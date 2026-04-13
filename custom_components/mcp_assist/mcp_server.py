"""MCP Server for Home Assistant entity discovery."""

import asyncio
import base64
from collections import Counter, defaultdict
import ipaddress
import json
import logging
import mimetypes
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
from datetime import date, datetime, time, timedelta

import aiohttp
from aiohttp import web, WSMsgType
from aiohttp.web_ws import WebSocketResponse
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    llm,
    service as service_helper,
)
from homeassistant.components.homeassistant import async_should_expose
from homeassistant.components.recorder import history
from homeassistant.util import dt as dt_util

try:
    from homeassistant.helpers import floor_registry as fr
except ImportError:  # pragma: no cover - older Home Assistant versions
    fr = None

try:
    from homeassistant.helpers import label_registry as lr
except ImportError:  # pragma: no cover - older Home Assistant versions
    lr = None

from .custom_tools.builtin_catalog import (
    BuiltInToolToggleSpec,
    is_builtin_package_enabled_for_shared_settings,
)
from .const import (
    DOMAIN,
    MCP_SERVER_NAME,
    MAX_ENTITIES_PER_DISCOVERY,
    CONF_API_KEY,
    CONF_LMSTUDIO_URL,
    CONF_MODEL_NAME,
    CONF_ALLOWED_IPS,
    CONF_SEARCH_PROVIDER,
    CONF_SERVER_TYPE,
    CONF_TIMEOUT,
    CONF_ENABLE_WEB_SEARCH,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
    CONF_ENABLE_WEATHER_FORECAST_TOOL,
    CONF_ENABLE_RECORDER_TOOLS,
    CONF_ENABLE_MEMORY_TOOLS,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    CONF_ENABLE_CUSTOM_TOOLS,
    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_MEMORY_DEFAULT_TTL_DAYS,
    CONF_MEMORY_MAX_TTL_DAYS,
    CONF_MEMORY_MAX_ITEMS,
    DEFAULT_API_KEY,
    DEFAULT_LMSTUDIO_URL,
    DEFAULT_MODEL_NAME,
    DEFAULT_OLLAMA_URL,
    DEFAULT_LLAMACPP_URL,
    DEFAULT_MOLTBOT_URL,
    DEFAULT_ALLOWED_IPS,
    DEFAULT_SERVER_TYPE,
    DEFAULT_SEARCH_PROVIDER,
    DEFAULT_TIMEOUT,
    DEFAULT_VLLM_URL,
    DEFAULT_ENABLE_ASSIST_BRIDGE,
    DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
    DEFAULT_ENABLE_WEATHER_FORECAST_TOOL,
    DEFAULT_ENABLE_RECORDER_TOOLS,
    DEFAULT_ENABLE_MEMORY_TOOLS,
    DEFAULT_ENABLE_UNIT_CONVERSION_TOOLS,
    DEFAULT_ENABLE_DEVICE_TOOLS,
    DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    DEFAULT_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
    DEFAULT_MEMORY_MAX_TTL_DAYS,
    DEFAULT_MEMORY_MAX_ITEMS,
    OPENAI_BASE_URL,
    GEMINI_BASE_URL,
    ANTHROPIC_BASE_URL,
    OPENROUTER_BASE_URL,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_OPENAI,
    SERVER_TYPE_GEMINI,
    SERVER_TYPE_ANTHROPIC,
    SERVER_TYPE_OPENROUTER,
    SERVER_TYPE_MOLTBOT,
    SERVER_TYPE_VLLM,
    TOOL_FAMILY_SHARED_SETTINGS,
    get_optional_tool_family,
)
from .discovery import EntityDiscovery
from .domain_registry import (
    validate_domain_action,
    validate_service_parameters,
    get_supported_domains,
    get_domain_info,
    get_domains_by_type,
    TYPE_CONTROLLABLE,
    TYPE_READ_ONLY,
)
from .memory_manager import MemoryManager

_LOGGER = logging.getLogger(__name__)

_MAX_INLINE_IMAGE_BYTES = 6 * 1024 * 1024


class MCPServer:
    """MCP Server for entity discovery."""

    def __init__(self, hass: HomeAssistant, port: int, entry=None) -> None:
        """Initialize MCP server."""
        self.hass = hass
        self.port = port
        self.entry = entry
        self.app: web.Application | None = None
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None
        self.discovery = EntityDiscovery(hass)
        self.sse_clients = []  # Track SSE connections for notifications
        self.progress_queues = set()  # Track progress SSE clients
        self._cached_tools_list: dict[str, Any] | None = None
        self._cached_tools_signature: tuple[Any, ...] | None = None
        self.memory_manager = MemoryManager(hass)

        # Extract allowed IPs from LM Studio URL
        self.allowed_ips = ["127.0.0.1", "::1"]  # Always allow localhost

        # Get LM Studio URL from config
        lmstudio_url = DEFAULT_LMSTUDIO_URL
        if entry:
            # Check options first, then data
            lmstudio_url = entry.options.get(
                CONF_LMSTUDIO_URL,
                entry.data.get(CONF_LMSTUDIO_URL, DEFAULT_LMSTUDIO_URL),
            )

        # Extract hostname/IP from LM Studio URL
        try:
            parsed = urlparse(lmstudio_url)
            lmstudio_host = parsed.hostname or parsed.netloc.split(":")[0]
            if lmstudio_host and lmstudio_host not in self.allowed_ips:
                self.allowed_ips.append(lmstudio_host)
                _LOGGER.info(
                    "MCP server automatically whitelisted LM Studio IP: %s",
                    lmstudio_host,
                )
        except Exception as e:
            _LOGGER.warning("Could not parse LM Studio URL '%s': %s", lmstudio_url, e)

        # Add user-configured allowed IPs/CIDR ranges (shared setting)
        allowed_ips_str = self._get_shared_setting(
            CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS
        )
        if allowed_ips_str:
            # Parse comma-separated list
            additional_ips = [
                ip.strip() for ip in allowed_ips_str.split(",") if ip.strip()
            ]
            for ip_entry in additional_ips:
                if ip_entry not in self.allowed_ips:
                    self.allowed_ips.append(ip_entry)
            if additional_ips:
                _LOGGER.info(
                    "MCP server added user-configured allowed IPs/ranges: %s",
                    additional_ips,
                )

        _LOGGER.info("MCP server allowed IPs/ranges: %s", self.allowed_ips)

        # Custom tools will be initialized in start() after system entry exists
        self.custom_tools = None

    def _get_shared_setting(self, key: str, default: Any) -> Any:
        """Get a shared setting from system entry with fallback to profile entry."""
        # Import here to avoid circular dependency
        from . import get_system_entry

        # Try to get from system entry first
        system_entry = get_system_entry(self.hass)
        if system_entry:
            value = system_entry.options.get(key, system_entry.data.get(key))
            if value is not None:
                return value

        # Fallback to profile entry for backward compatibility
        if self.entry:
            value = self.entry.options.get(key, self.entry.data.get(key))
            if value is not None:
                return value

        # Return default
        return default

    def _get_search_provider(self) -> str:
        """Get search provider (shared setting) with backward compatibility."""
        provider = self._get_shared_setting(CONF_SEARCH_PROVIDER, None)
        if provider:
            return provider

        # Backward compat: if old enable_custom_tools was True, default to "brave"
        if self._get_shared_setting(CONF_ENABLE_CUSTOM_TOOLS, False):
            return "brave"

        return "none"

    def _web_search_enabled(self) -> bool:
        """Return whether web-search tools are enabled."""
        explicit_enabled = self._get_shared_setting(CONF_ENABLE_WEB_SEARCH, None)
        if explicit_enabled is not None:
            return bool(explicit_enabled)
        provider = self._get_shared_setting(CONF_SEARCH_PROVIDER, DEFAULT_SEARCH_PROVIDER)
        if provider and str(provider).strip().casefold() != DEFAULT_SEARCH_PROVIDER:
            return True
        return bool(self._get_shared_setting(CONF_ENABLE_CUSTOM_TOOLS, False))

    def _music_assistant_support_enabled(self) -> bool:
        """Return whether Music Assistant-specific MCP support is enabled."""
        return bool(
            self._get_shared_setting(
                CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
                DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT,
            )
        )

    def _external_custom_tools_enabled(self) -> bool:
        """Return whether user-defined external custom tool packages are enabled."""
        return bool(
            self._get_shared_setting(
                CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                DEFAULT_ENABLE_EXTERNAL_CUSTOM_TOOLS,
            )
        )

    def _weather_forecast_tool_enabled(self) -> bool:
        """Return whether weather forecast MCP helpers are enabled."""
        return bool(
            self._get_shared_setting(
                CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
                DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
            )
        ) and bool(
            self._get_shared_setting(
                CONF_ENABLE_WEATHER_FORECAST_TOOL,
                DEFAULT_ENABLE_WEATHER_FORECAST_TOOL,
            )
        )

    def _assist_bridge_enabled(self) -> bool:
        """Return whether native Assist bridge tools are enabled."""
        return bool(
            self._get_shared_setting(
                CONF_ENABLE_ASSIST_BRIDGE,
                DEFAULT_ENABLE_ASSIST_BRIDGE,
            )
        )

    def _response_service_tools_enabled(self) -> bool:
        """Return whether native response-service tools are enabled."""
        return bool(
            self._get_shared_setting(
                CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
                DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
            )
        )

    def _recorder_tools_enabled(self) -> bool:
        """Return whether recorder/history tools are enabled."""
        return bool(
            self._get_shared_setting(
                CONF_ENABLE_RECORDER_TOOLS,
                DEFAULT_ENABLE_RECORDER_TOOLS,
            )
        )

    def _calculator_tools_enabled(self) -> bool:
        """Return whether calculator tools are enabled."""
        built_in_spec = self._get_builtin_toggle_spec("add")
        if built_in_spec is not None:
            return self._is_builtin_package_enabled(built_in_spec)
        return bool(
            self._get_shared_setting(
                CONF_ENABLE_CALCULATOR_TOOLS,
                False,
            )
        )

    def _memory_tools_enabled(self) -> bool:
        """Return whether persisted memory tools are enabled."""
        return bool(
            self._get_shared_setting(
                CONF_ENABLE_MEMORY_TOOLS,
                DEFAULT_ENABLE_MEMORY_TOOLS,
            )
        )

    def _memory_default_ttl_days(self) -> int:
        """Return the default TTL for new memories."""
        configured_max = self._memory_max_ttl_days()
        return self._coerce_int_arg(
            self._get_shared_setting(
                CONF_MEMORY_DEFAULT_TTL_DAYS,
                DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
            ),
            default=DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
            minimum=1,
            maximum=configured_max,
        )

    def _memory_max_ttl_days(self) -> int:
        """Return the maximum TTL allowed for memories."""
        return self._coerce_int_arg(
            self._get_shared_setting(
                CONF_MEMORY_MAX_TTL_DAYS,
                DEFAULT_MEMORY_MAX_TTL_DAYS,
            ),
            default=DEFAULT_MEMORY_MAX_TTL_DAYS,
            minimum=1,
            maximum=3650,
        )

    def _memory_max_items(self) -> int:
        """Return the maximum number of memories to keep."""
        return self._coerce_int_arg(
            self._get_shared_setting(
                CONF_MEMORY_MAX_ITEMS,
                DEFAULT_MEMORY_MAX_ITEMS,
            ),
            default=DEFAULT_MEMORY_MAX_ITEMS,
            minimum=10,
            maximum=5000,
        )

    def _unit_conversion_tools_enabled(self) -> bool:
        """Return whether unit-conversion tools are enabled."""
        built_in_spec = self._get_builtin_toggle_spec("convert_unit")
        if built_in_spec is not None:
            return self._is_builtin_package_enabled(built_in_spec)
        explicit_enabled = self._get_shared_setting(
            CONF_ENABLE_UNIT_CONVERSION_TOOLS,
            None,
        )
        if explicit_enabled is not None:
            return bool(explicit_enabled)
        return bool(
            self._get_shared_setting(
                CONF_ENABLE_CALCULATOR_TOOLS,
                DEFAULT_ENABLE_UNIT_CONVERSION_TOOLS,
            )
        )

    def _device_tools_enabled(self) -> bool:
        """Return whether Home Assistant device tools are enabled."""
        return bool(
            self._get_shared_setting(
                CONF_ENABLE_DEVICE_TOOLS,
                DEFAULT_ENABLE_DEVICE_TOOLS,
            )
        )

    def _get_builtin_toggle_spec(
        self,
        tool_name: str,
    ) -> BuiltInToolToggleSpec | None:
        """Return built-in packaged-tool metadata for a tool name, if any."""
        custom_tools = self.custom_tools
        if custom_tools is None:
            return None

        getter = getattr(custom_tools, "get_builtin_toggle_spec", None)
        if not callable(getter):
            return None

        try:
            return getter(tool_name)
        except Exception as err:
            _LOGGER.debug(
                "Unable to read built-in packaged tool metadata for %s: %s",
                tool_name,
                err,
            )
            return None

    def _get_builtin_toggle_specs(self) -> tuple[BuiltInToolToggleSpec, ...]:
        """Return built-in packaged-tool metadata from the custom tool loader."""
        custom_tools = self.custom_tools
        if custom_tools is None:
            return ()

        getter = getattr(custom_tools, "get_builtin_toggle_specs", None)
        if not callable(getter):
            return ()

        try:
            return tuple(getter() or ())
        except Exception as err:
            _LOGGER.debug("Unable to read built-in packaged tool specs: %s", err)
            return ()

    def _is_builtin_package_enabled(
        self,
        spec: BuiltInToolToggleSpec,
    ) -> bool:
        """Return whether a built-in packaged tool is enabled by shared settings."""
        return is_builtin_package_enabled_for_shared_settings(
            spec,
            self._get_shared_setting,
            search_provider=self._get_search_provider(),
        )

    def _get_domain_capability_error(self, domain: str) -> str | None:
        """Return a settings-based capability error for a domain, if any."""
        if (
            domain == "music_assistant"
            and not self._music_assistant_support_enabled()
        ):
            return (
                "Music Assistant support is disabled in shared MCP settings. "
                "Enable it to use Music Assistant actions or response services."
            )
        if domain == "weather" and not self._weather_forecast_tool_enabled():
            return (
                "Weather forecast support is disabled in shared MCP settings. "
                "Enable it to use weather forecast tools or weather response services."
            )

        return None

    def _is_tool_enabled(self, tool_name: str) -> bool:
        """Return whether an optional tool is enabled by settings."""
        built_in_spec = self._get_builtin_toggle_spec(tool_name)
        if built_in_spec is not None:
            return self._is_builtin_package_enabled(built_in_spec)

        if tool_name == "get_weather_forecast":
            return self._weather_forecast_tool_enabled()
        if tool_name == "convert_unit":
            return self._unit_conversion_tools_enabled()

        family = get_optional_tool_family(tool_name)
        if family is None:
            return True

        setting_key, default = TOOL_FAMILY_SHARED_SETTINGS[family]
        return bool(self._get_shared_setting(setting_key, default))

    def _get_tools_list_signature(self, max_limit: int) -> tuple[Any, ...]:
        """Return a cache signature for the current MCP tool surface."""
        custom_tool_signature: tuple[Any, ...] = ()
        if self.custom_tools:
            get_cache_signature = getattr(self.custom_tools, "get_cache_signature", None)
            if callable(get_cache_signature):
                try:
                    raw_signature = get_cache_signature()
                    if isinstance(raw_signature, tuple):
                        custom_tool_signature = raw_signature
                    else:
                        custom_tool_signature = (raw_signature,)
                except Exception as err:
                    _LOGGER.debug(
                        "Unable to build custom tool cache signature: %s", err
                    )
            else:
                custom_tool_store = getattr(self.custom_tools, "tools", {})
                if isinstance(custom_tool_store, dict):
                    custom_tool_signature = (tuple(sorted(custom_tool_store.keys())),)

        return (
            max_limit,
            self._get_search_provider(),
            self._web_search_enabled(),
            self._assist_bridge_enabled(),
            self._response_service_tools_enabled(),
            self._weather_forecast_tool_enabled(),
            self._recorder_tools_enabled(),
            self._memory_tools_enabled(),
            self._calculator_tools_enabled(),
            self._unit_conversion_tools_enabled(),
            self._device_tools_enabled(),
            self._music_assistant_support_enabled(),
            self._external_custom_tools_enabled(),
            tuple(
                (
                    spec.package_id,
                    self._is_builtin_package_enabled(spec),
                )
                for spec in self._get_builtin_toggle_specs()
            ),
            custom_tool_signature,
        )

    async def start(self) -> None:
        """Start the MCP server."""
        try:
            _LOGGER.info(
                "Starting MCP server on port %d, binding to all interfaces (0.0.0.0)",
                self.port,
            )

            # Create web application (IP checks are done per-handler, not via middleware)
            self.app = web.Application()
            self.app.router.add_post("/", self.handle_mcp_request)
            self.app.router.add_get("/sse", self.handle_sse)  # SSE endpoint
            self.app.router.add_get("/", self.handle_sse)  # Also handle root GET as SSE
            self.app.router.add_get("/ws", self.handle_websocket)
            self.app.router.add_get("/health", self.handle_health)
            self.app.router.add_get(
                "/external-tools/diagnostics",
                self.handle_external_tool_diagnostics,
            )
            self.app.router.add_get(
                "/progress", self.handle_progress_stream
            )  # Progress streaming

            self.runner = web.AppRunner(self.app)
            await self.runner.setup()

            # Bind to all interfaces so external machines can connect
            self.site = web.TCPSite(self.runner, "0.0.0.0", self.port)
            await self.site.start()

            # Create and initialize custom tools after system entry exists.
            # Calculator tools are optional; web tools depend on search provider.
            search_provider = self._get_search_provider()
            try:
                from .custom_tools import CustomToolsLoader

                self.custom_tools = CustomToolsLoader(self.hass, self.entry)
                await self.custom_tools.initialize()
                _LOGGER.info(
                    "✅ Custom tools initialized (search provider: %s, external enabled: %s)",
                    search_provider,
                    self._external_custom_tools_enabled(),
                )
            except Exception as e:
                _LOGGER.error(f"Failed to initialize custom tools: {e}")

            if self._memory_tools_enabled():
                try:
                    await self.memory_manager.async_initialize()
                    _LOGGER.info(
                        "✅ Memory tools initialized (default ttl: %s days, max ttl: %s days, max items: %s)",
                        self._memory_default_ttl_days(),
                        self._memory_max_ttl_days(),
                        self._memory_max_items(),
                    )
                except Exception as err:
                    _LOGGER.error("Failed to initialize memory tools: %s", err)

            _LOGGER.info(
                "✅ MCP server started successfully on http://0.0.0.0:%d", self.port
            )
            _LOGGER.info("🌐 MCP server is accessible from external machines")
            _LOGGER.info(
                "🔗 Health check available at: http://<your-ha-ip>:%d/health", self.port
            )
            _LOGGER.info("📡 WebSocket endpoint: ws://<your-ha-ip>:%d/ws", self.port)
            _LOGGER.info("📤 HTTP endpoint: http://<your-ha-ip>:%d/", self.port)

        except OSError as err:
            if err.errno == 98:  # Address already in use
                _LOGGER.error(
                    "❌ Port %d is already in use. Please choose a different port.",
                    self.port,
                )
                raise
            elif err.errno == 13:  # Permission denied
                _LOGGER.error(
                    "❌ Permission denied to bind to port %d. Try a port >= 1024.",
                    self.port,
                )
                raise
            else:
                _LOGGER.error(
                    "❌ Failed to bind MCP server to port %d: %s", self.port, err
                )
                raise
        except Exception as err:
            _LOGGER.error("❌ Failed to start MCP server: %s", err)
            raise

    async def stop(self) -> None:
        """Stop the MCP server."""
        _LOGGER.info("Stopping MCP server")

        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        if self.custom_tools:
            await self.custom_tools.shutdown()
        if self.memory_manager:
            await self.memory_manager.async_shutdown()

    def _is_ip_allowed(self, client_ip: str) -> bool:
        """Check if client IP is in the allowed list.

        Handles various formats:
        - IPv4: 192.168.1.7
        - IPv4 with port: 192.168.1.7:12345
        - IPv6: ::1 or 2001:db8::1
        - IPv6 with port: [2001:db8::1]:8080
        - CIDR ranges: 172.30.0.0/16, 192.168.1.0/24
        """
        if not self.allowed_ips:
            # If no IPs configured, allow all (backward compatible)
            return True

        if not client_ip:
            return False

        # Extract IP from various formats
        ip_only = client_ip

        # Handle IPv6 with port: [2001:db8::1]:8080 -> 2001:db8::1
        if ip_only.startswith("["):
            end_bracket = ip_only.find("]")
            if end_bracket > 0:
                ip_only = ip_only[1:end_bracket]
        # Handle IPv4 with port: 192.168.1.7:12345 -> 192.168.1.7
        # Only split on single colon (not IPv6 which has multiple colons)
        elif ip_only.count(":") == 1:
            ip_only = ip_only.split(":")[0]
        # Else: IPv6 without port (::1) or IPv4 without port - use as-is

        # Convert to IP address object for CIDR checking
        try:
            client_ip_obj = ipaddress.ip_address(ip_only)
        except ValueError:
            _LOGGER.warning("Invalid client IP format: %s", ip_only)
            return False

        # Check if client IP matches any allowed IP or CIDR range
        for allowed_entry in self.allowed_ips:
            # Check for exact IP match first (backward compatible)
            if ip_only == allowed_entry:
                return True

            # Check if it's a CIDR range
            if "/" in allowed_entry:
                try:
                    network = ipaddress.ip_network(allowed_entry, strict=False)
                    if client_ip_obj in network:
                        return True
                except ValueError:
                    # Invalid CIDR format, skip
                    _LOGGER.warning(
                        "Invalid CIDR format in allowed IPs: %s", allowed_entry
                    )
                    continue

        return False

    async def handle_health(self, request: web.Request) -> web.Response:
        """Handle health check requests."""
        client_ip = request.remote
        _LOGGER.info("🏥 Health check from %s", client_ip)

        health_info = {
            "status": "healthy",
            "server": MCP_SERVER_NAME,
            "port": self.port,
            "version": "0.1.0",
            "endpoints": {
                "websocket": f"ws://<host>:{self.port}/ws",
                "http": f"http://<host>:{self.port}/",
                "health": f"http://<host>:{self.port}/health",
            },
            "tools_available": len(await self._get_tools_list()),
            "timestamp": dt_util.now().isoformat(),
        }
        if self.custom_tools:
            health_info["external_custom_tools_enabled"] = (
                self._external_custom_tools_enabled()
            )
            get_loaded_builtin_tool_info = getattr(
                self.custom_tools,
                "get_loaded_builtin_tool_info",
                None,
            )
            if callable(get_loaded_builtin_tool_info):
                health_info["built_in_tool_packages_loaded"] = (
                    get_loaded_builtin_tool_info()
                )
            health_info["external_custom_tools_loaded"] = (
                self.custom_tools.get_loaded_external_tool_info()
            )
            get_package_diagnostics = getattr(
                self.custom_tools,
                "get_package_diagnostics",
                None,
            )
            if callable(get_package_diagnostics):
                health_info["tool_package_diagnostics"] = (
                    get_package_diagnostics()
                )
            get_external_diagnostics = getattr(
                self.custom_tools,
                "get_external_diagnostics",
                None,
            )
            if callable(get_external_diagnostics):
                health_info["external_custom_tool_diagnostics"] = (
                    get_external_diagnostics()
                )
        return web.json_response(health_info)

    async def handle_external_tool_diagnostics(
        self, request: web.Request
    ) -> web.Response:
        """Return detailed diagnostics for manifest-based tool packages."""
        client_ip = request.remote
        _LOGGER.info("🧰 External tool diagnostics request from %s", client_ip)

        if not self._is_ip_allowed(client_ip):
            _LOGGER.warning(
                "🚫 Blocked external tool diagnostics request from unauthorized IP: %s",
                client_ip,
            )
            return web.Response(status=403, text="Forbidden: IP not authorized")

        diagnostics: dict[str, Any] = {
            "enabled": self._external_custom_tools_enabled(),
            "loaded": [],
        }
        if self.custom_tools:
            get_package_diagnostics = getattr(
                self.custom_tools,
                "get_package_diagnostics",
                None,
            )
            if callable(get_package_diagnostics):
                diagnostics = get_package_diagnostics()
            get_external_diagnostics = getattr(
                self.custom_tools,
                "get_external_diagnostics",
                None,
            )
            if callable(get_external_diagnostics) and not callable(get_package_diagnostics):
                diagnostics = get_external_diagnostics()

        return web.json_response(diagnostics)

    async def reload_external_custom_tools(self) -> dict[str, Any]:
        """Reload external custom tools, clear caches, and notify clients."""
        if not self.custom_tools:
            return {
                "enabled": self._external_custom_tools_enabled(),
                "loaded_tools": [],
                "load_errors": ["Custom tools are not initialized"],
            }

        reload_tool_packages = getattr(self.custom_tools, "reload_tool_packages", None)
        if callable(reload_tool_packages):
            diagnostics = await reload_tool_packages()
        else:
            reload_external_tools = getattr(self.custom_tools, "reload_external_tools", None)
            if not callable(reload_external_tools):
                return {
                    "enabled": self._external_custom_tools_enabled(),
                    "loaded_tools": [],
                    "load_errors": ["Reload is not supported by the current custom tool loader"],
                }

            diagnostics = await reload_external_tools()

        if diagnostics is None:
            return {
                "enabled": self._external_custom_tools_enabled(),
                "loaded_tools": [],
                "load_errors": ["Reload is not supported by the current custom tool loader"],
            }
        self._cached_tools_list = None
        self._cached_tools_signature = None
        await self.broadcast_notification("notifications/tools/list_changed")
        return diagnostics

    async def handle_progress_stream(self, request: web.Request) -> web.StreamResponse:
        """SSE endpoint for progress updates during tool execution."""
        client_ip = request.remote
        _LOGGER.info("📊 Progress stream request from %s", client_ip)

        # Check IP whitelist
        if not self._is_ip_allowed(client_ip):
            _LOGGER.warning(
                "🚫 Blocked progress stream request from unauthorized IP: %s", client_ip
            )
            return web.Response(status=403, text="Forbidden: IP not authorized")

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )
        await response.prepare(request)

        # Create a queue for this client
        queue = asyncio.Queue()
        self.progress_queues.add(queue)

        try:
            # Send initial connection message
            data = f"data: {json.dumps({'type': 'connected', 'message': 'Progress stream connected'})}\n\n"
            await response.write(data.encode())

            # Stream progress updates
            while True:
                msg = await queue.get()
                data = f"data: {json.dumps(msg)}\n\n"
                await response.write(data.encode())

        except Exception as e:
            _LOGGER.debug(f"Progress stream closed: {e}")
        finally:
            self.progress_queues.discard(queue)

        return response

    def publish_progress(self, event_type: str, message: str, **kwargs):
        """Publish progress update to all progress SSE clients."""
        import time

        msg = {
            "type": event_type,
            "message": message,
            "timestamp": time.time(),
            **kwargs,
        }

        # Send to all progress clients
        for queue in list(self.progress_queues):
            try:
                queue.put_nowait(msg)
            except asyncio.QueueFull:
                _LOGGER.debug("Progress queue full, skipping")

    async def handle_sse(self, request: web.Request) -> web.StreamResponse:
        """Handle Server-Sent Events for MCP notifications."""
        client_ip = request.remote
        _LOGGER.info("🌊 SSE connection request from %s", client_ip)

        # Check IP whitelist
        if not self._is_ip_allowed(client_ip):
            _LOGGER.warning(
                "🚫 Blocked SSE connection from unauthorized IP: %s", client_ip
            )
            return web.Response(status=403, text="Forbidden: IP not authorized")

        response = web.StreamResponse()
        response.headers["Content-Type"] = "text/event-stream"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"  # Disable nginx buffering
        response.headers["Access-Control-Allow-Origin"] = "*"

        await response.prepare(request)

        # Store this client for notifications
        self.sse_clients.append(response)
        _LOGGER.info("✅ SSE client connected. Total clients: %d", len(self.sse_clients))

        try:
            # Send initial connection confirmation
            await response.write(b": connected\n\n")

            # Send tools list changed notification immediately
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/tools/list_changed",
            }
            await response.write(f"data: {json.dumps(notification)}\n\n".encode())
            _LOGGER.info("📤 Sent initial tools/list_changed notification")

            # Keep connection alive
            while True:
                await asyncio.sleep(30)
                await response.write(b": keepalive\n\n")

        except Exception as err:
            _LOGGER.info("📤 SSE client disconnected: %s", err)
        finally:
            if response in self.sse_clients:
                self.sse_clients.remove(response)
            _LOGGER.info("SSE clients remaining: %d", len(self.sse_clients))

        return response

    async def _get_tools_list(self) -> List[Dict[str, Any]]:
        """Get the tools list for health check."""
        tools_result = await self.handle_tools_list()
        return tools_result.get("tools", [])

    def _get_media_tool_definitions(self) -> list[dict[str, Any]]:
        """Return generic media/image MCP tools."""
        return [
            {
                "name": "analyze_image",
                "description": (
                    "Analyze an image, camera snapshot, or image-like entity with the "
                    "current profile's multimodal model. Use this for questions such as "
                    "'what is in the driveway?' or 'who is at the door?' when an image "
                    "source is available."
                ),
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": (
                                "Question to answer about the image. Defaults to a short factual description."
                            ),
                            "default": "Describe the image briefly and focus on factual observations.",
                        },
                        "camera_entity_id": {
                            "type": "string",
                            "description": "Camera entity to snapshot before analysis.",
                        },
                        "entity_id": {
                            "type": "string",
                            "description": (
                                "Image-like entity to resolve. Camera entities are supported directly. "
                                "Other entities may work when they expose a local or remote picture URL."
                            ),
                        },
                        "image_url": {
                            "type": "string",
                            "description": (
                                "Remote URL, data URL, or /local/... URL for the image to analyze."
                            ),
                        },
                        "image_path": {
                            "type": "string",
                            "description": (
                                "Local image path relative to the Home Assistant config directory, "
                                "or an absolute path inside that directory."
                            ),
                        },
                        "detail": {
                            "type": "string",
                            "enum": ["auto", "low", "high"],
                            "default": "auto",
                            "description": "Requested image detail level for OpenAI-compatible providers.",
                        },
                        "include_image": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include the source image as an MCP image content block in the result.",
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                "routingHints": {
                    "keywords": ["camera", "image", "vision", "driveway", "door"],
                    "example_queries": [
                        "What's in the driveway?",
                        "Who's at the front door?",
                    ],
                    "preferred_when": (
                        "Use when the user wants a live or static visual answer from a camera, URL, or image."
                    ),
                    "returns": "A factual answer plus optional structured image metadata.",
                },
            },
            {
                "name": "get_image",
                "description": (
                    "Fetch an image from a camera, image-like entity, URL, or local file "
                    "and return it as an MCP image content block for clients that can display images."
                ),
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "camera_entity_id": {
                            "type": "string",
                            "description": "Camera entity to snapshot.",
                        },
                        "entity_id": {
                            "type": "string",
                            "description": (
                                "Image-like entity to resolve. Camera entities are supported directly."
                            ),
                        },
                        "image_url": {
                            "type": "string",
                            "description": "Remote URL, data URL, or /local/... URL for the image.",
                        },
                        "image_path": {
                            "type": "string",
                            "description": (
                                "Local image path relative to the Home Assistant config directory, "
                                "or an absolute path inside that directory."
                            ),
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                "routingHints": {
                    "keywords": ["show", "display", "image", "qr", "camera"],
                    "example_queries": [
                        "Show me the guest wifi QR code.",
                        "Display the latest driveway snapshot.",
                    ],
                    "preferred_when": (
                        "Use when the client can render images or a downstream tool needs an image block."
                    ),
                    "returns": "An MCP image block plus lightweight source metadata.",
                },
            },
            {
                "name": "generate_image",
                "description": (
                    "Generate an image with the current profile's provider when it exposes "
                    "an OpenAI-compatible image generation API. Returns an MCP image content block when available."
                ),
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "What image to generate.",
                        },
                        "size": {
                            "type": "string",
                            "description": "Optional image size such as 1024x1024.",
                        },
                        "quality": {
                            "type": "string",
                            "description": "Optional provider-specific quality hint.",
                        },
                        "style": {
                            "type": "string",
                            "description": "Optional provider-specific style hint.",
                        },
                        "background": {
                            "type": "string",
                            "description": "Optional provider-specific background hint.",
                        },
                    },
                    "required": ["prompt"],
                    "additionalProperties": False,
                },
                "routingHints": {
                    "keywords": ["generate", "image", "draw", "illustration", "qr"],
                    "example_queries": [
                        "Generate a guest wifi QR code poster.",
                        "Create a simple front door instruction image.",
                    ],
                    "preferred_when": (
                        "Use when the user explicitly wants a new image and the provider supports image generation."
                    ),
                    "returns": "An MCP image block or a clear unsupported-provider error.",
                },
            },
        ]

    async def handle_websocket(self, request: web.Request) -> WebSocketResponse:
        """Handle WebSocket connections for MCP protocol."""
        client_ip = request.remote
        _LOGGER.info("🔌 New MCP WebSocket connection from %s", client_ip)

        # Check IP whitelist
        if not self._is_ip_allowed(client_ip):
            _LOGGER.warning(
                "🚫 Blocked WebSocket connection from unauthorized IP: %s", client_ip
            )
            return web.Response(status=403, text="Forbidden: IP not authorized")

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        _LOGGER.info("✅ MCP WebSocket connection established with %s", client_ip)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)

                        # Check if it's a notification (no id field)
                        if "id" not in data:
                            await self.process_mcp_notification(data)
                            # No response for notifications
                        else:
                            response = await self.process_mcp_message(data)
                            await ws.send_str(json.dumps(response))
                    except json.JSONDecodeError:
                        await ws.send_str(
                            json.dumps(
                                {"error": {"code": -32700, "message": "Parse error"}}
                            )
                        )
                    except Exception as err:
                        _LOGGER.exception("Error processing MCP message")
                        await ws.send_str(
                            json.dumps(
                                {
                                    "error": {
                                        "code": -32000,
                                        "message": f"Server error: {err}",
                                    }
                                }
                            )
                        )
                elif msg.type == WSMsgType.ERROR:
                    _LOGGER.error("WebSocket error: %s", ws.exception())
                    break

        except asyncio.CancelledError:
            pass
        except Exception:
            _LOGGER.exception("WebSocket handler error")

        return ws

    async def handle_mcp_request(self, request: web.Request) -> web.Response:
        """Handle HTTP MCP requests with proper JSON-RPC 2.0 protocol."""
        client_ip = request.remote
        _LOGGER.info("📨 MCP HTTP JSON-RPC request from %s", client_ip)

        # Check IP whitelist
        if not self._is_ip_allowed(client_ip):
            _LOGGER.warning("🚫 Blocked MCP request from unauthorized IP: %s", client_ip)
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32000,
                        "message": "Forbidden: IP not authorized",
                    },
                    "id": None,
                },
                status=403,
            )

        request_id = None
        try:
            data = await request.json()
            request_id = data.get("id")

            # Validate JSON-RPC 2.0 format
            if "jsonrpc" not in data or data["jsonrpc"] != "2.0":
                return web.json_response(
                    {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request: missing or invalid jsonrpc field",
                        },
                        "id": request_id,
                    },
                    status=400,
                )

            if "method" not in data:
                return web.json_response(
                    {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request: missing method field",
                        },
                        "id": request_id,
                    },
                    status=400,
                )

            # Check if this is a notification (no id field)
            is_notification = "id" not in data

            if is_notification:
                _LOGGER.debug("📮 MCP notification: %s", data.get("method"))
                # Process the notification but don't expect a response
                await self.process_mcp_notification(data)
                # Return 204 No Content for notifications
                return web.Response(status=204)
            else:
                _LOGGER.debug(
                    "📋 MCP method: %s (id: %s)", data.get("method"), request_id
                )
                response = await self.process_mcp_message(data)
                return web.json_response(response)

        except json.JSONDecodeError:
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "Parse error: invalid JSON"},
                    "id": None,
                },
                status=400,
            )
        except Exception as err:
            _LOGGER.exception("Error processing MCP request")
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": f"Internal error: {str(err)}"},
                    "id": request_id,
                },
                status=500,
            )

    async def process_mcp_notification(self, data: Dict[str, Any]) -> None:
        """Process MCP notification (no response expected)."""
        method = data.get("method")

        _LOGGER.info("Processing MCP notification: %s", method)

        try:
            # Handle both old and new MCP notification formats
            # Old format: "initialized"
            # New format: "notifications/initialized"
            if method in ("initialized", "notifications/initialized"):
                _LOGGER.info("✅ MCP client initialized successfully")
                # Send tools/list_changed to all SSE clients
                await self.broadcast_notification("notifications/tools/list_changed")
            elif method == "notifications/cancelled":
                # Client cancelled a pending request
                _LOGGER.debug("MCP client cancelled a request")
            else:
                _LOGGER.warning("Unknown notification method: %s", method)
        except Exception as err:
            _LOGGER.exception("Error processing notification %s: %s", method, err)

    async def broadcast_notification(
        self, method: str, params: Dict[str, Any] | None = None
    ) -> None:
        """Send notification to all SSE clients."""
        if not self.sse_clients:
            _LOGGER.debug("No SSE clients to notify for %s", method)
            return

        notification = {"jsonrpc": "2.0", "method": method}
        if params:
            notification["params"] = params

        data = f"data: {json.dumps(notification)}\n\n".encode()

        # Send to all clients, removing dead ones
        dead_clients = []
        for client in self.sse_clients:
            try:
                await client.write(data)
            except Exception as err:
                _LOGGER.debug("Failed to send to client: %s", err)
                dead_clients.append(client)

        # Remove dead clients
        for client in dead_clients:
            self.sse_clients.remove(client)

        if dead_clients:
            _LOGGER.info("Removed %d dead SSE clients", len(dead_clients))

    async def process_mcp_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process MCP message according to JSON-RPC 2.0 protocol."""
        method = data.get("method")
        params = data.get("params", {})
        msg_id = data.get("id")

        _LOGGER.debug("Processing MCP method: %s (id: %s)", method, msg_id)

        try:
            if method == "initialize":
                result = await self.handle_initialize(params)
            elif method == "tools/list":
                result = await self.handle_tools_list()
            elif method == "tools/call":
                result = await self.handle_tool_call(params)
            else:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                    "id": msg_id,
                }

            # Always include jsonrpc and id in successful responses
            response = {"jsonrpc": "2.0", "result": result, "id": msg_id}

            return response

        except Exception as err:
            _LOGGER.exception("Error in MCP method %s", method)
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error in {method}: {str(err)}",
                },
                "id": msg_id,
            }

    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request."""
        _LOGGER.info("🔌 MCP initialize request received")
        return {
            "protocolVersion": "2024-11-05",  # MCP uses date-based versioning
            "capabilities": {
                "tools": {
                    "listChanged": True  # Tell client that tools can change dynamically
                }
            },
            "serverInfo": {"name": MCP_SERVER_NAME, "version": "0.1.0"},
        }

    async def handle_tools_list(self) -> Dict[str, Any]:
        """Handle tools/list request."""
        _LOGGER.info("MCP tools/list request received")

        # Get configured max entities limit from system entry
        from .const import DOMAIN, CONF_MAX_ENTITIES_PER_DISCOVERY, DEFAULT_MAX_ENTITIES_PER_DISCOVERY
        max_limit = DEFAULT_MAX_ENTITIES_PER_DISCOVERY
        for entry in self.hass.config_entries.async_entries(DOMAIN):
            if entry.source == "system":
                max_limit = entry.data.get(CONF_MAX_ENTITIES_PER_DISCOVERY, DEFAULT_MAX_ENTITIES_PER_DISCOVERY)
                break

        signature = self._get_tools_list_signature(max_limit)
        if self._cached_tools_list is not None and self._cached_tools_signature == signature:
            _LOGGER.debug("Returning cached MCP tools/list response")
            return {"tools": list(self._cached_tools_list)}

        tools = [
            {
                "name": "discover_entities",
                "description": "Find and list Home Assistant entities by criteria like area, floor, label, type, domain, device_class, current state, or aliases. Prefer this for most direct control and status checks, including entities that do not belong to any device. This returns a compact summary plus paging metadata; call get_entity_details for full entity attributes.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "entity_type": {
                            "type": "string",
                            "description": "Type of entity to find (e.g., 'light', 'switch', 'sensor', 'climate')",
                        },
                        "area": {
                            "type": "string",
                            "description": "Area/room name or alias to search in - use names from the areas list provided in your system context (e.g., 'Kitchen', 'Back Garden', 'Living Room'). If the value matches a floor name or alias instead, it will search that floor.",
                        },
                        "floor": {
                            "type": "string",
                            "description": "Floor name or alias to search in (e.g., 'Upstairs', 'Basement', 'Ground Floor'). Check get_index() to see available floors.",
                        },
                        "label": {
                            "type": "string",
                            "description": "Label name to filter by (matches labels assigned directly to entities, their devices, or their areas). Check get_index() to see available labels.",
                        },
                        "domain": {
                            "type": "string",
                            "description": "Home Assistant domain to filter by (e.g., 'light', 'switch', 'climate', 'sensor')",
                        },
                        "state": {
                            "type": "string",
                            "description": "Current state to filter by (e.g., 'on', 'off', 'unavailable')",
                        },
                        "name_contains": {
                            "type": "string",
                            "description": "Text that an entity name or alias should contain. Also matches related device names, device aliases, area aliases, floor aliases, and labels (case-insensitive). Results are ranked by the strongest match.",
                        },
                        "device_class": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ],
                            "description": "Device class to filter by (e.g., 'temperature', 'motion', 'door', 'moisture'). Can be a single string or array of strings for OR logic. Check the index for available device classes per domain.",
                        },
                        "name_pattern": {
                            "type": "string",
                            "description": "Wildcard pattern to match entity IDs (e.g., '*_person_detected', 'sensor.*_ble_area'). Supports * for any characters.",
                        },
                        "inferred_type": {
                            "type": "string",
                            "description": "Inferred entity type from the index (e.g., 'person_detection', 'location_tracking'). The pattern will be looked up from the index's inferred_types. Check get_index() to see available inferred types.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": f"Maximum number of entities to return for this page (default: 20, max: {max_limit})",
                            "default": 20,
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Zero-based pagination offset. Use the next_offset from a previous discovery response to fetch more results.",
                            "default": 0,
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "discover_devices",
                "description": "Find and list Home Assistant devices by criteria like area, floor, label, related entity domain, manufacturer, model, name, or aliases. Use this when the user is referring to a physical device or when you want to inspect related entities on the same device. This returns compact results plus paging metadata.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "area": {
                            "type": "string",
                            "description": "Area/room name or alias to search in. If the value matches a floor name or alias instead, it will search that floor.",
                        },
                        "floor": {
                            "type": "string",
                            "description": "Floor name or alias to search in.",
                        },
                        "label": {
                            "type": "string",
                            "description": "Label name to filter by (matches labels assigned to the device or its area).",
                        },
                        "domain": {
                            "type": "string",
                            "description": "Filter devices by attached entity domain (e.g., 'light', 'climate', 'media_player').",
                        },
                        "name_contains": {
                            "type": "string",
                            "description": "Text that a device name or alias should contain. Also matches attached entity names/aliases, area aliases, floor aliases, and label names (case-insensitive). Results are ranked by the strongest match.",
                        },
                        "manufacturer": {
                            "type": "string",
                            "description": "Manufacturer name to filter by.",
                        },
                        "model": {
                            "type": "string",
                            "description": "Model name to filter by.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": f"Maximum number of devices to return for this page (default: 20, max: {max_limit})",
                            "default": 20,
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Zero-based pagination offset. Use the next_offset from a previous device discovery response to fetch more results.",
                            "default": 0,
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_entity_details",
                "description": "Get current state plus full serialized entity attributes, aliases, area, floor, labels, and device context for specific entities",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "entity_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of entity IDs to get details for",
                        }
                    },
                    "required": ["entity_ids"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_device_details",
                "description": "Get device metadata, aliases, area/floor/labels, and attached entities for specific Home Assistant devices so you can choose the right entity target for direct control",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "device_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of Home Assistant device IDs to inspect",
                        }
                    },
                    "required": ["device_ids"],
                    "additionalProperties": False,
                },
            },
        ]

        if self._music_assistant_support_enabled():
            tools.extend(
                [
                    {
                        "name": "list_music_assistant_players",
                        "description": "List Music Assistant media_player entities only. Use this to inspect or disambiguate valid Music Assistant playback targets by name, area, floor, or label without mixing in unrelated media_player entities.",
                        "inputSchema": {
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "type": "object",
                            "properties": {
                                "area": {
                                    "type": "string",
                                    "description": "Optional area name or alias to filter Music Assistant players. If the value matches a floor name or alias instead, it will search that floor.",
                                },
                                "floor": {
                                    "type": "string",
                                    "description": "Optional floor name or alias to filter Music Assistant players.",
                                },
                                "label": {
                                    "type": "string",
                                    "description": "Optional label name to filter Music Assistant players.",
                                },
                                "name_contains": {
                                    "type": "string",
                                    "description": "Optional text to match against Music Assistant player names, aliases, related device names, area aliases, and floor aliases.",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": f"Maximum number of Music Assistant players to return (default: 20, max: {max_limit})",
                                    "default": 20,
                                },
                            },
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "name": "play_music_assistant",
                        "description": "Play music using the Home Assistant Music Assistant integration. This resolves only Music Assistant players, supports area/floor/label targeting, and is safer than generic media_player playback when Music Assistant is in use.",
                        "inputSchema": {
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "type": "object",
                            "properties": {
                                "media_type": {
                                    "type": "string",
                                    "enum": ["track", "album", "artist", "playlist", "radio"],
                                    "description": "The type of content to play.",
                                },
                                "media_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "The Music Assistant media identifier, URI, name, or list of items to play.",
                                },
                                "artist": {
                                    "type": "string",
                                    "description": "Optional artist name to narrow track/album playback.",
                                },
                                "album": {
                                    "type": "string",
                                    "description": "Optional album name to narrow track playback.",
                                },
                                "media_description": {
                                    "type": "string",
                                    "description": "Optional natural-language description of the requested media for logging and result summaries.",
                                },
                                "area": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Optional area name or alias, resolved only to Music Assistant players in that area.",
                                },
                                "floor": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Optional floor name or alias, resolved only to Music Assistant players on that floor.",
                                },
                                "label": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Optional label name, resolved only to Music Assistant players carrying that label context.",
                                },
                                "media_player": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Optional Music Assistant player entity_id, friendly name, or alias. Only Music Assistant players are matched.",
                                },
                                "shuffle": {
                                    "type": "boolean",
                                    "description": "Optional shuffle state to apply after starting playback.",
                                },
                                "radio_mode": {
                                    "type": "boolean",
                                    "description": "Optional Music Assistant radio mode flag.",
                                },
                                "enqueue": {
                                    "type": "string",
                                    "enum": ["play", "replace", "next", "replace_next", "add"],
                                    "description": "Optional queue behavior for Music Assistant playback.",
                                },
                            },
                            "required": ["media_type", "media_id"],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "name": "list_music_assistant_instances",
                        "description": "List configured Music Assistant integration instances. Use this when multiple Music Assistant servers are configured and you need a specific instance for library discovery.",
                        "inputSchema": {
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "type": "object",
                            "properties": {},
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "name": "search_music_assistant",
                        "description": "Search the Music Assistant library and providers using a resolved Music Assistant instance. Prefer this over generic service calls when you want LLM-friendly music discovery results and automatic instance resolution.",
                        "inputSchema": {
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The track, artist, album, playlist, or radio station name to search for.",
                                },
                                "media_type": {
                                    "oneOf": [
                                        {
                                            "type": "string",
                                            "enum": ["track", "album", "artist", "playlist", "radio"],
                                        },
                                        {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "enum": ["track", "album", "artist", "playlist", "radio"],
                                            },
                                        },
                                    ],
                                    "description": "Optional Music Assistant media type or list of media types to narrow the search.",
                                },
                                "artist": {
                                    "type": "string",
                                    "description": "Optional artist constraint for track or album searches.",
                                },
                                "album": {
                                    "type": "string",
                                    "description": "Optional album constraint for track searches.",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Optional result limit per media type (default: 10, max: 50).",
                                    "default": 10,
                                },
                                "library_only": {
                                    "type": "boolean",
                                    "description": "When true, limit results to items already in the Music Assistant library.",
                                },
                                "config_entry_id": {
                                    "type": "string",
                                    "description": "Optional Music Assistant config entry ID. Use this when multiple Music Assistant instances exist.",
                                },
                                "instance": {
                                    "type": "string",
                                    "description": "Optional Music Assistant instance title/name. Use list_music_assistant_instances first if needed.",
                                },
                            },
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "name": "get_music_assistant_library",
                        "description": "Browse or filter the Music Assistant library using a resolved Music Assistant instance. Use this for curated discovery like favorite artists, random tracks, or filtered library views.",
                        "inputSchema": {
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "type": "object",
                            "properties": {
                                "media_type": {
                                    "type": "string",
                                    "enum": ["track", "album", "artist", "playlist", "radio"],
                                    "description": "The library media type to list.",
                                },
                                "search": {
                                    "type": "string",
                                    "description": "Optional filter text to narrow the library results.",
                                },
                                "favorite": {
                                    "type": "boolean",
                                    "description": "Optional favorite filter.",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Optional maximum number of results to return (default: 25, max: 100).",
                                    "default": 25,
                                },
                                "offset": {
                                    "type": "integer",
                                    "description": "Optional pagination offset.",
                                    "default": 0,
                                },
                                "order_by": {
                                    "type": "string",
                                    "description": "Optional Music Assistant sort field.",
                                },
                                "album_artists_only": {
                                    "type": "boolean",
                                    "description": "Optional Music Assistant album_artists_only flag for artist library views.",
                                },
                                "album_type": {
                                    "type": "string",
                                    "description": "Optional Music Assistant album_type filter for album library views.",
                                },
                                "config_entry_id": {
                                    "type": "string",
                                    "description": "Optional Music Assistant config entry ID. Use this when multiple Music Assistant instances exist.",
                                },
                                "instance": {
                                    "type": "string",
                                    "description": "Optional Music Assistant instance title/name. Use list_music_assistant_instances first if needed.",
                                },
                            },
                            "required": ["media_type"],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "name": "get_music_assistant_queue",
                        "description": "Read the current Music Assistant queue for specific Music Assistant players. This resolves only Music Assistant players and returns queue details for one or more target players.",
                        "inputSchema": {
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "type": "object",
                            "properties": {
                                "area": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Optional area name or alias to resolve Music Assistant players.",
                                },
                                "floor": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Optional floor name or alias to resolve Music Assistant players.",
                                },
                                "label": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Optional label name to resolve Music Assistant players.",
                                },
                                "media_player": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Optional Music Assistant player entity_id, friendly name, or alias.",
                                },
                            },
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                ]
            )

        tools.extend(
            [
            {
                "name": "list_areas",
                "description": "List all areas in the home with their aliases, entity counts, device counts, floor context, and area labels",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "list_domains",
                "description": "List all available domains with entity counts",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_index",
                "description": "Get the pre-generated system structure index. This index provides a lightweight overview of the Home Assistant system including areas, floors, labels, devices, domains, device classes, people, pets, calendars, zones, automations, scripts, and aliases for alias-capable objects. Call this ONCE at the start of a conversation to understand what exists in the system, then use discover_entities or discover_devices to query specifics. The index is much smaller than a full entity dump.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "list_assist_tools",
                "description": "List the native Home Assistant Assist tools exposed by the built-in Assist LLM API. Use this to inspect the core Assist tool surface or when deciding whether a native Assist tool is a better fit than the custom MCP Assist tools.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "call_assist_tool",
                "description": "Call a native Home Assistant Assist tool directly, using the built-in Assist LLM API rather than the custom MCP Assist tool surface. Use this as a fallback or compatibility path when the native Assist tool behavior is a better fit.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "The exact native Assist tool name to call, for example 'HassTurnOn' or 'GetLiveContext'. Use list_assist_tools first if you are unsure.",
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Arguments to pass to the native Assist tool. This should match the schema returned by list_assist_tools for that tool.",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["tool_name"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_assist_prompt",
                "description": "Get the native Home Assistant Assist prompt text from the built-in Assist LLM API. Use this sparingly for compatibility, debugging, or understanding the core Assist instructions.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_assist_context_snapshot",
                "description": "Get the native Home Assistant Assist live context snapshot, matching the built-in GetLiveContext tool output when available. Use this when a concise whole-home snapshot is helpful.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "perform_action",
                "description": "Control Home Assistant entities by calling services. Use after discovery to turn on/off lights, set temperatures, open/close covers, create calendar events, manage to-do lists, and other write/mutation actions. Prefer entity_id for most direct control; use device_id when intentionally targeting the physical device as a whole.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "The domain of the service to call (e.g., 'light', 'switch', 'climate', 'calendar', 'todo', 'vacuum', 'media_player', etc.)",
                        },
                        "action": {
                            "type": "string",
                            "description": "The service action (e.g., 'turn_on', 'turn_off', 'toggle', 'set_temperature', 'create_event', 'add_item')",
                        },
                        "target": {
                            "type": "object",
                            "description": "Target entities or selector IDs such as areas, floors, labels, or devices",
                            "properties": {
                                "entity_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Single entity ID or list of entity IDs. Preferred for most direct control and for entities that do not belong to a device.",
                                },
                                "area_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Single area ID or list of area IDs. Resolved to exposed entity IDs before the service call.",
                                },
                                "floor_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Single floor ID or list of floor IDs. Resolved to exposed entity IDs before the service call.",
                                },
                                "label_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Single label ID or list of label IDs. Resolved to exposed entity IDs before the service call.",
                                },
                                "device_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Single device ID or list of device IDs. Resolved to exposed attached entity IDs before the service call.",
                                },
                            },
                            "minProperties": 1,
                            "additionalProperties": False,
                        },
                        "data": {
                            "type": "object",
                            "description": "Additional parameters for the service (e.g., brightness: 50, temperature: 22)",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["domain", "action", "target"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_weather_forecast",
                "description": "Get a Home Assistant weather forecast in one call. Prefer this for user weather questions before web search. It finds the weather entity, chooses a supported forecast type, and summarizes today or tomorrow.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Optional specific weather entity ID to use.",
                        },
                        "area": {
                            "type": "string",
                            "description": "Optional area or room name if you have multiple weather entities.",
                        },
                        "floor": {
                            "type": "string",
                            "description": "Optional floor name if you have multiple weather entities.",
                        },
                        "label": {
                            "type": "string",
                            "description": "Optional label name if you have multiple weather entities.",
                        },
                        "name_contains": {
                            "type": "string",
                            "description": "Optional text to match a specific weather entity name or alias.",
                        },
                        "when": {
                            "type": "string",
                            "description": "Forecast day to summarize. Use 'today', 'tomorrow', or a local date like '2026-04-13'. Defaults to 'tomorrow'.",
                        },
                        "forecast_type": {
                            "type": "string",
                            "enum": ["daily", "twice_daily", "hourly"],
                            "description": "Optional forecast type override. If unsupported by the target entity, the tool falls back to a supported type.",
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_calendar_events",
                "description": "Get upcoming Home Assistant calendar events in one call. Prefer this for questions like 'When is the next Mariners game?' or 'What's on our calendar tomorrow?'. It discovers matching calendars, calls calendar.get_events, and summarizes the next matching event or agenda.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Optional specific calendar entity ID to use.",
                        },
                        "area": {
                            "type": "string",
                            "description": "Optional area or room name if you want calendars associated with a specific area.",
                        },
                        "floor": {
                            "type": "string",
                            "description": "Optional floor name if you want calendars associated with a specific floor.",
                        },
                        "label": {
                            "type": "string",
                            "description": "Optional label name if you want calendars tagged with a specific label.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Optional general text to match either calendar names or event details, for example 'Mariners' or 'dentist'.",
                        },
                        "name_contains": {
                            "type": "string",
                            "description": "Optional text to match a specific calendar entity name or alias.",
                        },
                        "event_text": {
                            "type": "string",
                            "description": "Optional text to match event summary, title, description, or location.",
                        },
                        "when": {
                            "type": "string",
                            "description": "Optional time window anchor. Use 'now', 'today', 'tomorrow', a local date like '2026-04-13', or an ISO datetime.",
                        },
                        "days": {
                            "type": "integer",
                            "description": "How many days forward to search when 'when' is omitted or when searching from a specific start time. Defaults to 60 for upcoming searches and 1 for day-specific lookups.",
                            "default": 60,
                            "minimum": 1,
                            "maximum": 365,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of matching events to summarize.",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "list_response_services",
                "description": "List Home Assistant services that currently support native response data. Use this when you need to discover which read/query-style services can be called with call_service_with_response.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "Optional domain filter, for example 'weather', 'calendar', or 'media_player'.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Optional text filter matching domain, service, name, or description.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of services to return (default: 50, max: 200).",
                            "default": 50,
                            "minimum": 1,
                            "maximum": 200,
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "call_service_with_response",
                "description": "Call a Home Assistant service that returns structured response data for read/query use cases. Use this for native service-response reads like calendar or to-do queries, media browsing/searching, or integration-specific query data. For normal weather questions, prefer get_weather_forecast.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "The Home Assistant domain for the response-returning service, for example 'weather', 'calendar', 'todo', or 'media_player'.",
                        },
                        "service": {
                            "type": "string",
                            "description": "The service/action name to call, for example 'get_forecasts', 'get_events', 'get_items', 'browse_media', or 'search_media'.",
                        },
                        "target": {
                            "type": "object",
                            "description": "Optional target entities or selector IDs such as areas, floors, labels, or devices. Selectors are resolved to exposed entity IDs, and may be narrowed using the service's target metadata when available.",
                            "properties": {
                                "entity_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Single entity ID or list of entity IDs.",
                                },
                                "area_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Single area ID or list of area IDs.",
                                },
                                "floor_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Single floor ID or list of floor IDs.",
                                },
                                "label_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Single label ID or list of label IDs.",
                                },
                                "device_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Single device ID or list of device IDs.",
                                },
                            },
                            "minProperties": 1,
                            "additionalProperties": False,
                        },
                        "data": {
                            "type": "object",
                            "description": "Additional service data for the response-returning service. Required fields are validated from Home Assistant's native service descriptions when available.",
                            "additionalProperties": True,
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 60, max: 300).",
                            "default": 60,
                            "minimum": 1,
                            "maximum": 300,
                        },
                    },
                    "required": ["domain", "service"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "set_conversation_state",
                "description": "Indicate whether you expect a response from the user after your message",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "expecting_response": {
                            "type": "boolean",
                            "description": "true if expecting user response, false if task is complete",
                        }
                    },
                    "required": ["expecting_response"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "run_script",
                "description": "Execute a Home Assistant script and return its response variables. Use this for scripts that return data (e.g., camera analysis, calculations). Returns the script's response variables.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "script_id": {
                            "type": "string",
                            "description": "The script entity ID discovered from Home Assistant (for example, 'script.some_script_name' or just 'some_script_name').",
                        },
                        "variables": {
                            "type": "object",
                            "description": "Variables to pass to the script",
                            "additionalProperties": True,
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 60)",
                            "default": 60,
                        },
                    },
                    "required": ["script_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "run_automation",
                "description": "Trigger a Home Assistant automation with optional variables. Use this to manually trigger automations.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "automation_id": {
                            "type": "string",
                            "description": "The automation entity ID (e.g., 'automation.notify_on_motion' or just 'notify_on_motion')",
                        },
                        "variables": {
                            "type": "object",
                            "description": "Variables to pass to the automation (available as trigger.variables)",
                            "additionalProperties": True,
                        },
                        "skip_conditions": {
                            "type": "boolean",
                            "description": "Whether to skip the automation's conditions (default: false)",
                            "default": False,
                        },
                    },
                    "required": ["automation_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_entity_history",
                "description": "Get recorder-backed history for a specific entity. By default this returns a recent timeline, and with mode='last_event' it returns only the most recent matching event or change.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The entity ID to get history for. Use discover_entities first and pass the discovered entity ID here.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["timeline", "last_event"],
                            "default": "timeline",
                            "description": "timeline returns recent history entries; last_event returns only the most recent matching event or change.",
                        },
                        "event": {
                            "type": "string",
                            "description": "Optional semantic event filter for last_event mode, such as opened, closed, on, off, locked, unlocked, detected, cleared, home, or away.",
                        },
                        "state": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            ],
                            "description": "Optional target state or states to filter by. Works with both timeline and last_event modes.",
                        },
                        "hours": {
                            "type": "integer",
                            "description": "Number of hours of recorder history to search. In timeline mode the default is 24 hours; in last_event mode the default is 720 hours (30 days). Max: 8760 hours / 1 year.",
                            "minimum": 1,
                            "maximum": 8760,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of timeline entries to return in timeline mode (default: 50, max: 100). Most recent changes shown first.",
                            "default": 50,
                            "minimum": 1,
                            "maximum": 100,
                        },
                    },
                    "required": ["entity_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "analyze_entity_history",
                "description": "Analyze Home Assistant recorder history for aggregate questions such as 'how many times was the door opened in the last hour?', 'how long has it been locked?', or 'how often did this sensor trigger today?'. Can count all changes or matching states/events.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The entity ID to analyze in recorder history.",
                        },
                        "event": {
                            "type": "string",
                            "description": "Optional semantic event to analyze, such as opened, closed, on, off, locked, unlocked, detected, cleared, home, or away.",
                        },
                        "state": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            ],
                            "description": "Optional specific target state or states to analyze. If omitted, counts all recorded state changes in the time window.",
                        },
                        "hours": {
                            "type": "integer",
                            "description": "How far back to analyze recorder history. Default is 24 hours for count/summary/duration/stats, and 720 hours (30 days) for streak. Max: 8760 hours / 1 year.",
                            "minimum": 1,
                            "maximum": 8760,
                        },
                        "analysis": {
                            "type": "string",
                            "enum": ["count", "summary", "duration", "streak", "stats"],
                            "default": "count",
                            "description": "count returns how many matching events occurred; summary also includes the first and last matching times in the window; duration reports total time spent in the matching state; streak reports how long the entity has continuously been in the matching state right now; stats reports numeric min/max/average over the window.",
                        },
                    },
                    "required": ["entity_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_entity_state_at_time",
                "description": "Look up the recorder state of an entity at a specific date/time. Use this for questions like 'was the gate open at 2 PM?' or 'what was the temperature at 9 this morning?'",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The entity ID to inspect in recorder history.",
                        },
                        "datetime": {
                            "type": "string",
                            "description": "The target date/time to inspect, preferably as an ISO 8601 timestamp. If no timezone is included, Home Assistant local time is assumed.",
                        },
                    },
                    "required": ["entity_id", "datetime"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "remember_memory",
                "description": "Store a short fact, preference, or instruction for later recall. Use this only when the user explicitly asks you to remember something. Memories persist across conversations and automatically expire after a TTL.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "memory": {
                            "type": "string",
                            "description": "The fact, preference, or instruction to store.",
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional short category such as 'preference', 'household', or 'schedule'.",
                        },
                        "ttl_days": {
                            "type": "integer",
                            "description": "Optional retention time in days. If omitted, the shared default TTL is used and capped by the shared maximum TTL.",
                            "minimum": 1,
                            "maximum": 3650,
                        },
                    },
                    "required": ["memory"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "recall_memories",
                "description": "Search active stored memories by query or category, or list recent memories when no query is given. Use this for requests like 'what do you remember about my coffee preference?'",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Optional search text to match against stored memory text.",
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional category filter.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of memories to return (default: 5).",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 5,
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "forget_memory",
                "description": "Delete one stored memory by id or by query/category match. Use this when the user asks you to forget or update something previously remembered.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "Specific memory id to delete.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search text to find a memory to delete when the id is not known.",
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional category filter when deleting by query.",
                        },
                        "forget_all_matches": {
                            "type": "boolean",
                            "description": "Delete every matching memory instead of only the best match.",
                            "default": False,
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            ]
        )
        tools.extend(self._get_media_tool_definitions())

        # Add custom tool definitions if enabled
        if self.custom_tools:
            try:
                custom_tool_defs = self.custom_tools.get_tool_definitions()
                tools.extend(custom_tool_defs)
            except Exception as e:
                _LOGGER.error(f"Failed to get custom tool definitions: {e}")

        tools = [tool for tool in tools if self._is_tool_enabled(tool["name"])]
        self._cached_tools_list = list(tools)
        self._cached_tools_signature = signature

        # nextCursor is optional - omit if not paginating
        return {"tools": tools}

    async def handle_tool_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        context = params.get("context") or {}

        _LOGGER.debug("Calling tool: %s with args: %s", tool_name, arguments)

        if not self._is_tool_enabled(tool_name):
            raise ValueError(
                f"Tool '{tool_name}' is disabled in shared MCP settings."
            )

        if tool_name == "discover_entities":
            return await self.tool_discover_entities(arguments)
        elif tool_name == "discover_devices":
            return await self.tool_discover_devices(arguments)
        elif tool_name == "get_entity_details":
            return await self.tool_get_entity_details(arguments)
        elif tool_name == "get_device_details":
            return await self.tool_get_device_details(arguments)
        elif tool_name == "list_music_assistant_players":
            return await self.tool_list_music_assistant_players(arguments)
        elif tool_name == "play_music_assistant":
            return await self.tool_play_music_assistant(arguments)
        elif tool_name == "list_music_assistant_instances":
            return await self.tool_list_music_assistant_instances(arguments)
        elif tool_name == "search_music_assistant":
            return await self.tool_search_music_assistant(arguments)
        elif tool_name == "get_music_assistant_library":
            return await self.tool_get_music_assistant_library(arguments)
        elif tool_name == "get_music_assistant_queue":
            return await self.tool_get_music_assistant_queue(arguments)
        elif tool_name == "list_areas":
            return await self.tool_list_areas()
        elif tool_name == "list_domains":
            return await self.tool_list_domains()
        elif tool_name == "get_index":
            return await self.tool_get_index()
        elif tool_name == "list_assist_tools":
            return await self.tool_list_assist_tools(arguments)
        elif tool_name == "call_assist_tool":
            return await self.tool_call_assist_tool(arguments)
        elif tool_name == "get_assist_prompt":
            return await self.tool_get_assist_prompt(arguments)
        elif tool_name == "get_assist_context_snapshot":
            return await self.tool_get_assist_context_snapshot(arguments)
        elif tool_name == "perform_action":
            return await self.tool_perform_action(arguments)
        elif tool_name == "get_weather_forecast":
            return await self.tool_get_weather_forecast(arguments)
        elif tool_name == "get_calendar_events":
            return await self.tool_get_calendar_events(arguments)
        elif tool_name == "list_response_services":
            return await self.tool_list_response_services(arguments)
        elif tool_name == "call_service_with_response":
            return await self.tool_call_service_with_response(arguments)
        elif tool_name == "set_conversation_state":
            return await self.tool_set_conversation_state(arguments)
        elif tool_name == "run_script":
            return await self.tool_run_script(arguments)
        elif tool_name == "run_automation":
            return await self.tool_run_automation(arguments)
        elif tool_name == "get_entity_history":
            return await self.tool_get_entity_history(arguments)
        elif tool_name == "get_last_entity_event":
            return await self.tool_get_last_entity_event(arguments)
        elif tool_name == "analyze_entity_history":
            return await self.tool_analyze_entity_history(arguments)
        elif tool_name == "get_entity_state_at_time":
            return await self.tool_get_entity_state_at_time(arguments)
        elif tool_name == "remember_memory":
            return await self.tool_remember_memory(arguments)
        elif tool_name == "recall_memories":
            return await self.tool_recall_memories(arguments)
        elif tool_name == "forget_memory":
            return await self.tool_forget_memory(arguments)
        elif tool_name == "analyze_image":
            return await self.tool_analyze_image(arguments, context=context)
        elif tool_name == "get_image":
            return await self.tool_get_image(arguments)
        elif tool_name == "generate_image":
            return await self.tool_generate_image(arguments, context=context)
        else:
            # Check if it's a custom tool
            if self.custom_tools and self.custom_tools.is_custom_tool(tool_name):
                return await self.custom_tools.handle_tool_call(
                    tool_name,
                    arguments,
                    context=context,
                )
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

    def _build_text_tool_result(
        self,
        text: str,
        *,
        is_error: bool = False,
        structured_content: dict[str, Any] | None = None,
        extra_content: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a standard MCP text result with optional structured content."""
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if extra_content:
            content.extend(extra_content)

        result: dict[str, Any] = {"content": content, "isError": is_error}
        if structured_content is not None:
            result["structuredContent"] = structured_content
        return result

    def _resolve_profile_entry(self, context: dict[str, Any] | None) -> Any:
        """Resolve the active conversation profile entry for a tool call."""
        if isinstance(context, dict):
            profile_entry_id = str(context.get("profile_entry_id") or "").strip()
            if profile_entry_id:
                entry = self.hass.config_entries.async_get_entry(profile_entry_id)
                if entry is not None:
                    return entry
        return self.entry

    @staticmethod
    def _get_entry_value(entry: Any, key: str, default: Any) -> Any:
        """Read a config-entry value from options first, then data."""
        if entry is None:
            return default
        value = entry.options.get(key, entry.data.get(key))
        return default if value is None else value

    def _get_model_provider_config(
        self, context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Resolve the current profile's model/provider config for media tools."""
        entry = self._resolve_profile_entry(context)
        server_type = str(
            self._get_entry_value(entry, CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)
        )
        model_name = str(
            self._get_entry_value(entry, CONF_MODEL_NAME, DEFAULT_MODEL_NAME)
        )
        api_key = str(self._get_entry_value(entry, CONF_API_KEY, DEFAULT_API_KEY) or "")
        timeout = int(self._get_entry_value(entry, CONF_TIMEOUT, DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT)
        configured_url = str(
            self._get_entry_value(entry, CONF_LMSTUDIO_URL, DEFAULT_LMSTUDIO_URL)
            or DEFAULT_LMSTUDIO_URL
        ).rstrip("/")

        if server_type == SERVER_TYPE_OPENAI:
            base_url = str(
                self._get_entry_value(entry, CONF_LMSTUDIO_URL, OPENAI_BASE_URL)
                or OPENAI_BASE_URL
            ).rstrip("/")
        elif server_type == SERVER_TYPE_GEMINI:
            base_url = GEMINI_BASE_URL.rstrip("/")
        elif server_type == SERVER_TYPE_ANTHROPIC:
            base_url = ANTHROPIC_BASE_URL.rstrip("/")
        elif server_type == SERVER_TYPE_OPENROUTER:
            base_url = OPENROUTER_BASE_URL.rstrip("/")
        elif server_type == SERVER_TYPE_MOLTBOT:
            base_url = configured_url or DEFAULT_MOLTBOT_URL
        elif server_type == SERVER_TYPE_VLLM:
            base_url = configured_url or DEFAULT_VLLM_URL
        elif server_type == SERVER_TYPE_OLLAMA:
            base_url = configured_url or DEFAULT_OLLAMA_URL
        elif server_type == "llamacpp":
            base_url = configured_url or DEFAULT_LLAMACPP_URL
        else:
            base_url = configured_url or DEFAULT_LMSTUDIO_URL

        return {
            "entry": entry,
            "server_type": server_type,
            "model_name": model_name,
            "api_key": api_key,
            "timeout": timeout,
            "base_url": base_url.rstrip("/"),
        }

    def _get_model_auth_headers(self, provider_config: dict[str, Any]) -> dict[str, str]:
        """Build provider auth headers using the same rules as the conversation agent."""
        server_type = provider_config["server_type"]
        api_key = provider_config["api_key"]
        if server_type == SERVER_TYPE_OPENAI:
            if (
                api_key
                and len(api_key) > 5
                and api_key.lower() not in {"none", "null", "fake", "na", "n/a"}
            ):
                return {"Authorization": f"Bearer {api_key}"}
            return {}
        if server_type in {
            SERVER_TYPE_GEMINI,
            SERVER_TYPE_ANTHROPIC,
            SERVER_TYPE_MOLTBOT,
        }:
            return {"Authorization": f"Bearer {api_key}"}
        if server_type == SERVER_TYPE_OPENROUTER:
            return {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/mike-nott/mcp-assist",
                "X-Title": "MCP Assist for Home Assistant",
            }
        return {}

    async def tool_get_image(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch an image and return it as an MCP image content block."""
        try:
            image_bytes, mime_type, source = await self._resolve_image_source(args)
        except Exception as err:
            return self._build_text_tool_result(str(err), is_error=True)

        return self._build_text_tool_result(
            f"Fetched image from {source['description']}.",
            structured_content={
                "source": source,
                "mime_type": mime_type,
                "size_bytes": len(image_bytes),
            },
            extra_content=[self._build_image_content_block(image_bytes, mime_type)],
        )

    async def tool_analyze_image(
        self,
        args: Dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Analyze an image or live camera snapshot with the active profile model."""
        try:
            image_bytes, mime_type, source = await self._resolve_image_source(args)
            answer = await self._analyze_image_with_provider(
                question=str(args.get("question") or "").strip()
                or "Describe the image briefly and focus on factual observations.",
                image_bytes=image_bytes,
                mime_type=mime_type,
                detail=str(args.get("detail") or "auto"),
                context=context,
            )
        except Exception as err:
            return self._build_text_tool_result(str(err), is_error=True)

        extra_content: list[dict[str, Any]] = []
        if bool(args.get("include_image")):
            extra_content.append(self._build_image_content_block(image_bytes, mime_type))

        return self._build_text_tool_result(
            answer,
            structured_content={
                "answer": answer,
                "source": source,
                "mime_type": mime_type,
                "size_bytes": len(image_bytes),
            },
            extra_content=extra_content,
        )

    async def tool_generate_image(
        self,
        args: Dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Generate an image with the active profile provider when supported."""
        prompt = str(args.get("prompt") or "").strip()
        if not prompt:
            return self._build_text_tool_result(
                "generate_image requires a prompt.",
                is_error=True,
            )

        try:
            image_bytes, mime_type, metadata = await self._generate_image_with_provider(
                prompt=prompt,
                size=str(args.get("size") or "").strip() or None,
                quality=str(args.get("quality") or "").strip() or None,
                style=str(args.get("style") or "").strip() or None,
                background=str(args.get("background") or "").strip() or None,
                context=context,
            )
        except Exception as err:
            return self._build_text_tool_result(str(err), is_error=True)

        return self._build_text_tool_result(
            "Generated image successfully.",
            structured_content=metadata,
            extra_content=[self._build_image_content_block(image_bytes, mime_type)],
        )

    async def _resolve_image_source(
        self,
        args: dict[str, Any],
    ) -> tuple[bytes, str, dict[str, Any]]:
        """Resolve one image source from a camera, entity, URL, or local path."""
        source_fields = {
            "camera_entity_id": str(args.get("camera_entity_id") or "").strip(),
            "entity_id": str(args.get("entity_id") or "").strip(),
            "image_url": str(args.get("image_url") or "").strip(),
            "image_path": str(args.get("image_path") or "").strip(),
        }
        provided = [key for key, value in source_fields.items() if value]
        if len(provided) != 1:
            raise ValueError(
                "Provide exactly one of camera_entity_id, entity_id, image_url, or image_path."
            )

        field_name = provided[0]
        field_value = source_fields[field_name]
        if field_name == "camera_entity_id":
            image_bytes, mime_type = await self._capture_camera_image(field_value)
            return (
                image_bytes,
                mime_type,
                {
                    "type": "camera_entity_id",
                    "value": field_value,
                    "description": f"camera {field_value}",
                },
            )
        if field_name == "entity_id":
            image_bytes, mime_type, description = await self._resolve_entity_image(
                field_value
            )
            return (
                image_bytes,
                mime_type,
                {
                    "type": "entity_id",
                    "value": field_value,
                    "description": description,
                },
            )
        if field_name == "image_url":
            image_bytes, mime_type = await self._fetch_image_reference(field_value)
            return (
                image_bytes,
                mime_type,
                {
                    "type": "image_url",
                    "value": field_value,
                    "description": f"URL {field_value}",
                },
            )

        local_path = self._resolve_local_image_path(field_value)
        image_bytes, mime_type = self._read_local_image_path(local_path)
        return (
            image_bytes,
            mime_type,
            {
                "type": "image_path",
                "value": str(local_path),
                "description": f"file {local_path.name}",
            },
        )

    async def _capture_camera_image(self, entity_id: str) -> tuple[bytes, str]:
        """Capture a camera snapshot using Home Assistant's native camera helper."""
        if not entity_id.startswith("camera."):
            raise ValueError("camera_entity_id must reference a camera entity.")

        from homeassistant.components.camera import async_get_image

        image = await async_get_image(self.hass, entity_id, timeout=10)
        image_bytes = getattr(image, "content", None)
        mime_type = getattr(image, "content_type", None)
        if not isinstance(image_bytes, (bytes, bytearray)):
            raise ValueError(f"Unable to capture image from {entity_id}.")
        return self._normalize_image_payload(bytes(image_bytes), mime_type, entity_id)

    async def _resolve_entity_image(
        self,
        entity_id: str,
    ) -> tuple[bytes, str, str]:
        """Resolve an image from a supported image-like entity."""
        state = self.hass.states.get(entity_id)
        if state is None:
            raise ValueError(f"Entity {entity_id!r} was not found.")

        if entity_id.startswith("camera."):
            image_bytes, mime_type = await self._capture_camera_image(entity_id)
            return image_bytes, mime_type, f"camera {entity_id}"

        entity_picture = (
            str(state.attributes.get("entity_picture_local") or "").strip()
            or str(state.attributes.get("entity_picture") or "").strip()
        )
        if not entity_picture:
            raise ValueError(
                f"Entity {entity_id!r} does not expose a usable picture URL."
            )

        image_bytes, mime_type = await self._fetch_image_reference(entity_picture)
        return image_bytes, mime_type, f"entity picture for {entity_id}"

    async def _fetch_image_reference(self, reference: str) -> tuple[bytes, str]:
        """Fetch image bytes from a data URL, local URL, or remote URL."""
        if reference.startswith("data:"):
            return self._parse_data_url_image(reference)
        if reference.startswith("/local/") or reference.startswith("/media/local/"):
            local_path = self._resolve_local_image_path(reference)
            return self._read_local_image_path(local_path)
        if not reference.startswith(("http://", "https://")):
            raise ValueError(
                "Only data URLs, /local URLs, /media/local URLs, and http(s) image URLs are supported."
            )

        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(reference) as response:
                if response.status != 200:
                    raise ValueError(
                        f"Unable to fetch image URL {reference!r}: HTTP {response.status}"
                    )
                image_bytes = await response.read()
                mime_type = response.headers.get("Content-Type")
        return self._normalize_image_payload(image_bytes, mime_type, reference)

    def _parse_data_url_image(self, reference: str) -> tuple[bytes, str]:
        """Decode a data URL into image bytes."""
        match = re.match(
            r"^data:(?P<mime>[-\w.+/]+);base64,(?P<data>[A-Za-z0-9+/=\s]+)$",
            reference,
            flags=re.IGNORECASE,
        )
        if not match:
            raise ValueError("Unsupported data URL format for image input.")
        mime_type = match.group("mime")
        image_bytes = base64.b64decode(match.group("data"), validate=False)
        return self._normalize_image_payload(image_bytes, mime_type, "data-url")

    def _resolve_local_image_path(self, reference: str) -> Path:
        """Resolve a local image path inside the Home Assistant config directory."""
        config_root = Path(self.hass.config.path("")).resolve()
        if reference.startswith("/local/"):
            candidate = (config_root / "www" / reference.removeprefix("/local/")).resolve()
        elif reference.startswith("/media/local/"):
            candidate = (
                config_root / "media" / reference.removeprefix("/media/local/")
            ).resolve()
        else:
            raw_path = Path(reference)
            if raw_path.is_absolute():
                candidate = raw_path.resolve()
            else:
                candidate = (config_root / raw_path).resolve()

        try:
            candidate.relative_to(config_root)
        except ValueError as err:
            raise ValueError(
                "Local image paths must stay inside the Home Assistant config directory."
            ) from err

        if not candidate.is_file():
            raise ValueError(f"Image file was not found: {candidate}")
        return candidate

    def _read_local_image_path(self, path: Path) -> tuple[bytes, str]:
        """Read and validate a local image file."""
        image_bytes = path.read_bytes()
        mime_type = mimetypes.guess_type(path.name)[0]
        return self._normalize_image_payload(image_bytes, mime_type, str(path))

    def _normalize_image_payload(
        self,
        image_bytes: bytes,
        mime_type: str | None,
        source_label: str,
    ) -> tuple[bytes, str]:
        """Validate an image payload and normalize its mime type."""
        if not image_bytes:
            raise ValueError(f"No image bytes were available from {source_label}.")
        if len(image_bytes) > _MAX_INLINE_IMAGE_BYTES:
            raise ValueError(
                f"Image source {source_label!r} is too large ({len(image_bytes)} bytes)."
            )

        normalized_mime = str(mime_type or "").split(";", 1)[0].strip().lower()
        if not normalized_mime:
            normalized_mime = mimetypes.guess_type(source_label)[0] or "image/jpeg"
        if not normalized_mime.startswith("image/"):
            raise ValueError(
                f"Source {source_label!r} did not resolve to an image mime type."
            )

        return image_bytes, normalized_mime

    def _build_image_content_block(
        self,
        image_bytes: bytes,
        mime_type: str,
    ) -> dict[str, Any]:
        """Build an MCP image content block."""
        return {
            "type": "image",
            "mimeType": mime_type,
            "data": base64.b64encode(image_bytes).decode("ascii"),
        }

    async def _analyze_image_with_provider(
        self,
        *,
        question: str,
        image_bytes: bytes,
        mime_type: str,
        detail: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Run image analysis through the active profile provider."""
        provider_config = self._get_model_provider_config(context)
        server_type = provider_config["server_type"]
        timeout = aiohttp.ClientTimeout(total=provider_config["timeout"])
        headers = self._get_model_auth_headers(provider_config)
        base_url = provider_config["base_url"]

        if server_type == SERVER_TYPE_OLLAMA:
            payload = {
                "model": provider_config["model_name"],
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                        "images": [base64.b64encode(image_bytes).decode("ascii")],
                    }
                ],
            }
            url = f"{base_url}/api/chat"
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        raise ValueError(
                            f"Image analysis failed for {server_type}: HTTP {response.status} {await response.text()}"
                        )
                    data = await response.json()
            return str(data.get("message", {}).get("content") or "").strip()

        payload = {
            "model": provider_config["model_name"],
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}",
                                "detail": detail if detail in {"auto", "low", "high"} else "auto",
                            },
                        },
                    ],
                }
            ],
        }
        url = f"{base_url}/v1/chat/completions"
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise ValueError(
                        f"Image analysis failed for {server_type}: HTTP {response.status} {await response.text()}"
                    )
                data = await response.json()

        message = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
        return self._extract_provider_message_text(message).strip()

    async def _generate_image_with_provider(
        self,
        *,
        prompt: str,
        size: str | None,
        quality: str | None,
        style: str | None,
        background: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[bytes, str, dict[str, Any]]:
        """Generate an image through an OpenAI-compatible images API when supported."""
        provider_config = self._get_model_provider_config(context)
        server_type = provider_config["server_type"]
        if server_type == SERVER_TYPE_OLLAMA:
            raise ValueError(
                "Image generation is not supported for Ollama profiles through MCP Assist yet."
            )

        payload: dict[str, Any] = {
            "model": provider_config["model_name"],
            "prompt": prompt,
            "response_format": "b64_json",
        }
        if size:
            payload["size"] = size
        if quality:
            payload["quality"] = quality
        if style:
            payload["style"] = style
        if background:
            payload["background"] = background

        timeout = aiohttp.ClientTimeout(total=provider_config["timeout"])
        headers = self._get_model_auth_headers(provider_config)
        url = f"{provider_config['base_url']}/v1/images/generations"

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise ValueError(
                        f"Image generation failed for {server_type}: HTTP {response.status} {await response.text()}"
                    )
                data = await response.json()

        items = data.get("data")
        if not isinstance(items, list) or not items:
            raise ValueError("The image generation provider did not return any images.")
        item = items[0]
        if not isinstance(item, dict):
            raise ValueError("Unexpected image generation payload from provider.")

        image_bytes: bytes
        mime_type = "image/png"
        if item.get("b64_json"):
            image_bytes = base64.b64decode(str(item["b64_json"]), validate=False)
        elif item.get("url"):
            image_bytes, mime_type = await self._fetch_image_reference(str(item["url"]))
        else:
            raise ValueError("The image generation provider returned no usable image data.")

        image_bytes, mime_type = self._normalize_image_payload(
            image_bytes,
            mime_type,
            "generated-image",
        )
        metadata = {
            "prompt": prompt,
            "provider": server_type,
            "model": provider_config["model_name"],
            "mime_type": mime_type,
            "size_bytes": len(image_bytes),
        }
        revised_prompt = str(item.get("revised_prompt") or "").strip()
        if revised_prompt:
            metadata["revised_prompt"] = revised_prompt

        return image_bytes, mime_type, metadata

    @staticmethod
    def _extract_provider_message_text(message_content: Any) -> str:
        """Extract text from provider chat-completion message content."""
        if isinstance(message_content, str):
            return message_content
        if isinstance(message_content, list):
            parts: list[str] = []
            for item in message_content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text = str(item.get("text") or "").strip()
                    if text:
                        parts.append(text)
            if parts:
                return "\n".join(parts)
        return json.dumps(message_content, ensure_ascii=False)

    async def tool_discover_entities(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Discover entities based on criteria with progress notifications."""
        limit = self._coerce_int_arg(
            args.get("limit"),
            default=20,
            minimum=1,
            maximum=MAX_ENTITIES_PER_DISCOVERY,
        )
        offset = self._coerce_int_arg(
            args.get("offset"),
            default=0,
            minimum=0,
            maximum=10000,
        )

        # Notify start
        self.publish_progress(
            "tool_start",
            "Starting entity discovery",
            tool="discover_entities",
            args=args,
        )

        page = await self.discovery.discover_entities_page(
            entity_type=args.get("entity_type"),
            area=args.get("area"),
            floor=args.get("floor"),
            label=args.get("label"),
            domain=args.get("domain"),
            state=args.get("state"),
            name_contains=args.get("name_contains"),
            limit=limit,
            offset=offset,
            device_class=args.get("device_class"),
            name_pattern=args.get("name_pattern"),
            inferred_type=args.get("inferred_type"),
        )
        entities = page["items"]

        # Notify completion
        self.publish_progress(
            "tool_complete",
            (
                "Discovery complete: "
                f"returned {page['returned_count']} of {page['total_found']} entities"
            ),
            tool="discover_entities",
            count=page["returned_count"],
            total=page["total_found"],
        )

        # Format results based on whether it's smart discovery or general
        return self._format_discovery_results(entities, args, page)

    async def tool_discover_devices(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Discover devices based on criteria."""
        limit = self._coerce_int_arg(
            args.get("limit"),
            default=20,
            minimum=1,
            maximum=MAX_ENTITIES_PER_DISCOVERY,
        )
        offset = self._coerce_int_arg(
            args.get("offset"),
            default=0,
            minimum=0,
            maximum=10000,
        )

        self.publish_progress(
            "tool_start",
            "Starting device discovery",
            tool="discover_devices",
            args=args,
        )

        page = await self.discovery.discover_devices_page(
            area=args.get("area"),
            floor=args.get("floor"),
            label=args.get("label"),
            domain=args.get("domain"),
            name_contains=args.get("name_contains"),
            manufacturer=args.get("manufacturer"),
            model=args.get("model"),
            limit=limit,
            offset=offset,
        )
        devices = page["items"]

        self.publish_progress(
            "tool_complete",
            (
                "Device discovery complete: "
                f"returned {page['returned_count']} of {page['total_found']} devices"
            ),
            tool="discover_devices",
            count=page["returned_count"],
            total=page["total_found"],
        )

        if not devices:
            if page["total_found"] > 0:
                page_header = self._build_paging_header(
                    noun="devices",
                    total_found=page["total_found"],
                    returned_count=page["returned_count"],
                    offset=page["offset"],
                    next_offset=page["next_offset"],
                )
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"{page_header}. No devices were returned for this page.",
                        }
                    ],
                    "devices": [],
                    "pagination": page,
                }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "No devices found matching the search criteria.",
                    }
                ]
            }

        header = self._build_paging_header(
            noun="devices",
            total_found=page["total_found"],
            returned_count=page["returned_count"],
            offset=page["offset"],
            next_offset=page["next_offset"],
        )
        text_parts = [
            (
                f"{header} (use get_device_details to inspect attached entities; "
                "prefer entity targets for most direct control):"
            )
        ]
        for device in devices:
            detail_parts = [f"{device['entity_count']} entities"]
            if device.get("domains"):
                detail_parts.append(f"Domains: {', '.join(device['domains'])}")
            if device.get("match_reasons"):
                detail_parts.append(f"Matched on: {', '.join(device['match_reasons'])}")
            if device.get("device_aliases"):
                detail_parts.append(f"Aliases: {', '.join(device['device_aliases'])}")
            if device.get("area"):
                detail_parts.append(f"Area: {device['area']}")
            if device.get("floor"):
                detail_parts.append(f"Floor: {device['floor']}")
            if device.get("manufacturer"):
                detail_parts.append(f"Manufacturer: {device['manufacturer']}")
            if device.get("model"):
                detail_parts.append(f"Model: {device['model']}")
            if device.get("labels"):
                detail_parts.append(f"Labels: {', '.join(device['labels'])}")
            if device.get("entities_preview"):
                preview_ids = [entity["entity_id"] for entity in device["entities_preview"][:3]]
                extra_count = max(device["entity_count"] - len(preview_ids), 0)
                preview_text = ", ".join(preview_ids)
                if extra_count:
                    preview_text += f", +{extra_count} more"
                detail_parts.append(f"Related entities: {preview_text}")
            text_parts.append(
                f"- {device['device_id']}: {device['name']} ({', '.join(detail_parts)})"
            )

        return {
            "content": [{"type": "text", "text": "\n".join(text_parts)}],
            "devices": devices,
            "pagination": page,
        }

    def _build_paging_header(
        self,
        *,
        noun: str,
        total_found: int,
        returned_count: int,
        offset: int,
        next_offset: int | None,
    ) -> str:
        """Build a compact human-readable paging header."""
        if total_found <= 0:
            return f"Found 0 {noun}"

        if returned_count <= 0:
            return f"No {noun} at offset {offset}; {total_found} total available"

        start_number = offset + 1
        end_number = offset + returned_count

        if total_found > returned_count or offset > 0:
            header = f"Showing {start_number}-{end_number} of {total_found} {noun}"
        else:
            header = f"Found {total_found} {noun}"

        if next_offset is not None:
            header += f"; {total_found - end_number} more available (next_offset={next_offset})"

        return header

    def _format_discovery_results(
        self,
        entities: List[Dict[str, Any]],
        args: Dict[str, Any],
        pagination: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Format discovery results for the LLM, handling both smart and general discovery."""
        pagination = pagination or {
            "total_found": len(entities),
            "returned_count": len(entities),
            "offset": 0,
            "next_offset": None,
        }

        if not entities:
            if pagination.get("total_found", 0) > 0:
                page_header = self._build_paging_header(
                    noun="entities",
                    total_found=pagination["total_found"],
                    returned_count=pagination.get("returned_count", 0),
                    offset=pagination.get("offset", 0),
                    next_offset=pagination.get("next_offset"),
                )
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"{page_header}. No entities were returned for this page.",
                        }
                    ],
                    "entities": [],
                    "pagination": pagination,
                }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "No entities found matching the search criteria.",
                    }
                ]
            }

        # Check if this is a smart discovery result (has summary metadata)
        has_summary = entities and entities[0].get("entity_id") == "_summary"

        if has_summary:
            # Smart discovery with grouping
            summary = entities[0]
            actual_entities = entities[1:]

            # Build formatted text
            text_parts = []

            # Add summary header
            query_type = summary.get("query_type", "general")
            query = summary.get("query", "")

            if query_type == "person":
                text_parts.append(f"🧑 Person Discovery: '{query}'")
            elif query_type == "pet":
                text_parts.append(f"🐾 Pet Discovery: '{query}'")
            elif query_type == "area":
                text_parts.append(f"🏠 Area Discovery: '{query}'")
            elif query_type == "aggregate":
                text_parts.append("📊 Aggregate Discovery")
            else:
                text_parts.append("🔍 Discovery Results")

            text_parts.append(
                self._build_paging_header(
                    noun="entities",
                    total_found=summary.get("total_found", 0),
                    returned_count=summary.get(
                        "returned_count", len(actual_entities)
                    ),
                    offset=summary.get("offset", 0),
                    next_offset=summary.get("next_offset"),
                )
            )

            # Group entities by relationship
            primary = [e for e in actual_entities if e.get("relationship") == "primary"]
            related = [e for e in actual_entities if e.get("relationship") != "primary"]

            # Add primary entities
            if primary:
                text_parts.append("\n📍 Primary Entities:")
                for entity in primary:
                    type_desc = (
                        f" ({entity.get('type', '')})" if entity.get("type") else ""
                    )
                    location = []
                    if entity.get("area"):
                        location.append(entity["area"])
                    if entity.get("floor"):
                        location.append(entity["floor"])
                    location_text = f" @ {' / '.join(location)}" if location else ""
                    labels = (
                        f" [Labels: {', '.join(entity['labels'])}]"
                        if entity.get("labels")
                        else ""
                    )
                    aliases = (
                        f" [Aliases: {', '.join(entity['aliases'])}]"
                        if entity.get("aliases")
                        else ""
                    )
                    text_parts.append(
                        f"  • {entity['entity_id']}: {entity['name']} - {entity['state']}{type_desc}{location_text}{labels}{aliases}"
                    )

            # Group related entities by category
            if related:
                categories = {}
                for entity in related:
                    cat = entity.get("relationship", "other")
                    categories.setdefault(cat, []).append(entity)

                text_parts.append("\n🔗 Related Entities:")
                for category, cat_entities in categories.items():
                    # Format category name
                    cat_name = category.replace("_", " ").title()
                    text_parts.append(f"\n  {cat_name}:")
                    for entity in cat_entities:
                        location = []
                        if entity.get("area"):
                            location.append(entity["area"])
                        if entity.get("floor"):
                            location.append(entity["floor"])
                        location_text = f" @ {' / '.join(location)}" if location else ""
                        labels = (
                            f" [Labels: {', '.join(entity['labels'])}]"
                            if entity.get("labels")
                            else ""
                        )
                        aliases = (
                            f" [Aliases: {', '.join(entity['aliases'])}]"
                            if entity.get("aliases")
                            else ""
                        )
                        text_parts.append(
                            f"    • {entity['entity_id']}: {entity['state']}{location_text}{labels}{aliases}"
                        )

            return {
                "content": [{"type": "text", "text": "\n".join(text_parts)}],
                "entities": actual_entities,
                "pagination": pagination,
            }
        else:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": self._format_general_discovery_results(
                            entities,
                            pagination=pagination,
                        ),
                    }
                ],
                "entities": entities,
                "pagination": pagination,
            }

    def _format_general_discovery_results(
        self,
        entities: List[Dict[str, Any]],
        *,
        pagination: Dict[str, Any] | None = None,
    ) -> str:
        """Format general discovery results in a stable, readable order."""
        sorted_entities = sorted(entities, key=self._discovery_entity_sort_key)
        grouped_entities = self._group_entities_for_display(sorted_entities)
        pagination = pagination or {
            "total_found": len(sorted_entities),
            "returned_count": len(sorted_entities),
            "offset": 0,
            "next_offset": None,
        }
        header = self._build_paging_header(
            noun="entities",
            total_found=pagination["total_found"],
            returned_count=pagination["returned_count"],
            offset=pagination["offset"],
            next_offset=pagination.get("next_offset"),
        )

        if len(grouped_entities) > 1:
            text_parts = [f"{header} across {len(grouped_entities)} groups:"]
            for group_name, group_items in grouped_entities:
                text_parts.append(f"\n{group_name} ({len(group_items)}):")
                for entity in group_items:
                    text_parts.append(
                        f"- {self._format_general_discovery_entity(entity)}"
                    )
            return "\n".join(text_parts)

        text_parts = [f"{header}:"]
        for entity in sorted_entities:
            text_parts.append(f"- {self._format_general_discovery_entity(entity)}")
        return "\n".join(text_parts)

    def _discovery_entity_sort_key(
        self, entity: Dict[str, Any]
    ) -> Tuple[int, str, str, str]:
        """Return a stable display sort key for discovered entities."""
        area = str(entity.get("area") or "").strip().casefold()
        floor = str(entity.get("floor") or "").strip().casefold()
        name = str(
            entity.get("name")
            or entity.get("attributes", {}).get("friendly_name")
            or entity.get("entity_id")
            or ""
        ).strip().casefold()
        entity_id = str(entity.get("entity_id") or "").strip().casefold()
        ungrouped = 1 if not area and not floor else 0
        return (ungrouped, area or floor, name, entity_id)

    def _group_entities_for_display(
        self, entities: List[Dict[str, Any]]
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """Group entities by room when available for more natural summaries."""
        floors = {
            str(entity.get("floor") or "").strip()
            for entity in entities
            if str(entity.get("floor") or "").strip()
        }
        show_floor_context = len(floors) > 1
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for entity in entities:
            group_name = self._discovery_group_name(
                entity, show_floor_context=show_floor_context
            )
            groups[group_name].append(entity)

        return sorted(
            groups.items(),
            key=lambda item: self._discovery_group_sort_key(item[0]),
        )

    def _discovery_group_name(
        self, entity: Dict[str, Any], *, show_floor_context: bool
    ) -> str:
        """Build a display label for a discovery group."""
        area = str(entity.get("area") or "").strip()
        floor = str(entity.get("floor") or "").strip()

        if area:
            if floor and show_floor_context:
                return f"{area} ({floor})"
            return area

        if floor:
            return f"No area ({floor})"

        return "No area"

    def _discovery_group_sort_key(self, group_name: str) -> Tuple[int, str]:
        """Sort named groups alphabetically and keep no-area buckets last."""
        normalized = group_name.casefold()
        is_no_area = 1 if normalized.startswith("no area") else 0
        return (is_no_area, normalized)

    def _format_general_discovery_entity(self, entity: Dict[str, Any]) -> str:
        """Format a single discovery result line."""
        detail_parts = [f"State: {entity['state']}"]
        if entity.get("device"):
            detail_parts.append(f"Device: {entity['device']}")
        if entity.get("floor") and not entity.get("area"):
            detail_parts.append(f"Floor: {entity['floor']}")
        if entity.get("attributes", {}).get("device_class"):
            detail_parts.append(
                f"Device class: {entity['attributes']['device_class']}"
            )
        if entity.get("match_reasons"):
            detail_parts.append(
                f"Matched on: {', '.join(entity['match_reasons'])}"
            )
        if entity.get("aliases"):
            detail_parts.append(f"Aliases: {', '.join(entity['aliases'])}")
        if entity.get("labels"):
            detail_parts.append(f"Labels: {', '.join(entity['labels'])}")
        if entity.get("forecast_service_supported"):
            forecast_types = entity.get("forecast_types") or []
            if forecast_types:
                detail_parts.append(
                    f"Forecast service: {', '.join(forecast_types)}"
                )
            else:
                detail_parts.append("Forecast service: supported")
        if entity.get("forecast_available"):
            detail_parts.append(
                f"Forecast available: {entity.get('forecast_entries', 0)} entries"
            )
        elif entity.get("attribute_keys"):
            preview_keys = entity["attribute_keys"][:6]
            detail_parts.append(
                "Extra attrs via get_entity_details: "
                + ", ".join(preview_keys)
                + (
                    "..."
                    if len(entity["attribute_keys"]) > len(preview_keys)
                    else ""
                )
            )

        display_name = entity.get("name") or entity.get("entity_id")
        return f"{display_name} ({entity['entity_id']}): {', '.join(detail_parts)}"

    async def tool_get_entity_details(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed information about specific entities."""
        entity_ids = args.get("entity_ids", [])
        details = await self.discovery.get_entity_details(entity_ids)

        return {"content": [{"type": "text", "text": json.dumps(details, indent=2)}]}

    async def tool_get_device_details(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed information about specific devices."""
        device_ids = args.get("device_ids", [])
        details = await self.discovery.get_device_details(device_ids)

        return {"content": [{"type": "text", "text": json.dumps(details, indent=2)}]}

    async def tool_list_music_assistant_players(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """List Music Assistant players only."""
        if not self._music_assistant_support_enabled():
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Music Assistant support is disabled in shared MCP settings.",
                    }
                ]
            }

        self.publish_progress(
            "tool_start",
            "Listing Music Assistant players",
            tool="list_music_assistant_players",
            args=args,
        )

        try:
            players = await self._discover_music_assistant_players(
                area=args.get("area"),
                floor=args.get("floor"),
                label=args.get("label"),
                name_contains=args.get("name_contains"),
                limit=self._coerce_int_arg(
                    args.get("limit"),
                    default=20,
                    minimum=1,
                    maximum=MAX_ENTITIES_PER_DISCOVERY,
                ),
            )
        except ValueError as err:
            return {"content": [{"type": "text", "text": f"❌ Error: {err}"}]}

        self.publish_progress(
            "tool_complete",
            f"Music Assistant player listing complete: found {len(players)} players",
            tool="list_music_assistant_players",
            count=len(players),
        )

        if not players:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "No exposed Music Assistant players matched that query.",
                    }
                ]
            }

        payload = {"count": len(players), "players": players}
        header = f"Found {len(players)} Music Assistant player{'s' if len(players) != 1 else ''}."
        return {
            "content": [
                {
                    "type": "text",
                    "text": header + "\n\n" + json.dumps(payload, indent=2, ensure_ascii=False),
                }
            ]
        }

    async def tool_play_music_assistant(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Play media using the Music Assistant integration."""
        if not self._music_assistant_support_enabled():
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Music Assistant support is disabled in shared MCP settings.",
                    }
                ]
            }

        if not self.hass.services.has_service("music_assistant", "play_media"):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "The Home Assistant Music Assistant integration is not available or does not expose music_assistant.play_media.",
                    }
                ]
            }

        media_type = str(args.get("media_type") or "").strip().lower()
        if media_type not in {"track", "album", "artist", "playlist", "radio"}:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: media_type must be one of track, album, artist, playlist, or radio.",
                    }
                ]
            }

        normalized_media_id = self._normalize_music_assistant_media_id(args.get("media_id"))
        if normalized_media_id in (None, "", []):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: media_id is required.",
                    }
                ]
            }

        try:
            resolved_player_ids, resolution_text = await self._resolve_music_assistant_player_targets(
                area=args.get("area"),
                floor=args.get("floor"),
                label=args.get("label"),
                media_player=args.get("media_player"),
            )
        except ValueError as err:
            return {"content": [{"type": "text", "text": f"❌ Error: {err}"}]}

        service_data: Dict[str, Any] = {
            "media_id": normalized_media_id,
            "media_type": media_type,
        }
        for key in ("artist", "album", "enqueue"):
            value = args.get(key)
            if isinstance(value, str):
                value = value.strip()
            if value not in (None, ""):
                service_data[key] = value

        radio_mode = args.get("radio_mode")
        if isinstance(radio_mode, bool):
            service_data["radio_mode"] = radio_mode

        shuffle = args.get("shuffle")
        shuffle_requested = isinstance(shuffle, bool)

        media_description = str(args.get("media_description") or "").strip()

        self.publish_progress(
            "tool_start",
            "Starting Music Assistant playback",
            tool="play_music_assistant",
            media_type=media_type,
            target_count=len(resolved_player_ids),
        )

        try:
            await self.hass.services.async_call(
                domain="music_assistant",
                service="play_media",
                service_data={**service_data, "entity_id": resolved_player_ids},
                blocking=True,
                return_response=False,
            )

            if shuffle_requested and self.hass.services.has_service("media_player", "shuffle_set"):
                await self.hass.services.async_call(
                    domain="media_player",
                    service="shuffle_set",
                    service_data={
                        "shuffle": shuffle,
                        "entity_id": resolved_player_ids,
                    },
                    blocking=True,
                    return_response=False,
                )
        except Exception as err:
            error_msg = f"Music Assistant playback failed: {err}"
            _LOGGER.exception(error_msg)
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}

        self.publish_progress(
            "tool_complete",
            "Music Assistant playback started",
            tool="play_music_assistant",
            success=True,
            target_count=len(resolved_player_ids),
        )

        player_names = self._friendly_names_for_entities(resolved_player_ids)
        target_text = ", ".join(player_names)
        description_text = media_description or (
            ", ".join(normalized_media_id)
            if isinstance(normalized_media_id, list)
            else str(normalized_media_id)
        )

        text_parts = [
            f"✅ Started Music Assistant playback for {description_text} on {target_text}.",
            f"Media type: {media_type}",
        ]
        if resolution_text:
            text_parts.append(resolution_text)
        if shuffle_requested:
            text_parts.append(f"Shuffle set to {'on' if shuffle else 'off'}.")

        return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

    async def tool_list_music_assistant_instances(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """List configured Music Assistant instances."""
        del args

        if not self._music_assistant_support_enabled():
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Music Assistant support is disabled in shared MCP settings.",
                    }
                ]
            }

        instances = self._get_music_assistant_instances()
        if not instances:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "No Music Assistant config entries are currently configured in Home Assistant.",
                    }
                ]
            }

        players = [
            dict(record["entity_info"])
            for record in self._get_music_assistant_player_catalog()
        ]
        player_counts: Dict[str, int] = {}
        for player in players:
            config_entry_id = player.get("config_entry_id")
            if config_entry_id:
                player_counts[config_entry_id] = player_counts.get(config_entry_id, 0) + 1

        payload = {
            "count": len(instances),
            "instances": [
                {
                    "config_entry_id": entry.entry_id,
                    "title": entry.title,
                    "player_count": player_counts.get(entry.entry_id, 0),
                }
                for entry in instances
            ],
        }
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, indent=2, ensure_ascii=False),
                }
            ]
        }

    async def tool_search_music_assistant(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Search Music Assistant content."""
        if not self._music_assistant_support_enabled():
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Music Assistant support is disabled in shared MCP settings.",
                    }
                ]
            }

        name = str(args.get("name") or "").strip()
        if not name:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: name is required for search_music_assistant.",
                    }
                ]
            }

        try:
            config_entry = self._resolve_music_assistant_instance(
                config_entry_id=args.get("config_entry_id"),
                instance=args.get("instance"),
            )
        except ValueError as err:
            return {"content": [{"type": "text", "text": f"❌ Error: {err}"}]}

        service_data: Dict[str, Any] = {
            "config_entry_id": config_entry.entry_id,
            "name": name,
            "limit": self._coerce_int_arg(
                args.get("limit"), default=10, minimum=1, maximum=50
            ),
        }
        for key in ("artist", "album"):
            value = args.get(key)
            if isinstance(value, str):
                value = value.strip()
            if value not in (None, ""):
                service_data[key] = value
        media_type = args.get("media_type")
        normalized_media_type = self._normalize_music_assistant_media_type_filter(media_type)
        if media_type is not None and normalized_media_type is None:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: media_type must be track, album, artist, playlist, radio, or a list of those values.",
                    }
                ]
            }
        if normalized_media_type not in (None, [], ""):
            service_data["media_type"] = normalized_media_type
        if isinstance(args.get("library_only"), bool):
            service_data["library_only"] = args["library_only"]

        return await self._call_music_assistant_response_service(
            service="search",
            service_data=service_data,
            summary_label=f"Music Assistant search for {name}",
        )

    async def tool_get_music_assistant_library(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Browse the Music Assistant library."""
        if not self._music_assistant_support_enabled():
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Music Assistant support is disabled in shared MCP settings.",
                    }
                ]
            }

        media_type = str(args.get("media_type") or "").strip().lower()
        if media_type not in {"track", "album", "artist", "playlist", "radio"}:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: media_type must be one of track, album, artist, playlist, or radio.",
                    }
                ]
            }

        try:
            config_entry = self._resolve_music_assistant_instance(
                config_entry_id=args.get("config_entry_id"),
                instance=args.get("instance"),
            )
        except ValueError as err:
            return {"content": [{"type": "text", "text": f"❌ Error: {err}"}]}

        service_data: Dict[str, Any] = {
            "config_entry_id": config_entry.entry_id,
            "media_type": media_type,
            "limit": self._coerce_int_arg(
                args.get("limit"), default=25, minimum=1, maximum=100
            ),
            "offset": self._coerce_int_arg(
                args.get("offset"), default=0, minimum=0, maximum=10000
            ),
        }
        for key in ("search", "order_by", "album_type"):
            value = args.get(key)
            if isinstance(value, str):
                value = value.strip()
            if value not in (None, ""):
                service_data[key] = value
        for key in ("favorite", "album_artists_only"):
            if isinstance(args.get(key), bool):
                service_data[key] = args[key]

        return await self._call_music_assistant_response_service(
            service="get_library",
            service_data=service_data,
            summary_label=f"Music Assistant {media_type} library",
        )

    async def tool_get_music_assistant_queue(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Read Music Assistant queue details for target players."""
        if not self._music_assistant_support_enabled():
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Music Assistant support is disabled in shared MCP settings.",
                    }
                ]
            }

        if not self.hass.services.has_service("music_assistant", "get_queue"):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "The Home Assistant Music Assistant integration is not available or does not expose music_assistant.get_queue.",
                    }
                ]
            }

        try:
            resolved_player_ids, resolution_text = await self._resolve_music_assistant_player_targets(
                area=args.get("area"),
                floor=args.get("floor"),
                label=args.get("label"),
                media_player=args.get("media_player"),
            )
        except ValueError as err:
            return {"content": [{"type": "text", "text": f"❌ Error: {err}"}]}

        result = await self._call_music_assistant_response_service(
            service="get_queue",
            service_data={"entity_id": resolved_player_ids},
            summary_label="Music Assistant queue",
        )

        if resolution_text and result.get("content"):
            result["content"][0]["text"] += f"\n\n{resolution_text}"

        return result

    async def tool_list_areas(self) -> Dict[str, Any]:
        """List all areas."""
        areas = await self.discovery.list_areas()

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Available areas ({len(areas)}):\n"
                    + "\n".join(
                        [
                            (
                                f"- {area['name']}"
                                + (
                                    f" (Aliases: {', '.join(area['aliases'])})"
                                    if area.get("aliases")
                                    else ""
                                )
                                + (
                                    f" (Floor: {area['floor']}"
                                    + (
                                        f"; Floor aliases: {', '.join(area['floor_aliases'])}"
                                        if area.get("floor_aliases")
                                        else ""
                                    )
                                    + ")"
                                    if area.get("floor")
                                    else ""
                                )
                                + (
                                    f" [Labels: {', '.join(area['labels'])}]"
                                    if area.get("labels")
                                    else ""
                                )
                                + f": {area['entity_count']} entities, {area.get('device_count', 0)} devices"
                            )
                            for area in areas
                        ]
                    ),
                }
            ]
        }

    async def tool_list_domains(self) -> Dict[str, Any]:
        """List all domains with entity counts and support status."""
        # Get domains that have entities in this HA instance
        entity_domains = [
            domain_info
            for domain_info in await self.discovery.list_domains()
            if not self._get_domain_capability_error(domain_info["domain"])
        ]
        entity_domain_map = {d["domain"]: d["count"] for d in entity_domains}

        # Get all supported domains from registry
        supported_domains = [
            domain
            for domain in get_supported_domains()
            if not self._get_domain_capability_error(domain)
        ]
        controllable_domains = {
            domain
            for domain in get_domains_by_type(TYPE_CONTROLLABLE)
            if not self._get_domain_capability_error(domain)
        }
        read_only_domains = {
            domain
            for domain in get_domains_by_type(TYPE_READ_ONLY)
            if not self._get_domain_capability_error(domain)
        }

        # Build comprehensive list
        result_text = f"Home Assistant Domains (Entities: {len(entity_domains)}, Supported: {len(supported_domains)}):\n\n"

        # Show domains with entities
        result_text += "📊 Domains with entities in your system:\n"
        for domain in entity_domains:
            support_status = "✅" if domain["domain"] in supported_domains else "⚠️"
            result_text += (
                f"  {support_status} {domain['domain']}: {domain['count']} entities\n"
            )

        # Show supported domains without entities
        result_text += "\n🔧 Additional supported domains (no entities found):\n"
        for domain in supported_domains:
            if domain not in entity_domain_map:
                domain_type = (
                    "controllable"
                    if domain in controllable_domains
                    else "read-only"
                    if domain in read_only_domains
                    else "service"
                )
                result_text += f"  ✅ {domain} ({domain_type})\n"

        result_text += "\n📈 Summary:\n"
        result_text += f"  - Total entity domains: {len(entity_domains)}\n"
        result_text += f"  - Supported domains: {len(supported_domains)}\n"
        result_text += f"  - Controllable: {len(controllable_domains)}\n"
        result_text += f"  - Read-only: {len(read_only_domains)}\n"

        return {"content": [{"type": "text", "text": result_text}]}

    async def tool_get_index(self) -> Dict[str, Any]:
        """Get the pre-generated system structure index."""
        # Get index manager from hass.data
        index_manager = self.hass.data.get(DOMAIN, {}).get("index_manager")

        if not index_manager:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Index manager not available. This feature requires MCP Assist 0.5.0 or later.",
                    }
                ]
            }

        # Get the index
        index = await index_manager.get_index()

        # Format as JSON for structured consumption
        return {"content": [{"type": "text", "text": json.dumps(index, indent=2)}]}

    async def tool_list_assist_tools(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List the native Home Assistant Assist tool surface."""
        del args

        llm_api = await self._get_assist_api_instance()
        tools_payload = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": self._format_assist_tool_input_schema(
                    tool, llm_api.custom_serializer
                ),
            }
            for tool in llm_api.tools
        ]
        payload = {
            "api_id": llm_api.api.id,
            "api_name": llm_api.api.name,
            "tool_count": len(tools_payload),
            "has_live_context_tool": self._assist_api_has_live_context_tool(llm_api),
            "tools": tools_payload,
        }
        header = (
            f"Found {len(tools_payload)} native Home Assistant Assist tools "
            f"from the {llm_api.api.name} API."
        )
        return {
            "content": [
                {
                    "type": "text",
                    "text": header + "\n\n" + json.dumps(payload, indent=2, ensure_ascii=False),
                }
            ]
        }

    async def tool_call_assist_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a native Home Assistant Assist tool directly."""
        tool_name = str(args.get("tool_name") or "").strip()
        if not tool_name:
            raise ValueError("tool_name is required")

        assist_arguments = args.get("arguments") or {}
        if not isinstance(assist_arguments, dict):
            raise ValueError("arguments must be an object")

        llm_api = await self._get_assist_api_instance()
        tool_response = await self._call_assist_api_tool(
            llm_api, tool_name, assist_arguments
        )
        serialized_response = self._serialize_service_response_value(tool_response)

        text_parts = [f"✅ Called native Assist tool `{tool_name}`."]
        summary_lines = self._build_assist_tool_response_summary(serialized_response)
        if summary_lines:
            text_parts.append("")
            text_parts.extend(summary_lines)
        text_parts.append("")
        text_parts.append("Response:")
        text_parts.append(json.dumps(serialized_response, indent=2, ensure_ascii=False))

        return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

    async def tool_get_assist_prompt(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get the native Home Assistant Assist prompt text."""
        del args

        llm_api = await self._get_assist_api_instance()
        description = f"Default prompt for Home Assistant {llm_api.api.name} API"
        text = f"{description}\n\n{llm_api.api_prompt}"
        return {"content": [{"type": "text", "text": text}]}

    async def tool_get_assist_context_snapshot(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get the native Home Assistant Assist live context snapshot."""
        del args

        llm_api = await self._get_assist_api_instance()
        if not self._assist_api_has_live_context_tool(llm_api):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The native Assist API does not currently expose the "
                            "`GetLiveContext` tool, so no Assist context snapshot is "
                            "available right now."
                        ),
                    }
                ]
            }

        tool_response = await self._call_assist_api_tool(
            llm_api, "GetLiveContext", {}
        )
        if (
            isinstance(tool_response, dict)
            and tool_response.get("success") is False
            and tool_response.get("error")
        ):
            raise HomeAssistantError(str(tool_response["error"]))
        snapshot = tool_response.get("result") if isinstance(tool_response, dict) else None
        if snapshot is None:
            snapshot = self._serialize_service_response_value(tool_response)
        if not isinstance(snapshot, str):
            snapshot = json.dumps(snapshot, indent=2, ensure_ascii=False)

        return {
            "content": [
                {
                    "type": "text",
                    "text": "Assist context snapshot:\n\n" + snapshot,
                }
            ]
        }

    async def tool_get_weather_forecast(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get a Home Assistant weather forecast in one call."""
        target_entity_id = str(args.get("entity_id") or "").strip() or None
        area = str(args.get("area") or "").strip() or None
        floor = str(args.get("floor") or "").strip() or None
        label = str(args.get("label") or "").strip() or None
        name_contains = str(args.get("name_contains") or "").strip() or None
        when_value = str(args.get("when") or "tomorrow").strip() or "tomorrow"
        forecast_type = str(args.get("forecast_type") or "").strip().lower() or None
        timeout = self._coerce_int_arg(
            args.get("timeout"), default=60, minimum=1, maximum=300
        )

        target_date, day_label, parse_error = self._parse_weather_forecast_when(
            when_value
        )
        if parse_error:
            return {
                "content": [{"type": "text", "text": f"❌ Error: {parse_error}"}]
            }

        try:
            resolved_target, entity_info = await self._resolve_weather_forecast_target(
                entity_id=target_entity_id,
                area=area,
                floor=floor,
                label=label,
                name_contains=name_contains,
            )
        except ValueError as err:
            return {"content": [{"type": "text", "text": f"❌ Error: {err}"}]}

        prepared_data = self._prepare_response_service_data(
            "weather",
            "get_forecasts",
            {"type": forecast_type} if forecast_type else {},
            resolved_target=resolved_target,
        )

        service_result = await self.tool_call_service_with_response(
            {
                "domain": "weather",
                "service": "get_forecasts",
                "target": resolved_target,
                "data": prepared_data,
                "timeout": timeout,
            }
        )

        if service_result.get("response") is None:
            return service_result

        response = service_result.get("response")
        if not isinstance(response, dict):
            return service_result

        payload = response.get(entity_info["entity_id"])
        forecast_entries = payload.get("forecast") if isinstance(payload, dict) else None
        if not isinstance(forecast_entries, list):
            return service_result

        summary_text = self._summarize_requested_weather_forecast(
            entity_name=entity_info["name"],
            entity_id=entity_info["entity_id"],
            forecast_entries=forecast_entries,
            forecast_type=prepared_data.get("type"),
            target_date=target_date,
            day_label=day_label,
        )
        if not summary_text:
            return service_result

        result: Dict[str, Any] = {
            "content": [{"type": "text", "text": summary_text}],
            "response": response,
        }
        return result

    async def tool_get_calendar_events(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get upcoming calendar events in one call."""
        target_entity_id = str(args.get("entity_id") or "").strip() or None
        area = str(args.get("area") or "").strip() or None
        floor = str(args.get("floor") or "").strip() or None
        label = str(args.get("label") or "").strip() or None
        query = str(args.get("query") or "").strip() or None
        name_contains = str(args.get("name_contains") or "").strip() or None
        event_text = str(args.get("event_text") or "").strip() or None
        when_value = str(args.get("when") or "").strip() or None
        days = self._coerce_int_arg(
            args.get("days"),
            default=60,
            minimum=1,
            maximum=365,
        )
        limit = self._coerce_int_arg(
            args.get("limit"),
            default=5,
            minimum=1,
            maximum=20,
        )
        timeout = self._coerce_int_arg(
            args.get("timeout"), default=60, minimum=1, maximum=300
        )

        try:
            (
                resolved_target,
                selected_calendars,
                fallback_used,
            ) = await self._resolve_calendar_event_targets(
                entity_id=target_entity_id,
                area=area,
                floor=floor,
                label=label,
                query=query,
                name_contains=name_contains,
            )
        except ValueError as err:
            return self._build_text_tool_result(str(err), is_error=True)

        window_start, window_end, window_label = self._build_calendar_search_window(
            when_value=when_value,
            days=days,
        )
        if window_start is None or window_end is None:
            return self._build_text_tool_result(
                f"Invalid calendar time window: {window_label}",
                is_error=True,
            )

        service_result = await self.tool_call_service_with_response(
            {
                "domain": "calendar",
                "service": "get_events",
                "target": resolved_target,
                "data": {
                    "start_date_time": window_start.isoformat(),
                    "end_date_time": window_end.isoformat(),
                },
                "timeout": timeout,
            }
        )

        response = service_result.get("response")
        if not isinstance(response, dict):
            return service_result

        active_event_filter = event_text
        if not active_event_filter and fallback_used and query:
            active_event_filter = query

        matches = self._collect_calendar_event_matches(
            response=response,
            selected_calendars=selected_calendars,
            event_text=active_event_filter,
        )
        serialized_matches = self._serialize_service_response_value(matches[:limit])
        if not matches:
            filter_label = active_event_filter or name_contains or query or "the request"
            calendars_label = (
                f"{len(selected_calendars)} calendar"
                f"{'' if len(selected_calendars) == 1 else 's'}"
            )
            return self._build_text_tool_result(
                (
                    f"No upcoming calendar events matched {filter_label!r} in {window_label}. "
                    f"Searched {calendars_label}."
                ),
                structured_content={
                    "query": query,
                    "name_contains": name_contains,
                    "event_text": active_event_filter,
                    "window": {
                        "start": window_start.isoformat(),
                        "end": window_end.isoformat(),
                        "label": window_label,
                    },
                    "selected_calendars": selected_calendars,
                    "events": [],
                },
            )

        summary_text = self._summarize_calendar_matches(
            matches=matches[:limit],
            query=query,
            event_text=active_event_filter,
            window_label=window_label,
        )
        return self._build_text_tool_result(
            summary_text,
            structured_content={
                "query": query,
                "name_contains": name_contains,
                "event_text": active_event_filter,
                "window": {
                    "start": window_start.isoformat(),
                    "end": window_end.isoformat(),
                    "label": window_label,
                },
                "selected_calendars": selected_calendars,
                "events": serialized_matches,
                "total_matching_events": len(matches),
            },
        )

    async def tool_list_response_services(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List dynamically available HA services that support response data."""
        domain_filter = str(args.get("domain") or "").strip().casefold()
        query = str(args.get("query") or "").strip().casefold()
        limit = self._coerce_int_arg(
            args.get("limit"), default=50, minimum=1, maximum=200
        )

        catalog = await self._get_response_service_catalog()
        rows: List[tuple[str, str, Dict[str, Any]]] = []

        for domain, services in catalog.items():
            if domain_filter and domain.casefold() != domain_filter:
                continue

            for service_name, description in services.items():
                haystacks = [
                    domain,
                    service_name,
                    str(description.get("name") or ""),
                    str(description.get("description") or ""),
                ]
                if query and not any(query in text.casefold() for text in haystacks):
                    continue

                rows.append((domain, service_name, description))

        rows.sort(key=lambda item: (item[0], item[1]))
        rows = rows[:limit]

        if not rows:
            filters = []
            if domain_filter:
                filters.append(f"domain='{domain_filter}'")
            if query:
                filters.append(f"query='{query}'")
            filter_text = f" for {', '.join(filters)}" if filters else ""
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"No response-capable services found{filter_text}.",
                    }
                ]
            }

        text_parts = [
            f"Found {len(rows)} response-capable Home Assistant services:"
        ]
        for domain, service_name, description in rows:
            detail_parts = [
                f"response: {description.get('supports_response', 'optional')}"
            ]
            target_domains = self._extract_service_target_domains(description)
            if target_domains:
                detail_parts.append(f"target domains: {', '.join(target_domains)}")
            required_fields = self._get_required_service_fields(description)
            if required_fields:
                detail_parts.append(f"required: {', '.join(required_fields)}")
            field_names = self._get_service_field_names(description)
            if field_names:
                preview_fields = field_names[:6]
                detail_parts.append(
                    "fields: "
                    + ", ".join(preview_fields)
                    + ("..." if len(field_names) > len(preview_fields) else "")
                )
            if description.get("description"):
                detail_parts.append(str(description["description"]))

            text_parts.append(
                f"- {domain}.{service_name} ({'; '.join(detail_parts)})"
            )

        return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

    async def tool_perform_action(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Perform an action on Home Assistant entities with progress notifications."""
        domain = args.get("domain")
        action = args.get("action")
        target = args.get("target", {})
        data = args.get("data", {})

        # Validate required parameters
        if not domain:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: Missing required parameter 'domain'. Use discover_entities to find the correct domain.",
                    }
                ]
            }

        if not action:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: Missing required parameter 'action'. Common actions: turn_on, turn_off, toggle.",
                    }
                ]
            }

        if not isinstance(target, dict):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: 'target' must be an object with entity_id, area_id, floor_id, label_id, or device_id.",
                    }
                ]
            }

        if not isinstance(data, dict):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: 'data' must be an object of service parameters.",
                    }
                ]
            }

        _LOGGER.info(f"🎯 Performing action: {domain}.{action} on {target}")

        # Notify start
        self.publish_progress(
            "tool_start",
            f"Performing action: {domain}.{action}",
            tool="perform_action",
            domain=domain,
            action=action,
        )

        # Validate the service and get the correct service name
        try:
            service = self.validate_service(domain, action)
        except ValueError as err:
            error_msg = str(err)
            _LOGGER.error(f"Service validation error: {error_msg}")
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}

        # Resolve target (convert areas to entity_ids if needed)
        try:
            resolved_target = await self.resolve_target(target)
            resolved_target = self._restrict_resolved_target_to_domain(
                resolved_target, domain
            )
            _LOGGER.debug(f"Resolved target: {resolved_target}")
        except Exception as err:
            error_msg = f"Failed to resolve target: {err}"
            _LOGGER.error(error_msg)
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}

        # Reject deprecated color_temp parameter
        if domain == "light" and "color_temp" in data:
            _LOGGER.warning(
                f"❌ Rejecting deprecated color_temp parameter: {data.get('color_temp')}"
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "❌ Error: color_temp is deprecated. Use "
                            "color_temp_kelvin instead. Examples: 2700 (warm white), "
                            "4000 (neutral white), 6500 (cool white). Lower Kelvin "
                            "values = warmer light, higher Kelvin values = cooler light."
                        ),
                    }
                ]
            }

        valid_params, validation_msg = validate_service_parameters(
            domain, service, data
        )
        if not valid_params:
            return {"content": [{"type": "text", "text": f"❌ Error: {validation_msg}"}]}

        try:
            # Prepare service data
            service_data = {**resolved_target, **data}

            # Call the Home Assistant service with the validated service name
            await self.hass.services.async_call(
                domain=domain,
                service=service,  # Use the mapped service name
                service_data=service_data,
                blocking=True,  # Wait for completion
                return_response=False,
            )

            # Notify completion
            self.publish_progress(
                "tool_complete",
                f"Action completed: {domain}.{service}",
                tool="perform_action",
                success=True,
            )

            # Check new states if we have entity_ids
            result_text = f"✅ Successfully executed {domain}.{service}"
            if service != action:
                result_text += f" (mapped from '{action}')"

            if "entity_id" in resolved_target:
                entity_ids = resolved_target["entity_id"]
                if isinstance(entity_ids, str):
                    entity_ids = [entity_ids]
                action_observation = await self._observe_action_outcome(
                    domain=domain,
                    service=service,
                    entity_ids=entity_ids,
                    action_data=data,
                )

                if action_observation["status"] == "pending":
                    result_text = f"✅ Sent {domain}.{service}"
                    if service != action:
                        result_text += f" (mapped from '{action}')"
                    result_text += (
                        f"\n\nFinal state is not yet confirmed; the device may still be "
                        f"{action_observation['progress_phrase']}."
                    )
                    if action_observation["state_lines"]:
                        result_text += (
                            "\n\nCurrent states right now:\n"
                            + "\n".join(action_observation["state_lines"])
                        )
                elif action_observation["state_lines"]:
                    heading = (
                        "Confirmed states:"
                        if action_observation["status"] == "confirmed"
                        else "Current states:"
                    )
                    result_text += (
                        f"\n\n{heading}\n" + "\n".join(action_observation["state_lines"])
                    )

            return {"content": [{"type": "text", "text": result_text}]}

        except Exception as err:
            error_msg = f"Service call failed: {err}"
            _LOGGER.exception(error_msg)
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}

    def _get_action_state_expectation(
        self, domain: str, service: str, action_data: Dict[str, Any] | None = None
    ) -> Dict[str, Any] | None:
        """Return final/transitional state expectations for slow mechanical actions."""
        action_data = action_data or {}

        if domain == "lock":
            if service == "lock":
                return {
                    "expected_states": {"locked"},
                    "transitional_states": {"locking"},
                    "progress_phrase": "locking",
                }
            if service == "unlock":
                return {
                    "expected_states": {"unlocked"},
                    "transitional_states": {"unlocking"},
                    "progress_phrase": "unlocking",
                }

        if domain == "cover":
            if service == "close_cover":
                return {
                    "expected_states": {"closed"},
                    "transitional_states": {"closing"},
                    "progress_phrase": "closing",
                }
            if service == "open_cover":
                return {
                    "expected_states": {"open"},
                    "transitional_states": {"opening"},
                    "progress_phrase": "opening",
                }
            if service == "set_cover_position":
                position = action_data.get("position")
                try:
                    position_value = int(position)
                except (TypeError, ValueError):
                    position_value = None

                if position_value is not None:
                    if position_value <= 0:
                        return {
                            "expected_states": {"closed"},
                            "transitional_states": {"closing"},
                            "progress_phrase": "closing",
                        }
                    if position_value >= 100:
                        return {
                            "expected_states": {"open"},
                            "transitional_states": {"opening"},
                            "progress_phrase": "opening",
                        }

        if domain == "valve":
            if service == "close_valve":
                return {
                    "expected_states": {"closed"},
                    "transitional_states": {"closing"},
                    "progress_phrase": "closing",
                }
            if service == "open_valve":
                return {
                    "expected_states": {"open"},
                    "transitional_states": {"opening"},
                    "progress_phrase": "opening",
                }

        return None

    def _format_action_state_lines(self, entity_ids: List[str]) -> List[str]:
        """Format a compact snapshot of current entity states."""
        lines: List[str] = []
        for entity_id in entity_ids[:10]:
            state = self.hass.states.get(entity_id)
            if state is None:
                lines.append(f"  • {entity_id}: unavailable")
                continue

            friendly_name = state.attributes.get("friendly_name")
            if friendly_name and str(friendly_name) != entity_id:
                lines.append(f"  • {friendly_name}: {state.state}")
            else:
                lines.append(f"  • {entity_id}: {state.state}")

        return lines

    async def _observe_action_outcome(
        self,
        *,
        domain: str,
        service: str,
        entity_ids: List[str],
        action_data: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Observe post-action state with transition-aware polling."""
        expectation = self._get_action_state_expectation(domain, service, action_data)
        if expectation is None:
            await asyncio.sleep(0.5)
            return {
                "status": "snapshot",
                "state_lines": self._format_action_state_lines(entity_ids),
                "progress_phrase": "",
            }

        deadline = asyncio.get_running_loop().time() + 3.0
        last_lines: List[str] = []

        while True:
            current_states = []
            for entity_id in entity_ids[:10]:
                state = self.hass.states.get(entity_id)
                current_states.append(state.state if state is not None else "unavailable")

            last_lines = self._format_action_state_lines(entity_ids)
            if current_states and all(
                state in expectation["expected_states"] for state in current_states
            ):
                return {
                    "status": "confirmed",
                    "state_lines": last_lines,
                    "progress_phrase": expectation["progress_phrase"],
                }

            if asyncio.get_running_loop().time() >= deadline:
                return {
                    "status": "pending",
                    "state_lines": last_lines,
                    "progress_phrase": expectation["progress_phrase"],
                }

            await asyncio.sleep(0.5)

    async def tool_call_service_with_response(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a native HA service that returns structured response data."""
        domain = str(args.get("domain") or "").strip().lower()
        service = str(args.get("service") or "").strip().lower()
        target = args.get("target")
        data = args.get("data", {}) or {}
        timeout = self._coerce_int_arg(
            args.get("timeout"), default=60, minimum=1, maximum=300
        )

        if not domain:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: Missing required parameter 'domain'.",
                    }
                ]
            }

        if not service:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: Missing required parameter 'service'.",
                    }
                ]
            }

        if target is None:
            target = {}
        elif not isinstance(target, dict):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: 'target' must be an object with entity_id, area_id, floor_id, label_id, or device_id.",
                    }
                ]
            }

        if not isinstance(data, dict):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: 'data' must be an object of service parameters.",
                    }
                ]
            }

        _LOGGER.info(
            "📖 Calling response service: %s.%s on %s with data %s",
            domain,
            service,
            target,
            data,
        )

        service_description, validation_error = await self._get_response_service_info(
            domain, service
        )
        if validation_error:
            return {
                "content": [
                    {"type": "text", "text": f"❌ Error: {validation_error}"}
                ]
            }

        resolved_target = {}
        try:
            if target:
                resolved_target = await self.resolve_target(target)
                resolved_target = self._restrict_resolved_target_for_service(
                    resolved_target,
                    service_description=service_description,
                )
        except Exception as err:
            error_msg = f"Failed to resolve target: {err}"
            _LOGGER.error(error_msg)
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}

        prepared_data = self._prepare_response_service_data(
            domain,
            service,
            data,
            resolved_target=resolved_target,
        )
        valid_params, validation_msg = self._validate_response_service_parameters(
            service_description, prepared_data
        )
        if not valid_params:
            return {
                "content": [{"type": "text", "text": f"❌ Error: {validation_msg}"}]
            }

        self.publish_progress(
            "tool_start",
            f"Calling response service: {domain}.{service}",
            tool="call_service_with_response",
            domain=domain,
            service=service,
        )

        try:
            service_data = {**resolved_target, **prepared_data}
            response = await asyncio.wait_for(
                self.hass.services.async_call(
                    domain=domain,
                    service=service,
                    service_data=service_data,
                    blocking=True,
                    return_response=True,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            error_msg = (
                f"Service-response call timed out after {timeout} seconds: "
                f"{domain}.{service}"
            )
            _LOGGER.error("❌ %s", error_msg)
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}
        except Exception as err:
            error_msg = f"Service-response call failed: {err}"
            _LOGGER.exception("❌ %s", error_msg)
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}

        self.publish_progress(
            "tool_complete",
            f"Response service completed: {domain}.{service}",
            tool="call_service_with_response",
            success=True,
        )

        serialized_response = self._serialize_service_response_value(response)
        result_text = self._format_service_response_result(
            domain,
            service,
            resolved_target,
            serialized_response,
            request_data=prepared_data,
        )
        result: Dict[str, Any] = {
            "content": [{"type": "text", "text": result_text}]
        }
        if serialized_response is not None:
            result["response"] = serialized_response

        return result

    async def tool_set_conversation_state(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Set whether the assistant expects a user response."""
        expecting_response = args.get("expecting_response", False)

        # Log the state for debugging
        _LOGGER.info(
            f"🔄 Conversation state set: expecting_response={expecting_response}"
        )

        # Return a marker that the agent can detect
        return {
            "content": [
                {"type": "text", "text": f"conversation_state:{expecting_response}"}
            ]
        }

    async def tool_run_script(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a Home Assistant script and return its response variables."""
        script_id = args.get("script_id")
        variables = args.get("variables", {})
        timeout = args.get("timeout", 60)

        # Extract script name (remove script. prefix if present)
        script_name = script_id.replace("script.", "")
        full_script_id = f"script.{script_name}"

        _LOGGER.info(f"📜 Running script: {full_script_id} with variables: {variables}")

        # Notify start
        self.publish_progress(
            "tool_start",
            f"Running script: {full_script_id}",
            tool="run_script",
            script_id=full_script_id,
        )

        try:
            # Call the script directly as a service (not script.turn_on)
            # Variables go directly in service_data, not nested
            response = await asyncio.wait_for(
                self.hass.services.async_call(
                    domain="script",
                    service=script_name,  # Call script directly
                    service_data=variables,  # Variables go directly here
                    blocking=True,
                    return_response=True,
                ),
                timeout=timeout,
            )

            # Notify completion
            self.publish_progress(
                "tool_complete",
                f"Script completed: {full_script_id}",
                tool="run_script",
                success=True,
            )

            # Format the response
            result_text = f"✅ Script {full_script_id} completed successfully"

            # If the script returned response variables, include them
            if response is not None:
                serialized_response = self._serialize_service_response_value(response)
                result_text += (
                    f"\n\nResponse:\n{json.dumps(serialized_response, indent=2)}"
                )
                return {
                    "content": [{"type": "text", "text": result_text}],
                    "response": serialized_response,
                }
            else:
                result_text += "\n\nNo response variables returned (script may not have response_variable defined)"
                return {"content": [{"type": "text", "text": result_text}]}

        except asyncio.TimeoutError:
            error_msg = f"Script execution timed out after {timeout} seconds"
            _LOGGER.error(f"❌ {error_msg}: {full_script_id}")
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}
        except Exception as err:
            error_msg = f"Script execution failed: {err}"
            _LOGGER.exception(f"❌ {error_msg}")
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}

    async def tool_run_automation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger a Home Assistant automation with optional variables."""
        automation_id = args.get("automation_id")
        variables = args.get("variables", {})
        skip_conditions = args.get("skip_conditions", False)

        # Normalize automation_id (add automation. prefix if missing)
        if not automation_id.startswith("automation."):
            automation_id = f"automation.{automation_id}"

        _LOGGER.info(
            f"🤖 Triggering automation: {automation_id} with variables: {variables}, skip_conditions: {skip_conditions}"
        )

        # Notify start
        self.publish_progress(
            "tool_start",
            f"Triggering automation: {automation_id}",
            tool="run_automation",
            automation_id=automation_id,
        )

        try:
            # Trigger the automation
            await self.hass.services.async_call(
                domain="automation",
                service="trigger",
                service_data={
                    "entity_id": automation_id,
                    "variables": variables,
                    "skip_condition": skip_conditions,
                },
                blocking=True,
            )

            # Notify completion
            self.publish_progress(
                "tool_complete",
                f"Automation triggered: {automation_id}",
                tool="run_automation",
                success=True,
            )

            result_text = f"✅ Automation {automation_id} triggered successfully"
            if skip_conditions:
                result_text += " (conditions skipped)"

            return {"content": [{"type": "text", "text": result_text}]}

        except Exception as err:
            error_msg = f"Automation trigger failed: {err}"
            _LOGGER.exception(f"❌ {error_msg}")
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}

    async def tool_get_entity_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get entity history with human-readable formatting."""
        mode = str(args.get("mode", "timeline")).strip().casefold()
        if mode not in {"timeline", "last_event"}:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Invalid mode. Use 'timeline' or 'last_event'.",
                    }
                ]
            }
        if mode == "last_event":
            return await self._tool_get_last_entity_event_impl(
                args,
                tool_name="get_entity_history",
            )

        entity_id = args.get("entity_id")
        hours = self._coerce_int_arg(args.get("hours"), default=24, minimum=1, maximum=8760)
        limit = self._coerce_int_arg(args.get("limit"), default=50, minimum=1, maximum=100)
        target_states = self._normalize_history_targets(
            args.get("state"), args.get("event")
        )

        _LOGGER.info(f"📜 Getting history for {entity_id}: {hours} hours, limit {limit}")

        # Notify start
        self.publish_progress(
            "tool_start",
            f"Retrieving history for {entity_id}",
            tool="get_entity_history",
            entity_id=entity_id,
        )

        # 1. Get current state
        current_state = self.hass.states.get(entity_id)
        if not current_state:
            return {
                "content": [
                    {"type": "text", "text": f"Entity '{entity_id}' not found."}
                ]
            }

        friendly_name = current_state.attributes.get("friendly_name", entity_id)

        # 2. Calculate time range (UTC)
        try:
            entity_states = await self._fetch_entity_history_states(
                entity_id,
                hours=hours,
                descending=True,
                limit=None if target_states else limit,
            )
        except Exception as e:
            _LOGGER.error(f"Failed to get history for {entity_id}: {e}")
            return {
                "content": [
                    {"type": "text", "text": f"Failed to retrieve history: {str(e)}"}
                ]
            }

        if target_states:
            entity_states = [
                state
                for state in entity_states
                if state.state.casefold() in target_states
            ]
            entity_states = entity_states[:limit]

        # Notify completion
        self.publish_progress(
            "tool_complete",
            f"History retrieved: {len(entity_states)} changes",
            tool="get_entity_history",
            success=True,
        )

        # 4. Format history (most recent first, limited)
        if not entity_states:
            if target_states:
                search_label = self._describe_history_target(
                    args.get("state"), args.get("event")
                )
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{friendly_name} ({entity_id})\n"
                                f"Current state: {current_state.state}\n\n"
                                f"No recorded {search_label} entries were found in the last {hours} hours."
                            ),
                        }
                    ]
                }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"{friendly_name} ({entity_id})\nCurrent state: {current_state.state}\n\nNo history available for the last {hours} hours.",
                    }
                ]
            }

        # Build formatted text
        text_parts = [
            f"{friendly_name} ({entity_id})",
            f"Current state: {current_state.state}",
            "",
            (
                f"Matching history for {self._describe_history_target(args.get('state'), args.get('event'))} "
                f"(last {hours} hours):"
                if target_states
                else f"Recent history (last {hours} hours):"
            ),
        ]

        for state in entity_states:
            when = state.last_changed or state.last_updated
            text_parts.append(
                f"• {self._format_relative_absolute_time(when)} → {state.state}"
            )

        text_parts.append("")
        text_parts.append(
            (
                f"Showing {len(entity_states)} matching entr{'ies' if len(entity_states) != 1 else 'y'}"
                if target_states
                else f"Showing {len(entity_states)} change{'s' if len(entity_states) != 1 else ''}"
            )
        )

        return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

    async def tool_get_last_entity_event(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Find the latest recorder event for an entity."""
        return await self._tool_get_last_entity_event_impl(
            args,
            tool_name="get_last_entity_event",
        )

    async def _tool_get_last_entity_event_impl(
        self, args: Dict[str, Any], *, tool_name: str
    ) -> Dict[str, Any]:
        """Shared implementation for latest-event recorder lookups."""
        entity_id = args.get("entity_id")
        hours = self._coerce_int_arg(args.get("hours"), default=720, minimum=1, maximum=8760)
        current_state = self.hass.states.get(entity_id)

        if not current_state:
            return {
                "content": [
                    {"type": "text", "text": f"Entity '{entity_id}' not found."}
                ]
            }

        target_states = self._normalize_history_targets(
            args.get("state"), args.get("event")
        )
        history_candidates = self._history_resolution_candidates(
            entity_id,
            args.get("state"),
            args.get("event"),
        )
        selected_candidate = history_candidates[0]
        history_entity_id = selected_candidate["entity_id"]
        resolution_note = selected_candidate["note"]
        current_state = self.hass.states.get(history_entity_id)
        friendly_name = current_state.attributes.get("friendly_name", history_entity_id)
        end_time = dt_util.utcnow()

        self.publish_progress(
            "tool_start",
            f"Searching recorder history for {history_entity_id}",
            tool=tool_name,
            entity_id=history_entity_id,
        )

        try:
            if target_states:
                matched_state = None
                for candidate in history_candidates:
                    candidate_entity_id = candidate["entity_id"]
                    candidate_target_states = target_states
                    for window_hours in self._build_history_search_windows(hours):
                        entity_states = await self._fetch_entity_history_states(
                            candidate_entity_id,
                            hours=window_hours,
                            end_time=end_time,
                            descending=True,
                        )
                        candidate_target_states = self._choose_history_count_states(
                            entity_states,
                            args.get("state"),
                            args.get("event"),
                        )
                        matched_state = next(
                            (
                                state
                                for state in entity_states
                                if state.state.casefold() in candidate_target_states
                            ),
                            None,
                        )
                        if matched_state is not None:
                            history_entity_id = candidate_entity_id
                            resolution_note = candidate["note"]
                            current_state = self.hass.states.get(history_entity_id)
                            friendly_name = current_state.attributes.get(
                                "friendly_name", history_entity_id
                            )
                            target_states = candidate_target_states
                            break
                    if matched_state is not None:
                        break
            else:
                entity_states = await self._fetch_entity_history_states(
                    history_entity_id,
                    hours=hours,
                    end_time=end_time,
                    descending=True,
                    limit=1,
                )
                matched_state = entity_states[0] if entity_states else None
        except Exception as err:
            _LOGGER.error(
                "Failed to get last recorder event for %s: %s", history_entity_id, err
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Failed to retrieve recorder history: {err}",
                    }
                ]
            }

        self.publish_progress(
            "tool_complete",
            "Recorder history search complete",
            tool=tool_name,
            success=True,
            found=matched_state is not None,
        )

        if not matched_state:
            search_label = self._describe_history_target(
                args.get("state"), args.get("event")
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{friendly_name} ({history_entity_id})\n"
                            f"Current state: {current_state.state}\n\n"
                            f"No recorded {search_label} event was found in the last {hours} hours."
                            + (
                                f"\n\n{resolution_note}"
                                if resolution_note
                                else ""
                            )
                        ),
                    }
                ]
            }

        when = matched_state.last_changed or matched_state.last_updated
        search_label = self._describe_history_target(
            args.get("state"), args.get("event")
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        (
                            f"{resolution_note}\n\n"
                            if resolution_note
                            else ""
                        )
                        + f"{friendly_name} ({history_entity_id})\n"
                        f"Current state: {current_state.state}\n"
                        f"Last recorded {search_label}: {self._format_relative_absolute_time(when)}\n"
                        f"Matched state: {matched_state.state}"
                    ),
                }
            ]
        }

    async def tool_analyze_entity_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze recorder history for counts and summaries."""
        analysis = str(args.get("analysis", "count")).strip().casefold()
        if analysis not in {"count", "summary", "duration", "stats", "streak"}:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Invalid analysis. Use 'count', 'summary', 'duration', 'streak', or 'stats'.",
                    }
                ]
            }
        entity_id = args.get("entity_id")
        default_hours = 720 if analysis == "streak" else 24
        hours = self._coerce_int_arg(
            args.get("hours"), default=default_hours, minimum=1, maximum=8760
        )
        history_candidates = self._history_resolution_candidates(
            entity_id,
            args.get("state"),
            args.get("event"),
        )
        selected_candidate = history_candidates[0]
        history_entity_id = selected_candidate["entity_id"]
        resolution_note = selected_candidate["note"]
        current_state = self.hass.states.get(history_entity_id)

        if not current_state:
            return {
                "content": [
                    {"type": "text", "text": f"Entity '{history_entity_id}' not found."}
                ]
            }

        friendly_name = current_state.attributes.get("friendly_name", history_entity_id)
        target_states = self._normalize_history_targets(
            args.get("state"), args.get("event")
        )

        self.publish_progress(
            "tool_start",
            f"Analyzing recorder history for {history_entity_id}",
            tool="analyze_entity_history",
            entity_id=history_entity_id,
        )

        query_end_time = dt_util.utcnow()
        query_start_time = query_end_time - timedelta(hours=hours)
        include_start_time_state = analysis in {"duration", "streak", "stats"}

        async def _load_candidate_history(candidate: Dict[str, Any]) -> Dict[str, Any] | None:
            candidate_entity_id = candidate["entity_id"]
            candidate_current_state = self.hass.states.get(candidate_entity_id)
            if candidate_current_state is None:
                return None

            candidate_entity_states = await self._fetch_entity_history_states(
                candidate_entity_id,
                hours=hours,
                end_time=query_end_time,
                descending=False,
                include_start_time_state=include_start_time_state,
            )
            if target_states:
                candidate_target_filter_states = self._choose_history_count_states(
                    candidate_entity_states,
                    args.get("state"),
                    args.get("event"),
                )
            else:
                candidate_target_filter_states = []

            return {
                "candidate": candidate,
                "entity_id": candidate_entity_id,
                "current_state": candidate_current_state,
                "friendly_name": candidate_current_state.attributes.get(
                    "friendly_name", candidate_entity_id
                ),
                "entity_states": candidate_entity_states,
                "target_filter_states": candidate_target_filter_states,
            }

        try:
            candidate_history = await _load_candidate_history(selected_candidate)
        except Exception as err:
            _LOGGER.error(
                "Failed to analyze history for %s: %s", history_entity_id, err
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Failed to analyze recorder history: {err}",
                    }
                ]
            }

        if candidate_history is None:
            return {
                "content": [
                    {"type": "text", "text": f"Entity '{history_entity_id}' not found."}
                ]
            }

        history_entity_id = candidate_history["entity_id"]
        current_state = candidate_history["current_state"]
        friendly_name = candidate_history["friendly_name"]
        entity_states = candidate_history["entity_states"]

        target_filter_states = candidate_history["target_filter_states"]

        if analysis == "duration":
            if not target_filter_states:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{friendly_name} ({history_entity_id})\n"
                                "Duration analysis requires a target event or state, such as opened, closed, on, or off."
                            ),
                        }
                    ]
                }

            duration_result = self._calculate_state_duration(
                entity_states,
                target_filter_states,
                query_start_time,
                query_end_time,
            )
            if duration_result["interval_count"] == 0 and len(history_candidates) > 1:
                for candidate in history_candidates[1:]:
                    candidate_history = await _load_candidate_history(candidate)
                    if candidate_history is None:
                        continue
                    candidate_duration_result = self._calculate_state_duration(
                        candidate_history["entity_states"],
                        candidate_history["target_filter_states"],
                        query_start_time,
                        query_end_time,
                    )
                    candidate_current_state = candidate_history["current_state"]
                    if (
                        candidate_duration_result["interval_count"] > 0
                        or (
                            candidate_history["target_filter_states"]
                            and candidate_current_state.state.casefold()
                            in candidate_history["target_filter_states"]
                        )
                    ):
                        history_entity_id = candidate_history["entity_id"]
                        current_state = candidate_current_state
                        friendly_name = candidate_history["friendly_name"]
                        entity_states = candidate_history["entity_states"]
                        target_filter_states = candidate_history["target_filter_states"]
                        resolution_note = candidate["note"]
                        duration_result = candidate_duration_result
                        break

            total_duration = duration_result["total_duration"]
            interval_count = duration_result["interval_count"]
            search_label = self._describe_history_target(
                args.get("state"), args.get("event")
            )

            self.publish_progress(
                "tool_complete",
                "Recorder history duration analysis complete",
                tool="analyze_entity_history",
                success=True,
                seconds=int(total_duration.total_seconds()),
            )

            text_parts = [
                f"{friendly_name} ({history_entity_id})",
                f"Current state: {current_state.state}",
                f"Total time in {search_label} state during the last {hours} hour{'s' if hours != 1 else ''}: {self._format_duration(total_duration)}",
                f"Matching interval{'s' if interval_count != 1 else ''}: {interval_count}",
            ]
            text_parts = self._prepend_resolution_note(text_parts, resolution_note)

            if interval_count:
                first_start = duration_result.get("first_start")
                last_end = duration_result.get("last_end")
                if first_start is not None:
                    text_parts.append(
                        "First matching interval in window started: "
                        f"{self._format_relative_absolute_time(first_start)}"
                    )
                if last_end is not None and last_end < query_end_time:
                    text_parts.append(
                        f"Last matching interval ended: {self._format_relative_absolute_time(last_end)}"
                    )

            if current_state.state.casefold() in target_filter_states:
                streak_start = current_state.last_changed or current_state.last_updated
                text_parts.append(
                    f"Current ongoing {search_label} streak: {self._format_duration(query_end_time - streak_start)}"
                )

            if not interval_count:
                text_parts.append("No matching recorder intervals were found in that window.")

            return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

        if analysis == "streak":
            if not target_filter_states:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{friendly_name} ({history_entity_id})\n"
                                "Streak analysis requires a target event or state, such as locked, opened, on, or home."
                            ),
                        }
                    ]
                }

            search_label = self._describe_history_target(
                args.get("state"), args.get("event")
            )
            if (
                current_state.state.casefold() not in target_filter_states
                and len(history_candidates) > 1
            ):
                for candidate in history_candidates[1:]:
                    candidate_history = await _load_candidate_history(candidate)
                    if candidate_history is None:
                        continue
                    candidate_current_state = candidate_history["current_state"]
                    if (
                        candidate_history["target_filter_states"]
                        and candidate_current_state.state.casefold()
                        in candidate_history["target_filter_states"]
                    ):
                        history_entity_id = candidate_history["entity_id"]
                        current_state = candidate_current_state
                        friendly_name = candidate_history["friendly_name"]
                        entity_states = candidate_history["entity_states"]
                        target_filter_states = candidate_history["target_filter_states"]
                        resolution_note = candidate["note"]
                        break

            if current_state.state.casefold() not in target_filter_states:
                text_parts = [
                    f"{friendly_name} ({history_entity_id})",
                    f"Current state: {current_state.state}",
                    f"It is not currently {search_label}, so there is no ongoing {search_label} streak to measure.",
                ]
                text_parts = self._prepend_resolution_note(text_parts, resolution_note)
                return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

            streak_start = query_start_time
            exact_start = False
            streak_reaches_window_start = True

            for window_hours in self._build_history_search_windows(hours):
                window_start = query_end_time - timedelta(hours=window_hours)
                window_states = await self._fetch_entity_history_states(
                    history_entity_id,
                    hours=window_hours,
                    end_time=query_end_time,
                    descending=False,
                    include_start_time_state=True,
                )
                if not window_states:
                    continue
                if window_states[-1].state.casefold() not in target_filter_states:
                    break

                start_idx = len(window_states) - 1
                while (
                    start_idx > 0
                    and window_states[start_idx - 1].state.casefold() in target_filter_states
                ):
                    start_idx -= 1

                streak_start = (
                    window_states[start_idx].last_changed
                    or window_states[start_idx].last_updated
                )
                if start_idx > 0:
                    exact_start = True
                    streak_reaches_window_start = False
                    break
                if streak_start > window_start:
                    exact_start = True
                    streak_reaches_window_start = False
                    break
                streak_reaches_window_start = True
                if window_hours >= hours:
                    break

            streak_duration = query_end_time - max(streak_start, query_start_time)

            self.publish_progress(
                "tool_complete",
                "Recorder history streak analysis complete",
                tool="analyze_entity_history",
                success=True,
                seconds=int(streak_duration.total_seconds()),
            )

            duration_text = self._format_duration(streak_duration)
            text_parts = [
                f"{friendly_name} ({history_entity_id})",
                f"Current state: {current_state.state}",
            ]
            if exact_start:
                text_parts.append(
                    f"Current {search_label} streak: {duration_text}"
                )
                text_parts.append(
                    f"Streak started: {self._format_relative_absolute_time(streak_start)}"
                )
            else:
                text_parts.append(
                    f"Current {search_label} streak: at least {duration_text}"
                )
                if streak_reaches_window_start:
                    text_parts.append(
                        f"The streak extends beyond the searched {hours}-hour window."
                    )
            text_parts = self._prepend_resolution_note(text_parts, resolution_note)
            return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

        if analysis == "stats":
            stats_result = self._calculate_numeric_history_stats(
                entity_states,
                query_start_time,
                query_end_time,
            )

            if stats_result is None:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{friendly_name} ({history_entity_id})\n"
                                "Recorder history did not contain numeric states in that window, so min/max/average could not be calculated."
                            ),
                        }
                    ]
                }

            self.publish_progress(
                "tool_complete",
                "Recorder history numeric analysis complete",
                tool="analyze_entity_history",
                success=True,
            )

            min_time = stats_result["min_time"]
            max_time = stats_result["max_time"]
            text_parts = [
                f"{friendly_name} ({history_entity_id})",
                f"Current state: {current_state.state}",
                f"Numeric recorder stats for the last {hours} hour{'s' if hours != 1 else ''}:",
                f"Minimum: {self._format_number(stats_result['min'])}"
                + (
                    f" at {self._format_relative_absolute_time(min_time)}"
                    if min_time is not None
                    else ""
                ),
                f"Maximum: {self._format_number(stats_result['max'])}"
                + (
                    f" at {self._format_relative_absolute_time(max_time)}"
                    if max_time is not None
                    else ""
                ),
                f"Average: {self._format_number(stats_result['average'])} (time-weighted)",
                f"Numeric state samples: {stats_result['sample_count']}",
            ]
            text_parts = self._prepend_resolution_note(text_parts, resolution_note)

            return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

        if target_filter_states:
            matched_states = [
                state
                for state in entity_states
                if state.state.casefold() in target_filter_states
            ]
        else:
            matched_states = entity_states

        if target_filter_states and not matched_states and len(history_candidates) > 1:
            for candidate in history_candidates[1:]:
                candidate_history = await _load_candidate_history(candidate)
                if candidate_history is None:
                    continue
                candidate_matched_states = [
                    state
                    for state in candidate_history["entity_states"]
                    if state.state.casefold() in candidate_history["target_filter_states"]
                ]
                if candidate_matched_states:
                    history_entity_id = candidate_history["entity_id"]
                    current_state = candidate_history["current_state"]
                    friendly_name = candidate_history["friendly_name"]
                    entity_states = candidate_history["entity_states"]
                    target_filter_states = candidate_history["target_filter_states"]
                    matched_states = candidate_matched_states
                    resolution_note = candidate["note"]
                    break

        match_count = len(matched_states)

        self.publish_progress(
            "tool_complete",
            "Recorder history analysis complete",
            tool="analyze_entity_history",
            success=True,
            count=match_count,
        )

        search_label = self._describe_history_target(
            args.get("state"), args.get("event")
        )
        noun = f"{search_label} event" if search_label != "change" else "state change"

        text_parts = [
            f"{friendly_name} ({history_entity_id})",
            f"Current state: {current_state.state}",
            f"Recorded {noun}{'s' if match_count != 1 else ''} in the last {hours} hour{'s' if hours != 1 else ''}: {match_count}",
        ]

        if target_filter_states:
            text_parts.append(
                f"Counted using recorder state{'s' if len(target_filter_states) != 1 else ''}: {', '.join(target_filter_states)}"
            )

        if analysis == "summary" and matched_states:
            first_match = matched_states[0]
            last_match = matched_states[-1]
            first_when = first_match.last_changed or first_match.last_updated
            last_when = last_match.last_changed or last_match.last_updated
            text_parts.append(
                f"First matching event in window: {self._format_relative_absolute_time(first_when)}"
            )
            text_parts.append(
                f"Most recent matching event: {self._format_relative_absolute_time(last_when)}"
            )

        if analysis == "summary" and not matched_states:
            text_parts.append("No matching recorder events were found in that window.")

        text_parts = self._prepend_resolution_note(text_parts, resolution_note)
        return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

    async def tool_get_entity_state_at_time(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return an entity's recorder state at a specific point in time."""
        entity_id = args.get("entity_id")
        raw_datetime = args.get("datetime")
        current_state = self.hass.states.get(entity_id)

        if not current_state:
            return {
                "content": [
                    {"type": "text", "text": f"Entity '{entity_id}' not found."}
                ]
            }

        target_time = self._parse_history_datetime(raw_datetime)
        if target_time is None:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Invalid datetime. Use an ISO 8601 timestamp or a Home Assistant-local datetime string.",
                    }
                ]
            }

        friendly_name = current_state.attributes.get("friendly_name", entity_id)
        lookup_start = target_time + timedelta(microseconds=1)

        self.publish_progress(
            "tool_start",
            f"Looking up recorder state for {entity_id} at {raw_datetime}",
            tool="get_entity_state_at_time",
            entity_id=entity_id,
        )

        try:
            entity_states = await self._fetch_entity_history_states(
                entity_id,
                start_time=lookup_start,
                end_time=lookup_start + timedelta(seconds=1),
                descending=False,
                include_start_time_state=True,
            )
        except Exception as err:
            _LOGGER.error("Failed to look up recorder state for %s: %s", entity_id, err)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Failed to retrieve recorder state: {err}",
                    }
                ]
            }

        state_at_time = entity_states[0] if entity_states else None
        if state_at_time is not None:
            state_when = state_at_time.last_changed or state_at_time.last_updated
            if state_when > target_time:
                state_at_time = None

        self.publish_progress(
            "tool_complete",
            "Recorder point-in-time lookup complete",
            tool="get_entity_state_at_time",
            success=True,
            found=state_at_time is not None,
        )

        target_local = self._format_absolute_time(target_time)
        if state_at_time is None:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{friendly_name} ({entity_id})\n"
                            f"No recorder state was available for {target_local}."
                        ),
                    }
                ]
            }

        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"{friendly_name} ({entity_id})\n"
                        f"State at {target_local}: {state_at_time.state}\n"
                        f"Current state: {current_state.state}"
                    ),
                }
            ]
        }

    async def tool_remember_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Store a persisted memory with TTL."""
        memory_text = " ".join(str(args.get("memory") or "").split()).strip()
        if not memory_text:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Memory text is required.",
                    }
                ],
                "isError": True,
            }

        ttl_days = args.get("ttl_days")
        category = args.get("category")
        self.publish_progress(
            "tool_start",
            "Storing memory",
            tool="remember_memory",
        )

        try:
            stored = await self.memory_manager.remember(
                memory_text,
                default_ttl_days=self._memory_default_ttl_days(),
                max_ttl_days=self._memory_max_ttl_days(),
                ttl_days=None if ttl_days is None else self._coerce_int_arg(
                    ttl_days,
                    default=self._memory_default_ttl_days(),
                    minimum=1,
                    maximum=self._memory_max_ttl_days(),
                ),
                category=category,
                max_items=self._memory_max_items(),
            )
        except Exception as err:
            _LOGGER.error("Failed to store memory: %s", err)
            return {
                "content": [{"type": "text", "text": f"Failed to store memory: {err}"}],
                "isError": True,
            }

        self.publish_progress(
            "tool_complete",
            "Memory stored",
            tool="remember_memory",
            memory_id=stored["id"],
        )

        expires_at = dt_util.parse_datetime(stored["expires_at"])
        expires_text = (
            self._format_relative_absolute_time(expires_at)
            if expires_at is not None
            else "later"
        )
        category_text = (
            f" Category: {stored['category']}."
            if stored.get("category")
            else ""
        )
        prune_text = (
            f" {stored['pruned_count']} old memories were pruned to stay within the configured limit."
            if stored.get("pruned_count")
            else ""
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Stored memory [{stored['id']}].{category_text} "
                        f"It expires {expires_text}.{prune_text}"
                    ),
                }
            ],
            "memory": stored,
        }

    async def tool_recall_memories(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Recall stored memories by query or category."""
        limit = self._coerce_int_arg(
            args.get("limit"),
            default=5,
            minimum=1,
            maximum=50,
        )
        query = args.get("query")
        category = args.get("category")

        self.publish_progress(
            "tool_start",
            "Searching stored memories",
            tool="recall_memories",
        )

        try:
            result = await self.memory_manager.recall(
                query=None if query is None else str(query),
                category=None if category is None else str(category),
                limit=limit,
            )
        except Exception as err:
            _LOGGER.error("Failed to recall memories: %s", err)
            return {
                "content": [{"type": "text", "text": f"Failed to recall memories: {err}"}],
                "isError": True,
            }

        items = result["items"]
        self.publish_progress(
            "tool_complete",
            "Memory recall complete",
            tool="recall_memories",
            count=result["returned_count"],
            total=result["total_found"],
        )

        if not items:
            return {
                "content": [{"type": "text", "text": "No active memories matched."}],
                "memories": [],
                "result_count": 0,
            }

        header = (
            f"Found {result['returned_count']} of {result['total_found']} active memories:"
            if result["remaining_count"] > 0
            else f"Found {result['returned_count']} active memories:"
        )
        lines = [header]
        for memory in items:
            expires_at = dt_util.parse_datetime(str(memory.get("expires_at") or ""))
            expires_text = (
                self._format_relative_absolute_time(expires_at)
                if expires_at is not None
                else "later"
            )
            category_text = (
                f" [{memory['category']}]" if memory.get("category") else ""
            )
            lines.append(
                f"- {memory['id']}{category_text}: {memory['text']} (expires {expires_text})"
            )
        if result["remaining_count"] > 0:
            lines.append(f"{result['remaining_count']} more memories matched but were not shown.")

        return {
            "content": [{"type": "text", "text": "\n".join(lines)}],
            "memories": items,
            "result_count": result["total_found"],
        }

    async def tool_forget_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Delete stored memories by id or query."""
        memory_id = args.get("memory_id")
        query = args.get("query")
        category = args.get("category")
        forget_all_matches = bool(args.get("forget_all_matches", False))

        self.publish_progress(
            "tool_start",
            "Deleting stored memory",
            tool="forget_memory",
        )

        try:
            result = await self.memory_manager.forget(
                memory_id=None if memory_id is None else str(memory_id),
                query=None if query is None else str(query),
                category=None if category is None else str(category),
                delete_all_matches=forget_all_matches,
            )
        except Exception as err:
            _LOGGER.error("Failed to forget memory: %s", err)
            return {
                "content": [{"type": "text", "text": f"Failed to forget memory: {err}"}],
                "isError": True,
            }

        self.publish_progress(
            "tool_complete",
            "Memory deletion complete",
            tool="forget_memory",
            deleted=result["deleted_count"],
        )

        if result["deleted_count"] == 0:
            return {
                "content": [{"type": "text", "text": "No matching memories were deleted."}],
                "deleted_count": 0,
                "deleted": [],
            }

        deleted = result["deleted"]
        lines = [f"Deleted {result['deleted_count']} memory item(s):"]
        for memory in deleted[:10]:
            category_text = (
                f" [{memory['category']}]" if memory.get("category") else ""
            )
            lines.append(f"- {memory['id']}{category_text}: {memory['text']}")
        if len(deleted) > 10:
            lines.append(f"{len(deleted) - 10} additional deleted memories were omitted.")

        return {
            "content": [{"type": "text", "text": "\n".join(lines)}],
            "deleted_count": result["deleted_count"],
            "deleted": deleted,
        }

    def _extract_history_states(
        self, history_result: Dict[str, Any], entity_id: str
    ) -> List[Any]:
        """Extract an entity's state list from recorder helper results."""
        return (
            history_result.get(entity_id)
            or history_result.get(entity_id.casefold())
            or next(iter(history_result.values()), [])
        )

    def _normalize_history_targets(
        self, raw_state: Any, raw_event: Any
    ) -> List[str]:
        """Normalize target states or semantic events into recorder states."""
        values: List[str] = []

        if isinstance(raw_state, list):
            values.extend(str(value) for value in raw_state)
        elif raw_state is not None:
            values.append(str(raw_state))

        if raw_event is not None:
            values.append(str(raw_event))

        expanded: List[str] = []
        for value in values:
            normalized = value.strip().casefold().replace(" ", "_")
            if not normalized:
                continue
            expanded.extend(self._history_state_aliases(normalized))

        deduped: List[str] = []
        seen = set()
        for value in expanded:
            if value not in seen:
                seen.add(value)
                deduped.append(value)
        return deduped

    def _normalize_history_request_values(
        self, raw_state: Any, raw_event: Any
    ) -> List[str]:
        """Normalize raw requested history values before alias expansion."""
        values: List[str] = []

        if isinstance(raw_state, list):
            values.extend(str(value) for value in raw_state)
        elif raw_state is not None:
            values.append(str(raw_state))

        if raw_event is not None:
            values.append(str(raw_event))

        normalized: List[str] = []
        seen = set()
        for value in values:
            item = value.strip().casefold().replace(" ", "_")
            if item and item not in seen:
                seen.add(item)
                normalized.append(item)
        return normalized

    def _choose_history_count_states(
        self, entity_states: List[Any], raw_state: Any, raw_event: Any
    ) -> List[str]:
        """Choose the best recorder states to count for semantic event requests."""
        present_states = {state.state.casefold() for state in entity_states}
        requested_values = self._normalize_history_request_values(raw_state, raw_event)
        chosen: List[str] = []
        seen = set()

        for value in requested_values:
            aliases = self._history_state_aliases(value)
            selected = next((alias for alias in aliases if alias in present_states), aliases[0])
            if selected not in seen:
                seen.add(selected)
                chosen.append(selected)

        return chosen or self._normalize_history_targets(raw_state, raw_event)

    def _calculate_state_duration(
        self,
        entity_states: List[Any],
        target_states: List[str],
        start_time,
        end_time,
    ) -> Dict[str, Any]:
        """Calculate total time spent in the target states over a window."""
        total_duration = timedelta(0)
        interval_count = 0
        first_start = None
        last_end = None

        for index, state in enumerate(entity_states):
            state_start = self._clip_time_to_window(
                state.last_changed or state.last_updated,
                start_time,
                end_time,
            )

            if index + 1 < len(entity_states):
                next_state = entity_states[index + 1]
                state_end = self._clip_time_to_window(
                    next_state.last_changed or next_state.last_updated,
                    start_time,
                    end_time,
                )
            else:
                state_end = end_time

            if state_end <= state_start:
                continue

            if state.state.casefold() not in target_states:
                continue

            total_duration += state_end - state_start
            interval_count += 1
            if first_start is None:
                first_start = state_start
            last_end = state_end

        return {
            "total_duration": total_duration,
            "interval_count": interval_count,
            "first_start": first_start,
            "last_end": last_end,
        }

    def _calculate_numeric_history_stats(
        self,
        entity_states: List[Any],
        start_time,
        end_time,
    ) -> Dict[str, Any] | None:
        """Calculate numeric min/max/average across recorder states."""
        numeric_points: List[tuple[float, Any]] = []
        weighted_sum = 0.0
        weighted_seconds = 0.0

        for index, state in enumerate(entity_states):
            numeric_value = self._coerce_numeric_state(state.state)
            if numeric_value is None:
                continue

            state_time = self._clip_time_to_window(
                state.last_changed or state.last_updated,
                start_time,
                end_time,
            )
            numeric_points.append((numeric_value, state_time))

            if index + 1 < len(entity_states):
                next_state = entity_states[index + 1]
                state_end = self._clip_time_to_window(
                    next_state.last_changed or next_state.last_updated,
                    start_time,
                    end_time,
                )
            else:
                state_end = end_time

            seconds = max((state_end - state_time).total_seconds(), 0)
            if seconds > 0:
                weighted_sum += numeric_value * seconds
                weighted_seconds += seconds

        if not numeric_points:
            return None

        min_value, min_time = min(numeric_points, key=lambda item: item[0])
        max_value, max_time = max(numeric_points, key=lambda item: item[0])

        if weighted_seconds > 0:
            average = weighted_sum / weighted_seconds
        else:
            average = sum(value for value, _ in numeric_points) / len(numeric_points)

        return {
            "min": min_value,
            "min_time": min_time,
            "max": max_value,
            "max_time": max_time,
            "average": average,
            "sample_count": len(numeric_points),
        }

    def _clip_time_to_window(self, when, start_time, end_time):
        """Clip a timestamp to the requested analysis window."""
        if when < start_time:
            return start_time
        if when > end_time:
            return end_time
        return when

    def _coerce_numeric_state(self, value: Any) -> float | None:
        """Convert a recorder state value into a numeric value when possible."""
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            return None

        if number != number or number in (float("inf"), float("-inf")):
            return None

        return number

    def _format_duration(self, duration: timedelta) -> str:
        """Format a duration in human-friendly units."""
        total_seconds = max(int(duration.total_seconds()), 0)
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts: List[str] = []
        if days:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds and not parts:
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

        return ", ".join(parts) if parts else "0 seconds"

    def _format_number(self, value: float) -> str:
        """Format numeric values compactly for responses."""
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.3f}".rstrip("0").rstrip(".")

    def _history_preferred_domains(self, requested_values: List[str]) -> Dict[str, int]:
        """Return domain preferences for semantic history requests."""
        domain_scores: Dict[str, int] = {}
        for value in requested_values:
            if value in {"locked", "unlocked"}:
                domain_scores["lock"] = max(domain_scores.get("lock", 0), 120)
            elif value in {"open", "opened", "opening", "close", "closed", "closing"}:
                domain_scores["cover"] = max(domain_scores.get("cover", 0), 90)
                domain_scores["binary_sensor"] = max(
                    domain_scores.get("binary_sensor", 0), 80
                )
                domain_scores["lock"] = max(domain_scores.get("lock", 0), 40)
            elif value in {"home", "away", "not_home"}:
                domain_scores["person"] = max(domain_scores.get("person", 0), 100)
                domain_scores["device_tracker"] = max(
                    domain_scores.get("device_tracker", 0), 95
                )
            elif value in {"detected", "triggered", "clear", "cleared"}:
                domain_scores["binary_sensor"] = max(
                    domain_scores.get("binary_sensor", 0), 90
                )
        return domain_scores

    def _history_preferred_device_classes(self, requested_values: List[str]) -> set[str]:
        """Return preferred device classes for semantic history requests."""
        device_classes: set[str] = set()
        for value in requested_values:
            if value in {"open", "opened", "opening", "close", "closed", "closing"}:
                device_classes.update({"door", "window", "opening", "garage_door"})
            elif value in {"detected", "triggered", "clear", "cleared"}:
                device_classes.update(
                    {
                        "motion",
                        "occupancy",
                        "presence",
                        "door",
                        "window",
                        "opening",
                        "garage_door",
                        "sound",
                        "vibration",
                    }
                )
        return device_classes

    def _score_entity_for_history_request(
        self,
        state_obj: Any,
        requested_values: List[str],
        target_states: List[str],
    ) -> int:
        """Score how appropriate an entity is for a semantic history request."""
        if state_obj is None or not requested_values:
            return 0

        domain = state_obj.entity_id.split(".", 1)[0]
        device_class = str(state_obj.attributes.get("device_class", "")).casefold()
        current_state = state_obj.state.casefold()
        preferred_domains = self._history_preferred_domains(requested_values)
        preferred_device_classes = self._history_preferred_device_classes(
            requested_values
        )

        score = preferred_domains.get(domain, 0)
        if device_class and device_class in preferred_device_classes:
            score += 25
        if current_state in target_states:
            score += 20

        return score

    def _history_relation_tokens(self, value: Any) -> List[str]:
        """Normalize an entity-related name into meaningful comparison tokens."""
        if value in (None, ""):
            return []

        stopwords = {
            "a",
            "an",
            "the",
            "entity",
            "device",
            "binary",
            "sensor",
            "contact",
            "contacts",
            "lock",
            "locks",
            "locked",
            "unlock",
            "unlocked",
            "deadbolt",
            "bolt",
        }
        raw_tokens = re.findall(r"[a-z0-9]+", str(value).casefold())
        filtered = [
            token
            for token in raw_tokens
            if len(token) > 1 and token not in stopwords
        ]
        return filtered or [token for token in raw_tokens if len(token) > 1]

    def _history_relation_texts(self, state_obj: Any) -> List[str]:
        """Collect stable entity name variants for related-entity matching."""
        if state_obj is None:
            return []

        texts: List[str] = []
        for value in (
            getattr(state_obj, "name", None),
            state_obj.attributes.get("friendly_name"),
            state_obj.entity_id.split(".", 1)[1].replace("_", " "),
        ):
            text = str(value).strip() if value not in (None, "") else ""
            if text and text not in texts:
                texts.append(text)
        return texts

    def _history_entity_location_signature(
        self, entity_id: str
    ) -> Tuple[str | None, str | None, str | None]:
        """Return device, area, and floor identifiers for an entity."""
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        area_registry = ar.async_get(self.hass)
        floor_registry = fr.async_get(self.hass) if fr else None

        entity_entry = entity_registry.async_get(entity_id)
        if entity_entry is None:
            return None, None, None

        device_entry = (
            device_registry.async_get(entity_entry.device_id)
            if entity_entry.device_id
            else None
        )
        area_id = entity_entry.area_id or (device_entry.area_id if device_entry else None)
        area_entry = area_registry.async_get_area(area_id) if area_id else None
        floor_id = getattr(area_entry, "floor_id", None) if area_entry else None
        if floor_id and floor_registry is not None:
            floor_entry = floor_registry.async_get_floor(floor_id)
            floor_id = floor_entry.floor_id if floor_entry else floor_id

        return entity_entry.device_id, area_id, floor_id

    def _score_history_entity_relatedness(
        self, current_state: Any, candidate_state: Any
    ) -> Tuple[int, bool]:
        """Score how likely two entities describe the same real-world thing."""
        if current_state is None or candidate_state is None:
            return 0, False

        current_device_id, current_area_id, current_floor_id = (
            self._history_entity_location_signature(current_state.entity_id)
        )
        candidate_device_id, candidate_area_id, candidate_floor_id = (
            self._history_entity_location_signature(candidate_state.entity_id)
        )

        best_name_score = 0
        for current_text in self._history_relation_texts(current_state):
            current_tokens = self._history_relation_tokens(current_text)
            current_canonical = " ".join(current_tokens)
            if not current_tokens:
                continue

            for candidate_text in self._history_relation_texts(candidate_state):
                candidate_tokens = self._history_relation_tokens(candidate_text)
                candidate_canonical = " ".join(candidate_tokens)
                if not candidate_tokens:
                    continue

                overlap = set(current_tokens) & set(candidate_tokens)
                if not overlap:
                    continue

                score = len(overlap) * 22
                coverage = len(overlap) / min(len(current_tokens), len(candidate_tokens))
                score += int(coverage * 55)

                if current_canonical and current_canonical == candidate_canonical:
                    score += 80
                elif (
                    current_canonical
                    and candidate_canonical
                    and min(len(current_canonical), len(candidate_canonical)) >= 5
                    and (
                        current_canonical in candidate_canonical
                        or candidate_canonical in current_canonical
                    )
                ):
                    score += 40

                best_name_score = max(best_name_score, score)

        same_device = bool(
            current_device_id
            and candidate_device_id
            and current_device_id == candidate_device_id
        )
        if same_device:
            best_name_score += 120
        elif best_name_score:
            if current_area_id and candidate_area_id and current_area_id == candidate_area_id:
                best_name_score += 10
            if (
                current_floor_id
                and candidate_floor_id
                and current_floor_id == candidate_floor_id
            ):
                best_name_score += 5

        return best_name_score, same_device

    def _resolve_history_entity_for_request(
        self,
        entity_id: str,
        raw_state: Any,
        raw_event: Any,
    ) -> tuple[str, str | None]:
        """Resolve the best entity for semantic history requests when needed."""
        candidates = self._history_resolution_candidates(entity_id, raw_state, raw_event)
        best_candidate = candidates[0]
        return best_candidate["entity_id"], best_candidate["note"]

    def _history_resolution_candidates(
        self,
        entity_id: str,
        raw_state: Any,
        raw_event: Any,
        *,
        max_candidates: int = 4,
    ) -> List[Dict[str, Any]]:
        """Return strong related-entity candidates for a semantic history request."""
        requested_values = self._normalize_history_request_values(raw_state, raw_event)
        if not requested_values:
            return [{"entity_id": entity_id, "note": None}]

        target_states = self._normalize_history_targets(raw_state, raw_event)
        current_state = self.hass.states.get(entity_id)
        if current_state is None:
            return [{"entity_id": entity_id, "note": None}]

        current_score = self._score_entity_for_history_request(
            current_state,
            requested_values,
            target_states,
        )
        candidates: List[Dict[str, Any]] = [
            {
                "entity_id": entity_id,
                "note": None,
                "priority": current_score + 150,
                "semantic_score": current_score,
                "relation_score": 0,
            }
        ]

        requested_label = self._describe_history_target(raw_state, raw_event)

        for sibling_state in self.hass.states.async_all():
            sibling_entity_id = sibling_state.entity_id
            if sibling_entity_id == entity_id:
                continue
            if not async_should_expose(self.hass, "conversation", sibling_entity_id):
                continue
            sibling_score = self._score_entity_for_history_request(
                sibling_state,
                requested_values,
                target_states,
            )
            if sibling_score <= 0:
                continue

            relation_score, same_device = self._score_history_entity_relatedness(
                current_state,
                sibling_state,
            )
            if relation_score <= 0:
                continue

            total_score = sibling_score + relation_score
            if same_device:
                if (
                    sibling_score <= current_score
                    or total_score < current_score + 40
                ):
                    continue
            else:
                if (
                    sibling_score < current_score + 20
                    or relation_score < 90
                    or total_score < current_score + 100
                ):
                    continue

            reason_text = (
                "is on the same device and"
                if same_device
                else "strongly matches the same named thing and"
            )
            candidates.append(
                {
                    "entity_id": sibling_entity_id,
                    "note": (
                        f"Using related entity {sibling_entity_id} because it {reason_text} "
                        f"the requested {requested_label} history applies more directly to that entity "
                        f"than {entity_id}."
                    ),
                    "priority": total_score + (40 if same_device else 0),
                    "semantic_score": sibling_score,
                    "relation_score": relation_score,
                }
            )

        candidates.sort(
            key=lambda item: (
                -int(item["priority"]),
                -int(item["semantic_score"]),
                item["entity_id"],
            )
        )
        return candidates[:max_candidates]

    def _prepend_resolution_note(
        self, text_parts: List[str], resolution_note: str | None
    ) -> List[str]:
        """Prepend a history-entity resolution note when a sibling entity was used."""
        if not resolution_note:
            return text_parts
        return [resolution_note, ""] + text_parts

    def _history_state_aliases(self, value: str) -> List[str]:
        """Map semantic event words to likely recorder states."""
        aliases = {
            "open": ["open", "opening", "on"],
            "opened": ["open", "opening", "on"],
            "opening": ["opening", "open", "on"],
            "close": ["closed", "closing", "off"],
            "closed": ["closed", "closing", "off"],
            "closing": ["closing", "closed", "off"],
            "on": ["on"],
            "turned_on": ["on"],
            "enabled": ["on"],
            "active": ["on"],
            "off": ["off"],
            "turned_off": ["off"],
            "disabled": ["off"],
            "locked": ["locked"],
            "unlocked": ["unlocked"],
            "detected": ["detected", "on", "open"],
            "triggered": ["triggered", "on"],
            "clear": ["clear", "off", "closed"],
            "cleared": ["clear", "off", "closed"],
            "home": ["home"],
            "away": ["away", "not_home"],
            "not_home": ["not_home", "away"],
        }
        return aliases.get(value, [value])

    def _describe_history_target(self, raw_state: Any, raw_event: Any) -> str:
        """Create a readable description of the recorder search target."""
        if raw_event:
            return str(raw_event).replace("_", " ").strip()
        if isinstance(raw_state, list):
            return " / ".join(str(value).replace("_", " ").strip() for value in raw_state)
        if raw_state:
            return str(raw_state).replace("_", " ").strip()
        return "change"

    def _format_relative_time(self, when) -> str:
        """Format a timestamp relative to now."""
        now = dt_util.utcnow()
        seconds = max((now - when).total_seconds(), 0)

        if seconds < 60:
            return "just now"
        if seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        if seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        if seconds < 604800:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"

        weeks = int(seconds / 604800)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"

    def _format_absolute_time(self, when) -> str:
        """Format a timestamp in the user's local time zone."""
        local_when = dt_util.as_local(when)
        now_local = dt_util.as_local(dt_util.utcnow())
        time_text = local_when.strftime("%I:%M %p %Z").lstrip("0")

        if local_when.date() == now_local.date():
            day_text = "today"
        elif local_when.date() == now_local.date() - timedelta(days=1):
            day_text = "yesterday"
        elif local_when.date() == now_local.date() + timedelta(days=1):
            day_text = "tomorrow"
        else:
            date_text = local_when.strftime("%b %d").replace(" 0", " ")
            if local_when.year != now_local.year:
                date_text += f", {local_when.year}"
            day_text = f"on {date_text}"

        return f"{time_text} {day_text}"

    def _format_relative_absolute_time(self, when) -> str:
        """Format a timestamp with both relative and absolute local time."""
        relative = self._format_relative_time(when)
        absolute = self._format_absolute_time(when)
        return f"{relative} at {absolute}"

    def _parse_history_datetime(self, value: Any):
        """Parse a recorder lookup datetime and assume local time if naive."""
        if value is None:
            return None

        parsed = dt_util.parse_datetime(str(value))
        if parsed is None:
            return None

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=getattr(dt_util, "DEFAULT_TIME_ZONE", dt_util.now().tzinfo)
            )

        return dt_util.as_utc(parsed)

    def _coerce_int_arg(
        self, value: Any, *, default: int, minimum: int, maximum: int
    ) -> int:
        """Coerce an integer-like tool argument safely."""
        if value is None:
            parsed = default
        elif isinstance(value, bool):
            parsed = default
        elif isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            parsed = int(value)
        else:
            try:
                parsed = int(str(value).strip())
            except (TypeError, ValueError):
                parsed = default

        return max(minimum, min(parsed, maximum))

    def _create_assist_llm_context(self) -> llm.LLMContext:
        """Create an LLM context for the native Home Assistant Assist API."""
        return llm.LLMContext(
            platform=DOMAIN,
            context=Context(),
            language="*",
            assistant=conversation.DOMAIN,
            device_id=None,
        )

    async def _get_assist_api_instance(self) -> llm.APIInstance:
        """Get the built-in Home Assistant Assist API instance."""
        return await llm.async_get_api(
            self.hass, llm.LLM_API_ASSIST, self._create_assist_llm_context()
        )

    def _assist_api_has_live_context_tool(self, llm_api: llm.APIInstance) -> bool:
        """Return whether the Assist API exposes GetLiveContext."""
        return any(tool.name == "GetLiveContext" for tool in llm_api.tools)

    def _format_assist_tool_input_schema(
        self,
        tool: llm.Tool,
        custom_serializer,
    ) -> Dict[str, Any]:
        """Convert an Assist tool schema to JSON schema for inspection."""
        try:
            input_schema = convert(
                tool.parameters, custom_serializer=custom_serializer
            )
        except Exception as err:
            _LOGGER.debug(
                "Failed to convert native Assist tool schema for %s: %s",
                tool.name,
                err,
            )
            return {"type": "object", "properties": {}}

        return (
            input_schema
            if isinstance(input_schema, dict)
            else {"type": "object", "properties": {}}
        )

    async def _call_assist_api_tool(
        self,
        llm_api: llm.APIInstance,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call a native Home Assistant Assist tool safely."""
        tool_input = llm.ToolInput(tool_name=tool_name, tool_args=arguments)
        _LOGGER.debug(
            "Calling native Assist tool: %s(%s)",
            tool_input.tool_name,
            tool_input.tool_args,
        )

        try:
            result = await llm_api.async_call_tool(tool_input)
        except (HomeAssistantError, vol.Invalid) as err:
            raise HomeAssistantError(
                f"Error calling native Assist tool '{tool_name}': {err}"
            ) from err

        if not isinstance(result, dict):
            return {"result": self._serialize_service_response_value(result)}
        return result

    def _build_assist_tool_response_summary(self, response: Any) -> List[str]:
        """Build a concise summary for a native Assist tool response."""
        if not isinstance(response, dict):
            return []

        lines: List[str] = []

        speech = response.get("speech")
        if isinstance(speech, dict):
            plain_speech = speech.get("plain")
            if isinstance(plain_speech, dict) and plain_speech.get("speech"):
                lines.append("Summary:")
                lines.append(f"- Speech: {plain_speech['speech']}")

        data = response.get("data")
        if isinstance(data, dict) and (
            "success" in data or "failed" in data or "targets" in data
        ):
            if not lines:
                lines.append("Summary:")
            detail_parts = []
            if "success" in data:
                detail_parts.append(f"success={data['success']}")
            if "failed" in data:
                detail_parts.append(f"failed={data['failed']}")
            targets = data.get("targets")
            if isinstance(targets, list):
                detail_parts.append(f"targets={len(targets)}")
            if detail_parts:
                lines.append("- Result: " + ", ".join(detail_parts))

        response_type = response.get("response_type")
        if response_type and not lines:
            lines.append("Summary:")
            lines.append(f"- Response type: {response_type}")

        return lines

    def _build_history_search_windows(self, max_hours: int) -> List[int]:
        """Build progressively larger recorder search windows."""
        windows = [24, 168, 720, max_hours]
        deduped: List[int] = []
        seen = set()

        for window in windows:
            clamped = min(window, max_hours)
            if clamped > 0 and clamped not in seen:
                seen.add(clamped)
                deduped.append(clamped)

        return deduped

    async def _fetch_entity_history_states(
        self,
        entity_id: str,
        hours: int | None = None,
        *,
        start_time=None,
        end_time=None,
        descending: bool = True,
        limit: int | None = None,
        include_start_time_state: bool = False,
    ) -> List[Any]:
        """Fetch recorder-backed history states for a single entity."""
        query_end_time = end_time or dt_util.utcnow()
        query_start_time = start_time
        if query_start_time is None:
            if hours is None:
                raise ValueError("Either hours or start_time must be provided")
            query_start_time = query_end_time - timedelta(hours=hours)

        states = await self.hass.async_add_executor_job(
            lambda: history.state_changes_during_period(
                self.hass,
                query_start_time,
                end_time=query_end_time,
                entity_id=entity_id,
                no_attributes=True,
                descending=descending,
                limit=limit,
                include_start_time_state=include_start_time_state,
            )
        )
        return self._extract_history_states(states, entity_id)

    def _get_music_assistant_instances(self) -> List[Any]:
        """Return configured Music Assistant config entries."""
        return list(self.hass.config_entries.async_entries("music_assistant"))

    def _resolve_music_assistant_instance(
        self,
        *,
        config_entry_id: Any = None,
        instance: Any = None,
    ) -> Any:
        """Resolve a single Music Assistant config entry."""
        instances = self._get_music_assistant_instances()
        if not instances:
            raise ValueError("No Music Assistant instances are configured in Home Assistant.")

        config_entry_id_text = str(config_entry_id or "").strip()
        instance_text = str(instance or "").strip().casefold()

        matched = instances
        if config_entry_id_text:
            matched = [
                entry for entry in matched if entry.entry_id == config_entry_id_text
            ]
            if not matched:
                raise ValueError(
                    f"No Music Assistant instance matched config_entry_id '{config_entry_id_text}'."
                )

        if instance_text:
            exact_matches = [
                entry for entry in matched if entry.title.casefold() == instance_text
            ]
            if exact_matches:
                matched = exact_matches
            else:
                partial_matches = [
                    entry for entry in matched if instance_text in entry.title.casefold()
                ]
                if partial_matches:
                    matched = partial_matches
                else:
                    raise ValueError(
                        f"No Music Assistant instance matched '{instance}'. Use list_music_assistant_instances first."
                    )

        if len(matched) == 1:
            return matched[0]

        raise ValueError(
            "Multiple Music Assistant instances are configured. Use list_music_assistant_instances and pass config_entry_id or instance."
        )

    def _is_music_assistant_entity(self, entity_entry: Any) -> bool:
        """Return whether an entity registry entry belongs to Music Assistant."""
        if entity_entry is None:
            return False

        platform = str(getattr(entity_entry, "platform", "") or "").strip()
        if platform == "music_assistant":
            return True

        config_entry_id = getattr(entity_entry, "config_entry_id", None)
        if config_entry_id:
            config_entry = self.hass.config_entries.async_get_entry(config_entry_id)
            if config_entry and config_entry.domain == "music_assistant":
                return True

        return False

    def _get_music_assistant_player_catalog(self) -> List[Dict[str, Any]]:
        """Build a catalog of exposed Music Assistant players."""
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        area_registry = ar.async_get(self.hass)
        floor_registry = fr.async_get(self.hass) if fr else None
        label_registry = lr.async_get(self.hass) if lr else None

        catalog: List[Dict[str, Any]] = []
        for state_obj in self.hass.states.async_all():
            if state_obj.domain != "media_player":
                continue
            if not async_should_expose(self.hass, "conversation", state_obj.entity_id):
                continue

            entity_entry = entity_registry.async_get(state_obj.entity_id)
            if not self._is_music_assistant_entity(entity_entry):
                continue

            entity_context = self.discovery._get_entity_context(
                entity_entry,
                device_registry,
                area_registry,
                floor_registry,
                label_registry,
            )
            entity_info = self.discovery._create_entity_info(
                state_obj,
                entity_entry=entity_entry,
                entity_context=entity_context,
            )

            config_entry_id = getattr(entity_entry, "config_entry_id", None)
            config_entry = (
                self.hass.config_entries.async_get_entry(config_entry_id)
                if config_entry_id
                else None
            )
            if config_entry_id:
                entity_info["config_entry_id"] = config_entry_id
            if config_entry:
                entity_info["instance_title"] = config_entry.title
            entity_info["integration"] = "music_assistant"
            for attr in (
                "media_title",
                "media_artist",
                "media_album_name",
                "source",
                "volume_level",
                "is_volume_muted",
            ):
                if attr in state_obj.attributes:
                    entity_info[attr] = self._serialize_service_response_value(
                        state_obj.attributes.get(attr)
                    )

            catalog.append(
                {
                    "entity_id": state_obj.entity_id,
                    "state_obj": state_obj,
                    "entity_entry": entity_entry,
                    "entity_context": entity_context,
                    "entity_info": entity_info,
                }
            )

        catalog.sort(
            key=lambda record: (
                record["entity_info"].get("name", "").casefold(),
                record["entity_id"],
            )
        )
        return catalog

    def _resolve_music_assistant_area_values(
        self,
        area_values: List[str],
        area_registry: Any,
        floor_registry: Any,
    ) -> Tuple[set[str], set[str]]:
        """Resolve Music Assistant area selectors, with floor fallback."""
        area_ids: set[str] = set()
        floor_ids: set[str] = set()

        for value in area_values:
            area_entry = self.discovery._resolve_area_entry(value, area_registry)
            if area_entry is not None:
                area_ids.add(area_entry.id)
                continue

            floor_entry = self.discovery._resolve_floor_entry(value, floor_registry)
            if floor_entry is not None:
                floor_ids.add(floor_entry.floor_id)
                continue

            raise ValueError(
                f"No Home Assistant area or floor matched '{value}' for Music Assistant targeting."
            )

        return area_ids, floor_ids

    def _match_music_assistant_player_term(
        self,
        catalog: List[Dict[str, Any]],
        search_term: str,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Resolve a player selector term to the strongest matching Music Assistant players."""
        normalized_term = str(search_term or "").strip().casefold()
        if not normalized_term:
            return [], []

        scored_matches: List[Tuple[int, Dict[str, Any], List[str]]] = []
        for record in catalog:
            score, reasons = self.discovery._get_entity_match_details(
                normalized_term,
                record["state_obj"],
                record["entity_entry"],
                record["entity_context"],
            )
            if score > 0:
                scored_matches.append((score, record, reasons))

        if not scored_matches:
            raise ValueError(
                f"No Music Assistant player matched '{search_term}'. Use list_music_assistant_players first."
            )

        best_score = max(score for score, _, _ in scored_matches)
        best_matches = [
            {
                **record,
                "match_score": score,
                "match_reasons": reasons,
            }
            for score, record, reasons in scored_matches
            if score == best_score
        ]

        if len(best_matches) > 5:
            raise ValueError(
                f"Music Assistant player selector '{search_term}' is too broad. Use list_music_assistant_players to narrow it down."
            )

        return best_matches, [match["entity_info"]["name"] for match in best_matches]

    async def _discover_music_assistant_players(
        self,
        *,
        area: Any = None,
        floor: Any = None,
        label: Any = None,
        name_contains: Any = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Discover Music Assistant players with alias-aware filters."""
        catalog = self._get_music_assistant_player_catalog()
        if not catalog:
            return []

        area_values = self._normalize_target_values(area)
        floor_values = self._normalize_target_values(floor)
        label_values = self._normalize_target_values(label)

        area_registry = ar.async_get(self.hass)
        floor_registry = fr.async_get(self.hass) if fr else None
        label_registry = lr.async_get(self.hass) if lr else None

        area_ids: set[str] = set()
        floor_ids: set[str] = set()
        if area_values:
            resolved_area_ids, resolved_floor_ids = self._resolve_music_assistant_area_values(
                area_values,
                area_registry,
                floor_registry,
            )
            area_ids.update(resolved_area_ids)
            floor_ids.update(resolved_floor_ids)

        for value in floor_values:
            floor_entry = self.discovery._resolve_floor_entry(value, floor_registry)
            if floor_entry is None:
                raise ValueError(
                    f"No Home Assistant floor matched '{value}' for Music Assistant filtering."
                )
            floor_ids.add(floor_entry.floor_id)

        label_ids: set[str] = set()
        for value in label_values:
            label_entry = self.discovery._resolve_label_entry(value, label_registry)
            if label_entry is None:
                raise ValueError(
                    f"No Home Assistant label matched '{value}' for Music Assistant filtering."
                )
            label_ids.add(label_entry.label_id)

        filtered_records = []
        for record in catalog:
            info = record["entity_info"]
            if area_ids and info.get("area_id") not in area_ids:
                continue
            if floor_ids and info.get("floor_id") not in floor_ids:
                continue
            if label_ids and not label_ids.intersection(set(info.get("label_ids", []))):
                continue
            filtered_records.append(record)

        search_term = str(name_contains or "").strip().casefold()
        if search_term:
            scored_records = []
            for record in filtered_records:
                score, reasons = self.discovery._get_entity_match_details(
                    search_term,
                    record["state_obj"],
                    record["entity_entry"],
                    record["entity_context"],
                )
                if score <= 0:
                    continue
                entity_info = dict(record["entity_info"])
                entity_info["match_score"] = score
                entity_info["match_reasons"] = reasons
                scored_records.append((score, entity_info))

            scored_records.sort(
                key=lambda item: (-item[0], item[1].get("name", "").casefold(), item[1]["entity_id"])
            )
            return [entity_info for _, entity_info in scored_records[:limit]]

        return [dict(record["entity_info"]) for record in filtered_records[:limit]]

    async def _resolve_music_assistant_player_targets(
        self,
        *,
        area: Any = None,
        floor: Any = None,
        label: Any = None,
        media_player: Any = None,
    ) -> Tuple[List[str], str]:
        """Resolve Music Assistant selectors to concrete player entity IDs."""
        catalog = self._get_music_assistant_player_catalog()
        if not catalog:
            raise ValueError("No exposed Music Assistant players are available.")

        area_values = self._normalize_target_values(area)
        floor_values = self._normalize_target_values(floor)
        label_values = self._normalize_target_values(label)
        media_player_values = self._normalize_target_values(media_player)

        area_registry = ar.async_get(self.hass)
        floor_registry = fr.async_get(self.hass) if fr else None
        label_registry = lr.async_get(self.hass) if lr else None

        selector_sets: List[set[str]] = []

        if area_values:
            area_ids, floor_ids = self._resolve_music_assistant_area_values(
                area_values,
                area_registry,
                floor_registry,
            )
            matched = {
                record["entity_id"]
                for record in catalog
                if (
                    (area_ids and record["entity_info"].get("area_id") in area_ids)
                    or (floor_ids and record["entity_info"].get("floor_id") in floor_ids)
                )
            }
            if not matched:
                raise ValueError(
                    f"No Music Assistant players matched area selector(s): {', '.join(area_values)}"
                )
            selector_sets.append(matched)

        if floor_values:
            floor_ids = set()
            for value in floor_values:
                floor_entry = self.discovery._resolve_floor_entry(value, floor_registry)
                if floor_entry is None:
                    raise ValueError(
                        f"No Home Assistant floor matched '{value}' for Music Assistant targeting."
                    )
                floor_ids.add(floor_entry.floor_id)

            matched = {
                record["entity_id"]
                for record in catalog
                if record["entity_info"].get("floor_id") in floor_ids
            }
            if not matched:
                raise ValueError(
                    f"No Music Assistant players matched floor selector(s): {', '.join(floor_values)}"
                )
            selector_sets.append(matched)

        if label_values:
            label_ids = set()
            for value in label_values:
                label_entry = self.discovery._resolve_label_entry(value, label_registry)
                if label_entry is None:
                    raise ValueError(
                        f"No Home Assistant label matched '{value}' for Music Assistant targeting."
                    )
                label_ids.add(label_entry.label_id)

            matched = {
                record["entity_id"]
                for record in catalog
                if label_ids.intersection(set(record["entity_info"].get("label_ids", [])))
            }
            if not matched:
                raise ValueError(
                    f"No Music Assistant players matched label selector(s): {', '.join(label_values)}"
                )
            selector_sets.append(matched)

        if media_player_values:
            matched_entity_ids: set[str] = set()
            resolved_names: List[str] = []
            for value in media_player_values:
                matches, match_names = self._match_music_assistant_player_term(catalog, value)
                matched_entity_ids.update(match["entity_id"] for match in matches)
                resolved_names.extend(match_names)
            selector_sets.append(matched_entity_ids)

        if selector_sets:
            resolved_entity_ids = sorted(set.intersection(*selector_sets))
            if not resolved_entity_ids:
                raise ValueError(
                    "No Music Assistant players matched the combined selectors."
                )
        else:
            if len(catalog) == 1:
                resolved_entity_ids = [catalog[0]["entity_id"]]
            else:
                raise ValueError(
                    "Music Assistant playback needs a target player, area, floor, or label when multiple Music Assistant players are exposed."
                )

        return (
            resolved_entity_ids,
            "Resolved Music Assistant players: "
            + ", ".join(self._friendly_names_for_entities(resolved_entity_ids)),
        )

    def _normalize_music_assistant_media_id(self, value: Any) -> Any:
        """Normalize Music Assistant media_id input, including semicolon-separated lists."""
        if value is None:
            return None

        if isinstance(value, (list, tuple, set)):
            normalized_values = []
            for item in value:
                item_text = str(item).strip()
                if item_text:
                    normalized_values.append(item_text)
            if not normalized_values:
                return None
            return normalized_values if len(normalized_values) > 1 else normalized_values[0]

        value_text = str(value).strip()
        if not value_text:
            return None

        if ";" in value_text:
            parts = [part.strip() for part in value_text.split(";") if part.strip()]
            if parts:
                return parts if len(parts) > 1 else parts[0]

        return value_text

    def _normalize_music_assistant_media_type_filter(self, value: Any) -> Any:
        """Normalize Music Assistant media_type filters."""
        allowed = {"track", "album", "artist", "playlist", "radio"}
        if value is None:
            return None

        if isinstance(value, (list, tuple, set)):
            normalized = []
            for item in value:
                item_text = str(item).strip().lower()
                if item_text in allowed and item_text not in normalized:
                    normalized.append(item_text)
            return normalized or None

        value_text = str(value).strip().lower()
        if not value_text:
            return None
        if "," in value_text:
            parts = [
                part.strip()
                for part in value_text.replace(";", ",").split(",")
                if part.strip()
            ]
            normalized = [part for part in parts if part in allowed]
            return normalized or None
        return value_text if value_text in allowed else None

    async def _call_music_assistant_response_service(
        self,
        *,
        service: str,
        service_data: Dict[str, Any],
        summary_label: str,
    ) -> Dict[str, Any]:
        """Call a Music Assistant response service and format the result."""
        if not self.hass.services.has_service("music_assistant", service):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"The Home Assistant Music Assistant integration does not expose music_assistant.{service}.",
                    }
                ]
            }

        self.publish_progress(
            "tool_start",
            f"Calling Music Assistant {service}",
            tool=f"music_assistant.{service}",
        )

        try:
            response = await self.hass.services.async_call(
                domain="music_assistant",
                service=service,
                service_data=service_data,
                blocking=True,
                return_response=True,
            )
        except Exception as err:
            error_msg = f"Music Assistant {service} failed: {err}"
            _LOGGER.exception(error_msg)
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}

        self.publish_progress(
            "tool_complete",
            f"Music Assistant {service} completed",
            tool=f"music_assistant.{service}",
            success=True,
        )

        serialized_response = self._serialize_service_response_value(response)
        text_parts = [f"✅ Retrieved {summary_label}."]
        summary_lines = self._summarize_music_assistant_response(service, serialized_response)
        if summary_lines:
            text_parts.append("")
            text_parts.extend(summary_lines)
        text_parts.append("")
        text_parts.append("Response:")
        text_parts.append(json.dumps(serialized_response, indent=2, ensure_ascii=False))

        return {
            "content": [{"type": "text", "text": "\n".join(text_parts)}],
            "response": serialized_response,
        }

    def _summarize_music_assistant_response(
        self, service: str, response: Any
    ) -> List[str]:
        """Build concise summaries for Music Assistant response payloads."""
        if service in {"search", "get_library"}:
            return self._summarize_music_assistant_collection_response(response)
        if service == "get_queue":
            return self._summarize_music_assistant_queue_response(response)
        return []

    def _summarize_music_assistant_collection_response(self, response: Any) -> List[str]:
        """Summarize Music Assistant search/library payloads."""
        if not isinstance(response, dict):
            return []

        collections: List[Tuple[str, Dict[str, Any]]] = []
        if isinstance(response.get("items"), list):
            collections.append(("items", response))
        else:
            for key, value in response.items():
                if isinstance(value, dict) and isinstance(value.get("items"), list):
                    collections.append((str(key), value))

        if not collections:
            return []

        lines = ["Summary:"]
        for key, payload in collections:
            items = payload.get("items") or []
            preview = self._describe_music_assistant_item(items[0]) if items else None
            line = f"- {key}: {len(items)} item{'s' if len(items) != 1 else ''}"
            if preview:
                line += f"; first: {preview}"
            lines.append(line)
        return lines

    def _summarize_music_assistant_queue_response(self, response: Any) -> List[str]:
        """Summarize Music Assistant queue payloads."""
        if not isinstance(response, dict):
            return []

        queue_payloads: List[Tuple[str, Dict[str, Any]]] = []
        if "items" in response or "current_item" in response:
            queue_payloads.append(("queue", response))
        else:
            for key, value in response.items():
                if isinstance(value, dict) and (
                    "items" in value or "current_item" in value
                ):
                    queue_payloads.append((str(key), value))

        if not queue_payloads:
            return []

        lines = ["Summary:"]
        for key, payload in queue_payloads:
            items = payload.get("items") or []
            current_item = payload.get("current_item")
            detail_parts = [f"{len(items)} queued item{'s' if len(items) != 1 else ''}"]
            current_preview = self._describe_music_assistant_item(current_item)
            if current_preview:
                detail_parts.append(f"current: {current_preview}")
            lines.append(f"- {key}: {'; '.join(detail_parts)}")
        return lines

    def _describe_music_assistant_item(self, item: Any) -> str | None:
        """Build a compact description for a Music Assistant media item."""
        if not isinstance(item, dict):
            return str(item) if item is not None else None

        name = item.get("name") or item.get("title") or item.get("uri")
        artist = item.get("artist")
        if isinstance(artist, dict):
            artist = artist.get("name")
        elif isinstance(artist, list) and artist:
            first_artist = artist[0]
            artist = first_artist.get("name") if isinstance(first_artist, dict) else str(first_artist)

        if name and artist and str(artist).strip().casefold() not in str(name).casefold():
            return f"{artist} - {name}"
        if name:
            return str(name)
        return None

    def _friendly_names_for_entities(self, entity_ids: List[str]) -> List[str]:
        """Resolve entity IDs to friendly names."""
        names = []
        for entity_id in entity_ids:
            state = self.hass.states.get(entity_id)
            if state and state.name:
                names.append(state.name)
            else:
                names.append(entity_id)
        return names

    def validate_service(self, domain: str, action: str) -> str:
        """Validate that a domain/action combination is allowed.

        Returns:
            The correct service name to use

        Raises:
            ValueError: If domain or action is invalid
        """
        capability_error = self._get_domain_capability_error(domain)
        if capability_error:
            raise ValueError(capability_error)

        valid, result = validate_domain_action(domain, action)
        if valid:
            _LOGGER.debug(
                f"Validated service: {domain}.{result} (from action: {action})"
            )
            return result  # Returns the correct service name
        else:
            _LOGGER.warning(f"Service validation failed: {result}")
            raise ValueError(result)  # Returns error message

    def _prepare_response_service_data(
        self,
        domain: str,
        service: str,
        data: Dict[str, Any] | None,
        *,
        resolved_target: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Normalize service-response data and fill safe defaults."""
        prepared = dict(data or {})

        if (
            domain == "weather"
            and service in {"get_forecast", "get_forecasts"}
        ):
            requested_type = str(prepared.get("type") or "").strip().lower() or None
            supported_types = self._get_weather_target_forecast_types(
                resolved_target
            )

            chosen_type = requested_type
            if supported_types:
                if requested_type not in supported_types:
                    chosen_type = self._select_preferred_weather_forecast_type(
                        supported_types
                    )
                    if requested_type and chosen_type and chosen_type != requested_type:
                        entity_ids = self._normalize_target_values(
                            (resolved_target or {}).get("entity_id")
                        )
                        _LOGGER.info(
                            "Weather forecast type '%s' is not supported by target %s; "
                            "using '%s' instead from supported types %s",
                            requested_type,
                            entity_ids or "entities",
                            chosen_type,
                            supported_types,
                        )

            if chosen_type:
                prepared["type"] = chosen_type

        return prepared

    async def _resolve_weather_forecast_target(
        self,
        *,
        entity_id: str | None = None,
        area: str | None = None,
        floor: str | None = None,
        label: str | None = None,
        name_contains: str | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, str]]:
        """Resolve the best weather entity target for a forecast request."""
        if entity_id:
            resolved_target = await self.resolve_target({"entity_id": entity_id})
            resolved_target = self._restrict_resolved_target_to_domain(
                resolved_target, "weather"
            )
            selected_entity_id = self._normalize_target_values(
                resolved_target.get("entity_id")
            )[0]
            state = self.hass.states.get(selected_entity_id)
            friendly_name = (
                state.attributes.get("friendly_name")
                if state and state.attributes.get("friendly_name")
                else selected_entity_id
            )
            return resolved_target, {
                "entity_id": selected_entity_id,
                "name": str(friendly_name),
            }

        candidates = await self.discovery.discover_entities(
            domain="weather",
            area=area,
            floor=floor,
            label=label,
            name_contains=name_contains,
            limit=10,
        )
        if not candidates:
            raise ValueError(
                "No exposed weather entity was found in Home Assistant. "
                "Try discover_entities(domain='weather') to inspect available weather entities."
            )

        selected = sorted(
            candidates,
            key=self._weather_entity_candidate_sort_key,
        )[0]
        return {"entity_id": [selected["entity_id"]]}, {
            "entity_id": str(selected["entity_id"]),
            "name": str(selected.get("name") or selected["entity_id"]),
        }

    async def _resolve_calendar_event_targets(
        self,
        *,
        entity_id: str | None = None,
        area: str | None = None,
        floor: str | None = None,
        label: str | None = None,
        query: str | None = None,
        name_contains: str | None = None,
    ) -> tuple[Dict[str, Any], list[Dict[str, str]], bool]:
        """Resolve one or more calendar targets for a search."""
        if entity_id:
            resolved_target = await self.resolve_target({"entity_id": entity_id})
            resolved_target = self._restrict_resolved_target_to_domain(
                resolved_target, "calendar"
            )
            entity_ids = self._normalize_target_values(resolved_target.get("entity_id"))
            selected = [
                {
                    "entity_id": candidate_entity_id,
                    "name": self._friendly_entity_name(candidate_entity_id),
                }
                for candidate_entity_id in entity_ids
            ]
            return resolved_target, selected, False

        explicit_name_filter = bool(name_contains)
        candidates: list[Dict[str, Any]] = []
        name_filters = self._build_calendar_name_filters(
            query=query,
            name_contains=name_contains,
        )
        for discovery_name in name_filters:
            candidates = await self.discovery.discover_entities(
                domain="calendar",
                area=area,
                floor=floor,
                label=label,
                name_contains=discovery_name,
                limit=25,
            )
            if candidates:
                break
        fallback_used = False
        if not candidates and query and not explicit_name_filter:
            candidates = await self.discovery.discover_entities(
                domain="calendar",
                area=area,
                floor=floor,
                label=label,
                limit=25,
            )
            fallback_used = True

        if not candidates:
            if explicit_name_filter:
                raise ValueError(
                    "No exposed calendar matched that name. "
                    "Try discover_entities(domain='calendar') to inspect available calendars."
                )
            raise ValueError(
                "No exposed calendar was found in Home Assistant. "
                "Try discover_entities(domain='calendar') to inspect available calendars."
            )

        selected = [
            {
                "entity_id": str(candidate["entity_id"]),
                "name": str(candidate.get("name") or candidate["entity_id"]),
            }
            for candidate in candidates
        ]
        selected.sort(key=lambda item: item["name"].casefold())
        return {"entity_id": [item["entity_id"] for item in selected]}, selected, fallback_used

    def _build_calendar_name_filters(
        self,
        *,
        query: str | None,
        name_contains: str | None,
    ) -> list[str | None]:
        """Build progressively broader calendar-name filters before event-text fallback."""
        explicit_name = str(name_contains or "").strip()
        if explicit_name:
            return [explicit_name]

        raw_query = str(query or "").strip()
        if not raw_query:
            return [None]

        filters: list[str | None] = [raw_query]
        generic_words = {
            "calendar",
            "event",
            "events",
            "game",
            "games",
            "match",
            "matches",
            "schedule",
            "schedules",
            "next",
            "upcoming",
        }
        query_tokens = re.findall(r"[a-z0-9']+", raw_query.casefold())
        simplified_tokens = [token for token in query_tokens if token not in generic_words]
        simplified_query = " ".join(simplified_tokens).strip()
        if simplified_query and simplified_query != raw_query.casefold():
            filters.append(simplified_query)

        return filters

    @staticmethod
    def _weather_entity_candidate_sort_key(entity: Dict[str, Any]) -> Tuple[int, int, str, str]:
        """Rank weather entity candidates with stable generic preferences."""
        name = str(entity.get("name") or "").casefold()
        entity_id = str(entity.get("entity_id") or "").casefold()
        service_hint = 0 if entity.get("forecast_service_supported") else 1
        attribute_hint = 0 if entity.get("forecast_available") else 1
        return (service_hint, attribute_hint, name or entity_id, entity_id)

    def _parse_weather_forecast_when(
        self, when_value: str | None
    ) -> tuple[date | None, str, str | None]:
        """Parse a weather forecast day request."""
        now_local = dt_util.as_local(dt_util.now())
        when_text = str(when_value or "tomorrow").strip().casefold()
        if not when_text or when_text == "tomorrow":
            return now_local.date() + timedelta(days=1), "tomorrow", None
        if when_text == "today":
            return now_local.date(), "today", None

        parsed_date = dt_util.parse_date(when_text)
        if parsed_date is not None:
            return parsed_date, self._format_weather_day_label(parsed_date), None

        parsed_datetime = dt_util.parse_datetime(when_text)
        if parsed_datetime is not None:
            if parsed_datetime.tzinfo is None:
                parsed_datetime = parsed_datetime.replace(
                    tzinfo=getattr(dt_util, "DEFAULT_TIME_ZONE", now_local.tzinfo)
                )
            local_when = dt_util.as_local(parsed_datetime)
            return local_when.date(), self._format_weather_day_label(local_when.date()), None

        return None, when_text, (
            "Unable to parse 'when'. Use 'today', 'tomorrow', or a local date like '2026-04-13'."
        )

    def _format_weather_day_label(self, target_date: date) -> str:
        """Format a forecast day label for user-facing summaries."""
        today = dt_util.as_local(dt_util.now()).date()
        if target_date == today:
            return "today"
        if target_date == today + timedelta(days=1):
            return "tomorrow"
        return target_date.strftime("%A, %b %d").replace(" 0", " ")

    def _extract_weather_entries_for_date(
        self,
        forecast_entries: List[Dict[str, Any]],
        *,
        target_date: date,
        forecast_type: str | None,
    ) -> List[Dict[str, Any]]:
        """Return forecast entries that map to the requested local date."""
        matching: List[tuple[datetime | None, Dict[str, Any]]] = []
        for entry in forecast_entries:
            entry_date = self._parse_weather_forecast_entry_date(entry)
            if entry_date and entry_date == target_date:
                matching.append((self._parse_weather_forecast_entry_datetime(entry), entry))

        if matching:
            matching.sort(key=lambda item: item[0] or datetime.min.replace(tzinfo=dt_util.UTC))
            return [entry for _, entry in matching]

        today = dt_util.as_local(dt_util.now()).date()
        day_offset = (target_date - today).days
        if day_offset < 0:
            return []

        if forecast_type == "daily" and day_offset < len(forecast_entries):
            return [forecast_entries[day_offset]]

        if forecast_type == "twice_daily":
            start = day_offset * 2
            if start < len(forecast_entries):
                return forecast_entries[start : start + 2]

        if forecast_type == "hourly":
            start = day_offset * 24
            if start < len(forecast_entries):
                return forecast_entries[start : start + 24]

        return []

    def _parse_weather_forecast_entry_datetime(
        self, entry: Dict[str, Any]
    ) -> datetime | None:
        """Parse a weather forecast entry datetime to local time."""
        raw_value = entry.get("datetime")
        if raw_value is None:
            raw_value = entry.get("date")
        if raw_value is None:
            return None

        parsed_datetime = dt_util.parse_datetime(str(raw_value))
        if parsed_datetime is not None:
            if parsed_datetime.tzinfo is None:
                parsed_datetime = parsed_datetime.replace(
                    tzinfo=getattr(dt_util, "DEFAULT_TIME_ZONE", dt_util.now().tzinfo)
                )
            return dt_util.as_local(parsed_datetime)

        parsed_date = dt_util.parse_date(str(raw_value))
        if parsed_date is None:
            return None

        local_tz = getattr(dt_util, "DEFAULT_TIME_ZONE", dt_util.now().tzinfo)
        return datetime.combine(parsed_date, time.min, tzinfo=local_tz)

    def _parse_weather_forecast_entry_date(
        self, entry: Dict[str, Any]
    ) -> date | None:
        """Parse just the local date for a weather forecast entry."""
        parsed_datetime = self._parse_weather_forecast_entry_datetime(entry)
        if parsed_datetime is not None:
            return parsed_datetime.date()
        return None

    def _summarize_requested_weather_forecast(
        self,
        *,
        entity_name: str,
        entity_id: str,
        forecast_entries: List[Dict[str, Any]],
        forecast_type: str | None,
        target_date: date,
        day_label: str,
    ) -> str | None:
        """Summarize a requested weather forecast day from returned forecast data."""
        day_entries = self._extract_weather_entries_for_date(
            forecast_entries,
            target_date=target_date,
            forecast_type=forecast_type,
        )
        if not day_entries:
            return None

        intro = f"{day_label.capitalize()} for {entity_name}:"
        if forecast_type == "twice_daily":
            parts = []
            for entry in day_entries[:2]:
                part_label = self._describe_weather_forecast_part(entry)
                parts.append(f"{part_label}: {self._format_weather_forecast_entry(entry)}")
            return f"{intro} {'; '.join(parts)}."

        if forecast_type == "hourly" and len(day_entries) > 2:
            conditions = [
                str(entry.get("condition")).replace("_", " ")
                for entry in day_entries
                if entry.get("condition")
            ]
            temps = [
                self._coerce_weather_temperature(entry.get("temperature"))
                for entry in day_entries
            ]
            temps = [temp for temp in temps if temp is not None]
            dominant_condition = None
            if conditions:
                dominant_condition = Counter(conditions).most_common(1)[0][0]

            detail_parts = []
            if dominant_condition:
                detail_parts.append(dominant_condition)
            if temps:
                detail_parts.append(
                    f"high {self._format_weather_temperature(max(temps))}, low {self._format_weather_temperature(min(temps))}"
                )
            preview_entries = [
                f"{self._describe_weather_forecast_part(entry)} {self._format_weather_forecast_entry(entry)}"
                for entry in day_entries[:3]
            ]
            if preview_entries:
                detail_parts.append("early outlook: " + "; ".join(preview_entries))
            return f"{intro} {'. '.join(detail_parts)}."

        return f"{intro} {self._format_weather_forecast_entry(day_entries[0])}."

    def _build_calendar_search_window(
        self,
        *,
        when_value: str | None,
        days: int,
    ) -> tuple[datetime | None, datetime | None, str]:
        """Build the UTC search window for a calendar query."""
        local_tz = getattr(dt_util, "DEFAULT_TIME_ZONE", dt_util.now().tzinfo)
        now_local = dt_util.as_local(dt_util.utcnow())
        raw_when = (when_value or "").strip()
        normalized = raw_when.casefold()

        if not raw_when or normalized == "now":
            window_start = dt_util.utcnow()
            window_end = window_start + timedelta(days=days)
            return window_start, window_end, f"the next {days} day{'s' if days != 1 else ''}"

        if normalized == "today":
            start_local = datetime.combine(now_local.date(), time.min, tzinfo=local_tz)
            end_local = start_local + timedelta(days=1)
            return dt_util.as_utc(start_local), dt_util.as_utc(end_local), "today"

        if normalized == "tomorrow":
            start_local = datetime.combine(
                now_local.date() + timedelta(days=1),
                time.min,
                tzinfo=local_tz,
            )
            end_local = start_local + timedelta(days=1)
            return dt_util.as_utc(start_local), dt_util.as_utc(end_local), "tomorrow"

        parsed_date = dt_util.parse_date(raw_when)
        if parsed_date is not None:
            start_local = datetime.combine(parsed_date, time.min, tzinfo=local_tz)
            end_local = start_local + timedelta(days=1)
            return dt_util.as_utc(start_local), dt_util.as_utc(end_local), raw_when

        parsed_datetime = dt_util.parse_datetime(raw_when)
        if parsed_datetime is None:
            return None, None, raw_when
        if parsed_datetime.tzinfo is None:
            parsed_datetime = parsed_datetime.replace(tzinfo=local_tz)
        window_start = dt_util.as_utc(parsed_datetime)
        window_end = window_start + timedelta(days=days)
        return window_start, window_end, raw_when

    def _collect_calendar_event_matches(
        self,
        *,
        response: Dict[str, Any],
        selected_calendars: list[Dict[str, str]],
        event_text: str | None,
    ) -> list[Dict[str, Any]]:
        """Collect matching calendar events sorted by upcoming start time."""
        selected_by_entity = {
            item["entity_id"]: item["name"] for item in selected_calendars
        }
        matches: list[Dict[str, Any]] = []
        for entity_id, payload in response.items():
            events = payload.get("events") if isinstance(payload, dict) else None
            if not isinstance(events, list):
                continue
            calendar_name = selected_by_entity.get(entity_id, entity_id)
            for event in events:
                if not isinstance(event, dict):
                    continue
                if event_text and not self._calendar_event_matches_text(event, event_text):
                    continue
                start_at = self._parse_calendar_event_start_datetime(event)
                if start_at is None:
                    continue
                matches.append(
                    {
                        "calendar_entity_id": entity_id,
                        "calendar_name": calendar_name,
                        "summary": str(
                            event.get("summary")
                            or event.get("title")
                            or "Untitled event"
                        ),
                        "description": str(event.get("description") or "").strip(),
                        "location": str(event.get("location") or "").strip(),
                        "start": self._serialize_service_response_value(
                            self._extract_calendar_event_start_value(event)
                        ),
                        "end": self._serialize_service_response_value(event.get("end")),
                        "all_day": self._calendar_event_is_all_day(event),
                        "start_at": start_at,
                    }
                )
        matches.sort(key=lambda item: item["start_at"])
        return matches

    def _summarize_calendar_matches(
        self,
        *,
        matches: list[Dict[str, Any]],
        query: str | None,
        event_text: str | None,
        window_label: str,
    ) -> str:
        """Summarize one or more calendar event matches."""
        if len(matches) == 1:
            event = matches[0]
            qualifier = "matching calendar event" if (query or event_text) else "calendar event"
            return (
                f"Next {qualifier}: {self._describe_calendar_match(event)}."
            )

        lead = "Upcoming matching calendar events" if (query or event_text) else "Upcoming calendar events"
        lines = [f"{lead} in {window_label}:"]
        for event in matches:
            lines.append(f"- {self._describe_calendar_match(event)}")
        return "\n".join(lines)

    def _describe_calendar_match(self, event: Dict[str, Any]) -> str:
        """Describe one calendar event compactly."""
        summary = str(event.get("summary") or "Untitled event")
        calendar_name = str(event.get("calendar_name") or event.get("calendar_entity_id") or "")
        when = self._format_calendar_event_when(event)
        parts = [summary]
        if calendar_name and calendar_name.casefold() not in summary.casefold():
            parts.append(f"on {calendar_name}")
        if when:
            parts.append(when)
        location = str(event.get("location") or "").strip()
        if location:
            parts.append(f"at {location}")
        return ", ".join(parts)

    def _format_calendar_event_when(self, event: Dict[str, Any]) -> str:
        """Format a calendar event start in friendly upcoming language."""
        start_at = event.get("start_at")
        if not isinstance(start_at, datetime):
            return ""
        if event.get("all_day"):
            return self._format_calendar_all_day(start_at)
        return self._format_future_absolute_time(start_at)

    def _format_calendar_all_day(self, when: datetime) -> str:
        """Format an all-day calendar date."""
        local_when = dt_util.as_local(when)
        today_local = dt_util.as_local(dt_util.utcnow()).date()
        day_delta = (local_when.date() - today_local).days
        if day_delta == 0:
            return "all day today"
        if day_delta == 1:
            return "all day tomorrow"
        if day_delta > 1 and day_delta < 7:
            return f"all day in {day_delta} days"
        date_text = local_when.strftime("%b %d").replace(" 0", " ")
        if local_when.year != today_local.year:
            date_text += f", {local_when.year}"
        return f"all day on {date_text}"

    def _format_future_absolute_time(self, when: datetime) -> str:
        """Format a future timestamp with relative and absolute context."""
        now = dt_util.utcnow()
        delta_seconds = (when - now).total_seconds()
        absolute = self._format_absolute_time(when)
        if delta_seconds <= 0:
            return absolute
        if delta_seconds < 3600:
            minutes = max(1, int(delta_seconds / 60))
            return f"in {minutes} minute{'s' if minutes != 1 else ''} at {absolute}"
        if delta_seconds < 86400:
            hours = max(1, int(delta_seconds / 3600))
            return f"in {hours} hour{'s' if hours != 1 else ''} at {absolute}"
        if delta_seconds < 604800:
            days = max(1, int(delta_seconds / 86400))
            return f"in {days} day{'s' if days != 1 else ''} at {absolute}"
        weeks = max(1, int(delta_seconds / 604800))
        return f"in {weeks} week{'s' if weeks != 1 else ''} at {absolute}"

    def _calendar_event_matches_text(self, event: Dict[str, Any], text: str) -> bool:
        """Return whether a calendar event matches free text."""
        query = text.casefold().strip()
        if not query:
            return True
        haystack = " ".join(
            str(event.get(key) or "").casefold()
            for key in ("summary", "title", "description", "location")
        )
        return query in haystack

    def _calendar_event_is_all_day(self, event: Dict[str, Any]) -> bool:
        """Return whether a calendar event is all-day."""
        start_value = event.get("start")
        if isinstance(start_value, dict):
            return bool(start_value.get("date")) and not bool(
                start_value.get("dateTime") or start_value.get("datetime")
            )
        if isinstance(start_value, str):
            return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", start_value.strip()))
        return False

    def _extract_calendar_event_start_value(self, event: Dict[str, Any]) -> Any:
        """Extract the raw calendar event start field."""
        start_value = event.get("start")
        if isinstance(start_value, dict):
            return (
                start_value.get("dateTime")
                or start_value.get("datetime")
                or start_value.get("date")
            )
        return start_value

    def _parse_calendar_event_start_datetime(self, event: Dict[str, Any]) -> datetime | None:
        """Parse a calendar event start into UTC for sorting."""
        start_value = self._extract_calendar_event_start_value(event)
        local_tz = getattr(dt_util, "DEFAULT_TIME_ZONE", dt_util.now().tzinfo)
        if self._calendar_event_is_all_day(event):
            parsed_date = dt_util.parse_date(str(start_value))
            if parsed_date is None:
                return None
            return dt_util.as_utc(
                datetime.combine(parsed_date, time.min, tzinfo=local_tz)
            )

        parsed_datetime = dt_util.parse_datetime(str(start_value))
        if parsed_datetime is None:
            return None
        if parsed_datetime.tzinfo is None:
            parsed_datetime = parsed_datetime.replace(tzinfo=local_tz)
        return dt_util.as_utc(parsed_datetime)

    def _friendly_entity_name(self, entity_id: str) -> str:
        """Return a friendly entity name when available."""
        state = self.hass.states.get(entity_id)
        if state is not None and state.attributes.get("friendly_name"):
            return str(state.attributes["friendly_name"])
        return entity_id

    def _describe_weather_forecast_part(self, entry: Dict[str, Any]) -> str:
        """Describe a weather forecast segment like morning/evening/time."""
        parsed_datetime = self._parse_weather_forecast_entry_datetime(entry)
        if parsed_datetime is not None:
            hour = parsed_datetime.hour
            if entry.get("is_daytime") is False or hour >= 18:
                return "evening"
            if 5 <= hour < 12:
                return "morning"
            if 12 <= hour < 18:
                return "afternoon"
            return parsed_datetime.strftime("%-I:%M %p")

        if entry.get("is_daytime") is True:
            return "daytime"
        if entry.get("is_daytime") is False:
            return "night"
        return "forecast"

    def _format_weather_forecast_entry(self, entry: Dict[str, Any]) -> str:
        """Format a single weather forecast entry compactly."""
        parts = []
        condition = entry.get("condition")
        if condition:
            parts.append(str(condition).replace("_", " "))

        high_temp = self._coerce_weather_temperature(entry.get("temperature"))
        low_temp = self._coerce_weather_temperature(entry.get("templow"))
        if high_temp is not None and low_temp is not None:
            parts.append(
                f"high {self._format_weather_temperature(high_temp)}, low {self._format_weather_temperature(low_temp)}"
            )
        elif high_temp is not None:
            parts.append(f"around {self._format_weather_temperature(high_temp)}")

        precipitation_probability = entry.get("precipitation_probability")
        if precipitation_probability not in (None, ""):
            parts.append(f"{precipitation_probability}% chance of rain")

        wind_speed = entry.get("wind_speed")
        if wind_speed not in (None, ""):
            parts.append(f"wind {wind_speed}")

        return ", ".join(parts) if parts else "forecast available"

    @staticmethod
    def _coerce_weather_temperature(value: Any) -> float | int | None:
        """Coerce a weather temperature value safely."""
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return value
        try:
            numeric = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        return int(numeric) if numeric.is_integer() else round(numeric, 1)

    @staticmethod
    def _format_weather_temperature(value: float | int) -> str:
        """Format a compact temperature value."""
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value)

    def _get_weather_target_forecast_types(
        self, resolved_target: Dict[str, Any] | None
    ) -> List[str]:
        """Return forecast types supported by all targeted weather entities."""
        entity_ids = self._normalize_target_values((resolved_target or {}).get("entity_id"))
        if not entity_ids:
            return []

        common_types: set[str] | None = None
        for entity_id in entity_ids:
            state = self.hass.states.get(entity_id)
            if state is None or state.domain != "weather":
                continue

            forecast_types = set(self.discovery._get_weather_forecast_types(state))
            if not forecast_types:
                continue

            common_types = (
                forecast_types
                if common_types is None
                else common_types.intersection(forecast_types)
            )

        if not common_types:
            return []

        return [
            forecast_type
            for forecast_type in ("daily", "twice_daily", "hourly")
            if forecast_type in common_types
        ]

    def _select_preferred_weather_forecast_type(
        self, forecast_types: List[str] | set[str] | tuple[str, ...]
    ) -> str | None:
        """Choose a stable forecast-type fallback for weather responses."""
        forecast_type_set = {str(item) for item in forecast_types if item}
        for candidate in ("daily", "twice_daily", "hourly"):
            if candidate in forecast_type_set:
                return candidate
        return next(iter(sorted(forecast_type_set)), None)

    async def _get_response_service_catalog(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Build a catalog of HA services that support response data."""
        descriptions = await service_helper.async_get_all_descriptions(self.hass)
        catalog: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for domain, services in self.hass.services.async_services().items():
            if self._get_domain_capability_error(domain):
                continue

            for service_name in services:
                supports_response = self.hass.services.supports_response(
                    domain, service_name
                )
                if supports_response == SupportsResponse.NONE:
                    continue

                description = dict(descriptions.get(domain, {}).get(service_name, {}))
                description["supports_response"] = (
                    "optional"
                    if supports_response == SupportsResponse.OPTIONAL
                    else "only"
                )
                catalog.setdefault(domain, {})[service_name] = description

        return catalog

    async def _get_response_service_info(
        self, domain: str, service: str
    ) -> tuple[Dict[str, Any], str | None]:
        """Get dynamic metadata for a response-capable HA service."""
        capability_error = self._get_domain_capability_error(domain)
        if capability_error:
            return {}, capability_error

        if not self.hass.services.has_service(domain, service):
            domain_info = get_domain_info(domain)
            if domain_info is None:
                return {}, (
                    f"Domain '{domain}' is not registered in Home Assistant right now. "
                    "Use list_response_services() to inspect available response-capable services."
                )
            return {}, (
                f"Service '{domain}.{service}' is not registered in Home Assistant right now. "
                "Use list_response_services() to inspect available response-capable services."
            )

        supports_response = self.hass.services.supports_response(domain, service)
        if supports_response == SupportsResponse.NONE:
            return {}, (
                f"Service '{domain}.{service}' does not support native response data. "
                "Use list_response_services() to find services that do."
            )

        catalog = await self._get_response_service_catalog()
        description = dict(catalog.get(domain, {}).get(service, {}))
        description["supports_response"] = (
            "optional"
            if supports_response == SupportsResponse.OPTIONAL
            else "only"
        )
        return description, None

    def _validate_response_service_parameters(
        self, description: Dict[str, Any], provided_params: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Validate required service parameters from HA's native service description."""
        required_fields = self._get_required_service_fields(description)
        missing = [field for field in required_fields if field not in provided_params]
        if missing:
            return (
                False,
                "Missing required parameters: " + ", ".join(missing),
            )

        return True, "Parameters valid"

    def _get_required_service_fields(self, description: Dict[str, Any]) -> List[str]:
        """Extract required field names from a service description."""
        fields = description.get("fields", {})
        if not isinstance(fields, dict):
            return []

        required_fields = []
        for field_name, metadata in fields.items():
            if isinstance(metadata, dict) and metadata.get("required") is True:
                required_fields.append(str(field_name))

        return sorted(required_fields)

    def _get_service_field_names(self, description: Dict[str, Any]) -> List[str]:
        """Extract service field names from a service description."""
        fields = description.get("fields", {})
        if not isinstance(fields, dict):
            return []

        return sorted(str(field_name) for field_name in fields.keys())

    def _extract_service_target_domains(
        self, description: Dict[str, Any]
    ) -> List[str]:
        """Extract allowed target entity domains from a service description."""
        target = description.get("target")
        if not isinstance(target, dict):
            return []

        entity_target = target.get("entity")
        if not isinstance(entity_target, dict):
            return []

        domain_value = entity_target.get("domain")
        if domain_value is None:
            return []
        if isinstance(domain_value, str):
            return [domain_value]
        if isinstance(domain_value, (list, tuple, set)):
            return [str(item) for item in domain_value if item]
        return [str(domain_value)]

    def _restrict_resolved_target_for_service(
        self,
        resolved_target: Dict[str, Any],
        *,
        default_domain: str | None = None,
        service_description: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Restrict resolved entity targets using known service target metadata."""
        allowed_domains = []
        if service_description:
            allowed_domains = self._extract_service_target_domains(service_description)
        if not allowed_domains and default_domain:
            allowed_domains = [default_domain]

        entity_ids = self._normalize_target_values(resolved_target.get("entity_id"))
        if not allowed_domains:
            resolved_domains = {entity_id.split(".", 1)[0] for entity_id in entity_ids}
            if len(resolved_domains) > 1:
                raise ValueError(
                    "Resolved target spans multiple domains, and this service does not "
                    "publish target-domain metadata. Use explicit entity_id values from discovery."
                )
            return {"entity_id": entity_ids}

        filtered_entity_ids = [
            entity_id
            for entity_id in entity_ids
            if entity_id.split(".", 1)[0] in allowed_domains
        ]

        if not filtered_entity_ids:
            raise ValueError(
                "Resolved target did not include any exposed entities accepted by this service."
            )

        return {"entity_id": filtered_entity_ids}

    def _restrict_resolved_target_to_domain(
        self, resolved_target: Dict[str, Any], domain: str
    ) -> Dict[str, Any]:
        """Restrict resolved entity targets to the requested domain."""
        return self._restrict_resolved_target_for_service(
            resolved_target, default_domain=domain
        )

    def _serialize_service_response_value(self, value: Any) -> Any:
        """Serialize HA service response data to JSON-safe values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        if isinstance(value, dict):
            return {
                str(key): self._serialize_service_response_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_service_response_value(item) for item in value]
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.hex()
        return str(value)

    def _format_service_response_result(
        self,
        domain: str,
        service: str,
        resolved_target: Dict[str, Any],
        response: Any,
        *,
        request_data: Dict[str, Any] | None = None,
    ) -> str:
        """Format a response-returning service call for the LLM."""
        entity_ids = self._normalize_target_values(resolved_target.get("entity_id"))
        if entity_ids:
            target_count = len(entity_ids)
            target_label = "entity" if target_count == 1 else "entities"
            header = (
                f"✅ Retrieved response from {domain}.{service} for "
                f"{target_count} target {target_label}."
            )
        else:
            header = f"✅ Retrieved response from {domain}.{service}."

        text_parts = [header]

        if domain == "weather" and service in {"get_forecast", "get_forecasts"}:
            forecast_type = str((request_data or {}).get("type") or "").strip()
            if forecast_type:
                text_parts.append("")
                text_parts.append(f"Forecast type used: {forecast_type}.")

        summary_lines = self._build_service_response_summary(domain, service, response)
        if summary_lines:
            text_parts.append("")
            text_parts.extend(summary_lines)

        if response is None:
            text_parts.append("")
            text_parts.append("No response data was returned.")
        else:
            text_parts.append("")
            text_parts.append("Response:")
            text_parts.append(json.dumps(response, indent=2))

        return "\n".join(text_parts)

    def _build_service_response_summary(
        self, domain: str, service: str, response: Any
    ) -> List[str]:
        """Build a concise summary for structured service responses."""
        if domain == "music_assistant":
            return self._summarize_music_assistant_response(service, response)
        if domain == "weather" and service in {"get_forecast", "get_forecasts"}:
            return self._summarize_weather_response(response)
        if domain == "calendar" and service == "get_events":
            return self._summarize_calendar_response(response)

        if isinstance(response, dict):
            return [f"Summary: {len(response)} top-level response entries."]
        if isinstance(response, list):
            return [f"Summary: {len(response)} response items."]
        return []

    def _summarize_weather_response(self, response: Any) -> List[str]:
        """Summarize weather forecast response data."""
        if not isinstance(response, dict):
            return []

        lines = ["Summary:"]
        for entity_id, payload in response.items():
            forecast = payload.get("forecast") if isinstance(payload, dict) else None
            if not isinstance(forecast, list):
                lines.append(f"- {entity_id}: no forecast entries returned")
                continue

            detail_parts = [f"{len(forecast)} forecast entries"]
            if forecast:
                first_forecast = forecast[0] if isinstance(forecast[0], dict) else {}
                preview_parts = []
                preview_time = self._format_service_response_datetime(
                    first_forecast.get("datetime")
                )
                if preview_time:
                    preview_parts.append(preview_time)
                if first_forecast.get("condition"):
                    preview_parts.append(str(first_forecast["condition"]))
                high_temp = first_forecast.get("temperature")
                low_temp = first_forecast.get("templow")
                if high_temp is not None and low_temp is not None:
                    preview_parts.append(f"{high_temp}/{low_temp}")
                elif high_temp is not None:
                    preview_parts.append(str(high_temp))
                if preview_parts:
                    detail_parts.append("first: " + ", ".join(preview_parts))

            lines.append(f"- {entity_id}: {'; '.join(detail_parts)}")

        return lines if len(lines) > 1 else []

    def _summarize_calendar_response(self, response: Any) -> List[str]:
        """Summarize calendar event response data."""
        if not isinstance(response, dict):
            return []

        lines = ["Summary:"]
        for entity_id, payload in response.items():
            events = payload.get("events") if isinstance(payload, dict) else None
            if not isinstance(events, list):
                lines.append(f"- {entity_id}: no events returned")
                continue

            detail_parts = [f"{len(events)} events"]
            if events:
                first_event = events[0] if isinstance(events[0], dict) else {}
                event_summary = first_event.get("summary") or first_event.get("title")
                event_start = self._extract_calendar_event_start(first_event)
                preview_parts = []
                if event_summary:
                    preview_parts.append(str(event_summary))
                if event_start:
                    preview_parts.append(event_start)
                if preview_parts:
                    detail_parts.append("next: " + " at ".join(preview_parts[:2]))

            lines.append(f"- {entity_id}: {'; '.join(detail_parts)}")

        return lines if len(lines) > 1 else []

    def _extract_calendar_event_start(self, event: Dict[str, Any]) -> str | None:
        """Extract and format the start time from a calendar event payload."""
        if not isinstance(event, dict):
            return None

        start_value = event.get("start")
        if isinstance(start_value, dict):
            start_value = (
                start_value.get("dateTime")
                or start_value.get("datetime")
                or start_value.get("date")
            )

        return self._format_service_response_datetime(start_value)

    def _format_service_response_datetime(self, value: Any) -> str | None:
        """Format a service-response date/time value in local time when possible."""
        if value is None:
            return None

        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, date):
            return value.isoformat()
        else:
            parsed = dt_util.parse_datetime(str(value))
            if parsed is None:
                return str(value)

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=getattr(dt_util, "DEFAULT_TIME_ZONE", dt_util.now().tzinfo)
            )

        return self._format_absolute_time(dt_util.as_utc(parsed))

    async def resolve_target(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve target selectors to exposed entity IDs."""
        explicit_entity_ids = self._normalize_target_values(target.get("entity_id"))
        selector_values = {
            "area_id": self._normalize_target_values(target.get("area_id")),
            "floor_id": self._normalize_target_values(target.get("floor_id")),
            "label_id": self._normalize_target_values(target.get("label_id")),
            "device_id": self._normalize_target_values(target.get("device_id")),
        }
        active_selectors = {
            key: values for key, values in selector_values.items() if values
        }

        resolved_entities = set()
        invalid_entity_ids = []
        for entity_id in explicit_entity_ids:
            state = self.hass.states.get(entity_id)
            if state is None:
                invalid_entity_ids.append(f"{entity_id} (not found)")
                continue
            if not async_should_expose(self.hass, "conversation", entity_id):
                invalid_entity_ids.append(f"{entity_id} (not exposed to conversation)")
                continue
            resolved_entities.add(entity_id)

        if invalid_entity_ids:
            raise ValueError(
                "Invalid entity targets: " + ", ".join(invalid_entity_ids)
            )

        if active_selectors:
            selector_matches = self._find_exposed_entities_for_target(active_selectors)
            selector_sets = []

            for selector_key, selector_ids in active_selectors.items():
                matched_entities = selector_matches.get(selector_key, set())
                if not matched_entities:
                    raise ValueError(
                        "No exposed conversation entities matched "
                        f"{selector_key}: {', '.join(selector_ids)}"
                    )
                selector_sets.append(matched_entities)

            combined_matches = set.intersection(*selector_sets)
            if not combined_matches:
                raise ValueError(
                    "No exposed conversation entities matched the combined target selectors."
                )

            resolved_entities.update(combined_matches)

        if not resolved_entities:
            raise ValueError(
                "Target did not resolve to any exposed entities. Use discover_entities first."
            )

        resolved_target = {"entity_id": sorted(resolved_entities)}
        _LOGGER.debug("Resolved target %s to entity_ids: %s", target, resolved_target["entity_id"])
        return resolved_target

    @staticmethod
    def _normalize_target_values(value: Any) -> List[str]:
        """Normalize scalar or list target selector values to unique strings."""
        if value is None:
            return []

        if isinstance(value, str):
            raw_values = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raw_values = [value]

        normalized = []
        seen = set()
        for item in raw_values:
            if item is None:
                continue
            item_text = str(item).strip()
            if not item_text or item_text in seen:
                continue
            seen.add(item_text)
            normalized.append(item_text)

        return normalized

    def _find_exposed_entities_for_target(
        self, selectors: Dict[str, List[str]]
    ) -> Dict[str, set[str]]:
        """Resolve area, floor, label, and device selectors to exposed entities."""
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        area_registry = ar.async_get(self.hass)
        selector_sets = {key: set(values) for key, values in selectors.items() if values}

        area_floor_ids = {}
        area_label_ids = {}
        for area_entry in area_registry.async_list_areas():
            area_floor_ids[area_entry.id] = getattr(area_entry, "floor_id", None)
            area_label_ids[area_entry.id] = set(getattr(area_entry, "labels", set()) or set())

        matches = {
            "area_id": set(),
            "floor_id": set(),
            "label_id": set(),
            "device_id": set(),
        }

        for state_obj in self.hass.states.async_all():
            entity_id = state_obj.entity_id
            if not async_should_expose(self.hass, "conversation", entity_id):
                continue

            entity_entry = entity_registry.async_get(entity_id)
            device_entry = (
                device_registry.async_get(entity_entry.device_id)
                if entity_entry and entity_entry.device_id
                else None
            )
            area_id = None
            if entity_entry and entity_entry.area_id:
                area_id = entity_entry.area_id
            elif device_entry and device_entry.area_id:
                area_id = device_entry.area_id

            floor_id = area_floor_ids.get(area_id)

            label_ids = set(getattr(entity_entry, "labels", set()) or set())
            if device_entry:
                label_ids.update(getattr(device_entry, "labels", set()) or set())
            if area_id:
                label_ids.update(area_label_ids.get(area_id, set()))

            if selector_sets.get("area_id") and area_id in selector_sets["area_id"]:
                matches["area_id"].add(entity_id)
            if selector_sets.get("floor_id") and floor_id in selector_sets["floor_id"]:
                matches["floor_id"].add(entity_id)
            if selector_sets.get("label_id") and label_ids.intersection(selector_sets["label_id"]):
                matches["label_id"].add(entity_id)
            if (
                selector_sets.get("device_id")
                and entity_entry
                and entity_entry.device_id in selector_sets["device_id"]
            ):
                matches["device_id"].add(entity_id)

        return matches
