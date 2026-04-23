"""MCP Server for Home Assistant entity discovery."""

import asyncio
import base64
from collections import defaultdict
import ipaddress
import json
import logging
import mimetypes
from pathlib import Path, PurePosixPath
import re
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
from datetime import timedelta

import aiohttp
from aiohttp import web, WSMsgType
from aiohttp.web_ws import WebSocketResponse
import voluptuous as vol
from voluptuous_openapi import convert
import yarl

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    llm,
    network as network_helper,
)
from homeassistant.components.homeassistant import async_should_expose
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
    get_builtin_toggle_spec_by_package_id,
    is_builtin_package_enabled_for_shared_settings,
)
from .const import (
    DOMAIN,
    MCP_SERVER_NAME,
    MAX_ENTITIES_PER_DISCOVERY,
    CONF_LMSTUDIO_URL,
    CONF_ALLOWED_IPS,
    CONF_SEARCH_PROVIDER,
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
    DEFAULT_LMSTUDIO_URL,
    DEFAULT_ALLOWED_IPS,
    DEFAULT_SEARCH_PROVIDER,
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
    SERVER_TYPE_OLLAMA,
    TOOL_FAMILY_SHARED_SETTINGS,
    get_optional_tool_family,
)
from .discovery import EntityDiscovery
from .domain_registry import (
    validate_domain_action,
    validate_service_parameters,
    get_supported_domains,
    get_domains_by_type,
    TYPE_CONTROLLABLE,
    TYPE_READ_ONLY,
)
from .memory_manager import MemoryManager
from .provider_runtime import (
    build_provider_auth_headers,
    resolve_provider_runtime_config,
)
from .server_tools.calendar import CalendarToolsMixin
from .server_tools.recorder import RecorderToolsMixin
from .server_tools.response_services import ResponseServicesMixin
from .server_tools.weather import WeatherToolsMixin

_LOGGER = logging.getLogger(__name__)

_MAX_INLINE_IMAGE_BYTES = 6 * 1024 * 1024
_HTTP_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_MAX_IMAGE_FETCH_REDIRECTS = 5


class MCPServer(
    CalendarToolsMixin,
    RecorderToolsMixin,
    ResponseServicesMixin,
    WeatherToolsMixin,
):
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
        built_in_spec = get_builtin_toggle_spec_by_package_id(
            "weather_forecast",
            self._get_builtin_toggle_specs(),
        )
        if built_in_spec is not None:
            return self._is_builtin_package_enabled(built_in_spec)
        return bool(
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
        built_in_spec = get_builtin_toggle_spec_by_package_id(
            "response_service",
            self._get_builtin_toggle_specs(),
        )
        if built_in_spec is not None:
            return self._is_builtin_package_enabled(built_in_spec)
        return bool(
            self._get_shared_setting(
                CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
                DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
            )
        )

    def _recorder_tools_enabled(self) -> bool:
        """Return whether recorder/history tools are enabled."""
        built_in_spec = get_builtin_toggle_spec_by_package_id(
            "recorder",
            self._get_builtin_toggle_specs(),
        )
        if built_in_spec is not None:
            return self._is_builtin_package_enabled(built_in_spec)
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
                "llmDescription": "Analyze an image or camera snapshot with the active multimodal model.",
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
                                "Other entities may work when they expose a Home Assistant-local or "
                                "allowlisted picture URL."
                            ),
                        },
                        "image_url": {
                            "type": "string",
                            "description": (
                                "Image URL to analyze. Supports data URLs, Home Assistant-local URLs, "
                                "/local/... URLs, /media/local/... URLs, and remote http(s) URLs only "
                                "when they are allowlisted in Home Assistant."
                            ),
                        },
                        "image_path": {
                            "type": "string",
                            "description": (
                                "Local image path relative to the Home Assistant config directory, "
                                "or an absolute path already inside that directory."
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
                "llmDescription": "Fetch an image from a camera, entity, URL, or local file.",
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
                            "description": (
                                "Image URL. Supports data URLs, Home Assistant-local URLs, "
                                "/local/... URLs, /media/local/... URLs, and remote http(s) URLs only "
                                "when they are allowlisted in Home Assistant."
                            ),
                        },
                        "image_path": {
                            "type": "string",
                            "description": (
                                "Local image path relative to the Home Assistant config directory, "
                                "or an absolute path already inside that directory."
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
                "llmDescription": "Generate an image with the active provider when supported.",
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
                "llmDescription": "Find Home Assistant entities by area, type, state, or name.",
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
                "llmDescription": "Find Home Assistant devices by area, domain, maker, model, or name.",
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
                "llmDescription": "Get full details for specific entities.",
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
                "llmDescription": "Get full details and attached entities for specific devices.",
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
                "description": "Get the pre-generated system structure index. This index provides a lightweight overview of the Home Assistant system including areas, floors, labels, devices, domains, device classes, people, pets, calendars, zones, automations, scripts, and aliases for alias-capable objects. Use only when a broad system overview is needed; do not call by default at the start of every conversation. Prefer discover_entities or discover_devices for specific lookups.",
                "llmDescription": "Get a compact overview of areas, labels, devices, domains, and aliases.",
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
                "llmDescription": "List native Home Assistant Assist tools.",
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
                "llmDescription": "Call a native Home Assistant Assist tool directly.",
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
                "llmDescription": "Get the native Home Assistant Assist prompt text.",
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
                "llmDescription": "Get a native Assist live context snapshot.",
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
                "llmDescription": "Call Home Assistant services to control entities or run write actions.",
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
        elif tool_name == "set_conversation_state":
            return await self.tool_set_conversation_state(arguments)
        elif tool_name == "run_script":
            return await self.tool_run_script(arguments)
        elif tool_name == "run_automation":
            return await self.tool_run_automation(arguments)
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

    def _get_model_provider_config(
        self, context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Resolve the current profile's model/provider config for media tools."""
        entry = self._resolve_profile_entry(context)
        runtime_config = resolve_provider_runtime_config(entry)

        return {
            "entry": entry,
            "server_type": runtime_config.server_type,
            "model_name": runtime_config.model_name,
            "api_key": runtime_config.api_key,
            "timeout": runtime_config.timeout,
            "base_url": runtime_config.base_url,
        }

    def _get_model_auth_headers(self, provider_config: dict[str, Any]) -> dict[str, str]:
        """Build provider auth headers using the same rules as the conversation agent."""
        return build_provider_auth_headers(
            str(provider_config.get("server_type") or ""),
            str(provider_config.get("api_key") or ""),
        )

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
        """Fetch image bytes from a supported image reference."""
        reference = str(reference or "").strip()
        if not reference:
            raise ValueError("Image reference is required.")

        if reference.startswith("data:"):
            return self._parse_data_url_image(reference)
        if reference.startswith("/"):
            if reference.startswith("/local/") or reference.startswith("/media/local/"):
                local_path = self._resolve_local_image_path(reference)
                return self._read_local_image_path(local_path)
            return await self._fetch_http_image_url(reference)
        if reference.startswith(("http://", "https://")):
            return await self._fetch_http_image_url(reference)
        if "://" in reference:
            raise ValueError(
                "Only data URLs, Home Assistant-local URLs, local image paths, and http(s) image URLs are supported."
            )

        local_path = self._resolve_local_image_path(reference)
        return self._read_local_image_path(local_path)

    async def _fetch_http_image_url(self, reference: str) -> tuple[bytes, str]:
        """Fetch an image from an allowed HTTP(S) URL, validating redirects."""
        current_base_url, current_target_url = self._resolve_fetchable_http_request_target(
            reference
        )
        timeout = aiohttp.ClientTimeout(total=20)
        for _redirect_count in range(_MAX_IMAGE_FETCH_REDIRECTS + 1):
            current_url = self._build_safe_http_request_url(
                current_base_url, current_target_url
            )
            request_path = self._build_safe_http_request_path(current_target_url)
            async with aiohttp.ClientSession(
                timeout=timeout,
                base_url=str(current_base_url),
            ) as session:
                async with session.get(request_path, allow_redirects=False) as response:
                    if response.status in _HTTP_REDIRECT_STATUSES:
                        location = str(response.headers.get("Location") or "").strip()
                        if not location:
                            raise ValueError(
                                f"Image URL {current_url!s} redirected without a Location header."
                            )
                        redirect_url = current_url.join(yarl.URL(location))
                        (
                            current_base_url,
                            current_target_url,
                        ) = self._resolve_fetchable_http_request_target(str(redirect_url))
                        continue

                    if response.status != 200:
                        raise ValueError(
                            f"Unable to fetch image URL {current_url!s}: HTTP {response.status}"
                        )

                    image_bytes = await response.read()
                    mime_type = response.headers.get("Content-Type")
                    return self._normalize_image_payload(
                        image_bytes,
                        mime_type,
                        str(current_url),
                    )

        raise ValueError(f"Image URL {reference!r} redirected too many times.")

    def _resolve_fetchable_http_image_url(self, reference: str) -> yarl.URL:
        """Resolve a supported HTTP(S) image URL into a safe absolute URL."""
        trusted_base_url, target_url = self._resolve_fetchable_http_request_target(
            reference
        )
        return self._build_safe_http_request_url(trusted_base_url, target_url)

    def _resolve_fetchable_http_request_target(
        self, reference: str
    ) -> tuple[yarl.URL, yarl.URL]:
        """Resolve a supported HTTP(S) image URL into a trusted base URL and request target."""
        raw_reference = str(reference or "").strip()
        if not raw_reference:
            raise ValueError("Image URL is required.")

        if raw_reference.startswith("/"):
            return yarl.URL(self._get_hass_base_url()).origin(), yarl.URL(raw_reference)

        parsed_url = yarl.URL(raw_reference)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.host:
            raise ValueError(
                "HTTP image URLs must use http or https and include a host."
            )
        if parsed_url.user or parsed_url.password:
            raise ValueError("HTTP image URLs must not embed credentials.")

        parsed_url = parsed_url.with_fragment(None)
        normalized_url = str(parsed_url)

        if network_helper.is_hass_url(self.hass, normalized_url):
            return yarl.URL(self._get_hass_base_url()).origin(), parsed_url

        allowlisted_base = self._get_allowlisted_external_base_url(parsed_url)
        if allowlisted_base is not None:
            return allowlisted_base.origin(), parsed_url

        raise ValueError(
            "Remote image URLs must either point to this Home Assistant instance or be allowlisted in Home Assistant."
        )

    def _build_safe_http_request_url(
        self,
        trusted_base_url: yarl.URL,
        target_url: yarl.URL,
    ) -> yarl.URL:
        """Build a request URL using a trusted authority and a validated target path."""
        request_url = trusted_base_url.origin().with_path(
            self._sanitize_http_request_path(target_url.path or "/")
        )
        if target_url.query_string:
            request_url = request_url.with_query(target_url.query_string)
        return request_url.with_fragment(None)

    def _build_safe_http_request_path(self, target_url: yarl.URL) -> str:
        """Build a request path using only a validated target path and query."""
        request_path = self._sanitize_http_request_path(target_url.path or "/")
        if target_url.query_string:
            request_path = f"{request_path}?{target_url.query_string}"
        return request_path

    def _sanitize_http_request_path(self, path: str) -> str:
        """Normalize an HTTP request path so it cannot replace the trusted authority.

        aiohttp accepts both absolute URLs and relative request paths. Collapse any
        leading slash run to a single rooted path so inputs like ``//evil.test/x``
        cannot be interpreted as a network-path reference against the session base URL.
        """
        normalized_path = path or "/"
        if not normalized_path.startswith("/"):
            normalized_path = f"/{normalized_path}"
        return "/" + normalized_path.lstrip("/")

    def _get_allowlisted_external_base_url(
        self, parsed_url: yarl.URL
    ) -> yarl.URL | None:
        """Return the matching allowlisted external base URL for a target URL."""
        normalized_url = str(parsed_url.with_fragment(None))
        if not self.hass.config.is_allowed_external_url(normalized_url):
            return None

        matching_allowlisted_urls = sorted(
            (
                yarl.URL(str(allowed).strip())
                for allowed in self.hass.config.allowlist_external_urls
                if allowed
            ),
            key=lambda item: len(self._normalized_http_path(item.path or "/")),
            reverse=True,
        )
        for allowlisted_url in matching_allowlisted_urls:
            if allowlisted_url.scheme not in {"http", "https"} or not allowlisted_url.host:
                continue
            if not self._has_same_http_origin(allowlisted_url, parsed_url):
                continue
            if not self._is_http_path_within_base(
                allowlisted_url.path or "/",
                parsed_url.path or "/",
            ):
                continue
            return allowlisted_url
        return None

    def _has_same_http_origin(self, allowed_url: yarl.URL, target_url: yarl.URL) -> bool:
        """Return whether two HTTP(S) URLs share the same origin."""
        return (
            allowed_url.scheme == target_url.scheme
            and allowed_url.host == target_url.host
            and self._normalized_http_port(allowed_url)
            == self._normalized_http_port(target_url)
        )

    def _normalized_http_port(self, url: yarl.URL) -> int | None:
        """Return the effective port for an HTTP(S) URL."""
        if url.explicit_port is not None:
            return url.explicit_port
        if url.scheme == "http":
            return 80
        if url.scheme == "https":
            return 443
        return None

    def _normalized_http_path(self, path: str) -> str:
        """Return a normalized HTTP path for prefix checks."""
        normalized_path = self._sanitize_http_request_path(path or "/")
        if normalized_path != "/" and normalized_path.endswith("/"):
            return normalized_path.rstrip("/")
        return normalized_path

    def _is_http_path_within_base(self, base_path: str, target_path: str) -> bool:
        """Return whether a target path stays within an allowlisted base path."""
        normalized_base = self._normalized_http_path(base_path)
        normalized_target = self._normalized_http_path(target_path)
        if normalized_base == "/":
            return normalized_target.startswith("/")
        return normalized_target == normalized_base or normalized_target.startswith(
            f"{normalized_base}/"
        )

    def _get_hass_base_url(self) -> str:
        """Return a base URL for this Home Assistant instance."""
        try:
            return network_helper.get_url(
                self.hass,
                allow_cloud=False,
                allow_external=True,
                allow_internal=True,
                prefer_external=False,
            )
        except HomeAssistantError:
            if self.hass.config.api is None:
                raise ValueError(
                    "Unable to determine a Home Assistant base URL for local image fetches."
                ) from None

            scheme = "https" if self.hass.config.api.use_ssl else "http"
            return str(
                yarl.URL.build(
                    scheme=scheme,
                    host="127.0.0.1",
                    port=self.hass.config.api.port,
                )
            )

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

    def _sanitize_relative_image_parts(self, relative_reference: str) -> tuple[str, ...]:
        """Return safe relative path parts for a config-scoped image path."""
        normalized = str(relative_reference or "").strip().replace("\\", "/")
        if not normalized:
            raise ValueError("Image path is required.")

        relative_path = PurePosixPath(normalized)
        parts = relative_path.parts
        if not parts or relative_path.is_absolute():
            raise ValueError(
                "Image paths must be relative to the Home Assistant config directory."
            )
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError(
                "Image paths must stay inside the Home Assistant config directory."
            )
        return tuple(parts)

    def _coerce_config_relative_image_path(
        self, reference: str, config_root: Path
    ) -> tuple[str, ...]:
        """Convert a relative or config-root-absolute image path into safe parts."""
        normalized = str(reference or "").strip()
        if not normalized:
            raise ValueError("Image path is required.")

        config_root_posix = config_root.as_posix().rstrip("/")
        normalized_posix = normalized.replace("\\", "/")

        if normalized_posix == config_root_posix or normalized_posix.startswith(
            f"{config_root_posix}/"
        ):
            normalized_posix = normalized_posix[len(config_root_posix) :].lstrip("/")

        return self._sanitize_relative_image_parts(normalized_posix)

    def _resolve_config_scoped_path(
        self,
        config_root: Path,
        base_root: Path,
        relative_parts: tuple[str, ...],
    ) -> Path:
        """Resolve and validate a config-scoped path from trusted relative parts."""
        candidate = base_root.joinpath(*relative_parts).resolve()

        try:
            candidate.relative_to(config_root)
        except ValueError as err:
            raise ValueError(
                "Local image paths must stay inside the Home Assistant config directory."
            ) from err

        if not candidate.is_file():
            raise ValueError(f"Image file was not found: {candidate}")
        return candidate

    def _resolve_local_image_path(self, reference: str) -> Path:
        """Resolve a local image path inside the Home Assistant config directory."""
        config_root = Path(self.hass.config.path("")).resolve()
        if reference.startswith("/local/"):
            relative_parts = self._sanitize_relative_image_parts(
                reference.removeprefix("/local/")
            )
            return self._resolve_config_scoped_path(
                config_root,
                config_root / "www",
                relative_parts,
            )
        if reference.startswith("/media/local/"):
            relative_parts = self._sanitize_relative_image_parts(
                reference.removeprefix("/media/local/")
            )
            return self._resolve_config_scoped_path(
                config_root,
                config_root / "media",
                relative_parts,
            )

        relative_parts = self._coerce_config_relative_image_path(reference, config_root)
        return self._resolve_config_scoped_path(
            config_root,
            config_root,
            relative_parts,
        )

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
