"""MCP Server for Home Assistant entity discovery."""

import asyncio
import ipaddress
import json
import logging
from typing import Any, Dict, List
from urllib.parse import urlparse

from aiohttp import web, WSMsgType
from aiohttp.web_ws import WebSocketResponse

from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar, entity_registry as er
from homeassistant.components.homeassistant import async_should_expose
from homeassistant.util import dt as dt_util

from .const import (
    MCP_SERVER_NAME,
    MCP_PROTOCOL_VERSION,
    MAX_ENTITIES_PER_DISCOVERY,
    CONF_LMSTUDIO_URL,
    CONF_ALLOWED_IPS,
    CONF_SEARCH_PROVIDER,
    CONF_ENABLE_CUSTOM_TOOLS,
    DEFAULT_LMSTUDIO_URL,
    DEFAULT_ALLOWED_IPS,
)
from .discovery import EntityDiscovery
from .domain_registry import (
    validate_domain_action,
    get_supported_domains,
    get_domain_info,
    get_domains_by_type,
    TYPE_CONTROLLABLE,
    TYPE_READ_ONLY,
    TYPE_SERVICE_ONLY
)

_LOGGER = logging.getLogger(__name__)


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

        # Extract allowed IPs from LM Studio URL
        self.allowed_ips = ["127.0.0.1", "::1"]  # Always allow localhost

        # Get LM Studio URL from config
        lmstudio_url = DEFAULT_LMSTUDIO_URL
        if entry:
            # Check options first, then data
            lmstudio_url = entry.options.get(CONF_LMSTUDIO_URL,
                                            entry.data.get(CONF_LMSTUDIO_URL, DEFAULT_LMSTUDIO_URL))

        # Extract hostname/IP from LM Studio URL
        try:
            parsed = urlparse(lmstudio_url)
            lmstudio_host = parsed.hostname or parsed.netloc.split(':')[0]
            if lmstudio_host and lmstudio_host not in self.allowed_ips:
                self.allowed_ips.append(lmstudio_host)
                _LOGGER.info("MCP server automatically whitelisted LM Studio IP: %s", lmstudio_host)
        except Exception as e:
            _LOGGER.warning("Could not parse LM Studio URL '%s': %s", lmstudio_url, e)

        # Add user-configured allowed IPs/CIDR ranges (shared setting)
        allowed_ips_str = self._get_shared_setting(CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS)
        if allowed_ips_str:
            # Parse comma-separated list
            additional_ips = [ip.strip() for ip in allowed_ips_str.split(',') if ip.strip()]
            for ip_entry in additional_ips:
                if ip_entry not in self.allowed_ips:
                    self.allowed_ips.append(ip_entry)
            if additional_ips:
                _LOGGER.info("MCP server added user-configured allowed IPs/ranges: %s", additional_ips)

        _LOGGER.info("MCP server allowed IPs/ranges: %s", self.allowed_ips)

        # Initialize custom tools if search provider is enabled
        self.custom_tools = None
        search_provider = self._get_search_provider()
        if search_provider in ["brave", "duckduckgo"]:
            try:
                from .custom_tools import CustomToolsLoader
                self.custom_tools = CustomToolsLoader(hass, entry)
            except Exception as e:
                _LOGGER.error(f"Failed to load custom tools: {e}")

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

    async def start(self) -> None:
        """Start the MCP server."""
        try:
            _LOGGER.info("Starting MCP server on port %d, binding to all interfaces (0.0.0.0)", self.port)

            # Create web application (IP checks are done per-handler, not via middleware)
            self.app = web.Application()
            self.app.router.add_post("/", self.handle_mcp_request)
            self.app.router.add_get("/sse", self.handle_sse)  # SSE endpoint
            self.app.router.add_get("/", self.handle_sse)  # Also handle root GET as SSE
            self.app.router.add_get("/ws", self.handle_websocket)
            self.app.router.add_get("/health", self.handle_health)
            self.app.router.add_get("/progress", self.handle_progress_stream)  # Progress streaming

            self.runner = web.AppRunner(self.app)
            await self.runner.setup()

            # Bind to all interfaces so external machines can connect
            self.site = web.TCPSite(self.runner, "0.0.0.0", self.port)
            await self.site.start()

            # Initialize custom tools if enabled
            if self.custom_tools:
                await self.custom_tools.initialize()

            _LOGGER.info("✅ MCP server started successfully on http://0.0.0.0:%d", self.port)
            _LOGGER.info("🌐 MCP server is accessible from external machines")
            _LOGGER.info("🔗 Health check available at: http://<your-ha-ip>:%d/health", self.port)
            _LOGGER.info("📡 WebSocket endpoint: ws://<your-ha-ip>:%d/ws", self.port)
            _LOGGER.info("📤 HTTP endpoint: http://<your-ha-ip>:%d/", self.port)

        except OSError as err:
            if err.errno == 98:  # Address already in use
                _LOGGER.error("❌ Port %d is already in use. Please choose a different port.", self.port)
                raise
            elif err.errno == 13:  # Permission denied
                _LOGGER.error("❌ Permission denied to bind to port %d. Try a port >= 1024.", self.port)
                raise
            else:
                _LOGGER.error("❌ Failed to bind MCP server to port %d: %s", self.port, err)
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
        if ip_only.startswith('['):
            end_bracket = ip_only.find(']')
            if end_bracket > 0:
                ip_only = ip_only[1:end_bracket]
        # Handle IPv4 with port: 192.168.1.7:12345 -> 192.168.1.7
        # Only split on single colon (not IPv6 which has multiple colons)
        elif ip_only.count(':') == 1:
            ip_only = ip_only.split(':')[0]
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
            if '/' in allowed_entry:
                try:
                    network = ipaddress.ip_network(allowed_entry, strict=False)
                    if client_ip_obj in network:
                        return True
                except ValueError:
                    # Invalid CIDR format, skip
                    _LOGGER.warning("Invalid CIDR format in allowed IPs: %s", allowed_entry)
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
                "health": f"http://<host>:{self.port}/health"
            },
            "tools_available": len(await self._get_tools_list()),
            "timestamp": dt_util.now().isoformat()
        }
        return web.json_response(health_info)

    async def handle_progress_stream(self, request: web.Request) -> web.StreamResponse:
        """SSE endpoint for progress updates during tool execution."""
        client_ip = request.remote
        _LOGGER.info("📊 Progress stream request from %s", client_ip)

        # Check IP whitelist
        if not self._is_ip_allowed(client_ip):
            _LOGGER.warning("🚫 Blocked progress stream request from unauthorized IP: %s", client_ip)
            return web.Response(status=403, text="Forbidden: IP not authorized")

        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
            }
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
            **kwargs
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
            _LOGGER.warning("🚫 Blocked SSE connection from unauthorized IP: %s", client_ip)
            return web.Response(status=403, text="Forbidden: IP not authorized")

        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        response.headers['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
        response.headers['Access-Control-Allow-Origin'] = '*'

        await response.prepare(request)

        # Store this client for notifications
        self.sse_clients.append(response)
        _LOGGER.info("✅ SSE client connected. Total clients: %d", len(self.sse_clients))

        try:
            # Send initial connection confirmation
            await response.write(b': connected\n\n')

            # Send tools list changed notification immediately
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/tools/list_changed"
            }
            await response.write(f"data: {json.dumps(notification)}\n\n".encode())
            _LOGGER.info("📤 Sent initial tools/list_changed notification")

            # Keep connection alive
            while True:
                await asyncio.sleep(30)
                await response.write(b': keepalive\n\n')

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

    async def handle_websocket(self, request: web.Request) -> WebSocketResponse:
        """Handle WebSocket connections for MCP protocol."""
        client_ip = request.remote
        _LOGGER.info("🔌 New MCP WebSocket connection from %s", client_ip)

        # Check IP whitelist
        if not self._is_ip_allowed(client_ip):
            _LOGGER.warning("🚫 Blocked WebSocket connection from unauthorized IP: %s", client_ip)
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
                        await ws.send_str(json.dumps({
                            "error": {"code": -32700, "message": "Parse error"}
                        }))
                    except Exception as err:
                        _LOGGER.exception("Error processing MCP message")
                        await ws.send_str(json.dumps({
                            "error": {"code": -32000, "message": f"Server error: {err}"}
                        }))
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
            return web.json_response({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32000,
                    "message": "Forbidden: IP not authorized"
                },
                "id": None
            }, status=403)

        request_id = None
        try:
            data = await request.json()
            request_id = data.get("id")

            # Validate JSON-RPC 2.0 format
            if "jsonrpc" not in data or data["jsonrpc"] != "2.0":
                return web.json_response({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request: missing or invalid jsonrpc field"
                    },
                    "id": request_id
                }, status=400)

            if "method" not in data:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request: missing method field"
                    },
                    "id": request_id
                }, status=400)

            # Check if this is a notification (no id field)
            is_notification = "id" not in data

            if is_notification:
                _LOGGER.debug("📮 MCP notification: %s", data.get("method"))
                # Process the notification but don't expect a response
                await self.process_mcp_notification(data)
                # Return 204 No Content for notifications
                return web.Response(status=204)
            else:
                _LOGGER.debug("📋 MCP method: %s (id: %s)", data.get("method"), request_id)
                response = await self.process_mcp_message(data)
                return web.json_response(response)

        except json.JSONDecodeError:
            return web.json_response({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,
                    "message": "Parse error: invalid JSON"
                },
                "id": None
            }, status=400)
        except Exception as err:
            _LOGGER.exception("Error processing MCP request")
            return web.json_response({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(err)}"
                },
                "id": request_id
            }, status=500)

    async def process_mcp_notification(self, data: Dict[str, Any]) -> None:
        """Process MCP notification (no response expected)."""
        method = data.get("method")
        params = data.get("params", {})

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

    async def broadcast_notification(self, method: str, params: Dict[str, Any] | None = None) -> None:
        """Send notification to all SSE clients."""
        if not self.sse_clients:
            _LOGGER.debug("No SSE clients to notify for %s", method)
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method
        }
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
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    },
                    "id": msg_id
                }

            # Always include jsonrpc and id in successful responses
            response = {
                "jsonrpc": "2.0",
                "result": result,
                "id": msg_id
            }

            return response

        except Exception as err:
            _LOGGER.exception("Error in MCP method %s", method)
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error in {method}: {str(err)}"
                },
                "id": msg_id
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
            "serverInfo": {
                "name": MCP_SERVER_NAME,
                "version": "0.1.0"
            }
        }

    async def handle_tools_list(self) -> Dict[str, Any]:
        """Handle tools/list request."""
        _LOGGER.info("MCP tools/list request received")

        tools = [
            {
                "name": "discover_entities",
                "description": "Find and list Home Assistant entities by various criteria like area, type, domain, device_class, or current state. Use this to discover what devices are available before trying to control them.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "entity_type": {
                            "type": "string",
                            "description": "Type of entity to find (e.g., 'light', 'switch', 'sensor', 'climate')"
                        },
                        "area": {
                            "type": "string",
                            "description": "Area/room name to search in - use exact names from the areas list provided in your system context (e.g., 'Kitchen', 'Back Garden', 'Living Room')"
                        },
                        "domain": {
                            "type": "string",
                            "description": "Home Assistant domain to filter by (e.g., 'light', 'switch', 'climate', 'sensor')"
                        },
                        "state": {
                            "type": "string",
                            "description": "Current state to filter by (e.g., 'on', 'off', 'unavailable')"
                        },
                        "name_contains": {
                            "type": "string",
                            "description": "Text that entity name should contain (case-insensitive)"
                        },
                        "device_class": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}}
                            ],
                            "description": "Device class to filter by (e.g., 'temperature', 'motion', 'door', 'moisture'). Can be a single string or array of strings for OR logic. Check the index for available device classes per domain."
                        },
                        "name_pattern": {
                            "type": "string",
                            "description": "Wildcard pattern to match entity IDs (e.g., '*_person_detected', 'sensor.*_ble_area'). Supports * for any characters."
                        },
                        "inferred_type": {
                            "type": "string",
                            "description": "Inferred entity type from the index (e.g., 'person_detection', 'location_tracking'). The pattern will be looked up from the index's inferred_types. Check get_index() to see available inferred types."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of entities to return (default: 20, max: 50)",
                            "default": 20
                        }
                    },
                    "required": [],
                    "additionalProperties": False
                }
            },
            {
                "name": "get_entity_details",
                "description": "Get current state and attributes of specific entities",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "entity_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of entity IDs to get details for"
                        }
                    },
                    "required": ["entity_ids"],
                    "additionalProperties": False
                }
            },
            {
                "name": "list_areas",
                "description": "List all areas in the home with their entities count",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            },
            {
                "name": "list_domains",
                "description": "List all available domains with entity counts",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            },
            {
                "name": "get_index",
                "description": "Get the pre-generated system structure index. This index provides a lightweight overview of the Home Assistant system including areas, domains, device classes, people, pets, calendars, zones, automations, and scripts. Call this ONCE at the start of a conversation to understand what exists in the system, then use discover_entities to query specific entities. The index is ~400-800 tokens vs ~15k tokens for a full entity dump.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            },
            {
                "name": "perform_action",
                "description": "Control Home Assistant devices by calling services. Use after discovering entities to turn on/off lights, set temperatures, open/close covers, etc.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "The domain of the service to call (e.g., 'light', 'switch', 'climate', 'vacuum', 'media_player', etc.)"
                        },
                        "action": {
                            "type": "string",
                            "description": "The service action (e.g., 'turn_on', 'turn_off', 'toggle', 'set_temperature')"
                        },
                        "target": {
                            "type": "object",
                            "description": "Target entities, areas, or devices",
                            "properties": {
                                "entity_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}}
                                    ],
                                    "description": "Single entity ID or list of entity IDs"
                                },
                                "area_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}}
                                    ],
                                    "description": "Single area ID or list of area IDs"
                                },
                                "device_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}}
                                    ],
                                    "description": "Single device ID or list of device IDs"
                                }
                            },
                            "minProperties": 1,
                            "additionalProperties": False
                        },
                        "data": {
                            "type": "object",
                            "description": "Additional parameters for the service (e.g., brightness: 50, temperature: 22)",
                            "additionalProperties": True
                        }
                    },
                    "required": ["domain", "action", "target"],
                    "additionalProperties": False
                }
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
                            "description": "true if expecting user response, false if task is complete"
                        }
                    },
                    "required": ["expecting_response"],
                    "additionalProperties": False
                }
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
                            "description": "The script entity ID (e.g., 'script.llm_camera_analysis' or just 'llm_camera_analysis')"
                        },
                        "variables": {
                            "type": "object",
                            "description": "Variables to pass to the script",
                            "additionalProperties": True
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 60)",
                            "default": 60
                        }
                    },
                    "required": ["script_id"],
                    "additionalProperties": False
                }
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
                            "description": "The automation entity ID (e.g., 'automation.notify_on_motion' or just 'notify_on_motion')"
                        },
                        "variables": {
                            "type": "object",
                            "description": "Variables to pass to the automation (available as trigger.variables)",
                            "additionalProperties": True
                        },
                        "skip_conditions": {
                            "type": "boolean",
                            "description": "Whether to skip the automation's conditions (default: false)",
                            "default": False
                        }
                    },
                    "required": ["automation_id"],
                    "additionalProperties": False
                }
            }
        ]

        # Add custom tool definitions if enabled
        if self.custom_tools:
            try:
                custom_tool_defs = self.custom_tools.get_tool_definitions()
                tools.extend(custom_tool_defs)
            except Exception as e:
                _LOGGER.error(f"Failed to get custom tool definitions: {e}")

        # nextCursor is optional - omit if not paginating
        return {"tools": tools}

    async def handle_tool_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        _LOGGER.debug("Calling tool: %s with args: %s", tool_name, arguments)

        if tool_name == "discover_entities":
            return await self.tool_discover_entities(arguments)
        elif tool_name == "get_entity_details":
            return await self.tool_get_entity_details(arguments)
        elif tool_name == "list_areas":
            return await self.tool_list_areas()
        elif tool_name == "list_domains":
            return await self.tool_list_domains()
        elif tool_name == "get_index":
            return await self.tool_get_index()
        elif tool_name == "perform_action":
            return await self.tool_perform_action(arguments)
        elif tool_name == "set_conversation_state":
            return await self.tool_set_conversation_state(arguments)
        elif tool_name == "run_script":
            return await self.tool_run_script(arguments)
        elif tool_name == "run_automation":
            return await self.tool_run_automation(arguments)
        else:
            # Check if it's a custom tool
            if self.custom_tools and self.custom_tools.is_custom_tool(tool_name):
                return await self.custom_tools.handle_tool_call(tool_name, arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

    async def tool_discover_entities(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Discover entities based on criteria with progress notifications."""
        # Notify start
        self.publish_progress("tool_start", "Starting entity discovery", tool="discover_entities", args=args)

        entities = await self.discovery.discover_entities(
            entity_type=args.get("entity_type"),
            area=args.get("area"),
            domain=args.get("domain"),
            state=args.get("state"),
            name_contains=args.get("name_contains"),
            limit=args.get("limit", 20),
            device_class=args.get("device_class"),
            name_pattern=args.get("name_pattern"),
            inferred_type=args.get("inferred_type"),
        )

        # Notify completion
        self.publish_progress(
            "tool_complete",
            f"Discovery complete: found {len(entities)} entities",
            tool="discover_entities",
            count=len(entities)
        )

        # Format results based on whether it's smart discovery or general
        return self._format_discovery_results(entities, args)

    def _format_discovery_results(self, entities: List[Dict[str, Any]], args: Dict[str, Any]) -> Dict[str, Any]:
        """Format discovery results for the LLM, handling both smart and general discovery."""
        if not entities:
            return {
                "content": [{
                    "type": "text",
                    "text": "No entities found matching the search criteria."
                }]
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
                text_parts.append(f"📊 Aggregate Discovery")
            else:
                text_parts.append(f"🔍 Discovery Results")

            text_parts.append(f"Found {summary.get('total_found', 0)} total entities")

            # Group entities by relationship
            primary = [e for e in actual_entities if e.get("relationship") == "primary"]
            related = [e for e in actual_entities if e.get("relationship") != "primary"]

            # Add primary entities
            if primary:
                text_parts.append("\n📍 Primary Entities:")
                for entity in primary:
                    type_desc = f" ({entity.get('type', '')})" if entity.get('type') else ""
                    text_parts.append(f"  • {entity['entity_id']}: {entity['name']} - {entity['state']}{type_desc}")

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
                        area = f" @ {entity.get('area')}" if entity.get('area') else ""
                        text_parts.append(f"    • {entity['entity_id']}: {entity['state']}{area}")

            return {
                "content": [{
                    "type": "text",
                    "text": "\n".join(text_parts)
                }]
            }
        else:
            # General discovery - simple list
            text_parts = [f"Found {len(entities)} entities:"]

            for entity in entities:
                area = entity.get('area', 'None')
                text_parts.append(
                    f"- {entity['entity_id']}: {entity['name']} (State: {entity['state']}, Area: {area})"
                )

            return {
                "content": [{
                    "type": "text",
                    "text": "\n".join(text_parts)
                }]
            }

    async def tool_get_entity_details(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed information about specific entities."""
        entity_ids = args.get("entity_ids", [])
        details = await self.discovery.get_entity_details(entity_ids)

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(details, indent=2)
                }
            ]
        }

    async def tool_list_areas(self) -> Dict[str, Any]:
        """List all areas."""
        areas = await self.discovery.list_areas()

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Available areas ({len(areas)}):\n" +
                           "\n".join([
                               f"- {area['name']}: {area['entity_count']} entities"
                               for area in areas
                           ])
                }
            ]
        }

    async def tool_list_domains(self) -> Dict[str, Any]:
        """List all domains with entity counts and support status."""
        # Get domains that have entities in this HA instance
        entity_domains = await self.discovery.list_domains()
        entity_domain_map = {d['domain']: d['count'] for d in entity_domains}

        # Get all supported domains from registry
        supported_domains = get_supported_domains()
        controllable_domains = get_domains_by_type(TYPE_CONTROLLABLE)
        read_only_domains = get_domains_by_type(TYPE_READ_ONLY)

        # Build comprehensive list
        result_text = f"Home Assistant Domains (Entities: {len(entity_domains)}, Supported: {len(supported_domains)}):\n\n"

        # Show domains with entities
        result_text += "📊 Domains with entities in your system:\n"
        for domain in entity_domains:
            support_status = "✅" if domain['domain'] in supported_domains else "⚠️"
            result_text += f"  {support_status} {domain['domain']}: {domain['count']} entities\n"

        # Show supported domains without entities
        result_text += "\n🔧 Additional supported domains (no entities found):\n"
        for domain in supported_domains:
            if domain not in entity_domain_map:
                domain_type = "controllable" if domain in controllable_domains else "read-only" if domain in read_only_domains else "service"
                result_text += f"  ✅ {domain} ({domain_type})\n"

        result_text += f"\n📈 Summary:\n"
        result_text += f"  - Total entity domains: {len(entity_domains)}\n"
        result_text += f"  - Supported domains: {len(supported_domains)}\n"
        result_text += f"  - Controllable: {len(controllable_domains)}\n"
        result_text += f"  - Read-only: {len(read_only_domains)}\n"

        return {
            "content": [
                {
                    "type": "text",
                    "text": result_text
                }
            ]
        }

    async def tool_get_index(self) -> Dict[str, Any]:
        """Get the pre-generated system structure index."""
        from .const import DOMAIN

        # Get index manager from hass.data
        index_manager = self.hass.data.get(DOMAIN, {}).get("index_manager")

        if not index_manager:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Index manager not available. This feature requires MCP Assist 0.5.0 or later."
                    }
                ]
            }

        # Get the index
        index = await index_manager.get_index()

        # Format as JSON for structured consumption
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(index, indent=2)
                }
            ]
        }

    async def tool_perform_action(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Perform an action on Home Assistant entities with progress notifications."""
        domain = args.get("domain")
        action = args.get("action")
        target = args.get("target", {})
        data = args.get("data", {})

        _LOGGER.info(f"🎯 Performing action: {domain}.{action} on {target}")

        # Notify start
        self.publish_progress(
            "tool_start",
            f"Performing action: {domain}.{action}",
            tool="perform_action",
            domain=domain,
            action=action
        )

        # Validate the service and get the correct service name
        try:
            service = self.validate_service(domain, action)
        except ValueError as err:
            error_msg = str(err)
            _LOGGER.error(f"Service validation error: {error_msg}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"❌ Error: {error_msg}"
                    }
                ]
            }

        # Resolve target (convert areas to entity_ids if needed)
        try:
            resolved_target = await self.resolve_target(target)
            _LOGGER.debug(f"Resolved target: {resolved_target}")
        except Exception as err:
            error_msg = f"Failed to resolve target: {err}"
            _LOGGER.error(error_msg)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"❌ Error: {error_msg}"
                    }
                ]
            }

        try:
            # Prepare service data
            service_data = {**resolved_target, **data}

            # Call the Home Assistant service with the validated service name
            await self.hass.services.async_call(
                domain=domain,
                service=service,  # Use the mapped service name
                service_data=service_data,
                blocking=True,  # Wait for completion
                return_response=False
            )

            # Wait briefly for state to update
            await asyncio.sleep(0.5)

            # Notify completion
            self.publish_progress(
                "tool_complete",
                f"Action completed: {domain}.{service}",
                tool="perform_action",
                success=True
            )

            # Check new states if we have entity_ids
            result_text = f"✅ Successfully executed {domain}.{service}"
            if service != action:
                result_text += f" (mapped from '{action}')"

            if "entity_id" in resolved_target:
                entity_ids = resolved_target["entity_id"]
                if isinstance(entity_ids, str):
                    entity_ids = [entity_ids]

                # Get new states
                states_info = []
                for entity_id in entity_ids[:10]:  # Limit to first 10 for response size
                    state = self.hass.states.get(entity_id)
                    if state:
                        states_info.append(f"  • {entity_id}: {state.state}")

                if states_info:
                    result_text += "\n\nNew states:\n" + "\n".join(states_info)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": result_text
                    }
                ]
            }

        except Exception as err:
            error_msg = f"Service call failed: {err}"
            _LOGGER.exception(error_msg)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"❌ Error: {error_msg}"
                    }
                ]
            }

    async def tool_set_conversation_state(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Set whether the assistant expects a user response."""
        expecting_response = args.get("expecting_response", False)

        # Log the state for debugging
        _LOGGER.info(f"🔄 Conversation state set: expecting_response={expecting_response}")

        # Return a marker that the agent can detect
        return {
            "content": [{
                "type": "text",
                "text": f"conversation_state:{expecting_response}"
            }]
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
            script_id=full_script_id
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
                timeout=timeout
            )

            # Notify completion
            self.publish_progress(
                "tool_complete",
                f"Script completed: {full_script_id}",
                tool="run_script",
                success=True
            )

            # Format the response
            result_text = f"✅ Script {full_script_id} completed successfully"

            # If the script returned response variables, include them
            if response:
                result_text += f"\n\nResponse:\n{json.dumps(response, indent=2)}"
                return {
                    "content": [{
                        "type": "text",
                        "text": result_text
                    }],
                    "response": response
                }
            else:
                result_text += "\n\nNo response variables returned (script may not have response_variable defined)"
                return {
                    "content": [{
                        "type": "text",
                        "text": result_text
                    }]
                }

        except asyncio.TimeoutError:
            error_msg = f"Script execution timed out after {timeout} seconds"
            _LOGGER.error(f"❌ {error_msg}: {full_script_id}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Error: {error_msg}"
                }]
            }
        except Exception as err:
            error_msg = f"Script execution failed: {err}"
            _LOGGER.exception(f"❌ {error_msg}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Error: {error_msg}"
                }]
            }

    async def tool_run_automation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger a Home Assistant automation with optional variables."""
        automation_id = args.get("automation_id")
        variables = args.get("variables", {})
        skip_conditions = args.get("skip_conditions", False)

        # Normalize automation_id (add automation. prefix if missing)
        if not automation_id.startswith("automation."):
            automation_id = f"automation.{automation_id}"

        _LOGGER.info(f"🤖 Triggering automation: {automation_id} with variables: {variables}, skip_conditions: {skip_conditions}")

        # Notify start
        self.publish_progress(
            "tool_start",
            f"Triggering automation: {automation_id}",
            tool="run_automation",
            automation_id=automation_id
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
                success=True
            )

            result_text = f"✅ Automation {automation_id} triggered successfully"
            if skip_conditions:
                result_text += " (conditions skipped)"

            return {
                "content": [{
                    "type": "text",
                    "text": result_text
                }]
            }

        except Exception as err:
            error_msg = f"Automation trigger failed: {err}"
            _LOGGER.exception(f"❌ {error_msg}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Error: {error_msg}"
                }]
            }

    def validate_service(self, domain: str, action: str) -> str:
        """Validate that a domain/action combination is allowed.

        Returns:
            The correct service name to use

        Raises:
            ValueError: If domain or action is invalid
        """
        valid, result = validate_domain_action(domain, action)
        if valid:
            _LOGGER.debug(f"Validated service: {domain}.{result} (from action: {action})")
            return result  # Returns the correct service name
        else:
            _LOGGER.warning(f"Service validation failed: {result}")
            raise ValueError(result)  # Returns error message

    async def resolve_target(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve areas and devices to entity_ids."""
        resolved = {}

        # Pass through entity_id as-is
        if "entity_id" in target:
            entity_id = target["entity_id"]
            # Ensure it's a list for consistency
            if isinstance(entity_id, str):
                entity_id = [entity_id]
            resolved["entity_id"] = entity_id
            _LOGGER.debug(f"Using entity_ids: {entity_id}")

        # Resolve area_id to entities
        if "area_id" in target:
            area_ids = target["area_id"]
            if isinstance(area_ids, str):
                area_ids = [area_ids]

            area_entities = []
            for area_id in area_ids:
                # Get entities in this area
                entities = await self.discovery.get_entities_by_area(area_id)
                area_entities.extend([e["entity_id"] for e in entities])
                _LOGGER.debug(f"Found {len(entities)} entities in area '{area_id}'")

            if area_entities:
                if "entity_id" in resolved:
                    # Merge with existing entity_ids
                    existing = resolved["entity_id"]
                    resolved["entity_id"] = list(set(existing + area_entities))
                else:
                    resolved["entity_id"] = area_entities

        # Pass through device_id if present (HA will handle it)
        if "device_id" in target:
            resolved["device_id"] = target["device_id"]
            _LOGGER.debug(f"Using device_id: {target['device_id']}")

        return resolved if resolved else target