"""Custom tools loader for MCP Assist."""
import logging
from typing import Dict, Any, List

_LOGGER = logging.getLogger(__name__)

class CustomToolsLoader:
    """Load and manage custom tools."""

    def __init__(self, hass, entry=None):
        """Initialize the custom tools loader."""
        self.hass = hass
        self.entry = entry
        self.tools = {}

    async def initialize(self):
        """Initialize custom tools based on search provider selection."""
        # Determine search provider
        search_provider = self._get_search_provider()

        # Load search tool based on provider
        if search_provider == "brave":
            try:
                from .brave_search import BraveSearchTool
                # Get Brave API key from system entry (shared setting)
                api_key = self._get_brave_api_key()
                self.tools["search"] = BraveSearchTool(self.hass, api_key)
                await self.tools["search"].initialize()
                _LOGGER.debug("✅ Brave Search tool initialized")
            except Exception as e:
                _LOGGER.error(f"Failed to initialize Brave Search tool: {e}")

        elif search_provider == "duckduckgo":
            try:
                from .duckduckgo_search import DuckDuckGoSearchTool
                self.tools["search"] = DuckDuckGoSearchTool(self.hass)
                await self.tools["search"].initialize()
                _LOGGER.debug("✅ DuckDuckGo Search tool initialized")
            except Exception as e:
                _LOGGER.error(f"Failed to initialize DuckDuckGo Search tool: {e}")

        # Load read_url tool if search is enabled
        if search_provider in ["brave", "duckduckgo"]:
            try:
                from .read_url import ReadUrlTool
                self.tools["read_url"] = ReadUrlTool(self.hass)
                await self.tools["read_url"].initialize()
                _LOGGER.debug("✅ Read URL tool initialized")
            except Exception as e:
                _LOGGER.error(f"Failed to initialize read_url tool: {e}")

    def _get_shared_setting(self, key: str, default: Any = None) -> Any:
        """Get a shared setting from system entry with fallback to profile entry."""
        # Import here to avoid circular dependency
        from .. import get_system_entry

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
        from ..const import CONF_SEARCH_PROVIDER, CONF_ENABLE_CUSTOM_TOOLS

        provider = self._get_shared_setting(CONF_SEARCH_PROVIDER)
        if provider:
            return provider

        # Backward compat: if old enable_custom_tools was True, default to "brave"
        if self._get_shared_setting(CONF_ENABLE_CUSTOM_TOOLS, False):
            return "brave"

        return "none"

    def _get_brave_api_key(self) -> str:
        """Get Brave API key (shared setting)."""
        from ..const import CONF_BRAVE_API_KEY, DEFAULT_BRAVE_API_KEY
        return self._get_shared_setting(CONF_BRAVE_API_KEY, DEFAULT_BRAVE_API_KEY)

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get MCP tool definitions for all enabled tools."""
        definitions = []
        for tool in self.tools.values():
            try:
                definitions.extend(tool.get_tool_definitions())
            except Exception as e:
                _LOGGER.error(f"Error getting tool definitions: {e}")
        return definitions

    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a custom tool call."""
        for tool in self.tools.values():
            if tool.handles_tool(tool_name):
                return await tool.handle_call(tool_name, arguments)

        raise ValueError(f"Unknown custom tool: {tool_name}")

    def is_custom_tool(self, tool_name: str) -> bool:
        """Check if a tool name is a custom tool."""
        for tool in self.tools.values():
            if tool.handles_tool(tool_name):
                return True
        return False