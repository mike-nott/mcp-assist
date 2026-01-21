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
                api_key = self.entry.options.get("brave_api_key") if self.entry else None
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

    def _get_search_provider(self) -> str:
        """Get search provider with backward compatibility."""
        if not self.entry:
            return "none"

        # Check for new search_provider config
        provider = self.entry.options.get("search_provider", self.entry.data.get("search_provider"))
        if provider:
            return provider

        # Backward compat: if old enable_custom_tools was True, default to "brave"
        if self.entry.options.get("enable_custom_tools", self.entry.data.get("enable_custom_tools", False)):
            return "brave"

        return "none"

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