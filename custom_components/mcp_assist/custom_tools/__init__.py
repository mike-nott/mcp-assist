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
        """Initialize all custom tools (brave_search and read_url)."""
        # Load brave_search tool
        try:
            from .brave_search import BraveSearchTool
            # Get API key from options
            api_key = None
            if self.entry:
                api_key = self.entry.options.get("brave_api_key")
            self.tools["brave_search"] = BraveSearchTool(self.hass, api_key)
            await self.tools["brave_search"].initialize()
            _LOGGER.debug("✅ Brave Search tool initialized")
        except Exception as e:
            _LOGGER.error(f"Failed to initialize brave_search tool: {e}")

        # Load read_url tool
        try:
            from .read_url import ReadUrlTool
            self.tools["read_url"] = ReadUrlTool(self.hass)
            await self.tools["read_url"].initialize()
            _LOGGER.debug("✅ Read URL tool initialized")
        except Exception as e:
            _LOGGER.error(f"Failed to initialize read_url tool: {e}")

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