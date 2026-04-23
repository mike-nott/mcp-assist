"""Built-in read_url package wrapper."""

from __future__ import annotations

from typing import Any

from custom_components.mcp_assist.custom_tool_api import MCPAssistExternalTool
from custom_components.mcp_assist.custom_tools.read_url import ReadUrlTool


class ReadUrlPackageTool(MCPAssistExternalTool):
    """Expose the legacy read_url tool through the package API."""

    def __init__(self, hass, manifest, tool_dir) -> None:
        """Initialize the wrapper and delegated read_url tool."""
        super().__init__(hass, manifest, tool_dir)
        self._delegate = ReadUrlTool(hass)

    async def initialize(self) -> None:
        """Initialize the delegated read_url tool."""
        await self._delegate.initialize()

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return the delegated read_url tool definition."""
        return self._delegate.get_tool_definitions()

    async def handle_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Delegate URL-reading tool calls to the legacy implementation."""
        return await self._delegate.handle_call(tool_name, arguments)
