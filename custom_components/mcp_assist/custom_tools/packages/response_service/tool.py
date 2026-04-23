"""Built-in Response Service Reads package wrapper."""

from __future__ import annotations

from typing import Any

from custom_components.mcp_assist.custom_tool_api import MCPAssistExternalTool
from custom_components.mcp_assist.custom_tools.response_services import (
    ResponseServiceTool,
)


class ResponseServicePackageTool(MCPAssistExternalTool):
    """Expose the built-in Response Service Reads tool bundle through the package API."""

    def __init__(self, hass, manifest, tool_dir) -> None:
        """Initialize the wrapper and delegated tool bundle."""
        super().__init__(hass, manifest, tool_dir)
        self._delegate = ResponseServiceTool(hass)

    async def initialize(self) -> None:
        """Initialize the delegated tool bundle."""
        await self._delegate.initialize()

    async def async_shutdown(self) -> None:
        """Shut down the delegated tool bundle."""
        await self._delegate.async_shutdown()

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return the delegated tool definitions."""
        return self._delegate.get_tool_definitions()

    async def handle_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Delegate tool calls to the packaged bundle."""
        return await self._delegate.handle_call(tool_name, arguments)
