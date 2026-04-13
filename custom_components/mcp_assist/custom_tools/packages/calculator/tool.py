"""Built-in calculator tool package wrapper."""

from __future__ import annotations

from typing import Any

from custom_components.mcp_assist.custom_tool_api import MCPAssistExternalTool
from custom_components.mcp_assist.custom_tools.calculator import CalculatorTool


class CalculatorPackageTool(MCPAssistExternalTool):
    """Expose the legacy calculator bundle through the package API."""

    def __init__(self, hass, manifest, tool_dir) -> None:
        """Initialize the wrapper and delegated calculator bundle."""
        super().__init__(hass, manifest, tool_dir)
        self._delegate = CalculatorTool(hass)

    async def initialize(self) -> None:
        """Initialize the delegated calculator bundle."""
        await self._delegate.initialize()

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return the delegated calculator tool definitions."""
        return self._delegate.get_tool_definitions()

    async def handle_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Delegate calculator tool calls to the legacy implementation."""
        return await self._delegate.handle_call(tool_name, arguments)
