"""Public API for user-defined MCP Assist custom tools.

This module is intentionally stable and documented so local tools placed under
the Home Assistant config directory can import it directly:

    <home-assistant-config>/mcp-assist-tools/<tool_id>/tool.py

Tool-package metadata should live in:

    <home-assistant-config>/mcp-assist-tools/<tool_id>/mcp_tool.json
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant


@dataclass(frozen=True)
class MCPAssistCustomToolManifest:
    """Validated metadata for a user-defined custom tool package."""

    schema_version: int
    tool_id: str
    name: str
    description: str
    version: str
    entrypoint: str
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    prompt_append_file: str | None = None


class MCPAssistExternalTool:
    """Base class for user-defined MCP Assist custom tool packages.

    Custom tool packages should subclass this class and implement:
    - get_tool_definitions()
    - handle_call()

    Optional hooks:
    - initialize()
    - async_shutdown()
    - get_prompt_instructions()
    """

    def __init__(
        self,
        hass: HomeAssistant,
        manifest: MCPAssistCustomToolManifest,
        tool_dir: Path,
    ) -> None:
        """Initialize the external tool instance."""
        self.hass = hass
        self.manifest = manifest
        self.tool_dir = tool_dir

    async def initialize(self) -> None:
        """Initialize the tool package.

        Override when the tool needs lightweight setup such as reading a local
        config file or preparing cached state.
        """

    async def async_shutdown(self) -> None:
        """Clean up any tool resources before MCP Assist shuts down."""

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return MCP tool definitions exposed by this package."""
        raise NotImplementedError

    def handles_tool(self, tool_name: str) -> bool:
        """Return whether this package handles the given tool name."""
        return tool_name in {
            str(tool.get("name") or "")
            for tool in self.get_tool_definitions()
        }

    async def handle_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle a tool call and return standard MCP content."""
        raise NotImplementedError

    def get_prompt_instructions(self) -> str:
        """Return optional prompt instructions for the LLM.

        Keep this short and procedural. The loader will append the returned
        text to the technical instructions only when external custom tools are
        enabled.
        """

        return ""
