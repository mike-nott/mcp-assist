"""Example external MCP Assist tool package."""

from __future__ import annotations

from typing import Any

from custom_components.mcp_assist.custom_tool_api import MCPAssistExternalTool


class SampleTool(MCPAssistExternalTool):
    """Minimal example custom tool."""

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Expose one namespaced example tool."""
        return [
            {
                "name": "sample_tool_status",
                "description": "Return the status exposed by the sample custom tool.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            }
        ]

    async def handle_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Return a normal MCP tool result."""
        return {
            "content": [
                {
                    "type": "text",
                    "text": "Sample custom tool is working.",
                }
            ],
            "isError": False,
        }

    def get_prompt_instructions(self) -> str:
        """Add short guidance for the LLM."""
        return "Use sample_tool_status for sample-status questions."
