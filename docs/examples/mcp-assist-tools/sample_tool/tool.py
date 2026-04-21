"""Example external MCP Assist tool package."""

from __future__ import annotations

from typing import Any

from custom_components.mcp_assist.custom_tool_api import MCPAssistExternalTool


class SampleTool(MCPAssistExternalTool):
    """Minimal example custom tool."""

    def get_settings_schema(self) -> dict[str, Any]:
        """Optional shared/profile settings schema for this package."""
        return {
            "type": "object",
            "properties": {
                "status_text": {
                    "type": "string",
                    "default": "Sample custom tool is working.",
                }
            },
        }

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
                "keywords": ["sample", "status", "custom"],
                "example_queries": ["What's the sample tool status?"],
                "preferred_when": "Use when the user asks about the sample package.",
                "returns": "A short status summary.",
            }
        ]

    async def handle_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Return a normal MCP tool result."""
        settings = self.get_settings()
        return {
            "content": [
                {
                    "type": "text",
                    "text": str(
                        settings.get("status_text")
                        or "Sample custom tool is working."
                    ),
                }
            ],
            "isError": False,
        }

    def get_prompt_instructions(self) -> str:
        """Add short guidance for the LLM."""
        return "Use sample_tool_status for sample-status questions."
