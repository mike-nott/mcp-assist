"""Built-in search tool package wrapper."""

from __future__ import annotations

from typing import Any

from custom_components.mcp_assist import get_system_entry
from custom_components.mcp_assist.const import (
    CONF_BRAVE_API_KEY,
    CONF_ENABLE_CUSTOM_TOOLS,
    CONF_ENABLE_WEB_SEARCH,
    CONF_SEARCH_PROVIDER,
    DEFAULT_BRAVE_API_KEY,
)
from custom_components.mcp_assist.custom_tool_api import MCPAssistExternalTool


class SearchPackageTool(MCPAssistExternalTool):
    """Expose the configured search provider through the package API."""

    def __init__(self, hass, manifest, tool_dir) -> None:
        """Initialize the search wrapper."""
        super().__init__(hass, manifest, tool_dir)
        self._delegate: Any | None = None

    async def initialize(self) -> None:
        """Initialize the configured search provider."""
        provider = self._get_search_provider()
        if provider == "brave":
            from custom_components.mcp_assist.custom_tools.brave_search import (
                BraveSearchTool,
            )

            self._delegate = BraveSearchTool(
                self.hass,
                self._get_shared_setting(CONF_BRAVE_API_KEY, DEFAULT_BRAVE_API_KEY),
            )
        elif provider == "duckduckgo":
            from custom_components.mcp_assist.custom_tools.duckduckgo_search import (
                DuckDuckGoSearchTool,
            )

            self._delegate = DuckDuckGoSearchTool(self.hass)
        else:
            raise ValueError(
                "Built-in search package loaded without a supported search provider."
            )

        await self._delegate.initialize()

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return the delegated search tool definition."""
        return self._require_delegate().get_tool_definitions()

    async def handle_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Delegate search tool calls to the configured provider."""
        return await self._require_delegate().handle_call(tool_name, arguments)

    def _get_shared_setting(self, key: str, default: Any = None) -> Any:
        """Read a shared MCP Assist setting from the system entry."""
        system_entry = get_system_entry(self.hass)
        if system_entry is None:
            return default

        value = system_entry.options.get(key, system_entry.data.get(key))
        if value is not None:
            return value
        return default

    def _get_search_provider(self) -> str:
        """Return the effective shared search provider."""
        provider = self._get_shared_setting(CONF_SEARCH_PROVIDER)
        if provider:
            explicit_enabled = self._get_shared_setting(CONF_ENABLE_WEB_SEARCH)
            if explicit_enabled is False:
                return "none"
            return str(provider)

        if self._get_shared_setting(CONF_ENABLE_CUSTOM_TOOLS, False):
            return "brave"

        return "none"

    def _require_delegate(self) -> Any:
        """Return the configured provider implementation."""
        if self._delegate is None:
            raise RuntimeError("Search package was not initialized")
        return self._delegate
