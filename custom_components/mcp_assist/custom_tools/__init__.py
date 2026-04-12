"""Built-in and external custom tool loading for MCP Assist."""

from __future__ import annotations

import logging
from typing import Any

from ..const import (
    CONF_BRAVE_API_KEY,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_CUSTOM_TOOLS,
    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_ENABLE_WEB_SEARCH,
    CONF_SEARCH_PROVIDER,
    DEFAULT_BRAVE_API_KEY,
    DEFAULT_ENABLE_CALCULATOR_TOOLS,
    DEFAULT_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    DEFAULT_ENABLE_WEB_SEARCH,
)
from .external_loader import ExternalCustomToolLoader, LoadedExternalTool, combine_prompt_instructions

_LOGGER = logging.getLogger(__name__)


class CustomToolsLoader:
    """Load and manage built-in and user-defined custom tools."""

    def __init__(self, hass, entry=None):
        """Initialize the custom tools loader."""
        self.hass = hass
        self.entry = entry
        self.tools: dict[str, Any] = {}
        self.external_tools: list[LoadedExternalTool] = []
        self._external_prompt_instructions = ""

    async def initialize(self):
        """Initialize built-in bundles and optional external tool packages."""
        await self._initialize_builtin_tools()
        await self._initialize_external_tools()

    async def _initialize_builtin_tools(self) -> None:
        """Load built-in tool bundles that ship with MCP Assist."""
        calculator_enabled = self._get_shared_setting(
            CONF_ENABLE_CALCULATOR_TOOLS, DEFAULT_ENABLE_CALCULATOR_TOOLS
        )
        unit_conversion_enabled = self._get_shared_setting(
            CONF_ENABLE_UNIT_CONVERSION_TOOLS, None
        )
        if unit_conversion_enabled is None:
            unit_conversion_enabled = calculator_enabled

        if calculator_enabled or unit_conversion_enabled:
            try:
                from .calculator import CalculatorTool

                self.tools["calculator"] = CalculatorTool(self.hass)
                await self.tools["calculator"].initialize()
                _LOGGER.debug("✅ Calculator and/or unit-conversion tools initialized")
            except Exception as err:
                _LOGGER.error(
                    "Failed to initialize calculator/unit-conversion tools: %s",
                    err,
                )
        else:
            _LOGGER.debug(
                "Calculator and unit-conversion tools disabled in shared MCP settings"
            )

        search_provider = self._get_search_provider()
        web_search_enabled = self._get_shared_setting(
            CONF_ENABLE_WEB_SEARCH,
            DEFAULT_ENABLE_WEB_SEARCH,
        )

        if not web_search_enabled:
            _LOGGER.debug("Web search tools disabled in shared MCP settings")
            return

        if search_provider == "brave":
            try:
                from .brave_search import BraveSearchTool

                api_key = self._get_brave_api_key()
                self.tools["search"] = BraveSearchTool(self.hass, api_key)
                await self.tools["search"].initialize()
                _LOGGER.debug("✅ Brave Search tool initialized")
            except Exception as err:
                _LOGGER.error("Failed to initialize Brave Search tool: %s", err)
        elif search_provider == "duckduckgo":
            try:
                from .duckduckgo_search import DuckDuckGoSearchTool

                self.tools["search"] = DuckDuckGoSearchTool(self.hass)
                await self.tools["search"].initialize()
                _LOGGER.debug("✅ DuckDuckGo Search tool initialized")
            except Exception as err:
                _LOGGER.error("Failed to initialize DuckDuckGo Search tool: %s", err)

        if search_provider in {"brave", "duckduckgo"}:
            try:
                from .read_url import ReadUrlTool

                self.tools["read_url"] = ReadUrlTool(self.hass)
                await self.tools["read_url"].initialize()
                _LOGGER.debug("✅ Read URL tool initialized")
            except Exception as err:
                _LOGGER.error("Failed to initialize read_url tool: %s", err)

    async def _initialize_external_tools(self) -> None:
        """Load opt-in user-defined tool packages from the HA config directory."""
        if not self._external_custom_tools_enabled():
            _LOGGER.debug("External custom tools disabled in shared MCP settings")
            self.external_tools = []
            self._external_prompt_instructions = ""
            return

        reserved_tool_names = {
            str(tool_definition.get("name") or "")
            for tool_definition in self.get_tool_definitions()
        }
        loader = ExternalCustomToolLoader(self.hass)
        self.external_tools = await loader.load(reserved_tool_names=reserved_tool_names)
        for loaded_tool in self.external_tools:
            self.tools[loaded_tool.manifest.tool_id] = loaded_tool.instance

        self._external_prompt_instructions = combine_prompt_instructions(
            self.external_tools
        )
        if self.external_tools:
            _LOGGER.info(
                "✅ Loaded %d external custom tool package(s) from %s",
                len(self.external_tools),
                loader.get_tools_root(),
            )
        else:
            _LOGGER.info("No external custom tool packages were loaded")

    async def shutdown(self) -> None:
        """Shut down any loaded tool packages cleanly."""
        for loaded_tool in self.external_tools:
            try:
                await loaded_tool.instance.async_shutdown()
            except Exception as err:
                _LOGGER.warning(
                    "External custom tool %s failed during shutdown: %s",
                    loaded_tool.manifest.tool_id,
                    err,
                )

    def _get_shared_setting(self, key: str, default: Any = None) -> Any:
        """Get a shared setting from system entry with fallback to profile entry."""
        from .. import get_system_entry

        system_entry = get_system_entry(self.hass)
        if system_entry:
            value = system_entry.options.get(key, system_entry.data.get(key))
            if value is not None:
                return value

        if self.entry:
            value = self.entry.options.get(key, self.entry.data.get(key))
            if value is not None:
                return value

        return default

    def _external_custom_tools_enabled(self) -> bool:
        """Return whether user-defined external tool packages are enabled."""
        return bool(
            self._get_shared_setting(
                CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                DEFAULT_ENABLE_EXTERNAL_CUSTOM_TOOLS,
            )
        )

    def _get_search_provider(self) -> str:
        """Get search provider (shared setting) with backward compatibility."""
        provider = self._get_shared_setting(CONF_SEARCH_PROVIDER)
        if provider:
            explicit_enabled = self._get_shared_setting(CONF_ENABLE_WEB_SEARCH)
            if explicit_enabled is False:
                return "none"
            return provider

        # Backward compat: older versions used enable_custom_tools for Brave search.
        if self._get_shared_setting(CONF_ENABLE_CUSTOM_TOOLS, False):
            return "brave"

        return "none"

    def _get_brave_api_key(self) -> str:
        """Get Brave API key (shared setting)."""
        return self._get_shared_setting(CONF_BRAVE_API_KEY, DEFAULT_BRAVE_API_KEY)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get MCP tool definitions for all enabled tools."""
        definitions: list[dict[str, Any]] = []
        external_tool_definitions = {
            loaded_tool.manifest.tool_id: loaded_tool.tool_definitions
            for loaded_tool in self.external_tools
        }
        for tool_key, tool in self.tools.items():
            try:
                if tool_key in external_tool_definitions:
                    definitions.extend(external_tool_definitions[tool_key])
                else:
                    definitions.extend(tool.get_tool_definitions())
            except Exception as err:
                _LOGGER.error("Error getting tool definitions from %s: %s", tool_key, err)
        return definitions

    async def handle_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle a custom tool call."""
        for tool in self.tools.values():
            if tool.handles_tool(tool_name):
                return await tool.handle_call(tool_name, arguments)

        raise ValueError(f"Unknown custom tool: {tool_name}")

    def is_custom_tool(self, tool_name: str) -> bool:
        """Check if a tool name is provided by the custom tool loader."""
        return any(tool.handles_tool(tool_name) for tool in self.tools.values())

    def get_external_prompt_instructions(self) -> str:
        """Return aggregated prompt additions from loaded external tool packages."""
        return self._external_prompt_instructions

    def get_loaded_external_tool_info(self) -> list[dict[str, Any]]:
        """Return metadata for loaded external tool packages."""
        return [
            {
                "id": loaded_tool.manifest.tool_id,
                "name": loaded_tool.manifest.name,
                "version": loaded_tool.manifest.version,
                "description": loaded_tool.manifest.description,
                "tool_names": list(loaded_tool.tool_names),
                "capabilities": list(loaded_tool.manifest.capabilities),
            }
            for loaded_tool in self.external_tools
        ]
