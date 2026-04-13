"""Built-in and external custom tool loading for MCP Assist."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

from ..custom_tool_api import MCPAssistExternalTool
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
from .external_loader import (
    ExternalCustomToolLoader,
    LoadedToolPackage,
    combine_prompt_instructions,
)
from .schema_utils import SchemaValidationError, validate_and_normalize_json_value

_LOGGER = logging.getLogger(__name__)


@dataclass
class _RegisteredTool:
    """Validated custom-tool registry entry."""

    name: str
    definition: dict[str, Any]
    instance: Any
    provider_key: str
    is_external: bool
    loaded_tool_package: LoadedToolPackage | None = None
    package_loader: ExternalCustomToolLoader | None = None


class CustomToolsLoader:
    """Load and manage built-in and user-defined custom tools."""

    def __init__(self, hass, entry=None):
        """Initialize the custom tools loader."""
        self.hass = hass
        self.entry = entry
        self.tools: dict[str, Any] = {}
        self.builtin_packages: list[LoadedToolPackage] = []
        self.external_tools: list[LoadedToolPackage] = []
        self._builtin_loader = ExternalCustomToolLoader(
            self.hass,
            tools_root=Path(__file__).resolve().parent / "packages",
            module_namespace="mcp_assist_builtin_tools",
            require_tool_name_prefix=False,
            package_log_label="built-in tool package",
            prompt_package_label="Built-in tool package",
        )
        self._external_loader = ExternalCustomToolLoader(self.hass)
        self._external_prompt_instructions = ""
        self._tool_registry: dict[str, _RegisteredTool] = {}
        self._tool_definitions_cache: list[dict[str, Any]] = []

    async def initialize(self):
        """Initialize built-in bundles and optional external tool packages."""
        await self._initialize_builtin_tools()
        await self._initialize_external_tools()
        self._refresh_tool_registry()

    async def _initialize_builtin_tools(self) -> None:
        """Load built-in tool bundles that ship with MCP Assist."""
        await self._shutdown_builtin_tools()

        selected_tool_ids: set[str] = set()
        calculator_enabled = self._get_shared_setting(
            CONF_ENABLE_CALCULATOR_TOOLS, DEFAULT_ENABLE_CALCULATOR_TOOLS
        )
        unit_conversion_enabled = self._get_shared_setting(
            CONF_ENABLE_UNIT_CONVERSION_TOOLS, None
        )
        if unit_conversion_enabled is None:
            unit_conversion_enabled = calculator_enabled

        if calculator_enabled or unit_conversion_enabled:
            selected_tool_ids.add("calculator")
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
        elif search_provider in {"brave", "duckduckgo"}:
            selected_tool_ids.update({"search", "read_url"})
        else:
            _LOGGER.debug("Web search enabled, but no supported search provider is configured")

        if not selected_tool_ids:
            return

        self.builtin_packages = await self._builtin_loader.load(
            allowed_tool_ids=selected_tool_ids,
        )
        for loaded_tool in self.builtin_packages:
            self.tools[loaded_tool.manifest.tool_id] = loaded_tool.instance

        if self.builtin_packages:
            _LOGGER.info(
                "✅ Loaded %d built-in tool package(s)",
                len(self.builtin_packages),
            )

    async def _initialize_external_tools(self) -> None:
        """Load opt-in user-defined tool packages from the HA config directory."""
        await self._shutdown_external_tools()

        if not self._external_custom_tools_enabled():
            _LOGGER.debug("External custom tools disabled in shared MCP settings")
            self.external_tools = []
            self._external_prompt_instructions = ""
            return

        reserved_tool_names = {
            str(tool_definition.get("name") or "")
            for tool_definition in self._get_builtin_tool_definitions()
        }
        reserved_tool_ids = {
            loaded_tool.manifest.tool_id for loaded_tool in self.builtin_packages
        }
        self.external_tools = await self._external_loader.load(
            reserved_tool_names=reserved_tool_names,
            reserved_tool_ids=reserved_tool_ids,
        )
        for loaded_tool in self.external_tools:
            self.tools[loaded_tool.manifest.tool_id] = loaded_tool.instance

        self._external_prompt_instructions = combine_prompt_instructions(
            self.external_tools
        )
        if self.external_tools:
            _LOGGER.info(
                "✅ Loaded %d external custom tool package(s) from %s",
                len(self.external_tools),
                self._external_loader.get_tools_root(),
            )
        else:
            _LOGGER.info("No external custom tool packages were loaded")

    async def shutdown(self) -> None:
        """Shut down any loaded tool packages cleanly."""
        await self._shutdown_builtin_tools()
        await self._shutdown_external_tools()

    async def _shutdown_builtin_tools(self) -> None:
        """Shut down the currently loaded built-in tool packages."""
        for loaded_tool in self.builtin_packages:
            try:
                await loaded_tool.instance.async_shutdown()
            except Exception as err:
                _LOGGER.warning(
                    "Built-in tool package %s failed during shutdown: %s",
                    loaded_tool.manifest.tool_id,
                    err,
                )
            finally:
                self.tools.pop(loaded_tool.manifest.tool_id, None)

        self.builtin_packages = []

    async def _shutdown_external_tools(self) -> None:
        """Shut down only the currently loaded external custom tool packages."""
        for loaded_tool in self.external_tools:
            try:
                await loaded_tool.instance.async_shutdown()
            except Exception as err:
                _LOGGER.warning(
                    "External custom tool %s failed during shutdown: %s",
                    loaded_tool.manifest.tool_id,
                    err,
                )
            finally:
                self.tools.pop(loaded_tool.manifest.tool_id, None)

        self.external_tools = []
        self._external_prompt_instructions = ""

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
        if not self._tool_definitions_cache:
            self._refresh_tool_registry()
        return list(self._tool_definitions_cache)

    async def handle_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Handle a custom tool call."""
        registry_entry = self._tool_registry.get(tool_name)
        if registry_entry is None:
            raise ValueError(f"Unknown custom tool: {tool_name}")

        normalized_arguments = dict(arguments or {})
        if registry_entry.loaded_tool_package is not None:
            try:
                normalized_arguments = validate_and_normalize_json_value(
                    registry_entry.definition.get("inputSchema") or {},
                    normalized_arguments,
                    path=f"{tool_name} arguments",
                )
            except SchemaValidationError as err:
                return self._build_error_result(str(err))

            return await self._handle_package_tool_call(
                registry_entry,
                tool_name,
                normalized_arguments,
                context=context,
            )

        return await registry_entry.instance.handle_call(tool_name, normalized_arguments)

    def is_custom_tool(self, tool_name: str) -> bool:
        """Check if a tool name is provided by the custom tool loader."""
        return tool_name in self._tool_registry

    def is_external_custom_tool(self, tool_name: str) -> bool:
        """Check if a tool name comes from an external custom tool package."""
        registry_entry = self._tool_registry.get(tool_name)
        return bool(registry_entry and registry_entry.is_external)

    def get_external_prompt_instructions(self) -> str:
        """Return aggregated prompt additions from loaded external tool packages."""
        return self._external_prompt_instructions

    def get_cache_signature(self) -> tuple[Any, ...]:
        """Return a stable cache signature for the current tool surface."""
        try:
            tool_definitions = tuple(
                json.dumps(tool_definition, sort_keys=True, separators=(",", ":"))
                for tool_definition in self.get_tool_definitions()
            )
        except Exception as err:
            _LOGGER.debug("Unable to serialize tool definitions for cache key: %s", err)
            tool_definitions = ()

        try:
            external_prompt_instructions = self.get_external_prompt_instructions()
        except Exception as err:
            _LOGGER.debug(
                "Unable to read external prompt instructions for cache key: %s", err
            )
            external_prompt_instructions = ""

        return (tool_definitions, external_prompt_instructions)

    async def reload_external_tools(self) -> dict[str, Any]:
        """Reload external custom-tool packages and return diagnostics."""
        await self._initialize_external_tools()
        self._refresh_tool_registry()
        return self.get_external_diagnostics()

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

    def get_external_diagnostics(self) -> dict[str, Any]:
        """Return detailed diagnostics for external custom-tool loading."""
        return {
            "enabled": self._external_custom_tools_enabled(),
            "tools_root": str(self._external_loader.get_tools_root()),
            "settings_root": str(self._external_loader.get_settings_root()),
            "last_loaded_at": self._external_loader.last_loaded_at,
            "scanned_packages": list(self._external_loader.last_scanned_packages),
            "load_errors": list(self._external_loader.last_load_errors),
            "loaded_tools": [
                {
                    "id": loaded_tool.manifest.tool_id,
                    "name": loaded_tool.manifest.name,
                    "version": loaded_tool.manifest.version,
                    "tool_names": list(loaded_tool.tool_names),
                    "capabilities": list(loaded_tool.manifest.capabilities),
                    "shared_settings_path": loaded_tool.shared_settings_path,
                    "has_settings_schema": bool(loaded_tool.settings_schema),
                }
                for loaded_tool in self.external_tools
            ],
        }

    def _get_builtin_tool_definitions(self) -> list[dict[str, Any]]:
        """Return built-in MCP tool definitions for reserved-name checks."""
        builtin_definitions: list[dict[str, Any]] = []
        for loaded_tool in self.builtin_packages:
            builtin_definitions.extend(loaded_tool.tool_definitions)
        return builtin_definitions

    def _refresh_tool_registry(self) -> None:
        """Build a validated tool-name registry for dispatch and caching."""
        registry: dict[str, _RegisteredTool] = {}
        ordered_definitions: list[dict[str, Any]] = []
        packaged_tool_keys = {
            loaded_tool.manifest.tool_id
            for loaded_tool in [*self.builtin_packages, *self.external_tools]
        }

        for tool_key, tool in self.tools.items():
            if tool_key in packaged_tool_keys:
                continue
            try:
                tool_definitions = tool.get_tool_definitions()
            except Exception as err:
                _LOGGER.error("Error getting tool definitions from %s: %s", tool_key, err)
                continue

            for tool_definition in tool_definitions:
                tool_name = str(tool_definition.get("name") or "")
                if not tool_name:
                    continue
                if tool_name in registry:
                    _LOGGER.warning("Skipping duplicate custom tool definition %s", tool_name)
                    continue
                registry[tool_name] = _RegisteredTool(
                    name=tool_name,
                    definition=tool_definition,
                    instance=tool,
                    provider_key=tool_key,
                    is_external=False,
                )
                ordered_definitions.append(tool_definition)

        for loaded_tool in self.builtin_packages:
            for tool_definition in loaded_tool.tool_definitions:
                tool_name = str(tool_definition.get("name") or "")
                if not tool_name:
                    continue
                if tool_name in registry:
                    _LOGGER.warning(
                        "Skipping duplicate built-in custom tool definition %s",
                        tool_name,
                    )
                    continue
                registry[tool_name] = _RegisteredTool(
                    name=tool_name,
                    definition=tool_definition,
                    instance=loaded_tool.instance,
                    provider_key=loaded_tool.manifest.tool_id,
                    is_external=False,
                    loaded_tool_package=loaded_tool,
                    package_loader=self._builtin_loader,
                )
                ordered_definitions.append(tool_definition)

        for loaded_tool in self.external_tools:
            for tool_definition in loaded_tool.tool_definitions:
                tool_name = str(tool_definition.get("name") or "")
                if not tool_name:
                    continue
                if tool_name in registry:
                    _LOGGER.warning(
                        "Skipping duplicate external custom tool definition %s",
                        tool_name,
                    )
                    continue
                registry[tool_name] = _RegisteredTool(
                    name=tool_name,
                    definition=tool_definition,
                    instance=loaded_tool.instance,
                    provider_key=loaded_tool.manifest.tool_id,
                    is_external=True,
                    loaded_tool_package=loaded_tool,
                    package_loader=self._external_loader,
                )
                ordered_definitions.append(tool_definition)

        self._tool_registry = registry
        self._tool_definitions_cache = ordered_definitions

    async def _handle_package_tool_call(
        self,
        registry_entry: _RegisteredTool,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a manifest-based tool package with scoped call context."""
        loaded_tool = registry_entry.loaded_tool_package
        package_loader = registry_entry.package_loader
        instance = registry_entry.instance
        if (
            loaded_tool is None
            or package_loader is None
            or not isinstance(instance, MCPAssistExternalTool)
        ):
            return self._build_error_result(
                f"Tool package for {tool_name!r} is not available right now."
            )

        call_context = dict(context or {})
        profile_entry_id = str(call_context.get("profile_entry_id") or "").strip()
        try:
            (
                effective_settings,
                profile_settings,
                profile_settings_path,
            ) = await package_loader.load_profile_settings(
                loaded_tool,
                profile_entry_id,
            )
        except Exception as err:
            _LOGGER.error(
                "Failed to load settings for %s %s: %s",
                "external custom tool" if registry_entry.is_external else "built-in tool package",
                tool_name,
                err,
            )
            return self._build_error_result(str(err))

        call_context.update(
            {
                "tool_id": loaded_tool.manifest.tool_id,
                "tool_name": tool_name,
                "settings": effective_settings,
                "shared_settings": loaded_tool.shared_settings,
                "profile_settings": profile_settings,
                "shared_settings_path": loaded_tool.shared_settings_path,
                "profile_settings_path": profile_settings_path,
            }
        )

        token = instance._push_call_context(call_context)
        try:
            return await instance.handle_call(tool_name, arguments)
        except Exception as err:
            _LOGGER.error("Error executing tool package %s: %s", tool_name, err)
            return self._build_error_result(str(err))
        finally:
            instance._reset_call_context(token)

    @staticmethod
    def _build_error_result(message: str) -> dict[str, Any]:
        """Return a standard MCP error payload."""
        return {
            "isError": True,
            "content": [{"type": "text", "text": str(message)}],
        }
