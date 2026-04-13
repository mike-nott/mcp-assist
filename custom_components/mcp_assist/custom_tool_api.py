"""Public API for user-defined MCP Assist custom tools.

This module is intentionally stable and documented so local tools placed under
the Home Assistant config directory can import it directly:

    <home-assistant-config>/mcp-assist-tools/<tool_id>/tool.py

Tool-package metadata should live in:

    <home-assistant-config>/mcp-assist-tools/<tool_id>/mcp_tool.json
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
from typing import Any

from homeassistant.core import HomeAssistant

from .const import CUSTOM_TOOL_SHARED_DIRECTORY, CUSTOM_TOOLS_DIRECTORY, DOMAIN

_CURRENT_EXTERNAL_TOOL_CALL_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "mcp_assist_external_tool_call_context",
    default=None,
)


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
    - get_settings_schema()
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
        self.settings: dict[str, Any] = {}

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

    def get_settings_schema(self) -> dict[str, Any]:
        """Return an optional shared-settings schema for this package.

        When provided, MCP Assist will load package settings from
        `<config>/mcp-assist-tool-settings/<tool_id>.json`, validate them, and
        make the merged shared/profile settings available via `get_settings()`.
        """

        return {}

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

    def get_settings(self) -> dict[str, Any]:
        """Return the effective settings for the current tool call."""
        context = self.get_call_context()
        settings = context.get("settings") if isinstance(context, dict) else None
        if isinstance(settings, dict):
            return dict(settings)
        return dict(self.settings)

    def get_shared_settings(self) -> dict[str, Any]:
        """Return the package's shared settings."""
        return dict(self.settings)

    def get_profile_settings(self) -> dict[str, Any]:
        """Return the current profile override settings, if any."""
        context = self.get_call_context()
        profile_settings = (
            context.get("profile_settings") if isinstance(context, dict) else None
        )
        if isinstance(profile_settings, dict):
            return dict(profile_settings)
        return {}

    def get_call_context(self) -> dict[str, Any]:
        """Return MCP Assist metadata for the current tool call."""
        return get_external_tool_call_context()

    async def call_mcp_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke any MCP Assist tool using the current profile call context."""
        return await call_mcp_tool(
            self.hass,
            tool_name,
            arguments,
            context=self.get_call_context(),
        )

    def _set_loaded_settings(self, settings: dict[str, Any]) -> None:
        """Internal helper used by the loader to attach validated settings."""
        self.settings = dict(settings or {})

    def _push_call_context(self, context: dict[str, Any] | None) -> Token:
        """Internal helper used by the loader to scope call metadata."""
        return _CURRENT_EXTERNAL_TOOL_CALL_CONTEXT.set(dict(context or {}))

    def _reset_call_context(self, token: Token) -> None:
        """Internal helper used by the loader to restore prior call metadata."""
        _CURRENT_EXTERNAL_TOOL_CALL_CONTEXT.reset(token)


def load_external_shared_module(
    caller_file: str | Path,
    module_name: str,
) -> Any:
    """Load a reusable helper module from `<config>/mcp-assist-tools/__shared__`.

    This lets multiple narrow external tool packages share code without custom
    `sys.path` shims.
    """

    caller_path = Path(caller_file).resolve()
    tools_root = _find_external_tools_root(caller_path)
    shared_root = (tools_root / CUSTOM_TOOL_SHARED_DIRECTORY).resolve()
    if not shared_root.is_dir():
        raise ImportError(
            f"Shared helper directory does not exist: {shared_root}"
        )

    relative_path = Path(*module_name.split("."))
    module_path = (shared_root / relative_path).with_suffix(".py")
    if not module_path.is_file():
        module_path = shared_root / relative_path / "__init__.py"
    if not module_path.is_file():
        raise ImportError(
            f"Unable to resolve shared helper module {module_name!r} in {shared_root}"
        )

    try:
        module_path.resolve().relative_to(shared_root)
    except ValueError as err:
        raise ImportError("Shared helper module must stay within __shared__") from err

    unique_module_name = (
        f"mcp_assist_external_tools.shared.{module_name.replace('.', '_')}"
    )
    module = sys.modules.get(unique_module_name)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(unique_module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import shared helper from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(unique_module_name, None)
        raise
    return module


def get_external_tool_call_context() -> dict[str, Any]:
    """Return the current scoped external-tool call context."""
    context = _CURRENT_EXTERNAL_TOOL_CALL_CONTEXT.get()
    if isinstance(context, dict):
        return dict(context)
    return {}


async def call_mcp_tool(
    hass: HomeAssistant,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke any MCP Assist tool through the shared MCP server."""
    server = hass.data.get(DOMAIN, {}).get("shared_mcp_server")
    if server is None:
        raise RuntimeError("Shared MCP server is not running.")

    handle_tool_call = getattr(server, "handle_tool_call", None)
    if not callable(handle_tool_call):
        raise RuntimeError(
            "This MCP Assist build does not support tool invocation from external packages."
        )

    return await handle_tool_call(
        {
            "name": tool_name,
            "arguments": dict(arguments or {}),
            "context": dict(context or {}),
        }
    )


def _find_external_tools_root(caller_path: Path) -> Path:
    for candidate in [caller_path.parent, *caller_path.parents]:
        if candidate.name == CUSTOM_TOOLS_DIRECTORY:
            return candidate
    raise ImportError(
        f"Unable to locate {CUSTOM_TOOLS_DIRECTORY!r} from {caller_path}"
    )
