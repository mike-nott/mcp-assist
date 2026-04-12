"""Loader for user-defined MCP Assist custom tools."""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..const import (
    CUSTOM_TOOL_MANIFEST_FILENAME,
    CUSTOM_TOOL_SCHEMA_VERSION,
    CUSTOM_TOOLS_DIRECTORY,
)
from ..custom_tool_api import MCPAssistCustomToolManifest, MCPAssistExternalTool

_LOGGER = logging.getLogger(__name__)

_TOOL_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_PROMPT_CHARS_PER_TOOL = 1600
_MAX_PROMPT_CHARS_TOTAL = 5000


@dataclass
class LoadedExternalTool:
    """Loaded and validated external custom tool package."""

    manifest: MCPAssistCustomToolManifest
    instance: MCPAssistExternalTool
    tool_definitions: tuple[dict[str, Any], ...]
    tool_names: tuple[str, ...]
    prompt_instructions: str


class ExternalCustomToolLoader:
    """Load user-defined custom tool packages from the HA config directory."""

    def __init__(self, hass) -> None:
        """Initialize the loader."""
        self.hass = hass

    def get_tools_root(self) -> Path:
        """Return the shared custom tools directory inside HA config."""
        return Path(self.hass.config.path(CUSTOM_TOOLS_DIRECTORY))

    async def load(
        self, *, reserved_tool_names: set[str] | None = None
    ) -> list[LoadedExternalTool]:
        """Load and validate all external custom tool packages."""
        reserved = set(reserved_tool_names or set())
        seen_tool_ids: set[str] = set()
        tools_root = self.get_tools_root()
        if not tools_root.exists():
            _LOGGER.info(
                "External custom tools enabled, but %s does not exist yet.",
                tools_root,
            )
            return []

        if not tools_root.is_dir():
            _LOGGER.warning(
                "External custom tools path %s exists but is not a directory. Skipping.",
                tools_root,
            )
            return []

        loaded: list[LoadedExternalTool] = []
        for tool_dir in sorted(tools_root.iterdir(), key=lambda item: item.name.casefold()):
            if tool_dir.name.startswith((".", "__")):
                continue
            if tool_dir.is_symlink():
                _LOGGER.warning(
                    "Skipping external custom tool package %s because symlinked directories are not allowed.",
                    tool_dir,
                )
                continue
            if not tool_dir.is_dir():
                continue

            tool: MCPAssistExternalTool | None = None
            try:
                manifest = self._load_manifest(tool_dir)
                if manifest.tool_id in seen_tool_ids:
                    raise ValueError(
                        f"Duplicate manifest id {manifest.tool_id!r} is not allowed"
                    )
                tool = self._instantiate_tool(tool_dir, manifest)
                await tool.initialize()
                tool_definitions = self._normalize_tool_definitions(
                    manifest,
                    tool.get_tool_definitions(),
                    reserved_tool_names=reserved,
                )
                tool_names = tuple(
                    str(tool_definition["name"]) for tool_definition in tool_definitions
                )
                reserved.update(tool_names)
                prompt_instructions = self._build_prompt_instructions(
                    tool_dir,
                    manifest,
                    tool,
                )
                loaded.append(
                    LoadedExternalTool(
                        manifest=manifest,
                        instance=tool,
                        tool_definitions=tuple(tool_definitions),
                        tool_names=tool_names,
                        prompt_instructions=prompt_instructions,
                    )
                )
                seen_tool_ids.add(manifest.tool_id)
                _LOGGER.info(
                    "Loaded external custom tool package %s with %d tool(s)",
                    manifest.tool_id,
                    len(tool_names),
                )
            except Exception as err:
                if tool is not None:
                    try:
                        await tool.async_shutdown()
                    except Exception as shutdown_err:
                        _LOGGER.debug(
                            "External custom tool %s also failed during cleanup: %s",
                            getattr(getattr(tool, "manifest", None), "tool_id", tool_dir.name),
                            shutdown_err,
                        )
                _LOGGER.error(
                    "Failed to load external custom tool package from %s: %s",
                    tool_dir,
                    err,
                )

        return loaded

    def _load_manifest(self, tool_dir: Path) -> MCPAssistCustomToolManifest:
        """Load and validate a custom tool manifest."""
        manifest_path = tool_dir / CUSTOM_TOOL_MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ValueError(f"{CUSTOM_TOOL_MANIFEST_FILENAME} is required")

        try:
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as err:
            raise ValueError(f"Invalid {manifest_path.name}: {err}") from err

        if not isinstance(raw_manifest, dict):
            raise ValueError(f"{manifest_path.name} must contain an object")

        schema_version = raw_manifest.get("schema_version")
        if schema_version != CUSTOM_TOOL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {schema_version!r}; expected {CUSTOM_TOOL_SCHEMA_VERSION}"
            )

        tool_id = str(raw_manifest.get("id") or "").strip()
        if not _TOOL_ID_PATTERN.match(tool_id):
            raise ValueError(
                "manifest id must contain only lowercase letters, numbers, and underscores"
            )
        if tool_dir.name != tool_id:
            raise ValueError(
                f"tool directory name must match manifest id ({tool_id!r})"
            )

        name = str(raw_manifest.get("name") or "").strip()
        description = str(raw_manifest.get("description") or "").strip()
        version = str(raw_manifest.get("version") or "").strip()
        entrypoint = str(raw_manifest.get("entrypoint") or "").strip()

        if not name:
            raise ValueError("manifest name is required")
        if not description:
            raise ValueError("manifest description is required")
        if not version:
            raise ValueError("manifest version is required")
        if ":" not in entrypoint:
            raise ValueError("entrypoint must be in 'module:ClassName' format")

        raw_capabilities = raw_manifest.get("capabilities") or []
        if not isinstance(raw_capabilities, list):
            raise ValueError("manifest capabilities must be a list of strings")
        capabilities = tuple(
            str(item).strip()
            for item in raw_capabilities
            if str(item).strip()
        )

        prompt_append_file = raw_manifest.get("prompt_append_file")
        if prompt_append_file is not None:
            prompt_append_file = str(prompt_append_file).strip() or None

        return MCPAssistCustomToolManifest(
            schema_version=CUSTOM_TOOL_SCHEMA_VERSION,
            tool_id=tool_id,
            name=name,
            description=description,
            version=version,
            entrypoint=entrypoint,
            capabilities=capabilities,
            prompt_append_file=prompt_append_file,
        )

    def _instantiate_tool(
        self,
        tool_dir: Path,
        manifest: MCPAssistCustomToolManifest,
    ) -> MCPAssistExternalTool:
        """Import and instantiate a tool class from a package entrypoint."""
        module_name, class_name = manifest.entrypoint.split(":", 1)
        module_name = module_name.strip()
        class_name = class_name.strip()
        if not module_name or not class_name:
            raise ValueError("entrypoint must include both module and class name")

        module_path = self._resolve_module_path(tool_dir, module_name)
        try:
            module_path.resolve().relative_to(tool_dir.resolve())
        except ValueError as err:
            raise ValueError("entrypoint module must stay within the tool directory") from err
        unique_module_name = (
            f"mcp_assist_external_tools.{manifest.tool_id}.{module_name.replace('.', '_')}"
        )
        spec = importlib.util.spec_from_file_location(unique_module_name, module_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Unable to import module from {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(unique_module_name, None)
            raise

        tool_class = getattr(module, class_name, None)
        if tool_class is None:
            raise ValueError(f"Entrypoint class {class_name!r} not found")
        if not isinstance(tool_class, type) or not issubclass(
            tool_class, MCPAssistExternalTool
        ):
            raise ValueError(
                f"{class_name!r} must subclass MCPAssistExternalTool"
            )

        return tool_class(self.hass, manifest, tool_dir)

    def _resolve_module_path(self, tool_dir: Path, module_name: str) -> Path:
        """Resolve an entrypoint module inside the tool package directory."""
        relative_path = Path(*module_name.split("."))
        file_path = (tool_dir / relative_path).with_suffix(".py")
        if file_path.is_file():
            return file_path

        package_init = tool_dir / relative_path / "__init__.py"
        if package_init.is_file():
            return package_init

        raise ValueError(f"Unable to resolve entrypoint module {module_name!r}")

    def _normalize_tool_definitions(
        self,
        manifest: MCPAssistCustomToolManifest,
        tool_definitions: Any,
        *,
        reserved_tool_names: set[str],
    ) -> list[dict[str, Any]]:
        """Validate, normalize, and de-duplicate external tool definitions."""
        if not isinstance(tool_definitions, list) or not tool_definitions:
            raise ValueError("get_tool_definitions() must return a non-empty list")

        normalized: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for tool_definition in tool_definitions:
            if not isinstance(tool_definition, dict):
                raise ValueError("Each tool definition must be an object")

            tool_name = str(tool_definition.get("name") or "").strip()
            if not _TOOL_NAME_PATTERN.match(tool_name):
                raise ValueError(
                    f"Tool name {tool_name!r} contains unsupported characters"
                )
            if not tool_name.startswith(f"{manifest.tool_id}_"):
                raise ValueError(
                    f"Tool name {tool_name!r} must be prefixed with '{manifest.tool_id}_'"
                )
            if tool_name in reserved_tool_names or tool_name in seen_names:
                raise ValueError(
                    f"Tool name {tool_name!r} conflicts with an existing tool"
                )

            description = str(tool_definition.get("description") or "").strip()
            if not description:
                raise ValueError(f"Tool {tool_name!r} is missing a description")

            input_schema = tool_definition.get("inputSchema") or {}
            if not isinstance(input_schema, dict):
                raise ValueError(
                    f"Tool {tool_name!r} inputSchema must be an object"
                )
            normalized_schema = dict(input_schema)
            if not normalized_schema:
                normalized_schema = {"type": "object", "properties": {}}
            elif normalized_schema.get("type") == "object" and "properties" not in normalized_schema:
                normalized_schema["properties"] = {}

            normalized.append(
                {
                    **tool_definition,
                    "name": tool_name,
                    "description": description,
                    "inputSchema": normalized_schema,
                }
            )
            seen_names.add(tool_name)

        return normalized

    def _build_prompt_instructions(
        self,
        tool_dir: Path,
        manifest: MCPAssistCustomToolManifest,
        tool: MCPAssistExternalTool,
    ) -> str:
        """Build a compact prompt appendix for an external tool package."""
        parts: list[str] = [
            f"Custom tool package '{manifest.name}' ({manifest.tool_id}) is enabled."
        ]

        if manifest.capabilities:
            parts.append("Capabilities:")
            parts.extend(
                f"- {self._compact_text(capability, max_len=180)}"
                for capability in manifest.capabilities
            )

        prompt_file_text = self._read_prompt_append_file(tool_dir, manifest)
        runtime_instructions = str(tool.get_prompt_instructions() or "").strip()
        if prompt_file_text:
            parts.append(prompt_file_text)
        if runtime_instructions:
            parts.append(runtime_instructions)

        combined = "\n".join(part for part in parts if part).strip()
        if not combined:
            return ""
        if len(combined) > _MAX_PROMPT_CHARS_PER_TOOL:
            combined = combined[:_MAX_PROMPT_CHARS_PER_TOOL].rstrip()
            combined += "\n\n[Prompt appendix truncated to keep context small.]"
        return combined

    def _read_prompt_append_file(
        self,
        tool_dir: Path,
        manifest: MCPAssistCustomToolManifest,
    ) -> str:
        """Read optional static prompt instructions from the package."""
        if not manifest.prompt_append_file:
            return ""

        prompt_path = (tool_dir / manifest.prompt_append_file).resolve()
        try:
            prompt_path.relative_to(tool_dir.resolve())
        except ValueError as err:
            raise ValueError("prompt_append_file must stay within the tool directory") from err

        if not prompt_path.is_file():
            raise ValueError(
                f"prompt_append_file {manifest.prompt_append_file!r} was not found"
            )

        return prompt_path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _compact_text(text: str, *, max_len: int) -> str:
        """Compact capability text for prompt use."""
        normalized = " ".join(str(text).split()).strip()
        if len(normalized) <= max_len:
            return normalized
        trimmed = normalized[: max_len - 1].rstrip()
        last_space = trimmed.rfind(" ")
        if last_space > 40:
            trimmed = trimmed[:last_space]
        return trimmed.rstrip(" ,;:.") + "."


def combine_prompt_instructions(loaded_tools: list[LoadedExternalTool]) -> str:
    """Combine external-tool prompt appendices into one compact block."""
    prompt_sections = [
        tool.prompt_instructions.strip()
        for tool in loaded_tools
        if tool.prompt_instructions.strip()
    ]
    if not prompt_sections:
        return ""

    combined = "## External Custom Tools\n" + "\n\n".join(prompt_sections)
    if len(combined) <= _MAX_PROMPT_CHARS_TOTAL:
        return combined

    truncated = combined[:_MAX_PROMPT_CHARS_TOTAL].rstrip()
    return truncated + "\n\n[External custom tool instructions truncated.]"
