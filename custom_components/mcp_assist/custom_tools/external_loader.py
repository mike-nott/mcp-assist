"""Loader for manifest-based MCP Assist tool packages."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
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
    CUSTOM_TOOL_SETTINGS_DIRECTORY,
    CUSTOM_TOOLS_DIRECTORY,
)
from ..custom_tool_api import MCPAssistCustomToolManifest, MCPAssistExternalTool
from .schema_utils import SchemaValidationError, validate_and_normalize_json_value

_LOGGER = logging.getLogger(__name__)

_TOOL_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_PROMPT_CHARS_PER_TOOL = 220
_MAX_PROMPT_CHARS_TOTAL = 2200


@dataclass
class LoadedToolPackage:
    """Loaded and validated MCP Assist tool package."""

    manifest: MCPAssistCustomToolManifest
    instance: MCPAssistExternalTool
    tool_definitions: tuple[dict[str, Any], ...]
    tool_names: tuple[str, ...]
    prompt_instructions: str
    settings_schema: dict[str, Any]
    shared_settings: dict[str, Any]
    shared_settings_path: str | None


LoadedExternalTool = LoadedToolPackage


class ExternalCustomToolLoader:
    """Load manifest-based tool packages from a configured root directory."""

    def __init__(
        self,
        hass,
        *,
        tools_root: Path | None = None,
        settings_root: Path | None = None,
        module_namespace: str = "mcp_assist_external_tools",
        require_tool_name_prefix: bool = True,
        package_log_label: str = "external custom tool package",
        prompt_package_label: str = "Custom tool package",
    ) -> None:
        """Initialize the loader."""
        self.hass = hass
        self._tools_root = tools_root
        self._settings_root = settings_root
        self._module_namespace = module_namespace
        self._require_tool_name_prefix = require_tool_name_prefix
        self._package_log_label = package_log_label
        self._prompt_package_label = prompt_package_label
        self.last_load_errors: list[dict[str, str]] = []
        self.last_tools_root: Path = self.get_tools_root()
        self.last_scanned_packages: tuple[str, ...] = ()
        self.last_loaded_at: str = ""

    def get_tools_root(self) -> Path:
        """Return the package root directory for this loader."""
        if self._tools_root is not None:
            return self._tools_root
        return Path(self.hass.config.path(CUSTOM_TOOLS_DIRECTORY))

    def get_settings_root(self) -> Path:
        """Return the package-settings directory for this loader."""
        if self._settings_root is not None:
            return self._settings_root
        return Path(self.hass.config.path(CUSTOM_TOOL_SETTINGS_DIRECTORY))

    async def load(
        self,
        *,
        reserved_tool_names: set[str] | None = None,
        reserved_tool_ids: set[str] | None = None,
        allowed_tool_ids: set[str] | None = None,
    ) -> list[LoadedToolPackage]:
        """Load and validate all matching tool packages."""
        reserved = set(reserved_tool_names or set())
        reserved_ids = set(reserved_tool_ids or set())
        seen_tool_ids: set[str] = set()
        tools_root = self.get_tools_root()
        self.last_tools_root = tools_root
        self.last_load_errors = []
        self.last_loaded_at = datetime.now(timezone.utc).isoformat()
        if not await self.hass.async_add_executor_job(tools_root.exists):
            _LOGGER.info(
                "External custom tools enabled, but %s does not exist yet.",
                tools_root,
            )
            self.last_scanned_packages = ()
            return []

        if not await self.hass.async_add_executor_job(tools_root.is_dir):
            _LOGGER.warning(
                "External custom tools path %s exists but is not a directory. Skipping.",
                tools_root,
            )
            self.last_scanned_packages = ()
            return []

        loaded: list[LoadedToolPackage] = []
        package_dirs = await self.hass.async_add_executor_job(
            self._discover_package_dirs,
            tools_root,
        )
        self.last_scanned_packages = tuple(
            tool_dir.name for tool_dir, _is_symlink in package_dirs
        )
        for tool_dir, is_symlink in package_dirs:
            if allowed_tool_ids is not None and tool_dir.name not in allowed_tool_ids:
                continue

            if is_symlink:
                _LOGGER.warning(
                    "Skipping %s %s because symlinked directories are not allowed.",
                    self._package_log_label,
                    tool_dir,
                )
                self.last_load_errors.append(
                    {"tool_dir": tool_dir.name, "error": "Symlinked package directories are not allowed"}
                )
                continue

            tool: MCPAssistExternalTool | None = None
            try:
                manifest = await self._load_manifest(tool_dir)
                if manifest.tool_id in reserved_ids or manifest.tool_id in seen_tool_ids:
                    raise ValueError(
                        f"Manifest id {manifest.tool_id!r} conflicts with an existing tool package"
                    )
                tool = self._instantiate_tool(tool_dir, manifest)
                settings_schema = self._normalize_settings_schema(
                    manifest,
                    tool.get_settings_schema(),
                )
                shared_settings, shared_settings_path = await self._load_shared_settings(
                    manifest,
                    settings_schema,
                )
                tool._set_loaded_settings(shared_settings)
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
                prompt_instructions = await self._build_prompt_instructions(
                    tool_dir,
                    manifest,
                    tool,
                    tool_names,
                )
                loaded.append(
                    LoadedToolPackage(
                        manifest=manifest,
                        instance=tool,
                        tool_definitions=tuple(tool_definitions),
                        tool_names=tool_names,
                        prompt_instructions=prompt_instructions,
                        settings_schema=settings_schema,
                        shared_settings=shared_settings,
                        shared_settings_path=shared_settings_path,
                    )
                )
                seen_tool_ids.add(manifest.tool_id)
                _LOGGER.info(
                    "Loaded %s %s with %d tool(s)",
                    self._package_log_label,
                    manifest.tool_id,
                    len(tool_names),
                )
            except Exception as err:
                if tool is not None:
                    try:
                        await tool.async_shutdown()
                    except Exception as shutdown_err:
                        _LOGGER.debug(
                            "%s %s also failed during cleanup: %s",
                            self._package_log_label,
                            getattr(getattr(tool, "manifest", None), "tool_id", tool_dir.name),
                            shutdown_err,
                        )
                _LOGGER.error(
                    "Failed to load %s from %s: %s",
                    self._package_log_label,
                    tool_dir,
                    err,
                )
                self.last_load_errors.append(
                    {"tool_dir": tool_dir.name, "error": str(err)}
                )

        return loaded

    def _discover_package_dirs(self, tools_root: Path) -> list[tuple[Path, bool]]:
        """Return sorted candidate package directories and whether they are symlinks."""
        return [
            (tool_dir, tool_dir.is_symlink())
            for tool_dir in sorted(
                tools_root.iterdir(),
                key=lambda item: item.name.casefold(),
            )
            if tool_dir.is_dir() and not tool_dir.name.startswith((".", "__"))
        ]

    async def _load_manifest(self, tool_dir: Path) -> MCPAssistCustomToolManifest:
        """Load and validate a custom tool manifest without blocking the event loop."""
        return await self.hass.async_add_executor_job(
            self._load_manifest_from_disk,
            tool_dir,
        )

    def _load_manifest_from_disk(self, tool_dir: Path) -> MCPAssistCustomToolManifest:
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
            f"{self._module_namespace}.{manifest.tool_id}.{module_name.replace('.', '_')}"
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
        """Validate, normalize, and de-duplicate tool definitions."""
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
            if self._require_tool_name_prefix and not tool_name.startswith(
                f"{manifest.tool_id}_"
            ):
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
            llm_description = self._compact_text(
                str(
                    tool_definition.get(
                        "llmDescription", tool_definition.get("llm_description")
                    )
                    or ""
                ),
                max_len=180,
            )

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
            routing_hints = self._normalize_routing_hints(
                tool_name,
                tool_definition,
            )

            normalized.append(
                {
                    **tool_definition,
                    "name": tool_name,
                    "description": description,
                    **({"llmDescription": llm_description} if llm_description else {}),
                    "inputSchema": normalized_schema,
                    **({"routingHints": routing_hints} if routing_hints else {}),
                }
            )
            seen_names.add(tool_name)

        return normalized

    def _normalize_routing_hints(
        self,
        tool_name: str,
        tool_definition: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize optional tool-selection hints without bloating prompts."""
        routing = tool_definition.get("routing")
        if not isinstance(routing, dict):
            routing = {}

        keywords = self._normalize_string_list(
            routing.get("keywords", tool_definition.get("keywords")),
            max_items=8,
            max_len=40,
        )
        example_queries = self._normalize_string_list(
            routing.get("example_queries", tool_definition.get("example_queries")),
            max_items=4,
            max_len=100,
        )
        preferred_when = self._compact_text(
            str(
                routing.get("preferred_when", tool_definition.get("preferred_when"))
                or ""
            ),
            max_len=180,
        )
        returns = self._compact_text(
            str(routing.get("returns", tool_definition.get("returns")) or ""),
            max_len=160,
        )

        hints: dict[str, Any] = {}
        if keywords:
            hints["keywords"] = keywords
        if example_queries:
            hints["example_queries"] = example_queries
        if preferred_when:
            hints["preferred_when"] = preferred_when
        if returns:
            hints["returns"] = returns

        raw_routing = tool_definition.get("routing")
        if raw_routing is not None and not isinstance(raw_routing, dict):
            raise ValueError(
                f"Tool {tool_name!r} routing metadata must be an object when provided"
            )

        return hints

    @staticmethod
    def _normalize_string_list(
        value: Any,
        *,
        max_items: int,
        max_len: int,
    ) -> list[str]:
        """Normalize a short list of compact routing strings."""
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]

        normalized: list[str] = []
        for item in value:
            compact = ExternalCustomToolLoader._compact_text(
                str(item or ""),
                max_len=max_len,
            )
            if not compact or compact in normalized:
                continue
            normalized.append(compact)
            if len(normalized) >= max_items:
                break
        return normalized

    def _normalize_settings_schema(
        self,
        manifest: MCPAssistCustomToolManifest,
        schema: Any,
    ) -> dict[str, Any]:
        """Normalize an optional settings schema declared by the tool."""
        if schema in (None, {}):
            return {}
        if not isinstance(schema, dict):
            raise ValueError(
                f"Tool package {manifest.tool_id!r} get_settings_schema() must return an object"
            )
        normalized = dict(schema)
        if not normalized:
            return {}
        if not normalized.get("type"):
            normalized["type"] = "object"
        if normalized.get("type") == "object" and "properties" not in normalized:
            normalized["properties"] = {}
        return normalized

    async def _load_shared_settings(
        self,
        manifest: MCPAssistCustomToolManifest,
        settings_schema: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        """Load and validate shared package settings for a tool."""
        settings_path = self.get_settings_root() / f"{manifest.tool_id}.json"
        settings, file_exists = await self.hass.async_add_executor_job(
            self._load_settings_file_with_status,
            settings_path,
            settings_schema,
            True,
            f"{manifest.tool_id} shared settings",
        )
        return settings, str(settings_path) if file_exists else None

    async def load_profile_settings(
        self,
        loaded_tool: LoadedExternalTool,
        profile_entry_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        """Load an optional per-profile settings overlay for a tool."""
        if not profile_entry_id:
            return loaded_tool.shared_settings, {}, None

        profile_path = (
            self.get_settings_root()
            / "profiles"
            / profile_entry_id
            / f"{loaded_tool.manifest.tool_id}.json"
        )
        profile_settings, file_exists = await self.hass.async_add_executor_job(
            self._load_settings_file_with_status,
            profile_path,
            {},
            True,
            f"{loaded_tool.manifest.tool_id} profile settings for {profile_entry_id}",
        )
        merged_settings = self._deep_merge_dicts(
            loaded_tool.shared_settings,
            profile_settings,
        )
        if loaded_tool.settings_schema:
            merged_settings = validate_and_normalize_json_value(
                loaded_tool.settings_schema,
                merged_settings,
                path=f"{loaded_tool.manifest.tool_id} settings",
            )
        return (
            merged_settings,
            profile_settings,
            str(profile_path) if file_exists else None,
        )

    def _load_settings_file_with_status(
        self,
        path: Path,
        schema: dict[str, Any],
        allow_missing: bool,
        path_label: str,
    ) -> tuple[dict[str, Any], bool]:
        """Load a JSON settings file and validate it."""
        file_exists = path.exists()
        if not file_exists:
            if allow_missing:
                if schema:
                    return (
                        validate_and_normalize_json_value(schema, {}, path=path_label),
                        False,
                    )
                return {}, False
            raise ValueError(f"{path_label} file was not found: {path}")

        try:
            raw_data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as err:
            raise ValueError(f"Invalid JSON in {path}: {err}") from err

        if not isinstance(raw_data, dict):
            raise ValueError(f"{path} must contain a JSON object")

        try:
            if schema:
                return (
                    validate_and_normalize_json_value(
                        schema,
                        raw_data,
                        path=path_label,
                    ),
                    True,
                )
            return (
                validate_and_normalize_json_value(
                    {"type": "object", "properties": {}},
                    raw_data,
                    path=path_label,
                ),
                True,
            )
        except SchemaValidationError as err:
            raise ValueError(str(err)) from err

    def _deep_merge_dicts(
        self,
        base: dict[str, Any],
        overlay: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge nested settings dictionaries with overlay precedence."""
        merged = deepcopy(base)
        for key, value in overlay.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged

    async def _build_prompt_instructions(
        self,
        tool_dir: Path,
        manifest: MCPAssistCustomToolManifest,
        tool: MCPAssistExternalTool,
        tool_names: tuple[str, ...] = (),
    ) -> str:
        """Build a compact prompt appendix for a tool package."""
        prompt_file_text = await self._read_prompt_append_file(tool_dir, manifest)
        runtime_instructions = str(tool.get_prompt_instructions() or "").strip()
        instruction_parts = self._dedupe_instruction_parts(
            [prompt_file_text, runtime_instructions]
        )
        if not instruction_parts:
            fallback = self._build_prompt_fallback(manifest, tool_names)
            if fallback:
                instruction_parts = [fallback]

        if not instruction_parts:
            return ""

        combined_body = " ".join(instruction_parts).strip()
        label = str(manifest.name or manifest.tool_id).strip()
        combined = f"- {label}: {combined_body}"
        return self._compact_text(combined, max_len=_MAX_PROMPT_CHARS_PER_TOOL)

    async def _read_prompt_append_file(
        self,
        tool_dir: Path,
        manifest: MCPAssistCustomToolManifest,
    ) -> str:
        """Read optional static prompt instructions from the package asynchronously."""
        return await self.hass.async_add_executor_job(
            self._read_prompt_append_file_from_disk,
            tool_dir,
            manifest,
        )

    def _read_prompt_append_file_from_disk(
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
        for separator in (". ", "\n", "; "):
            if separator in normalized:
                normalized = normalized.split(separator, 1)[0].strip()
                break
        if len(normalized) <= max_len:
            return normalized
        trimmed = normalized[: max_len - 1].rstrip()
        last_space = trimmed.rfind(" ")
        if last_space > 40:
            trimmed = trimmed[:last_space]
        return trimmed.rstrip(" ,;:.") + "."

    def _dedupe_instruction_parts(self, raw_parts: list[str]) -> list[str]:
        """Normalize and deduplicate prompt-instruction fragments."""
        normalized_parts: list[str] = []
        seen: set[str] = set()
        for part in raw_parts:
            compact = self._compact_text(part, max_len=180)
            if not compact:
                continue
            key = compact.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized_parts.append(compact)
        return normalized_parts

    def _build_prompt_fallback(
        self,
        manifest: MCPAssistCustomToolManifest,
        tool_names: tuple[str, ...],
    ) -> str:
        """Build a minimal fallback prompt line when no prompt text is supplied."""
        if tool_names:
            preview = ", ".join(tool_names[:2])
            if len(tool_names) > 2:
                preview += ", ..."
            return f"Use {preview} for {manifest.tool_id.replace('_', ' ')} questions."

        if manifest.capabilities:
            return self._compact_text(manifest.capabilities[0], max_len=140)

        return ""


def combine_prompt_instructions(
    loaded_tools: list[LoadedExternalTool],
    *,
    heading: str = "## External Custom Tools",
    truncated_notice: str = "[External custom tool instructions truncated.]",
) -> str:
    """Combine tool-package prompt appendices into one compact block."""
    prompt_sections = [
        tool.prompt_instructions.strip()
        for tool in loaded_tools
        if tool.prompt_instructions.strip()
    ]
    if not prompt_sections:
        return ""

    combined_lines = [heading.strip(), *prompt_sections]
    combined = "\n".join(combined_lines)
    if len(combined) <= _MAX_PROMPT_CHARS_TOTAL:
        return combined

    kept_lines = [heading.strip()]
    current_len = len(heading.strip())
    for section in prompt_sections:
        candidate_len = current_len + 1 + len(section)
        if candidate_len > _MAX_PROMPT_CHARS_TOTAL:
            break
        kept_lines.append(section)
        current_len = candidate_len

    if len(kept_lines) == 1:
        truncated = heading.strip()[:_MAX_PROMPT_CHARS_TOTAL].rstrip()
        return truncated + f"\n{truncated_notice}"

    kept_lines.append(truncated_notice)
    return "\n".join(kept_lines)
