"""Tests for custom tool loading."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from custom_components.mcp_assist.const import (
    CONF_BRAVE_API_KEY,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_CUSTOM_TOOLS,
    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_ENABLE_WEB_SEARCH,
    CONF_SEARCH_PROVIDER,
    CUSTOM_TOOL_MANIFEST_FILENAME,
    CUSTOM_TOOL_SETTINGS_DIRECTORY,
    CUSTOM_TOOLS_DIRECTORY,
)
from custom_components.mcp_assist.custom_tools import CustomToolsLoader


class _StubTool:
    """Simple stub custom tool implementation."""

    def __init__(self, hass, *args) -> None:
        self.hass = hass
        self.args = args

    async def initialize(self) -> None:
        return None

    def get_tool_definitions(self):
        return [{"name": self.__class__.__name__}]

    def handles_tool(self, tool_name: str) -> bool:
        return tool_name == self.__class__.__name__

    async def handle_call(self, tool_name, arguments):
        return {"content": [{"type": "text", "text": tool_name}]}


def _write_external_tool_package(
    config_root: Path,
    *,
    tool_id: str = "sample_tool",
    tool_name: str | None = None,
    include_prompt_file: bool = True,
    directory_name: str | None = None,
    manifest_filename: str = CUSTOM_TOOL_MANIFEST_FILENAME,
) -> Path:
    """Write a valid external custom tool package to the temp config root."""
    package_dir = config_root / CUSTOM_TOOLS_DIRECTORY / (directory_name or tool_id)
    package_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "id": tool_id,
        "name": "Sample Tool",
        "description": "Example external tool package for testing.",
        "version": "1.0.0",
        "entrypoint": "tool:SampleTool",
        "capabilities": ["Provides a custom sample status lookup."],
    }
    if include_prompt_file:
        manifest["prompt_append_file"] = "prompt.md"

    (package_dir / manifest_filename).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    if include_prompt_file:
        (package_dir / "prompt.md").write_text(
            "Use sample_tool_status when the user asks for the sample system status.",
            encoding="utf-8",
        )

    resolved_tool_name = tool_name or f"{tool_id}_status"
    (package_dir / "tool.py").write_text(
        f'''from custom_components.mcp_assist.custom_tool_api import MCPAssistExternalTool


class SampleTool(MCPAssistExternalTool):
    async def initialize(self) -> None:
        return None

    def get_settings_schema(self):
        return {{
            "type": "object",
            "properties": {{
                "status_text": {{
                    "type": "string",
                    "default": "sample ok",
                }}
            }},
        }}

    def get_tool_definitions(self):
        return [{{
            "name": "{resolved_tool_name}",
            "description": "Return a sample status.",
            "inputSchema": {{"type": "object", "properties": {{}}}},
            "keywords": ["sample", "status"],
            "example_queries": ["What's the sample status?"],
            "preferred_when": "Use for sample package status questions.",
            "returns": "A short sample status string.",
        }}]

    async def handle_call(self, tool_name, arguments):
        settings = self.get_settings()
        return {{
            "content": [{{"type": "text", "text": settings.get("status_text", "sample ok")}}],
            "isError": False,
        }}

    def get_prompt_instructions(self) -> str:
        return "Prefer {resolved_tool_name} for sample-status questions."
''',
        encoding="utf-8",
    )
    return package_dir


@pytest.mark.asyncio
async def test_initialize_skips_calculator_when_disabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Calculator tools should not be loaded when the shared toggle is disabled."""
    profile_entry = profile_entry_factory()
    system_entry_factory(data={CONF_ENABLE_CALCULATOR_TOOLS: False})
    loader = CustomToolsLoader(hass, profile_entry)

    await loader.initialize()

    assert "calculator" not in loader.tools


@pytest.mark.asyncio
async def test_initialize_loads_calculator_bundle_when_only_unit_conversion_enabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Unit conversion should still load the shared calculator tool bundle."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: True,
        }
    )
    loader = CustomToolsLoader(hass, profile_entry)

    await loader.initialize()

    assert "calculator" in loader.tools


@pytest.mark.asyncio
async def test_initialize_loads_search_and_read_url_for_brave(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Search-enabled setups should load the provider and read_url tools."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_WEB_SEARCH: True,
            CONF_SEARCH_PROVIDER: "brave",
            CONF_BRAVE_API_KEY: "secret",
            CONF_ENABLE_CALCULATOR_TOOLS: False,
        }
    )

    brave_module = types.SimpleNamespace(BraveSearchTool=type("BraveSearchTool", (_StubTool,), {}))
    read_url_module = types.SimpleNamespace(ReadUrlTool=type("ReadUrlTool", (_StubTool,), {}))
    monkeypatch.setitem(
        sys.modules,
        "custom_components.mcp_assist.custom_tools.brave_search",
        brave_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "custom_components.mcp_assist.custom_tools.read_url",
        read_url_module,
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert set(loader.tools) == {"search", "read_url"}


@pytest.mark.asyncio
async def test_initialize_skips_search_when_web_search_disabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Search tools should not load when the web-search tool family is disabled."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_WEB_SEARCH: False,
            CONF_SEARCH_PROVIDER: "brave",
            CONF_BRAVE_API_KEY: "secret",
            CONF_ENABLE_CALCULATOR_TOOLS: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert "search" not in loader.tools
    assert "read_url" not in loader.tools


def test_get_search_provider_keeps_backward_compatibility(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Legacy enable_custom_tools should still imply Brave search when search_provider is unset."""
    profile_entry = profile_entry_factory(options={CONF_ENABLE_CUSTOM_TOOLS: True})
    system_entry_factory(data={CONF_SEARCH_PROVIDER: None}, options={})
    loader = CustomToolsLoader(hass, profile_entry)

    assert loader._get_search_provider() == "brave"


@pytest.mark.asyncio
async def test_external_custom_tools_are_disabled_by_default(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """User-defined custom tools should not load unless explicitly enabled."""
    _write_external_tool_package(tmp_path)
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: False,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert loader.external_tools == []
    assert loader.get_external_prompt_instructions() == ""
    assert "sample_tool" not in loader.tools


@pytest.mark.asyncio
async def test_external_custom_tool_package_loads_and_handles_calls(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Enabled external custom tools should load from the HA config directory."""
    _write_external_tool_package(tmp_path)
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    tool_names = {
        tool_definition["name"] for tool_definition in loader.get_tool_definitions()
    }
    assert "sample_tool_status" in tool_names
    assert "External Custom Tools" in loader.get_external_prompt_instructions()
    assert "Prefer sample_tool_status" in loader.get_external_prompt_instructions()
    assert loader.get_loaded_external_tool_info() == [
        {
            "id": "sample_tool",
            "name": "Sample Tool",
            "version": "1.0.0",
            "description": "Example external tool package for testing.",
            "tool_names": ["sample_tool_status"],
            "capabilities": ["Provides a custom sample status lookup."],
        }
    ]

    result = await loader.handle_tool_call("sample_tool_status", {})
    assert result["isError"] is False
    assert result["content"][0]["text"] == "sample ok"

    tool_definition = next(
        tool
        for tool in loader.get_tool_definitions()
        if tool["name"] == "sample_tool_status"
    )
    assert tool_definition["routingHints"]["keywords"] == ["sample", "status"]
    assert tool_definition["routingHints"]["preferred_when"] == (
        "Use for sample package status questions."
    )


@pytest.mark.asyncio
async def test_invalid_external_tool_package_is_skipped_without_crashing(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Invalid external packages should be skipped while valid loading continues."""
    _write_external_tool_package(tmp_path, directory_name="wrong_dir_name")
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert loader.external_tools == []
    assert loader.get_tool_definitions() == []


@pytest.mark.asyncio
async def test_external_tool_name_must_be_namespaced(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """External tool names must be prefixed with the manifest id."""
    _write_external_tool_package(tmp_path, tool_name="status")
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert loader.external_tools == []


@pytest.mark.asyncio
async def test_legacy_manifest_filename_is_not_supported(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Only the canonical mcp_tool.json manifest filename should be accepted."""
    _write_external_tool_package(
        tmp_path,
        manifest_filename="manifest.json",
    )
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert loader.get_loaded_external_tool_info() == []


@pytest.mark.asyncio
async def test_external_custom_tool_uses_shared_and_profile_settings(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """External tools should receive merged shared and per-profile settings."""
    _write_external_tool_package(tmp_path)
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    settings_root = tmp_path / CUSTOM_TOOL_SETTINGS_DIRECTORY
    settings_root.mkdir(parents=True, exist_ok=True)
    (settings_root / "sample_tool.json").write_text(
        json.dumps({"status_text": "shared status"}),
        encoding="utf-8",
    )
    profile_settings_dir = settings_root / "profiles" / profile_entry.entry_id
    profile_settings_dir.mkdir(parents=True, exist_ok=True)
    (profile_settings_dir / "sample_tool.json").write_text(
        json.dumps({"status_text": "profile status"}),
        encoding="utf-8",
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    shared_result = await loader.handle_tool_call("sample_tool_status", {})
    profile_result = await loader.handle_tool_call(
        "sample_tool_status",
        {},
        context={"profile_entry_id": profile_entry.entry_id},
    )

    assert shared_result["content"][0]["text"] == "shared status"
    assert profile_result["content"][0]["text"] == "profile status"


@pytest.mark.asyncio
async def test_external_custom_tool_argument_validation_returns_mcp_error(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Invalid external-tool arguments should be rejected before tool execution."""
    package_dir = tmp_path / CUSTOM_TOOLS_DIRECTORY / "typed_tool"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / CUSTOM_TOOL_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "typed_tool",
                "name": "Typed Tool",
                "description": "Typed external tool for schema validation tests.",
                "version": "1.0.0",
                "entrypoint": "tool:TypedTool",
            }
        ),
        encoding="utf-8",
    )
    (package_dir / "tool.py").write_text(
        """from custom_components.mcp_assist.custom_tool_api import MCPAssistExternalTool


class TypedTool(MCPAssistExternalTool):
    def get_tool_definitions(self):
        return [{
            "name": "typed_tool_echo",
            "description": "Echo an integer value.",
            "inputSchema": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
                "additionalProperties": False,
            },
        }]

    async def handle_call(self, tool_name, arguments):
        return {
            "content": [{"type": "text", "text": str(arguments["count"])}],
            "isError": False,
        }
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    result = await loader.handle_tool_call("typed_tool_echo", {"count": "nope"})

    assert result["isError"] is True
    assert "typed_tool_echo arguments.count must be an integer" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_external_custom_tools_can_import_shared_helpers(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Packages should be able to import helper modules from __shared__ safely."""
    shared_dir = tmp_path / CUSTOM_TOOLS_DIRECTORY / "__shared__"
    shared_dir.mkdir(parents=True, exist_ok=True)
    (shared_dir / "helpers.py").write_text(
        "def message():\n    return 'shared helper ok'\n",
        encoding="utf-8",
    )

    package_dir = tmp_path / CUSTOM_TOOLS_DIRECTORY / "shared_tool"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / CUSTOM_TOOL_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "shared_tool",
                "name": "Shared Tool",
                "description": "External tool that uses a shared helper.",
                "version": "1.0.0",
                "entrypoint": "tool:SharedTool",
            }
        ),
        encoding="utf-8",
    )
    (package_dir / "tool.py").write_text(
        """from custom_components.mcp_assist.custom_tool_api import (
    MCPAssistExternalTool,
    load_external_shared_module,
)

helpers = load_external_shared_module(__file__, "helpers")


class SharedTool(MCPAssistExternalTool):
    def get_tool_definitions(self):
        return [{
            "name": "shared_tool_status",
            "description": "Return shared helper output.",
            "inputSchema": {"type": "object", "properties": {}},
        }]

    async def handle_call(self, tool_name, arguments):
        return {
            "content": [{"type": "text", "text": helpers.message()}],
            "isError": False,
        }
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    result = await loader.handle_tool_call("shared_tool_status", {})

    assert result["isError"] is False
    assert result["content"][0]["text"] == "shared helper ok"
