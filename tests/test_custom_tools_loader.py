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
    CUSTOM_TOOLS_DIRECTORY,
    LEGACY_CUSTOM_TOOL_MANIFEST_FILENAME,
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

    def get_tool_definitions(self):
        return [{{
            "name": "{resolved_tool_name}",
            "description": "Return a sample status.",
            "inputSchema": {{"type": "object", "properties": {{}}}},
        }}]

    async def handle_call(self, tool_name, arguments):
        return {{
            "content": [{{"type": "text", "text": "sample ok"}}],
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
async def test_legacy_manifest_filename_is_still_supported(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Older external tool packages using manifest.json should keep loading."""
    _write_external_tool_package(
        tmp_path,
        manifest_filename=LEGACY_CUSTOM_TOOL_MANIFEST_FILENAME,
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

    assert [item["id"] for item in loader.get_loaded_external_tool_info()] == [
        "sample_tool"
    ]
