"""Tests for custom tool loading."""

from __future__ import annotations

import sys
import types

import pytest

from custom_components.mcp_assist.const import (
    CONF_BRAVE_API_KEY,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_CUSTOM_TOOLS,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_ENABLE_WEB_SEARCH,
    CONF_SEARCH_PROVIDER,
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
