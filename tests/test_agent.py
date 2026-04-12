"""Tests for profile-specific tool gating in the conversation agent."""

from __future__ import annotations

import pytest

from custom_components.mcp_assist.agent import MCPAssistConversationEntity
from custom_components.mcp_assist.const import (
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_PROFILE_ENABLE_ASSIST_BRIDGE,
    CONF_PROFILE_ENABLE_DEVICE_TOOLS,
)


def _tool(name: str) -> dict[str, object]:
    """Build a minimal MCP tool definition."""
    return {
        "name": name,
        "description": name,
        "inputSchema": {"type": "object", "properties": {}},
    }


def test_profile_tool_enablement_respects_shared_and_profile_settings(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """A profile can only use optional tools when both shared and profile settings allow it."""
    system_entry_factory(
        data={
            CONF_ENABLE_ASSIST_BRIDGE: True,
            CONF_ENABLE_DEVICE_TOOLS: False,
        }
    )
    entry = profile_entry_factory(
        options={
            CONF_PROFILE_ENABLE_ASSIST_BRIDGE: False,
            CONF_PROFILE_ENABLE_DEVICE_TOOLS: True,
        }
    )

    agent = MCPAssistConversationEntity(hass, entry)

    assert agent.assist_bridge_enabled is False
    assert agent.device_tools_enabled is False


def test_profile_tool_filtering_hides_disabled_optional_tools(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Only profile-enabled optional tools should be exposed to the LLM."""
    system_entry_factory(
        data={
            CONF_ENABLE_ASSIST_BRIDGE: True,
            CONF_ENABLE_DEVICE_TOOLS: True,
        }
    )
    entry = profile_entry_factory(
        options={
            CONF_PROFILE_ENABLE_ASSIST_BRIDGE: False,
            CONF_PROFILE_ENABLE_DEVICE_TOOLS: False,
        }
    )

    agent = MCPAssistConversationEntity(hass, entry)
    filtered = agent._filter_mcp_tools_for_profile(
        [
            _tool("discover_entities"),
            _tool("discover_devices"),
            _tool("list_assist_tools"),
        ]
    )

    tool_names = {tool["name"] for tool in filtered}
    assert "discover_entities" in tool_names
    assert "discover_devices" not in tool_names
    assert "list_assist_tools" not in tool_names


@pytest.mark.asyncio
async def test_profile_disabled_tool_is_rejected_before_mcp_call(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Direct tool calls should also fail closed when the profile disabled that family."""
    system_entry_factory(data={CONF_ENABLE_ASSIST_BRIDGE: True})
    entry = profile_entry_factory(options={CONF_PROFILE_ENABLE_ASSIST_BRIDGE: False})

    agent = MCPAssistConversationEntity(hass, entry)
    result = await agent._call_mcp_tool("list_assist_tools", {})

    assert result["isError"] is True
    assert "disabled for this profile" in result["content"][0]["text"]
