"""Tests for profile-specific tool gating in the conversation agent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_components.mcp_assist.agent import MCPAssistConversationEntity
from custom_components.mcp_assist.const import (
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_CLEAN_RESPONSES,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_MAX_HISTORY,
    CONF_TECHNICAL_PROMPT,
    CONF_TECHNICAL_PROMPT_MODE,
    CONF_PROFILE_ENABLE_ASSIST_BRIDGE,
    CONF_PROFILE_ENABLE_DEVICE_TOOLS,
    DOMAIN,
    PROMPT_MODE_CUSTOM,
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


def test_build_messages_respects_configured_max_history(
    hass, profile_entry_factory
) -> None:
    """Conversation message building should honor the configured history limit."""
    entry = profile_entry_factory(options={CONF_MAX_HISTORY: 2})
    agent = MCPAssistConversationEntity(hass, entry)

    history = [
        {"user": "u1", "assistant": "a1"},
        {"user": "u2", "assistant": "a2"},
        {"user": "u3", "assistant": "a3"},
    ]

    messages = agent._build_messages("system", "current", history)

    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "current"},
    ]


def test_build_messages_supports_zero_history(
    hass, profile_entry_factory
) -> None:
    """A zero history limit should omit prior turns entirely."""
    entry = profile_entry_factory(options={CONF_MAX_HISTORY: 0})
    agent = MCPAssistConversationEntity(hass, entry)

    history = [{"user": "u1", "assistant": "a1"}]

    messages = agent._build_messages("system", "current", history)

    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "current"},
    ]


@pytest.mark.asyncio
async def test_default_prompt_does_not_fetch_index_unless_requested(
    hass, profile_entry_factory
) -> None:
    """The built-in prompt path should not inline the smart index on every turn."""

    class FailIndexManager:
        async def get_index(self) -> dict[str, object]:
            raise AssertionError("get_index should not be called for default prompts")

    hass.data.setdefault(DOMAIN, {})["index_manager"] = FailIndexManager()
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    prompt = await agent._build_system_prompt_with_context(SimpleNamespace(device_id=None))

    assert "Current area:" in prompt
    assert "get_index()" in prompt
    assert "## Index" not in prompt


@pytest.mark.asyncio
async def test_custom_prompt_with_index_placeholder_fetches_index(
    hass, profile_entry_factory
) -> None:
    """Custom prompts that explicitly request {index} should still work."""

    class StubIndexManager:
        async def get_index(self) -> dict[str, object]:
            return {"areas": ["Kitchen"], "domains": {"light": 3}}

    hass.data.setdefault(DOMAIN, {})["index_manager"] = StubIndexManager()
    entry = profile_entry_factory(
        options={
            CONF_TECHNICAL_PROMPT_MODE: PROMPT_MODE_CUSTOM,
            CONF_TECHNICAL_PROMPT: "Index:{index}",
        }
    )
    agent = MCPAssistConversationEntity(hass, entry)

    prompt = await agent._build_system_prompt_with_context(SimpleNamespace(device_id=None))

    assert 'Index:{"areas":["Kitchen"],"domains":{"light":3}}' in prompt


def test_convert_mcp_tools_to_llm_tools_compacts_schema(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """The LLM-facing tool schema should drop nonessential verbosity."""
    system_entry_factory(data={CONF_ENABLE_CALCULATOR_TOOLS: True})
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    tools = [
        {
            "name": "discover_entities",
            "description": (
                "Find and list Home Assistant entities by criteria like area, floor, "
                "label, type, domain, device_class, current state, or aliases. "
                "Prefer this for most direct control."
            ),
            "inputSchema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": "Area name or alias to search in. Check get_index() to see available areas.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of entities to return.",
                        "default": 20,
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        }
    ]

    compact_tools = agent._convert_mcp_tools_to_llm_tools(tools)
    function = compact_tools[0]["function"]
    parameters = function["parameters"]

    assert function["description"] == "Find and list Home Assistant entities by criteria like area, floor, label, type, domain, device_class, current state, or aliases"
    assert "$schema" not in parameters
    assert "additionalProperties" not in parameters
    assert "required" not in parameters
    assert "description" not in parameters["properties"]["area"]
    assert "default" not in parameters["properties"]["limit"]


def test_convert_mcp_tools_to_llm_tools_keeps_empty_object_properties(
    hass, profile_entry_factory
) -> None:
    """OpenAI requires object schemas to include properties, even when empty."""
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    tools = [
        {
            "name": "list_music_assistant_instances",
            "description": "List configured Music Assistant instances.",
            "inputSchema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }
    ]

    compact_tools = agent._convert_mcp_tools_to_llm_tools(tools)
    parameters = compact_tools[0]["function"]["parameters"]

    assert parameters == {"type": "object", "properties": {}}


def test_clean_text_for_tts_removes_spaces_before_punctuation(
    hass, profile_entry_factory
) -> None:
    """Final speech text should not keep stray spaces before punctuation."""
    entry = profile_entry_factory(data={CONF_CLEAN_RESPONSES: False})
    agent = MCPAssistConversationEntity(hass, entry)

    cleaned = agent._clean_text_for_tts(
        "I can use it , the weather entity is available ."
    )

    assert cleaned == "I can use it, the weather entity is available."
