"""Tests for profile-specific tool gating in the conversation agent."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.mcp_assist.agent import MCPAssistConversationEntity
from custom_components.mcp_assist.const import (
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_INCLUDE_CURRENT_USER,
    CONF_INCLUDE_HOME_LOCATION,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_CLEAN_RESPONSES,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_MAX_HISTORY,
    CONF_TECHNICAL_PROMPT,
    CONF_TECHNICAL_PROMPT_MODE,
    CONF_PROFILE_ENABLE_ASSIST_BRIDGE,
    CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS,
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


def test_unit_conversion_tool_enablement_has_backward_compatible_fallbacks(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Unit conversion should inherit older calculator settings when its own flags are absent."""
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: True,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: None,
        }
    )
    entry = profile_entry_factory(
        options={
            CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS: None,
        }
    )

    agent = MCPAssistConversationEntity(hass, entry)

    assert agent.unit_conversion_tools_enabled is True


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


def test_optional_technical_instructions_include_external_custom_tool_guidance(
    hass, profile_entry_factory
) -> None:
    """Loaded external custom tools should be able to extend prompt guidance."""
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(
        custom_tools=SimpleNamespace(
            get_external_prompt_instructions=lambda: "## External Custom Tools\nUse sample_tool_status when asked for custom status."
        )
    )

    instructions = agent._build_optional_technical_instructions("Kitchen")

    assert "External Custom Tools" in instructions
    assert "sample_tool_status" in instructions


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
async def test_default_prompt_includes_current_user_and_home_location_context(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Default prompts should include current HA user and home location when enabled."""
    system_entry_factory(
        data={
            CONF_INCLUDE_CURRENT_USER: True,
            CONF_INCLUDE_HOME_LOCATION: True,
        }
    )
    hass.config.location_name = "Test Home"
    hass.config.latitude = 47.6205
    hass.config.longitude = -122.3493
    monkeypatch.setattr(
        hass.auth,
        "async_get_user",
        AsyncMock(return_value=SimpleNamespace(name="Jason")),
    )
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    prompt = await agent._build_system_prompt_with_context(
        SimpleNamespace(device_id=None, context=SimpleNamespace(user_id="user-123"))
    )

    assert "Current user: Jason" in prompt
    assert "Home location: Test Home (47.6205, -122.3493)" in prompt


@pytest.mark.asyncio
async def test_default_prompt_omits_optional_identity_context_when_disabled(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Shared privacy settings should allow identity/location prompt context to be omitted."""
    system_entry_factory(
        data={
            CONF_INCLUDE_CURRENT_USER: False,
            CONF_INCLUDE_HOME_LOCATION: False,
        }
    )
    monkeypatch.setattr(
        hass.auth,
        "async_get_user",
        AsyncMock(return_value=SimpleNamespace(name="Jason")),
    )
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    prompt = await agent._build_system_prompt_with_context(
        SimpleNamespace(device_id=None, context=SimpleNamespace(user_id="user-123"))
    )

    assert "Current user:" not in prompt
    assert "Home location:" not in prompt


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


@pytest.mark.asyncio
async def test_get_mcp_tools_uses_short_lived_cache(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """Repeated tool fetches with the same profile surface should reuse the cache."""
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)
    fetch_mock = AsyncMock(return_value=[{"type": "function", "function": {"name": "discover_entities"}}])
    monkeypatch.setattr(agent, "_fetch_mcp_tools_from_server", fetch_mock)

    first = await agent._get_mcp_tools()
    second = await agent._get_mcp_tools()

    assert first == second
    fetch_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_mcp_tools_uses_stale_cache_on_refresh_failure(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """A transient tools/list failure should fall back to the last cached tool surface."""
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)
    cached_tools = [{"type": "function", "function": {"name": "discover_entities"}}]
    agent._cached_llm_tools = list(cached_tools)
    agent._cached_llm_tools_key = agent._build_mcp_tool_cache_key()
    agent._cached_llm_tools_fetched_at = 0
    monkeypatch.setattr(agent, "_fetch_mcp_tools_from_server", AsyncMock(return_value=None))
    monkeypatch.setattr("custom_components.mcp_assist.agent.time.monotonic", lambda: 9999.0)

    result = await agent._get_mcp_tools()

    assert result == cached_tools


def test_compact_tool_result_for_llm_truncates_large_payloads(
    hass, profile_entry_factory
) -> None:
    """Oversized tool results should be trimmed before re-entering the model loop."""
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    large_result = "\n".join(f"line {index}" for index in range(300))
    compacted = agent._compact_tool_result_for_llm(
        "discover_entities", large_result
    )

    assert "Tool result truncated for model context" in compacted
    assert "Use limit/offset paging" in compacted
    assert len(compacted) < len(large_result)


@pytest.mark.asyncio
async def test_trigger_tts_is_a_noop(
    hass, profile_entry_factory
) -> None:
    """Interim streaming TTS should simply return without doing extra work."""
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    assert await agent._trigger_tts("Hello there.") is None


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
