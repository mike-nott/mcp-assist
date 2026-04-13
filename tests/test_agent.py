"""Tests for profile-specific tool gating in the conversation agent."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.mcp_assist.agent import MCPAssistConversationEntity
from custom_components.mcp_assist.custom_tools.builtin_catalog import (
    load_builtin_tool_toggle_specs,
)
from custom_components.mcp_assist.const import (
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_INCLUDE_CURRENT_USER,
    CONF_INCLUDE_HOME_LOCATION,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_ENABLE_WEB_SEARCH,
    CONF_CLEAN_RESPONSES,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_MAX_HISTORY,
    CONF_PROFILE_NAME,
    CONF_TECHNICAL_PROMPT,
    CONF_TECHNICAL_PROMPT_MODE,
    CONF_PROFILE_ENABLE_ASSIST_BRIDGE,
    CONF_PROFILE_ENABLE_CALCULATOR_TOOLS,
    CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_PROFILE_ENABLE_DEVICE_TOOLS,
    CONF_PROFILE_ENABLE_WEB_SEARCH,
    DOMAIN,
    PROMPT_MODE_CUSTOM,
)

BUILTIN_SPECS = load_builtin_tool_toggle_specs()


def _builtin_spec(tool_name: str):
    """Return the built-in spec that declares a tool name."""
    for spec in BUILTIN_SPECS:
        if tool_name in spec.tool_names:
            return spec
    raise AssertionError(f"Missing built-in spec for {tool_name}")


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


def test_profile_tool_filtering_can_hide_convert_unit_without_hiding_add(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Profiles should still be able to disable unit conversion independently of math tools."""
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: True,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: True,
        }
    )
    entry = profile_entry_factory(
        options={
            CONF_PROFILE_ENABLE_CALCULATOR_TOOLS: True,
            CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS: False,
        }
    )

    agent = MCPAssistConversationEntity(hass, entry)
    filtered = agent._filter_mcp_tools_for_profile(
        [
            _tool("add"),
            _tool("convert_unit"),
        ]
    )

    tool_names = {tool["name"] for tool in filtered}
    assert "add" in tool_names
    assert "convert_unit" not in tool_names


def test_profile_tool_filtering_hides_web_search_tools_for_disabled_profile(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Profiles should still be able to hide search and read_url together."""
    system_entry_factory(
        data={
            CONF_ENABLE_WEB_SEARCH: True,
        }
    )
    entry = profile_entry_factory(
        options={
            CONF_PROFILE_ENABLE_WEB_SEARCH: False,
        }
    )

    agent = MCPAssistConversationEntity(hass, entry)
    filtered = agent._filter_mcp_tools_for_profile(
        [
            _tool("search"),
            _tool("read_url"),
            _tool("discover_entities"),
        ]
    )

    tool_names = {tool["name"] for tool in filtered}
    assert "discover_entities" in tool_names
    assert "search" not in tool_names
    assert "read_url" not in tool_names


def test_optional_technical_instructions_include_external_custom_tool_guidance(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Loaded external custom tools should be able to extend prompt guidance."""
    system_entry_factory(data={CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True})
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(
        custom_tools=SimpleNamespace(
            is_external_custom_tool=lambda name: name == "sample_tool_status",
            get_external_prompt_instructions=lambda: "## External Custom Tools\nUse sample_tool_status when asked for custom status."
        )
    )

    instructions = agent._build_optional_technical_instructions("Kitchen")

    assert "External Custom Tools" in instructions
    assert "sample_tool_status" in instructions


def test_optional_technical_instructions_include_built_in_package_guidance(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Built-in packaged tools should contribute prompt guidance through the loader."""
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: True,
        }
    )
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(
        custom_tools=SimpleNamespace(
            get_builtin_prompt_instructions=lambda: (
                "## Optional Built-In Tool Packages\n"
                "Use calculator tools for arithmetic questions."
            ),
            get_builtin_toggle_specs=lambda: BUILTIN_SPECS,
        )
    )

    instructions = agent._build_optional_technical_instructions("Kitchen")

    assert "Optional Built-In Tool Packages" in instructions
    assert "calculator tools" in instructions


def test_profile_tool_filtering_hides_disabled_external_custom_tools(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """External custom tools should also respect profile-level tool disables."""
    system_entry_factory(data={CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True})
    entry = profile_entry_factory(
        options={CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS: False}
    )
    agent = MCPAssistConversationEntity(hass, entry)
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(
        custom_tools=SimpleNamespace(
            is_external_custom_tool=lambda name: name == "sample_tool_status"
        )
    )

    filtered = agent._filter_mcp_tools_for_profile(
        [
            _tool("discover_entities"),
            _tool("sample_tool_status"),
        ]
    )

    tool_names = {tool["name"] for tool in filtered}
    assert "discover_entities" in tool_names
    assert "sample_tool_status" not in tool_names


def test_optional_technical_instructions_omit_external_custom_tool_guidance_when_disabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """External custom tool prompt guidance should disappear when the profile disables it."""
    system_entry_factory(data={CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True})
    entry = profile_entry_factory(
        options={CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS: False}
    )
    agent = MCPAssistConversationEntity(hass, entry)
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(
        custom_tools=SimpleNamespace(
            is_external_custom_tool=lambda name: name == "sample_tool_status",
            get_external_prompt_instructions=lambda: "## External Custom Tools\nUse sample_tool_status when asked for custom status.",
        )
    )

    instructions = agent._build_optional_technical_instructions("Kitchen")

    assert "External Custom Tools" not in instructions
    assert "sample_tool_status" not in instructions


def test_profile_tool_filtering_can_disable_search_without_hiding_read_url(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Built-in packaged tools should be independently gateable per profile."""
    system_entry_factory(
        data={
            "enable_search_tool": True,
            "enable_read_url_tool": True,
        }
    )
    entry = profile_entry_factory(
        options={
            "profile_enable_search_tool": False,
            "profile_enable_read_url_tool": True,
        }
    )
    agent = MCPAssistConversationEntity(hass, entry)
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(
        custom_tools=SimpleNamespace(
            get_builtin_toggle_spec=lambda name: _builtin_spec(name),
            get_builtin_toggle_specs=lambda: BUILTIN_SPECS,
        )
    )

    filtered = agent._filter_mcp_tools_for_profile(
        [
            _tool("search"),
            _tool("read_url"),
        ]
    )

    tool_names = {tool["name"] for tool in filtered}
    assert "search" not in tool_names
    assert "read_url" in tool_names


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


@pytest.mark.asyncio
async def test_profile_disabled_unit_conversion_is_rejected_before_mcp_call(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Direct MCP calls should still respect the profile unit-conversion toggle."""
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: True,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: True,
        }
    )
    entry = profile_entry_factory(
        options={
            CONF_PROFILE_ENABLE_CALCULATOR_TOOLS: True,
            CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS: False,
        }
    )

    agent = MCPAssistConversationEntity(hass, entry)
    result = await agent._call_mcp_tool("convert_unit", {"value": 1, "from_unit": "m", "to_unit": "ft"})

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
async def test_get_mcp_tools_refetches_when_external_custom_tool_signature_changes(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """External custom tool changes should invalidate the profile MCP-tool cache immediately."""
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)
    state = {"signature": ("v1",)}
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(
        custom_tools=SimpleNamespace(get_cache_signature=lambda: state["signature"])
    )
    fetch_mock = AsyncMock(
        side_effect=[
            [{"type": "function", "function": {"name": "sample_tool_status"}}],
            [{"type": "function", "function": {"name": "sample_tool_history"}}],
        ]
    )
    monkeypatch.setattr(agent, "_fetch_mcp_tools_from_server", fetch_mock)

    first = await agent._get_mcp_tools()
    state["signature"] = ("v2",)
    second = await agent._get_mcp_tools()

    assert first != second
    assert first[0]["function"]["name"] == "sample_tool_status"
    assert second[0]["function"]["name"] == "sample_tool_history"
    assert fetch_mock.await_count == 2


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


def test_convert_mcp_tools_to_llm_tools_appends_routing_hints(
    hass, profile_entry_factory
) -> None:
    """Routing hints should improve tool selection without needing long prompt text."""
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    tools = [
        {
            "name": "sample_tool_status",
            "description": "Return custom status.",
            "inputSchema": {"type": "object", "properties": {}},
            "routingHints": {
                "keywords": ["status", "custom"],
                "preferred_when": "Use when the user asks for custom package health.",
                "returns": "A short status summary.",
                "example_queries": ["What's the custom status?"],
            },
        }
    ]

    compact_tools = agent._convert_mcp_tools_to_llm_tools(tools)
    description = compact_tools[0]["function"]["description"]

    assert "Return custom status" in description
    assert "Use for: Use when the user asks for custom package health" in description
    assert "Keywords:" not in description
    assert "Example:" not in description


def test_format_tool_result_for_llm_preserves_structured_results_without_binary(
    hass, profile_entry_factory
) -> None:
    """Structured MCP results should survive, but binary image payloads should be compacted."""
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    formatted = agent._format_tool_result_for_llm(
        "analyze_image",
        {
            "content": [
                {"type": "text", "text": "White SUV in the driveway."},
                {"type": "image", "mimeType": "image/jpeg", "data": "a" * 4096},
            ],
            "structuredContent": {"source": {"camera_entity_id": "camera.driveway"}},
            "isError": False,
        },
    )

    assert "White SUV in the driveway." in formatted
    assert "[binary image omitted:" in formatted
    assert "camera.driveway" in formatted


@pytest.mark.asyncio
async def test_call_mcp_tool_includes_profile_context(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """MCP tool calls should identify the active profile for settings-aware tools."""
    entry = profile_entry_factory(data={CONF_PROFILE_NAME: "Kitchen Profile"})
    agent = MCPAssistConversationEntity(hass, entry)
    captured: dict[str, object] = {}

    class _FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self):
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{"type": "text", "text": "ok"}],
                    "isError": False,
                },
            }

    class _FakeSession:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            captured["url"] = url
            captured["payload"] = json
            return _FakeResponse()

    monkeypatch.setattr(
        "custom_components.mcp_assist.agent.aiohttp.ClientSession",
        _FakeSession,
    )

    result = await agent._call_mcp_tool("sample_tool_status", {})

    assert result["content"][0]["text"] == "ok"
    assert captured["payload"]["params"]["context"] == {
        "profile_entry_id": entry.entry_id,
        "profile_name": "Kitchen Profile",
    }


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
