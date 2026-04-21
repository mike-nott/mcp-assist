"""Config flow for MCP Assist integration."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult, section
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.selector import (
    BooleanSelector,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .custom_tools.builtin_catalog import (
    BuiltInToolToggleSpec,
    get_builtin_profile_setting_value,
    get_builtin_shared_setting_value,
    load_builtin_tool_toggle_specs,
)
from .localization import get_language_instruction, get_follow_up_phrases, get_end_words

from .const import (
    DOMAIN,
    SYSTEM_ENTRY_UNIQUE_ID,
    CONF_PROFILE_NAME,
    CONF_SERVER_TYPE,
    CONF_API_KEY,
    CONF_LMSTUDIO_URL,
    CONF_MODEL_NAME,
    CONF_MCP_PORT,
    CONF_AUTO_START,
    CONF_SYSTEM_PROMPT,
    CONF_TECHNICAL_PROMPT,
    CONF_SYSTEM_PROMPT_MODE,
    CONF_TECHNICAL_PROMPT_MODE,
    CONF_CONTROL_HA,
    CONF_FOLLOW_UP_MODE,
    CONF_RESPONSE_MODE,
    CONF_TEMPERATURE,
    CONF_MAX_TOKENS,
    CONF_MAX_HISTORY,
    CONF_MAX_ITERATIONS,
    CONF_DEBUG_MODE,
    CONF_ENABLE_CUSTOM_TOOLS,
    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_BRAVE_API_KEY,
    CONF_ALLOWED_IPS,
    CONF_INCLUDE_CURRENT_USER,
    CONF_INCLUDE_HOME_LOCATION,
    CONF_SEARCH_PROVIDER,
    CONF_ENABLE_WEB_SEARCH,
    CONF_ENABLE_GAP_FILLING,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
    CONF_ENABLE_WEATHER_FORECAST_TOOL,
    CONF_ENABLE_RECORDER_TOOLS,
    CONF_ENABLE_MEMORY_TOOLS,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    CONF_MEMORY_DEFAULT_TTL_DAYS,
    CONF_MEMORY_MAX_TTL_DAYS,
    CONF_MEMORY_MAX_ITEMS,
    CONF_MAX_ENTITIES_PER_DISCOVERY,
    DEFAULT_MAX_ENTITIES_PER_DISCOVERY,
    CONF_OLLAMA_KEEP_ALIVE,
    CONF_OLLAMA_NUM_CTX,
    CONF_FOLLOW_UP_PHRASES,
    CONF_END_WORDS,
    CONF_CLEAN_RESPONSES,
    CONF_TIMEOUT,
    SERVER_TYPE_LMSTUDIO,
    SERVER_TYPE_LLAMACPP,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_OPENAI,
    SERVER_TYPE_GEMINI,
    SERVER_TYPE_ANTHROPIC,
    SERVER_TYPE_OPENROUTER,
    SERVER_TYPE_OPENCLAW,
    SERVER_TYPE_VLLM,
    DEFAULT_SERVER_TYPE,
    DEFAULT_LMSTUDIO_URL,
    DEFAULT_LLAMACPP_URL,
    DEFAULT_OLLAMA_URL,
    CONF_OPENCLAW_HOST,
    CONF_OPENCLAW_PORT,
    CONF_OPENCLAW_TOKEN,
    CONF_OPENCLAW_USE_SSL,
    CONF_OPENCLAW_SESSION_KEY,
    DEFAULT_OPENCLAW_HOST,
    DEFAULT_OPENCLAW_PORT,
    DEFAULT_OPENCLAW_USE_SSL,
    DEFAULT_OPENCLAW_SESSION_KEY,
    DEFAULT_VLLM_URL,
    DEFAULT_MCP_PORT,
    DEFAULT_MODEL_NAME,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TECHNICAL_PROMPT,
    PROMPT_MODE_DEFAULT,
    PROMPT_MODE_CUSTOM,
    DEFAULT_CONTROL_HA,
    DEFAULT_RESPONSE_MODE,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_HISTORY,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_DEBUG_MODE,
    DEFAULT_BRAVE_API_KEY,
    DEFAULT_ALLOWED_IPS,
    DEFAULT_INCLUDE_CURRENT_USER,
    DEFAULT_INCLUDE_HOME_LOCATION,
    DEFAULT_SEARCH_PROVIDER,
    DEFAULT_ENABLE_WEB_SEARCH,
    DEFAULT_ENABLE_GAP_FILLING,
    DEFAULT_ENABLE_ASSIST_BRIDGE,
    DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
    DEFAULT_ENABLE_WEATHER_FORECAST_TOOL,
    DEFAULT_ENABLE_RECORDER_TOOLS,
    DEFAULT_ENABLE_MEMORY_TOOLS,
    DEFAULT_ENABLE_CALCULATOR_TOOLS,
    DEFAULT_ENABLE_UNIT_CONVERSION_TOOLS,
    DEFAULT_ENABLE_DEVICE_TOOLS,
    DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    DEFAULT_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
    DEFAULT_MEMORY_MAX_TTL_DAYS,
    DEFAULT_MEMORY_MAX_ITEMS,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_NUM_CTX,
    DEFAULT_FOLLOW_UP_PHRASES,
    DEFAULT_END_WORDS,
    DEFAULT_CLEAN_RESPONSES,
    DEFAULT_TIMEOUT,
    TOOL_FAMILY_ASSIST_BRIDGE,
    TOOL_FAMILY_DEVICE,
    TOOL_FAMILY_EXTERNAL_CUSTOM,
    TOOL_FAMILY_MEMORY,
    TOOL_FAMILY_PROFILE_SETTINGS,
    TOOL_FAMILY_SHARED_SETTINGS,
    OPENAI_BASE_URL,
    OPENROUTER_BASE_URL,
)

_LOGGER = logging.getLogger(__name__)


def _prompt_mode_selector() -> SelectSelector:
    """Build a prompt source selector."""
    return SelectSelector(
        SelectSelectorConfig(
            options=[PROMPT_MODE_DEFAULT, PROMPT_MODE_CUSTOM],
            mode=SelectSelectorMode.DROPDOWN,
            translation_key="prompt_source_mode",
        )
    )


def _get_default_system_prompt(hass: HomeAssistant) -> str:
    """Get the localized default system prompt."""
    return get_language_instruction(hass.config.language) or DEFAULT_SYSTEM_PROMPT


def _infer_prompt_mode(
    explicit_mode: Any, stored_prompt: Any, default_prompt: str
) -> str:
    """Infer prompt mode with backward compatibility for older entries."""
    if explicit_mode in {PROMPT_MODE_DEFAULT, PROMPT_MODE_CUSTOM}:
        return explicit_mode
    if stored_prompt in (None, "", default_prompt):
        return PROMPT_MODE_DEFAULT
    return PROMPT_MODE_CUSTOM


def _get_current_prompt_mode(
    current_values: dict[str, Any] | None,
    *,
    mode_key: str,
    stored_mode: Any,
    stored_prompt: Any,
    default_prompt: str,
) -> str:
    """Get the prompt mode from current form values or stored settings."""
    if current_values and mode_key in current_values:
        current_mode = current_values.get(mode_key)
        if current_mode in {PROMPT_MODE_DEFAULT, PROMPT_MODE_CUSTOM}:
            return current_mode

    return _infer_prompt_mode(stored_mode, stored_prompt, default_prompt)


def _get_prompt_text_default(
    current_values: dict[str, Any] | None,
    *,
    prompt_key: str,
    stored_mode: Any = None,
    stored_prompt: Any,
    default_prompt: str = "",
) -> str:
    """Get the effective prompt text to prefill in the form."""
    if current_values and prompt_key in current_values:
        value = current_values.get(prompt_key)
        return "" if value is None else str(value)

    inferred_mode = _infer_prompt_mode(stored_mode, stored_prompt, default_prompt)
    if inferred_mode == PROMPT_MODE_DEFAULT or stored_prompt in (None, ""):
        return str(default_prompt)

    return str(stored_prompt)


def _normalize_prompt_inputs(
    user_input: dict[str, Any], server_type: str, default_system_prompt: str
) -> dict[str, Any]:
    """Normalize prompt override inputs before storing."""
    normalized = dict(user_input)

    def _normalize_prompt(prompt_key: str, mode_key: str, default_prompt: str) -> None:
        raw_value = normalized.get(prompt_key, "")
        text = "" if raw_value is None else str(raw_value)
        if not text.strip() or text.strip() == default_prompt.strip():
            normalized.pop(prompt_key, None)
            normalized[mode_key] = PROMPT_MODE_DEFAULT
        else:
            normalized[prompt_key] = text
            normalized[mode_key] = PROMPT_MODE_CUSTOM

    if server_type == SERVER_TYPE_OPENCLAW:
        normalized[CONF_SYSTEM_PROMPT_MODE] = PROMPT_MODE_DEFAULT
        normalized.pop(CONF_SYSTEM_PROMPT, None)
    else:
        _normalize_prompt(
            CONF_SYSTEM_PROMPT,
            CONF_SYSTEM_PROMPT_MODE,
            default_system_prompt,
        )

    _normalize_prompt(
        CONF_TECHNICAL_PROMPT,
        CONF_TECHNICAL_PROMPT_MODE,
        DEFAULT_TECHNICAL_PROMPT,
    )

    return normalized


def _get_form_value(
    current_values: dict[str, Any] | None, key: str, fallback: Any
) -> Any:
    """Prefer in-progress form values over stored defaults."""
    if current_values and key in current_values:
        return current_values[key]
    return fallback


def _optional_with_suggested_value(key: str, suggested_value: str | None) -> vol.Optional:
    """Build an optional marker that pre-fills a value without forcing it."""
    if suggested_value not in (None, ""):
        return vol.Optional(key, description={"suggested_value": suggested_value})
    return vol.Optional(key)


TOOLS_SECTION_KEY = "tools"
DISCOVERY_SECTION_KEY = "discovery"
CONTEXT_SECTION_KEY = "context"
MEMORY_SECTION_KEY = "memory"
PROFILE_SECTION_KEY = "profile"
CONNECTION_SECTION_KEY = "connection"
MODEL_SECTION_KEY = "model_fields"
PROMPTS_SECTION_KEY = "prompts"
CONVERSATION_SECTION_KEY = "conversation"
PERFORMANCE_SECTION_KEY = "performance"
PROVIDER_SECTION_KEY = "provider"
ADVANCED_SECTION_KEY = "advanced_settings"
DISABLE_ASSIST_BRIDGE_FIELD = "disable_assist_bridge"
DISABLE_CUSTOM_TOOLS_FIELD = "disable_custom_tools"
DISABLE_DEVICE_FIELD = "disable_device"
DISABLE_MEMORY_FIELD = "disable_memory"
DISABLE_MUSIC_ASSISTANT_FIELD = "disable_music_assistant"
DISABLE_RECORDER_FIELD = "disable_recorder"
DISABLE_RESPONSE_SERVICE_FIELD = "disable_response_service"
DISABLE_WEATHER_FORECAST_FIELD = "disable_weather_forecast"

STATIC_TOOL_FAMILY_ALPHABETICAL = [
    TOOL_FAMILY_ASSIST_BRIDGE,
    TOOL_FAMILY_EXTERNAL_CUSTOM,
    TOOL_FAMILY_DEVICE,
    TOOL_FAMILY_MEMORY,
]

PROFILE_DISABLE_FIELD_BY_FAMILY = {
    TOOL_FAMILY_ASSIST_BRIDGE: DISABLE_ASSIST_BRIDGE_FIELD,
    TOOL_FAMILY_EXTERNAL_CUSTOM: DISABLE_CUSTOM_TOOLS_FIELD,
    TOOL_FAMILY_DEVICE: DISABLE_DEVICE_FIELD,
    TOOL_FAMILY_MEMORY: DISABLE_MEMORY_FIELD,
}

STATIC_TOOL_FAMILY_SHARED_LABELS = {
    TOOL_FAMILY_ASSIST_BRIDGE: "Assist Bridge",
    TOOL_FAMILY_EXTERNAL_CUSTOM: "Custom Tools",
    TOOL_FAMILY_DEVICE: "Device Tools",
    TOOL_FAMILY_MEMORY: "Memory",
}

STATIC_TOOL_FAMILY_PROFILE_DISABLE_LABELS = {
    TOOL_FAMILY_ASSIST_BRIDGE: "Disable Assist Bridge",
    TOOL_FAMILY_EXTERNAL_CUSTOM: "Disable Custom Tools",
    TOOL_FAMILY_DEVICE: "Disable Device Tools",
    TOOL_FAMILY_MEMORY: "Disable Memory",
}


def _flatten_section_values(
    user_input: dict[str, Any], *section_keys: str
) -> dict[str, Any]:
    """Flatten nested section payloads into a plain config dict."""
    normalized = dict(user_input)

    for section_key in section_keys:
        section_values = normalized.pop(section_key, None)
        if isinstance(section_values, dict):
            normalized.update(section_values)

    return normalized


async def _async_load_builtin_tool_toggle_specs(
    hass: HomeAssistant,
) -> tuple[BuiltInToolToggleSpec, ...]:
    """Load built-in packaged-tool metadata asynchronously."""
    return await hass.async_add_executor_job(load_builtin_tool_toggle_specs)


def _builtin_shared_field_key(spec: BuiltInToolToggleSpec) -> str:
    """Return the stable shared-form checkbox key for a built-in package."""
    return spec.shared_setting_key


def _builtin_profile_disable_field_key(spec: BuiltInToolToggleSpec) -> str:
    """Return the stable profile disable checkbox key for a built-in package."""
    return f"disable_{spec.package_id}"


def _profile_tool_disabled_default(
    current_values: dict[str, Any] | None,
    family: str,
    options: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> bool:
    """Return whether a profile tool family should default to disabled in the form."""
    options = options or {}
    data = data or {}
    disable_field = PROFILE_DISABLE_FIELD_BY_FAMILY[family]
    if current_values and disable_field in current_values:
        return bool(current_values[disable_field])

    setting_key, _default = TOOL_FAMILY_PROFILE_SETTINGS[family]
    stored_value = options.get(setting_key, data.get(setting_key))
    return stored_value is False


def _builtin_profile_tool_disabled_default(
    current_values: dict[str, Any] | None,
    spec: BuiltInToolToggleSpec,
    options: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> bool:
    """Return whether a built-in packaged tool should default to disabled."""
    options = options or {}
    data = data or {}
    disable_field = _builtin_profile_disable_field_key(spec)
    if current_values and disable_field in current_values:
        return bool(current_values[disable_field])

    stored_value = get_builtin_profile_setting_value(
        spec,
        lambda key, default=None: options.get(key, data.get(key, default)),
    )
    return stored_value is False


def _apply_profile_tool_disables(
    user_input: dict[str, Any],
    built_in_specs: tuple[BuiltInToolToggleSpec, ...] = (),
) -> dict[str, Any]:
    """Map profile disable checkboxes to stored profile enable flags."""
    normalized = dict(user_input)
    for family in STATIC_TOOL_FAMILY_ALPHABETICAL:
        disable_field = PROFILE_DISABLE_FIELD_BY_FAMILY[family]
        disabled = bool(normalized.pop(disable_field, False))
        setting_key, _default = TOOL_FAMILY_PROFILE_SETTINGS[family]
        if disabled:
            normalized[setting_key] = False
        else:
            normalized.pop(setting_key, None)

    for spec in built_in_specs:
        disable_field = _builtin_profile_disable_field_key(spec)
        disabled = bool(normalized.pop(disable_field, False))
        if not disabled and spec.profile_disable_label in normalized:
            disabled = bool(normalized.pop(spec.profile_disable_label, False))
        if disabled:
            normalized[spec.profile_setting_key] = False
        else:
            normalized.pop(spec.profile_setting_key, None)

    return normalized


def _normalize_search_provider(value: Any) -> str:
    """Normalize a stored search provider value."""
    normalized = str(value or "").strip().casefold()
    return normalized or DEFAULT_SEARCH_PROVIDER


def _infer_web_search_enabled(
    explicit_enabled: Any,
    search_provider: Any,
    legacy_enable_custom_tools: Any = False,
) -> bool:
    """Infer whether web search should be enabled."""
    if explicit_enabled is not None:
        return bool(explicit_enabled)

    provider = _normalize_search_provider(search_provider)
    if provider != DEFAULT_SEARCH_PROVIDER:
        return True

    return bool(legacy_enable_custom_tools)


def _normalize_shared_tool_inputs(
    user_input: dict[str, Any],
    built_in_specs: tuple[BuiltInToolToggleSpec, ...] = (),
) -> dict[str, Any]:
    """Normalize shared tool-family settings before storing."""
    normalized = dict(user_input)

    for spec in built_in_specs:
        field_key = _builtin_shared_field_key(spec)
        if field_key in normalized:
            normalized[spec.shared_setting_key] = bool(normalized.pop(field_key))
            continue

        legacy_label_key = spec.shared_label
        if legacy_label_key in normalized:
            normalized[spec.shared_setting_key] = bool(normalized.pop(legacy_label_key))

    search_provider = _normalize_search_provider(
        normalized.get(CONF_SEARCH_PROVIDER, DEFAULT_SEARCH_PROVIDER)
    )

    built_in_search_enabled = any(
        bool(normalized.get(spec.shared_setting_key, spec.shared_default))
        for spec in built_in_specs
        if spec.requires_search_provider
    )
    legacy_web_search_enabled = bool(
        normalized.get(CONF_ENABLE_WEB_SEARCH, DEFAULT_ENABLE_WEB_SEARCH)
    )

    if (
        built_in_search_enabled or legacy_web_search_enabled
    ) and search_provider == DEFAULT_SEARCH_PROVIDER:
        normalized[CONF_SEARCH_PROVIDER] = "duckduckgo"
    else:
        normalized[CONF_SEARCH_PROVIDER] = search_provider

    memory_max_ttl = normalized.get(
        CONF_MEMORY_MAX_TTL_DAYS,
        DEFAULT_MEMORY_MAX_TTL_DAYS,
    )
    try:
        memory_max_ttl = max(1, int(memory_max_ttl))
    except (TypeError, ValueError):
        memory_max_ttl = DEFAULT_MEMORY_MAX_TTL_DAYS
    normalized[CONF_MEMORY_MAX_TTL_DAYS] = memory_max_ttl

    memory_default_ttl = normalized.get(
        CONF_MEMORY_DEFAULT_TTL_DAYS,
        DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
    )
    try:
        memory_default_ttl = int(memory_default_ttl)
    except (TypeError, ValueError):
        memory_default_ttl = DEFAULT_MEMORY_DEFAULT_TTL_DAYS
    normalized[CONF_MEMORY_DEFAULT_TTL_DAYS] = max(1, min(memory_default_ttl, memory_max_ttl))

    return normalized


def _build_profile_tools_section(
    current_values: dict[str, Any] | None,
    built_in_specs: tuple[BuiltInToolToggleSpec, ...],
    options: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> section:
    """Build the per-profile tool preferences section."""
    options = options or {}
    data = data or {}
    profile_tool_entries: list[tuple[str, vol.Optional, type[bool]]] = []
    for family in STATIC_TOOL_FAMILY_ALPHABETICAL:
        disable_field = PROFILE_DISABLE_FIELD_BY_FAMILY[family]
        profile_tool_entries.append(
            (
                STATIC_TOOL_FAMILY_PROFILE_DISABLE_LABELS[family].casefold(),
                vol.Optional(
                    disable_field,
                    default=_profile_tool_disabled_default(
                        current_values, family, options, data
                    ),
                ),
                bool,
            )
        )

    for spec in built_in_specs:
        profile_tool_entries.append(
            (
                spec.profile_disable_label.casefold(),
                vol.Optional(
                    _builtin_profile_disable_field_key(spec),
                    default=_builtin_profile_tool_disabled_default(
                        current_values,
                        spec,
                        options,
                        data,
                    ),
                ),
                bool,
            )
        )

    profile_tool_fields = {
        marker: value_type
        for _label, marker, value_type in sorted(
            profile_tool_entries,
            key=lambda item: item[0],
        )
    }

    return section(
        vol.Schema(profile_tool_fields),
        {"collapsed": False},
    )


def _build_shared_tools_section(
    defaults: dict[str, Any],
    built_in_specs: tuple[BuiltInToolToggleSpec, ...],
) -> section:
    """Build the shared MCP server optional tools section."""
    shared_tool_entries: list[tuple[str, vol.Optional, type[bool]]] = []
    for family in STATIC_TOOL_FAMILY_ALPHABETICAL:
        setting_key, default = TOOL_FAMILY_SHARED_SETTINGS[family]
        shared_tool_entries.append(
            (
                STATIC_TOOL_FAMILY_SHARED_LABELS[family].casefold(),
                vol.Optional(
                    setting_key,
                    default=_get_form_value(defaults, setting_key, default),
                ),
                bool,
            )
        )

    for spec in built_in_specs:
        shared_tool_entries.append(
            (
                spec.shared_label.casefold(),
                vol.Optional(
                    _builtin_shared_field_key(spec),
                    default=_get_form_value(
                        defaults,
                        spec.shared_setting_key,
                        spec.shared_default,
                    ),
                ),
                bool,
            )
        )

    shared_tool_fields = {
        marker: value_type
        for _label, marker, value_type in sorted(
            shared_tool_entries,
            key=lambda item: item[0],
        )
    }

    return section(
        vol.Schema(
            {
                **shared_tool_fields,
                vol.Required(
                    CONF_SEARCH_PROVIDER,
                    default=_get_form_value(
                        defaults, CONF_SEARCH_PROVIDER, DEFAULT_SEARCH_PROVIDER
                    ),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            {"value": "duckduckgo", "label": "DuckDuckGo"},
                            {
                                "value": "brave",
                                "label": "Brave Search (requires API key)",
                            },
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_BRAVE_API_KEY,
                    default=_get_form_value(
                        defaults, CONF_BRAVE_API_KEY, DEFAULT_BRAVE_API_KEY
                    ),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
            }
        ),
        {"collapsed": False},
    )


def _build_shared_discovery_section(defaults: dict[str, Any]) -> section:
    """Build the shared MCP server discovery settings section."""
    return section(
        vol.Schema(
            {
                vol.Optional(
                    CONF_ENABLE_GAP_FILLING,
                    default=defaults[CONF_ENABLE_GAP_FILLING],
                ): bool,
                vol.Optional(
                    CONF_MAX_ENTITIES_PER_DISCOVERY,
                    default=defaults[CONF_MAX_ENTITIES_PER_DISCOVERY],
                ): vol.All(vol.Coerce(int), vol.Range(min=20, max=500)),
            }
        ),
        {"collapsed": False},
    )


def _build_shared_context_section(defaults: dict[str, Any]) -> section:
    """Build the shared AI-context settings section."""
    return section(
        vol.Schema(
            {
                vol.Optional(
                    CONF_INCLUDE_CURRENT_USER,
                    default=defaults[CONF_INCLUDE_CURRENT_USER],
                ): bool,
                vol.Optional(
                    CONF_INCLUDE_HOME_LOCATION,
                    default=defaults[CONF_INCLUDE_HOME_LOCATION],
                ): bool,
            }
        ),
        {"collapsed": False},
    )


def _build_shared_memory_section(defaults: dict[str, Any]) -> section:
    """Build shared persisted-memory settings."""
    return section(
        vol.Schema(
            {
                vol.Optional(
                    CONF_MEMORY_DEFAULT_TTL_DAYS,
                    default=defaults[CONF_MEMORY_DEFAULT_TTL_DAYS],
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=3650)),
                vol.Optional(
                    CONF_MEMORY_MAX_TTL_DAYS,
                    default=defaults[CONF_MEMORY_MAX_TTL_DAYS],
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=3650)),
                vol.Optional(
                    CONF_MEMORY_MAX_ITEMS,
                    default=defaults[CONF_MEMORY_MAX_ITEMS],
                ): vol.All(vol.Coerce(int), vol.Range(min=10, max=5000)),
            }
        ),
        {"collapsed": False},
    )


def _build_profile_identity_section(profile_name: str) -> section:
    """Build the profile identity section."""
    return section(
        vol.Schema(
            {
                vol.Required(CONF_PROFILE_NAME, default=profile_name): str,
            }
        ),
        {"collapsed": False},
    )


def _build_connection_section(schema_items: dict[Any, Any]) -> section:
    """Wrap connection-related fields in a section."""
    return section(vol.Schema(schema_items), {"collapsed": False})


def _build_model_section(current_model: str, model_field: Any) -> section:
    """Build the model-selection section."""
    return section(
        vol.Schema(
            {
                vol.Required(CONF_MODEL_NAME, default=current_model): model_field,
            }
        ),
        {"collapsed": False},
    )


def _build_prompt_section(
    *,
    system_prompt_value: str | None = None,
    technical_prompt_value: str | None,
    include_system_prompt: bool = True,
) -> section:
    """Build the prompt editing section."""
    schema_items: dict[Any, Any] = {}
    if include_system_prompt:
        schema_items[
            _optional_with_suggested_value(CONF_SYSTEM_PROMPT, system_prompt_value)
        ] = TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True))
    schema_items[
        _optional_with_suggested_value(
            CONF_TECHNICAL_PROMPT, technical_prompt_value
        )
    ] = TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True))
    return section(vol.Schema(schema_items), {"collapsed": False})


def _build_conversation_section(schema_items: dict[Any, Any]) -> section:
    """Wrap conversation-behavior fields in a section."""
    return section(vol.Schema(schema_items), {"collapsed": False})


def _build_performance_section(schema_items: dict[Any, Any]) -> section:
    """Wrap performance-related fields in a section."""
    return section(vol.Schema(schema_items), {"collapsed": False})


def _build_provider_section(schema_items: dict[Any, Any]) -> section:
    """Wrap provider-specific fields in a section."""
    return section(vol.Schema(schema_items), {"collapsed": False})


def _build_advanced_section(schema_items: dict[Any, Any]) -> section:
    """Wrap advanced settings in a collapsed section."""
    return section(vol.Schema(schema_items), {"collapsed": True})


def _needs_prompt_followup(
    user_input: dict[str, Any], server_type: str
) -> bool:
    """Config-flow forms are static per step; prompt followup is no longer used."""
    del user_input, server_type
    return False


async def fetch_models_from_lmstudio(hass: HomeAssistant, url: str) -> list[str]:
    """Fetch available models from local inference server (LM Studio/Ollama)."""
    _LOGGER.info("🌐 FETCH: Starting model fetch from %s", url)
    try:
        # Small delay to ensure server is ready
        await asyncio.sleep(0.5)

        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            _LOGGER.info("📡 FETCH: Sending request to %s/v1/models", url)
            async with session.get(f"{url}/v1/models") as resp:
                _LOGGER.info("📥 FETCH: Got response with status %d", resp.status)
                if resp.status != 200:
                    _LOGGER.warning("⚠️ FETCH: Non-200 status, returning empty list")
                    return []

                models = await resp.json()
                model_ids = [m.get("id", "") for m in models.get("data", [])]
                sorted_models = sorted(model_ids) if model_ids else []
                _LOGGER.info(
                    "✨ FETCH: Returning %d sorted models: %s",
                    len(sorted_models),
                    sorted_models,
                )
                return sorted_models
    except Exception as err:
        _LOGGER.error("💥 FETCH: Exception during fetch: %s", err, exc_info=True)
        return []


async def fetch_models_from_openai(
    hass: HomeAssistant,
    api_key: str,
    base_url: str = OPENAI_BASE_URL
) -> list[str]:
    """Fetch available models from OpenAI API."""
    _LOGGER.info("🌐 FETCH: Starting OpenAI model fetch")
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        headers = {
            "Content-Type": "application/json",
        }

        # Only include Authorization header if API key looks valid
        # Some custom OpenAI-compatible services don't require authentication
        if api_key and len(api_key) > 5 and api_key.lower() not in ["none", "null", "fake", "na", "n/a"]:
            headers["Authorization"] = f"Bearer {api_key}"

        async with aiohttp.ClientSession(timeout=timeout) as session:
            _LOGGER.info("📡 FETCH: Requesting OpenAI models")
            async with session.get(
                f"{base_url}/v1/models", headers=headers
            ) as resp:
                _LOGGER.info("📥 FETCH: OpenAI response status %d", resp.status)
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.warning(
                        "⚠️ FETCH: OpenAI API error %d: %s",
                        resp.status,
                        error_text[:200],
                    )
                    return []

                data = await resp.json()
                # Filter for chat models only (exclude embeddings, whisper, etc.)
                all_models = [m.get("id", "") for m in data.get("data", [])]

                # Only filter for GPT models when using official OpenAI URL
                # Custom OpenAI-compatible services may use different naming schemes
                if base_url == OPENAI_BASE_URL:
                    chat_models = [m for m in all_models if m.startswith("gpt-")]
                else:
                    # For custom URLs, return all models (user's service defines what's available)
                    chat_models = all_models

                sorted_models = sorted(chat_models, reverse=True) if chat_models else []
                _LOGGER.info("✨ FETCH: Found %d OpenAI chat models", len(sorted_models))
                return sorted_models
    except Exception as err:
        _LOGGER.error("💥 FETCH: OpenAI fetch failed: %s", err)
        return []


async def fetch_models_from_gemini(hass: HomeAssistant, api_key: str) -> list[str]:
    """Fetch available models from Gemini API."""
    _LOGGER.info("🌐 FETCH: Starting Gemini model fetch")
    try:
        timeout = aiohttp.ClientTimeout(total=10)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            _LOGGER.info("📡 FETCH: Requesting Gemini models")
            # Gemini uses native API for model listing, not OpenAI-compatible endpoint
            # API key goes in query parameter for native API
            async with session.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            ) as resp:
                _LOGGER.info("📥 FETCH: Gemini response status %d", resp.status)
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.warning(
                        "⚠️ FETCH: Gemini API error %d: %s",
                        resp.status,
                        error_text[:200],
                    )
                    return []

                data = await resp.json()
                # Gemini native API response format: {"models": [{"name": "models/gemini-..."}]}
                all_models = []
                for model in data.get("models", []):
                    # Extract model ID from "models/gemini-pro" format
                    model_name = model.get("name", "")
                    if model_name.startswith("models/"):
                        model_id = model_name.replace("models/", "")
                        all_models.append(model_id)

                # Filter for gemini models only
                gemini_models = [m for m in all_models if "gemini" in m.lower()]
                sorted_models = (
                    sorted(gemini_models, reverse=True) if gemini_models else []
                )
                _LOGGER.info("✨ FETCH: Found %d Gemini models", len(sorted_models))
                return sorted_models
    except Exception as err:
        _LOGGER.error("💥 FETCH: Gemini fetch failed: %s", err)
        return []


async def fetch_models_from_openrouter(hass: HomeAssistant, api_key: str) -> list[str]:
    """Fetch available models from OpenRouter API."""
    _LOGGER.info("🌐 FETCH: Starting OpenRouter model fetch")
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/mike-nott/mcp-assist",
            "X-Title": "MCP Assist for Home Assistant",
        }

        async with aiohttp.ClientSession(timeout=timeout) as session:
            _LOGGER.info("📡 FETCH: Requesting OpenRouter models")
            async with session.get(
                f"{OPENROUTER_BASE_URL}/v1/models", headers=headers
            ) as resp:
                _LOGGER.info("📥 FETCH: OpenRouter response status %d", resp.status)
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.warning(
                        "⚠️ FETCH: OpenRouter API error %d: %s",
                        resp.status,
                        error_text[:200],
                    )
                    return []

                data = await resp.json()
                # OpenRouter returns models in OpenAI-compatible format
                all_models = [m.get("id", "") for m in data.get("data", [])]
                # Filter out empty strings and sort
                models = [m for m in all_models if m]
                sorted_models = sorted(models) if models else []
                _LOGGER.info("✨ FETCH: Found %d OpenRouter models", len(sorted_models))
                return sorted_models
    except Exception as err:
        _LOGGER.error("💥 FETCH: OpenRouter fetch failed: %s", err)
        return []


def validate_allowed_ips(allowed_ips_str: str) -> tuple[bool, str]:
    """Validate comma-separated list of IP addresses and CIDR ranges.

    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is empty string
    """
    if not allowed_ips_str or not allowed_ips_str.strip():
        # Empty is valid (no additional IPs)
        return True, ""

    # Parse comma-separated values
    ip_list = [ip.strip() for ip in allowed_ips_str.split(",") if ip.strip()]

    for ip_entry in ip_list:
        try:
            # Try parsing as IP network (handles both individual IPs and CIDR)
            ipaddress.ip_network(ip_entry, strict=False)
        except ValueError:
            # Invalid IP or CIDR format
            return False, f"Invalid IP address or CIDR range: {ip_entry}"

    return True, ""


STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_PROFILE_NAME): str,
        vol.Required(CONF_SERVER_TYPE, default=DEFAULT_SERVER_TYPE): SelectSelector(
            SelectSelectorConfig(
                options=[
                    {"value": "lmstudio", "label": "LM Studio"},
                    {"value": "llamacpp", "label": "llama.cpp"},
                    {"value": "ollama", "label": "Ollama"},
                    {"value": "openai", "label": "OpenAI"},
                    {"value": "gemini", "label": "Google Gemini"},
                    {"value": "anthropic", "label": "Anthropic (Claude)"},
                    {"value": "openrouter", "label": "OpenRouter"},
                    {"value": "openclaw", "label": "OpenClaw"},
                    {"value": "vllm", "label": "vLLM"},
                ],
                mode=SelectSelectorMode.LIST,
            )
        ),
    }
)

STEP_MCP_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_MCP_PORT, default=DEFAULT_MCP_PORT): vol.Coerce(int),
        vol.Required(CONF_AUTO_START, default=True): bool,
    }
)


async def validate_lmstudio_connection(
    hass: HomeAssistant, data: dict[str, Any]
) -> dict[str, Any]:
    """Validate LM Studio connection."""
    url = data[CONF_LMSTUDIO_URL].rstrip("/")
    model_name = data[CONF_MODEL_NAME]

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Test models endpoint
            async with session.get(f"{url}/v1/models") as resp:
                if resp.status != 200:
                    raise CannotConnect(
                        f"LM Studio not responding (status {resp.status})"
                    )

                models = await resp.json()
                model_ids = [m.get("id", "") for m in models.get("data", [])]

                if not model_ids:
                    raise NoModelsLoaded("No models loaded in LM Studio")

                # Check if specified model exists
                if model_name not in model_ids:
                    _LOGGER.warning(
                        "Model '%s' not found. Available models: %s",
                        model_name,
                        model_ids,
                    )

            # Test chat completions endpoint
            test_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
                "stream": False,
            }

            async with session.post(
                f"{url}/v1/chat/completions", json=test_payload
            ) as resp:
                if resp.status != 200:
                    raise InvalidModel(
                        f"Model '{model_name}' not working (status {resp.status})"
                    )

    except aiohttp.ClientError as err:
        raise CannotConnect(f"Failed to connect to LM Studio: {err}") from err

    return {"title": f"LM Studio ({model_name})"}


class MCPAssistConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for MCP Assist."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize."""
        self.step1_data: dict[str, Any] = {}
        self.step2_data: dict[str, Any] = {}
        self.step3_data: dict[str, Any] = {}
        self.step4_data: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle step 1 - profile name and server type."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate profile name is not empty
            profile_name = user_input.get(CONF_PROFILE_NAME, "").strip()
            if not profile_name:
                errors[CONF_PROFILE_NAME] = "profile_name_required"
            else:
                # Store data and move to step 2
                self.step1_data = user_input
                return await self.async_step_server()

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
        )

    async def async_step_server(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle step 2 - server configuration (URL for local, API key for cloud)."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Store data and move to next step
            self.step2_data = user_input
            server_type = self.step1_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)
            if server_type == SERVER_TYPE_OPENCLAW:
                return await self.async_step_openclaw_pairing()
            return await self.async_step_model()

        # Get server type from step 1 to build dynamic schema
        server_type = self.step1_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)

        # Build schema based on server type
        if server_type == SERVER_TYPE_OPENCLAW:
            # OpenClaw Gateway - host, port, token, SSL
            server_schema = vol.Schema(
                {
                    vol.Required(CONF_OPENCLAW_HOST, default=DEFAULT_OPENCLAW_HOST): str,
                    vol.Required(CONF_OPENCLAW_PORT, default=DEFAULT_OPENCLAW_PORT): vol.Coerce(int),
                    vol.Required(CONF_OPENCLAW_TOKEN): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    ),
                    vol.Required(CONF_OPENCLAW_USE_SSL, default=DEFAULT_OPENCLAW_USE_SSL): BooleanSelector(),
                }
            )
        elif server_type in [
            SERVER_TYPE_LMSTUDIO,
            SERVER_TYPE_LLAMACPP,
            SERVER_TYPE_OLLAMA,
            SERVER_TYPE_VLLM,
        ]:
            # Local servers - show URL field
            if server_type == SERVER_TYPE_OLLAMA:
                default_url = DEFAULT_OLLAMA_URL
            elif server_type == SERVER_TYPE_LLAMACPP:
                default_url = DEFAULT_LLAMACPP_URL
            elif server_type == SERVER_TYPE_VLLM:
                default_url = DEFAULT_VLLM_URL
            else:
                default_url = DEFAULT_LMSTUDIO_URL

            server_schema = vol.Schema(
                {
                    vol.Required(CONF_LMSTUDIO_URL, default=default_url): str,
                }
            )
        elif server_type == SERVER_TYPE_OPENAI:
            # OpenAI - hybrid like OpenClaw (URL + API key)
            # Pre-fill with official OpenAI URL but allow users to edit for custom endpoints
            server_schema = vol.Schema(
                {
                    vol.Required(CONF_LMSTUDIO_URL, default=OPENAI_BASE_URL): str,
                    vol.Required(CONF_API_KEY): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    ),
                }
            )
        else:
            # Other cloud providers (Gemini, Anthropic, OpenRouter) - API key only
            server_schema = vol.Schema(
                {
                    vol.Required(CONF_API_KEY): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    ),
                }
            )

        return self.async_show_form(
            step_id="server",
            data_schema=server_schema,
            errors=errors,
        )

    async def async_step_openclaw_pairing(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle OpenClaw device pairing step."""
        errors: dict[str, str] = {}

        from .openclaw_client import (
            OpenClawClient, OpenClawDeviceAuth, DevicePairingRequiredError,
            OpenClawConnectionError, OpenClawAuthError,
        )

        # Get or create device auth
        if "openclaw_device_auth" not in self.hass.data.get(DOMAIN, {}):
            self.hass.data.setdefault(DOMAIN, {})
            device_auth = OpenClawDeviceAuth(self.hass)
            await device_auth.async_load()
            self.hass.data[DOMAIN]["openclaw_device_auth"] = device_auth

        device_auth = self.hass.data[DOMAIN]["openclaw_device_auth"]
        device_id = device_auth.device_id

        if user_input is not None or not hasattr(self, "_pairing_attempted"):
            # Attempt connection to test pairing
            self._pairing_attempted = True
            client = OpenClawClient(
                host=self.step2_data.get(CONF_OPENCLAW_HOST, DEFAULT_OPENCLAW_HOST),
                port=self.step2_data.get(CONF_OPENCLAW_PORT, DEFAULT_OPENCLAW_PORT),
                token=self.step2_data.get(CONF_OPENCLAW_TOKEN, ""),
                use_ssl=self.step2_data.get(CONF_OPENCLAW_USE_SSL, DEFAULT_OPENCLAW_USE_SSL),
                device_auth=device_auth,
                timeout=30,
            )

            try:
                await client.connect()
                await client.disconnect()
                # Device is approved — proceed to model step
                return await self.async_step_model()
            except DevicePairingRequiredError:
                errors["base"] = "openclaw_not_paired"
            except OpenClawAuthError as err:
                _LOGGER.error("OpenClaw auth error: %s", err)
                errors["base"] = "openclaw_connection_failed"
            except OpenClawConnectionError as err:
                _LOGGER.error("OpenClaw connection error: %s", err)
                errors["base"] = "openclaw_connection_failed"
            except Exception as err:
                _LOGGER.error("OpenClaw unexpected error: %s: %s", type(err).__name__, err)
                errors["base"] = "openclaw_connection_failed"

        return self.async_show_form(
            step_id="openclaw_pairing",
            data_schema=vol.Schema({}),
            errors=errors,
            description_placeholders={
                "device_id": device_id,
            },
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle step 3 - model selection and prompts."""
        errors: dict[str, str] = {}

        # Get server type to determine model source
        server_type = self.step1_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)
        default_system_prompt = _get_default_system_prompt(self.hass)

        if user_input is not None:
            user_input = _flatten_section_values(
                user_input, MODEL_SECTION_KEY, PROMPTS_SECTION_KEY
            )
            user_input = _normalize_prompt_inputs(
                user_input, server_type, default_system_prompt
            )
            # Store data and move to step 4 (advanced)
            self.step3_data = user_input
            return await self.async_step_advanced()

        models = []
        current_values = getattr(self, "step3_data", {})

        if server_type == SERVER_TYPE_OPENCLAW:
            self.step3_data = {
                CONF_MODEL_NAME: "main",
                CONF_SYSTEM_PROMPT: "",
                CONF_TECHNICAL_PROMPT: "",
                CONF_SYSTEM_PROMPT_MODE: PROMPT_MODE_DEFAULT,
                CONF_TECHNICAL_PROMPT_MODE: PROMPT_MODE_DEFAULT,
            }
            return await self.async_step_advanced()
        elif server_type in [
            SERVER_TYPE_LMSTUDIO,
            SERVER_TYPE_LLAMACPP,
            SERVER_TYPE_OLLAMA,
            SERVER_TYPE_VLLM,
        ]:
            # Local servers - fetch models from API
            server_url = self.step2_data.get(
                CONF_LMSTUDIO_URL, DEFAULT_LMSTUDIO_URL
            ).rstrip("/")
            _LOGGER.debug("Attempting to fetch models from %s", server_url)
            models = await fetch_models_from_lmstudio(self.hass, server_url)
            _LOGGER.debug("Fetched %d models: %s", len(models), models)
            # Show error if fetch failed
            if not models:
                errors["base"] = "cannot_connect"
        elif server_type == SERVER_TYPE_OPENAI:
            # OpenAI - fetch models from API with authentication
            api_key = self.step2_data.get(CONF_API_KEY, "")
            # Get custom URL from step 2 (uses same CONF_LMSTUDIO_URL field as local servers)
            base_url = self.step2_data.get(CONF_LMSTUDIO_URL, OPENAI_BASE_URL).rstrip("/")
            _LOGGER.debug("Fetching OpenAI models from %s", base_url)
            models = await fetch_models_from_openai(self.hass, api_key, base_url)
            _LOGGER.debug("Fetched %d OpenAI models: %s", len(models), models)
            # Show error if fetch failed
            if not models:
                errors["base"] = "invalid_api_key"
        elif server_type == SERVER_TYPE_GEMINI:
            # Gemini - fetch models from API with authentication
            api_key = self.step2_data.get(CONF_API_KEY, "")
            _LOGGER.debug("Fetching Gemini models with API key")
            models = await fetch_models_from_gemini(self.hass, api_key)
            _LOGGER.debug("Fetched %d Gemini models: %s", len(models), models)
            # Show error if fetch failed
            if not models:
                errors["base"] = "invalid_api_key"
        elif server_type == SERVER_TYPE_OPENROUTER:
            # OpenRouter - fetch models from API with authentication
            api_key = self.step2_data.get(CONF_API_KEY, "")
            _LOGGER.debug("Fetching OpenRouter models with API key")
            models = await fetch_models_from_openrouter(self.hass, api_key)
            _LOGGER.debug("Fetched %d OpenRouter models: %s", len(models), models)
            # Show error if fetch failed
            if not models:
                errors["base"] = "invalid_api_key"

        # Build dynamic schema based on whether models were fetched
        current_model = current_values.get(CONF_MODEL_NAME, DEFAULT_MODEL_NAME)
        if models:
            # Show dropdown with available models (custom_value allows free text input)
            _LOGGER.info("Showing model dropdown with %d models", len(models))
            model_field = SelectSelector(
                SelectSelectorConfig(
                    options=models,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            )
        else:
            # Show text input as fallback
            _LOGGER.info("No models fetched, showing text input")
            model_field = str

        system_prompt_suggestion = _get_prompt_text_default(
            current_values,
            prompt_key=CONF_SYSTEM_PROMPT,
            stored_prompt=None,
            default_prompt=default_system_prompt,
        )
        technical_prompt_suggestion = _get_prompt_text_default(
            current_values,
            prompt_key=CONF_TECHNICAL_PROMPT,
            stored_prompt=None,
            default_prompt=DEFAULT_TECHNICAL_PROMPT,
        )

        schema_dict: dict[Any, Any] = {
            MODEL_SECTION_KEY: _build_model_section(current_model, model_field),
            PROMPTS_SECTION_KEY: _build_prompt_section(
                system_prompt_value=system_prompt_suggestion,
                technical_prompt_value=technical_prompt_suggestion,
            ),
        }

        model_schema = vol.Schema(schema_dict)

        return self.async_show_form(
            step_id="model",
            data_schema=model_schema,
            errors=errors,
            description_placeholders={
                "server_info": "Select a model. The prompt fields are prefilled with the current effective prompts so you can review, copy, or edit them directly. If you leave a prompt unchanged, the integration keeps using the built-in version from code. Models are automatically loaded from your server."
            },
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle step 4 - advanced settings."""
        errors: dict[str, str] = {}
        built_in_specs = await _async_load_builtin_tool_toggle_specs(self.hass)

        # Get server type to determine which fields to show
        server_type = self.step1_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)

        if user_input is not None:
            user_input = _flatten_section_values(
                user_input,
                CONVERSATION_SECTION_KEY,
                PERFORMANCE_SECTION_KEY,
                PROVIDER_SECTION_KEY,
                TOOLS_SECTION_KEY,
            )
            user_input = _apply_profile_tool_disables(user_input, built_in_specs)

            # For OpenClaw, set defaults for LLM-specific fields (not shown in UI)
            if server_type == SERVER_TYPE_OPENCLAW:
                user_input[CONF_TEMPERATURE] = DEFAULT_TEMPERATURE
                user_input[CONF_MAX_TOKENS] = DEFAULT_MAX_TOKENS
                user_input[CONF_MAX_HISTORY] = DEFAULT_MAX_HISTORY
                user_input[CONF_MAX_ITERATIONS] = DEFAULT_MAX_ITERATIONS
                if CONF_TIMEOUT not in user_input:
                    user_input[CONF_TIMEOUT] = 60

            # Validate MCP port
            mcp_port = user_input.get(CONF_MCP_PORT, DEFAULT_MCP_PORT)
            if not 1024 <= mcp_port <= 65535:
                errors[CONF_MCP_PORT] = "invalid_port"

            # Validate allowed IPs
            allowed_ips_str = user_input.get(CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS)
            is_valid, error_msg = validate_allowed_ips(allowed_ips_str)
            if not is_valid:
                errors[CONF_ALLOWED_IPS] = "invalid_ip"
                _LOGGER.warning("Invalid allowed IPs: %s", error_msg)

            if not errors:
                # Check if this is the first profile (MCP server doesn't exist yet)
                is_first_profile = "shared_mcp_server" not in self.hass.data.get(
                    DOMAIN, {}
                )

                if is_first_profile:
                    # First profile - store step 4 data and proceed to MCP server config
                    self.step4_data = user_input
                    return await self.async_step_mcp_server()
                else:
                    # Subsequent profile - use existing shared MCP server settings
                    # Get MCP settings from shared server
                    mcp_port = self.hass.data[DOMAIN].get("mcp_port", DEFAULT_MCP_PORT)

                    # Get search provider from any existing entry (they all share it)
                    # Find first entry to copy shared settings from
                    existing_entry = None
                    for entry in self.hass.config_entries.async_entries(DOMAIN):
                        existing_entry = entry
                        break

                    # Copy shared settings from existing entry
                    shared_settings = {}
                    if existing_entry:
                        shared_settings = {
                            CONF_MCP_PORT: existing_entry.data.get(
                                CONF_MCP_PORT, mcp_port
                            ),
                            CONF_SEARCH_PROVIDER: existing_entry.data.get(
                                CONF_SEARCH_PROVIDER, DEFAULT_SEARCH_PROVIDER
                            ),
                            CONF_BRAVE_API_KEY: existing_entry.data.get(
                                CONF_BRAVE_API_KEY, DEFAULT_BRAVE_API_KEY
                            ),
                            CONF_ALLOWED_IPS: existing_entry.data.get(
                                CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS
                            ),
                            CONF_INCLUDE_CURRENT_USER: existing_entry.data.get(
                                CONF_INCLUDE_CURRENT_USER,
                                DEFAULT_INCLUDE_CURRENT_USER,
                            ),
                            CONF_INCLUDE_HOME_LOCATION: existing_entry.data.get(
                                CONF_INCLUDE_HOME_LOCATION,
                                DEFAULT_INCLUDE_HOME_LOCATION,
                            ),
                            CONF_ENABLE_GAP_FILLING: existing_entry.data.get(
                                CONF_ENABLE_GAP_FILLING, DEFAULT_ENABLE_GAP_FILLING
                            ),
                            CONF_ENABLE_ASSIST_BRIDGE: existing_entry.data.get(
                                CONF_ENABLE_ASSIST_BRIDGE,
                                DEFAULT_ENABLE_ASSIST_BRIDGE,
                            ),
                            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: existing_entry.data.get(
                                CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
                                DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
                            ),
                            CONF_ENABLE_WEATHER_FORECAST_TOOL: existing_entry.data.get(
                                CONF_ENABLE_WEATHER_FORECAST_TOOL,
                                DEFAULT_ENABLE_WEATHER_FORECAST_TOOL,
                            ),
                            CONF_ENABLE_RECORDER_TOOLS: existing_entry.data.get(
                                CONF_ENABLE_RECORDER_TOOLS,
                                DEFAULT_ENABLE_RECORDER_TOOLS,
                            ),
                            CONF_ENABLE_CALCULATOR_TOOLS: existing_entry.data.get(
                                CONF_ENABLE_CALCULATOR_TOOLS,
                                DEFAULT_ENABLE_CALCULATOR_TOOLS,
                            ),
                            CONF_ENABLE_UNIT_CONVERSION_TOOLS: existing_entry.data.get(
                                CONF_ENABLE_UNIT_CONVERSION_TOOLS,
                                existing_entry.data.get(
                                    CONF_ENABLE_CALCULATOR_TOOLS,
                                    DEFAULT_ENABLE_UNIT_CONVERSION_TOOLS,
                                ),
                            ),
                            CONF_ENABLE_DEVICE_TOOLS: existing_entry.data.get(
                                CONF_ENABLE_DEVICE_TOOLS,
                                DEFAULT_ENABLE_DEVICE_TOOLS,
                            ),
                            CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: existing_entry.data.get(
                                CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
                                DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT,
                            ),
                        }
                        for spec in built_in_specs:
                            shared_settings[spec.shared_setting_key] = (
                                existing_entry.data.get(
                                    spec.shared_setting_key,
                                    get_builtin_shared_setting_value(
                                        spec,
                                        lambda key, default=None: existing_entry.data.get(
                                            key, default
                                        ),
                                    ),
                                )
                            )

                    # Combine data from steps 1-4 + shared settings
                    combined_data = {
                        **self.step1_data,
                        **self.step2_data,
                        **self.step3_data,
                        **user_input,  # Step 4 data
                        **shared_settings,  # Copy from existing entry
                    }

                    # Create config entry (same as before)
                    profile_name = combined_data[CONF_PROFILE_NAME]
                    server_type = combined_data.get(
                        CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE
                    )

                    server_display_map = {
                        SERVER_TYPE_LMSTUDIO: "LM Studio",
                        SERVER_TYPE_LLAMACPP: "llama.cpp",
                        SERVER_TYPE_OLLAMA: "Ollama",
                        SERVER_TYPE_OPENAI: "OpenAI",
                        SERVER_TYPE_GEMINI: "Gemini",
                        SERVER_TYPE_ANTHROPIC: "Claude",
                        SERVER_TYPE_OPENROUTER: "OpenRouter",
                        SERVER_TYPE_OPENCLAW: "OpenClaw",
                        SERVER_TYPE_VLLM: "vLLM",
                    }
                    server_display = server_display_map.get(server_type, "LM Studio")

                    unique_id = f"{DOMAIN}_{server_type}_{profile_name.lower().replace(' ', '_')}"
                    await self.async_set_unique_id(unique_id)
                    self._abort_if_unique_id_configured()

                    return self.async_create_entry(
                        title=f"{server_display} - {profile_name}",
                        data=combined_data,
                    )

        # Gemini requires temperature=1.0 for optimal performance (Google's guidance)
        default_temp = 1.0 if server_type == SERVER_TYPE_GEMINI else DEFAULT_TEMPERATURE

        # Build schema based on server type
        if server_type == SERVER_TYPE_OPENCLAW:
            advanced_schema_dict = {
                CONVERSATION_SECTION_KEY: _build_conversation_section(
                    {
                        vol.Required(CONF_CONTROL_HA, default=DEFAULT_CONTROL_HA): bool,
                        vol.Required(
                            CONF_RESPONSE_MODE, default=DEFAULT_RESPONSE_MODE
                        ): SelectSelector(
                            SelectSelectorConfig(
                                options=[
                                    {"value": "none", "label": "None"},
                                    {"value": "default", "label": "Smart"},
                                    {"value": "always", "label": "Always"},
                                ],
                                mode=SelectSelectorMode.DROPDOWN,
                            )
                        ),
                        vol.Optional(
                            CONF_FOLLOW_UP_PHRASES,
                            default=get_follow_up_phrases(self.hass.config.language),
                        ): TextSelector(TextSelectorConfig(multiline=True)),
                        vol.Optional(
                            CONF_END_WORDS,
                            default=get_end_words(self.hass.config.language),
                        ): TextSelector(TextSelectorConfig(multiline=True)),
                        vol.Optional(
                            CONF_CLEAN_RESPONSES, default=DEFAULT_CLEAN_RESPONSES
                        ): bool,
                    }
                ),
                PERFORMANCE_SECTION_KEY: _build_performance_section(
                    {
                        vol.Required(CONF_TIMEOUT, default=60): vol.All(
                            vol.Coerce(int), vol.Range(min=5, max=300)
                        ),
                        vol.Required(
                            CONF_DEBUG_MODE, default=DEFAULT_DEBUG_MODE
                        ): bool,
                    }
                ),
                PROVIDER_SECTION_KEY: _build_provider_section(
                    {
                        vol.Optional(
                            CONF_OPENCLAW_SESSION_KEY,
                            default=DEFAULT_OPENCLAW_SESSION_KEY,
                        ): str,
                    }
                ),
            }
        else:
            performance_schema_items: dict[Any, Any] = {
                vol.Required(CONF_TEMPERATURE, default=default_temp): vol.All(
                    vol.Coerce(float), vol.Range(min=0.0, max=1.0)
                ),
                vol.Required(CONF_MAX_TOKENS, default=DEFAULT_MAX_TOKENS): vol.Coerce(
                    int
                ),
                vol.Required(CONF_MAX_HISTORY, default=DEFAULT_MAX_HISTORY): vol.Coerce(
                    int
                ),
                vol.Required(
                    CONF_MAX_ITERATIONS, default=DEFAULT_MAX_ITERATIONS
                ): vol.Coerce(int),
                vol.Required(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): vol.All(
                    vol.Coerce(int), vol.Range(min=5, max=300)
                ),
                vol.Required(CONF_DEBUG_MODE, default=DEFAULT_DEBUG_MODE): bool,
            }
            conversation_schema_items: dict[Any, Any] = {
                vol.Required(CONF_CONTROL_HA, default=DEFAULT_CONTROL_HA): bool,
                vol.Required(
                    CONF_RESPONSE_MODE, default=DEFAULT_RESPONSE_MODE
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            {"value": "none", "label": "None"},
                            {"value": "default", "label": "Smart"},
                            {"value": "always", "label": "Always"},
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_FOLLOW_UP_PHRASES,
                    default=get_follow_up_phrases(self.hass.config.language),
                ): TextSelector(TextSelectorConfig(multiline=True)),
                vol.Optional(
                    CONF_END_WORDS, default=get_end_words(self.hass.config.language)
                ): TextSelector(TextSelectorConfig(multiline=True)),
                vol.Optional(CONF_CLEAN_RESPONSES, default=DEFAULT_CLEAN_RESPONSES): bool,
            }
            advanced_schema_dict = {
                CONVERSATION_SECTION_KEY: _build_conversation_section(
                    conversation_schema_items
                ),
                PERFORMANCE_SECTION_KEY: _build_performance_section(
                    performance_schema_items
                ),
            }

            # Add Ollama-specific fields in correct position (after Max Tokens)
            if server_type == SERVER_TYPE_OLLAMA:
                advanced_schema_dict[PROVIDER_SECTION_KEY] = _build_provider_section(
                    {
                        vol.Optional(
                            CONF_OLLAMA_NUM_CTX, default=DEFAULT_OLLAMA_NUM_CTX
                        ): vol.Coerce(int),
                        vol.Optional(
                            CONF_OLLAMA_KEEP_ALIVE,
                            default=DEFAULT_OLLAMA_KEEP_ALIVE,
                        ): str,
                    }
                )

        advanced_schema_dict[TOOLS_SECTION_KEY] = _build_profile_tools_section(
            getattr(self, "step4_data", {}),
            built_in_specs,
        )

        advanced_schema = vol.Schema(advanced_schema_dict)

        # Set description based on server type
        if server_type == SERVER_TYPE_OPENCLAW:
            description_placeholders = {
                "advanced_info": (
                    "OpenClaw manages model selection, token limits, history, and "
                    "tool execution on the gateway. Only conversation, timeout, "
                    "session, and profile-level tool settings are shown here. The "
                    "Tools section still lets this profile disable specific shared "
                    "MCP tool families."
                )
            }
        else:
            description_placeholders = {
                "advanced_info": (
                    "These settings are organized by conversation behavior, performance, "
                    "provider-specific options, and tools. The Tools section only affects "
                    "this profile and can disable specific shared MCP tool families for "
                    "smaller models."
                )
            }

        return self.async_show_form(
            step_id="advanced",
            data_schema=advanced_schema,
            errors=errors,
            description_placeholders=description_placeholders,
        )

    async def async_step_mcp_server(self, user_input=None) -> FlowResult:
        """Handle step 5 - shared MCP server settings (first profile only)."""
        errors: dict[str, str] = {}
        current_values = user_input or {}
        built_in_specs = await _async_load_builtin_tool_toggle_specs(self.hass)

        if user_input is not None:
            user_input = _flatten_section_values(
                user_input,
                CONTEXT_SECTION_KEY,
                DISCOVERY_SECTION_KEY,
                MEMORY_SECTION_KEY,
                TOOLS_SECTION_KEY,
            )
            user_input = _normalize_shared_tool_inputs(user_input, built_in_specs)
            current_values = user_input

            # Validate MCP port
            mcp_port = user_input.get(CONF_MCP_PORT, DEFAULT_MCP_PORT)
            if not 1024 <= mcp_port <= 65535:
                errors[CONF_MCP_PORT] = "invalid_port"

            # Validate allowed IPs
            allowed_ips_str = user_input.get(CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS)
            is_valid, error_msg = validate_allowed_ips(allowed_ips_str)
            if not is_valid:
                errors[CONF_ALLOWED_IPS] = "invalid_ip"
                _LOGGER.warning("Invalid allowed IPs: %s", error_msg)

            if not errors:
                # Create/update system entry with shared settings
                from . import get_system_entry

                system_entry = get_system_entry(self.hass)

                if not system_entry:
                    # Create system entry with shared settings
                    await self.hass.config_entries.flow.async_init(
                        DOMAIN, context={"source": "system"}, data=user_input
                    )
                    _LOGGER.info(
                        "Created system entry with shared MCP settings from initial setup"
                    )
                else:
                    # Update existing system entry
                    self.hass.config_entries.async_update_entry(
                        system_entry, data={**system_entry.data, **user_input}
                    )
                    _LOGGER.info(
                        "Updated existing system entry with shared MCP settings"
                    )

                # Combine data from steps 1-4 (profile settings only, no shared settings)
                combined_data = {
                    **self.step1_data,
                    **self.step2_data,
                    **self.step3_data,
                    **self.step4_data,
                }

                # Create profile config entry
                profile_name = combined_data[CONF_PROFILE_NAME]
                server_type = combined_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)

                server_display_map = {
                    SERVER_TYPE_LMSTUDIO: "LM Studio",
                    SERVER_TYPE_LLAMACPP: "llama.cpp",
                    SERVER_TYPE_OLLAMA: "Ollama",
                    SERVER_TYPE_OPENAI: "OpenAI",
                    SERVER_TYPE_GEMINI: "Gemini",
                    SERVER_TYPE_ANTHROPIC: "Claude",
                    SERVER_TYPE_OPENROUTER: "OpenRouter",
                    SERVER_TYPE_OPENCLAW: "OpenClaw",
                    SERVER_TYPE_VLLM: "vLLM",
                }
                server_display = server_display_map.get(server_type, "LM Studio")

                unique_id = (
                    f"{DOMAIN}_{server_type}_{profile_name.lower().replace(' ', '_')}"
                )
                await self.async_set_unique_id(unique_id)
                self._abort_if_unique_id_configured()

                return self.async_create_entry(
                    title=f"{server_display} - {profile_name}",
                    data=combined_data,
                )

        shared_defaults = {
            CONF_SEARCH_PROVIDER: _get_form_value(
                current_values,
                CONF_SEARCH_PROVIDER,
                DEFAULT_SEARCH_PROVIDER,
            ),
            CONF_ENABLE_WEB_SEARCH: _get_form_value(
                current_values,
                CONF_ENABLE_WEB_SEARCH,
                _infer_web_search_enabled(
                    current_values.get(CONF_ENABLE_WEB_SEARCH),
                    current_values.get(CONF_SEARCH_PROVIDER),
                ),
            ),
            CONF_BRAVE_API_KEY: _get_form_value(
                current_values,
                CONF_BRAVE_API_KEY,
                DEFAULT_BRAVE_API_KEY,
            ),
            CONF_INCLUDE_CURRENT_USER: _get_form_value(
                current_values,
                CONF_INCLUDE_CURRENT_USER,
                DEFAULT_INCLUDE_CURRENT_USER,
            ),
            CONF_INCLUDE_HOME_LOCATION: _get_form_value(
                current_values,
                CONF_INCLUDE_HOME_LOCATION,
                DEFAULT_INCLUDE_HOME_LOCATION,
            ),
            CONF_ENABLE_GAP_FILLING: _get_form_value(
                current_values,
                CONF_ENABLE_GAP_FILLING,
                DEFAULT_ENABLE_GAP_FILLING,
            ),
            CONF_MAX_ENTITIES_PER_DISCOVERY: _get_form_value(
                current_values,
                CONF_MAX_ENTITIES_PER_DISCOVERY,
                DEFAULT_MAX_ENTITIES_PER_DISCOVERY,
            ),
            CONF_ENABLE_ASSIST_BRIDGE: _get_form_value(
                current_values,
                CONF_ENABLE_ASSIST_BRIDGE,
                DEFAULT_ENABLE_ASSIST_BRIDGE,
            ),
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
                DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
            ),
            CONF_ENABLE_WEATHER_FORECAST_TOOL: _get_form_value(
                current_values,
                CONF_ENABLE_WEATHER_FORECAST_TOOL,
                DEFAULT_ENABLE_WEATHER_FORECAST_TOOL,
            ),
            CONF_ENABLE_RECORDER_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_RECORDER_TOOLS,
                DEFAULT_ENABLE_RECORDER_TOOLS,
            ),
            CONF_ENABLE_MEMORY_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_MEMORY_TOOLS,
                DEFAULT_ENABLE_MEMORY_TOOLS,
            ),
            CONF_ENABLE_DEVICE_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_DEVICE_TOOLS,
                DEFAULT_ENABLE_DEVICE_TOOLS,
            ),
            CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: _get_form_value(
                current_values,
                CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
                DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT,
            ),
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                DEFAULT_ENABLE_EXTERNAL_CUSTOM_TOOLS,
            ),
            CONF_MEMORY_DEFAULT_TTL_DAYS: _get_form_value(
                current_values,
                CONF_MEMORY_DEFAULT_TTL_DAYS,
                DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
            ),
            CONF_MEMORY_MAX_TTL_DAYS: _get_form_value(
                current_values,
                CONF_MEMORY_MAX_TTL_DAYS,
                DEFAULT_MEMORY_MAX_TTL_DAYS,
            ),
            CONF_MEMORY_MAX_ITEMS: _get_form_value(
                current_values,
                CONF_MEMORY_MAX_ITEMS,
                DEFAULT_MEMORY_MAX_ITEMS,
            ),
        }
        for spec in built_in_specs:
            shared_defaults[spec.shared_setting_key] = _get_form_value(
                current_values,
                spec.shared_setting_key,
                get_builtin_shared_setting_value(
                    spec,
                    lambda key, default=None: current_values.get(key, default),
                ),
            )

        # Build schema for MCP server settings
        mcp_schema = vol.Schema(
            {
                vol.Required(
                    CONF_MCP_PORT,
                    default=_get_form_value(
                        current_values, CONF_MCP_PORT, DEFAULT_MCP_PORT
                    ),
                ): vol.Coerce(int),
                vol.Optional(
                    CONF_ALLOWED_IPS,
                    default=_get_form_value(
                        current_values, CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS
                    ),
                ): str,
                CONTEXT_SECTION_KEY: _build_shared_context_section(shared_defaults),
                DISCOVERY_SECTION_KEY: _build_shared_discovery_section(shared_defaults),
                MEMORY_SECTION_KEY: _build_shared_memory_section(shared_defaults),
                TOOLS_SECTION_KEY: _build_shared_tools_section(
                    shared_defaults,
                    built_in_specs,
                ),
            }
        )

        return self.async_show_form(
            step_id="mcp_server",
            data_schema=mcp_schema,
            errors=errors,
            description_placeholders={
                "info": (
                    "⚠️ These settings define the shared MCP server capabilities "
                    "available to all profiles and external MCP clients. Individual "
                    "profiles can still disable specific tool families later."
                )
            },
        )

    async def async_step_system(self, data: dict[str, Any]) -> FlowResult:
        """Handle programmatic creation of system entry (no UI)."""
        # Set unique ID for system entry
        await self.async_set_unique_id(SYSTEM_ENTRY_UNIQUE_ID)
        self._abort_if_unique_id_configured()

        # Create system entry with provided data
        return self.async_create_entry(
            title="Shared MCP Server Settings",
            data=data,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get options flow for this handler."""
        return MCPAssistOptionsFlow()


class MCPAssistOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for MCP Assist integration."""

    def __init__(self) -> None:
        """Initialize options flow."""
        super().__init__()
        self.profile_options: dict[str, Any] = {}

    def _get_search_provider_default(self, options: dict, data: dict) -> str:
        """Get default search provider with backward compatibility."""
        # Check if search_provider is already set
        provider = options.get(CONF_SEARCH_PROVIDER, data.get(CONF_SEARCH_PROVIDER))
        if provider:
            return provider

        # Backward compat: if old enable_custom_tools was True, default to "brave"
        if options.get(
            CONF_ENABLE_CUSTOM_TOOLS, data.get(CONF_ENABLE_CUSTOM_TOOLS, False)
        ):
            return "brave"

        return DEFAULT_SEARCH_PROVIDER

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        # Check if this is the system entry - skip directly to MCP server settings
        if self.config_entry.unique_id == SYSTEM_ENTRY_UNIQUE_ID:
            return await self.async_step_mcp_server()

        errors: dict[str, str] = {}
        server_type = self.config_entry.data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)
        default_system_prompt = _get_default_system_prompt(self.hass)
        built_in_specs = await _async_load_builtin_tool_toggle_specs(self.hass)

        if user_input is not None:
            user_input = _flatten_section_values(
                user_input,
                PROFILE_SECTION_KEY,
                CONNECTION_SECTION_KEY,
                MODEL_SECTION_KEY,
                PROMPTS_SECTION_KEY,
                CONVERSATION_SECTION_KEY,
                PROVIDER_SECTION_KEY,
                ADVANCED_SECTION_KEY,
                TOOLS_SECTION_KEY,
            )
            user_input = _apply_profile_tool_disables(user_input, built_in_specs)
            user_input = _normalize_prompt_inputs(
                user_input, server_type, default_system_prompt
            )

            if not errors:
                # Support both old and new config keys
                if (
                    CONF_FOLLOW_UP_MODE in user_input
                    and CONF_RESPONSE_MODE not in user_input
                ):
                    user_input[CONF_RESPONSE_MODE] = user_input[CONF_FOLLOW_UP_MODE]
                    del user_input[CONF_FOLLOW_UP_MODE]

                # For OpenClaw, ensure model name and empty system prompt are set
                server_type = self.config_entry.data.get(
                    CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE
                )
                if server_type == SERVER_TYPE_OPENCLAW:
                    if CONF_MODEL_NAME not in user_input:
                        user_input[CONF_MODEL_NAME] = "main"
                    user_input[CONF_SYSTEM_PROMPT] = ""
                    user_input[CONF_TECHNICAL_PROMPT] = ""
                    user_input[CONF_SYSTEM_PROMPT_MODE] = PROMPT_MODE_DEFAULT
                    user_input[CONF_TECHNICAL_PROMPT_MODE] = PROMPT_MODE_DEFAULT

                # Store profile settings and proceed to MCP server settings
                self.profile_options = user_input
                return await self.async_step_mcp_server()

        # Get current values from options, then data, then defaults
        options = self.config_entry.options
        data = self.config_entry.data
        current_values = self.profile_options or {}

        # Handle backward compatibility
        response_mode_value = options.get(
            CONF_RESPONSE_MODE, options.get(CONF_FOLLOW_UP_MODE, DEFAULT_RESPONSE_MODE)
        )

        # Fetch models based on server type
        models = []
        current_model = current_values.get(
            CONF_MODEL_NAME,
            options.get(CONF_MODEL_NAME, data.get(CONF_MODEL_NAME, DEFAULT_MODEL_NAME)),
        )

        # OpenClaw doesn't have /v1/models - skip model fetching
        if server_type == SERVER_TYPE_OPENCLAW:
            # Don't fetch models, don't show model field
            pass
        elif server_type in [
            SERVER_TYPE_LMSTUDIO,
            SERVER_TYPE_LLAMACPP,
            SERVER_TYPE_OLLAMA,
            SERVER_TYPE_VLLM,
        ]:
            # Local servers - fetch from URL
            server_url = _get_form_value(
                current_values,
                CONF_LMSTUDIO_URL,
                options.get(
                    CONF_LMSTUDIO_URL, data.get(CONF_LMSTUDIO_URL, DEFAULT_LMSTUDIO_URL)
                ),
            ).rstrip("/")
            _LOGGER.info(
                f"🔍 OPTIONS: Attempting to fetch models from {server_type} at {server_url}"
            )
            try:
                models = await fetch_models_from_lmstudio(self.hass, server_url)
                _LOGGER.info(f"✅ OPTIONS: Successfully fetched {len(models)} models")
            except Exception as err:
                _LOGGER.error(f"❌ OPTIONS: Failed to fetch models: {err}")
        elif server_type == SERVER_TYPE_OPENAI:
            # OpenAI - fetch from API
            api_key = _get_form_value(
                current_values,
                CONF_API_KEY,
                options.get(CONF_API_KEY, data.get(CONF_API_KEY, "")),
            )
            if api_key:
                _LOGGER.info("🔍 OPTIONS: Attempting to fetch models from OpenAI")
                try:
                    models = await fetch_models_from_openai(self.hass, api_key)
                    _LOGGER.info(
                        f"✅ OPTIONS: Successfully fetched {len(models)} OpenAI models"
                    )
                except Exception as err:
                    _LOGGER.error(f"❌ OPTIONS: Failed to fetch OpenAI models: {err}")
        elif server_type == SERVER_TYPE_GEMINI:
            # Gemini - fetch from API
            api_key = _get_form_value(
                current_values,
                CONF_API_KEY,
                options.get(CONF_API_KEY, data.get(CONF_API_KEY, "")),
            )
            if api_key:
                _LOGGER.info("🔍 OPTIONS: Attempting to fetch models from Gemini")
                try:
                    models = await fetch_models_from_gemini(self.hass, api_key)
                    _LOGGER.info(
                        f"✅ OPTIONS: Successfully fetched {len(models)} Gemini models"
                    )
                except Exception as err:
                    _LOGGER.error(f"❌ OPTIONS: Failed to fetch Gemini models: {err}")
        elif server_type == SERVER_TYPE_OPENROUTER:
            # OpenRouter - fetch from API
            api_key = _get_form_value(
                current_values,
                CONF_API_KEY,
                options.get(CONF_API_KEY, data.get(CONF_API_KEY, "")),
            )
            if api_key:
                _LOGGER.info("🔍 OPTIONS: Attempting to fetch models from OpenRouter")
                try:
                    models = await fetch_models_from_openrouter(self.hass, api_key)
                    _LOGGER.info(
                        f"✅ OPTIONS: Successfully fetched {len(models)} OpenRouter models"
                    )
                except Exception as err:
                    _LOGGER.error(
                        f"❌ OPTIONS: Failed to fetch OpenRouter models: {err}"
                    )

        # Build model selector based on whether models were fetched
        if models:
            # Show dropdown with available models (custom_value allows free text input)
            model_selector = SelectSelector(
                SelectSelectorConfig(
                    options=models,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            )
        else:
            # Show text input as fallback
            model_selector = str

        schema_dict: dict[Any, Any] = {
            PROFILE_SECTION_KEY: _build_profile_identity_section(
                _get_form_value(
                    current_values,
                    CONF_PROFILE_NAME,
                    options.get(
                        CONF_PROFILE_NAME,
                        data.get(CONF_PROFILE_NAME, "Default"),
                    ),
                )
            ),
        }

        system_prompt_suggestion = _get_prompt_text_default(
            current_values,
            prompt_key=CONF_SYSTEM_PROMPT,
            stored_mode=options.get(
                CONF_SYSTEM_PROMPT_MODE, data.get(CONF_SYSTEM_PROMPT_MODE)
            ),
            stored_prompt=options.get(CONF_SYSTEM_PROMPT, data.get(CONF_SYSTEM_PROMPT)),
            default_prompt=default_system_prompt,
        )
        technical_prompt_suggestion = _get_prompt_text_default(
            current_values,
            prompt_key=CONF_TECHNICAL_PROMPT,
            stored_mode=options.get(
                CONF_TECHNICAL_PROMPT_MODE,
                data.get(CONF_TECHNICAL_PROMPT_MODE),
            ),
            stored_prompt=options.get(
                CONF_TECHNICAL_PROMPT, data.get(CONF_TECHNICAL_PROMPT)
            ),
            default_prompt=DEFAULT_TECHNICAL_PROMPT,
        )

        connection_schema_items: dict[Any, Any] = {}
        if server_type == SERVER_TYPE_OPENCLAW:
            connection_schema_items[
                vol.Required(
                    CONF_OPENCLAW_HOST,
                    default=_get_form_value(
                        current_values,
                        CONF_OPENCLAW_HOST,
                        options.get(
                            CONF_OPENCLAW_HOST,
                            data.get(CONF_OPENCLAW_HOST, DEFAULT_OPENCLAW_HOST),
                        ),
                    ),
                )
            ] = str
            connection_schema_items[
                vol.Required(
                    CONF_OPENCLAW_PORT,
                    default=_get_form_value(
                        current_values,
                        CONF_OPENCLAW_PORT,
                        options.get(
                            CONF_OPENCLAW_PORT,
                            data.get(CONF_OPENCLAW_PORT, DEFAULT_OPENCLAW_PORT),
                        ),
                    ),
                )
            ] = vol.Coerce(int)
            connection_schema_items[
                vol.Required(
                    CONF_OPENCLAW_TOKEN,
                    default=_get_form_value(
                        current_values,
                        CONF_OPENCLAW_TOKEN,
                        options.get(
                            CONF_OPENCLAW_TOKEN,
                            data.get(CONF_OPENCLAW_TOKEN, ""),
                        ),
                    ),
                )
            ] = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))
            connection_schema_items[
                vol.Required(
                    CONF_OPENCLAW_USE_SSL,
                    default=_get_form_value(
                        current_values,
                        CONF_OPENCLAW_USE_SSL,
                        options.get(
                            CONF_OPENCLAW_USE_SSL,
                            data.get(
                                CONF_OPENCLAW_USE_SSL,
                                DEFAULT_OPENCLAW_USE_SSL,
                            ),
                        ),
                    ),
                )
            ] = BooleanSelector()
        elif server_type in [
            SERVER_TYPE_LMSTUDIO,
            SERVER_TYPE_LLAMACPP,
            SERVER_TYPE_OLLAMA,
            SERVER_TYPE_VLLM,
        ]:
            server_url = options.get(
                CONF_LMSTUDIO_URL, data.get(CONF_LMSTUDIO_URL, DEFAULT_LMSTUDIO_URL)
            )
            connection_schema_items[
                vol.Required(
                    CONF_LMSTUDIO_URL,
                    default=_get_form_value(
                        current_values, CONF_LMSTUDIO_URL, server_url
                    ),
                )
            ] = str
        elif server_type == SERVER_TYPE_OPENAI:
            # OpenAI - hybrid (URL + API key)
            server_url = options.get(
                CONF_LMSTUDIO_URL,
                data.get(CONF_LMSTUDIO_URL, OPENAI_BASE_URL)
            )
            connection_schema_items[
                vol.Required(
                    CONF_LMSTUDIO_URL,
                    default=_get_form_value(
                        current_values, CONF_LMSTUDIO_URL, server_url
                    ),
                )
            ] = str

            api_key = options.get(CONF_API_KEY, data.get(CONF_API_KEY, ""))
            connection_schema_items[
                vol.Required(
                    CONF_API_KEY,
                    default=_get_form_value(current_values, CONF_API_KEY, api_key),
                )
            ] = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))
        else:
            # Other cloud providers (Gemini, Anthropic, OpenRouter) - API key only
            api_key = options.get(CONF_API_KEY, data.get(CONF_API_KEY, ""))
            connection_schema_items[
                vol.Required(
                    CONF_API_KEY,
                    default=_get_form_value(current_values, CONF_API_KEY, api_key),
                )
            ] = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))

        schema_dict[CONNECTION_SECTION_KEY] = _build_connection_section(
            connection_schema_items
        )

        if server_type != SERVER_TYPE_OPENCLAW:
            schema_dict[MODEL_SECTION_KEY] = _build_model_section(
                current_model, model_selector
            )

        if server_type != SERVER_TYPE_OPENCLAW:
            schema_dict[PROMPTS_SECTION_KEY] = _build_prompt_section(
                include_system_prompt=True,
                system_prompt_value=system_prompt_suggestion,
                technical_prompt_value=technical_prompt_suggestion,
            )

        provider_schema_items: dict[Any, Any] = {}

        if server_type == SERVER_TYPE_OPENCLAW:
            schema_dict[CONVERSATION_SECTION_KEY] = _build_conversation_section(
                {
                    vol.Required(
                        CONF_CONTROL_HA,
                        default=_get_form_value(
                            current_values,
                            CONF_CONTROL_HA,
                            options.get(
                                CONF_CONTROL_HA,
                                data.get(CONF_CONTROL_HA, DEFAULT_CONTROL_HA),
                            ),
                        ),
                    ): bool,
                    vol.Required(
                        CONF_RESPONSE_MODE, default=response_mode_value
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                {"value": "none", "label": "None"},
                                {"value": "default", "label": "Smart"},
                                {"value": "always", "label": "Always"},
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_FOLLOW_UP_PHRASES,
                        default=options.get(
                            CONF_FOLLOW_UP_PHRASES,
                            data.get(CONF_FOLLOW_UP_PHRASES, DEFAULT_FOLLOW_UP_PHRASES),
                        ),
                    ): TextSelector(TextSelectorConfig(multiline=True)),
                    vol.Optional(
                        CONF_END_WORDS,
                        default=options.get(
                            CONF_END_WORDS, data.get(CONF_END_WORDS, DEFAULT_END_WORDS)
                        ),
                    ): TextSelector(TextSelectorConfig(multiline=True)),
                    vol.Optional(
                        CONF_CLEAN_RESPONSES,
                        default=_get_form_value(
                            current_values,
                            CONF_CLEAN_RESPONSES,
                            options.get(
                                CONF_CLEAN_RESPONSES,
                                data.get(
                                    CONF_CLEAN_RESPONSES,
                                    DEFAULT_CLEAN_RESPONSES,
                                ),
                            ),
                        ),
                    ): bool,
                }
            )
            provider_schema_items = {
                vol.Optional(
                    CONF_OPENCLAW_SESSION_KEY,
                    default=_get_form_value(
                        current_values,
                        CONF_OPENCLAW_SESSION_KEY,
                        options.get(
                            CONF_OPENCLAW_SESSION_KEY,
                            data.get(
                                CONF_OPENCLAW_SESSION_KEY,
                                DEFAULT_OPENCLAW_SESSION_KEY,
                            ),
                        ),
                    ),
                ): str,
            }
            advanced_schema_items: dict[Any, Any] = {
                vol.Required(
                    CONF_TIMEOUT,
                    default=_get_form_value(
                        current_values,
                        CONF_TIMEOUT,
                        options.get(
                            CONF_TIMEOUT,
                            data.get(CONF_TIMEOUT, 60),
                        ),
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=5, max=300)),
                vol.Required(
                    CONF_DEBUG_MODE,
                    default=_get_form_value(
                        current_values,
                        CONF_DEBUG_MODE,
                        options.get(
                            CONF_DEBUG_MODE,
                            data.get(CONF_DEBUG_MODE, DEFAULT_DEBUG_MODE),
                        ),
                    ),
                ): bool,
            }
        else:
            schema_dict[CONVERSATION_SECTION_KEY] = _build_conversation_section(
                {
                    vol.Required(
                        CONF_CONTROL_HA,
                        default=_get_form_value(
                            current_values,
                            CONF_CONTROL_HA,
                            options.get(
                                CONF_CONTROL_HA,
                                data.get(CONF_CONTROL_HA, DEFAULT_CONTROL_HA),
                            ),
                        ),
                    ): bool,
                    vol.Required(
                        CONF_RESPONSE_MODE,
                        default=_get_form_value(
                            current_values,
                            CONF_RESPONSE_MODE,
                            response_mode_value,
                        ),
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                {"value": "none", "label": "None"},
                                {"value": "default", "label": "Smart"},
                                {"value": "always", "label": "Always"},
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_FOLLOW_UP_PHRASES,
                        default=_get_form_value(
                            current_values,
                            CONF_FOLLOW_UP_PHRASES,
                            options.get(
                                CONF_FOLLOW_UP_PHRASES,
                                data.get(
                                    CONF_FOLLOW_UP_PHRASES,
                                    DEFAULT_FOLLOW_UP_PHRASES,
                                ),
                            ),
                        ),
                    ): TextSelector(TextSelectorConfig(multiline=True)),
                    vol.Optional(
                        CONF_END_WORDS,
                        default=_get_form_value(
                            current_values,
                            CONF_END_WORDS,
                            options.get(
                                CONF_END_WORDS,
                                data.get(CONF_END_WORDS, DEFAULT_END_WORDS),
                            ),
                        ),
                    ): TextSelector(TextSelectorConfig(multiline=True)),
                    vol.Required(
                        CONF_CLEAN_RESPONSES,
                        default=_get_form_value(
                            current_values,
                            CONF_CLEAN_RESPONSES,
                            options.get(
                                CONF_CLEAN_RESPONSES,
                                data.get(
                                    CONF_CLEAN_RESPONSES,
                                    DEFAULT_CLEAN_RESPONSES,
                                ),
                            ),
                        ),
                    ): bool,
                }
            )
            advanced_schema_items = {
                vol.Required(
                    CONF_TEMPERATURE,
                    default=_get_form_value(
                        current_values,
                        CONF_TEMPERATURE,
                        options.get(
                            CONF_TEMPERATURE,
                            data.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                        ),
                    ),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
                vol.Required(
                    CONF_MAX_TOKENS,
                    default=_get_form_value(
                        current_values,
                        CONF_MAX_TOKENS,
                        options.get(
                            CONF_MAX_TOKENS,
                            data.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                        ),
                    ),
                ): vol.Coerce(int),
                vol.Required(
                    CONF_MAX_HISTORY,
                    default=_get_form_value(
                        current_values,
                        CONF_MAX_HISTORY,
                        options.get(
                            CONF_MAX_HISTORY,
                            data.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY),
                        ),
                    ),
                ): vol.Coerce(int),
                vol.Required(
                    CONF_MAX_ITERATIONS,
                    default=_get_form_value(
                        current_values,
                        CONF_MAX_ITERATIONS,
                        options.get(
                            CONF_MAX_ITERATIONS,
                            data.get(
                                CONF_MAX_ITERATIONS,
                                DEFAULT_MAX_ITERATIONS,
                            ),
                        ),
                    ),
                ): vol.Coerce(int),
                vol.Required(
                    CONF_TIMEOUT,
                    default=_get_form_value(
                        current_values,
                        CONF_TIMEOUT,
                        options.get(
                            CONF_TIMEOUT,
                            data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                        ),
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=5, max=300)),
                vol.Required(
                    CONF_DEBUG_MODE,
                    default=_get_form_value(
                        current_values,
                        CONF_DEBUG_MODE,
                        options.get(
                            CONF_DEBUG_MODE,
                            data.get(CONF_DEBUG_MODE, DEFAULT_DEBUG_MODE),
                        ),
                    ),
                ): bool,
            }
            if server_type == SERVER_TYPE_OLLAMA:
                provider_schema_items = {
                    vol.Optional(
                        CONF_OLLAMA_NUM_CTX,
                        default=_get_form_value(
                            current_values,
                            CONF_OLLAMA_NUM_CTX,
                            options.get(
                                CONF_OLLAMA_NUM_CTX,
                                data.get(
                                    CONF_OLLAMA_NUM_CTX,
                                    DEFAULT_OLLAMA_NUM_CTX,
                                ),
                            ),
                        ),
                    ): vol.Coerce(int),
                    vol.Optional(
                        CONF_OLLAMA_KEEP_ALIVE,
                        default=_get_form_value(
                            current_values,
                            CONF_OLLAMA_KEEP_ALIVE,
                            options.get(
                                CONF_OLLAMA_KEEP_ALIVE,
                                data.get(
                                    CONF_OLLAMA_KEEP_ALIVE,
                                    DEFAULT_OLLAMA_KEEP_ALIVE,
                                ),
                            ),
                        ),
                    ): str,
                }

        if provider_schema_items:
            schema_dict[PROVIDER_SECTION_KEY] = _build_provider_section(
                provider_schema_items
            )
        schema_dict[TOOLS_SECTION_KEY] = _build_profile_tools_section(
            current_values,
            built_in_specs,
            options,
            data,
        )
        schema_dict[ADVANCED_SECTION_KEY] = _build_advanced_section(
            advanced_schema_items
        )

        options_schema = vol.Schema(schema_dict)

        # Set description based on server type
        if server_type == SERVER_TYPE_OPENCLAW:
            description_placeholders = {
                "server_info": (
                    "OpenClaw manages the model and system prompt on the gateway. "
                    "Use the connection, conversation, provider, and advanced "
                    "sections here to control how this Home Assistant profile "
                    "connects and follows up. The Tools section can still disable "
                    "specific shared MCP tool families for this profile."
                )
            }
        else:
            description_placeholders = {
                "server_info": (
                    "Configure this conversation profile. These settings only affect "
                    "this profile. The prompt fields are prefilled with the current "
                    "effective prompts so you can review, copy, or edit them directly. "
                    "If you leave a prompt unchanged, the integration keeps using the "
                    "built-in version from code. "
                    "The Tools section can disable specific shared MCP tool families "
                    "for smaller models."
                )
            }

        return self.async_show_form(
            step_id="init",
            data_schema=options_schema,
            errors=errors,
            description_placeholders=description_placeholders,
        )

    async def async_step_mcp_server(self, user_input=None):
        """Configure shared MCP server settings (affects all profiles)."""
        errors: dict[str, str] = {}
        current_values = user_input or {}
        built_in_specs = await _async_load_builtin_tool_toggle_specs(self.hass)

        if user_input is not None:
            user_input = _flatten_section_values(
                user_input,
                CONTEXT_SECTION_KEY,
                DISCOVERY_SECTION_KEY,
                MEMORY_SECTION_KEY,
                TOOLS_SECTION_KEY,
            )
            user_input = _normalize_shared_tool_inputs(user_input, built_in_specs)
            current_values = user_input

            # Validate allowed IPs
            allowed_ips_str = user_input.get(CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS)
            is_valid, error_msg = validate_allowed_ips(allowed_ips_str)
            if not is_valid:
                errors[CONF_ALLOWED_IPS] = "invalid_ip"
                _LOGGER.warning("Invalid allowed IPs in options: %s", error_msg)

            if not errors:
                # Import get_system_entry
                from . import get_system_entry

                # Update system entry with shared MCP settings
                system_entry = get_system_entry(self.hass)
                if system_entry:
                    self.hass.config_entries.async_update_entry(
                        system_entry, data={**system_entry.data, **user_input}
                    )
                    _LOGGER.info("Updated system entry with shared MCP settings")
                else:
                    _LOGGER.error("System entry not found when saving shared settings")

                # Update profile entry with per-profile settings only
                # Update entry title if profile name changed
                new_profile_name = self.profile_options.get(CONF_PROFILE_NAME)
                old_profile_name = self.config_entry.options.get(
                    CONF_PROFILE_NAME, self.config_entry.data.get(CONF_PROFILE_NAME)
                )
                if new_profile_name and new_profile_name != old_profile_name:
                    server_type = self.config_entry.data.get(
                        CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE
                    )
                    server_display_map = {
                        SERVER_TYPE_LMSTUDIO: "LM Studio",
                        SERVER_TYPE_LLAMACPP: "llama.cpp",
                        SERVER_TYPE_OLLAMA: "Ollama",
                        SERVER_TYPE_OPENAI: "OpenAI",
                        SERVER_TYPE_GEMINI: "Gemini",
                        SERVER_TYPE_ANTHROPIC: "Claude",
                        SERVER_TYPE_OPENROUTER: "OpenRouter",
                        SERVER_TYPE_OPENCLAW: "OpenClaw",
                        SERVER_TYPE_VLLM: "vLLM",
                    }
                    server_display = server_display_map.get(server_type, "LM Studio")
                    self.hass.config_entries.async_update_entry(
                        self.config_entry,
                        title=f"{server_display} - {new_profile_name}",
                    )

                # Save profile settings only (not shared settings)
                return self.async_create_entry(title="", data=self.profile_options)

        # Get current values from system entry
        from . import get_system_entry

        system_entry = get_system_entry(self.hass)

        # Get shared settings from system entry (with fallback to profile for backward compat)
        if system_entry:
            sys_options = system_entry.options
            sys_data = system_entry.data
        else:
            # Fallback to profile entry for backward compatibility
            sys_options = self.config_entry.options
            sys_data = self.config_entry.data

        shared_defaults = {
            CONF_SEARCH_PROVIDER: _get_form_value(
                current_values,
                CONF_SEARCH_PROVIDER,
                self._get_search_provider_default(sys_options, sys_data),
            ),
            CONF_ENABLE_WEB_SEARCH: _get_form_value(
                current_values,
                CONF_ENABLE_WEB_SEARCH,
                _infer_web_search_enabled(
                    sys_options.get(
                        CONF_ENABLE_WEB_SEARCH,
                        sys_data.get(CONF_ENABLE_WEB_SEARCH),
                    ),
                    self._get_search_provider_default(sys_options, sys_data),
                    sys_options.get(
                        CONF_ENABLE_CUSTOM_TOOLS,
                        sys_data.get(CONF_ENABLE_CUSTOM_TOOLS, False),
                    ),
                ),
            ),
            CONF_BRAVE_API_KEY: _get_form_value(
                current_values,
                CONF_BRAVE_API_KEY,
                sys_options.get(
                    CONF_BRAVE_API_KEY,
                    sys_data.get(CONF_BRAVE_API_KEY, DEFAULT_BRAVE_API_KEY),
                ),
            ),
            CONF_INCLUDE_CURRENT_USER: _get_form_value(
                current_values,
                CONF_INCLUDE_CURRENT_USER,
                sys_options.get(
                    CONF_INCLUDE_CURRENT_USER,
                    sys_data.get(
                        CONF_INCLUDE_CURRENT_USER,
                        DEFAULT_INCLUDE_CURRENT_USER,
                    ),
                ),
            ),
            CONF_INCLUDE_HOME_LOCATION: _get_form_value(
                current_values,
                CONF_INCLUDE_HOME_LOCATION,
                sys_options.get(
                    CONF_INCLUDE_HOME_LOCATION,
                    sys_data.get(
                        CONF_INCLUDE_HOME_LOCATION,
                        DEFAULT_INCLUDE_HOME_LOCATION,
                    ),
                ),
            ),
            CONF_ENABLE_GAP_FILLING: _get_form_value(
                current_values,
                CONF_ENABLE_GAP_FILLING,
                sys_options.get(
                    CONF_ENABLE_GAP_FILLING,
                    sys_data.get(CONF_ENABLE_GAP_FILLING, DEFAULT_ENABLE_GAP_FILLING),
                ),
            ),
            CONF_ENABLE_ASSIST_BRIDGE: _get_form_value(
                current_values,
                CONF_ENABLE_ASSIST_BRIDGE,
                sys_options.get(
                    CONF_ENABLE_ASSIST_BRIDGE,
                    sys_data.get(
                        CONF_ENABLE_ASSIST_BRIDGE, DEFAULT_ENABLE_ASSIST_BRIDGE
                    ),
                ),
            ),
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
                sys_options.get(
                    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
                    sys_data.get(
                        CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
                        DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
                    ),
                ),
            ),
            CONF_ENABLE_WEATHER_FORECAST_TOOL: _get_form_value(
                current_values,
                CONF_ENABLE_WEATHER_FORECAST_TOOL,
                sys_options.get(
                    CONF_ENABLE_WEATHER_FORECAST_TOOL,
                    sys_data.get(
                        CONF_ENABLE_WEATHER_FORECAST_TOOL,
                        DEFAULT_ENABLE_WEATHER_FORECAST_TOOL,
                    ),
                ),
            ),
            CONF_ENABLE_RECORDER_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_RECORDER_TOOLS,
                sys_options.get(
                    CONF_ENABLE_RECORDER_TOOLS,
                    sys_data.get(
                        CONF_ENABLE_RECORDER_TOOLS,
                        DEFAULT_ENABLE_RECORDER_TOOLS,
                    ),
                ),
            ),
            CONF_ENABLE_MEMORY_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_MEMORY_TOOLS,
                sys_options.get(
                    CONF_ENABLE_MEMORY_TOOLS,
                    sys_data.get(
                        CONF_ENABLE_MEMORY_TOOLS,
                        DEFAULT_ENABLE_MEMORY_TOOLS,
                    ),
                ),
            ),
            CONF_ENABLE_DEVICE_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_DEVICE_TOOLS,
                sys_options.get(
                    CONF_ENABLE_DEVICE_TOOLS,
                    sys_data.get(
                        CONF_ENABLE_DEVICE_TOOLS, DEFAULT_ENABLE_DEVICE_TOOLS
                    ),
                ),
            ),
            CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: _get_form_value(
                current_values,
                CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
                sys_options.get(
                    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
                    sys_data.get(
                        CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
                        DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT,
                    ),
                ),
            ),
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                sys_options.get(
                    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                    sys_data.get(
                        CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                        DEFAULT_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                    ),
                ),
            ),
            CONF_MEMORY_DEFAULT_TTL_DAYS: _get_form_value(
                current_values,
                CONF_MEMORY_DEFAULT_TTL_DAYS,
                sys_options.get(
                    CONF_MEMORY_DEFAULT_TTL_DAYS,
                    sys_data.get(
                        CONF_MEMORY_DEFAULT_TTL_DAYS,
                        DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
                    ),
                ),
            ),
            CONF_MEMORY_MAX_TTL_DAYS: _get_form_value(
                current_values,
                CONF_MEMORY_MAX_TTL_DAYS,
                sys_options.get(
                    CONF_MEMORY_MAX_TTL_DAYS,
                    sys_data.get(
                        CONF_MEMORY_MAX_TTL_DAYS,
                        DEFAULT_MEMORY_MAX_TTL_DAYS,
                    ),
                ),
            ),
            CONF_MEMORY_MAX_ITEMS: _get_form_value(
                current_values,
                CONF_MEMORY_MAX_ITEMS,
                sys_options.get(
                    CONF_MEMORY_MAX_ITEMS,
                    sys_data.get(
                        CONF_MEMORY_MAX_ITEMS,
                        DEFAULT_MEMORY_MAX_ITEMS,
                    ),
                ),
            ),
            CONF_MAX_ENTITIES_PER_DISCOVERY: _get_form_value(
                current_values,
                CONF_MAX_ENTITIES_PER_DISCOVERY,
                sys_options.get(
                    CONF_MAX_ENTITIES_PER_DISCOVERY,
                    sys_data.get(
                        CONF_MAX_ENTITIES_PER_DISCOVERY,
                        DEFAULT_MAX_ENTITIES_PER_DISCOVERY,
                    ),
                ),
            ),
        }
        for spec in built_in_specs:
            shared_defaults[spec.shared_setting_key] = _get_form_value(
                current_values,
                spec.shared_setting_key,
                get_builtin_shared_setting_value(
                    spec,
                    lambda key, default=None: sys_options.get(
                        key, sys_data.get(key, default)
                    ),
                ),
            )

        # Build schema for MCP server settings
        mcp_schema = vol.Schema(
            {
                vol.Required(
                    CONF_MCP_PORT,
                    default=_get_form_value(
                        current_values,
                        CONF_MCP_PORT,
                        sys_options.get(
                            CONF_MCP_PORT,
                            sys_data.get(CONF_MCP_PORT, DEFAULT_MCP_PORT),
                        ),
                    ),
                ): vol.Coerce(int),
                vol.Optional(
                    CONF_ALLOWED_IPS,
                    default=_get_form_value(
                        current_values,
                        CONF_ALLOWED_IPS,
                        sys_options.get(
                            CONF_ALLOWED_IPS,
                            sys_data.get(CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS),
                        ),
                    ),
                ): str,
                CONTEXT_SECTION_KEY: _build_shared_context_section(shared_defaults),
                DISCOVERY_SECTION_KEY: _build_shared_discovery_section(shared_defaults),
                MEMORY_SECTION_KEY: _build_shared_memory_section(shared_defaults),
                TOOLS_SECTION_KEY: _build_shared_tools_section(
                    shared_defaults,
                    built_in_specs,
                ),
            }
        )

        return self.async_show_form(
            step_id="mcp_server",
            data_schema=mcp_schema,
            errors=errors,
            description_placeholders={
                "warning": (
                    "⚠️ These settings are shared across ALL MCP Assist profiles and "
                    "external MCP clients. Individual profiles can still opt into a "
                    "smaller subset in their own settings."
                )
            },
        )


class CannotConnect(HomeAssistantError):
    """Error to indicate we cannot connect."""


class NoModelsLoaded(HomeAssistantError):
    """Error to indicate no models are loaded."""


class InvalidModel(HomeAssistantError):
    """Error to indicate the model is invalid."""
