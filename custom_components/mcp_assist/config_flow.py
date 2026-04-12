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
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
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
    CONF_BRAVE_API_KEY,
    CONF_ALLOWED_IPS,
    CONF_SEARCH_PROVIDER,
    CONF_ENABLE_GAP_FILLING,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
    CONF_ENABLE_RECORDER_TOOLS,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
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
    SERVER_TYPE_MOLTBOT,
    SERVER_TYPE_VLLM,
    DEFAULT_SERVER_TYPE,
    DEFAULT_LMSTUDIO_URL,
    DEFAULT_LLAMACPP_URL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_MOLTBOT_URL,
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
    DEFAULT_SEARCH_PROVIDER,
    DEFAULT_ENABLE_GAP_FILLING,
    DEFAULT_ENABLE_ASSIST_BRIDGE,
    DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
    DEFAULT_ENABLE_RECORDER_TOOLS,
    DEFAULT_ENABLE_CALCULATOR_TOOLS,
    DEFAULT_ENABLE_DEVICE_TOOLS,
    DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_NUM_CTX,
    DEFAULT_FOLLOW_UP_PHRASES,
    DEFAULT_END_WORDS,
    DEFAULT_CLEAN_RESPONSES,
    DEFAULT_TIMEOUT,
    TOOL_FAMILY_ASSIST_BRIDGE,
    TOOL_FAMILY_CALCULATOR,
    TOOL_FAMILY_DEVICE,
    TOOL_FAMILY_MUSIC_ASSISTANT,
    TOOL_FAMILY_PROFILE_SETTINGS,
    TOOL_FAMILY_RECORDER,
    TOOL_FAMILY_RESPONSE_SERVICE,
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
    """Get the current default text for a prompt override field."""
    if current_values and prompt_key in current_values:
        value = current_values.get(prompt_key)
        return "" if value is None else str(value)

    inferred_mode = _infer_prompt_mode(stored_mode, stored_prompt, default_prompt)
    if inferred_mode == PROMPT_MODE_DEFAULT or stored_prompt in (None, ""):
        return ""

    return str(stored_prompt)


def _normalize_prompt_inputs(
    user_input: dict[str, Any], server_type: str
) -> dict[str, Any]:
    """Normalize prompt override inputs before storing."""
    normalized = dict(user_input)

    def _normalize_prompt(prompt_key: str, mode_key: str) -> None:
        raw_value = normalized.get(prompt_key, "")
        text = "" if raw_value is None else str(raw_value)
        if text.strip():
            normalized[prompt_key] = text
            normalized[mode_key] = PROMPT_MODE_CUSTOM
        else:
            normalized.pop(prompt_key, None)
            normalized[mode_key] = PROMPT_MODE_DEFAULT

    if server_type == SERVER_TYPE_MOLTBOT:
        normalized[CONF_SYSTEM_PROMPT_MODE] = PROMPT_MODE_DEFAULT
        normalized.pop(CONF_SYSTEM_PROMPT, None)
    else:
        _normalize_prompt(CONF_SYSTEM_PROMPT, CONF_SYSTEM_PROMPT_MODE)

    _normalize_prompt(CONF_TECHNICAL_PROMPT, CONF_TECHNICAL_PROMPT_MODE)

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
SEARCH_SECTION_KEY = "search"
ENABLED_TOOLS_FIELD = "enabled_tools"
TOOL_FAMILY_OPTIONS = [
    TOOL_FAMILY_DEVICE,
    TOOL_FAMILY_RESPONSE_SERVICE,
    TOOL_FAMILY_RECORDER,
    TOOL_FAMILY_ASSIST_BRIDGE,
    TOOL_FAMILY_CALCULATOR,
    TOOL_FAMILY_MUSIC_ASSISTANT,
]


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


def _tool_family_selector() -> SelectSelector:
    """Build a multi-select for optional tool families."""
    return SelectSelector(
        SelectSelectorConfig(
            options=TOOL_FAMILY_OPTIONS,
            multiple=True,
            mode=SelectSelectorMode.DROPDOWN,
            translation_key="tool_family",
        )
    )


def _selected_tool_families(
    current_values: dict[str, Any] | None,
    setting_map: dict[str, tuple[str, bool]],
    options: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    *,
    explicit_only: bool = False,
) -> list[str]:
    """Return the selected tool families for a config form default."""
    options = options or {}
    data = data or {}
    if explicit_only:
        has_explicit_values = any(
            (
                current_values and setting_key in current_values
            ) or setting_key in options or setting_key in data
            for setting_key, _default in setting_map.values()
        )
        if not has_explicit_values:
            return []
    return [
        family
        for family, (setting_key, default) in setting_map.items()
        if _get_form_value(
            current_values,
            setting_key,
            options.get(setting_key, data.get(setting_key, default)),
        )
    ]


def _apply_tool_family_selection(
    user_input: dict[str, Any],
    setting_map: dict[str, tuple[str, bool]],
    *,
    inherit_when_empty: bool = False,
) -> dict[str, Any]:
    """Expand a multi-select tool list into stored per-family booleans."""
    normalized = dict(user_input)
    selected = normalized.pop(ENABLED_TOOLS_FIELD, None)
    if selected is None:
        return normalized

    selected_set = {
        str(value)
        for value in selected
        if str(value) in setting_map
    }

    if inherit_when_empty and not selected_set:
        for _family, (setting_key, _default) in setting_map.items():
            normalized.pop(setting_key, None)
        return normalized

    for family, (setting_key, _default) in setting_map.items():
        normalized[setting_key] = family in selected_set

    return normalized


def _build_profile_tools_section(
    current_values: dict[str, Any] | None,
    options: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> section:
    """Build the per-profile tool preferences section."""
    options = options or {}
    data = data or {}
    return section(
        vol.Schema(
            {
                vol.Optional(
                    ENABLED_TOOLS_FIELD,
                    default=_selected_tool_families(
                        current_values,
                        TOOL_FAMILY_PROFILE_SETTINGS,
                        options,
                        data,
                        explicit_only=True,
                    ),
                ): _tool_family_selector(),
            }
        ),
        {"collapsed": False},
    )


def _build_shared_tools_section(
    defaults: dict[str, Any],
) -> section:
    """Build the shared MCP server optional tools section."""
    return section(
        vol.Schema(
            {
                vol.Optional(
                    ENABLED_TOOLS_FIELD,
                    default=_selected_tool_families(
                        defaults, TOOL_FAMILY_SHARED_SETTINGS
                    ),
                ): _tool_family_selector(),
            }
        ),
        {"collapsed": False},
    )


def _build_shared_search_section(current_values: dict[str, Any] | None) -> section:
    """Build the shared MCP server search settings section."""
    return section(
        vol.Schema(
            {
                vol.Required(
                    CONF_SEARCH_PROVIDER,
                    default=_get_form_value(
                        current_values, CONF_SEARCH_PROVIDER, DEFAULT_SEARCH_PROVIDER
                    ),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            {"value": "none", "label": "Disabled"},
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
                        current_values, CONF_BRAVE_API_KEY, DEFAULT_BRAVE_API_KEY
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
                    {"value": "moltbot", "label": "Moltbot"},
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
            # Store data and move to step 3
            self.step2_data = user_input
            return await self.async_step_model()

        # Get server type from step 1 to build dynamic schema
        server_type = self.step1_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)

        # Build schema based on server type
        if server_type in [
            SERVER_TYPE_LMSTUDIO,
            SERVER_TYPE_LLAMACPP,
            SERVER_TYPE_OLLAMA,
            SERVER_TYPE_MOLTBOT,
            SERVER_TYPE_VLLM,
        ]:
            # Local servers - show URL field
            if server_type == SERVER_TYPE_OLLAMA:
                default_url = DEFAULT_OLLAMA_URL
            elif server_type == SERVER_TYPE_LLAMACPP:
                default_url = DEFAULT_LLAMACPP_URL
            elif server_type == SERVER_TYPE_MOLTBOT:
                default_url = DEFAULT_MOLTBOT_URL
            elif server_type == SERVER_TYPE_VLLM:
                default_url = DEFAULT_VLLM_URL
            else:
                default_url = DEFAULT_LMSTUDIO_URL

            # Moltbot needs both URL and API key (bearer token)
            if server_type == SERVER_TYPE_MOLTBOT:
                server_schema = vol.Schema(
                    {
                        vol.Required(CONF_LMSTUDIO_URL, default=default_url): str,
                        vol.Required(CONF_API_KEY): TextSelector(
                            TextSelectorConfig(type=TextSelectorType.PASSWORD)
                        ),
                    }
                )
            else:
                # Other local servers (LM Studio, Ollama, llamacpp, vLLM) - just URL
                server_schema = vol.Schema(
                    {
                        vol.Required(CONF_LMSTUDIO_URL, default=default_url): str,
                    }
                )
        elif server_type == SERVER_TYPE_OPENAI:
            # OpenAI - hybrid like Moltbot (URL + API key)
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

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle step 3 - model selection and prompts."""
        errors: dict[str, str] = {}

        # Get server type to determine model source
        server_type = self.step1_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)

        if user_input is not None:
            user_input = _normalize_prompt_inputs(user_input, server_type)
            if server_type == SERVER_TYPE_MOLTBOT:
                user_input[CONF_MODEL_NAME] = "main"

            # Store data and move to step 4 (advanced)
            self.step3_data = user_input
            return await self.async_step_advanced()

        models = []
        current_values = getattr(self, "step3_data", {})
        default_system_prompt = _get_default_system_prompt(self.hass)

        # Moltbot doesn't have /v1/models endpoint - skip model selection
        if server_type == SERVER_TYPE_MOLTBOT:
            technical_prompt_suggestion = _get_prompt_text_default(
                current_values,
                prompt_key=CONF_TECHNICAL_PROMPT,
                stored_prompt=None,
                default_prompt=DEFAULT_TECHNICAL_PROMPT,
            )
            model_schema = vol.Schema(
                {
                    _optional_with_suggested_value(
                        CONF_TECHNICAL_PROMPT, technical_prompt_suggestion
                    ): TextSelector(
                        TextSelectorConfig(
                            type=TextSelectorType.TEXT, multiline=True
                        )
                    ),
                }
            )

            return self.async_show_form(
                step_id="model",
                data_schema=model_schema,
                errors=errors,
                description_placeholders={
                    "server_info": "Moltbot's model and system prompt are configured on the Moltbot server. Leave Technical Instructions blank to use the built-in default from the integration."
                },
            )
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
            vol.Required(CONF_MODEL_NAME, default=current_model): model_field,
            _optional_with_suggested_value(
                CONF_SYSTEM_PROMPT, system_prompt_suggestion
            ): TextSelector(
                TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
            ),
            _optional_with_suggested_value(
                CONF_TECHNICAL_PROMPT, technical_prompt_suggestion
            ): TextSelector(
                TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
            ),
        }

        model_schema = vol.Schema(schema_dict)

        return self.async_show_form(
            step_id="model",
            data_schema=model_schema,
            errors=errors,
            description_placeholders={
                "server_info": "Select a model. Leave System Prompt or Technical Instructions blank to use the built-in defaults from the integration. Models are automatically loaded from your server."
            },
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle step 4 - advanced settings."""
        errors: dict[str, str] = {}

        # Get server type to determine which fields to show
        server_type = self.step1_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)

        if user_input is not None:
            user_input = _flatten_section_values(user_input, TOOLS_SECTION_KEY)
            user_input = _apply_tool_family_selection(
                user_input,
                TOOL_FAMILY_PROFILE_SETTINGS,
                inherit_when_empty=True,
            )

            # For Moltbot, set defaults for hidden fields
            if server_type == SERVER_TYPE_MOLTBOT:
                user_input[CONF_TEMPERATURE] = DEFAULT_TEMPERATURE
                user_input[CONF_MAX_TOKENS] = DEFAULT_MAX_TOKENS
                user_input[CONF_MAX_HISTORY] = DEFAULT_MAX_HISTORY
                user_input[CONF_MAX_ITERATIONS] = DEFAULT_MAX_ITERATIONS
                user_input[CONF_RESPONSE_MODE] = "none"  # Moltbot manages this
                user_input[CONF_FOLLOW_UP_PHRASES] = DEFAULT_FOLLOW_UP_PHRASES
                user_input[CONF_END_WORDS] = DEFAULT_END_WORDS
                # Timeout is shown for Moltbot, so use provided value or default to 60
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
                            CONF_ENABLE_RECORDER_TOOLS: existing_entry.data.get(
                                CONF_ENABLE_RECORDER_TOOLS,
                                DEFAULT_ENABLE_RECORDER_TOOLS,
                            ),
                            CONF_ENABLE_CALCULATOR_TOOLS: existing_entry.data.get(
                                CONF_ENABLE_CALCULATOR_TOOLS,
                                DEFAULT_ENABLE_CALCULATOR_TOOLS,
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
                        SERVER_TYPE_MOLTBOT: "Moltbot",
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
        if server_type == SERVER_TYPE_MOLTBOT:
            # Moltbot - only show Control HA, Timeout, Clean Responses, Debug
            advanced_schema_dict = {
                vol.Required(CONF_CONTROL_HA, default=DEFAULT_CONTROL_HA): bool,
                vol.Optional(
                    CONF_CLEAN_RESPONSES, default=DEFAULT_CLEAN_RESPONSES
                ): bool,
                vol.Required(CONF_TIMEOUT, default=60): vol.All(
                    vol.Coerce(int), vol.Range(min=5, max=300)
                ),
                vol.Required(CONF_DEBUG_MODE, default=DEFAULT_DEBUG_MODE): bool,
            }
        else:
            # Other servers - show all fields
            advanced_schema_dict = {
                vol.Required(CONF_TEMPERATURE, default=default_temp): vol.All(
                    vol.Coerce(float), vol.Range(min=0.0, max=1.0)
                ),
                vol.Required(CONF_MAX_TOKENS, default=DEFAULT_MAX_TOKENS): vol.Coerce(
                    int
                ),
            }

            # Add Ollama-specific fields in correct position (after Max Tokens)
            if server_type == SERVER_TYPE_OLLAMA:
                advanced_schema_dict[
                    vol.Optional(CONF_OLLAMA_NUM_CTX, default=DEFAULT_OLLAMA_NUM_CTX)
                ] = vol.Coerce(int)
                advanced_schema_dict[
                    vol.Optional(
                        CONF_OLLAMA_KEEP_ALIVE, default=DEFAULT_OLLAMA_KEEP_ALIVE
                    )
                ] = str

            # Continue with remaining fields
            advanced_schema_dict.update(
                {
                    vol.Required(
                        CONF_MAX_HISTORY, default=DEFAULT_MAX_HISTORY
                    ): vol.Coerce(int),
                    vol.Required(CONF_CONTROL_HA, default=DEFAULT_CONTROL_HA): bool,
                    vol.Required(
                        CONF_MAX_ITERATIONS, default=DEFAULT_MAX_ITERATIONS
                    ): vol.Coerce(int),
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
                    vol.Optional(
                        CONF_CLEAN_RESPONSES, default=DEFAULT_CLEAN_RESPONSES
                    ): bool,
                    vol.Required(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): vol.All(
                        vol.Coerce(int), vol.Range(min=5, max=300)
                    ),
                    vol.Required(CONF_DEBUG_MODE, default=DEFAULT_DEBUG_MODE): bool,
                }
            )

        advanced_schema_dict[TOOLS_SECTION_KEY] = _build_profile_tools_section(
            getattr(self, "step4_data", {})
        )

        advanced_schema = vol.Schema(advanced_schema_dict)

        # Set description based on server type
        if server_type == SERVER_TYPE_MOLTBOT:
            description_placeholders = {
                "advanced_info": (
                    "Moltbot manages temperature, token limits, history, and tool "
                    "iterations internally. Only essential settings are shown. The "
                    "Tools section still lets this profile opt into a smaller MCP "
                    "tool set."
                )
            }
        else:
            description_placeholders = {
                "advanced_info": (
                    "Configure temperature, token limits, and other advanced options. "
                    "The Tools section only affects this profile and can narrow the "
                    "shared MCP server tool set for smaller models."
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

        if user_input is not None:
            user_input = _flatten_section_values(
                user_input, SEARCH_SECTION_KEY, DISCOVERY_SECTION_KEY, TOOLS_SECTION_KEY
            )
            user_input = _apply_tool_family_selection(
                user_input, TOOL_FAMILY_SHARED_SETTINGS
            )
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
                    SERVER_TYPE_MOLTBOT: "Moltbot",
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
            CONF_BRAVE_API_KEY: _get_form_value(
                current_values,
                CONF_BRAVE_API_KEY,
                DEFAULT_BRAVE_API_KEY,
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
            CONF_ENABLE_RECORDER_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_RECORDER_TOOLS,
                DEFAULT_ENABLE_RECORDER_TOOLS,
            ),
            CONF_ENABLE_CALCULATOR_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_CALCULATOR_TOOLS,
                DEFAULT_ENABLE_CALCULATOR_TOOLS,
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
        }

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
                SEARCH_SECTION_KEY: _build_shared_search_section(shared_defaults),
                DISCOVERY_SECTION_KEY: _build_shared_discovery_section(shared_defaults),
                TOOLS_SECTION_KEY: _build_shared_tools_section(shared_defaults),
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
                    "profiles can still opt into a smaller subset later."
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

        if user_input is not None:
            user_input = _flatten_section_values(user_input, TOOLS_SECTION_KEY)
            user_input = _apply_tool_family_selection(
                user_input,
                TOOL_FAMILY_PROFILE_SETTINGS,
                inherit_when_empty=True,
            )
            user_input = _normalize_prompt_inputs(user_input, server_type)

            if not errors:
                # Support both old and new config keys
                if (
                    CONF_FOLLOW_UP_MODE in user_input
                    and CONF_RESPONSE_MODE not in user_input
                ):
                    user_input[CONF_RESPONSE_MODE] = user_input[CONF_FOLLOW_UP_MODE]
                    del user_input[CONF_FOLLOW_UP_MODE]

                # For Moltbot, ensure model name and empty system prompt are set
                if server_type == SERVER_TYPE_MOLTBOT:
                    if CONF_MODEL_NAME not in user_input:
                        user_input[CONF_MODEL_NAME] = "main"
                    user_input[CONF_SYSTEM_PROMPT] = ""  # Moltbot manages its own
                    user_input[CONF_SYSTEM_PROMPT_MODE] = PROMPT_MODE_DEFAULT

                # Store profile settings and proceed to MCP server settings
                self.profile_options = user_input
                return await self.async_step_mcp_server()

        # Get current values from options, then data, then defaults
        options = self.config_entry.options
        data = self.config_entry.data
        current_values = self.profile_options or {}
        default_system_prompt = _get_default_system_prompt(self.hass)

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

        # Moltbot doesn't have /v1/models - skip model fetching
        if server_type == SERVER_TYPE_MOLTBOT:
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

        # Build schema based on server type
        schema_dict = {
            # 1. Profile Name
            vol.Required(
                CONF_PROFILE_NAME,
                default=_get_form_value(
                    current_values,
                    CONF_PROFILE_NAME,
                    options.get(CONF_PROFILE_NAME, data.get(CONF_PROFILE_NAME, "Default")),
                ),
            ): str,
        }

        system_prompt_suggestion = _get_prompt_text_default(
            current_values,
            prompt_key=CONF_SYSTEM_PROMPT,
            stored_mode=options.get(CONF_SYSTEM_PROMPT_MODE, data.get(CONF_SYSTEM_PROMPT_MODE)),
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

        # 2. Server URL or API Key (based on server type)
        if server_type in [
            SERVER_TYPE_LMSTUDIO,
            SERVER_TYPE_LLAMACPP,
            SERVER_TYPE_OLLAMA,
            SERVER_TYPE_MOLTBOT,
            SERVER_TYPE_VLLM,
        ]:
            server_url = options.get(
                CONF_LMSTUDIO_URL, data.get(CONF_LMSTUDIO_URL, DEFAULT_LMSTUDIO_URL)
            )
            schema_dict[
                vol.Required(
                    CONF_LMSTUDIO_URL,
                    default=_get_form_value(current_values, CONF_LMSTUDIO_URL, server_url),
                )
            ] = str

            # Moltbot also needs bearer token (required)
            if server_type == SERVER_TYPE_MOLTBOT:
                api_key = options.get(CONF_API_KEY, data.get(CONF_API_KEY, ""))
                schema_dict[
                    vol.Required(
                        CONF_API_KEY,
                        default=_get_form_value(current_values, CONF_API_KEY, api_key),
                    )
                ] = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))
        elif server_type == SERVER_TYPE_OPENAI:
            # OpenAI - hybrid like Moltbot (URL + API key)
            server_url = options.get(
                CONF_LMSTUDIO_URL,
                data.get(CONF_LMSTUDIO_URL, OPENAI_BASE_URL)
            )
            schema_dict[
                vol.Required(
                    CONF_LMSTUDIO_URL,
                    default=_get_form_value(current_values, CONF_LMSTUDIO_URL, server_url),
                )
            ] = str

            api_key = options.get(CONF_API_KEY, data.get(CONF_API_KEY, ""))
            schema_dict[
                vol.Required(
                    CONF_API_KEY,
                    default=_get_form_value(current_values, CONF_API_KEY, api_key),
                )
            ] = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))
        else:
            # Other cloud providers (Gemini, Anthropic, OpenRouter) - API key only
            api_key = options.get(CONF_API_KEY, data.get(CONF_API_KEY, ""))
            schema_dict[
                vol.Required(
                    CONF_API_KEY,
                    default=_get_form_value(current_values, CONF_API_KEY, api_key),
                )
            ] = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))

        # 3. Model Name (skip for Moltbot - hardcoded to "main")
        if server_type != SERVER_TYPE_MOLTBOT:
            schema_dict[
                vol.Required(CONF_MODEL_NAME, default=current_model)
            ] = model_selector

        # Continue with remaining common fields
        # 4. System Prompt (skip for Moltbot - it manages its own)
        if server_type != SERVER_TYPE_MOLTBOT:
            schema_dict[
                _optional_with_suggested_value(
                    CONF_SYSTEM_PROMPT, system_prompt_suggestion
                )
            ] = TextSelector(
                TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
            )

        # 5. Technical Instructions
        schema_dict[
            _optional_with_suggested_value(
                CONF_TECHNICAL_PROMPT, technical_prompt_suggestion
            )
        ] = TextSelector(
            TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
        )

        # For Moltbot, only show Control HA, Timeout, Clean Responses, and Debug Mode
        if server_type == SERVER_TYPE_MOLTBOT:
            schema_dict.update(
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
                    vol.Optional(
                        CONF_CLEAN_RESPONSES,
                        default=_get_form_value(
                            current_values,
                            CONF_CLEAN_RESPONSES,
                            options.get(
                                CONF_CLEAN_RESPONSES,
                                data.get(CONF_CLEAN_RESPONSES, DEFAULT_CLEAN_RESPONSES),
                            ),
                        ),
                    ): bool,
                    vol.Required(
                        CONF_TIMEOUT,
                        default=_get_form_value(
                            current_values,
                            CONF_TIMEOUT,
                            options.get(CONF_TIMEOUT, data.get(CONF_TIMEOUT, 60)),
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
            )
        else:
            # Other servers - show all fields
            schema_dict.update(
                {
                    # 6. Temperature
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
                    # 7. Max Response Tokens
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
                }
            )

            # Add Ollama-specific fields in correct position (after Max Tokens)
            if server_type == SERVER_TYPE_OLLAMA:
                schema_dict[
                    vol.Optional(
                        CONF_OLLAMA_NUM_CTX,
                        default=_get_form_value(
                            current_values,
                            CONF_OLLAMA_NUM_CTX,
                            options.get(
                                CONF_OLLAMA_NUM_CTX,
                                data.get(CONF_OLLAMA_NUM_CTX, DEFAULT_OLLAMA_NUM_CTX),
                            ),
                        ),
                    )
                ] = vol.Coerce(int)
                schema_dict[
                    vol.Optional(
                        CONF_OLLAMA_KEEP_ALIVE,
                        default=_get_form_value(
                            current_values,
                            CONF_OLLAMA_KEEP_ALIVE,
                            options.get(
                                CONF_OLLAMA_KEEP_ALIVE,
                                data.get(CONF_OLLAMA_KEEP_ALIVE, DEFAULT_OLLAMA_KEEP_ALIVE),
                            ),
                        ),
                    )
                ] = str

            # Continue with remaining fields
            schema_dict.update(
                {
                    # 8/10. Max History Messages
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
                    # 9/11. Control Home Assistant
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
                    # 10/12. Max Tool Iterations
                    vol.Required(
                        CONF_MAX_ITERATIONS,
                        default=_get_form_value(
                            current_values,
                            CONF_MAX_ITERATIONS,
                            options.get(
                                CONF_MAX_ITERATIONS,
                                data.get(CONF_MAX_ITERATIONS, DEFAULT_MAX_ITERATIONS),
                            ),
                        ),
                    ): vol.Coerce(int),
                    # 11/13. Response Mode
                    vol.Required(
                        CONF_RESPONSE_MODE,
                        default=_get_form_value(
                            current_values, CONF_RESPONSE_MODE, response_mode_value
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
                    # 12/14. Follow-up Phrases
                    vol.Optional(
                        CONF_FOLLOW_UP_PHRASES,
                        default=_get_form_value(
                            current_values,
                            CONF_FOLLOW_UP_PHRASES,
                            options.get(
                                CONF_FOLLOW_UP_PHRASES,
                                data.get(CONF_FOLLOW_UP_PHRASES, DEFAULT_FOLLOW_UP_PHRASES),
                            ),
                        ),
                    ): TextSelector(TextSelectorConfig(multiline=True)),
                    # 13/15. End Conversation Words
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
                    # 14/16. Clean Responses
                    vol.Required(
                        CONF_CLEAN_RESPONSES,
                        default=_get_form_value(
                            current_values,
                            CONF_CLEAN_RESPONSES,
                            options.get(
                                CONF_CLEAN_RESPONSES,
                                data.get(CONF_CLEAN_RESPONSES, DEFAULT_CLEAN_RESPONSES),
                            ),
                        ),
                    ): bool,
                    # 15/17. Response Time Out
                    vol.Required(
                        CONF_TIMEOUT,
                        default=_get_form_value(
                            current_values,
                            CONF_TIMEOUT,
                            options.get(CONF_TIMEOUT, data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)),
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=5, max=300)),
                    # 16/18. Debug Mode
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
            )

        # Create the schema from the built dictionary
        schema_dict[TOOLS_SECTION_KEY] = _build_profile_tools_section(
            current_values, options, data
        )

        options_schema = vol.Schema(schema_dict)

        # Set description based on server type
        if server_type == SERVER_TYPE_MOLTBOT:
            description_placeholders = {
                "server_info": (
                    "Moltbot's model and system prompt are configured on the Moltbot "
                    "server. Leave Technical Instructions blank to use the built-in "
                    "default from the integration. The Tools section can still keep "
                    "this profile on a smaller MCP tool set."
                )
            }
        else:
            description_placeholders = {
                "server_info": (
                    "Configure this conversation profile. These settings only affect "
                    "this profile. Leave System Prompt or Technical Instructions "
                    "blank to use the built-in defaults from the integration code. "
                    "The Tools section can keep smaller models on a leaner subset of "
                    "the shared MCP server."
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

        if user_input is not None:
            user_input = _flatten_section_values(
                user_input, SEARCH_SECTION_KEY, DISCOVERY_SECTION_KEY, TOOLS_SECTION_KEY
            )
            user_input = _apply_tool_family_selection(
                user_input, TOOL_FAMILY_SHARED_SETTINGS
            )
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
                        SERVER_TYPE_MOLTBOT: "Moltbot",
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
            CONF_BRAVE_API_KEY: _get_form_value(
                current_values,
                CONF_BRAVE_API_KEY,
                sys_options.get(
                    CONF_BRAVE_API_KEY,
                    sys_data.get(CONF_BRAVE_API_KEY, DEFAULT_BRAVE_API_KEY),
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
            CONF_ENABLE_CALCULATOR_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_CALCULATOR_TOOLS,
                sys_options.get(
                    CONF_ENABLE_CALCULATOR_TOOLS,
                    sys_data.get(
                        CONF_ENABLE_CALCULATOR_TOOLS,
                        DEFAULT_ENABLE_CALCULATOR_TOOLS,
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
                SEARCH_SECTION_KEY: _build_shared_search_section(shared_defaults),
                DISCOVERY_SECTION_KEY: _build_shared_discovery_section(shared_defaults),
                TOOLS_SECTION_KEY: _build_shared_tools_section(shared_defaults),
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
