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
from homeassistant.data_entry_flow import FlowResult
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
    BooleanSelector,
)

from .const import (
    DOMAIN,
    CONF_PROFILE_NAME,
    CONF_SERVER_TYPE,
    CONF_API_KEY,
    CONF_LMSTUDIO_URL,
    CONF_MODEL_NAME,
    CONF_MCP_PORT,
    CONF_AUTO_START,
    CONF_SYSTEM_PROMPT,
    CONF_TECHNICAL_PROMPT,
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
    CONF_ENABLE_GAP_FILLING,
    SERVER_TYPE_LMSTUDIO,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_OPENAI,
    SERVER_TYPE_GEMINI,
    SERVER_TYPE_ANTHROPIC,
    DEFAULT_SERVER_TYPE,
    DEFAULT_LMSTUDIO_URL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_MCP_PORT,
    DEFAULT_MODEL_NAME,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TECHNICAL_PROMPT,
    DEFAULT_CONTROL_HA,
    DEFAULT_FOLLOW_UP_MODE,
    DEFAULT_RESPONSE_MODE,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_HISTORY,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_DEBUG_MODE,
    DEFAULT_ENABLE_CUSTOM_TOOLS,
    DEFAULT_BRAVE_API_KEY,
    DEFAULT_ALLOWED_IPS,
    DEFAULT_ENABLE_GAP_FILLING,
    DEFAULT_API_KEY,
    OPENAI_BASE_URL,
    GEMINI_BASE_URL,
)

_LOGGER = logging.getLogger(__name__)


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
                _LOGGER.info("✨ FETCH: Returning %d sorted models: %s", len(sorted_models), sorted_models)
                return sorted_models
    except Exception as err:
        _LOGGER.error("💥 FETCH: Exception during fetch: %s", err, exc_info=True)
        return []


async def fetch_models_from_openai(hass: HomeAssistant, api_key: str) -> list[str]:
    """Fetch available models from OpenAI API."""
    _LOGGER.info("🌐 FETCH: Starting OpenAI model fetch")
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession(timeout=timeout) as session:
            _LOGGER.info("📡 FETCH: Requesting OpenAI models")
            async with session.get(
                f"{OPENAI_BASE_URL}/v1/models",
                headers=headers
            ) as resp:
                _LOGGER.info("📥 FETCH: OpenAI response status %d", resp.status)
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.warning("⚠️ FETCH: OpenAI API error %d: %s", resp.status, error_text[:200])
                    return []

                data = await resp.json()
                # Filter for chat models only (exclude embeddings, whisper, etc.)
                all_models = [m.get("id", "") for m in data.get("data", [])]
                # Only include GPT models suitable for chat
                chat_models = [m for m in all_models if m.startswith("gpt-")]
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
                    _LOGGER.warning("⚠️ FETCH: Gemini API error %d: %s", resp.status, error_text[:200])
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
                sorted_models = sorted(gemini_models, reverse=True) if gemini_models else []
                _LOGGER.info("✨ FETCH: Found %d Gemini models", len(sorted_models))
                return sorted_models
    except Exception as err:
        _LOGGER.error("💥 FETCH: Gemini fetch failed: %s", err)
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
    ip_list = [ip.strip() for ip in allowed_ips_str.split(',') if ip.strip()]

    for ip_entry in ip_list:
        try:
            # Try parsing as IP network (handles both individual IPs and CIDR)
            ipaddress.ip_network(ip_entry, strict=False)
        except ValueError:
            # Invalid IP or CIDR format
            return False, f"Invalid IP address or CIDR range: {ip_entry}"

    return True, ""


STEP_USER_DATA_SCHEMA = vol.Schema({
    vol.Required(CONF_PROFILE_NAME): str,
    vol.Required(CONF_SERVER_TYPE, default=DEFAULT_SERVER_TYPE): SelectSelector(
        SelectSelectorConfig(
            options=[
                {"value": "lmstudio", "label": "LM Studio"},
                {"value": "ollama", "label": "Ollama"},
                {"value": "openai", "label": "OpenAI"},
                {"value": "gemini", "label": "Google Gemini"},
                {"value": "anthropic", "label": "Anthropic (Claude)"},
            ],
            mode=SelectSelectorMode.LIST,
        )
    ),
})

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
                    raise CannotConnect(f"LM Studio not responding (status {resp.status})")

                models = await resp.json()
                model_ids = [m.get("id", "") for m in models.get("data", [])]

                if not model_ids:
                    raise NoModelsLoaded("No models loaded in LM Studio")

                # Check if specified model exists
                if model_name not in model_ids:
                    _LOGGER.warning(
                        "Model '%s' not found. Available models: %s",
                        model_name,
                        model_ids
                    )

            # Test chat completions endpoint
            test_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
                "stream": False
            }

            async with session.post(f"{url}/v1/chat/completions", json=test_payload) as resp:
                if resp.status != 200:
                    raise InvalidModel(f"Model '{model_name}' not working (status {resp.status})")

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
        if server_type in [SERVER_TYPE_LMSTUDIO, SERVER_TYPE_OLLAMA]:
            # Local servers - show URL field
            default_url = DEFAULT_OLLAMA_URL if server_type == SERVER_TYPE_OLLAMA else DEFAULT_LMSTUDIO_URL
            server_schema = vol.Schema({
                vol.Required(CONF_LMSTUDIO_URL, default=default_url): str,
            })
        else:
            # Cloud providers - show API key field
            server_schema = vol.Schema({
                vol.Required(CONF_API_KEY): str,
            })

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

        if user_input is not None:
            # Store data and move to step 4 (advanced)
            self.step3_data = user_input
            return await self.async_step_advanced()

        # Get server type to determine model source
        server_type = self.step1_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)
        models = []

        if server_type in [SERVER_TYPE_LMSTUDIO, SERVER_TYPE_OLLAMA]:
            # Local servers - fetch models from API
            server_url = self.step2_data.get(CONF_LMSTUDIO_URL, DEFAULT_LMSTUDIO_URL).rstrip("/")
            _LOGGER.debug("Attempting to fetch models from %s", server_url)
            models = await fetch_models_from_lmstudio(self.hass, server_url)
            _LOGGER.debug("Fetched %d models: %s", len(models), models)
        elif server_type == SERVER_TYPE_OPENAI:
            # OpenAI - fetch models from API with authentication
            api_key = self.step2_data.get(CONF_API_KEY, "")
            _LOGGER.debug("Fetching OpenAI models with API key")
            models = await fetch_models_from_openai(self.hass, api_key)
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

        # Build dynamic schema based on whether models were fetched
        if models:
            # Show dropdown with available models
            _LOGGER.info("Showing model dropdown with %d models", len(models))
            model_schema = vol.Schema({
                vol.Required(CONF_MODEL_NAME): SelectSelector(
                    SelectSelectorConfig(
                        options=models,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(CONF_SYSTEM_PROMPT, default=DEFAULT_SYSTEM_PROMPT): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
                ),
                vol.Required(CONF_TECHNICAL_PROMPT, default=DEFAULT_TECHNICAL_PROMPT): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
                ),
            })
        else:
            # Show text input as fallback
            _LOGGER.info("No models fetched, showing text input")
            model_schema = vol.Schema({
                vol.Required(CONF_MODEL_NAME, default=DEFAULT_MODEL_NAME): str,
                vol.Required(CONF_SYSTEM_PROMPT, default=DEFAULT_SYSTEM_PROMPT): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
                ),
                vol.Required(CONF_TECHNICAL_PROMPT, default=DEFAULT_TECHNICAL_PROMPT): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
                ),
            })

        return self.async_show_form(
            step_id="model",
            data_schema=model_schema,
            errors=errors,
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle step 4 - advanced settings."""
        errors: dict[str, str] = {}

        if user_input is not None:
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
                # Combine data from all 4 steps
                combined_data = {
                    **self.step1_data,
                    **self.step2_data,
                    **self.step3_data,
                    **user_input
                }

                # Create unique ID based on server type + profile name
                profile_name = combined_data[CONF_PROFILE_NAME]
                server_type = combined_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)

                # Map server type to display name
                server_display_map = {
                    SERVER_TYPE_LMSTUDIO: "LM Studio",
                    SERVER_TYPE_OLLAMA: "Ollama",
                    SERVER_TYPE_OPENAI: "OpenAI",
                    SERVER_TYPE_GEMINI: "Gemini",
                    SERVER_TYPE_ANTHROPIC: "Claude",
                }
                server_display = server_display_map.get(server_type, "LM Studio")

                # Include server type in unique ID to allow same profile name across providers
                unique_id = f"{DOMAIN}_{server_type}_{profile_name.lower().replace(' ', '_')}"
                await self.async_set_unique_id(unique_id)
                self._abort_if_unique_id_configured()

                return self.async_create_entry(
                    title=f"{server_display} - {profile_name}",
                    data=combined_data,
                )

        advanced_schema = vol.Schema({
            vol.Required(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): vol.All(
                vol.Coerce(float), vol.Range(min=0.0, max=1.0)
            ),
            vol.Required(CONF_MAX_TOKENS, default=DEFAULT_MAX_TOKENS): vol.Coerce(int),
            vol.Required(CONF_RESPONSE_MODE, default=DEFAULT_RESPONSE_MODE): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        {"value": "default", "label": "Smart"},
                        {"value": "always", "label": "Always"},
                        {"value": "none", "label": "None"},
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Required(CONF_MAX_HISTORY, default=DEFAULT_MAX_HISTORY): vol.Coerce(int),
            vol.Required(CONF_CONTROL_HA, default=DEFAULT_CONTROL_HA): bool,
            vol.Required(CONF_MAX_ITERATIONS, default=DEFAULT_MAX_ITERATIONS): vol.Coerce(int),
            vol.Optional(CONF_ENABLE_CUSTOM_TOOLS, default=DEFAULT_ENABLE_CUSTOM_TOOLS): bool,
            vol.Optional(CONF_BRAVE_API_KEY, default=DEFAULT_BRAVE_API_KEY): str,
            vol.Optional(CONF_ALLOWED_IPS, default=DEFAULT_ALLOWED_IPS): str,
            vol.Optional(CONF_ENABLE_GAP_FILLING, default=DEFAULT_ENABLE_GAP_FILLING): bool,
            vol.Required(CONF_MCP_PORT, default=DEFAULT_MCP_PORT): vol.Coerce(int),
            vol.Required(CONF_DEBUG_MODE, default=DEFAULT_DEBUG_MODE): bool,
        })

        return self.async_show_form(
            step_id="advanced",
            data_schema=advanced_schema,
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get options flow for this handler."""
        return MCPAssistOptionsFlow()


class MCPAssistOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for MCP Assist integration."""

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate allowed IPs if provided
            allowed_ips_str = user_input.get(CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS)
            is_valid, error_msg = validate_allowed_ips(allowed_ips_str)
            if not is_valid:
                errors[CONF_ALLOWED_IPS] = "invalid_ip"
                _LOGGER.warning("Invalid allowed IPs in options: %s", error_msg)

            if not errors:
                # Support both old and new config keys
                if CONF_FOLLOW_UP_MODE in user_input and CONF_RESPONSE_MODE not in user_input:
                    user_input[CONF_RESPONSE_MODE] = user_input[CONF_FOLLOW_UP_MODE]
                    del user_input[CONF_FOLLOW_UP_MODE]

                # Update entry title if profile name changed
                new_profile_name = user_input.get(CONF_PROFILE_NAME)
                old_profile_name = self.config_entry.options.get(CONF_PROFILE_NAME,
                                                                  self.config_entry.data.get(CONF_PROFILE_NAME))
                if new_profile_name and new_profile_name != old_profile_name:
                    # Get server type from original data (can't be changed in options)
                    server_type = self.config_entry.data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)
                    server_display_map = {
                        SERVER_TYPE_LMSTUDIO: "LM Studio",
                        SERVER_TYPE_OLLAMA: "Ollama",
                        SERVER_TYPE_OPENAI: "OpenAI",
                        SERVER_TYPE_GEMINI: "Gemini",
                        SERVER_TYPE_ANTHROPIC: "Claude",
                    }
                    server_display = server_display_map.get(server_type, "LM Studio")
                    self.hass.config_entries.async_update_entry(
                        self.config_entry,
                        title=f"{server_display} - {new_profile_name}"
                    )

                return self.async_create_entry(title="", data=user_input)

        # Get current values from options, then data, then defaults
        options = self.config_entry.options
        data = self.config_entry.data

        # Get server type (can't be changed in options)
        server_type = data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)

        # Handle backward compatibility
        response_mode_value = options.get(CONF_RESPONSE_MODE,
                                         options.get(CONF_FOLLOW_UP_MODE,
                                         DEFAULT_RESPONSE_MODE))

        # Fetch models based on server type
        models = []
        current_model = options.get(CONF_MODEL_NAME, data.get(CONF_MODEL_NAME, DEFAULT_MODEL_NAME))

        if server_type in [SERVER_TYPE_LMSTUDIO, SERVER_TYPE_OLLAMA]:
            # Local servers - fetch from URL
            server_url = options.get(CONF_LMSTUDIO_URL, data.get(CONF_LMSTUDIO_URL, DEFAULT_LMSTUDIO_URL)).rstrip("/")
            _LOGGER.info(f"🔍 OPTIONS: Attempting to fetch models from {server_type} at {server_url}")
            try:
                models = await fetch_models_from_lmstudio(self.hass, server_url)
                _LOGGER.info(f"✅ OPTIONS: Successfully fetched {len(models)} models")
            except Exception as err:
                _LOGGER.error(f"❌ OPTIONS: Failed to fetch models: {err}")
        elif server_type == SERVER_TYPE_OPENAI:
            # OpenAI - fetch from API
            api_key = options.get(CONF_API_KEY, data.get(CONF_API_KEY, ""))
            if api_key:
                _LOGGER.info("🔍 OPTIONS: Attempting to fetch models from OpenAI")
                try:
                    models = await fetch_models_from_openai(self.hass, api_key)
                    _LOGGER.info(f"✅ OPTIONS: Successfully fetched {len(models)} OpenAI models")
                except Exception as err:
                    _LOGGER.error(f"❌ OPTIONS: Failed to fetch OpenAI models: {err}")
        elif server_type == SERVER_TYPE_GEMINI:
            # Gemini - fetch from API
            api_key = options.get(CONF_API_KEY, data.get(CONF_API_KEY, ""))
            if api_key:
                _LOGGER.info("🔍 OPTIONS: Attempting to fetch models from Gemini")
                try:
                    models = await fetch_models_from_gemini(self.hass, api_key)
                    _LOGGER.info(f"✅ OPTIONS: Successfully fetched {len(models)} Gemini models")
                except Exception as err:
                    _LOGGER.error(f"❌ OPTIONS: Failed to fetch Gemini models: {err}")

        # Build model selector based on whether models were fetched
        if models:
            # Show dropdown with available models
            model_selector = SelectSelector(
                SelectSelectorConfig(
                    options=models,
                    mode=SelectSelectorMode.DROPDOWN,
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
                default=options.get(CONF_PROFILE_NAME, data.get(CONF_PROFILE_NAME, "Default"))
            ): str,
        }

        # 2. Server URL or API Key (based on server type)
        if server_type in [SERVER_TYPE_LMSTUDIO, SERVER_TYPE_OLLAMA]:
            server_url = options.get(CONF_LMSTUDIO_URL, data.get(CONF_LMSTUDIO_URL, DEFAULT_LMSTUDIO_URL))
            schema_dict[vol.Required(CONF_LMSTUDIO_URL, default=server_url)] = str
        else:
            # Cloud providers use API key
            api_key = options.get(CONF_API_KEY, data.get(CONF_API_KEY, ""))
            schema_dict[vol.Required(CONF_API_KEY, default=api_key)] = str

        # 3. Model Name (dynamic: dropdown if models found, text input if not)
        schema_dict[vol.Required(CONF_MODEL_NAME, default=current_model)] = model_selector

        # Continue with remaining common fields
        schema_dict.update({

                # 4. System Prompt
                vol.Required(
                    CONF_SYSTEM_PROMPT,
                    default=options.get(CONF_SYSTEM_PROMPT, data.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT))
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)),

                # 5. Technical Instructions
                vol.Required(
                    CONF_TECHNICAL_PROMPT,
                    default=options.get(CONF_TECHNICAL_PROMPT, data.get(CONF_TECHNICAL_PROMPT, DEFAULT_TECHNICAL_PROMPT))
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)),

                # 6. Temperature
                vol.Required(
                    CONF_TEMPERATURE,
                    default=options.get(CONF_TEMPERATURE, data.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),

                # 7. Max Response Tokens
                vol.Required(
                    CONF_MAX_TOKENS,
                    default=options.get(CONF_MAX_TOKENS, data.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
                ): vol.Coerce(int),

                # 8. Response Mode
                vol.Required(
                    CONF_RESPONSE_MODE,
                    default=response_mode_value
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            {"value": "default", "label": "Smart"},
                            {"value": "always", "label": "Always"},
                            {"value": "none", "label": "None"},
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),

                # 9. Max History Messages
                vol.Required(
                    CONF_MAX_HISTORY,
                    default=options.get(CONF_MAX_HISTORY, data.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY))
                ): vol.Coerce(int),

                # 10. Control Home Assistant
                vol.Required(
                    CONF_CONTROL_HA,
                    default=options.get(CONF_CONTROL_HA, data.get(CONF_CONTROL_HA, DEFAULT_CONTROL_HA))
                ): bool,

                # 11. Max Tool Iterations
                vol.Required(
                    CONF_MAX_ITERATIONS,
                    default=options.get(CONF_MAX_ITERATIONS, data.get(CONF_MAX_ITERATIONS, DEFAULT_MAX_ITERATIONS))
                ): vol.Coerce(int),

                # 12. Enable Custom Tools
                vol.Optional(
                    CONF_ENABLE_CUSTOM_TOOLS,
                    default=options.get(CONF_ENABLE_CUSTOM_TOOLS, data.get(CONF_ENABLE_CUSTOM_TOOLS, DEFAULT_ENABLE_CUSTOM_TOOLS))
                ): bool,

                # 13. Brave API Key
                vol.Optional(
                    CONF_BRAVE_API_KEY,
                    default=options.get(CONF_BRAVE_API_KEY, data.get(CONF_BRAVE_API_KEY, DEFAULT_BRAVE_API_KEY))
                ): str,

                # 14. Allowed IPs
                vol.Optional(
                    CONF_ALLOWED_IPS,
                    default=options.get(CONF_ALLOWED_IPS, data.get(CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS))
                ): str,

                # 15. Enable Gap-Filling
                vol.Optional(
                    CONF_ENABLE_GAP_FILLING,
                    default=options.get(CONF_ENABLE_GAP_FILLING, data.get(CONF_ENABLE_GAP_FILLING, DEFAULT_ENABLE_GAP_FILLING))
                ): bool,

                # 16. MCP Server Port
                vol.Required(
                    CONF_MCP_PORT,
                    default=options.get(CONF_MCP_PORT, data.get(CONF_MCP_PORT, DEFAULT_MCP_PORT))
                ): vol.Coerce(int),

                # 17. Debug Mode
                vol.Required(
                    CONF_DEBUG_MODE,
                    default=options.get(CONF_DEBUG_MODE, data.get(CONF_DEBUG_MODE, DEFAULT_DEBUG_MODE))
                ): bool,
        })

        # Create the schema from the built dictionary
        options_schema = vol.Schema(schema_dict)

        return self.async_show_form(
            step_id="init",
            data_schema=options_schema,
            errors=errors,
        )


class CannotConnect(HomeAssistantError):
    """Error to indicate we cannot connect."""


class NoModelsLoaded(HomeAssistantError):
    """Error to indicate no models are loaded."""


class InvalidModel(HomeAssistantError):
    """Error to indicate the model is invalid."""