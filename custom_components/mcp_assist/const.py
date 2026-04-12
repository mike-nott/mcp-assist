"""Constants for the MCP Assist integration."""

DOMAIN = "mcp_assist"
SYSTEM_ENTRY_UNIQUE_ID = "mcp_assist_system_settings"

# Server type options
SERVER_TYPE_LMSTUDIO = "lmstudio"
SERVER_TYPE_LLAMACPP = "llamacpp"
SERVER_TYPE_OLLAMA = "ollama"
SERVER_TYPE_OPENAI = "openai"
SERVER_TYPE_GEMINI = "gemini"
SERVER_TYPE_ANTHROPIC = "anthropic"
SERVER_TYPE_OPENROUTER = "openrouter"
SERVER_TYPE_MOLTBOT = "moltbot"
SERVER_TYPE_VLLM = "vllm"

# Configuration keys
CONF_PROFILE_NAME = "profile_name"
CONF_SERVER_TYPE = "server_type"
CONF_API_KEY = "api_key"
CONF_LMSTUDIO_URL = "lmstudio_url"
CONF_MODEL_NAME = "model_name"
CONF_MCP_PORT = "mcp_port"
CONF_AUTO_START = "auto_start"
CONF_SYSTEM_PROMPT = "system_prompt"
CONF_TECHNICAL_PROMPT = "technical_prompt"
CONF_SYSTEM_PROMPT_MODE = "system_prompt_mode"
CONF_TECHNICAL_PROMPT_MODE = "technical_prompt_mode"
CONF_CONTROL_HA = "control_home_assistant"
CONF_RESPONSE_MODE = "response_mode"
CONF_FOLLOW_UP_MODE = "follow_up_mode"  # Keep for backward compatibility
CONF_TEMPERATURE = "temperature"
CONF_MAX_TOKENS = "max_tokens"
CONF_MAX_HISTORY = "max_history"
CONF_MAX_ITERATIONS = "max_iterations"
CONF_DEBUG_MODE = "debug_mode"
CONF_ENABLE_CUSTOM_TOOLS = "enable_custom_tools"
CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS = "enable_external_custom_tools"
CONF_BRAVE_API_KEY = "brave_api_key"
CONF_ALLOWED_IPS = "allowed_ips"
CONF_INCLUDE_CURRENT_USER = "include_current_user"
CONF_INCLUDE_HOME_LOCATION = "include_home_location"
CONF_SEARCH_PROVIDER = "search_provider"
CONF_ENABLE_WEB_SEARCH = "enable_web_search"
CONF_ENABLE_GAP_FILLING = "enable_gap_filling"
CONF_ENABLE_ASSIST_BRIDGE = "enable_assist_bridge"
CONF_ENABLE_RESPONSE_SERVICE_TOOLS = "enable_response_service_tools"
CONF_ENABLE_WEATHER_FORECAST_TOOL = "enable_weather_forecast_tool"
CONF_ENABLE_RECORDER_TOOLS = "enable_recorder_tools"
CONF_ENABLE_MEMORY_TOOLS = "enable_memory_tools"
CONF_ENABLE_CALCULATOR_TOOLS = "enable_calculator_tools"
CONF_ENABLE_UNIT_CONVERSION_TOOLS = "enable_unit_conversion_tools"
CONF_ENABLE_DEVICE_TOOLS = "enable_device_tools"
CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT = "enable_music_assistant_support"
CONF_MEMORY_DEFAULT_TTL_DAYS = "memory_default_ttl_days"
CONF_MEMORY_MAX_TTL_DAYS = "memory_max_ttl_days"
CONF_MEMORY_MAX_ITEMS = "memory_max_items"
CONF_PROFILE_ENABLE_WEB_SEARCH = "profile_enable_web_search"
CONF_PROFILE_ENABLE_ASSIST_BRIDGE = "profile_enable_assist_bridge"
CONF_PROFILE_ENABLE_RESPONSE_SERVICE_TOOLS = "profile_enable_response_service_tools"
CONF_PROFILE_ENABLE_WEATHER_FORECAST_TOOL = "profile_enable_weather_forecast_tool"
CONF_PROFILE_ENABLE_RECORDER_TOOLS = "profile_enable_recorder_tools"
CONF_PROFILE_ENABLE_MEMORY_TOOLS = "profile_enable_memory_tools"
CONF_PROFILE_ENABLE_CALCULATOR_TOOLS = "profile_enable_calculator_tools"
CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS = "profile_enable_unit_conversion_tools"
CONF_PROFILE_ENABLE_DEVICE_TOOLS = "profile_enable_device_tools"
CONF_PROFILE_ENABLE_MUSIC_ASSISTANT_SUPPORT = "profile_enable_music_assistant_support"
CONF_OLLAMA_KEEP_ALIVE = "ollama_keep_alive"
CONF_OLLAMA_NUM_CTX = "ollama_num_ctx"
CONF_FOLLOW_UP_PHRASES = "follow_up_phrases"
CONF_END_WORDS = "end_words"
CONF_CLEAN_RESPONSES = "clean_responses"
CONF_TIMEOUT = "timeout"

# Default values
DEFAULT_SERVER_TYPE = "lmstudio"
DEFAULT_LMSTUDIO_URL = "http://localhost:1234"
DEFAULT_LLAMACPP_URL = "http://localhost:8080"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MOLTBOT_URL = "http://localhost:18789"
DEFAULT_VLLM_URL = "http://localhost:8000"
DEFAULT_MCP_PORT = 8090
DEFAULT_API_KEY = ""

# Cloud provider base URLs
OPENAI_BASE_URL = "https://api.openai.com"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"

# No hardcoded model lists - models are fetched dynamically from provider APIs
DEFAULT_MODEL_NAME = "model"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful Home Assistant voice assistant. Respond naturally and "
    "conversationally to user requests."
)
PROMPT_MODE_DEFAULT = "default"
PROMPT_MODE_CUSTOM = "custom"
DEFAULT_SYSTEM_PROMPT_MODE = PROMPT_MODE_DEFAULT
DEFAULT_TECHNICAL_PROMPT_MODE = PROMPT_MODE_DEFAULT
DEFAULT_CONTROL_HA = True
DEFAULT_RESPONSE_MODE = "default"
DEFAULT_FOLLOW_UP_MODE = "default"  # Keep for backward compatibility
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 500
DEFAULT_MAX_HISTORY = 10
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_DEBUG_MODE = False
DEFAULT_ENABLE_CUSTOM_TOOLS = False
DEFAULT_ENABLE_EXTERNAL_CUSTOM_TOOLS = False
DEFAULT_BRAVE_API_KEY = ""
DEFAULT_ALLOWED_IPS = ""
DEFAULT_INCLUDE_CURRENT_USER = True
DEFAULT_INCLUDE_HOME_LOCATION = True
DEFAULT_SEARCH_PROVIDER = "none"
DEFAULT_ENABLE_WEB_SEARCH = False
DEFAULT_ENABLE_GAP_FILLING = True
DEFAULT_ENABLE_ASSIST_BRIDGE = False
DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS = True
DEFAULT_ENABLE_WEATHER_FORECAST_TOOL = True
DEFAULT_ENABLE_RECORDER_TOOLS = True
DEFAULT_ENABLE_MEMORY_TOOLS = False
DEFAULT_ENABLE_CALCULATOR_TOOLS = False
DEFAULT_ENABLE_UNIT_CONVERSION_TOOLS = False
DEFAULT_ENABLE_DEVICE_TOOLS = True
DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT = False
DEFAULT_MEMORY_DEFAULT_TTL_DAYS = 30
DEFAULT_MEMORY_MAX_TTL_DAYS = 365
DEFAULT_MEMORY_MAX_ITEMS = 500
DEFAULT_PROFILE_ENABLE_WEB_SEARCH = True
DEFAULT_PROFILE_ENABLE_ASSIST_BRIDGE = True
DEFAULT_PROFILE_ENABLE_RESPONSE_SERVICE_TOOLS = True
DEFAULT_PROFILE_ENABLE_WEATHER_FORECAST_TOOL = True
DEFAULT_PROFILE_ENABLE_RECORDER_TOOLS = True
DEFAULT_PROFILE_ENABLE_MEMORY_TOOLS = True
DEFAULT_PROFILE_ENABLE_CALCULATOR_TOOLS = True
DEFAULT_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS = True
DEFAULT_PROFILE_ENABLE_DEVICE_TOOLS = True
DEFAULT_PROFILE_ENABLE_MUSIC_ASSISTANT_SUPPORT = True
DEFAULT_OLLAMA_KEEP_ALIVE = "5m"  # 5 minutes
DEFAULT_OLLAMA_NUM_CTX = 0  # 0 = use model default
DEFAULT_FOLLOW_UP_PHRASES = (
    "anything else, what else, would you, do you, should i, can i, which, "
    "how can, what about, is there"
)
DEFAULT_END_WORDS = (
    "stop, cancel, no, nope, thanks, thank you, bye, goodbye, done, never mind, "
    "nevermind, forget it, that's all, that's it"
)
DEFAULT_CLEAN_RESPONSES = False
DEFAULT_TIMEOUT = 30

CUSTOM_TOOLS_DIRECTORY = "mcp-assist-tools"
CUSTOM_TOOL_SCHEMA_VERSION = 1

TOOL_FAMILY_DEVICE = "device"
TOOL_FAMILY_ASSIST_BRIDGE = "assist_bridge"
TOOL_FAMILY_RESPONSE_SERVICE = "response_service"
TOOL_FAMILY_WEATHER_FORECAST = "weather_forecast"
TOOL_FAMILY_RECORDER = "recorder"
TOOL_FAMILY_MEMORY = "memory"
TOOL_FAMILY_CALCULATOR = "calculator"
TOOL_FAMILY_UNIT_CONVERSION = "unit_conversion"
TOOL_FAMILY_MUSIC_ASSISTANT = "music_assistant"
TOOL_FAMILY_WEB_SEARCH = "web_search"

OPTIONAL_TOOL_FAMILY_TOOL_NAMES = {
    TOOL_FAMILY_DEVICE: frozenset({"discover_devices", "get_device_details"}),
    TOOL_FAMILY_ASSIST_BRIDGE: frozenset(
        {
            "list_assist_tools",
            "call_assist_tool",
            "get_assist_prompt",
            "get_assist_context_snapshot",
        }
    ),
    TOOL_FAMILY_RESPONSE_SERVICE: frozenset(
        {
            "list_response_services",
            "call_service_with_response",
        }
    ),
    TOOL_FAMILY_WEATHER_FORECAST: frozenset({"get_weather_forecast"}),
    TOOL_FAMILY_RECORDER: frozenset(
        {
            "get_entity_history",
            "get_last_entity_event",
            "analyze_entity_history",
            "get_entity_state_at_time",
        }
    ),
    TOOL_FAMILY_MEMORY: frozenset(
        {
            "remember_memory",
            "recall_memories",
            "forget_memory",
        }
    ),
    TOOL_FAMILY_CALCULATOR: frozenset(
        {
            "add",
            "subtract",
            "multiply",
            "divide",
            "sqrt",
            "power",
            "round_number",
            "average",
            "min_value",
            "max_value",
            "evaluate_expression",
        }
    ),
    TOOL_FAMILY_UNIT_CONVERSION: frozenset({"convert_unit"}),
    TOOL_FAMILY_MUSIC_ASSISTANT: frozenset(
        {
            "list_music_assistant_players",
            "play_music_assistant",
            "list_music_assistant_instances",
            "search_music_assistant",
            "get_music_assistant_library",
            "get_music_assistant_queue",
        }
    ),
    TOOL_FAMILY_WEB_SEARCH: frozenset({"search", "read_url"}),
}

OPTIONAL_TOOL_NAME_TO_FAMILY = {
    tool_name: family
    for family, tool_names in OPTIONAL_TOOL_FAMILY_TOOL_NAMES.items()
    for tool_name in tool_names
}

TOOL_FAMILY_SHARED_SETTINGS = {
    TOOL_FAMILY_DEVICE: (
        CONF_ENABLE_DEVICE_TOOLS,
        DEFAULT_ENABLE_DEVICE_TOOLS,
    ),
    TOOL_FAMILY_ASSIST_BRIDGE: (
        CONF_ENABLE_ASSIST_BRIDGE,
        DEFAULT_ENABLE_ASSIST_BRIDGE,
    ),
    TOOL_FAMILY_RESPONSE_SERVICE: (
        CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
        DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
    ),
    TOOL_FAMILY_RECORDER: (
        CONF_ENABLE_RECORDER_TOOLS,
        DEFAULT_ENABLE_RECORDER_TOOLS,
    ),
    TOOL_FAMILY_MEMORY: (
        CONF_ENABLE_MEMORY_TOOLS,
        DEFAULT_ENABLE_MEMORY_TOOLS,
    ),
    TOOL_FAMILY_WEATHER_FORECAST: (
        CONF_ENABLE_WEATHER_FORECAST_TOOL,
        DEFAULT_ENABLE_WEATHER_FORECAST_TOOL,
    ),
    TOOL_FAMILY_CALCULATOR: (
        CONF_ENABLE_CALCULATOR_TOOLS,
        DEFAULT_ENABLE_CALCULATOR_TOOLS,
    ),
    TOOL_FAMILY_UNIT_CONVERSION: (
        CONF_ENABLE_UNIT_CONVERSION_TOOLS,
        DEFAULT_ENABLE_UNIT_CONVERSION_TOOLS,
    ),
    TOOL_FAMILY_MUSIC_ASSISTANT: (
        CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
        DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    ),
    TOOL_FAMILY_WEB_SEARCH: (
        CONF_ENABLE_WEB_SEARCH,
        DEFAULT_ENABLE_WEB_SEARCH,
    ),
}

TOOL_FAMILY_PROFILE_SETTINGS = {
    TOOL_FAMILY_DEVICE: (
        CONF_PROFILE_ENABLE_DEVICE_TOOLS,
        DEFAULT_PROFILE_ENABLE_DEVICE_TOOLS,
    ),
    TOOL_FAMILY_ASSIST_BRIDGE: (
        CONF_PROFILE_ENABLE_ASSIST_BRIDGE,
        DEFAULT_PROFILE_ENABLE_ASSIST_BRIDGE,
    ),
    TOOL_FAMILY_RESPONSE_SERVICE: (
        CONF_PROFILE_ENABLE_RESPONSE_SERVICE_TOOLS,
        DEFAULT_PROFILE_ENABLE_RESPONSE_SERVICE_TOOLS,
    ),
    TOOL_FAMILY_WEATHER_FORECAST: (
        CONF_PROFILE_ENABLE_WEATHER_FORECAST_TOOL,
        DEFAULT_PROFILE_ENABLE_WEATHER_FORECAST_TOOL,
    ),
    TOOL_FAMILY_RECORDER: (
        CONF_PROFILE_ENABLE_RECORDER_TOOLS,
        DEFAULT_PROFILE_ENABLE_RECORDER_TOOLS,
    ),
    TOOL_FAMILY_MEMORY: (
        CONF_PROFILE_ENABLE_MEMORY_TOOLS,
        DEFAULT_PROFILE_ENABLE_MEMORY_TOOLS,
    ),
    TOOL_FAMILY_CALCULATOR: (
        CONF_PROFILE_ENABLE_CALCULATOR_TOOLS,
        DEFAULT_PROFILE_ENABLE_CALCULATOR_TOOLS,
    ),
    TOOL_FAMILY_UNIT_CONVERSION: (
        CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS,
        DEFAULT_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS,
    ),
    TOOL_FAMILY_MUSIC_ASSISTANT: (
        CONF_PROFILE_ENABLE_MUSIC_ASSISTANT_SUPPORT,
        DEFAULT_PROFILE_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    ),
    TOOL_FAMILY_WEB_SEARCH: (
        CONF_PROFILE_ENABLE_WEB_SEARCH,
        DEFAULT_PROFILE_ENABLE_WEB_SEARCH,
    ),
}


def get_optional_tool_family(tool_name: str) -> str | None:
    """Return the optional tool family for a tool name, if any."""
    return OPTIONAL_TOOL_NAME_TO_FAMILY.get(tool_name)

# MCP Server settings
MCP_SERVER_NAME = "ha-entity-discovery"
MCP_PROTOCOL_VERSION = "2024-11-05"

# Entity discovery limits
MAX_ENTITIES_PER_DISCOVERY = 50  # Default, can be overridden in system settings
MAX_DISCOVERY_RESULTS = 100
CONF_MAX_ENTITIES_PER_DISCOVERY = "max_entities_per_discovery"
DEFAULT_MAX_ENTITIES_PER_DISCOVERY = 50

RESPONSE_MODE_INSTRUCTIONS = {
    "none": (
        "Follow-up behavior: do not ask follow-up questions. Complete the task and end."
    ),
    "default": (
        "Follow-up behavior: ask a short, specific follow-up only when it is "
        "genuinely helpful. If you ask one, call set_conversation_state. "
        "Otherwise end after completing the task."
    ),
    "always": (
        "Follow-up behavior: usually ask a short, specific follow-up after the "
        "task. If you ask one, call set_conversation_state. End naturally when "
        "the user indicates they are done."
    ),
}

DEFAULT_TECHNICAL_PROMPT = """You control a Home Assistant smart home through MCP tools.

Rules:
- Never invent entity IDs or claim an action happened unless a tool confirmed it.
- For Home Assistant tasks, discover the target first.
- Prefer entity-first control. Use device tools only when physical-device context matters.
- Floors, labels, and aliases are valid discovery inputs.
- Call get_index() only when you need a broad system overview.

Core workflow:
1. discover_entities(...) for most requests.
2. perform_action(...) for changes.
3. get_entity_details(...) when exact state or attributes matter.
4. run_script(...) for scripts with return data.
5. run_automation(...) for manual automation triggering.
6. Discover calendar or todo entities first, then use perform_action(...) for supported writes.

Responses:
- Keep replies short and plain text.
- Use friendly names, not raw entity IDs.
- For time answers, prefer relative time plus local absolute time when available.
- When listing multiple items, group by area when possible, otherwise use a stable order.

{response_mode}

{current_user_context}
Current area: {current_area}
{home_location_context}
Current time: {time}
Current date: {date}"""

DEVICE_TECHNICAL_INSTRUCTIONS = """
Device tools are enabled.
- Use discover_devices / get_device_details when the user means a physical device or when you need related entities on the same device.
- Still prefer discover_entities for most direct control because some entities have no device.
"""

RESPONSE_SERVICE_TECHNICAL_INSTRUCTIONS = """
Response-service tools are enabled.
- Use list_response_services / call_service_with_response for structured Home Assistant reads such as forecasts, calendar events, todo items, or integration-specific query data.
- Prefer native service responses over web search when the answer already exists in Home Assistant.
- Prefer get_weather_forecast for user weather questions. It discovers the weather entity, chooses a supported forecast type, and summarizes the result in one call.
- For weather forecasts, discover the weather entity first. If forecast_types are available, use a supported type and never assume daily is supported.
- For a general "tomorrow" forecast, prefer daily if supported, otherwise twice_daily, otherwise hourly, and summarize the supported result instead of asking again.
"""

RECORDER_ANALYSIS_TECHNICAL_INSTRUCTIONS = """
Recorder tools are enabled.
- Use get_entity_history(mode="last_event") for the latest matching event.
- Use analyze_entity_history for counts, durations, streaks, and numeric summaries.
- Use get_entity_state_at_time for point-in-time questions.
- Choose the entity domain that matches the question, such as lock for locked or contact/opening for open.
"""

MEMORY_TECHNICAL_INSTRUCTIONS = """
Memory tools are enabled.
- Use remember_memory only when the user explicitly asks you to remember something or confirms a preference should be stored.
- Use recall_memories for questions about previously stored facts or preferences.
- Use forget_memory when the user asks you to remove or update stored memory.
- Prefer a specific ttl_days only when the user gives a duration; otherwise let the server default apply.
"""

ASSIST_BRIDGE_TECHNICAL_INSTRUCTIONS = """
Assist bridge tools are enabled.
- Use list_assist_tools / call_assist_tool only as fallback or debugging.
- Prefer MCP Assist discovery and control tools first.
"""

CALCULATOR_TECHNICAL_INSTRUCTIONS = """
Calculator tools are enabled.
- Use calculator tools for exact arithmetic and compound expressions instead of mental math.
"""

UNIT_CONVERSION_TECHNICAL_INSTRUCTIONS = """
Unit conversion tools are enabled.
- Use convert_unit for temperatures, measurements, data sizes, rates, and other exact unit conversions.
"""

MUSIC_ASSISTANT_TECHNICAL_INSTRUCTIONS = """
Music Assistant support is enabled.
- Prefer play_music_assistant for playback, list_music_assistant_players for target discovery, search_music_assistant for finding music, get_music_assistant_library for library browsing, and get_music_assistant_queue for queue questions.
- Only target Music Assistant players, not arbitrary media_player entities.
- If no target is given and the current area is known, use area="{current_area}".
"""
