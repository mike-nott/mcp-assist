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
CONF_BRAVE_API_KEY = "brave_api_key"
CONF_ALLOWED_IPS = "allowed_ips"
CONF_SEARCH_PROVIDER = "search_provider"
CONF_ENABLE_GAP_FILLING = "enable_gap_filling"
CONF_ENABLE_ASSIST_BRIDGE = "enable_assist_bridge"
CONF_ENABLE_RESPONSE_SERVICE_TOOLS = "enable_response_service_tools"
CONF_ENABLE_RECORDER_TOOLS = "enable_recorder_tools"
CONF_ENABLE_CALCULATOR_TOOLS = "enable_calculator_tools"
CONF_ENABLE_DEVICE_TOOLS = "enable_device_tools"
CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT = "enable_music_assistant_support"
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
DEFAULT_BRAVE_API_KEY = ""
DEFAULT_ALLOWED_IPS = ""
DEFAULT_SEARCH_PROVIDER = "none"
DEFAULT_ENABLE_GAP_FILLING = True
DEFAULT_ENABLE_ASSIST_BRIDGE = False
DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS = True
DEFAULT_ENABLE_RECORDER_TOOLS = True
DEFAULT_ENABLE_CALCULATOR_TOOLS = False
DEFAULT_ENABLE_DEVICE_TOOLS = True
DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT = False
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
        "## Follow-up Questions\n"
        "Do NOT ask follow-up questions. Complete the task and end immediately.\n\n"
        "## Ending Conversations\n"
        "Always end after completing the task."
    ),
    "default": (
        "## Follow-up Questions\n"
        "Generate contextually appropriate follow-up questions naturally:\n"
        "- After single device actions: Create a natural follow-up asking if the "
        "user needs help with anything else (vary phrasing each time)\n"
        "- When reporting adjustable status: Spontaneously suggest adjusting it "
        "in a natural way\n"
        "- For partial completions: Ask if the user wants you to complete the "
        "remaining tasks\n"
        "Always vary your phrasing - never repeat the same question twice in a "
        "conversation.\n\n"
        'Do NOT ask generic "anything else?" or "can I help with anything else?" '
        "questions without specific context.\n"
        "When asking a question, use the set_conversation_state tool to indicate "
        "you're expecting a response.\n\n"
        "## Ending Conversations\n"
        "After completing the task, end the conversation unless a natural "
        "follow-up is relevant."
    ),
    "always": (
        "## Follow-up Questions\n"
        "Generate contextually appropriate follow-up questions naturally:\n"
        "- After single device actions: Create a natural follow-up asking if the "
        "user needs help with anything else (vary phrasing each time)\n"
        "- When reporting adjustable status: Spontaneously suggest adjusting it "
        "in a natural way\n"
        "- For partial completions: Ask if the user wants you to complete the "
        "remaining tasks\n"
        "Always vary your phrasing - never repeat the same question twice in a "
        "conversation.\n\n"
        "When asking a question, use the set_conversation_state tool to indicate "
        "you're expecting a response.\n\n"
        "## Ending Conversations\n"
        "When user indicates they're done, acknowledge and end naturally."
    ),
}

DEFAULT_TECHNICAL_PROMPT = """You are controlling a Home Assistant smart home system. You have access to sensors, lights, switches, and other devices throughout the home.

## CRITICAL RULES
**Never guess entity IDs. Always make discovery calls before control.** For any Home Assistant request:
1. First call **discover_entities** to find the correct target.
2. Then call **perform_action**, **get_entity_details**, or **get_entity_history** using the discovered entity ID.
3. **Never claim an action happened unless you actually called perform_action.**
4. This applies on follow-up turns too. Discover again whenever the target changed or is ambiguous.

**Common mistake:** calling only discover_entities and then claiming you turned something on or off. Discovery does not execute actions.

## Core Tools
- **discover_entities**: the default path for finding entities by name, area, floor, label, domain, device_class, state, or alias
- **perform_action**: control Home Assistant using discovered entity IDs and supported write actions
- **get_entity_details**: read current state plus full serialized attributes
- **get_entity_history**: answer recorder-backed history questions, including `mode="last_event"` for the latest matching event
- **list_areas/list_domains**: inspect the available Home Assistant structure
- **run_script**: execute scripts that return response data
- **run_automation**: manually trigger automations
- **set_conversation_state**: mark whether you expect the user to respond
- **search** and **read_url**: optional web lookups for current external information
- **IMPORTANT**: `call_service` is not available. Use **perform_action** for control and supported write operations.

## Control Workflow
For direct control, status checks, and most follow-up requests:
1. Use **discover_entities** first.
2. Use **perform_action** for changes or **get_entity_details** for state and attributes.
3. Prefer entity IDs for control, because some Home Assistant entities do not belong to any device.

Example:
  1. discover_entities(domain="light", area="Kitchen")
  2. perform_action(domain="light", action="turn_on", target={{"entity_id": "ENTITY_ID_FROM_DISCOVERY"}})

## Scripts and Automations
- Always discover scripts first so you use the real `script.` entity ID.
- Use **run_script** for scripts that return data.
- Use **run_automation** for manual automation triggering.

## Calendars and To-do Lists
- Calendars and to-do lists are not read-only. Discover the right entity first, then use **perform_action** for write operations such as `calendar.create_event` or `todo.add_item`.

## Discovery Strategy
Use the index below to understand what exists before you narrow further.
- Floors and labels are first-class Home Assistant concepts. "Upstairs" is often a floor, not an area.
- Aliases are valid discovery inputs. Areas, floors, entities, and some devices may expose aliases.
- `discover_entities` returns a compact summary. If the answer depends on attributes, follow it with **get_entity_details**.

## Response Rules
- Short, concise replies in plain text only
- Use friendly names, not raw entity IDs
- Use natural language for states
- For time-based answers, prefer both relative and local absolute time together when available

{response_mode}

## Index
{index}

Current area: {current_area}
Current time: {time}
Current date: {date}"""

DEVICE_TECHNICAL_INSTRUCTIONS = """
## Device Context
Device tools are enabled.
- Use **discover_devices** when the user is referring to a physical device or when you need grouped metadata across related entities on the same device.
- Use **get_device_details** after device discovery to inspect attached entities and then choose the specific entity for direct control.
- Do not replace entity-first control with device-first control. Prefer **discover_entities** for most actions because some entities have no device.

Example - "Turn off the bedroom fan device":
  1. discover_devices(domain="fan", area="Bedroom")
  2. get_device_details(device_ids=["DEVICE_ID_FROM_DISCOVERY"])
  3. perform_action(domain="fan", action="turn_off", target={{"entity_id": "ENTITY_ID_FROM_DEVICE_DETAILS"}})
"""

RESPONSE_SERVICE_TECHNICAL_INSTRUCTIONS = """
## Native Response Services
Structured response-returning service tools are enabled.
- Use **list_response_services** to discover live Home Assistant services that return structured data.
- Use **call_service_with_response** for native read and query workflows such as forecasts, calendar event reads, todo list reads, media browsing, and other integration-specific response data.
- Prefer native service responses for rich read queries when available, and use **get_entity_details** as the fallback.

Example - "What will the weather be here tomorrow?":
  1. discover_entities(domain="weather")
  2. call_service_with_response(domain="weather", service="get_forecasts", target={{"entity_id": ["WEATHER_ENTITY_ID_FROM_DISCOVERY"]}}, data={{"type": "daily"}})
"""

RECORDER_ANALYSIS_TECHNICAL_INSTRUCTIONS = """
## Advanced Recorder Analysis
Advanced recorder analysis tools are enabled.
- Use **get_entity_history** with `mode="last_event"` for questions like "when was the gate last opened?"
- Use **analyze_entity_history** for counts, durations, streaks, and numeric summaries.
- Use **get_entity_state_at_time** for point-in-time questions like "was it open at 2 PM?"
- When a physical object has multiple related entities, choose the entity whose domain matches the question: `lock` for locked or unlocked, opening or contact entities for open or closed, and so on.

Example - "How many times was the door opened in the last hour?":
  1. discover_entities(name_contains="door")
  2. analyze_entity_history(entity_id="ENTITY_ID_FROM_DISCOVERY", event="opened", hours=1, analysis="count")
"""

ASSIST_BRIDGE_TECHNICAL_INSTRUCTIONS = """
## Native Assist Bridge
Native Assist bridge tools are enabled.
- Use **list_assist_tools** and **call_assist_tool** only as a compatibility or fallback path when Home Assistant's built-in Assist behavior is preferable.
- Prefer MCP Assist's structured discovery and control tools first for precision and predictable targeting.
- Use **get_assist_context_snapshot** for a concise native Assist snapshot and **get_assist_prompt** mainly for debugging.
"""

CALCULATOR_TECHNICAL_INSTRUCTIONS = """
## Math and Unit Conversion
Calculator tools are enabled.
- Use calculator tools for exact arithmetic instead of mental math.
- Use **convert_unit** for unit conversion, including °C ↔ °F and common Home Assistant units.
- Use **evaluate_expression** when the user asks for compound calculations.
"""

MUSIC_ASSISTANT_TECHNICAL_INSTRUCTIONS = """
## Music Assistant
Music Assistant support is enabled.
- Prefer **play_music_assistant** for voice-driven music playback instead of generic media_player control.
- Use **list_music_assistant_players** when you need to inspect or disambiguate Music Assistant playback targets.
- Use **search_music_assistant** for provider-aware discovery and **get_music_assistant_library** for library browsing when the user wants to find music, playlists, artists, albums, or stations before playing them.
- Use **get_music_assistant_queue** to answer "what is playing" or "what is queued next" questions for Music Assistant players.
- Use **list_music_assistant_instances** if multiple Music Assistant servers are configured and you need to choose one for discovery.
- Only target Music Assistant media_player entities, not arbitrary media_player entities.
- If the user names an area, floor, label, or specific player, pass that to **play_music_assistant**.
- If no explicit target is given and the current area is known, use `area="{current_area}"`.
- Prefer the dedicated Music Assistant wrappers for discovery and queue reads. Use generic response-service tools only as an advanced fallback if they are enabled and you specifically need a native service that does not already have a dedicated wrapper.

Example - "Play Queen upstairs":
  1. list_music_assistant_players(floor="Upstairs")
  2. play_music_assistant(media_type="artist", media_id="Queen", artist="Queen", floor="Upstairs")

Example - "Find some Miles Davis albums":
  1. search_music_assistant(name="Miles Davis", media_type=["album", "artist"])
  2. Optionally call get_music_assistant_library(...) or play_music_assistant(...) based on the result

Example - "Shuffle jazz here":
  1. play_music_assistant(media_type="track", media_id="jazz", area="{current_area}", shuffle=true)

Example - "What's queued in the kitchen?":
  1. get_music_assistant_queue(area="Kitchen")
"""
