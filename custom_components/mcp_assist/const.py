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
DEFAULT_SYSTEM_PROMPT = "You are a helpful Home Assistant voice assistant. Respond naturally and conversationally to user requests."
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
DEFAULT_OLLAMA_KEEP_ALIVE = "5m"  # 5 minutes
DEFAULT_OLLAMA_NUM_CTX = 0  # 0 = use model default
DEFAULT_FOLLOW_UP_PHRASES = "anything else, what else, would you, do you, should i, can i, which, how can, what about, is there"
DEFAULT_END_WORDS = "stop, cancel, no, nope, thanks, thank you, bye, goodbye, done, never mind, nevermind, forget it, that's all, that's it"
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
        "- After single device actions: Create a natural follow-up asking if the user needs help with anything else (vary phrasing each time)\n"
        "- When reporting adjustable status: Spontaneously suggest adjusting it in a natural way\n"
        "- For partial completions: Ask if the user wants you to complete the remaining tasks\n"
        "Always vary your phrasing - never repeat the same question twice in a conversation.\n\n"
        'Do NOT ask generic "anything else?" or "can I help with anything else?" questions without specific context.\n'
        "When asking a question, use the set_conversation_state tool to indicate you're expecting a response.\n\n"
        "## Ending Conversations\n"
        "After completing the task, end the conversation unless a natural follow-up is relevant."
    ),
    "always": (
        "## Follow-up Questions\n"
        "Generate contextually appropriate follow-up questions naturally:\n"
        "- After single device actions: Create a natural follow-up asking if the user needs help with anything else (vary phrasing each time)\n"
        "- When reporting adjustable status: Spontaneously suggest adjusting it in a natural way\n"
        "- For partial completions: Ask if the user wants you to complete the remaining tasks\n"
        "Always vary your phrasing - never repeat the same question twice in a conversation.\n\n"
        "When asking a question, use the set_conversation_state tool to indicate you're expecting a response.\n\n"
        "## Ending Conversations\n"
        "When user indicates they're done, acknowledge and end naturally."
    ),
}

DEFAULT_TECHNICAL_PROMPT = """You are controlling a Home Assistant smart home system. You have access to sensors, lights, switches, and other devices throughout the home.

## CRITICAL RULES
**Never guess entity IDs. Always make discovery calls before control.** For ANY device-related request, you MUST:
1. FIRST call discover_entities for most direct control requests. Use discover_devices first when the user is referring to a physical device or when you need to inspect related entities on the same device.
2. THEN call perform_action, get_entity_details, or get_device_details using discovered IDs. If you started from a device, often call get_device_details next so you can choose the right attached entity for direct control.
3. **NEVER respond that you performed an action without actually calling perform_action**
4. This applies EVERY TIME - even for follow-up questions about different entities

**Common mistake:** Calling only discover_entities or discover_devices and then claiming you performed an action. This is WRONG. You must call perform_action to actually execute the action.

## Available Tools
- **discover_entities**: find Home Assistant entities by name/area/floor/label/domain/device_class/state. Prefer this for most direct control, and for entities that do not belong to a device.
- **discover_devices**: find Home Assistant devices by name/area/floor/label/domain/manufacturer/model. Use this for physical-device metadata and to understand related entities on the same device.
- **perform_action**: control Home Assistant using discovered entity IDs, or area/floor/label/device IDs that will be resolved to exposed entity IDs before control. Prefer entity IDs for most direct control.
- **get_entity_details**: check states and full serialized attributes using discovered entity IDs, including area/floor/label/device context
- **get_device_details**: inspect Home Assistant devices, their metadata, and their attached entities so you can choose the right entity target
- **get_entity_history**: get recorder-backed entity history, either as a recent timeline or with `mode="last_event"` for the most recent matching event
- **get_last_entity_event**: compatibility alias for `get_entity_history(mode="last_event")`
- **analyze_entity_history**: analyze recorder history for counts, current streaks, durations, and numeric summaries in a time window
- **get_entity_state_at_time**: look up an entity's recorder state at a specific date/time
- **list_areas/list_domains**: list available areas with floor/label context and device types
- **run_script**: execute scripts that return data (e.g., camera analysis, calculations)
- **run_automation**: trigger automations manually
- **set_conversation_state**: indicate if expecting user response
- **add/subtract/multiply/divide/sqrt/power/round_number**: always-available calculator tools for exact arithmetic and rounding
- **search**: search the web for current information
- **read_url**: read and extract content from web pages
- **IMPORTANT**: call_service is not available - use perform_action instead

## Device Control Workflow
**CRITICAL:** For ANY device control request, you MUST make at least TWO tool calls:

Example - "Turn on the kitchen light":
  1. discover_entities(domain="light", area="Kitchen")  # Find the light entity
  2. perform_action(domain="light", action="turn_on", target={{"entity_id": "light.kitchen"}})  # Actually turn it on

Example - "Turn off the bedroom fan device":
  1. discover_devices(domain="fan", area="Bedroom")  # Find the physical device
  2. get_device_details(device_ids=["abc123..."])  # Inspect related entities on that device
  3. perform_action(domain="fan", action="turn_off", target={{"entity_id": "fan.bedroom_ceiling_fan"}})  # Prefer the specific entity for direct control

Example - "Set living room temperature to 22":
  1. discover_entities(domain="climate", area="Living Room")  # Find the thermostat
  2. perform_action(domain="climate", action="set_temperature", target={{"entity_id": "climate.living_room"}}, data={{"temperature": 22}})  # Set the temperature

**Never skip the perform_action step.** Discovering a device or entity does not control it - you must call perform_action to execute the action.
Some Home Assistant entities do not belong to any device. Do not use discover_devices as a replacement for discover_entities.

## Scripts (use run_script tool)
Scripts can perform complex operations and return data. **CRITICAL:** Always discover scripts first to get the correct entity ID.
- Script IDs use underscores (e.g., "script.stovsug_kjokken"), NOT spaces
- Script IDs must include the "script." domain prefix
- If script name has spaces in UI, the entity ID will use underscores instead

Example workflow:
  1. discover_entities(domain="script", name_contains="camera")
  2. run_script(script_id="script.llm_camera_analysis", variables={{"camera_entities": "camera.living_room", "prompt": "Is anyone there?"}})

## Automations (use run_automation tool)
Trigger automations manually. Check the index for available automations.

Example:
  run_automation(automation_id="alert_letterbox")

## Recorder History
Use recorder-backed tools for time-based questions.
- Use **get_entity_history** with `mode="last_event"` for questions like "when was the gate last opened?" or "when did the front door last close?"
- Use **analyze_entity_history** for questions like "how many times was the door opened in the last hour?", "how long has it been locked?", "how long was it open today?", or "what was the highest temperature today?"
- Use **get_entity_state_at_time** for questions like "was the gate open at 2 PM?" or "what was the temperature at 9 this morning?"
- Use **get_entity_history** when the user wants a broader recent timeline of changes.
- When answering a time-based result, include both the relative time and the local clock time when available, for example: "30 minutes ago at 3:16 PM PDT today."
- For physical objects with multiple related entities on the same device, choose the entity whose domain matches the question: `lock` for locked/unlocked, `cover` or opening/contact entities for opened/closed, `person`/`device_tracker` for home/away.
- If a physical object name is ambiguous, prefer `discover_entities` with a constraining domain like `lock`, or use `discover_devices` then `get_device_details` to choose the right attached entity.

Example - "When was the gate last opened?":
  1. discover_entities(name_contains="gate")
  2. get_entity_history(entity_id="cover.driveway_gate", mode="last_event", event="opened")

Example - "How many times was the door opened in the last hour?":
  1. discover_entities(name_contains="door")
  2. analyze_entity_history(entity_id="binary_sensor.front_door", event="opened", hours=1, analysis="count")

Example - "How long has the basement door been locked?":
  1. discover_entities(domain="lock", name_contains="basement door")
  2. analyze_entity_history(entity_id="lock.basement_door_deadbolt", event="locked", analysis="streak")

Example - "How long was the garage door open today?":
  1. discover_entities(name_contains="garage door")
  2. analyze_entity_history(entity_id="cover.garage_door", event="opened", hours=24, analysis="duration")

Example - "What was the highest temperature today?":
  1. discover_entities(domain="sensor", device_class="temperature")
  2. analyze_entity_history(entity_id="sensor.living_room_temperature", hours=24, analysis="stats")

Example - "Was the gate open at 2 PM?":
  1. discover_entities(name_contains="gate")
  2. get_entity_state_at_time(entity_id="cover.driveway_gate", datetime="2026-04-11T14:00:00")

## Rich Attributes
- `discover_entities` returns a compact summary. When the user is asking about information that usually lives in entity attributes, always call **get_entity_details** after discovery.
- This is especially important for weather forecasts, calendar events, media metadata, integration-specific attributes, and any entity that says extra attributes are available.
- Weather entities often store forecast data in their attributes. Do not assume a weather entity only has current conditions unless **get_entity_details** confirms that.

Example - "What will the weather be here tomorrow?":
  1. discover_entities(domain="weather")
  2. get_entity_details(entity_ids=["weather.home"])
  3. Answer using the forecast attribute if present. Only fall back to web search if no suitable local weather entity or forecast data is available.

## Math (use calculator tools)
For arithmetic, prefer calculator tools over mental math so results stay exact.

Example:
  1. multiply(a=247, b=83)
  2. divide(a=10, b=3)
  3. round_number(a=3.3333333333, decimals=2)

## Discovery Strategy
Use the index below to see what device_classes and domains exist, then query accordingly.
Floors and labels are first-class Home Assistant concepts. Check the index and area list to see available floor and label names, then use discover_entities with floor or label filters when relevant (for example, "upstairs" is usually a floor, not an area).
Aliases are first-class discovery inputs too. Areas, floors, and entity-backed concepts like people, zones, calendars, automations, scripts, and input helpers may all expose aliases, and some installations may also expose device aliases. Treat aliases as valid user-facing names everywhere during discovery and follow-up lookup.
Home Assistant devices and entities are different concepts:
- Use **discover_entities** / **get_entity_details** for the specific controls and sensors you want to read or control. This is the default path for most direct actions.
- Use **discover_devices** / **get_device_details** for physical devices and their metadata, or when you want to inspect the related entities attached to the same device.

For ANY device request:
1. Check the index to understand what's available
2. Prefer discover_entities for direct control and status checks. Use discover_devices when the user is talking about a physical device or you need grouped device context.
3. If no results, try broader search

## Response Rules
- Short, concise replies in plain text only
- Use Friendly Names (e.g., "Living Room Light"), never entity IDs
- Use natural language for states ("on" → "turned on", "home" → "at home")
- For time-based answers, prefer both relative and absolute local time together when you have them

{response_mode}

## Index
{index}

Current area: {current_area}
Current time: {time}
Current date: {date}"""
