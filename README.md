# MCP Assist for Home Assistant

A Home Assistant conversation agent that uses MCP (Model Context Protocol) for efficient entity discovery, achieving **95% token reduction** compared to traditional methods. Works with LM Studio, llama.cpp, Ollama, OpenAI, Google Gemini, Anthropic Claude, OpenRouter, Moltbot, and vLLM.

## Key Features

- ✅ **95% Token Reduction**: Uses MCP tools for dynamic entity discovery instead of sending all entities
- ✅ **No Entity Dumps**: Never sends 12,000+ token entity lists to the LLM
- ✅ **Smart Entity Index**: Pre-generated system structure index (~400-800 tokens) for context-aware queries
- ✅ **Multi-Platform Support**: Works with LM Studio, llama.cpp, Ollama, OpenAI, Google Gemini, Anthropic Claude, OpenRouter, Moltbot, and vLLM
- ✅ **Multilingual Support**: 21 languages with localized UI, system prompts, and speech detection
- ✅ **Multi-turn Conversations**: Maintains conversation context and history
- ✅ **Dynamic Discovery**: Finds entities and devices by area, floor, label, domain, device_class, state, or alias-aware name search on-demand
- ✅ **Native History & Service Reads**: Uses recorder-backed history plus Home Assistant response-capable services for things like event history, forecasts, calendars, and to-do queries
- ✅ **Calculator Tools**: Built-in arithmetic, unit conversion, and expression-evaluation tools for exact math through tool calling
- ✅ **Web Search Tools**: Optional DuckDuckGo or Brave Search integration for current information
- ✅ **Works with 1000+ Entities**: Efficient even with large Home Assistant installations
- ✅ **Multi-Profile Support**: Run multiple conversation agents with different models

## The Problem MCP Assist Solves

Traditional voice assistants send your **entire entity list** (lights, switches, sensors, etc.) to the LLM with every request. For a typical home with 200+ devices, this means:
- **12,000+ tokens** sent every time
- Expensive API costs (cloud LLMs)
- Slow response times
- Context window limitations
- Poor performance with large homes

## How MCP Assist Works

Instead of dumping all entities, MCP Assist:

1. **Starts an MCP Server** on Home Assistant that exposes entity discovery tools
2. **Your LLM connects** to the MCP server and gets access to these tools:
   - `get_index` - Get system structure index (areas, floors, labels, devices, domains, device_classes, people, etc.) including aliases for alias-capable objects
   - `discover_entities` - Find entities by type, area, floor, label, domain, device_class, state, or alias-aware name search. This is the preferred path for most direct control
   - `discover_devices` - Find Home Assistant devices by area, floor, label, domain, name, or alias-aware search when you want physical-device context or related entities on the same device
   - `get_entity_details` - Get current entity state and full serialized attributes
   - `get_device_details` - Get device metadata and attached entities so you can choose the right entity target
   - `perform_action` - Control Home Assistant entities and perform supported write actions
   - `list_response_services` / `call_service_with_response` - Discover and call Home Assistant services that return structured response data for things like weather forecasts, calendar event reads, to-do list queries, and other native reads
   - `get_entity_history` / `analyze_entity_history` / `get_entity_state_at_time` - Query recorder history for last events, counts, streaks, durations, numeric summaries, and point-in-time answers
   - `list_assist_tools` / `call_assist_tool` / `get_assist_prompt` / `get_assist_context_snapshot` - Optional native Assist bridge tools for compatibility and debugging
   - Calculator tools - Optional exact arithmetic, unit conversion, and expression evaluation through tool calling
   - `run_script` - Execute scripts and return response data
   - `run_automation` - Trigger automations manually
   - `list_areas` - List all areas in your home with entity/device counts
   - `list_domains` - List all entity types
   - `set_conversation_state` - Smart follow-up handling
3. **LLM uses the index for smart queries** - Understands what exists without full context dump
4. **LLM discovers on-demand** - Only fetches the entities it needs for each request
5. **Token usage drops** from 12,000+ to ~400 tokens per request

The custom MCP Assist tools remain the primary path for precise Home Assistant work. The native Assist bridge tools are complementary: they let the model inspect or call the built-in Home Assistant Assist tool surface when compatibility, debugging, or native Assist behavior is useful.

## Token Usage Comparison

| Method | Token Usage | Description |
|--------|-------------|-------------|
| **Traditional** | 12,000+ tokens | Sends all entity states |
| **MCP Assist** | ~400 tokens | Uses MCP tools for discovery |
| **Reduction** | **95%** | Massive efficiency gain |

## Smart Entity Index (v0.5.0+)

The Smart Entity Index provides a lightweight (~400-800 tokens) snapshot of your Home Assistant system structure, enabling context-aware queries without full entity dumps. The index includes areas, floors, labels, domains, device classes, devices, people, calendars, zones, automations, scripts, and alias metadata for alias-capable Home Assistant objects. For entities without standardized device_class attributes (like custom integrations), LLM-powered gap-filling automatically infers semantic categories from naming patterns. This results in faster, more accurate queries that use ~95% fewer tokens compared to traditional entity dumps.

## Multilingual Support (v0.12.0+)

MCP Assist supports **21 languages** with localized configuration interfaces, language-aware system prompts, and region-specific speech detection patterns. The integration automatically detects your Home Assistant system language and provides appropriate defaults for system prompts, follow-up phrases, and end conversation words. Supported languages include: Arabic, Chinese (Simplified), Czech, Danish, Dutch, Finnish, Filipino, French, German, Greek, Hindi, Italian, Japanese, Korean, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, and Turkish.

## Requirements

- Home Assistant 2024.1+
- One of:
  - **Local LLMs**: LM Studio v0.3.17+, llama.cpp, Ollama, Moltbot, or vLLM
  - **Cloud LLMs**: OpenAI, Google Gemini, Anthropic Claude, or OpenRouter (API key required)
- Python 3.11+

## Installation

### Add to HACS

[![Open your Home Assistant instance and add this repository to HACS.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=mike-nott&repository=mcp-assist&category=integration)

### Option A: HACS (Recommended)
1. Click the badge above to add this repository to HACS, or manually add it as a custom repository
2. Install "MCP Assist" from HACS
3. Restart Home Assistant

### Option B: Manual Installation
1. Copy the `custom_components/mcp_assist` folder to your Home Assistant `custom_components` directory
2. Restart Home Assistant

## Configuration

### 1. Add the Integration

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for "MCP Assist" and select it

### 2. Setup Flow

**Step 1 - Profile & Server Type:**
- Profile Name: Give your assistant a name (e.g., "Living Room Assistant")
- Server Type: Choose your LLM provider
  - **LM Studio** - Local, free, runs on your machine
  - **llama.cpp** - Local, free, official llama.cpp server
  - **Ollama** - Local, free, command-line based
  - **OpenAI** - Cloud, paid
  - **Google Gemini** - Cloud, paid/free tier
  - **Anthropic Claude** - Cloud, paid
  - **OpenRouter** - Cloud, multi-model gateway with access to many models
  - **Moltbot** - Local or self-hosted server with its own model/session handling
  - **vLLM** - Local or self-hosted OpenAI-compatible inference server

**Step 2 - Server Configuration:**

*For Local/OpenAI-Compatible Servers (LM Studio / llama.cpp / Ollama / Moltbot / vLLM):*
- Server URL: Where your LLM server is running
  - LM Studio: `http://localhost:1234` (default)
  - llama.cpp: `http://localhost:8080` (default)
  - Ollama: `http://localhost:11434` (default)
  - Moltbot: `http://localhost:18789` (default)
  - vLLM: `http://localhost:8000` (default)

*For Cloud Providers (OpenAI / Gemini / Anthropic / OpenRouter):*
- API Key: Your provider API key (see below for setup)

**Step 3 - Model & Prompts:**
- Model Name: Select from auto-loaded models or enter manually
- System Prompt: Prefilled with the current built-in prompt so you can review, copy, or customize it
- Technical Instructions: Prefilled with the current built-in instructions so you can review, copy, or customize them

If you leave the prompt text effectively unchanged, MCP Assist continues using the built-in prompt from the integration code.

**Step 4 - Conversation, Tools, and Advanced Settings:**
- Conversation settings:
  - Response Mode: None / Smart / Always (conversation continuation behavior)
  - Follow-up Phrases: Configurable phrases for pattern detection
  - End Conversation Words: Words/phrases that end conversations
  - Control Home Assistant: Enable/disable device control
  - Clean Responses: Optional cleanup for voice-friendly output
- Tools section:
  - Tool Family Overrides: Optionally narrow this profile to a smaller subset of the shared MCP server's optional tools
  - Leave blank to inherit the shared MCP server tool families
- Advanced section:
  - Temperature: Response randomness (0.0-1.0)
  - Max Response Tokens: Maximum length of responses
  - Max History Messages: Conversation memory depth
  - Max Tool Iterations: How many tool calls are allowed per request
  - Timeout: Maximum wait time for provider responses
  - Debug Mode: Extra logging for troubleshooting
  - **Ollama Keep Alive** (Ollama only): Control how long models stay loaded in memory
    - `-1` = Keep loaded indefinitely
    - `0` = Unload immediately after response
    - `"5m"` = Keep for 5 minutes (default)
    - Duration strings like `"24h"`, `"168h"` also supported
  - **Ollama Context Window** (Ollama only): Custom context window size (`0` = use model default)

**Step 5 - Shared MCP Server Settings** (created with the first profile, editable later):  
These settings affect all profiles because they share one MCP server.

- Top-level shared settings:
  - MCP Server Port
  - Additional Allowed IPs/Ranges
- Discovery section:
  - Enable Smart Entity Index
  - Max Entities Per Discovery
- Tools section:
  - Web Search Provider: None, DuckDuckGo, or Brave Search
  - Brave Search API Key
  - Shared optional tool families such as device tools, response-service tools, recorder tools, Assist bridge tools, calculator tools, and Music Assistant tools

### 3. Set as Voice Assistant

1. In Home Assistant, go to **Settings** → **Voice Assistants**
2. Set your preferred assistant to your MCP Assist profile name
3. Test with commands!

## Usage Examples

### Basic Commands
- "Turn on the kitchen lights"
- "Turn off all the lights in the bedroom"
- "What's the temperature in the living room?"
- "Are any lights on upstairs?"

### Multi-Turn Conversations
- **User**: "What lights are on?"
- **Assistant**: "The kitchen and living room lights are on."
- **User**: "Turn off the kitchen one"
- **Assistant**: "I've turned off the kitchen light."

### Complex Query Example

**User**: "Do we have a leak?"

**Behind the scenes:**
```
1. LLM calls get_index → Sees moisture sensors and water flow monitors exist in system
2. LLM calls discover_entities(device_class="moisture")
   → Returns: binary_sensor.bathroom_leak, binary_sensor.kitchen_sink_leak, binary_sensor.laundry_leak
3. LLM calls discover_entities(name_contains="water flow")
   → Returns: sensor.water_flow_rate
4. LLM calls get_entity_details for each sensor
   → bathroom leak: "off", kitchen leak: "off", laundry leak: "on", water flow: "2.5 gpm"
5. LLM synthesizes response
```

**Assistant**: "Yes, the laundry room leak sensor is detecting water and water is flowing at 2.5 gallons per minute. The bathroom and kitchen sensors are dry."

**Follow-up User**: "Turn off the water main"

**Behind the scenes:**
```
1. LLM calls discover_entities(name_contains="water main")
   → Returns: switch.water_main_shutoff
2. LLM calls perform_action(entity_id="switch.water_main_shutoff", action="turn_off")
   → Success
```

**Assistant**: "I've shut off the main water valve."

### History & Native Read Examples
- "When was the front door last opened?"
- "How many times was the garage door opened in the last hour?"
- "How long has the basement door deadbolt been locked?"
- "What will the weather be here tomorrow?"

### Web Search (if enabled)
- "Search for the latest Home Assistant updates"
- "What time does the store close?"

For weather, calendar, and to-do queries, the assistant should prefer Home Assistant's native response-capable services before falling back to web search. Web search is most useful when the answer is genuinely external or current information outside Home Assistant.

## Configuration Options

### Profile Settings
- **Profile Name**: Unique name for this assistant
- **Server Type**: LM Studio, llama.cpp, Ollama, OpenAI, Gemini, Anthropic Claude, OpenRouter, Moltbot, or vLLM
- **Server URL / API Key**: How the selected provider is reached
- **Model Name**: Which model to use

### Prompts
- **System Prompt**: Prefilled with the built-in prompt so you can review, copy, or customize it
- **Technical Instructions**: Prefilled with the built-in tool-usage instructions so you can review, copy, or customize them

### Conversation & Tools
- **Response Mode**:
  - **None**: Never ask follow-ups, end immediately
  - **Smart** (default): Contextual follow-ups when relevant, user can end with "bye"/"thanks"
  - **Always**: Natural conversational follow-ups, user can end with "bye"/"thanks"
- **Follow-up Phrases**: Comma-separated phrases for detecting when the assistant wants to continue
- **End Conversation Words**: Comma-separated words/phrases for user-initiated ending
- **Control Home Assistant**: Enable or disable device control
- **Tool Family Overrides**: Optional per-profile narrowing of the shared MCP server tool families

### Advanced Settings
- **Temperature**: Limit or increase randomness (default depends on provider)
- **Max Response Tokens**: Limit response length
- **Max History Messages**: How many conversation turns to remember
- **Max Tool Iterations**: Prevent infinite loops
- **Timeout**: Provider response timeout
- **Debug Mode**: Extra logging for troubleshooting
- **Ollama Keep Alive** / **Ollama Context Window**: Ollama-specific tuning

### MCP Server Settings
- **MCP Server Port**: Default 8090 (change if port conflict)
- **Additional Allowed IPs/Ranges**: Whitelist Docker containers (e.g., `172.30.0.0/16`) or specific IPs for external MCP clients like Claude Code add-on
- **Enable Smart Entity Index**: Context-aware entity discovery with automatic gap-filling for uncommon devices (default: enabled)
- **Max Entities Per Discovery**: Cap how many entities a single discovery call may return
- **Tools**: Shared optional capabilities exposed by the MCP server, including web search and optional tool families

### Web Search
- Configured in the shared **Tools** section of the MCP server settings
- **Web Search Provider**: Choose between:
  - **None**: Search disabled
  - **DuckDuckGo**: Free web search (no API key required)
  - **Brave Search**: Requires API key from https://brave.com/search/api/
- **Brave Search API Key**: Required only if using Brave Search

### Shared vs Per-Profile Settings

MCP Assist has two types of settings:

**Per-Profile Settings** (independent per conversation agent):
- Model name and prompt overrides
- Conversation behavior and response mode
- Temperature, max tokens, timeout, and other advanced tuning
- Debug mode and max iterations
- Server URL or API key
- Optional per-profile tool family overrides

**Shared Settings** (affect ALL profiles):
- MCP server port
- Web search provider and Brave API key
- Allowed IPs/CIDR ranges
- Smart entity index settings
- Max entities per discovery
- Shared optional tool families exposed by the MCP server

When you change shared settings in one profile's options, they apply to all profiles. This is intentional since all profiles share the same MCP server.

## Model Compatibility Guide

Not all LLM models support tool calling (function calling) equally well. **This integration works best with frontier models** (GPT-5.2, Claude Opus 4.5, Gemini 3 Flash) **or higher-spec local models**. Smaller models may struggle with complex multi-entity queries that require synthesizing large tool result sets.  

### Understanding Tool Calling Requirements

Tool calling (function calling) requires the model to:
1. Understand the user's request
2. Decide which tool to call
3. Format the tool arguments correctly as JSON
4. Interpret the tool results
5. Generate a natural response

**Factors affecting tool calling success**:
- **Model size**: Larger models (8B+) generally handle tool calling better
- **Model architecture**: Vision-Language (VL) models behave differently than standard models
- **Inference engine**: LM Studio and Ollama optimize models differently
- **Quantization level**: Q4 vs Q8 can affect instruction following

### Instruct vs Thinking/Reasoning Models

**Instruct Models** (e.g., `qwen3-8b-instruct`):
- Fast response times
- Best for simple, single-action requests ("turn on the kitchen lights")
- May struggle with complex queries requiring multiple tool calls
- Good for basic voice commands

**Thinking/Reasoning Models** (e.g., `qwen3-8b-thinking`):
- Slower response times (more deliberate reasoning)
- **Much better at complex requests** requiring multiple tool calls and context
- Handles multi-step queries reliably ("check all rooms for open windows, then turn off lights in those rooms")
- **Recommended for Home Assistant** where queries often involve discovery + action combinations

Choose the model type that best fits your use case. Thinking/reasoning models offer better reliability with complex multi-tool queries, while instruct models provide faster responses for simple commands.

### Recommended Models

**Consistently Reliable**:
- ✅ **Qwen3 VL 32B Instruct** - Excellent tool calling
- ✅ **Qwen3 30B A3B Instruct** - Very good tool calling
- ✅ **Qwen3 8B Instruct** - Good balance, works reliably
- ✅ **Anthropic Opus 4.5** - The very best at tool calling (cloud)
- ✅ **OpenAI GPT-5.2** - Excellent tool calling, very fast (cloud)
- ✅ **Google Gemini 3 Flash** - Excellent tool calling, fast, cost-effective (cloud)

### Testing Your Model

When tool calling **doesn't work**, you'll see:
- Model claims "I turned on the lights" but nothing happens
- No `perform_action` tool calls in the logs
- Actions don't execute, only narration

When tool calling **works correctly**, you'll see in logs:
- `discover_entities` called to find devices
- `perform_action` called to control them
- "✅ Successfully executed" messages
- Devices actually change state
- Tool calls visible in Settings → Voice Assistants → (profile) → Debug

### General Guidelines

**Start with larger models** (30B) if your hardware supports it - they work consistently across platforms.

**If using smaller models** (4B-8B), test thoroughly:
- Try a simple command like "turn on the kitchen lights"
- Check logs to verify tools are being called
- Confirm the device actually changes state
- If it doesn't work, try the same model on a different platform (LM Studio vs Ollama)

**Vision-Language (VL) models** are optimized for multimodal tasks and may have different tool calling behavior than standard models.

### Dynamic Model Switching

One of MCP Assist's features is **dynamic model switching** - you can change models in the configuration UI and it takes effect immediately without restarting Home Assistant. This makes it easy to:
- Test different models
- Switch between fast (Q4) and quality (Q8) quantizations
- Try new models as they're released

## Troubleshooting

### Integration Won't Start
- Check that the MCP port isn't already in use
- Verify Home Assistant has permission to bind to the port
- Check the Home Assistant logs for specific error messages

### LM Studio Can't Connect to MCP
- Ensure the MCP server is running (check integration status)
- Verify the MCP configuration in LM Studio is correct
- Check that the URL in LM Studio matches your MCP port (default: 8090)
- Restart LM Studio after changing MCP configuration

### Ollama Connection Issues
- Verify Ollama is running: `ollama list`
- Check the URL matches where Ollama is running (default: `http://localhost:11434`)
- Ensure the model is loaded in Ollama

### Cloud Provider Connection Issues
- Verify your API key is valid with your provider
- Check you have sufficient credits/quota remaining
- Check for rate limit errors in Home Assistant logs
- Try regenerating your API key if authentication fails
- Verify your internet connection is working
- Check Home Assistant logs for specific error codes

### No Response from Assistant
- Verify your LLM has a model loaded
- Check that the model name in the integration matches exactly
- Ensure your entities are exposed to the conversation assistant
- Check Home Assistant logs for API errors
- Enable Debug Mode for more detailed logging

### Poor Response Quality
- Try a different/larger model
- Adjust the temperature setting (lower = more focused)
- Ensure the model supports tool calling/function calling
- Check that Technical Instructions are not modified

### Tools Not Working
- Verify "Control Home Assistant" is enabled
- Check that entities are exposed (Settings → Voice Assistants → Expose)
- Look for MCP server errors in logs
- Ensure Max Tool Iterations isn't set too low

### History, Forecast, or Calendar Questions Feel Incomplete
- Make sure the relevant optional tool families are enabled on the shared MCP server
- Recorder-backed questions depend on Home Assistant recorder data being available
- Weather, calendar, and to-do reads work best through Home Assistant's native response-capable services

## Entity Exposure

The integration only discovers entities that are exposed to the "conversation" assistant. To expose entities:

1. Go to **Settings** → **Voice Assistants** → **Expose**
2. Select entities you want the assistant to control
3. The integration will automatically discover these when needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/mike-nott/mcp-assist/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mike-nott/mcp-assist/discussions)
- **Home Assistant Community**: [Community Forum](https://community.home-assistant.io/)
