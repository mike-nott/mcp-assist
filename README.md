# MCP Assist for Home Assistant

A Home Assistant conversation agent that uses MCP (Model Context Protocol) for efficient entity discovery, achieving **95% token reduction** compared to traditional methods. Works with LM Studio, llama.cpp, Ollama, OpenAI, Google Gemini, Anthropic Claude, and OpenRouter.

## Key Features

- ✅ **95% Token Reduction**: Uses MCP tools for dynamic entity discovery instead of sending all entities
- ✅ **Fast Path for Simple Commands**: Instant responses (~50ms) for common commands without calling the LLM
- ✅ **Instant Entity Recognition**: Pre-resolves device names before the LLM call - faster responses, fewer API calls
- ✅ **No Entity Dumps**: Never sends 12,000+ token entity lists to the LLM
- ✅ **Smart Entity Index**: Pre-generated system structure index (~400-800 tokens) for context-aware queries
- ✅ **Multi-Platform Support**: Works with LM Studio, llama.cpp, Ollama, OpenAI, Google Gemini, Anthropic Claude, and OpenRouter
- ✅ **Multi-turn Conversations**: Maintains conversation context and history
- ✅ **Dynamic Discovery**: Finds entities by area, type, device_class, state, or name on-demand
- ✅ **Web Search Tools**: Optional DuckDuckGo or Brave Search integration for current information
- ✅ **Works with 1000+ Entities**: Efficient even with large Home Assistant installations
- ✅ **Multi-Profile Support**: Run multiple conversation agents with different models
- ✅ **Multi-Language**: Fast Path supports DE, EN, FR, ES, NL with custom keywords

## The Problem MCP Assist Solves

Traditional voice assistants send your **entire entity list** (lights, switches, sensors, etc.) to the LLM with every request. For a typical home with 200+ devices, this means:
- **12,000+ tokens** sent every time
- Expensive API costs (cloud LLMs)
- Slow response times
- Context window limitations
- Poor performance with large homes

## How MCP Assist Works

Instead of dumping all entities, MCP Assist:

1. **Pre-resolves entity names** - When you say "kitchen light", MCP Assist instantly knows you mean `light.kitchen` before even calling the LLM
2. **Starts an MCP Server** on Home Assistant that exposes entity discovery tools
3. **Your LLM connects** to the MCP server and gets access to these tools:
   - `get_index` - Get system structure index (areas, domains, device_classes, people, etc.)
   - `discover_entities` - Find entities by type, area, domain, device_class, or state (fallback when pre-resolution doesn't match)
   - `get_entity_details` - Get current state and attributes
   - `perform_action` - Control devices
   - `run_script` - Execute scripts and return response data
   - `run_automation` - Trigger automations manually
   - `list_areas` - List all areas in your home
   - `list_domains` - List all entity types
   - `set_conversation_state` - Smart follow-up handling
4. **LLM uses pre-resolved entities directly** - For common requests, no discovery call needed
5. **Falls back to smart discovery** - When pre-resolution doesn't find a match, uses the index and discovery tools
6. **Token usage drops** from 12,000+ to ~400 tokens per request

## Token Usage Comparison

| Method | Token Usage | Description |
|--------|-------------|-------------|
| **Traditional** | 12,000+ tokens | Sends all entity states |
| **MCP Assist** | ~400 tokens | Uses MCP tools for discovery |
| **Reduction** | **95%** | Massive efficiency gain |

## Smart Entity Index (v0.5.0+)

The Smart Entity Index provides a lightweight (~400-800 tokens) snapshot of your Home Assistant system structure, enabling context-aware queries without full entity dumps. The index includes areas, domains, device classes, people, calendars, zones, automations, and scripts. For entities without standardized device_class attributes (like custom integrations), LLM-powered gap-filling automatically infers semantic categories from naming patterns. This results in faster, more accurate queries that use ~95% fewer tokens compared to traditional entity dumps.

## Entity Pre-Resolution

Pre-Resolution makes your assistant **faster and cheaper** by recognizing device names before the LLM is called.

### Before Pre-Resolution (2 steps)
```
You: "Turn on the kitchen light"
  → LLM calls discover_entities("kitchen light")
  → LLM calls perform_action("light.kitchen", "turn_on")
```

### With Pre-Resolution (1 step)
```
You: "Turn on the kitchen light"
  → Pre-resolution: "kitchen light" = light.kitchen ✓
  → LLM calls perform_action("light.kitchen", "turn_on")
```

**Result**: Faster responses, fewer API calls, lower costs.

### Features

- **Handles typos**: "kichen lite" still finds the kitchen light
- **Multiple devices**: "Turn off kitchen and bedroom lights" resolves both at once
- **Multi-language support**: Works with entity names in any language
- **Automatic fallback**: If no match found, uses normal discovery

### Settings

| Setting | Default | What it does |
|---------|---------|-------------|
| Enable Pre-Resolution | On | Turn the feature on/off |
| Confidence Threshold | 0.90 | How similar a name must be to match |
| Match Margin | 0.08 | How much better the best match must be than alternatives |

## Fast Path for Simple Commands

Fast Path handles simple commands **without calling the LLM at all** - instant responses in ~50ms instead of 1-3 seconds.

### How It Works

```
Traditional:  User → Pre-Resolution → LLM → Action → Response (1-3 sec)
Fast Path:    User → Pre-Resolution → Fast Path → Response (~50ms)
```

### Supported Commands

| Command Type | Examples |
|--------------|----------|
| **On/Off** | "Kitchen light on", "Turn off bedroom" |
| **Toggle** | "Toggle the fan" |
| **Open/Close** | "Open garage door", "Close blinds" |
| **Lock/Unlock** | "Lock front door" |
| **Brightness** | "Dim living room to 50%", "Bedroom 30%" |
| **Temperature** | "Set thermostat to 22°", "Heating 20 degrees" |
| **Position** | "Blinds to 40%", "Shutter 80%" |
| **Color Temp** | "Warm light in kitchen", "Daylight in office" |

### Multi-Language Support

Fast Path includes keywords for 5 languages out of the box:

| Language | Examples |
|----------|----------|
| 🇩🇪 German | "Küche an", "Dimme Wohnzimmer auf 50%" |
| 🇬🇧 English | "Kitchen on", "Dim living room to 50%" |
| 🇫🇷 French | "Allumer cuisine", "Salon à 50%" |
| 🇪🇸 Spanish | "Encender cocina", "Sala al 50%" |
| 🇳🇱 Dutch | "Keuken aan", "Woonkamer op 50%" |

### Custom Keywords

Add your own keywords by creating `custom_components/mcp_assist/fast_path/keywords/custom.yaml`:

```yaml
actions:
  turn_on:
    de:
      - anknipsen
      - hochfahren
    en:
      - fire up
```

### When Fast Path Falls Back to LLM

Fast Path gracefully falls back to the LLM for:
- Questions ("Is the light on?", "What's the temperature?")
- Complex requests (more than ~12 words)
- Unknown entities (not found by Pre-Resolution)
- Unrecognized actions

### Settings

| Setting | Default | What it does |
|---------|---------|-------------|
| Enable Fast Path | On | Turn the feature on/off |
| Fast Path Language | Auto | Auto-detect from HA config, or select manually |

### Performance

| Path | Response Time | Token Cost |
|------|---------------|------------|
| **Fast Path** | ~50ms | 0 tokens |
| LLM (local) | 500-2000ms | ~400 tokens |
| LLM (cloud) | 1000-3000ms | ~400 tokens |

## Requirements

- Home Assistant 2024.1+
- One of:
  - **Local LLMs**: LM Studio v0.3.17+, llama.cpp, or Ollama
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
  - **OpenAI** - Cloud, paid, GPT-5.2 series
  - **Google Gemini** - Cloud, paid/free tier, Gemini 3.0 series
  - **OpenRouter** - Cloud, multi-model gateway with access to 200+ models

**Step 2 - Server Configuration:**

*For Local Servers (LM Studio / llama.cpp / Ollama):*
- Server URL: Where your LLM server is running
  - LM Studio: `http://localhost:1234` (default)
  - llama.cpp: `http://localhost:8080` (default)
  - Ollama: `http://localhost:11434` (default)
- MCP Server Port: Port for the MCP server (default: 8090)

*For Cloud Providers (OpenAI / Gemini / Anthropic / OpenRouter):*
- API Key: Your provider API key (see below for setup)
- MCP Server Port: Port for the MCP server (default: 8090)

**Step 3 - Model & Prompts:**
- Model Name: Select from auto-loaded models or enter manually
- System Prompt: Customize the assistant's personality
- Technical Instructions: Advanced prompt for tool usage (pre-configured)

**Step 4 - Advanced Settings:**
- Temperature: Response randomness (0.0-1.0)
- Max Response Tokens: Maximum length of responses
- Response Mode: Smart / Always / None (follow-up question behavior)
- Control Home Assistant: Enable/disable device control
- Max Tool Iterations: How many tool calls allowed per request
- Web Search Provider: Choose none, DuckDuckGo, or Brave Search
- Brave Search API Key: Your API key (if using Brave Search)
- Debug Mode: Extra logging for troubleshooting
- **Ollama Keep Alive** (Ollama only): Control how long models stay loaded in memory
  - `-1` = Keep loaded indefinitely
  - `0` = Unload immediately after response
  - `"5m"` = Keep for 5 minutes (default)
  - Duration strings like `"24h"`, `"168h"` also supported
- **Ollama Context Window** (Ollama only): Custom context window size (0 = use model default)

### 3. Set as Voice Assistant

1. In Home Assistant, go to **Settings** → **Voice Assistants**
2. Set your preferred assistant to your MCP Assist profile name
3. Test with commands!

## Usage Examples

### Basic Commands
- "Turn on the kitchen lights"
- "Turn off all the lights in the bedroom"
- "What's the temperature in the living room?"

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

### Web Search (if enabled)
- "What's the weather forecast for tomorrow?"
- "Search for the latest Home Assistant updates"
- "What time does the store close?"

## Configuration Options

### Profile Settings
- **Profile Name**: Unique name for this assistant
- **Server Type**: LM Studio, Ollama (more coming)
- **Server URL**: Where your LLM is running
- **Model Name**: Which model to use

### Prompts
- **System Prompt**: Sets the assistant's personality and behavior
- **Technical Instructions**: Low-level instructions for tool usage (usually leave as default)

### Advanced Settings
- **Max Response Tokens**: Limit response length (default: 500)
- **Max History Messages**: How many conversation turns to remember (default: 10)
- **Max Tool Iterations**: Prevent infinite loops (default: 10)
- **Response Mode**:
  - **Smart** (default): LLM decides if follow-up needed
  - **Always**: Always wait for follow-up
  - **None**: Never wait for follow-up
- **Enable Smart Entity Index**: Context-aware entity discovery with automatic gap-filling for uncommon devices (default: enabled)
- **Enable Pre-Resolution**: Automatically recognize device names before LLM call for faster responses (default: enabled)
- **Pre-Resolution Threshold**: How confident the match must be (0.5-1.0, default: 0.90)
- **Pre-Resolution Margin**: How much better the best match must be than the second-best (0.0-0.5, default: 0.08)
- **Enable Fast Path**: Handle simple commands without LLM for instant responses (default: enabled)
- **Fast Path Language**: Language for keywords (auto-detect from HA, or DE/EN/FR/ES/NL)

### Temperature Settings

Temperature controls response randomness (0.0 = deterministic, 1.0 = creative). Different providers have different optimal values:

| Provider | Default | Reason |
|----------|---------|--------|
| **Gemini** | `1.0` | Google requires 1.0 for Gemini 3 to avoid "looping or degraded performance" |
| **OpenAI (GPT-4)** | `0.5` | Balanced for reliable tool calling |
| **OpenAI (GPT-5/o1)** | N/A | Reasoning models don't use temperature |
| **Anthropic Claude** | `0.5` | Works well across 0.5-1.0 range |
| **LM Studio / llama.cpp** | `0.5` | Lower temps improve tool calling accuracy |
| **Ollama** | `0.5` | Model-dependent, lower is safer for tools |
| **OpenRouter** | `0.5` | Depends on underlying model |

**Note**: You can always override these defaults in Advanced Settings. For Home Assistant voice control, lower temperatures (0.5-0.7) generally provide more consistent tool calling and accurate entity control.

### MCP Server Settings
- **MCP Server Port**: Default 8090 (change if port conflict)
- **Additional Allowed IPs/Ranges**: Whitelist Docker containers (e.g., `172.30.0.0/16`) or specific IPs for external MCP clients like Claude Code add-on

### Web Search
- **Web Search Provider**: Choose between:
  - **None**: Search disabled
  - **DuckDuckGo**: Free web search (no API key required)
  - **Brave Search**: Requires API key from https://brave.com/search/api/
- **Brave Search API Key**: Required only if using Brave Search

### Shared vs Per-Profile Settings

MCP Assist has two types of settings:

**Per-Profile Settings** (independent per conversation agent):
- Model name, system prompt, technical instructions
- Temperature, max tokens, response mode
- Debug mode, max iterations
- Server URL (for local LLMs)

**Shared Settings** (affect ALL profiles):
- MCP server port
- Web search provider (none/duckduckgo/brave)
- Brave API key
- Allowed IPs/CIDR ranges
- Smart entity index (gap-filling)

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
