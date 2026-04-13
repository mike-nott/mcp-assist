# External Custom Tools for MCP Assist

MCP Assist can load additional MCP tool packages from your Home Assistant config directory.

This is intended for advanced users who want to:

- expose integration-specific read helpers
- add custom component behavior
- bundle local house-specific helper logic
- extend MCP Assist without forking the integration

It also keeps user extensions out of the HACS-managed integration directory, which is the recommended upgrade-safe pattern for Home Assistant customizations.

## What Belongs Where

Use the built-in/core MCP Assist repo for capabilities that are broadly reusable across many Home Assistant installations, such as:

- generic web, math, calendar, weather, media, or image-analysis helpers
- tools that operate on standard Home Assistant entities or services without assuming one specific house layout
- features that should be trusted and available as part of MCP Assist itself

Use external custom tool packages for capabilities that are installation-specific, such as:

- local custom components, MariaDB audit tables, or house-specific scripts
- family relationships, vehicle nicknames, room aliases, or local naming conventions
- city-specific schedules, site-specific camera zones, or custom dashboards

If a capability only works because it knows details about one household or one set of custom entities, it should stay in `<home-assistant-config>/mcp-assist-tools` instead of being added to the core integration.

## Safety Model

External custom tools are intentionally **safe by default**:

- Disabled by default in the shared MCP server settings
- Loaded only from `<home-assistant-config>/mcp-assist-tools`
- Skipped when invalid instead of crashing the MCP server
- Cannot override built-in MCP Assist tool names
- Must use namespaced tool names
- Cannot load from symlinked package directories
- Cannot reference entrypoint modules or prompt files outside the package directory

Important: once enabled, these packages run as Python code inside Home Assistant. Only install or write packages you trust.

## How to Enable

1. Create one or more tool packages under `<home-assistant-config>/mcp-assist-tools`
2. In MCP Assist shared MCP server settings, enable **Custom Tools**
3. Reload the integration or restart Home Assistant

When enabled, MCP Assist will:

- discover valid packages at startup
- expose their MCP tools alongside built-in tools
- append short tool-specific prompt guidance when provided
- validate external-tool arguments against `inputSchema` before dispatch
- merge optional shared/profile settings when a package declares a settings schema

## Directory Layout

Each tool package lives in its own folder:

```text
<home-assistant-config>/
  mcp-assist-tools/
    __shared__/
      helpers.py
    my_tool/
      mcp_tool.json
      tool.py
      prompt.md
```

Rules:

- The package folder name must match the manifest `id`
- Hidden folders and `__*` folders are ignored
- Symlinked package folders are rejected
- `__shared__/` is reserved for helper modules that multiple packages can import

## Manifest Format

Each package must contain an `mcp_tool.json`.

This namespaced filename avoids collisions with Home Assistant validation tools and generic manifest conventions.

Current schema:

```json
{
  "schema_version": 1,
  "id": "my_tool",
  "name": "My Tool",
  "description": "Short description of what this package adds.",
  "version": "1.0.0",
  "entrypoint": "tool:MyTool",
  "capabilities": [
    "Optional short capability summary shown to the model"
  ],
  "prompt_append_file": "prompt.md"
}
```

### Manifest Fields

- `schema_version`
  - Required
  - Must currently be `1`
- `id`
  - Required
  - Lowercase letters, numbers, and underscores only
  - Must match the package folder name
- `name`
  - Required
  - Human-readable package name
- `description`
  - Required
  - Human-readable package description
- `version`
  - Required
  - Free-form version string
- `entrypoint`
  - Required
  - Must be in `module:ClassName` format
- `capabilities`
  - Optional
  - List of short strings describing what the package can do
- `prompt_append_file`
  - Optional
  - Relative file path inside the package
  - Used to append compact technical guidance for the model

## Python API

Custom tools should import the stable MCP Assist tool API:

```python
from custom_components.mcp_assist.custom_tool_api import MCPAssistExternalTool
```

The entrypoint class must subclass `MCPAssistExternalTool`.

### Required methods

- `get_tool_definitions(self) -> list[dict[str, Any]]`
- `handle_call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]`

### Optional hooks

- `initialize(self) -> None`
- `async_shutdown(self) -> None`
- `get_prompt_instructions(self) -> str`
- `get_settings_schema(self) -> dict[str, Any]`

### Shared helpers

If you want several narrow packages to share code, use the first-class helper loader:

```python
from custom_components.mcp_assist.custom_tool_api import (
    MCPAssistExternalTool,
    load_external_shared_module,
)

shared = load_external_shared_module(__file__, "helpers")
```

MCP Assist loads these helpers from:

- `<home-assistant-config>/mcp-assist-tools/__shared__/`

This avoids ad-hoc `sys.path` shims and keeps package imports collision-safe.

### Package settings

External packages can declare an optional settings schema:

```python
def get_settings_schema(self) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "default_camera": {"type": "string"},
            "zones": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        },
    }
```

When present, MCP Assist will load and validate settings from:

- Shared: `<home-assistant-config>/mcp-assist-tool-settings/<tool_id>.json`
- Per-profile override: `<home-assistant-config>/mcp-assist-tool-settings/profiles/<profile_entry_id>/<tool_id>.json`

During a tool call, packages can read:

- `self.get_settings()`
- `self.get_shared_settings()`
- `self.get_profile_settings()`
- `self.get_call_context()`

### Reusing core MCP tools

External packages can call built-in MCP Assist tools instead of reimplementing
shared behavior:

```python
result = await self.call_mcp_tool(
    "analyze_image",
    {
        "camera_entity_id": "camera.driveway",
        "question": "What is in the driveway right now?",
    },
)
```

Shared helper modules can do the same with the module-level helpers:

```python
from custom_components.mcp_assist.custom_tool_api import (
    call_mcp_tool,
    get_external_tool_call_context,
)

result = await call_mcp_tool(
    hass,
    "analyze_image",
    {"camera_entity_id": "camera.driveway"},
    context=get_external_tool_call_context(),
)
```

This keeps package prompts smaller, preserves active-profile model selection,
and lets external suites compose generic MCP Assist capabilities such as image
analysis without duplicating provider logic.

## Tool Definition Rules

Each tool definition must:

- be an MCP tool definition object
- include `name`
- include `description`
- include `inputSchema`

Tool names must:

- contain only letters, numbers, `_`, or `-`
- be namespaced with the manifest id
- start with `<id>_`

Example:

- valid: `my_tool_status`
- invalid: `status`
- invalid: `discover_entities`

MCP Assist rejects tool-name collisions with built-in tools or other external packages.

## Routing Metadata

To keep the initial prompt small, external tools can add lightweight routing hints directly on the tool definition instead of inflating prompt appendices:

```python
{
    "name": "my_tool_status",
    "description": "Return package status.",
    "inputSchema": {"type": "object", "properties": {}},
    "keywords": ["status", "health", "custom"],
    "example_queries": ["What's the custom system status?"],
    "preferred_when": "Use when the user asks about this package's state.",
    "returns": "A short status summary.",
}
```

You can also place the same fields under a nested `routing` object. MCP Assist normalizes these hints into `routingHints` and uses them when converting MCP tools into compact LLM tool descriptions.

## Prompt Guidance

External tools can teach the model how to use them in two ways:

1. Static text file via `prompt_append_file`
2. Dynamic text from `get_prompt_instructions()`

These appendices should be:

- short
- procedural
- tool-usage focused

They should not:

- restate the entire system prompt
- include long examples unless absolutely necessary
- dump large environment-specific data

Prefer routing metadata over longer prompt text when a hint can be expressed structurally.

MCP Assist automatically truncates overly long external prompt additions to keep context usage small.

## Return Format

`handle_call()` should return standard MCP tool results, for example:

```python
return {
    "content": [
        {"type": "text", "text": "Kitchen scene is active."}
    ],
    "isError": False,
}
```

For errors, prefer returning a normal tool error result instead of raising when the failure is expected:

```python
return {
    "content": [
        {"type": "text", "text": "Scene is not available right now."}
    ],
    "isError": True,
}
```

External tools may also return structured MCP results such as:

- multiple content blocks
- `structuredContent`
- image content blocks for clients that support image rendering

MCP Assist preserves these results in the MCP server response, and conversation profiles now compact them for the LLM without flattening everything to the first text block.

## Best Practices

Follow these guidelines to keep packages portable and maintainable:

- Prefer small, narrow tools over one huge catch-all tool
- Use Home Assistant APIs and current state instead of hardcoded assumptions
- Keep tool names stable
- Keep prompt additions concise
- Prefer routing metadata for discoverability before adding more prompt text
- Use async-safe Home Assistant patterns
- Avoid blocking I/O in tool calls
- Avoid installing or expecting third-party Python dependencies
- Do not assume a specific entity ID, area name, floor name, or user
- Return structured, factual results and let the model phrase them naturally

## Reloading and Diagnostics

You no longer need a full restart just to iterate on package code:

- Home Assistant service: `mcp_assist.reload_external_custom_tools`
- Diagnostics endpoint: `GET /external-tools/diagnostics`

The diagnostics endpoint reports:

- whether external tools are enabled
- which package folders were scanned
- current load errors
- loaded package ids, tool names, and settings status

## Example Package

A working example lives here:

- [docs/examples/mcp-assist-tools/sample_tool/mcp_tool.json](examples/mcp-assist-tools/sample_tool/mcp_tool.json)
- [docs/examples/mcp-assist-tools/sample_tool/tool.py](examples/mcp-assist-tools/sample_tool/tool.py)
- [docs/examples/mcp-assist-tools/sample_tool/prompt.md](examples/mcp-assist-tools/sample_tool/prompt.md)

## Troubleshooting

### My custom tools do not appear

Check:

- **Custom Tools** is enabled in shared MCP server settings
- the package lives under `<home-assistant-config>/mcp-assist-tools/<tool_id>/`
- the folder name matches the manifest `id`
- the manifest file is named `mcp_tool.json` and uses `schema_version: 1`
- the tool names are properly prefixed with `<tool_id>_`
- Home Assistant logs for `Failed to load external custom tool package`
- the diagnostics endpoint or reload service output for package-specific load errors

### My tool package loads but the model never uses it

Check:

- the tool description is clear and action-oriented
- the package adds concise prompt guidance
- the tool is narrow enough for the model to choose confidently
- the tool name and description match the user intent it should handle

### Can packages add profile-specific settings?

Yes. Declare `get_settings_schema()` and use the shared/profile JSON settings paths described above.
