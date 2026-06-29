"""Built-in weather packaged tools for MCP Assist."""

from __future__ import annotations

import copy
from typing import Any, Dict

from ..server_tools.response_services import ResponseServicesMixin
from ..server_tools.weather import WeatherToolsMixin
from .tool_runtime import HomeAssistantToolRuntime

WEATHER_TOOL_DEFINITIONS: list[dict[str, Any]] = [{'name': 'get_weather_forecast',
  'description': 'Get a Home Assistant weather forecast in one call. Prefer this for user weather '
                 'questions before web search. It finds the weather entity, chooses a supported '
                 'forecast type, and summarizes today or tomorrow.',
  'llmDescription': 'Get a Home Assistant weather forecast.',
  'inputSchema': {'$schema': 'http://json-schema.org/draft-07/schema#',
                  'type': 'object',
                  'properties': {'entity_id': {'type': 'string',
                                               'description': 'Optional specific weather entity ID '
                                                              'to use.'},
                                 'area': {'type': 'string',
                                          'description': 'Optional area or room name if you have '
                                                         'multiple weather entities.'},
                                 'floor': {'type': 'string',
                                           'description': 'Optional floor name if you have '
                                                          'multiple weather entities.'},
                                 'label': {'type': 'string',
                                           'description': 'Optional label name if you have '
                                                          'multiple weather entities.'},
                                 'name_contains': {'type': 'string',
                                                   'description': 'Optional text to match a '
                                                                  'specific weather entity name or '
                                                                  'alias.'},
                                 'when': {'type': 'string',
                                          'description': "Forecast day to summarize. Use 'today', "
                                                         "'tomorrow', or a local date like "
                                                         "'2026-04-13'. Defaults to 'tomorrow'."},
                                 'forecast_type': {'type': 'string',
                                                   'enum': ['daily', 'twice_daily', 'hourly'],
                                                   'description': 'Optional forecast type '
                                                                  'override. If unsupported by the '
                                                                  'target entity, the tool falls '
                                                                  'back to a supported type.'}},
                  'required': [],
                  'additionalProperties': False}}]

WEATHER_TOOL_NAMES = {
    str(tool_definition["name"]) for tool_definition in WEATHER_TOOL_DEFINITIONS
}


class WeatherTool(WeatherToolsMixin, ResponseServicesMixin, HomeAssistantToolRuntime):
    """Built-in packaged weather tools."""

    def __init__(self, hass) -> None:
        """Initialize the weather tool bundle."""
        super().__init__(hass)

    async def initialize(self) -> None:
        """Initialize the tool bundle."""
        return None

    async def async_shutdown(self) -> None:
        """Clean up tool resources."""
        return None

    def handles_tool(self, tool_name: str) -> bool:
        """Return whether this bundle handles a tool."""
        return tool_name in WEATHER_TOOL_NAMES

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return weather MCP tool definitions."""
        return copy.deepcopy(WEATHER_TOOL_DEFINITIONS)

    async def handle_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle a weather tool call."""
        if tool_name == "get_weather_forecast":
            return await self.tool_get_weather_forecast(arguments)
        return self._build_text_tool_result(
            f"Unknown weather tool: {tool_name}",
            is_error=True,
        )
