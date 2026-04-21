"""Built-in response-service packaged tools for MCP Assist."""

from __future__ import annotations

import copy
from typing import Any, Dict

from ..server_tools.calendar import CalendarToolsMixin
from ..server_tools.response_services import ResponseServicesMixin
from ..server_tools.weather import WeatherToolsMixin
from .tool_runtime import HomeAssistantToolRuntime

RESPONSE_SERVICE_TOOL_DEFINITIONS: list[dict[str, Any]] = [{'name': 'get_calendar_events',
  'description': 'Get upcoming Home Assistant calendar events in one call. Prefer this for '
                 "questions like 'When is the next Mariners game?' or 'What's on our calendar "
                 "tomorrow?'. It discovers matching calendars, calls calendar.get_events, and "
                 'summarizes the next matching event or agenda.',
  'llmDescription': 'Get upcoming Home Assistant calendar events, schedules, or subscribed team '
                    'games.',
  'routingHints': {'preferred_when': 'Use for calendar, schedule, or next-game questions when Home '
                                     'Assistant may already have the answer.',
                   'example_queries': ['When is the next Mariners game?']},
  'inputSchema': {'$schema': 'http://json-schema.org/draft-07/schema#',
                  'type': 'object',
                  'properties': {'entity_id': {'type': 'string',
                                               'description': 'Optional specific calendar entity '
                                                              'ID to use.'},
                                 'area': {'type': 'string',
                                          'description': 'Optional area or room name if you want '
                                                         'calendars associated with a specific '
                                                         'area.'},
                                 'floor': {'type': 'string',
                                           'description': 'Optional floor name if you want '
                                                          'calendars associated with a specific '
                                                          'floor.'},
                                 'label': {'type': 'string',
                                           'description': 'Optional label name if you want '
                                                          'calendars tagged with a specific '
                                                          'label.'},
                                 'query': {'type': 'string',
                                           'description': 'Optional general text to match either '
                                                          'calendar names or event details, for '
                                                          "example 'Mariners' or 'dentist'."},
                                 'name_contains': {'type': 'string',
                                                   'description': 'Optional text to match a '
                                                                  'specific calendar entity name '
                                                                  'or alias.'},
                                 'event_text': {'type': 'string',
                                                'description': 'Optional text to match event '
                                                               'summary, title, description, or '
                                                               'location.'},
                                 'when': {'type': 'string',
                                          'description': "Optional time window anchor. Use 'now', "
                                                         "'today', 'tomorrow', a local date like "
                                                         "'2026-04-13', or an ISO datetime."},
                                 'days': {'type': 'integer',
                                          'description': 'How many days forward to search when '
                                                         "'when' is omitted or when searching from "
                                                         'a specific start time. Defaults to 60 '
                                                         'for upcoming searches and 1 for '
                                                         'day-specific lookups.',
                                          'default': 60,
                                          'minimum': 1,
                                          'maximum': 365},
                                 'limit': {'type': 'integer',
                                           'description': 'Maximum number of matching events to '
                                                          'summarize.',
                                           'default': 5,
                                           'minimum': 1,
                                           'maximum': 20}},
                  'required': [],
                  'additionalProperties': False}},
 {'name': 'list_response_services',
  'description': 'List Home Assistant services that currently support native response data. Use '
                 'this when you need to discover which read/query-style services can be called '
                 'with call_service_with_response.',
  'llmDescription': 'List services that return structured response data.',
  'inputSchema': {'$schema': 'http://json-schema.org/draft-07/schema#',
                  'type': 'object',
                  'properties': {'domain': {'type': 'string',
                                            'description': 'Optional domain filter, for example '
                                                           "'weather', 'calendar', or "
                                                           "'media_player'."},
                                 'query': {'type': 'string',
                                           'description': 'Optional text filter matching domain, '
                                                          'service, name, or description.'},
                                 'limit': {'type': 'integer',
                                           'description': 'Maximum number of services to return '
                                                          '(default: 50, max: 200).',
                                           'default': 50,
                                           'minimum': 1,
                                           'maximum': 200}},
                  'required': [],
                  'additionalProperties': False}},
 {'name': 'call_service_with_response',
  'description': 'Call a Home Assistant service that returns structured response data for '
                 'read/query use cases. Use this for native service-response reads like calendar '
                 'or to-do queries, media browsing/searching, or integration-specific query data. '
                 'For normal weather questions, prefer get_weather_forecast.',
  'llmDescription': 'Call a Home Assistant service that returns structured data.',
  'inputSchema': {'$schema': 'http://json-schema.org/draft-07/schema#',
                  'type': 'object',
                  'properties': {'domain': {'type': 'string',
                                            'description': 'The Home Assistant domain for the '
                                                           'response-returning service, for '
                                                           "example 'weather', 'calendar', 'todo', "
                                                           "or 'media_player'."},
                                 'service': {'type': 'string',
                                             'description': 'The service/action name to call, for '
                                                            "example 'get_forecasts', "
                                                            "'get_events', 'get_items', "
                                                            "'browse_media', or 'search_media'."},
                                 'target': {'type': 'object',
                                            'description': 'Optional target entities or selector '
                                                           'IDs such as areas, floors, labels, or '
                                                           'devices. Selectors are resolved to '
                                                           'exposed entity IDs, and may be '
                                                           "narrowed using the service's target "
                                                           'metadata when available.',
                                            'properties': {'entity_id': {'oneOf': [{'type': 'string'},
                                                                                   {'type': 'array',
                                                                                    'items': {'type': 'string'}}],
                                                                         'description': 'Single '
                                                                                        'entity ID '
                                                                                        'or list '
                                                                                        'of entity '
                                                                                        'IDs.'},
                                                           'area_id': {'oneOf': [{'type': 'string'},
                                                                                 {'type': 'array',
                                                                                  'items': {'type': 'string'}}],
                                                                       'description': 'Single area '
                                                                                      'ID or list '
                                                                                      'of area '
                                                                                      'IDs.'},
                                                           'floor_id': {'oneOf': [{'type': 'string'},
                                                                                  {'type': 'array',
                                                                                   'items': {'type': 'string'}}],
                                                                        'description': 'Single '
                                                                                       'floor ID '
                                                                                       'or list of '
                                                                                       'floor '
                                                                                       'IDs.'},
                                                           'label_id': {'oneOf': [{'type': 'string'},
                                                                                  {'type': 'array',
                                                                                   'items': {'type': 'string'}}],
                                                                        'description': 'Single '
                                                                                       'label ID '
                                                                                       'or list of '
                                                                                       'label '
                                                                                       'IDs.'},
                                                           'device_id': {'oneOf': [{'type': 'string'},
                                                                                   {'type': 'array',
                                                                                    'items': {'type': 'string'}}],
                                                                         'description': 'Single '
                                                                                        'device ID '
                                                                                        'or list '
                                                                                        'of device '
                                                                                        'IDs.'}},
                                            'minProperties': 1,
                                            'additionalProperties': False},
                                 'data': {'type': 'object',
                                          'description': 'Additional service data for the '
                                                         'response-returning service. Required '
                                                         'fields are validated from Home '
                                                         "Assistant's native service descriptions "
                                                         'when available.',
                                          'additionalProperties': True},
                                 'timeout': {'type': 'integer',
                                             'description': 'Timeout in seconds (default: 60, max: '
                                                            '300).',
                                             'default': 60,
                                             'minimum': 1,
                                             'maximum': 300}},
                  'required': ['domain', 'service'],
                  'additionalProperties': False}}]

RESPONSE_SERVICE_TOOL_NAMES = {
    str(tool_definition["name"]) for tool_definition in RESPONSE_SERVICE_TOOL_DEFINITIONS
}


class ResponseServiceTool(
    CalendarToolsMixin,
    ResponseServicesMixin,
    WeatherToolsMixin,
    HomeAssistantToolRuntime,
):
    """Built-in packaged response-service tools."""

    def __init__(self, hass) -> None:
        """Initialize the response-service tool bundle."""
        super().__init__(hass)

    async def initialize(self) -> None:
        """Initialize the tool bundle."""
        return None

    async def async_shutdown(self) -> None:
        """Clean up tool resources."""
        return None

    def handles_tool(self, tool_name: str) -> bool:
        """Return whether this bundle handles a tool."""
        return tool_name in RESPONSE_SERVICE_TOOL_NAMES

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return response-service MCP tool definitions."""
        return copy.deepcopy(RESPONSE_SERVICE_TOOL_DEFINITIONS)

    async def handle_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle a response-service tool call."""
        if tool_name == "get_calendar_events":
            return await self.tool_get_calendar_events(arguments)
        if tool_name == "list_response_services":
            return await self.tool_list_response_services(arguments)
        if tool_name == "call_service_with_response":
            return await self.tool_call_service_with_response(arguments)
        return self._build_text_tool_result(
            f"Unknown response-service tool: {tool_name}",
            is_error=True,
        )
