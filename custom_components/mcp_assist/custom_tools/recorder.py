"""Built-in recorder/history packaged tools for MCP Assist."""

from __future__ import annotations

import copy
from typing import Any, Dict

from ..server_tools.recorder import RecorderToolsMixin
from .tool_runtime import HomeAssistantToolRuntime

RECORDER_TOOL_DEFINITIONS: list[dict[str, Any]] = [{'name': 'get_entity_history',
  'description': 'Get recorder-backed history for a specific entity. By default this returns a '
                 "recent timeline, and with mode='last_event' it returns only the most recent "
                 'matching event or change.',
  'llmDescription': 'Get recorder-backed history for an entity.',
  'inputSchema': {'$schema': 'http://json-schema.org/draft-07/schema#',
                  'type': 'object',
                  'properties': {'entity_id': {'type': 'string',
                                               'description': 'The entity ID to get history for. '
                                                              'Use discover_entities first and '
                                                              'pass the discovered entity ID '
                                                              'here.'},
                                 'mode': {'type': 'string',
                                          'enum': ['timeline', 'last_event'],
                                          'default': 'timeline',
                                          'description': 'timeline returns recent history entries; '
                                                         'last_event returns only the most recent '
                                                         'matching event or change.'},
                                 'event': {'type': 'string',
                                           'description': 'Optional semantic event filter for '
                                                          'last_event mode, such as opened, '
                                                          'closed, on, off, locked, unlocked, '
                                                          'detected, cleared, home, or away.'},
                                 'state': {'oneOf': [{'type': 'string'},
                                                     {'type': 'array',
                                                      'items': {'type': 'string'}}],
                                           'description': 'Optional target state or states to '
                                                          'filter by. Works with both timeline and '
                                                          'last_event modes.'},
                                 'hours': {'type': 'integer',
                                           'description': 'Number of hours of recorder history to '
                                                          'search. In timeline mode the default is '
                                                          '24 hours; in last_event mode the '
                                                          'default is 720 hours (30 days). Max: '
                                                          '8760 hours / 1 year.',
                                           'minimum': 1,
                                           'maximum': 8760},
                                 'limit': {'type': 'integer',
                                           'description': 'Maximum number of timeline entries to '
                                                          'return in timeline mode (default: 50, '
                                                          'max: 100). Most recent changes shown '
                                                          'first.',
                                           'default': 50,
                                           'minimum': 1,
                                           'maximum': 100}},
                  'required': ['entity_id'],
                  'additionalProperties': False}},
 {'name': 'analyze_entity_history',
  'description': "Analyze Home Assistant recorder history for aggregate questions such as 'how "
                 "many times was the door opened in the last hour?', 'how long has it been "
                 "locked?', or 'how often did this sensor trigger today?'. Can count all changes "
                 'or matching states/events.',
  'llmDescription': 'Analyze recorder history for counts, durations, or matching transitions.',
  'inputSchema': {'$schema': 'http://json-schema.org/draft-07/schema#',
                  'type': 'object',
                  'properties': {'entity_id': {'type': 'string',
                                               'description': 'The entity ID to analyze in '
                                                              'recorder history.'},
                                 'event': {'type': 'string',
                                           'description': 'Optional semantic event to analyze, '
                                                          'such as opened, closed, on, off, '
                                                          'locked, unlocked, detected, cleared, '
                                                          'home, or away.'},
                                 'state': {'oneOf': [{'type': 'string'},
                                                     {'type': 'array',
                                                      'items': {'type': 'string'}}],
                                           'description': 'Optional specific target state or '
                                                          'states to analyze. If omitted, counts '
                                                          'all recorded state changes in the time '
                                                          'window.'},
                                 'hours': {'type': 'integer',
                                           'description': 'How far back to analyze recorder '
                                                          'history. Default is 24 hours for '
                                                          'count/summary/duration/stats, and 720 '
                                                          'hours (30 days) for streak. Max: 8760 '
                                                          'hours / 1 year.',
                                           'minimum': 1,
                                           'maximum': 8760},
                                 'analysis': {'type': 'string',
                                              'enum': ['count',
                                                       'summary',
                                                       'duration',
                                                       'streak',
                                                       'stats'],
                                              'default': 'count',
                                              'description': 'count returns how many matching '
                                                             'events occurred; summary also '
                                                             'includes the first and last matching '
                                                             'times in the window; duration '
                                                             'reports total time spent in the '
                                                             'matching state; streak reports how '
                                                             'long the entity has continuously '
                                                             'been in the matching state right '
                                                             'now; stats reports numeric '
                                                             'min/max/average over the window.'}},
                  'required': ['entity_id'],
                  'additionalProperties': False}},
 {'name': 'get_entity_state_at_time',
  'description': 'Look up the recorder state of an entity at a specific date/time. Use this for '
                 "questions like 'was the gate open at 2 PM?' or 'what was the temperature at 9 "
                 "this morning?'",
  'llmDescription': "Look up an entity's recorder state at a specific time.",
  'inputSchema': {'$schema': 'http://json-schema.org/draft-07/schema#',
                  'type': 'object',
                  'properties': {'entity_id': {'type': 'string',
                                               'description': 'The entity ID to inspect in '
                                                              'recorder history.'},
                                 'datetime': {'type': 'string',
                                              'description': 'The target date/time to inspect, '
                                                             'preferably as an ISO 8601 timestamp. '
                                                             'If no timezone is included, Home '
                                                             'Assistant local time is assumed.'}},
                  'required': ['entity_id', 'datetime'],
                  'additionalProperties': False}}]

RECORDER_TOOL_NAMES = {
    str(tool_definition["name"]) for tool_definition in RECORDER_TOOL_DEFINITIONS
}


class RecorderTool(RecorderToolsMixin, HomeAssistantToolRuntime):
    """Built-in packaged recorder/history tools."""

    def __init__(self, hass) -> None:
        """Initialize the recorder tool bundle."""
        super().__init__(hass)

    async def initialize(self) -> None:
        """Initialize the tool bundle."""
        return None

    async def async_shutdown(self) -> None:
        """Clean up tool resources."""
        return None

    def handles_tool(self, tool_name: str) -> bool:
        """Return whether this bundle handles a tool."""
        return tool_name in RECORDER_TOOL_NAMES

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return recorder MCP tool definitions."""
        return copy.deepcopy(RECORDER_TOOL_DEFINITIONS)

    async def handle_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle a recorder tool call."""
        if tool_name == "get_entity_history":
            return await self.tool_get_entity_history(arguments)
        if tool_name == "analyze_entity_history":
            return await self.tool_analyze_entity_history(arguments)
        if tool_name == "get_entity_state_at_time":
            return await self.tool_get_entity_state_at_time(arguments)
        return self._build_text_tool_result(
            f"Unknown recorder tool: {tool_name}",
            is_error=True,
        )
