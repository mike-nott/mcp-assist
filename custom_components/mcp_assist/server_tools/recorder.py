"""Recorder/history MCP server tools."""

from __future__ import annotations

from datetime import timedelta
import logging
import re
from typing import Any, Dict, List, Tuple

from homeassistant.components.homeassistant import async_should_expose
from homeassistant.components.recorder import history
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)
from homeassistant.util import dt as dt_util

try:
    from homeassistant.helpers import floor_registry as fr
except ImportError:  # pragma: no cover - older Home Assistant versions
    fr = None

_LOGGER = logging.getLogger(__name__)


class RecorderToolsMixin:
    """Recorder/history MCP server tool implementations."""

    async def tool_get_entity_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get entity history with human-readable formatting."""
        mode = str(args.get("mode", "timeline")).strip().casefold()
        if mode not in {"timeline", "last_event"}:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Invalid mode. Use 'timeline' or 'last_event'.",
                    }
                ]
            }
        if mode == "last_event":
            return await self._tool_get_last_entity_event_impl(
                args,
                tool_name="get_entity_history",
            )

        entity_id = args.get("entity_id")
        hours = self._coerce_int_arg(args.get("hours"), default=24, minimum=1, maximum=8760)
        limit = self._coerce_int_arg(args.get("limit"), default=50, minimum=1, maximum=100)
        target_states = self._normalize_history_targets(
            args.get("state"), args.get("event")
        )

        _LOGGER.info(f"📜 Getting history for {entity_id}: {hours} hours, limit {limit}")

        # Notify start
        self.publish_progress(
            "tool_start",
            f"Retrieving history for {entity_id}",
            tool="get_entity_history",
            entity_id=entity_id,
        )

        # 1. Get current state
        current_state = self.hass.states.get(entity_id)
        if not current_state:
            return {
                "content": [
                    {"type": "text", "text": f"Entity '{entity_id}' not found."}
                ]
            }

        friendly_name = current_state.attributes.get("friendly_name", entity_id)

        # 2. Calculate time range (UTC)
        try:
            entity_states = await self._fetch_entity_history_states(
                entity_id,
                hours=hours,
                descending=True,
                limit=None if target_states else limit,
            )
        except Exception as e:
            _LOGGER.error(f"Failed to get history for {entity_id}: {e}")
            return {
                "content": [
                    {"type": "text", "text": f"Failed to retrieve history: {str(e)}"}
                ]
            }

        if target_states:
            entity_states = [
                state
                for state in entity_states
                if state.state.casefold() in target_states
            ]
            entity_states = entity_states[:limit]

        # Notify completion
        self.publish_progress(
            "tool_complete",
            f"History retrieved: {len(entity_states)} changes",
            tool="get_entity_history",
            success=True,
        )

        # 4. Format history (most recent first, limited)
        if not entity_states:
            if target_states:
                search_label = self._describe_history_target(
                    args.get("state"), args.get("event")
                )
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{friendly_name} ({entity_id})\n"
                                f"Current state: {current_state.state}\n\n"
                                f"No recorded {search_label} entries were found in the last {hours} hours."
                            ),
                        }
                    ]
                }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"{friendly_name} ({entity_id})\nCurrent state: {current_state.state}\n\nNo history available for the last {hours} hours.",
                    }
                ]
            }

        # Build formatted text
        text_parts = [
            f"{friendly_name} ({entity_id})",
            f"Current state: {current_state.state}",
            "",
            (
                f"Matching history for {self._describe_history_target(args.get('state'), args.get('event'))} "
                f"(last {hours} hours):"
                if target_states
                else f"Recent history (last {hours} hours):"
            ),
        ]

        for state in entity_states:
            when = state.last_changed or state.last_updated
            text_parts.append(
                f"• {self._format_relative_absolute_time(when)} → {state.state}"
            )

        text_parts.append("")
        text_parts.append(
            (
                f"Showing {len(entity_states)} matching entr{'ies' if len(entity_states) != 1 else 'y'}"
                if target_states
                else f"Showing {len(entity_states)} change{'s' if len(entity_states) != 1 else ''}"
            )
        )

        return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

    async def tool_get_last_entity_event(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Find the latest recorder event for an entity."""
        return await self._tool_get_last_entity_event_impl(
            args,
            tool_name="get_last_entity_event",
        )

    async def _tool_get_last_entity_event_impl(
        self, args: Dict[str, Any], *, tool_name: str
    ) -> Dict[str, Any]:
        """Shared implementation for latest-event recorder lookups."""
        entity_id = args.get("entity_id")
        hours = self._coerce_int_arg(args.get("hours"), default=720, minimum=1, maximum=8760)
        current_state = self.hass.states.get(entity_id)

        if not current_state:
            return {
                "content": [
                    {"type": "text", "text": f"Entity '{entity_id}' not found."}
                ]
            }

        target_states = self._normalize_history_targets(
            args.get("state"), args.get("event")
        )
        history_candidates = self._history_resolution_candidates(
            entity_id,
            args.get("state"),
            args.get("event"),
        )
        selected_candidate = history_candidates[0]
        history_entity_id = selected_candidate["entity_id"]
        resolution_note = selected_candidate["note"]
        current_state = self.hass.states.get(history_entity_id)
        friendly_name = current_state.attributes.get("friendly_name", history_entity_id)
        end_time = dt_util.utcnow()

        self.publish_progress(
            "tool_start",
            f"Searching recorder history for {history_entity_id}",
            tool=tool_name,
            entity_id=history_entity_id,
        )

        try:
            if target_states:
                matched_state = None
                for candidate in history_candidates:
                    candidate_entity_id = candidate["entity_id"]
                    candidate_target_states = target_states
                    for window_hours in self._build_history_search_windows(hours):
                        entity_states = await self._fetch_entity_history_states(
                            candidate_entity_id,
                            hours=window_hours,
                            end_time=end_time,
                            descending=True,
                        )
                        candidate_target_states = self._choose_history_count_states(
                            entity_states,
                            args.get("state"),
                            args.get("event"),
                        )
                        matched_state = next(
                            (
                                state
                                for state in entity_states
                                if state.state.casefold() in candidate_target_states
                            ),
                            None,
                        )
                        if matched_state is not None:
                            history_entity_id = candidate_entity_id
                            resolution_note = candidate["note"]
                            current_state = self.hass.states.get(history_entity_id)
                            friendly_name = current_state.attributes.get(
                                "friendly_name", history_entity_id
                            )
                            target_states = candidate_target_states
                            break
                    if matched_state is not None:
                        break
            else:
                entity_states = await self._fetch_entity_history_states(
                    history_entity_id,
                    hours=hours,
                    end_time=end_time,
                    descending=True,
                    limit=1,
                )
                matched_state = entity_states[0] if entity_states else None
        except Exception as err:
            _LOGGER.error(
                "Failed to get last recorder event for %s: %s", history_entity_id, err
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Failed to retrieve recorder history: {err}",
                    }
                ]
            }

        self.publish_progress(
            "tool_complete",
            "Recorder history search complete",
            tool=tool_name,
            success=True,
            found=matched_state is not None,
        )

        if not matched_state:
            search_label = self._describe_history_target(
                args.get("state"), args.get("event")
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{friendly_name} ({history_entity_id})\n"
                            f"Current state: {current_state.state}\n\n"
                            f"No recorded {search_label} event was found in the last {hours} hours."
                            + (
                                f"\n\n{resolution_note}"
                                if resolution_note
                                else ""
                            )
                        ),
                    }
                ]
            }

        when = matched_state.last_changed or matched_state.last_updated
        search_label = self._describe_history_target(
            args.get("state"), args.get("event")
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        (
                            f"{resolution_note}\n\n"
                            if resolution_note
                            else ""
                        )
                        + f"{friendly_name} ({history_entity_id})\n"
                        f"Current state: {current_state.state}\n"
                        f"Last recorded {search_label}: {self._format_relative_absolute_time(when)}\n"
                        f"Matched state: {matched_state.state}"
                    ),
                }
            ]
        }

    async def tool_analyze_entity_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze recorder history for counts and summaries."""
        analysis = str(args.get("analysis", "count")).strip().casefold()
        if analysis not in {"count", "summary", "duration", "stats", "streak"}:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Invalid analysis. Use 'count', 'summary', 'duration', 'streak', or 'stats'.",
                    }
                ]
            }
        entity_id = args.get("entity_id")
        default_hours = 720 if analysis == "streak" else 24
        hours = self._coerce_int_arg(
            args.get("hours"), default=default_hours, minimum=1, maximum=8760
        )
        history_candidates = self._history_resolution_candidates(
            entity_id,
            args.get("state"),
            args.get("event"),
        )
        selected_candidate = history_candidates[0]
        history_entity_id = selected_candidate["entity_id"]
        resolution_note = selected_candidate["note"]
        current_state = self.hass.states.get(history_entity_id)

        if not current_state:
            return {
                "content": [
                    {"type": "text", "text": f"Entity '{history_entity_id}' not found."}
                ]
            }

        friendly_name = current_state.attributes.get("friendly_name", history_entity_id)
        target_states = self._normalize_history_targets(
            args.get("state"), args.get("event")
        )

        self.publish_progress(
            "tool_start",
            f"Analyzing recorder history for {history_entity_id}",
            tool="analyze_entity_history",
            entity_id=history_entity_id,
        )

        query_end_time = dt_util.utcnow()
        query_start_time = query_end_time - timedelta(hours=hours)
        include_start_time_state = analysis in {"duration", "streak", "stats"}

        async def _load_candidate_history(candidate: Dict[str, Any]) -> Dict[str, Any] | None:
            candidate_entity_id = candidate["entity_id"]
            candidate_current_state = self.hass.states.get(candidate_entity_id)
            if candidate_current_state is None:
                return None

            candidate_entity_states = await self._fetch_entity_history_states(
                candidate_entity_id,
                hours=hours,
                end_time=query_end_time,
                descending=False,
                include_start_time_state=include_start_time_state,
            )
            if target_states:
                candidate_target_filter_states = self._choose_history_count_states(
                    candidate_entity_states,
                    args.get("state"),
                    args.get("event"),
                )
            else:
                candidate_target_filter_states = []

            return {
                "candidate": candidate,
                "entity_id": candidate_entity_id,
                "current_state": candidate_current_state,
                "friendly_name": candidate_current_state.attributes.get(
                    "friendly_name", candidate_entity_id
                ),
                "entity_states": candidate_entity_states,
                "target_filter_states": candidate_target_filter_states,
            }

        try:
            candidate_history = await _load_candidate_history(selected_candidate)
        except Exception as err:
            _LOGGER.error(
                "Failed to analyze history for %s: %s", history_entity_id, err
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Failed to analyze recorder history: {err}",
                    }
                ]
            }

        if candidate_history is None:
            return {
                "content": [
                    {"type": "text", "text": f"Entity '{history_entity_id}' not found."}
                ]
            }

        history_entity_id = candidate_history["entity_id"]
        current_state = candidate_history["current_state"]
        friendly_name = candidate_history["friendly_name"]
        entity_states = candidate_history["entity_states"]

        target_filter_states = candidate_history["target_filter_states"]

        if analysis == "duration":
            if not target_filter_states:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{friendly_name} ({history_entity_id})\n"
                                "Duration analysis requires a target event or state, such as opened, closed, on, or off."
                            ),
                        }
                    ]
                }

            duration_result = self._calculate_state_duration(
                entity_states,
                target_filter_states,
                query_start_time,
                query_end_time,
            )
            if duration_result["interval_count"] == 0 and len(history_candidates) > 1:
                for candidate in history_candidates[1:]:
                    candidate_history = await _load_candidate_history(candidate)
                    if candidate_history is None:
                        continue
                    candidate_duration_result = self._calculate_state_duration(
                        candidate_history["entity_states"],
                        candidate_history["target_filter_states"],
                        query_start_time,
                        query_end_time,
                    )
                    candidate_current_state = candidate_history["current_state"]
                    if (
                        candidate_duration_result["interval_count"] > 0
                        or (
                            candidate_history["target_filter_states"]
                            and candidate_current_state.state.casefold()
                            in candidate_history["target_filter_states"]
                        )
                    ):
                        history_entity_id = candidate_history["entity_id"]
                        current_state = candidate_current_state
                        friendly_name = candidate_history["friendly_name"]
                        entity_states = candidate_history["entity_states"]
                        target_filter_states = candidate_history["target_filter_states"]
                        resolution_note = candidate["note"]
                        duration_result = candidate_duration_result
                        break

            total_duration = duration_result["total_duration"]
            interval_count = duration_result["interval_count"]
            search_label = self._describe_history_target(
                args.get("state"), args.get("event")
            )

            self.publish_progress(
                "tool_complete",
                "Recorder history duration analysis complete",
                tool="analyze_entity_history",
                success=True,
                seconds=int(total_duration.total_seconds()),
            )

            text_parts = [
                f"{friendly_name} ({history_entity_id})",
                f"Current state: {current_state.state}",
                f"Total time in {search_label} state during the last {hours} hour{'s' if hours != 1 else ''}: {self._format_duration(total_duration)}",
                f"Matching interval{'s' if interval_count != 1 else ''}: {interval_count}",
            ]
            text_parts = self._prepend_resolution_note(text_parts, resolution_note)

            if interval_count:
                first_start = duration_result.get("first_start")
                last_end = duration_result.get("last_end")
                if first_start is not None:
                    text_parts.append(
                        "First matching interval in window started: "
                        f"{self._format_relative_absolute_time(first_start)}"
                    )
                if last_end is not None and last_end < query_end_time:
                    text_parts.append(
                        f"Last matching interval ended: {self._format_relative_absolute_time(last_end)}"
                    )

            if current_state.state.casefold() in target_filter_states:
                streak_start = current_state.last_changed or current_state.last_updated
                text_parts.append(
                    f"Current ongoing {search_label} streak: {self._format_duration(query_end_time - streak_start)}"
                )

            if not interval_count:
                text_parts.append("No matching recorder intervals were found in that window.")

            return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

        if analysis == "streak":
            if not target_filter_states:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{friendly_name} ({history_entity_id})\n"
                                "Streak analysis requires a target event or state, such as locked, opened, on, or home."
                            ),
                        }
                    ]
                }

            search_label = self._describe_history_target(
                args.get("state"), args.get("event")
            )
            if (
                current_state.state.casefold() not in target_filter_states
                and len(history_candidates) > 1
            ):
                for candidate in history_candidates[1:]:
                    candidate_history = await _load_candidate_history(candidate)
                    if candidate_history is None:
                        continue
                    candidate_current_state = candidate_history["current_state"]
                    if (
                        candidate_history["target_filter_states"]
                        and candidate_current_state.state.casefold()
                        in candidate_history["target_filter_states"]
                    ):
                        history_entity_id = candidate_history["entity_id"]
                        current_state = candidate_current_state
                        friendly_name = candidate_history["friendly_name"]
                        entity_states = candidate_history["entity_states"]
                        target_filter_states = candidate_history["target_filter_states"]
                        resolution_note = candidate["note"]
                        break

            if current_state.state.casefold() not in target_filter_states:
                text_parts = [
                    f"{friendly_name} ({history_entity_id})",
                    f"Current state: {current_state.state}",
                    f"It is not currently {search_label}, so there is no ongoing {search_label} streak to measure.",
                ]
                text_parts = self._prepend_resolution_note(text_parts, resolution_note)
                return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

            streak_start = query_start_time
            exact_start = False
            streak_reaches_window_start = True

            for window_hours in self._build_history_search_windows(hours):
                window_start = query_end_time - timedelta(hours=window_hours)
                window_states = await self._fetch_entity_history_states(
                    history_entity_id,
                    hours=window_hours,
                    end_time=query_end_time,
                    descending=False,
                    include_start_time_state=True,
                )
                if not window_states:
                    continue
                if window_states[-1].state.casefold() not in target_filter_states:
                    break

                start_idx = len(window_states) - 1
                while (
                    start_idx > 0
                    and window_states[start_idx - 1].state.casefold() in target_filter_states
                ):
                    start_idx -= 1

                streak_start = (
                    window_states[start_idx].last_changed
                    or window_states[start_idx].last_updated
                )
                if start_idx > 0:
                    exact_start = True
                    streak_reaches_window_start = False
                    break
                if streak_start > window_start:
                    exact_start = True
                    streak_reaches_window_start = False
                    break
                streak_reaches_window_start = True
                if window_hours >= hours:
                    break

            streak_duration = query_end_time - max(streak_start, query_start_time)

            self.publish_progress(
                "tool_complete",
                "Recorder history streak analysis complete",
                tool="analyze_entity_history",
                success=True,
                seconds=int(streak_duration.total_seconds()),
            )

            duration_text = self._format_duration(streak_duration)
            text_parts = [
                f"{friendly_name} ({history_entity_id})",
                f"Current state: {current_state.state}",
            ]
            if exact_start:
                text_parts.append(
                    f"Current {search_label} streak: {duration_text}"
                )
                text_parts.append(
                    f"Streak started: {self._format_relative_absolute_time(streak_start)}"
                )
            else:
                text_parts.append(
                    f"Current {search_label} streak: at least {duration_text}"
                )
                if streak_reaches_window_start:
                    text_parts.append(
                        f"The streak extends beyond the searched {hours}-hour window."
                    )
            text_parts = self._prepend_resolution_note(text_parts, resolution_note)
            return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

        if analysis == "stats":
            stats_result = self._calculate_numeric_history_stats(
                entity_states,
                query_start_time,
                query_end_time,
            )

            if stats_result is None:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{friendly_name} ({history_entity_id})\n"
                                "Recorder history did not contain numeric states in that window, so min/max/average could not be calculated."
                            ),
                        }
                    ]
                }

            self.publish_progress(
                "tool_complete",
                "Recorder history numeric analysis complete",
                tool="analyze_entity_history",
                success=True,
            )

            min_time = stats_result["min_time"]
            max_time = stats_result["max_time"]
            text_parts = [
                f"{friendly_name} ({history_entity_id})",
                f"Current state: {current_state.state}",
                f"Numeric recorder stats for the last {hours} hour{'s' if hours != 1 else ''}:",
                f"Minimum: {self._format_number(stats_result['min'])}"
                + (
                    f" at {self._format_relative_absolute_time(min_time)}"
                    if min_time is not None
                    else ""
                ),
                f"Maximum: {self._format_number(stats_result['max'])}"
                + (
                    f" at {self._format_relative_absolute_time(max_time)}"
                    if max_time is not None
                    else ""
                ),
                f"Average: {self._format_number(stats_result['average'])} (time-weighted)",
                f"Numeric state samples: {stats_result['sample_count']}",
            ]
            text_parts = self._prepend_resolution_note(text_parts, resolution_note)

            return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

        if target_filter_states:
            matched_states = [
                state
                for state in entity_states
                if state.state.casefold() in target_filter_states
            ]
        else:
            matched_states = entity_states

        if target_filter_states and not matched_states and len(history_candidates) > 1:
            for candidate in history_candidates[1:]:
                candidate_history = await _load_candidate_history(candidate)
                if candidate_history is None:
                    continue
                candidate_matched_states = [
                    state
                    for state in candidate_history["entity_states"]
                    if state.state.casefold() in candidate_history["target_filter_states"]
                ]
                if candidate_matched_states:
                    history_entity_id = candidate_history["entity_id"]
                    current_state = candidate_history["current_state"]
                    friendly_name = candidate_history["friendly_name"]
                    entity_states = candidate_history["entity_states"]
                    target_filter_states = candidate_history["target_filter_states"]
                    matched_states = candidate_matched_states
                    resolution_note = candidate["note"]
                    break

        match_count = len(matched_states)

        self.publish_progress(
            "tool_complete",
            "Recorder history analysis complete",
            tool="analyze_entity_history",
            success=True,
            count=match_count,
        )

        search_label = self._describe_history_target(
            args.get("state"), args.get("event")
        )
        noun = f"{search_label} event" if search_label != "change" else "state change"

        text_parts = [
            f"{friendly_name} ({history_entity_id})",
            f"Current state: {current_state.state}",
            f"Recorded {noun}{'s' if match_count != 1 else ''} in the last {hours} hour{'s' if hours != 1 else ''}: {match_count}",
        ]

        if target_filter_states:
            text_parts.append(
                f"Counted using recorder state{'s' if len(target_filter_states) != 1 else ''}: {', '.join(target_filter_states)}"
            )

        if analysis == "summary" and matched_states:
            first_match = matched_states[0]
            last_match = matched_states[-1]
            first_when = first_match.last_changed or first_match.last_updated
            last_when = last_match.last_changed or last_match.last_updated
            text_parts.append(
                f"First matching event in window: {self._format_relative_absolute_time(first_when)}"
            )
            text_parts.append(
                f"Most recent matching event: {self._format_relative_absolute_time(last_when)}"
            )

        if analysis == "summary" and not matched_states:
            text_parts.append("No matching recorder events were found in that window.")

        text_parts = self._prepend_resolution_note(text_parts, resolution_note)
        return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

    async def tool_get_entity_state_at_time(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return an entity's recorder state at a specific point in time."""
        entity_id = args.get("entity_id")
        raw_datetime = args.get("datetime")
        current_state = self.hass.states.get(entity_id)

        if not current_state:
            return {
                "content": [
                    {"type": "text", "text": f"Entity '{entity_id}' not found."}
                ]
            }

        target_time = self._parse_history_datetime(raw_datetime)
        if target_time is None:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Invalid datetime. Use an ISO 8601 timestamp or a Home Assistant-local datetime string.",
                    }
                ]
            }

        friendly_name = current_state.attributes.get("friendly_name", entity_id)
        lookup_start = target_time + timedelta(microseconds=1)

        self.publish_progress(
            "tool_start",
            f"Looking up recorder state for {entity_id} at {raw_datetime}",
            tool="get_entity_state_at_time",
            entity_id=entity_id,
        )

        try:
            entity_states = await self._fetch_entity_history_states(
                entity_id,
                start_time=lookup_start,
                end_time=lookup_start + timedelta(seconds=1),
                descending=False,
                include_start_time_state=True,
            )
        except Exception as err:
            _LOGGER.error("Failed to look up recorder state for %s: %s", entity_id, err)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Failed to retrieve recorder state: {err}",
                    }
                ]
            }

        state_at_time = entity_states[0] if entity_states else None
        if state_at_time is not None:
            state_when = state_at_time.last_changed or state_at_time.last_updated
            if state_when > target_time:
                state_at_time = None

        self.publish_progress(
            "tool_complete",
            "Recorder point-in-time lookup complete",
            tool="get_entity_state_at_time",
            success=True,
            found=state_at_time is not None,
        )

        target_local = self._format_absolute_time(target_time)
        if state_at_time is None:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{friendly_name} ({entity_id})\n"
                            f"No recorder state was available for {target_local}."
                        ),
                    }
                ]
            }

        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"{friendly_name} ({entity_id})\n"
                        f"State at {target_local}: {state_at_time.state}\n"
                        f"Current state: {current_state.state}"
                    ),
                }
            ]
        }

    def _extract_history_states(
        self, history_result: Dict[str, Any], entity_id: str
    ) -> List[Any]:
        """Extract an entity's state list from recorder helper results."""
        return (
            history_result.get(entity_id)
            or history_result.get(entity_id.casefold())
            or next(iter(history_result.values()), [])
        )

    def _normalize_history_targets(
        self, raw_state: Any, raw_event: Any
    ) -> List[str]:
        """Normalize target states or semantic events into recorder states."""
        values: List[str] = []

        if isinstance(raw_state, list):
            values.extend(str(value) for value in raw_state)
        elif raw_state is not None:
            values.append(str(raw_state))

        if raw_event is not None:
            values.append(str(raw_event))

        expanded: List[str] = []
        for value in values:
            normalized = value.strip().casefold().replace(" ", "_")
            if not normalized:
                continue
            expanded.extend(self._history_state_aliases(normalized))

        deduped: List[str] = []
        seen = set()
        for value in expanded:
            if value not in seen:
                seen.add(value)
                deduped.append(value)
        return deduped

    def _normalize_history_request_values(
        self, raw_state: Any, raw_event: Any
    ) -> List[str]:
        """Normalize raw requested history values before alias expansion."""
        values: List[str] = []

        if isinstance(raw_state, list):
            values.extend(str(value) for value in raw_state)
        elif raw_state is not None:
            values.append(str(raw_state))

        if raw_event is not None:
            values.append(str(raw_event))

        normalized: List[str] = []
        seen = set()
        for value in values:
            item = value.strip().casefold().replace(" ", "_")
            if item and item not in seen:
                seen.add(item)
                normalized.append(item)
        return normalized

    def _choose_history_count_states(
        self, entity_states: List[Any], raw_state: Any, raw_event: Any
    ) -> List[str]:
        """Choose the best recorder states to count for semantic event requests."""
        present_states = {state.state.casefold() for state in entity_states}
        requested_values = self._normalize_history_request_values(raw_state, raw_event)
        chosen: List[str] = []
        seen = set()

        for value in requested_values:
            aliases = self._history_state_aliases(value)
            selected = next((alias for alias in aliases if alias in present_states), aliases[0])
            if selected not in seen:
                seen.add(selected)
                chosen.append(selected)

        return chosen or self._normalize_history_targets(raw_state, raw_event)

    def _calculate_state_duration(
        self,
        entity_states: List[Any],
        target_states: List[str],
        start_time,
        end_time,
    ) -> Dict[str, Any]:
        """Calculate total time spent in the target states over a window."""
        total_duration = timedelta(0)
        interval_count = 0
        first_start = None
        last_end = None

        for index, state in enumerate(entity_states):
            state_start = self._clip_time_to_window(
                state.last_changed or state.last_updated,
                start_time,
                end_time,
            )

            if index + 1 < len(entity_states):
                next_state = entity_states[index + 1]
                state_end = self._clip_time_to_window(
                    next_state.last_changed or next_state.last_updated,
                    start_time,
                    end_time,
                )
            else:
                state_end = end_time

            if state_end <= state_start:
                continue

            if state.state.casefold() not in target_states:
                continue

            total_duration += state_end - state_start
            interval_count += 1
            if first_start is None:
                first_start = state_start
            last_end = state_end

        return {
            "total_duration": total_duration,
            "interval_count": interval_count,
            "first_start": first_start,
            "last_end": last_end,
        }

    def _calculate_numeric_history_stats(
        self,
        entity_states: List[Any],
        start_time,
        end_time,
    ) -> Dict[str, Any] | None:
        """Calculate numeric min/max/average across recorder states."""
        numeric_points: List[tuple[float, Any]] = []
        weighted_sum = 0.0
        weighted_seconds = 0.0

        for index, state in enumerate(entity_states):
            numeric_value = self._coerce_numeric_state(state.state)
            if numeric_value is None:
                continue

            state_time = self._clip_time_to_window(
                state.last_changed or state.last_updated,
                start_time,
                end_time,
            )
            numeric_points.append((numeric_value, state_time))

            if index + 1 < len(entity_states):
                next_state = entity_states[index + 1]
                state_end = self._clip_time_to_window(
                    next_state.last_changed or next_state.last_updated,
                    start_time,
                    end_time,
                )
            else:
                state_end = end_time

            seconds = max((state_end - state_time).total_seconds(), 0)
            if seconds > 0:
                weighted_sum += numeric_value * seconds
                weighted_seconds += seconds

        if not numeric_points:
            return None

        min_value, min_time = min(numeric_points, key=lambda item: item[0])
        max_value, max_time = max(numeric_points, key=lambda item: item[0])

        if weighted_seconds > 0:
            average = weighted_sum / weighted_seconds
        else:
            average = sum(value for value, _ in numeric_points) / len(numeric_points)

        return {
            "min": min_value,
            "min_time": min_time,
            "max": max_value,
            "max_time": max_time,
            "average": average,
            "sample_count": len(numeric_points),
        }

    def _clip_time_to_window(self, when, start_time, end_time):
        """Clip a timestamp to the requested analysis window."""
        if when < start_time:
            return start_time
        if when > end_time:
            return end_time
        return when

    def _coerce_numeric_state(self, value: Any) -> float | None:
        """Convert a recorder state value into a numeric value when possible."""
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            return None

        if number != number or number in (float("inf"), float("-inf")):
            return None

        return number

    def _format_duration(self, duration: timedelta) -> str:
        """Format a duration in human-friendly units."""
        total_seconds = max(int(duration.total_seconds()), 0)
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts: List[str] = []
        if days:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds and not parts:
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

        return ", ".join(parts) if parts else "0 seconds"

    def _format_number(self, value: float) -> str:
        """Format numeric values compactly for responses."""
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.3f}".rstrip("0").rstrip(".")

    def _history_preferred_domains(self, requested_values: List[str]) -> Dict[str, int]:
        """Return domain preferences for semantic history requests."""
        domain_scores: Dict[str, int] = {}
        for value in requested_values:
            if value in {"locked", "unlocked"}:
                domain_scores["lock"] = max(domain_scores.get("lock", 0), 120)
            elif value in {"open", "opened", "opening", "close", "closed", "closing"}:
                domain_scores["cover"] = max(domain_scores.get("cover", 0), 90)
                domain_scores["binary_sensor"] = max(
                    domain_scores.get("binary_sensor", 0), 80
                )
                domain_scores["lock"] = max(domain_scores.get("lock", 0), 40)
            elif value in {"home", "away", "not_home"}:
                domain_scores["person"] = max(domain_scores.get("person", 0), 100)
                domain_scores["device_tracker"] = max(
                    domain_scores.get("device_tracker", 0), 95
                )
            elif value in {"detected", "triggered", "clear", "cleared"}:
                domain_scores["binary_sensor"] = max(
                    domain_scores.get("binary_sensor", 0), 90
                )
        return domain_scores

    def _history_preferred_device_classes(self, requested_values: List[str]) -> set[str]:
        """Return preferred device classes for semantic history requests."""
        device_classes: set[str] = set()
        for value in requested_values:
            if value in {"open", "opened", "opening", "close", "closed", "closing"}:
                device_classes.update({"door", "window", "opening", "garage_door"})
            elif value in {"detected", "triggered", "clear", "cleared"}:
                device_classes.update(
                    {
                        "motion",
                        "occupancy",
                        "presence",
                        "door",
                        "window",
                        "opening",
                        "garage_door",
                        "sound",
                        "vibration",
                    }
                )
        return device_classes

    def _score_entity_for_history_request(
        self,
        state_obj: Any,
        requested_values: List[str],
        target_states: List[str],
    ) -> int:
        """Score how appropriate an entity is for a semantic history request."""
        if state_obj is None or not requested_values:
            return 0

        domain = state_obj.entity_id.split(".", 1)[0]
        device_class = str(state_obj.attributes.get("device_class", "")).casefold()
        current_state = state_obj.state.casefold()
        preferred_domains = self._history_preferred_domains(requested_values)
        preferred_device_classes = self._history_preferred_device_classes(
            requested_values
        )

        score = preferred_domains.get(domain, 0)
        if device_class and device_class in preferred_device_classes:
            score += 25
        if current_state in target_states:
            score += 20

        return score

    def _history_relation_tokens(self, value: Any) -> List[str]:
        """Normalize an entity-related name into meaningful comparison tokens."""
        if value in (None, ""):
            return []

        stopwords = {
            "a",
            "an",
            "the",
            "entity",
            "device",
            "binary",
            "sensor",
            "contact",
            "contacts",
            "lock",
            "locks",
            "locked",
            "unlock",
            "unlocked",
            "deadbolt",
            "bolt",
        }
        raw_tokens = re.findall(r"[a-z0-9]+", str(value).casefold())
        filtered = [
            token
            for token in raw_tokens
            if len(token) > 1 and token not in stopwords
        ]
        return filtered or [token for token in raw_tokens if len(token) > 1]

    def _history_relation_texts(self, state_obj: Any) -> List[str]:
        """Collect stable entity name variants for related-entity matching."""
        if state_obj is None:
            return []

        texts: List[str] = []
        for value in (
            getattr(state_obj, "name", None),
            state_obj.attributes.get("friendly_name"),
            state_obj.entity_id.split(".", 1)[1].replace("_", " "),
        ):
            text = str(value).strip() if value not in (None, "") else ""
            if text and text not in texts:
                texts.append(text)
        return texts

    def _history_entity_location_signature(
        self, entity_id: str
    ) -> Tuple[str | None, str | None, str | None]:
        """Return device, area, and floor identifiers for an entity."""
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        area_registry = ar.async_get(self.hass)
        floor_registry = fr.async_get(self.hass) if fr else None

        entity_entry = entity_registry.async_get(entity_id)
        if entity_entry is None:
            return None, None, None

        device_entry = (
            device_registry.async_get(entity_entry.device_id)
            if entity_entry.device_id
            else None
        )
        area_id = entity_entry.area_id or (device_entry.area_id if device_entry else None)
        area_entry = area_registry.async_get_area(area_id) if area_id else None
        floor_id = getattr(area_entry, "floor_id", None) if area_entry else None
        if floor_id and floor_registry is not None:
            floor_entry = floor_registry.async_get_floor(floor_id)
            floor_id = floor_entry.floor_id if floor_entry else floor_id

        return entity_entry.device_id, area_id, floor_id

    def _score_history_entity_relatedness(
        self, current_state: Any, candidate_state: Any
    ) -> Tuple[int, bool]:
        """Score how likely two entities describe the same real-world thing."""
        if current_state is None or candidate_state is None:
            return 0, False

        current_device_id, current_area_id, current_floor_id = (
            self._history_entity_location_signature(current_state.entity_id)
        )
        candidate_device_id, candidate_area_id, candidate_floor_id = (
            self._history_entity_location_signature(candidate_state.entity_id)
        )

        best_name_score = 0
        for current_text in self._history_relation_texts(current_state):
            current_tokens = self._history_relation_tokens(current_text)
            current_canonical = " ".join(current_tokens)
            if not current_tokens:
                continue

            for candidate_text in self._history_relation_texts(candidate_state):
                candidate_tokens = self._history_relation_tokens(candidate_text)
                candidate_canonical = " ".join(candidate_tokens)
                if not candidate_tokens:
                    continue

                overlap = set(current_tokens) & set(candidate_tokens)
                if not overlap:
                    continue

                score = len(overlap) * 22
                coverage = len(overlap) / min(len(current_tokens), len(candidate_tokens))
                score += int(coverage * 55)

                if current_canonical and current_canonical == candidate_canonical:
                    score += 80
                elif (
                    current_canonical
                    and candidate_canonical
                    and min(len(current_canonical), len(candidate_canonical)) >= 5
                    and (
                        current_canonical in candidate_canonical
                        or candidate_canonical in current_canonical
                    )
                ):
                    score += 40

                best_name_score = max(best_name_score, score)

        same_device = bool(
            current_device_id
            and candidate_device_id
            and current_device_id == candidate_device_id
        )
        if same_device:
            best_name_score += 120
        elif best_name_score:
            if current_area_id and candidate_area_id and current_area_id == candidate_area_id:
                best_name_score += 10
            if (
                current_floor_id
                and candidate_floor_id
                and current_floor_id == candidate_floor_id
            ):
                best_name_score += 5

        return best_name_score, same_device

    def _resolve_history_entity_for_request(
        self,
        entity_id: str,
        raw_state: Any,
        raw_event: Any,
    ) -> tuple[str, str | None]:
        """Resolve the best entity for semantic history requests when needed."""
        candidates = self._history_resolution_candidates(entity_id, raw_state, raw_event)
        best_candidate = candidates[0]
        return best_candidate["entity_id"], best_candidate["note"]

    def _history_resolution_candidates(
        self,
        entity_id: str,
        raw_state: Any,
        raw_event: Any,
        *,
        max_candidates: int = 4,
    ) -> List[Dict[str, Any]]:
        """Return strong related-entity candidates for a semantic history request."""
        requested_values = self._normalize_history_request_values(raw_state, raw_event)
        if not requested_values:
            return [{"entity_id": entity_id, "note": None}]

        target_states = self._normalize_history_targets(raw_state, raw_event)
        current_state = self.hass.states.get(entity_id)
        if current_state is None:
            return [{"entity_id": entity_id, "note": None}]

        current_score = self._score_entity_for_history_request(
            current_state,
            requested_values,
            target_states,
        )
        candidates: List[Dict[str, Any]] = [
            {
                "entity_id": entity_id,
                "note": None,
                "priority": current_score + 150,
                "semantic_score": current_score,
                "relation_score": 0,
            }
        ]

        requested_label = self._describe_history_target(raw_state, raw_event)

        for sibling_state in self.hass.states.async_all():
            sibling_entity_id = sibling_state.entity_id
            if sibling_entity_id == entity_id:
                continue
            if not async_should_expose(self.hass, "conversation", sibling_entity_id):
                continue
            sibling_score = self._score_entity_for_history_request(
                sibling_state,
                requested_values,
                target_states,
            )
            if sibling_score <= 0:
                continue

            relation_score, same_device = self._score_history_entity_relatedness(
                current_state,
                sibling_state,
            )
            if relation_score <= 0:
                continue

            total_score = sibling_score + relation_score
            if same_device:
                if (
                    sibling_score <= current_score
                    or total_score < current_score + 40
                ):
                    continue
            else:
                if (
                    sibling_score < current_score + 20
                    or relation_score < 90
                    or total_score < current_score + 100
                ):
                    continue

            reason_text = (
                "is on the same device and"
                if same_device
                else "strongly matches the same named thing and"
            )
            candidates.append(
                {
                    "entity_id": sibling_entity_id,
                    "note": (
                        f"Using related entity {sibling_entity_id} because it {reason_text} "
                        f"the requested {requested_label} history applies more directly to that entity "
                        f"than {entity_id}."
                    ),
                    "priority": total_score + (40 if same_device else 0),
                    "semantic_score": sibling_score,
                    "relation_score": relation_score,
                }
            )

        candidates.sort(
            key=lambda item: (
                -int(item["priority"]),
                -int(item["semantic_score"]),
                item["entity_id"],
            )
        )
        return candidates[:max_candidates]

    def _prepend_resolution_note(
        self, text_parts: List[str], resolution_note: str | None
    ) -> List[str]:
        """Prepend a history-entity resolution note when a sibling entity was used."""
        if not resolution_note:
            return text_parts
        return [resolution_note, ""] + text_parts

    def _history_state_aliases(self, value: str) -> List[str]:
        """Map semantic event words to likely recorder states."""
        aliases = {
            "open": ["open", "opening", "on"],
            "opened": ["open", "opening", "on"],
            "opening": ["opening", "open", "on"],
            "close": ["closed", "closing", "off"],
            "closed": ["closed", "closing", "off"],
            "closing": ["closing", "closed", "off"],
            "on": ["on"],
            "turned_on": ["on"],
            "enabled": ["on"],
            "active": ["on"],
            "off": ["off"],
            "turned_off": ["off"],
            "disabled": ["off"],
            "locked": ["locked"],
            "unlocked": ["unlocked"],
            "detected": ["detected", "on", "open"],
            "triggered": ["triggered", "on"],
            "clear": ["clear", "off", "closed"],
            "cleared": ["clear", "off", "closed"],
            "home": ["home"],
            "away": ["away", "not_home"],
            "not_home": ["not_home", "away"],
        }
        return aliases.get(value, [value])

    def _describe_history_target(self, raw_state: Any, raw_event: Any) -> str:
        """Create a readable description of the recorder search target."""
        if raw_event:
            return str(raw_event).replace("_", " ").strip()
        if isinstance(raw_state, list):
            return " / ".join(str(value).replace("_", " ").strip() for value in raw_state)
        if raw_state:
            return str(raw_state).replace("_", " ").strip()
        return "change"

    def _parse_history_datetime(self, value: Any):
        """Parse a recorder lookup datetime and assume local time if naive."""
        if value is None:
            return None

        parsed = dt_util.parse_datetime(str(value))
        if parsed is None:
            return None

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=getattr(dt_util, "DEFAULT_TIME_ZONE", dt_util.now().tzinfo)
            )

        return dt_util.as_utc(parsed)

    def _build_history_search_windows(self, max_hours: int) -> List[int]:
        """Build progressively larger recorder search windows."""
        windows = [24, 168, 720, max_hours]
        deduped: List[int] = []
        seen = set()

        for window in windows:
            clamped = min(window, max_hours)
            if clamped > 0 and clamped not in seen:
                seen.add(clamped)
                deduped.append(clamped)

        return deduped

    async def _fetch_entity_history_states(
        self,
        entity_id: str,
        hours: int | None = None,
        *,
        start_time=None,
        end_time=None,
        descending: bool = True,
        limit: int | None = None,
        include_start_time_state: bool = False,
    ) -> List[Any]:
        """Fetch recorder-backed history states for a single entity."""
        query_end_time = end_time or dt_util.utcnow()
        query_start_time = start_time
        if query_start_time is None:
            if hours is None:
                raise ValueError("Either hours or start_time must be provided")
            query_start_time = query_end_time - timedelta(hours=hours)

        states = await self.hass.async_add_executor_job(
            lambda: history.state_changes_during_period(
                self.hass,
                query_start_time,
                end_time=query_end_time,
                entity_id=entity_id,
                no_attributes=True,
                descending=descending,
                limit=limit,
                include_start_time_state=include_start_time_state,
            )
        )
        return self._extract_history_states(states, entity_id)
