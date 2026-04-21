"""Calendar MCP server tools."""

from __future__ import annotations

from datetime import datetime, time, timedelta
import re
from typing import Any, Dict, List

from homeassistant.util import dt as dt_util


class CalendarToolsMixin:
    """Calendar MCP server tool implementations."""

    async def tool_get_calendar_events(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get upcoming calendar events in one call."""
        target_entity_id = str(args.get("entity_id") or "").strip() or None
        area = str(args.get("area") or "").strip() or None
        floor = str(args.get("floor") or "").strip() or None
        label = str(args.get("label") or "").strip() or None
        query = str(args.get("query") or "").strip() or None
        name_contains = str(args.get("name_contains") or "").strip() or None
        event_text = str(args.get("event_text") or "").strip() or None
        when_value = str(args.get("when") or "").strip() or None
        days = self._coerce_int_arg(
            args.get("days"),
            default=60,
            minimum=1,
            maximum=365,
        )
        limit = self._coerce_int_arg(
            args.get("limit"),
            default=5,
            minimum=1,
            maximum=20,
        )
        timeout = self._coerce_int_arg(
            args.get("timeout"), default=60, minimum=1, maximum=300
        )

        try:
            (
                resolved_target,
                selected_calendars,
                fallback_used,
            ) = await self._resolve_calendar_event_targets(
                entity_id=target_entity_id,
                area=area,
                floor=floor,
                label=label,
                query=query,
                name_contains=name_contains,
            )
        except ValueError as err:
            return self._build_text_tool_result(str(err), is_error=True)

        window_start, window_end, window_label = self._build_calendar_search_window(
            when_value=when_value,
            days=days,
        )
        if window_start is None or window_end is None:
            return self._build_text_tool_result(
                f"Invalid calendar time window: {window_label}",
                is_error=True,
            )

        service_result = await self.tool_call_service_with_response(
            {
                "domain": "calendar",
                "service": "get_events",
                "target": resolved_target,
                "data": {
                    "start_date_time": window_start.isoformat(),
                    "end_date_time": window_end.isoformat(),
                },
                "timeout": timeout,
            }
        )

        response = service_result.get("response")
        if not isinstance(response, dict):
            return service_result

        active_event_filter = event_text
        if not active_event_filter and fallback_used and query:
            active_event_filter = query

        matches = self._collect_calendar_event_matches(
            response=response,
            selected_calendars=selected_calendars,
            event_text=active_event_filter,
        )
        serialized_matches = self._serialize_service_response_value(matches[:limit])
        if not matches:
            filter_label = active_event_filter or name_contains or query or "the request"
            calendars_label = (
                f"{len(selected_calendars)} calendar"
                f"{'' if len(selected_calendars) == 1 else 's'}"
            )
            return self._build_text_tool_result(
                (
                    f"No upcoming calendar events matched {filter_label!r} in {window_label}. "
                    f"Searched {calendars_label}."
                ),
                structured_content={
                    "query": query,
                    "name_contains": name_contains,
                    "event_text": active_event_filter,
                    "window": {
                        "start": window_start.isoformat(),
                        "end": window_end.isoformat(),
                        "label": window_label,
                    },
                    "selected_calendars": selected_calendars,
                    "events": [],
                },
            )

        summary_text = self._summarize_calendar_matches(
            matches=matches[:limit],
            query=query,
            event_text=active_event_filter,
            window_label=window_label,
        )
        return self._build_text_tool_result(
            summary_text,
            structured_content={
                "query": query,
                "name_contains": name_contains,
                "event_text": active_event_filter,
                "window": {
                    "start": window_start.isoformat(),
                    "end": window_end.isoformat(),
                    "label": window_label,
                },
                "selected_calendars": selected_calendars,
                "events": serialized_matches,
                "total_matching_events": len(matches),
            },
        )

    async def _resolve_calendar_event_targets(
        self,
        *,
        entity_id: str | None = None,
        area: str | None = None,
        floor: str | None = None,
        label: str | None = None,
        query: str | None = None,
        name_contains: str | None = None,
    ) -> tuple[Dict[str, Any], list[Dict[str, str]], bool]:
        """Resolve one or more calendar targets for a search."""
        if entity_id:
            resolved_target = await self.resolve_target({"entity_id": entity_id})
            resolved_target = self._restrict_resolved_target_to_domain(
                resolved_target, "calendar"
            )
            entity_ids = self._normalize_target_values(resolved_target.get("entity_id"))
            selected = [
                {
                    "entity_id": candidate_entity_id,
                    "name": self._friendly_entity_name(candidate_entity_id),
                }
                for candidate_entity_id in entity_ids
            ]
            return resolved_target, selected, False

        explicit_name_filter = bool(name_contains)
        candidates: list[Dict[str, Any]] = []
        name_filters = self._build_calendar_name_filters(
            query=query,
            name_contains=name_contains,
        )
        for discovery_name in name_filters:
            candidates = await self.discovery.discover_entities(
                domain="calendar",
                area=area,
                floor=floor,
                label=label,
                name_contains=discovery_name,
                limit=25,
            )
            if candidates:
                break
        fallback_used = False
        if not candidates and query and not explicit_name_filter:
            candidates = await self.discovery.discover_entities(
                domain="calendar",
                area=area,
                floor=floor,
                label=label,
                limit=25,
            )
            fallback_used = True

        if not candidates:
            if explicit_name_filter:
                raise ValueError(
                    "No exposed calendar matched that name. "
                    "Try discover_entities(domain='calendar') to inspect available calendars."
                )
            raise ValueError(
                "No exposed calendar was found in Home Assistant. "
                "Try discover_entities(domain='calendar') to inspect available calendars."
            )

        selected = [
            {
                "entity_id": str(candidate["entity_id"]),
                "name": str(candidate.get("name") or candidate["entity_id"]),
            }
            for candidate in candidates
        ]
        selected.sort(key=lambda item: item["name"].casefold())
        return {"entity_id": [item["entity_id"] for item in selected]}, selected, fallback_used

    def _build_calendar_name_filters(
        self,
        *,
        query: str | None,
        name_contains: str | None,
    ) -> list[str | None]:
        """Build progressively broader calendar-name filters before event-text fallback."""
        explicit_name = str(name_contains or "").strip()
        if explicit_name:
            return [explicit_name]

        raw_query = str(query or "").strip()
        if not raw_query:
            return [None]

        filters: list[str | None] = [raw_query]
        generic_words = {
            "calendar",
            "event",
            "events",
            "game",
            "games",
            "match",
            "matches",
            "schedule",
            "schedules",
            "next",
            "upcoming",
        }
        query_tokens = re.findall(r"[a-z0-9']+", raw_query.casefold())
        simplified_tokens = [token for token in query_tokens if token not in generic_words]
        simplified_query = " ".join(simplified_tokens).strip()
        if simplified_query and simplified_query != raw_query.casefold():
            filters.append(simplified_query)

        return filters

    def _build_calendar_search_window(
        self,
        *,
        when_value: str | None,
        days: int,
    ) -> tuple[datetime | None, datetime | None, str]:
        """Build the UTC search window for a calendar query."""
        local_tz = getattr(dt_util, "DEFAULT_TIME_ZONE", dt_util.now().tzinfo)
        now_local = dt_util.as_local(dt_util.utcnow())
        raw_when = (when_value or "").strip()
        normalized = raw_when.casefold()

        if not raw_when or normalized == "now":
            window_start = dt_util.utcnow()
            window_end = window_start + timedelta(days=days)
            return window_start, window_end, f"the next {days} day{'s' if days != 1 else ''}"

        if normalized == "today":
            start_local = datetime.combine(now_local.date(), time.min, tzinfo=local_tz)
            end_local = start_local + timedelta(days=1)
            return dt_util.as_utc(start_local), dt_util.as_utc(end_local), "today"

        if normalized == "tomorrow":
            start_local = datetime.combine(
                now_local.date() + timedelta(days=1),
                time.min,
                tzinfo=local_tz,
            )
            end_local = start_local + timedelta(days=1)
            return dt_util.as_utc(start_local), dt_util.as_utc(end_local), "tomorrow"

        parsed_date = dt_util.parse_date(raw_when)
        if parsed_date is not None:
            start_local = datetime.combine(parsed_date, time.min, tzinfo=local_tz)
            end_local = start_local + timedelta(days=1)
            return dt_util.as_utc(start_local), dt_util.as_utc(end_local), raw_when

        parsed_datetime = dt_util.parse_datetime(raw_when)
        if parsed_datetime is None:
            return None, None, raw_when
        if parsed_datetime.tzinfo is None:
            parsed_datetime = parsed_datetime.replace(tzinfo=local_tz)
        window_start = dt_util.as_utc(parsed_datetime)
        window_end = window_start + timedelta(days=days)
        return window_start, window_end, raw_when

    def _collect_calendar_event_matches(
        self,
        *,
        response: Dict[str, Any],
        selected_calendars: list[Dict[str, str]],
        event_text: str | None,
    ) -> list[Dict[str, Any]]:
        """Collect matching calendar events sorted by upcoming start time."""
        selected_by_entity = {
            item["entity_id"]: item["name"] for item in selected_calendars
        }
        matches: list[Dict[str, Any]] = []
        for entity_id, payload in response.items():
            events = payload.get("events") if isinstance(payload, dict) else None
            if not isinstance(events, list):
                continue
            calendar_name = selected_by_entity.get(entity_id, entity_id)
            for event in events:
                if not isinstance(event, dict):
                    continue
                if event_text and not self._calendar_event_matches_text(event, event_text):
                    continue
                start_at = self._parse_calendar_event_start_datetime(event)
                if start_at is None:
                    continue
                matches.append(
                    {
                        "calendar_entity_id": entity_id,
                        "calendar_name": calendar_name,
                        "summary": str(
                            event.get("summary")
                            or event.get("title")
                            or "Untitled event"
                        ),
                        "description": str(event.get("description") or "").strip(),
                        "location": str(event.get("location") or "").strip(),
                        "start": self._serialize_service_response_value(
                            self._extract_calendar_event_start_value(event)
                        ),
                        "end": self._serialize_service_response_value(event.get("end")),
                        "all_day": self._calendar_event_is_all_day(event),
                        "start_at": start_at,
                    }
                )
        matches.sort(key=lambda item: item["start_at"])
        return matches

    def _summarize_calendar_matches(
        self,
        *,
        matches: list[Dict[str, Any]],
        query: str | None,
        event_text: str | None,
        window_label: str,
    ) -> str:
        """Summarize one or more calendar event matches."""
        if len(matches) == 1:
            event = matches[0]
            qualifier = "matching calendar event" if (query or event_text) else "calendar event"
            return (
                f"Next {qualifier}: {self._describe_calendar_match(event)}."
            )

        lead = "Upcoming matching calendar events" if (query or event_text) else "Upcoming calendar events"
        lines = [f"{lead} in {window_label}:"]
        for event in matches:
            lines.append(f"- {self._describe_calendar_match(event)}")
        return "\n".join(lines)

    def _describe_calendar_match(self, event: Dict[str, Any]) -> str:
        """Describe one calendar event compactly."""
        summary = str(event.get("summary") or "Untitled event")
        calendar_name = str(event.get("calendar_name") or event.get("calendar_entity_id") or "")
        when = self._format_calendar_event_when(event)
        parts = [summary]
        if calendar_name and calendar_name.casefold() not in summary.casefold():
            parts.append(f"on {calendar_name}")
        if when:
            parts.append(when)
        location = str(event.get("location") or "").strip()
        if location:
            parts.append(f"at {location}")
        return ", ".join(parts)

    def _format_calendar_event_when(self, event: Dict[str, Any]) -> str:
        """Format a calendar event start in friendly upcoming language."""
        start_at = event.get("start_at")
        if not isinstance(start_at, datetime):
            return ""
        if event.get("all_day"):
            return self._format_calendar_all_day(start_at)
        return self._format_future_absolute_time(start_at)

    def _format_calendar_all_day(self, when: datetime) -> str:
        """Format an all-day calendar date."""
        local_when = dt_util.as_local(when)
        today_local = dt_util.as_local(dt_util.utcnow()).date()
        day_delta = (local_when.date() - today_local).days
        if day_delta == 0:
            return "all day today"
        if day_delta == 1:
            return "all day tomorrow"
        if day_delta > 1 and day_delta < 7:
            return f"all day in {day_delta} days"
        date_text = local_when.strftime("%b %d").replace(" 0", " ")
        if local_when.year != today_local.year:
            date_text += f", {local_when.year}"
        return f"all day on {date_text}"

    def _format_future_absolute_time(self, when: datetime) -> str:
        """Format a future timestamp with relative and absolute context."""
        now = dt_util.utcnow()
        delta_seconds = (when - now).total_seconds()
        absolute = self._format_absolute_time(when)
        if delta_seconds <= 0:
            return absolute
        if delta_seconds < 3600:
            minutes = max(1, int(delta_seconds / 60))
            return f"in {minutes} minute{'s' if minutes != 1 else ''} at {absolute}"
        if delta_seconds < 86400:
            hours = max(1, int(delta_seconds / 3600))
            return f"in {hours} hour{'s' if hours != 1 else ''} at {absolute}"
        if delta_seconds < 604800:
            days = max(1, int(delta_seconds / 86400))
            return f"in {days} day{'s' if days != 1 else ''} at {absolute}"
        weeks = max(1, int(delta_seconds / 604800))
        return f"in {weeks} week{'s' if weeks != 1 else ''} at {absolute}"

    def _calendar_event_matches_text(self, event: Dict[str, Any], text: str) -> bool:
        """Return whether a calendar event matches free text."""
        query = text.casefold().strip()
        if not query:
            return True
        haystack = " ".join(
            str(event.get(key) or "").casefold()
            for key in ("summary", "title", "description", "location")
        )
        return query in haystack

    def _calendar_event_is_all_day(self, event: Dict[str, Any]) -> bool:
        """Return whether a calendar event is all-day."""
        start_value = event.get("start")
        if isinstance(start_value, dict):
            return bool(start_value.get("date")) and not bool(
                start_value.get("dateTime") or start_value.get("datetime")
            )
        if isinstance(start_value, str):
            return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", start_value.strip()))
        return False

    def _extract_calendar_event_start_value(self, event: Dict[str, Any]) -> Any:
        """Extract the raw calendar event start field."""
        start_value = event.get("start")
        if isinstance(start_value, dict):
            return (
                start_value.get("dateTime")
                or start_value.get("datetime")
                or start_value.get("date")
            )
        return start_value

    def _parse_calendar_event_start_datetime(self, event: Dict[str, Any]) -> datetime | None:
        """Parse a calendar event start into UTC for sorting."""
        start_value = self._extract_calendar_event_start_value(event)
        local_tz = getattr(dt_util, "DEFAULT_TIME_ZONE", dt_util.now().tzinfo)
        if self._calendar_event_is_all_day(event):
            parsed_date = dt_util.parse_date(str(start_value))
            if parsed_date is None:
                return None
            return dt_util.as_utc(
                datetime.combine(parsed_date, time.min, tzinfo=local_tz)
            )

        parsed_datetime = dt_util.parse_datetime(str(start_value))
        if parsed_datetime is None:
            return None
        if parsed_datetime.tzinfo is None:
            parsed_datetime = parsed_datetime.replace(tzinfo=local_tz)
        return dt_util.as_utc(parsed_datetime)

    def _friendly_entity_name(self, entity_id: str) -> str:
        """Return a friendly entity name when available."""
        state = self.hass.states.get(entity_id)
        if state is not None and state.attributes.get("friendly_name"):
            return str(state.attributes["friendly_name"])
        return entity_id

    def _summarize_calendar_response(self, response: Any) -> List[str]:
        """Summarize calendar event response data."""
        if not isinstance(response, dict):
            return []

        lines = ["Summary:"]
        for entity_id, payload in response.items():
            events = payload.get("events") if isinstance(payload, dict) else None
            if not isinstance(events, list):
                lines.append(f"- {entity_id}: no events returned")
                continue

            detail_parts = [f"{len(events)} events"]
            if events:
                first_event = events[0] if isinstance(events[0], dict) else {}
                event_summary = first_event.get("summary") or first_event.get("title")
                event_start = self._extract_calendar_event_start(first_event)
                preview_parts = []
                if event_summary:
                    preview_parts.append(str(event_summary))
                if event_start:
                    preview_parts.append(event_start)
                if preview_parts:
                    detail_parts.append("next: " + " at ".join(preview_parts[:2]))

            lines.append(f"- {entity_id}: {'; '.join(detail_parts)}")

        return lines if len(lines) > 1 else []

    def _extract_calendar_event_start(self, event: Dict[str, Any]) -> str | None:
        """Extract and format the start time from a calendar event payload."""
        if not isinstance(event, dict):
            return None

        start_value = event.get("start")
        if isinstance(start_value, dict):
            start_value = (
                start_value.get("dateTime")
                or start_value.get("datetime")
                or start_value.get("date")
            )

        return self._format_service_response_datetime(start_value)
