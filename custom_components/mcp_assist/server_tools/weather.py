"""Weather MCP server tools."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Tuple

from homeassistant.util import dt as dt_util


class WeatherToolsMixin:
    """Weather MCP server tool implementations."""

    async def tool_get_weather_forecast(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get a Home Assistant weather forecast in one call."""
        target_entity_id = str(args.get("entity_id") or "").strip() or None
        area = str(args.get("area") or "").strip() or None
        floor = str(args.get("floor") or "").strip() or None
        label = str(args.get("label") or "").strip() or None
        name_contains = str(args.get("name_contains") or "").strip() or None
        when_value = str(args.get("when") or "tomorrow").strip() or "tomorrow"
        forecast_type = str(args.get("forecast_type") or "").strip().lower() or None
        timeout = self._coerce_int_arg(
            args.get("timeout"), default=60, minimum=1, maximum=300
        )

        target_date, day_label, parse_error = self._parse_weather_forecast_when(
            when_value
        )
        if parse_error:
            return {
                "content": [{"type": "text", "text": f"❌ Error: {parse_error}"}]
            }

        try:
            resolved_target, entity_info = await self._resolve_weather_forecast_target(
                entity_id=target_entity_id,
                area=area,
                floor=floor,
                label=label,
                name_contains=name_contains,
            )
        except ValueError as err:
            return {"content": [{"type": "text", "text": f"❌ Error: {err}"}]}

        prepared_data = self._prepare_response_service_data(
            "weather",
            "get_forecasts",
            {"type": forecast_type} if forecast_type else {},
            resolved_target=resolved_target,
        )

        service_result = await self.tool_call_service_with_response(
            {
                "domain": "weather",
                "service": "get_forecasts",
                "target": resolved_target,
                "data": prepared_data,
                "timeout": timeout,
            }
        )

        if service_result.get("response") is None:
            return service_result

        response = service_result.get("response")
        if not isinstance(response, dict):
            return service_result

        payload = response.get(entity_info["entity_id"])
        forecast_entries = payload.get("forecast") if isinstance(payload, dict) else None
        if not isinstance(forecast_entries, list):
            return service_result

        summary_text = self._summarize_requested_weather_forecast(
            entity_name=entity_info["name"],
            entity_id=entity_info["entity_id"],
            forecast_entries=forecast_entries,
            forecast_type=prepared_data.get("type"),
            target_date=target_date,
            day_label=day_label,
        )
        if not summary_text:
            return service_result

        result: Dict[str, Any] = {
            "content": [{"type": "text", "text": summary_text}],
            "response": response,
        }
        return result

    async def _resolve_weather_forecast_target(
        self,
        *,
        entity_id: str | None = None,
        area: str | None = None,
        floor: str | None = None,
        label: str | None = None,
        name_contains: str | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, str]]:
        """Resolve the best weather entity target for a forecast request."""
        if entity_id:
            resolved_target = await self.resolve_target({"entity_id": entity_id})
            resolved_target = self._restrict_resolved_target_to_domain(
                resolved_target, "weather"
            )
            selected_entity_id = self._normalize_target_values(
                resolved_target.get("entity_id")
            )[0]
            state = self.hass.states.get(selected_entity_id)
            friendly_name = (
                state.attributes.get("friendly_name")
                if state and state.attributes.get("friendly_name")
                else selected_entity_id
            )
            return resolved_target, {
                "entity_id": selected_entity_id,
                "name": str(friendly_name),
            }

        candidates = await self.discovery.discover_entities(
            domain="weather",
            area=area,
            floor=floor,
            label=label,
            name_contains=name_contains,
            limit=10,
        )
        if not candidates:
            raise ValueError(
                "No exposed weather entity was found in Home Assistant. "
                "Try discover_entities(domain='weather') to inspect available weather entities."
            )

        selected = sorted(
            candidates,
            key=self._weather_entity_candidate_sort_key,
        )[0]
        return {"entity_id": [selected["entity_id"]]}, {
            "entity_id": str(selected["entity_id"]),
            "name": str(selected.get("name") or selected["entity_id"]),
        }

    @staticmethod
    def _weather_entity_candidate_sort_key(entity: Dict[str, Any]) -> Tuple[int, int, str, str]:
        """Rank weather entity candidates with stable generic preferences."""
        name = str(entity.get("name") or "").casefold()
        entity_id = str(entity.get("entity_id") or "").casefold()
        service_hint = 0 if entity.get("forecast_service_supported") else 1
        attribute_hint = 0 if entity.get("forecast_available") else 1
        return (service_hint, attribute_hint, name or entity_id, entity_id)

    def _parse_weather_forecast_when(
        self, when_value: str | None
    ) -> tuple[date | None, str, str | None]:
        """Parse a weather forecast day request."""
        now_local = dt_util.as_local(dt_util.now())
        when_text = str(when_value or "tomorrow").strip().casefold()
        if not when_text or when_text == "tomorrow":
            return now_local.date() + timedelta(days=1), "tomorrow", None
        if when_text == "today":
            return now_local.date(), "today", None

        parsed_date = dt_util.parse_date(when_text)
        if parsed_date is not None:
            return parsed_date, self._format_weather_day_label(parsed_date), None

        parsed_datetime = dt_util.parse_datetime(when_text)
        if parsed_datetime is not None:
            if parsed_datetime.tzinfo is None:
                parsed_datetime = parsed_datetime.replace(
                    tzinfo=getattr(dt_util, "DEFAULT_TIME_ZONE", now_local.tzinfo)
                )
            local_when = dt_util.as_local(parsed_datetime)
            return local_when.date(), self._format_weather_day_label(local_when.date()), None

        return None, when_text, (
            "Unable to parse 'when'. Use 'today', 'tomorrow', or a local date like '2026-04-13'."
        )

    def _format_weather_day_label(self, target_date: date) -> str:
        """Format a forecast day label for user-facing summaries."""
        today = dt_util.as_local(dt_util.now()).date()
        if target_date == today:
            return "today"
        if target_date == today + timedelta(days=1):
            return "tomorrow"
        return target_date.strftime("%A, %b %d").replace(" 0", " ")

    def _extract_weather_entries_for_date(
        self,
        forecast_entries: List[Dict[str, Any]],
        *,
        target_date: date,
        forecast_type: str | None,
    ) -> List[Dict[str, Any]]:
        """Return forecast entries that map to the requested local date."""
        matching: List[tuple[datetime | None, Dict[str, Any]]] = []
        for entry in forecast_entries:
            entry_date = self._parse_weather_forecast_entry_date(entry)
            if entry_date and entry_date == target_date:
                matching.append((self._parse_weather_forecast_entry_datetime(entry), entry))

        if matching:
            matching.sort(key=lambda item: item[0] or datetime.min.replace(tzinfo=dt_util.UTC))
            return [entry for _, entry in matching]

        today = dt_util.as_local(dt_util.now()).date()
        day_offset = (target_date - today).days
        if day_offset < 0:
            return []

        if forecast_type == "daily" and day_offset < len(forecast_entries):
            return [forecast_entries[day_offset]]

        if forecast_type == "twice_daily":
            start = day_offset * 2
            if start < len(forecast_entries):
                return forecast_entries[start : start + 2]

        if forecast_type == "hourly":
            start = day_offset * 24
            if start < len(forecast_entries):
                return forecast_entries[start : start + 24]

        return []

    def _parse_weather_forecast_entry_datetime(
        self, entry: Dict[str, Any]
    ) -> datetime | None:
        """Parse a weather forecast entry datetime to local time."""
        raw_value = entry.get("datetime")
        if raw_value is None:
            raw_value = entry.get("date")
        if raw_value is None:
            return None

        parsed_datetime = dt_util.parse_datetime(str(raw_value))
        if parsed_datetime is not None:
            if parsed_datetime.tzinfo is None:
                parsed_datetime = parsed_datetime.replace(
                    tzinfo=getattr(dt_util, "DEFAULT_TIME_ZONE", dt_util.now().tzinfo)
                )
            return dt_util.as_local(parsed_datetime)

        parsed_date = dt_util.parse_date(str(raw_value))
        if parsed_date is None:
            return None

        local_tz = getattr(dt_util, "DEFAULT_TIME_ZONE", dt_util.now().tzinfo)
        return datetime.combine(parsed_date, time.min, tzinfo=local_tz)

    def _parse_weather_forecast_entry_date(
        self, entry: Dict[str, Any]
    ) -> date | None:
        """Parse just the local date for a weather forecast entry."""
        parsed_datetime = self._parse_weather_forecast_entry_datetime(entry)
        if parsed_datetime is not None:
            return parsed_datetime.date()
        return None

    def _summarize_requested_weather_forecast(
        self,
        *,
        entity_name: str,
        entity_id: str,
        forecast_entries: List[Dict[str, Any]],
        forecast_type: str | None,
        target_date: date,
        day_label: str,
    ) -> str | None:
        """Summarize a requested weather forecast day from returned forecast data."""
        day_entries = self._extract_weather_entries_for_date(
            forecast_entries,
            target_date=target_date,
            forecast_type=forecast_type,
        )
        if not day_entries:
            return None

        intro = f"{day_label.capitalize()} for {entity_name}:"
        if forecast_type == "twice_daily":
            parts = []
            for entry in day_entries[:2]:
                part_label = self._describe_weather_forecast_part(entry)
                parts.append(f"{part_label}: {self._format_weather_forecast_entry(entry)}")
            return f"{intro} {'; '.join(parts)}."

        if forecast_type == "hourly" and len(day_entries) > 2:
            conditions = [
                str(entry.get("condition")).replace("_", " ")
                for entry in day_entries
                if entry.get("condition")
            ]
            temps = [
                self._coerce_weather_temperature(entry.get("temperature"))
                for entry in day_entries
            ]
            temps = [temp for temp in temps if temp is not None]
            dominant_condition = None
            if conditions:
                dominant_condition = Counter(conditions).most_common(1)[0][0]

            detail_parts = []
            if dominant_condition:
                detail_parts.append(dominant_condition)
            if temps:
                detail_parts.append(
                    f"high {self._format_weather_temperature(max(temps))}, low {self._format_weather_temperature(min(temps))}"
                )
            preview_entries = [
                f"{self._describe_weather_forecast_part(entry)} {self._format_weather_forecast_entry(entry)}"
                for entry in day_entries[:3]
            ]
            if preview_entries:
                detail_parts.append("early outlook: " + "; ".join(preview_entries))
            return f"{intro} {'. '.join(detail_parts)}."

        return f"{intro} {self._format_weather_forecast_entry(day_entries[0])}."

    def _describe_weather_forecast_part(self, entry: Dict[str, Any]) -> str:
        """Describe a weather forecast segment like morning/evening/time."""
        parsed_datetime = self._parse_weather_forecast_entry_datetime(entry)
        if parsed_datetime is not None:
            hour = parsed_datetime.hour
            if entry.get("is_daytime") is False or hour >= 18:
                return "evening"
            if 5 <= hour < 12:
                return "morning"
            if 12 <= hour < 18:
                return "afternoon"
            return parsed_datetime.strftime("%-I:%M %p")

        if entry.get("is_daytime") is True:
            return "daytime"
        if entry.get("is_daytime") is False:
            return "night"
        return "forecast"

    def _format_weather_forecast_entry(self, entry: Dict[str, Any]) -> str:
        """Format a single weather forecast entry compactly."""
        parts = []
        condition = entry.get("condition")
        if condition:
            parts.append(str(condition).replace("_", " "))

        high_temp = self._coerce_weather_temperature(entry.get("temperature"))
        low_temp = self._coerce_weather_temperature(entry.get("templow"))
        if high_temp is not None and low_temp is not None:
            parts.append(
                f"high {self._format_weather_temperature(high_temp)}, low {self._format_weather_temperature(low_temp)}"
            )
        elif high_temp is not None:
            parts.append(f"around {self._format_weather_temperature(high_temp)}")

        precipitation_probability = entry.get("precipitation_probability")
        if precipitation_probability not in (None, ""):
            parts.append(f"{precipitation_probability}% chance of rain")

        wind_speed = entry.get("wind_speed")
        if wind_speed not in (None, ""):
            parts.append(f"wind {wind_speed}")

        return ", ".join(parts) if parts else "forecast available"

    @staticmethod
    def _coerce_weather_temperature(value: Any) -> float | int | None:
        """Coerce a weather temperature value safely."""
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return value
        try:
            numeric = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        return int(numeric) if numeric.is_integer() else round(numeric, 1)

    @staticmethod
    def _format_weather_temperature(value: float | int) -> str:
        """Format a compact temperature value."""
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value)

    def _get_weather_target_forecast_types(
        self, resolved_target: Dict[str, Any] | None
    ) -> List[str]:
        """Return forecast types supported by all targeted weather entities."""
        entity_ids = self._normalize_target_values((resolved_target or {}).get("entity_id"))
        if not entity_ids:
            return []

        common_types: set[str] | None = None
        for entity_id in entity_ids:
            state = self.hass.states.get(entity_id)
            if state is None or state.domain != "weather":
                continue

            forecast_types = set(self.discovery._get_weather_forecast_types(state))
            if not forecast_types:
                continue

            common_types = (
                forecast_types
                if common_types is None
                else common_types.intersection(forecast_types)
            )

        if not common_types:
            return []

        return [
            forecast_type
            for forecast_type in ("daily", "twice_daily", "hourly")
            if forecast_type in common_types
        ]

    def _select_preferred_weather_forecast_type(
        self, forecast_types: List[str] | set[str] | tuple[str, ...]
    ) -> str | None:
        """Choose a stable forecast-type fallback for weather responses."""
        forecast_type_set = {str(item) for item in forecast_types if item}
        for candidate in ("daily", "twice_daily", "hourly"):
            if candidate in forecast_type_set:
                return candidate
        return next(iter(sorted(forecast_type_set)), None)

    def _summarize_weather_response(self, response: Any) -> List[str]:
        """Summarize weather forecast response data."""
        if not isinstance(response, dict):
            return []

        lines = ["Summary:"]
        for entity_id, payload in response.items():
            forecast = payload.get("forecast") if isinstance(payload, dict) else None
            if not isinstance(forecast, list):
                lines.append(f"- {entity_id}: no forecast entries returned")
                continue

            detail_parts = [f"{len(forecast)} forecast entries"]
            if forecast:
                first_forecast = forecast[0] if isinstance(forecast[0], dict) else {}
                preview_parts = []
                preview_time = self._format_service_response_datetime(
                    first_forecast.get("datetime")
                )
                if preview_time:
                    preview_parts.append(preview_time)
                if first_forecast.get("condition"):
                    preview_parts.append(str(first_forecast["condition"]))
                high_temp = first_forecast.get("temperature")
                low_temp = first_forecast.get("templow")
                if high_temp is not None and low_temp is not None:
                    preview_parts.append(f"{high_temp}/{low_temp}")
                elif high_temp is not None:
                    preview_parts.append(str(high_temp))
                if preview_parts:
                    detail_parts.append("first: " + ", ".join(preview_parts))

            lines.append(f"- {entity_id}: {'; '.join(detail_parts)}")

        return lines if len(lines) > 1 else []
