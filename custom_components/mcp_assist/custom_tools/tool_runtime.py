"""Shared runtime helpers for built-in Home Assistant tool packages."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List

from homeassistant.components.homeassistant import async_should_expose
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)
from homeassistant.util import dt as dt_util

from ..const import DOMAIN
from ..discovery import EntityDiscovery

try:
    from homeassistant.helpers import floor_registry as fr
except ImportError:  # pragma: no cover - older Home Assistant versions
    fr = None


class HomeAssistantToolRuntime:
    """Common Home Assistant helpers for vendored built-in tools."""

    def __init__(self, hass) -> None:
        """Initialize the runtime."""
        self.hass = hass
        self.discovery = EntityDiscovery(hass)

    @property
    def _server(self) -> Any | None:
        """Return the shared MCP server when the integration has finished startup."""
        return self.hass.data.get(DOMAIN, {}).get("shared_mcp_server")

    def publish_progress(self, stage: str, message: str, **payload: Any) -> None:
        """Forward progress updates to the shared MCP server when available."""
        server = self._server
        publish_progress = getattr(server, "publish_progress", None) if server else None
        if callable(publish_progress):
            publish_progress(stage, message, **payload)

    def _get_domain_capability_error(self, domain: str) -> str | None:
        """Delegate domain capability checks to the shared MCP server when available."""
        server = self._server
        checker = getattr(server, "_get_domain_capability_error", None) if server else None
        if callable(checker):
            return checker(domain)
        return None

    @staticmethod
    def _build_text_tool_result(
        text: str,
        *,
        is_error: bool = False,
        structured_content: dict[str, Any] | None = None,
        extra_content: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a standard MCP text result with optional structured content."""
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if extra_content:
            content.extend(extra_content)

        result: dict[str, Any] = {"content": content, "isError": is_error}
        if structured_content is not None:
            result["structuredContent"] = structured_content
        return result

    @staticmethod
    def _coerce_int_arg(
        value: Any, *, default: int, minimum: int, maximum: int
    ) -> int:
        """Coerce an integer-like tool argument safely."""
        if value is None:
            parsed = default
        elif isinstance(value, bool):
            parsed = default
        elif isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            parsed = int(value)
        else:
            try:
                parsed = int(str(value).strip())
            except (TypeError, ValueError):
                parsed = default

        return max(minimum, min(parsed, maximum))

    @staticmethod
    def _normalize_target_values(value: Any) -> List[str]:
        """Normalize scalar or list target selector values to unique strings."""
        if value is None:
            return []

        if isinstance(value, str):
            raw_values = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raw_values = [value]

        normalized = []
        seen = set()
        for item in raw_values:
            if item is None:
                continue
            item_text = str(item).strip()
            if not item_text or item_text in seen:
                continue
            seen.add(item_text)
            normalized.append(item_text)

        return normalized

    def _friendly_names_for_entities(self, entity_ids: List[str]) -> List[str]:
        """Resolve entity IDs to friendly names."""
        names = []
        for entity_id in entity_ids:
            state = self.hass.states.get(entity_id)
            if state and state.name:
                names.append(state.name)
            else:
                names.append(entity_id)
        return names

    def _friendly_entity_name(self, entity_id: str) -> str:
        """Return a friendly entity name when available."""
        state = self.hass.states.get(entity_id)
        if state is not None and state.attributes.get("friendly_name"):
            return str(state.attributes["friendly_name"])
        return entity_id

    def _format_relative_time(self, when) -> str:
        """Format a timestamp relative to now."""
        now = dt_util.utcnow()
        seconds = max((now - when).total_seconds(), 0)

        if seconds < 60:
            return "just now"
        if seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        if seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        if seconds < 604800:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"

        weeks = int(seconds / 604800)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"

    def _format_absolute_time(self, when) -> str:
        """Format a timestamp in the user's local time zone."""
        local_when = dt_util.as_local(when)
        now_local = dt_util.as_local(dt_util.utcnow())
        time_text = local_when.strftime("%I:%M %p %Z").lstrip("0")

        if local_when.date() == now_local.date():
            day_text = "today"
        elif local_when.date() == now_local.date() - timedelta(days=1):
            day_text = "yesterday"
        elif local_when.date() == now_local.date() + timedelta(days=1):
            day_text = "tomorrow"
        else:
            date_text = local_when.strftime("%b %d").replace(" 0", " ")
            if local_when.year != now_local.year:
                date_text += f", {local_when.year}"
            day_text = f"on {date_text}"

        return f"{time_text} {day_text}"

    def _format_relative_absolute_time(self, when) -> str:
        """Format a timestamp with both relative and absolute local time."""
        relative = self._format_relative_time(when)
        absolute = self._format_absolute_time(when)
        return f"{relative} at {absolute}"

    async def resolve_target(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve target selectors to exposed entity IDs."""
        explicit_entity_ids = self._normalize_target_values(target.get("entity_id"))
        selector_values = {
            "area_id": self._normalize_target_values(target.get("area_id")),
            "floor_id": self._normalize_target_values(target.get("floor_id")),
            "label_id": self._normalize_target_values(target.get("label_id")),
            "device_id": self._normalize_target_values(target.get("device_id")),
        }
        active_selectors = {
            key: values for key, values in selector_values.items() if values
        }

        resolved_entities = set()
        invalid_entity_ids = []
        for entity_id in explicit_entity_ids:
            state = self.hass.states.get(entity_id)
            if state is None:
                invalid_entity_ids.append(f"{entity_id} (not found)")
                continue
            if not async_should_expose(self.hass, "conversation", entity_id):
                invalid_entity_ids.append(f"{entity_id} (not exposed to conversation)")
                continue
            resolved_entities.add(entity_id)

        if invalid_entity_ids:
            raise ValueError("Invalid entity targets: " + ", ".join(invalid_entity_ids))

        if active_selectors:
            selector_matches = self._find_exposed_entities_for_target(active_selectors)
            selector_sets = []

            for selector_key, selector_ids in active_selectors.items():
                matched_entities = selector_matches.get(selector_key, set())
                if not matched_entities:
                    raise ValueError(
                        "No exposed conversation entities matched "
                        f"{selector_key}: {', '.join(selector_ids)}"
                    )
                selector_sets.append(matched_entities)

            combined_matches = set.intersection(*selector_sets)
            if not combined_matches:
                raise ValueError(
                    "No exposed conversation entities matched the combined target selectors."
                )

            resolved_entities.update(combined_matches)

        if not resolved_entities:
            raise ValueError(
                "Target did not resolve to any exposed entities. Use discover_entities first."
            )

        return {"entity_id": sorted(resolved_entities)}

    def _find_exposed_entities_for_target(
        self, selectors: Dict[str, List[str]]
    ) -> Dict[str, set[str]]:
        """Resolve area, floor, label, and device selectors to exposed entities."""
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        area_registry = ar.async_get(self.hass)
        selector_sets = {key: set(values) for key, values in selectors.items() if values}

        area_floor_ids = {}
        area_label_ids = {}
        for area_entry in area_registry.async_list_areas():
            area_floor_ids[area_entry.id] = getattr(area_entry, "floor_id", None)
            area_label_ids[area_entry.id] = set(getattr(area_entry, "labels", set()) or set())

        matches = {
            "area_id": set(),
            "floor_id": set(),
            "label_id": set(),
            "device_id": set(),
        }

        for state_obj in self.hass.states.async_all():
            entity_id = state_obj.entity_id
            if not async_should_expose(self.hass, "conversation", entity_id):
                continue

            entity_entry = entity_registry.async_get(entity_id)
            device_entry = (
                device_registry.async_get(entity_entry.device_id)
                if entity_entry and entity_entry.device_id
                else None
            )
            area_id = None
            if entity_entry and entity_entry.area_id:
                area_id = entity_entry.area_id
            elif device_entry and device_entry.area_id:
                area_id = device_entry.area_id

            floor_id = area_floor_ids.get(area_id)

            label_ids = set(getattr(entity_entry, "labels", set()) or set())
            if device_entry:
                label_ids.update(getattr(device_entry, "labels", set()) or set())
            if area_id:
                label_ids.update(area_label_ids.get(area_id, set()))

            if selector_sets.get("area_id") and area_id in selector_sets["area_id"]:
                matches["area_id"].add(entity_id)
            if selector_sets.get("floor_id") and floor_id in selector_sets["floor_id"]:
                matches["floor_id"].add(entity_id)
            if selector_sets.get("label_id") and label_ids.intersection(selector_sets["label_id"]):
                matches["label_id"].add(entity_id)
            if (
                selector_sets.get("device_id")
                and entity_entry
                and entity_entry.device_id in selector_sets["device_id"]
            ):
                matches["device_id"].add(entity_id)

        return matches
