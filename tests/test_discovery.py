"""Tests for discovery helpers and entity summarization."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from custom_components.mcp_assist.discovery import SmartDiscovery


def test_entity_match_details_include_alias_and_floor_context(hass) -> None:
    """Search scoring should account for aliases and floor context."""
    discovery = SmartDiscovery(hass)
    state_obj = SimpleNamespace(
        entity_id="lock.basement_door",
        name="Basement Door Lock",
        attributes={"friendly_name": "Basement Door Lock"},
    )
    entity_entry = SimpleNamespace(aliases={"Deadbolt"})
    entity_context = {
        "device": None,
        "device_name": None,
        "device_name_by_user": None,
        "device_aliases": [],
        "area": "Basement",
        "area_aliases": [],
        "floor": "Lower Level",
        "floor_aliases": ["Downstairs"],
        "labels": [],
    }

    score, reasons = discovery._get_entity_match_details(
        "deadbolt",
        state_obj,
        entity_entry,
        entity_context,
    )
    assert score > 0
    assert "entity alias" in reasons

    score, reasons = discovery._get_entity_match_details(
        "downstairs",
        state_obj,
        entity_entry,
        entity_context,
    )
    assert score > 0
    assert "floor alias" in reasons


def test_create_entity_info_includes_weather_forecast_hints(hass) -> None:
    """Weather discovery summaries should expose forecast availability and preview."""
    discovery = SmartDiscovery(hass)
    state_obj = SimpleNamespace(
        entity_id="weather.home",
        name="Home Weather",
        domain="weather",
        state="sunny",
        attributes={
            "friendly_name": "Home Weather",
            "temperature": 54,
            "forecast": [
                {"datetime": date(2026, 4, 12), "condition": "rainy", "temperature": 58},
                {"datetime": date(2026, 4, 13), "condition": "sunny", "temperature": 62},
                {"datetime": date(2026, 4, 14), "condition": "cloudy", "temperature": 60},
            ],
        },
    )

    dummy_registry = SimpleNamespace(async_get=lambda *_args, **_kwargs: None)
    dummy_area_registry = SimpleNamespace(async_get_area=lambda *_args, **_kwargs: None)

    with (
        patch("custom_components.mcp_assist.discovery.er.async_get", return_value=dummy_registry),
        patch("custom_components.mcp_assist.discovery.dr.async_get", return_value=dummy_registry),
        patch(
            "custom_components.mcp_assist.discovery.ar.async_get",
            return_value=dummy_area_registry,
        ),
    ):
        entity_info = discovery._create_entity_info(
            state_obj,
            entity_context={},
        )

    assert entity_info["forecast_available"] is True
    assert entity_info["forecast_entries"] == 3
    assert entity_info["forecast_types"] == ["daily"]
    assert entity_info["forecast_service_supported"] is True
    assert entity_info["forecast_preview"] == [
        {"datetime": "2026-04-12", "condition": "rainy", "temperature": 58},
        {"datetime": "2026-04-13", "condition": "sunny", "temperature": 62},
    ]


def test_format_smart_results_page_includes_paging_metadata(hass) -> None:
    """Smart discovery pagination should expose counts and the next offset."""
    discovery = SmartDiscovery(hass)

    page = discovery._format_smart_results_page(
        {
            "query": "alex",
            "query_type": "person",
            "primary_entities": [
                {"entity_id": "person.alex", "name": "Alex", "state": "home"},
            ],
            "related_entities": {
                "presence": [
                    {
                        "entity_id": "binary_sensor.alex_home",
                        "name": "Alex Home",
                        "state": "on",
                    }
                ],
                "room_tracking": [
                    {
                        "entity_id": "input_text.alex_room",
                        "name": "Alex Room",
                        "state": "Office",
                    }
                ],
            },
        },
        limit=2,
        offset=0,
    )

    assert page["total_found"] == 3
    assert page["returned_count"] == 2
    assert page["remaining_count"] == 1
    assert page["next_offset"] == 2
    assert page["items"][0]["entity_id"] == "_summary"
    assert page["items"][0]["next_offset"] == 2
    assert len(page["items"]) == 3
