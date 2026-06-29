"""Tests for the Home Assistant domain registry helpers."""

from __future__ import annotations

from custom_components.mcp_assist.domain_registry import (
    get_response_services,
    map_action_to_service,
    validate_domain_action,
    validate_service_parameters,
)


def test_map_action_to_service_supports_aliases() -> None:
    """Action aliases should map to the correct domain service."""
    assert map_action_to_service("calendar", "create") == "create_event"
    assert map_action_to_service("todo", "cleanup") == "remove_completed_items"
    assert map_action_to_service("vacuum", "dock") == "return_to_base"


def test_validate_domain_action_rejects_read_only_domains() -> None:
    """Read-only domains should provide a helpful error."""
    valid, message = validate_domain_action("sensor", "turn_on")

    assert valid is False
    assert "read-only" in message


def test_validate_domain_action_suggests_similar_domains() -> None:
    """Invalid domains should guide the caller toward a valid choice."""
    valid, message = validate_domain_action("lights", "turn_on")

    assert valid is False
    assert "Did you mean" in message


def test_validate_service_parameters_for_calendar_modes() -> None:
    """Calendar event creation should enforce exactly one scheduling mode."""
    valid, message = validate_service_parameters(
        "calendar",
        "create_event",
        {"summary": "Dentist"},
    )
    assert valid is False
    assert "Missing calendar time fields" in message

    valid, message = validate_service_parameters(
        "calendar",
        "create_event",
        {
            "summary": "Dentist",
            "start_date": "2026-04-12",
            "end_date": "2026-04-12",
            "start_date_time": "2026-04-12T15:00:00",
            "end_date_time": "2026-04-12T16:00:00",
        },
    )
    assert valid is False
    assert "only one scheduling mode" in message


def test_validate_service_parameters_for_todo_updates() -> None:
    """Todo helpers should validate mutually exclusive and required fields."""
    valid, message = validate_service_parameters(
        "todo",
        "add_item",
        {"item": "Milk", "due_date": "2026-04-12", "due_datetime": "2026-04-12T08:00:00"},
    )
    assert valid is False
    assert "only one of due_date or due_datetime" in message

    valid, message = validate_service_parameters(
        "todo",
        "update_item",
        {"item": "Milk"},
    )
    assert valid is False
    assert "requires at least one update field" in message


def test_get_response_services_uses_registry_metadata() -> None:
    """Response-returning services should come from registry metadata."""
    assert "browse_media" in get_response_services("media_player")
    assert "get_events" in get_response_services("calendar")
