"""Tests for custom-tool schema validation helpers."""

from __future__ import annotations

import pytest

from custom_components.mcp_assist.custom_tools.schema_utils import (
    SchemaValidationError,
    validate_and_normalize_json_value,
)


def test_validate_and_normalize_json_value_returns_value_for_empty_schema() -> None:
    """Empty schemas should behave like pass-through validation."""
    payload = {"keep": "me"}

    assert validate_and_normalize_json_value({}, payload) == payload
    assert validate_and_normalize_json_value(None, payload) == payload


def test_validate_object_applies_defaults_and_validates_extra_properties() -> None:
    """Object schemas should apply defaults and validate typed extra fields."""
    normalized = validate_and_normalize_json_value(
        {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "default": 3},
                "name": {"type": "string", "default": "sample"},
            },
            "additionalProperties": {"type": "boolean"},
        },
        {"enabled": "yes"},
        path="settings",
    )

    assert normalized == {
        "count": 3,
        "name": "sample",
        "enabled": True,
    }


def test_validate_object_rejects_missing_required_fields_and_unknown_keys() -> None:
    """Object validation should fail closed for missing or disallowed fields."""
    with pytest.raises(
        SchemaValidationError,
        match=r"payload is missing required field\(s\): name",
    ):
        validate_and_normalize_json_value(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            {},
            path="payload",
        )

    with pytest.raises(
        SchemaValidationError,
        match=r"payload.extra is not allowed",
    ):
        validate_and_normalize_json_value(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
            },
            {"name": "ok", "extra": "nope"},
            path="payload",
        )


def test_validate_array_accepts_tuples_and_enforces_item_limits() -> None:
    """Arrays should normalize tuples and enforce min/max item counts."""
    normalized = validate_and_normalize_json_value(
        {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 3,
        },
        ("1", 2),
        path="items",
    )

    assert normalized == [1, 2]

    with pytest.raises(
        SchemaValidationError,
        match=r"items must contain at least 2 item\(s\)",
    ):
        validate_and_normalize_json_value(
            {"type": "array", "minItems": 2},
            [1],
            path="items",
        )

    with pytest.raises(
        SchemaValidationError,
        match=r"items must contain at most 1 item\(s\)",
    ):
        validate_and_normalize_json_value(
            {"type": "array", "maxItems": 1},
            [1, 2],
            path="items",
        )


def test_validate_scalar_types_normalize_strings_numbers_and_booleans() -> None:
    """Scalar validators should coerce supported string and numeric values."""
    assert (
        validate_and_normalize_json_value({"type": "string"}, 42, path="value") == "42"
    )
    assert (
        validate_and_normalize_json_value({"type": "integer"}, "7", path="value") == 7
    )
    assert (
        validate_and_normalize_json_value({"type": "number"}, "2.5", path="value")
        == 2.5
    )
    assert (
        validate_and_normalize_json_value({"type": "boolean"}, "on", path="value")
        is True
    )
    assert (
        validate_and_normalize_json_value({"type": "boolean"}, 0, path="value")
        is False
    )


def test_validate_scalar_types_reject_invalid_values() -> None:
    """Scalar validators should reject incompatible inputs with clear errors."""
    with pytest.raises(SchemaValidationError, match=r"value must be an integer"):
        validate_and_normalize_json_value({"type": "integer"}, True, path="value")

    with pytest.raises(SchemaValidationError, match=r"value must be a number"):
        validate_and_normalize_json_value({"type": "number"}, False, path="value")

    with pytest.raises(SchemaValidationError, match=r"value must be a boolean"):
        validate_and_normalize_json_value({"type": "boolean"}, "maybe", path="value")

    with pytest.raises(
        SchemaValidationError, match=r"value must be at least 3 character\(s\)"
    ):
        validate_and_normalize_json_value(
            {"type": "string", "minLength": 3},
            "hi",
            path="value",
        )


def test_validate_and_normalize_json_value_supports_union_schemas() -> None:
    """The validator should support oneOf, anyOf, and list-style type unions."""
    assert (
        validate_and_normalize_json_value(
            {"type": ["integer", "string"]},
            "hello",
            path="value",
        )
        == "hello"
    )
    assert (
        validate_and_normalize_json_value(
            {
                "oneOf": [
                    {"type": "integer", "minimum": 10},
                    {"type": "string", "const": "ten"},
                ]
            },
            "ten",
            path="value",
        )
        == "ten"
    )
    assert (
        validate_and_normalize_json_value(
            {
                "anyOf": [
                    {"type": "boolean"},
                    {"type": "integer", "minimum": 1},
                ]
            },
            "true",
            path="value",
        )
        is True
    )

    with pytest.raises(
        SchemaValidationError,
        match=r"value did not match any allowed schema option",
    ):
        validate_and_normalize_json_value(
            {"oneOf": [{"type": "integer"}, {"type": "boolean"}]},
            {"bad": "shape"},
            path="value",
        )


def test_validate_enum_and_const_constraints() -> None:
    """Enum and const restrictions should apply after normalization."""
    assert (
        validate_and_normalize_json_value(
            {"type": "string", "enum": ["north", "south"]},
            "north",
            path="direction",
        )
        == "north"
    )
    assert (
        validate_and_normalize_json_value(
            {"type": "integer", "const": 5},
            "5",
            path="count",
        )
        == 5
    )

    with pytest.raises(
        SchemaValidationError,
        match=r"direction must be one of: north, south",
    ):
        validate_and_normalize_json_value(
            {"type": "string", "enum": ["north", "south"]},
            "east",
            path="direction",
        )

    with pytest.raises(SchemaValidationError, match=r"count must equal 5"):
        validate_and_normalize_json_value(
            {"type": "integer", "const": 5},
            "4",
            path="count",
        )


def test_validate_and_normalize_json_value_deep_copies_defaults() -> None:
    """Mutable defaults should be copied so validation does not leak shared state."""
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "default": ["a"],
            }
        },
    }

    first = validate_and_normalize_json_value(schema, {}, path="payload")
    second = validate_and_normalize_json_value(schema, {}, path="payload")
    first["items"].append("b")

    assert second["items"] == ["a"]
