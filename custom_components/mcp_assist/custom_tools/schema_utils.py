"""Lightweight JSON-schema-style validation helpers for external tools."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class SchemaValidationError(ValueError):
    """Raised when data does not match a supported schema shape."""


def validate_and_normalize_json_value(
    schema: dict[str, Any] | None,
    value: Any,
    *,
    path: str = "value",
) -> Any:
    """Validate a limited JSON-schema-style payload and apply defaults."""
    if not isinstance(schema, dict) or not schema:
        return value

    if value is None and "default" in schema:
        value = deepcopy(schema["default"])

    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        errors: list[str] = []
        for option in one_of:
            try:
                return validate_and_normalize_json_value(option, value, path=path)
            except SchemaValidationError as err:
                errors.append(str(err))
        raise SchemaValidationError(
            f"{path} did not match any allowed schema option"
        )

    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        errors: list[str] = []
        for option in any_of:
            try:
                return validate_and_normalize_json_value(option, value, path=path)
            except SchemaValidationError as err:
                errors.append(str(err))
        raise SchemaValidationError(
            f"{path} did not match any allowed schema option"
        )

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        last_error: SchemaValidationError | None = None
        for option in schema_type:
            try:
                return validate_and_normalize_json_value(
                    {**schema, "type": option},
                    value,
                    path=path,
                )
            except SchemaValidationError as err:
                last_error = err
        raise last_error or SchemaValidationError(
            f"{path} did not match any allowed type"
        )

    if schema_type == "object" or (
        schema_type is None and isinstance(schema.get("properties"), dict)
    ):
        return _validate_object(schema, value, path=path)
    if schema_type == "array":
        return _validate_array(schema, value, path=path)
    if schema_type == "string":
        return _validate_string(schema, value, path=path)
    if schema_type == "integer":
        return _validate_integer(schema, value, path=path)
    if schema_type == "number":
        return _validate_number(schema, value, path=path)
    if schema_type == "boolean":
        return _validate_boolean(schema, value, path=path)

    normalized = value
    _validate_enum_and_const(schema, normalized, path=path)
    return normalized


def _validate_object(schema: dict[str, Any], value: Any, *, path: str) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{path} must be an object")

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required")
    if not isinstance(required, list):
        required = []
    additional_properties = schema.get("additionalProperties", True)

    normalized: dict[str, Any] = {}
    missing: list[str] = []
    for prop_name, prop_schema in properties.items():
        prop_path = f"{path}.{prop_name}"
        if prop_name in value:
            normalized[prop_name] = validate_and_normalize_json_value(
                prop_schema,
                value[prop_name],
                path=prop_path,
            )
        elif isinstance(prop_schema, dict) and "default" in prop_schema:
            normalized[prop_name] = validate_and_normalize_json_value(
                prop_schema,
                deepcopy(prop_schema.get("default")),
                path=prop_path,
            )
        elif prop_name in required:
            missing.append(prop_name)

    if missing:
        raise SchemaValidationError(
            f"{path} is missing required field(s): {', '.join(sorted(missing))}"
        )

    for extra_key, extra_value in value.items():
        if extra_key in properties:
            continue
        extra_path = f"{path}.{extra_key}"
        if additional_properties is False:
            raise SchemaValidationError(f"{extra_path} is not allowed")
        if isinstance(additional_properties, dict):
            normalized[extra_key] = validate_and_normalize_json_value(
                additional_properties,
                extra_value,
                path=extra_path,
            )
        else:
            normalized[extra_key] = extra_value

    _validate_enum_and_const(schema, normalized, path=path)
    return normalized


def _validate_array(schema: dict[str, Any], value: Any, *, path: str) -> list[Any]:
    if value is None and "default" in schema:
        value = deepcopy(schema["default"])
    if not isinstance(value, list):
        if isinstance(value, tuple):
            value = list(value)
        else:
            raise SchemaValidationError(f"{path} must be an array")

    items_schema = schema.get("items")
    normalized = [
        validate_and_normalize_json_value(
            items_schema if isinstance(items_schema, dict) else {},
            item,
            path=f"{path}[{index}]",
        )
        for index, item in enumerate(value)
    ]

    min_items = schema.get("minItems")
    if isinstance(min_items, int) and len(normalized) < min_items:
        raise SchemaValidationError(f"{path} must contain at least {min_items} item(s)")
    max_items = schema.get("maxItems")
    if isinstance(max_items, int) and len(normalized) > max_items:
        raise SchemaValidationError(f"{path} must contain at most {max_items} item(s)")

    _validate_enum_and_const(schema, normalized, path=path)
    return normalized


def _validate_string(schema: dict[str, Any], value: Any, *, path: str) -> str:
    if value is None:
        raise SchemaValidationError(f"{path} must be a string")
    if isinstance(value, (dict, list, tuple, set)):
        raise SchemaValidationError(f"{path} must be a string")

    normalized = str(value)
    min_length = schema.get("minLength")
    if isinstance(min_length, int) and len(normalized) < min_length:
        raise SchemaValidationError(f"{path} must be at least {min_length} character(s)")
    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and len(normalized) > max_length:
        raise SchemaValidationError(f"{path} must be at most {max_length} character(s)")

    _validate_enum_and_const(schema, normalized, path=path)
    return normalized


def _validate_integer(schema: dict[str, Any], value: Any, *, path: str) -> int:
    if isinstance(value, bool):
        raise SchemaValidationError(f"{path} must be an integer")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, float) and value.is_integer():
        normalized = int(value)
    elif isinstance(value, str):
        text_value = value.strip()
        try:
            normalized = int(text_value)
        except ValueError as err:
            raise SchemaValidationError(f"{path} must be an integer") from err
    else:
        raise SchemaValidationError(f"{path} must be an integer")

    minimum = schema.get("minimum")
    if isinstance(minimum, (int, float)) and normalized < minimum:
        raise SchemaValidationError(f"{path} must be >= {minimum}")
    maximum = schema.get("maximum")
    if isinstance(maximum, (int, float)) and normalized > maximum:
        raise SchemaValidationError(f"{path} must be <= {maximum}")

    _validate_enum_and_const(schema, normalized, path=path)
    return normalized


def _validate_number(schema: dict[str, Any], value: Any, *, path: str) -> float | int:
    if isinstance(value, bool):
        raise SchemaValidationError(f"{path} must be a number")
    if isinstance(value, (int, float)):
        normalized: float | int = value
    elif isinstance(value, str):
        text_value = value.strip()
        try:
            normalized = float(text_value)
        except ValueError as err:
            raise SchemaValidationError(f"{path} must be a number") from err
    else:
        raise SchemaValidationError(f"{path} must be a number")

    minimum = schema.get("minimum")
    if isinstance(minimum, (int, float)) and normalized < minimum:
        raise SchemaValidationError(f"{path} must be >= {minimum}")
    maximum = schema.get("maximum")
    if isinstance(maximum, (int, float)) and normalized > maximum:
        raise SchemaValidationError(f"{path} must be <= {maximum}")

    _validate_enum_and_const(schema, normalized, path=path)
    return normalized


def _validate_boolean(schema: dict[str, Any], value: Any, *, path: str) -> bool:
    if isinstance(value, bool):
        normalized = value
    elif isinstance(value, (int, float)) and value in {0, 1}:
        normalized = bool(value)
    elif isinstance(value, str):
        text_value = value.strip().lower()
        if text_value in {"true", "1", "yes", "on"}:
            normalized = True
        elif text_value in {"false", "0", "no", "off"}:
            normalized = False
        else:
            raise SchemaValidationError(f"{path} must be a boolean")
    else:
        raise SchemaValidationError(f"{path} must be a boolean")

    _validate_enum_and_const(schema, normalized, path=path)
    return normalized


def _validate_enum_and_const(schema: dict[str, Any], value: Any, *, path: str) -> None:
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        raise SchemaValidationError(
            f"{path} must be one of: {', '.join(str(item) for item in enum_values)}"
        )
    if "const" in schema and value != schema["const"]:
        raise SchemaValidationError(f"{path} must equal {schema['const']!r}")
