"""Tests for calculator custom tools."""

from __future__ import annotations

import pytest

from custom_components.mcp_assist.custom_tools.calculator import CalculatorTool


@pytest.fixture
def calculator() -> CalculatorTool:
    """Create a calculator tool instance."""
    return CalculatorTool(None)


@pytest.mark.asyncio
async def test_handle_call_returns_success_and_error_flags(calculator: CalculatorTool) -> None:
    """handle_call should report errors via the MCP response shape."""
    success = await calculator.handle_call("add", {"numbers": [1, 2, 3.5]})
    error = await calculator.handle_call("divide", {"a": 1, "b": 0})

    assert success["content"][0]["text"] == "6.5"
    assert success["isError"] is False
    assert error["content"][0]["text"] == "Error: division by zero"
    assert error["isError"] is True


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected"),
    [
        ("multiply", {"operands": ["2", 3, 4]}, "24"),
        ("round_number", {"a": "3.14159", "decimals": "2"}, "3.14"),
        ("convert_unit", {"value": 22, "from_unit": "°C", "to_unit": "°F"}, "71.6 °F"),
        ("convert_unit", {"value": 1250000, "from_unit": "B/s", "to_unit": "Mbps"}, "10 Mbps"),
        ("evaluate_expression", {"expression": "(68 - 32) * 5 / 9"}, "20"),
        ("evaluate_expression", {"expression": "2^10"}, "1024"),
    ],
)
def test_execute_supported_operations(
    calculator: CalculatorTool,
    tool_name: str,
    arguments: dict[str, object],
    expected: str,
) -> None:
    """Supported calculator operations should return normalized results."""
    assert calculator._execute(tool_name, arguments) == expected


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected"),
    [
        ("subtract", {"a": 5}, "Error: subtract requires b"),
        ("power", {"a": 2}, "Error: power requires b"),
        ("round_number", {"decimals": 2}, "Error: round_number requires a"),
        (
            "convert_unit",
            {"from_unit": "°C", "to_unit": "°F"},
            "Error: convert_unit requires value",
        ),
        (
            "convert_unit",
            {"value": 1, "from_unit": "°C", "to_unit": "meter"},
            "Error: cannot convert °C to m",
        ),
    ],
)
def test_execute_validates_required_arguments(
    calculator: CalculatorTool,
    tool_name: str,
    arguments: dict[str, object],
    expected: str,
) -> None:
    """Calculator tools should fail clearly when required arguments are missing."""
    assert calculator._execute(tool_name, arguments) == expected


def test_extract_numbers_from_named_keys_ignores_unrelated_fields(
    calculator: CalculatorTool,
) -> None:
    """Aggregate helpers should only consider intended numeric collections."""
    result = calculator._execute(
        "average",
        {"numbers": [1, 2, 3], "unexpected": 999, "metadata": {"ignore": 100}},
    )

    assert result == "2"


def test_evaluate_expression_rejects_unsupported_names(calculator: CalculatorTool) -> None:
    """Expression evaluation must stay sandboxed."""
    result = calculator._execute("evaluate_expression", {"expression": "__import__('os')"})

    assert result.startswith("Error:")
    assert "not supported" in result
