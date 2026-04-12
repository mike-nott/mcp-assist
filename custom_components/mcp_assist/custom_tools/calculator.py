"""Calculator custom tools for MCP Assist."""

import logging
import math
from typing import Any, Dict, List

_LOGGER = logging.getLogger(__name__)

NUM_SCHEMA = {"type": "number"}

CALCULATOR_TOOLS = [
    {
        "name": "add",
        "description": "Add two numbers. Example: add(a=10, b=3) returns 13.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"a": NUM_SCHEMA, "b": NUM_SCHEMA},
            "required": ["a", "b"],
            "additionalProperties": True,
        },
    },
    {
        "name": "subtract",
        "description": "Subtract b from a. Example: subtract(a=10, b=3) returns 7.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"a": NUM_SCHEMA, "b": NUM_SCHEMA},
            "required": ["a", "b"],
            "additionalProperties": True,
        },
    },
    {
        "name": "multiply",
        "description": "Multiply two numbers. Example: multiply(a=247, b=83) returns 20501.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"a": NUM_SCHEMA, "b": NUM_SCHEMA},
            "required": ["a", "b"],
            "additionalProperties": True,
        },
    },
    {
        "name": "divide",
        "description": "Divide a by b. Example: divide(a=10, b=4) returns 2.5.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"a": NUM_SCHEMA, "b": NUM_SCHEMA},
            "required": ["a", "b"],
            "additionalProperties": True,
        },
    },
    {
        "name": "sqrt",
        "description": "Square root of a number. Example: sqrt(a=144) returns 12.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"a": NUM_SCHEMA},
            "required": ["a"],
            "additionalProperties": True,
        },
    },
    {
        "name": "power",
        "description": "Raise a to the power of b. Example: power(a=2, b=10) returns 1024.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"a": NUM_SCHEMA, "b": NUM_SCHEMA},
            "required": ["a", "b"],
            "additionalProperties": True,
        },
    },
    {
        "name": "round_number",
        "description": "Round a number to a given number of decimal places. Example: round_number(a=3.14159, decimals=2) returns 3.14.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "a": NUM_SCHEMA,
                "decimals": {"type": "integer"},
            },
            "required": ["a"],
            "additionalProperties": True,
        },
    },
]


class CalculatorTool:
    """Calculator toolset with small arithmetic tools."""

    def __init__(self, hass) -> None:
        """Initialize calculator tools."""
        self.hass = hass
        self._tool_names = {tool["name"] for tool in CALCULATOR_TOOLS}

    async def initialize(self) -> None:
        """Initialize calculator tools."""
        pass

    def handles_tool(self, tool_name: str) -> bool:
        """Check if this class handles the given tool."""
        return tool_name in self._tool_names

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get MCP tool definitions for calculator tools."""
        return CALCULATOR_TOOLS

    async def handle_call(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a calculator tool."""
        result = self._execute(tool_name, arguments or {})
        is_error = isinstance(result, str) and result.startswith("Error:")

        return {
            "content": [{"type": "text", "text": result}],
            "isError": is_error,
        }

    def _extract_numbers(self, arguments: Dict[str, Any]) -> List[float]:
        """Extract numeric values from tolerant argument shapes."""
        numbers: List[float] = []
        for value in arguments.values():
            self._collect_numbers(value, numbers)
        return numbers

    def _collect_numbers(self, value: Any, numbers: List[float]) -> None:
        """Collect numbers from nested argument values."""
        parsed = self._coerce_number(value)
        if parsed is not None:
            numbers.append(parsed)
            return

        if isinstance(value, dict):
            for item in value.values():
                self._collect_numbers(item, numbers)
            return

        if isinstance(value, list):
            for item in value:
                self._collect_numbers(item, numbers)

    def _coerce_number(self, value: Any) -> float | int | None:
        """Coerce a value into a number when possible."""
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return float(value) if "." in value else int(value)
            except ValueError:
                return None
        return None

    def _normalize_result(self, result: float | int) -> str:
        """Normalize numeric results into compact text."""
        if result == 0:
            result = 0
        if isinstance(result, float) and math.isfinite(result) and result == int(result):
            result = int(result)
        return str(result)

    def _get_first_number(
        self,
        arguments: Dict[str, Any],
        keys: List[str],
        numbers: List[float],
        fallback_index: int = 0,
        default: float | int = 0,
    ) -> float | int:
        """Get the first matching numeric value from preferred keys or fallbacks."""
        for key in keys:
            parsed = self._coerce_number(arguments.get(key))
            if parsed is not None:
                return parsed

        if len(numbers) > fallback_index:
            return numbers[fallback_index]

        return default

    def _has_collection_operand(self, arguments: Dict[str, Any]) -> bool:
        """Check if the call provided list-like operands instead of named a/b args."""
        for key in ("numbers", "operands", "values", "args"):
            value = arguments.get(key)
            if isinstance(value, list):
                return True
        return False

    def _execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a calculator operation."""
        numbers = self._extract_numbers(arguments)
        a = self._get_first_number(
            arguments,
            ["a", "x", "value", "number", "base", "first", "first_number"],
            numbers,
        )
        b = self._get_first_number(
            arguments,
            ["b", "y", "exponent", "second", "second_number"],
            numbers,
            fallback_index=1,
        )

        try:
            if tool_name == "add":
                if self._has_collection_operand(arguments):
                    result = sum(numbers)
                else:
                    result = a + b
            elif tool_name == "subtract":
                result = a - b
            elif tool_name == "multiply":
                if self._has_collection_operand(arguments):
                    result = 1
                    for value in numbers:
                        result *= value
                else:
                    result = a * b
            elif tool_name == "divide":
                if b == 0:
                    return "Error: division by zero"
                result = a / b
            elif tool_name == "sqrt":
                a = self._get_first_number(
                    arguments,
                    ["a", "value", "number", "x"],
                    numbers,
                )
                result = math.sqrt(a)
            elif tool_name == "power":
                result = a**b
            elif tool_name == "round_number":
                a = self._get_first_number(
                    arguments,
                    ["a", "value", "number", "x"],
                    numbers,
                )
                decimals_value = self._get_first_number(
                    arguments,
                    ["decimals", "n", "digits", "places"],
                    numbers,
                    fallback_index=1,
                )
                decimals = int(decimals_value)
                result = round(a, decimals)
            else:
                return f"Error: unknown tool '{tool_name}'"
        except Exception as err:
            _LOGGER.debug("Calculator error for %s: %s", tool_name, err)
            return f"Error: {err}"

        return self._normalize_result(result)
