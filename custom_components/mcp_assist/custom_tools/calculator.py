"""Calculator and unit conversion custom tools for MCP Assist."""

from __future__ import annotations

import ast
import logging
import math
import re
from typing import Any, Dict, List

_LOGGER = logging.getLogger(__name__)

NUM_SCHEMA = {"type": "number"}
STRING_SCHEMA = {"type": "string"}


CALCULATOR_TOOLS = [
    {
        "name": "add",
        "description": "Add numbers. Accepts a and b, or a list like numbers=[1, 2, 3.5]. Example: add(a=10, b=3) returns 13.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "a": NUM_SCHEMA,
                "b": NUM_SCHEMA,
                "numbers": {"type": "array", "items": NUM_SCHEMA},
            },
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
        "description": "Multiply numbers. Accepts a and b, or a list like operands=[2, 3, 4]. Example: multiply(a=247, b=83) returns 20501.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "a": NUM_SCHEMA,
                "b": NUM_SCHEMA,
                "operands": {"type": "array", "items": NUM_SCHEMA},
            },
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
    {
        "name": "average",
        "description": "Calculate the arithmetic mean of numbers. Accepts a list such as numbers=[1, 2, 3]. Example: average(numbers=[1, 2, 3]) returns 2.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "numbers": {"type": "array", "items": NUM_SCHEMA},
                "values": {"type": "array", "items": NUM_SCHEMA},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "min_value",
        "description": "Return the smallest value from a list of numbers. Example: min_value(numbers=[5, 2, 9]) returns 2.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "numbers": {"type": "array", "items": NUM_SCHEMA},
                "values": {"type": "array", "items": NUM_SCHEMA},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "max_value",
        "description": "Return the largest value from a list of numbers. Example: max_value(numbers=[5, 2, 9]) returns 9.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "numbers": {"type": "array", "items": NUM_SCHEMA},
                "values": {"type": "array", "items": NUM_SCHEMA},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "convert_unit",
        "description": "Convert between common units including temperature (°C, °F, K), length, weight, volume, speed, pressure, energy, power, time, area, illuminance, electrical units, storage sizes, and data rates. Example: convert_unit(value=22, from_unit='°C', to_unit='°F') returns 71.6 °F.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "value": NUM_SCHEMA,
                "from_unit": STRING_SCHEMA,
                "to_unit": STRING_SCHEMA,
            },
            "required": ["value", "from_unit", "to_unit"],
            "additionalProperties": True,
        },
    },
    {
        "name": "evaluate_expression",
        "description": "Safely evaluate a math expression with parentheses and operators such as +, -, *, /, %, //, and **. Also supports abs(), round(), min(), max(), sqrt(), ceil(), floor(), and constants pi, e, and tau. Example: evaluate_expression(expression='(68 - 32) * 5 / 9') returns 20.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "expression": STRING_SCHEMA,
            },
            "required": ["expression"],
            "additionalProperties": True,
        },
    },
]


UNIT_DISPLAY = {
    "c": "°C",
    "f": "°F",
    "k": "K",
    "mm": "mm",
    "cm": "cm",
    "m": "m",
    "km": "km",
    "in": "in",
    "ft": "ft",
    "yd": "yd",
    "mi": "mi",
    "mg": "mg",
    "g": "g",
    "kg": "kg",
    "oz": "oz",
    "lb": "lb",
    "ml": "mL",
    "l": "L",
    "fl_oz": "fl oz",
    "cup": "cup",
    "pint": "pint",
    "quart": "quart",
    "gallon": "gal",
    "m_s": "m/s",
    "km_h": "km/h",
    "mph": "mph",
    "ft_s": "ft/s",
    "knot": "knot",
    "pa": "Pa",
    "kpa": "kPa",
    "hpa": "hPa",
    "bar": "bar",
    "psi": "psi",
    "inhg": "inHg",
    "j": "J",
    "kj": "kJ",
    "mj": "MJ",
    "wh": "Wh",
    "kwh": "kWh",
    "w": "W",
    "kw": "kW",
    "mw": "MW",
    "ms": "ms",
    "s": "s",
    "min": "min",
    "h": "h",
    "day": "day",
    "week": "week",
    "sq_cm": "cm²",
    "sq_m": "m²",
    "sq_km": "km²",
    "sq_in": "in²",
    "sq_ft": "ft²",
    "sq_yd": "yd²",
    "acre": "acre",
    "hectare": "hectare",
    "lx": "lx",
    "fc": "fc",
    "ma": "mA",
    "a": "A",
    "mv": "mV",
    "v": "V",
    "kv": "kV",
    "hz": "Hz",
    "khz": "kHz",
    "mhz": "MHz",
    "ghz": "GHz",
    "b": "B",
    "kb": "KB",
    "mb": "MB",
    "gb": "GB",
    "tb": "TB",
    "kib": "KiB",
    "mib": "MiB",
    "gib": "GiB",
    "tib": "TiB",
    "bps": "bps",
    "kbps": "kbps",
    "mbps": "Mbps",
    "gbps": "Gbps",
    "tbps": "Tbps",
    "Bps": "B/s",
    "KBps": "KB/s",
    "MBps": "MB/s",
    "GBps": "GB/s",
    "TBps": "TB/s",
    "KiBps": "KiB/s",
    "MiBps": "MiB/s",
    "GiBps": "GiB/s",
    "TiBps": "TiB/s",
}

UNIT_GROUPS = {
    "length": {
        "mm": 0.001,
        "cm": 0.01,
        "m": 1.0,
        "km": 1000.0,
        "in": 0.0254,
        "ft": 0.3048,
        "yd": 0.9144,
        "mi": 1609.344,
    },
    "mass": {
        "mg": 0.001,
        "g": 1.0,
        "kg": 1000.0,
        "oz": 28.349523125,
        "lb": 453.59237,
    },
    "volume": {
        "ml": 0.001,
        "l": 1.0,
        "fl_oz": 0.0295735295625,
        "cup": 0.2365882365,
        "pint": 0.473176473,
        "quart": 0.946352946,
        "gallon": 3.785411784,
    },
    "speed": {
        "m_s": 1.0,
        "km_h": 0.2777777777777778,
        "mph": 0.44704,
        "ft_s": 0.3048,
        "knot": 0.5144444444444445,
    },
    "pressure": {
        "pa": 1.0,
        "kpa": 1000.0,
        "hpa": 100.0,
        "bar": 100000.0,
        "psi": 6894.757293168,
        "inhg": 3386.389,
    },
    "energy": {
        "j": 1.0,
        "kj": 1000.0,
        "mj": 1000000.0,
        "wh": 3600.0,
        "kwh": 3600000.0,
    },
    "power": {
        "w": 1.0,
        "kw": 1000.0,
        "mw": 1000000.0,
    },
    "time": {
        "ms": 0.001,
        "s": 1.0,
        "min": 60.0,
        "h": 3600.0,
        "day": 86400.0,
        "week": 604800.0,
    },
    "area": {
        "sq_cm": 0.0001,
        "sq_m": 1.0,
        "sq_km": 1000000.0,
        "sq_in": 0.00064516,
        "sq_ft": 0.09290304,
        "sq_yd": 0.83612736,
        "acre": 4046.8564224,
        "hectare": 10000.0,
    },
    "illuminance": {
        "lx": 1.0,
        "fc": 10.763910416709722,
    },
    "current": {
        "ma": 0.001,
        "a": 1.0,
    },
    "voltage": {
        "mv": 0.001,
        "v": 1.0,
        "kv": 1000.0,
    },
    "frequency": {
        "hz": 1.0,
        "khz": 1000.0,
        "mhz": 1000000.0,
        "ghz": 1000000000.0,
    },
    "data_size": {
        "b": 1.0,
        "kb": 1000.0,
        "mb": 1000000.0,
        "gb": 1000000000.0,
        "tb": 1000000000000.0,
        "kib": 1024.0,
        "mib": 1048576.0,
        "gib": 1073741824.0,
        "tib": 1099511627776.0,
    },
    "data_rate": {
        "bps": 1.0,
        "kbps": 1000.0,
        "mbps": 1000000.0,
        "gbps": 1000000000.0,
        "tbps": 1000000000000.0,
        "Bps": 8.0,
        "KBps": 8000.0,
        "MBps": 8000000.0,
        "GBps": 8000000000.0,
        "TBps": 8000000000000.0,
        "KiBps": 8192.0,
        "MiBps": 8388608.0,
        "GiBps": 8589934592.0,
        "TiBps": 8796093022208.0,
    },
}

TEMPERATURE_UNITS = {"c", "f", "k"}
UNIT_CATEGORY_BY_ID = {
    unit_id: category
    for category, units in UNIT_GROUPS.items()
    for unit_id in units.keys()
}
for unit_id in TEMPERATURE_UNITS:
    UNIT_CATEGORY_BY_ID[unit_id] = "temperature"

UNIT_ALIASES: Dict[str, str] = {}
CASE_SENSITIVE_UNIT_ALIASES: Dict[str, str] = {}


def _normalize_case_sensitive_unit_alias(value: Any) -> str | None:
    """Normalize a unit alias while preserving letter case for case-sensitive units."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    text = (
        text.replace("μ", "u")
        .replace("µ", "u")
        .replace("º", "°")
        .replace("²", "2")
        .replace("³", "3")
    )
    text = text.replace("^2", "2").replace("^3", "3")
    text = re.sub(r"\bsquared\b", "2", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcubed\b", "3", text, flags=re.IGNORECASE)
    text = text.replace("/", " per ")
    text = text.replace("°", "")
    text = re.sub(r"\bdegrees?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdeg\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9]+", "", text)
    return text or None


def _normalize_unit_alias(value: Any) -> str | None:
    """Normalize a unit alias for lookup."""
    if value is None:
        return None

    text = str(value).strip().casefold()
    if not text:
        return None

    text = (
        text.replace("μ", "u")
        .replace("µ", "u")
        .replace("º", "°")
        .replace("²", "2")
        .replace("³", "3")
    )
    text = text.replace("^2", "2").replace("^3", "3")
    text = re.sub(r"\bsquared\b", "2", text)
    text = re.sub(r"\bcubed\b", "3", text)
    text = text.replace("º", "°")
    text = text.replace("/", " per ")
    text = text.replace("°", "")
    text = re.sub(r"\bdegrees?\b", "", text)
    text = re.sub(r"\bdeg\b", "", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text or None


def _register_unit_aliases(canonical: str, aliases: List[str]) -> None:
    """Register unit aliases."""
    for alias in aliases:
        normalized = _normalize_unit_alias(alias)
        if normalized:
            UNIT_ALIASES[normalized] = canonical


def _register_case_sensitive_unit_aliases(canonical: str, aliases: List[str]) -> None:
    """Register aliases that need case-sensitive matching."""
    for alias in aliases:
        normalized = _normalize_case_sensitive_unit_alias(alias)
        if normalized:
            CASE_SENSITIVE_UNIT_ALIASES[normalized] = canonical


_register_unit_aliases("c", ["c", "°c", "ºc", "celsius", "centigrade"])
_register_unit_aliases("f", ["f", "°f", "ºf", "fahrenheit"])
_register_unit_aliases("k", ["k", "kelvin"])
_register_unit_aliases("mm", ["mm", "millimeter", "millimeters", "millimetre", "millimetres"])
_register_unit_aliases("cm", ["cm", "centimeter", "centimeters", "centimetre", "centimetres"])
_register_unit_aliases("m", ["m", "meter", "meters", "metre", "metres"])
_register_unit_aliases("km", ["km", "kilometer", "kilometers", "kilometre", "kilometres"])
_register_unit_aliases("in", ["in", "inch", "inches"])
_register_unit_aliases("ft", ["ft", "foot", "feet"])
_register_unit_aliases("yd", ["yd", "yard", "yards"])
_register_unit_aliases("mi", ["mi", "mile", "miles"])
_register_unit_aliases("mg", ["mg", "milligram", "milligrams"])
_register_unit_aliases("g", ["g", "gram", "grams"])
_register_unit_aliases("kg", ["kg", "kilogram", "kilograms"])
_register_unit_aliases("oz", ["oz", "ounce", "ounces"])
_register_unit_aliases("lb", ["lb", "lbs", "pound", "pounds"])
_register_unit_aliases("ml", ["ml", "milliliter", "milliliters", "millilitre", "millilitres"])
_register_unit_aliases("l", ["l", "liter", "liters", "litre", "litres"])
_register_unit_aliases("fl_oz", ["fl oz", "floz", "fluid ounce", "fluid ounces"])
_register_unit_aliases("cup", ["cup", "cups"])
_register_unit_aliases("pint", ["pint", "pints", "pt"])
_register_unit_aliases("quart", ["quart", "quarts", "qt"])
_register_unit_aliases("gallon", ["gallon", "gallons", "gal"])
_register_unit_aliases("m_s", ["m/s", "meter per second", "meters per second", "metre per second", "metres per second"])
_register_unit_aliases("km_h", ["km/h", "kph", "kmh", "kilometer per hour", "kilometers per hour", "kilometre per hour", "kilometres per hour"])
_register_unit_aliases("mph", ["mph", "mile per hour", "miles per hour"])
_register_unit_aliases("ft_s", ["ft/s", "fps", "foot per second", "feet per second"])
_register_unit_aliases("knot", ["knot", "knots", "kt", "kts"])
_register_unit_aliases("pa", ["pa", "pascal", "pascals"])
_register_unit_aliases("kpa", ["kpa", "kilopascal", "kilopascals"])
_register_unit_aliases("hpa", ["hpa", "hectopascal", "hectopascals", "mbar", "millibar", "millibars"])
_register_unit_aliases("bar", ["bar", "bars"])
_register_unit_aliases("psi", ["psi"])
_register_unit_aliases("inhg", ["inhg", "in hg", "inch mercury", "inches mercury", "inch of mercury", "inches of mercury"])
_register_unit_aliases("j", ["j", "joule", "joules"])
_register_unit_aliases("kj", ["kj", "kilojoule", "kilojoules"])
_register_unit_aliases("mj", ["mj", "megajoule", "megajoules"])
_register_unit_aliases("wh", ["wh", "watt hour", "watt hours"])
_register_unit_aliases("kwh", ["kwh", "kilowatt hour", "kilowatt hours"])
_register_unit_aliases("w", ["w", "watt", "watts"])
_register_unit_aliases("kw", ["kw", "kilowatt", "kilowatts"])
_register_unit_aliases("mw", ["mw", "megawatt", "megawatts"])
_register_unit_aliases("ms", ["ms", "millisecond", "milliseconds"])
_register_unit_aliases("s", ["s", "sec", "secs", "second", "seconds"])
_register_unit_aliases("min", ["min", "mins", "minute", "minutes"])
_register_unit_aliases("h", ["h", "hr", "hrs", "hour", "hours"])
_register_unit_aliases("day", ["day", "days", "d"])
_register_unit_aliases("week", ["week", "weeks", "wk", "wks"])
_register_unit_aliases("sq_cm", ["cm2", "cm²", "square centimeter", "square centimeters", "square centimetre", "square centimetres"])
_register_unit_aliases("sq_m", ["m2", "m²", "sqm", "square meter", "square meters", "square metre", "square metres"])
_register_unit_aliases("sq_km", ["km2", "km²", "square kilometer", "square kilometers", "square kilometre", "square kilometres"])
_register_unit_aliases("sq_in", ["in2", "in²", "square inch", "square inches", "sqin", "sq in"])
_register_unit_aliases("sq_ft", ["ft2", "ft²", "sqft", "sq ft", "square foot", "square feet"])
_register_unit_aliases("sq_yd", ["yd2", "yd²", "sqyd", "sq yd", "square yard", "square yards"])
_register_unit_aliases("acre", ["acre", "acres"])
_register_unit_aliases("hectare", ["hectare", "hectares", "ha"])
_register_unit_aliases("lx", ["lx", "lux"])
_register_unit_aliases("fc", ["fc", "foot candle", "foot candles", "footcandle", "footcandles"])
_register_unit_aliases("ma", ["ma", "milliamp", "milliamps", "milliampere", "milliamperes"])
_register_unit_aliases("a", ["a", "amp", "amps", "ampere", "amperes"])
_register_unit_aliases("mv", ["mv", "millivolt", "millivolts"])
_register_unit_aliases("v", ["v", "volt", "volts"])
_register_unit_aliases("kv", ["kv", "kilovolt", "kilovolts"])
_register_unit_aliases("hz", ["hz", "hertz"])
_register_unit_aliases("khz", ["khz", "kilohertz"])
_register_unit_aliases("mhz", ["mhz", "megahertz"])
_register_unit_aliases("ghz", ["ghz", "gigahertz"])
_register_unit_aliases("b", ["b", "byte", "bytes"])
_register_unit_aliases("kb", ["kb", "kilobyte", "kilobytes"])
_register_unit_aliases("mb", ["mb", "megabyte", "megabytes"])
_register_unit_aliases("gb", ["gb", "gigabyte", "gigabytes"])
_register_unit_aliases("tb", ["tb", "terabyte", "terabytes"])
_register_unit_aliases("kib", ["kib", "kibibyte", "kibibytes", "kib"])
_register_unit_aliases("mib", ["mib", "mebibyte", "mebibytes"])
_register_unit_aliases("gib", ["gib", "gibibyte", "gibibytes"])
_register_unit_aliases("tib", ["tib", "tebibyte", "tebibytes"])
_register_unit_aliases("bps", ["bps", "bit per second", "bits per second"])
_register_unit_aliases("kbps", ["kbps", "kilobit per second", "kilobits per second"])
_register_unit_aliases("mbps", ["mbps", "megabit per second", "megabits per second"])
_register_unit_aliases("gbps", ["gbps", "gigabit per second", "gigabits per second"])
_register_unit_aliases("tbps", ["tbps", "terabit per second", "terabits per second"])
_register_unit_aliases("Bps", ["byte per second", "bytes per second"])
_register_unit_aliases("KBps", ["kilobyte per second", "kilobytes per second"])
_register_unit_aliases("MBps", ["megabyte per second", "megabytes per second"])
_register_unit_aliases("GBps", ["gigabyte per second", "gigabytes per second"])
_register_unit_aliases("TBps", ["terabyte per second", "terabytes per second"])
_register_unit_aliases("KiBps", ["kibibyte per second", "kibibytes per second"])
_register_unit_aliases("MiBps", ["mebibyte per second", "mebibytes per second"])
_register_unit_aliases("GiBps", ["gibibyte per second", "gibibytes per second"])
_register_unit_aliases("TiBps", ["tebibyte per second", "tebibytes per second"])

_register_case_sensitive_unit_aliases("bps", ["b/s"])
_register_case_sensitive_unit_aliases("kbps", ["kb/s", "Kb/s"])
_register_case_sensitive_unit_aliases("mbps", ["mb/s", "Mb/s"])
_register_case_sensitive_unit_aliases("gbps", ["gb/s", "Gb/s"])
_register_case_sensitive_unit_aliases("tbps", ["tb/s", "Tb/s"])
_register_case_sensitive_unit_aliases("Bps", ["B/s", "Bps"])
_register_case_sensitive_unit_aliases("KBps", ["KB/s", "KBps", "kB/s", "kBps"])
_register_case_sensitive_unit_aliases("MBps", ["MB/s", "MBps"])
_register_case_sensitive_unit_aliases("GBps", ["GB/s", "GBps"])
_register_case_sensitive_unit_aliases("TBps", ["TB/s", "TBps"])
_register_case_sensitive_unit_aliases("KiBps", ["KiB/s", "KiBps"])
_register_case_sensitive_unit_aliases("MiBps", ["MiB/s", "MiBps"])
_register_case_sensitive_unit_aliases("GiBps", ["GiB/s", "GiBps"])
_register_case_sensitive_unit_aliases("TiBps", ["TiB/s", "TiBps"])


ALLOWED_AST_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
}

ALLOWED_AST_UNARY_OPERATORS = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}

ALLOWED_AST_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
}

ALLOWED_AST_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


class CalculatorTool:
    """Calculator toolset with arithmetic, summary, and unit conversion tools."""

    def __init__(self, hass) -> None:
        """Initialize calculator tools."""
        self.hass = hass
        self._tool_names = {tool["name"] for tool in CALCULATOR_TOOLS}

    async def initialize(self) -> None:
        """Initialize calculator tools."""
        return None

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

    def _extract_numbers(self, value: Any) -> List[float]:
        """Extract numeric values from a tolerant value shape."""
        numbers: List[float] = []
        self._collect_numbers(value, numbers)
        return numbers

    def _extract_numbers_from_keys(
        self, arguments: Dict[str, Any], keys: List[str]
    ) -> List[float]:
        """Extract numeric values only from specific argument keys."""
        numbers: List[float] = []
        for key in keys:
            if key in arguments:
                self._collect_numbers(arguments.get(key), numbers)
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

        if isinstance(value, (list, tuple, set)):
            for item in value:
                self._collect_numbers(item, numbers)

    def _coerce_number(self, value: Any) -> float | int | None:
        """Coerce a value into a number when possible."""
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            if not math.isfinite(value):
                return None
            return value
        if isinstance(value, str):
            text = value.strip().replace(",", "").replace("_", "")
            if not text:
                return None
            try:
                parsed = float(text)
            except ValueError:
                return None
            if not math.isfinite(parsed):
                return None
            if parsed.is_integer():
                return int(parsed)
            return parsed
        return None

    def _normalize_result(self, result: float | int) -> str:
        """Normalize numeric results into compact text."""
        if isinstance(result, float):
            if not math.isfinite(result):
                return str(result)
            if result == 0:
                result = 0.0
            if result.is_integer():
                return str(int(result))
            return format(result, ".12g")
        return str(result)

    def _get_number(
        self,
        arguments: Dict[str, Any],
        keys: List[str],
    ) -> float | int | None:
        """Get the first matching numeric value from preferred keys."""
        for key in keys:
            if key not in arguments:
                continue
            parsed = self._coerce_number(arguments.get(key))
            if parsed is not None:
                return parsed
        return None

    def _coerce_integer(self, value: Any) -> int | None:
        """Coerce a value into an integer when possible."""
        parsed = self._coerce_number(value)
        if parsed is None:
            return None
        if isinstance(parsed, float) and not parsed.is_integer():
            return None
        return int(parsed)

    def _get_first_text(
        self, arguments: Dict[str, Any], keys: List[str]
    ) -> str | None:
        """Get the first non-empty text value from preferred keys."""
        for key in keys:
            value = arguments.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    def _has_collection_operand(self, arguments: Dict[str, Any]) -> bool:
        """Check if the call provided list-like operands instead of named a/b args."""
        for key in ("numbers", "operands", "values", "args"):
            value = arguments.get(key)
            if isinstance(value, (list, tuple, set)):
                return True
        return False

    def _require_numbers(self, numbers: List[float], tool_name: str) -> str | None:
        """Validate that a list-based tool received at least one number."""
        if numbers:
            return None
        return f"Error: {tool_name} requires one or more numbers"

    def _require_number(
        self, arguments: Dict[str, Any], keys: List[str], label: str, tool_name: str
    ) -> float | int | str:
        """Require a named numeric argument."""
        value = self._get_number(arguments, keys)
        if value is None:
            return f"Error: {tool_name} requires {label}"
        return value

    def _normalize_unit(self, value: Any) -> str | None:
        """Normalize a unit name into a canonical internal unit id."""
        case_sensitive_normalized = _normalize_case_sensitive_unit_alias(value)
        if case_sensitive_normalized:
            case_sensitive_match = CASE_SENSITIVE_UNIT_ALIASES.get(case_sensitive_normalized)
            if case_sensitive_match:
                return case_sensitive_match

        normalized = _normalize_unit_alias(value)
        if not normalized:
            return None
        return UNIT_ALIASES.get(normalized)

    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert temperatures through Celsius."""
        if from_unit == to_unit:
            return value

        if from_unit == "c":
            celsius = value
        elif from_unit == "f":
            celsius = (value - 32.0) * 5.0 / 9.0
        elif from_unit == "k":
            celsius = value - 273.15
        else:
            raise ValueError(f"Unsupported temperature unit '{from_unit}'")

        if to_unit == "c":
            return celsius
        if to_unit == "f":
            return celsius * 9.0 / 5.0 + 32.0
        if to_unit == "k":
            return celsius + 273.15

        raise ValueError(f"Unsupported temperature unit '{to_unit}'")

    def _convert_linear_unit(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert units using a linear factor table."""
        category = UNIT_CATEGORY_BY_ID.get(from_unit)
        if category is None or category != UNIT_CATEGORY_BY_ID.get(to_unit):
            raise ValueError(f"Cannot convert {from_unit} to {to_unit}")

        factors = UNIT_GROUPS.get(category)
        if factors is None:
            raise ValueError(f"No conversion table for category '{category}'")

        base_value = value * factors[from_unit]
        return base_value / factors[to_unit]

    def _convert_unit(self, value: float, from_unit_raw: Any, to_unit_raw: Any) -> str:
        """Convert between supported units."""
        from_unit = self._normalize_unit(from_unit_raw)
        to_unit = self._normalize_unit(to_unit_raw)

        if not from_unit:
            return f"Error: unsupported from_unit '{from_unit_raw}'"
        if not to_unit:
            return f"Error: unsupported to_unit '{to_unit_raw}'"

        from_category = UNIT_CATEGORY_BY_ID.get(from_unit)
        to_category = UNIT_CATEGORY_BY_ID.get(to_unit)
        if from_category != to_category:
            return (
                f"Error: cannot convert {UNIT_DISPLAY.get(from_unit, from_unit)} to "
                f"{UNIT_DISPLAY.get(to_unit, to_unit)}"
            )

        try:
            if from_category == "temperature":
                converted = self._convert_temperature(value, from_unit, to_unit)
            else:
                converted = self._convert_linear_unit(value, from_unit, to_unit)
        except Exception as err:
            return f"Error: {err}"

        if not math.isfinite(converted):
            return "Error: conversion result is not finite"

        return f"{self._normalize_result(converted)} {UNIT_DISPLAY.get(to_unit, to_unit)}"

    def _safe_power(self, left: float | int, right: float | int) -> float | int:
        """Limit pathological exponent operations in expression evaluation."""
        if abs(right) > 1000:
            raise ValueError("exponent is too large")
        if abs(left) > 1000000 and abs(right) > 20:
            raise ValueError("power expression is too large")
        if right > 1 and abs(left) > 1:
            estimated_bits = abs(right) * math.log2(abs(left))
            if estimated_bits > 4096:
                raise ValueError("expression result is too large")

        result = left**right
        if isinstance(result, complex):
            raise ValueError("complex results are not supported")
        if isinstance(result, int) and result.bit_length() > 4096:
            raise ValueError("expression result is too large")
        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("expression result is not finite")
        return result

    def _evaluate_expression_node(self, node: ast.AST) -> float | int:
        """Safely evaluate an AST node for arithmetic expressions."""
        if isinstance(node, ast.Expression):
            return self._evaluate_expression_node(node.body)

        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("expression may only contain numeric constants")
            return value

        if type(node).__name__ == "Num":  # pragma: no cover - compat for older Python ASTs
            return node.n

        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                left = self._evaluate_expression_node(node.left)
                right = self._evaluate_expression_node(node.right)
                return self._safe_power(left, right)

            operator = ALLOWED_AST_BINARY_OPERATORS.get(type(node.op))
            if operator is None:
                raise ValueError("expression contains an unsupported operator")
            left = self._evaluate_expression_node(node.left)
            right = self._evaluate_expression_node(node.right)
            return operator(left, right)

        if isinstance(node, ast.UnaryOp):
            operator = ALLOWED_AST_UNARY_OPERATORS.get(type(node.op))
            if operator is None:
                raise ValueError("expression contains an unsupported unary operator")
            return operator(self._evaluate_expression_node(node.operand))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("expression contains an unsupported function call")
            function = ALLOWED_AST_FUNCTIONS.get(node.func.id)
            if function is None:
                raise ValueError(f"function '{node.func.id}' is not supported")
            if node.keywords:
                raise ValueError("keyword arguments are not supported in expressions")
            return function(*[self._evaluate_expression_node(node_arg) for node_arg in node.args])

        if isinstance(node, ast.Name):
            if node.id in ALLOWED_AST_NAMES:
                return ALLOWED_AST_NAMES[node.id]
            raise ValueError(f"name '{node.id}' is not supported")

        raise ValueError("expression contains unsupported syntax")

    def _evaluate_expression(self, expression_raw: Any) -> str:
        """Safely evaluate a user-provided arithmetic expression."""
        if expression_raw is None:
            return "Error: evaluate_expression requires an expression"

        expression = str(expression_raw).strip()
        if not expression:
            return "Error: evaluate_expression requires a non-empty expression"
        if len(expression) > 256:
            return "Error: expression is too long"

        expression = (
            expression.replace("×", "*")
            .replace("÷", "/")
            .replace("−", "-")
            .replace("^", "**")
        )
        expression = re.sub(r"(?<=\d),(?=\d)", "", expression)

        try:
            parsed = ast.parse(expression, mode="eval")
            if sum(1 for _ in ast.walk(parsed)) > 128:
                return "Error: expression is too complex"
            result = self._evaluate_expression_node(parsed)
        except ZeroDivisionError:
            return "Error: division by zero"
        except Exception as err:
            return f"Error: {err}"

        return self._normalize_result(result)

    def _execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a calculator operation."""
        pair_number_keys = [
            "a",
            "b",
            "x",
            "y",
            "value",
            "number",
            "base",
            "exponent",
            "first",
            "second",
            "first_number",
            "second_number",
        ]
        collection_number_keys = ["numbers", "values", "operands", "args"]

        try:
            if tool_name == "add":
                if self._has_collection_operand(arguments):
                    numbers = self._extract_numbers_from_keys(arguments, collection_number_keys)
                    error = self._require_numbers(numbers, tool_name)
                    if error:
                        return error
                    result = math.fsum(numbers)
                else:
                    a = self._require_number(
                        arguments,
                        ["a", "x", "value", "number", "first", "first_number"],
                        "a",
                        tool_name,
                    )
                    if isinstance(a, str):
                        return a
                    b = self._require_number(
                        arguments,
                        ["b", "y", "second", "second_number"],
                        "b",
                        tool_name,
                    )
                    if isinstance(b, str):
                        return b
                    result = a + b
            elif tool_name == "subtract":
                a = self._require_number(
                    arguments,
                    ["a", "x", "value", "number", "first", "first_number"],
                    "a",
                    tool_name,
                )
                if isinstance(a, str):
                    return a
                b = self._require_number(
                    arguments,
                    ["b", "y", "second", "second_number"],
                    "b",
                    tool_name,
                )
                if isinstance(b, str):
                    return b
                result = a - b
            elif tool_name == "multiply":
                if self._has_collection_operand(arguments):
                    numbers = self._extract_numbers_from_keys(arguments, collection_number_keys)
                    error = self._require_numbers(numbers, tool_name)
                    if error:
                        return error
                    result = 1.0
                    for value in numbers:
                        result *= value
                else:
                    a = self._require_number(
                        arguments,
                        ["a", "x", "value", "number", "first", "first_number"],
                        "a",
                        tool_name,
                    )
                    if isinstance(a, str):
                        return a
                    b = self._require_number(
                        arguments,
                        ["b", "y", "second", "second_number"],
                        "b",
                        tool_name,
                    )
                    if isinstance(b, str):
                        return b
                    result = a * b
            elif tool_name == "divide":
                a = self._require_number(
                    arguments,
                    ["a", "x", "value", "number", "first", "first_number"],
                    "a",
                    tool_name,
                )
                if isinstance(a, str):
                    return a
                b = self._require_number(
                    arguments,
                    ["b", "y", "second", "second_number"],
                    "b",
                    tool_name,
                )
                if isinstance(b, str):
                    return b
                if b == 0:
                    return "Error: division by zero"
                result = a / b
            elif tool_name == "sqrt":
                a = self._require_number(
                    arguments,
                    ["a", "value", "number", "x"],
                    "a",
                    tool_name,
                )
                if isinstance(a, str):
                    return a
                if a < 0:
                    return "Error: square root requires a non-negative number"
                result = math.sqrt(a)
            elif tool_name == "power":
                a = self._require_number(
                    arguments,
                    ["a", "x", "value", "number", "base", "first", "first_number"],
                    "a",
                    tool_name,
                )
                if isinstance(a, str):
                    return a
                b = self._require_number(
                    arguments,
                    ["b", "y", "exponent", "second", "second_number"],
                    "b",
                    tool_name,
                )
                if isinstance(b, str):
                    return b
                try:
                    result = self._safe_power(a, b)
                except ValueError as err:
                    return f"Error: {err}"
            elif tool_name == "round_number":
                a = self._require_number(
                    arguments,
                    ["a", "value", "number", "x"],
                    "a",
                    tool_name,
                )
                if isinstance(a, str):
                    return a
                decimals = 0
                for key in ["decimals", "n", "digits", "places"]:
                    if key not in arguments:
                        continue
                    parsed = self._coerce_integer(arguments.get(key))
                    if parsed is None:
                        return "Error: round_number requires an integer decimals value"
                    decimals = parsed
                    break
                result = round(a, decimals)
            elif tool_name == "average":
                numbers = self._extract_numbers_from_keys(arguments, collection_number_keys)
                if not numbers:
                    numbers = self._extract_numbers_from_keys(arguments, pair_number_keys)
                error = self._require_numbers(numbers, tool_name)
                if error:
                    return error
                result = math.fsum(numbers) / len(numbers)
            elif tool_name == "min_value":
                numbers = self._extract_numbers_from_keys(arguments, collection_number_keys)
                if not numbers:
                    numbers = self._extract_numbers_from_keys(arguments, pair_number_keys)
                error = self._require_numbers(numbers, tool_name)
                if error:
                    return error
                result = min(numbers)
            elif tool_name == "max_value":
                numbers = self._extract_numbers_from_keys(arguments, collection_number_keys)
                if not numbers:
                    numbers = self._extract_numbers_from_keys(arguments, pair_number_keys)
                error = self._require_numbers(numbers, tool_name)
                if error:
                    return error
                result = max(numbers)
            elif tool_name == "convert_unit":
                value = self._require_number(
                    arguments,
                    ["value", "a", "number", "x"],
                    "value",
                    tool_name,
                )
                if isinstance(value, str):
                    return value
                from_unit = self._get_first_text(
                    arguments,
                    ["from_unit", "from", "source_unit", "unit_from"],
                )
                to_unit = self._get_first_text(
                    arguments,
                    ["to_unit", "to", "target_unit", "unit_to"],
                )
                if not from_unit or not to_unit:
                    return "Error: convert_unit requires from_unit and to_unit"
                return self._convert_unit(float(value), from_unit, to_unit)
            elif tool_name == "evaluate_expression":
                expression = self._get_first_text(
                    arguments,
                    ["expression", "expr", "formula"],
                )
                return self._evaluate_expression(expression)
            else:
                return f"Error: unknown tool '{tool_name}'"
        except Exception as err:
            _LOGGER.exception("Calculator tool '%s' failed", tool_name)
            return f"Error: {err}"

        return self._normalize_result(result)
