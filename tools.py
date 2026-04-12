"""
Fourth Umpire AI — Tool Definitions & Execution

Provides tools that Claude can call during ruling generation:
  1. calculator      — General-purpose arithmetic
  2. overs_to_balls  — Cricket overs notation → total balls
  3. balls_to_overs  — Total balls → cricket overs notation
"""


# ═══════════════════════════════════════════════════════════
# TOOL SCHEMAS (Anthropic API format)
# ═══════════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    {
        "name": "calculator",
        "description": (
            "Evaluates a mathematical expression and returns the exact numeric result. "
            "Use this for any arithmetic: run rates, strike rates, economy rates, "
            "required run rates, percentages, averages, Net Run Rate, etc. "
            "Pass a valid Python math expression as a string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A mathematical expression to evaluate, "
                        "e.g. '(240 - 110) / 25' or '(45 / 30) * 100'"
                    ),
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "overs_to_balls",
        "description": (
            "Converts cricket overs notation to total number of balls. "
            "In cricket, 25.3 overs means 25 complete overs and 3 balls, "
            "which equals 153 balls (NOT 25.3 * 6). "
            "Use this whenever you need to convert overs to balls for further calculation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "overs": {
                    "type": "number",
                    "description": (
                        "Overs in cricket notation, e.g. 25.3 means 25 overs and 3 balls"
                    ),
                }
            },
            "required": ["overs"],
        },
    },
    {
        "name": "balls_to_overs",
        "description": (
            "Converts a total number of balls into cricket overs notation. "
            "For example, 153 balls = 25.3 overs (25 complete overs and 3 balls). "
            "Use this when you need to express a ball count in overs format."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "balls": {
                    "type": "integer",
                    "description": "Total number of balls, e.g. 153",
                }
            },
            "required": ["balls"],
        },
    },
]


# ═══════════════════════════════════════════════════════════
# EXECUTION FUNCTIONS
# ═══════════════════════════════════════════════════════════

def execute_calculator(expression: str) -> str:
    """Safely evaluate a math expression. Returns the result as a string."""
    # Only allow safe characters: digits, operators, parens, decimal points, spaces
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return f"Error: expression contains disallowed characters. Only digits and +-*/.() are permitted."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        # Format nicely: if it's a whole number, show without decimal
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        if isinstance(result, float):
            return str(round(result, 4))
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def execute_overs_to_balls(overs: float) -> str:
    """Convert cricket overs notation to total balls.

    25.3 overs = 25 complete overs + 3 balls = 153 balls.
    Uses round() to handle floating-point precision (25.3 - 25 = 0.29999...).
    """
    complete_overs = int(overs)
    remaining_balls = round((overs - complete_overs) * 10)
    if remaining_balls < 0 or remaining_balls > 5:
        return f"Error: invalid overs notation {overs}. The decimal part must be 0-5 (balls in an over)."
    total = complete_overs * 6 + remaining_balls
    return str(total)


def execute_balls_to_overs(balls: int) -> str:
    """Convert total balls to cricket overs notation.

    153 balls = 25 complete overs + 3 remaining = 25.3 overs.
    """
    complete_overs = balls // 6
    remaining = balls % 6
    return f"{complete_overs}.{remaining}"


# ═══════════════════════════════════════════════════════════
# DISPATCHER
# ═══════════════════════════════════════════════════════════

def _get_input(tool_input: dict, expected_key: str):
    """Get tool input value by expected key, falling back to first dict value."""
    if expected_key in tool_input:
        return tool_input[expected_key]
    # Fallback: try the first value in the dict (model used a different key name)
    if tool_input:
        return next(iter(tool_input.values()))
    return None


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Route a tool call to the right function. Returns result as string."""
    try:
        if tool_name == "calculator":
            expr = _get_input(tool_input, "expression")
            if expr is None:
                return "Error: no expression provided"
            return execute_calculator(str(expr))
        elif tool_name == "overs_to_balls":
            overs = _get_input(tool_input, "overs")
            if overs is None:
                return "Error: no overs value provided"
            return execute_overs_to_balls(float(overs))
        elif tool_name == "balls_to_overs":
            balls = _get_input(tool_input, "balls")
            if balls is None:
                return "Error: no balls value provided"
            return execute_balls_to_overs(int(balls))
        else:
            return f"Error: unknown tool '{tool_name}'"
    except Exception as e:
        return f"Error executing {tool_name}: {e}"
