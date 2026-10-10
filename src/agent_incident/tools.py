"""Tool registry and execution. Everything the model asks for is validated here."""

from agent_incident.incident import INCIDENT_LOGS

MAX_QUERY_LENGTH = 100
MAX_RESULTS = 10


class ToolError(Exception):
    """A tool call was invalid. The message is written to be shown to the model."""


def search_logs(query: str) -> dict:
    """Case-insensitive literal substring search over the incident logs."""
    needle = query.lower()
    matches = [line for line in INCIDENT_LOGS if needle in line.lower()]
    return {
        "query": query,
        "match_count": len(matches),
        "lines": matches[:MAX_RESULTS],
        "truncated": len(matches) > MAX_RESULTS,
    }


def _validate_search_logs(arguments: dict) -> dict:
    unexpected = set(arguments) - {"query"}
    if unexpected:
        raise ToolError(
            f"Unexpected arguments: {sorted(unexpected)}. Only 'query' is allowed."
        )

    query = arguments.get("query")
    if not isinstance(query, str):
        raise ToolError("'query' is required and must be a string.")

    query = query.strip()
    if not query:
        raise ToolError("'query' must not be empty.")
    if len(query) > MAX_QUERY_LENGTH:
        raise ToolError(f"'query' must be at most {MAX_QUERY_LENGTH} characters.")

    return {"query": query}


TOOLS = {"search_logs": search_logs}
VALIDATORS = {"search_logs": _validate_search_logs}

# What we describe to the model. It is a schema, not code: the model can only
# ask for a call, and execute_tool decides whether to honor it.
SUBMIT_TOOL_NAME = "submit_root_cause"

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_logs",
            "description": "Search application logs for lines containing a text query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for in log lines.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": SUBMIT_TOOL_NAME,
            "description": (
                "Submit your final root-cause conclusion. Call this once, when "
                "you have found evidence for the cause. This ends the investigation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One or two sentences stating the root cause.",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> dict:
    """Validate a model-requested tool call, then run it."""
    if name not in TOOLS:
        raise ToolError(f"Unknown tool {name!r}. Available tools: {sorted(TOOLS)}.")
    if not isinstance(arguments, dict):
        raise ToolError("Tool arguments must be a JSON object.")

    clean_arguments = VALIDATORS[name](arguments)
    return TOOLS[name](**clean_arguments)
