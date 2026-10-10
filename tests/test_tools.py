import pytest

from agent_incident.tools import MAX_QUERY_LENGTH, MAX_RESULTS, ToolError, execute_tool


def test_search_finds_the_root_cause_lines():
    result = execute_tool("search_logs", {"query": "pool"})

    assert result["match_count"] == 4
    assert all("pool" in line.lower() for line in result["lines"])


def test_search_is_case_insensitive():
    lower = execute_tool("search_logs", {"query": "db_pool_size"})
    upper = execute_tool("search_logs", {"query": "DB_POOL_SIZE"})

    assert lower["match_count"] == upper["match_count"] == 1


def test_query_with_no_matches_is_a_valid_empty_result_not_an_error():
    # The exact query the real model produced in our probe.
    result = execute_tool("search_logs", {"query": "checkout-api p99 latency 4s"})

    assert result["match_count"] == 0
    assert result["lines"] == []


def test_query_is_stripped_of_surrounding_whitespace():
    result = execute_tool("search_logs", {"query": "  pool  "})

    assert result["query"] == "pool"
    assert result["match_count"] == 4


def test_results_are_capped_and_flagged_as_truncated():
    result = execute_tool("search_logs", {"query": "2026"})  # every line matches

    assert result["match_count"] == 20
    assert len(result["lines"]) == MAX_RESULTS
    assert result["truncated"] is True


def test_query_at_exactly_the_maximum_length_is_accepted():
    result = execute_tool("search_logs", {"query": "x" * MAX_QUERY_LENGTH})

    assert result["match_count"] == 0


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("search_logs", {}),
        ("search_logs", {"query": ""}),
        ("search_logs", {"query": "   "}),
        ("search_logs", {"query": 42}),
        ("search_logs", {"query": None}),
        ("search_logs", {"query": "pool", "limit": 5}),
        ("search_logs", {"query": "x" * (MAX_QUERY_LENGTH + 1)}),
        ("search_logs", "pool"),
        ("delete_logs", {"query": "pool"}),
    ],
    ids=[
        "missing-query",
        "empty-query",
        "whitespace-query",
        "non-string-query",
        "null-query",
        "unexpected-argument",
        "query-too-long",
        "arguments-not-an-object",
        "unknown-tool",
    ],
)
def test_invalid_tool_calls_are_rejected(name, arguments):
    with pytest.raises(ToolError):
        execute_tool(name, arguments)
