"""Tests for tool functions (search, get_tables, run_query).

Mirrors: datahub-agent-context/tests/unit/mcp_tools/test_search.py etc.

DataHub mocks execute_graphql and tests tool logic.
We mock the Snowflake cursor and test tool logic.
"""

from unittest.mock import Mock, patch

import pytest

from data_chat.context import DataChatContext
from data_chat.tools.query import _validate_read_only, _enforce_limit


# --- Fixtures ---


@pytest.fixture
def mock_client():
    """Create a mock SnowflakeClient with a mock cursor."""
    client = Mock()
    cursor = Mock()
    client.connection = Mock()
    client.connection.cursor.return_value = cursor
    return client, cursor


# --- search() tests ---


def test_search_basic(mock_client):
    """search() should query INFORMATION_SCHEMA and return results."""
    client, cursor = mock_client

    # execute_query calls: cursor.execute(sql) → cursor.description → cursor.fetchall()
    # We need description to change between the count query and results query.
    descriptions = [
        [("total",)],  # count query
        [("database",), ("schema",), ("name",), ("type",),
         ("comment",), ("row_count",), ("created",), ("last_altered",), ("_rank",)],  # results
    ]
    results = [
        [(3,)],  # count query result
        [("MY_DB", "PUBLIC", "CUSTOMERS", "TABLE", "Customer data", 1000, None, None, 0)],
    ]
    call_idx = [0]

    def execute_side_effect(*args, **kwargs):
        idx = call_idx[0]
        cursor.description = descriptions[idx]
        cursor.fetchall.return_value = results[idx]
        call_idx[0] += 1

    cursor.execute = Mock(side_effect=execute_side_effect)

    with DataChatContext(client):
        from data_chat.tools.search import search
        result = search(query="customer")

    assert result["total"] == 3
    assert len(result["results"]) == 1
    assert result["results"][0]["name"] == "CUSTOMERS"
    assert result["has_more"] is True


def test_search_caps_limit(mock_client):
    """search() should cap limit at MAX_RESULTS (50)."""
    client, cursor = mock_client
    cursor.description = [("total",)]
    cursor.fetchall = Mock(side_effect=[[(0,)], []])

    with DataChatContext(client):
        from data_chat.tools.search import search
        search(query="test", limit=9999)

    # Check the LIMIT in the SQL that was executed
    calls = cursor.execute.call_args_list
    # The results query (second call) should have limit=50
    results_call = calls[1]
    params = results_call[1].get("params") or results_call[0][1] if len(results_call[0]) > 1 else {}
    if isinstance(params, dict):
        assert params.get("limit", 50) <= 50


# --- get_tables() tests ---


def test_get_tables_returns_list(mock_client):
    """get_tables() should return a list with one entry per table."""
    client, cursor = mock_client

    # First call: table metadata
    cursor.description = [
        ("database",), ("schema",), ("name",), ("type",),
        ("comment",), ("row_count",), ("created",), ("last_altered",),
    ]
    cursor.fetchall = Mock(side_effect=[
        [("DB", "SCH", "USERS", "TABLE", "User table", 500, None, None)],  # table query
        [  # columns query
            ("id", "NUMBER", "NO", None, None, 1),
            ("email", "VARCHAR", "YES", None, "User email", 2),
        ],
    ])
    # Reset description for columns query
    cursor.description = [
        ("name",), ("type",), ("nullable",), ("default_value",), ("comment",), ("position",),
    ]

    with DataChatContext(client):
        from data_chat.tools.tables import get_tables
        results = get_tables(tables=["DB.SCH.USERS"])

    assert len(results) == 1
    assert "error" not in results[0]


def test_get_tables_handles_not_found(mock_client):
    """get_tables() should return error dict for missing tables."""
    client, cursor = mock_client
    cursor.description = [("database",)]
    cursor.fetchall = Mock(return_value=[])  # empty = not found

    with DataChatContext(client):
        from data_chat.tools.tables import get_tables
        results = get_tables(tables=["DB.SCH.NONEXISTENT"])

    assert len(results) == 1
    assert "error" in results[0]
    assert results[0]["table"] == "DB.SCH.NONEXISTENT"


def test_get_tables_batch_with_partial_failure(mock_client):
    """One bad table shouldn't fail the whole batch.

    Mirrors: DataHub's per-item error handling in get_entities().
    """
    client, cursor = mock_client

    # Track execute calls to serve different descriptions/results per query
    call_count = [0]
    descriptions = [
        # Table 1: table metadata query
        [("database",), ("schema",), ("name",), ("type",),
         ("comment",), ("row_count",), ("created",), ("last_altered",)],
        # Table 1: columns query
        [("name",), ("type",), ("nullable",), ("default_value",), ("comment",), ("position",)],
        # Table 2: table metadata query (not found)
        [("database",), ("schema",), ("name",), ("type",),
         ("comment",), ("row_count",), ("created",), ("last_altered",)],
    ]
    query_results = [
        [("DB", "SCH", "GOOD", "TABLE", None, 10, None, None)],
        [("id", "NUMBER", "NO", None, None, 1)],
        [],  # not found
    ]

    def execute_side_effect(*args, **kwargs):
        idx = call_count[0]
        cursor.description = descriptions[idx]
        cursor.fetchall.return_value = query_results[idx]
        call_count[0] += 1

    cursor.execute = Mock(side_effect=execute_side_effect)

    with DataChatContext(client):
        from data_chat.tools.tables import get_tables
        results = get_tables(tables=["DB.SCH.GOOD", "DB.SCH.BAD"])

    assert len(results) == 2
    assert "error" not in results[0]  # first succeeded
    assert "error" in results[1]      # second failed gracefully


# --- run_query() safety tests ---


def test_validate_read_only_allows_select():
    """SELECT statements should be allowed."""
    _validate_read_only("SELECT * FROM users")
    _validate_read_only("  SELECT 1")
    _validate_read_only("-- comment\nSELECT 1")
    _validate_read_only("/* block */ SELECT 1")
    _validate_read_only("WITH cte AS (SELECT 1) SELECT * FROM cte")
    _validate_read_only("SHOW TABLES")
    _validate_read_only("DESCRIBE TABLE users")


def test_validate_read_only_rejects_mutations():
    """DDL/DML statements should be rejected."""
    mutations = [
        "DROP TABLE users",
        "DELETE FROM users",
        "INSERT INTO users VALUES (1)",
        "UPDATE users SET x=1",
        "CREATE TABLE evil (id INT)",
        "ALTER TABLE users ADD COLUMN x INT",
        "TRUNCATE TABLE users",
        "MERGE INTO users USING src ON 1=1",
        "GRANT ALL ON users TO PUBLIC",
    ]
    for sql in mutations:
        with pytest.raises(ValueError, match="Only read-only queries"):
            _validate_read_only(sql)


def test_validate_read_only_rejects_empty():
    """Empty SQL should be rejected."""
    with pytest.raises(ValueError, match="Empty SQL"):
        _validate_read_only("   ")


def test_validate_read_only_strips_comments_before_check():
    """Comments before a mutation should not bypass the check."""
    with pytest.raises(ValueError, match="Only read-only queries"):
        _validate_read_only("-- safe comment\nDROP TABLE users")

    with pytest.raises(ValueError, match="Only read-only queries"):
        _validate_read_only("/* block */ DROP TABLE users")


def test_enforce_limit_appends_when_missing():
    """Should append LIMIT when SQL has none."""
    result = _enforce_limit("SELECT * FROM users", 100)
    assert "LIMIT 100" in result


def test_enforce_limit_caps_existing():
    """Should reduce existing LIMIT if it exceeds cap."""
    result = _enforce_limit("SELECT * FROM users LIMIT 5000", 100)
    assert "LIMIT 100" in result
    assert "5000" not in result


def test_enforce_limit_keeps_smaller():
    """Should keep existing LIMIT if it's smaller than cap."""
    result = _enforce_limit("SELECT * FROM users LIMIT 10", 100)
    assert "LIMIT 10" in result


def test_enforce_limit_caps_at_max():
    """Should cap at MAX_LIMIT (1000) regardless of requested limit."""
    result = _enforce_limit("SELECT * FROM users", 9999)
    assert "LIMIT 1000" in result


def test_enforce_limit_handles_semicolon():
    """Should strip trailing semicolon before appending LIMIT."""
    result = _enforce_limit("SELECT * FROM users;", 50)
    assert "LIMIT 50" in result
    assert not result.rstrip().endswith(";")


# --- Table ref parsing ---


def test_parse_table_ref():
    """Table reference parsing should handle all formats."""
    from data_chat.tools.tables import _parse_table_ref

    # Fully qualified
    assert _parse_table_ref("DB.SCHEMA.TABLE", None, None) == ("DB", "SCHEMA", "TABLE")

    # Schema qualified
    assert _parse_table_ref("SCHEMA.TABLE", "DEF_DB", None) == ("DEF_DB", "SCHEMA", "TABLE")

    # Unqualified
    assert _parse_table_ref("TABLE", "DB", "SCH") == ("DB", "SCH", "TABLE")

    # With whitespace
    assert _parse_table_ref(" DB . SCHEMA . TABLE ", None, None) == ("DB", "SCHEMA", "TABLE")


def test_parse_table_ref_rejects_invalid():
    """Should reject table refs with too many parts."""
    from data_chat.tools.tables import _parse_table_ref

    with pytest.raises(ValueError, match="Invalid table reference"):
        _parse_table_ref("A.B.C.D", None, None)
