"""Tests for get_lineage() tool and validate_identifier()."""

from unittest.mock import Mock

import pytest

from data_chat.context import DataChatContext
from data_chat.tools.base import validate_identifier


# --- Fixtures ---


@pytest.fixture
def mock_client():
    """Create a mock SnowflakeClient with a mock cursor."""
    client = Mock()
    cursor = Mock()
    client.connection = Mock()
    client.connection.cursor.return_value = cursor
    return client, cursor


# --- validate_identifier() tests ---


def test_validate_identifier_valid():
    """Should accept and uppercase valid identifiers."""
    assert validate_identifier("my_db.my_schema.my_table") == "MY_DB.MY_SCHEMA.MY_TABLE"
    assert validate_identifier("DB") == "DB"
    assert validate_identifier("DB.SCHEMA") == "DB.SCHEMA"
    assert validate_identifier("DB.SCHEMA.TABLE") == "DB.SCHEMA.TABLE"
    assert validate_identifier("abc_123") == "ABC_123"


def test_validate_identifier_rejects_injection():
    """Should reject identifiers with SQL injection characters."""
    bad_names = [
        "DB; DROP TABLE --",
        "DB.SCHEMA.TABLE'",
        "DB.SCHEMA.TABLE; DELETE FROM x",
        "DB SCHEMA TABLE",
        "DB.SCHEMA.TABLE.EXTRA.PARTS",
        "",
        "DB..TABLE",
        ".SCHEMA.TABLE",
    ]
    for name in bad_names:
        with pytest.raises(ValueError, match="Invalid"):
            validate_identifier(name)


# --- get_lineage() tests ---


def test_get_lineage_basic(mock_client):
    """get_lineage() should query GET_LINEAGE and return structured results."""
    client, cursor = mock_client

    cursor.description = [
        ("SOURCE_OBJECT_DATABASE",), ("SOURCE_OBJECT_SCHEMA",), ("SOURCE_OBJECT_NAME",),
        ("SOURCE_COLUMN_NAME",),
        ("TARGET_OBJECT_DATABASE",), ("TARGET_OBJECT_SCHEMA",), ("TARGET_OBJECT_NAME",),
        ("TARGET_COLUMN_NAME",),
        ("DISTANCE",),
    ]
    cursor.fetchall.return_value = [
        ("DB", "SCH", "SOURCE_TABLE", None, "DB", "SCH", "TARGET_VIEW", None, 1),
        ("DB", "SCH", "TARGET_VIEW", None, "DB", "SCH", "FINAL_TABLE", None, 2),
    ]

    with DataChatContext(client):
        from data_chat.tools.lineage import get_lineage
        result = get_lineage("DB.SCH.SOURCE_TABLE")

    assert result["object_name"] == "DB.SCH.SOURCE_TABLE"
    assert result["direction"] == "DOWNSTREAM"
    assert result["distance"] == 3
    assert result["total"] == 2
    assert result["returned"] == 2
    assert result["has_more"] is False
    assert len(result["lineage"]) == 2
    assert result["lineage"][0]["source_name"] == "SOURCE_TABLE"
    assert result["lineage"][0]["target_name"] == "TARGET_VIEW"
    assert result["lineage"][0]["distance"] == 1


def test_get_lineage_validates_direction(mock_client):
    """Invalid direction should raise ValueError."""
    client, _ = mock_client

    with DataChatContext(client):
        from data_chat.tools.lineage import get_lineage
        with pytest.raises(ValueError, match="Invalid direction"):
            get_lineage("DB.SCH.TABLE", direction="SIDEWAYS")


def test_get_lineage_validates_object_domain(mock_client):
    """Invalid object_domain should raise ValueError."""
    client, _ = mock_client

    with DataChatContext(client):
        from data_chat.tools.lineage import get_lineage
        with pytest.raises(ValueError, match="Invalid object_domain"):
            get_lineage("DB.SCH.TABLE", object_domain="FUNCTION")


def test_get_lineage_caps_distance(mock_client):
    """Distance should be capped at MAX_DISTANCE (5)."""
    client, cursor = mock_client

    cursor.description = [
        ("SOURCE_OBJECT_DATABASE",), ("SOURCE_OBJECT_SCHEMA",), ("SOURCE_OBJECT_NAME",),
        ("SOURCE_COLUMN_NAME",),
        ("TARGET_OBJECT_DATABASE",), ("TARGET_OBJECT_SCHEMA",), ("TARGET_OBJECT_NAME",),
        ("TARGET_COLUMN_NAME",),
        ("DISTANCE",),
    ]
    cursor.fetchall.return_value = []

    with DataChatContext(client):
        from data_chat.tools.lineage import get_lineage
        result = get_lineage("DB.SCH.TABLE", distance=99)

    # Check the SQL contains distance=5, not 99
    executed_sql = cursor.execute.call_args[0][0]
    assert "99" not in executed_sql
    assert ", 5\n" in executed_sql
    assert result["distance"] == 5


def test_get_lineage_validates_object_name(mock_client):
    """SQL injection in object_name should raise ValueError."""
    client, _ = mock_client

    with DataChatContext(client):
        from data_chat.tools.lineage import get_lineage
        with pytest.raises(ValueError, match="Invalid object_name"):
            get_lineage("DB.SCH.TABLE'; DROP TABLE --")


def test_get_lineage_uppercases_name(mock_client):
    """Lowercase object names should be uppercased in SQL."""
    client, cursor = mock_client

    cursor.description = [
        ("SOURCE_OBJECT_DATABASE",), ("SOURCE_OBJECT_SCHEMA",), ("SOURCE_OBJECT_NAME",),
        ("SOURCE_COLUMN_NAME",),
        ("TARGET_OBJECT_DATABASE",), ("TARGET_OBJECT_SCHEMA",), ("TARGET_OBJECT_NAME",),
        ("TARGET_COLUMN_NAME",),
        ("DISTANCE",),
    ]
    cursor.fetchall.return_value = []

    with DataChatContext(client):
        from data_chat.tools.lineage import get_lineage
        result = get_lineage("db.sch.my_table")

    executed_sql = cursor.execute.call_args[0][0]
    assert "'DB.SCH.MY_TABLE'" in executed_sql
    assert result["object_name"] == "DB.SCH.MY_TABLE"


def test_get_lineage_enterprise_error(mock_client):
    """ProgrammingError mentioning GET_LINEAGE should return error dict."""
    client, cursor = mock_client

    from snowflake.connector.errors import ProgrammingError
    cursor.execute.side_effect = ProgrammingError(
        "Unknown function SNOWFLAKE.CORE.GET_LINEAGE"
    )

    with DataChatContext(client):
        from data_chat.tools.lineage import get_lineage
        result = get_lineage("DB.SCH.TABLE")

    assert "error" in result
    assert "Enterprise Edition" in result["error"]
    assert result["object_name"] == "DB.SCH.TABLE"


def test_get_lineage_empty_result(mock_client):
    """No lineage should return empty list with total=0."""
    client, cursor = mock_client

    cursor.description = [
        ("SOURCE_OBJECT_DATABASE",), ("SOURCE_OBJECT_SCHEMA",), ("SOURCE_OBJECT_NAME",),
        ("SOURCE_COLUMN_NAME",),
        ("TARGET_OBJECT_DATABASE",), ("TARGET_OBJECT_SCHEMA",), ("TARGET_OBJECT_NAME",),
        ("TARGET_COLUMN_NAME",),
        ("DISTANCE",),
    ]
    cursor.fetchall.return_value = []

    with DataChatContext(client):
        from data_chat.tools.lineage import get_lineage
        result = get_lineage("DB.SCH.ISOLATED_TABLE")

    assert result["lineage"] == []
    assert result["total"] == 0
    assert result["returned"] == 0
    assert result["has_more"] is False
