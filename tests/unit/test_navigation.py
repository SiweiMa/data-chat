"""Tests for list_databases() and list_schemas() tools."""

from unittest.mock import Mock

import pytest

from data_chat.context import DataChatContext


# --- Fixtures ---


@pytest.fixture
def mock_client():
    """Create a mock SnowflakeClient with a mock cursor."""
    client = Mock()
    cursor = Mock()
    client.connection = Mock()
    client.connection.cursor.return_value = cursor
    # Default database for list_schemas fallback
    client.connection.database = "DEFAULT_DB"
    return client, cursor


# --- list_databases() tests ---


def test_list_databases_basic(mock_client):
    """list_databases() should return database names from SHOW DATABASES."""
    client, cursor = mock_client

    cursor.description = [
        ("created_on",), ("name",), ("is_default",), ("is_current",),
        ("origin",), ("owner",), ("comment",), ("options",), ("retention_time",),
    ]
    cursor.fetchall.return_value = [
        ("2024-01-01", "PROD_DB", "N", "N", "", "SYSADMIN", "Production", "", "1"),
        ("2024-01-02", "DEV_DB", "N", "N", "", "SYSADMIN", "Development", "", "1"),
        ("2024-01-03", "AI_DATAMART", "N", "Y", "", "SYSADMIN", "AI data", "", "1"),
    ]

    with DataChatContext(client):
        from data_chat.tools.navigation import list_databases
        result = list_databases()

    assert result["total"] == 3
    assert result["returned"] == 3
    assert result["has_more"] is False
    assert len(result["databases"]) == 3
    assert result["databases"][0]["name"] == "PROD_DB"


def test_list_databases_with_filter(mock_client):
    """list_databases(name_filter=...) should filter by name."""
    client, cursor = mock_client

    cursor.description = [
        ("created_on",), ("name",), ("is_default",), ("is_current",),
        ("origin",), ("owner",), ("comment",), ("options",), ("retention_time",),
    ]
    cursor.fetchall.return_value = [
        ("2024-01-01", "PROD_DB", "N", "N", "", "SYSADMIN", "", "", "1"),
        ("2024-01-02", "DEV_DB", "N", "N", "", "SYSADMIN", "", "", "1"),
        ("2024-01-03", "PROD_ANALYTICS", "N", "N", "", "SYSADMIN", "", "", "1"),
    ]

    with DataChatContext(client):
        from data_chat.tools.navigation import list_databases
        result = list_databases(name_filter="prod")

    assert result["total"] == 2
    assert result["returned"] == 2
    names = [d["name"] for d in result["databases"]]
    assert "PROD_DB" in names
    assert "PROD_ANALYTICS" in names
    assert "DEV_DB" not in names


def test_list_databases_caps_limit(mock_client):
    """Limit should be capped at MAX_RESULTS (50)."""
    client, cursor = mock_client

    cursor.description = [("created_on",), ("name",), ("owner",), ("comment",)]
    cursor.fetchall.return_value = [
        ("2024-01-01", f"DB_{i}", "SYSADMIN", "")
        for i in range(60)
    ]

    with DataChatContext(client):
        from data_chat.tools.navigation import list_databases
        result = list_databases(limit=9999)

    assert result["returned"] == 50
    assert result["total"] == 60
    assert result["has_more"] is True


def test_list_databases_pagination(mock_client):
    """Offset should skip initial results."""
    client, cursor = mock_client

    cursor.description = [("created_on",), ("name",), ("owner",), ("comment",)]
    cursor.fetchall.return_value = [
        ("2024-01-01", f"DB_{i}", "SYSADMIN", "")
        for i in range(5)
    ]

    with DataChatContext(client):
        from data_chat.tools.navigation import list_databases
        result = list_databases(offset=3, limit=10)

    assert result["returned"] == 2
    assert result["total"] == 5
    assert result["offset"] == 3
    assert result["has_more"] is False


# --- list_schemas() tests ---


def test_list_schemas_basic(mock_client):
    """list_schemas() should return schemas from INFORMATION_SCHEMA."""
    client, cursor = mock_client

    cursor.description = [
        ("name",), ("database",), ("owner",), ("created_on",), ("comment",),
    ]
    cursor.fetchall.return_value = [
        ("PUBLIC", "DEFAULT_DB", "SYSADMIN", "2024-01-01", "Default schema"),
        ("RAW", "DEFAULT_DB", "SYSADMIN", "2024-01-02", "Raw data"),
    ]

    with DataChatContext(client):
        from data_chat.tools.navigation import list_schemas
        result = list_schemas()

    assert result["database"] == "DEFAULT_DB"
    assert result["total"] == 2
    assert result["returned"] == 2
    assert len(result["schemas"]) == 2
    assert result["schemas"][0]["name"] == "PUBLIC"


def test_list_schemas_with_database(mock_client):
    """list_schemas(database=...) should query the specified database."""
    client, cursor = mock_client

    cursor.description = [
        ("name",), ("database",), ("owner",), ("created_on",), ("comment",),
    ]
    cursor.fetchall.return_value = [
        ("PUBLIC", "MY_DB", "SYSADMIN", "2024-01-01", None),
    ]

    with DataChatContext(client):
        from data_chat.tools.navigation import list_schemas
        result = list_schemas(database="my_db")

    # Should uppercase the database name
    assert result["database"] == "MY_DB"
    # SQL should reference MY_DB.INFORMATION_SCHEMA.SCHEMATA
    executed_sql = cursor.execute.call_args[0][0]
    assert "MY_DB.INFORMATION_SCHEMA.SCHEMATA" in executed_sql


def test_list_schemas_validates_database_name(mock_client):
    """Invalid database name should raise ValueError."""
    client, _ = mock_client

    with DataChatContext(client):
        from data_chat.tools.navigation import list_schemas
        with pytest.raises(ValueError, match="Invalid database"):
            list_schemas(database="DB; DROP TABLE --")


def test_list_schemas_with_filter(mock_client):
    """list_schemas(name_filter=...) should filter by name."""
    client, cursor = mock_client

    cursor.description = [
        ("name",), ("database",), ("owner",), ("created_on",), ("comment",),
    ]
    cursor.fetchall.return_value = [
        ("PUBLIC", "DB", "SYSADMIN", "2024-01-01", None),
        ("RAW_DATA", "DB", "SYSADMIN", "2024-01-02", None),
        ("RAW_STAGING", "DB", "SYSADMIN", "2024-01-03", None),
    ]

    with DataChatContext(client):
        from data_chat.tools.navigation import list_schemas
        result = list_schemas(name_filter="raw")

    assert result["total"] == 2
    names = [s["name"] for s in result["schemas"]]
    assert "RAW_DATA" in names
    assert "RAW_STAGING" in names
    assert "PUBLIC" not in names


def test_list_schemas_excludes_information_schema(mock_client):
    """INFORMATION_SCHEMA should be excluded via the SQL WHERE clause."""
    client, cursor = mock_client

    cursor.description = [
        ("name",), ("database",), ("owner",), ("created_on",), ("comment",),
    ]
    cursor.fetchall.return_value = []

    with DataChatContext(client):
        from data_chat.tools.navigation import list_schemas
        list_schemas()

    executed_sql = cursor.execute.call_args[0][0]
    assert "INFORMATION_SCHEMA" in executed_sql
    assert "!= 'INFORMATION_SCHEMA'" in executed_sql
