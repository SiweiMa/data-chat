"""Tools for navigating the Snowflake catalog hierarchy.

Provides list_databases() and list_schemas() for discovering available
databases and schemas before using search() to find specific tables.

TYPICAL WORKFLOW:
    1. list_databases() → see what databases are available
    2. list_schemas(database="MY_DB") → see schemas in a database
    3. search("customer", database="MY_DB", schema="PUBLIC") → find tables
    4. get_tables(["MY_DB.PUBLIC.CUSTOMERS"]) → get column details
    5. run_query("SELECT * FROM MY_DB.PUBLIC.CUSTOMERS LIMIT 10") → see data
"""

import logging
from typing import Optional

from data_chat.context import get_connection
from data_chat.tools.base import clean_response, execute_query, validate_identifier

logger = logging.getLogger(__name__)

MAX_RESULTS = 50


def list_databases(
    name_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """List all databases accessible to the current user.

    Use this tool to discover which databases are available before searching
    for tables. This is the starting point for exploring an unfamiliar
    Snowflake account.

    WHEN TO USE:
    - User asks "what databases do I have access to?"
    - User asks "show me all databases"
    - You need to know which databases exist before searching for tables
    - User references a database you haven't seen before

    FILTERING:
    - Use name_filter to search by database name (case-insensitive substring match)
    - Example: list_databases(name_filter="prod") → databases with "prod" in the name

    PAGINATION:
    - limit: Max results per page (max 50, default 50)
    - offset: Starting position (default 0)
    - Use has_more to determine if another page is available

    Args:
        name_filter: Optional substring to filter database names (case-insensitive)
        limit: Maximum number of databases to return (max 50)
        offset: Pagination offset

    Returns:
        Dictionary with:
        - databases: List of database info dicts (name, created, comment, etc.)
        - total: Total matching databases
        - returned: Number in this page
        - has_more: Whether more pages are available
        - offset: The offset used

    Example:
        from data_chat import DataChatContext
        from data_chat.client import SnowflakeClient

        client = SnowflakeClient.from_env()
        with DataChatContext(client):
            result = list_databases(name_filter="prod")
    """
    conn = get_connection()
    limit = min(limit, MAX_RESULTS)

    # SHOW DATABASES returns a fixed set of columns; we use execute_query
    # which normalizes the result into list-of-dicts.
    rows = execute_query(conn, sql="SHOW DATABASES")

    # Filter by name if requested
    if name_filter:
        filter_upper = name_filter.upper()
        rows = [r for r in rows if filter_upper in r.get("name", "").upper()]

    total = len(rows)

    # Paginate
    page = rows[offset : offset + limit]

    # Clean and simplify — SHOW DATABASES returns many columns,
    # keep only the useful ones to save tokens.
    cleaned = []
    for row in page:
        entry = {
            "name": row.get("name"),
            "comment": row.get("comment"),
            "created_on": row.get("created_on"),
            "owner": row.get("owner"),
        }
        cleaned.append(clean_response(entry))

    return {
        "databases": cleaned,
        "total": total,
        "returned": len(cleaned),
        "has_more": (offset + len(cleaned)) < total,
        "offset": offset,
    }


def list_schemas(
    database: Optional[str] = None,
    name_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """List all schemas in a Snowflake database.

    Use this tool to discover which schemas are available in a database
    before searching for tables within a specific schema.

    WHEN TO USE:
    - User asks "what schemas are in <database>?"
    - User asks "show me all schemas"
    - You need to narrow down which schema to search before calling search()
    - User references a schema you haven't seen before

    DATABASE PARAMETER:
    - If database is provided, lists schemas in that database
    - If database is None, lists schemas in the connection's default database
    - The database name is validated to prevent SQL injection

    FILTERING:
    - Use name_filter to search by schema name (case-insensitive substring match)
    - INFORMATION_SCHEMA is automatically excluded (it's a system schema)

    Args:
        database: Database to list schemas from (default: connection's database)
        name_filter: Optional substring to filter schema names (case-insensitive)
        limit: Maximum number of schemas to return (max 50)
        offset: Pagination offset

    Returns:
        Dictionary with:
        - database: The database queried
        - schemas: List of schema info dicts (name, created, comment, etc.)
        - total: Total matching schemas
        - returned: Number in this page
        - has_more: Whether more pages are available
        - offset: The offset used

    Example:
        from data_chat import DataChatContext
        from data_chat.client import SnowflakeClient

        client = SnowflakeClient.from_env()
        with DataChatContext(client):
            result = list_schemas(database="AI_DATAMART")
    """
    conn = get_connection()
    limit = min(limit, MAX_RESULTS)

    # Validate database name if provided (prevents SQL injection since
    # we embed it in the FROM clause as an identifier)
    if database:
        database = validate_identifier(database, "database")
    else:
        # Use connection's default database
        database = conn.database

    sql = f"""
        SELECT
            SCHEMA_NAME as name,
            CATALOG_NAME as database,
            SCHEMA_OWNER as owner,
            CREATED as created_on,
            COMMENT as comment
        FROM {database}.INFORMATION_SCHEMA.SCHEMATA
        WHERE SCHEMA_NAME != 'INFORMATION_SCHEMA'
        ORDER BY SCHEMA_NAME
    """

    rows = execute_query(conn, sql=sql)

    # Filter by name if requested
    if name_filter:
        filter_upper = name_filter.upper()
        rows = [r for r in rows if filter_upper in r.get("name", "").upper()]

    total = len(rows)

    # Paginate
    page = rows[offset : offset + limit]

    cleaned = []
    for row in page:
        cleaned.append(clean_response(row))

    return {
        "database": database,
        "schemas": cleaned,
        "total": total,
        "returned": len(cleaned),
        "has_more": (offset + len(cleaned)) < total,
        "offset": offset,
    }
