"""Search tool for discovering tables and columns in Snowflake.

Mirrors: datahub-agent-context/mcp_tools/search.py

DataHub's search() queries Elasticsearch via GraphQL.
Ours queries Snowflake INFORMATION_SCHEMA via SQL.

The pattern is identical:
    1. get_connection() from context (not a function parameter)
    2. Execute query
    3. Clean response
    4. Return with pagination metadata
"""

import logging
from typing import List, Optional

from data_chat.context import get_connection
from data_chat.tools.base import clean_response, execute_query
from data_chat.tools.helpers import truncate_descriptions

logger = logging.getLogger(__name__)

# Hard cap — prevents the LLM from requesting 10,000 results
MAX_RESULTS = 50


def search(
    query: str,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    search_columns: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """Search for tables and views in Snowflake by name, comment, or column name.

    This is the primary discovery tool. Use it to find tables before calling
    get_tables() for detailed column information.

    SEARCH BEHAVIOR:
    - Matches table names and comments using case-insensitive substring matching
    - When search_columns=True, also searches column names
    - Results are ordered by match relevance (name matches first, then comment)

    TYPICAL WORKFLOW:
    1. Search: search(query="customer") → find tables with "customer" in the name
    2. Details: get_tables(["DB.SCHEMA.CUSTOMERS"]) → get columns and types
    3. Query: run_query("SELECT * FROM DB.SCHEMA.CUSTOMERS LIMIT 10") → see data

    SCOPING:
    - By default, searches the database/schema configured on the connection
    - Use database= and schema= to search a different scope
    - Omit schema to search all schemas in a database

    PAGINATION:
    - limit: Number of results per page (max: 50, default: 20)
    - offset: Starting position (default: 0)
    - Examples:
      * First page:  search("customer", limit=20, offset=0)
      * Second page: search("customer", limit=20, offset=20)

    Args:
        query: Search term (matched against table name and comment)
        database: Snowflake database to search (default: connection's database)
        schema: Snowflake schema to search (default: connection's schema, None = all schemas)
        search_columns: If True, also match against column names
        limit: Max results to return (max 50)
        offset: Pagination offset

    Returns:
        Dictionary with:
        - results: List of matching tables with name, type, comment, database, schema
        - total: Total number of matches (for pagination)
        - returned: Number of results in this page
        - has_more: Whether more results are available
        - offset: The offset used

    Example:
        from data_chat import DataChatContext
        from data_chat.client import SnowflakeClient

        client = SnowflakeClient.from_env()
        with DataChatContext(client):
            result = search(query="customer")
    """
    conn = get_connection()

    # Hard cap — DataHub does the same: min(num_results, 50)
    limit = min(limit, MAX_RESULTS)

    # Build WHERE clauses
    conditions: List[str] = []
    params: dict = {"query": f"%{query}%"}

    # Table name or comment match
    conditions.append(
        "(t.TABLE_NAME ILIKE %(query)s OR t.COMMENT ILIKE %(query)s)"
    )

    if search_columns:
        # Also match column names via a subquery
        conditions.append(
            "OR EXISTS ("
            "  SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS c"
            "  WHERE c.TABLE_CATALOG = t.TABLE_CATALOG"
            "  AND c.TABLE_SCHEMA = t.TABLE_SCHEMA"
            "  AND c.TABLE_NAME = t.TABLE_NAME"
            "  AND c.COLUMN_NAME ILIKE %(query)s"
            ")"
        )

    # Scope to database/schema
    scope_conditions: List[str] = []
    if database:
        scope_conditions.append("t.TABLE_CATALOG = %(database)s")
        params["database"] = database
    if schema:
        scope_conditions.append("t.TABLE_SCHEMA = %(schema)s")
        params["schema"] = schema

    # Exclude Snowflake internal schemas
    scope_conditions.append("t.TABLE_SCHEMA != 'INFORMATION_SCHEMA'")

    # Build the full query
    where_parts = " ".join(conditions)
    if scope_conditions:
        where_parts = f"({where_parts}) AND {' AND '.join(scope_conditions)}"

    # Count query (for pagination metadata)
    count_sql = f"""
        SELECT COUNT(*) as total
        FROM INFORMATION_SCHEMA.TABLES t
        WHERE {where_parts}
    """

    # Results query with ordering:
    #   - Name matches rank higher than comment-only matches
    #   - Then alphabetical by full path
    results_sql = f"""
        SELECT
            t.TABLE_CATALOG as database,
            t.TABLE_SCHEMA as schema,
            t.TABLE_NAME as name,
            t.TABLE_TYPE as type,
            t.COMMENT as comment,
            t.ROW_COUNT as row_count,
            t.CREATED as created,
            t.LAST_ALTERED as last_altered,
            CASE WHEN t.TABLE_NAME ILIKE %(query)s THEN 0 ELSE 1 END as _rank
        FROM INFORMATION_SCHEMA.TABLES t
        WHERE {where_parts}
        ORDER BY _rank, t.TABLE_CATALOG, t.TABLE_SCHEMA, t.TABLE_NAME
        LIMIT %(limit)s OFFSET %(offset)s
    """
    params["limit"] = limit
    params["offset"] = offset

    # Execute both queries
    count_rows = execute_query(conn, sql=count_sql, params=params)
    total = count_rows[0]["total"] if count_rows else 0

    result_rows = execute_query(conn, sql=results_sql, params=params)

    # Clean results — strip None values, truncate long comments
    cleaned = []
    for row in result_rows:
        # Remove the internal _rank column (used for ordering only)
        row.pop("_rank", None)
        cleaned.append(clean_response(row))

    truncate_descriptions(cleaned)

    return {
        "results": cleaned,
        "total": total,
        "returned": len(cleaned),
        "has_more": (offset + len(cleaned)) < total,
        "offset": offset,
    }
