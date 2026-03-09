"""Tool for getting detailed table information from Snowflake.

Mirrors: datahub-agent-context/mcp_tools/entities.py (get_entities)

DataHub's get_entities() takes a list of URNs and returns detailed metadata
for each. Ours takes fully-qualified table names and returns column details.

Key patterns copied from DataHub:
    1. Batch operation — list in, list out (reduces LLM round-trips)
    2. Per-item error handling — one bad table doesn't fail the whole call
    3. Per-table token budget — wide tables get columns truncated
    4. Truncation flag — LLM knows to call list_columns() for more detail
"""

import logging
from typing import List, Optional

from data_chat.context import get_connection
from data_chat.tools.base import clean_response, execute_query
from data_chat.tools.helpers import (
    select_columns_within_budget,
    truncate_descriptions,
)

logger = logging.getLogger(__name__)


def get_tables(
    tables: List[str],
    database: Optional[str] = None,
    schema: Optional[str] = None,
) -> List[dict]:
    """Get detailed information about one or more Snowflake tables.

    IMPORTANT: Pass multiple table names in a single call — this is much more
    efficient than calling this tool multiple times. When examining search
    results, always pass the top 3-10 table names together to compare.

    Returns column details (name, type, nullable, comment) for each table,
    with per-table token budget enforcement. If a table has too many columns,
    the response includes columns_truncated=True — use list_columns() to
    explore specific columns by keyword.

    TABLE NAME FORMATS:
    - Fully qualified: "MY_DB.MY_SCHEMA.MY_TABLE"
    - Schema-qualified: "MY_SCHEMA.MY_TABLE" (uses connection's database)
    - Unqualified: "MY_TABLE" (uses database= and schema= params, or connection defaults)

    TYPICAL WORKFLOW:
    1. search("customer") → find tables
    2. get_tables(["DB.SCHEMA.CUSTOMERS", "DB.SCHEMA.CUSTOMER_ORDERS"]) → compare schemas
    3. run_query("SELECT * FROM DB.SCHEMA.CUSTOMERS LIMIT 10") → see data

    WHEN COLUMNS ARE TRUNCATED:
    If you see columns_truncated=True in the response, it means the table has
    more columns than could fit in the token budget. To find specific columns:
    - Call get_tables() again with just that one table (gets full per-table budget)
    - Or use search(query="email", search_columns=True) to find columns by name

    Args:
        tables: List of table names (fully qualified, schema-qualified, or unqualified)
        database: Default database for unqualified names (default: connection's database)
        schema: Default schema for unqualified names (default: connection's schema)

    Returns:
        List of dicts, one per table. Each contains:
        - database, schema, name: Fully qualified identity
        - type: TABLE or VIEW
        - comment: Table comment (if any)
        - row_count: Approximate row count
        - columns: List of column dicts (name, type, nullable, comment, position)
        - total_columns: Total number of columns in the table
        - columns_truncated: True if not all columns are shown (token budget hit)
        - Or {"error": "...", "table": "..."} if the table was not found

    Example:
        from data_chat import DataChatContext
        from data_chat.client import SnowflakeClient

        client = SnowflakeClient.from_env()
        with DataChatContext(client):
            results = get_tables(tables=["MY_DB.MY_SCHEMA.USERS"])
    """
    conn = get_connection()

    results = []
    for table_ref in tables:
        try:
            result = _get_single_table(conn, table_ref, database, schema)
            results.append(result)
        except Exception as e:
            # Per-item error handling: one bad table doesn't fail the batch.
            # Mirrors DataHub's pattern at entities.py:120-122.
            logger.warning(f"Error fetching table {table_ref}: {e}")
            results.append({"error": str(e), "table": table_ref})

    return results


def _parse_table_ref(
    table_ref: str,
    default_database: Optional[str],
    default_schema: Optional[str],
) -> tuple:
    """Parse a table reference into (database, schema, table).

    Handles:
    - "DB.SCHEMA.TABLE" → (DB, SCHEMA, TABLE)
    - "SCHEMA.TABLE"    → (default_database, SCHEMA, TABLE)
    - "TABLE"           → (default_database, default_schema, TABLE)
    """
    parts = [p.strip() for p in table_ref.split(".")]

    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        return default_database, parts[0], parts[1]
    elif len(parts) == 1:
        return default_database, default_schema, parts[0]
    else:
        raise ValueError(
            f"Invalid table reference: {table_ref!r}. "
            f"Expected format: DATABASE.SCHEMA.TABLE, SCHEMA.TABLE, or TABLE"
        )


def _get_single_table(
    conn,
    table_ref: str,
    default_database: Optional[str],
    default_schema: Optional[str],
) -> dict:
    """Fetch detailed info for a single table.

    Separated from get_tables() so per-item error handling is clean.
    DataHub does the same: the loop in get_entities() calls per-entity
    logic wrapped in try/except.
    """
    db, sch, tbl = _parse_table_ref(table_ref, default_database, default_schema)

    # --- Fetch table metadata ---
    table_sql = """
        SELECT
            t.TABLE_CATALOG as database,
            t.TABLE_SCHEMA as schema,
            t.TABLE_NAME as name,
            t.TABLE_TYPE as type,
            t.COMMENT as comment,
            t.ROW_COUNT as row_count,
            t.CREATED as created,
            t.LAST_ALTERED as last_altered
        FROM INFORMATION_SCHEMA.TABLES t
        WHERE t.TABLE_NAME = %(table)s
    """
    params: dict = {"table": tbl}

    if db:
        table_sql += " AND t.TABLE_CATALOG = %(database)s"
        params["database"] = db
    if sch:
        table_sql += " AND t.TABLE_SCHEMA = %(schema)s"
        params["schema"] = sch

    table_rows = execute_query(conn, sql=table_sql, params=params)

    if not table_rows:
        fqn = ".".join(filter(None, [db, sch, tbl]))
        raise ValueError(f"Table {fqn} not found")

    table_info = clean_response(table_rows[0])
    truncate_descriptions(table_info)

    # --- Fetch columns ---
    columns_sql = """
        SELECT
            c.COLUMN_NAME as name,
            c.DATA_TYPE as type,
            c.IS_NULLABLE as nullable,
            c.COLUMN_DEFAULT as default_value,
            c.COMMENT as comment,
            c.ORDINAL_POSITION as position
        FROM INFORMATION_SCHEMA.COLUMNS c
        WHERE c.TABLE_NAME = %(table)s
    """
    col_params: dict = {"table": tbl}

    if db:
        columns_sql += " AND c.TABLE_CATALOG = %(database)s"
        col_params["database"] = db
    if sch:
        columns_sql += " AND c.TABLE_SCHEMA = %(schema)s"
        col_params["schema"] = sch

    columns_sql += " ORDER BY c.ORDINAL_POSITION"

    column_rows = execute_query(conn, sql=columns_sql, params=col_params)

    # Clean each column row
    cleaned_columns = []
    for col in column_rows:
        cleaned = clean_response(col)
        truncate_descriptions(cleaned)
        cleaned_columns.append(cleaned)

    # Apply per-table token budget (the two-level budget trick).
    # This prevents a 500-column table from eating the entire response budget.
    # Mirrors: DataHub's clean_get_entities_response() schema field loop.
    budget_result = select_columns_within_budget(cleaned_columns)

    # Merge table info + column info into final response
    table_info["columns"] = budget_result["columns"]
    table_info["total_columns"] = budget_result["total_columns"]

    # Only include truncation flag when actually truncated.
    # This is the signal that tells the LLM: "there are more columns,
    # call a follow-up tool to see them."
    if budget_result.get("columns_truncated"):
        table_info["columns_truncated"] = True

    return table_info
