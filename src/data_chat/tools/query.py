"""Tool for executing read-only SQL queries against Snowflake.

NO DIRECT DATAHUB EQUIVALENT — DataHub never executes SQL against warehouses.
However, this tool uses all the same infrastructure patterns:
    - get_connection() from contextvars (Step 2)
    - select_results_within_budget() for row-level token budget (Step 3)
    - truncate_cell_values() for oversized cell values (Step 3)
    - clean_response() on every row (Step 3)
    - Rich docstring as LLM instruction manual (Step 4 pattern)

SAFETY:
    This is the most dangerous tool — it takes raw SQL from the LLM.
    Three safety layers:
    1. Statement validation: reject DDL/DML (DROP, DELETE, INSERT, UPDATE, etc.)
    2. LIMIT enforcement: inject or cap LIMIT to prevent full table scans
    3. Token budget: stop returning rows when budget is exhausted
"""

import logging
import re
from typing import Optional

from data_chat.context import get_connection
from data_chat.tools.base import clean_response
from data_chat.tools.helpers import (
    select_results_within_budget,
    truncate_cell_values,
)

logger = logging.getLogger(__name__)

# Hard cap on rows — even if the LLM asks for more
MAX_LIMIT = 1000

# Statements that modify data or schema. Case-insensitive check.
# We check the FIRST keyword after stripping comments/whitespace.
_FORBIDDEN_PREFIXES = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "ALTER",
    "TRUNCATE",
    "REPLACE",
    "MERGE",
    "GRANT",
    "REVOKE",
    "COPY",
    "PUT",
    "REMOVE",
    "CALL",
    "EXEC",
)


def _validate_read_only(sql: str) -> None:
    """Reject SQL statements that could modify data or schema.

    Checks the first meaningful keyword in the statement.
    Not foolproof (a CTE could hide a DML), but catches the common
    cases where the LLM generates a mutation by mistake.

    Raises:
        ValueError: If the statement appears to be a mutation
    """
    # Strip leading comments (-- and /* */) and whitespace
    cleaned = re.sub(r"--[^\n]*", "", sql)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()

    if not cleaned:
        raise ValueError("Empty SQL statement")

    # Get the first word
    first_word = cleaned.split()[0].upper()

    if first_word in _FORBIDDEN_PREFIXES:
        raise ValueError(
            f"Only read-only queries are allowed. "
            f"Statement starts with '{first_word}', which could modify data. "
            f"Use SELECT, SHOW, DESCRIBE, or WITH (CTE) statements only."
        )


def _enforce_limit(sql: str, limit: int) -> str:
    """Ensure the SQL has a LIMIT clause, capped at MAX_LIMIT.

    Three cases:
    1. SQL already has LIMIT N → cap it at min(N, limit, MAX_LIMIT)
    2. SQL has no LIMIT → append LIMIT
    3. SQL ends with semicolon → insert LIMIT before it

    This prevents the LLM from running SELECT * on a billion-row table.
    """
    limit = min(limit, MAX_LIMIT)

    # Check if LIMIT already exists (case-insensitive, word boundary)
    limit_match = re.search(
        r"\bLIMIT\s+(\d+)\b", sql, flags=re.IGNORECASE
    )

    if limit_match:
        existing_limit = int(limit_match.group(1))
        capped = min(existing_limit, limit)
        # Replace the existing LIMIT value with the capped one
        sql = sql[: limit_match.start(1)] + str(capped) + sql[limit_match.end(1) :]
        return sql

    # No LIMIT found — append one
    sql = sql.rstrip().rstrip(";")
    return f"{sql}\nLIMIT {limit}"


def run_query(
    sql: str,
    limit: int = 100,
) -> dict:
    """Execute a read-only SQL query against Snowflake and return results.

    SAFETY: Only SELECT, SHOW, DESCRIBE, and WITH (CTE) statements are allowed.
    Mutations (INSERT, UPDATE, DELETE, DROP, etc.) are rejected.

    A LIMIT clause is automatically enforced to prevent full table scans.
    If the query already has a LIMIT, it is capped at the specified limit.

    QUERY GUIDELINES:
    - Always include specific column names instead of SELECT *
    - Use LIMIT to control result size (default: 100, max: 1000)
    - Use WHERE clauses to filter data before it reaches the token budget
    - For large results, prefer aggregations (COUNT, SUM, AVG, GROUP BY)

    TOKEN BUDGET:
    Results are subject to the tool response token budget. If a query returns
    many rows, later rows may be omitted. The response includes:
    - rows_returned: How many rows are in the response
    - rows_truncated: True if token budget was hit before all rows were returned
    - total_row_count: Total rows the query would return (when available)

    TYPICAL WORKFLOW:
    1. search("customer") → find tables
    2. get_tables(["DB.SCHEMA.CUSTOMERS"]) → see columns and types
    3. run_query("SELECT customer_id, email FROM DB.SCHEMA.CUSTOMERS LIMIT 10") → see data
    4. run_query("SELECT status, COUNT(*) FROM DB.SCHEMA.CUSTOMERS GROUP BY status") → aggregate

    Args:
        sql: SQL query (SELECT, SHOW, DESCRIBE, or WITH only)
        limit: Maximum number of rows to return (default: 100, max: 1000)

    Returns:
        Dictionary with:
        - rows: List of result dicts (one per row)
        - columns: List of column names in the result
        - rows_returned: Number of rows in the response
        - rows_truncated: True if not all rows fit in token budget
        - sql_executed: The actual SQL that was run (with LIMIT applied)

    Raises:
        ValueError: If the SQL is a mutation statement or is empty

    Example:
        from data_chat import DataChatContext
        from data_chat.client import SnowflakeClient

        client = SnowflakeClient.from_env()
        with DataChatContext(client):
            result = run_query("SELECT * FROM MY_DB.MY_SCHEMA.USERS LIMIT 10")
    """
    conn = get_connection()

    # Safety layer 1: reject mutations
    _validate_read_only(sql)

    # Safety layer 2: enforce LIMIT
    sql = _enforce_limit(sql, limit)

    # Execute
    logger.debug("Executing query: %s", sql)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)

        # Get column names
        columns = (
            [desc[0].lower() for desc in cursor.description]
            if cursor.description
            else []
        )

        # Fetch all rows (LIMIT already caps this)
        raw_rows = cursor.fetchall()
    finally:
        cursor.close()

    # Convert to list of dicts and apply per-row cleaning
    all_rows = []
    for row in raw_rows:
        row_dict = dict(zip(columns, row))
        row_dict = truncate_cell_values(row_dict)
        row_dict = clean_response(row_dict)
        all_rows.append(row_dict)

    # Safety layer 3: token budget enforcement
    # Use select_results_within_budget to stop when budget is hit.
    # The fetch_entity lambda just returns the row itself for token counting.
    selected = list(
        select_results_within_budget(
            results=iter(all_rows),
            fetch_entity=lambda r: r,
            max_results=len(all_rows),
        )
    )

    truncated = len(selected) < len(all_rows)

    result: dict = {
        "columns": columns,
        "rows": selected,
        "rows_returned": len(selected),
        "sql_executed": sql,
    }

    if truncated:
        result["rows_truncated"] = True
        result["total_row_count"] = len(all_rows)

    return result
