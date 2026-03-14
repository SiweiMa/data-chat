"""Base utilities for tool query execution and response cleaning.

Mirrors datahub-agent-context/mcp_tools/base.py.

DataHub's base.py has two core functions:
    1. execute_graphql() — runs GraphQL, handles retries/fallback
    2. clean_gql_response() — strips __typename, None, empty values

Our equivalents:
    1. execute_query() — runs SQL via Snowflake cursor
    2. clean_response() — strips None and empty values from dicts

WHY CENTRALIZE THIS:
    Every tool needs to run a query and clean the result. Without this,
    each tool would duplicate cursor management and cleaning logic.
    DataHub has 19 tools — imagine 19 copies of try/except/cursor.close()!
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Regex for validating Snowflake identifiers (database, schema, table names).
# Allows dotted names like "DB.SCHEMA.TABLE". Each part must be alphanumeric + underscore.
# Shared by lineage.py and navigation.py to prevent SQL injection in string-literal contexts.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_]+(\.[A-Za-z0-9_]+){0,2}$")


def validate_identifier(name: str, label: str = "identifier") -> str:
    """Validate and uppercase a Snowflake identifier.

    Used for contexts where bind parameters are not supported (e.g., GET_LINEAGE
    requires string literals, INFORMATION_SCHEMA.SCHEMATA requires embedded database names).

    Args:
        name: The identifier to validate (e.g., "DB.SCHEMA.TABLE")
        label: Human-readable label for error messages (e.g., "object_name")

    Returns:
        The uppercased identifier.

    Raises:
        ValueError: If the identifier contains invalid characters.
    """
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid {label}: {name!r}. "
            f"Must contain only alphanumeric characters, underscores, and dots."
        )
    return name.upper()


def execute_query(
    conn: Any,
    *,
    sql: str,
    params: Optional[Union[Dict[str, Any], tuple]] = None,
) -> List[Dict[str, Any]]:
    """Execute a SQL query and return results as a list of dicts.

    Mirrors: datahub-agent-context execute_graphql(graph, query=..., variables=...)

    DataHub's version handles GraphQL-specific concerns (cloud detection,
    field fallback). Ours is simpler — just run SQL and return rows as dicts.

    Args:
        conn: Snowflake connection (from get_connection())
        sql: SQL query string
        params: Optional query parameters (dict for named, tuple for positional)

    Returns:
        List of dicts, one per row. Keys are column names.

    Raises:
        Exception: Any Snowflake query error (propagated to caller)
    """
    logger.debug("Executing SQL: %s (params: %s)", sql, params)

    cursor = conn.cursor()
    try:
        cursor.execute(sql, params)

        # Get column names from cursor description
        columns = [desc[0].lower() for desc in cursor.description] if cursor.description else []

        # Convert rows to list of dicts
        rows = []
        for row in cursor.fetchall():
            rows.append(dict(zip(columns, row)))

        logger.debug("Query returned %d rows", len(rows))
        return rows

    finally:
        cursor.close()


def clean_response(response: Any) -> Any:
    """Clean a response dict/list by removing None values and empty containers.

    Mirrors: datahub-agent-context clean_gql_response()

    DataHub's version also removes __typename (GraphQL metadata) and
    base64 images from descriptions. We don't have those concerns,
    but we keep the same recursive cleaning pattern.

    Why clean responses?
    - None values waste tokens ("field": null is 15+ chars for zero info)
    - Empty arrays/dicts add structural noise
    - LLMs parse cleaner JSON more reliably

    Args:
        response: Raw response (dict, list, or primitive)

    Returns:
        Cleaned response with same structure but without noise
    """
    if isinstance(response, dict):
        cleaned = {}
        for k, v in response.items():
            if v is None or v == []:
                continue
            cleaned_v = clean_response(v)
            if cleaned_v is not None and cleaned_v != {}:
                cleaned[k] = cleaned_v
        return cleaned
    elif isinstance(response, list):
        return [clean_response(item) for item in response]
    else:
        return response
