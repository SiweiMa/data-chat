"""Tool for querying data lineage from Snowflake.

Uses Snowflake's built-in SNOWFLAKE.CORE.GET_LINEAGE table function
to trace upstream and downstream dependencies between tables, views,
and columns.

NOTE: GET_LINEAGE requires Snowflake Enterprise Edition or higher.
The tool returns a clear error message if the function is unavailable.

IMPORTANT — GET_LINEAGE requires string literals, not bind parameters:
    TABLE(SNOWFLAKE.CORE.GET_LINEAGE('DB.SCHEMA.TABLE', 'TABLE', 'DOWNSTREAM', 3))
We validate object_name with a strict regex before embedding it in SQL
to prevent injection. The shared validate_identifier() in base.py
handles this.
"""

import logging

from data_chat.context import get_connection
from data_chat.tools.base import clean_response, execute_query, validate_identifier
from data_chat.tools.helpers import select_results_within_budget, truncate_descriptions

logger = logging.getLogger(__name__)

MAX_RESULTS = 100
MAX_DISTANCE = 5

_VALID_DOMAINS = {"TABLE", "VIEW", "COLUMN"}
_VALID_DIRECTIONS = {"UPSTREAM", "DOWNSTREAM"}


def get_lineage(
    object_name: str,
    object_domain: str = "TABLE",
    direction: str = "DOWNSTREAM",
    distance: int = 3,
) -> dict:
    """Trace data lineage for a Snowflake table, view, or column.

    Returns the upstream or downstream dependencies of the specified object,
    showing how data flows between tables and views in your warehouse.

    WHAT IS LINEAGE:
    - DOWNSTREAM: "What depends on this table?" — find all tables/views that
      read from the given object. Useful for impact analysis before changing
      a table.
    - UPSTREAM: "Where does this table's data come from?" — trace the data
      sources feeding into the given object. Useful for understanding data
      provenance and debugging data quality issues.

    TYPICAL WORKFLOW:
    1. search("revenue") → find tables related to revenue
    2. get_tables(["DB.SCHEMA.REVENUE_SUMMARY"]) → examine columns
    3. get_lineage("DB.SCHEMA.REVENUE_SUMMARY", direction="UPSTREAM")
       → see what source tables feed into the revenue summary
    4. get_lineage("DB.SCHEMA.RAW_TRANSACTIONS", direction="DOWNSTREAM")
       → see what depends on raw transactions

    COLUMN-LEVEL LINEAGE:
    - Set object_domain="COLUMN" and pass a fully qualified column name:
      get_lineage("DB.SCHEMA.TABLE.COLUMN_NAME", object_domain="COLUMN")
    - Returns which columns in downstream tables are derived from this column

    DISTANCE:
    - distance=1: direct dependencies only (one hop)
    - distance=3: up to 3 hops (default, good for most use cases)
    - distance=5: maximum — traces the full dependency chain

    ENTERPRISE EDITION REQUIRED:
    This tool uses SNOWFLAKE.CORE.GET_LINEAGE, which is only available on
    Snowflake Enterprise Edition or higher. If the function is not available,
    the tool returns {"error": "Requires Enterprise Edition"} instead of
    raising an exception, so the agent can inform the user gracefully.

    OBJECT NAME FORMAT:
    - Fully qualified: "DATABASE.SCHEMA.TABLE" (recommended)
    - The name is automatically uppercased to match Snowflake conventions

    Args:
        object_name: Fully qualified object name (e.g., "MY_DB.MY_SCHEMA.MY_TABLE").
            Must contain only alphanumeric characters, underscores, and dots.
        object_domain: Type of object — "TABLE", "VIEW", or "COLUMN"
            (default: "TABLE")
        direction: Direction to trace — "DOWNSTREAM" (what depends on this)
            or "UPSTREAM" (what feeds into this) (default: "DOWNSTREAM")
        distance: Maximum number of hops to trace (1-5, default: 3)

    Returns:
        Dictionary with:
        - object_name: The validated, uppercased object name
        - direction: The direction traced
        - distance: The distance used
        - lineage: List of lineage edges, each with source/target info and distance
        - total: Total number of lineage edges found
        - returned: Number of edges in this response
        - has_more: Whether more edges exist beyond what was returned

    Example:
        from data_chat import DataChatContext
        from data_chat.client import SnowflakeClient

        client = SnowflakeClient.from_env()
        with DataChatContext(client):
            result = get_lineage("MY_DB.MY_SCHEMA.REVENUE", direction="UPSTREAM")
    """
    conn = get_connection()

    # Validate inputs
    validated_name = validate_identifier(object_name, "object_name")

    object_domain = object_domain.upper()
    if object_domain not in _VALID_DOMAINS:
        raise ValueError(
            f"Invalid object_domain: {object_domain!r}. "
            f"Must be one of: {', '.join(sorted(_VALID_DOMAINS))}"
        )

    direction = direction.upper()
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"Invalid direction: {direction!r}. "
            f"Must be one of: {', '.join(sorted(_VALID_DIRECTIONS))}"
        )

    distance = min(distance, MAX_DISTANCE)

    # Build SQL — GET_LINEAGE requires string literals, not bind params
    sql = f"""
        SELECT
            SOURCE_OBJECT_DATABASE, SOURCE_OBJECT_SCHEMA, SOURCE_OBJECT_NAME,
            SOURCE_COLUMN_NAME,
            TARGET_OBJECT_DATABASE, TARGET_OBJECT_SCHEMA, TARGET_OBJECT_NAME,
            TARGET_COLUMN_NAME,
            DISTANCE
        FROM TABLE(SNOWFLAKE.CORE.GET_LINEAGE(
            '{validated_name}', '{object_domain}', '{direction}', {distance}
        ))
        ORDER BY DISTANCE, TARGET_OBJECT_DATABASE, TARGET_OBJECT_SCHEMA, TARGET_OBJECT_NAME
    """

    try:
        rows = execute_query(conn, sql=sql)
    except Exception as e:
        # GET_LINEAGE is Enterprise Edition only
        if "GET_LINEAGE" in str(e):
            return {
                "object_name": validated_name,
                "error": (
                    "Lineage is not available. SNOWFLAKE.CORE.GET_LINEAGE requires "
                    "Snowflake Enterprise Edition or higher."
                ),
            }
        raise

    # Transform rows into lineage edges
    all_edges = []
    for row in rows:
        edge = {
            "source_database": row.get("source_object_database"),
            "source_schema": row.get("source_object_schema"),
            "source_name": row.get("source_object_name"),
            "source_column": row.get("source_column_name"),
            "target_database": row.get("target_object_database"),
            "target_schema": row.get("target_object_schema"),
            "target_name": row.get("target_object_name"),
            "target_column": row.get("target_column_name"),
            "distance": row.get("distance"),
        }
        all_edges.append(clean_response(edge))

    truncate_descriptions(all_edges)

    total = len(all_edges)

    # Apply token budget
    selected = list(
        select_results_within_budget(
            results=iter(all_edges),
            fetch_entity=lambda r: r,
            max_results=MAX_RESULTS,
        )
    )

    return {
        "object_name": validated_name,
        "direction": direction,
        "distance": distance,
        "lineage": selected,
        "total": total,
        "returned": len(selected),
        "has_more": len(selected) < total,
    }
