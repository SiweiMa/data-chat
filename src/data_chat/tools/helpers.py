"""Helper utilities for tool response management.

Mirrors datahub-agent-context/mcp_tools/helpers.py.

Essential LLM agent tricks from DataHub's helpers.py:

1. TWO-LEVEL TOKEN BUDGET (lines 161-234, 237-345 in DataHub)
   - TOOL_RESPONSE_TOKEN_LIMIT: caps the total response to the LLM
   - PER_TABLE_TOKEN_BUDGET: caps each table's columns independently
   Without per-table budget, one 500-column table eats the entire response.

2. TRUNCATION METADATA FLAGS (line 329 in DataHub)
   Tools signal "I couldn't show you everything" via flags like
   {"columns_truncated": true}. The LLM sees this and knows to call
   a follow-up tool for more detail.

3. SANITIZE BEFORE TRUNCATE (lines 72-88 in DataHub)
   Strip HTML/markdown noise FIRST, then truncate. Otherwise you waste
   the 1000-char budget on <div class="..."> tags instead of content.
   Uses bounded regex to prevent ReDoS attacks.

4. EXPLICIT TRUNCATION SUFFIX (line 106-110 in DataHub)
   Use "... [truncated]" not "..." for SQL and large values. The word
   "truncated" tells the LLM "don't analyze the last few lines, they're
   incomplete."

5. FLATTEN FOR LLM COMPREHENSION (lines 366-466 in DataHub)
   Reshape deeply nested structures into flat lists. LLMs parse flat
   JSON much more reliably than 4-level-deep nesting.
"""

import html
import logging
import os
import re
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, TypeVar

from data_chat.tools._token_estimator import TokenCountEstimator

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Budget constants
# ---------------------------------------------------------------------------

# Total response budget. 80K tokens ≈ safe for most LLMs.
# Configurable: TOOL_RESPONSE_TOKEN_LIMIT=120000 for Claude's 200K context.
TOOL_RESPONSE_TOKEN_LIMIT = int(os.getenv("TOOL_RESPONSE_TOKEN_LIMIT", "80000"))

# Per-table budget for column listing.
# DataHub calls this ENTITY_SCHEMA_TOKEN_BUDGET (16K default).
# WHY: Without this, one wide table (500 columns) eats the entire 80K budget,
# leaving nothing for other tables in a multi-table response.
PER_TABLE_TOKEN_BUDGET = int(os.getenv("PER_TABLE_TOKEN_BUDGET", "16000"))

# Max length for description/comment fields.
DESCRIPTION_LENGTH_HARD_LIMIT = 1000

# Max length for SQL in results (view definitions, query text).
# DataHub uses 5000. The suffix "... [truncated]" tells the LLM explicitly
# that the content is incomplete (vs "..." which could look intentional).
QUERY_LENGTH_HARD_LIMIT = 5000

# Max length for individual cell values in query results.
# JSON columns, large text fields, etc. can be enormous.
CELL_VALUE_LENGTH_LIMIT = 2000


# ---------------------------------------------------------------------------
# Sanitization — clean data before showing it to the LLM
# ---------------------------------------------------------------------------


def sanitize_html_content(text: str) -> str:
    """Strip HTML tags and decode HTML entities from text.

    Mirrors: datahub helpers.py sanitize_html_content() (lines 30-47)

    Uses a BOUNDED regex to prevent ReDoS (Regular Expression Denial of
    Service). The {0,100} limit means: if there are more than 100 chars
    between < and >, don't try to match — just skip it. This prevents
    catastrophic backtracking on malicious input like "<" + "a"*1000000.

    WHY THIS MATTERS FOR AGENTS:
        Tool inputs come from the database. Snowflake table/column comments
        could contain HTML from automated documentation tools. Sending raw
        HTML to the LLM wastes tokens on tags and confuses the model.
    """
    if not text:
        return text

    # Bounded regex: max 100 chars between < and > to prevent ReDoS
    text = re.sub(r"<[^<>]{0,100}>", "", text)

    # Decode HTML entities: &amp; → &, &lt; → <, etc.
    text = html.unescape(text)

    return text.strip()


# ---------------------------------------------------------------------------
# Truncation — cap field lengths to save tokens
# ---------------------------------------------------------------------------


def truncate_with_ellipsis(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text and add suffix if it exceeds max_length.

    Mirrors: datahub helpers.py truncate_with_ellipsis()
    """
    if not text or len(text) <= max_length:
        return text
    actual_max = max_length - len(suffix)
    return text[:actual_max] + suffix


def sanitize_and_truncate(text: str, max_length: int) -> str:
    """Sanitize HTML/noise FIRST, then truncate to max_length.

    Mirrors: datahub helpers.py sanitize_and_truncate_description() (lines 72-88)

    ORDER MATTERS:
        sanitize first → truncate second.
        If you truncate first, you might cut in the middle of
        '<div class="very-long-class">' — leaving broken HTML that
        wastes the budget on noise. Sanitizing first strips the tags,
        so the 1000-char budget goes to actual content.
    """
    if not text:
        return text
    try:
        sanitized = sanitize_html_content(text)
        return truncate_with_ellipsis(sanitized, max_length)
    except Exception as e:
        logger.warning(f"Error sanitizing text: {e}")
        return text[:max_length] if len(text) > max_length else text


def truncate_descriptions(
    data: dict | list, max_length: int = DESCRIPTION_LENGTH_HARD_LIMIT
) -> None:
    """Recursively sanitize+truncate 'description' and 'comment' values (in place).

    Mirrors: datahub helpers.py truncate_descriptions() (lines 91-103)

    WHY IN-PLACE: Avoids copying large structures. The response dict is
    modified directly, which is fine because we're in a tool function that
    owns the data.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ("description", "comment") and isinstance(value, str):
                data[key] = sanitize_and_truncate(value, max_length)
            elif isinstance(value, (dict, list)):
                truncate_descriptions(value)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                truncate_descriptions(item)


def truncate_query(query: str) -> str:
    """Truncate SQL text with an explicit [truncated] suffix.

    Mirrors: datahub helpers.py truncate_query() (lines 106-110)

    WHY "[truncated]" NOT "...":
        "..." could look like the SQL intentionally ends with ellipsis.
        "... [truncated]" tells the LLM explicitly: "this is incomplete,
        don't try to analyze the last few lines."
    """
    return truncate_with_ellipsis(
        query, QUERY_LENGTH_HARD_LIMIT, suffix="... [truncated]"
    )


def truncate_cell_values(row: Dict[str, Any]) -> Dict[str, Any]:
    """Truncate oversized cell values in a query result row.

    NOT IN DATAHUB — new for data-chat.

    WHY: Snowflake columns can hold huge values: JSON blobs, large TEXT
    fields, VARIANT columns with nested data. A single cell could be
    50K+ characters. Without truncation, one row eats the entire budget.

    Applies to string values only (numbers, bools, None pass through).
    """
    truncated = {}
    for key, value in row.items():
        if isinstance(value, str) and len(value) > CELL_VALUE_LENGTH_LIMIT:
            truncated[key] = truncate_with_ellipsis(
                value, CELL_VALUE_LENGTH_LIMIT, suffix="... [truncated]"
            )
        else:
            truncated[key] = value
    return truncated


# ---------------------------------------------------------------------------
# Token budget enforcement
# ---------------------------------------------------------------------------


def select_results_within_budget(
    results: Iterator[T],
    fetch_entity: Callable[[T], dict],
    max_results: int = 10,
    token_budget: Optional[int] = None,
) -> Generator[T, None, None]:
    """Yield results until token budget is exhausted.

    Mirrors: datahub helpers.py _select_results_within_budget() (lines 161-234)

    HOW IT WORKS:
        1. For each result, call fetch_entity() to get the dict for counting
           (fetch_entity can also clean/mutate the result — see Trick 3)
        2. Estimate its token cost
        3. If adding it would exceed the budget AND we have >=1 result, stop
        4. Always yield at least 1 result (even if it exceeds budget alone)

    WHY A GENERATOR:
        Memory efficient. The caller iterates lazily — if we stop early,
        we never even look at the remaining results.

    Args:
        results: Iterator of result objects
        fetch_entity: Extracts dict for token counting. CAN MUTATE the result
                      to clean it in place (DataHub's "clean-while-counting"
                      trick — count tokens on the cleaned version, not raw).
        max_results: Hard cap on number of results
        token_budget: Defaults to 90% of TOOL_RESPONSE_TOKEN_LIMIT

    Yields:
        Results within budget
    """
    if token_budget is None:
        # 90% buffer — absorbs estimation inaccuracy + response wrapper overhead.
        # DataHub uses the same 90% factor.
        token_budget = int(TOOL_RESPONSE_TOKEN_LIMIT * 0.9)

    total_tokens = 0
    results_count = 0

    for i, result in enumerate(results):
        if i >= max_results:
            break

        entity = fetch_entity(result)
        entity_tokens = TokenCountEstimator.estimate_dict_tokens(entity)

        if total_tokens + entity_tokens > token_budget:
            if results_count == 0:
                # Always yield at least 1 result
                logger.warning(
                    f"First result ({entity_tokens:,} tokens) exceeds budget "
                    f"({token_budget:,}), yielding it anyway"
                )
                yield result
                results_count += 1
                total_tokens += entity_tokens
            else:
                logger.info(
                    f"Stopping at {results_count} results "
                    f"(next would exceed {token_budget:,} token budget)"
                )
                break
        else:
            yield result
            results_count += 1
            total_tokens += entity_tokens

    logger.info(
        f"Selected {results_count} results using {total_tokens:,} tokens "
        f"(budget: {token_budget:,})"
    )


def select_columns_within_budget(
    columns: List[dict],
    offset: int = 0,
    limit: Optional[int] = None,
    token_budget: Optional[int] = None,
) -> dict:
    """Select columns within per-table token budget, with pagination.

    Mirrors: datahub helpers.py clean_get_entities_response() (lines 237-345)
    Specifically the schema field processing loop at lines 289-334.

    THIS IS THE TWO-LEVEL BUDGET TRICK:
        select_results_within_budget() caps TOTAL response tokens.
        This function caps PER-TABLE column tokens.

        Without this, one 500-column table eats the entire 80K budget:
            get_tables(["wide_table", "narrow_table"])
            → wide_table uses 60K tokens for 500 columns
            → narrow_table gets 0 tokens (budget exhausted)

        With per-table budget (16K default):
            → wide_table shows 100 columns (16K budget) + truncated flag
            → narrow_table shows all 10 columns (well within 16K)

    Args:
        columns: List of column dicts from INFORMATION_SCHEMA
        offset: Skip first N columns (for pagination)
        limit: Max columns to return (None = unlimited, but still budget-capped)
        token_budget: Per-table budget (defaults to PER_TABLE_TOKEN_BUDGET)

    Returns:
        Dict with:
        - columns: selected column list
        - total_columns: total available
        - returned: how many we included
        - columns_truncated: True if not all columns are shown
        - offset: the offset used

        THE TRUNCATION FLAG IS CRITICAL:
        When the LLM sees {"columns_truncated": true}, it knows to call
        a follow-up tool for specific columns instead of assuming the
        truncated list is complete. DataHub uses "schemaFieldsTruncated"
        for the same purpose (line 330).
    """
    if token_budget is None:
        token_budget = PER_TABLE_TOKEN_BUDGET

    total_columns = len(columns)
    selected: List[dict] = []
    total_tokens = 0
    truncated = False

    for i, col in enumerate(columns):
        # Skip before offset
        if i < offset:
            continue

        # Check limit
        if limit is not None and len(selected) >= limit:
            truncated = True
            break

        # Estimate tokens for this column
        col_tokens = TokenCountEstimator.estimate_dict_tokens(col)

        # Check per-table budget
        if total_tokens + col_tokens > token_budget:
            if len(selected) == 0:
                # Always include at least one column
                logger.warning(
                    f"First column ({col_tokens:,} tokens) exceeds per-table "
                    f"budget ({token_budget:,}), including it anyway"
                )
                selected.append(col)
                total_tokens += col_tokens
            truncated = True
            break

        selected.append(col)
        total_tokens += col_tokens

    if truncated:
        logger.info(
            f"Truncated columns: showing {len(selected)} of {total_columns} "
            f"({total_tokens:,} tokens, budget: {token_budget:,})"
        )

    result: dict = {
        "columns": selected,
        "total_columns": total_columns,
        "returned": len(selected),
        "offset": offset,
    }

    # Only include truncation flag when actually truncated.
    # This is the COMMUNICATION CHANNEL between tool and LLM.
    if truncated:
        result["columns_truncated"] = True

    return result
