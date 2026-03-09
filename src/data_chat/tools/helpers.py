"""Helper utilities for tool response management.

Mirrors datahub-agent-context/mcp_tools/helpers.py (simplified).

DataHub's helpers.py has ~467 lines covering:
    - Token budget enforcement (_select_results_within_budget)
    - HTML sanitization (sanitize_html_content)
    - Description truncation (truncate_descriptions)
    - URL injection for Cloud instances (inject_urls_for_urns)
    - Schema field processing (clean_get_entities_response)
    - Lineage column extraction (_extract_lineage_columns_from_paths)

We keep the core patterns (token budget + truncation) and skip
DataHub-specific concerns (URL injection, schema field merging, lineage).
"""

import logging
import os
from typing import Callable, Generator, Iterator, Optional, TypeVar

from data_chat.tools._token_estimator import TokenCountEstimator

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Maximum token count for tool responses.
# DataHub defaults to 80K. Configurable via env var so you can tune for
# different LLMs (GPT-4 has 128K context, Claude has 200K, etc.).
TOOL_RESPONSE_TOKEN_LIMIT = int(os.getenv("TOOL_RESPONSE_TOKEN_LIMIT", "80000"))

# Max length for description/comment fields before truncation.
DESCRIPTION_LENGTH_HARD_LIMIT = 1000


def truncate_with_ellipsis(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text and add suffix if it exceeds max_length.

    Mirrors: datahub helpers.py truncate_with_ellipsis()
    """
    if not text or len(text) <= max_length:
        return text
    actual_max = max_length - len(suffix)
    return text[:actual_max] + suffix


def truncate_descriptions(
    data: dict | list, max_length: int = DESCRIPTION_LENGTH_HARD_LIMIT
) -> None:
    """Recursively truncate 'description' and 'comment' values in a dict (in place).

    Mirrors: datahub helpers.py truncate_descriptions()

    WHY: Table/column comments from Snowflake can be arbitrarily long.
    Truncating to 1000 chars keeps tool responses within budget.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ("description", "comment") and isinstance(value, str):
                data[key] = truncate_with_ellipsis(value, max_length)
            elif isinstance(value, (dict, list)):
                truncate_descriptions(value)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                truncate_descriptions(item)


def select_results_within_budget(
    results: Iterator[T],
    fetch_entity: Callable[[T], dict],
    max_results: int = 10,
    token_budget: Optional[int] = None,
) -> Generator[T, None, None]:
    """Yield results until token budget is exhausted.

    Mirrors: datahub helpers.py _select_results_within_budget()

    This is a KEY pattern for agent tools. Without it, a query returning
    1000 rows would generate a response so large it breaks the LLM.

    HOW IT WORKS:
        1. For each result, estimate its token cost
        2. If adding it would exceed the budget AND we have >=1 result, stop
        3. Always yield at least 1 result (even if it exceeds budget alone)

    WHY A GENERATOR:
        Memory efficient — doesn't load all results into memory.
        The caller can iterate and stop early.

    Args:
        results: Iterator of result objects
        fetch_entity: Function to extract the dict for token counting.
                      Can mutate the result (e.g., to clean it in place).
        max_results: Hard cap on number of results
        token_budget: Token budget (defaults to 90% of TOOL_RESPONSE_TOKEN_LIMIT)

    Yields:
        Results within budget
    """
    if token_budget is None:
        # 90% buffer — accounts for estimation inaccuracy + response wrapper overhead
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
