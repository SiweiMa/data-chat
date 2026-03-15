"""Conversation memory management — sliding window over message history.

Prevents the message history from exceeding the LLM's context window
by trimming oldest messages while preserving tool_use/tool_result pairs
atomically (never split a pair).

Uses TokenCountEstimator from tools/_token_estimator.py for fast,
dependency-free token estimation.
"""

import json
import logging
from typing import Dict, List

from data_chat.tools._token_estimator import TokenCountEstimator

logger = logging.getLogger(__name__)

# Reserve ~100K tokens for message history out of a 200K context window.
# The remaining ~100K is split between system prompt, tool definitions,
# and the current response.
DEFAULT_MESSAGE_TOKEN_BUDGET = 100_000


def _estimate_message_tokens(message: dict) -> int:
    """Estimate token count for a single message."""
    content = message.get("content", "")
    if isinstance(content, str):
        return TokenCountEstimator.estimate_dict_tokens({"role": message.get("role", ""), "content": content})
    elif isinstance(content, list):
        return TokenCountEstimator.estimate_dict_tokens({"role": message.get("role", ""), "content": content})
    else:
        return TokenCountEstimator.estimate_dict_tokens(message)


def _group_into_exchanges(messages: List[dict]) -> List[List[dict]]:
    """Group messages into atomic exchanges that should not be split.

    An exchange is:
    - A user message (simple text)
    - An assistant message with tool_use blocks + the following user message
      containing tool_results (these two form an atomic pair)
    - An assistant message with final text

    This ensures we never leave orphaned tool_use or tool_result messages.
    """
    exchanges: List[List[dict]] = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        if msg["role"] == "assistant":
            # Check if this assistant message contains tool_use blocks
            content = msg.get("content", [])
            has_tool_use = False
            if isinstance(content, list):
                has_tool_use = any(
                    (isinstance(block, dict) and block.get("type") == "tool_use")
                    for block in content
                )

            if has_tool_use and i + 1 < len(messages):
                # Group: assistant (tool_use) + user (tool_result)
                exchanges.append([messages[i], messages[i + 1]])
                i += 2
            else:
                # Standalone assistant message (final text)
                exchanges.append([messages[i]])
                i += 1
        else:
            # User message (text input)
            exchanges.append([messages[i]])
            i += 1

    return exchanges


def trim_messages(
    messages: List[dict],
    token_budget: int = DEFAULT_MESSAGE_TOKEN_BUDGET,
) -> List[dict]:
    """Trim oldest messages to stay within token budget.

    Groups tool_use/tool_result pairs atomically — never splits them.
    Always keeps at least the most recent exchange (even if it exceeds budget).

    Args:
        messages: List of message dicts (Anthropic API format).
            Modified in place and also returned.
        token_budget: Maximum tokens to allow for message history.

    Returns:
        The trimmed messages list (same object, modified in place).
    """
    if not messages:
        return messages

    # Group into atomic exchanges
    exchanges = _group_into_exchanges(messages)

    if not exchanges:
        return messages

    # Calculate total tokens per exchange
    exchange_tokens: List[int] = []
    for exchange in exchanges:
        tokens = sum(_estimate_message_tokens(msg) for msg in exchange)
        exchange_tokens.append(tokens)

    total_tokens = sum(exchange_tokens)

    if total_tokens <= token_budget:
        return messages

    # Trim from the front (oldest) until we're within budget.
    # Always keep at least the last exchange.
    trimmed_tokens = total_tokens
    trim_count = 0

    for i in range(len(exchanges) - 1):  # Never trim the last exchange
        if trimmed_tokens <= token_budget:
            break
        trimmed_tokens -= exchange_tokens[i]
        trim_count += 1

    if trim_count > 0:
        logger.info(
            "Trimmed %d exchanges (%d tokens) from message history. "
            "Remaining: %d exchanges, ~%d tokens (budget: %d)",
            trim_count,
            total_tokens - trimmed_tokens,
            len(exchanges) - trim_count,
            trimmed_tokens,
            token_budget,
        )

        # Rebuild messages from remaining exchanges
        remaining = exchanges[trim_count:]
        messages.clear()
        for exchange in remaining:
            messages.extend(exchange)

    return messages
