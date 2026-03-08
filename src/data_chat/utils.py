"""Utility functions for data-chat.

This is a near-exact copy of datahub-agent-context/utils.py.

WHY THIS EXISTS:
    Framework builders (LangChain, Google ADK) need to wrap each tool so
    that the client is automatically set in context before the tool runs.

    Without this, you'd have to write:
        with DataChatContext(client):
            result = search(query="customers")

    With this, the builder does:
        wrapped_search = create_context_wrapper(search, client)
        # Now wrapped_search() sets context automatically

HOW IT WORKS:
    create_context_wrapper(func, client) returns a NEW function that:
    1. Calls set_client(client) → gets a Token
    2. Calls the original func(*args, **kwargs)
    3. Calls reset_client(token) in a finally block (always runs, even on error)

    @functools.wraps preserves the original function's name, docstring, and
    type hints — this is critical because LangChain reads the docstring to
    generate the tool description shown to the LLM.
"""

import contextvars
import functools
from typing import TYPE_CHECKING, Callable

from data_chat.context import reset_client, set_client

if TYPE_CHECKING:
    from data_chat.client import SnowflakeClient


def create_context_wrapper(func: Callable, client: "SnowflakeClient") -> Callable:
    """Create a wrapper that sets client context before calling the function.

    Mirrors: datahub_agent_context.utils.create_context_wrapper

    Args:
        func: The tool function (e.g., search, get_tables)
        client: SnowflakeClient instance to inject

    Returns:
        Wrapped function with same signature, name, and docstring
    """

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        token: contextvars.Token = set_client(client)
        try:
            return func(*args, **kwargs)
        finally:
            reset_client(token)

    return wrapper
