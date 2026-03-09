"""Context management for data-chat tools.

This is a near-exact copy of datahub-agent-context/context.py.

WHY THIS EXISTS:
    Tools need a Snowflake connection, but we don't want "client" in their
    function signature (LLMs would try to fill it). So we store the client
    in a contextvars.ContextVar — a thread-safe, async-safe global.

    Tool code:        conn = get_connection()   # reads from context
    Framework code:   with DataChatContext(client): tool()  # sets context

HOW IT WORKS:
    1. ContextVar is like a thread-local, but also works with asyncio
    2. DataChatContext.__enter__ stores the client via set_client()
    3. Tools call get_client() or get_connection() to retrieve it
    4. DataChatContext.__exit__ restores the previous value via Token

    The Token mechanism ensures nested contexts work correctly:
        with DataChatContext(client_a):
            # get_client() returns client_a
            with DataChatContext(client_b):
                # get_client() returns client_b
            # get_client() returns client_a again (restored via Token)
"""

import contextvars
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import snowflake.connector
    from data_chat.client import SnowflakeClient

# The single ContextVar that stores the current client.
# Default is None — tools will get a RuntimeError if no context is set.
_client_context: contextvars.ContextVar[Optional["SnowflakeClient"]] = (
    contextvars.ContextVar("snowflake_client", default=None)
)


def get_client() -> "SnowflakeClient":
    """Get the current SnowflakeClient from context.

    This is what tools call when they need the client object.
    Mirrors: datahub_agent_context.context.get_datahub_client()
    """
    client = _client_context.get()
    if client is None:
        raise RuntimeError(
            "No SnowflakeClient in context. "
            "Use DataChatContext or set_client() before calling tools."
        )
    return client


def get_connection() -> "snowflake.connector.SnowflakeConnection":
    """Get the current Snowflake connection from context (convenience).

    Most tools want the raw connection, not the wrapper.
    Mirrors: datahub_agent_context.context.get_graph()
    """
    return get_client().connection


def set_client(client: "SnowflakeClient") -> contextvars.Token:
    """Set the SnowflakeClient in context.

    Returns a Token — you MUST pass this to reset_client() later.
    The Token remembers the previous value so nesting works.
    """
    return _client_context.set(client)


def reset_client(token: contextvars.Token) -> None:
    """Reset the context to its previous value using the Token."""
    _client_context.reset(token)


class DataChatContext:
    """Context manager that sets the client for the duration of a with-block.

    This is the primary API for framework integrations and examples.
    Mirrors: datahub_agent_context.context.DataHubContext

    Example:
        client = SnowflakeClient.from_env()
        with DataChatContext(client):
            results = search(query="customers")  # No client param needed!
    """

    def __init__(self, client: "SnowflakeClient"):
        self.client = client
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> "SnowflakeClient":
        self._token = set_client(self.client)
        return self.client

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token is not None:
            reset_client(self._token)
            self._token = None
