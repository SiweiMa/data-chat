"""Data Chat - AI agent tools for querying Snowflake.

Public API mirrors datahub-agent-context/__init__.py:
    - DataChatContext: context manager for tool execution
    - get_client / get_connection: retrieve client from context
    - set_client / reset_client: low-level context manipulation
    - create_context_wrapper: used by framework builders
"""

from data_chat._version import __version__
from data_chat.context import (
    DataChatContext,
    get_client,
    get_connection,
    reset_client,
    set_client,
)
from data_chat.utils import create_context_wrapper

__all__ = [
    "__version__",
    "DataChatContext",
    "create_context_wrapper",
    "get_client",
    "get_connection",
    "set_client",
    "reset_client",
]
