"""Tools for querying Snowflake metadata and data.

Mirrors: datahub-agent-context/mcp_tools/__init__.py

Each tool is a plain function that calls get_connection() from context.
Framework builders (LangChain, Google ADK) wrap these with
create_context_wrapper() to inject the client automatically.
"""

from data_chat.tools.query import run_query
from data_chat.tools.search import search
from data_chat.tools.tables import get_tables

__all__ = [
    "search",
    "get_tables",
    "run_query",
]
