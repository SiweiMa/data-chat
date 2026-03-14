"""Builder for Google ADK tools from data-chat tools.

Mirrors: datahub-agent-context/google_adk_tools/builder.py

DIFFERENCE FROM LANGCHAIN BUILDER:
    LangChain needs: tool(create_context_wrapper(func, client))
        → @tool decorator converts function into BaseTool object

    Google ADK needs: create_context_wrapper(func, client)
        → ADK takes plain callables directly, no decorator needed

    That's the ONLY difference. The wrapping pattern is identical.
"""

from typing import TYPE_CHECKING, Callable, List

from data_chat.utils import create_context_wrapper

if TYPE_CHECKING:
    from data_chat.client import SnowflakeClient

from data_chat.tools.lineage import get_lineage
from data_chat.tools.query import run_query
from data_chat.tools.search import search
from data_chat.tools.tables import get_tables


def build_google_adk_tools(
    client: "SnowflakeClient",
    include_query_execution: bool = True,
) -> List[Callable]:
    """Build Google ADK tools with automatic context management.

    Returns plain Python functions wrapped to automatically set the
    SnowflakeClient in context before execution. These can be passed
    directly to a Google ADK Agent's ``tools`` parameter.

    Args:
        client: SnowflakeClient instance
        include_query_execution: Whether to include run_query() tool
            (default: True). Set to False for metadata-only agents.

    Returns:
        List of callables ready for use as Google ADK tools

    Example:
        from google.adk.agents import Agent
        from data_chat.client import SnowflakeClient
        from data_chat.google_adk_tools import build_google_adk_tools

        client = SnowflakeClient.from_env()
        tools = build_google_adk_tools(client)

        agent = Agent(
            model="gemini-2.0-flash",
            name="data_chat_agent",
            instruction="You are a data assistant with access to Snowflake.",
            tools=tools,
        )
    """
    # No @tool decorator needed — ADK takes plain callables.
    # Just wrap with create_context_wrapper to inject client.
    tools: List[Callable] = [
        create_context_wrapper(search, client),
        create_context_wrapper(get_tables, client),
        create_context_wrapper(get_lineage, client),
    ]

    if include_query_execution:
        tools.append(create_context_wrapper(run_query, client))

    return tools
