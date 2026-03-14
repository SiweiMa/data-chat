"""Builder for LangChain tools from data-chat tools.

Mirrors: datahub-agent-context/langchain_tools/builder.py

This is the GLUE between our plain tool functions (Steps 4-6) and
LangChain's agent framework.

HOW IT WORKS:
    1. Import each tool function (search, get_tables, run_query)
    2. Wrap each with create_context_wrapper(func, client) — injects
       the SnowflakeClient into contextvars before the tool runs
    3. Wrap with LangChain's @tool decorator — converts the plain
       function into a BaseTool with name, description, and schema
       derived from the function signature + docstring

    The result: LangChain sees a list of BaseTool objects. Each tool's
    description comes from the docstring we wrote in Steps 4-6.

WHY THIS IS SMALL (~30 lines of logic):
    All the complexity is in the layers below:
    - Tool functions handle query execution (Steps 4-6)
    - create_context_wrapper handles client injection (Step 2)
    - helpers.py handles token budgets (Step 3)
    This builder just wires them together.
"""

from typing import TYPE_CHECKING

from data_chat.utils import create_context_wrapper

if TYPE_CHECKING:
    from data_chat.client import SnowflakeClient

# Lazy import: fail with a clear message if langchain-core isn't installed.
# DataHub does the same thing at langchain_tools/builder.py:18-25.
# This lets the base package stay light — users only install langchain
# if they need it: pip install 'data-chat[langchain]'
try:
    from langchain_core.tools import tool  # type: ignore[import-not-found]
    from langchain_core.tools.base import BaseTool  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(
        "langchain-core is required for LangChain tools. "
        "Install with: pip install 'data-chat[langchain]'"
    ) from e

from data_chat.tools.lineage import get_lineage
from data_chat.tools.navigation import list_databases, list_schemas
from data_chat.tools.query import run_query
from data_chat.tools.search import search
from data_chat.tools.tables import get_tables


def build_langchain_tools(
    client: "SnowflakeClient",
    include_query_execution: bool = True,
) -> list[BaseTool]:
    """Build LangChain tools with automatic context management.

    Each tool is wrapped to automatically set the SnowflakeClient in
    context before execution. The LLM never sees the client parameter.

    Args:
        client: SnowflakeClient instance
        include_query_execution: Whether to include run_query() tool
            (default: True). Set to False for metadata-only agents.

    Returns:
        List of LangChain BaseTool instances

    Example:
        from data_chat.client import SnowflakeClient
        from data_chat.langchain_tools import build_langchain_tools

        client = SnowflakeClient.from_env()
        tools = build_langchain_tools(client)

        # Use with LangChain agents
        agent = create_react_agent(llm, tools, prompt)
        result = agent.invoke({"input": "find customer tables"})
    """
    # tool() converts a plain function → BaseTool using its signature + docstring.
    # create_context_wrapper() injects client before each call.
    # Combined: tool(create_context_wrapper(search, client))
    tools: list[BaseTool] = [
        tool(create_context_wrapper(list_databases, client)),
        tool(create_context_wrapper(list_schemas, client)),
        tool(create_context_wrapper(search, client)),
        tool(create_context_wrapper(get_tables, client)),
        tool(create_context_wrapper(get_lineage, client)),
    ]

    if include_query_execution:
        tools.append(tool(create_context_wrapper(run_query, client)))

    return tools
