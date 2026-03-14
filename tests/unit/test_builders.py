"""Tests for framework builders (LangChain + Google ADK).

Mirrors: datahub-agent-context/tests/unit/langchain_tools/test_builder.py
         datahub-agent-context/tests/unit/google_adk_tools/test_builder.py

Tests that builders:
1. Return the correct number of tools
2. Preserve function metadata (name, docstring)
3. Respect include_query_execution flag
4. Context wrapper works end-to-end
"""

from unittest.mock import Mock

import pytest

from data_chat.context import get_client


@pytest.fixture
def mock_client():
    """Create a mock SnowflakeClient."""
    mock = Mock()
    mock.connection = Mock()
    return mock


# --- Google ADK builder (no framework dependency) ---


def test_google_adk_returns_correct_count(mock_client):
    """Should return 4 tools with query execution, 3 without."""
    from data_chat.google_adk_tools import build_google_adk_tools

    tools = build_google_adk_tools(mock_client, include_query_execution=True)
    assert len(tools) == 4

    tools_safe = build_google_adk_tools(mock_client, include_query_execution=False)
    assert len(tools_safe) == 3


def test_google_adk_tool_names(mock_client):
    """Tools should have correct function names."""
    from data_chat.google_adk_tools import build_google_adk_tools

    tools = build_google_adk_tools(mock_client)
    names = [t.__name__ for t in tools]
    assert "search" in names
    assert "get_tables" in names
    assert "run_query" in names
    assert "get_lineage" in names


def test_google_adk_excludes_run_query(mock_client):
    """include_query_execution=False should exclude run_query.

    Mirrors: DataHub's include_mutations=False excluding write tools.
    """
    from data_chat.google_adk_tools import build_google_adk_tools

    tools = build_google_adk_tools(mock_client, include_query_execution=False)
    names = [t.__name__ for t in tools]
    assert "run_query" not in names
    assert "search" in names
    assert "get_tables" in names
    assert "get_lineage" in names


def test_google_adk_preserves_docstrings(mock_client):
    """Each tool's docstring should be preserved (LLM reads it).

    Mirrors: DataHub test_create_context_wrapper_preserves_function_metadata
    """
    from data_chat.google_adk_tools import build_google_adk_tools

    tools = build_google_adk_tools(mock_client)
    for tool in tools:
        doc = tool.__doc__ or ""
        assert len(doc) > 100, f"{tool.__name__} docstring too short ({len(doc)} chars)"


def test_google_adk_wrapper_manages_context(mock_client):
    """Wrapped tools should set/reset context automatically.

    Mirrors: DataHub test_wrapped_tools_manage_context_automatically
    """
    from data_chat.google_adk_tools import build_google_adk_tools

    # Context should not be set initially
    with pytest.raises(RuntimeError):
        get_client()

    tools = build_google_adk_tools(mock_client)
    # Find the search tool
    search_tool = next(t for t in tools if t.__name__ == "search")

    # We can't call search_tool (it needs a real Snowflake cursor),
    # but we can verify the wrapper structure
    assert search_tool.__wrapped__.__name__ == "search"  # functools.wraps sets this

    # Context should still not be set (no tool was called)
    with pytest.raises(RuntimeError):
        get_client()


# --- LangChain builder (may not be installed) ---


def test_langchain_import_error():
    """Should give clear install instructions if langchain-core is missing."""
    try:
        from data_chat.langchain_tools import build_langchain_tools
        # If we get here, langchain is installed — skip this test
        pytest.skip("langchain-core is installed")
    except ImportError as e:
        assert "pip install" in str(e)
        assert "data-chat[langchain]" in str(e)


def test_langchain_builder_if_available(mock_client):
    """If langchain is installed, test the builder works."""
    try:
        from data_chat.langchain_tools import build_langchain_tools
    except ImportError:
        pytest.skip("langchain-core not installed")

    tools = build_langchain_tools(mock_client)
    assert isinstance(tools, list)
    assert len(tools) > 0

    # Each tool should have LangChain BaseTool attributes
    for tool in tools:
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")

    # Check tool names
    tool_names = {t.name for t in tools}
    assert "search" in tool_names
    assert "get_tables" in tool_names
    assert "run_query" in tool_names
    assert "get_lineage" in tool_names


def test_langchain_excludes_run_query_if_available(mock_client):
    """include_query_execution=False should exclude run_query."""
    try:
        from data_chat.langchain_tools import build_langchain_tools
    except ImportError:
        pytest.skip("langchain-core not installed")

    tools = build_langchain_tools(mock_client, include_query_execution=False)
    tool_names = {t.name for t in tools}
    assert "run_query" not in tool_names
    assert "search" in tool_names
    assert "get_lineage" in tool_names
