"""Tests for context layer (context.py + utils.py).

Mirrors: datahub-agent-context/tests/unit/test_utils.py

Tests the core architectural pattern:
- contextvars set/get/reset
- DataChatContext manager
- create_context_wrapper preserves metadata and manages context
"""

from unittest.mock import Mock

import pytest

from data_chat.context import DataChatContext, get_client, get_connection, set_client, reset_client
from data_chat.utils import create_context_wrapper


@pytest.fixture
def mock_client():
    """Create a mock SnowflakeClient."""
    mock = Mock()
    mock.connection = Mock()
    return mock


# --- Low-level: set_client / get_client / reset_client ---


def test_get_client_raises_when_no_context():
    """get_client() should raise RuntimeError when no context is set."""
    with pytest.raises(RuntimeError, match="No SnowflakeClient in context"):
        get_client()


def test_set_and_get_client(mock_client):
    """set_client + get_client round-trip should work."""
    token = set_client(mock_client)
    try:
        assert get_client() is mock_client
    finally:
        reset_client(token)


def test_reset_restores_previous_value(mock_client):
    """reset_client should restore None (no context)."""
    token = set_client(mock_client)
    reset_client(token)

    with pytest.raises(RuntimeError, match="No SnowflakeClient in context"):
        get_client()


def test_get_connection_returns_underlying_connection(mock_client):
    """get_connection() should return client.connection."""
    token = set_client(mock_client)
    try:
        assert get_connection() is mock_client.connection
    finally:
        reset_client(token)


# --- DataChatContext manager ---


def test_context_manager_sets_client(mock_client):
    """DataChatContext should make client available via get_client()."""
    with DataChatContext(mock_client):
        assert get_client() is mock_client


def test_context_manager_resets_on_exit(mock_client):
    """DataChatContext should reset context after exiting."""
    with DataChatContext(mock_client):
        pass

    with pytest.raises(RuntimeError, match="No SnowflakeClient in context"):
        get_client()


def test_context_manager_resets_on_exception(mock_client):
    """DataChatContext should reset context even when exception occurs."""
    with pytest.raises(ValueError, match="boom"):
        with DataChatContext(mock_client):
            raise ValueError("boom")

    with pytest.raises(RuntimeError, match="No SnowflakeClient in context"):
        get_client()


def test_context_manager_nesting(mock_client):
    """Nested DataChatContext should restore outer context on inner exit."""
    outer_client = Mock()
    outer_client.connection = Mock()

    with DataChatContext(outer_client):
        assert get_client() is outer_client

        with DataChatContext(mock_client):
            assert get_client() is mock_client

        # Inner exited — should be back to outer
        assert get_client() is outer_client


# --- create_context_wrapper ---


def test_wrapper_sets_client_during_call(mock_client):
    """Wrapper should set client in context for the duration of the call.

    Mirrors: datahub test_sets_client_in_context_during_call
    """
    def tool():
        return get_client()

    wrapped = create_context_wrapper(tool, mock_client)
    result = wrapped()
    assert result is mock_client


def test_wrapper_resets_context_after_call(mock_client):
    """Wrapper should reset context after function returns.

    Mirrors: datahub test_resets_context_after_call
    """
    wrapped = create_context_wrapper(lambda: None, mock_client)
    wrapped()

    with pytest.raises(RuntimeError, match="No SnowflakeClient in context"):
        get_client()


def test_wrapper_resets_context_on_exception(mock_client):
    """Wrapper should reset context even when function raises.

    Mirrors: datahub test_resets_context_on_exception
    """
    def failing_tool():
        raise ValueError("boom")

    wrapped = create_context_wrapper(failing_tool, mock_client)
    with pytest.raises(ValueError, match="boom"):
        wrapped()

    with pytest.raises(RuntimeError, match="No SnowflakeClient in context"):
        get_client()


def test_wrapper_preserves_function_metadata(mock_client):
    """Wrapper should preserve name and docstring (critical for LLM).

    Mirrors: datahub test_preserves_function_metadata
    """
    def my_tool(x: int, y: str = "default") -> str:
        """My tool docstring."""
        return f"{x} {y}"

    wrapped = create_context_wrapper(my_tool, mock_client)
    assert wrapped.__name__ == "my_tool"
    assert wrapped.__doc__ == "My tool docstring."


def test_wrapper_passes_args_and_kwargs(mock_client):
    """Wrapper should forward all args and kwargs.

    Mirrors: datahub test_passes_args_and_kwargs
    """
    def tool(a, b, c=10):
        return (a, b, c)

    wrapped = create_context_wrapper(tool, mock_client)
    assert wrapped(1, 2, c=3) == (1, 2, 3)
