"""Tests for agent error handling — LLM error wrapping and session expiry."""

from unittest.mock import MagicMock, Mock

import anthropic
import pytest

from data_chat.agent import run_agent
from data_chat.exceptions import (
    LLMAPIError,
    LLMConnectionError,
    LLMRateLimitError,
    SnowflakeSessionExpiredError,
)


@pytest.fixture
def mock_sf_client():
    client = Mock()
    client.connection = Mock()
    return client


def _make_tool_use_block(name, input_dict, tool_id="tool_1"):
    block = Mock()
    block.type = "tool_use"
    block.name = name
    block.input = input_dict
    block.id = tool_id
    block.model_dump = Mock(return_value={
        "type": "tool_use", "id": tool_id,
        "name": name, "input": input_dict,
    })
    return block


def _make_mock_response(stop_reason, content_blocks):
    response = Mock()
    response.stop_reason = stop_reason
    response.usage = Mock(input_tokens=100, output_tokens=50)
    response.content = content_blocks
    return response


def dummy_tool(query: str) -> dict:
    """A tool."""
    return {}


def test_agent_wraps_rate_limit_error(mock_sf_client):
    """RateLimitError should be wrapped as LLMRateLimitError."""
    client = MagicMock()
    mock_response = Mock()
    mock_response.status_code = 429
    mock_response.headers = {"retry-after": "30"}
    error = anthropic.RateLimitError(
        message="Rate limited",
        response=mock_response,
        body=None,
    )
    client.messages.create.side_effect = error

    with pytest.raises(LLMRateLimitError) as exc_info:
        run_agent(client, "test-model", mock_sf_client, [dummy_tool], "test")
    assert exc_info.value.retry_after == 30.0


def test_agent_wraps_api_connection_error(mock_sf_client):
    """APIConnectionError should be wrapped as LLMConnectionError."""
    client = MagicMock()
    client.messages.create.side_effect = anthropic.APIConnectionError(
        request=Mock(),
    )

    with pytest.raises(LLMConnectionError):
        run_agent(client, "test-model", mock_sf_client, [dummy_tool], "test")


def test_agent_wraps_api_status_error(mock_sf_client):
    """APIStatusError should be wrapped as LLMAPIError."""
    client = MagicMock()
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.headers = {}
    mock_response.json.return_value = {"error": {"message": "Internal error"}}
    error = anthropic.APIStatusError(
        message="Server error",
        response=mock_response,
        body=None,
    )
    client.messages.create.side_effect = error

    with pytest.raises(LLMAPIError) as exc_info:
        run_agent(client, "test-model", mock_sf_client, [dummy_tool], "test")
    assert exc_info.value.status_code == 500


def test_agent_session_expired_raises(mock_sf_client):
    """Session expired error should raise SnowflakeSessionExpiredError."""
    client = MagicMock()

    tool_block = _make_tool_use_block("session_tool", {"query": "test"})
    response = _make_mock_response("tool_use", [tool_block])
    client.messages.create.return_value = response

    def session_tool(query: str) -> dict:
        """A tool."""
        raise Exception("Session does not exist or has expired")

    with pytest.raises(SnowflakeSessionExpiredError, match="session has expired"):
        run_agent(client, "test-model", mock_sf_client, [session_tool], "test")
