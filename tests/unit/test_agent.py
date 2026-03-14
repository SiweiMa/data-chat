"""Tests for shared agent module."""

import json
from typing import List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from data_chat.agent import (
    NoOpCallbacks,
    PrintCallbacks,
    function_to_tool_schema,
    run_agent,
)


# --- function_to_tool_schema() tests ---


def test_function_to_tool_schema_basic():
    """Should generate correct schema for a simple function."""
    def my_tool(query: str) -> dict:
        """Search for something."""
        pass

    schema = function_to_tool_schema(my_tool)
    assert schema["name"] == "my_tool"
    assert "Search for something" in schema["description"]
    assert schema["input_schema"]["properties"]["query"]["type"] == "string"
    assert "query" in schema["input_schema"]["required"]


def test_function_to_tool_schema_optional_params():
    """Optional params should not be in required list."""
    def my_tool(query: str, limit: int = 10, name: Optional[str] = None) -> dict:
        """Do something."""
        pass

    schema = function_to_tool_schema(my_tool)
    assert schema["input_schema"]["required"] == ["query"]
    assert "limit" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["properties"]["limit"]["type"] == "integer"


def test_function_to_tool_schema_list_param():
    """List[str] should become array type."""
    def my_tool(tables: List[str]) -> dict:
        """Get tables."""
        pass

    schema = function_to_tool_schema(my_tool)
    assert schema["input_schema"]["properties"]["tables"]["type"] == "array"
    assert schema["input_schema"]["properties"]["tables"]["items"]["type"] == "string"


# --- run_agent() tests ---


def _make_mock_response(stop_reason, content_blocks):
    """Helper to create a mock Anthropic response."""
    response = Mock()
    response.stop_reason = stop_reason
    response.usage = Mock(input_tokens=100, output_tokens=50)
    response.content = content_blocks
    return response


def _make_text_block(text):
    block = Mock()
    block.type = "text"
    block.text = text
    block.model_dump = Mock(return_value={"type": "text", "text": text})
    return block


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


@pytest.fixture
def mock_sf_client():
    client = Mock()
    client.connection = Mock()
    return client


@pytest.fixture
def mock_anthropic_client():
    return MagicMock()


def test_run_agent_single_turn(mock_anthropic_client, mock_sf_client):
    """Agent should return text when LLM responds without tool calls."""
    text_block = _make_text_block("Hello, I can help you with data questions.")
    response = _make_mock_response("end_turn", [text_block])
    mock_anthropic_client.messages.create.return_value = response

    def dummy_tool(query: str) -> dict:
        """A tool."""
        return {}

    answer, messages = run_agent(
        mock_anthropic_client, "test-model", mock_sf_client,
        [dummy_tool], "Hello",
    )
    assert answer == "Hello, I can help you with data questions."
    assert len(messages) == 2  # user + assistant


def test_run_agent_with_tool_call(mock_anthropic_client, mock_sf_client):
    """Agent should execute tools and return final text."""
    # First response: tool call
    tool_block = _make_tool_use_block("my_tool", {"query": "test"})
    response1 = _make_mock_response("tool_use", [tool_block])

    # Second response: text
    text_block = _make_text_block("Found 3 tables.")
    response2 = _make_mock_response("end_turn", [text_block])

    mock_anthropic_client.messages.create.side_effect = [response1, response2]

    def my_tool(query: str) -> dict:
        """A tool."""
        return {"results": [1, 2, 3]}

    answer, messages = run_agent(
        mock_anthropic_client, "test-model", mock_sf_client,
        [my_tool], "Find tables",
    )
    assert answer == "Found 3 tables."
    # user + assistant (tool_use) + user (tool_result) + assistant (text)
    assert len(messages) == 4


def test_run_agent_max_turns(mock_anthropic_client, mock_sf_client):
    """Agent should stop after max_turns even if LLM keeps calling tools."""
    tool_block = _make_tool_use_block("my_tool", {"query": "test"})
    response = _make_mock_response("tool_use", [tool_block])
    mock_anthropic_client.messages.create.return_value = response

    def my_tool(query: str) -> dict:
        """A tool."""
        return {"result": "ok"}

    answer, messages = run_agent(
        mock_anthropic_client, "test-model", mock_sf_client,
        [my_tool], "test", max_turns=2,
    )
    assert answer == "(Max turns reached)"
    assert mock_anthropic_client.messages.create.call_count == 2


def test_run_agent_callbacks_called(mock_anthropic_client, mock_sf_client):
    """Callbacks should be invoked at each stage."""
    text_block = _make_text_block("Done.")
    response = _make_mock_response("end_turn", [text_block])
    mock_anthropic_client.messages.create.return_value = response

    cb = Mock(spec=NoOpCallbacks)

    def dummy(query: str) -> dict:
        """Tool."""
        return {}

    run_agent(
        mock_anthropic_client, "test-model", mock_sf_client,
        [dummy], "test", callbacks=cb,
    )
    cb.on_turn_start.assert_called_once_with(1, 10)
    cb.on_llm_response.assert_called_once()
    cb.on_final_response.assert_called_once_with("Done.")


def test_run_agent_returns_messages_for_multiturn(mock_anthropic_client, mock_sf_client):
    """Messages returned should be reusable for multi-turn."""
    text_block = _make_text_block("First answer.")
    response = _make_mock_response("end_turn", [text_block])
    mock_anthropic_client.messages.create.return_value = response

    def dummy(query: str) -> dict:
        """Tool."""
        return {}

    _, messages1 = run_agent(
        mock_anthropic_client, "test-model", mock_sf_client,
        [dummy], "First question",
    )

    # Pass messages back for second turn
    text_block2 = _make_text_block("Second answer.")
    response2 = _make_mock_response("end_turn", [text_block2])
    mock_anthropic_client.messages.create.return_value = response2

    answer2, messages2 = run_agent(
        mock_anthropic_client, "test-model", mock_sf_client,
        [dummy], "Follow-up", messages=messages1,
    )
    assert answer2 == "Second answer."
    # Original 2 + new user + new assistant = 4
    assert len(messages2) == 4


def test_run_agent_tool_error_returns_error_json(mock_anthropic_client, mock_sf_client):
    """Tool errors should be sent back to LLM as error JSON."""
    tool_block = _make_tool_use_block("failing_tool", {"query": "test"})
    response1 = _make_mock_response("tool_use", [tool_block])

    text_block = _make_text_block("I encountered an error.")
    response2 = _make_mock_response("end_turn", [text_block])

    mock_anthropic_client.messages.create.side_effect = [response1, response2]

    def failing_tool(query: str) -> dict:
        """A tool that fails."""
        raise ValueError("Table not found")

    answer, messages = run_agent(
        mock_anthropic_client, "test-model", mock_sf_client,
        [failing_tool], "test",
    )
    # The tool error should have been sent to the LLM, which responded with text
    assert answer == "I encountered an error."
    # Check the tool result message contains the error
    tool_result_msg = messages[2]  # user message with tool_result
    assert tool_result_msg["role"] == "user"
    result_content = tool_result_msg["content"][0]["content"]
    assert "Table not found" in result_content


def test_run_agent_raises_on_consecutive_sf_errors(mock_anthropic_client, mock_sf_client):
    """Should raise SnowflakeConnectionError after 3 consecutive connection failures."""
    from data_chat.exceptions import SnowflakeConnectionError

    tool_block = _make_tool_use_block("failing_tool", {"query": "test"})
    response = _make_mock_response("tool_use", [tool_block])
    mock_anthropic_client.messages.create.return_value = response

    def failing_tool(query: str) -> dict:
        """A tool."""
        raise ConnectionError("Connection refused")

    with pytest.raises(SnowflakeConnectionError, match="3 consecutive"):
        run_agent(
            mock_anthropic_client, "test-model", mock_sf_client,
            [failing_tool], "test", max_turns=5,
        )
