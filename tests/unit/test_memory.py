"""Tests for conversation memory (trim_messages)."""

from data_chat.memory import _group_into_exchanges, trim_messages


def _make_user_msg(text):
    return {"role": "user", "content": text}


def _make_assistant_msg(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _make_tool_use_msg(tool_name, tool_id="t1"):
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {"query": "x"}},
        ],
    }


def _make_tool_result_msg(tool_id="t1"):
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tool_id, "content": '{"result": "ok"}'},
        ],
    }


def test_trim_messages_empty():
    """Empty messages should be returned as-is."""
    messages = []
    result = trim_messages(messages)
    assert result == []


def test_trim_messages_within_budget():
    """Messages within budget should not be trimmed."""
    messages = [
        _make_user_msg("hello"),
        _make_assistant_msg("hi there"),
    ]
    original_len = len(messages)
    result = trim_messages(messages, token_budget=100_000)
    assert len(result) == original_len


def test_trim_messages_trims_oldest():
    """Should trim oldest messages when over budget."""
    messages = [
        _make_user_msg("first question " * 100),
        _make_assistant_msg("first answer " * 100),
        _make_user_msg("second question " * 100),
        _make_assistant_msg("second answer " * 100),
        _make_user_msg("third question"),
        _make_assistant_msg("third answer"),
    ]
    # Use a very small budget to force trimming
    result = trim_messages(messages, token_budget=100)
    # Should have trimmed some messages but kept at least the last exchange
    assert len(result) < 6
    assert len(result) >= 1


def test_trim_messages_keeps_minimum():
    """Should always keep at least the last exchange."""
    messages = [
        _make_user_msg("huge message " * 1000),
        _make_assistant_msg("huge response " * 1000),
    ]
    result = trim_messages(messages, token_budget=1)
    # Even with budget=1, should keep at least the last exchange
    assert len(result) >= 1


def test_trim_messages_preserves_tool_pairs():
    """Tool use + tool result should be kept together as atomic pairs."""
    messages = [
        _make_user_msg("find tables"),
        _make_tool_use_msg("search", "t1"),
        _make_tool_result_msg("t1"),
        _make_assistant_msg("Found tables."),
        _make_user_msg("show me columns"),
        _make_assistant_msg("Here are the columns."),
    ]
    # With a moderate budget, should trim as atomic units
    result = trim_messages(messages, token_budget=200)
    # Verify no orphaned tool_use or tool_result messages
    for i, msg in enumerate(result):
        content = msg.get("content", [])
        if isinstance(content, list):
            has_tool_use = any(
                isinstance(b, dict) and b.get("type") == "tool_use"
                for b in content
            )
            if has_tool_use and i + 1 < len(result):
                next_content = result[i + 1].get("content", [])
                if isinstance(next_content, list):
                    has_tool_result = any(
                        isinstance(b, dict) and b.get("type") == "tool_result"
                        for b in next_content
                    )
                    assert has_tool_result, "tool_use without matching tool_result"


def test_group_into_exchanges_basic():
    """Basic user/assistant messages should each be their own exchange."""
    messages = [
        _make_user_msg("hello"),
        _make_assistant_msg("hi"),
        _make_user_msg("bye"),
        _make_assistant_msg("goodbye"),
    ]
    exchanges = _group_into_exchanges(messages)
    assert len(exchanges) == 4
    assert all(len(e) == 1 for e in exchanges)


def test_group_into_exchanges_tool_pair():
    """Tool use + tool result should be grouped together."""
    messages = [
        _make_user_msg("find tables"),
        _make_tool_use_msg("search"),
        _make_tool_result_msg(),
        _make_assistant_msg("Found tables."),
    ]
    exchanges = _group_into_exchanges(messages)
    assert len(exchanges) == 3  # user, (tool_use + tool_result), assistant
    # The tool pair should be in one exchange
    tool_exchange = exchanges[1]
    assert len(tool_exchange) == 2
    assert tool_exchange[0]["role"] == "assistant"
    assert tool_exchange[1]["role"] == "user"


def test_trim_messages_realistic_conversation():
    """Simulate a realistic multi-turn conversation with tools."""
    messages = []

    # Turn 1: user asks, agent calls search, then responds
    messages.append(_make_user_msg("What customer tables do we have?"))
    messages.append(_make_tool_use_msg("search", "t1"))
    messages.append(_make_tool_result_msg("t1"))
    messages.append(_make_assistant_msg("I found 3 customer tables."))

    # Turn 2: user asks for details, agent calls get_tables, then responds
    messages.append(_make_user_msg("Show me the columns of CUSTOMERS."))
    messages.append(_make_tool_use_msg("get_tables", "t2"))
    messages.append(_make_tool_result_msg("t2"))
    messages.append(_make_assistant_msg("CUSTOMERS has 15 columns."))

    # Turn 3: simple question
    messages.append(_make_user_msg("Thanks!"))
    messages.append(_make_assistant_msg("You're welcome!"))

    # With large budget, nothing should be trimmed
    original_len = len(messages)
    result = trim_messages(messages, token_budget=100_000)
    assert len(result) == original_len

    # With tiny budget, should keep at least the last exchange
    messages_copy = list(messages)
    result = trim_messages(messages_copy, token_budget=10)
    assert len(result) >= 1
