"""Shared agent module for data-chat.

Extracted from examples/raw_api/basic_agent.py to be reusable across
the CLI example, Streamlit app, and any future interfaces.

Provides:
- AgentCallbacks protocol for UI integration
- function_to_tool_schema() for generating Anthropic tool definitions
- run_agent() for the core agent loop
"""

import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Protocol

import anthropic

from data_chat import DataChatContext
from data_chat.client import SnowflakeClient

logger = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 10

DEFAULT_SYSTEM_PROMPT = """You are a helpful data analyst assistant with access to Snowflake.

You can help users:
- Discover databases and schemas (list_databases, list_schemas)
- Find tables by searching keywords (search)
- Understand table schemas and column types (get_tables)
- Trace data lineage between tables (get_lineage)
- Run SQL queries to answer data questions (run_query)

When searching, always examine the top results to find the most relevant table.
When asked about data, first find the right table, then examine its schema,
then write and run a query to answer the question.
Be concise but informative in your responses."""

# Maximum consecutive Snowflake connection failures before aborting
_MAX_CONSECUTIVE_SF_FAILURES = 3


# ---------------------------------------------------------------------------
# Callbacks protocol
# ---------------------------------------------------------------------------


class AgentCallbacks(Protocol):
    """Protocol for agent event callbacks.

    Implement this to integrate the agent loop with any UI (CLI, Streamlit, etc.).
    All methods have default no-op implementations via NoOpCallbacks.
    """

    def on_turn_start(self, turn: int, max_turns: int) -> None: ...
    def on_llm_response(self, response: Any) -> None: ...
    def on_tool_start(self, name: str, input: dict) -> None: ...
    def on_tool_result(self, name: str, result: str) -> None: ...
    def on_tool_error(self, name: str, error: Exception) -> None: ...
    def on_final_response(self, text: str) -> None: ...


class NoOpCallbacks:
    """Default no-op callbacks. Subclass and override what you need."""

    def on_turn_start(self, turn: int, max_turns: int) -> None:
        pass

    def on_llm_response(self, response: Any) -> None:
        pass

    def on_tool_start(self, name: str, input: dict) -> None:
        pass

    def on_tool_result(self, name: str, result: str) -> None:
        pass

    def on_tool_error(self, name: str, error: Exception) -> None:
        pass

    def on_final_response(self, text: str) -> None:
        pass


class PrintCallbacks(NoOpCallbacks):
    """Callbacks that print to stdout. For CLI examples."""

    def on_turn_start(self, turn: int, max_turns: int) -> None:
        print(f"\n{'─' * 60}")
        print(f"  Turn {turn}/{max_turns}")
        print(f"{'─' * 60}")

    def on_llm_response(self, response: Any) -> None:
        print(f"  stop_reason : {response.stop_reason}")
        print(f"  usage       : input={response.usage.input_tokens}, output={response.usage.output_tokens}")
        print(f"  content blocks ({len(response.content)}):")
        for i, block in enumerate(response.content):
            if block.type == "text":
                preview = block.text[:100] + "..." if len(block.text) > 100 else block.text
                print(f'    [{i}] text: "{preview}"')
            elif block.type == "tool_use":
                print(f"    [{i}] tool_use: {block.name}(id={block.id})")
                print(f"         input: {json.dumps(block.input, default=str)}")

    def on_tool_start(self, name: str, input: dict) -> None:
        print(f"\n  > Executing {name}({json.dumps(input, default=str)})")

    def on_tool_result(self, name: str, result: str) -> None:
        preview = result[:200] + "..." if len(result) > 200 else result
        print(f"  < Result: {preview}")

    def on_tool_error(self, name: str, error: Exception) -> None:
        print(f"  ! Error in {name}: {error}")

    def on_final_response(self, text: str) -> None:
        print(f"\n  Done.")


# ---------------------------------------------------------------------------
# Tool schema generation
# ---------------------------------------------------------------------------


def _python_type_to_json_schema(annotation: Any) -> dict:
    """Convert a Python type annotation to a JSON Schema type."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = getattr(annotation, "__origin__", None)

    # List[str], List[int], etc.
    if origin is list:
        args = getattr(annotation, "__args__", (Any,))
        return {"type": "array", "items": _python_type_to_json_schema(args[0])}

    # Optional[X] = Union[X, None]
    if origin is type(None):
        return {"type": "string"}

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    if isinstance(annotation, type) and annotation in type_map:
        return {"type": type_map[annotation]}

    return {"type": "string"}


def function_to_tool_schema(func: Callable) -> dict:
    """Convert a Python function into an Anthropic tool definition.

    The tool definition has three parts:
    - name: function name (Claude uses this to call it)
    - description: docstring (Claude reads this to decide WHEN to call it)
    - input_schema: JSON Schema derived from function signature
    """
    sig = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""

    properties: Dict[str, dict] = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        prop = _python_type_to_json_schema(param.annotation)
        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": func.__name__,
        "description": docstring,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_agent(
    anthropic_client: anthropic.Anthropic,
    model: str,
    sf_client: SnowflakeClient,
    tools: List[Callable],
    user_message: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_turns: int = DEFAULT_MAX_TURNS,
    messages: Optional[List[dict]] = None,
    callbacks: Optional[AgentCallbacks] = None,
) -> tuple[str, List[dict]]:
    """Run one agent turn: user message -> (tool calls)* -> final text.

    This is the core agent loop extracted from basic_agent.py.

    Args:
        anthropic_client: Anthropic API client
        model: Model name (e.g., "claude-sonnet-4-20250514")
        sf_client: SnowflakeClient for tool execution
        tools: List of tool functions to make available
        user_message: The user's input message
        system_prompt: System prompt for the agent
        max_turns: Maximum tool-calling rounds
        messages: Existing conversation history for multi-turn.
            If None, starts fresh. Messages are modified in place.
        callbacks: Optional callbacks for UI integration

    Returns:
        Tuple of (final_text, updated_messages). The messages list
        includes the full conversation history for multi-turn use.

    Raises:
        SnowflakeConnectionError: After _MAX_CONSECUTIVE_SF_FAILURES consecutive
            Snowflake connection errors during tool dispatch.
        LLMRateLimitError: When the LLM API returns 429.
        LLMAPIError: When the LLM API returns a server error.
        LLMConnectionError: When the LLM API is unreachable.
    """
    from data_chat.exceptions import (
        LLMAPIError,
        LLMConnectionError,
        LLMRateLimitError,
        SnowflakeConnectionError,
        SnowflakeSessionExpiredError,
    )

    cb = callbacks or NoOpCallbacks()

    # Build tool definitions
    tool_definitions = [function_to_tool_schema(t) for t in tools]
    tool_dispatch: Dict[str, Callable] = {t.__name__: t for t in tools}

    # Initialize or extend messages
    if messages is None:
        messages = []
    messages.append({"role": "user", "content": user_message})

    consecutive_sf_failures = 0

    for turn in range(1, max_turns + 1):
        cb.on_turn_start(turn, max_turns)

        # Call LLM with error wrapping
        try:
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                tools=tool_definitions,
                messages=messages,
            )
        except anthropic.RateLimitError as e:
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                retry_after_str = e.response.headers.get("retry-after")
                if retry_after_str:
                    try:
                        retry_after = float(retry_after_str)
                    except (ValueError, TypeError):
                        pass
            raise LLMRateLimitError(str(e), retry_after=retry_after) from e
        except anthropic.APIStatusError as e:
            status_code = e.status_code if hasattr(e, "status_code") else None
            raise LLMAPIError(str(e), status_code=status_code) from e
        except anthropic.APIConnectionError as e:
            raise LLMConnectionError(str(e)) from e

        cb.on_llm_response(response)

        # Serialize ContentBlocks to dicts for JSON-serializable message history
        content = [block.model_dump() for block in response.content]
        messages.append({"role": "assistant", "content": content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_id = block.id

                    cb.on_tool_start(tool_name, tool_input)

                    try:
                        with DataChatContext(sf_client):
                            func = tool_dispatch[tool_name]
                            result = func(**tool_input)
                        result_str = json.dumps(result, default=str)
                        consecutive_sf_failures = 0  # Reset on success
                        cb.on_tool_result(tool_name, result_str)
                    except Exception as e:
                        # Check for session expired
                        if "session does not exist" in str(e).lower():
                            raise SnowflakeSessionExpiredError(
                                "Snowflake session has expired. Please reconnect."
                            ) from e

                        # Check for connection errors (DatabaseError subclasses)
                        error_str = str(e).lower()
                        is_sf_connection_error = any(
                            phrase in error_str
                            for phrase in [
                                "connection",
                                "timeout",
                                "network",
                                "socket",
                            ]
                        )
                        if is_sf_connection_error:
                            consecutive_sf_failures += 1
                            if consecutive_sf_failures >= _MAX_CONSECUTIVE_SF_FAILURES:
                                raise SnowflakeConnectionError(
                                    f"Snowflake connection failed {consecutive_sf_failures} "
                                    f"consecutive times. Last error: {e}"
                                ) from e

                        # Return error to LLM for self-correction
                        result_str = json.dumps({"error": str(e)})
                        cb.on_tool_error(tool_name, e)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_str,
                    })

            messages.append({"role": "user", "content": tool_results})

        else:
            # Final text response
            for block in response.content:
                if hasattr(block, "text"):
                    cb.on_final_response(block.text)
                    return block.text, messages

            return "(No text in response)", messages

    return "(Max turns reached)", messages
