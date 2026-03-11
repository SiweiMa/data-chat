#!/usr/bin/env python3
"""
Raw API agent example — no framework, just the Anthropic SDK + manual tool loop.

This example shows EXACTLY what LangChain and Google ADK do under the hood:
1. Convert tool functions into JSON schemas (tool definitions)
2. Send messages + tool definitions to the LLM
3. When the LLM returns a tool_use block, execute the tool
4. Send the tool result back to the LLM
5. Repeat until the LLM returns a text response (no more tool calls)

This is the most educational example because nothing is hidden.

Prerequisites:
    pip install anthropic data-chat
    Snowflake connection configured via environment variables

Environment variables (Snowflake):
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_WAREHOUSE,
    SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA

Environment variables (Anthropic):
    ANTHROPIC_API_KEY: Your Anthropic API key
"""

import inspect
import json
import os
from typing import Any, Callable, Dict, List

import anthropic

from data_chat import DataChatContext
from data_chat.client import SnowflakeClient
from data_chat.tools import get_tables, run_query, search

SYSTEM_PROMPT = """You are a helpful data analyst assistant with access to Snowflake.

You can help users:
- Find tables by searching keywords
- Understand table schemas and column types
- Run SQL queries to answer data questions

When searching, always examine the top results to find the most relevant table.
When asked about data, first find the right table, then examine its schema,
then write and run a query to answer the question.
Be concise but informative in your responses."""

# Maximum tool-calling rounds before forcing a text response.
# Prevents infinite loops if the LLM keeps calling tools.
MAX_TURNS = 10


# ---------------------------------------------------------------------------
# Tool schema generation
# ---------------------------------------------------------------------------
# LangChain does this automatically from the function signature + docstring.
# Here we do it manually so you can see exactly what the LLM receives.


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

    This is what LangChain's @tool decorator does automatically.
    We do it manually here so you can see the exact JSON that gets
    sent to Claude's API.

    The tool definition has three parts:
    - name: function name (Claude uses this to call it)
    - description: docstring (Claude reads this to decide WHEN to call it)
    - input_schema: JSON Schema derived from function signature
    """
    sig = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""

    # Build JSON Schema properties from function parameters
    properties: Dict[str, dict] = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        prop = _python_type_to_json_schema(param.annotation)

        # Add description from docstring Args section if available
        properties[name] = prop

        # Parameters without defaults are required
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
    client_anthropic: anthropic.Anthropic,
    sf_client: SnowflakeClient,
    tools_list: List[Callable],
    user_message: str,
) -> str:
    """Run one agent turn: user message → (tool calls)* → final text.

    This is the core loop that LangChain's create_agent() hides from you.

    HOW IT WORKS:
    1. Send the user message + tool definitions to Claude
    2. If Claude returns tool_use blocks → execute each tool
    3. Send tool results back to Claude
    4. Repeat until Claude returns only text (stop_reason="end_turn")
    """
    # Build tool definitions (sent with every API call)
    tool_definitions = [function_to_tool_schema(t) for t in tools_list]

    # Build a name → function lookup for dispatching tool calls
    tool_dispatch: Dict[str, Callable] = {t.__name__: t for t in tools_list}

    # Conversation history for this turn
    messages: List[dict] = [
        {"role": "user", "content": user_message},
    ]

    for turn in range(MAX_TURNS):
        print(f"\n{'─' * 60}")
        print(f"  Turn {turn + 1}/{MAX_TURNS}")
        print(f"{'─' * 60}")
        print(f"  Sending {len(messages)} message(s) to Claude...")

        response = client_anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tool_definitions,
            messages=messages,
        )

        print(f"  stop_reason : {response.stop_reason}")
        print(f"  usage       : input={response.usage.input_tokens}, output={response.usage.output_tokens}")
        print(f"  content blocks ({len(response.content)}):")

        for i, block in enumerate(response.content):
            if block.type == "text":
                print(f"    [{i}] text: \"{block.text}\"")
            elif block.type == "tool_use":
                print(f"    [{i}] tool_use: {block.name}(id={block.id})")
                print(f"         input: {json.dumps(block.input, default=str)}")
            else:
                print(f"    [{i}] {block.type}: {block}")

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_id = block.id

                    print(f"\n  ▶ Executing {tool_name}({json.dumps(tool_input, default=str)})")

                    try:
                        with DataChatContext(sf_client):
                            func = tool_dispatch[tool_name]
                            result = func(**tool_input)
                        result_str = json.dumps(result, default=str)
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})

                    print(f"  ◀ Result: {result_str}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_str,
                    })

            messages.append({"role": "user", "content": tool_results})

        else:
            print(f"\n  ✓ Final response (stop_reason={response.stop_reason})")
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

            return "(No text in response)"

    return "(Max turns reached)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run the raw API agent example."""
    # Step 1: Connect to Snowflake
    print("Connecting to Snowflake...")
    sf_client = SnowflakeClient.from_env()

    # Step 2: Set up tools
    # No builder needed — we use the plain functions directly.
    # In the LangChain/ADK examples, the builder wraps these with
    # create_context_wrapper. Here we call DataChatContext manually
    # in the agent loop.
    tools_list = [search, get_tables, run_query]
    print(f"Loaded {len(tools_list)} tools: {[t.__name__ for t in tools_list]}")

    # Step 3: Initialize Anthropic client
    client_anthropic = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )

    examples = [
        "Find all tables related to 'customers'",
        "What columns does the top customer table have?",
        "How many rows are in that table?",
    ]

    print("\n" + "=" * 80)
    print("Data Chat - Raw Anthropic API Agent")
    print("=" * 80)

    print("\nExample queries:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")

    # Interactive mode
    print("\n" + "=" * 80)
    print("Interactive Mode - Type 'quit' to exit")
    print("=" * 80)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            print()
            answer = run_agent(client_anthropic, sf_client, tools_list, user_input)
            print(f"\nAgent: {answer}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

    sf_client.close()


if __name__ == "__main__":
    main()
