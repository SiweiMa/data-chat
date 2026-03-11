#!/usr/bin/env python3
"""
Raw API agent example — no framework, just the Anthropic SDK + manual tool loop.

Uses SoFi's LLM Proxy (LiteLLM) by default, with fallback to direct Anthropic API.

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

Environment variables (LLM — pick one):
    Option A: SoFi LLM Proxy (default, no env vars needed)
        Uses llm-proxy-keys CLI + https://internal.sofitest.com/llm-proxy
    Option B: Direct Anthropic API
        ANTHROPIC_API_KEY: Your Anthropic API key
"""

import inspect
import json
import os
import subprocess
from typing import Any, Callable, Dict, List, Optional

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

# SoFi LLM Proxy config
LLM_PROXY_BASE_URL = "https://internal.sofitest.com/llm-proxy"
LLM_PROXY_KEYS_CLI = os.path.expanduser("~/.local/bin/llm-proxy-keys")
LLM_PROXY_MODEL = "claude-sonnet-4-6"

# Direct Anthropic API config (fallback)
DIRECT_MODEL = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# LLM Client setup
# ---------------------------------------------------------------------------


def _get_proxy_api_key() -> Optional[str]:
    """Get an API key from SoFi's llm-proxy-keys CLI.

    The CLI caches keys and handles Okta authentication automatically.
    Returns None if the CLI is not installed or fails.
    """
    if not os.path.exists(LLM_PROXY_KEYS_CLI):
        return None

    try:
        result = subprocess.run(
            [LLM_PROXY_KEYS_CLI, "--no-refresh", "--quiet"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        key = result.stdout.strip()
        if key and key.startswith("sk-"):
            return key
    except Exception:
        pass

    # Try with refresh (may open browser for Okta)
    try:
        result = subprocess.run(
            [LLM_PROXY_KEYS_CLI, "--quiet"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        key = result.stdout.strip()
        if key and key.startswith("sk-"):
            return key
    except Exception:
        pass

    return None


def create_anthropic_client() -> tuple[anthropic.Anthropic, str]:
    """Create an Anthropic client, preferring SoFi LLM Proxy.

    Returns:
        Tuple of (client, model_name)
    """
    # Option A: SoFi LLM Proxy
    proxy_key = _get_proxy_api_key()
    if proxy_key:
        print(f"Using SoFi LLM Proxy: {LLM_PROXY_BASE_URL}")
        print(f"Model: {LLM_PROXY_MODEL}")
        client = anthropic.Anthropic(
            api_key=proxy_key,
            base_url=LLM_PROXY_BASE_URL,
        )
        return client, LLM_PROXY_MODEL

    # Option B: Direct Anthropic API
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        print("Using direct Anthropic API")
        print(f"Model: {DIRECT_MODEL}")
        client = anthropic.Anthropic(api_key=api_key)
        return client, DIRECT_MODEL

    raise RuntimeError(
        "No LLM API key found. Either:\n"
        "  - Install llm-proxy-keys CLI (SoFi employees)\n"
        "  - Set ANTHROPIC_API_KEY environment variable"
    )


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
    model: str,
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
        # Call Claude API
        response = client_anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tool_definitions,
            messages=messages,
        )

        # Check if Claude wants to call tools
        if response.stop_reason == "tool_use":
            # Claude returned one or more tool_use blocks
            # Add Claude's full response to conversation history
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_id = block.id

                    print(f"  [Tool call] {tool_name}({json.dumps(tool_input, default=str)[:100]}...)")

                    # Execute the tool within our DataChatContext
                    # This is what create_context_wrapper does automatically
                    try:
                        with DataChatContext(sf_client):
                            func = tool_dispatch[tool_name]
                            result = func(**tool_input)
                        result_str = json.dumps(result, default=str)
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_str,
                    })

            # Send tool results back to Claude
            messages.append({"role": "user", "content": tool_results})

        else:
            # Claude returned a final text response (stop_reason="end_turn")
            # Extract the text from the response
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

    # Step 3: Initialize LLM client (prefers SoFi proxy, falls back to direct API)
    client_anthropic, model = create_anthropic_client()

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
            answer = run_agent(client_anthropic, model, sf_client, tools_list, user_input)
            print(f"\nAgent: {answer}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

    sf_client.close()


if __name__ == "__main__":
    main()
