#!/usr/bin/env python3
"""
Basic LangChain agent example using data-chat tools.

Supports three LLM backends:
1. SoFi LLM Proxy (default) — uses llm-proxy-keys CLI + ChatAnthropic
2. AWS Bedrock — uses boto3 + ChatBedrock
3. Direct Anthropic API — uses ANTHROPIC_API_KEY + ChatAnthropic

Prerequisites:
    pip install 'data-chat[langchain]' langchain-anthropic
    Snowflake connection configured via environment variables

Environment variables (Snowflake):
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_WAREHOUSE,
    SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA

Environment variables (LLM — pick one):
    Option A: SoFi LLM Proxy (default, no env vars needed)
    Option B: AWS Bedrock — set AWS_REGION, requires langchain-aws + boto3
              Set LLM_BACKEND=bedrock to use this
    Option C: Direct Anthropic API — set ANTHROPIC_API_KEY
"""

import os
import subprocess

from langchain.agents import create_agent

from data_chat.client import SnowflakeClient
from data_chat.langchain_tools import build_langchain_tools

LLM_PROXY_BASE_URL = "https://internal.sofitest.com/llm-proxy"
LLM_PROXY_KEYS_CLI = os.path.expanduser("~/.local/bin/llm-proxy-keys")
LLM_PROXY_MODEL = "claude-sonnet-4-6"


def _get_proxy_api_key():
    """Get API key from SoFi's llm-proxy-keys CLI."""
    if not os.path.exists(LLM_PROXY_KEYS_CLI):
        return None
    try:
        result = subprocess.run(
            [LLM_PROXY_KEYS_CLI, "--no-refresh", "--quiet"],
            capture_output=True, text=True, timeout=5,
        )
        key = result.stdout.strip()
        return key if key and key.startswith("sk-") else None
    except Exception:
        return None


def create_llm():
    """Create a LangChain chat model, preferring SoFi LLM Proxy."""
    backend = os.getenv("LLM_BACKEND", "auto")

    # Option A: SoFi LLM Proxy (default)
    if backend in ("auto", "proxy"):
        proxy_key = _get_proxy_api_key()
        if proxy_key:
            from langchain_anthropic import ChatAnthropic

            print(f"Using SoFi LLM Proxy: {LLM_PROXY_BASE_URL}")
            return ChatAnthropic(
                model=LLM_PROXY_MODEL,
                anthropic_api_key=proxy_key,
                anthropic_api_url=LLM_PROXY_BASE_URL,
                max_tokens=4096,
                temperature=0,
            )

    # Option B: AWS Bedrock
    if backend in ("auto", "bedrock") and os.getenv("AWS_REGION"):
        import boto3
        from langchain_aws import ChatBedrock

        aws_region = os.getenv("AWS_REGION", "us-west-2")
        print(f"Using AWS Bedrock in {aws_region}")
        return ChatBedrock(
            client=boto3.client("bedrock-runtime", region_name=aws_region),
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            model_kwargs={"max_tokens": 4096, "temperature": 0},
        )

    # Option C: Direct Anthropic API
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        from langchain_anthropic import ChatAnthropic

        print("Using direct Anthropic API")
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            anthropic_api_key=api_key,
            max_tokens=4096,
            temperature=0,
        )

    raise RuntimeError(
        "No LLM backend found. Either:\n"
        "  - Install llm-proxy-keys CLI (SoFi employees)\n"
        "  - Set LLM_BACKEND=bedrock + AWS_REGION\n"
        "  - Set ANTHROPIC_API_KEY"
    )


def main():
    """Run the basic agent example."""
    # Step 1: Connect to Snowflake
    print("Connecting to Snowflake...")
    client = SnowflakeClient.from_env()

    # Step 2: Build tools
    tools = build_langchain_tools(client)
    print(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")

    # Step 3: Initialize LLM (auto-detects backend)
    model = create_llm()

    # Step 4: Create agent
    system_prompt = """You are a helpful data analyst assistant with access to Snowflake.

You can help users:
- Find tables by searching keywords
- Understand table schemas and column types
- Run SQL queries to answer data questions

When searching, always examine the top results to find the most relevant table.
When asked about data, first find the right table, then examine its schema,
then write and run a query to answer the question.
Be concise but informative in your responses."""

    agent = create_agent(model, tools=tools, system_prompt=system_prompt)

    examples = [
        "Find all tables related to 'customers'",
        "What columns does the top customer table have?",
        "How many rows are in that table?",
    ]

    print("\n" + "=" * 80)
    print("Data Chat - LangChain Agent")
    print("=" * 80)

    print("\nExample queries:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")

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

            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )
            final_message = response["messages"][-1]
            print(f"\nAgent: {final_message.content}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

    client.close()


if __name__ == "__main__":
    main()
