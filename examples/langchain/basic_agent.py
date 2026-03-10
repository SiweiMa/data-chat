#!/usr/bin/env python3
"""
Basic LangChain agent example using data-chat tools with AWS Bedrock.

This example demonstrates how to create a LangChain agent that can:
- Search for tables in Snowflake
- Get detailed column information
- Execute read-only SQL queries

Prerequisites:
    pip install 'data-chat[langchain]' langchain-aws boto3
    AWS credentials configured (via ~/.aws/credentials or environment variables)
    Snowflake connection configured via environment variables

Environment variables (Snowflake):
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_WAREHOUSE,
    SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA
    Optional: SNOWFLAKE_ROLE, SNOWFLAKE_AUTHENTICATOR, SNOWFLAKE_PRIVATE_KEY_PATH

Environment variables (AWS):
    AWS_REGION: AWS region for Bedrock (default: us-west-2)
"""

import os

import boto3
from langchain.agents import create_agent
from langchain_aws import ChatBedrock

from data_chat.client import SnowflakeClient
from data_chat.langchain_tools import build_langchain_tools


def main():
    """Run the basic agent example."""
    # Step 1: Connect to Snowflake
    # One line — all config comes from env vars.
    # Mirrors: DataHubClient.from_env() in DataHub's example.
    print("Connecting to Snowflake...")
    client = SnowflakeClient.from_env()

    # Step 2: Build tools
    # One line — creates all 3 tools with context injection.
    # Each tool's docstring becomes its LLM description automatically.
    tools = build_langchain_tools(client)
    print(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")

    # Step 3: Initialize LLM
    aws_region = os.getenv("AWS_REGION", "us-west-2")
    print(f"Connecting to AWS Bedrock in {aws_region}...")

    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime", region_name=aws_region
    )

    model = ChatBedrock(
        client=bedrock_runtime,
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        model_kwargs={"max_tokens": 4096, "temperature": 0},
    )

    # Step 4: Create agent
    # The system prompt guides the LLM's behavior.
    # Tool docstrings (from Steps 4-6) tell it HOW to use each tool.
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

    # Example queries that demonstrate the tool chain:
    #   search() → get_tables() → run_query()
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
