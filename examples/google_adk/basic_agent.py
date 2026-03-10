#!/usr/bin/env python3
"""
Basic Google ADK agent example using data-chat tools with Gemini.

This example demonstrates how to create a Google ADK agent that can:
- Search for tables in Snowflake
- Get detailed column information
- Execute read-only SQL queries

Prerequisites:
    pip install 'data-chat[google-adk]'
    Set GOOGLE_API_KEY environment variable (or use Vertex AI ADC)
    Snowflake connection configured via environment variables

Environment variables (Snowflake):
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_WAREHOUSE,
    SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA
    Optional: SNOWFLAKE_ROLE, SNOWFLAKE_AUTHENTICATOR, SNOWFLAKE_PRIVATE_KEY_PATH

Environment variables (Google):
    GOOGLE_API_KEY: Google AI API key (for Gemini Developer API)
"""

import asyncio
import os

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from data_chat.client import SnowflakeClient
from data_chat.google_adk_tools import build_google_adk_tools

SYSTEM_PROMPT = """You are a helpful data analyst assistant with access to Snowflake.

You can help users:
- Find tables by searching keywords
- Understand table schemas and column types
- Run SQL queries to answer data questions

When searching, always examine the top results to find the most relevant table.
When asked about data, first find the right table, then examine its schema,
then write and run a query to answer the question.
Be concise but informative in your responses."""


async def run_query(runner: Runner, session_id: str, query: str) -> str:
    """Send a query to the agent and return the final response text."""
    response_text = ""
    async for event in runner.run_async(
        user_id="user",
        session_id=session_id,
        new_message=types.Content(
            role="user", parts=[types.Part(text=query)]
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text
    return response_text


async def main() -> None:
    """Run the basic agent example."""
    # Step 1: Connect to Snowflake
    print("Connecting to Snowflake...")
    client = SnowflakeClient.from_env()

    # Step 2: Build tools
    # Google ADK takes plain callables — no @tool decorator needed.
    tools = build_google_adk_tools(client)
    print(f"Loaded {len(tools)} tools: {[t.__name__ for t in tools]}")

    # Step 3: Create agent
    agent = Agent(
        model="gemini-2.0-flash",
        name="data_chat_agent",
        description="A data analyst assistant with access to Snowflake.",
        instruction=SYSTEM_PROMPT,
        tools=tools,
    )

    # Step 4: Set up session and runner
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="data_chat",
        user_id="user",
    )
    runner = Runner(
        agent=agent,
        app_name="data_chat",
        session_service=session_service,
    )

    examples = [
        "Find all tables related to 'customers'",
        "What columns does the top customer table have?",
        "How many rows are in that table?",
    ]

    print("\n" + "=" * 80)
    print("Data Chat - Google ADK Agent")
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

            # New session per turn keeps memory bounded
            turn_session = await session_service.create_session(
                app_name="data_chat",
                user_id="user",
            )
            answer = await run_query(runner, turn_session.id, user_input)
            print(f"\nAgent: {answer}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
