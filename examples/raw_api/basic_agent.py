#!/usr/bin/env python3
"""
Raw API agent example — uses the shared agent module.

Previously this file contained ~380 lines of inline agent logic.
Now it's a thin wrapper around data_chat.agent.run_agent() and
data_chat.llm.create_anthropic_client().

Prerequisites:
    pip install 'data-chat[agent]'
    Snowflake connection configured via environment variables
"""

from data_chat.agent import PrintCallbacks, run_agent
from data_chat.client import SnowflakeClient
from data_chat.llm import create_anthropic_client
from data_chat.tools import (
    get_lineage,
    get_tables,
    list_databases,
    list_schemas,
    run_query,
    search,
)


def main():
    """Run the raw API agent example."""
    print("Connecting to Snowflake...")
    sf_client = SnowflakeClient.from_env()

    tools = [list_databases, list_schemas, search, get_tables, get_lineage, run_query]
    print(f"Loaded {len(tools)} tools: {[t.__name__ for t in tools]}")

    client_anthropic, model = create_anthropic_client()

    print("\n" + "=" * 80)
    print("Data Chat - Raw Anthropic API Agent")
    print("=" * 80)
    print(f"Model: {model}")

    examples = [
        "What databases do I have access to?",
        "Find all tables related to 'customers'",
        "Show me lineage for that table",
    ]
    print("\nExample queries:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")

    print("\n" + "=" * 80)
    print("Interactive Mode - Type 'quit' to exit")
    print("=" * 80)

    messages = None  # Multi-turn conversation history

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            answer, messages = run_agent(
                client_anthropic, model, sf_client, tools, user_input,
                messages=messages,
                callbacks=PrintCallbacks(),
            )
            print(f"\nAgent: {answer}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

    sf_client.close()


if __name__ == "__main__":
    main()
