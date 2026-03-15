# data-chat

AI agent tools for querying Snowflake — a conversational interface for data discovery, lineage, and SQL execution.

## Quick Start

### Streamlit Web App

```bash
pip install -e '.[streamlit]'
streamlit run app/streamlit_app.py
```

1. Click **Connect to Snowflake** (opens Okta SSO in browser)
2. Click **Connect to LLM** (auto-detects SoFi Proxy or Anthropic API)
3. Ask questions: *"What databases do I have?"*, *"Show me lineage for REVENUE_SUMMARY"*

### CLI Agent

```bash
pip install -e '.[agent]'
python examples/raw_api/basic_agent.py
```

## Tools

| Tool | What it does |
|------|-------------|
| `list_databases()` | Show accessible databases |
| `list_schemas()` | Show schemas in a database |
| `search()` | Find tables by name, comment, or column |
| `get_tables()` | Get column details for specific tables |
| `get_lineage()` | Trace upstream/downstream dependencies |
| `run_query()` | Execute read-only SQL with safety guards |

The agent chains these automatically:

```
list_databases → list_schemas → search("customer") → get_tables([...]) → run_query("SELECT ...")
   explore         narrow down       find it              zoom in              see data
```

## Architecture

```
src/data_chat/
├── agent.py               # run_agent() loop + AgentCallbacks protocol
├── llm.py                 # LLM client factory (SoFi Proxy / Anthropic)
├── memory.py              # trim_messages() — sliding window for context
├── exceptions.py          # DataChatError hierarchy
├── client.py              # SnowflakeClient — connection wrapper
├── context.py             # contextvars — client injection
├── utils.py               # create_context_wrapper — bridge to frameworks
├── tools/
│   ├── base.py            # execute_query + clean_response + validate_identifier
│   ├── helpers.py         # token budget, sanitization, truncation
│   ├── _token_estimator.py
│   ├── navigation.py      # list_databases, list_schemas
│   ├── search.py          # search
│   ├── tables.py          # get_tables
│   ├── lineage.py         # get_lineage (SNOWFLAKE.CORE.GET_LINEAGE)
│   └── query.py           # run_query (read-only SQL)
├── langchain_tools/
│   └── builder.py         # build_langchain_tools(client)
└── google_adk_tools/
    └── builder.py         # build_google_adk_tools(client)

app/
└── streamlit_app.py       # Web chat UI
```

### How It Works

1. **Tools are plain functions** that call `get_connection()` from `contextvars` — no `client` parameter in the signature
2. **Framework builders** wrap each tool with `create_context_wrapper(func, client)` — injects the client before each call
3. **`run_agent()`** handles the LLM loop: send messages, execute tool calls, return final text
4. **`AgentCallbacks`** protocol lets any UI (CLI, Streamlit, etc.) hook into the agent loop without modifying it
5. **`trim_messages()`** keeps conversation history within the context window by trimming oldest exchanges

## Snowflake Connection

Auth auto-detection priority:
1. `SNOWFLAKE_PRIVATE_KEY_PATH` → key pair (no browser)
2. `SNOWFLAKE_AUTHENTICATOR` → externalbrowser or other
3. `SNOWFLAKE_PASSWORD` → password
4. None → defaults to `externalbrowser` (Okta SSO)

```bash
export SNOWFLAKE_ACCOUNT=gna62195.us-east-1
export SNOWFLAKE_USER=sma
export SNOWFLAKE_WAREHOUSE=DATA_SCIENCE_WAREHOUSE
export SNOWFLAKE_DATABASE=AI_DATAMART
export SNOWFLAKE_SCHEMA=PUBLIC
export SNOWFLAKE_ROLE=SNOWFLAKE_GALILEO_ST_DATASCIENCE
```

## LLM Backend

`create_anthropic_client()` auto-detects:
1. **SoFi LLM Proxy** (default) — uses `llm-proxy-keys` CLI
2. **Direct Anthropic** — set `ANTHROPIC_API_KEY`

## Safety

`run_query()` has three safety layers:
1. **Read-only validation** — rejects DROP, DELETE, INSERT, UPDATE, etc.
2. **LIMIT enforcement** — injects or caps LIMIT (max 1000)
3. **Token budget** — stops returning rows when budget is exhausted

`get_lineage()` and `list_schemas()` use `validate_identifier()` to prevent SQL injection in contexts where bind parameters are not supported.

## Installation Options

```bash
pip install -e '.'              # Base (tools only)
pip install -e '.[agent]'       # + Anthropic SDK (CLI agent)
pip install -e '.[streamlit]'   # + Streamlit (web app)
pip install -e '.[langchain]'   # + LangChain framework
pip install -e '.[google-adk]'  # + Google ADK framework
pip install -e '.[dev]'         # + ruff, mypy, pytest
```

## Testing

```bash
PYTHONPATH=src python3 -m pytest tests/unit/ -v   # 101 tests
```
