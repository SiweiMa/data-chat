# data-chat — Agent Development Guide

AI agent tools for querying Snowflake, modeled after [datahub-agent-context](https://github.com/datahub-project/datahub/tree/master/datahub-agent-context).

## Essential Commands

```bash
# Run all tests (101 tests, ~2s)
PYTHONPATH=src python3 -m pytest tests/unit/ -v

# Run a specific test file
PYTHONPATH=src python3 -m pytest tests/unit/test_context.py -v

# Lint (requires dev install)
pip install -e '.[dev]'
ruff check src/ tests/
ruff format src/ tests/
```

**Verifying changes:**
- Always run tests after modifying any `src/` file
- Run `ruff check` before committing

## Architecture

```
src/data_chat/
├── agent.py               # run_agent() loop + AgentCallbacks protocol
├── llm.py                 # LLM client factory (SoFi Proxy / Anthropic)
├── memory.py              # trim_messages() — sliding window for context
├── exceptions.py          # DataChatError hierarchy
├── client.py              # SnowflakeClient — connection wrapper
├── context.py             # contextvars — client injection (THE core pattern)
├── utils.py               # create_context_wrapper — bridge to frameworks
├── tools/
│   ├── base.py            # execute_query + clean_response + validate_identifier
│   ├── helpers.py         # token budget, sanitization, truncation
│   ├── _token_estimator.py # fast token counting
│   ├── navigation.py      # list_databases, list_schemas
│   ├── search.py          # search() — discover tables
│   ├── tables.py          # get_tables() — column details
│   ├── lineage.py         # get_lineage() — data lineage
│   └── query.py           # run_query() — read-only SQL execution
├── langchain_tools/
│   └── builder.py         # build_langchain_tools(client) → List[BaseTool]
└── google_adk_tools/
    └── builder.py         # build_google_adk_tools(client) → List[Callable]

app/
└── streamlit_app.py       # Streamlit web chat UI
```

### How It Works

1. **Tools are plain functions** that call `get_connection()` from `contextvars` — no `client` parameter in the signature (LLMs can't see it)
2. **Framework builders** wrap each tool with `create_context_wrapper(func, client)` — injects the client before each call
3. **Token budgets** prevent tools from returning responses that blow up the LLM context window

### Tool Chain

```
list_databases → list_schemas → search("customer") → get_tables([...]) → run_query("SELECT ...")
   explore        narrow down      find it              zoom in              see data

get_lineage("DB.SCH.TABLE", direction="UPSTREAM") → trace data provenance
```

## Adding a New Tool

Follow the pattern in `tools/search.py`:

1. Create `tools/my_tool.py` with a function that calls `get_connection()`
2. Write a **rich docstring** (50+ lines) — the LLM reads it as its instruction manual
3. Apply `clean_response()` + `truncate_descriptions()` to the output
4. Use `select_results_within_budget()` if returning multiple items
5. Add to `tools/__init__.py`
6. Add to both builders (`langchain_tools/builder.py`, `google_adk_tools/builder.py`)
7. Add tests in `tests/unit/`

**Key rules for tool functions:**
- NO `conn` / `client` parameter — use `get_connection()` from context
- Cap results with a hard limit (e.g., `min(limit, 50)`)
- Return pagination metadata (`total`, `has_more`, `offset`)
- Set truncation flags when data is cut (`columns_truncated`, `rows_truncated`)

## Snowflake Connection

Auth auto-detection priority:
1. `SNOWFLAKE_PRIVATE_KEY_PATH` → key pair (no browser)
2. `SNOWFLAKE_AUTHENTICATOR` → externalbrowser or other
3. `SNOWFLAKE_PASSWORD` → password
4. None → defaults to `externalbrowser` (Okta SSO)

```bash
# Required
export SNOWFLAKE_ACCOUNT=gna62195.us-east-1
export SNOWFLAKE_USER=sma
export SNOWFLAKE_WAREHOUSE=DATA_SCIENCE_WAREHOUSE
export SNOWFLAKE_DATABASE=AI_DATAMART
export SNOWFLAKE_SCHEMA=PUBLIC
export SNOWFLAKE_ROLE=SNOWFLAKE_GALILEO_ST_DATASCIENCE
```

## LLM Backend

Examples auto-detect the LLM backend:
1. **SoFi LLM Proxy** (default) — uses `llm-proxy-keys` CLI + `https://internal.sofitest.com/llm-proxy`
2. **AWS Bedrock** — set `LLM_BACKEND=bedrock` + `AWS_REGION`
3. **Direct Anthropic** — set `ANTHROPIC_API_KEY`

## Running Examples

```bash
# Raw API (no framework, most educational)
pip install anthropic
python examples/raw_api/basic_agent.py

# LangChain
pip install 'data-chat[langchain]' langchain-anthropic
python examples/langchain/basic_agent.py

# Google ADK
pip install 'data-chat[google-adk]'
GOOGLE_API_KEY=... python examples/google_adk/basic_agent.py
```

## Safety (run_query)

`run_query()` has three safety layers:
1. **Read-only validation** — rejects DROP, DELETE, INSERT, UPDATE, etc.
2. **LIMIT enforcement** — injects or caps LIMIT (max 1000)
3. **Token budget** — stops returning rows when budget is exhausted

## DataHub Lineage

This project is modeled after `datahub-agent-context`. Key mapping:

| DataHub | data-chat | Layer |
|---------|-----------|-------|
| `DataHubClient` | `SnowflakeClient` | Connection |
| `DataHubContext` | `DataChatContext` | Context manager |
| `get_graph()` | `get_connection()` | Context retrieval |
| `execute_graphql()` | `execute_query()` | Query execution |
| `clean_gql_response()` | `clean_response()` | Response cleaning |
| `ENTITY_SCHEMA_TOKEN_BUDGET` | `PER_TABLE_TOKEN_BUDGET` | Per-entity budget |
| `schemaFieldsTruncated` | `columns_truncated` | Truncation flag |
| `search()` | `search()` | Discovery tool |
| `get_entities()` | `get_tables()` | Detail tool |
| (none) | `run_query()` | SQL execution |
| `build_langchain_tools()` | `build_langchain_tools()` | Framework builder |
| `build_google_adk_tools()` | `build_google_adk_tools()` | Framework builder |
