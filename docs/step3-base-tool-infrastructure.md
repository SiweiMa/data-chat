# Step 3: Base Tool Infrastructure

## The Problem

Every tool in the agent needs to:
1. Execute a query against Snowflake
2. Clean the response (remove noise)
3. Stay within the LLM's context window (token budget)

Without a shared base layer, each tool would duplicate this logic. DataHub has 19 tools — that's 19 copies of cursor management, response cleaning, and token counting. The base layer eliminates that duplication.

## Architecture

```
Tool Function (search, get_tables, run_query)
    │
    ├── calls execute_query(conn, sql)          ← base.py
    │       │
    │       ├── conn.cursor()
    │       ├── cursor.execute(sql)
    │       ├── cursor.fetchall() → list of tuples
    │       └── zip(column_names, row) → list of dicts
    │
    ├── calls clean_response(data)              ← base.py
    │       │
    │       └── recursively removes:
    │           • None values      ("comment": null → removed)
    │           • empty lists      ("tags": [] → removed)
    │           • empty dicts      ("metadata": {} → removed)
    │
    ├── calls truncate_descriptions(data)       ← helpers.py
    │       │
    │       └── recursively finds "description"/"comment" keys
    │           and caps them at 1000 chars
    │
    └── calls select_results_within_budget()    ← helpers.py
            │
            └── yields results one-by-one, tracking token count
                stops when budget would be exceeded
                uses TokenCountEstimator for fast approximation
```

## Files Created

### `_token_estimator.py` — Token Counting Without a Tokenizer

**DataHub source**: `datahub-agent-context/src/datahub_agent_context/mcp_tools/_token_estimator.py`

**Problem**: We need to know how many tokens a response will cost before sending it to the LLM. Real tokenizers (tiktoken, sentencepiece) are slow and add heavy dependencies.

**Solution**: Approximate using character counts. The heuristic:

```
tokens ≈ 1.3 × characters / 4
```

The 1.3 multiplier accounts for JSON structural overhead — quotes, colons, commas, and braces often tokenize as separate tokens, making JSON "more expensive" per character than plain text.

**Why precision doesn't matter**: DataHub uses a 90% budget (72K of 80K limit). The 10% buffer absorbs any estimation error. Being off by 20% is fine when you have a 10% safety margin on a generous limit.

**Example**:
```python
small = {"name": "users", "columns": ["id", "email"]}
# → 20 tokens (estimated)

large = {"name": "users", "columns": [{"name": f"col_{i}", "type": "VARCHAR"} for i in range(100)]}
# → 1,765 tokens (estimated)
```

### `base.py` — Query Execution and Response Cleaning

**DataHub source**: `datahub-agent-context/src/datahub_agent_context/mcp_tools/base.py`

#### `execute_query(conn, sql, params)`

DataHub's equivalent is `execute_graphql(graph, query, variables)`. Both serve the same purpose: **centralize query execution so tools don't manage cursors/connections themselves**.

DataHub's version is more complex because GraphQL has version-specific concerns:
- Cloud vs OSS field detection (`#[CLOUD]` markers)
- Automatic fallback when newer GMS fields aren't supported
- Field validation error recovery

Our version is simpler — Snowflake SQL doesn't have these concerns. But the pattern is the same: take a connection + query, return structured data, handle errors in one place.

**Key design choice — returning list-of-dicts**:
```python
# Raw cursor gives you: [(1, "alice"), (2, "bob")]
# We convert to:        [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
```
Dicts are self-describing — the LLM sees column names without needing a separate schema. Tuples would require the LLM to remember positional mapping.

#### `clean_response(response)`

DataHub's equivalent is `clean_gql_response()`. Both recursively walk a dict/list and strip noise:

```
Before: {"name": "users", "comment": null, "tags": [], "meta": {}}
After:  {"name": "users"}
```

**Why this matters for agents**: Every `null` and `[]` in the response costs tokens but carries zero information. On a 50-row response with 20 columns, removing nulls can save 30-40% of tokens.

DataHub's version also strips `__typename` (GraphQL metadata) and base64 images from descriptions. We don't have those concerns with Snowflake SQL.

### `helpers.py` — Token Budget Enforcement

**DataHub source**: `datahub-agent-context/src/datahub_agent_context/mcp_tools/helpers.py`

#### `select_results_within_budget()` — The Key Pattern

This is the most important helper. It solves: **how do you return as much data as possible without blowing up the LLM's context window?**

```
50 query results
    │
    ▼
select_results_within_budget(results, budget=72000)
    │
    ├── result 1:  2,100 tokens  (total:  2,100) ✓ yield
    ├── result 2:  1,800 tokens  (total:  3,900) ✓ yield
    ├── result 3:  2,300 tokens  (total:  6,200) ✓ yield
    │   ...
    ├── result 11: 1,900 tokens  (total: 71,500) ✓ yield
    ├── result 12: 2,000 tokens  (total: 73,500) ✗ STOP (exceeds 72,000)
    │
    ▼
11 results returned (within budget)
```

**Design decisions**:

1. **Generator, not list** — Memory efficient. Doesn't load all results at once. The caller iterates and the generator stops when budget is hit.

2. **Always yield at least 1** — Even if the first result alone exceeds the budget. Returning nothing is worse than returning something too large. The LLM can at least see partial data.

3. **90% of limit as budget** — The 10% buffer accounts for:
   - Token estimation inaccuracy
   - Response wrapper overhead (the tool output gets wrapped in framework-specific JSON)
   - Better to return slightly fewer results than to exceed the limit

4. **Configurable via env var** — `TOOL_RESPONSE_TOKEN_LIMIT=80000` default. You can tune this for different LLMs without changing code.

#### `truncate_descriptions()` — In-Place String Truncation

Recursively walks a dict and truncates any key named `description` or `comment` to 1000 characters. Applied in-place (mutates the input) to avoid copying large structures.

**Why 1000 chars?** A Snowflake table comment could be an entire documentation page. 1000 chars (~250 tokens) is enough for the LLM to understand the table's purpose without wasting budget on prose.

## DataHub ↔ data-chat Mapping

| DataHub | data-chat | Notes |
|---------|-----------|-------|
| `execute_graphql(graph, query=q, variables=v)` | `execute_query(conn, sql=s, params=p)` | GraphQL → SQL, same pattern |
| `clean_gql_response()` | `clean_response()` | Drops `__typename` (DataHub) vs simpler (us) |
| `_select_results_within_budget()` | `select_results_within_budget()` | Identical logic |
| `truncate_descriptions()` | `truncate_descriptions()` | Identical, we also handle `comment` key |
| `TokenCountEstimator` | `TokenCountEstimator` | Exact copy |
| `TOOL_RESPONSE_TOKEN_LIMIT = 80000` | `TOOL_RESPONSE_TOKEN_LIMIT = 80000` | Same default |
| `sanitize_html_content()` | Not included | Snowflake doesn't return HTML |
| `inject_urls_for_urns()` | Not included | DataHub Cloud-specific |
| `clean_get_entities_response()` | Not included | Schema field merging — DataHub-specific |

## What DataHub Has That We Skipped

| Feature | Why DataHub needs it | Why we don't |
|---------|---------------------|--------------|
| `#[CLOUD]` / `#[NEWER_GMS]` field markers | DataHub Cloud has fields that OSS doesn't | Snowflake schema is consistent |
| GraphQL retry with fallback | Older GMS versions reject unknown fields | SQL doesn't have field versioning |
| `inject_urls_for_urns()` | Converts URNs to clickable Cloud URLs | No URN concept in our tools |
| `clean_get_entities_response()` | Merges `editableSchemaMetadata` into `schemaMetadata`, deduplicates | Snowflake `INFORMATION_SCHEMA` is flat |
| `sanitize_html_content()` with ReDoS protection | DataHub descriptions can contain HTML from ingested sources | Snowflake comments are plain text |
| `_extract_lineage_columns_from_paths()` | Column-level lineage path processing | No lineage concept |
