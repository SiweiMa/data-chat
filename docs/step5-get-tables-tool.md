# Step 5: Second Tool — `get_tables()`

## The Problem

After `search()` returns a list of table names, the LLM knows *what* exists but not *what's inside*. It needs column names, types, and comments to write correct SQL. Without a schema inspection tool, the LLM would have to guess column names — or worse, run `SELECT *` and infer the schema from data.

`get_tables()` is the bridge between discovery and querying: search finds the tables, get_tables reveals their structure, and then the LLM can write precise SQL.

## How DataHub Does It

**Source**: `datahub-agent-context/src/datahub_agent_context/mcp_tools/entities.py` (get_entities)

DataHub's `get_entities()` takes a list of URNs (Uniform Resource Names — DataHub's unique identifiers) and returns detailed metadata for each entity. It supports:
- Batch lookup (multiple URNs in a single call)
- Per-entity error handling (one bad URN doesn't crash the batch)
- Schema field listing with per-entity token budget
- Truncation flag (`schemaFieldsTruncated`) to signal incomplete column lists
- Response flattening (deeply nested GraphQL → flat LLM-friendly dicts)

## How data-chat Does It

**Source**: `src/data_chat/tools/tables.py`

Our `get_tables()` takes a list of table names (fully qualified, schema-qualified, or unqualified) and returns column details from `INFORMATION_SCHEMA`. It supports:
- Batch lookup (multiple table names in one call)
- Per-table error handling (one missing table doesn't fail the batch)
- Flexible name resolution (`DB.SCHEMA.TABLE`, `SCHEMA.TABLE`, or `TABLE`)
- Per-table token budget (16K default) to prevent wide tables from eating the entire response
- Truncation flag (`columns_truncated`) to tell the LLM "there are more columns"

## Key Design Patterns

### Pattern 1: Batch In, Batch Out (Reduce LLM Round-Trips)

```python
# DataHub — list of URNs in:
get_entities(urns=["urn:li:dataset:1", "urn:li:dataset:2", "urn:li:dataset:3"])

# Ours — list of table names in:
get_tables(tables=["DB.SCHEMA.CUSTOMERS", "DB.SCHEMA.ORDERS", "DB.SCHEMA.PRODUCTS"])
```

Both return a list of results — one per input. This is critical for agent efficiency. Without batching:

```
LLM: get_table("CUSTOMERS")    → round-trip 1
LLM: get_table("ORDERS")       → round-trip 2
LLM: get_table("PRODUCTS")     → round-trip 3
```

Each round-trip is an LLM inference call (~1-5 seconds) + a tool execution. With batching:

```
LLM: get_tables(["CUSTOMERS", "ORDERS", "PRODUCTS"])  → round-trip 1
```

Three tables in one call. The docstring explicitly tells the LLM to do this:

```python
"""IMPORTANT: Pass multiple table names in a single call — this is much more
efficient than calling this tool multiple times. When examining search
results, always pass the top 3-10 table names together to compare."""
```

Without this instruction, the LLM will often make one call per table — it's the "obvious" thing to do. The docstring overrides that instinct.

### Pattern 2: Per-Item Error Handling

```python
# DataHub (entities.py ~line 120):
for urn in urns:
    try:
        result = fetch_entity(graph, urn)
        results.append(result)
    except Exception as e:
        results.append({"error": str(e), "urn": urn})

# Ours (tables.py line 87-95):
for table_ref in tables:
    try:
        result = _get_single_table(conn, table_ref, database, schema)
        results.append(result)
    except Exception as e:
        logger.warning(f"Error fetching table {table_ref}: {e}")
        results.append({"error": str(e), "table": table_ref})
```

If the LLM calls `get_tables(["CUSTOMERS", "TYPO_TABLE", "ORDERS"])`, only `TYPO_TABLE` fails. The response contains:

```json
[
  {"database": "DB", "schema": "PUBLIC", "name": "CUSTOMERS", "columns": [...]},
  {"error": "Table TYPO_TABLE not found", "table": "TYPO_TABLE"},
  {"database": "DB", "schema": "PUBLIC", "name": "ORDERS", "columns": [...]}
]
```

The LLM gets useful data for 2 out of 3 tables and can self-correct the typo. Without per-item error handling, one typo would throw an exception and return nothing.

### Pattern 3: The Two-Level Token Budget

This is the most important pattern in this tool. It's the reason `helpers.py` has both `select_results_within_budget()` (total) and `select_columns_within_budget()` (per-table).

**The problem**: A table with 500 columns costs ~40K tokens. If you request 3 such tables, the response is 120K tokens — exceeding any LLM context window.

**DataHub's solution**: Per-entity token budget. Each entity's schema fields are independently capped:

```
Total response budget: 80K tokens (TOOL_RESPONSE_TOKEN_LIMIT)
    │
    └── Entity 1 budget: 16K tokens (ENTITY_SCHEMA_TOKEN_BUDGET)
    │       → 100 columns shown, 400 truncated
    │
    └── Entity 2 budget: 16K tokens
    │       → all 10 columns shown
    │
    └── Entity 3 budget: 16K tokens
            → 120 columns shown, 380 truncated
```

**Our implementation** does the same:

```python
# tables.py line 208:
budget_result = select_columns_within_budget(cleaned_columns)
```

Which calls `helpers.py`'s `select_columns_within_budget()`:

```
cleaned_columns (500 columns)
    │
    ▼
select_columns_within_budget(columns, token_budget=16000)
    │
    ├── col 1:  80 tokens  (total:     80) ✓ include
    ├── col 2:  65 tokens  (total:    145) ✓ include
    │   ...
    ├── col 98: 90 tokens  (total: 15,900) ✓ include
    ├── col 99: 85 tokens  (total: 15,985) ✓ include
    ├── col 100: 95 tokens (total: 16,080) ✗ exceeds 16,000 → STOP
    │
    └── return {
            "columns": [...98 cols...],
            "total_columns": 500,
            "columns_truncated": True   ← critical signal
        }
```

Without the per-table budget, requesting `get_tables(["WIDE_TABLE", "NARROW_TABLE"])`:
- `WIDE_TABLE` uses 40K tokens for 500 columns
- `NARROW_TABLE` gets nothing (total budget exhausted)

With the per-table budget:
- `WIDE_TABLE` uses 16K tokens, shows 100 columns + truncation flag
- `NARROW_TABLE` uses 2K tokens, shows all 10 columns

### Pattern 4: The Truncation Flag as a Communication Channel

```python
# DataHub (entities.py ~line 330):
if schema_fields_truncated:
    result["schemaFieldsTruncated"] = True

# Ours (tables.py line 217-218):
if budget_result.get("columns_truncated"):
    table_info["columns_truncated"] = True
```

The truncation flag is *not just metadata* — it's a **prompt for the LLM to take action**. The docstring teaches the LLM what to do when it sees this flag:

```python
"""WHEN COLUMNS ARE TRUNCATED:
If you see columns_truncated=True in the response, it means the table has
more columns than could fit in the token budget. To find specific columns:
- Call get_tables() again with just that one table (gets full per-table budget)
- Or use search(query="email", search_columns=True) to find columns by name"""
```

This creates a feedback loop:
1. LLM calls `get_tables(["WIDE_TABLE", "NARROW_TABLE"])`
2. Response includes `columns_truncated: true` for WIDE_TABLE
3. LLM reads the flag, calls `get_tables(["WIDE_TABLE"])` alone (full budget for 1 table)
4. Or uses `search(query="email", search_columns=True)` to find specific columns

Without the flag, the LLM would assume the truncated column list is complete and write incorrect SQL referencing only the visible columns.

### Pattern 5: Flexible Name Resolution

DataHub uses URNs — globally unique identifiers like `urn:li:dataset:(urn:li:dataPlatform:snowflake,mydb.myschema.mytable,PROD)`. There's no ambiguity.

Snowflake table names are contextual — `CUSTOMERS` could exist in many databases and schemas. Our `_parse_table_ref()` handles this:

```python
# Fully qualified — unambiguous
_parse_table_ref("MY_DB.PUBLIC.CUSTOMERS", None, None)
→ ("MY_DB", "PUBLIC", "CUSTOMERS")

# Schema-qualified — uses connection's database
_parse_table_ref("PUBLIC.CUSTOMERS", "MY_DB", None)
→ ("MY_DB", "PUBLIC", "CUSTOMERS")

# Unqualified — uses provided defaults or connection defaults
_parse_table_ref("CUSTOMERS", "MY_DB", "PUBLIC")
→ ("MY_DB", "PUBLIC", "CUSTOMERS")
```

This matters because `search()` returns fully qualified names (`database`, `schema`, `name` as separate fields). The LLM can pass them through in any format:
- Copy-paste the FQN from search results: `"MY_DB.PUBLIC.CUSTOMERS"`
- Use just the table name when the database/schema is obvious: `"CUSTOMERS"`

### Pattern 6: Separate the Loop from the Logic

```python
# The public function handles batching + error handling:
def get_tables(tables, database, schema):
    for table_ref in tables:
        try:
            result = _get_single_table(conn, table_ref, database, schema)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "table": table_ref})

# The private function handles single-table logic:
def _get_single_table(conn, table_ref, default_database, default_schema):
    # parse name → query metadata → query columns → budget → return
```

DataHub follows the same split. This separation matters because:
1. **Error boundaries are clean** — the try/except in the loop catches everything `_get_single_table` might throw
2. **Testability** — you can unit-test `_get_single_table` with a single table without the batch scaffolding
3. **Readability** — the loop is trivial to understand; the per-table logic is self-contained

## Architecture Diagram

```
LLM: "What columns does CUSTOMERS have?"
 │
 ▼
LangChain: get_tables(tables=["DB.SCHEMA.CUSTOMERS", "DB.SCHEMA.ORDERS"])
 │
 ├─ create_context_wrapper injects SnowflakeClient
 │
 ▼
get_tables() — the batch loop
 │
 ├── for each table_ref in tables:
 │       │
 │       ▼
 │   _get_single_table(conn, table_ref, database, schema)
 │       │
 │       ├── _parse_table_ref("DB.SCHEMA.CUSTOMERS")
 │       │       → (DB, SCHEMA, CUSTOMERS)
 │       │
 │       ├── Query 1: Table metadata
 │       │   SELECT TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME, ...
 │       │   FROM INFORMATION_SCHEMA.TABLES
 │       │   WHERE TABLE_NAME = 'CUSTOMERS'
 │       │     AND TABLE_CATALOG = 'DB'
 │       │     AND TABLE_SCHEMA = 'SCHEMA'
 │       │
 │       ├── clean_response() + truncate_descriptions()
 │       │
 │       ├── Query 2: Column details
 │       │   SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, ...
 │       │   FROM INFORMATION_SCHEMA.COLUMNS
 │       │   WHERE TABLE_NAME = 'CUSTOMERS' ...
 │       │   ORDER BY ORDINAL_POSITION
 │       │
 │       ├── clean_response() + truncate_descriptions() for each column
 │       │
 │       ├── select_columns_within_budget()    ← per-table 16K cap
 │       │       │
 │       │       ├── Walk columns, summing estimated tokens
 │       │       ├── Stop when budget exceeded (always include ≥1)
 │       │       └── Set columns_truncated=True if stopped early
 │       │
 │       └── Merge: table_info + columns + truncation flag
 │
 └── return [result_1, result_2, ...]
```

## The Response Shape

```json
[
  {
    "database": "MY_DB",
    "schema": "PUBLIC",
    "name": "CUSTOMERS",
    "type": "BASE TABLE",
    "comment": "Core customer records",
    "row_count": 1500000,
    "created": "2024-01-15T10:30:00",
    "last_altered": "2024-06-01T14:22:00",
    "columns": [
      {"name": "CUSTOMER_ID", "type": "NUMBER", "nullable": "NO", "position": 1},
      {"name": "EMAIL", "type": "VARCHAR", "nullable": "NO", "position": 2},
      {"name": "CREATED_AT", "type": "TIMESTAMP_NTZ", "nullable": "YES", "position": 3}
    ],
    "total_columns": 3
  },
  {
    "error": "Table TYPO_TABLE not found",
    "table": "TYPO_TABLE"
  }
]
```

Notice:
- `columns_truncated` is **absent** when all columns fit — not `false`, absent. This saves tokens and avoids confusing the LLM with unnecessary flags.
- Error entries are inline with success entries — the LLM sees them in order and can self-correct.
- `clean_response()` has already stripped any null values, so you won't see `"comment": null` or `"default_value": null`.

## What DataHub Has That We Don't (Yet)

### Entity Type Diversity

DataHub's `get_entities()` handles datasets, dashboards, charts, data products, users, and more — each with different metadata shapes. It uses a generalized approach:

```python
# DataHub dispatches by entity type:
if entity_type == "dataset":
    process_dataset_fields(result)
elif entity_type == "dashboard":
    process_dashboard_fields(result)
```

We only handle Snowflake tables/views. If we add support for Snowflake stages, streams, or tasks later, we'd need a similar dispatch pattern.

### Response Flattening (helpers.py lines 366-466)

DataHub's GraphQL responses are deeply nested:

```json
{
  "schemaMetadata": {
    "fields": [
      {
        "fieldPath": "user.address.city",
        "nativeDataType": "varchar",
        "type": {"type": {"com.linkedin.schema.StringType": {}}}
      }
    ]
  }
}
```

DataHub flattens this to:

```json
{
  "fields": [
    {"name": "user.address.city", "type": "varchar"}
  ]
}
```

Our Snowflake responses from `INFORMATION_SCHEMA.COLUMNS` are already flat — no flattening needed.

### Schema Field Merging

DataHub merges `schemaMetadata` (system-generated) with `editableSchemaMetadata` (user-curated descriptions) into a single field list. This handles the case where a data steward adds a description to a column that the ingestion pipeline doesn't know about.

In Snowflake, `INFORMATION_SCHEMA.COLUMNS` already has the `COMMENT` field (which is the user-curated description), so no merging is needed.

### Lineage Information

DataHub can include upstream/downstream lineage in entity responses — "this table is built from these sources." Snowflake has lineage via `ACCESS_HISTORY` and `OBJECT_DEPENDENCIES`, but we don't surface it yet. This would be a separate tool.

## DataHub ↔ data-chat Mapping

| DataHub entities.py | data-chat tables.py | Notes |
|---|---|---|
| `get_entities(urns=[...])` | `get_tables(tables=[...])` | Both batch: list in, list out |
| URN-based lookup | Name-based lookup with `_parse_table_ref()` | DataHub uses globally unique URNs |
| `get_graph()` | `get_connection()` | Both from contextvars |
| Per-URN try/except | Per-table try/except | Same error boundary pattern |
| `ENTITY_SCHEMA_TOKEN_BUDGET = 16000` | `PER_TABLE_TOKEN_BUDGET = 16000` | Same per-entity budget |
| `schemaFieldsTruncated = True` | `columns_truncated = True` | Same truncation signal |
| `clean_get_entities_response()` schema loop | `select_columns_within_budget()` | Same budget enforcement, different structure |
| `execute_graphql(graph, query=...)` | `execute_query(conn, sql=...)` | GraphQL → SQL |
| Handles datasets, dashboards, charts, etc. | Handles tables and views only | DataHub is multi-entity |
| Merges `schemaMetadata` + `editableSchemaMetadata` | Not needed | Snowflake `COMMENT` is already unified |
| `inject_urls_for_urns()` | Not applicable | DataHub Cloud link injection |
| 51-line docstring | 51-line docstring | Both serve as LLM instruction manuals |

## How search() and get_tables() Work Together

The two tools form a discovery → inspection pipeline. The docstrings explicitly teach the LLM this workflow:

```
Step 1: search("customer")
        │
        └──→ { "results": [
                 {"database": "DB", "schema": "PUBLIC", "name": "CUSTOMERS"},
                 {"database": "DB", "schema": "PUBLIC", "name": "CUSTOMER_ORDERS"},
                 {"database": "DB", "schema": "ANALYTICS", "name": "CUSTOMER_SEGMENTS"}
               ]}

Step 2: get_tables(["DB.PUBLIC.CUSTOMERS", "DB.PUBLIC.CUSTOMER_ORDERS"])
        │
        └──→ [
               {"name": "CUSTOMERS", "columns": [
                 {"name": "CUSTOMER_ID", "type": "NUMBER"},
                 {"name": "EMAIL", "type": "VARCHAR"}, ...]},
               {"name": "CUSTOMER_ORDERS", "columns": [
                 {"name": "ORDER_ID", "type": "NUMBER"},
                 {"name": "CUSTOMER_ID", "type": "NUMBER"}, ...]}
             ]

Step 3: LLM writes SQL using the discovered schema
        SELECT c.EMAIL, COUNT(o.ORDER_ID)
        FROM DB.PUBLIC.CUSTOMERS c
        JOIN DB.PUBLIC.CUSTOMER_ORDERS o ON c.CUSTOMER_ID = o.CUSTOMER_ID
        GROUP BY c.EMAIL
```

Both docstrings include this workflow in their `TYPICAL WORKFLOW:` section. This is deliberate — repeating the pattern in both tools reinforces the behavior. The LLM sees the same workflow regardless of which tool description it reads first.
