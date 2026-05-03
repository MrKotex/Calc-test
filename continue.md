<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# continue

Based on what I've now fetched directly from the repo, here is the consolidated compatibility audit for the **new `sys/` binary SQL features**.

***

## What exists in `sys/` right now

| File | Role |
| :-- | :-- |
| `build_binary_memory.py` | Writes `.bin` index + content files |
| `scout_agent_binary.py` | Reads `.bin` files, performs graph traversal + scoring |
| `navigator_agent.py` | Higher-level agent wrapping the scout |
| `build_graph-new.py` | Builds the symbolic JSON graph |
| `export_graph.py` | Exports graph to Neo4j / GraphML / JSON |
| `benchmark_runner.py` | Runs benchmark queries against the agent |
| `benchmark_questions.json` | Test question set |
| `prompts_navigator.md` | Navigator system prompt |
| `prompts_orchestrator.md` | Orchestrator system prompt |
| `scout_agent-qwen-new.py` | **Old** text-based agent (superseded) |


***

## Cross-file compatibility issues \& required fixes

### 1. Binary format contract (`build_binary_memory.py` ↔ `scout_agent_binary.py`)

**Issue:** `build_binary_memory.py` writes SQL table/column nodes using its own string-packing format (`write_string` + `write_string_list`), but `scout_agent_binary.py` `read_node()` only handles node types that existed before SQL was added (`file`, `class`, `function`, `method`). Any `table`, `column`, `schema`, or `view` node read back will fall into the default branch with no type-specific field parsing — resulting in corrupted field offsets for every node after the first SQL node in the file.

**Fix:** Add matching `read` branches in `scout_agent_binary.py` for every new type `build_binary_memory.py` writes:

```python
elif node_type in ("table", "view", "schema"):
    node["db"]       = read_string(f)
    node["columns"]  = read_string_list(f)
    node["snippet"]  = read_string(f)
elif node_type == "column":
    node["table"]    = read_string(f)
    node["dtype"]    = read_string(f)
    node["nullable"] = struct.unpack("?", f.read(1))[0]
```

The field order **must exactly match** what `build_binary_memory.py` writes — verify byte-for-byte.

***

### 2. Edge type vocabulary (`build_binary_memory.py` ↔ `scout_agent_binary.py` ↔ `navigator_agent.py`)

**Issue:** `build_binary_memory.py` writes SQL edges as `"table_column"`, `"foreign_key"`, `"view_base"`, `"etl_reads"`, `"etl_writes"`. The scoring logic in `scout_agent_binary.py` only boosts `"calls"`, `"contains"`, `"imports"` edges. SQL lineage edges receive no traversal weight, so they are effectively invisible to the ranker.

**Fix:** In `scout_agent_binary.py`'s edge weight / traversal function, add:

```python
SQL_LINEAGE_EDGES = {"foreign_key", "view_base", "etl_reads", "etl_writes", "table_column"}
if edge_type in SQL_LINEAGE_EDGES:
    weight = 1.6   # lower than "calls" (2.0) but higher than default (1.0)
```

Also add `"table_lineage"` and `"column_lineage"` as recognised `query_type` values in `navigator_agent.py`.

***

### 3. `export_graph.py` missing SQL node types

**Issue:** `export_graph.py` builds its Neo4j `nodes.csv` and `graph_viz.json` by iterating the graph JSON, but its `NODE_COLORS` and `NODE_LABELS` dicts don't include `"table"`, `"column"`, `"schema"`, `"view"`, or `"etl_job"`. These nodes export as blank/unknown type, breaking the browser explorer's filter panel.

**Fix:** Add to the colour/label maps in `export_graph.py`:

```python
NODE_COLORS = {
    ...existing...,
    "table":   "#F59E0B",
    "column":  "#FCD34D",
    "schema":  "#6EE7B7",
    "view":    "#93C5FD",
    "etl_job": "#C4B5FD",
}
```


***

### 4. `benchmark_runner.py` ↔ `scout_agent_binary.py` interface mismatch

**Issue:** `benchmark_runner.py` calls the agent via subprocess and parses its stdout expecting the old JSON schema `{"top_nodes": [...], "score": ...}`. `scout_agent_binary.py` now outputs `{"results": [...], "query_type": ..., "elapsed_ms": ...}`. The runner will silently score every SQL-related question as a miss (KeyError caught → Hit@1 = 0).

**Fix:** Either update `benchmark_runner.py` to read `response["results"]`, or add a compatibility shim in `scout_agent_binary.py`:

```python
output["top_nodes"] = output["results"]   # backwards-compat key
```


***

### 5. `prompts_navigator.md` scoring formula references non-existent node fields

**Issue:** The prompt documents a scoring formula using `caller_count` as a boost signal. For SQL node types (`table`, `column`, `view`), no `caller_count` field exists in the binary format — those nodes use `reference_count` (number of foreign keys / ETL jobs that reference them). The Navigator agent will produce `KeyError` or silently apply 0 boost for all SQL nodes.

**Fix:** In `navigator_agent.py` normalise the field on read:

```python
node["caller_count"] = node.get("caller_count") or node.get("reference_count", 0)
```

And update `prompts_navigator.md` to document this unified field.

***

### 6. `prompts_orchestrator.md` tool signatures don't match `navigator_agent.py`

**Issue:** `prompts_orchestrator.md` documents `graph_query({type: "column_lineage", table: ..., column: ...})` as a valid call. `navigator_agent.py` only implements `symbol_lookup` and `table_lineage` — `column_lineage` will return `{"error": "unknown query type"}` silently, causing the orchestrator to hallucinate an answer with no grounding.

**Fix:** Either implement `column_lineage` in `navigator_agent.py`:

```python
elif query["type"] == "column_lineage":
    return _column_lineage(query["table"], query["column"], direction=query.get("direction","both"))
```

Or remove it from `prompts_orchestrator.md` until it is implemented. **Do not document capabilities that don't exist.**

***

### 7. Magic number version guard missing for SQL extension

**Issue:** `build_binary_memory.py` writes a 4-byte magic `0x42494E4D` at offset 0, but no version byte follows it. When SQL fields were added, the binary layout changed. Any `.bin` file built with the old layout will silently produce wrong results when read by the new `scout_agent_binary.py` — no error, just garbage node data.

**Fix:** Add a 2-byte version field immediately after the magic:

```python
# Writer (build_binary_memory.py)
f.write(b'\x42\x49\x4E\x4D')  # magic
f.write(struct.pack(">H", 2))   # format version 2 = includes SQL types

# Reader (scout_agent_binary.py)
magic = f.read(4)
version = struct.unpack(">H", f.read(2))[0]
if version < 2:
    raise RuntimeError("Binary memory file is v1 (pre-SQL). Re-run build_binary_memory.py.")
```


***

## Summary priority order

| \# | Severity | Fix needed in |
| :-- | :-- | :-- |
| 1 | 🔴 **Critical** | `scout_agent_binary.py` — add SQL node read branches |
| 7 | 🔴 **Critical** | Both — add version guard to binary format |
| 4 | 🔴 **Critical** | `benchmark_runner.py` — fix output key mismatch |
| 2 | 🟠 **High** | `scout_agent_binary.py` — add SQL edge weights |
| 6 | 🟠 **High** | `navigator_agent.py` — implement `column_lineage` or remove from prompt |
| 5 | 🟡 **Medium** | `navigator_agent.py` — normalise `caller_count` / `reference_count` |
| 3 | 🟡 **Medium** | `export_graph.py` — add SQL node colours/labels |

Fix issues 1, 7, and 4 first — they are silent data-corruption bugs that will make everything else appear to work while producing wrong results. Issues 2, 5, 6, and 3 can follow in a single PR after.

