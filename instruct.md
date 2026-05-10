You are an AI engineer working on the repository `Calc-test`.

Goal
====

Refactor the binary memory builder into an AI‑oriented, two‑phase indexing system:

1. Phase 1: **Pure extractor + graph indexer**
   - Parse `.py`, `.html`, and `.sql` files.
   - Extract structured SQL + HTML information and relationships.
   - Build a compact, AI‑oriented binary index (no embeddings, no large models).

2. Phase 2: **Separate AI / embedding pipeline** (stub only in this pass)
   - Provide clear extension points and minimal code for a future script that will:
     - Read the binary index.
     - Reconstruct textual representations of nodes.
     - Build a vector index / embedding store (but do NOT implement heavy model logic now).


Context
=======

Key repo facts:

- Current indexer is `sys/build_binary_memory.py`.
- It currently:
  - Walks `ROOT_DIR` and parses `.py`, `.html`, `.sql`.
  - Builds nodes and edges in memory.
  - Optionally generates embeddings using a Qwen model.
  - Writes `.context-tree/memory_index.bin` and `.context-tree/memory_content.bin`.

Problems with the current design:

- The embedding step loads a large model, consumes a lot of RAM, and can fail before export.
- Nodes are somewhat human‑oriented (`sx` strings) and not optimized as structured features for AI.
- HTML parsing logic has been evolving to handle T‑SQL examples, but needs to be more clearly structured and AI‑friendly.


Target Architecture
===================

You will implement the following architecture:

Phase 1: AI‑oriented binary index (no embeddings)
-------------------------------------------------

1. Keep `sys/build_binary_memory.py` as the **index builder**.
2. Remove any dependency on large embedding models and ensure Phase 1:
   - Does NOT import `transformers` or `sentence_transformers`.
   - Does NOT allocate or store high‑dimensional embedding vectors.
3. For each node, move away from opaque text strings and toward structured fields.

   For SQL objects, store at least:

   - `object_type`: e.g., `"table"`, `"view"`, `"procedure"`, `"function"`, `"column"`.
   - `db_name`: database name if known.
   - `schema_name`: schema name if distinguishable, otherwise empty.
   - `object_name`: e.g. `PROCEDURE_NAME`, `TABLE_PRODUCTS`.
   - For procedures/functions:
     - `params`: structured list of `(name, type, has_default)` or similar.
   - For tables:
     - `columns`: structured list of `(name, type, nullable)`.

   For relationships, store:

   - `reads_from`: node IDs of tables/views this proc/view selects from.
   - `writes_to`: node IDs of tables this proc updates/inserts into (if you can detect).
   - `uses_functions`: node IDs of functions/procs called from within this proc/view.

4. For HTML SQL documentation:

   - Treat each logical SQL example block as its own node.
   - Node ID scheme example:
     - `html:templates/sql_data_example.html#block_1`
   - Fields:
     - `source_file`: e.g. `"templates/sql_data_example.html"`.
     - `block_id`: numeric or anchor‑based.
     - `kind`: e.g. `"example_delete_proc"`, `"example_create_proc"`, `"example_grant_rights"`.
     - `sql_object_id`: link to the canonical SQL object node this example documents (if recognizable).

5. Node ID and type ontology:

   - Keep or refine `NODE_TYPE` as a stable small integer enum, but make sure it covers:
     - `root`, `file`, `table`, `view`, `procedure`, `function`, `column`, `html_block`, etc.
   - Adopt stable, AI‑friendly ID schemes such as:
     - `sql:db.schema.table`
     - `proc:db.schema.proc_name`
     - `col:db.schema.table.col_name`
     - `html:relative/path.html#block_n`

   Ensure IDs are unique and reversible enough to reconstruct file + object from an ID.

6. Binary format:

   - Keep using `memory_index.bin` + `memory_content.bin`, but:
     - Separate **structural metadata** (IDs, types, relationships, small descriptors) from any large text.
     - Store only compact text in `sx` or a similar field (short signature, not full SQL code).
   - You can reuse the existing header and node loop, but adapt what is written:
     - Serialize structured SQL metadata (db, schema, object_name, params, columns, relationships) in a compact binary or length‑prefixed JSON per node.
   - Do *not* write any embedding vectors in this phase.


Phase 2: Embedding / AI consumer stub
-------------------------------------

Create a new script, e.g. `sys/build_embeddings.py`, with the following responsibilities:

- Reads `memory_index.bin` and `memory_content.bin`.
- Provides helper functions to:
  - Iterate over nodes of certain types (e.g. all procedures, all tables).
  - Reconstruct concise textual descriptions for those nodes, such as:
    - `"Procedure DATABASE.PROCEDURE_NAME(@KodTowaru VARCHAR(40), @NazwaCennika VARCHAR(100)=NULL) reading from TABLE_PRODUCTS, TABLE_PRICES_HEADER."`
- Defines a clear extension point function, e.g. `def generate_embeddings(output_path: str) -> None`, which currently:
  - Builds in‑memory Python objects or prints out the would‑be chunks.
  - Contains TODO comments for where a future embedding model would plug in.
- This script must not actually load any large models in this refactor; focus on structure and clarity.


Specific Implementation Tasks
=============================

Task 1: Cleanly disable embeddings in build_binary_memory.py
------------------------------------------------------------

- Remove or comment out any imports from `transformers`, `torch`, or `sentence_transformers`.
- Make `BinaryMemoryBuilder.generate_embeddings` a safe no‑op with a clear log message:
  - It should not allocate large data structures, not iterate over all nodes, and not change state except maybe logging.

- Ensure `build()` does **not** call `generate_embeddings()` by default.

Task 2: Make parse_sql and regex extractor produce structured SQL metadata
--------------------------------------------------------------------------

- In `ParserRegistry.parse_sql` and `_extract_sql_details_regex`, besides building nodes, compute and attach:

  - `db_name`, `schema_name`, `object_name`, `object_type`.
  - For procedures/functions: structured parameter list.
  - For CREATE TABLE: structured column list.
  - For SELECT/UPDATE/INSERT: lists of referenced table names.

- Do not store large SQL text in every node.
  - Instead, store a short normalized signature (e.g. `"PROC DATABASE.PROCEDURE_NAME(@KodTowaru:VARCHAR(40), @NazwaCennika:VARCHAR(100)=NULL)"`) and/or a short snippet.
  - If you need full SQL text later, store it once per logical block (e.g., in an HTML example node) and reference it.

- Normalize object and table names:
  - Strip brackets `[ ]`.
  - Combine into `db.schema.table` when possible.

Task 3: Make parse_html emit block-level SQL example nodes
----------------------------------------------------------

- For HTML like `sql_data_example.html` / `sql_data_example_proc.html`, ensure `parse_html`:

  - Detects each SQL example block (e.g., `<pre>...</pre>` or fallback over full text) and creates a `html_block` node per block.
  - Assigns stable IDs like `html:relative/path.html#block_i`.

- For each block:

  - Store `source_file`, `block_id`, `kind` (inferred from nearby headings like “Usuwanie procedury”, “Tworzenie procedury”, “Przyznanie praw do funkcji”).
  - Try to link it to its SQL object via `sql_object_id` using the procedure name inside the SQL.

- Do not attach huge `ch` lists to row nodes; ensure that table children (`columns`, `rows`, `blocks`) are attached to the appropriate table or file node, not the last row.

Task 4: Adjust export_binary to write structured, AI-friendly metadata
----------------------------------------------------------------------

- Update `export_binary()` so that, for each node:

  - Writes:
    - Node ID (string).
    - Parent ID.
    - Type ID.
    - A compact serialized blob of structured metadata:
      - For SQL objects: db, schema, object_name, cols/params, relationships.
      - For HTML blocks: source file, block_id, kind, sql_object_id.
  - Still writes edge lists (`contains`, `calls`, `feeds`) in a way that remains compatible with your existing graph tooling where feasible.

- Ensure that the content file (`memory_content.bin`) only stores necessary short content, such as:
  - `sx` short signatures.
  - Optional small snippets (e.g. truncated SQL block for display).

Task 5: Add sys/build_embeddings.py stub
----------------------------------------

- Create `sys/build_embeddings.py` with:

  - A small `BinaryIndexReader` class that can:
    - Open `memory_index.bin` and `memory_content.bin`.
    - Iterate over nodes and reconstruct:
      - Node IDs, types, metadata.
      - Short text descriptions suitable for embedding.

  - A `main` section that:
    - Loads the index.
    - Prints a few sample textual representations for:
      - A procedure.
      - A table.
      - A HTML example block.

  - A clearly documented function `generate_embeddings(output_path: str)` that currently just collects the text chunks and writes them to a simple JSON/CSV for inspection. Do NOT call any embedding models in this pass; leave TODO comments indicating where the embedding step would later go.

Constraints
==========

- Preserve current behavior of the graph as much as possible:
  - The root node, file nodes, and basic edges should still be present.
- It is acceptable for the on-disk binary format to change, as long as it is:
  - Clearly documented in comments.
  - Consistent and AI‑oriented (structured fields, stable IDs).
- Prefer small, composable functions over adding complex logic in a single place.

Output Expectations
===================

- Update `sys/build_binary_memory.py` in small, coherent commits or changes:
  - One logical change per section: disable embeddings, restructure SQL metadata, refine HTML parsing, adjust export.
- Add `sys/build_embeddings.py` with:
  - A minimal working CLI:
    - `python -m sys.build_embeddings` or `python sys/build_embeddings.py` should run without errors and print some example reconstructed text chunks.
- Ensure that running:

  - `python sys/build_binary_memory.py` from the repo root:
    - Completes without heavy RAM usage.
    - Produces `.context-tree/memory_index.bin` and `.context-tree/memory_content.bin`.

  - `python sys/build_embeddings.py`:
    - Reads the above files.
    - Shows example AI‑friendly text representations for a few nodes.

When you start, first:
- Read `sys/build_binary_memory.py` and understand the current node and edge structure.
- Then propose a short plan in comments at the top of that file and implement it step by step.
