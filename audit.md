Context:
I have a custom binary graph memory system comprising build_binary_memory.py (writer), scout_agent_binary.py (reader/agent), export_graph.py (exporter), and navigator_agent.py (orchestrator).

The Problem:

    Byte Mismatch: Embeddings are packed as 32-bit floats (<f) in the builder, but export_graph.py tries to unpack them as 64-bit doubles (<d), causing fatal byte-alignment errors.

    Unresolved Edges: build_binary_memory.py leaves raw string names in node["cl"] instead of resolving them to Node IDs, which breaks the graph.

    Hardcoded Binary Format: The binary header strictly expects arrays for imports, calls, and called_by. I recently added SQL parsing for references and feeds edge types, but they are completely ignored by the binary writer/reader.

The Goal: The Scalable Refactor
I want to change the binary format from hardcoded relationship arrays to a single, generic edge array. Please refactor the code according to these rules:

1. build_binary_memory.py

    Fix Call Edges: Update resolve_call_edges() so that it replaces the raw string names in node["cl"] with the actual, resolved target Node IDs.

    Refactor export_binary(): Instead of writing imports, calls, and called_by counts and strings, group self.edges by their source node. Write an edges_count (unsigned int), and then loop through that node's edges, writing the edge_type (1 byte, unsigned char), followed by the target_id string (length + string bytes).

2. scout_agent_binary.py

    Refactor _load_index(): Remove the old reads for imports, calls, and called_by. Instead, read the new edges_count. Loop through the edges, unpacking the 1-byte edge_type and the target_id string.

    Preserve Compatibility: To ensure navigator_agent.py doesn't break, dynamically map these parsed edges back into lists inside node_data (e.g., if edge_type == EDGE_TYPE["calls"], append to node_data["calls"]; if edge_type == EDGE_TYPE["feeds"], append to node_data["feeds"]).

3. export_graph.py

    Fix Byte Mismatch: In load_index(), change the embedding unpacker from <d to <f to read 32-bit floats.

    Refactor load_index(): Update the binary reading logic to match the new generic edge format (read edges_count, then edge_type and target_id). Attach this raw edge list to the returned node dictionary.

    Refactor build_edges(): Simplify this dramatically. Instead of reconstructing edges from strings, simply iterate over the node's parsed generic edges and format them using the EDGE_LABELS mapping.

Please output the fully refactored Python code for these files.
