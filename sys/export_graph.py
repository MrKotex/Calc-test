"""
export_graph.py
===============
Reads memory_index.bin + memory_content.bin from your BinaryScoutAgent
and exports to:

  1. neo4j_import/nodes.csv + neo4j_import/edges.csv  — Neo4j bulk import
  2. neo4j_import/schema.cypher                        — schema + constraints
  3. graph.graphml                                      — Gephi / yEd import
  4. graph_viz.json                                     — standalone browser viewer

Usage:
    python export_graph.py \
        --index  .context-tree/memory_index.bin \
        --content .context-tree/memory_content.bin \
        --out    ./graph-export-output

Requirements: none (stdlib only)
"""

import argparse
import csv
import json
import os
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Node type labels (must match build_binary_memory.py) ──────────────────────
NODE_LABELS = {
    0: "Root",
    1: "File",
    2: "Class",
    3: "Function",
    4: "AsyncFunction",
    # Extended types for SQL/HTML (future-proof)
    5: "Table",
    6: "Column",
    7: "View",
    8: "Schema",
    9: "Database",
    10: "ETLJob",
}

EDGE_LABELS = {
    1: "CONTAINS",
    2: "CALLS",
    3: "IMPORTS",
    4: "REFERENCES",   # FK
    5: "FEEDS",        # ETL/view lineage
}

NODE_COLORS = {
    0: "#6b7280",   # Root       — gray
    1: "#2563eb",   # File       — blue
    2: "#d97706",   # Class      — amber
    3: "#16a34a",   # Function   — green
    4: "#0d9488",   # AsyncFn    — teal
    5: "#F59E0B",   # Table      — amber (updated)
    6: "#FCD34D",   # Column     — light yellow (updated)
    7: "#93C5FD",   # View       — light blue (updated)
    8: "#6EE7B7",   # Schema     — mint (updated)
    9: "#65a30d",   # Database   — lime
    10: "#C4B5FD",  # ETLJob     — lavender (updated)
}


def read_string(f) -> str:
    length = struct.unpack('<I', f.read(4))[0]
    return f.read(length).decode('utf-8', errors='replace')


def read_string_list(f) -> List[str]:
    count = struct.unpack('<I', f.read(4))[0]
    return [read_string(f) for _ in range(count)]


def load_index(index_path: str) -> Tuple[List[Dict], Dict[str, Dict]]:
    nodes = []
    node_map = {}

    with open(index_path, 'rb') as f:
        magic, version, count = struct.unpack('<III', f.read(12))
        if magic != 0x42494E4D:
            raise ValueError(f"Invalid magic number: {hex(magic)} — not a valid memory_index.bin")

        for i in range(count):
            nid      = read_string(f)
            parent   = read_string(f)
            ntype    = struct.unpack('<B', f.read(1))[0]
            
            # ── REFACORED: Generic Edge Array ──
            edges_count = struct.unpack('<I', f.read(4))[0]
            raw_edges = []
            for _ in range(edges_count):
                edge_type = struct.unpack('<B', f.read(1))[0]
                target_len = struct.unpack('<I', f.read(4))[0]
                target_id = f.read(target_len).decode('utf-8')
                raw_edges.append({"type": edge_type, "target": target_id})
            # ───────────────────────────────────

            offset   = struct.unpack('<Q', f.read(8))[0]
            length   = struct.unpack('<I', f.read(4))[0]

            vec_len  = struct.unpack('<I', f.read(4))[0]
            vector   = []
            if vec_len > 0:
                # FIX: Use <f (float32) instead of <d (float64) to match builder
                vector = [struct.unpack('<f', f.read(4))[0] for _ in range(vec_len)]

            node = {
                "id": nid,
                "parent": parent,
                "type": ntype,
                "label": NODE_LABELS.get(ntype, f"Type{ntype}"),
                "raw_edges": raw_edges, # Attach raw edges for simplified build_edges
                "offset": offset,
                "length": length,
                "has_embedding": len(vector) > 0,
                "embedding_dim": len(vector),
                "short_name": nid.split("::")[-1] if "::" in nid else nid,
            }
            nodes.append(node)
            node_map[nid] = node

    return nodes, node_map


def load_content(content_path: str, node_map: Dict[str, Dict]) -> Dict[str, str]:
    content = {}
    with open(content_path, 'rb') as f:
        for nid, node in node_map.items():
            if node["length"] > 0:
                f.seek(node["offset"])
                raw = f.read(node["length"])
                text = raw.decode('utf-8', errors='replace').strip()
                content[nid] = text[:500]  # truncate for export — full text via Neo4j query
    return content


def build_edges(nodes: List[Dict], node_map: Dict) -> List[Dict]:
    edges = []
    edge_id = 0

    for node in nodes:
        nid = node["id"]
        parent = node["parent"]

        # contains edge (from parent)
        if parent and parent in node_map:
            edges.append({
                "id": f"e{edge_id}", "source": parent, "target": nid,
                "type": "CONTAINS", "type_code": 1
            })
            edge_id += 1

        # REFACORED: Iterate over raw generic edges
        for edge in node.get("raw_edges", []):
            target = edge["target"]
            etype_code = edge["type"]
            
            if target in node_map:
                edges.append({
                    "id": f"e{edge_id}", "source": nid, "target": target,
                    "type": EDGE_LABELS.get(etype_code, f"TYPE{etype_code}"),
                    "type_code": etype_code
                })
                edge_id += 1

    return edges


# ── Neo4j CSV export ──────────────────────────────────────────────────────────

def export_neo4j_csv(nodes, edges, content, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    nodes_path = os.path.join(out_dir, "nodes.csv")
    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([
            "nodeId:ID", ":LABEL", "shortName", "fullPath",
            "type:INT", "hasEmbedding:BOOLEAN", "embeddingDim:INT",
            "snippet"
        ])
        for node in nodes:
            snippet = content.get(node["id"], "").replace("\n", " ")[:300]
            writer.writerow([
                node["id"],
                node["label"],
                node["short_name"],
                node["id"],
                node["type"],
                str(node["has_embedding"]).lower(),
                node["embedding_dim"],
                snippet,
            ])
    print(f"  ✓ nodes.csv  ({len(nodes)} nodes)")

    edges_path = os.path.join(out_dir, "edges.csv")
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([":START_ID", ":END_ID", ":TYPE"])
        for edge in edges:
            writer.writerow([edge["source"], edge["target"], edge["type"]])
    print(f"  ✓ edges.csv  ({len(edges)} edges)")

    schema_path = os.path.join(out_dir, "schema.cypher")
    with open(schema_path, "w", encoding="utf-8") as f:
        f.write("""// ── Neo4j Schema & Constraints ─────────────────────────────────────────────
// Run these in Neo4j Browser BEFORE importing data.

// Unique constraint on node ID
CREATE CONSTRAINT node_id_unique IF NOT EXISTS
  FOR (n:File) REQUIRE n.nodeId IS UNIQUE;

// Indexes for fast lookup
CREATE INDEX node_short_name IF NOT EXISTS FOR (n:File) ON (n.shortName);
CREATE INDEX node_type IF NOT EXISTS FOR (n:File) ON (n.type);

// ── Bulk import command (run from terminal, not Neo4j Browser) ────────────────
//
//   neo4j-admin database import full \\
//     --nodes=nodes.csv \\
//     --relationships=edges.csv \\
//     --delimiter="," \\
//     --array-delimiter="|" \\
//     --quote='"' \\
//     neo4j
//
// ── Useful starter queries ────────────────────────────────────────────────────

// All file nodes
// MATCH (n:File) RETURN n LIMIT 25;

// Full graph (caution on large repos — add LIMIT)
// MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 200;

// Find a specific node by name
// MATCH (n) WHERE n.shortName CONTAINS 'Calculator' RETURN n;

// Callers of a function
// MATCH (caller)-[:CALLS]->(fn {shortName: 'divide'}) RETURN caller, fn;

// Everything a file contains
// MATCH (f:File {shortName: 'calculator.py'})-[:CONTAINS*]->(child) RETURN f, child;

// Shortest path between two nodes
// MATCH p = shortestPath(
//   (a {shortName: 'main.py'})-[*]-(b {shortName: 'divide'})
// ) RETURN p;

// Nodes with embeddings
// MATCH (n) WHERE n.hasEmbedding = true RETURN n.fullPath, n.embeddingDim;
""")
    print(f"  ✓ schema.cypher")

    readme_path = os.path.join(out_dir, "IMPORT_README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("""HOW TO IMPORT INTO NEO4J
========================

Option A — Neo4j Desktop (recommended for first-time use)
----------------------------------------------------------
1. Install Neo4j Desktop: https://neo4j.com/download/
2. Create a new database (Neo4j 5.x).
3. Open Neo4j Browser.
4. Run schema.cypher contents first (copy-paste each CREATE statement).
5. Use LOAD CSV for smaller graphs:

   LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
   CALL apoc.create.node([row[':LABEL']], {
     nodeId: row['nodeId:ID'],
     shortName: row['shortName'],
     fullPath: row['fullPath'],
     type: toInteger(row['type:INT']),
     hasEmbedding: row['hasEmbedding:BOOLEAN'] = 'true',
     snippet: row['snippet']
   }) YIELD node RETURN count(node);

   LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
   MATCH (a {nodeId: row[':START_ID']}), (b {nodeId: row[':END_ID']})
   CALL apoc.create.relationship(a, row[':TYPE'], {}, b) YIELD rel
   RETURN count(rel);

Option B — neo4j-admin bulk import (for large graphs, 10k+ nodes)
------------------------------------------------------------------
See the command in schema.cypher.
Requires stopping the database first.

Option C — Gephi
----------------
Open graph.graphml in Gephi.
Layout → ForceAtlas2 or Yifan Hu.
Appearance → Nodes → Color by "type" attribute.
""")
    print(f"  ✓ IMPORT_README.txt")


# ── GraphML export (Gephi / yEd) ─────────────────────────────────────────────

def export_graphml(nodes, edges, content, out_path):
    root = ET.Element("graphml", {
        "xmlns": "http://graphml.graphdrawing.org/graphml",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://graphml.graphdrawing.org/graphml http://graphml.graphdrawing.org/graphml/graphml.xsd"
    })

    def key(id_, for_, name, type_):
        k = ET.SubElement(root, "key", {"id": id_, "for": for_, "attr.name": name, "attr.type": type_})
        return k

    key("label",     "node", "label",     "string")
    key("type",      "node", "type",      "int")
    key("typeName",  "node", "typeName",  "string")
    key("color",     "node", "color",     "string")
    key("snippet",   "node", "snippet",   "string")
    key("hasEmb",    "node", "hasEmb",    "boolean")
    key("edgeType",  "edge", "edgeType",  "string")

    graph_el = ET.SubElement(root, "graph", {"id": "G", "edgedefault": "directed"})

    for node in nodes:
        n = ET.SubElement(graph_el, "node", {"id": node["id"]})
        def d(k, v): ET.SubElement(n, "data", {"key": k}).text = str(v)
        d("label",    node["short_name"])
        d("type",     node["type"])
        d("typeName", node["label"])
        d("color",    NODE_COLORS.get(node["type"], "#6b7280"))
        d("snippet",  content.get(node["id"], "")[:200].replace("\n", " "))
        d("hasEmb",   str(node["has_embedding"]).lower())

    for edge in edges:
        e = ET.SubElement(graph_el, "edge", {
            "id": edge["id"], "source": edge["source"], "target": edge["target"]
        })
        ET.SubElement(e, "data", {"key": "edgeType"}).text = edge["type"]

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="unicode", xml_declaration=True)
    print(f"  ✓ graph.graphml  ({len(nodes)} nodes, {len(edges)} edges)")


# ── JSON export for standalone browser viewer ─────────────────────────────────

def export_json(nodes, edges, content, out_path):
    data = {
        "nodes": [
            {
                "id": n["id"],
                "label": n["short_name"],
                "fullPath": n["id"],
                "type": n["type"],
                "typeName": n["label"],
                "color": NODE_COLORS.get(n["type"], "#6b7280"),
                "snippet": content.get(n["id"], "")[:400],
                "hasEmbedding": n["has_embedding"],
            }
            for n in nodes
        ],
        "edges": [
            {"source": e["source"], "target": e["target"], "type": e["type"]}
            for e in edges
        ],
        "meta": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_types": NODE_LABELS,
            "edge_types": {v: v for v in ["CONTAINS", "CALLS", "IMPORTS", "REFERENCES", "FEEDS"]},
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ graph_viz.json")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export binary memory map to Neo4j / Gephi / browser")
    parser.add_argument("--index",   default=".context-tree/memory_index.bin")
    parser.add_argument("--content", default=".context-tree/memory_content.bin")
    parser.add_argument("--out",     default="./graph-export-output")
    args = parser.parse_args()

    print(f"\nReading binary index: {args.index}")
    nodes, node_map = load_index(args.index)
    print(f"  → {len(nodes)} nodes loaded")

    print(f"Reading content: {args.content}")
    content = load_content(args.content, node_map)
    print(f"  → {len(content)} content blobs loaded")

    print(f"\nBuilding edge list...")
    edges = build_edges(nodes, node_map)
    print(f"  → {len(edges)} edges built")

    out = args.out
    os.makedirs(out, exist_ok=True)

    print(f"\nExporting to: {out}")
    export_neo4j_csv(nodes, edges, content, os.path.join(out, "neo4j_import"))
    export_graphml(nodes, edges, content, os.path.join(out, "graph.graphml"))
    export_json(nodes, edges, content, os.path.join(out, "graph_viz.json"))

    print(f"\nDone. Files written to: {out}/")
    print("  neo4j_import/nodes.csv     → Neo4j nodes")
    print("  neo4j_import/edges.csv     → Neo4j edges")
    print("  neo4j_import/schema.cypher → Schema + starter queries")
    print("  neo4j_import/IMPORT_README.txt")
    print("  graph.graphml              → Gephi / yEd")
    print("  graph_viz.json             → Standalone browser viewer")


if __name__ == "__main__":
    main()