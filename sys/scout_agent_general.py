"""
scout_agent_general.py
~~~~~~~~~~~~~~~~~~~~~~
Generalised binary-memory retrieval agent.
No repo-specific scoring rules — all signals come from graph structure
and token overlap between the query and node metadata.

Usage:
    python scout_agent_general.py --query "Where is reset defined?"
    python scout_agent_general.py --query "table lineage orders" --top 5
"""
from __future__ import annotations

import argparse
import os
import re
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

INDEX_FILE   = os.path.join(".context-tree", "memory_index.bin")
CONTENT_FILE = os.path.join(".context-tree", "memory_content.bin")
MAGIC_NUMBER = 0x42494E4D

# Words that carry no retrieval signal
QUERY_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "of", "for", "by", "with", "from", "that",
    "this", "it", "its", "where", "what", "how", "which", "who", "does",
    "do", "did", "has", "have", "had", "can", "could", "will", "would",
    "should", "and", "or", "not", "no", "get", "set", "use", "used",
    "define", "defined", "call", "called",
}

# Node-type codes that are system/tooling — never surface as results
META_PATTERNS = [
    "test_", "_test.py", "/tests/", "benchmark",
    "runner", "build_graph", "scout_agent", "inspect_graph",
    "build_binary", "export_graph", "__init__",
]


# ---------------------------------------------------------------------------
# Binary reader
# ---------------------------------------------------------------------------

def _read_str(f) -> str:
    length = struct.unpack("<I", f.read(4))[0]
    return f.read(length).decode("utf-8", errors="replace")


def load_graph(index_path: str, content_path: str) -> Tuple[List[Dict], Dict[str, int]]:
    """Read memory_index.bin and return (nodes, node_idx)."""
    nodes:    List[Dict]        = []
    node_idx: Dict[str, int]    = {}

    content = Path(content_path).read_bytes() if Path(content_path).exists() else b""

    with open(index_path, "rb") as f:
        magic, version, count = struct.unpack("<III", f.read(12))
        if magic != MAGIC_NUMBER:
            raise ValueError(f"Bad magic number: 0x{magic:08X}")

        for _ in range(count):
            nid    = _read_str(f)
            pid    = _read_str(f)
            ntype  = struct.unpack("<B", f.read(1))[0]

            edge_count = struct.unpack("<I", f.read(4))[0]
            edges: List[Tuple[int, str]] = []
            for _ in range(edge_count):
                etype = struct.unpack("<B", f.read(1))[0]
                tid   = _read_str(f)
                edges.append((etype, tid))

            offset = struct.unpack("<Q", f.read(8))[0]
            length = struct.unpack("<I", f.read(4))[0]

            vec_len = struct.unpack("<I", f.read(4))[0]
            vec: List[float] = []
            for _ in range(vec_len):
                vec.append(struct.unpack("<f", f.read(4))[0])

            sx = content[offset: offset + length].decode("utf-8", errors="replace")

            node: Dict = {
                "id":    nid,
                "p":     pid,
                "t":     ntype,
                "edges": edges,
                "sx":    sx,
                "vec":   vec,
            }
            node_idx[nid] = len(nodes)
            nodes.append(node)

    return nodes, node_idx


# ---------------------------------------------------------------------------
# Query decomposition
# ---------------------------------------------------------------------------

def decompose_query(query_text: str) -> Tuple[str, List[str]]:
    """
    Split query into (primary_identifier, context_tokens).
    Primary identifier = the last meaningful token (most specific term).
    Context tokens     = the rest (give domain/file context).
    """
    raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query_text)
    useful = [t.lower() for t in raw_tokens if t.lower() not in QUERY_STOPWORDS]
    if not useful:
        return "", []
    return useful[-1], useful[:-1]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

TYPE_SCORE = {
    3: 1.5,   # function
    4: 1.5,   # async_function
    2: 1.3,   # class
    5: 1.4,   # table
    6: 1.2,   # column
    7: 1.3,   # view
    1: 1.0,   # file
    0: 0.1,   # root
}


def _tokenise(text: str) -> set:
    return set(re.split(r"[^a-z0-9]+", text.lower())) - {""}


def name_score(node: Dict, ident: str) -> float:
    nid  = node["id"].lower()
    base = nid.rsplit("::", 1)[-1].rsplit(".", 1)[-1]  # last segment after :: or .
    if base == ident:
        return 8.0
    if f"::{ident}" in nid:
        return 6.5
    if ident in base:
        return 1.8
    return 0.5


def context_score(node: Dict, context_toks: List[str]) -> float:
    if not context_toks:
        return 1.0
    nid_toks    = _tokenise(node["id"])
    parent_toks = _tokenise(node.get("p", ""))
    all_toks    = nid_toks | parent_toks
    matches     = sum(1 for ct in context_toks if ct in all_toks)
    total       = len(context_toks)
    if matches == total:
        return 2.8
    if matches > 0:
        return 1.0 + (matches / total) * 1.5
    return 0.4


def file_stem_score(node: Dict, ident: str) -> float:
    stem = re.sub(r"\.py(:.*)?$", "", node["id"]).lstrip("./")
    stem = re.split(r"[/\\]", stem)[-1].split("::")[0]
    if stem == ident:
        return 2.0
    if ident in stem:
        return 1.4
    return 1.0


def depth_penalty(node: Dict) -> float:
    # Prefer mid-depth nodes (functions inside classes/files) over root
    sx = node.get("sx", "")
    m  = re.search(r"\|D:(\d+)", sx)
    d  = int(m.group(1)) if m else 1
    return 1.0 if d >= 1 else 0.3


def caller_boost(node: Dict, node_idx: Dict, nodes: List[Dict]) -> float:
    """Slightly boost nodes that are called by others (higher connectivity)."""
    nid = node["id"]
    callers = [
        1 for n in nodes
        if any(e == 2 and t == nid for e, t in n.get("edges", []))
    ]
    return 1.0 + min(len(callers) * 0.15, 0.6)


def is_meta(node: Dict) -> bool:
    nid = node["id"].lower()
    return any(p in nid for p in META_PATTERNS)


def score_node(
    node: Dict,
    ident: str,
    context_toks: List[str],
    nodes: List[Dict],
    node_idx: Dict,
) -> float:
    if is_meta(node):
        return -1.0
    s  = name_score(node, ident)
    s *= context_score(node, context_toks)
    s *= file_stem_score(node, ident)
    s *= TYPE_SCORE.get(node["t"], 1.0)
    s *= depth_penalty(node)
    s *= caller_boost(node, node_idx, nodes)
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    query_text: str,
    top_k: int = 5,
    index_path: str = INDEX_FILE,
    content_path: str = CONTENT_FILE,
) -> List[Dict]:
    """Return top_k ranked nodes for query_text."""
    nodes, node_idx = load_graph(index_path, content_path)
    ident, context_toks = decompose_query(query_text)

    scored = [
        (score_node(n, ident, context_toks, nodes, node_idx), n)
        for n in nodes
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": round(s, 3), **n} for s, n in scored[:top_k] if s > 0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generalised binary-memory retrieval")
    ap.add_argument("--query",   required=True, help="Natural-language query")
    ap.add_argument("--top",     type=int, default=5, help="Number of results")
    ap.add_argument("--index",   default=INDEX_FILE)
    ap.add_argument("--content", default=CONTENT_FILE)
    args = ap.parse_args()

    if not Path(args.index).exists():
        print(f"[scout] Index not found: {args.index}")
        print("[scout] Run: python sys/build_binary_memory.py first.")
        raise SystemExit(1)

    results = retrieve(args.query, top_k=args.top,
                       index_path=args.index, content_path=args.content)

    print(f"\nQuery: {args.query!r}")
    print(f"Top {len(results)} results:\n")
    for r in results:
        print(f"  [{r['score']:6.2f}]  {r['id']}")
        print(f"           sx: {r['sx'][:80]}")
        print()
