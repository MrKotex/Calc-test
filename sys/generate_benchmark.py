"""
generate_benchmark.py
~~~~~~~~~~~~~~~~~~~~~
Auto-generates benchmark_questions.json from a binary memory graph.
No hand-written questions needed — samples real node IDs as gold answers.

Usage:
    python sys/generate_benchmark.py \\
        --index  .context-tree/memory_index.bin \\
        --content .context-tree/memory_content.bin \\
        --out    sys/benchmark_questions.json \\
        --limit  40
"""
from __future__ import annotations

import argparse
import json
import random
import re
import struct
from pathlib import Path
from typing import Dict, List, Tuple

MAGIC_NUMBER = 0x42494E4D

NODE_TYPE_NAMES = {
    0: "root", 1: "file", 2: "class", 3: "function",
    4: "async_function", 5: "table", 6: "column", 7: "view",
    8: "schema", 9: "database",
}

# Question templates per node type
QUESTION_TEMPLATES: Dict[int, List[str]] = {
    3: [
        "Where is {name} defined?",
        "What does the function {name} do?",
        "Find the implementation of {name}",
    ],
    4: [
        "Where is async function {name} defined?",
        "Find the async implementation of {name}",
    ],
    2: [
        "Where is class {name} defined?",
        "What methods does {name} have?",
    ],
    5: [
        "What columns does table {name} have?",
        "Describe the schema of {name}",
        "What is the structure of table {name}?",
    ],
    6: [
        "What table contains the {name} column?",
        "Find the column {name}",
    ],
    7: [
        "What tables feed view {name}?",
        "Describe view {name}",
    ],
    1: [
        "What does the file {name} contain?",
        "What is the purpose of {name}?",
    ],
}


def _read_str(f) -> str:
    length = struct.unpack("<I", f.read(4))[0]
    return f.read(length).decode("utf-8", errors="replace")


def load_nodes(index_path: str, content_path: str) -> List[Dict]:
    nodes: List[Dict] = []
    content = Path(content_path).read_bytes() if Path(content_path).exists() else b""
    with open(index_path, "rb") as f:
        magic, _ver, count = struct.unpack("<III", f.read(12))
        if magic != MAGIC_NUMBER:
            raise ValueError(f"Bad magic: 0x{magic:08X}")
        for _ in range(count):
            nid   = _read_str(f)
            pid   = _read_str(f)
            ntype = struct.unpack("<B", f.read(1))[0]
            ec    = struct.unpack("<I", f.read(4))[0]
            for _ in range(ec):
                f.read(1)
                _read_str(f)
            offset = struct.unpack("<Q", f.read(8))[0]
            length = struct.unpack("<I", f.read(4))[0]
            vl = struct.unpack("<I", f.read(4))[0]
            f.read(vl * 4)
            sx = content[offset: offset + length].decode("utf-8", errors="replace")
            m  = re.search(r"N:([^|]+)", sx)
            name = m.group(1).strip() if m else nid.rsplit("::", 1)[-1]
            nodes.append({"id": nid, "t": ntype, "n": name, "p": pid})
    return nodes


def generate(
    nodes: List[Dict],
    limit: int = 40,
    seed: int = 42,
) -> List[Dict]:
    random.seed(seed)
    # Exclude root, meta/system nodes
    META = {
        "test_", "_test", "benchmark", "runner", "build_graph",
        "scout_agent", "inspect_graph", "build_binary", "export_graph",
    }
    candidates = [
        n for n in nodes
        if n["t"] in QUESTION_TEMPLATES
        and not any(p in n["id"].lower() for p in META)
        and n["n"] not in {"-", "", "root", "unnamed"}
    ]

    if len(candidates) > limit:
        candidates = random.sample(candidates, limit)

    questions: List[Dict] = []
    for node in candidates:
        templates = QUESTION_TEMPLATES.get(node["t"], ["Find {name}"])
        q_text = random.choice(templates).format(name=node["n"])
        questions.append({
            "id":   f"q{len(questions) + 1:03d}",
            "q":    q_text,
            "gold": [node["id"]],
            "type": NODE_TYPE_NAMES.get(node["t"], "unknown"),
        })

    return questions


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index",   default=".context-tree/memory_index.bin")
    ap.add_argument("--content", default=".context-tree/memory_content.bin")
    ap.add_argument("--out",     default="sys/benchmark_questions.json")
    ap.add_argument("--limit",   type=int, default=40)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    if not Path(args.index).exists():
        print(f"[generate_benchmark] Index not found: {args.index}")
        print("[generate_benchmark] Run build_binary_memory.py first.")
        raise SystemExit(1)

    nodes     = load_nodes(args.index, args.content)
    questions = generate(nodes, limit=args.limit, seed=args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(questions, fh, indent=2, ensure_ascii=False)

    print(f"[generate_benchmark] Written {len(questions)} questions to {args.out}")
