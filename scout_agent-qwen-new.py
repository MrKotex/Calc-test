import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


GRAPH_PATH = ".context-tree/code_graph_ai.json"
RUN_RESULT_PATH = ".context-tree/run_result.json"

EDGE_TYPE = {
    "contains": 1,
    "calls": 2,
    "imports": 3,
}

NODE_TYPE = {
    "root": 0,
    "file": 1,
    "class": 2,
    "function": 3,
    "async_function": 4,
}


def stable_seed(seed: int = 42):
    np.random.seed(seed)


def compact_tokens(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    parts = re.split(r"[^a-z0-9_./:+*-]+", text)
    return [p for p in parts if p]


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def cosine_from_token_sets(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    vocab = sorted(set(a) | set(b))
    idx = {t: i for i, t in enumerate(vocab)}
    va = np.zeros(len(vocab), dtype=np.float32)
    vb = np.zeros(len(vocab), dtype=np.float32)
    for t in a:
        va[idx[t]] += 1.0
    for t in b:
        vb[idx[t]] += 1.0
    va = normalize(va)
    vb = normalize(vb)
    return float(np.dot(va, vb))


def extract_identifier(query_text: str) -> Optional[str]:
    m = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query_text.strip())
    if not m:
        return None
    stop = {
        "where", "what", "how", "is", "the", "a", "an", "defined",
        "used", "called", "function", "method", "class", "file",
        "does", "do", "in", "of", "and", "to", "which", "contains",
        "find", "locate", "definition"
    }
    candidates = [x for x in m if x.lower() not in stop]
    if not candidates:
        return None
    return candidates[-1].lower()


def intent_is_symbol_lookup(query_text: str) -> bool:
    q = query_text.lower()
    return any(x in q for x in ["where is", "defined", "definition", "find", "locate"])


def is_meta_file(node_data: Dict) -> bool:
    nid = node_data.get("id", "").lower()
    name = node_data.get("n", "").lower()
    meta_names = {
        "build_graph.py",
        "scout_agent-qwen.py",
        "benchmark_runner.py",
        "check_hit.py",
    }
    return any(x in nid for x in meta_names) or name in meta_names


def name_score(node_data: Dict, ident: Optional[str]) -> float:
    if not ident:
        return 1.0
    name = node_data.get("n", "").lower()
    nid = node_data.get("id", "").lower()

    if name == ident:
        return 8.0
    if name.endswith(f".{ident}"):
        return 7.0
    if name.split(".")[-1] == ident:
        return 7.0
    if f"::{ident}" in nid:
        return 6.0
    if name.startswith(ident):
        return 2.5
    if ident in name:
        return 1.5
    return 0.5


class CompactScoutAgent:
    def __init__(
        self,
        graph_path: str,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.1,
    ):
        self.graph_path = graph_path
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.meta = {}
        self.graph = self._load_graph()

    def _load_graph(self) -> nx.DiGraph:
        with open(self.graph_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.meta = payload.get("m", {})
        g = nx.DiGraph()

        for node in payload.get("nodes", []):
            g.add_node(node["id"], **node)

        for edge in payload.get("edges", []):
            if len(edge) != 3:
                continue
            s, t, e = edge
            g.add_edge(s, t, edge_type=e, pheromone_weight=1.0)

        print(
            f"[Phase 3] AI graph loaded with {g.number_of_nodes()} nodes and "
            f"{g.number_of_edges()} edges."
        )
        return g

    def _query_tokens(self, query_text: str) -> List[str]:
        q = query_text.lower()
        tokens = compact_tokens(q)

        if "function" in q or "method" in q:
            tokens += ["fn"]
        if "class" in q:
            tokens += ["cl"]
        if "file" in q or ".py" in q:
            tokens += ["fl"]
        if "call" in q or "use" in q or "used" in q:
            tokens += ["calls"]
        if "import" in q:
            tokens += ["imports"]

        ident = extract_identifier(query_text)
        if ident:
            tokens.append(ident)

        return tokens

    def _node_tokens(self, node_data: Dict) -> List[str]:
        toks = []
        toks += compact_tokens(node_data.get("sx", ""))
        toks += compact_tokens(node_data.get("n", ""))
        return toks[:64]

    def _type_bonus(self, node_data: Dict, query_tokens: List[str]) -> float:
        t = node_data.get("t", -1)
        bonus = 1.0

        if "fn" in query_tokens and t == NODE_TYPE["function"]:
            bonus += 0.35
        if "fn" in query_tokens and t == NODE_TYPE["async_function"]:
            bonus += 0.35
        if "cl" in query_tokens and t == NODE_TYPE["class"]:
            bonus += 0.35
        if "fl" in query_tokens and t == NODE_TYPE["file"]:
            bonus += 0.35

        if t in {NODE_TYPE["function"], NODE_TYPE["async_function"]}:
            bonus += 0.05

        return bonus

    def _edge_bonus(self, edge_type: int, query_tokens: List[str]) -> float:
        bonus = 1.0
        if "calls" in query_tokens and edge_type == EDGE_TYPE["calls"]:
            bonus += 0.45
        if "imports" in query_tokens and edge_type == EDGE_TYPE["imports"]:
            bonus += 0.45
        if edge_type == EDGE_TYPE["contains"]:
            bonus += 0.05
        return bonus

    def _depth_bonus(self, node_data: Dict) -> float:
        d = node_data.get("d", 0)
        if d <= 0:
            return 1.0
        return 1.0 + min(d, 4) * 0.04

    def semantic_score(self, query_text: str, node_id: str) -> float:
        q_tokens = self._query_tokens(query_text)
        node_data = self.graph.nodes[node_id]
        n_tokens = self._node_tokens(node_data)

        ident = extract_identifier(query_text)
        sim = cosine_from_token_sets(q_tokens, n_tokens)
        sim = max(sim, 0.01)

        t = node_data.get("t", -1)

        score = sim
        score *= self._type_bonus(node_data, q_tokens)
        score *= self._depth_bonus(node_data)
        score *= name_score(node_data, ident)

        if intent_is_symbol_lookup(query_text):
            if ident:
                if t == NODE_TYPE["file"]:
                    score *= 0.25
                elif t in {NODE_TYPE["function"], NODE_TYPE["async_function"]}:
                    score *= 1.8

        if is_meta_file(node_data):
            score *= 0.15

        return float(score)

    def rank_nodes(self, query_text: str, top_k: int = 8) -> List[Tuple[str, float]]:
        ident = extract_identifier(query_text)
        symbol_lookup = intent_is_symbol_lookup(query_text)

        exact_symbol_nodes = []
        if ident:
            for node_id in self.graph.nodes:
                if node_id == ".":
                    continue
                node = self.graph.nodes[node_id]
                name = node.get("n", "").lower().split(".")[-1]
                if name == ident and node.get("t") in {NODE_TYPE["function"], NODE_TYPE["async_function"], NODE_TYPE["class"]}:
                    exact_symbol_nodes.append(node_id)

        candidates = exact_symbol_nodes if exact_symbol_nodes else list(self.graph.nodes)

        scored = []
        for node_id in candidates:
            if node_id == ".":
                continue

            node = self.graph.nodes[node_id]
            t = node.get("t", -1)

            if symbol_lookup and ident and t == NODE_TYPE["file"] and not exact_symbol_nodes:
                continue

            score = self.semantic_score(query_text, node_id)
            scored.append((node_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _update_pheromones(self, path: List[str], success: bool):
        for u, v in self.graph.edges():
            cur = self.graph[u][v].get("pheromone_weight", 1.0)
            self.graph[u][v]["pheromone_weight"] = max(0.1, cur * (1.0 - self.evaporation_rate))

        if success:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if self.graph.has_edge(u, v):
                    self.graph[u][v]["pheromone_weight"] += 2.0

    def navigate(
    self,
    query_text: str,
    start_node: str = ".",
    target_node: Optional[str] = None,
    max_steps: int = 12,
) -> List[str]:
        ranked = self.rank_nodes(query_text, top_k=8)
    
        if target_node is None:
            if not ranked:
                return [start_node]
            target_node = ranked[0][0]
    
        if target_node not in self.graph.nodes:
            return [start_node]
    
        path = []
        cur = target_node
    
        while True:
            path.append(cur)
            parent = self.graph.nodes[cur].get("p")
            if not parent:
                break
            cur = parent
            if cur == start_node:
                path.append(cur)
                break
    
        path.reverse()
        self._update_pheromones(path, success=True)
        return path

    def build_ai_prompt(self, path_taken: List[str], query_text: str, top_k_ranked: int = 6) -> str:
        ranked = self.rank_nodes(query_text, top_k=top_k_ranked)
        selected = []
        seen = set()

        for nid in path_taken:
            if nid in self.graph.nodes and nid not in seen:
                selected.append(nid)
                seen.add(nid)

        for nid, _ in ranked:
            if nid not in seen:
                selected.append(nid)
                seen.add(nid)

        blocks = []
        for nid in selected:
            node = self.graph.nodes[nid]
            block = (
                f"ID={nid}\n"
                f"T={node.get('t')}\n"
                f"D={node.get('d')}\n"
                f"SX={node.get('sx', '')}\n"
                f"CL={','.join(node.get('cl', [])[:8])}\n"
                f"CB={','.join(node.get('cb', [])[:8])}\n"
            )
            blocks.append(block)

        return (
            f"Q={query_text}\n"
            f"MEMORY_MODE=AI_NATIVE\n"
            f"GRAPH_REPO={self.meta.get('repo', 'unknown')}\n\n"
            + "\n---\n".join(blocks)
        )


def save_run_result(
    query: str,
    ranked: List[Tuple[str, float]],
    path_taken: List[str],
    prompt: str,
    duration_sec: float,
    output_path: str,
):
    payload = {
        "query": query,
        "top_1": ranked[0][0] if ranked else None,
        "ranked_nodes": [[nid, float(score)] for nid, score in ranked],
        "path_taken": path_taken,
        "prompt_char_count": len(prompt),
        "duration_sec": duration_sec,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, default=GRAPH_PATH)
    parser.add_argument("--query", type=str, default="Where is add defined?")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-run", type=str, default=RUN_RESULT_PATH)
    args = parser.parse_args()

    stable_seed(args.seed)

    t0 = time.perf_counter()
    agent = CompactScoutAgent(args.graph)
    ranked = agent.rank_nodes(args.query, top_k=5)

    print("\n[Phase 3] Top ranked nodes:")
    for nid, score in ranked:
        print(f"  - {nid} :: {score:.4f}")

    path_taken = agent.navigate(
        query_text=args.query,
        start_node=".",
        target_node=args.target,
        max_steps=args.steps,
    )

    print(f"\n[Agent] Path taken: {path_taken}\n")

    prompt = agent.build_ai_prompt(path_taken, args.query, top_k_ranked=6)
    dt = time.perf_counter() - t0

    save_run_result(
        query=args.query,
        ranked=ranked,
        path_taken=path_taken,
        prompt=prompt,
        duration_sec=dt,
        output_path=args.save_run,
    )
    print(f"[Run] saved {args.save_run}")


if __name__ == "__main__":
    main()
