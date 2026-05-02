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

def intent_is_callers_query(query_text: str) -> bool:
    q = query_text.lower()
    return "what calls" in q or "who calls" in q or "where is" in q and "called" in q

def intent_is_file_lookup(query_text: str) -> bool:
    q = query_text.lower()
    return "which file" in q or "what file" in q
def query_parts(query_text: str) -> Tuple[List[str], List[str]]:
    """Split query into (primary_identifier, all_context_keywords)"""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query_text)
    stop = {
        "where", "what", "how", "is", "the", "a", "an", "defined",
        "used", "called", "function", "method", "class", "file",
        "does", "do", "in", "of", "and", "to", "which", "contains",
        "find", "locate", "definition", "show", "tell", "list", "endpoint", "route", "path"
    }
    useful = [t.lower() for t in tokens if t.lower() not in stop]
    primary = useful[-1] if useful else None
    return primary, useful



def context_score(node_data: Dict, context_toks: list) -> float:
    """Generic context anchoring — no hardcoded file/symbol names."""
    if not context_toks:
        return 1.0
    nid = node_data.get("id", "").lower()
    parent = node_data.get("p", "").lower()
    all_tokens = set(re.split(r"[^a-z0-9]+", nid)) | set(re.split(r"[^a-z0-9]+", parent))
    matches = sum(1 for ct in context_toks if ct in all_tokens)
    total = len(context_toks)
    if matches == total:
        return 2.8
    elif matches > 0:
        return 1.0 + (matches / total) * 1.5
    return 0.4


def file_stem_score(node_data: Dict, ident) -> float:
    """Boost nodes in a file whose stem matches the identifier."""
    if not ident:
        return 1.0
    nid = node_data.get("id", "").lower()
    stem = re.sub(r"\.py(::.*)?$", "", nid).lstrip("./")
    if stem == ident or stem.replace("_", "") == ident:
        return 2.0
    if ident in stem:
        return 1.4
    return 1.0

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
    meta_parts = [
        "build_graph",
        "benchmark_runner",
        "scout_agent",
        "check_hit",
        "test_",
    ]
    return any(x in nid for x in meta_parts)

def file_prior_score(node_data: Dict, ident: Optional[str]) -> float:
    if not ident:
        return 1.0
    nid = node_data.get("id", "").lower()
    if f"./{ident}.py::" in nid:
        return 2.5
    if f"./{ident}.py" in nid:
        return 1.8
    return 1.0


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

        ident, all_keywords = query_parts(query_text)
        name = node_data.get("n", "").lower()
        nid = node_data.get("id", "").lower()

        # 🔑 Direct & synonym keyword matching
        SYNONYMS = {
            "division": "divide", "subtract": "minus", "multiply": "times", 
            "add": "plus", "endpoint": "route", "path": "route", "handler": "function"
        }
        direct_match = 0.0
        for kw in all_keywords:
            if not kw: continue
            if name == kw or nid == kw:
                direct_match = max(direct_match, 10.0)
            elif kw in name or kw in nid:
                direct_match = max(direct_match, 4.0)
            elif SYNONYMS.get(kw) and SYNONYMS[kw] in name:
                direct_match = max(direct_match, 3.5)
            elif len(set(kw.lower()) & set(name.lower())) > 0.5 * len(kw):
                direct_match = max(direct_match, 2.5)

        # Cosine similarity (only triggers on token overlap)
        sim = cosine_from_token_sets(q_tokens, n_tokens)

        # 🔑 FIX: Guarantee non-zero base to prevent collapse
        base_score = max(sim, direct_match * 0.1, 0.01)

        t = node_data.get("t", -1)
        score = base_score
        score *= self._type_bonus(node_data, q_tokens)
        score *= self._depth_bonus(node_data)
        score *= context_score(node_data, all_keywords)
        score *= file_stem_score(node_data, ident)
        score *= name_score(node_data, ident)
        score *= file_prior_score(node_data, ident)

        if intent_is_symbol_lookup(query_text):
            if ident:
                if t == NODE_TYPE["file"]:
                    score *= 0.15
                elif t in {NODE_TYPE["function"], NODE_TYPE["async_function"]}:
                    score *= 2.2
                elif t == NODE_TYPE["class"]:
                    score *= 0.75

        if is_meta_file(node_data):
            score *= 0.03

        return float(score)



    def rank_nodes(self, query_text: str, top_k: int = 8) -> List[Tuple[str, float]]:
        ident, _ = query_parts(query_text)
        symbol_lookup = intent_is_symbol_lookup(query_text)
        callers_query = intent_is_callers_query(query_text)
        file_lookup = intent_is_file_lookup(query_text)
    
        symbol_nodes = []
        if ident:
            for node_id in self.graph.nodes:
                if node_id == ".":
                    continue
                node = self.graph.nodes[node_id]
                base = node.get("n", "").lower().split(".")[-1]
                if base == ident:
                    symbol_nodes.append(node_id)
    
        if callers_query and symbol_nodes:
            caller_scores = {}
            for sym in symbol_nodes:
                node = self.graph.nodes[sym]
                for caller in node.get("cb", []):
                    caller_scores[caller] = max(
                        caller_scores.get(caller, 0.0),
                        self.semantic_score(query_text, caller) * 2.0
                    )
            ranked = sorted(caller_scores.items(), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]
    
        scored = []
        for node_id in self.graph.nodes:
            if node_id == ".":
                continue
    
            node = self.graph.nodes[node_id]
            t = node.get("t", -1)
    
            if symbol_lookup and not file_lookup and t == NODE_TYPE["file"]:
                continue
            if file_lookup and t != NODE_TYPE["file"]:
                continue
    
            score = self.semantic_score(query_text, node_id)
            scored.append((node_id, score))
    
        # 🔑 Deterministic tie-breaking
        scored.sort(key=lambda x: (
            -x[1],  # score descending
            -int(x[0].split("::")[-1] == ident),  # exact match descending
            len(x[0]),  # shorter name descending
            x[0]  # alphabetical ascending
        ))
        
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
    
        if not ranked:
            return [start_node]
        target_node = target_node or ranked[0][0]
    
        if target_node not in self.graph.nodes:
            return [start_node]
    
        path = []
        cur = target_node
        steps = 0
    
        while cur and cur != start_node and steps < max_steps:
            path.append(cur)
            # Find parent via 'contains' edges
            parents = [
                u for u, v, d in self.graph.in_edges(cur, data=True)
                if d.get("edge_type") == EDGE_TYPE["contains"]
            ]
            if not parents:
                break
            # Prioritize edges with higher pheromone weight
            best_parent = max(parents, key=lambda p: self.graph[p][cur].get("pheromone_weight", 1.0))
            cur = best_parent
            steps += 1
    
        if start_node not in path:
            path.append(start_node)
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
