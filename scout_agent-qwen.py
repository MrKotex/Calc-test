import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


GRAPH_PATH = ".context-tree/code_graph_ai.json"
CAPSULE_PATH = ".context-tree/context_capsule_ai.json"

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
    torch.manual_seed(seed)


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

        return tokens

    def _node_tokens(self, node_data: Dict) -> List[str]:
        toks = []
        toks += compact_tokens(node_data.get("sx", ""))
        toks += compact_tokens(node_data.get("n", ""))
        toks += compact_tokens(node_data.get("id", ""))
        parent = node_data.get("p", "")
        if parent:
            toks += compact_tokens(parent)
        return toks

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

        sim = cosine_from_token_sets(q_tokens, n_tokens)
        sim = max(sim, 0.01)

        exact_name = node_data.get("n", "").lower()
        if exact_name and exact_name in query_text.lower():
            sim += 0.25

        return sim * self._type_bonus(node_data, q_tokens) * self._depth_bonus(node_data)

    def rank_nodes(self, query_text: str, top_k: int = 8) -> List[Tuple[str, float]]:
        scored = []
        for node_id in self.graph.nodes:
            if node_id == ".":
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

        current_node = start_node
        path = [current_node]
        q_tokens = self._query_tokens(query_text)

        for step in range(max_steps):
            if current_node == target_node:
                print(f"[Phase 3] Reached target {target_node} in {step} steps.")
                self._update_pheromones(path, success=True)
                return path

            neighbors = list(self.graph.successors(current_node))
            if not neighbors:
                break

            scores = []
            for neighbor in neighbors:
                edge_data = self.graph.get_edge_data(current_node, neighbor) or {}
                pheromone = edge_data.get("pheromone_weight", 1.0)
                edge_type = edge_data.get("edge_type", EDGE_TYPE["contains"])

                sem = self.semantic_score(query_text, neighbor)
                e_bonus = self._edge_bonus(edge_type, q_tokens)

                target_bonus = 1.0
                if target_node and neighbor == target_node:
                    target_bonus += 1.0
                elif target_node and nx.has_path(self.graph, neighbor, target_node):
                    target_bonus += 0.25

                score = (pheromone ** self.alpha) * (sem ** self.beta) * e_bonus * target_bonus
                scores.append(max(score, 1e-6))

            probs = np.array(scores, dtype=np.float64)
            probs = probs / probs.sum()

            next_node = str(np.random.choice(neighbors, p=probs))
            path.append(next_node)
            current_node = next_node

        if target_node and nx.has_path(self.graph, start_node, target_node):
            fallback_path = nx.shortest_path(self.graph, source=start_node, target=target_node)
            print("[Phase 3] Falling back to shortest path.")
            self._update_pheromones(fallback_path, success=True)
            return fallback_path

        print("[Phase 3] Navigation ended without exact target.")
        self._update_pheromones(path, success=False)
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

        prompt = (
            f"Q={query_text}\n"
            f"MEMORY_MODE=AI_NATIVE\n"
            f"GRAPH_REPO={self.meta.get('repo', 'unknown')}\n\n"
            + "\n---\n".join(blocks)
        )
        return prompt


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def load_qwen_model():
    print("[Phase 4] Downloading RAW Qwen3.5-0.8B from ModelScope...")
    model_dir = snapshot_download("qwen/Qwen3.5-0.8B")

    print("[Phase 4] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype="auto",
    )
    return tokenizer, model


def extract_last_hidden_states(model, tokenizer, prompt: str) -> np.ndarray:
    device = get_model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("[Phase 4] Running forward pass and extracting hidden states...")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[-1].squeeze(0).float().cpu().numpy()
    print(f"[Phase 4] Hidden state shape: {hidden.shape}")
    return hidden


def realign_latent_vectors(
    hidden_states: np.ndarray,
    source_embed_matrix: np.ndarray,
    target_embed_matrix: np.ndarray,
    lambda_reg: float = 0.5,
) -> np.ndarray:
    print("[Phase 4] Realigning latent vectors via Ridge Regression...")
    x = source_embed_matrix
    y = target_embed_matrix

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch: source={x.shape[0]}, target={y.shape[0]}")

    x_t = x.T
    identity = np.eye(x.shape[1], dtype=np.float32)
    w = np.linalg.inv(x_t @ x + lambda_reg * identity) @ x_t @ y
    aligned = hidden_states @ w
    print(f"[Phase 4] Aligned tensor shape: {aligned.shape}")
    return aligned


def round_trip_cosine(
    aligned_tensor: np.ndarray,
    original_hidden: np.ndarray,
    source_embed_matrix: np.ndarray,
    target_embed_matrix: np.ndarray,
    lambda_reg: float = 0.5,
) -> float:
    y_t = target_embed_matrix.T
    identity = np.eye(target_embed_matrix.shape[1], dtype=np.float32)
    w_back = np.linalg.inv(y_t @ target_embed_matrix + lambda_reg * identity) @ y_t @ source_embed_matrix
    round_trip = aligned_tensor @ w_back

    a = round_trip.flatten()
    b = original_hidden.flatten()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def save_capsule(
    path_taken: List[str],
    query_text: str,
    ai_prompt: str,
    hidden_shape: Tuple[int, int],
    aligned_shape: Tuple[int, int],
    round_trip: float,
):
    capsule = {
        "query": query_text,
        "graph_path_taken": path_taken,
        "prompt_preview": ai_prompt[:4000],
        "source_model": "Qwen3.5-0.8B",
        "memory_format": "code_graph_ai.json",
        "alignment_method": "ridge_regression",
        "source_hidden_shape": list(hidden_shape),
        "aligned_latent_shape": list(aligned_shape),
        "round_trip_cosine": round_trip,
    }

    Path(CAPSULE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(CAPSULE_PATH, "w", encoding="utf-8") as f:
        json.dump(capsule, f, indent=2, ensure_ascii=False)

    print(f"[Phase 4] Saved capsule to {CAPSULE_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph",
        type=str,
        default=GRAPH_PATH,
        help="Path to compact AI-native graph json",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Where is add defined and how is it used?",
        help="Natural language query for graph navigation",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional explicit target node id",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Max navigation steps",
    )
    parser.add_argument(
        "--target-dim",
        type=int,
        default=8192,
        help="Synthetic target embedding dimension for POC alignment",
    )
    args = parser.parse_args()

    stable_seed(42)

    # ---------------------------------------------------------
    # 1. LOAD AI-NATIVE GRAPH + NAVIGATE
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 2. BUILD AI-NATIVE PROMPT
    # ---------------------------------------------------------
    ai_prompt = agent.build_ai_prompt(path_taken, args.query, top_k_ranked=6)
    print("[Phase 4] Built AI-native prompt from compact graph memory.")

    # ---------------------------------------------------------
    # 3. LATENT EXTRACTION
    # ---------------------------------------------------------
    tokenizer, model = load_qwen_model()
    hidden = extract_last_hidden_states(model, tokenizer, ai_prompt)

    # ---------------------------------------------------------
    # 4. ALIGNMENT SETUP
    # ---------------------------------------------------------
    source_embed = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    rng = np.random.default_rng(42)
    target_embed = rng.standard_normal((source_embed.shape[0], args.target_dim), dtype=np.float32)

    # ---------------------------------------------------------
    # 5. LATENT ALIGNMENT
    # ---------------------------------------------------------
    aligned = realign_latent_vectors(hidden, source_embed, target_embed)
    rt = round_trip_cosine(aligned, hidden, source_embed, target_embed)
    print(f"[Phase 4] Round-trip cosine: {rt:.4f}")

    # ---------------------------------------------------------
    # 6. EXPORT CAPSULE
    # ---------------------------------------------------------
    save_capsule(
        path_taken=path_taken,
        query_text=args.query,
        ai_prompt=ai_prompt,
        hidden_shape=hidden.shape,
        aligned_shape=aligned.shape,
        round_trip=rt,
    )


if __name__ == "__main__":
    main()
