import json
from typing import List

import networkx as nx
import numpy as np
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==========================================
# PHASE 3: ACO SCOUT AGENT
# ==========================================

class ScoutAgent:
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
        self.graph = self._load_and_initialize_graph()

    def _load_and_initialize_graph(self) -> nx.DiGraph:
        with open(self.graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        graph = nx.node_link_graph(data, edges="links")

        for u, v, attrs in graph.edges(data=True):
            if "pheromone_weight" not in attrs:
                graph[u][v]["pheromone_weight"] = 1.0

        print(
            f"[Phase 3] Graph loaded with {graph.number_of_nodes()} nodes and "
            f"{graph.number_of_edges()} edges."
        )
        return graph

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        v1, v2 = np.array(vec1), np.array(vec2)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def navigate(
        self,
        query_vector: List[float],
        start_node: str,
        target_node: str,
        max_steps: int = 15,
    ) -> List[str]:
        current_node = start_node
        path = [current_node]

        for step in range(max_steps):
            if current_node == target_node:
                print(f"[Phase 3] Target {target_node} reached successfully in {step} steps.")
                self._update_pheromones(path, success=True)
                return path

            neighbors = list(self.graph.successors(current_node))
            if not neighbors:
                break

            scores = []
            for neighbor in neighbors:
                edge_data = self.graph.get_edge_data(current_node, neighbor)
                pheromone = edge_data.get("pheromone_weight", 1.0)

                neighbor_vec = self.graph.nodes[neighbor].get(
                    "vector", np.zeros(len(query_vector))
                )
                similarity = self._cosine_similarity(query_vector, neighbor_vec)
                heuristic = max(similarity, 0.01)

                score = (pheromone ** self.alpha) * (heuristic ** self.beta)
                scores.append(score)

            total_score = sum(scores)
            if total_score == 0:
                probabilities = [1.0 / len(neighbors)] * len(neighbors)
            else:
                probabilities = [s / total_score for s in scores]

            next_node = np.random.choice(neighbors, p=probabilities)
            path.append(next_node)
            current_node = next_node

        print("[Phase 3] Navigation failed to reach target within max steps.")
        self._update_pheromones(path, success=False)
        return path

    def _update_pheromones(self, path: List[str], success: bool):
        for u, v in self.graph.edges():
            current_ph = self.graph[u][v]["pheromone_weight"]
            self.graph[u][v]["pheromone_weight"] = max(
                0.1, current_ph * (1.0 - self.evaporation_rate)
            )

        if success:
            boost_value = 2.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if self.graph.has_edge(u, v):
                    self.graph[u][v]["pheromone_weight"] += boost_value

        print("[Phase 3] Pheromone matrix updated.")


# ==========================================
# PHASE 4: MATHEMATICAL ALIGNMENT
# ==========================================

def realign_latent_vectors(
    hidden_states: np.ndarray,
    source_embed_matrix: np.ndarray,
    target_embed_matrix: np.ndarray,
    lambda_reg: float = 0.5,
) -> np.ndarray:
    print("[Phase 4] Realigning latent vectors via Ridge Regression...")

    X = source_embed_matrix
    Y = target_embed_matrix

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Embedding row mismatch: source rows={X.shape[0]}, target rows={Y.shape[0]}"
        )

    x_t = X.T
    identity = np.eye(X.shape[1], dtype=np.float32)
    w = np.linalg.inv(x_t @ X + lambda_reg * identity) @ x_t @ Y

    aligned_latents = hidden_states @ w
    print(f"[Phase 4] Latent vectors realigned. Output shape: {aligned_latents.shape}")
    return aligned_latents


def build_context_prompt(agent: ScoutAgent, path_taken: List[str]) -> str:
    path_code_snippets = []

    for node_id in path_taken:
        node_data = agent.graph.nodes.get(node_id, {})

        snippet_parts = [
            f"NODE: {node_id}",
            f"TYPE: {node_data.get('type', 'unknown')}",
        ]

        label = node_data.get("label")
        if label:
            snippet_parts.append(f"LABEL: {label}")

        name = node_data.get("name")
        if name:
            snippet_parts.append(f"NAME: {name}")

        docstring = node_data.get("docstring")
        if docstring:
            snippet_parts.append(f"DOC: {docstring}")

        path_code_snippets.append("\n".join(snippet_parts))

    context_prompt = "\n\n".join(path_code_snippets).strip()

    if not context_prompt:
        context_prompt = "def add(a, b):\n    return a + b\n"

    return context_prompt


def extract_last_hidden_states(model, tokenizer, prompt: str) -> np.ndarray:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("[Phase 4] Running forward pass and extracting latents...")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[-1].squeeze(0).float().cpu().numpy()
    print(f"[Phase 4] SUCCESS! Extracted real latent tensor of shape: {hidden.shape}")
    return hidden


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    np.random.seed(42)

    # ---------------------------------------------------------
    # 1. SCOUT AGENT NAVIGATION
    # ---------------------------------------------------------
    target_graph = ".context-tree/code_graph.json"
    agent = ScoutAgent(target_graph)

    dummy_query_vector = list(np.random.uniform(-1, 1, 384))

    root_node = "."
    target_node = "./calculator.py::add"

    path_taken = agent.navigate(dummy_query_vector, root_node, target_node)
    print(f"\n[Agent] Path taken: {path_taken}\n")

    # ---------------------------------------------------------
    # 2. BUILD PROMPT FROM GRAPH PATH
    # ---------------------------------------------------------
    context_prompt = build_context_prompt(agent, path_taken)
    print("[Phase 4] Built context prompt from graph traversal.")

    # ---------------------------------------------------------
    # 3. REAL LATENT EXTRACTION
    # ---------------------------------------------------------
    print("[Phase 4] Downloading RAW Qwen3.5-0.8B from Alibaba ModelScope...")
    model_dir = snapshot_download("qwen/Qwen3.5-0.8B")

    print("[Phase 4] Loading model into local memory...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype="auto",
    )

    extracted_latents_np = extract_last_hidden_states(model, tokenizer, context_prompt)
    sequence_length, slm_dim = extracted_latents_np.shape
    print(f"[Phase 4] Detected SLM dimension: {slm_dim}")

    # ---------------------------------------------------------
    # 4. ALIGNMENT SPACE SETUP
    # ---------------------------------------------------------
    large_model_dim = 8192

    source_embed_matrix = (
        model.get_output_embeddings().weight.detach().float().cpu().numpy()
    )
    print(f"[Phase 4] Source embedding matrix shape: {source_embed_matrix.shape}")

    rng = np.random.default_rng(42)
    target_embed_matrix = rng.standard_normal(
        (source_embed_matrix.shape[0], large_model_dim),
        dtype=np.float32,
    )
    print(f"[Phase 4] Synthetic target embedding matrix shape: {target_embed_matrix.shape}")

    # ---------------------------------------------------------
    # 5. REALIGN LATENTS
    # ---------------------------------------------------------
    aligned_tensor = realign_latent_vectors(
        extracted_latents_np,
        source_embed_matrix,
        target_embed_matrix,
    )

    # ---------------------------------------------------------
    # 6. ROUND-TRIP VALIDATION
    # ---------------------------------------------------------
    lambda_reg = 0.5
    y_t = target_embed_matrix.T
    identity_back = np.eye(target_embed_matrix.shape[1], dtype=np.float32)

    w_back = np.linalg.inv(y_t @ target_embed_matrix + lambda_reg * identity_back) @ y_t @ source_embed_matrix
    round_trip = aligned_tensor @ w_back

    numerator = float(np.dot(round_trip.flatten(), extracted_latents_np.flatten()))
    denominator = float(
        np.linalg.norm(round_trip.flatten()) * np.linalg.norm(extracted_latents_np.flatten())
    )
    cosine_check = numerator / denominator if denominator > 0 else 0.0
    print(f"[Phase 4] Round-trip fidelity (cosine sim): {cosine_check:.4f}")

    # ---------------------------------------------------------
    # 7. EXPORT POC CAPSULE
    # ---------------------------------------------------------
    capsule = {
        "graph_path_taken": path_taken,
        "context_prompt": context_prompt[:4000],
        "source_model": "Qwen3.5-0.8B",
        "alignment_method": "ridge_regression",
        "source_hidden_shape": list(extracted_latents_np.shape),
        "aligned_latent_shape": list(aligned_tensor.shape),
        "target_dim": large_model_dim,
        "round_trip_cosine": cosine_check,
    }

    with open(".context-tree/context_capsule.json", "w", encoding="utf-8") as f:
        json.dump(capsule, f, indent=2)

    print("[Phase 4] Context capsule saved to .context-tree/context_capsule.json")
