import json
import numpy as np
import networkx as nx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

# ==========================================
# PHASE 3: ACO SCOUT AGENT
# ==========================================

class ScoutAgent:
    def __init__(self, graph_path: str, alpha: float = 1.0, beta: float = 2.0, evaporation_rate: float = 0.1):
        """
        Initializes the Scout Agent with Ant Colony Optimization routing.
        alpha: Importance of pheromones.
        beta: Importance of heuristic (cosine similarity).
        evaporation_rate: Rate at which unchosen paths lose pheromones.
        """
        self.graph_path = graph_path
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.graph = self._load_and_initialize_graph()

    def _load_and_initialize_graph(self) -> nx.DiGraph:
        """Loads the vector tree and initializes ACO pheromones on all edges."""
        with open(self.graph_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Explicitly set edges="links" to silence the NetworkX 3.6 FutureWarning
        graph = nx.node_link_graph(data, edges="links")
        
        # Initialize default pheromone weights cleanly
        for u, v, attrs in graph.edges(data=True):
            if 'pheromone_weight' not in attrs:
                graph[u][v]['pheromone_weight'] = 1.0
                
        print(f"[Phase 3] Graph loaded with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        return graph

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculates cosine similarity between two continuous vectors."""
        v1, v2 = np.array(vec1), np.array(vec2)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def navigate(self, query_vector: List[float], start_node: str, target_node: str, max_steps: int = 15) -> List[str]:
        """
        Navigates the graph using ACO probability distributions.
        Equation: P(i,j) = (Pheromone^alpha * Similarity^beta) / Sum(Pheromone^alpha * Similarity^beta)
        """
        current_node = start_node
        path = [current_node]
        
        for step in range(max_steps):
            if current_node == target_node:
                print(f"[Phase 3] Target {target_node} reached successfully in {step} steps.")
                self._update_pheromones(path, success=True)
                return path

            neighbors = list(self.graph.successors(current_node))
            if not neighbors:
                break # Dead end

            # Calculate transition probabilities
            scores = []
            for neighbor in neighbors:
                edge_data = self.graph.get_edge_data(current_node, neighbor)
                pheromone = edge_data.get('pheromone_weight', 1.0)
                
                # Heuristic is the vector similarity between query and the neighbor node
                neighbor_vec = self.graph.nodes[neighbor].get('vector', np.zeros(len(query_vector)))
                similarity = self._cosine_similarity(query_vector, neighbor_vec)
                
                # Ensure similarity is positive for probability calculation
                heuristic = max(similarity, 0.01) 
                
                # Math: Score = \tau^\alpha * \eta^\beta
                score = (pheromone ** self.alpha) * (heuristic ** self.beta)
                scores.append(score)

            # Normalize to create a probability distribution
            total_score = sum(scores)
            if total_score == 0:
                probabilities = [1.0 / len(neighbors)] * len(neighbors)
            else:
                probabilities = [s / total_score for s in scores]

            # Choose next node based on distribution
            next_node = np.random.choice(neighbors, p=probabilities)
            path.append(next_node)
            current_node = next_node

        print("[Phase 3] Navigation failed to reach target within max steps. (Expected behavior with dummy vectors)")
        self._update_pheromones(path, success=False)
        return path

    def _update_pheromones(self, path: List[str], success: bool):
        """Applies positive feedback and evaporation logic (AMRO-S inspired)."""
        # Global Evaporation: Penalty to all edges
        for u, v in self.graph.edges():
            current_ph = self.graph[u][v]['pheromone_weight']
            # Math: \tau = (1 - \rho) * \tau
            self.graph[u][v]['pheromone_weight'] = max(0.1, current_ph * (1.0 - self.evaporation_rate))

        # Positive Feedback Loop: Boost chosen path if successful
        if success:
            boost_value = 2.0 # Arbitrary reward value for PoC
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                self.graph[u][v]['pheromone_weight'] += boost_value
        print("[Phase 3] Pheromone matrix updated.")


# ==========================================
# PHASE 4: MATHEMATICAL ALIGNMENT
# ==========================================

def realign_latent_vectors(hidden_states: np.ndarray, input_embeds_matrix: np.ndarray, output_embeds_matrix: np.ndarray) -> np.ndarray:
    """
    Applies a simulated linear alignment operator via ridge regression.
    Prepares the SLM's tensor for injection into the larger downstream model.
    """
    print("[Phase 4] Realigning latent vectors via Ridge Regression...")
    
    # We want to find a transformation matrix W such that: (hidden_states * W) aligns with the target space.
    # Math: W = (X^T X + \lambda I)^{-1} X^T Y
    
    lambda_reg = 0.5
    X = input_embeds_matrix
    Y = output_embeds_matrix
    
    # Calculate Ridge Regression weights (W)
    X_T = X.T
    identity = np.eye(X.shape[1])
    W = np.linalg.inv(X_T.dot(X) + lambda_reg * identity).dot(X_T).dot(Y)
    
    # Apply alignment operator to the extracted hidden states
    aligned_latents = hidden_states.dot(W)
    
    print(f"[Phase 4] Latent vectors realigned. Output shape: {aligned_latents.shape}")
    return aligned_latents

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. SCOUT AGENT NAVIGATION
    # ---------------------------------------------------------
    target_graph = ".context-tree/code_graph.json"
    agent = ScoutAgent(target_graph)
    
    # Still using dummy vectors for the PoC navigation until we upgrade build_graph.py
    dummy_query_vector = list(np.random.uniform(-1, 1, 384))
    
    root_node = "." 
    target_node = "./calculator.py::add" 
    
    path_taken = agent.navigate(dummy_query_vector, root_node, target_node)
    print(f"\n[Agent] Path taken: {path_taken}\n")
    
    # ---------------------------------------------------------
    # 2. REAL LATENT EXTRACTION (Using Qwen3.5-0.8B)
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # 2. REAL LATENT EXTRACTION (Using ModelScope / Alibaba)
    # ---------------------------------------------------------
    print("[Phase 4] Downloading RAW Qwen3.5-0.8B from Alibaba ModelScope...")
    from modelscope import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    # This downloads from Alibaba instead of Hugging Face
    model_dir = snapshot_download('qwen/Qwen3.5-0.8B')
    
    print("[Phase 4] Loading model into local memory...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        device_map="auto", 
        torch_dtype="auto"
    )
    
    prompt = "def add(a, b):\n    return a + b\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("[Phase 4] Running forward pass and extracting latents...")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    real_hidden_states = outputs.hidden_states[-1] 
    
    print(f"[Phase 4] SUCCESS! Extracted real latent tensor of shape: {real_hidden_states.shape}")
    
    # ---------------------------------------------------------
    # 3. REALIGN LATENTS
    extracted_latents_np = real_hidden_states.squeeze(0).float().cpu().numpy() 
    
    # DYNAMIC DIMENSION DETECTION
    # extracted_latents_np shape is (sequence_length, hidden_dimension)
    sequence_length, slm_dim = extracted_latents_np.shape
    
    print(f"[Phase 4] Detected SLM dimension: {slm_dim}")

    large_model_dim = 8192 # Your target "large model" dimension
    
    # Ensure our dummy alignment matrices match the REAL model dimensions
    # Math check: (sequence_length, slm_dim) dot (slm_dim, large_model_dim)
    X_matrix = np.random.randn(sequence_length, slm_dim)
    Y_matrix = np.random.randn(sequence_length, large_model_dim)
    
    aligned_tensor = realign_latent_vectors(extracted_latents_np, X_matrix, Y_matrix)
