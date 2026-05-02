import json
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from typing import List, Tuple, Dict

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
            
        # networkx 3.x+ uses 'links' as default for edges
        graph = nx.node_link_graph(data)
        
        # Initialize default pheromone weights
        for u, v, attrs in graph.edges(data=True):
            if 'pheromone_weight' not in attrs:
                graph[edges[u, v]]['pheromone_weight'] = 1.0
                
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

        print("[Phase 3] Navigation failed to reach target within max steps.")
        self._update_pheromones(path, success=False)
        return path

    def _update_pheromones(self, path: List[str], success: bool):
        """Applies positive feedback and evaporation logic (AMRO-S inspired)."""
        # Global Evaporation: Penalty to all edges
        for u, v in self.graph.edges():
            current_ph = self.graph[u][v]['pheromone_weight']
            # Math: \tau = (1 - \rho) * \tau
            self.graph[u][v]['pheromone_weight'] = max(0.1, current_ph * (1.0 - self.蒸发_rate))

        # Positive Feedback Loop: Boost chosen path if successful
        if success:
            boost_value = 2.0 # Arbitrary reward value for PoC
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                self.graph[u][v]['pheromone_weight'] += boost_value
        print("[Phase 3] Pheromone matrix updated.")


# ==========================================
# PHASE 4: LATENT EXTRACTION & HANDOFF
# ==========================================

class CacheOnlyAttentionBackend(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Linear layers to simulate projection to Key/Value space
        self.to_fake_k = nn.Linear(hidden_size, hidden_size)
        self.to_fake_v = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Intercepts hidden states, reshapes into K/V, simulates cache writing,
        and returns the UNALTERED hidden states.
        """
        # Extract and detach to avoid messing up the main computational graph's gradients
        latent_thoughts = hidden_states.detach().clone()
        
        # Reshape into fake Key/Value pairs
        fake_keys = self.to_fake_k(latent_thoughts)
        fake_values = self.to_fake_v(latent_thoughts)
        
        # [MOCK] Here you would call vLLM's paged attention cache block manager
        # e.g., cache_engine.write_to_cache(fake_keys, fake_values, block_table)
        print(f"[Phase 4] CacheOnlyBackend: Extracted {latent_thoughts.shape} and simulated KV write.")
        
        # Return original hidden_states to allow the forward pass to continue cleanly
        return hidden_states

# MOCK: Subclassing the vLLM Qwen execution model
# In a real environment, you import Qwen2ForCausalLM from vllm.model_executor
class LatentQwenForCausalLM(nn.Module): 
    def __init__(self, hidden_size: int = 4096):
        super().__init__()
        self.hidden_size = hidden_size
        # Simulating the main LLM layers
        self.transformer_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        # Inject our custom cache-only attention layer
        self.latent_extractor = CacheOnlyAttentionBackend(hidden_size)
        self.lm_head = nn.Linear(hidden_size, 50256) # Vocab size

    def forward(self, input_ids_or_embeds: torch.Tensor) -> torch.Tensor:
        """Custom forward pass implementing the extraction."""
        x = self.transformer_layers(input_ids_or_embeds)
        
        # Extract latents mid-pass
        x = self.latent_extractor(x)
        
        logits = self.lm_head(x)
        return logits

def realign_latent_vectors(hidden_states: np.ndarray, input_embeds_matrix: np.ndarray, output_embeds_matrix: np.ndarray) -> np.ndarray:
    """
    Applies a simulated linear alignment operator via ridge regression.
    Prepares the SLM's tensor for injection into the larger downstream model.
    """
    print("[Phase 4] Realigning latent vectors via Ridge Regression...")
    
    # We want to find a transformation matrix W such that: (hidden_states * W) aligns with the target space.
    # Math: W = (X^T X + \lambda I)^{-1} X^T Y
    
    # For PoC, we will use a dummy ridge regression analytic solution
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
    # 1. Setup Scout Agent
    target_graph = ".context-tree/code_graph.json"
    agent = ScoutAgent(target_graph)
    
    # Mocking a user query embedded into a 384-dimensional vector
    dummy_query_vector = list(np.random.uniform(-1, 1, 384))
    
    # In the graph from phase 2, the root is usually the directory name (e.g., ".")
    # We will pretend the agent is looking for "calculator.py::add"
    root_node = "." 
    target_node = "./calculator.py::add" # Ensure this matches your actual node IDs
    
    path_taken = agent.navigate(dummy_query_vector, root_node, target_node)
    print(f"Path taken by agent: {path_taken}")
    
    # 2. Setup Latent Handoff
    # Simulate a batch of 1 sequence, 10 tokens, 4096 hidden size
    mock_hidden_states = torch.randn(1, 10, 4096) 
    
    custom_qwen = LatentQwenForCausalLM(hidden_size=4096)
    _ = custom_qwen(mock_hidden_states)
    
    # 3. Realign Latents
    # Mock embedding matrices for the SLM (input) and the larger Model (output)
    slm_dim = 4096
    large_model_dim = 8192
    tokens = 100 # Sample vocabulary overlap for regression
    
    X_matrix = np.random.randn(tokens, slm_dim)
    Y_matrix = np.random.randn(tokens, large_model_dim)
    
    extracted_latents_np = mock_hidden_states.squeeze(0).numpy() # Shape (10, 4096)
    
    aligned_tensor = realign_latent_vectors(extracted_latents_np, X_matrix, Y_matrix)
