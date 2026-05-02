import struct
import os
import math
import re
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# --- Constants & Helpers from scout_agent-qwen-new.py ---
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

SYNONYMS = {
    "division": "divide", "subtract": "minus", "multiply": "times", 
    "add": "plus", "endpoint": "route", "path": "route", "handler": "function"
}

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

def intent_is_callers_query(query_text: str) -> bool:
    q = query_text.lower()
    return "what calls" in q or "who calls" in q or ("where is" in q and "called" in q)

def intent_is_file_lookup(query_text: str) -> bool:
    q = query_text.lower()
    return "which file" in q or "what file" in q

def intent_is_symbol_lookup(query_text: str) -> bool:
    q = query_text.lower()
    return any(x in q for x in ["where is", "defined", "definition", "find", "locate"])

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

def query_parts(query_text: str) -> Tuple[List[str], List[str]]:
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

def is_meta_file(node_data: Dict) -> bool:
    nid = node_data.get("id", "").lower()
    meta_parts = ["build_graph", "benchmark_runner", "scout_agent", "check_hit", "test_"]
    return any(x in nid for x in meta_parts)

class BinaryScoutAgent:
    def __init__(self, index_path: str, content_path: str, embedding_dim: int = 768, alpha: float = 1.0, beta: float = 2.0, evaporation_rate: float = 0.1):
        self.index_path = index_path
        self.content_path = content_path
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.nodes = []
        self.node_map = {} # id -> node_data
        self.adj = {}      # id -> [child_ids]
        self.rev_adj = {}  # id -> [parent_ids]
        self.embeddings = {} # id -> [floats]
        self._load_index()

    def _load_index(self):
        """Load index into RAM and build adjacency lists."""
        with open(self.index_path, 'rb') as f:
            magic, version, count = struct.unpack('<III', f.read(12))
            
            self.nodes = [None] * count
            self.node_map = {}
            self.adj = {}
            self.rev_adj = {}
            self.embeddings = {}

            for i in range(count):
                # 1. Node ID
                id_len = struct.unpack('<I', f.read(4))[0]
                nid = f.read(id_len).decode('utf-8')
                
                # 2. Parent ID
                parent_len = struct.unpack('<I', f.read(4))[0]
                parent_id = f.read(parent_len).decode('utf-8')
                
                # 3. Type
                ntype = struct.unpack('<B', f.read(1))[0]
                
                # 4. Imports
                imp_count = struct.unpack('<I', f.read(4))[0]
                imports = [f.read(struct.unpack('<I', f.read(4))[0]).decode('utf-8') for _ in range(imp_count)]
                
                # 5. Calls
                call_count = struct.unpack('<I', f.read(4))[0]
                calls = [f.read(struct.unpack('<I', f.read(4))[0]).decode('utf-8') for _ in range(call_count)]
                
                # 6. Called By
                cb_count = struct.unpack('<I', f.read(4))[0]
                called_by = [f.read(struct.unpack('<I', f.read(4))[0]).decode('utf-8') for _ in range(cb_count)]

                # 7. Content Offset & Length
                offset = struct.unpack('<Q', f.read(8))[0]
                length = struct.unpack('<I', f.read(4))[0]
                
                # 8. Embedding (Optional)
                vec_len = struct.unpack('<I', f.read(4))[0]
                vector = []
                if vec_len > 0:
                    # Read as doubles (8 bytes each)
                    vector = [struct.unpack('<d', f.read(8))[0] for _ in range(vec_len)]
                    self.embeddings[nid] = vector

                node_data = {
                    "id": nid, "parent_id": parent_id, "type": ntype,
                    "offset": offset, "length": length,
                    "imports": imports, "calls": calls, "called_by": called_by
                }
                
                self.nodes[i] = node_data
                self.node_map[nid] = node_data
                
                # Build Adjacency Lists
                if parent_id:
                    self.adj.setdefault(parent_id, []).append(nid)
                    self.rev_adj.setdefault(nid, []).append(parent_id)

    def get_node_content(self, node_id: str) -> Optional[str]:
        node = self.node_map.get(node_id)
        if not node: return None
        with open(self.content_path, 'rb') as f:
            f.seek(node["offset"])
            return f.read(node["length"]).decode('utf-8')

    def plan_path(self, start_id: str, target_id: str) -> Optional[List[str]]:
        """BFS on In-RAM Adjacency List."""
        queue = [(start_id, [start_id])]
        visited = set([start_id])
        
        while queue:
            current, path = queue.pop(0)
            if current == target_id:
                return path
            
            for child in self.adj.get(current, []):
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))
        return None

    def get_callers(self, node_id: str, max_depth: int = 3) -> List[str]:
        """Find all nodes that call the target node."""
        queue = [(node_id, 0)]
        callers = []
        visited = set([node_id])
        
        while queue:
            current, depth = queue.pop(0)
            if depth > max_depth: continue
            
            for caller in self.node_map.get(current, {}).get("called_by", []):
                if caller not in visited:
                    visited.add(caller)
                    callers.append(caller)
                    queue.append((caller, depth + 1))
        return callers

    def semantic_score(self, query_vector: List[float], node_id: str) -> float:
        """Cosine similarity between query and node embedding."""
        if node_id not in self.embeddings:
            return 0.0
        
        v1 = query_vector
        v2 = self.embeddings[node_id]
        
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        
        if norm1 == 0 or norm2 == 0: return 0.0
        return dot / (norm1 * norm2)

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
        toks += compact_tokens(node_data.get("id", ""))
        return toks[:64]

    def _type_bonus(self, node_data: Dict, query_tokens: List[str]) -> float:
        t = node_data.get("type", -1)
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

    def context_score(self, node_data: Dict, context_toks: list) -> float:
        if not context_toks:
            return 1.0
        nid = node_data.get("id", "").lower()
        parent = node_data.get("parent_id", "").lower()
        all_tokens = set(re.split(r"[^a-z0-9]+", nid)) | set(re.split(r"[^a-z0-9]+", parent))
        matches = sum(1 for ct in context_toks if ct in all_tokens)
        total = len(context_toks)
        if matches == total:
            return 2.8
        elif matches > 0:
            return 1.0 + (matches / total) * 1.5
        return 0.4

    def file_stem_score(self, node_data: Dict, ident) -> float:
        if not ident:
            return 1.0
        nid = node_data.get("id", "").lower()
        stem = re.sub(r"\.py(::.*)?$", "", nid).lstrip("./")
        if stem == ident or stem.replace("_", "") == ident:
            return 2.0
        if ident in stem:
            return 1.4
        return 1.0

    def file_prior_score(self, node_data: Dict, ident: Optional[str]) -> float:
        if not ident:
            return 1.0
        nid = node_data.get("id", "").lower()
        if f"./{ident}.py::" in nid:
            return 2.5
        if f"./{ident}.py" in nid:
            return 1.8
        return 1.0

    def name_score(self, node_data: Dict, ident: Optional[str]) -> float:
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

    def semantic_score(self, query_text: str, node_id: str) -> float:
        q_tokens = self._query_tokens(query_text)
        node_data = self.node_map[node_id]
        n_tokens = self._node_tokens(node_data)

        ident, all_keywords = query_parts(query_text)
        name = node_data.get("n", "").lower()
        nid = node_data.get("id", "").lower()

        # 🔑 Direct & synonym keyword matching
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

        # Cosine similarity (token-based)
        sim = cosine_from_token_sets(q_tokens, n_tokens)

        # 🔑 FIX: Guarantee non-zero base to prevent collapse
        base_score = max(sim, direct_match * 0.1, 0.01)

        t = node_data.get("type", -1)
        score = base_score
        score *= self._type_bonus(node_data, q_tokens)
        score *= self._depth_bonus(node_data)
        score *= self.context_score(node_data, all_keywords)
        score *= self.file_stem_score(node_data, ident)
        score *= self.name_score(node_data, ident)
        score *= self.file_prior_score(node_data, ident)

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
            for node_id in self.node_map:
                if node_id == ".":
                    continue
                node = self.node_map[node_id]
                base = node.get("n", "").lower().split(".")[-1]
                if base == ident:
                    symbol_nodes.append(node_id)
    
        if callers_query and symbol_nodes:
            caller_scores = {}
            for sym in symbol_nodes:
                node = self.node_map[sym]
                for caller in node.get("called_by", []):
                    caller_scores[caller] = max(
                        caller_scores.get(caller, 0.0),
                        self.semantic_score(query_text, caller) * 2.0
                    )
            ranked = sorted(caller_scores.items(), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]
    
        scored = []
        for node_id in self.node_map:
            if node_id == ".":
                continue
            node = self.node_map[node_id]
            t = node.get("type", -1)
    
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
        for u, v in self.adj.items():
            for child in v:
                # Simulate pheromone on 'contains' edges
                pass # In binary mode, we skip heavy graph updates for performance, 
                      # or we could store them in a separate dict if needed.
                      # For now, we keep it lightweight.

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
    
        if target_node not in self.node_map:
            return [start_node]
    
        path = []
        cur = target_node
        steps = 0
    
        while cur and cur != start_node and steps < max_steps:
            path.append(cur)
            parents = self.rev_adj.get(cur, [])
            if not parents:
                break
            # Prioritize parents with higher implicit score (depth/content match)
            best_parent = max(parents, key=lambda p: self.node_map.get(p, {}).get("d", 0))
            cur = best_parent
            steps += 1
    
        if start_node not in path:
            path.append(start_node)
        path.reverse()
        return path

    def build_ai_prompt(self, path: List[str], query: str, top_k_ranked: int = 6) -> str:
        ranked = self.rank_nodes(query, top_k=top_k_ranked)
        selected = []
        seen = set()

        for nid in path:
            if nid in self.node_map and nid not in seen:
                selected.append(nid)
                seen.add(nid)

        for nid, _ in ranked:
            if nid not in seen:
                selected.append(nid)
                seen.add(nid)

        blocks = []
        for nid in selected:
            node = self.node_map[nid]
            content = self.get_node_content(nid) or ""
            block = (
                f"ID={nid}\n"
                f"T={node.get('type')}\n"
                f"D={node.get('d', 0)}\n"
                f"SX={node.get('sx', '')}\n"
                f"CL={','.join(node.get('calls', [])[:8])}\n"
                f"CB={','.join(node.get('called_by', [])[:8])}\n"
            )
            blocks.append(block)

        return (
            f"Q={query}\n"
            f"MEMORY_MODE=AI_NATIVE\n"
            f"GRAPH_REPO=unknown\n\n"
            + "\n---\n".join(blocks)
        )

# Usage
if __name__ == "__main__":
    agent = BinaryScoutAgent(".context-tree/memory_index.bin", ".context-tree/memory_content.bin")
    
    # 1. Test Path Planning
    path = agent.navigate("Where is add defined?")
    print(f"Path: {path}")
    
    # 2. Test Ranking
    ranked = agent.rank_nodes("Where is add defined?")
    print(f"Top ranked: {ranked[:3]}")
    
    # 3. Test Prompt
    prompt = agent.build_ai_prompt(path, "Where is add defined?")
    print(f"Prompt length: {len(prompt)}")
