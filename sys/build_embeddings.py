"""
Phase 2: Fully Local, Graph-First AI Memory System
===================================================
Architecture:
1. BinaryIndexReader: Load Phase 1 index
2. GraphTraversalEngine: Navigate relationships (reads_from, writes_to, calls)
3. LocalEmbeddingStore: FAISS vector store for semantic search
4. HybridRouter: Merge graph + vector results
5. ContextCompressor: Token-optimized output for expert models

Usage:
    python sys/build_embeddings.py --mode generate
    python sys/build_embeddings.py --mode query --query "price calculation" --top-k 5
    python sys/build_embeddings.py --mode compress --input chunks.json --output context.json
"""
import os
import sys
import json
import struct
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import deque

# Local dependencies (no cloud APIs)
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# Paths (must match build_binary_memory.py)
ROOT_DIR = "./sql_data"
OUTPUT_DIR = ".context-tree"
INDEX_FILE = os.path.join(OUTPUT_DIR, "memory_index.bin")
CONTENT_FILE = os.path.join(OUTPUT_DIR, "memory_content.bin")
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "embeddings.faiss")
METADATA_FILE = os.path.join(OUTPUT_DIR, "embeddings_metadata.json")

# Node types mapping
NODE_TYPE = {
    0: "root", 1: "file", 2: "class", 3: "function", 4: "async_function",
    5: "table", 6: "column", 7: "view", 8: "schema", 9: "database",
    10: "html_block", 11: "param_node",
}

EDGE_TYPE = {
    1: "contains", 2: "calls", 3: "imports", 4: "references", 5: "feeds",
}

MAGIC_NUMBER = 0x42494E4D


class BinaryIndexReader:
    """Load and parse Phase 1 binary index."""
    
    def __init__(self, index_path: str, content_path: str):
        self.index_path = index_path
        self.content_path = content_path
        self.nodes: List[Dict] = []
        self.edges: List[List] = []
        self.node_idx: Dict[str, int] = {}
        
    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.content_path):
            raise FileNotFoundError(f"Index or content file not found. Run build_binary_memory.py first.")
        
        # Load Index File
        with open(self.index_path, "rb") as f:
            magic, version, count = struct.unpack('<III', f.read(12))
            if magic != MAGIC_NUMBER:
                raise ValueError("Invalid binary index file")
            print(f"[Reader] Loaded version {version} index with {count} nodes.")
            
            for _ in range(count):
                nid_len = struct.unpack('<I', f.read(4))[0]
                nid = f.read(nid_len).decode('utf-8')
                
                pid_len = struct.unpack('<I', f.read(4))[0]
                pid = f.read(pid_len).decode('utf-8')
                
                node_type = struct.unpack('<B', f.read(1))[0]
                
                meta_len = struct.unpack('<I', f.read(4))[0]
                meta_bytes = f.read(meta_len)
                meta = json.loads(meta_bytes.decode('utf-8'))
                
                edge_count = struct.unpack('<I', f.read(4))[0]
                node_edges = []
                for _ in range(edge_count):
                    etype = struct.unpack('<B', f.read(1))[0]
                    tlen = struct.unpack('<I', f.read(4))[0]
                    tid = f.read(tlen).decode('utf-8')
                    node_edges.append([etype, tid])
                
                sx_offset = struct.unpack('<Q', f.read(8))[0]
                sx_len = struct.unpack('<I', f.read(4))[0]
                
                self.nodes.append({
                    "id": nid, "pid": pid, "type": node_type, "meta": meta,
                    "edges": node_edges, "sx_offset": sx_offset, "sx_len": sx_len
                })
                self.node_idx[nid] = len(self.nodes) - 1
        
        # Build edge index for fast lookup
        self.edge_index: Dict[str, List[Tuple[int, str]]] = {}
        for node in self.nodes:
            for etype, tid in node["edges"]:
                self.edge_index.setdefault(node["id"], []).append((etype, tid))
        
        return self
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        idx = self.node_idx.get(node_id)
        return self.nodes[idx] if idx is not None else None
    
    def get_nodes_by_type(self, node_type: int) -> List[Dict]:
        return [n for n in self.nodes if n["type"] == node_type]
    
    def get_relationships(self, node_id: str) -> Dict[str, List[str]]:
        """Get all relationships for a node."""
        rels = {}
        for etype, tid in self.edge_index.get(node_id, []):
            rels.setdefault(EDGE_TYPE.get(etype, str(etype)), []).append(tid)
        return rels


class GraphTraversalEngine:
    """Navigate relationships in the index."""
    
    def __init__(self, reader: BinaryIndexReader):
        self.reader = reader
        self.adjacency: Dict[str, List[Tuple[int, str]]] = reader.edge_index
        
    def find_procedures_using_table(self, table_id: str) -> List[str]:
        """Find all procedures that read/write to a table."""
        results = []
        for node in self.reader.nodes:
            if node["type"] in (NODE_TYPE["function"], NODE_TYPE["param_node"]):
                meta = node.get("meta", {})
                if table_id in meta.get("reads_from", []) or table_id in meta.get("writes_to", []):
                    results.append(node["id"])
        return results
    
    def get_procedure_dependencies(self, proc_id: str, depth: int = 2) -> Dict:
        """BFS traversal of procedure dependencies."""
        visited = set()
        queue = deque([(proc_id, 0)])
        dependencies = {}
        
        while queue:
            current, d = queue.popleft()
            if d > depth or current in visited:
                continue
            visited.add(current)
            
            node = self.reader.get_node(current)
            if not node:
                continue
            
            dependencies[current] = {
                "type": NODE_TYPE.get(node["type"], "unknown"),
                "meta": node.get("meta", {}),
                "depth": d
            }
            
            # Traverse edges
            for etype, tid in self.adjacency.get(current, []):
                if etype in (EDGE_TYPE["calls"], EDGE_TYPE["references"]):
                    queue.append((tid, d + 1))
        
        return dependencies
    
    def find_related_procedures(self, proc_id: str, max_results: int = 10) -> List[str]:
        """Find procedures that share tables or call each other."""
        proc = self.reader.get_node(proc_id)
        if not proc:
            return []
        
        # Get tables this procedure uses
        tables = set(proc.get("meta", {}).get("reads_from", []) + proc.get("meta", {}).get("writes_to", []))
        
        # Find other procedures using same tables
        related = set()
        for node in self.reader.nodes:
            if node["id"] == proc_id:
                continue
            if node["type"] in (NODE_TYPE["function"], NODE_TYPE["param_node"]):
                node_tables = set(node.get("meta", {}).get("reads_from", []) + node.get("meta", {}).get("writes_to", []))
                if node_tables & tables:
                    related.add(node["id"])
        
        return list(related)[:max_results]


class LocalEmbeddingStore:
    """FAISS-based vector store for local embeddings."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.metadata: List[Dict] = []
        
    def create_index(self):
        if not HAS_FAISS:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")
        self.index = faiss.IndexFlatL2(self.dimension)
    
    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict]):
        if self.index is None:
            self.create_index()
        
        # FAISS expects float32
        vectors = [v for v in vectors if len(v) == self.dimension]
        if vectors:
            self.index.add(faiss.normalize_L2(faiss.vector_to_array(vectors)).astype('float32'))
            self.metadata.extend(metadata)
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        if self.index is None or len(self.metadata) == 0:
            return []
        
        query = faiss.normalize_L2(faiss.array_to_vector(query_vector)).astype('float32')
        distances, indices = self.index.search(query.reshape(1, -1), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "metadata": self.metadata[idx],
                    "distance": distances[0][i],
                    "score": 1.0 / (1.0 + distances[0][i])  # Convert to similarity
                })
        return results
    
    def save(self, path: str):
        if self.index is not None:
            faiss.write_index(self.index, path)
        with open(path.replace('.faiss', '_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def load(self, path: str):
        self.index = faiss.read_index(path)
        with open(path.replace('.faiss', '_metadata.json'), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)


class HybridRouter:
    """Merge graph traversal and vector search results."""
    
    def __init__(self, reader: BinaryIndexReader, graph_engine: GraphTraversalEngine, embedding_store: LocalEmbeddingStore):
        self.reader = reader
        self.graph = graph_engine
        self.embeddings = embedding_store
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Hybrid search: graph + vector."""
        # 1. Graph traversal results
        graph_results = self._graph_search(query_text)
        
        # 2. Vector search results (simplified - would need actual embedding model)
        vector_results = self._vector_search(query_text, top_k)
        
        # 3. Merge and rank
        merged = self._merge_results(graph_results, vector_results)
        return merged[:top_k]
    
    def _graph_search(self, query: str) -> List[Dict]:
        """Simple keyword-based graph search."""
        results = []
        keywords = query.lower().split()
        
        for node in self.reader.nodes:
            if node["type"] in (NODE_TYPE["function"], NODE_TYPE["table"]):
                score = 0
                meta = node.get("meta", {})
                
                # Check metadata for keywords
                for kw in keywords:
                    if kw in str(meta).lower():
                        score += 1
                
                # Check relationships
                rels = self.graph.get_relationships(node["id"])
                for rel_type, targets in rels.items():
                    for t in targets:
                        if any(kw in t.lower() for kw in keywords):
                            score += 0.5
                
                if score > 0:
                    results.append({
                        "node_id": node["id"],
                        "type": NODE_TYPE.get(node["type"], "unknown"),
                        "score": score,
                        "source": "graph"
                    })
        
        return sorted(results, key=lambda x: x["score"], reverse=True)
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Placeholder for vector search."""
        # TODO: Implement actual embedding generation
        # For now, return empty
        return []
    
    def _merge_results(self, graph_results: List[Dict], vector_results: List[Dict]) -> List[Dict]:
        """Merge and deduplicate results."""
        merged = {}
        
        for r in graph_results:
            merged[r["node_id"]] = r
        
        for r in vector_results:
            nid = r["metadata"].get("node_id")
            if nid not in merged:
                merged[nid] = r
            else:
                # Boost score if found in both
                merged[nid]["score"] += r.get("score", 0) * 0.5
                merged[nid]["source"] = "hybrid"
        
        return list(merged.values())


class ContextCompressor:
    """Token-optimized context for expert models."""
    
    def __init__(self):
        self.chunks: List[Dict] = []
    
    def add_chunk(self, chunk: Dict):
        self.chunks.append(chunk)
    
    def compress(self) -> Dict:
        """Deduplicate and format for AI consumption."""
        # Deduplicate by node_id
        seen = set()
        unique_chunks = []
        for chunk in self.chunks:
            nid = chunk.get("node_id")
            if nid and nid not in seen:
                seen.add(nid)
                unique_chunks.append(chunk)
        
        # Format for AI
        return {
            "metadata": {
                "total_nodes": len(unique_chunks),
                "generated_at": "auto",
                "compression": "deduplicated"
            },
            "chunks": unique_chunks
        }


def generate_embeddings_from_index(reader: BinaryIndexReader) -> LocalEmbeddingStore:
    """Generate embeddings for all nodes (stub for local model)."""
    store = LocalEmbeddingStore(dimension=384)
    store.create_index()
    
    vectors = []
    metadata = []
    
    for node in reader.nodes:
        # Generate simple hash-based embedding (stub)
        # TODO: Replace with actual local embedding model
        text = f"{NODE_TYPE.get(node['type'], 'unknown')} {node['id']} {json.dumps(node.get('meta', {}))}"
        hash_bytes = hashlib.sha256(text.encode()).digest()
        vector = [float(b) / 255.0 for b in hash_bytes[:384]]  # Normalize to [0, 1]
        
        vectors.append(vector)
        metadata.append({"node_id": node["id"], "type": NODE_TYPE.get(node["type"], "unknown"), "text": text})
    
    store.add_vectors(vectors, metadata)
    return store


def main():
    parser = argparse.ArgumentParser(description="Phase 2: AI Memory System")
    parser.add_argument("--mode", choices=["generate", "query", "compress"], required=True)
    parser.add_argument("--query", type=str, help="Query text for search mode")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--input", type=str, help="Input file for compress mode")
    parser.add_argument("--output", type=str, help="Output file")
    args = parser.parse_args()
    
    if args.mode == "generate":
        print("[Phase 2] Generating embeddings...")
        reader = BinaryIndexReader(INDEX_FILE, CONTENT_FILE).load()
        store = generate_embeddings_from_index(reader)
        store.save(EMBEDDINGS_FILE)
        print(f"[Phase 2] Saved embeddings to {EMBEDDINGS_FILE}")
        
    elif args.mode == "query":
        if not args.query:
            print("[Error] --query is required for query mode")
            sys.exit(1)
        
        reader = BinaryIndexReader(INDEX_FILE, CONTENT_FILE).load()
        graph = GraphTraversalEngine(reader)
        store = LocalEmbeddingStore()
        store.load(EMBEDDINGS_FILE)
        
        router = HybridRouter(reader, graph, store)
        results = router.query(args.query, args.top_k)
        
        print(f"\n[Phase 2] Query: '{args.query}'")
        print("-" * 50)
        for i, res in enumerate(results, 1):
            print(f"{i}. [{res['type']}] {res['node_id']} (score: {res['score']:.2f})")
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"[Phase 2] Results saved to {args.output}")
        
    elif args.mode == "compress":
        if not args.input:
            print("[Error] --input is required for compress mode")
            sys.exit(1)
        
        with open(args.input, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        compressor = ContextCompressor()
        for chunk in chunks:
            compressor.add_chunk(chunk)
        
        compressed = compressor.compress()
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(compressed, f, indent=2, ensure_ascii=False)
            print(f"[Phase 2] Compressed context saved to {args.output}")
        else:
            print(json.dumps(compressed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()