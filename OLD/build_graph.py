import os
import ast
import json
import random
from pathlib import Path
import networkx as nx

class VectorTreeBuilder:
    def __init__(self, target_dir):
        self.target_dir = Path(target_dir)
        self.graph = nx.DiGraph()
        
        # Initialize root node
        self.root_name = self.target_dir.name
        self.graph.add_node(self.root_name, type='directory', name=self.root_name)

    def parse_and_build(self):
        """Walks the directory, parsing .py files to build nodes and structural edges."""
        print(f"Scanning directory: {self.target_dir}")
        
        for root, _, files in os.walk(self.target_dir):
            current_dir = Path(root)
            
            # Skip hidden directories like .git or .context-tree and cache files
            if current_dir.name.startswith('.') or '__pycache__' in current_dir.parts:
                continue

            # Add directory nodes
            dir_node_id = str(current_dir)
            if dir_node_id != self.root_name:
                self.graph.add_node(dir_node_id, type='directory', name=current_dir.name)
                parent_id = str(current_dir.parent)
                self.graph.add_edge(parent_id, dir_node_id, relationship='contains')

            for file in files:
                if not file.endswith('.py'):
                    continue
                
                filepath = current_dir / file
                file_node_id = str(filepath)
                
                # Create file node and link to its directory
                self.graph.add_node(file_node_id, type='file', name=filepath.name)
                self.graph.add_edge(dir_node_id, file_node_id, relationship='contains')
                
                self._parse_python_file(filepath, file_node_id)
                
        # Run a post-process pass to establish complex relationships (imports, invokes)
        self._map_complex_edges()

    def _parse_python_file(self, filepath, file_node_id):
        """Uses AST to extract classes and functions, creating contains edges."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(filepath))
        except Exception as e:
            print(f"Warning: Failed to parse {filepath}: {e}")
            return

        for node in tree.body:
            # Handle Classes
            if isinstance(node, ast.ClassDef):
                class_node_id = f"{file_node_id}::{node.name}"
                self.graph.add_node(class_node_id, type='class', name=node.name)
                self.graph.add_edge(file_node_id, class_node_id, relationship='contains')
                
                # Handle Methods inside Classes
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        func_node_id = f"{class_node_id}::{child.name}"
                        self.graph.add_node(func_node_id, type='function', name=child.name)
                        self.graph.add_edge(class_node_id, func_node_id, relationship='contains')
            
            # Handle Top-level Functions
            elif isinstance(node, ast.FunctionDef):
                func_node_id = f"{file_node_id}::{node.name}"
                self.graph.add_node(func_node_id, type='function', name=node.name)
                self.graph.add_edge(file_node_id, func_node_id, relationship='contains')

            # Handle Imports (simplified mapping for PoC)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                module_name = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                if module_name:
                    # We create a loose module node; complex resolution requires deep static analysis
                    module_node_id = f"module::{module_name}"
                    if not self.graph.has_node(module_node_id):
                        self.graph.add_node(module_node_id, type='module', name=module_name)
                    self.graph.add_edge(file_node_id, module_node_id, relationship='imports')

    def _map_complex_edges(self):
        """
        Placeholder for advanced static analysis.
        Simulates drawing an 'invokes' edge from a Flask route to a calculator method.
        """
        print("Mapping complex semantic edges (imports/invokes)...")
        # In a real scenario, you would use a Call Graph analyzer. 
        # Here we simulate finding a Flask route in main.py invoking calculator.py
        nodes = list(self.graph.nodes(data=True))
        flask_route_node = next((n[0] for n in nodes if 'main.py' in n[0] and n[1].get('type') == 'function'), None)
        calc_method_node = next((n[0] for n in nodes if 'calculator.py' in n[0] and n[1].get('type') == 'function'), None)
        
        if flask_route_node and calc_method_node:
            self.graph.add_edge(flask_route_node, calc_method_node, relationship='invokes')

    def simulate_llm_summary(self, node_type, name):
        """Placeholder: Simulates an LLM call to generate a 1-sentence summary."""
        templates = {
            'directory': f"Directory containing project files for {name}.",
            'file': f"Python source code file named {name}.",
            'class': f"Class definition '{name}' handling grouped state and logic.",
            'function': f"Function '{name}' that extracts inputs, processes data, and returns a result.",
            'module': f"External or internal dependency module named {name}."
        }
        return templates.get(node_type, f"A project node representing {name}.")

    def simulate_text_embedding(self, text, dimensions=384):
        """Placeholder: Simulates generating a continuous vector array (e.g., via sentence-transformers)."""
        # Returns a dummy array of floats
        return [round(random.uniform(-1.0, 1.0), 6) for _ in range(dimensions)]

    def enrich_nodes(self):
        """Iterates through all nodes, attaching semantic summaries and vector embeddings."""
        print("Enriching nodes with semantic summaries and vectors...")
        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_name = data.get('name', 'unknown')
            
            # 1. Semantic Enrichment
            summary = self.simulate_llm_summary(node_type, node_name)
            self.graph.nodes[node_id]['semantic_summary'] = summary
            
            # 2. Vectorization
            # We embed the combination of the node's name and its semantic summary
            content_to_embed = f"{node_name}: {summary}"
            vector = self.simulate_text_embedding(content_to_embed)
            self.graph.nodes[node_id]['vector'] = vector

    def export_graph(self):
        """Exports the networkx graph to a hidden .context-tree directory."""
        out_dir = self.target_dir / '.context-tree'
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / 'code_graph.json'
        
        # networkx provides node_link_data to cleanly export graph topology and attributes to JSON
        graph_data = nx.node_link_data(self.graph)
        
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
            
        print(f"Success! Graph exported to: {out_file}")
        print(f"Total Nodes: {self.graph.number_of_nodes()}")
        print(f"Total Edges: {self.graph.number_of_edges()}")

if __name__ == "__main__":
    # Assuming the script is run in the root of the target project directory
    target_directory = "." 
    
    builder = VectorTreeBuilder(target_directory)
    builder.parse_and_build()
    builder.enrich_nodes()
    builder.export_graph()
