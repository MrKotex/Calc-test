import tree_sitter_python as tspython
from tree_sitter import Language, Parser

class TreeSitterBuilder:
    def __init__(self, root_dir):
        self.root = root_dir
        self.parser = Parser()
        self.language = Language(tspython.language_python())
        self.parser.set_language(self.language)

    def build(self):
        # ... (Walk files) ...
        for path in files:
            with open(path, 'rb') as f:
                tree = self.parser.parse(f.read())
                # ... (Traverse tree_sitter nodes instead of ast) ...
                self.process_node(tree.root_node)

    def process_node(self, node):
        if node.type == "function_definition":
            nid = f"{self.current_file}::{node.child_by_field_name('name').text.decode()}"
            # ... (Extract code and metadata) ...
            pass
