import ast
import hashlib
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

ROOT_DIR = "."
OUTPUT_DIR = ".context-tree"
INDEX_FILE = os.path.join(OUTPUT_DIR, "memory_index.bin")
CONTENT_FILE = os.path.join(OUTPUT_DIR, "memory_content.bin")

EXCLUDED_DIRS = {
    ".git",
    ".context-tree",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "OLD",
    "sys",
}

INCLUDED_EXTENSIONS = {".py", ".html", ".sql"}

NODE_TYPE = {
    "root": 0,
    "file": 1,
    "class": 2,
    "function": 3,
    "async_function": 4,
    "table": 5,
    "column": 6,
    "view": 7,
    "schema": 8,
    "database": 9,
}

EDGE_TYPE = {
    "contains": 1,
    "calls": 2,
    "imports": 3,
    "references": 4,
    "feeds": 5,
}

TYPE_TOKEN = {
    0: "RT",
    1: "FL",
    2: "CL",
    3: "FN",
    4: "AF",
}

MAGIC_NUMBER = 0x42494E4D  # "BINM"

def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def rel_path(path: Path, root: Path) -> str:
    # Ensure root is absolute so it works with absolute `path` arguments
    if not root.is_absolute():
        root = Path.cwd() / root
    return f"./{path.relative_to(root).as_posix()}"

def token_est(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))

def src_segment(lines: List[str], start: Optional[int], end: Optional[int]) -> str:
    if start is None or end is None:
        return ""
    return "".join(lines[max(0, start - 1): min(len(lines), end)]).strip()

def get_argc(node: ast.AST) -> int:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return 0
    argc = len(node.args.args) + len(node.args.kwonlyargs)
    if node.args.vararg:
        argc += 1
    if node.args.kwarg:
        argc += 1
    return argc

def get_sig(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        return node.name
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    args = []
    for a in node.args.args:
        args.append(a.arg)
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    for a in node.args.kwonlyargs:
        args.append(a.arg)
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    return f"{node.name}({','.join(args)})"

def collect_imports(tree: ast.AST) -> List[str]:
    out = []
    class V(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                out.append(alias.name)
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            mod = node.module or ""
            for alias in node.names:
                out.append(f"{mod}.{alias.name}" if mod else alias.name)
            self.generic_visit(node)

    V().visit(tree)
    return out

def collect_calls(node: ast.AST) -> List[str]:
    out = []
    class V(ast.NodeVisitor):
        def visit_Call(self, n):
            name = None
            if isinstance(n.func, ast.Name):
                name = n.func.id
            elif isinstance(n.func, ast.Attribute):
                name = n.func.attr
            if name:
                out.append(name)
            self.generic_visit(n)

    V().visit(node)
    return out

def op_code_from_tree(node: ast.AST) -> str:
    ops = []
    class V(ast.NodeVisitor):
        def visit_BinOp(self, n):
            if isinstance(n.op, ast.Add):
                ops.append("+")
            elif isinstance(n.op, ast.Sub):
                ops.append("-")
            elif isinstance(n.op, ast.Mult):
                ops.append("*")
            elif isinstance(n.op, ast.Div):
                ops.append("/")
            elif isinstance(n.op, ast.Mod):
                ops.append("%")
            elif isinstance(n.op, ast.Pow):
                ops.append("**")
            self.generic_visit(n)

        def visit_Return(self, n):
            ops.append("RET")
            self.generic_visit(n)

        def visit_Raise(self, n):
            ops.append("RAISE")
            self.generic_visit(n)

        def visit_Try(self, n):
            ops.append("TRY")
            self.generic_visit(n)

    V().visit(node)
    if not ops:
        return "-"
    seen = []
    for op in ops:
        if op not in seen:
            seen.append(op)
    return ",".join(seen[:6])

def make_sx(
    t: int,
    n: str,
    f: str,
    sig: str,
    argc: int,
    start: Optional[int],
    end: Optional[int],
    doc: bool,
    op: str,
) -> str:
    line = f"{start}-{end}" if start is not None and end is not None else "?"
    s = sig.replace(" ", "")[:64] if sig else "-"
    d = 1 if doc else 0
    return f"{TYPE_TOKEN[t]}|N:{n}|F:{f}|S:{s}|A:{argc}|L:{line}|D:{d}|O:{op}"

def make_sx_generic(t: int, n: str, f: str, extra: str = "") -> str:
    """Generic sx for non-Python files."""
    return f"TYPE:{t}|N:{n}|F:{f}|X:{extra}"

class ParserRegistry:
    def __init__(self):
        self.parsers = {}
        self.register('.py', self.parse_python)
        if HAS_BS4:
            self.register('.html', self.parse_html)
            self.register('.htm', self.parse_html)
        if HAS_SQLGLOT:
            self.register('.sql', self.parse_sql)

    def register(self, ext, func):
        self.parsers[ext] = func

    def parse_python(self, path: Path, src: str) -> List[Dict]:
        nodes = []
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return nodes
        
        imports = collect_imports(tree)
        calls = collect_calls(tree)
        nodes.append({
            "id": rel_path(path, Path(".")),
            "t": NODE_TYPE["file"],
            "n": path.name,
            "p": ".",
            "l": [1, len(src.splitlines())],
            "a": 0,
            "d": 1,
            "sx": make_sx(NODE_TYPE["file"], path.name, rel_path(path, Path(".")), "", 0, 1, len(src.splitlines()), False, "MOD"),
            "h": sha16(src),
            "tc": token_est(src),
            "ch": [],
            "im": imports,
            "cl": calls,
            "cb": [],
        })
        
        module_name = path.stem
        # We'll handle imports in a separate pass in the builder if needed, 
        # but for now we just return the file node.
        return nodes

    def parse_html(self, path: Path, src: str) -> List[Dict]:
        nodes = []
        if not HAS_BS4:
            return nodes
        
        soup = BeautifulSoup(src, 'html.parser')
        file_id = rel_path(path, Path("."))
        
        # Extract tables
        for table in soup.find_all('table'):
            table_id = f"{file_id}::table.{table.get('id', 'unnamed')}"
            nodes.append({
                "id": table_id,
                "t": NODE_TYPE["table"],
                "n": table.get('id', 'unnamed'),
                "p": file_id,
                "l": [0, 0],
                "a": 0,
                "d": 2,
                "sx": make_sx_generic(NODE_TYPE["table"], table.get('id', 'unnamed'), file_id, "html_table"),
                "h": sha16(str(table)),
                "tc": token_est(str(table)),
                "ch": [],
                "im": [],
                "cl": [],
                "cb": [],
            })
            # Extract columns (headers)
            headers = table.find_all('th')
            for i, th in enumerate(headers):
                col_id = f"{table_id}::col.{th.get_text(strip=True)}"
                nodes.append({
                    "id": col_id,
                    "t": NODE_TYPE["column"],
                    "n": th.get_text(strip=True),
                    "p": table_id,
                    "l": [0, 0],
                    "a": 0,
                    "d": 3,
                    "sx": make_sx_generic(NODE_TYPE["column"], th.get_text(strip=True), file_id, "html_header"),
                    "h": sha16(str(th)),
                    "tc": token_est(str(th)),
                    "ch": [],
                    "im": [],
                    "cl": [],
                    "cb": [],
                })
                nodes[-1]["ch"] = [] # Ensure list exists
                nodes[self.node_idx[table_id]]["ch"].append(col_id) # Add to parent if exists
        
        return nodes

    def parse_sql(self, path: Path, src: str) -> List[Dict]:
        nodes = []
        if not HAS_SQLGLOT:
            return nodes

        file_id = rel_path(path, Path("."))
        try:
            parsed = sqlglot.parse(src, read='postgres') # Default to postgres dialect
        except Exception:
            return nodes

        for expression in parsed:
            if isinstance(expression, sqlglot.exp.CreateTable):
                table_name = expression.this.name
                table_id = f"{file_id}::table.{table_name}"
                
                # Columns
                columns = []
                for col in expression.find_all(sqlglot.exp.ColumnDef):
                    col_name = col.name
                    col_id = f"{table_id}::col.{col_name}"
                    columns.append({
                        "id": col_id,
                        "t": NODE_TYPE["column"],
                        "n": col_name,
                        "p": table_id,
                        "l": [0, 0],
                        "a": 0,
                        "d": 3,
                        "sx": make_sx_generic(NODE_TYPE["column"], col_name, file_id, "sql_column"),
                        "h": sha16(str(col)),
                        "tc": token_est(str(col)),
                        "ch": [],
                        "im": [],
                        "cl": [],
                        "cb": [],
                    })
                    nodes.append(columns[-1])
                
                # Foreign Keys
                for fk in expression.find_all(sqlglot.exp.ForeignKey):
                    for target in fk.find_all(sqlglot.exp.Table):
                        target_name = target.name
                        nodes.append({
                            "id": f"{table_id}::fk.{target_name}",
                            "t": NODE_TYPE["table"], # Or a specific FK type if needed, but table is fine for now
                            "n": f"fk_{target_name}",
                            "p": table_id,
                            "l": [0, 0],
                            "a": 0,
                            "d": 3,
                            "sx": make_sx_generic(NODE_TYPE["references"], f"fk_{target_name}", file_id, "fk_reference"),
                            "h": sha16(str(fk)),
                            "tc": token_est(str(fk)),
                            "ch": [],
                            "im": [],
                            "cl": [],
                            "cb": [],
                        })
                        nodes.append(nodes[-1]) # Add to list
                        # Add edge references
                        # We'll handle edges in the builder

                table_node = {
                    "id": table_id,
                    "t": NODE_TYPE["table"],
                    "n": table_name,
                    "p": file_id,
                    "l": [0, 0],
                    "a": 0,
                    "d": 2,
                    "sx": make_sx_generic(NODE_TYPE["table"], table_name, file_id, "sql_create_table"),
                    "h": sha16(str(expression)),
                    "tc": token_est(str(expression)),
                    "ch": [c["id"] for c in columns],
                    "im": [],
                    "cl": [],
                    "cb": [],
                }
                nodes.insert(0, table_node)
                nodes[0]["ch"] = [c["id"] for c in columns] # Update ch
                # Add columns to nodes list after table
                nodes.extend(columns)

            elif isinstance(expression, sqlglot.exp.Create) and isinstance(expression.this, sqlglot.exp.Table):
                view_name = expression.this.name
                view_id = f"{file_id}::view.{view_name}"
                nodes.append({
                    "id": view_id,
                    "t": NODE_TYPE["view"],
                    "n": view_name,
                    "p": file_id,
                    "l": [0, 0],
                    "a": 0,
                    "d": 2,
                    "sx": make_sx_generic(NODE_TYPE["view"], view_name, file_id, "sql_view"),
                    "h": sha16(str(expression)),
                    "tc": token_est(str(expression)),
                    "ch": [],
                    "im": [],
                    "cl": [],
                    "cb": [],
                })
                # Find referenced tables
                for table in expression.find_all(sqlglot.exp.Table):
                    if table.name != view_name:
                        nodes.append({
                            "id": f"{view_id}::feeds.{table.name}",
                            "t": NODE_TYPE["table"],
                            "n": f"feeds_{table.name}",
                            "p": view_id,
                            "l": [0, 0],
                            "a": 0,
                            "d": 3,
                            "sx": make_sx_generic(NODE_TYPE["feeds"], f"feeds_{table.name}", file_id, "feeds_reference"),
                            "h": sha16(str(table)),
                            "tc": token_est(str(table)),
                            "ch": [],
                            "im": [],
                            "cl": [],
                            "cb": [],
                        })
                        nodes.append(nodes[-1])

        return nodes

class BinaryMemoryBuilder:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir).resolve()
        self.nodes: List[Dict] = []
        self.edges: List[List] = []
        self.node_idx: Dict[str, int] = {}
        self.sym_idx: Dict[str, List[str]] = {}
        self.imp_idx: Dict[str, str] = {}
        self.topo = 0
        self.content_buffer = b""
        self.content_offsets = {} # Map node_id -> (offset, length)
        self.embeddings = {}
        self.parser_registry = ParserRegistry()

    def set_embedding(self, node_id: str, vector: List[float]):
            """Store embedding for a node."""
            self.embeddings[node_id] = vector

    def add_node(self, obj: Dict):
        self.node_idx[obj["id"]] = len(self.nodes)
        self.nodes.append(obj)

    def add_edge(self, s: str, t: str, e: int):
        self.edges.append([s, t, e])

    def discover(self) -> List[Path]:
        out = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix in INCLUDED_EXTENSIONS:
                    out.append(p)
        out.sort()
        return out

    def build(self):
        files = self.discover()

        self.add_node({
            "id": ".",
            "t": NODE_TYPE["root"],
            "n": "root",
            "p": "",
            "l": [0, 0],
            "a": 0,
            "d": 0,
            "ti": self.topo,
            "sx": "RT|N:root|F:.|S:-|A:0|L:0-0|D:0|O:-",
            "h": sha16("root"),
            "tc": 1,
            "ch": [],
            "im": [],
            "cl": [],
            "cb": [],
        })
        self.topo += 1

        for path in files:
            self.process_file(path)

        self.resolve_import_edges()
        self.resolve_call_edges()
        self.compute_called_by()
        self.compute_depths()
        self.generate_embeddings()

    def process_file(self, path: Path):
        rp = rel_path(path, self.root)
        src = read_text(path)
        
        # Use Parser Registry
        nodes = self.parser_registry.parsers.get(path.suffix, self.parse_python)(path, src)
        
        for node in nodes:
            self.add_node(node)
            self.add_edge(".", node["id"], EDGE_TYPE["contains"])
            self.nodes[self.node_idx["."]]["ch"].append(node["id"])
            self.topo += 1
            
            # Store imports/calls for resolution
            if path.suffix == '.py':
                self.imp_idx[path.stem] = node["id"]
                # Store calls for resolution
                for call in node.get("cl", []):
                    self.sym_idx.setdefault(call, []).append(node["id"]) # This is a simplification, usually calls are from functions

    def parse_python(self, path: Path, src: str) -> List[Dict]:
        nodes = []
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return nodes
        
        imports = collect_imports(tree)
        calls = collect_calls(tree)
        nodes.append({
            "id": rel_path(path, Path(".")),
            "t": NODE_TYPE["file"],
            "n": path.name,
            "p": ".",
            "l": [1, len(src.splitlines())],
            "a": 0,
            "d": 1,
            "sx": make_sx(NODE_TYPE["file"], path.name, rel_path(path, Path(".")), "", 0, 1, len(src.splitlines()), False, "MOD"),
            "h": sha16(src),
            "tc": token_est(src),
            "ch": [],
            "im": imports,
            "cl": calls,
            "cb": [],
        })
        
        module_name = path.stem
        # We'll handle imports in a separate pass in the builder if needed, 
        # but for now we just return the file node.
        return nodes

    def resolve_import_edges(self):
        for node in self.nodes:
            src_id = node["id"]
            for imp in node.get("im", []):
                base = imp.split(".")[0]
                if base in self.imp_idx:
                    self.add_edge(src_id, self.imp_idx[base], EDGE_TYPE["imports"])

    def resolve_call_edges(self):
        seen: Set[Tuple[str, str, int]] = set()
        for node in self.nodes:
            src_id = node["id"]
            for call in node.get("cl", []):
                for target in self.sym_idx.get(call, []):
                    if src_id == target:
                        continue
                    trip = (src_id, target, EDGE_TYPE["calls"])
                    if trip not in seen:
                        self.edges.append([src_id, target, EDGE_TYPE["calls"]])
                        seen.add(trip)

    def compute_called_by(self):
        for n in self.nodes:
            n["cb"] = []
        for s, t, e in self.edges:
            if e == EDGE_TYPE["calls"] and t in self.node_idx:
                self.nodes[self.node_idx[t]]["cb"].append(s)

    def compute_depths(self):
        q = ["."]
        seen = {"."}
        while q:
            cur = q.pop(0)
            cur_d = self.nodes[self.node_idx[cur]]["d"]
            for s, t, e in self.edges:
                if s == cur and e == EDGE_TYPE["contains"] and t not in seen:
                    self.nodes[self.node_idx[t]]["d"] = cur_d + 1
                    q.append(t)
                    seen.add(t)

    def generate_embeddings(self):
        print("[build_binary_memory] Generating embeddings...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            for node in self.nodes:
                nid = node["id"]
                sx = node.get("sx", "")
                content = sx # Use sx as content for embedding
                vec = model.encode(content).tolist()
                self.set_embedding(nid, vec)
        except Exception as e:
            print(f"[build_binary_memory] Warning: Could not generate embeddings: {e}")

    def export_binary(self):
        os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
        
        # 1. Write Content Blob
        content_data = b""
        offsets = []
        
        for node in self.nodes:
            sx = node.get("sx", "")
            offset = len(content_data)
            content_bytes = sx.encode("utf-8")
            content_data += content_bytes
            offsets.append((offset, len(content_bytes)))

        with open(CONTENT_FILE, "wb") as f:
            f.write(content_data)

        # 2. Write Index
        with open(INDEX_FILE, "wb") as f:
            # Header: Magic (4), Version (4), Count (4)
            f.write(struct.pack('<III', MAGIC_NUMBER, 1, len(self.nodes)))
            
            for i, node in enumerate(self.nodes):
                nid = node["id"]
                pid = node.get("p", "")
                imports = node.get("im", [])
                calls = node.get("cl", [])
                called_by = node.get("cb", [])
                
                # Write ID
                nid_bytes = nid.encode('utf-8')
                f.write(struct.pack('<I', len(nid_bytes)))
                f.write(nid_bytes)
                
                # Write Parent
                pid_bytes = pid.encode('utf-8')
                f.write(struct.pack('<I', len(pid_bytes)))
                f.write(pid_bytes)
                
                # Write Type
                f.write(struct.pack('<B', node["t"]))
                
                # Write Imports
                f.write(struct.pack('<I', len(imports)))
                for imp in imports:
                    imp_bytes = imp.encode('utf-8')
                    f.write(struct.pack('<I', len(imp_bytes)))
                    f.write(imp_bytes)
                
                # Write Calls
                f.write(struct.pack('<I', len(calls)))
                for call in calls:
                    call_bytes = call.encode('utf-8')
                    f.write(struct.pack('<I', len(call_bytes)))
                    f.write(call_bytes)
                
                # Write Called By
                f.write(struct.pack('<I', len(called_by)))
                for cb in called_by:
                    cb_bytes = cb.encode('utf-8')
                    f.write(struct.pack('<I', len(cb_bytes)))
                    f.write(cb_bytes)
                
                # Write Offset & Length
                offset = offsets[i][0]
                length = offsets[i][1]
                f.write(struct.pack('<Q', offset))
                f.write(struct.pack('<I', length))

                # Write Embedding (NEW) - float32
                vec = self.embeddings.get(nid, [])
                f.write(struct.pack('<I', len(vec))) # Vector Length
                for val in vec:
                    f.write(struct.pack('<f', val))   # Write as float32

        print(f"[build_binary_memory] Saved {INDEX_FILE} and {CONTENT_FILE}")
        print(f"[build_binary_memory] nodes={len(self.nodes)} edges={len(self.edges)}")

if __name__ == "__main__":
    b = BinaryMemoryBuilder(ROOT_DIR)
    b.build()
    b.export_binary()
