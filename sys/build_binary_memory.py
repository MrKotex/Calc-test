import ast
import hashlib
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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

INCLUDED_EXTENSIONS = {".py"}

NODE_TYPE = {
    "root": 0,
    "file": 1,
    "class": 2,
    "function": 3,
    "async_function": 4,
}

EDGE_TYPE = {
    "contains": 1,
    "calls": 2,
    "imports": 3,
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

    def process_file(self, path: Path):
        rp = rel_path(path, self.root)
        src = read_text(path)
        lines = src.splitlines(keepends=True)

        try:
            tree = ast.parse(src)
        except SyntaxError:
            return

        imports = collect_imports(tree)
        calls = collect_calls(tree)
        node = {
            "id": rp,
            "t": NODE_TYPE["file"],
            "n": path.name,
            "p": ".",
            "l": [1, len(lines)],
            "a": 0,
            "d": 1,
            "ti": self.topo,
            "sx": make_sx(
                NODE_TYPE["file"], path.name, rp, "", 0, 1, len(lines), False, "MOD"
            ),
            "h": sha16(src),
            "tc": token_est(src),
            "ch": [],
            "im": imports,
            "cl": calls,
            "cb": [],
        }
        self.add_node(node)
        self.add_edge(".", rp, EDGE_TYPE["contains"])
        self.nodes[self.node_idx["."]]["ch"].append(rp)
        self.topo += 1

        module_name = path.stem
        self.imp_idx[module_name] = rp

        for item in tree.body:
            if isinstance(item, ast.ClassDef):
                self.process_class(item, rp, lines)
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.process_function(item, rp, lines, parent=rp, owner=None)

    def process_class(self, node: ast.ClassDef, rp: str, lines: List[str]):
        nid = f"{rp}::{node.name}"
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", start)
        code = src_segment(lines, start, end)
        sig = get_sig(node)
        doc = bool(ast.get_docstring(node) or "")
        sx = make_sx(
            NODE_TYPE["class"], node.name, rp, sig, 0, start, end, doc, "CLS"
        )

        obj = {
            "id": nid,
            "t": NODE_TYPE["class"],
            "n": node.name,
            "p": rp,
            "l": [start or 0, end or 0],
            "a": 0,
            "d": 2,
            "ti": self.topo,
            "sx": sx,
            "h": sha16(code or node.name),
            "tc": token_est(code),
            "ch": [],
            "im": [],
            "cl": collect_calls(node),
            "cb": [],
        }
        self.add_node(obj)
        self.add_edge(rp, nid, EDGE_TYPE["contains"])
        self.nodes[self.node_idx[rp]]["ch"].append(nid)
        self.sym_idx.setdefault(node.name, []).append(nid)
        self.topo += 1

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.process_function(item, rp, lines, parent=nid, owner=node.name)

    def process_function(self, node: ast.AST, rp: str, lines: List[str], parent: str, owner: Optional[str]):
        base = node.name
        full = f"{owner}.{base}" if owner else base
        nid = f"{rp}::{full}"
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", start)
        code = src_segment(lines, start, end)
        sig = get_sig(node)
        argc = get_argc(node)
        doc = bool(ast.get_docstring(node) or "")
        t = NODE_TYPE["async_function"] if isinstance(node, ast.AsyncFunctionDef) else NODE_TYPE["function"]
        
        # 🔍 NEW: Extract decorators & optional route paths
        decs = []
        routes = []
        for dec in getattr(node, "decorator_list", []):
            if isinstance(dec, ast.Name):
                decs.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decs.append(dec.attr)
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                decs.append(dec.func.attr)
                # Extract route if present: @app.route("/calc/add", methods=["GET"])
                if dec.args and isinstance(dec.args[0], ast.Constant):
                    routes.append(str(dec.args[0].value))
        
        sx = make_sx(t, full, rp, sig, argc, start, end, doc, op_code_from_tree(node))

        obj = {
            "id": nid,
            "t": t,
            "n": full,
            "p": parent,
            "l": [start or 0, end or 0],
            "a": argc,
            "d": 3 if owner else 2,
            "ti": self.topo,
            "sx": sx,
            "h": sha16(code or full),
            "tc": token_est(code),
            "ch": [],
            "im": [],
            "cl": collect_calls(node),
            "cb": [],
            "dec": decs,          # 🔍 NEW
            "routes": routes,     # 🔍 NEW
        }
        self.add_node(obj)
        self.add_edge(parent, nid, EDGE_TYPE["contains"])
        self.nodes[self.node_idx[parent]]["ch"].append(nid)
        self.sym_idx.setdefault(base, []).append(nid)
        self.topo += 1


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

    def export_binary(self):
        os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
        
        # 1. Write Content Blob
        # We accumulate content in memory first, then write to file
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

                # Write Embedding (NEW)
                vec = self.embeddings.get(nid, [])
                f.write(struct.pack('<I', len(vec))) # Vector Length
                for val in vec:
                    f.write(struct.pack('<d', val))   # Write as double

        print(f"[build_binary_memory] Saved {INDEX_FILE} and {CONTENT_FILE}")
        print(f"[build_binary_memory] nodes={len(self.nodes)} edges={len(self.edges)}")

if __name__ == "__main__":
    b = BinaryMemoryBuilder(ROOT_DIR)
    b.build()
    b.export_binary()