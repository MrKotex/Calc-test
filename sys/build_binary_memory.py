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
    import sqlglot.expressions as exp
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

ROOT_DIR = "."
OUTPUT_DIR = ".context-tree"
INDEX_FILE = os.path.join(OUTPUT_DIR, "memory_index.bin")
CONTENT_FILE = os.path.join(OUTPUT_DIR, "memory_content.bin")

EXCLUDED_DIRS = {
    ".git", ".context-tree", "__pycache__",
    ".venv", "venv", "node_modules",
    "dist", "build", "OLD", "sys",
}

INCLUDED_EXTENSIONS = {".py", ".html", ".sql"}

NODE_TYPE = {
    "root":           0,
    "file":           1,
    "class":          2,
    "function":       3,
    "async_function": 4,
    "table":          5,
    "column":         6,
    "view":           7,
    "schema":         8,
    "database":       9,
}

EDGE_TYPE = {
    "contains":   1,
    "calls":      2,
    "imports":    3,
    "references": 4,
    "feeds":      5,
}

# TYPE_TOKEN must cover every NODE_TYPE value used in sx strings
TYPE_TOKEN = {
    0: "RT",
    1: "FL",
    2: "CL",
    3: "FN",
    4: "AF",
    5: "TB",
    6: "CO",
    7: "VW",
    8: "SC",
    9: "DB",
}

MAGIC_NUMBER = 0x42494E4D  # "BINM"


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def rel_path(path: Path, root: Path) -> str:
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
    args = [a.arg for a in node.args.args]
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
            t = type(n.op)
            ops.append({ast.Add: "+", ast.Sub: "-", ast.Mult: "*",
                         ast.Div: "/", ast.Mod: "%", ast.Pow: "**"}.get(t, "?"))
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
    seen: list = []
    for op in ops:
        if op not in seen:
            seen.append(op)
    return ",".join(seen[:6])


def make_sx(
    t: int, n: str, f: str, sig: str, argc: int,
    start: Optional[int], end: Optional[int], doc: bool, op: str,
) -> str:
    line = f"{start}-{end}" if start is not None and end is not None else "?"
    s = sig.replace(" ", "")[:64] if sig else "-"
    d = 1 if doc else 0
    return f"{TYPE_TOKEN[t]}|N:{n}|F:{f}|S:{s}|A:{argc}|L:{line}|D:{d}|O:{op}"


def make_sx_generic(t: int, n: str, f: str, extra: str = "") -> str:
    token = TYPE_TOKEN.get(t, f"T{t}")
    return f"{token}|N:{n}|F:{f}|X:{extra}"


# ---------------------------------------------------------------------------
# Parser Registry
# ---------------------------------------------------------------------------

class ParserRegistry:
    """Holds per-extension parse functions.  Each function returns List[Dict]."""

    def __init__(self, node_idx: Dict[str, int], nodes_list: list):
        # References to the builder's shared state so parsers can look up parents.
        self._node_idx = node_idx
        self._nodes = nodes_list
        self.parsers: Dict[str, object] = {}
        self.register(".py", self.parse_python)
        if HAS_BS4:
            self.register(".html", self.parse_html)
            self.register(".htm",  self.parse_html)
        if HAS_SQLGLOT:
            self.register(".sql", self.parse_sql)

    def register(self, ext: str, func) -> None:
        self.parsers[ext] = func

    # ------------------------------------------------------------------ Python
    def parse_python(self, path: Path, src: str) -> List[Dict]:
        nodes: List[Dict] = []
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return nodes

        imports = collect_imports(tree)
        file_id = rel_path(path, Path("."))
        file_node: Dict = {
            "id":  file_id,
            "t":   NODE_TYPE["file"],
            "n":   path.name,
            "p":   ".",
            "l":   [1, len(src.splitlines())],
            "a":   0,
            "d":   1,
            "sx":  make_sx(NODE_TYPE["file"], path.name, file_id,
                           "", 0, 1, len(src.splitlines()), False, "MOD"),
            "h":   sha16(src),
            "tc":  token_est(src),
            "ch":  [],
            "im":  imports,
            "cl":  [],
            "cb":  [],
        }
        nodes.append(file_node)

        lines = src.splitlines(keepends=True)
        for ast_node in ast.walk(tree):
            if not isinstance(ast_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            is_async = isinstance(ast_node, ast.AsyncFunctionDef)
            if isinstance(ast_node, ast.ClassDef):
                ntype = NODE_TYPE["class"]
                op = "-"
            else:
                ntype = NODE_TYPE["async_function"] if is_async else NODE_TYPE["function"]
                op = op_code_from_tree(ast_node)

            sig  = get_sig(ast_node)
            argc = get_argc(ast_node)
            doc  = bool(ast.get_docstring(ast_node))
            name = ast_node.name
            s_l  = ast_node.lineno
            e_l  = getattr(ast_node, "end_lineno", None)
            child_id = f"{file_id}::{name}"
            calls = collect_calls(ast_node) if not isinstance(ast_node, ast.ClassDef) else []

            child: Dict = {
                "id":  child_id,
                "t":   ntype,
                "n":   name,
                "p":   file_id,
                "l":   [s_l, e_l],
                "a":   0,
                "d":   2,
                "sx":  make_sx(ntype, name, file_id, sig, argc, s_l, e_l, doc, op),
                "h":   sha16(src_segment(lines, s_l, e_l)),
                "tc":  token_est(src_segment(lines, s_l, e_l)),
                "ch":  [],
                "im":  [],
                "cl":  calls,
                "cb":  [],
            }
            nodes.append(child)
            file_node["ch"].append(child_id)

        return nodes

    # ------------------------------------------------------------------- HTML
    def parse_html(self, path: Path, src: str) -> List[Dict]:
        nodes: List[Dict] = []
        if not HAS_BS4:
            return nodes

        soup    = BeautifulSoup(src, "html.parser")
        file_id = rel_path(path, Path("."))

        for table in soup.find_all("table"):
            tid   = table.get("id", "unnamed")
            table_id = f"{file_id}::table.{tid}"
            table_node: Dict = {
                "id":  table_id,
                "t":   NODE_TYPE["table"],
                "n":   tid,
                "p":   file_id,
                "l":   [0, 0],
                "a":   0,
                "d":   2,
                "sx":  make_sx_generic(NODE_TYPE["table"], tid, file_id, "html_table"),
                "h":   sha16(str(table)),
                "tc":  token_est(str(table)),
                "ch":  [],
                "im":  [],
                "cl":  [],
                "cb":  [],
            }
            nodes.append(table_node)

            for th in table.find_all("th"):
                col_name = th.get_text(strip=True)
                col_id   = f"{table_id}::col.{col_name}"
                col_node: Dict = {
                    "id":  col_id,
                    "t":   NODE_TYPE["column"],
                    "n":   col_name,
                    "p":   table_id,
                    "l":   [0, 0],
                    "a":   0,
                    "d":   3,
                    "sx":  make_sx_generic(NODE_TYPE["column"], col_name, file_id, "html_header"),
                    "h":   sha16(str(th)),
                    "tc":  token_est(str(th)),
                    "ch":  [],
                    "im":  [],
                    "cl":  [],
                    "cb":  [],
                }
                nodes.append(col_node)
                table_node["ch"].append(col_id)

        return nodes

    # -------------------------------------------------------------------- SQL
    def parse_sql(self, path: Path, src: str) -> List[Dict]:
        nodes: List[Dict] = []
        if not HAS_SQLGLOT:
            return nodes

        file_id = rel_path(path, Path("."))
        try:
            parsed = sqlglot.parse(src, read="postgres")
        except Exception:
            return nodes

        for expression in parsed:
            if expression is None:
                continue

            # ---- CREATE TABLE -----------------------------------------------
            if isinstance(expression, exp.Create) and isinstance(expression.this, exp.Schema):
                schema_node = expression.this
                table_name  = schema_node.this.name
                table_id    = f"{file_id}::table.{table_name}"

                col_nodes: List[Dict] = []
                for col_def in expression.find_all(exp.ColumnDef):
                    col_name = col_def.name
                    col_id   = f"{table_id}::col.{col_name}"
                    col_node: Dict = {
                        "id":  col_id,
                        "t":   NODE_TYPE["column"],
                        "n":   col_name,
                        "p":   table_id,
                        "l":   [0, 0],
                        "a":   0,
                        "d":   3,
                        "sx":  make_sx_generic(NODE_TYPE["column"], col_name, file_id, "sql_column"),
                        "h":   sha16(str(col_def)),
                        "tc":  token_est(str(col_def)),
                        "ch":  [],
                        "im":  [],
                        "cl":  [],
                        "cb":  [],
                    }
                    col_nodes.append(col_node)

                table_node: Dict = {
                    "id":  table_id,
                    "t":   NODE_TYPE["table"],
                    "n":   table_name,
                    "p":   file_id,
                    "l":   [0, 0],
                    "a":   0,
                    "d":   2,
                    "sx":  make_sx_generic(NODE_TYPE["table"], table_name, file_id, "sql_create_table"),
                    "h":   sha16(str(expression)),
                    "tc":  token_est(str(expression)),
                    "ch":  [c["id"] for c in col_nodes],
                    "im":  [],
                    "cl":  [],
                    "cb":  [],
                }
                nodes.append(table_node)
                nodes.extend(col_nodes)

                # Foreign keys — emit as reference edges (stored in cl for resolution)
                for fk in expression.find_all(exp.ForeignKey):
                    for ref_table in fk.find_all(exp.Table):
                        if ref_table.name and ref_table.name != table_name:
                            table_node["cl"].append(f"fk_ref::{ref_table.name}")

            # ---- CREATE VIEW ------------------------------------------------
            elif isinstance(expression, exp.Create) and isinstance(expression.this, exp.Table):
                view_name = expression.this.name
                view_id   = f"{file_id}::view.{view_name}"

                # Tables referenced inside the SELECT
                ref_names = [
                    t.name for t in expression.find_all(exp.Table)
                    if t.name and t.name != view_name
                ]

                view_node: Dict = {
                    "id":  view_id,
                    "t":   NODE_TYPE["view"],
                    "n":   view_name,
                    "p":   file_id,
                    "l":   [0, 0],
                    "a":   0,
                    "d":   2,
                    "sx":  make_sx_generic(NODE_TYPE["view"], view_name, file_id, "sql_view"),
                    "h":   sha16(str(expression)),
                    "tc":  token_est(str(expression)),
                    "ch":  [],
                    "im":  [],
                    # Store references as strings; builder resolves them to edges.
                    "cl":  [f"feeds_ref::{n}" for n in ref_names],
                    "cb":  [],
                }
                nodes.append(view_node)

        return nodes


# ---------------------------------------------------------------------------
# Binary Memory Builder
# ---------------------------------------------------------------------------

class BinaryMemoryBuilder:
    def __init__(self, root_dir: str):
        self.root   = Path(root_dir).resolve()
        self.nodes: List[Dict]          = []
        self.edges: List[List]          = []
        self.node_idx: Dict[str, int]   = {}
        self.sym_idx:  Dict[str, List[str]] = {}  # name -> [node_id]
        self.imp_idx:  Dict[str, str]   = {}      # module_stem -> file node_id
        self.topo      = 0
        self.embeddings: Dict[str, List[float]] = {}
        # Pass shared state into registry so parsers can reference it if needed.
        self.parser_registry = ParserRegistry(self.node_idx, self.nodes)

    # ------------------------------------------------------------------
    def set_embedding(self, node_id: str, vector: List[float]) -> None:
        self.embeddings[node_id] = vector

    def add_node(self, obj: Dict) -> None:
        self.node_idx[obj["id"]] = len(self.nodes)
        self.nodes.append(obj)

    def add_edge(self, s: str, t: str, e: int) -> None:
        self.edges.append([s, t, e])

    # ------------------------------------------------------------------
    def discover(self) -> List[Path]:
        out: List[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix in INCLUDED_EXTENSIONS:
                    out.append(p)
        out.sort()
        return out

    # ------------------------------------------------------------------
    def build(self) -> None:
        files = self.discover()

        self.add_node({
            "id": ".", "t": NODE_TYPE["root"], "n": "root",
            "p": "", "l": [0, 0], "a": 0, "d": 0, "ti": self.topo,
            "sx": "RT|N:root|F:.|S:-|A:0|L:0-0|D:0|O:-",
            "h": sha16("root"), "tc": 1,
            "ch": [], "im": [], "cl": [], "cb": [],
        })
        self.topo += 1

        for path in files:
            self._process_file(path)

        self._resolve_import_edges()
        self._resolve_call_edges()
        self._compute_called_by()
        self._compute_depths()
        self._generate_embeddings()

    # ------------------------------------------------------------------
    def _process_file(self, path: Path) -> None:
        parser = self.parser_registry.parsers.get(path.suffix)
        if parser is None:
            return

        src  = read_text(path)
        file_id = rel_path(path, self.root)
        parsed_nodes: List[Dict] = parser(path, src)

        for node in parsed_nodes:
            self.add_node(node)
            self.topo += 1

        if not parsed_nodes:
            return

        # Register root -> file edge
        top_node = parsed_nodes[0]
        self.add_edge(".", top_node["id"], EDGE_TYPE["contains"])
        self.nodes[self.node_idx["."]].setdefault("ch", []).append(top_node["id"])

        # Register child -> parent containment edges
        for node in parsed_nodes[1:]:
            parent_id = node.get("p", top_node["id"])
            if parent_id in self.node_idx:
                self.add_edge(parent_id, node["id"], EDGE_TYPE["contains"])

        # Build symbol index from functions/classes
        for node in parsed_nodes:
            self.sym_idx.setdefault(node["n"], []).append(node["id"])

        # Register module stem for import resolution
        if path.suffix == ".py":
            self.imp_idx[path.stem] = top_node["id"]

    # ------------------------------------------------------------------
    def _resolve_import_edges(self) -> None:
        for node in self.nodes:
            src_id = node["id"]
            for imp in node.get("im", []):
                base = imp.split(".")[0]
                if base in self.imp_idx and self.imp_idx[base] != src_id:
                    self.add_edge(src_id, self.imp_idx[base], EDGE_TYPE["imports"])

    def _resolve_call_edges(self) -> None:
        seen: Set[Tuple[str, str, int]] = set()
        for node in self.nodes:
            src_id = node["id"]
            resolved: List[str] = []
            for call in node.get("cl", []):
                # SQL feed/fk references stored as special strings
                if call.startswith("fk_ref::") or call.startswith("feeds_ref::"):
                    ref_name = call.split("::", 1)[1]
                    etype    = EDGE_TYPE["references"] if call.startswith("fk") else EDGE_TYPE["feeds"]
                    for target in self.sym_idx.get(ref_name, []):
                        if src_id != target:
                            trip = (src_id, target, etype)
                            if trip not in seen:
                                self.add_edge(src_id, target, etype)
                                seen.add(trip)
                    continue
                # Regular Python calls
                for target in self.sym_idx.get(call, []):
                    if src_id == target:
                        continue
                    resolved.append(target)
                    trip = (src_id, target, EDGE_TYPE["calls"])
                    if trip not in seen:
                        self.add_edge(src_id, target, EDGE_TYPE["calls"])
                        seen.add(trip)
            node["cl"] = list(set(resolved))

    def _compute_called_by(self) -> None:
        for n in self.nodes:
            n["cb"] = []
        for s, t, e in self.edges:
            if e == EDGE_TYPE["calls"] and t in self.node_idx:
                self.nodes[self.node_idx[t]]["cb"].append(s)

    def _compute_depths(self) -> None:
        q    = ["."]
        seen = {"."}
        while q:
            cur   = q.pop(0)
            cur_d = self.nodes[self.node_idx[cur]]["d"]
            for s, t, e in self.edges:
                if s == cur and e == EDGE_TYPE["contains"] and t not in seen:
                    self.nodes[self.node_idx[t]]["d"] = cur_d + 1
                    q.append(t)
                    seen.add(t)

    def _generate_embeddings(self) -> None:
        print("[build_binary_memory] Generating embeddings...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            for node in self.nodes:
                nid = node["id"]
                vec = model.encode(node.get("sx", "")).tolist()
                self.set_embedding(nid, vec)
        except Exception as e:
            print(f"[build_binary_memory] Warning: embeddings skipped ({e})")

    # ------------------------------------------------------------------
    def export_binary(self) -> None:
        os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)

        # --- Content blob (sx strings) ---
        content_data  = b""
        content_meta: List[Tuple[int, int]] = []
        for node in self.nodes:
            sx_bytes = node.get("sx", "").encode("utf-8")
            content_meta.append((len(content_data), len(sx_bytes)))
            content_data += sx_bytes

        with open(CONTENT_FILE, "wb") as f:
            f.write(content_data)

        # --- Index file ---
        with open(INDEX_FILE, "wb") as f:
            # Header: Magic(4) | Version(4) | NodeCount(4)
            f.write(struct.pack("<III", MAGIC_NUMBER, 1, len(self.nodes)))

            for i, node in enumerate(self.nodes):
                nid = node["id"]
                pid = node.get("p", "")

                # Node ID
                nid_b = nid.encode("utf-8")
                f.write(struct.pack("<I", len(nid_b)))
                f.write(nid_b)

                # Parent ID
                pid_b = pid.encode("utf-8")
                f.write(struct.pack("<I", len(pid_b)))
                f.write(pid_b)

                # Type (1 byte)
                f.write(struct.pack("<B", node["t"]))

                # Edges for this node
                # Collect (edge_type: int, target_id: str) pairs
                node_edges: List[Tuple[int, str]] = []
                for s, t, e in self.edges:
                    if s == nid:
                        node_edges.append((e, t))

                f.write(struct.pack("<I", len(node_edges)))
                for etype, target_id in node_edges:
                    f.write(struct.pack("<B", etype))
                    t_b = target_id.encode("utf-8")
                    f.write(struct.pack("<I", len(t_b)))
                    f.write(t_b)

                # Content offset + length
                offset, length = content_meta[i]
                f.write(struct.pack("<Q", offset))
                f.write(struct.pack("<I", length))

                # Embedding (float32 vector)
                vec = self.embeddings.get(nid, [])
                f.write(struct.pack("<I", len(vec)))
                for val in vec:
                    f.write(struct.pack("<f", val))

        print(f"[build_binary_memory] Written: {INDEX_FILE}, {CONTENT_FILE}")
        print(f"[build_binary_memory] nodes={len(self.nodes)}  edges={len(self.edges)}")


if __name__ == "__main__":
    b = BinaryMemoryBuilder(ROOT_DIR)
    b.build()
    b.export_binary()
