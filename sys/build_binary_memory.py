import ast
import hashlib
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import re

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
        
        # 1. File Node
        nodes.append({
            "id": file_id, "t": NODE_TYPE["file"], "n": path.name, "p": ".",
            "l": [0, 0], "a": 0, "d": 1,
            "sx": make_sx_generic(NODE_TYPE["file"], path.name, file_id, "html_doc"),
            "h": sha16(src), "tc": token_est(src), "ch": [], "im": [], "cl": [], "cb": [],
        })

        # 2. Extract Tables & Data
        for i, table in enumerate(soup.find_all('table')):
            table_name = table.get('id', f'unnamed_table_{i}')
            table_id = f"{file_id}::table.{table_name}"
            
            nodes.append({
                "id": table_id, "t": NODE_TYPE["table"], "n": table_name, "p": file_id,
                "l": [0, 0], "a": 0, "d": 2,
                "sx": make_sx_generic(NODE_TYPE["table"], table_name, file_id, "html_table"),
                "h": sha16(str(table)), "tc": token_est(str(table)), "ch": [], "im": [], "cl": [], "cb": [],
            })
            
            table_children = []
            headers = table.find_all('th')
            
            for j, th in enumerate(headers):
                col_name = th.get_text(strip=True)
                col_id = f"{table_id}::col.{col_name}"
                nodes.append({
                    "id": col_id, "t": NODE_TYPE["column"], "n": col_name, "p": table_id,
                    "l": [0, 0], "a": 0, "d": 3,
                    "sx": make_sx_generic(NODE_TYPE["column"], col_name, file_id, "html_header"),
                    "h": sha16(str(th)), "tc": token_est(str(th)), "ch": [], "im": [], "cl": [], "cb": [],
                })
                table_children.append(col_id)
                # Edge logic removed: BinaryMemoryBuilder.process_file handles this via node["ch"]

            rows = table.find_all('tr')
            for r_idx, row in enumerate(rows):
                row_id = f"{table_id}::row.{r_idx}"
                row_data = [td.get_text(strip=True) for td in row.find_all('td')]
                nodes.append({
                    "id": row_id, "t": NODE_TYPE["file"], "n": f"row_{r_idx}", "p": table_id,
                    "l": [0, 0], "a": 0, "d": 3,
                    "sx": make_sx_generic(NODE_TYPE["file"], f"row_{r_idx}", file_id, f"row_data:{','.join(row_data[:3])}..."),
                    "h": sha16(str(row)), "tc": token_est(str(row)), "ch": [], "im": [], "cl": [], "cb": [],
                })
                table_children.append(row_id)
                # Edge logic removed: BinaryMemoryBuilder.process_file handles this via node["ch"]
                
                for c_idx, td in enumerate(row.find_all('td')):
                    if c_idx < len(headers):
                        # Edge logic removed: BinaryMemoryBuilder.process_file handles this via node["ch"]
                        pass

            nodes[-1]["ch"] = table_children # Link table children
            # Edge logic removed: BinaryMemoryBuilder.process_file handles this via node["ch"]

        # 3. Extract SQL from <PRE> / <CODE> tags
        for tag in soup.find_all(['pre', 'code', 'textarea']):
            anchor = tag.find_previous('a')
            block_name = anchor.get('name', 'unnamed_block') if anchor else 'unnamed_block'
            
            sql_text = tag.get_text(strip=True)
            if not sql_text: 
                continue
                
            # Robust keyword detection
            clean_text = re.sub(r'/\*.*?\*/', '', sql_text, flags=re.DOTALL).strip()
            if not any(clean_text.upper().startswith(kw) for kw in ['CREATE', 'SELECT', 'UPDATE', 'INSERT', 'DROP', 'ALTER', 'GRANT', 'IF']):
                continue
                
            # Use 'tsql' for Microsoft SQL Server
            sql_nodes = self.parse_sql(Path(f"{file_id}::sql_block"), sql_text, file_id=file_id, dialect='tsql')
            nodes.extend(sql_nodes)
            
            # Link extracted SQL nodes to the parent HTML file
            for sql_node in sql_nodes:
                # Edge logic removed: BinaryMemoryBuilder.process_file handles this via node["ch"]
                pass
        
        return nodes

    def parse_sql(self, path: Path, src: str, file_id: str = None, dialect: str = 'tsql') -> List[Dict]:
        nodes = []
        if not HAS_SQLGLOT:
            return self._extract_sql_details_regex(file_id, src)

        if file_id is None:
            file_id = rel_path(path, Path("."))
        
        # 1. Try sqlglot first (handles CREATE TABLE, VIEW, PROCEDURE, FUNCTION)
        try:
            parsed = sqlglot.parse(src, read=dialect)
            for expression in parsed:
                if isinstance(expression, sqlglot.exp.Create):
                    kind = expression.kind
                    this = expression.this
                    
                    if kind == 'TABLE':
                        table_name = this.name
                        table_id = f"{file_id}::table.{table_name}"
                        columns = []
                        for col in expression.find_all(sqlglot.exp.ColumnDef):
                            col_name = col.name
                            col_id = f"{table_id}::col.{col_name}"
                            columns.append({
                                "id": col_id, "t": NODE_TYPE["column"], "n": col_name, "p": table_id,
                                "l": [0, 0], "a": 0, "d": 3,
                                "sx": make_sx_generic(NODE_TYPE["column"], col_name, file_id, "sql_column"),
                                "h": sha16(str(col)), "tc": token_est(str(col)), "ch": [], "im": [], "cl": [], "cb": [],
                            })
                            nodes.append(columns[-1])
                        table_node = {
                            "id": table_id, "t": NODE_TYPE["table"], "n": table_name, "p": file_id,
                            "l": [0, 0], "a": 0, "d": 2,
                            "sx": make_sx_generic(NODE_TYPE["table"], table_name, file_id, "sql_create_table"),
                            "h": sha16(str(expression)), "tc": token_est(str(expression)),
                            "ch": [c["id"] for c in columns], "im": [], "cl": [], "cb": [],
                        }
                        nodes.insert(0, table_node)
                        nodes[0]["ch"] = [c["id"] for c in columns]

                    elif kind == 'VIEW':
                        view_name = this.name
                        view_id = f"{file_id}::view.{view_name}"
                        nodes.append({
                            "id": view_id, "t": NODE_TYPE["view"], "n": view_name, "p": file_id,
                            "l": [0, 0], "a": 0, "d": 2,
                            "sx": make_sx_generic(NODE_TYPE["view"], view_name, file_id, "sql_view"),
                            "h": sha16(str(expression)), "tc": token_est(str(expression)),
                            "ch": [], "im": [], "cl": [], "cb": [],
                        })
                        for table in expression.find_all(sqlglot.exp.Table):
                            if table.name != view_name:
                                nodes.append({
                                    "id": f"{view_id}::feeds.{table.name}", "t": NODE_TYPE["table"],
                                    "n": f"feeds_{table.name}", "p": view_id,
                                    "l": [0, 0], "a": 0, "d": 3,
                                    "sx": make_sx_generic(NODE_TYPE["feeds"], f"feeds_{table.name}", file_id, "feeds_reference"),
                                    "h": sha16(str(table)), "tc": token_est(str(table)),
                                    "ch": [], "im": [], "cl": [], "cb": [],
                                })
                                nodes.append(nodes[-1])

                    elif kind in ('PROCEDURE', 'FUNCTION'):
                        obj_type = kind.lower()
                        proc_name = this.name if hasattr(this, 'name') else str(this)
                        proc_id = f"{file_id}::{obj_type}.{proc_name}"
                        nodes.append({
                            "id": proc_id, "t": NODE_TYPE["function"], "n": proc_name, "p": file_id,
                            "l": [0, 0], "a": 0, "d": 2,
                            "sx": make_sx_generic(NODE_TYPE["function"], proc_name, file_id, f"sql_create_{obj_type}"),
                            "h": sha16(str(expression)), "tc": token_est(str(expression)),
                            "ch": [], "im": [], "cl": [], "cb": [],
                        })
                        # Trigger regex to get parameters and table refs for this procedure
                        nodes.extend(self._extract_sql_details_regex(file_id, src, proc_name))
        except Exception as e:
            print(f"[build_binary_memory] Warning: sqlglot failed to parse SQL in {file_id}: {e}")

        # 2. Fallback: Extract details using Regex for anything sqlglot missed or failed on
        # This now correctly catches CREATE TABLE, Procedures, Parameters, and References
        regex_nodes = self._extract_sql_details_regex(file_id, src)
        nodes.extend(regex_nodes)
        
        return nodes

    def _extract_sql_details_regex(self, file_id: str, src: str, target_name: str = None) -> List[Dict]:
        """
        Robust Regex extractor for T-SQL documentation.
        Extracts: CREATE TABLE, Procedure/Function Names, Parameters, Referenced Tables, and Columns.
        """
        nodes = []
        
        # 1. Extract CREATE TABLE statements (Fixed: was previously ignored in regex fallback)
        table_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_.]+)\s*\(([^)]+)\)'
        for match in re.finditer(table_pattern, src, re.IGNORECASE):
            table_name = match.group(1)
            columns_str = match.group(2)
            table_id = f"{file_id}::table.{table_name}"
            
            col_nodes = []
            for col_line in columns_str.split(','):
                col_line = col_line.strip()
                if not col_line: continue
                col_match = re.match(r'(@?\w+)\s+(\w+)', col_line)
                if col_match:
                    col_name = col_match.group(1)
                    col_type = col_match.group(2)
                    col_id = f"{table_id}::col.{col_name}"
                    col_nodes.append({
                        "id": col_id, "t": NODE_TYPE["column"], "n": col_name, "p": table_id,
                        "l": [0, 0], "a": 0, "d": 3,
                        "sx": make_sx_generic(NODE_TYPE["column"], col_name, file_id, f"col:{col_type}"),
                        "h": sha16(col_name), "tc": token_est(col_name),
                        "ch": [], "im": [], "cl": [], "cb": [],
                    })
                    nodes.append(col_nodes[-1])
            
            nodes.insert(0, {
                "id": table_id, "t": NODE_TYPE["table"], "n": table_name, "p": file_id,
                "l": [0, 0], "a": 0, "d": 2,
                "sx": make_sx_generic(NODE_TYPE["table"], table_name, file_id, "regex_create_table"),
                "h": sha16(table_name), "tc": token_est(table_name),
                "ch": [c["id"] for c in col_nodes], "im": [], "cl": [], "cb": [],
            })

        # 2. Extract Procedure/Function Name
        proc_match = re.search(r'CREATE\s+(PROCEDURE|FUNCTION)\s+([^\s@]+)', src, re.IGNORECASE)
        if not proc_match:
            return nodes # Return early if no proc/function found, but keep TABLES extracted above
            
        obj_type = proc_match.group(1)
        obj_name = proc_match.group(2)
        
        if target_name and obj_name != target_name:
            return nodes 

        node_type = NODE_TYPE["function"]
        obj_id = f"{file_id}::{obj_type.lower()}.{obj_name}"

        # 3. Extract Parameters
        param_pattern = r'@(\w+)\s+(VARCHAR|INT|DECIMAL|DATETIME|SMALLINT|BIT|NVARCHAR|CHAR|NCHAR|FLOAT|REAL|NUMERIC|MONEY|UNIQUEIDENTIFIER|TIMESTAMP|IMAGE|TEXT|VARBINARY|BINARY)\s*(=\s*[^,\n]+)?'
        params = re.findall(param_pattern, src)
        
        param_nodes = []
        for p_name, p_type, p_default in params:
            param_id = f"{obj_id}::param.{p_name}"
            param_nodes.append({
                "id": param_id,
                "t": NODE_TYPE["function"],
                "n": f"@{p_name}",
                "p": obj_id,
                "l": [0, 0], "a": 0, "d": 3,
                "sx": make_sx_generic(NODE_TYPE["function"], f"@{p_name}", file_id, f"param:{p_type}"),
                "h": sha16(p_name), "tc": token_est(p_name),
                "ch": [], "im": [], "cl": [], "cb": [],
            })
            nodes.append(param_nodes[-1])
            # Note: self.add_edge removed. BinaryMemoryBuilder.process_file handles child linking via node["ch"]

        # 4. Extract Referenced Tables
        table_pattern = r'(?:FROM|JOIN|INSERT\s+INTO|UPDATE)\s+([A-Za-z0-9_.]+)'
        tables = re.findall(table_pattern, src)
        
        for table_name in tables:
            if table_name.startswith('@') or table_name.upper() in ('SELECT', 'WHERE', 'SET', 'VALUES'):
                continue
            table_id = f"{file_id}::table.{table_name}"
            nodes.append({
                "id": table_id,
                "t": NODE_TYPE["table"],
                "n": table_name,
                "p": file_id,
                "l": [0, 0], "a": 0, "d": 2,
                "sx": make_sx_generic(NODE_TYPE["table"], table_name, file_id, "regex_table_ref"),
                "h": sha16(table_name), "tc": token_est(table_name),
                "ch": [], "im": [], "cl": [], "cb": [],
            })

        # 5. Extract Referenced Columns
        select_pattern = r'SELECT\s+([\w\s,]+)\s+FROM'
        selects = re.findall(select_pattern, src)
        
        for col_list in selects:
            cols = [c.strip() for c in col_list.split(',')]
            for col_name in cols:
                if not col_name or col_name.startswith('@'):
                    continue
                if '(' not in col_name:
                    col_id = f"{obj_id}::col.{col_name}"
                    nodes.append({
                        "id": col_id,
                        "t": NODE_TYPE["column"],
                        "n": col_name,
                        "p": obj_id,
                        "l": [0, 0], "a": 0, "d": 3,
                        "sx": make_sx_generic(NODE_TYPE["column"], col_name, file_id, "regex_col_ref"),
                        "h": sha16(col_name), "tc": token_est(col_name),
                        "ch": [], "im": [], "cl": [], "cb": [],
                    })

        # Add the main Procedure/Function node
        nodes.insert(0, {
            "id": obj_id,
            "t": node_type,
            "n": obj_name,
            "p": file_id,
            "l": [0, 0], "a": 0, "d": 2,
            "sx": make_sx_generic(node_type, obj_name, file_id, "regex_extract"),
            "h": sha16(obj_name), "tc": token_est(obj_name),
            "ch": [n["id"] for n in param_nodes],
            "im": [], "cl": [], "cb": [],
        })

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
        nodes = self.parser_registry.parsers.get(path.suffix, self.parse_python)(path, src)
        
        for node in nodes:
            self.add_node(node)
            self.add_edge(".", node["id"], EDGE_TYPE["contains"])
            self.nodes[self.node_idx["."]]["ch"].append(node["id"])
            self.topo += 1
            
            # Link extracted children (tables, rows, columns) to the file
            for child_id in node.get("ch", []):
                self.add_edge(node["id"], child_id, EDGE_TYPE["contains"])
                self.nodes[self.node_idx[node["id"]]]["ch"].append(child_id)
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
            resolved_calls = [] # Create a new list for actual IDs
            for call in node.get("cl", []):
                for target in self.sym_idx.get(call, []):
                    if src_id == target:
                        continue
                    resolved_calls.append(target) # Store the Node ID
                    trip = (src_id, target, EDGE_TYPE["calls"])
                    if trip not in seen:
                        self.edges.append([src_id, target, EDGE_TYPE["calls"]])
                        seen.add(trip)
            node["cl"] = list(set(resolved_calls)) # Replace strings with IDs

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
        print("[build_binary_memory] Generating embeddings with Qwen3.5-0.8B...")
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # 1. Configuration
            MODEL_NAME = "Qwen/Qwen3.5-0.8B"
            
            # Automatically detects GPU if available, otherwise uses CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[build_binary_memory] Loading {MODEL_NAME} on {device}...")
            
            # Load model and tokenizer (trust_remote_code is required for Qwen)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                MODEL_NAME, 
                device_map=device, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            model.eval()

            # 2. Collect texts from nodes
            texts = []
            node_ids = []
            for node in self.nodes:
                sx = node.get("sx", "")
                if sx:
                    texts.append(sx)
                    node_ids.append(node["id"])

            if not texts:
                print("[build_binary_memory] No content to embed.")
                return

            # 3. Batch Processing (Crucial for speed and memory)
            batch_size = 16 
            print(f"[build_binary_memory] Encoding {len(texts)} nodes in batches of {batch_size}...")

            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize inputs
                inputs = tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(device)

                # Inference
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                    # Mean Pooling for Sentence Embeddings:
                    # 1. Get last hidden states (token embeddings)
                    token_embeddings = outputs.last_hidden_state
                    
                    # 2. Apply attention mask to ignore padding tokens
                    input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                    
                    # 3. Sum and divide by length to get mean vector
                    embeddings = (token_embeddings * input_mask_expanded).sum(1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                all_embeddings.append(embeddings.cpu())

            # 4. Store results back into the builder
            idx = 0
            for nid in node_ids:
                vec = all_embeddings[idx].tolist()
                self.set_embedding(nid, vec)
                idx += 1
            
            print("[build_binary_memory] Embeddings generated successfully.")

        except Exception as e:
            import traceback
            print(f"[build_binary_memory] Warning: Could not generate embeddings: {e}")
            traceback.print_exc()




    def export_binary(self):
        os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
        
        # Prepare temporary storage for offsets
        content_offsets = []
        
        # 1. Stream Content to disk immediately (avoids massive RAM usage)
        with open(CONTENT_FILE, "wb") as content_f:
            for node in self.nodes:
                sx = node.get("sx", "")
                offset = content_f.tell()
                content_bytes = sx.encode("utf-8")
                content_f.write(content_bytes)
                content_offsets.append((offset, len(content_bytes)))
        
        # 2. Write Index
        with open(INDEX_FILE, "wb") as idx_f:
            # Header: Magic (4), Version (4), Count (4)
            idx_f.write(struct.pack('<III', MAGIC_NUMBER, 2, len(self.nodes)))
            
            for i, node in enumerate(self.nodes):
                nid = node["id"]
                pid = node.get("p", "")
                
                nid_bytes = nid.encode('utf-8')
                idx_f.write(struct.pack('<I', len(nid_bytes)))
                idx_f.write(nid_bytes)
                
                pid_bytes = pid.encode('utf-8')
                idx_f.write(struct.pack('<I', len(pid_bytes)))
                idx_f.write(pid_bytes)
                
                idx_f.write(struct.pack('<B', node["t"]))
                
                # Handle SQL node types
                if node["t"] in (NODE_TYPE["table"], NODE_TYPE["view"], NODE_TYPE["schema"]):
                    db_name = node.get("db", "")
                    db_bytes = db_name.encode('utf-8')
                    idx_f.write(struct.pack('<I', len(db_bytes)))
                    idx_f.write(db_bytes)
                    
                    columns = node.get("columns", [])
                    idx_f.write(struct.pack('<I', len(columns)))
                    for col in columns:
                        col_bytes = col.encode('utf-8')
                        idx_f.write(struct.pack('<I', len(col_bytes)))
                        idx_f.write(col_bytes)
                    
                    snippet = node.get("snippet", "")
                    snippet_bytes = snippet.encode('utf-8')
                    idx_f.write(struct.pack('<I', len(snippet_bytes)))
                    idx_f.write(snippet_bytes)
                    
                elif node["t"] == NODE_TYPE["column"]:
                    table_name = node.get("table", "")
                    table_bytes = table_name.encode('utf-8')
                    idx_f.write(struct.pack('<I', len(table_bytes)))
                    idx_f.write(table_bytes)
                    
                    dtype = node.get("dtype", "")
                    dtype_bytes = dtype.encode('utf-8')
                    idx_f.write(struct.pack('<I', len(dtype_bytes)))
                    idx_f.write(dtype_bytes)
                    
                    nullable = node.get("nullable", False)
                    idx_f.write(struct.pack('?', nullable))
                
                # Generic Edge Array
                node_edges = {}
                for s, t, e in self.edges:
                    if s == nid:
                        node_edges.setdefault(e, []).append(t)
                for child_id in node.get("ch", []):
                    node_edges.setdefault(EDGE_TYPE["contains"], []).append(child_id)

                total_edges = sum(len(v) for v in node_edges.values())
                idx_f.write(struct.pack('<I', total_edges))
                
                for edge_type, targets in node_edges.items():
                    for target_id in targets:
                        idx_f.write(struct.pack('<B', edge_type))
                        target_bytes = target_id.encode('utf-8')
                        idx_f.write(struct.pack('<I', len(target_bytes)))
                        idx_f.write(target_bytes)
                
                offset, length = content_offsets[i]
                idx_f.write(struct.pack('<Q', offset))
                idx_f.write(struct.pack('<I', length))

                vec = self.embeddings.get(nid, [])
                idx_f.write(struct.pack('<I', len(vec)))
                for val in vec:
                    idx_f.write(struct.pack('<f', val))

        print(f"[build_binary_memory] Saved {INDEX_FILE} and {CONTENT_FILE}")
        print(f"[build_binary_memory] nodes={len(self.nodes)} edges={len(self.edges)}")


if __name__ == "__main__":
    b = BinaryMemoryBuilder(ROOT_DIR)
    b.build()
    b.export_binary()