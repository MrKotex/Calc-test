"""
PLAN: Two-Phase AI-Oriented Indexing System
============================================
Phase 1 (Current): Pure extractor + graph indexer
- Parse .py, .html, and .sql files.
- Extract structured SQL + HTML information and relationships.
- Build compact binary index (no embeddings, no large models).
- Nodes use AI-friendly IDs: sql:db.schema.table, proc:db.schema.proc_name, html:path.html#block_n
- Metadata stored as compact JSON per node.

Phase 2 (Stub): Separate AI / embedding pipeline
- See sys/build_embeddings.py
"""
import ast
import threading
from concurrent.futures import ThreadPoolExecutor
from pipeline_save import PipelineSaver
import hashlib
import json
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

ROOT_DIR = "./sql_data"
OUTPUT_DIR = ".context-tree"
INDEX_FILE = os.path.join(OUTPUT_DIR, "memory_index.bin")
CONTENT_FILE = os.path.join(OUTPUT_DIR, "memory_content.bin")

EXCLUDED_DIRS = {
    ".git", ".context-tree", "__pycache__", ".venv", "venv",
    "node_modules", "dist", "build", "OLD", "sys",
}
INCLUDED_EXTENSIONS = {".html", ".py", ".sql"}

# Stable, AI-oriented node type ontology
NODE_TYPE = {
    "root": 0, "file": 1, "class": 2, "function": 3, "async_function": 4,
    "table": 5, "column": 6, "view": 7, "schema": 8, "database": 9,
    "html_block": 10, "param_node": 11,
}

# Edge types preserved for graph compatibility
EDGE_TYPE = {
    "contains": 1, "calls": 2, "imports": 3, "references": 4, "feeds": 5,
}

TYPE_TOKEN = {
    0: "RT", 1: "FL", 2: "CL", 3: "FN", 4: "AF", 5: "TB", 6: "CO",
    7: "VI", 8: "SC", 9: "DB", 10: "HB", 11: "PM",
}

MAGIC_NUMBER = 0x42494E4D  # "BINM"
CURRENT_FORMAT_VERSION = 2

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

def normalize_db_schema_obj(name: str) -> Tuple[str, str, str]:
    """Strip brackets and split into db, schema, obj."""
    name = name.strip("[]")
    parts = [p.strip() for p in name.split(".")]
    db = parts[0] if len(parts) >= 3 else ""
    schema = parts[1] if len(parts) >= 3 else (parts[0] if len(parts) == 2 else "")
    obj = parts[-1]
    return db, schema, obj

def safe_sql_id(raw_name: str, prefix: str) -> str:
    """Generate a safe, stable SQL ID by stripping path artifacts."""
    # Remove file path prefix artifacts (e.g., "./sql_data/filename.")
    clean = raw_name.replace("./sql_data/", "").replace(".html", "").replace(".", "_")
    db, schema, obj = normalize_db_schema_obj(clean)
    if db or schema:
        return f"{prefix}:{db}.{schema}.{obj}"
    return f"{prefix}:{obj}"

def make_sx(t: int, n: str, f: str, extra: str = "") -> str:
    """Compact signature for AI indexing."""
    return f"{TYPE_TOKEN.get(t, '??')}|N:{n}|F:{f}|X:{extra[:64]}"

def make_sx_generic(t: int, n: str, f: str, extra: str = "") -> str:
    return make_sx(t, n, f, extra)

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
        
        imports = []
        calls = []
        class V(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names: imports.append(alias.name)
                self.generic_visit(node)
            def visit_ImportFrom(self, node):
                mod = node.module or ""
                for alias in node.names: imports.append(f"{mod}.{alias.name}" if mod else alias.name)
                self.generic_visit(node)
            def visit_Call(self, n):
                name = None
                if isinstance(n.func, ast.Name): name = n.func.id
                elif isinstance(n.func, ast.Attribute): name = n.func.attr
                if name: calls.append(name)
                self.generic_visit(n)
        V().visit(tree)

        nodes.append({
            "id": rel_path(path, Path(".")),
            "t": NODE_TYPE["file"], "n": path.name, "p": ".",
            "l": [1, len(src.splitlines())], "a": 0, "d": 1,
            "sx": make_sx(NODE_TYPE["file"], path.name, rel_path(path, Path(".")), "py_mod"),
            "h": sha16(src), "tc": token_est(src), "ch": [], "im": imports, "cl": calls, "cb": [],
            "meta": {"object_type": "module", "lines": len(src.splitlines())}
        })
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
            "meta": {"source_file": path.name}
        })

        # 2. Extract Tables & Data (preserved for compatibility)
        for i, table in enumerate(soup.find_all('table')):
            table_name = table.get('id', f'unnamed_table_{i}')
            table_id = f"{file_id}::table.{table_name}"
            nodes.append({
                "id": table_id, "t": NODE_TYPE["table"], "n": table_name, "p": file_id,
                "l": [0, 0], "a": 0, "d": 2,
                "sx": make_sx_generic(NODE_TYPE["table"], table_name, file_id, "html_table"),
                "h": sha16(str(table)), "tc": token_est(str(table)), "ch": [], "im": [], "cl": [], "cb": [],
                "meta": {"source_table_id": table_name}
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
                    "meta": {"source_table": table_name}
                })
                table_children.append(col_id)

            rows = table.find_all('tr')
            for r_idx, row in enumerate(rows):
                row_id = f"{table_id}::row.{r_idx}"
                row_data = [td.get_text(strip=True) for td in row.find_all('td')]
                nodes.append({
                    "id": row_id, "t": NODE_TYPE["file"], "n": f"row_{r_idx}", "p": table_id,
                    "l": [0, 0], "a": 0, "d": 3,
                    "sx": make_sx_generic(NODE_TYPE["file"], f"row_{r_idx}", file_id, f"data:{','.join(row_data[:3])}..."),
                    "h": sha16(str(row)), "tc": token_est(str(row)), "ch": [], "im": [], "cl": [], "cb": [],
                })
                table_children.append(row_id)
            nodes[-1]["ch"] = table_children

        # 3. Extract SQL from <PRE> / <CODE> tags & create canonical SQL nodes
        block_counter = 1
        seen_tables = set()
        for tag in soup.find_all(['pre', 'code']):
            sql_text = tag.get_text(strip=True)
            if not sql_text: 
                continue
            clean_text = re.sub(r'/\*.*?\*/', '', sql_text, flags=re.DOTALL).strip()
            if not any(clean_text.upper().startswith(kw) for kw in ['CREATE', 'SELECT', 'UPDATE', 'INSERT', 'DROP', 'ALTER', 'GRANT', 'IF']):
                continue
            
            block_id = f"block_{block_counter}"
            block_id_full = f"html:{file_id}#{block_id}"
            block_counter += 1

            # Infer kind from nearby headings or context
            kind = "sql_example"
            prev_heading = tag.find_previous(['h1','h2','h3','h4','h5','h6'])
            if prev_heading:
                kind = prev_heading.get_text(strip=True).lower().replace(' ', '_')

            # Extract canonical SQL objects from this block
            sql_object_id = ""
            obj_nodes = []
            if 'PROCEDURE' in clean_text.upper() or 'FUNCTION' in clean_text.upper():
                proc_match = re.search(r'CREATE\s+(PROCEDURE|FUNCTION)\s+([^\s@\(]+)', clean_text, re.IGNORECASE)
                if proc_match:
                    obj_type = proc_match.group(1).lower()
                    obj_name = proc_match.group(2)
                    sql_object_id = safe_sql_id(obj_name, "proc")

                    # Extract parameters robustly (case-insensitive types)
                    param_pattern = r'@(\w+)\s+(VARCHAR|INT|DECIMAL|DATETIME|SMALLINT|BIT|NVARCHAR|CHAR|NCHAR|FLOAT|REAL|NUMERIC|MONEY|UNIQUEIDENTIFIER|TIMESTAMP|IMAGE|TEXT|VARBINARY|BINARY)\s*(=\s*[^,\n]+)?'
                    params = re.findall(param_pattern, clean_text, re.IGNORECASE)

                    # Extract referenced tables (bulletproof flattening)
                    raw_reads = []
                    raw_writes = []
                    for table_match in re.finditer(r'(?:FROM|JOIN|INSERT\s+INTO|UPDATE)\s+([A-Za-z0-9_.]+)', clean_text, re.IGNORECASE):
                        t_name = table_match.group(1)
                        if t_name.startswith('@') or t_name.upper() in ('SELECT', 'WHERE', 'SET', 'VALUES'): continue
                        t_id = safe_sql_id(t_name, "sql")
                        if re.search(rf'UPDATE\s+{re.escape(t_name)}', clean_text, re.IGNORECASE) or re.search(rf'INSERT\s+INTO\s+{re.escape(t_name)}', clean_text, re.IGNORECASE):
                            if t_id not in raw_writes: raw_writes.append(t_id)
                        else:
                            if t_id not in raw_reads: raw_reads.append(t_id)

                    obj_nodes.append({
                        "id": sql_object_id, "t": NODE_TYPE["function"], "n": obj_name, "p": file_id,
                        "l": [0, 0], "a": 0, "d": 2,
                        "sx": make_sx_generic(NODE_TYPE["function"], obj_name, file_id, f"sql_create_{obj_type}"),
                        "h": sha16(clean_text), "tc": token_est(clean_text), "ch": [], "im": [], "cl": [], "cb": [],
                        "meta": {
                            "db_name": "", "schema_name": "", "object_name": obj_name, "object_type": obj_type,
                            "params": [{"name": p[0], "type": p[1], "has_default": bool(p[2])} for p in params],
                            "reads_from": raw_reads, "writes_to": raw_writes
                        }
                    })
            elif 'TABLE' in clean_text.upper():
                table_match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_.]+)', clean_text, re.IGNORECASE)
                if table_match:
                    table_name = table_match.group(1)
                    sql_object_id = safe_sql_id(table_name, "sql")

                    # Extract columns from CREATE TABLE
                    col_pattern = r'(@?\w+)\s+(\w+)'
                    cols_str = re.search(rf'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{re.escape(table_name)}\s*\(([^)]+)\)', clean_text, re.IGNORECASE)
                    columns = []
                    if cols_str:
                        for col_match in re.finditer(col_pattern, cols_str.group(1)):
                            columns.append({"name": col_match.group(1), "type": col_match.group(2)})

                    obj_nodes.append({
                        "id": sql_object_id, "t": NODE_TYPE["table"], "n": table_name, "p": file_id,
                        "l": [0, 0], "a": 0, "d": 2,
                        "sx": make_sx_generic(NODE_TYPE["table"], table_name, file_id, "sql_create_table"),
                        "h": sha16(clean_text), "tc": token_est(clean_text), "ch": [], "im": [], "cl": [], "cb": [],
                        "meta": {"db_name": "", "schema_name": "", "object_name": table_name, "object_type": "table", "columns": columns}
                    })

            # Create nodes for referenced tables that weren't defined in this block
            for t_name in re.findall(r'(?:FROM|JOIN|INSERT\s+INTO|UPDATE)\s+([A-Za-z0-9_.]+)', clean_text, re.IGNORECASE):
                if t_name.startswith('@') or t_name.upper() in ('SELECT', 'WHERE', 'SET', 'VALUES'): continue
                t_id = safe_sql_id(t_name, "sql")
                if t_id not in seen_tables:
                    seen_tables.add(t_id)
                    # Extract columns from UPDATE SET, SELECT, and JOIN clauses
                    inferred_cols = set()
                    # UPDATE table SET col = ...
                    for col_match in re.finditer(rf'UPDATE\s+{re.escape(t_name)}\s+SET\s+([\w\s,=]+)', clean_text, re.IGNORECASE):
                        for c in col_match.group(1).split(','):
                            c = c.strip().split('=')[0].strip()
                            if c and not c.startswith('@') and not c.isdigit(): inferred_cols.add(c)
                    # SELECT col1, col2 FROM table
                    for col_match in re.finditer(rf'SELECT\s+([\w\s,]+)\s+FROM\s+{re.escape(t_name)}', clean_text, re.IGNORECASE):
                        for c in col_match.group(1).split(','):
                            c = c.strip().split('.')[0].strip()
                            if c and not c.startswith('@') and c.upper() not in ('TOP', 'DISTINCT', 'NULL') and not c.isdigit(): inferred_cols.add(c)
                    # JOIN table ON ... (extract columns from ON clause)
                    for col_match in re.finditer(rf'JOIN\s+{re.escape(t_name)}\s+ON\s+([\w\s.=<>!]+)', clean_text, re.IGNORECASE):
                        on_clause = col_match.group(1)
                        # Split by AND/OR to handle multiple conditions
                        for condition in re.split(r'\s+(?:AND|OR)\s+', on_clause, flags=re.IGNORECASE):
                            # Extract column names from table.column patterns
                            for col in re.finditer(r'(\w+)\.(\w+)', condition):
                                col_name = col.group(2)
                                if col_name and not col_name.startswith('@') and not col_name.isdigit() and col_name.upper() not in ('ON', 'AND', 'OR', 'WHERE', 'SET'):
                                    inferred_cols.add(col_name)
                            # Also extract standalone column names (without table prefix)
                            for col in re.finditer(r'(?<!\w)(\w+)(?!\w)', condition):
                                col_name = col.group(1)
                                if col_name and not col_name.startswith('@') and not col_name.isdigit() and col_name.upper() not in ('ON', 'AND', 'OR', 'WHERE', 'SET', 'NULL', 'IS', 'NOT', 'IN', 'BETWEEN', 'LIKE', 'EXISTS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS', 'SELECT', 'FROM', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'CROSS', 'FULL'):
                                    inferred_cols.add(col_name)

                    obj_nodes.append({
                        "id": t_id, "t": NODE_TYPE["table"], "n": t_name, "p": file_id,
                        "l": [0, 0], "a": 0, "d": 2,
                        "sx": make_sx_generic(NODE_TYPE["table"], t_name, file_id, "sql_referenced"),
                        "h": sha16(t_name), "tc": token_est(t_name), "ch": [], "im": [], "cl": [], "cb": [],
                        "meta": {
                            "db_name": "", "schema_name": "", "object_name": t_name, "object_type": "table",
                            "columns": [{"name": c, "type": "inferred", "nullable": True} for c in sorted(inferred_cols)] if inferred_cols else [],
                            "schema_source": "inferred_from_scripts" if inferred_cols else "external_reference"
                        }
                    })

            nodes.append({
                "id": block_id_full,
                "t": NODE_TYPE["html_block"],
                "n": kind,
                "p": file_id,
                "l": [0, 0], "a": 0, "d": 2,
                "sx": make_sx_generic(NODE_TYPE["html_block"], kind, file_id, f"sql_block_{block_id}"),
                "h": sha16(sql_text), "tc": token_est(sql_text), "ch": [], "im": [], "cl": [], "cb": [],
                "meta": {
                    "source_file": path.name,
                    "block_id": block_id,
                    "kind": kind,
                    "sql_object_id": sql_object_id,
                    "sql_sample": sql_text[:200]
                }
            })
            nodes.extend(obj_nodes)

        # 4. Fallback: scan full HTML text for SQL blocks
        try:
            full_text = soup.get_text("\n", strip=True)
        except Exception:
            full_text = ""
    
        if full_text:
            sql_block_pattern = re.compile(
                r"(?:/\*.*?\*/\s*)*"
                r"(?:IF\s+EXISTS[^\n]*\n)?"
                r".{0,200}?"
                r"CREATE\s+(PROCEDURE|FUNCTION|TABLE)\b"
                r"[\s\S]+?END\b",
                re.IGNORECASE
            )
            for match in sql_block_pattern.finditer(full_text):
                sql_text = match.group(0)
                block_id = f"block_{block_counter}"
                block_id_full = f"html:{file_id}#{block_id}"
                block_counter += 1

                obj_type = match.group(1).lower()
                obj_name_match = re.search(r'CREATE\s+(?:PROCEDURE|FUNCTION)\s+([^\s@]+)', sql_text, re.IGNORECASE)
                obj_name = obj_name_match.group(1) if obj_name_match else "unknown"
                sql_object_id = f"proc:{file_id.split('.')[0]}.{file_id.split('.')[-1]}.{obj_name}"

                nodes.append({
                    "id": block_id_full,
                    "t": NODE_TYPE["html_block"],
                    "n": f"example_{obj_type}_{obj_name}",
                    "p": file_id,
                    "l": [0, 0], "a": 0, "d": 2,
                    "sx": make_sx_generic(NODE_TYPE["html_block"], f"example_{obj_type}", file_id, f"fallback_{block_id}"),
                    "h": sha16(sql_text), "tc": token_est(sql_text), "ch": [], "im": [], "cl": [], "cb": [],
                    "meta": {
                        "source_file": path.name,
                        "block_id": block_id,
                        "kind": f"example_{obj_type}",
                        "sql_object_id": sql_object_id,
                        "sql_sample": sql_text[:200]
                    }
                })
        return nodes

    def parse_sql(self, path: Path, src: str, file_id: str = None, dialect: str = 'tsql') -> List[Dict]:
        nodes = []
        if not HAS_SQLGLOT:
            return self._extract_sql_details_regex(file_id, src)

        if file_id is None:
            file_id = rel_path(path, Path("."))
        
        try:
            parsed = sqlglot.parse(src, read=dialect)
            for expression in parsed:
                if isinstance(expression, sqlglot.exp.Create):
                    kind = expression.kind
                    this = expression.this
                    
                    if kind == 'TABLE':
                        table_name = this.name
                        db, schema, obj = normalize_db_schema_obj(table_name)
                        table_id = f"sql:{db}.{schema}.{obj}" if db or schema else f"sql:{obj}"
                        columns = []
                        for col in expression.find_all(sqlglot.exp.ColumnDef):
                            col_name = col.name
                            col_type = col.kind.name if col.kind else "UNKNOWN"
                            col_id = f"col:{table_name}.{col_name}"
                            columns.append({
                                "id": col_id, "t": NODE_TYPE["column"], "n": col_name, "p": table_id,
                                "l": [0, 0], "a": 0, "d": 3,
                                "sx": make_sx_generic(NODE_TYPE["column"], col_name, file_id, f"col:{col_type}"),
                                "h": sha16(str(col)), "tc": token_est(str(col)), "ch": [], "im": [], "cl": [], "cb": [],
                                "meta": {"dtype": col_type, "nullable": True, "table_name": table_name}
                            })
                            nodes.append(columns[-1])
                        table_node = {
                            "id": table_id, "t": NODE_TYPE["table"], "n": obj, "p": file_id,
                            "l": [0, 0], "a": 0, "d": 2,
                            "sx": make_sx_generic(NODE_TYPE["table"], obj, file_id, "sql_create_table"),
                            "h": sha16(str(expression)), "tc": token_est(str(expression)),
                            "ch": [c["id"] for c in columns], "im": [], "cl": [], "cb": [],
                            "meta": {"db_name": db, "schema_name": schema, "object_name": obj, "object_type": "table", "columns": columns}
                        }
                        nodes.insert(0, table_node)

                    elif kind == 'VIEW':
                        view_name = this.name
                        db, schema, obj = normalize_db_schema_obj(view_name)
                        view_id = f"sql:{db}.{schema}.{obj}" if db or schema else f"sql:{obj}"
                        nodes.append({
                            "id": view_id, "t": NODE_TYPE["view"], "n": obj, "p": file_id,
                            "l": [0, 0], "a": 0, "d": 2,
                            "sx": make_sx_generic(NODE_TYPE["view"], obj, file_id, "sql_view"),
                            "h": sha16(str(expression)), "tc": token_est(str(expression)), "ch": [], "im": [], "cl": [], "cb": [],
                            "meta": {"db_name": db, "schema_name": schema, "object_name": obj, "object_type": "view"}
                        })
                        for table in expression.find_all(sqlglot.exp.Table):
                            if table.name != view_name:
                                nodes.append({
                                    "id": f"{view_id}::feeds.{table.name}", "t": NODE_TYPE["table"],
                                    "n": f"feeds_{table.name}", "p": view_id,
                                    "l": [0, 0], "a": 0, "d": 3,
                                    "sx": make_sx_generic(NODE_TYPE["feeds"], f"feeds_{table.name}", file_id, "feeds_reference"),
                                    "h": sha16(str(table)), "tc": token_est(str(table)), "ch": [], "im": [], "cl": [], "cb": [],
                                    "meta": {"object_name": table.name, "object_type": "table"}
                                })
                                nodes.append(nodes[-1])

                    elif kind in ('PROCEDURE', 'FUNCTION'):
                        obj_type = kind.lower()
                        proc_name = this.name if hasattr(this, 'name') else str(this)
                        db, schema, obj = normalize_db_schema_obj(proc_name)
                        proc_id = f"proc:{db}.{schema}.{obj}" if db or schema else f"proc:{obj}"
                        nodes.append({
                            "id": proc_id, "t": NODE_TYPE["function"], "n": obj, "p": file_id,
                            "l": [0, 0], "a": 0, "d": 2,
                            "sx": make_sx_generic(NODE_TYPE["function"], obj, file_id, f"sql_create_{obj_type}"),
                            "h": sha16(str(expression)), "tc": token_est(str(expression)), "ch": [], "im": [], "cl": [], "cb": [],
                            "meta": {"db_name": db, "schema_name": schema, "object_name": obj, "object_type": obj_type}
                        })
                        # Trigger regex to get parameters and table refs for this procedure
                        nodes.extend(self._extract_sql_details_regex(file_id, src, proc_name, proc_id))
        except Exception as e:
            print(f"[build_binary_memory] Warning: sqlglot failed to parse SQL in {file_id}: {e}")

        # Fallback regex extraction
        regex_nodes = self._extract_sql_details_regex(file_id, src)
        nodes.extend(regex_nodes)
        
        return nodes

    def _extract_sql_details_regex(self, file_id: str, src: str, target_name: str = None, canonical_id: str = None) -> List[Dict]:
        nodes = []
        
        # 1. Extract CREATE TABLE statements
        table_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_.]+)\s*\(([^)]+)\)'
        for match in re.finditer(table_pattern, src, re.IGNORECASE):
            table_name = match.group(1)
            columns_str = match.group(2)
            db, schema, obj = normalize_db_schema_obj(table_name)
            table_id = f"sql:{db}.{schema}.{obj}" if db or schema else f"sql:{obj}"
            
            col_nodes = []
            for col_line in columns_str.split(','):
                col_line = col_line.strip()
                if not col_line: continue
                col_match = re.match(r'(@?\w+)\s+(\w+)', col_line)
                if col_match:
                    col_name = col_match.group(1)
                    col_type = col_match.group(2)
                    col_id = f"col:{table_name}.{col_name}"
                    col_nodes.append({
                        "id": col_id, "t": NODE_TYPE["column"], "n": col_name, "p": table_id,
                        "l": [0, 0], "a": 0, "d": 3,
                        "sx": make_sx_generic(NODE_TYPE["column"], col_name, file_id, f"col:{col_type}"),
                        "h": sha16(col_name), "tc": token_est(col_name), "ch": [], "im": [], "cl": [], "cb": [],
                        "meta": {"dtype": col_type, "nullable": True, "table_name": table_name}
                    })
                    nodes.append(col_nodes[-1])
            
            nodes.insert(0, {
                "id": table_id, "t": NODE_TYPE["table"], "n": obj, "p": file_id,
                "l": [0, 0], "a": 0, "d": 2,
                "sx": make_sx_generic(NODE_TYPE["table"], obj, file_id, "regex_create_table"),
                "h": sha16(table_name), "tc": token_est(table_name),
                "ch": [c["id"] for c in col_nodes], "im": [], "cl": [], "cb": [],
                "meta": {"db_name": db, "schema_name": schema, "object_name": obj, "object_type": "table", "columns": col_nodes}
            })

        # 2. Extract Procedure/Function Name
        proc_match = re.search(r'CREATE\s+(PROCEDURE|FUNCTION)\s+([^\s@]+)', src, re.IGNORECASE)
        if not proc_match:
            return nodes
            
        obj_type = proc_match.group(1)
        obj_name = proc_match.group(2)
        
        if target_name and obj_name != target_name:
            return nodes 

        db, schema, obj = normalize_db_schema_obj(obj_name)
        obj_id = canonical_id if canonical_id else (f"proc:{db}.{schema}.{obj}" if db or schema else f"proc:{obj}")
        node_type = NODE_TYPE["function"]

        # 3. Extract Parameters
        param_pattern = r'@(\w+)\s+(VARCHAR|INT|DECIMAL|DATETIME|SMALLINT|BIT|NVARCHAR|CHAR|NCHAR|FLOAT|REAL|NUMERIC|MONEY|UNIQUEIDENTIFIER|TIMESTAMP|IMAGE|TEXT|VARBINARY|BINARY)\s*(=\s*[^,\n]+)?'
        params = re.findall(param_pattern, src)
        
        param_nodes = []
        for p_name, p_type, p_default in params:
            param_id = f"{obj_id}::param.{p_name}"
            param_nodes.append({
                "id": param_id, "t": NODE_TYPE["param_node"], "n": f"@{p_name}", "p": obj_id,
                "l": [0, 0], "a": 0, "d": 3,
                "sx": make_sx_generic(NODE_TYPE["param_node"], f"@{p_name}", file_id, f"param:{p_type}"),
                "h": sha16(p_name), "tc": token_est(p_name), "ch": [], "im": [], "cl": [], "cb": [],
                "meta": {"param_name": p_name, "param_type": p_type, "has_default": bool(p_default)}
            })
            nodes.append(param_nodes[-1])

        # 4. Extract Referenced Tables
        table_pattern = r'(?:FROM|JOIN|INSERT\s+INTO|UPDATE)\s+([A-Za-z0-9_.]+)'
        tables = re.findall(table_pattern, src)
        reads_from = []
        writes_to = []
        for table_name in tables:
            if table_name.startswith('@') or table_name.upper() in ('SELECT', 'WHERE', 'SET', 'VALUES'):
                continue
            t_db, t_schema, t_obj = normalize_db_schema_obj(table_name)
            t_id = f"sql:{t_db}.{t_schema}.{t_obj}" if t_db or t_schema else f"sql:{t_obj}"
            # Simple heuristic for writes
            if re.search(rf'UPDATE\s+{re.escape(table_name)}', src, re.IGNORECASE) or re.search(rf'INSERT\s+INTO\s+{re.escape(table_name)}', src, re.IGNORECASE):
                writes_to.append(t_id)
            else:
                reads_from.append(t_id)
            nodes.append({
                "id": t_id, "t": NODE_TYPE["table"], "n": t_obj, "p": file_id,
                "l": [0, 0], "a": 0, "d": 2,
                "sx": make_sx_generic(NODE_TYPE["table"], t_obj, file_id, "regex_table_ref"),
                "h": sha16(table_name), "tc": token_est(table_name), "ch": [], "im": [], "cl": [], "cb": [],
                "meta": {"db_name": t_db, "schema_name": t_schema, "object_name": t_obj, "object_type": "table"}
            })

        # Add the main Procedure/Function node
        nodes.insert(0, {
            "id": obj_id, "t": node_type, "n": obj, "p": file_id,
            "l": [0, 0], "a": 0, "d": 2,
            "sx": make_sx_generic(node_type, obj, file_id, "regex_extract"),
            "h": sha16(obj_name), "tc": token_est(obj_name), "ch": [n["id"] for n in param_nodes], "im": [], "cl": [], "cb": [],
            "meta": {
                "db_name": db, "schema_name": schema, "object_name": obj, "object_type": obj_type.lower(),
                "params": [{"name": p[0], "type": p[1], "has_default": bool(p[2])} for p in params],
                "reads_from": reads_from, "writes_to": writes_to, "uses_functions": []
            }
        })
        return nodes

class BinaryMemoryBuilder:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir).resolve()
        self.sym_idx: Dict[str, List[str]] = {}
        self.imp_idx: Dict[str, str] = {}
        self.topo = 0
        self.embeddings = {}
        self.parser_registry = ParserRegistry()
        self.saver = PipelineSaver(OUTPUT_DIR)
        self.builder_lock = threading.Lock()

    def set_embedding(self, node_id: str, vector: List[float]):
        self.embeddings[node_id] = vector

    def add_node(self, obj: Dict):
        self.saver.save_node(obj)

    def add_edge(self, s: str, t: str, e: int):
        self.saver.save_edge(s, t, e)

    def discover(self) -> List[Path]:
        out = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix in INCLUDED_EXTENSIONS or p.suffix == '.py':
                    out.append(p)
        out.sort()
        return out

    def build(self, max_workers: int = 4):
        files = self.discover()

        # Log file extensions
        ext_counts = {}
        for f in files:
            ext_counts[f.suffix] = ext_counts.get(f.suffix, 0) + 1
        print(f"[Builder] Discovered {len(files)} files. Extensions: {ext_counts}")

        # Add root node
        self.add_node({
            "id": ".", "t": NODE_TYPE["root"], "n": "root", "p": "",
            "l": [0, 0], "a": 0, "d": 0, "ti": self.topo,
            "sx": "RT|N:root|F:.|S:-|A:0|L:0-0|D:0|O:-",
            "h": sha16("root"), "tc": 1, "ch": [], "im": [], "cl": [], "cb": [],
            "meta": {}
        })
        self.topo += 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.process_file, files)

        self.resolve_import_edges()
        self.resolve_call_edges()
        self.compute_called_by()
        self.compute_depths()
        # Phase 1 does not generate embeddings. Phase 2 handles this via sys/build_embeddings.py

    def process_file(self, path: Path):
        rp = rel_path(path, self.root)
        src = read_text(path)
        nodes = self.parser_registry.parsers.get(path.suffix, self.parse_python)(path, src)
        
        for node in nodes:
            self.add_node(node)
            self.add_edge(".", node["id"], EDGE_TYPE["contains"])

            # Update root node to include child
            root_node = self.saver.get_node(".")
            if root_node:
                with self.builder_lock:
                    # In a highly concurrent scenario, another thread might have updated it
                    # So we fetch again under lock or just update the list
                    current_root = self.saver.get_node(".")
                    if current_root:
                        current_root.setdefault("ch", []).append(node["id"])
                        self.saver.update_node(".", current_root)

            with self.builder_lock:
                self.topo += 1
            
            for child_id in node.get("ch", []):
                self.add_edge(node["id"], child_id, EDGE_TYPE["contains"])
                with self.builder_lock:
                    self.topo += 1
            
            if path.suffix == '.py':
                with self.builder_lock:
                    self.imp_idx[path.stem] = node["id"]
                    # Store calls for resolution
                    for call in node.get("cl", []):
                        self.sym_idx.setdefault(call, []).append(node["id"])

    def parse_python(self, path: Path, src: str) -> List[Dict]:
        return self.parser_registry.parse_python(path, src)

    def resolve_import_edges(self):
        for node in self.saver.get_all_nodes():
            src_id = node["id"]
            for imp in node.get("im", []):
                base = imp.split(".")[0]
                if base in self.imp_idx:
                    self.add_edge(src_id, self.imp_idx[base], EDGE_TYPE["imports"])

    def resolve_call_edges(self):
        seen: Set[Tuple[str, str, int]] = set()
        for node in self.saver.get_all_nodes():
            src_id = node["id"]
            resolved_calls = []
            for call in node.get("cl", []):
                for target in self.sym_idx.get(call, []):
                    if src_id == target: continue
                    resolved_calls.append(target)
                    trip = (src_id, target, EDGE_TYPE["calls"])
                    if trip not in seen:
                        self.saver.save_edge(src_id, target, EDGE_TYPE["calls"])
                        seen.add(trip)
            node["cl"] = list(set(resolved_calls))
            self.saver.update_node(src_id, node)

    def compute_called_by(self):
        # Initialize cb
        nodes_to_update = []
        for n in self.saver.get_all_nodes():
            n["cb"] = []
            nodes_to_update.append(n)
        for n in nodes_to_update:
            self.saver.update_node(n["id"], n)

        # Collect edges before mutating the nodes
        call_edges = list(self.saver.get_edges(edge_type=EDGE_TYPE["calls"]))
        for s, t, e in call_edges:
            target_node = self.saver.get_node(t)
            if target_node:
                target_node.setdefault("cb", []).append(s)
                self.saver.update_node(t, target_node)

    def compute_depths(self):
        q = ["."]
        seen = {"."}
        while q:
            cur = q.pop(0)
            cur_node = self.saver.get_node(cur)
            if not cur_node: continue
            cur_d = cur_node.get("d", 0)

            contain_edges = list(self.saver.get_edges(source=cur, edge_type=EDGE_TYPE["contains"]))
            for s, t, e in contain_edges:
                if t not in seen:
                    target_node = self.saver.get_node(t)
                    if target_node:
                        target_node["d"] = cur_d + 1
                        self.saver.update_node(t, target_node)
                        q.append(t)
                        seen.add(t)

    def generate_embeddings(self):
        print("[build_binary_memory] Generating embeddings with sentence-transformers...")
        # Since we are using pipeline saver we iterate over self.saver.get_all_nodes()
        # for node in self.saver.get_all_nodes():
        #    ...
        return




    def export_binary(self):
        self.saver.finalize()
        os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
        content_offsets = []
        
        print("[Builder] Starting streaming export...")

        # 1. Stream Content (sx signatures) to disk immediately
        num_nodes_total = self.saver.get_node_count()
        with open(CONTENT_FILE, "wb") as content_f:
            for j, node in enumerate(self.saver.get_all_nodes()):
                sx = node.get("sx", "")
                offset = content_f.tell()
                content_bytes = sx.encode("utf-8")
                content_f.write(content_bytes)
                content_offsets.append((offset, len(content_bytes)))

                # Progress every 100 nodes
                if (j + 1) % 100 == 0:
                    print(f"\r[Builder] Content: {j+1}/{num_nodes_total} nodes", end="", flush=True)
        print(f"\r[Builder] Content: {num_nodes_total}/{num_nodes_total} nodes done\n", flush=True)
        
        # 2. Write Index incrementally
        with open(INDEX_FILE, "wb") as idx_f:
            # Header: Magic (4), Version (4), Count (4)
            num_nodes = self.saver.get_node_count()
            idx_f.write(struct.pack('<III', MAGIC_NUMBER, 2, num_nodes))
            
            i = -1
            for i, node in enumerate(self.saver.get_all_nodes()):
                nid = node["id"]
                pid = node.get("p", "")
                
                nid_bytes = nid.encode('utf-8')
                idx_f.write(struct.pack('<I', len(nid_bytes)))
                idx_f.write(nid_bytes)
                
                pid_bytes = pid.encode('utf-8')
                idx_f.write(struct.pack('<I', len(pid_bytes)))
                idx_f.write(pid_bytes)
                
                idx_f.write(struct.pack('<B', node["t"]))
                
                # Handle SQL node types (reverted back to original from the new version we somehow grabbed)
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
                for s, t, e in self.saver.get_edges(source=nid):
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
                
                offset_t, length_t = content_offsets[i]
                idx_f.write(struct.pack('<Q', offset_t))
                idx_f.write(struct.pack('<I', length_t))

                # Write 0 for embedding vectors to match binary format expectation
                idx_f.write(struct.pack('<I', 0))

                # Progress every 100 nodes
                if (i + 1) % 100 == 0:
                    print(f"\r[Builder] Index: {i+1}/{num_nodes} nodes", end="", flush=True)
        try:
            if i >= 0:
                print(f"\r[Builder] Index: {num_nodes}/{num_nodes} nodes done\n", flush=True)
        except NameError:
            pass

        print(f"[build_binary_memory] Saved {INDEX_FILE} and {CONTENT_FILE}")
        print(f"[build_binary_memory] nodes={self.saver.get_node_count()} edges={self.saver.get_edge_count()}")

        # Output Regression Snapshot
        stats = {
            "total_nodes": self.saver.get_node_count(),
            "total_edges": self.saver.get_edge_count(),
            "nodes_per_type": {},
            "avg_degree": self.saver.get_edge_count() / self.saver.get_node_count() if self.saver.get_node_count() > 0 else 0
        }

        for node in self.saver.get_all_nodes():
            t = str(node.get("t", "unknown"))
            stats["nodes_per_type"][t] = stats["nodes_per_type"].get(t, 0) + 1

        stats_file = os.path.join(os.path.dirname(INDEX_FILE), "build_stats.json")
        with open(stats_file, "w") as sf:
            json.dump(stats, sf, indent=2)
        print(f"[build_binary_memory] Regression snapshot saved to {stats_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1: Binary Index Builder")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    args = parser.parse_args()

    b = BinaryMemoryBuilder(ROOT_DIR)
    b.build(max_workers=args.workers if args.workers is not None else 4)
    b.export_binary()
