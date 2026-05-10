import json
import os
import sqlite3
from typing import Dict, Any, Optional

class PipelineSaver:
    """
    A pipeline save module that saves extracted data as it goes.
    It uses an SQLite backend for indexing nodes and edges to avoid
    storing massive amounts of data in RAM.
    Content is written to the SQLite database.
    """
    def __init__(self, output_dir: str, db_name: str = "pipeline.db"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.db_path = os.path.join(self.output_dir, db_name)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA journal_mode = MEMORY")

        self._setup_schema()

    def _setup_schema(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                data TEXT
            )
        """)
        c.execute("""
            CREATE TABLE edges (
                source TEXT,
                target TEXT,
                type INTEGER,
                UNIQUE(source, target, type)
            )
        """)
        c.execute("CREATE INDEX idx_edges_source ON edges(source)")
        c.execute("CREATE INDEX idx_edges_target ON edges(target)")
        self.conn.commit()

    def save_node(self, node: Dict[str, Any]):
        """Save a node to the pipeline."""
        self.conn.execute(
            "INSERT OR REPLACE INTO nodes (id, data) VALUES (?, ?)",
            (node["id"], json.dumps(node))
        )
        self.commit()

    def save_edge(self, source: str, target: str, edge_type: int):
        """Save an edge to the pipeline."""
        self.conn.execute(
            "INSERT OR IGNORE INTO edges (source, target, type) VALUES (?, ?, ?)",
            (source, target, edge_type)
        )
        self.commit()

    def update_node(self, node_id: str, node: Dict[str, Any]):
        """Update a node's metadata."""
        self.conn.execute("UPDATE nodes SET data = ? WHERE id = ?", (json.dumps(node), node_id))
        self.commit()

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute("SELECT data FROM nodes WHERE id = ?", (node_id,))
        row = cur.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def get_all_nodes(self):
        cur = self.conn.execute("SELECT data FROM nodes")
        for row in cur:
            yield json.loads(row[0])

    def get_edges(self, source=None, target=None, edge_type=None):
        query = "SELECT source, target, type FROM edges WHERE 1=1"
        params = []
        if source is not None:
            query += " AND source = ?"
            params.append(source)
        if target is not None:
            query += " AND target = ?"
            params.append(target)
        if edge_type is not None:
            query += " AND type = ?"
            params.append(edge_type)

        cur = self.conn.execute(query, params)
        for row in cur:
            yield row

    def get_node_count(self):
        cur = self.conn.execute("SELECT COUNT(*) FROM nodes")
        return cur.fetchone()[0]

    def get_edge_count(self):
        cur = self.conn.execute("SELECT COUNT(*) FROM edges")
        return cur.fetchone()[0]

    def commit(self):
        """Commit current database transaction."""
        self.conn.commit()

    def finalize(self):
        """Commit the db and we are done."""
        self.commit()
