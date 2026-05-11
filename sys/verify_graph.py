import json
import re
import sqlite3
import argparse
import random
import os

NODE_TYPE = {
    0: "root", 1: "file", 2: "class", 3: "function",
    4: "async_function", 5: "table", 6: "column",
    7: "view", 8: "schema", 9: "database"
}

def verify_graph(db_path=".context-tree/pipeline.db", n_samples=5):
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Dangling Edges Check
    cur.execute("SELECT COUNT(*) FROM edges WHERE source NOT IN (SELECT id FROM nodes) OR target NOT IN (SELECT id FROM nodes)")
    dangling_count = cur.fetchone()[0]
    if dangling_count > 0:
        print(f"[ERROR] Found {dangling_count} dangling edges!")
    else:
        print("[OK] No dangling edges found.")

    # 2. Node Types Check
    cur.execute("SELECT id, data FROM nodes")
    type_counts = {}
    total_nodes = 0
    file_nodes = []

    for row in cur:
        total_nodes += 1
        data = json.loads(row[1])
        t = data.get("t", -1)
        t_name = NODE_TYPE.get(t, f"unknown_{t}")
        type_counts[t_name] = type_counts.get(t_name, 0) + 1
        if t == 1:
            file_nodes.append(data)

    print(f"\n[Info] Total Nodes: {total_nodes}")
    print("[Info] Node Type Distribution:")
    for t_name, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {t_name}: {count}")

    if type_counts.get("root", 0) == 0:
        print("[ERROR] Missing root node!")
    if type_counts.get("file", 0) == 0:
        print("[WARNING] No file nodes found. Did discovery fail?")

    # 3. Degree Sanity Check
    cur.execute('''
        SELECT n.id,
               (SELECT COUNT(*) FROM edges WHERE source = n.id) as out_degree,
               (SELECT COUNT(*) FROM edges WHERE target = n.id) as in_degree
        FROM nodes n
    ''')
    zero_degree = 0
    high_degree = 0
    for row in cur:
        nid, out_d, in_d = row
        total_d = out_d + in_d
        if total_d == 0 and nid != ".":
            zero_degree += 1
        if total_d > 5000:
            high_degree += 1

    print(f"\n[Info] Degree Sanity:")
    print(f"  - Nodes with 0 edges (isolated): {zero_degree}")
    print(f"  - Nodes with > 5000 edges (hub): {high_degree}")
    if high_degree > 0:
        print("[WARNING] Huge hub nodes detected. Check for runaway clustering.")

    # 4. Semantic Random Sampling
    print(f"\n[Semantic Check] Sampling {min(n_samples, len(file_nodes))} random file nodes:")
    samples = random.sample(file_nodes, min(n_samples, len(file_nodes)))
    for fnode in samples:
        fid = fnode["id"]
        print(f"  File: {fnode.get('n', fid)}")
        cur.execute("SELECT target FROM edges WHERE source = ? AND type = 1", (fid,))
        children = [r[0] for r in cur.fetchall()]
        print(f"    Children count: {len(children)}")
        for child in children[:5]:
            print(f"      -> {child}")
        if len(children) > 5:
            print(f"      ... and {len(children) - 5} more")


def run_baseline_regex(root_dir="sql_data"):
    print(f"\n[Baseline] Running regex scanner on raw files in '{root_dir}'...")
    if not os.path.exists(root_dir):
        print(f"  -> Directory '{root_dir}' not found. Skipping baseline.")
        return

    table_count = 0
    proc_count = 0

    table_pattern = re.compile(r"CREATE\s+TABLE", re.IGNORECASE)
    proc_pattern = re.compile(r"CREATE\s+(PROCEDURE|FUNCTION)", re.IGNORECASE)

    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".sql") or f.endswith(".html"):
                path = os.path.join(dirpath, f)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as file:
                        content = file.read()
                        table_count += len(table_pattern.findall(content))
                        proc_count += len(proc_pattern.findall(content))
                except Exception as e:
                    print(f"  -> Error reading {path}: {e}")

    print(f"  -> Regex found ~{table_count} CREATE TABLE statements.")
    print(f"  -> Regex found ~{proc_count} CREATE PROCEDURE/FUNCTION statements.")
    return table_count, proc_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=".context-tree/pipeline.db")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--root", default="sql_data")
    args = parser.parse_args()
    verify_graph(args.db, args.samples)
    run_baseline_regex(args.root)
