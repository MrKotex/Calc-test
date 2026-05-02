import json

with open(".context-tree/code_graph_ai.json", "r", encoding="utf-8") as f:
    g = json.load(f)

for n in g["nodes"]:
    print(n["id"])
