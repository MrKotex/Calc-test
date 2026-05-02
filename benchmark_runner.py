import json
import subprocess
from pathlib import Path

QUESTIONS_FILE = "benchmark_questions.json"
RUN_RESULT_FILE = ".context-tree/run_result.json"
GRAPH_FILE = ".context-tree/code_graph_ai.json"
OUT_FILE = ".context-tree/benchmark_results.json"


def hit_at_k(ranked_nodes, gold_nodes, k):
    top = [x[0] for x in ranked_nodes[:k]]
    return any(g in top for g in gold_nodes)


def path_hit(path_taken, gold_nodes):
    return any(g in path_taken for g in gold_nodes)


def main():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    total_h1 = 0
    total_h3 = 0
    total_ph = 0

    for item in questions:
        query = item["query"]
        gold = item["gold_nodes"]

        cmd = [
            "python",
            "scout_agent-qwen.py",
            "--graph", GRAPH_FILE,
            "--query", query,
            "--save-run", RUN_RESULT_FILE
        ]
        subprocess.run(cmd, check=True)

        with open(RUN_RESULT_FILE, "r", encoding="utf-8") as f:
            run = json.load(f)

        ranked = run.get("ranked_nodes", [])
        path = run.get("path_taken", [])

        h1 = hit_at_k(ranked, gold, 1)
        h3 = hit_at_k(ranked, gold, 3)
        ph = path_hit(path, gold)

        total_h1 += int(h1)
        total_h3 += int(h3)
        total_ph += int(ph)

        results.append({
            "id": item["id"],
            "query": query,
            "gold_nodes": gold,
            "top_1": run.get("top_1"),
            "ranked_nodes": ranked,
            "path_taken": path,
            "hit@1": h1,
            "hit@3": h3,
            "path_hit": ph,
            "prompt_char_count": run.get("prompt_char_count"),
            "duration_sec": run.get("duration_sec"),
        })

    summary = {
        "total_queries": len(questions),
        "hit@1_rate": total_h1 / len(questions) if questions else 0.0,
        "hit@3_rate": total_h3 / len(questions) if questions else 0.0,
        "path_hit_rate": total_ph / len(questions) if questions else 0.0,
        "results": results,
    }

    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "total_queries": summary["total_queries"],
        "hit@1_rate": summary["hit@1_rate"],
        "hit@3_rate": summary["hit@3_rate"],
        "path_hit_rate": summary["path_hit_rate"],
        "out_file": OUT_FILE,
    }, indent=2))


if __name__ == "__main__":
    main()
