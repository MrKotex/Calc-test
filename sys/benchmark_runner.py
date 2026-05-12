import json
import time
from pathlib import Path
from scout_agent_binary import BinaryScoutAgent

QUESTIONS_FILE = "benchmark_questions.json"
OUT_FILE = ".context-tree/benchmark_results.json"

def hit_at_k(ranked_nodes, gold_nodes, k):
    top = [x[0] for x in ranked_nodes[:k]]
    return any(g in top for g in gold_nodes)

def path_hit(path_taken, gold_nodes):
    return any(g in path_taken for g in gold_nodes)

def main():
    # Initialize BinaryScoutAgent directly
    agent = BinaryScoutAgent(
        index_path=".context-tree/memory_index.bin",
        content_path=".context-tree/memory_content.bin"
    )

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    total_h1 = 0
    total_h3 = 0
    total_ph = 0
    total_duration = 0.0

    for item in questions:
        query = item["query"]
        gold = item["gold_nodes"]

        t0 = time.perf_counter()
        
        # Direct method calls instead of subprocess
        ranked = agent.rank_nodes(query, top_k=5)
        path = agent.navigate(query_text=query, start_node=".")
        prompt = agent.build_ai_prompt(path, query, top_k_ranked=6)
        
        dt = time.perf_counter() - t0

        h1 = hit_at_k(ranked, gold, 1)
        h3 = hit_at_k(ranked, gold, 3)
        ph = path_hit(path, gold)

        total_h1 += int(h1)
        total_h3 += int(h3)
        total_ph += int(ph)
        total_duration += dt

        # Add backwards compatibility for the output format
        results.append({
            "id": item["id"],
            "query": query,
            "gold_nodes": gold,
            "top_1": ranked[0][0] if ranked else None,
            "ranked_nodes": [[nid, float(score)] for nid, score in ranked],
            "path_taken": path,
            "hit@1": h1,
            "hit@3": h3,
            "path_hit": ph,
            "prompt_char_count": len(prompt),
            "duration_sec": dt,
            # Backwards compatibility
            "top_nodes": [[nid, float(score)] for nid, score in ranked],
        })

    summary = {
        "total_queries": len(questions),
        "hit@1_rate": total_h1 / len(questions) if questions else 0.0,
        "hit@3_rate": total_h3 / len(questions) if questions else 0.0,
        "path_hit_rate": total_ph / len(questions) if questions else 0.0,
        "avg_duration_sec": total_duration / len(questions) if questions else 0.0,
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
        "avg_duration_sec": summary["avg_duration_sec"],
        "out_file": OUT_FILE,
    }, indent=2))


if __name__ == "__main__":
    main()
