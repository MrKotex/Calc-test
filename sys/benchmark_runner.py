"""
benchmark_runner.py
~~~~~~~~~~~~~~~~~~~
Runs the retrieval agent against benchmark_questions.json
and reports Hit@1, Hit@3, Path Hit, and MRR.

Usage:
    python sys/benchmark_runner.py \\
        --questions sys/benchmark_questions.json \\
        --index     .context-tree/memory_index.bin \\
        --content   .context-tree/memory_content.bin \\
        --top       5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from scout_agent_general import retrieve


def run(
    questions_path: str,
    index_path: str,
    content_path: str,
    top_k: int = 5,
) -> Dict:
    with open(questions_path, encoding="utf-8") as fh:
        questions: List[Dict] = json.load(fh)

    hit1 = hit3 = path_hit = mrr_sum = 0
    misses: List[Dict] = []

    for q in questions:
        query_text = q["q"]
        gold_set   = set(q["gold"])

        results    = retrieve(query_text, top_k=top_k,
                              index_path=index_path, content_path=content_path)
        top_ids    = [r["id"] for r in results]

        # Reciprocal rank
        rr = 0.0
        for rank, rid in enumerate(top_ids, start=1):
            if rid in gold_set:
                rr = 1.0 / rank
                break
        mrr_sum += rr

        # Hit@1
        if top_ids and top_ids[0] in gold_set:
            hit1 += 1

        # Hit@3
        if any(rid in gold_set for rid in top_ids[:3]):
            hit3 += 1

        # Path Hit — gold node's file path somewhere in top results
        gold_paths = {g.split("::")[0] for g in gold_set}
        if any(r["id"].split("::")[0] in gold_paths for r in results):
            path_hit += 1

        if not (top_ids and top_ids[0] in gold_set):
            misses.append({
                "id":   q["id"],
                "q":    query_text,
                "gold": list(gold_set),
                "top1": top_ids[0] if top_ids else "(no results)",
            })

    total = len(questions)

    print("\n" + "=" * 50)
    print(f"  Results  ({total} queries)")
    print("=" * 50)
    print(f"  Hit@1:    {hit1 / total:.2%}")
    print(f"  Hit@3:    {hit3 / total:.2%}")
    print(f"  Path Hit: {path_hit / total:.2%}")
    print(f"  MRR:      {mrr_sum / total:.4f}")

    if misses:
        print(f"\n  Misses ({len(misses)}):")
        for m in misses:
            print(f"    [{m['id']}] {m['q']!r}")
            print(f"         gold:  {m['gold']}")
            print(f"         top1:  {m['top1']}")

    return {
        "hit_at_1":  hit1 / total,
        "hit_at_3":  hit3 / total,
        "path_hit":  path_hit / total,
        "mrr":       mrr_sum / total,
        "misses":    misses,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="sys/benchmark_questions.json")
    ap.add_argument("--index",     default=".context-tree/memory_index.bin")
    ap.add_argument("--content",   default=".context-tree/memory_content.bin")
    ap.add_argument("--top",       type=int, default=5)
    args = ap.parse_args()

    run(
        questions_path=args.questions,
        index_path=args.index,
        content_path=args.content,
        top_k=args.top,
    )
