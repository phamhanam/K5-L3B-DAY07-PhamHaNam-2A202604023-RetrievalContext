import argparse
import json
import statistics
import time
from pathlib import Path

from adapters import HybridRRFRetriever, PgDenseRetriever, RerankRetriever
from baseline import TfidfIndex, hit, pack_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval methods with baseline.py metrics")
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--method", required=True, choices=("tfidf", "dense", "hybrid", "rerank"))
    parser.add_argument("--strategy", required=True, help="Day 6 strategy name in Postgres")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    # Only answerable questions have gold pointers for Hit@k / Context Hit
    selected = [q for q in questions if q.get("split") == args.split and q.get("gold")]

    t0 = time.perf_counter()
    if args.method == "tfidf":
        tfidf = TfidfIndex(chunks)
        search_fn = lambda q, k: tfidf.search(q, top_k=k)
    elif args.method == "dense":
        r = PgDenseRetriever(strategy=args.strategy)
        search_fn = lambda q, k: r.search(q, filters=None, top_k=k)
    elif args.method == "hybrid":
        r = HybridRRFRetriever(strategy=args.strategy)
        search_fn = lambda q, k: r.search(q, filters=None, top_k=k)
    else:  # rerank
        r = RerankRetriever(HybridRRFRetriever(strategy=args.strategy))
        search_fn = lambda q, k: r.search(q, filters=None, top_k=k)
    index_seconds = time.perf_counter() - t0

    traces = []
    for q in selected:
        start = time.perf_counter()
        results = search_fn(q["question"], args.top_k)
        context, ids = pack_context(results, budget=args.budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        marked = [{**c, "hit": hit(c, q["gold"])} for c in results]
        traces.append({
            "query_id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "strategy": args.method,
            "top_k": args.top_k,
            "budget_chars": args.budget,
            "gold": q["gold"],
            "results": marked,
            "context": context,
            "context_ids": ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "hit1": bool(marked and marked[0]["hit"]),
            "hit3": any(c["hit"] for c in marked),
            "context_hit": any(c["hit"] and c["id"] in ids for c in marked),
        })

    summary = {
        "strategy": args.method,
        "split": args.split,
        "n": len(traces),
        "top_k": args.top_k,
        "budget": args.budget,
        "hit1": sum(t["hit1"] for t in traces),
        "hit3": sum(t["hit3"] for t in traces),
        "context_hit": sum(t["context_hit"] for t in traces),
        "avg_context_chars": statistics.mean(t["context_chars"] for t in traces),
        "median_ms": statistics.median(t["elapsed_ms"] for t in traces),
        "index_seconds": index_seconds,
        "service_cost_usd": 0,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    for name, obj in (("traces.json", traces), ("summary.json", summary)):
        (args.output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
