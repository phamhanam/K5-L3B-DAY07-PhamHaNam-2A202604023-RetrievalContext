import argparse
import json
import os
import time
from pathlib import Path

from adapters import HybridRRFRetriever, LLMController, PgDenseRetriever, RerankRetriever
from agentic import run_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="starter/data/converted/questions.json")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output", default="starter/results/dev-dense")
    parser.add_argument("--strategy", required=True, help="Day 6 strategy name in Postgres chunks table")
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--method", choices=["dense", "hybrid", "rerank"], default="dense")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    split_questions = [q for q in questions if q.get("split") == args.split]

    if args.method == "dense":
        retriever = PgDenseRetriever(strategy=args.strategy)
    elif args.method == "hybrid":
        retriever = HybridRRFRetriever(strategy=args.strategy)
    elif args.method == "rerank":
        retriever = RerankRetriever(HybridRRFRetriever(strategy=args.strategy))
        
    controller = LLMController()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = []
    counts = {"answered": 0, "abstain": 0, "error": 0}

    for q in split_questions:
        qid = q.get("id") or q.get("question_id")
        qtext = q["question"]
        t0 = time.perf_counter()
        try:
            res = run_agent(
                qtext,
                controller,
                retriever,
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget,
            )
            status = res["status"]
            counts[status] = counts.get(status, 0) + 1
            traces.append({
                "id": qid,
                "split": args.split,
                "question": qtext,
                "answerable": q.get("answerable"),
                "gold_evidence": q.get("gold_evidence") or q.get("evidence"),
                "status": status,
                "answer": res["answer"],
                "citations": res["citations"],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "events": res["events"],
            })
        except Exception as exc:
            counts["error"] += 1
            traces.append({
                "id": qid,
                "question": qtext,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            })
        print(f"[{qid}] {traces[-1]['status']} ({traces[-1]['latency_ms']} ms)")

    (out_dir / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "split": args.split,
        "n": len(split_questions),
        "counts": counts,
        "max_searches": args.max_searches,
        "top_k": args.top_k,
        "budget": args.budget,
        "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved:", out_dir / "traces.json")


if __name__ == "__main__":
    main()
