"""Metric 1 - end-to-end search latency of the Pi Guard API (POST /query).

Each sample is the full HTTP round trip as seen by a client: CLIP text encoding,
ChromaDB vector search, llama.cpp re-ranking and frame preview extraction.

Usage:
    python benchmarks/search_latency.py --user admin --password secret --n 50 --label idle
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests

DEFAULT_QUERIES = [
    "a person walking",
    "a car parked in front of the house",
    "someone carrying a bag",
    "a dog in the yard",
    "a person wearing a red shirt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search latency benchmark (POST /query)")
    parser.add_argument("--url", default="http://localhost:3000")
    parser.add_argument("--user", default=os.environ.get("BENCH_USER"))
    parser.add_argument("--password", default=os.environ.get("BENCH_PASSWORD"))
    parser.add_argument("--n", type=int, default=50, help="measured requests")
    parser.add_argument("--warmup", type=int, default=3, help="discarded requests before measuring")
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES, help="queries used round-robin")
    parser.add_argument("--label", default="idle", help="scenario tag, e.g. idle / under_load")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", default=None, help="per-request CSV (default results/search_latency_<label>.csv)")

    args = parser.parse_args()

    if not args.user or not args.password:
        parser.error("--user/--password (or BENCH_USER/BENCH_PASSWORD) are required")

    return args


def login(session: requests.Session, url: str, user: str, password: str) -> None:
    response = session.post(f"{url}/login", json={"username": user, "password": password}, timeout=30)
    response.raise_for_status()

    token = response.json()["access_token"]
    session.headers["Authorization"] = f"Bearer {token}"


def run_query(session: requests.Session, url: str, text: str, timeout: float) -> dict:
    start = time.perf_counter()
    error = ""
    status = 0
    n_results = 0

    try:
        response = session.post(f"{url}/query", json={"text": text}, timeout=timeout)
        status = response.status_code

        body = response.json()
        n_results = len(body.get("results", []))

        if status != 200:
            error = str(body.get("detail", "http error"))
        elif body.get("error"):
            error = str(body["error"])
    except (requests.RequestException, ValueError) as e:
        error = str(e)

    latency_ms = (time.perf_counter() - start) * 1000

    return {"status": status, "latency_ms": latency_ms, "n_results": n_results, "error": error}


def summarize(latencies: list[float], failures: int) -> dict:
    summary = {"n": len(latencies), "failures": failures}

    if not latencies:
        return summary

    summary.update({
        "mean_ms": statistics.fmean(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "median_ms": statistics.median(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    })

    if len(latencies) > 1:
        cuts = statistics.quantiles(latencies, n=100, method="inclusive")
        summary["p90_ms"] = cuts[89]
        summary["p99_ms"] = cuts[98]

    return {k: round(v, 2) if isinstance(v, float) else v for k, v in summary.items()}


def main() -> int:
    args = parse_args()
    url = args.url.rstrip("/")
    out_path = Path(args.out or f"results/search_latency_{args.label}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    login(session, url, args.user, args.password)

    for i in range(args.warmup):
        sample = run_query(session, url, args.queries[i % len(args.queries)], args.timeout)
        print(f"[warmup {i + 1}/{args.warmup}] {sample['latency_ms']:.0f} ms")

    latencies: list[float] = []
    failures = 0
    empty = 0

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["i", "query", "status", "latency_ms", "n_results", "error"])
        writer.writeheader()

        for i in range(args.n):
            query = args.queries[i % len(args.queries)]
            sample = run_query(session, url, query, args.timeout)

            writer.writerow({"i": i + 1, "query": query, **sample, "latency_ms": round(sample["latency_ms"], 2)})
            f.flush()

            if sample["error"]:
                failures += 1
            else:
                latencies.append(sample["latency_ms"])
                empty += sample["n_results"] == 0

            flag = f" ERROR: {sample['error']}" if sample["error"] else ""
            print(f"[{i + 1}/{args.n}] {sample['latency_ms']:.0f} ms, {sample['n_results']} results{flag}")

    summary = {"label": args.label, **summarize(latencies, failures), "empty_results": empty}
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Search latency summary ===")
    for key, value in summary.items():
        print(f"{key:>10}: {value}")

    if empty:
        print(f"\nWARNING: {empty} requests returned no results; those skip the llama.cpp re-ranking,"
              " so their latency is not representative. Index some video before measuring.")

    if len(latencies) < 100:
        print(f"\nNote: with n={len(latencies)} the p99 is interpolated between the two slowest samples.")

    print(f"\nPer-request CSV: {out_path}\nSummary JSON:    {summary_path}")

    return 0 if latencies else 1


if __name__ == "__main__":
    sys.exit(main())
