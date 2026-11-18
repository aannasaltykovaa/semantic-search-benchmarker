"""Source-evidence metrics keep labels independent of chunk size."""
import time
import numpy as np
import pandas as pd
from src.models import EvalQuery, SearchResult


def metrics(matches: list[set[int]], relevant_count: int, k: int) -> dict[str, float]:
    if k < 1 or relevant_count < 1:
        raise ValueError("K and relevant_count must be positive")
    ranks = [rank for rank, found in enumerate(matches, 1) if found]
    found = set().union(*matches[:k]) if matches[:k] else set()
    return {"hit_rate": float(bool(ranks)), "recall": len(found)/relevant_count,
            "mrr": 1/ranks[0] if ranks else 0.0}


def match_evidence(results: list[SearchResult], query: EvalQuery) -> list[set[int]]:
    return [{i for i, evidence in enumerate(query.evidence)
             if result.chunk.parent_doc == evidence.parent_doc and evidence.text in result.chunk.content}
            for result in results]


def validate_labels(queries: list[EvalQuery], documents: dict[str, str]) -> None:
    if len({q.query_id for q in queries}) != len(queries):
        raise ValueError("Query IDs must be unique")
    for q in queries:
        for evidence in q.evidence:
            if not evidence.text.strip() or documents.get(evidence.parent_doc, "").count(evidence.text) != 1:
                raise ValueError(f"Evidence must be a unique nonempty source span: {q.query_id}")


def benchmark(indexes: dict, queries: list[EvalQuery], ks=(3, 5, 10), repeats=3):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows = []
    for name, index in indexes.items():
        index.search(queries[0].query, max(ks))  # warm-up excluded
        for q in queries:
            timings = []
            for _ in range(repeats):
                start = time.perf_counter()
                results = index.search(q.query, max(ks))
                timings.append((time.perf_counter()-start)*1000)
            for k in ks:
                rows.append({"configuration": name, "query_id": q.query_id, "kind": q.kind,
                             "query": q.query, "k": k,
                             **metrics(match_evidence(results, q), len(q.evidence), k),
                             "latency_ms": float(np.mean(timings)), "timings_ms": timings})
    details = pd.DataFrame(rows)
    summaries = []
    for (name, k), group in details.groupby(["configuration", "k"], sort=False):
        times = [t for ts in group.timings_ms for t in ts]
        summaries.append({"configuration": name, "k": k,
                          **{metric: float(group[metric].mean()) for metric in ["hit_rate", "recall", "mrr"]},
                          "mean_ms": float(np.mean(times)), "p50_ms": float(np.percentile(times, 50)),
                          "p95_ms": float(np.percentile(times, 95))})
    return pd.DataFrame(summaries), details
