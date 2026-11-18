import json
import platform
from datetime import datetime, timezone
from src.search import ROOT, build_indexes
from src.benchmarker import benchmark


def main():
    indexes, queries = build_indexes()
    summary, details = benchmark(indexes, queries)
    (ROOT/'reports').mkdir(exist_ok=True)
    summary.to_csv(ROOT/'reports/benchmark_summary.csv', index=False)
    details.to_json(ROOT/'reports/benchmark_details.json', orient='records', indent=2)
    (ROOT/'reports/run_metadata.json').write_text(json.dumps({
        'run_at': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
        'platform': platform.platform(), 'model': 'sentence-transformers/all-MiniLM-L6-v2',
        'vector_store': 'FAISS IndexFlatIP (exact)', 'repeats': 3,
        'chunk_counts': {name: len(index.chunks) for name,index in indexes.items()},
        'latency_scope': 'warm query processing including query embedding; excludes startup/index construction',
    }, indent=2))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
