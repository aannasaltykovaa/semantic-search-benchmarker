"""Build four configurations from exactly the same source documents."""
import json
from pathlib import Path
from src.chunking import FixedSizeChunker, StructureAwareChunker
from src.indexer import BM25Index, DenseIndex, LocalEmbedder
from src.models import EvalQuery
from src.benchmarker import validate_labels

ROOT = Path(__file__).resolve().parents[1]


def load_data():
    documents = {p.name: p.read_text(encoding='utf-8') for p in sorted((ROOT/'corpus').glob('*.md'))}
    if len(documents) < 20:
        raise ValueError('Expected at least 20 corpus pages. Run python -m scripts.download_corpus.')
    queries = [EvalQuery.model_validate(q) for q in json.loads((ROOT/'data/eval_queries.json').read_text())]
    validate_labels(queries, documents)
    return documents, queries


def build_indexes():
    documents, queries = load_data()
    groups = {}
    for name, chunker in [('256 tokens', FixedSizeChunker(256, 32)),
                          ('1024 tokens', FixedSizeChunker(1024, 128)),
                          ('Structure', StructureAwareChunker())]:
        groups[name] = [chunk for doc, text in documents.items() for chunk in chunker.split(text, doc)]
    embedder = LocalEmbedder()
    indexes = {'BM25 · Structure': BM25Index(groups['Structure'])}
    for name, chunks in groups.items():
        indexes[f'Dense · {name}'] = DenseIndex(chunks, embedder)
    return indexes, queries
