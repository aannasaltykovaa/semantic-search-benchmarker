"""BM25 and exact FAISS cosine retrieval (scores higher = better)."""
import re
from typing import Protocol
import numpy as np
from rank_bm25 import BM25Okapi
from src.models import DocumentChunk, SearchResult


def tokenize(text: str) -> list[str]:
    """Shared corpus/query tokenizer; retain underscore identifiers."""
    # Markdown escapes literal underscores; normalize their source representation.
    return re.findall(r"\w+", text.replace("\\_", "_").casefold())


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class LocalEmbedder:
    """Pool model-sized windows so large chunks are never silently truncated."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        windows, owners, weights = [], [], []
        tokenizer = self.model.tokenizer
        limit = self.model.max_seq_length-tokenizer.num_special_tokens_to_add(pair=False)
        for owner, text in enumerate(texts):
            ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
            for start in range(0, max(1, len(ids)), limit):
                part = ids[start:start+limit]
                windows.append(tokenizer.decode(part))
                owners.append(owner)
                weights.append(max(1, len(part)))
        vectors = self.model.encode(windows, normalize_embeddings=True, convert_to_numpy=True,
                                    batch_size=32, show_progress_bar=False)
        pooled = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
        for owner, weight, vector in zip(owners, weights, vectors):
            pooled[owner] += weight*vector
        return normalize(pooled)


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(lengths == 0):
        raise ValueError("Embeddings must be finite and nonzero")
    return np.ascontiguousarray(vectors/lengths)


class BM25Index:
    def __init__(self, chunks: list[DocumentChunk]):
        if not chunks:
            raise ValueError("Cannot index an empty corpus")
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(c.content) for c in chunks])

    def search(self, query: str, k: int) -> list[SearchResult]:
        if k < 1:
            raise ValueError("K must be positive")
        if not query.strip():
            return []
        scores = self.index.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.chunks[i].chunk_id))[:k]
        return [SearchResult(chunk=self.chunks[i], score=float(scores[i])) for i in order]


class DenseIndex:
    def __init__(self, chunks: list[DocumentChunk], embedder: Embedder):
        import faiss
        if not chunks:
            raise ValueError("Cannot index an empty corpus")
        self.chunks, self.embedder = chunks, embedder
        vectors = normalize(embedder.encode([c.content for c in chunks]))
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, k: int) -> list[SearchResult]:
        if k < 1:
            raise ValueError("K must be positive")
        if not query.strip():
            return []
        vector = normalize(self.embedder.encode([query]))
        # Retrieve all for deterministic tie-breaking on this small corpus.
        scores, indices = self.index.search(vector, len(self.chunks))
        rows = sorted(zip(scores[0], indices[0]), key=lambda row: (-row[0], self.chunks[row[1]].chunk_id))[:k]
        return [SearchResult(chunk=self.chunks[int(i)], score=float(s)) for s, i in rows]
