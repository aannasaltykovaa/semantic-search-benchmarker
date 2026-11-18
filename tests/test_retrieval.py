import numpy as np
import pytest
from src.indexer import BM25Index, DenseIndex, LocalEmbedder, normalize, tokenize
from src.models import DocumentChunk


def chunks():
    return [DocumentChunk(chunk_id=str(i), parent_doc='test.md', content=text)
            for i, text in enumerate(['CUSTOM_CONFIG_FLAG_XYZ enables retries',
                                     'Cats nap quietly', 'Dogs run outdoors', 'Birds sing loudly'])]


def test_exact_identifier_and_case_punctuation():
    index = BM25Index(chunks())
    assert index.search('CUSTOM_CONFIG_FLAG_XYZ!', 1)[0].chunk.chunk_id == '0'
    assert index.search('CUSTOM_CONFIG_FLAG_XYZ!', 1)[0].score > 0
    assert tokenize('VACUUM!') == tokenize('vacuum')


def test_markdown_escaped_identifier_matches_plain_query():
    assert tokenize(r'pg\_dump') == tokenize('pg_dump')


def test_cosine_order_and_normalization():
    class FakeEmbedder:
        def encode(self, texts):
            return np.array([[10, 0] if text in ['query', 'CUSTOM_CONFIG_FLAG_XYZ enables retries']
                             else [0, 7] for text in texts], dtype=np.float32)
    index = DenseIndex(chunks(), FakeEmbedder())
    results = index.search('query', 10)
    assert len(results) == 4
    assert results[0].chunk.chunk_id == '0'
    assert results[0].score == pytest.approx(1)
    assert [r.chunk.chunk_id for r in results[1:]] == ['1', '2', '3']
    with pytest.raises(ValueError):
        normalize(np.zeros((1, 2)))


@pytest.mark.integration
def test_real_dense_zero_keyword_paraphrase():
    texts = ['An automobile requires gasoline.', 'Bananas grow in tropical regions.',
             'Musicians perform melodies on stage.', 'Snow covers mountain peaks.']
    query = 'Cars need fuel'
    assert set(tokenize(query)).isdisjoint(tokenize(texts[0]))
    docs = [DocumentChunk(chunk_id=str(i), parent_doc='fixture.md', content=t) for i,t in enumerate(texts)]
    assert DenseIndex(docs, LocalEmbedder()).search(query, 1)[0].chunk.chunk_id == '0'
