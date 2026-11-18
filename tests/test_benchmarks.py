import pytest
from src.benchmarker import metrics, match_evidence, validate_labels
from src.models import EvalQuery, Evidence, DocumentChunk, SearchResult


def test_metric_math():
    assert metrics([set(), {0}, {1}], 2, 3) == {'hit_rate': 1.0, 'recall': 1.0, 'mrr': .5}
    assert metrics([set(), set()], 1, 2) == {'hit_rate': 0.0, 'recall': 0.0, 'mrr': 0.0}
    assert metrics([{0}, {0}], 2, 2)['recall'] == .5


def test_top_k_cutoff():
    assert metrics([set(), set(), {0}], 1, 2) == {'hit_rate': 0.0, 'recall': 0.0, 'mrr': 0.0}


def test_labels_follow_source_not_chunk_number():
    q = EvalQuery(query_id='q', query='why', kind='conceptual', evidence=[Evidence(parent_doc='a.md', text='answer')])
    def result(identity, text, doc='a.md'):
        return SearchResult(chunk=DocumentChunk(chunk_id=identity, content=text, parent_doc=doc), score=1)
    assert match_evidence([result('different-id', 'longer answer text')], q) == [{0}]
    assert match_evidence([result('same-id', 'wrong text')], q) == [set()]
    assert match_evidence([result('other-doc', 'answer', 'b.md')], q) == [set()]
    with pytest.raises(ValueError):
        validate_labels([q], {'a.md': 'answer and answer'})
