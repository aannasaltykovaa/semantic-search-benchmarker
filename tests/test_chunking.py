import pytest
from src.chunking import FixedSizeChunker, StructureAwareChunker, ENCODING


def test_exact_overlap_and_reconstruction():
    text = " ".join(f"word{i}" for i in range(150))
    chunks = FixedSizeChunker(40, 8).split(text, "example.md")
    rebuilt = chunks[0].metadata["token_ids"][:]
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.metadata["token_ids"][-8:] == current.metadata["token_ids"][:8]
        rebuilt.extend(current.metadata["token_ids"][8:])
    assert ENCODING.decode(rebuilt) == text
    assert all(c.content == text[c.metadata['start_char']:c.metadata['end_char']] for c in chunks)


def test_structure_preserves_fences_and_hierarchy():
    text = "# Guide\nIntro\n## SQL\n```sql\n# not a heading\nSELECT 1;\n```\n## Next\nDone\n"
    chunks = StructureAwareChunker().split(text, "guide.md")
    assert len(chunks) == 3
    assert "```sql\n# not a heading\nSELECT 1;\n```" in chunks[1].content
    assert chunks[1].metadata['headers'] == ['Guide', 'SQL']
    assert ''.join(c.content for c in chunks) == text


def test_empty_invalid_and_unicode():
    assert FixedSizeChunker().split('', 'empty.md') == []
    with pytest.raises(ValueError):
        FixedSizeChunker(10, 10)
    text = '数据库 café 🌍 ' * 50
    chunks = FixedSizeChunker(30, 8).split(text, 'unicode.md')
    assert all('�' not in c.content for c in chunks)
    assert chunks[0].metadata['start_char'] == 0
    assert chunks[-1].metadata['end_char'] == len(text)


def test_fence_with_suffix_is_not_a_closing_fence():
    text = '# Guide\n```text\n```python\n# still code\n```\n## Next\nDone\n'
    chunks = StructureAwareChunker().split(text, 'fences.md')
    assert len(chunks) == 2
    assert '# still code' in chunks[0].content
