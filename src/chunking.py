"""Token windows and Markdown sections with source offsets for evaluation."""
import hashlib
import re
import tiktoken
from src.models import DocumentChunk

ENCODING = tiktoken.get_encoding("cl100k_base")


def sections(text: str) -> list[tuple[int, int, list[str]]]:
    """Find heading boundaries, ignoring headings inside fenced code."""
    boundaries = []
    stack: list[tuple[int, str]] = []
    fence = None
    offset = 0
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            value = marker.group(1)
            if fence is None:
                fence = value
            elif (value[0] == fence[0] and len(value) >= len(fence)
                  and not line[marker.end():].strip()):
                fence = None
        elif fence is None:
            heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if heading:
                depth = len(heading.group(1))
                stack = [(level, title) for level, title in stack if level < depth]
                stack.append((depth, heading.group(2)))
                boundaries.append((offset, [title for _, title in stack]))
        offset += len(line)
    if not boundaries or boundaries[0][0] != 0:
        boundaries.insert(0, (0, ["Preamble"]))
    return [(start, boundaries[i+1][0] if i+1 < len(boundaries) else len(text), headers)
            for i, (start, headers) in enumerate(boundaries)]


def make_chunk(text: str, doc: str, start: int, end: int, strategy: str,
               headers: list[str], **extra) -> DocumentChunk:
    identity = hashlib.sha256(f"{doc}:{strategy}:{start}:{end}:{text[start:end]}".encode()).hexdigest()[:20]
    return DocumentChunk(chunk_id=identity, content=text[start:end], parent_doc=doc,
                         metadata={"strategy": strategy, "start_char": start, "end_char": end,
                                   "line_start": text.count("\n", 0, start)+1,
                                   "line_end": text.count("\n", 0, max(start, end-1))+1,
                                   "headers": headers,
                                   "token_count": len(ENCODING.encode(text[start:end])), **extra})


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32):
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Require 0 <= overlap < chunk_size")
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap

    def split(self, text: str, parent_doc: str) -> list[DocumentChunk]:
        tokens = ENCODING.encode(text)
        if not tokens:
            return []
        _, offsets = ENCODING.decode_with_offsets(tokens)
        offsets.append(len(text))
        result = []
        doc_sections = sections(text)
        for start in range(0, len(tokens), self.chunk_size-self.chunk_overlap):
            end = min(start+self.chunk_size, len(tokens))
            a, b = offsets[start], offsets[end]
            headers = [" / ".join(h) for x, y, h in doc_sections if a < y and b > x]
            result.append(make_chunk(text, parent_doc, a, b, f"fixed-{self.chunk_size}", headers,
                                     token_start=start, token_end=end, token_ids=tokens[start:end]))
            if end == len(tokens):
                break
        return result


class StructureAwareChunker:
    def split(self, text: str, parent_doc: str) -> list[DocumentChunk]:
        return [make_chunk(text, parent_doc, a, b, "structure", headers)
                for a, b, headers in sections(text) if text[a:b].strip()]
