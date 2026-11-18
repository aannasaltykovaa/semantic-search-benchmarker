# AI assistance and audit log

## Provenance

This project was generated with ChatGPT using a supplied semantic-search assignment specification. The implementation uses PostgreSQL documentation, a free local embedding model, and FAISS. The findings and fixes below were produced during AI-assisted review.

Two genuine review findings are documented first. Earlier intentionally seeded regression examples are separately labeled and are not accidental discoveries.

## Actual review findings

### A. Markdown-escaped identifiers mismatch plain queries

The downloaded PostgreSQL Markdown contains identifiers such as `pg\_dump`, but a person types `pg_dump`. Even with one shared tokenizer, the backslash split the corpus identifier into two tokens while the query remained one token. This was discovered by inspecting the corpus after the first benchmark, not planted in advance.

The regression `test_markdown_escaped_identifier_matches_plain_query` failed. The fix normalizes Markdown-escaped underscores before tokenization. This is especially relevant to the exact-identifier part of the assignment. Source Markdown is kept unchanged for evidence-span matching.

### B. Incorrect closing code-fence recognition

Review of the structure parser found that any matching backtick prefix could close a code fence, even with a non-whitespace suffix. A line such as triple-backticks followed by `python` inside an already-open fence is not a valid closing fence. The old code closed the block anyway and incorrectly interpreted a later `#` line as a section heading, splitting the code.

The regression `test_fence_with_suffix_is_not_a_closing_fence` failed with three chunks instead of two. The fix also requires the remainder of a closing-fence line to be whitespace. These findings, their failing tests, and the fixes were produced during AI-assisted review.

Evidence: `audit/chunking_before_review.py`, `audit/indexer_before_review.py`, `reports/review_failures.txt` (two failures), `audit/review_fixes.diff`, and `reports/final_tests.txt`.

## Reusable baseline prompt

The following is a reconstruction of the implemented specification, not a verbatim additional user prompt:

> Build a modular Python semantic-search-benchmarker for PostgreSQL technical documentation. Compare BM25 with FAISS cosine search using all-MiniLM-L6-v2. Include 256-token windows with 32 overlap, 1024-token windows with 128 overlap, and header-aware sections that preserve fenced code. Use Pydantic models, source-text evidence labels, Hit Rate/Recall/MRR at 3/5/10, latency percentiles, a Streamlit comparator, failure analysis, tests, Docker, and transparent reporting. Avoid API-key requirements and silent embedding truncation.

## Earlier seeded regression examples (not accidental findings)

### 1. BM25 tokenization asymmetry

`audit/indexer_initial.py` lowercases corpus tokens, but passes `query.split()` to BM25. An uppercase identifier followed by punctuation receives a zero score. A rank-only assertion initially failed to detect this because tied zero scores happened to place the expected document first. The score assertion exposes the flaw.

Fix in `src/indexer.py`: use `tokenize(query)`, the same function used for document indexing. Test: `test_exact_identifier_and_case_punctuation` checks both rank and positive score.

### 2. MRR / Hit Rate outside top K

`audit/benchmarker_initial.py` searches the entire ranking for a relevant result. A relevant result at rank 3 incorrectly earns Hit Rate 1 and MRR 1/3 when K is 2.

Fix in `src/benchmarker.py`: enumerate only `matches[:k]`. Test: `test_top_k_cutoff` requires zero for a match outside K. Misses return zero, avoiding division by zero. Duplicate evidence in overlapping chunks is counted once for recall.

Evidence: `reports/initial_audit_tests.txt` has two failing tests; `reports/final_tests.txt` records the corrected suite. Exact before/after changes are in `audit/fixes.diff`.

## Additional design review

- Ground truth uses a document filename plus a unique source-text span. No labels contain chunk array positions. `validate_labels` rejects ambiguous/missing spans. The complete answer span must be contained in a result to receive credit.
- Both query and document vectors are normalized. FAISS IndexFlatIP returns inner-product similarity, sorted descending. It is exact search, not ANN. Synthetic vectors test ranking independently of the real-model integration test.
- A Markdown state machine ignores headings within backtick or tilde code fences. Structure-aware splitting preserves complete sections, including code blocks. Fixed windows intentionally may split a code block; this is a documented baseline limitation, not a promise that all chunkers preserve code.
- MiniLM has a short context limit. Long chunks are split into model-sized windows, embedded, and combined with token-weighted pooling before normalization. This avoids simply discarding the end of 1024-token chunks, but pooling can still dilute fine details.
- Twenty AI-proposed evaluation queries were authored before the benchmark. There are ten identifier/keyword queries and ten conceptual queries. They are not independently human-validated. No result-driven rewriting was used to force an engine to win.
- Current queries each have one labeled evidence span, so Recall@K equals Hit Rate@K. The metric implementation supports multiple spans and tests partial recall, but the provided dataset does not demonstrate that distinction.
- Ground truth is intentionally limited to the labeled passage. Equivalent answers elsewhere may be counted as nonrelevant; incomplete relevance judgments limit the interpretation of the results.
- Fixed windows store the exact tiktoken token intervals. Their text uses original source character boundaries to avoid replacement characters when a token boundary occurs inside a multi-byte character; in that edge case displayed text can differ slightly from the nominal token interval. ASCII overlap is verified exactly.
