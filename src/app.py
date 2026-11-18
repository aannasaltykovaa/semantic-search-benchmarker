"""Run with: python -m streamlit run src/app.py"""
import html
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from src.search import build_indexes
from src.benchmarker import benchmark
from src.indexer import tokenize

st.set_page_config(page_title='SearchWorkbench', page_icon='🔎', layout='wide')
st.title('SearchWorkbench')
st.caption('PostgreSQL documentation • Keyword versus semantic retrieval')
st.info('First load downloads a free local model and builds the indexes. No API key needed.')


@st.cache_resource
def resources():
    return build_indexes()


def highlighted(text, query):
    terms = set(tokenize(query))
    return ''.join('<mark>'+html.escape(part)+'</mark>' if part.casefold() in terms else html.escape(part)
                   for part in re.split(r'(\w+)', text))


try:
    with st.spinner('Loading documentation and search indexes…'):
        indexes, queries = resources()
except Exception as error:
    st.error(f'Could not initialize search: {error}')
    st.stop()

search_tab, benchmark_tab, about_tab = st.tabs(['Live search', 'Benchmark', 'How it works'])
with search_tab:
    dense_name = st.selectbox('Dense chunking strategy', list(indexes)[1:])
    k = st.slider('Results (K)', 1, 10, 5)
    query = st.text_input('Search the documentation', 'How can I reclaim space after deleting rows?')
    if st.button('Search', type='primary') and query.strip():
        for column, name in zip(st.columns(2), ['BM25 · Structure', dense_name]):
            with column:
                st.subheader(name)
                st.caption('Higher scores rank first. Scores from different engines are not directly comparable.')
                for rank, result in enumerate(indexes[name].search(query, k), 1):
                    chunk = result.chunk
                    with st.expander(f'{rank}. {chunk.parent_doc} · {result.score:.4f}', expanded=rank == 1):
                        st.caption(' / '.join(chunk.metadata['headers']))
                        st.caption(f"Lines {chunk.metadata['line_start']}–{chunk.metadata['line_end']} · {chunk.metadata['token_count']} tokens")
                        st.markdown('<pre style="white-space:pre-wrap">'+highlighted(chunk.content, query)+'</pre>', unsafe_allow_html=True)

with benchmark_tab:
    st.write(f'{len(queries)} labeled queries; K = 3, 5, 10; three timed runs per query.')
    st.caption('Labels are unique source-text evidence spans. A chunk must contain the complete evidence to count.')
    if st.button('Run benchmark', type='primary'):
        with st.spinner('Running all four configurations…'):
            st.session_state['benchmark'] = benchmark(indexes, queries)
    if 'benchmark' in st.session_state:
        summary, details = st.session_state['benchmark']
        cutoff = st.selectbox('Benchmark K', [3, 5, 10], index=1)
        selected = summary[summary.k == cutoff]
        st.dataframe(selected, hide_index=True, width='stretch')
        st.bar_chart(selected.set_index('configuration')[['hit_rate', 'recall', 'mrr']])
        st.bar_chart(selected.set_index('configuration')[['mean_ms']])
        st.download_button('Download benchmark CSV', summary.to_csv(index=False), 'benchmark_summary.csv', 'text/csv')
        st.subheader('Failure analysis')
        compare = st.selectbox('Compare baseline with', list(indexes)[1:], index=2)
        subset = details[details.k == cutoff]
        table = subset.pivot(index=['query_id', 'query', 'kind'], columns='configuration', values='mrr').reset_index()
        differences = table[table['BM25 · Structure'] != table[compare]]
        st.dataframe(differences[['query_id', 'query', 'kind', 'BM25 · Structure', compare]], hide_index=True)
        st.caption('A higher MRR means the first relevant result ranked earlier. Explanations below are hypotheses to check against the retrieved text.')
        for _, row in differences.iterrows():
            winner = 'BM25' if row['BM25 · Structure'] > row[compare] else 'Dense'
            explanation = ('Exact identifiers and repeated lexical terms may favor BM25.' if winner == 'BM25'
                           else 'Paraphrasing and related meanings may favor dense retrieval; chunk context also affects rankings.')
            with st.expander(f"{row['query_id']}: {winner} ranks relevant evidence earlier"):
                st.write(row['query'])
                st.write(explanation)
                target = next(q for q in queries if q.query_id == row['query_id'])
                st.write('Expected evidence:', [e.model_dump() for e in target.evidence])
                for name in ['BM25 · Structure', compare]:
                    results = indexes[name].search(row['query'], cutoff)
                    st.write(name, [{'document': r.chunk.parent_doc, 'score': r.score,
                                    'preview': r.chunk.content[:400]} for r in results])

with about_tab:
    st.markdown('''### Four configurations
- BM25 with complete Markdown sections.
- Dense search with 256-token windows and 32-token overlap.
- Dense search with 1024-token windows and 128-token overlap.
- Dense search with complete Markdown sections.

Embeddings use all-MiniLM-L6-v2 and a FAISS exact inner-product index. Both query and document vectors are normalized, so inner products are cosine similarities. Long chunks are embedded as model-sized windows and pooled, avoiding silent truncation.

**Hit Rate:** fraction of queries with any relevant result in the first K.
**Recall:** fraction of labeled evidence spans recovered, averaged over queries.
**MRR:** average reciprocal rank of the first relevant result; misses score zero.

This is a small, manually labeled classroom benchmark, not proof that one engine is universally better. Labels need human review. See README.md for methodology and limitations.''')
