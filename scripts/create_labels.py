"""AI-proposed labels, written before benchmarking; review against the source."""
import json
from src.search import ROOT

SPECS = [
('exact', 'Which index supports equality and range comparisons such as <= and >=?', 'indexes-types', 'B-trees can handle equality and range queries on data that can be sorted into some ordering.'),
('exact', 'Can a Hash index handle range queries or only equality?', 'indexes-types', 'Hence, such indexes can only handle simple equality comparisons.'),
('exact', 'How do I index lower(col1) for case-insensitive comparisons?', 'indexes-expressional', 'CREATE INDEX test1_lower_col1_idx ON test1 (lower(col1));'),
('exact', 'What does the INCLUDE clause add to a covering index?', 'indexes-index-only-scans', 'This is done by adding an `INCLUDE` clause listing the extra columns.'),
('exact', 'Which index type is preferred for full text search, GIN or GiST?', 'textsearch-indexes', 'GIN indexes are the preferred text search index type.'),
('exact', 'Show the command pg_dump to write a database backup file.', 'backup-dump', 'pg_dump dbname > dumpfile'),
('exact', 'How does ANALYZE help the query planner?', 'sql-analyze', 'Subsequently, the query planner uses these statistics to help determine the most efficient execution plans for queries.'),
('exact', 'How do I turn on enable_partition_pruning?', 'ddl-partitioning', 'SET enable_partition_pruning = on;                 -- the default'),
('exact', 'Which PostgreSQL index supports ORDER BY sorted output?', 'indexes-ordering', 'only B-tree can produce sorted output'),
('exact', 'How does PostgreSQL resolve deadlocks?', 'explicit-locking', 'PostgreSQL automatically detects deadlock situations and resolves them by aborting one of the transactions involved, allowing the other(s) to complete.'),
('conceptual', 'How can I avoid indexing the majority of records that nobody searches for?', 'indexes-partial', 'One major reason for using a partial index is to avoid indexing common values.'),
('conceptual', 'How can the database recover from a crash without losing committed changes?', 'wal-intro', 'changes to data files (where tables and indexes reside) must be written only after those changes have been logged'),
('conceptual', 'How do I prevent a child record from referring to a nonexistent parent?', 'ddl-constraints', 'A foreign key constraint specifies that the values in a column (or a group of columns) must match the values appearing in some row of another table.'),
('conceptual', 'How can I reuse storage occupied by obsolete records?', 'routine-vacuuming', 'The standard form of `VACUUM` removes dead row versions in tables and indexes and marks the space available for future reuse.'),
('conceptual', 'Why is the binary representation of a document faster to query than its original text?', 'datatype-json', 'significantly faster to process, since no reparsing is needed.'),
('conceptual', 'How can I accelerate a lookup when uppercase and lowercase should be treated alike?', 'indexes-expressional', 'CREATE INDEX test1_lower_col1_idx ON test1 (lower(col1));'),
('conceptual', 'How can I read the answer straight from the lookup structure without visiting the underlying records?', 'indexes-index-only-scans', 'CREATE INDEX tab_x_y ON tab(x) INCLUDE (y);'),
('conceptual', 'What happens when two sessions each wait for a resource held by the other?', 'explicit-locking', 'PostgreSQL automatically detects deadlock situations and resolves them by aborting one of the transactions involved, allowing the other(s) to complete.'),
('conceptual', 'How can I export all my records into commands that recreate them later?', 'backup-dump', 'pg_dump dbname > dumpfile'),
('conceptual', 'How does the optimizer get information about the distribution of table contents?', 'sql-analyze', 'Subsequently, the query planner uses these statistics to help determine the most efficient execution plans for queries.'),
]


def main():
    rows = []
    for i, (kind, query, doc, span) in enumerate(SPECS, 1):
        text = (ROOT/'corpus'/f'{doc}.md').read_text()
        if text.count(span) != 1:
            raise ValueError(f'Label must uniquely match source: {doc}: {span}')
        rows.append(dict(query_id=f'q{i:02}', query=query, kind=kind,
                         evidence=[dict(parent_doc=f'{doc}.md', text=span)]))
    (ROOT/'data/eval_queries.json').write_text(json.dumps(rows, indent=2))
    print(f'Validated {len(rows)} unique evidence spans')


if __name__ == '__main__':
    main()
