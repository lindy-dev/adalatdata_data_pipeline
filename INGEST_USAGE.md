# Ingest Usage

This file documents how to run `ingest.py`, inspect the database, test vector search, and reset the data.

The pipeline uses PostgreSQL with three layers:
- **Relational tables** for structured metadata and entity lookups
- **pgvector** for semantic similarity search on judgment embeddings
- **AGE (Apache AGE)** for graph traversal via Cypher queries

Defaults used by the code:

- Database: `adalatdata` (PostgreSQL)
- AGE graph name: `adalatdata`
- Embedding model: `text-embedding-qwen3-embedding-0.6b`
- Vector dimensions: `1024`
- Embedding endpoint: `http://localhost:1234/v1/embeddings`
- JSON input: `/home/shreyas/Documents/02_Adalatdata/data_pipeline/extracted_jsons`
- Markdown input: `/home/shreyas/Documents/02_Adalatdata/data_pipeline/extracted_mds`

## Prerequisites

1. Start PostgreSQL with the `vector` and `age` extensions enabled (Docker or system install).
2. Set environment variables from `.env.example`:

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=adalatdata
export DB_USER=postgres
export DB_PASS=your_password_here
```

Or place them in a `.env` file in the project root (loaded automatically via `python-dotenv`).

3. Initialize the database schema:

```bash
uv run python init_db.py
```

This creates all tables, indexes, triggers, and the AGE graph. It is idempotent -- safe to run multiple times.

## Run The Ingest Script

Run a small smoke test with one file:

```bash
uv run python ingest.py --limit 1
```

Run the full ingest:

```bash
uv run python ingest.py
```

Run with semantic demo search after ingestion:

```bash
uv run python ingest.py --demo-search "bail conditions in criminal cases"
```

Run with explicit input directories:

```bash
uv run python ingest.py \
  --json-dir /media/shreyas/E/Adalatdata_data/extracted_jsons \
  --md-dir /media/shreyas/E/Adalatdata_data/extracted_mds
```

## Query Relational Tables With psql

Connect to the database:

```bash
psql postgresql://postgres:your_password@localhost:5432/adalatdata
```

Count imported judgments:

```sql
SELECT COUNT(*) FROM judgments;
```

Inspect a few judgments:

```sql
SELECT doc_id, case_title, year, month, filename FROM judgments LIMIT 10;
```

Inspect a known document:

```sql
SELECT id, doc_id, case_title, summary FROM judgments WHERE doc_id = '2016_10_1_856_EN';
```

Check judge relationships:

```sql
SELECT j.doc_id, j.case_title, array_agg(g.name) AS judges
FROM judgments j
JOIN judgment_judges jj ON j.id = jj.judgment_id
JOIN judges g ON jj.judge_id = g.id
GROUP BY j.id LIMIT 5;
```

Check party relationships:

```sql
SELECT j.doc_id, p.name, jp.role
FROM judgments j
JOIN judgment_parties jp ON j.id = jp.judgment_id
JOIN parties p ON jp.party_id = p.id
LIMIT 10;
```

Check cited sections:

```sql
SELECT j.doc_id, ls.section, ls.act
FROM judgments j
JOIN judgment_sections js ON j.id = js.judgment_id
JOIN legal_sections ls ON js.section_id = ls.id
LIMIT 10;
```

Check keywords:

```sql
SELECT j.doc_id, k.text, jk.category
FROM judgments j
JOIN judgment_keywords jk ON j.id = jk.judgment_id
JOIN keywords k ON jk.keyword_id = k.id
LIMIT 10;
```

Show full neighborhood for one judgment (all related entities):

```sql
SELECT 'Judge' AS entity_type, g.name
FROM judgments j JOIN judgment_judges jj ON j.id = jj.judgment_id JOIN judges g ON jj.judge_id = g.id
WHERE j.doc_id = '2016_10_1_856_EN'
UNION ALL
SELECT 'Party', p.name
FROM judgments j JOIN judgment_parties jp ON j.id = jp.judgment_id JOIN parties p ON jp.party_id = p.id
WHERE j.doc_id = '2016_10_1_856_EN'
UNION ALL
SELECT 'Section', ls.section || ' of ' || ls.act
FROM judgments j JOIN judgment_sections js ON j.id = js.judgment_id JOIN legal_sections ls ON js.section_id = ls.id
WHERE j.doc_id = '2016_10_1_856_EN'
UNION ALL
SELECT 'Keyword', k.text
FROM judgments j JOIN judgment_keywords jk ON j.id = jk.judgment_id JOIN keywords k ON jk.keyword_id = k.id
WHERE j.doc_id = '2016_10_1_856_EN';
```

## Vector Search With psql

The query vector must match the index dimension of `1024`. For a quick sanity check, query against the stored embedding of a known judgment:

```sql
SELECT doc_id, case_title, embedding <=> (SELECT embedding FROM judgments LIMIT 1) AS distance
FROM judgments
WHERE embedding IS NOT NULL
ORDER BY distance
LIMIT 5;
```

For real semantic search, generate a 1024-dimensional embedding with the same model used by `ingest.py` and pass it as a vector literal:

```sql
SELECT doc_id, case_title, summary,
       embedding <=> '[1.0, 0.0, ...]'::vector AS distance
FROM judgments
WHERE embedding IS NOT NULL
ORDER BY distance
LIMIT 10;
```

Combine full-text search with vector ranking:

```sql
SELECT doc_id, case_title, summary,
       embedding <=> '[1.0, 0.0, ...]'::vector AS distance
FROM judgments
WHERE search_tsvector @@ plainto_tsquery('english', 'bail conditions')
  AND embedding IS NOT NULL
ORDER BY distance
LIMIT 10;
```

## Query The AGE Graph (Cypher)

Run Cypher queries through `psql` against the AGE graph:

```sql
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

SELECT * FROM cypher('adalatdata', $$
  MATCH (j:Judgment)
  RETURN count(j)
$$) AS (result agtype);
```

Inspect judgments with judges:

```sql
SELECT * FROM cypher('adalatdata', $$
  MATCH (j:Judgment)-[:DELIVERED_BY]->(g:Judge)
  RETURN j.doc_id, j.case_title, collect(g.name) AS judges
  LIMIT 5
$$) AS (result agtype);
```

Show neighborhood for one judgment:

```sql
SELECT * FROM cypher('adalatdata', $$
  MATCH (j:Judgment {doc_id: '2016_10_1_856_EN'})-[r]->(n)
  RETURN type(r) AS rel_type, n
$$) AS (result agtype);
```

## Query With The PostgresClient (Python)

```python
from psycopg import Connection
import psycopg

conn = psycopg.connect("postgresql://postgres:your_password@localhost:5432/adalatdata")
# Import the PostgresClient from ingest
from ingest import PostgresClient

client = PostgresClient(conn)

# Cypher graph query
results = client.query(
    "MATCH (j:Judgment)-[:DELIVERED_BY]->(g:Judge) "
    "RETURN j.doc_id, j.case_title, collect(g.name) AS judges LIMIT 5"
)
print(results)

# Vector search
query_vector = [...]  # 1024-dimensional embedding
results = client.vector_search("Judgment", "embedding", query_vector, 5)
for doc_id, distance in results:
    print(f"  {doc_id} | distance={distance:.6f}")

# Entity counts
for label in ["Judgment", "Judge", "Party", "LegalSection", "Keyword"]:
    print(f"  {label}: {client.node_count(label)}")
```

## Reset / Drop Commands

Delete all data from relational tables (keeps schema):

```sql
TRUNCATE judgment_keywords, judgment_sections, judgment_parties, judgment_judges RESTART IDENTITY;
TRUNCATE judgments, judges, parties, legal_sections, keywords RESTART IDENTITY;
```

Drop the pgvector HNSW index:

```sql
DROP INDEX IF EXISTS idx_judgments_embedding;
```

Delete all nodes from the AGE graph:

```sql
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

SELECT * FROM cypher('adalatdata', $$
  MATCH (n) DETACH DELETE n
$$) AS (result agtype);
```

Drop the entire AGE graph:

```sql
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

SELECT * FROM cypher('adalatdata', $$
  MATCH (n) DETACH DELETE n
$$) AS (result agtype);

SELECT drop_graph('adalatdata', true);
```

Drop everything (extensions, tables, graph) to start fresh:

```bash
psql -d adalatdata -c "DROP EXTENSION IF EXISTS age CASCADE;"
psql -d adalatdata -c "DROP EXTENSION IF EXISTS vector CASCADE;"
```

Then re-run `uv run python init_db.py` to recreate from scratch.

## Useful One-Line Smoke Tests

Relational sanity check:

```bash
psql postgresql://postgres:your_password@localhost:5432/adalatdata -c "SELECT doc_id, case_title, summary FROM judgments LIMIT 5;"
```

Relationship sanity check:

```bash
psql postgresql://postgres:your_password@localhost:5432/adalatdata -c "SELECT j.doc_id, j.case_title, array_agg(g.name) AS judges FROM judgments j JOIN judgment_judges jj ON j.id = jj.judgment_id JOIN judges g ON jj.judge_id = g.id GROUP BY j.id LIMIT 5;"
```

Script sanity check:

```bash
uv run python ingest.py --limit 1
```

## Notes

- `init_db.py` must be run before the first ingest. It is idempotent.
- `ingest.py` writes to both relational tables and the AGE graph `adalatdata`.
- The pgvector index is `idx_judgments_embedding` (HNSW, cosine distance).
- Vector search requires a query vector with exactly `1024` dimensions, matching the embedding model output.
- Embedding model: `text-embedding-qwen3-embedding-0.6b`, served at `http://localhost:1234/v1/embeddings`.
- The AGE graph mirrors all relational data, so Cypher queries from `nlq.py` can traverse relationships while the relational layer handles metadata filtering and vector search.
