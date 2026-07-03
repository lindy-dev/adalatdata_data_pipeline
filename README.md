# AdalatData Pipeline

A comprehensive, structured dataset of **589 judgments** delivered by the Supreme Court of India during the calendar year **2016**, plus a small number of spill-over judgments from **2017** bundled in the original source PDFs. Each judgment is available in three forms: the original scanned PDF as downloaded from the official eCourts portal, a full text extracted via OCR as Markdown, and a fully structured JSON with extracted entities (case title, summary, judges, parties, legal sections, and topic labels across 11 categories).

The dataset was curated by downloading the 2016 subset from the [Indian Supreme Court Judgments registry on AWS Open Data](https://registry.opendata.aws/indian-supreme-court-judgments/) (managed by Dattam Labs, CC-BY-4.0), then OCR-processing every PDF through **PaddleOCR-VL-1.6** to produce the Markdown and structured JSON outputs.

This pipeline ingests those JSON and Markdown outputs into PostgreSQL with pgvector (semantic search) and AGE (Cypher graph queries).

## Dataset overview

| Metric           | Value                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Judgments        | 589 (2016, + a few 2017 spill-over)                                                                                      |
| Source PDFs      | Scanned images from eCourts portal                                                                                       |
| OCR engine       | PaddleOCR-VL-1.6                                                                                                         |
| Topic categories | 11 (procedural, administrative, criminal, constitutional, civil, property, tax, corporate, other, family, environmental) |
| Total topics     | 3041                                                                                                                     |
| License          | CC-BY-4.0 (source data via Dattam Labs)                                                                                  |

## Download

The raw PDFs, OCR-processed Markdown, and structured JSON outputs are available on Hugging Face:

- [Shreyasrao/Indian-law-supreme-court-judgements-2016](https://huggingface.co/datasets/Shreyasrao/Indian-law-supreme-court-judgements-2016)

## Directory Structure

```
Adalatdata_data/
  raw_data/year=YYYY/           # Input: original PDFs
  data_pipeline/
    extract.py                  # Stage 1 (combined): PDF -> JSON + Markdown
    ocr_extract.py              # Stage 1a: PDF -> Markdown (PaddleOCR)
    md_to_json.py               # Stage 1b: Markdown -> JSON (LLM extraction)
    ingest.py                   # Stage 2: JSON -> PostgreSQL + pgvector + AGE
    init_db.py                  # Database initialization (idempotent)
    nlq.py                      # Natural language -> Cypher query generator
  extracted_jsons/year=YYYY/    # Stage 1 output: structured JSON
  extracted_mds/year=YYYY/      # Stage 1a output: Markdown documents
```

## Setup

```bash
# Install dependencies
uv pip install pymupdf instructor openai psycopg pydantic python-dotenv

# Start PostgreSQL with pgvector + AGE extensions
docker run -d --name adalatdata-postgres \
  -e POSTGRES_PASSWORD=your_password_here \
  -p 5432:5432 \
  vishva123/postgres-for-ai

# Or use the image directly:
# https://hub.docker.com/r/vishva123/postgres-for-ai

# Configure database connection (see .env.example)
cp .env.example .env
# Edit .env with your PostgreSQL credentials

# Initialize database schema (tables, indexes, AGE graph)
uv run python init_db.py
```

Database connection is controlled by environment variables (or `.env`):

| Variable  | Default      | Description       |
| --------- | ------------ | ----------------- |
| `DB_HOST` | `localhost`  | PostgreSQL host   |
| `DB_PORT` | `5432`       | PostgreSQL port   |
| `DB_NAME` | `adalatdata` | Database name     |
| `DB_USER` | `postgres`   | Database user     |
| `DB_PASS` | _(required)_ | Database password |

## Running

### Step-by-step pipeline (Stage 1 split into 1a + 1b)

`extract.py` runs the full pipeline end-to-end. To inspect each step individually, run the two stages separately:

```bash
# Stage 1a: PDF -> Markdown (PaddleOCR-VL-1.6)
# Requires: paddleocr_venv, PaddleOCR-VL llama-server on port 8083
~/paddleocr_venv/bin/python3 ocr_extract.py

# Custom paths / options
~/paddleocr_venv/bin/python3 ocr_extract.py \
  --input-dir /path/to/raw_data \
  --output-dir /path/to/extracted_mds \
  --max-pages 9999 \
  --dpi 72

# Stage 1b: Markdown -> JSON (LLM entity extraction)
uv run python md_to_json.py

# Custom paths / LLM
uv run python md_to_json.py \
  --input-dir /path/to/extracted_mds \
  --output-dir /path/to/extracted_jsons \
  --llm-base-url http://localhost:8080/v1 \
  --llm-model qwen3.6-27b
```

**What each step does:**

| Script                | Input                | Output                   | Engine                            |
| --------------------- | -------------------- | ------------------------ | --------------------------------- |
| `ocr_extract.py` (1a) | `raw_data/*.pdf`     | `extracted_mds/*.md`     | PaddleOCR-VL-1.6 via llama-server |
| `md_to_json.py` (1b)  | `extracted_mds/*.md` | `extracted_jsons/*.json` | Local LLM + instructor + Pydantic |

Both scripts skip files that already have output, so you can resume interrupted runs safely. `ocr_extract.py` auto-relaunches from `paddleocr_venv` if needed and checks that the PaddleOCR server is running before processing. `md_to_json.py` strips OCR metadata headers and page markers before sending text to the LLM.

### Stage 1: Extract (combined)

Reads PDFs, extracts text, sends to local LLM via instructor+Pydantic, outputs JSON + Markdown.

```bash
cd /home/shreyas/Documents/02_Adalatdata/data_pipeline

uv run python extract.py --year 2016

# Custom LLM endpoint
uv run python extract.py --year 2016 --llm-base-url http://localhost:8080/v1 --llm-model your-model
```

### Stage 1.5: Post-processing and sanity checks

Between extraction and ingestion, run these scripts against `extracted_jsons/` to clean up OCR artifacts and verify data quality. Run them **after Stage 1, before Stage 2**.

```bash
# 1. Capitalize judge names (title case)
uv run python scripts/capitalize_judge_names.py

# 2. Normalize judge names (fix OCR errors, canonicalize variants)
#    e.g. "Prallulla C. Pant" -> "Prafulla C. Pant", "Kurian" -> "Kurian Joseph"
uv run python scripts/normalize_judge_names.py

# 3. Extract unique judge names for review
uv run python scripts/extract_judge_names.py
#    Output: scripts/judge_names.txt (34 unique judges)

# 4. Extract topic category distribution for review
uv run python scripts/extract_categories.py
#    Output: scripts/categories.txt (11 categories, 3041 total topics)
```

**What each script does:**

| Script                      | Action                                                                                                      |
| --------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `capitalize_judge_names.py` | Converts all judge names to title case (e.g. "justice xyz" -> "Justice Xyz")                                |
| `normalize_judge_names.py`  | Maps OCR errors and name variants to canonical names (35 known variants, e.g. 11 Prafulla C. Pant variants) |
| `extract_judge_names.py`    | Dumps a deduplicated, sorted list of all judge names to `judge_names.txt`                                   |
| `extract_categories.py`     | Counts topic categories and outputs `categories.txt` for sanity-checking coverage                           |

The normalization map in `normalize_judge_names.py` handles initial-spacing fixes (`A.K. Sikri` -> `A. K. Sikri`), OCR misspellings (`Pratlulla` -> `Prafulla`), and short-form aliases (`Kurian` -> `Kurian Joseph`). Review `judge_names.txt` after running to catch any remaining variants.

### Stage 2: Ingest (embeddings + PostgreSQL + pgvector + AGE)

Reads JSON files, generates 1024-dim embeddings via local embedding endpoint, stores structured metadata in PostgreSQL relational tables, vector embeddings in pgvector, and mirrors the graph in AGE for Cypher queries.

```bash
# Smoke test (one file)
uv run python ingest.py --limit 1

# Full ingest
uv run python ingest.py

# With demo semantic search
uv run python ingest.py --demo-search "bail conditions in criminal cases"

# Custom input directories
uv run python ingest.py \
  --json-dir /path/to/extracted_jsons \
  --md-dir /path/to/extracted_mds
```

### Natural Language Queries

Generate and optionally execute Cypher queries from natural language questions:

```bash
# Generate Cypher only
uv run python nlq.py "What judgments did Justice XYZ deliver in 2016?"

# Generate and execute
uv run python nlq.py "What judgments did Justice XYZ deliver in 2016?" --execute

# Custom LLM
uv run python nlq.py "..." --llm-model your-model --llm-base-url http://localhost:8080/v1
```

### Full pipeline for all years

```bash
for year in {2016..2026}; do
  uv run python extract.py --year $year
done
uv run python ingest.py
```

## Architecture

The backend uses three PostgreSQL layers:

- **Relational tables** — structured metadata, entity lookups, junction tables for relationships
- **pgvector** — 1024-dim cosine-similarity search on judgment embeddings (HNSW index)
- **AGE (Apache AGE)** — Cypher graph queries for multi-hop relationship traversal

### Database tables

| Table               | Purpose                                            |
| ------------------- | -------------------------------------------------- |
| `judgments`         | Core document metadata + `embedding` vector column |
| `judges`            | Named judges                                       |
| `parties`           | Case parties                                       |
| `legal_sections`    | Cited law sections                                 |
| `keywords`          | Legal topics/keywords                              |
| `judgment_judges`   | Judgment-Judge junction                            |
| `judgment_parties`  | Judgment-Party junction (with `role`)              |
| `judgment_sections` | Judgment-Section junction                          |
| `judgment_keywords` | Judgment-Keyword junction (with `category`)        |

### AGE graph schema (Cypher)

**Nodes:**

- `Judgment` - Core document. Properties: `doc_id`, `year`, `month`, `page_start`, `page_end`, `filename`, `language`, `case_title`, `summary`. Has 1024-dim vector embedding stored in relational layer.
- `Judge` - Named judge. Property: `name`
- `Party` - Case party. Property: `name`
- `LegalSection` - Cited law section. Properties: `section`, `act`
- `Keyword` - Legal topic. Property: `text`

**Relationships:**

- `Judgment -[:DELIVERED_BY]-> Judge`
- `Judgment -[:INVOLVES_PARTY]-> Party`
- `Judgment -[:CITES_SECTION]-> LegalSection`
- `Judgment -[:HAS_KEYWORD]-> Keyword`

Full query examples and psql commands are in [INGEST_USAGE.md](INGEST_USAGE.md).

## JSON Output Format (Stage 1)

```json
{
  "filename": "2016_1_1_23_EN.pdf",
  "metadata": {
    "year": 2016,
    "month": 1,
    "page_start": 1,
    "page_end": 23,
    "language": "EN"
  },
  "entities": {
    "case_title": { "title": "State of Maharashtra v. XYZ" },
    "summary": { "summary": "Case about..." },
    "judges": [{ "name": "Justice ABC", "role": "author" }],
    "parties": [{ "name": "State of Maharashtra", "role": "respondent" }],
    "sections": [{ "section": "302", "act": "Indian Penal Code" }],
    "topics": [{ "text": "bail", "category": "criminal" }]
  },
  "raw_text_preview": "..."
}
```

## Dependencies

| Package         | Purpose                                                       |
| --------------- | ------------------------------------------------------------- |
| `pymupdf`       | PDF text extraction                                           |
| `instructor`    | LLM structured output (Pydantic validation + retries)         |
| `openai`        | OpenAI-compatible client for local LLM and embedding endpoint |
| `psycopg`       | PostgreSQL adapter (relational queries + AGE/Cypher)          |
| `pydantic`      | Data validation models                                        |
| `python-dotenv` | Load database credentials from `.env`                         |

Embeddings are generated by a local OpenAI-compatible endpoint (`text-embedding-qwen3-embedding-0.6b`, 1024-dim) via embedding server run on LM Studio on cpu.
