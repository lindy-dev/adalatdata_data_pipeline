#!/usr/bin/env python3
"""
AdalatData Pipeline - Stage 2: Ingest

Reads extracted JSON and Markdown judgments, stores structured metadata in
PostgreSQL (relational tables + pgvector + AGE graph), and attaches vector
embeddings to Judgment rows.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
import psycopg

try:
    from psycopg.types.vector import Vector
except ImportError:
    Vector = None  # type: ignore

load_dotenv()

BASE_DIR = "/home/shreyas/Documents/02_Adalatdata/data_pipeline"
JSON_DIR = os.path.join(BASE_DIR, "extracted_jsons")
MD_DIR = os.path.join(BASE_DIR, "extracted_mds")
GRAPH_NAME = "adalatdata"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-0.6b"
VECTOR_DIM = 1024
OPENAI_BASE_URL = "http://localhost:1234/v1/embeddings"

EMBEDDING_TEXT_LIMIT = 16000
DEMO_RESULT_COUNT = 5

api_key = os.environ.get("OPENAI_API_KEY", "")
openai_client = OpenAI(base_url="http://localhost:1234/v1", api_key=api_key)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def cypher_string(value: str) -> str:
    """Escape a string for a Cypher literal used with AGE."""
    normalized = normalize_text(value).replace("\x00", "")
    # Replace ASCII apostrophes to avoid Cypher parser issues.
    return "'" + normalized.replace("'", "\u2019") + "'"


def cypher_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(cypher_literal(item) for item in value) + "]"
    if isinstance(value, (int, float)):
        return str(value)
    return cypher_string(as_text(value))


def properties_clause(properties: dict[str, Any]) -> str:
    return ", ".join(f"{key}: {cypher_literal(value)}" for key, value in properties.items())


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def resolve_markdown_path(json_path: Path, json_root: Path, md_root: Path) -> Path:
    relative_path = json_path.relative_to(json_root)
    return (md_root / relative_path).with_suffix(".md")


def read_json_file(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON value must be an object")
    return data


def read_markdown_file(md_path: Path) -> str:
    if not md_path.exists():
        return ""
    return normalize_text(md_path.read_text(encoding="utf-8"))


def extract_markdown_body(markdown_text: str) -> str:
    if not markdown_text:
        return ""
    marker = "## Judgment Text"
    if marker in markdown_text:
        return markdown_text.split(marker, 1)[1].strip()
    return markdown_text.strip()


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def build_embedding_text(data: dict[str, Any], markdown_text: str) -> str:
    entities = as_dict(data.get("entities"))
    case_title = as_text(as_dict(entities.get("case_title")).get("title"))
    summary = as_text(as_dict(entities.get("summary")).get("summary"))
    raw_text_preview = as_text(data.get("raw_text_preview"))

    chunks: list[str] = []

    markdown_body = extract_markdown_body(markdown_text)
    if markdown_body:
        chunks.append(markdown_body)

    if summary:
        chunks.append(f"Summary: {summary}")

    if case_title:
        chunks.append(f"Case title: {case_title}")

    judges = [
        as_text(as_dict(judge).get("name"))
        for judge in as_list(entities.get("judges"))
        if as_text(as_dict(judge).get("name"))
    ]
    if judges:
        chunks.append("Judges: " + "; ".join(judges))

    parties = []
    for party in as_list(entities.get("parties")):
        party_dict = as_dict(party)
        name = as_text(party_dict.get("name"))
        role = as_text(party_dict.get("role"))
        if not name:
            continue
        parties.append(f"{role}: {name}" if role else name)
    if parties:
        chunks.append("Parties: " + "; ".join(parties))

    sections = []
    for section in as_list(entities.get("sections")):
        section_dict = as_dict(section)
        section_text = as_text(section_dict.get("section"))
        act = as_text(section_dict.get("act"))
        if not section_text:
            continue
        sections.append(f"{section_text} of {act}" if act else section_text)
    if sections:
        chunks.append("Sections: " + "; ".join(sections))

    topics = []
    for topic in as_list(entities.get("topics")):
        topic_dict = as_dict(topic)
        text = as_text(topic_dict.get("text"))
        category = as_text(topic_dict.get("category"))
        if not text:
            continue
        topics.append(f"{text} ({category})" if category else text)
    if topics:
        chunks.append("Topics: " + "; ".join(topics))

    if not markdown_body and raw_text_preview:
        chunks.append("Raw text preview: " + raw_text_preview)

    combined = "\n\n".join(chunk for chunk in chunks if chunk).strip()
    return combined[:EMBEDDING_TEXT_LIMIT].strip()


def get_embedding(text: str, model: str = "text-embedding-qwen3-embedding-0.6b") -> list[float]:
    text = text.replace("\n", " ")
    result = openai_client.embeddings.create(input=[text], model=model)
    embedding = result.data[0].embedding
    if embedding is None or len(embedding) == 0:
        raise RuntimeError(
            f"embedding server returned null/empty vector for text "
            f"({len(text)} chars). Check model server logs."
        )
    return embedding


# ---------------------------------------------------------------------------
# PostgreSQL connection
# ---------------------------------------------------------------------------


def build_conninfo() -> str:
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ.get("DB_NAME", "adalatdata")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASS", "")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


# ---------------------------------------------------------------------------
# PostgreSQL client
# ---------------------------------------------------------------------------


class PostgresClient:
    """Combines relational tables, pgvector, and AGE graph operations."""

    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    @classmethod
    def connect(cls) -> "PostgresClient":
        if Vector is not None:
            Vector.register()
        conn = psycopg.connect(build_conninfo())
        # Load AGE extension for Cypher queries
        with conn.cursor() as cur:
            cur.execute("LOAD 'age'")
            cur.execute('SET search_path = ag_catalog, "$user", public')
        return cls(conn)

    def status(self) -> str:
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1")
        return "connected"

    # ---- Cypher / AGE graph ----

    def query(self, cypher: str, params: Optional[dict] = None) -> list[dict]:
        """Execute a Cypher query against the AGE graph."""
        if params:
            raise ValueError("parameterized Cypher is not supported by this wrapper")
        return self._cypher(cypher)

    # ---- Vector index ----

    def create_vector_index(
        self, label: str = "Judgment", prop: str = "embedding",
        dimensions: int = 1024, metric: str = "cosine",
    ) -> None:
        """Create pgvector HNSW index (idempotent)."""
        with self.conn.cursor() as cur:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_judgments_embedding "
                "ON judgments USING hnsw (embedding vector_cosine_ops)"
            )
        self.conn.commit()

    # ---- Entity upserts (relational + AGE mirror) ----

    def upsert_judgment(self, doc_id: str, props: dict[str, Any]) -> int:
        """Upsert a judgment row. Returns the primary key."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO judgments (
                    doc_id, filename, year, month, page_start, page_end,
                    language, case_title, summary, md_path, json_path, updated_at
                ) VALUES (
                    %(doc_id)s, %(filename)s, %(year)s, %(month)s, %(page_start)s,
                    %(page_end)s, %(language)s, %(case_title)s, %(summary)s,
                    %(md_path)s, %(json_path)s, NOW()
                )
                ON CONFLICT (doc_id) DO UPDATE SET
                    filename   = EXCLUDED.filename,
                    year       = EXCLUDED.year,
                    month      = EXCLUDED.month,
                    page_start = EXCLUDED.page_start,
                    page_end   = EXCLUDED.page_end,
                    language   = EXCLUDED.language,
                    case_title = EXCLUDED.case_title,
                    summary    = EXCLUDED.summary,
                    md_path    = EXCLUDED.md_path,
                    json_path  = EXCLUDED.json_path,
                    updated_at = NOW()
                RETURNING id
                """,
                props,
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("upsert_judgment returned no row")
            pk = row[0]

        # Mirror to AGE graph
        age_props = dict(props)
        age_props["doc_id"] = doc_id
        self._age_upsert_node("Judgment", age_props)

        return pk

    def upsert_entity(self, label: str, identity: dict[str, Any],
                      props: dict[str, Any]) -> int:
        """Upsert an entity row (Judge, Party, LegalSection, Keyword)."""
        table_map = {
            "Judge": ("judges", {"name": "name"}),
            "Party": ("parties", {"name": "name"}),
            "LegalSection": ("legal_sections", {"section": "section", "act": "act"}),
            "Keyword": ("keywords", {"text": "text"}),
        }

        if label not in table_map:
            raise ValueError(f"unknown entity label: {label}")

        table, col_map = table_map[label]
        all_props = dict(identity)
        all_props.update(props)

        cols = ", ".join(col_map.values())
        placeholders = ", ".join(f"%({k})s" for k in col_map.keys())
        conflict_cols = ", ".join(col_map.values())
        update_cols = ", ".join(
            f"{c} = EXCLUDED.{c}" for c in col_map.values()
        )

        with self.conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
                f"ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_cols} "
                f"RETURNING id",
                all_props,
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"upsert_entity returned no row for {label}")
            pk = row[0]

        # Mirror to AGE graph
        self._age_upsert_node(label, all_props)

        return pk

    def upsert_relationship(
        self, judgment_pk: int, doc_id: str, rel_type: str,
        target_label: str, target_identity: dict[str, Any],
        target_pk: int, rel_props: Optional[dict[str, Any]] = None,
    ) -> None:
        """Upsert a relationship in the junction table and AGE graph."""
        junction_map = {
            "DELIVERED_BY": "judgment_judges",
            "INVOLVES_PARTY": "judgment_parties",
            "CITES_SECTION": "judgment_sections",
            "HAS_KEYWORD": "judgment_keywords",
        }
        junction = junction_map.get(rel_type)
        if not junction:
            raise ValueError(f"unknown relationship type: {rel_type}")

        with self.conn.cursor() as cur:
            if rel_type == "INVOLVES_PARTY":
                role = rel_props.get("role") if rel_props else None
                cur.execute(
                    "INSERT INTO judgment_parties "
                    "(judgment_id, party_id, role) "
                    "VALUES (%s, %s, %s) "
                    "ON CONFLICT (judgment_id, party_id) "
                    "DO UPDATE SET role = EXCLUDED.role",
                    (judgment_pk, target_pk, role),
                )
            elif rel_type == "HAS_KEYWORD":
                category = rel_props.get("category") if rel_props else None
                cur.execute(
                    "INSERT INTO judgment_keywords "
                    "(judgment_id, keyword_id, category) "
                    "VALUES (%s, %s, %s) "
                    "ON CONFLICT (judgment_id, keyword_id) "
                    "DO UPDATE SET category = EXCLUDED.category",
                    (judgment_pk, target_pk, category),
                )
            elif rel_type == "DELIVERED_BY":
                cur.execute(
                    "INSERT INTO judgment_judges "
                    "(judgment_id, judge_id) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
                    (judgment_pk, target_pk),
                )
            elif rel_type == "CITES_SECTION":
                cur.execute(
                    "INSERT INTO judgment_sections "
                    "(judgment_id, section_id) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
                    (judgment_pk, target_pk),
                )

        # Mirror to AGE graph
        self._age_ensure_relationship(
            "Judgment", {"doc_id": doc_id},
            rel_type, target_label, target_identity, rel_props,
        )

    # ---- Embedding storage ----

    def add_embedding(self, judgment_pk: int, embedding: list[float]) -> None:
        """Store a vector embedding for a judgment."""
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE judgments SET embedding = %s, updated_at = NOW() "
                "WHERE id = %s",
                (embedding, judgment_pk),
            )

    def add_vector(
        self, label: str, prop: str, node_id: int, vector: list[float],
    ) -> None:
        """Backward-compatible alias for add_embedding."""
        self.add_embedding(node_id, vector)

    def vector_search(
        self, label: str = "Judgment", prop: str = "embedding",
        query_vector: list[float] | None = None, k: int = 10,
    ) -> list[tuple[str, float]]:
        """Semantic search using pgvector. Returns [(doc_id, distance), ...]."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, embedding <=> %s AS distance "
                "FROM judgments "
                "WHERE embedding IS NOT NULL "
                "ORDER BY distance LIMIT %s",
                (query_vector, k),
            )
            return [(row[0], float(row[1])) for row in cur.fetchall()]

    def node_count(self, label: str) -> int:
        """Count rows in the relational table for a given label."""
        table_map = {
            "Judgment": "judgments",
            "Judge": "judges",
            "Party": "parties",
            "LegalSection": "legal_sections",
            "Keyword": "keywords",
        }
        table = table_map.get(label)
        if not table:
            return 0
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")  # type: ignore[arg-type]
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def fetch_all_judgments(self) -> dict[str, dict[str, Any]]:
        """Fetch all judgments for lookup. Returns {doc_id: info}."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, case_title, summary, filename FROM judgments"
            )
            return {
                row[0]: {
                    "doc_id": row[0],
                    "case_title": row[1],
                    "summary": row[2],
                    "filename": row[3],
                }
                for row in cur.fetchall()
            }

    # ---- Internal AGE helpers ----

    def _cypher(self, query: str) -> list[dict]:
        """Execute a Cypher query against the AGE graph.

        Returns a list where each element is one row.
        - Single-column RETURN: each row is the scalar value (str, int, list, etc.)
        - Multi-column RETURN: each row is a list of values
        """
        # Count columns in the RETURN clause to build matching column definitions
        col_defs = self._cypher_column_defs(query)

        sql = (
            f"SELECT * FROM cypher('{GRAPH_NAME}', $$\n{query}\n$$) "
            f"AS ({col_defs})"
        )

        with self.conn.cursor() as cur:
            cur.execute(sql)
            results = []
            for row in cur.fetchall():
                # Convert agtype JSON strings to Python objects
                converted = []
                for val in row:
                    if isinstance(val, str):
                        # agtype-encoded strings come back as JSON-quoted strings
                        # like '"hello"' - unquote them
                        try:
                            converted.append(json.loads(val))
                        except (json.JSONDecodeError, ValueError):
                            converted.append(val)
                    else:
                        converted.append(val)
                # Return single value directly for 1-col, list for multi-col
                results.append(converted[0] if len(converted) == 1 else converted)
            return results

    def _cypher_column_defs(self, query: str) -> str:
        """Parse RETURN clause of a Cypher query to generate AGE column definitions.

        Returns something like 'col1 agtype, col2 agtype' or 'result agtype'.
        """
        # Find the last RETURN clause (handle WITH ... RETURN)
        import re
        # Match RETURN ... up to ORDER BY, LIMIT, or end
        match = re.search(
            r"RETURN\s+(.+?)(?:\s+ORDER\s+BY|\s+LIMIT|\s+SKIP|\s+UNION|\s*;|\s*$)",
            query,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return "result agtype"

        return_clause = match.group(1).strip()
        # Split by comma, respecting parentheses depth (collect(...), etc.)
        cols: list[str] = []
        depth = 0
        current = ""
        for ch in return_clause:
            if ch == "(":
                depth += 1
                current += ch
            elif ch == ")":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                cols.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            cols.append(current.strip())

        if not cols:
            return "result agtype"

        # Extract column names (alias if present, else expression)
        col_names = []
        for col in cols:
            alias_match = re.search(r"\bAS\s+(\w+)\s*$", col, re.IGNORECASE)
            if alias_match:
                col_names.append(alias_match.group(1).lower())
            else:
                # Strip SQL keywords (DISTINCT, etc.) and simplify to safe identifier
                expr = re.sub(r"\b(DISTINCT|ALL)\b", "", col, flags=re.IGNORECASE).strip()
                name_match = re.match(r"(\w+)", expr.lower())
                if name_match:
                    col_names.append(name_match.group(1))
                else:
                    col_names.append(f"col{len(col_names)}")

        # Ensure unique names
        seen: set[str] = set()
        unique_names: list[str] = []
        for name in col_names:
            base = name
            counter = 0
            while name in seen:
                counter += 1
                name = f"{base}_{counter}"
            seen.add(name)
            unique_names.append(name)

        return ", ".join(f"{n} agtype" for n in unique_names)

    def _age_upsert_node(self, label: str, properties: dict[str, Any]) -> None:
        """Upsert a node in the AGE graph using MERGE."""
        props_str = properties_clause(properties)
        self._cypher(f"MERGE (n:{label} {{{props_str}}})")

    def _age_ensure_relationship(
        self, source_label: str, source_identity: dict[str, Any],
        rel_type: str, target_label: str, target_identity: dict[str, Any],
        rel_props: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create a relationship in AGE if it doesn't exist."""
        s_props = properties_clause(source_identity)
        t_props = properties_clause(target_identity)

        if rel_props:
            r_props = properties_clause(rel_props)
            query = (
                f"MATCH (s:{source_label} {{{s_props}}}), "
                f"(t:{target_label} {{{t_props}}}) "
                f"MERGE (s)-[r:{rel_type} {{{r_props}}}]->(t)"
            )
        else:
            query = (
                f"MATCH (s:{source_label} {{{s_props}}}), "
                f"(t:{target_label} {{{t_props}}}) "
                f"MERGE (s)-[:{rel_type}]->(t)"
            )
        self._cypher(query)


# ---------------------------------------------------------------------------
# Free functions (backward-compatible interface)
# ---------------------------------------------------------------------------


def upsert_node(
    client: PostgresClient,
    label: str,
    identity_properties: dict[str, Any],
    properties: dict[str, Any],
    alias: str = "n",
) -> int:
    """Upsert entity in relational table and AGE graph. Returns PK."""
    if label == "Judgment":
        return client.upsert_judgment(identity_properties["doc_id"], properties)
    return client.upsert_entity(label, identity_properties, properties)


def ensure_relationship(
    client: PostgresClient,
    doc_id: str,
    judgment_pk: int,
    rel_type: str,
    target_label: str,
    target_identity: dict[str, Any],
    target_pk: int,
    rel_properties: Optional[dict[str, Any]] = None,
) -> None:
    """Upsert relationship in junction table and AGE graph."""
    client.upsert_relationship(
        judgment_pk, doc_id, rel_type, target_label,
        target_identity, target_pk, rel_properties,
    )


# ---------------------------------------------------------------------------
# Ingestion logic
# ---------------------------------------------------------------------------


def judgment_properties(
    data: dict[str, Any], json_path: Path, md_path: Path,
) -> dict[str, Any]:
    metadata = as_dict(data.get("metadata"))
    entities = as_dict(data.get("entities"))

    year = as_int(metadata.get("year"))
    month = as_int(metadata.get("month"))
    page_start = as_int(metadata.get("page_start"))
    page_end = as_int(metadata.get("page_end"))
    language = as_text(metadata.get("language")) or "EN"

    return {
        "doc_id": json_path.stem,
        "filename": as_text(data.get("filename")) or f"{json_path.stem}.pdf",
        "year": year,
        "month": month,
        "page_start": page_start,
        "page_end": page_end,
        "page_range": f"{page_start}-{page_end}",
        "language": language,
        "case_title": as_text(as_dict(entities.get("case_title")).get("title")),
        "summary": as_text(as_dict(entities.get("summary")).get("summary")),
        "md_path": str(md_path),
        "json_path": str(json_path),
    }


def ingest_file(
    client: PostgresClient, json_path: Path, md_root: Path, json_root: Path,
) -> None:
    data = read_json_file(json_path)
    md_path = resolve_markdown_path(json_path, json_root, md_root)
    markdown_text = read_markdown_file(md_path)
    embedding_text = build_embedding_text(data, markdown_text)

    if not embedding_text:
        raise ValueError("no embedding text available")

    embedding = get_embedding(embedding_text)
    props = judgment_properties(data, json_path, md_path)
    doc_id = as_text(props["doc_id"])

    judgment_pk = upsert_node(
        client, "Judgment", {"doc_id": doc_id}, props, alias="j",
    )
    client.add_vector("Judgment", "embedding", judgment_pk, embedding)

    entities = as_dict(data.get("entities"))

    for judge in as_list(entities.get("judges")):
        judge_dict = as_dict(judge)
        judge_name = as_text(judge_dict.get("name"))
        if not judge_name:
            continue
        judge_pk = upsert_node(
            client, "Judge", {"name": judge_name}, {"name": judge_name}, alias="g",
        )
        ensure_relationship(
            client, doc_id, judgment_pk, "DELIVERED_BY",
            "Judge", {"name": judge_name}, judge_pk,
        )

    for party in as_list(entities.get("parties")):
        party_dict = as_dict(party)
        party_name = as_text(party_dict.get("name"))
        role = as_text(party_dict.get("role"))
        if not party_name:
            continue
        party_pk = upsert_node(
            client, "Party", {"name": party_name}, {"name": party_name}, alias="p",
        )
        rel_props = {"role": role} if role else None
        ensure_relationship(
            client, doc_id, judgment_pk, "INVOLVES_PARTY",
            "Party", {"name": party_name}, party_pk, rel_props,
        )

    for section in as_list(entities.get("sections")):
        section_dict = as_dict(section)
        section_text = as_text(section_dict.get("section"))
        act = as_text(section_dict.get("act"))
        if not section_text:
            continue
        identity = {"section": section_text, "act": act}
        section_pk = upsert_node(
            client, "LegalSection", identity, identity, alias="s",
        )
        ensure_relationship(
            client, doc_id, judgment_pk, "CITES_SECTION",
            "LegalSection", identity, section_pk,
        )

    for topic in as_list(entities.get("topics")):
        topic_dict = as_dict(topic)
        text = as_text(topic_dict.get("text"))
        category = as_text(topic_dict.get("category"))
        if not text:
            continue
        keyword_pk = upsert_node(
            client, "Keyword", {"text": text}, {"text": text}, alias="k",
        )
        rel_props = {"category": category} if category else None
        ensure_relationship(
            client, doc_id, judgment_pk, "HAS_KEYWORD",
            "Keyword", {"text": text}, keyword_pk, rel_props,
        )


# ---------------------------------------------------------------------------
# Demo search
# ---------------------------------------------------------------------------


def run_demo_search(client: PostgresClient, query_text: str) -> None:
    query_embedding = get_embedding(query_text)
    results = client.vector_search(
        "Judgment", "embedding", query_embedding, DEMO_RESULT_COUNT,
    )
    if not results:
        print("Demo search returned no results.")
        return

    lookup = client.fetch_all_judgments()
    print("\nDemo search results:")
    for rank, (doc_id, distance) in enumerate(results, start=1):
        judgment = lookup.get(doc_id, {})
        title = as_text(judgment.get("case_title")) or "(untitled judgment)"
        summary = as_text(judgment.get("summary"))
        print(f"  {rank}. {doc_id} | distance={distance:.6f} | {title}")
        if summary:
            print(f"     {summary[:180]}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run(
    json_dir: Path,
    md_dir: Path,
    limit: Optional[int],
    demo_search: Optional[str],
) -> int:
    json_files = sorted(json_dir.rglob("*.json"))
    if limit is not None:
        json_files = json_files[:limit]

    total = len(json_files)
    if total == 0:
        print(f"No JSON files found in {json_dir}")
        return 1

    print(f"Found {total} JSON files in {json_dir}")
    print(f"Markdown root: {md_dir}")
    print(f"Embedding endpoint: {OPENAI_BASE_URL}")
    print(f"Database: {build_conninfo()}")
    print(f"AGE graph: {GRAPH_NAME}")

    client = PostgresClient.connect()
    print(f"PostgreSQL status: {client.status()}")
    client.create_vector_index("Judgment", "embedding", VECTOR_DIM, "cosine")

    success_count = 0
    failure_count = 0

    for index, json_path in enumerate(json_files, start=1):
        try:
            with client.conn.transaction():
                ingest_file(client, json_path, md_dir, json_dir)
            print(f"[{index}/{total}] OK   {json_path.stem}")
            success_count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[{index}/{total}] FAIL {json_path.stem} | {exc}")
            failure_count += 1

    print("\nSummary")
    print(f"  Judgments processed: {total}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failure_count}")
    for label in ["Judgment", "Judge", "Party", "LegalSection", "Keyword"]:
        print(f"  {label}: {client.node_count(label)}")

    if demo_search:
        try:
            run_demo_search(client, demo_search)
        except Exception as exc:  # noqa: BLE001
            print(f"Demo search failed: {exc}")

    return 0 if failure_count == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: ingest extracted judgments into PostgreSQL + pgvector + AGE"
    )
    parser.add_argument("--json-dir", type=str, default=JSON_DIR, help="JSON input root")
    parser.add_argument("--md-dir", type=str, default=MD_DIR, help="Markdown input root")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of JSON files")
    parser.add_argument(
        "--demo-search",
        type=str,
        default=None,
        help="Run a small semantic search demo after ingestion",
    )
    args = parser.parse_args()

    json_dir = Path(args.json_dir).expanduser().resolve()
    md_dir = Path(args.md_dir).expanduser().resolve()

    if not json_dir.is_dir():
        print(f"ERROR: JSON directory not found: {json_dir}")
        sys.exit(1)

    if not md_dir.is_dir():
        print(f"ERROR: Markdown directory not found: {md_dir}")
        sys.exit(1)

    sys.exit(run(json_dir, md_dir, args.limit, args.demo_search))


if __name__ == "__main__":
    main()
