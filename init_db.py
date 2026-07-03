#!/usr/bin/env python3
"""
AdalatData Pipeline - Database Initialization

Creates PostgreSQL extensions, tables, indexes, triggers, and the AGE graph.
Idempotent — safe to run multiple times.
"""

import os
import sys

from dotenv import load_dotenv

import psycopg

try:
    from psycopg.types.vector import Vector
except ImportError:
    # Fallback: some psycopg builds register Vector automatically.
    Vector = None  # type: ignore

load_dotenv()

GRAPH_NAME = "adalatdata"


def build_conninfo():
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ.get("DB_NAME", "adalatdata")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASS", "")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def create_tables(conn):
    with conn.cursor() as cur:
        # Core judgments table with pgvector embedding
        cur.execute("""
            CREATE TABLE IF NOT EXISTS judgments (
                id              SERIAL PRIMARY KEY,
                doc_id          TEXT NOT NULL UNIQUE,
                filename        TEXT,
                json_path       TEXT,
                md_path         TEXT,
                year            INT,
                month           INT,
                page_start      INT,
                page_end        INT,
                language        TEXT DEFAULT 'EN',
                case_title      TEXT,
                summary         TEXT,
                embedding       VECTOR(1024),
                search_tsvector TSVECTOR,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                updated_at      TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # Entity tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS judges (
                id   SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS parties (
                id   SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS legal_sections (
                id      SERIAL PRIMARY KEY,
                section TEXT NOT NULL,
                act     TEXT,
                UNIQUE (section, act)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                id   SERIAL PRIMARY KEY,
                text TEXT NOT NULL UNIQUE
            )
        """)

        # Junction tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS judgment_judges (
                judgment_id INT NOT NULL REFERENCES judgments(id),
                judge_id    INT NOT NULL REFERENCES judges(id),
                PRIMARY KEY (judgment_id, judge_id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS judgment_parties (
                judgment_id INT NOT NULL REFERENCES judgments(id),
                party_id    INT NOT NULL REFERENCES parties(id),
                role        TEXT,
                PRIMARY KEY (judgment_id, party_id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS judgment_sections (
                judgment_id INT NOT NULL REFERENCES judgments(id),
                section_id  INT NOT NULL REFERENCES legal_sections(id),
                PRIMARY KEY (judgment_id, section_id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS judgment_keywords (
                judgment_id INT NOT NULL REFERENCES judgments(id),
                keyword_id  INT NOT NULL REFERENCES keywords(id),
                category    TEXT,
                PRIMARY KEY (judgment_id, keyword_id)
            )
        """)

        # Indexes
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_judgments_year ON judgments(year)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_judgments_tsv "
            "ON judgments USING GIN(search_tsvector)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_judgments_embedding "
            "ON judgments USING hnsw (embedding vector_cosine_ops)"
        )

        # FTS trigger
        cur.execute("""
            CREATE OR REPLACE FUNCTION update_judgment_search_vector()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.search_tsvector :=
                    setweight(to_tsvector('english', coalesce(NEW.case_title, '')), 'A') ||
                    setweight(to_tsvector('english', coalesce(NEW.summary, '')), 'B');
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
        """)

        cur.execute("DROP TRIGGER IF EXISTS trg_update_judgment_search_vector ON judgments")
        cur.execute("""
            CREATE TRIGGER trg_update_judgment_search_vector
                BEFORE INSERT OR UPDATE OF case_title, summary ON judgments
                FOR EACH ROW EXECUTE FUNCTION update_judgment_search_vector()
        """)

    conn.commit()
    print("  Tables, indexes, and triggers created")


def create_age_graph(conn):
    with conn.cursor() as cur:
        cur.execute("LOAD 'age'")
        cur.execute('SET search_path = ag_catalog, "$user", public')
        cur.execute("SELECT 1 FROM ag_graph WHERE name = %s", (GRAPH_NAME,))
        if cur.fetchone():
            print(f"  AGE graph '{GRAPH_NAME}' already exists")
        else:
            cur.execute("SELECT create_graph(%s)", (GRAPH_NAME,))
            print(f"  AGE graph '{GRAPH_NAME}' created")
    conn.commit()


def main():
    if Vector is not None:
        Vector.register()
    conninfo = build_conninfo()
    print(f"Connecting to PostgreSQL: {conninfo.split('@')[1] if '@' in conninfo else conninfo}")

    conn = psycopg.connect(conninfo)

    # Enable extensions
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE EXTENSION IF NOT EXISTS age")
    conn.commit()
    print("  Extensions enabled: vector, age")

    create_tables(conn)
    create_age_graph(conn)

    conn.close()
    print("Database initialization complete.")


if __name__ == "__main__":
    main()
