#!/usr/bin/env python3
"""
AdalatData Pipeline - NLQ: Natural Language to Cypher

Converts natural language questions into Cypher queries for the adalatdata AGE graph,
using the local LLM endpoint. Optionally executes the generated query.

Usage:
  uv run python nlq.py "What judgments did Justice XYZ deliver in 2016?"
  uv run python nlq.py "What judgments did Justice XYZ deliver in 2016?" --execute
"""

import argparse
import sys
from typing import Optional

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from dotenv import load_dotenv

from ingest import PostgresClient

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_LLM_BASE_URL = "http://localhost:8080/v1"
DEFAULT_LLM_MODEL = "qwen2.5"

# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class CypherResult(BaseModel):
    cypher: str = Field(description="The Cypher query generated from the natural language question")
    explanation: str = Field(description="Brief explanation of what the query does")

# ---------------------------------------------------------------------------
# Schema discovery
# ---------------------------------------------------------------------------

_SCHEMA_CACHE: Optional[str] = None


def discover_schema(client: PostgresClient) -> str:
    """Query the AGE graph via PostgresClient to build a schema summary string."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE

    lines: list[str] = []

    # 1. Get all node labels
    # Returns list of lists: [["Judgment"], ["Judge"], ...]
    label_result = client.query("MATCH (n) RETURN DISTINCT labels(n) AS labelGroup")
    all_labels: list[str] = []
    seen_labels: set[str] = set()
    for row in label_result:
        # row is a list like ["Judgment"] or a single-item list containing a list
        group = row[0] if isinstance(row, list) else row
        if isinstance(group, list):
            for lbl in group:
                if lbl not in seen_labels:
                    seen_labels.add(lbl)
                    all_labels.append(lbl)
        elif isinstance(group, str) and group not in seen_labels:
            seen_labels.add(group)
            all_labels.append(group)
    lines.append(f"Node labels: {', '.join(all_labels)}")

    # 2. Get all relationship types
    # Returns list of strings: ["CITES_SECTION", "DELIVERED_BY", ...]
    rel_result = client.query("MATCH ()-[r]->() RETURN DISTINCT type(r) AS rel_type")
    rel_types: list[str] = []
    seen_rels: set[str] = set()
    for row in rel_result:
        rel = row[0] if isinstance(row, list) else row
        if rel and rel not in seen_rels:
            seen_rels.add(rel)
            rel_types.append(str(rel))
    lines.append(f"Relationship types: {', '.join(rel_types)}")

    # 2b. Get relationship direction (source->target labels)
    dir_result = client.query(
        "MATCH (a)-[r]->(b) "
        "RETURN DISTINCT head(labels(a)) AS src, type(r) AS rel, head(labels(b)) AS tgt"
    )
    lines.append("Relationship directions:")
    for row in dir_result:
        if isinstance(row, list):
            lines.append(f"  {row[0]} -[:{row[1]}]-> {row[2]}")

    # 3. Get sample properties per label
    # For each label, query one node to discover its properties
    lines.append("Properties:")
    for lbl in all_labels:
        try:
            prop_result = client.query(
                f"MATCH (n:{lbl}) RETURN keys(n) AS properties LIMIT 1"
            )
            if prop_result:
                props = prop_result[0]
                if isinstance(props, list):
                    # Result is the properties list directly
                    prop_list = [str(p) for p in props]
                else:
                    prop_list = []
                lines.append(f"  {lbl}: {prop_list}")
        except Exception:
            lines.append(f"  {lbl}: (error reading properties)")

    _SCHEMA_CACHE = "\n".join(lines)
    return _SCHEMA_CACHE

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


def create_llm_client(base_url: str):
    openai_client = OpenAI(base_url=base_url, api_key="not-needed")
    return instructor.from_openai(openai_client, mode=instructor.Mode.JSON)

# ---------------------------------------------------------------------------
# Text-to-Cypher
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a Cypher query generator for a graph database containing Indian Supreme Court judgments.
Given a graph schema and a natural language question, generate a valid Cypher query.

Schema:
{schema_summary}

Rules:
- Only use labels and properties from the schema above
- Keep queries simple and focused
- Use RETURN to project useful properties (e.g. case_title, year, name, summary)
- Avoid returning raw node objects; always RETURN specific properties
- If the question asks for counts, use COUNT()
- If the question asks for a listing, add ORDER BY and LIMIT 20 unless a specific count is requested
- IMPORTANT: In Apache AGE, you cannot ORDER BY a column alias defined in the RETURN clause. Instead, use the expression directly in ORDER BY (e.g., ORDER BY count(j) DESC instead of ORDER BY cnt DESC)
- Do NOT use WITH clauses; they are not supported in this AGE version
"""


def text_to_cypher(question: str, schema_summary: str, llm_model: str, llm_base_url: str) -> CypherResult:
    """Call the LLM to convert a natural language question to Cypher."""
    client = create_llm_client(llm_base_url)
    try:
        return client.chat.completions.create(
            response_model=CypherResult,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(schema_summary=schema_summary)},
                {"role": "user", "content": question},
            ],
            model=llm_model,
            max_retries=2,
        )
    except Exception as e:
        print(f"LLM error: {e}", file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------


import re as _re


def _extract_headers(cypher: str) -> list[str]:
    """Extract column names from a Cypher RETURN clause."""
    # Find RETURN ... (ORDER|LIMIT|WITH|UNION|$)
    match = _re.search(r"RETURN\s+(.+?)(?:\s+(?:ORDER\s+BY|LIMIT|WITH|UNION)|$)", cypher, _re.IGNORECASE | _re.DOTALL)
    if not match:
        return []
    return_clause = match.group(1).strip()
    # Split by comma, handling nested parens (e.g. collect(...))
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
    # Extract alias if present: expr AS alias  ->  alias, else expr
    headers = []
    for col in cols:
        alias_match = _re.search(r"\bAS\s+(\w+)\s*$", col, _re.IGNORECASE)
        if alias_match:
            headers.append(alias_match.group(1))
        else:
            # Strip keywords (DISTINCT, ALL) and use the expression
            expr = _re.sub(r"\b(DISTINCT|ALL)\b", "", col, _re.IGNORECASE).strip()
            headers.append(expr)
    return headers


def print_results(query_result: list, cypher: str = "") -> None:
    """Print query results in a readable table format.

    Handles both dict rows (legacy) and list rows (positional from AGE).
    If cypher is provided, column headers are extracted from its RETURN clause.
    """
    if not query_result:
        print("(no results)")
        return

    # Determine headers
    first_row = query_result[0]
    if isinstance(first_row, dict):
        str_headers = [str(h) for h in first_row.keys()]
    elif cypher:
        str_headers = _extract_headers(cypher)
    else:
        str_headers = [f"col{i}" for i in range(len(first_row))] if isinstance(first_row, list) else []

    # Build string rows
    str_rows: list[list[str]] = []
    for item in query_result:
        if isinstance(item, dict):
            str_row = [_format_value(item.get(h)) for h in str_headers]
        elif isinstance(item, list):
            str_row = [_format_value(v) for v in item]
        else:
            str_row = [_format_value(item)]
        str_rows.append(str_row)

    col_widths = [len(h) for h in str_headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))
            else:
                col_widths.append(len(cell))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(str_headers, col_widths)) + " |"

    print(sep)
    print(header_line)
    print(sep)
    for row in str_rows:
        line = "| " + " | ".join(cell.ljust(w) for cell, w in zip(row, col_widths)) + " |"
        print(line)
    print(sep)
    print(f"{len(str_rows)} row(s)")


def _format_value(val) -> str:
    if val is None:
        return "(null)"
    if isinstance(val, (list, dict)):
        import json
        return json.dumps(val, ensure_ascii=False)[:200]
    s = str(val)
    if len(s) > 200:
        return s[:197] + "..."
    return s

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert natural language questions to Cypher queries for adalatdata AGE graph"
    )
    parser.add_argument("query", type=str, help="The natural language question")
    parser.add_argument(
        "--execute", "-e", action="store_true",
        help="Execute the generated Cypher against the AGE graph and print results"
    )
    parser.add_argument(
        "--llm-model", type=str, default=DEFAULT_LLM_MODEL,
        help=f"LLM model name (default: {DEFAULT_LLM_MODEL})"
    )
    parser.add_argument(
        "--llm-base-url", type=str, default=DEFAULT_LLM_BASE_URL,
        help=f"LLM base URL (default: {DEFAULT_LLM_BASE_URL})"
    )
    args = parser.parse_args()

    # Connect to PostgreSQL + AGE
    print(f"Connecting to PostgreSQL (env: DB_HOST, DB_PORT, DB_NAME)...")
    client = PostgresClient.connect()
    print(f"PostgreSQL status: {client.status()}")

    # Discover schema
    print("Discovering schema...")
    schema_summary = discover_schema(client)
    print(schema_summary)
    print()

    # Generate Cypher
    print(f"Generating Cypher (model: {args.llm_model})...")
    result = text_to_cypher(args.query, schema_summary, args.llm_model, args.llm_base_url)

    print(f"\nExplanation: {result.explanation}")
    print(f"\nCypher:")
    print(f"  {result.cypher}")

    # Execute if requested
    if args.execute:
        print(f"\nExecuting query...")
        try:
            qr = client.query(result.cypher)
            print_results(qr, result.cypher)
        except Exception as e:
            print(f"Query execution failed: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
