#!/usr/bin/env python3
"""
AdalatData Pipeline - Stage 1b: Markdown to JSON via LLM entity extraction

Reads OCR-generated markdown files from extracted_mds, runs LLM-powered
entity extraction using instructor + Pydantic, and outputs structured JSON.

Pipeline:
  Markdown (from OCR) -> entity extraction (LLM) -> JSON

Input:  extracted_mds/*.md
Output: extracted_jsons/*.json

Usage:
  cd /home/shreyas/Documents/02_Adalatdata/data_pipeline
  .venv/bin/python3 md_to_json.py

  # Custom paths
  .venv/bin/python3 md_to_json.py --input-dir /path/to/mds --output-dir /path/to/jsons

  # Custom LLM endpoint
  .venv/bin/python3 md_to_json.py --llm-base-url http://localhost:8080/v1 --llm-model qwen3.6-27b
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

import instructor
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = "/home/shreyas/Documents/02_Adalatdata/data_pipeline"
MD_DIR = os.path.join(BASE_DIR, "extracted_mds")
JSON_DIR = os.path.join(BASE_DIR, "extracted_jsons")

LLM_BASE_URL = "http://127.0.0.1:8080/v1"
LLM_MODEL = "qwen3.6-27b"

# ---------------------------------------------------------------------------
# Pydantic models for structured entity extraction
# ---------------------------------------------------------------------------

class Party(BaseModel):
    """A party involved in the case."""
    name: str = Field(description="Name of the party")
    role: str = Field(
        description="Role: petitioner, respondent, appellant, respondent-in-appeal, etc."
    )


class Judge(BaseModel):
    """A judge who participated in deciding the case."""
    name: str = Field(description="Full name of the judge")
    role: str = Field(
        default="bench_member",
        description="Role: author, bench_member, dissenting",
    )


class LegalSection(BaseModel):
    """A section of an Act or law cited in the judgment."""
    section: str = Field(
        description="Section number, e.g. '302', '498A', 'Article 21', 'Section 34'"
    )
    act: Optional[str] = Field(
        default=None,
        description="Name of the Act if identifiable, e.g. 'Indian Penal Code', 'Constitution of India', 'CrPC'",
    )


class LegalTopic(BaseModel):
    """A legal topic or concept relevant to the case."""
    text: str = Field(description="Legal topic keyword or phrase")
    category: str = Field(
        description="Category: constitutional, criminal, civil, property, family, administrative, tax, corporate, environmental, procedural, other"
    )


class CaseTitle(BaseModel):
    """The title/heading of the case."""
    title: Optional[str] = Field(
        default=None,
        description="Formal case title, e.g. 'State of Maharashtra v. Bharat Raghoba Bhor'",
    )


class JudgmentSummary(BaseModel):
    """A brief summary of the judgment."""
    summary: str = Field(default="", description="A concise 2-3 sentence summary covering the core legal issue and ruling/outcome")


class ExtractedEntities(BaseModel):
    """Complete structured extraction from a Supreme Court judgment."""
    case_title: CaseTitle = Field(default_factory=CaseTitle)
    summary: JudgmentSummary = Field(default_factory=JudgmentSummary)
    judges: List[Judge] = Field(default_factory=list)
    parties: List[Party] = Field(default_factory=list)
    sections: List[LegalSection] = Field(default_factory=list)
    topics: List[LegalTopic] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

LLM_CLIENT = None


def create_llm_client():
    openai_client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
    return instructor.from_openai(openai_client, mode=instructor.Mode.JSON)


def get_llm_client():
    global LLM_CLIENT
    if LLM_CLIENT is None:
        print(f"  Connecting to LLM at {LLM_BASE_URL} (model: {LLM_MODEL})...")
        try:
            LLM_CLIENT = create_llm_client()
            # Test connection
            LLM_CLIENT.chat.completions.create(
                response_model=ExtractedEntities,
                messages=[{"role": "user", "content": "Empty text."}],
                model=LLM_MODEL,
            )
            print("  LLM connection OK.")
        except Exception as e:
            print(f"  ERROR: LLM connection failed: {e}")
            print("  Extraction will output empty entities.")
            LLM_CLIENT = None
    return LLM_CLIENT

# ---------------------------------------------------------------------------
# Markdown text extraction
# ---------------------------------------------------------------------------

def extract_judgment_text(md_content: str) -> str:
    """Strip OCR metadata headers and page markers, keep judgment text."""
    lines = md_content.split("\n")
    cleaned = []
    skip = True

    for line in lines:
        # Skip file-level metadata header
        if line.startswith("# 2016_") or line.startswith("# 2016-"):
            continue
        if line.startswith("*Converted via"):
            continue
        if line == "---":
            # First --- marks end of metadata
            if skip:
                skip = False
                continue
            else:
                cleaned.append(line)
                continue
        if skip:
            continue
        cleaned.append(line)

    # Clean up: remove page headers (## Page N) but keep other headers
    text = "\n".join(cleaned)
    text = re.sub(r'\n## Page \d+\n', '\n', text)
    return text.strip()

# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a legal NLP extractor for the Indian Supreme Court. From the judgment text below,
extract structured entities. Be precise and only include entities clearly present in the text.

Return as few false positives as possible. If an entity type is not found, return an empty list.

Guidelines:
- Judges: Look for "Per Justice X", "X, J.", "delivered by", "[NAME, JJ.]". Names are Capitalized case.
- Parties: Look for case title format "A v. B", petitioner/respondent sections, appellant/respondent.
- Sections: Look for "Section X of Y Act", "s. X", "Article X", etc.
- Topics: Identify 3-8 key legal topics/categories that apply to this case.
- Summary: Write a concise 2-3 sentence summary covering the core issue and ruling.
- Case title: Extract the formal case title if present (e.g. "State of X v. Y").

Judgment text:
---
{text}
---
"""


def extract_entities(md_text: str) -> ExtractedEntities:
    """Use LLM to extract structured entities from judgment markdown."""
    client = get_llm_client()
    if client is None:
        return ExtractedEntities()

    # Truncate to fit LLM context window
    truncated = md_text[:15000]
    try:
        return client.chat.completions.create(
            response_model=ExtractedEntities,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=truncated)}],
            model=LLM_MODEL,
        )
    except Exception as e:
        print(f"    LLM extraction error: {e}")
        return ExtractedEntities()

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(filename: str) -> dict:
    """Parse filename like '2016-1-49-55-en' or '2016_1_49_55_EN' into metadata."""
    name = os.path.splitext(filename)[0]
    # Handle both hyphen and underscore separators
    if "-" in name:
        parts = name.split("-")
    else:
        parts = name.split("_")

    if len(parts) >= 4:
        return {
            "year": int(parts[0]),
            "month": int(parts[1]),
            "page_start": int(parts[2]),
            "page_end": int(parts[3]),
            "language": parts[4] if len(parts) > 4 else "EN",
        }
    return {"year": 0, "month": 0, "page_start": 0, "page_end": 0, "language": "EN"}

# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def to_json(entities: ExtractedEntities, filename: str, info: dict, raw_text: str) -> dict:
    """Build JSON output dictionary."""
    return {
        "filename": filename,
        "metadata": info,
        "entities": entities.model_dump(),
        "raw_text_preview": raw_text[:5000],
    }

# ---------------------------------------------------------------------------
# Process single file
# ---------------------------------------------------------------------------

def process_md(md_path, json_dir, llm_model):
    """Extract entities from a single markdown file and write JSON."""
    global LLM_MODEL
    if llm_model:
        LLM_MODEL = llm_model

    filename = os.path.basename(md_path)

    # Skip if JSON already exists (try both naming conventions)
    json_name = os.path.splitext(filename)[0] + ".json"
    json_path = os.path.join(json_dir, json_name)

    if os.path.exists(json_path):
        return False, "exists"

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    info = parse_filename(filename)
    md_text = extract_judgment_text(md_content)

    if not md_text.strip():
        print(f"  SKIP (empty): {filename}")
        return False, "empty"

    entities = extract_entities(md_text)

    # Write JSON
    json_output = to_json(entities, filename, info, md_text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False, default=str)

    # Progress
    judges_str = ", ".join(j.name for j in entities.judges[:2])
    has_data = bool(entities.judges or entities.sections or entities.topics)
    if has_data:
        print(f"  {filename} | judges: [{judges_str}] "
              f"sections: {len(entities.sections)} topics: {len(entities.topics)}")
    else:
        print(f"  {filename} | (text extracted, no entities)")

    return True, "extracted"

# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def run(input_dir, output_dir, llm_model):
    """Process all markdown files in input_dir."""
    # Recursively find all .md files
    md_files = sorted(Path(input_dir).rglob("*.md"))
    total = len(md_files)

    if not total:
        print(f"No markdown files found in {input_dir}")
        return

    print(f"\nProcessing {total} markdown files from {input_dir}")
    print(f"  LLM:      {LLM_BASE_URL} (model: {llm_model or LLM_MODEL})")
    print(f"  JSON out: {output_dir}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Verify LLM connection before batch
    get_llm_client()

    extracted = 0
    skipped = 0

    for i, md_path in enumerate(md_files, 1):
        filename = md_path.name
        print(f"  [{i}/{total}] Extract: {filename} ...")

        success, reason = process_md(str(md_path), output_dir, llm_model)
        if success:
            extracted += 1
        else:
            skipped += 1
            if reason == "exists":
                print(f"  [{i}/{total}] SKIP (JSON exists): {filename}")
            elif reason == "empty":
                print(f"  [{i}/{total}] SKIP (empty): {filename}")

    print(f"\nDone! Extracted {extracted}, skipped {skipped} out of {total} files.")
    print(f"  JSON output: {output_dir}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1b: Markdown to JSON via LLM entity extraction"
    )
    parser.add_argument("--input-dir", type=str, default=None, help="Input markdown directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output JSON directory")
    parser.add_argument("--llm-base-url", type=str, default=None, help="LLM API base URL")
    parser.add_argument("--llm-model", type=str, default=None, help="LLM model name")
    args = parser.parse_args()

    global LLM_BASE_URL, LLM_MODEL
    if args.llm_base_url:
        LLM_BASE_URL = args.llm_base_url
    if args.llm_model:
        LLM_MODEL = args.llm_model

    input_dir = args.input_dir or MD_DIR
    output_dir = args.output_dir or JSON_DIR

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    print(f"LLM:   {LLM_BASE_URL} / {LLM_MODEL}")
    print(f"Input: {input_dir}")
    print(f"Out:   {output_dir}")
    run(input_dir, output_dir, args.llm_model)

if __name__ == "__main__":
    main()
