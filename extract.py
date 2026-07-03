#!/usr/bin/env python3
"""
AdalatData Pipeline - Stage 1: Extract

Reads Supreme Court judgment PDFs, renders pages as images, runs glm-ocr
for OCR-to-markdown, then runs LLM-powered entity extraction, and outputs
structured JSON + Markdown files.

Pipeline:
  PDF -> page images (PyMuPDF) -> OCR markdown (glm-ocr) ->
  entity extraction (LLM) -> JSON + Markdown

Input:  raw_data/*.pdf
Output: extracted_jsons/*.json
        extracted_mds/*.md

Usage:
  pip install pymupdf instructor openai
  python extract.py
"""

import os
import sys
import base64
import argparse
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

import fitz  # PyMuPDF
import instructor
from openai import OpenAI
import json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = "/home/shreyas/Documents/02_Adalatdata/data_pipeline"
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
JSON_DIR = os.path.join(BASE_DIR, "extracted_jsons")
MD_DIR = os.path.join(BASE_DIR, "extracted_mds")

# OCR endpoint (glm-ocr)
OCR_BASE_URL = "http://127.0.0.1:1234/v1"
OCR_MODEL = "glm-ocr"

# Entity extraction LLM
LLM_BASE_URL = "http://127.0.0.1:8080/v1"
LLM_MODEL = "qwen3.5-9b"

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
    summary: str = Field(
        description="A concise 2-3 sentence summary covering the core legal issue and ruling/outcome"
    )


class ExtractedEntities(BaseModel):
    """Complete structured extraction from a Supreme Court judgment."""
    case_title: CaseTitle = Field(default_factory=CaseTitle)
    summary: JudgmentSummary = Field(default_factory=JudgmentSummary)
    judges: List[Judge] = Field(default_factory=list)
    parties: List[Party] = Field(default_factory=list)
    sections: List[LegalSection] = Field(default_factory=list)
    topics: List[LegalTopic] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

ocr_client = OpenAI(base_url=OCR_BASE_URL, api_key="not-needed")


def create_llm_client():
    openai_client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
    return instructor.from_openai(openai_client, mode=instructor.Mode.JSON)


LLM_CLIENT = None


def get_llm_client():
    global LLM_CLIENT
    if LLM_CLIENT is None:
        print(f"  Connecting to LLM at {LLM_BASE_URL} (model: {LLM_MODEL})...")
        try:
            LLM_CLIENT = create_llm_client()
            LLM_CLIENT.chat.completions.create(
                response_model=ExtractedEntities,
                messages=[{"role": "user", "content": "Empty text."}],
                model=LLM_MODEL,
            )
            print("  LLM connection OK.")
        except Exception as e:
            print(f"  ERROR: LLM connection failed: {e}")
            print("  Extraction will be skipped; text will still be saved as JSON.")
            LLM_CLIENT = None
    return LLM_CLIENT


# ---------------------------------------------------------------------------
# OCR via glm-ocr
# ---------------------------------------------------------------------------

def pdf_to_ocr_text(pdf_path, max_pages=30):
    """Render PDF pages as images and send to glm-ocr for OCR markdown."""
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_process = min(total_pages, max_pages)
    except Exception as e:
        print(f"  WARN: Failed to open {os.path.basename(pdf_path)}: {e}")
        return ""

    all_text = []
    for page_index in range(pages_to_process):
        page = doc[page_index]
        try:
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            image_url = f"data:image/png;base64,{img_b64}"
        except Exception as e:
            print(f"    WARN: Failed to render page {page_index + 1}: {e}")
            continue

        try:
            response = ocr_client.chat.completions.create(
                model=OCR_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                            {
                                "type": "text",
                                "text": "OCR markdown",
                            },
                        ],
                    }
                ],
            )
            ocr_text = response.choices[0].message.content
            if ocr_text and ocr_text.strip():
                all_text.append(ocr_text)
        except Exception as e:
            print(f"    WARN: OCR failed for page {page_index + 1}: {e}")
            continue

    doc.close()
    return "\n\n".join(all_text)


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a legal NLP extractor for the Indian Supreme Court. From the judgment text below,
extract structured entities. Be precise and only include entities clearly present in the text.

Return as few false positives as possible. If an entity type is not found, return an empty list.

Guidelines:
- Judges: Look for "Per Justice X", "X, J.", "delivered by", "bench", etc. Make sure the names are Capitalized case
- Parties: Look for case title format "A v. B", petitioner/respondent sections.
- Sections: Look for "Section X of Y Act", "Article X", etc.
- Topics: Identify 3-8 key legal topics/categories that apply to this case.
- Summary: Write a concise 2-3 sentence summary covering the core issue and ruling.
- Case title: Extract the formal case title if present.

Judgment text:
---
{text}
---
"""


def parse_filename(filename):
    """Parse '2016_1_1_23_EN.pdf' into structured metadata."""
    parts = filename.replace(".pdf", "").split("_")
    if len(parts) >= 4:
        return {
            "year": int(parts[0]),
            "month": int(parts[1]),
            "page_start": int(parts[2]),
            "page_end": int(parts[3]),
            "language": parts[4] if len(parts) > 4 else "EN",
        }
    return {"year": 0, "month": 0, "page_start": 0, "page_end": 0, "language": "EN"}


def extract_entities_llm(text: str) -> ExtractedEntities:
    """Use LLM to extract structured entities from judgment text."""
    client = get_llm_client()
    if client is None:
        return ExtractedEntities()

    truncated = text[:6000]
    try:
        return client.chat.completions.create(
            response_model=ExtractedEntities,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=truncated)}],
            model=LLM_MODEL,
        )
    except Exception as e:
        print(f"    LLM extraction error: {e}")
        return ExtractedEntities()


def to_markdown(entities: ExtractedEntities, info: dict, raw_text: str) -> str:
    """Convert extracted entities to a clean Markdown document."""
    lines = []
    year = info.get("year", "?")
    month = info.get("month", "?")
    pages = f"{info.get('page_start', '?')}-{info.get('page_end', '?')}"

    # Title
    title = entities.case_title.title or f"SC Judgment {year}/{month} pp.{pages}"
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Year:** {year}  ")
    lines.append(f"**Month:** {month}  ")
    lines.append(f"**Pages:** {pages}  ")
    lines.append(f"**Language:** {info.get('language', 'EN')}")
    lines.append("")

    # Summary
    if entities.summary.summary:
        lines.append("## Summary")
        lines.append("")
        lines.append(entities.summary.summary)
        lines.append("")

    # Judges
    if entities.judges:
        lines.append("## Bench")
        lines.append("")
        for j in entities.judges:
            role_label = j.role.replace("_", " ").title() if j.role else ""
            lines.append(f"- **{j.name}** ({role_label})")
        lines.append("")

    # Parties
    if entities.parties:
        lines.append("## Parties")
        lines.append("")
        for p in entities.parties:
            lines.append(f"- **{p.name}** ({p.role})")
        lines.append("")

    # Sections
    if entities.sections:
        lines.append("## Legal Sections Cited")
        lines.append("")
        for s in entities.sections:
            act_str = f" of {s.act}" if s.act else ""
            lines.append(f"- Section **{s.section}**{act_str}")
        lines.append("")

    # Topics
    if entities.topics:
        lines.append("## Legal Topics")
        lines.append("")
        for t in entities.topics:
            lines.append(f"- **{t.text}** ({t.category})")
        lines.append("")

    # Raw text (truncated)
    if raw_text:
        lines.append("---")
        lines.append("")
        lines.append("## Judgment Text")
        lines.append("")
        lines.append(raw_text[:15000])

    return "\n".join(lines)


def run(raw_dir, json_dir, md_dir):
    """Main extraction pipeline."""
    pdf_files = sorted(Path(raw_dir).glob("*.pdf"))
    total = len(pdf_files)

    if not total:
        print(f"No PDFs found in {raw_dir}")
        return

    print(f"\nProcessing {total} PDFs from {raw_dir}")
    print(f"  OCR endpoint:  {OCR_BASE_URL} (model: {OCR_MODEL})")
    print(f"  JSON output:   {json_dir}")
    print(f"  MD   output:   {md_dir}")
    print("=" * 60)

    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(md_dir, exist_ok=True)

    for i, pdf_path in enumerate(pdf_files, 1):
        filename = pdf_path.name
        info = parse_filename(filename)

        # OCR via glm-ocr (PDF -> images -> OCR markdown)
        print(f"  [{i}/{total}] OCR: {filename} ...")
        raw_text = pdf_to_ocr_text(str(pdf_path))
        if not raw_text.strip():
            print(f"  [{i}/{total}] SKIP (empty): {filename}")
            continue

        # Entity extraction via LLM
        entities = extract_entities_llm(raw_text)

        # Build JSON output
        json_output = {
            "filename": filename,
            "metadata": info,
            "entities": entities.model_dump(),
            "raw_text_preview": raw_text[:5000],
        }

        # Write JSON
        json_file = os.path.join(json_dir, filename.replace(".pdf", ".json"))
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False, default=str)

        # Write Markdown
        md_content = to_markdown(entities, info, raw_text)
        md_file = os.path.join(md_dir, filename.replace(".pdf", ".md"))
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        # Progress
        judges_str = ", ".join(j.name for j in entities.judges[:2])
        has_data = bool(entities.judges or entities.sections or entities.topics)
        if has_data:
            print(f"  [{i}/{total}] {filename} | judges: [{judges_str}] "
                  f"sections: {len(entities.sections)} topics: {len(entities.topics)}")
        else:
            print(f"  [{i}/{total}] {filename} | (text extracted, no entities)")

    print(f"\nDone! Extracted {i}/{total} files.")
    print(f"  JSON files: {json_dir}")
    print(f"  MD   files: {md_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Extract entities from SC judgment PDFs")
    parser.add_argument("--raw-dir", type=str, default=None, help="Override raw data dir")
    parser.add_argument("--json-dir", type=str, default=None, help="Override JSON output dir")
    parser.add_argument("--md-dir", type=str, default=None, help="Override MD output dir")
    parser.add_argument("--ocr-base-url", type=str, default=None, help="Override OCR base URL")
    parser.add_argument("--ocr-model", type=str, default=None, help="Override OCR model name")
    parser.add_argument("--llm-model", type=str, default=None, help="Override LLM model name")
    parser.add_argument("--llm-base-url", type=str, default=None, help="Override LLM base URL")
    args = parser.parse_args()

    global OCR_BASE_URL, OCR_MODEL, LLM_BASE_URL, LLM_MODEL
    if args.ocr_base_url:
        OCR_BASE_URL = args.ocr_base_url
    if args.ocr_model:
        OCR_MODEL = args.ocr_model
    if args.llm_base_url:
        LLM_BASE_URL = args.llm_base_url
    if args.llm_model:
        LLM_MODEL = args.llm_model

    # Re-create OCR client with updated URL/model
    global ocr_client
    ocr_client = OpenAI(base_url=OCR_BASE_URL, api_key="not-needed")

    raw_dir = args.raw_dir or RAW_DIR
    json_dir = args.json_dir or JSON_DIR
    md_dir = args.md_dir or MD_DIR

    if not os.path.isdir(raw_dir):
        print(f"ERROR: Raw data directory not found: {raw_dir}")
        sys.exit(1)

    print(f"OCR:   {OCR_BASE_URL} / {OCR_MODEL}")
    print(f"LLM:   {LLM_BASE_URL} / {LLM_MODEL}")
    print(f"Input: {raw_dir}")
    print(f"Output: {json_dir}, {md_dir}")
    run(raw_dir, json_dir, md_dir)


if __name__ == "__main__":
    main()
