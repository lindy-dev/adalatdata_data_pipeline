#!/usr/bin/env python3
"""
AdalatData Pipeline - Stage 1a: PDF to Markdown via PaddleOCR-VL

Reads Supreme Court judgment PDFs, renders pages as images, runs
PaddleOCR-VL-1.6 for OCR-to-markdown, and outputs markdown files.

Pipeline:
  PDF -> page images (PyMuPDF) -> OCR markdown (PaddleOCR-VL) -> .md files

Input:  raw_data/*.pdf
Output: extracted_mds/*.md

Requirements:
  - Run from paddleocr_venv (Python 3.12): ~/paddleocr_venv/bin/python3 ocr_extract.py
  - PaddleOCR-VL llama-server must be running on port 8083
  - pip install paddleocr paddlex paddlepaddle pymupdf

Server startup (in separate terminal):
  llama-server \\
    -m /media/shreyas/E/Models/models--PaddlePaddle--PaddleOCR-VL-1.6-GGUF/snapshots/c75b67fac5d3c5a389857cf82243011251fec612/PaddleOCR-VL-1.6-GGUF.gguf \\
    --mmproj /media/shreyas/E/Models/models--PaddlePaddle--PaddleOCR-VL-1.6-GGUF/snapshots/c75b67fac5d3c5a389857cf82243011251fec612/PaddleOCR-VL-1.6-GGUF-mmproj.gguf \\
    --port 8083 --host 127.0.0.1 --temp 0 -t 16 -ngl 999

Usage:
  cd /home/shreyas/Documents/02_Adalatdata/data_pipeline
  ~/paddleocr_venv/bin/python3 ocr_extract.py

  # Custom paths
  ~/paddleocr_venv/bin/python3 ocr_extract.py --input-dir /path/to/pdfs --output-dir /path/to/mds

  # Process all pages (default: 30)
  ~/paddleocr_venv/bin/python3 ocr_extract.py --max-pages 9999
"""

import os
import sys
import re
import shutil
import tempfile
import argparse
import time
import base64
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

# Auto-detect and relaunch from paddleocr_venv if needed
PADDLEOCR_VENV_PYTHON = "/home/shreyas/.hermes/profiles/local-llama/home/paddleocr_venv/bin/python3"

def ensure_paddleocr_venv():
    """Relaunch from paddleocr_venv if paddleocr is not available."""
    if PADDLEOCR_VENV_PYTHON in sys.executable:
        return
    try:
        import paddleocr
    except ImportError:
        if os.path.exists(PADDLEOCR_VENV_PYTHON):
            print(f"paddleocr not available in current Python ({sys.executable}).")
            print(f"Relaunching from paddleocr_venv...")
            os.execv(PADDLEOCR_VENV_PYTHON, [PADDLEOCR_VENV_PYTHON, __file__] + sys.argv[1:])
        else:
            print(f"ERROR: paddleocr not installed and paddleocr_venv not found at:")
            print(f"  {PADDLEOCR_VENV_PYTHON}")
            sys.exit(1)

ensure_paddleocr_venv()

from paddleocr import PaddleOCRVL
import fitz

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = "/home/shreyas/Documents/02_Adalatdata/data_pipeline"
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
MD_DIR = os.path.join(BASE_DIR, "extracted_mds")

SERVER_URL = "http://127.0.0.1:8083/v1"
SERVER_PORT = 8083
DPI = 72
MAX_PAGES = 30

# ---------------------------------------------------------------------------
# Server check
# ---------------------------------------------------------------------------

def check_server():
    """Verify PaddleOCR-VL llama-server is running."""
    try:
        resp = urlopen(f"{SERVER_URL}/models", timeout=5)
        if resp.getcode() == 200:
            print(f"  PaddleOCR server OK on port {SERVER_PORT}")
            return True
    except (URLError, Exception):
        pass
    print(f"  ERROR: PaddleOCR server not responding on port {SERVER_PORT}")
    print(f"  Start it first:")
    print(f"    llama-server -m model.gguf --mmproj mmproj.gguf")
    print(f"    --port {SERVER_PORT} --host 127.0.0.1 --temp 0 -t 16 -ngl 999")
    return False

# ---------------------------------------------------------------------------
# OCR per page
# ---------------------------------------------------------------------------

def extract_page_text(image_path, pdf_path, page_num, pipeline, dpi):
    """Run PaddleOCRVL on an image and return markdown with embedded images."""
    tmp_dir = tempfile.mkdtemp()
    try:
        output = pipeline.predict(image_path)

        # Result object is dict-like — access with dict syntax
        res_dict = dict(output[0]) if output else {}
        result_data = res_dict.get('res', res_dict)
        layout_boxes = result_data.get("layout_det_res", {}).get("boxes", [])

        # Find image blocks from layout detection
        image_blocks = [b for b in layout_boxes if b.get("label") == "image"]

        # Extract image regions from the original PDF page
        pdf_doc = fitz.open(pdf_path)
        pdf_page = pdf_doc[page_num]
        embedded_images = []
        for idx, img_block in enumerate(image_blocks):
            coords = img_block["coordinate"]
            clip = fitz.Rect(coords)
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = pdf_page.get_pixmap(matrix=mat, clip=clip)
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode()
            embedded_images.append((coords, b64))
        pdf_doc.close()

        # Build markdown from PaddleOCR results
        # save_to_markdown writes a .md file to tmp_dir (returns file path)
        md_parts = []
        for res in output:
            res.save_to_markdown(save_path=tmp_dir)
            md_files = [f for f in os.listdir(tmp_dir) if f.endswith('.md')]
            for mf in md_files:
                with open(os.path.join(tmp_dir, mf), 'r') as f:
                    md_parts.append(f.read())
            for mf in md_files:
                os.remove(os.path.join(tmp_dir, mf))
        base_md = "\n".join(md_parts)

        # Strip broken HTML image references from PaddleOCR
        base_md = re.sub(r'<div[^>]*><img[^>]*/?[\s>]*/div>', '', base_md)
        base_md = re.sub(r'<img[^/]*/>', '', base_md)
        base_md = re.sub(r'<div[^>]*>\s*</div>', '', base_md)

        # Embed images at the end with named references
        image_refs = []
        for i, (coords, b64) in enumerate(embedded_images):
            ref_name = f"img_page{page_num}_{i+1}"
            image_refs.append((ref_name, b64))

        if image_refs:
            base_md += "\n\n<!-- Page images -->\n\n"
            for ref_name, b64 in image_refs:
                base_md += f"\n\n![{ref_name}](data:image/png;base64,{b64})\n\n"

        return base_md
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Full PDF processing
# ---------------------------------------------------------------------------

def pdf_to_markdown(pdf_path, output_path, max_pages, dpi):
    """Render PDF pages as images, run PaddleOCR-VL, output markdown."""
    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_backend="llama-cpp-server",
        vl_rec_server_url=SERVER_URL,
    )

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages_to_process = min(total_pages, max_pages)

    if total_pages == 0:
        print(f"  WARN: Empty PDF: {os.path.basename(pdf_path)}")
        doc.close()
        return False

    print(f"  Pages: {total_pages} (processing {pages_to_process})")
    results = []
    start_time = time.time()

    for page_index in range(pages_to_process):
        page = doc[page_index]

        # Render page as PNG
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_path = os.path.join(tempfile.gettempdir(), f"page_{page_index}.png")
        pix.save(img_path)

        # OCR via PaddleOCR-VL
        page_num = page_index + 1
        t_page = time.time()

        try:
            ocr_text = extract_page_text(img_path, pdf_path, page_index, pipeline, dpi)
            elapsed = time.time() - t_page
            results.append(ocr_text)
            print(f"    Page {page_num}/{pages_to_process} ({len(ocr_text)} chars, {elapsed:.1f}s)")
        except Exception as e:
            print(f"    WARN: OCR failed for page {page_num}: {e}")
            results.append(f"[Error processing page {page_num}]")

        os.unlink(img_path)

    doc.close()
    total_time = time.time() - start_time

    # Write markdown file
    filename = os.path.basename(pdf_path)
    name_without_ext = os.path.splitext(filename)[0]
    output_name = name_without_ext + ".md"
    output_path_full = os.path.join(os.path.dirname(output_path), output_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path_full, "w", encoding="utf-8") as f:
        f.write(f"# {name_without_ext.replace('_', '-')}\n\n")
        f.write(f"*Converted via PaddleOCR-VL-1.6 | {pages_to_process} pages | {total_time:.0f}s*\n\n")
        for i, text in enumerate(results):
            f.write(f"---\n\n## Page {i + 1}\n\n")
            f.write(text + "\n\n")

    print(f"  Saved: {output_path_full}")
    return True

def run(input_dir, output_dir, max_pages, dpi, server_url):
    """Process all PDFs in input_dir."""
    global SERVER_URL
    if server_url:
        SERVER_URL = server_url

    pdf_files = sorted(Path(input_dir).glob("*.pdf"))
    total = len(pdf_files)

    if not total:
        print(f"No PDFs found in {input_dir}")
        return

    # Check server before processing
    if not check_server():
        sys.exit(1)

    print(f"\nProcessing {total} PDFs")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  DPI:    {dpi}, Max pages: {max_pages}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    success = 0
    for i, pdf_path in enumerate(pdf_files, 1):
        filename = pdf_path.name

        # Skip if markdown already exists
        output_path = os.path.join(output_dir, filename.replace(".pdf", ".md"))
        if os.path.exists(output_path):
            print(f"  [{i}/{total}] SKIP (exists): {filename}")
            continue

        print(f"  [{i}/{total}] OCR: {filename}")
        if pdf_to_markdown(str(pdf_path), output_path, max_pages, dpi):
            success += 1

    print(f"\nDone! Converted {success}/{total} PDFs to markdown.")
    print(f"  Output: {output_dir}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1a: PDF to Markdown via PaddleOCR-VL"
    )
    parser.add_argument("--input-dir", type=str, default=None, help="Input PDF directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output markdown directory")
    parser.add_argument("--server-url", type=str, default=None, help="PaddleOCR server URL")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages per PDF (default: 30)")
    parser.add_argument("--dpi", type=int, default=72, help="Image DPI for page rendering")
    parser.add_argument("--venv-python", type=str, default=None,
        help="Path to paddleocr_venv Python (default: auto-detect)")
    args = parser.parse_args()

    global PADDLEOCR_VENV_PYTHON
    if args.venv_python:
        PADDLEOCR_VENV_PYTHON = args.venv_python

    input_dir = args.input_dir or RAW_DIR
    output_dir = args.output_dir or MD_DIR
    server_url = args.server_url or SERVER_URL
    max_pages = args.max_pages if args.max_pages is not None else MAX_PAGES

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    run(input_dir, output_dir, max_pages, args.dpi, server_url)

if __name__ == "__main__":
    main()
