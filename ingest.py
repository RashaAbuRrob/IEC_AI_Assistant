# -*- coding: utf-8 -*-
"""
Ingestion pipeline for the IEC Arabic RAG system.

Usage:
    python ingest.py --input-dir ./source_pdfs --db-path ./chroma_db

Per page:
  1. Extract native text via PyMuPDF.
  2. Compute char_count, char_density (chars / sq inch), image_count.
  3. Flag page if char_count<40 OR char_density<8 OR image_count>0.
  4. Flagged pages: rasterize at 200 DPI, OCR with pytesseract (lang="ara").
  5. If still <40 chars after OCR AND image_count>0: send the page image to
     qwen2.5vl (a vision model -- qwen3:14b is text-only) for transcription.
  6. Spot-check pages containing numbers/dates for RTL-reversal bugs (logged,
     never auto-corrected).
  7. Log per-page extraction method to <db_path>/ingestion_log.jsonl.

Chunks are normalized (arabic_utils.normalize_arabic) before being embedded
and stored, so retrieval-time normalization matches exactly. The original,
un-normalized page text is kept in chunk metadata as `raw_text` so citations
can show the real source wording if needed.
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from arabic_utils import chunk_page_text, looks_like_font_corruption, normalize_arabic, spot_check_rtl_reversal
from chroma_utils import get_collection
from ollama_utils import prewarm_embedder, vision_transcribe

MIN_CHARS = 40
MIN_DENSITY = 8.0  # chars per square inch
OCR_DPI = 200

# Windows: tesseract.exe isn't on PATH by default, and the Arabic language
# pack isn't in the system tessdata dir (no admin rights required this way).
# Point pytesseract at the local install + the ara.traineddata shipped in
# ./tessdata. Harmless no-op on systems where tesseract is already on PATH
# with "ara" installed system-wide (e.g. Linux/macOS per the README).
_DEFAULT_WIN_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_LOCAL_TESSDATA_DIR = str(Path(__file__).parent / "tessdata")

if os.name == "nt" and os.path.isfile(_DEFAULT_WIN_TESSERACT):
    pytesseract.pytesseract.tesseract_cmd = _DEFAULT_WIN_TESSERACT
if os.path.isfile(os.path.join(_LOCAL_TESSDATA_DIR, "ara.traineddata")):
    os.environ.setdefault("TESSDATA_PREFIX", _LOCAL_TESSDATA_DIR)


def compute_page_stats(page: "fitz.Page", native_text: str):
    rect = page.rect
    width_in = rect.width / 72.0
    height_in = rect.height / 72.0
    area_sqin = max(width_in * height_in, 1e-6)
    char_count = len(native_text.strip())
    char_density = char_count / area_sqin
    image_count = len(page.get_images(full=True))
    return char_count, char_density, image_count


def needs_fallback(char_count: int, char_density: float, image_count: int) -> bool:
    return char_count < MIN_CHARS or char_density < MIN_DENSITY or image_count > 0


def rasterize_page(page: "fitz.Page", dpi: int = OCR_DPI) -> bytes:
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    return pix.tobytes("png")


def ocr_page(png_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(png_bytes))
    return pytesseract.image_to_string(img, lang="ara")


def extract_page_text(page: "fitz.Page", log_entry: dict):
    """Returns (final_text, method) where method in {native, ocr, vision}."""
    native_text = page.get_text("text")
    char_count, char_density, image_count = compute_page_stats(page, native_text)
    # Some PDFs embed a subset Arabic font with a broken/missing
    # ToUnicode CMap: PyMuPDF then extracts the WRONG characters at
    # roughly the right length, so char_count/char_density look fine and
    # the checks below never trigger even though the "text" is nonsense.
    native_corrupt = looks_like_font_corruption(native_text)

    log_entry.update(
        char_count=char_count,
        char_density=round(char_density, 2),
        image_count=image_count,
    )
    if native_corrupt:
        log_entry["native_font_corruption_suspected"] = True

    if not needs_fallback(char_count, char_density, image_count) and not native_corrupt:
        return native_text, "native"

    # --- OCR fallback ---
    png_bytes = rasterize_page(page)
    ocr_text = ocr_page(png_bytes)
    ocr_char_count = len(ocr_text.strip())

    if native_corrupt:
        # Native text is known-bad regardless of length -- prefer OCR
        # outright rather than picking whichever has more characters.
        if ocr_char_count >= MIN_CHARS:
            return ocr_text, "ocr"
        # OCR also came up short -- fall through to vision below.
    elif ocr_char_count >= MIN_CHARS or image_count == 0:
        # OCR gave us enough text, OR there's no embedded image to justify
        # trying vision (i.e. this is just a sparse/mostly-blank native page).
        final_text = ocr_text if ocr_char_count > char_count else native_text
        method = "ocr" if ocr_char_count > char_count else "native"
        return final_text, method

    # --- Vision fallback (last resort) ---
    try:
        vision_text = vision_transcribe(png_bytes)
    except Exception as e:  # noqa: BLE001 -- log and degrade gracefully
        log_entry["vision_error"] = str(e)
        vision_text = ""

    if len(vision_text.strip()) > ocr_char_count:
        return vision_text, "vision"
    return ocr_text, "ocr"  # vision didn't help; keep whatever OCR produced


def ingest_pdf(pdf_path: Path, collection, log_fh, chunk_target_chars: int, chunk_max_chars: int):
    doc = fitz.open(pdf_path)
    total_chunks = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1  # 1-indexed for human-facing citations

        log_entry = {"source_file": pdf_path.name, "page": page_num}
        t0 = time.time()

        final_text, method = extract_page_text(page, log_entry)

        rtl_warnings = spot_check_rtl_reversal(final_text)
        if rtl_warnings:
            log_entry["rtl_warnings"] = rtl_warnings

        log_entry["method"] = method
        log_entry["final_char_count"] = len(final_text.strip())
        log_entry["elapsed_sec"] = round(time.time() - t0, 2)
        log_fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        log_fh.flush()

        if not final_text.strip():
            continue

        chunks = chunk_page_text(final_text, target_chars=chunk_target_chars, max_chars=chunk_max_chars)
        if not chunks:
            continue

        ids, documents, metadatas = [], [], []
        for chunk_index, (raw_chunk, article_number) in enumerate(chunks):
            normalized = normalize_arabic(raw_chunk)
            if not normalized:
                continue
            ids.append(f"{pdf_path.stem}_p{page_num}_c{chunk_index}")
            documents.append(normalized)
            metadatas.append(
                {
                    "source_file": pdf_path.name,
                    "page": page_num,
                    "article_number": article_number or "",
                    "extraction_method": method,
                    "raw_text": raw_chunk,
                }
            )

        if documents:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
            total_chunks += len(documents)

    doc.close()
    return total_chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the IEC Arabic RAG ChromaDB collection.")
    parser.add_argument("--input-dir", required=True, help="Directory containing source PDF files.")
    parser.add_argument("--db-path", required=True, help="Path for the on-disk ChromaDB PersistentClient store.")
    parser.add_argument("--chunk-target-chars", type=int, default=900)
    parser.add_argument("--chunk-max-chars", type=int, default=1400)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")

    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        sys.exit(f"No PDF files found in {input_dir}")

    db_path = Path(args.db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    log_path = db_path / "ingestion_log.jsonl"

    print(f"Pre-warming bge-m3 embedder via Ollama...")
    prewarm_embedder()

    collection = get_collection(str(db_path))

    grand_total = 0
    with open(log_path, "a", encoding="utf-8") as log_fh:
        for pdf_path in pdf_paths:
            print(f"Ingesting {pdf_path.name} ...")
            n = ingest_pdf(pdf_path, collection, log_fh, args.chunk_target_chars, args.chunk_max_chars)
            print(f"  -> {n} chunks added.")
            grand_total += n

    print(f"\nDone. {grand_total} total chunks across {len(pdf_paths)} file(s).")
    print(f"Collection now has {collection.count()} documents.")
    print(f"Per-page ingestion log: {log_path}")
    print(
        "\nReminder: scan ingestion_log.jsonl for entries with 'rtl_warnings' "
        "or method != 'native' and spot-check those pages manually."
    )


if __name__ == "__main__":
    main()
