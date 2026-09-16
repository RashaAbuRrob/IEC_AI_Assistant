# IEC Arabic RAG (local, offline, Ollama-only)

A fully local, offline retrieval-augmented QA system for Arabic PDF
documents (built for الهيئة المستقلة للانتخاب). No cloud calls: generation,
embeddings, and (when needed) vision transcription all run through a local
Ollama server.

## Stack
- **Generation:** `qwen3:14b` (text-only)
- **Embeddings:** `bge-m3` (via custom Chroma `EmbeddingFunction` — Chroma's
  built-in default embedder is not bge-m3)
- **Vision fallback (OCR-of-last-resort):** `qwen2.5vl`
- **Vector DB:** ChromaDB `PersistentClient` (on-disk, not ephemeral)
- **PDF parsing:** PyMuPDF (`fitz`)
- **OCR:** `pytesseract`, `lang="ara"`

## One-time setup

```bash
pip install -r requirements.txt

# System dependency for pytesseract's Arabic OCR:
#   Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-ara
#   macOS:         brew install tesseract tesseract-lang

# Pull the three models via Ollama:
ollama pull bge-m3
ollama pull qwen3:14b
ollama pull qwen2.5vl

ollama serve   # if not already running
```

## 1. Ingest your PDFs

```bash
python ingest.py --input-dir ./source_pdfs --db-path ./chroma_db
```

For each page, this:
1. Extracts native text with PyMuPDF and computes `char_count`,
   `char_density` (chars / sq inch), and `image_count`.
2. Flags the page if `char_count < 40` OR `char_density < 8` OR
   `image_count > 0`.
3. Flagged pages are rasterized at 200 DPI and OCR'd with `pytesseract`
   (`lang="ara"`).
4. If the page still has fewer than 40 characters after OCR **and** it
   contains at least one embedded image, the page image is sent to
   `qwen2.5vl` for transcription (last resort — vision inference is slow).
5. Runs a heuristic RTL-reversal spot-check on any digit runs found (common
   PDF-extraction bug where dates/numbers come out reversed). This only
   **logs a warning** for manual review — it never auto-"fixes" a number,
   since that risks silently corrupting a genuinely correct figure.
6. Text is chunked on paragraph → sentence boundaries only (see
   `arabic_utils.chunk_page_text`) — never on a fixed character count,
   which would otherwise split words or detached diacritics mid-token.
7. Each chunk is normalized (`arabic_utils.normalize_arabic`) before being
   embedded and stored. The **same normalization function** is called again
   at query time in `rag_service.py`, so index-time and query-time text
   are guaranteed to go through identical rules. The original (raw) chunk
   text is preserved in metadata for accurate citation display.
8. Every page's outcome (method used: `native` / `ocr` / `vision`, char
   stats, RTL warnings) is appended to `<db_path>/ingestion_log.jsonl`.

**After ingesting, check `ingestion_log.jsonl`** for any `rtl_warnings` or
`method != "native"` entries and spot-check those specific pages by eye —
this is the one step that can't be fully automated away.

## 2. Ask questions

```bash
# Interactive:
python rag_service.py --db-path ./chroma_db

# One-shot:
python rag_service.py --db-path ./chroma_db --query "ما هي شروط الترشح؟" --k 5
```

The system prompt instructs `qwen3:14b` to answer **only** from the
retrieved chunks, to cite `(المصدر: source_file، صفحة N)` or
`(المصدر: source_file، المادة N)` for every claim, and to respond with the
exact fixed refusal string —
`"لا تتوفر لدي معلومات كافية للإجابة على هذا السؤال"` — when the retrieved
chunks don't cover the question. If retrieval returns zero results, the
refusal is returned directly without calling the LLM.

## 3. Evaluate

1. Write 20–30 real Q&A pairs (plus a handful of deliberately out-of-scope
   questions) into a JSON file shaped like `eval_qa_template.json`. Pull the
   in-scope questions from the actual source documents so you can fill in
   `expected_source_file` / `expected_page` / `expected_article` correctly —
   these are what the eval script checks retrieval and citations against.

2. Run:

```bash
python eval.py --db-path ./chroma_db --qa-file eval_qa.json --k 5 \
    --report-out eval_report.md --json-out eval_report.json
```

This reports three metrics **separately** (never blended into one score),
per spec:
- **Retrieval hit-rate** — did the top-k retrieved chunks include the
  expected source (file + page or article) for in-scope questions?
- **Citation accuracy** — did the generated answer's citation actually
  reference the expected source?
- **Correct-refusal rate** — for the deliberately out-of-scope questions,
  did the system correctly emit the exact refusal string instead of
  fabricating an answer?

It also reports a false-refusal rate on in-scope questions as a sanity
check (a system that refuses everything would otherwise look great on
correct-refusal rate alone).

## 4. Chat UI (optional)

A small Flask API (`api_server.py`) wraps the retrieval + generation logic
behind the exact NDJSON streaming contract that `static/index.html` (the
Arabic RTL chat UI) expects:

```bash
python api_server.py --db-path ./chroma_db --port 5001
```

Then open **http://localhost:5001/** in a browser — Flask serves
`static/index.html` directly, and it's already wired to call
`POST /api/assistant/query`.

Request body: `{"query": "...", "history": [...], "stream": true}`
Response: newline-delimited JSON, one object per line:
- `{"type": "sources", "context_sources": [{"filename": "...", "page": N, "article_number": "12"}]}` — sent once, right after retrieval
- `{"type": "text", "text": "<fragment>"}` — streamed as `qwen3:14b` generates the answer

**Known simplification:** the UI sends conversation `history` with every
request, but the backend currently answers each question independently
(no multi-turn context folded into retrieval or the prompt) — this keeps
the strict "answer only from retrieved chunks, cite every claim" behavior
simple and auditable. If you want follow-up questions like "وماذا عن
المادة التالية؟" to work, the natural extension point is
`api_server.assistant_query()`: pass the last user/assistant turn into the
retrieval query and into `user_prompt` alongside the retrieved context.

## Design notes / known trade-offs

- **Embedding calls are one-at-a-time** (per the required `BgeM3EF`
  signature calling `/api/embed` per text). For very large corpora this is
  the ingestion bottleneck; if that matters, batch pages before calling
  `collection.add()` and consider parallelizing `embed_text` calls with a
  small thread pool — not done here to keep the reference implementation
  simple and match the spec exactly.
- **Arabic normalization** folds tashkeel, tatweel, and hamza-carrying alef
  variants (أ/إ/آ/ٱ → ا) for search purposes, but deliberately leaves `ة`
  and `ى` untouched, since those can carry legal meaning in regulatory
  Arabic text. This is a fixed, documented rule applied identically at
  index and query time — if you want to change it, change it in exactly
  one place (`arabic_utils.normalize_arabic`) so the two never drift apart.
- **`qwen3:14b`'s "thinking" mode is turned off** (`"think": False` in the
  Ollama call) since this is a grounded-QA task where we want a direct,
  citation-bound answer rather than exposed chain-of-thought.
- **Vision transcription is the slowest path** and is only invoked when
  both OCR and image_count>0 conditions are met — expect it to be used
  rarely (e.g. scanned stamps, embedded photos of signatures/seals).

## Out of scope (per spec)
No UI, no auth, no hosting/deployment tooling — this is ingestion +
CLI retrieval/generation + eval, meant to run entirely on one local
machine against local Ollama.
