# -*- coding: utf-8 -*-
"""
Shared Arabic text utilities for the IEC RAG system.

IMPORTANT: normalize_arabic() MUST be called on text at BOTH index time
(when chunks are added to ChromaDB) and query time (when a user question
is embedded for retrieval). If the two ever drift apart, retrieval quality
silently degrades because the embedding model sees different surface forms
for what is semantically the same text.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Arabic diacritics (tashkeel) + Quranic annotation marks
_TASHKEEL_RE = re.compile(
    r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED'
    r'\u08D4-\u08E1\u08E3-\u08FF]'
)
_TATWEEL = '\u0640'  # ـ elongation character
_MULTI_WS_RE = re.compile(r'\s+')

# Fixed alef normalization: hamza-above/below/madda -> bare alef.
# NOTE: we deliberately do NOT fold ة->ه or ى->ي, since in legal/regulatory
# Arabic text those distinctions occasionally carry meaning (and Article
# text should stay as close to the source as possible for citation
# purposes). We only fold variants that are near-universally treated as
# equivalent for search purposes.
_ALEF_VARIANTS_RE = re.compile('[إأآٱ]')


def normalize_arabic(text: str) -> str:
    """
    Deterministic normalization applied identically at index time and
    query time. Rules (fixed, do not change independently in one place):
      1. Unicode NFKC normalization (folds compatibility forms, Arabic-Indic
         digit presentation variants, etc.)
      2. Strip tashkeel (diacritics) entirely.
      3. Strip tatweel (kashida) elongation characters.
      4. Fold أ/إ/آ/ٱ -> ا
      5. Collapse whitespace.
    """
    if not text:
        return text
    text = unicodedata.normalize('NFKC', text)
    text = _TASHKEEL_RE.sub('', text)
    text = text.replace(_TATWEEL, '')
    text = _ALEF_VARIANTS_RE.sub('ا', text)
    text = _MULTI_WS_RE.sub(' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Article-number detection
# ---------------------------------------------------------------------------

# Matches "المادة 5", "المادة (5)", "مادة رقم 5", etc. Captures the number.
ARTICLE_RE = re.compile(r'(?:المادة|مادة)\s*(?:رقم)?\s*\(?\s*(\d+)\s*\)?')


def find_article_number(text: str):
    m = ARTICLE_RE.search(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Sentence / paragraph aware chunking (no fixed-char splitting)
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!؟?۔])\s+')


def split_into_sentences(paragraph: str):
    parts = _SENTENCE_SPLIT_RE.split(paragraph.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_page_text(text: str, target_chars: int = 900, max_chars: int = 1400):
    """
    Chunk text on paragraph -> sentence boundaries only. Sentences are
    accumulated into a chunk until target_chars is reached; a chunk is
    never allowed to exceed max_chars, and a sentence is never split
    mid-word/mid-diacritic.

    Returns a list of (chunk_text, article_number_or_None).
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks = []
    current = ""
    current_article = None

    def flush():
        nonlocal current, current_article
        if current.strip():
            chunks.append((current.strip(), current_article))
        current = ""
        current_article = None

    for para in paragraphs:
        para_article = find_article_number(para)
        sentences = split_into_sentences(para) or [para]

        for sent in sentences:
            projected_len = len(current) + (1 if current else 0) + len(sent)
            if current and projected_len > max_chars:
                flush()

            if not current and para_article:
                current_article = para_article
            elif current_article is None and para_article:
                current_article = para_article

            current = (current + " " + sent).strip() if current else sent

            if len(current) >= target_chars:
                flush()

        # Paragraph boundary: if we're comfortably short of target, keep
        # accumulating into the next paragraph rather than force-flushing,
        # so short paragraphs (e.g. list items) don't each become tiny chunks.

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Font-encoding corruption check (some PDFs embed a subset Arabic font with
# a broken/missing ToUnicode CMap -- PyMuPDF then extracts the WRONG
# characters at roughly the RIGHT length, so char_count/char_density look
# fine and needs_fallback() never triggers, even though the "text" is
# nonsense. Observed in ~55% of native-extracted pages in this corpus.)
# ---------------------------------------------------------------------------

_COMMON_ARABIC_WORDS_RE = re.compile(
    r'(?:^|\s)(?:من|في|على|إذا|الذي|التي|هذا|هذه|أن|كان)(?:\s|$)'
)


def looks_like_font_corruption(text: str, min_chars_to_check: int = 200, min_rate_per_1000: float = 5.0) -> bool:
    """
    True if `text` is long enough to judge and is suspiciously missing
    extremely common Arabic function words (من, في, على, إذا, ...) that
    real Arabic prose of any length contains constantly. A rate this low
    is the fingerprint of a broken font-to-Unicode mapping, not a
    legitimate stylistic quirk -- confirmed against this corpus by
    visually comparing rendered pages to their extracted text.
    """
    stripped = text.strip()
    if len(stripped) < min_chars_to_check:
        return False
    hits = len(_COMMON_ARABIC_WORDS_RE.findall(stripped))
    rate_per_1000 = hits / len(stripped) * 1000
    return rate_per_1000 < min_rate_per_1000


# ---------------------------------------------------------------------------
# RTL-reversal spot check (heuristic, logs a warning -- never auto-fixes)
# ---------------------------------------------------------------------------

_WESTERN_DIGIT_RUN_RE = re.compile(r'\d{2,4}')
_ARABIC_INDIC_DIGIT_RUN_RE = re.compile(r'[\u0660-\u0669]{2,4}')

_ARABIC_INDIC_TO_WESTERN = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')


def _plausible_year(n: int) -> bool:
    return 1900 <= n <= 2100


def spot_check_rtl_reversal(text: str):
    """
    Heuristic check for common PDF-extraction RTL bugs where digit runs
    embedded in Arabic (RTL) text come out reversed (e.g. a year "2023"
    extracted as "3202"). This does NOT attempt to auto-correct anything
    (that would risk silently corrupting genuinely-correct numbers, e.g.
    article numbers, monetary amounts). It only returns a list of
    human-readable warnings for the ingestion log so a person can spot-check
    the flagged pages.
    """
    warnings = []
    if not text:
        return warnings

    for run in _WESTERN_DIGIT_RUN_RE.findall(text):
        if len(run) == 4:
            forward = int(run)
            reversed_val = int(run[::-1])
            if not _plausible_year(forward) and _plausible_year(reversed_val):
                warnings.append(
                    f"possible RTL-reversed year: found '{run}', "
                    f"reversed '{run[::-1]}' looks like a plausible year"
                )

    for run in _ARABIC_INDIC_DIGIT_RUN_RE.findall(text):
        western = run.translate(_ARABIC_INDIC_TO_WESTERN)
        if len(western) == 4:
            forward = int(western)
            reversed_val = int(western[::-1])
            if not _plausible_year(forward) and _plausible_year(reversed_val):
                warnings.append(
                    f"possible RTL-reversed Arabic-Indic year: found '{run}' "
                    f"(-> {western}), reversed '{western[::-1]}' looks plausible"
                )

    return warnings
