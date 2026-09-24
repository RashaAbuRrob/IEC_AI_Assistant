# -*- coding: utf-8 -*-
"""
Thin wrapper around the local Ollama HTTP API. No cloud calls anywhere in
this module -- everything talks to http://localhost:11434 (or OLLAMA_HOST).
"""

import base64
import json
import os
import time

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

EMBED_MODEL = "bge-m3"
GEN_MODEL = "qwen3:14b"
GEN_MODEL_QWEN25 = "qwen2.5:1.5b-instruct-q8_0"
VISION_MODEL = "qwen2.5vl"

# Keep bge-m3 resident in memory since it is hit on every add() and every
# query(). -1 tells Ollama to keep the model loaded indefinitely (must be an
# int or a unit-suffixed duration string like "-1s" -- a bare "-1" string is
# rejected by the server with "time: missing unit in duration").
EMBED_KEEP_ALIVE = -1


def prewarm_embedder():
    """Send a throwaway embedding request at startup so the first real
    add()/query() isn't stuck paying the model-load cold-start cost."""
    try:
        embed_text("تهيئة النظام")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_HOST} to pre-warm bge-m3. "
            f"Is `ollama serve` running and has `ollama pull bge-m3` been run? "
            f"Original error: {e}"
        )


def embed_text(text: str, retries: int = 3, backoff: float = 1.5):
    """Embed a single string via Ollama's bge-m3, with basic retry."""
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{OLLAMA_HOST}/api/embed",
                json={
                    "model": EMBED_MODEL,
                    "input": text,
                    "keep_alive": EMBED_KEEP_ALIVE,
                },
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            return data["embeddings"][0]
        except (requests.exceptions.RequestException, KeyError, IndexError) as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"bge-m3 embedding failed after {retries} attempts: {last_err}")


def generate_answer(system_prompt: str, user_prompt: str, temperature: float = 0.0, model: str = GEN_MODEL):
    """
    Call a local Ollama model (qwen3:14b by default) for the final answer
    generation. temperature=0.0 for factual, low-variance answers (this is
    a citation-bound QA task, not creative generation).
    """
    r = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {"temperature": temperature},
            # qwen3 supports an explicit thinking toggle; we don't need
            # chain-of-thought in the output for a grounded-QA task.
            "think": False,
        },
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


def stream_generate_answer(system_prompt: str, user_prompt: str, temperature: float = 0.0, model: str = GEN_MODEL):
    """
    Same call as generate_answer(), but streams response fragments as they
    arrive from Ollama instead of waiting for the full completion. Yields
    plain text chunks (already extracted from Ollama's per-line JSON
    envelope) suitable for forwarding straight to an HTTP client.
    """
    with requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": True,
            "options": {"temperature": temperature},
            "think": False,
        },
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            chunk = data.get("response")
            if chunk:
                yield chunk
            if data.get("done"):
                break


def vision_transcribe(image_bytes: bytes, prompt: str = None):
    """
    Send a page image to qwen2.5vl for transcription. Used only as a last
    resort, when a page still has fewer than 40 characters after OCR AND
    contains at least one embedded image (per the ingestion spec).
    """
    if prompt is None:
        prompt = (
            "انسخ كل النص الموجود في هذه الصورة بدقة، بنفس اللغة العربية، "
            "دون إضافة أي شرح أو تعليق. أعد النص فقط."
        )
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    r = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": VISION_MODEL,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()
