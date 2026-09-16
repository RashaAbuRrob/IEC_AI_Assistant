# -*- coding: utf-8 -*-
"""
Local API server for the IEC RAG chat UI (static/index.html).

Matches the contract the frontend already expects:
  POST /api/assistant/query
  body: {"query": "...", "history": [...], "stream": true}
  response: newline-delimited JSON (NDJSON), one object per line:
    {"type": "sources", "context_sources": [{"filename": "...", "page": N, "article_number": "..."}]}
    {"type": "text", "text": "<fragment>"}   -- zero or more, streamed as generated

Usage:
    python api_server.py --db-path ./chroma_db --port 5001

Then open http://localhost:5001/ in a browser (served from static/index.html),
or open static/index.html directly as a file:// page -- CORS is enabled so
either works.
"""

import argparse
import json

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from arabic_utils import normalize_arabic
from chroma_utils import get_collection
from lipsync_utils import generate_lipsync_video
from ollama_utils import prewarm_embedder, stream_generate_answer
from rag_service import REFUSAL, SYSTEM_PROMPT, build_where_filter, format_context
from voice_clone_utils import synthesize_cloned_speech
from voice_output_utils import synthesize_speech
from whisper_utils import prewarm as prewarm_whisper, transcribe_arabic

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Populated in main() before app.run(), or via init_app() if imported.
_state = {"collection": None, "k": 5}


def init_app(db_path: str, k: int = 5):
    print("Pre-warming bge-m3 embedder via Ollama...")
    prewarm_embedder()
    print("Pre-warming faster-whisper (speech-to-text)...")
    prewarm_whisper()
    _state["collection"] = get_collection(db_path)
    _state["k"] = k
    print(f"Loaded collection with {_state['collection'].count()} chunks.")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    collection = _state["collection"]
    return jsonify(
        {
            "status": "ok" if collection is not None else "not_initialized",
            "chunks": collection.count() if collection is not None else 0,
        }
    )


@app.route("/api/tts/status")
def tts_status():
    """Lets the frontend know the human Jordanian voice endpoint exists
    (no config/key needed -- it's edge-tts, always available as long as
    the server has internet access to Microsoft's TTS service)."""
    return jsonify({"voice_available": True})


@app.route("/api/assistant/speak", methods=["POST"])
def speak():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    try:
        audio_bytes = synthesize_speech(text)
    except Exception as e:  # noqa: BLE001 -- surface any failure to the UI instead of hanging it
        # Network/service hiccup -- the frontend falls back to the
        # browser's built-in voice on any non-2xx response here.
        return jsonify({"error": str(e)}), 502
    return Response(audio_bytes, mimetype="audio/mpeg")


@app.route("/api/assistant/lipsync", methods=["POST"])
def lipsync():
    """Generates a real lip-synced spokesperson video (SadTalker, ~2-3
    minutes per answer on this machine's 4GB GPU) instead of just audio.
    The frontend falls back to /api/assistant/speak (audio + looping
    clip) if this fails or times out."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    try:
        audio_bytes = synthesize_speech(text)
        video_bytes = generate_lipsync_video(audio_bytes)
    except Exception as e:  # noqa: BLE001 -- surface any failure to the UI instead of hanging it
        return jsonify({"error": str(e)}), 502
    return Response(video_bytes, mimetype="video/mp4")


@app.route("/api/assistant/speak_cloned", methods=["POST"])
def speak_cloned():
    """Generates audio in the real, authorized human voice reference
    (chatterbox-tts, CPU) instead of the stock edge-tts voice. Slow --
    roughly 25-30s model load (once) plus ~10s per second of output
    audio -- so this is opt-in per-message in the UI, not the automatic
    default voice."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    try:
        audio_bytes = synthesize_cloned_speech(text)
    except Exception as e:  # noqa: BLE001 -- surface any failure to the UI instead of hanging it
        return jsonify({"error": str(e)}), 502
    return Response(audio_bytes, mimetype="audio/wav")


@app.route("/api/assistant/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "audio file is required"}), 400
    try:
        text = transcribe_arabic(request.files["audio"].stream)
    except Exception as e:  # noqa: BLE001 -- surface any failure to the UI instead of hanging it
        return jsonify({"error": str(e)}), 500
    return jsonify({"text": text})


@app.route("/api/assistant/query", methods=["POST"])
def assistant_query():
    collection = _state["collection"]
    if collection is None:
        return jsonify({"error": "Server not initialized yet."}), 503

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    k = _state["k"]

    def event_stream():
        def emit(obj):
            return json.dumps(obj, ensure_ascii=False) + "\n"

        try:
            normalized_q = normalize_arabic(query)
            query_kwargs = {"query_texts": [normalized_q], "n_results": k, "include": ["documents", "metadatas", "distances"]}
            where = build_where_filter(query)
            if where:
                query_kwargs["where"] = where
            results = collection.query(**query_kwargs)
            docs = results["documents"][0]

            if not docs:
                yield emit({"type": "sources", "context_sources": []})
                yield emit({"type": "text", "text": REFUSAL})
                return

            metas = results["metadatas"][0]
            context_sources = [
                {
                    "filename": m.get("source_file"),
                    "page": m.get("page"),
                    "article_number": m.get("article_number") or None,
                }
                for m in metas
            ]
            yield emit({"type": "sources", "context_sources": context_sources})

            context = format_context(results)
            user_prompt = f"المقاطع المسترجعة:\n\n{context}\n\nالسؤال: {query}\n\nالإجابة:"

            for chunk in stream_generate_answer(SYSTEM_PROMPT, user_prompt):
                yield emit({"type": "text", "text": chunk})

        except Exception as e:  # noqa: BLE001 -- surface any failure to the UI instead of hanging it
            yield emit({"type": "text", "text": f"\n\n[خطأ في الخادم: {e}]"})

    return Response(event_stream(), mimetype="application/x-ndjson")


def main():
    parser = argparse.ArgumentParser(description="Run the IEC RAG API server.")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    init_app(args.db_path, k=args.k)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
