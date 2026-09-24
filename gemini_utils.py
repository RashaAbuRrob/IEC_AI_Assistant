# -*- coding: utf-8 -*-
"""
Answer generation via Google's Gemini Flash API. This is the primary
generation path; ollama_utils.stream_generate_answer (the local qwen3:14b)
is the fallback used automatically when Gemini fails -- no API key, no
network, quota/rate limit, etc. -- and is also selectable explicitly from
the UI.

Requires GEMINI_API_KEY (loaded from .env by api_server.py at startup).
"""

import os

from dotenv import load_dotenv

load_dotenv()  # populates os.environ from a local .env file, if present

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
GEMINI_TTS_VOICE = os.environ.get("GEMINI_TTS_VOICE", "Achird")  # male, "Friendly" per Google's voice list

_client = None


def is_configured() -> bool:
    return bool(GEMINI_API_KEY)


def _get_client():
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def stream_generate_answer(system_prompt: str, user_prompt: str, temperature: float = 0.0):
    """
    Yields text chunks from Gemini Flash as they arrive. Raises on any
    failure (missing key, network, quota, ...) so the caller can fall back
    to the local Ollama model instead of silently returning nothing.
    """
    from google.genai import types

    client = _get_client()
    response = client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        ),
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


def synthesize_speech(text: str, voice: str = None) -> bytes:
    """Returns WAV bytes of `text` spoken by a Gemini native TTS voice --
    the same model family behind ChatGPT-style voice mode, with much more
    natural prosody than edge-tts. Raises on any failure (missing key,
    network, quota, ...) so the caller can fall back to edge-tts instead
    of silently returning nothing."""
    import io
    import wave

    from google.genai import types

    client = _get_client()
    response = client.models.generate_content(
        model=GEMINI_TTS_MODEL,
        # The "Say ...:" prefix is required -- without it the model
        # sometimes tries to respond to/continue the text instead of just
        # reading it aloud, and the API rejects the request. The accent
        # instruction steers delivery toward Jordanian colloquial (Ammani)
        # rather than a neutral MSA newsreader tone.
        contents=f"Say in a warm, casual Jordanian Arabic (Ammani) dialect accent, like a friendly local speaker: {text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice or GEMINI_TTS_VOICE)
                )
            ),
        ),
    )
    pcm = response.candidates[0].content.parts[0].inline_data.data

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm)
    return buf.getvalue()
