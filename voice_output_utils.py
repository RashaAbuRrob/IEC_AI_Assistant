# -*- coding: utf-8 -*-
"""
Voice output, preferring Gemini's native TTS (natural, ChatGPT-voice-mode-
style prosody) with an automatic fallback to edge-tts's free Jordanian
neural voice (ar-JO-TaimNeural) when Gemini is unconfigured or fails --
same "best model, then local/free fallback" pattern gemini_utils.py uses
for text generation.

edge-tts is the one part of this project that is NOT offline: it needs an
internet connection to reach Microsoft's TTS service. Gemini TTS also
needs internet + GEMINI_API_KEY. If both fail, the frontend falls back to
the browser's built-in SpeechSynthesis.
"""

import asyncio
import io

import edge_tts

import gemini_utils

DEFAULT_VOICE = "ar-JO-TaimNeural"


async def _synthesize_async(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def synthesize_speech(text: str, voice: str = DEFAULT_VOICE) -> tuple[bytes, str]:
    """Returns (audio_bytes, mimetype) for `text` spoken aloud. Tries
    Gemini's native TTS first when GEMINI_API_KEY is configured, falling
    back to the edge-tts Jordanian neural voice on any failure (missing
    key, network, quota, ...)."""
    if gemini_utils.is_configured():
        try:
            return gemini_utils.synthesize_speech(text), "audio/wav"
        except Exception:  # noqa: BLE001 -- fall back to edge-tts below
            pass
    return asyncio.run(_synthesize_async(text, voice)), "audio/mpeg"
