# -*- coding: utf-8 -*-
"""
Human-sounding Jordanian voice output via edge-tts, which calls the same
neural voices Microsoft Edge's built-in "Read Aloud" feature uses
(ar-JO-TaimNeural / ar-JO-SanaNeural) for free -- no Azure account or API
key required.

This is the one part of this project that is NOT offline: it needs an
internet connection to reach Microsoft's TTS service. Everything else
(Ollama generation/embeddings, faster-whisper speech-to-text, ChromaDB)
stays fully local regardless. If this call fails (no internet, service
hiccup), the frontend falls back to the browser's built-in SpeechSynthesis.
"""

import asyncio
import io

import edge_tts

DEFAULT_VOICE = "ar-JO-TaimNeural"


async def _synthesize_async(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def synthesize_speech(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Returns MP3 audio bytes for `text` spoken by a Jordanian neural voice."""
    return asyncio.run(_synthesize_async(text, voice))
