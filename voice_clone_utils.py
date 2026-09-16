# -*- coding: utf-8 -*-
"""
Voice cloning via chatterbox-tts (Resemble AI, MIT licensed), using a real
human voice reference clip (voice_reference.wav, ~7.5s) so the
spokesperson can speak in that person's actual voice instead of a stock
TTS voice. Use is authorized by the voice's owner for this project.

Runs on CPU on this machine (no CUDA-enabled torch installed here) --
roughly 25-30s to load the model once, then ~10s of generation per
second of output audio. That means a typical multi-sentence answer can
take a few minutes, so this is wired up as an opt-in per-message action
in the UI, not the automatic default voice (see voice_output_utils for
the fast edge-tts path used automatically).
"""

import io
import os
import threading

import torchaudio as ta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_REFERENCE_PATH = os.path.join(BASE_DIR, "voice_reference.wav")

_model = None
_model_lock = threading.Lock()
_generate_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                _model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    return _model


def prewarm():
    """Load the model at startup so the first real request isn't stuck
    paying the ~25-30s load cost on top of generation time."""
    _get_model()


def synthesize_cloned_speech(text: str) -> bytes:
    """Returns WAV bytes of `text` spoken in the cloned reference voice.
    Only one generation runs at a time (CPU-bound, can take minutes) --
    raises RuntimeError immediately if another is already in progress
    rather than letting requests pile up and slow each other down."""
    if not _generate_lock.acquire(blocking=False):
        raise RuntimeError("A cloned-voice reply is already being generated -- please wait for it to finish first.")
    try:
        model = _get_model()
        wav = model.generate(text, language_id="ar", audio_prompt_path=VOICE_REFERENCE_PATH)
        buf = io.BytesIO()
        ta.save(buf, wav, model.sr, format="wav")
        return buf.getvalue()
    finally:
        _generate_lock.release()
