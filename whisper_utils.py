# -*- coding: utf-8 -*-
"""
Local speech-to-text via faster-whisper (CTranslate2 runtime). Runs fully
on-device -- the model weights are fetched from Hugging Face once and
cached locally (~/.cache/huggingface); no audio ever leaves the machine.
"""

from faster_whisper import WhisperModel

# "small" balances Arabic transcription accuracy against CPU latency on a
# machine with no GPU (this project's whole stack runs CPU-only via
# Ollama). Bump to "medium"/"large-v3" for better accuracy if latency
# allows; drop to "base" if "small" is too slow.
WHISPER_MODEL_SIZE = "small"

_model = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def prewarm():
    """Load the model at startup so the first real request isn't stuck
    paying the load cost (and, on a fresh machine, the one-time download)."""
    get_model()


def transcribe_arabic(audio_file) -> str:
    """
    audio_file: a path, or a binary file-like object (e.g. a browser
    MediaRecorder blob in webm/ogg/wav -- faster-whisper decodes any
    common container via PyAV under the hood).
    """
    model = get_model()
    segments, _info = model.transcribe(audio_file, language="ar")
    return "".join(segment.text for segment in segments).strip()
