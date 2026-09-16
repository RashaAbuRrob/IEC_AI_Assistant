# -*- coding: utf-8 -*-
"""
Full lip-synced spokesperson video generation via SadTalker, running in
its own isolated Python 3.10 venv (SadTalker's old dependency stack --
numpy==1.23.4, basicsr==1.4.2, etc. -- is incompatible with this
project's Python 3.13, so it cannot be imported directly; it is invoked
as a subprocess instead).

This is slow -- roughly 2.5 minutes for a ~9 second answer on this
machine's 4GB RTX 2050 -- trading latency for a materially more natural
result (real head motion driven by the audio, not just a looping clip)
than the plain audio+loop fallback in voice_output_utils.
"""

import glob
import os
import subprocess
import tempfile
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SADTALKER_DIR = os.path.join(BASE_DIR, "sadtalker_lib")
SADTALKER_PYTHON = r"C:\sadtalker_venv\Scripts\python.exe"
SOURCE_IMAGE = os.path.join(SADTALKER_DIR, "source_frame.png")

INFERENCE_TIMEOUT_SEC = 600

# The 4GB GPU cannot run two SadTalker jobs at once -- a second job doesn't
# just queue politely, it fights the first for VRAM and both slow to a
# crawl or crash. This serializes generations and fails fast (rather than
# silently piling up orphaned processes) if one is already running.
_gpu_lock = threading.Lock()


def _kill_process_tree(pid: int):
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        capture_output=True,
    )


def _run_sadtalker(audio_wav: str, result_dir: str):
    proc = subprocess.Popen(
        [
            SADTALKER_PYTHON, "inference.py",
            "--driven_audio", audio_wav,
            "--source_image", SOURCE_IMAGE,
            "--result_dir", result_dir,
            "--size", "256",
            "--still",
            # Default preprocessing ("crop") zooms in tight on just the
            # face. "full" keeps the source image's original framing
            # (upper body, not just the face) and pastes the animated
            # face back into it, matching the idle-loop video's framing.
            "--preprocess", "full",
        ],
        cwd=SADTALKER_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        _out, err = proc.communicate(timeout=INFERENCE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        # Popen.kill() on Windows only terminates the direct child; SadTalker
        # can spawn worker subprocesses that survive that and keep holding
        # GPU memory. Kill the whole tree instead.
        _kill_process_tree(proc.pid)
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"SadTalker failed (exit {proc.returncode}): {err.decode(errors='replace')[-2000:]}")


def generate_lipsync_video(audio_bytes: bytes) -> bytes:
    """audio_bytes: any ffmpeg-readable audio (e.g. the mp3 from edge-tts).
    Returns mp4 bytes of the lip-synced spokesperson video (SadTalker's
    own muxed audio+video output). Raises RuntimeError immediately if
    another generation is already in progress -- see _gpu_lock."""
    if not _gpu_lock.acquire(blocking=False):
        raise RuntimeError("A lip-sync video is already being generated -- please wait for it to finish first.")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            audio_in = os.path.join(tmp, "in_audio.mp3")
            audio_wav = os.path.join(tmp, "in_audio.wav")
            with open(audio_in, "wb") as f:
                f.write(audio_bytes)

            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_in, audio_wav],
                check=True, capture_output=True,
            )

            result_dir = os.path.join(tmp, "result")
            _run_sadtalker(audio_wav, result_dir)

            # SadTalker writes a silent "temp_*.mp4" plus the final muxed
            # "<source>##<audio>.mp4" -- we want the latter.
            candidates = [
                m for m in glob.glob(os.path.join(result_dir, "**", "*.mp4"), recursive=True)
                if not os.path.basename(m).startswith("temp_")
            ]
            if not candidates:
                raise RuntimeError("SadTalker did not produce an output video")
            with open(candidates[0], "rb") as f:
                return f.read()
    finally:
        _gpu_lock.release()
