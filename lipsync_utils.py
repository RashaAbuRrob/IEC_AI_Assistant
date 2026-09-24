# -*- coding: utf-8 -*-
"""
Full lip-synced spokesperson video generation: SadTalker for natural head
motion/framing, running in its own isolated Python 3.10 venv (SadTalker's
old dependency stack -- numpy==1.23.4, basicsr==1.4.2, etc. -- is
incompatible with this project's Python 3.13, so it cannot be imported
directly; it is invoked as a subprocess instead) -- followed by a Wav2Lip
refinement pass over just the mouth region.

Why two stages: SadTalker's audio2expression model is tuned for natural,
varied head motion, not frame-accurate lip sync, and that shows up as the
mouth visibly not tracking the words. Wav2Lip is trained specifically
against a lip-sync discriminator (SyncNet) and is much more precise at
mapping audio to mouth shape, but on its own produces a flat, motionless
video. Wav2Lip also happily re-syncs an existing *video* (not just a still
image) to new audio, so running it over SadTalker's output keeps
SadTalker's head motion and framing while correcting the mouth region to
actually match the audio. Wav2Lip's deps (onnxruntime, insightface,
opencv) are already importable in this project's own Python -- unlike
SadTalker, no separate venv is needed for this stage.

This is slow -- roughly 2.5 minutes for a ~9 second answer on this
machine's 4GB RTX 2050 for the SadTalker pass, plus well under a minute
for the Wav2Lip refinement pass -- trading latency for a materially more
natural, better-synced result than the plain audio+loop fallback in
voice_output_utils.
"""

import glob
import os
import subprocess
import sys
import tempfile
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SADTALKER_DIR = os.path.join(BASE_DIR, "sadtalker_lib")
SADTALKER_PYTHON = r"C:\sadtalker_venv\Scripts\python.exe"
SOURCE_IMAGE = os.path.join(SADTALKER_DIR, "source_frame.png")
SADTALKER_TIMEOUT_SEC = 600

WAV2LIP_DIR = os.path.join(BASE_DIR, "wav2lip_onnx_lib")
WAV2LIP_CHECKPOINT = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.onnx")
WAV2LIP_TIMEOUT_SEC = 180

# The 4GB GPU cannot run two SadTalker jobs at once -- a second job doesn't
# just queue politely, it fights the first for VRAM and both slow to a
# crawl or crash. This serializes generations (SadTalker + the Wav2Lip
# refinement pass that follows it) and fails fast (rather than silently
# piling up orphaned processes) if one is already running.
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
        _out, err = proc.communicate(timeout=SADTALKER_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        # Popen.kill() on Windows only terminates the direct child; SadTalker
        # can spawn worker subprocesses that survive that and keep holding
        # GPU memory. Kill the whole tree instead.
        _kill_process_tree(proc.pid)
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"SadTalker failed (exit {proc.returncode}): {err.decode(errors='replace')[-2000:]}")


def _run_wav2lip_refine(face_video: str, audio_wav: str, out_path: str):
    """Re-syncs the mouth region of `face_video` to `audio_wav` using
    Wav2Lip (GAN checkpoint -- less blurry than the base checkpoint), and
    muxes the result with that same audio into `out_path`."""
    proc = subprocess.Popen(
        [
            sys.executable, "inference_onnxModel.py",
            "--checkpoint_path", WAV2LIP_CHECKPOINT,
            "--face", face_video,
            "--audio", audio_wav,
            "--outfile", out_path,
        ],
        cwd=WAV2LIP_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        _out, err = proc.communicate(timeout=WAV2LIP_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc.pid)
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"Wav2Lip refinement failed (exit {proc.returncode}): {err.decode(errors='replace')[-2000:]}")


def generate_lipsync_video(audio_bytes: bytes) -> bytes:
    """audio_bytes: any ffmpeg-readable audio (e.g. the mp3 from edge-tts).
    Returns mp4 bytes of the lip-synced spokesperson video: SadTalker for
    head motion/framing, then Wav2Lip to correct the mouth region against
    the same audio. Raises RuntimeError immediately if another generation
    is already in progress -- see _gpu_lock."""
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

            # SadTalker may write a silent "temp_*.mp4" alongside the final
            # muxed output (named by timestamp) -- exclude those, then pick
            # the most recently written match rather than an arbitrary
            # glob() order, in case more than one real candidate remains.
            candidates = [
                m for m in glob.glob(os.path.join(result_dir, "**", "*.mp4"), recursive=True)
                if not os.path.basename(m).startswith("temp_")
            ]
            if not candidates:
                raise RuntimeError("SadTalker did not produce an output video")
            sadtalker_video = max(candidates, key=os.path.getmtime)

            refined_video = os.path.join(tmp, "refined.mp4")
            _run_wav2lip_refine(sadtalker_video, audio_wav, refined_video)

            with open(refined_video, "rb") as f:
                return f.read()
    finally:
        _gpu_lock.release()
