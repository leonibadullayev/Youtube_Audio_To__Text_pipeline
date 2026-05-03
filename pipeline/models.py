"""
Load and cache all ML models used in the pipeline.

Models are loaded once on first access and reused across the entire run.
"""

import torch
import whisperx
from demucs.pretrained import get_model

from config import DEVICE, WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE, DEMUCS_MODEL_NAME, LANG

# ── WhisperX transcription ──────────────────────────────────────────

print("[INFO] Loading WhisperX transcription model...")
whisper_model = whisperx.load_model(
    WHISPER_MODEL_SIZE, DEVICE, compute_type=WHISPER_COMPUTE_TYPE
)

# ── Demucs vocal separation ─────────────────────────────────────────

print("[INFO] Loading Demucs vocal separation model...")
demucs_model = get_model(DEMUCS_MODEL_NAME)
demucs_model.to(DEVICE)

# ── Silero VAD ──────────────────────────────────────────────────────

print("[INFO] Loading Silero VAD model...")
vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
)
get_speech_timestamps, _, read_audio, _, _ = vad_utils

# ── WhisperX alignment ──────────────────────────────────────────────

print("[INFO] Loading WhisperX alignment model...")
try:
    align_model, align_metadata = whisperx.load_align_model(
        language_code=LANG, device=DEVICE
    )
    print(f"[INFO] Alignment model loaded for '{LANG}'")
except Exception as e:
    print(f"[WARN] No alignment model for '{LANG}': {e}")
    print("[WARN] Falling back to transcription-only timestamps")
    align_model, align_metadata = None, None
