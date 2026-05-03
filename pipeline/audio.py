"""
Audio cleaning pipeline: vocal separation, noise reduction, and silence removal.
"""

import os
import uuid
import tempfile

import numpy as np
import torch
import torchaudio
import soundfile as sf
import noisereduce as nr
from demucs.apply import apply_model

from config import (
    DEVICE,
    TARGET_SAMPLE_RATE,
    NOISE_REDUCE_PROP,
    MIN_SPEECH_MS,
    MIN_SILENCE_MS,
    SPEECH_PAD_MS,
    INTER_CHUNK_GAP_SEC,
)
from pipeline.models import demucs_model, vad_model, get_speech_timestamps


def clean_audio(wav_path):
    """Remove music/background and reduce noise using Demucs + noisereduce.

    Args:
        wav_path: Path to input WAV file.

    Returns:
        str: Path to cleaned 16 kHz mono WAV. Original file is deleted.
    """
    print("[CLEAN] Separating vocals with Demucs...")

    audio_np, sr = sf.read(wav_path)
    if len(audio_np.shape) > 1:
        audio_np = audio_np[:, 0]

    # demucs expects (batch, 2, samples) — duplicate mono to stereo
    mono_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
    wav_tensor = torch.stack([mono_tensor, mono_tensor], dim=1)

    # resample to demucs model sample rate if needed
    if sr != demucs_model.samplerate:
        wav_tensor = torchaudio.functional.resample(
            wav_tensor, sr, demucs_model.samplerate
        )

    # separate stems: drums, bass, other, vocals
    with torch.no_grad():
        stems = apply_model(demucs_model, wav_tensor, device=DEVICE)

    # extract vocals, average stereo to mono
    vocals_idx = demucs_model.sources.index("vocals")
    vocals = stems[0, vocals_idx].mean(dim=0).numpy()

    # resample vocals to 16 kHz for downstream models
    vocals_tensor = torch.tensor(vocals, dtype=torch.float32).unsqueeze(0)
    vocals_16k = torchaudio.functional.resample(
        vocals_tensor, demucs_model.samplerate, TARGET_SAMPLE_RATE
    ).squeeze(0).numpy()

    # noise reduction
    print("[CLEAN] Reducing remaining noise...")
    vocals_clean = nr.reduce_noise(
        y=vocals_16k, sr=TARGET_SAMPLE_RATE, prop_decrease=NOISE_REDUCE_PROP
    )

    clean_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_clean.wav")
    sf.write(clean_path, vocals_clean, TARGET_SAMPLE_RATE)

    os.remove(wav_path)
    print("[CLEAN] Done")
    return clean_path


def strip_silence(wav_path):
    """Remove non-speech segments using Silero VAD.

    Concatenates only speech parts with small padding between them.

    Args:
        wav_path: Path to input WAV file.

    Returns:
        str: Path to trimmed WAV. Original file is deleted.
    """
    print("[VAD] Detecting speech segments...")

    audio_np, sr = sf.read(wav_path)
    if len(audio_np.shape) > 1:
        audio_np = audio_np[:, 0]

    wav_tensor = torch.tensor(audio_np, dtype=torch.float32)

    speech_timestamps = get_speech_timestamps(
        wav_tensor,
        vad_model,
        sampling_rate=sr,
        min_speech_duration_ms=MIN_SPEECH_MS,
        min_silence_duration_ms=MIN_SILENCE_MS,
        speech_pad_ms=SPEECH_PAD_MS,
    )

    if not speech_timestamps:
        print("[VAD] No speech detected!")
        return wav_path

    # extract and concatenate speech chunks
    chunks = []
    for ts in speech_timestamps:
        chunks.append(audio_np[ts["start"]:ts["end"]])

    gap = np.zeros(int(INTER_CHUNK_GAP_SEC * sr), dtype=audio_np.dtype)
    speech_only = []
    for i, chunk in enumerate(chunks):
        speech_only.append(chunk)
        if i < len(chunks) - 1:
            speech_only.append(gap)

    speech_only = np.concatenate(speech_only)

    total_dur = len(audio_np) / sr
    speech_dur = len(speech_only) / sr
    removed = total_dur - speech_dur
    print(f"[VAD] {total_dur:.1f}s -> {speech_dur:.1f}s (removed {removed:.1f}s silence)")

    trimmed_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_trimmed.wav")
    sf.write(trimmed_path, speech_only, sr)

    os.remove(wav_path)
    return trimmed_path
