"""
Save processed audio segments and metadata to disk.
"""

import os
import json
import uuid

import numpy as np
import soundfile as sf

from config import AUDIO_DIR, JSONL_PATH


def save_segments(wav_path, segments):
    """Split audio into per-segment WAV files.

    Args:
        wav_path: Path to full audio WAV.
        segments: List of {"start", "end", "text"} dicts.

    Returns:
        list[dict]: List of {"audio_path", "text_raw"} entries.
    """
    audio, sr = sf.read(wav_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    outputs = []

    for seg in segments:
        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)

        if start >= len(audio) or end > len(audio):
            continue

        chunk = audio[start:end]

        if len(chunk) < 2000:
            continue

        if np.mean(np.abs(chunk)) < 0.01:
            continue

        filename = f"{uuid.uuid4()}.wav"
        path = os.path.join(AUDIO_DIR, filename)
        sf.write(path, chunk, sr)

        outputs.append({
            "audio_path": path,
            "text_raw": seg["text"].strip(),
        })

    return outputs


def save_jsonl(entries):
    """Append entries to the dataset JSONL file.

    Args:
        entries: List of dicts to write (one JSON object per line).
    """
    if not entries:
        return

    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
