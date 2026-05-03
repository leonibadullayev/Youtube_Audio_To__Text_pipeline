"""
WhisperX transcription and forced alignment.
"""

import os
import uuid

import numpy as np
import soundfile as sf
import whisperx

from config import DEVICE, BATCH_SIZE, LANG, AUDIO_DIR
from pipeline.models import whisper_model, align_model, align_metadata


def align_segments(wav_path, segments):
    """Use WhisperX forced alignment for precise word-level timestamps.

    Args:
        wav_path: Path to audio WAV file.
        segments: List of {"start", "end", "text"} dicts.

    Returns:
        list[dict]: Aligned segments with updated timestamps.
    """
    audio = whisperx.load_audio(wav_path)

    result = whisperx.align(
        segments,
        align_model,
        align_metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )

    return result["segments"]


def transcribe_asr(wav_path):
    """Full ASR pipeline: transcribe + optional alignment.

    Used as fallback when no subtitles are available.

    Args:
        wav_path: Path to cleaned audio WAV file.

    Returns:
        list[dict]: List of {"audio_path", "text_raw"} entries.
    """
    audio = whisperx.load_audio(wav_path)

    result = whisper_model.transcribe(audio, batch_size=BATCH_SIZE, language=LANG)

    if align_model is not None:
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            DEVICE,
            return_char_alignments=False,
        )

    audio_np, sr = sf.read(wav_path)
    if len(audio_np.shape) > 1:
        audio_np = audio_np[:, 0]

    outputs = []
    for seg in result["segments"]:
        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)

        if start >= len(audio_np) or end > len(audio_np):
            continue

        chunk = audio_np[start:end]

        if len(chunk) < 2000:
            continue

        filename = f"{uuid.uuid4()}.wav"
        path = os.path.join(AUDIO_DIR, filename)
        sf.write(path, chunk, sr)

        outputs.append({
            "audio_path": path,
            "text_raw": seg["text"].strip(),
        })

    return outputs
