"""
Main entry point for the Azerbaijani speech dataset pipeline.

Usage:
    python main.py
"""

import os
import re
import time
import uuid

import whisperx

from config import (
    AUDIO_DIR,
    BATCH_SIZE,
    LANG,
    DELAY_BETWEEN_VIDEOS,
    MAX_RETRIES,
    RATE_LIMIT_BASE_WAIT,
)
from pipeline.models import whisper_model
from pipeline.download import download_audio, get_playlist
from pipeline.subtitles import get_subtitle_text, get_subtitle_downsub
from pipeline.audio import clean_audio, strip_silence
from pipeline.storage import save_jsonl


def process_video(url):
    """Process a single video: download, clean, extract text, save.

    Tries subtitles first, then DownSub, then ASR as final fallback.
    """
    for attempt in range(MAX_RETRIES):
        try:
            print(f"\n[VIDEO] {url}")

            # 1. download audio + get info
            wav, info = download_audio(url)

            # 2. fetch subtitle text immediately (URLs expire)
            subs = get_subtitle_text(info)

            # 3. remove music and noise
            wav = clean_audio(wav)

            # 4. strip silence and non-speech
            wav = strip_silence(wav)

            # 5. extract text: subtitles → DownSub → ASR
            if subs:
                print("[MODE] SUBTITLES")
                full_text = " ".join(seg["text"].strip() for seg in subs)

            else:
                downsub_text = get_subtitle_downsub(url)

                if downsub_text:
                    print("[MODE] DOWNSUB SUBTITLES")
                    full_text = downsub_text
                else:
                    print("[MODE] ASR FALLBACK")
                    audio = whisperx.load_audio(wav)
                    result = whisper_model.transcribe(
                        audio, batch_size=BATCH_SIZE, language=LANG
                    )
                    full_text = " ".join(
                        seg["text"].strip() for seg in result["segments"]
                    )

            # clean whitespace
            full_text = re.sub(r"\s+", " ", full_text).strip()

            if not full_text:
                print("[SKIP] No text extracted")
                os.remove(wav)
                return

            # 6. move cleaned audio to dataset
            filename = f"{uuid.uuid4()}.wav"
            final_path = os.path.join(AUDIO_DIR, filename)
            os.rename(wav, final_path)

            # 7. save entry
            save_jsonl([{
                "audio_path": final_path,
                "text_raw": full_text,
            }])

            print(f"[DONE] 1 sample, {len(full_text)} chars")
            return

        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                wait = RATE_LIMIT_BASE_WAIT * (attempt + 1)
                print(f"[RATE LIMITED] Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"[SKIP] {url} -> {e}")
                return


def run(urls):
    """Process a list of video URLs sequentially."""
    print(f"[INFO] {len(urls)} videos")

    for i, url in enumerate(urls):
        process_video(url)
        if i < len(urls) - 1:
            time.sleep(DELAY_BETWEEN_VIDEOS)


if __name__ == "__main__":
    playlist_urls = [
        "https://www.youtube.com/playlist?list=",
    ]

    for p in playlist_urls:
        videos = get_playlist(p, start=290)
        run(videos)
