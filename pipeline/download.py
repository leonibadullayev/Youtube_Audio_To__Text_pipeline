"""
Download audio from YouTube and extract playlist URLs.
"""

import os
import uuid
import tempfile
import subprocess

import yt_dlp

from config import COOKIE_FILE, DEMUCS_DOWNLOAD_SAMPLE_RATE


def _base_ydl_opts():
    """Shared yt-dlp options for all calls."""
    return {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "ignoreerrors": True,
        "cookiefile": COOKIE_FILE,
        "extractor_args": {
            "youtube": {
                "player_client": ["web", "android", "ios"],
            }
        },
    }


def download_audio(url):
    """Download audio via yt-dlp and convert to 44.1 kHz mono WAV.

    Returns:
        tuple: (wav_path, info_dict)
    """
    tmp_wav = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
    tmp_dl = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}")

    opts = _base_ydl_opts()
    opts["format"] = "bestaudio/best/worst"
    opts["outtmpl"] = tmp_dl + ".%(ext)s"

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

        if info is None:
            raise Exception("Skipped (restricted or failed)")

        path = ydl.prepare_filename(info)

    # handle cases where yt-dlp changes the extension
    if not os.path.exists(path):
        base = os.path.splitext(path)[0]
        parent = os.path.dirname(path)
        candidates = [
            f for f in os.listdir(parent)
            if f.startswith(os.path.basename(base))
        ]
        if candidates:
            path = os.path.join(parent, candidates[0])
        else:
            raise Exception(f"Downloaded file not found: {path}")

    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar",
         str(DEMUCS_DOWNLOAD_SAMPLE_RATE), tmp_wav],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    os.remove(path)
    return tmp_wav, info


def get_playlist(url, start=1):
    """Extract video URLs from a YouTube playlist.

    Args:
        url: Playlist URL.
        start: 1-based index to start from.

    Returns:
        list[str]: Individual video URLs.
    """
    opts = _base_ydl_opts()
    opts["extract_flat"] = True
    opts["playliststart"] = start

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if info is None:
        return []

    return [
        f"https://www.youtube.com/watch?v={e['id']}"
        for e in info.get("entries", [])
        if e and "id" in e
    ]
