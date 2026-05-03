# Youtube Speech Dataset Pipeline

An automated pipeline for building Azerbaijani (az) speech-to-text datasets from YouTube videos. The pipeline downloads audio from playlists or individual videos, isolates vocals from background music, removes silence, extracts transcription text from subtitles or ASR, and outputs paired audio–text samples in a format ready for fine-tuning speech recognition models.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Pipeline Architecture](#pipeline-architecture)
- [Output Format](#output-format)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Module Reference](#module-reference)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## How It Works

For each video the pipeline runs through six stages:

1. **Download** — `yt-dlp` pulls the best available audio stream and `ffmpeg` converts it to a 44.1 kHz mono WAV.
2. **Subtitle extraction** — The pipeline tries to grab subtitles from the yt-dlp info dict in `json3` format (manual subs first, then auto-captions, in Azerbaijani → Azerbaijani-orig → English priority). If that fails, it falls back to scraping [downsub.com](https://downsub.com) via headless Chrome.
3. **Vocal separation** — [Demucs](https://github.com/facebookresearch/demucs) (`htdemucs` model) separates the audio into stems (drums, bass, other, vocals) and keeps only the vocal track.
4. **Noise reduction** — `noisereduce` applies spectral gating to clean residual noise from the isolated vocals.
5. **Silence stripping** — [Silero VAD](https://github.com/snakers4/silero-vad) detects speech regions and the pipeline concatenates only those regions, removing all silence and non-speech segments.
6. **Save** — The cleaned audio file and its paired transcription text are written to disk as a JSONL entry.

If no subtitles are available from either source, WhisperX (`large-v3`) performs full ASR transcription as a final fallback.

### Text Source Priority

```
yt-dlp subtitles (json3)
        │
        ▼ not found
DownSub.com (Selenium scrape)
        │
        ▼ not found
WhisperX ASR (large-v3)
```

---

## Pipeline Architecture

```
YouTube URL
     │
     ▼
┌──────────────┐
│  yt-dlp      │──→ audio (WAV 44.1kHz) + subtitle metadata
└──────────────┘
     │
     ▼
┌──────────────┐
│  Demucs      │──→ isolated vocal track
└──────────────┘
     │
     ▼
┌──────────────┐
│  noisereduce │──→ cleaned vocals (16kHz)
└──────────────┘
     │
     ▼
┌──────────────┐
│  Silero VAD  │──→ speech-only audio (silence removed)
└──────────────┘
     │
     ▼
┌──────────────┐
│  Save        │──→ dataset/audio/*.wav + dataset/data.jsonl
└──────────────┘
```

---

## Output Format

The pipeline produces a flat dataset directory:

```
dataset/
├── audio/
│   ├── 3a1f8c2e-...wav
│   ├── 7b4d9e01-...wav
│   └── ...
└── data.jsonl
```

Each line in `data.jsonl` is a JSON object:

```json
{
  "audio_path": "dataset/audio/3a1f8c2e-...-clean.wav",
  "text_raw": "Salam, bu gün biz danışacağıq..."
}
```

Audio files are 16 kHz mono WAV with vocals only (no music, no background noise, no silence gaps).

---

## Requirements

### System Dependencies

| Dependency | Purpose | Install |
|-------|---|---|
| **Python 3.11** | Runtime | [python.org](https://www.python.org) | | Audio format conversion | `sudo apt install ffmpeg` / `brew install ffmpeg` |
| **Chrome/Chromium** | DownSub subtitle fallback | `sudo apt install chromium-browser` |
| **ChromeDriver** | Selenium browser automation | Must match your Chrome version |

### Hardware

The pipeline is configured for CPU by default. All models (WhisperX large-v3, Demucs htdemucs, Silero VAD) will run on CPU but processing is significantly slower than on GPU.

For GPU acceleration, change `DEVICE = "cuda"` in `config.py` and ensure you have a CUDA-capable GPU with sufficient VRAM (8 GB+ recommended for large-v3). You may also want to change `WHISPER_COMPUTE_TYPE` from `"int8"` to `"float16"` on GPU for better speed.

### Disk Space

Budget approximately 5–15 MB per video for the cleaned audio. A 500-video playlist will produce roughly 2.5–7.5 GB of WAV files.

---

## FFMPEG Download & Installation & Configuration
The project includes the use of ffmpeg which enables performing operations on audio files.

Open the link below:

https://www.gyan.dev/ffmpeg/builds/

In the **release builds** section click on

ffmpeg-release-full-shared.7z

for the download to start.

Upon download, extract them to a particular folder, **Desktop**,
Then add their **\bin** directory path to Path in user and system variables in your computer's **environmental variables**.

## Installation

```bash
# Clone the repository
git clone https://github.com/leonibadullayev/youtube_audio_to_text_pipeline.git
cd youtube-audio-to-text-pipeline

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install Python dependencies
pip install -r requirements.txt
```

### Verifying system dependencies

```bash
# Check ffmpeg
ffmpeg -version

# Check Chrome
google-chrome --version         # or chromium-browser --version

# Check ChromeDriver
chromedriver --version
```

### YouTube cookies (optional but recommended)

If you encounter HTTP 429 rate limits, age-restricted video errors, or sign-in walls, export your YouTube cookies to a `cookies.txt` file in Netscape format. Browser extensions like "Get cookies.txt LOCALLY" can do this.

Place the file in the project root:

```
az-speech-dataset/
├── cookies.txt       ← here
├── config.py
└── ...
```

---

## Configuration

All tuneable parameters live in `config.py`. Here is a full reference:

### Paths

| Parameter | Default | Description |
|---|---|---|
| `OUTPUT_DIR` | `"dataset"` | Root directory for all output |
| `AUDIO_DIR` | `"dataset/audio"` | Where cleaned WAV files are saved |
| `JSONL_PATH` | `"dataset/data.jsonl"` | Path to the metadata file |

### Model / Inference

| Parameter | Default | Description |
|---|---|---|
| `DEVICE` | `"cpu"` | Torch device — `"cpu"` or `"cuda"` |
| `BATCH_SIZE` | `4` | WhisperX transcription batch size (increase on GPU) |
| `LANG` | `"az"` | Target language code (ISO 639-1) |
| `WHISPER_MODEL_SIZE` | `"large-v3"` | Whisper model variant (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `WHISPER_COMPUTE_TYPE` | `"int8"` | Quantization — `"int8"` for CPU, `"float16"` for GPU |
| `DEMUCS_MODEL_NAME` | `"htdemucs"` | Demucs model variant |

### Audio Processing

| Parameter | Default | Description |
|---|---|---|
| `TARGET_SAMPLE_RATE` | `16000` | Output sample rate (Hz) for cleaned audio |
| `DEMUCS_DOWNLOAD_SAMPLE_RATE` | `44100` | Sample rate for initial download (Demucs input) |
| `NOISE_REDUCE_PROP` | `0.7` | Noise reduction strength (0.0–1.0) |
| `MIN_SPEECH_MS` | `250` | Minimum speech segment duration for VAD (ms) |
| `MIN_SILENCE_MS` | `300` | Minimum silence gap to split on (ms) |
| `SPEECH_PAD_MS` | `100` | Padding around detected speech (ms) |
| `INTER_CHUNK_GAP_SEC` | `0.15` | Silence gap inserted between concatenated speech chunks (sec) |

### Downloading

| Parameter | Default | Description |
|---|---|---|
| `COOKIE_FILE` | `"cookies.txt"` | Path to Netscape-format cookies file |
| `DELAY_BETWEEN_VIDEOS` | `5` | Seconds to wait between videos to avoid rate limiting |
| `MAX_RETRIES` | `3` | Number of retry attempts on failure |
| `RATE_LIMIT_BASE_WAIT` | `60` | Base wait time (sec) on HTTP 429; multiplied by attempt number |

---

## Usage

### Processing a playlist

Edit the playlist URLs and start index in `main.py`:

```python
if __name__ == "__main__":
    playlist_urls = [
        "https://www.youtube.com/playlist?list=PLxxxxxxxxxxxxxxxx",
    ]

    for p in playlist_urls:
        videos = get_playlist(p, start=1)   # start=1 begins from the first video
        run(videos)
```

Then run:

```bash
python main.py
```

### Processing individual videos

```python
from main import process_video

process_video("https://www.youtube.com/watch?v=XXXXXXXXXXX")
```

### Using modules independently

Each pipeline stage can be used on its own:

```python
from pipeline.download import download_audio, get_playlist
from pipeline.subtitles import get_subtitle_text, get_subtitle_downsub
from pipeline.audio import clean_audio, strip_silence
from pipeline.transcribe import transcribe_asr, align_segments
from pipeline.storage import save_jsonl, save_segments

# Download just the audio
wav_path, info = download_audio("https://www.youtube.com/watch?v=...")

# Clean an existing WAV file
cleaned = clean_audio("/path/to/audio.wav")

# Run ASR on a pre-cleaned file
entries = transcribe_asr("/path/to/cleaned.wav")

# Extract a playlist without downloading
urls = get_playlist("https://www.youtube.com/playlist?list=...", start=50)
```

### Resuming interrupted runs

The pipeline appends to `data.jsonl` rather than overwriting it, so you can safely restart. To skip already-processed videos, use the `start` parameter in `get_playlist()` to jump ahead in the playlist.

---

## Project Structure

```
az-speech-dataset/
├── config.py                  # All constants, paths, and hyperparameters
├── main.py                    # Entry point — orchestrates the full pipeline
├── requirements.txt           # Python dependencies
├── .gitignore                 # Ignores dataset/, cookies, caches
├── README.md                  # This file
│
└── pipeline/                  # Core processing modules
    ├── __init__.py
    ├── models.py              # Loads all ML models at import time
    ├── download.py            # yt-dlp audio download + playlist URL extraction
    ├── subtitles.py           # Subtitle text extraction (yt-dlp + DownSub fallback)
    ├── audio.py               # Demucs vocal separation + noisereduce + Silero VAD
    ├── transcribe.py          # WhisperX ASR transcription + forced alignment
    └── storage.py             # Audio segment saving + JSONL writer
```

---

## Module Reference

### `config.py`

Central configuration file. Every tuneable value is defined here — no magic numbers are scattered across modules. Importing `config` also creates the output directories.

### `pipeline/models.py`

Loads four models at import time and exposes them as module-level singletons. This ensures models are loaded exactly once regardless of how many videos are processed.

Models loaded:

- **WhisperX transcription** — `large-v3` with `int8` quantization for CPU-friendly inference.
- **Demucs** — `htdemucs` hybrid transformer model for source separation.
- **Silero VAD** — Lightweight voice activity detection model from PyTorch Hub.
- **WhisperX alignment** — Language-specific forced alignment model (gracefully skipped if unavailable for the target language).

### `pipeline/download.py`

**`download_audio(url)`** — Downloads the best audio stream via yt-dlp, converts to 44.1 kHz mono WAV with ffmpeg. Returns a tuple of `(wav_path, info_dict)`. The info dict contains subtitle metadata needed by the subtitle extractor. Temporary files are cleaned up automatically.

**`get_playlist(url, start=1)`** — Extracts individual video URLs from a YouTube playlist using flat extraction (no download). The `start` parameter is 1-indexed and allows resuming from a specific position.

### `pipeline/subtitles.py`

**`get_subtitle_text(info)`** — Parses subtitle data from the yt-dlp info dict. Looks for `json3` format subtitles in priority order: `az` → `az-orig` → `en`, checking manual subs before auto-captions. Words are grouped into sentence-like segments of 3–15 seconds based on punctuation and duration heuristics. Returns a list of `{"start", "end", "text"}` segment dicts, or `None`.

**`get_subtitle_downsub(video_url)`** — Fallback subtitle source. Launches headless Chrome, navigates to downsub.com, submits the video URL, waits for processing, and downloads the Azerbaijani TXT subtitle file. Returns the full text as a string, or `None` on failure. All temporary files and the browser instance are cleaned up in a `finally` block.

### `pipeline/audio.py`

**`clean_audio(wav_path)`** — Runs the Demucs vocal separation model to isolate the vocal stem from drums, bass, and other instruments. The stereo vocal output is averaged to mono, resampled to 16 kHz, and passed through `noisereduce` spectral gating. The original file is deleted and the path to the cleaned file is returned.

**`strip_silence(wav_path)`** — Uses Silero VAD to detect speech regions and concatenates only those segments. A configurable 150 ms gap is inserted between chunks to prevent words from running together. Prints before/after duration statistics (e.g., `[VAD] 245.3s -> 198.1s (removed 47.2s silence)`). The original file is deleted.

### `pipeline/transcribe.py`

**`align_segments(wav_path, segments)`** — WhisperX forced alignment for precise word-level timestamps given pre-existing subtitle text. Requires the alignment model to be available for the target language.

**`transcribe_asr(wav_path)`** — Full ASR fallback: runs WhisperX transcription followed by optional forced alignment. Splits the audio into per-segment WAV files and returns a list of `{"audio_path", "text_raw"}` entries. Segments shorter than 2000 samples are filtered out.

### `pipeline/storage.py`

**`save_segments(wav_path, segments)`** — Splits a full audio file into individual WAV clips based on segment timestamps. Filters out segments that are too short (< 2000 samples) or near-silent (mean absolute amplitude < 0.01).

**`save_jsonl(entries)`** — Appends entries to the JSONL file. Each entry is written as a single JSON line with `ensure_ascii=False` to preserve Azerbaijani characters (ə, ı, ö, ü, ç, ş, ğ).

---

## Troubleshooting

### `HTTP Error 429: Too Many Requests`

YouTube is rate-limiting your requests. The pipeline automatically retries with linear backoff (60s, 120s, 180s). To reduce the chance of hitting limits, increase `DELAY_BETWEEN_VIDEOS` in `config.py`. Using a `cookies.txt` file from an authenticated browser session also helps significantly.

### `Downloaded file not found`

yt-dlp sometimes changes the file extension during download (e.g., `.webm` instead of `.m4a`). The pipeline handles this by scanning for files matching the expected basename, but if the temp directory is cluttered it may pick the wrong file. Clearing `/tmp` can help.

### `No alignment model for 'az'`

WhisperX may not have a forced alignment model for Azerbaijani. This is non-fatal — the pipeline falls back to transcription-only timestamps, which are less precise but still functional. The warning is printed once at startup and can be safely ignored.

### DownSub fallback not working

Ensure Chrome/Chromium and a matching ChromeDriver are installed and accessible on your `PATH`. The DownSub scraper uses headless Chrome with eager page loading, so it can tolerate slow page loads, but the site structure may change over time and break the CSS selectors.

### Out of memory on CPU

Demucs and WhisperX large-v3 are memory-intensive. If you run out of RAM on long videos, try switching to a smaller Whisper model (`medium` or `small`) via `WHISPER_MODEL_SIZE` in `config.py`. You can also process shorter playlists in batches.

### Adapting to a different language

1. Change `LANG` in `config.py` to your target ISO 639-1 language code (e.g., `"tr"` for Turkish, `"en"` for English).
2. Update the DownSub button selector in `subtitles.py` — replace `"Azerbaijani"` in the CSS selector with the language name used by DownSub's UI (e.g., `"Turkish"`).
3. The subtitle extraction priority, ASR language, and alignment model will all follow the `LANG` setting automatically.

---

## License

MIT
