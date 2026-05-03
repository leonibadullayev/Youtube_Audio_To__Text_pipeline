import os

# =========================
# PATHS
# =========================
OUTPUT_DIR = "dataset"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
JSONL_PATH = os.path.join(OUTPUT_DIR, "data.jsonl")

# =========================
# MODEL / INFERENCE
# =========================
DEVICE = "cpu"
BATCH_SIZE = 4
LANG = "az"
WHISPER_MODEL_SIZE = "large-v3"
WHISPER_COMPUTE_TYPE = "int8"
DEMUCS_MODEL_NAME = "htdemucs"

# =========================
# AUDIO PROCESSING
# =========================
TARGET_SAMPLE_RATE = 16000
DEMUCS_DOWNLOAD_SAMPLE_RATE = 44100
NOISE_REDUCE_PROP = 0.7
MIN_SPEECH_MS = 250
MIN_SILENCE_MS = 300
SPEECH_PAD_MS = 100
INTER_CHUNK_GAP_SEC = 0.15

# =========================
# DOWNLOADING
# =========================
COOKIE_FILE = "cookies.txt"
DELAY_BETWEEN_VIDEOS = 5
MAX_RETRIES = 3
RATE_LIMIT_BASE_WAIT = 60

# =========================
# INIT
# =========================
os.makedirs(AUDIO_DIR, exist_ok=True)
