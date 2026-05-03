"""
Extract subtitle text from YouTube videos.

Two strategies:
  1. yt-dlp json3 subtitle parsing (fast, preferred).
  2. DownSub.com Selenium scraper (fallback when no subs via yt-dlp).
"""

import os
import re
import uuid
import time
import shutil
import tempfile

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions

from config import LANG


def get_subtitle_text(info):
    """Parse subtitle text from a yt-dlp info dict (json3 format).

    Args:
        info: yt-dlp info dictionary (must be from a download, not flat extract).

    Returns:
        list[dict] | None: List of {"start", "end", "text"} segments, or None.
    """
    try:
        subs = info.get("subtitles") or {}
        auto = info.get("automatic_captions") or {}

        # find the best subtitle URL
        chosen = None
        for lang_key in [LANG, LANG + "-orig", "en"]:
            for source in [subs, auto]:
                if lang_key in source:
                    for fmt in source[lang_key]:
                        if fmt.get("ext") == "json3":
                            chosen = fmt["url"]
                            break
                    break
            if chosen:
                break

        if not chosen:
            return None

        data = requests.get(chosen).json()

        # collect words with timestamps
        words = []
        for event in data.get("events", []):
            start_ms = event.get("tStartMs", 0)
            for w in event.get("segs", []):
                text = w.get("utf8", "").strip()
                if not text or text == "\n":
                    continue
                offset = w.get("tOffsetMs", 0)
                words.append({
                    "time": (start_ms + offset) / 1000.0,
                    "text": text,
                })

        if not words:
            return None

        # group words into sentence-like segments (~3-15s each)
        segments = []
        current_words = []
        current_start = words[0]["time"]

        for word in words:
            current_words.append(word["text"])
            duration = word["time"] - current_start
            is_sentence_end = word["text"].rstrip().endswith((".", "!", "?"))

            if (is_sentence_end and duration >= 3.0) or duration >= 15.0:
                segments.append({
                    "start": current_start,
                    "end": word["time"] + 0.5,
                    "text": " ".join(current_words),
                })
                current_words = []
                current_start = word["time"] + 0.5

        if current_words:
            segments.append({
                "start": current_start,
                "end": words[-1]["time"] + 0.5,
                "text": " ".join(current_words),
            })

        # clean whitespace
        for seg in segments:
            seg["text"] = re.sub(r"\s+", " ", seg["text"]).strip()

        segments = [s for s in segments if s["text"]]
        return segments if segments else None

    except Exception as e:
        print(f"[SUB FAIL] {e}")
        return None


def get_subtitle_downsub(video_url):
    """Scrape downsub.com via headless Chrome to download subtitle text.

    Args:
        video_url: YouTube video URL.

    Returns:
        str | None: Full subtitle text, or None on failure.
    """
    driver = None
    download_dir = os.path.join(tempfile.gettempdir(), f"downsub_{uuid.uuid4()}")
    os.makedirs(download_dir, exist_ok=True)

    try:
        print("[DOWNSUB] Trying downsub.com...")

        chrome_opts = ChromeOptions()
        chrome_opts.add_argument("--headless=new")
        chrome_opts.add_argument("--no-sandbox")
        chrome_opts.add_argument("--disable-dev-shm-usage")
        chrome_opts.add_argument("--disable-gpu")
        chrome_opts.add_argument("--disable-extensions")
        chrome_opts.add_argument("--disable-images")
        chrome_opts.add_argument("--blink-settings=imagesEnabled=false")
        chrome_opts.page_load_strategy = "eager"
        chrome_opts.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        chrome_opts.add_experimental_option("prefs", {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
        })

        driver = webdriver.Chrome(options=chrome_opts)
        driver.set_page_load_timeout(60)

        # open downsub.com
        try:
            driver.get("https://downsub.com/")
        except Exception:
            pass
        time.sleep(3)

        # enter URL
        input_el = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[name="url"]'))
        )
        input_el.clear()
        input_el.send_keys(video_url)

        # submit
        submit_btn = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
        driver.execute_script("arguments[0].click();", submit_btn)

        # wait for download buttons
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "button.download-button"))
        )
        time.sleep(2)

        # find TXT Azerbaijani button
        buttons = driver.find_elements(
            By.CSS_SELECTOR,
            'button.download-button[data-title*="TXT"][data-title*="Azerbaijani"]',
        )
        target_btn = buttons[0] if buttons else None

        if not target_btn:
            print("[DOWNSUB] No download button found")
            return None

        chosen_title = target_btn.get_attribute("data-title") or ""
        print(f"[DOWNSUB] Clicking: {chosen_title}")
        driver.execute_script("arguments[0].click();", target_btn)

        # wait for file to download
        srt_path = None
        for _ in range(30):
            time.sleep(1)
            files = os.listdir(download_dir)
            done = [f for f in files if not f.endswith(".crdownload")]
            if done:
                srt_path = os.path.join(download_dir, done[0])
                break

        driver.quit()
        driver = None

        if not srt_path or not os.path.exists(srt_path):
            print("[DOWNSUB] Download timed out")
            return None

        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            print("[DOWNSUB] Empty subtitle file")
            return None

        full_text = re.sub(r"\s+", " ", content).strip()
        if full_text:
            print(f"[DOWNSUB] Got {len(full_text)} chars")

        return full_text

    except Exception as e:
        print(f"[DOWNSUB FAIL] {e}")
        return None

    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
        shutil.rmtree(download_dir, ignore_errors=True)
