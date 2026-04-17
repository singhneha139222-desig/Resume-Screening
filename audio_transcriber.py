"""
audio_transcriber.py
--------------------
Converts audio files to text using OpenAI's Whisper model.
Supports both local file paths and URLs.
Requires ffmpeg to be installed on the system.
"""

import logging
import os
import tempfile
import threading
import time
from urllib.parse import urlparse

import requests
import whisper

# ── Auto-inject ffmpeg PATH (Windows) ──
# If ffmpeg is installed at C:\ffmpeg but not in system PATH (common when
# installed without admin rights), add it to the current process PATH.
_FFMPEG_COMMON_PATHS = [
    r"C:\ffmpeg",
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    r"C:\Program Files (x86)\ffmpeg\bin",
]
for _fp in _FFMPEG_COMMON_PATHS:
    if os.path.isfile(os.path.join(_fp, "ffmpeg.exe")):
        if _fp not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _fp + os.pathsep + os.environ.get("PATH", "")
            break

# Cache the model so it's only loaded once
_model = None
_model_lock = threading.Lock()
logger = logging.getLogger(__name__)

# Map Content-Type to file extensions
CONTENT_TYPE_TO_EXT = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-m4a": ".m4a",
    "audio/mp4": ".m4a",
    "audio/m4a": ".m4a",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "video/mp4": ".mp4",
    "application/octet-stream": ".mp3",  # fallback for generic binary
}


def _get_model(model_name="base"):
    """
    Load and cache the Whisper model.
    
    Available models (smallest to largest):
        tiny, base, small, medium, large
    
    Args:
        model_name (str): Which Whisper model to use.
    
    Returns:
        whisper.Whisper: The loaded model.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info("Loading Whisper model '%s'...", model_name)
                _model = whisper.load_model(model_name)
                logger.info("Whisper model loaded.")
    return _model


def _download_audio_to_tempfile(source_url, timeout=120, max_retries=3):
    """
    Download audio from a URL to a temporary file.
    Includes retry logic, proper extension detection, and User-Agent header.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Downloading audio (attempt %d/%d): %s", attempt, max_retries, source_url
            )
            response = requests.get(
                source_url, stream=True, timeout=timeout, headers=headers
            )
            response.raise_for_status()

            # Determine file extension from Content-Type header
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
            ext = CONTENT_TYPE_TO_EXT.get(content_type)

            # Fallback: try to extract extension from URL path
            if not ext:
                parsed = urlparse(source_url)
                url_ext = os.path.splitext(parsed.path)[1].lower()
                if url_ext in {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}:
                    ext = url_ext
                else:
                    ext = ".mp3"  # safe default for Whisper

            logger.info("Detected content type: %s -> extension: %s", content_type, ext)

            # Write to temp file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            total_bytes = 0
            try:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        tmp.write(chunk)
                        total_bytes += len(chunk)
            finally:
                tmp.close()

            logger.info("Downloaded %d bytes to %s", total_bytes, tmp.name)

            if total_bytes == 0:
                os.unlink(tmp.name)
                raise RuntimeError("Downloaded file is empty (0 bytes)")

            return tmp.name

        except Exception as e:
            last_error = e
            logger.warning("Download attempt %d failed: %s", attempt, e)
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.info("Retrying in %d seconds...", wait)
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to download audio after {max_retries} attempts. Last error: {last_error}"
    )


def _check_ffmpeg():
    """Check if ffmpeg is installed and available in PATH."""
    import subprocess
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5
        )
        return True
    except (FileNotFoundError, OSError):
        return False


def transcribe_from_local_file(local_path, model_name="base"):
    """
    Transcribe audio directly from a local file path.
    This is the PRIMARY method — saves the audio locally first,
    then runs Whisper on it.  No network download needed.
    
    Args:
        local_path (str): Absolute path to the audio file on disk.
        model_name (str): Whisper model size (default: "base").
    
    Returns:
        dict: {
            "text": str,        # Full transcription
            "language": str,    # Detected language
            "success": bool,    # Whether transcription succeeded
            "error": str|None   # Error message if failed
        }
    """
    try:
        # --- Check ffmpeg first (WinError 2 prevention) ---
        if not _check_ffmpeg():
            msg = (
                "ffmpeg not found! Whisper needs ffmpeg to decode audio. "
                "Install it: Windows → 'winget install ffmpeg' then restart your terminal. "
                "Or download from https://ffmpeg.org/download.html and add to PATH."
            )
            logger.error(msg)
            return {"text": "", "language": "", "success": False, "error": msg}

        if not os.path.exists(local_path):
            return {
                "text": "",
                "language": "",
                "success": False,
                "error": f"Local audio file not found: {local_path}",
            }

        file_size = os.path.getsize(local_path)
        if file_size == 0:
            return {
                "text": "",
                "language": "",
                "success": False,
                "error": "Audio file is empty (0 bytes)",
            }

        logger.info("Transcribing local file: %s (%d bytes)", local_path, file_size)
        model = _get_model(model_name)
        result = model.transcribe(local_path)
        text = result.get("text", "").strip()
        language = result.get("language", "unknown")
        logger.info("Transcription complete. Language: %s, Length: %d chars", language, len(text))

        return {
            "text": text,
            "language": language,
            "success": True,
            "error": None,
        }

    except Exception as e:
        error_msg = str(e)
        logger.exception("Transcription failed for local file: %s", local_path)

        # WinError 2 = file not found = typically ffmpeg not in PATH
        if "winerror 2" in error_msg.lower() or "[winerror 2]" in error_msg.lower() or "cannot find the file" in error_msg.lower():
            error_msg = (
                "ffmpeg not found in PATH (WinError 2). "
                "Install it: run 'winget install ffmpeg' in PowerShell as Administrator, then restart. "
                "Or download from https://ffmpeg.org/download.html"
            )
        elif "ffmpeg" in error_msg.lower():
            error_msg = (
                "ffmpeg error. Please install ffmpeg and ensure it is in your system PATH. "
                "Windows: 'winget install ffmpeg'"
            )

        return {
            "text": "",
            "language": "",
            "success": False,
            "error": error_msg,
        }


def transcribe_audio(audio_source, model_name="base"):
    """
    Transcribe an audio file to text using Whisper.
    
    Supported formats: mp3, wav, m4a, flac, ogg, webm
    
    Args:
        audio_source (str): Path or URL to the audio file.
        model_name (str): Whisper model size (default: "base").
    
    Returns:
        dict: {
            "text": str,        # Full transcription
            "language": str,    # Detected language
            "success": bool,    # Whether transcription succeeded
            "error": str|None   # Error message if failed
        }
    """
    local_path = None
    is_url = False

    try:
        model = _get_model(model_name)
        is_url = str(audio_source).startswith(("http://", "https://"))

        if is_url:
            logger.info("Audio source is a URL, downloading: %s", audio_source)
            local_path = _download_audio_to_tempfile(audio_source)
        else:
            local_path = audio_source
            if not os.path.exists(local_path):
                return {
                    "text": "",
                    "language": "",
                    "success": False,
                    "error": f"Local audio file not found: {local_path}",
                }

        logger.info("Starting Whisper transcription on: %s", local_path)
        result = model.transcribe(local_path)
        text = result.get("text", "").strip()
        language = result.get("language", "unknown")
        logger.info("Transcription complete. Language: %s, Length: %d chars", language, len(text))

        return {
            "text": text,
            "language": language,
            "success": True,
            "error": None,
        }

    except Exception as e:
        error_msg = str(e)
        logger.exception("Transcription failed for source: %s", audio_source)

        # Check for common ffmpeg error
        if "ffmpeg" in error_msg.lower():
            error_msg = (
                "ffmpeg is not installed or not found in PATH. "
                "Please install ffmpeg: https://ffmpeg.org/download.html"
            )

        return {
            "text": "",
            "language": "",
            "success": False,
            "error": error_msg,
        }

    finally:
        # Clean up temp file only if we downloaded it
        if is_url and local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
                logger.info("Cleaned up temp file: %s", local_path)
            except Exception:
                pass
