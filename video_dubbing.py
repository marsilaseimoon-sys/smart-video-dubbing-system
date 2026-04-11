import os
import sys
import subprocess
import tempfile
import math
import re
import asyncio
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# BASE DIRECTORY
# ══════════════════════════════════════════════════════════════

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_DIR  = os.path.join(BASE_DIR, "Wav2Lip")
OUTPUT_DIR   = os.path.join(BASE_DIR, "dubbed_videos")
HF_API_TOKEN = None

# ── Cookies.txt path (optional — rakhein toh YouTube better kaam karega) ──
# Export karne ka tarika:
#   Chrome extension: "Get cookies.txt LOCALLY"
#   Firefox extension: "cookies.txt"
#   Phir yahan path dein:
COOKIES_TXT_PATH = os.path.join(BASE_DIR, "cookies.txt")   # optional

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Dependencies
# ══════════════════════════════════════════════════════════════

try:
    import whisper
    WHISPER_AVAILABLE = True
    print("[INFO] Loading Whisper model (small)...")
    WHISPER_MODEL = whisper.load_model("small")
    print("[INFO] Whisper model loaded.")
except Exception as e:
    print("[WARN] Whisper not available:", e)
    WHISPER_AVAILABLE = False
    WHISPER_MODEL = None

try:
    from deep_translator import GoogleTranslator
    print("[INFO] deep_translator loaded.")
except Exception as e:
    print("[WARN] deep_translator not available:", e)
    GoogleTranslator = None

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
    print("[INFO] edge_tts loaded.")
except Exception as e:
    print("[WARN] edge_tts not available — install: pip install edge-tts")
    EDGE_TTS_AVAILABLE = False

try:
    from gtts import gTTS
    print("[INFO] gTTS loaded.")
except Exception as e:
    print("[WARN] gTTS not available:", e)
    gTTS = None

try:
    from pydub import AudioSegment
    print("[INFO] pydub loaded.")
except Exception as e:
    print("[WARN] pydub not available:", e)
    AudioSegment = None

try:
    from pyannote.audio import Pipeline as PyannotePipeline
    PYANNOTE_AVAILABLE = True
    print("[INFO] pyannote loaded.")
except Exception as e:
    print("[INFO] pyannote not available (diarization disabled):", e)
    PYANNOTE_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
# Edge TTS — voice map
# ══════════════════════════════════════════════════════════════

EDGE_LANG_VOICES = {
    "ur":    ("ur-PK-AsadNeural",     "ur-PK-UzmaNeural"),
    "en":    ("en-US-GuyNeural",      "en-US-JennyNeural"),
    "ar":    ("ar-SA-HamedNeural",    "ar-SA-ZariyahNeural"),
    "hi":    ("hi-IN-MadhurNeural",   "hi-IN-SwaraNeural"),
    "fr":    ("fr-FR-HenriNeural",    "fr-FR-DeniseNeural"),
    "de":    ("de-DE-ConradNeural",   "de-DE-KatjaNeural"),
    "es":    ("es-ES-AlvaroNeural",   "es-ES-ElviraNeural"),
    "it":    ("it-IT-DiegoNeural",    "it-IT-ElsaNeural"),
    "pt":    ("pt-BR-AntonioNeural",  "pt-BR-FranciscaNeural"),
    "ja":    ("ja-JP-KeitaNeural",    "ja-JP-NanamiNeural"),
    "ko":    ("ko-KR-InJoonNeural",   "ko-KR-SunHiNeural"),
    "zh-cn": ("zh-CN-YunxiNeural",    "zh-CN-XiaoxiaoNeural"),
    "tr":    ("tr-TR-AhmetNeural",    "tr-TR-EmelNeural"),
}

CHARACTER_GENDER = {
    "Ali":   0,
    "Ahmed": 0,
    "Sara":  1,
    "Zoya":  1,
    "Child": 1,
}
DEFAULT_GENDER = 0


def get_edge_voice(label: str, target_lang: str) -> str:
    lang        = target_lang.lower()
    voices      = EDGE_LANG_VOICES.get(lang, EDGE_LANG_VOICES["en"])
    label_clean = label.strip().rstrip(":").strip()
    if label_clean in CHARACTER_GENDER:
        return voices[CHARACTER_GENDER[label_clean]]
    if label_clean.upper().startswith("SPEAKER"):
        digits = ''.join(filter(str.isdigit, label_clean))
        idx    = int(digits) if digits else 0
        return voices[idx % 2]
    lc = label_clean.lower()
    if any(k in lc for k in ("female", "woman", "sara", "zoya")):
        return voices[1]
    if any(k in lc for k in ("male", "man", "ali", "ahmed")):
        return voices[0]
    return voices[DEFAULT_GENDER]


async def _edge_tts_synthesize(text: str, voice: str, output_path: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def synthesize_with_edge_tts(text: str, voice: str, output_path: str) -> bool:
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run,
                                         _edge_tts_synthesize(text, voice, output_path))
                    future.result(timeout=60)
            else:
                loop.run_until_complete(_edge_tts_synthesize(text, voice, output_path))
        except RuntimeError:
            asyncio.run(_edge_tts_synthesize(text, voice, output_path))
        return os.path.exists(output_path) and os.path.getsize(output_path) > 100
    except Exception as e:
        print(f"[ERROR] Edge TTS failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# URL helpers
# ══════════════════════════════════════════════════════════════

def is_url(path_or_url: str) -> bool:
    return re.match(r'^https?://', path_or_url.strip()) is not None


def is_youtube_url(url: str) -> bool:
    patterns = [
        r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/',
        r'(https?://)?(www\.)?youtube\.com/shorts/',
        r'(https?://)?(m\.)?youtube\.com/',
    ]
    return any(re.search(p, url) for p in patterns)


# ══════════════════════════════════════════════════════════════
# yt-dlp helpers  —  FIXED YouTube 403
# ══════════════════════════════════════════════════════════════

def check_yt_dlp() -> bool:
    try:
        subprocess.run(["yt-dlp", "--version"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def ensure_yt_dlp_updated():
    """Update yt-dlp quickly — 30s timeout only."""
    print("[INFO] Checking yt-dlp version...")
    for extra_args in [[], ["--user"]]:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp", "-q"]
                + extra_args,
                capture_output=True, text=True,
                timeout=30,
            )
            if result.returncode == 0:
                print("[INFO] yt-dlp is up to date.")
                return True
        except subprocess.TimeoutExpired:
            print("[WARN] yt-dlp update timed out — using current version.")
            return False
        except Exception as e:
            print(f"[WARN] pip upgrade exception: {e}")
    print("[WARN] Could not update yt-dlp.")
    return False


def _get_cookies_args() -> list:
    """
    Returns cookie args for yt-dlp.
    Priority: cookies.txt file → browser cookies → no cookies
    """
    if os.path.isfile(COOKIES_TXT_PATH):
        print(f"[INFO] Using cookies.txt: {COOKIES_TXT_PATH}")
        return ["--cookies", COOKIES_TXT_PATH]

    for browser in ["chrome", "firefox", "edge", "brave", "opera", "chromium"]:
        try:
            result = subprocess.run(
                ["yt-dlp", "--cookies-from-browser", browser,
                 "--simulate", "--quiet",
                 "https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(f"[INFO] Using cookies from browser: {browser}")
                return ["--cookies-from-browser", browser]
        except Exception:
            continue

    print("[INFO] No cookies found — will try without cookies.")
    return []


def _run_yt_dlp(cmd: list, timestamp: str, progress_callback=None) -> str | None:
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        downloaded_file = None

        for line in process.stdout:
            line = line.strip()
            if line:
                print(f"[yt-dlp] {line}")

            pct_match = re.search(r'(\d+\.?\d*)%', line)
            if pct_match and progress_callback:
                pct = float(pct_match.group(1))
                progress_callback(stage='download', status='active',
                                  progress=5 + int(pct * 0.04),
                                  message=f'Downloading... {pct:.1f}%')

            if '[download] Destination:' in line:
                downloaded_file = line.split('[download] Destination:')[-1].strip()
            if 'Merging formats into' in line:
                downloaded_file = line.split('Merging formats into')[-1].strip().strip('"')
            if 'ExtractAudio] Destination:' in line:
                downloaded_file = line.split('Destination:')[-1].strip()

        process.wait()
        if process.returncode != 0:
            return None

        if not downloaded_file or not os.path.exists(downloaded_file):
            import glob
            files = sorted(
                glob.glob(os.path.join(OUTPUT_DIR, f"downloaded_{timestamp}.*")),
                key=os.path.getmtime, reverse=True,
            )
            downloaded_file = files[0] if files else None

        return downloaded_file if downloaded_file and os.path.exists(downloaded_file) else None

    except Exception as e:
        print(f"[ERROR] yt-dlp run exception: {e}")
        return None


def _build_yt_strategies(output_tmpl: str, cookie_args: list, url: str) -> list:
    """
    5 download strategies in order of reliability for YouTube 403 fixes.
    android_creator client bypasses 403 best without cookies.
    """
    base = [
        "yt-dlp",
        "--no-playlist",
        "--merge-output-format", "mp4",
        "--no-check-certificate",
        "--geo-bypass",
        "--retries", "3",
        "--fragment-retries", "3",
        "--socket-timeout", "30",
        "--no-warnings",
        "--progress",
        "--output", output_tmpl,
    ]

    return [
        # Strategy 1: android_creator — best 403 bypass
        (
            "android_creator client (best for 403)",
            base + [
                "--extractor-args", "youtube:player_client=android_creator",
                "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            ] + cookie_args + [url],
        ),
        # Strategy 2: iOS client
        (
            "ios client",
            base + [
                "--extractor-args", "youtube:player_client=ios",
                "-f", "best[ext=mp4]/best",
            ] + cookie_args + [url],
        ),
        # Strategy 3: android_embedded
        (
            "android_embedded client",
            base + [
                "--extractor-args", "youtube:player_client=android_embedded",
                "-f", "best[ext=mp4]/best",
            ] + cookie_args + [url],
        ),
        # Strategy 4: web + cookies
        (
            "web client + cookies",
            base + [
                "--extractor-args", "youtube:player_client=web",
                "-f", "best[ext=mp4]/best",
            ] + cookie_args + [url],
        ),
        # Strategy 5: bare — no client override, no cookies
        (
            "bare fallback (no client override)",
            [
                "yt-dlp", "--no-playlist",
                "--merge-output-format", "mp4",
                "--no-check-certificate", "--geo-bypass",
                "--retries", "2", "--no-warnings", "--progress",
                "-f", "best",
                "--output", output_tmpl,
                url,
            ],
        ),
    ]


def download_video_from_url(url: str, progress_callback=None) -> str | None:
    if progress_callback:
        progress_callback(stage='download', status='active', progress=2,
                          message='Checking download tool...')

    if not check_yt_dlp():
        print("[INFO] yt-dlp not found — installing...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "yt-dlp", "-q", "--user"],
                check=True, timeout=60
            )
        except Exception as e:
            print(f"[ERROR] Could not install yt-dlp: {e}")
            if progress_callback:
                progress_callback(stage='download', status='error', progress=0,
                                  message='yt-dlp not found. Run: pip install yt-dlp')
            return None

    ensure_yt_dlp_updated()

    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_tmpl = os.path.join(OUTPUT_DIR, f"downloaded_{timestamp}.%(ext)s")
    print(f"[INFO] Downloading: {url}")

    cookie_args = _get_cookies_args() if is_youtube_url(url) else []

    if is_youtube_url(url):
        strategies = _build_yt_strategies(output_tmpl, cookie_args, url)
    else:
        strategies = [
            (
                "direct best",
                [
                    "yt-dlp", "--no-playlist", "-f", "best[ext=mp4]/best",
                    "--merge-output-format", "mp4",
                    "--no-check-certificate", "--retries", "3",
                    "--no-warnings", "--progress",
                    "--output", output_tmpl, url,
                ],
            )
        ]

    for attempt, (desc, cmd) in enumerate(strategies, 1):
        if progress_callback:
            progress_callback(stage='download', status='active', progress=5,
                              message=f'Download attempt {attempt}: {desc}...')

        print(f"\n[INFO] Attempt {attempt}/{len(strategies)}: {desc}")
        result = _run_yt_dlp(cmd, timestamp, progress_callback)

        if result:
            size_mb = os.path.getsize(result) / (1024 * 1024)
            print(f"[INFO] Download complete: {result} ({size_mb:.1f} MB)")
            if progress_callback:
                progress_callback(stage='download', status='completed', progress=10,
                                  message=f'Download complete ({size_mb:.1f} MB)')
            return result

        print(f"[WARN] Attempt {attempt} failed.")

    # All failed
    print("\n[ERROR] All download attempts failed.")
    print("=" * 55)
    print("[FIX OPTIONS]")
    print("  Option 1 (easiest): Upload the video file directly")
    print("    — Download the YouTube video using any online tool")
    print("    — Upload the .mp4 file in the app")
    print()
    print("  Option 2: Add cookies.txt")
    print("    1. Open Chrome/Firefox and go to youtube.com (log in)")
    print("    2. Install extension: 'Get cookies.txt LOCALLY' (Chrome)")
    print("       or 'cookies.txt' (Firefox)")
    print("    3. Export cookies for youtube.com")
    print(f"    4. Save as: {COOKIES_TXT_PATH}")
    print("    5. Try again")
    print()
    print("  Option 3: Update yt-dlp manually")
    print("    Run:  yt-dlp -U")
    print("=" * 55)

    if progress_callback:
        progress_callback(
            stage='download', status='error', progress=0,
            message=(
                'YouTube 403 error. Fix: '
                '(1) Upload .mp4 directly, OR '
                '(2) Add cookies.txt next to APP.py (export from browser). '
                'See console for details.'
            )
        )
    return None


# ══════════════════════════════════════════════════════════════
# Audio extraction
# ══════════════════════════════════════════════════════════════

def extract_audio(video_path: str, out_wav: str, progress_callback=None) -> str | None:
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return None

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        out_wav
    ]

    try:
        if progress_callback:
            progress_callback(stage='extract', status='active', progress=15,
                              message='Extracting audio from video...')

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[ERROR] FFmpeg audio extract failed:\n{result.stderr}")
            if progress_callback:
                progress_callback(stage='extract', status='error', progress=0,
                                  message='Audio extraction failed')
            return None

        if not os.path.exists(out_wav) or os.path.getsize(out_wav) < 100:
            print("[ERROR] Extracted WAV is missing or empty.")
            return None

        print(f"[INFO] Audio extracted: {out_wav} ({os.path.getsize(out_wav) // 1024} KB)")
        if progress_callback:
            progress_callback(stage='extract', status='completed', progress=20,
                              message='Audio extraction completed')
        return out_wav

    except Exception as e:
        print(f"[ERROR] extract_audio exception: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# Transcription
# ══════════════════════════════════════════════════════════════

def transcribe_long_audio(audio_path: str, chunk_seconds: int = 30,
                          progress_callback=None) -> str:
    if not WHISPER_AVAILABLE or WHISPER_MODEL is None:
        print("[WARN] Whisper unavailable.")
        return ""
    if AudioSegment is None:
        print("[WARN] pydub unavailable.")
        return ""
    if not os.path.exists(audio_path):
        print(f"[ERROR] Audio file missing: {audio_path}")
        return ""

    if progress_callback:
        progress_callback(stage='transcribe', status='active', progress=25,
                          message='Starting transcription...')

    audio       = AudioSegment.from_file(audio_path)
    duration_ms = len(audio)
    chunk_ms    = chunk_seconds * 1000
    parts       = math.ceil(duration_ms / chunk_ms)
    transcripts = []

    for i in range(parts):
        chunk = audio[i * chunk_ms: min((i + 1) * chunk_ms, duration_ms)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            chunk_path = tf.name
        chunk.export(chunk_path, format="wav")
        try:
            res  = WHISPER_MODEL.transcribe(chunk_path)
            text = res.get("text", "").strip()
            transcripts.append(text)
        except Exception as e:
            print(f"[WARN] Whisper chunk {i} failed: {e}")
            transcripts.append("")
        finally:
            try:
                os.remove(chunk_path)
            except Exception:
                pass

        if progress_callback:
            progress_callback(stage='transcribe', status='active',
                              progress=25 + int((i + 1) / parts * 15),
                              message=f'Transcribing... ({i + 1}/{parts} chunks)')

    if progress_callback:
        progress_callback(stage='transcribe', status='completed', progress=40,
                          message='Transcription completed')

    full_text = " ".join(t for t in transcripts if t)
    print(f"[INFO] Transcription ({len(full_text)} chars): {full_text[:100]}...")
    return full_text


# ══════════════════════════════════════════════════════════════
# Translation
# ══════════════════════════════════════════════════════════════

def translate_text_chunked(text: str, target_lang: str = "ur",
                            max_chunk_chars: int = 1500,
                            progress_callback=None) -> str:
    if not GoogleTranslator:
        print("[WARN] GoogleTranslator unavailable — returning original text.")
        return text
    if not text.strip():
        return text

    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks    = []
    current   = ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chunk_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)

    translated_parts = []
    total = len(chunks)
    for idx, ch in enumerate(chunks):
        try:
            translated = GoogleTranslator(source='auto', target=target_lang).translate(ch)
        except Exception as e:
            print(f"[WARN] Translation error chunk {idx}: {e}")
            translated = ch
        translated_parts.append(translated or ch)
        if progress_callback:
            progress_callback(stage='translate', status='active',
                              progress=45 + int((idx + 1) / total * 15),
                              message=f'Translating... ({idx + 1}/{total} chunks)')

    if progress_callback:
        progress_callback(stage='translate', status='completed', progress=60,
                          message='Translation completed')

    return " ".join(translated_parts)


# ══════════════════════════════════════════════════════════════
# Diarization (optional)
# ══════════════════════════════════════════════════════════════

def run_diarization(audio_path: str, hf_token: str, progress_callback=None):
    if not PYANNOTE_AVAILABLE or not hf_token or not hf_token.strip():
        return None
    try:
        if progress_callback:
            progress_callback(stage='transcribe', status='active', progress=35,
                              message='Running speaker diarization...')
        pipeline    = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1", use_auth_token=hf_token)
        diarization = pipeline(audio_path)
        speakers    = [(speaker, turn.start, turn.end)
                       for turn, _, speaker in diarization.itertracks(yield_label=True)]
        print(f"[INFO] Diarization: {len(speakers)} segments found")
        return speakers
    except Exception as e:
        print(f"[ERROR] Diarization failed: {e}")
        return None


def assign_sentences_to_speakers(translated_text: str,
                                  diarization_segments=None) -> str:
    sentences = [s.strip()
                 for s in re.split(r'(?<=[\.\?\!])\s+', translated_text)
                 if s.strip()]
    if not sentences:
        return translated_text

    if diarization_segments:
        speakers_order = []
        for sp, _, __ in diarization_segments:
            if sp not in speakers_order:
                speakers_order.append(sp)
        speakers_order = speakers_order or ["SPEAKER_00"]
        return "\n".join(
            f"{speakers_order[i % len(speakers_order)]}: {s}"
            for i, s in enumerate(sentences)
        )

    default_chars = ["Ali", "Sara", "Ahmed"]
    return "\n".join(
        f"{default_chars[i % len(default_chars)]}: {s}"
        for i, s in enumerate(sentences)
    )


# ══════════════════════════════════════════════════════════════
# TTS  —  Edge TTS (primary) → gTTS (fallback)
# ══════════════════════════════════════════════════════════════

def synthesize_speech_character_wise(text: str, target_lang: str = "en",
                                     progress_callback=None) -> str:
    if AudioSegment is None:
        raise RuntimeError("pydub is required — pip install pydub")
    if not text.strip():
        raise RuntimeError("Empty text — nothing to synthesize.")

    if progress_callback:
        progress_callback(stage='synthesize', status='active', progress=65,
                          message='Starting speech synthesis...')

    lines   = [l.strip() for l in text.split("\n") if l.strip()]
    outputs = []
    total   = len(lines)

    for idx, line in enumerate(lines):
        if ":" in line:
            label, content = line.split(":", 1)
            label, content = label.strip(), content.strip()
        else:
            label, content = "Ali", line

        if not content:
            continue

        temp_out = os.path.join(
            OUTPUT_DIR,
            f"tts_part_{idx}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.mp3"
        )
        success = False

        if EDGE_TTS_AVAILABLE:
            voice   = get_edge_voice(label, target_lang)
            success = synthesize_with_edge_tts(content, voice, temp_out)
            if success:
                print(f"[INFO] Edge TTS OK [{voice}] line {idx}")
            else:
                print(f"[WARN] Edge TTS failed line {idx} — trying gTTS")

        if not success and gTTS is not None:
            try:
                gtts_lang = target_lang.split("-")[0] if "-" in target_lang else target_lang
                gTTS(text=content, lang=gtts_lang).save(temp_out)
                success = True
                print(f"[INFO] gTTS fallback OK line {idx}")
            except Exception as e:
                print(f"[ERROR] gTTS line {idx}: {e}")

        if success:
            outputs.append(temp_out)
        else:
            print(f"[ERROR] Could not synthesize line {idx} — skipping")

        if progress_callback and total > 0:
            progress_callback(
                stage='synthesize', status='active',
                progress=65 + int((idx + 1) / total * 20),
                message=f'Synthesizing speech... ({idx + 1}/{total} lines)'
            )

    if not outputs:
        raise RuntimeError("No TTS outputs generated.")

    combined = None
    for p in outputs:
        try:
            seg      = AudioSegment.from_file(p)
            combined = seg if combined is None else combined + seg
        except Exception as e:
            print(f"[WARN] Could not load audio segment {p}: {e}")

    if combined is None:
        raise RuntimeError("All audio segments failed to load.")

    final_out = os.path.join(
        OUTPUT_DIR,
        f"dubbed_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    )
    combined.export(final_out, format="wav")

    for p in outputs:
        try:
            os.remove(p)
        except Exception:
            pass

    size = os.path.getsize(final_out)
    print(f"[INFO] Final dubbed audio: {final_out} ({size // 1024} KB)")

    if size < 1000:
        raise RuntimeError(f"Dubbed audio too small ({size} bytes).")

    if progress_callback:
        progress_callback(stage='synthesize', status='completed', progress=85,
                          message='Speech synthesis completed')

    return final_out


# ══════════════════════════════════════════════════════════════
# Wav2Lip
# ══════════════════════════════════════════════════════════════

def run_wav2lip(video_path: str, audio_path: str, output_path: str,
                progress_callback=None) -> str | None:
    abs_video   = os.path.abspath(video_path)
    abs_audio   = os.path.abspath(audio_path)
    abs_output  = os.path.abspath(output_path)
    checkpoint  = os.path.join(WAV2LIP_DIR, "Wav2Lip.pth")
    inference   = os.path.join(WAV2LIP_DIR, "inference.py")

    missing = []
    if not os.path.isdir(WAV2LIP_DIR):
        missing.append(f"Wav2Lip directory: {WAV2LIP_DIR}")
    if not os.path.isfile(checkpoint):
        missing.append(f"Checkpoint: {checkpoint}")
    if not os.path.isfile(inference):
        missing.append(f"inference.py: {inference}")
    if not os.path.isfile(abs_video):
        missing.append(f"Video: {abs_video}")
    if not os.path.isfile(abs_audio):
        missing.append(f"Audio: {abs_audio}")

    if missing:
        print("[ERROR] Wav2Lip pre-flight failed — missing:")
        for m in missing:
            print(f"        - {m}")
        print("[INFO] Falling back to FFmpeg merge.")
        return merge_audio_video(abs_video, abs_audio, abs_output, progress_callback)

    try:
        if progress_callback:
            progress_callback(stage='merge', status='active', progress=90,
                              message='Running Wav2Lip lip-sync...')

        env         = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = WAV2LIP_DIR + (os.pathsep + existing_pp if existing_pp else "")

        cmd = [
            sys.executable, "inference.py",
            "--checkpoint_path", checkpoint,
            "--face",    abs_video,
            "--audio",   abs_audio,
            "--outfile", abs_output,
        ]

        print(f"[INFO] Running Wav2Lip (cwd={WAV2LIP_DIR})")

        result = subprocess.run(
            cmd, cwd=WAV2LIP_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=600,
        )

        for line in (result.stdout or "").strip().splitlines():
            print(f"[Wav2Lip] {line}")

        if (result.returncode == 0
                and os.path.exists(abs_output)
                and os.path.getsize(abs_output) > 10_000):
            print(f"[SUCCESS] Wav2Lip done: {abs_output}")
            if progress_callback:
                progress_callback(stage='merge', status='completed', progress=100,
                                  message='Lip-sync completed successfully!')
            return abs_output

        print(f"[ERROR] Wav2Lip code {result.returncode} — falling back to FFmpeg.")
        return merge_audio_video(abs_video, abs_audio, abs_output, progress_callback)

    except subprocess.TimeoutExpired:
        print("[ERROR] Wav2Lip timed out — falling back to FFmpeg.")
        return merge_audio_video(abs_video, abs_audio, abs_output, progress_callback)
    except Exception as e:
        print(f"[ERROR] Wav2Lip exception: {e} — falling back to FFmpeg.")
        return merge_audio_video(abs_video, abs_audio, abs_output, progress_callback)


# ══════════════════════════════════════════════════════════════
# FFmpeg merge
# ══════════════════════════════════════════════════════════════

def merge_audio_video(video_path: str, dubbed_audio: str,
                      output_name: str | None = None,
                      progress_callback=None) -> str | None:
    if output_name is None:
        output_name = os.path.join(
            OUTPUT_DIR,
            f"dubbed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )

    video_path   = os.path.abspath(video_path)
    dubbed_audio = os.path.abspath(dubbed_audio)
    output_name  = os.path.abspath(output_name)

    print("\n========== MERGE DEBUG ==========")
    print(f"  Video  : {video_path}  exists={os.path.exists(video_path)}")
    print(f"  Audio  : {dubbed_audio}  exists={os.path.exists(dubbed_audio)}"
          + (f"  size={os.path.getsize(dubbed_audio)}" if os.path.exists(dubbed_audio) else ""))
    print(f"  Output : {output_name}")
    print("==================================\n")

    if not os.path.exists(video_path):
        print("[ERROR] Video file missing!")
        return None
    if not os.path.exists(dubbed_audio) or os.path.getsize(dubbed_audio) < 1000:
        print("[ERROR] Dubbed audio missing or too small!")
        return None

    if progress_callback:
        progress_callback(stage='merge', status='active', progress=90,
                          message='Merging audio with video...')

    os.makedirs(os.path.dirname(output_name), exist_ok=True)

    # Attempt 1: copy video (fast)
    cmd1 = [
        "ffmpeg", "-y",
        "-i", video_path, "-i", dubbed_audio,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_name,
    ]
    result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=300)

    if (result1.returncode == 0
            and os.path.exists(output_name)
            and os.path.getsize(output_name) > 10_000):
        size_mb = os.path.getsize(output_name) / (1024 * 1024)
        print(f"[SUCCESS] Merge complete: {output_name} ({size_mb:.1f} MB)")
        if progress_callback:
            progress_callback(stage='merge', status='completed', progress=100,
                              message='Video dubbing completed successfully!')
        return output_name

    print(f"[WARN] FFmpeg copy failed — retrying with re-encode...")

    # Attempt 2: re-encode video (handles codec mismatches)
    cmd2 = [
        "ffmpeg", "-y",
        "-i", video_path, "-i", dubbed_audio,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_name,
    ]
    result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)

    if (result2.returncode == 0
            and os.path.exists(output_name)
            and os.path.getsize(output_name) > 10_000):
        size_mb = os.path.getsize(output_name) / (1024 * 1024)
        print(f"[SUCCESS] Re-encode merge: {output_name} ({size_mb:.1f} MB)")
        if progress_callback:
            progress_callback(stage='merge', status='completed', progress=100,
                              message='Video dubbing completed successfully!')
        return output_name

    print(f"[ERROR] Both merge attempts failed.\n{result2.stderr[-2000:]}")
    if progress_callback:
        progress_callback(stage='merge', status='error', progress=0,
                          message='Video merge failed')
    return None


# ══════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════

def process_video_pipeline(video_path_or_url: str, target_lang: str = "en",
                            use_diarization: bool = False,
                            use_wav2lip: bool = True,
                            progress_callback=None) -> str | None:

    downloaded_temp = None
    tmp_wav         = None
    video_path      = None

    try:
        if is_url(video_path_or_url):
            url = video_path_or_url.strip()
            print(f"[INFO] Input URL: {url}")
            video_path = download_video_from_url(url, progress_callback=progress_callback)
            if not video_path:
                return None
            downloaded_temp = video_path
        else:
            video_path = os.path.abspath(video_path_or_url)
            if not os.path.exists(video_path):
                print(f"[ERROR] Local file not found: {video_path}")
                if progress_callback:
                    progress_callback(stage='upload', status='error', progress=0,
                                      message=f'File not found: {video_path}')
                return None
            if progress_callback:
                progress_callback(stage='upload', status='completed', progress=10,
                                  message='Video loaded successfully')
            print(f"[INFO] Local video: {video_path} ({os.path.getsize(video_path) // 1024} KB)")

        tmp_wav = os.path.join(
            OUTPUT_DIR,
            f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        if not extract_audio(video_path, tmp_wav, progress_callback):
            return None

        transcription = transcribe_long_audio(tmp_wav, chunk_seconds=30,
                                              progress_callback=progress_callback)
        if not transcription.strip():
            if progress_callback:
                progress_callback(stage='transcribe', status='error', progress=0,
                                  message='Transcription failed — no speech found')
            return None

        if use_diarization and HF_API_TOKEN and PYANNOTE_AVAILABLE:
            diar           = run_diarization(tmp_wav, HF_API_TOKEN, progress_callback)
            translated     = translate_text_chunked(transcription, target_lang=target_lang,
                                                    progress_callback=progress_callback)
            speaker_tagged = assign_sentences_to_speakers(translated, diar)
        else:
            translated     = translate_text_chunked(transcription, target_lang=target_lang,
                                                    progress_callback=progress_callback)
            speaker_tagged = assign_sentences_to_speakers(translated, None)

        dubbed_audio = synthesize_speech_character_wise(
            speaker_tagged, target_lang=target_lang,
            progress_callback=progress_callback
        )

        final_video = os.path.join(
            OUTPUT_DIR,
            f"dubbed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )

        if use_wav2lip:
            final_video = run_wav2lip(video_path, dubbed_audio, final_video, progress_callback)
        else:
            final_video = merge_audio_video(video_path, dubbed_audio, final_video, progress_callback)

        if not final_video:
            return None

        print(f"\n[SUCCESS] Dubbed video saved: {final_video}")
        return final_video

    except Exception as e:
        import traceback
        print(f"[ERROR] Pipeline failed: {e}")
        traceback.print_exc()
        if progress_callback:
            progress_callback(stage='extract', status='error', progress=0,
                              message=f'Pipeline error: {e}')
        return None

    finally:
        for path in [tmp_wav, downloaded_temp]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
                    print(f"[INFO] Cleaned up: {path}")
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python video_dubbing.py <video_or_url> [lang] [diarization] [wav2lip]\n\n"
            "Examples:\n"
            "  python video_dubbing.py input.mp4 ur false true\n"
            "  python video_dubbing.py https://youtu.be/xxx ur false true\n"
            "  python video_dubbing.py input.mp4 hi false false\n"
        )
        sys.exit(1)

    r = process_video_pipeline(
        video_path_or_url = sys.argv[1],
        target_lang       = sys.argv[2] if len(sys.argv) > 2 else "en",
        use_diarization   = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False,
        use_wav2lip       = sys.argv[4].lower() != "false" if len(sys.argv) > 4 else True,
    )
    print(f"\n{'OK Final: ' + r if r else 'FAILED — check logs above.'}")
    sys.exit(0 if r else 1)