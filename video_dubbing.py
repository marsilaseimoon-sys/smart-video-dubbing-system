import os, sys, subprocess, tempfile, math, re, asyncio
from datetime import datetime

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_DIR      = os.path.join(BASE_DIR, "Wav2Lip")
OUTPUT_DIR       = os.path.join(BASE_DIR, "dubbed_videos")
HF_API_TOKEN     = None
COOKIES_TXT_PATH = os.path.join(BASE_DIR, "cookies.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    print("[WARN] edge_tts not available — pip install edge-tts")
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
    print("[INFO] pyannote not available:", e)
    PYANNOTE_AVAILABLE = False

try:
    from pytubefix import YouTube
    PYTUBEFIX_AVAILABLE = True
    print("[INFO] pytubefix loaded.")
except Exception as e:
    print("[WARN] pytubefix not available:", e)
    print("[HINT] Run: pip install pytubefix")
    PYTUBEFIX_AVAILABLE = False

EDGE_LANG_VOICES = {
    "ur":    ("ur-PK-AsadNeural",    "ur-PK-UzmaNeural"),
    "en":    ("en-US-GuyNeural",     "en-US-JennyNeural"),
    "ar":    ("ar-SA-HamedNeural",   "ar-SA-ZariyahNeural"),
    "hi":    ("hi-IN-MadhurNeural",  "hi-IN-SwaraNeural"),
    "fr":    ("fr-FR-HenriNeural",   "fr-FR-DeniseNeural"),
    "de":    ("de-DE-ConradNeural",  "de-DE-KatjaNeural"),
    "es":    ("es-ES-AlvaroNeural",  "es-ES-ElviraNeural"),
    "it":    ("it-IT-DiegoNeural",   "it-IT-ElsaNeural"),
    "pt":    ("pt-BR-AntonioNeural", "pt-BR-FranciscaNeural"),
    "ja":    ("ja-JP-KeitaNeural",   "ja-JP-NanamiNeural"),
    "ko":    ("ko-KR-InJoonNeural",  "ko-KR-SunHiNeural"),
    "zh-cn": ("zh-CN-YunxiNeural",   "zh-CN-XiaoxiaoNeural"),
    "tr":    ("tr-TR-AhmetNeural",   "tr-TR-EmelNeural"),
}
CHARACTER_GENDER = {"Ali":0,"Ahmed":0,"Sara":1,"Zoya":1,"Child":1}
DEFAULT_GENDER   = 0

def get_edge_voice(label, target_lang):
    voices = EDGE_LANG_VOICES.get(target_lang.lower(), EDGE_LANG_VOICES["en"])
    lc     = label.strip().rstrip(":").strip()
    if lc in CHARACTER_GENDER: return voices[CHARACTER_GENDER[lc]]
    if lc.upper().startswith("SPEAKER"):
        d = ''.join(filter(str.isdigit, lc))
        return voices[(int(d) if d else 0) % 2]
    ll = lc.lower()
    if any(k in ll for k in ("female","woman","sara","zoya")): return voices[1]
    if any(k in ll for k in ("male","man","ali","ahmed")):     return voices[0]
    return voices[DEFAULT_GENDER]

async def _edge_synth(text, voice, out):
    await edge_tts.Communicate(text, voice).save(out)

def synthesize_with_edge_tts(text, voice, out):
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as p:
                    p.submit(asyncio.run, _edge_synth(text,voice,out)).result(timeout=60)
            else:
                loop.run_until_complete(_edge_synth(text,voice,out))
        except RuntimeError:
            asyncio.run(_edge_synth(text,voice,out))
        return os.path.exists(out) and os.path.getsize(out) > 100
    except Exception as e:
        print(f"[ERROR] Edge TTS: {e}")
        return False

def is_url(s): return bool(re.match(r'^https?://', s.strip()))
def is_youtube_url(u): return bool(re.search(r'(youtube\.com|youtu\.be)', u, re.I))

# ══ DOWNLOADER ═══════════════════════════════════════════════

def _install_pkg(pkg):
    try:
        r = subprocess.run([sys.executable,"-m","pip","install",pkg,"-q"],
                           capture_output=True,text=True,timeout=60)
        return r.returncode == 0
    except Exception: return False

def _download_pytubefix(url, progress_callback=None):
    global PYTUBEFIX_AVAILABLE
    if not PYTUBEFIX_AVAILABLE:
        if _install_pkg("pytubefix"):
            try:
                from pytubefix import YouTube as YT2
                globals()['YouTube'] = YT2
                PYTUBEFIX_AVAILABLE = True
            except Exception: return None
        else: return None
    try:
        from pytubefix import YouTube as YT
        print(f"[INFO] pytubefix connecting: {url}")
        yt = YT(url, use_oauth=False, allow_oauth_cache=True)
        print(f"[INFO] Title: {yt.title}  |  Length: {yt.length}s")

        # Try progressive (video+audio together) first
        stream = yt.streams.filter(progressive=True,file_extension="mp4")\
                            .order_by("resolution").last()

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        if stream:
            if progress_callback:
                progress_callback(stage='download',status='active',progress=5,
                                  message='Downloading via pytubefix...')
            out = stream.download(output_path=OUTPUT_DIR, filename=f"downloaded_{ts}.mp4")
            if out and os.path.exists(out) and os.path.getsize(out)>10000:
                print(f"[INFO] pytubefix progressive OK: {out}")
                return out

        # Adaptive: separate video + audio then merge
        vs = yt.streams.filter(adaptive=True,file_extension="mp4",only_video=True)\
                        .order_by("resolution").last()
        as_ = yt.streams.filter(adaptive=True,only_audio=True)\
                         .order_by("abr").last()
        if not vs or not as_:
            print("[WARN] pytubefix: no adaptive streams"); return None

        if progress_callback:
            progress_callback(stage='download',status='active',progress=4,
                              message='Downloading video stream...')
        vp = vs.download(output_path=OUTPUT_DIR, filename=f"_vtmp_{ts}.mp4")
        if progress_callback:
            progress_callback(stage='download',status='active',progress=7,
                              message='Downloading audio stream...')
        ap = as_.download(output_path=OUTPUT_DIR, filename=f"_atmp_{ts}.mp4")

        out = os.path.join(OUTPUT_DIR, f"downloaded_{ts}.mp4")
        mr = subprocess.run(["ffmpeg","-y","-i",vp,"-i",ap,
                              "-c:v","copy","-c:a","aac","-shortest",out],
                            capture_output=True,text=True,timeout=300)
        for p in [vp,ap]:
            try: os.remove(p)
            except: pass
        if mr.returncode==0 and os.path.exists(out):
            print(f"[INFO] pytubefix adaptive OK: {out}")
            return out
        print(f"[WARN] pytubefix adaptive merge failed: {mr.stderr[-300:]}")
        return None
    except Exception as e:
        print(f"[WARN] pytubefix: {e}")
        return None

def _check_ytdlp():
    try:
        subprocess.run(["yt-dlp","--version"],stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,check=True)
        return True
    except: return False

def _get_cookies_args():
    if os.path.isfile(COOKIES_TXT_PATH):
        print(f"[INFO] cookies.txt: {COOKIES_TXT_PATH}")
        return ["--cookies", COOKIES_TXT_PATH]
    for b in ["chrome","firefox","edge","brave","opera","chromium"]:
        try:
            r = subprocess.run(["yt-dlp","--cookies-from-browser",b,"--simulate","--quiet",
                                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
                               capture_output=True,text=True,timeout=10)
            if r.returncode==0:
                print(f"[INFO] Browser cookies: {b}")
                return ["--cookies-from-browser",b]
        except: continue
    print("[INFO] No cookies found.")
    return []

def _run_ytdlp_cmd(cmd, ts, progress_callback=None):
    try:
        proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,text=True,bufsize=1)
        dl_file = None
        for line in proc.stdout:
            line = line.strip()
            if line: print(f"[yt-dlp] {line}")
            m = re.search(r'(\d+\.?\d*)%', line)
            if m and progress_callback:
                pct = float(m.group(1))
                progress_callback(stage='download',status='active',
                                  progress=5+int(pct*0.04),
                                  message=f'Downloading... {pct:.1f}%')
            if '[download] Destination:' in line:
                dl_file = line.split('[download] Destination:')[-1].strip()
            if 'Merging formats into' in line:
                dl_file = line.split('Merging formats into')[-1].strip().strip('"')
        proc.wait()
        if proc.returncode != 0: return None
        if not dl_file or not os.path.exists(dl_file):
            import glob
            fs = sorted(glob.glob(os.path.join(OUTPUT_DIR,f"downloaded_{ts}.*")),
                        key=os.path.getmtime,reverse=True)
            dl_file = fs[0] if fs else None
        return dl_file if dl_file and os.path.exists(dl_file) else None
    except Exception as e:
        print(f"[ERROR] yt-dlp run: {e}")
        return None

def _ytdlp_strategies(out_tmpl, cookie_args, url):
    base = ["yt-dlp","--no-playlist","--merge-output-format","mp4",
            "--no-check-certificate","--geo-bypass",
            "--retries","3","--fragment-retries","3",
            "--socket-timeout","30","--no-warnings","--progress",
            "--output",out_tmpl]
    return [
        ("android_creator (best 403 bypass)",
         base+["--extractor-args","youtube:player_client=android_creator",
               "-f","bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best"]
         +cookie_args+[url]),
        ("ios client",
         base+["--extractor-args","youtube:player_client=ios",
               "-f","best[ext=mp4]/best"]+cookie_args+[url]),
        ("android_embedded",
         base+["--extractor-args","youtube:player_client=android_embedded",
               "-f","best[ext=mp4]/best"]+cookie_args+[url]),
        ("web + cookies",
         base+["--extractor-args","youtube:player_client=web",
               "-f","best[ext=mp4]/best"]+cookie_args+[url]),
        ("bare fallback",
         ["yt-dlp","--no-playlist","--merge-output-format","mp4",
          "--no-check-certificate","--geo-bypass","--retries","2",
          "--no-warnings","--progress","-f","best","--output",out_tmpl,url]),
    ]

def _download_ytdlp(url, progress_callback=None):
    if not _check_ytdlp():
        print("[INFO] Installing yt-dlp...")
        try:
            subprocess.run([sys.executable,"-m","pip","install","yt-dlp","-q","--user"],
                           check=True,timeout=60)
        except Exception as e:
            print(f"[ERROR] yt-dlp install: {e}"); return None
    # Quick update
    try:
        subprocess.run([sys.executable,"-m","pip","install","--upgrade","yt-dlp","-q"],
                       capture_output=True,text=True,timeout=30)
    except: pass

    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_tmpl = os.path.join(OUTPUT_DIR, f"downloaded_{ts}.%(ext)s")
    cookies  = _get_cookies_args()
    strats   = _ytdlp_strategies(out_tmpl, cookies, url)

    for i,(desc,cmd) in enumerate(strats,1):
        if progress_callback:
            progress_callback(stage='download',status='active',progress=5,
                              message=f'yt-dlp attempt {i}: {desc}...')
        print(f"\n[INFO] yt-dlp attempt {i}/{len(strats)}: {desc}")
        r = _run_ytdlp_cmd(cmd, ts, progress_callback)
        if r: return r
        print(f"[WARN] yt-dlp attempt {i} failed.")
    return None

def download_video_from_url(url, progress_callback=None):
    if progress_callback:
        progress_callback(stage='download',status='active',progress=2,
                          message='Starting download...')
    print(f"[INFO] Downloading: {url}")

    if is_youtube_url(url):
        # 1. pytubefix (primary — no 403)
        print("[INFO] Trying pytubefix...")
        r = _download_pytubefix(url, progress_callback)
        if r:
            mb = os.path.getsize(r)/(1024*1024)
            print(f"[SUCCESS] pytubefix OK: {r} ({mb:.1f} MB)")
            if progress_callback:
                progress_callback(stage='download',status='completed',progress=10,
                                  message=f'Download complete ({mb:.1f} MB)')
            return r

        # 2. yt-dlp fallback with 5 strategies
        print("[INFO] pytubefix failed — trying yt-dlp...")
        r = _download_ytdlp(url, progress_callback)
        if r:
            mb = os.path.getsize(r)/(1024*1024)
            print(f"[SUCCESS] yt-dlp OK: {r} ({mb:.1f} MB)")
            if progress_callback:
                progress_callback(stage='download',status='completed',progress=10,
                                  message=f'Download complete ({mb:.1f} MB)')
            return r
    else:
        # Non-YouTube
        if not _check_ytdlp(): _install_pkg("yt-dlp")
        ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_tmpl = os.path.join(OUTPUT_DIR, f"downloaded_{ts}.%(ext)s")
        cmd = ["yt-dlp","--no-playlist","-f","best[ext=mp4]/best",
               "--merge-output-format","mp4","--no-check-certificate",
               "--retries","3","--no-warnings","--progress",
               "--output",out_tmpl,url]
        r = _run_ytdlp_cmd(cmd, ts, progress_callback)
        if r:
            mb = os.path.getsize(r)/(1024*1024)
            if progress_callback:
                progress_callback(stage='download',status='completed',progress=10,
                                  message=f'Download complete ({mb:.1f} MB)')
            return r

    # All failed
    print("\n"+"="*58)
    print("[ERROR] All download methods failed.")
    print()
    print("  QUICK FIX OPTIONS:")
    print("  1. pip install pytubefix   (recommended, then retry)")
    print("  2. Download manually from savefrom.net or y2mate.com")
    print("     and upload the .mp4 file directly in the app")
    print("  3. Add cookies.txt:")
    print("     - Chrome: install 'Get cookies.txt LOCALLY' extension")
    print("     - Export cookies from youtube.com (while logged in)")
    print(f"     - Save to: {COOKIES_TXT_PATH}")
    print("="*58)

    if progress_callback:
        progress_callback(stage='download',status='error',progress=0,
                          message='Download failed. Fix: pip install pytubefix  OR  upload .mp4 directly. See console.')
    return None

# ══ AUDIO EXTRACTION ══════════════════════════════════════════

def extract_audio(video_path, out_wav, progress_callback=None):
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}"); return None
    cmd = ["ffmpeg","-y","-i",video_path,"-vn","-acodec","pcm_s16le",
           "-ar","16000","-ac","1",out_wav]
    try:
        if progress_callback:
            progress_callback(stage='extract',status='active',progress=15,
                              message='Extracting audio...')
        r = subprocess.run(cmd,capture_output=True,text=True)
        if r.returncode!=0:
            print(f"[ERROR] FFmpeg: {r.stderr}")
            if progress_callback:
                progress_callback(stage='extract',status='error',progress=0,
                                  message='Audio extraction failed')
            return None
        if not os.path.exists(out_wav) or os.path.getsize(out_wav)<100:
            print("[ERROR] WAV empty."); return None
        print(f"[INFO] Audio: {out_wav} ({os.path.getsize(out_wav)//1024} KB)")
        if progress_callback:
            progress_callback(stage='extract',status='completed',progress=20,
                              message='Audio extracted')
        return out_wav
    except Exception as e:
        print(f"[ERROR] extract_audio: {e}"); return None

# ══ TRANSCRIPTION ═════════════════════════════════════════════

def transcribe_long_audio(audio_path, chunk_seconds=30, progress_callback=None):
    if not WHISPER_AVAILABLE or WHISPER_MODEL is None or AudioSegment is None:
        print("[WARN] Whisper/pydub unavailable."); return ""
    if not os.path.exists(audio_path):
        print(f"[ERROR] Audio missing: {audio_path}"); return ""
    if progress_callback:
        progress_callback(stage='transcribe',status='active',progress=25,
                          message='Starting transcription...')
    audio   = AudioSegment.from_file(audio_path)
    dur_ms  = len(audio)
    chunk_ms= chunk_seconds*1000
    parts   = math.ceil(dur_ms/chunk_ms)
    txts    = []
    for i in range(parts):
        chunk = audio[i*chunk_ms:min((i+1)*chunk_ms,dur_ms)]
        with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as tf:
            cp = tf.name
        chunk.export(cp,format="wav")
        try:
            txts.append(WHISPER_MODEL.transcribe(cp).get("text","").strip())
        except Exception as e:
            print(f"[WARN] Whisper chunk {i}: {e}"); txts.append("")
        finally:
            try: os.remove(cp)
            except: pass
        if progress_callback:
            progress_callback(stage='transcribe',status='active',
                              progress=25+int((i+1)/parts*15),
                              message=f'Transcribing... ({i+1}/{parts})')
    if progress_callback:
        progress_callback(stage='transcribe',status='completed',progress=40,
                          message='Transcription done')
    full = " ".join(t for t in txts if t)
    print(f"[INFO] Transcription ({len(full)} chars): {full[:100]}...")
    return full

# ══ TRANSLATION ═══════════════════════════════════════════════

def translate_text_chunked(text, target_lang="ur", max_chunk_chars=1500, progress_callback=None):
    if not GoogleTranslator or not text.strip(): return text
    sents = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks, cur = [], ""
    for s in sents:
        if len(cur)+len(s)+1<=max_chunk_chars: cur=(cur+" "+s).strip()
        else:
            if cur: chunks.append(cur)
            cur=s
    if cur: chunks.append(cur)
    parts, total = [], len(chunks)
    for i,ch in enumerate(chunks):
        try: t=GoogleTranslator(source='auto',target=target_lang).translate(ch)
        except Exception as e: print(f"[WARN] Trans {i}: {e}"); t=ch
        parts.append(t or ch)
        if progress_callback:
            progress_callback(stage='translate',status='active',
                              progress=45+int((i+1)/total*15),
                              message=f'Translating... ({i+1}/{total})')
    if progress_callback:
        progress_callback(stage='translate',status='completed',progress=60,
                          message='Translation done')
    return " ".join(parts)

# ══ DIARIZATION ═══════════════════════════════════════════════

def run_diarization(audio_path, hf_token, progress_callback=None):
    if not PYANNOTE_AVAILABLE or not hf_token: return None
    try:
        if progress_callback:
            progress_callback(stage='transcribe',status='active',progress=35,
                              message='Diarization...')
        pl  = PyannotePipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                               use_auth_token=hf_token)
        dia = pl(audio_path)
        return [(sp,t.start,t.end) for t,_,sp in dia.itertracks(yield_label=True)]
    except Exception as e:
        print(f"[ERROR] Diarization: {e}"); return None

def assign_sentences_to_speakers(text, segs=None):
    sents = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+',text) if s.strip()]
    if not sents: return text
    if segs:
        order=[]
        for sp,_,__ in segs:
            if sp not in order: order.append(sp)
        order=order or ["SPEAKER_00"]
        return "\n".join(f"{order[i%len(order)]}: {s}" for i,s in enumerate(sents))
    chars=["Ali","Sara","Ahmed"]
    return "\n".join(f"{chars[i%len(chars)]}: {s}" for i,s in enumerate(sents))

# ══ TTS ═══════════════════════════════════════════════════════

def synthesize_speech_character_wise(text, target_lang="en", progress_callback=None):
    if AudioSegment is None: raise RuntimeError("pip install pydub")
    if not text.strip(): raise RuntimeError("Empty text.")
    if progress_callback:
        progress_callback(stage='synthesize',status='active',progress=65,
                          message='Starting TTS...')
    lines   = [l.strip() for l in text.split("\n") if l.strip()]
    outputs = []
    total   = len(lines)
    for idx,line in enumerate(lines):
        if ":" in line:
            label,content = line.split(":",1)
            label,content = label.strip(),content.strip()
        else:
            label,content = "Ali",line
        if not content: continue
        tmp = os.path.join(OUTPUT_DIR,f"tts_{idx}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.mp3")
        ok  = False
        if EDGE_TTS_AVAILABLE:
            voice = get_edge_voice(label,target_lang)
            ok    = synthesize_with_edge_tts(content,voice,tmp)
            if ok: print(f"[INFO] Edge TTS OK [{voice}] line {idx}")
            else:  print(f"[WARN] Edge TTS failed line {idx}")
        if not ok and gTTS is not None:
            try:
                gTTS(text=content,lang=target_lang.split("-")[0]).save(tmp)
                ok=True
                print(f"[INFO] gTTS OK line {idx}")
            except Exception as e: print(f"[ERROR] gTTS {idx}: {e}")
        if ok: outputs.append(tmp)
        else:  print(f"[ERROR] Skipping line {idx}")
        if progress_callback and total>0:
            progress_callback(stage='synthesize',status='active',
                              progress=65+int((idx+1)/total*20),
                              message=f'Synthesizing... ({idx+1}/{total})')
    if not outputs: raise RuntimeError("No TTS output.")
    combined=None
    for p in outputs:
        try:
            seg=AudioSegment.from_file(p)
            combined=seg if combined is None else combined+seg
        except Exception as e: print(f"[WARN] Load {p}: {e}")
    if combined is None: raise RuntimeError("All segments failed.")
    final=os.path.join(OUTPUT_DIR,f"dubbed_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
    combined.export(final,format="wav")
    for p in outputs:
        try: os.remove(p)
        except: pass
    size=os.path.getsize(final)
    print(f"[INFO] Dubbed audio: {final} ({size//1024} KB)")
    if size<1000: raise RuntimeError(f"Audio too small ({size} bytes).")
    if progress_callback:
        progress_callback(stage='synthesize',status='completed',progress=85,
                          message='TTS done')
    return final

# ══ WAV2LIP ═══════════════════════════════════════════════════

def run_wav2lip(video_path, audio_path, output_path, progress_callback=None):
    av=os.path.abspath(video_path); aa=os.path.abspath(audio_path)
    ao=os.path.abspath(output_path)
    ck=os.path.join(WAV2LIP_DIR,"Wav2Lip.pth")
    inf=os.path.join(WAV2LIP_DIR,"inference.py")
    miss=[]
    for lbl,p in [("Wav2Lip dir",WAV2LIP_DIR),("Wav2Lip.pth",ck),
                  ("inference.py",inf),("Video",av),("Audio",aa)]:
        if not os.path.exists(p): miss.append(f"{lbl}: {p}")
    if miss:
        print("[ERROR] Wav2Lip missing:")
        for m in miss: print(f"   - {m}")
        print("[INFO] Falling back to FFmpeg.")
        return merge_audio_video(av,aa,ao,progress_callback)
    try:
        if progress_callback:
            progress_callback(stage='merge',status='active',progress=90,
                              message='Running Wav2Lip...')
        env=os.environ.copy()
        pp=env.get("PYTHONPATH","")
        env["PYTHONPATH"]=WAV2LIP_DIR+(os.pathsep+pp if pp else "")
        r=subprocess.run([sys.executable,"inference.py",
                          "--checkpoint_path",ck,"--face",av,"--audio",aa,"--outfile",ao],
                         cwd=WAV2LIP_DIR,env=env,
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                         text=True,timeout=600)
        for line in (r.stdout or "").strip().splitlines():
            print(f"[Wav2Lip] {line}")
        if r.returncode==0 and os.path.exists(ao) and os.path.getsize(ao)>10000:
            print(f"[SUCCESS] Wav2Lip: {ao}")
            if progress_callback:
                progress_callback(stage='merge',status='completed',progress=100,
                                  message='Lip-sync done!')
            return ao
        print(f"[WARN] Wav2Lip code {r.returncode} — FFmpeg fallback.")
        return merge_audio_video(av,aa,ao,progress_callback)
    except subprocess.TimeoutExpired:
        print("[ERROR] Wav2Lip timeout.")
        return merge_audio_video(av,aa,ao,progress_callback)
    except Exception as e:
        print(f"[ERROR] Wav2Lip: {e}")
        return merge_audio_video(av,aa,ao,progress_callback)

# ══ FFMPEG MERGE ══════════════════════════════════════════════

def merge_audio_video(video_path, dubbed_audio, output_name=None, progress_callback=None):
    if output_name is None:
        output_name=os.path.join(OUTPUT_DIR,f"dubbed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    video_path=os.path.abspath(video_path)
    dubbed_audio=os.path.abspath(dubbed_audio)
    output_name=os.path.abspath(output_name)
    print(f"\n[MERGE] video={video_path} | audio={dubbed_audio}")
    if not os.path.exists(video_path): print("[ERROR] Video missing!"); return None
    if not os.path.exists(dubbed_audio) or os.path.getsize(dubbed_audio)<1000:
        print("[ERROR] Audio missing/small!"); return None
    if progress_callback:
        progress_callback(stage='merge',status='active',progress=90,
                          message='Merging audio & video...')
    os.makedirs(os.path.dirname(output_name),exist_ok=True)
    # Attempt 1: copy
    r1=subprocess.run(["ffmpeg","-y","-i",video_path,"-i",dubbed_audio,
                        "-map","0:v:0","-map","1:a:0","-c:v","copy",
                        "-c:a","aac","-b:a","192k","-shortest",output_name],
                       capture_output=True,text=True,timeout=300)
    if r1.returncode==0 and os.path.exists(output_name) and os.path.getsize(output_name)>10000:
        mb=os.path.getsize(output_name)/(1024*1024)
        print(f"[SUCCESS] Merged: {output_name} ({mb:.1f} MB)")
        if progress_callback:
            progress_callback(stage='merge',status='completed',progress=100,
                              message='Dubbing complete!')
        return output_name
    print("[WARN] Copy failed — re-encoding...")
    # Attempt 2: re-encode
    r2=subprocess.run(["ffmpeg","-y","-i",video_path,"-i",dubbed_audio,
                        "-map","0:v:0","-map","1:a:0",
                        "-c:v","libx264","-crf","23","-preset","fast",
                        "-c:a","aac","-b:a","192k","-shortest",output_name],
                       capture_output=True,text=True,timeout=600)
    if r2.returncode==0 and os.path.exists(output_name) and os.path.getsize(output_name)>10000:
        mb=os.path.getsize(output_name)/(1024*1024)
        print(f"[SUCCESS] Re-encoded: {output_name} ({mb:.1f} MB)")
        if progress_callback:
            progress_callback(stage='merge',status='completed',progress=100,
                              message='Dubbing complete!')
        return output_name
    print(f"[ERROR] Both merges failed.\n{r2.stderr[-1500:]}")
    if progress_callback:
        progress_callback(stage='merge',status='error',progress=0,message='Merge failed')
    return None

# ══ MAIN PIPELINE ══════════════════════════════════════════════

def process_video_pipeline(video_path_or_url, target_lang="en",
                            use_diarization=False, use_wav2lip=True,
                            progress_callback=None):
    downloaded_temp=tmp_wav=video_path=None
    try:
        if is_url(video_path_or_url):
            video_path=download_video_from_url(video_path_or_url.strip(),
                                               progress_callback=progress_callback)
            if not video_path: return None
            downloaded_temp=video_path
        else:
            video_path=os.path.abspath(video_path_or_url)
            if not os.path.exists(video_path):
                print(f"[ERROR] File not found: {video_path}")
                if progress_callback:
                    progress_callback(stage='upload',status='error',progress=0,
                                      message=f'File not found: {video_path}')
                return None
            if progress_callback:
                progress_callback(stage='upload',status='completed',progress=10,
                                  message='Video loaded')
            print(f"[INFO] Local video: {video_path} ({os.path.getsize(video_path)//1024} KB)")

        tmp_wav=os.path.join(OUTPUT_DIR,f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        if not extract_audio(video_path,tmp_wav,progress_callback): return None

        transcription=transcribe_long_audio(tmp_wav,progress_callback=progress_callback)
        if not transcription.strip():
            if progress_callback:
                progress_callback(stage='transcribe',status='error',progress=0,
                                  message='No speech found')
            return None

        if use_diarization and HF_API_TOKEN and PYANNOTE_AVAILABLE:
            diar  =run_diarization(tmp_wav,HF_API_TOKEN,progress_callback)
            trans =translate_text_chunked(transcription,target_lang,progress_callback=progress_callback)
            tagged=assign_sentences_to_speakers(trans,diar)
        else:
            trans =translate_text_chunked(transcription,target_lang,progress_callback=progress_callback)
            tagged=assign_sentences_to_speakers(trans,None)

        dubbed_audio=synthesize_speech_character_wise(tagged,target_lang=target_lang,
                                                      progress_callback=progress_callback)
        final_video=os.path.join(OUTPUT_DIR,f"dubbed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

        if use_wav2lip:
            final_video=run_wav2lip(video_path,dubbed_audio,final_video,progress_callback)
        else:
            final_video=merge_audio_video(video_path,dubbed_audio,final_video,progress_callback)

        if not final_video: return None
        print(f"\n[SUCCESS] Final: {final_video}")
        return final_video

    except Exception as e:
        import traceback
        print(f"[ERROR] Pipeline: {e}")
        traceback.print_exc()
        if progress_callback:
            progress_callback(stage='extract',status='error',progress=0,
                              message=f'Pipeline error: {e}')
        return None
    finally:
        for p in [tmp_wav,downloaded_temp]:
            try:
                if p and os.path.exists(p):
                    os.remove(p); print(f"[INFO] Cleaned: {p}")
            except: pass

# ══ CLI ════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python video_dubbing.py <video_or_url> [lang] [diarization] [wav2lip]")
        sys.exit(1)
    r=process_video_pipeline(
        video_path_or_url=sys.argv[1],
        target_lang=sys.argv[2] if len(sys.argv)>2 else "en",
        use_diarization=sys.argv[3].lower()=="true" if len(sys.argv)>3 else False,
        use_wav2lip=sys.argv[4].lower()!="false" if len(sys.argv)>4 else True,
    )
    print(f"\n{'OK: '+r if r else 'FAILED — see logs.'}")
    sys.exit(0 if r else 1)