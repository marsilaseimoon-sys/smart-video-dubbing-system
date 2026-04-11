from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import re
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import threading

# Import the video dubbing backend
from video_dubbing import (
    process_video_pipeline,
    download_video_from_url,
    is_url,
    is_youtube_url,
    OUTPUT_DIR,
    HF_API_TOKEN,
)

app = Flask("V_Dub",
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2 GB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# In-memory job store
processing_status = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ═══════════════════════════════════════════════════════════════
# PAGE ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/studio')
def studio():
    return render_template('studio.html')

@app.route('/features')
def features():
    return render_template('feature.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# ═══════════════════════════════════════════════════════════════
# API — FILE UPLOAD
# ═══════════════════════════════════════════════════════════════

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video file upload from disk."""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400

        filename  = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename  = f"{timestamp}_{filename}"
        filepath  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        return jsonify({
            'success':  True,
            'filename': filename,
            'filepath': filepath,
            'message':  'Video uploaded successfully'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# API — URL LOADER
# ═══════════════════════════════════════════════════════════════

@app.route('/api/load_url', methods=['POST'])
def load_url():
    """
    Validate a YouTube / direct-video URL from the frontend.
    The actual download happens inside /api/process.
    Here we just verify the URL looks valid and return confirmation.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        url = (data.get('url') or '').strip()
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400

        if not re.match(r'^https?://', url):
            return jsonify({
                'success': False,
                'error': 'Invalid URL — must start with http:// or https://'
            }), 400

        source_type = 'youtube' if is_youtube_url(url) else 'direct'

        return jsonify({
            'success':     True,
            'url':         url,
            'source_type': source_type,
            'filepath':    url,   # frontend stores this as uploadedFilePath
            'message':     f'{"YouTube" if source_type == "youtube" else "Direct"} URL validated successfully'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# API — PROCESS  (accepts file path OR URL)
# ═══════════════════════════════════════════════════════════════

@app.route('/api/process', methods=['POST'])
def process_video():
    """
    Start video dubbing.

    Accepts EITHER:
      { "filepath": "/uploads/video.mp4", "language": "ur", "diarization": false }
    OR:
      { "url": "https://youtu.be/...",     "language": "ur", "diarization": false }

    A URL passed under the "filepath" key is also handled transparently,
    which is how the frontend sends it after /api/load_url.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        language        = data.get('language', 'en')
        use_diarization = data.get('diarization', False)

        # ── Determine input source ──────────────────────────────────────
        raw_url      = (data.get('url')      or '').strip()
        raw_filepath = (data.get('filepath') or '').strip()

        # Priority: explicit "url" key → "filepath" that looks like URL → local path
        if raw_url and re.match(r'^https?://', raw_url):
            video_input  = raw_url
            input_is_url = True

        elif raw_filepath and re.match(r'^https?://', raw_filepath):
            # Frontend sent the URL stored as filepath (expected behaviour after /api/load_url)
            video_input  = raw_filepath
            input_is_url = True

        elif raw_filepath and os.path.exists(raw_filepath):
            video_input  = raw_filepath
            input_is_url = False

        else:
            return jsonify({
                'success': False,
                'error':   'Provide a valid "url" or an existing local "filepath"'
            }), 400

        # ── Create job ──────────────────────────────────────────────────
        job_id = f"job_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        if input_is_url:
            initial = {
                'stage':        'download',
                'stage_status': 'active',
                'progress':     3,
                'message':      'Preparing to download video…',
            }
        else:
            initial = {
                'stage':        'upload',
                'stage_status': 'completed',
                'progress':     10,
                'message':      'Starting video dubbing process…',
            }

        processing_status[job_id] = {
            'status': 'processing',
            **initial,
        }

        # ── Progress callback ───────────────────────────────────────────
        def update_progress(stage, status, progress, message):
            processing_status[job_id].update({
                'status':       'processing',
                'stage':        stage,
                'stage_status': status,
                'progress':     progress,
                'message':      message,
            })

        # ── Background worker ───────────────────────────────────────────
        def process_async():
            try:
                result = process_video_pipeline(
                    video_input,
                    target_lang=language,
                    use_diarization=use_diarization,
                    progress_callback=update_progress,
                )

                if result and os.path.exists(result):
                    processing_status[job_id] = {
                        'status':       'completed',
                        'stage':        'merge',
                        'stage_status': 'completed',
                        'progress':     100,
                        'message':      'Video dubbing completed successfully!',
                        'output_path':  result,
                        'download_url': f'/api/download/{os.path.basename(result)}',
                        'file_size':    os.path.getsize(result),
                    }
                else:
                    processing_status[job_id] = {
                        'status':       'failed',
                        'stage':        processing_status[job_id].get('stage', 'extract'),
                        'stage_status': 'error',
                        'progress':     0,
                        'message':      'Video processing failed — check server logs.',
                    }

            except Exception as exc:
                processing_status[job_id] = {
                    'status':       'failed',
                    'stage':        processing_status[job_id].get('stage', 'extract'),
                    'stage_status': 'error',
                    'progress':     0,
                    'message':      f'Error: {exc}',
                }

        thread = threading.Thread(target=process_async, daemon=True)
        thread.start()

        return jsonify({
            'success':    True,
            'job_id':     job_id,
            'message':    'Processing started',
            'input_type': 'url' if input_is_url else 'file',
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# API — STATUS / CANCEL / DOWNLOAD / MISC
# ═══════════════════════════════════════════════════════════════

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    if job_id not in processing_status:
        return jsonify({'success': False, 'error': 'Job not found'}), 404
    return jsonify({'success': True, 'job': processing_status[job_id]})


@app.route('/api/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    if job_id in processing_status:
        processing_status[job_id]['status'] = 'cancelled'
        return jsonify({'success': True, 'message': 'Job cancelled'})
    return jsonify({'success': False, 'error': 'Job not found'}), 404


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        return send_file(filepath, as_attachment=True, download_name=f"dubbed_{filename}")
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/check-diarization', methods=['GET'])
def check_diarization():
    available = bool(HF_API_TOKEN and HF_API_TOKEN.strip())
    return jsonify({'success': True, 'available': available})


@app.route('/api/languages', methods=['GET'])
def get_languages():
    languages = [
        {'code': 'en',    'name': 'English',    'flag': '🇬🇧'},
        {'code': 'ur',    'name': 'Urdu',        'flag': '🇵🇰'},
        {'code': 'es',    'name': 'Spanish',     'flag': '🇪🇸'},
        {'code': 'fr',    'name': 'French',      'flag': '🇫🇷'},
        {'code': 'de',    'name': 'German',      'flag': '🇩🇪'},
        {'code': 'it',    'name': 'Italian',     'flag': '🇮🇹'},
        {'code': 'pt',    'name': 'Portuguese',  'flag': '🇵🇹'},
        {'code': 'ja',    'name': 'Japanese',    'flag': '🇯🇵'},
        {'code': 'ko',    'name': 'Korean',      'flag': '🇰🇷'},
        {'code': 'ar',    'name': 'Arabic',      'flag': '🇸🇦'},
        {'code': 'hi',    'name': 'Hindi',       'flag': '🇮🇳'},
        {'code': 'zh-cn', 'name': 'Chinese',     'flag': '🇨🇳'},
        {'code': 'tr',    'name': 'Turkish',     'flag': '🇹🇷'},
    ]
    return jsonify({'success': True, 'languages': languages})


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import webbrowser
    from threading import Timer

    print("=" * 50)
    print("DubStudio Server Starting...")
    print("=" * 50)
    print(f"Upload folder : {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Output folder : {os.path.abspath(OUTPUT_DIR)}")
    print("Server running on http://localhost:5000")
    print("=" * 50)

    Timer(1, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)