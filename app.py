"""
app.py
------
Main Flask application for HireScope AI — Resume Screening System.
Includes MongoDB, Auth, Async Processing, and Sentence Transformers.
"""

import os
import logging
import concurrent.futures
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify
)
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Load environment variables
load_dotenv()

# Cloudinary Configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)

from db import (
    create_user, authenticate_user, insert_candidate, 
    get_all_candidates, get_candidate_by_id, update_candidate_audio,
    update_candidate_audio_error, set_candidate_audio_processing,
    clear_all_candidates
)
from werkzeug.utils import secure_filename
from resume_parser import extract_text, clean_text
from skill_extractor import extract_all, SKILLS_LIST
from job_matcher import calculate_match_score, find_skill_gaps, rank_candidates
from audio_transcriber import transcribe_from_local_file

app = Flask(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Secret key
app.secret_key = os.getenv("SECRET_KEY")

# Hugging Face / Proxy Configuration
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Session Configuration for iframe compatibility (Hugging Face)
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_SAMESITE='None',
    SESSION_COOKIE_HTTPONLY=True,
)

# Optional: Initialize Google Generative AI if key is present
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception:
        pass

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER_RESUMES = os.path.join(BASE_DIR, "uploads", "resumes")
UPLOAD_FOLDER_AUDIO = os.path.join(BASE_DIR, "uploads", "audio")
os.makedirs(UPLOAD_FOLDER_RESUMES, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_AUDIO, exist_ok=True)

ALLOWED_RESUME_EXTENSIONS = {"pdf", "docx"}
ALLOWED_AUDIO_EXTENSIONS = {"mp3", "wav", "m4a", "flac", "ogg", "webm"}

# Thread pool for async audio transcription
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

# ── Authentication Helper ──
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = authenticate_user(email, password)
        if user:
            session["user_id"] = user["_id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            flash("Logged in successfully!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid email or password", "error")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        success, msg = create_user(username, email, password)
        if success:
            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))
        else:
            flash(msg, "error")
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))

@app.route("/")
@login_required
def index():
    candidates = get_all_candidates()
    # Calculate stats
    avg_score = 0
    if candidates:
        scores = [c.get("match_score", 0) for c in candidates]
        avg_score = round(sum(scores) / len(scores), 1)
    return render_template(
        "index.html",
        candidate_count=len(candidates),
        avg_score=avg_score,
        recent_candidates=candidates[:5]
    )

@app.route("/upload", methods=["POST"])
@login_required
def upload_resume():
    if "resume" not in request.files:
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    file = request.files["resume"]
    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename, ALLOWED_RESUME_EXTENSIONS):
        flash("Invalid file type. Please upload PDF or DOCX.", "error")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    if not filename:
        filename = "resume_file"
    filepath = os.path.join(UPLOAD_FOLDER_RESUMES, filename)
    file.save(filepath)

    raw_text = extract_text(filepath)
    if not raw_text.strip():
        flash("Could not extract text. Please ensure the PDF/DOCX is not just scanned images.", "error")
        return redirect(url_for("index"))

    cleaned_text = clean_text(raw_text)
    extracted_info = extract_all(cleaned_text)

    job_description = request.form.get("job_description", "").strip()
    match_score = 0.0
    skill_gaps = {"matched": [], "missing": []}
    jd_skills = []

    if job_description:
        from skill_extractor import extract_skills
        jd_skills = extract_skills(job_description)
        match_score = calculate_match_score(cleaned_text, job_description, extracted_info["skills"], jd_skills)
        skill_gaps = find_skill_gaps(extracted_info["skills"], job_description, SKILLS_LIST, jd_skills)

    # --- Cloudinary Upload Resume ---
    resume_url = ""
    try:
        cloudinary_response = cloudinary.uploader.upload(
            filepath,
            resource_type="auto",
            folder="resume_screener/resumes",
            use_filename=True,
            unique_filename=True,
        )
        # Simply use the secure_url provided by Cloudinary
        resume_url = cloudinary_response.get("secure_url", "")
    except Exception as e:
        logger.error("Cloudinary upload failed: %s", e)
        resume_url = ""

    # Clean up local file after processing
    try:
        os.remove(filepath)
    except Exception:
        pass

    # AI Summary using Google Gen AI (if configured)
    ai_summary = ""
    if GOOGLE_API_KEY:
        try:
            import google.generativeai as genai
            model = genai.GenerativeModel('gemini-flash-latest')
            prompt = f"Summarize this candidate in 2 to 3 short sentences emphasizing their top skills, experience, and education based on this resume text:\n{cleaned_text[:3000]}"
            response = model.generate_content(prompt)
            ai_summary = response.text.strip()
        except Exception as e:
            logger.error(f"Generative AI Error (Summary): {e}")

    candidate_name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()
    candidate_data = {
        "name": candidate_name,
        "filename": filename,
        "resume_url": resume_url,
        "resume_text": cleaned_text,
        "raw_text_preview": raw_text[:500],
        "ai_summary": ai_summary,
        "skills": extracted_info["skills"],
        "education": extracted_info["education"],
        "experience": extracted_info["experience"],
        "match_score": match_score,
        "skill_gaps": skill_gaps,
        "job_description": job_description,
        "audio_transcription": None,
        "uploaded_by": session["user_id"]
    }
    
    # Save to MongoDB
    candidate_id = insert_candidate(candidate_data)
    session["last_candidate_id"] = str(candidate_id)

    flash("Resume analyzed successfully!", "success")
    return redirect(url_for("results"))


def process_audio_local(candidate_id, local_audio_path, audio_url):
    """
    Process audio transcription from a LOCAL file (not URL).
    This avoids Cloudinary download issues entirely.
    The audio is saved locally first, transcribed, then cleaned up.
    """
    logger.info("Starting LOCAL transcription for candidate_id=%s, file=%s", candidate_id, local_audio_path)
    result = transcribe_from_local_file(local_audio_path)
    
    if result["success"]:
        update_candidate_audio(candidate_id, result["text"], result["language"], audio_url)
        logger.info("Transcription saved for candidate_id=%s", candidate_id)
    else:
        update_candidate_audio_error(candidate_id, result["error"], audio_url)
        logger.error("Transcription failed for candidate_id=%s: %s", candidate_id, result["error"])

    # Clean up local audio file after transcription
    try:
        if local_audio_path and os.path.exists(local_audio_path):
            os.remove(local_audio_path)
            logger.info("Cleaned up local audio: %s", local_audio_path)
    except Exception:
        pass


def _handle_transcription_future(future, candidate_id, audio_url):
    exc = future.exception()
    if exc is None:
        return
    error_msg = f"Background transcription crashed: {exc}"
    logger.exception("Unhandled transcription error for candidate_id=%s", candidate_id)
    update_candidate_audio_error(candidate_id, error_msg, audio_url)

@app.route("/upload_audio", methods=["POST"])
@login_required
def upload_audio():
    # Get candidate_id from form (sent from results page) or session
    candidate_id = request.form.get("candidate_id") or session.get("last_candidate_id")

    if not candidate_id:
        flash("Please upload and analyze a resume first before attaching audio.", "error")
        return redirect(url_for("index"))

    candidate = get_candidate_by_id(candidate_id)
    if not candidate:
        flash("Candidate not found. Please upload a resume first.", "error")
        return redirect(url_for("index"))

    if "audio" not in request.files:
        flash("No audio file selected.", "error")
        return redirect(url_for("results"))

    file = request.files["audio"]
    if file.filename == "":
        flash("No audio file selected.", "error")
        return redirect(url_for("results"))

    if not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
        flash("Invalid audio format. Supported: MP3, WAV, M4A, FLAC, OGG, WEBM", "error")
        return redirect(url_for("results"))

    filename = secure_filename(file.filename) or "audio_file"
    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        ext = ".mp3"
    
    # === KEY FIX: Save audio LOCALLY first, then transcribe from local file ===
    local_audio_path = os.path.join(UPLOAD_FOLDER_AUDIO, f"{candidate_id}_{filename}")
    file.save(local_audio_path)
    logger.info("Audio saved locally at: %s (%d bytes)", local_audio_path, os.path.getsize(local_audio_path))

    # Upload to Cloudinary for storage (non-blocking for transcription)
    audio_url = ""
    try:
        cloudinary_response = cloudinary.uploader.upload(
            local_audio_path,
            resource_type="video",
            folder="resume_screener/audio",
            public_id=f"{candidate_id}_{os.path.splitext(filename)[0]}",
            use_filename=False,
            overwrite=True,
        )
        audio_url = cloudinary_response.get("secure_url", "")
    except Exception as exc:
        logger.warning("Audio Cloudinary upload failed (will still transcribe locally): %s", exc)
        audio_url = ""  # Not critical — transcription uses local file

    set_candidate_audio_processing(candidate_id, audio_url)
    
    # === Transcribe from LOCAL file (not from Cloudinary URL) ===
    try:
        future = executor.submit(process_audio_local, candidate_id, local_audio_path, audio_url)
        future.add_done_callback(
            lambda f, cid=candidate_id, aurl=audio_url: _handle_transcription_future(f, cid, aurl)
        )
    except Exception as exc:
        error_msg = f"Failed to queue transcription task: {exc}"
        logger.exception(error_msg)
        update_candidate_audio_error(candidate_id, error_msg, audio_url)
        flash(error_msg, "error")
        return redirect(url_for("results"))

    session["last_candidate_id"] = str(candidate_id)
    session["awaiting_transcription_for"] = str(candidate["_id"])
    logger.info("Queued LOCAL transcription for candidate_id=%s", candidate_id)
    flash("Audio uploaded successfully! Transcription is processing in the background.", "info")
    return redirect(url_for("results"))

@app.route("/results")
@login_required
def results():
    candidate_id = session.get("last_candidate_id")
    candidate = get_candidate_by_id(candidate_id) if candidate_id else None
    transcription_pending = False
    awaiting_for = session.get("awaiting_transcription_for")

    if candidate and awaiting_for == str(candidate["_id"]):
        audio_transcription = candidate.get("audio_transcription")
        if audio_transcription and audio_transcription.get("status") in {"completed", "failed"}:
            session.pop("awaiting_transcription_for", None)
        else:
            transcription_pending = True
    elif not candidate:
        session.pop("awaiting_transcription_for", None)

    candidates = get_all_candidates()
    return render_template(
        "results.html",
        candidate=candidate,
        candidate_count=len(candidates),
        transcription_pending=transcription_pending
    )

# ── API: Transcription Status (AJAX polling) ──
@app.route("/api/transcription_status/<candidate_id>")
@login_required
def transcription_status(candidate_id):
    candidate = get_candidate_by_id(candidate_id)
    if not candidate:
        return jsonify({"status": "not_found"}), 404
    
    audio = candidate.get("audio_transcription")
    if not audio:
        return jsonify({"status": "none"})
    
    return jsonify({
        "status": audio.get("status", "unknown"),
        "text": audio.get("text", ""),
        "language": audio.get("language", ""),
        "error": audio.get("error"),
    })

# ── API: Candidate Profile (for modal) ──
@app.route("/api/candidate/<candidate_id>")
@login_required
def candidate_profile(candidate_id):
    candidate = get_candidate_by_id(candidate_id)
    if not candidate:
        return jsonify({"error": "not found"}), 404
    
    # Don't send the full resume text to keep response small
    return jsonify({
        "_id": candidate["_id"],
        "name": candidate.get("name", "Unknown"),
        "filename": candidate.get("filename", ""),
        "resume_url": candidate.get("resume_url", ""),
        "ai_summary": candidate.get("ai_summary", ""),
        "skills": candidate.get("skills", []),
        "education": candidate.get("education", []),
        "experience": candidate.get("experience", []),
        "match_score": candidate.get("match_score", 0),
        "skill_gaps": candidate.get("skill_gaps", {"matched": [], "missing": []}),
        "job_description": candidate.get("job_description", ""),
        "audio_transcription": candidate.get("audio_transcription"),
        "raw_text_preview": candidate.get("raw_text_preview", ""),
    })

@app.route("/ranking", methods=["GET", "POST"])
@login_required
def ranking():
    candidates = get_all_candidates()
    job_description = ""
    ranked = list(candidates)

    if request.method == "POST":
        job_description = request.form.get("job_description", "").strip()
        if job_description and candidates:
            from skill_extractor import extract_skills, SKILLS_LIST
            from job_matcher import calculate_match_score, find_skill_gaps, rank_candidates
            from db import candidates_collection
            from bson.objectid import ObjectId
            jd_skills = extract_skills(job_description)
            
            # Recalculate score and gaps for all candidates and update in DB
            for candidate in candidates:
                candidate_id = candidate["_id"]
                candidate_skills = candidate.get("skills", [])
                resume_text = candidate.get("resume_text", "")
                
                # Calculate new metrics based on new JD
                new_match_score = calculate_match_score(resume_text, job_description, candidate_skills, jd_skills)
                new_skill_gaps = find_skill_gaps(candidate_skills, job_description, SKILLS_LIST, jd_skills)
                
                # Update DB directly
                try:
                    candidates_collection.update_one(
                        {"_id": ObjectId(candidate_id)},
                        {"$set": {
                            "job_description": job_description,
                            "match_score": new_match_score,
                            "skill_gaps": new_skill_gaps
                        }}
                    )
                except Exception as e:
                    logger.error(f"Error updating candidate {candidate_id} during re-ranking: {e}")

            # Re-fetch the updated candidates from the database
            candidates = get_all_candidates()
            # Rank the newly fetched candidates
            ranked = rank_candidates(candidates, job_description, jd_skills)

    return render_template("ranking.html", ranked=ranked, job_description=job_description, candidate_count=len(candidates))

@app.route("/clear")
@login_required
def clear():
    clear_all_candidates()
    session.pop("last_candidate_id", None)
    flash("All candidate data cleared from database.", "info")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
