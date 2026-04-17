---
title: HireScope AI
emoji: 🔍
colorFrom: indigo
colorTo: cyan
sdk: docker
app_file: app.py
python_version: "3.9"
pinned: false
---

# 🔍 HireScope AI — Intelligent Resume Screening

An AI-powered web application that parses resumes, extracts 250+ skills, matches candidates to job descriptions using **Sentence-Transformers** semantic embeddings, ranks them with hybrid scoring, and transcribes audio intros with **OpenAI Whisper** — built with Flask, MongoDB Atlas, Cloudinary, and Tailwind CSS.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-green)
![Whisper](https://img.shields.io/badge/Whisper-OpenAI-orange)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green)
![Tailwind](https://img.shields.io/badge/Tailwind-CSS_v3-blue)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Resume Upload** | Upload PDF or DOCX files for automatic text extraction (including tables) |
| **250+ Skill Extraction** | Regex-based extraction across 24 BTech categories (Software, ML, DevOps, Embedded, etc.) |
| **Skill Normalization** | Infers higher-level skills (e.g., "TensorFlow" → "Deep Learning" added) |
| **Semantic Matching** | Sentence-Transformer (`all-MiniLM-L6-v2`) embeddings + Cosine Similarity |
| **Hybrid Scoring** | 50% Semantic Score + 50% Exact Skill Overlap → Score 0–100 |
| **Skill Gap Analysis** | Shows matched ✅ and missing ❌ skills for each candidate |
| **Audio Transcription** | Upload voice recordings → transcribed via OpenAI Whisper (local-file-first approach) |
| **Async Processing** | Audio transcription runs in background thread via ThreadPoolExecutor |
| **AJAX Polling** | Real-time transcription status updates without page reload |
| **Candidate Ranking** | Leaderboard with re-ranking via custom JD |
| **Candidate Profiles** | Click any candidate → full modal with skills, education, experience, audio |
| **Resume Download** | Cloudinary-hosted with `fl_attachment` for direct download |
| **Auth System** | Login/Register with Werkzeug password hashing |
| **Premium UI** | Tailwind CSS light theme + Inter font + glassmorphism + animations |

---

## 🗂️ Project Structure

```
HireScope-AI/
├── app.py                    # Flask main app (routes, auth, APIs)
├── resume_parser.py          # PDF/DOCX text extraction + name/email/phone extraction
├── skill_extractor.py        # 250+ skills regex extraction + normalization
├── job_matcher.py            # Sentence-Transformers matching + ranking engine
├── audio_transcriber.py      # Whisper speech-to-text (local-file-first approach)
├── db.py                     # MongoDB CRUD operations
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker deployment config (HuggingFace Spaces)
├── .env                      # Sensitive credentials (NOT in Git)
├── docs/
│   ├── SRS.md                # Software Requirements Specification
│   ├── JIRA_PLAN.md          # Sprint planning & bug tracker
│   └── REVISION.md           # Viva revision guide (Hinglish + English)
├── templates/
│   ├── base.html             # Shared layout (nav, footer, flash messages)
│   ├── login.html            # Auth - sign in
│   ├── register.html         # Auth - sign up
│   ├── index.html            # Dashboard + resume upload (drag & drop)
│   ├── results.html          # Analysis results + audio upload + transcription
│   └── ranking.html          # Candidate leaderboard + profile modal
├── static/css/style.css      # Supplemental CSS
└── uploads/
    ├── resumes/              # Temporary resume storage
    └── audio/                # Temporary audio storage (for Whisper)
```

---

## 🚀 How to Run

### Prerequisites

- **Python 3.9+** installed
- **ffmpeg** installed (required for Whisper audio processing)
  - Windows: `winget install ffmpeg`
  - Mac: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/HireScope-AI.git
cd HireScope-AI

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download NLTK data (one-time)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 6. Create .env file with your credentials
# MONGO_URI=mongodb+srv://...
# CLOUDINARY_CLOUD_NAME=...
# CLOUDINARY_API_KEY=...
# CLOUDINARY_API_SECRET=...
# GOOGLE_API_KEY=...  (optional)
# SECRET_KEY=your-secret-key

# 7. Run the application
python app.py
```

### Open in Browser

```
http://127.0.0.1:5000
```

---

## 📖 How It Works

### 1. Resume Parsing (`resume_parser.py`)
- **PDF** files parsed using `PyPDF2`
- **DOCX** files parsed using `python-docx` (paragraphs + tables)
- Extracts candidate name, email, phone from raw text

### 2. Skill Extraction (`skill_extractor.py`)
- **250+ skills** across 24 categories covering all BTech career paths
- Regex-based **word-boundary matching** for precision
- **Skill normalization** infers related skills

### 3. Semantic Matching (`job_matcher.py`)
```
Resume Text  →  Sentence Embedding (384-dim)  ─┐
                                                ├→  50% Cosine Similarity + 50% Skill Overlap → Score (0-100)
Job Description  →  Sentence Embedding (384-dim) ─┘
```

### 4. Audio Transcription (`audio_transcriber.py`)
- Uses OpenAI's Whisper (`base` model)
- **Local-file-first approach** — audio saved to disk, transcribed from local path
- Supports MP3, WAV, M4A, FLAC, OGG, WEBM
- Runs asynchronously via `ThreadPoolExecutor`

### 5. Candidate Ranking
- Leaderboard with re-ranking via custom JD
- Clickable candidate profiles with full biodata modal

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Backend | Python 3.9, Flask 3.x |
| Text Extraction | PyPDF2, python-docx |
| NLP | Regex (250+ skills), NLTK |
| Semantic ML | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Audio ASR | OpenAI Whisper (base model) |
| Database | MongoDB Atlas (PyMongo) |
| File Storage | Cloudinary |
| Frontend | Tailwind CSS v3, Inter Font, Vanilla JS, AJAX |
| Auth | Werkzeug password hashing, Flask sessions |
| Deployment | Docker, Gunicorn, HuggingFace Spaces |

---

## 📄 License

This project is for educational purposes. Feel free to use and modify.
