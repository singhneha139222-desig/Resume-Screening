FROM python:3.9-slim

WORKDIR /app

# Install system dependencies — ffmpeg required for Whisper audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (needed for text processing)
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')" 2>/dev/null || true

# Copy all project files
COPY . .

# Create upload directories
RUN mkdir -p uploads/resumes uploads/audio

# Expose port required by Hugging Face Spaces (Docker SDK)
EXPOSE 7860

# Environment variables for Flask
ENV FLASK_RUN_PORT=7860
ENV FLASK_RUN_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Production server with gunicorn — 120s timeout for ML model loading + Whisper
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--timeout", "120", "--workers", "2", "--threads", "4"]
