"""
db.py
-----
Handles MongoDB connection and operations for HireScope AI.
"""

import logging
import os
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from dotenv import load_dotenv
from bson.objectid import ObjectId

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI", "")
if not MONGO_URI:
    logger.error("MONGO_URI environment variable is not set!")
    raise RuntimeError("MONGO_URI environment variable is required. Set it in your .env file.")

client = MongoClient(MONGO_URI)

# Database Name
db = client.get_database("resume_screener")

# Collections
users_collection = db.get_collection("users")
jobs_collection = db.get_collection("jobs")
candidates_collection = db.get_collection("candidates")
trash_candidates_collection = db.get_collection("trash_candidates")

# ── User Operations ──

def create_user(username, email, password, role="recruiter"):
    if users_collection.find_one({"email": email}):
        return False, "Email already exists"
    
    hashed_password = generate_password_hash(password)
    user_data = {
        "username": username,
        "email": email,
        "password": hashed_password,
        "role": role,
        "created_at": datetime.utcnow()
    }
    users_collection.insert_one(user_data)
    return True, "User created successfully"

def authenticate_user(email, password):
    user = users_collection.find_one({"email": email})
    if user and check_password_hash(user["password"], password):
        # Don't return the password hash in the user object
        user['_id'] = str(user['_id'])
        del user['password']
        return user
    return None

# ── Candidate Operations ──

def insert_candidate(candidate_data):
    """
    Candidate data should include name, filename, text, skills, match_score, etc.
    """
    candidate_data["created_at"] = datetime.utcnow()
    result = candidates_collection.insert_one(candidate_data)
    return str(result.inserted_id)

def get_all_candidates(user_id=None):
    query = {}
    if user_id:
        query["uploaded_by"] = user_id
    candidates = list(candidates_collection.find(query).sort("match_score", -1))
    for c in candidates:
        c['_id'] = str(c['_id'])
    return candidates

def get_candidate_by_id(candidate_id):
    try:
        candidate = candidates_collection.find_one({"_id": ObjectId(candidate_id)})
        if candidate:
            candidate['_id'] = str(candidate['_id'])
        return candidate
    except Exception:
        return None

def _candidate_filter(candidate_id):
    return {"_id": ObjectId(candidate_id)}


def set_candidate_audio_processing(candidate_id, audio_url=None):
    candidates_collection.update_one(
        _candidate_filter(candidate_id),
        {
            "$set": {
                "audio_transcription": {
                    "status": "processing",
                    "text": "",
                    "language": "",
                    "error": None,
                    "audio_url": audio_url,
                    "updated_at": datetime.utcnow(),
                }
            }
        },
    )


def update_candidate_audio(candidate_id, audio_text, language, audio_url=None):
    candidates_collection.update_one(
        _candidate_filter(candidate_id),
        {
            "$set": {
                "audio_transcription": {
                    "status": "completed",
                    "text": audio_text,
                    "language": language,
                    "error": None,
                    "audio_url": audio_url,
                    "updated_at": datetime.utcnow(),
                }
            }
        },
    )


def update_candidate_audio_error(candidate_id, error_msg, audio_url=None):
    candidates_collection.update_one(
        _candidate_filter(candidate_id),
        {
            "$set": {
                "audio_transcription": {
                    "status": "failed",
                    "text": "",
                    "language": "",
                    "error": error_msg,
                    "audio_url": audio_url,
                    "updated_at": datetime.utcnow(),
                }
            }
        },
    )

def clear_all_candidates(user_id=None):
    query = {}
    if user_id:
        query["uploaded_by"] = user_id
        
    candidates_to_trash = list(candidates_collection.find(query))
    if candidates_to_trash:
        for c in candidates_to_trash:
            c["deleted_at"] = datetime.utcnow()
        trash_candidates_collection.insert_many(candidates_to_trash)
        
    candidates_collection.delete_many(query)
