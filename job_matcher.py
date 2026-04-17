"""
job_matcher.py
--------------
Matches resumes to job descriptions using Advanced NLP embeddings (sentence-transformers)
and performs Exact Skill Overlap measurement. Also provides precise skill gap analysis.
"""
import re

from sentence_transformers import SentenceTransformer, util

# Load the model once when the file is imported
# This will download ~90MB on first run.
print("Loading sentence-transformers model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_match_score(resume_text, job_description, resume_skills=None, jd_skills=None):
    """
    Calculate how well a resume matches a job description using a Hybrid Score:
    50% Sentence Embeddings (Semantic Match) + 50% Exact Skill Overlap.
    """
    if not resume_text.strip() or not job_description.strip():
        return 0.0

    # 1. Compute Semantic Score (Cosine Similarity)
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    job_emb = model.encode(job_description, convert_to_tensor=True)
    cosine_scores = util.cos_sim(resume_emb, job_emb)
    semantic_score = max(0.0, float(cosine_scores[0][0])) * 100
    
    # 2. Compute Exact Overlap Score
    overlap_score = 0.0
    if jd_skills is not None and resume_skills is not None:
        jd_skills_lower = [s.lower() for s in jd_skills]
        resume_skills_lower = [s.lower() for s in resume_skills]
        if len(jd_skills_lower) > 0:
            matched_skills = [s for s in jd_skills_lower if s in resume_skills_lower]
            overlap_score = (len(matched_skills) / len(jd_skills_lower)) * 100
        
        # Hybrid Approach: Average of semantic and exact overlap
        final_score = (semantic_score + overlap_score) / 2
    else:
        final_score = semantic_score

    return round(final_score, 1)


def find_skill_gaps(resume_skills, job_description, all_skills_list=None, jd_skills=None):
    """
    Compare resume skills against skills mentioned in the job description using precise word boundaries.
    """
    job_lower = job_description.lower()
    resume_skills_lower = [s.lower() for s in resume_skills]

    # If jd_skills are explicitly provided, use them
    if jd_skills is not None:
        job_skills = jd_skills
    # Otherwise fallback to regex matching against the all_skills_list
    elif all_skills_list:
        job_skills = []
        for skill in all_skills_list:
            pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
            if re.search(pattern, job_lower):
                job_skills.append(skill)
    else:
        job_skills = []

    matched = [s for s in job_skills if s.lower() in resume_skills_lower]
    missing = [s for s in job_skills if s.lower() not in resume_skills_lower]

    return {
        "matched": matched,
        "missing": missing,
    }


def rank_candidates(candidates, job_description, jd_skills=None):
    """
    Rank multiple candidates based on their semantic and exact skill match to a job description.
    """
    ranked = []
    
    if not job_description.strip():
        return sorted(candidates, key=lambda x: x.get("match_score", 0), reverse=True)

    # Pre-encode the JD once
    job_emb = model.encode(job_description, convert_to_tensor=True)

    jd_skills_lower = []
    if jd_skills:
        jd_skills_lower = [s.lower() for s in jd_skills]

    for candidate in candidates:
        resume_emb = model.encode(candidate["resume_text"], convert_to_tensor=True)
        cosine_scores = util.cos_sim(resume_emb, job_emb)
        semantic_score = max(0.0, float(cosine_scores[0][0])) * 100
        
        # Calculate overlap score
        overlap_score = 0.0
        resume_skills = candidate.get("skills", [])
        resume_skills_lower = [s.lower() for s in resume_skills]
        
        if jd_skills_lower:
            matched_skills = [s for s in jd_skills_lower if s in resume_skills_lower]
            overlap_score = (len(matched_skills) / len(jd_skills_lower)) * 100
            final_score = (semantic_score + overlap_score) / 2
        else:
            final_score = semantic_score
            
        score = round(final_score, 1)
        
        ranked.append({
            "_id": candidate["_id"],
            "name": candidate["name"],
            "match_score": score,
            "skill_count": len(resume_skills),
            "skills": resume_skills,
        })

    # Sort by match score descending, then by skill count descending
    ranked.sort(key=lambda x: (x["match_score"], x["skill_count"]), reverse=True)
    return ranked
 