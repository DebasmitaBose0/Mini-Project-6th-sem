from fastapi import APIRouter
from app.services.similarity import compute_similarity, sentence_level_similarity

router = APIRouter()

@router.post("/analyze")
def analyze(data: dict):
    original = data["original"]
    rewritten = data["rewritten"]

    sim = compute_similarity(original, rewritten)
    sentence_sim = sentence_level_similarity(original, rewritten)

    # Logic to show "Proper" values:
    # Plagiarism: Verbatim matches (exact sentence overlap) relative to original length
    # Similarity: Overall structural/semantic overlap
    
    orig_sentences = [s.strip() for s in original.split('.') if s.strip()]
    plagiarism_score = (len(sentence_sim) / max(len(orig_sentences), 1)) * 100
    
    # Cap plagiarism by overall similarity to stay realistic
    plagiarism_score = min(plagiarism_score, sim * 100)
    
    return {
        "similarity": round(sim * 100, 2),
        "plagiarism_percent": round(plagiarism_score, 1),
        "sentence_matches": sentence_sim
    }