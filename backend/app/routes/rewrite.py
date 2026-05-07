from fastapi import APIRouter, HTTPException
from app.services.llm_service import rewrite_text
from app.services.similarity import compute_similarity
from app.db.memory_store import save_entry   # ✅ import

router = APIRouter()

@router.post("/rewrite")
def rewrite(data: dict):
    text = data.get("text", "")   # ✅ defined here
    
    if len(text.split()) > 400:
        raise HTTPException(status_code=400, detail="Text exceeds the 400 words limit.")

    rewritten = rewrite_text(text)
    similarity = compute_similarity(text, rewritten)

    # ✅ SAVE INSIDE FUNCTION
    save_entry(text, rewritten, similarity)

    return {
        "original": text,
        "rewritten": rewritten,
        "similarity": similarity,
        "plagiarism_percent": similarity * 100
    }