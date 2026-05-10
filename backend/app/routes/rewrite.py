from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.services.llm_service import rewrite_text
from app.services.similarity import compute_similarity
from app.services.text_utils import extract_from_url, extract_from_file
from app.db.memory_store import save_entry
import json
from typing import Optional

router = APIRouter()

@router.post("/rewrite")
async def rewrite(
    text: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    source_text = ""
    
    if text:
        source_text = text
    elif url:
        source_text = extract_from_url(url)
        if not source_text:
            raise HTTPException(status_code=400, detail="Could not extract text from the provided URL.")
    elif file:
        content = await file.read()
        source_text = extract_from_file(content, file.filename)
        if not source_text:
            raise HTTPException(status_code=400, detail="Could not extract text from the uploaded file.")
    else:
        raise HTTPException(status_code=400, detail="No source provided (text, url, or file).")

    if len(source_text.split()) > 1000: # Increased limit for documents
        raise HTTPException(status_code=400, detail="Source text exceeds the 1000 words limit.")

    rewritten = rewrite_text(source_text)
    similarity = compute_similarity(source_text, rewritten)

    save_entry(source_text, rewritten, similarity)

    return {
        "original": source_text,
        "rewritten": rewritten,
        "similarity": similarity,
        "plagiarism_percent": similarity * 100
    }