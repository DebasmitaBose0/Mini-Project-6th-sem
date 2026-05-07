from fastapi import APIRouter
from app.db.memory_store import get_history as db_get_history, delete_entry as db_delete_entry

router = APIRouter()

@router.get("/history")
def get_history():
    return db_get_history()

@router.delete("/history/{entry_id}")
def delete_entry(entry_id: str):
    db_delete_entry(entry_id)
    return {"status": "deleted"}

@router.delete("/history")
def clear():
    history = db_get_history()
    history.clear()
    return {"status": "cleared"}