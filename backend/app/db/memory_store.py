import uuid
from datetime import datetime

history = []

def save_entry(original, rewritten, similarity):
    entry_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    history.append({
        "id": entry_id,
        "timestamp": timestamp,
        "original": original,
        "rewritten": rewritten,
        "similarity": similarity
    })

def get_history():
    return history

def delete_entry(entry_id: str):
    global history
    history = [entry for entry in history if entry["id"] != entry_id]