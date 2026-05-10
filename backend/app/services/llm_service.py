import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

def rewrite_text(text):
    """
    Rewrites the input text using Ollama to remove plagiarism while preserving meaning.
    """
    prompt = f"""
### Task: Professional Content Paraphrasing
You are a world-class academic editor and creative writer. Your goal is to take the provided text and rewrite it from scratch.

### Requirements:
1. **Zero Plagiarism**: Change the sentence structure, vocabulary, and flow completely.
2. **Meaning Preservation**: Ensure the core message, facts, and tone remain identical to the original.
3. **Natural Flow**: The output must sound professional, coherent, and human-written.
4. **No Meta-Talk**: Provide ONLY the rewritten text. Do not include phrases like "Here is the rewritten text" or "Original:".

### Original Text:
{text}

### Rewritten Version:
"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 1024,
                    "temperature": 0.6,
                    "top_p": 0.9,
                }
            },
            timeout=300 # Increased to 5 minutes to allow for model loading/slow CPUs
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        return f"Error: Ollama took too long to respond (Timeout). Your computer might be slow at processing 400 words. Try a shorter text or check if your CPU/GPU is overloaded."
    except Exception as e:
        print(f"Ollama Service Error: {e}")
        return f"Error: Unable to reach Ollama at {OLLAMA_BASE_URL}. Ensure the Ollama app is running in your system tray."
