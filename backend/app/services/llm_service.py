import requests

def rewrite_text(text):
    prompt = f"""
Rewrite the following text to reduce plagiarism.
Keep meaning but change structure.

Text:
{text}
"""

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:8b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        res.raise_for_status()
        return res.json()["response"]
    except Exception as e:
        print(f"Ollama error: {e}")
        return f"Fallback rewritten text due to error: {str(e)}"