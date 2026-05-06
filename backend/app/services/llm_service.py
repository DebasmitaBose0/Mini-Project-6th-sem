import requests

def rewrite_text(text):
    prompt = f"""
You are an expert AI editor and paraphrasing tool.
Your task is to rewrite the following text completely to remove any trace of plagiarism while preserving the original meaning, tone, and key information.
Ensure the structure, vocabulary, and sentence flow are significantly altered to produce highly original content. Provide ONLY the rewritten text, without any introductory or concluding remarks.

Original Text:
{text}

Rewritten Text:
"""

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 512,
                    "temperature": 0.5
                }
            },
            timeout=300
        )
        res.raise_for_status()
        return res.json()["response"]
    except Exception as e:
        print(f"Ollama error: {e}")
        return f"Fallback rewritten text due to error: {str(e)}"