import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rewrite_text(text):
    prompt = f"""
Rewrite the following text to reduce plagiarism.

Rules:
- Keep the original meaning
- Change sentence structure
- Use natural human language
- Avoid simple synonym replacement
- Ensure the output is significantly different

Text:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()