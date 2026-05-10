import requests
from bs4 import BeautifulSoup
import io
# pyrefly: ignore [missing-import]
import PyPDF2
# pyrefly: ignore [missing-import]
import docx
import re

def clean_text(text):
    """Basic text cleaning to remove extra whitespaces."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_from_url(url):
    """
    Fetch and extract readable text from a URL.
    Uses a browser-like User-Agent to avoid common 403 blocks.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Detect encoding from content if not provided in headers
        if response.encoding == 'ISO-8859-1':
            response.encoding = response.apparent_encoding

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove non-text elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.extract()

        # Get text with a separator to avoid merging words
        text = soup.get_text(separator='\n')
        
        # Clean up text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)
        
    except Exception as e:
        print(f"URL extraction error: {e}")
        return None

def extract_from_file(file_content, filename):
    """
    Extract text from PDF, DOCX, or TXT files.
    Includes encoding fallbacks for text files.
    """
    try:
        if filename.lower().endswith('.pdf'):
            reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return "\n".join(text)
            
        elif filename.lower().endswith('.docx'):
            doc = docx.Document(io.BytesIO(file_content))
            return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            
        elif filename.lower().endswith('.txt'):
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    return file_content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return None
            
        return None
    except Exception as e:
        print(f"File extraction error for {filename}: {e}")
        return None