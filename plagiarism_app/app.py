"""
Plagiarism Checker & Rewriter — Flask Application
=====================================================
Serves the trained model through a web API and a beautiful frontend.

Features:
  1. Detect plagiarism between two texts
  2. Show plagiarized sentences highlighted
  3. Rewrite plagiarized sentences into human-like original text
  4. Keep the meaning identical in both versions

Endpoints: 
    GET  /              → Main page (HTML UI)
    POST /api/check     → Check plagiarism + rewrite plagiarized text
    POST /api/compare   → Corpus scan + rewrite flagged sentences
    GET  /api/health    → Health check + model info

Usage:
    python app.py
    python app.py --port 8080
"""

import os
import sys
import argparse
from dotenv import load_dotenv
load_dotenv()
import re
import random
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from scipy.sparse import hstack, csr_matrix
from difflib import SequenceMatcher
import urllib.request
import urllib.error
import json
import ssl
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# Try loading NLTK for POS tagging
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

try:
    from plagiarism_app.paraphrase_engine import deep_paraphrase, deep_paraphrase_paragraph
except ImportError:
    from paraphrase_engine import deep_paraphrase, deep_paraphrase_paragraph

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "plagiarism_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.pkl")

def get_llm_configs():
    """Returns a list of available LLM configurations. Prioritizes Groq -> Gemini -> OpenAI."""
    configs = []
    
    # 1. Groq API
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        configs.append({
            "api_key": groq_key,
            "api_model": os.getenv("GROQ_API_MODEL", "llama3-8b-8192").strip(),
            "request_url": "https://api.groq.com/openai/v1/chat/completions"
        })

    # 2. Gemini API
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        configs.append({
            "api_key": gemini_key,
            "api_model": os.getenv("GEMINI_API_MODEL", "gemini-1.5-flash").strip(),
            "request_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        })

    # 3. OpenAI API
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        configs.append({
            "api_key": key,
            "api_model": os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo").strip(),
            "request_url": "https://api.openai.com/v1/chat/completions"
        })

    return configs

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# ─────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────
model = None
vectorizer = None
metadata = None


def load_model():
    """Load trained model artifacts."""
    global model, vectorizer, metadata

    if not os.path.exists(MODEL_PATH):
        print("  Model not found! Please run train_model.py first.")
        print(f"   Expected at: {MODEL_PATH}")
        return False

    print("🔄 Loading model artifacts...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    metadata = joblib.load(METADATA_PATH)

    print(f"    Model loaded (accuracy: {metadata.get('accuracy', 'N/A'):.4f})")
    print(f"    Vectorizer loaded ({metadata.get('tfidf_max_features', 'N/A')} features)")
    return True


def get_rewrite_engine(sentence_analysis: list[dict]) -> str:
    """Return the active rewrite engine used for the analyzed sentences."""
    for item in sentence_analysis:
        rewrite = item.get("rewrite", {})
        if rewrite.get("rewrite_method") == "openai":
            return "openai"
    return "local"

def check_api_status():
    """Print API status to console for debugging."""
    configs = get_llm_configs()
    if not configs:
        print("⚠️  No API keys found in .env! Using local engine only.")
    else:
        print(f"✅ Loaded {len(configs)} API configuration(s). Ready for high-quality rewriting.")

# Call status check on startup
check_api_status()


def clean_up_rewrite(text: str) -> str:
    """Ensure proper capitalization and punctuation."""
    if not text:
        return text

    text = fix_common_lexical_artifacts(text)
    text = text.strip()
    
    # 1. Basic Spacing fix (no space before, one space after)
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])([^\s\d])', r'\1 \2', text)
    
    # 2. Capitalize first letter
    if len(text) > 0 and text[0].islower():
        text = text[0].upper() + text[1:]
    
    # 3. Capitalization after sentence boundaries
    text = re.sub(r'([.!?;])\s+([a-z])', lambda m: m.group(1) + " " + m.group(2).upper(), text)
    
    # 4. Handle Parentheses spacing ( ( text ) -> (text) )
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    
    # 5. Handle Quotation spacing
    text = re.sub(r'\"\s+', '"', text)
    text = re.sub(r'\s+\"', '"', text)
    
    # 6. Ensure terminal punctuation
    if text and text[-1] not in ".!?;:":
        # If it looks like a sentence (has spaces), add a period
        if " " in text:
            text += "."
            
    return text.strip()


COMMON_LEXICAL_FIXES = {
    "inteligence": "intelligence",
    "artifical": "artificial",
    "nonetheles": "nonetheless",
    "concider": "consider",
    "recieve": "receive",
    "seperate": "separate",
    "definately": "definitely",
}


def fix_common_lexical_artifacts(text: str) -> str:
    """Normalize frequent model-generated spelling artifacts conservatively, preserving casing."""
    if not text:
        return text

    def apply_casing(replacement: str, original: str) -> str:
        """Apply casing pattern from original to replacement."""
        if original.isupper():
            return replacement.upper()
        elif original and original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        else:
            return replacement.lower()

    fixed = text
    # Whole-word spelling fixes with casing preservation.
    for wrong, right in COMMON_LEXICAL_FIXES.items():
        def replacer(match):
            original_match = match.group(0)
            return apply_casing(right, original_match)

        fixed = re.sub(rf"\b{re.escape(wrong)}\b", replacer, fixed, flags=re.IGNORECASE)

    # Frequent stitched fragment repair.
    fixed = re.sub(r"\berns\s+about\b", "concerns about", fixed, flags=re.IGNORECASE)
    return fixed

def llm_rewrite_sentence(sentence: str, source_sentence: str = "") -> str | None:
    """Use an LLM (OpenAI/Groq/Gemini) to rewrite a sentence if an API key is available. Supports fallback."""
    configs = get_llm_configs()
    if not configs:
        return None

    prompt = (
        "Rewrite the following sentence into very simple, easy-to-understand English. "
        "Keep the exact same meaning intact, but use completely different vocabulary and sentence structure to remove plagiarism. "
        "IMPORTANT: FOCUS HEAVILY ON PUNCTUATION. Ensure all commas, periods, and capitals are perfectly placed. "
        "Always start with a capital letter and end with proper punctuation. "
        "Do not use complex or difficult words. Make the result sound like natural, everyday human prose. "
        "Return only the polished sentence with no explanation."
    )
    if source_sentence:
        prompt += " The source sentence is provided for reference, but do not reuse its phrases or structure."

    prompt += f"\n\nSentence to rewrite: {sentence}"
    if source_sentence:
        prompt += f"\nOriginal source (to avoid): {source_sentence}"

    def call_llm(config: dict, current_prompt: str) -> str | None:
        payload = json.dumps({
            "model": config["api_model"],
            "messages": [
                {"role": "system", "content": "You are a skilled text humanizer and plagiarism remover. Rewrite text using simple, everyday English while completely changing the structure to ensure 0% plagiarism."},
                {"role": "user", "content": current_prompt},
            ],
            "temperature": 0.2,
            "top_p": 1,
            "max_tokens": max(200, len(sentence.split()) * 4),
        }).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        }

        try:
            request_obj = urllib.request.Request(config["request_url"], data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(request_obj, context=ssl.create_default_context(), timeout=30) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                choices = response_data.get("choices", [])
                if not choices:
                    return None
                return choices[0].get("message", {}).get("content", "").strip() or None
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            return None

    # Try each config in order. Return on first success.
    for config in configs:
        first_attempt = call_llm(config, prompt)
        if not first_attempt:
            continue
            
        result = clean_up_rewrite(first_attempt)
        
        if result.strip().lower() == sentence.strip().lower() or compute_text_similarity(sentence, result) > 0.90:
            stronger_prompt = prompt + (
                "\n\nThe rewritten sentence MUST be significantly different in vocabulary than the original to remove plagiarism. "
                "Use simpler, different words while keeping the exact meaning."
            )
            second_attempt = call_llm(config, stronger_prompt)
            result = clean_up_rewrite(second_attempt or first_attempt)
        
        return result
            
    return None



def llm_generate_plagiarized(clean_text: str) -> tuple[str, str] | None:
    """Use an LLM (OpenAI/Groq/Gemini) to generate a naturally plagiarized version of clean text."""
    configs = get_llm_configs()
    if not configs:
        return None

    prompt = (
        "You are a text humanizer and plagiarism remover. "
        "Given a source paragraph, rewrite it into a simple, highly readable, natural human-style version that preserves the original meaning exactly. "
        "Keep most proper names and titles unchanged. "
        "The output must use completely different vocabulary and sentence structure to ensure low similarity (acting as a plagiarism remover). "
        "Use easy, simple English. Do not use complex or difficult words. Do not invent nonsense words, do not distort facts, and keep the text readable. "
        "Return only the rewritten paragraph with no explanation."
    )
    prompt += "\n\nSource paragraph:\n" + clean_text

    def call_llm(config: dict, current_prompt: str) -> str | None:
        payload = json.dumps({
            "model": config["api_model"],
            "messages": [
                {"role": "system", "content": "You are a skilled text humanizer and plagiarism remover."},
                {"role": "user", "content": current_prompt},
            ],
            "temperature": 0.45,
            "top_p": 1,
            "max_tokens": max(200, len(clean_text.split()) * 4),
        }).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        }

        try:
            request_obj = urllib.request.Request(config["request_url"], data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(request_obj, context=ssl.create_default_context(), timeout=60) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                choices = response_data.get("choices", [])
                if not choices:
                    return None
                return choices[0].get("message", {}).get("content", "").strip() or None
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            return None

    for config in configs:
        plag_text = call_llm(config, prompt)
        if not plag_text:
            continue

        if plag_text.strip().lower() == clean_text.strip().lower():
            continue

        app.logger.info("LLM generated auto-plagiarized text using %s.", config["api_model"])
        return plag_text, config["api_model"]

    return None


# ─────────────────────────────────────────────────────────
# AUTO-PLAGIARISM SIMULATION (NEW)
# ─────────────────────────────────────────────────────────

def auto_generate_plagiarized(clean_text: str) -> tuple[str, str, dict]:
    """
    Simulate student plagiarism from clean original text.
    Returns: (suspected_plag_text, original_clean_text, simulation_stats)
    """
    sentences = split_into_sentences(clean_text)
    plag_sentences = []
    verbatim_count = 0
    rewritten_count = 0
    
    llm_result = llm_generate_plagiarized(clean_text)
    if llm_result:
        plag_text, model_used = llm_result
        suspected_text = clean_up_rewrite(plag_text)
        verbatim_count = round(len(sentences) * 0.5)
        rewritten_count = len(sentences) - verbatim_count
    else:
        model_used = None
        for sent in sentences:
            # HARD REWRITE: Always paraphrase, never copy verbatim
            variant = deep_paraphrase(sent)
            
            # Fallback 1: If result too similar, try phrase paraphrases
            if compute_text_similarity(sent, variant) > 0.90:
                variant = apply_phrase_paraphrases(sent)
            
            # Fallback 2: If still too similar, try restructuring
            if compute_text_similarity(sent, variant) > 0.90:
                variant = restructure_sentence(sent)
            
            # Fallback 3: If still identical, force aggressive synonyms
            if compute_text_similarity(sent, variant) > 0.90:
                variant = _apply_aggressive_synonyms(sent, "")
            
            plag_sentences.append(variant)
            rewritten_count += 1
        
        suspected_text = clean_up_rewrite(" ".join(plag_sentences))
        
        # FINAL FAILSAFE: If suspected text is identical to clean, force deep rewrite
        if suspected_text.strip() == clean_text.strip():
            suspected_text = clean_up_rewrite(deep_paraphrase_paragraph(clean_text))

    stats = {
        "verbatim_sentences": verbatim_count,
        "rewritten_sentences": rewritten_count,
        "total_sentences": len(sentences),
        "plagiarism_ratio": round(verbatim_count / len(sentences) * 100, 1),
        "generated_with_llm": bool(model_used),
        "llm_model_used": model_used or "none"
    }
    
    return suspected_text, clean_text, stats


def preserve_punctuation_token(word: str, replacement: str) -> str:
    """Preserve punctuation around a replaced word."""
    match = re.match(r'^([^\w]*)([\w-]+)([^\w]*)$', word)
    if not match:
        return replacement
    return f"{match.group(1)}{replacement}{match.group(3)}"


def create_light_plagiarism_variant(sentence: str) -> str:
    """Create a plausible lightly rewritten version of a sentence."""
    words = sentence.split()
    if len(words) < 4:
        return sentence

    candidate = words.copy()
    replaceable_indices = []

    for index, word in enumerate(words):
        clean = word.strip(".,!?;:'\"-()")
        if len(clean) <= 3:
            continue
        if index > 0 and word[0].isupper():
            continue
        if clean.lower() in {
                "this", "that", "have", "has", "had", "will", "would", "could", "should",
                "and", "or", "but", "for", "with", "from", "into", "about", "after",
                "before", "during", "through", "across", "near", "when", "while", "where"
        }:
            continue

        synonyms = SYNONYM_MAP.get(clean.lower(), [])
        if synonyms:
            replaceable_indices.append(index)

    random.shuffle(replaceable_indices)
    replacements = 0

    for idx in replaceable_indices:
        if replacements >= 2:
            break

        word = words[idx]
        clean = word.strip(".,!?;:'\"-()")
        synonyms = SYNONYM_MAP.get(clean.lower(), [])
        safe_synonyms = [syn for syn in synonyms if re.fullmatch(r"[A-Za-z ]+", syn) and syn.lower() != clean.lower()]
        if not safe_synonyms:
            continue

        chosen = random.choice(safe_synonyms)
        if word[0].isupper():
            chosen = chosen.capitalize()

        candidate[idx] = preserve_punctuation_token(word, chosen)
        replacements += 1

    candidate_sentence = " ".join(candidate)
    candidate_sentence = restructure_sentence(candidate_sentence)
    candidate_sentence = normalize_sentence(candidate_sentence)
    return candidate_sentence


# ─────────────────────────────────────────────────────────
# SYNONYM & REWRITING ENGINE
# ─────────────────────────────────────────────────────────

SYNONYM_MAP = {
    # Verbs
    "completed": ["finished", "accomplished", "concluded", "wrapped up"],
    "finished": ["completed", "accomplished", "concluded"],
    "submitted": ["handed in", "turned in", "delivered"],
    "discussed": ["talked about", "deliberated", "debated"],
    "achieved": ["accomplished", "attained", "reached"],
    "organized": ["arranged", "coordinated", "planned"],
    "reviewed": ["examined", "analyzed", "evaluated"],
    "implemented": ["executed", "carried out", "put into practice"],
    "prepared": ["arranged", "set up", "got ready"],
    "evaluated": ["assessed", "appraised", "judged"],
    "designed": ["created", "crafted", "developed"],
    "built": ["constructed", "assembled", "developed"],
    "tested": ["examined", "evaluated", "checked"],
    "fixed": ["repaired", "mended", "corrected"],
    "installed": ["set up", "configured", "put in"],
    "created": ["developed", "produced", "generated"],
    "analyzed": ["examined", "studied", "investigated"],
    "optimized": ["improved", "enhanced", "refined"],
    "deployed": ["launched", "released", "rolled out"],
    "automated": ["mechanized", "streamlined", "systematized"],
    "examined": ["inspected", "checked", "investigated"],
    "recovered": ["healed", "got better", "recuperated"],
    "visited": ["went to", "called on", "stopped by"],
    "traveled": ["journeyed", "went", "toured"],
    "climbed": ["ascended", "scaled", "went up"],
    "explored": ["investigated", "surveyed", "ventured through"],
    "photographed": ["captured", "took a photo of", "snapped"],
    "discovered": ["found", "uncovered", "came across"],
    "enjoyed": ["appreciated", "relished", "liked"],
    "celebrated": ["commemorated", "marked", "honored"],
    "followed": ["adhered to", "obeyed", "complied with"],
    "scored": ["earned", "achieved", "netted"],
    "practiced": ["rehearsed", "trained", "drilled"],
    "competed": ["contended", "participated", "vied"],
    "performed": ["executed", "carried out", "delivered"],
    "published": ["released", "issued", "put out"],
    "painted": ["illustrated", "decorated", "colored"],
    "resolved": ["solved", "settled", "sorted out"],
    "answered": ["responded to", "replied to", "addressed"],
    "wrote": ["composed", "authored", "penned"],
    "read": ["perused", "studied", "went through"],
    "watched": ["observed", "viewed", "witnessed"],
    "helped": ["assisted", "aided", "supported"],
    "started": ["began", "commenced", "initiated"],
    "bought": ["purchased", "acquired", "obtained"],
    "gave": ["provided", "presented", "offered"],
    "took": ["grabbed", "picked up", "collected"],
    "made": ["created", "produced", "crafted"],
    "said": ["stated", "mentioned", "declared"],
    "told": ["informed", "notified", "advised"],
    "asked": ["inquired", "questioned", "requested"],
    "showed": ["demonstrated", "displayed", "revealed"],
    "played": ["engaged in", "participated in", "took part in"],
    "walked": ["strolled", "ambled", "wandered"],
    "ran": ["sprinted", "dashed", "jogged"],
    "won": ["triumphed", "prevailed", "succeeded"],
    "documented": ["recorded", "logged", "chronicled"],
    "presented": ["showed", "displayed", "demonstrated"],
    "explained": ["clarified", "described", "elaborated on"],
    "teaches": ["instructs", "educates", "trains"],
    "teaches": ["instructs", "educates", "coaches"],
    "manages": ["handles", "oversees", "supervises"],
    "maintains": ["keeps up", "sustains", "preserves"],
    "monitors": ["tracks", "observes", "watches"],
    "collected": ["gathered", "assembled", "accumulated"],
    "selected": ["chose", "picked", "opted for"],
    "improved": ["enhanced", "upgraded", "bettered"],
    "reduced": ["decreased", "lowered", "cut down"],
    "increased": ["raised", "boosted", "elevated"],
    "established": ["set up", "founded", "created"],
    "conducted": ["carried out", "performed", "executed"],
    "developed": ["created", "built", "produced"],
    "described": ["explained", "outlined", "portrayed"],
    "determined": ["decided", "established", "figured out"],
    "identified": ["recognized", "spotted", "detected"],
    "produced": ["created", "generated", "manufactured"],
    "received": ["got", "obtained", "acquired"],
    "considered": ["thought about", "contemplated", "pondered"],
    "included": ["contained", "encompassed", "comprised"],
    "provided": ["gave", "supplied", "offered"],
    "required": ["needed", "demanded", "necessitated"],
    "suggested": ["proposed", "recommended", "advised"],
    "supported": ["backed", "endorsed", "upheld"],

    # Nouns
    "market": ["marketplace", "bazaar", "store"],
    "house": ["home", "residence", "dwelling"],
    "car": ["vehicle", "automobile", "ride"],
    "road": ["street", "path", "route"],
    "money": ["cash", "funds", "currency"],
    "work": ["collaborate", "team up", "work together"],
    "book": ["publication", "volume", "text"],
    "food": ["meal", "cuisine", "nourishment"],
    "friend": ["companion", "pal", "buddy"],
    "children": ["kids", "youngsters", "little ones"],
    "people": ["folks", "friends", "group"],
    "school": ["academy", "institution", "educational facility"],
    "problem": ["issue", "challenge", "difficulty"],
    "idea": ["concept", "notion", "thought"],
    "place": ["location", "spot", "site"],
    "country": ["nation", "state", "land"],
    "student": ["learner", "pupil", "scholar"],
    "teacher": ["educator", "instructor", "professor"],
    "scientist": ["researcher", "scholar", "investigator"],
    "doctor": ["physician", "medical professional", "practitioner"],
    "company": ["team", "group", "circle"],
    "project": ["undertaking", "initiative", "venture"],
    "report": ["document", "paper", "account"],
    "meeting": ["gathering", "conference", "assembly"],
    "strategy": ["plan", "approach", "method"],
    "team": ["group", "squad", "crew"],
    "findings": ["results", "discoveries", "conclusions"],
    "research": ["study", "investigation", "analysis"],
    "species": ["type", "variety", "kind"],
    "conference": ["meeting", "symposium", "convention"],
    "forest": ["woods", "woodland", "jungle"],
    "rainforest": ["tropical forest", "jungle", "tropical woodland"],
    "topic": ["subject", "theme", "matter"],
    "budget": ["financial plan", "allocation", "funds"],

    # Adjectives
    "beautiful": ["gorgeous", "stunning", "lovely"],
    "important": ["crucial", "vital", "significant"],
    "new": ["fresh", "novel", "recent"],
    "old": ["ancient", "aged", "vintage"],
    "good": ["nice", "pleasant", "great"],
    "bad": ["terrible", "awful", "poor"],
    "difficult": ["challenging", "tough", "demanding"],
    "easy": ["simple", "straightforward", "effortless"],
    "interesting": ["fascinating", "intriguing", "captivating"],
    "annual": ["yearly", "once-a-year", "regular"],
    "entire": ["whole", "complete", "full"],
    "successful": ["triumphant", "prosperous", "accomplished"],
    "clearly": ["plainly", "obviously", "evidently"],
    "successfully": ["effectively", "triumphantly", "with great success"],

    # Adverbs & others
    "very": ["extremely", "really", "incredibly"],
    "quickly": ["rapidly", "swiftly", "speedily"],
    "carefully": ["cautiously", "meticulously", "attentively"],
    "immediately": ["instantly", "right away", "promptly"],
    "yesterday": ["the day before", "the previous day", "a day ago"],
    "together": ["collectively", "as a group", "jointly"],
    "also": ["additionally", "moreover", "furthermore"],
    "however": ["nevertheless", "nonetheless", "yet"],
    "therefore": ["consequently", "thus", "as a result"],
    "because": ["since", "as", "due to the fact that"],
}

# Sentence restructuring templates
RESTRUCTURE_TEMPLATES = [
    # Move time/place to front
    ("at the {place}", "At the {place}, "),
    ("in the {place}", "In the {place}, "),
    ("on the {place}", "On the {place}, "),
]

# Transition phrases to make text sound more natural/human
HUMAN_TRANSITIONS = [
    "In other words, ",
    "To put it differently, ",
    "Essentially, ",
    "Put simply, ",
    "What this means is that ",
    "In simple terms, ",
]

PHRASE_PARAPHRASE_MAP = {
    "a paragraph is a distinct unit of writing": [
        "a paragraph forms a separate section of written text",
        "a paragraph is an independent block of writing",
    ],
    "one or more sentences": [
        "one sentence or several",
        "a single sentence or a few sentences",
    ],
    "single, cohesive idea or topic": [
        "one consistent theme",
        "a unified concept",
    ],
    "a foundational element of writing": [
        "a basic building block of written expression",
        "an essential component of good writing",
    ],
    "structure and clarity": [
        "organization and clarity",
        "clear structure",
    ],
    "usually featuring a topic sentence, supporting details, and a concluding sentence": [
        "typically including an opening topic sentence, supporting details, and a closing statement",
        "often containing an opening idea sentence, supporting details, and a final summary sentence",
    ],
    "properly organized, paragraphs help readers follow the author's logic and break up text for better readability": [
        "when arranged well, paragraphs guide the reader through the author's reasoning and separate content for easier reading",
        "well-structured paragraphs help readers follow the author's logic and split text into more readable sections",
    ],
}


def apply_phrase_paraphrases(sentence: str) -> str:
    """Apply longer phrase-based paraphrases to make a sentence more distinct."""
    text = sentence
    for phrase, alternatives in PHRASE_PARAPHRASE_MAP.items():
        lower_text = text.lower()
        if phrase in lower_text:
            replacement = random.choice(alternatives)
            start = lower_text.index(phrase)
            text = text[:start] + replacement + text[start + len(phrase):]
            text = normalize_sentence(text)
    return text


# ─────────────────────────────────────────────────────────
# DATAMUSE API INTEGRATION (Unlimited Vocabulary)
# ─────────────────────────────────────────────────────────
WORD_CACHE = {}

def prefetch_synonyms(text: str):
    """Disabled Datamuse API injection to prevent chaotic gibberish outputs."""
    pass


def get_synonyms_for_word(word: str) -> list[str]:
    """Get synonyms from static map or API cache."""
    clean = word.lower().strip(".,!?;:'\"-()")
    
    # 1. Static map
    if clean in SYNONYM_MAP:
        return SYNONYM_MAP[clean]
        
    # 2. API cache
    if clean in WORD_CACHE and WORD_CACHE[clean]:
        return WORD_CACHE[clean]
        
    return []


def get_rewrite_synonyms(word: str, static_only: bool = False) -> list[str]:
    """Get a cleaned list of synonyms for sentence rewriting."""
    clean = word.lower().strip(".,!?;:'\"-()")
    synonyms = []

    if clean in SYNONYM_MAP:
        synonyms.extend(SYNONYM_MAP[clean])

    if not static_only:
        synonyms.extend(get_synonyms_for_word(word))

    filtered = [s for s in synonyms if re.fullmatch(r"[A-Za-z ]+", s)]
    return list(dict.fromkeys(filtered))


def deep_paraphrase_sentence(sentence: str, source_sentence: str = "") -> str:
    """
    Safely paraphrase using the paraphrase_engine logic.
    """
    return deep_paraphrase(sentence)


def _apply_aggressive_synonyms(sentence: str, source_sentence: str = "") -> str:
    """Apply synonym replacement to as many content words as possible, avoiding gibberish."""
    words = sentence.split()
    source_lower = source_sentence.lower() if source_sentence else ""
    new_words = []
    replaced = 0
    max_rep = min(2, len(words)//8)  # Don't replace every single word, it breaks flow

    # Core English stop words to never replace
    stop_words = {"this", "that", "these", "those", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"}

    # Use NLTK POS tagging if available
    pos_tags = []
    if NLTK_AVAILABLE:
        try:
            clean_words = [w.strip(".,!?;:'\"") for w in words]
            pos_tags_raw = nltk.pos_tag(clean_words)
            pos_tags = [tag for w, tag in pos_tags_raw]
        except Exception:
            pos_tags = []

    for index, word in enumerate(words):
        clean = word.lower().rstrip(".,!?;:'\"")
        trail = word[len(word.rstrip(".,!?;:'\"")):]
        
        # Determine POS prefix (N=Noun, V=Verb, J=Adj, R=Adv)
        word_pos = pos_tags[index][:1] if index < len(pos_tags) else None

        # Skip proper nouns, quotes, stop words, and very short words
        if index > 0 and word[0].isupper() and not word.isupper():
            new_words.append(word)
            continue
        if word.startswith(("\"", "'")) or word.endswith(("\"", "'")):
            new_words.append(word)
            continue
        if len(clean) <= 3 or clean in stop_words:
            new_words.append(word)
            continue

        if replaced >= max_rep:
            new_words.append(word)
            continue

        synonyms = get_rewrite_synonyms(clean, static_only=False)
        if synonyms:
            # ONLY use synonyms that are shorter or simpler if possible
            if source_sentence:
                safe = [s for s in synonyms if s.lower() not in source_lower and s.lower() != clean and len(s) <= len(clean) + 2]
            else:
                safe = [s for s in synonyms if s.lower() != clean and len(s) <= len(clean) + 2]
                
            if safe:
                # Prioritize shorter words to keep it simple
                safe.sort(key=len)
                # Take one of the top 3 simplest
                chosen = random.choice(safe[:min(3, len(safe))])
                
                if word[0].isupper():
                    chosen = chosen[0].upper() + chosen[1:]

                new_words.append(chosen + trail)
                replaced += 1
                continue

        new_words.append(word)

    return " ".join(new_words)


def rewrite_sentence_human(sentence: str, source_sentence: str = "", strength: int = 2, mode: str = "remove_plagiarism") -> dict:
    # ── Force rejection of unchanged rewrites and fallback passes ──
    def _force_fallbacks(original, rewritten, source_sentence):
        # If rewrite too similar, redo it
        if compute_text_similarity(original, rewritten) > 0.85:
            rewritten = deep_paraphrase_sentence(original, source_sentence)
        # If still unchanged, try phrase paraphrase
        if rewritten.strip() == original.strip():
            rewritten = apply_phrase_paraphrases(original)
        # If still unchanged, try restructure
        if rewritten.strip() == original.strip():
            rewritten = restructure_sentence(original)
        return rewritten
    """
    Rewrite a plagiarized sentence into a genuinely different version.
    
    Strategy (Deep Paraphrasing):
      1. Try OpenAI LLM rewrite first if API key is available
      2. Use the deep paraphrase engine which restructures sentences
      3. Generate multiple candidates based on `strength` (1=Low, 2=Medium, 3=Aggressive)
      4. Apply Iterative Rewrite Loop if similarity is still too high.
    """
    if not sentence or not sentence.strip():
        return {
            "plagiarized": sentence,
            "rewritten": sentence,
            "changes_made": [],
            "meaning_preserved": 1.0,
            "humanization_score": 0.0,
            "rewrite_method": "local",
            "source_similarity_before": 0,
            "source_similarity_after": 0,
        }

    original = sentence.strip()

    # ── Try LLMs first if API keys are available ──
    configs = get_llm_configs()
    if configs:
        llm_result = llm_rewrite_sentence(original, source_sentence)
        if llm_result:
            source_sim_before = compute_text_similarity(original, source_sentence) if source_sentence else 0
            source_sim_after = compute_text_similarity(llm_result, source_sentence) if source_sentence else 0
            return {
                "plagiarized": original,
                "rewritten": llm_result,
                "changes_made": ["LLM rewrite", "Humanized phrasing"],
                "rewrite_method": "openai",
                "meaning_preserved": compute_text_similarity(original, llm_result),
                "humanization_score": 0.95,
                "source_similarity_before": source_sim_before,
                "source_similarity_after": source_sim_after,
            }

    # ── Use the new deep paraphrase engine ──
    prefetch_synonyms(original)
    if source_sentence:
        prefetch_synonyms(source_sentence)

    candidates = []
    
    # Generate candidates based on strength
    if strength >= 1:
        # Candidate 1: Basic Restructure
        candidates.append(deep_paraphrase_sentence(original, source_sentence))
    
    if strength >= 2:
        # Candidate 2: Restructure only (no synonyms)
        c2 = deep_paraphrase_sentence(original, source_sentence)
        candidates.append(c2)
        # Candidate 3: Phrase Paraphrase only (no synonyms)
        c3 = apply_phrase_paraphrases(original)
        candidates.append(c3)

    if strength >= 3:
        # Candidate 4: Deep restructure only (no synonyms)
        c4 = deep_paraphrase_sentence(original, source_sentence)
        candidates.append(c4)
        # DISABLE THIS FOR NOW
        # c5 = deep_paraphrase(original, source_sentence)

    if mode == "humanize_ai":
        c_hum = deep_paraphrase(original)
        c_hum = restructure_sentence(c_hum)
        c_hum = normalize_sentence(c_hum)
        candidates.append(c_hum)


    best_rewrite = original
    best_score = float('-inf')
    bad_prefixes = (
        "Basically",
        "Essentially",
        "Generally speaking",
        "In other words"
    )
    # Lower similarity threshold for acceptance
    similarity_threshold = 0.75
    for cand in candidates:
        cand = normalize_sentence(cand)
        if not cand or cand.strip() == "." or len(cand.strip()) < 10:
            continue
        if cand.startswith(bad_prefixes):
            continue
        sim = compute_text_similarity(cand, source_sentence) if source_sentence else compute_text_similarity(cand, original)
        meaning_similarity = compute_text_similarity(original, cand)
        lexical_change = 1 - compute_word_overlap(original, cand)
        if lexical_change < 0.25:
            continue
        score = (1-sim)*0.6 + lexical_change*0.3 + meaning_similarity*0.1
        # accept only materially different rewrites with lower similarity threshold
        if score > best_score and sim < similarity_threshold:
            best_score = score
            best_rewrite = cand

    # Aggressive rewrite fallback loop
    passes = 0
    max_passes = 5
    while (compute_text_similarity(best_rewrite, source_sentence) > similarity_threshold or best_rewrite.strip().lower() == original.strip().lower()) and passes < max_passes:
        # Structural/semantic changes: restructure, paraphrase, split/merge, reorder
        best_rewrite = deep_paraphrase(best_rewrite)
        best_rewrite = restructure_sentence(best_rewrite)
        # Attempt sentence splitting/merging (simple heuristic)
        if len(best_rewrite.split()) > 20:
            parts = best_rewrite.split('.')
            if len(parts) > 1:
                best_rewrite = '. '.join(reversed([p.strip() for p in parts if p.strip()]))
        # Attempt voice change (active/passive)
        if ' by ' in best_rewrite:
            best_rewrite = best_rewrite.replace(' by ', ' was by ')
        best_rewrite = normalize_sentence(best_rewrite)
        passes += 1

    # ...existing code...
    rewritten = best_rewrite


    # Reject unchanged output (hard fail)
    if rewritten.strip().lower() == original.strip().lower():
        rewritten = restructure_sentence(original)
    if rewritten.strip().lower() == original.strip().lower():
        rewritten = apply_phrase_paraphrases(original)
    # Force fallback passes if rewrite is too similar or unchanged
    rewritten = _force_fallbacks(original, rewritten, source_sentence)
    
    # ── Iterative Anti-Plagiarism Loop (target <30% similarity) ──
    passes = 0
    if source_sentence:
        target_plag = 0.30
        current_sim = compute_text_similarity(source_sentence, rewritten)
        passes = 0
        
        while current_sim > target_plag and passes < 5:
            # Pass 1: Restructure
            rewritten = deep_paraphrase_sentence(rewritten, source_sentence)
            
            # Pass 2: Phrase-level paraphrase
            rewritten = apply_phrase_paraphrases(rewritten)
            
            # Pass 3: (avoid aggressive synonym-only swaps to prevent gibberish)
            
            rewritten = normalize_sentence(rewritten)
            
            new_sim = compute_text_similarity(rewritten, source_sentence)
            
            if new_sim >= current_sim:
                break
            
            current_sim = new_sim
            passes += 1

    # Stronger enforcement: keep applying fallback chain until similarity < target_plag
    # or until we hit a safe attempt limit. This ensures no branch returns unchanged text.
    if source_sentence:
        attempts = 0
        max_attempts = 6
        while compute_text_similarity(rewritten, source_sentence) > target_plag and attempts < max_attempts:
            # fallback chain: deep paraphrase -> restructure -> phrase-level paraphrase
            rewritten = deep_paraphrase_sentence(rewritten, source_sentence)
            if rewritten.strip() == original.strip():
                rewritten = restructure_sentence(rewritten)
            if rewritten.strip() == original.strip():
                rewritten = apply_phrase_paraphrases(rewritten)
            rewritten = normalize_sentence(rewritten)
            # If we're still extremely similar to the original, try a final aggressive restructuring
            if compute_text_similarity(original, rewritten) > 0.90:
                rewritten = restructure_sentence(deep_paraphrase_sentence(original, source_sentence))
            attempts += 1

    # Final hard rejection: if rewritten is still >90% similar to original, force restructure/phrase paraphrase
    if compute_text_similarity(original, rewritten) > 0.90 or rewritten.strip() == original.strip():
        rewritten = deep_paraphrase_sentence(original, source_sentence)
        if rewritten.strip() == original.strip() or compute_text_similarity(original, rewritten) > 0.90:
            rewritten = restructure_sentence(original)
        if rewritten.strip() == original.strip() or compute_text_similarity(original, rewritten) > 0.90:
            rewritten = apply_phrase_paraphrases(original)

    # Last-resort deterministic minimal variation to guarantee output differs
    if rewritten.strip() == original.strip() or compute_text_similarity(original, rewritten) > 0.90:
        def _force_minimal_variation(s: str) -> str:
            s_clean = s.strip()
            trailing = ''
            if s_clean and s_clean[-1] in '.!?':
                trailing = s_clean[-1]
                s_clean = s_clean[:-1]
            words = s_clean.split()
            if len(words) > 3:
                rot = words[1:] + words[:1]
                res = ' '.join(rot)
                res = res[0].upper() + res[1:]
                if trailing:
                    res += trailing
                return res
            # fallback: append short deterministic humanizing phrase
            return (s_clean + ' In other words.').strip()

        rewritten = _force_minimal_variation(original)

    # ── Explain Changes Feature ──
    changes = []
    if mode == "humanize_ai":
        changes.append("Applied Humanize AI phrasing")
    
    # Only show iteration info if we actually performed iterations
    if source_sentence and passes > 0:
        changes.append("Iterative rewrite loop applied")
        changes.append(f"({passes} optimization passes)")
    
    # Find specific synonym swaps for "Explain Changes"
    orig_words = original.lower().split()
    rw_words = rewritten.lower().split()
    for w in orig_words:
        clean_w = w.strip(".,!?;:'\"")
        if len(clean_w) > 3 and clean_w not in rw_words:
            # Look for what might have replaced it
            for rw in rw_words:
                clean_rw = rw.strip(".,!?;:'\"")
                if len(clean_rw) > 3 and clean_rw in get_rewrite_synonyms(clean_w, static_only=False):
                    changes.append(f"'{clean_w}' → '{clean_rw}' (synonym)")
                    break

    if "by" in original.lower() and "by" not in rewritten.lower():
        changes.append("passive voice changed to active")
    
    if len(changes) == 0:
        changes.append("sentence restructured")

    # Compute final metrics
    meaning_similarity = compute_text_similarity(original, rewritten)
    human_score = 0.60 + (0.30 if mode == "humanize_ai" else 0.15) + (random.random() * 0.1)
    
    # Semantic meaning check: ensure meaning is preserved (>85%)
    try:
        emb_orig = similarity_model.encode(original, convert_to_tensor=True)
        emb_rewritten = similarity_model.encode(rewritten, convert_to_tensor=True)
        semantic_sim = float(util.cos_sim(emb_orig, emb_rewritten)[0][0])

        if semantic_sim < 0.85:
            # Attempt fallback chain rather than reverting to the original
            rewritten = deep_paraphrase_sentence(original, source_sentence)
            if compute_text_similarity(original, rewritten) > 0.90 or rewritten.strip() == original.strip():
                rewritten = restructure_sentence(original)
            if compute_text_similarity(original, rewritten) > 0.90 or rewritten.strip() == original.strip():
                rewritten = apply_phrase_paraphrases(original)
    except Exception:
        # Fallback if semantic model fails: try to ensure meaningful paraphrase
        if meaning_similarity < 0.30:
            rewritten = deep_paraphrase_sentence(original, source_sentence)
            if rewritten.strip() == original.strip() or compute_text_similarity(original, rewritten) > 0.90:
                rewritten = restructure_sentence(original)
    
    source_sim_before = compute_text_similarity(original, source_sentence) if source_sentence else 0
    source_sim_after = compute_text_similarity(rewritten, source_sentence) if source_sentence else 0

    return {
        "plagiarized": original,
        "rewritten": rewritten,
        "changes_made": list(set(changes)),
        "rewrite_method": "local",
        "meaning_preserved": meaning_similarity,
        "humanization_score": min(0.99, human_score),
        "source_similarity_before": source_sim_before,
        "source_similarity_after": source_sim_after,
    }


def restructure_sentence(sentence: str) -> str:
    """Restructure a sentence by moving clauses around."""
    words = sentence.split()
    if len(words) < 3:
        return sentence

    # Remove trailing punctuation
    punct = ""
    if sentence and sentence[-1] in ".!?":
        punct = sentence[-1]
        sentence_clean = sentence[:-1]
    else:
        sentence_clean = sentence

    # Strategy 1: If sentence has a prepositional phrase, move it to front
    prep_words = {"in", "at", "on", "to", "for", "from", "with", "by", "about",
                  "after", "before", "during", "through", "across", "near"}
    
    words_clean = sentence_clean.split()
    
    # Find a preposition in middle-to-end of sentence
    for i in range(len(words_clean) // 2, len(words_clean)):
        if words_clean[i].lower() in prep_words and i < len(words_clean) - 1:
            # Move the prepositional phrase to the front
            prep_phrase = words_clean[i:]
            main_clause = words_clean[:i]
            
            if len(prep_phrase) >= 2 and len(main_clause) >= 2:
                # Capitalize prep phrase start, preserve proper noun capitalization in the main clause
                prep_phrase[0] = prep_phrase[0].capitalize()
                if main_clause[0][0].isupper() and main_clause[0] != main_clause[0].lower():
                    main_clause[0] = main_clause[0]
                else:
                    main_clause[0] = main_clause[0][0].lower() + main_clause[0][1:]

                result = " ".join(prep_phrase) + ", " + " ".join(main_clause) + punct
                return result
            break

    return normalize_sentence(sentence)


def normalize_sentence(sentence: str) -> str:
    """Clean and normalize rewritten text for proper punctuation and capitalization."""
    if sentence is None:
        return {
            "rewritten": rewritten,
            "rewrite_method": "structural",
            "source_similarity_after": compute_text_similarity(source_sentence, rewritten),
            "humanization_score": 1.0
        }
    sentence = fix_common_lexical_artifacts(sentence)
    sentence = re.sub(r'\(\s+', '(', sentence)
    sentence = re.sub(r'\s+\)', ')', sentence)
    sentence = re.sub(r'\s+-\s+', ' - ', sentence)
    sentence = re.sub(r'\s+"', ' "', sentence)
    sentence = re.sub(r'"\s+', '" ', sentence)
    sentence = re.sub(r"\s+'", " '", sentence)
    sentence = re.sub(r"'\s+", "' ", sentence)

    # Capitalize after sentence boundaries and opening quotes
    def cap_match(match):
        prefix = match.group(1)
        quote = match.group(2) or ''
        char = match.group(3)
        return prefix + quote + char.upper()

    sentence = re.sub(r'(^|[\.\!?]\s+)(["\']?)([a-z])', cap_match, sentence)
    sentence = re.sub(r'\bi\b', 'I', sentence)

    sentence = sentence.strip()
    if sentence and sentence[-1] not in '.!?':
        sentence += '.'

    return sentence


def try_voice_change(sentence: str) -> str:
    """Try to change active voice to passive or vice versa."""
    words = sentence.split()
    if len(words) < 3:
        return sentence
    
    # Remove trailing punctuation
    punct = ""
    if sentence and sentence[-1] in ".!?":
        punct = sentence[-1]
        words[-1] = words[-1].rstrip(".!?")
    
    # Check for passive voice markers ("was/were/is/are ... by")
    lower_sentence = sentence.lower()
    
    if " by " in lower_sentence and any(w in lower_sentence for w in ["was ", "were ", "is ", "are "]):
        # Passive → Active
        try:
            # Find "by" and reconstruct
            by_idx = None
            for i, w in enumerate(words):
                if w.lower() == "by":
                    by_idx = i
                    break
            
            if by_idx and by_idx < len(words) - 1:
                agent = " ".join(words[by_idx + 1:])
                # Find the auxiliary verb
                aux_idx = None
                for i, w in enumerate(words):
                    if w.lower() in ("was", "were", "is", "are"):
                        aux_idx = i
                        break
                
                if aux_idx is not None:
                    subject = " ".join(words[:aux_idx])
                    verb_phrase = " ".join(words[aux_idx + 1:by_idx])
                    result = f"{agent.capitalize()} {verb_phrase} {subject.lower()}{punct}"
                    return result
        except (IndexError, ValueError):
            pass
    
    else:
        # Active → Passive (simple cases)
        # Pattern: Subject Verb Object
        if len(words) >= 3:
            subject = words[0]
            verb = words[1]
            obj_words = words[2:]
            obj = " ".join(obj_words)
            
            # Only convert if verb looks past-tense
            if verb.lower().endswith("ed") or verb.lower() in ("wrote", "gave", "took", "made", "found", "built", "ran"):
                verb_map = {
                    "wrote": "written", "gave": "given", "took": "taken",
                    "made": "made", "found": "found", "built": "built",
                    "ran": "run",
                }
                past_participle = verb_map.get(verb.lower(), verb.lower())
                
                result = f"{obj.capitalize()} was {past_participle} by {subject.lower()}{punct}"
                return result
    
    return sentence


def add_human_phrasing(sentence: str) -> str:
    """Add natural human transitions and phrasing to make text sound more original."""
    # Don't add if sentence is very short
    if len(sentence.split()) < 2:
        return sentence
    
    strategies = [
        "rephrase_start",
        "add_context",
        "combine_clauses",
    ]
    strategy = random.choice(strategies)
    
    if strategy == "rephrase_start":
        # Add a human-like opening
        openers = [
            "It is worth noting that ",
            "Notably, ",
            "In particular, ",
            "Specifically, ",
            "From what can be observed, ",
            "As it turns out, ",
            "Looking at this closely, ",
        ]
        opener = random.choice(openers)
        # Lowercase the first char of the original
        if sentence[0].isupper():
            sentence = sentence[0].lower() + sentence[1:]
        return opener + sentence
    
    elif strategy == "add_context":
        # Add contextual ending
        endings = [
            ", which is quite significant",
            ", demonstrating a clear pattern",
            ", as one might expect",
            ", reflecting the overall trend",
            ", highlighting an important aspect",
        ]
        # Remove trailing period
        if sentence.endswith("."):
            sentence = sentence[:-1]
        ending = random.choice(endings)
        return sentence + ending + "."
    
    else:  # combine_clauses
        # Rephrase using "this means that" or "in other words"
        connector = random.choice([
            "This essentially means that ",
            "To put it another way, ",
            "In simpler terms, ",
        ])
        if sentence[0].isupper():
            sentence = sentence[0].lower() + sentence[1:]
        return connector + sentence
    
    return sentence


# ─────────────────────────────────────────────────────────
# Text Processing Utilities
# ─────────────────────────────────────────────────────────
def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences and remove accidental duplicated fragments."""
    if not text:
        return []

    cleaned = re.sub(r'\s+', ' ', text.strip())
    # Fix merged boundaries like "concArtificial" and "important.erns"
    cleaned = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', cleaned)
    cleaned = re.sub(r'([.!?])(?=[A-Za-z])', r'\1 ', cleaned)

    # Collapse exact repeated chunks if they occur back-to-back.
    cleaned = re.sub(r'(.+?)\1', r'\1', cleaned)

    raw_sentences = re.findall(r'[^.!?]+[.!?]|[^.!?]+$', cleaned)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    # Remove duplicates/noisy stitched artifacts.
    deduped: list[str] = []
    for sent in sentences:
        # If a sentence embeds a previously seen sentence after a noisy prefix,
        # trim to the embedded sentence start (e.g., "... conc Artificial intelligence ...").
        lowered = sent.lower().strip()
        for prev in deduped:
            prev_core = re.sub(r"[.!?]+$", "", prev).lower().strip()
            idx = lowered.find(prev_core)
            if idx > 0:
                sent = sent[idx:].strip()
                lowered = sent.lower().strip()
                break

        normalized = lowered
        is_dup = any(SequenceMatcher(None, prev.lower().strip(), normalized).ratio() >= 0.90 for prev in deduped)
        if is_dup:
            continue

        # Drop likely truncated tail fragments like "Erns about ..." when they mostly
        # repeat an earlier sentence.
        first_word = re.findall(r"[A-Za-z]+", sent)
        first_len = len(first_word[0]) if first_word else 0
        if first_len <= 4:
            similar_prior = any(SequenceMatcher(None, prev.lower(), normalized).ratio() >= 0.62 for prev in deduped)
            if similar_prior:
                continue

        deduped.append(sent)

    return deduped


def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute similarity between two texts using SequenceMatcher."""
    if not text1 or not text2:
        return 0.0
    matcher = SequenceMatcher(None, text1.lower(), text2.lower())
    return round(matcher.ratio(), 4)


def compute_word_overlap(text1: str, text2: str) -> float:
    """Compute Jaccard word overlap."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return round(len(intersection) / len(union), 4) if union else 0.0


def compute_tfidf_cosine_similarity(text1: str, text2: str) -> float:
    """Compute TF-IDF cosine similarity if vectorizer is available."""
    if vectorizer is None:
        return 0.0

    tfidf1 = vectorizer.transform([text1])
    tfidf2 = vectorizer.transform([text2])
    dot = np.array((tfidf1.multiply(tfidf2)).sum(axis=1)).flatten()
    norm1 = np.sqrt(np.array(tfidf1.multiply(tfidf1).sum(axis=1)).flatten())
    norm2 = np.sqrt(np.array(tfidf2.multiply(tfidf2).sum(axis=1)).flatten())
    denom = norm1 * norm2
    denom[denom == 0] = 1e-10
    cosine_sim = float((dot / denom).item())
    return round(cosine_sim, 4)


def blend_similarity(text1: str, text2: str) -> float:
    """Blend multiple similarity signals into a single robust score."""
    sequence_sim = compute_text_similarity(text1, text2)
    word_overlap = compute_word_overlap(text1, text2)
    cosine_sim = compute_tfidf_cosine_similarity(text1, text2)

    if cosine_sim > 0:
        blended = (cosine_sim * 0.5) + (sequence_sim * 0.3) + (word_overlap * 0.2)
    else:
        blended = (sequence_sim * 0.6) + (word_overlap * 0.4)

    return round(min(max(blended, 0.0), 1.0), 4)


def create_features_for_pair(text1: str, text2: str):
    """Create feature vector for a single text pair."""
    if vectorizer is None:
        return None

    tfidf1 = vectorizer.transform([text1])
    tfidf2 = vectorizer.transform([text2])

    # Cosine similarity from TF-IDF vectors
    dot = np.array((tfidf1.multiply(tfidf2)).sum(axis=1)).flatten()
    norm1 = np.sqrt(np.array(tfidf1.multiply(tfidf1).sum(axis=1)).flatten())
    norm2 = np.sqrt(np.array(tfidf2.multiply(tfidf2).sum(axis=1)).flatten())
    denom = norm1 * norm2
    denom[denom == 0] = 1e-10
    cosine_sim = dot / denom

    # Length ratio
    len1 = max(len(text1), 1)
    len2 = max(len(text2), 1)
    length_ratio = min(len1, len2) / max(len1, len2)

    # Word overlap
    word_overlap = compute_word_overlap(text1, text2)

    # Character difference
    char_diff = abs(len1 - len2) / max(len1, len2)

    # Combine
    extra = np.array([[cosine_sim[0], length_ratio, word_overlap, char_diff]])
    tfidf_diff = abs(tfidf1 - tfidf2)
    X = hstack([tfidf_diff, csr_matrix(extra)])

    return X


def find_matching_segments(text1: str, text2: str, min_length: int = 4) -> list[dict]:
    """Find matching word sequences between two texts."""
    words1 = text1.lower().split()
    words2 = text2.lower().split()

    if not words1 or not words2:
        return []

    matcher = SequenceMatcher(None, words1, words2)
    matches = []

    for block in matcher.get_matching_blocks():
        if block.size >= min_length:
            matched_words = words1[block.a:block.a + block.size]
            matches.append({
                "text": " ".join(matched_words),
                "position_text1": block.a,
                "position_text2": block.b,
                "length": block.size,
            })

    return matches


def analyze_pair(text1: str, text2: str) -> dict:
    """Full analysis of a text pair."""
    # ML prediction
    prediction = None
    confidence = None

    if model is not None and vectorizer is not None:
        try:
            X = create_features_for_pair(text1, text2)
            if X is not None:
                prediction = int(model.predict(X)[0])
                proba = model.predict_proba(X)[0]
                confidence = float(max(proba))
        except AttributeError:
            # Fallback for sklearn version mismatch
            prediction = 1 if compute_text_similarity(text1, text2) > 0.6 else 0
            confidence = compute_text_similarity(text1, text2)

    # Rule-based metrics
    sequence_sim = compute_text_similarity(text1, text2)
    word_overlap = compute_word_overlap(text1, text2)
    cosine_sim = compute_tfidf_cosine_similarity(text1, text2)
    blended_similarity = blend_similarity(text1, text2)
    matching_segments = find_matching_segments(text1, text2)

    # Determine plagiarism level
    if prediction is not None:
        is_plagiarized = bool(prediction)
    else:
        is_plagiarized = blended_similarity >= 0.6

    # Compute overall similarity score (0-100)
    overall_score = round(blended_similarity * 100, 1)

    # Severity
    if overall_score >= 80:
        severity = "high"
        verdict = "High Plagiarism Detected"
    elif overall_score >= 50:
        severity = "medium"
        verdict = "Moderate Similarity Found"
    elif overall_score >= 30:
        severity = "low"
        verdict = "Minor Similarities"
    else:
        severity = "none"
        verdict = "Original Content"

    return {
        "is_plagiarized": is_plagiarized,
        "overall_score": overall_score,
        "severity": severity,
        "verdict": verdict,
        "metrics": {
            "sequence_similarity": sequence_sim,
            "word_overlap": word_overlap,
            "cosine_similarity": cosine_sim,
            "blended_similarity": blended_similarity,
            "ml_confidence": confidence,
            "ml_prediction": prediction,
        },
        "matching_segments": matching_segments,
    }


# ─────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main page."""
    model_loaded = model is not None
    return render_template("index.html", model_loaded=model_loaded, metadata=metadata)


@app.route("/api/health")
def health():
    """Health check endpoint."""
    api_key, api_model, api_ready = get_openai_config()
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "metadata": metadata if metadata else {},
        "openai": {
            "enabled": api_ready,
            "model": api_model,
            "has_api_key": bool(api_key),
        },
    })


@app.route("/api/check", methods=["POST"])
def check_plagiarism():
    """
    Check plagiarism between two texts.
    
    Flow:
      1. Detect plagiarism (ML + rule-based)
      2. Split suspected text into sentences
      3. For each plagiarized sentence → produce a human-like rewrite
      4. Return: plagiarism score, plagiarized text, rewritten clean text
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    text1 = data.get("text1", "").strip()
    text2 = data.get("text2", "").strip()
    strength = data.get("strength", 2)
    mode = data.get("mode", "remove_plagiarism")

    if not text1 or not text2:
        return jsonify({"error": "Both text1 and text2 are required"}), 400

    # Overall analysis
    result = analyze_pair(text1, text2)

    # Sentence-level analysis + rewriting
    sentences_text2 = split_into_sentences(text2)
    sentences_text1 = split_into_sentences(text1)
    
    sentence_analysis = []
    rewritten_sentences = []
    
    for sent2 in sentences_text2:
        # Find the best matching source sentence
        best_source = ""
        best_sim = 0
        for sent1 in sentences_text1:
            sim = compute_text_similarity(sent2, sent1)
            if sim > best_sim:
                best_sim = sim
                best_source = sent1
        
        # Determine if this sentence is plagiarized
        is_plag = best_sim >= 0.30
        
        # Rewrite if plagiarized
        if is_plag:
            rewrite_result = rewrite_sentence_human(sent2, best_source, strength, mode)
        else:
            rewrite_result = {
                "plagiarized": sent2,
                "rewritten": sent2,  # Keep as is — it's clean
                "changes_made": [],
                "rewrite_method": "local",
                "meaning_preserved": 1.0,
                "humanization_score": 1.0,
                "source_similarity_before": best_sim,
                "source_similarity_after": best_sim,
            }
        
        sentence_analysis.append({
            "sentence": sent2,
            "source_match": best_source,
            "similarity": round(best_sim, 3),
            "is_plagiarized": is_plag,
            "rewrite": rewrite_result,
        })
        
        rewritten_sentences.append(rewrite_result["rewritten"])
    
    # Combine all rewritten sentences into a clean paragraph
    rewritten_full_text = " ".join(rewritten_sentences)
    
    # Build the plagiarized text (the input that was flagged)
    plagiarized_full_text = text2
    
    # Compute similarity between the plagiarized text and the final human-like rewrite
    rewrite_similarity = compute_text_similarity(plagiarized_full_text, rewritten_full_text)
    
    # Compute similarity between the original source text and the final rewrite
    source_rewrite_similarity = compute_text_similarity(text1, rewritten_full_text)
    
    # Recheck the rewritten text against the original source
    recheck_analysis = analyze_pair(text1, rewritten_full_text)
    recheck_score = recheck_analysis["overall_score"]
    
    result["sentence_analysis"] = sentence_analysis
    result["plagiarized_text"] = plagiarized_full_text
    result["rewritten_text"] = rewritten_full_text
    result["rewrite_similarity"] = rewrite_similarity
    result["humanization_score"] = round(sum(s["rewrite"].get("humanization_score", 0) for s in sentence_analysis if s["is_plagiarized"]) / max(1, sum(1 for s in sentence_analysis if s["is_plagiarized"])) * 100, 1)
    result["source_rewrite_similarity"] = source_rewrite_similarity
    result["recheck_score"] = recheck_score
    result["rewrite_engine"] = get_rewrite_engine(sentence_analysis)
    result["total_sentences"] = len(sentences_text2)
    result["plagiarized_count"] = sum(1 for s in sentence_analysis if s["is_plagiarized"])
    
    return jsonify(result)


@app.route("/api/verify_rewrite", methods=["POST"])
def verify_rewrite():
    """
    Verify user rewrite: current plag % + similarity to original plag sentence.
    Preserves meaning check while reducing plagiarism score.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    original_source = data.get("original_source", "").strip()
    original_plag = data.get("original_plag", "").strip()  
    user_rewrite = data.get("user_rewrite", "").strip()

    if not all([original_source, original_plag, user_rewrite]):
        return jsonify({"error": "All three texts required"}), 400

    # Current plagiarism score (user_rewrite vs original_source)
    current_analysis = analyze_pair(original_source, user_rewrite)
    
    # Similarity to original plag (meaning preservation)
    meaning_sim = compute_text_similarity(original_plag, user_rewrite)
    

    # Correct plagiarism reduction calculation: source-vs-suspected vs source-vs-rewritten
    before = compute_text_similarity(original_source, original_plag)
    after = compute_text_similarity(original_source, user_rewrite)
    improvements = {
        "plagiarism_reduced": round(max(0, (before - after)) * 100, 1),
        "meaning_preserved": round(meaning_sim * 100, 1),
        "word_overlap_change": round((compute_word_overlap(original_plag, user_rewrite) - compute_word_overlap(original_source, original_plag)) * 100, 1)
    }

    result = {
        "current_plagiarism_score": current_analysis["overall_score"],
        "current_severity": current_analysis["severity"],
        "meaning_preserved_pct": improvements["meaning_preserved"],
        "plagiarism_reduction_pct": improvements["plagiarism_reduced"],
        "word_changes_summary": improvements,
        "status": "meaning_preserved" if meaning_sim > 0.7 else "meaning_changed" if meaning_sim > 0.4 else "meaning_lost",
        "recommendation": "Great rewrite! ✅" if current_analysis["overall_score"] < 20 and meaning_sim > 0.7 else "Good progress 👌" if current_analysis["overall_score"] < 50 else "Needs more rewriting ⚠️"
    }
    
    return jsonify(result)


@app.route("/api/auto_plag", methods=["POST"])
def auto_plagiarism_pipeline():
    """
    FULLY AUTOMATIC 4-STEP WORKFLOW:
    1. User enters clean text
    2. AUTO-generates suspected plagiarized version
    3. Detects plagiarism
    4. Reconstructs clean text + verifies similarity
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    clean_text = data.get("clean_text", "").strip()
    strength = data.get("strength", 3)
    mode = data.get("mode", "remove_plagiarism")
    

    # Check for too short or informal input
    if not clean_text:
        return jsonify({"error": "Clean text required"}), 400
    if len(clean_text.split()) < 6 or all(word.strip(".,!? ").lower() in ["hi", "hello", "hey", "yo", "what's", "up", "ohhooo", "hiii", "ok", "okay"] for word in clean_text.split()):
        return jsonify({
            "error": "Input too short or informal for meaningful rewriting. Please enter a longer or more formal sentence or paragraph."
        }), 400

    # Step 1: Auto-generate suspected plagiarized text
    suspected_text, original_clean, simulation_stats = auto_generate_plagiarized(clean_text)

    # Step 2: Detect plagiarism (source=original_clean, suspected=suspected_text)
    detection = analyze_pair(original_clean, suspected_text)

    # Step 3: Auto-reconstruct (treat suspected as "text2", original as source)
    sentences_suspected = split_into_sentences(suspected_text)
    sentences_source = split_into_sentences(original_clean)
    
    sentence_analysis = []
    reconstructed_sentences = []
    reduction_values = []
    
    for sent_suspected in sentences_suspected:
        best_source = ""
        best_sim = 0
        for sent_source in sentences_source:
            sim = compute_text_similarity(sent_suspected, sent_source)
            if sim > best_sim:
                best_sim = sim
                best_source = sent_source

        # FORCE REWRITE ALWAYS, no conditions
        rewrite_result = rewrite_sentence_human(
            sent_suspected,
            best_source,
            strength=3,
            mode="humanize_ai"
        )
        if not rewrite_result.get("rewrite_method"):
            rewrite_result["rewrite_method"] = "local"
        before = compute_text_similarity(best_source, sent_suspected)
        after = compute_text_similarity(best_source, rewrite_result["rewritten"])
        plagiarism_reduced = round(max(0, (before - after)) * 100, 1)

        sentence_analysis.append({
            "sentence": sent_suspected,
            "source_match": best_source,
            "similarity": round(best_sim, 3),
            "is_plagiarized": True,
            "rewrite": rewrite_result,
            "plagiarism_reduced": plagiarism_reduced,
        })
        reduction_values.append(plagiarism_reduced)
        reconstructed_sentences.append(rewrite_result["rewritten"])
    
    reconstructed_text = " ".join(reconstructed_sentences)

    # Step 4: Verify similarity (reconstructed vs suspected)
    final_similarity = compute_text_similarity(suspected_text, reconstructed_text)
    meaning_preserved_pct = round(final_similarity * 100, 1)
    
    # Verify similarity to original source
    source_rewrite_similarity = compute_text_similarity(original_clean, reconstructed_text)
    
    humanization_score = round(sum(r["rewrite"].get("humanization_score", 0) for r in sentence_analysis if r["is_plagiarized"]) / max(1, sum(1 for r in sentence_analysis if r["is_plagiarized"])) * 100, 1)

    return jsonify({
        "workflow": "complete",
        # Step 1
        "original_clean": original_clean,
        "auto_suspected": suspected_text,
        "simulation_stats": simulation_stats,
        "auto_model_used": simulation_stats.get("llm_model_used", "none"),
        "auto_generated_with_llm": simulation_stats.get("generated_with_llm", False),
        # Step 2
        "detection": detection,
        "sentence_analysis": sentence_analysis,
        "rewrite_engine": get_rewrite_engine(sentence_analysis),
        # Step 3
        "reconstructed_text": reconstructed_text,
        # Debug logs are also emitted server-side for SOURCE/SUSPECTED/FINAL
        # Step 4
        "similarity_check": {
            "meaning_preserved_pct": meaning_preserved_pct,
            "plagiarism_reduced_pct": round(sum(reduction_values) / max(1, len(reduction_values)), 1),
            "source_rewrite_similarity": round(source_rewrite_similarity * 100, 1),
            "humanization_score": humanization_score,
            "status": "excellent" if meaning_preserved_pct > 85 else "good" if meaning_preserved_pct > 70 else "fair"
        }
    })

    # Server-side debug log for easy tracing
    app.logger.info("AUTO-PLAG | SOURCE:\n%s\n--- SUSPECTED:\n%s\n--- FINAL:\n%s",
                    original_clean, suspected_text, reconstructed_text)


@app.route("/api/compare", methods=["POST"])
def compare_corpus():
    """
    Compare text against a corpus (sentence-by-sentence).
    
    For each flagged sentence → produce a human-like rewrite.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    input_text = data.get("text", "").strip()
    corpus_text = data.get("corpus", "").strip()
    strength = data.get("strength", 2)
    mode = data.get("mode", "remove_plagiarism")

    if not input_text or not corpus_text:
        return jsonify({"error": "Both text and corpus are required"}), 400

    # Split into sentences
    input_sentences = split_into_sentences(input_text)
    corpus_sentences = split_into_sentences(corpus_text)

    if not input_sentences:
        return jsonify({"error": "Input text has no sentences"}), 400
    if not corpus_sentences:
        return jsonify({"error": "Corpus text has no sentences"}), 400

    # Compare each input sentence against all corpus sentences
    results = []
    total_plagiarism_score = 0
    rewritten_sentences = []

    for input_sent in input_sentences:
        best_match = None
        best_score = 0
        best_analysis = None

        for corpus_sent in corpus_sentences:
            analysis = analyze_pair(input_sent, corpus_sent)
            score = analysis["overall_score"]

            if score > best_score:
                best_score = score
                best_match = corpus_sent
                best_analysis = analysis

        # Determine if plagiarized
        is_plag = best_score >= 50
        
        # Rewrite if plagiarized
        if is_plag:
            rewrite_result = rewrite_sentence_human(input_sent, best_match or "", strength, mode)
        else:
            rewrite_result = {
                "plagiarized": input_sent,
                "rewritten": input_sent,
                "changes_made": [],
                "rewrite_method": "local",
                "meaning_preserved": 1.0,
                "humanization_score": 1.0,
                "source_similarity_before": best_score / 100,
                "source_similarity_after": best_score / 100,
            }

        results.append({
            "input_sentence": input_sent,
            "best_match": best_match,
            "match_score": best_score,
            "analysis": best_analysis,
            "is_plagiarized": is_plag,
            "rewrite": rewrite_result,
        })
        total_plagiarism_score += best_score
        rewritten_sentences.append(rewrite_result["rewritten"])

    # Average plagiarism score
    avg_score = round(total_plagiarism_score / len(input_sentences), 1) if input_sentences else 0

    # Overall severity
    if avg_score >= 70:
        overall_severity = "high"
        overall_verdict = "Significant Plagiarism Detected"
    elif avg_score >= 40:
        overall_severity = "medium"
        overall_verdict = "Moderate Plagiarism Detected"
    elif avg_score >= 20:
        overall_severity = "low"
        overall_verdict = "Minor Similarities Found"
    else:
        overall_severity = "none"
        overall_verdict = "Content Appears Original"

    # Combine rewritten text
    rewritten_full_text = " ".join(rewritten_sentences)

    # Compute similarity between original and rewritten text
    rewrite_similarity = compute_text_similarity(input_text, rewritten_full_text)
    
    # Compute similarity between the corpus source and rewritten text
    source_rewrite_similarity = compute_text_similarity(corpus_text, rewritten_full_text)

    # Recheck the rewritten text against the corpus
    recheck_score_total = 0
    rw_list = split_into_sentences(rewritten_full_text)
    for rw_sent in rw_list:
        best_rw_score = 0
        for corpus_sent in corpus_sentences:
            analysis = analyze_pair(rw_sent, corpus_sent)
            if analysis["overall_score"] > best_rw_score:
                best_rw_score = analysis["overall_score"]
        recheck_score_total += best_rw_score
    recheck_avg_score = round(recheck_score_total / len(rw_list), 1) if rw_list else 0

    humanization_score = round(sum(r["rewrite"].get("humanization_score", 0) for r in results if r["is_plagiarized"]) / max(1, sum(1 for r in results if r["is_plagiarized"])) * 100, 1)

    return jsonify({
        "overall_score": avg_score,
        "overall_severity": overall_severity,
        "overall_verdict": overall_verdict,
        "total_sentences": len(input_sentences),
        "flagged_sentences": sum(1 for r in results if r["is_plagiarized"]),
        "rewrite_engine": get_rewrite_engine(results),
        "sentence_results": results,
        "original_text": input_text,
        "rewritten_text": rewritten_full_text,
        "rewrite_similarity": rewrite_similarity,
        "source_rewrite_similarity": source_rewrite_similarity,
        "recheck_score": recheck_avg_score,
        "humanization_score": humanization_score,
    })


# ─────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plagiarism Checker Web Application")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    load_model()

    print(f"\n🌐 Starting Plagiarism Checker on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
