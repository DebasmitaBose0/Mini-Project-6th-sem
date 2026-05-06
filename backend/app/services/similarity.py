from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

def compute_similarity(t1, t2):
    # TF-IDF Cosine Similarity (Structural/Keyword)
    vec = TfidfVectorizer(stop_words='english')
    try:
        v = vec.fit_transform([t1, t2])
        cos_sim = cosine_similarity(v[0], v[1])[0][0]
    except:
        cos_sim = 0.0

    # Sequence Matching (Literal similarity)
    seq = difflib.SequenceMatcher(None, t1, t2)
    seq_sim = seq.ratio()

    # Blend them: emphasis on sequence matching for "Plagiarism" detection
    return (cos_sim * 0.4) + (seq_sim * 0.6)

def sentence_level_similarity(t1, t2):
    s1 = t1.split('.')
    s2 = t2.split('.')
    matches = []
    for sent in s1:
        if sent.strip() and sent in t2:
            matches.append(sent.strip())
    return matches