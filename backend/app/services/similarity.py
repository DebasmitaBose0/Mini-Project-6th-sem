from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('punkt')

def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]


def sentence_level_similarity(original, rewritten):
    orig_sent = nltk.sent_tokenize(original)
    rew_sent = nltk.sent_tokenize(rewritten)

    matches = []

    for o in orig_sent:
        for r in rew_sent:
            score = compute_similarity(o, r)
            if score > 0.6:
                matches.append({
                    "original": o,
                    "rewritten": r,
                    "score": score
                })

    return matches