from difflib import SequenceMatcher
import re

PRONOUN_MAP = {
    "I": "me",
    "me": "I",
    "my": "mine",
    "mine": "my",
    "she": "her",
    "her": "she",
    "he": "him",
    "him": "he",
    "they": "them",
    "them": "they",
    "we": "us",
    "us": "we"
}

ACTIVE_TO_PASSIVE_PATTERNS = [
    (r"^(?P<subj>\w+) (?P<verb>\w+) (?P<obj>.+)$", "{obj} is {verb} by {subj}.")
]


def sentence_similarity(a: str, b: str) -> float:
    matcher = SequenceMatcher(None, a.lower(), b.lower())
    return round(matcher.ratio(), 3)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [part.strip() for part in parts if part.strip()]


def transform_reorder(sentence: str) -> str:
    punctuation = ""
    if sentence and sentence[-1] in ".!?":
        punctuation = sentence[-1]
        sentence = sentence[:-1]
    words = sentence.split()
    if len(words) < 4:
        return sentence + punctuation
    pivot = len(words) // 2
    reordered = words[pivot:] + words[:pivot]
    return " ".join(reordered).capitalize() + punctuation


def transform_pronoun_change(sentence: str) -> str:
    tokens = sentence.rstrip(".?!").split()
    changed = [PRONOUN_MAP.get(word, PRONOUN_MAP.get(word.lower(), word)) for word in tokens]
    result = " ".join(changed)
    return result.capitalize() + (sentence[-1] if sentence and sentence[-1] in ".!?" else "")


def transform_active_to_passive(sentence: str) -> str:
    text = sentence.rstrip(".?!")
    words = text.split()
    if len(words) < 3:
        return sentence
    subject = words[0]
    verb = words[1]
    obj = " ".join(words[2:])
    if subject.lower() in ["i", "she", "he", "they", "we", "you"]:
        return f"{obj.capitalize()} is {verb} by {subject}."
    return sentence


def transform_passive_to_active(sentence: str) -> str:
    text = sentence.rstrip(".?!")
    if " by " not in text.lower():
        return sentence
    before_by, after_by = text.rsplit(" by ", 1)
    parts = before_by.split()
    if len(parts) < 2:
        return sentence
    verb = parts[1]
    subject = after_by.capitalize()
    obj = " ".join(parts[0:1] + parts[2:])
    return f"{subject} {verb} {obj}."


def rewrite_text(paragraph: str) -> dict:
    sentences = split_sentences(paragraph)
    suggestions = []
    for sentence in sentences:
        candidate = transform_reorder(sentence)
        pronoun = transform_pronoun_change(sentence)
        active_to_passive = transform_active_to_passive(sentence)
        passive_to_active = transform_passive_to_active(sentence)
        best = max(
            [candidate, pronoun, active_to_passive, passive_to_active],
            key=lambda s: sentence_similarity(sentence, s)
        )
        suggestions.append({
            "original": sentence,
            "reorder": candidate,
            "pronoun_change": pronoun,
            "active_to_passive": active_to_passive,
            "passive_to_active": passive_to_active,
            "best_rewrite": best,
            "similarity": sentence_similarity(sentence, best)
        })
    return {
        "paragraph": paragraph,
        "sentences": suggestions
    }


def print_rewrite_report(report: dict) -> None:
    print("\nOriginal paragraph:\n", report["paragraph"], "\n")
    for item in report["sentences"]:
        print("Original sentence:", item["original"])
        print("  Reorder:", item["reorder"])
        print("  Pronoun change:", item["pronoun_change"])
        print("  Active->Passive:", item["active_to_passive"])
        print("  Passive->Active:", item["passive_to_active"])
        print("  Best rewrite:", item["best_rewrite"])
        print("  Similarity score:", item["similarity"])
        print()


def main() -> None:
    print("Paragraph Rewriter for plagiarism-style rewriting")
    paragraph = input("Enter a paragraph or sentence: ").strip()
    if not paragraph:
        print("No input provided. Exiting.")
        return
    report = rewrite_text(paragraph)
    print_rewrite_report(report)


if __name__ == "__main__":
    main()
