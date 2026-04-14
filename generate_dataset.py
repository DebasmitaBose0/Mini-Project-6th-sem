import csv
import random
from difflib import SequenceMatcher

BASE_SENTENCES = [
    "My name is Monika.",
    "I wrote the letter.",
    "She loves music.",
    "They are playing football.",
    "He gave the book to me.",
    "The teacher teaches the class.",
    "The cat chased the mouse.",
    "He will finish the work tomorrow.",
    "She can answer the question.",
    "The child is drawing a picture."
]

TRANSFORMATION_TYPES = [
    "reorder",
    "active_to_passive",
    "passive_to_active",
    "pronoun_change"
]

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


def sentence_similarity(a: str, b: str) -> float:
    matcher = SequenceMatcher(None, a.lower(), b.lower())
    return round(matcher.ratio(), 3)


def transform_reorder(sentence: str) -> str:
    words = sentence.strip().rstrip(".").split()
    if len(words) < 4:
        return sentence
    pivot = len(words) // 2
    reordered = words[pivot:] + words[:pivot]
    result = " ".join(reordered)
    if not result.endswith("."):
        result += "."
    return result


def transform_pronoun_change(sentence: str) -> str:
    tokens = sentence.strip().rstrip(".").split()
    changed = [PRONOUN_MAP.get(word, PRONOUN_MAP.get(word.lower(), word)) for word in tokens]
    result = " ".join(changed)
    if not result.endswith("."):
        result += "."
    return result


def transform_active_to_passive(sentence: str) -> str:
    sentence = sentence.strip().rstrip(".")
    words = sentence.split()
    if len(words) < 4:
        return sentence + "."

    if words[1].lower() == "am" or words[1].lower() == "is" or words[1].lower() == "are":
        # simple copula case, just reverse elements
        return f"{words[-1]} {words[1]} {' '.join(words[2:-1])}."

    subject = words[0]
    verb = words[1]
    rest = words[2:]
    if len(rest) == 0:
        return sentence + "."

    if len(rest) == 1:
        return f"{rest[0].capitalize()} is {verb} by {subject}."

    obj = " ".join(rest)
    return f"{obj.capitalize()} is {verb} by {subject}."


def transform_passive_to_active(sentence: str) -> str:
    sentence = sentence.strip().rstrip(".")
    words = sentence.split()
    if " by " not in sentence.lower():
        return sentence + "."
    try:
        before_by, after_by = sentence.rsplit(" by ", 1)
        object_phrase = before_by
        subject_phrase = after_by
        if object_phrase.lower().startswith("the "):
            object_phrase = object_phrase[4:]
        return f"{subject_phrase.capitalize()} {words[2]} {object_phrase}."
    except ValueError:
        return sentence + "."


def generate_variations(base_sentence: str) -> list[dict]:
    variations = []
    transformations = [transform_reorder, transform_pronoun_change, transform_active_to_passive, transform_passive_to_active]
    labels = ["reorder", "pronoun_change", "active_to_passive", "passive_to_active"]

    for transform, label in zip(transformations, labels):
        transformed = transform(base_sentence)
        similarity = sentence_similarity(base_sentence, transformed)
        variations.append({
            "original_sentence": base_sentence,
            "transformed_sentence": transformed,
            "transformation_type": label,
            "similarity_score": similarity,
            "same_meaning": True
        })

    # Add one extra human-like rewrite for each base sentence
    rewritten = base_sentence.replace("My name is", "I am").replace("I wrote", "I have written")
    if rewritten != base_sentence:
        similarity = sentence_similarity(base_sentence, rewritten)
        variations.append({
            "original_sentence": base_sentence,
            "transformed_sentence": rewritten,
            "transformation_type": "human_rewrite",
            "similarity_score": similarity,
            "same_meaning": True
        })

    return variations


def save_dataset(rows: list[dict], filename: str = "dataset.csv") -> None:
    fieldnames = ["original_sentence", "transformed_sentence", "transformation_type", "similarity_score", "same_meaning"]
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    random.seed(42)
    dataset = []
    for sentence in BASE_SENTENCES:
        dataset.extend(generate_variations(sentence))

    save_dataset(dataset)
    print(f"Dataset generated with {len(dataset)} rows: dataset.csv")


if __name__ == "__main__":
    main()
