import argparse
import csv
import random
import re
from difflib import SequenceMatcher

PARAGRAPH_PATTERNS = [
    "{subject} {verb} {object} {detail}. {subject2} {verb2} {object2} {detail2}. {closing}.",
    "During {event}, {subject} {verb} {object} {detail}. {subject2} {verb2} {object2} {detail2}. {closing}.",
    "The {subject} {verb} {object} {detail}, and {subject2} {verb2} {object2} {detail2}. {closing}.",
    "{subject} {verb} {object} {detail}. {subject2} {verb2} {object2} {detail2}, and {subject3} {verb3} {object3} {detail3}. {closing}."
]

SUBJECTS = [
    "The research team",
    "Our marketing department",
    "The hotel manager",
    "The development team",
    "The city council",
    "The executive chef",
    "The professor",
    "The service crew",
    "The nonprofit",
    "The keynote speaker",
    "The clinical team",
    "The logistics provider",
    "The bank",
    "A university research group",
    "The construction crew",
    "The retail team",
    "The health clinic",
    "The publishing house",
    "The legal team",
    "The automotive manufacturer",
    "The operations staff",
    "The training team",
    "The outreach team",
    "The editorial board",
    "The facilities team"
]

VERBS = [
    "launched",
    "completed",
    "scheduled",
    "approved",
    "prepared",
    "assigned",
    "monitored",
    "organized",
    "emphasized",
    "conducted",
    "optimized",
    "introduced",
    "released",
    "tested",
    "coordinated",
    "redesigned",
    "reviewed",
    "finalized",
    "presented",
    "maintained"
]

OBJECTS = [
    "a new campaign",
    "a major report",
    "a software update",
    "a green space project",
    "a seasonal menu",
    "a research study",
    "the power lines",
    "a fundraiser",
    "a leadership seminar",
    "a clinical trial",
    "shipment routes",
    "an application upgrade",
    "a white paper",
    "an eco-friendly housing project",
    "the storefront layout",
    "a wellness program",
    "the publishing schedule",
    "the acquisition documents",
    "electric vehicles",
    "the regulatory submission"
]

DETAILS = [
    "with strong community support",
    "after careful review",
    "under tight deadlines",
    "with expert guidance",
    "using local resources",
    "with measurable performance goals",
    "to improve customer satisfaction",
    "while meeting compliance standards",
    "with reliable tracking systems",
    "through an inclusive planning process",
    "with minimal disruption",
    "supported by cross-functional teams",
    "using data-driven analysis",
    "to enhance public trust",
    "with an expanded support network",
    "to improve operational efficiency",
    "with a strong communications plan",
    "while preserving the budget",
    "with sustainability goals",
    "through collaborative decision-making"
]

EVENTS = [
    "the annual review",
    "the emergency meeting",
    "the product launch",
    "the regulatory audit",
    "a community forum",
    "the client briefing",
    "the software deployment",
    "the policy rollout",
    "the editorial review",
    "the board meeting"
]

CLOSINGS = [
    "The outcome exceeded expectations",
    "The results improved overall performance",
    "Stakeholders praised the final delivery",
    "The effort strengthened long-term relationships",
    "The project demonstrated clear value",
    "This approach set a new standard",
    "The initiative delivered measurable benefits",
    "The implementation received widespread acclaim",
    "The team achieved a strong outcome",
    "The final report received positive feedback"
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
    "us": "we",
    "our": "their",
    "their": "our",
    "theirs": "ours",
    "you": "you",
    "your": "your"
}

PHRASE_MAP = {
    "published their findings": "shared their research results",
    "careful experimentation": "methodical testing",
    "marketing department": "promotion team",
    "customer testimonials": "client feedback",
    "hotel manager": "property manager",
    "personal recommendations": "tailored suggestions",
    "development team": "engineering staff",
    "software update": "system release",
    "city council approved": "municipal leaders authorized",
    "green space project": "community park initiative",
    "executive chef": "head chef",
    "seasonal menu": "spring menu",
    "research sustainable energy solutions": "study renewable power options",
    "storm": "severe weather",
    "service crew": "maintenance team",
    "nonprofit organized": "charity arranged",
    "fundraiser": "benefit event",
    "annual conference": "yearly summit",
    "keynote speaker": "main presenter",
    "ethical leadership": "responsible management"
}

FORMAL_PHRASE_MAP = {
    "within days": "shortly thereafter",
    "left glowing reviews": "received positive feedback",
    "fast and efficient work": "rapid and effective work",
    "the client reported": "the client noted",
    "a balanced entrée": "a well-balanced entrée",
    "the class discussed": "the class examined",
    "restored electricity": "reestablished power",
    "raised enough funds": "generated sufficient funding",
    "after the session": "subsequent to the session",
    "shared their own stories": "recounted their own experiences",
    "goals": "objectives",
    "staff": "personnel",
    "project": "initiative",
    "results": "outcomes",
    "feedback": "response"
}

TRANSFORMATION_TYPES = [
    "sentence_reorder",
    "paragraph_pronoun_change",
    "phrase_level_rewrite",
    "human_like_rewrite",
    "formal_rewrite"
]


def sentence_similarity(a: str, b: str) -> float:
    matcher = SequenceMatcher(None, a.lower(), b.lower())
    return round(matcher.ratio(), 3)


def split_sentences(paragraph: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    return [part.strip() for part in parts if part.strip()]


def transform_sentence_reorder(paragraph: str) -> str:
    sentences = split_sentences(paragraph)
    if len(sentences) < 2:
        return paragraph
    reordered = sentences[1:] + sentences[:1]
    return " ".join(reordered)


def transform_paragraph_pronoun_change(paragraph: str) -> str:
    def replace_word(word: str) -> str:
        stripped = re.sub(r"[^A-Za-z']", "", word)
        mapped = PRONOUN_MAP.get(stripped, PRONOUN_MAP.get(stripped.lower()))
        if mapped:
            return word.replace(stripped, mapped)
        return word

    return " ".join(replace_word(word) for word in paragraph.split())


def transform_phrase_level(paragraph: str) -> str:
    text = paragraph
    for phrase, replacement in PHRASE_MAP.items():
        text = re.sub(re.escape(phrase), replacement, text, flags=re.IGNORECASE)
    return text


def transform_human_like(paragraph: str) -> str:
    sentences = split_sentences(paragraph)
    variations = []
    for sentence in sentences:
        sentence = sentence.replace("It was", "The experience was")
        sentence = sentence.replace("They", "The team")
        sentence = sentence.replace("She", "The manager")
        sentence = sentence.replace("He", "The lead")
        sentence = sentence.replace("We", "Our team")
        sentence = sentence.replace("I ", "My team ")
        sentence = sentence.replace("This ", "The following ")
        variations.append(sentence)
    return " ".join(variations)


def transform_formal_rewrite(paragraph: str) -> str:
    text = paragraph
    for phrase, replacement in FORMAL_PHRASE_MAP.items():
        text = re.sub(re.escape(phrase), replacement, text, flags=re.IGNORECASE)
    text = text.replace("goals", "objectives")
    text = text.replace("staff", "personnel")
    return text


def clean_paragraph(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def build_paragraph() -> str:
    pattern = random.choice(PARAGRAPH_PATTERNS)
    values = {
        "subject": random.choice(SUBJECTS),
        "verb": random.choice(VERBS),
        "object": random.choice(OBJECTS),
        "detail": random.choice(DETAILS),
        "subject2": random.choice(SUBJECTS),
        "verb2": random.choice(VERBS),
        "object2": random.choice(OBJECTS),
        "detail2": random.choice(DETAILS),
        "subject3": random.choice(SUBJECTS),
        "verb3": random.choice(VERBS),
        "object3": random.choice(OBJECTS),
        "detail3": random.choice(DETAILS),
        "event": random.choice(EVENTS),
        "closing": random.choice(CLOSINGS)
    }
    return clean_paragraph(pattern.format(**values))


def generate_base_paragraphs(count: int) -> list[str]:
    paragraphs = set()
    while len(paragraphs) < count:
        paragraphs.add(build_paragraph())
    return list(paragraphs)


def save_dataset(rows: list[dict], filename: str) -> None:
    fieldnames = [
        "original_paragraph",
        "transformed_paragraph",
        "transformation_type",
        "similarity_score",
        "same_meaning"
    ]
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a paragraph dataset for plagiarism analysis.")
    parser.add_argument("--rows", "-r", type=int, default=10000, help="Number of rows to generate.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV filename.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(42)

    target_rows = args.rows
    base_count = max(2000, (target_rows // len(TRANSFORMATION_TYPES)) + 5)
    base_paragraphs = generate_base_paragraphs(base_count)

    rows = []
    transform_functions = [
        (transform_sentence_reorder, "sentence_reorder"),
        (transform_paragraph_pronoun_change, "paragraph_pronoun_change"),
        (transform_phrase_level, "phrase_level_rewrite"),
        (transform_human_like, "human_like_rewrite"),
        (transform_formal_rewrite, "formal_rewrite")
    ]

    index = 0
    while len(rows) < target_rows:
        paragraph = base_paragraphs[index % len(base_paragraphs)]
        transform_func, label = transform_functions[index % len(transform_functions)]
        transformed = clean_paragraph(transform_func(paragraph))
        similarity = sentence_similarity(paragraph, transformed)
        rows.append({
            "original_paragraph": paragraph,
            "transformed_paragraph": transformed,
            "transformation_type": label,
            "similarity_score": similarity,
            "same_meaning": True
        })
        index += 1

    output_file = args.output or f"paragraph_dataset_{target_rows}.csv"
    save_dataset(rows, output_file)
    print(f"Generated paragraph dataset with {len(rows)} rows: {output_file}")


if __name__ == "__main__":
    main()
