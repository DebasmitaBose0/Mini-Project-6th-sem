# Sentence Paraphrase & Similarity Dataset

This project creates a synthetic dataset of sentence pairs that preserve meaning while changing word order or voice. It is designed for a plagiarism/rewriting task where you want to avoid copying text exactly but keep the same meaning.

## What this dataset contains

- `original_sentence`: the source sentence
- `transformed_sentence`: a rewritten version with the same meaning
- `transformation_type`: the method used, such as `reorder`, `active_to_passive`, `passive_to_active`, or `pronoun_change`
- `similarity_score`: a numeric estimate of how similar the rewritten sentence is to the original, based on sequence similarity

## How to use

1. Run the generator:

```bash
python generate_dataset.py
```

2. The script will create `dataset.csv` in the same folder.

3. You can open `dataset.csv` in Excel, a text editor, or load it into Python for analysis.

## Why this is your own dataset

- The dataset is generated from your own sentence templates and word lists.
- No online scraping is used.
- You control the examples, transformations, and similarity labels.

## Extending the dataset

To add new examples, edit the `base_sentences` list or add new transformation rules in `generate_dataset.py`.

## Paragraph rewriter tool

Use `rewrite_para.py` to enter a paragraph or sentence and see rewritten alternatives:

```bash
python rewrite_para.py
```

The script shows:
- a reordered version
- a pronoun-change version
- an active-to-passive conversion
- a passive-to-active conversion
- a best rewrite and similarity score

This helps you practice rewriting text so the meaning stays the same while the wording changes.
