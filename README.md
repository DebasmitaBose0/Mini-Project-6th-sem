# PlagiarismAI — Premium AI Assistant

A polished, enterprise-grade plagiarism detection and rewrite assistant built for academic authenticity and advanced linguistic reconstruction. This project features a sophisticated Flask-driven backend paired with a premium, glassmorphism-inspired frontend.

**Authors:** Manisha, Joita, Debasmita, Suchitra

---

## 🌟 Premium Features

`PlagiarismAI` provides a state-of-the-art experience with:
- **Full Auto Workflow**: A 4-step automated pipeline that generates plagiarism, detects it, reconstructs clean text, and verifies meaning preservation.
- **Deep Neural Analysis**: Multi-step loading sequences showing real-time decomposition, semantic vector scanning, and cross-referencing.
- **Advanced Rewriting Engine**: Adjustable rewrite strength (Low, Medium, Aggressive) and specialized modes for **Removing Plagiarism** or **Humanizing AI Text**.
- **Human-Like Reconstruction**: Preserves semantic integrity with a target meaning match of 90%+.
- **Activity History**: Local storage integration to track your previous scans and rewrites.
- **Professional UI**: Glassmorphism design, pulsing preloader, and high-performance top-bar progress indicators.

---

## ⚙️ Prerequisites

Recommended Python version: `3.10+`

Install the core dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install flask pandas numpy scipy scikit-learn joblib
```

For advanced rewriting (OpenAI/Gemini support), create a `.env` file in the `plagiarism_app/` directory:

```text
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

---

## 🚀 Run the Application

Start the Flask app from the `plagiarism_app` folder:

```bash
cd plagiarism_app
python app.py
```

Access the dashboard at: `http://localhost:5000`

> **Note**: The app initializes with a professional neural engine preloader. During analysis, you can follow the progress through the dynamic multi-step status track.

---

## 🧠 Model & Training

The system uses a TF-IDF + Logistic Regression pipeline trained on **1,000,000+** samples for high-accuracy detection.

### Training Artifacts (`plagiarism_app/models/`)
- `plagiarism_model.pkl` — Trained classifier
- `tfidf_vectorizer.pkl` — Feature extractor
- `model_metadata.pkl` — Accuracy and F1 metrics displayed on the dashboard

### Retraining
```bash
python train_model.py --sample 500000
```

---

## 📂 Dataset and Utility Scripts

This repository includes tools for generating training data and testing rewrite behavior.

- `generate_million.py` — Script to build the 1M+ sample dataset.
- `generate_paragraph_dataset.py` — Creates paragraph-level datasets for broader similarity testing.
- `rewrite_para.py` — Runs a local rewrite test outside of the Flask UI.

---

## 🧩 Repository Structure

- `plagiarism_app/` — Core application (Flask, Templates, Static)
- `plagiarism_app/paraphrase_engine.py` — The "brain" behind linguistic reconstruction
- `plagiarism_app/models/` — Serialized AI weights and metadata
- `dataset.csv` — Generated training dataset
- `.gitignore` — Configured to protect `.env` and local caches

---

## 📄 License
This project is licensed under the MIT License. See `LICENSE` for details.
