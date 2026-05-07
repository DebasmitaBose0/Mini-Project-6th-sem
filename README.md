# Plagiarism AI Detection & Rewriting System

```text
╔════════════════════╗
   Plagiarism AI
╚════════════════════╝
```

## 📌 Project Overview
The **Plagiarism AI Detection & Rewriting System** is an advanced application that detects text similarities and identifies potential plagiarism, while also offering AI-powered text rewriting capabilities. The system analyzes input text (via direct paste, URL, or file upload) and compares it with reference documents to calculate similarity scores using Natural Language Processing (NLP) techniques.

## 🚀 Features
* **Plagiarism Detection:** Paste text, upload documents, or provide URLs to cleanly scan for plagiarism.
* **AI Text Rewriting:** Automatically rewrite flagged content to improve originality.
* **Smart History:** Keep track of recent scans with a convenient history sidebar.
* **Modern UI/UX:** Responsive, accessible, and animated interface featuring dark mode support.
* **Highlighting & Copying:** Easily copy rewritten text and view detailed similarity metrics instantly.

## 🛠️ Technologies Used

**Frontend**
* React 18 & TypeScript
* Vite
* Tailwind CSS & shadcn/ui
* Framer Motion (Animations)
* React Router DOM (Routing)

**Backend**
* Python 3
* FastAPI (High-performance API framework)
* Uvicorn (ASGI web server)

## 📂 Project Structure

```text
Mini-Project-6th-sem/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   ├── models/           # Data schemas
│   │   ├── routes/           # API Endpoints (analyze, history, rewrite)
│   │   └── services/         # LLM & Similarity Services
│   └── requirements.txt      # Python dependencies
├── frontend-new/new-look/
│   ├── public/               # Static assets
│   ├── src/
│   │   ├── components/       # Reusable React components & UI elements
│   │   ├── pages/            # Application views (Dashboard, etc.)
│   │   └── App.tsx           # Main application routing
│   ├── package.json          # Node dependencies
│   └── tailwind.config.ts    # Tailwind styling configuration
└── README.md
```

## ⚙️ Installation & Setup

Follow these steps to run the project locally. You will need two terminal windows running simultaneously.

### 1. Backend Setup

Open your first terminal and navigate to the backend directory:
```bash
cd backend
```

Create and activate a virtual environment (Windows):
```bash
python -m venv .venv
../.venv/Scripts/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```
The backend will run on `http://localhost:8000`.

### 2. Frontend Setup

Open your second terminal and navigate to the frontend directory:
```bash
cd frontend-new/new-look
```

Install Node.js dependencies:
```bash
npm install
# or if using bun
bun install
```

Start the Vite development server:
```bash
npm run dev
# or if using bun
bun run dev
```

Access the application at `http://localhost:5173`.
