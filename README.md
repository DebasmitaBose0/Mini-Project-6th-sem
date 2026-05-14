# Plagiarism AI Detection & Rewriting System

```text
╔════════════════════╗
   Plagiarism AI
╚════════════════════╝
```

## 📌 Project Overview

The **Plagiarism AI Detection & Rewriting System** is an advanced open-source application that detects text similarities and identifies potential plagiarism, while offering AI-powered text rewriting capabilities to help users produce original content.

This platform was built as a third year mini-project by **Debasmita, Manisha and Joita**. It operates completely free of charge, leveraging a privacy-first local AI engine powered by **Ollama (Llama 3)** and FastAPI to ensure your data never leaves your computer.

## 🚀 Features

* **AI Plagiarism Scanner:** Paste your text directly into our clean, focused uploader to instantly calculate similarity scores and plagiarism removal metrics.
* **Context-Aware Rewriting:** Automatically rewrite flagged content using our local LLM integration to improve originality while preserving meaning.
* **Dedicated History Page:** Keep track of recent scans and review past rewrites on a centralized history dashboard.
* **Modern UI/UX:** Responsive, accessible, and animated interface featuring dark mode support and interactive tooltips.
* **100% Free & Open Source:** Built by students, for students. Enterprise-grade security and zero subscription fees.

## 🛠️ Technologies Used

### Frontend

* React 18 & TypeScript
* Vite
* Tailwind CSS & shadcn/ui
* Framer Motion (Animations)
* React Router DOM (Routing)

### Backend

* Python 3
* FastAPI (High-performance API framework)
* Uvicorn (ASGI web server)

## 📂 Project Structure

```text
Mini-Project-6th-sem/
├── README.md
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       ├── db/
│       │   └── memory_store.py
│       ├── models/
│       │   └── schema.py
│       ├── routes/
│       │   ├── analyze.py
│       │   ├── history.py
│       │   └── rewrite.py
│       └── services/
│           ├── llm_service.py
│           ├── similarity.py
│           └── text_utils.py
└── frontend-new/
    └── new-look/
        ├── bun.lockb
        ├── components.json
        ├── eslint.config.js
        ├── index.html
        ├── package.json
        ├── postcss.config.js
        ├── tailwind.config.ts
        ├── tsconfig.app.json
        ├── tsconfig.json
        ├── tsconfig.node.json
        ├── vite.config.ts
        ├── vitest.config.ts
        ├── public/
        └── src/
            ├── App.css
            ├── App.tsx
            ├── index.css
            ├── main.tsx
            ├── vite-env.d.ts
            ├── components/
            │   ├── NavLink.tsx
            │   ├── ScrollToTop.tsx
            │   └── plagiarism/
            │       ├── Footer.tsx
            │       ├── Hero.tsx
            │       ├── History.tsx
            │       ├── Navbar.tsx
            │       ├── Preloader.tsx
            │       └── Uploader.tsx
            │   └── ui/
            │       ├── accordion.tsx
            │       ├── alert-dialog.tsx
            │       ├── alert.tsx
            │       ├── aspect-ratio.tsx
            │       ├── avatar.tsx
            │       ├── badge.tsx
            │       ├── breadcrumb.tsx
            │       ├── button.tsx
            │       ├── calendar.tsx
            │       ├── carousel.tsx
            │       ├── chart.tsx
            │       ├── checkbox.tsx
            │       ├── collapsible.tsx
            │       ├── command.tsx
            │       ├── context-menu.tsx
            │       ├── dialog.tsx
            │       ├── drawer.tsx
            │       ├── dropdown-menu.tsx
            │       ├── form.tsx
            │       ├── hover-card.tsx
            │       ├── input-otp.tsx
            │       ├── input.tsx
            │       ├── label.tsx
            │       ├── menubar.tsx
            │       ├── navigation-menu.tsx
            │       ├── pagination.tsx
            │       ├── popover.tsx
            │       ├── progress.tsx
            │       ├── radio-group.tsx
            │       ├── resizable.tsx
            │       ├── scroll-area.tsx
            │       ├── select.tsx
            │       ├── separator.tsx
            │       ├── sheet.tsx
            │       ├── sidebar.tsx
            │       ├── skeleton.tsx
            │       ├── slider.tsx
            │       ├── sonner.tsx
            │       ├── switch.tsx
            │       ├── table.tsx
            │       ├── tabs.tsx
            │       ├── textarea.tsx
            │       ├── toaster.tsx
            │       ├── toggle-group.tsx
            │       ├── toggle.tsx
            │       ├── tooltip.tsx
            │       └── use-toast.ts
            ├── hooks/
            │   ├── use-mobile.tsx
            │   └── use-toast.ts
            ├── lib/
            │   └── utils.ts
            ├── pages/
            │   ├── ApiInfo.tsx
            │   ├── HistoryPage.tsx
            │   ├── Index.tsx
            │   ├── NotFound.tsx
            │   └── PlaceholderPage.tsx
            └── test/
                ├── example.test.ts
                └── setup.ts
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
