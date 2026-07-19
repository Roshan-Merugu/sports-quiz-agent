# AI-Powered Sports Quiz Generation Agent

An AI-powered Streamlit application that generates sports multiple-choice quizzes with Retrieval-Augmented Generation (RAG). It combines historical sports facts from a persistent ChromaDB vector database, recent sports snippets from DuckDuckGo Search, and Google Gemini quiz generation.

## Project Overview

The app lets a user select a sport and difficulty level, then generates exactly five grounded multiple-choice questions. Each quiz includes options A through D, the correct answer, an explanation, and an expandable RAG context view.

Supported sports:

- Cricket
- Football
- Tennis
- Basketball
- Badminton

Supported difficulty levels:

- Easy
- Medium
- Hard

## Architecture

The project is split into small modules:

- `app.py`: Streamlit dashboard and session state workflow
- `src/config.py`: Paths, supported options, and environment loading
- `src/database.py`: ChromaDB setup, embedding, population, and retrieval
- `src/search.py`: DuckDuckGo recent-context retrieval
- `src/generator.py`: Gemini prompt construction and quiz generation
- `data/sports_facts.json`: Historical sports facts used for vector search

## RAG Workflow

1. `setup_database()` creates a persistent ChromaDB collection if it does not already exist.
2. Historical facts are loaded from `data/sports_facts.json`.
3. Facts are embedded with `sentence-transformers/all-MiniLM-L6-v2`.
4. `query_database()` retrieves relevant facts filtered by sport metadata.
5. `get_recent_context()` searches DuckDuckGo for the top three recent snippets.
6. `generate_quiz()` combines historical and recent context in a strict prompt.
7. Gemini returns a structured Markdown quiz with exactly five questions.

## Installation

Use Python 3.11.

```bash
python -m venv venv
```

Activate the virtual environment.

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup `.env`

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

The app never hardcodes API keys.

## Run

```bash
streamlit run app.py
```

The first run downloads the sentence-transformers embedding model and creates the persistent `chroma_db/` directory.

## Folder Structure

```text
sports-quiz-agent/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── data/
│   └── sports_facts.json
├── chroma_db/
└── src/
    ├── __init__.py
    ├── config.py
    ├── database.py
    ├── search.py
    └── generator.py
```

## Screenshots

Add screenshots here after running the Streamlit app:

- Dashboard with sport and difficulty selectors
- Generated quiz result
- Expanded RAG context section

## Future Improvements

- Add quiz export to PDF or Markdown.
- Add answer hiding and reveal mode for classroom use.
- Cache DuckDuckGo results for repeat queries.
- Store generated quiz history.
- Add automated tests for prompt validation and retrieval behavior.
- Support more sports and user-uploaded facts.

