"""Application configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
SPORTS_FACTS_PATH = DATA_DIR / "sports_facts.json"

COLLECTION_NAME = "sports_facts"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"

SUPPORTED_SPORTS = ["Cricket", "Football", "Tennis", "Basketball", "Badminton"]
SUPPORTED_DIFFICULTIES = ["Easy", "Medium", "Hard"]


def load_environment() -> None:
    """Load environment variables from the project .env file when present."""
    load_dotenv(BASE_DIR / ".env")


def get_gemini_api_key() -> str:
    """Return the configured Gemini API key or an empty string."""
    load_environment()
    return os.getenv("GEMINI_API_KEY", "").strip()


def get_gemini_model_name() -> str:
    """Return the Gemini model name configured for quiz generation."""
    load_environment()
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL

