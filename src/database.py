"""Persistent ChromaDB setup and retrieval for historical sports facts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME, SPORTS_FACTS_PATH


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence-transformers embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Create vector embeddings for a list of text documents."""
    if not texts:
        return []
    embeddings = get_embedding_model().encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def _load_sports_facts(data_path: Path = SPORTS_FACTS_PATH) -> list[dict[str, str]]:
    """Load sports facts from JSON and validate the required fields."""
    if not data_path.exists():
        raise FileNotFoundError(f"Sports facts file not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as file:
        facts = json.load(file)

    if not isinstance(facts, list):
        raise ValueError("sports_facts.json must contain a list of fact objects.")

    required_fields = {"id", "sport", "fact"}
    validated_facts: list[dict[str, str]] = []
    for item in facts:
        if not isinstance(item, dict) or not required_fields.issubset(item):
            raise ValueError("Each sports fact must include id, sport, and fact fields.")
        validated_facts.append(
            {
                "id": str(item["id"]),
                "sport": str(item["sport"]),
                "fact": str(item["fact"]),
            }
        )
    return validated_facts


def get_chroma_collection() -> Any:
    """Return the persistent ChromaDB collection, creating it when needed."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Historical sports facts for RAG quiz generation"},
    )


def setup_database() -> Any:
    """Create and populate the persistent ChromaDB collection from sports_facts.json."""
    collection = get_chroma_collection()
    facts = _load_sports_facts()

    ids = [fact["id"] for fact in facts]
    documents = [fact["fact"] for fact in facts]
    metadatas = [{"sport": fact["sport"], "category": "history"} for fact in facts]
    embeddings = _embed_texts(documents)

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return collection


def query_database(sport: str, query_text: str | None = None, n_results: int = 8) -> str:
    """Retrieve historical context from ChromaDB, filtered by sport metadata."""
    collection = setup_database()
    query = query_text or f"{sport} historical winners records milestones facts"
    query_embeddings = _embed_texts([query])

    results: dict[str, Any] = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        where={"sport": sport},
        include=["documents", "metadatas"],
    )

    documents = results.get("documents", [[]])
    if not documents or not documents[0]:
        return f"No historical facts were found for {sport}."

    return "\n".join(f"- {document}" for document in documents[0])
