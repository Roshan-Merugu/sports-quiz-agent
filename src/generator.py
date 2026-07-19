"""Gemini-powered quiz generation using retrieved RAG context."""

from __future__ import annotations

import google.generativeai as genai

from src.config import get_gemini_api_key, get_gemini_model_name


def build_quiz_prompt(
    sport: str,
    difficulty: str,
    historical_context: str,
    recent_context: str,
) -> str:
    """Build a grounded prompt for Gemini quiz generation."""
    return f"""
You are an expert sports quiz writer.

Generate exactly FIVE multiple-choice quiz questions.

Strict grounding rules:
- Use only the retrieved information in Historical Context and Recent News Context.
- Never hallucinate, guess, invent facts, or add unsupported statistics.
- If a fact is not present in the contexts, do not use it.
- Generate unique questions with no repeated answer patterns.
- Match the requested difficulty level.
- Return structured Markdown only.

Sport: {sport}
Difficulty: {difficulty}

Historical Context:
{historical_context}

Recent News Context:
{recent_context}

Markdown output format:

## Sport
{sport}

## Difficulty
{difficulty}

## Questions

### Question 1
Question: ...

A. ...
B. ...
C. ...
D. ...

Correct Answer: ...

Explanation: ...

Repeat this structure through Question 5.
""".strip()


def generate_quiz(
    sport: str,
    difficulty: str,
    historical_context: str,
    recent_context: str,
) -> str:
    """Generate a five-question quiz with Google Gemini."""
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to a .env file before generating a quiz."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(get_gemini_model_name())
    prompt = build_quiz_prompt(
        sport=sport,
        difficulty=difficulty,
        historical_context=historical_context,
        recent_context=recent_context,
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.35,
                "top_p": 0.9,
                "max_output_tokens": 4096,
            },
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini quiz generation failed: {exc}") from exc

    quiz_text = getattr(response, "text", "").strip()
    if not quiz_text:
        raise RuntimeError("Gemini returned an empty response.")

    return quiz_text

