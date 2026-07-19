"""DuckDuckGo search integration for recent sports context."""

from __future__ import annotations

from duckduckgo_search import DDGS


def get_recent_context(sport: str) -> str:
    """Return top DuckDuckGo snippets for recent sport news and records."""
    query = f"{sport} latest tournament winners news records"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as exc:
        return f"Recent web context unavailable for {sport}: {exc}"

    snippets: list[str] = []
    for result in results[:3]:
        title = result.get("title", "").strip()
        body = (result.get("body") or result.get("snippet") or "").strip()
        href = result.get("href", "").strip()

        if body:
            if title and href:
                snippets.append(f"{title}: {body} Source: {href}")
            elif title:
                snippets.append(f"{title}: {body}")
            else:
                snippets.append(body)

    if not snippets:
        return f"No recent DuckDuckGo snippets were found for {sport}."

    return "\n".join(f"- {snippet}" for snippet in snippets)

