"""Streamlit dashboard for the AI-Powered Sports Quiz Generation Agent."""

from __future__ import annotations

import streamlit as st  # type: ignore[reportMissingImports]

from src.config import SUPPORTED_DIFFICULTIES, SUPPORTED_SPORTS, load_environment
from src.database import query_database, setup_database
from src.generator import generate_quiz
from src.search import get_recent_context


st.set_page_config(
    page_title="AI Sports Quiz Agent",
    page_icon="🏆",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def initialize_database() -> bool:
    """Initialize ChromaDB once per Streamlit session."""
    setup_database()
    return True


def initialize_session_state() -> None:
    """Create the Streamlit session keys used by the dashboard."""
    defaults = {
        "quiz": "",
        "historical_context": "",
        "recent_context": "",
        "selected_sport": SUPPORTED_SPORTS[0],
        "selected_difficulty": SUPPORTED_DIFFICULTIES[1],
        "generation_count": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def create_quiz(sport: str, difficulty: str) -> None:
    """Retrieve RAG context and generate a quiz for the selected inputs."""
    with st.spinner("Retrieving sports context and generating your quiz..."):
        historical_context = query_database(sport=sport)
        recent_context = get_recent_context(sport=sport)
        quiz = generate_quiz(
            sport=sport,
            difficulty=difficulty,
            historical_context=historical_context,
            recent_context=recent_context,
        )

    st.session_state.quiz = quiz
    st.session_state.historical_context = historical_context
    st.session_state.recent_context = recent_context
    st.session_state.selected_sport = sport
    st.session_state.selected_difficulty = difficulty
    st.session_state.generation_count += 1


def render_sidebar_controls() -> tuple[str, str, bool, bool]:
    """Render sport and difficulty selectors plus generation controls."""
    with st.sidebar:
        st.header("⚙️ Quiz Settings")
        sport = st.selectbox(
            "Sport",
            SUPPORTED_SPORTS,
            index=SUPPORTED_SPORTS.index(st.session_state.selected_sport),
        )
        difficulty = st.selectbox(
            "Difficulty",
            SUPPORTED_DIFFICULTIES,
            index=SUPPORTED_DIFFICULTIES.index(st.session_state.selected_difficulty),
        )

        st.divider()
        generate_clicked = st.button("✨ Generate Quiz", use_container_width=True)
        regenerate_clicked = st.button(
            "🔄 Regenerate Quiz",
            use_container_width=True,
            disabled=not bool(st.session_state.quiz),
        )

    return sport, difficulty, generate_clicked, regenerate_clicked


def render_quiz() -> None:
    """Display the generated quiz and retrieved RAG context."""
    if not st.session_state.quiz:
        with st.container(border=True):
            st.subheader("Ready when you are")
            st.write(
                "Choose a sport and difficulty, then generate a grounded five-question quiz."
            )
        return

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Sport", st.session_state.selected_sport)
    metric_col_2.metric("Difficulty", st.session_state.selected_difficulty)
    metric_col_3.metric("Questions", "5")

    st.container(border=True).markdown(st.session_state.quiz)

    with st.expander("📚 Retrieved RAG Context", expanded=False):
        historical_col, recent_col = st.columns(2)
        with historical_col:
            st.subheader("Historical Context")
            st.markdown(st.session_state.historical_context)
        with recent_col:
            st.subheader("Recent News Context")
            st.markdown(st.session_state.recent_context)


def main() -> None:
    """Run the Streamlit dashboard."""
    load_environment()
    initialize_session_state()

    st.title("🏆 AI-Powered Sports Quiz Generation Agent")
    st.write(
        "Generate fact-aware sports quizzes with historical facts from ChromaDB, "
        "recent snippets from DuckDuckGo Search, and grounded quiz writing from Gemini."
    )

    try:
        initialize_database()
    except Exception as exc:
        st.error(f"Database setup failed: {exc}")
        st.stop()

    sport, difficulty, generate_clicked, regenerate_clicked = render_sidebar_controls()

    if generate_clicked or regenerate_clicked:
        try:
            create_quiz(sport=sport, difficulty=difficulty)
            st.success("Quiz generated successfully.")
        except Exception as exc:
            st.error(str(exc))

    render_quiz()


if __name__ == "__main__":
    main()

