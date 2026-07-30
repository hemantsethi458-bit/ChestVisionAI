"""ChestVision AI Streamlit dashboard entry point."""

import streamlit as st

from components.ui import render_sidebar
from pages import history_page, model_info_page, performance_page, predict_page, settings_page

st.set_page_config(
    page_title="ChestVision AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "predict": predict_page.render,
    "history": history_page.render,
    "model_info": model_info_page.render,
    "performance": performance_page.render,
    "settings": settings_page.render,
}


def main() -> None:
    """Launch selected dashboard page."""
    selected = render_sidebar()
    PAGES[selected]()


if __name__ == "__main__":
    main()
