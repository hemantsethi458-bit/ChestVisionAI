"""Reusable Streamlit UI components."""

from pathlib import Path

import streamlit as st

from configs.config import get_config


def render_sidebar() -> str:
    """Render sidebar navigation and return selected page key."""
    config = get_config()
    st.sidebar.title("ChestVision AI")
    st.sidebar.caption("Multi-label Chest X-ray Analysis")
    st.sidebar.markdown("---")

    pages = {
        "predict": "Prediction",
        "history": "History",
        "model_info": "Model Info",
        "performance": "Performance",
        "settings": "Settings",
    }
    selection = st.sidebar.radio("Navigation", options=list(pages.keys()), format_func=lambda key: pages[key])
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Project Paths**")
    st.sidebar.text(f"Data: {config.data.data_root}")
    st.sidebar.text(f"Weights: {config.paths.weights_dir.name}/")
    return selection


def render_header(title: str, subtitle: str = "") -> None:
    """Render a consistent page header."""
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def load_image_uploader(label: str = "Upload Chest X-ray"):
    """Render file uploader and return uploaded file object."""
    return st.file_uploader(
        label,
        type=["png", "jpg", "jpeg"],
        help="Upload a frontal chest X-ray image in PNG or JPEG format.",
    )


def checkpoint_exists() -> bool:
    """Return True if a trained checkpoint is available."""
    config = get_config()
    return config.paths.best_model_path.exists()


def show_missing_model_warning() -> None:
    """Display guidance when model weights are unavailable."""
    st.warning(
        "No trained model checkpoint found. Train the model first:\n\n"
        "`python -m training.train`"
    )
