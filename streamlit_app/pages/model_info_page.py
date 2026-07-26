"""Model information page."""

import streamlit as st

from configs.config import get_config
from configs.constants import DISEASE_LABELS, MODEL_VERSION
from streamlit_app.components.ui import checkpoint_exists, render_header


def render() -> None:
    """Display model architecture and dataset details."""
    config = get_config()
    render_header("Model Information", "Architecture, dataset, and checkpoint details.")

    st.markdown("### Architecture")
    st.markdown(
        """
        - **Backbone:** DenseNet121 (ImageNet pretrained)
        - **Task:** Multi-label classification (14 thoracic diseases)
        - **Loss:** BCEWithLogitsLoss
        - **Output:** 14 independent sigmoid probabilities
        - **Explainability:** Grad-CAM on final dense block
        """
    )

    st.markdown("### Dataset")
    st.markdown(
        """
        - **Source:** NIH ChestX-ray14
        - **Split strategy:** Patient-level stratified split (70/15/15)
        - **Input size:** 224 × 224 RGB
        - **Normalization:** ImageNet mean/std
        """
    )

    st.markdown("### Disease Labels")
    st.write(", ".join(DISEASE_LABELS))

    st.markdown("### Checkpoint Status")
    if checkpoint_exists():
        st.success(f"Checkpoint available: `{config.paths.best_model_path}`")
    else:
        st.error("No checkpoint found. Train the model to enable inference.")

    st.markdown("### Version")
    st.code(MODEL_VERSION)
