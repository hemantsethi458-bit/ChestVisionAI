"""Settings page for runtime configuration."""

import os

import streamlit as st

from configs.config import get_config
from streamlit_app.components.ui import render_header


def render() -> None:
    """Render editable runtime settings stored in session state."""
    config = get_config()
    render_header("Settings", "Adjust inference thresholds and environment overrides.")

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(config.inference.threshold),
        step=0.05,
        help="Probability threshold used to mark a disease as positive.",
    )
    top_k = st.number_input("Top-K Predictions", min_value=1, max_value=14, value=config.inference.top_k)
    batch_size = st.number_input("Training Batch Size", min_value=4, max_value=128, value=config.training.batch_size)
    learning_rate = st.number_input(
        "Learning Rate",
        min_value=1e-6,
        max_value=1e-2,
        value=float(config.training.learning_rate),
        format="%.6f",
    )

    st.markdown("### Environment Variables")
    st.code(
        "\n".join(
            [
                f"CHESTVISION_DATA_ROOT={config.data.data_root}",
                f"CHESTVISION_BATCH_SIZE={batch_size}",
                f"CHESTVISION_LR={learning_rate}",
                f"CHESTVISION_THRESHOLD={threshold}",
            ]
        )
    )

    if st.button("Apply Session Settings", type="primary"):
        config.inference.threshold = threshold
        config.inference.top_k = int(top_k)
        config.training.batch_size = int(batch_size)
        config.training.learning_rate = float(learning_rate)
        os.environ["CHESTVISION_THRESHOLD"] = str(threshold)
        os.environ["CHESTVISION_BATCH_SIZE"] = str(batch_size)
        os.environ["CHESTVISION_LR"] = str(learning_rate)
        st.success("Settings applied for the current Streamlit session.")

    st.markdown("### Retraining")
    st.markdown(
        "To retrain the model with updated settings:\n\n"
        "```bash\n"
        "python -m training.train --epochs 50 --batch-size 32 --lr 1e-4\n"
        "```"
    )
