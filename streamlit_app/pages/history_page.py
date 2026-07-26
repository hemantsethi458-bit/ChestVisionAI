"""Prediction history page."""

from pathlib import Path

import pandas as pd
import streamlit as st

from configs.config import get_config
from inference.history import PredictionHistory
from streamlit_app.components.ui import render_header


def render() -> None:
    """Render prediction history table and record details."""
    render_header("Prediction History", "Review previously generated inference results.")
    config = get_config()
    history = PredictionHistory(config.inference.history_db_path)
    records = history.list_records()

    if not records:
        st.info("No prediction history yet. Run an analysis from the Prediction page.")
        return

    dataframe = pd.DataFrame(
        [
            {
                "Timestamp": record["timestamp"],
                "Patient ID": record["patient_id"],
                "Image": record["image_name"],
                "Top Prediction": record["predictions"]["top_predictions"][0]["label"],
                "Confidence": f"{record['predictions']['top_predictions'][0]['confidence_pct']}%",
                "Model": record["model_version"],
            }
            for record in records
        ]
    )
    st.dataframe(dataframe, use_container_width=True, hide_index=True)

    st.subheader("Record Details")
    record_ids = [record["id"] for record in records]
    selected_id = st.selectbox("Select record", options=record_ids)
    selected = history.get_record(selected_id)
    if selected:
        st.json(selected["predictions"])
        if selected.get("report_path") and Path(selected["report_path"]).exists():
            with open(selected["report_path"], "rb") as report_file:
                st.download_button(
                    "Download Report",
                    data=report_file.read(),
                    file_name=Path(selected["report_path"]).name,
                    mime="application/pdf",
                )

    if st.button("Clear History", type="secondary"):
        history.clear()
        st.warning("History cleared.")
        st.rerun()
