"""Prediction page for Streamlit dashboard."""

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from configs.config import get_config

from streamlit_app.components.ui import checkpoint_exists, load_image_uploader, render_header, show_missing_model_warning


@st.cache_resource
def get_predictor():
    """Load and cache predictor instance for the Streamlit session."""

    from inference.predictor import ChestXrayPredictor

    config = get_config()
    return ChestXrayPredictor(config=config)
config = get_config()
return ChestXrayPredictor(config=config)

def render() -> None:
    """Render prediction workflow page."""
    render_header("Prediction", "Upload a chest X-ray to generate multi-label predictions.")
    if not checkpoint_exists():
        show_missing_model_warning()
        return

    patient_id = st.text_input("Patient ID", value="DEMO-001")
    uploaded = load_image_uploader()

    col1, col2 = st.columns([1, 1])
    with col1:
        generate_heatmap = st.checkbox("Generate Grad-CAM heatmap", value=True)
    with col2:
        generate_report = st.checkbox("Generate PDF report", value=True)

    if uploaded is None:
        st.info("Upload an X-ray image to begin analysis.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)

    if st.button("Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Running inference..."):
            with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                temp_path = tmp.name

            try:
                predictor = get_predictor()
                result = predictor.predict(
                    image_path=temp_path,
                    patient_id=patient_id,
                    generate_heatmap=generate_heatmap,
                    generate_report=generate_report,
                )
            finally:
                Path(temp_path).unlink(missing_ok=True)

        st.success("Analysis complete.")
        predictions = result.predictions

        st.subheader("Top Predictions")
        for item in predictions["top_predictions"]:
            st.progress(min(max(item["confidence"], 0.0), 1.0), text=f"{item['label']} — {item['confidence_pct']}%")

        positive = predictions.get("positive_labels", [])
        st.subheader("Detected Conditions")
        if positive:
            st.table([{"Disease": p["label"], "Confidence": f"{p['confidence_pct']}%"} for p in positive])
        else:
            st.write("No conditions exceeded the configured threshold.")

        if result.overlay_path and Path(result.overlay_path).exists():
            st.subheader("Grad-CAM Overlay")
            st.image(result.overlay_path, use_container_width=True)

        if result.report_path and Path(result.report_path).exists():
            with open(result.report_path, "rb") as report_file:
                st.download_button(
                    label="Download PDF Report",
                    data=report_file.read(),
                    file_name=Path(result.report_path).name,
                    mime="application/pdf",
                    use_container_width=True,
                )
