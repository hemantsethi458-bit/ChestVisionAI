"""Training performance visualization page."""

from pathlib import Path

import streamlit as st
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from configs.config import get_config
from streamlit_app.components.ui import render_header


def _load_scalar(tag: str, log_dir: Path):
    """Load scalar time series from TensorBoard event files."""
    if not log_dir.exists():
        return None
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return None
    events = accumulator.Scalars(tag)
    steps = [event.step for event in events]
    values = [event.value for event in events]
    return {"steps": steps, "values": values}


def render() -> None:
    """Render TensorBoard-derived training metrics."""
    config = get_config()
    render_header("Performance", "Training and validation metrics from TensorBoard logs.")
    log_dir = config.paths.tensorboard_dir

    if not log_dir.exists() or not any(log_dir.iterdir()):
        st.info(
            "No TensorBoard logs found. Start training with:\n\n"
            "`python -m training.train`"
        )
        return

    metrics = {
        "Loss/train": _load_scalar("Loss/train", log_dir),
        "Loss/val": _load_scalar("Loss/val", log_dir),
        "F1/val": _load_scalar("F1/val", log_dir),
        "Accuracy/val": _load_scalar("Accuracy/val", log_dir),
    }

    available = {name: data for name, data in metrics.items() if data is not None}
    if not available:
        st.warning("TensorBoard directory exists but no scalar metrics were found yet.")
        return

    for name, data in available.items():
        st.markdown(f"**{name}**")
        st.line_chart({"value": data["values"]}, x=data["steps"])

    eval_dir = config.paths.logs_dir / "evaluation"
    if eval_dir.exists():
        st.markdown("### Saved Evaluation Artifacts")
        for image_path in eval_dir.rglob("*.png"):
            st.image(str(image_path), caption=image_path.name, use_container_width=True)

        metrics_files = list(eval_dir.rglob("metrics.json"))
        if metrics_files:
            st.markdown("### Latest Evaluation Metrics")
            import json

            with metrics_files[0].open("r", encoding="utf-8") as handle:
                st.json(json.load(handle))
