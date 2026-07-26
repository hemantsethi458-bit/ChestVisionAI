"""Prediction history persistence."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from utils.io import read_json, write_json
from utils.paths import ensure_dir


class PredictionHistory:
    """JSON-backed store for inference history records."""

    def __init__(self, db_path: Path) -> None:
        """Initialize history store at the given path."""
        self.db_path = ensure_dir(db_path.parent) / db_path.name
        if not self.db_path.exists():
            write_json(self.db_path, {"records": []})

    def _load(self) -> Dict[str, List[dict]]:
        """Load all records from disk."""
        return read_json(self.db_path)

    def add_record(
        self,
        patient_id: str,
        image_name: str,
        predictions: Dict[str, object],
        model_version: str,
        report_path: Optional[str] = None,
        heatmap_path: Optional[str] = None,
    ) -> dict:
        """Append a new prediction record.

        Args:
            patient_id: Patient identifier supplied by user or metadata.
            image_name: Source image filename.
            predictions: Structured prediction output.
            model_version: Model version string.
            report_path: Optional generated PDF report path.
            heatmap_path: Optional saved heatmap image path.

        Returns:
            Newly created history record.
        """
        payload = self._load()
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patient_id": patient_id,
            "image_name": image_name,
            "predictions": predictions,
            "model_version": model_version,
            "report_path": report_path,
            "heatmap_path": heatmap_path,
        }
        payload["records"].insert(0, record)
        write_json(self.db_path, payload)
        return record

    def list_records(self, limit: Optional[int] = None) -> List[dict]:
        """Return prediction records, optionally truncated."""
        records = self._load().get("records", [])
        return records[:limit] if limit is not None else records

    def get_record(self, record_id: str) -> Optional[dict]:
        """Fetch a single record by UUID."""
        for record in self.list_records():
            if record["id"] == record_id:
                return record
        return None

    def clear(self) -> None:
        """Remove all stored prediction records."""
        write_json(self.db_path, {"records": []})
