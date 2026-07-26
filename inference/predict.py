"""CLI entry point for single-image inference."""

import argparse
import json

from configs.config import get_config
from inference.predictor import ChestXrayPredictor
from utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(description="Run ChestVision AI inference")
    parser.add_argument("--image", type=str, required=True, help="Path to chest X-ray image")
    parser.add_argument("--patient-id", type=str, default="UNKNOWN", help="Patient identifier")
    parser.add_argument("--no-heatmap", action="store_true", help="Disable Grad-CAM generation")
    parser.add_argument("--no-report", action="store_true", help="Disable PDF report generation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path")
    return parser.parse_args()


def main() -> None:
    """Run inference from the command line."""
    args = parse_args()
    config = get_config()
    checkpoint = config.paths.project_root / args.checkpoint if args.checkpoint else None
    predictor = ChestXrayPredictor(config=config, checkpoint_path=checkpoint)
    result = predictor.predict(
        image_path=args.image,
        patient_id=args.patient_id,
        generate_heatmap=not args.no_heatmap,
        generate_report=not args.no_report,
    )
    print(json.dumps(result.predictions, indent=2))
    logger.info("Report saved to: %s", result.report_path)


if __name__ == "__main__":
    main()
