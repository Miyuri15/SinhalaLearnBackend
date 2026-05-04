"""Command-line sanity check for the printed vs handwritten classifier."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from app.components.document_processing.services.classify_text_type_ml import (
    MobileNetV2TextTypeClassifier,
)
from app.components.document_processing.services.text_extraction import classify_text_type


DEFAULT_MODEL_PATH = Path("utils") / "printed_vs_handwritten.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test utils/printed_vs_handwritten.pth against one image file."
    )
    parser.add_argument(
        "image_path",
        help="Path to the image to classify as printed or handwritten.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to the .pth classifier model. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use, for example cpu or cuda. Default: auto-detect.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum ML confidence for classify_text_type to trust the model. Default: 0.7",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image_path)

    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}", file=sys.stderr)
        return 1

    if not image_path.exists():
        print(f"ERROR: Image file not found: {image_path}", file=sys.stderr)
        return 1

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not read image file: {image_path}", file=sys.stderr)
        return 1

    try:
        load_started = time.perf_counter()
        classifier = MobileNetV2TextTypeClassifier(
            model_path=str(model_path),
            device=args.device,
        )
        load_seconds = time.perf_counter() - load_started

        ml_started = time.perf_counter()
        ml_label, ml_confidence = classifier.predict(image)
        ml_seconds = time.perf_counter() - ml_started

        app_started = time.perf_counter()
        app_label, app_confidence = classify_text_type(
            image,
            ml_model_predict=classifier.predict,
            ml_conf_threshold=args.threshold,
            return_confidence=True,
        )
        app_seconds = time.perf_counter() - app_started
    except Exception as exc:
        print(f"ERROR: Classification failed: {exc}", file=sys.stderr)
        return 1

    print("Printed vs handwritten classifier test")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Device: {classifier.device}")
    print(f"ML prediction: {ml_label}")
    print(f"ML confidence: {ml_confidence:.4f}")
    print(f"ML prediction time: {ml_seconds:.4f}s")
    print(f"classify_text_type prediction: {app_label}")
    print(f"classify_text_type confidence: {app_confidence:.4f}")
    print(f"classify_text_type time: {app_seconds:.4f}s")
    print(f"classify_text_type ML threshold: {args.threshold:.4f}")
    print(f"Model load time: {load_seconds:.4f}s")
    print(f"Total time: {(load_seconds + ml_seconds + app_seconds):.4f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
