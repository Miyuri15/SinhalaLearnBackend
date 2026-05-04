"""Evaluate printed vs handwritten classification against a metadata.csv folder."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2

from app.components.document_processing.services.classify_text_type_ml import (
    MobileNetV2TextTypeClassifier,
)
from app.components.document_processing.services.text_extraction import classify_text_type


DEFAULT_MODEL_PATH = Path("utils") / "printed_vs_handwritten.pth"
REQUIRED_COLUMNS = {"file_name", "text", "is_handwritten"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate utils/printed_vs_handwritten.pth and classify_text_type() "
            "against a dataset folder containing metadata.csv."
        )
    )
    parser.add_argument(
        "dataset_folder",
        help="Folder containing metadata.csv and image folders such as train_images.",
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of metadata rows to evaluate. Default: all rows.",
    )
    parser.add_argument(
        "--errors-output",
        default=None,
        help="Optional CSV path for incorrectly classified rows.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many evaluated rows. Default: 100.",
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "handwritten"}:
        return True
    if normalized in {"false", "0", "no", "n", "printed"}:
        return False
    raise ValueError(f"Unsupported boolean value for is_handwritten: {value!r}")


def expected_label(is_handwritten: bool) -> str:
    return "handwritten" if is_handwritten else "printed"


def iter_metadata_rows(metadata_path: Path, limit: int | None) -> Iterable[dict[str, str]]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"metadata.csv missing required columns: {missing_cols}")

        for index, row in enumerate(reader, start=1):
            if limit is not None and index > limit:
                break
            yield row


def count_metadata_rows(metadata_path: Path, limit: int | None) -> int:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        total = max(sum(1 for _ in file) - 1, 0)

    if limit is not None:
        return min(total, limit)
    return total


def resolve_image_path(dataset_folder: Path, file_name: str) -> Path | None:
    candidates = [
        dataset_folder / file_name,
        dataset_folder / "train_images" / file_name,
        dataset_folder / "ground_truth" / file_name,
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = list(dataset_folder.rglob(file_name))
    if matches:
        return matches[0]

    return None


def empty_counts() -> dict[str, int]:
    return {
        "total": 0,
        "correct": 0,
        "printed_as_printed": 0,
        "printed_as_handwritten": 0,
        "printed_as_unknown": 0,
        "handwritten_as_printed": 0,
        "handwritten_as_handwritten": 0,
        "handwritten_as_unknown": 0,
    }


def update_counts(counts: dict[str, int], expected: str, predicted: str) -> None:
    counts["total"] += 1
    if expected == predicted:
        counts["correct"] += 1

    key = f"{expected}_as_{predicted}"
    if key in counts:
        counts[key] += 1


def print_counts(title: str, counts: dict[str, int]) -> None:
    total = counts["total"]
    accuracy = (counts["correct"] / total * 100.0) if total else 0.0

    print()
    print(title)
    print(f"Total: {total}")
    print(f"Correct: {counts['correct']}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Printed -> printed: {counts['printed_as_printed']}")
    print(f"Printed -> handwritten: {counts['printed_as_handwritten']}")
    print(f"Printed -> unknown: {counts['printed_as_unknown']}")
    print(f"Handwritten -> handwritten: {counts['handwritten_as_handwritten']}")
    print(f"Handwritten -> printed: {counts['handwritten_as_printed']}")
    print(f"Handwritten -> unknown: {counts['handwritten_as_unknown']}")


def print_progress(
    processed_rows: int,
    total_rows: int,
    evaluated_rows: int,
    start_time: float,
) -> None:
    elapsed = max(time.time() - start_time, 0.001)
    rows_per_second = processed_rows / elapsed
    percent = (processed_rows / total_rows * 100.0) if total_rows else 100.0

    if rows_per_second > 0 and total_rows:
        remaining_seconds = max((total_rows - processed_rows) / rows_per_second, 0.0)
        eta_text = f"{remaining_seconds:.0f}s"
    else:
        eta_text = "unknown"

    print(
        f"Progress: {processed_rows}/{total_rows} rows "
        f"({percent:.1f}%) | evaluated: {evaluated_rows} | "
        f"speed: {rows_per_second:.2f} rows/s | ETA: {eta_text}",
        flush=True,
    )


def main() -> int:
    args = parse_args()

    dataset_folder = Path(args.dataset_folder)
    metadata_path = dataset_folder / "metadata.csv"
    model_path = Path(args.model)

    if not dataset_folder.exists():
        print(f"ERROR: Dataset folder not found: {dataset_folder}", file=sys.stderr)
        return 1

    if not metadata_path.exists():
        print(f"ERROR: metadata.csv not found: {metadata_path}", file=sys.stderr)
        return 1

    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}", file=sys.stderr)
        return 1

    try:
        classifier = MobileNetV2TextTypeClassifier(
            model_path=str(model_path),
            device=args.device,
        )
    except Exception as exc:
        print(f"ERROR: Failed to load classifier: {exc}", file=sys.stderr)
        return 1

    ml_counts = empty_counts()
    app_counts = empty_counts()
    missing_images = 0
    unreadable_images = 0
    invalid_rows = 0
    errors: list[dict[str, object]] = []
    processed_rows = 0
    total_rows = count_metadata_rows(metadata_path, args.limit)
    start_time = time.time()

    print("Starting printed vs handwritten folder evaluation")
    print(f"Dataset: {dataset_folder}")
    print(f"Rows to process: {total_rows}")
    print(f"Progress interval: every {args.progress_every} rows")
    print("", flush=True)

    try:
        rows = iter_metadata_rows(metadata_path, args.limit)
        for row_number, row in enumerate(rows, start=1):
            processed_rows += 1
            file_name = str(row.get("file_name", "")).strip()

            try:
                expected = expected_label(parse_bool(row.get("is_handwritten", "")))
            except ValueError:
                invalid_rows += 1
                continue

            image_path = resolve_image_path(dataset_folder, file_name)
            if image_path is None:
                missing_images += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                unreadable_images += 1
                continue

            try:
                ml_label, ml_confidence = classifier.predict(image)
                app_label, app_confidence = classify_text_type(
                    image,
                    ml_model_predict=classifier.predict,
                    ml_conf_threshold=args.threshold,
                    return_confidence=True,
                )
            except Exception as exc:
                print(
                    f"ERROR: Classification failed for {image_path}: {exc}",
                    file=sys.stderr,
                )
                return 1

            update_counts(ml_counts, expected, ml_label)
            update_counts(app_counts, expected, app_label)

            if ml_label != expected or app_label != expected:
                errors.append(
                    {
                        "row_number": row_number,
                        "file_name": file_name,
                        "expected": expected,
                        "ml_prediction": ml_label,
                        "ml_confidence": f"{ml_confidence:.4f}",
                        "classify_text_type_prediction": app_label,
                        "classify_text_type_confidence": f"{app_confidence:.4f}",
                        "text": row.get("text", ""),
                    }
                )

            if (
                args.progress_every > 0
                and processed_rows % args.progress_every == 0
            ):
                print_progress(
                    processed_rows,
                    total_rows,
                    app_counts["total"],
                    start_time,
                )
    except Exception as exc:
        print(f"ERROR: Failed to evaluate dataset: {exc}", file=sys.stderr)
        return 1

    if processed_rows == 0 or processed_rows % args.progress_every != 0:
        print_progress(
            processed_rows,
            total_rows,
            app_counts["total"],
            start_time,
        )

    print("Printed vs handwritten folder evaluation")
    print(f"Dataset: {dataset_folder}")
    print(f"Metadata: {metadata_path}")
    print(f"Model: {model_path}")
    print(f"Device: {classifier.device}")
    print(f"classify_text_type ML threshold: {args.threshold:.4f}")

    print_counts("Raw ML model result", ml_counts)
    print_counts("classify_text_type result", app_counts)

    print()
    print("Skipped rows")
    print(f"Missing images: {missing_images}")
    print(f"Unreadable images: {unreadable_images}")
    print(f"Invalid metadata rows: {invalid_rows}")

    if args.errors_output:
        errors_output = Path(args.errors_output)
        with errors_output.open("w", encoding="utf-8", newline="") as file:
            fieldnames = [
                "row_number",
                "file_name",
                "expected",
                "ml_prediction",
                "ml_confidence",
                "classify_text_type_prediction",
                "classify_text_type_confidence",
                "text",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(errors)
        print(f"Incorrect rows written: {errors_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
