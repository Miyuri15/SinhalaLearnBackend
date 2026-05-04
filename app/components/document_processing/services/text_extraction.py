import pdfplumber
import pytesseract
import cv2
import numpy as np
import os
from typing import Literal, Optional, Callable, Dict, Any, Tuple, Union

from app.components.document_processing.services.table_detection import detect_tables_with_yolo
from app.components.document_processing.services.trocr_extraction import trocr_predict as default_trocr_predict
from app.components.document_processing.ocr_config import OCR_LANG, OCR_CONFIG_EXTRA

import logging
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception as e:
    logger.warning(f"Ultralytics YOLO not available: {e}")
    YOLO = None
    ULTRALYTICS_AVAILABLE = False
layout_model = None
if ULTRALYTICS_AVAILABLE:
    try:
        layout_model = YOLO("utils/yolov8m-doclaynet.pt")
    except Exception as e:
        logger.error(f"Failed to load layout YOLO model: {e}")
        layout_model = None

lang = OCR_LANG  # Tesseract language setting from config

TextTypeLabel = Literal["handwritten", "printed", "unknown"]
TextTypeResult = Tuple[TextTypeLabel, float]


def _format_text_type_result(
    label: TextTypeLabel,
    confidence: float,
    return_confidence: bool,
) -> Union[TextTypeLabel, TextTypeResult]:
    if return_confidence:
        return label, confidence
    return label

def classify_text_type(
    image_input: Union[str, np.ndarray],
    ml_model_predict: Optional[Callable[[np.ndarray], TextTypeResult]] = None,
    ml_conf_threshold: float = 0.7,
    return_confidence: bool = False,
) -> Union[TextTypeLabel, TextTypeResult]:
    """
    Classify text as handwritten or printed.
    
    Args:
        image_input: Either a file path (str) or numpy array (from PIL/cv2)
        ml_model_predict: Optional ML model callback returning (label, confidence)
        ml_conf_threshold: Minimum confidence to trust ML output
        return_confidence: If True, return (label, confidence)
    """
    try:
        # Handle both file path and numpy array inputs
        if isinstance(image_input, str):
            img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Cannot read image for classification: {image_input}")
                return _format_text_type_result("unknown", 0.0, return_confidence)
        else:
            img = image_input
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Normalize contrast
        img = cv2.equalizeHist(img)

        # Binarize (text = white)
        _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Remove noise
        bw = cv2.medianBlur(bw, 3)

        # ---- Stroke Width Estimation (distance transform) ----
        dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)

        # stroke width = distance * 2 (approx)
        stroke_width = dist[bw > 0] * 2

        mean_sw = np.mean(stroke_width)
        var_sw = np.var(stroke_width)

        # ---- Edge density ----
        edges = cv2.Canny(img, 80, 160)
        edge_density = np.sum(edges > 0) / edges.size

        # ---- Component irregularity ----
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        aspect_ratios = []
        for c in contours:
            _, _, w, h = cv2.boundingRect(c)
            if w * h < 20:  # ignore tiny dots
                continue
            aspect_ratios.append(w / float(h))

        var_ar = np.var(aspect_ratios) if aspect_ratios else 0

        # -------- Decision Rules --------
        # Tuned for Sinhala printed vs handwriting
        handwritten_score = 0

        if var_sw > 6:              # handwriting strokes vary more
            handwritten_score += 1
        if edge_density < 0.02:     # handwriting usually has fewer sharp edges
            handwritten_score += 1
        if var_ar > 0.35:           # handwritten characters vary shape more
            handwritten_score += 1

        if handwritten_score >= 2:
            rule_result: TextTypeLabel = "handwritten"
        else:
            rule_result = "printed"

        # Heuristic confidence from the margin in voting rules.
        rule_confidence = 0.55 + (handwritten_score / 3.0) * 0.4 if rule_result == "handwritten" else 0.75

        result = rule_result
        confidence = float(min(max(rule_confidence, 0.0), 1.0))

        if ml_model_predict is not None:
            try:
                ml_label, ml_confidence = ml_model_predict(img)
                if ml_label in {"printed", "handwritten", "unknown"} and ml_confidence >= ml_conf_threshold:
                    result = ml_label
                    confidence = float(min(max(ml_confidence, 0.0), 1.0))
            except Exception as ml_err:
                logger.warning("ML text type prediction failed, using rule-based fallback: %s", ml_err)

        logger.info(
            f"Text classification: {result} (confidence={confidence:.3f}) "
            f"(mean_sw={mean_sw:.2f}, var_sw={var_sw:.2f}, "
            f"edge_density={edge_density:.4f}, var_ar={var_ar:.4f})"
        )

        return _format_text_type_result(result, confidence, return_confidence)

    except Exception as e:
        logger.error(f"Error classifying text type: {e}")
        return _format_text_type_result("unknown", 0.0, return_confidence)


def detect_language_from_text(text: str) -> Literal["sinhala", "english", "mixed", "unknown"]:
    """Lightweight heuristic to flag dominant script in extracted text."""
    sinhala = sum(1 for ch in text if 0x0D80 <= ord(ch) <= 0x0DFF)
    latin = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    total = sinhala + latin

    if total == 0:
        return "unknown"

    sinhala_ratio = sinhala / total
    latin_ratio = latin / total

    if sinhala_ratio >= 0.7:
        return "sinhala"
    if latin_ratio >= 0.7:
        return "english"
    return "mixed"

def extract_text_from_pdf(file_path: str) -> tuple:
    with pdfplumber.open(file_path) as pdf:
        text = ""
        logger.info("Extracting text from %d pages...", len(pdf.pages))
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text() or ""
            text += f"\n\n--- PAGE {page_num} ---\n{page_text}"
    return text, len(pdf.pages)


def crop_region(img, region):
    """Safely crop an [x1, y1, x2, y2] region from an image array."""
    x1, y1, x2, y2 = map(int, region)
    h, w = img.shape[:2]

    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return img[0:0, 0:0]

    return img[y1:y2, x1:x2]

def process_ocr_for_images_with_tables(
    images, 
    force_layout_analysis: bool = False,
    progress_callback: Optional[Callable[[str, float, Optional[Dict[str, Any]]], None]] = None,
    ml_model_predict: Optional[Callable[[np.ndarray], TextTypeResult]] = None,
    ml_conf_threshold: float = 0.7,
    trocr_predict: Optional[Callable[[np.ndarray], str]] = None,
    region_conf_threshold: float = 0.7,
) -> tuple:
    print(f"LANG VALUE: {lang}")
    """
    OCR pipeline with:
    - Table detection
    - Layout detection (DocLayNet)
    - Column clustering
    - Reading order reconstruction
    - Block-wise OCR
    """

    extracted_text = ""
    page_count = 0
    total_images = len(images) if images else 1

    if trocr_predict is None:
        trocr_predict = default_trocr_predict

     # Fixed progress ranges
    START_PERCENT = 12.0  # Start of OCR phase
    END_PERCENT = 40.0     # End of OCR phase (before "Completed OCR Extraction")
    RANGE = END_PERCENT - START_PERCENT

    PROGRESS_PER_PAGE = RANGE / total_images

    for idx, pil_img in enumerate(images):
        page_count += 1
        
        # Calculate base progress for this page
        page_base = START_PERCENT + (idx * PROGRESS_PER_PAGE)
        
        # Layout analysis gets 40% of this page's allocation
        # OCR extraction gets 60% of this page's allocation
        layout_share = PROGRESS_PER_PAGE * 0.4
        ocr_share = PROGRESS_PER_PAGE * 0.6

        layout_progress = page_base + (layout_share * 0.5)  # Mid-point of layout share
        ocr_progress = page_base + layout_share + (ocr_share * 0.5)  # After layout

        if progress_callback:
            details = {
                "current_page": page_count,
                "total_pages": total_images,
                "current_action": f"Layout analysis page {page_count} of {total_images}"
            }

            progress_callback("Layout Analysis", layout_progress, details)

        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        logger.debug("TESSDATA_PREFIX: %s", os.environ.get("TESSDATA_PREFIX"))

        tess_config = (
            "--oem 1 "
            "--psm 6 "
            "-c preserve_interword_spaces=1 "
        )

        # -------------------------------------
        # 1️⃣ Detect tables
        # -------------------------------------
        table_coords, num_tables = detect_tables_with_yolo(img)
        logger.info(f"Page {page_count}: Detected {num_tables} tables.")

        # Create a mask for the table areas to exclude them from OCR
        table_mask = np.zeros_like(gray)
        for coords in table_coords:
            x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
            table_mask[y1:y2, x1:x2] = 255  # Mark table areas with white in the mask

        # If force_layout_analysis is True, always perform layout detection
        if force_layout_analysis:
            # -------------------------------------
            # 2️⃣ Detect layout excluding tables
            # -------------------------------------
            text_regions = detect_layout_excluding_tables(img, table_coords)
            classified_regions: list[dict[str, Any]] = []

            for idx, region in enumerate(text_regions, start=1):
                crop = crop_region(img, region)
                region_text_type, region_confidence = classify_text_type(
                    crop,
                    ml_model_predict=ml_model_predict,
                    ml_conf_threshold=ml_conf_threshold,
                    return_confidence=True,
                )
                classified_regions.append(
                    {
                        "box": region,
                        "type": region_text_type,
                        "confidence": float(region_confidence),
                    }
                )
                logger.debug(
                    "Region %d classified as %s (confidence=%.3f)",
                    idx,
                    region_text_type,
                    region_confidence,
                )

            # -------------------------------------
            # 3️⃣ Column clustering
            # -------------------------------------
            columns = detect_columns([region["box"] for region in classified_regions], img.shape[1])

            # -------------------------------------
            # 4️⃣ Reading order reconstruction
            # -------------------------------------
            reading_order_boxes = sort_regions_by_reading_order(columns)
            region_lookup = {tuple(r["box"]): r for r in classified_regions}
            reading_order = [
                region_lookup.get(tuple(box), {"box": box, "type": "unknown", "confidence": 0.0})
                for box in reading_order_boxes
            ]

        else:
            # If force_layout_analysis is False, skip layout analysis
            reading_order = []

        # -------------------------------------
        # 5️⃣ OCR TEXT BLOCKS IN ORDER
        # -------------------------------------
        if progress_callback:
            details = {
                "current_page": page_count,
                "total_pages": total_images,
                "current_action": f"OCR extraction page {page_count} of {total_images}",
                "tables_detected": num_tables,
            }

            progress_callback("OCR Extraction", ocr_progress, details)

        page_text = ""

        if reading_order:
            for region_info in reading_order:
                box = region_info["box"]
                region_text_type = region_info.get("type", "unknown")
                region_confidence = float(region_info.get("confidence", 0.0))

                x1, y1, x2, y2 = box

                crop = gray[y1:y2, x1:x2]

                use_trocr = (
                    trocr_predict is not None
                    and region_text_type == "handwritten"
                    and region_confidence >= region_conf_threshold
                )

                if use_trocr:
                    try:
                        text = trocr_predict(crop)
                    except Exception as trocr_error:
                        logger.warning(
                            "TrOCR failed for region; using Tesseract fallback: %s",
                            trocr_error,
                        )
                        text = pytesseract.image_to_string(crop, lang=lang, config=tess_config)
                else:
                    if region_text_type == "unknown" or region_confidence < region_conf_threshold:
                        fallback_tess_config = (
                            "--oem 1 "
                            "--psm 11 "
                            "-c preserve_interword_spaces=1 "
                        )
                        text = pytesseract.image_to_string(crop, lang=lang, config=fallback_tess_config)
                    else:
                        text = pytesseract.image_to_string(crop, lang=lang, config=tess_config)

                page_text += text.strip() + "\n\n"
        else:
            # Perform OCR on the entire page excluding table areas
            # Use bitwise NOT to exclude the table areas from OCR
            non_table_area = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(table_mask))

            logger.debug("TESSDATA_PREFIX: %s", os.environ.get("TESSDATA_PREFIX"))

            page_text_type, page_confidence = classify_text_type(
                non_table_area,
                ml_model_predict=ml_model_predict,
                ml_conf_threshold=ml_conf_threshold,
                return_confidence=True,
            )
            use_trocr = (
                trocr_predict is not None
                and page_text_type == "handwritten"
                and page_confidence >= region_conf_threshold
            )

            if use_trocr:
                try:
                    full_page_text = trocr_predict(non_table_area)
                except Exception as trocr_error:
                    logger.warning(
                        "TrOCR failed for page; using Tesseract fallback: %s",
                        trocr_error,
                    )
                    full_page_text = pytesseract.image_to_string(
                        non_table_area,
                        lang=lang,
                        config=tess_config
                    )
            else:
                full_page_text = pytesseract.image_to_string(
                    non_table_area,
                    lang=lang,
                    config=tess_config
                )
            page_text = full_page_text.strip()

        # -------------------------------------
        # 6️⃣ OCR TABLES SEPARATELY
        # -------------------------------------
        table_texts = []

        for t_idx, coords in enumerate(table_coords):
            x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]

            table_crop = gray[y1:y2, x1:x2]

            logger.debug("TESSDATA_PREFIX: %s", os.environ.get("TESSDATA_PREFIX"))

            table_config = (
                "--oem 1 "
                "--psm 6 "
                "-c preserve_interword_spaces=1 "
                "-c textord_tablefind_good_text_size=12 "
            )

            t_text = pytesseract.image_to_string(
                table_crop,
                lang=lang,
                config=table_config
            )

            table_texts.append(
                f"\n\n--- TABLE {t_idx + 1} (Page {page_count}) ---\n{t_text.strip()}"
            )

        # -------------------------------------
        # 7️⃣ Merge page result
        # -------------------------------------
        page_output = (
            f"\n\n--- PAGE {page_count} ---\n"
            + page_text
            + "\n".join(table_texts)
        )

        extracted_text += page_output

    return extracted_text, page_count

def detect_layout_excluding_tables(img, table_coords, conf_threshold=0.6):
    """
    Detect layout regions (Text, Title, Section-header)
    excluding areas overlapping with already detected tables.
    """

    results = layout_model(img, imgsz=1024)

    text_regions = []

    def overlaps(box, table):
        x1, y1, x2, y2 = box
        tx1, ty1, tx2, ty2 = table
        return not (x2 < tx1 or x1 > tx2 or y2 < ty1 or y1 > ty2)

    # Convert table coords format
    formatted_tables = [
        (t["x1"], t["y1"], t["x2"], t["y2"])
        for t in table_coords
    ]

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = layout_model.names[cls_id]
            conf = float(box.conf[0])

            if conf < conf_threshold:
                continue

            xyxy = list(map(int, box.xyxy[0].tolist()))

            if class_name in ["Text", "Title", "Section-header"]:

                # Exclude if overlapping with table
                if any(overlaps(xyxy, t) for t in formatted_tables):
                    continue

                text_regions.append(xyxy)

    return text_regions

def detect_columns(text_blocks, image_width):
    """
    Cluster text blocks into columns using horizontal CENTER distance.
    More stable than x_min clustering.
    """

    if not text_blocks:
        return []

    # Sort by horizontal center
    text_blocks = sorted(
        text_blocks,
        key=lambda b: (b[0] + b[2]) / 2
    )

    columns = []

    # Slightly larger threshold for robustness
    column_threshold = image_width * 0.12

    for box in text_blocks:
        box_center = (box[0] + box[2]) / 2
        placed = False

        for col in columns:
            if abs(box_center - col["center"]) < column_threshold:
                col["boxes"].append(box)

                # Update column center using box centers
                col["center"] = np.mean(
                    [(b[0] + b[2]) / 2 for b in col["boxes"]]
                )

                placed = True
                break

        if not placed:
            columns.append({
                "center": box_center,
                "boxes": [box]
            })

    logger.info(
        f"Clustered {len(text_blocks)} text blocks into {len(columns)} columns."
    )

    return columns

def sort_regions_by_reading_order(columns):
    """
    Sort columns left-to-right,
    then blocks top-to-bottom within each column.
    """

    if not columns:
        return []

    # Sort columns by horizontal position
    columns = sorted(columns, key=lambda c: c["center"])

    reading_order = []

    for col in columns:
        sorted_boxes = sorted(col["boxes"], key=lambda b: b[1])
        reading_order.extend(sorted_boxes)

    logger.info(f"Sorted {len(reading_order)} text regions into reading order.")
    return reading_order
