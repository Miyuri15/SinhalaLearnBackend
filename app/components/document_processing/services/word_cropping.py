import cv2
from pathlib import Path


def crop_words_from_line(image_path: Path, output_dir: Path):
    """
    Crop individual words from a document line image using contour detection.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save cropped word images
        
    Returns:
        Number of words successfully cropped
    """
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Text becomes white, background becomes black
    _, th = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Dilate horizontally to connect characters into words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 3))
    dilated = cv2.dilate(th, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Remove small noise
        if w > 10 and h > 10:
            boxes.append((x, y, w, h))

    # Sort words from left to right
    boxes = sorted(boxes, key=lambda b: b[0])

    crop_count = 0

    for i, (x, y, w, h) in enumerate(boxes, start=1):
        padding = 5

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, image.shape[1])
        y2 = min(y + h + padding, image.shape[0])

        word_crop = image[y1:y2, x1:x2]

        if word_crop.size == 0:
            continue

        output_name = f"{image_path.stem}_word_{i:03d}.png"
        output_path = output_dir / output_name

        cv2.imwrite(str(output_path), word_crop)
        crop_count += 1

    return crop_count
