"""
extractors/image_extractor.py
------------------------------
Detects table structure in images using OpenCV, then dispatches to OCR.

Error routing logic (unified for all image formats):

  GATE 1 — Resolution hard-minimum (before any heavy processing):
    If image is too small to be readable → reject with resolution error.
    This prevents spending 25s on a guaranteed-failure.

  GATE 2 — Table structure check:
    If image has no table → image_no_table(), regardless of OCR mode.
    This gives the correct message to flower.jpg, cat.gif, etc.
    A cat photo is rejected because "no table", NOT because "OCR disabled".

  GATE 3 — OCR mode check (only reached if table IS present):
    If table detected but allow_ocr=False → image_ocr_disabled().
    This message only appears when OCR would actually help.

This order ensures all image formats (.jpg, .bmp, .png, .gif) get
consistent, accurate rejection messages.
"""

import cv2
import pandas as pd

from bullpholio.core.errors import (
    image_ocr_disabled,
    image_no_table,
    image_ocr_no_results,
    image_resolution_too_low,
)
from bullpholio.extractors.normaliser import _normalise_dataframe

# Minimum dimensions for OCR to be worth attempting.
# EasyOCR's CRAFT detector needs ~20px character height.
# A 400×250px image after 3× upscale gives ~750px, giving ~15px char height —
# marginally workable. Below this, readtext burns 18s and returns nothing.
_MIN_OCR_WIDTH  = 400
_MIN_OCR_HEIGHT = 250


def _get_image_dimensions(image_path: str) -> tuple[int, int]:
    """Return (width, height) of an image, or (0, 0) on failure."""
    img = cv2.imread(image_path)
    if img is None:
        return 0, 0
    h, w = img.shape[:2]
    return w, h


def _has_table_structure(image_path: str, min_lines: int = 4) -> bool:
    """
    Use OpenCV to decide whether an image contains a table.

    Stage 1 — grid lines: morphological line detection + connectedComponents.
    Stage 2 — borderless text-alignment fallback: contour-based column/row
               bucket detection (pure OpenCV, no tesseract required).

    Returns False if cv2 cannot read the image (e.g. corrupt/unsupported GIF frames).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_h, img_w = binary.shape

    # Stage 1: grid line detection
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(img_w // 10, 20), 1))
    h_lines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(img_h // 10, 20)))
    v_lines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    h_count = min(cv2.connectedComponents(h_lines)[0] - 1, 200)
    v_count = min(cv2.connectedComponents(v_lines)[0] - 1, 200)

    if h_count >= min_lines and v_count >= min_lines:
        return True

    # Stage 2: borderless text-alignment heuristic
    cell_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(img_w // 40, 8), max(img_h // 80, 4)),
    )
    dilated  = cv2.dilate(binary, cell_kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area   = (img_w * img_h) * 0.0001
    max_area   = (img_w * img_h) * 0.10
    max_blob_h = img_h * 0.15
    max_blob_w = img_w * 0.90
    MIN_ASPECT = 2.0

    total_contours = len(contours)
    cx_list: list[int] = []
    cy_list: list[int] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if not (min_area < area < max_area):
            continue
        if bw < 6 or bh < 4:
            continue
        if bh > max_blob_h or bw > max_blob_w:
            continue
        if (bw / bh) < MIN_ASPECT:
            continue
        cx_list.append(x + bw // 2)
        cy_list.append(y + bh // 2)

    if total_contours > 0 and (len(cx_list) / total_contours) < 0.15:
        return False
    if len(cx_list) < 6:
        return False

    x_bucket = max(1, img_w // 33)
    y_bucket = max(1, img_h // 50)
    col_buckets = len(set(cx // x_bucket for cx in cx_list))
    row_buckets = len(set(cy // y_bucket for cy in cy_list))

    if not (col_buckets >= 2 and row_buckets >= 2 and col_buckets * row_buckets >= 4):
        return False

    total_x_buckets = max(1, img_w // x_bucket)
    x_fill_ratio    = col_buckets / total_x_buckets
    if col_buckets > 8 and x_fill_ratio > 0.55:
        return False

    col_bucket_counts: dict[int, int] = {}
    for cx in cx_list:
        b = cx // x_bucket
        col_bucket_counts[b] = col_bucket_counts.get(b, 0) + 1
    multi_row_cols = sum(1 for cnt in col_bucket_counts.values() if cnt >= 2)
    if multi_row_cols < 2:
        return False

    return True


def _extract_tables_from_image(
    file_path: str,
    allow_ocr: bool = False,
    warnings: list[str] | None = None,
) -> list[pd.DataFrame]:
    """
    Extract tables from an image file.

    Gate order (same for ALL image formats — jpg, bmp, png, gif):

      1. Resolution check  → image_resolution_too_low (fast fail, <0.1s)
      2. Table check       → image_no_table           (catches photos, GIFs, charts)
      3. OCR mode check    → image_ocr_disabled        (only shown when table IS present)
      4. OCR extraction    → image_ocr_no_results      (shown after actual attempt)

    Gate 2 runs BEFORE Gate 3 so that a flower photo or cat GIF always gets
    "no table" — never "OCR disabled" — regardless of OCR mode setting.
    """
    if warnings is None:
        warnings = []

    # ── GATE 1: resolution hard minimum ──────────────────────────────────────
    # Do this before any heavy processing. OCR cannot read images below this
    # threshold even after upscaling (CRAFT needs ~20px character height).
    # This prevents burning 25s on a guaranteed failure like trans.png.
    w, h = _get_image_dimensions(file_path)
    is_too_small = (w > 0 and h > 0 and w < _MIN_OCR_WIDTH and h < _MIN_OCR_HEIGHT)
    if is_too_small:
        # Even if allow_ocr=True, OCR would fail anyway — reject with resolution error.
        raise ValueError(image_resolution_too_low(w, h, file_path))

    # Non-fatal warning for borderline images (small in one dimension only)
    if w > 0 and (w < _MIN_OCR_WIDTH or h < _MIN_OCR_HEIGHT):
        warnings.append(image_resolution_too_low(w, h, file_path))

    # ── GATE 2: table structure check (BEFORE OCR mode check) ────────────────
    # A non-table image (photo, logo, animation) gets image_no_table()
    # regardless of whether OCR is enabled.
    # This is the correct message: the issue is "no table", not "no OCR".
    if not _has_table_structure(file_path):
        raise ValueError(image_no_table(file_path))

    # ── GATE 3: OCR mode check ────────────────────────────────────────────────
    # Only reached when a real table structure was detected.
    # Now it makes sense to tell the user "enable OCR to read this table".
    if not allow_ocr:
        raise ValueError(image_ocr_disabled())

    # ── GATE 4: OCR extraction ────────────────────────────────────────────────
    from bullpholio.extractors.ocr_extractor import _ocr_to_dataframe

    df = _ocr_to_dataframe(file_path)
    df = _normalise_dataframe(df)
    if df is None:
        raise ValueError(image_ocr_no_results(file_path))
    return [df]
