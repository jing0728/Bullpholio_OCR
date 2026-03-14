"""
extractors/image_extractor.py
------------------------------
Image validation, table detection, and OCR dispatch.

Unified rejection path — ALL image formats (jpg/png/bmp/gif/webp/tiff) go
through the same gate in this exact order:

  1. FAST QUALITY GATE  (~5-50ms, OpenCV only, no EasyOCR)
     a. Dimension check  — width OR height below hard floor → fast fail
     b. Text density     — estimate text pixel coverage; too sparse → fast fail
     c. Table structure  — morphological line/contour detection → no table → fail

  2. OCR MODE CHECK  (only reached if table was detected)
     → allow_ocr=False → image_ocr_disabled()

  3. OCR  (only reached for table images with OCR enabled and good quality)

This order guarantees:
  - cat.gif / flower.jpg / flower.bmp  → rejected at step 1c ("no table"),
    regardless of allow_ocr setting
  - trans.png (330×180px)              → rejected at step 1a ("resolution too low"),
    no OCR attempted at all  →  ~50ms instead of ~40s
  - stock.png / SPDR.png               → pass all gates, run OCR
"""

import cv2
import numpy as np
import pandas as pd

from bullpholio.core.errors import (
    image_ocr_disabled,
    image_no_table,
    image_ocr_no_results,
    image_resolution_too_low,
)
from bullpholio.extractors.normaliser import _normalise_dataframe


# ── Thresholds ────────────────────────────────────────────────────────────────

# Hard floor: width below this means EasyOCR cannot recover useful text even
# after aggressive upscaling.  Testing shows 330px-wide images (trans.png)
# produce 0 tokens regardless of preprocessing; 400px+ images work well.
# Using width (not short side) because tables are always wider than tall.
_MIN_WIDTH_FOR_OCR = 600   # pixels — fast fail if image width < this
                           # 404×125 (trans.png) proved unreadable at 400; 600 is safe floor
_MIN_OCR_WIDTH     = _MIN_WIDTH_FOR_OCR   # alias used by ocr_extractor
_MIN_OCR_HEIGHT    = 100                  # alias used by ocr_extractor (height floor)

# Soft floor warning (non-fatal, just a note in warnings)
_SOFT_MIN_WIDTH = 600      # pixels

# Text density: fraction of pixels classified as dark after adaptive threshold.
# Pure photos/logos score < 0.005; financial table screenshots score > 0.02.
_MIN_TEXT_DENSITY = 0.01   # 1% — very permissive; catches blank/photo images


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_image_dimensions(image_path: str) -> tuple[int, int]:
    """Return (width, height), or (0, 0) on failure (e.g. animated GIF)."""
    img = cv2.imread(image_path)
    if img is None:
        return 0, 0
    h, w = img.shape[:2]
    return w, h


def _estimate_text_density(image_path: str) -> float:
    """
    Quick text-presence estimate via adaptive threshold.
    Returns fraction of pixels classified as 'dark' (text/line) in [0, 1].
    Pure photos/logos typically score < 0.005; table screenshots score > 0.02.
    Takes ~5-10ms on a typical image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    # Resize to max 400px wide for speed
    h, w = img.shape[:2]
    if w > 400:
        scale = 400 / w
        img = cv2.resize(img, (400, int(h * scale)), interpolation=cv2.INTER_AREA)
    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=4,
    )
    return float(binary.sum() // 255) / binary.size


def _has_table_structure(image_path: str, min_lines: int = 4,
                         text_density: float | None = None) -> bool:
    """
    Use OpenCV to decide whether an image contains a table.

    Stage 1 — grid lines: morphological line detection + connectedComponents.
    Stage 2 — borderless text-alignment fallback: contour-based column/row
               bucket detection (pure OpenCV, no OCR required).
              Parameters relaxed vs original to handle modern web UI screenshots
              (Morningstar, Koyfin, StockRover, etc.) which have:
                - no visible grid lines (borderless / zebra-stripe rows)
                - navigation bars, tabs, and sidebar elements
    Stage 3 — high-density text grid fallback: if text_density > 0.03, the
              image is almost certainly a document/UI screenshot (not a photo).
              Apply much more lenient contour analysis with no aspect-ratio
              filter and coarser spatial buckets.
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

    # ── Shared contour extraction for Stage 2 + 3 ────────────────
    cell_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(img_w // 40, 8), max(img_h // 80, 4)),
    )
    dilated = cv2.dilate(binary, cell_kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area   = (img_w * img_h) * 0.0001
    max_area   = (img_w * img_h) * 0.10
    max_blob_h = img_h * 0.15
    max_blob_w = img_w * 0.90

    total_contours = len(contours)

    # Stage 2: borderless text-alignment heuristic (RELAXED)
    # Changes vs original thresholds:
    #   - aspect ratio filter:  (bw / bh) < 2.0  ->  < 1.0
    #     Web UI cells can be nearly square (short ticker + padding).
    #   - contour ratio gate:   < 0.15  ->  < 0.08
    #     Web UIs have many non-table contours (buttons, nav, icons).
    #   - min blobs:  6  ->  4
    #     Smaller visible table area in full-page screenshots.
    #   - col spread gate:  > 0.55  ->  > 0.70
    #     Dense financial tables legitimately fill most horizontal space.
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
        if (bw / bh) < 1.0:           # relaxed: allow square-ish web UI cells
            continue
        cx_list.append(x + bw // 2)
        cy_list.append(y + bh // 2)

    if total_contours > 0 and (len(cx_list) / total_contours) < 0.08:  # relaxed for web UIs
        pass  # do not return False yet — Stage 3 may rescue
    elif len(cx_list) < 4:  # relaxed: smaller visible table area in full-page screenshots
        pass  # do not return False yet — Stage 3 may rescue
    else:
        x_bucket = max(1, img_w // 33)
        y_bucket = max(1, img_h // 50)
        col_buckets = len(set(cx // x_bucket for cx in cx_list))
        row_buckets = len(set(cy // y_bucket for cy in cy_list))

        if col_buckets >= 2 and row_buckets >= 2 and col_buckets * row_buckets >= 4:
            total_x_buckets = max(1, img_w // x_bucket)
            if not (col_buckets > 8 and (col_buckets / total_x_buckets) > 0.70):  # relaxed for dense tables
                col_bucket_counts: dict[int, int] = {}
                for cx in cx_list:
                    b = cx // x_bucket
                    col_bucket_counts[b] = col_bucket_counts.get(b, 0) + 1
                if sum(1 for c in col_bucket_counts.values() if c >= 2) >= 2:
                    return True

    # ── Stage 3: high-density text grid fallback ─────────────────
    # If text density > 3%, the image is almost certainly a document or
    # UI screenshot, NOT a photo.  Apply lenient contour analysis that
    # skips the aspect-ratio filter entirely and uses coarser buckets.
    # This rescues web UI screenshots (Morningstar, Koyfin, StockRover)
    # that fail Stage 2 due to nav bars, tabs, and dark themes.
    if text_density is not None and text_density > 0.03:
        cx3: list[int] = []
        cy3: list[int] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if not (min_area < area < max_area):
                continue
            if bw < 4 or bh < 3:
                continue
            if bh > max_blob_h or bw > max_blob_w:
                continue
            # No aspect ratio filter — allow any blob shape
            cx3.append(x + bw // 2)
            cy3.append(y + bh // 2)

        if len(cx3) >= 6:
            x_bucket3 = max(1, img_w // 25)   # coarser buckets than Stage 2
            y_bucket3 = max(1, img_h // 40)
            col3 = len(set(cx // x_bucket3 for cx in cx3))
            row3 = len(set(cy // y_bucket3 for cy in cy3))
            if col3 >= 2 and row3 >= 3:
                return True

    return False


# ── Main extractor ────────────────────────────────────────────────────────────

def _extract_tables_from_image(
    file_path: str,
    allow_ocr: bool = False,
    warnings: list[str] | None = None,
) -> list[pd.DataFrame]:
    """
    Extract tables from an image file.

    Gate order (all fast OpenCV — total overhead < 100ms):
      1a. Min short-side check  → fast fail if too small for OCR
      1b. Text density check    → fast fail if image is a photo/logo
      1c. Table structure check → fast fail if no table layout detected
      2.  OCR mode check        → fail if table found but allow_ocr=False
      3.  OCR extraction

    For non-table images (photos/logos/gifs), the error is ALWAYS
    image_no_table() — never image_ocr_disabled() — regardless of allow_ocr.
    This prevents confusing messages like "enable OCR" on a flower photo.
    """
    if warnings is None:
        warnings = []

    w, h = _get_image_dimensions(file_path)
    short_side = min(w, h) if (w > 0 and h > 0) else 0

    # ── Gate 1a: hard minimum width ──────────────────────────────────────────
    # Images narrower than 400px cannot be read by EasyOCR even after 4x
    # upscaling — CRAFT text detector produces 0 tokens at that resolution.
    # This saves ~40s for trans.png (330x180) by skipping all OCR entirely.
    # Also catches animated GIFs / corrupt files where cv2 returns (0, 0).
    if w < _MIN_WIDTH_FOR_OCR:
        if allow_ocr:
            raise ValueError(image_resolution_too_low(w, h, file_path))
        else:
            raise ValueError(image_no_table(file_path))

    # Soft warning for borderline images (pass but warn)
    if w < _SOFT_MIN_WIDTH:
        warnings.append(image_resolution_too_low(w, h, file_path))

    # ── Gate 1b: text density pre-check ──────────────────────────────────────
    # Pure photos, logos, and blank images have very few dark pixels after
    # adaptive threshold.  Reject before the heavier table structure check.
    density = _estimate_text_density(file_path)
    if density < _MIN_TEXT_DENSITY:
        raise ValueError(image_no_table(file_path))

    # ── Gate 1c: table structure ──────────────────────────────────────────────
    # Full morphological table detection.  A non-table image always gets
    # image_no_table(), regardless of allow_ocr — the absence of a table is the
    # real reason, not the OCR setting.
    # Pass pre-computed density so Stage 3 (high-density fallback) can use it.
    if not _has_table_structure(file_path, text_density=density):
        raise ValueError(image_no_table(file_path))

    # ── Gate 2: OCR mode check ────────────────────────────────────────────────
    # Table exists — now check if we're allowed to read it.
    if not allow_ocr:
        raise ValueError(image_ocr_disabled())

    # ── Gate 3: OCR ──────────────────────────────────────────────────────────
    from bullpholio.extractors.ocr_extractor import _ocr_to_dataframe

    df = _ocr_to_dataframe(file_path)
    df = _normalise_dataframe(df)
    if df is None:
        raise ValueError(image_ocr_no_results(file_path))
    return [df]
