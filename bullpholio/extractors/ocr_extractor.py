"""
extractors/ocr_extractor.py
----------------------------
Image preprocessing and PaddleOCR-based table extraction.

Preprocessing uses a multi-strategy approach:
  Strategy 1 (default):  tiered upscale + auto-invert + CLAHE + unsharp mask
  Strategy 2 (adaptive): adaptive threshold on grayscale (high contrast boost)
  Strategy 3 (plain):    plain grayscale, no enhancement (fallback)

The strategy producing the most OCR tokens is used for table reconstruction.
This improves accuracy on very low-resolution images like broker screenshots.
"""

import difflib
import tempfile
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from bullpholio.constants.column_aliases import (
    HOLDING_COLUMN_ALIASES,
    TRANSACTION_COLUMN_ALIASES,
)

# ── Preprocessing helpers ─────────────────────────────────────────────────────

def _unsharp_mask(gray: np.ndarray, sigma: float = 1.0,
                  strength: float = 1.5) -> np.ndarray:
    blurred   = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharpened = cv2.addWeighted(gray, strength, blurred, -(strength - 1), 0)
    return sharpened


def _preprocess_strategy_1(image_path: str) -> tuple[str, bool]:
    """
    Default strategy: tiered upscale + auto-invert + CLAHE + unsharp mask.

    Upscale targets:
      Tier C (w < 400):  target 1600px
      Tier B (w < 800):  target 1800px
      Tier A (w >= 800): no upscale unless soft

    Returns (output_path, was_modified).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path, False

        h, w = img.shape[:2]
        modified = False

        if w < 400:
            scale = max(2.0, 1600 / w)
            img   = cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LANCZOS4)
            modified = True
            _tier = "C"
        elif w < 800:
            scale = max(2.0, 1800 / w)
            img   = cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LANCZOS4)
            modified = True
            _tier = "B"
        else:
            _tier = "A"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Auto-invert dark backgrounds
        if (gray < 80).sum() / gray.size > 0.55:
            gray     = cv2.bitwise_not(gray)
            modified = True

        # CLAHE only for genuinely low-contrast images
        _clahe_lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        if gray.std() < 50 and _clahe_lap < 200:
            clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray     = clahe.apply(gray)
            modified = True

        # Unsharp mask
        if _tier in ("B", "C"):
            strength = 2.0 if _tier == "C" else 1.5
            gray     = _unsharp_mask(gray, sigma=1.0, strength=strength)
            modified = True
        else:
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var < 100:
                gray     = _unsharp_mask(gray, sigma=1.0, strength=1.5)
                modified = True

        if not modified:
            return image_path, False

        tmp = Path(tempfile.gettempdir()) / "_bullpholio_s1.png"
        cv2.imwrite(str(tmp), gray)
        return str(tmp), True

    except Exception:
        return image_path, False


def _preprocess_strategy_2(image_path: str) -> tuple[str, bool]:
    """
    Adaptive threshold strategy: high contrast boost via adaptive binarisation.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path, False

        h, w = img.shape[:2]

        if w < 400:
            scale = max(2.0, 1600 / w)
            img   = cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LANCZOS4)
        elif w < 800:
            scale = max(2.0, 1800 / w)
            img   = cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LANCZOS4)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if (gray < 80).sum() / gray.size > 0.55:
            gray = cv2.bitwise_not(gray)

        gray = cv2.fastNlMeansDenoising(gray, h=7,
                                        templateWindowSize=7,
                                        searchWindowSize=21)

        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15, C=4,
        )

        tmp = Path(tempfile.gettempdir()) / "_bullpholio_s2.png"
        cv2.imwrite(str(tmp), binary)
        return str(tmp), True

    except Exception:
        return image_path, False


def _preprocess_strategy_3(image_path: str) -> tuple[str, bool]:
    """
    Plain grayscale fallback — no enhancements except upscale.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path, False

        h, w = img.shape[:2]
        if w >= 800:
            return image_path, False

        scale = max(2.0, 1600 / w)
        img   = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_LANCZOS4)
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        tmp = Path(tempfile.gettempdir()) / "_bullpholio_s3.png"
        cv2.imwrite(str(tmp), gray)
        return str(tmp), True

    except Exception:
        return image_path, False


# ── Module-level init ────────────────────────────────────────────────────────
# Nothing needed for PaddleOCR 2.x on Windows — oneDNN issues only affect 3.x.

# ── PaddleOCR Reader cache ────────────────────────────────────────────────────
# Loading PaddleOCR models takes a few seconds on first call.
# We cache the instance at module level so the cost is paid once per process.

_READER_CACHE: object = None  # PaddleOCR instance, lazily initialised


def _get_reader():
    """
    Return the cached PaddleOCR instance, initialising it on first call.
    Targets PaddleOCR 2.x (tested on 2.8.1 + paddlepaddle 2.6.2).
    """
    global _READER_CACHE
    if _READER_CACHE is None:
        import logging
        try:
            from paddleocr import PaddleOCR  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("Missing dependency. Run: pip install paddleocr==2.8.1 paddlepaddle==2.6.2")

        # Mute verbose ppocr INFO output.
        for noisy in ("ppocr", "paddle", "paddleocr", "ppstructure"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _READER_CACHE = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            show_log=False,
        )
    return _READER_CACHE


def warmup_ocr() -> None:
    """
    Pre-warm the PaddleOCR model so the first real image doesn't pay
    the model-initialisation cost (~4s on first call after install).
    Call once at program/server startup or at the top of a test suite.
    """
    reader = _get_reader()
    dummy  = np.ones((64, 256, 3), dtype=np.uint8) * 255
    tmp    = Path(tempfile.gettempdir()) / "_ocr_warmup.png"
    cv2.imwrite(str(tmp), dummy)
    _run_paddleocr(reader, str(tmp), conf_min=0.0)


def _run_paddleocr(reader, path: str, conf_min: float = 0.05) -> list[tuple]:
    """
    Run PaddleOCR (2.x) on a single image and return
    (bbox, text, confidence) tuples with conf >= conf_min.

    PaddleOCR 2.x layout:
        raw[0]  — list of [bbox, (text, score)] for the page
        bbox    — [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    try:
        raw = reader.ocr(path, cls=True)

        if not raw or raw[0] is None:
            return []

        results = []
        for line in raw[0]:
            bbox, (text, conf) = line
            text = text.strip()
            if conf >= conf_min and text:
                results.append((bbox, text, conf))

        return results

    except Exception:
        return []


def _ocr_to_dataframe(image_path: str) -> pd.DataFrame:
    """
    Run PaddleOCR on a table image and reconstruct a DataFrame.

    Fast-fail gates (checked before loading the OCR model):
      • Image unreadable by OpenCV → empty DataFrame immediately
      • Both dimensions below OCR minimum → empty DataFrame immediately

    Table reconstruction — two-pass header-first approach:
      Phase 1  Y-cluster all OCR tokens into row groups.
      Phase 2  Score each row group against known column aliases to locate
               the header row (no column assignment needed yet).
      Phase 3  Use only the header row's X positions to define column
               centres.  This avoids false clusters created by dense
               numeric data (commas, decimal points, multi-digit numbers).
      Phase 4  Compute column boundaries as midpoints between adjacent
               centres.  Assign every token via boundaries rather than
               pure nearest-centre to prevent cross-column absorption.
      Phase 5  Build the 2-D grid and return as a DataFrame.
    """
    # ── Pre-OCR resolution gate ───────────────────────────────────────────────
    img_check = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_check is None:
        return pd.DataFrame()
    img_h, img_w = img_check.shape
    from bullpholio.extractors.image_extractor import _MIN_OCR_WIDTH, _MIN_OCR_HEIGHT
    if img_w < _MIN_OCR_WIDTH and img_h < _MIN_OCR_HEIGHT:
        return pd.DataFrame()

    reader = _get_reader()

    # ── Multi-strategy competition ────────────────────────────────
    # Run ALL strategies and pick the one producing the most OCR tokens.
    # conf_min=0.05 is intentionally permissive here — noisy tokens are
    # filtered later by header scoring and alias matching; the bigger
    # risk is having zero tokens at all on low-quality screenshots.

    # ── Strategy selection ────────────────────────────────────────
    # Token threshold for "good enough" — if the original image already
    # produces this many tokens, it is a high-quality screenshot and no
    # preprocessing strategy will meaningfully improve it.  Skipping all
    # strategies for these images saves 5-10 s per call (strategy_2 /
    # adaptive-threshold is the dominant cost and rarely wins on clear
    # screenshots).
    #
    # Empirical calibration:
    #   stock.png (1366px clear):  original=142  → skip → saves ~6 s
    #   SPDR.png  (clear):         original=78   → skip → saves ~6 s
    #   warehouse.png (clear):     original=130  → skip → saves ~7 s
    #   Low-res / blurry images:   original<20   → try strategies as before
    _GOOD_ENOUGH_TOKENS = 20

    orig_results = _run_paddleocr(reader, image_path, conf_min=0.05)
    candidates: list[tuple[str, list]] = [("original", orig_results)]

    if len(orig_results) < _GOOD_ENOUGH_TOKENS:
        # Original is weak — try preprocessing strategies to recover more tokens.

        # Strategy 1: tiered upscale + CLAHE + unsharp mask
        # Only adds a candidate when the image was actually modified (i.e. small
        # or blurry); large clear images fall straight through with modified=False.
        s1_path, s1_modified = _preprocess_strategy_1(image_path)
        if s1_modified:
            s1_results = _run_paddleocr(reader, s1_path, conf_min=0.05)
            candidates.append(("strategy_1", s1_results))

        # Strategy 2: adaptive threshold — only worthwhile for low-token images.
        # Always produces a modified image so we gate it on token count, not
        # modification flag, to avoid a wasted ~5 s OCR call on clear images.
        s2_path, s2_modified = _preprocess_strategy_2(image_path)
        if s2_modified:
            s2_results = _run_paddleocr(reader, s2_path, conf_min=0.05)
            candidates.append(("strategy_2", s2_results))

        # Strategy 3: plain grayscale upscale (small images only)
        s3_path, s3_modified = _preprocess_strategy_3(image_path)
        if s3_modified:
            s3_results = _run_paddleocr(reader, s3_path, conf_min=0.05)
            candidates.append(("strategy_3", s3_results))

    best_label, best_results = max(candidates, key=lambda x: len(x[1]))

    if not best_results:
        return pd.DataFrame()

    # ── Phase 1: Y-cluster tokens into row groups ─────────────────
    img   = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_h = img.shape[0] if img is not None else 500
    img_w = img.shape[1] if img is not None else 500

    row_threshold = max(12, int(img_h * 0.018))

    rows_raw: list[tuple[float, float, str]] = []
    for bbox, text, conf in best_results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        x_center = (bbox[0][0] + bbox[2][0]) / 2
        rows_raw.append((y_center, x_center, text))

    if not rows_raw:
        return pd.DataFrame()

    rows_raw.sort(key=lambda r: r[0])
    grouped: list[list[tuple[float, float, str]]] = []
    current_group = [rows_raw[0]]
    for item in rows_raw[1:]:
        if abs(item[0] - current_group[-1][0]) <= row_threshold:
            current_group.append(item)
        else:
            grouped.append(current_group)
            current_group = [item]
    grouped.append(current_group)

    if len(grouped) < 2:
        return pd.DataFrame()

    # ── Phase 2: Find header row from raw token groups ────────────
    # Score each row group directly (no column assignment needed yet).
    # Using raw text means number-heavy data rows score 0, header rows
    # score high — we get the header index before touching column layout.
    all_aliases: set[str] = set()
    for alias_list in (list(HOLDING_COLUMN_ALIASES.values()) +
                       list(TRANSACTION_COLUMN_ALIASES.values())):
        all_aliases.update(a.lower() for a in alias_list)

    def _score_group(group: list[tuple]) -> int:
        score = 0
        for _, _, text in group:
            c = text.lower().strip()
            if c in all_aliases:
                score += 2
            elif difflib.get_close_matches(c, all_aliases, n=1, cutoff=0.7):
                score += 1
        return score

    header_group_idx = 0
    best_header_score = _score_group(grouped[0])
    for i in range(1, min(10, len(grouped))):
        s = _score_group(grouped[i])
        if s > best_header_score:
            best_header_score, header_group_idx = s, i

    if best_header_score == 0:
        return pd.DataFrame()

    # ── Phase 3: Derive column centers from header token X positions ──
    # The header row contains one token per column (e.g. "Symbol", "Shares",
    # "Cost Basis"). Using these X positions — rather than all tokens across
    # all rows — avoids false clusters created by dense numeric columns
    # (commas, decimal points, multi-digit numbers).
    #
    # Fallback: if the header row has too few tokens, fall back to global
    # X clustering from all rows. This handles the case where a header has
    # been partially OCR'd.
    col_gap = max(18, int(img_w * 0.02))

    def _cluster_xs(xs: list[float], gap: float) -> list[float]:
        if not xs:
            return []
        cols = [xs[0]]
        for x in xs[1:]:
            if x - cols[-1] > gap:
                cols.append(x)
            else:
                cols[-1] = (cols[-1] + x) / 2
        return cols

    header_xs = sorted(item[1] for item in grouped[header_group_idx])
    col_centers = _cluster_xs(header_xs, gap=col_gap)

    if len(col_centers) < 2:
        # Header too sparse — fall back to global X clustering
        all_x = sorted(set(round(item[1]) for group in grouped for item in group))
        col_centers = _cluster_xs([float(x) for x in all_x], gap=col_gap)
        if len(col_centers) < 2:
            return pd.DataFrame()

    # ── Phase 4: Column boundary midpoints ───────────────────────
    # Assign each token to a column using boundary midpoints, not pure
    # nearest-center. This prevents a token near the edge of col N from
    # being absorbed into col N+1 just because the centres happen to be
    # slightly closer together on that row.
    #
    # Boundary layout:
    #   -inf  |  mid(col0,col1)  |  mid(col1,col2)  | ... |  +inf
    #    col0       col1               col2                   col_last
    col_boundaries = [float("-inf")]
    for i in range(len(col_centers) - 1):
        col_boundaries.append((col_centers[i] + col_centers[i + 1]) / 2)
    col_boundaries.append(float("inf"))

    def col_idx_for_x(x: float) -> int:
        for i in range(len(col_centers)):
            if col_boundaries[i] <= x < col_boundaries[i + 1]:
                return i
        return len(col_centers) - 1

    # ── Phase 5: Build 2-D grid and emit DataFrame ────────────────
    table: list[list[str]] = []
    for group in grouped:
        row: list[str] = [""] * len(col_centers)
        for _, x, text in group:
            ci = col_idx_for_x(x)
            row[ci] = (row[ci] + " " + text).strip() if row[ci] else text
        table.append(row)

    if len(table) < 2:
        return pd.DataFrame()

    headers = table[header_group_idx]
    seen: dict[str, int] = {}
    clean_headers: list[str] = []
    for h in headers:
        h = h.strip() or "col"
        count = seen.get(h, 0)
        clean_headers.append(h if count == 0 else f"{h}_{count}")
        seen[h] = count + 1

    data_rows = table[header_group_idx + 1:]
    if not data_rows:
        return pd.DataFrame()

    return pd.DataFrame(data_rows, columns=clean_headers)
