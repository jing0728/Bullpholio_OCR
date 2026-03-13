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

print("[DEBUG] LOADED bullpholio.extractors.ocr_extractor")

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


# ── PaddleOCR startup env var ─────────────────────────────────────────────────
# Must be set BEFORE `from paddleocr import PaddleOCR` is executed anywhere,
# otherwise PaddleX fires its 10-second connectivity probe on first import.
import os as _os
_os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# ── PaddleOCR Reader cache ────────────────────────────────────────────────────
# Loading PaddleOCR models takes a few seconds on first call.
# We cache the instance at module level so the cost is paid once per process.

_READER_CACHE: object = None  # PaddleOCR instance, lazily initialised


def _get_reader():
    """
    Return the cached PaddleOCR instance, initialising it on first call.

    PaddleOCR 2.x constructor: PaddleOCR(use_angle_cls, lang, use_gpu, show_log)
    PaddleOCR 3.x constructor: PaddleOCR(lang)  — gpu/angle handled automatically,
                                other kwargs raise TypeError "Unknown argument".

    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK must be set BEFORE the import statement
    fires; setting it here is too late if the module was already imported elsewhere.
    Call _ensure_paddle_env() at module import time (bottom of this file) instead.
    """
    global _READER_CACHE
    if _READER_CACHE is None:
        import logging
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("Missing dependency. Run: pip install paddleocr")

        # Mute verbose paddle/ppocr INFO output regardless of version.
        for noisy in ("ppocr", "paddle", "paddleocr", "ppstructure"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        # Try PaddleOCR 3.x signature first (minimal args).
        # Fall back to 2.x signature if that raises TypeError.
        try:
            _READER_CACHE = PaddleOCR(lang="en")
        except TypeError:
            _READER_CACHE = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=False,
            )
    return _READER_CACHE


def warmup_ocr() -> None:
    """
    Pre-warm the PaddleOCR model so the first real image doesn't pay
    the model-initialisation cost (~4-24s on first call).

    Call this once at program/server startup or at the top of a test suite.
    A blank white image is used so CRAFT detects nothing and returns immediately —
    this is about loading the model weights, not producing OCR results.
    """
    import tempfile
    reader = _get_reader()
    dummy  = np.ones((64, 256, 3), dtype=np.uint8) * 255
    tmp    = Path(tempfile.gettempdir()) / "_ocr_warmup.png"
    cv2.imwrite(str(tmp), dummy)
    _run_paddleocr(reader, str(tmp), conf_min=0.0)


def _run_paddleocr(reader, path: str, conf_min: float = 0.05) -> list[tuple]:
    """
    Run PaddleOCR on a single image path and return filtered results as
    a list of (bbox, text, confidence) tuples.

    PaddleOCR 2.x API:  reader.ocr(path, cls=True)
                         raw[0] → list of [bbox, (text, score)]

    PaddleOCR 3.x API:  reader.predict(path)
                         raw    → list of Result objects (one per image)
                         result.boxes / result.rec_scores / result.rec_texts
    """
    try:
        # ── Try 3.x predict() API first ──────────────────────────
        if hasattr(reader, "predict"):
            raw = reader.predict(path)
            print(f"\n[DEBUG] OCR path = {path}")
            print(f"[DEBUG] predict() raw type={type(raw)}, len={len(raw) if raw else 0}")

            if not raw:
                print("[DEBUG] predict() returned empty")
                return []

            # raw is a list of Result objects, one per image.
            # Each Result has .boxes (Nx4x2 or Nx8 array), .rec_texts, .rec_scores
            results = []
            for page_result in raw:
                print(f"[DEBUG] page_result type={type(page_result)}")
                # Attribute names vary slightly by version; check both spellings
                texts  = (getattr(page_result, "rec_texts",  None) or
                          getattr(page_result, "texts",       None) or [])
                scores = (getattr(page_result, "rec_scores",  None) or
                          getattr(page_result, "scores",      None) or [])
                boxes  = (getattr(page_result, "boxes",       None) or
                          getattr(page_result, "dt_boxes",    None) or [])

                print(f"[DEBUG] texts={texts}, scores={scores}")

                for i, (text, conf) in enumerate(zip(texts, scores)):
                    text = str(text).strip()
                    conf = float(conf)
                    print(f"[DEBUG]   text={text!r}  conf={conf:.3f}")
                    if conf >= conf_min and text:
                        bbox = boxes[i].tolist() if i < len(boxes) else [[0,0],[0,0],[0,0],[0,0]]
                        results.append((bbox, text, conf))

            print(f"[DEBUG] kept {len(results)} tokens (conf_min={conf_min})")
            return results

        # ── Fall back to 2.x ocr() API ────────────────────────────
        raw = reader.ocr(path, cls=True)
        print(f"\n[DEBUG] OCR path = {path}")
        print(f"[DEBUG] ocr() raw type={type(raw)}, page len={len(raw[0]) if raw and raw[0] else 0}")

        if not raw or raw[0] is None:
            return []

        results = []
        for line in raw[0]:
            bbox, (text, conf) = line
            text = str(text).strip()
            print(f"[DEBUG]   text={text!r}  conf={conf:.3f}")
            if conf >= conf_min and text:
                results.append((bbox, text, conf))

        print(f"[DEBUG] kept {len(results)} tokens (conf_min={conf_min})")
        return results

    except Exception as e:
        print(f"[DEBUG] OCR exception: {type(e).__name__}: {e}")
        return []


def _ocr_to_dataframe(image_path: str) -> pd.DataFrame:
    """
    Run PaddleOCR on a table image and reconstruct a DataFrame.

    Fast-fail gates (checked before loading the OCR model):
      • Image unreadable by OpenCV → empty DataFrame immediately
      • Both dimensions below OCR minimum → empty DataFrame immediately

    Multi-strategy pipeline (only reached for viable images):
      1. Strategy 1: tiered upscale + CLAHE + unsharp mask
      2. Strategy 2: adaptive threshold (only if S1 yields 1–4 tokens)
      Best result used for table reconstruction via Y/X clustering.
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

    # Baseline: original image, no preprocessing
    orig_results = _run_paddleocr(reader, image_path, conf_min=0.05)
    candidates: list[tuple[str, list]] = [("original", orig_results)]

    # Strategy 1: tiered upscale + CLAHE + unsharp mask
    s1_path, s1_modified = _preprocess_strategy_1(image_path)
    if s1_modified:
        s1_results = _run_paddleocr(reader, s1_path, conf_min=0.05)
        candidates.append(("strategy_1", s1_results))

    # Strategy 2: adaptive threshold
    s2_path, s2_modified = _preprocess_strategy_2(image_path)
    if s2_modified:
        s2_results = _run_paddleocr(reader, s2_path, conf_min=0.05)
        candidates.append(("strategy_2", s2_results))

    # Strategy 3: plain grayscale upscale
    s3_path, s3_modified = _preprocess_strategy_3(image_path)
    if s3_modified:
        s3_results = _run_paddleocr(reader, s3_path, conf_min=0.05)
        candidates.append(("strategy_3", s3_results))

    for label, results in candidates:
        print(f"[DEBUG] {label}: {len(results)} tokens")

    best_label, best_results = max(candidates, key=lambda x: len(x[1]))
    print(f"[DEBUG] best OCR strategy = {best_label}, tokens = {len(best_results)}")

    if not best_results:
        return pd.DataFrame()

    # ── Reconstruct table from best results ──────────────────────
    img   = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_h = img.shape[0] if img is not None else 500
    img_w = img.shape[1] if img is not None else 500

    row_threshold = max(12, int(img_h * 0.018))

    rows_raw: list[tuple[float, float, str]] = []
    for bbox, text, conf in best_results:
        # bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        x_center = (bbox[0][0] + bbox[2][0]) / 2
        rows_raw.append((y_center, x_center, text))

    if not rows_raw:
        return pd.DataFrame()

    # Cluster by Y (rows)
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

    # Column clustering via gap scan
    col_gap = max(18, int(img_w * 0.02))
    all_x   = sorted(set(round(item[1]) for group in grouped for item in group))

    def _cluster_xs(xs: list[float], gap: float = col_gap) -> list[float]:
        if not xs:
            return []
        cols = [xs[0]]
        for x in xs[1:]:
            if x - cols[-1] > gap:
                cols.append(x)
            else:
                cols[-1] = (cols[-1] + x) / 2
        return cols

    col_centers = _cluster_xs([float(x) for x in all_x])

    def nearest_col_idx(x: float) -> int:
        return int(np.argmin([abs(x - c) for c in col_centers]))

    # Build 2-D grid
    table: list[list[str]] = []
    for group in grouped:
        row: list[str] = [""] * len(col_centers)
        for _, x, text in group:
            ci = nearest_col_idx(x)
            row[ci] = (row[ci] + " " + text).strip() if row[ci] else text
        table.append(row)

    if len(table) < 2:
        return pd.DataFrame()

    # Score rows to find the best header (search up to first 10 rows)
    all_aliases: set[str] = set()
    for alias_list in (list(HOLDING_COLUMN_ALIASES.values()) +
                       list(TRANSACTION_COLUMN_ALIASES.values())):
        all_aliases.update(a.lower() for a in alias_list)

    def _score_header(row: list[str]) -> int:
        score = 0
        for cell in row:
            c = cell.lower().strip()
            if c in all_aliases:
                score += 2
            elif difflib.get_close_matches(c, all_aliases, n=1, cutoff=0.7):
                score += 1
        return score

    header_idx = 0
    best = _score_header(table[0])
    for i in range(1, min(10, len(table))):
        s = _score_header(table[i])
        if s > best:
            best, header_idx = s, i

    if best == 0:
        return pd.DataFrame()

    headers = table[header_idx]
    seen: dict[str, int] = {}
    clean_headers: list[str] = []
    for h in headers:
        h = h.strip() or "col"
        count = seen.get(h, 0)
        clean_headers.append(h if count == 0 else f"{h}_{count}")
        seen[h] = count + 1

    return pd.DataFrame(table[header_idx + 1:], columns=clean_headers)
