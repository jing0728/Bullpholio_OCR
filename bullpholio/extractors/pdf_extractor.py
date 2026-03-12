"""
extractors/pdf_extractor.py
----------------------------
PDF table extraction: 4 passes (bordered, relaxed, word-layout, OCR).
"""

import difflib
import logging
import tempfile
import pandas as pd
from pathlib import Path
from typing import Optional

from bullpholio.constants.column_aliases import (
    HOLDING_COLUMN_ALIASES,
    TRANSACTION_COLUMN_ALIASES,
)
from bullpholio.extractors.normaliser import _normalise_dataframe, _normalise_pdf_table


def _pdf_words_to_dataframe(page) -> Optional[pd.DataFrame]:
    """
    Reconstruct a table from a pdfplumber page using word positions.
    Used for borderless / whitespace-aligned PDFs where extract_tables()
    finds nothing because there are no visible grid lines.
    """
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    if len(words) < 4:
        return None

    # ── Cluster into rows by Y ────────────────────────────────────
    words_sorted = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
    tops = sorted(set(round(w["top"]) for w in words))
    if len(tops) >= 2:
        gaps = [tops[i+1] - tops[i] for i in range(len(tops)-1) if tops[i+1] - tops[i] > 0]
        median_gap = sorted(gaps)[len(gaps)//2] if gaps else 10
    else:
        median_gap = 10
    row_tol = max(4, median_gap * 0.6)

    rows_raw: list[list[dict]] = []
    cur_row: list[dict] = [words_sorted[0]]
    for w in words_sorted[1:]:
        if abs(w["top"] - cur_row[-1]["top"]) <= row_tol:
            cur_row.append(w)
        else:
            rows_raw.append(cur_row)
            cur_row = [w]
    rows_raw.append(cur_row)

    if len(rows_raw) < 2:
        return None

    # ── Score each of the first rows to find the real header ─────
    all_aliases: set[str] = set()
    for alias_list in (
        list(HOLDING_COLUMN_ALIASES.values()) +
        list(TRANSACTION_COLUMN_ALIASES.values())
    ):
        all_aliases.update(a.lower() for a in alias_list)

    def _score_as_header(row_words: list[dict]) -> int:
        score = 0
        for w in row_words:
            cell_l = w["text"].lower().strip()
            if not cell_l:
                continue
            if cell_l in all_aliases:
                score += 2
                continue
            if difflib.get_close_matches(cell_l, all_aliases, n=1, cutoff=0.7):
                score += 1
        return score

    header_row_idx = 0
    best_score = _score_as_header(rows_raw[0])
    for i in range(1, min(3, len(rows_raw))):
        s = _score_as_header(rows_raw[i])
        if s > best_score:
            best_score = s
            header_row_idx = i

    header_words  = sorted(rows_raw[header_row_idx], key=lambda w: w["x0"])
    data_rows_raw = rows_raw[header_row_idx + 1:]
    if not data_rows_raw:
        return None

    # ── Determine column count & centres ─────────────────────────
    hw_sorted     = sorted(header_words, key=lambda w: w["x0"])
    header_groups: list[list[dict]] = []
    if hw_sorted:
        cur_grp = [hw_sorted[0]]
        for w in hw_sorted[1:]:
            if w["x0"] - cur_grp[-1]["x1"] < 20:
                cur_grp.append(w)
            else:
                header_groups.append(cur_grp)
                cur_grp = [w]
        header_groups.append(cur_grp)

    n_cols = len(header_groups)
    if n_cols < 2:
        return None

    raw_hdr_names = [
        " ".join(w["text"] for w in grp).strip()
        for grp in header_groups
    ]

    all_x_mid_list = sorted(set(round((w["x0"] + w["x1"]) / 2) for w in words))
    WITHIN = 25
    clusters: list[float] = []
    if all_x_mid_list:
        clusters = [float(all_x_mid_list[0])]
        for x in all_x_mid_list[1:]:
            if x - clusters[-1] <= WITHIN:
                clusters[-1] = (clusters[-1] + x) / 2
            else:
                clusters.append(float(x))

    if len(clusters) == n_cols:
        col_centers = clusters
    else:
        mids = all_x_mid_list
        if len(mids) < n_cols:
            return None
        gaps_indexed = sorted(
            ((mids[i+1] - mids[i], i) for i in range(len(mids)-1)),
            reverse=True
        )
        split_at = sorted(idx for _, idx in gaps_indexed[:n_cols - 1])
        col_centers = []
        start = 0
        for si in split_at:
            seg = mids[start:si + 1]
            col_centers.append(float(sum(seg) / len(seg)))
            start = si + 1
        tail = mids[start:]
        if tail:
            col_centers.append(float(sum(tail) / len(tail)))

    if len(col_centers) < 2:
        return None

    def _col_idx(x_mid: float) -> int:
        return int(min(range(len(col_centers)),
                       key=lambda i: abs(x_mid - col_centers[i])))

    # ── Build 2-D grid ────────────────────────────────────────────
    grid: list[list[str]] = []
    for row_words in data_rows_raw:
        row_words_sorted = sorted(row_words, key=lambda w: w["x0"])
        cells = [""] * len(col_centers)
        for w in row_words_sorted:
            x_mid = (w["x0"] + w["x1"]) / 2
            ci    = _col_idx(x_mid)
            cells[ci] = (cells[ci] + " " + w["text"]).strip()
        grid.append(cells)

    # ── Clean headers ─────────────────────────────────────────────
    seen: dict[str, int] = {}
    clean_headers: list[str] = []
    for h in raw_hdr_names:
        h = h.strip() or f"col_{len(clean_headers)}"
        c = seen.get(h, 0)
        clean_headers.append(h if c == 0 else f"{h}_{c}")
        seen[h] = c + 1

    if not grid:
        return None
    df = pd.DataFrame(grid, columns=clean_headers)
    return _normalise_dataframe(df)


def _extract_tables_from_pdf(
    file_path: str,
    logger: logging.Logger,
    allow_ocr: bool = False,
) -> list[pd.DataFrame]:
    """
    Extract all tables from a PDF using four passes in order.
    Stops as soon as any pass produces results.

      Pass 1 — pdfplumber default (bordered tables, fast).
      Pass 2 — pdfplumber relaxed tolerances.
      Pass 3 — word-layout reconstruction (borderless text PDFs).
      Pass 4 — EasyOCR (scanned/image-only PDFs, only if allow_ocr=True).
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Missing dependency. Run: pip install pdfplumber")

    # ── Pass 1: default extraction ────────────────────────────────
    tables: list[pd.DataFrame] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            for raw_table in page.extract_tables():
                df = _normalise_pdf_table(raw_table)
                if df is not None:
                    tables.append(df)

    if tables:
        logger.debug(f"PDF pass 1 found {len(tables)} table(s)")
        return tables

    # ── Pass 2: relaxed tolerances ────────────────────────────────
    RELAXED = {
        "vertical_strategy":    "lines_strict",
        "horizontal_strategy":  "lines_strict",
        "snap_tolerance":       10,
        "join_tolerance":       10,
        "edge_min_length":      10,
        "min_words_vertical":   1,
        "min_words_horizontal": 1,
    }
    logger.debug("PDF pass 1 → 0 tables; retrying with relaxed settings")
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            for raw_table in page.extract_tables(table_settings=RELAXED):
                df = _normalise_pdf_table(raw_table)
                if df is not None:
                    tables.append(df)

    if tables:
        logger.debug(f"PDF pass 2 (relaxed) found {len(tables)} table(s)")
        return tables

    # ── Pass 3: word-layout reconstruction ───────────────────────
    logger.debug("PDF pass 2 → 0 tables; trying word-layout reconstruction")
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            df = _pdf_words_to_dataframe(page)
            if df is not None:
                tables.append(df)

    if tables:
        logger.debug(f"PDF pass 3 (word-layout) found {len(tables)} table(s)")
        return tables

    # ── Pass 4: OCR — only when explicitly enabled ────────────────
    if not allow_ocr:
        logger.warning(
            "No tables found in PDF (passes 1-3 all failed). "
            "If this is a scanned/image PDF, re-run with allow_ocr=True."
        )
        return []

    logger.info("PDF passes 1-3 found nothing — falling back to EasyOCR (scanned PDF)")
    import fitz
    from bullpholio.extractors.image_extractor import _has_table_structure
    from bullpholio.extractors.ocr_extractor import _ocr_to_dataframe

    doc = fitz.open(file_path)
    try:
        for page in doc:
            pix      = page.get_pixmap(dpi=200)
            img_path = str(Path(tempfile.gettempdir()) / f"_bullpholio_page_{page.number}.png")
            pix.save(img_path)
            if _has_table_structure(img_path):
                df = _ocr_to_dataframe(img_path)
                df = _normalise_dataframe(df)
                if df is not None:
                    tables.append(df)
    finally:
        doc.close()

    return tables
