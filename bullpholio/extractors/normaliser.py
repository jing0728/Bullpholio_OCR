"""
extractors/normaliser.py
------------------------
Normalises raw DataFrames and pdfplumber table outputs before column mapping.
"""

import re
from typing import Optional

import pandas as pd


def _normalise_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Standardise a raw extracted DataFrame before column mapping.

    Steps:
      1. Normalise column names: lower, strip, collapse whitespace/newlines.
      2. Rename empty/Unnamed columns to col_N.
      3. Drop all-empty rows and all-empty columns.
      4. Normalise cell values: strip, unify null-ish strings to "".
      5. Reject DataFrames that are too sparse or have too few columns.

    Returns None if the DataFrame is deemed invalid.
    """
    if df is None or df.empty:
        return None

    # 1+2: Normalise column names
    new_cols: list[str] = []
    seen: dict[str, int] = {}
    for i, col in enumerate(df.columns):
        c = re.sub(r"\s+", " ", str(col).lower().strip().replace("\n", " "))
        if not c or c.startswith("unnamed:") or c == "nan":
            c = f"col_{i}"
        count = seen.get(c, 0)
        new_cols.append(c if count == 0 else f"{c}_{count}")
        seen[c] = count + 1
    df = df.copy()
    df.columns = new_cols  # type: ignore[assignment]

    # 3: Drop fully empty rows
    df = df.dropna(how="all").reset_index(drop=True)

    if df.empty:
        return None

    # 4: Normalise cell values
    NULL_STRINGS = {"", "nan", "none", "n/a", "na", "-", "—", "null"}

    def _clean_cell(x: object) -> str:
        s = str(x).strip()
        return "" if s.lower() in NULL_STRINGS else s

    df = df.apply(lambda col: col.map(_clean_cell))

    # Drop columns that are entirely empty strings
    df = df.loc[:, df.astype(bool).any(axis=0)]

    # 5: Reject obviously bad tables
    if df.shape[1] < 2:
        return None
    total_cells = df.shape[0] * df.shape[1]
    non_empty   = (df != "").sum().sum()
    if total_cells > 0 and (non_empty / total_cells) < 0.2:
        return None

    # Reject if >60% of column names are still "col_N" placeholders
    placeholder_ratio = sum(
        1 for c in df.columns if re.match(r"^col_\d+$", c)
    ) / len(df.columns)
    if placeholder_ratio > 0.6:
        return None

    return df


def _normalise_pdf_table(raw_table: list[list]) -> Optional[pd.DataFrame]:
    """
    Clean up pdfplumber extract_tables() output.
    Handles: split headers, empty col names, duplicate col names, sparse tables.

    Detects two-row headers common in broker PDFs (e.g. "Avg" / "Cost"
    on separate rows). If row-0 has too many placeholders and row-1 looks
    like a continuation, merges them col-by-col as the header.
    """
    if not raw_table or len(raw_table) < 2:
        return None

    def _is_placeholder(s: str) -> bool:
        return not s or s.lower().startswith("col_") or s.lower() == "nan"

    def _build_headers(row: list) -> list[str]:
        return [re.sub(r"\s+", " ", str(c).strip()) if c else f"col_{i}"
                for i, c in enumerate(row)]

    row0_headers = _build_headers(raw_table[0])

    HEADER_KEYWORDS = {
        "date", "qty", "price", "amount", "total", "cost", "symbol",
        "type", "action", "shares", "value", "name", "ticker",
        "日期", "数量", "价格", "金额", "成本", "代码", "类型",
    }
    data_start = 1

    if len(raw_table) >= 3:
        placeholder_ratio = sum(
            1 for h in row0_headers if _is_placeholder(h)
        ) / max(len(row0_headers), 1)
        row1 = [str(c).strip() if c else "" for c in raw_table[1]]
        row1_nonempty   = sum(1 for v in row1 if v)
        row1_nonempty_r = row1_nonempty / max(len(row1), 1)
        numeric_in_row1 = sum(1 for v in row1 if re.match(r"^-?[\d,\.]+$", v))
        numeric_r       = numeric_in_row1 / max(len(row1), 1)
        row1_has_header_token = any(
            v.lower() in HEADER_KEYWORDS for v in row1 if v
        )
        is_two_row_header = (
            placeholder_ratio > 0.3
            and row1_nonempty_r > 0.5
            and (numeric_r < 0.3 or row1_has_header_token)
        )
        if is_two_row_header:
            merged = []
            for h, r in zip(row0_headers, row1):
                if _is_placeholder(h) and r:
                    merged.append(r)
                elif h and r and r.lower() != h.lower():
                    merged.append(f"{h} {r}".strip())
                else:
                    merged.append(h or r)
            row0_headers = merged
            data_start = 2

    # De-duplicate headers
    seen: dict[str, int] = {}
    clean_headers: list[str] = []
    for h in row0_headers:
        h = h or f"col_{len(clean_headers)}"
        count = seen.get(h, 0)
        clean_headers.append(h if count == 0 else f"{h}_{count}")
        seen[h] = count + 1

    rows = [
        [re.sub(r"\s+", " ", str(c).strip()) if c else "" for c in row]
        for row in raw_table[data_start:]
    ]

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=clean_headers)
    return _normalise_dataframe(df)
