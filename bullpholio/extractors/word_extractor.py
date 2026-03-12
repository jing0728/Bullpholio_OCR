"""
extractors/word_extractor.py
-----------------------------
Extracts tables from Word (.docx) files using python-docx.
"""

import pandas as pd

from bullpholio.extractors.normaliser import _normalise_dataframe


def _extract_tables_from_word(file_path: str) -> list[pd.DataFrame]:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Missing dependency. Run: pip install python-docx")

    doc    = Document(file_path)
    tables = []
    for tbl in doc.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in tbl.rows]
        if len(rows) < 2:
            continue
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df = _normalise_dataframe(df)
        if df is not None:
            tables.append(df)

    if not tables:
        raise ValueError("No usable tables found in the Word document.")
    return tables
