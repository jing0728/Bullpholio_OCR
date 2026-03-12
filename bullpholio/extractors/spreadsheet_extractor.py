"""
extractors/spreadsheet_extractor.py
-------------------------------------
Extracts tables from CSV and Excel (.xlsx) files using pandas.
"""

import pandas as pd
from pathlib import Path

from bullpholio.extractors.normaliser import _normalise_dataframe


def _extract_tables_from_spreadsheet(file_path: str) -> list[pd.DataFrame]:
    suffix = Path(file_path).suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path, dtype=str).fillna("")
        df = _normalise_dataframe(df)
        return [df] if df is not None else []

    xl  = pd.ExcelFile(file_path)
    dfs = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet, dtype=str).fillna("")
        df = _normalise_dataframe(df)
        if df is not None:
            dfs.append(df)
    return dfs
