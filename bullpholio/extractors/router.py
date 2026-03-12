"""
extractors/router.py
---------------------
Routes a file to the correct extractor based on its type.
"""

import logging
import pandas as pd
from pathlib import Path

from bullpholio.config.file_types import get_file_category
from bullpholio.core.errors import (
    file_not_found,
    unsupported_file_type,
    unknown_file_type,
)
from bullpholio.extractors.pdf_extractor import _extract_tables_from_pdf
from bullpholio.extractors.image_extractor import _extract_tables_from_image
from bullpholio.extractors.spreadsheet_extractor import _extract_tables_from_spreadsheet
from bullpholio.extractors.word_extractor import _extract_tables_from_word


def extract_tables(
    file_path: str,
    logger: logging.Logger,
    allow_ocr: bool = False,
    warnings: list[str] | None = None,
) -> list[pd.DataFrame]:
    """
    Route the file to the appropriate extractor and return DataFrames.
    Raises ValueError for unsupported/unknown types, FileNotFoundError if missing.
    """
    if warnings is None:
        warnings = []

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_not_found(file_path))

    suffix   = path.suffix.lower()
    category = get_file_category(suffix)
    logger.debug(f"[router] extension={suffix} category={category}")

    if category == "unsupported":
        raise ValueError(unsupported_file_type(suffix))
    if category == "unknown":
        raise ValueError(unknown_file_type(suffix))
    if category == "pdf":
        return _extract_tables_from_pdf(file_path, logger, allow_ocr=allow_ocr)
    if category == "image":
        return _extract_tables_from_image(file_path, allow_ocr=allow_ocr, warnings=warnings)
    if category == "spreadsheet":
        return _extract_tables_from_spreadsheet(file_path)
    if category == "word":
        return _extract_tables_from_word(file_path)

    raise ValueError(f"Router internal error: unhandled category '{category}'")
