"""
core/errors.py
--------------
User-readable error and warning message builders.

All messages are designed to be shown directly in a frontend UI —
no stack traces, no internal field names, no Python jargon.

Each builder returns a plain string. The pipeline attaches these
strings to StageError.message or ClassificationResult.reason.
"""

from pathlib import Path


# ── Extraction errors ────────────────────────────────────────────────────────

def file_not_found(file_path: str) -> str:
    return (
        f"File not found: '{Path(file_path).name}'. "
        "Please check the file path and try again."
    )


def unsupported_file_type(suffix: str) -> str:
    return (
        f"'{suffix}' files are not supported. "
        "Accepted formats: PDF, Excel (.xlsx), CSV, Word (.docx), "
        "and images (.jpg, .png, .bmp, .tiff, .webp, .gif)."
    )


def unknown_file_type(suffix: str) -> str:
    return (
        f"Unrecognised file type '{suffix}'. "
        "Please upload a PDF, Excel, CSV, Word, or image file."
    )


def no_tables_found(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    hints = {
        ".pdf": (
            "The PDF appears to have no extractable tables. "
            "If it is a scanned document, try re-uploading with OCR enabled."
        ),
        ".docx": (
            "No tables were found in the Word document. "
            "Make sure the data is in a Word table (Insert → Table), not plain text."
        ),
    }
    return hints.get(suffix, (
        "No tables could be extracted from the file. "
        "Please check that the file contains tabular data."
    ))


def missing_dependency(package: str, install_cmd: str) -> str:
    return (
        f"Required package '{package}' is not installed. "
        f"Run: {install_cmd}"
    )


# ── Image / OCR errors ───────────────────────────────────────────────────────

def image_ocr_disabled() -> str:
    return (
        "This image file requires OCR to extract data, but OCR is currently disabled. "
        "Enable OCR mode and try again."
    )


def image_no_table(file_path: str) -> str:
    name = Path(file_path).name
    return (
        f"'{name}' does not appear to contain a table or spreadsheet. "
        "Only images of financial tables (e.g. broker screenshots) are supported. "
        "Photos, logos, and charts are not accepted."
    )


def image_resolution_too_low(width: int, height: int, file_path: str) -> str:
    name = Path(file_path).name
    return (
        f"'{name}' has very low resolution ({width}×{height}px). "
        "OCR accuracy may be poor. For best results, use an image at least 800px wide. "
        "The pipeline will attempt extraction anyway."
    )


def image_ocr_no_results(file_path: str) -> str:
    name = Path(file_path).name
    return (
        f"A table was detected in '{name}', but OCR could not read any text. "
        "Try uploading a higher-resolution version of the image."
    )


# ── Classification errors ────────────────────────────────────────────────────

def not_financial(missing_cols: list[str], available_cols: list[str]) -> str:
    missing_str = ", ".join(sorted(missing_cols)) if missing_cols else "symbol, shares"
    # Show at most 6 available columns so the message stays readable
    sample = [c for c in available_cols if not c.startswith("col_")][:6]
    sample_str = ", ".join(sample) if sample else "none recognised"
    return (
        f"This does not appear to be a portfolio holdings or transaction file. "
        f"Missing required columns: {missing_str}. "
        f"Columns found: {sample_str}."
    )


def not_financial_no_columns() -> str:
    return (
        "Tables were found in the file, but no financial columns were detected. "
        "Expected columns such as: Symbol, Shares, Avg Cost, Date, Commission."
    )


# ── Parse warnings (shown in result.warnings) ────────────────────────────────

def missing_required_columns(input_type: str, missing: list[str], available: list[str]) -> str:
    missing_str   = ", ".join(missing)
    available_str = ", ".join(available[:8])
    return (
        f"Cannot parse as {input_type}: required column(s) not found: {missing_str}. "
        f"Columns available: {available_str}."
    )


def rows_skipped(count: int, input_type: str) -> str:
    noun = "row" if count == 1 else "rows"
    return f"{count} {noun} skipped during {input_type} parsing (missing or invalid data)."


def unrecognised_transaction_move(raw_value: str, row_idx: int) -> str:
    return (
        f"Row {row_idx}: unrecognised transaction type '{raw_value}'. "
        "Expected: buy, sell, short sell, cover, or dividend."
    )


def missing_executed_at(row_idx: int) -> str:
    return f"Row {row_idx}: trade date is missing or could not be parsed — row skipped."


def missing_shares(row_idx: int) -> str:
    return f"Row {row_idx}: shares value is missing or empty — row skipped."
