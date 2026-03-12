"""
pipeline.py
-----------
Main pipeline orchestrator: coordinates extraction, classification,
parsing, and result assembly.

Stage flow:
    1. extract_tables()        — raw DataFrames from file
    2. DocumentClassifier      — lightweight classify before any DTO work
    3. parse_dataframe()       — DataFrame → DTOs, guided by classifier hint
    4. PipelineResult          — unified result envelope with confidence metadata

Usage:
    from bullpholio.pipeline import run_pipeline

    result = run_pipeline("path/to/file.pdf")
    print(result.status, result.record_count)
    print(result.classification.reason)         # frontend-ready message
    for s in result.table_summaries:
        print(s.parse_confidence, s.suspicious_rows, s.confidence_notes)
"""

import logging
import time
from pathlib import Path
from typing import Optional

from bullpholio.config.file_types import get_file_category
from bullpholio.core.classifier import DocumentClassifier
from bullpholio.core.df_parser import ParseDetail, parse_dataframe
from bullpholio.extractors.router import extract_tables
from bullpholio.models.results import PipelineResult, StageError, TableParseSummary


def run_pipeline(
    file_path: str,
    logger: Optional[logging.Logger] = None,
    allow_ocr: bool = False,
) -> PipelineResult:
    if logger is None:
        logger = logging.getLogger("pipeline")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

    pipeline_start  = time.monotonic()
    stage_latency: dict[str, float] = {}
    all_records:   list             = []
    all_warnings:  list[str]        = []
    all_summaries: list[TableParseSummary] = []
    detected_type: Optional[str]    = None

    path     = Path(file_path)
    suffix   = path.suffix.lower()
    category = get_file_category(suffix)

    def elapsed_ms(start: float) -> float:
        return round((time.monotonic() - start) * 1000, 2)

    # ── Stage 1: Table Extraction ─────────────────────────────────────────────
    logger.info(f"[1] Extracting tables from: {file_path}")
    t = time.monotonic()
    try:
        dataframes = extract_tables(file_path, logger, allow_ocr=allow_ocr, warnings=all_warnings)
    except Exception as e:
        stage_latency["extraction"] = elapsed_ms(t)
        return PipelineResult(
            status="failed",
            file_path=file_path,
            file_extension=suffix,
            file_category=category,
            total_latency_ms=elapsed_ms(pipeline_start),
            stage_latency_ms=stage_latency,
            errors=[StageError(stage="extraction", error_type=type(e).__name__, message=str(e))],
            warnings=all_warnings,
        )
    stage_latency["extraction"] = elapsed_ms(t)
    logger.info(f"Found {len(dataframes)} table(s) ({stage_latency['extraction']} ms)")

    # ── Stage 2: Document Classification ─────────────────────────────────────
    # Lightweight pass — columns only, no DTO construction.
    # Result guides which parser to call and provides the frontend reason string.
    logger.info("[2] Classifying document...")
    t = time.monotonic()
    classifier     = DocumentClassifier()
    classification = classifier.classify(dataframes)
    stage_latency["classification"] = elapsed_ms(t)
    logger.info(f"Classification: {classification.doc_type} (confidence={classification.confidence})")
    logger.debug(f"Reason: {classification.reason}")

    # Early exit for non-financial / unsupported
    if not classification.should_parse:
        return PipelineResult(
            status="failed",
            file_path=file_path,
            file_extension=suffix,
            file_category=category,
            classification=classification,
            total_latency_ms=elapsed_ms(pipeline_start),
            stage_latency_ms=stage_latency,
            errors=[StageError(
                stage="classification",
                error_type=classification.doc_type.title().replace("_", ""),
                message=classification.reason,
            )],
            warnings=all_warnings,
        )

    # ── Stage 3: Parse Each DataFrame ────────────────────────────────────────
    # Pass the classifier's per-table type hint so parse_dataframe can skip
    # detect_input_type() when the classifier is confident.
    logger.info("[3] Parsing tables...")
    t = time.monotonic()
    parse_errors: list[StageError] = []

    for i, df in enumerate(dataframes):
        table_idx   = i + 1
        table_class = classification.table_results[i] if i < len(classification.table_results) else None

        if table_class and table_class.doc_type == "non_financial":
            logger.debug(f"  Table {table_idx}: skipped (non_financial per classifier)")
            continue

        # Pass classifier hint — high-confidence hits skip detect_input_type()
        type_hint = None
        if table_class and table_class.confidence == "high":
            type_hint = table_class.doc_type  # "holding" | "transaction" | ...

        logger.info(f"  Table {table_idx}: {len(df)} rows × {len(df.columns)} cols"
                    + (f" (hint={type_hint})" if type_hint else ""))
        try:
            input_type, records, detail = parse_dataframe(df, logger, all_warnings, type_hint=type_hint)

            if detected_type is None:
                detected_type = input_type
            elif detected_type != input_type:
                detected_type = "mixed"

            all_records.extend(records)
            logger.info(f"  → {len(records)} {input_type} record(s) parsed "
                        f"(confidence={detail.parse_confidence}, "
                        f"suspicious={detail.suspicious_rows})")

            # Derive parse_status from record count + detail
            # Rules:
            #   0 records              → failed
            #   any skipped rows       → partial_success (data was lost)
            #   suspicious rows > 30%  → partial_success (most values untrustworthy)
            #   suspicious rows ≤ 30%  → success with notes (isolated outliers are OK)
            total_rows = max(len(records) + detail.skipped_rows, 1)
            suspicious_ratio = detail.suspicious_rows / total_rows
            if len(records) == 0:
                parse_status = "failed"
            elif detail.skipped_rows > 0 or suspicious_ratio > 0.30:
                parse_status = "partial_success"
            else:
                parse_status = "success"

            skipped = len(records) == 0
            all_summaries.append(TableParseSummary(
                table_index=table_idx,
                input_type=input_type,
                row_count=len(df),
                record_count=len(records),
                skipped=skipped,
                parse_status=parse_status,
                parse_confidence=detail.parse_confidence,
                suspicious_rows=detail.suspicious_rows,
                confidence_notes=detail.notes[:5],  # cap at 5 to keep result lean
            ))
            if skipped:
                all_warnings.append(
                    f"Table {table_idx}: parsed as {input_type} but produced 0 records. "
                    "Check warnings for MissingRequiredColumns details."
                )
        except Exception as e:
            parse_errors.append(StageError(
                stage=f"parse_table_{table_idx}",
                error_type=type(e).__name__,
                message=str(e),
            ))
            all_summaries.append(TableParseSummary(
                table_index=table_idx,
                input_type="unknown",
                row_count=len(df),
                record_count=0,
                skipped=True,
                parse_status="failed",
                parse_confidence="low",
            ))
            logger.error(f"  → Parse failed: {e}")

    stage_latency["parse"] = elapsed_ms(t)
    total_ms = elapsed_ms(pipeline_start)
    logger.info(f"[4] Done — {len(all_records)} record(s) total ({total_ms} ms)")

    # Top-level status stays "success" | "partial" | "failed":
    #   failed  → no records at all
    #   partial → records extracted but parse_errors OR any table is partial_success
    #   success → all tables clean
    has_partial_table = any(
        s.parse_status == "partial_success" for s in all_summaries
    )
    status = (
        "failed"  if not all_records else
        "partial" if parse_errors or has_partial_table else
        "success"
    )

    return PipelineResult(
        status=status,
        file_path=file_path,
        file_extension=suffix,
        file_category=category,
        classification=classification,
        input_type=detected_type,
        total_latency_ms=total_ms,
        stage_latency_ms=stage_latency,
        record_count=len(all_records),
        data=all_records,
        errors=parse_errors,
        warnings=all_warnings,
        table_summaries=all_summaries,
    )
