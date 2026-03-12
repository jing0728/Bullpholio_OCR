"""
core/df_parser.py
-----------------
Converts a normalised DataFrame into DTO records.

Parse path decision:
    1. type_hint from DocumentClassifier (if confident), else detect_input_type()
    2a. "holding"  + shares present  →  BrokerHoldingDTO  (with sanity check)
    2b. "holding"  + shares absent   →  ConstituentHoldingDTO fallback
    2c. "transaction"                →  TransactionDTO
    2d. "constituent_holding" hint   →  ConstituentHoldingDTO directly

Each parser also returns a ParseDetail object carrying confidence metadata
(suspicious_rows, notes) so the pipeline can populate TableParseSummary.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from bullpholio.constants.column_aliases import (
    CONSTITUENT_COLUMN_ALIASES,
    CONSTITUENT_REQUIRED,
    HOLDING_COLUMN_ALIASES,
    HOLDING_REQUIRED,
    TRANSACTION_COLUMN_ALIASES,
    TRANSACTION_REQUIRED,
    TRANSACTION_MOVE_VALID,
)
from bullpholio.core.column_mapper import map_columns
from bullpholio.core.errors import (
    missing_required_columns,
    rows_skipped,
    unrecognised_transaction_move,
    missing_executed_at,
    missing_shares,
)
from bullpholio.core.type_detector import detect_input_type
from bullpholio.models.dtos import (
    BrokerHoldingDTO,
    ConstituentHoldingDTO,
    TransactionDTO,
    _normalise_transaction_move,
)


# ── Sanity check thresholds ───────────────────────────────────────────────────

_SHARES_MAX          = 1_000_000_000   # >1B shares in a personal portfolio = suspect
_AVG_COST_MAX        = 1_000_000       # >$1M per share = suspect (Berkshire A ~$700k)
_AVG_COST_MIN        = 0.0001          # <$0.0001 = likely OCR garbage
_TOTAL_COST_TOLERANCE = 0.15           # 15% tolerance on shares × avg_cost ≈ total_cost


def _sanity_check_broker_holding(dto: BrokerHoldingDTO, idx) -> tuple[bool, str]:
    """
    Cross-check numeric fields on a BrokerHoldingDTO.

    Returns (is_suspicious: bool, note: str).
    Empty note means the row passed all checks.

    Checks (in order of severity):
      1. shares in reasonable range (0 < shares < 1B)
      2. avg_cost: cross-check with total_cost if available (±15%)
      3. avg_cost standalone: flag if it looks like an OCR digit-drop/shift
         (e.g. 1033.19 vs 33.19 — value > $1000 for a fund/ETF is suspicious)
         Heuristic: if avg_cost > 500 AND symbol looks like a fund (>4 chars
         or ends in X), flag as possibly misread.
    """
    issues: list[str] = []

    if dto.shares < 0 or dto.shares > _SHARES_MAX:
        issues.append(f"shares={dto.shares:.4g} out of expected range [0, 1B)")

    if dto.avg_cost_per_share > 0:
        if dto.avg_cost_per_share > _AVG_COST_MAX:
            issues.append(f"avg_cost={dto.avg_cost_per_share:.2f} > ${_AVG_COST_MAX:,} — unrealistic")
        elif dto.avg_cost_per_share < _AVG_COST_MIN:
            issues.append(f"avg_cost={dto.avg_cost_per_share} < {_AVG_COST_MIN} — possible OCR garbage")

        # Cross-check with total_cost when available
        if dto.total_cost > 0 and dto.shares > 0:
            expected = dto.shares * dto.avg_cost_per_share
            ratio    = abs(dto.total_cost - expected) / max(expected, 1)
            if ratio > _TOTAL_COST_TOLERANCE:
                issues.append(
                    f"total_cost {dto.total_cost:.2f} ≠ "
                    f"shares×avg_cost ({dto.shares:.4g}×{dto.avg_cost_per_share:.2f}"
                    f"={expected:.2f}, diff {ratio*100:.0f}%)"
                )
        else:
            # No total_cost to cross-check — apply standalone heuristics.
            # Flag if avg_cost is suspiciously high for a likely fund/ETF symbol.
            # Fund/ETF tickers: >4 chars, or ends in X (GTLOX, VTIAX, etc.)
            # These are rarely priced > $200/share; prices like 1033.19 likely
            # indicate OCR merged a leading digit from an adjacent column.
            is_fund_like = (len(dto.symbol) > 4 or dto.symbol.endswith("X"))
            if is_fund_like and dto.avg_cost_per_share > 200:
                issues.append(
                    f"avg_cost={dto.avg_cost_per_share:.2f} seems high for "
                    f"fund-like symbol '{dto.symbol}' — possible OCR misread "
                    f"(e.g. leading digit from adjacent column)"
                )
            # Also flag any ticker where avg_cost has an implausible leading digit:
            # cost > 1000 but shares < 10 is unusual outside of Berkshire/NVR
            elif dto.avg_cost_per_share > 1000 and dto.shares < 100:
                issues.append(
                    f"avg_cost={dto.avg_cost_per_share:.2f} with only "
                    f"{dto.shares:.4g} shares — cross-check manually"
                )

    if issues:
        note = f"[sanity] Row {idx} [{dto.symbol}]: " + "; ".join(issues)
        return True, note
    return False, ""


# ── ParseDetail ───────────────────────────────────────────────────────────────

@dataclass
class ParseDetail:
    """
    Carries per-table confidence metadata back to the pipeline.
    Populated by each parser and used to fill TableParseSummary fields.
    """
    skipped_rows:    int        = 0
    suspicious_rows: int        = 0
    notes:           list[str]  = field(default_factory=list)

    @property
    def parse_confidence(self) -> str:
        total = self.skipped_rows + self.suspicious_rows
        if total == 0:
            return "high"
        if total <= 2:
            return "medium"
        return "low"

    @property
    def parse_status(self, record_count: int = 0) -> str:
        # Called externally with record_count; return value used by pipeline
        return "high"  # placeholder — pipeline sets this based on record_count


# ── BrokerHoldingDTO parser ───────────────────────────────────────────────────

def _df_to_broker_holdings(
    df: pd.DataFrame,
    col_map: dict,
    logger: logging.Logger,
    warnings: list[str],
) -> tuple[list[BrokerHoldingDTO], ParseDetail]:
    records: list[BrokerHoldingDTO] = []
    errors:  list[str] = []
    detail = ParseDetail()

    for idx, row in df.iterrows():
        raw = {
            canonical: (row[raw_col] if raw_col and raw_col in row.index else None)
            for canonical, raw_col in col_map.items()
        }
        if not raw.get("symbol") or str(raw["symbol"]).strip() in ("", "nan"):
            continue
        try:
            dto = BrokerHoldingDTO(**{k: v for k, v in raw.items() if v is not None})
            raw_shares = raw.get("shares")
            if dto.shares == 0.0 and (raw_shares is None or str(raw_shares).strip() in ("", "nan")):
                errors.append(missing_shares(idx))
                detail.skipped_rows += 1
                continue

            # Sanity check — flag but don't skip
            suspicious, note = _sanity_check_broker_holding(dto, idx)
            if suspicious:
                detail.suspicious_rows += 1
                detail.notes.append(note)
                logger.warning(f"  {note}")
                warnings.append(note)

            records.append(dto)
        except Exception as e:
            errors.append(f"Row {idx}: {e}")
            detail.skipped_rows += 1

    if errors:
        msg = rows_skipped(len(errors), "holding")
        logger.warning(msg)
        for err in errors[:5]:
            logger.warning(f"  {err}")
        warnings.append(msg)
        warnings.extend(errors[:5])

    return records, detail


# ── ConstituentHoldingDTO parser ──────────────────────────────────────────────

def _df_to_constituent_holdings(
    df: pd.DataFrame,
    col_map: dict,
    logger: logging.Logger,
    warnings: list[str],
) -> tuple[list[ConstituentHoldingDTO], ParseDetail]:
    records: list[ConstituentHoldingDTO] = []
    errors:  list[str] = []
    detail = ParseDetail()

    for idx, row in df.iterrows():
        raw = {
            canonical: (row[raw_col] if raw_col and raw_col in row.index else None)
            for canonical, raw_col in col_map.items()
        }
        if not raw.get("symbol") or str(raw["symbol"]).strip() in ("", "nan"):
            continue

        has_weight = raw.get("weight") not in (None, "", "nan")
        has_price  = raw.get("price")  not in (None, "", "nan")
        if not has_weight and not has_price:
            errors.append(f"Row {idx}: neither weight nor price found — row skipped.")
            detail.skipped_rows += 1
            continue

        try:
            dto = ConstituentHoldingDTO(**{k: v for k, v in raw.items() if v is not None})
            records.append(dto)
        except Exception as e:
            errors.append(f"Row {idx}: {e}")
            detail.skipped_rows += 1

    if errors:
        msg = rows_skipped(len(errors), "constituent holding")
        logger.warning(msg)
        for err in errors[:5]:
            logger.warning(f"  {err}")
        warnings.append(msg)
        warnings.extend(errors[:5])

    return records, detail


# ── TransactionDTO parser ─────────────────────────────────────────────────────

def _df_to_transactions(
    df: pd.DataFrame,
    col_map: dict,
    logger: logging.Logger,
    warnings: list[str],
) -> tuple[list[TransactionDTO], ParseDetail]:
    records: list[TransactionDTO] = []
    errors:  list[str] = []
    detail = ParseDetail()

    for idx, row in df.iterrows():
        raw = {
            canonical: (row[raw_col] if raw_col and raw_col in row.index else None)
            for canonical, raw_col in col_map.items()
        }
        if not raw.get("symbol") or str(raw["symbol"]).strip() in ("", "nan"):
            continue

        raw_move_val = raw.get("transaction_move")
        if raw_move_val is not None:
            normalised = _normalise_transaction_move(raw_move_val)
            if normalised not in TRANSACTION_MOVE_VALID:
                errors.append(unrecognised_transaction_move(raw_move_val, idx))
                detail.skipped_rows += 1
                continue

        try:
            dto = TransactionDTO(**{k: v for k, v in raw.items() if v is not None})
            if not dto.executed_at:
                errors.append(missing_executed_at(idx))
                detail.skipped_rows += 1
                continue
            raw_shares = raw.get("shares")
            if dto.shares == 0.0 and (raw_shares is None or str(raw_shares).strip() in ("", "nan")):
                errors.append(missing_shares(idx))
                detail.skipped_rows += 1
                continue
            records.append(dto)
        except Exception as e:
            errors.append(f"Row {idx}: {e}")
            detail.skipped_rows += 1

    if errors:
        msg = rows_skipped(len(errors), "transaction")
        logger.warning(msg)
        for err in errors[:5]:
            logger.warning(f"  {err}")
        warnings.append(msg)
        warnings.extend(errors[:5])

    return records, detail


# ── Main parse entry point ────────────────────────────────────────────────────

def parse_dataframe(
    df: pd.DataFrame,
    logger: logging.Logger,
    warnings: list[str],
    type_hint: Optional[str] = None,
) -> tuple[str, list, ParseDetail]:
    """
    Detect input type, map columns, and convert DataFrame to DTOs.

    type_hint: optional classification from DocumentClassifier.
      If provided and the required columns are present, skip detect_input_type()
      and route directly.  Falls back to detect_input_type() if hint routing fails.

      Accepted hint values: "holding", "transaction", "constituent_holding"

    Returns (input_type, records, ParseDetail).
    """
    columns = list(df.columns)

    # ── Type detection ────────────────────────────────────────────
    # Use classifier hint if trustworthy; fall back to heuristic scorer.
    if type_hint in ("holding", "transaction", "constituent_holding"):
        input_type = type_hint
        logger.debug(f"Using classifier type_hint={input_type} (skipping detect_input_type)")
        warnings.append(f"input_type={input_type} (from classifier)")
    else:
        input_type, confidence = detect_input_type(columns)
        logger.debug(f"Detected input_type={input_type} ({confidence})")
        warnings.append(f"input_type={input_type} ({confidence})")

    # ── Transaction path ──────────────────────────────────────────
    if input_type == "transaction":
        col_map = map_columns(columns, TRANSACTION_COLUMN_ALIASES)
        missing = [f for f in TRANSACTION_REQUIRED if col_map.get(f) is None]
        if missing:
            # Hint might be wrong — retry with heuristic
            if type_hint == "transaction":
                input_type, _ = detect_input_type(columns)
                if input_type != "transaction":
                    return parse_dataframe(df, logger, warnings)  # re-route without hint
            msg = missing_required_columns("transaction", missing, columns)
            logger.warning(msg)
            warnings.append(f"[MissingRequiredColumns] {msg}")
            return "transaction", [], ParseDetail()
        _log_extra(columns, col_map, logger)
        records, detail = _df_to_transactions(df, col_map, logger, warnings)
        return "transaction", records, detail

    # ── Constituent holding path (explicit hint) ──────────────────
    if input_type == "constituent_holding":
        c_col_map = map_columns(columns, CONSTITUENT_COLUMN_ALIASES)
        c_missing = [f for f in CONSTITUENT_REQUIRED if c_col_map.get(f) is None]
        has_weight_or_price = (
            c_col_map.get("weight") is not None or
            c_col_map.get("price")  is not None
        )
        if not c_missing and has_weight_or_price:
            _log_extra(columns, c_col_map, logger)
            records, detail = _df_to_constituent_holdings(df, c_col_map, logger, warnings)
            return "constituent_holding", records, detail
        # Fall through to standard holding path

    # ── Holding path ──────────────────────────────────────────────
    h_col_map = map_columns(columns, HOLDING_COLUMN_ALIASES)
    h_missing = [f for f in HOLDING_REQUIRED if h_col_map.get(f) is None]

    if not h_missing:
        _log_extra(columns, h_col_map, logger)
        records, detail = _df_to_broker_holdings(df, h_col_map, logger, warnings)
        return "holding", records, detail

    # Shares missing — try ConstituentHoldingDTO fallback
    c_col_map = map_columns(columns, CONSTITUENT_COLUMN_ALIASES)
    c_missing = [f for f in CONSTITUENT_REQUIRED if c_col_map.get(f) is None]
    has_weight_or_price = (
        c_col_map.get("weight") is not None or
        c_col_map.get("price")  is not None
    )

    if not c_missing and has_weight_or_price:
        logger.info(
            "Shares column absent — falling back to ConstituentHoldingDTO "
            f"(weight={c_col_map.get('weight')}, price={c_col_map.get('price')})"
        )
        warnings.append(
            "Parsed as constituent holding (index/fund constituent view): "
            "shares column not found, using weight/price instead."
        )
        _log_extra(columns, c_col_map, logger)
        records, detail = _df_to_constituent_holdings(df, c_col_map, logger, warnings)
        return "constituent_holding", records, detail

    # Neither path worked
    msg = missing_required_columns("holding", h_missing, columns)
    logger.warning(msg)
    warnings.append(f"[MissingRequiredColumns] {msg}")
    return "holding", [], ParseDetail()


def _log_extra(columns: list[str], col_map: dict, logger: logging.Logger) -> None:
    mapped = {v for v in col_map.values() if v is not None}
    extra  = [c for c in columns if c not in mapped]
    if extra:
        logger.debug(f"Ignoring {len(extra)} unrecognised column(s): {extra}")
