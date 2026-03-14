"""
core/df_parser.py
-----------------
Converts a normalised DataFrame into DTO records.

Parse path decision:
    1. type_hint from DocumentClassifier (if confident), else detect_input_type()
    2a. "rebalance_plan"      →  RebalancePlanDTO  (target_weight/drift/action)
    2b. "holding"  + shares   →  BrokerHoldingDTO  (with sanity check)
    2c. "holding"  no shares  →  ConstituentHoldingDTO fallback
    2d. "transaction"         →  TransactionDTO
    2e. "constituent_holding" →  ConstituentHoldingDTO directly

Each parser also returns a ParseDetail object carrying confidence metadata
(suspicious_rows, notes) so the pipeline can populate TableParseSummary.

Warning hygiene
───────────────
Only user-actionable messages go into the `warnings` list (which flows into
PipelineResult.warnings and is shown in the frontend):
  • MissingRequiredColumns — tells the user what's missing
  • rows_skipped           — tells the user how many rows were dropped and why
  • sanity notes           — flags numeric values that look like OCR misreads

Internal routing decisions (which type_hint was used, detect_input_type score)
are written to logger.debug only and never appear in `warnings`.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from bullpholio.constants.column_aliases import (
    CONSTITUENT_COLUMN_ALIASES,
    CONSTITUENT_REQUIRED,
    HOLDING_COLUMN_ALIASES,
    HOLDING_REQUIRED,
    REBALANCE_COLUMN_ALIASES,
    REBALANCE_REQUIRED,
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
    RebalancePlanDTO,
    TransactionDTO,
    _normalise_transaction_move,
)


# ── Sanity check thresholds ───────────────────────────────────────────────────

_SHARES_MAX           = 1_000_000_000   # >1B shares in a personal portfolio = suspect
_AVG_COST_MAX         = 1_000_000       # >$1M per share = suspect (Berkshire A ~$700k)
_AVG_COST_MIN         = 0.0001          # <$0.0001 = likely OCR garbage
_TOTAL_COST_TOLERANCE = 0.15            # 15% tolerance on shares × avg_cost ≈ total_cost


def _sanity_check_broker_holding(dto: BrokerHoldingDTO, idx) -> tuple[bool, str]:
    """
    Cross-check numeric fields on a BrokerHoldingDTO.

    Returns (is_suspicious: bool, note: str).
    Empty note means the row passed all checks.

    Checks (in order of severity):
      1. shares in reasonable range (0 ≤ shares < 1B)
      2. avg_cost: cross-check with total_cost if available (±15%)
      3. avg_cost standalone: flag if it looks like an OCR digit-drop/shift
    """
    issues: list[str] = []

    if dto.shares < 0 or dto.shares > _SHARES_MAX:
        issues.append(f"shares={dto.shares:.4g} out of expected range [0, 1B)")

    if dto.avg_cost_per_share > 0:
        if dto.avg_cost_per_share > _AVG_COST_MAX:
            issues.append(f"avg_cost={dto.avg_cost_per_share:.2f} > ${_AVG_COST_MAX:,} — unrealistic")
        elif dto.avg_cost_per_share < _AVG_COST_MIN:
            issues.append(f"avg_cost={dto.avg_cost_per_share} < {_AVG_COST_MIN} — possible OCR garbage")

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
            is_fund_like = (len(dto.symbol) > 4 or dto.symbol.endswith("X"))
            if is_fund_like and dto.avg_cost_per_share > 200:
                issues.append(
                    f"avg_cost={dto.avg_cost_per_share:.2f} seems high for "
                    f"fund-like symbol '{dto.symbol}' — possible OCR misread"
                )
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


# ── BrokerHoldingDTO parser ───────────────────────────────────────────────────

# Fields that exist in HOLDING_COLUMN_ALIASES for column-detection purposes
# but are NOT part of BrokerHoldingDTO (display-only / never stored in DB).
# Only truly display-only fields stay here; day_gain and overall_gain have been
# promoted to proper BrokerHoldingDTO fields (Go may wish to display them).
_HOLDING_DISPLAY_ONLY: frozenset[str] = frozenset({
    "last_price",    # current market price — not a cost/position field
    "market_value",  # shares × last_price — always recomputable, not stored
})

# Aggregation / summary row labels that should be silently skipped.
# These are not real positions — they are UI totals added by broker exports.
_AGGREGATION_SYMBOLS: frozenset[str] = frozenset({
    "total", "grand total", "subtotal", "sub-total",
    "portfolio total", "account total", "合计", "总计",
})

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
            if canonical not in _HOLDING_DISPLAY_ONLY   # strip display-only fields
        }
        if not raw.get("symbol") or str(raw["symbol"]).strip() in ("", "nan"):
            continue
        # Skip broker-UI aggregation rows ("Total", "Grand Total", etc.)
        if str(raw["symbol"]).strip().lower() in _AGGREGATION_SYMBOLS:
            logger.debug(f"Row {idx}: skipping aggregation row '{raw['symbol']}'")
            continue
        try:
            dto = BrokerHoldingDTO(**{k: v for k, v in raw.items() if v is not None})
            raw_shares = raw.get("shares")
            if dto.shares == 0.0 and (raw_shares is None or str(raw_shares).strip() in ("", "nan")):
                errors.append(missing_shares(idx))
                detail.skipped_rows += 1
                continue

            # If avg_cost_per_share is absent but total_cost + shares are both
            # present, derive it.  This is common in broker screenshots that show
            # "Cost Basis" (= total_cost) but no "Avg Cost" column — e.g. the
            # Fidelity / personal-finance portfolio view in stock.png.
            if dto.avg_cost_per_share == 0.0 and dto.total_cost > 0 and dto.shares > 0:
                dto.avg_cost_per_share = round(dto.total_cost / dto.shares, 6)

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
            logger.debug(f"  {err}")
        warnings.append(msg)

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
            logger.debug(f"  {err}")
        warnings.append(msg)

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
            logger.debug(f"  {err}")
        warnings.append(msg)

    return records, detail


# ── OCR column recovery helpers ───────────────────────────────────────────────
#
# After OCR reconstruction, two common failure modes:
#
# 1. Symbol drift: the "Symbol" header X-position ≠ ticker data X-position
#    (header text is centred/wide; ticker values are short and left-aligned).
#    Result: tickers land in an adjacent blank column; symbol cells are empty.
#
# 2. Shares absorption: a shares value near a column boundary gets absorbed
#    into the neighbouring (often numeric) column.
#
# These helpers are called by parse_dataframe() when the OCR path is detected
# (i.e. when required columns are missing but pattern-matching finds candidates).

_TICKER_RE = re.compile(r'^[A-Z]{1,6}$')   # strict ticker: 1–6 uppercase letters


def _find_stray_symbol_column(df: pd.DataFrame, symbol_col: Optional[str]) -> Optional[str]:
    """
    If the mapped symbol column has mostly empty rows, scan other string
    columns for one whose values mostly look like stock tickers.

    Returns the column name that looks like the real symbol column, or None.
    """
    n = max(len(df), 1)

    def _ticker_ratio(col: str) -> float:
        vals = df[col].astype(str).str.strip()
        hits = vals.apply(lambda v: bool(_TICKER_RE.match(v))).sum()
        return hits / n

    # Current symbol column already fine?
    if symbol_col and _ticker_ratio(symbol_col) >= 0.5:
        return None   # nothing to fix

    # Find the best candidate among OTHER columns
    best_col, best_ratio = None, 0.3   # minimum threshold
    for col in df.columns:
        if col == symbol_col:
            continue
        r = _ticker_ratio(col)
        if r > best_ratio:
            best_ratio, best_col = r, col

    return best_col


def _repair_ocr_symbol(df: pd.DataFrame, symbol_col: Optional[str]) -> pd.DataFrame:
    """
    If OCR put ticker values in the wrong column, remap them.
    Returns a (possibly modified) DataFrame; always safe to call.
    """
    stray = _find_stray_symbol_column(df, symbol_col)
    if stray is None:
        return df     # nothing to fix

    df = df.copy()
    if symbol_col and symbol_col in df.columns:
        # Overwrite empty symbol cells with the stray column's value
        mask = df[symbol_col].astype(str).str.strip().isin(["", "nan", "None"])
        df.loc[mask, symbol_col] = df.loc[mask, stray]
    else:
        # No symbol column at all — rename the stray column
        df = df.rename(columns={stray: "Symbol"})

    logger_repair = logging.getLogger("bullpholio.ocr_repair")
    logger_repair.debug(
        f"OCR symbol repair: moved tickers from '{stray}' → '{symbol_col or 'Symbol'}'"
    )
    return df


def _repair_ocr_shares(
    df: pd.DataFrame,
    symbol_col: Optional[str],
    shares_col: Optional[str],
) -> pd.DataFrame:
    """
    If the shares column has mostly empty cells, look for a numeric column
    that contains realistic share counts (0.001–1,000,000).

    Only activates when the shares column is genuinely missing data, not when
    it just has some 0-share rows (which are valid).
    """
    if not shares_col or shares_col not in df.columns:
        return df

    n = max(len(df), 1)

    def _numeric_ratio(col: str) -> float:
        vals = df[col].astype(str).str.replace(",", "").str.strip()
        hits = 0
        for v in vals:
            try:
                f = float(v)
                if 0.001 < abs(f) < 1_000_000:
                    hits += 1
            except (ValueError, TypeError):
                pass
        return hits / n

    # Check if current shares col is already fine
    if _numeric_ratio(shares_col) >= 0.3:
        return df   # nothing to fix

    # Find best numeric candidate (skip symbol col)
    best_col, best_ratio = None, 0.4
    for col in df.columns:
        if col in (shares_col, symbol_col):
            continue
        r = _numeric_ratio(col)
        if r > best_ratio:
            best_ratio, best_col = r, col

    if best_col is None:
        return df

    df = df.copy()
    mask = df[shares_col].astype(str).str.strip().isin(["", "nan", "None", "0"])
    df.loc[mask, shares_col] = df.loc[mask, best_col]
    return df


# ── Rebalance plan parser ─────────────────────────────────────────────────────

def _df_to_rebalance_plan(
    df: pd.DataFrame,
    col_map: dict,
    logger: logging.Logger,
    warnings: list[str],
) -> tuple[list[RebalancePlanDTO], ParseDetail]:
    """
    Parse a rebalance plan DataFrame into RebalancePlanDTO records.

    Required: symbol + at least one of (current_weight, target_weight).
    Optional: drift, action, trade_amount, trade_shares.

    If drift is absent but both current_weight and target_weight are present,
    it is auto-computed as target - current.
    """
    records: list[RebalancePlanDTO] = []
    errors:  list[str] = []
    detail = ParseDetail()

    for idx, row in df.iterrows():
        raw = {
            canonical: (row[raw_col] if raw_col and raw_col in row.index else None)
            for canonical, raw_col in col_map.items()
        }
        sym = str(raw.get("symbol") or "").strip()
        if not sym or sym.lower() in ("", "nan"):
            continue
        # Skip aggregation/total rows
        if sym.lower() in _AGGREGATION_SYMBOLS:
            logger.debug(f"Row {idx}: skipping aggregation row '{sym}'")
            continue

        # Require at least one weight field
        has_cur = raw.get("current_weight") not in (None, "", "nan")
        has_tgt = raw.get("target_weight")  not in (None, "", "nan")
        if not has_cur and not has_tgt:
            errors.append(f"Row {idx}: neither current_weight nor target_weight found")
            detail.skipped_rows += 1
            continue

        try:
            dto = RebalancePlanDTO(**{k: v for k, v in raw.items() if v is not None})
            # Auto-derive drift when absent
            if dto.drift == 0.0 and dto.current_weight != 0.0 and dto.target_weight != 0.0:
                dto.drift = round(dto.target_weight - dto.current_weight, 6)
            records.append(dto)
        except Exception as e:
            errors.append(f"Row {idx}: {e}")
            detail.skipped_rows += 1

    if errors:
        msg = rows_skipped(len(errors), "rebalance plan")
        logger.warning(msg)
        for err in errors[:5]:
            logger.debug(f"  {err}")
        warnings.append(msg)

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

      Accepted hint values:
        "holding", "transaction", "constituent_holding", "rebalance_plan"

    Returns (input_type, records, ParseDetail).
    """
    columns = list(df.columns)

    # ── Type detection ────────────────────────────────────────────
    # Routing decisions are internal — written to logger.debug only,
    # never appended to `warnings` (which is user-facing).
    if type_hint in ("holding", "transaction", "constituent_holding", "rebalance_plan"):
        input_type = type_hint
        logger.debug(f"Using classifier type_hint={input_type}")
    else:
        input_type, confidence_note = detect_input_type(columns)
        logger.debug(f"Detected input_type={input_type} ({confidence_note})")

    # ── Rebalance plan path (probe first — distinct column vocabulary) ────────
    # Probe unconditionally: rebalance plans share "symbol" with holdings but
    # have "target_weight" / "drift" which are never in pure holding tables.
    # If the explicit hint is rebalance_plan, or if the columns strongly suggest
    # a rebalance plan even without a hint, route here.
    _r_col_map = map_columns(columns, REBALANCE_COLUMN_ALIASES)
    _has_rebalance_signal = (
        _r_col_map.get("target_weight") is not None or
        _r_col_map.get("drift")         is not None
    )
    if (input_type == "rebalance_plan" or _has_rebalance_signal) and _r_col_map.get("symbol"):
        has_weight = (
            _r_col_map.get("current_weight") is not None or
            _r_col_map.get("target_weight")  is not None
        )
        if has_weight:
            _log_extra(columns, _r_col_map, logger)
            records, detail = _df_to_rebalance_plan(df, _r_col_map, logger, warnings)
            return "rebalance_plan", records, detail

    # ── Transaction path ──────────────────────────────────────────
    if input_type == "transaction":
        col_map = map_columns(columns, TRANSACTION_COLUMN_ALIASES)
        missing = [f for f in TRANSACTION_REQUIRED if col_map.get(f) is None]
        if missing:
            if type_hint == "transaction":
                input_type, _ = detect_input_type(columns)
                if input_type != "transaction":
                    return parse_dataframe(df, logger, warnings)
            msg = missing_required_columns("transaction", missing, columns)
            logger.warning(msg)
            warnings.append(f"[MissingRequiredColumns] {msg}")
            return "transaction", [], ParseDetail()
        _log_extra(columns, col_map, logger)
        records, detail = _df_to_transactions(df, col_map, logger, warnings)
        return "transaction", records, detail

    # ── Constituent holding path (explicit hint) ──────────────────
    if input_type == "constituent_holding":
        # Guard: if the DataFrame also has a "shares" column (unambiguous
        # broker-holding signal), override the hint and go to broker_holding.
        # This fixes screenshots like stock.png where "Last Price" + "Change"
        # cause the classifier to emit constituent_holding, but "Shares" and
        # "Cost basis" clearly indicate a broker account view.
        h_col_map_pre = map_columns(columns, HOLDING_COLUMN_ALIASES)
        has_shares = h_col_map_pre.get("shares") is not None
        has_symbol = h_col_map_pre.get("symbol") is not None
        if has_symbol and has_shares:
            logger.debug(
                "constituent_holding hint overridden: "
                "both symbol and shares columns found — routing to broker_holding"
            )
            input_type = "holding"
        else:
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

    # ── Holding path ──────────────────────────────────────────────
    # ── OCR column repair (symbol + shares) ───────────────────────
    # Run before column mapping so the mapper sees repaired column names/values.
    # These repairs are cheap and safe: if nothing needs fixing, df is returned
    # unchanged. They activate when:
    #   • Symbol column exists but has mostly empty cells (drift from OCR X-cluster)
    #   • Shares column exists but has mostly empty cells (absorbed by adjacent col)
    _pre_map = map_columns(columns, HOLDING_COLUMN_ALIASES)
    df_repaired = _repair_ocr_symbol(df, _pre_map.get("symbol"))
    if df_repaired is not df:
        df      = df_repaired
        columns = list(df.columns)
        _pre_map = map_columns(columns, HOLDING_COLUMN_ALIASES)

    h_col_map = map_columns(columns, HOLDING_COLUMN_ALIASES)
    h_missing = [f for f in HOLDING_REQUIRED if h_col_map.get(f) is None]

    if "shares" in h_missing:
        df_repaired = _repair_ocr_shares(df, h_col_map.get("symbol"), h_col_map.get("shares"))
        if df_repaired is not df:
            df        = df_repaired
            columns   = list(df.columns)
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

    msg = missing_required_columns("holding", h_missing, columns)
    logger.warning(msg)
    warnings.append(f"[MissingRequiredColumns] {msg}")
    return "holding", [], ParseDetail()


def _log_extra(columns: list[str], col_map: dict, logger: logging.Logger) -> None:
    mapped = {v for v in col_map.values() if v is not None}
    extra  = [c for c in columns if c not in mapped]
    if extra:
        logger.debug(f"Ignoring {len(extra)} unrecognised column(s): {extra}")
