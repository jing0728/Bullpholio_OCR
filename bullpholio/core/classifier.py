"""
core/classifier.py
------------------
Lightweight document classifier that runs AFTER table extraction but BEFORE
parsing. Determines document type without building any DTOs.

Classification types:
    "holding"       — all tables look like portfolio holdings
    "transaction"   — all tables look like trade history
    "mixed"         — file contains both holdings and transactions
    "non_financial" — tables found but no financial columns detected
    "unsupported"   — no usable tables extracted at all

Confidence levels:
    "high"   — required fields (symbol + shares/date) clearly present
    "medium" — some financial columns matched but required fields unclear
    "low"    — weak signal, best-guess classification

This classifier is intentionally cheap: it only runs map_columns() at a
loose threshold and counts hits — no DTO construction, no row iteration.
"""

import pandas as pd
from dataclasses import dataclass, field

from bullpholio.constants.column_aliases import (
    HOLDING_COLUMN_ALIASES,
    HOLDING_REQUIRED,
    TRANSACTION_COLUMN_ALIASES,
    TRANSACTION_REQUIRED,
)
from bullpholio.core.column_mapper import map_columns
from bullpholio.core.errors import not_financial, not_financial_no_columns


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class TableClassification:
    """Classification result for a single DataFrame."""
    table_index: int           # 1-based
    doc_type: str              # "holding" | "transaction" | "non_financial"
    confidence: str            # "high" | "medium" | "low"
    holding_score: int
    transaction_score: int
    matched_holding_cols: list[str]
    matched_transaction_cols: list[str]
    missing_required: list[str]


@dataclass
class ClassificationResult:
    """Aggregated classification for the whole document."""
    doc_type: str              # "holding"|"transaction"|"mixed"|"non_financial"|"unsupported"
    confidence: str            # "high" | "medium" | "low"
    reason: str                # human-readable, suitable for frontend display
    table_results: list[TableClassification] = field(default_factory=list)

    # Convenience flags for pipeline branching
    @property
    def is_financial(self) -> bool:
        return self.doc_type in ("holding", "transaction", "mixed")

    @property
    def should_parse(self) -> bool:
        """True when the pipeline should proceed to DTO parsing."""
        return self.is_financial


# ── Scorer ────────────────────────────────────────────────────────────────────

def _score_table(df: pd.DataFrame, table_index: int) -> TableClassification:
    """
    Score a single DataFrame against holding and transaction alias maps.

    Scoring (mirrors detect_input_type tiers but stops before DTO construction):

    For transactions (strict threshold 0.75):
      +3  commission, net_amount, settled_at   (exclusive tx fields)
      +2  transaction_move, executed_at        (strong tx signals)
      +1  price_per_share                      (weak tx signal)

    For holdings (strict threshold 0.75):
      +2  avg_cost_per_share, total_cost       (cost-basis fields)
      +2  last_price, market_value             (broker UI fields)
      +1  first_trading_date, day_gain, total_gain

    Required-field check uses loose threshold 0.6.
    """
    columns = list(df.columns)

    tx_strict = map_columns(columns, TRANSACTION_COLUMN_ALIASES, threshold=0.75)
    tx_loose  = map_columns(columns, TRANSACTION_COLUMN_ALIASES, threshold=0.6)
    h_strict  = map_columns(columns, HOLDING_COLUMN_ALIASES,     threshold=0.75)
    h_loose   = map_columns(columns, HOLDING_COLUMN_ALIASES,     threshold=0.6)

    # Transaction score
    tx_score = (
        (3 if tx_strict.get("commission")       else 0) +
        (3 if tx_strict.get("net_amount")        else 0) +
        (3 if tx_strict.get("settled_at")        else 0) +
        (2 if tx_strict.get("transaction_move")  else 0) +
        (2 if tx_strict.get("executed_at")       else 0) +
        (1 if tx_loose.get("price_per_share")    else 0)
    )

    # Holding score
    h_score = (
        (2 if h_strict.get("avg_cost_per_share") else 0) +
        (2 if h_strict.get("total_cost")         else 0) +
        (2 if h_strict.get("last_price")         else 0) +
        (2 if h_strict.get("market_value")       else 0) +
        (1 if h_strict.get("first_trading_date") else 0) +
        (1 if h_strict.get("day_gain")           else 0) +
        (1 if h_strict.get("total_gain")         else 0)
    )

    # Determine raw type
    if tx_score > h_score and tx_score >= 2:
        raw_type   = "transaction"
        col_map    = tx_loose
        required   = TRANSACTION_REQUIRED
    elif h_score > 0 or tx_score > 0:
        raw_type   = "holding"
        col_map    = h_loose
        required   = HOLDING_REQUIRED
    else:
        # No financial signal at all
        return TableClassification(
            table_index=table_index,
            doc_type="non_financial",
            confidence="high",
            holding_score=0,
            transaction_score=0,
            matched_holding_cols=[],
            matched_transaction_cols=[],
            missing_required=list(HOLDING_REQUIRED | TRANSACTION_REQUIRED),
        )

    # Check required fields
    missing = [f for f in required if col_map.get(f) is None]

    # Matched column names (for diagnostics)
    matched_h  = [v for v in h_loose.values()  if v is not None]
    matched_tx = [v for v in tx_loose.values() if v is not None]

    # Confidence
    max_score  = max(h_score, tx_score)
    if not missing and max_score >= 4:
        confidence = "high"
    elif not missing and max_score >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    # Downgrade to non_financial when required fields are completely absent
    # and there's very weak signal overall
    if missing and max_score < 2:
        raw_type   = "non_financial"
        confidence = "high"

    return TableClassification(
        table_index=table_index,
        doc_type=raw_type,
        confidence=confidence,
        holding_score=h_score,
        transaction_score=tx_score,
        matched_holding_cols=matched_h,
        matched_transaction_cols=matched_tx,
        missing_required=missing,
    )


# ── Classifier ────────────────────────────────────────────────────────────────

class DocumentClassifier:
    """
    Classifies a list of DataFrames (one per extracted table) without
    building any DTOs or iterating over data rows.

    Usage:
        classifier = DocumentClassifier()
        result = classifier.classify(dataframes)
        if not result.should_parse:
            return early_failure(result.reason)
    """

    def classify(self, dataframes: list[pd.DataFrame]) -> ClassificationResult:
        if not dataframes:
            return ClassificationResult(
                doc_type="unsupported",
                confidence="high",
                reason="No tables could be extracted from the file.",
            )

        table_results = [
            _score_table(df, i + 1)
            for i, df in enumerate(dataframes)
        ]

        types = {t.doc_type for t in table_results}

        # ── All non-financial ─────────────────────────────────────
        if types == {"non_financial"}:
            cols = table_results[0].matched_holding_cols + table_results[0].matched_transaction_cols
            return ClassificationResult(
                doc_type="non_financial",
                confidence="high",
                reason=_reason_non_financial(table_results),
                table_results=table_results,
            )

        # ── Financial tables only ─────────────────────────────────
        financial = [t for t in table_results if t.doc_type != "non_financial"]
        fin_types = {t.doc_type for t in financial}

        if fin_types == {"holding"}:
            doc_type = "holding"
        elif fin_types == {"transaction"}:
            doc_type = "transaction"
        else:
            doc_type = "mixed"

        # Overall confidence = weakest among financial tables
        conf_rank  = {"high": 2, "medium": 1, "low": 0}
        confidence = min(
            (t.confidence for t in financial),
            key=lambda c: conf_rank[c],
        )

        reason = _reason_financial(doc_type, financial)
        return ClassificationResult(
            doc_type=doc_type,
            confidence=confidence,
            reason=reason,
            table_results=table_results,
        )


# ── Human-readable reason builders ───────────────────────────────────────────

def _reason_financial(doc_type: str, tables: list[TableClassification]) -> str:
    total   = len(tables)
    records = "table" if total == 1 else "tables"

    if doc_type == "holding":
        sample = tables[0].matched_holding_cols[:4]
        return (
            f"Detected {total} holdings {records}. "
            f"Matched columns: {', '.join(sample)}."
        )
    if doc_type == "transaction":
        sample = tables[0].matched_transaction_cols[:4]
        return (
            f"Detected {total} transaction {records}. "
            f"Matched columns: {', '.join(sample)}."
        )
    # mixed
    h_count  = sum(1 for t in tables if t.doc_type == "holding")
    tx_count = sum(1 for t in tables if t.doc_type == "transaction")
    return (
        f"Detected mixed document: {h_count} holdings table(s) "
        f"and {tx_count} transaction table(s)."
    )


def _reason_non_financial(tables: list[TableClassification]) -> str:
    all_cols: list[str] = []
    for t in tables:
        all_cols.extend(t.matched_holding_cols)
        all_cols.extend(t.matched_transaction_cols)

    missing_all: set[str] = set()
    for t in tables:
        missing_all.update(t.missing_required)

    if not all_cols and not missing_all:
        return not_financial_no_columns()

    available = [c for c in all_cols if not c.startswith("col_")]
    return not_financial(list(missing_all), available)
