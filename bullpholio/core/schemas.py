"""
api/schemas.py
--------------
Request / response schemas for the Bullpholio parse API.

Design goals
────────────
• Every field is a JSON-native type (str, float, int, bool, list, dict).
  No Python-specific objects cross the wire.
• snake_case field names throughout — Go uses struct tags to remap if needed.
• The `data` array uses a `dto_type` discriminator so the Go backend can
  switch on the value and unmarshal into the correct struct:
    "broker_holding"      → BrokerHolding
    "constituent_holding" → ConstituentHolding
    "transaction"         → Transaction
• `ClassificationSummary` flattens the internal dataclass into a plain dict
  so the Go side does not need to care about Python internals.
• HTTP error responses always follow `ErrorResponse` so the Go client has
  one shape to unmarshal on non-2xx status codes.
"""

from __future__ import annotations

from typing import Any, Optional, Union
from pydantic import BaseModel, Field


# ── Record schemas (mirror the DTO models) ────────────────────────────────────

class BrokerHoldingRecord(BaseModel):
    dto_type:           str   = "broker_holding"
    symbol:             str
    name:               str   = ""
    shares:             float
    avg_cost_per_share: float = 0.0
    total_cost:         float = 0.0
    side:               str   = "long"
    first_trading_date: Optional[str] = None
    last_trading_date:  Optional[str] = None
    # Broker UI display fields — 0.0 when the source document omits them.
    # Not stored in the Holding DB table but useful for the frontend to
    # show current performance without a separate market-data call.
    day_gain:           float = 0.0   # today's unrealised P&L ($)
    overall_gain:       float = 0.0   # total unrealised P&L ($)
    overall_gain_pct:   float = 0.0   # total return %


class RebalancePlanRecord(BaseModel):
    """
    One row from a portfolio rebalance plan document.
    `dto_type` = "rebalance_plan" — Go backend switches on this.
    """
    dto_type:       str   = "rebalance_plan"
    symbol:         str
    name:           str   = ""
    current_weight: float = 0.0   # current % allocation
    target_weight:  float = 0.0   # target % allocation
    drift:          float = 0.0   # target - current (negative = over-weight)
    action:         str   = ""    # "buy" / "sell" / "hold" / ""
    trade_amount:   float = 0.0   # dollar value to trade
    trade_shares:   float = 0.0   # share count to trade


class ConstituentHoldingRecord(BaseModel):
    dto_type:     str   = "constituent_holding"
    symbol:       str
    name:         str   = ""
    weight:       float = 0.0
    price:        float = 0.0
    holding_type: str   = ""
    change:       float = 0.0


class TransactionRecord(BaseModel):
    dto_type:          str   = "transaction"
    symbol:            str
    transaction_move:  str
    shares:            float
    price_per_share:   float = 0.0
    total_amount:      float = 0.0
    commission:        float = 0.0
    fees:              float = 0.0
    net_amount:        float = 0.0
    executed_at:       Optional[str] = None
    settled_at:        Optional[str] = None
    notes:             str   = ""


# ── Supporting schemas ────────────────────────────────────────────────────────

class StageErrorSchema(BaseModel):
    stage:      str
    error_type: str
    message:    str


class TableSummarySchema(BaseModel):
    table_index:      int
    input_type:       str
    row_count:        int
    record_count:     int
    skipped:          bool  = False
    parse_status:     str   = "success"      # "success" | "partial_success" | "failed"
    parse_confidence: str   = "high"         # "high" | "medium" | "low"
    suspicious_rows:  int   = 0
    confidence_notes: list[str] = Field(default_factory=list)


class ClassificationSummary(BaseModel):
    """
    Flattened view of the internal ClassificationResult dataclass.
    Gives the Go backend enough context to render a user-facing message
    without needing to understand the Python classifier internals.
    """
    doc_type:   str          # "holding" | "transaction" | "mixed" | "non_financial"
    confidence: str          # "high" | "medium" | "low"
    reason:     str          # human-readable, suitable for frontend display


# ── Main response schema ──────────────────────────────────────────────────────

class ParseResponse(BaseModel):
    """
    Unified response envelope for POST /v1/parse.

    Always returned with HTTP 200 — the `status` field carries the
    semantic outcome.  The Go backend should branch on `status`:

        "success"               → records are clean; display immediately
        "partial"               → records extracted; some rows were skipped
        "low_confidence_partial"→ records extracted; but majority flagged by
                                  sanity checks — surface manual-review prompt
        "failed"                → no records; check `errors` for the reason

    The `data` array is heterogeneous: inspect each element's `dto_type`
    to determine the concrete struct:

        "broker_holding"      → BrokerHolding
        "constituent_holding" → ConstituentHolding
        "transaction"         → Transaction
    """
    status:           str                   # "success"|"partial"|"low_confidence_partial"|"failed"
    input_type:       Optional[str]         # "holding"|"transaction"|"mixed"|None
    record_count:     int
    total_latency_ms: float
    stage_latency_ms: dict[str, float]
    classification:   Optional[ClassificationSummary]
    data:             list[Union[
                          BrokerHoldingRecord,
                          ConstituentHoldingRecord,
                          TransactionRecord,
                          RebalancePlanRecord,
                      ]] = Field(default_factory=list)
    table_summaries:  list[TableSummarySchema] = Field(default_factory=list)
    errors:           list[StageErrorSchema]   = Field(default_factory=list)
    warnings:         list[str]                = Field(default_factory=list)

    model_config = {"populate_by_name": True}


# ── Health check response ─────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str


# ── HTTP error envelope (non-2xx responses) ───────────────────────────────────

class ErrorResponse(BaseModel):
    """
    Returned on HTTP 4xx / 5xx.
    The Go client can always unmarshal non-2xx responses into this shape.
    """
    error:   str           # short machine-readable code, e.g. "unsupported_file_type"
    message: str           # human-readable detail
    detail:  Optional[Any] = None   # extra context when available
