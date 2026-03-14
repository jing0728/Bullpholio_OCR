"""
models/dtos.py
--------------
Data Transfer Objects produced by the parser stage.

BrokerHoldingDTO
    Standard broker account view (Fidelity, Schwab, IBKR, etc.).
    Fields mirror the Go Holding struct.  Two additional read-only fields
    that brokers show in their UI but that are not stored in the DB:
        day_gain     — unrealised gain/loss for today (dollars)
        overall_gain — total unrealised gain/loss since purchase (dollars)
    Both default to 0.0 and are zero when the source document does not
    include these columns.

ConstituentHoldingDTO
    Index / ETF constituent view (e.g. SPDR sector breakdown).
    Requires symbol + (weight OR price).

TransactionDTO
    Individual trade record.

dto_type discriminator
    Each DTO carries a `dto_type` literal so the Go backend can switch on
    it after JSON unmarshalling without reflection tricks:
        "broker_holding"      → BrokerHolding
        "constituent_holding" → ConstituentHolding
        "transaction"         → Transaction
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_float(v) -> float:
    """
    Coerce a value to float, stripping common broker formatting.
    Handles: None, '', 'nan', '$1,234.56', '(567.89)' (negative), '19.71%'
    Returns 0.0 for unparseable values.
    """
    if v is None:
        return 0.0
    s = str(v).strip()
    if not s or s.lower() in ("nan", "n/a", "--", "-", ""):
        return 0.0
    # Strip currency symbols, thousands separators, percent signs
    s = s.replace("$", "").replace(",", "").replace("%", "").strip()
    # Parentheses → negative  e.g. (567.89) → -567.89
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _to_str(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "n/a") else s


def _normalise_transaction_move(raw) -> str:
    """
    Map raw broker action strings to canonical transaction_move values.
    Returns the original lowercased string if no mapping is found
    (caller checks against TRANSACTION_MOVE_VALID).
    """
    from bullpholio.constants.column_aliases import TRANSACTION_MOVE_ALIASES
    s = _to_str(raw).lower().strip()
    for canonical, aliases in TRANSACTION_MOVE_ALIASES.items():
        if s in [a.lower() for a in aliases]:
            return canonical
    return s


# ── BrokerHoldingDTO ──────────────────────────────────────────────────────────

class BrokerHoldingDTO(BaseModel):
    """
    Mirrors the Go Holding struct (minus DB/audit fields).

    Fields
    ──────
    symbol             TICKER (required)
    name               Company name — populated when a Name/Description column exists
    shares             Position size
    avg_cost_per_share Per-share average cost.  Auto-derived from total_cost/shares
                       when the source document only shows "Cost Basis" (total).
    total_cost         Total amount paid for the position ("Cost Basis" in Fidelity/Schwab)
    side               "long" | "short" | "flat"  — defaults to "long"
    first_trading_date ISO date string or None
    last_trading_date  ISO date string or None
    day_gain           Today's unrealised P&L in dollars (0.0 if not in source)
    overall_gain       Total unrealised P&L in dollars (0.0 if not in source)
    """
    dto_type: Literal["broker_holding"] = "broker_holding"

    symbol:             str
    name:               str   = ""
    shares:             float = 0.0
    avg_cost_per_share: float = 0.0
    total_cost:         float = 0.0
    side:               str   = "long"
    first_trading_date: Optional[str] = None
    last_trading_date:  Optional[str] = None
    day_gain:           float = 0.0   # today's unrealised gain / loss
    overall_gain:       float = 0.0   # total unrealised gain / loss
    overall_gain_pct:   float = 0.0   # total return % (0.0 when not in source)

    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, v):
        return _to_str(v).upper()

    @field_validator("name", mode="before")
    @classmethod
    def _clean_name(cls, v):
        return _to_str(v)

    @field_validator("shares", "avg_cost_per_share", "total_cost",
                     "day_gain", "overall_gain", "overall_gain_pct", mode="before")
    @classmethod
    def _clean_float(cls, v):
        return _to_float(v)

    @field_validator("side", mode="before")
    @classmethod
    def _clean_side(cls, v):
        s = _to_str(v).lower()
        return s if s in ("long", "short", "flat") else "long"

    @field_validator("first_trading_date", "last_trading_date", mode="before")
    @classmethod
    def _clean_date(cls, v):
        s = _to_str(v)
        return s if s else None

    model_config = {"populate_by_name": True}


# ── ConstituentHoldingDTO ─────────────────────────────────────────────────────

class ConstituentHoldingDTO(BaseModel):
    """
    Index / ETF constituent holding (e.g. SPDR sector weights).
    Requires symbol + at least one of weight or price.
    """
    dto_type: Literal["constituent_holding"] = "constituent_holding"

    symbol:       str
    name:         str   = ""
    weight:       float = 0.0   # % allocation (e.g. 7.02 for 7.02%)
    price:        float = 0.0
    holding_type: str   = ""
    change:       float = 0.0   # day change in dollars or percent

    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, v):
        return _to_str(v).upper()

    @field_validator("name", "holding_type", mode="before")
    @classmethod
    def _clean_str(cls, v):
        return _to_str(v)

    @field_validator("weight", "price", "change", mode="before")
    @classmethod
    def _clean_float(cls, v):
        return _to_float(v)

    model_config = {"populate_by_name": True}


# ── TransactionDTO ────────────────────────────────────────────────────────────

class TransactionDTO(BaseModel):
    """
    Single trade / transaction record.
    Mirrors the Go Transaction struct (minus DB/audit fields).
    """
    dto_type: Literal["transaction"] = "transaction"

    symbol:           str
    transaction_move: str   = "buy"   # "buy"|"sell"|"short_sell"|"cover"|"dividend"
    shares:           float = 0.0
    price_per_share:  float = 0.0
    total_amount:     float = 0.0
    commission:       float = 0.0
    fees:             float = 0.0
    net_amount:       float = 0.0
    executed_at:      Optional[str] = None
    settled_at:       Optional[str] = None
    notes:            str   = ""

    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, v):
        return _to_str(v).upper()

    @field_validator("transaction_move", mode="before")
    @classmethod
    def _clean_move(cls, v):
        return _normalise_transaction_move(v)

    @field_validator("shares", "price_per_share", "total_amount",
                     "commission", "fees", "net_amount", mode="before")
    @classmethod
    def _clean_float(cls, v):
        return _to_float(v)

    @field_validator("notes", mode="before")
    @classmethod
    def _clean_notes(cls, v):
        return _to_str(v)

    @field_validator("executed_at", "settled_at", mode="before")
    @classmethod
    def _clean_date(cls, v):
        s = _to_str(v)
        return s if s else None

    model_config = {"populate_by_name": True}


# ── RebalancePlanDTO ──────────────────────────────────────────────────────────

class RebalancePlanDTO(BaseModel):
    """
    One row in a portfolio rebalance plan.

    Distinguishing columns vs constituent_holding:
      - target_weight: the desired allocation (constituent_holding rarely has this)
      - drift: difference between current and target (unique to rebalance plans)
      - action: trade instruction ("buy" / "sell" / "hold")
      - trade_amount / trade_shares: dollar or share quantity to trade

    Any combination of current_weight, target_weight, drift is valid.
    At least one of current_weight or target_weight must be non-zero.
    """
    dto_type: Literal["rebalance_plan"] = "rebalance_plan"

    symbol:         str
    name:           str   = ""
    current_weight: float = 0.0   # current % allocation
    target_weight:  float = 0.0   # desired % allocation
    drift:          float = 0.0   # target - current (can be negative)
    action:         str   = ""    # "buy" / "sell" / "hold" / ""
    trade_amount:   float = 0.0   # dollar value to trade (positive = buy)
    trade_shares:   float = 0.0   # share count to trade

    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, v):
        return _to_str(v).upper()

    @field_validator("name", "action", mode="before")
    @classmethod
    def _clean_str(cls, v):
        return _to_str(v).lower() if v else ""

    @field_validator("current_weight", "target_weight", "drift",
                     "trade_amount", "trade_shares", mode="before")
    @classmethod
    def _clean_float(cls, v):
        return _to_float(v)

    model_config = {"populate_by_name": True}
