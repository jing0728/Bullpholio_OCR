"""
models/dtos.py
--------------
Data Transfer Objects — matches the Go structs consumed by the backend.

Holding schema is layered:

    BrokerHoldingDTO       — broker account view (requires shares)
                             e.g. Fidelity/Schwab portfolio CSV/screenshots
    ConstituentHoldingDTO  — index/fund constituent view (requires weight OR price)
                             e.g. SPY holdings from stockanalysis.com, ETF fact sheets

    TransactionDTO         — trade history (unchanged)

The dto_type field on each record tells the backend and frontend which
variant was parsed, so they can render the right columns.
"""

import re
import numpy as np
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, field_validator

from bullpholio.constants.column_aliases import TRANSACTION_MOVE_ALIASES


# ── Shared field-cleaning helpers ─────────────────────────────────────────────

def _clean_decimal(v: object) -> float:
    """
    Strip currency symbols, commas, whitespace; return float.
    Handles parentheses negatives:  (1,234.56) → -1234.56
    Handles trailing minus sign:    1,234.56-  → -1234.56
    Returns 0.0 on failure.
    """
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        is_paren_negative = s.startswith("(") and s.endswith(")")
        if is_paren_negative:
            s = s[1:-1]
        cleaned = re.sub(r"[^\d.\-]", "", s.replace(",", ""))
        if cleaned.endswith("-"):
            cleaned = "-" + cleaned[:-1]
        if is_paren_negative and not cleaned.startswith("-"):
            cleaned = "-" + cleaned
        match = re.search(r"-?\d+(\.\d+)?", cleaned)
        return float(match.group()) if match else 0.0
    return 0.0


def _clean_date(v: object) -> Optional[str]:
    """Normalise a date value to ISO string YYYY-MM-DD. Returns None if unparseable."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, datetime):
        return v.date().isoformat()
    if hasattr(v, "isoformat"):
        return str(v)[:10]
    s = str(v).strip()
    if not s or s.lower() in ("", "nan", "none", "n/a", "na", "-", "—"):
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d",
                "%m-%d-%Y", "%d-%m-%Y", "%b %d %Y", "%d %b %Y",
                "%Y/%m/%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _normalise_side(v: object) -> str:
    s = str(v).strip().lower()
    if s in ("long", "l", "buy", "多", "做多"):
        return "long"
    if s in ("short", "s", "sell", "空", "做空"):
        return "short"
    return "flat"


def _normalise_transaction_move(v: object) -> str:
    """Map raw action strings to canonical transaction_move values."""
    s = str(v).strip().lower()
    for canonical, aliases in TRANSACTION_MOVE_ALIASES.items():
        if s in aliases:
            return canonical
    best_canonical, best_len = None, 0
    for canonical, aliases in TRANSACTION_MOVE_ALIASES.items():
        for alias in aliases:
            if alias in s and len(alias) > best_len:
                best_len = len(alias)
                best_canonical = canonical
    return best_canonical if best_canonical else s


# ── Holding DTOs ──────────────────────────────────────────────────────────────

class BrokerHoldingDTO(BaseModel):
    """
    Broker account holding — requires symbol + shares.
    Produced from Fidelity/Schwab/IBKR portfolio exports and screenshots.

    Required:  symbol, shares
    Optional:  name, avg_cost_per_share, total_cost, side, trading dates
    """
    dto_type: str = "broker_holding"  # discriminator for frontend/backend routing

    symbol: str
    name: str = ""
    shares: float
    avg_cost_per_share: float = 0.0
    total_cost: float = 0.0
    side: str = "long"
    first_trading_date: Optional[str] = None
    last_trading_date: Optional[str] = None

    @field_validator("shares", "avg_cost_per_share", "total_cost", mode="before")
    @classmethod
    def clean_decimal(cls, v):
        return _clean_decimal(v)

    @field_validator("side", mode="before")
    @classmethod
    def clean_side(cls, v):
        return _normalise_side(v)

    @field_validator("first_trading_date", "last_trading_date", mode="before")
    @classmethod
    def clean_date(cls, v):
        return _clean_date(v)


class ConstituentHoldingDTO(BaseModel):
    """
    Index / fund constituent holding — requires symbol + (weight OR price).
    Produced from ETF/index fact sheets and fund screener screenshots
    (e.g. SPY holdings from stockanalysis.com, MSCI index exports).

    Required:  symbol + at least one of: weight, price
    Optional:  name, holding_type, change
    """
    dto_type: str = "constituent_holding"  # discriminator

    symbol: str
    name: str = ""
    weight: float = 0.0       # % weight in the index/fund (e.g. 6.5 for 6.5%)
    price: float = 0.0        # last price / market price
    holding_type: str = ""    # e.g. "Common Stock", "ETF"
    change: float = 0.0       # day change %

    @field_validator("weight", "price", "change", mode="before")
    @classmethod
    def clean_decimal(cls, v):
        return _clean_decimal(v)


# ── Backwards-compatible alias ────────────────────────────────────────────────
# Existing code that imports HoldingDTO continues to work unchanged.
HoldingDTO = BrokerHoldingDTO


# ── Transaction DTO ───────────────────────────────────────────────────────────

class TransactionDTO(BaseModel):
    """Trade history record — unchanged from original schema."""
    dto_type: str = "transaction"  # discriminator

    symbol: str
    transaction_move: str  # "buy"|"sell"|"short_sell"|"cover"|"dividend"
    shares: float
    price_per_share: float = 0.0
    total_amount: float = 0.0
    commission: float = 0.0
    fees: float = 0.0
    net_amount: float = 0.0
    executed_at: Optional[str] = None
    settled_at: Optional[str] = None
    notes: str = ""

    @field_validator("shares", "price_per_share", "total_amount",
                     "commission", "fees", "net_amount", mode="before")
    @classmethod
    def clean_decimal(cls, v):
        return _clean_decimal(v)

    @field_validator("transaction_move", mode="before")
    @classmethod
    def clean_move(cls, v):
        return _normalise_transaction_move(v)

    @field_validator("executed_at", "settled_at", mode="before")
    @classmethod
    def clean_date(cls, v):
        return _clean_date(v)
