"""
core/type_detector.py
---------------------
Detects whether a DataFrame represents Holding or Transaction data.
"""

from bullpholio.constants.column_aliases import (
    HOLDING_COLUMN_ALIASES,
    TRANSACTION_COLUMN_ALIASES,
)
from bullpholio.core.column_mapper import map_columns


def detect_input_type(columns: list[str]) -> tuple[str, str]:
    """
    Decide whether the table looks like Holding or Transaction data.

    Scoring uses three tiers:

    Tier 1 — exclusive tx fields (only appear in transaction tables):
      commission, net_amount, settled_at → +3 each, strict threshold 0.75.

    Tier 2 — strong discriminating fields (strict threshold 0.75):
      transaction_move, executed_at → +2 tx
      avg_cost_per_share, total_cost, last_price, market_value → +2 holding

    Tier 3 — breadth tiebreaker: count total matched columns per alias map.

    Returns: (input_type, confidence_note) for logging.
    """
    tx_strict = map_columns(columns, TRANSACTION_COLUMN_ALIASES, threshold=0.75)
    tx_loose  = map_columns(columns, TRANSACTION_COLUMN_ALIASES, threshold=0.6)
    h_strict  = map_columns(columns, HOLDING_COLUMN_ALIASES,     threshold=0.75)
    h_loose   = map_columns(columns, HOLDING_COLUMN_ALIASES,     threshold=0.6)

    # Tier 1: exclusive tx fields
    tx_exclusive = (
        (3 if tx_strict.get("commission") else 0) +
        (3 if tx_strict.get("net_amount") else 0) +
        (3 if tx_strict.get("settled_at") else 0)
    )

    # Tier 2: discriminating fields
    tx_hits = tx_exclusive + (
        (2 if tx_strict.get("transaction_move") else 0) +
        (2 if tx_strict.get("executed_at")      else 0) +
        (1 if tx_loose.get("price_per_share")   else 0)
    )
    holding_hits = (
        (2 if h_strict.get("avg_cost_per_share") else 0) +
        (2 if h_strict.get("total_cost")         else 0) +
        (2 if h_strict.get("last_price")         else 0) +
        (2 if h_strict.get("market_value")       else 0) +
        (1 if h_strict.get("first_trading_date") else 0) +
        (1 if h_strict.get("day_gain")           else 0) +
        (1 if h_strict.get("total_gain")         else 0)
    )

    note = f"tx_hits={tx_hits}, holding_hits={holding_hits}"

    if tx_hits > holding_hits and tx_hits >= 2:
        return "transaction", note

    if holding_hits > tx_hits:
        return "holding", note

    # Tier 3: breadth tiebreaker
    tx_breadth = sum(1 for v in tx_loose.values()  if v is not None)
    h_breadth  = sum(1 for v in h_loose.values()   if v is not None)
    note += f", tx_breadth={tx_breadth}, h_breadth={h_breadth}"

    if tx_breadth > h_breadth:
        return "transaction", note
    return "holding", note
