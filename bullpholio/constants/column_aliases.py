"""
constants/column_aliases.py
----------------------------
All column name alias mappings and required field definitions.

Change log
──────────
2024-03  "cost basis" moved from avg_cost_per_share → total_cost.
         Rationale: in all major broker UIs (Fidelity, Schwab, IBKR),
         "Cost Basis" = total amount paid for the position (total_cost).
         "Avg Cost" / "Average Cost" = per-share cost (avg_cost_per_share).
         Misclassifying it caused OCR sanity warnings on fund tickers
         (e.g. GTLOX showing avg_cost=1033 when actual total_cost=1033).
2024-04  Added REBALANCE_COLUMN_ALIASES for rebalance plan documents.
         Broadened shares/transaction_move aliases to catch OCR variants.
"""

HOLDING_COLUMN_ALIASES: dict[str, list[str]] = {
    "symbol":             ["symbol", "ticker", "stock", "security", "stock symbol",
                           "ticker symbol", "instrument", "security id",
                           "股票代码", "代码", "证券代码"],
    "name":               ["name", "company", "security name", "stock name",
                           "description", "company name", "公司名称", "名称", "证券名称"],
    "shares":             ["shares", "quantity", "qty", "units", "position",
                           "shares held", "number of shares", "# shares",
                           "no. of shares", "holding", "pos", "position size",
                           "股数", "数量", "持股数"],
    "avg_cost_per_share": ["avg cost", "average cost", "avg price", "average price",
                           "avg cost per share", "average cost per share",
                           "unit cost", "均价", "平均成本", "成本价"],
    "total_cost":         ["total cost", "cost", "book value", "total value",
                           "book cost", "total book value",
                           # "Cost Basis" in broker UIs = total amount paid, not per-share.
                           # Fidelity, Schwab, IBKR all use this convention.
                           "cost basis", "total cost basis",
                           "总成本", "账面价值", "成本基础"],
    "side":               ["side", "position type", "long/short", "direction",
                           "position side", "多空", "方向"],
    "first_trading_date": ["first trading date", "open date", "first purchase",
                           "start date", "inception date", "首次交易日"],
    "last_trading_date":  ["last trading date", "last purchase", "last trade date",
                           "most recent trade", "最近交易日"],
    # Portfolio display-only fields — NEVER appear in transaction tables.
    "last_price":   ["last price", "current price", "market price", "latest price",
                     "price", "last", "现价", "最新价"],
    "market_value": ["market value", "mkt value", "mkt val", "current value",
                     "portfolio value", "市值"],
    "day_gain":     ["day's gain", "day gain", "daily gain", "today's gain",
                     "change today", "日收益"],
    "overall_gain": ["overall gain", "total gain", "total return", "gain/loss",
                     "unrealized gain", "总收益", "浮动盈亏"],
    # Performance display-only fields — in performance reports / position summaries
    "overall_gain_pct": ["overall gain %", "gain %", "return %", "total return %",
                         "% gain", "% return", "% change", "pct return",
                         "overall gain percent", "unrealized gain %", "收益率"],
}

TRANSACTION_COLUMN_ALIASES: dict[str, list[str]] = {
    "symbol":           ["symbol", "ticker", "stock", "security", "instrument",
                         "stock symbol", "security id", "股票代码", "代码"],
    "transaction_move": [
        # Standard column headers
        "type", "action", "transaction type", "move", "buy/sell",
        "side", "transaction", "trade type", "order type",
        "activity", "activity type", "trans type",
        # Spreadsheet / bank export variants
        "description",     # catch "BUY 100 AAPL" style descriptions
        "debit/credit",    # some banks use this
        "交易类型", "操作类型",
    ],
    "shares":           ["shares", "quantity", "qty", "units", "volume",
                         "number of shares", "# shares", "no. of shares",
                         "amount", "share count",
                         "股数", "数量"],
    "price_per_share":  ["price", "price per share", "unit price", "execution price",
                         "fill price", "trade price", "单价", "成交价"],
    "total_amount":     ["total", "total amount", "gross amount", "gross",
                         "value", "total value", "debit", "credit",
                         "总金额", "总价"],
    "commission":       ["commission", "comm", "brokerage", "broker fee", "佣金"],
    "fees":             ["fees", "fee", "charges", "other fees", "regulatory fee",
                         "手续费", "费用"],
    "net_amount":       ["net amount", "net", "settlement amount", "net proceeds",
                         "net debit", "net credit", "净额"],
    "executed_at":      ["date", "execution date", "trade date", "executed at",
                         "transaction date", "trade time", "settled date",
                         "交易日期", "成交日期"],
    "settled_at":       ["settlement date", "settle date", "settled at",
                         "value date", "清算日期"],
    "notes":            ["notes", "memo", "remarks", "comment",
                         "备注", "说明"],
}

TRANSACTION_MOVE_ALIASES: dict[str, list[str]] = {
    "buy":        ["buy", "purchase", "purchased", "bought", "b",
                   "buy order", "market buy", "limit buy",
                   "买入", "买", "申购"],
    "sell":       ["sell", "sold", "s", "sell order", "market sell", "limit sell",
                   "卖出", "卖", "赎回"],
    "short_sell": ["short sell", "short sale", "short", "ss", "做空", "融券"],
    "cover":      ["cover", "buy to cover", "btc", "cover short", "平仓", "回补"],
    "dividend":   ["dividend", "div", "income", "distribution", "reinvest",
                   "reinvestment", "股息", "分红"],
}

TRANSACTION_MOVE_VALID: set[str] = set(TRANSACTION_MOVE_ALIASES.keys())

HOLDING_REQUIRED: set[str]     = {"symbol", "shares"}
TRANSACTION_REQUIRED: set[str] = {"symbol", "transaction_move", "shares", "executed_at"}

# ConstituentHoldingDTO only needs symbol + weight OR price
CONSTITUENT_COLUMN_ALIASES: dict[str, list[str]] = {
    "symbol":       ["symbol", "ticker", "stock", "security", "instrument",
                     "stock symbol", "security id", "股票代码", "代码"],
    "name":         ["name", "company", "security name", "stock name",
                     "description", "公司名称", "名称"],
    "weight":       ["weight", "% weight", "wt", "allocation", "% alloc",
                     "portfolio weight", "index weight", "权重", "占比"],
    "price":        ["price", "last price", "current price", "market price",
                     "latest price", "last", "现价", "最新价"],
    "holding_type": ["holding type", "type", "asset type", "security type",
                     "asset class", "类型"],
    "change":       ["change", "% change", "chg", "day change", "daily change",
                     "变动", "涨跌幅"],
}

CONSTITUENT_REQUIRED: set[str] = {"symbol"}  # weight OR price checked at row level


# ── Rebalance plan ─────────────────────────────────────────────────────────────
# A rebalance plan shows current vs target allocation with trade instructions.
# Distinguishing signals vs constituent_holding: "target weight" AND either
# "drift"/"difference"/"action"/"trade amount".
#
# Examples:
#   Personal Capital "Rebalance" export
#   M1 Finance portfolio slice table
#   Any hand-built spreadsheet with Target %/Current %/Difference columns

REBALANCE_COLUMN_ALIASES: dict[str, list[str]] = {
    "symbol":         ["symbol", "ticker", "stock", "security", "instrument",
                       "stock symbol", "security id", "股票代码", "代码"],
    "name":           ["name", "description", "security name", "company",
                       "公司名称", "名称"],
    "current_weight": ["current weight", "current %", "current allocation",
                       "cur wt", "cur %", "current pct", "actual weight",
                       "actual %", "actual allocation", "current", "existing %",
                       "持仓权重", "当前占比"],
    "target_weight":  ["target weight", "target %", "target allocation",
                       "tgt wt", "tgt %", "target pct", "desired weight",
                       "desired %", "goal %", "new %", "ideal %",
                       "目标权重", "目标占比"],
    "drift":          ["drift", "difference", "diff", "over/under", "deviation",
                       "delta", "off by", "deviation %", "rebalance %",
                       "偏差", "差值"],
    "action":         ["action", "trade", "rebalance action", "buy/sell",
                       "direction", "type", "to do", "recommended action",
                       "操作", "建议"],
    "trade_amount":   ["trade amount", "amount", "dollar amount", "trade value",
                       "$ to trade", "trade $", "value", "notional",
                       "金额", "交易金额"],
    "trade_shares":   ["shares to trade", "shares", "qty", "quantity",
                       "units", "share count", "# shares",
                       "股数", "交易股数"],
}

# Required: symbol + (target_weight OR current_weight) — caller checks the OR
REBALANCE_REQUIRED: set[str] = {"symbol"}
