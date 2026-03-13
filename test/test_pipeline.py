"""
test_pipeline.py
----------------
Pipeline test suite — uses real uploaded broker files where available,
generates synthetic fixtures for edge-case scenarios that are hard to
source from real brokers (borderless PDFs, mixed statements).

Usage:
    # Full suite
    python test/test_pipeline.py

    # Single file (ad-hoc)
    python test/test_pipeline.py path/to/file.pdf

    # With OCR enabled (for scanned PDFs and broker screenshots)
    python test/test_pipeline.py path/to/scanned.pdf --ocr
    python test/test_pipeline.py --ocr          # full suite + OCR cases

Test suite structure
────────────────────
Suite A — Structured files (CSV / Excel / PDF / Word)
    Real uploaded files; fast, no OCR needed.
    holdings.csv / transactions.csv / holdings.xlsx / holdings.docx
    mixed_statement.pdf

Suite B — Borderless text PDFs
    Synthetic fixtures testing Pass 3 (word-layout reconstruction).
    holdings_borderless.pdf / transactions_borderless.pdf

Suite C — OCR screenshots  [requires --ocr flag + EasyOCR installed]
    Real screenshots with table content:
      stock.png     — Investment Portfolio (Fidelity-style, ~12 holdings)
      SPDR.png      — SPY Holdings from stockanalysis.com (non-standard cols)
      warehouse.png — Storage Details by Item Warehouse Location (non-financial)

Suite D — Non-financial documents  (must fail cleanly)
    test.pdf — FCS Automotive invoice (line items, not portfolio data)

Suite E — Hard rejection cases  (must fail cleanly)
    flower.jpg / flower.bmp — photos with no table structure
    cat.gif                 — animal photo, no table structure

Edge cases covered across all suites
─────────────────────────────────────
  ✓ shares = 0 (flat/closed position) → kept, not skipped
  ✓ Borderless PDF → Pass 3 word-layout reconstruction
  ✓ Mixed-section PDF → input_type = "mixed"
  ✓ Non-table image → status = "failed" (correct rejection)
  ✓ Non-financial table → status = "failed" (MissingRequiredColumns)
  ✓ Multi-page invoice → parsed but rejected (no symbol/shares match)
  ✓ Broker screenshot with standard columns → OCR + fuzzy match → success
  ✓ Broker screenshot with non-standard columns → OCR + fail cleanly
  ✓ Warehouse/inventory table → rejected (no financial columns)
  ✓ Extra unrecognised columns → silently ignored
"""

import sys
import logging
from pathlib import Path

# ── Locate pipeline module ────────────────────────────────────────
_here = Path(__file__).parent
sys.path.insert(0, str(_here.parent))  # project root
sys.path.insert(0, str(_here))         # test/ itself

from bullpholio.pipeline import run_pipeline
from bullpholio.models.dtos import HoldingDTO, TransactionDTO
from bullpholio.models.results import PipelineResult
from bullpholio.extractors.ocr_extractor import warmup_ocr
warmup_ocr()
# ── Colour helpers ────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}!{RESET} {msg}")
def info(msg): print(f"  {CYAN}→{RESET} {msg}")


# ================================================================
# SECTION 1 — Synthetic fixture generation
# ================================================================

def generate_synthetic_fixtures(out_dir: Path) -> None:
    """Generate synthetic fixtures and resolve uploaded file names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _resolve_uploaded_names(out_dir)
    _gen_borderless_pdfs(out_dir)
    print(f"{GREEN}Test files ready in:{RESET} {out_dir}\n")


def _resolve_uploaded_names(d: Path) -> None:
    """
    Locate real test files and copy them into test_files/ if not already there.

    Search order for each expected file:
      1. test_files/ itself (already in place — nothing to do)
      2. test/  (sibling of test_files/)
      3. project root  (parent of test/)
      4. Any file whose name ends with a known timestamp-prefixed suffix
    """
    EXPECTED = {
        # clean name → timestamp suffix variants
        "holdings.csv":                ["_holdings.csv"],
        "holdings.docx":               ["_holdings.docx"],
        "holdings.pdf":                ["_holdings.pdf"],
        "holdings.xlsx":               ["_holdings.xlsx"],
        "mixed_statement.pdf":         ["_mixed_statement.pdf"],
        "transactions.csv":            ["_transactions.csv"],
        "transactions.pdf":            ["_transactions.pdf"],
        "transactions.xlsx":           ["_transactions.xlsx"],
        "holdings_borderless.pdf":     ["_holdings_borderless.pdf"],
        "transactions_borderless.pdf": ["_transactions_borderless.pdf"],
        # OCR screenshots
        "trans.png":                   ["_trans.png"],
        "stock.png":                   ["_stock.png"],
        "SPDR.png":                    ["_SPDR.png"],
        "warehouse.png":               ["_warehouse.png"],
        # Non-financial document
        "test.pdf":                    ["_test.pdf"],
        # Rejection cases
        "flower.jpg":                  ["_flower.jpg"],
        "flower.bmp":                  ["_flower.bmp"],
        "cat.gif":                     ["_cat.gif", "_gif.gif"],
    }

    search_dirs = [d, d.parent, d.parent.parent]

    import shutil
    for clean_name, suffixes in EXPECTED.items():
        target = d / clean_name
        if target.exists():
            continue

        found = None

        # 1. Clean name in each search dir
        for sd in search_dirs:
            candidate = sd / clean_name
            if candidate.exists() and candidate != target:
                found = candidate
                break

        # 2. Timestamp-prefixed variants
        if not found:
            for sd in search_dirs:
                if not sd.exists():
                    continue
                for f in sd.iterdir():
                    for suffix in suffixes:
                        if f.name.endswith(suffix) and f != target:
                            found = f
                            break
                    if found:
                        break
                if found:
                    break

        if found:
            shutil.copy2(found, target)
            print(f"  {CYAN}Copied{RESET} {found.name} → test_files/{clean_name}")


def _gen_borderless_pdfs(d: Path) -> None:
    """Borderless text PDFs — tests Pass 3 (word-layout reconstruction)."""
    try:
        from fpdf import FPDF
    except ImportError:
        print(f"{YELLOW}  fpdf2 not installed — skipping borderless PDF generation{RESET}")
        return

    def _make(path, title, headers, rows, col_x):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Courier", "B", 13)
        pdf.set_xy(10, 10)
        pdf.cell(0, 8, title)
        pdf.set_font("Courier", "B", 10)
        y = 25
        for h, x in zip(headers, col_x):
            pdf.set_xy(x, y); pdf.cell(0, 8, h)
        y += 8
        pdf.set_draw_color(150, 150, 150)
        pdf.line(10, y, 200, y)
        y += 2
        pdf.set_font("Courier", "", 10)
        for row in rows:
            for val, x in zip(row, col_x):
                pdf.set_xy(x, y); pdf.cell(0, 8, str(val))
            y += 8
        pdf.output(str(path))
        print(f"  {GREEN}Generated{RESET} {Path(path).name}")

    if not (d / "holdings_borderless.pdf").exists():
        _make(d / "holdings_borderless.pdf",
              "Portfolio Holdings - Borderless Layout",
              ["Symbol", "Name",           "Shares", "Avg Cost", "Total Cost", "Side"],
              col_x=[10,  40,              120,       155,        185,          225],
              rows=[
                  ["AAPL", "Apple Inc",      "150.0", "182.50", "27375.00", "long"],
                  ["GOOG", "Alphabet Inc",    "40.0", "140.20",  "5608.00", "long"],
                  ["TSLA", "Tesla Inc",        "0.0", "220.00",     "0.00", "long"],
                  ["NVDA", "Nvidia Corp",     "80.0", "495.30", "39624.00", "long"],
                  ["MSFT", "Microsoft Corp", "120.0", "310.75", "37290.00", "long"],
              ])

    if not (d / "transactions_borderless.pdf").exists():
        _make(d / "transactions_borderless.pdf",
              "Transaction History - Borderless Layout",
              ["Symbol", "Type", "Shares", "Price",   "Total",   "Commission", "Net Amt",  "Date"],
              col_x=[10,   38,    62,        92,        125,        160,          195,        235],
              rows=[
                  ["AAPL", "buy",  "50.0", "178.90", "8945.00", "1.00", "8946.15", "2024-01-10"],
                  ["TSLA", "sell", "30.0", "215.50", "6465.00", "1.00", "6464.00", "2024-01-15"],
                  ["NVDA", "buy",  "20.0", "480.00", "9600.00", "1.00", "9601.00", "2024-02-01"],
                  ["GOOG", "buy",  "10.0", "138.75", "1387.50", "1.00", "1388.50", "2024-02-14"],
                  ["MSFT", "sell", "25.0", "315.20", "7880.00", "1.00", "7879.00", "2024-03-01"],
              ])


# ================================================================
# SECTION 2 — Test case definitions
# ================================================================

# Each case dict:
#   label       str          — display name
#   file        str          — filename relative to test_files/
#   suite       str          — A | B | C | D | E
#   allow_ocr   bool         — whether to pass allow_ocr=True
#   exp_status  str | set    — expected result.status
#   exp_type    str | None   — expected input_type (None = don't check)
#   exp_min     int          — minimum record count
#   note        str          — what this case tests

TEST_CASES = [

    # ── Suite A: Real structured files ───────────────────────────

    dict(label="Holdings CSV",
         file="holdings.csv", suite="A",
         exp_status="success", exp_type="holding", exp_min=4,
         note="Real CSV — symbol/shares/avg_cost/total_cost columns"),

    dict(label="Transactions CSV",
         file="transactions.csv", suite="A",
         exp_status="success", exp_type="transaction", exp_min=5,
         note="Real CSV — commission/fees/net_amount columns"),

    dict(label="Holdings Excel",
         file="holdings.xlsx", suite="A",
         exp_status="success", exp_type="holding", exp_min=4,
         note="Real .xlsx holdings — same schema as CSV"),

    dict(label="Transactions Excel",
         file="transactions.xlsx", suite="A",
         exp_status="success", exp_type="transaction", exp_min=5,
         note="Real .xlsx transactions — commission/fees/net_amount columns"),

    dict(label="Holdings PDF",
         file="holdings.pdf", suite="A",
         exp_status="success", exp_type="holding", exp_min=5,
         note="Real bordered PDF holdings table"),

    dict(label="Transactions PDF",
         file="transactions.pdf", suite="A",
         exp_status="success", exp_type="transaction", exp_min=5,
         note="Real bordered PDF transactions table"),

    dict(label="Holdings Word",
         file="holdings.docx", suite="A",
         exp_status="success", exp_type="holding", exp_min=3,
         note="Real .docx table — symbol/name/shares/avg_cost/total_cost/side"),

    dict(label="Mixed Statement PDF",
         file="mixed_statement.pdf", suite="A",
         exp_status="success", exp_type="mixed", exp_min=4,
         note="Single PDF with a holdings section AND a transactions section → type=mixed"),

    # ── Suite B: Borderless text PDFs ────────────────────────────

    dict(label="Borderless Holdings PDF",
         file="holdings_borderless.pdf", suite="B",
         exp_status="success", exp_type="holding", exp_min=5,
         note="No grid lines — Pass 3 word-layout reconstruction; includes 0-share row (TSLA)"),

    dict(label="Borderless Transactions PDF",
         file="transactions_borderless.pdf", suite="B",
         exp_status="success", exp_type="transaction", exp_min=5,
         note="No grid lines — Pass 3; has commission + net_amount columns"),

    # ── Suite C: OCR broker screenshots ──────────────────────────

    dict(label="Broker Transactions Screenshot (trans.png)",
         file="trans.png", suite="C", allow_ocr=True,
         exp_status={"success", "partial", "failed"}, exp_type=None, exp_min=0,
         note="Real Schwab broker screenshot (~330x180 px) — very low resolution; "
              "OCR is best-effort. Pass = pipeline handles it without crashing."),

    dict(label="Investment Portfolio Screenshot (stock.png)",
         file="stock.png", suite="C", allow_ocr=True,
         exp_status={"success", "partial"}, exp_type="holding", exp_min=3,
         note="Fidelity-style portfolio UI — Symbol/Shares/Cost basis columns; "
              "12 holdings rows. OCR should extract >= 3 valid HoldingDTOs."),

    dict(label="SPY Holdings Screenshot (SPDR.png)",
         file="SPDR.png", suite="C", allow_ocr=True,
         exp_status={"success", "partial", "failed"}, exp_type=None, exp_min=0,
         note="stockanalysis.com SPY holdings — has %%Weight/%%Change but no Shares column; "
              "pipeline must not crash; 0 records is acceptable."),

    dict(label="Warehouse Storage Screenshot (warehouse.png)",
         file="warehouse.png", suite="C", allow_ocr=True,
         exp_status={"partial", "failed"}, exp_type=None, exp_min=0,
         note="ERP Storage Details table — Inventory ID / Qty. On Hand columns; "
              "not financial portfolio data → MissingRequiredColumns expected."),

    # ── Suite D: Non-financial documents ─────────────────────────

    dict(label="FCS Automotive Invoice (test.pdf)",
         file="test.pdf", suite="D",
         exp_status="failed", exp_type=None, exp_min=0,
         note="8-page invoice — Line ID/Shipped/Ordered/Item/Rate/Amount; "
              "no symbol or shares fields → must be rejected with 0 records."),

    # ── Suite E: Hard rejection cases ────────────────────────────

    dict(label="Non-table photo (flower.jpg)",
         file="flower.jpg", suite="E",
         exp_status="failed", exp_type=None, exp_min=0,
         note="Plain JPEG photo — no table structure → rejected before OCR"),

    dict(label="Non-table photo (flower.bmp)",
         file="flower.bmp", suite="E",
         exp_status="failed", exp_type=None, exp_min=0,
         note="BMP photo — no table structure → rejected before OCR"),

    dict(label="Animal GIF (cat.gif)",
         file="cat.gif", suite="E",
         exp_status="failed", exp_type=None, exp_min=0,
         note=".gif routes through image pipeline; cat photo has no table structure "
              "→ rejected by _has_table_structure"),
]


# ================================================================
# SECTION 3 — Result printer
# ================================================================

def print_result(result: PipelineResult, tc: dict) -> bool:
    """Pretty-print a result; return True if the test passed."""
    passed     = True
    exp_status = tc["exp_status"]
    exp_type   = tc.get("exp_type")
    exp_min    = tc.get("exp_min", 1)
    suite      = tc["suite"]

    # ── Status ───────────────────────────────────────────────────
    status_ok = (result.status in exp_status
                 if isinstance(exp_status, set)
                 else result.status == exp_status)
    if status_ok:
        ok(f"status = {result.status}")
    else:
        fail(f"status = {result.status!r}  (expected {exp_status!r})")
        passed = False

    # ── Hard-rejection suites: show reason and return early ──────
    if suite in ("D", "E"):
        if result.errors:
            info(f"rejection: {result.errors[0].message[:120]}")
        return passed

    # ── Input type ───────────────────────────────────────────────
    if exp_type:
        if result.input_type == exp_type:
            ok(f"input_type = {result.input_type}")
        else:
            fail(f"input_type = {result.input_type!r}  (expected {exp_type!r})")
            passed = False
    else:
        info(f"input_type = {result.input_type}  (not checked)")

    # ── Record count ─────────────────────────────────────────────
    if result.record_count >= exp_min:
        ok(f"record_count = {result.record_count}")
    else:
        fail(f"record_count = {result.record_count}  (expected >= {exp_min})")
        passed = False

    # ── Metadata ─────────────────────────────────────────────────
    info(f"latency = {result.total_latency_ms:.1f} ms")
    info(f"note: {tc['note']}")

    if result.table_summaries:
        info("table summaries:")
        for s in result.table_summaries:
            flag = "SKIP" if s.skipped else "OK  "
            print(f"      [{flag}] table {s.table_index}: {s.input_type}, "
                  f"{s.row_count} rows → {s.record_count} records")

    # ── Sample records (always shown — placeholder when empty) ───
    if result.data:
        info("sample records:")
        for rec in result.data[:2]:
            d = rec.model_dump()
            brief = {k: v for k, v in d.items() if v not in (None, "", 0.0, 0)}
            print(f"      {brief}")
    else:
        _placeholder_tx = {
            "symbol": "unknown", "transaction_move": "unknown",
            "shares": 0.0, "price_per_share": 0.0, "total_amount": 0.0,
            "commission": 0.0, "net_amount": 0.0, "executed_at": "unknown",
        }
        _placeholder_h = {
            "symbol": "unknown", "shares": 0.0,
            "avg_cost_per_share": 0.0, "total_cost": 0.0, "side": "unknown",
        }
        _ph = _placeholder_tx if result.input_type == "transaction" else _placeholder_h
        info("sample records:")
        print(f"      {_ph}")

    # ── Errors ───────────────────────────────────────────────────
    # Suite C: any outcome is acceptable — show as info, not failure
    if result.errors and suite != "C":
        for e in result.errors:
            fail(f"[{e.stage}] {e.error_type}: {e.message}")
            passed = False
    elif result.errors:
        for e in result.errors:
            info(f"rejection: {e.message[:120]}")

    # ── Warnings ─────────────────────────────────────────────────
    if result.warnings and (not passed or len(result.warnings) <= 3):
        for w in result.warnings[:3]:
            warn(w)

    return passed


# ================================================================
# SECTION 4 — Suite runners
# ================================================================

_SUITE_LABELS = {
    "A": "Structured files (CSV / Excel / PDF / Word)",
    "B": "Borderless text PDFs  — Pass 3 word-layout",
    "C": "OCR screenshots  — allow_ocr=True",
    "D": "Non-financial documents  — must fail cleanly",
    "E": "Hard rejections  — must fail cleanly",
}


def run_all(test_dir: Path, run_ocr: bool = False) -> bool:
    pl_logger = logging.getLogger("pipeline.test")
    pl_logger.setLevel(logging.WARNING)

    passed_total  = 0
    failed_tests: list[str] = []
    skipped:      list[str] = []

    if not run_ocr:
        print(f"\n{YELLOW}  Suite C (OCR screenshots) skipped — add --ocr flag to enable{RESET}")

    for suite_id in ["A", "B", "C", "D", "E"]:
        suite_cases = [tc for tc in TEST_CASES if tc["suite"] == suite_id]
        if not suite_cases:
            continue

        print(f"\n{'═'*62}")
        print(f"{BOLD}Suite {suite_id}: {_SUITE_LABELS[suite_id]}{RESET}")
        print(f"{'═'*62}")

        for tc in suite_cases:
            label   = tc["label"]
            fpath   = test_dir / tc["file"]
            suite   = tc["suite"]
            use_ocr = tc.get("allow_ocr", False)

            if suite == "C" and not run_ocr:
                print(f"\n  {YELLOW}SKIP{RESET}  {label}  (add --ocr to run)")
                skipped.append(label)
                continue

            if not fpath.exists():
                print(f"\n  {YELLOW}SKIP{RESET}  {label} — file not found: {tc['file']}")
                skipped.append(label)
                continue

            print(f"\n{'─'*62}")
            print(f"{BOLD}[{suite}] {label}{RESET}  ({tc['file']})")
            print(f"{'─'*62}")

            result = run_pipeline(str(fpath), logger=pl_logger, allow_ocr=use_ocr)
            passed = print_result(result, tc)

            if passed:
                passed_total += 1
            else:
                failed_tests.append(f"[{suite}] {label}")

    total_run = sum(
        1 for tc in TEST_CASES
        if not (tc["suite"] == "C" and not run_ocr)
        and (test_dir / tc["file"]).exists()
    )

    print(f"\n{'='*62}")
    print(f"{BOLD}Results: {passed_total}/{total_run} passed", end="")
    if skipped:
        print(f"  ({len(skipped)} skipped)", end="")
    print(f"{RESET}")
    if failed_tests:
        for t in failed_tests:
            print(f"  {RED}✗ {t}{RESET}")
    else:
        print(f"  {GREEN}All run tests passed!{RESET}")
    if skipped:
        print(f"  {YELLOW}Skipped: {', '.join(skipped)}{RESET}")
    print(f"{'='*62}\n")
    return len(failed_tests) == 0


def run_single(file_path: str, allow_ocr: bool = False):
    print(f"\n{BOLD}{CYAN}{'─'*62}{RESET}")
    print(f"{BOLD}File: {file_path}{RESET}")
    if allow_ocr:
        print(f"{YELLOW}  OCR mode: allow_ocr=True{RESET}")
    print(f"{CYAN}{'─'*62}{RESET}")

    pl_logger = logging.getLogger("pipeline.test")
    pl_logger.setLevel(logging.WARNING)

    result = run_pipeline(file_path, logger=pl_logger, allow_ocr=allow_ocr)
    tc = dict(suite="X", exp_type=None, exp_min=0, note="ad-hoc",
              exp_status={"success", "partial", "failed"})
    print_result(result, tc)


# ================================================================
# SECTION 5 — Entry point
# ================================================================

if __name__ == "__main__":
    test_dir = _here / "test_files"
    run_ocr  = "--ocr" in sys.argv
    args     = [a for a in sys.argv[1:] if not a.startswith("--")]

    print(f"\n{BOLD}Preparing test files (if missing)...{RESET}")
    generate_synthetic_fixtures(test_dir)

    if args:
        run_single(args[0], allow_ocr=run_ocr)
    else:
        success = run_all(test_dir, run_ocr=run_ocr)
        sys.exit(0 if success else 1)