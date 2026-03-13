"""
test_pipeline.py
----------------
Pipeline test suite — uses real uploaded broker files where available,
generates synthetic fixtures for edge-case scenarios.

Usage:
    python test/test_pipeline.py                   # full suite, compact output
    python test/test_pipeline.py --ocr             # include Suite C OCR cases
    python test/test_pipeline.py --ocr --verbose   # full warnings + table summaries
    python test/test_pipeline.py --quiet           # pass/fail only, no records
    python test/test_pipeline.py path/to/file.pdf  # single file ad-hoc
    python test/test_pipeline.py path/to/file.pdf --ocr --verbose

Output levels
─────────────
  default   status · input_type · record_count · latency · 2 sample records
            · warning count + first warning · errors (always shown)
  --verbose all of the above + full warnings + table summaries
  --quiet   status line only (pass/fail), no records

Test suite structure
────────────────────
Suite A — Structured files (CSV / Excel / PDF / Word)
Suite B — Borderless text PDFs  (Pass 3 word-layout)
Suite C — OCR screenshots        [requires --ocr]
Suite D — Non-financial docs     (must fail cleanly)
Suite E — Hard rejections        (must fail cleanly)
"""

import sys
import logging
from pathlib import Path

# ── Locate pipeline module ────────────────────────────────────────
_here = Path(__file__).parent
sys.path.insert(0, str(_here.parent))
sys.path.insert(0, str(_here))

from bullpholio.pipeline import run_pipeline
from bullpholio.models.results import PipelineResult

# ── CLI flags ─────────────────────────────────────────────────────
VERBOSE = "--verbose" in sys.argv
QUIET   = "--quiet"   in sys.argv

# ── Colour helpers ────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):  print(f"  {RED}✗{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}!{RESET} {msg}")
def info(msg):  print(f"  {CYAN}→{RESET} {msg}")
def dim(msg):   print(f"  {DIM}{msg}{RESET}")


# ================================================================
# SECTION 1 — Synthetic fixture generation
# ================================================================

def generate_synthetic_fixtures(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _resolve_uploaded_names(out_dir)
    _gen_borderless_pdfs(out_dir)
    print(f"{GREEN}Test files ready in:{RESET} {out_dir}\n")


def _resolve_uploaded_names(d: Path) -> None:
    EXPECTED = {
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
        "trans.png":                   ["_trans.png"],
        "stock.png":                   ["_stock.png"],
        "SPDR.png":                    ["_SPDR.png"],
        "warehouse.png":               ["_warehouse.png"],
        "test.pdf":                    ["_test.pdf"],
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
        for sd in search_dirs:
            candidate = sd / clean_name
            if candidate.exists() and candidate != target:
                found = candidate
                break
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

TEST_CASES = [
    # ── Suite A ──────────────────────────────────────────────────
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

    # ── Suite B ──────────────────────────────────────────────────
    dict(label="Borderless Holdings PDF",
         file="holdings_borderless.pdf", suite="B",
         exp_status="success", exp_type="holding", exp_min=5,
         note="No grid lines — Pass 3 word-layout; includes 0-share row (TSLA)"),
    dict(label="Borderless Transactions PDF",
         file="transactions_borderless.pdf", suite="B",
         exp_status="success", exp_type="transaction", exp_min=5,
         note="No grid lines — Pass 3; has commission + net_amount columns"),

    # ── Suite C ──────────────────────────────────────────────────
    dict(label="Broker Transactions Screenshot (trans.png)",
         file="trans.png", suite="C", allow_ocr=True,
         exp_status={"success", "partial", "low_confidence_partial", "failed"},
         exp_type=None, exp_min=0,
         note="Very low-res Schwab screenshot (404×125px) — expect fast rejection"),
    dict(label="Investment Portfolio Screenshot (stock.png)",
         file="stock.png", suite="C", allow_ocr=True,
         exp_status={"success", "partial", "low_confidence_partial"},
         exp_type="holding", exp_min=3,
         note="Fidelity-style portfolio — Symbol/Shares/Cost Basis; 12 holdings"),
    dict(label="SPY Holdings Screenshot (SPDR.png)",
         file="SPDR.png", suite="C", allow_ocr=True,
         exp_status={"success", "partial", "low_confidence_partial", "failed"},
         exp_type=None, exp_min=0,
         note="stockanalysis.com SPY — %Weight/%Change, no Shares; 0 records OK"),
    dict(label="Warehouse Storage Screenshot (warehouse.png)",
         file="warehouse.png", suite="C", allow_ocr=True,
         exp_status={"partial", "failed"}, exp_type=None, exp_min=0,
         note="ERP table — Inventory ID/Qty; not financial → MissingRequiredColumns"),

    # ── Suite D ──────────────────────────────────────────────────
    dict(label="FCS Automotive Invoice (test.pdf)",
         file="test.pdf", suite="D",
         exp_status="failed", exp_type=None, exp_min=0,
         note="8-page invoice — no symbol/shares → must be rejected"),

    # ── Suite E ──────────────────────────────────────────────────
    dict(label="Non-table photo (flower.jpg)",
         file="flower.jpg", suite="E",
         exp_status="failed", exp_type=None, exp_min=0,
         note="Plain JPEG photo → rejected before OCR"),
    dict(label="Non-table photo (flower.bmp)",
         file="flower.bmp", suite="E",
         exp_status="failed", exp_type=None, exp_min=0,
         note="BMP photo → rejected before OCR"),
    dict(label="Animal GIF (cat.gif)",
         file="cat.gif", suite="E",
         exp_status="failed", exp_type=None, exp_min=0,
         note="Cat photo GIF → rejected by _has_table_structure"),
]


# ================================================================
# SECTION 3 — Record formatter
# ================================================================

# Core fields shown per DTO type — keeps sample output human-readable
_CORE_FIELDS: dict[str, list[str]] = {
    "broker_holding":      ["symbol", "shares", "avg_cost_per_share", "total_cost", "side"],
    "transaction":         ["symbol", "transaction_move", "shares", "price_per_share",
                            "total_amount", "executed_at"],
    "constituent_holding": ["symbol", "weight", "price", "change"],
}

def _format_record(rec: dict) -> dict:
    """Return only core fields for the DTO type, dropping nulls/zeros."""
    keys = _CORE_FIELDS.get(rec.get("dto_type", ""), list(rec.keys()))
    return {k: rec[k] for k in keys if k in rec and rec[k] not in (None, "", 0.0, 0)}


# ================================================================
# SECTION 4 — Result printer
# ================================================================

def print_result(result: PipelineResult, tc: dict) -> bool:
    """Print a result at the configured verbosity level. Returns True if passed."""
    passed     = True
    exp_status = tc["exp_status"]
    exp_type   = tc.get("exp_type")
    exp_min    = tc.get("exp_min", 1)
    suite      = tc["suite"]

    if QUIET:
        # ── Quiet: single status line ─────────────────────────────
        status_ok = (result.status in exp_status
                     if isinstance(exp_status, set)
                     else result.status == exp_status)
        if not status_ok:
            fail(f"status={result.status!r}  expected={exp_status!r}")
            passed = False
        elif result.record_count < exp_min:
            fail(f"record_count={result.record_count}  expected>={exp_min}")
            passed = False
        return passed

    # ── Status ────────────────────────────────────────────────────
    status_ok = (result.status in exp_status
                 if isinstance(exp_status, set)
                 else result.status == exp_status)
    if status_ok:
        ok(f"status = {result.status}")
    else:
        fail(f"status = {result.status!r}  (expected {exp_status!r})")
        passed = False

    # ── Hard-rejection suites: show reason and return early ───────
    if suite in ("D", "E"):
        if result.errors:
            dim(f"rejection: {result.errors[0].message[:110]}")
        return passed

    # ── Input type ────────────────────────────────────────────────
    if exp_type:
        if result.input_type == exp_type:
            ok(f"input_type = {result.input_type}")
        else:
            fail(f"input_type = {result.input_type!r}  (expected {exp_type!r})")
            passed = False
    else:
        info(f"input_type = {result.input_type}  (not checked)")

    # ── Record count ──────────────────────────────────────────────
    if result.record_count >= exp_min:
        ok(f"record_count = {result.record_count}")
    else:
        fail(f"record_count = {result.record_count}  (expected >= {exp_min})")
        passed = False

    # ── Latency ───────────────────────────────────────────────────
    info(f"latency = {result.total_latency_ms:.0f} ms")

    # ── Sample records ────────────────────────────────────────────
    if result.data:
        # Suite C: show parsed table type + prominent OCR label
        if suite == "C":
            info(f"parsed_table_type = {result.input_type or 'unknown'}")
            info("OCR extracted:")
        else:
            info("sample records:")
        for rec in result.data[:3 if suite == "C" else 2]:
            print(f"      {_format_record(rec.model_dump())}")
    else:
        info("sample records: none")

    # ── Warnings: summary by default, full list in verbose ────────
    if result.warnings:
        # Filter to user-actionable warnings only — suppress routing noise
        # (lines that only contain "input_type=..." with no other context)
        actionable = [
            w for w in result.warnings
            if not (w.startswith("input_type=") and "(" in w and len(w) < 60)
        ]
        if actionable:
            if VERBOSE:
                for w in actionable:
                    warn(w)
            else:
                warn(f"{len(actionable)} warning(s)  — run --verbose to see all")
                # Representative summary line: prefer a MissingRequiredColumns or
                # rows_skipped message over a per-row sanity detail, because those
                # are structural issues that affect the whole table.
                summary = next(
                    (w for w in actionable
                     if w.startswith("[MissingRequiredColumns]")
                     or "row" in w.lower() and "skipped" in w.lower()),
                    None,
                )
                if summary is None and any("[sanity]" in w for w in actionable):
                    # All warnings are sanity notes — show a generic message
                    # instead of a per-row detail so the output reads like a report
                    n_sanity = sum(1 for w in actionable if "[sanity]" in w)
                    summary = (
                        f"OCR extracted table, but {n_sanity} row(s) have "
                        "numeric fields that may be misaligned — verify manually."
                    )
                if summary is None:
                    summary = actionable[0]
                warn(summary[:120])

    # ── Table summaries (verbose only) ────────────────────────────
    if VERBOSE and result.table_summaries:
        info("table summaries:")
        for s in result.table_summaries:
            flag = "SKIP" if s.skipped else "OK  "
            conf = f"conf={s.parse_confidence}" if s.parse_confidence != "high" else ""
            susp = f"suspicious={s.suspicious_rows}" if s.suspicious_rows else ""
            meta = "  ".join(x for x in [conf, susp] if x)
            print(f"      [{flag}] table {s.table_index}: {s.input_type}, "
                  f"{s.row_count} rows → {s.record_count} records"
                  + (f"  [{meta}]" if meta else ""))

    # ── Errors ────────────────────────────────────────────────────
    # Suite C: any outcome is acceptable — show as info
    if result.errors and suite != "C":
        for e in result.errors:
            fail(f"[{e.stage}] {e.error_type}: {e.message[:100]}")
            passed = False
    elif result.errors and suite == "C":
        for e in result.errors:
            info(f"rejection: {e.message[:110]}")

    return passed


# ================================================================
# SECTION 5 — Suite runners
# ================================================================

_SUITE_LABELS = {
    "A": "Structured files (CSV / Excel / PDF / Word)",
    "B": "Borderless text PDFs  — Pass 3 word-layout",
    "C": "OCR screenshots  — allow_ocr=True",
    "D": "Non-financial documents  — must fail cleanly",
    "E": "Hard rejections  — must fail cleanly",
    "F": "Auto-discovered files  — no expected outcome",
}

# Extensions the pipeline can handle — used by the directory scanner.
_SCANNABLE_EXTENSIONS: set[str] = {
    ".csv", ".xlsx", ".pdf", ".docx",
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp",
}

# Filenames to silently ignore during auto-scan (OS/editor artefacts).
_SCAN_IGNORE: set[str] = {
    "desktop.ini", ".ds_store", "thumbs.db",
}


def _build_suite_f(test_dir: Path, run_ocr: bool) -> list[dict]:
    """
    Scan test_dir for files not already covered by TEST_CASES A–E.

    Each discovered file becomes a Suite F test case with:
      • exp_status = any (success / partial / low_confidence_partial / failed)
      • exp_min    = 0  (we have no prior knowledge of what it contains)
      • allow_ocr  = same as --ocr flag (images need OCR to produce records)

    Files that do not match a scannable extension are silently skipped.
    """
    _ANY_STATUS = {"success", "partial", "low_confidence_partial", "failed"}

    # Names already owned by TEST_CASES — skip them in the scan
    known_files: set[str] = {tc["file"].lower() for tc in TEST_CASES}

    cases: list[dict] = []
    if not test_dir.exists():
        return cases

    for fpath in sorted(test_dir.iterdir()):
        if not fpath.is_file():
            continue
        if fpath.name.lower() in _SCAN_IGNORE:
            continue
        if fpath.suffix.lower() not in _SCANNABLE_EXTENSIONS:
            continue
        if fpath.name.lower() in known_files:
            continue

        is_image = fpath.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"
        }
        cases.append(dict(
            label=fpath.name,
            file=fpath.name,
            suite="F",
            allow_ocr=run_ocr and is_image,
            exp_status=_ANY_STATUS,
            exp_type=None,
            exp_min=0,
            note="auto-discovered — no expected outcome",
        ))
    return cases


def run_all(test_dir: Path, run_ocr: bool = False) -> bool:
    pl_logger = logging.getLogger("pipeline.test")
    pl_logger.setLevel(logging.WARNING)

    passed_total = 0
    failed_tests: list[str] = []
    skipped:      list[str] = []

    if not run_ocr:
        print(f"\n{YELLOW}  Suite C (OCR screenshots) skipped — add --ocr to enable{RESET}")

    # Build Suite F dynamically from whatever is in test_files/
    suite_f_cases = _build_suite_f(test_dir, run_ocr)

    all_suites: dict[str, list[dict]] = {}
    for suite_id in ["A", "B", "C", "D", "E"]:
        cases = [tc for tc in TEST_CASES if tc["suite"] == suite_id]
        if cases:
            all_suites[suite_id] = cases
    if suite_f_cases:
        all_suites["F"] = suite_f_cases

    for suite_id, suite_cases in all_suites.items():
        print(f"\n{'═'*62}")
        print(f"{BOLD}Suite {suite_id}: {_SUITE_LABELS[suite_id]}{RESET}")
        if suite_id == "F":
            print(f"{DIM}  {len(suite_cases)} file(s) found in test_files/ not in TEST_CASES{RESET}")
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

            if not QUIET:
                print(f"\n{'─'*62}")
                print(f"{BOLD}[{suite}] {label}{RESET}  ({tc['file']})")
                print(f"{'─'*62}")
            else:
                print(f"  [{suite}] {label} ...", end=" ", flush=True)

            result = run_pipeline(str(fpath), logger=pl_logger, allow_ocr=use_ocr)
            passed = print_result(result, tc)

            if QUIET:
                print(f"{GREEN}OK{RESET}" if passed else f"{RED}FAIL{RESET}")

            if passed:
                passed_total += 1
            else:
                failed_tests.append(f"[{suite}] {label}")

    # total_run = known cases that ran + all Suite F cases
    total_run = (
        sum(
            1 for tc in TEST_CASES
            if not (tc["suite"] == "C" and not run_ocr)
            and (test_dir / tc["file"]).exists()
        )
        + len(suite_f_cases)
    )

    print(f"\n{'='*62}")
    print(f"{BOLD}Results: {passed_total}/{total_run} passed", end="")
    if skipped:
        print(f"  ({len(skipped)} skipped)", end="")
    if suite_f_cases:
        print(f"  ({len(suite_f_cases)} auto-discovered)", end="")
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
    # Treat as Suite X: accept any status, show everything
    tc = dict(suite="X", exp_type=None, exp_min=0, note="ad-hoc",
              exp_status={"success", "partial", "low_confidence_partial", "failed"})
    print_result(result, tc)


# ================================================================
# SECTION 6 — Entry point
# ================================================================

if __name__ == "__main__":
    test_dir = _here / "test_files"
    run_ocr  = "--ocr" in sys.argv
    args     = [a for a in sys.argv[1:]
                if not a.startswith("--")]

    print(f"\n{BOLD}Preparing test files (if missing)...{RESET}")
    generate_synthetic_fixtures(test_dir)

    if args:
        run_single(args[0], allow_ocr=run_ocr)
    else:
        success = run_all(test_dir, run_ocr=run_ocr)
        sys.exit(0 if success else 1)
