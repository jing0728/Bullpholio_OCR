"""
test_pipeline.py
----------------
Production testing harness for the Bullpholio document ingestion pipeline.

Architecture
────────────
  TestCase     — typed descriptor: what file, what to expect, which suite
  TestResult   — outcome of one run: pass/fail, latency, sanity flags
  SuiteStats   — per-suite aggregates (pass rate, avg latency)
  RunStats     — global statistics collector driving the summary report

Auto-scan
─────────
  Suite F is built at runtime by scanning test_files/ for any file not
  already registered in suites A–E.  Drop a new file in test_files/ and
  it is picked up automatically on the next run — no code change needed.

  Auto-classification assigns each discovered file to the most appropriate
  category based on its extension:
    structured (CSV/Excel/PDF/Word) → Suite F-S (open verdict, no OCR)
    image                           → Suite F-I (open verdict, OCR if --ocr)

OCR statistics
──────────────
  All cases that ran with allow_ocr=True (Suite C + any Suite F image)
  are collected into a dedicated OCR breakdown table in the summary:
    • success rate (images that produced ≥1 record)
    • average records per successful image
    • average latency
    • per-image status / record count / sanity warning count

Expected pass/fail enforcement
───────────────────────────────
  Suites A–E have strict expectations (exp_status, exp_type, exp_min).
  Failing to meet them marks the case RED and increments the failure count.
  Suite C is always "open verdict" — any status is acceptable.
  Suite F is always "open verdict" — crash-free is the only requirement.

Usage
─────
  python test/test_pipeline.py                   # full suite, default output
  python test/test_pipeline.py --ocr             # include OCR cases
  python test/test_pipeline.py --ocr --verbose   # full warnings + table summaries
  python test/test_pipeline.py --quiet           # one line per test + summary
  python test/test_pipeline.py path/to/file.pdf  # single file ad-hoc
  python test/test_pipeline.py path/to/file.pdf --ocr --verbose
"""

from __future__ import annotations

import sys
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

# ── Locate pipeline module ────────────────────────────────────────────────────
_here = Path(__file__).parent
sys.path.insert(0, str(_here.parent))
sys.path.insert(0, str(_here))

from bullpholio.pipeline import run_pipeline
from bullpholio.models.results import PipelineResult

# ── CLI flags (parsed once at import time) ────────────────────────────────────
VERBOSE = "--verbose" in sys.argv
QUIET   = "--quiet"   in sys.argv
RUN_OCR = "--ocr"     in sys.argv

# ── Colour helpers ────────────────────────────────────────────────────────────
G  = "\033[92m"   # green
R  = "\033[91m"   # red
Y  = "\033[93m"   # yellow
C  = "\033[96m"   # cyan
W  = "\033[0m"    # reset
B  = "\033[1m"    # bold
D  = "\033[2m"    # dim

def _ok(s):   print(f"  {G}✓{W} {s}")
def _fail(s): print(f"  {R}✗{W} {s}")
def _warn(s): print(f"  {Y}!{W} {s}")
def _info(s): print(f"  {C}→{W} {s}")
def _dim(s):  print(f"  {D}{s}{W}")

# Sentinel meaning "accept any pipeline status"
_ANY: set[str] = {"success", "partial", "low_confidence_partial", "failed"}

# ── Supported extensions ──────────────────────────────────────────────────────
_IMAGE_EXTS:  set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}
_STRUCT_EXTS: set[str] = {".csv", ".xlsx", ".pdf", ".docx"}
_SCANNABLE:   set[str] = _IMAGE_EXTS | _STRUCT_EXTS
_IGNORE_FILES: set[str] = {"desktop.ini", ".ds_store", "thumbs.db"}


# ================================================================
# SECTION 1 — Data model
# ================================================================

@dataclass
class TestCase:
    """
    Describes a single pipeline test.

    Fields
    ──────
    label       display name shown in output
    file        filename relative to test_files/
    suite       A | B | C | D | E | F
    exp_status  expected PipelineResult.status — str or set[str]
                pass _ANY to accept any status (Suite C, F)
    exp_type    expected input_type, or None to skip that check
    exp_min     minimum acceptable record_count  (0 = don't check)
    note        one-line description of what this case tests
    allow_ocr   whether to pass allow_ocr=True to run_pipeline
    open_verdict if True, test always passes regardless of output
                 (used for Suite C and F — we just want no crash)
    """
    label:        str
    file:         str
    suite:        str
    exp_status:   Union[str, set[str]]
    exp_type:     Optional[str]
    exp_min:      int
    note:         str
    allow_ocr:    bool = False
    open_verdict: bool = False   # Suite C & F
    base_dir:     Optional[Path] = None  # set by _build_suite_f for Suite F

    def status_ok(self, got: str) -> bool:
        if isinstance(self.exp_status, set):
            return got in self.exp_status
        return got == self.exp_status


@dataclass
class TestResult:
    """
    Outcome of running one TestCase.

    passed       False only when a strict expectation was not met.
                 Cases with open_verdict=True are always True.
    skipped      File missing, or Suite C without --ocr.
    assertions   List of (assertion_label, ok_bool) for the case detail view.
    """
    tc:          TestCase
    result:      Optional[PipelineResult]
    passed:      bool
    skipped:     bool       = False
    skip_reason: str        = ""
    wall_ms:     float      = 0.0
    assertions:  list[tuple[str, bool]] = field(default_factory=list)

    # ── Convenience properties ────────────────────────────────────
    @property
    def label(self)        -> str:            return self.tc.label
    @property
    def suite(self)        -> str:            return self.tc.suite
    @property
    def is_ocr(self)       -> bool:           return self.tc.allow_ocr
    @property
    def record_count(self) -> int:
        return self.result.record_count if self.result else 0
    @property
    def sanity_count(self) -> int:
        if not self.result:
            return 0
        return sum(s.suspicious_rows for s in self.result.table_summaries)
    @property
    def latency_ms(self)   -> float:
        return self.result.total_latency_ms if self.result else 0.0


@dataclass
class SuiteStats:
    """Per-suite pass / latency aggregates."""
    suite_id:  str
    label:     str
    total:     int         = 0
    run:       int         = 0   # total - skipped
    passed:    int         = 0
    failed:    int         = 0
    skipped:   int         = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.run * 100) if self.run else 0.0

    @property
    def avg_ms(self) -> Optional[float]:
        vals = [v for v in self.latencies if v > 0]
        return sum(vals) / len(vals) if vals else None


# ================================================================
# SECTION 2 — Synthetic fixture generation
# ================================================================

def generate_synthetic_fixtures(test_dir: Path) -> None:
    known_dir = test_dir / "known"
    new_dir   = test_dir / "new"
    known_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)
    _resolve_uploaded_names(known_dir)
    _gen_borderless_pdfs(known_dir)
    print(f"{G}Test files ready:{W}")
    print(f"  regression fixtures → {known_dir}")
    print(f"  new files to test   → {new_dir}\n")


def _resolve_uploaded_names(d: Path) -> None:
    """
    Locate canonical test files and copy them into test_files/known/ if absent.
    Handles timestamp-prefixed upload filenames (e.g. '20240315_holdings.csv').
    Search order: known/ → test_files/ → test/ → project root.
    """
    EXPECTED: dict[str, list[str]] = {
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
    import shutil
    # d is known_dir; also search test_files/ (d.parent), test/ (d.parent.parent),
    # and project root (d.parent.parent.parent) for legacy flat layouts.
    search_dirs = [d, d.parent, d.parent.parent, d.parent.parent.parent]
    for clean_name, suffixes in EXPECTED.items():
        target = d / clean_name
        if target.exists():
            continue
        found: Optional[Path] = None
        for sd in search_dirs:
            cand = sd / clean_name
            if cand.exists() and cand != target:
                found = cand; break
        if not found:
            for sd in search_dirs:
                if not sd.exists():
                    continue
                for f in sd.iterdir():
                    if any(f.name.endswith(sfx) for sfx in suffixes) and f != target:
                        found = f; break
                if found:
                    break
        if found:
            shutil.copy2(found, target)
            print(f"  {C}Copied{W} {found.name} → test_files/{clean_name}")


def _gen_borderless_pdfs(d: Path) -> None:
    """Generate synthetic borderless PDFs for Suite B (Pass 3 tests)."""
    try:
        from fpdf import FPDF
    except ImportError:
        print(f"{Y}  fpdf2 not installed — skipping borderless PDF generation{W}")
        return

    def _make(path: Path, title: str, headers: list, rows: list, col_x: list) -> None:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Courier", "B", 13)
        pdf.set_xy(10, 10);  pdf.cell(0, 8, title)
        pdf.set_font("Courier", "B", 10)
        y = 25
        for h, x in zip(headers, col_x):
            pdf.set_xy(x, y);  pdf.cell(0, 8, h)
        y += 8
        pdf.set_draw_color(150, 150, 150)
        pdf.line(10, y, 200, y)
        y += 2
        pdf.set_font("Courier", "", 10)
        for row in rows:
            for val, x in zip(row, col_x):
                pdf.set_xy(x, y);  pdf.cell(0, 8, str(val))
            y += 8
        pdf.output(str(path))
        print(f"  {G}Generated{W} {path.name}")

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
              ["Symbol", "Type",  "Shares", "Price",   "Total",   "Commission", "Net Amt",  "Date"],
              col_x=[10,   38,     62,        92,        125,        160,          195,        235],
              rows=[
                  ["AAPL", "buy",  "50.0", "178.90", "8945.00", "1.00", "8946.15", "2024-01-10"],
                  ["TSLA", "sell", "30.0", "215.50", "6465.00", "1.00", "6464.00", "2024-01-15"],
                  ["NVDA", "buy",  "20.0", "480.00", "9600.00", "1.00", "9601.00", "2024-02-01"],
                  ["GOOG", "buy",  "10.0", "138.75", "1387.50", "1.00", "1388.50", "2024-02-14"],
                  ["MSFT", "sell", "25.0", "315.20", "7880.00", "1.00", "7879.00", "2024-03-01"],
              ])


# ================================================================
# SECTION 3 — Test case registry  (Suites A–E)
# ================================================================

REGISTRY: list[TestCase] = [

    # ── Suite A: Structured files — strict expectations ───────────────────────
    TestCase("Holdings CSV",        "holdings.csv",       "A",
             "success", "holding",     4,
             "CSV with symbol / shares / avg_cost / total_cost columns"),
    TestCase("Transactions CSV",    "transactions.csv",   "A",
             "success", "transaction", 5,
             "CSV with commission / fees / net_amount columns"),
    TestCase("Holdings Excel",      "holdings.xlsx",      "A",
             "success", "holding",     4,
             "xlsx — same schema as holdings.csv"),
    TestCase("Transactions Excel",  "transactions.xlsx",  "A",
             "success", "transaction", 5,
             "xlsx with commission / fees / net_amount"),
    TestCase("Holdings PDF",        "holdings.pdf",       "A",
             "success", "holding",     5,
             "Bordered PDF with holdings table"),
    TestCase("Transactions PDF",    "transactions.pdf",   "A",
             "success", "transaction", 5,
             "Bordered PDF with transactions table"),
    TestCase("Holdings Word",       "holdings.docx",      "A",
             "success", "holding",     3,
             "Word table with symbol / name / shares / avg_cost / side"),
    TestCase("Mixed Statement PDF", "mixed_statement.pdf","A",
             "success", "mixed",       4,
             "One PDF containing both a holdings and a transactions section"),

    # ── Suite B: Borderless PDFs — Pass 3 word-layout reconstruction ──────────
    TestCase("Borderless Holdings PDF",
             "holdings_borderless.pdf",     "B",
             "success", "holding",     5,
             "No grid lines; TSLA 0-share row must be kept"),
    TestCase("Borderless Transactions PDF",
             "transactions_borderless.pdf", "B",
             "success", "transaction", 5,
             "No grid lines; has commission + net_amount columns"),

    # ── Suite C: OCR broker screenshots — open verdict ────────────────────────
    # Any pipeline status is acceptable; we only check it does not crash.
    TestCase("Broker Transactions Screenshot",
             "trans.png",    "C", _ANY, None, 0,
             "Very low-res Schwab screenshot (404×125 px) → fast rejection expected",
             allow_ocr=True, open_verdict=True),
    TestCase("Investment Portfolio Screenshot",
             "stock.png",    "C",
             {"success", "partial", "low_confidence_partial"}, "holding", 3,
             "Fidelity-style UI — Symbol / Shares / Cost Basis; 12 holdings rows",
             allow_ocr=True, open_verdict=True),
    TestCase("SPY Holdings Screenshot",
             "SPDR.png",     "C", _ANY, None, 0,
             "stockanalysis.com — %Weight / %Change, no Shares; 0 records OK",
             allow_ocr=True, open_verdict=True),
    TestCase("Warehouse Storage Screenshot",
             "warehouse.png","C",
             {"partial", "failed"}, None, 0,
             "ERP table — Inventory ID / Qty; not financial → MissingRequiredColumns",
             allow_ocr=True, open_verdict=True),

    # ── Suite D: Non-financial documents — must fail cleanly ──────────────────
    TestCase("FCS Automotive Invoice",
             "test.pdf",     "D",
             "failed", None, 0,
             "8-page invoice — Line ID / Rate / Amount; no symbol or shares field"),

    # ── Suite E: Hard rejection images — must fail cleanly ───────────────────
    TestCase("Non-table photo (flower.jpg)",
             "flower.jpg",   "E",
             "failed", None, 0,
             "Plain JPEG flower photo → rejected at image_no_table gate"),
    TestCase("Non-table photo (flower.bmp)",
             "flower.bmp",   "E",
             "failed", None, 0,
             "BMP photo → rejected at image_no_table gate"),
    TestCase("Animal GIF (cat.gif)",
             "cat.gif",      "E",
             "failed", None, 0,
             "Cat photo GIF → rejected by _has_table_structure"),
]

# ── Suite labels ──────────────────────────────────────────────────────────────
_SUITE_LABELS: dict[str, str] = {
    "A": "Structured files — strict expectations",
    "B": "Borderless text PDFs — Pass 3 word-layout",
    "C": "OCR screenshots — allow_ocr=True (open verdict)",
    "D": "Non-financial documents — must fail cleanly",
    "E": "Hard rejections (photos / GIFs) — must fail cleanly",
    "F": "Auto-discovered — open verdict, crash-free required",
}


# ================================================================
# SECTION 4 — Auto-discovery  (Suite F)
# ================================================================

def _classify_discovered(fpath: Path) -> str:
    """
    Assign a sub-label for display inside Suite F.

    image + --ocr  → "image/OCR"
    image, no OCR  → "image/no-OCR"
    structured     → extension upper-case, e.g. "CSV", "PDF"
    """
    ext = fpath.suffix.lower()
    if ext in _IMAGE_EXTS:
        return "image/OCR" if RUN_OCR else "image/no-OCR"
    return ext.lstrip(".").upper()


def _build_suite_f(new_dir: Path) -> list[TestCase]:
    """
    Scan test_files/new/ for files not in REGISTRY and return a Suite F TestCase
    for each one.  Keeping new files in new/ means they never interfere with the
    A–E regression suite in known/.

    Auto-classification:
      image  → allow_ocr = RUN_OCR  (enabled when --ocr flag is present)
      other  → allow_ocr = False
    All Suite F cases have open_verdict=True (crash-free is the only requirement).
    """
    known: set[str] = {tc.file.lower() for tc in REGISTRY}
    cases: list[TestCase] = []
    if not new_dir.exists():
        return cases

    for fpath in sorted(new_dir.iterdir()):
        if not fpath.is_file():
            continue
        if fpath.name.lower() in _IGNORE_FILES:
            continue
        if fpath.suffix.lower() not in _SCANNABLE:
            continue
        if fpath.name.lower() in known:
            continue

        is_image = fpath.suffix.lower() in _IMAGE_EXTS
        cat      = _classify_discovered(fpath)
        cases.append(TestCase(
            label=f"{fpath.name}  [{cat}]",
            file=fpath.name,
            suite="F",
            exp_status=_ANY,
            exp_type=None,
            exp_min=0,
            note="auto-discovered",
            allow_ocr=is_image and RUN_OCR,
            open_verdict=True,
            base_dir=new_dir,   # Suite F files live in new/
        ))
    return cases


# ================================================================
# SECTION 5 — Assertion engine
# ================================================================

def _evaluate(tc: TestCase, result: PipelineResult) -> tuple[bool, list[tuple[str, bool]]]:
    """
    Check all expectations for a TestCase against a PipelineResult.

    Returns (overall_passed, list_of_(label, ok) tuples).
    Cases with open_verdict=True always return (True, [...]).
    """
    assertions: list[tuple[str, bool]] = []

    # Status
    s_ok = tc.status_ok(result.status)
    exp_s = (
        tc.exp_status if isinstance(tc.exp_status, str)
        else "{" + ", ".join(sorted(tc.exp_status)) + "}"
    )
    assertions.append((f"status == {exp_s}", s_ok))

    # Input type (skip if exp_type is None)
    if tc.exp_type is not None:
        t_ok = result.input_type == tc.exp_type
        assertions.append((f"input_type == {tc.exp_type!r}", t_ok))

    # Minimum record count (skip if exp_min == 0)
    if tc.exp_min > 0:
        c_ok = result.record_count >= tc.exp_min
        assertions.append((f"record_count >= {tc.exp_min}", c_ok))

    overall = all(ok for _, ok in assertions)
    if tc.open_verdict:
        overall = True   # Suite C / F — never mark as failed
    return overall, assertions


# ================================================================
# SECTION 6 — Case detail printer
# ================================================================

# Core fields to show per DTO type (omit boilerplate zeros)
_DTO_FIELDS: dict[str, list[str]] = {
    "broker_holding":      ["symbol", "shares", "avg_cost_per_share", "total_cost", "side"],
    "transaction":         ["symbol", "transaction_move", "shares",
                            "price_per_share", "total_amount", "executed_at"],
    "constituent_holding": ["symbol", "weight", "price", "change"],
}

def _fmt_record(rec) -> dict:
    d    = rec.model_dump()
    keys = _DTO_FIELDS.get(d.get("dto_type", ""), list(d.keys()))
    return {k: d[k] for k in keys if k in d and d[k] not in (None, "", 0.0, 0)}


def _print_case_detail(tr: TestResult) -> None:
    """Print assertion results + records + warnings for one test."""
    if QUIET or tr.skipped:
        return

    result = tr.result

    # ── Assertions ────────────────────────────────────────────────
    for label, ok in tr.assertions:
        if ok:
            _ok(label)
        else:
            _fail(label)

    # Hard-rejection suites: show rejection reason then stop
    if tr.suite in ("D", "E"):
        if result and result.errors:
            _dim(f"rejection: {result.errors[0].message[:110]}")
        return

    # ── Latency + record count ────────────────────────────────────
    _info(f"latency       = {result.total_latency_ms:.0f} ms")
    _info(f"record_count  = {result.record_count}")
    if result.input_type:
        _info(f"input_type    = {result.input_type}")

    # ── Sample records ────────────────────────────────────────────
    if result.data:
        prefix = "OCR extracted" if tr.is_ocr else "sample records"
        _info(f"{prefix}:")
        n_show = 3 if tr.is_ocr else 2
        for rec in result.data[:n_show]:
            print(f"      {_fmt_record(rec)}")
    else:
        _info("sample records: none")

    # ── Warnings ─────────────────────────────────────────────────
    # Filter out routing noise (e.g. "input_type=holding (from classifier)")
    actionable = [
        w for w in result.warnings
        if not (w.startswith("input_type=") and len(w) < 70)
    ]
    if actionable:
        if VERBOSE:
            for w in actionable:
                _warn(w)
        else:
            # Pick the single most informative warning to surface
            priority = next(
                (w for w in actionable
                 if w.startswith("[MissingRequiredColumns]")
                    or ("skipped" in w.lower() and "row" in w.lower())),
                None,
            )
            if priority is None and any("[sanity]" in w for w in actionable):
                n = sum(1 for w in actionable if "[sanity]" in w)
                priority = (
                    f"OCR: {n} row(s) have numeric fields that may be misaligned"
                    " — verify manually."
                )
            if priority is None:
                priority = actionable[0]
            _warn(f"{len(actionable)} warning(s): {priority[:100]}")
            if len(actionable) > 1:
                _dim("  run --verbose to see all warnings")

    # ── Table summaries (verbose only) ────────────────────────────
    if VERBOSE and result.table_summaries:
        _info("table summaries:")
        for s in result.table_summaries:
            flag  = "SKIP" if s.skipped else "OK  "
            extra = []
            if s.parse_confidence != "high":
                extra.append(f"conf={s.parse_confidence}")
            if s.suspicious_rows:
                extra.append(f"suspicious={s.suspicious_rows}")
            meta = ("  [" + "  ".join(extra) + "]") if extra else ""
            print(f"      [{flag}] table {s.table_index}: "
                  f"{s.input_type}, {s.row_count} rows → "
                  f"{s.record_count} records{meta}")

    # ── Errors ────────────────────────────────────────────────────
    if result.errors:
        if tr.suite == "C" or tr.tc.open_verdict:
            _info(f"pipeline note: {result.errors[0].message[:110]}")
        else:
            for e in result.errors:
                _fail(f"[{e.stage}] {e.error_type}: {e.message[:100]}")


# ================================================================
# SECTION 7 — Summary report
# ================================================================

def _print_summary(
    all_results:  list[TestResult],
    suite_stats:  dict[str, SuiteStats],
    run_ocr:      bool,
) -> None:
    run     = [r for r in all_results if not r.skipped]
    passed  = [r for r in run if r.passed]
    failed  = [r for r in run if not r.passed]
    skipped = [r for r in all_results if r.skipped]

    ocr_run = [r for r in run if r.is_ocr]

    W66 = "═" * 66

    # ── Header ────────────────────────────────────────────────────
    print(f"\n{W66}")
    print(f"{B}  SUMMARY REPORT{W}")
    print(W66)

    # Overall
    colour = G if not failed else R
    print(f"\n  {B}Overall{W}  "
          f"{colour}{len(passed)}/{len(run)} passed{W}"
          + (f"   {R}{len(failed)} FAILED{W}" if failed else "")
          + (f"   {Y}{len(skipped)} skipped{W}" if skipped else ""))

    # ── Per-suite table ───────────────────────────────────────────
    print(f"\n  {'STE':<4}  {'Description':<42}  {'Result':>14}  {'Avg ms':>7}")
    print(f"  {'─'*4}  {'─'*42}  {'─'*14}  {'─'*7}")

    for sid in ["A", "B", "C", "D", "E", "F"]:
        st = suite_stats.get(sid)
        if st is None or st.total == 0:
            continue

        if sid == "C" and not run_ocr:
            res_str = f"{Y}skipped (--ocr){W}"
            lat_str = "     —"
        else:
            if st.failed == 0:
                res_str = f"{G}{st.passed}/{st.run}{W}"
            else:
                res_str = f"{R}{st.passed}/{st.run}  ✗{st.failed}{W}"
            lat_str = f"{st.avg_ms:>6.0f}" if st.avg_ms else "     —"

        label = _SUITE_LABELS[sid][:42]
        # Pad result string accounting for invisible ANSI codes
        print(f"  {sid:<4}  {label:<42}  {res_str:<22}  {lat_str}")

    # ── OCR breakdown ─────────────────────────────────────────────
    if ocr_run:
        print(f"\n  {'─'*66}")
        print(f"  {B}OCR Breakdown{W}"
              + (f"  {D}(--ocr flag was active){W}" if run_ocr else ""))
        print(f"  {'─'*66}")

        with_data    = [r for r in ocr_run if r.record_count > 0]
        success_rate = len(with_data) / len(ocr_run) * 100 if ocr_run else 0
        avg_rec      = (sum(r.record_count for r in with_data) / len(with_data)
                        if with_data else 0)
        avg_lat      = (sum(r.latency_ms for r in ocr_run) / len(ocr_run)
                        if ocr_run else 0)

        print(f"\n  Images attempted      {len(ocr_run)}")
        print(f"  Produced ≥1 record    {len(with_data)}/{len(ocr_run)}"
              f"   ({success_rate:.0f}% success rate)")
        if with_data:
            print(f"  Avg records / image   {avg_rec:.1f}  (when successful)")
        print(f"  Avg pipeline latency  {avg_lat:.0f} ms")

        # Per-image table
        col = f"  {'File':<28}  {'Status':<26}  {'Recs':>4}  {'ms':>6}  {'⚠ sanity':>8}"
        print(f"\n{col}")
        print(f"  {'─'*28}  {'─'*26}  {'─'*4}  {'─'*6}  {'─'*8}")
        for r in ocr_run:
            if not r.result:
                continue
            s = r.result.status
            sc = (G if s == "success"
                  else Y if s in ("partial", "low_confidence_partial")
                  else R)
            lat_s  = f"{r.latency_ms/1000:.1f}s"
            sanity = f"{r.sanity_count:>8}" if r.sanity_count else "       —"
            fname  = r.tc.file[:28]
            print(f"  {fname:<28}  {sc}{s:<16}{W}  "
                  f"{r.record_count:>4}  {lat_s:>6}  {sanity}")

    # ── Failed cases detail ───────────────────────────────────────
    if failed:
        print(f"\n  {R}{'─'*66}{W}")
        print(f"  {B}{R}Failed cases{W}")
        for r in failed:
            print(f"  {R}✗{W}  [{r.suite}] {r.label}")
            if r.result and r.result.errors:
                for e in r.result.errors[:2]:
                    print(f"       {D}{e.stage}: {e.message[:80]}{W}")
            for assertion_label, ok in r.assertions:
                if not ok:
                    print(f"       {D}✗ {assertion_label}{W}")

    # ── Footer ────────────────────────────────────────────────────
    verdict = f"{G}ALL PASSED{W}" if not failed else f"{R}FAILURES DETECTED{W}"
    print(f"\n  {verdict}")
    print(f"{W66}\n")


# ================================================================
# SECTION 8 — Test runner
# ================================================================

def run_all(test_dir: Path, run_ocr: bool = False) -> bool:
    """
    Run all suites.

    Directory layout:
      test_files/known/  — regression fixtures for Suites A–E
      test_files/new/    — new files scanned automatically into Suite F

    Each TestCase resolves its file against:
      tc.base_dir  if set (Suite F cases point to new/)
      known_dir    otherwise (Suites A–E)
    """
    known_dir = test_dir / "known"
    new_dir   = test_dir / "new"

    logger = logging.getLogger("pipeline.test")
    logger.setLevel(logging.WARNING)

    if not run_ocr:
        print(f"\n{Y}  Suite C (OCR screenshots) skipped — pass --ocr to enable{W}")

    # Build complete case list: registry + auto-discovered Suite F
    all_cases: list[TestCase] = list(REGISTRY) + _build_suite_f(new_dir)

    # Initialise per-suite stats
    suite_stats: dict[str, SuiteStats] = {
        sid: SuiteStats(sid, _SUITE_LABELS.get(sid, sid))
        for sid in ["A", "B", "C", "D", "E", "F"]
    }

    all_results: list[TestResult] = []

    # Group cases by suite for block printing
    from collections import defaultdict
    grouped: dict[str, list[TestCase]] = defaultdict(list)
    for tc in all_cases:
        grouped[tc.suite].append(tc)

    for suite_id in ["A", "B", "C", "D", "E", "F"]:
        cases = grouped.get(suite_id, [])
        if not cases:
            continue

        st = suite_stats[suite_id]
        st.total = len(cases)

        # Suite header
        if not QUIET:
            print(f"\n{'═'*62}")
            print(f"{B}Suite {suite_id}: {_SUITE_LABELS[suite_id]}{W}")
            if suite_id == "F":
                print(f"{D}  {len(cases)} file(s) auto-discovered in test_files/new/{W}")
            print(f"{'═'*62}")

        for tc in cases:
            # Suite F TestCases carry their own base_dir (new/);
            # all other cases resolve against known/.
            fpath = (tc.base_dir / tc.file) if tc.base_dir else (known_dir / tc.file)

            # ── Determine if we should skip ───────────────────────
            skip_reason = ""
            if tc.suite == "C" and not run_ocr:
                skip_reason = "add --ocr to run"
            elif not fpath.exists():
                skip_reason = f"file not found: {tc.file}"

            if skip_reason:
                st.skipped += 1
                tr = TestResult(tc=tc, result=None, passed=True,
                                skipped=True, skip_reason=skip_reason)
                all_results.append(tr)
                if not QUIET:
                    print(f"\n  {Y}SKIP{W}  {tc.label}  ({skip_reason})")
                continue

            st.run += 1

            # ── Case header ───────────────────────────────────────
            if not QUIET:
                print(f"\n{'─'*62}")
                print(f"{B}[{suite_id}] {tc.label}{W}  ({tc.file})")
                if tc.note and VERBOSE:
                    print(f"{D}  {tc.note}{W}")
                print(f"{'─'*62}")
            else:
                label_trunc = tc.label[:44]
                print(f"  [{suite_id}] {label_trunc:<44}", end=" ", flush=True)

            # ── Execute pipeline ──────────────────────────────────
            t0 = time.monotonic()
            result = run_pipeline(str(fpath), logger=logger, allow_ocr=tc.allow_ocr)
            wall_ms = (time.monotonic() - t0) * 1000

            # ── Evaluate expectations ─────────────────────────────
            passed, assertions = _evaluate(tc, result)

            tr = TestResult(
                tc=tc, result=result, passed=passed,
                wall_ms=wall_ms, assertions=assertions,
            )
            all_results.append(tr)

            # ── Update stats ──────────────────────────────────────
            if passed:
                st.passed += 1
            else:
                st.failed += 1
            st.latencies.append(result.total_latency_ms)

            # ── Print detail / quiet line ─────────────────────────
            if not QUIET:
                _print_case_detail(tr)
            else:
                status_tag = f"{G}OK  {W}" if passed else f"{R}FAIL{W}"
                print(f"{status_tag}  "
                      f"{result.record_count:>4} recs  "
                      f"{result.total_latency_ms:>6.0f} ms  "
                      f"[{result.status}]")

    _print_summary(all_results, suite_stats, run_ocr)

    return all(r.passed for r in all_results if not r.skipped)


# ================================================================
# SECTION 9 — Single-file ad-hoc mode
# ================================================================

def run_single(file_path: str, allow_ocr: bool = False) -> None:
    print(f"\n{B}{C}{'─'*62}{W}")
    print(f"{B}File: {file_path}{W}")
    if allow_ocr:
        print(f"{Y}  OCR mode: allow_ocr=True{W}")
    print(f"{C}{'─'*62}{W}")

    logger = logging.getLogger("pipeline.test")
    logger.setLevel(logging.WARNING)

    result = run_pipeline(file_path, logger=logger, allow_ocr=allow_ocr)
    tc = TestCase(
        label="ad-hoc", file=Path(file_path).name,
        suite="X", exp_status=_ANY,
        exp_type=None, exp_min=0, note="single file",
        allow_ocr=allow_ocr, open_verdict=True,
    )
    _, assertions = _evaluate(tc, result)
    tr = TestResult(tc=tc, result=result, passed=True, assertions=assertions)
    _print_case_detail(tr)


# ================================================================
# SECTION 10 — Entry point
# ================================================================

if __name__ == "__main__":
    test_dir = _here / "test_files"
    args     = [a for a in sys.argv[1:] if not a.startswith("--")]

    print(f"\n{B}Preparing test files (if missing)...{W}")
    generate_synthetic_fixtures(test_dir)   # creates known/ and new/ if absent

    if args:
        run_single(args[0], allow_ocr=RUN_OCR)
    else:
        success = run_all(test_dir, run_ocr=RUN_OCR)
        sys.exit(0 if success else 1)
