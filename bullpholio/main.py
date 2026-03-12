"""
main.py
-------
Command-line entry point for the Bullpholio document parser.

Usage:
    python main.py <file_path> [--ocr]

Examples:
    python main.py holdings.pdf
    python main.py scanned_statement.pdf --ocr
    python main.py transactions.csv
"""

import sys
from bullpholio.pipeline import run_pipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <file_path> [--ocr]")
        sys.exit(1)

    file_path = sys.argv[1]
    allow_ocr = "--ocr" in sys.argv

    result = run_pipeline(file_path, allow_ocr=allow_ocr)

    print("\n===== Pipeline Result =====")
    print(f"status       : {result.status}")
    print(f"input_type   : {result.input_type}")
    print(f"record_count : {result.record_count}")
    print(f"total_ms     : {result.total_latency_ms}")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"  {w}")

    if result.errors:
        print("\nErrors:")
        for err in result.errors:
            print(f"  [{err.stage}] {err.error_type}: {err.message}")

    if result.data:
        print("\nFirst 3 records:")
        for rec in result.data[:3]:
            print(" ", rec.model_dump())


if __name__ == "__main__":
    main()
