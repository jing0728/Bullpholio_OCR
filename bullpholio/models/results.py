"""
models/results.py
-----------------
Pipeline result envelope and supporting models.
"""

from typing import TYPE_CHECKING, Optional, Union
from pydantic import BaseModel, Field

from bullpholio.models.dtos import BrokerHoldingDTO, ConstituentHoldingDTO, TransactionDTO

# Backwards-compatible alias
HoldingDTO = BrokerHoldingDTO

if TYPE_CHECKING:
    from bullpholio.core.classifier import ClassificationResult


class StageError(BaseModel):
    stage: str
    error_type: str
    message: str


class TableParseSummary(BaseModel):
    """
    Per-table metadata attached to PipelineResult.

    parse_confidence:
      "high"   — all rows passed; numeric cross-checks OK
      "medium" — some rows skipped, or minor sanity warnings
      "low"    — majority of rows skipped, or multiple sanity failures

    parse_status:
      "success"        — record_count > 0, no suspicious rows
      "partial_success"— record_count > 0, but some rows skipped or low-confidence values
      "failed"         — record_count = 0
    """
    table_index: int       # 1-based index matching "Table N" in logs
    input_type: str        # "holding" | "transaction" | "constituent_holding"
    row_count: int         # rows in the raw DataFrame
    record_count: int      # DTOs successfully produced
    skipped: bool = False  # True when required columns were absent

    # Confidence + partial-success tracking
    parse_status: str = "success"          # "success" | "partial_success" | "failed"
    parse_confidence: str = "high"         # "high" | "medium" | "low"
    suspicious_rows: int = 0               # rows flagged by sanity check
    confidence_notes: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """
    Unified result envelope — always returned, never raises.

    status:
      "success"               — all records extracted cleanly; no sanity warnings
      "partial"               — records extracted; some tables failed or had
                                skipped rows, but numeric values look plausible
      "low_confidence_partial"— records extracted; but the majority of rows
                                triggered sanity warnings (e.g. OCR digit misreads,
                                implausible avg_cost values).  Downstream consumers
                                should surface a manual-review prompt to the user.
      "failed"                — no records could be extracted
    """
    status: str
    file_path: str
    file_extension: str
    file_category: str
    input_type: Optional[str] = None
    classification: Optional[object] = None
    total_latency_ms: float
    stage_latency_ms: dict[str, float]
    record_count: int = 0
    data: list[Union[BrokerHoldingDTO, ConstituentHoldingDTO, TransactionDTO]] = Field(default_factory=list)
    errors: list[StageError] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    table_summaries: list[TableParseSummary] = Field(default_factory=list)
