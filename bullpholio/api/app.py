"""
api/app.py
----------
FastAPI application exposing the Bullpholio pipeline as an HTTP API.

Endpoints
─────────
  GET  /health           liveness probe — returns {"status": "ok", "version": "..."}
  POST /v1/parse         upload a file; run the pipeline; return structured records

POST /v1/parse
──────────────
  Content-Type:  multipart/form-data
  Field:         file  (the document to parse)
  Query params:
    allow_ocr    bool  default=False
                       Enable EasyOCR for scanned PDFs and broker screenshots.
                       Adds ~5-20 s of latency for images.
    warnings     bool  default=True
                       Include the warnings array in the response.
                       Set to False to slim down the payload.

  Response (always HTTP 200):
    ParseResponse JSON — see schemas.py for the full shape.
    Branch on the `status` field:
      "success"               → display records immediately
      "partial"               → records exist; surface a "some rows skipped" note
      "low_confidence_partial"→ records exist; surface a manual-review prompt
      "failed"                → no records; show errors[0].message to the user

  HTTP error codes:
    400  missing file, empty file, or unsupported MIME / extension
    413  file exceeds MAX_FILE_SIZE_MB
    500  unexpected internal error (bug; should not happen in production)

Configuration (environment variables)
──────────────────────────────────────
  MAX_FILE_SIZE_MB   int     default=50    hard upload size limit
  LOG_LEVEL          str     default=INFO  Python logging level
  CORS_ORIGINS       str     default="*"  comma-separated allowed origins
                             Set to your Go backend's origin in production,
                             e.g. "http://localhost:8080,https://app.example.com"
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bullpholio.pipeline import run_pipeline
from bullpholio.config.file_types import get_file_category

from api.schemas import (
    ClassificationSummary,
    ErrorResponse,
    HealthResponse,
    ParseResponse,
    StageErrorSchema,
    TableSummarySchema,
    BrokerHoldingRecord,
    ConstituentHoldingRecord,
    TransactionRecord,
)

# ── Configuration ─────────────────────────────────────────────────────────────

VERSION = "1.0.0"

MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_MB", "50")) * 1024 * 1024

_raw_origins = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = (
    ["*"] if _raw_origins.strip() == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("bullpholio.api")

# Pipeline logger: WARNING in production so per-row details don't flood stdout
pipeline_logger = logging.getLogger("bullpholio.pipeline")
pipeline_logger.setLevel(logging.WARNING)

# Accepted MIME types → extension mapping.
# Used as a fast pre-check before writing the temp file.
_MIME_TO_EXT: dict[str, str] = {
    "text/csv":                                                ".csv",
    "application/csv":                                         ".csv",
    "application/vnd.ms-excel":                               ".xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/pdf":                                         ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "image/png":                                               ".png",
    "image/jpeg":                                              ".jpg",
    "image/bmp":                                               ".bmp",
    "image/gif":                                               ".gif",
    "image/tiff":                                              ".tiff",
    "image/webp":                                              ".webp",
    # Browsers sometimes send these for CSV / Excel
    "application/octet-stream":                                None,   # fallback: use filename ext
    "text/plain":                                              ".csv",
}

_SUPPORTED_EXTENSIONS: set[str] = {
    ".csv", ".xlsx", ".pdf", ".docx",
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp",
}


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info(f"Bullpholio API v{VERSION} starting up")
    logger.info(f"Max upload size: {MAX_FILE_SIZE_BYTES // (1024*1024)} MB")
    logger.info(f"CORS origins: {CORS_ORIGINS}")
    yield
    logger.info("Bullpholio API shutting down")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Bullpholio Document Parser",
    description=(
        "Extracts structured portfolio holdings and transaction records "
        "from uploaded financial documents (PDF, Excel, CSV, Word, images)."
    ),
    version=VERSION,
    lifespan=_lifespan,
    # Disable the default /docs and /redoc in production if desired
    # docs_url=None, redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request ID middleware ─────────────────────────────────────────────────────

@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    """
    Attach a UUID request ID to every request.
    Returned in the X-Request-ID response header so the Go backend
    can correlate logs between services.
    """
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=_status_to_code(exc.status_code),
            message=exc.detail,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "unknown")
    logger.exception(f"Unhandled error  request_id={req_id}: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred. Please try again or contact support.",
            detail=req_id,
        ).model_dump(),
    )


def _status_to_code(status: int) -> str:
    return {
        400: "bad_request",
        413: "file_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
        500: "internal_error",
    }.get(status, "error")


# ── Helper: resolve file extension ───────────────────────────────────────────

def _resolve_extension(upload: UploadFile) -> str:
    """
    Determine the canonical file extension from the upload.

    Priority:
      1. Original filename extension (most reliable)
      2. Content-Type header (fallback for clients that strip extensions)

    Raises HTTPException(400) if neither resolves to a supported extension.
    Raises HTTPException(415) if the type is recognised but unsupported.
    """
    # 1. Try filename
    if upload.filename:
        ext = Path(upload.filename).suffix.lower()
        if ext in _SUPPORTED_EXTENSIONS:
            return ext
        if ext:  # has an extension but it's not supported
            raise HTTPException(
                status_code=415,
                detail=(
                    f"File type '{ext}' is not supported. "
                    f"Accepted: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
                ),
            )

    # 2. Try Content-Type
    ct = (upload.content_type or "").split(";")[0].strip().lower()
    if ct in _MIME_TO_EXT:
        ext = _MIME_TO_EXT[ct]
        if ext and ext in _SUPPORTED_EXTENSIONS:
            return ext

    raise HTTPException(
        status_code=400,
        detail=(
            "Could not determine file type from filename or Content-Type. "
            "Please upload a PDF, Excel (.xlsx), CSV, Word (.docx), or image file."
        ),
    )


# ── Helper: serialise PipelineResult → ParseResponse ─────────────────────────

def _serialise_record(dto) -> dict:
    """
    Convert a DTO (BrokerHoldingDTO / ConstituentHoldingDTO / TransactionDTO)
    into a plain dict matching the corresponding *Record schema.
    Uses model_dump() which is guaranteed JSON-serialisable.
    """
    return dto.model_dump()


def _serialise_classification(classification) -> Optional[ClassificationSummary]:
    """
    Convert the internal ClassificationResult dataclass to a plain schema.
    Handles both dataclass instances and None gracefully.
    """
    if classification is None:
        return None
    try:
        return ClassificationSummary(
            doc_type=classification.doc_type,
            confidence=classification.confidence,
            reason=classification.reason,
        )
    except Exception:
        return None


def _build_response(result, include_warnings: bool) -> ParseResponse:
    """Assemble a ParseResponse from a PipelineResult."""
    records = []
    for dto in result.data:
        raw = dto.model_dump()
        dto_type = raw.get("dto_type", "")
        if dto_type == "broker_holding":
            records.append(BrokerHoldingRecord(**raw))
        elif dto_type == "constituent_holding":
            records.append(ConstituentHoldingRecord(**raw))
        elif dto_type == "transaction":
            records.append(TransactionRecord(**raw))
        else:
            # Unknown DTO type — pass through as broker_holding best-effort
            records.append(BrokerHoldingRecord(**raw))

    return ParseResponse(
        status=result.status,
        input_type=result.input_type,
        record_count=result.record_count,
        total_latency_ms=result.total_latency_ms,
        stage_latency_ms=result.stage_latency_ms,
        classification=_serialise_classification(result.classification),
        data=records,
        table_summaries=[
            TableSummarySchema(**s.model_dump())
            for s in result.table_summaries
        ],
        errors=[
            StageErrorSchema(**e.model_dump())
            for e in result.errors
        ],
        warnings=result.warnings if include_warnings else [],
    )


# ================================================================
# Routes
# ================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["ops"],
)
async def health():
    """
    Returns `{"status": "ok", "version": "..."}`.
    Use as a Kubernetes/Docker liveness or readiness probe.
    """
    return HealthResponse(status="ok", version=VERSION)


@app.post(
    "/v1/parse",
    response_model=ParseResponse,
    summary="Parse a financial document",
    tags=["pipeline"],
    responses={
        200: {"description": "Parsed successfully (check `status` field for outcome)"},
        400: {"model": ErrorResponse, "description": "Bad request (missing/empty file, unknown type)"},
        413: {"model": ErrorResponse, "description": "File exceeds size limit"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def parse_document(
    file: UploadFile = File(..., description="Financial document to parse"),
    allow_ocr: bool = Query(
        default=False,
        description=(
            "Enable OCR for scanned PDFs and broker screenshots. "
            "Adds 5–20 s of latency for image files."
        ),
    ),
    include_warnings: bool = Query(
        default=True,
        alias="warnings",
        description="Include the warnings array in the response. Set to false to slim the payload.",
    ),
    request: Request = None,
):
    """
    Upload a financial document and receive structured portfolio records.

    **Supported formats**: PDF, Excel (.xlsx), CSV, Word (.docx),
    PNG, JPG, BMP, GIF, TIFF, WebP

    **Response `status` values**:
    - `success` — all records clean; display immediately
    - `partial` — records extracted; some rows were skipped
    - `low_confidence_partial` — records extracted but majority flagged by
      sanity checks; surface a manual-review prompt to the user
    - `failed` — no records; inspect `errors[0].message` for the reason

    **`data` array** contains heterogeneous records; use `dto_type` to
    discriminate:
    - `broker_holding` — standard broker account view (Fidelity / Schwab style)
    - `constituent_holding` — index / ETF constituent view (weight / price)
    - `transaction` — trade history entry
    """
    req_id = getattr(request.state, "request_id", "?") if request else "?"

    # ── Validate: file present ────────────────────────────────────────────────
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    # ── Validate: file extension / MIME ──────────────────────────────────────
    ext = _resolve_extension(file)

    # ── Read and validate size ────────────────────────────────────────────────
    # Read in one shot — for large files (PDF / Excel) streaming would be
    # better, but the pipeline itself needs a real path on disk anyway.
    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File size {len(content) // (1024*1024)} MB exceeds the {max_mb} MB limit.",
        )

    logger.info(
        f"request_id={req_id}  file={file.filename!r}  "
        f"size={len(content)/1024:.1f}KB  ext={ext}  allow_ocr={allow_ocr}"
    )

    # ── Write to temp file ────────────────────────────────────────────────────
    # The pipeline expects a real filesystem path; temp file is cleaned up in
    # the finally block regardless of success or error.
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=ext,
            delete=False,
            prefix="bullpholio_",
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # ── Run pipeline (CPU-bound; runs synchronously in the event loop) ────
        # For a high-throughput deployment, wrap in asyncio.to_thread():
        #   result = await asyncio.to_thread(run_pipeline, str(tmp_path), ...)
        # For a typical broker-file use case with <10 concurrent requests,
        # synchronous is fine and easier to reason about.
        result = run_pipeline(
            str(tmp_path),
            logger=pipeline_logger,
            allow_ocr=allow_ocr,
        )

        response = _build_response(result, include_warnings=include_warnings)

        logger.info(
            f"request_id={req_id}  status={result.status}  "
            f"records={result.record_count}  latency={result.total_latency_ms:.0f}ms"
        )
        return response

    finally:
        # Always clean up the temp file
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# ── Dev server entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=_LOG_LEVEL.lower(),
    )
