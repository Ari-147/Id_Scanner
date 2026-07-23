"""
FastAPI ID-card scanner backend.
- easyocr (CPU) OCR, in-memory (no disk I/O)                       -> ocr.py
- image preprocessing + confidence filtering + adaptive retry      -> ocr.py
- 3-tier dynamic heuristic parsing                                 -> parser.py
- pre-indexed rapidfuzz payer lookup (sub-ms), strict              -> payer_matching.py
- optional Claude field-refinement step (skips cleanly w/o a key)  -> llm_refine.py

This module is now only the FastAPI wiring + route handlers, which orchestrate
calls into the modules above.

Run: uvicorn app:app --host 127.0.0.1 --port 8080
"""
import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from config import DEFAULT_MIN_CONF
from ocr import run_adaptive_ocr
from parser import parse_fields
from payer_matching import match_payer, resolve_payer_candidate, PAYER_CHOICES
from llm_refine import needs_refinement, maybe_refine
from result_summary import build_result_summary
from fax_parser import parser_fax_with_regex
from fax_llm import parse_fax_with_llm, is_configured as fax_llm_configured
from pdf_converter import convert_pdf_to_images

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_FILE = Path(__file__).parent / "app.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),                     # console
        logging.FileHandler(LOG_FILE, encoding="utf-8")  # file
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="ID Card Scanner")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok", "payers_indexed": len(PAYER_CHOICES)}


@app.get("/log", response_class=PlainTextResponse)
def get_log():
    """Return the entire log file as plain text."""
    if LOG_FILE.exists():
        return LOG_FILE.read_text(encoding="utf-8")
    return "No log file yet."


@app.post("/scan")
async def scan(file: UploadFile = File(...), min_conf: float = DEFAULT_MIN_CONF):
    logger.info("Scan request received: file=%s, min_conf=%.2f", file.filename, min_conf)
    raw = await file.read()

    try:
        # Adaptive OCR (single or two-pass, decided internally) -> reading-order lines.
        detections = run_adaptive_ocr(raw, min_conf)
        logger.info("OCR completed: %d detections (before confidence filter)", len(detections))

        lines, dropped = [], []
        for d in detections:
            entry = {"text": d["text"], "conf": round(d["conf"], 2)}
            (lines if d["conf"] >= min_conf else dropped).append(entry)

        clean_lines = [d["text"] for d in lines]
        logger.info("Kept %d lines after confidence filter", len(clean_lines))

        parsed = parse_fields(clean_lines)
        logger.info("Parsed fields: %s", {k: v for k, v in parsed.items() if k != "extra_fields"})

        payer = match_payer(clean_lines)
        if payer:
            parsed["payer_id"] = payer["payer_id"]
            parsed["payer_name"] = payer["payer_name"]
            parsed["payer_match"] = payer
            logger.info("Payer matched: %s (ID %s, score %.1f)",
                        payer["payer_name"], payer["payer_id"], payer["score"])
        else:
            parsed["payer_match"] = None
            logger.info("No payer matched")

        # Optional LLM refinement
        if needs_refinement(parsed):
            parsed, refinement_status = maybe_refine(clean_lines, parsed)
            logger.info("LLM refinement status: %s", refinement_status)
        else:
            refinement_status = "skipped"
            logger.info("LLM refinement skipped (heuristic complete)")

        # Compiled result
        candidate = resolve_payer_candidate(clean_lines, strict=payer)
        result = build_result_summary(
            parsed, candidate, refinement_status, clean_lines, lines
        )
        logger.info("Scan completed successfully for %s", file.filename)

        return JSONResponse(
            {
                "raw_lines": clean_lines,
                "fields": parsed,
                "ocr_confidence": lines,
                "dropped_low_confidence": dropped,
                "refinement": refinement_status,
                "result": result,
            }
        )

    except Exception as e:
        logger.error("Scan failed for %s: %s", file.filename, str(e), exc_info=True)
        raise

# @app.on_event("shutdown")
# def shutdown_cleanup():
#     if os.path.exists(LOG_FILE):
#         try:
#             os.remove(LOG_FILE)
#             print("Session Ended/ Removed log File")
#         except Exception as e:
#                 print(f"Error removing log file: {e}")

# ---------------------------------------------------------------------------
# Fax parsing
#
# Two endpoints, one shared OCR front-end. Both accept an image OR a PDF and
# return the SAME response shape so the frontend can render either identically:
#   { "method": "regex" | "claude", "fields": {...}, "ocr_confidence": [...] }
# ---------------------------------------------------------------------------
def _ocr_fax(raw: bytes, content_type: str | None) -> list[dict]:
    """Run OCR over a fax upload (PDF -> per-page images, else a single image)."""
    if content_type == "application/pdf":
        detections = []
        for image in convert_pdf_to_images(raw):
            detections.extend(run_adaptive_ocr(image, min_conf=DEFAULT_MIN_CONF))
        return detections
    return run_adaptive_ocr(raw, min_conf=DEFAULT_MIN_CONF)


@app.post("/extract-data-v1")
async def extract_data_v1(file: UploadFile = File(...)):
    """Regex-based fax parsing (zero-cost baseline)."""
    logger.info("Fax extract (regex) request: file=%s", file.filename)
    raw = await file.read()
    try:
        detections = _ocr_fax(raw, file.content_type)
        lines = [d["text"] for d in detections]
        fields = parser_fax_with_regex(lines)
        logger.info("Fax regex parse done: %d fields populated",
                    sum(1 for v in fields.values() if v))
        return JSONResponse({
            "method": "regex",
            "fields": fields,
            "ocr_confidence": [round(d["conf"], 2) for d in detections],
        })
    except Exception as e:
        logger.error("Fax regex extract failed for %s: %s", file.filename, e,
                     exc_info=True)
        raise


@app.post("/extract-data-v2")
async def extract_data_v2(file: UploadFile = File(...)):
    """Claude-based fax parsing (token-optimized). Requires ANTHROPIC_API_KEY."""
    logger.info("Fax extract (claude) request: file=%s", file.filename)
    if not fax_llm_configured():
        raise HTTPException(
            503,
            "Claude fax parsing is not configured. Set ANTHROPIC_API_KEY (and "
            "install the `anthropic` package), or use the regex parser instead.",
        )
    raw = await file.read()
    try:
        detections = _ocr_fax(raw, file.content_type)
        lines = [d["text"] for d in detections]
        fields = parse_fax_with_llm(lines)
        logger.info("Fax claude parse done: %d fields populated",
                    sum(1 for v in fields.values() if v))
        return JSONResponse({
            "method": "claude",
            "fields": fields,
            "ocr_confidence": [round(d["conf"], 2) for d in detections],
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Fax claude extract failed for %s: %s", file.filename, e,
                     exc_info=True)
        raise HTTPException(502, f"Claude fax parsing failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)