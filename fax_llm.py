"""
Claude-powered medical-referral-fax parser (token-optimized).

Targets the exact same flat field set as fax_parser.py so `/extract-data-v1`
(regex) and `/extract-data-v2` (Claude) return an identical shape.

Token discipline:
  - Only the OCR text lines are sent — never the image, never the PDF.
  - The schema is sent as a terse comma-separated key list (built from
    FAX_FIELDS), not a verbose JSON template, keeping the system prompt small.
  - A small/fast model (config.LLM_MODEL, i.e. Haiku) with a bounded
    max_tokens is used; JSON is the only thing we ask for back.
"""
import json
import logging
import re

import config
from fax_parser import FAX_FIELDS, FAX_BOOLEAN_FIELDS, _blank_result

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None

log = logging.getLogger("fax_llm")

_STRING_FIELDS = [f for f in FAX_FIELDS if f not in FAX_BOOLEAN_FIELDS]

SYSTEM_PROMPT = (
    "You extract structured data from the OCR text of a medical referral fax. "
    "Return ONLY one JSON object, no markdown or commentary.\n"
    "String fields (use null when absent), comma-separated:\n"
    + ", ".join(_STRING_FIELDS) + "\n"
    "Boolean fields (true only if that order type is clearly present, else false):\n"
    + ", ".join(sorted(FAX_BOOLEAN_FIELDS)) + "\n"
    "Copy values verbatim from the text; never invent or reformat data that is "
    "not present."
)


def is_configured() -> bool:
    """True when a Claude call is actually possible."""
    return bool(config.ANTHROPIC_API_KEY) and Anthropic is not None


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _coerce(llm_fields: dict) -> dict:
    """Merge the model output onto a stable blank result, keeping only known
    keys and coercing booleans."""
    result = _blank_result()
    for field in FAX_FIELDS:
        if field not in llm_fields:
            continue
        val = llm_fields[field]
        if field in FAX_BOOLEAN_FIELDS:
            result[field] = bool(val)
        elif val not in (None, "", []):
            result[field] = str(val).strip()
    return result


def parse_fax_with_llm(ocr_lines: list[str]) -> dict:
    """Extract the canonical fax field set from OCR lines using Claude.

    Raises RuntimeError if the LLM is not configured or the call fails — the
    `/extract-data-v2` endpoint's whole purpose is the LLM, so failures are
    surfaced rather than silently falling back.
    """
    if not is_configured():
        raise RuntimeError(
            "Claude is not configured (set ANTHROPIC_API_KEY and install "
            "the `anthropic` package)."
        )

    # Drop blank lines to trim input tokens.
    text = "\n".join(l for l in ocr_lines if l and l.strip())

    client = Anthropic()
    resp = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=config.FAX_LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    parsed = json.loads(_strip_fences(raw))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response was not a JSON object")
    return _coerce(parsed)
