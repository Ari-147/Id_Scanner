"""
Claude-powered medical-referral-fax parser (vision).

`/extract-data-v2` sends the fax page image(s) DIRECTLY to Claude's multimodal
model — no OCR step. Seeing the real document lets the model read the form
layout and, crucially, the ORDER-TYPE CHECKBOXES (which text-only OCR can't
detect), while returning the exact same flat field set as fax_parser.py so v1
(regex) and v2 (Claude) share one response shape.

Token discipline:
  - Page images are downscaled (config.LLM_IMAGE_MAX_DIM) and JPEG-encoded
    before sending; input tokens scale with pixel area, so this is the main knob.
  - At most config.FAX_LLM_MAX_PAGES pages are sent.
  - The schema is a terse comma-separated key list, and output is capped at
    config.FAX_LLM_IMAGE_MAX_TOKENS.
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
    "You extract structured data from the image(s) of a medical referral fax. "
    "Read the form directly from the image. A field's label is next to (usually "
    "above) its value; lab results are in a two-column grid. Ignore the repeating "
    "page header (From/To/Fax/Page x of y/date-time) except to fill the fax "
    "metadata fields.\n"
    "Return ONLY one JSON object, no markdown or commentary.\n"
    "String fields (use null when absent), comma-separated:\n"
    + ", ".join(_STRING_FIELDS) + "\n"
    "Boolean order-type fields — set true ONLY when that option's checkbox is "
    "actually ticked/filled in the image, otherwise false:\n"
    + ", ".join(sorted(FAX_BOOLEAN_FIELDS)) + "\n"
    "Copy values verbatim as printed; never invent or reformat data that is not "
    "present."
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


def parse_fax_with_llm(images: list[bytes]) -> dict:
    """Extract the canonical fax field set from fax page image(s) using Claude.

    `images` is a list of raw image bytes (one per page). Raises RuntimeError if
    the LLM is not configured or the call fails — the `/extract-data-v2`
    endpoint's whole purpose is the LLM, so failures are surfaced rather than
    silently falling back.
    """
    if not is_configured():
        raise RuntimeError(
            "Claude is not configured (set ANTHROPIC_API_KEY and install "
            "the `anthropic` package)."
        )
    if not images:
        raise ValueError("No fax pages to parse")

    from llm_image import encode_image_block

    pages = images[: config.FAX_LLM_MAX_PAGES]
    content = [encode_image_block(img) for img in pages]
    content.append({"type": "text", "text": "Extract the fields as instructed."})

    client = Anthropic()
    resp = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=config.FAX_LLM_IMAGE_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    parsed = json.loads(_strip_fences(raw))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response was not a JSON object")
    return _coerce(parsed)
