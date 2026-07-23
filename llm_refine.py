"""
Optional Claude-powered refinement of the heuristic parse.

This is a *text-sorting* task, never an OCR task: after the local heuristic
parser runs, we optionally ask a small, fast Claude model to double-check the
field assignment using ONLY the handful of OCR text lines already extracted —
never the image, never the payer database, never per-detection metadata.

Hard guarantees (see maybe_refine):
  - No ANTHROPIC_API_KEY in the environment  -> clean no-op ("not_configured").
  - Any failure (no network, expired key, timeout, malformed JSON, exception)
    -> caught, logged as a warning, heuristic result returned unchanged
    ("failed"). An LLM failure NEVER breaks or degrades a scan.

Token discipline (see needs_refinement): the call is skipped entirely on scans
that already look complete, so a clean card costs zero tokens — exactly like
today's app with no key configured.
"""
import json
import logging
import re

import config

# Import defensively: if the `anthropic` package isn't installed, the whole
# module still imports and refinement degrades to a clean no-op.
try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - exercised only without the package
    Anthropic = None

log = logging.getLogger("llm_refine")

# The exact field names parse_fields() produces. The LLM must return ONLY these
# keys; we merge them back field-by-field. Payer fields (payer_id, payer_name,
# payer_match) are intentionally excluded — they're added after parsing and are
# not the LLM's concern.
SCHEMA_FIELDS = (
    "name",
    "id_number",
    "dob",
    "expiry",
    "issue_date",
    "sex",
    "address",
    "organization",
    "group_number",
    "dates",
    "extra_fields",
)

_SCHEMA_INSTRUCTION = (
    "Return ONLY a single JSON object with EXACTLY these keys: "
    "name, id_number, dob, expiry, issue_date, sex, address, organization, "
    "group_number, dates, extra_fields. Use null for a missing scalar field, "
    "[] for dates, and {} for extra_fields. No markdown, no code fences, no "
    "commentary, no extra keys."
)

# Text-only refinement (no image available).
SYSTEM_PROMPT = (
    "You are a data-cleaning assistant for an ID / insurance card scanner. "
    "You are given the OCR text lines from a single card (`raw_lines`) and a "
    "heuristic parser's current best-guess `fields`. Your job is to correct or "
    "fill the field assignment using ONLY the literal text present in "
    "`raw_lines`. Never invent, translate, or reformat data that is not present "
    "in the provided lines.\n\n" + _SCHEMA_INSTRUCTION
)

# Refinement WITH the card image attached — the OCR was weak (many empty
# fields), so let the model read the image as the source of truth and use the
# OCR lines / current parse only as hints.
IMAGE_SYSTEM_PROMPT = (
    "You are a data-cleaning assistant for an ID / insurance card scanner. "
    "You are given the IMAGE of a single card, plus the OCR text lines "
    "(`raw_lines`) and a heuristic parser's current best-guess `fields`. The OCR "
    "was unreliable, so read the values primarily FROM THE IMAGE, using the OCR "
    "text and current fields only as hints. Copy values exactly as printed; "
    "never invent or translate data that is not on the card.\n\n"
    + _SCHEMA_INSTRUCTION
)


def needs_refinement(parsed_fields: dict) -> bool:
    """
    Trigger condition for the (paid) LLM call — tune it here, in one place.

    Only call the model when the heuristic parse looks incomplete or ambiguous:
      - `name` is missing, OR
      - `id_number` is missing, OR
      - more than REFINE_MAX_EXTRA_FIELDS unclassified lines are left over.

    Otherwise the heuristic result is returned as-is (no API call), exactly
    like today.
    """
    if not parsed_fields.get("name"):
        return True
    if not parsed_fields.get("id_number"):
        return True
    extra = parsed_fields.get("extra_fields") or {}
    if len(extra) > config.REFINE_MAX_EXTRA_FIELDS:
        return True
    return False


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` fences if the model added them despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _merge(parsed: dict, llm_fields: dict) -> dict:
    """
    Field-by-field merge: prefer an LLM value ONLY where it actually filled
    something in. A sparse/empty LLM response must never blank out a field the
    heuristic parser already got right. Payer fields and anything outside
    SCHEMA_FIELDS are preserved untouched.
    """
    merged = dict(parsed)
    for key in SCHEMA_FIELDS:
        val = llm_fields.get(key)
        if val:  # non-empty string / non-empty list / non-empty dict / truthy
            merged[key] = val
    return merged


def maybe_refine(
    raw_lines: list[str], parsed: dict, image_bytes: bytes | None = None
) -> tuple[dict, str]:
    """
    Optionally refine `parsed` with a Claude call. Returns (fields, status),
    where status is one of:
      "not_configured" | "applied" | "applied_image" | "failed".

    Callers should only invoke this when needs_refinement(parsed) is True — the
    same threshold therefore gates the (heavier) image send. When `image_bytes`
    is supplied the card image is attached so the model can read the card
    directly (larger token budget); otherwise it's the original text-only call.
    """
    # --- Fallback 1: not configured -> clean no-op, nothing scary logged ---
    if not config.ANTHROPIC_API_KEY or Anthropic is None:
        return parsed, "not_configured"

    try:
        # The current best-guess fields (parser schema keys only — no payer DB).
        payload = {
            "raw_lines": raw_lines,
            "fields": {k: parsed.get(k) for k in SCHEMA_FIELDS if k in parsed},
        }
        payload_text = json.dumps(payload, ensure_ascii=False)

        use_image = image_bytes is not None
        if use_image:
            from llm_image import encode_image_block
            content = [encode_image_block(image_bytes), {"type": "text", "text": payload_text}]
            system = IMAGE_SYSTEM_PROMPT
            max_tokens = config.LLM_IMAGE_MAX_TOKENS
        else:
            content = payload_text
            system = SYSTEM_PROMPT
            max_tokens = config.LLM_MAX_TOKENS

        client = Anthropic()  # picks up ANTHROPIC_API_KEY from the environment
        resp = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        llm_fields = json.loads(_strip_fences(text))
        if not isinstance(llm_fields, dict):
            raise ValueError("LLM response was not a JSON object")
        return _merge(parsed, llm_fields), ("applied_image" if use_image else "applied")
    except Exception as e:  # noqa: BLE001 - any failure must fall back safely
        # Never let an LLM failure break or degrade a scan.
        log.warning("LLM refinement failed (%s); using heuristic result", e)
        return parsed, "failed"
