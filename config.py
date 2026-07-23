"""
Central configuration: environment vars and tunables in one place.

Nothing here requires configuration for the app to keep working exactly as it
does today. The only *new* optional setting is ANTHROPIC_API_KEY — leave it
unset and the LLM refinement step (llm_refine.py) is a clean no-op.
"""
import os

# ---------------------------------------------------------------------------
# LLM refinement (optional) — see llm_refine.py
# ---------------------------------------------------------------------------
# Read straight from the environment, no hardcoded default. When this is None,
# llm_refine.maybe_refine() short-circuits to "not_configured" and the app
# behaves exactly as it does with no API key at all.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Fast/cheap model — the refinement step is a small text-sorting task, not OCR.
LLM_MODEL = "claude-haiku-4-5-20251001"

# Small: enough for one compact JSON object back.
LLM_MAX_TOKENS = 300

# A fax has many more fields than an ID card, so its JSON response needs a
# larger (but still bounded) token budget. See fax_llm.py.
FAX_LLM_MAX_TOKENS = 1024

# Refinement trigger: call the LLM only when the heuristic parse left more than
# this many unclassified lines in extra_fields (in addition to the missing
# name / id_number checks). Keeps token usage near-zero on clean scans.
REFINE_MAX_EXTRA_FIELDS = 2

# ---------------------------------------------------------------------------
# OCR tunables
# ---------------------------------------------------------------------------
# Long-side px; small phone captures are the #1 cause of bad OCR, so we upscale
# much more aggressively than a naive "just big enough" target.
UPSCALE_TARGET = 2200

# Below this many *confident* lines, a capture is almost certainly a poor photo
# (blur / low light / too far away) rather than a clean card, so it's worth
# paying for a second OCR pass with a different preprocessing variant. A clean,
# well-lit card typically yields 8+ confident lines; cards with few printed
# fields still tend to clear this easily, so 6 is a safe floor that mostly only
# trips for genuinely hard captures.
MIN_CONFIDENT_LINES_FOR_SINGLE_PASS = 6

# ---------------------------------------------------------------------------
# Scan defaults
# ---------------------------------------------------------------------------
# OCR confidence threshold below which detections are dropped from parsing.
DEFAULT_MIN_CONF = 0.35
