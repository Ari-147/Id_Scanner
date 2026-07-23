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

# Fast/cheap model — also multimodal, so it can read a card/fax image directly.
LLM_MODEL = "claude-haiku-4-5-20251001"

# --- Output token budgets (kept intentionally small = token-friendly) -------
# Different limits for the three call shapes; the image/fax calls get a bit more
# headroom because vision answers tend to be slightly longer.
LLM_MAX_TOKENS = 300              # ID refine, OCR text only
LLM_IMAGE_MAX_TOKENS = 512        # ID refine, with the card image attached
FAX_LLM_MAX_TOKENS = 1024         # fax from OCR text (fallback path)
FAX_LLM_IMAGE_MAX_TOKENS = 1536   # fax sent as image(s) to Claude vision

# --- Input token control for vision calls -----------------------------------
# Downscale an image's long edge to at most this many px before sending. Image
# input tokens scale with pixel area (~w*h/750), so this is the main knob for
# keeping vision calls cheap. Fax text is large, so 1300px stays very legible.
LLM_IMAGE_MAX_DIM = 1300
# Safety cap on how many fax pages we send to the vision model in one call.
FAX_LLM_MAX_PAGES = 10

# Refinement trigger: call the LLM only when the heuristic parse left more than
# this many unclassified lines in extra_fields (in addition to the missing
# name / id_number checks). Keeps token usage near-zero on clean scans. When it
# does fire, the card image is attached (same threshold gates the image send).
REFINE_MAX_EXTRA_FIELDS = 2

# ---------------------------------------------------------------------------
# OCR tunables
# ---------------------------------------------------------------------------
# Long-side px; small phone captures are the #1 cause of bad OCR, so we upscale
# much more aggressively than a naive "just big enough" target.
UPSCALE_TARGET = 2200

# --- EasyOCR model loading (startup speed) ---------------------------------
# Where EasyOCR's model weights live. Defaults to EasyOCR's own cache
# (~/.EasyOCR/model). When the weights are already present there, the Reader is
# built with download_enabled=False, which skips the per-start download/verify
# handshake and just loads the local files — noticeably faster server startup.
EASYOCR_MODEL_DIR = os.environ.get(
    "EASYOCR_MODEL_DIR",
    os.path.join(os.path.expanduser("~"), ".EasyOCR", "model"),
)

# CPU inference threads for torch. 0 = leave torch's default untouched. Set a
# value (e.g. your physical core count) only if you want to cap/pin threads;
# this is accuracy-neutral — it only affects speed/scheduling.
OCR_NUM_THREADS = int(os.environ.get("OCR_NUM_THREADS", "0"))

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
