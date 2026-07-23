"""
In-memory OCR pipeline: image bytes -> preprocessed numpy -> easyocr -> lines.

No disk I/O. Includes the adaptive two-pass strategy: a good, clean photo scans
in one pass; a hard capture (blur / low light / too far away) automatically gets
a second pass with a different preprocessing variant, and the two are merged.

The easyocr reader is a lazy singleton and lives here (not in app.py) so that
run_adaptive_ocr can use it without a circular import back into the app module.
"""
import logging
import os
import re

import numpy as np
import cv2
import easyocr
from fastapi import HTTPException

import config
from config import UPSCALE_TARGET, MIN_CONFIDENT_LINES_FOR_SINGLE_PASS
from parser import parse_fields

log = logging.getLogger("ocr")

# ---------------------------------------------------------------------------
# OCR engine (lazy singleton — the model is loaded once per process).
#
# Call get_reader() once at server startup (see app.py lifespan) to warm this
# singleton, so no request pays the multi-second model-load stall. When the
# weights already exist in EASYOCR_MODEL_DIR we build with
# download_enabled=False, which skips EasyOCR's per-start download/verify
# handshake and just loads the local files.
# ---------------------------------------------------------------------------
_reader = None

# The two weight files EasyOCR needs for an English CPU reader.
_MODEL_FILES = ("craft_mlt_25k.pth", "english_g2.pth")


def _models_present(model_dir: str) -> bool:
    return all(os.path.isfile(os.path.join(model_dir, f)) for f in _MODEL_FILES)


def get_reader():
    global _reader
    if _reader is not None:
        return _reader

    # Optional, accuracy-neutral CPU thread cap/pin.
    if config.OCR_NUM_THREADS and config.OCR_NUM_THREADS > 0:
        try:
            import torch
            torch.set_num_threads(config.OCR_NUM_THREADS)
        except Exception as e:  # noqa: BLE001 - never let a tuning knob break OCR
            log.warning("Could not set torch threads: %s", e)

    model_dir = config.EASYOCR_MODEL_DIR
    have_models = _models_present(model_dir)
    log.info(
        "Loading EasyOCR reader (model_dir=%s, models_present=%s, download=%s)",
        model_dir, have_models, not have_models,
    )
    _reader = easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=model_dir,
        # Skip the download/verify handshake when weights are already local;
        # allow a one-time download only if they're missing (first-ever setup).
        download_enabled=not have_models,
        quantize=True,        # CPU int8 quantization (EasyOCR default; explicit)
        verbose=False,
    )
    return _reader


# ---------------------------------------------------------------------------
# Image -> preprocessed numpy, streamed, no disk.
#
# Two variants are used because they fail on *different* kinds of bad
# captures, and a full retry is only paid for when it's actually needed:
#   - "primary"     grayscale, upscaled, denoised, contrast-boosted, sharpened.
#                    Best default for typical phone photos (soft focus, low
#                    light, small/far-away captures).
#   - "alt_color"    color, upscaled only, no grayscale/denoise/sharpen.
#                    Sometimes recovers text the primary variant blurs away
#                    (thin strokes, colored print on colored backgrounds).
# ---------------------------------------------------------------------------
def _decode(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    return img


def _upscale(img: np.ndarray, target: int = UPSCALE_TARGET) -> np.ndarray:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side >= target:
        return img
    scale = target / long_side
    # LANCZOS4 holds up better than CUBIC for large upscale factors
    # (small/far-away captures need >2x); CUBIC is fine for mild upscales.
    interp = cv2.INTER_LANCZOS4 if scale > 2 else cv2.INTER_CUBIC
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)


def preprocess_primary(raw: bytes) -> np.ndarray:
    img = _decode(raw)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orig_long_side = max(img.shape[:2])
    gray = _upscale(gray)
    if orig_long_side < 700:
        # denoising is somewhat costly; only worth it on small/noisy captures
        gray = cv2.fastNlMeansDenoising(gray, h=10)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    # unsharp mask — counteracts the softening from upscale + denoise and
    # from genuine motion/focus blur in the original capture
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return gray


def preprocess_alt_color(raw: bytes) -> np.ndarray:
    img = _decode(raw)
    return _upscale(img)


# Back-compat alias (old single-variant name), used by any external caller
# that imports `preprocess` directly.
preprocess = preprocess_primary


def _norm_key(text: str) -> str:
    """Loose key for de-duplicating near-identical detections across passes."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _run_ocr(image: np.ndarray) -> list[tuple[str, float, float, float]]:
    """Returns (text, conf, y_norm, x_norm) so results can be put back into
    reading order (top-to-bottom, left-to-right) regardless of the order the
    detector happened to return them in, or which preprocessing pass found
    them. The label-adjacent-line parsing tier depends on reading order."""
    h, w = image.shape[:2]
    detections = get_reader().readtext(image, detail=1, paragraph=False)
    out = []
    for bbox, text, conf in detections:
        t = (text or "").strip()
        if not t:
            continue
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        out.append((t, float(conf), min(ys) / h, min(xs) / w))
    return out


def run_adaptive_ocr(raw: bytes, min_conf: float) -> list[dict]:
    """
    Full adaptive OCR + merge, returning detections as {"text", "conf"} dicts
    already sorted into reading order (top-to-bottom, left-to-right).

    Conf values are raw (unrounded) so the caller can split kept vs dropped at
    exactly the same threshold the retry decision used.
    """
    primary = preprocess_primary(raw)
    detections = _run_ocr(primary)
    confident_count = sum(1 for _, c, _, _ in detections if c >= min_conf)

    # A quick peek at what the primary pass alone would extract, just to
    # decide whether a retry is worth it (cheap: reuses the same parser).
    primary_preview = [t for t, c, _, _ in detections if c >= min_conf]
    primary_fields = parse_fields(primary_preview)
    missing_critical = not primary_fields.get("name") and not primary_fields.get(
        "id_number"
    )

    # Adaptive retry: only pay for the second (alt-color) OCR pass when the
    # first pass looks like it came from a poor-quality capture — either too
    # few confident lines, or the fields that actually identify the
    # cardholder didn't come through. Good, clean photos scan in one pass;
    # hard ones get a second chance automatically.
    if confident_count < MIN_CONFIDENT_LINES_FOR_SINGLE_PASS or missing_critical:
        alt = preprocess_alt_color(raw)
        alt_detections = _run_ocr(alt)
        merged: dict[str, tuple[str, float, float, float]] = {}
        for t, c, y, x in detections + alt_detections:
            key = _norm_key(t)
            if not key:
                continue
            if key not in merged or c > merged[key][1]:
                merged[key] = (t, c, y, x)
        detections = list(merged.values())

    # restore reading order (top-to-bottom, left-to-right) — required for the
    # parser's "label on one line, value on the next" tier to work correctly
    detections.sort(key=lambda d: (d[2], d[3]))

    # `x`/`y` are the normalized (0-1) top-left position of each detection.
    # The ID-card path ignores them (it only reads "text"); the fax parser uses
    # them to reconstruct the form's 2-D layout (label above / value below, and
    # two-column lab grids), which reading order alone gets wrong.
    return [{"text": t, "conf": c, "y": y, "x": x} for t, c, y, x in detections]
