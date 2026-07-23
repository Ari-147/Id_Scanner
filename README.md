# ID Card Scanner & Fax Extractor

A lightweight FastAPI backend + single-file HTML/JS frontend that scans ID / insurance cards
through a device camera (or an uploaded photo), runs OCR, extracts structured fields with dynamic
heuristics, and cross-references the text against a payer database for an instant `payer_id` lookup.

It also parses **medical referral faxes** (image or PDF) into a structured field set via two
interchangeable engines: a fast, layout-aware **regex/geometry** parser and a **Claude vision**
parser that reads the fax image directly (including its order-type checkboxes). See
[Fax parsing](#fax-parsing-medical-referral-faxes).

## Features

- **In-memory OCR pipeline** — image streamed straight into OpenCV/NumPy buffers (no disk I/O),
  preprocessed (grayscale → upscale small captures → CLAHE contrast), then read with
  [`easyocr`](https://github.com/JaidedAI/EasyOCR) (CPU mode).
- **Confidence filtering** — each OCR detection carries a confidence; anything below `min_conf`
  (default `0.35`, tunable per request) is set aside in `dropped_low_confidence` instead of
  polluting the parsed fields.
- **3-tier heuristic parser** for varying card layouts:
  1. **Fuzzy label match** — matches known labels (`Name`, `DOB`, `Member ID`, …) on the same line
     (`Label: value`) or the next line.
  2. **Format heuristics** — regex for dates, digit-density for ID numbers, and merging of
     consecutive ALL-CAPS lines into names/organizations.
  3. **Catch-all** — unclassified lines land in an `extra_fields` dict (nothing is lost).
- **Smart date handling** — every date on a line is captured (`MM/DD/YYYY`, `YYYY-MM-DD`,
  `DD Mon YYYY`, `MM/YYYY`). Labeled dates win; unlabeled dates are classified by year:
  future → `expiry`, ≥16 years ago → `dob`, recent past → `issue_date`. All raw dates are also
  echoed in a `dates` list.
- **ID normalization** — alphabetic characters in ID numbers are uppercased (OCR often lowercases,
  e.g. `Y`→`y`).
- **Strict payer lookup** — see [Payer matching](#payer-matching-behavior) below.
- **Optional Claude refinement (now vision-capable)** — an *additive*, off-by-default step that asks
  a fast Claude model to double-check the field assignment when the heuristic parse looks incomplete.
  When it fires, the **card image is attached** so Claude can read the card directly instead of
  trusting shaky OCR. Strictly token-disciplined and fully graceful — see
  [LLM refinement](#optional-llm-refinement) below.
- **Fax parsing** — two endpoints turn a medical referral fax (image **or PDF**) into the same
  structured JSON: `/extract-data-v1` (regex + geometry, zero-cost) and `/extract-data-v2`
  (Claude vision). See [Fax parsing](#fax-parsing-medical-referral-faxes).
- **Fast startup** — the EasyOCR model is preloaded once at server boot and constructed without the
  per-start download/verify handshake, so the first request isn't slow. See
  [OCR startup & performance](#ocr-startup--performance).
- **Single-file frontend** — Tailwind CSS, HTML5 `getUserMedia` (back camera, high resolution),
  live preview, capture button, camera-flip, **image-upload fallback**, a **Fax Document** tab
  (PDF/image upload + engine selector), and on-screen JSON with a payer-match banner.

## Payer matching behavior

On startup, every `payer_name` + `alt_names` from `payers_data.py` is flattened and **pre-indexed**
into a normalized lookup map (~7,000 keys) for sub-millisecond matching. Each OCR line is then
matched as follows:

1. **Exact (normalized) match wins instantly** — even short names like `AARP` or `CIGNA`.
2. **Fuzzy fallback** uses `rapidfuzz` `token_sort_ratio` with `score_cutoff=90`, and only for
   **multi-word** lines. A length-comparability guard blocks long-vs-tiny spurious hits.

**Why multi-word-only fuzzy matching?** A single common English word on a card must not
fuzzy-match a one-word payer. For example a US **Medicare** card prints the word `MEDICAL`
(as in "MEDICAL (PART B)"), which is one letter away from the payer **`Medica`** (a Minnesota
commercial plan, id `94265`). Matching those would be a false positive — `MEDICAL` is not an
alias of `Medica` (verified: it appears in **zero** `alt_names`). The strict rule returns
`payer_match: null` for such cards, which is the correct answer.

> If a card is genuinely not in `payers_data.py`, `payer_match` is `null` and the frontend shows a
> "No payer match — card not found in database" banner. **All scanned fields are still returned.**
> To make a new card matchable, add an entry (with `alt_names`) to `payers_data.py`; it is picked
> up on the next restart — no code change needed.

This strict behavior (and the `payer_match` object) is unchanged. Separately, the compiled `result`
block adds a *graded* fallback for cards that are a near-superset of a known name/alt-name (e.g.
"BlueCare Plus" ~ "BlueCare") — see [Compiled result & confidence](#compiled-result--confidence) →
partial / alternate-name case. That's additive and never alters `payer_match`.

## Compiled result & confidence

Alongside the existing flat `fields` object (unchanged — `result` is purely additive), each `/scan`
response includes a single, already-organized `result` block: the thing a consumer actually wants to
read without reassembling it from `fields`. It's built by `build_result_summary()` in
`result_summary.py` — a pure view layer that does no OCR, parsing, matching, or API calls.

```json
"result": {
  "status": "matched",                 // "matched" | "partial_match" | "not_found"
  "confidence": 92.3,                    // percentage, meaningful for ALL outcomes — shown on top
  "confidence_basis": "fuzzy_match",     // "exact_match" | "fuzzy_match" | "partial_match" | "absence_estimate"
  "payer": { "payer_id": "94265", "payer_name": "Medica" },  // null only when not_found
  "cardholder": { "name": "...", "id_number": "...", "dob": "...", "sex": "...", "address": "..." },
  "coverage":   { "organization": "...", "group_number": "...", "issue_date": "...", "expiry": "...", "dates": ["..."] },
  "unclassified_fields": { },            // same as fields.extra_fields
  "refinement": "skipped"
}
```

Every field with no extracted value is `null` (not omitted), so the shape is predictable for
frontend consumption.

**Confidence — matched case.** Surfaced directly from the existing `match_payer` score (no new math):
`100.0` for an exact normalized hit (`confidence_basis: "exact_match"`), or the `token_sort_ratio`
score in the 90–100 range for a fuzzy hit (`"fuzzy_match"`). The path is recorded by an additive
`match_type` field on the payer-match object.

**Confidence — partial / alternate-name case (`status: "partial_match"`).** A card often prints a
*superset* of a known `payer_name` or `alt_name` — e.g. a **"BlueCare Plus"** card versus the
alt-name **"BlueCare"**, or "Blue Shield of California PPO" versus "Blue Shield of California". The
strict `match_payer` deliberately rejects these (to avoid confident false positives), so they used
to read as a flat "not found". `resolve_payer_candidate()` now falls back to
`partial_payer_candidate()`, which surfaces the closest known payer as a graded partial match:

- Candidates are ranked with `token_set_ratio` at a **high** cutoff (95) — i.e. a known name must be
  *almost fully contained* in the card text. That high bar is what distinguishes a genuine superset
  (`"BlueCare Plus" ⊃ "BlueCare"`, set-ratio 100) from a coincidental generic-word overlap
  (`"MEDICARE HEALTH INSURANCE"` vs `"WPS HEALTH INSURANCE"`, set-ratio ~88 → rejected).
- The same length-comparability guard as the strict path blocks tiny-substring hits (e.g. matching
  just `"AETNA"` out of a much longer line).
- The reported `confidence` is the **length-sensitive** `token_sort_ratio`, so extra words on the
  card correctly pull it below 100 ("somewhat matches") — e.g. `"BlueCare Plus"` → **BlueCare**
  (owned by the `TN BCBS` payer) at ~76%.

This path is `confidence_basis: "partial_match"`. Because it's not a strict database hit,
`fields.payer_match` stays `null` for a partial match — the graded candidate appears only in the
richer `result` block (`result.payer`), and the frontend labels it "Likely match … not an exact
database entry".

**Confidence — not-found case.** A `"not found"` used to give no sense of *how sure* we are. A clean,
well-lit card whose payer legitimately isn't in `payers_data.py` should read very differently from a
blurry photo where we simply failed to read enough text. `estimate_absence_confidence()` computes a
**local, deterministic** score (no LLM/API call, no tokens) from scan-quality evidence:

- starts at a neutral baseline (50);
- **+** if an `organization`/payer-ish name was extracted yet still didn't match (strongest signal of
  a *genuine* absence);
- **+** if the average confidence of the kept OCR lines is high (the card was read well);
- **+** if little was left unclassified in `extra_fields`;
- **−** (toward uncertain) if there are few kept lines, low average confidence, or no organization —
  i.e. we may just not have read enough of the card to know;
- clamped to **30–97** (never claims near-certainty of absence from a heuristic, never implies a
  random guess).

This path is flagged `confidence_basis: "absence_estimate"` so it's clear the number is a heuristic
estimate, not a database certainty. The frontend renders it with an "estimated from scan quality"
caption.

## Optional LLM refinement

After the heuristic parser and payer lookup run, the app can optionally ask a fast, multimodal Claude
model (`claude-haiku-4-5-20251001`) to **double-check and correct the field assignment**. When it
fires, the **card image is attached** so Claude can read the card directly — this is the recovery
path for exactly the scans where OCR did poorly, so seeing the pixels (not just the shaky text) is
what makes the correction reliable.

**Off by default.** It runs only if `ANTHROPIC_API_KEY` is set in the environment. With no key, the
app behaves exactly as it always has (the `.env.example` shows the only new setting).

**Token discipline — the call is skipped on scans that already look clean.** The trigger lives in
one place (`needs_refinement()` in `llm_refine.py`) so it's easy to see and tune. The model is only
called when the heuristic parse looks incomplete or ambiguous:

- `name` is missing, **or**
- `id_number` is missing, **or**
- more than ~2 unclassified lines are left over in `extra_fields`.

Otherwise the heuristic result is returned as-is with **no API call**. A clean card therefore costs
zero tokens.

**What gets sent (kept minimal):** the confident OCR `raw_lines` (typically < 20 short lines), the
parser's current best-guess `fields` (as compact JSON), and the **card image** — downscaled to a
1300 px long edge and JPEG-encoded (`llm_image.py`), which is the main lever keeping the vision call
cheap (≈ 1.3 k image tokens). The payer database, `dropped_low_confidence` lines, and per-detection
metadata (bbox / confidence) are **never** sent. Output is capped separately per call shape
(`LLM_MAX_TOKENS` for a text-only call, `LLM_IMAGE_MAX_TOKENS` when the image is attached).

**Merge behavior:** results are merged field-by-field — an LLM value is used only where it actually
filled something in, so a sparse response can never blank out a field the heuristic parser already
got right. Payer fields (`payer_id` / `payer_name` / `payer_match`) are preserved untouched.

**Fallback guarantee (an LLM failure never breaks or degrades a scan):**

| Situation | Behavior | `refinement` value |
|-----------|----------|--------------------|
| Heuristic parse already complete | API not called | `"skipped"` |
| No `ANTHROPIC_API_KEY` set (or `anthropic` not installed) | Clean no-op | `"not_configured"` |
| LLM ran (image attached) and returned usable JSON | Merged into `fields` | `"applied_image"` |
| LLM ran (text only) and returned usable JSON | Merged into `fields` | `"applied"` |
| API error / timeout / no network / expired key / malformed JSON | Caught, warning logged, heuristic result returned unchanged | `"failed"` |

Set the key via a standard environment variable (see `.env.example`):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

The `refinement` field is the **only** addition to the `/scan` response schema.

## Fax parsing (medical referral faxes)

Two endpoints extract a **medical referral fax** into one flat, structured field set (fax metadata,
organization, referring provider, patient, order-type checkboxes, and laboratory values). Both
accept an **image or a PDF** (multi-page PDFs are rasterized per page via `pdf_converter.py`, which
needs Poppler — see [Requirements](#requirements)) and return the **same response shape** so the
frontend renders either identically:

```json
{ "method": "regex" | "claude", "fields": { ... }, "ocr_confidence": [ ... ] }
```

The canonical field list lives in one place, `fax_parser.py::FAX_FIELDS`, and is shared by both
engines.

### `/extract-data-v1` — regex + geometry (zero-cost baseline)

Referral faxes are *forms*, not prose: a field's label sits on one line and its value in the box
**directly below** it, and lab results are a **two-column grid**. A naive "Label: value" regex misses
almost everything, so `fax_parser.py` is **layout-aware** and uses each OCR detection's normalized
`x`/`y` position (now returned by `run_adaptive_ocr`):

- a field's value is the detection(s) **directly below its label in the same x-column** — this is
  what correctly pairs the two-column labs (e.g. `LDL → 91`, `HDL → 60`, `ESR → 56`, `CRP → 100`),
  which a plain reading-order scan gets wrong;
- fragments on the same row are merged left-to-right (e.g. a phone split by OCR into `+1` and
  `7869899867` → `+17869899867`);
- generic labels (`Email` / `Phone` / `Name`) are disambiguated by which form **section**
  (Provider / Patient / Labs) their y-position falls in.

> **Checkbox limitation (v1 only):** OCR captures the printed *text* of every order-type option but
> not the checkbox tick, so v1 reports which options the form *lists*, not which are *selected*
> (all listed options read `true`). Use `/extract-data-v2` for correct checkbox state.

### `/extract-data-v2` — Claude vision

Sends the fax page image(s) **directly to Claude** (no OCR step) so the model reads the form —
including the **order-type checkboxes** — from the pixels. This resolves the v1 checkbox limitation
and tends to recover details OCR mangles (accented/segmented phones and emails). Requires
`ANTHROPIC_API_KEY`; returns `503` if unset.

- **Token discipline:** each page is downscaled to a 1300 px long edge and JPEG-encoded
  (`llm_image.py`); at most `FAX_LLM_MAX_PAGES` pages are sent; output is capped at
  `FAX_LLM_IMAGE_MAX_TOKENS`.
- The response adds `"source": "image"` and `"pages": N`, and `ocr_confidence` is `[]` (there is no
  OCR pass). The frontend shows *"Read directly by Claude vision (N pages)"* in place of a
  confidence percentage.

### Fax UI

The frontend adds a **Fax Document** tab (alongside Live Camera and Upload Image) with a drop zone
for PDF/PNG/JPG and an **engine selector** (Regex v1 / Claude AI v2). Results render into grouped,
fax-appropriate sections (Fax Metadata, Organization, Provider, Patient, Order Type, Lab Values,
Support) rather than the ID-card layout.

## OCR startup & performance

EasyOCR runs on CPU and its model is loaded once per process. Two changes keep startup fast without
changing OCR accuracy:

- **Warm preload at boot** — a FastAPI `lifespan` handler calls `get_reader()` at startup, so the
  model is ready before the first request instead of stalling it for several seconds.
- **No per-start download/verify handshake** — when the weights already exist in
  `EASYOCR_MODEL_DIR` (default `~/.EasyOCR/model`), the reader is built with `download_enabled=False`
  and just loads the local files. A one-time download still happens automatically on first-ever
  setup (or if the directory is empty). `easyocr` is pinned in `requirements.txt` so an upgrade can't
  silently expect different model files and trigger a real re-download.
- **Accuracy-neutral knobs** — `quantize=True` (EasyOCR's default CPU int8 speedup, made explicit)
  and an optional `OCR_NUM_THREADS` cap for torch.

DPI and the adaptive two-pass OCR logic are unchanged, so per-page OCR output is identical to before.

```
Task_2_OCR/
├── app.py             # FastAPI app + lifespan preload; routes: / /health /scan
│                      #   /extract-data-v1 /extract-data-v2 — orchestration only
├── config.py          # env vars + tunables (ANTHROPIC_API_KEY, model dir, token/image limits)
├── ocr.py             # preprocessing + adaptive two-pass OCR + merge (run_adaptive_ocr, get_reader)
├── parser.py          # 3-tier dynamic heuristic parser for ID cards (parse_fields)
├── payer_matching.py  # pre-indexed rapidfuzz payer lookup (match_payer)
├── llm_refine.py      # optional Claude ID-card refinement, incl. vision (needs_refinement / maybe_refine)
├── fax_parser.py      # layout/geometry-aware regex fax parser (parser_fax_with_regex, FAX_FIELDS)
├── fax_llm.py         # Claude-vision fax parser (parse_fax_with_llm)
├── pdf_converter.py   # PDF → per-page images for faxes (needs Poppler)
├── llm_image.py       # shared image → downscaled JPEG → Anthropic vision block
├── index.html         # Camera + upload + Fax Document tabs + result UI (served at /)
├── payers_data.py     # PAYERS_DATA list (payer_name, payer_id, alt_names)
├── requirements.txt
├── .env.example       # optional ANTHROPIC_API_KEY (+ optional EASYOCR_MODEL_DIR / OCR_NUM_THREADS)
├── test_llm_refine.py # mocked tests for the refinement step (no key/network needed)
└── README.md
```

The scan pipeline (`app.py`) is pure orchestration: `run_adaptive_ocr()` → `parse_fields()` →
`match_payer()` → (optionally) `maybe_refine()`. The fax pipeline is `run_adaptive_ocr()` (v1) or
`convert_pdf_to_images()` (v2) → `parser_fax_with_regex()` / `parse_fax_with_llm()`. The `/scan`
output on a given image is identical to before, plus the `refinement` field.

## Requirements

- Python 3.10+
- Packages: `fastapi`, `uvicorn`, `easyocr` (pinned `==1.7.2`), `rapidfuzz`,
  `opencv-python-headless`, `numpy`, `python-multipart`, `pillow`, `anthropic`, `pdf2image`
  (see `requirements.txt`)
- **Poppler** — required by `pdf2image` to rasterize PDF faxes. Install it and point
  `pdf_converter.py::POPPLER_PATH` at its `bin` directory (Windows), or install
  `poppler-utils` on Linux / `brew install poppler` on macOS. Only needed if you upload **PDF**
  faxes; image faxes and the ID-card `/scan` path don't use it.
- **`anthropic` is only used by the Claude paths** (ID-card image refinement and the `/extract-data-v2`
  fax parser). If you never set `ANTHROPIC_API_KEY`, it's imported harmlessly and those paths become
  a clean no-op / `503` — the core OCR / parsing / payer pipeline has no new dependency.
- **`pillow`** encodes/downscales images for the vision calls (`llm_image.py`) in addition to its
  prior use.

## Setup

```bash
# from the project root
python -m venv .venv

# activate the virtualenv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Git Bash):
source .venv/Scripts/activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8080
```

Then open **http://127.0.0.1:8080** (or **http://localhost:8080**) in your browser.

### Camera access note

`getUserMedia` requires a **secure context**. It works on `localhost` out of the box.
To open it from a **phone over your LAN**, you need HTTPS — use a tunnel such as
[ngrok](https://ngrok.com/) (`ngrok http 8000`) or Cloudflare Tunnel, or serve behind a
reverse proxy with a TLS certificate. For a quick test without a camera, use the **📁 upload**
button in the UI.

## Configuration

All settings live in `config.py`; the only ones read from the environment are optional:

| Env var | Default | Purpose |
|---------|---------|---------|
| `ANTHROPIC_API_KEY` | *(unset)* | Enables the Claude paths (ID-card image refinement + `/extract-data-v2`). Unset = clean no-op / `503`. |
| `EASYOCR_MODEL_DIR` | `~/.EasyOCR/model` | Where EasyOCR weights live. If present there, startup skips the download/verify handshake. |
| `OCR_NUM_THREADS` | `0` (torch default) | Optional CPU thread cap for torch. Accuracy-neutral; only affects speed. |

Token/image budgets (`LLM_MAX_TOKENS`, `LLM_IMAGE_MAX_TOKENS`, `FAX_LLM_IMAGE_MAX_TOKENS`,
`LLM_IMAGE_MAX_DIM`, `FAX_LLM_MAX_PAGES`, …) are plain constants in `config.py` — tune them there.

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # optional — turns on the Claude paths
```

## API

| Method | Path               | Description                                                              |
|--------|--------------------|--------------------------------------------------------------------------|
| GET    | `/`                | Serves the scanner frontend (`index.html`)                               |
| GET    | `/health`          | Health check — returns status and number of indexed payers               |
| POST   | `/scan`            | Multipart form (`file`) with a card image → structured JSON              |
| POST   | `/extract-data-v1` | Multipart form (`file`, image or PDF) → fax fields via regex + geometry  |
| POST   | `/extract-data-v2` | Multipart form (`file`, image or PDF) → fax fields via Claude vision (needs `ANTHROPIC_API_KEY`; `503` if unset) |

**`POST /scan` query param:** `min_conf` (float, default `0.35`) — OCR confidence threshold below
which detections are dropped from parsing.

### Example

```bash
# ID / insurance card
curl -X POST "http://127.0.0.1:8080/scan?min_conf=0.35" -F "file=@card.jpg"

# Fax (image or PDF) — regex/geometry engine
curl -X POST "http://127.0.0.1:8080/extract-data-v1" -F "file=@referral_fax.pdf"

# Fax (image or PDF) — Claude vision engine (requires ANTHROPIC_API_KEY)
curl -X POST "http://127.0.0.1:8080/extract-data-v2" -F "file=@referral_fax.pdf"
```

### Sample response (card found in payers_data.py)

```json
{
  "raw_lines": ["Name: JOHN A SMITH", "Member ID: 123456789", "DOB: 03/15/1985", "AARP"],
  "fields": {
    "name": "JOHN A SMITH",
    "id_number": "123456789",
    "dob": "03/15/1985",
    "extra_fields": {},
    "payer_id": "36273",
    "payer_name": "AARP",
    "payer_match": {
      "payer_id": "36273",
      "payer_name": "AARP",
      "matched_on": "AARP",
      "score": 100.0
    }
  },
  "ocr_confidence": [{ "text": "AARP", "conf": 0.99 }],
  "dropped_low_confidence": [],
  "refinement": "skipped",
  "result": {
    "status": "matched",
    "confidence": 100.0,
    "confidence_basis": "exact_match",
    "payer": { "payer_id": "36273", "payer_name": "AARP" },
    "cardholder": { "name": "JOHN A SMITH", "id_number": "123456789", "dob": "03/15/1985", "sex": null, "address": null },
    "coverage": { "organization": null, "group_number": null, "issue_date": null, "expiry": null, "dates": null },
    "unclassified_fields": {},
    "refinement": "skipped"
  }
}
```

### Sample response (card NOT in payers_data.py — e.g. a Medicare card)

```json
{
  "raw_lines": ["MEDICARE HEALTH INSURANCE", "RODNEY G MULLINS", "66J9-V66-FY35", "MEDICAL", "(PART B)", "02-01-2024"],
  "fields": {
    "name": "RODNEY G MULLINS",
    "id_number": "66J9-V66-FY35",
    "organization": "MEDICARE HEALTH INSURANCE",
    "issue_date": "04-01-2023",
    "dates": ["04-01-2023", "02-01-2024"],
    "extra_fields": { "line_10": "MEDICAL (PART B)" },
    "payer_match": null
  },
  "refinement": "skipped",
  "result": {
    "status": "not_found",
    "confidence": 85.0,
    "confidence_basis": "absence_estimate",
    "payer": null,
    "cardholder": { "name": "RODNEY G MULLINS", "id_number": "66J9-V66-FY35", "dob": null, "sex": null, "address": null },
    "coverage": {
      "organization": "MEDICARE HEALTH INSURANCE",
      "group_number": null,
      "issue_date": "04-01-2023",
      "expiry": null,
      "dates": ["04-01-2023", "02-01-2024"]
    },
    "unclassified_fields": { "line_10": "MEDICAL (PART B)" },
    "refinement": "skipped"
  }
}
```

> In the not-found sample above, `confidence` is a heuristic estimate (basis `absence_estimate`): a
> payer-looking `organization` was read and the OCR confidence was high, yet nothing matched the
> database — so we're fairly (not certainly) confident the card genuinely isn't a known payer.

### Sample fax response (`/extract-data-v2`, Claude vision)

```json
{
  "method": "claude",
  "source": "image",
  "pages": 3,
  "fields": {
    "sender_name": "Preventi.AI",
    "fax_date": "07/14/2026",
    "fax_time": "10:32 AM",
    "website": "preventi.ai",
    "email": "scaninfo@preventi.ai",
    "provider_name": "Bilal",
    "provider_email": "Bilal@yopmail.com",
    "provider_phone": "+15445634243",
    "npi": "91528473",
    "patient_name": "jack",
    "date_of_birth": "04/04/1999",
    "patient_phone": "+17869899867",
    "ct_with_calcium_scoring_and_cardiology_e_consult": true,
    "ccta_with_cardiology_e_consult": false,
    "ccta_ai_analysis_with_clearly": false,
    "recent_creatinine": "0.9",
    "ldl": "91", "hdl": "60", "total_cholesterol": "135", "triglycerides": "54",
    "apo_a": "200", "apo_b": "60", "esr": "56", "crp": "100",
    "support_email": "support@preventi.ai"
  },
  "ocr_confidence": []
}
```

> Note the checkbox booleans: only `ct_with_calcium_scoring…` is `true` (the box actually ticked on
> the fax), the other two `false` — the vision engine reads the ticks from the image, which the
> regex engine (`v1`) cannot. Fields absent from the fax are `null` (omitted above for brevity); the
> full flat `FAX_FIELDS` set is always present.

## Response fields

| Field                     | Description                                                            |
|---------------------------|------------------------------------------------------------------------|
| `raw_lines`               | Confidence-filtered OCR text lines                                     |
| `fields`                  | Structured extraction (see below)                                      |
| `fields.payer_match`      | Matched payer object, or `null` if the card isn't in the database      |
| `fields.dates`            | All dates found (raw), regardless of classification                    |
| `fields.extra_fields`     | Unclassified lines (tier-3 catch-all)                                  |
| `ocr_confidence`          | Kept detections with their confidence scores                           |
| `dropped_low_confidence`  | Detections discarded because `conf < min_conf`                         |
| `refinement`              | LLM refinement status: `skipped` / `not_configured` / `applied` / `applied_image` / `failed` (see [Optional LLM refinement](#optional-llm-refinement)) |
| `result`                  | Compiled, frontend-ready summary + a confidence % for **all** outcomes — match, partial / alternate-name match, and no-match (see [Compiled result & confidence](#compiled-result--confidence)) |

## How it works

1. The browser captures a high-resolution frame from the back camera (or you upload an image) and
   POSTs it as a JPEG blob to `/scan`.
2. `app.py` decodes the bytes into a NumPy array, preprocesses (grayscale → upscale → CLAHE), and
   runs easyocr with per-line confidence.
3. Low-confidence detections are set aside; the rest pass through the 3-tier parser.
4. Each line is checked against the pre-indexed payer map (exact first, then strict multi-word
   fuzzy); a confident hit adds `payer_id` / `payer_name`.
5. If (and only if) the heuristic parse looks incomplete **and** `ANTHROPIC_API_KEY` is set, a fast
   Claude model refines the field assignment — **with the card image attached** so it can read the
   card directly (see [Optional LLM refinement](#optional-llm-refinement)); otherwise this step is
   skipped entirely.
6. The frontend renders the JSON and a green (matched) or amber (no match) payer banner.

Faxes follow a separate flow — see [Fax parsing](#fax-parsing-medical-referral-faxes): the
**Fax Document** tab POSTs an image/PDF to `/extract-data-v1` (regex + geometry) or
`/extract-data-v2` (Claude vision, image sent directly), and the results render into
fax-specific sections.

## Accuracy tips

- **Fill the frame with the card** and hold steady — small/blurry captures are the #1 cause of poor
  OCR. The frontend requests up to 4K and shows the actual capture resolution.
- Good, even lighting; avoid glare on laminated cards.
- If a genuine payer isn't matching, confirm its name/`alt_names` exist in `payers_data.py`.
