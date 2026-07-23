"""
Layout-aware medical-referral-fax parser (heuristic / no LLM).

Referral faxes are *forms*, not prose: a field's label sits on one line and its
value sits in the box directly below it, and lab values are laid out in a
two-column grid. A naive "Label: value on one line" regex therefore misses
almost everything. This parser instead uses the OCR detections' geometry
(normalized x/y from ocr.run_adaptive_ocr):

  - the value for a labelled field is the detection(s) directly BELOW the label
    in the SAME x-column (this is what fixes two-column labs — LDL's value is the
    number under LDL, not merely the next number in reading order);
  - fragments on the same row (e.g. a phone split into "+1" and "7869899867")
    are merged left-to-right;
  - generic labels ("Email", "Phone", "Name") are disambiguated by which form
    section (Provider / Patient / Labs) their y-position falls in.

Shares the canonical field set with fax_llm.py so `/extract-data-v1` (this) and
`/extract-data-v2` (Claude) return an identical shape.
"""
import re

# ---------------------------------------------------------------------------
# Canonical field set — the single source of truth shared with fax_llm.py.
# ---------------------------------------------------------------------------
FAX_FIELDS = [
    # Fax metadata
    "sender_name", "sender_fax", "recipient_name", "recipient_fax",
    "page_number", "total_pages", "fax_date", "fax_time",
    # Cover page
    "note", "fax_number",
    # Organization
    "website", "email", "organization_fax",
    # Referring provider
    "provider_name", "provider_email", "provider_phone", "provider_fax", "npi",
    # Patient
    "patient_name", "date_of_birth", "patient_phone", "patient_email",
    # Order
    "order_type",
    "ct_with_calcium_scoring_and_cardiology_e_consult",
    "ccta_with_cardiology_e_consult",
    "ccta_ai_analysis_with_clearly",
    # Laboratory values
    "recent_creatinine", "ldl", "hdl", "total_cholesterol", "triglycerides",
    "apo_a", "apo_b", "esr", "crp",
    # Support
    "support_email",
]

FAX_BOOLEAN_FIELDS = {
    "ct_with_calcium_scoring_and_cardiology_e_consult",
    "ccta_with_cardiology_e_consult",
    "ccta_ai_analysis_with_clearly",
}

# ---------------------------------------------------------------------------
# Regex building blocks / value validators
# ---------------------------------------------------------------------------
EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.?[A-Za-z]{2,}"
# Optional +country code, then a 3-3-4 grouping (with optional (), space/-/. )
PHONE = r"\+?\d{0,2}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
DATE = r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
TIME = r"\d{1,2}[:;]\d{2}\s*[AaPp]\.?[Mm]\.?"
_EMAIL_RE = re.compile(EMAIL)
_PHONE_RE = re.compile(PHONE)
_DATE_RE = re.compile(DATE)
_TIME_RE = re.compile(TIME)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Geometry tolerances (all in normalized 0-1 units; y is page-offset for PDFs).
_MIN_CONF = 0.30          # drop OCR garbage (blank-line separators come as 0.0)
_ROW_DY = 0.018           # detections within this Δy are on the same visual row
_VALUE_MAX_DY = 0.055     # a value box sits at most this far below its label
_COL_DX = 0.22            # label and its value share an x-column within this Δx


def _blank_result() -> dict:
    return {f: (False if f in FAX_BOOLEAN_FIELDS else None) for f in FAX_FIELDS}


def _norm(text: str) -> str:
    """Lowercase, keep alnum + spaces, collapse whitespace — for label matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


# ---------------------------------------------------------------------------
# Section layout: field -> (section, [label variants], value type)
# Sections scope generic labels (Email/Phone/Name) to the right field.
# ---------------------------------------------------------------------------
_FIELD_SPECS = [
    # provider
    ("provider_name",  "provider", ["provider name"],                 "text"),
    ("provider_email", "provider", ["email"],                         "email"),
    ("provider_phone", "provider", ["phone"],                         "phone"),
    ("provider_fax",   "provider", ["fax number", "fax"],             "phone"),
    ("npi",            "provider", ["npi"],                           "digits"),
    # patient
    ("patient_name",   "patient",  ["patient name", "name"],          "text"),
    ("date_of_birth",  "patient",  ["date of birth", "dob"],          "date"),
    ("patient_phone",  "patient",  ["phone"],                         "phone"),
    ("patient_email",  "patient",  ["email"],                         "email"),
    # labs (note: OCR often reads "HDL" as "KDL")
    ("recent_creatinine", "labs",  ["creatinine"],                    "number"),
    ("ldl",            "labs",     ["ldl"],                           "number"),
    ("hdl",            "labs",     ["hdl", "kdl"],                     "number"),
    ("total_cholesterol", "labs",  ["total cholesterol", "cholesterol"], "number"),
    ("triglycerides",  "labs",     ["triglycerides", "tg"],           "number"),
    ("apo_a",          "labs",     ["apo a", "apoa"],                 "number"),
    ("apo_b",          "labs",     ["apo b", "apob"],                 "number"),
    ("esr",            "labs",     ["esr"],                           "number"),
    ("crp",            "labs",     ["crp"],                           "number"),
]

# Every label variant + section-header word — used to reject a candidate value
# that is really another label.
_ALL_LABEL_TERMS = {v for _, _, variants, _ in _FIELD_SPECS for v in variants} | {
    "referring provider", "patient information", "order type",
    "laboratory", "provider information", "website", "npi",
}


def _extract_value(text: str, vtype: str):
    """Validate/clean a candidate string for the expected value type."""
    s = text.strip()
    if vtype == "email":
        m = _EMAIL_RE.search(s)
        return m.group(0) if m else None
    if vtype == "phone":
        digits = re.sub(r"\D", "", s)
        return s if len(digits) >= 7 else None
    if vtype == "digits":
        digits = re.sub(r"\D", "", s)
        return digits if len(digits) >= 6 else None
    if vtype == "date":
        m = _DATE_RE.search(s)
        return m.group(0) if m else None
    if vtype == "number":
        m = _NUM_RE.search(s)
        return m.group(0) if m else None
    if vtype == "text":
        if not s or _EMAIL_RE.search(s) or re.fullmatch(r"[\d\W]+", s):
            return None
        if any(term in _norm(s) for term in _ALL_LABEL_TERMS):
            return None
        return s
    return None


def _is_label_like(text: str) -> bool:
    n = _norm(text)
    return any(term in n for term in _ALL_LABEL_TERMS)


def _value_below(label, dets, vtype):
    """Find the value directly below `label` in the same x-column.

    Gathers detections that sit just below the label and roughly share its x,
    takes the nearest such row, merges its fragments left-to-right, and validates.
    """
    cands = [
        d for d in dets
        if _MIN_CONF <= d.get("conf", 1)
        and 0 < (d["y"] - label["y"]) <= _VALUE_MAX_DY
        and abs(d["x"] - label["x"]) <= _COL_DX
        and not _is_label_like(d["text"])
    ]
    if not cands:
        return None
    top_y = min(d["y"] for d in cands)
    row = sorted([d for d in cands if d["y"] - top_y <= _ROW_DY], key=lambda d: d["x"])
    joiner = "" if vtype in ("phone", "digits") else " "
    merged = joiner.join(d["text"].strip() for d in row)
    return _extract_value(merged, vtype)


def _section_bounds(dets):
    """Return the y of each section header (or None if absent)."""
    def find(kw):
        ys = [d["y"] for d in dets if kw in _norm(d["text"])]
        return min(ys) if ys else None
    return {
        "provider": find("referring provider"),
        "patient": find("patient information"),
        "order": find("order type"),
        "labs": find("laboratory"),
    }


def _section_range(section, bounds, ymax):
    """(y_start, y_end) window for a section, given detected header positions."""
    prov, pat, order, labs = (bounds["provider"], bounds["patient"],
                              bounds["order"], bounds["labs"])
    if section == "provider":
        return (prov if prov is not None else 0.0,
                pat if pat is not None else ymax)
    if section == "patient":
        return (pat if pat is not None else 0.0,
                order if order is not None else (labs if labs is not None else ymax))
    if section == "labs":
        return (labs if labs is not None else 0.0, ymax)
    return (0.0, ymax)


def parser_fax_with_regex(detections: list[dict]) -> dict:
    """Extract the canonical fax field set from positioned OCR detections."""
    result = _blank_result()

    # Tolerate a plain list of strings (no geometry) for robustness/tests.
    dets = []
    for i, d in enumerate(detections):
        if isinstance(d, str):
            dets.append({"text": d, "conf": 1.0, "y": i * 0.02, "x": 0.0})
        else:
            dets.append(d)
    dets = [d for d in dets if d.get("text", "").strip()]
    if not dets:
        return result

    ymax = max(d["y"] for d in dets) + 1.0
    bounds = _section_bounds(dets)

    # --- Sectioned label -> value-below extraction ---
    for field, section, variants, vtype in _FIELD_SPECS:
        y0, y1 = _section_range(section, bounds, ymax)
        in_section = [d for d in dets if y0 <= d["y"] < y1 and d.get("conf", 1) >= _MIN_CONF]
        label = next(
            (d for d in sorted(in_section, key=lambda d: (d["y"], d["x"]))
             if any(v in _norm(d["text"]) for v in variants)),
            None,
        )
        if not label:
            continue
        # Inline "Label: value" on the label line itself takes precedence.
        if ":" in label["text"]:
            inline = _extract_value(label["text"].split(":", 1)[1], vtype)
            if inline:
                result[field] = inline
                continue
        val = _value_below(label, dets, vtype)
        if val:
            result[field] = val

    # Metadata / cover / organization fields are genuine "Label: value" lines,
    # but often OCR-fragmented; the reconstructed layout re-joins them per row.
    _parse_inline_and_meta(reconstruct_layout(dets), dets, bounds, result)
    _parse_order_flags(dets, result)
    return result


def _parse_inline_and_meta(full, dets, bounds, result):
    """Header/cover/organization fields — these are genuine 'Label: value' lines."""
    prov_y = bounds["provider"]
    cover = [d for d in dets if prov_y is None or d["y"] < prov_y]

    def search(pattern, text=full, flags=re.IGNORECASE, group=1):
        m = re.search(pattern, text, flags)
        return m.group(group).strip() if m else None

    # Organization block (inline "Label: value")
    result["website"] = search(r"website\s*[:;]\s*(\S+)")
    org_email = search(rf"email\s*[:;]\s*({EMAIL})")
    if org_email:
        result["email"] = org_email

    # Two "Fax Number:" lines appear inline: cover then organization.
    fax_nums = re.findall(rf"fax\s*number\s*[:;]\s*({PHONE})", full, re.IGNORECASE)
    if fax_nums:
        result["fax_number"] = fax_nums[0].strip()
        if len(fax_nums) > 1:
            result["organization_fax"] = fax_nums[1].strip()
        else:
            result["organization_fax"] = fax_nums[0].strip()
    # Cover fax number is often OCR-fragmented as "(NNN)" + "NNN-NNNN".
    if not result["fax_number"]:
        area = next((re.search(r"\(\d{3}\)", d["text"]) for d in cover
                     if re.search(r"\(\d{3}\)", d["text"])), None)
        rest = next((re.search(r"\d{3}[-.\s]\d{4}", d["text"]) for d in cover
                     if re.search(r"\d{3}[-.\s]\d{4}", d["text"])), None)
        if area and rest:
            result["fax_number"] = f"{area.group(0)} {rest.group(0)}"

    # Support email (anywhere).
    result["support_email"] = next(
        (e for e in _EMAIL_RE.findall(full) if e.lower().startswith("support")), None
    )

    # Page X of Y
    m = re.search(r"page\s*[:;]?\s*(\d+)\s*of\s*(\d+)", full, re.IGNORECASE)
    if m:
        result["page_number"], result["total_pages"] = m.group(1), m.group(2)
    if not result["total_pages"]:
        result["total_pages"] = search(r"pages\s*[:;]\s*(\d+)")

    # Date / time (metadata date, not the patient DOB which lives in its section)
    result["fax_date"] = search(rf"({DATE})")
    t = search(rf"({TIME})")
    if t:
        result["fax_time"] = t.replace(";", ":")

    # Sender / recipient from the repeating page header row. Match "Fax:" but
    # not "Fax Number:" (the [:;] must follow "fax" directly).
    result["sender_name"] = search(r"from\s*[:;]\s*([^\n]+?)(?:\s+fax\b|\s+to\b|$)")
    header_faxes = [f.strip() for f in
                    re.findall(rf"\bfax\s*[:;]\s*({PHONE})", full, re.IGNORECASE)]
    if header_faxes:
        result["sender_fax"] = header_faxes[0]
        if len(header_faxes) > 1:
            result["recipient_fax"] = header_faxes[1]


def _parse_order_flags(dets, result):
    """Order-type checkboxes.

    NOTE: OCR yields the printed *text* of every option but not the checkbox
    tick, so this reports which options the form lists, not which were ticked.
    Reliable "which box is checked" detection needs the checkbox glyph, which
    text-only OCR does not capture — so all listed options read as True.
    """
    n = _norm(" ".join(d["text"] for d in dets))
    result["ct_with_calcium_scoring_and_cardiology_e_consult"] = "calcium scoring" in n
    result["ccta_with_cardiology_e_consult"] = "ccta with cardiology" in n
    result["ccta_ai_analysis_with_clearly"] = (
        "analysis with cleerly" in n or "analysis with clearly" in n or "ai analysis" in n
    )


def reconstruct_layout(detections: list[dict], width: int = 110) -> str:
    """Render detections back into column-aligned plain text using x/y.

    Used by the LLM parser so the model sees the form's real 2-D structure
    (labels above values, two-column lab grid) instead of a scrambled reading
    order. Falls back to newline-joined text if no geometry is present.
    """
    dets = [d for d in detections
            if not isinstance(d, str) and d.get("text", "").strip()
            and d.get("conf", 1) >= _MIN_CONF]
    if not dets:
        return "\n".join(
            (d if isinstance(d, str) else d.get("text", "")).strip()
            for d in detections
        ).strip()

    dets.sort(key=lambda d: (d["y"], d["x"]))
    rows, cur = [], []
    for d in dets:
        if cur and d["y"] - cur[0]["y"] > _ROW_DY:
            rows.append(cur)
            cur = []
        cur.append(d)
    if cur:
        rows.append(cur)

    lines = []
    for row in rows:
        line = ""
        for d in sorted(row, key=lambda d: d["x"]):
            col = int(d["x"] * width)
            if len(line) < col:
                line += " " * (col - len(line))
            elif line and not line.endswith(" "):
                line += " "
            line += d["text"].strip()
        lines.append(line.rstrip())
    return "\n".join(lines)
