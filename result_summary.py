"""
Compiled, frontend-ready `result` block + confidence.

This is purely a *view* layer over what the pipeline already produced — it does
no OCR, no parsing, no payer matching, and makes no API calls. It reorganizes
the flat `fields` dict into cardholder / coverage groups and attaches a single
confidence number that is meaningful for BOTH outcomes:

  - matched       -> the real payer-match score (surfaced as-is), basis
                     "exact_match" / "fuzzy_match".
  - partial_match -> a graded partial / alternate-name match (e.g. a
                     "BlueCare Plus" card ~ the alt-name "BlueCare"), basis
                     "partial_match". Score is the real similarity %.
  - not_found     -> a local, deterministic estimate of how confident we are
                     that the absence is genuine (basis "absence_estimate").
                     No API call.
"""

# match_type (from payer_matching) -> (result status, confidence_basis)
_MATCH_TYPE_MAP = {
    "exact": ("matched", "exact_match"),
    "fuzzy": ("matched", "fuzzy_match"),
    "partial": ("partial_match", "partial_match"),
}

# Absence-confidence tunables (kept together so the heuristic is easy to tune).
_ABSENCE_BASE = 50.0            # neutral starting point ("we're not sure")
_ABSENCE_FLOOR = 30.0           # never imply we're guessing randomly...
_ABSENCE_CEIL = 97.0           # ...never claim near-certain absence from a heuristic
_HIGH_CONF = 0.7               # avg OCR conf above this reads the card "well"
_LOW_CONF = 0.5                # ...below this, we probably misread
_FEW_LINES = 4                 # fewer kept lines than this -> we may not have read enough
_MANY_LINES = 8                # a well-populated read
_SMALL_EXTRA = 2               # <= this many leftover lines -> most text got classified
_LARGE_EXTRA = 5               # a lot left uninspected


def estimate_absence_confidence(
    clean_lines: list[str],
    ocr_confidence: list[dict],
    fields: dict,
) -> float:
    """
    Deterministic, local estimate (0-ish..100) of how confident we are that a
    "not found" is a *genuine* absence rather than a failure to read the card.

    Strongest positive signal: we did extract an `organization`/payer-ish name
    and it still didn't match. Strongest negative signal: we barely read the
    card (few, low-confidence lines), so we can't really tell.
    """
    confs = [d.get("conf", 0.0) for d in (ocr_confidence or [])]
    n = len(confs)
    avg_conf = (sum(confs) / n) if n else 0.0
    extra = fields.get("extra_fields") or {}
    has_org = bool(fields.get("organization"))

    score = _ABSENCE_BASE

    # + We found an org/payer-looking name and it still didn't match -> the
    #   card is most likely just not in the database. Strongest signal.
    #   Missing it means we may not have captured the payer name at all.
    score += 20 if has_org else -10

    # + The OCR read the card well, so a non-match is more likely to be real.
    if n:
        if avg_conf > _HIGH_CONF:
            score += 15
        elif avg_conf < _LOW_CONF:
            score -= 15

    # +/- How much text did we actually get to inspect?
    if n < _FEW_LINES:
        score -= 15
    elif n >= _MANY_LINES:
        score += 5

    # + Most of what we read got classified into known fields (little left over).
    if len(extra) <= _SMALL_EXTRA:
        score += 10
    elif len(extra) >= _LARGE_EXTRA:
        score -= 10

    return round(max(_ABSENCE_FLOOR, min(_ABSENCE_CEIL, score)), 1)


def build_result_summary(
    fields: dict,
    payer,
    refinement_status: str,
    clean_lines: list[str],
    ocr_confidence: list[dict],
) -> dict:
    """
    Assemble the single compiled `result` block. `fields` is the flat parsed
    dict (parse_fields output, possibly augmented by refinement); `payer` is the
    best payer candidate — a strict match_payer(...) hit OR a graded partial /
    alternate-name candidate (see payer_matching.resolve_payer_candidate) — or
    None when nothing came close.

    Every field with no extracted value is `null` (not omitted) so the shape is
    predictable for frontend consumption.
    """
    if payer is not None:
        status, confidence_basis = _MATCH_TYPE_MAP.get(
            payer.get("match_type"), ("matched", "fuzzy_match")
        )
        confidence = payer["score"]
        payer_block = {
            "payer_id": payer["payer_id"],
            "payer_name": payer["payer_name"],
        }
    else:
        status = "not_found"
        confidence = estimate_absence_confidence(clean_lines, ocr_confidence, fields)
        confidence_basis = "absence_estimate"
        payer_block = None

    return {
        "status": status,
        "confidence": confidence,
        "confidence_basis": confidence_basis,
        "payer": payer_block,
        "cardholder": {
            "name": fields.get("name"),
            "id_number": fields.get("id_number"),
            "dob": fields.get("dob"),
            "sex": fields.get("sex"),
            "address": fields.get("address"),
        },
        "coverage": {
            "organization": fields.get("organization"),
            "group_number": fields.get("group_number"),
            "issue_date": fields.get("issue_date"),
            "expiry": fields.get("expiry"),
            "dates": fields.get("dates"),
        },
        "unclassified_fields": fields.get("extra_fields"),
        "refinement": refinement_status,
    }
