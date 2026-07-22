"""
3-tier dynamic heuristic parser.

Turns a list of OCR text lines (in reading order) into a structured fields dict.
Tiers, in order:
  1. Fuzzy label match — "Label: value" on the same line, or on the next line.
  2. Format heuristics — dates (classified by year), digit-density -> ID number,
     merged ALL-CAPS runs -> name / organization.
  3. Catch-all — anything left over lands in extra_fields (nothing is lost).
"""
import re
from datetime import date

from rapidfuzz import process, fuzz

# ---------------------------------------------------------------------------
# 3-tier heuristic parser
# ---------------------------------------------------------------------------
LABELS = {
    "name": ["name", "full name", "cardholder", "member name", "holder", "subscriber"],
    "id_number": ["id", "id no", "id number", "member id", "card no", "identification"],
    "dob": ["dob", "date of birth", "birth", "born"],
    "expiry": ["exp", "expiry", "expires", "expiration", "valid thru", "valid until"],
    "issue_date": ["issue", "issued", "date of issue", "eff", "effective"],
    "sex": ["sex", "gender"],
    "address": ["address", "addr", "residence"],
    "organization": ["organization", "org", "company", "employer", "issuer", "plan"],
    "group_number": ["group", "group no", "group number", "grp"],
}

DATE_RE = re.compile(
    r"\b("
    r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}"     # 03/15/1985, 15-06-24
    r"|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}"       # 1985-03-15
    r"|\d{1,2}\s?[A-Za-z]{3,9}\s?\d{2,4}"     # 12 Jan 2024
    r"|\d{1,2}[/\-.]\d{4}"                    # 03/2024, 01-2027 (MM/YYYY)
    r")\b"
)
LABEL_SPLIT_RE = re.compile(r"[:#]\s*")


def _fuzzy_label(token: str, threshold: int = 84):
    token = token.strip().lower()
    if not token or len(token) < 2:
        return None
    best_field, best_score = None, threshold
    for field, variants in LABELS.items():
        hit = process.extractOne(token, variants, scorer=fuzz.WRatio, score_cutoff=best_score)
        if hit and hit[1] >= best_score:
            best_field, best_score = field, hit[1]
    return best_field


def _digit_density(s: str) -> float:
    digits = sum(c.isdigit() for c in s)
    return digits / max(len(s), 1)


def _norm_id(s: str) -> str:
    """Uppercase alphabetic chars in an ID (OCR often lowercases, e.g. Y->y)."""
    return re.sub(r"[a-z]", lambda m: m.group(0).upper(), s.strip())


def _year_of(datestr: str) -> int | None:
    """Best-effort 4-digit year from a date string (for classification only)."""
    m4 = re.search(r"(?:19|20)\d{2}", datestr)
    if m4:
        return int(m4.group(0))
    groups = re.findall(r"\d{1,2}", datestr)  # e.g. 12/31/27 -> ['12','31','27']
    if groups:
        yy = int(groups[-1])
        if yy <= 99:
            full = 2000 + yy
            return full if full <= date.today().year + 20 else 1900 + yy
    return None


def _classify_date(datestr: str) -> str:
    """
    Route an UNLABELED date to a field by its year:
      - future            -> expiry
      - >= 16 years ago    -> dob (person old enough to hold a card)
      - recent past        -> issue_date (manufacture / effective)
    """
    y = _year_of(datestr)
    if y is None:
        return "issue_date"
    this_year = date.today().year
    if y > this_year:
        return "expiry"
    if this_year - y >= 16:
        return "dob"
    return "issue_date"


def _is_caps_line(s: str) -> bool:
    """All-caps alphabetic line (names/orgs printed on cards), tolerates
    spaces/dots. Also tolerant of OCR case noise: real-world OCR on
    all-caps print frequently mis-cases a handful of characters (e.g.
    "MULLINS" -> "MuLLINS"), so this only requires the line to be
    *mostly* uppercase rather than strictly uppercase — a genuine
    sentence-case line will still fail this easily."""
    letters = re.sub(r"[^A-Za-z]", "", s)
    if not letters or len(letters) < 2 or _digit_density(s) >= 0.2:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio >= 0.7


def parse_fields(lines: list[str]) -> dict:
    result: dict = {}
    extra_fields: dict = {}
    consumed = set()
    dates: list[str] = []

    # ---- Tier 1: fuzzy label match, same line ("Label: value") or next line
    for i, line in enumerate(lines):
        if i in consumed:
            continue
        parts = LABEL_SPLIT_RE.split(line, maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            field = _fuzzy_label(parts[0])
            if field and field not in result:
                val = parts[1].strip()
                dm = DATE_RE.search(val)
                if dm and field in ("dob", "expiry", "issue_date"):
                    val = dm.group(0)
                elif field == "id_number":
                    val = _norm_id(val)
                result[field] = val
                consumed.add(i)
                continue
        field = _fuzzy_label(line)
        if field and field not in result and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if nxt and not _fuzzy_label(nxt):
                result[field] = _norm_id(nxt) if field == "id_number" else nxt
                consumed.update({i, i + 1})

    # ---- Tier 2a: unlabeled dates -> capture ALL on a line, classify by year
    for i, line in enumerate(lines):
        if i in consumed:
            continue
        found = DATE_RE.findall(line)
        if not found:
            continue
        for d in found:
            dates.append(d)
            field = _classify_date(d)
            if field == "expiry":  # keep the latest future date
                result[field] = d
            elif field not in result:  # first dob / issue_date wins
                result[field] = d
        consumed.add(i)

    # ---- Tier 2b: high digit-density -> ID number (uppercase OCR'd letters)
    for i, line in enumerate(lines):
        if i in consumed:
            continue
        s = line.strip()
        if _digit_density(s) >= 0.5 and len(re.sub(r"\D", "", s)) >= 5 and "id_number" not in result:
            result["id_number"] = _norm_id(s)
            consumed.add(i)

    # ---- Tier 2c: merge consecutive ALL-CAPS lines -> name, then organization
    i = 0
    while i < len(lines):
        if i in consumed or not _is_caps_line(lines[i].strip()):
            i += 1
            continue
        run, j = [], i
        while j < len(lines) and j not in consumed and _is_caps_line(lines[j].strip()):
            run.append(lines[j].strip())
            j += 1
        block = " ".join(run).strip()
        tokens = block.split()
        if 1 <= len(tokens) <= 6:
            if "name" not in result:
                result["name"] = block
            elif "organization" not in result:
                result["organization"] = block
            else:
                extra_fields[f"line_{i}"] = block
            consumed.update(range(i, j))
        i = j

    # ---- Tier 3: catch-all -> extra_fields
    for i, line in enumerate(lines):
        if i in consumed:
            continue
        s = line.strip()
        if s:
            extra_fields[f"line_{i}"] = s

    if dates:
        result["dates"] = dates
    result["extra_fields"] = extra_fields
    return result
