"""
Pre-indexed rapidfuzz payer lookup (sub-ms), strict to avoid false positives.

On import, every payer_name + alt_name from payers_data.py is flattened into ONE
normalized lookup map for O(1) exact matching, plus a fuzzy fallback that is
deliberately strict (multi-word only, length-comparability guard) so a substring
on a card can't spuriously match a different payer's alt-name.
"""
import re

from rapidfuzz import process, fuzz

from payers_data import PAYERS_DATA

# ---------------------------------------------------------------------------
# Payer pre-index: flatten every payer_name + alt_name into ONE lookup map.
# key = UPPERCASED name string -> {"payer_id", "payer_name"} (canonical)
# ---------------------------------------------------------------------------
PAYER_INDEX: dict[str, dict] = {}


def build_payer_index():
    for p in PAYERS_DATA:
        canonical = {"payer_id": p.get("payer_id"), "payer_name": p.get("payer_name")}
        names = [p.get("payer_name", "")] + list(p.get("alt_names") or [])
        for n in names:
            if not n:
                continue
            key = n.strip().upper()
            if len(key) < 4:  # skip tiny/ambiguous keys (e.g. "CAS", "PPO")
                continue
            PAYER_INDEX.setdefault(key, canonical)


build_payer_index()
PAYER_CHOICES = list(PAYER_INDEX.keys())

_ALNUM_RE = re.compile(r"[^A-Z0-9 ]")
_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    """Normalize to uppercase alphanumeric + single spaces for comparison."""
    return _WS_RE.sub(" ", _ALNUM_RE.sub(" ", s.upper())).strip()


# normalized-key index for O(1) exact matching (punctuation-insensitive)
NORM_INDEX: dict[str, dict] = {}
for _k, _v in PAYER_INDEX.items():
    NORM_INDEX.setdefault(_norm(_k), _v)
NORM_CHOICES = list(NORM_INDEX.keys())


def match_payer(lines: list[str], score_cutoff: int = 90):
    """
    Cross-reference OCR lines against the payer map.

    1. Exact (normalized) hit wins instantly — even short names like "AARP".
    2. Otherwise a fuzzy pass with token_sort_ratio (full-string, length-
       sensitive) so a substring on the card ("BlueCarePlus") does NOT falsely
       match a different payer's alt-name ("BlueCare"). Length-comparability
       guard blocks long-vs-tiny spurious hits. Unknown cards return None.
    """
    best = None  # (matched_key, score)
    for raw in lines:
        q = _norm(raw)
        if len(q) < 4:
            continue
        # --- Tier 1: exact normalized match ---
        if q in NORM_INDEX:
            return _payer_result(NORM_INDEX[q], q, 100.0, "exact")
        # --- Tier 2: fuzzy ---
        # Only multi-word lines may fuzzy-match: a single common word like
        # "MEDICAL" must NOT fuzzy-hit a one-word payer ("Medica"). Real payer
        # names on cards are multi-word; single tokens must match exactly above.
        if len(q.split()) < 2:
            continue
        if sum(c.isalpha() for c in q) < 5:
            continue
        hit = process.extractOne(
            q, NORM_CHOICES, scorer=fuzz.token_sort_ratio, score_cutoff=score_cutoff
        )
        if not hit:
            continue
        ratio = len(q) / max(len(hit[0]), 1)
        if ratio < 0.6 or ratio > 1.7:
            continue
        if best is None or hit[1] > best[1]:
            best = (hit[0], hit[1])
    if not best:
        return None
    return _payer_result(NORM_INDEX[best[0]], best[0], round(best[1], 1), "fuzzy")


def _payer_result(
    canonical: dict, matched_on: str, score: float, match_type: str
) -> dict:
    # `match_type` ("exact" | "fuzzy" | "partial") records which path produced
    # the hit, so downstream code can label confidence basis without re-deriving
    # it from the score (a fuzzy token-sort match can also score 100.0). Additive
    # metadata — the strict matching logic and score are unchanged.
    return {
        "payer_id": canonical["payer_id"],
        "payer_name": canonical["payer_name"],
        "matched_on": matched_on,
        "score": score,
        "match_type": match_type,
    }


# ---------------------------------------------------------------------------
# Partial / alternate-name matching (graded, deliberately NOT part of the strict
# match_payer path above — kept separate so fields.payer_match stays strict).
#
# A card often prints a *superset* of a known payer_name or alt_name — e.g. a
# "BlueCare Plus" card vs the alt-name "BlueCare", or "Blue Shield of California
# PPO" vs "Blue Shield of California". The strict path rejects these on purpose
# (to avoid confident false positives), so they'd otherwise read as a flat "not
# found" with no indication the card is *nearly* a known payer.
#
# partial_payer_candidate() surfaces them as a clearly-labelled "partial" match
# with a real similarity percentage. Ranking uses token_set_ratio with a very
# high cutoff — near-complete containment of a known name. That high bar is what
# separates a genuine superset ("BlueCare Plus" ⊃ "BlueCare", set-ratio 100)
# from a coincidental generic-word overlap ("MEDICARE HEALTH INSURANCE" vs
# "WPS HEALTH INSURANCE", set-ratio ~88, which we reject). The same length-
# comparability guard as the strict path blocks tiny-substring hits. The
# *reported* score is the length-sensitive token_sort_ratio, so extra words on
# the card correctly pull it below 100 ("somewhat matches").
# ---------------------------------------------------------------------------
PARTIAL_SET_CUTOFF = 95  # token_set_ratio bar: near-complete containment only


def partial_payer_candidate(lines: list[str]):
    best = None  # (matched_key, display_score)
    for raw in lines:
        q = _norm(raw)
        # Same guards as the strict fuzzy path: multi-word, enough letters.
        if len(q) < 4 or len(q.split()) < 2 or sum(c.isalpha() for c in q) < 5:
            continue
        for key, set_score, _ in process.extract(
            q, NORM_CHOICES, scorer=fuzz.token_set_ratio, limit=15
        ):
            if set_score < PARTIAL_SET_CUTOFF:
                break  # process.extract is sorted desc — nothing better follows
            ratio = len(q) / max(len(key), 1)
            if ratio < 0.6 or ratio > 1.7:  # length-comparability guard
                continue
            disp = fuzz.token_sort_ratio(q, key)  # length-sensitive "somewhat" %
            if best is None or disp > best[1]:
                best = (key, disp)
    if not best:
        return None
    return _payer_result(NORM_INDEX[best[0]], best[0], round(best[1], 1), "partial")


def resolve_payer_candidate(lines: list[str], strict=None):
    """
    Best payer for the compiled `result` block: the strict match_payer() hit if
    there is one, otherwise a graded partial / alternate-name candidate,
    otherwise None. Pass the already-computed strict result in as `strict` to
    avoid re-running it. This never affects `fields.payer_match`, which stays
    strict-only for backward compatibility.
    """
    if strict is None:
        strict = match_payer(lines)
    if strict:
        return strict
    return partial_payer_candidate(lines)
