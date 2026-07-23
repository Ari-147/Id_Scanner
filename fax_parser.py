"""
Regex-based medical-referral-fax parser.

Given the OCR text lines of a fax, pull out a flat set of well-known fields
using labelled-line patterns ("Label: value"). This is the zero-cost baseline
parser; the LLM variant (fax_llm.py) targets the same field set so both
`/extract-data-v1` (regex) and `/extract-data-v2` (Claude) return an identical
shape to the frontend.
"""
import re

# ---------------------------------------------------------------------------
# Canonical field set — the single source of truth shared with fax_llm.py.
# Order here is the order the frontend renders sections in.
# ---------------------------------------------------------------------------
FAX_FIELDS = [
    # Fax metadata
    "sender_name",
    "sender_fax",
    "recipient_name",
    "recipient_fax",
    "page_number",
    "total_pages",
    "fax_date",
    "fax_time",
    # Cover page
    "note",
    "fax_number",
    # Organization
    "website",
    "email",
    "organization_fax",
    # Referring provider
    "provider_name",
    "provider_email",
    "provider_phone",
    "provider_fax",
    "npi",
    # Patient
    "patient_name",
    "date_of_birth",
    "patient_phone",
    "patient_email",
    # Order
    "order_type",
    "ct_with_calcium_scoring_and_cardiology_e_consult",
    "ccta_with_cardiology_e_consult",
    "ccta_ai_analysis_with_clearly",
    # Laboratory values
    "recent_creatinine",
    "ldl",
    "hdl",
    "total_cholesterol",
    "triglycerides",
    "apo_a",
    "apo_b",
    "esr",
    "crp",
    # Support
    "support_email",
]

# Fields that are a yes/no presence check rather than an extracted string.
FAX_BOOLEAN_FIELDS = {
    "ct_with_calcium_scoring_and_cardiology_e_consult",
    "ccta_with_cardiology_e_consult",
    "ccta_ai_analysis_with_clearly",
}

# ---------------------------------------------------------------------------
# Regex building blocks
# ---------------------------------------------------------------------------
EMAIL = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
PHONE = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
FAX = PHONE
DATE = r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
TIME = r"\d{1,2}:\d{2}(?:\s?[AP]M)?"
NPI = r"\d{10}"

# Per-field patterns. Each pattern's LAST capturing group is taken as the value.
_PATTERNS = {
    # Fax metadata
    "sender_name": r"Sender\s*(?:Name)?\s*:\s*(.+)",
    "sender_fax": rf"Sender\s*Fax\s*:\s*({FAX})",
    "recipient_name": r"(?:Recipient|To)\s*(?:Name)?\s*:\s*(.+)",
    "recipient_fax": rf"(?:Recipient\s*Fax|To\s*Fax)\s*:\s*({FAX})",
    "page_number": r"Page\s*:?\s*(\d+)",
    "total_pages": r"(?:Total\s*Pages|of)\s*:?\s*(\d+)",
    "fax_date": rf"(?:Fax\s*Date|Date)\s*:\s*({DATE})",
    "fax_time": rf"(?:Fax\s*Time|Time)\s*:\s*({TIME})",

    # Cover page
    "note": r"(?:Note|Message)\s*:\s*(.+)",
    "fax_number": rf"Fax\s*(?:Number)?\s*:\s*({FAX})",

    # Organization
    "website": r"(?:Website|Web)\s*:\s*(\S+)",
    "email": rf"Email\s*:\s*({EMAIL})",
    "organization_fax": rf"(?:Organization\s*Fax|Fax)\s*:\s*({FAX})",

    # Referring provider
    "provider_name": r"(?:Referring\s*Provider|Provider|Physician|Doctor)\s*:\s*(.+)",
    "provider_email": rf"(?:Provider\s*Email|Email)\s*:\s*({EMAIL})",
    "provider_phone": rf"(?:Provider\s*Phone|Phone)\s*:\s*({PHONE})",
    "provider_fax": rf"(?:Provider\s*Fax|Fax)\s*:\s*({FAX})",
    "npi": rf"NPI\s*:\s*({NPI})",

    # Patient
    "patient_name": r"(?:Patient\s*Name|Patient|Name)\s*:\s*(.+)",
    "date_of_birth": rf"(?:DOB|Date\s*of\s*Birth)\s*:\s*({DATE})",
    "patient_phone": rf"(?:Patient\s*Phone|Phone)\s*:\s*({PHONE})",
    "patient_email": rf"(?:Patient\s*Email|Email)\s*:\s*({EMAIL})",

    # Order
    "order_type": r"(?:Order\s*Type|Exam)\s*:\s*(.+)",
    "ct_with_calcium_scoring_and_cardiology_e_consult":
        r"CT.*Calcium.*Scoring.*Cardiology.*Consult",
    "ccta_with_cardiology_e_consult": r"CCTA.*Cardiology.*Consult",
    "ccta_ai_analysis_with_clearly": r"CCTA.*AI.*Clearly",

    # Laboratory values
    "recent_creatinine": r"(?:Recent\s*Creatinine|Creatinine)\s*:\s*(.+)",
    "ldl": r"LDL\s*:\s*(.+)",
    "hdl": r"HDL\s*:\s*(.+)",
    "total_cholesterol": r"(?:Total\s*Cholesterol|Cholesterol)\s*:\s*(.+)",
    "triglycerides": r"Triglycerides\s*:\s*(.+)",
    "apo_a": r"Apo\s*A\s*:\s*(.+)",
    "apo_b": r"Apo\s*B\s*:\s*(.+)",
    "esr": r"ESR\s*:\s*(.+)",
    "crp": r"CRP\s*:\s*(.+)",

    # Support
    "support_email": rf"(?:Support\s*Email|Support)\s*:\s*({EMAIL})",
}


def _blank_result() -> dict:
    """A full result with every field present (None / False), so the shape is
    stable regardless of what the OCR actually contained."""
    return {f: (False if f in FAX_BOOLEAN_FIELDS else None) for f in FAX_FIELDS}


def parser_fax_with_regex(ocr_lines: list[str]) -> dict:
    """Extract the canonical fax field set from OCR lines using regex."""
    full_text = "\n".join(ocr_lines)
    result = _blank_result()

    for field, pattern in _PATTERNS.items():
        if field in FAX_BOOLEAN_FIELDS:
            # Presence check — the order-type keywords may be spread over
            # several lines, so match across newlines with DOTALL.
            result[field] = bool(
                re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
            )
            continue

        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            value = match.groups()[-1]
            if value:
                result[field] = value.strip()

    return result
