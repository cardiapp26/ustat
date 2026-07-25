"""
Column-name suggestion: turn raw/cryptic headers into readable Sentence-case
labels the user can accept as a rename.

Examples:
    "RECORD NO"          -> "Record no"
    "blood_urea"         -> "Blood urea"
    "ejectionFraction"   -> "Ejection fraction"
    "LDL CATEGORIES"     -> "LDL categories"   (LDL kept as a known acronym)
    "SEX"                -> "Sex"
    "fu_days"            -> "Fu days"          (caller can edit in the modal)

The suggestion is advisory — the UI shows old → new and only renames on the
user's confirmation.
"""

import re

# Medical / clinical acronyms kept uppercase when they appear as a whole token.
_ACRONYMS = {
    "LDL", "HDL", "VLDL", "BMI", "BSA", "BP", "SBP", "DBP", "MAP",
    "HT", "HTN", "DM", "AF", "CAD", "CKD", "EF", "LVEF", "NYHA",
    "BUN", "GFR", "EGFR", "CRP", "ESR", "AST", "ALT", "ALP", "GGT",
    "TSH", "HBA1C", "INR", "PT", "APTT", "WBC", "RBC", "MCV", "RDW",
    "ACEI", "ARB", "BB", "CCB", "MRA", "PCI", "CABG", "MI", "HF",
    "ID", "TCHOL", "TG", "UA", "NA", "K", "CL", "CA",
}

# Tokens that should not be capitalised mid-sentence (sentence case).
_LOWER_STOP = {"no", "of", "per", "vs", "and", "or", "the", "at", "in", "to"}


def _split_tokens(name: str) -> list[str]:
    """Split a raw header into word tokens on separators and camelCase."""
    # Normalise separators to spaces.
    s = re.sub(r"[_\-.]+", " ", name.strip())
    # Insert a space at lower→Upper camelCase boundaries (ejectionFraction).
    s = re.sub(r"(?<=[a-zçğıöşü])(?=[A-ZÇĞİÖŞÜ])", " ", s)
    # Split a digit/letter boundary (LDL2 -> LDL 2, fu30 -> fu 30).
    s = re.sub(r"(?<=[A-Za-zÇĞİÖŞÜçğıöşü])(?=\d)", " ", s)
    s = re.sub(r"(?<=\d)(?=[A-Za-zÇĞİÖŞÜçğıöşü])", " ", s)
    return [t for t in s.split() if t]


def suggest_display_name(name: str) -> str:
    """Return a Sentence-case suggestion for a raw column name.

    First meaningful token is capitalised; known acronyms stay uppercase;
    everything else is lower-cased. Returns the original (trimmed) name when
    there is nothing sensible to suggest.
    """
    if not name or not name.strip():
        return name
    tokens = _split_tokens(name)
    if not tokens:
        return name.strip()

    out: list[str] = []
    for i, tok in enumerate(tokens):
        upper = tok.upper()
        if upper in _ACRONYMS:
            out.append(upper)
        elif i == 0:
            out.append(tok[:1].upper() + tok[1:].lower())
        elif tok.lower() in _LOWER_STOP:
            out.append(tok.lower())
        else:
            out.append(tok.lower())

    suggestion = " ".join(out)
    # Ensure the very first character is uppercase even if token 0 was a stop word.
    suggestion = suggestion[:1].upper() + suggestion[1:] if suggestion else suggestion
    return suggestion


def suggest_names(columns: list[str]) -> dict[str, str]:
    """Map each column name to its suggestion, omitting no-op suggestions."""
    out: dict[str, str] = {}
    for c in columns:
        s = suggest_display_name(c)
        if s and s != c:
            out[c] = s
    return out
