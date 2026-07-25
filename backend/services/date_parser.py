"""Robust date parsing for mixed-format columns (SPSS/Excel paste & import).

Ported from the drtr Excel date-repair tool. Recognises:
  * numeric separators — 15.03.2024, 15/03/2024, 15-03-2024, 2024-03-15
  * Turkish & English month names — "5 Ocak 2024", "Jan 2, 2022"
  * Excel serial numbers — 45000 → 2023-07-18 (1899-12-30 epoch, 1900 leap bug)
  * 2-digit years — resolved against a century threshold (default 50)
  * DMY/MDY ambiguity — resolved per-column by scanning every row

Pure / immutable: callers pass a Series, get a new datetime64 Series + stats.
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import Optional, Tuple

import pandas as pd

# Turkish month names incl. ASCII fallbacks (subat/agustos/…) and 3-letter abbrevs.
_TR_MONTHS = {
    "ocak": 1, "şubat": 2, "subat": 2, "mart": 3, "nisan": 4, "mayıs": 5, "mayis": 5,
    "haziran": 6, "temmuz": 7, "ağustos": 8, "agustos": 8, "eylül": 9, "eylul": 9,
    "ekim": 10, "kasım": 11, "kasim": 11, "aralık": 12, "aralik": 12,
    "oca": 1, "şub": 2, "sub": 2, "mar": 3, "nis": 4, "may": 5, "haz": 6, "tem": 7,
    "ağu": 8, "agu": 8, "eyl": 9, "eki": 10, "kas": 11, "ara": 12,
}
_EN_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_INT_RE = re.compile(r"^-?\d+(\.\d+)?$")
_ISO_RE = re.compile(r"^(\d{4})[\-/.](\d{1,2})[\-/.](\d{1,2})\b")
_DMY_RE = re.compile(r"^(\d{1,2})[\-/. ](\d{1,2})[\-/. ](\d{2,4})\b")
_DAY_MON_YEAR = re.compile(r"^(\d{1,2})\s+([a-zçşğıöü]+)\.?\s+(\d{2,4})\b")
_MON_DAY_YEAR = re.compile(r"^([a-zçşğıöü]+)\.?\s+(\d{1,2})\s+(\d{2,4})\b")

_EXCEL_EPOCH = _dt.date(1899, 12, 30)  # accounts for the Excel 1900 leap-year bug


def _century(yy: int, threshold: int) -> int:
    return 2000 + yy if yy <= threshold else 1900 + yy


def _valid(y: int, m: int, d: int) -> bool:
    if not (1 <= m <= 12 and 1 <= d <= 31 and 1900 <= y <= 2999):
        return False
    try:
        _dt.date(y, m, d)
        return True
    except ValueError:
        return False


def parse_one(raw, threshold: int = 50) -> Optional[dict]:
    """Parse a single value → {y, mo, d, ambig[, alt]} or None.

    `ambig` marks a numeric date that is valid under both DMY and MDY; `alt`
    carries the MDY interpretation so a whole-column scan can disambiguate.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = re.sub(r"\s+", " ", s).replace(",", " ").strip()

    # Excel serial (pure number)
    if _INT_RE.match(s):
        n = float(s)
        if 59 < n < 200000:
            dt = _EXCEL_EPOCH + _dt.timedelta(days=int(n))
            return {"y": dt.year, "mo": dt.month, "d": dt.day, "ambig": False}
        return None

    m = _ISO_RE.match(s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _valid(y, mo, d):
            return {"y": y, "mo": mo, "d": d, "ambig": False}

    m = _DMY_RE.match(s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        y = int(m.group(3))
        if y < 100:
            y = _century(y, threshold)
        dmy_ok = _valid(y, b, a)
        mdy_ok = _valid(y, a, b)
        if dmy_ok and mdy_ok:
            return {"y": y, "mo": b, "d": a, "ambig": True, "alt": {"y": y, "mo": a, "d": b}}
        if dmy_ok:
            return {"y": y, "mo": b, "d": a, "ambig": False}
        if mdy_ok:
            return {"y": y, "mo": a, "d": b, "ambig": False}
        return None

    low = s.lower()
    mt = _DAY_MON_YEAR.match(low)
    if mt:
        d = int(mt.group(1))
        mo = _TR_MONTHS.get(mt.group(2)) or _EN_MONTHS.get(mt.group(2))
        y = int(mt.group(3))
        if mo:
            if y < 100:
                y = _century(y, threshold)
            if _valid(y, mo, d):
                return {"y": y, "mo": mo, "d": d, "ambig": False}
    mt = _MON_DAY_YEAR.match(low)
    if mt:
        mo = _TR_MONTHS.get(mt.group(1)) or _EN_MONTHS.get(mt.group(1))
        d = int(mt.group(2))
        y = int(mt.group(3))
        if mo:
            if y < 100:
                y = _century(y, threshold)
            if _valid(y, mo, d):
                return {"y": y, "mo": mo, "d": d, "ambig": False}
    return None


def parse_series(
    series: pd.Series, order: str = "auto", threshold: int = 50
) -> Tuple[pd.Series, dict]:
    """Parse a whole column → (datetime64 Series, stats dict).

    order: 'auto' (scan to pick DMY vs MDY), 'dmy', or 'mdy'.
    stats: n_total, n_ok, n_bad, n_empty, order_used.
    """
    raws = list(series)
    parsed = [parse_one(r, threshold) for r in raws]

    if order == "dmy":
        order_used = "dmy"
        resolved = [
            (None if p is None else {"y": p["y"], "mo": p["mo"], "d": p["d"]}) for p in parsed
        ]
    elif order == "mdy":
        order_used = "mdy"
        resolved = [
            (None if p is None else (p["alt"] if p["ambig"] else {"y": p["y"], "mo": p["mo"], "d": p["d"]}))
            for p in parsed
        ]
    else:
        # auto: any ambiguous row whose DMY day > 12 forces DMY; whose month > 12 forces MDY.
        dmy_sig = sum(1 for p in parsed if p and p["ambig"] and p["d"] > 12)
        mdy_sig = sum(1 for p in parsed if p and p["ambig"] and p["mo"] > 12)
        use_dmy = dmy_sig >= mdy_sig
        order_used = "dmy" if use_dmy else "mdy"
        resolved = []
        for p in parsed:
            if p is None:
                resolved.append(None)
            elif not p["ambig"]:
                resolved.append({"y": p["y"], "mo": p["mo"], "d": p["d"]})
            elif use_dmy:
                resolved.append({"y": p["y"], "mo": p["mo"], "d": p["d"]})
            else:
                resolved.append(p["alt"])

    out, n_ok, n_bad, n_empty = [], 0, 0, 0
    for raw, p in zip(raws, resolved):
        is_empty = raw is None or (isinstance(raw, float) and pd.isna(raw)) or str(raw).strip() == ""
        if is_empty:
            out.append(pd.NaT)
            n_empty += 1
        elif p is None:
            out.append(pd.NaT)
            n_bad += 1
        else:
            out.append(pd.Timestamp(p["y"], p["mo"], p["d"]))
            n_ok += 1

    ser = pd.Series(out, index=series.index, dtype="datetime64[ns]")
    stats = {
        "n_total": len(raws),
        "n_ok": n_ok,
        "n_bad": n_bad,
        "n_empty": n_empty,
        "order_used": order_used,
    }
    return ser, stats
