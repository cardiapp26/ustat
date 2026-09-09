"""
What import did to the file, said out loud.

THE BUG THIS EXISTS TO CLOSE. A column that was at least 98% numeric had its
remaining cells rewritten to NaN and nothing said so. For dirty exports that is
right: ``"NA"``, ``"?"`` and ``"."`` mean missing and should not pin a whole
column to text. For a clinical measurement it is data loss. ``<0.1`` is not a
missing value; it is a result -- the analyte was below the assay's limit of
quantification, which is a *number the reader knows something about*. Turning
it into a blank moves that row from "measured, below the limit" to "never
measured", and the count of observed values silently drops by one.

So this module splits the values a column cannot parse into kinds, and lets the
kind decide:

  missing code   "NA", "n/a", "?", "." ...   mean missing; map to NaN, report
                                             which codes and how many
  censored       "<0.1", ">= 50", "≤ 2"      a measurement limit; the column is
                                             NOT coerced, because substituting
                                             at the limit (or dropping the row)
                                             is a methodological choice that
                                             belongs to whoever is writing the
                                             paper, not to the importer
  unit-bearing   "12 mg/dL", "37 °C"         a number plus a unit; the column is
                                             NOT coerced, for the same reason --
                                             stripping the unit assumes every
                                             cell shares it, and mixed mg/L and
                                             mg/dL in one column is a real and
                                             silent tenfold error
  unparseable    "n.d.", "see notes"         genuinely unrecognised; the cell
                                             does become NaN, but its text is
                                             kept verbatim in the report so the
                                             original is recoverable

Every decision, count and example lands in a report the caller stores with the
session and shows the user. Nothing is dropped without a line saying so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

#: Text values that mean "missing" in dirty CSV/SPSS/SAS exports.
TEXT_MISSING = frozenset({"", "na", "n/a", "?", "-", ".", "null", "missing", "none"})

#: Coverage threshold for the "almost-all numeric, a few text" case.
NUMERIC_THRESHOLD = 0.98

#: ``0123`` and friends: identifier codes, never arithmetic.
LEADING_ZERO_RE = re.compile(r"^0\d")

#: ``<0.1``, ``> 50``, ``≤2,5``. The operator is what makes it a limit rather
#: than a number, so it is captured and reported separately from the value.
CENSOR_RE = re.compile(r"^(<=|>=|<|>|≤|≥|≦|≧)\s*(-?\d+(?:[.,]\d+)?)$")

#: ``12 mg/dL``, ``37°C``, ``98 %``. The suffix must carry no digits, so an ISO
#: date or a hyphenated code cannot be mistaken for a united measurement.
UNIT_RE = re.compile(r"^(-?\d+(?:[.,]\d+)?)\s*([A-Za-zµμ%°][A-Za-zµμ%°/·^\-]*)$")

#: Verbatim originals are kept for cells that would otherwise be lost. The cap
#: bounds a hostile file's memory: a coerced column is >=98% numeric by
#: construction, so a real dataset never comes near it.
MAX_PRESERVED_CELLS = 50_000

#: How many examples of each finding a report carries. Enough to recognise the
#: pattern, few enough that the payload stays a report and not a second copy.
MAX_EXAMPLES = 5


@dataclass
class ColumnReport:
    """What happened to one column, and why."""

    column: str
    decision: str                       # "numeric" | "kept_text"
    reason: str
    n_values: int = 0                   # non-blank cells examined
    n_changed: int = 0                  # cells whose stored value != file text
    n_missing_coded: int = 0
    n_discarded: int = 0                # became NaN and were NOT a missing code
    decimal_separator: Optional[str] = None
    missing_codes: Dict[str, int] = field(default_factory=dict)
    discarded_examples: List[dict] = field(default_factory=list)
    censored: Optional[dict] = None
    units: Optional[dict] = None

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "column": self.column,
            "decision": self.decision,
            "reason": self.reason,
            "n_values": self.n_values,
            "n_changed": self.n_changed,
            "n_missing_coded": self.n_missing_coded,
            "n_discarded": self.n_discarded,
            "decimal_separator": self.decimal_separator,
            "missing_codes": self.missing_codes,
            "discarded_examples": self.discarded_examples,
        }
        if self.censored:
            out["censored"] = self.censored
        if self.units:
            out["units"] = self.units
        return out


def _classify(text: str) -> Tuple[str, Any]:
    """One cell's kind, from the value alone.

    Order matters: a censoring operator is checked before the plain parse so
    ``<0.1`` can never be read as anything but a limit.
    """
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered in TEXT_MISSING:
        return "missing_code", stripped
    m = CENSOR_RE.match(stripped)
    if m:
        return "censored", (m.group(1), m.group(2))
    if pd.notna(pd.to_numeric(pd.Series([stripped]), errors="coerce").iloc[0]):
        return "numeric", stripped
    swapped = stripped.replace(",", ".")
    if pd.notna(pd.to_numeric(pd.Series([swapped]), errors="coerce").iloc[0]):
        return "numeric_comma", swapped
    m = UNIT_RE.match(stripped)
    if m:
        return "unit", (m.group(1), m.group(2))
    return "unparseable", stripped


def _tally(values: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _analyse_column(series: pd.Series) -> Tuple[ColumnReport, Optional[pd.Series], Dict[int, str]]:
    """Decide a single object column's fate.

    Returns the report, the coerced series (``None`` when the column is kept as
    text), and the verbatim originals of any cell the coercion would drop.
    """
    name = str(series.name)
    as_str = series.astype(str).str.strip()
    blank = series.isna()

    kinds: Dict[str, List[Any]] = {
        "numeric": [], "numeric_comma": [], "censored": [],
        "unit": [], "unparseable": [], "missing_code": [],
    }
    positions: Dict[str, List[int]] = {k: [] for k in kinds}
    for pos, (is_blank, raw) in enumerate(zip(blank.to_numpy(), as_str.to_numpy())):
        if is_blank:
            continue
        kind, payload = _classify(str(raw))
        kinds[kind].append(payload)
        positions[kind].append(pos)

    n_values = sum(len(v) for v in kinds.values())
    if n_values == 0:
        return ColumnReport(name, "kept_text", "no values to examine"), None, {}

    n_meaningful = n_values - len(kinds["missing_code"])
    report = ColumnReport(
        column=name,
        decision="kept_text",
        reason="",
        n_values=n_values,
        n_missing_coded=len(kinds["missing_code"]),
        missing_codes=_tally([str(v) for v in kinds["missing_code"]]),
    )

    def _examples(kind: str) -> List[dict]:
        return [
            {"row": positions[kind][i], "value": str(as_str.iloc[positions[kind][i]])}
            for i in range(min(MAX_EXAMPLES, len(positions[kind])))
        ]

    if kinds["censored"]:
        report.censored = {
            "n": len(kinds["censored"]),
            "operators": _tally([op for op, _ in kinds["censored"]]),
            "limits": sorted({float(str(v).replace(",", ".")) for _, v in kinds["censored"]}),
            "examples": _examples("censored"),
        }
    if kinds["unit"]:
        report.units = {
            "n": len(kinds["unit"]),
            "suffixes": _tally([suffix for _, suffix in kinds["unit"]]),
            "examples": _examples("unit"),
        }

    # A measurement limit or a unit is information. Coercing the column would
    # delete it, so the column is left exactly as the file had it and the user
    # is told what is in there.
    if kinds["censored"] or kinds["unit"]:
        parts = []
        if kinds["censored"]:
            parts.append(f"{len(kinds['censored'])} value(s) at a measurement limit (e.g. <, >)")
        if kinds["unit"]:
            parts.append(f"{len(kinds['unit'])} value(s) carrying a unit")
        report.reason = (
            "kept as text: " + " and ".join(parts) +
            ". Converting would have discarded them; decide how to handle them first."
        )
        return report, None, {}

    # Identifier codes with a leading zero are not arithmetic.
    if any(LEADING_ZERO_RE.match(str(v)) for v in kinds["numeric"] + kinds["unparseable"]):
        report.reason = "kept as text: values with a leading zero look like identifier codes"
        return report, None, {}

    parsed = len(kinds["numeric"]) + len(kinds["numeric_comma"])
    if n_meaningful == 0 or parsed / n_meaningful < NUMERIC_THRESHOLD:
        pct = round(parsed / n_meaningful * 100, 1) if n_meaningful else 0.0
        report.reason = (
            f"kept as text: only {pct}% of values parse as a number "
            f"(needs {NUMERIC_THRESHOLD * 100:.0f}%)"
        )
        # Nothing is discarded on this path -- the column is kept exactly as
        # the file had it -- so it stays out of the report, which is reserved
        # for changes the user has to know about.
        return report, None, {}

    # Coerce. Everything that is not a number becomes NaN -- and everything
    # that is lost by that is recorded first.
    swapped = as_str.where(~blank).str.replace(",", ".", regex=False)
    coerced = pd.to_numeric(swapped, errors="coerce")

    preserved: Dict[int, str] = {}
    for pos in positions["unparseable"][:MAX_PRESERVED_CELLS]:
        preserved[pos] = str(as_str.iloc[pos])

    report.decision = "numeric"
    report.decimal_separator = "," if kinds["numeric_comma"] else "."
    report.n_discarded = len(kinds["unparseable"])
    report.discarded_examples = _examples("unparseable")
    report.n_changed = (
        len(kinds["numeric_comma"]) + len(kinds["missing_code"]) + len(kinds["unparseable"])
    )
    bits = [f"converted to numeric from text ({parsed}/{n_meaningful} values parse)"]
    if kinds["numeric_comma"]:
        bits.append(f"{len(kinds['numeric_comma'])} comma-decimal(s) rewritten with a dot")
    if kinds["missing_code"]:
        bits.append(f"{len(kinds['missing_code'])} missing code(s) mapped to blank")
    if kinds["unparseable"]:
        bits.append(
            f"{len(kinds['unparseable'])} unrecognised value(s) blanked "
            "(originals kept in this report)"
        )
    report.reason = "; ".join(bits)
    return report, coerced, preserved


def coerce_with_report(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, Dict[str, Dict[int, str]]]:
    """Restore numeric dtype where it is safe, and say what that cost.

    Returns ``(frame, report, preserved)``. ``preserved`` maps column name to
    ``{row position: original text}`` for every cell the coercion blanked, so
    the raw value is recoverable even though the frame now holds a NaN.
    """
    out = df.copy()
    columns: List[dict] = []
    preserved_all: Dict[str, Dict[int, str]] = {}
    n_converted = 0
    n_cells_changed = 0
    n_cells_discarded = 0
    n_examined = 0

    for col in out.columns:
        if out[col].dtype != object:
            continue
        n_examined += 1
        report, coerced, preserved = _analyse_column(out[col])
        if coerced is not None:
            out[col] = coerced
            n_converted += 1
        if preserved:
            preserved_all[str(col)] = preserved
        n_cells_changed += report.n_changed
        n_cells_discarded += report.n_discarded
        # A clean text column that was always going to stay text has nothing to
        # report; a report full of those is a report nobody reads.
        if report.decision == "numeric" or report.censored or report.units or report.n_discarded:
            columns.append(report.as_dict())

    summary = {
        "n_columns_examined": n_examined,
        "n_columns_converted": n_converted,
        "n_cells_changed": n_cells_changed,
        "n_cells_discarded": n_cells_discarded,
        "needs_review": any(
            c.get("censored") or c.get("units") or c.get("n_discarded") for c in columns
        ),
        "columns": columns,
    }
    return out, summary, preserved_all
