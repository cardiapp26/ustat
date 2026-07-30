"""
Compute / Create New Variable router.

Endpoints
---------
POST /{session_id}/formula          — formula builder via pandas df.eval()
POST /{session_id}/transform        — single-column math transforms (log, sqrt, …)
POST /{session_id}/recode           — IF-THEN rule builder via numpy np.select()
POST /{session_id}/clinical/{calc}  — preset clinical calculators (BMI, eGFR, CHA₂DS₂-VASc)
DELETE /{session_id}/column/{col}   — remove a computed (or any) column from session
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import store
from services.dirty_value_guard import coerce_numeric, flag_sentinels, mask_sentinels, plausibility_max_for_column, sentinel_values

router = APIRouter()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _col_kind(series: pd.Series) -> str:
    from routers.upload import _detect_kind
    return _detect_kind(series)


def _jsonable_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _build_result(df: pd.DataFrame, col: str) -> dict:
    """Build the standard response dict after adding a new column."""
    series = df[col]
    preview_vals = [_jsonable_value(v) for v in series.head(2000).tolist()]
    return {
        "name": col,
        "dtype": str(series.dtype),
        "kind": _col_kind(series),
        "preview_values": preview_vals,
        "n_computed": int(series.notna().sum()),
        "n_missing": int(series.isna().sum()),
    }


def _validate_col_name(new_col: str):
    if not new_col or not new_col.strip():
        raise HTTPException(status_code=422, detail="New column name cannot be empty")
    return new_col.strip()


def _quantile_groups(col: pd.Series, q: int) -> pd.Series:
    """Return 1-based quantile groups while preserving missing source values."""
    result = pd.Series(np.nan, index=col.index, dtype="float64")
    valid = col.dropna()
    if valid.empty:
        raise HTTPException(status_code=422, detail="No numeric values available for this transform")
    if valid.nunique(dropna=True) < 2:
        raise HTTPException(status_code=422, detail="Need at least two distinct numeric values for quantile grouping")

    try:
        grouped = pd.qcut(valid, q=q, labels=False, duplicates="drop")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Could not create quantile groups: {exc}")

    if grouped.notna().sum() == 0:
        raise HTTPException(status_code=422, detail="Could not create quantile groups from this column")
    result.loc[grouped.index] = grouped.astype(float) + 1
    return result


# ── 1. Formula Builder ────────────────────────────────────────────────────────

class FormulaRequest(BaseModel):
    formula: str
    new_col: str


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _resolve_awkward_column_names(
    df: pd.DataFrame, formula: str, names: dict
) -> tuple:
    """Let formulas reference columns whose names aren't Python identifiers.

    Two ways in, both mapped onto generated placeholders the parser accepts:
      1. Backticks — ```p wave mv_2` / 2`` (explicit, always wins).
      2. The bare name — ``p wave mv_2 / 2``. Matched longest-first so that a
         dataset with both "p wave" and "p wave mv_2" resolves the longer one.
    Returns the rewritten formula plus the extended name map.
    """
    names = dict(names)
    counter = 0

    def _placeholder(col: str) -> str:
        nonlocal counter
        key = f"_col_{counter}_"
        counter += 1
        names[key] = df[col]
        return key

    # 1. Backtick-quoted names.
    def _sub_quoted(match: "re.Match") -> str:
        col = match.group(1)
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
        return _placeholder(col)

    formula = re.sub(r"`([^`]+)`", _sub_quoted, formula)

    # 2. Bare names that can't be parsed as identifiers. Longest first so a
    #    shorter column name can't shadow a longer one that contains it.
    awkward = sorted(
        (c for c in df.columns if not _IDENTIFIER_RE.match(str(c))),
        key=lambda c: len(str(c)),
        reverse=True,
    )
    for col in awkward:
        pattern = re.compile(
            r"(?<![A-Za-z0-9_])" + re.escape(str(col)) + r"(?![A-Za-z0-9_])"
        )
        if pattern.search(formula):
            formula = pattern.sub(lambda _m, c=col: _placeholder(c), formula)

    return formula, names


def _syntax_error_help(df: pd.DataFrame, formula: str, exc: SyntaxError) -> str:
    """Turn "invalid syntax (<unknown>, line 1)" into something actionable."""
    awkward = [str(c) for c in df.columns if not _IDENTIFIER_RE.match(str(c))]
    hint = ""
    if awkward:
        sample = ", ".join(f"`{c}`" for c in awkward[:3])
        hint = (
            " If you meant a column whose name has spaces or symbols, wrap it in "
            f"backticks — e.g. {sample}."
        )
    offset = getattr(exc, "offset", None)
    where = ""
    if isinstance(offset, int) and 0 < offset <= len(formula) + 1:
        where = f" near position {offset}: \"{formula[max(0, offset - 12):offset + 12]}\""
    return f"Could not parse the formula{where}.{hint}"


def _eval_formula_with_custom_functions(df: pd.DataFrame, formula: str) -> pd.Series:
    """Evaluate a column-arithmetic formula safely.

    Uses simpleeval instead of Python's eval(): there is no __builtins__, no
    imports, no attribute access (so dunder traversal like
    ``().__class__.__bases__`` is impossible), and the only callable surface is
    a fixed whitelist of spreadsheet-style functions. Operators apply directly
    to pandas Series so column arithmetic still vectorises, and the only
    resolvable identifiers are the dataframe's own column names.
    """
    import ast
    import operator as op

    from simpleeval import DEFAULT_OPERATORS, InvalidExpression, SimpleEval

    def _days(d1, d2):
        return (pd.to_datetime(d1) - pd.to_datetime(d2)).dt.days

    functions = {
        "IF":    lambda cond, a, b: np.where(cond, a, b),
        "ISNA":  lambda x: pd.isna(x),
        "DAYS":  _days,
        "ABS":   np.abs,
        "LOG":   np.log,
        "LOG10": np.log10,
        "LOG2":  np.log2,
        "EXP":   np.exp,
        "SQRT":  np.sqrt,
        "ROUND": np.round,
        "MIN":   np.minimum,
        "MAX":   np.maximum,
        "FLOOR": np.floor,
        "CEIL":  np.ceil,
    }
    # Allow element-wise boolean combination of Series conditions (&, |, ^, ~)
    # in addition to simpleeval's defaults; needed for IF(A>0 & B<5, ...).
    # Also override ast.Pow: simpleeval's default safe_power calls abs(base),
    # which raises "truth value of a Series is ambiguous" when the base is a
    # column Series (e.g. bmi**2). operator.pow applies ** element-wise to the
    # Series with no guard, which is what we want for column arithmetic.
    operators = {
        **DEFAULT_OPERATORS,
        ast.BitAnd: op.and_,
        ast.BitOr:  op.or_,
        ast.BitXor: op.xor,
        ast.Invert: op.invert,
        ast.Pow:    op.pow,
    }
    names = {col: df[col] for col in df.columns}

    # simpleeval parses with Python's ast, so a column whose name isn't a valid
    # identifier ("p wave mv_2", "BMI (kg/m2)") can never be referenced directly
    # — it just dies with "invalid syntax" and no hint about which part failed.
    # Swap such names out for safe placeholders first, both when the user quotes
    # them in backticks (pandas-style, always unambiguous) and when they simply
    # type the real name.
    formula, names = _resolve_awkward_column_names(df, formula, names)

    evaluator = SimpleEval(operators=operators, functions=functions, names=names)
    try:
        result = evaluator.eval(formula)
    except SyntaxError as exc:
        raise ValueError(_syntax_error_help(df, formula, exc))
    except InvalidExpression as exc:
        # NameNotDefined / FunctionNotDefined / FeatureNotAvailable / numeric
        # guards all subclass InvalidExpression.
        raise ValueError(str(exc))

    if isinstance(result, np.ndarray):
        result = pd.Series(result, index=df.index)
    if not isinstance(result, pd.Series):
        raise ValueError("Formula did not produce a series result")

    return result


@router.post("/{session_id}/formula")
def formula_compute(session_id: str, req: FormulaRequest):
    """
    Evaluate a pandas-safe formula expression and save as a new column.
    Uses df.eval() — safe, no arbitrary Python execution.
    NaN propagation is automatic: if any source cell is NaN, result is NaN.
    Supports custom functions: IF(cond, true_val, false_val), ISNA(x), DAYS(date1, date2)
    """
    df = _get_df(session_id)
    new_col = _validate_col_name(req.new_col)

    try:
        result = _eval_formula_with_custom_functions(df, req.formula)
        # eval() may return a scalar if formula has no column refs
        if not isinstance(result, pd.Series):
            raise HTTPException(status_code=422, detail="Formula did not produce a column result. Make sure to reference existing column names.")
        df = df.copy()
        df[new_col] = result
    except HTTPException:
        raise
    except Exception as exc:
        msg = str(exc)
        # Make common errors more user-friendly
        if "not defined" in msg.lower() or "undefined" in msg.lower() or "UndefinedVariable" in msg:
            # Extract the offending name
            m = re.search(r"'(\w+)'", msg)
            bad = f" Column '{m.group(1)}' not found." if m else ""
            raise HTTPException(status_code=422, detail=f"Unknown column name in formula.{bad} Check spelling and use exact column names as they appear in the dataset.")
        raise HTTPException(status_code=422, detail=f"Formula error: {msg}")

    store.save(session_id, df)
    return _build_result(df, new_col)


# ── 2. Transformations ────────────────────────────────────────────────────────

TRANSFORMS = {
    "ln":           "Ln (natural log)",
    "log10":        "Log₁₀",
    "sqrt":         "√ Square root",
    "square":       "x² Square",
    "exp":          "eˣ Exponential",
    "abs":          "|x| Absolute value",
    "zscore":       "Z-score",
    "tertile":      "Tertile (3 groups)",
    "quartile":     "Quartile (4 groups)",
    "median_split": "Median split (2 groups)",
}


class TransformRequest(BaseModel):
    source_col: str
    transform: str          # one of the TRANSFORMS keys
    new_col: str


@router.post("/{session_id}/transform")
def transform_compute(session_id: str, req: TransformRequest):
    df = _get_df(session_id)
    new_col = _validate_col_name(req.new_col)

    if req.source_col not in df.columns:
        raise HTTPException(status_code=422, detail=f"Column '{req.source_col}' not found")
    if new_col == req.source_col:
        # Writing the result back over its own input destroys the input. The
        # UI used to suggest exactly this name for the transforms it had no
        # prefix for (tertile, quartile, median split), so one click replaced
        # the source column with its own groups and the original values were
        # gone. There is no reading of this that the user wants.
        raise HTTPException(
            status_code=422,
            detail=(
                f"'{new_col}' is the source column. Writing the result there "
                "would replace the values the transform is computed from. "
                "Give the result its own column name."
            ),
        )
    if req.transform not in TRANSFORMS:
        raise HTTPException(status_code=422, detail=f"Unknown transform '{req.transform}'. Valid: {list(TRANSFORMS.keys())}")

    max_plausible = plausibility_max_for_column(req.source_col)
    sentinel_mask = flag_sentinels(df[req.source_col], max_plausible)
    col = mask_sentinels(df[req.source_col], max_plausible)
    df = df.copy()

    if req.transform == "ln":
        df[new_col] = np.log(col.where(col > 0))       # ≤0 → NaN
    elif req.transform == "log10":
        df[new_col] = np.log10(col.where(col > 0))
    elif req.transform == "sqrt":
        df[new_col] = np.sqrt(col.where(col >= 0))     # <0 → NaN
    elif req.transform == "square":
        df[new_col] = col ** 2
    elif req.transform == "exp":
        df[new_col] = np.exp(col)
    elif req.transform == "abs":
        df[new_col] = col.abs()
    elif req.transform == "zscore":
        mu, sd = col.mean(), col.std()
        if sd == 0:
            raise HTTPException(status_code=422, detail="Standard deviation is 0 — cannot compute Z-score for a constant column")
        df[new_col] = (col - mu) / sd
    elif req.transform == "tertile":
        df[new_col] = _quantile_groups(col, 3)
    elif req.transform == "quartile":
        df[new_col] = _quantile_groups(col, 4)
    elif req.transform == "median_split":
        med = col.median()
        if pd.isna(med):
            raise HTTPException(status_code=422, detail="No numeric values available for median split")
        df[new_col] = (col > med).where(col.notna(), np.nan).astype(float)  # 0 = ≤ median, 1 = > median

    store.save(session_id, df)
    result = _build_result(df, new_col)
    if sentinel_mask.any():
        result["warnings"] = [
            f"{int(sentinel_mask.sum())} implausible value(s) in '{req.source_col}' were treated as missing for this transform."
        ]
        result["n_implausible"] = int(sentinel_mask.sum())
    return result


# ── 3. Recode / Binning ───────────────────────────────────────────────────────

class Condition(BaseModel):
    col: str
    op: str       # one of: < <= > >= == !=
    val: Any      # string or number

class Rule(BaseModel):
    conditions: List[Condition]   # all joined with AND
    result: Any                   # the value to assign when all conditions are true

class RecodeRequest(BaseModel):
    rules: List[Rule]
    else_val: Optional[Any] = None   # None → NaN; or numeric/string
    new_col: str


_OPS = {
    "<":        lambda s, v: s < v,
    "<=":       lambda s, v: s <= v,
    ">":        lambda s, v: s > v,
    ">=":       lambda s, v: s >= v,
    "==":       lambda s, v: s == v,
    "!=":       lambda s, v: s != v,
    "contains": lambda s, v: s.astype("string").str.contains(str(v), case=False, na=False),
    "!contains": lambda s, v: ~s.astype("string").str.contains(str(v), case=False, na=False),
}


def _cast_val(col_series: pd.Series, val: Any) -> Any:
    """Try to cast the threshold value to the column's dtype."""
    if pd.api.types.is_numeric_dtype(col_series):
        try:
            return float(val)
        except (TypeError, ValueError):
            return val
    return val


@router.post("/{session_id}/recode")
def recode_compute(session_id: str, req: RecodeRequest):
    df = _get_df(session_id)
    new_col = _validate_col_name(req.new_col)

    if not req.rules:
        raise HTTPException(status_code=422, detail="At least one rule is required")

    # Validate all referenced columns exist
    all_cols = {c.col for r in req.rules for c in r.conditions}
    missing = all_cols - set(df.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"Column(s) not found: {', '.join(missing)}")

    conditions: list = []
    choices: list = []

    for rule in req.rules:
        mask = pd.Series([True] * len(df), index=df.index)
        for cond in rule.conditions:
            if cond.op not in _OPS:
                raise HTTPException(status_code=422, detail=f"Unknown operator '{cond.op}'")

            raw_col = df[cond.col]
            val = cond.val

            # Decide whether to compare as numeric or string
            # If the value looks numeric, try numeric comparison
            val_is_numeric = False
            try:
                val_num = float(val)
                val_is_numeric = True
            except (TypeError, ValueError):
                pass

            if val_is_numeric and cond.op in ("<", "<=", ">", ">="):
                # Numeric comparison — coerce column to numeric
                col_s = pd.to_numeric(raw_col, errors="coerce")
                v = val_num
            elif val_is_numeric and cond.op in ("==", "!="):
                # For ==  / !=, try numeric first, fall back to string
                col_num = pd.to_numeric(raw_col, errors="coerce")
                if col_num.notna().sum() > col_num.isna().sum():
                    col_s = col_num
                    v = val_num
                else:
                    col_s = raw_col.astype("string").str.strip()
                    v = str(val).strip()
            else:
                # String comparison (value is text, or == / != on text column)
                col_s = raw_col.astype("string").str.strip()
                v = str(val).strip()

            try:
                cond_mask = _OPS[cond.op](col_s, v)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Condition error ({cond.col} {cond.op} {cond.val}): {exc}")
            # NaN in source → False (row not matched)
            cond_mask = cond_mask.fillna(False) & raw_col.notna()
            mask = mask & cond_mask
        conditions.append(mask)
        # Try to cast result to numeric
        try:
            choices.append(float(rule.result))
        except (TypeError, ValueError):
            choices.append(rule.result)

    # Determine default
    default = np.nan
    has_default = False
    if req.else_val is not None and str(req.else_val).strip() != "":
        has_default = True
        try:
            default = float(req.else_val)
        except (TypeError, ValueError):
            default = req.else_val

    df = df.copy()

    # Rules are first-match-wins, like np.select. Build with pandas so missing
    # defaults and string choices do not trigger NumPy dtype promotion errors.
    all_numeric = all(isinstance(c, (int, float)) for c in choices)
    if has_default:
        try:
            float(default)
        except (TypeError, ValueError):
            all_numeric = False

    if all_numeric:
        result = pd.Series(default, index=df.index, dtype="float64")
        unmatched = pd.Series(True, index=df.index)
        for cond_mask, choice in zip(conditions, choices):
            assign_mask = cond_mask & unmatched
            result.loc[assign_mask] = choice
            unmatched &= ~assign_mask
        df[new_col] = result
        # Convert int-like float columns to int if no NaN
        if df[new_col].notna().all():
            try:
                vals = df[new_col].astype(float)
                if (vals % 1 == 0).all():
                    df[new_col] = vals.astype(int)
            except (ValueError, TypeError):
                pass
    else:
        result = pd.Series(pd.NA if not has_default else str(default), index=df.index, dtype="object")
        unmatched = pd.Series(True, index=df.index)
        for cond_mask, choice in zip(conditions, choices):
            assign_mask = cond_mask & unmatched
            result.loc[assign_mask] = str(choice)
            unmatched &= ~assign_mask
        df[new_col] = result

    store.save(session_id, df)
    return _build_result(df, new_col)


# ── 4. Clinical Calculators ───────────────────────────────────────────────────

class ClinicalRequest(BaseModel):
    column_map: Dict[str, str]   # logical_name → actual df column name
    female_value: Optional[str] = None  # which value in sex column = Female
    new_col: Optional[str] = None       # override output column name


def _req_cols(column_map: dict, *keys: str):
    missing = [k for k in keys if not column_map.get(k)]
    if missing:
        raise HTTPException(status_code=422, detail=f"Required column mapping(s) missing: {', '.join(missing)}")


def _is_female(df: pd.DataFrame, sex_col: str, female_value: Optional[str]) -> pd.Series:
    """Return boolean Series indicating Female rows."""
    col = df[sex_col].astype(str)
    if female_value is not None:
        return col == str(female_value)
    # Auto-detect common patterns
    return col.str.lower().isin(["f", "female", "kadın", "kadin", "women", "w", "2"])


@router.post("/{session_id}/clinical/bmi")
def clinical_bmi(session_id: str, req: ClinicalRequest):
    df = _get_df(session_id)
    cm = req.column_map
    _req_cols(cm, "weight", "height")

    weight = pd.to_numeric(df[cm["weight"]], errors="coerce")
    height = pd.to_numeric(df[cm["height"]], errors="coerce")

    df = df.copy()
    new_col = req.new_col or "BMI"
    df[new_col] = (weight / ((height / 100) ** 2)).round(2)

    store.save(session_id, df)
    return _build_result(df, new_col)


@router.post("/{session_id}/clinical/egfr")
def clinical_egfr(session_id: str, req: ClinicalRequest):
    """Race-free CKD-EPI 2021 eGFR formula."""
    df = _get_df(session_id)
    cm = req.column_map
    _req_cols(cm, "age", "sex", "creatinine")

    age = pd.to_numeric(df[cm["age"]], errors="coerce")
    scr = pd.to_numeric(df[cm["creatinine"]], errors="coerce")
    is_f = _is_female(df, cm["sex"], req.female_value)

    kappa = np.where(is_f, 0.7, 0.9)
    alpha = np.where(is_f, -0.241, -0.302)
    ratio = scr.values / kappa

    egfr = (
        142
        * np.minimum(ratio, 1) ** alpha
        * np.maximum(ratio, 1) ** (-1.200)
        * 0.9938 ** age.values
        * np.where(is_f, 1.012, 1.0)
    )

    df = df.copy()
    new_col = req.new_col or "eGFR"
    df[new_col] = np.round(egfr, 1)

    store.save(session_id, df)
    return _build_result(df, new_col)


@router.post("/{session_id}/clinical/chadsvasc")
def clinical_chadsvasc(session_id: str, req: ClinicalRequest):
    """CHA₂DS₂-VASc score for AF stroke risk."""
    df = _get_df(session_id)
    cm = req.column_map
    _req_cols(cm, "age", "sex")

    age = pd.to_numeric(df[cm["age"]], errors="coerce")
    is_f = _is_female(df, cm["sex"], req.female_value)

    # Age score: ≥75 → 2, 65-74 → 1, <65 → 0
    age_score = np.where(age >= 75, 2, np.where(age >= 65, 1, 0))

    def _binary(key: str) -> pd.Series:
        col_name = cm.get(key)
        if not col_name:
            return pd.Series(0, index=df.index)
        s = pd.to_numeric(df[col_name], errors="coerce").fillna(0)
        return s.clip(0, 1).astype(int)

    score = (
        _binary("chf")           # CHF = 1
        + _binary("htn")         # Hypertension = 1
        + age_score              # Age score
        + _binary("dm")          # Diabetes = 1
        + _binary("stroke") * 2  # Stroke/TIA = 2
        + _binary("vasc")        # Vascular disease = 1
        + is_f.astype(int)       # Female sex = 1
    )

    df = df.copy()
    new_col = req.new_col or "CHA2DS2VASc"
    df[new_col] = score

    store.save(session_id, df)
    return _build_result(df, new_col)


# ── shared binary helper used by all clinical calculators ─────────────────────

def _bin(df: pd.DataFrame, cm: dict, key: str) -> pd.Series:
    """Return an integer 0/1 Series for a binary column; 0 if column not mapped."""
    col_name = cm.get(key)
    if not col_name:
        return pd.Series(0, index=df.index)
    s = pd.to_numeric(df[col_name], errors="coerce").fillna(0)
    return s.clip(0, 1).astype(int)


def _num(df: pd.DataFrame, cm: dict, key: str) -> pd.Series:
    """Return a numeric Series for a column; NaN if not mapped."""
    col_name = cm.get(key)
    if not col_name:
        return pd.Series(np.nan, index=df.index)
    max_plausible = plausibility_max_for_column(key) or plausibility_max_for_column(col_name)
    return mask_sentinels(df[col_name], max_plausible)


def _clinical_warnings(df: pd.DataFrame, cm: dict) -> list[str]:
    warnings: list[str] = []
    for key, col_name in (cm or {}).items():
        if col_name not in df.columns:
            continue
        max_plausible = plausibility_max_for_column(key) or plausibility_max_for_column(col_name)
        mask = flag_sentinels(df[col_name], max_plausible)
        if mask.any():
            vals = sorted(sentinel_values(df[col_name], max_plausible))
            warnings.append(
                f"{int(mask.sum())} implausible value(s) in '{col_name}' treated as missing: {vals}"
            )
    return warnings


def _build_clinical_result(df: pd.DataFrame, new_col: str, cm: dict) -> dict:
    result = _build_result(df, new_col)
    warnings = _clinical_warnings(df, cm)
    if warnings:
        result["warnings"] = warnings
    return result


# ── BSA (Mosteller formula) ───────────────────────────────────────────────────

@router.post("/{session_id}/clinical/bsa")
def clinical_bsa(session_id: str, req: ClinicalRequest):
    """Body Surface Area = sqrt(height_cm × weight_kg / 3600)"""
    df = _get_df(session_id)
    _req_cols(req.column_map, "weight", "height")
    weight = _num(df, req.column_map, "weight")
    height = _num(df, req.column_map, "height")
    df = df.copy()
    new_col = req.new_col or "BSA"
    df[new_col] = np.sqrt(height * weight / 3600).round(2)
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── MAP (Mean Arterial Pressure) ──────────────────────────────────────────────

@router.post("/{session_id}/clinical/map")
def clinical_map(session_id: str, req: ClinicalRequest):
    """MAP = (SBP + 2 × DBP) / 3"""
    df = _get_df(session_id)
    _req_cols(req.column_map, "sbp", "dbp")
    sbp = _num(df, req.column_map, "sbp")
    dbp = _num(df, req.column_map, "dbp")
    df = df.copy()
    new_col = req.new_col or "MAP"
    df[new_col] = ((sbp + 2 * dbp) / 3).round(1)
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── CHA₂DS₂-VA (2024 ESC updated — sex category removed) ─────────────────────

@router.post("/{session_id}/clinical/chadsva")
def clinical_chadsva(session_id: str, req: ClinicalRequest):
    """CHA₂DS₂-VA score (2024 ESC guideline update — sex no longer counted)."""
    df = _get_df(session_id)
    cm = req.column_map
    _req_cols(cm, "age")
    age = _num(df, cm, "age")
    age_score = np.where(age >= 75, 2, np.where(age >= 65, 1, 0))
    score = (
        _bin(df, cm, "chf")           # CHF = 1
        + _bin(df, cm, "htn")         # Hypertension = 1
        + age_score                   # Age ≥75 = 2, 65-74 = 1
        + _bin(df, cm, "dm")          # Diabetes = 1
        + _bin(df, cm, "stroke") * 2  # Stroke/TIA = 2
        + _bin(df, cm, "vasc")        # Vascular disease = 1
    )
    df = df.copy()
    new_col = req.new_col or "CHA2DS2VA"
    df[new_col] = score
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── HAS-BLED Score ────────────────────────────────────────────────────────────

@router.post("/{session_id}/clinical/hasbled")
def clinical_hasbled(session_id: str, req: ClinicalRequest):
    """HAS-BLED bleeding risk score (0-9)."""
    df = _get_df(session_id)
    cm = req.column_map
    # Age-based elderly criterion: >65
    age_col = cm.get("age")
    if age_col:
        age = _num(df, cm, "age")
        elderly = (age > 65).astype(int).fillna(0)
    else:
        elderly = _bin(df, cm, "elderly")
    score = (
        _bin(df, cm, "htn")       # H: uncontrolled hypertension
        + _bin(df, cm, "renal")   # A: abnormal renal function (1 each)
        + _bin(df, cm, "liver")   # A: abnormal liver function (1 each)
        + _bin(df, cm, "stroke")  # S: stroke history
        + _bin(df, cm, "bleeding") # B: bleeding history
        + _bin(df, cm, "labile_inr") # L: labile INR
        + elderly                  # E: age > 65
        + _bin(df, cm, "drugs")   # D: drugs (antiplatelets/NSAIDs)
        + _bin(df, cm, "alcohol") # D: alcohol use
    )
    df = df.copy()
    new_col = req.new_col or "HAS_BLED"
    df[new_col] = score
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── GRACE Score (in-hospital mortality) ───────────────────────────────────────

def _grace_lookup(series: pd.Series, breakpoints: list, points: list) -> np.ndarray:
    """Map a numeric series to integer points using a step lookup table."""
    result = np.zeros(len(series), dtype=int)
    for i, (bp, pt) in enumerate(zip(breakpoints, points)):
        if i == 0:
            result = np.where(series < bp, pt, result)
        else:
            result = np.where(series >= breakpoints[i - 1], pt, result)
    return result


@router.post("/{session_id}/clinical/grace")
def clinical_grace(session_id: str, req: ClinicalRequest):
    """GRACE 2.0 integer risk score for ACS (in-hospital mortality)."""
    df = _get_df(session_id)
    cm = req.column_map
    _req_cols(cm, "age", "hr", "sbp", "creatinine")

    age = _num(df, cm, "age").values
    hr  = _num(df, cm, "hr").values
    sbp = _num(df, cm, "sbp").values
    scr = _num(df, cm, "creatinine").values   # mg/dL

    # Age lookup (points for upper boundary of each bracket)
    age_pts = np.select(
        [age < 30, age < 40, age < 50, age < 60, age < 70, age < 80, age < 90],
        [0,        8,        25,        41,        58,        75,        91],
        default=100,
    )
    # Heart rate
    hr_pts = np.select(
        [hr < 50, hr < 70, hr < 90, hr < 110, hr < 150, hr < 200],
        [0,       3,       9,       15,        24,        38],
        default=46,
    )
    # Systolic BP
    sbp_pts = np.select(
        [sbp < 80, sbp < 100, sbp < 120, sbp < 140, sbp < 160, sbp < 200],
        [63,       58,        47,         37,         26,         11],
        default=0,
    )
    # Creatinine (mg/dL)
    scr_pts = np.select(
        [scr < 0.4, scr < 0.8, scr < 1.2, scr < 1.6, scr < 2.0, scr < 4.0],
        [2,         5,         8,          11,         14,         23],
        default=31,
    )
    # Killip class (1-4 → 0, 20, 39, 59)
    killip_col = cm.get("killip")
    if killip_col:
        killip = pd.to_numeric(df[killip_col], errors="coerce").fillna(1).clip(1, 4).astype(int)
        killip_pts = np.select(
            [killip == 1, killip == 2, killip == 3],
            [0,           20,          39],
            default=59,
        )
    else:
        killip_pts = np.zeros(len(df), dtype=int)

    score = (
        age_pts
        + hr_pts
        + sbp_pts
        + scr_pts
        + killip_pts
        + _bin(df, cm, "cardiac_arrest").values * 43
        + _bin(df, cm, "st_deviation").values   * 30
        + _bin(df, cm, "cardiac_markers").values * 15
    )

    df = df.copy()
    new_col = req.new_col or "GRACE_Score"
    df[new_col] = score.astype(int)
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── TIMI Risk Score for NSTEMI / UA ──────────────────────────────────────────

@router.post("/{session_id}/clinical/timi_nstemi")
def clinical_timi_nstemi(session_id: str, req: ClinicalRequest):
    """TIMI risk score for NSTEMI/UA (0-7). Each criterion = 1 point."""
    df = _get_df(session_id)
    cm = req.column_map

    # Age ≥65 from numeric column
    age_col = cm.get("age")
    if age_col:
        age_pts = (_num(df, cm, "age") >= 65).astype(int).fillna(0)
    else:
        age_pts = _bin(df, cm, "age_ge65")

    score = (
        age_pts                        # 1. Age ≥ 65
        + _bin(df, cm, "risk_factors") # 2. ≥3 CAD risk factors
        + _bin(df, cm, "known_cad")    # 3. Known CAD (stenosis ≥50%)
        + _bin(df, cm, "aspirin")      # 4. Aspirin use in last 7 days
        + _bin(df, cm, "severe_angina")# 5. ≥2 anginal events in last 24h
        + _bin(df, cm, "st_deviation") # 6. ST deviation ≥0.5 mm
        + _bin(df, cm, "markers")      # 7. Elevated cardiac markers
    )
    df = df.copy()
    new_col = req.new_col or "TIMI_NSTEMI"
    df[new_col] = score
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── TIMI Risk Score for STEMI ─────────────────────────────────────────────────

@router.post("/{session_id}/clinical/timi_stemi")
def clinical_timi_stemi(session_id: str, req: ClinicalRequest):
    """TIMI risk score for STEMI (0-14). Points as per original publication."""
    df = _get_df(session_id)
    cm = req.column_map

    age = _num(df, cm, "age")
    age_pts = np.where(age >= 75, 3, np.where(age >= 65, 2, 0))

    sbp = _num(df, cm, "sbp")
    sbp_pts = (sbp < 100).astype(int).fillna(0) * 3

    hr = _num(df, cm, "hr")
    hr_pts = (hr > 100).astype(int).fillna(0) * 2

    # Killip class II-IV = 2 points
    killip_col = cm.get("killip")
    if killip_col:
        killip = pd.to_numeric(df[killip_col], errors="coerce").fillna(1)
        killip_pts = (killip > 1).astype(int) * 2
    else:
        killip_pts = pd.Series(0, index=df.index)

    weight = _num(df, cm, "weight")
    weight_pts = (weight < 67).astype(int).fillna(0)

    score = (
        age_pts
        + _bin(df, cm, "dm_htn_angina") * 1  # DM, HTN, or angina = 1
        + sbp_pts                              # SBP < 100 = 3
        + hr_pts                               # HR > 100 = 2
        + killip_pts                           # Killip II-IV = 2
        + weight_pts                           # Weight < 67 kg = 1
        + _bin(df, cm, "anterior_stemi") * 1  # Anterior ST elevation or LBBB = 1
        + _bin(df, cm, "late_treatment") * 1  # Time to treatment > 4h = 1
    )
    df = df.copy()
    new_col = req.new_col or "TIMI_STEMI"
    df[new_col] = score
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── H2FPEF Score (HFpEF diagnosis) ───────────────────────────────────────────

@router.post("/{session_id}/clinical/h2fpef")
def clinical_h2fpef(session_id: str, req: ClinicalRequest):
    """H2FPEF score for HFpEF probability (0-9)."""
    df = _get_df(session_id)
    cm = req.column_map

    # H: Heavy — BMI > 30 = 2 points
    bmi_col = cm.get("bmi")
    if bmi_col:
        bmi = _num(df, cm, "bmi")
        heavy = (bmi > 30).astype(int).fillna(0) * 2
    else:
        heavy = _bin(df, cm, "obese") * 2   # or direct binary

    # E: Elderly — age > 60 = 1 point
    age_col = cm.get("age")
    if age_col:
        age = _num(df, cm, "age")
        elderly = (age > 60).astype(int).fillna(0)
    else:
        elderly = _bin(df, cm, "elderly")

    score = (
        heavy                            # H²: obese (BMI > 30) = 2
        + _bin(df, cm, "htn_meds") * 1  # H: ≥2 antihypertensive meds = 1
        + _bin(df, cm, "af") * 3        # F: Atrial fibrillation = 3
        + _bin(df, cm, "pulm_htn") * 1  # P: Pulmonary HTN (PASP > 35) = 1
        + elderly                        # E: Age > 60 = 1
        + _bin(df, cm, "ee_ratio") * 1  # F: E/e' > 9 = 1
    )
    df = df.copy()
    new_col = req.new_col or "H2FPEF"
    df[new_col] = score
    store.save(session_id, df)
    return _build_clinical_result(df, new_col, cm)


# ── MAGGIC Heart Failure Risk Score ──────────────────────────────────────────

def _maggic_age_pts(age: np.ndarray) -> np.ndarray:
    return np.select(
        [age < 55, age < 60, age < 65, age < 70, age < 75, age < 80],
        [0,        1,        2,        4,        6,        8],
        default=10,
    )

def _maggic_sbp_pts(sbp: np.ndarray) -> np.ndarray:
    return np.select(
        [sbp < 100, sbp < 110, sbp < 120, sbp < 130, sbp < 140],
        [5,         4,         3,         2,         1],
        default=0,
    )

def _maggic_bmi_pts(bmi: np.ndarray) -> np.ndarray:
    return np.select(
        [bmi < 15, bmi < 20, bmi < 25, bmi < 30],
        [6,        5,        3,        1],
        default=0,
    )

def _maggic_creatinine_pts(scr_umol: np.ndarray) -> np.ndarray:
    """Creatinine in μmol/L."""
    return np.select(
        [scr_umol < 90, scr_umol < 110, scr_umol < 130, scr_umol < 150, scr_umol < 170, scr_umol < 210],
        [0,             1,              2,              3,              4,              5],
        default=8,
    )

def _maggic_ef_pts(ef: np.ndarray) -> np.ndarray:
    return np.select(
        [ef < 15, ef < 20, ef < 25, ef < 30, ef < 35, ef < 40, ef < 45],
        [7,       6,       5,       4,       3,       2,       1],
        default=0,
    )

def _maggic_nyha_pts(nyha: np.ndarray) -> np.ndarray:
    return np.select(
        [nyha == 1, nyha == 2, nyha == 3],
        [0,         2,         6],
        default=8,  # NYHA IV
    )


@router.post("/{session_id}/clinical/maggic")
def clinical_maggic(session_id: str, req: ClinicalRequest):
    """MAGGIC Heart Failure Risk Score (Pocock et al. 2013, EHJ)."""
    df = _get_df(session_id)
    cm = req.column_map
    _req_cols(cm, "age", "sbp", "bmi", "creatinine", "ef")

    age = _num(df, cm, "age").values
    sbp = _num(df, cm, "sbp").values
    bmi_vals = _num(df, cm, "bmi").values
    ef  = _num(df, cm, "ef").values

    # Creatinine: auto-detect mg/dL vs μmol/L (mg/dL values are typically <20)
    scr_raw = _num(df, cm, "creatinine").values
    scr_umol = np.where(np.nanmax(scr_raw) < 20, scr_raw * 88.4, scr_raw)

    # NYHA class (1-4); default to 2 if not mapped
    nyha_col = cm.get("nyha")
    if nyha_col:
        nyha = pd.to_numeric(df[nyha_col], errors="coerce").fillna(2).clip(1, 4).values
    else:
        nyha = np.full(len(df), 2.0)

    # Sex: male = +1
    sex_col = cm.get("sex")
    if sex_col:
        is_male = ~_is_female(df, sex_col, req.female_value)
        male_pts = is_male.astype(int).values
    else:
        male_pts = np.zeros(len(df), dtype=int)

    # Not on BB = +3; we accept a "bb" column (1=on BB, 0=not on BB)
    bb = _bin(df, cm, "bb").values
    not_on_bb = (1 - bb) * 3

    # Not on ACE/ARB = +1
    ace = _bin(df, cm, "ace_arb").values
    not_on_ace = (1 - ace)

    score = (
        _maggic_age_pts(age)
        + male_pts
        + _maggic_nyha_pts(nyha)
        + np.where(cm.get("current_smoker"), _bin(df, cm, "current_smoker").values, 0)
        + _bin(df, cm, "diabetes").values   * 3
        + _bin(df, cm, "copd").values       * 2
        + _bin(df, cm, "hf_lt18m").values   * 2  # HF diagnosed < 18 months ago
        + not_on_ace
        + not_on_bb
        + _maggic_sbp_pts(sbp)
        + _maggic_bmi_pts(bmi_vals)
        + _maggic_creatinine_pts(scr_umol)
        + _maggic_ef_pts(ef)
    )

    df = df.copy()
    new_col = req.new_col or "MAGGIC_Score"
    df[new_col] = score.astype(int)
    store.save(session_id, df)
    return _build_clinical_result(df, new_col, cm)


# ── QTc — Bazett's formula ────────────────────────────────────────────────────

@router.post("/{session_id}/clinical/qtc")
def clinical_qtc(session_id: str, req: ClinicalRequest):
    """Corrected QT interval (Bazett): QTc = QT_ms / sqrt(RR_s) = QT / sqrt(60/HR)"""
    df = _get_df(session_id)
    cm = req.column_map
    _req_cols(cm, "qt", "hr")
    qt = _num(df, cm, "qt")   # QT in milliseconds
    hr = _num(df, cm, "hr")   # Heart rate in bpm
    rr = 60.0 / hr            # RR interval in seconds
    df = df.copy()
    new_col = req.new_col or "QTc_Bazett"
    df[new_col] = (qt / np.sqrt(rr)).round(1)
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── 5. Delete column ──────────────────────────────────────────────────────────

@router.delete("/{session_id}/column/{col_name:path}")
def delete_column(session_id: str, col_name: str):
    df = _get_df(session_id)
    if col_name not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{col_name}' not found")
    df = store.delete_dataframe_columns(session_id, [col_name])
    store.log_action(session_id, "delete_column", {"column": col_name})
    conditions = store.get_filter(session_id)
    filtered = store.get_filtered(session_id)
    case_filter = {
        "conditions": conditions,
        "selected": len(filtered),
        "total": len(df),
    } if conditions else None
    return {"deleted": col_name, "case_filter": case_filter}


@router.get("/{session_id}/column_values/{col_name:path}")
def column_values(session_id: str, col_name: str):
    """Every value in a column, in row order.

    "Copy column" needs the WHOLE column: the frontend's `preview` is capped at
    2000 rows, so copying from it silently truncated larger datasets.
    """
    df = _get_df(session_id)
    if col_name not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{col_name}' not found")
    return {
        "name": col_name,
        "values": [_jsonable_value(v) for v in df[col_name].tolist()],
        "rows": len(df),
    }


class PasteColumnRequest(BaseModel):
    name: str
    values: List[Optional[str]]  # raw strings in row order; "" / None → missing
    position: int = -1  # -1 = append at end, otherwise insert at this index


@router.post("/{session_id}/paste_column")
def paste_column(session_id: str, req: PasteColumnRequest):
    """Insert a whole column (name + per-row values) — the paste side of
    "Copy column", including pasting across sessions/windows via the clipboard.

    Values are matched to rows by position. A longer payload is truncated and a
    shorter one is padded with blanks; both counts are reported so the UI can
    tell the user what happened instead of silently mangling the data.
    """
    df = _get_df(session_id)
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Column name cannot be empty")
    if name in df.columns:
        raise HTTPException(status_code=422, detail=f"Column '{name}' already exists")

    n_rows = len(df)
    supplied = len(req.values)
    trimmed = [None if (v is None or str(v).strip() == "") else str(v).strip() for v in req.values[:n_rows]]
    padded = trimmed + [None] * (n_rows - len(trimmed))
    series = pd.Series(padded, index=df.index, dtype="object")

    # Cast to numeric only when every observed value parses, so a text column
    # pasted onto a numeric-looking dataset isn't silently coerced to NaN.
    observed = series.notna()
    if observed.any():
        numeric = coerce_numeric(series)
        if bool(numeric[observed].notna().all()):
            # coerce_numeric hands back pandas' NULLABLE dtypes (Int64/Float64),
            # but the rest of the app — _detect_kind included — only recognises
            # numpy dtypes, so an Int64 column would be mislabelled "text".
            # Downcast to numpy: float64 when blanks need NaN, else the natural
            # integer/float dtype.
            if bool(numeric.isna().any()):
                series = pd.Series(numeric.to_numpy(dtype="float64", na_value=np.nan), index=df.index)
            else:
                series = pd.Series(numeric.to_numpy(), index=df.index)

    df = df.copy()
    if 0 <= req.position < len(df.columns):
        df.insert(req.position, name, series)
    else:
        df[name] = series
    store.save(session_id, df)
    store.log_action(session_id, "paste_column", {"name": name, "n_values": int(observed.sum())})

    result = _build_result(df, name)
    result["n_supplied"] = supplied
    result["n_truncated"] = max(0, supplied - n_rows)
    result["n_padded"] = max(0, n_rows - supplied)
    return result


class DeleteColumnsRequest(BaseModel):
    columns: List[str]  # column names to drop


@router.post("/{session_id}/delete_columns")
def delete_columns(session_id: str, req: DeleteColumnsRequest):
    """Drop several columns in one atomic mutation (bulk tick-and-delete)."""
    df = _get_df(session_id)
    if not req.columns:
        raise HTTPException(status_code=422, detail="No columns provided")
    # De-duplicate while preserving order so a repeated name doesn't error.
    unique_cols = list(dict.fromkeys(req.columns))
    missing = [c for c in unique_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=404, detail=f"Columns not found: {missing}")
    if len(unique_cols) >= len(df.columns):
        raise HTTPException(status_code=422, detail="Cannot delete every column")
    df = store.delete_dataframe_columns(session_id, unique_cols)
    store.log_action(session_id, "delete_columns", {"n_deleted": len(unique_cols)})
    conditions = store.get_filter(session_id)
    filtered = store.get_filtered(session_id)
    case_filter = {
        "conditions": conditions,
        "selected": len(filtered),
        "total": len(df),
    } if conditions else None
    return {
        "deleted": unique_cols,
        "remaining_columns": list(df.columns),
        "case_filter": case_filter,
    }


# ── 6. Fill blanks ──────────────────────────────────────────────────────────

class FillBlanksRequest(BaseModel):
    column: str
    value: str  # fill value (will be cast to match column dtype)
    # When set, the original column is left untouched and the filled result is
    # written to this NEW column (the original is copied first, then imputed).
    new_column: Optional[str] = None


@router.post("/{session_id}/fill_blanks")
def fill_blanks(session_id: str, req: FillBlanksRequest):
    df = _get_df(session_id)
    if req.column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{req.column}' not found")

    df = df.copy()
    # Write into a new column when requested (keeps the original intact); the
    # rest of the routine operates on `target`.
    target = _validate_col_name(req.new_column) if req.new_column else req.column
    if req.new_column:
        if target in df.columns:
            raise HTTPException(status_code=422, detail=f"Column '{target}' already exists")
        source_pos = list(df.columns).index(req.column)
        df.insert(source_pos + 1, target, df[req.column].copy())
    col = df[target]
    max_plausible = plausibility_max_for_column(req.column)
    sentinel_mask = flag_sentinels(col, max_plausible)
    blank_mask = col.astype(str).str.strip() == ""
    n_before = int((col.isna() | blank_mask | sentinel_mask).sum())

    method_label = req.value

    # Special fill strategies
    if req.value == "__mean__":
        num_col = mask_sentinels(col, max_plausible)
        fill_val = float(num_col.mean())
        method_label = f"mean ({fill_val:.2f})"
        df[target] = num_col.fillna(fill_val)
    elif req.value == "__median__":
        num_col = mask_sentinels(col, max_plausible)
        fill_val = float(num_col.median())
        method_label = f"median ({fill_val:.2f})"
        df[target] = num_col.fillna(fill_val)
    elif req.value == "__mode__":
        observed = col.dropna()
        observed = observed[observed.astype(str).str.strip() != ""]
        if observed.empty:
            raise HTTPException(status_code=422, detail=f"Column '{req.column}' has no values to impute from.")
        fill_val = observed.mode().iloc[0]
        df[target] = col.fillna(fill_val)
        if col.dtype == object:
            df.loc[df[target].astype(str).str.strip() == "", target] = fill_val
        method_label = f"most frequent ({fill_val})"
    elif req.value == "__rownum__":
        # Sequential case number: each blank cell gets its 1-based row
        # position. On a fully-empty column this numbers every case 1..n,
        # giving a ready-made patient/case ID.
        fill_mask = col.isna() | blank_mask
        positions = pd.Series(range(1, len(df) + 1), index=df.index)
        df.loc[fill_mask, target] = positions[fill_mask]
        # All-blank object column just filled with ints → numeric column.
        if not pd.api.types.is_numeric_dtype(df[target]):
            coerced_all = pd.to_numeric(df[target], errors="coerce")
            if coerced_all.notna().sum() == df[target].notna().sum():
                df[target] = coerced_all
        method_label = "sequential row number (1…n)"
    elif req.value == "__mice__":
        coerced = mask_sentinels(col, max_plausible)
        is_numeric_col = pd.api.types.is_numeric_dtype(col) or (
            col.notna().any() and coerced.notna().mean() >= 0.8
        )
        if not is_numeric_col:
            # MICE is undefined on text/categorical → impute with the most
            # frequent value (mode). Never crash; never silently no-op.
            mode = col.dropna()
            mode = mode[mode.astype(str).str.strip() != ""]
            if mode.empty:
                raise HTTPException(status_code=422, detail=f"Column '{req.column}' has no values to impute from.")
            fill_val = mode.mode().iloc[0]
            df[target] = col.fillna(fill_val)
            if col.dtype == object:
                df.loc[df[target].astype(str).str.strip() == "", target] = fill_val
            method_label = f"most frequent ({fill_val})"
        else:
            # Numeric → MICE using the other numeric feature columns that have
            # data. Pre-filter all-NaN features so the imputer can't drop a
            # column and misalign positions (the previous IndexError).
            from sklearn.experimental import enable_iterative_imputer  # noqa
            from sklearn.impute import IterativeImputer
            work = df.copy()
            work[target] = coerced
            feat_cols = [c for c in work.select_dtypes(include="number").columns if work[c].notna().any()]
            if target not in feat_cols:
                feat_cols = [target, *feat_cols]
            if len(feat_cols) >= 2 and work[target].notna().any():
                try:
                    imp = IterativeImputer(max_iter=10, random_state=42)
                    out = pd.DataFrame(imp.fit_transform(work[feat_cols]), columns=feat_cols, index=work.index)
                    df[target] = out[target]
                    method_label = "MICE (multiple imputation)"
                except Exception:
                    med = coerced.median()
                    df[target] = coerced.fillna(med)
                    method_label = f"median fallback ({med:.2f})" if pd.notna(med) else "median fallback"
            else:
                # Too few numeric features for chained equations → median.
                med = coerced.median()
                if pd.isna(med):
                    raise HTTPException(status_code=422, detail=f"Column '{req.column}' has no numeric values to impute from.")
                df[target] = coerced.fillna(med)
                method_label = f"median fallback ({med:.2f})"
    else:
        # Custom value — try numeric cast first
        try:
            fill_val = float(req.value)
            if fill_val == int(fill_val):
                fill_val = int(fill_val)
        except (ValueError, TypeError):
            fill_val = req.value

        df[target] = col.fillna(fill_val)
        if col.dtype == object:
            df.loc[df[target].astype(str).str.strip() == "", target] = fill_val

    n_after = int(df[target].isna().sum())
    n_filled = n_before - n_after

    store.save(session_id, df)
    store.log_action(session_id, "fill_blanks",
                     {"column": req.column, "target": target, "method": method_label, "n_filled": n_filled})
    result = _build_result(df, target)
    result.update({"column": target, "source_column": req.column,
                   "fill_value": method_label, "n_filled": n_filled,
                   "new_column": bool(req.new_column)})
    if sentinel_mask.any():
        result["n_implausible"] = int(sentinel_mask.sum())
        result["warnings"] = [
            f"{int(sentinel_mask.sum())} implausible value(s) in '{req.column}' were treated as missing for imputation."
        ]
    return result


# ── 6b. Missing-data diagnostics (MCAR vs MAR heuristic) ────────────────────────

class MissingDiagnosticsRequest(BaseModel):
    columns: Optional[List[str]] = None


@router.post("/{session_id}/missing_diagnostics")
def missing_diagnostics(session_id: str, req: Optional[MissingDiagnosticsRequest] = None):
    """Heuristic MCAR-vs-MAR hint (no AI). For each column with missing values,
    test whether its missingness indicator is associated with the OTHER numeric
    columns (Welch t-test of each other column, missing vs observed rows). Any
    association → the data depend on observed values → consistent with MAR, so
    MICE is appropriate; none → consistent with MCAR."""
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    from scipy import stats as _stats

    target_cols = req.columns if req and req.columns else list(df.columns)
    missing_cols = [c for c in target_cols if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Columns not found: {missing_cols}")

    n = len(df)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]
    columns = []
    any_mar = False
    for c in target_cols:
        max_plausible = plausibility_max_for_column(c)
        raw_miss = df[c].isna() | (df[c].astype(str).str.strip() == "")
        implausible = flag_sentinels(df[c], max_plausible)
        miss = raw_miss | implausible
        n_miss = int(miss.sum())
        n_raw_miss = int(raw_miss.sum())
        n_implausible = int(implausible.sum())
        if n_miss == 0:
            continue
        depends_on = []
        for o in num_cols:
            if o == c:
                continue
            a = pd.to_numeric(df.loc[miss, o], errors="coerce").dropna()
            b = pd.to_numeric(df.loc[~miss, o], errors="coerce").dropna()
            if len(a) >= 3 and len(b) >= 3:
                try:
                    _, p = _stats.ttest_ind(a, b, equal_var=False)
                    if pd.notna(p) and p < 0.05:
                        depends_on.append(o)
                except Exception:
                    pass
        likely = "MAR" if depends_on else "MCAR-consistent"
        if depends_on:
            any_mar = True
        columns.append({
            "name": c,
            "n_missing": n_miss,
            "n_missing_raw": n_raw_miss,
            "n_implausible": n_implausible,
            "implausible_values": sorted(sentinel_values(df[c], max_plausible)),
            "review_flag": "implausible (review)" if n_implausible else None,
            "pct": round(100.0 * n_miss / n, 1) if n else 0.0,
            "kind": _col_kind(df[c]),
            "is_numeric": bool(pd.api.types.is_numeric_dtype(df[c])),
            "depends_on": depends_on,
            "likely": likely,
        })

    if not columns:
        overall = "No missing values detected in the selected variables."
        recommendation = ""
    elif any_mar:
        overall = ("At least one selected variable's missingness is associated with other observed "
                   "variables — consistent with MAR (not MCAR).")
        recommendation = ("MAR → MICE (multiple imputation) is the appropriate choice. Mean/median "
                          "or listwise deletion can bias results when the missing fraction is non-trivial.")
    else:
        overall = ("For the selected variables, no association was detected between missingness and "
                   "the observed numeric variables "
                   "— consistent with MCAR.")
        recommendation = ("MCAR → listwise deletion is unbiased; MICE is still valid and more "
                          "efficient (keeps the full sample).")

    return {
        "columns": columns,
        "analyzed_columns": target_cols,
        "overall_hint": overall,
        "recommendation": recommendation,
        "any_mar": any_mar,
    }


# ── 7. Delete rows ──────────────────────────────────────────────────────────

class DeleteRowsRequest(BaseModel):
    row_indices: List[int]  # 0-based indices to delete


@router.post("/{session_id}/delete_rows")
def delete_rows(session_id: str, req: DeleteRowsRequest):
    df = _get_df(session_id)
    if not req.row_indices:
        raise HTTPException(status_code=422, detail="No row indices provided")
    invalid = [i for i in req.row_indices if i < 0 or i >= len(df)]
    if invalid:
        raise HTTPException(status_code=422, detail=f"Row indices out of range: {invalid}")
    df = df.drop(df.index[req.row_indices]).reset_index(drop=True)
    store.save(session_id, df)
    store.log_action(session_id, "delete_rows", {"n_deleted": len(req.row_indices)})
    return {"deleted": len(req.row_indices), "remaining_rows": len(df)}


# ── 8. Add row ─────────────────────────────────────────────────────────────

class AddRowRequest(BaseModel):
    position: int = -1  # -1 = append at end, otherwise insert at this index


@router.post("/{session_id}/add_row")
def add_row(session_id: str, req: AddRowRequest):
    df = _get_df(session_id)
    # New row with all NaN/None values
    new_row = pd.DataFrame([{col: None for col in df.columns}])
    if req.position < 0 or req.position >= len(df):
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        top = df.iloc[:req.position]
        bottom = df.iloc[req.position:]
        df = pd.concat([top, new_row, bottom], ignore_index=True)
    store.save(session_id, df)
    store.log_action(session_id, "add_row", {"position": req.position})
    return {"rows": len(df), "position": req.position}


# ── 9. Add column ──────────────────────────────────────────────────────────

class AddColumnRequest(BaseModel):
    name: str
    default_value: Optional[Any] = None  # None → all NaN
    position: int = -1  # -1 = append at end, otherwise insert at this index


@router.post("/{session_id}/add_column")
def add_column(session_id: str, req: AddColumnRequest):
    df = _get_df(session_id)
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Column name cannot be empty")
    if name in df.columns:
        raise HTTPException(status_code=422, detail=f"Column '{name}' already exists")
    df = df.copy()
    if req.position >= 0 and req.position < len(df.columns):
        df.insert(req.position, name, req.default_value)
    else:
        df[name] = req.default_value
    store.save(session_id, df)
    store.log_action(session_id, "add_column", {"name": name})
    return _build_result(df, name)


# ── 10. Paste rows (from clipboard TSV/CSV) ─────────────────────────────────

class PasteRequest(BaseModel):
    tsv: str  # tab or comma separated text (with optional header row)
    has_header: bool = True
    mode: str = "append"  # "append" or "replace"


@router.post("/{session_id}/paste")
def paste_rows(session_id: str, req: PasteRequest):
    import io as _io
    df = _get_df(session_id)

    text = req.tsv.strip()
    if not text:
        raise HTTPException(status_code=422, detail="No data to paste")

    # Auto-detect separator (tab or comma)
    first_line = text.split("\n")[0]
    sep = "\t" if "\t" in first_line else ","

    try:
        if req.has_header:
            pasted = pd.read_csv(_io.StringIO(text), sep=sep)
        else:
            pasted = pd.read_csv(_io.StringIO(text), sep=sep, header=None)
            # Assign column names from existing df if column count matches
            if len(pasted.columns) == len(df.columns):
                pasted.columns = df.columns
            else:
                pasted.columns = [f"Col_{i+1}" for i in range(len(pasted.columns))]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse pasted data: {exc}")

    if req.mode == "replace":
        df = pasted
    else:
        # Append — align columns (add missing as NaN, ignore extra)
        for col in df.columns:
            if col not in pasted.columns:
                pasted[col] = None
        for col in pasted.columns:
            if col not in df.columns:
                df[col] = None
        df = pd.concat([df, pasted[df.columns]], ignore_index=True)

    store.save(session_id, df)
    store.log_action(session_id, "paste_rows", {"n_pasted": len(pasted), "mode": req.mode})
    return {"n_pasted": len(pasted), "total_rows": len(df)}


# ── 11. Rename column ──────────────────────────────────────────────────────

class RenameRequest(BaseModel):
    old_name: str
    new_name: str


@router.post("/{session_id}/rename")
def rename_column(session_id: str, req: RenameRequest):
    df = _get_df(session_id)
    if req.old_name not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{req.old_name}' not found")
    new = req.new_name.strip()
    if not new:
        raise HTTPException(status_code=422, detail="New column name cannot be empty")
    if new in df.columns and new != req.old_name:
        raise HTTPException(status_code=422, detail=f"Column '{new}' already exists")
    df = store.rename_dataframe_column(session_id, req.old_name, new)
    store.log_action(session_id, "rename_column", {"old": req.old_name, "new": new})
    conditions = store.get_filter(session_id)
    filtered = store.get_filtered(session_id)
    case_filter = {
        "conditions": conditions,
        "selected": len(filtered),
        "total": len(df),
    } if conditions else None
    return {
        "old_name": req.old_name,
        "new_name": new,
        "case_filter": case_filter,
    }


# ── 12. Duplicate column ──────────────────────────────────────────────────────

class DuplicateColumnRequest(BaseModel):
    column: str


@router.post("/{session_id}/duplicate_column")
def duplicate_column(session_id: str, req: DuplicateColumnRequest):
    df = _get_df(session_id)
    col = req.column
    if col not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{col}' not found")

    # Generate unique name
    base = f"{col}_copy"
    new_name = base
    i = 2
    while new_name in df.columns:
        new_name = f"{base}_{i}"
        i += 1

    # Insert right after the original column
    pos = list(df.columns).index(col) + 1
    df = df.copy()
    df.insert(pos, new_name, df[col].values.copy())
    store.save(session_id, df)
    store.log_action(session_id, "duplicate_column", {"source": col, "new": new_name})
    return _build_result(df, new_name)


# ── 13. Paste cells (copy-paste within the grid) ─────────────────────────────

class PasteCellsRequest(BaseModel):
    start_row: Optional[int] = None
    start_col: Optional[str] = None
    tsv: str  # tab-separated values grid
    # Optional explicit targets preserve the visible grid order when the
    # frontend is sorted or filtered.
    row_indices: Optional[List[int]] = None
    target_columns: Optional[List[str]] = None


@router.post("/{session_id}/paste_cells")
def paste_cells(session_id: str, req: PasteCellsRequest):
    """Paste a TSV grid of values starting at a given cell position."""
    df = _get_df(session_id)
    col_list = list(df.columns)

    if req.target_columns:
        invalid_cols = [c for c in req.target_columns if c not in df.columns]
        if invalid_cols:
            raise HTTPException(status_code=400, detail=f"Columns not found: {invalid_cols}")
        target_cols = req.target_columns
    else:
        if req.start_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{req.start_col}' not found")
        start_ci = col_list.index(req.start_col)
        target_cols = col_list[start_ci:]

    if req.row_indices is not None:
        invalid_rows = [r for r in req.row_indices if r < 0 or r >= len(df)]
        if invalid_rows:
            raise HTTPException(status_code=400, detail=f"Row indices out of range: {invalid_rows}")
        target_rows = req.row_indices
    else:
        if req.start_row is None or req.start_row < 0:
            raise HTTPException(status_code=400, detail="A valid start_row is required")
        target_rows = list(range(req.start_row, len(df)))

    text = req.tsv.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    if text == "":
        return {"pasted": 0}
    lines = text.split("\n")

    df = df.copy()
    pasted = 0

    for dr, line in enumerate(lines):
        if dr >= len(target_rows):
            break
        ri = target_rows[dr]
        vals = line.split("\t")
        for dc, val in enumerate(vals):
            if dc >= len(target_cols):
                break
            col_name = target_cols[dc]
            # Coerce value
            v: Any = val.strip()
            if v == "" or v.lower() == "null":
                v = np.nan
            else:
                col_dtype = df[col_name].dtype
                try:
                    if col_dtype.kind in ("i", "u"):
                        v = int(float(v))
                    elif col_dtype.kind == "f":
                        v = float(v)
                except (ValueError, TypeError):
                    pass
            df.at[ri, col_name] = v
            pasted += 1

    store.save(session_id, df)
    store.log_action(session_id, "paste_cells", {"n_pasted": pasted})
    return {"pasted": pasted}


# ── 7. List unique values (for sex mapping UI) ────────────────────────────────

@router.get("/{session_id}/unique/{col_name:path}")
def unique_values(session_id: str, col_name: str):
    df = _get_df(session_id)
    if col_name not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{col_name}' not found")
    vals = sorted(df[col_name].dropna().unique().tolist(), key=lambda x: (str(type(x).__name__), x))
    return {"values": [str(v) for v in vals[:200]]}


# ── Advanced Data Cleaning & Imputation ───────────────────────────────────────

class DropMissingRequest(BaseModel):
    columns: List[str]


@router.post("/{session_id}/drop_missing")
def drop_missing(session_id: str, req: DropMissingRequest):
    df = _get_df(session_id)
    for c in req.columns:
        if c not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{c}' not found")
    df = df.copy()
    n_before = len(df)
    df = df.dropna(subset=req.columns).reset_index(drop=True)
    n_deleted = n_before - len(df)
    store.save(session_id, df)
    store.log_action(session_id, "drop_missing", {"columns": req.columns, "n_deleted": n_deleted})
    return {"deleted": n_deleted, "remaining_rows": len(df)}


class OutliersRequest(BaseModel):
    columns: List[str]
    method: str = "iqr"  # iqr | zscore
    threshold: float = 1.5  # 1.5 * IQR or 3.0 * SD


@router.post("/{session_id}/clean_outliers")
def clean_outliers(session_id: str, req: OutliersRequest):
    df = _get_df(session_id)
    df = df.copy()
    n_before = len(df)
    
    keep_mask = np.ones(len(df), dtype=bool)
    per_column_deleted: Dict[str, int] = {}
    warnings: List[str] = []
    for c in req.columns:
        if c not in df.columns:
            continue
        col = pd.to_numeric(df[c], errors="coerce")
        col_drop = pd.Series(False, index=df.index)
        max_plausible = plausibility_max_for_column(c)
        if max_plausible is not None:
            impossible = (col > max_plausible).fillna(False)
            if impossible.any():
                col_drop |= impossible
                warnings.append(
                    f"{c}: removed {int(impossible.sum())} value(s) above the plausible maximum ({max_plausible:g})."
                )

        body = col[~col_drop]
        if req.method == "iqr":
            q1 = body.quantile(0.25)
            q3 = body.quantile(0.75)
            iqr = q3 - q1
            low = q1 - req.threshold * iqr
            high = q3 + req.threshold * iqr
            if pd.notna(low) and pd.notna(high):
                col_drop |= (col.notna() & ((col < low) | (col > high))).fillna(False)
        else:  # zscore
            mean = body.mean()
            std = body.std(ddof=1)
            if std > 0:
                z = np.abs((col - mean) / std)
                col_drop |= (col.notna() & (z > req.threshold)).fillna(False)
        per_column_deleted[c] = int(col_drop.sum())
        keep_mask &= ~col_drop.to_numpy(dtype=bool)
                
    df = df[keep_mask].reset_index(drop=True)
    n_deleted = n_before - len(df)
    store.save(session_id, df)
    store.log_action(session_id, "clean_outliers", {
        "columns": req.columns,
        "method": req.method,
        "n_deleted": n_deleted,
        "per_column_deleted": per_column_deleted,
    })
    return {
        "deleted": n_deleted,
        "deleted_rows": n_deleted,
        "remaining_rows": len(df),
        "per_column_deleted": per_column_deleted,
        "warnings": warnings,
    }


class FindReplaceRequest(BaseModel):
    columns: List[str]
    find_value: str
    replace_value: str


@router.post("/{session_id}/find_replace")
def find_replace(session_id: str, req: FindReplaceRequest):
    df = _get_df(session_id)
    df = df.copy()
    replaced_count = 0
    per_column_replaced: Dict[str, int] = {}
    missing_columns: List[str] = []
    
    for c in req.columns:
        if c not in df.columns:
            missing_columns.append(c)
            continue
        
        # Try to coerce find/replace values if column is numeric
        f_val: Any = req.find_value
        r_val: Any = req.replace_value
        
        if pd.api.types.is_numeric_dtype(df[c]):
            try:
                f_val = float(req.find_value)
                if f_val == int(f_val):
                    f_val = int(f_val)
            except ValueError:
                pass
            try:
                r_val = float(req.replace_value)
                if r_val == int(r_val):
                    r_val = int(r_val)
            except ValueError:
                if req.replace_value == "" or req.replace_value.lower() == "nan":
                    r_val = np.nan
                    
        # Count replacements
        n_col = int((df[c] == f_val).sum())
        replaced_count += n_col
        per_column_replaced[c] = n_col
        df[c] = df[c].replace(f_val, r_val)
        
    store.save(session_id, df)
    store.log_action(session_id, "find_replace", {"columns": req.columns, "replaced_count": replaced_count})
    warnings = []
    if missing_columns:
        warnings.append(f"Skipped missing column(s): {', '.join(missing_columns)}.")
    if replaced_count == 0:
        warnings.append("No matching values found; dataset was unchanged.")
    return {
        "replaced_count": replaced_count,
        "found": replaced_count > 0,
        "changed": replaced_count > 0,
        "per_column_replaced": per_column_replaced,
        "warnings": warnings,
    }


# ── Per-column value-map replace (in place) ─────────────────────────────────────

class ReplaceValuesRequest(BaseModel):
    column: str
    # old display value (as shown in the grid) → new value, both as strings.
    mapping: Dict[str, str]


def _norm_key(series: pd.Series) -> pd.Series:
    """String view of a column for matching: trims and normalises integer-coded
    floats so "1.0" matches a user-entered "1"."""
    s = series.astype(str).str.strip()
    return s.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)


@router.post("/{session_id}/replace_values")
def replace_values(session_id: str, req: ReplaceValuesRequest):
    """Replace cell values in ONE column via a value→value map, in place.

    Backs the data-grid 'Find & Replace' modal. After replacing, if every
    non-null value parses as a number the column is cast to numeric (so e.g.
    female→0 / male→1 yields a real 0/1 predictor, not object strings). Any
    existing value labels have their keys remapped so they keep matching.
    """
    df = _get_df(session_id)
    col = req.column
    if col not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{col}' not found")
    if not req.mapping:
        raise HTTPException(status_code=422, detail="At least one replacement is required")

    df = df.copy()
    # Match against the original (normalised) string view so replacements never
    # chain (a value mapped to another mapped value is matched on the original).
    as_str = _norm_key(df[col])
    new_vals = df[col].astype(object).copy()
    n_replaced = 0
    for old, new in req.mapping.items():
        mask = as_str == str(old).strip()
        n = int(mask.sum())
        if n:
            new_vals[mask] = new
            n_replaced += n
    df[col] = new_vals

    # Auto-cast to numeric when every non-null value is a number.
    nonnull = int(df[col].notna().sum())
    coerced = pd.to_numeric(df[col], errors="coerce")
    if nonnull > 0 and int(coerced.notna().sum()) == nonnull:
        if df[col].isna().any():
            df[col] = coerced  # NaN forces float
        elif (coerced % 1 == 0).all():
            df[col] = coerced.astype(int)
        else:
            df[col] = coerced

    store.save(session_id, df)

    # Remap existing value-label keys through the same mapping so labels follow.
    meta = store.get_metadata(session_id) or {}
    vl = (meta.get(col, {}) or {}).get("value_labels")
    if vl:
        norm = {str(k).strip(): v for k, v in req.mapping.items()}
        new_vl = {str(norm.get(str(k).strip(), k)): label for k, label in vl.items()}
        store.save_metadata(session_id, {col: {"value_labels": new_vl}})

    store.log_action(session_id, "replace_values", {"column": col, "replaced_count": n_replaced})
    result = _build_result(df, col)
    result["n_replaced"] = n_replaced
    final_vl = (store.get_metadata(session_id).get(col, {}) or {}).get("value_labels")
    if final_vl:
        result["value_labels"] = final_vl
    return result


# ── Parse a text column to real dates (datetime64, in place) ────────────────────

class ParseDatesRequest(BaseModel):
    column: str
    order: str = "auto"            # auto | dmy | mdy (gg/aa ambiguity)
    century_threshold: int = 50    # 2-digit year cutoff (≤ → 2000s)
    preview_only: bool = False


def _iso_or_none(ts) -> Optional[str]:
    return None if pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d")


@router.post("/{session_id}/parse_dates")
def parse_dates(session_id: str, req: ParseDatesRequest):
    """Convert a column of mixed-format date text into real datetime64, in place.

    Backs the data-grid 'Parse as date' modal. Recognises numeric separators,
    TR/EN month names, Excel serial numbers and 2-digit years, and resolves
    DMY/MDY ambiguity across the whole column (see services.date_parser).
    Stored as datetime64 (ISO) so survival / time-series read it directly.
    """
    from services.date_parser import parse_series

    df = _get_df(session_id)
    if req.column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{req.column}' not found")

    ser, stats = parse_series(df[req.column], order=req.order, threshold=req.century_threshold)

    # Always return a small raw→parsed sample for the live preview.
    raws = list(df[req.column].head(20))
    sample = [
        {
            "raw": (None if (r is None or (isinstance(r, float) and pd.isna(r))) else str(r)),
            "parsed": _iso_or_none(v),
        }
        for r, v in zip(raws, ser.head(20))
    ]
    if req.preview_only:
        return {"column": req.column, "stats": stats, "sample": sample}

    df = df.copy()
    df[req.column] = ser
    store.save(session_id, df)
    store.log_action(session_id, "parse_dates", {"column": req.column, **stats})

    preview_values = [_iso_or_none(v) for v in ser.head(2000)]
    return {
        "name": req.column,
        "dtype": "datetime64[ns]",
        "kind": "date",
        "preview_values": preview_values,
        "n_computed": stats["n_ok"],
        "n_missing": stats["n_bad"] + stats["n_empty"],
        "stats": stats,
        "sample": sample,
    }
