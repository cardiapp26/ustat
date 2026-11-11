"""Small shared utilities for regression-related code."""

from __future__ import annotations

import pandas as pd
import numpy as np
from fastapi import HTTPException


def design_with_constant(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add the intercept, first dropping any predictor that has no variance.

    `sm.add_constant` defaults to has_constant="skip": if the design already
    contains a column of a single repeated value, it adds nothing and that
    column BECOMES the intercept. A predictor that is constant across every
    row — a single-site study still carrying its `site` column, a filtered
    cohort where a flag is 1 everywhere — was then reported as a predictor,
    with the intercept's estimate divided by its value, its standard error,
    and its p-value. On the audit frame a column holding 5.0 in all 30 rows
    came back at p = 1.2e-12 in the linear model and p = 0.044 in the
    logistic one, and no intercept row appeared at all.

    A constant column carries no information about the outcome. R aliases it
    away and reports NA; this drops it and says so.

    Returns (design with a `const` column, names of the dropped predictors).
    """
    import statsmodels.api as sm

    kept, dropped = drop_constant_columns(X)
    return sm.add_constant(kept.astype(float), has_constant="add"), dropped


def drop_constant_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove predictors that take the same value in every row.

    For models without an intercept column of their own — the ordinal fit
    carries thresholds instead — a constant predictor is collinear with those
    thresholds rather than with an intercept, but it is just as uninformative.
    """
    dropped = [
        c for c in X.columns
        if float(np.nanstd(X[c].astype(float).values)) == 0.0
    ]
    return (X.drop(columns=dropped) if dropped else X), dropped


def require_full_rank_design(X: pd.DataFrame) -> None:
    """Refuse a rank-deficient design matrix before it reaches `.fit()`.

    A design with more parameters than complete cases -- or with predictors
    that are an exact linear combination of one another -- has no unique
    least-squares solution. statsmodels does not raise for this; it fits
    anyway via a pseudo-inverse and comes back with some standard errors and
    t-statistics as NaN/Inf. That NaN/Inf then fails JSON serialization, and
    main.py's generic handler for that failure blames "a very small subgroup
    or a degenerate stratum" -- a diagnosis about tiny groups that has
    nothing to do with a model that simply had too many predictors. R's
    lm() instead drops the excess columns and reports their coefficients as
    NA; this app does not silently drop coefficients, so it refuses up
    front with the actual reason instead.
    """
    n, p = X.shape
    if n < p:
        raise HTTPException(
            status_code=400,
            detail=(
                f"The model has {p} parameters (including the intercept) but only "
                f"{n} complete cases after removing missing data -- there are more "
                "predictors than observations, so the design matrix cannot be "
                "inverted. Reduce the number of predictors or provide more data."
            ),
        )
    rank = int(np.linalg.matrix_rank(X.to_numpy(dtype=float)))
    if rank < p:
        raise HTTPException(
            status_code=400,
            detail=(
                "The design matrix is rank-deficient: one or more predictors (or "
                "encoded category levels) are an exact linear combination of the "
                "others, so their coefficients cannot be estimated separately. "
                "Check for duplicated, derived, or redundant predictor columns."
            ),
        )


def constant_column_warnings(dropped: list[str]) -> list[str]:
    """One user-facing sentence per predictor dropped for having no variance."""
    return [
        f"Predictor '{c}' has the same value in every row and carries no "
        "information about the outcome; it was excluded from the model."
        for c in dropped
    ]


def compute_vif(X: pd.DataFrame) -> dict:
    """Variance Inflation Factor per predictor, as R's car::vif defines it.

    The intercept has to be IN the matrix the auxiliary regressions are run
    on, and out of the results. Dropping the constant column first — which is
    what this used to do, and what four copies of it in the model routers
    also did — makes every auxiliary regression pass through the origin, so
    its R-squared absorbs the mean structure and the VIF comes out inflated.

    On the audit dataset that reported age at 21.17 and bmi at 21.69 where
    car::vif gives 1.008 and 1.012. Ten is the conventional "severe
    collinearity, drop this predictor" threshold, so the numbers were telling
    users to throw away sound predictors on data with no collinearity at all.

    Returns {column_name: vif_float}, intercept excluded from the output.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    Xn = X.copy().astype(float)
    if "const" not in Xn.columns:
        Xn.insert(0, "const", 1.0)
    predictors = [c for c in Xn.columns if c != "const"]
    if len(predictors) < 2:
        return {c: 1.0 for c in predictors}

    arr = Xn.values
    cols = list(Xn.columns)
    out: dict = {}
    for col in predictors:
        try:
            v = float(variance_inflation_factor(arr, cols.index(col)))
            if not np.isfinite(v):
                v = None
        except Exception:
            v = None
        out[str(col)] = v
    return out


# ── Predictor-to-design-column map ───────────────────────────────────────────

def dummy_column_map(df: pd.DataFrame, predictors: list) -> dict:
    """Design columns each predictor produces under get_dummies(drop_first=True).

    Looking columns up by the prefix "<name>_" also matches other predictors
    whose name starts the same way: with `sex` (categorical) and `sex_partner`
    in one model, the lookup for `sex` returned `sex_partner` as well, which
    created interaction terms nobody asked for and read p-values from the
    wrong variable. Encoding each predictor on its own gives the exact map.
    """
    return {
        pred: [str(c) for c in pd.get_dummies(df[[pred]], drop_first=True).columns]
        for pred in predictors
        if pred in df.columns
    }


def members_for(name: str, columns, column_map: dict | None) -> list:
    """Design columns in `columns` that belong to predictor `name`."""
    present = [str(c) for c in columns]
    if column_map is not None and name in column_map:
        return [c for c in column_map[name] if c in present]
    if name in present:
        return [name]
    prefix = f"{name}_"
    return [c for c in present if c.startswith(prefix)]


# ── Stepwise selection helpers (pure functions) ────────────────────────────────

STEPWISE_MODEL_TYPES = ("logistic", "linear")
STEPWISE_CRITERIA = ("p", "aic")


def _uni_p_for_pred(pred: str, uni_results: dict, column_map: dict | None = None) -> float:
    """Smallest univariable p among the design columns of `pred`."""
    matching = [uni_results[c]["p"] for c in members_for(pred, uni_results.keys(), column_map)]
    return min(matching) if matching else 1.0


def _compute_aic(model) -> float:
    try:
        return float(model.aic)
    except Exception:
        return float("nan")


def _fit_terms(y, df: pd.DataFrame, terms: list, model_type: str):
    """Fit outcome ~ terms with an intercept, as OLS or logistic regression."""
    from statsmodels import api as sm

    if terms:
        X_enc = pd.get_dummies(df[terms], drop_first=True).astype(float)
    else:
        X_enc = pd.DataFrame(index=df.index)
    X_const = sm.add_constant(X_enc, has_constant="add")
    if model_type == "linear":
        return sm.OLS(y, X_const).fit()
    return sm.Logit(y, X_const).fit(disp=False, maxiter=200)


def _term_p(term: str, model, column_map: dict) -> float:
    """p-value for one predictor: its own p, or a joint Wald test over its dummies."""
    names = [str(c) for c in model.params.index]
    cols = members_for(term, names, column_map)
    if not cols:
        return 1.0
    if len(cols) == 1:
        return float(model.pvalues[cols[0]])
    restriction = np.zeros((len(cols), len(names)))
    for row, col in enumerate(cols):
        restriction[row, names.index(col)] = 1.0
    return float(model.wald_test(restriction, scalar=True).pvalue)


def _check_stepwise_args(model_type: str, criterion: str) -> None:
    if model_type not in STEPWISE_MODEL_TYPES:
        raise ValueError(f"model_type must be one of {STEPWISE_MODEL_TYPES}. Got: {model_type}")
    if criterion not in STEPWISE_CRITERIA:
        raise ValueError(f"criterion must be one of {STEPWISE_CRITERIA}. Got: {criterion}")


def _safe_fit(y, df: pd.DataFrame, terms: list, model_type: str):
    try:
        return _fit_terms(y, df, terms, model_type)
    except Exception:
        return None


def _best_addition(y, df, selected, remaining, model_type, criterion, p_enter, column_map):
    """The candidate to add next, or None when none improves the model."""
    current = _safe_fit(y, df, selected, model_type) if criterion == "aic" else None
    best_var = None
    best_score = p_enter if criterion == "p" else (_compute_aic(current) if current is not None else float("inf"))
    for var in remaining:
        model = _safe_fit(y, df, selected + [var], model_type)
        if model is None:
            continue
        score = _term_p(var, model, column_map) if criterion == "p" else _compute_aic(model)
        if np.isfinite(score) and score < best_score:
            best_score, best_var = score, var
    return best_var


def stepwise_forward(
    y,
    df: pd.DataFrame,
    pred_list: list,
    p_enter: float = 0.05,
    *,
    model_type: str = "logistic",
    criterion: str = "p",
    base: list | None = None,
) -> list:
    """Forward selection. Does not mutate inputs.

    `model_type` picks the model the candidates are tested in. It used to be
    logistic regardless: a linear stepwise fitted Logit to a continuous
    outcome, every fit raised, and forward selection returned nothing.
    `criterion` "p" adds the term with the smallest p below `p_enter`; "aic"
    adds the term that lowers AIC the most. `base` terms are always in the
    model and are returned first.
    """
    _check_stepwise_args(model_type, criterion)
    selected: list = list(base or [])
    remaining = [v for v in pred_list if v not in selected]
    column_map = dummy_column_map(df, selected + remaining)
    while remaining:
        best_var = _best_addition(y, df, selected, remaining, model_type, criterion, p_enter, column_map)
        if best_var is None:
            break
        selected = selected + [best_var]
        remaining = [v for v in remaining if v != best_var]
    return selected


def _worst_term(y, df, selected, model, model_type, criterion, p_remove, column_map):
    """The term to drop next, or None when every term should stay."""
    if criterion == "p":
        worst_var, worst_p = None, p_remove
        for var in selected:
            p = _term_p(var, model, column_map)
            if p > worst_p:
                worst_p, worst_var = p, var
        return worst_var
    worst_var, best_aic = None, _compute_aic(model)
    for var in selected:
        reduced = _safe_fit(y, df, [v for v in selected if v != var], model_type)
        aic = _compute_aic(reduced) if reduced is not None else float("nan")
        if np.isfinite(aic) and aic < best_aic:
            best_aic, worst_var = aic, var
    return worst_var


def stepwise_backward(
    y,
    df: pd.DataFrame,
    pred_list: list,
    p_remove: float = 0.10,
    *,
    model_type: str = "logistic",
    criterion: str = "p",
) -> list:
    """Backward elimination. Does not mutate inputs. See `stepwise_forward`."""
    _check_stepwise_args(model_type, criterion)
    selected = list(pred_list)
    column_map = dummy_column_map(df, selected)
    while selected:
        model = _safe_fit(y, df, selected, model_type)
        if model is None:
            break
        worst_var = _worst_term(y, df, selected, model, model_type, criterion, p_remove, column_map)
        if worst_var is None:
            break
        selected = [v for v in selected if v != worst_var]
    return selected


# ── Binary outcome coding ────────────────────────────────────────────────────

def encode_binary_outcome(y: pd.Series, model_label: str = "Logistic regression") -> tuple[pd.Series, dict]:
    """Code a binary outcome as 0/1 and say which level is the event.

    Numeric (and boolean) outcomes must already be 0/1; 1 is the event. A text
    or categorical outcome needs exactly two levels and is coded in sorted
    order, the second level being the event, which is what LabelEncoder did
    here before. That choice was never reported: "Case"/"Control" modelled
    P(Control) and every odds ratio came out inverted with nothing in the
    response to show it. The returned info names both levels.

    Missing values stay NaN in the returned float Series (same index as `y`).

    Raises:
        HTTPException 422 when the outcome is not binary.
    """
    if pd.api.types.is_bool_dtype(y) or pd.api.types.is_numeric_dtype(y):
        coded = pd.to_numeric(y, errors="coerce").astype(float)
        values = sorted(coded.dropna().unique().tolist())
        if set(values) - {0.0, 1.0}:
            raise HTTPException(
                status_code=422,
                detail=f"{model_label} requires a binary 0/1 outcome. Found values: {values[:10]}",
            )
        return coded, {"event_level": "1", "reference_level": "0", "coded_from_text": False}

    as_text = y.astype(object).where(y.notna(), None)
    levels = sorted({str(v) for v in as_text if v is not None})
    if len(levels) != 2:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{model_label} requires a binary outcome with exactly two levels. "
                f"Found {len(levels)}: {levels[:10]}"
            ),
        )
    mapping = {levels[0]: 0.0, levels[1]: 1.0}
    coded = pd.Series(
        [np.nan if v is None else mapping[str(v)] for v in as_text],
        index=y.index, dtype=float,
    )
    return coded, {"event_level": levels[1], "reference_level": levels[0], "coded_from_text": True}


def outcome_level_sentence(outcome: str, info: dict) -> str:
    """Sentence naming the modelled level of a text outcome, else empty."""
    if not info.get("coded_from_text"):
        return ""
    return (
        f" The modelled event is {outcome} = '{info['event_level']}' "
        f"(reference: '{info['reference_level']}'); odds ratios above 1 mean "
        f"higher odds of '{info['event_level']}'."
    )


# ── Pairwise interactions ────────────────────────────────────────────────────

def add_pairwise_interactions(
    enc: pd.DataFrame,
    interactions: list | None,
    requested_predictors: list,
    column_map: dict | None = None,
) -> tuple[pd.DataFrame, list]:
    """Append pairwise interaction columns to a dummy-coded design matrix.

    `column_map` (from `dummy_column_map`) says which design columns belong to
    each predictor, so `sex` never picks up the columns of `sex_partner`.
    """
    if not interactions:
        return enc, []

    out = enc.copy()
    added: list = []
    requested_set = set(requested_predictors)
    for pair in interactions:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise HTTPException(status_code=422, detail=f"Each interaction must be a [colA, colB] pair. Got: {pair}")
        a_name, b_name = pair
        for nm in (a_name, b_name):
            if nm not in requested_set:
                raise HTTPException(
                    status_code=422,
                    detail=f"Interaction '{a_name} × {b_name}': '{nm}' must already be in the predictors list.",
                )
        a_members = members_for(a_name, enc.columns, column_map)
        b_members = members_for(b_name, enc.columns, column_map)
        if not a_members or not b_members:
            raise HTTPException(
                status_code=422,
                detail=f"Interaction '{a_name} × {b_name}': one or both columns did not survive dummy encoding.",
            )
        for a in a_members:
            for b in b_members:
                col = f"{a}:{b}"
                if col in out.columns:
                    continue
                out[col] = out[a] * out[b]
                added.append(col)
    return out, added
