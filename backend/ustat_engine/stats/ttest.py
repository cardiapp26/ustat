"""Independent-samples and one-sample t-test.

Extracted from the `POST /api/stats/ttest` handler in
routers/stats/inferential.py. The computational body is verbatim: both
branches, every expression and every string, are the ones that were there
before. That is the whole point -- this module has to be provably the same
arithmetic as the code it replaced, now runnable in either runtime.

First analysis to move that actually reads a dataset. Power and meta-analysis
compute from numbers typed into a form; this one needs rows, so it is the first
real exercise of the frame path: `build_envelope` on the server,
`frame_from_envelope` in the worker, and `registry.run` refusing the frame if
the Select Cases it was cut under has moved on.

Translation from the FastAPI original, and nothing else:
  - `class TTestRequest(BaseModel)` became the `_P` shim below, with the same
    field defaults and the same `group_col` alias, so the copied body still
    says `req.<field>`. Keeping those references intact is what lets this file
    be diffed against its source.
  - `_get_df(req.session_id)` disappeared: the frame is a parameter now. The
    session lookup is the server's business and stayed in the router.
  - `raise HTTPException(status_code=X, detail=Y)` became
    `raise EngineError(Y, status_hint=X)`, detail strings unchanged.
  - `_sanitize(ret)` became `jsonsafe.sanitize(ret)` -- the same recursion over
    the same non-finite floats, plus numpy scalars the original left to
    FastAPI's encoder. The browser has no such encoder behind it.

pandas and `clean_two_level` are imported inside `_two_level_work` rather than
at module scope. `ustat_engine/__init__.py` imports this module so the registry
is complete before anything asks it a question, and a module-level `import
pandas` would therefore make `import ustat_engine` fail in a browser that had
only loaded numpy/scipy/statsmodels for a power analysis.
"""
from __future__ import annotations

from scipy import stats as scipy_stats

from ..errors import EngineError
from ..jsonsafe import sanitize
from ..registry import register
from ..spec import AnalysisSpec
from ..text.ttest import (
    methods_ttest_ind, methods_ttest_one,
    results_ttest_ind, results_ttest_one,
    r_ttest_ind, r_ttest_one,
)
from .assumptions import check_equal_variances, check_normality
from .effect_sizes import (
    cohen_d, cohen_d_one_sample, group_summary, welch_satterthwaite_df,
)

_METHODS = ("auto", "student", "welch")


class _P:
    """Shim standing in for the pydantic `TTestRequest`, so the copied
    function body below can keep using `req.<field>` unchanged. Defaults
    mirror the field declarations of `TTestRequest` exactly, including the
    `group_col` validation alias."""

    def __init__(self, params):
        if not params.get("column"):
            raise EngineError("Field 'column' is required.", status_hint=422)
        self.column = params["column"]
        # AliasChoices("group_column", "group_col") — either spelling arrives
        # from a caller that predates the rename.
        self.group_column = params.get("group_column")
        if self.group_column is None:
            self.group_column = params.get("group_col")
        mu = params.get("mu", 0.0)
        self.mu = 0.0 if mu is None else mu
        # "auto" lets Levene pick Student vs Welch; the other two force it.
        self.method = params.get("method") or "auto"
        if self.method not in _METHODS:
            raise EngineError(
                f"Field 'method' must be one of {_METHODS}.", status_hint=422
            )
        # Legacy alias, kept so existing callers keep working. None means "not
        # supplied" — it used to default to True while the handler ignored it
        # entirely and always let Levene decide.
        self.equal_var = params.get("equal_var")


def _two_level_work(df, value_col: str, group_col: str):
    import pandas as pd

    from ..frame.category_health import clean_two_level

    work = df[[value_col, group_col]].copy()
    cleaned = clean_two_level(work[group_col])
    work[group_col] = cleaned.series
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    return work.dropna(), cleaned.warnings


# ─────────────────────────────────────────────────────────────────────────
# Everything below this line is copied verbatim (HTTPException raises
# rewritten to EngineError, `_sanitize` to `sanitize`, `df` now a parameter
# instead of a session lookup, and `req` bound to the `_P` shim instead of a
# pydantic model) from the `ttest` handler in
# backend/routers/stats/inferential.py.
# ─────────────────────────────────────────────────────────────────────────

def run_ttest(frame, params: dict) -> dict:
    # Deferred for the same reason as pandas in `_two_level_work`:
    # `frame.levels` imports pandas at module scope, and this module is
    # imported at `import ustat_engine` time.
    from ..frame.levels import sorted_groups

    req = _P(params)
    df = frame
    col = df[req.column].dropna()

    if req.group_column:
        work, warnings = _two_level_work(df, req.column, req.group_column)
        groups = sorted_groups(work[req.group_column])
        if len(groups) != 2:
            raise EngineError("Group column must have exactly 2 groups", status_hint=400)
        g1 = work[work[req.group_column] == groups[0]][req.column].values.astype(float)
        g2 = work[work[req.group_column] == groups[1]][req.column].values.astype(float)

        # Assumption checks
        assumptions = [check_normality(g1, str(groups[0])), check_normality(g2, str(groups[1])),
                       check_equal_variances([g1, g2], [str(groups[0]), str(groups[1])],
                                             on_violation="Welch correction applied")]
        # Precedence: explicit method > legacy equal_var > Levene.
        if req.method == "welch":
            use_welch, chosen_by = True, "request (method)"
        elif req.method == "student":
            use_welch, chosen_by = False, "request (method)"
        elif req.equal_var is not None:
            use_welch, chosen_by = (not req.equal_var), "request (equal_var)"
        else:
            use_welch, chosen_by = (not assumptions[2]["met"]), "auto (Levene)"
        stat, p = scipy_stats.ttest_ind(g1, g2, equal_var=not use_welch)
        sig = bool(p < 0.05)
        es = cohen_d(g1, g2)
        p_str = '<0.001' if p < 0.001 else f'{p:.4f}'

        ret = {
            "test": f"Independent samples t-test{' (Welch)' if use_welch else ''}",
            "group1": str(groups[0]), "n1": len(g1), "mean1": float(g1.mean()),
            "group2": str(groups[1]), "n2": len(g2), "mean2": float(g2.mean()),
            # df must match the test that produced t and p. Welch uses the
            # fractional Satterthwaite df; only the pooled test uses n1+n2-2.
            "t": float(stat), "p": float(p),
            "df": welch_satterthwaite_df(g1, g2) if use_welch else float(len(g1) + len(g2) - 2),
            "df_method": "welch_satterthwaite" if use_welch else "pooled",
            "variance_assumption": "welch" if use_welch else "student",
            "variance_assumption_selected_by": chosen_by,
            "significant": sig,
            "effect_sizes": [es],
            "assumptions": assumptions,
            "summary": {str(groups[0]): group_summary(g1, str(groups[0])),
                        str(groups[1]): group_summary(g2, str(groups[1]))},
            "interpretation": f"{'Significant' if sig else 'No significant'} difference between groups (t = {stat:.3f}, p = {p_str}, Hedges' g = {es['value']:.3f} [{es['magnitude']}])",
            "methods_text": methods_ttest_ind(req.column, req.group_column, use_welch),
            "r_code": r_ttest_ind(req.column, req.group_column, use_welch),
        }
        if warnings:
            ret["warnings"] = warnings
        ret["result_text"] = results_ttest_ind(ret)
        return sanitize(ret)
    else:
        x = col.astype(float).values
        stat, p = scipy_stats.ttest_1samp(x, req.mu)
        sig = bool(p < 0.05)
        es = cohen_d_one_sample(x, req.mu)
        p_str = '<0.001' if p < 0.001 else f'{p:.4f}'

        ret = {
            "test": "One-sample t-test",
            "mu": req.mu, "n": len(x),
            "mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "t": float(stat), "p": float(p), "df": int(len(x) - 1),
            "significant": sig,
            "effect_sizes": [es],
            "assumptions": [check_normality(x, req.column)],
            "summary": {"sample": group_summary(x, "Sample")},
            "interpretation": f"Mean {'differs from' if sig else 'does not differ from'} {req.mu} (t = {stat:.3f}, p = {p_str}, Cohen's d = {es['value']:.3f} [{es['magnitude']}])",
            "methods_text": methods_ttest_one(req.column, req.mu),
            "r_code": r_ttest_one(req.column, req.mu),
        }
        ret["result_text"] = results_ttest_one(ret)
        return sanitize(ret)


register(AnalysisSpec(
    id="stats.ttest",
    fn=run_ttest,
    needs_frame=True,
    # pandas because the frame arrives as one; scipy for the test itself and
    # for Shapiro/Levene.
    #
    # statsmodels is here reluctantly, and it is the one place this spec is not
    # the obvious minimum. `check_normality` switches to the Lilliefors test at
    # n >= 50 and imports statsmodels *inside the function* to do it -- which on
    # the server is a lazy import and in the browser is a hard failure, because
    # a Pyodide package that was not in the boot plan cannot be fetched from
    # inside a synchronous `runPython`. Any group of 50 or more would crash a
    # local t-test that declared only numpy/scipy/pandas, so the dependency is
    # declared for the branch rather than for the common case.
    deps=("numpy", "scipy", "pandas", "statsmodels"),
    required_columns=lambda p: [
        p.get("column"),
        p.get("group_column") or p.get("group_col"),
    ],
    cost_key="stats.ttest",
    doc=(
        "Independent-samples t-test (Student or Welch, chosen by Levene unless "
        "the caller forces one) or a one-sample t-test against mu."
    ),
    tags=("inferential", "parametric"),
))


__all__ = ["run_ttest"]
