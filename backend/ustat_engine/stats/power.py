"""Statistical power and sample size.

Extracted from routers/stats/inferential.py (the `POST /api/stats/power`
handler and its `_power_result_text` helper). The computational body is
verbatim: every expression, branch and constant is the one that was there
before, because this module's whole purpose is to be provably the same
arithmetic as the code it replaced, now runnable in either runtime.

First analysis to move, because it is the only interesting one that reads no
dataset at all -- every number it needs was typed into the form. That makes it
the honest test of the two-runtime split without any of the data-marshalling
question mixed in.

Translation from the FastAPI original, and nothing else:
  - `class PowerRequest(BaseModel)` became the `_P` shim below, applying the
    same field defaults, so the copied body still says `req.<field>`. Keeping
    those references intact is what lets this file be diffed against its
    source.
  - `raise HTTPException(status_code=X, detail=Y)` became
    `raise EngineError(Y, status_hint=X)`, detail strings unchanged.
"""
from __future__ import annotations

from scipy import stats as scipy_stats

from .._logging import logger
from ..errors import EngineError
from ..registry import register
from ..spec import AnalysisSpec


class _P:
    """Shim standing in for the pydantic `PowerRequest`, so the copied
    function body below can keep using `req.<field>` unchanged. Defaults
    mirror the field declarations of `PowerRequest` exactly."""

    def __init__(self, params):
        if "test" not in params:
            raise EngineError("Field 'test' is required.", status_hint=422)
        if "solve_for" not in params:
            raise EngineError("Field 'solve_for' is required.", status_hint=422)
        self.test = params["test"]
        self.solve_for = params["solve_for"]
        self.alpha = params.get("alpha", 0.05)
        self.power = params.get("power", None)
        self.effect_size = params.get("effect_size", None)
        self.n = params.get("n", None)
        self.tails = params.get("tails", 2)
        self.k_groups = params.get("k_groups", 3)
        self.ratio = params.get("ratio", 1.0)
        self.p1 = params.get("p1", None)
        self.p2 = params.get("p2", None)
        self.log_or = params.get("log_or", None)
        self.p_event = params.get("p_event", None)
        self.r2_other = params.get("r2_other", 0.0)
        self.hr = params.get("hr", None)
        self.event_rate = params.get("event_rate", None)
        self.p_exposed = params.get("p_exposed", 0.5)
        self.attrition = params.get("attrition", 0.0)


# ─────────────────────────────────────────────────────────────────────────
# Everything below this line is copied verbatim (module-level HTTPException
# raises rewritten to EngineError; `req` now bound to the `_P` shim instead
# of a pydantic model) from run_power / _power_result_text in
# backend/routers/stats/inferential.py.
# ─────────────────────────────────────────────────────────────────────────

def run_power(params: dict) -> dict:
    req = _P(params)

    import numpy as np
    from scipy.stats import norm
    from statsmodels.stats.power import (
        TTestIndPower, TTestPower,
        FTestAnovaPower, NormalIndPower, GofChisquarePower,
    )

    alt = "two-sided" if req.tails == 2 else "larger"
    a   = req.alpha

    if req.attrition is not None and not (0.0 <= req.attrition < 1.0):
        # 1.0 would divide by zero, and a rate above it is not a rate. Caught
        # here rather than left to produce an infinite recruitment target.
        raise EngineError(
            "Expected attrition must be at least 0 and below 1 (0.10 = 10%).",
            status_hint=422,
        )

    # ── Upfront field validation for the statsmodels-backed power classes ──
    # solve_power() raises a bare ValueError ("need exactly one keyword that
    # is None") when the caller omits a required field. Left unhandled, that
    # crashes with a 500 instead of a usable client error. Validate the
    # solve_for-appropriate required fields before dispatching.
    if req.test in ("t_two", "t_one", "anova", "chi2", "proportion"):
        missing = []
        if req.solve_for == "n":
            if req.power is None:
                missing.append("power")
            if req.test != "proportion" and req.effect_size is None:
                missing.append("effect_size")
        elif req.solve_for == "power":
            if req.n is None:
                missing.append("n")
            if req.test != "proportion" and req.effect_size is None:
                missing.append("effect_size")
        elif req.solve_for == "effect_size":
            if req.n is None:
                missing.append("n")
            if req.power is None:
                missing.append("power")
        if missing:
            raise EngineError(
                f"Power analysis for test='{req.test}', solve_for='{req.solve_for}' "
                f"requires: {', '.join(missing)}.",
                status_hint=400,
            )

    def _ceil(x): return int(np.ceil(float(x)))

    def _curve(pw_fn, n_end, n_start=4, steps=80):
        pts, step = [], max(1, (n_end - n_start) // steps)
        for n in range(n_start, n_end + 1, step):
            try:
                pwr = float(pw_fn(n))
                if 0 <= pwr <= 1:
                    pts.append({"n": n, "power": round(pwr, 4)})
            except Exception as exc:
                logger.debug("Power curve point failed for n={}: {}", n, exc)
        return pts

    result, label, curve = None, "", []

    # ── Two-sample t-test ──
    if req.test == "t_two":
        ana = TTestIndPower()
        ratio = req.ratio or 1.0
        def pw(n): return ana.solve_power(effect_size=req.effect_size, nobs1=n, alpha=a, power=None, ratio=ratio, alternative=alt)

        if req.solve_for == "n":
            n1 = _ceil(ana.solve_power(effect_size=req.effect_size, nobs1=None, alpha=a, power=req.power, ratio=ratio, alternative=alt))
            result = n1
            label  = f"n₁ = {n1},  n₂ = {_ceil(n1*ratio)},  total N = {n1 + _ceil(n1*ratio)}"
            curve  = _curve(pw, max(n1 * 4, 100))
        elif req.solve_for == "power":
            result = float(ana.solve_power(effect_size=req.effect_size, nobs1=req.n, alpha=a, power=None, ratio=ratio, alternative=alt))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n) * 4, 100))
        else:
            result = float(ana.solve_power(effect_size=None, nobs1=req.n, alpha=a, power=req.power, ratio=ratio, alternative=alt))
            label  = f"Minimum detectable Cohen's d = {result:.4f}"
            d = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=d, nobs1=n, alpha=a, power=None, ratio=ratio, alternative=alt), max(int(req.n)*4, 100))

    # ── One-sample / paired t-test ──
    elif req.test == "t_one":
        ana = TTestPower()
        def pw(n): return ana.solve_power(effect_size=req.effect_size, nobs=n, alpha=a, power=None, alternative=alt)

        if req.solve_for == "n":
            n = _ceil(ana.solve_power(effect_size=req.effect_size, nobs=None, alpha=a, power=req.power, alternative=alt))
            result, label, curve = n, f"n = {n}", _curve(pw, max(n*4, 100))
        elif req.solve_for == "power":
            result = float(ana.solve_power(effect_size=req.effect_size, nobs=req.n, alpha=a, power=None, alternative=alt))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n)*4, 100))
        else:
            result = float(ana.solve_power(effect_size=None, nobs=req.n, alpha=a, power=req.power, alternative=alt))
            label  = f"Minimum detectable Cohen's d = {result:.4f}"
            d = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=d, nobs=n, alpha=a, power=None, alternative=alt), max(int(req.n)*4, 100))

    # ── One-way ANOVA ──
    elif req.test == "anova":
        # statsmodels' FTestAnovaPower takes `nobs` as the TOTAL sample size.
        # The panel asks the user for participants PER GROUP — its own field
        # help says so and its own label reports "n/group" — and the two were
        # never translated. With four groups that made every answer wrong by a
        # factor of k, in both directions and in the dangerous one:
        #
        #   power for f = 0.25, 52/group, k = 4 → reported 0.275, truth 0.864
        #   n for f = 0.25, 80% power, k = 4    → reported 179/group (716 total)
        #                                         where 45/group (179 total) does
        #
        # So a properly powered study was called hopeless, and a study needing
        # 179 participants was told to recruit 716. Both agree with
        # pwr.anova.test once the per-group count is converted here.
        ana, k = FTestAnovaPower(), req.k_groups

        def pw(n_per_group):
            return ana.solve_power(effect_size=req.effect_size,
                                   nobs=n_per_group * k, alpha=a, power=None,
                                   k_groups=k)

        if req.solve_for == "n":
            total = ana.solve_power(effect_size=req.effect_size, nobs=None,
                                    alpha=a, power=req.power, k_groups=k)
            n = _ceil(total / k)
            result, label, curve = n, f"n/group = {n},  total N = {n*k}", _curve(pw, max(n*4, 100))
        elif req.solve_for == "power":
            result = float(pw(int(req.n)))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n)*4, 100))
        else:
            result = float(ana.solve_power(effect_size=None, nobs=int(req.n) * k,
                                           alpha=a, power=req.power, k_groups=k))
            label  = f"Minimum detectable Cohen's f = {result:.4f}"
            f_es = result
            curve  = _curve(
                lambda n_per_group: ana.solve_power(
                    effect_size=f_es, nobs=n_per_group * k, alpha=a,
                    power=None, k_groups=k),
                max(int(req.n)*4, 100))

    # ── Pearson correlation (Fisher-z) ──
    elif req.test == "correlation":
        tails = req.tails

        def corr_power(r, n):
            """Cohen (1988) as pwr.r.test implements it.

            Two things were missing from the plain Fisher-z version that was
            here. The critical value came from a normal, where the test that
            will actually be run is a t on n-2 df; and the Fisher z of a
            sample correlation is biased upward by r/(2(n-1)), which the
            transform is normally applied with. Both matter most at the small
            n a power calculation is usually about: at n = 12, r = 0.5 this
            reported 0.378 against pwr.r.test's 0.400, and at n = 85, r = 0.3
            it reported 0.800 — landing exactly on the 80% convention that
            decides whether a study goes ahead — where the correct value is
            0.804.
            """
            if abs(r) >= 1 or n <= 3:
                return float("nan")
            r = abs(r)
            tside = 1 if tails == 1 else 2
            t_c = scipy_stats.t.ppf(1 - a / tside, df=n - 2)
            r_c = np.sqrt(t_c ** 2 / (t_c ** 2 + n - 2))
            z_r = np.arctanh(r) + r / (2 * (n - 1))
            z_rc = np.arctanh(r_c)
            power = norm.cdf((z_r - z_rc) * np.sqrt(n - 3))
            if tails == 2:
                power += norm.cdf((-z_r - z_rc) * np.sqrt(n - 3))
            return float(power)

        def corr_solve_n(r, pwr):
            for n in range(4, 100001):
                if corr_power(r, n) >= pwr:
                    return n
            return 100001

        def corr_solve_r(n, pwr):
            from scipy.optimize import brentq
            try:
                return float(brentq(lambda r: corr_power(r, n) - pwr, 1e-6, 1 - 1e-6))
            except Exception as exc:
                logger.debug("Correlation power root solve failed: {}", exc)
                return None

        r_es = req.effect_size
        if req.solve_for == "n":
            n = corr_solve_n(r_es, req.power)
            result, label = n, f"n = {n}"
            curve = _curve(lambda n: corr_power(r_es, n), max(n*4, 100))
        elif req.solve_for == "power":
            result = corr_power(r_es, req.n)
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(lambda n: corr_power(r_es, n), max(int(req.n)*4, 100))
        else:
            r_sol = corr_solve_r(req.n, req.power)
            result = r_sol
            label  = f"Minimum detectable r = {r_sol:.4f}" if r_sol else "Could not converge"
            if r_sol:
                curve = _curve(lambda n: corr_power(r_sol, n), max(int(req.n)*4, 100))

    # ── Two proportions (Cohen's h) ──
    elif req.test == "proportion":
        ana   = NormalIndPower()
        ratio = req.ratio or 1.0
        p1    = req.p1 if req.p1 is not None else 0.5
        p2    = req.p2 if req.p2 is not None else 0.3
        h_from_p = abs(float(2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p2))))

        if req.solve_for == "effect_size":
            eff = float(ana.solve_power(effect_size=None, nobs1=req.n, alpha=a, power=req.power, ratio=ratio, alternative=alt))
            result = abs(eff)
            label  = f"Minimum detectable Cohen's h = {result:.4f}"
            h_sol = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=h_sol, nobs1=n, alpha=a, power=None, ratio=ratio, alternative=alt), max(int(req.n)*4, 100))
        else:
            eff = req.effect_size if req.effect_size is not None else h_from_p
            def pw(n): return ana.solve_power(effect_size=eff, nobs1=n, alpha=a, power=None, ratio=ratio, alternative=alt)
            if req.solve_for == "n":
                n1 = _ceil(ana.solve_power(effect_size=eff, nobs1=None, alpha=a, power=req.power, ratio=ratio, alternative=alt))
                result, label, curve = n1, f"n₁ = {n1},  n₂ = {_ceil(n1*ratio)},  total N = {n1+_ceil(n1*ratio)}", _curve(pw, max(n1*4, 100))
            else:
                result = float(ana.solve_power(effect_size=eff, nobs1=req.n, alpha=a, power=None, ratio=ratio, alternative=alt))
                label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
                curve  = _curve(pw, max(int(req.n)*4, 100))

    # ── Logistic regression ──
    elif req.test == "logistic":
        from scipy.stats import norm as _norm

        def _required_n(log_or, p_event, power_target, alpha_target, r2_other, tails):
            z_a = _norm.ppf(1 - alpha_target / (2 if tails == 2 else 1))
            z_b = _norm.ppf(power_target)
            return float(((z_a + z_b) ** 2) / (p_event * (1 - p_event) * (log_or ** 2) * (1 - (r2_other or 0.0))))

        def _power_from_n(log_or, p_event, n_total, alpha_target, r2_other, tails):
            z_a = _norm.ppf(1 - alpha_target / (2 if tails == 2 else 1))
            se = float(np.sqrt(1.0 / (n_total * p_event * (1 - p_event) * (1 - (r2_other or 0.0)))))
            z = abs(log_or) / se if se > 0 else 0.0
            return float(_norm.cdf(z - z_a))

        if req.solve_for == "effect_size":
            log_or = None
        elif not req.log_or and req.effect_size is not None:
            log_or = float(np.log(req.effect_size))
        elif req.log_or is not None:
            log_or = float(req.log_or) if req.log_or <= 0 else float(np.log(req.log_or))
        else:
            raise EngineError("Logistic power needs 'log_or' (or 'effect_size' = OR).", status_hint=400)
        if req.p_event is None or not (0 < req.p_event < 1):
            raise EngineError("Logistic power needs 'p_event' in (0, 1).", status_hint=400)
        r2 = req.r2_other if req.r2_other is not None else 0.0

        def pw(n_): return _power_from_n(log_or, req.p_event, n_, a, r2, req.tails)
        if req.solve_for == "n":
            n_req = _ceil(_required_n(log_or, req.p_event, req.power or 0.8, a, r2, req.tails))
            result, label = n_req, f"n = {n_req}"
            curve = _curve(pw, max(n_req * 4, 200))
        elif req.solve_for == "power":
            result = float(pw(int(req.n)))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n) * 4, 200))
        else:
            from scipy.optimize import brentq
            if req.n is None:
                raise EngineError("Solving minimum detectable OR needs 'n'.", status_hint=400)
            try:
                or_solved = brentq(
                    lambda lo: _power_from_n(lo, req.p_event, int(req.n), a, r2, req.tails) - (req.power or 0.8),
                    1e-3, 5.0,
                )
                result = float(np.exp(or_solved))
                label  = f"Minimum detectable OR = {result:.3f}"
                ll = float(or_solved)
                curve = _curve(lambda n_: _power_from_n(ll, req.p_event, n_, a, r2, req.tails), max(int(req.n)*4, 200))
            except Exception:
                logger.exception("Solving OR in power analysis failed")
                result = None
                label = "Could not solve for OR — try different power / n combination."

    # ── Cox PH ──
    elif req.test == "survival_cox":
        from scipy.stats import norm as _norm

        if req.solve_for != "effect_size" and (req.hr is None or req.hr <= 0):
            raise EngineError("Cox power needs 'hr' > 0.", status_hint=400)
        if req.event_rate is None or not (0 < req.event_rate < 1):
            raise EngineError("Cox power needs 'event_rate' in (0, 1).", status_hint=400)
        p_exp = req.p_exposed if req.p_exposed is not None else 0.5
        if not (0 < p_exp < 1):
            raise EngineError("'p_exposed' must be in (0, 1).", status_hint=400)
        r2 = req.r2_other or 0.0
        log_hr = float(np.log(req.hr)) if req.hr is not None else None

        def _events_required(power_target):
            z_a = _norm.ppf(1 - a / (2 if req.tails == 2 else 1))
            z_b = _norm.ppf(power_target)
            return ((z_a + z_b) ** 2) / (p_exp * (1 - p_exp) * (log_hr ** 2))

        def _n_required(power_target):
            d = _events_required(power_target)
            return d / (req.event_rate * (1 - r2))

        def _power_from_n(n_total):
            z_a = _norm.ppf(1 - a / (2 if req.tails == 2 else 1))
            d = n_total * req.event_rate * (1 - r2)
            if d <= 0:
                return 0.0
            se = float(np.sqrt(1.0 / (d * p_exp * (1 - p_exp))))
            z = abs(log_hr) / se if se > 0 else 0.0
            return float(_norm.cdf(z - z_a))

        def pw(n_): return _power_from_n(n_)
        if req.solve_for == "n":
            n_req = _ceil(_n_required(req.power or 0.8))
            d_req = _ceil(_events_required(req.power or 0.8))
            result, label = n_req, f"n = {n_req} (events = {d_req})"
            curve = _curve(pw, max(n_req * 4, 200))
        elif req.solve_for == "power":
            result = float(pw(int(req.n)))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n) * 4, 200))
        else:
            d_total = int(req.n) * req.event_rate * (1 - r2)
            if d_total > 0:
                z_a = _norm.ppf(1 - a / (2 if req.tails == 2 else 1))
                z_b = _norm.ppf(req.power or 0.8)
                lh = (z_a + z_b) / np.sqrt(d_total * p_exp * (1 - p_exp))
                result = float(np.exp(lh))
                label  = f"Minimum detectable HR = {result:.3f}"
                def _power_from_n_at_lh(n_total):
                    d = n_total * req.event_rate * (1 - r2)
                    if d <= 0:
                        return 0.0
                    se = float(np.sqrt(1.0 / (d * p_exp * (1 - p_exp))))
                    z = abs(lh) / se if se > 0 else 0.0
                    return float(_norm.cdf(z - z_a))
                curve = _curve(_power_from_n_at_lh, max(int(req.n) * 4, 200))
            else:
                result, label = None, "Insufficient events to solve for HR."

    # ── Chi-square GOF ──
    elif req.test == "chi2":
        ana    = GofChisquarePower()
        n_bins = req.k_groups
        def pw(n): return ana.solve_power(effect_size=req.effect_size, nobs=n, alpha=a, power=None, n_bins=n_bins)

        if req.solve_for == "n":
            n = _ceil(ana.solve_power(effect_size=req.effect_size, nobs=None, alpha=a, power=req.power, n_bins=n_bins))
            result, label, curve = n, f"n = {n}", _curve(pw, max(n*4, 100))
        elif req.solve_for == "power":
            result = float(ana.solve_power(effect_size=req.effect_size, nobs=req.n, alpha=a, power=None, n_bins=n_bins))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n)*4, 100))
        else:
            result = float(ana.solve_power(effect_size=None, nobs=req.n, alpha=a, power=req.power, n_bins=n_bins))
            label  = f"Minimum detectable Cohen's w = {result:.4f}"
            w_es = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=w_es, nobs=n, alpha=a, power=None, n_bins=n_bins), max(int(req.n)*4, 100))
    else:
        raise EngineError(f"Unknown test: {req.test}", status_hint=400)

    # Attrition correction: one place, after whichever branch produced the n,
    # so a test cannot be added later that quietly skips it.
    attrition = float(req.attrition or 0.0)
    n_corrected = None
    if req.solve_for == "n" and attrition > 0 and result is not None:
        per_group = int(np.ceil(float(result) / (1.0 - attrition)))
        n_corrected = per_group
        if req.test in ("t_two", "proportion"):
            ratio = req.ratio or 1.0
            label += (
                f"  ·  allowing {attrition*100:.0f}% attrition: enrol n₁ = {per_group}, "
                f"n₂ = {_ceil(per_group*ratio)}, total N = {per_group + _ceil(per_group*ratio)}"
            )
        else:
            label += f"  ·  allowing {attrition*100:.0f}% attrition: enrol {per_group}"

    result_text = _power_result_text(req, result, n_corrected)
    return {
        "result": float(result) if result is not None else None,
        "label": label,
        "curve": curve,
        "result_text": result_text,
        # The raw n stays `result`; this is the recruitment target.
        "n_corrected": n_corrected,
        "attrition": attrition if attrition > 0 else None,
    }


def _power_result_text(req, result, n_corrected=None) -> str:
    import numpy as np

    if result is None:
        return ""

    def _attrition_note() -> str:
        if not n_corrected:
            return ""
        pct = float(req.attrition or 0.0) * 100
        return (
            f" Allowing for {pct:.0f}% attrition, enrol {n_corrected} per group "
            f"({int(np.ceil(float(result)))} / {1 - float(req.attrition):.2f}) so that "
            f"{int(np.ceil(float(result)))} complete the study."
            if req.test in ("t_two", "proportion") else
            f" Allowing for {pct:.0f}% attrition, enrol {n_corrected} "
            f"({int(np.ceil(float(result)))} / {1 - float(req.attrition):.2f})."
        )

    a_str = f"{req.alpha}" if req.alpha else "0.05"
    pw_pct = int((req.power or 0.8) * 100)

    # Regression powers have their own effect metric (OR / HR) and design inputs.
    if req.test == "logistic":
        orr = req.log_or if (req.log_or and req.log_or > 0) else None
        or_txt = f"OR = {orr}" if orr else "the specified odds ratio"
        if req.solve_for == "n":
            return (f"You need a total N = {int(np.ceil(result))} for a logistic regression to "
                    f"detect {or_txt} (event prevalence {req.p_event}) with {pw_pct}% power at "
                    f"alpha = {a_str}.")
        if req.solve_for == "power":
            return (f"With N = {req.n} and event prevalence {req.p_event}, your logistic regression "
                    f"has {round(result*100,1)}% power to detect {or_txt} at alpha = {a_str}.")
        return (f"With N = {req.n} at {pw_pct}% power (event prevalence {req.p_event}, alpha = {a_str}), "
                f"the smallest detectable odds ratio is {result:.3f}.")
    if req.test == "survival_cox":
        hr_txt = f"HR = {req.hr}" if req.hr else "the specified hazard ratio"
        if req.solve_for == "n":
            return (f"You need a total N = {int(np.ceil(result))} (event rate {req.event_rate}, "
                    f"exposed fraction {req.p_exposed}) for a Cox model to detect {hr_txt} with "
                    f"{pw_pct}% power at alpha = {a_str}.")
        if req.solve_for == "power":
            return (f"With N = {req.n} (event rate {req.event_rate}), your Cox model has "
                    f"{round(result*100,1)}% power to detect {hr_txt} at alpha = {a_str}.")
        return (f"With N = {req.n} at {pw_pct}% power (event rate {req.event_rate}, alpha = {a_str}), "
                f"the smallest detectable hazard ratio is {result:.3f}.")

    test_names = {
        "t_two": "two-sample t-test", "t_one": "one-sample/paired t-test",
        "anova": "one-way ANOVA", "correlation": "correlation test",
        "proportion": "two-proportion z-test", "chi2": "chi-square test",
    }
    test_name = test_names.get(req.test, req.test)

    if req.solve_for == "n":
        n = int(np.ceil(result))
        total = n * 2 if req.test in ("t_two", "proportion") else n
        ratio_note = f" (ratio {req.ratio}:1)" if hasattr(req, "ratio") and req.ratio and req.ratio != 1 else ""
        return (
            f"You need {n} participants per group{ratio_note} (total N = {total}) "
            f"for a {test_name} to detect an effect size of {req.effect_size} "
            f"with {int((req.power or 0.8) * 100)}% power at alpha = {a_str}."
            + _attrition_note()
        )
    elif req.solve_for == "power":
        pwr = round(result * 100, 1)
        return (
            f"With n = {req.n} per group and effect size = {req.effect_size}, "
            f"your {test_name} has {pwr}% power to detect a real effect at alpha = {a_str}. "
            f"{'This exceeds the 80% minimum standard.' if result >= 0.8 else 'This is below the 80% minimum — consider increasing your sample size.'}"
        )
    elif req.solve_for == "effect_size":
        return (
            f"With n = {req.n} per group at {int((req.power or 0.8) * 100)}% power (alpha = {a_str}), "
            f"your {test_name} can detect a minimum effect size of {result:.3f}. "
            f"Effects smaller than this will likely be missed."
        )
    return ""


register(
    AnalysisSpec(
        id="stats.power",
        fn=lambda params: run_power(params),
        needs_frame=False,
        deps=("numpy", "scipy", "statsmodels"),
        required_columns=lambda params: [],
        cost_key="stats.power",
        doc="Power and sample size for t, ANOVA, correlation, proportion, "
            "chi-square, logistic and Cox designs.",
        tags=("power", "planning"),
    )
)
