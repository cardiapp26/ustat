"""
Joint Longitudinal-Survival Models (Phase 8 - Enhanced)

Provides advanced joint modeling of longitudinal biomarkers and time-to-event outcomes
using a pragmatic, dependency-light (statsmodels + lifelines) approach.

Capabilities:
- Time-varying two-stage joint models (counting-process format)
- Association structures: value, slope, area
- Non-linear trajectories (Natural Cubic Splines)
- Multivariate longitudinal markers
- Joint Latent Class Models (JLCM) via Gaussian Mixture Modeling of random effects
- Model comparison (AIC, BIC)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from sklearn.mixture import GaussianMixture

def _safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v

def _build_spline_formula(time_col: str, spline: bool) -> str:
    if spline:
        # Natural cubic spline with 3 degrees of freedom
        return f"cr({time_col}, df=3)"
    return time_col

ASSOCIATIONS = ("value", "slope", "area")
# Trapezoid nodes for the cumulative ("area") association; exact for a linear
# trajectory, close for a natural cubic spline.
_AREA_NODES = 33


def _clean_longitudinal(long_df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Rows usable for one mixed model: every model column observed.

    mixedlm drops NaN rows from the formula frame but takes `groups` at full
    length, so a single missing Y crashed with a shape mismatch.
    """
    present = list(dict.fromkeys(cols))
    return long_df.dropna(subset=present).reset_index(drop=True)


def _subject_covariates(long_clean: pd.DataFrame, id_col: str, predictors: List[str]) -> pd.DataFrame:
    """One row per subject with the longitudinal predictors (first record)."""
    cols = [id_col] + [c for c in dict.fromkeys(predictors) if c != id_col]
    return long_clean[cols].drop_duplicates(subset=[id_col], keep="first")


class _Trajectory:
    """Subject-specific fitted trajectory m_i(t) = x_i(t)'beta + z(t)'b_i.

    The fixed part comes from the fitted formula, so longitudinal predictors
    and spline terms are included; the random part rebuilds the random-effects
    design at the requested times with the knots of the fit. Subjects without
    longitudinal data get the population curve (b_i = 0).
    """

    def __init__(self, mdf: Any, long_clean: pd.DataFrame, id_col: str, time_col: str,
                 re_rhs: str, predictors: List[str]) -> None:
        import patsy

        self._mdf = mdf
        self._id_col = id_col
        self._time_col = time_col
        self._re_info = patsy.dmatrix(re_rhs, long_clean).design_info
        self._cov = _subject_covariates(long_clean, id_col, predictors)
        self._n_re = len(self._re_info.column_names)
        self._re = {g: np.asarray(v, dtype=float) for g, v in mdf.random_effects.items()}

    def value(self, ids: np.ndarray, times: np.ndarray) -> np.ndarray:
        import patsy

        frame = pd.DataFrame({self._id_col: ids, self._time_col: np.asarray(times, dtype=float)})
        frame = frame.merge(self._cov, on=self._id_col, how="left")
        fixed = np.asarray(self._mdf.predict(exog=frame), dtype=float)
        z = np.asarray(patsy.build_design_matrices([self._re_info], frame[[self._time_col]])[0], dtype=float)
        zeros = np.zeros(self._n_re)
        b = np.vstack([self._re.get(g, zeros) for g in ids])
        return fixed + np.einsum("ij,ij->i", z, b)

    def slope(self, ids: np.ndarray, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        h = 1e-4 * max(1.0, float(np.nanmax(np.abs(times))) if len(times) else 1.0)
        return (self.value(ids, times + h) - self.value(ids, times - h)) / (2.0 * h)

    def area(self, ids: np.ndarray, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        nodes = np.linspace(0.0, 1.0, _AREA_NODES)
        grid = np.outer(times, nodes)
        vals = self.value(np.repeat(ids, _AREA_NODES), grid.ravel()).reshape(grid.shape)
        widths = np.diff(grid, axis=1)
        return np.sum(0.5 * (vals[:, 1:] + vals[:, :-1]) * widths, axis=1)


def _extract_lmm_predictions(
    trajectory: _Trajectory, df: pd.DataFrame, time_col: str, id_col: str,
    association: List[str],
) -> pd.DataFrame:
    """Value, slope and area of each subject's fitted trajectory at each row's time."""
    preds = df[[id_col, time_col]].copy()
    ids = df[id_col].to_numpy()
    times = df[time_col].to_numpy(dtype=float)
    for name in association:
        preds[name] = getattr(trajectory, name)(ids, times)
    return preds


def _encode_cox_predictors(cox_df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
    """Reference-coded indicators for every non-numeric Cox predictor.

    Integer category codes made a nominal variable one linear term.
    """
    nominal = [c for c in predictors if not pd.api.types.is_numeric_dtype(cox_df[c])]
    if not nominal:
        return cox_df
    dummies = pd.get_dummies(cox_df[nominal].astype(str), drop_first=True, dtype=float)
    return pd.concat([cox_df.drop(columns=nominal), dummies], axis=1)


def _fit_cox(cox_df: pd.DataFrame, **fit_kwargs: Any) -> tuple[CoxPHFitter, float]:
    """Unpenalised Cox fit; a small ridge only when that does not converge.

    Returns the fitter and the penalizer actually used, which the response
    reports. A fixed, unreported penalizer of 0.05 shrank every coefficient
    and its standard error.
    """
    from lifelines.exceptions import ConvergenceError

    try:
        return CoxPHFitter().fit(cox_df, **fit_kwargs), 0.0
    except (ConvergenceError, np.linalg.LinAlgError, ValueError):
        return CoxPHFitter(penalizer=_FALLBACK_PENALIZER).fit(cox_df, **fit_kwargs), _FALLBACK_PENALIZER


_FALLBACK_PENALIZER = 0.05


def _penalizer_note(penalizer: float) -> str:
    if penalizer == 0.0:
        return "Cox stage fitted without penalisation."
    return (
        f"The unpenalised Cox stage did not converge; a ridge penalty of {penalizer} was applied, "
        "so the hazard ratios and their standard errors are shrunk towards the null."
    )


def _to_counting_process(
    long_df: pd.DataFrame, surv_df: pd.DataFrame, 
    id_col: str, time_col: str, duration_col: str, event_col: str
) -> pd.DataFrame:
    """
    Converts survival data into counting process format (start, stop, event) 
    split by longitudinal measurement times.
    """
    records = []
    
    surv_df_idx = surv_df.set_index(id_col, drop=True)
    long_groups = long_df.groupby(id_col)
    
    for id_val, s_row in surv_df_idx.iterrows():
        event_time = s_row[duration_col]
        event_status = s_row[event_col]
        
        if id_val in long_groups.groups:
            l_df = long_groups.get_group(id_val)
            m_times = sorted(l_df[l_df[time_col] <= event_time][time_col].unique())
        else:
            m_times = []
            
        if not m_times or m_times[0] > 0:
            m_times = [0.0] + m_times
            
        if m_times[-1] < event_time:
            m_times.append(event_time)
            
        for i in range(len(m_times) - 1):
            start = m_times[i]
            stop = m_times[i+1]
            if start == stop:
                continue
                
            is_last = (i == len(m_times) - 2)
            records.append({
                id_col: id_val,
                "start": float(start),
                "stop": float(stop),
                event_col: int(event_status) if is_last else 0,
                duration_col: float(stop)  # map for evaluation
            })
            
    cp_df = pd.DataFrame(records)
    
    # Join baseline predictors - use original surv_df which is not indexed
    baseline_cols = [c for c in surv_df.columns if c not in [id_col, duration_col, event_col]]
    if baseline_cols:
        cp_df = cp_df.merge(surv_df[[id_col] + baseline_cols], on=id_col, how="left")
        
    return cp_df


def fit_time_varying_joint_model(
    long_df: pd.DataFrame,
    surv_df: pd.DataFrame,
    id_col: str = "id",
    time_col: str = "time",
    y_cols: List[str] = ["Y"],
    long_predictors: Optional[List[str]] = None,
    surv_predictors: Optional[List[str]] = None,
    duration_col: str = "duration",
    event_col: str = "event",
    association: List[str] = ["value"],
    time_spline: bool = False
) -> Dict[str, Any]:
    """
    Fits a time-varying two-stage joint model.
    """
    if not long_predictors:
        long_predictors = []
    if not surv_predictors:
        surv_predictors = []
    if not association:
        association = ["value"]
    
    unknown = [a for a in association if a not in ASSOCIATIONS]
    if unknown:
        raise ValueError(f"Unknown association structure(s): {unknown}. Use any of {list(ASSOCIATIONS)}.")

    long_all = long_df.loc[:, ~long_df.columns.duplicated()].copy()

    lmm_summaries = {}
    total_loglike_lmm = 0.0

    cp_df = _to_counting_process(long_df, surv_df, id_col, time_col, duration_col, event_col)

    eval_df = cp_df[[id_col, duration_col]].rename(columns={duration_col: time_col})
    tv_covariates = []
    time_term = _build_spline_formula(time_col, time_spline)

    for y_col in y_cols:
        # Sort groups so statsmodels handles properly
        long_clean = _clean_longitudinal(
            long_all, [id_col, time_col, y_col] + long_predictors,
        ).sort_values([id_col, time_col]).reset_index(drop=True)
        formula = f"{y_col} ~ {time_term}"
        if long_predictors:
            formula += " + " + " + ".join(long_predictors)

        md = smf.mixedlm(formula, long_clean, groups=long_clean[id_col], re_formula=f"~{time_term}")
        try:
            mdf = md.fit(method=["lbfgs"], reml=False) # ML for correct AIC
        except Exception:
            mdf = md.fit(method=["cg"], reml=False)

        total_loglike_lmm += mdf.llf

        lmm_summaries[y_col] = {
            "params": mdf.params.to_dict(),
            "bse": mdf.bse.to_dict(),
            "pvalues": mdf.pvalues.to_dict(),
        }

        trajectory = _Trajectory(mdf, long_clean, id_col, time_col, time_term, long_predictors)
        preds = _extract_lmm_predictions(trajectory, eval_df, time_col, id_col, association)

        for a in association:
            col_name = f"{y_col}_{a}"
            cp_df[col_name] = preds[a].to_numpy()
            tv_covariates.append(col_name)

    cox_predictors = surv_predictors + tv_covariates
    # A subject with no value for a longitudinal predictor has no trajectory.
    cox_df = cp_df[[id_col, "start", "stop", event_col] + cox_predictors].dropna()
    cox_df = _encode_cox_predictors(cox_df, cox_predictors)
    n_subjects_cox = int(cox_df[id_col].nunique())

    cph, penalizer = _fit_cox(
        cox_df,
        duration_col="stop",
        event_col=event_col,
        entry_col="start",
        cluster_col=id_col,
        robust=True,
    )

    coefs = []
    for var in cph.params_.index:
        beta = float(cph.params_[var])
        coefs.append({
            "variable": str(var),
            "coef": round(beta, 5),
            "hr": round(np.exp(beta), 4),
            "se": round(float(cph.standard_errors_[var]), 5),
            "p": _safe(cph.summary.loc[var, "p"] if "p" in cph.summary.columns else None),
        })
        
    num_params = cph.summary.shape[0] + sum(len(s['params']) for s in lmm_summaries.values())
    total_ll = cph.log_likelihood_ + total_loglike_lmm
    aic = 2 * num_params - 2 * total_ll
    bic = num_params * np.log(max(n_subjects_cox, 1)) - 2 * total_ll

    return {
        "model": "time_varying_joint_lmm_cox",
        "lmm_summaries": lmm_summaries,
        "cox_coefficients": coefs,
        "cox_concordance": round(float(cph.concordance_index_), 4),
        "n_subjects": n_subjects_cox,
        "cox_penalizer": penalizer,
        "aic": round(aic, 2),
        "bic": round(bic, 2),
        "log_likelihood": round(total_ll, 2),
        "note": (
            "Time-varying two-stage joint model. Association evaluated on each subject's fitted "
            "trajectory (fixed effects plus random effects) via the counting process. "
            + _penalizer_note(penalizer)
        ),
    }

def fit_latent_class_joint_model(
    long_df: pd.DataFrame,
    surv_df: pd.DataFrame,
    id_col: str = "id",
    time_col: str = "time",
    y_cols: List[str] = ["Y"],
    long_predictors: Optional[List[str]] = None,
    surv_predictors: Optional[List[str]] = None,
    duration_col: str = "duration",
    event_col: str = "event",
    latent_classes: int = 2
) -> Dict[str, Any]:
    """
    Fits a pragmatic Joint Latent Class Model (JLCM) using GMM on extracted random effects.
    """
    if not long_predictors:
        long_predictors = []
    if not surv_predictors:
        surv_predictors = []
    
    long_all = long_df.loc[:, ~long_df.columns.duplicated()].copy()

    lmm_summaries = {}
    re_dfs = []
    total_loglike_lmm = 0.0

    for y_col in y_cols:
        long_clean = _clean_longitudinal(
            long_all, [id_col, time_col, y_col] + long_predictors,
        ).sort_values([id_col, time_col]).reset_index(drop=True)
        formula = f"{y_col} ~ {time_col}"
        if long_predictors:
            formula += " + " + " + ".join(long_predictors)
            
        md = smf.mixedlm(formula, long_clean, groups=long_clean[id_col], re_formula=f"~{time_col}")
        mdf = md.fit(method=["lbfgs"], reml=False)
        total_loglike_lmm += mdf.llf
        
        lmm_summaries[y_col] = {
            "params": mdf.params.to_dict(),
        }
        
        re_dict = mdf.random_effects
        re_df = pd.DataFrame.from_dict(re_dict, orient="index")
        re_df.columns = [f"{y_col}_RE_{c}" for c in re_df.columns]
        re_df.index.name = id_col
        re_dfs.append(re_df)
        
    combined_re = pd.concat(re_dfs, axis=1).reset_index()
    
    # GMM
    re_features = combined_re.drop(columns=[id_col]).fillna(0)
    gmm = GaussianMixture(n_components=latent_classes, random_state=42)
    gmm.fit(re_features)
    
    combined_re['latent_class'] = gmm.predict(re_features)
    
    surv_work = surv_df.merge(combined_re[[id_col, 'latent_class']], on=id_col, how="inner")
    
    cox_predictors = surv_predictors + ['latent_class']
    cox_df = surv_work[[duration_col, event_col] + cox_predictors].dropna()
    # Class labels are arbitrary GMM indices: indicator-code them (reference
    # = class 0) so each class gets its own hazard ratio. As one integer
    # term, three classes produced a single HR per "step" in label order.
    cox_df["latent_class"] = cox_df["latent_class"].astype(int).astype(str)
    cox_df = _encode_cox_predictors(cox_df, cox_predictors)

    cph, penalizer = _fit_cox(cox_df, duration_col=duration_col, event_col=event_col, robust=True)
    
    coefs = []
    for var in cph.params_.index:
        beta = float(cph.params_[var])
        coefs.append({
            "variable": str(var),
            "coef": round(beta, 5),
            "hr": round(np.exp(beta), 4),
            "se": round(float(cph.standard_errors_[var]), 5),
            "p": _safe(cph.summary.loc[var, "p"] if "p" in cph.summary.columns else None),
        })
        
    num_params = cph.summary.shape[0] + sum(len(s['params']) for s in lmm_summaries.values()) + (latent_classes - 1)
    total_ll = cph.log_likelihood_ + total_loglike_lmm
    aic = 2 * num_params - 2 * total_ll
    bic = num_params * np.log(len(surv_df)) - 2 * total_ll
        
    return {
        "model": "latent_class_joint_lmm_cox",
        "lmm_summaries": lmm_summaries,
        "cox_coefficients": coefs,
        "cox_concordance": round(float(cph.concordance_index_), 4),
        "n_subjects": int(len(surv_work)),
        "n_classes": latent_classes,
        "cox_penalizer": penalizer,
        "class_proportions": {str(k): round(float(v), 3) for k, v in combined_re['latent_class'].value_counts(normalize=True).items()},
        "aic": round(aic, 2),
        "bic": round(bic, 2),
        "log_likelihood": round(total_ll, 2),
        "note": (
            "JLCM using LMM random effects + GMM clustering + Cox; latent classes enter the Cox "
            "stage as indicators against class 0. " + _penalizer_note(penalizer)
        ),
    }
