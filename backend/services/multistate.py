"""
Multi-State Models and Dynamic Prediction (Phase 7)

Provides tools for illness-death and other multi-state survival analyses.

Current focus (pragmatic, no new heavy dependencies):
- Transition-specific modeling using stacked long-format data + cause-specific Cox or Weibull.
- Estimation of cumulative transition probabilities (Aalen-Johansen style generalization).
- Simple dynamic prediction from a landmark time given current state.
- Expected Length of Stay (ELOS) and Markov assumption testing.
- Individual-level Monte Carlo Microsimulation.
- Cluster-robust Bootstrap Confidence Intervals.

All functions return immutable structures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter


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


def _fit_multistate_models(
    long_df: pd.DataFrame,
    id_col: str,
    from_state_col: str,
    to_state_col: str,
    entry_col: str,
    exit_col: str,
    event_col: str,
    predictors: List[str],
    transitions: Optional[List[Tuple[int, int]]] = None,
    model_type: str = "cox",
) -> Tuple[Dict[Tuple[int, int], Any], Dict[str, Any]]:
    """
    Fits either Cox or Weibull transition-specific models and returns both
    the model objects (for predictions) and JSON-serializable statistics.
    """
    if transitions is None:
        # Infer transitions from data
        transitions = (
            long_df[[from_state_col, to_state_col]]
            .drop_duplicates()
            .apply(tuple, axis=1)
            .tolist()
        )
        
    # Get all unique states from transitions
    all_states = set(long_df[from_state_col].unique()) | set(long_df[to_state_col].unique()) - {-1}
    states = sorted(list(all_states))
    # Filter transitions to valid ones (state to state, not censored)
    transitions = [(f, t) for (f, t) in transitions if f in states and t in states and f != t]

    results: Dict[str, Any] = {}
    models = {}

    for (from_s, to_s) in transitions:
        mask = (long_df[from_state_col] == from_s) & (long_df[to_state_col] == to_s)
        sub = long_df[mask].copy()

        if len(sub) < 15 or sub[event_col].sum() < 3:
            results[f"{from_s}->{to_s}"] = {"warning": "Too few events for reliable estimation"}
            continue

        # Prepare design matrix
        X = sub[predictors].copy()
        for c in predictors:
            if X[c].dtype == object or str(X[c].dtype).startswith("category"):
                X[c] = pd.Categorical(X[c]).codes
            X[c] = pd.to_numeric(X[c], errors="coerce")

        work = pd.concat([
            sub[[entry_col, exit_col, event_col]].reset_index(drop=True),
            X.reset_index(drop=True)
        ], axis=1)
        work = work.dropna()

        if len(work) < 10:
            results[f"{from_s}->{to_s}"] = {"warning": "Too few rows after dropping missing values"}
            continue

        try:
            if model_type == "cox":
                cph = CoxPHFitter()
                cph.fit(
                    work,
                    duration_col=exit_col,
                    event_col=event_col,
                    entry_col=entry_col,
                    robust=True,
                )
                coef_table = []
                summary = cph.summary
                for var in cph.params_.index:
                    beta = float(cph.params_[var])
                    se_val = float(summary.loc[var, "se(coef)"]) if "se(coef)" in summary.columns else float(cph.standard_errors_[var])
                    coef_table.append({
                        "variable": str(var),
                        "coef": round(beta, 5),
                        "hr": round(np.exp(beta), 4),
                        "se": round(se_val, 5),
                        "p": _safe(summary.loc[var, "p"] if "p" in summary.columns else None),
                    })
                models[(from_s, to_s)] = cph
                results[f"{from_s}->{to_s}"] = {
                    "n_transitions": int(len(work)),
                    "n_events": int(work[event_col].sum()),
                    "coefficients": coef_table,
                    "concordance": round(float(cph.concordance_index_), 4),
                    "aic": round(float(cph.AIC_partial_), 2) if hasattr(cph, "AIC_partial_") else None,
                }
            elif model_type == "weibull":
                from lifelines import WeibullAFTFitter
                aft = WeibullAFTFitter()
                # Create duration column
                work["duration"] = (work[exit_col] - work[entry_col]).clip(lower=1e-5)
                aft_data = work[["duration", event_col] + predictors]
                aft.fit(
                    aft_data,
                    duration_col="duration",
                    event_col=event_col,
                )
                
                coef_table = []
                summary = aft.summary
                for var in predictors:
                    if ("lambda_", var) in summary.index:
                        row = summary.loc[("lambda_", var)]
                        beta = float(row["coef"])
                        se_val = float(row["se(coef)"]) if "se(coef)" in row else float(row.get("se", 0.0))
                        coef_table.append({
                            "variable": str(var),
                            "coef": round(beta, 5),
                            "aft_factor": round(np.exp(beta), 4),
                            "se": round(se_val, 5),
                            "p": _safe(row["p"]),
                        })
                
                rho_val = float(np.exp(aft.params_[("rho_", "Intercept")]))
                models[(from_s, to_s)] = aft
                results[f"{from_s}->{to_s}"] = {
                    "n_transitions": int(len(work)),
                    "n_events": int(work[event_col].sum()),
                    "coefficients": coef_table,
                    "shape_rho": round(rho_val, 4),
                    "aic": round(float(aft.AIC_), 2) if hasattr(aft, "AIC_") else None,
                }
        except Exception as e:
            results[f"{from_s}->{to_s}"] = {"error": str(e)}

    return models, results


def test_markov_assumption(
    long_df: pd.DataFrame,
    id_col: str,
    from_state_col: str,
    to_state_col: str,
    entry_col: str,
    exit_col: str,
    event_col: str,
    predictors: List[str],
    transitions: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """
    Test the Markov assumption for transitions in a multi-state model.
    For each transition s -> k, we fit a Cox model including predictors and
    the entry time (time at which the subject entered state s).
    If the coefficient of the entry time is statistically significant (p < 0.05),
    the Markov assumption is violated (i.e. transition hazard depends on when
    the state was entered, meaning clock-reset/semi-Markov might be more appropriate).
    """
    if transitions is None:
        transitions = (
            long_df[[from_state_col, to_state_col]]
            .drop_duplicates()
            .apply(tuple, axis=1)
            .tolist()
        )

    # Get unique valid states
    all_states = set(long_df[from_state_col].unique()) | set(long_df[to_state_col].unique()) - {-1}
    states = sorted(list(all_states))
    transitions = [(f, t) for (f, t) in transitions if f in states and t in states and f != t]

    results = {}

    for (from_s, to_s) in transitions:
        mask = (long_df[from_state_col] == from_s) & (long_df[to_state_col] == to_s)
        sub = long_df[mask].copy()

        # We can only test entry time if there's sufficient variation and enough events
        if len(sub) < 30 or sub[event_col].sum() < 5:
            results[f"{from_s}->{to_s}"] = {
                "status": "Skipped",
                "reason": "Too few transitions or events to test the Markov assumption."
            }
            continue

        # Prepare design matrix
        X = sub[predictors].copy()
        for c in X.columns:
            if X[c].dtype == object or str(X[c].dtype).startswith("category"):
                X[c] = pd.Categorical(X[c]).codes
            X[c] = pd.to_numeric(X[c], errors="coerce")

        work = pd.concat([
            sub[[entry_col, exit_col, event_col]].reset_index(drop=True),
            X.reset_index(drop=True)
        ], axis=1).dropna()
        
        # Add entry time as a covariate with a different name to avoid duplicate columns
        work["_entry_covariate"] = work[entry_col]

        # Check variation in entry times
        if work["_entry_covariate"].std() < 1e-4:
            results[f"{from_s}->{to_s}"] = {
                "status": "Skipped",
                "reason": "No variation in entry time (all subjects entered at the same moment)."
            }
            continue

        try:
            cph = CoxPHFitter()
            cph.fit(
                work,
                duration_col=exit_col,
                event_col=event_col,
                entry_col=entry_col,
                robust=True,
            )
            
            # Extract stats for the '_entry_covariate' variable
            if "_entry_covariate" in cph.summary.index:
                row = cph.summary.loc["_entry_covariate"]
                coef = float(row["coef"])
                se_val = float(row["se(coef)"]) if "se(coef)" in row else float(row.get("se", 0.0))
                p_val = float(row["p"])
                hr = float(np.exp(coef))
                
                is_violated = p_val < 0.05
                results[f"{from_s}->{to_s}"] = {
                    "status": "Tested",
                    "coef": round(coef, 5),
                    "hr": round(hr, 4),
                    "se": round(se_val, 5),
                    "p_value": round(p_val, 5),
                    "markov_assumption_violated": bool(is_violated),
                    "interpretation": (
                        f"Entry time as a covariate gave p = {p_val:.4f}. "
                        + (
                            "Since p < 0.05 the Markov assumption is VIOLATED — a clock-reset "
                            "(semi-Markov) model may fit better."
                            if is_violated else
                            "Since p >= 0.05 the Markov assumption holds."
                        )
                    )
                }
            else:
                results[f"{from_s}->{to_s}"] = {
                    "status": "Error",
                    "reason": "The entry-time covariate could not be estimated."
                }
        except Exception as e:
            results[f"{from_s}->{to_s}"] = {
                "status": "Error",
                "reason": str(e)
            }

    return results


def fit_multistate_transitions(
    long_df: pd.DataFrame,
    id_col: str,
    from_state_col: str,
    to_state_col: str,
    entry_col: str,
    exit_col: str,
    event_col: str,
    predictors: List[str],
    transitions: Optional[List[Tuple[int, int]]] = None,
    transition_model_type: str = "cox",
) -> Dict[str, Any]:
    """
    Fit transition-specific Cox or Weibull models on long-format multi-state data.
    """
    models, results = _fit_multistate_models(
        long_df=long_df,
        id_col=id_col,
        from_state_col=from_state_col,
        to_state_col=to_state_col,
        entry_col=entry_col,
        exit_col=exit_col,
        event_col=event_col,
        predictors=predictors,
        transitions=transitions,
        model_type=transition_model_type,
    )
    
    # Run Markov assumption tests
    markov_tests = test_markov_assumption(
        long_df=long_df,
        id_col=id_col,
        from_state_col=from_state_col,
        to_state_col=to_state_col,
        entry_col=entry_col,
        exit_col=exit_col,
        event_col=event_col,
        predictors=predictors,
        transitions=transitions,
    )

    return {
        "transitions_estimated": list(results.keys()),
        "results": results,
        "markov_assumption_tests": markov_tests,
        "model_type": transition_model_type,
        "note": f"Transition-specific cause-specific {transition_model_type} models.",
    }


def compute_elos(
    state_probabilities: pd.DataFrame,
    times: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Expected Length of Stay (ELOS) in each state over the prediction horizon.
    ELOS_j = \\int_{t_start}^{t_end} P_j(u) du
    """
    results = {}
    for col in state_probabilities.columns:
        if col.startswith("state_"):
            state_name = col.split("_")[1]
            probs = state_probabilities[col].values
            # Compute integral using trapezoidal rule
            if hasattr(np, "trapezoid"):
                elos_val = float(np.trapezoid(probs, times))
            else:
                elos_val = float(np.trapz(probs, times))
            results[state_name] = round(elos_val, 4)
    return results


def _compute_state_occupation_probabilities_internal(
    cumhaz: Dict[Tuple[int, int], np.ndarray],
    times: np.ndarray,
    states: List[int],
    initial_state: int,
) -> pd.DataFrame:
    """
    Internal helper to compute state occupation probabilities using Aalen-Johansen-style discrete integration.
    """
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}
    
    P = np.zeros((len(times), n_states))
    P[0, state_to_idx[initial_state]] = 1.0
    
    for i in range(1, len(times)):
        trans_probs = np.eye(n_states)
        
        for (from_s, to_s), ch_series in cumhaz.items():
            if from_s not in state_to_idx or to_s not in state_to_idx:
                continue
            f_idx = state_to_idx[from_s]
            t_idx = state_to_idx[to_s]
            
            ch_prev = ch_series[i-1]
            ch_now = ch_series[i]
            d_ch = max(0.0, ch_now - ch_prev)
            
            if d_ch > 0:
                p_trans = min(0.99, 1.0 - np.exp(-d_ch))
                trans_probs[f_idx, f_idx] -= p_trans
                trans_probs[f_idx, t_idx] += p_trans
                
        P[i] = P[i-1] @ trans_probs
        # Clip negative probabilities and re-normalize to prevent numerical errors
        P[i] = np.clip(P[i], 0.0, 1.0)
        row_sum = P[i].sum()
        if row_sum > 0:
            P[i] = P[i] / row_sum
            
    cols = [f"state_{s}" for s in states]
    return pd.DataFrame(np.round(P, 5), index=times, columns=cols)


def compute_state_occupation_probabilities(
    baseline_hazards: Dict[Tuple[int, int], pd.DataFrame],
    beta_dict: Dict[Tuple[int, int], np.ndarray],
    covariate_vector: np.ndarray,
    times: np.ndarray,
    initial_state: int = 0,
) -> pd.DataFrame:
    """
    Compute state occupation probabilities for a multi-state process (especially illness-death)
    using a discrete approximation to the Aalen-Johansen estimator.
    """
    if len(times) < 2:
        raise ValueError("times must have at least 2 points")

    # Determine states
    all_states = set()
    for (f, t) in baseline_hazards.keys():
        all_states.add(f)
        all_states.add(t)
    states = sorted(all_states)
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    # Prepare cumulative hazards on the time grid (linear interpolation)
    cumhaz = {}
    for trans, bh in baseline_hazards.items():
        if trans not in beta_dict:
            continue
        beta = beta_dict[trans]
        lp = float(np.dot(covariate_vector, beta)) if len(beta) > 0 else 0.0
        t_grid = np.sort(bh.index.values) if hasattr(bh, 'index') else np.array(bh['time'])
        ch = np.cumsum(np.array(bh['hazard']) * np.exp(lp)) if 'hazard' in bh.columns else np.zeros_like(t_grid)
        cumhaz[trans] = (t_grid, ch)

    # Discrete time grid
    grid = np.sort(times)
    P = np.zeros((len(grid), n_states))
    P[0, state_to_idx[initial_state]] = 1.0

    for i in range(1, len(grid)):
        t_prev, t_now = grid[i-1], grid[i]
        trans_probs = np.eye(n_states)

        for (from_s, to_s), (t_grid, ch) in cumhaz.items():
            if from_s not in state_to_idx or to_s not in state_to_idx:
                continue
            f_idx = state_to_idx[from_s]
            t_idx = state_to_idx[to_s]

            ch_prev = np.interp(t_prev, t_grid, ch, left=0.0, right=ch[-1])
            ch_now = np.interp(t_now, t_grid, ch, left=0.0, right=ch[-1])
            d_ch = max(0.0, ch_now - ch_prev)

            if d_ch > 0:
                p_trans = min(0.99, 1 - np.exp(-d_ch))
                trans_probs[f_idx, f_idx] -= p_trans
                trans_probs[f_idx, t_idx] += p_trans

        P[i] = P[i-1] @ trans_probs

    P = np.clip(P, 0.0, 1.0)
    P = P / P.sum(axis=1, keepdims=True)

    cols = [f"state_{s}" for s in states]
    return pd.DataFrame(np.round(P, 5), index=grid, columns=cols)


def run_multistate_microsimulation(
    models: Dict[Tuple[int, int], Any],
    covariate_df: pd.DataFrame,
    initial_state: int,
    times: np.ndarray,
    model_type: str = "cox",
    n_simulations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run an individual-level Monte Carlo microsimulation.
    Simulates the forward state trajectory of n_simulations individuals
    sharing the same covariate vector, starting from initial_state at times[0].
    """
    rng = np.random.default_rng(seed)
    states_set = set(f for f, t in models.keys()) | set(t for f, t in models.keys())
    states = sorted(list(states_set))
    state_to_idx = {s: i for i, s in enumerate(states)}
    
    # Pre-calculate cumulative hazards for each transition on the time grid
    cumhaz = {}
    for (from_s, to_s), fitter in models.items():
        try:
            if model_type == "cox":
                pred = fitter.predict_cumulative_hazard(covariate_df, times=times)
            else:
                durations = times - times[0]
                pred = fitter.predict_cumulative_hazard(covariate_df, times=durations)
            
            ch = pred.iloc[:, 0].values
            ch = np.maximum.accumulate(ch)
            cumhaz[(from_s, to_s)] = ch
        except Exception:
            cumhaz[(from_s, to_s)] = np.zeros_like(times)

    trajectories = np.full((n_simulations, len(times)), initial_state)
    sample_paths = []
    
    for sim_idx in range(n_simulations):
        current_state = initial_state
        entry_time = times[0]
        
        path = [{"state": int(current_state), "time": float(entry_time)}]
        
        for t_idx in range(1, len(times)):
            t_prev = times[t_idx - 1]
            t_curr = times[t_idx]
            
            possible = [trans for trans in models.keys() if trans[0] == current_state]
            if not possible:
                trajectories[sim_idx, t_idx:] = current_state
                break
            
            transition_probs = {}
            total_prob = 0.0
            
            for trans in possible:
                from_s, to_s = trans
                ch_series = cumhaz[trans]
                
                if model_type == "cox":
                    ch_prev = ch_series[t_idx - 1]
                    ch_curr = ch_series[t_idx]
                else:
                    dur_prev = t_prev - entry_time
                    dur_curr = t_curr - entry_time
                    durations_grid = times - times[0]
                    ch_prev = np.interp(dur_prev, durations_grid, ch_series)
                    ch_curr = np.interp(dur_curr, durations_grid, ch_series)
                
                d_ch = max(0.0, ch_curr - ch_prev)
                p_trans = min(0.99, 1.0 - np.exp(-d_ch))
                transition_probs[to_s] = p_trans
                total_prob += p_trans
            
            if total_prob > 0:
                if total_prob > 1.0:
                    scale = 1.0 / total_prob
                    for k in transition_probs:
                        transition_probs[k] *= scale
                    p_any = 1.0
                else:
                    p_any = total_prob
                
                u = rng.uniform(0, 1)
                if u < p_any:
                    states_dest = list(transition_probs.keys())
                    probs_dest = [transition_probs[k] / p_any for k in states_dest]
                    next_state = rng.choice(states_dest, p=probs_dest)
                    
                    current_state = next_state
                    entry_time = t_curr
                    path.append({"state": int(current_state), "time": float(t_curr)})
            
            trajectories[sim_idx, t_idx] = current_state
            
        if sim_idx < 5:
            sample_paths.append({
                "simulation_id": sim_idx + 1,
                "path": path
            })
            
    sim_probs = np.zeros((len(times), len(states)))
    for t_idx in range(len(times)):
        counts = pd.Series(trajectories[:, t_idx]).value_counts()
        for s in states:
            s_idx = state_to_idx[s]
            sim_probs[t_idx, s_idx] = counts.get(s, 0) / n_simulations
            
    cols = [f"state_{s}" for s in states]
    sim_probs_df = pd.DataFrame(np.round(sim_probs, 5), index=times, columns=cols)
    
    return {
        "state_probabilities": sim_probs_df.to_dict(orient="list"),
        "sample_paths": sample_paths,
        "n_simulations": n_simulations
    }


def compute_multistate_bootstrap_ci(
    long_df: pd.DataFrame,
    id_col: str,
    from_state_col: str,
    to_state_col: str,
    entry_col: str,
    exit_col: str,
    event_col: str,
    predictors: List[str],
    covariate_df: pd.DataFrame,
    times: np.ndarray,
    initial_state: int,
    model_type: str = "cox",
    n_bootstrap: int = 50,
    seed: int = 42,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Compute cluster-robust bootstrap confidence intervals for state occupation probabilities.
    """
    rng = np.random.default_rng(seed)
    unique_ids = long_df[id_col].unique()
    n_subjects = len(unique_ids)
    
    all_states = set(long_df[from_state_col].unique()) | set(long_df[to_state_col].unique()) - {-1}
    states = sorted(list(all_states))
    boot_results = {f"state_{s}": [] for s in states}
    
    for boot_idx in range(n_bootstrap):
        sampled_ids = rng.choice(unique_ids, size=n_subjects, replace=True)
        
        boot_dfs = []
        for new_id, old_id in enumerate(sampled_ids):
            subj_df = long_df[long_df[id_col] == old_id].copy()
            subj_df[id_col] = new_id
            boot_dfs.append(subj_df)
        
        if not boot_dfs:
            continue
        boot_df = pd.concat(boot_dfs, ignore_index=True)
        
        try:
            models = {}
            transitions = (
                boot_df[[from_state_col, to_state_col]]
                .drop_duplicates()
                .apply(tuple, axis=1)
                .tolist()
            )
            transitions = [(f, t) for (f, t) in transitions if f in states and t in states and f != t]
            
            for (from_s, to_s) in transitions:
                mask = (boot_df[from_state_col] == from_s) & (boot_df[to_state_col] == to_s)
                sub = boot_df[mask].copy()
                if len(sub) < 15 or sub[event_col].sum() < 3:
                    continue
                    
                X = sub[predictors].copy()
                for c in predictors:
                    if X[c].dtype == object or str(X[c].dtype).startswith("category"):
                        X[c] = pd.Categorical(X[c]).codes
                    X[c] = pd.to_numeric(X[c], errors="coerce")
                
                work = pd.concat([
                    sub[[entry_col, exit_col, event_col]].reset_index(drop=True),
                    X.reset_index(drop=True)
                ], axis=1).dropna()
                
                if model_type == "cox":
                    cph = CoxPHFitter()
                    cph.fit(work, duration_col=exit_col, event_col=event_col, entry_col=entry_col, robust=False)
                    models[(from_s, to_s)] = cph
                else:
                    from lifelines import WeibullAFTFitter
                    aft = WeibullAFTFitter()
                    work["duration"] = (work[exit_col] - work[entry_col]).clip(lower=1e-5)
                    aft.fit(work[["duration", event_col] + predictors], duration_col="duration", event_col=event_col)
                    models[(from_s, to_s)] = aft
            
            if not models:
                continue
                
            boot_cumhaz = {}
            for trans, fitter in models.items():
                if model_type == "cox":
                    pred = fitter.predict_cumulative_hazard(covariate_df, times=times)
                else:
                    durations = times - times[0]
                    pred = fitter.predict_cumulative_hazard(covariate_df, times=durations)
                
                ch = pred.iloc[:, 0].values
                ch = np.maximum.accumulate(ch)
                boot_cumhaz[trans] = ch
                
            boot_probs = _compute_state_occupation_probabilities_internal(
                boot_cumhaz,
                times,
                states,
                initial_state
            )
            
            for col in boot_probs.columns:
                boot_results[col].append(boot_probs[col].values)
                
        except Exception:
            continue
            
    lower_ci = {}
    upper_ci = {}
    
    for col in boot_results.keys():
        curves = boot_results[col]
        if not curves:
            lower_ci[col] = [0.0] * len(times)
            upper_ci[col] = [1.0] * len(times)
            continue
            
        curves_stack = np.vstack(curves)
        lower = np.percentile(curves_stack, 2.5, axis=0)
        upper = np.percentile(curves_stack, 97.5, axis=0)
        
        lower_ci[col] = np.round(lower, 5).tolist()
        upper_ci[col] = np.round(upper, 5).tolist()
        
    return lower_ci, upper_ci


def dynamic_prediction_from_landmark(
    long_df: pd.DataFrame,
    landmark_time: float,
    current_state: int,
    predictors: List[str],
    id_col: str = "id",
    from_state_col: str = "from_state",
    to_state_col: str = "to_state",
    entry_col: str = "entry",
    exit_col: str = "exit",
    event_col: str = "event",
    horizon_times: Optional[np.ndarray] = None,
    transition_model_type: str = "cox",
    run_bootstrap: bool = False,
    n_bootstrap: int = 50,
    run_microsimulation: bool = False,
    n_simulations: int = 1000,
) -> Dict[str, Any]:
    """
    Perform dynamic prediction: given that a subject is in `current_state` at `landmark_time`,
    compute future state occupation probabilities forward in time using Cox or Weibull transition models.
    """
    if horizon_times is None:
        horizon_times = np.linspace(landmark_time, landmark_time + 5, 20)
    else:
        horizon_times = np.sort(horizon_times)

    # Filter to subjects still at risk at landmark in the current state
    at_risk = long_df[
        (long_df[entry_col] <= landmark_time) &
        (long_df[exit_col] > landmark_time) &
        (long_df[from_state_col] == current_state)
    ].copy()

    if len(at_risk) < 10:
        return {"error": f"Too few subjects ({len(at_risk)}) at risk at landmark for reliable dynamic prediction"}

    # Fit transitions on the overall long-format data for maximum statistical stability
    try:
        models, trans_res = _fit_multistate_models(
            long_df=long_df,
            id_col=id_col,
            from_state_col=from_state_col,
            to_state_col=to_state_col,
            entry_col=entry_col,
            exit_col=exit_col,
            event_col=event_col,
            predictors=predictors,
            model_type=transition_model_type,
        )
    except Exception as e:
        return {"error": f"Failed to fit transition models: {str(e)}"}

    if not models:
        return {"error": "No valid transitions could be estimated."}

    # Dynamic prediction: compute forward probability curve assuming average covariates of at-risk subjects
    avg_cov = at_risk[predictors].mean().to_frame().T
    
    # Predict actual cumulative hazards on the horizon grid
    cumhaz = {}
    states_set = set(f for f, t in models.keys()) | set(t for f, t in models.keys())
    states = sorted(list(states_set))
    
    for trans, fitter in models.items():
        try:
            if transition_model_type == "cox":
                pred = fitter.predict_cumulative_hazard(avg_cov, times=horizon_times)
            else:
                durations = horizon_times - horizon_times[0]
                pred = fitter.predict_cumulative_hazard(avg_cov, times=durations)
            
            ch = pred.iloc[:, 0].values
            ch = np.maximum.accumulate(ch)
            cumhaz[trans] = ch
        except Exception:
            cumhaz[trans] = np.zeros_like(horizon_times)

    # Compute analytical state probabilities using Aalen-Johansen
    probs = _compute_state_occupation_probabilities_internal(
        cumhaz,
        horizon_times,
        states,
        current_state
    )

    # Compute ELOS (Expected Length of Stay)
    elos = compute_elos(probs, horizon_times)

    # Run Bootstrap CI if requested
    bootstrap_results = {}
    if run_bootstrap:
        try:
            lower_ci, upper_ci = compute_multistate_bootstrap_ci(
                long_df=long_df,
                id_col=id_col,
                from_state_col=from_state_col,
                to_state_col=to_state_col,
                entry_col=entry_col,
                exit_col=exit_col,
                event_col=event_col,
                predictors=predictors,
                covariate_df=avg_cov,
                times=horizon_times,
                initial_state=current_state,
                model_type=transition_model_type,
                n_bootstrap=n_bootstrap,
            )
            bootstrap_results = {
                "lower": lower_ci,
                "upper": upper_ci
            }
        except Exception as e:
            bootstrap_results = {"error": f"Bootstrap failed: {str(e)}"}

    # Run Microsimulation if requested
    microsim_results = {}
    if run_microsimulation:
        try:
            microsim_results = run_multistate_microsimulation(
                models=models,
                covariate_df=avg_cov,
                initial_state=current_state,
                times=horizon_times,
                model_type=transition_model_type,
                n_simulations=n_simulations,
            )
        except Exception as e:
            microsim_results = {"error": f"Microsimulation failed: {str(e)}"}

    # Compute prediction error metric
    obs_records = []
    for _, subj in at_risk.iterrows():
        subj_id = subj[id_col]
        subj_all_records = long_df[long_df[id_col] == subj_id].sort_values(by=entry_col)
        
        for ht in horizon_times:
            state_at_ht = None
            for _, rec in subj_all_records.iterrows():
                if rec[entry_col] <= ht < rec[exit_col]:
                    state_at_ht = int(rec[from_state_col])
                    break
                elif ht >= rec[exit_col] and rec[event_col] == 1:
                    state_at_ht = int(rec[to_state_col])
            
            if state_at_ht is None:
                last_rec = subj_all_records.iloc[-1]
                state_at_ht = int(last_rec[to_state_col]) if last_rec[event_col] == 1 else int(last_rec[from_state_col])
                
            obs_records.append({
                "time": ht,
                "observed_state": state_at_ht
            })

    obs_df = pd.DataFrame(obs_records) if obs_records else pd.DataFrame(columns=["time", "observed_state"])

    error_metrics = {}
    if not obs_df.empty:
        error_metrics = compute_multistate_prediction_error(
            probs, obs_df, time_col="time", state_col="observed_state"
        )

    return {
        "landmark_time": landmark_time,
        "current_state": current_state,
        "n_at_risk": len(at_risk),
        # `probs` is indexed by time and to_dict(orient="list") drops the index,
        # which left the curves with no x-axis for any consumer to plot against.
        "times": [float(t) for t in probs.index],
        "state_probabilities": probs.to_dict(orient="list"),
        "elos": elos,
        "bootstrap": bootstrap_results,
        "microsimulation": microsim_results,
        "prediction_error": error_metrics,
        "transition_models": trans_res,
        "model_type": transition_model_type,
        "note": f"Dynamic prediction from landmark using transition-specific {transition_model_type} models."
    }


def compute_multistate_prediction_error(
    predicted_probs: pd.DataFrame,
    observed_data: pd.DataFrame,
    time_col: str = "time",
    state_col: str = "observed_state",
) -> Dict[str, Any]:
    """
    Basic multi-state prediction error (generalized Brier score).
    """
    if predicted_probs.empty or observed_data.empty:
        return {"error": "Empty input data"}

    errors = []
    times = predicted_probs.index.values

    for t in times:
        at_risk = observed_data[observed_data[time_col] >= t]
        if len(at_risk) == 0:
            continue

        pred_row = predicted_probs.loc[t].values
        states = [int(c.split("_")[1]) for c in predicted_probs.columns]

        sq_errors = []
        for _, row in at_risk.iterrows():
            obs_state = int(row[state_col])
            indicator = np.array([1.0 if s == obs_state else 0.0 for s in states])
            sq = np.sum((pred_row - indicator) ** 2)
            sq_errors.append(sq)

        if sq_errors:
            mean_err = float(np.mean(sq_errors))
            errors.append({
                "time": round(float(t), 4),
                "n_at_risk": len(at_risk),
                "mean_squared_error": round(mean_err, 5),
            })

    if not errors:
        return {"error": "No overlapping prediction and observation times"}

    overall = float(np.mean([e["mean_squared_error"] for e in errors]))

    return {
        "per_time": errors,
        "overall_mean_error": round(overall, 5),
        "n_time_points": len(errors),
        "interpretation": "Multi-state Brier-style score. Lower values indicate better probabilistic calibration and discrimination across states.",
    }
