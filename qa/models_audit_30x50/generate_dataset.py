"""Deterministic 30-row x 50-column dataset for cross-validating the
regression and survival endpoints against R at a deliberately small n.

Thirty rows is not an accident. It is where the large-sample shortcuts break:
t against z, exact against asymptotic, a rare category with two observations,
a design matrix one column away from singular, a Cox fit with eighteen events.
Anything that only shows up at n = 300 is not being tested here; anything that
only shows up at n = 30 is.

Missing values are generated with three different mechanisms, and which column
carries which is fixed and documented, so a shifted result can be traced back
to the mechanism that shifted it:

  clean   no missing at all - the control
  MCAR    missing completely at random
  MAR     missing conditional on an observed column (age, or diabetes)

Outcomes are always generated from the TRUE covariate values before any
masking, so the missingness is a property of what the analyst sees rather than
of what generated the data.

No patient data. Fixed seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20260730
N = 30
N_EXT = 40  # external validation cohort

# column -> (mechanism, rate). Everything not listed is clean.
MISSINGNESS = {
    "bmi": ("MCAR", 0.10),
    "stage": ("MCAR", 0.10),
    "smoke": ("MCAR", 0.10),
    "region": ("MCAR", 0.25),
    "prior_tx": ("MCAR", 0.10),
    "dbp_base": ("MCAR", 0.10),
    "ldl": ("MCAR", 0.10),
    "hdl": ("MCAR", 0.25),
    "hb": ("MCAR", 0.10),
    "alb": ("MCAR", 0.25),
    "potassium": ("MCAR", 0.40),   # deliberately severe
    "tumor_size": ("MCAR", 0.10),
    "sbp": ("MCAR", 0.10),         # a missing OUTCOME
    "qol": ("MCAR", 0.25),
    "grade": ("MCAR", 0.10),
    "grade_num": ("MCAR", 0.10),   # masked on the same rows as `grade`
    "crp": ("MAR_age", 0.20),
    "biomarker": ("MAR_age", 0.20),
    "egfr": ("MAR_age", 0.25),
    "glucose": ("MAR_diabetes", 0.30),
}


def _mask_mcar(rng: np.random.Generator, n: int, rate: float) -> np.ndarray:
    """Exactly round(n * rate) rows, so the count is reproducible."""
    k = int(round(n * rate))
    idx = rng.choice(n, size=k, replace=False)
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m


def _mask_mar(rng: np.random.Generator, driver: np.ndarray, rate: float) -> np.ndarray:
    """Missing with probability increasing in `driver`, at the target rate on
    average. The driver is standardised so the tilt does not depend on scale."""
    z = (driver - driver.mean()) / (driver.std() or 1.0)
    p = 1.0 / (1.0 + np.exp(-(np.log(rate / (1 - rate)) + 1.2 * z)))
    return rng.random(len(driver)) < p


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    pid = np.arange(1, N + 1)
    site = rng.choice(["s1", "s2", "s3", "s4", "s5"], N)
    wt = np.round(rng.uniform(0.6, 2.4, N), 3)

    age = np.round(rng.normal(61, 12, N), 1)
    sex = rng.choice(["F", "M"], N)
    arm = np.array(["control", "treat"] * (N // 2))          # exactly balanced
    rng.shuffle(arm)
    bmi = np.round(rng.normal(27.4, 4.5, N), 2)
    stage = rng.choice(["I", "II", "III"], N, p=[0.45, 0.35, 0.20])
    smoke = rng.choice(["never", "former", "current"], N, p=[0.5, 0.3, 0.2])
    region = rng.choice(["r1", "r2", "r3", "r4", "r5", "r6"], N)

    # A category whose third level holds a single observation. At n = 30 this
    # is what a real subgroup looks like, and it is where dummy coding, exact
    # tests and separation handling either work or do not.
    rare_grp = np.array(["a"] * 20 + ["b"] * 9 + ["c"] * 1)
    rng.shuffle(rare_grp)
    solo_grp = np.array(["only"] * N)                        # zero variance

    diabetes = rng.binomial(1, 0.30, N)
    prior_tx = rng.binomial(1, 0.40, N)
    comorbid = rng.binomial(1, 0.55, N)

    sbp_base = np.round(rng.normal(134, 15, N), 1)
    dbp_base = np.round(rng.normal(80, 9, N), 1)
    ldl = np.round(rng.normal(3.2, 0.8, N), 2)
    hdl = np.round(rng.normal(1.3, 0.3, N), 2)
    crp = np.round(np.exp(rng.normal(0.6, 0.9, N)), 3)       # right-skewed
    biomarker = np.round(np.exp(rng.normal(1.1, 0.7, N)), 3)
    egfr = np.round(np.clip(105 - 0.45 * age + rng.normal(0, 9, N), 15, None), 1)
    hb = np.round(rng.normal(13.1, 1.5, N), 1)
    alb = np.round(rng.normal(39, 4.5, N), 1)
    sodium = np.round(rng.normal(139, 3.0, N), 1)
    potassium = np.round(rng.normal(4.2, 0.45, N), 2)
    glucose = np.round(rng.normal(5.6, 1.1, N) + 2.1 * diabetes, 2)
    tumor_size = np.round(np.clip(rng.gamma(3.0, 1.2, N), 0.3, None), 2)
    dose = np.round(rng.choice([10.0, 20.0, 40.0], N), 1)
    duration_days = rng.integers(30, 400, N)

    score1 = np.round(rng.normal(50, 10, N), 2)
    # Near-collinear by construction: VIF on (score1, score2) should be large
    # but finite, which is exactly the case a VIF implementation gets wrong.
    score2 = np.round(2.0 * score1 + rng.normal(0, 0.8, N), 2)
    const_num = np.full(N, 5.0)

    followup_years = np.round(np.clip(rng.gamma(4.0, 0.55, N), 0.3, None), 3)

    # ── outcomes, generated from the TRUE covariates ──────────────────────
    sbp = np.round(
        101.0 + 0.40 * age + 0.62 * bmi + 5.5 * (arm == "treat")
        + rng.normal(0, 8.0, N), 2)

    qol = np.round(
        72.0 - 0.22 * age + 4.0 * (arm == "treat") - 3.0 * comorbid
        + rng.normal(0, 6.0, N), 2)

    cost = np.round(rng.gamma(shape=2.5, scale=np.exp(1.1 + 0.021 * age) / 2.5), 2)

    log_mu = -1.2 + 0.018 * age + 0.42 * (arm == "treat") + np.log(followup_years)
    admissions = rng.poisson(np.exp(log_mu))

    nb_mu = np.exp(-0.5 + 0.028 * age)
    visits = rng.negative_binomial(2.0, 2.0 / (2.0 + nb_mu))

    lin = 0.045 * age + 0.9 * (arm == "treat")
    cuts = np.quantile(lin, [0.40, 0.75])
    grade_num = np.digitize(lin + rng.logistic(0, 0.8, N), cuts) + 1
    grade = np.array([{1: "mild", 2: "moderate", 3: "severe"}[g] for g in grade_num])

    z = -5.4 + 0.058 * age + 1.0 * (arm == "treat") + 0.08 * bmi
    event_binary = rng.binomial(1, 1 / (1 + np.exp(-z)))

    z2 = -6.0 + 0.11 * score1 + 0.7 * (sex == "M")
    resp_binary = rng.binomial(1, 1 / (1 + np.exp(-z2)))

    # Perfectly separated by `prior_tx`: every treated subject is a 1 and no
    # untreated one is. Ordinary logistic diverges here; this is the case
    # Firth exists for, and it must be present in a small-n audit.
    sep_binary = prior_tx.copy()

    # ── time to event, with administrative censoring ──────────────────────
    lp = 0.030 * (age - 61) + 0.60 * (arm == "treat") - 0.30 * (sex == "M")
    t_event = rng.exponential(1 / (0.055 * np.exp(lp)))
    t_cens = rng.uniform(3, 26, N)
    time = np.round(np.minimum(t_event, t_cens), 3)
    status = (t_event <= t_cens).astype(int)

    # Competing risks: among the events, a fraction are cause 2 rather than 1.
    cause = rng.binomial(1, 0.30, N)          # 1 -> competing cause
    cmp_status = np.where(status == 0, 0, np.where(cause == 1, 2, 1))

    # Interval censoring: the same event time, observed only between two
    # scheduled visits. Right-censored subjects get an open upper bound.
    grid = np.arange(0, 40, 3.0)
    ic_l, ic_r = [], []
    for t, s in zip(time, status):
        if s == 1:
            lo = grid[grid <= t].max()
            hi = grid[grid > t].min()
            ic_l.append(round(float(lo), 3))
            ic_r.append(round(float(hi), 3))
        else:
            ic_l.append(round(float(grid[grid <= t].max()), 3))
            ic_r.append(np.nan)               # open to the right
    ic_l = np.array(ic_l, dtype=float)
    ic_r = np.array(ic_r, dtype=float)

    # A pre-existing risk score, for calibration / external validation.
    pred_risk = np.round(1 / (1 + np.exp(-(-5.0 + 0.055 * age + 0.9 * (arm == "treat")))), 4)

    df = pd.DataFrame({
        "pid": pid, "site": site, "wt": wt,
        "age": age, "sex": sex, "arm": arm, "bmi": bmi,
        "stage": stage, "smoke": smoke, "region": region,
        "rare_grp": rare_grp, "solo_grp": solo_grp,
        "diabetes": diabetes, "prior_tx": prior_tx, "comorbid": comorbid,
        "sbp_base": sbp_base, "dbp_base": dbp_base, "ldl": ldl, "hdl": hdl,
        "crp": crp, "biomarker": biomarker, "egfr": egfr, "hb": hb, "alb": alb,
        "sodium": sodium, "potassium": potassium, "glucose": glucose,
        "tumor_size": tumor_size, "dose": dose, "duration_days": duration_days,
        "score1": score1, "score2": score2, "const_num": const_num,
        "followup_years": followup_years,
        "sbp": sbp, "qol": qol, "cost": cost,
        "admissions": admissions, "visits": visits,
        "grade": grade, "grade_num": grade_num,
        "event_binary": event_binary, "resp_binary": resp_binary,
        "sep_binary": sep_binary,
        "time": time, "status": status, "cmp_status": cmp_status,
        "ic_l": ic_l, "ic_r": ic_r, "pred_risk": pred_risk,
    })

    df = _apply_missingness(df, rng)
    assert df.shape == (N, 50), df.shape
    return df


def _apply_missingness(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Mask after the outcomes were generated, never before."""
    out = df.copy()
    age = df["age"].to_numpy()
    diabetes = df["diabetes"].to_numpy().astype(float)
    grade_mask = None
    for col, (mech, rate) in MISSINGNESS.items():
        if mech == "MCAR":
            m = _mask_mcar(rng, len(df), rate)
        elif mech == "MAR_age":
            m = _mask_mar(rng, age, rate)
        elif mech == "MAR_diabetes":
            m = _mask_mar(rng, diabetes, rate)
        else:  # pragma: no cover - the table above is exhaustive
            raise ValueError(mech)
        if col == "grade":
            grade_mask = m
        if col == "grade_num" and grade_mask is not None:
            m = grade_mask   # the ordinal outcome and its numeric twin agree
        out.loc[m, col] = np.nan
    return out


def build_long(df: pd.DataFrame) -> pd.DataFrame:
    """Three visits per subject: mixed model, GEE, repeated measures."""
    rng = np.random.default_rng(SEED + 1)
    intercepts = rng.normal(0, 5.5, len(df))
    rows = []
    for i, r in df.reset_index(drop=True).iterrows():
        for k, visit in enumerate(("v1", "v2", "v3")):
            eta = (55 + intercepts[i] + 3.0 * k
                   + 2.8 * (r["arm"] == "treat") + 0.07 * (r["age"] - 61))
            rows.append({
                "pid": int(r["pid"]),
                "visit": visit,
                "arm": r["arm"],
                "sex": r["sex"],
                "site": r["site"],
                "age": float(r["age"]),
                "score": round(float(eta + rng.normal(0, 3.0)), 3),
                "count_y": int(rng.poisson(np.exp(0.4 + 0.25 * k))),
                "resp": int(rng.binomial(1, 1 / (1 + np.exp(-(-1.0 + 0.5 * k))))),
            })
    return pd.DataFrame(rows)


def build_recurrent(df: pd.DataFrame) -> pd.DataFrame:
    """Counting-process intervals for the recurrent-event model."""
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for _, r in df.iterrows():
        t, horizon = 0.0, float(r["time"])
        rate = 0.09 * np.exp(0.5 * (r["arm"] == "treat"))
        while True:
            gap = float(rng.exponential(1 / rate))
            if t + gap >= horizon:
                if horizon > t:
                    rows.append({"pid": int(r["pid"]), "start": round(t, 3),
                                 "stop": round(horizon, 3), "event": 0,
                                 "arm": r["arm"], "age": float(r["age"]),
                                 "sex": r["sex"]})
                break
            t += gap
            rows.append({"pid": int(r["pid"]), "start": round(t - gap, 3),
                         "stop": round(t, 3), "event": 1,
                         "arm": r["arm"], "age": float(r["age"]), "sex": r["sex"]})
    return pd.DataFrame(rows)


def build_multistate(df: pd.DataFrame) -> pd.DataFrame:
    """Illness-death transitions, one row per subject per transition at risk.

    State 1 alive and well, 2 relapsed, 3 dead. `cmp_status` 1 is a relapse
    and 2 a death without relapse, so the same subjects drive the competing
    risk analysis and the multistate one.
    """
    rng = np.random.default_rng(SEED + 4)
    rows = []
    for _, r in df.iterrows():
        t, cs = float(r["time"]), int(r["cmp_status"])
        base = {"id": int(r["pid"]), "age": float(r["age"]),
                "arm": r["arm"], "sex": r["sex"]}
        # at risk for 1->2 and 1->3 from entry to t
        rows.append({**base, "from_state": 1, "to_state": 2, "entry": 0.0,
                     "exit": t, "event": int(cs == 1)})
        rows.append({**base, "from_state": 1, "to_state": 3, "entry": 0.0,
                     "exit": t, "event": int(cs == 2)})
        if cs == 1:  # relapsed, then at risk for 2->3
            extra = float(rng.exponential(6.0))
            rows.append({**base, "from_state": 2, "to_state": 3, "entry": t,
                         "exit": round(t + extra, 3),
                         "event": int(rng.random() < 0.45)})
    return pd.DataFrame(rows)


def build_raters(df: pd.DataFrame) -> pd.DataFrame:
    """Three raters scoring the same 30 subjects, continuous and categorical.

    Rater B carries a systematic +2.1 offset and rater C a −1.4 one. That is
    the whole point: a consistency ICC ignores rater bias and an absolute
    agreement ICC does not, so the two forms have to come out different here.
    If they agree on this frame, the endpoint is not computing what it says.
    """
    rng = np.random.default_rng(SEED + 5)
    true_score = rng.normal(50, 9, N)
    a = np.round(true_score + rng.normal(0, 3.0, N), 2)
    b = np.round(true_score + 2.1 + rng.normal(0, 3.0, N), 2)
    c = np.round(true_score - 1.4 + rng.normal(0, 4.5, N), 2)

    # Ordered categories, with the raters agreeing most of the time.
    base = np.digitize(true_score, np.quantile(true_score, [0.35, 0.7]))
    def _rate(flip: float, seed_shift: int) -> np.ndarray:
        r = np.random.default_rng(SEED + seed_shift)
        out = base.copy()
        move = r.random(N) < flip
        out[move] = np.clip(out[move] + r.choice([-1, 1], move.sum()), 0, 2)
        return np.array(["low", "mid", "high"], dtype=object)[out]

    out = pd.DataFrame({
        "pid": df["pid"].to_numpy(),
        "rater_a": a, "rater_b": b, "rater_c": c,
        "cat_a": _rate(0.00, 6), "cat_b": _rate(0.20, 7), "cat_c": _rate(0.35, 8),
    })
    # Missing on one rater only, so complete-case n differs per pair.
    out.loc[_mask_mcar(rng, N, 0.10), "rater_c"] = np.nan
    out.loc[_mask_mcar(rng, N, 0.10), "cat_c"] = np.nan
    return out


def build_external() -> pd.DataFrame:
    """A second cohort for external validation: same schema, different draw."""
    rng = np.random.default_rng(SEED + 3)
    age = np.round(rng.normal(64, 11, N_EXT), 1)
    sex = rng.choice(["F", "M"], N_EXT)
    arm = rng.choice(["control", "treat"], N_EXT)
    bmi = np.round(rng.normal(28.0, 4.2, N_EXT), 2)
    stage = rng.choice(["I", "II", "III"], N_EXT, p=[0.4, 0.35, 0.25])
    z = -5.4 + 0.058 * age + 1.0 * (arm == "treat") + 0.08 * bmi
    event_binary = rng.binomial(1, 1 / (1 + np.exp(-z)))
    lp = 0.030 * (age - 61) + 0.60 * (arm == "treat") - 0.30 * (sex == "M")
    t_event = rng.exponential(1 / (0.055 * np.exp(lp)))
    t_cens = rng.uniform(3, 26, N_EXT)
    return pd.DataFrame({
        "pid": np.arange(1001, 1001 + N_EXT),
        "age": age, "sex": sex, "arm": arm, "bmi": bmi, "stage": stage,
        "event_binary": event_binary,
        "time": np.round(np.minimum(t_event, t_cens), 3),
        "status": (t_event <= t_cens).astype(int),
        # The development model's linear predictor, carried over unchanged.
        # External validation exists to score a frozen model on new data, so
        # this column is an input, not something refitted here.
        "pred_lp": np.round(lp - lp.mean(), 5),
    })


if __name__ == "__main__":
    import hashlib
    import pathlib

    here = pathlib.Path(__file__).parent
    df = build()
    lf = build_long(df)
    rf = build_recurrent(df)
    ms = build_multistate(df)
    ef = build_external()
    ra = build_raters(df)
    files = {
        "dataset.csv": df, "dataset_long.csv": lf,
        "dataset_recurrent.csv": rf, "dataset_multistate.csv": ms,
        "dataset_external.csv": ef, "dataset_raters.csv": ra,
    }
    for name, frame in files.items():
        frame.to_csv(here / name, index=False)
        h = hashlib.sha256((here / name).read_bytes()).hexdigest()[:16]
        print(f"{name:24s} {str(frame.shape):12s} sha256:{h}")

    print("\nevents:", int(df.status.sum()),
          "| competing:", int((df.cmp_status == 2).sum()),
          "| binary:", int(df.event_binary.sum()),
          "| separated:", int(df.sep_binary.sum()))
    miss = df.isna().sum()
    print("columns with missing:", int((miss > 0).sum()), "of", df.shape[1])
    print("complete rows:", int(df.notna().all(axis=1).sum()))
    print("rare_grp:", df.rare_grp.value_counts().to_dict())
