"""Generate the 100 x 40 walkthrough dataset used by TEST_PLANI.md.

One synthetic cohort that carries a suitable variable for every tab of the
app: descriptive, tests, correlation, ROC, regression, RCS, survival,
PSM / IPTW, causal, DCA, missing data, reliability and factor analysis.

Run from the repo root:
    backend/.venv/bin/python qa/full_walkthrough_100x40/generate_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261112
N = 100
ADMIN_CENSOR_MONTHS = 36.0
OUT = Path(__file__).with_name("dataset.csv")


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _baseline(rng: np.random.Generator) -> pd.DataFrame:
    age = rng.normal(62, 10, N).clip(35, 88).round(1)
    sex = rng.choice(["F", "M"], N, p=[0.45, 0.55])
    bmi = rng.normal(27.5, 4.2, N).clip(17, 42).round(1)
    stage = rng.choice(["I", "II", "III"], N, p=[0.4, 0.35, 0.25])
    smoke = rng.choice(["never", "former", "current"], N, p=[0.45, 0.33, 0.22])
    diabetes = rng.binomial(1, _expit(-1.6 + 0.09 * (bmi - 27.5) + 0.02 * (age - 62)))
    htn = rng.binomial(1, _expit(-0.4 + 0.04 * (age - 62) + 0.05 * (bmi - 27.5)))
    return pd.DataFrame({
        "pid": np.arange(1, N + 1),
        "site": rng.choice(["s1", "s2", "s3", "s4", "s5"], N),
        "age": age, "sex": sex, "bmi": bmi, "stage": stage, "smoke": smoke,
        "diabetes": diabetes, "htn": htn,
    })


def _treatment(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Confounded assignment (PSM / IPTW) driven partly by an instrument (IV)."""
    instrument = rng.normal(0, 1, N).round(3)
    stage_num = df["stage"].map({"I": 0, "II": 1, "III": 2}).to_numpy()
    p_treat = _expit(-0.2 + 0.9 * instrument - 0.03 * (df["age"] - 62) + 0.5 * df["diabetes"] + 0.3 * stage_num)
    treat = rng.binomial(1, p_treat)
    dose = np.where(treat == 1, rng.choice([10, 20, 40], N, p=[0.3, 0.4, 0.3]), 0)
    mediator = (55 + 18 * treat - 0.2 * (df["age"] - 62) + rng.normal(0, 10, N)).clip(5, 100).round(1)
    return df.assign(
        arm=np.where(treat == 1, "treat", "control"),
        dose=dose, instrument=instrument, mediator=mediator,
    )


def _labs(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    treat = (df["arm"] == "treat").to_numpy().astype(float)
    male = (df["sex"] == "M").to_numpy().astype(float)
    sbp_base = (128 + 0.5 * (df["age"] - 62) + 8 * df["htn"] + rng.normal(0, 11, N)).round(1)
    sbp_m3 = (sbp_base - 3 - 5 * treat + rng.normal(0, 6, N)).round(1)
    sbp_m6 = (sbp_base - 4 - 9 * treat + rng.normal(0, 7, N)).round(1)
    ldl = (3.4 + 0.03 * (df["bmi"] - 27.5) + rng.normal(0, 0.7, N)).clip(1.2, 6.5).round(2)
    hdl = (1.35 - 0.2 * male - 0.02 * (df["bmi"] - 27.5) + rng.normal(0, 0.25, N)).clip(0.5, 2.6).round(2)
    crp = np.exp(0.6 + 0.05 * (df["bmi"] - 27.5) + rng.normal(0, 0.9, N)).round(2)
    egfr = (92 - 0.9 * (df["age"] - 62) - 6 * df["diabetes"] + rng.normal(0, 12, N)).clip(18, 125).round(1)
    hb = (13.2 + 1.2 * male + rng.normal(0, 1.1, N)).round(1)
    return df.assign(sbp_base=sbp_base, sbp_m3=sbp_m3, sbp_m6=sbp_m6, ldl=ldl, hdl=hdl, crp=crp, egfr=egfr, hb=hb)


def _outcomes(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    treat = (df["arm"] == "treat").to_numpy().astype(float)
    stage_num = df["stage"].map({"I": 0, "II": 1, "III": 2}).to_numpy()
    # Shared risk score; the U-shaped BMI term gives the RCS panels a curve to find.
    risk = (
        0.035 * (df["age"] - 62) + 0.55 * stage_num + 0.5 * df["diabetes"]
        + 0.012 * (df["bmi"] - 27.5) ** 2 - 0.6 * treat + 0.15 * np.log(df["crp"])
    ).to_numpy()
    biomarker = (10 + 2.6 * risk + rng.normal(0, 2.4, N)).round(2)
    biomarker2 = (5 + 1.0 * risk + rng.normal(0, 2.6, N)).round(2)
    event_binary = rng.binomial(1, _expit(-1.3 + 1.1 * risk))
    qol = (72 - 6 * risk + 0.25 * (df["mediator"] - 60) + rng.normal(0, 8, N)).clip(10, 100).round(1)
    cost = rng.gamma(2.0, np.exp(7.0 + 0.35 * risk) / 2.0).round(0)
    followup_years = rng.uniform(0.5, 3.0, N).round(2)
    nb_mean = followup_years * np.exp(-0.5 + 0.45 * risk)
    admissions = rng.poisson(rng.gamma(1.5, nb_mean / 1.5))
    severity = 0.9 * risk + rng.logistic(0, 1, N)
    grade = pd.cut(severity, [-np.inf, 0.2, 1.6, np.inf], labels=["mild", "moderate", "severe"]).astype(str)
    return df.assign(
        biomarker=biomarker, biomarker2=biomarker2, event_binary=event_binary, qol=qol,
        cost=cost, admissions=admissions, followup_years=followup_years, grade=grade,
        _risk=risk,
    )


def _survival(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    risk = df["_risk"].to_numpy()
    t_event = rng.weibull(1.3, N) * 30 * np.exp(-0.55 * risk)
    t_compete = rng.exponential(110, N)
    t_censor = np.minimum(rng.uniform(8, 60, N), ADMIN_CENSOR_MONTHS)
    time = np.minimum.reduce([t_event, t_compete, t_censor])
    cmp_status = np.where(time == t_event, 1, np.where(time == t_compete, 2, 0))
    time = np.maximum(time, 0.1).round(2)
    # Interval censoring: events are only seen at 3-monthly visits.
    left = np.floor(time / 3) * 3
    ic_l = np.where(cmp_status == 1, left, time)
    ic_r = np.where(cmp_status == 1, left + 3, np.nan)
    return df.assign(
        time=time, status=(cmp_status == 1).astype(int), cmp_status=cmp_status,
        ic_l=ic_l.round(2), ic_r=ic_r,
    )


def _ratings_and_items(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    truth = df["event_binary"].to_numpy()
    rater1 = np.where(rng.random(N) < 0.85, truth, 1 - truth)
    rater2 = np.where(rng.random(N) < 0.80, truth, 1 - truth)
    latent = (df["qol"].to_numpy() - 65) / 12
    items = {
        f"item{i}": np.clip(np.rint(3 + 0.9 * latent + rng.normal(0, 0.8, N)), 1, 5).astype(int)
        for i in range(1, 5)
    }
    return df.assign(rater1=rater1, rater2=rater2, **items)


def _add_missing(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """MCAR holes plus one MAR pattern (qol missing more often in older patients)."""
    out = df.copy()
    for col, share in {"ldl": 0.08, "hdl": 0.06, "crp": 0.10, "smoke": 0.05, "sbp_m6": 0.07, "biomarker2": 0.05}.items():
        out.loc[rng.random(N) < share, col] = np.nan
    p_qol = _expit(-2.6 + 0.07 * (out["age"] - 62))
    out.loc[rng.random(N) < p_qol, "qol"] = np.nan
    return out


COLUMNS = [
    "pid", "site", "age", "sex", "arm", "bmi", "stage", "smoke", "diabetes", "htn",
    "sbp_base", "sbp_m3", "sbp_m6", "ldl", "hdl", "crp", "egfr", "hb",
    "biomarker", "biomarker2", "dose", "instrument", "mediator",
    "qol", "cost", "admissions", "followup_years", "grade", "event_binary",
    "rater1", "rater2", "time", "status", "cmp_status", "ic_l", "ic_r",
    "item1", "item2", "item3", "item4",
]


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    df = _baseline(rng)
    df = _treatment(df, rng)
    df = _labs(df, rng)
    df = _outcomes(df, rng)
    df = _survival(df, rng)
    df = _ratings_and_items(df, rng)
    df = _add_missing(df, rng)
    return df[COLUMNS]


if __name__ == "__main__":
    data = build()
    if data.shape != (N, len(COLUMNS)) or len(COLUMNS) != 40:
        raise SystemExit(f"unexpected shape {data.shape}")
    data.to_csv(OUT, index=False)
    print(f"wrote {OUT} {data.shape}")
