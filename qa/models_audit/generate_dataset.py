"""Deterministic sample dataset for cross-validating the Models endpoints
against R. No randomness beyond a fixed seed; no patient data.

Carries one outcome of every kind the Models panel fits:
  continuous, binary, count, positive-skewed, ordinal, time-to-event,
  plus a repeated-measures long form for the mixed model.
"""
import numpy as np
import pandas as pd

SEED = 20260729
N = 300


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    age = rng.normal(62, 11, N).round(1)
    sex = rng.choice(["F", "M"], N)
    bmi = rng.normal(27.5, 4.2, N).round(2)
    arm = rng.choice(["control", "treat"], N)
    stage = rng.choice(["I", "II", "III"], N, p=[0.45, 0.35, 0.20])
    biomarker = np.exp(rng.normal(1.1, 0.65, N)).round(3)

    z = (
        -6.2
        + 0.055 * age
        + 0.9 * (arm == "treat")
        + 0.11 * bmi
        + 0.6 * (sex == "M")
    )
    event_binary = rng.binomial(1, 1 / (1 + np.exp(-z)))

    # Continuous outcome with a genuine linear structure.
    sbp = (
        95.0
        + 0.42 * age
        + 0.75 * bmi
        + 6.5 * (arm == "treat")
        + rng.normal(0, 9.0, N)
    ).round(2)

    # Count outcome (Poisson-ish) plus an offset for exposure time.
    log_mu = -1.4 + 0.02 * age + 0.45 * (arm == "treat")
    admissions = rng.poisson(np.exp(log_mu))
    followup_years = np.clip(rng.gamma(4.0, 0.6, N), 0.25, None).round(3)

    # Overdispersed count for the negative binomial.
    nb_mu = np.exp(-0.6 + 0.03 * age)
    visits = rng.negative_binomial(2.0, 2.0 / (2.0 + nb_mu))

    # Strictly positive skewed outcome for the Gamma model.
    cost = np.round(rng.gamma(shape=2.5, scale=np.exp(1.0 + 0.02 * age) / 2.5), 2)

    # Ordinal outcome, three ordered grades.
    lin = 0.04 * age + 0.8 * (arm == "treat")
    cuts = np.quantile(lin, [0.4, 0.75])
    noise = rng.logistic(0, 0.9, N)
    grade_num = np.digitize(lin + noise, cuts) + 1
    grade = pd.Categorical(
        [{1: "mild", 2: "moderate", 3: "severe"}[g] for g in grade_num],
        categories=["mild", "moderate", "severe"], ordered=True,
    )

    # Time to event with administrative censoring.
    lp = 0.035 * (age - 62) + 0.55 * (arm == "treat") - 0.25 * (sex == "M")
    t_event = rng.exponential(1 / (0.05 * np.exp(lp)))
    t_cens = rng.uniform(2, 30, N)
    time = np.minimum(t_event, t_cens).round(3)
    status = (t_event <= t_cens).astype(int)

    df = pd.DataFrame({
        "pid": np.arange(1, N + 1),
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "arm": arm,
        "stage": stage,
        "biomarker": biomarker,
        "sbp": sbp,
        "event_binary": event_binary,
        "admissions": admissions,
        "followup_years": followup_years,
        "visits": visits,
        "cost": cost,
        "grade": grade.astype(str),
        "grade_num": grade_num,
        "time": time,
        "status": status,
    })
    return df


def build_long(df: pd.DataFrame) -> pd.DataFrame:
    """Repeated measures for the mixed model: 3 visits per subject."""
    rng = np.random.default_rng(SEED + 1)
    rows = []
    intercepts = rng.normal(0, 6.0, len(df))
    for i, r in df.iterrows():
        for k, visit in enumerate(("v1", "v2", "v3")):
            rows.append({
                "pid": int(r["pid"]),
                "visit": visit,
                "arm": r["arm"],
                "age": r["age"],
                "score": round(
                    50 + intercepts[i] + 3.2 * k
                    + 2.5 * (r["arm"] == "treat")
                    + 0.08 * (r["age"] - 62)
                    + rng.normal(0, 3.0), 3),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import pathlib, hashlib
    here = pathlib.Path(__file__).parent
    df = build()
    lf = build_long(df)
    df.to_csv(here / "dataset.csv", index=False)
    lf.to_csv(here / "dataset_long.csv", index=False)
    for f in ("dataset.csv", "dataset_long.csv"):
        h = hashlib.sha256((here / f).read_bytes()).hexdigest()
        print(f"{f}: {h}")
    print(df.shape, lf.shape)
    print("events:", int(df.status.sum()), "binary:", int(df.event_binary.sum()))
    print(df.grade.value_counts().to_dict())
