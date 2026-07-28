"""200x20 dogrulama veri seti uretici.

Survival ciftini de sayan 10 surekli + 10 kategorik degisken, bilincli gercek
iliskiler (ground truth) ve karisik MCAR/MAR eksiklik desenleri icerir. Her
testin beklenen sonucu ground_truth.json'a yazilir.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 200
BASE = Path(__file__).resolve().parent

# --- Kategorik degiskenler ---
sex = rng.choice(["Male", "Female"], N, p=[0.48, 0.52])
group = rng.choice(["A", "B"], N, p=[0.5, 0.5])                # tedavi grubu (ikili)
# RNG draw retained to keep every downstream synthetic value identical to the
# original 200x21 audit fixture. The redundant treatment column is intentionally
# not emitted: group already supplies the independent two-arm exposure.
_treatment = rng.choice(["Drug", "Placebo"], N, p=[0.5, 0.5])
edu = rng.choice(["Low", "Mid", "High"], N, p=[0.3, 0.45, 0.25])  # 3 duzeyli
smoking = rng.choice(["Never", "Former", "Current"], N, p=[0.45, 0.3, 0.25])
diabetes = rng.choice([0, 1], N, p=[0.82, 0.18]).astype(float)
hypertension = rng.choice([0, 1], N, p=[0.7, 0.3]).astype(float)
stage = rng.choice(["I", "II", "III", "IV"], N, p=[0.35, 0.3, 0.2, 0.15])
center = rng.choice(["C1", "C2", "C3", "C4", "C5"], N, p=[0.25, 0.22, 0.2, 0.18, 0.15])
response = rng.choice([0, 1], N, p=[0.55, 0.45]).astype(float)

sex_m = (sex == "Male").astype(float)
grp_b = (group == "B").astype(float)
smoke_now = (smoking == "Current").astype(float)

# --- Surekli degiskenler (bilincli gercek iliskiler) ---
age = np.clip(rng.normal(55, 12, N), 18, 90)
bmi = np.clip(24 + 3.0 * sex_m + 0.08 * (age - 55) + rng.normal(0, 3.5, N), 15, 45)
sbp = np.clip(118 + 0.55 * (age - 55) + 1.8 * (bmi - 25) + rng.normal(0, 9, N), 85, 210)
cholesterol = np.clip(185 + 0.7 * (age - 55) + 2.2 * (bmi - 25) + rng.normal(0, 28, N), 110, 320)
glucose = np.clip(92 + 22 * diabetes + 0.6 * (bmi - 25) + rng.normal(0, 10, N), 60, 260)

# score: group B icin +5 gercek fark (t-test/ANOVA/Mann-Whitney anlamli cikmali)
score = 50 + 5.0 * grp_b + 2.5 * sex_m + rng.normal(0, 10, N)
# biomarker: gercek fark yok (p > 0.05 beklenir)
biomarker = rng.normal(10, 2, N)

# paired (tekrarli olcum): pre -> post ~3 puan dusus
pre = 70 + 0.3 * (age - 55) + rng.normal(0, 8, N)
post = pre - 3.0 + rng.normal(0, 6, N)

# survival (veri icinde hazir; eksiklik koymuyoruz)
base_haz = 0.05
hr_group = 1.8
linpred = np.log(hr_group) * grp_b + 0.03 * (age - 55) / 10
event_time = rng.exponential(1 / (base_haz * np.exp(linpred)))
censor = rng.exponential(1 / 0.02, N)
event = (event_time <= censor).astype(int)
time = np.round(np.minimum(event_time, censor) + 0.5, 1)

df = pd.DataFrame({
    "age": np.round(age, 1),
    "bmi": np.round(bmi, 1),
    "sbp": np.round(sbp, 1),
    "cholesterol": np.round(cholesterol, 1),
    "glucose": np.round(glucose, 1),
    "score": np.round(score, 2),
    "biomarker": np.round(biomarker, 3),
    "pre": np.round(pre, 2),
    "post": np.round(post, 2),
    "time": time,
    "event": event,
    "sex": sex,
    "group": group,
    "education": edu,
    "smoking": smoking,
    "diabetes": diabetes,
    "hypertension": hypertension,
    "stage": stage,
    "center": center,
    "response": response,
})

# --- Eksiklik desenleri ---
def mcar(col, frac):
    idx = rng.choice(df.index, int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

def mar(col, driver, frac):
    p = np.where(driver > np.median(driver), frac * 2, 0.0)
    mask = rng.random(N) < p
    df.loc[mask, col] = np.nan

mcar("bmi", 0.05)          # MCAR ~5%
mcar("score", 0.07)        # MCAR ~7%
mar("sbp", age, 0.06)      # MAR: yasi yukseklerde daha cok eksik
mar("cholesterol", bmi, 0.05)
mcar("post", 0.06)         # paired testte listwise kayip
mcar("biomarker", 0.04)
mcar("glucose", 0.03)
mcar("smoking", 0.05)
mcar("education", 0.04)
mcar("stage", 0.03)
mcar("response", 0.04)

df.to_csv(BASE / "dataset_200x20.csv", index=False)

truth = {
    "n_rows": int(N),
    "n_cols": int(df.shape[1]),
    "continuous": ["age", "bmi", "sbp", "cholesterol", "glucose", "score", "biomarker", "pre", "post", "time"],
    "categorical": ["event", "sex", "group", "education", "smoking", "diabetes", "hypertension", "stage", "center", "response"],
    "missing_frac": {c: round(float(df[c].isna().mean()), 3) for c in df.columns if df[c].isna().any()},
    "relations": {
        "score~group": {"true_diff_B_minus_A": 5.0, "expect": "significant"},
        "biomarker~group": {"true_diff": 0.0, "expect": "not significant"},
        "bmi~sex": {"true_diff_M_minus_F": 3.0, "expect": "significant"},
        "pre_post": {"true_diff_pre_minus_post": 3.0, "expect": "significant (paired)"},
        "glucose~diabetes": {"true_diff": 22.0, "expect": "significant"},
        "corr(age,sbp)": {"expect": "positive significant"},
        "corr(age,cholesterol)": {"expect": "positive significant"},
        "corr(bmi,glucose)": {"expect": "positive"},
        "cox group HR": {"true_HR_B_vs_A": 1.8, "expect": "HR>1, likely significant"},
        "chi sex x group": {"expect": "not significant (independent by design)"},
        "survival": {"note": "time/event eksiksiz birakildi", "event_rate": round(float(event.mean()), 3)},
    },
}
with (BASE / "ground_truth.json").open("w") as f:
    json.dump(truth, f, indent=2, ensure_ascii=False)

print(df.shape)
print(df.isna().sum()[df.isna().sum() > 0])
print("event rate:", event.mean().round(3))
