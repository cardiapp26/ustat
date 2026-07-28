#!/usr/bin/env python3
"""Generate a deterministic 200x20 clinical-biostatistics audit dataset.

Mirrors the 10x20 fixture's variable schema and missingness intent, but at
a size that lets large-sample approximations, repeated-measures, mixed ANOVA,
and two-way ANOVA all actually run (instead of being rejected by sample-size
guards).

No randomness without a fixed seed. No patient data. Source-controlled values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUT = "qa/statistical_audit_200x20/dataset.csv"
N = 200
SEED = 20260728
rng = np.random.default_rng(SEED)

# Deterministic group allocation, balanced across arms.
arm = np.array(["A", "B"])[np.arange(N) % 2]
# Three-level group, near-balanced.
group3 = np.array(["Low", "Medium", "High"])[np.arange(N) % 3]
# Two-level factor for two-way ANOVA. Must be balanced WITHIN each arm AND not
# a deterministic function of arm, otherwise the design matrix is rank-deficient
# and R drops the arm main effect (F = NA). Use a fixed-seed shuffle so X/Y
# appears in both arms independently.
perm = rng.permutation(N)
factor2_arr = np.array(["X", "Y"])[np.arange(N) % 2]
factor2 = factor2_arr[perm]

# Demographics: age centered ~55, mild arm-B skew.
age = 55 + 3 * (arm == "B") + rng.normal(0, 10, N)
age = np.clip(age, 18, 90).round(0)

# Continuous biomarker ~ Normal(10, 0.5): clear arm difference (B higher).
biomarker_normal = 10.0 + 1.2 * (arm == "B") + rng.normal(0, 0.5, N)

# Right-skewed biomarker (lognormal): arm B markedly higher.
biomarker_skew = np.exp(
    0.4 + 0.9 * (arm == "B") + rng.normal(0, 0.35, N)
)

# ANCOVA outcome driven by age and arm.
ancova_outcome = 120 + 0.6 * (age - 55) + 10 * (arm == "B") + rng.normal(0, 4, N)
outcome2 = 50 + 0.3 * (age - 55) + 4 * (arm == "B") + rng.normal(0, 2, N)

# Repeated measures (wide): increasing over time, arm B steeper slope.
pre_score = 5.0 + rng.normal(0, 0.8, N)
post_score = pre_score + 1.5 + 1.0 * (arm == "B") + rng.normal(0, 0.5, N)
followup_score = post_score + 1.0 + 0.8 * (arm == "B") + rng.normal(0, 0.5, N)

# Binary event: arm B higher event rate.
logit = -1.0 + 1.1 * (arm == "B")
p_event = 1 / (1 + np.exp(-logit))
event_binary = (rng.uniform(size=N) < p_event).astype(int)

# Categoricals.
category_binary = np.where(event_binary == 1, "Yes", "No")
cat3_pool = np.array(["Alpha", "Beta", "Gamma"])
category_three = cat3_pool[rng.integers(0, 3, N)]

# Paired binary columns for McNemar / Cochran Q (3 raters).
paired_pre = (rng.uniform(size=N) < 0.4).astype(int)
paired_post = (rng.uniform(size=N) < 0.55).astype(int)
paired_third = (rng.uniform(size=N) < 0.50).astype(int)

# Stratification (2 strata) and ordinal dose (0/1/2).
stratum = np.where(np.arange(N) % 2 == 0, "S1", "S2")
ordinal_dose = (np.arange(N) % 3).astype(int)

df = pd.DataFrame({
    "id": [f"P{i+1:03d}" for i in range(N)],
    "arm": arm,
    "group3": group3,
    "factor2": factor2,
    "age": age.astype(int),
    "biomarker_normal": np.round(biomarker_normal, 2),
    "biomarker_skew": np.round(biomarker_skew, 2),
    "ancova_outcome": np.round(ancova_outcome, 1),
    "outcome2": np.round(outcome2, 1),
    "pre_score": np.round(pre_score, 1),
    "post_score": np.round(post_score, 1),
    "followup_score": np.round(followup_score, 1),
    "event_binary": event_binary,
    "category_binary": category_binary,
    "category_three": category_three,
    "paired_pre": paired_pre,
    "paired_post": paired_post,
    "paired_third": paired_third,
    "stratum": stratum,
    "ordinal_dose": ordinal_dose,
})

# Inject missingness deterministically by row index, mirroring the 10x20 intent:
#   biomarker_normal, biomarker_skew, category_three, followup_score.
MISS = {
    "biomarker_normal": [9, 41, 88, 134, 177],      # ~2.5%
    "biomarker_skew": [27, 95, 156],                 # ~1.5%
    "category_three": [13, 70, 119, 188],            # ~2%
    "followup_score": [5, 60, 123, 150, 199],        # ~2.5%
}
for col, idxs in MISS.items():
    df.loc[idxs, col] = np.nan

assert df.shape == (200, 20), df.shape
# Each arm must retain >=2 complete repeated-measure rows etc.
df.to_csv(OUT, index=False, float_format="%.4g")
print(f"wrote {OUT}: {df.shape}")
print("missing by column:")
print(df.isna().sum()[df.isna().sum() > 0])
print("arm counts:", df["arm"].value_counts().to_dict())
print("group3 counts:", df["group3"].value_counts().to_dict())
