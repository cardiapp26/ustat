# uSTAT 200×20 statistical audit fixture

Deterministic synthetic dataset (200 rows × 20 columns) for read-only
validation of uSTAT's Tests and Table tabs at a sample size where large-sample
approximations, repeated-measures, mixed ANOVA, and two-way ANOVA all actually
run (the 10×20 fixture could not exercise two-way ANOVA or large-sample
non-inferiority paths).

## Files

- `dataset.csv` — the fixture (200 × 20). SHA-256 recorded in `report.md`.
- `generate_dataset.py` — produces `dataset.csv` deterministically (seed
  `20260728`, no randomness, no patient data). Re-runnable.
- `reference.R` — independent reference calculations (one-sample/independent/
  paired t, one-way + Welch ANOVA, Mann-Whitney, Kruskal-Wallis, chi-square
  Yates/uncorrected/MC, Fisher, Friedman, ANCOVA, MANCOVA, two-way ANOVA,
  split-plot mixed ANOVA, RM ANOVA, and correct one-sided non-inferiority p).
- `audit_continuous.py` — descriptive + continuous inferential checks vs R.
- `audit_categorical.py` — categorical/contingency + non-inferiority checks.
- `audit_repeated.py` — paired/RM/mixed/two-way ANOVA/ANCOVA checks.
- `audit_table1.py` — Table 1 parity with Tests panel + missing-value handling.
- `report.md` — human-facing findings report (Turkish).

## Run

```bash
# independent reference
Rscript qa/statistical_audit_200x20/reference.R \
  qa/statistical_audit_200x20/dataset.csv > /tmp/r_refs_200.txt

# each audit script uses TestClient in-process and compares to the R refs
.venv/bin/python qa/statistical_audit_200x20/audit_continuous.py
.venv/bin/python qa/statistical_audit_200x20/audit_categorical.py
.venv/bin/python qa/statistical_audit_200x20/audit_repeated.py
.venv/bin/python qa/statistical_audit_200x20/audit_table1.py
```

## Schema (mirrors the 10×20 fixture)

- Continuous: `age`, `biomarker_normal`, `biomarker_skew`, `ancova_outcome`,
  `outcome2`, `pre_score`, `post_score`, `followup_score`.
- Categorical: `arm` (A/B, grouping), `group3` (Low/Medium/High), `factor2`
  (X/Y), `event_binary` (0/1), `category_binary` (Yes/No),
  `category_three` (Alpha/Beta/Gamma), `stratum` (S1/S2), `ordinal_dose`
  (0/1/2).
- Paired-binary: `paired_pre`, `paired_post`, `paired_third`.
- Missing: `biomarker_normal` 5, `biomarker_skew` 3, `followup_score` 5,
  `category_three` 4.

No patient data. Source-controlled values. Product code is read-only during
this audit.
