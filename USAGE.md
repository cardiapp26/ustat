# uSTAT — Usage Guide

A step-by-step companion to the [README](README.md). Each section assumes you
have <https://ustat.drtr.uk> open in a recent Chrome / Firefox / Safari / Edge.

> **Looking for "which test do I run, and how?"** See **[MANUAL.md](MANUAL.md)** —
> a complete, screenshot-illustrated reference that maps every analysis to its
> tab and exact steps. It is written to be scanned by an AI assistant: tell a
> chatbot *"I uploaded my data to uSTAT, which tests should I run and how?"* and
> it can answer directly from that file.

> ⚠️ uSTAT is **not a medical device** and has **not yet been validated through
> peer-reviewed publications**. Verify any clinically or scientifically
> important result against an established statistics package (SPSS, R, Stata,
> SAS) before reporting.

---

## 0. Preflight

Before uploading anything sensitive:

1. **Anonymise.** Strip names, MRNs, dates of birth, free-text identifiers.
2. **Close other tabs** on shared machines. Use a private / incognito window.
3. **Disable** untrusted browser extensions on the uSTAT domain.
4. Read [`/privacy.html`](frontend/public/privacy.html) — the formal Privacy
   Policy, especially §6 (regulated data).

uSTAT keeps your data only in server RAM, never on disk, and auto-discards
it 30 minutes after your last activity.

---

## 1. Upload your data

Supported formats (auto-detected from the file extension):

| Extension                | Notes                                                  |
|--------------------------|--------------------------------------------------------|
| `.csv`                   | Comma or tab. UTF-8 preferred.                         |
| `.xlsx` / `.xls`         | Excel.                                                 |
| `.sav`                   | SPSS — value labels and variable labels are preserved. |
| `.sas7bdat`              | SAS.                                                   |
| `.dta`                   | Stata.                                                 |
| `.json` (uSTAT session)  | Reload a previously saved session (variables, labels, filters, audit). |

Drop the file on the **Statistical Analysis** tile on the splash screen, or
click to browse.

The server detects variable types (numeric / categorical / date) and shows a
data preview. Variables can be re-typed in the Dictionary modal at any time.

---

## 2. Inspect & clean

Open the **Data** tab. The toolbar (left to right):

- **Dictionary** — variable label, value labels, type override, missing
  recoding for each column.
- **+ Row** / **+ Column** — insert blank rows / columns.
- **Undo / Redo** — full history of edits in this session.
- **Save Session** — download the JSON, reload it later via the splash screen.
- **Select cases** — filter to a subset for all downstream analyses.

Right-click any column header to:

- Rename, recode, fill blanks (mean / median / MICE), delete.
- Insert a column to the left / right.
- Copy a column / row to the clipboard.

Right-click any cell to copy / paste a rectangular block. Ctrl+V pastes from
Excel or another CSV.

---

## 3. Descriptive statistics

**Summary** tab — point at any variable to get:

- For numeric: N, mean, SD, median, IQR, min, max, Shapiro-Wilk p, skewness,
  kurtosis, Q-Q plot.
- For categorical: counts, percent, bar chart.

Use the **search box** at the top of the variable list to jump quickly when
you have hundreds of columns.

---

## 4. Hypothesis tests

**Tests** tab. Pick the sub-tab that matches your design:

| Sub-tab        | Tests                                                          |
|----------------|----------------------------------------------------------------|
| Hypothesis     | t-test (independent / paired), Mann-Whitney U, Wilcoxon signed-rank, one-way ANOVA, Kruskal-Wallis, Tukey HSD, two-way ANOVA, ANCOVA |
| Repeated       | Repeated-measures ANOVA, mixed ANOVA, Friedman                 |
| Categorical    | χ², Fisher's exact, McNemar, Cochran's Q, Mantel-Haenszel       |
| Reliability    | Cronbach's α, ICC, Cohen's κ                                    |

uSTAT runs Shapiro-Wilk + Levene first and **auto-selects** the parametric
vs non-parametric path. You can override the choice in the panel.

Every result includes effect size with 95% CI and a plain-English summary
under the table.

---

## 5. Correlation

**Correlation** tab. Pick:

- **Pairwise** — Pearson, Spearman, or Kendall, with significance.
- **Matrix** — heat-map of pairwise correlations for many variables.

Toggle log scales, axis units, and palette via the chart toolbar.

---

## 6. ROC & diagnostic accuracy

**ROC** tab.

1. Pick the predictor and the binary outcome.
2. Optional: stratify by a group variable to compare ROC curves.
3. Read AUC + 95% CI, sensitivity / specificity at the Youden index, and the
   coordinates table.
4. Use **Compare** to test two ROC curves on the same cohort (paired DeLong
   1988) or two cohorts on different curves (unpaired).
5. Use **Combine** to derive a logistic-combined score from several
   predictors and evaluate its ROC.

---

## 7. Regression models

**Models** tab. Pick the model on the left.

### Linear

- Outcome (continuous) + predictors. Categorical predictors are auto-dummied.
- Toggle **Robust SE (HC3)** when residuals look heteroscedastic.
- Output: β, SE, t, p, 95% CI, R², F. Diagnostic plots (residuals vs fitted,
  Q-Q, scale-location, leverage) on a separate card.

### Logistic / OR Table

- Outcome must be 0/1 (the panel warns if it is not).
- **OR Table** runs univariate logistic per predictor, then a multivariate
  model with one of several selection strategies (`p < 0.10`, `p < 0.05`,
  forward, backward, all). Standard for clinical papers.

### Poisson / negative binomial / gamma / polynomial / LMM

Same shape as the others: predictors checklist, optional offset / scale, run.

### RCS dose-response

- Pick a continuous **predictor** and an **outcome type** — Logistic, Linear,
  or **Cox** (with duration + event columns).
- **Knots**: 3 / 4 (★ clinical default) / 5.
- **Knot positions**: *Harrell percentiles* (default — 5/35/65/95 for 4
  knots) or *Custom* (comma-separated, e.g. `70, 100, 130, 160` mg/dL for
  LDL).
- Optional **reference value** for the OR / HR = 1.0 anchor (defaults to the
  median).
- Optional **covariates** (linear terms only).

Output: dose-response curve with 95% CI, nonlinearity p, AIC, knot positions.

### Cox-RCS (multivariable)

The big one — supports the full clinical pipeline:

```
Surv(time, event) ~ rcs(LDL, 4) * rcs(AGE, 4) + SEX + DM + HT + SMOKER
```

1. Pick the duration and event columns.
2. Add one or two **Spline term cards** (LDL, AGE, …). Each card has its own
   knot count and optional custom knot positions.
3. Check additive **linear covariates** (SEX, DM, HT, SMOKER).
4. Toggle **Include RCS × RCS interaction** to add the tensor-product columns
   plus an LR test against the main-effects-only model.
5. Read:
   - Coefficient table (β, HR, 95% CI, p).
   - Per-term Wald **nonlinearity** badge.
   - **Interaction LR test** badge.
   - 1D HR curve for each spline term.
   - 2D HR contour plot (when interaction is on).

The same workflow exists as four pre-filled templates in the **Code** tab if
you prefer to drive lifelines directly.

---

## 8. Survival

**Survival** is split across two places:

- **Models** tab → Cox-RCS — the parametric / spline path (above).
- **Survival Advanced** sub-tab of Models — Kaplan-Meier, log-rank,
  multivariable Cox, Fine-Gray competing risks, landmark, E-value.

The **PSM** tab also has a Survival outcome path (see §10).

---

## 9. Table 1 (publication-ready)

**Table** tab.

1. Pick a **Group by** column (optional; if blank, you get a single column of
   overall statistics).
2. Pick which **Statistics** to display (Mean ± SD, Median [IQR], counts,
   percentiles, …).
3. (Optional) Tick **Show SMD** to display the standardised mean difference
   for each variable.
4. (Optional) Tick **Test normality within each group** — runs Shapiro-Wilk
   / Lilliefors on **each group separately** and takes the parametric path
   only when every group passes p > 0.05. Stricter than the default
   pooled-sample check.
5. Select variables and press **Generate Table**.

Output: variable names down the left, statistic per group across the top,
p-values + test names + (optional) SMD.

Hit **Format for Journal (AMA)** to convert to AMA style and download Excel
or Word.

---

## 10. Propensity Score Matching (PSM)

**PSM** tab.

### Setup

1. **Treatment variable** — must be binary 0/1.
2. **Outcome type**: Binary (matched-pair GEE logistic) or Survival
   (stratified Cox PH on the matched cohort).
3. **Covariates** — the confounders to balance.
4. **Score model**: Logistic ★ (default), Probit, or GBM.
5. **Caliper**: width in SD units (Cochran & Rubin standard = 0.2).
6. **Caliper scale**: Logit ★ (Austin 2011 recommendation) or Raw PS.
7. **Trim to common support** (Crump 2009).
8. **Ratio**: 1:1 through 1:5.
9. **Seed** for reproducibility.
10. **Matching**: Greedy ★ (NN with caliper) or Optimal (Hungarian, 1:1 only).
11. **Exact-match strata**: categorical columns that must agree before NN.
12. For binary outcomes at 1:1, optionally enable **Rosenbaum bounds**
    (sensitivity to hidden bias).

### Output

- **Summary banner** — n matched / n unmatched / trimmed count, caliper info.
- **Love plot** — SMD before vs after for every covariate.
- **Balance table** — SMD (before / after), **Rubin variance ratio**, **KS
  p-value**, reduction %, balanced flag.
- **Propensity-score overlap** histogram.
- **Outcome panel** — GEE logistic OR for binary outcomes, stratified Cox HR
  for survival outcomes.
- **Rosenbaum bounds** — critical Γ + full Γ-vs-p curve (when enabled).

The matched cohort is auto-saved as a JSON session
(`session_id + "_psm"`) you can reload as a fresh dataset.

---

## 11. Power analysis

**Power** tab. No data required.

1. Pick the test family (t-test, ANOVA, two-proportion, correlation, Cox /
   log-rank).
2. Enter any three of: power, effect size, sample size, α.
3. uSTAT solves for the fourth and shows the power curve.

References: `statsmodels.stats.power` and, for survival, Schoenfeld 1981 /
Freedman 1982 formulas.

---

## 12. Code sandbox (advanced, optional)

Off by default. When the deployment owner sets `ENABLE_CODE_RUNNER=1`, a
**Code** tab appears.

The editor has four pre-filled templates that exactly map to the [Cox-RCS
pipeline above](README.md#statistical-pipeline-worked-example):

1. Step 1 — Univariate Cox-RCS, Harrell knots.
2. Step 1b — Univariate Cox-RCS, clinical knots (70 / 100 / 130 / 160).
3. Step 2 — Multivariable Cox-RCS.
4. Step 3 — RCS × RCS interaction with the LR test.

The session DataFrame is exposed as `df`. Allowed imports: `numpy`,
`pandas`, `scipy`, `statsmodels`, `lifelines`, `sklearn`, `matplotlib`,
`seaborn`, plus a safe subset of the stdlib. Everything network-, OS-,
subprocess-, or pickle-related is denied. CPU 30 s / RAM 512 MB hard
limits. See [`backend/SECURITY.md`](backend/SECURITY.md) for the full
threat model.

Output goes to two tabs:

- **Console** — `stdout`, `stderr`, any traceback.
- **Figures** — every open Matplotlib figure as a 120 DPI PNG you can
  right-click → Save image.

---

## 13. Save & resume

Top toolbar **Save** menu:

- **Save as CSV** — dataset only (current edits, no metadata).
- **Save as XLSX** — same, in Excel with a value-labels sheet.
- **Save Session (.json)** — dataset + variable labels + value labels +
  filters + audit trail + Dictionary metadata. Drop the JSON back on the
  splash screen later to resume exactly where you left off.

The matched cohort from PSM is also stored as a session under
`<original_session_id>_psm`, downloadable from its own toolbar.

---

## 14. Troubleshooting

| Symptom                                                                                    | Likely cause / fix |
|--------------------------------------------------------------------------------------------|--------------------|
| **"Cannot connect to backend (localhost:8000)"**                                            | Local dev only — start uvicorn.                                                       |
| **"Logistic RCS requires binary 0/1 outcome"**                                              | Recode the outcome variable in the Dictionary modal, or use Cox / Linear outcome.     |
| **"Predictor 'X' has only N unique values — need ≥ N+2 for k-knot spline"**                 | Drop one knot or pick a richer variable.                                              |
| **No matches found in PSM**                                                                | Widen the caliper, switch to logit-PS scale, or enable common-support trimming.       |
| **"Server busy: max X concurrent code runs"**                                              | Code-runner concurrency cap hit — wait and retry, or raise `CODE_RUNNER_MAX_CONCURRENT`. |
| **XLSX export does nothing**                                                                | Old browser cache — hard reload (Cmd-Shift-R). The fix landed in v1.7.0.              |
| **Code tab is missing**                                                                    | The deployment owner has not set `ENABLE_CODE_RUNNER=1`. Off by default in production. |

If something else breaks, open an issue at
<https://github.com/afstudy20-gif/ustat/issues> with the steps to reproduce,
the browser console output, and (if relevant) a minimal anonymised dataset.

---

## 15. Where to go next

- [README](README.md) — architecture, statistical methods, citation.
- [SECURITY.md](backend/SECURITY.md) — threat model + production hardening.
- [Privacy Policy](frontend/public/privacy.html) — formal data-handling
  statement.
- [Terms of Use](frontend/public/terms.html) — no-warranty + acceptable use.
- [Security Overview](frontend/public/security.html) — public-facing
  hardening summary.
- Open an issue or email
  [adycovs@gmail.com](mailto:adycovs@gmail.com) for anything else.
