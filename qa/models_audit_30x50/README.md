# Regression and survival at n = 30 — R cross-validation

Runs every regression and survival endpoint on a 30-row × 50-column dataset
with missing values, and puts the output next to the same model fitted in R.

```bash
backend/.venv/bin/python qa/models_audit_30x50/generate_dataset.py   # fixed seed
Rscript                   qa/models_audit_30x50/reference_regression.R
Rscript                   qa/models_audit_30x50/reference_advanced.R
backend/.venv/bin/python qa/models_audit_30x50/audit.py              # uSTAT
backend/.venv/bin/python qa/models_audit_30x50/compare.py            # differences
```

`compare.py` exits with the number of mismatches, so it can gate a check.
`backend/tests/test_models_vs_r_30x50.py` locks the agreed values as literals,
so the suite needs neither R nor a checked-in CSV.

## Why thirty rows

Thirty is where the large-sample shortcuts break: t against z, exact against
asymptotic, a category whose third level holds one observation, a design
matrix one column from singular, a Cox fit with eighteen events. The 300-row
audit under `qa/models_audit` found four wrong models. This one found nine,
and none of them needed a rare input — a text treatment column, a constant
column, a landmark plot grouped by the variable it adjusts for.

## Data

`dataset.csv` — 30 rows, 50 columns, 21 of them carrying missing values under
three named mechanisms, so a shifted result can be traced to the mechanism
that shifted it:

* **clean** — the control: `age`, `sex`, `arm`, `time`, `status`, `site`
* **MCAR** — 10% (`bmi`, `stage`, `ldl`), 25% (`hdl`, `region`, `qol`), 40%
  (`potassium`, deliberately severe)
* **MAR** — missing conditional on `age` (`crp`, `biomarker`, `egfr`) or on
  `diabetes` (`glucose`)

Outcomes are generated from the TRUE covariates before any masking, so the
missingness is a property of what the analyst sees, not of the data-generating
process. Complete-case n ranges from 30 down to 16 depending on the model.

Also carried, because each one breaks something: `rare_grp` (third level n=1),
`solo_grp` and `const_num` (no variance), `score2 = 2·score1 + noise` (VIF),
`sep_binary` (perfectly separated by `prior_tx`), `cmp_status` (competing
risks), `ic_l`/`ic_r` (interval censoring).

Companion frames: `dataset_long.csv` (3 visits × 30 subjects),
`dataset_recurrent.csv` (counting process), `dataset_multistate.csv`
(illness–death transitions), `dataset_external.csv` (40 rows with a frozen
linear predictor, for external validation).

## What was wrong

| endpoint | defect | R says |
|---|---|---|
| `/linear` (robust) | HC3 SEs tested against a normal instead of the residual-df t | age p 0.016 → 0.025 |
| `/linear`, `/logistic`, `/poisson`, `/gamma`, `/negbinom`, `/ordinal`, `/multi_outcome_regression` | a zero-variance predictor silently became the intercept and was reported as a predictor | `const_num` p = 1.2e-12 in OLS, 0.044 in logistic; no intercept row at all |
| `/survival_advanced/fine_gray` | augmented rows had no start time, so competing-event subjects sat in every earlier risk set and none of their own | 1000 rows, 543 competing: 0.306 vs crr's 0.610 |
| `/survival_advanced/frailty` | reported an L2-penalised fit whose penalty grew with the estimated heterogeneity; the cluster never entered the likelihood | 1.073 (SE 0.373) vs 1.232 (SE 0.459) |
| `/survival_advanced/multistate`, `/dynamic_prediction`, `/frailty` | categorical predictors integer-coded, so a 3-level factor got one "per step" coefficient | `stage` → one HR instead of stageII and stageIII |
| `/survival_advanced/dynamic_prediction` | averaged raw covariates, so any categorical predictor was a 500 | — |
| `/survival_advanced/landmark` | grouping by a variable and adjusting for it duplicated the column; 500 on `.unique()` | — |
| `/models/psm`, `/models/iptw` | the "must be binary" check was unreachable — the cast raised first, giving a 500 with a full traceback | — |
| `/models/survival/cox_rcs` | covariates coerced with `to_numeric`, so a categorical became NaN, every row dropped, and the refusal blamed the data | — |

The Correlation panel's four tabs — Pairwise, Matrix, ICC, Cohen's κ — were
audited the same way, against `irr` and `cor.test` on
`dataset_raters.csv` (three raters scoring the same 30 subjects, rater B
carrying a deliberate +2.1 offset so the ICC forms cannot coincide):

| endpoint | defect | R says |
|---|---|---|
| `/stats/icc` | point estimate is ICC(A,1) absolute agreement, interval was the CONSISTENCY one | [0.814, 0.955] reported where the agreement interval is [0.800, 0.952] |
| `/stats/cohens_kappa` | `pe` returned `po` — expected agreement reported as the observed one | 0.90 shown as "expected" where chance agreement is 0.334 |
| `/stats/cohens_kappa` | interval unbounded above; no test at all | upper limit 1.011 for a statistic capped at 1; `irr` gives z = 6.587 |
| `/stats/fleiss_kappa` | rounded to four decimals on the way out; a different published null variance than the reference | SE 0.0813 against `irr`'s 0.0792, z 9.07 against 9.30 |
| `/stats/correlation_pair` | every method other than "pearson" fell through to Spearman in silence | "kendall" answered 0.372 where τ is 0.242; "banana" answered too |
| `/stats/correlation_pair` | Fisher-z limits used a hardcoded 1.96 | limits differed from `cor.test` in the sixth decimal |

Fifteen fixes in all. Every one is verified against R and pinned in
`backend/tests/test_models_vs_r_30x50.py`.

## Agreement now

Exact to 1e-6 or better: linear (plain and HC3), polynomial, logistic,
Poisson, GEE (`geeglm`, both gaussian and binomial), LMM fixed effects and
variance components, Kaplan-Meier, Cox, Fine-Gray (`cmprsk::crr`, coefficients
and standard errors), the frailty endpoint against
`coxph(..., cluster=site)`, multistate transitions, RMST, landmark,
interval-censored, external validation.

Cox coefficients agree to about 1e-4 relative, not better: lifelines stops its
Newton iteration slightly earlier than `survival` does. There are no ties in
this data, so it is not a tie-handling difference.

**Kaplan-Meier was not being compared at all** until this was noticed: the
comparator matches models by their coefficient table, KM has none, and it
therefore counted as agreement without a single value being checked.
`_compare_km` now checks the log-rank statistic, the per-group n, events and
median, and survival with its confidence limits at each requested time. All
of it agrees exactly. The confidence limits are log-log (cloglog) on both
sides — R's `survfit` defaults to `conf.type="log"`, so the reference script
asks for log-log explicitly; otherwise the comparison flags a convention as
an error.

## Differences that are real and deliberate

**Firth standard errors.** uSTAT inverts the unpenalised Fisher information at
the penalised estimate; `logistf` inverts the penalised information. uSTAT's
come out about 5% larger — conservative, but not the same quantity. On top of
that `logistf` reports penalised *profile* likelihood p-values where uSTAT
reports Wald, and under complete separation the two are five orders of
magnitude apart (0.0018 against 4.6e-08). Firth exists for separated data, so
this is where it matters most. The correct test constrains the coefficient to
zero while evaluating the penalty on the full design, which needs a
constrained Firth fit.

**Recurrent events (LWYY).** The robust standard errors come from lifelines'
cluster sandwich on left-truncated `(start, stop]` data, which does not
reproduce `survival::coxph(..., cluster=id)` — verified by computing R's
value by hand from the dfbeta residuals. Here the age SE is 0.0125 against
R's 0.0145, about 14% smaller, i.e. anti-conservative. The point estimates
match R exactly. lifelines cannot compute residuals with entry times, and
statsmodels' `PHReg` with `entry` does not reproduce R's coefficients either,
so this needs a hand-written score-residual sandwich. The response carries an
`se_note` saying so.

**Fine-Gray standard errors at small n.** The estimator agrees with `crr`
exactly, and so do the standard errors on 1000 rows. At n = 30 they differ by
about 8%, because the IPCW reformulation's sandwich does not carry the
uncertainty in the estimated censoring distribution, a term that vanishes as
n grows.

**Shared frailty is not a frailty model.** What the endpoint now reports is a
marginal cluster-robust Cox fit, which is a defensible analysis of clustered
survival data and matches R's `cluster()` term to six decimals. It is not
`coxph(..., frailty(site))`: on this data R's frailty fit gives 1.232 (SE
0.459) against the marginal 1.236 (SE 0.515). A proper shared frailty needs a
penalised partial likelihood with cluster-specific random effects and a
variance component estimated by marginal likelihood; a naive EM was tried and
collapses to theta = 0 even on data simulated with a variance of 0.64, where
R recovers 0.34. Not shipped rather than shipped wrong.

**Separated logistic.** Ordinary ML diverges, in uSTAT and in R alike, and the
two diverge to different places (coefficient 50.3 against 51.1, SE 8.2e5
against 8.6e4). Neither is a number to report. uSTAT raises a warning; the
right analysis is Firth.

**Spearman's p-value.** scipy uses the t approximation; R's `cor.test` uses
the AS89 Edgeworth approximation for n > 9. At n = 27 that is 0.05619 against
0.05698 — a 1.4% relative difference, and on the anti-conservative side. Both
are approximations to the same exact permutation distribution, which is not
computable at this n; AS89 is the closer of the two. Left as is, and recorded
here because it sits right at the 0.05 boundary.

**Fleiss p-value.** `irr` computes `2*(1-pnorm(|z|))`, which underflows to a
flat 0 past about z = 8.3. uSTAT uses the survival function and reports
1.3e-20. uSTAT is the more useful of the two; the comparator treats a tiny p
against R's 0 as agreement.

**Negative binomial, Gamma AIC, ordinal.** Unchanged from the 300-row audit and
written up in `qa/models_audit/README.md`.
