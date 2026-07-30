# uSTAT Models — R cross-validation

Runs every Models endpoint on a fixed 300-row sample dataset and puts the
output next to the same model fitted in R.

```bash
backend/.venv/bin/python qa/models_audit/generate_dataset.py   # fixed seed
Rscript                   qa/models_audit/reference.R          # R numbers
backend/.venv/bin/python qa/models_audit/audit.py              # uSTAT numbers
backend/.venv/bin/python qa/models_audit/compare.py            # differences
```

`compare.py` exits with the number of mismatches, so it can gate a check.

## Data

`dataset.csv` — 300 rows, one outcome of every kind the panel fits:
continuous (`sbp`), binary (`event_binary`), count (`admissions`),
overdispersed count (`visits`), positive-skewed (`cost`), ordinal (`grade`),
and time-to-event (`time` / `status`, 182 events). `dataset_long.csv` carries
three visits per subject for the mixed model. Fixed seed, no patient data.

## What R can and cannot check here

Installed: `survival`, `MASS`, `lme4`, `logistf`, `ordinal`, `geepack`,
`MatchIt`, `survey`, `car`, `rms`, `pROC`, `sandwich`, `lmtest`. Every model
the panel offers now has an R counterpart except propensity-score matching,
which is compared on balance rather than on the matched set (see below).

## Agreement

Exact to at least 1e-5 relative: linear, polynomial, logistic, Poisson, Cox
(coefficients, SEs, HRs, concordance), Kaplan-Meier, and the mixed model's
fixed effects and variance components. Linear AIC/BIC now match R to 1e-9.
GEE matches `geepack::geeglm` to 1e-8 on every coefficient and standard
error. The IPTW treatment effect matches `survey::svyglm` to about 0.2%,
the small gap coming from a slightly different propensity-score fit.
VIF matches `car::vif` to 1e-6. ROC matches `pROC` on the AUC, its DeLong
interval, the Youden operating point and the two-sample DeLong test.
Restricted cubic splines match `rms::lrm` on Harrell knot placement and on
the nonlinearity Wald test.

## Propensity score matching

uSTAT retains 121 matched pairs where `MatchIt` retains 138 on the same
caliper. The two use different matching implementations — greedy nearest
neighbour against MatchIt's, and the caliper is applied on a different scale
— so the matched sets are not expected to be identical. Balance after
matching is comparable (average SMD 0.016). The endpoint is not held to
MatchIt's sample size in the tests; only its balance and its reported effect
are checked.

## Differences that are real and deliberate

**Negative binomial standard errors.** uSTAT estimates the dispersion by
maximum likelihood jointly with the coefficients, and its standard errors
therefore carry the uncertainty in that estimate. R's `glm.nb` reports
standard errors conditional on the fitted theta, so R's come out slightly
smaller (about 2%). The coefficients and theta itself agree to 7 decimals.
uSTAT's are the conservative side of the difference.

**Gamma AIC.** R's `Gamma()$aic` estimates the shape by maximum likelihood
before evaluating the log-likelihood; statsmodels uses the Pearson
dispersion. On this dataset that leaves a gap of 0.77. Both count the
dispersion as a parameter.

**Firth p-values and CIs.** R's `logistf` reports penalised *profile*
likelihood p-values and intervals; uSTAT reports Wald, and says so in the
response. On this dataset the p-values differ by up to 22%. Firth exists
for separated data, where Wald inference is least trustworthy, so this is a
genuine limitation rather than a rounding difference. A drop-column
likelihood-ratio test is **not** a valid substitute — deleting a column also
changes the Jeffreys penalty, and trying it moved the age p-value from
0.0077 to 5.6e-05 against R's 0.0065, i.e. further away. The correct test
constrains the coefficient to zero while evaluating the penalty on the full
design, which needs a constrained Firth fit.

**Ordinal.** statsmodels' `OrderedModel` and `MASS::polr` are both ML fits of
the same model and converge to slightly different points; the estimates agree
and the standard errors differ in the fourth significant digit.
