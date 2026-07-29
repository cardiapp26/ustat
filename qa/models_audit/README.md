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

Installed: `survival`, `MASS`, `lme4`, `logistf`, `ordinal`.
Absent: `geepack`, `MatchIt`, `survey`, `car`, `rms` — so **GEE, PSM and IPTW
have no R counterpart in this harness** and are not covered.

## Agreement

Exact to at least 1e-5 relative: linear, polynomial, logistic, Poisson, Cox
(coefficients, SEs, HRs, concordance), Kaplan-Meier, and the mixed model's
fixed effects and variance components. Linear AIC/BIC now match R to 1e-9.

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
