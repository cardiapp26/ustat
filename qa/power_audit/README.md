# Power Analysis — R cross-validation

Runs `/api/stats/power` over a fixed grid of cases and puts each answer next to
the same calculation in R's `pwr`.

```bash
backend/.venv/bin/python qa/power_audit/audit.py       # uSTAT
Rscript                   qa/power_audit/reference.R   # R
backend/.venv/bin/python qa/power_audit/compare.py     # differences
```

`compare.py` exits with the number of mismatches, so it can gate a check.
`backend/tests/test_power_vs_r.py` locks the agreed values as literals, so the
suite needs neither R nor this harness.

## Cases

25, in `cases.json`: every one of the eight test families the panel offers
(two-sample t, one-sample t, one-way ANOVA, correlation, two proportions,
chi-square, Cox, logistic), each solved in every direction it supports — for
power, for sample size, and for the minimum detectable effect — with one- and
two-tailed variants and a deliberately small n in each family, because a power
calculation is usually about a study that is small.

## What was wrong

**One-way ANOVA was wrong by a factor of k, in both directions.**
statsmodels' `FTestAnovaPower` takes `nobs` as the TOTAL sample size. The panel
asks the user for participants *per group* — its own field help says so, and
its own label reports "n/group" — and the two were never translated. With four
groups:

| | reported | correct (`pwr.anova.test`) |
|---|---|---|
| power, f = 0.25, 52/group | 0.275 | **0.864** |
| n, f = 0.25, 80% power | 179/group, 716 total | **45/group, 180 total** |

So a properly powered study was called hopeless, and a study needing 179
participants was told to recruit 716. Both directions now agree with `pwr` to
1e-9. The minimum-detectable-effect branch carried the same mistake and is
converted too, as is the power curve, whose x axis is per-group.

**Correlation power used the wrong reference distribution.** The critical
value came from a normal, where the test that will actually be run is a t on
n-2 df, and the Fisher z of a sample correlation is biased upward by
r/(2(n-1)) — a term the transform is normally applied with. Both matter most
at small n. At n = 12, r = 0.5 the panel reported 0.378 against
`pwr.r.test`'s 0.400; at n = 85, r = 0.3 it reported exactly 0.800 — landing
on the conventional threshold that decides whether a study goes ahead — where
the correct value is 0.804. Now Cohen (1988) as `pwr.r.test` implements it.

## Agreement now

All 25 cases match to 1e-6 relative or better, except solving for the effect
size, where both sides root-find (brentq against uniroot) and stop within
their own tolerances at about 1e-5. Sample sizes are integers after a ceiling
and match exactly.

Two-sample and one-sample t match `pwr.t.test`, including one-tailed and at
n = 10. Two proportions match `pwr.2p.test` on Cohen's h, including a rare
outcome (5% against 1%). Chi-square matches `pwr.chisq.test` with df = k - 1.
Cox matches Schoenfeld's events formula, including unbalanced exposure.

## A naming trap that is not a bug

The logistic family posts its effect in a field called `log_or`, but the panel
labels that field "Odds ratio (OR)" and sends the odds ratio itself; the
endpoint takes any positive value as an OR and logs it, and only treats a
value ≤ 0 as an already-logged coefficient. The UI and the endpoint agree, so
nothing is wrong with what a user sees — but anyone calling the API directly
and passing a genuine log odds ratio of, say, 1.5 will get the answer for
OR = 1.5. Recorded here rather than renamed, because the field name is part of
the published request shape.

Cox and logistic both implement a normal-approximation formula (Schoenfeld,
and Hsieh's continuous-covariate form) rather than wrapping an R package, so
the reference reproduces those formulas explicitly instead of calling
`powerSurvEpi` or `powerMediation`, whose accrual and covariate assumptions
differ.
