<div align="center">

<img src="frontend/public/logo.png" alt="uSTAT logo" width="120" />

# uSTAT — Statistical Analysis Platform

**Free, browser-based SPSS / R / Stata alternative for clinicians and researchers.**

[![Live](https://img.shields.io/badge/live-ustat.drtr.uk-4f46e5?style=flat-square)](https://ustat.drtr.uk)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21610852.svg)](https://doi.org/10.5281/zenodo.21610852)
[![Security scan](https://img.shields.io/badge/CI-bandit%20%2B%20pip--audit%20%2B%20semgrep-22c55e?style=flat-square)](.github/workflows/security-scan.yml)
[![No medical device](https://img.shields.io/badge/not-a%20medical%20device-dc2626?style=flat-square)](frontend/public/terms.html)

</div>

uSTAT is a free, server-hosted clinical biostatistics platform aimed at clinicians,
biostatisticians, and medical researchers. Upload a `CSV`, `Excel`, `SPSS`,
`SAS`, or `Stata` file and run 100+ analyses — from basic hypothesis testing to
advanced causal inference, missing-data sensitivity, external validation,
survival ML benchmarking, and decision curve analysis — without installing anything.

The backend has been completely refactored into a clean, maintainable services layer
(20+ focused pure modules) with thin routers. All statistical work happens server-side
in pure Python using peer-reviewed libraries (scipy, statsmodels, lifelines, scikit-learn).

> ⚠️ **Not a medical device.** uSTAT has **not yet been validated through
> peer-reviewed publications**. Independent validation against SPSS, R, and
> Stata is ongoing. Verify any clinically or scientifically important result
> against an established statistics package before reporting.

---

## Table of contents

- [Live instance](#live-instance)
- [Features](#features)
- [Quick start](#quick-start-end-user)
- [Statistical pipeline (worked example)](#statistical-pipeline-worked-example)
- [Local development](#local-development)
- [Self-hosting](#self-hosting)
- [Architecture](#architecture)
- [Statistical methods & references](#statistical-methods--references)
- [Security & privacy](#security--privacy)
- [Code-execution sandbox](#code-execution-sandbox)
- [Contributing](#contributing)
- [Citation](#citation)
- [License & attribution](#license--attribution)

---

## Live instance

Production deployment: **<https://ustat.drtr.uk>**

Operated by **Dr. Yusuf Hoşoğlu**.
Contact: [adycovs@gmail.com](mailto:adycovs@gmail.com)
Vulnerability disclosure: [adycovs@gmail.com](mailto:adycovs@gmail.com?subject=%5BuSTAT-security%5D) (prefix subject with `[uSTAT-security]`) · RFC 9116 contact at `/.well-known/security.txt`.

---

## Features

| Area                          | What you can do |
|-------------------------------|-----------------|
| **Data I/O**                  | CSV, Excel (.xlsx / .xls), SPSS (.sav), SAS (.sas7bdat), Stata (.dta), JSON sessions |
| **Cleaning**                  | Rename, recode, fill blanks (mean / median / MICE), filter cases, compute new variables, Ctrl+V paste from Excel/CSV |
| **Descriptive & survey**      | Mean / SD / median / IQR, weighted (survey) descriptives, Shapiro-Wilk, KS (Lilliefors), Q-Q plots |
| **Hypothesis tests**          | t-test, ANOVA/ANCOVA, repeated-measures, non-parametric, gatekeeping (truncated Hochberg/Holm), non-inferiority/TOST |
| **Categorical & agreement**   | χ² / Fisher / McNemar / Cochran Q / Mantel-Haenszel, Bland-Altman, Deming, Cronbach's α, McDonald's ω, Fleiss' κ |
| **Regression**                | Linear (HC3), logistic (+ Firth), Poisson/NB, ordinal, GEE, **RCS dose-response** (with interactions) |
| **Survival (advanced)**       | KM + log-rank (stratified), Cox (linear + RCS + TV + interactions), Fine-Gray, RMST, Landmark + dynamic prediction, Recurrent (LWYY), **Shared frailty**, **Multistate illness-death**, **Joint longitudinal-survival (two-stage)** |
| **External validation**       | C-index, calibration slope/intercept, IPCW Integrated Brier Score, time-dependent AUC, transportability diagnostics (Phase 9 framework) |
| **Survival ML**               | Gradient Boosting ranking benchmark vs Cox + permutation importance + direct feed into Phase 9 validation (risk scores → survival probabilities → full IBS/tdAUC on shifted cohorts) |
| **ROC / Prediction / Utility**| ROC + DeLong, calibration + Hosmer-Lemeshow, Brier, **professional Decision Curve Analysis** (binary + survival modes, standardized net benefit, interventions avoided — Phase 13) |
| **Causal inference**          | **PSM** (greedy/optimal + Rosenbaum bounds), **IPTW** (ATE/ATT/overlap + weighted outcome models), E-value + quantitative bias analysis (MNAR sensitivity) |
| **Missing data**              | MICE + Little's MCAR + delta-adjustment MNAR sensitivity analysis |
| **Tables & reporting**        | Publication-ready Table 1 (with SMD), Methods appendix DOCX from audit log, high-DPI chart export |
| **Power & meta**              | Power (t/ANOVA/proportions/Cox), full meta-analysis (DL/PM random effects, prediction interval, subgroup, meta-regression, Egger/Begg/funnel/trim-and-fill) |
| **Time series**               | ARIMA/SARIMA (auto + manual), STL decomposition, ADF + KPSS stationarity |
| **Machine learning**          | Random Forest & Gradient Boosting (honest CV performance, permutation importance) |
| **Code sandbox**              | Optional server-side Python runner with `df` pre-injected (off by default) |

---

## Quick start (end user)

1. Open <https://ustat.drtr.uk>.
2. Drop a file on the **Statistical Analysis** tile (or click to browse). uSTAT auto-detects variable types (numeric / categorical / date).
3. Use the left sidebar to pick an analysis (**Models**, **Tests**, **Survival**, **PSM**, **Power**, **Table**, …). Each panel lets you pick the variables and runs the appropriate test automatically — including normality-based selection between parametric and non-parametric paths.
4. Read the result. Every output includes effect sizes, 95% CIs, assumption diagnostics, and a plain-English summary.
5. Export charts at 600 DPI, copy tables to Word / Excel, or save the whole session as a JSON file to resume later.

Your data stays in server RAM only, is never written to disk, and is automatically discarded 30 minutes after your last activity (see [SECURITY.md](backend/SECURITY.md)).

---

## Statistical pipeline (worked example)

**Question:** how does serum LDL relate to all-cause mortality, after adjusting for age and other covariates?

| Step | UI path | Backend route | What it does |
|------|---------|----------------|--------------|
| 1a. Univariate Cox-RCS, Harrell knots | Models → RCS Dose-Response → Outcome **Cox** | `POST /api/models/rcs` | Cox PH on `rcs(LDL, 4)` only. Knots at the 5/35/65/95 percentile (Harrell standard). Returns HR curve + 95% CI + nonlinearity p. |
| 1b. Univariate Cox-RCS, clinical knots | Same panel, **Knot positions = Custom (70, 100, 130, 160)** | `POST /api/models/rcs` | Sensitivity analysis with clinically meaningful cut-points. |
| 2. Multivariable Cox-RCS | Models → **Cox-RCS (multivariable)** → LDL + AGE + covariates | `POST /api/models/survival/cox_rcs` | `Surv(time, event) ~ rcs(LDL,4) + rcs(AGE,4) + SEX + DM + HT + SMOKER`. Per-term Wald nonlinearity test. |
| 3. Interaction (primary hypothesis) | Same panel, toggle **Include RCS × RCS interaction** | `POST /api/models/survival/cox_rcs` | Tensor-product interaction columns + LR test of full vs main-effects model + 2D HR contour plot of `LDL × AGE`. |

Each step also exists as a code-runner template (Step 1 / Step 1b / Step 2 / Step 3) if you prefer to drive the analysis with lifelines directly — see [Code-execution sandbox](#code-execution-sandbox).

---

## Local development

### Prerequisites

- Python 3.11+
- Node 20+ (or compatible)
- ~3 GB free disk for the Python scientific stack

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API is now on `http://localhost:8000`. Sanity check:

```bash
curl http://localhost:8000/api/health
# {"status":"ok","active_sessions":0,"memory":{...}}
```

### Frontend

```bash
cd frontend
npm install --legacy-peer-deps
npm run dev
```

Vite serves the UI on `http://localhost:5173` and proxies `/api/*` to the
backend automatically (`vite.config.ts`).

### Run both with one command

The repo ships a `.claude/launch.json` consumed by the Claude Preview MCP, but
in plain shell:

```bash
# terminal 1
cd backend && source .venv/bin/activate && uvicorn main:app --reload --port 8000
# terminal 2
cd frontend && npm run dev
```

### Enable the optional code sandbox (off by default)

```bash
ENABLE_CODE_RUNNER=1 uvicorn main:app --reload --port 8000
```

The frontend `Code` tab will then appear; before that, it is hidden by the
`/api/code/status` probe.

---

## Self-hosting

uSTAT is designed to be easy to self-host with Docker.

### Recommended: Coolify (Self-hosted PaaS)

The project is currently deployed on **Coolify**.

**Recommended Coolify settings:**

- **Project Type**: Docker Compose (recommended) or plain Dockerfile
- **Compose File**: Use the `docker-compose.yml` in the repo root
- **Health Check**: `/api/health`
- **Port**: 8000 (Coolify's proxy will handle external access)

**Important Environment Variables** (set these in Coolify → Environment Variables):

```env
CORS_ALLOWED_ORIGINS=https://your-domain.com,http://localhost:5173
SECURITY_CONTACT_EMAIL=security@your-domain.com
ENABLE_CODE_RUNNER=0
CSP_ENFORCE=1
```

**Resource Recommendations** (in Coolify UI under Resources):
- CPU: 1–2 cores
- Memory: 512MB – 1GB (the app is quite lightweight)
- Only increase if you enable the code runner sandbox heavily

**Update Strategy**:
- Push to `main` → Coolify should automatically rebuild and deploy (if Git integration is set up).
- For manual deploys, use the "Deploy" button in the Coolify service dashboard.

A `.env.example` file is included in the repository to help with variable names.

### Quick local / manual Docker

```bash
docker build -t ustat .
docker run -p 8000:8000 \
  -e CORS_ALLOWED_ORIGINS="http://localhost:5173" \
  -e ENABLE_CODE_RUNNER=0 \
  ustat
```

### Environment Variables

See [`backend/SECURITY.md`](backend/SECURITY.md) for the full list of available variables (especially sandbox limits and rate limiting).

Key variables:

| Variable                    | Default     | Recommendation for production      |
|----------------------------|-------------|------------------------------------|
| `ENABLE_CODE_RUNNER`       | `0`         | `0` (keep disabled unless needed)  |
| `CSP_ENFORCE`              | `0`         | `1`                                |
| `CORS_ALLOWED_ORIGINS`     | (dev + prod domain) | Set to your actual domains     |
| `SECURITY_CONTACT_EMAIL`   | ...         | Your security email                |

```bash
# Example hardened production values
ENABLE_CODE_RUNNER=0
CSP_ENFORCE=1
CODE_RUNNER_PER_MIN=2
CODE_RUNNER_MAX_CONCURRENT=1
```

---

## Architecture

uSTAT follows a clean **services-first** architecture (major refactor completed in v3.0):

```
┌────────────────────────────────────────────────────────────────────┐
│ Browser (React 19 + Plotly + zustand)                              │
│   – UI state, charts, and result rendering only                    │
└──────────────────────────────────────────────┬─────────────────────┘
                                               │ HTTPS + security headers
┌──────────────────────────────────────────────▼─────────────────────┐
│ FastAPI (thin routers)                                             │
│   /api/decision_curve, /api/survival_advanced, /api/models/* …     │
├────────────────────────────────────────────────────────────────────┤
│ Pure services layer (20+ focused modules — no god files)           │
│   decision_curve.py, survival_ml.py, external_validation.py,       │
│   multistate.py, frailty.py, joint_model.py, causal_sensitivity.py,│
│   missing_data_sensitivity.py, model_validation.py, psm.py …       │
│   + store.py (in-RAM 30 min TTL sessions)                          │
├────────────────────────────────────────────────────────────────────┤
│ Statistical engines (peer-reviewed, no custom black boxes)         │
│   scipy · statsmodels · lifelines · scikit-learn · pandas · numpy  │
└────────────────────────────────────────────────────────────────────┘
```

Key invariants:

- **Server-side computation.** The frontend renders results; it does not run
  statistical algorithms.
- **No persistence (default).** Uploaded data lives only in server RAM and is
  discarded after `SESSION_TTL_SECONDS` (default 1800). Nothing is written to
  disk. (An optional `SESSION_DISK_CACHE=1` operator flag can snapshot sessions
  to a host volume so edits survive a redeploy — off by default; if you enable
  it, that volume holds unencrypted data at rest, so harden and disclose it.)
- **No accounts.** Sessions are an opaque UUID handed back to the browser; the
  backend never knows who you are.
- **Pure-Python deps.** All statistical computation runs on peer-reviewed
  open-source libraries — no custom black-box estimators.

---

## Statistical methods & references

Every test maps to a specific upstream function. The full table lives in the
in-app **About → Tests & Methods** modal. Highlights:

| Method                                           | Implementation                                  |
|--------------------------------------------------|-------------------------------------------------|
| Independent / paired t-test                      | `scipy.stats.ttest_ind` / `ttest_rel`           |
| ANOVA / ANCOVA / mixed                           | `statsmodels.formula.api.ols` + `anova_lm`, `mixedlm` |
| Mann-Whitney / Kruskal-Wallis                    | `scipy.stats.mannwhitneyu` / `kruskal`          |
| Shapiro-Wilk / Lilliefors                        | `scipy.stats.shapiro` / `statsmodels.stats.diagnostic.lilliefors` |
| χ² / Fisher's exact                              | `scipy.stats.chi2_contingency` / `fisher_exact` |
| Pearson / Spearman / Kendall                     | `scipy.stats.pearsonr` / `spearmanr` / `kendalltau` |
| Linear regression (HC3)                          | `statsmodels.OLS` with `cov_type="HC3"`         |
| Logistic regression (with OR + CI)               | `statsmodels.formula.api.logit`                 |
| Cox proportional hazards                         | `lifelines.CoxPHFitter`                         |
| Kaplan-Meier + log-rank                          | `lifelines.KaplanMeierFitter`, `logrank_test`   |
| Fine-Gray / Multistate / Frailty / Joint models  | lifelines + custom (services/multistate.py, frailty.py, joint_model.py) |
| External validation (IBS, tdAUC, transportability) | services/external_validation.py (Phase 9)     |
| Survival ML benchmark + Phase 9 integration      | services/survival_ml.py (Phase 12)              |
| Professional Decision Curve Analysis (binary + survival) | services/decision_curve.py (Phase 13)         |
| Restricted cubic splines (Harrell)               | `services/rcs_basis.py` (Harrell 2015 §2.4.4)   |
| Propensity-score matching + IPTW                 | sklearn + custom (services/psm.py)              |
| Rosenbaum bounds + E-value + QBA                 | custom (services/causal_sensitivity.py)         |
| MNAR sensitivity (delta adjustment)              | services/missing_data_sensitivity.py            |
| MICE + bootstrap optimism validation             | services/missing_data.py + model_validation.py  |

Methodological references applied:

- Austin PC (2011). *Multivariate Behavioral Research* — PSM balance, logit-PS caliper, pooled-SD denominator.
- Crump RK et al. (2009). *Biometrika* — common-support trimming.
- Rosenbaum PR (2002). *Observational Studies* — bounds on hidden bias.
- Harrell FE (2015). *Regression Modeling Strategies* — RCS knot placement, Cox-RCS, nonlinearity tests.
- Vickers AJ, Elkin EB (2006). *Med Decis Making* — decision curve analysis.
- DeLong ER et al. (1988). *Biometrics* — paired ROC comparison.
- Schoenfeld DA (1981) — Cox model sample size.
- Hosmer DW, Lemeshow S — logistic calibration test.
- Benjamini Y, Hochberg Y (1995) — FDR control.

---

## Security & privacy

A summary; the canonical documents are:

- [`/privacy.html`](frontend/public/privacy.html) — Privacy Policy (data flow, retention, what we do NOT do)
- [`/terms.html`](frontend/public/terms.html) — Terms of Use (no medical device, no warranty, citation)
- [`/security.html`](frontend/public/security.html) — Security Overview (architecture, hardening, automated scans)
- [`backend/SECURITY.md`](backend/SECURITY.md) — sandbox threat model + env knobs
- [`/.well-known/security.txt`](backend/main.py) — RFC 9116 disclosure contact

Highlights:

- HSTS 1-year preload, CSP (report-only until tuned), X-Frame-Options DENY,
  X-Content-Type-Options nosniff, Permissions-Policy denying camera / mic /
  geolocation / payment / USB / motion sensors, COOP same-origin.
- Continuous scans on every push & weekly:
  [`.github/workflows/security-scan.yml`](.github/workflows/security-scan.yml)
  runs `bandit`, `pip-audit`, `npm audit`, `semgrep` (OWASP Top 10), and
  `gitleaks`.
- No persistent storage, no accounts, no third-party analytics or trackers.

If you discover a vulnerability, email
[adycovs@gmail.com](mailto:adycovs@gmail.com?subject=%5BuSTAT-security%5D)
with the `[uSTAT-security]` subject prefix. We acknowledge within 5 business
days.

---

## Code-execution sandbox

uSTAT optionally exposes a sandboxed Python runner — a "Code" tab that
accepts arbitrary Python and runs it against your session DataFrame
(injected as `df`). Off by default in production.

| Knob                            | Default | Description                              |
|---------------------------------|---------|------------------------------------------|
| `ENABLE_CODE_RUNNER`            | 0       | Set to 1 to expose the endpoint          |
| `SANDBOX_MEM_BYTES`             | 512 MB  | Address-space rlimit per run             |
| `SANDBOX_CPU_SEC`               | 30 s    | CPU rlimit (max 60 s)                    |
| `CODE_RUNNER_PER_MIN`           | 6       | Runs / minute / session                  |
| `CODE_RUNNER_PER_HOUR`          | 30      | Runs / hour / session                    |
| `CODE_RUNNER_IP_PER_MIN`        | 10      | Runs / minute / IP                       |
| `CODE_RUNNER_IP_PER_HOUR`       | 60      | Runs / hour / IP                         |
| `CODE_RUNNER_GLOBAL_PER_MIN`    | 30      | Runs / minute server-wide                |
| `CODE_RUNNER_MAX_CONCURRENT`    | 2       | Concurrent in-flight runs                |

Defense-in-depth: subprocess with `python -I -u`, `resource.setrlimit` for
CPU / memory / fds / nproc / fsize, `sys.meta_path` import allowlist
(numpy / pandas / scipy / statsmodels / lifelines / sklearn / matplotlib /
seaborn + safe stdlib) with an explicit deny-list for `socket`, `urllib*`,
`http*`, `subprocess`, `ctypes`, `os`, `shutil`, `pickle`, `importlib`, …
On Linux the wrapper additionally runs the child under
`unshare --user --net` when available, stripping the network namespace.

Read the full threat model in [`backend/SECURITY.md`](backend/SECURITY.md)
before enabling in production.

---

## Contributing

Pull requests welcome. Before opening one:

1. Run the linters and the local checks the CI workflow runs.
2. Update [`backend/SECURITY.md`](backend/SECURITY.md) if you touch the
   sandbox, auth, or rate-limit layers.
3. Add or update a test if you change a statistical method.
4. If you add a new statistical method, also add a row to the Tests & Methods
   table in [`AboutModal.tsx`](frontend/src/components/AboutModal.tsx) and a
   line to the methods table in this README.

---

## Citation

If you publish results obtained with uSTAT, please cite the tool:

> Hoşoğlu Y, Hoşoğlu A. *uSTAT — browser-based statistical analysis platform.*
> Version 3.4.0. Zenodo; 2026. doi:10.5281/zenodo.21610852

BibTeX:

```bibtex
@software{ustat,
  author    = {Hoşoğlu, Yusuf and Hoşoğlu, Ayşe},
  title     = {uSTAT — browser-based statistical analysis platform},
  version   = {3.4.0},
  doi       = {10.5281/zenodo.21610852},
  url       = {https://ustat.drtr.uk},
  publisher = {Zenodo},
  year      = {2026}
}
```

`10.5281/zenodo.21610852` is the concept DOI — it always resolves to the most
recent release. To cite the exact version you ran, use that release's own DOI
(v3.3.0 is `10.5281/zenodo.21610853`), which is listed on the Zenodo record.

---

## License & attribution

MIT License — © 2026 Dr. Yusuf Hoşoğlu. See [LICENSE](LICENSE).

Statistical computation builds on `scipy`, `statsmodels`, `lifelines`,
`scikit-learn`, `pandas`, `numpy`, `patsy`, `pyreadstat` (each under its own
license — see the respective project pages).
