import {
  X, BookOpen, FlaskConical, Brain, Settings, ShieldCheck, HelpCircle, Code,
  CheckCircle2, Info, Target, Activity, LineChart, GitBranch, MonitorPlay,
} from "lucide-react";
import { useState } from "react";

type TabId =
  | "quickstart"
  | "tours"
  | "hypothesis"
  | "regression"
  | "causal"
  | "prediction"
  | "specialized"
  | "rhub";

export default function HelpModal({ onClose }: { onClose: () => void }) {
  const [activeTab, setActiveTab] = useState<TabId>("quickstart");

  const tabs = [
    { id: "quickstart",  label: "Quick Start",          icon: BookOpen },
    { id: "tours",       label: "Tab Tours",            icon: MonitorPlay },
    { id: "hypothesis",  label: "Hypothesis Tests",     icon: FlaskConical },
    { id: "regression",  label: "Regression & Survival",icon: LineChart },
    { id: "causal",      label: "Causal Inference",     icon: GitBranch },
    { id: "prediction",  label: "Prediction & Validation", icon: Target },
    { id: "specialized", label: "EFA, Bayes & Meta",    icon: Settings },
    { id: "rhub",        label: "R Replication Hub",    icon: Code },
  ] as const;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl flex flex-col h-[720px] max-h-[92vh] overflow-hidden animate-in fade-in zoom-in duration-200">

        {/* Header */}
        <div className="bg-slate-900 text-white px-6 py-4 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-2">
            <div className="bg-indigo-500 p-1.5 rounded-lg">
              <HelpCircle size={18} className="text-white" />
            </div>
            <div>
              <h2 className="font-bold text-sm tracking-tight">uSTAT Help &amp; Analysis Guide</h2>
              <p className="text-[10px] text-slate-400 font-mono">Clinical-biostatistics walkthrough &amp; method reference</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors p-1 hover:bg-slate-800 rounded-lg cursor-pointer"
          >
            <X size={18} />
          </button>
        </div>

        {/* Inner Content Area */}
        <div className="flex-1 flex min-h-0">

          {/* Sidebar Navigation */}
          <div className="w-60 bg-slate-50 border-r border-slate-200 flex flex-col p-3 gap-1 flex-shrink-0 overflow-y-auto">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider px-2.5 pb-2">
              Sections
            </p>
            {tabs.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-xs font-semibold transition-all text-left cursor-pointer ${
                  activeTab === id
                    ? "bg-indigo-600 text-white shadow-md shadow-indigo-100"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
                }`}
              >
                <Icon size={14} className={activeTab === id ? "text-white" : "text-slate-400"} />
                {label}
              </button>
            ))}

            <div className="mt-auto p-2 bg-indigo-50 rounded-xl border border-indigo-100">
              <p className="text-[10px] font-bold text-indigo-900 uppercase">💡 Tip</p>
              <p className="text-[9px] text-indigo-700 mt-0.5 leading-relaxed">
                Hover the <span className="font-semibold">ⓘ</span> / question-mark icons in any panel for detailed clinical hints. Use the
                <span className="font-semibold"> search bar</span> in the top toolbar to jump to any method by name (English or Turkish alias).
              </p>
            </div>
          </div>

          {/* Tab Panels */}
          <div className="flex-1 p-6 overflow-y-auto bg-white font-sans text-xs text-slate-700 space-y-4">

            {/* ─────────────────────────────────────────── QUICK START ─── */}
            {activeTab === "quickstart" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    🚀 Quick Start &amp; Data Workflow
                  </h3>
                  <p className="text-[10px] text-slate-500 mt-1">From file → cleaned dataset → reproducible analysis in five steps.</p>
                </div>

                <div className="space-y-3">
                  <Step n="1" title="Data upload">
                    Drag &amp; drop <code>.xlsx</code>, <code>.csv</code>, <code>.tsv</code>, or SPSS <code>.sav</code> files. SPSS value labels
                    and variable descriptions are read automatically. Multi-sheet Excel files prompt for a sheet picker. A
                    starter <span className="font-semibold">Sample dataset</span> is one click away on the upload zone for demos and tests.
                  </Step>

                  <Step n="2" title="Variable types &amp; data dictionary">
                    uSTAT auto-classifies each column as <span className="text-indigo-600 font-semibold">Numeric</span> (continuous),
                    <span className="text-teal-600 font-semibold"> Categorical</span>, <span className="font-semibold">Date</span>, or
                    <span className="font-semibold"> ID</span>. To override, click the badge under the column header in the Data tab or open the
                    <span className="font-semibold"> Data Dictionary</span>. Labels can be edited, recoded, and ordered (important for ordinal
                    variables and reference groups in regression).
                  </Step>

                  <Step n="3" title="Filters &amp; cohort locking">
                    Use the <span className="font-semibold">Filter</span> button to subset rows (e.g. age ≥ 18, diabetics only).
                    Once active, every test, plot, and model is restricted to that subset and a banner appears in the header.
                    PSM and IPTW additionally support <em>locking</em> the matched / weighted cohort so subsequent analyses use the balanced sample.
                  </Step>

                  <Step n="4" title="Compute &amp; cleaning">
                    Build new variables (formulas, transforms, recodes, clinical calculators such as BMI / eGFR / NEWS) in the
                    <span className="font-semibold"> Compute</span> tab. Data-quality tools — listwise deletion, IQR / Z-score outlier capping,
                    find &amp; replace, and visual missingness maps — live in the <span className="font-semibold">Missing</span> tab alongside
                    <span className="font-semibold"> MICE</span> multiple imputation (pooled estimates via Rubin's rules).
                  </Step>

                  <Step n="5" title="Save &amp; reproduce">
                    The header <span className="font-semibold">Save</span> menu exports the current dataset (xlsx / csv) and a portable
                    <span className="font-semibold"> session JSON</span> (variables, filters, dictionary, recent analyses). Re-open it later or
                    share it for review. See the <span className="font-semibold">R Replication Hub</span> tab on the left for the
                    auto-generated <code>.R</code> script.
                  </Step>
                </div>
              </div>
            )}

            {/* ────────────────────────────────────── TAB TOURS ──── */}
            {activeTab === "tours" && <TabTours />}

            {/* ─────────────────────────────────── HYPOTHESIS / TESTS ─── */}
            {activeTab === "hypothesis" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    🧪 Hypothesis &amp; Categorical Tests
                  </h3>
                  <p className="text-[10px] text-slate-500 mt-1">Continuous and categorical comparisons, post-hoc, multiplicity control.</p>
                </div>

                <div className="space-y-3">
                  <div className="flex gap-2 bg-indigo-50/50 p-3 rounded-xl border border-indigo-100/50">
                    <Info size={16} className="text-indigo-600 flex-shrink-0 mt-0.5" />
                    <p className="text-indigo-800 leading-relaxed text-[11px]">
                      <strong>Auto-mode parametric/non-parametric switch.</strong> For n ≤ 2000, normality is checked via Shapiro-Wilk; for n &gt; 2000,
                      via skewness + Lilliefors KS. Variance equality is checked via Levene. Based on the result, uSTAT runs a t-test/ANOVA
                      <em> or </em> Mann-Whitney/Kruskal-Wallis automatically and prints which path was chosen in the result text.
                    </p>
                  </div>

                  <Block title="Two-group comparisons" body={
                    <>Independent <em>t</em>-test (Welch by default; Student's when variances equal), Paired <em>t</em>-test for repeated
                    measures, Mann-Whitney U / Wilcoxon signed-rank for non-normal data. Effect sizes printed: Cohen's <em>d</em>,
                    Hedges' <em>g</em>, rank-biserial <em>r</em>. CI for the median difference (Hodges-Lehmann).</>
                  } />

                  <Block title="Multi-group comparisons &amp; post-hoc" body={
                    <>One-way ANOVA (Welch's option for unequal variances), Kruskal-Wallis, Brown-Forsythe. Post-hoc:
                    Tukey HSD, Bonferroni, Games-Howell (unequal variances), Dunn's test (non-parametric). Repeated-measures ANOVA with
                    Mauchly's sphericity check; Greenhouse-Geisser / Huynh-Feldt corrections applied automatically when violated.</>
                  } />

                  <Block title="Categorical tests" body={
                    <>Chi-square independence (with Yates / continuity correction), Fisher's exact (incl. Monte-Carlo for large tables),
                    Cochran-Armitage trend, McNemar / Bowker (paired binary), Stuart-Maxwell (paired multinomial),
                    Cochran's Q (k matched groups). Effect sizes: φ, Cramér's V, odds ratio with exact CI.</>
                  } />

                  <Block title="ANCOVA / MANCOVA" body={
                    <>Adjust group means for continuous covariates; check homogeneity of regression slopes (group × covariate term).
                    MANCOVA: Pillai / Wilks / Lawley-Hotelling / Roy statistics for multivariate outcome tests.</>
                  } />

                  <Block title="Multiplicity &amp; gatekeeping" body={
                    <>Family-wise control (Bonferroni, Holm, Hochberg, Hommel) and false-discovery (Benjamini-Hochberg, Benjamini-Yekutieli).
                    Hierarchical gatekeeping: assign primary / secondary endpoint families with weights to preserve overall α.</>
                  } />
                </div>
              </div>
            )}

            {/* ────────────────────────────── REGRESSION & SURVIVAL ─── */}
            {activeTab === "regression" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    📈 Regression &amp; Survival Models
                  </h3>
                  <p className="text-[10px] text-slate-500 mt-1">Linear / GLM / Cox / mixed / time-series — under <span className="font-semibold">Models</span> tab.</p>
                </div>

                <div className="space-y-3">
                  <Block title="Linear / GLM family" body={
                    <>Linear (OLS) with robust HC0–HC4 standard errors, Logistic (binary), Firth penalized logistic for separation /
                    rare events, Poisson and Negative Binomial for counts (IRR with offset support), Gamma GLM, Ordinal (proportional odds),
                    Multinomial. VIF / collinearity diagnostics, residual plots, influence (Cook's D, leverage), Box-Tidwell linearity check,
                    and an <span className="font-semibold">OR table (Uni + Multi)</span> for one-click univariable → multivariable workflow.</>
                  } />

                  <Block title="Restricted Cubic Splines (RCS)" body={
                    <>Model non-linear continuous predictors via Harrell-style RCS (3–7 knots). Produces dose-response plots with point-wise CIs
                    and likelihood-ratio test of overall non-linearity. Also available as <span className="font-semibold">Cox-RCS</span>
                    for hazard modelling.</>
                  } />

                  <Block title="Stepwise &amp; penalized selection" body={
                    <>Forward / backward / bidirectional stepwise by AIC, BIC, or p-value. Penalized regression (ridge / lasso / elastic-net)
                    with cross-validated λ. Use cautiously — see the Validation tab for honest internal performance.</>
                  } />

                  <Block title="Mixed effects &amp; GEE" body={
                    <>Linear and generalized linear mixed-effects (LMM / GLMM) with random intercepts &amp; slopes, REML / ML estimation, ICC
                    reporting. GEE for population-averaged models with exchangeable / AR(1) / unstructured working correlation.</>
                  } />

                  <Block title="Survival — KM, Cox, time-varying, RMST" body={
                    <>Kaplan-Meier curves with log-rank / Wilcoxon-Gehan, Cox PH with Schoenfeld residual PH test and Breslow / Efron ties,
                    Cox time-varying coefficients, frailty terms, stratification. Restricted Mean Survival Time (RMST) when PH is violated.
                    Fine-Gray subdistribution hazards for competing risks. Recurrent events (Andersen-Gill, LWYY robust).</>
                  } />

                  <Block title="Time series" body={
                    <>ARIMA / SARIMA forecasting with auto-order (AIC / BIC), STL seasonal decomposition, ADF / KPSS stationarity tests,
                    ACF / PACF plots, Ljung-Box residual diagnostics.</>
                  } />

                  <Block title="Machine learning" body={
                    <>Random Forest and Gradient Boosting (classification + regression) with stratified k-fold CV, OOF probability outputs,
                    permutation feature importance, partial-dependence plots, and an out-of-sample ROC for honest discrimination.</>
                  } />
                </div>
              </div>
            )}

            {/* ──────────────────────────────────── CAUSAL INFERENCE ─── */}
            {activeTab === "causal" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    🧠 Causal Inference Toolkit
                  </h3>
                  <p className="text-[10px] text-slate-500 mt-1">DAG-driven design + estimands beyond regression adjustment.</p>
                </div>

                <div className="space-y-3">
                  <div className="flex gap-2 bg-amber-50 p-3 rounded-xl border border-amber-100">
                    <Info size={16} className="text-amber-600 flex-shrink-0 mt-0.5" />
                    <p className="text-amber-800 leading-relaxed text-[11px]">
                      <strong>Start with the DAG.</strong> Draw confounders, mediators, moderators, and colliders on paper or in the
                      <span className="font-semibold"> DAG / Backdoor</span> panel. Adjust only for the backdoor set — never for mediators or colliders.
                    </p>
                  </div>

                  <Block title="Propensity Score Matching (PSM)" body={
                    <>1:1 or 1:N nearest-neighbour, caliper, optimal, and full matching on the logit propensity score. Diagnostics: Love plot,
                    standardized mean differences (SMD), variance ratios, common support density.
                    The <span className="font-semibold text-indigo-600">"View &amp; Analyze Matched Cohort"</span> button locks the whole app to the
                    matched pair list; the back-arrow returns to the original dataset.</>
                  } />

                  <Block title="Inverse Probability Weighting (IPTW)" body={
                    <>Weight each subject by 1/π̂ (ATE) or π̂/(1-π̂) (ATT). Stabilized + truncated weights to control extreme values.
                    Outputs balance diagnostics (SMD after weighting), effective sample size, and weighted regression-friendly weights you can lock
                    into the session.</>
                  } />

                  <Block title="Instrumental Variables (2SLS)" body={
                    <>Two-stage least squares with first-stage F-statistic, Sargan / Hansen over-identification, Wu-Hausman endogeneity test, and
                    weak-instrument warnings. Good for unmeasured-confounder scenarios when a valid instrument exists.</>
                  } />

                  <Block title="Causal Mediation" body={
                    <>Decompose total effect into ACME (indirect) and ADE (direct); proportion mediated and 95% percentile / BCa bootstrap CIs.
                    Sensitivity analysis (ρ) for unmeasured mediator-outcome confounders.</>
                  } />

                  <Block title="Target Trial Emulation" body={
                    <>Specify the protocol of a hypothetical RCT (eligibility, treatment strategies, time zero, follow-up, outcome, causal contrast),
                    then emulate with clone-censor-weight. Reduces immortal-time and prevalent-user biases.</>
                  } />

                  <Block title="Difference-in-Differences (DiD) &amp; RDD" body={
                    <>DiD with parallel-trends pre-period plot and event-study coefficients. Regression Discontinuity (sharp / fuzzy) with optimal
                    bandwidth (Imbens-Kalyanaraman / Calonico-Cattaneo-Titiunik) and McCrary density test for manipulation.</>
                  } />

                  <Block title="E-value &amp; sensitivity" body={
                    <>E-value for the minimum unmeasured-confounder strength needed to nullify the observed effect. Complements bias analyses
                    and tipping-point sensitivity.</>
                  } />
                </div>
              </div>
            )}

            {/* ─────────────────────────── PREDICTION & VALIDATION ─── */}
            {activeTab === "prediction" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    🎯 Prediction Models &amp; Validation
                  </h3>
                  <p className="text-[10px] text-slate-500 mt-1">Discrimination, calibration, clinical utility, and internal / external validation.</p>
                </div>

                <div className="space-y-3">
                  <div className="flex gap-2 bg-emerald-50 p-3 rounded-xl border border-emerald-100">
                    <Activity size={16} className="text-emerald-600 flex-shrink-0 mt-0.5" />
                    <p className="text-emerald-800 leading-relaxed text-[11px]">
                      <strong>Five questions a prediction model must answer:</strong>
                      &nbsp;(1) <em>discrimination</em> — does it separate cases from non-cases?
                      &nbsp;(2) <em>calibration</em> — do predicted risks match observed risks?
                      &nbsp;(3) <em>clinical utility</em> — would using it help patients?
                      &nbsp;(4) <em>overfitting</em> — does it work outside its training data?
                      &nbsp;(5) <em>transportability</em> — does it work in <em>another</em> cohort?
                    </p>
                  </div>

                  <Block title="ROC / AUC &amp; calibration" body={
                    <>ROC curves with DeLong CI for AUC, Youden J optimum threshold, calibration plot (deciles + smoothed), Brier score,
                    Hosmer-Lemeshow goodness-of-fit, calibration slope &amp; intercept, observed / expected (O/E) ratio.</>
                  } />

                  <Block title="Reclassification — NRI &amp; IDI" body={
                    <>Net Reclassification Improvement (categorical &amp; continuous), Integrated Discrimination Improvement, and the
                    <span className="font-semibold"> Added Predictive Value</span> panel for ΔAUC against a baseline model, with bootstrap CIs.</>
                  } />

                  <Block title="Decision Curve Analysis (DCA)" body={
                    <>Net benefit vs threshold probability against "treat all" / "treat none". Available for both binary and survival outcomes
                    (incl. integrated DCA over a fixed horizon).</>
                  } />

                  <div className="flex gap-2 bg-indigo-50 p-3 rounded-xl border border-indigo-200">
                    <Target size={16} className="text-indigo-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-indigo-900 text-[11px]">
                        Internal Validation — Models tab → <span className="underline">Validation (internal / external)</span> → <em>Internal</em>
                      </p>
                      <p className="text-indigo-800 leading-relaxed text-[10.5px] mt-1">
                        <strong>Harrell bootstrap optimism correction:</strong> each of <em>n_boot</em> resamples refits the model and scores the
                        original sample → reports <span className="font-semibold">apparent</span> AUC / C-index,
                        <span className="font-semibold"> optimism</span>, and <span className="font-semibold">optimism-corrected</span> performance
                        plus calibration slope. <strong>k-fold CV</strong> (StratifiedKFold logistic / KFold Cox, 5 or 10 folds) gives out-of-fold
                        AUC / C-index for cross-checking. The "overfitting gap" tile (apparent − corrected) flags shrinkage candidates.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2 bg-rose-50 p-3 rounded-xl border border-rose-200">
                    <Target size={16} className="text-rose-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-rose-900 text-[11px]">
                        External Validation — Models tab → <span className="underline">Validation</span> → <em>External (logistic)</em>
                      </p>
                      <p className="text-rose-800 leading-relaxed text-[10.5px] mt-1">
                        Load the validation cohort as the active dataset and provide its <span className="font-mono">prob_column</span>
                        &nbsp;(predicted probabilities from the development model applied to this cohort). The panel reports AUC + DeLong CI,
                        calibration slope / intercept, O/E, Hosmer-Lemeshow, Brier, decile calibration plot, and (optionally) the dev → val drop
                        when development AUC and slope are supplied.
                      </p>
                    </div>
                  </div>

                  <Block title="Model comparison" body={
                    <>Pair-wise DeLong AUC test, likelihood-ratio &amp; AIC / BIC comparisons, NRI / IDI, side-by-side calibration
                    against a benchmark model in the <span className="font-semibold">Model Compare</span> panel.</>
                  } />
                </div>
              </div>
            )}

            {/* ────────────────────────────── EFA, BAYES & META ──────── */}
            {activeTab === "specialized" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    ⚙️ EFA, Bayesian &amp; Meta-analysis
                  </h3>
                  <p className="text-[10px] text-slate-500 mt-1">Latent structure, evidence weighting, and pooling across studies.</p>
                </div>

                <div className="space-y-3">
                  <Block title="EFA &amp; PCA" body={
                    <>Suitability: KMO and Bartlett's sphericity. Extraction: Principal Axis, Maximum Likelihood, Principal Components.
                    Rotation: Varimax / Quartimax (orthogonal) or Promax / Oblimin (oblique). Outputs: factor loadings, communalities,
                    Scree plot with Kaiser / parallel-analysis cutoffs, biplot, Cronbach's α per factor.</>
                  } />

                  <Block title="Bayesian hypothesis tests" body={
                    <>JZS Bayes Factor for t-tests, ANOVA, correlation, and proportions; BF₁₀ / BF₀₁ with evidence categories,
                    prior (Cauchy) vs posterior density overlay, Savage-Dickey ratio, and robustness sweep over the prior scale <em>r</em>.</>
                  } />

                  <Block title="Meta-analysis" body={
                    <>Pool OR / RR / SMD / MD / proportions with fixed-effect, DerSimonian-Laird, Paule-Mandel, or Hartung-Knapp random-effects.
                    Heterogeneity: τ², I², Q, prediction interval. Forest plot, subgroup &amp; cumulative meta-analysis, sensitivity (leave-one-out).</>
                  } />

                  <Block title="Meta-regression &amp; bias" body={
                    <>Mixed-effects meta-regression on study-level moderators with R² (Knapp-Hartung). Publication bias: Egger's &amp; Begg's tests,
                    Trim-and-Fill, contour-enhanced funnel plot.</>
                  } />

                  <Block title="Network meta-analysis" body={
                    <>Frequentist NMA with consistency checking (node-split, side-split), league table, SUCRA rankings,
                    and forest plot vs a chosen reference comparator.</>
                  } />
                </div>
              </div>
            )}

            {/* ─────────────────────────────────────────── R HUB ─────── */}
            {activeTab === "rhub" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    💻 R Replication Hub &amp; Reporting
                  </h3>
                  <p className="text-[10px] text-slate-500 mt-1">Reproducibility, methods appendix, and audit trail.</p>
                </div>

                <div className="space-y-3 bg-indigo-950 text-indigo-100 p-4 rounded-2xl border border-indigo-900 shadow-lg">
                  <div className="flex gap-2.5 items-start">
                    <Info size={18} className="text-indigo-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <h4 className="font-bold text-sm text-white">One-to-one R replication script</h4>
                      <p className="text-xs text-indigo-300 mt-1 leading-relaxed">
                        Every filter, recode, imputation, test, and model you run is recorded chronologically. The R Replication Hub
                        compiles them into a clean, commented <span className="font-mono text-white bg-indigo-900/60 px-1 py-0.5 rounded">.R</span>
                        &nbsp;script — package <code>library()</code> calls, seed pinning, then step-by-step calls to
                        &nbsp;<code>stats</code>, <code>survival</code>, <code>rms</code>, <code>lme4</code>, <code>metafor</code>,
                        &nbsp;<code>MatchIt</code>, <code>WeightIt</code>, <code>geepack</code>, and friends — so RStudio replays the same numbers.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="space-y-3 pt-2">
                  <Block title="Methods appendix (DOCX)" body={
                    <>Alongside the script, generate an academic Word document for your paper's Methods section — listing software
                    versions, random seeds, sample size, missing-data handling, and one paragraph per analysis with the appropriate
                    citation and the parameters actually used.</>
                  } />

                  <Block title="Per-result exports" body={
                    <>Each panel offers PNG / SVG / TIFF / PDF chart export with publication-grade themes (white background, vector fonts) and
                    a CSV / XLSX export of the result table. Result Exporter combines tables + figures into a single docx.</>
                  } />

                  <Block title="Session JSON &amp; resume" body={
                    <>Save the full session (dataset, dictionary, filters, recent analyses) as JSON. Re-load any time to continue,
                    or share for peer review / audit.</>
                  } />
                </div>
              </div>
            )}

          </div>

        </div>

        {/* Footer */}
        <div className="bg-slate-50 border-t border-slate-200 px-6 py-3 flex items-center justify-between flex-shrink-0 text-[10px] text-slate-500 font-medium">
          <div className="flex items-center gap-1.5">
            <ShieldCheck size={13} className="text-emerald-600" />
            <span>Data stays in your local browser session; only anonymized vectors are sent to the compute service.</span>
          </div>
          <span className="font-semibold text-slate-400">uSTAT v3.3.0 Guide</span>
        </div>

      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// Small presentational helpers
// ─────────────────────────────────────────────────────────────────────

function Step({ n, title, children }: { n: string; title: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-2">
      <div className="flex-shrink-0 w-6 h-6 rounded-full bg-indigo-600 text-white text-[11px] font-bold flex items-center justify-center shadow-sm">
        {n}
      </div>
      <div>
        <p className="font-bold text-slate-800">{title}</p>
        <p className="text-slate-500 leading-relaxed mt-0.5 text-[11px]">{children}</p>
      </div>
    </div>
  );
}

function Block({ title, body }: { title: string; body: React.ReactNode }) {
  return (
    <div className="flex gap-2">
      <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
      <div>
        <p className="font-bold text-slate-800">{title}</p>
        <p className="text-slate-500 leading-relaxed mt-0.5 text-[11px]">{body}</p>
      </div>
    </div>
  );
}

// Silence unused-icon imports (kept in case theming changes need them).
void Brain;

// ─── Tab Tours: per-app-tab screenshot + step-by-step how-to ──────────

interface TourTab {
  id: string;
  label: string;
  img: string;
  blurb: string;
  steps: string[];
  read?: string;
}

const TOUR_TABS: TourTab[] = [
  {
    id: "data",
    label: "Data",
    img: "/manual/01-data.png",
    blurb: "Load, type, and clean your dataset — the foundation every other tab reads.",
    steps: [
      "Upload is automatic — drag your CSV/XLSX/SAV onto the dropzone, or click Browse. The Data tab opens by default.",
      "Type each variable: in any column header, click the small badge (num / cat / txt / date) to cycle to the correct type. Do this for every column.",
      "Add labels (optional): click a column name → Dictionary to set a label and value labels (0 → No, 1 → Yes).",
      "Check missingness: each header shows a red badge like '103✕ · 13%'. Click ⚠ Missing to see only rows with gaps.",
      "Clean if needed: use Select Cases (rule-based row filter) or per-column Filter. +Row/+Column to add; double-click a cell to edit in place.",
    ],
    read: "Type and clean here first — a mistyped variable (e.g. a number as text) silently disappears from downstream analyses.",
  },
  {
    id: "summary",
    label: "Summary",
    img: "/manual/02-summary.png",
    blurb: "Descriptive statistics and distribution plots. Use it to judge normality before picking a test.",
    steps: [
      "Go to Summary → Descriptive (Weighted sub-tab for survey weights).",
      "Pick one numeric variable from the Variable dropdown.",
      "Read the stat table: n, mean, SD, median, IQR, range, skew, kurtosis, Shapiro-Wilk p.",
      "Switch the plot with the buttons above: Histogram, Boxplot, Violin, Q-Q.",
      "Optional: choose a categorical Group by to split distributions side-by-side.",
    ],
    read: "Use the Q-Q plot to choose your test: points on the line ≈ normal → parametric (t-test/ANOVA); strong S-curve → non-parametric (Mann-Whitney).",
  },
  {
    id: "table1",
    label: "Table",
    img: "/manual/03-table1.png",
    blurb: 'Clinical baseline "Table 1" by group, with auto-selected summary + test per row.',
    steps: [
      "Go to the Table tab.",
      "Pick the Grouping variable (e.g. treatment arm). Leave blank for a single-arm table.",
      "Tick the baseline variables you want in the left list.",
      "uSTAT auto-selects mean±SD / median[IQR] / n(%) and the correct test (t / Mann-Whitney / χ² / Fisher) per row, with a p-value column.",
      "Click Export for a publication-ready Word/CSV table.",
    ],
  },
  {
    id: "tests",
    label: "Tests",
    img: "/manual/04-tests.png",
    blurb: "Classic hypothesis tests: t, ANOVA, Mann-Whitney, Kruskal, χ², Fisher, McNemar, reliability, Bayesian.",
    steps: [
      "Go to the Tests tab and pick the right sub-tab (Hypothesis / Repeated Measures / Categorical / Reliability / Non-Inferiority / Gatekeeping / Factor / Bayesian).",
      "In the left list, click the test name you want.",
      "In the form, assign the variables: Outcome (+ Group / Paired column / Strata as needed).",
      "Read the When-to-use / Assumptions / How-to-read card on the right.",
      "Click Run → statistic, p-value, effect size, plain-English conclusion.",
    ],
    read: "Parametric vs non-parametric: check the Q-Q in Summary first. Normal → t/ANOVA; not normal → Mann-Whitney/Kruskal.",
  },
  {
    id: "correlation",
    label: "Correlation",
    img: "/manual/05-correlation.png",
    blurb: "Pearson / Spearman / Kendall correlation matrix with p-values and heatmap.",
    steps: [
      "Go to the Correlation tab.",
      "Tick ≥2 numeric or ordinal variables in the left list.",
      "Pick the method: Pearson (linear, normal), Spearman (monotonic/ranked), or Kendall (robust, small samples).",
      "Click Compute → coefficient matrix with p-values + color-coded heatmap.",
      "Click any cell for the scatter with a fitted line.",
    ],
    read: "|r| < 0.3 weak, 0.3–0.6 moderate, > 0.6 strong. p < 0.05 says it's non-zero, not that it's strong.",
  },
  {
    id: "roc",
    label: "ROC",
    img: "/manual/06-roc.png",
    blurb: "ROC curve, AUC, DeLong comparison, combined multi-marker model.",
    steps: [
      "Go to the ROC tab.",
      "Choose a numeric Score/marker and a binary Outcome (0/1).",
      "Click Compute → AUC with 95% CI, optimal cutoff (Youden J), sensitivity/specificity.",
      "Multi-curve: tick several scores to overlay. DeLong tests whether two AUCs differ.",
      "Combined model: fits a logistic combination of markers → combined AUC.",
    ],
    read: "AUC 0.5 = none, 0.7 = acceptable, 0.8 = good, 0.9 = excellent. If CI crosses 0.5, the marker is not informative.",
  },
  {
    id: "models-reg",
    label: "Models · Regression",
    img: "/manual/07-models-regression.png",
    blurb: "Linear, logistic, Firth, Poisson, gamma, ordinal, polynomial — full coefficient tables.",
    steps: [
      "Go to Models → Regression.",
      "Pick the model matching your outcome (Linear / Logistic / Firth / Poisson / NegBinom / Gamma / Ordinal).",
      "Choose the Outcome, then tick the Predictors.",
      "Optional: Robust SE, imputation, interactions, scale factors.",
      "Click Fit → coefficient table (β / OR / IRR) with 95% CI + p, model-fit (R², AIC), plain-English summary.",
    ],
    read: "Ordinal logistic: check the Brant test — green = proportional-odds holds, amber = violated (offending predictors named).",
  },
  {
    id: "models-surv",
    label: "Models · Survival",
    img: "/manual/08-models-survival.png",
    blurb: "Kaplan-Meier, Cox PH, time-horizon HR, Fine-Gray, RMST, interval-censored.",
    steps: [
      "Go to Models → Survival Advanced. Left list holds the methods.",
      "Common inputs: numeric Duration (time) + binary Event (0/1) + optional Group/Stratify.",
      "Kaplan-Meier: Duration + Event + Group → curves with log-rank p.",
      "Cox PH: add predictors → HR table with 95% CI + p + PH assumption test.",
      "Interval-censored: pick lower + upper bound columns + covariates → Turnbull curve + Weibull HR.",
    ],
  },
  {
    id: "psm",
    label: "PSM",
    img: "/manual/09-psm.png",
    blurb: "Propensity-score matching: balance confounders, estimate the matched treatment effect.",
    steps: [
      "Go to the PSM tab.",
      "Choose the Treatment variable (binary 0/1).",
      "Tick the Covariates to balance on.",
      "Choose the Outcome + caliper (0.2 default) + matching ratio.",
      "Run → balance diagnostics (SMDs, Love plot) + matched ATT with CI.",
    ],
    read: "Post-match SMDs all < 0.1 → balanced → estimate is trustworthy. Otherwise tighten the caliper.",
  },
  {
    id: "iptw",
    label: "IPTW",
    img: "/manual/10-iptw.png",
    blurb: "Inverse-probability-of-treatment weighting — the ATE without discarding subjects.",
    steps: [
      "Go to the IPTW tab. Same inputs as PSM: Treatment + Covariates + Outcome.",
      "Pick the estimand: ATE (whole population), ATT (treated only), or overlap.",
      "Optional: Stabilize weights + truncation if tails are extreme.",
      "Run → propensity model, weight distribution (ESS), balance after weighting, weighted effect.",
    ],
    read: "Use IPTW over PSM when you don't want to discard unmatched subjects, or want the ATE rather than the ATT.",
  },
  {
    id: "causal",
    label: "Causal+",
    img: "/manual/11-causal.png",
    blurb: "IV/2SLS, mediation, SEM / Path analysis, target-trial emulation, DiD, RDD, DAG backdoor.",
    steps: [
      "Go to the Causal+ tab and pick a method: IV/2SLS, Mediation, SEM / Path analysis, Target-trial, DiD, RDD, DAG.",
      "Assign the method-specific roles (e.g. IV needs outcome + endogenous exposure + instrument(s); Mediation needs treatment + mediator + outcome; SEM needs ≥1 treatments + mediators + outcomes).",
      "Click Run → estimate, identifying assumption, sensitivity check.",
    ],
  },
  {
    id: "sem",
    label: "SEM / Path analysis",
    img: "/manual/11-causal.png", // TODO: SEM-specific screenshot (orchestrator will swap)
    blurb: "SEM / path analysis: multi-treatment, parallel or serial mediators, multi-outcome; lavaan models + bootstrap CIs + fit indices.",
    steps: [
      "Go to Causal+ → SEM / Path analysis. Pick ≥1 Treatments, ≥1 Mediators (enable Serial chain for chains), ≥1 continuous Outcomes (+ optional covariates).",
      "Set bootstrap (default 5000) or paste custom lavaan spec in Advanced. Run.",
      "Inspect indirect effects table (95% bootstrap CI excluding 0 = significant); check fit: CFI/TLI ≥0.95, RMSEA ≤0.06, SRMR ≤0.08.",
    ],
  },
  {
    id: "dca",
    label: "DCA",
    img: "/manual/12-dca.png",
    blurb: "Decision-curve analysis — net benefit across threshold probabilities.",
    steps: [
      "Go to the DCA tab.",
      "Provide predicted risks directly, or give predictors + binary outcome (uSTAT fits logistic).",
      "Optional: set threshold probability range (default 1–99%).",
      "Compute → net-benefit curve vs treat-all / treat-none.",
    ],
    read: "The model is clinically useful over a threshold range where its net benefit beats both treat-all and treat-none.",
  },
  {
    id: "meta",
    label: "Meta",
    img: "/manual/13-meta.png",
    blurb: "Random/fixed-effects meta-analysis, forest plot, subgroup, meta-regression, bias.",
    steps: [
      "Go to the Meta tab.",
      "Enter one row per study: effect + 95% CI (or +SE), or a 2×2 table (e1,n1,e2,n2).",
      "Pick the measure (OR / RR / RD / SMD / MD).",
      "Analyze → pooled estimate (DerSimonian-Laird), forest plot, I²/τ²/Q.",
      "Subgroup: add a subgroup label per study. Regression: add a numeric moderator. Bias: Egger, Begg, funnel, trim-and-fill.",
    ],
  },
  {
    id: "missing",
    label: "Missing",
    img: "/manual/14-missing.png",
    blurb: "Missingness audit, Little's MCAR test, MICE imputation, MNAR sensitivity.",
    steps: [
      "Go to the Missing tab.",
      "The pattern view shows a heatmap of which cells are missing + per-variable counts.",
      "Run Little's MCAR test: p > 0.05 → MCAR (simple imputation OK); p < 0.05 → MAR (use MICE).",
      "Impute with MICE: choose columns + number of imputations (default 5). Downstream models pool via Rubin's rules.",
      "Compare strategies side-by-side: mean / median / MICE / listwise.",
    ],
  },
  {
    id: "visual",
    label: "Visual",
    img: "/manual/15-visual.png",
    blurb: "Model diagnostics, custom charts, subgroup bar, forest builder, added predictive value.",
    steps: [
      "Models & Diagnostics — pick a fitted model → residual, Q-Q, leverage, VIF plots.",
      "Charts — build a custom plot (Histogram/Box/Violin/Scatter/Bar/Line) from scratch.",
      "Subgroup Bar — compare a mean+CI or % outcome across two factors.",
      "Forest plot — paste rows (label, estimate, CI, weight) or load from a model. Toggle log/linear, sort, add group sub-headings, run built-in meta-analysis for a pooled diamond.",
      "Added Predictive Value — base vs base+marker: ΔAUC, NRI, IDI.",
    ],
  },
  {
    id: "power",
    label: "Power",
    img: "/manual/16-power.png",
    blurb: "Power / sample-size calculator (no data upload needed).",
    steps: [
      "Go to the Power tab (also on the splash screen — no data needed).",
      "Pick the test: t-test, ANOVA, correlation, proportions, χ², logistic, Cox.",
      "Choose solve for: sample size / power / minimum detectable effect.",
      "Enter α, power, effect size + test-specific fields (logistic: OR + event prevalence; Cox: HR + event rate + exposed fraction).",
      "Read the result + power curve (how power changes with n).",
    ],
    read: "Use this for protocol / grant sample-size justification.",
  },
];

function TabTours() {
  const [activeTour, setActiveTour] = useState(TOUR_TABS[0].id);
  const tour = TOUR_TABS.find((t) => t.id === activeTour) ?? TOUR_TABS[0];

  return (
    <div className="flex flex-col h-full">
      <div className="border-b pb-2 mb-3">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
          🖼 Tab Tours — screenshot + how-to for every tab
        </h3>
        <p className="text-[10px] text-slate-500 mt-1">
          Pick a tab on the left, see its screenshot and the exact click-by-click steps.
        </p>
      </div>
      <div className="flex gap-3 flex-1 min-h-0">
        {/* Left: tab selector */}
        <div className="w-32 flex-shrink-0 overflow-y-auto pr-1 space-y-0.5">
          {TOUR_TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setActiveTour(t.id)}
              className={`w-full text-left text-[11px] px-2 py-1.5 rounded-lg transition-colors ${
                activeTour === t.id
                  ? "bg-indigo-600 text-white font-semibold"
                  : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        {/* Right: screenshot + steps */}
        <div className="flex-1 min-w-0 overflow-y-auto pr-1">
          <div key={tour.id} className="space-y-3">
            <div>
              <p className="text-[11px] text-slate-600">{tour.blurb}</p>
            </div>
            <img
              src={tour.img}
              alt={`${tour.label} tab`}
              className="w-full rounded-lg border border-slate-200 shadow-sm"
              loading="lazy"
            />
            <div className="space-y-2">
              <p className="text-[10px] font-bold uppercase tracking-wider text-indigo-600">
                How to do it
              </p>
              {tour.steps.map((step, i) => (
                <div key={i} className="flex gap-2">
                  <div className="flex-shrink-0 w-5 h-5 rounded-full bg-indigo-600 text-white text-[9px] font-bold flex items-center justify-center">
                    {i + 1}
                  </div>
                  <p className="text-[11px] text-slate-600 leading-relaxed">{step}</p>
                </div>
              ))}
            </div>
            {tour.read && (
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-2.5 mt-2">
                <p className="text-[10px] text-amber-800 leading-relaxed">
                  <span className="font-bold">💡 Read it: </span>
                  {tour.read}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
