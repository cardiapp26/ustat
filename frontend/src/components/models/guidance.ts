/** Static per-model copy shown in the Models panel. Extracted from
 *  ModelsPanel.tsx so the panel file holds logic, not prose. */

/** Default figure title when a model's estimates are sent to the Forest
 *  Builder — the penalty and the adjustment are part of what the figure
 *  claims, so they belong in the title rather than only in the methods. */
export const MODEL_FOREST_TITLE: Record<string, string> = {
  firth: "Firth penalized logistic regression",
  firth_ortable: "Firth penalized logistic regression",
  logistic: "Logistic regression",
  ortable: "Logistic regression",
  cox: "Cox proportional hazards",
  ordinal: "Ordinal logistic regression",
};

export const MODEL_GUIDANCE: Record<string, { use: string; check: string; interpret: string }> = {
  linear: {
    use: "Predict a continuous outcome from one or more predictors. Best for dose-response, biomarker prediction, and adjusted mean comparisons.",
    check: "Residuals vs Fitted should show no pattern (linearity). Q-Q plot should be roughly diagonal (normality). Scale-Location should be flat (homoscedasticity). Use Robust SE if heteroscedastic.",
    interpret: "Each coefficient = change in outcome per 1-unit increase in predictor, holding others constant. R² = proportion of variance explained. Check p-values and 95% CIs.",
  },
  logistic: {
    use: "Model a binary outcome (0/1) — e.g. death, readmission, disease presence. Returns Odds Ratios (OR) with 95% CI.",
    check: "Outcome must be binary 0/1. Check for multicollinearity (VIF > 5). Sample size rule of thumb: ≥ 10 events per predictor variable (EPV).",
    interpret: "OR > 1 = higher odds of outcome. OR < 1 = protective. OR = 1 = no effect. Report: OR (95% CI), p-value. Pseudo-R² is NOT comparable to linear R².",
  },
  ortable: {
    use: "Publication-standard univariate + multivariate OR table. Shows each predictor's effect both alone and adjusted for all others.",
    check: "Same as logistic. The forest plot visually compares unadjusted vs adjusted ORs — large shifts suggest confounding.",
    interpret: "Univariate OR = crude effect. Multivariate OR = adjusted effect. If they differ substantially, the variable is confounded by others in the model.",
  },
  poisson: {
    use: "Model count outcomes (0, 1, 2, 3...) — e.g. number of events, hospital visits, complications. Returns Incidence Rate Ratios (IRR).",
    check: "Outcome must be non-negative integers. Check the overdispersion warning (Pearson chi2/df well above 1): if variance >> mean, use Negative Binomial instead. Robust SE helps with mild overdispersion. If follow-up differs between rows, pick the follow-up time column as the exposure (offset).",
    interpret: "IRR > 1 = higher rate. IRR = 1.5 means 50% more events. With an exposure column the IRR is a rate ratio (events per unit of follow-up time). Report: IRR (95% CI), p-value.",
  },
  zip: {
    use: "Count outcome with more zeros than a Poisson model allows, where some zeros are 'structural' (the person could never have the event) and others are chance zeros. Fits a count part (IRR) and a zero-inflation part (OR of a structural zero).",
    check: "Outcome must be non-negative integers with at least one zero. Read the Vuong test: only keep the zero-inflated model if it is preferred over the standard Poisson. Compare observed and expected zeros. Use few zero-inflation predictors (or intercept only) to keep the fit stable.",
    interpret: "Count table: IRR is the rate ratio among rows that are not structural zeros. Zero-inflation table: OR above 1 means higher odds of a structural zero. Report both tables and the Vuong verdict.",
  },
  zinb: {
    use: "Zero-inflated counts that are also overdispersed (variance well above the mean among the non-structural rows). Adds a negative-binomial dispersion alpha to the zero-inflated Poisson model.",
    check: "Outcome must be non-negative integers with at least one zero. Check the Vuong test against the standard negative binomial and the alpha estimate: alpha near 0 means a zero-inflated Poisson would do. Keep the zero-inflation part small.",
    interpret: "Count table: IRR among non-structural rows. Zero-inflation table: OR of a structural zero. alpha is the dispersion (theta = 1/alpha). Report both tables, alpha and the Vuong verdict.",
  },
  km: {
    use: "Visualise time-to-event data. The survival curve shows the probability of surviving beyond each time point. Log-rank test compares curves between groups.",
    check: "Event column must be binary 0/1 (1 = event occurred). Duration must be positive. Censoring is assumed non-informative.",
    interpret: "Curves that separate early = strong effect. Log-rank p < 0.05 = significant difference between groups. Median survival = time at which 50% have had the event.",
  },
  cox: {
    use: "Regression for time-to-event data with multiple predictors. Returns Hazard Ratios (HR) — the multiplicative effect on the event rate.",
    check: "Proportional hazards assumption: HR should be constant over time (check Schoenfeld residuals). Event column must be binary 0/1.",
    interpret: "HR > 1 = higher hazard (worse prognosis). HR < 1 = protective. HR = 1 = no effect. Report: HR (95% CI), p-value.",
  },
  ordinal: {
    use: "Ordinal outcome with ≥3 ordered categories (e.g. NYHA I–IV, Killip, none/mild/severe). Proportional-odds model — one OR per predictor shared across the cumulative thresholds.",
    check: "Outcome must be an ordered categorical (mark it 'Ordered Categorical' in the Data tab). Proportional-odds assumption: each predictor's effect is constant across the category cut-points.",
    interpret: "OR > 1 = higher odds of being in a HIGHER category per unit increase. One OR per predictor (not one per category). Report OR (95% CI), p.",
  },
  multinomial: {
    use: "Unordered categorical outcome with 3 or more levels (cause of death, AF pattern, stent type), or an ordinal outcome whose proportional-odds assumption fails (Brant test).",
    check: "At least about 10 cases per estimated parameter in every category. Very large coefficients or SEs signal separation: merge sparse categories. The model assumes independence of irrelevant alternatives.",
    interpret: "One equation per category against the reference. RRR > 1 = higher odds of that category rather than the reference per unit increase. Judge each predictor overall by its likelihood-ratio test, then read the RRR rows.",
  },
  hrtable: {
    use: "Publication HR table (Table 3): each predictor's univariable HR, its parsimonious-model HR (a subset you tick), and its fully-adjusted HR (all predictors together) side by side.",
    check: "Event column must be binary 0/1, duration positive. Tick which predictors enter the parsimonious column. Categorical predictors expand to one row per level vs the reference.",
    interpret: "Univariable = crude effect. Parsimonious = adjusted for the chosen subset. Fully adjusted = adjusted for everything. A blank (—) cell means the predictor was not in that model.",
  },
  multi_outcome: {
    use: "Regress multiple continuous outcomes together on the same predictors/covariates. Produces one consolidated coefficient table (rows = predictors in model order, columns = outcomes).",
    check: "All outcomes must be continuous. Predictors and covariates are mutually exclusive with outcomes. Use listwise or imputation as needed.",
    interpret: "B = unstandardized coefficient. SE = standard error. β = standardized (when toggle on). 95% CI and p per outcome. Bottom model-fit per outcome.",
  },
  rcs: {
    use: "Model non-linear (U/J-shaped) dose-response relationships using Restricted Cubic Splines. Essential for continuous biomarkers where the effect is not a straight line.",
    check: "Predictor must have enough unique values (≥ knots + 2). Logistic RCS requires binary 0/1 outcome. 4 knots is the standard default.",
    interpret: "The dose-response curve shows how OR (or outcome) changes across the predictor range. Reference value = OR 1.0. Non-linearity p-value tests whether the curve is significantly non-linear.",
  },
};
