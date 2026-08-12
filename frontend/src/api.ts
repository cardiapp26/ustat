import axios from "axios";
import { runColumnStructureMutation } from "./lib/columnStructureLock";

const api = axios.create({ baseURL: "" });  // Vite proxy: /api → localhost:8000

export default api;

export const uploadFile = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/api/upload/", form);
};

export const getDescriptive = (sessionId: string, column?: string) =>
  api.get(`/api/stats/${sessionId}/descriptive`, { params: column ? { column } : {} });

export const getFrequency = (sessionId: string, column: string) =>
  api.get(`/api/stats/${sessionId}/frequency`, { params: { column } });

export const getCorrelation = (sessionId: string, method = "pearson") =>
  api.get(`/api/stats/${sessionId}/correlation`, { params: { method } });

export interface TTestRequest {
  session_id: string;
  column: string;
  group_column?: string;
  mu?: number;
  method?: "auto" | "student" | "welch";
  equal_var?: boolean;
}

export interface ChiSquareRequest {
  session_id: string;
  row_column: string;
  col_column: string;
}

export interface FisherRequest {
  session_id: string;
  row_column: string;
  col_column: string;
}

export interface AnovaRequest {
  session_id: string;
  column: string;
  group_column: string;
}

export interface MannWhitneyRequest {
  session_id: string;
  column: string;
  group_column: string;
}

export interface KruskalRequest extends MannWhitneyRequest {
  posthoc_correction?: "holm" | "bonferroni" | "fdr" | "none";
}

export interface JonckheereRequest extends MannWhitneyRequest {
  scores?: number[];
  alpha?: number;
}

export interface ROCRequest {
  session_id: string;
  score_column: string;
  outcome_column: string;
  direction?: "auto" | "higher" | "lower";
  manual_cutoff?: number;
  imputation?: string;
  stratify_by?: string;
  stratify_values?: unknown[];
}

export interface ROCCompareRequest {
  session_id: string;
  score_column_1: string;
  score_column_2: string;
  outcome_column: string;
  direction_1?: "auto" | "higher" | "lower";
  direction_2?: "auto" | "higher" | "lower";
}

export interface ROCMultiCompareRequest {
  session_id: string;
  score_columns: string[];
  outcome_column: string;
  directions?: Array<"auto" | "higher" | "lower">;
  p_adjust?: "holm" | "bonferroni" | "none";
}

export interface ROCCombinedRequest {
  session_id: string;
  predictor_columns: string[];
  outcome_column: string;
  model_name?: string;
}

export const runTTest = (data: TTestRequest) => api.post("/api/stats/ttest", data);
export const runChiSquare = (data: ChiSquareRequest) => api.post("/api/stats/chisquare", data);
export const runAnova = (data: AnovaRequest) => api.post("/api/stats/anova", data);
export const runMannWhitney = (data: MannWhitneyRequest) => api.post("/api/stats/mannwhitney", data);
export const runFisher = (data: FisherRequest) => api.post("/api/stats/fisher", data);
export const runKruskal = (data: KruskalRequest) => api.post("/api/stats/kruskal", data);
export const runJonckheereTerpstra = (data: JonckheereRequest) => api.post("/api/stats/jonckheere_terpstra", data);
export const runROC = (data: ROCRequest) => api.post("/api/stats/roc", data);
export const runROCCompare = (data: ROCCompareRequest) => api.post("/api/stats/roc_compare", data);
export const runROCMultiCompare = (data: ROCMultiCompareRequest) => api.post("/api/stats/roc_multi_compare", data);
export const runROCCombined = (data: ROCCombinedRequest) => api.post("/api/stats/roc_combined", data);

export const getHistogram = (data: object) => api.post("/api/charts/histogram", data);
export const getScatter = (data: object) => api.post("/api/charts/scatter", data);
export const getBoxplot = (data: object) => api.post("/api/charts/boxplot", data);
export const getPairedBox = (data: object) => api.post("/api/charts/paired_box", data);
export const getDumbbell = (data: object) => api.post("/api/charts/dumbbell", data);
export const getCompareMeans = (data: object) => api.post("/api/charts/compare_means", data);
export const getErrorPlot = (data: object) => api.post("/api/charts/errorplot", data);
export const getEcdf = (data: object) => api.post("/api/charts/ecdf", data);
export const getPie = (data: object) => api.post("/api/charts/pie", data);
export const getBalloon = (data: object) => api.post("/api/charts/balloon", data);
export const getSummaryStats = (data: object) => api.post("/api/charts/summary_stats", data);
export const getFacet = (data: object) => api.post("/api/charts/facet", data);
export const getLinePlot = (data: object) => api.post("/api/charts/lineplot", data);
export const getSlopePlot = (data: object) => api.post("/api/charts/slopeplot", data);
export const getSankey = (data: object) => api.post("/api/charts/sankey", data);
export const getStackPlot = (data: object) => api.post("/api/charts/stackplot", data);
export const getRidgePlot = (data: object) => api.post("/api/charts/ridgeplot", data);
export const getSets = (data: object) => api.post("/api/charts/sets", data);
export const getBar = (data: object) => api.post("/api/charts/bar", data);
export const runSubgroupBar = (data: object) => api.post("/api/charts/subgroup_bar", data);
export const runScoreComposite = (data: object) => api.post("/api/charts/score_composite", data);
export const runKMComposite = (data: object) => api.post("/api/charts/km_composite", data);


export const runLinear   = (data: object) => api.post("/api/models/linear", data);
export const runMultiOutcomeRegression = (data: object) => api.post("/api/models/multi_outcome_regression", data);
export const runRCS      = (data: object) => api.post("/api/models/rcs", data);
export const runLogistic = (data: object) => api.post("/api/models/logistic", data);
export const runFirthLogistic = (data: object) => api.post("/api/models/firth_logistic", data);
export const runLogisticTable = (data: object) => api.post("/api/models/logistic_table", data);
export const runPoisson  = (data: object) => api.post("/api/models/poisson", data);
export const runKM = (data: object) => api.post("/api/models/survival/km", data);
export const runCox = (data: object) => api.post("/api/models/survival/cox", data);
export const runCoxRCS = (data: object) => api.post("/api/models/survival/cox_rcs", data);
export const runCoxHorizons = (data: object) => api.post("/api/models/survival/cox_horizons", data);
export const runCoxUniMulti = (data: object) => api.post("/api/models/survival/cox_uni_multi", data);
export const runCoxModelSpecs = (data: object) => api.post("/api/models/survival/cox_model_specs", data);
export const runPolynomial  = (data: object) => api.post("/api/models/polynomial", data);
export const runLMM         = (data: object) => api.post("/api/models/lmm", data);
export const runGamma       = (data: object) => api.post("/api/models/gamma", data);
export const runNegBinom    = (data: object) => api.post("/api/models/negbinom", data);
export const runLinearDiag  = (data: object) => api.post("/api/models/linear_diag", data);
export const runMelt          = (data: object) => api.post("/api/models/melt", data);

// Machine learning (predictive modeling)
export const runRandomForest      = (data: object) => api.post("/api/ml/random_forest", data);
export const runGradientBoosting  = (data: object) => api.post("/api/ml/gradient_boosting", data);
export const runFeatureImportance = (data: object) => api.post("/api/ml/feature_importance", data);
export const runMLSurvivalBenchmark = (data: object) => api.post("/api/survival_advanced/ml_survival_benchmark", data);

// Time series
export const runArima        = (data: object) => api.post("/api/timeseries/arima", data);
export const runDecompose    = (data: object) => api.post("/api/timeseries/decompose", data);
export const runStationarity = (data: object) => api.post("/api/timeseries/stationarity", data);

// Weighted descriptives (survey weights)
export const runWeightedDescriptive = (data: object) => api.post("/api/stats/weighted_descriptive", data);

// Non-inferiority / margin testing
export interface NonInferiorityRequest {
  session_id: string;
  outcome_col: string;
  group_col: string;
  test_group?: string;
  ref_group?: string;
  outcome_type?: "binary" | "continuous";
  effect?: "RR" | "RD" | "OR";
  margin?: number;
  bound?: "upper" | "lower";
  alpha?: number;
  imputation?: string;
}

export const runNonInferiority = (data: NonInferiorityRequest) =>
  api.post("/api/stats/noninferiority", data);

// Multiplicity / gatekeeping
export const runGatekeeping = (data: object) => api.post("/api/multiplicity/gatekeeping", data);

// Meta-analysis (study-level)
export const runMetaAnalyze    = (data: object) => api.post("/api/meta/analyze", data);
export const runMetaSubgroup   = (data: object) => api.post("/api/meta/subgroup", data);
export const runMetaRegression = (data: object) => api.post("/api/meta/regression", data);
export const runMetaBias       = (data: object) => api.post("/api/meta/bias", data);
export const refreshSession   = (sessionId: string) => api.get(`/api/stats/${sessionId}/refresh`);
export const runPSM           = (data: object) => api.post("/api/models/psm", data);
export const runIPTW          = (data: object) => api.post("/api/models/iptw", data);

export const getSparklines = (sessionId: string) =>
  api.get(`/api/stats/${sessionId}/sparklines`);

export const getColumnBadges = (sessionId: string) =>
  api.get(`/api/stats/${sessionId}/column_badges`);

export const getRawColumns = (sessionId: string, columns: string[]) =>
  api.get(`/api/stats/${sessionId}/raw`, { params: { columns: columns.join(",") } });

export const getMissing = (sessionId: string, columns: string[]) =>
  api.get(`/api/stats/${sessionId}/missing`, { params: { columns: columns.join(",") } });

// ── Compute / Create New Variable ──────────────────────────────────────────
export const computeFormula = (sessionId: string, data: object) =>
  runColumnStructureMutation(
    sessionId,
    () => api.post(`/api/compute/${sessionId}/formula`, data),
  );
export const computeTransform = (sessionId: string, data: object) =>
  runColumnStructureMutation(
    sessionId,
    () => api.post(`/api/compute/${sessionId}/transform`, data),
  );
export const computeRecode = (sessionId: string, data: object) =>
  runColumnStructureMutation(
    sessionId,
    () => api.post(`/api/compute/${sessionId}/recode`, data),
  );
export const computeClinical = (sessionId: string, calc: string, data: object) =>
  runColumnStructureMutation(
    sessionId,
    () => api.post(`/api/compute/${sessionId}/clinical/${calc}`, data),
  );
export const deleteColumn      = (sessionId: string, col: string) => api.delete(`/api/compute/${sessionId}/column/${encodeURIComponent(col)}`);
export const getUniqueValues   = (sessionId: string, col: string) => api.get(`/api/compute/${sessionId}/unique/${encodeURIComponent(col)}`);

export const runCorrelationPair = (data: object) => api.post("/api/stats/correlation_pair", data);
export const runCorrelationMatrix = (data: object) => api.post("/api/stats/correlation_matrix", data);
export interface ICCRequest {
  session_id: string;
  rater1_col: string;
  rater2_col: string;
}

export interface KappaRequest {
  session_id: string;
  rater1_col: string;
  rater2_col: string;
}

export interface FleissKappaRequest {
  session_id: string;
  rater_cols: string[];
}

export const runICC = (data: ICCRequest) => api.post("/api/stats/icc", data);
export const runCohensKappa = (data: KappaRequest) => api.post("/api/stats/cohens_kappa", data);
export const runFleissKappa = (data: FleissKappaRequest) => api.post("/api/stats/fleiss_kappa", data);
export const runPower       = (data: object) => api.post("/api/stats/power", data);
export const runHosmerLemeshow = (data: object) => api.post("/api/decision_curve/hosmer_lemeshow", data);
export interface TOSTRequest {
  session_id: string;
  column: string;
  group_column?: string;
  paired_column?: string;
  low: number;
  high: number;
  mu?: number;
  test_type?: "independent" | "paired" | "one_sample";
}

export const runTOST = (data: TOSTRequest) => api.post("/api/stats/tost", data);
export const runGEE            = (data: object) => api.post("/api/models/gee", data);
export const runOrdinal        = (data: object) => api.post("/api/models/ordinal", data);
export const runCoxTV          = (data: object) => api.post("/api/models/survival/cox_tv", data);
export const runStepwise       = (data: object) => api.post("/api/models/stepwise", data);
export const runForest         = (data: object) => api.post("/api/charts/forest", data);
export const downloadMethodAppendix = (sessionId: string, title?: string) =>
  api.post("/api/pub_export/method_appendix", { session_id: sessionId, title: title ?? "Statistical Methods" }, { responseType: "blob" });

// Repeated measures
export const runPairedTTest  = (data: object) => api.post("/api/repeated/paired_ttest", data);
export const runWilcoxonSR   = (data: object) => api.post("/api/repeated/wilcoxon_signed_rank", data);
export const runFriedman     = (data: object) => api.post("/api/repeated/friedman", data);
export const runRMAnova      = (data: object) => api.post("/api/repeated/rm_anova", data);
export const runMixedAnova   = (data: object) => api.post("/api/repeated/mixed_anova", data);

// Advanced ANOVA
export const runAncova       = (data: object) => api.post("/api/advanced_anova/ancova", data);
export const runTwoWayAnova  = (data: object) => api.post("/api/advanced_anova/two_way_anova", data);
export const runMancova      = (data: object) => api.post("/api/advanced_anova/mancova", data);

// Categorical
export const runBinomial     = (data: object) => api.post("/api/categorical/binomial", data);
export const runOneProportion = (data: object) => api.post("/api/categorical/one_proportion", data);
export const runTwoProportions = (data: object) => api.post("/api/categorical/two_proportions", data);
export const runMcNemar      = (data: object) => api.post("/api/categorical/mcnemar", data);
export const runCochranQ     = (data: object) => api.post("/api/categorical/cochran_q", data);
export const runMantelHaenszel = (data: object) => api.post("/api/categorical/mantel_haenszel", data);
export const runCochranArmitage = (data: object) => api.post("/api/categorical/cochran_armitage", data);

// Agreement
export const runBlandAltman  = (data: object) => api.post("/api/agreement/bland_altman", data);
export const runDeming       = (data: object) => api.post("/api/agreement/deming", data);
export const runPassingBablok = (data: object) => api.post("/api/agreement/passing_bablok", data);
export const runConcordance  = (data: object) => api.post("/api/agreement/concordance", data);

// Reliability
export const runCronbach     = (data: object) => api.post("/api/reliability/cronbach", data);

// Progressive adjustment (crude / model 1 / model 2 …)
export interface MultiModelRequest {
  session_id: string;
  outcome: string;
  exposure: string;
  models: { label: string; covariates: string[] }[];
  outcome_kind?: "continuous" | "binary" | "survival";
  time_col?: string;
  categorical?: string[];
  exposure_categorical?: boolean;
}
export const runMultiModel = (data: MultiModelRequest) =>
  api.post("/api/multimodel/analyze", data);

// Threshold (two-piecewise) regression
export interface ThresholdRequest {
  session_id: string;
  outcome: string;
  exposure: string;
  outcome_kind?: "continuous" | "binary" | "survival";
  time_col?: string;
  covariates?: string[];
  categorical?: string[];
}
export const runThreshold = (data: ThresholdRequest) =>
  api.post("/api/threshold/analyze", data);

// Normality
export interface NormalityRequest {
  session_id: string;
  variables: string[];
  group_column?: string;
  alpha?: number;
}
export const runNormality = (data: NormalityRequest) =>
  api.post("/api/stats/normality", data);

// Missing data
export const runMissingPattern = (data: object) => api.post("/api/missing_data/pattern", data);
export const runMCARTest     = (data: object) => api.post("/api/missing_data/mcar_test", data);
export const runImputationCompare = (data: object) => api.post("/api/missing_data/imputation_compare", data);
// The /api/models/ copy is async and degrades gracefully; the /api/missing_data/
// twin 500s when a sub-analysis fails, so prefer this one.
export const runMnarSensitivity = (data: object) => api.post("/api/models/mnar_sensitivity", data);
export const getExternalImputeReferenceColumns = (file: File) => {
  const fd = new FormData();
  fd.append("file", file);
  return api.post("/api/missing_data/external_impute_reference_columns", fd);
};
export const runExternalImputePreview = (data: {
  sessionId: string;
  target: string;
  referenceTarget?: string;
  predictors: string[];
  predictorMappings?: Record<string, string>;
  method: string;
  mechanism: string;
  maxIter: number;
  randomState: number;
  stratifyBy?: string;
  file: File;
}) => {
  const fd = new FormData();
  fd.append("session_id", data.sessionId);
  fd.append("target", data.target);
  if (data.referenceTarget) fd.append("reference_target", data.referenceTarget);
  fd.append("predictors", JSON.stringify(data.predictors));
  fd.append("predictor_mappings", JSON.stringify(data.predictorMappings ?? {}));
  fd.append("method", data.method);
  fd.append("mechanism", data.mechanism);
  fd.append("max_iter", String(data.maxIter));
  fd.append("random_state", String(data.randomState));
  if (data.stratifyBy) fd.append("stratify_by", data.stratifyBy);
  fd.append("file", data.file);
  return api.post("/api/missing_data/external_impute_preview", fd);
};
export const runExternalImputeApply = (data: {
  sessionId: string;
  target: string;
  referenceTarget?: string;
  predictors: string[];
  predictorMappings?: Record<string, string>;
  method: string;
  mechanism: string;
  maxIter: number;
  randomState: number;
  stratifyBy?: string;
  file: File;
}) => {
  const fd = new FormData();
  fd.append("session_id", data.sessionId);
  fd.append("target", data.target);
  if (data.referenceTarget) fd.append("reference_target", data.referenceTarget);
  fd.append("predictors", JSON.stringify(data.predictors));
  fd.append("predictor_mappings", JSON.stringify(data.predictorMappings ?? {}));
  fd.append("method", data.method);
  fd.append("mechanism", data.mechanism);
  fd.append("max_iter", String(data.maxIter));
  fd.append("random_state", String(data.randomState));
  if (data.stratifyBy) fd.append("stratify_by", data.stratifyBy);
  fd.append("file", data.file);
  return api.post("/api/missing_data/external_impute_apply", fd);
};
export const runExternalImputeTransfer = (data: {
  sessionId: string;
  target: string;
  previewRows: Array<{ row_index: number; imputed_value: unknown }>;
}) => api.post("/api/missing_data/external_impute_transfer", {
  session_id: data.sessionId,
  target: data.target,
  preview_rows: data.previewRows,
});
export const runMissingDiagnostics = (sessionId: string, columns?: string[]) =>
  api.post(`/api/compute/${sessionId}/missing_diagnostics`, { columns });
export const fillBlanks = (sessionId: string, column: string, value: string, newColumn?: string) =>
  api.post(`/api/compute/${sessionId}/fill_blanks`, {
    column,
    value,
    ...(newColumn ? { new_column: newColumn } : {}),
  });

// Diagnostics
export const runLinearDiagFull = (data: object) => api.post("/api/diagnostics/linear_full", data);
export const runLogisticDiag   = (data: object) => api.post("/api/model_diagnostics/logistic_diagnostics", data);
export const runCoxDiag        = (data: object) => api.post("/api/model_diagnostics/cox_diagnostics", data);
export const runModelValidation = (data: object) => api.post("/api/model_diagnostics/model_validation", data);
export const runExternalValidationLogistic = (data: object) => api.post("/api/model_diagnostics/external_validation_logistic", data);
export const runNriIdi         = (data: object) => api.post("/api/model_diagnostics/nri_idi", data);

// Decision curve
export const runCalibration    = (data: object) => api.post("/api/decision_curve/calibration", data);
export const runDCA            = (data: object) => api.post("/api/decision_curve/dca", data);
export const runIntegratedExtValDCA = (data: object) => api.post("/api/decision_curve/integrated_extval_dca", data);

// Model comparison
export const runNestedLR       = (data: object) => api.post("/api/model_compare/nested_lr_test", data);
export const runCompareModels  = (data: object) => api.post("/api/model_compare/compare_models", data);
export const runAddedValue     = (data: object) => api.post("/api/model_compare/added_value", data);
export const runIV2SLS         = (data: object) => api.post("/api/causal/iv_2sls", data);
// Full sensitivity suite. Three routes expose this; /api/models/ is the async,
// strictly-validated one (the model_diagnostics route is a smaller subset).
export const runCausalSensitivity = (data: object) => api.post("/api/models/causal_sensitivity", data);
export const runMediation      = (data: object) => api.post("/api/causal/mediation", data);
export const runTargetTrial    = (data: object) => api.post("/api/causal/target_trial", data);
export const runDiD            = (data: object) => api.post("/api/causal/did", data);
export const runRDD            = (data: object) => api.post("/api/causal/rdd", data);
export const runDAGAdjustment  = (data: object) => api.post("/api/causal/dag_adjustment", data);
export const runSEM            = (data: object) => api.post("/api/causal/sem", data);

// Survival advanced
export const runMICE         = (data: object) => api.post("/api/survival_advanced/mice", data);
export const runMICEPreview  = (data: object) => api.post("/api/survival_advanced/mice_preview", data);
export const runMICETransfer = (data: { session_id: string; preview_rows: Array<{ row_index: number; column: string; imputed_value: unknown }> }) =>
  api.post("/api/survival_advanced/mice_transfer", data);
export const runFineGray   = (data: object) => api.post("/api/survival_advanced/fine_gray", data);
export const runEValue     = (data: object) => api.post("/api/survival_advanced/evalue", data);
export const runLandmark   = (data: object) => api.post("/api/survival_advanced/landmark", data);
export const runRMST       = (data: object) => api.post("/api/survival_advanced/rmst", data);
export const runRecurrentLWYY = (data: object) => api.post("/api/survival_advanced/recurrent_lwyy", data);
export const runIntervalCensored = (data: object) => api.post("/api/survival_advanced/interval_censored", data);
export const runFrailty    = (data: object) => api.post("/api/survival_advanced/frailty", data);
export const runMultistate = (data: object) => api.post("/api/survival_advanced/multistate", data);
export const runDynamicPrediction = (data: object) => api.post("/api/survival_advanced/dynamic_prediction", data);
export const runJointModel = (data: object) => api.post("/api/survival_advanced/joint_model", data);
export const runExternalValidationSurvival = (data: object) => api.post("/api/survival_advanced/external_validation", data);

// Article parser
export const parseArticle = (file: File) => {
  const fd = new FormData();
  fd.append("file", file);
  return api.post("/api/article_parser/parse", fd);
};

// Column operations
export const renameColumn = (sessionId: string, oldName: string, newName: string) =>
  api.post(`/api/compute/${sessionId}/rename`, { old_name: oldName, new_name: newName });
export const getNameSuggestions = (sessionId: string) =>
  api.get(`/api/sessions/${sessionId}/name_suggestions`);

// Session management
export const saveSession   = (sessionId: string) => api.get(`/api/sessions/${sessionId}/save_session`, { responseType: "blob" });
export const getSessionInfo = (sessionId: string) => api.get(`/api/sessions/${sessionId}`);
export const createBlankSession = () => api.post("/api/sessions/blank");
export const loadSession   = (file: File) => { const fd = new FormData(); fd.append("file", file); return api.post("/api/sessions/load_session", fd); };
export const getAuditTrail = (sessionId: string) => api.get(`/api/sessions/${sessionId}/audit`);
export const saveMetadata  = (sessionId: string, columns: Record<string, unknown>) => api.post(`/api/sessions/${sessionId}/metadata`, { columns });
export const setColumnKind = (sessionId: string, column: string, kind: string) => api.post(`/api/sessions/${sessionId}/kind`, { column, kind });
export const setColumnDecimalsApi = (sessionId: string, column: string, decimals: number | null) =>
  api.post(`/api/sessions/${sessionId}/decimals`, { column, decimals });
export const getColumnDecimalsApi = (sessionId: string) =>
  api.get<Record<string, number>>(`/api/sessions/${sessionId}/decimals`);
export const deleteRow     = (sessionId: string, rowIndex: number) => api.delete(`/api/sessions/${sessionId}/row/${rowIndex}`);

// Publication export
export const exportTableDocx = (data: object) => api.post("/api/pub_export/table_docx", data, { responseType: "blob" });
export const exportStyledTable = (data: object) => api.post("/api/pub_export/styled_table", data, { responseType: "blob" });
export const getFigureCaption = (data: object) => api.post("/api/pub_export/figure_caption", data);

// Nomogram
export const buildNomogram = (data: object) => api.post("/api/nomogram/build", data);

export const selectCases = (sessionId: string, conditions: object[], apply = true) =>
  api.post(`/api/sessions/${sessionId}/select_cases`, { conditions, apply });
export const clearCases  = (sessionId: string) =>
  api.delete(`/api/sessions/${sessionId}/select_cases`);

// Factor Analysis / PCA
export const runFactorPCA = (data: object) => api.post("/api/factor/factor_pca", data);

// Bayesian Statistics
export const runBayesian = (data: object) => api.post("/api/bayesian", data);

// Advanced Cleaning & Imputation
export const runDropMissing = (sessionId: string, data: object) => api.post(`/api/compute/${sessionId}/drop_missing`, data);
export const runCleanOutliers = (sessionId: string, data: object) => api.post(`/api/compute/${sessionId}/clean_outliers`, data);
export const runFindReplace = (sessionId: string, data: object) => api.post(`/api/compute/${sessionId}/find_replace`, data);
export const replaceColumnValues = (sessionId: string, data: object) => api.post(`/api/compute/${sessionId}/replace_values`, data);
export const parseColumnDates = (sessionId: string, data: object) => api.post(`/api/compute/${sessionId}/parse_dates`, data);

// ── Code runner ────────────────────────────────────────────────────────────

export interface CodeRunnerStatus {
  enabled: boolean;
  max_timeout_s: number;
  max_code_bytes: number;
  rate_limit_per_min: number;
  rate_limit_per_hour: number;
}

export interface CodeRunRequest {
  session_id: string;
  code: string;
  timeout?: number;
}

export interface CodeRunResponse {
  stdout: string;
  stderr: string;
  figures: string[];   // base64 PNGs
  exit_code: number;
  time_used_s: number;
  error: string | null;
  timed_out: boolean;
}

export const codeRunnerStatus = () => api.get<CodeRunnerStatus>("/api/code/status");
export const runCode = (data: CodeRunRequest, signal?: AbortSignal) =>
  api.post<CodeRunResponse>("/api/code/run", data, { signal });
