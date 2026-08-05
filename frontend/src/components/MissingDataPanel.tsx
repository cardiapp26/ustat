import { useState, type ReactNode } from "react";
import { useStore, isNumericKind } from "../store";
import {
  fillBlanks,
  getExternalImputeReferenceColumns,
  runExternalImputePreview,
  runExternalImputeTransfer,
  runImputationCompare,
  runMCARTest,
  runMICEPreview,
  runMICETransfer,
  runMissingDiagnostics,
  runMnarSensitivity,
} from "../api";
import ResultExporter from "./ResultExporter";
import api from "../api";
import { CleaningTab } from "./CleaningTab";
import { fmtP } from "../lib/format";
import { useMissing } from "./MissingGuard";

interface DiagCol { name: string; n_missing: number; pct: number; kind: string; is_numeric: boolean; depends_on: string[]; likely: string }
interface DiagResult { columns: DiagCol[]; overall_hint: string; recommendation: string; any_mar: boolean }
type MissingSort = "missing-desc" | "missing-asc" | "name-asc" | "name-desc";
type QuickMethod = "__mean__" | "__median__" | "__mode__" | "__mice__";

interface MicePreviewRow {
  row_index: number;
  column: string;
  imputed_value: unknown;
}
interface MiceExportResult {
  result_text?: string;
  methods_text?: string;
  export_rows?: unknown[][];
  preview_rows?: MicePreviewRow[];
  columns?: Array<{ column: string; method: string; n_imputed: number; mean_imputed?: number | null; min_imputed?: number | null; max_imputed?: number | null; mode_imputed?: unknown }>;
  total_imputed?: number;
  preview_only?: boolean;
  applied?: boolean;
}
interface McarResult {
  statistic: number | string;
  df: number;
  p: number | string;
  significant: boolean;
}
interface CompareColumn {
  col: string;
  before?: { mean?: number | string };
  after?: { mean?: number | string };
  ks_p?: number | null;
}
interface CompareEntry { strategy: string; columns: CompareColumn[] }
interface CompareResult { comparisons?: CompareEntry[] }
interface ExternalPreviewRow { row_index: number; imputed_value: unknown; predictors_missing: number; stratum?: string }
interface ExternalReferenceColumn { name: string; dtype: string; kind: string; n_missing: number }
interface ExternalReferenceColumnsResult { columns: ExternalReferenceColumn[]; n_rows: number }
interface ExternalImputeResult {
  target: string;
  reference_target?: string;
  predictors: string[];
  reference_predictors?: string[];
  predictor_mappings?: Record<string, string>;
  method: string;
  mechanism: string;
  n_missing_target: number;
  n_imputed: number;
  reference_rows: number;
  reference_complete_rows: number;
  preview_rows: ExternalPreviewRow[];
  warnings?: string[];
  result_text?: string;
  methods_text?: string;
  export_rows?: unknown[][];
  applied?: boolean;
}

/** Every heavy sub-analysis of /api/models/mnar_sensitivity is individually
 *  guarded server-side and degrades to `{ available: false, reason: "..." }`
 *  instead of failing the whole request, so each block must be able to render
 *  its own reason. */
interface MnarBlockBase { available?: boolean; reason?: string }
interface MnarPmmScenario { delta: number; pooled_means?: Record<string, number | null> }
interface MnarPatternMixture extends MnarBlockBase {
  n_imputations?: number;
  delta_values?: number[];
  scenarios?: MnarPmmScenario[];
  interpretation?: string;
}
interface MnarModelDeltaRow {
  delta: number;
  estimate?: number; log_odds?: number; odds_ratio?: number; hr?: number; se?: number; error?: string;
}
interface MnarModelDelta extends MnarBlockBase {
  model_type?: string;
  results?: MnarModelDeltaRow[];
  interpretation?: string;
}
interface MnarHeckman extends MnarBlockBase {
  n_total?: number;
  n_observed_outcome?: number;
  selection_rate?: number;
  inverse_mills_ratio_p?: number | null;
  selection_bias_signal?: boolean;
  outcome_coefficients?: Array<{ variable: string; estimate: number; se: number; p: number }>;
  interpretation?: string;
}
interface MnarIsniRow {
  variable: string;
  missingness_indicator_coef?: number;
  target_coefficient?: number | null;
  isni?: number;
  high_sensitivity?: boolean;
  error?: string;
}
interface MnarIsni extends MnarBlockBase { indices?: MnarIsniRow[]; interpretation?: string }
interface MnarConvergence extends MnarBlockBase {
  variables?: Record<string, { r_hat_proxy?: number | null; converged?: boolean }>;
  warning?: string;
}
interface MnarPpcCheck {
  variable: string;
  available?: boolean;
  reason?: string;
  observed_mean?: number;
  imputed_mean?: number;
  mean_difference?: number;
  ks_p?: number | null;
  flag_distribution_shift?: boolean;
}
interface MnarPpc extends MnarBlockBase { checks?: MnarPpcCheck[] }
interface MnarCongeniality extends MnarBlockBase {
  congenial?: boolean;
  analysis_variables_missing_from_imputation?: string[];
  passive_variables?: string[];
  recommendation?: string;
}
interface MnarAux extends MnarBlockBase {
  recommended_auxiliary_variables?: Array<{
    target: string; candidate: string; missingness_corr_abs: number; value_corr_abs: number; priority_score: number;
  }>;
  method_note?: string;
}
interface MnarSurvivalRow {
  delta: number;
  censored_weight_multiplier?: number;
  concordance?: number;
  coefficients?: Array<{ variable: string; hr: number }>;
  error?: string;
}
interface MnarSurvival extends MnarBlockBase { results?: MnarSurvivalRow[]; interpretation?: string }
interface MnarResult {
  test?: string;
  n?: number;
  columns?: string[];
  pattern_mixture_model?: MnarPatternMixture | null;
  model_delta_sensitivity?: MnarModelDelta | null;
  heckman_selection_model?: MnarHeckman | null;
  isni?: MnarIsni | null;
  mice_convergence_diagnostics?: MnarConvergence | null;
  imputation_model_diagnostics?: MnarPpc | null;
  congeniality_assessment?: MnarCongeniality | null;
  passive_imputation?: {
    formulas?: Record<string, string>;
    preview?: Record<string, { n_nonmissing?: number; mean?: number | null }>;
  } | null;
  survival_specific_imputation?: { enabled?: boolean; auxiliary_variables?: string[] } | null;
  auxiliary_variable_guidance?: MnarAux | null;
  survival_mnar_sensitivity?: MnarSurvival | null;
  warnings?: string[];
  assumptions?: Array<{ name: string; met: boolean; detail: string }>;
  result_text?: string;
  r_code?: string;
}
type MnarModelType = "linear" | "logistic" | "cox";

const QUICK_SUFFIX: Record<QuickMethod, string> = {
  __mean__: "mean",
  __median__: "median",
  __mode__: "mode",
  __mice__: "imp",
};

const errText = (e: unknown): string =>
  (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Request failed";

const normColumnName = (name: string) => name.trim().toLowerCase();

/** Mirrors the backend default `delta_values` of /api/models/mnar_sensitivity. */
const MNAR_DEFAULT_DELTAS = "-1, 0, 1";

/** Parse the comma-separated delta input; null means "not a valid delta list". */
const parseDeltaValues = (raw: string): number[] | null => {
  const parts = raw.split(",").map((p) => p.trim()).filter((p) => p.length > 0);
  if (parts.length === 0) return null;
  const values = parts.map(Number);
  return values.some((v) => !Number.isFinite(v)) ? null : values;
};

/** Reason a guarded sub-analysis could not be produced, or null when it has results. */
const mnarUnavailableReason = (block: MnarBlockBase | null | undefined, fallback: string): string | null => {
  if (block == null) return fallback;
  if (block.available === false) return block.reason || fallback;
  return null;
};

const fmtNum = (v: number | null | undefined): string =>
  v == null || !Number.isFinite(v) ? "—" : String(v);

export default function MissingDataPanel() {
  const session = useStore((s) => s.session);
  const columns = session?.columns ?? [];
  const numCols = columns.filter((c) => isNumericKind(c.kind));
  const sid = session?.session_id ?? "";
  const dataVersion = useStore((s) => s.dataVersion);
  const columnNames = columns.map((c) => c.name);
  const { info: fullMissingInfo } = useMissing(sid, columnNames, dataVersion);
  const [activeSubTab, setActiveSubTab] = useState<"overview" | "cleaning" | "reference" | "mnar">("overview");

  const preview = session?.preview ?? [];
  const missingInfo = columns
    .map((c) => {
      const previewMissing = preview.filter(
        (r) => r[c.name] === null || r[c.name] === undefined || r[c.name] === ""
      ).length;
      const serverMissing = fullMissingInfo?.per_column[c.name];
      const nMiss = serverMissing?.count ?? previewMissing;
      return {
        name: c.name,
        kind: c.kind,
        isNum: isNumericKind(c.kind),
        nMiss,
        pct: serverMissing?.pct ?? (preview.length > 0 ? (nMiss / preview.length) * 100 : 0),
      };
    })
    .filter((m) => m.nMiss > 0);

  // Selection + MICE state
  const [selected, setSelected] = useState<string[]>([]);
  const [missingSort, setMissingSort] = useState<MissingSort>("missing-desc");
  const [miceIter, setMiceIter] = useState(20);
  const [miceSeed, setMiceSeed] = useState(42);
  const [miceMechanism, setMiceMechanism] = useState<"unknown" | "MCAR" | "MAR" | "MNAR">("unknown");
  const [miceLoading, setMiceLoading] = useState(false);
  const [micePreviewResult, setMicePreviewResult] = useState<MiceExportResult | null>(null);
  const [miceTransferLoading, setMiceTransferLoading] = useState(false);
  const [busy, setBusy] = useState<string | null>(null); // per-row action in flight
  const [err, setErr] = useState<string | null>(null);
  const [mutationNotice, setMutationNotice] = useState<string | null>(null);

  // Diagnostics state
  const [diag, setDiag] = useState<DiagResult | null>(null);
  const [mcar, setMcar] = useState<McarResult | null>(null);
  const [mcarNote, setMcarNote] = useState<string | null>(null);
  const [compare, setCompare] = useState<CompareResult | null>(null);
  const [externalTarget, setExternalTarget] = useState("");
  const [externalPredictors, setExternalPredictors] = useState<string[]>([]);
  const [externalReferenceTarget, setExternalReferenceTarget] = useState("");
  const [externalPredictorMappings, setExternalPredictorMappings] = useState<Record<string, string>>({});
  const [externalFile, setExternalFile] = useState<File | null>(null);
  const [externalReferenceMeta, setExternalReferenceMeta] = useState<ExternalReferenceColumnsResult | null>(null);
  const [externalMethod, setExternalMethod] = useState<"pmm" | "mice">("pmm");
  const [externalStratifyBy, setExternalStratifyBy] = useState("");
  const [externalResult, setExternalResult] = useState<ExternalImputeResult | null>(null);
  const [externalLoading, setExternalLoading] = useState<"columns" | "preview" | "apply" | null>(null);

  // MNAR sensitivity state
  const [mnarColumns, setMnarColumns] = useState<string[]>([]);
  const [mnarDeltaText, setMnarDeltaText] = useState(MNAR_DEFAULT_DELTAS);
  const [mnarModelType, setMnarModelType] = useState<MnarModelType>("logistic");
  // Without an outcome model the backend skips model_delta_sensitivity, Heckman
  // and ISNI entirely — and model_type then has nothing to act on. These two
  // pickers are what make those three blocks (and the selector above) live.
  const [mnarOutcome, setMnarOutcome] = useState("");
  const [mnarPredictors, setMnarPredictors] = useState<string[]>([]);
  const [mnarLoading, setMnarLoading] = useState(false);
  const [mnarResult, setMnarResult] = useState<MnarResult | null>(null);

  if (!session) return <p className="text-gray-400 text-sm p-6">Upload data first.</p>;

  const refresh = async () => {
    const r = await api.get(`/api/stats/${sid}/refresh`);
    useStore.getState().setSession({ ...useStore.getState().session!, ...r.data });
    useStore.setState((s) => ({ dataVersion: s.dataVersion + 1 }));
  };

  const clearDiagnostics = () => {
    setDiag(null);
    setMcar(null);
    setMcarNote(null);
  };

  const toggle = (name: string) => {
    clearDiagnostics();
    setSelected((p) => (p.includes(name) ? p.filter((c) => c !== name) : [...p, name]));
  };

  const sortedMissingInfo = [...missingInfo].sort((a, b) => {
    if (missingSort === "name-asc" || missingSort === "name-desc") {
      const order = a.name.localeCompare(b.name, "tr", { numeric: true, sensitivity: "base" });
      return missingSort === "name-asc" ? order : -order;
    }
    const order = a.pct - b.pct || a.name.localeCompare(b.name, "tr", { numeric: true, sensitivity: "base" });
    return missingSort === "missing-asc" ? order : -order;
  });

  const toggleNameSort = () =>
    setMissingSort((current) => current === "name-asc" ? "name-desc" : "name-asc");

  const toggleMissingSort = () =>
    setMissingSort((current) => current === "missing-asc" ? "missing-desc" : "missing-asc");

  const sortArrow = (asc: MissingSort, desc: MissingSort) =>
    missingSort === asc ? "↑" : missingSort === desc ? "↓" : "↕";

  const nextColumnName = (source: string, method: QuickMethod): string => {
    const base = `${source}_${QUICK_SUFFIX[method]}`;
    const existing = new Set(columns.map((c) => c.name));
    let candidate = base;
    let index = 2;
    while (existing.has(candidate)) candidate = `${base}_${index++}`;
    return candidate;
  };

  // Per-row quick imputation (acts immediately on one column).
  const quickFill = async (col: string, method: QuickMethod) => {
    setBusy(`${col}:${method}`); setErr(null); setMutationNotice(null);
    try {
      const newColumn = nextColumnName(col, method);
      const response = await fillBlanks(sid, col, method, newColumn);
      await refresh();
      setMutationNotice(
        `${col} kept; ${response.data.column} created with ${response.data.n_filled} missing value(s) filled.`
      );
    } catch (e: unknown) {
      setErr(errText(e));
    } finally {
      setBusy(null);
    }
  };

  const runDiagnostics = async () => {
    if (selected.length === 0) { setErr("Select at least one column to analyze"); return; }
    const selectedNumeric = selected.filter((name) => missingInfo.some((m) => m.name === name && m.isNum));
    setBusy("diag"); setErr(null); setDiag(null); setMcar(null); setMcarNote(null);
    try {
      const diagRequest = runMissingDiagnostics(sid, selected);
      const mcarRequest = selectedNumeric.length >= 2
        ? runMCARTest({ session_id: sid, columns: selectedNumeric })
        : null;
      const [d, m] = await Promise.allSettled([
        diagRequest,
        ...(mcarRequest ? [mcarRequest] : []),
      ]);
      if (d.status === "fulfilled") setDiag(d.value.data);
      else setErr(errText(d.reason));
      if (mcarRequest) {
        if (m?.status === "fulfilled") setMcar(m.value.data);
        else if (m?.status === "rejected") setMcarNote(`Little's MCAR test could not be calculated: ${errText(m.reason)}`);
      } else {
        setMcarNote("Little's MCAR test requires at least two selected numeric variables. The dependence analysis below is limited to the selected variable(s).");
      }
    } finally {
      setBusy(null);
    }
  };

  const runCompare = async () => {
    if (selected.length === 0) { setErr("Select columns to compare"); return; }
    setBusy("compare"); setErr(null); setCompare(null);
    try {
      const r = await runImputationCompare({ session_id: sid, columns: selected, strategies: ["median", "mice"] });
      setCompare(r.data);
    } catch (e: unknown) {
      setErr(errText(e));
    } finally {
      setBusy(null);
    }
  };

  const externalTargetName = externalTarget || missingInfo[0]?.name || "";
  const currentColumnNames = new Set(columns.map((c) => c.name));
  const currentColumnByNorm = new Map(columns.map((c) => [normColumnName(c.name), c.name]));
  const externalReferenceColumns = externalReferenceMeta?.columns ?? [];
  const autoReferenceTarget = externalReferenceColumns.find(
    (c) => normColumnName(c.name) === normColumnName(externalTargetName)
  )?.name ?? "";
  const externalReferenceTargetName = externalReferenceTarget || autoReferenceTarget;
  const externalPredictorColumns = externalReferenceColumns.filter(
    (c) => normColumnName(c.name) !== normColumnName(externalReferenceTargetName)
  );
  const predictorMappings = Object.fromEntries(
    externalPredictors.map((name) => [
      name,
      externalPredictorMappings[name] || currentColumnByNorm.get(normColumnName(name)) || "",
    ])
  );
  const externalPayload = () => ({
    sessionId: sid,
    target: externalTargetName,
    referenceTarget: externalReferenceTargetName,
    predictors: externalPredictors,
    predictorMappings,
    method: externalMethod,
    mechanism: miceMechanism,
    maxIter: miceIter,
    randomState: miceSeed,
    stratifyBy: externalStratifyBy || undefined,
    file: externalFile!,
  });

  const validateExternal = (): boolean => {
    if (!externalTargetName) { setErr("Select a target column with missing values"); return false; }
    if (externalPredictors.length === 0) { setErr("Select at least one predictor for the target"); return false; }
    if (!externalFile) { setErr("Upload a reference dataset first"); return false; }
    if (!externalReferenceTargetName) { setErr("Select matching target column in reference dataset"); return false; }
    if (!externalReferenceColumns.some((c) => c.name === externalReferenceTargetName)) {
      setErr(`Reference dataset must contain target column '${externalReferenceTargetName}'`);
      return false;
    }
    const missingCurrent = Object.entries(predictorMappings)
      .filter(([, currentName]) => !currentName || !currentColumnNames.has(currentName))
      .map(([refName]) => refName);
    if (missingCurrent.length > 0) {
      setErr(`Select current data match for: ${missingCurrent.join(", ")}`);
      return false;
    }
    return true;
  };

  const loadExternalReferenceColumns = async (file: File | null) => {
    setExternalFile(file);
    setExternalReferenceMeta(null);
    setExternalPredictors([]);
    setExternalReferenceTarget("");
    setExternalPredictorMappings({});
    setExternalResult(null);
    if (!file) return;
    setExternalLoading("columns"); setErr(null);
    try {
      const res = await getExternalImputeReferenceColumns(file);
      const meta = res.data as ExternalReferenceColumnsResult;
      setExternalReferenceMeta(meta);
      setExternalReferenceTarget(
        meta.columns.find((c) => normColumnName(c.name) === normColumnName(externalTargetName))?.name ?? ""
      );
      setExternalPredictorMappings(Object.fromEntries(
        meta.columns.map((c) => [c.name, currentColumnByNorm.get(normColumnName(c.name)) ?? ""])
      ));
    } catch (e: unknown) {
      setErr(errText(e));
    } finally {
      setExternalLoading(null);
    }
  };

  const runExternalPreview = async () => {
    if (!validateExternal()) return;
    setExternalLoading("preview"); setErr(null); setExternalResult(null); setMutationNotice(null);
    try {
      const res = await runExternalImputePreview(externalPayload());
      setExternalResult(res.data);
    } catch (e: unknown) {
      setErr(errText(e));
    } finally {
      setExternalLoading(null);
    }
  };

  const applyExternalImputation = async () => {
    if (!externalResult?.preview_rows?.length) {
      setErr("Preview target estimates before transferring data");
      return;
    }
    setExternalLoading("apply"); setErr(null); setMutationNotice(null);
    try {
      const res = await runExternalImputeTransfer({
        sessionId: sid,
        target: externalResult.target || externalTargetName,
        previewRows: externalResult.preview_rows.map((row) => ({
          row_index: row.row_index,
          imputed_value: row.imputed_value,
        })),
      });
      setExternalResult((current) => current ? { ...current, applied: true } : current);
      await refresh();
      setMutationNotice(`${res.data.n_imputed} value(s) transferred into ${res.data.target}.`);
    } catch (e: unknown) {
      setErr(errText(e));
    } finally {
      setExternalLoading(null);
    }
  };

  const handleMICEPreview = async () => {
    if (selected.length === 0) { setErr("Select columns to impute"); return; }
    setMiceLoading(true); setErr(null); setMutationNotice(null); setMicePreviewResult(null);
    try {
      const res = await runMICEPreview({
        session_id: sid, columns: selected,
        max_iter: miceIter, random_state: miceSeed, mechanism: miceMechanism,
      });
      setMicePreviewResult(res.data);
    } catch (e: unknown) {
      setErr(errText(e));
    } finally {
      setMiceLoading(false);
    }
  };

  const handleMICETransfer = async () => {
    if (!micePreviewResult?.preview_rows?.length) {
      setErr("Preview PMM estimates before transferring");
      return;
    }
    setMiceTransferLoading(true); setErr(null); setMutationNotice(null);
    try {
      const res = await runMICETransfer({
        session_id: sid,
        preview_rows: micePreviewResult.preview_rows.map((r) => ({
          row_index: r.row_index,
          column: r.column,
          imputed_value: r.imputed_value,
        })),
      });
      setMicePreviewResult((current) => current ? { ...current, applied: true } : current);
      await refresh();
      setMutationNotice(`${res.data.total_imputed} value(s) transferred into original columns: ${res.data.columns.join(", ")}.`);
    } catch (e: unknown) {
      setErr(errText(e));
    } finally {
      setMiceTransferLoading(false);
    }
  };

  const toggleMnarColumn = (name: string) => {
    setMnarResult(null);
    setMnarColumns((p) => (p.includes(name) ? p.filter((c) => c !== name) : [...p, name]));
  };

  const runMnar = async () => {
    if (mnarColumns.length === 0) { setErr("Select at least one variable with missing data"); return; }
    const deltaValues = parseDeltaValues(mnarDeltaText);
    if (!deltaValues) { setErr("Enter delta values as comma-separated numbers, e.g. -1, 0, 1"); return; }
    setErr(null); setMnarResult(null); setMnarLoading(true);
    try {
      const payload: Record<string, unknown> = {
        session_id: sid,
        columns: mnarColumns,
        delta_values: deltaValues,
        model_type: mnarModelType,
      };
      // Only send the outcome model when it is complete; a half-specified one
      // makes the backend fall back to the same placeholders as sending none.
      if (mnarOutcome && mnarPredictors.length > 0) {
        payload.outcome_col = mnarOutcome;
        payload.predictors = mnarPredictors;
      }
      const res = await runMnarSensitivity(payload);
      setMnarResult(res.data);
    } catch (e: unknown) {
      setErr(errText(e));
    } finally {
      setMnarLoading(false);
    }
  };

  const pctClass = (pct: number) =>
    pct > 30 ? "bg-red-100 text-red-600" : pct > 10 ? "bg-amber-100 text-amber-600" : "bg-gray-100 text-gray-500";
  const QuickBtn = ({ col, method, label, show }: { col: string; method: QuickMethod; label: string; show: boolean }) =>
    !show ? null : (
      <button
        onClick={() => quickFill(col, method)}
        disabled={busy === `${col}:${method}`}
        className="text-[10px] px-1.5 py-0.5 rounded border border-gray-200 text-gray-500 hover:bg-indigo-50 hover:text-indigo-600 hover:border-indigo-300 disabled:opacity-40 transition-colors"
      >
        {busy === `${col}:${method}` ? "…" : label}
      </button>
    );

  /** Renders one MNAR sub-analysis, falling back to the backend's own
   *  `{ available: false, reason }` explanation instead of an empty tile. */
  const MnarBlock = ({ title, block, fallback, children }: {
    title: string;
    block: MnarBlockBase | null | undefined;
    fallback: string;
    children: ReactNode;
  }) => {
    const reason = mnarUnavailableReason(block, fallback);
    return (
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <div className="px-4 py-2 bg-gray-50 border-b border-gray-100">
          <h4 className="text-xs font-semibold text-gray-700">{title}</h4>
        </div>
        <div className="px-4 py-3 space-y-2">
          {reason ? (
            <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-[11px] text-amber-700">
              <span className="font-semibold">Not available — </span>{reason}
            </div>
          ) : children}
        </div>
      </div>
    );
  };

  const mnarAnalysedColumns = mnarResult?.columns ?? mnarColumns;

  return (
    <div className="max-w-4xl mx-auto p-4">
      <div
        className="mb-5 flex items-center gap-1 border-b border-gray-200"
        role="tablist"
        aria-label="Missing data sections"
      >
        {([
          ["overview", "Missing Data Overview"],
          ["cleaning", "Data Cleaning"],
          ["reference", "Reference Imputation"],
          ["mnar", "MNAR Sensitivity"],
        ] as const).map(([id, label]) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={activeSubTab === id}
            onClick={() => setActiveSubTab(id)}
            className={`relative px-4 py-2.5 text-sm font-medium transition-colors ${
              activeSubTab === id
                ? "text-indigo-700"
                : "text-gray-500 hover:text-gray-800"
            }`}
          >
            {label}
            {activeSubTab === id && (
              <span className="absolute inset-x-0 -bottom-px h-0.5 rounded-full bg-indigo-600" />
            )}
          </button>
        ))}
      </div>

      <div className={activeSubTab === "overview" ? "space-y-5" : "hidden"} role="tabpanel">
        {activeSubTab === "overview" && err && <div className="bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-xs text-red-600">{err}</div>}
        {activeSubTab === "overview" && mutationNotice && <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-3 py-2 text-xs text-emerald-700">{mutationNotice}</div>}

        {/* ── Overview — list ── */}
        <div className="border border-gray-200 rounded-xl overflow-hidden">
          <div className="px-5 py-3.5 bg-gray-50 border-b border-gray-100 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-gray-800">Missing Data Overview</h3>
              <p className="text-[11px] text-gray-400 mt-0.5">Tick rows for MICE / comparison, or impute a single column inline.</p>
            </div>
            {missingInfo.length > 0 && (
              <div className="flex items-center gap-2">
                <label className="flex items-center gap-1.5 text-[10px] text-gray-500">
                  <span>Sort</span>
                  <select value={missingSort} onChange={(e) => setMissingSort(e.target.value as MissingSort)}
                    className="border border-gray-300 rounded px-2 py-1 bg-white text-[10px] text-gray-600">
                    <option value="missing-desc">Missing %: high to low</option>
                    <option value="missing-asc">Missing %: low to high</option>
                    <option value="name-asc">Name: A to Z</option>
                    <option value="name-desc">Name: Z to A</option>
                  </select>
                </label>
                <button onClick={() => {
                  clearDiagnostics();
                  setSelected(selected.length === missingInfo.length ? [] : missingInfo.map((m) => m.name));
                }}
                  className="text-[10px] px-2 py-1 rounded border border-gray-300 text-gray-500 hover:bg-gray-100">
                  {selected.length === missingInfo.length ? "Clear all" : "Select all"}
                </button>
              </div>
            )}
          </div>
          {missingInfo.length === 0 ? (
            <div className="px-5 py-4">
              <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-4 py-3 text-sm text-emerald-700">
                ✅ No missing values detected in any column.
              </div>
            </div>
          ) : (
            <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-gray-400 border-b border-gray-100">
                <th className="px-3 py-2 w-8"></th>
                <th className="px-3 py-2">
                  <button
                    type="button"
                    onClick={toggleNameSort}
                    className="inline-flex items-center gap-1 hover:text-indigo-600 transition-colors"
                    title="Sort by variable name"
                  >
                    Variable <span aria-hidden="true">{sortArrow("name-asc", "name-desc")}</span>
                  </button>
                </th>
                <th className="px-3 py-2">Type</th>
                <th className="px-3 py-2 text-right">
                  <button
                    type="button"
                    onClick={toggleMissingSort}
                    className="inline-flex items-center justify-end gap-1 hover:text-indigo-600 transition-colors"
                    title="Sort by missing percentage"
                  >
                    Missing <span aria-hidden="true">{sortArrow("missing-asc", "missing-desc")}</span>
                  </button>
                </th>
                <th className="px-3 py-2 w-28">
                  <button
                    type="button"
                    onClick={toggleMissingSort}
                    className="inline-flex items-center gap-1 hover:text-indigo-600 transition-colors"
                    title="Sort by missing percentage"
                  >
                    % <span aria-hidden="true">{sortArrow("missing-asc", "missing-desc")}</span>
                  </button>
                </th>
                <th className="px-3 py-2 text-right">Quick impute</th>
              </tr>
            </thead>
            <tbody>
              {sortedMissingInfo.map((m) => (
                <tr key={m.name} className={`border-b border-gray-50 ${selected.includes(m.name) ? "bg-indigo-50/40" : "hover:bg-gray-50"}`}>
                  <td className="px-3 py-1.5">
                    <input type="checkbox" checked={selected.includes(m.name)} onChange={() => toggle(m.name)} className="accent-indigo-500" />
                  </td>
                  <td className="px-3 py-1.5 font-medium text-gray-800 truncate max-w-[10rem]">{m.name}</td>
                  <td className="px-3 py-1.5 text-gray-500">{m.kind}</td>
                  <td className="px-3 py-1.5 text-right text-gray-600">{m.nMiss}</td>
                  <td className="px-3 py-1.5">
                    <div className="flex items-center gap-1.5">
                      <div className="flex-1 h-1.5 rounded-full bg-gray-100 overflow-hidden">
                        <div className={`h-full ${m.pct > 30 ? "bg-red-400" : m.pct > 10 ? "bg-amber-400" : "bg-gray-300"}`} style={{ width: `${Math.min(100, m.pct)}%` }} />
                      </div>
                      <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${pctClass(m.pct)}`}>{m.pct.toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="px-3 py-1.5">
                    <div className="flex items-center justify-end gap-1">
                      <QuickBtn col={m.name} method="__mean__" label="Mean" show={m.isNum} />
                      <QuickBtn col={m.name} method="__median__" label="Median" show={m.isNum} />
                      <QuickBtn col={m.name} method="__mode__" label="Mode" show={!m.isNum} />
                      <QuickBtn col={m.name} method="__mice__" label="MICE" show={m.isNum} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
            </table>
          )}
        </div>

        {missingInfo.length > 0 && (
          <>
          {/* ── Mechanism + diagnostics ── */}
          <div className="border border-gray-200 rounded-xl overflow-hidden">
            <div className="px-5 py-3.5 bg-gray-50 border-b border-gray-100 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-gray-800">Missing Mechanism</h3>
                <p className="text-[11px] text-gray-400 mt-0.5">Determines which imputation is valid. Analyze, or set it manually.</p>
              </div>
              <button onClick={runDiagnostics} disabled={busy === "diag" || selected.length === 0}
                className="text-xs px-3 py-1.5 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50">
                {busy === "diag" ? "Analyzing…" : `Analyze missingness${selected.length ? ` (${selected.length})` : ""}`}
              </button>
            </div>
            <div className="px-5 py-4 space-y-3">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {([
                  { id: "unknown", label: "Unknown", desc: "Analyze to decide" },
                  { id: "MCAR", label: "MCAR", desc: "Completely at random" },
                  { id: "MAR", label: "MAR", desc: "At random (on observed)" },
                  { id: "MNAR", label: "MNAR", desc: "Not at random" },
                ] as const).map(({ id, label, desc }) => (
                  <button key={id} onClick={() => setMiceMechanism(id)}
                    className={`flex flex-col items-start gap-0.5 px-3 py-2 rounded-lg border text-left transition-colors ${
                      miceMechanism === id ? "border-indigo-400 bg-indigo-50" : "border-gray-200 bg-white hover:border-gray-300"
                    }`}>
                    <span className={`text-xs font-semibold ${miceMechanism === id ? "text-indigo-700" : "text-gray-700"}`}>{label}</span>
                    <span className="text-[10px] text-gray-400 leading-tight">{desc}</span>
                  </button>
                ))}
              </div>

              {/* Data-driven hint (heuristic + Little's MCAR), no AI */}
              {(diag || mcar) && (
                <div className="space-y-2">
                  {mcar && (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 text-[11px] text-blue-800">
                      <span className="font-semibold">Little's MCAR test:</span> χ²={Number(mcar.statistic).toFixed(2)}, df={mcar.df}, <i>p</i>={fmtP(Number(mcar.p))}.{" "}
                      {mcar.significant
                        ? "p < 0.05 → MCAR rejected; data are likely MAR (or MNAR). MICE is appropriate."
                        : "p ≥ 0.05 → consistent with MCAR; listwise deletion is unbiased (MICE still fine)."}
                    </div>
                  )}
                  {diag && (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 text-[11px] text-blue-800">
                      <span className="font-semibold">Dependence check:</span> {diag.overall_hint} {diag.recommendation}
                      {diag.columns.some((c) => c.depends_on.length > 0) && (
                        <ul className="mt-1 ml-3 list-disc">
                          {diag.columns.filter((c) => c.depends_on.length > 0).map((c) => (
                            <li key={c.name}>{c.name}: missingness related to {c.depends_on.join(", ")} → MAR signal</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                </div>
              )}
              {mcarNote && (
                <div className="bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-[11px] text-gray-600">
                  {mcarNote}
                </div>
              )}
              {miceMechanism === "MNAR" && (
                <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-[11px] text-amber-700">
                  ⚠️ MNAR: MICE assumes MAR and may bias results. For &gt;40–50% missing, use a dedicated MNAR sensitivity analysis (pattern-mixture / selection model).
                </div>
              )}
            </div>
          </div>

          {/* ── MICE (multi-column) ── */}
          <div className="border border-indigo-200 rounded-xl overflow-hidden">
            <div className="px-5 py-3.5 bg-indigo-50 border-b border-indigo-100">
              <h3 className="text-sm font-semibold text-indigo-800">PMM Imputation (selected columns)</h3>
              <p className="text-[11px] text-indigo-400 mt-0.5">
                Preview PMM estimates for the selected columns, then transfer the imputed values directly into the original columns. For variance-correct inference, prefer the model panels' MICE option (m datasets + Rubin's-rules pooling).
              </p>
            </div>
            <div className="px-5 py-4 space-y-4">
              <div className="flex gap-4 flex-wrap">
                {([
                  ["Max iterations", miceIter, setMiceIter, 1, 100],
                  ["Seed", miceSeed, setMiceSeed, 0, 999999],
                ] as Array<[string, number, (v: number) => void, number, number]>).map(([lab, val, set, mn, mx]) => (
                  <label key={lab} className="flex flex-col gap-1">
                    <span className="text-xs text-gray-500 font-medium">{lab}</span>
                    <input type="number" value={val} onChange={(e) => set(Number(e.target.value))} min={mn} max={mx}
                      className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 w-24 focus:outline-none focus:border-indigo-400" />
                  </label>
                ))}
              </div>
              <div className="flex items-center gap-3 flex-wrap">
                <button onClick={handleMICEPreview} disabled={miceLoading || selected.length === 0}
                  className="px-4 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50">
                  {miceLoading ? "Calculating…" : `Preview PMM for ${selected.length || ""} column(s)`}
                </button>
                <button onClick={runCompare} disabled={busy === "compare" || selected.length === 0}
                  className="px-4 py-2 text-sm font-medium border border-indigo-300 text-indigo-600 rounded-lg hover:bg-indigo-50 disabled:opacity-50">
                  {busy === "compare" ? "Comparing…" : "Compare: complete-case vs MICE"}
                </button>
                {selected.length === 0 && <p className="text-xs text-gray-400">Select columns above</p>}
              </div>

              {micePreviewResult?.result_text && (
                <div className="bg-emerald-50 border border-emerald-200 rounded-xl px-4 py-3 text-sm text-emerald-800">{micePreviewResult.result_text}</div>
              )}
              {micePreviewResult?.methods_text && (
                <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-4 py-3">
                  <div className="flex items-center justify-between gap-3 mb-1.5">
                    <p className="text-xs font-semibold text-indigo-800">Methods</p>
                    <button
                      onClick={() => navigator.clipboard.writeText(micePreviewResult.methods_text ?? "")}
                      className="text-[10px] px-2 py-0.5 rounded border border-indigo-200 text-indigo-600 hover:bg-white transition-colors"
                    >
                      Copy
                    </button>
                  </div>
                  <p className="text-xs text-indigo-800 leading-relaxed">{micePreviewResult.methods_text}</p>
                </div>
              )}
              {micePreviewResult?.export_rows && micePreviewResult.export_rows.length > 1 && (
                <>
                  <div className="overflow-auto rounded-lg border border-gray-200">
                    <table className="text-xs w-full">
                      <thead><tr className="bg-gray-50">{(micePreviewResult.export_rows[0] as unknown[]).map((h, i: number) => <th key={i} className="px-3 py-1.5 text-left text-gray-500 font-medium">{String(h)}</th>)}</tr></thead>
                      <tbody>{micePreviewResult.export_rows.slice(1).map((row, ri: number) => (
                        <tr key={ri} className="border-t border-gray-100">{(row as unknown[]).map((v, ci: number) => <td key={ci} className="px-3 py-1 text-gray-700">{(v as ReactNode) ?? "—"}</td>)}</tr>
                      ))}</tbody>
                    </table>
                  </div>
                  <ResultExporter
                    title="MICE_imputation_preview"
                    headers={micePreviewResult.export_rows[0].map(String)}
                    rows={micePreviewResult.export_rows.slice(1).map((row) => row.map((v) =>
                      v == null || typeof v === "string" || typeof v === "number" ? v : String(v)
                    ))}
                  />
                </>
              )}
              {micePreviewResult?.preview_rows && micePreviewResult.preview_rows.length > 0 && (
                <div className="space-y-3">
                  <p className="text-xs font-semibold text-gray-700">Previewed imputed values</p>
                  <div className="overflow-auto rounded-lg border border-gray-200 max-h-64">
                    <table className="text-xs w-full">
                      <thead>
                        <tr className="bg-gray-50 text-left text-gray-500">
                          <th className="px-3 py-1.5">Row</th>
                          <th className="px-3 py-1.5">Column</th>
                          <th className="px-3 py-1.5">Estimated value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {micePreviewResult.preview_rows.map((row) => (
                          <tr key={`${row.row_index}:${row.column}`} className="border-t border-gray-100">
                            <td className="px-3 py-1 text-gray-700">{row.row_index}</td>
                            <td className="px-3 py-1 text-gray-700">{row.column}</td>
                            <td className="px-3 py-1 text-gray-700">{String(row.imputed_value)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="flex justify-end">
                    <button
                      onClick={handleMICETransfer}
                      disabled={miceTransferLoading || micePreviewResult.applied}
                      className="px-4 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
                    >
                      {miceTransferLoading ? "Transferring…" : micePreviewResult.applied ? "Transferred" : "Transfer to original columns"}
                    </button>
                  </div>
                </div>
              )}

              {/* CCA vs MI comparison (sensitivity) */}
              {compare?.comparisons && (
                <div className="space-y-2">
                  <p className="text-xs font-semibold text-gray-700">Sensitivity — observed vs imputed distribution</p>
                  <div className="overflow-auto rounded-lg border border-gray-200">
                    <table className="text-xs w-full">
                      <thead><tr className="bg-gray-50 text-left text-gray-500">
                        <th className="px-3 py-1.5">Strategy</th><th className="px-3 py-1.5">Column</th>
                        <th className="px-3 py-1.5">Mean (obs→after)</th><th className="px-3 py-1.5">KS p</th>
                      </tr></thead>
                      <tbody>
                        {compare.comparisons.flatMap((cmp) => cmp.columns.map((c) => (
                          <tr key={`${cmp.strategy}:${c.col}`} className="border-t border-gray-100">
                            <td className="px-3 py-1 text-gray-700">{cmp.strategy}</td>
                            <td className="px-3 py-1 text-gray-700">{c.col}</td>
                            <td className="px-3 py-1 text-gray-700">{c.before?.mean ?? "—"} → {c.after?.mean ?? "—"}</td>
                            <td className={`px-3 py-1 ${c.ks_p != null && c.ks_p < 0.05 ? "text-red-500" : "text-gray-700"}`}>{c.ks_p ?? "—"}</td>
                          </tr>
                        )))}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-[10px] text-gray-400">KS p &lt; 0.05 = imputed distribution differs from observed (expected for MAR; large shifts warrant a closer look).</p>
                </div>
              )}
            </div>
          </div>
          </>
        )}
      </div>

      <div className={activeSubTab === "cleaning" ? "" : "hidden"} role="tabpanel">
        <CleaningTab sessionId={sid} columns={columns} numCols={numCols} />
      </div>

      <div className={activeSubTab === "reference" ? "space-y-5" : "hidden"} role="tabpanel">
        {activeSubTab === "reference" && (
          <>
        {err && <div className="bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-xs text-red-600">{err}</div>}
        {mutationNotice && <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-3 py-2 text-xs text-emerald-700">{mutationNotice}</div>}

        {missingInfo.length === 0 ? (
          <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-4 py-3 text-sm text-emerald-700">
            No missing values detected in any column.
          </div>
        ) : (
          <div className="border border-sky-200 rounded-xl overflow-hidden">
            <div className="px-5 py-3.5 bg-sky-50 border-b border-sky-100">
              <h3 className="text-sm font-semibold text-sky-800">Reference Dataset Imputation</h3>
              <p className="text-[11px] text-sky-500 mt-0.5">
                Upload a similar dataset, map reference variables to current variables, preview PMM/MICE estimates, then transfer values into the Data tab.
              </p>
            </div>
            <div className="px-5 py-4 space-y-4">
              <div className="grid gap-3 md:grid-cols-[1fr_1fr]">
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Current missing target</span>
                  <select
                    value={externalTargetName}
                    onChange={(e) => {
                      const nextTarget = e.target.value;
                      const nextReferenceTarget = externalReferenceColumns.find(
                        (c) => normColumnName(c.name) === normColumnName(nextTarget)
                      )?.name ?? "";
                      setExternalTarget(nextTarget);
                      setExternalReferenceTarget(nextReferenceTarget);
                      setExternalPredictors((prev) => prev.filter((name) => name !== nextReferenceTarget));
                      setExternalResult(null);
                    }}
                    className="text-sm border border-gray-300 rounded-lg px-3 py-2 bg-white focus:outline-none focus:border-sky-400"
                  >
                    {missingInfo.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Reference dataset</span>
                  <input
                    type="file"
                    accept=".csv,.xlsx,.xls,.sas7bdat,.sav,.dta"
                    onChange={(e) => void loadExternalReferenceColumns(e.target.files?.[0] ?? null)}
                    className="text-xs border border-gray-300 rounded-lg px-3 py-2 bg-white file:mr-3 file:border-0 file:bg-sky-50 file:text-sky-700 file:px-2 file:py-1 file:rounded"
                  />
                </label>
              </div>

              <div className="grid gap-3 md:grid-cols-[1fr_1fr]">
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Reference target match</span>
                  <select
                    value={externalReferenceTargetName}
                    onChange={(e) => {
                      setExternalReferenceTarget(e.target.value);
                      setExternalPredictors((prev) => prev.filter((name) => name !== e.target.value));
                      setExternalResult(null);
                    }}
                    disabled={!externalReferenceMeta}
                    className="text-sm border border-gray-300 rounded-lg px-3 py-2 bg-white focus:outline-none focus:border-sky-400 disabled:bg-gray-50"
                  >
                    <option value="">Select reference target</option>
                    {externalReferenceColumns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Method</span>
                  <select
                    value={externalMethod}
                    onChange={(e) => { setExternalMethod(e.target.value as "pmm" | "mice"); setExternalResult(null); }}
                    className="text-sm border border-gray-300 rounded-lg px-3 py-2 bg-white focus:outline-none focus:border-sky-400"
                  >
                    <option value="pmm">PMM</option>
                    <option value="mice">MICE / PMM</option>
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Stratify by (optional)</span>
                  <select
                    value={externalStratifyBy}
                    onChange={(e) => { setExternalStratifyBy(e.target.value); setExternalResult(null); }}
                    className="text-sm border border-gray-300 rounded-lg px-3 py-2 bg-white focus:outline-none focus:border-sky-400"
                  >
                    <option value="">No stratification</option>
                    {columns
                      .filter((c) => c.name !== externalTargetName)
                      .map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                  </select>
                </label>
              </div>

              <div>
                <div className="flex items-center justify-between gap-3 mb-2">
                  <p className="text-xs text-gray-500 font-medium">Reference predictors and current matches</p>
                  {externalReferenceMeta && (
                    <p className="text-[10px] text-gray-400">
                      {externalReferenceMeta.n_rows} rows, {externalReferenceMeta.columns.length} columns
                    </p>
                  )}
                </div>
                <div className="max-h-64 overflow-y-auto rounded-lg border border-gray-200">
                  {externalLoading === "columns" && (
                    <div className="px-3 py-2 text-xs text-gray-400">Reading columns...</div>
                  )}
                  {!externalLoading && externalFile && externalPredictorColumns.length === 0 && (
                    <div className="px-3 py-2 text-xs text-gray-400">No reference predictors found.</div>
                  )}
                  {!externalFile && (
                    <div className="px-3 py-2 text-xs text-gray-400">Upload reference dataset first.</div>
                  )}
                  {externalPredictorColumns.map((c) => {
                    const currentMatch = externalPredictorMappings[c.name] || currentColumnByNorm.get(normColumnName(c.name)) || "";
                    const checked = externalPredictors.includes(c.name);
                    return (
                      <div key={c.name} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-3 px-3 py-2 border-t first:border-t-0 border-gray-100 items-center">
                        <label className="flex items-center gap-2 text-xs text-gray-700 min-w-0">
                          <input
                            type="checkbox"
                            className="accent-sky-600"
                            checked={checked}
                            onChange={() => {
                              setExternalResult(null);
                              setExternalPredictors((prev) =>
                                prev.includes(c.name) ? prev.filter((x) => x !== c.name) : [...prev, c.name]
                              );
                            }}
                          />
                          <span className="truncate">{c.name}</span>
                        </label>
                        <select
                          value={currentMatch}
                          onChange={(e) => {
                            setExternalResult(null);
                            setExternalPredictorMappings((prev) => ({ ...prev, [c.name]: e.target.value }));
                          }}
                          className="text-xs border border-gray-300 rounded-md px-2 py-1 bg-white focus:outline-none focus:border-sky-400"
                        >
                          <option value="">Match current variable</option>
                          {columns
                            .filter((col) => col.name !== externalTargetName)
                            .map((col) => <option key={col.name} value={col.name}>{col.name}</option>)}
                        </select>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="flex items-center gap-3 justify-end">
                <button
                  onClick={runExternalPreview}
                  disabled={externalLoading !== null}
                  className="px-4 py-2 text-sm font-medium border border-sky-300 text-sky-700 rounded-lg hover:bg-sky-50 disabled:opacity-50"
                >
                  {externalLoading === "preview" ? "Calculating…" : "Preview target estimates"}
                </button>
              </div>

              {externalResult?.result_text && (
                <div className="bg-sky-50 border border-sky-200 rounded-xl px-4 py-3 text-sm text-sky-800">
                  {externalResult.result_text}
                </div>
              )}
              {externalResult?.warnings?.map((w) => (
                <div key={w} className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-[11px] text-amber-700">{w}</div>
              ))}
              {externalResult?.preview_rows && externalResult.preview_rows.length > 0 && (
                <div className="space-y-3">
                  <div className="overflow-auto rounded-lg border border-gray-200">
                    <table className="text-xs w-full">
                      <thead>
                        <tr className="bg-gray-50 text-left text-gray-500">
                          <th className="px-3 py-1.5">Row</th>
                          <th className="px-3 py-1.5">Estimated value</th>
                          <th className="px-3 py-1.5">Predictors missing</th>
                          {externalResult.preview_rows.some((r) => r.stratum) && (
                            <th className="px-3 py-1.5">Stratum</th>
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        {externalResult.preview_rows.map((row) => (
                          <tr key={row.row_index} className="border-t border-gray-100">
                            <td className="px-3 py-1 text-gray-700">{row.row_index}</td>
                            <td className="px-3 py-1 text-gray-700">{String(row.imputed_value)}</td>
                            <td className="px-3 py-1 text-gray-700">{row.predictors_missing}</td>
                            {externalResult.preview_rows.some((r) => r.stratum) && (
                              <td className="px-3 py-1 text-gray-700">{row.stratum ?? ""}</td>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="flex justify-end">
                    <button
                      onClick={applyExternalImputation}
                      disabled={externalLoading !== null || externalResult.applied}
                      className="px-4 py-2 text-sm font-medium bg-sky-600 text-white rounded-lg hover:bg-sky-700 disabled:opacity-50"
                    >
                      {externalLoading === "apply" ? "Transferring…" : externalResult.applied ? "Transferred" : "Transfer data"}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
          </>
        )}
      </div>

      <div className={activeSubTab === "mnar" ? "space-y-5" : "hidden"} role="tabpanel">
        {activeSubTab === "mnar" && (
          <>
        {err && <div className="bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-xs text-red-600">{err}</div>}

        {missingInfo.length === 0 ? (
          <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-4 py-3 text-sm text-emerald-700">
            No missing values detected in any column.
          </div>
        ) : (
          <div className="border border-purple-200 rounded-xl overflow-hidden">
            <div className="px-5 py-3.5 bg-purple-50 border-b border-purple-100">
              <h3 className="text-sm font-semibold text-purple-800">MNAR Sensitivity Analysis</h3>
              <p className="text-[11px] text-purple-500 mt-0.5">
                Pattern-mixture delta adjustment, selection model, ISNI and imputation diagnostics for data that may be
                missing not at random. Each sub-analysis is reported separately; ones that cannot be computed explain why.
              </p>
            </div>
            <div className="px-5 py-4 space-y-4">
              <div>
                <p className="text-xs text-gray-500 font-medium mb-2">Variables to analyse</p>
                <div
                  role="group"
                  aria-label="Variables to analyse"
                  className="max-h-48 overflow-y-auto rounded-lg border border-gray-200"
                >
                  {missingInfo.map((m) => (
                    <label
                      key={m.name}
                      className="flex items-center gap-2 px-3 py-2 border-t first:border-t-0 border-gray-100 text-xs text-gray-700"
                    >
                      <input
                        type="checkbox"
                        className="accent-purple-600"
                        checked={mnarColumns.includes(m.name)}
                        onChange={() => toggleMnarColumn(m.name)}
                      />
                      <span className="truncate">{m.name}</span>
                      <span className="text-[10px] text-gray-400">{m.pct.toFixed(1)}% missing</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="grid gap-3 md:grid-cols-[1fr_1fr]">
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Delta values</span>
                  <input
                    type="text"
                    value={mnarDeltaText}
                    onChange={(e) => { setMnarDeltaText(e.target.value); setMnarResult(null); }}
                    placeholder={MNAR_DEFAULT_DELTAS}
                    className="text-sm border border-gray-300 rounded-lg px-3 py-2 bg-white focus:outline-none focus:border-purple-400"
                  />
                  <span className="text-[10px] text-gray-400">
                    Comma-separated shifts applied to originally missing cells; 0 is the MAR reference scenario.
                  </span>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Model type</span>
                  <select
                    value={mnarModelType}
                    onChange={(e) => { setMnarModelType(e.target.value as MnarModelType); setMnarResult(null); }}
                    className="text-sm border border-gray-300 rounded-lg px-3 py-2 bg-white focus:outline-none focus:border-purple-400"
                  >
                    <option value="logistic">Logistic</option>
                    <option value="linear">Linear</option>
                    <option value="cox">Cox</option>
                  </select>
                  <span className="text-[10px] text-gray-400">
                    Applies to the model-based delta sensitivity below. Pick an outcome and
                    predictors to activate it.
                  </span>
                </label>
              </div>

              {/* Outcome model — optional, but without it the backend returns
                  placeholders for delta sensitivity, Heckman and ISNI. */}
              <div className="grid gap-3 md:grid-cols-[1fr_1.4fr]">
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Outcome (optional)</span>
                  <select
                    value={mnarOutcome}
                    onChange={(e) => { setMnarOutcome(e.target.value); setMnarResult(null); }}
                    className="text-sm border border-gray-300 rounded-lg px-3 py-2 bg-white focus:outline-none focus:border-purple-400"
                  >
                    <option value="">— none —</option>
                    {columns.map((c) => (
                      <option key={c.name} value={c.name}>{c.name}</option>
                    ))}
                  </select>
                  <span className="text-[10px] text-gray-400">
                    Unlocks model-based delta sensitivity, Heckman and ISNI.
                  </span>
                </label>
                <div>
                  <p className="text-xs text-gray-500 font-medium mb-2">Outcome-model predictors</p>
                  <div
                    role="group"
                    aria-label="Outcome-model predictors"
                    className="max-h-28 overflow-y-auto border border-gray-200 rounded-lg p-2 grid grid-cols-2 gap-1 bg-white"
                  >
                    {columns.filter((c) => c.name !== mnarOutcome).map((c) => (
                      <label key={c.name} className="flex items-center gap-1.5 text-xs text-gray-600">
                        <input
                          type="checkbox"
                          className="accent-purple-500"
                          checked={mnarPredictors.includes(c.name)}
                          onChange={() => {
                            setMnarResult(null);
                            setMnarPredictors((p) =>
                              p.includes(c.name) ? p.filter((x) => x !== c.name) : [...p, c.name]);
                          }}
                        />
                        <span className="truncate">{c.name}</span>
                      </label>
                    ))}
                  </div>
                  {mnarOutcome && mnarPredictors.length === 0 && (
                    <p className="text-[10px] text-amber-600 mt-1">
                      Select at least one predictor, or the outcome model is ignored.
                    </p>
                  )}
                </div>
              </div>

              <div className="flex items-center gap-3 flex-wrap">
                <button
                  onClick={runMnar}
                  disabled={mnarLoading || mnarColumns.length === 0}
                  className="px-4 py-2 text-sm font-medium bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50"
                >
                  {mnarLoading ? "Running MNAR sensitivity…" : `Run MNAR sensitivity${mnarColumns.length ? ` (${mnarColumns.length})` : ""}`}
                </button>
                {mnarColumns.length === 0 && <p className="text-xs text-gray-400">Select variables above</p>}
                <p className="text-[10px] text-gray-400">
                  Runs MICE plus several models — this can take a minute or more on larger datasets.
                </p>
              </div>
              {mnarLoading && (
                <div className="bg-purple-50 border border-purple-200 rounded-lg px-3 py-2 text-[11px] text-purple-700">
                  Running multiple imputation and sensitivity models… please keep this tab open.
                </div>
              )}

              {mnarResult && (
                <div className="space-y-3">
                  {mnarResult.result_text && (
                    <div className="bg-emerald-50 border border-emerald-200 rounded-xl px-4 py-3 text-sm text-emerald-800">
                      {mnarResult.result_text}
                    </div>
                  )}
                  {mnarResult.warnings?.map((w) => (
                    <div key={w} className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-[11px] text-amber-700">{w}</div>
                  ))}

                  <MnarBlock
                    title="Pattern-mixture delta scenarios"
                    block={mnarResult.pattern_mixture_model}
                    fallback="Pattern-mixture model was not returned."
                  >
                    {mnarResult.pattern_mixture_model?.scenarios?.length ? (
                      <div className="overflow-auto rounded-lg border border-gray-200">
                        <table className="text-xs w-full">
                          <thead>
                            <tr className="bg-gray-50 text-left text-gray-500">
                              <th className="px-3 py-1.5">Delta</th>
                              {mnarAnalysedColumns.map((c) => <th key={c} className="px-3 py-1.5">{c} (pooled mean)</th>)}
                            </tr>
                          </thead>
                          <tbody>
                            {mnarResult.pattern_mixture_model.scenarios.map((s) => (
                              <tr key={s.delta} className="border-t border-gray-100">
                                <td className="px-3 py-1 text-gray-700">{s.delta}</td>
                                {mnarAnalysedColumns.map((c) => (
                                  <td key={c} className="px-3 py-1 text-gray-700">{fmtNum(s.pooled_means?.[c])}</td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : <p className="text-[11px] text-gray-400">No delta scenarios returned.</p>}
                    {mnarResult.pattern_mixture_model?.interpretation && (
                      <p className="text-[10px] text-gray-400">{mnarResult.pattern_mixture_model.interpretation}</p>
                    )}
                  </MnarBlock>

                  <MnarBlock
                    title="Model-based delta sensitivity"
                    block={mnarResult.model_delta_sensitivity}
                    fallback="Not run — requires an outcome model with predictors."
                  >
                    <div className="overflow-auto rounded-lg border border-gray-200">
                      <table className="text-xs w-full">
                        <thead>
                          <tr className="bg-gray-50 text-left text-gray-500">
                            <th className="px-3 py-1.5">Delta</th>
                            <th className="px-3 py-1.5">Estimate</th>
                            <th className="px-3 py-1.5">SE</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(mnarResult.model_delta_sensitivity?.results ?? []).map((r, i) => (
                            <tr key={i} className="border-t border-gray-100">
                              <td className="px-3 py-1 text-gray-700">{r.delta}</td>
                              <td className="px-3 py-1 text-gray-700">
                                {r.error ?? fmtNum(r.estimate ?? r.log_odds ?? r.hr)}
                              </td>
                              <td className="px-3 py-1 text-gray-700">{fmtNum(r.se)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </MnarBlock>

                  <MnarBlock
                    title="Heckman selection model"
                    block={mnarResult.heckman_selection_model}
                    fallback="Heckman selection model was not run."
                  >
                    <p className="text-[11px] text-gray-600">
                      Observed outcomes: {mnarResult.heckman_selection_model?.n_observed_outcome ?? "—"} / {mnarResult.heckman_selection_model?.n_total ?? "—"}
                      {" · "}Selection rate: {fmtNum(mnarResult.heckman_selection_model?.selection_rate)}
                      {" · "}Inverse Mills ratio p: {fmtNum(mnarResult.heckman_selection_model?.inverse_mills_ratio_p)}
                    </p>
                    {mnarResult.heckman_selection_model?.outcome_coefficients?.length ? (
                      <div className="overflow-auto rounded-lg border border-gray-200">
                        <table className="text-xs w-full">
                          <thead>
                            <tr className="bg-gray-50 text-left text-gray-500">
                              <th className="px-3 py-1.5">Variable</th>
                              <th className="px-3 py-1.5">Estimate</th>
                              <th className="px-3 py-1.5">SE</th>
                              <th className="px-3 py-1.5">p</th>
                            </tr>
                          </thead>
                          <tbody>
                            {mnarResult.heckman_selection_model.outcome_coefficients.map((c) => (
                              <tr key={c.variable} className="border-t border-gray-100">
                                <td className="px-3 py-1 text-gray-700">{c.variable}</td>
                                <td className="px-3 py-1 text-gray-700">{fmtNum(c.estimate)}</td>
                                <td className="px-3 py-1 text-gray-700">{fmtNum(c.se)}</td>
                                <td className="px-3 py-1 text-gray-700">{fmtP(Number(c.p))}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : null}
                  </MnarBlock>

                  <MnarBlock
                    title="ISNI — index of sensitivity to non-ignorability"
                    block={mnarResult.isni}
                    fallback="ISNI was not computed."
                  >
                    <div className="overflow-auto rounded-lg border border-gray-200">
                      <table className="text-xs w-full">
                        <thead>
                          <tr className="bg-gray-50 text-left text-gray-500">
                            <th className="px-3 py-1.5">Variable</th>
                            <th className="px-3 py-1.5">ISNI</th>
                            <th className="px-3 py-1.5">High sensitivity</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(mnarResult.isni?.indices ?? []).map((r) => (
                            <tr key={r.variable} className="border-t border-gray-100">
                              <td className="px-3 py-1 text-gray-700">{r.variable}</td>
                              <td className="px-3 py-1 text-gray-700">{r.error ?? fmtNum(r.isni)}</td>
                              <td className={`px-3 py-1 ${r.high_sensitivity ? "text-red-500" : "text-gray-700"}`}>
                                {r.high_sensitivity ? "Yes" : "No"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </MnarBlock>

                  <MnarBlock
                    title="MICE convergence diagnostics"
                    block={mnarResult.mice_convergence_diagnostics}
                    fallback="Convergence diagnostics were not returned."
                  >
                    <div className="overflow-auto rounded-lg border border-gray-200">
                      <table className="text-xs w-full">
                        <thead>
                          <tr className="bg-gray-50 text-left text-gray-500">
                            <th className="px-3 py-1.5">Variable</th>
                            <th className="px-3 py-1.5">R-hat proxy</th>
                            <th className="px-3 py-1.5">Converged</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(mnarResult.mice_convergence_diagnostics?.variables ?? {}).map(([name, v]) => (
                            <tr key={name} className="border-t border-gray-100">
                              <td className="px-3 py-1 text-gray-700">{name}</td>
                              <td className="px-3 py-1 text-gray-700">{fmtNum(v.r_hat_proxy)}</td>
                              <td className={`px-3 py-1 ${v.converged ? "text-gray-700" : "text-red-500"}`}>
                                {v.converged ? "Yes" : "No"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {mnarResult.mice_convergence_diagnostics?.warning && (
                      <p className="text-[10px] text-gray-400">{mnarResult.mice_convergence_diagnostics.warning}</p>
                    )}
                  </MnarBlock>

                  <MnarBlock
                    title="Imputation model diagnostics"
                    block={mnarResult.imputation_model_diagnostics}
                    fallback="Imputation diagnostics were not returned."
                  >
                    {(mnarResult.imputation_model_diagnostics?.checks ?? []).map((c) => (
                      c.available === false ? (
                        <div key={c.variable} className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-[11px] text-amber-700">
                          <span className="font-semibold">{c.variable} — </span>{c.reason || "Not available."}
                        </div>
                      ) : (
                        <p key={c.variable} className="text-[11px] text-gray-600">
                          <span className="font-semibold">{c.variable}:</span> observed mean {fmtNum(c.observed_mean)} → imputed mean {fmtNum(c.imputed_mean)}
                          {" · "}KS <i>p</i> {c.ks_p == null ? "—" : fmtP(Number(c.ks_p))}
                          {c.flag_distribution_shift ? " · distribution shift flagged" : ""}
                        </p>
                      )
                    ))}
                  </MnarBlock>

                  <MnarBlock
                    title="Congeniality assessment"
                    block={mnarResult.congeniality_assessment}
                    fallback="Congeniality assessment was not returned."
                  >
                    <p className="text-[11px] text-gray-600">
                      <span className="font-semibold">{mnarResult.congeniality_assessment?.congenial ? "Congenial" : "Not congenial"}.</span>{" "}
                      {mnarResult.congeniality_assessment?.recommendation}
                    </p>
                    {mnarResult.congeniality_assessment?.analysis_variables_missing_from_imputation?.length ? (
                      <p className="text-[11px] text-gray-600">
                        Missing from imputation model: {mnarResult.congeniality_assessment.analysis_variables_missing_from_imputation.join(", ")}
                      </p>
                    ) : null}
                  </MnarBlock>

                  <MnarBlock
                    title="Auxiliary variable guidance"
                    block={mnarResult.auxiliary_variable_guidance}
                    fallback="Auxiliary variable guidance was not returned."
                  >
                    {mnarResult.auxiliary_variable_guidance?.recommended_auxiliary_variables?.length ? (
                      <div className="overflow-auto rounded-lg border border-gray-200">
                        <table className="text-xs w-full">
                          <thead>
                            <tr className="bg-gray-50 text-left text-gray-500">
                              <th className="px-3 py-1.5">Target</th>
                              <th className="px-3 py-1.5">Candidate</th>
                              <th className="px-3 py-1.5">Priority</th>
                            </tr>
                          </thead>
                          <tbody>
                            {mnarResult.auxiliary_variable_guidance.recommended_auxiliary_variables.map((r) => (
                              <tr key={`${r.target}:${r.candidate}`} className="border-t border-gray-100">
                                <td className="px-3 py-1 text-gray-700">{r.target}</td>
                                <td className="px-3 py-1 text-gray-700">{r.candidate}</td>
                                <td className="px-3 py-1 text-gray-700">{fmtNum(r.priority_score)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : <p className="text-[11px] text-gray-400">No auxiliary variables recommended.</p>}
                  </MnarBlock>

                  <MnarBlock
                    title="Survival MNAR sensitivity (informative censoring)"
                    block={mnarResult.survival_mnar_sensitivity}
                    fallback="Survival MNAR sensitivity was not run."
                  >
                    <div className="overflow-auto rounded-lg border border-gray-200">
                      <table className="text-xs w-full">
                        <thead>
                          <tr className="bg-gray-50 text-left text-gray-500">
                            <th className="px-3 py-1.5">Delta</th>
                            <th className="px-3 py-1.5">Censored weight ×</th>
                            <th className="px-3 py-1.5">Concordance</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(mnarResult.survival_mnar_sensitivity?.results ?? []).map((r) => (
                            <tr key={r.delta} className="border-t border-gray-100">
                              <td className="px-3 py-1 text-gray-700">{r.delta}</td>
                              <td className="px-3 py-1 text-gray-700">{r.error ?? fmtNum(r.censored_weight_multiplier)}</td>
                              <td className="px-3 py-1 text-gray-700">{fmtNum(r.concordance)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </MnarBlock>

                  {mnarResult.survival_specific_imputation?.enabled && (
                    <div className="bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-[11px] text-gray-600">
                      Survival-specific imputation auxiliaries added:{" "}
                      {mnarResult.survival_specific_imputation.auxiliary_variables?.join(", ") || "none"}
                    </div>
                  )}
                  {mnarResult.passive_imputation?.formulas && Object.keys(mnarResult.passive_imputation.formulas).length > 0 && (
                    <div className="bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-[11px] text-gray-600">
                      Passive imputation formulas:{" "}
                      {Object.entries(mnarResult.passive_imputation.formulas).map(([k, v]) => `${k} = ${v}`).join("; ")}
                    </div>
                  )}

                  {mnarResult.assumptions?.length ? (
                    <div className="border border-gray-200 rounded-lg px-4 py-3 space-y-1">
                      <p className="text-xs font-semibold text-gray-700">Assumptions</p>
                      <ul className="ml-3 list-disc text-[11px] text-gray-600">
                        {mnarResult.assumptions.map((a) => (
                          <li key={a.name}>
                            <span className="font-medium">{a.name}</span> ({a.met ? "met" : "not met"}): {a.detail}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : null}

                  {mnarResult.r_code && (
                    <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-4 py-3">
                      <div className="flex items-center justify-between gap-3 mb-1.5">
                        <p className="text-xs font-semibold text-indigo-800">R code</p>
                        <button
                          onClick={() => navigator.clipboard.writeText(mnarResult.r_code ?? "")}
                          className="text-[10px] px-2 py-0.5 rounded border border-indigo-200 text-indigo-600 hover:bg-white transition-colors"
                        >
                          Copy
                        </button>
                      </div>
                      <pre className="text-[11px] text-indigo-800 whitespace-pre-wrap">{mnarResult.r_code}</pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
          </>
        )}
      </div>
    </div>
  );
}
