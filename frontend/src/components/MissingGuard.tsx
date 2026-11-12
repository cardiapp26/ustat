/* eslint-disable react-refresh/only-export-components -- shared hook + component co-located by design */
/**
 * MissingGuard — Missing-data awareness component.
 *
 * Usage:
 *   <MissingGuard sessionId={id} columns={[...]} imputation={imp} onImputation={setImp}>
 *     {(imp) => <RunButton onClick={() => run(imp)} />}
 *   </MissingGuard>
 *
 * Shows nothing when there are no missing values.
 * Shows an amber/red warning with per-column breakdown and a strategy picker
 * when missing data is found. The children render-prop always gets the
 * currently selected imputation strategy so callers don't need to track it.
 */

import { useEffect, useState, useCallback, useRef } from "react";
import { getMissing } from "../api";
import { useStore } from "../store";
import { Tip } from "./Tip";

// ── Types ──────────────────────────────────────────────────────────────────────

export type ImputationStrategy = "listwise" | "median" | "mice";

export interface PerColumnInfo {
  count: number;
  pct: number;
  raw_count?: number;
  n_implausible?: number;
}

export interface MissingInfo {
  total_rows: number;
  rows_affected: number;
  pct_affected: number;
  raw_rows_affected?: number;
  per_column: Record<string, PerColumnInfo>;
}

interface MissingGuardProps {
  sessionId: string;
  columns: string[];                              // columns to check
  imputation: ImputationStrategy;
  onImputation: (s: ImputationStrategy) => void;
  /** Render-prop: receives current imputation and renders the run button / content. */
  children?: React.ReactNode;
}

// ── Strategy descriptions ──────────────────────────────────────────────────────

const STRATEGIES: {
  value: ImputationStrategy;
  label: string;
  desc: string;
  tip: string;
}[] = [
  {
    value: "listwise",
    label: "Listwise deletion",
    desc: "Removes every row with any missing value. Default in R/SPSS. Unbiased when data are missing completely at random (MCAR), but shrinks sample size.",
    tip: "Drop every row that has at least one missing value in the selected columns. R and SPSS default. Safe but reduces sample size.",
  },
  {
    value: "median",
    label: "Median imputation",
    desc: "Fills each missing cell with its column median. Robust to outliers — preferred for skewed clinical variables (Troponin, CRP, BMI). Fast, but underestimates variance.",
    tip: "Replace missing values with the column median. Recommended for skewed clinical variables (e.g. Troponin, CRP) because the median is not affected by extreme outliers.",
  },
  {
    value: "mice",
    label: "MICE",
    desc: "Multiple Imputation by Chained Equations. Predicts each missing value from all other variables using regression — statistically rigorous and minimises bias. Slower; falls back to median if it fails.",
    tip: "Multiple Imputation by Chained Equations. Predicts each missing value from all other columns using regression. Most accurate but slowest. Falls back to median if it fails.",
  },
];

// ── Hook ───────────────────────────────────────────────────────────────────────

export function useMissing(sessionId: string, columns: string[], refreshKey = 0) {
  const [info, setInfo] = useState<MissingInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);
  const caseFilter = useStore((s) => s.caseFilter);
  const caseWeight = useStore((s) => s.caseWeight);
  const splitFile = useStore((s) => s.splitFile);

  const refetch = useCallback(() => {
    const id = ++requestId.current;
    setInfo(null);
    setError(null);
    if (!sessionId || columns.length === 0) {
      setLoading(false);
      return;
    }
    setLoading(true);
    getMissing(sessionId, columns)
      .then((r) => {
        if (id === requestId.current) setInfo(r.data as MissingInfo);
      })
      .catch((e) => {
        if (id === requestId.current) setError(e?.response?.data?.detail ?? "Failed to check missing data");
      })
      .finally(() => {
        if (id === requestId.current) setLoading(false);
      });
  }, [sessionId, columns.join(","), refreshKey, caseFilter, caseWeight, splitFile]);  // eslint-disable-line

  useEffect(() => {
    refetch();
    return () => { requestId.current += 1; };
  }, [refetch]);

  return { info, loading, error, refetch };
}

// ── Component ─────────────────────────────────────────────────────────────────

export function MissingGuard({
  sessionId,
  columns,
  imputation,
  onImputation,
  children,
}: MissingGuardProps) {
  const dataVersion = useStore((s) => s.dataVersion);
  const { info, loading } = useMissing(sessionId, columns, dataVersion);
  const missingRows = info?.raw_rows_affected ?? info?.rows_affected ?? 0;
  const flaggedColumns = Object.entries(info?.per_column ?? {}).filter(([, col]) => (col.n_implausible ?? 0) > 0);

  // `children` (the run button) stays in the same slot in every state. When it
  // moved between the loading, clean and warning layouts React remounted it,
  // so a click that landed as the missing-data check finished hit the old,
  // detached button and was lost.
  return (
    <>
      {loading && (
        <div className="text-xs text-gray-400 animate-pulse mb-2">Checking missing data…</div>
      )}
      {!loading && flaggedColumns.length > 0 && (
        <div className="mb-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          <p>Flagged values are separate from missing values. Listwise deletion keeps them. MICE can treat them as missing in supported analyses.</p>
          <ul>
            {flaggedColumns.map(([name, { n_implausible }]) => (
              <li key={name}>{name}: {n_implausible} flagged {n_implausible === 1 ? "value" : "values"}</li>
            ))}
          </ul>
        </div>
      )}
      {!loading && info && missingRows > 0 && (
        <MissingWarning info={info} missingRows={missingRows} imputation={imputation} onImputation={onImputation} />
      )}
      {children}
    </>
  );
}

interface MissingWarningProps {
  info: MissingInfo;
  missingRows: number;
  imputation: ImputationStrategy;
  onImputation: (s: ImputationStrategy) => void;
}

function percentage(count: number, total: number) {
  return total > 0 ? Number((100 * count / total).toFixed(1)) : 0;
}

function MissingWarning({ info, missingRows, imputation, onImputation }: MissingWarningProps) {
  const pct = percentage(missingRows, info.total_rows);
  const severity = pct >= 20 ? "red" : "amber";

  return (
    <div className="space-y-3 mb-3">
      {/* ── Warning banner ── */}
      <div
        className={`rounded-lg border px-3 py-2.5 text-xs leading-relaxed ${
          severity === "red"
            ? "bg-red-50 border-red-200 text-red-800"
            : "bg-amber-50 border-amber-200 text-amber-800"
        }`}
      >
        <div className="flex items-start gap-2">
          <span className="mt-0.5 flex-shrink-0 text-base">
            {severity === "red" ? "⚠️" : "⚠️"}
          </span>
          <div className="space-y-1 w-full">
            <p className="font-semibold">
              {missingRows} of {info.total_rows} rows have missing values ({pct}%)
            </p>

            {/* Per-column breakdown */}
            <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 mt-1">
              {Object.entries(info.per_column)
                .map(([col, v]) => [col, v.raw_count ?? v.count] as const)
                .filter(([, count]) => count > 0)
                .map(([col, count]) => (
                  <div key={col} className="flex justify-between">
                    <span className="truncate max-w-[120px]" title={col}>
                      {col}
                    </span>
                    <span className="font-mono ml-2 opacity-80">
                      {count} ({percentage(count, info.total_rows)}%)
                    </span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── Strategy picker ── */}
      <div className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2.5 space-y-1.5">
        <p className="text-xs font-medium text-gray-600 mb-1">
          How should missing values be handled?
          <Tip
            wide
            text="Choose a strategy before running the analysis. Listwise is the safest default; Median is better for skewed clinical variables; MICE is the most statistically rigorous."
          />
        </p>
        {STRATEGIES.map((s) => (
          <label
            key={s.value}
            className="flex items-start gap-2 cursor-pointer group"
          >
            <input
              type="radio"
              name="imputation"
              value={s.value}
              checked={imputation === s.value}
              onChange={() => onImputation(s.value)}
              className="mt-0.5 accent-indigo-600 flex-shrink-0"
            />
            <span className="text-xs leading-snug">
              <span className="font-medium text-gray-800 group-hover:text-gray-900">{s.label}</span>
              <Tip text={s.tip} wide />
              {imputation === s.value && (
                <p className="text-[10px] text-gray-500 mt-0.5 leading-relaxed">{s.desc}</p>
              )}
            </span>
          </label>
        ))}
      </div>
    </div>
  );
}
