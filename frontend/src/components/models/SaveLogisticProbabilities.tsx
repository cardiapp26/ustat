import { useState } from "react";
import { saveLogisticProbabilities } from "../../api";
import { staleExportTitle, useStaleGuard } from "../../lib/staleGuard";
import { runColumnStructureMutation, useStore } from "../../store";
import type { ModelResult } from "./shared";

interface Props {
  sessionId: string;
  outcome: string;
  fit: NonNullable<ModelResult["probability_fit"]>;
}

export default function SaveLogisticProbabilities({ sessionId, outcome, fit }: Props) {
  const { stale, reason } = useStaleGuard();
  const [busy, setBusy] = useState(false);
  const [savedColumn, setSavedColumn] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const save = async () => {
    if (stale || busy || savedColumn) return;
    setBusy(true);
    setError(null);
    try {
      await runColumnStructureMutation(sessionId, async () => {
        const { data } = await saveLogisticProbabilities(sessionId, fit.token);
        if (useStore.getState().session?.session_id !== sessionId) return;
        useStore.setState((state) => ({
          session: { ...state.session!, rows: data.rows, columns: data.columns, preview: data.preview },
          dataVersion: state.dataVersion + 1,
          undoDepth: data.undo_depth,
          redoDepth: 0,
          columnMutationRedo: [],
        }));
        setSavedColumn(data.name);
      });
    } catch (cause) {
      const detail = (cause as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
      setError(typeof detail === "string" ? detail : cause instanceof Error ? cause.message : "Could not save probabilities.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mt-3 space-y-1">
      <button
        type="button"
        className="btn-secondary"
        onClick={save}
        disabled={stale || busy || savedColumn !== null}
        title={stale ? staleExportTitle(reason) : undefined}
      >
        {busy ? "Saving…" : "Save predicted probabilities"}
      </button>
      <p className="text-xs text-gray-500">
        Probabilities for {outcome} = {fit.event_level}. Rows outside the fit stay blank.
        {fit.source === "first imputed dataset" && " Predictions use the first imputed dataset."}
      </p>
      {savedColumn && (
        <p role="status" className="text-xs text-emerald-700">
          Saved {savedColumn}. Select it as the probability column in Binary DCA.
        </p>
      )}
      {error && <p role="alert" className="text-xs text-red-700">{error}</p>}
    </div>
  );
}
