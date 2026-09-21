/** Small presentational pieces shared by the Models panel's forms and result
 *  cards. Extracted from ModelsPanel.tsx. */
import { useStore, paletteOf } from "../../store";

const _pal = () => paletteOf(useStore.getState().plotTheme);

/** Send-to-Forest-Builder control, shared by both forest cards. */
export function ForestBuilderButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      title="Add these estimates as rows in the Forest Builder, keeping whatever is already there — so a figure can combine several fits (e.g. a continuous exposure and its dichotomised form, which cannot share one model)."
      className="flex-shrink-0 whitespace-nowrap rounded-lg border border-indigo-200 bg-indigo-50 px-2.5 py-1 text-xs font-medium text-indigo-700 transition-colors hover:bg-indigo-100"
    >
      → Forest Builder
    </button>
  );
}

/** Sparkline mini distribution bar shown beside column pickers. */
export function SparklineMini({ data, type }: { data: number[]; type: string }) {
  const W = 44, H = 14;
  if (!data || data.length === 0) return null;
  const max = Math.max(...data);
  if (max === 0) return null;
  if (type === "numeric") {
    const bw = W / data.length;
    return (
      <svg width={W} height={H} style={{ display: "block", flexShrink: 0 }}>
        {data.map((v, i) => {
          const bh = Math.max(1, (v / max) * H);
          return <rect key={i} x={i * bw} y={H - bh} width={Math.max(bw - 0.5, 0.5)} height={bh} fill="#ef4444" opacity={0.65} />;
        })}
      </svg>
    );
  }
  // categorical → stacked horizontal proportion bars
  const total = data.reduce((a, b) => a + b, 0);
  const CATS  = _pal();
  // Cumulative left offset per segment, precomputed immutably (no reassigned
  // closure variable inside the render map) — react-hooks/immutability.
  const offsets = data.reduce<number[]>(
    (acc, _v, i) => [...acc, i === 0 ? 0 : acc[i - 1] + (data[i - 1] / total) * W],
    [],
  );
  return (
    <svg width={W} height={H} style={{ display: "block", flexShrink: 0 }}>
      {data.map((v, i) => {
        const w = (v / total) * W;
        return <rect key={i} x={offsets[i]} y={0} width={Math.max(w, 0.5)} height={H} fill={CATS[i % CATS.length]} />;
      })}
    </svg>
  );
}

/** How a model's intervals were computed, stated beside the table that shows
 *  them. Firth fits report Wald intervals where R's logistf defaults to
 *  penalised profile likelihood; a note that only lived in the API response
 *  never reached the reader. */
export function CiMethodNote({ method, note }: { method?: string; note?: string }) {
  if (!note) return null;
  return (
    <div role="note" className="mb-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] leading-snug text-amber-800">
      {method && <span className="font-semibold">{method} intervals. </span>}
      {note}
    </div>
  );
}
