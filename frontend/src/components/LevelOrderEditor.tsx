/** Low-to-high order of a categorical column's levels, stored in the Data
 *  Dictionary as metadata `level_order`. Ordinal logistic regression,
 *  Jonckheere-Terpstra and Cochran-Armitage read it on the server; without it
 *  text categories have no order the server will assume (an ordinal model
 *  refuses to run rather than sort them alphabetically). */

import { arrangeLevels, mergeLevelOrder, swap } from "../lib/levelOrder";

interface LevelOrderEditorProps {
  column: string;
  values: string[];
  order: string[] | undefined;
  labels: Record<string, string>;
  /** New saved order; an empty list clears it. */
  onChange: (order: string[]) => void;
}

export default function LevelOrderEditor({ column, values, order, labels, onChange }: LevelOrderEditorProps) {
  if (values.length < 2) return null;
  const isSet = !!order && order.length > 0;
  const { levels, unplaced } = arrangeLevels(values, order);
  const moveTo = (i: number, j: number) => onChange(mergeLevelOrder(order, swap(levels, i, j)));

  return (
    <section aria-label={`Category order for ${column}`} className="mt-3 border-t border-indigo-100 pt-3">
      <div className="mb-1 flex items-center justify-between">
        <p className="text-xs font-semibold text-gray-700">Category order (low → high)</p>
        {isSet ? (
          <button onClick={() => onChange([])}
            className="rounded border border-orange-200 px-2 py-0.5 text-[10px] text-orange-600 hover:bg-orange-50">
            Clear order
          </button>
        ) : (
          <button onClick={() => onChange(levels)}
            className="rounded border border-indigo-200 px-2 py-0.5 text-[10px] text-indigo-700 hover:bg-indigo-50">
            Use this order
          </button>
        )}
      </div>
      <p className="mb-2 text-[10px] leading-snug text-gray-500">
        {isSet
          ? "Used by ordinal logistic regression, Jonckheere-Terpstra and Cochran-Armitage. Save Metadata to apply."
          : "Not set. Numeric codes and known scales (mild < moderate < severe) are ordered automatically; other text categories need an order here before an ordinal model will run."}
      </p>
      {unplaced.length > 0 && (
        <p className="mb-2 text-[10px] text-amber-700">
          New in the data and placed last: {unplaced.join(", ")}. Check their position, then save.
        </p>
      )}
      <ol className="max-w-md space-y-1">
        {levels.map((v, i) => (
          <li key={v} className="flex items-center gap-2 rounded border border-gray-200 bg-white px-2 py-1 text-xs">
            <span className="w-5 tabular-nums text-gray-400">{i + 1}.</span>
            <span className="font-mono text-gray-700">{v}</span>
            {labels[v] && <span className="text-gray-500">{labels[v]}</span>}
            <span className="ml-auto flex gap-1">
              <button aria-label={`Move ${v} up`} disabled={i === 0} onClick={() => moveTo(i, i - 1)}
                className="rounded border border-gray-200 px-1.5 text-gray-500 hover:bg-gray-50 disabled:opacity-30">↑</button>
              <button aria-label={`Move ${v} down`} disabled={i === levels.length - 1} onClick={() => moveTo(i, i + 1)}
                className="rounded border border-gray-200 px-1.5 text-gray-500 hover:bg-gray-50 disabled:opacity-30">↓</button>
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}
