import { useState } from "react";
import { MAX_GOF_CATEGORIES, parseGofRatio, type GofCategory } from "../lib/gofProportions";

interface GofExpectedInputsProps {
  mode: "equal" | "custom";
  onModeChange: (mode: "equal" | "custom") => void;
  /** Observed categories of the chosen column; null while loading. */
  cats: GofCategory[] | null;
  loadError: string | null;
  tooMany: boolean;
  weights: Record<string, string>;
  parsed: { value: string; weight: number | null }[];
  onWeight: (category: string, text: string) => void;
  /** Weights in the order the categories are listed. */
  onApplyRatio: (weights: number[]) => void;
}

/** Expected-proportion editor of the goodness-of-fit test: equal shares or one weight per category. */
export default function GofExpectedInputs({ mode, onModeChange, cats, loadError, tooMany, weights, parsed, onWeight, onApplyRatio }: GofExpectedInputsProps) {
  const [ratioText, setRatioText] = useState("");
  const [ratioError, setRatioError] = useState<string | null>(null);
  const custom = mode === "custom";
  const total = parsed.reduce((acc, c) => acc + (c.weight ?? 0), 0);

  const applyRatio = () => {
    const parts = parseGofRatio(ratioText);
    if (!parts) { setRatioError("Enter positive numbers separated by colons, e.g. 9:3:3:1."); return; }
    if (parts.length !== (cats?.length ?? 0)) {
      setRatioError(`The ratio has ${parts.length} part(s) but the column has ${cats?.length ?? 0} categories.`);
      return;
    }
    setRatioError(null);
    onApplyRatio(parts);
  };

  return (
    <div className="space-y-2">
      <label className="text-xs text-gray-400 block">Expected proportions</label>
      <div className="flex gap-0 rounded overflow-hidden border border-gray-200" role="group" aria-label="Expected proportions mode">
        {([["equal", "Equal proportions"], ["custom", "Custom weights"]] as const).map(([id, label]) => (
          <button key={id} type="button" aria-pressed={mode === id} onClick={() => onModeChange(id)}
            className={`flex-1 text-[10px] py-1 transition-colors ${mode === id ? "bg-indigo-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"}`}>
            {label}
          </button>
        ))}
      </div>
      {loadError && <p className="text-red-400 text-[11px]">{loadError}</p>}
      {!loadError && cats == null && <p className="text-[11px] text-gray-400">Loading categories{"\u2026"}</p>}
      {tooMany && (
        <p className="text-[11px] text-amber-700">
          This column has more than {MAX_GOF_CATEGORIES} categories; custom weights are not offered. Pick a different column or use equal proportions.
        </p>
      )}
      {cats != null && !tooMany && (
        <div className="space-y-1">
          <div className="max-h-48 overflow-auto rounded border border-gray-200">
            <table className="w-full text-xs">
              <thead><tr className="bg-gray-50">
                <th className="px-2 py-1 text-left">Category</th>
                <th className="px-2 py-1 text-right">n</th>
                <th className="px-2 py-1 text-right">{custom ? "Weight" : "Expected"}</th>
              </tr></thead>
              <tbody>
                {cats.map((c, i) => {
                  const invalid = custom && parsed[i]?.weight === null;
                  return (
                    <tr key={c.value} className="border-t border-gray-100">
                      <td className="px-2 py-1 font-medium break-all">{c.value}</td>
                      <td className="px-2 py-1 text-right font-mono">{c.count}</td>
                      <td className="px-2 py-1 text-right">
                        {custom ? (
                          <input
                            className={`select w-20 text-xs text-right ${invalid ? "border-red-400" : ""}`}
                            aria-label={`Expected weight for ${c.value}`}
                            aria-invalid={invalid}
                            value={weights[c.value] ?? "1"}
                            onChange={(e) => onWeight(c.value, e.target.value)}
                          />
                        ) : (
                          <span className="font-mono text-gray-500">{(100 / cats.length).toFixed(1)}%</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {custom && (
            <>
              <div className="flex gap-1">
                <input className="input flex-1 text-xs" aria-label="Ratio in the order listed" placeholder="e.g. 9:3:3:1"
                  value={ratioText} onChange={(e) => setRatioText(e.target.value)} />
                <button type="button" className="text-[10px] px-2 rounded border border-gray-300 text-gray-600 hover:bg-gray-50" onClick={applyRatio}>
                  Apply ratio
                </button>
              </div>
              {ratioError && <p className="text-red-400 text-[11px]">{ratioError}</p>}
              <p className="text-[10px] text-gray-400">
                Weights may be proportions (0.25), percentages (25%), fractions (1/4) or ratio parts (9, 3, 3, 1).
                They are rescaled to sum to 1{total > 0 ? ` (current total ${Number(total.toPrecision(6))})` : ""}.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
