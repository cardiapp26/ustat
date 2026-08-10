/**
 * PlotThemeBar – compact header widget for choosing the global chart palette + font.
 * Rendered inside the App header next to the grid toggle.
 */
import { useState } from "react";
import { useStore, PALETTES, DEFAULT_THEME, paletteOf, type PaletteName } from "../store";
import { Palette } from "lucide-react";

const PALETTE_LABELS: Record<PaletteName, string> = {
  indigo:    "Indigo",
  clinical:  "Clinical Blue",
  nature:    "Nature",
  grayscale: "Grayscale",
  warm:      "Warm",
  jama:      "JAMA",
  custom:    "Custom",
};

const FONTS = [
  { label: "System (default)", value: "system-ui, sans-serif" },
  { label: "Inter",            value: "Inter, sans-serif" },
  { label: "Georgia (serif)",  value: "Georgia, serif" },
  { label: "Courier (mono)",   value: "Courier New, monospace" },
];

export default function PlotThemeBar() {
  const { plotTheme, setPlotTheme } = useStore();
  const [open, setOpen] = useState(false);
  const [pinName, setPinName] = useState("");
  const [pinColour, setPinColour] = useState("#c0392b");

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className={`p-1.5 rounded-lg transition-colors ${open ? "bg-indigo-50 text-indigo-600" : "text-gray-400 hover:text-gray-700 hover:bg-gray-100"}`}
        title="Chart theme"
      >
        <Palette size={16} />
      </button>

      {open && (
        <>
          {/* Backdrop */}
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          {/* Panel */}
          <div className="absolute right-0 top-9 z-20 bg-white border border-gray-200 rounded-xl shadow-2xl p-4 w-72 space-y-4">
            <p className="text-sm font-semibold text-gray-800">Chart Theme</p>

            {/* Palette swatches */}
            <div>
              <p className="text-xs text-gray-500 mb-2">Color Palette</p>
              <div className="space-y-1.5">
                {(Object.keys(PALETTES) as PaletteName[]).map((p) => (
                  <button
                    key={p}
                    onClick={() => setPlotTheme({ palette: p })}
                    className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-lg transition-colors
                      ${plotTheme.palette === p ? "bg-indigo-50 border border-indigo-200" : "hover:bg-gray-50 border border-transparent"}`}
                  >
                    {/* Color swatches — the custom row shows the live list,
                        not the module constant, so it reflects edits. */}
                    <div className="flex gap-0.5 flex-shrink-0">
                      {(p === "custom" ? paletteOf({ ...plotTheme, palette: "custom" }) : PALETTES[p])
                        .slice(0, 5).map((c, i) => (
                          <div key={i} className="w-4 h-4 rounded-sm" style={{ backgroundColor: c }} />
                        ))}
                    </div>
                    <span className={`text-xs ${plotTheme.palette === p ? "text-indigo-700 font-semibold" : "text-gray-600"}`}>
                      {PALETTE_LABELS[p]}
                    </span>
                    {plotTheme.palette === p && <span className="ml-auto text-indigo-500 text-xs">✓</span>}
                  </button>
                ))}
              </div>
            </div>

            {/* Custom colours — only where they apply */}
            {plotTheme.palette === "custom" && (
              <div>
                <p className="text-xs text-gray-500 mb-1.5">
                  Custom colours <span className="text-gray-400">(trace order)</span>
                </p>
                <div className="grid grid-cols-4 gap-1.5">
                  {plotTheme.customPalette.map((c, i) => (
                    <input
                      key={i}
                      type="color"
                      aria-label={`Custom colour ${i + 1}`}
                      value={c}
                      onChange={(e) => {
                        const next = [...plotTheme.customPalette];
                        next[i] = e.target.value;
                        setPlotTheme({ customPalette: next });
                      }}
                      className="h-7 w-full cursor-pointer rounded border border-gray-200 bg-white p-0.5"
                    />
                  ))}
                </div>
                <div className="mt-1.5 flex gap-2">
                  <button
                    onClick={() => setPlotTheme({ customPalette: [...plotTheme.customPalette, "#111827"] })}
                    className="flex-1 rounded border border-gray-200 py-1 text-[10px] text-gray-500 hover:bg-gray-50">
                    Add a colour
                  </button>
                  <button
                    disabled={plotTheme.customPalette.length <= 1}
                    onClick={() => setPlotTheme({ customPalette: plotTheme.customPalette.slice(0, -1) })}
                    className="flex-1 rounded border border-gray-200 py-1 text-[10px] text-gray-500 hover:bg-gray-50 disabled:opacity-40">
                    Remove the last
                  </button>
                </div>
              </div>
            )}

            {/* Colours pinned to a named group. Palette order follows how the
                groups happen to sort, so "treatment must be red" cannot be
                expressed by a palette — adding a third arm would recolour the
                first two. */}
            <div>
              <p className="text-xs text-gray-500 mb-1.5">Fixed colour for a group</p>
              {Object.entries(plotTheme.seriesColors).length > 0 && (
                <div className="mb-1.5 space-y-1">
                  {Object.entries(plotTheme.seriesColors).map(([name, colour]) => (
                    <div key={name} className="flex items-center gap-1.5">
                      <input
                        type="color"
                        aria-label={`Colour for ${name}`}
                        value={colour}
                        onChange={(e) => setPlotTheme({
                          seriesColors: { ...plotTheme.seriesColors, [name]: e.target.value },
                        })}
                        className="h-6 w-8 shrink-0 cursor-pointer rounded border border-gray-200 bg-white p-0.5"
                      />
                      <span className="truncate text-[11px] text-gray-600" title={name}>{name}</span>
                      <button
                        aria-label={`Unpin ${name}`}
                        onClick={() => {
                          const next = { ...plotTheme.seriesColors };
                          delete next[name];
                          setPlotTheme({ seriesColors: next });
                        }}
                        className="ml-auto px-1 text-[11px] text-gray-400 hover:text-red-500">
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              )}
              <form
                className="flex gap-1.5"
                onSubmit={(e) => {
                  e.preventDefault();
                  const name = pinName.trim();
                  if (!name) return;
                  setPlotTheme({ seriesColors: { ...plotTheme.seriesColors, [name]: pinColour } });
                  setPinName("");
                }}>
                <input
                  value={pinName}
                  onChange={(e) => setPinName(e.target.value)}
                  placeholder="Group label, e.g. Male"
                  aria-label="Group label to pin"
                  className="min-w-0 flex-1 rounded border border-gray-300 px-1.5 py-1 text-[11px] outline-none focus:border-indigo-400"
                />
                <input
                  type="color"
                  aria-label="Colour to pin"
                  value={pinColour}
                  onChange={(e) => setPinColour(e.target.value)}
                  className="h-7 w-8 shrink-0 cursor-pointer rounded border border-gray-200 bg-white p-0.5"
                />
                <button type="submit"
                  className="rounded bg-indigo-600 px-2 text-[11px] font-medium text-white hover:bg-indigo-700">
                  Pin
                </button>
              </form>
              <p className="mt-1 text-[10px] text-gray-400">
                Matched against the label printed in the legend, exactly as shown.
              </p>
            </div>

            {/* Font */}
            <div>
              <p className="text-xs text-gray-500 mb-1.5">Chart Font</p>
              <select
                value={plotTheme.fontFamily}
                onChange={(e) => setPlotTheme({ fontFamily: e.target.value })}
                className="select w-full text-xs"
              >
                {FONTS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
              </select>
            </div>

            {/* Font size */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <p className="text-xs text-gray-500">Font Size</p>
                <span className="text-xs font-mono text-gray-600">{plotTheme.fontSize}px</span>
              </div>
              <input type="range" min={8} max={16} step={1} value={plotTheme.fontSize}
                onChange={(e) => setPlotTheme({ fontSize: +e.target.value })}
                className="w-full accent-indigo-500" />
            </div>

            {/* Line width */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <p className="text-xs text-gray-500">Line Width</p>
                <span className="text-xs font-mono text-gray-600">{plotTheme.lineWidth}px</span>
              </div>
              <input type="range" min={1} max={5} step={0.5} value={plotTheme.lineWidth}
                onChange={(e) => setPlotTheme({ lineWidth: +e.target.value })}
                className="w-full accent-indigo-500" />
            </div>

            {/* Marker size */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <p className="text-xs text-gray-500">Marker Size</p>
                <span className="text-xs font-mono text-gray-600">{plotTheme.markerSize}px</span>
              </div>
              <input type="range" min={2} max={14} step={1} value={plotTheme.markerSize}
                onChange={(e) => setPlotTheme({ markerSize: +e.target.value })}
                className="w-full accent-indigo-500" />
            </div>

            {/* Plot background */}
            <div>
              <p className="text-xs text-gray-500 mb-1.5">Plot Background</p>
              <div className="flex gap-2">
                {[["#ffffff","White"],["#f9fafb","Off-white"],["#1f2937","Dark"]].map(([bg, lbl]) => (
                  <button key={bg} onClick={() => setPlotTheme({ plotBg: bg })}
                    className={`flex-1 text-[10px] py-1 rounded border transition-colors
                      ${plotTheme.plotBg === bg ? "border-indigo-400 text-indigo-700 bg-indigo-50" : "border-gray-200 text-gray-500 hover:bg-gray-50"}`}
                    style={{ backgroundColor: bg, color: bg === "#1f2937" ? "#e5e7eb" : undefined }}>
                    {lbl}
                  </button>
                ))}
              </div>
            </div>

            {/* Reset */}
            <button onClick={() => setPlotTheme(DEFAULT_THEME)}
              className="w-full text-xs py-1 text-gray-400 hover:text-gray-700 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
              Reset to defaults
            </button>
          </div>
        </>
      )}
    </div>
  );
}
