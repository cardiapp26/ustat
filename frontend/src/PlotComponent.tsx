// Vite requires explicit default export handling for react-plotly.js (CJS module)
import { forwardRef } from "react";
import _Plot from "react-plotly.js";
import type { PlotParams } from "react-plotly.js";
import { useStaleGuard } from "./lib/staleGuard";

type PlotModule = typeof _Plot & { default?: typeof _Plot };
const PlotBase = (_Plot as PlotModule).default ?? _Plot;

const Plot = forwardRef<InstanceType<typeof _Plot>, PlotParams>((props, ref) => {
  // The modebar camera is an export path like any other: inside an
  // out-of-date result (a <StaleGuard>) it is removed.
  const guard = useStaleGuard();
  const removed = (props.config?.modeBarButtonsToRemove ?? []) as string[];
  const modeBarButtonsToRemove = guard.stale && !removed.includes("toImage")
    ? [...removed, "toImage"]
    : props.config?.modeBarButtonsToRemove;
  return (
    <PlotBase
      {...props}
      ref={ref}
      // Plotly.toImage clones the layout, not the surrounding DOM background.
      // A transparent paper layer therefore stays transparent even when
      // config.setBackground is "opaque". Force the actual figure paper white
      // so every export path (modebar, PNG, SVG, JPEG, TIFF and clipboard) has
      // a real white background.
      layout={{ ...props.layout, paper_bgcolor: "#ffffff" }}
      config={{ ...props.config, setBackground: "opaque", modeBarButtonsToRemove } as PlotParams["config"]}
    />
  );
});

Plot.displayName = "Plot";

export default Plot;
