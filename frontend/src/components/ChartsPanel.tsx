import { useState, useRef } from "react";
import { useStore, isNumericKind, isCategoricalKind, type Session } from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import {
  usePlotLayout, usePalette, useTraceDefaults, applySeriesPins, applyHighlight, ordinalLadder,
} from "../plotStyle";
import { getHistogram, getScatter, getBoxplot, getBar, getPairedBox, getDumbbell, getCompareMeans, getErrorPlot, getEcdf, getPie, getBalloon, getSummaryStats, getFacet, getLinePlot, getSlopePlot, getSankey, getStackPlot, getRidgePlot, getSets } from "../api";
import type { PlotData, PlotLayout, PlotCaptureHandle } from "../lib/plotTypes";
import TitledPlot from "./TitledPlot";
import ChartTypeIcon from "./charts/ChartTypeIcon";
import { fmtP } from "../lib/format";
import { labelFor } from "../lib/valueLabels";
import { categoryColors } from "../lib/categoryColors";
import { CHART_TYPES } from "../lib/chartTypes";
import { parseRefValue, referenceLineOverlay, type RefLine } from "../lib/referenceLines";

/** Charts drawn on one x/y pair, where a reference line has a place to go.
 *  A pie has no axes; a facet has several; a Sankey's axes are not values. */
const REF_LINE_CHARTS = new Set([
  "histogram", "scatter", "boxplot", "violin", "raincloud", "strip", "bar", "paired",
  "dumbbell", "errorplot", "ecdf", "lineplot", "slopeplot", "stackplot", "ridgeplot",
]);

/** A reference line as typed: the value stays text until it parses. */
interface RefLineDraft { axis: "x" | "y"; value: string; label: string }

/** Charts whose request actually carries the Color / Group column. */
const COLOUR_AWARE_CHARTS = new Set([
  "histogram", "scatter", "boxplot", "violin", "raincloud", "strip", "bar", "paired",
  "dumbbell", "errorplot", "ecdf", "facet", "lineplot", "slopeplot", "ridgeplot",
]);

export default function ChartsPanel() {
  const session  = useStore((s) => s.session);
  if (!session) return null;
  return <ChartsPanelBody session={session} />;
}

function ChartsPanelBody({ session }: { session: Session }) {
  const layout   = usePlotLayout();
  const pal      = usePalette();
  const td       = useTraceDefaults();

  const numCols = session.columns.filter((c) => isNumericKind(c.kind) && !c.analysis_excluded).map((c) => c.name);
  const catCols = session.columns.filter((c) => isCategoricalKind(c.kind) && !c.analysis_excluded).map((c) => c.name);

  const [chartType, setChartType] = usePersistedPanelState<string>("charts", "chartType", "histogram");
  const [x, setX] = usePersistedPanelState<string>("charts", "x", numCols[0] ?? "");
  const [y, setY] = usePersistedPanelState<string>("charts", "y", numCols[1] ?? "");
  const [color, setColor] = usePersistedPanelState<string>("charts", "color", "");
  const [pairId, setPairId] = usePersistedPanelState<string>("charts", "pairId", session.columns[0]?.name ?? "");
  const [bins, setBins] = usePersistedPanelState<number>("charts", "bins", 20);
  // Agreement-plot options (scatter): log axes for values spanning orders of
  // magnitude, y = x when both axes carry the same quantity, and a per-point
  // label so the outliers can be named on the figure.
  const [logX, setLogX] = usePersistedPanelState<boolean>("charts", "logX", false);
  const [logY, setLogY] = usePersistedPanelState<boolean>("charts", "logY", false);
  const [identityLine, setIdentityLine] = usePersistedPanelState<boolean>("charts", "identityLine", false);
  const [labelCol, setLabelCol] = usePersistedPanelState<string>("charts", "labelCol", "");
  // Dumbbell: two numeric columns per category, drawn as a gap.
  const [dbStart, setDbStart] = usePersistedPanelState<string>("charts", "dbStart", numCols[0] ?? "");
  const [dbEnd, setDbEnd] = usePersistedPanelState<string>("charts", "dbEnd", numCols[1] ?? "");
  const [dbSort, setDbSort] = usePersistedPanelState<string>("charts", "dbSort", "gap");
  // Significance brackets over box / violin / bar — ggpubr's stat_compare_means.
  const [showBrackets, setShowBrackets] = usePersistedPanelState<boolean>("charts", "showBrackets", false);
  const [cmpMethod, setCmpMethod] = usePersistedPanelState<string>("charts", "cmpMethod", "auto");
  const [cmpAdjust, setCmpAdjust] = usePersistedPanelState<string>("charts", "cmpAdjust", "holm");
  const [cmpLabel, setCmpLabel] = usePersistedPanelState<string>("charts", "cmpLabel", "stars");
  const [showPoints, setShowPoints] = usePersistedPanelState<boolean>("charts", "showPoints", false);
  // The box draws the median. A published box plot usually marks the mean too,
  // and the gap between the two is what tells the reader the distribution is
  // skewed — which is the same thing the test choice rests on.
  const [showMean, setShowMean] = usePersistedPanelState<boolean>("charts", "showMean", false);
  const [horizontal, setHorizontal] = usePersistedPanelState<boolean>("charts", "horizontal", false);
  // Log scale on the VALUE axis of a grouped chart. Distinct from the
  // scatter's log_x/log_y, which are axis-per-variable; here there is one
  // value axis and its orientation moves with `horizontal`.
  const [logValue, setLogValue] = usePersistedPanelState<boolean>("charts", "logValue", false);
  const [barMode, setBarMode] = usePersistedPanelState<string>("charts", "barMode", "mean");
  const [barTarget, setBarTarget] = usePersistedPanelState<string>("charts", "barTarget", "");
  // Whisker on a bar of means (geom_col + errorbar) and coord_flip for bars.
  const [barError, setBarError] = usePersistedPanelState<string>("charts", "barError", "none");
  const [barHorizontal, setBarHorizontal] = usePersistedPanelState<boolean>("charts", "barHorizontal", false);
  // geom_hline / geom_vline: thresholds drawn in data coordinates.
  const [refLines, setRefLines] = usePersistedPanelState<RefLineDraft[]>("charts", "refLines", []);
  // Error plot / ECDF
  const [errCentre, setErrCentre] = usePersistedPanelState<string>("charts", "errCentre", "mean");
  const [errSpread, setErrSpread] = usePersistedPanelState<string>("charts", "errSpread", "ci");
  // Scatter cloud description (ellipse / marginal) and marker shape.
  const [ellipse, setEllipse] = usePersistedPanelState<boolean>("charts", "ellipse", false);
  const [marginal, setMarginal] = usePersistedPanelState<boolean>("charts", "marginal", false);
  // geom_smooth: the straight line (lm), the local curve (loess), or none;
  // and whether each colour group gets its own.
  const [fitMethod, setFitMethod] = usePersistedPanelState<string>("charts", "fitMethod", "lm");
  const [fitPerGroup, setFitPerGroup] = usePersistedPanelState<boolean>("charts", "fitPerGroup", false);
  const [loessSpan, setLoessSpan] = usePersistedPanelState<number>("charts", "loessSpan", 0.75);
  // Histogram: what the y axis counts, how groups are laid over each other,
  // whether bars / density / both are drawn, a rug, and a bin width in data
  // units that overrides the bin count when set.
  const [histStat, setHistStat] = usePersistedPanelState<string>("charts", "histStat", "count");
  const [histPosition, setHistPosition] = usePersistedPanelState<string>("charts", "histPosition", "overlay");
  const [histDisplay, setHistDisplay] = usePersistedPanelState<string>("charts", "histDisplay", "both");
  const [histRug, setHistRug] = usePersistedPanelState<boolean>("charts", "histRug", false);
  const [binwidth, setBinwidth] = usePersistedPanelState<string>("charts", "binwidth", "");
  // facet_wrap(scales =, ncol =).
  const [facetScales, setFacetScales] = usePersistedPanelState<string>("charts", "facetScales", "fixed");
  const [facetNcol, setFacetNcol] = usePersistedPanelState<string>("charts", "facetNcol", "auto");
  // A numeric column mapped to a colour ramp on the scatter — a continuous
  // scale where Color / Group gives a discrete one.
  const [gradientCol, setGradientCol] = usePersistedPanelState<string>("charts", "gradientCol", "");
  const [gradientScale, setGradientScale] = usePersistedPanelState<string>("charts", "gradientScale", "Viridis");
  // geom_bin2d: the cloud as a grid of counts once the points are a blob.
  const [bin2d, setBin2d] = usePersistedPanelState<boolean>("charts", "bin2d", false);
  const [bin2dBins, setBin2dBins] = usePersistedPanelState<number>("charts", "bin2dBins", 30);
  // A dash pattern per group, so a figure printed in greyscale still
  // separates its lines — ggplot2's aes(linetype = group).
  const [varyLineStyle, setVaryLineStyle] = usePersistedPanelState<boolean>("charts", "varyLineStyle", false);
  // coord_cartesian(xlim =, ylim =): a manual window on the axes. Kept as
  // typed text so a half-entered number is not read as a limit.
  const [xMin, setXMin] = usePersistedPanelState<string>("charts", "xMin", "");
  const [xMax, setXMax] = usePersistedPanelState<string>("charts", "xMax", "");
  const [yMin, setYMin] = usePersistedPanelState<string>("charts", "yMin", "");
  const [yMax, setYMax] = usePersistedPanelState<string>("charts", "yMax", "");
  const [shapeCol, setShapeCol] = usePersistedPanelState<string>("charts", "shapeCol", "");
  // Pie / donut, balloon, facet
  const [pieValue, setPieValue] = usePersistedPanelState<string>("charts", "pieValue", "");
  const [donut, setDonut] = usePersistedPanelState<boolean>("charts", "donut", false);
  const [balloonCol, setBalloonCol] = usePersistedPanelState<string>("charts", "balloonCol", "");
  const [facetCol, setFacetCol] = usePersistedPanelState<string>("charts", "facetCol", "");
  const [facetKind, setFacetKind] = usePersistedPanelState<string>("charts", "facetKind", "boxplot");
  // "level" splits one measurement across the levels of a column; "variable"
  // puts a different measurement in each panel — the layout of a published
  // multi-panel figure (QT | QRS | index, each over the same two groups).
  const [facetMode, setFacetMode] = usePersistedPanelState<string>("charts", "facetMode", "level");
  const [facetVars, setFacetVars] = usePersistedPanelState<string[]>("charts", "facetVars", []);
  const [showSummary, setShowSummary] = usePersistedPanelState<boolean>("charts", "showSummary", false);
  // cnsplots shapes: line over an ordered axis, before/after slopes, flow,
  // stacked composition, stacked densities, set overlap.
  const [lineY, setLineY] = usePersistedPanelState<string>("charts", "lineY", numCols[0] ?? "");
  const [slopeBefore, setSlopeBefore] = usePersistedPanelState<string>("charts", "slopeBefore", numCols[0] ?? "");
  const [slopeAfter, setSlopeAfter] = usePersistedPanelState<string>("charts", "slopeAfter", numCols[1] ?? "");
  const [stage2, setStage2] = usePersistedPanelState<string>("charts", "stage2", "");
  const [stage3, setStage3] = usePersistedPanelState<string>("charts", "stage3", "");
  const [fillCol, setFillCol] = usePersistedPanelState<string>("charts", "fillCol", "");
  const [stackNormalize, setStackNormalize] = usePersistedPanelState<boolean>("charts", "stackNormalize", false);
  const [setCols, setSetCols] = usePersistedPanelState<string[]>("charts", "setCols", []);
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null);
  const [comparisons, setComparisons] = useState<Record<string, unknown> | null>(null);
  const [plotData, setPlotData] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // States for custom labels
  const [customTitle, setCustomTitle] = useState("");
  const [customXLabel, setCustomXLabel] = useState("");
  const [customYLabel, setCustomYLabel] = useState("");

  const run = async () => {
    if (chartType === "paired") {
      if (!color) { setError("Select a Group/Color column with exactly two levels (e.g. treatment status)."); return; }
      if (!pairId) { setError("Select a Pair ID column linking each matched pair (e.g. match_set_id)."); return; }
    }
    if (chartType === "dumbbell") {
      if (!x) { setError("Select the Category column — one row per level (e.g. the variable name)."); return; }
      if (!dbStart || !dbEnd) { setError("Select both value columns: the open marker and the filled one."); return; }
      if (dbStart === dbEnd) { setError("The two value columns must differ — a dumbbell shows the gap between them."); return; }
    }
    if (chartType === "ecdf" || chartType === "errorplot") {
      if (!x) { setError("Select the numeric variable to summarise."); return; }
    }
    if (chartType === "balloon" && (!x || !balloonCol)) {
      setError("Select both a row and a column variable — a balloon plot is a cross-tabulation."); return;
    }
    if (chartType === "balloon" && x === balloonCol) {
      setError("Row and column variables must differ."); return;
    }
    if (chartType === "facet" && facetMode === "variable" && facetKind === "boxplot") {
      if (facetVars.length < 1) {
        setError("Tick at least one variable to give a panel to."); return;
      }
    } else if (chartType === "facet" && !facetCol) {
      setError("Select the column to split into panels."); return;
    }
    if (chartType === "slopeplot") {
      if (!slopeBefore || !slopeAfter) { setError("Select both the before and after measurements."); return; }
      if (slopeBefore === slopeAfter) { setError("The two measurements must be different columns."); return; }
    }
    if (chartType === "sankey" && !stage2) {
      setError("A Sankey needs at least two stages — select the second one."); return;
    }
    if (chartType === "stackplot") {
      if (!fillCol) { setError("Select what to stack inside each bar."); return; }
      if (fillCol === x) { setError("The axis and the stacked variable must differ."); return; }
    }
    if (chartType === "ridgeplot" && !color) {
      setError("Select a Color / Group column — a ridge plot draws one density per group."); return;
    }
    if (chartType === "sets" && setCols.length < 2) {
      setError("Tick at least two membership columns."); return;
    }
    setLoading(true);
    setError(null);
    setComparisons(null);
    setSummary(null);
    try {
      const base = { session_id: session.session_id, x, bins };
      let res;
      if (chartType === "histogram") {
        const bw = parseRefValue(binwidth);
        res = await getHistogram({
          ...base, color: color || undefined, rug: histRug,
          ...(bw !== null && bw > 0 ? { binwidth: bw } : {}),
        });
      }
      else if (chartType === "scatter") res = await getScatter({
        ...base, y, color: color || undefined,
        log_x: logX, log_y: logY, identity_line: identityLine,
        label: labelCol || undefined, shape: shapeCol || undefined,
        ellipse, marginal,
        fit: fitMethod, fit_per_group: fitPerGroup && Boolean(color), loess_span: loessSpan,
        ...(gradientCol && !color && !bin2d ? { gradient: gradientCol } : {}),
        ...(bin2d ? { bin2d: true, bin2d_bins: bin2dBins } : {}),
      });
      else if (chartType === "boxplot" || chartType === "violin" || chartType === "raincloud" || chartType === "strip") res = await getBoxplot({ ...base, color: color || undefined });
      else if (chartType === "paired") res = await getPairedBox({ session_id: session.session_id, y: x, group: color, pair_id: pairId });
      else if (chartType === "dumbbell") res = await getDumbbell({
        session_id: session.session_id, category: x,
        start: dbStart, end: dbEnd, group: color || undefined, sort: dbSort,
      });
      else if (chartType === "errorplot") res = await getErrorPlot({
        session_id: session.session_id, y: x, group: color || undefined,
        centre: errCentre, spread: errSpread,
      });
      else if (chartType === "ecdf") res = await getEcdf({
        session_id: session.session_id, x, group: color || undefined,
      });
      else if (chartType === "pie") res = await getPie({
        session_id: session.session_id, category: x, value: pieValue || undefined,
      });
      else if (chartType === "balloon") res = await getBalloon({
        session_id: session.session_id, row: x, col: balloonCol,
      });
      else if (chartType === "lineplot") res = await getLinePlot({
        session_id: session.session_id, x, y: lineY, group: color || undefined,
        centre: errCentre, spread: errSpread === "iqr" && errCentre === "mean" ? "ci" : errSpread,
      });
      else if (chartType === "slopeplot") res = await getSlopePlot({
        session_id: session.session_id, before: slopeBefore, after: slopeAfter,
        group: color || undefined,
      });
      else if (chartType === "sankey") res = await getSankey({
        session_id: session.session_id,
        stages: [x, stage2, ...(stage3 ? [stage3] : [])],
      });
      else if (chartType === "stackplot") res = await getStackPlot({
        session_id: session.session_id, x, fill: fillCol,
        value: pieValue || undefined, normalize: stackNormalize,
      });
      else if (chartType === "ridgeplot") res = await getRidgePlot({
        session_id: session.session_id, x, group: color,
      });
      else if (chartType === "sets") res = await getSets({
        session_id: session.session_id, columns: setCols,
      });
      else if (chartType === "facet") {
        const grid = {
          scales: facetScales,
          ...(facetNcol !== "auto" ? { ncol: Number(facetNcol) } : {}),
        };
        res = await getFacet(
          facetMode === "variable" && facetKind === "boxplot"
            ? {
              session_id: session.session_id, kind: "boxplot",
              variables: facetVars, color: color || undefined, ...grid,
            }
            : {
              session_id: session.session_id, kind: facetKind, x,
              y: facetKind === "scatter" ? y : undefined,
              facet: facetCol, color: color || undefined, ...grid,
            });
      }
      else res = await getBar({
        ...base, y: y || undefined, color: color || undefined,
        y_mode: barMode,
        ...(barMode === "percentage" && barTarget.trim() ? { target_value: barTarget.trim() } : {}),
        ...(barMode === "mean" && y && barError !== "none" ? { error: barError } : {}),
      });
      setPlotData(res.data);

      // The summary table is a separate result printed under the plot, so a
      // failure there must not cost the user the chart.
      if (showSummary && ["boxplot", "violin", "raincloud", "errorplot", "ecdf"].includes(chartType)) {
        try {
          const s = await getSummaryStats({
            session_id: session.session_id, y: x, group: color || undefined,
          });
          setSummary(s.data);
        } catch { /* the chart stands on its own */ }
      }

      // Brackets are a second call: the comparison is a statistical result in
      // its own right, and a failure there must not lose the plot the user
      // already has.
      if (showBrackets && color && (chartType === "boxplot" || chartType === "violin" || chartType === "raincloud" || chartType === "strip")) {
        try {
          const cmp = await getCompareMeans({
            session_id: session.session_id, y: x, group: color,
            method: cmpMethod, p_adjust: cmpAdjust, label: cmpLabel,
          });
          setComparisons(cmp.data);
        } catch (e: unknown) {
          const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
          setError(`Chart drawn, but the comparisons failed: ${detail ?? "unknown error"}`);
        }
      }

      // Auto-generate beautiful defaults
      const xMeta = session.columns.find((c) => c.name === x);
      const xLabelText = xMeta?.label || xMeta?.name || x;
      
      const yMeta = y ? session.columns.find((c) => c.name === y) : null;
      const yLabelText = yMeta ? (yMeta.label || yMeta.name || y) : "Count";
      
      const colorMeta = color ? session.columns.find((c) => c.name === color) : null;
      const colorLabelText = colorMeta ? (colorMeta.label || colorMeta.name || color) : "";

      let autoTitle = "";
      let autoX = "";
      let autoY = "";

      if (chartType === "histogram") {
        autoTitle = colorLabelText
          ? `Distribution of ${xLabelText} by ${colorLabelText}`
          : `Distribution of ${xLabelText}`;
        autoX = xLabelText;
        autoY = histStat === "density" ? "Density" : histStat === "percent" ? "% of observations" : "Count";
      } else if (chartType === "scatter") {
        autoTitle = `${yLabelText} vs ${xLabelText}`;
        autoX = xLabelText;
        autoY = yLabelText;
      } else if (chartType === "boxplot" || chartType === "violin" || chartType === "raincloud" || chartType === "strip") {
        autoTitle = colorLabelText ? `Distribution of ${xLabelText} by ${colorLabelText}` : `Distribution of ${xLabelText}`;
        autoX = colorLabelText || "Overall";
        autoY = xLabelText;
      } else if (chartType === "bar") {
        // A percentage axis captioned with the outcome's name reads as the
        // outcome itself — "Malign" against a 0-100 axis is not what the bar
        // shows. The share, and what of, is the caption.
        const pct = barMode === "percentage" && y;
        const pctText = barTarget.trim()
          ? `% ${yLabelText} = ${barTarget.trim()}`
          : `% ${yLabelText}`;
        autoTitle = pct
          ? `${pctText} by ${xLabelText}`
          : (y ? `${yLabelText} by ${xLabelText}` : `Count by ${xLabelText}`);
        autoX = xLabelText;
        autoY = pct ? pctText : yLabelText;
      } else if (chartType === "paired") {
        autoTitle = `Matched-pair ${xLabelText} by ${colorLabelText}`;
        autoX = colorLabelText;
        autoY = xLabelText;
      } else if (chartType === "errorplot") {
        autoTitle = colorLabelText ? `${xLabelText} by ${colorLabelText}` : xLabelText;
        autoX = colorLabelText || "Overall";
        autoY = xLabelText;
      } else if (chartType === "ecdf") {
        autoTitle = colorLabelText ? `Cumulative distribution of ${xLabelText} by ${colorLabelText}` : `Cumulative distribution of ${xLabelText}`;
        autoX = xLabelText;
        autoY = "Cumulative proportion";
      } else if (chartType === "dumbbell") {
        const startMeta = session.columns.find((c) => c.name === dbStart);
        const endMeta = session.columns.find((c) => c.name === dbEnd);
        const startText = startMeta?.label || dbStart;
        const endText = endMeta?.label || dbEnd;
        autoTitle = `${endText} vs ${startText}, by ${xLabelText}`;
        autoX = `${startText} (open) → ${endText} (filled)`;
        autoY = xLabelText;
      }

      setCustomTitle(autoTitle);
      // A transposed chart swaps which axis carries the values, so the titles
      // have to swap with it — otherwise the value axis is captioned with the
      // grouping variable's name, which is worse than no caption.
      const transposed = (horizontal
        && (chartType === "boxplot" || chartType === "violin"
            || chartType === "raincloud" || chartType === "strip"))
        || (chartType === "bar" && barHorizontal);
      setCustomXLabel(transposed ? autoY : autoX);
      setCustomYLabel(transposed ? autoX : autoY);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail ?? "Error generating chart");
    } finally {
      setLoading(false);
    }
  };

  const chartRef = useRef<PlotCaptureHandle | null>(null);
  const seriesColors = useStore((s) => s.plotTheme.seriesColors);
  const highlightGroup = useStore((s) => s.plotTheme.highlightGroup);
  const highlightColor = useStore((s) => s.plotTheme.highlightColor);
  // An ORDERED grouping column takes a lightness ladder of the palette's
  // first hue instead of the palette's unrelated hues, so tertile 3 is
  // visibly "more" than tertile 1 without a trip to the legend.
  const colourMeta = color ? session.columns.find((c) => c.name === color) : undefined;
  const groupLevels = colourMeta?.kind === "ordinal" ? countGroupLevels(plotData) : 0;
  const chartPalette = groupLevels >= 2 ? ordinalLadder(pal[0], groupLevels) : pal;
  const traces = applyHighlight(
    applySeriesPins(
      plotData ? buildTraces(
        plotData, chartType, chartPalette, td, session, showPoints, donut ? 0.45 : 0, showMean,
        chartType === "bar" ? barHorizontal : horizontal,
        { stat: histStat, display: histDisplay, rug: histRug, gradientScale, varyLineStyle },
      ) : null,
      seriesColors,
    ),
    highlightGroup,
    highlightColor,
  );
  const brackets = buildBrackets(plotData, comparisons);
  const groupedChart = chartType === "boxplot" || chartType === "violin"
    || chartType === "raincloud" || chartType === "strip";
  const valueAxisIsLog = groupedChart && logValue;
  const barFlipped = chartType === "bar" && barHorizontal && plotData?.type === "bar";
  const numericAxes = numericAxesFor(chartType, horizontal, barHorizontal);
  const xIsLog = Boolean(plotData?.log_x) || (valueAxisIsLog && horizontal);
  const yIsLog = Boolean(plotData?.log_y) || (valueAxisIsLog && !horizontal);
  const xWindow = numericAxes.x ? axisWindow(xMin, xMax, xIsLog) : {};
  const yWindow = numericAxes.y ? axisWindow(yMin, yMax, yIsLog) : {};
  const marginalAxes = marginalLayout(plotData);
  const facetOverlay = facetLayout(plotData);
  // Reference lines and significance brackets are both shapes + annotations;
  // the facet grid adds panel titles as annotations too. Merged once here so
  // none of the three overwrites another.
  const refOverlay = referenceLineOverlay(
    REF_LINE_CHARTS.has(chartType)
      ? refLines.flatMap((l): RefLine[] => {
        const v = parseRefValue(l.value);
        return v === null ? [] : [{ axis: l.axis, value: v, label: l.label }];
      })
      : [],
  );
  const overlayShapes = [...brackets.shapes, ...refOverlay.shapes];
  const overlayAnnotations = [
    ...((facetOverlay.annotations as Record<string, unknown>[] | undefined) ?? []),
    ...brackets.annotations,
    ...refOverlay.annotations,
  ];

  return (
    <div className="flex gap-4 h-full">
      {/* Controls */}
      <div className="w-60 flex-shrink-0 space-y-4 overflow-y-auto pr-1" style={{ maxHeight: "calc(100vh - 120px)" }}>
        <div className="panel space-y-3 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
          <h3 className="text-sm font-semibold text-gray-700">Chart Type</h3>
          {CHART_TYPES.map((t) => (
            <label key={t} className="flex items-center gap-2 cursor-pointer">
              <input type="radio" name="chartType" value={t} checked={chartType === t}
                onChange={() => setChartType(t)} className="accent-indigo-500" />
              {/* Thumbnail of the shape this chart makes. Muted until the row
                  is picked, so the list reads as labels first and the choice
                  still stands out. */}
              <ChartTypeIcon type={t} className={`w-6 h-4 flex-shrink-0 transition-colors ${
                chartType === t ? "text-indigo-600" : "text-gray-400"}`} />
              <span className="text-sm text-gray-700 capitalize">
                {t === "paired" ? "Paired box" : t === "errorplot" ? "Error plot"
                  : t === "ecdf" ? "ECDF" : t === "pie" ? (donut ? "Donut" : "Pie")
                  : t === "balloon" ? "Balloon" : t === "facet" ? "Facet grid"
                  : t === "lineplot" ? "Line (over visits)" : t === "slopeplot" ? "Slope (before / after)"
                  : t === "sankey" ? "Sankey (flow)" : t === "stackplot" ? "Stacked bar"
                  : t === "ridgeplot" ? "Ridge" : t === "sets" ? "Set overlap"
                  : t === "raincloud" ? "Raincloud" : t === "strip" ? "Strip (points + median)" : t}
              </span>
            </label>
          ))}
        </div>

        <div className="panel space-y-3 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
          <h3 className="text-sm font-semibold text-gray-700">Variables</h3>
          <div>
            <label className="text-xs text-gray-400 block mb-1">
              {chartType === "paired" ? "Outcome (Y)"
                : chartType === "dumbbell" ? "Category (one row each)"
                : chartType === "pie" ? "Category"
                : chartType === "balloon" ? "Rows"
                : chartType === "lineplot" ? "Ordered axis (visit / time)"
                : chartType === "sankey" ? "Stage 1"
                : chartType === "stackplot" ? "Bars (axis category)"
                : chartType === "ridgeplot" ? "Value"
                : chartType === "sets" ? "— use the tick list below —"
                : "X axis"}
            </label>
            <select className="select w-full" value={x} onChange={(e) => setX(e.target.value)}>
              {(chartType === "boxplot" || chartType === "violin" || chartType === "raincloud" || chartType === "paired"
                || chartType === "errorplot" || chartType === "ecdf" || chartType === "facet"
                || chartType === "ridgeplot" ? numCols
                : chartType === "dumbbell" || chartType === "pie" || chartType === "balloon"
                || chartType === "lineplot" || chartType === "sankey" || chartType === "stackplot"
                  ? [...catCols, ...numCols]
                : [...numCols, ...catCols]).map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            {chartType === "dumbbell" && (
              <p className="text-[10px] text-gray-400 mt-1">
                Must have exactly one row per value — e.g. a variable-name column.
              </p>
            )}
          </div>
          {chartType === "dumbbell" && (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Open marker (reference)</label>
                <select className="select w-full" value={dbStart} onChange={(e) => setDbStart(e.target.value)}>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Filled marker (observed)</label>
                <select className="select w-full" value={dbEnd} onChange={(e) => setDbEnd(e.target.value)}>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Order rows by</label>
                <select className="select w-full" value={dbSort} onChange={(e) => setDbSort(e.target.value)}>
                  <option value="gap">Gap (largest first)</option>
                  <option value="end">Observed value</option>
                  <option value="start">Reference value</option>
                  <option value="category">Category name</option>
                </select>
              </div>
            </>
          )}
          {(chartType === "scatter" || chartType === "bar") && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Y axis</label>
              <select className="select w-full" value={y} onChange={(e) => setY(e.target.value)}>
                <option value="">— count —</option>
                {/* In percentage mode the outcome is a condition, not a
                    quantity, so a coded categorical column belongs here too —
                    "Histology = malignant" is the usual form of the question. */}
                {(chartType === "bar" && barMode === "percentage"
                  ? [...numCols, ...catCols.filter((c) => !numCols.includes(c))]
                  : numCols
                ).map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}
          {/* Not gated on a Y column being chosen: percentage mode is what
              ADDS the categorical columns to the Y list, so gating it on Y
              made the one outcome it exists for unreachable. */}
          {chartType === "bar" && (
            <div className="space-y-2">
              <div>
                <label
                  className="text-xs text-gray-400 block mb-1"
                  title="Mean plots the average of the Y column per group. Percentage plots the share of each group meeting a condition on Y — the form a risk-factor figure needs. A mean of a 0/1 column answers the same question but reports 0.37 where the figure wants 37%."
                >Bar height</label>
                <select className="select w-full" value={barMode} onChange={(e) => setBarMode(e.target.value)}>
                  <option value="mean">Mean of Y</option>
                  <option value="percentage">% of group where Y is…</option>
                </select>
              </div>
              {barMode === "percentage" && y && (
                <div>
                  <label
                    className="text-xs text-gray-400 block mb-1"
                    title="The value of Y that counts towards the percentage, e.g. 1, or the code for malignant. Leave blank to treat Y as a 0/1 flag and count every non-zero."
                  >Counts as a hit</label>
                  <input
                    className="select w-full text-sm"
                    placeholder="blank = any non-zero"
                    value={barTarget}
                    onChange={(e) => setBarTarget(e.target.value)}
                  />
                </div>
              )}
              {barMode === "mean" && y && (
                <div>
                  <label
                    className="text-xs text-gray-400 block mb-1"
                    title="Whisker on each bar. A bar of means with no spread — the dynamite plot — says nothing about how firm each mean is. SD describes the sample; SE and CI describe the estimate and differ from SD by √n, so name the one shown."
                  >Whisker</label>
                  <select className="select w-full" value={barError} onChange={(e) => setBarError(e.target.value)}>
                    <option value="none">None</option>
                    <option value="sd">SD (spread of the sample)</option>
                    <option value="se">SE (precision of the mean)</option>
                    <option value="ci">95% CI (precision of the mean)</option>
                  </select>
                </div>
              )}
              <label className="flex items-center gap-2 cursor-pointer" title="Puts the categories down the side and the bars across — coord_flip. Worth it as soon as the category names are words rather than codes.">
                <input type="checkbox" checked={barHorizontal} onChange={(e) => setBarHorizontal(e.target.checked)} className="accent-indigo-500" />
                <span className="text-xs text-gray-600">Horizontal bars</span>
              </label>
            </div>
          )}
          {/* Only where it does something. It used to appear on every chart
              but the histogram, including five that never read it — a pie
              with "Color / Group: Sex" chosen drew exactly the same pie, and
              said nothing. A control that silently does nothing is worse
              than an absent one: it reads as a setting that was applied. */}
          {COLOUR_AWARE_CHARTS.has(chartType) && catCols.length > 0 && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">{chartType === "paired" ? "Group (2 levels)" : "Color / Group"}</label>
              <select className="select w-full" value={color} onChange={(e) => setColor(e.target.value)}>
                <option value="">{chartType === "paired" ? "— select —" : "None"}</option>
                {catCols.map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}
          {chartType === "paired" && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Pair ID</label>
              <select className="select w-full" value={pairId} onChange={(e) => setPairId(e.target.value)}>
                <option value="">— select —</option>
                {session.columns.map((c) => <option key={c.name}>{c.name}</option>)}
              </select>
              <p className="text-[10px] text-gray-400 mt-1">Links each matched pair — e.g. PSM's <code>match_set_id</code>, or a case-number column.</p>
            </div>
          )}
          {LINE_STYLE_CHARTS.has(chartType) && color && (
            <label className="flex items-center gap-2 cursor-pointer" title="A dash pattern per group as well as a colour — aes(linetype = group). A figure printed in greyscale, or read by someone who cannot separate the palette, keeps its groups apart.">
              <input type="checkbox" checked={varyLineStyle} onChange={(e) => setVaryLineStyle(e.target.checked)} className="accent-indigo-500" />
              <span className="text-xs text-gray-600">Vary line style by group</span>
            </label>
          )}
          {chartType === "lineplot" && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Value (Y)</label>
              <select className="select w-full" value={lineY} onChange={(e) => setLineY(e.target.value)}>
                {numCols.map((c) => <option key={c}>{c}</option>)}
              </select>
              <p className="text-[10px] text-gray-400 mt-1">
                Each point carries its n on hover — in longitudinal data the n usually falls.
              </p>
            </div>
          )}
          {chartType === "slopeplot" && (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Before</label>
                <select className="select w-full" value={slopeBefore} onChange={(e) => setSlopeBefore(e.target.value)}>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">After</label>
                <select className="select w-full" value={slopeAfter} onChange={(e) => setSlopeAfter(e.target.value)}>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
                <p className="text-[10px] text-gray-400 mt-1">
                  Rows missing either value are excluded and counted below the chart.
                </p>
              </div>
            </>
          )}
          {chartType === "sankey" && (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Stage 2</label>
                <select className="select w-full" value={stage2} onChange={(e) => setStage2(e.target.value)}>
                  <option value="">— select —</option>
                  {[...catCols, ...numCols].map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Stage 3 (optional)</label>
                <select className="select w-full" value={stage3} onChange={(e) => setStage3(e.target.value)}>
                  <option value="">None</option>
                  {[...catCols, ...numCols].map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
            </>
          )}
          {chartType === "stackplot" && (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Stack within each bar</label>
                <select className="select w-full" value={fillCol} onChange={(e) => setFillCol(e.target.value)}>
                  <option value="">— select —</option>
                  {[...catCols, ...numCols].map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Bar height</label>
                <select className="select w-full" value={pieValue} onChange={(e) => setPieValue(e.target.value)}>
                  <option value="">Row count</option>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={stackNormalize} onChange={(e) => setStackNormalize(e.target.checked)} className="accent-indigo-500" />
                <span className="text-xs text-gray-600">Scale every bar to 100%</span>
              </label>
              {stackNormalize && (
                <p className="text-[10px] text-amber-700">
                  A 100% bar hides how many rows it rests on. The per-bar n is printed below.
                </p>
              )}
            </>
          )}
          {chartType === "sets" && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Membership columns</label>
              <div className="max-h-48 overflow-y-auto border border-gray-200 rounded p-2 space-y-1">
                {session.columns.map((c) => (
                  <label key={c.name} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      className="accent-indigo-500"
                      checked={setCols.includes(c.name)}
                      onChange={(e) =>
                        setSetCols(e.target.checked
                          ? [...setCols, c.name]
                          : setCols.filter((n) => n !== c.name))
                      }
                    />
                    <span className="text-xs text-gray-700 truncate">{c.name}</span>
                  </label>
                ))}
              </div>
              <p className="text-[10px] text-gray-400 mt-1">
                Read as members when numeric and non-zero, or yes / true / evet / var.
                Up to 6 columns — regions double with each one.
              </p>
            </div>
          )}
          {chartType === "pie" && (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Slice size</label>
                <select className="select w-full" value={pieValue} onChange={(e) => setPieValue(e.target.value)}>
                  <option value="">Row count</option>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
                <p className="text-[10px] text-gray-400 mt-1">
                  A summed column must be non-negative — a pie splits a whole into parts.
                </p>
              </div>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={donut} onChange={(e) => setDonut(e.target.checked)} className="accent-indigo-500" />
                <span className="text-xs text-gray-600">Donut (hollow centre)</span>
              </label>
            </>
          )}
          {chartType === "balloon" && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Columns</label>
              <select className="select w-full" value={balloonCol} onChange={(e) => setBalloonCol(e.target.value)}>
                <option value="">— select —</option>
                {[...catCols, ...numCols].map((c) => <option key={c}>{c}</option>)}
              </select>
              <p className="text-[10px] text-gray-400 mt-1">
                Dot area is the count; colour is the standardised residual, so cells that
                depart from independence stand out from cells that are merely large.
              </p>
            </div>
          )}
          {chartType === "facet" && (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Panel type</label>
                <select className="select w-full" value={facetKind} onChange={(e) => setFacetKind(e.target.value)}>
                  <option value="boxplot">Box plot</option>
                  <option value="scatter">Scatter</option>
                </select>
              </div>
              {facetKind === "boxplot" && (
                <div>
                  <label className="text-xs text-gray-400 block mb-1">One panel per</label>
                  <select className="select w-full" value={facetMode} onChange={(e) => setFacetMode(e.target.value)}>
                    <option value="level">Level of a column</option>
                    <option value="variable">Variable</option>
                  </select>
                </div>
              )}
              {facetKind === "scatter" && (
                <div>
                  <label className="text-xs text-gray-400 block mb-1">Y axis</label>
                  <select className="select w-full" value={y} onChange={(e) => setY(e.target.value)}>
                    {numCols.map((c) => <option key={c}>{c}</option>)}
                  </select>
                </div>
              )}
              {facetKind === "boxplot" && facetMode === "variable" ? (
                <div>
                  <label className="text-xs text-gray-400 block mb-1">Variables (one panel each)</label>
                  <div className="max-h-48 overflow-y-auto border border-gray-200 rounded p-2 space-y-1">
                    {numCols.map((c) => (
                      <label key={c} className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          className="accent-indigo-500"
                          checked={facetVars.includes(c)}
                          onChange={(e) =>
                            setFacetVars(e.target.checked
                              ? [...facetVars, c]
                              : facetVars.filter((v) => v !== c))}
                        />
                        <span className="text-xs text-gray-600">{c}</span>
                      </label>
                    ))}
                  </div>
                  <p className="text-[10px] text-gray-400 mt-1">
                    Each panel keeps its own axis — milliseconds and a unitless index do not
                    share a scale. Set Color / Group below to split every panel by the same groups.
                  </p>
                </div>
              ) : (
                <div>
                  <label className="text-xs text-gray-400 block mb-1">Split into panels by</label>
                  <select className="select w-full" value={facetCol} onChange={(e) => setFacetCol(e.target.value)}>
                    <option value="">— select —</option>
                    {catCols.map((c) => <option key={c}>{c}</option>)}
                  </select>
                  <p className="text-[10px] text-gray-400 mt-1">
                    Every panel shows the same measurement, so they share one axis range and
                    can be compared by eye.
                  </p>
                </div>
              )}
              {/* facet_wrap(scales =). Panels of one measurement are only
                  comparable while they share an axis, so freeing one is
                  deliberate — it shows each panel's own shape and gives up the
                  comparison. A panel per variable is always free: milliseconds
                  and a unitless index have nothing to share. */}
              {!(facetKind === "boxplot" && facetMode === "variable") && (
                <div>
                  <label className="text-xs text-gray-400 block mb-1">Panel scales</label>
                  <select className="select w-full" aria-label="Panel scales" value={facetScales}
                    onChange={(e) => setFacetScales(e.target.value)}>
                    <option value="fixed">Shared (comparable)</option>
                    <option value="free">Free — each panel its own</option>
                    {facetKind === "scatter" && <option value="free_x">Free X only</option>}
                    <option value="free_y">Free Y only</option>
                  </select>
                  {facetScales !== "fixed" && (
                    <p className="text-[10px] text-amber-700 mt-1">
                      Panels can no longer be compared by eye. Say so in the caption.
                    </p>
                  )}
                </div>
              )}
              <div>
                <label className="text-xs text-gray-400 block mb-1">Columns</label>
                <select className="select w-full" aria-label="Panel columns" value={facetNcol}
                  onChange={(e) => setFacetNcol(e.target.value)}>
                  <option value="auto">Auto (up to 3)</option>
                  {[1, 2, 3, 4, 5, 6].map((n) => <option key={n} value={String(n)}>{n}</option>)}
                </select>
              </div>
            </>
          )}
          {chartType === "errorplot" && (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Centre</label>
                <select className="select w-full" value={errCentre} onChange={(e) => setErrCentre(e.target.value)}>
                  <option value="mean">Mean</option>
                  <option value="median">Median</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Whisker</label>
                <select className="select w-full" value={errSpread} onChange={(e) => setErrSpread(e.target.value)}>
                  <option value="ci">95% CI (precision of the mean)</option>
                  <option value="se">SE (precision of the mean)</option>
                  <option value="sd">SD (spread of the sample)</option>
                  <option value="iqr">IQR (with median)</option>
                </select>
                <p className="text-[10px] text-gray-400 mt-1">
                  SD describes the data; SE and CI describe how well the mean is pinned down.
                  They differ by √n, so say which one the figure shows.
                </p>
              </div>
            </>
          )}
          {(chartType === "boxplot" || chartType === "violin" || chartType === "raincloud" || chartType === "strip") && (
            <div className="pt-2 border-t border-gray-100 space-y-2">
              <p className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Axes</p>
              <label className="flex items-center gap-2 cursor-pointer" title="Puts the groups down the side instead of along the bottom. Worth it as soon as the category names are words rather than codes — a rotated tick label is slower to read than a horizontal one.">
                <input type="checkbox" checked={horizontal} onChange={(e) => setHorizontal(e.target.checked)} className="accent-indigo-500" />
                <span className="text-xs text-gray-600">Horizontal (groups down the side)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer" title="Log scale on the value axis. Ratios and composite indices (SII, PLR) span an order of magnitude or more; on a linear axis a handful of large values squash the rest against one end.">
                <input type="checkbox" checked={logValue} onChange={(e) => setLogValue(e.target.checked)} className="accent-indigo-500" />
                <span className="text-xs text-gray-600">Log scale on the value axis</span>
              </label>
              <p className="text-[10px] font-bold text-gray-400 uppercase tracking-wider pt-1">Comparisons</p>
              {/* A raincloud IS the scattered points, so the toggle would be a
                  control that cannot change anything. */}
              {chartType !== "raincloud" && (
                <label className="flex items-center gap-2 cursor-pointer" title="Draws every raw observation over the box. A box alone hides the sample size and any clustering; with small clinical samples showing the points is now expected.">
                  <input type="checkbox" checked={showPoints} onChange={(e) => setShowPoints(e.target.checked)} className="accent-indigo-500" />
                  <span className="text-xs text-gray-600">Show every point</span>
                </label>
              )}
              {chartType === "boxplot" && (
                <label className="flex items-center gap-2 cursor-pointer" title="Marks each group's mean with a diamond. The box draws the median; the distance between the two is the skew, and the skew is what the parametric-vs-non-parametric choice rests on.">
                  <input type="checkbox" checked={showMean} onChange={(e) => setShowMean(e.target.checked)} className="accent-indigo-500" />
                  <span className="text-xs text-gray-600">Mark the mean</span>
                </label>
              )}
              <label className="flex items-center gap-2 cursor-pointer" title="Adds a significance bracket over each pair of groups, like ggpubr's stat_compare_means. Needs a Color / Group column.">
                <input type="checkbox" checked={showBrackets} onChange={(e) => setShowBrackets(e.target.checked)} className="accent-indigo-500" />
                <span className="text-xs text-gray-600">Significance brackets</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer" title="Prints n, mean ± SD, median and IQR per group under the chart — ggpubr's ggsummarystats.">
                <input type="checkbox" checked={showSummary} onChange={(e) => setShowSummary(e.target.checked)} className="accent-indigo-500" />
                <span className="text-xs text-gray-600">Summary table below</span>
              </label>
              {showBrackets && (
                <div className="space-y-2 pl-1">
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Test</label>
                    <select className="select w-full" value={cmpMethod} onChange={(e) => setCmpMethod(e.target.value)}>
                      <option value="auto">Auto (Shapiro-Wilk decides)</option>
                      <option value="welch">Welch t-test</option>
                      <option value="t">Student t-test</option>
                      <option value="wilcoxon">Mann-Whitney U</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Correct for multiplicity</label>
                    <select className="select w-full" value={cmpAdjust} onChange={(e) => setCmpAdjust(e.target.value)}>
                      <option value="holm">Holm</option>
                      <option value="fdr">Benjamini-Hochberg (FDR)</option>
                      <option value="bonferroni">Bonferroni</option>
                      <option value="none">None (raw p)</option>
                    </select>
                    {cmpAdjust === "none" && (
                      <p className="text-[10px] text-amber-700 mt-1">
                        Every extra pair raises the chance of a false star. Say in the caption
                        that these are unadjusted.
                      </p>
                    )}
                  </div>
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Label</label>
                    <select className="select w-full" value={cmpLabel} onChange={(e) => setCmpLabel(e.target.value)}>
                      <option value="stars">Stars (*, **, ***)</option>
                      <option value="p">p-value</option>
                    </select>
                  </div>
                  {!color && (
                    <p className="text-[10px] text-amber-700">
                      Select a Color / Group column — brackets compare its levels.
                    </p>
                  )}
                </div>
              )}
            </div>
          )}
          {chartType === "scatter" && (
            <div className="pt-2 border-t border-gray-100 space-y-2">
              <p className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Agreement options</p>
              <label className="flex items-center gap-2 cursor-pointer" title="Use when the values span orders of magnitude — p-values, concentrations, counts. Zero and negative values cannot be shown and are dropped with a warning.">
                <input type="checkbox" checked={logX} onChange={(e) => setLogX(e.target.checked)} className="accent-indigo-500" />
                <span className="text-xs text-gray-600">Log X axis</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer" title="Same as above for the vertical axis. The fit is recomputed in log space so it stays a straight line.">
                <input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} className="accent-indigo-500" />
                <span className="text-xs text-gray-600">Log Y axis</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer" title="Draws y = x. Only meaningful when both axes carry the same quantity — a reported value against a recomputed one, or a method against a reference. Points off the line are the disagreements.">
                <input type="checkbox" checked={identityLine} onChange={(e) => setIdentityLine(e.target.checked)} className="accent-indigo-500" />
                <span className="text-xs text-gray-600">y = x reference line</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer" title="Draws the region expected to contain 95% of a bivariate normal cloud, per group. It describes the spread, not the fit.">
                <input type="checkbox" checked={ellipse} onChange={(e) => setEllipse(e.target.checked)} className="accent-indigo-500" />
                <span className="text-xs text-gray-600">95% confidence ellipse</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer" title="Adds a histogram along each axis, so the marginal distribution of each variable is visible alongside their joint behaviour.">
                <input type="checkbox" checked={marginal} onChange={(e) => setMarginal(e.target.checked)} className="accent-indigo-500" />
                <span className="text-xs text-gray-600">Marginal histograms</span>
              </label>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Marker shape by</label>
                <select className="select w-full" value={shapeCol} onChange={(e) => setShapeCol(e.target.value)}>
                  <option value="">None</option>
                  {catCols.map((c) => <option key={c}>{c}</option>)}
                </select>
                <p className="text-[10px] text-gray-400 mt-1">
                  Survives printing in greyscale and is readable without colour vision.
                </p>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Point labels</label>
                <select className="select w-full" value={labelCol} onChange={(e) => setLabelCol(e.target.value)}>
                  <option value="">None</option>
                  {session.columns.map((c) => <option key={c.name}>{c.name}</option>)}
                </select>
                <p className="text-[10px] text-gray-400 mt-1">Names each point on the figure — use it to call out the rows that miss the line.</p>
              </div>
              {/* A continuous colour scale. Offered only with no Color /
                  Group column: one set of markers cannot carry a legend of
                  groups and a colour bar of values at once, and a control that
                  silently loses to another is worse than an absent one. */}
              <label className="flex items-center gap-2 cursor-pointer" title="geom_bin2d. Past a few thousand points a scatter is a solid blob that hides where the mass sits; the grid counts the points in each cell and colours by the count. The fit is still computed from every row.">
                <input type="checkbox" checked={bin2d} onChange={(e) => setBin2d(e.target.checked)} className="accent-indigo-500" />
                <span className="text-xs text-gray-600">Bin into a 2-D grid (dense clouds)</span>
              </label>
              {bin2d && (
                <div>
                  <label className="text-xs text-gray-400 block mb-1">Grid: {bin2dBins} × {bin2dBins}</label>
                  <input type="range" min={5} max={100} step={5} value={bin2dBins}
                    aria-label="Grid bins"
                    onChange={(e) => setBin2dBins(+e.target.value)} className="w-full accent-indigo-500" />
                  <p className="text-[10px] text-gray-400 mt-1">
                    The points are replaced by the grid, so the marker shape and point labels no longer apply.
                  </p>
                </div>
              )}
              {!color && !bin2d && (
                <>
                  <div>
                    <label className="text-xs text-gray-400 block mb-1"
                      title="Colours every point by the value of a numeric column, with a colour bar — ggplot2's continuous scale. Use it for a third quantity: age over a risk plot, follow-up over an agreement plot.">
                      Colour by value
                    </label>
                    <select className="select w-full" aria-label="Colour by value" value={gradientCol}
                      onChange={(e) => setGradientCol(e.target.value)}>
                      <option value="">None</option>
                      {numCols.map((c) => <option key={c}>{c}</option>)}
                    </select>
                  </div>
                  {gradientCol && (
                    <div>
                      <label className="text-xs text-gray-400 block mb-1">Colour ramp</label>
                      <select className="select w-full" aria-label="Colour ramp" value={gradientScale}
                        onChange={(e) => setGradientScale(e.target.value)}>
                        {GRADIENT_SCALES.map((g) => <option key={g.value} value={g.value}>{g.label}</option>)}
                      </select>
                      <p className="text-[10px] text-gray-400 mt-1">
                        Viridis and Cividis stay ordered in greyscale and for colour-blind readers;
                        Red-Blue is for a value with a meaningful middle.
                      </p>
                    </div>
                  )}
                </>
              )}
              {bin2d && (
                <div>
                  <label className="text-xs text-gray-400 block mb-1">Colour ramp</label>
                  <select className="select w-full" aria-label="Colour ramp" value={gradientScale}
                    onChange={(e) => setGradientScale(e.target.value)}>
                    {GRADIENT_SCALES.map((g) => <option key={g.value} value={g.value}>{g.label}</option>)}
                  </select>
                </div>
              )}
              <p className="text-[10px] font-bold text-gray-400 uppercase tracking-wider pt-1">Trend line</p>
              <div>
                <label
                  className="text-xs text-gray-400 block mb-1"
                  title="geom_smooth. The straight line comes with the 95% CI of the fitted line. LOESS follows the data locally — the shape to show when the relation bends — and carries no band, because it has none it can justify."
                >Method</label>
                <select className="select w-full" aria-label="Trend line method" value={fitMethod} onChange={(e) => setFitMethod(e.target.value)}>
                  <option value="lm">Linear (lm) with 95% CI</option>
                  <option value="loess">LOESS (local curve)</option>
                  <option value="none">None</option>
                </select>
              </div>
              {fitMethod === "loess" && (
                <div>
                  <label className="text-xs text-gray-400 block mb-1">Span: {loessSpan.toFixed(2)}</label>
                  <input type="range" min={0.2} max={1} step={0.05} value={loessSpan}
                    aria-label="LOESS span"
                    onChange={(e) => setLoessSpan(+e.target.value)} className="w-full accent-indigo-500" />
                  <p className="text-[10px] text-gray-400 mt-1">
                    The share of points each local fit sees. Smaller follows every wiggle; 0.75 is ggplot2's default.
                  </p>
                </div>
              )}
              {fitMethod !== "none" && color && (
                <label className="flex items-center gap-2 cursor-pointer" title="One trend per level of the Color / Group column, in that group's colour — geom_smooth(aes(colour = group)). Replaces the single overall line.">
                  <input type="checkbox" checked={fitPerGroup} onChange={(e) => setFitPerGroup(e.target.checked)} className="accent-indigo-500" />
                  <span className="text-xs text-gray-600">One trend per group</span>
                </label>
              )}
            </div>
          )}
          {chartType === "histogram" && (
            <div className="space-y-2">
              <div>
                <label className="text-xs text-gray-400 block mb-1">Bins: {bins}</label>
                <input type="range" min={5} max={100} value={bins} onChange={(e) => setBins(+e.target.value)}
                  disabled={parseRefValue(binwidth) !== null} className="w-full accent-indigo-500 disabled:opacity-40" />
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1" title="ggplot2's binwidth. A width in the variable's own units — 5 mmHg, 1 year — beats a bin count when the units mean something; edges are aligned to multiples of it. Leave blank to use the bin count.">
                  Bin width (overrides bins)
                </label>
                <input className="select w-full text-sm" aria-label="Bin width" placeholder="blank = use bin count"
                  value={binwidth} onChange={(e) => setBinwidth(e.target.value)} />
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1" title="after_stat(count | density). Density makes groups of different sizes comparable by shape; percent does the same in units a reader can quote.">Y axis</label>
                <select className="select w-full" aria-label="Histogram y axis" value={histStat} onChange={(e) => setHistStat(e.target.value)}>
                  <option value="count">Count</option>
                  <option value="density">Density</option>
                  <option value="percent">Percent of observations</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Show</label>
                <select className="select w-full" aria-label="Histogram display" value={histDisplay} onChange={(e) => setHistDisplay(e.target.value)}>
                  <option value="both">Bars with KDE curve</option>
                  <option value="bars">Bars only</option>
                  <option value="density">Density curve only (geom_density)</option>
                </select>
              </div>
              {color && histDisplay !== "density" && (
                <div>
                  <label className="text-xs text-gray-400 block mb-1" title="position = identity (overlaid, translucent), stack, or dodge (side by side).">Groups</label>
                  <select className="select w-full" aria-label="Histogram group position" value={histPosition} onChange={(e) => setHistPosition(e.target.value)}>
                    <option value="overlay">Overlaid (translucent)</option>
                    <option value="stack">Stacked</option>
                    <option value="dodge">Side by side</option>
                  </select>
                </div>
              )}
              <label className="flex items-center gap-2 cursor-pointer" title="geom_rug: a tick per observation along the x axis, so the reader sees where the data actually sit — gaps, clumps, a lone outlier — which bins can hide.">
                <input type="checkbox" checked={histRug} onChange={(e) => setHistRug(e.target.checked)} className="accent-indigo-500" />
                <span className="text-xs text-gray-600">Rug (a tick per observation)</span>
              </label>
            </div>
          )}
          <button className="btn-primary w-full mt-2" onClick={run} disabled={loading}>
            {loading ? "Generating…" : "Generate Chart"}
          </button>
          {error && <p className="text-red-500 text-xs mt-2">{error}</p>}
        </div>

        {/* Reference lines — geom_hline / geom_vline. A threshold the reader
            is meant to judge the data against: LDL 100, BMI 30, a cut-off on
            the axis. Applied live; no regeneration needed. */}
        {REF_LINE_CHARTS.has(chartType) && (
          <div className="panel space-y-2 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-700">Reference lines</h3>
              <button
                type="button"
                className="text-[11px] text-indigo-600 hover:text-indigo-800"
                onClick={() => setRefLines([...refLines, { axis: "y", value: "", label: "" }])}
              >+ Add</button>
            </div>
            {refLines.length === 0 && (
              <p className="text-[10px] text-gray-400">
                A dashed line at a value on the X or Y axis, with a label — a clinical threshold or a cut-off.
              </p>
            )}
            {refLines.map((line, i) => {
              const update = (patch: Partial<RefLineDraft>) =>
                setRefLines(refLines.map((l, j) => (j === i ? { ...l, ...patch } : l)));
              const bad = line.value.trim() !== "" && parseRefValue(line.value) === null;
              return (
                <div key={i} className="space-y-1 border-t border-gray-100 pt-2">
                  <div className="flex gap-1.5">
                    <select
                      className="select text-xs w-14"
                      aria-label={`Reference line ${i + 1} axis`}
                      value={line.axis}
                      onChange={(e) => update({ axis: e.target.value as "x" | "y" })}
                    >
                      <option value="y">Y =</option>
                      <option value="x">X =</option>
                    </select>
                    <input
                      className={`select w-full text-xs ${bad ? "border-red-400" : ""}`}
                      aria-label={`Reference line ${i + 1} value`}
                      placeholder="value"
                      value={line.value}
                      onChange={(e) => update({ value: e.target.value })}
                    />
                    <button
                      type="button"
                      aria-label={`Remove reference line ${i + 1}`}
                      className="px-1 text-gray-400 hover:text-red-500 text-xs"
                      onClick={() => setRefLines(refLines.filter((_, j) => j !== i))}
                    >✕</button>
                  </div>
                  <input
                    className="select w-full text-xs"
                    aria-label={`Reference line ${i + 1} label`}
                    placeholder="label (optional)"
                    value={line.label}
                    onChange={(e) => update({ label: e.target.value })}
                  />
                  {bad && <p className="text-[10px] text-red-500">Not a number.</p>}
                </div>
              );
            })}
          </div>
        )}

        {/* coord_cartesian(xlim =, ylim =). A window on the axes that zooms
            rather than dropping rows: the statistics printed under the chart
            are still computed from every row, which is the difference between
            this and filtering the data. */}
        {(numericAxes.x || numericAxes.y) && (
          <div className="panel space-y-2 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
            <h3 className="text-sm font-semibold text-gray-700">Axis limits</h3>
            {numericAxes.x && (
              <div className="flex items-center gap-1.5">
                <span className="w-4 text-[11px] text-gray-400">X</span>
                <input className="select w-full text-xs" aria-label="X axis minimum" placeholder="min"
                  value={xMin} onChange={(e) => setXMin(e.target.value)} />
                <input className="select w-full text-xs" aria-label="X axis maximum" placeholder="max"
                  value={xMax} onChange={(e) => setXMax(e.target.value)} />
              </div>
            )}
            {numericAxes.y && (
              <div className="flex items-center gap-1.5">
                <span className="w-4 text-[11px] text-gray-400">Y</span>
                <input className="select w-full text-xs" aria-label="Y axis minimum" placeholder="min"
                  value={yMin} onChange={(e) => setYMin(e.target.value)} />
                <input className="select w-full text-xs" aria-label="Y axis maximum" placeholder="max"
                  value={yMax} onChange={(e) => setYMax(e.target.value)} />
              </div>
            )}
            <p className="text-[10px] text-gray-400">
              Zooms the view; no row is dropped, so the statistics under the chart are unchanged.
              Leave one end blank to let the data set it.
            </p>
            {(xMin || xMax || yMin || yMax) && (
              <button type="button"
                className="text-[10px] text-gray-500 hover:text-indigo-600"
                onClick={() => { setXMin(""); setXMax(""); setYMin(""); setYMax(""); }}>
                Clear limits
              </button>
            )}
          </div>
        )}

        {/* Custom Labels Panel */}
        {plotData && (
          <div className="panel space-y-3 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
            <h3 className="text-sm font-semibold text-gray-700 border-b pb-2">Custom Labels</h3>
            
            {/* Custom Chart Title */}
            <div>
              <label className="text-xs font-medium text-gray-500 block mb-1">Chart Title</label>
              <input
                type="text"
                className="select w-full text-xs py-1 px-2 border rounded"
                value={customTitle}
                onChange={(e) => setCustomTitle(e.target.value)}
                placeholder="Title..."
              />
            </div>

            {/* Custom X Axis Label */}
            <div>
              <label className="text-xs font-medium text-gray-500 block mb-1">X-Axis Label</label>
              <input
                type="text"
                className="select w-full text-xs py-1 px-2 border rounded"
                value={customXLabel}
                onChange={(e) => setCustomXLabel(e.target.value)}
                placeholder="X-axis label..."
              />
            </div>

            {/* Custom Y Axis Label */}
            <div>
              <label className="text-xs font-medium text-gray-500 block mb-1">Y-Axis Label</label>
              <input
                type="text"
                className="select w-full text-xs py-1 px-2 border rounded"
                value={customYLabel}
                onChange={(e) => setCustomYLabel(e.target.value)}
                placeholder="Y-axis label..."
              />
            </div>
          </div>
        )}

        {/* Chart guidance */}
        <div className="panel bg-gray-50 border-gray-200 p-4 rounded-2xl">
          <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">Tip</p>
          <p className="text-xs text-gray-600 leading-relaxed">{
            chartType === "histogram" ? "Shows the frequency distribution of a single numeric variable. The KDE (kernel density) overlay estimates the smooth probability density. Skewed histograms suggest using median instead of mean." :
            chartType === "scatter" ? "Reveals relationships between two continuous variables. The regression line and R² show linear fit strength. Add a Color variable to see group-specific patterns." :
            chartType === "boxplot" ? "Compares distributions across groups. The box shows Q1–Q3 (IQR), the line is the median, whiskers extend to 1.5×IQR. Points beyond whiskers are outliers." :
            chartType === "strip" ? "Every observation as a point, with a rule at the group median and nothing else. Use it where a box or violin would over-claim: small or very uneven groups, where a kernel drawn through seven cases asserts a shape the data does not support. With groups down the side and a log value axis it is the standard figure for a ratio or composite index (SII, PLR) across histology subtypes." :
            chartType === "violin" ? "Combines a box plot with a kernel density estimate. The wider the violin, the more data points at that value. Better than box plots for showing bimodal or skewed distributions." :
            chartType === "paired" ? "Matched-pair comparison: a box per group plus a line connecting each pair's two values (e.g. PSM-matched cohorts). Pair ID must link exactly one row per group — PSM's match_set_id or any per-case ID column works." :
            chartType === "lineplot" ? "Group means across an ordered axis — the repeated-measures figure: one line per arm across visits, with a band for the uncertainty. Each point's n is on hover, and a warning appears when a group loses half its subjects along the axis, because a thinning line looks identical to a stable one." :
            chartType === "slopeplot" ? "Before and after, one line per subject. Shows what a mean change conceals: whether everyone moved a little or a few moved a lot, and who moved the other way. Rows missing either measurement are excluded and counted — a paired test run on whoever happened to have both values is a different analysis from the one the figure implies." :
            chartType === "sankey" ? "Flow between successive states — treatment lines, stage transitions, care pathways. A level appearing at two stages becomes two separate nodes, so 'Medical → Medical' reads as staying put rather than as a loop." :
            chartType === "stackplot" ? "Composition within each bar. Useful when the parts matter as much as the total. Scaling every bar to 100% makes the shares comparable but hides how many rows each bar rests on, so the per-bar n is printed below the chart." :
            chartType === "ridgeplot" ? "One density curve per group, stacked. Good for comparing the shape of many distributions at once — shifts, skew, bimodality — where a box plot would show only quartiles. Every group is evaluated on the same grid so widths are comparable, and groups too small to smooth are named rather than drawn." :
            chartType === "sets" ? "How membership columns overlap. Each bar is an exclusive region: rows in exactly that combination and no other, which is what a Venn region means too. Up to three sets a Venn is readable; past that these bars are the honest rendering." :
            chartType === "pie" ? "Composition of a single categorical variable. Readers judge angles poorly, so percentages are printed on each slice and a long tail of thin wedges is folded into 'Other'. If the point is to compare categories rather than show shares of a whole, a bar chart reads more accurately." :
            chartType === "balloon" ? "A cross-tabulation drawn as dots: area is the cell count, colour is the standardised residual (observed − expected, scaled). The residual is what makes this more than a restatement of the marginals — it shows which cells actually depart from independence. The χ² test appears below; watch for the warning when an expected count falls under 5." :
            chartType === "facet" ? "One panel per level of a grouping variable — small multiples. Every panel shares a single axis range computed across all of them, because per-panel autoscaling makes different distributions look identical. Panels beyond the limit are dropped with a warning rather than quietly omitted." :
            chartType === "errorplot" ? "Centre and spread per group, without the box. Pick the whisker deliberately: SD says how spread the observations are, SE and CI say how precisely the mean is estimated. They differ by a factor of √n, so an SE plot looks far tighter than an SD plot on identical data — journals ask which one you used." :
            chartType === "ecdf" ? "The empirical cumulative distribution: for each value on the x axis, the proportion of observations at or below it. Unlike a histogram it involves no binning choice, so it cannot be made to tell a different story by changing bin width. With exactly two groups the largest vertical gap between the curves is the Kolmogorov-Smirnov D, reported below the chart." :
            chartType === "dumbbell" ? "Two values per category, joined by a line — the line length is the point. Use it to compare a reference against an observation across many variables at once: an expected effect size against the one computed from the raw data, a baseline against follow-up, model A against model B. Rows are ranked so the largest disagreement sits at the top. Needs exactly one row per category." :
            "Shows counts or aggregated values for categories. Use for comparing frequencies across groups. Add a Color variable for stacked/grouped comparisons."
          }</p>
          {chartType === "scatter" && (
            <p className="text-xs text-gray-600 leading-relaxed mt-2 pt-2 border-t border-gray-200">
              For an <strong>agreement plot</strong> — a reported value against a recomputed one —
              turn on both log axes and the y = x line, and set Point labels to the variable name.
              Points below the line are the ones reported smaller than they should be.
            </p>
          )}
        </div>

        {/* Backend warnings (e.g. points a log axis cannot show) */}
        {Array.isArray(plotData?.warnings) && (plotData.warnings as unknown[]).length > 0 && (
          <div className="panel bg-amber-50 border border-amber-200 p-3 rounded-2xl">
            {(plotData.warnings as Array<{ message: string }>).map((w, i) => (
              <p key={i} className="text-xs text-amber-800 leading-relaxed">{w.message}</p>
            ))}
          </div>
        )}

        {/* Slope plot: the paired test and how many pairs it rests on */}
        {plotData?.type === "slopeplot" && (() => {
          const t = (plotData.test_result ?? {}) as { test?: string; p?: number | null; selected_by?: string; note?: string };
          return (
            <div className="panel bg-gray-50 border-gray-200 p-4 rounded-2xl">
              <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">Change</p>
              <p className="text-xs text-gray-600 leading-relaxed">
                {String(plotData.n_pairs)} complete pairs · mean change {Number(plotData.mean_change).toFixed(2)},
                median {Number(plotData.median_change).toFixed(2)} · {String(plotData.n_decreased)} down,
                {" "}{String(plotData.n_increased)} up, {String(plotData.n_unchanged)} unchanged.
                {t.test && (
                  <> {t.test} ({t.selected_by}):{" "}
                    {t.p == null ? t.note : `p = ${t.p < 1e-4 ? t.p.toExponential(2) : t.p.toFixed(4)}`}</>
                )}
              </p>
            </div>
          );
        })()}

        {/* Stacked bar: the denominator a 100% bar hides */}
        {plotData?.type === "stackplot" && Boolean(plotData.normalize) && (
          <div className="panel bg-gray-50 border-gray-200 p-4 rounded-2xl">
            <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">Rows per bar</p>
            <p className="text-xs text-gray-600 leading-relaxed">
              {Object.entries(plotData.totals as Record<string, number>)
                .map(([k, v]) => `${k}: ${v}`).join(" · ")}
            </p>
          </div>
        )}

        {/* Set overlap: what the bars leave out */}
        {plotData?.type === "sets" && (
          <div className="panel bg-gray-50 border-gray-200 p-4 rounded-2xl">
            <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">Sets</p>
            <p className="text-xs text-gray-600 leading-relaxed">
              {Object.entries(plotData.set_sizes as Record<string, number>)
                .map(([k, v]) => `${k}: ${v}`).join(" · ")}.
              {" "}{String(plotData.n_in_no_set)} of {String(plotData.n_rows)} rows belong to no set.
              {!plotData.renderable_as_venn && " Too many sets for a Venn; shown as exclusive-region bars."}
            </p>
          </div>
        )}

        {/* Balloon plot: the test behind the colours */}
        {plotData?.type === "balloon" && (
          <div className="panel bg-gray-50 border-gray-200 p-4 rounded-2xl">
            <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">Independence</p>
            <p className="text-xs text-gray-600 leading-relaxed">
              χ² = {Number(plotData.chi2).toFixed(2)}, df = {String(plotData.df)},{" "}
              p = {Number(plotData.p) < 1e-4 ? Number(plotData.p).toExponential(2) : Number(plotData.p).toFixed(4)}{" "}
              (n = {String(plotData.n)}). Blue cells hold more observations than independence predicts,
              red fewer; a residual beyond ±2 is the usual threshold for calling a cell a contributor.
            </p>
          </div>
        )}

        {/* Summary table under the plot — ggsummarystats */}
        {summary && (
          <div className="panel bg-white border border-gray-200 p-3 rounded-2xl overflow-x-auto">
            <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">Summary</p>
            <table className="text-[11px] w-full">
              <thead>
                <tr className="text-gray-400 text-left">
                  <th className="pr-2 font-medium">Group</th>
                  <th className="pr-2 font-medium">n</th>
                  <th className="pr-2 font-medium">Mean ± SD</th>
                  <th className="pr-2 font-medium">Median (IQR)</th>
                </tr>
              </thead>
              <tbody>
                {(summary.rows as Array<Record<string, number | string>>).map((r) => (
                  <tr key={String(r.group)} className="text-gray-700 border-t border-gray-100">
                    <td className="pr-2 py-0.5">{String(r.group)}</td>
                    <td className="pr-2 py-0.5">
                      {String(r.n)}{Number(r.n_missing) > 0 && <span className="text-amber-600"> (+{String(r.n_missing)} missing)</span>}
                    </td>
                    <td className="pr-2 py-0.5">{Number(r.mean).toFixed(2)} ± {Number(r.sd).toFixed(2)}</td>
                    <td className="pr-2 py-0.5">
                      {Number(r.median).toFixed(2)} ({Number(r.q1).toFixed(2)}–{Number(r.q3).toFixed(2)})
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* What the stars actually mean — a figure legend the user can copy */}
        {comparisons && (
          <div className="panel bg-gray-50 border-gray-200 p-4 rounded-2xl">
            <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">Comparisons</p>
            <p className="text-xs text-gray-600 leading-relaxed">
              {String(comparisons.test)} ({String(comparisons.test_selected_by)}).{" "}
              {comparisons.p_shown_is_adjusted
                ? `p-values adjusted for ${(comparisons.comparisons as unknown[]).length} comparisons (${String(comparisons.p_adjust)}).`
                : `p-values are unadjusted across ${(comparisons.comparisons as unknown[]).length} comparisons.`}
              {(comparisons.omnibus as { test?: string; p?: number })?.test && (
                <> Omnibus {String((comparisons.omnibus as { test: string }).test)}{" "}
                  p = {Number((comparisons.omnibus as { p: number }).p).toExponential(2)}.</>
              )}
            </p>
            <p className="text-[10px] text-gray-500 mt-1">
              **** ≤ 0.0001 · *** ≤ 0.001 · ** ≤ 0.01 · * ≤ 0.05 · ns otherwise.
            </p>
          </div>
        )}

        {/* Dumbbell summary — the numbers behind the picture */}
        {plotData?.type === "dumbbell" && (() => {
          const s = plotData.summary as Record<string, number | string>;
          return (
            <div className="panel bg-gray-50 border-gray-200 p-4 rounded-2xl">
              <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">Gaps</p>
              <p className="text-xs text-gray-600 leading-relaxed">
                {s.n} categories · median absolute gap {Number(s.median_abs_gap).toFixed(3)} ·
                largest {Number(s.max_abs_gap).toFixed(3)} at <strong>{String(s.largest_gap_category)}</strong>.
                {" "}{s.n_end_above_start} above the reference, {s.n_end_below_start} below.
              </p>
            </div>
          );
        })()}
      </div>

      {/* Plot area */}
      <div className="flex-1 panel min-h-0 relative bg-white border border-gray-200 shadow-sm rounded-2xl p-4 overflow-y-auto">
        {traces ? (
          <TitledPlot
            // Remount when the chart type changes so the height below is
            // applied; without it the first mount's height sticks.
            key={chartType}
            plotRefOut={chartRef}
            storageKey={`charts:${chartType}:${x}`}
            data={traces}
            layout={{
              ...layout,
              // A dumbbell needs one legible label per row. At a fixed height
              // Plotly starts dropping tick labels past ~15 rows, which leaves
              // markers the reader cannot attribute to anything.
              ...(plotData?.type === "dumbbell"
                ? { height: Math.min(1400, Math.max(360, 26 * ((plotData.rows as unknown[])?.length ?? 0) + 140)) }
                : {}),
              xaxis: {
                ...(layout.xaxis as PlotLayout),
                ...pairedXAxisOverride(chartType, plotData, session),
                ...(plotData?.log_x ? { type: "log" } : {}),
                // The grouped charts have one value axis, and which of the two
                // it is moves with the orientation. Categories must never be
                // logged — that axis carries names, not numbers.
                ...(valueAxisIsLog && horizontal ? { type: "log" } : {}),
                ...(groupedChart && !horizontal ? { type: "category", automargin: true } : {}),
                ...(marginalAxes.xDomain ? { domain: marginalAxes.xDomain } : {}),
                // Last, so a manual window wins over the data-driven range.
                ...xWindow,
              },
              yaxis: {
                ...(layout.yaxis as PlotLayout),
                ...(plotData?.log_y ? { type: "log" } : {}),
                ...(valueAxisIsLog && !horizontal ? { type: "log" } : {}),
                // Category order comes from the trace arrays, not alphabetical.
                ...(plotData?.type === "dumbbell" ? { type: "category", automargin: true } : {}),
                // automargin so a long histology name is not clipped.
                ...(groupedChart && horizontal ? { type: "category", automargin: true } : {}),
                // Flipped bars: categories down the side, values along the bottom.
                ...(barFlipped ? { type: "category", automargin: true } : {}),
                ...(marginalAxes.yDomain ? { domain: marginalAxes.yDomain } : {}),
                ...yWindow,
              },
              ...facetOverlay,
              ...(overlayShapes.length ? { shapes: overlayShapes } : {}),
              ...(overlayAnnotations.length ? { annotations: overlayAnnotations } : {}),
              ...marginalAxes.extra,
              ...(plotData?.type === "stackplot" ? { barmode: "stack" } : {}),
              ...(plotData?.type === "histogram"
                ? { barmode: histPosition === "stack" ? "stack" : histPosition === "dodge" ? "group" : "overlay", bargap: 0.05 }
                : {}),
              ...(plotData?.type === "ridgeplot"
                ? { yaxis: { showticklabels: false, title: { text: "" } } }
                : {}),
            }}
            config={{ responsive: true, displayModeBar: true, displaylogo: false }}
            defaultTitle={customTitle || (plotData?.x ? String(plotData.x) : "")}
            defaultSubtitle=""
            defaultXAxis={customXLabel}
            defaultYAxis={customYLabel}
          />
        ) : (
          <div className="h-full flex items-center justify-center text-gray-400">
            Configure and generate a chart
          </div>
        )}
        {/* The numbers a scatter caption is written from. Pearson answers a
            straight-line question and Spearman a monotone one; on skewed
            clinical data they part company, and quoting one while the reader
            assumes the other is how a figure legend goes wrong. */}
        {plotData?.type === "scatter" && (() => {
          const reg = plotData.regression as {
            r?: number | null; r2?: number | null; p?: number | null; n?: number | null;
            spearman?: { rho?: number | null; p?: number | null };
            note?: string;
          } | undefined;
          if (!reg) return null;
          if (reg.r === null || reg.r === undefined) {
            return reg.note
              ? <p className="mt-2 text-[11px] text-gray-400">{reg.note}</p>
              : null;
          }
          const rho = reg.spearman?.rho;
          const fit = String(plotData.fit ?? "lm");
          const perGroup = (plotData.regressions as Array<{ group: unknown; r?: number | null; r2?: number | null; n?: number; note?: string }> | undefined) ?? [];
          const fitNote = fit === "loess"
            ? " The curve is a LOESS fit; it carries no band."
            : fit === "none" ? "" : " The shaded band is the 95% CI of the fitted line.";
          return (
            <p className="mt-2 text-[11px] text-gray-500">
              Pearson r = {reg.r.toFixed(3)} (R² = {(reg.r2 ?? 0).toFixed(3)}), p = {fmtP(reg.p ?? NaN)}
              {typeof rho === "number" && (
                <> · Spearman ρ = {rho.toFixed(3)}, p = {fmtP(reg.spearman?.p ?? NaN)}</>
              )}
              {typeof reg.n === "number" && <> · n = {reg.n}</>}
              .{fitNote}
              {perGroup.length > 0 && (
                <> Per group: {perGroup.map((g) => (
                  typeof g.r === "number"
                    ? `${String(g.group)} r = ${g.r.toFixed(3)} (n = ${g.n ?? "?"})`
                    : `${String(g.group)}: ${g.note ?? "no fit"}`
                )).join(" · ")}.</>
              )}
            </p>
          );
        })()}
      </div>
    </div>
  );
}

/** Domains that shrink the scatter to make room for the marginal strips.
 *
 *  Returns only the extra axes plus the domains; the caller merges the domain
 *  into the main axis objects rather than replacing them, so a log scale or
 *  any other axis setting survives.
 */
function marginalLayout(plotData: Record<string, unknown> | null): {
  xDomain?: number[];
  yDomain?: number[];
  extra: Record<string, unknown>;
} {
  const marg = (plotData?.marginal ?? {}) as Record<string, unknown[]>;
  if (!plotData || plotData.type !== "scatter") return { extra: {} };
  const hasX = Array.isArray(marg.x) && marg.x.length > 0;
  const hasY = Array.isArray(marg.y) && marg.y.length > 0;
  if (!hasX && !hasY) return { extra: {} };
  const main = 0.82;
  return {
    xDomain: [0, hasY ? main : 1],
    yDomain: [0, hasX ? main : 1],
    extra: {
      xaxis2: { domain: [main + 0.02, 1], anchor: "y", showticklabels: false },
      yaxis2: { domain: [main + 0.02, 1], anchor: "x", showticklabels: false },
      bargap: 0.05,
    },
  };
}

/** Grid of subplots for a faceted chart.
 *
 *  When every panel shows the SAME measurement they are given one range,
 *  computed by the backend across all of them: letting Plotly autoscale each
 *  panel is the classic small-multiples error, where two panels look alike
 *  while their axes differ tenfold. When each panel is a DIFFERENT
 *  measurement the backend sends no shared range and each panel keeps its
 *  own — milliseconds and a unitless index on one axis would flatten the
 *  index into a line at the bottom.
 */
function facetLayout(plotData: Record<string, unknown> | null): Record<string, unknown> {
  if (!plotData || plotData.type !== "facet") return {};
  const panels = (plotData.panels as Array<Record<string, unknown>>) ?? [];
  if (!panels.length) return {};
  const shared = (plotData.shared_range ?? {}) as { x?: number[]; y?: number[] };
  // A freed axis simply has no shared range, so nothing is set on it and
  // Plotly autoscales that panel — which is what a free scale means, and
  // it keeps the padding an explicit range would lose.
  const requested = Number(plotData.ncol);
  const cols = Math.min(Number.isFinite(requested) && requested > 0 ? requested : 3, panels.length);
  const rows = Math.ceil(panels.length / cols);
  const kind = String(plotData.kind);
  const pad = 0.06;
  const height = Math.max(360, rows * 260 + 120);
  // Rows need a real gap, not the 3% margin a single row wants: between two
  // rows sit the upper panel's tick labels AND the lower panel's title, and
  // at 3% those collide. Sized in pixels, then expressed as a fraction.
  const outer = pad / 2;
  const gap = rows > 1 ? Math.min(0.18, 76 / height) : 0;
  const rowHeight = (1 - 2 * outer - gap * (rows - 1)) / rows;
  const out: Record<string, unknown> = {
    height,
    annotations: [] as Record<string, unknown>[],
    showlegend: kind === "boxplot",
  };
  const anns = out.annotations as Record<string, unknown>[];

  panels.forEach((p, i) => {
    const r = Math.floor(i / cols);
    const c = i % cols;
    const x0 = c / cols + pad / 2;
    const x1 = (c + 1) / cols - pad / 2;
    const yTop = 1 - outer - r * (rowHeight + gap);
    const yBot = yTop - rowHeight;
    const suffix = i === 0 ? "" : String(i + 1);
    out[`xaxis${suffix}`] = {
      domain: [x0, x1],
      anchor: `y${suffix}`,
      ...(kind === "scatter" && shared.x ? { range: shared.x } : {}),
    };
    out[`yaxis${suffix}`] = {
      domain: [yBot, yTop],
      anchor: `x${suffix}`,
      ...(shared.y ? { range: shared.y } : shared.x && kind === "boxplot" ? { range: shared.x } : {}),
    };
    anns.push({
      text: `${String(p.panel)} (n=${p.n})`,
      x: (x0 + x1) / 2,
      y: yTop,
      xref: "paper",
      yref: "paper",
      showarrow: false,
      yanchor: "bottom",
      font: { size: 11 },
    });
  });
  return out;
}

/** Significance brackets over a grouped plot, as Plotly shapes + annotations.
 *
 *  Positions come from the data, not the axis range, so the brackets clear the
 *  tallest whisker rather than floating at a guessed height. Each level is one
 *  step higher; the backend already ordered them shortest-span-first so short
 *  brackets sit under long ones instead of crossing them.
 */
function buildBrackets(
  plotData: Record<string, unknown> | null,
  comparisons: Record<string, unknown> | null,
): { shapes: Record<string, unknown>[]; annotations: Record<string, unknown>[] } {
  const empty = { shapes: [], annotations: [] };
  if (!plotData || !comparisons) return empty;
  const groups = plotData.groups as Array<{ values: number[] }> | undefined;
  const rows = comparisons.comparisons as Array<Record<string, number | string>> | undefined;
  if (!groups?.length || !rows?.length) return empty;

  const all = groups.flatMap((g) => g.values.map(Number)).filter(Number.isFinite);
  if (!all.length) return empty;
  const top = Math.max(...all);
  const bottom = Math.min(...all);
  const span = top - bottom || Math.abs(top) || 1;
  const step = span * 0.09;
  const base = top + step * 0.6;
  const tick = step * 0.22;

  const shapes: Record<string, unknown>[] = [];
  const annotations: Record<string, unknown>[] = [];
  rows.forEach((r) => {
    const yTop = base + step * Number(r.level);
    const x1 = Number(r.x1);
    const x2 = Number(r.x2);
    // Three segments: two short verticals and the horizontal joining them.
    shapes.push(
      { type: "line", xref: "x", yref: "y", x0: x1, x1: x1, y0: yTop - tick, y1: yTop, line: { color: "#4b5563", width: 1 } },
      { type: "line", xref: "x", yref: "y", x0: x2, x1: x2, y0: yTop - tick, y1: yTop, line: { color: "#4b5563", width: 1 } },
      { type: "line", xref: "x", yref: "y", x0: x1, x1: x2, y0: yTop, y1: yTop, line: { color: "#4b5563", width: 1 } },
    );
    annotations.push({
      x: (x1 + x2) / 2,
      y: yTop,
      xref: "x",
      yref: "y",
      text: String(r.label),
      showarrow: false,
      yanchor: "bottom",
      font: { size: 11, color: "#374151" },
    });
  });
  return { shapes, annotations };
}

function buildTraces(
  d: Record<string, unknown> | null,
  chartType: string,
  C: string[],
  td: { lineWidth: number; markerSize: number; markerOpacity: number },
  session: Session,
  showPoints = false,
  donutHole = 0,
  showMean = false,
  // Groups down the side rather than along the bottom. A published figure
  // does this whenever the category names are words — "PTC follicular
  // variant" does not fit under a tick and rotating it costs the reader.
  horizontal = false,
  opts: ChartOptions = {
    stat: "count", display: "both", rug: false, gradientScale: "Viridis", varyLineStyle: false,
  },
): PlotData[] | null {
  if (!d) return null;

  const valueLabelsFor = (colName: unknown): Record<string, string> => {
    const meta = session.columns.find((c) => c.name === colName);
    return (meta?.value_labels as Record<string, string> | undefined) ?? {};
  };

  if (d.type === "histogram") {
    return histogramTraces(d, C, td, opts, valueLabelsFor(d.color));
  }

  if (d.type === "scatter") {
    const points = d.points as Array<Record<string, unknown>>;
    const regression = d.regression as FitResult;
    // One fit per colour group replaces the overall line — ggplot2's
    // geom_smooth(aes(colour = group)).
    const groupFits = (d.regressions as Array<FitResult & { group: unknown }> | undefined) ?? [];
    const xKey = String(d.x);
    const yKey = String(d.y);
    const labelKey = d.label ? String(d.label) : null;
    // Labels ride on the markers; hover keeps them legible when they collide.
    const textFor = (rows: Array<Record<string, unknown>>) =>
      labelKey ? rows.map((p) => String(p[labelKey] ?? "")) : undefined;
    const markerMode = labelKey ? "markers+text" : "markers";
    // Confidence ellipses come back as closed polygons, one per group.
    const ellipseTraces: PlotData[] = ((d.ellipses as Array<Record<string, unknown>>) ?? [])
      .filter((e) => Array.isArray(e.x) && (e.x as unknown[]).length > 0)
      .map((e, i) => ({
        type: "scatter", mode: "lines",
        x: e.x, y: e.y,
        line: { color: C[i % C.length], width: 1.5, dash: "dot" },
        name: `${String(e.group)} ${Math.round(Number(d.ellipse_level ?? 0.95) * 100)}% ellipse`,
        hoverinfo: "name",
      } as PlotData));
    // Marker shape by a second column — distinguishable in print and for
    // readers who cannot separate the palette by colour alone.
    const SHAPES = ["circle", "diamond", "square", "triangle-up", "cross", "x", "star"];
    const shapeKey = d.shape ? String(d.shape) : null;
    const shapeLevels = shapeKey
      ? [...new Set(points.map((p) => String(p[shapeKey])))]
      : [];
    const symbolFor = (rows: Array<Record<string, unknown>>) =>
      shapeKey ? rows.map((p) => SHAPES[shapeLevels.indexOf(String(p[shapeKey])) % SHAPES.length]) : undefined;
    // Marginal histograms live on their own axes: one strip above the plot for
    // x, one to the right for y. Bars, not densities — the counts are what the
    // backend computed and a KDE would imply smoothing nobody asked for.
    // A continuous colour scale over a numeric column, with the colour bar
    // that makes it readable. Mutually exclusive with the group colouring, so
    // it only ever reaches the ungrouped branch below.
    const gradientKey = d.gradient ? String(d.gradient) : null;
    const gradientMarker = gradientKey
      ? {
        color: points.map((p) => Number(p[gradientKey])),
        colorscale: opts.gradientScale,
        showscale: true,
        colorbar: {
          title: { text: columnLabel(session, gradientKey), side: "right" },
          thickness: 12,
        },
        ...(Array.isArray(d.gradient_range) && (d.gradient_range as number[]).length === 2
          ? { cmin: (d.gradient_range as number[])[0], cmax: (d.gradient_range as number[])[1] }
          : {}),
      }
      : null;
    const marg = (d.marginal ?? {}) as {
      x?: Array<Record<string, number>>;
      y?: Array<Record<string, number>>;
    };
    const marginalTraces: PlotData[] = [];
    if (marg.x?.length) {
      marginalTraces.push({
        type: "bar",
        x: marg.x.map((b) => b.centre),
        y: marg.x.map((b) => b.count),
        marker: { color: C[0], opacity: 0.55 },
        xaxis: "x", yaxis: "y2",
        showlegend: false,
        hovertemplate: "%{y} points<extra></extra>",
      } as PlotData);
    }
    if (marg.y?.length) {
      marginalTraces.push({
        type: "bar",
        orientation: "h",
        y: marg.y.map((b) => b.centre),
        x: marg.y.map((b) => b.count),
        marker: { color: C[0], opacity: 0.55 },
        xaxis: "x2", yaxis: "y",
        showlegend: false,
        hovertemplate: "%{x} points<extra></extra>",
      } as PlotData);
    }
    const identity = (d.identity ?? {}) as { line_x?: unknown[]; line_y?: unknown[] };
    const identityTrace: PlotData[] =
      identity.line_x && (identity.line_x as unknown[]).length
        ? [{
            type: "scatter", mode: "lines",
            x: identity.line_x, y: identity.line_y,
            line: { color: "#6b7280", width: 1.5 },
            name: "y = x",
            hoverinfo: "name",
          } as PlotData]
        : [];
    // The grid replaces the cloud: the backend sends no points with it.
    const grid = (d.bin2d ?? {}) as { x?: number[]; y?: number[]; z?: number[][]; n?: number };
    if (grid.z?.length) {
      return [
        {
          type: "heatmap",
          x: grid.x, y: grid.y, z: grid.z,
          colorscale: opts.gradientScale,
          // Empty cells stay empty rather than taking the ramp's lowest
          // colour, which would read as "a few points here".
          zmin: 0,
          zauto: false,
          zmax: Math.max(1, ...(grid.z.flat())),
          hoverongaps: false,
          colorbar: { title: { text: "Points", side: "right" }, thickness: 12 },
          hovertemplate: "%{x}, %{y}<br>%{z} points<extra></extra>",
        } as PlotData,
        ...fitTraces(regression, "#f97316", Math.max(2, td.lineWidth), "", "solid").band,
        ...fitTraces(regression, "#f97316", Math.max(2, td.lineWidth), "", "solid").line,
        ...identityTrace,
        ...ellipseTraces,
        ...marginalTraces,
      ];
    }
    if (d.color) {
      const colorKey = String(d.color);
      const colorLabels = valueLabelsFor(d.color);
      const groups = [...new Set(points.map((p) => p[colorKey]))];
      const fits = groupFits.length
        ? groupFits.map((g) => {
          const idx = groups.findIndex((name) => String(name) === String(g.group));
          const colour = idx >= 0 ? C[idx % C.length] : "#374151";
          const label = idx >= 0 ? labelFor(colorLabels, groups[idx], String(g.group)) : String(g.group);
          return fitTraces(g, colour, td.lineWidth, label, "dash");
        })
        : [fitTraces(regression, "#374151", 1.5, "", "dash")];
      // Bands under the points, lines over them.
      return [
        ...fits.flatMap((f) => f.band),
        ...groups.map((g, i) => {
          const rows = points.filter((p) => p[colorKey] === g);
          return {
            type: "scatter",
            mode: markerMode,
            name: labelFor(colorLabels, g, String(g)),
            x: rows.map((p) => p[xKey]),
            y: rows.map((p) => p[yKey]),
            text: textFor(rows),
            textposition: "top center",
            textfont: { size: 9 },
            marker: { color: C[i % C.length], size: td.markerSize, opacity: td.markerOpacity, symbol: symbolFor(rows) },
          } as PlotData;
        }),
        ...fits.flatMap((f) => f.line),
        ...identityTrace,
        ...ellipseTraces,
        ...marginalTraces,
      ];
    }
    const fit = fitTraces(regression, C[1], td.lineWidth, "", "solid");
    return [
      ...fit.band,
      {
        type: "scatter", mode: markerMode,
        x: points.map((p) => p[xKey]),
        y: points.map((p) => p[yKey]),
        text: textFor(points),
        textposition: "top center",
        textfont: { size: 9 },
        marker: gradientMarker
          ? { ...gradientMarker, size: td.markerSize, opacity: td.markerOpacity, symbol: symbolFor(points) }
          : { color: C[0], size: td.markerSize, opacity: td.markerOpacity, symbol: symbolFor(points) },
        name: yKey,
      } as PlotData,
      ...fit.line,
      ...identityTrace,
      ...ellipseTraces,
      ...marginalTraces,
    ];
  }

  if (d.type === "lineplot") {
    const series = (d.series as Array<Record<string, unknown>>) ?? [];
    const out: PlotData[] = [];
    series.forEach((s, i) => {
      const pts = (s.points as Array<Record<string, number | string>>) ?? [];
      const xs = pts.map((p) => String(p.x));
      const hasBand = pts.some((p) => Number(p.upper) !== Number(p.lower));
      if (hasBand) {
        // One closed polygon per group: up the uppers, back down the lowers.
        out.push({
          type: "scatter", mode: "lines",
          x: [...xs, ...[...xs].reverse()],
          y: [...pts.map((p) => Number(p.upper)), ...[...pts].reverse().map((p) => Number(p.lower))],
          fill: "toself",
          fillcolor: C[i % C.length] + "22",
          line: { width: 0 },
          hoverinfo: "skip",
          showlegend: false,
        } as PlotData);
      }
      out.push({
        type: "scatter", mode: "lines+markers",
        x: xs,
        y: pts.map((p) => Number(p.centre)),
        line: {
          color: C[i % C.length], width: td.lineWidth,
          ...(dashFor(i, opts.varyLineStyle) ? { dash: dashFor(i, opts.varyLineStyle) } : {}),
        },
        marker: { color: C[i % C.length], size: td.markerSize },
        name: String(s.group),
        // n per point, because attrition is the thing a line hides.
        text: pts.map((p) => `n = ${p.n}`),
        hovertemplate: "%{x}: %{y:.3g}<br>%{text}<extra>%{fullData.name}</extra>",
      } as PlotData);
    });
    return out;
  }

  if (d.type === "slopeplot") {
    const pairs = (d.pairs as Array<Record<string, unknown>>) ?? [];
    const groups = [...new Set(pairs.map((p) => String(p.group ?? "All")))];
    const groupIndex = (p: Record<string, unknown>) => groups.indexOf(String(p.group ?? "All"));
    const colourOf = (p: Record<string, unknown>) => C[groupIndex(p) % C.length];
    // One two-point trace per subject; legend carries the group, not 200 lines.
    const lines: PlotData[] = pairs.map((p, i) => ({
      type: "scatter", mode: "lines+markers",
      x: [String(d.before), String(d.after)],
      y: [Number(p.before), Number(p.after)],
      line: {
        color: colourOf(p), width: 1,
        ...(dashFor(groupIndex(p), opts.varyLineStyle) ? { dash: dashFor(groupIndex(p), opts.varyLineStyle) } : {}),
      },
      marker: { color: colourOf(p), size: 5 },
      opacity: 0.5,
      name: String(p.group ?? "All"),
      legendgroup: String(p.group ?? "All"),
      showlegend: groups.indexOf(String(p.group ?? "All")) === i
        || pairs.findIndex((q) => String(q.group ?? "All") === String(p.group ?? "All")) === i,
      hovertemplate: `%{x}: %{y:.3g}${p.label ? `<br>${String(p.label)}` : ""}<extra></extra>`,
    } as PlotData));
    return lines;
  }

  if (d.type === "sankey") {
    const links = (d.links as Array<Record<string, number | string>>) ?? [];
    return [{
      type: "sankey",
      orientation: "h",
      node: {
        label: d.labels,
        pad: 14,
        thickness: 14,
        color: ((d.labels as string[]) ?? []).map((_, i) => C[i % C.length]),
        line: { color: "#e5e7eb", width: 1 },
      },
      link: {
        source: links.map((l) => Number(l.source)),
        target: links.map((l) => Number(l.target)),
        value: links.map((l) => Number(l.value)),
        color: links.map((l) => C[Number(l.source) % C.length] + "44"),
      },
    } as PlotData];
  }

  if (d.type === "stackplot") {
    const series = (d.series as Array<Record<string, unknown>>) ?? [];
    const usePct = Boolean(d.normalize);
    return series.map((s, i) => ({
      type: "bar",
      x: d.x_levels,
      y: usePct ? s.percent : s.value,
      name: String(s.fill),
      marker: { color: C[i % C.length] },
      hovertemplate: usePct
        ? "%{x} · %{fullData.name}: %{y:.1f}%<extra></extra>"
        : "%{x} · %{fullData.name}: %{y}<extra></extra>",
    } as PlotData));
  }

  if (d.type === "ridgeplot") {
    const ridges = (d.ridges as Array<Record<string, unknown>>) ?? [];
    const maxPeak = Number(d.max_peak) || 1;
    const step = 1.0;
    // Drawn top-down so the first group sits at the top, as ridgelines read.
    return [...ridges].reverse().map((r, i) => {
      const dens = (r.density as number[]).map((v) => (v / maxPeak) * step * 0.95 + i * step);
      const baseline = i * step;
      return {
        type: "scatter", mode: "lines",
        x: r.x,
        y: dens,
        fill: "tonexty",
        fillcolor: C[(ridges.length - 1 - i) % C.length] + "55",
        line: { color: C[(ridges.length - 1 - i) % C.length], width: 1.2 },
        name: `${String(r.group)} (n=${r.n})`,
        // A flat trace beneath gives `tonexty` something to fill against.
        customdata: [baseline],
        hovertemplate: `${String(r.group)}<extra></extra>`,
      } as PlotData;
    });
  }

  if (d.type === "sets") {
    const inter = (d.intersections as Array<Record<string, unknown>>) ?? [];
    // UpSet form: one bar per exclusive region, labelled by the sets it spans.
    return [{
      type: "bar",
      x: inter.map((r) => (r.sets as string[]).join(" ∩ ")),
      y: inter.map((r) => Number(r.count)),
      marker: { color: inter.map((r) => C[(Number(r.degree) - 1) % C.length]) },
      text: inter.map((r) => String(r.count)),
      textposition: "outside",
      hovertemplate: "%{x}<br>%{y} rows<extra></extra>",
    } as PlotData];
  }

  if (d.type === "pie") {
    const slices = (d.slices as Array<Record<string, unknown>>) ?? [];
    // The pie was the one chart still drawing raw codes for a labelled
    // column — 0.0, 1.0, 8.0 where the legend everywhere else said Benign.
    const sliceLabels = valueLabelsFor(d.category);
    return [{
      type: "pie",
      labels: slices.map((s) => labelFor(sliceLabels, s.label, String(s.label))),
      values: slices.map((s) => Number(s.value)),
      // Percentages on the slices: a pie is hard to read past three wedges,
      // and the number is what the reader is trying to recover anyway.
      textinfo: "label+percent",
      hole: donutHole,
      // Cycling the palette gives two slices of one pie the same colour, which
      // makes them read as one category. Past the end of the palette the hues
      // come back at a different lightness instead.
      marker: { colors: categoryColors(C, slices.length) },
      // Without this the leader lines of thin slices are drawn past the plot
      // area and clipped, so the smallest categories lose their labels.
      automargin: true,
      hovertemplate: `%{label}: %{value} (%{percent})<extra></extra>`,
    } as PlotData];
  }

  if (d.type === "balloon") {
    const cells = (d.cells as Array<Record<string, unknown>>) ?? [];
    const counts = cells.map((c) => Number(c.count));
    const maxCount = Math.max(...counts, 1);
    const resid = cells.map((c) => Number(c.residual));
    const maxAbsResid = Math.max(...resid.map(Math.abs), 1);
    return [{
      type: "scatter",
      mode: "markers",
      x: cells.map((c) => String(c.col)),
      y: cells.map((c) => String(c.row)),
      marker: {
        // Area, not diameter, tracks the count — sizing by diameter
        // exaggerates large cells by the square.
        size: counts.map((n) => 6 + 42 * Math.sqrt(n / maxCount)),
        color: resid,
        cmin: -maxAbsResid,
        cmax: maxAbsResid,
        colorscale: "RdBu",
        reversescale: true,
        showscale: true,
        colorbar: { title: { text: "Std. residual" }, thickness: 12 },
        line: { color: "#9ca3af", width: 1 },
      },
      text: cells.map(
        (c) => `n = ${c.count}<br>expected ${Number(c.expected).toFixed(1)}<br>residual ${Number(c.residual).toFixed(2)}`,
      ),
      hovertemplate: "%{y} × %{x}<br>%{text}<extra></extra>",
    } as PlotData];
  }

  if (d.type === "facet") {
    const panels = (d.panels as Array<Record<string, unknown>>) ?? [];
    const kind = String(d.kind);
    const traces: PlotData[] = [];
    panels.forEach((p, pi) => {
      const axis = pi === 0 ? "" : String(pi + 1);
      if (kind === "boxplot") {
        const groups = (p.groups as Array<Record<string, unknown>>) ?? [];
        groups.forEach((g, gi) => {
          traces.push({
            type: "box",
            y: g.values,
            name: String(g.group),
            legendgroup: String(g.group),
            showlegend: pi === 0,
            marker: { color: C[gi % C.length] },
            xaxis: `x${axis}`,
            yaxis: `y${axis}`,
          } as PlotData);
        });
      } else {
        traces.push({
          type: "scatter",
          mode: "markers",
          x: p.x,
          y: p.y,
          name: String(p.panel),
          showlegend: false,
          marker: { color: C[pi % C.length], size: td.markerSize, opacity: td.markerOpacity },
          xaxis: `x${axis}`,
          yaxis: `y${axis}`,
        } as PlotData);
      }
    });
    return traces;
  }

  if (d.type === "errorplot") {
    const rows = (d.rows as Array<Record<string, unknown>>) ?? [];
    return [{
      type: "scatter",
      mode: "markers",
      x: rows.map((r) => String(r.group)),
      y: rows.map((r) => Number(r.centre)),
      error_y: {
        type: "data",
        symmetric: false,
        array: rows.map((r) => Number(r.upper) - Number(r.centre)),
        arrayminus: rows.map((r) => Number(r.centre) - Number(r.lower)),
        thickness: 1.5,
        width: 6,
      },
      marker: { color: C[0], size: td.markerSize + 2 },
      name: String(d.spread_label ?? ""),
      hovertemplate: rows.map((r) => `${String(r.group)} (n=${r.n})<extra></extra>`),
    } as PlotData];
  }

  if (d.type === "ecdf") {
    const curves = (d.curves as Array<Record<string, unknown>>) ?? [];
    return curves.map((c, i) => ({
      type: "scatter",
      mode: "lines",
      // A step shape is the honest rendering: the ECDF jumps at each
      // observation and is flat between them.
      line: {
        shape: "hv", color: C[i % C.length], width: td.lineWidth,
        ...(dashFor(i, opts.varyLineStyle) ? { dash: dashFor(i, opts.varyLineStyle) } : {}),
      },
      x: c.x,
      y: c.y,
      name: `${String(c.group)} (n=${c.n})`,
    } as PlotData));
  }

  if (d.type === "dumbbell") {
    const rows = (d.rows as Array<Record<string, unknown>>) ?? [];
    // Plotly stacks the first category at the bottom; the backend ranks worst
    // first, so reverse to put the largest gap at the top of the figure.
    const ordered = [...rows].reverse();
    const cats = ordered.map((r) => String(r.category));
    const starts = ordered.map((r) => Number(r.start));
    const ends = ordered.map((r) => Number(r.end));
    const groupKey = d.group ? String(d.group) : null;
    const groupsSeen = groupKey ? [...new Set(ordered.map((r) => String(r.group)))] : [];
    const colorOf = (i: number) =>
      groupKey ? C[groupsSeen.indexOf(String(ordered[i].group)) % C.length] : C[0];

    // One connector per row: a separate two-point trace, so each can take the
    // colour of its own group rather than one colour for the whole set.
    const connectors: PlotData[] = ordered.map((r, i) => ({
      type: "scatter",
      mode: "lines",
      x: [Number(r.start), Number(r.end)],
      y: [String(r.category), String(r.category)],
      line: { color: colorOf(i), width: 2 },
      showlegend: false,
      hoverinfo: "skip",
    } as PlotData));

    return [
      ...connectors,
      {
        type: "scatter", mode: "markers",
        x: starts, y: cats,
        marker: {
          color: "#ffffff", size: td.markerSize + 2,
          line: { color: ordered.map((_, i) => colorOf(i)), width: 2 },
        },
        name: String(d.start),
        hovertemplate: `%{y}<br>${String(d.start)}: %{x}<extra></extra>`,
      } as PlotData,
      {
        type: "scatter", mode: "markers",
        x: ends, y: cats,
        marker: { color: ordered.map((_, i) => colorOf(i)), size: td.markerSize + 2 },
        name: String(d.end),
        hovertemplate: `%{y}<br>${String(d.end)}: %{x}<extra></extra>`,
      } as PlotData,
    ];
  }

  if (d.type === "boxplot") {
    const colorLabels = valueLabelsFor(d.color);
    const groups = d.groups as Array<{ values: unknown[]; group: unknown; row_indices?: number[] }>;

    // A raincloud is a violin taken apart: the density is drawn on one side
    // only, the box sits on the centre line, and every raw observation is
    // scattered on the other side. One trace does all three, which keeps the
    // category axis — and so the significance brackets — exactly as they are
    // for a box plot.
    if (chartType === "raincloud") {
      const n = groups.reduce((acc, g) => acc + g.values.length, 0);
      return groups.map((g, i) => ({
        type: "violin",
        ...(horizontal ? { x: g.values } : { y: g.values }),
        name: labelFor(colorLabels, g.group, String(g.group)),
        side: "positive",
        width: 1.0,
        points: "all",
        pointpos: -0.75,
        jitter: 0.5,
        // Without this the kernel runs past the smallest and largest
        // observation — on a strictly positive measure such as CMI the
        // density then reaches below zero, which is not a value the variable
        // can take.
        spanmode: "hard",
        // White-filled box on the centre line, as the published raincloud
        // draws it: filled in the group colour it disappears into the violin.
        box: { visible: true, width: 0.12, fillcolor: "#ffffff", line: { color: C[i % C.length], width: 1 } },
        meanline: { visible: false },
        scalemode: "width",
        line: { color: C[i % C.length], width: 1 },
        fillcolor: C[i % C.length] + "55",
        // The cloud is the point of the figure, so the markers are never
        // thinned out — only made smaller and fainter as they crowd.
        marker: {
          color: C[i % C.length],
          size: n > 2000 ? 2 : n > 500 ? 3 : 5,
          opacity: n > 2000 ? 0.35 : n > 500 ? 0.45 : 0.6,
        },
        hoveron: "points",
        text: g.row_indices?.map((idx) => `Row ${idx + 1}`),
      }));
    }

    if (chartType === "strip") {
      // Points and a median rule, nothing else. A box or a violin asserts a
      // shape the reader is meant to compare; with the small, uneven groups a
      // histology breakdown produces (seven Hurthle cases against two hundred
      // benign), the raw points are the honest picture and a kernel drawn
      // through seven of them is not.
      const median = (vals: unknown[]): number | null => {
        const nums = vals.map(Number).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
        if (!nums.length) return null;
        const mid = Math.floor(nums.length / 2);
        return nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
      };
      const names = groups.map((g) => labelFor(colorLabels, g.group, String(g.group)));
      // A transparent box is the jitter engine: Plotly scatter has no jitter,
      // and hand-rolling it needs numeric category positions that break the
      // moment a group is renamed.
      const points: PlotData[] = groups.map((g, i) => ({
        type: "box",
        ...(horizontal ? { x: g.values } : { y: g.values }),
        name: names[i],
        boxpoints: "all",
        pointpos: 0,
        jitter: 0.65,
        fillcolor: "rgba(0,0,0,0)",
        line: { color: "rgba(0,0,0,0)" },
        marker: { color: C[i % C.length], size: 5, opacity: 0.55 },
        hoveron: "points",
        text: g.row_indices?.map((idx) => `Row ${idx + 1}`),
        hovertemplate: horizontal
          ? "%{x}<br>%{text}<extra>%{fullData.name}</extra>"
          : "%{y}<br>%{text}<extra>%{fullData.name}</extra>",
        showlegend: false,
      } as PlotData));
      const medians = groups.map((g) => median(g.values as unknown[]));
      return [...points, {
        type: "scatter",
        mode: "markers",
        ...(horizontal ? { x: medians, y: names } : { x: names, y: medians }),
        marker: {
          symbol: horizontal ? "line-ns-open" : "line-ew-open",
          size: 26,
          // Both, deliberately: an open symbol takes its stroke from
          // marker.color, and leaving that unset lets the layout colourway
          // tint the median rule to match whichever series came next.
          color: "#111827",
          line: { color: "#111827", width: 2.5 },
        },
        name: "Median",
        hovertemplate: horizontal
          ? "median %{x}<extra>%{y}</extra>"
          : "median %{y}<extra>%{x}</extra>",
      } as PlotData];
    }

    if (chartType === "violin") {
      return groups.map((g, i) => ({
        type: "violin",
        ...(horizontal ? { x: g.values } : { y: g.values }),
        name: labelFor(colorLabels, g.group, String(g.group)),
        box: { visible: true },
        meanline: { visible: true },
        line: { color: C[i % C.length] },
        fillcolor: C[i % C.length] + "25",
        // The checkbox above offers the points, so it has to decide them here
        // too; the n cap only stops a violin turning into a smear.
        points: showPoints && g.values.length < 2000 ? "all" : false,
        jitter: 0.3,
        pointpos: -1.5,
        marker: { color: C[i % C.length], size: 3, opacity: 0.5 },
      }));
    }
    // One diamond per group at its mean, as a published box plot marks it.
    // Plotly's own boxmean draws a dashed line that disappears against the
    // median when the two are close — which is exactly when the reader most
    // needs to see that they are close.
    const meanTraces: PlotData[] = showMean
      ? [{
        type: "scatter",
        mode: "markers",
        ...(() => {
          const names = groups.map((g) => labelFor(colorLabels, g.group, String(g.group)));
          const means = groups.map((g) => {
            const nums = (g.values as unknown[]).map(Number).filter((v) => Number.isFinite(v));
            return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
          });
          return horizontal ? { x: means, y: names } : { x: names, y: means };
        })(),
        marker: { symbol: "diamond", size: 9, color: "#ffffff", line: { color: "#111827", width: 1.5 } },
        name: "Mean",
        hovertemplate: horizontal ? "mean %{x}<extra></extra>" : "mean %{y}<extra></extra>",
      } as PlotData]
      : [];

    return [...groups.map((g, i) => ({
      type: "box",
      ...(horizontal ? { x: g.values } : { y: g.values }),
      name: labelFor(colorLabels, g.group, String(g.group)),
      marker: { color: C[i % C.length], size: showPoints ? 4 : undefined, opacity: showPoints ? 0.55 : undefined },
      // "Show every point" is the modern default for small clinical samples —
      // a box alone hides n and any clustering. Above ~500 the overplotting
      // costs more than it tells, so the outliers-only view stays.
      boxpoints: showPoints
        ? (groups[0].values.length < 500 ? "all" : "outliers")
        : (groups[0].values.length < 500 ? "outliers" : false),
      jitter: showPoints ? 0.35 : undefined,
      pointpos: showPoints ? 0 : undefined,
      text: g.row_indices?.map((idx) => `Row ${idx + 1}`),
      hovertemplate: horizontal
        ? "%{x}<br>%{text}<extra>%{fullData.name}</extra>"
        : "%{y}<br>%{text}<extra>%{fullData.name}</extra>",
    })), ...meanTraces];
  }

  if (d.type === "bar" && Array.isArray(d.series)) {
    const xLabels = valueLabelsFor(d.x);
    const groupLabels = valueLabelsFor(d.color);
    const isPct = d.y_mode === "percentage";
    const series = d.series as Array<{ group: unknown; data: BarRow[] }>;
    return series.map((sr, i) => ({
      type: "bar",
      name: labelFor(groupLabels, sr.group, String(sr.group)),
      ...barGeometry(sr.data.map((r) => labelFor(xLabels, r.label, String(r.label))), sr.data, horizontal),
      marker: { color: C[i % C.length] },
      text: sr.data.map((r) => (isPct ? `${Number(r.value).toFixed(0)}%` : String(r.value))),
      ...barTextPlacement(sr.data),
      cliponaxis: false,
      hovertemplate: isPct
        ? `${barHoverCore(horizontal)}%` + (sr.data[0]?.n != null ? " (%{customdata[0]}/%{customdata[1]})" : "") + "<extra>%{fullData.name}</extra>"
        : `${barHoverCore(horizontal)}<extra>%{fullData.name}</extra>`,
      ...(isPct && sr.data[0]?.n != null
        ? { customdata: sr.data.map((r) => [r.k ?? 0, r.n ?? 0]) }
        : {}),
    } as PlotData));
  }

  if (d.type === "bar") {
    const xLabels = valueLabelsFor(d.x);
    const data = d.data as BarRow[];
    const isPct = d.y_mode === "percentage";
    return [{
      type: "bar",
      ...barGeometry(data.map((r) => labelFor(xLabels, r.label, String(r.label))), data, horizontal),
      marker: { color: C[0] },
      // The number over the bar. Reading a percentage off a gridline is a
      // guess, and a figure that states 36% is not making the reader guess.
      text: data.map((r) => (isPct ? `${Number(r.value).toFixed(0)}%` : String(r.value))),
      ...barTextPlacement(data),
      cliponaxis: false,
      // n and k travel with a percentage because 37% of 8 and 37% of 800 are
      // the same bar and not the same finding.
      hovertemplate: isPct
        ? `${barHoverCore(horizontal)}%` + (data[0]?.n != null ? " (%{customdata[0]}/%{customdata[1]})" : "") + "<extra></extra>"
        : `${barHoverCore(horizontal)}<extra></extra>`,
      ...(isPct && data[0]?.n != null
        ? { customdata: data.map((r) => [r.k ?? 0, r.n ?? 0]) }
        : {}),
      ...(d.error_label ? { name: String(d.error_label) } : {}),
    }];
  }

  if (d.type === "paired_box") {
    const colorLabels = valueLabelsFor(d.group);
    const groups = d.groups as Array<{ group: unknown; values: number[]; row_indices: number[]; pair_ids: (string | null)[] }>;
    const pairs = d.pairs as Array<{ pair_id: string; y0: number; y1: number }>;

    const boxTraces: PlotData[] = groups.map((g, i) => ({
      type: "box",
      x: g.values.map(() => i),
      y: g.values,
      name: labelFor(colorLabels, g.group, String(g.group)),
      marker: { color: C[i % C.length] },
      fillcolor: C[i % C.length] + "40",
      boxpoints: false,
      width: 0.5,
    }));

    const lineTraces: PlotData[] = pairs.map((pr) => ({
      type: "scatter",
      mode: "lines",
      x: [0 + pairJitter(pr.pair_id), 1 + pairJitter(pr.pair_id)],
      y: [pr.y0, pr.y1],
      line: { color: "#9ca3af", width: 1 },
      opacity: 0.55,
      showlegend: false,
      hoverinfo: "skip",
    }));

    const markerTraces: PlotData[] = groups.map((g, i) => ({
      type: "scatter",
      mode: "markers",
      x: g.values.map((_, idx) => i + pairJitter(g.pair_ids[idx] ?? `${String(g.group)}-${g.row_indices[idx]}`)),
      y: g.values,
      marker: { color: "#1f2937", size: td.markerSize, opacity: 0.85, line: { color: "#fff", width: 0.5 } },
      showlegend: false,
      hovertemplate: "%{y}<extra></extra>",
    }));

    return [...boxTraces, ...lineTraces, ...markerTraces];
  }

  return null;
}

/** One trend from the backend: the overall regression or a per-group fit. */
interface FitResult {
  method?: string;
  line_x?: unknown[];
  line_y?: unknown[];
  r2?: number | null;
  span?: number;
  note?: string;
  band?: { x?: number[]; lo?: number[]; hi?: number[]; level?: number };
}

/** Line plus band for one fit. Nothing when the fit has no line (method
 *  "none", or a group too small to fit). The band is the 95% CI of the fitted
 *  LINE, drawn as one ribbon: an invisible lower edge and an upper edge filled
 *  down to it. A bare line says nothing about how well the slope is pinned
 *  down, and it is pinned down least at the ends — where readers extrapolate. */
function fitTraces(
  fit: FitResult,
  colour: string,
  width: number,
  groupLabel: string,
  dash: "solid" | "dash",
): { band: PlotData[]; line: PlotData[] } {
  if (!fit.line_x?.length || !fit.line_y?.length) return { band: [], line: [] };
  const prefix = groupLabel ? `${groupLabel} · ` : "";
  const name = fit.method === "loess"
    ? `${prefix}LOESS (span ${(fit.span ?? 0.75).toFixed(2)})`
    : `${prefix}Fit (R²=${(fit.r2 ?? 0).toFixed(3)})`;
  const band = fit.band;
  const bandTraces: PlotData[] =
    band?.x?.length && band.lo?.length && band.hi?.length
      ? [
        { type: "scatter", mode: "lines", x: band.x, y: band.lo, line: { width: 0 }, hoverinfo: "skip", showlegend: false },
        {
          type: "scatter", mode: "lines", x: band.x, y: band.hi,
          line: { width: 0 }, fill: "tonexty",
          // The group's own colour at low alpha, so two bands stay tellable apart.
          fillcolor: /^#[0-9a-f]{6}$/i.test(colour) ? `${colour}2e` : "rgba(107,114,128,0.18)",
          name: `${prefix}${Math.round((band.level ?? 0.95) * 100)}% CI of the fit`,
          hoverinfo: "skip",
        },
      ]
      : [];
  return {
    band: bandTraces,
    line: [{
      type: "scatter", mode: "lines",
      x: fit.line_x, y: fit.line_y,
      line: { color: colour, width, ...(dash === "dash" ? { dash: "dash" } : {}) },
      name,
    }],
  };
}

/** Per-chart drawing choices that live only on the client — the backend
 *  neither needs them nor echoes them. */
/** How many groups a chart response carries, whatever shape it came in —
 *  the count a lightness ladder needs before a single trace is built. Zero
 *  when the response has no grouping, so the caller keeps the palette. */
function countGroupLevels(d: Record<string, unknown> | null): number {
  if (!d) return 0;
  const lengthOf = (key: string): number => {
    const v = d[key];
    return Array.isArray(v) ? v.length : 0;
  };
  // Box, violin, strip, raincloud, histogram: one entry per group.
  if (Array.isArray(d.groups)) return lengthOf("groups");
  // Grouped bar, line plot: one series per group.
  if (Array.isArray(d.series)) return lengthOf("series");
  if (Array.isArray(d.curves)) return lengthOf("curves");
  if (Array.isArray(d.ridges)) return lengthOf("ridges");
  // Scatter, slope plot: a group key on each row.
  const key = typeof d.color === "string" ? d.color : null;
  const rows = (Array.isArray(d.points) ? d.points : Array.isArray(d.pairs) ? d.pairs : null) as
    Array<Record<string, unknown>> | null;
  if (rows && rows.length) {
    const field = Array.isArray(d.pairs) ? "group" : key;
    if (!field) return 0;
    return new Set(rows.map((r) => String(r[field]))).size;
  }
  return 0;
}

/** A column's display label, falling back to its name. */
function columnLabel(session: Session, name: string): string {
  const meta = session.columns.find((c) => c.name === name);
  return meta?.label || name;
}

interface ChartOptions {
  /** Histogram y axis: count | density | percent — after_stat(). */
  stat: string;
  /** Histogram: both | bars | density — geom_histogram, bars alone, geom_density. */
  display: string;
  /** Histogram: geom_rug. */
  rug: boolean;
  /** Scatter: named Plotly colourscale for the continuous colour column. */
  gradientScale: string;
  /** Line charts: a dash pattern per group as well as a colour. */
  varyLineStyle: boolean;
}

/** Which axes of a chart carry numbers, and so can take a manual window.
 *  The other axis holds category names, where a numeric limit means nothing —
 *  and an input that cannot work is worse than an absent one. */
function numericAxesFor(
  chartType: string,
  horizontal: boolean,
  barHorizontal: boolean,
): { x: boolean; y: boolean } {
  switch (chartType) {
    case "histogram":
    case "scatter":
    case "ecdf":
      return { x: true, y: true };
    case "boxplot":
    case "violin":
    case "raincloud":
    case "strip":
      return horizontal ? { x: true, y: false } : { x: false, y: true };
    case "bar":
      return barHorizontal ? { x: true, y: false } : { x: false, y: true };
    case "paired":
    case "errorplot":
    case "lineplot":
    case "slopeplot":
    case "stackplot":
      return { x: false, y: true };
    case "dumbbell":
    case "ridgeplot":
      return { x: true, y: false };
    // A pie has no axes, a Sankey's are not values, and a facet has one pair
    // per panel — a single window would silently apply to the first only.
    default:
      return { x: false, y: false };
  }
}

/** A manual window on one axis — ggplot2's coord_cartesian(xlim = ), which
 *  zooms rather than dropping the rows outside it.
 *
 *  Plotly wants a log axis's range in log10 units, so a limit typed in data
 *  units is converted; a non-positive limit on a log axis has no logarithm and
 *  is ignored rather than rendering the axis blank. One end on its own becomes
 *  an autorange bound, so the other end still follows the data. */
function axisWindow(min: string, max: string, isLog: boolean): Record<string, unknown> {
  const convert = (v: number | null): number | null => {
    if (v === null) return null;
    if (!isLog) return v;
    return v > 0 ? Math.log10(v) : null;
  };
  const lo = convert(parseRefValue(min));
  const hi = convert(parseRefValue(max));
  if (lo !== null && hi !== null) {
    return lo === hi ? {} : { range: lo < hi ? [lo, hi] : [hi, lo], autorange: false };
  }
  if (lo !== null) return { autorangeoptions: { minallowed: lo } };
  if (hi !== null) return { autorangeoptions: { maxallowed: hi } };
  return {};
}

/** Charts that draw one line per group, where a dash pattern can carry the
 *  grouping alongside the colour. */
const LINE_STYLE_CHARTS = new Set(["lineplot", "ecdf", "slopeplot"]);

/** Dash patterns in the order groups take them. The first is solid, so a
 *  single-group figure is unchanged. */
const LINE_DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"];

/** The dash for group `i`, or undefined when the feature is off. */
function dashFor(i: number, on: boolean): string | undefined {
  return on ? LINE_DASHES[i % LINE_DASHES.length] : undefined;
}

/** Colour ramps offered for a continuous scale. Viridis and Cividis are
 *  perceptually uniform and survive greyscale printing; RdBu is diverging,
 *  for a value with a meaningful middle. */
const GRADIENT_SCALES = [
  { value: "Viridis", label: "Viridis" },
  { value: "Cividis", label: "Cividis (colour-blind safe)" },
  { value: "Plasma", label: "Plasma" },
  { value: "RdBu", label: "Red-Blue (diverging)" },
] as const;

interface HistGroup {
  group: string;
  n: number;
  counts: number[];
  kde: Array<{ x: number; y: number }>;
  values?: number[];
}

/** Bars, KDE curve and rug for one or more groups on shared bin edges.
 *
 *  Counts arrive from the backend; density and percent are rescalings of the
 *  same numbers (count / (n · width), count · 100 / n) so the y axis can be
 *  switched without another request. The KDE integrates to 1, so it is scaled
 *  the opposite way to sit on the bars. A response from before groups existed
 *  (bins + kde only) is read as a single group, so nothing old breaks. */
function histogramTraces(
  d: Record<string, unknown>,
  C: string[],
  td: { lineWidth: number; markerSize: number; markerOpacity: number },
  opts: ChartOptions,
  labels: Record<string, string>,
): PlotData[] {
  const bins = d.bins as Array<Record<string, number>>;
  const binWidth = typeof d.bin_width === "number" ? d.bin_width : bins[0].x1 - bins[0].x0;
  const centres = bins.map((b) => (b.x0 + b.x1) / 2);
  const groups: HistGroup[] = (d.groups as HistGroup[] | undefined)?.length
    ? (d.groups as HistGroup[])
    : [{
      group: "All",
      n: bins.reduce((a, b) => a + b.count, 0),
      counts: bins.map((b) => b.count),
      kde: (d.kde as Array<{ x: number; y: number }>) ?? [],
    }];
  const single = groups.length === 1 && !d.color;
  const barScale = (n: number) =>
    opts.stat === "density" ? 1 / (n * binWidth) : opts.stat === "percent" ? 100 / n : 1;
  const kdeScale = (n: number) =>
    opts.stat === "density" ? 1 : opts.stat === "percent" ? 100 * binWidth : n * binWidth;
  const statName = opts.stat === "density" ? "Density" : opts.stat === "percent" ? "Percent" : "Count";
  // Counts printed on the bars rather than only on hover: a histogram is
  // usually read for "how many are in this category", and a printed figure
  // has no hover at all. Above the bar rather than inside it, because a bar
  // one pixel tall has no inside — and `cliponaxis: false` so the label on
  // the tallest bar is not cut off by the top of the plot.
  //
  // Only for a single ungrouped count histogram of at most 30 bins: with
  // groups or finer bins the labels collide into an unreadable band, and a
  // density has no integer to print.
  const showCounts = single && opts.stat === "count" && bins.length <= 30;
  const hoverUnit = opts.stat === "percent" ? "%" : opts.stat === "density" ? "" : " observations";
  const out: PlotData[] = [];
  groups.forEach((g, i) => {
    const colour = C[i % C.length];
    const name = single ? statName : labelFor(labels, g.group, g.group);
    if (opts.display !== "density") {
      const scale = barScale(g.n);
      out.push({
        type: "bar",
        x: centres,
        y: g.counts.map((c) => c * scale),
        width: binWidth,
        marker: { color: colour, opacity: groups.length > 1 ? 0.55 : 0.8 },
        name,
        legendgroup: g.group,
        hovertemplate: `%{x}<br>%{y:.3~g}${hoverUnit}<extra>${name}</extra>`,
        ...(showCounts ? {
          // Empty bins print nothing: a row of zeroes along the axis is noise.
          text: g.counts.map((c) => (c ? String(c) : "")),
          textposition: "outside",
          cliponaxis: false,
          textfont: { size: Math.max(8, (td.markerSize ?? 6) + 3) },
          hovertemplate: "%{x}<br>%{y} observations<extra></extra>",
        } : {}),
      });
    }
    if (opts.display !== "bars" && g.kde.length) {
      const scale = kdeScale(g.n);
      const densityOnly = opts.display === "density";
      out.push({
        type: "scatter",
        x: g.kde.map((k) => k.x),
        y: g.kde.map((k) => k.y * scale),
        mode: "lines",
        line: { color: single && !densityOnly ? C[1] : colour, width: td.lineWidth },
        ...(densityOnly ? { fill: "tozeroy", fillcolor: `${colour}33` } : {}),
        name: single ? "KDE" : `${name} KDE`,
        legendgroup: g.group,
        showlegend: single || densityOnly,
        hoverinfo: "skip",
      });
    }
    if (opts.rug && g.values?.length) {
      out.push({
        type: "scatter",
        mode: "markers",
        x: g.values,
        y: g.values.map(() => 0),
        marker: { symbol: "line-ns-open", size: 9, color: colour, opacity: 0.6, line: { width: 1 } },
        name: `${name} rug`,
        legendgroup: g.group,
        showlegend: false,
        hovertemplate: "%{x}<extra></extra>",
        cliponaxis: false,
      });
    }
  });
  return out;
}

interface BarRow {
  label: unknown;
  value: unknown;
  n?: number;
  k?: number;
  lower?: number;
  upper?: number;
}

/** Category and value arrays for a bar trace, flipped when horizontal, plus
 *  the whisker whenever the backend sent one (geom_col + geom_errorbar). The
 *  whisker rides on the value axis, so it flips with the bars. */
function barGeometry(labels: string[], rows: BarRow[], horizontal: boolean): PlotData {
  const values = rows.map((r) => r.value);
  const hasWhisker = rows.some((r) => typeof r.lower === "number" && typeof r.upper === "number");
  const whisker = hasWhisker
    ? {
      type: "data",
      symmetric: false,
      array: rows.map((r) => Number(r.upper) - Number(r.value)),
      arrayminus: rows.map((r) => Number(r.value) - Number(r.lower)),
      thickness: 1.5,
      width: 6,
      color: "#374151",
    }
    : undefined;
  return horizontal
    ? { orientation: "h", x: values, y: labels, ...(whisker ? { error_x: whisker } : {}) }
    : { x: labels, y: values, ...(whisker ? { error_y: whisker } : {}) };
}

/** Where the bar's number goes. Outside the end as usual; but a whisker
 *  starts exactly there, so with one the number moves to the bar's middle
 *  rather than sit on top of the error bar. */
function barTextPlacement(rows: BarRow[]): PlotData {
  const hasWhisker = rows.some((r) => typeof r.lower === "number" && typeof r.upper === "number");
  return hasWhisker
    ? { textposition: "inside", insidetextanchor: "middle" }
    : { textposition: "outside" };
}

/** "%{category}<br>%{value}" with the axes the right way round. */
function barHoverCore(horizontal: boolean): string {
  return horizontal ? "%{y}<br>%{x}" : "%{x}<br>%{y}";
}

// Deterministic per-point horizontal jitter (seeded by pair/row id) so the
// same point renders at the same x offset in both its group's marker trace
// and its connector line — otherwise a line would miss its own markers.
function pairJitter(seed: string, amplitude = 0.16): number {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) | 0;
  const frac = ((h >>> 0) % 10000) / 10000;
  return (frac - 0.5) * 2 * amplitude;
}

// Paired box plots use two fixed numeric x positions (0/1) instead of a
// categorical axis, so tick labels must be supplied manually as the group
// values' display labels.
function pairedXAxisOverride(
  chartType: string,
  plotData: Record<string, unknown> | null,
  session: Session,
): Partial<PlotLayout> {
  if (chartType !== "paired" || !plotData || plotData.type !== "paired_box") return {};
  const groupCol = plotData.group as string;
  const meta = session.columns.find((c) => c.name === groupCol);
  const labels = (meta?.value_labels as Record<string, string> | undefined) ?? {};
  const groups = (plotData.groups as Array<{ group: unknown }>) ?? [];
  return {
    tickvals: groups.map((_, i) => i),
    ticktext: groups.map((g) => labelFor(labels, g.group, String(g.group))),
    range: [-0.7, groups.length - 0.3],
  };
}

