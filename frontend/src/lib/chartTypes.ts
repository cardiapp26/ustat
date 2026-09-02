/** Chart types offered in the picker, in the order they are listed. Kept beside
 *  the panels rather than inside ChartsPanel so the icon set can be checked for
 *  coverage (see ChartTypeIcon.test.tsx) without importing a component module. */
export const CHART_TYPES = [
  "histogram", "scatter", "boxplot", "violin", "raincloud", "strip", "bar", "paired",
  "dumbbell", "errorplot", "ecdf", "pie", "balloon", "facet", "lineplot",
  "slopeplot", "sankey", "stackplot", "ridgeplot", "sets", "waffle", "waterfall",
] as const;
