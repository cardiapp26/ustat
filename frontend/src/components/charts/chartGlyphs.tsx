/**
 * The glyph shapes behind ChartTypeIcon.
 *
 * Each one draws the shape that distinguishes its chart, not a decorative
 * symbol: the point is that someone who has never heard of a dumbbell plot or
 * a balloon plot can tell from the picture what they would get. Names alone
 * do not carry that — "Ridge", "Balloon" and "Set overlap" mean nothing until
 * you have seen one.
 *
 * Inline SVG on a 24x16 canvas, stroked in `currentColor` so a glyph follows
 * the label's colour and the selected row highlights as one unit. No image
 * assets: at this size a bitmap would be blurry on a HiDPI screen and would
 * need a second copy for the dark theme.
 */

/** Filled dot — used by the scatter, dumbbell, error and raincloud glyphs. */
const dot = (cx: number, cy: number, r = 1.3, key?: string | number) => (
  <circle key={key} cx={cx} cy={cy} r={r} fill="currentColor" stroke="none" />
);

/** One box-and-whisker at `x`, spanning `top`..`bottom` with the box inset. */
const box = (x: number, top: number, boxTop: number, boxH: number, bottom: number, median: number) => (
  <>
    <line x1={x + 3} y1={top} x2={x + 3} y2={bottom} />
    <rect x={x} y={boxTop} width="6" height={boxH} rx="0.6" />
    <line x1={x} y1={median} x2={x + 6} y2={median} />
  </>
);

export const GLYPHS: Record<string, React.ReactNode> = {
  // Adjacent bars — the touching edges are what separate a histogram from a
  // bar chart, so the gap here is deliberately hairline.
  histogram: (
    <>
      <rect x="2" y="9" width="4.2" height="5" fill="currentColor" stroke="none" />
      <rect x="6.6" y="5" width="4.2" height="9" fill="currentColor" stroke="none" />
      <rect x="11.2" y="2.5" width="4.2" height="11.5" fill="currentColor" stroke="none" />
      <rect x="15.8" y="7" width="4.2" height="7" fill="currentColor" stroke="none" />
    </>
  ),
  scatter: (
    <>
      {[[4, 12], [8, 7], [12, 9.5], [15, 5], [19, 6.5], [21, 3]].map(([x, y], i) => dot(x, y, 1.3, i))}
    </>
  ),
  boxplot: (
    <>
      {box(3, 2, 5, 6, 14, 8.5)}
      {box(15, 3.5, 6, 5, 13, 9)}
    </>
  ),
  violin: (
    <>
      <path d="M6 2 C1.5 5.5 1.5 10.5 6 14 C10.5 10.5 10.5 5.5 6 2 Z" />
      <path d="M18 3 C14 6 14 11 18 14 C22 11 22 6 18 3 Z" />
    </>
  ),
  // Cloud over rain: the half-density above, the raw observations below.
  // Points scattered in three columns with a rule through each — the median
  // is the only summary the chart draws, so the glyph shows exactly that and
  // nothing box-shaped, or it reads as a box plot.
  strip: (
    <>
      {[[4, 5.5], [4, 8], [4, 11], [4, 13], [12, 4.5], [12, 7], [12, 9.5], [12, 12],
        [20, 6], [20, 8.5], [20, 11.5]].map(([x, y], i) => dot(x, y, 1, i))}
      <path d="M1.5 9.2 H6.5" strokeWidth="1.6" />
      <path d="M9.5 8.2 H14.5" strokeWidth="1.6" />
      <path d="M17.5 8.8 H22.5" strokeWidth="1.6" />
    </>
  ),
  raincloud: (
    <>
      <path d="M2.5 9 C5.5 2 18.5 2 21.5 9" />
      <path d="M2.5 9 H21.5" opacity="0.45" />
      {[[5, 13], [8, 12], [11, 13.5], [14, 12.3], [17, 13.2], [20, 12.5]].map(([x, y], i) => dot(x, y, 1, i))}
    </>
  ),
  // Same bars as the histogram but separated — categories, not bins.
  bar: (
    <>
      <rect x="2.5" y="7" width="4" height="7" fill="currentColor" stroke="none" />
      <rect x="10" y="3" width="4" height="11" fill="currentColor" stroke="none" />
      <rect x="17.5" y="9" width="4" height="5" fill="currentColor" stroke="none" />
    </>
  ),
  // Two boxes joined subject by subject — the connectors are the whole point.
  paired: (
    <>
      <rect x="2" y="5" width="5" height="7" rx="0.6" />
      <rect x="17" y="2.5" width="5" height="7" rx="0.6" />
      <path d="M7 7 L17 4 M7 9 L17 6.5 M7 11 L17 8.5" opacity="0.55" />
    </>
  ),
  dumbbell: (
    <>
      <line x1="5" y1="4.5" x2="19" y2="4.5" opacity="0.55" />
      {dot(5, 4.5, 1.7)}{dot(19, 4.5, 1.7)}
      <line x1="7" y1="11.5" x2="17" y2="11.5" opacity="0.55" />
      {dot(7, 11.5, 1.7)}{dot(17, 11.5, 1.7)}
    </>
  ),
  errorplot: (
    <>
      <line x1="5" y1="3" x2="5" y2="11" />
      <line x1="3.2" y1="3" x2="6.8" y2="3" /><line x1="3.2" y1="11" x2="6.8" y2="11" />
      {dot(5, 7, 1.7)}
      <line x1="12" y1="5.5" x2="12" y2="13.5" />
      <line x1="10.2" y1="5.5" x2="13.8" y2="5.5" /><line x1="10.2" y1="13.5" x2="13.8" y2="13.5" />
      {dot(12, 9.5, 1.7)}
      <line x1="19" y1="2" x2="19" y2="10" />
      <line x1="17.2" y1="2" x2="20.8" y2="2" /><line x1="17.2" y1="10" x2="20.8" y2="10" />
      {dot(19, 6, 1.7)}
    </>
  ),
  ecdf: <path d="M2 14 H6 V10.5 H9.5 V8 H13 V5.5 H16.5 V3 H20 V2 H22" />,
  pie: (
    <>
      <circle cx="12" cy="8" r="6" />
      <path d="M12 8 L12 2 A6 6 0 0 1 17.2 11 Z" fill="currentColor" stroke="none" />
    </>
  ),
  // A grid where the value is the area of each disc.
  balloon: (
    <>
      {dot(5, 4.5, 2.6)}{dot(12, 4.5, 1.2)}{dot(19, 4.5, 2)}
      {dot(5, 11.5, 1)}{dot(12, 11.5, 2.8)}{dot(19, 11.5, 1.7)}
    </>
  ),
  facet: (
    <>
      <rect x="2" y="2" width="9" height="5.5" rx="0.6" />
      <rect x="13" y="2" width="9" height="5.5" rx="0.6" />
      <rect x="2" y="8.5" width="9" height="5.5" rx="0.6" />
      <rect x="13" y="8.5" width="9" height="5.5" rx="0.6" />
    </>
  ),
  lineplot: (
    <>
      <path d="M2.5 12 L7.5 6.5 L12 9 L16.5 3.5 L21.5 7" />
      {[[2.5, 12], [7.5, 6.5], [12, 9], [16.5, 3.5], [21.5, 7]].map(([x, y], i) => dot(x, y, 1.2, i))}
    </>
  ),
  // Two moments, one line per subject — including the crossing that a pair of
  // group means would hide.
  slopeplot: (
    <>
      <line x1="4" y1="1.5" x2="4" y2="14.5" opacity="0.3" />
      <line x1="20" y1="1.5" x2="20" y2="14.5" opacity="0.3" />
      {/* Mostly parallel with one crosser. Three lines through a common
          centre draw a perfect X, which reads as a bowtie, not a slope. */}
      <path d="M4 3.5 L20 8 M4 8 L20 12.5 M4 12.5 L20 5" />
    </>
  ),
  sankey: (
    <>
      <rect x="2" y="2" width="2.4" height="12" fill="currentColor" stroke="none" />
      <rect x="19.6" y="1.5" width="2.4" height="4.5" fill="currentColor" stroke="none" />
      <rect x="19.6" y="10" width="2.4" height="4.5" fill="currentColor" stroke="none" />
      {/* Two separated flows. Fatter strokes merge into one blob at 16px and
          the split — the thing a Sankey is for — disappears. */}
      <path d="M4.4 5 C12 5 12 3.75 19.6 3.75" strokeWidth="2.2" opacity="0.45" />
      <path d="M4.4 11 C12 11 12 12.25 19.6 12.25" strokeWidth="2.8" opacity="0.45" />
    </>
  ),
  stackplot: (
    <>
      <rect x="2.5" y="9" width="4" height="5" fill="currentColor" stroke="none" />
      <rect x="2.5" y="5" width="4" height="3.6" fill="currentColor" stroke="none" opacity="0.45" />
      <rect x="10" y="7" width="4" height="7" fill="currentColor" stroke="none" />
      <rect x="10" y="2.5" width="4" height="4.1" fill="currentColor" stroke="none" opacity="0.45" />
      <rect x="17.5" y="10" width="4" height="4" fill="currentColor" stroke="none" />
      <rect x="17.5" y="5.5" width="4" height="4.1" fill="currentColor" stroke="none" opacity="0.45" />
    </>
  ),
  // Stacked densities. Each keeps a flat baseline and a single dominant peak:
  // a symmetric repeated bump reads as a wave, which is a different chart.
  ridgeplot: (
    <>
      <path d="M1.5 5 H5 Q8 -0.5 11 5 H22.5" />
      <path d="M1.5 9.5 H8 Q11 4 14 9.5 H22.5" opacity="0.75" />
      <path d="M1.5 14 H4 Q7.5 8.5 11 14 H22.5" opacity="0.5" />
    </>
  ),
  sets: (
    <>
      <circle cx="9" cy="8" r="5" />
      <circle cx="15" cy="8" r="5" />
    </>
  ),
};

/** Chart types that have a glyph — the source of truth for the coverage test. */
export const ICON_TYPES = Object.keys(GLYPHS);
