/** Outcome-type metadata for the univariate RCS model. */

export type RCSModelType = "logistic" | "linear" | "poisson" | "quasipoisson" | "cox";

export interface RCSModelOption {
  key: RCSModelType;
  /** Short button label. */
  label: string;
  /** Accessible name and tooltip. */
  full: string;
}

export const RCS_MODEL_TYPES: readonly RCSModelOption[] = [
  { key: "logistic", label: "Logistic", full: "Logistic" },
  { key: "linear", label: "Linear", full: "Linear" },
  { key: "poisson", label: "Poisson", full: "Poisson" },
  { key: "quasipoisson", label: "Quasi-P", full: "Quasi-Poisson" },
  { key: "cox", label: "Cox", full: "Cox" },
];

export const isCountModel = (mt: string): boolean => mt === "poisson" || mt === "quasipoisson";

export interface EffectMeta {
  label: string;
  abbr: string;
  refValue: number;
  axisType: "log" | "linear";
}

export function effectMeta(mt: string): EffectMeta {
  if (mt === "cox") return { label: "Hazard Ratio", abbr: "HR", refValue: 1, axisType: "log" };
  if (mt === "linear") return { label: "Mean difference", abbr: "Δ", refValue: 0, axisType: "linear" };
  if (isCountModel(mt)) return { label: "Incidence Rate Ratio", abbr: "IRR", refValue: 1, axisType: "log" };
  return { label: "Odds Ratio", abbr: "OR", refValue: 1, axisType: "log" };
}

export function modelTitle(mt: string): string {
  const titles: Record<string, string> = {
    cox: "Cox-RCS", linear: "Linear RCS", poisson: "Poisson RCS", quasipoisson: "Quasi-Poisson RCS",
  };
  return titles[mt] ?? "Logistic RCS";
}

/**
 * Pearson dispersion above this under Poisson means the counts vary more than
 * the model allows, so its confidence bands and p-values are too optimistic.
 */
export const OVERDISPERSION_THRESHOLD = 1.5;

export interface DispersionNote {
  warn: boolean;
  text: string;
}

export function dispersionNote(mt: string, dispersion: number | null | undefined): DispersionNote | null {
  if (!isCountModel(mt) || dispersion == null || !Number.isFinite(dispersion)) return null;
  const value = `dispersion = ${dispersion.toFixed(2)}`;
  if (mt === "quasipoisson") {
    return { warn: false, text: `${value}: standard errors scaled by √${dispersion.toFixed(2)}` };
  }
  if (dispersion > OVERDISPERSION_THRESHOLD) {
    return { warn: true, text: `${value}: overdispersed, the Poisson CI is too narrow; refit as quasi-Poisson` };
  }
  return { warn: false, text: `${value}: consistent with Poisson` };
}

export interface NestedTest {
  test?: "chi2" | "F";
  lr_stat?: number;
  f?: number;
  df?: number;
  df_resid?: number;
}

/** "χ²(3) = 3.62" for a likelihood-ratio test, "F(3, 290) = 0.55" for quasi-Poisson. */
export function formatNestedTest(t: NestedTest): string {
  if (t.test === "F") return `F(${t.df}, ${t.df_resid}) = ${t.f?.toFixed(2) ?? "NA"}`;
  return `χ²(${t.df}) = ${t.lr_stat?.toFixed(2) ?? "NA"}`;
}

export function nestedTestName(t: NestedTest): string {
  return t.test === "F" ? "F test" : "LR test";
}
