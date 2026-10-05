/** Zero-inflated Poisson / negative binomial result: a count-part table (IRR),
 *  a zero-inflation table (OR of a structural zero), the zeros check, and the
 *  Vuong test against the standard model, stated in plain words. */
import type { ReactNode } from "react";
import ResultExporter from "../ResultExporter";
import { StyledTableExporter } from "../StyledTableExporter";
import CopyTextButton from "../CopyTextButton";
import { Tip, InfoBanner } from "../Tip";
import { RateModelNote } from "./CountModelNotes";
import type { Provenance } from "../../lib/engine/provenance";
import type { StyledTableData } from "../../lib/styledTable";
import { fmtP, fmtPubP, pCellTitle } from "../../lib/format";

type Num = number | null;

interface CountCoef {
  variable: string;
  log_irr: Num; irr: Num; se: Num; z: Num; p: Num;
  ci_low: Num; ci_high: Num; irr_ci_low: Num; irr_ci_high: Num;
}

interface InflationCoef {
  variable: string;
  logit: Num; or: Num; se: Num; z: Num; p: Num;
  ci_low: Num; ci_high: Num; or_ci_low: Num; or_ci_high: Num;
}

type VuongVerdict = "zero_inflated" | "standard" | "neither";

export interface VuongTest {
  statistic: { raw: Num; aic: Num; bic: Num };
  p: { raw: Num; aic: Num; bic: Num };
  favours?: { raw?: VuongVerdict; aic?: VuongVerdict; bic?: VuongVerdict };
  n?: number;
  standard_model?: string;
  preferred?: VuongVerdict;
  preferred_label?: string;
  note?: string;
}

export interface ZeroInflatedResultData {
  model?: string;
  outcome?: string;
  n?: number;
  n_excluded?: number;
  aic?: Num;
  bic?: Num;
  loglik?: Num;
  converged?: boolean;
  exposure_col?: string | null;
  rate_model?: boolean;
  inflation_predictors?: string[];
  n_zeros?: number;
  observed_zero_fraction?: Num;
  expected_zeros?: Num;
  expected_zero_fraction?: Num;
  expected_zeros_standard?: Num;
  standard_model_aic?: Num;
  standard_model_bic?: Num;
  alpha?: Num;
  alpha_se?: Num;
  theta?: Num;
  vuong?: VuongTest | null;
  warnings?: string[];
  count_coefficients?: CountCoef[];
  inflation_coefficients?: InflationCoef[];
  result_text?: string;
}

interface Props {
  result: ZeroInflatedResultData;
  stale?: boolean;
  staleReason?: string;
  provenance?: Provenance | null;
}

const fmt = (v: Num | undefined, digits = 3): string =>
  v == null || !Number.isFinite(v) ? "n/a" : v.toFixed(digits);

const fmtCi = (lo: Num, hi: Num, digits = 2): string =>
  lo == null || hi == null || !Number.isFinite(lo) || !Number.isFinite(hi)
    ? "n/a" : `${lo.toFixed(digits)}–${hi.toFixed(digits)}`;

const pct = (frac: Num | undefined): string =>
  frac == null || !Number.isFinite(frac) ? "" : ` (${(frac * 100).toFixed(1)}%)`;

const isConst = (name: string) => name === "const" || name === "Intercept";

const VERDICT_TEXT: Record<VuongVerdict, string> = {
  zero_inflated: "zero-inflated model",
  standard: "standard model",
  neither: "neither model",
};

/** Plain-words reading of the Vuong test (AIC-corrected statistic). */
function vuongVerdict(v: VuongTest, zeroInflatedLabel: string): { tone: "zi" | "std" | "neither"; text: string } {
  const std = stdName(v.standard_model ?? "standard");
  const stat = `Vuong V = ${fmt(v.statistic.aic, 2)}, p = ${fmtP(v.p.aic)}`;
  const preferred = v.preferred ?? v.favours?.aic ?? "neither";
  if (preferred === "zero_inflated") {
    return {
      tone: "zi",
      text: `The ${zeroInflatedLabel} fits significantly better than the standard ${std} model (${stat}). `
        + "Some zeros look structural (a separate process produces them), so keep the zero-inflation part.",
    };
  }
  if (preferred === "standard") {
    return {
      tone: "std",
      text: `The standard ${std} model fits significantly better than the ${zeroInflatedLabel} (${stat}). `
        + "Zero-inflation is not supported by these data; prefer the simpler model.",
    };
  }
  return {
    tone: "neither",
    text: `The Vuong test cannot tell the ${zeroInflatedLabel} from the standard ${std} model (${stat}). `
      + `Neither is clearly better, so by parsimony prefer the simpler standard ${std} model unless structural zeros are expected on subject-matter grounds.`,
  };
}

const TONE_CLS: Record<"zi" | "std" | "neither", string> = {
  zi: "bg-emerald-50 text-emerald-800 border-emerald-200",
  std: "bg-sky-50 text-sky-800 border-sky-200",
  neither: "bg-amber-50 text-amber-800 border-amber-200",
};

/** "Poisson" keeps its capital; "Negative binomial" reads as a common noun mid-sentence. */
const stdName = (m: string): string => (m === "Poisson" ? m : m.toLowerCase());

const th = "text-right py-1 font-medium";

export default function ZeroInflatedResult({ result, stale = false, staleReason, provenance }: Props) {
  const count = result.count_coefficients ?? [];
  const infl = result.inflation_coefficients ?? [];
  const label = result.model ?? "Zero-inflated model";
  const isNb = result.alpha != null;
  const inflationLabel = (result.inflation_predictors ?? []).length > 0
    ? `Zero-inflation predictors: ${(result.inflation_predictors ?? []).join(", ")}.`
    : "Zero-inflation part: intercept only (the same zero probability for every row).";

  const countHeaders = ["Variable", "Log-IRR", "SE", "z", "p-value", "IRR", "CI_low", "CI_high"];
  const countRows = count.map((c) => [
    c.variable, fmt(c.log_irr, 4), fmt(c.se, 4), fmt(c.z, 3), fmtP(c.p),
    fmt(c.irr, 3), fmt(c.irr_ci_low, 3), fmt(c.irr_ci_high, 3),
  ]);
  const inflHeaders = ["Variable", "Logit", "SE", "z", "p-value", "OR", "CI_low", "CI_high"];
  const inflRows = infl.map((c) => [
    c.variable, fmt(c.logit, 4), fmt(c.se, 4), fmt(c.z, 3), fmtP(c.p),
    fmt(c.or, 3), fmt(c.or_ci_low, 3), fmt(c.or_ci_high, 3),
  ]);

  const countStyled = (): StyledTableData => ({
    title: `${label}: count model`,
    columns: ["Variable", "IRR (95% CI)", "p"],
    rows: count.map((c) => [c.variable, `${fmt(c.irr, 2)} (${fmtCi(c.irr_ci_low, c.irr_ci_high)})`, fmtPubP(c.p)]),
    filename: "ZeroInflated_Count_Model",
  });
  const inflStyled = (): StyledTableData => ({
    title: `${label}: zero-inflation model`,
    columns: ["Variable", "OR (95% CI)", "p"],
    rows: infl.map((c) => [c.variable, `${fmt(c.or, 2)} (${fmtCi(c.or_ci_low, c.or_ci_high)})`, fmtPubP(c.p)]),
    filename: "ZeroInflated_Inflation_Model",
  });

  const cards: Array<[string, string, string]> = [
    ["N", String(result.n ?? "n/a"), "Observations used to fit the model."],
    ["AIC", fmt(result.aic, 2), "Lower is better when comparing models on the same data."],
    ["BIC", fmt(result.bic, 2), "Like AIC with a larger penalty for extra parameters."],
    ["Log-likelihood", fmt(result.loglik, 2), "Maximised log-likelihood of the zero-inflated model."],
  ];
  if (isNb) {
    cards.push(
      ["alpha (dispersion)", `${fmt(result.alpha, 4)}${result.alpha_se != null ? ` (SE ${fmt(result.alpha_se, 4)})` : ""}`,
        "Negative-binomial dispersion of the count part. Near 0 means the counts are close to Poisson."],
      ["theta (1/alpha)", fmt(result.theta, 4), "The same dispersion in R's MASS::glm.nb parameterisation."],
    );
  }

  const renderTable = (
    heading: string, tip: string, ratio: "IRR" | "OR",
    rows: Array<{ variable: string; est: Num; se: Num; z: Num; p: Num; ratio: Num; lo: Num; hi: Num }>,
    exporters: ReactNode,
  ) => (
    <div className="panel">
      <div className="flex items-center justify-between mb-2">
        <h4 className="font-semibold text-gray-900 flex items-center">
          {heading}
          <Tip wide text={tip} />
        </h4>
        <div className="flex items-center gap-1.5">{exporters}</div>
      </div>
      <div className="overflow-auto rounded border border-gray-200">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-500 border-b border-gray-200">
              <th className="text-left py-1 pl-2 font-medium">Variable</th>
              <th className={th}>{ratio === "IRR" ? "Log-IRR" : "Logit"}</th>
              <th className={th}>SE</th>
              <th className={th}>z</th>
              <th className={th}><i>p</i>-value</th>
              <th className={th}>{ratio}</th>
              <th className={`${th} pr-2`}>95% CI ({ratio})</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.variable} className={`border-b border-gray-100 ${r.p != null && r.p < 0.05 && !isConst(r.variable) ? "bg-indigo-50/40" : ""}`}>
                <td className="py-1 pl-2 font-mono text-gray-700">{r.variable}</td>
                <td className="py-1 text-right tabular-nums">{fmt(r.est, 4)}</td>
                <td className="py-1 text-right tabular-nums">{fmt(r.se, 4)}</td>
                <td className="py-1 text-right tabular-nums">{fmt(r.z, 3)}</td>
                <td className="py-1 text-right tabular-nums" title={pCellTitle(r.p)}>
                  <span className={r.p != null && r.p < 0.05 ? "badge-sig" : "badge-ns"}>{fmtP(r.p)}</span>
                </td>
                <td className="py-1 text-right tabular-nums font-semibold">{fmt(r.ratio, 3)}</td>
                <td className="py-1 pr-2 text-right tabular-nums text-gray-500">{fmtCi(r.lo, r.hi, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  const verdict = result.vuong ? vuongVerdict(result.vuong, label) : null;
  const std = result.vuong?.standard_model ?? (isNb ? "Negative binomial" : "Poisson");

  return (
    <div className="space-y-4">
      <div className="panel">
        <h4 className="font-semibold text-gray-900 mb-3">{label}</h4>
        <div className="grid grid-cols-3 gap-3">
          {cards.map(([k, v, tip]) => (
            <div key={k} className="bg-gray-50 border border-gray-200 rounded-lg p-3">
              <p className="text-xs text-gray-400 flex items-center">{k}<Tip text={tip} wide /></p>
              <p className="text-gray-900 font-semibold tabular-nums">{v}</p>
            </div>
          ))}
        </div>
        <RateModelNote exposureCol={result.exposure_col} />
        <p className="mt-3 text-xs text-gray-600">{inflationLabel}</p>
        {result.converged === false && (
          <div className="mt-2 text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded px-2 py-1.5" role="alert">
            The optimiser did not report convergence; interpret the estimates with caution.
          </div>
        )}
        {result.n_excluded != null && result.n_excluded > 0 && (
          <div className="mt-3">
            <InfoBanner>
              {result.n_excluded} row{result.n_excluded !== 1 ? "s" : ""} excluded for missing values;
              the model used <strong>{result.n}</strong>.
            </InfoBanner>
          </div>
        )}
        {(result.warnings ?? []).map((w) => (
          <div key={w} className="mt-2 text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded px-2 py-1.5" role="status">
            {w}
          </div>
        ))}
      </div>

      {renderTable(
        "Count model (incidence rate ratios)",
        "The count part: how each predictor changes the expected count among rows that are not structural zeros. IRR = exp(beta); IRR above 1 means a higher rate.",
        "IRR",
        count.map((c) => ({ variable: c.variable, est: c.log_irr, se: c.se, z: c.z, p: c.p, ratio: c.irr, lo: c.irr_ci_low, hi: c.irr_ci_high })),
        <>
          <StyledTableExporter data={countStyled} />
          <ResultExporter title="ZeroInflated_Count_Model" headers={countHeaders} rows={countRows}
            stale={stale} staleReason={staleReason} provenance={provenance} />
        </>,
      )}

      {renderTable(
        "Zero-inflation model (odds of a structural zero)",
        "The logit part: how each predictor changes the odds that a row is a structural zero (always zero, regardless of the count process). OR above 1 means more excess zeros.",
        "OR",
        infl.map((c) => ({ variable: c.variable, est: c.logit, se: c.se, z: c.z, p: c.p, ratio: c.or, lo: c.or_ci_low, hi: c.or_ci_high })),
        <>
          <StyledTableExporter data={inflStyled} />
          <ResultExporter title="ZeroInflated_Inflation_Model" headers={inflHeaders} rows={inflRows}
            stale={stale} staleReason={staleReason} provenance={provenance} />
        </>,
      )}

      {/* Zeros check: observed against what each model predicts. */}
      <div className="panel">
        <h4 className="font-semibold text-gray-900 mb-2 flex items-center">
          Zeros check
          <Tip wide text="Observed zeros against the number of zeros each fitted model predicts. A good model predicts about as many zeros as the data contain; a standard model that predicts far fewer is the usual sign of zero inflation." />
        </h4>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-500 border-b border-gray-200">
              <th className="text-left py-1 font-medium">Source</th>
              <th className={th}>Zeros</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-gray-100">
              <td className="py-1 text-gray-700">Observed</td>
              <td className="py-1 text-right tabular-nums">{result.n_zeros ?? "n/a"}{pct(result.observed_zero_fraction)}</td>
            </tr>
            <tr className="border-b border-gray-100">
              <td className="py-1 text-gray-700">Expected, {label}</td>
              <td className="py-1 text-right tabular-nums">{fmt(result.expected_zeros, 1)}{pct(result.expected_zero_fraction)}</td>
            </tr>
            {result.expected_zeros_standard != null && (
              <tr className="border-b border-gray-100">
                <td className="py-1 text-gray-700">Expected, standard {stdName(std)} model</td>
                <td className="py-1 text-right tabular-nums">{fmt(result.expected_zeros_standard, 1)}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Vuong test: zero-inflated against the standard model. */}
      <div className="panel">
        <h4 className="font-semibold text-gray-900 mb-2 flex items-center">
          Vuong test (zero-inflated vs standard {stdName(std)})
          <Tip wide text={result.vuong?.note ?? "Vuong (1989) non-nested test on per-observation log-likelihood differences. V above 1.96 favours the zero-inflated model, below -1.96 the standard model."} />
        </h4>
        {result.vuong && verdict ? (
          <>
            <div className={`rounded-lg border px-3 py-2 text-sm ${TONE_CLS[verdict.tone]}`} role="status">
              <strong>Preferred: {result.vuong.preferred_label ?? VERDICT_TEXT[result.vuong.preferred ?? "neither"]}.</strong>{" "}
              {verdict.text}
            </div>
            <table className="w-full text-xs mt-2">
              <thead>
                <tr className="text-gray-500 border-b border-gray-200">
                  <th className="text-left py-1 font-medium">Statistic</th>
                  <th className={th}>V</th>
                  <th className={th}><i>p</i></th>
                  <th className={th}>Favours</th>
                </tr>
              </thead>
              <tbody>
                {([["raw", "Uncorrected"], ["aic", "AIC-corrected (used)"], ["bic", "BIC-corrected"]] as const).map(([k, name]) => {
                  const fav = result.vuong?.favours?.[k];
                  return (
                    <tr key={k} className="border-b border-gray-100">
                      <td className="py-1 text-gray-700">{name}</td>
                      <td className="py-1 text-right tabular-nums">{fmt(result.vuong?.statistic[k], 2)}</td>
                      <td className="py-1 text-right tabular-nums" title={pCellTitle(result.vuong?.p[k])}>{fmtP(result.vuong?.p[k])}</td>
                      <td className="py-1 text-right text-gray-600">{fav ? VERDICT_TEXT[fav] : "n/a"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {result.standard_model_aic != null && (
              <p className="mt-2 text-[11px] text-gray-400">
                Standard {stdName(std)} model: AIC {fmt(result.standard_model_aic, 2)}, BIC {fmt(result.standard_model_bic, 2)}
                {" "}(zero-inflated: AIC {fmt(result.aic, 2)}, BIC {fmt(result.bic, 2)}).
              </p>
            )}
          </>
        ) : (
          <p className="text-xs text-gray-500">The Vuong test could not be computed for this fit.</p>
        )}
      </div>

      {result.result_text && (
        <div className="panel">
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-semibold text-gray-900">Results Paragraph</h4>
            <CopyTextButton text={result.result_text} />
          </div>
          <p className="text-sm text-gray-700 leading-relaxed bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">{result.result_text}</p>
        </div>
      )}
    </div>
  );
}
