/** Multinomial logistic regression result: one RRR table per non-reference
 *  category, a likelihood-ratio test per predictor across all of them, model
 *  fit and the classification table. */
import ResultExporter from "../ResultExporter";
import CopyTextButton from "../CopyTextButton";
import { Tip, InfoBanner } from "../Tip";
import type { Provenance } from "../../lib/engine/provenance";
import { fmtP, pCellTitle } from "../../lib/format";

interface MultinomialCoef {
  variable: string;
  log_rrr: number | null;
  se: number | null;
  z: number | null;
  p: number | null;
  rrr: number | null;
  rrr_ci_low: number | null;
  rrr_ci_high: number | null;
}

interface MultinomialEquation {
  category: string;
  vs: string;
  n: number;
  coefficients: MultinomialCoef[];
}

interface MultinomialLR {
  variable: string;
  lr_chi2: number;
  df: number;
  p: number;
}

export interface MultinomialResultData {
  model?: string;
  outcome?: string;
  n?: number;
  n_excluded?: number;
  categories?: string[];
  reference?: string;
  level_order_source?: string;
  category_counts?: Record<string, number>;
  equations?: MultinomialEquation[];
  lr_tests?: MultinomialLR[];
  model_lr_chi2?: number | null;
  model_lr_df?: number | null;
  model_lr_p?: number | null;
  pseudo_r2?: number | null;
  aic?: number | null;
  bic?: number | null;
  classification_table?: { labels: string[]; table: number[][]; accuracy: number };
  warnings?: string[];
  result_text?: string;
}

interface Props {
  result: MultinomialResultData;
  valueLabels?: Record<string, string>;
  stale?: boolean;
  staleReason?: string;
  provenance?: Provenance | null;
}

const fmt = (v: number | null | undefined, digits = 3): string =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(digits);

const ciText = (lo: number | null, hi: number | null): string =>
  lo == null || hi == null ? "—" : `${fmt(lo, 2)} – ${fmt(hi, 2)}`;

export default function MultinomialResult({ result, valueLabels, stale = false, staleReason, provenance }: Props) {
  const label = (code: string): string => valueLabels?.[code] ?? code;
  const equations = result.equations ?? [];
  const lrTests = result.lr_tests ?? [];
  const reference = result.reference ?? "";
  const counts = result.category_counts ?? {};

  const exportHeaders = ["Category (vs reference)", "Variable", "B", "SE", "RRR", "95% CI", "p"];
  const exportRows = equations.flatMap((eq) =>
    eq.coefficients.map((c) => [
      `${label(eq.category)} vs ${label(eq.vs)}`, c.variable, c.log_rrr, c.se, c.rrr,
      ciText(c.rrr_ci_low, c.rrr_ci_high), c.p,
    ]),
  );

  const cards: Array<[string, string, string]> = [
    ["N", String(result.n ?? "—"), "Observations used to fit the model."],
    ["LR χ²", result.model_lr_chi2 != null ? `${fmt(result.model_lr_chi2, 2)} (df ${result.model_lr_df})` : "—",
      "Likelihood-ratio test of the whole model against the intercept-only model."],
    ["Model p", fmtP(result.model_lr_p), "p of the likelihood-ratio test above."],
    ["McFadden R²", fmt(result.pseudo_r2, 3), "1 − logL(model)/logL(intercept only). Not comparable to linear R²."],
    ["AIC", fmt(result.aic, 2), "Lower is better when comparing models on the same data."],
    ["BIC", fmt(result.bic, 2), "Like AIC with a larger penalty for extra parameters."],
  ];

  return (
    <div className="space-y-4">
      <div className="panel">
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-semibold text-gray-900">{result.model ?? "Multinomial Logistic Regression"}</h4>
          <ResultExporter title="multinomial_logistic" headers={exportHeaders} rows={exportRows}
            stale={stale} staleReason={staleReason} provenance={provenance} />
        </div>
        <div className="grid grid-cols-3 gap-3">
          {cards.map(([k, v, tip]) => (
            <div key={k} className="bg-gray-50 border border-gray-200 rounded-lg p-3">
              <p className="text-xs text-gray-400 flex items-center">{k}<Tip text={tip} wide /></p>
              <p className="text-gray-900 font-semibold tabular-nums">{v}</p>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs text-gray-600">
          Reference category: <strong>{label(reference)}</strong>. Categories:{" "}
          {(result.categories ?? []).map((c) => `${label(c)} (n=${counts[c] ?? 0})`).join(", ")}.
        </p>
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

      {lrTests.length > 0 && (
        <div className="panel">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center">
            Likelihood-ratio tests
            <Tip wide text="Each predictor dropped from every equation at once. Answers 'is this predictor related to the outcome at all?', which no single RRR row does. Matches anova() of nested nnet::multinom fits in R." />
          </h4>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-500 border-b border-gray-200">
                <th className="text-left py-1 font-medium">Predictor</th>
                <th className="text-right py-1 font-medium">LR χ²</th>
                <th className="text-right py-1 font-medium">df</th>
                <th className="text-right py-1 font-medium"><i>p</i></th>
              </tr>
            </thead>
            <tbody>
              {lrTests.map((t) => (
                <tr key={t.variable} className={`border-b border-gray-100 ${t.p < 0.05 ? "bg-indigo-50/40" : ""}`}>
                  <td className="py-1 font-mono text-gray-700">{t.variable}</td>
                  <td className="py-1 text-right tabular-nums">{fmt(t.lr_chi2, 2)}</td>
                  <td className="py-1 text-right tabular-nums">{t.df}</td>
                  <td className="py-1 text-right tabular-nums" title={pCellTitle(t.p)}>{fmtP(t.p)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {equations.map((eq) => (
        <div key={eq.category} className="panel">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center">
            {label(eq.category)} vs {label(eq.vs)}
            <span className="ml-2 text-xs font-normal text-gray-400">n = {eq.n}</span>
            <Tip wide text={`Relative risk ratio (RRR): how the odds of '${label(eq.category)}' rather than '${label(eq.vs)}' change per unit of the predictor, others held fixed. RRR > 1 favours '${label(eq.category)}'.`} />
          </h4>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-500 border-b border-gray-200">
                <th className="text-left py-1 font-medium">Variable</th>
                <th className="text-right py-1 font-medium">B</th>
                <th className="text-right py-1 font-medium">SE</th>
                <th className="text-right py-1 font-medium">RRR</th>
                <th className="text-right py-1 font-medium">95% CI</th>
                <th className="text-right py-1 font-medium"><i>p</i></th>
              </tr>
            </thead>
            <tbody>
              {eq.coefficients.map((c) => {
                const isIntercept = c.variable === "Intercept";
                const sig = !isIntercept && c.p != null && c.p < 0.05;
                return (
                  <tr key={c.variable} className={`border-b border-gray-100 ${sig ? "bg-indigo-50/40" : ""}`}>
                    <td className="py-1 font-mono text-gray-700">{c.variable}</td>
                    <td className="py-1 text-right tabular-nums">{fmt(c.log_rrr)}</td>
                    <td className="py-1 text-right tabular-nums text-gray-600">{fmt(c.se)}</td>
                    <td className="py-1 text-right tabular-nums">{isIntercept ? "—" : fmt(c.rrr, 2)}</td>
                    <td className="py-1 text-right tabular-nums text-gray-600">{isIntercept ? "—" : ciText(c.rrr_ci_low, c.rrr_ci_high)}</td>
                    <td className="py-1 text-right tabular-nums" title={pCellTitle(c.p)}>{fmtP(c.p)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ))}

      {result.classification_table && (
        <div className="panel">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center">
            Classification table
            <span className="ml-2 text-xs font-normal text-gray-400">
              {(result.classification_table.accuracy * 100).toFixed(1)}% correctly classified
            </span>
            <Tip wide text="Each case assigned to its most probable category. Accuracy should be read against the largest category's share, which a model with no predictors would already reach." />
          </h4>
          <table className="text-xs">
            <thead>
              <tr className="text-gray-500 border-b border-gray-200">
                <th className="text-left py-1 pr-4 font-medium">Observed \ Predicted</th>
                {result.classification_table.labels.map((l) => (
                  <th key={l} className="text-right py-1 px-3 font-medium">{label(l)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.classification_table.table.map((row, i) => (
                <tr key={result.classification_table!.labels[i]} className="border-b border-gray-100">
                  <td className="py-1 pr-4 text-gray-700">{label(result.classification_table!.labels[i])}</td>
                  {row.map((v, j) => (
                    <td key={j} className={`py-1 px-3 text-right tabular-nums ${i === j ? "font-semibold text-indigo-700" : "text-gray-600"}`}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

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
