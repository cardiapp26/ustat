/** Small notes shared by the count-model result views (Poisson, negative
 *  binomial): which column acted as the exposure offset, the Poisson
 *  overdispersion warning, and the negative-binomial dispersion readout. */
import { Tip } from "../Tip";

const fmt = (v: number | null | undefined, digits = 3): string =>
  v == null || !Number.isFinite(v) ? "n/a" : v.toFixed(digits);

/** States that the fit is a rate model, so the IRRs are rate ratios. */
export function RateModelNote({ exposureCol }: { exposureCol?: string | null }) {
  if (!exposureCol) return null;
  return (
    <p className="mt-3 text-xs text-gray-600" role="note">
      <span className="font-semibold text-gray-800">Rate model: exposure = {exposureCol}</span>
      <span className="text-gray-400">
        {" "}· log({exposureCol}) enters as an offset, so the IRRs are incidence rate ratios (events per unit of follow-up time).
      </span>
    </p>
  );
}

/** Amber warning when the Poisson Pearson chi2/df says the SEs are too small. */
export function OverdispersionWarning({ overdispersed, note }: { overdispersed?: boolean; note?: string | null }) {
  if (!overdispersed) return null;
  return (
    <div className="mt-3 text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded px-2 py-1.5 leading-snug" role="alert">
      <strong>Overdispersion warning. </strong>
      {note ?? "The variance exceeds the mean, so the Poisson standard errors are likely too small. Consider negative binomial regression or robust standard errors."}
    </div>
  );
}

/** alpha / theta of a negative-binomial fit, with the server's explanation. */
export function NegBinDispersion({ alpha, alphaSe, theta, note }: {
  alpha?: number | null; alphaSe?: number | null; theta?: number | null; note?: string | null;
}) {
  if (alpha == null) return null;
  return (
    <div className="mt-3 rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-xs text-gray-600 leading-snug">
      <p>
        <span className="font-semibold text-gray-800 inline-flex items-center">
          Dispersion
          <Tip wide text="alpha is the negative-binomial dispersion estimated by maximum likelihood (variance = mu + alpha mu^2). theta = 1 / alpha is the same quantity in R's MASS::glm.nb parameterisation. alpha near 0 means the counts are close to Poisson." />
        </span>
        {": "}
        alpha = {fmt(alpha, 4)}
        {alphaSe != null && <> (SE {fmt(alphaSe, 4)})</>}
        {theta != null && <>, theta = {fmt(theta, 4)}</>}
      </p>
      {note && <p className="mt-1 text-gray-400">{note}</p>}
    </div>
  );
}
