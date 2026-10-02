/** Gray's test of equal cumulative incidence, run by R's cmprsk::cuminc.
 *
 *  It exists only in the R engine (see backend/ustat_engine_r/analyses/
 *  gray.R for why the server has no copy). In an R session the card runs it
 *  in the browser; in a Python session it says so and offers to move the
 *  session to R, which keeps the server session and swaps the in-browser
 *  engine on the next local run. */
import { useState } from "react";
import { useStore } from "../../store";
import { runGrayTest, type GrayTestResult } from "../../api";
import { fmtP } from "../../lib/format";
import CopyTextButton from "../CopyTextButton";
import { Tip } from "../Tip";

interface Props {
  sessionId: string;
  durationCol: string;
  eventCol: string;
  groupCol: string;
  eventOfInterest: number;
}

function errorText(e: unknown): string {
  const detail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  return e instanceof Error ? e.message : "Gray's test failed.";
}

export default function GrayTestCard({ sessionId, durationCol, eventCol, groupCol, eventOfInterest }: Props) {
  const engine = useStore((s) => s.engine);
  const setEngine = useStore((s) => s.setEngine);
  const caseWeight = useStore((s) => s.caseWeight);
  const [result, setResult] = useState<GrayTestResult | null>(null);
  const [resultKey, setResultKey] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const key = [durationCol, eventCol, groupCol, eventOfInterest].join("|");
  const shown = resultKey === key ? result : null;
  const ready = Boolean(durationCol && eventCol && groupCol);

  const run = async () => {
    setBusy(true); setError(null);
    try {
      const res = await runGrayTest({
        session_id: sessionId, duration_col: durationCol, event_col: eventCol,
        group_col: groupCol, event_of_interest: eventOfInterest,
      });
      setResult(res.data); setResultKey(key);
    } catch (e: unknown) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const moveToRAndRun = async () => {
    setEngine("r");
    // Start the runtime download now; the run below waits for it either way.
    void import("../../lib/engine/r/client").then(({ prefetchRRuntime }) => prefetchRRuntime()).catch(() => {});
    await run();
  };

  return (
    <div className="border border-teal-200 bg-teal-50/30 rounded-lg p-3 space-y-2">
      <div className="flex items-center gap-1">
        <h4 className="text-sm font-semibold text-gray-800">Gray&apos;s test</h4>
        <Tip wide text="Compares the cumulative incidence of the event of interest across groups (Gray 1988), the test that matches the CIF curves. The cause-specific log-rank compares hazards instead and can disagree when groups differ in their competing-event rates. Computed by R's cmprsk::cuminc." />
      </div>

      {!ready ? (
        <p className="text-[11px] text-gray-500">Choose a group column to compare cumulative incidence.</p>
      ) : caseWeight ? (
        <p className="text-[11px] text-gray-600 leading-snug">
          Gray&apos;s test does not use Weight Cases: it runs in the R engine, which receives the rows
          unweighted. Turn weighting off to run it, or analyse the expanded data.
        </p>
      ) : engine === "r" ? (
        <button onClick={run} disabled={busy}
          className="text-xs px-2.5 py-1 rounded border border-teal-300 text-teal-700 hover:bg-teal-100 disabled:opacity-50">
          {busy ? "Running in R…" : "Run Gray's test (R, cmprsk)"}
        </button>
      ) : (
        <div className="space-y-1.5">
          <p className="text-[11px] text-gray-600 leading-snug">
            Analyse this in R: Gray&apos;s test is computed by R&apos;s <code>cmprsk</code>, which runs in the
            R engine. Switching keeps your data. Results already on screen are marked out of date and need a recompute;
            analyses the R engine does not cover still run on the server.
          </p>
          <button onClick={moveToRAndRun} disabled={busy}
            className="text-xs px-2.5 py-1 rounded border border-teal-300 text-teal-700 hover:bg-teal-100 disabled:opacity-50">
            {busy ? "Loading R…" : "Switch session to R and run"}
          </button>
        </div>
      )}

      {error && <p className="text-[11px] text-red-600" role="alert">{error}</p>}

      {shown && (
        <div className="space-y-2">
          <p className="text-xs text-gray-700">
            χ²({shown.df}) = <span className="font-mono">{shown.statistic.toFixed(3)}</span>,{" "}
            <i>p</i> = <span className={`font-mono ${shown.p < 0.05 ? "font-semibold text-teal-800" : ""}`}>{fmtP(shown.p)}</span>
            <span className="block text-[10px] text-gray-500">{shown.hypothesis}</span>
          </p>
          <table className="w-full text-[11px] bg-white rounded border border-gray-200">
            <thead>
              <tr className="text-gray-500 border-b border-gray-200">
                <th className="px-1.5 py-1 text-left font-medium">Group</th>
                <th className="px-1.5 py-1 text-right font-medium">n</th>
                <th className="px-1.5 py-1 text-right font-medium">Event</th>
                <th className="px-1.5 py-1 text-right font-medium">Competing</th>
                <th className="px-1.5 py-1 text-right font-medium">Censored</th>
              </tr>
            </thead>
            <tbody>
              {shown.groups.map((g) => (
                <tr key={g.group} className="border-b border-gray-100">
                  <td className="px-1.5 py-1 text-gray-700">{g.group}</td>
                  <td className="px-1.5 py-1 text-right tabular-nums">{g.n}</td>
                  <td className="px-1.5 py-1 text-right tabular-nums">{g.event_of_interest}</td>
                  <td className="px-1.5 py-1 text-right tabular-nums">{g.competing_events}</td>
                  <td className="px-1.5 py-1 text-right tabular-nums">{g.censored}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {shown.by_cause.length > 1 && (
            <p className="text-[10px] text-gray-500">
              Other event types:{" "}
              {shown.by_cause.filter((c) => c.event !== shown.event_of_interest)
                .map((c) => `event ${c.event}: χ²(${c.df}) = ${c.statistic.toFixed(2)}, p = ${fmtP(c.p)}`).join("; ")}
            </p>
          )}
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-gray-400">{shown.engine}</span>
            <CopyTextButton text={shown.r_code} />
          </div>
        </div>
      )}
    </div>
  );
}
