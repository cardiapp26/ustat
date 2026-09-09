import { useState } from "react";
import { AlertTriangle, ChevronDown, ChevronRight, X } from "lucide-react";
import { useStore } from "../store";
import type { IngestColumnReport } from "../api";

/**
 * What import changed, shown at the top of the session that it changed.
 *
 * The bar the panel has to clear is not "was a report produced" but "would the
 * person who is about to run a t-test find out". So it opens itself when a
 * column hit a measurement limit, carried a unit, or had a cell blanked -- the
 * three cases where a number on screen no longer matches the file -- and stays
 * a single quiet line when the only changes were comma-decimals and declared
 * missing codes, which lose nothing.
 */
function cellCount(n: number, noun = "value"): string {
  return `${n} ${noun}${n === 1 ? "" : "s"}`;
}

function ColumnRow({ c }: { c: IngestColumnReport }) {
  const examples = (rows: { row: number; value: string }[]) =>
    rows.map((e) => `“${e.value}” (row ${e.row + 1})`).join(", ");

  return (
    <li className="border-t border-amber-200/60 py-1.5 first:border-t-0">
      <div className="flex items-baseline gap-2">
        <span className="font-mono font-medium text-gray-800">{c.column}</span>
        <span
          className={`text-[9px] uppercase tracking-wider font-semibold rounded px-1 py-px ${
            c.decision === "numeric"
              ? "bg-indigo-100 text-indigo-700"
              : "bg-gray-200 text-gray-600"
          }`}
        >
          {c.decision === "numeric" ? "converted to numbers" : "kept as text"}
        </span>
      </div>
      <p className="text-gray-600 leading-snug">{c.reason}</p>

      {c.censored && (
        <p className="text-amber-800 leading-snug">
          {cellCount(c.censored.n)} at a measurement limit: {examples(c.censored.examples)}.
          A value below a limit of quantification is a result, not a blank — decide
          how to handle it (substitute at the limit, censor, exclude) before analysing
          the column.
        </p>
      )}

      {c.units && (
        <p className="text-amber-800 leading-snug">
          {cellCount(c.units.n)} carrying a unit ({Object.keys(c.units.suffixes).join(", ")}):{" "}
          {examples(c.units.examples)}. Stripping the unit would assume every cell shares
          it, and two units in one column is a silent order-of-magnitude error.
        </p>
      )}

      {c.n_discarded > 0 && (
        <p className="text-amber-800 leading-snug">
          {cellCount(c.n_discarded)} could not be read and {c.n_discarded === 1 ? "was" : "were"}{" "}
          blanked: {examples(c.discarded_examples)}. The original text is kept with the
          session and is readable at <code>/api/upload/&lt;id&gt;/ingest_report</code>.
        </p>
      )}

      {Object.keys(c.missing_codes).length > 0 && (
        <p className="text-gray-500 leading-snug">
          Read as missing:{" "}
          {Object.entries(c.missing_codes)
            .map(([code, n]) => `“${code}” ×${n}`)
            .join(", ")}
          .
        </p>
      )}

      {c.decimal_separator === "," && (
        <p className="text-gray-500 leading-snug">Decimal separator read as a comma.</p>
      )}
    </li>
  );
}

export default function IngestReportNotice() {
  const report = useStore((s) => s.ingestReport);
  const setReport = useStore((s) => s.setIngestReport);
  const [open, setOpen] = useState(true);

  // Nothing was rewritten and nothing needs a decision: no line at all.
  if (!report || report.columns.length === 0) return null;

  const review = report.needs_review;
  const headline = review
    ? "Import changed values that need a decision"
    : `Import tidied ${cellCount(report.n_cells_changed, "cell")} in ${cellCount(
        report.n_columns_converted, "column")}`;

  return (
    <div
      className={`flex-shrink-0 border-b px-4 py-1.5 text-[11px] ${
        review ? "border-amber-200 bg-amber-50" : "border-gray-200 bg-gray-50"
      }`}
    >
      <div className="flex items-center gap-1.5">
        {review && <AlertTriangle size={13} className="flex-shrink-0 text-amber-500" />}
        <button
          onClick={() => setOpen((v) => !v)}
          className={`flex items-center gap-1 font-medium ${
            review ? "text-amber-800" : "text-gray-600"
          }`}
          aria-expanded={open}
        >
          {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          {headline}
        </button>
        <span className="ml-auto flex items-center gap-2 text-gray-400">
          <span>
            {report.n_columns_examined} text column
            {report.n_columns_examined === 1 ? "" : "s"} examined
          </span>
          <button
            onClick={() => setReport(null)}
            title="Dismiss — the report stays readable from the session"
            className="rounded p-0.5 hover:bg-black/5"
          >
            <X size={12} />
          </button>
        </span>
      </div>

      {open && (
        <ul className="mt-1 space-y-0.5 pl-5">
          {report.columns.map((c) => (
            <ColumnRow key={c.column} c={c} />
          ))}
        </ul>
      )}
    </div>
  );
}
