import { AlertTriangle } from "lucide-react";
import { useStore } from "../store";
import { describeProvenance, provenanceLines, type Provenance } from "../lib/engine/provenance";

interface Props {
  provenance: Provenance | null | undefined;
}

/**
 * What computed the number directly above it.
 *
 * WHY EVERY RESULT AND NOT ONE HEADER CHIP. The chip says which engine the
 * session chose. Only two analyses are actually routed to a local engine, so
 * in an R session every regression, Cox model and forest is the server's
 * Python -- and a chip reading "R-based statistics" over a Python number is
 * not a rounding error in the interface, it is a claim that will not reproduce.
 * So the claim is made where it can be checked: beside the result, from what
 * the run itself reported, versions included.
 *
 * Amber only on a mismatch. A Python session computing in Python needs one
 * grey line for the methods section; an R session looking at a Python number
 * needs to be told.
 */
export default function ResultProvenanceLine({ provenance }: Props) {
  const sessionEngine = useStore((s) => s.engine);
  if (!provenance) return null;

  const mismatch = provenance.engine !== sessionEngine;
  const full = provenanceLines(provenance).join("\n");

  return (
    <p
      title={full}
      className={`flex items-start gap-1.5 text-[10px] leading-snug ${
        mismatch ? "text-amber-700" : "text-gray-500"
      }`}
    >
      {mismatch && <AlertTriangle size={11} className="mt-px flex-shrink-0 text-amber-500" />}
      <span>
        {mismatch && (
          <b>
            Computed with {provenance.engine === "r" ? "R" : "Python"}, not{" "}
            {sessionEngine === "r" ? "R" : "Python"}.{" "}
          </b>
        )}
        {describeProvenance(provenance)}
        {provenance.fellBackBecause ? ` · ${provenance.fellBackBecause}` : ""}
      </span>
    </p>
  );
}
