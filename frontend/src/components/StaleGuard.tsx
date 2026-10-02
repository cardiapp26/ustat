import { useId, useMemo, type ReactNode } from "react";
import { StaleGuardContext, useStaleGuard } from "../lib/staleGuard";
import { OUTPUT_END_ATTR, OUTPUT_START_ATTR } from "../lib/outputCapture";

interface StaleGuardProps {
  stale: boolean;
  reason?: string;
  children: ReactNode;
}

/**
 * Marks everything inside as belonging to a result that may be out of date,
 * so every export control within refuses to run while it is (see
 * `lib/staleGuard`). Guards nest: an inner result inside a stale outer one is
 * stale too, because it was computed from the same moment.
 *
 * The outermost guard also brackets its children with two hidden markers, so
 * "Add to output" can copy exactly this result area. Markers rather than a
 * wrapper element: `hidden` elements take no space and are skipped by
 * Tailwind's space-y rules, where a wrapper would have changed the layout of
 * every panel using the guard.
 */
export default function StaleGuard({ stale, reason, children }: StaleGuardProps) {
  const outer = useStaleGuard();
  const ownId = useId();
  const isOutermost = outer.captureId === undefined;
  const captureId = outer.captureId ?? ownId;
  const value = useMemo(
    () => (outer.stale ? outer : { stale, reason: stale ? reason : undefined, captureId }),
    [outer, stale, reason, captureId],
  );
  return (
    <StaleGuardContext.Provider value={value}>
      {isOutermost && <span hidden {...{ [OUTPUT_START_ATTR]: captureId }} />}
      {children}
      {isOutermost && <span hidden {...{ [OUTPUT_END_ATTR]: captureId }} />}
    </StaleGuardContext.Provider>
  );
}
