import { useMemo, type ReactNode } from "react";
import { StaleGuardContext, useStaleGuard } from "../lib/staleGuard";

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
 */
export default function StaleGuard({ stale, reason, children }: StaleGuardProps) {
  const outer = useStaleGuard();
  const value = useMemo(
    () => (outer.stale ? outer : { stale, reason: stale ? reason : undefined }),
    [outer, stale, reason],
  );
  return <StaleGuardContext.Provider value={value}>{children}</StaleGuardContext.Provider>;
}
