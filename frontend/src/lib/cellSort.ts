/**
 * Cell comparison for the data grid's sort.
 *
 * Missing values sink to the end whichever way the column is sorted: the
 * direction applies to the values, not to the gaps. For a date column the
 * caller passes the column's date keys (see dateSortKeys) and values compare
 * chronologically; a value that does not read as a date sorts after every one
 * that does, and before the missing ones.
 */

export type SortDir = "asc" | "desc";

function isMissing(v: unknown): boolean {
  return v == null || v === "";
}

function compareValues(av: unknown, bv: unknown): number {
  return typeof av === "number" && typeof bv === "number"
    ? av - bv
    : String(av).localeCompare(String(bv), undefined, { numeric: true, sensitivity: "base" });
}

export function compareCells(
  av: unknown,
  bv: unknown,
  dir: SortDir,
  dateKeys?: Map<string, number>,
): number {
  const aMissing = isMissing(av);
  const bMissing = isMissing(bv);
  if (aMissing || bMissing) return aMissing === bMissing ? 0 : aMissing ? 1 : -1;

  const sign = dir === "asc" ? 1 : -1;
  if (!dateKeys) return sign * compareValues(av, bv);

  const ak = dateKeys.get(String(av));
  const bk = dateKeys.get(String(bv));
  if (ak !== undefined && bk !== undefined) return sign * (ak - bk);
  if (ak !== undefined) return -1;
  if (bk !== undefined) return 1;
  return sign * compareValues(av, bv);
}
