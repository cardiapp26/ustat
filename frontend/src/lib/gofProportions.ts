/**
 * Expected-proportion helpers for the chi-square goodness-of-fit test: turning
 * the frequency table of a column into the categories the server will test, and
 * reading the weights a user types against them.
 */
import type { FrequencyTable } from "../api";

/** Cap on the categories the weight editor lists; a goodness-of-fit test on more is almost surely the wrong variable. */
export const MAX_GOF_CATEGORIES = 40;

export interface GofCategory {
  value: string;
  count: number;
}

/** The label the server will match: an integer-valued float column reads 3.0 as "3". */
export function gofLabel(value: string, floatColumn: boolean): string {
  return floatColumn && /^-?\d+\.0+$/.test(value) ? value.replace(/\.0+$/, "") : value.trim();
}

/** Same order the server uses for equal proportions: numeric labels by value, then text. */
export function sortGofCategories(cats: GofCategory[]): GofCategory[] {
  const numeric = (v: string) => v.trim() !== "" && Number.isFinite(Number(v));
  return [...cats].sort((a, b) => {
    const an = numeric(a.value);
    const bn = numeric(b.value);
    if (an !== bn) return an ? -1 : 1;
    if (an) return Number(a.value) - Number(b.value) || a.value.localeCompare(b.value);
    return a.value < b.value ? -1 : a.value > b.value ? 1 : 0;
  });
}

/** Frequency-table categories as the goodness-of-fit endpoint would see them (missing dropped). */
export function gofCategoriesFrom(table: FrequencyTable, floatColumn: boolean): GofCategory[] {
  let missingDropped = table.missing === 0;
  const kept: GofCategory[] = [];
  for (const c of table.categories) {
    if (!missingDropped && c.value === "Missing" && c.count === table.missing) {
      missingDropped = true;
      continue;
    }
    kept.push({ value: gofLabel(c.value, floatColumn), count: c.count });
  }
  return sortGofCategories(kept);
}

/** "0.25", "25%", "1/4" or "9" -> a positive weight; anything else -> null. */
export function parseGofWeight(raw: string): number | null {
  const t = raw.trim().replace(/%$/, "").trim();
  if (t === "") return null;
  const frac = /^(\d*\.?\d+)\s*\/\s*(\d*\.?\d+)$/.exec(t);
  const v = frac ? Number(frac[1]) / Number(frac[2]) : Number(t);
  return Number.isFinite(v) && v > 0 ? v : null;
}

/** "9:3:3:1", "9, 3, 3, 1" or "9 3 3 1" -> [9, 3, 3, 1]; null when any part is not a positive number. */
export function parseGofRatio(raw: string): number[] | null {
  const parts = raw.split(/[:,\s]+/).filter(Boolean);
  if (parts.length === 0) return null;
  const weights = parts.map(parseGofWeight);
  return weights.every((w): w is number => w !== null) ? weights : null;
}

/** Weights are kept per column so another column's "0"/"1" labels never inherit them. */
export const gofKey = (col: string, category: string) => `${col}::${category}`;

