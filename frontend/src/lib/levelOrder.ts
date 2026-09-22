/** Pure helpers for the Data Dictionary's category-order editor
 *  (components/LevelOrderEditor.tsx). */

export interface ArrangedLevels {
  /** Levels present in the data, in display order. */
  levels: string[];
  /** Present levels the saved order does not place (new in the data). */
  unplaced: string[];
}

/** The present levels in saved order, anything the order misses appended. */
export function arrangeLevels(values: string[], order: string[] | undefined): ArrangedLevels {
  if (!order || order.length === 0) return { levels: values, unplaced: [] };
  const present = new Set(values);
  const placed = order.filter((v) => present.has(v));
  const unplaced = values.filter((v) => !order.includes(v));
  return { levels: [...placed, ...unplaced], unplaced };
}

/** Write a re-ordered list of present levels back into the saved order.
 *  Levels the saved order names but the (filtered) data lacks keep their
 *  slots, so editing under a case filter does not drop or shift them. */
export function mergeLevelOrder(saved: string[] | undefined, present: string[]): string[] {
  if (!saved || saved.length === 0) return present;
  const presentSet = new Set(present);
  let next = 0;
  const kept = saved.map((v) => (presentSet.has(v) ? present[next++] : v));
  return [...kept, ...present.slice(next)];
}

/** Swap two positions; an out-of-range target leaves the list unchanged. */
export function swap(list: string[], i: number, j: number): string[] {
  if (j < 0 || j >= list.length) return list;
  const out = [...list];
  [out[i], out[j]] = [out[j], out[i]];
  return out;
}
