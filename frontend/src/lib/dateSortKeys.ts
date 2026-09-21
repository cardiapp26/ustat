/**
 * Chronological sort keys for a column the server labelled "date".
 *
 * On import a text column whose values look like dates is labelled "date" but
 * keeps its text: converting is opt-in through "Parse as date", which writes
 * real datetimes. Sorting that text as text orders "14.09.2022" before
 * "23.09.2013" (by day, not by year), so a date column sorted in the grid came
 * out in an order that is no order at all.
 *
 * The rules mirror backend/services/date_parser.py (parse_one and
 * parse_series(order="auto")), so the grid sorts a value as the date "Parse as
 * date" would turn it into: ISO, day/month/year with . / - or space, Turkish
 * and English month names, 2-digit years against a threshold of 50, and Excel
 * serial numbers. Two additions matter only for ordering: a time of day after
 * the date breaks ties, and a bare time ("13:45") sorts by time of day.
 */

const TR_MONTHS: Record<string, number> = {
  ocak: 1, şubat: 2, subat: 2, mart: 3, nisan: 4, mayıs: 5, mayis: 5,
  haziran: 6, temmuz: 7, ağustos: 8, agustos: 8, eylül: 9, eylul: 9,
  ekim: 10, kasım: 11, kasim: 11, aralık: 12, aralik: 12,
  oca: 1, şub: 2, sub: 2, mar: 3, nis: 4, may: 5, haz: 6, tem: 7,
  ağu: 8, agu: 8, eyl: 9, eki: 10, kas: 11, ara: 12,
};
const EN_MONTHS: Record<string, number> = {
  january: 1, february: 2, march: 3, april: 4, may: 5, june: 6,
  july: 7, august: 8, september: 9, october: 10, november: 11, december: 12,
  jan: 1, feb: 2, mar: 3, apr: 4, jun: 6, jul: 7, aug: 8, sep: 9,
  sept: 9, oct: 10, nov: 11, dec: 12,
};

const PURE_NUMBER = /^-?\d+(\.\d+)?$/;
const ISO = /^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b/;
const DMY = /^(\d{1,2})[-/. ](\d{1,2})[-/. ](\d{2,4})\b/;
const DAY_MONTH_YEAR = /^(\d{1,2})\s+([a-zçşğıöü]+)\.?\s+(\d{2,4})\b/;
const MONTH_DAY_YEAR = /^([a-zçşğıöü]+)\.?\s+(\d{1,2})\s+(\d{2,4})\b/;
const TIME = /(\d{1,2}):(\d{2})(?::(\d{2}))?/;
const TIME_ONLY = /^(\d{1,2}):(\d{2})(?::(\d{2}))?$/;

const CENTURY_THRESHOLD = 50;
const EXCEL_EPOCH_MS = Date.UTC(1899, 11, 30); // the Excel 1900 leap-year bug
const DAY_MS = 86_400_000;

interface Ymd { y: number; mo: number; d: number }

/** One parsed value. `alt` is the month-first reading of a value valid both
 *  ways; `order` records which reading an unambiguous d/m/y value needed. */
interface Parsed extends Ymd {
  ambig: boolean;
  alt?: Ymd;
  order?: "dmy" | "mdy";
  timeMs: number;
}

function century(yy: number): number {
  return yy <= CENTURY_THRESHOLD ? 2000 + yy : 1900 + yy;
}

function valid(y: number, mo: number, d: number): boolean {
  if (mo < 1 || mo > 12 || d < 1 || d > 31 || y < 1900 || y > 2999) return false;
  const t = new Date(Date.UTC(y, mo - 1, d));
  return t.getUTCMonth() === mo - 1 && t.getUTCDate() === d;
}

function timeOfDay(match: RegExpMatchArray | null): number {
  if (!match) return 0;
  const [h, m, s] = [Number(match[1]), Number(match[2]), Number(match[3] ?? 0)];
  return ((h * 60 + m) * 60 + s) * 1000;
}

function monthNumber(name: string): number | undefined {
  return TR_MONTHS[name] ?? EN_MONTHS[name];
}

/** Port of date_parser.parse_one, plus the time of day after the date. */
function parseOne(raw: string): Parsed | null {
  const s = raw.trim().replace(/\s+/g, " ").replace(/,/g, " ").trim();
  if (!s) return null;

  if (PURE_NUMBER.test(s)) {
    const n = Number(s);
    if (n <= 59 || n >= 200_000) return null;
    const t = new Date(EXCEL_EPOCH_MS + Math.trunc(n) * DAY_MS);
    return { y: t.getUTCFullYear(), mo: t.getUTCMonth() + 1, d: t.getUTCDate(), ambig: false, timeMs: 0 };
  }

  const timeMs = timeOfDay(s.match(TIME));

  const iso = s.match(ISO);
  if (iso) {
    const [y, mo, d] = [Number(iso[1]), Number(iso[2]), Number(iso[3])];
    if (valid(y, mo, d)) return { y, mo, d, ambig: false, timeMs };
  }

  const dmy = s.match(DMY);
  if (dmy) {
    const [a, b] = [Number(dmy[1]), Number(dmy[2])];
    const rawYear = Number(dmy[3]);
    const y = rawYear < 100 ? century(rawYear) : rawYear;
    const dmyOk = valid(y, b, a);
    const mdyOk = valid(y, a, b);
    if (dmyOk && mdyOk) return { y, mo: b, d: a, ambig: true, alt: { y, mo: a, d: b }, timeMs };
    if (dmyOk) return { y, mo: b, d: a, ambig: false, order: "dmy", timeMs };
    if (mdyOk) return { y, mo: a, d: b, ambig: false, order: "mdy", timeMs };
    return null;
  }

  const low = s.toLowerCase();
  const dmyName = low.match(DAY_MONTH_YEAR);
  if (dmyName) {
    const mo = monthNumber(dmyName[2]);
    const rawYear = Number(dmyName[3]);
    const y = rawYear < 100 ? century(rawYear) : rawYear;
    const d = Number(dmyName[1]);
    if (mo && valid(y, mo, d)) return { y, mo, d, ambig: false, timeMs };
  }
  const mdyName = low.match(MONTH_DAY_YEAR);
  if (mdyName) {
    const mo = monthNumber(mdyName[1]);
    const rawYear = Number(mdyName[3]);
    const y = rawYear < 100 ? century(rawYear) : rawYear;
    const d = Number(mdyName[2]);
    if (mo && valid(y, mo, d)) return { y, mo, d, ambig: false, timeMs };
  }
  return null;
}

/**
 * Sort key (milliseconds) for every value in the column that reads as a date
 * or a time; values that do not are absent from the map.
 *
 * Day-first versus month-first is decided once for the column, as
 * parse_series does: values that only work one way are the evidence, ties go
 * to day-first. So "03/04/2024" sorts as 3 April beside "15/06/2024" and as
 * 4 March beside "03/15/2024".
 */
export function dateSortKeys(values: Iterable<unknown>): Map<string, number> {
  const parsed = new Map<string, Parsed>();
  const times = new Map<string, number>();
  let dmyEvidence = 0;
  let mdyEvidence = 0;
  const unparseable = new Set<string>();
  for (const value of values) {
    if (value == null) continue;
    const text = String(value);
    if (times.has(text) || unparseable.has(text)) continue;
    let p = parsed.get(text);
    if (!p) {
      const bareTime = text.trim().match(TIME_ONLY);
      if (bareTime) {
        times.set(text, timeOfDay(bareTime));
        continue;
      }
      const fresh = parseOne(text);
      if (!fresh) {
        unparseable.add(text);
        continue;
      }
      parsed.set(text, fresh);
      p = fresh;
    }
    // Every row counts, repeats included, as in parse_series.
    if (p.order === "dmy") dmyEvidence += 1;
    if (p.order === "mdy") mdyEvidence += 1;
  }

  const dayFirst = dmyEvidence >= mdyEvidence;
  const keys = new Map<string, number>(times);
  for (const [text, p] of parsed) {
    const { y, mo, d } = p.ambig && !dayFirst && p.alt ? p.alt : p;
    keys.set(text, Date.UTC(y, mo - 1, d) + p.timeMs);
  }
  return keys;
}
