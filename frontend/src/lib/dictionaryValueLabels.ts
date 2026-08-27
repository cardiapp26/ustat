import type { ColMeta } from "../store";

export type DictionaryCell = string | number | boolean | null | undefined;

export interface ParsedDictionaryColumn {
  sourceName: string;
  labels: Record<string, string>;
  warnings: string[];
}

export interface ParsedDictionary {
  columns: ParsedDictionaryColumn[];
  warnings: string[];
}

export interface DictionaryColumnMatch {
  sourceName: string;
  targetName: string;
  score: number;
  reason: "name" | "label" | "fuzzy" | "unmatched";
}

const NUMERIC_CODE = /^-?\d+(?:[.,]\d+)?$/;

function canonicalCode(value: string): string {
  const trimmed = value.trim();
  if (!NUMERIC_CODE.test(trimmed)) return trimmed;
  const numeric = Number(trimmed.replace(",", "."));
  return Number.isFinite(numeric) ? String(numeric) : trimmed;
}

function parseDelimitedCell(text: string): { code: string; label: string } | null {
  const delimiters = ["->", "→", "=", ":"];
  for (const delimiter of delimiters) {
    const positions: number[] = [];
    let start = 0;
    while (start < text.length) {
      const index = text.indexOf(delimiter, start);
      if (index < 0) break;
      positions.push(index);
      start = index + delimiter.length;
    }

    for (const index of positions) {
      const left = text.slice(0, index).trim();
      const right = text.slice(index + delimiter.length).trim();
      if (!left || !right) continue;
      const leftIsCode = NUMERIC_CODE.test(left);
      const rightIsCode = NUMERIC_CODE.test(right);

      if (leftIsCode && !rightIsCode) {
        return { code: canonicalCode(left), label: right };
      }
      if (rightIsCode && !leftIsCode) {
        return { code: canonicalCode(right), label: left };
      }
      if (delimiter === ":") {
        return { code: canonicalCode(right), label: left };
      }
      return { code: canonicalCode(left), label: right };
    }
  }
  return null;
}

function parseCell(value: DictionaryCell): {
  code: string;
  label: string;
  recovered: boolean;
} | null {
  if (value === null || value === undefined) return null;
  const text = String(value).trim();
  if (!text) return null;

  const delimited = parseDelimitedCell(text);
  if (delimited) return { ...delimited, recovered: false };

  // Conservative recovery for a common spreadsheet typo: "PEG dışı.1"
  // among otherwise label:integer cells. Decimal-looking labels remain
  // visible as warnings in the import preview before users apply anything.
  const trailingInteger = /^(.+\D)\.(-?\d+)$/.exec(text);
  if (trailingInteger) {
    return {
      code: canonicalCode(trailingInteger[2]),
      label: trailingInteger[1].trim(),
      recovered: true,
    };
  }
  return null;
}

export function parseWideDictionaryRows(rows: DictionaryCell[][]): ParsedDictionary {
  if (rows.length === 0) return { columns: [], warnings: ["File is empty."] };

  const headers = rows[0] ?? [];
  const seenHeaders = new Set<string>();
  const columns: ParsedDictionaryColumn[] = [];
  const warnings: string[] = [];

  headers.forEach((rawHeader, columnIndex) => {
    const sourceName = String(rawHeader ?? "").trim();
    if (!sourceName) return;
    if (seenHeaders.has(sourceName)) {
      warnings.push(`Duplicate variable column ignored: ${sourceName}`);
      return;
    }
    seenHeaders.add(sourceName);

    const labels: Record<string, string> = Object.create(null) as Record<string, string>;
    const columnWarnings: string[] = [];
    for (let rowIndex = 1; rowIndex < rows.length; rowIndex += 1) {
      const rawValue = rows[rowIndex]?.[columnIndex];
      if (rawValue === null || rawValue === undefined || String(rawValue).trim() === "") continue;
      const parsed = parseCell(rawValue);
      if (!parsed) {
        columnWarnings.push(`Row ${rowIndex + 1}: could not parse “${String(rawValue)}”`);
        continue;
      }
      if (Object.prototype.hasOwnProperty.call(labels, parsed.code)) {
        columnWarnings.push(`Row ${rowIndex + 1}: duplicate code ${parsed.code} ignored`);
        continue;
      }
      labels[parsed.code] = parsed.label;
      if (parsed.recovered) {
        columnWarnings.push(
          `Row ${rowIndex + 1}: interpreted “${String(rawValue)}” as ${parsed.label}:${parsed.code}`,
        );
      }
    }

    if (Object.keys(labels).length > 0) {
      columns.push({ sourceName, labels, warnings: columnWarnings });
    } else {
      warnings.push(`No usable value labels found under ${sourceName}.`);
      warnings.push(...columnWarnings.map((warning) => `${sourceName}, ${warning}`));
    }
  });

  return { columns, warnings };
}

export function normalizeDictionaryName(value: string): string {
  return value
    .toLocaleLowerCase("tr-TR")
    .replaceAll("ı", "i")
    .normalize("NFKD")
    .replace(/\p{M}/gu, "")
    .replace(/[^a-z0-9]+/g, "");
}

function levenshtein(a: string, b: string): number {
  if (!a) return b.length;
  if (!b) return a.length;
  const previous = Array.from({ length: b.length + 1 }, (_, index) => index);
  for (let i = 1; i <= a.length; i += 1) {
    let diagonal = previous[0];
    previous[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const above = previous[j];
      previous[j] = Math.min(
        previous[j] + 1,
        previous[j - 1] + 1,
        diagonal + (a[i - 1] === b[j - 1] ? 0 : 1),
      );
      diagonal = above;
    }
  }
  return previous[b.length];
}

function similarity(a: string, b: string): number {
  if (!a || !b) return 0;
  if (a === b) return 1;
  return 1 - levenshtein(a, b) / Math.max(a.length, b.length);
}

export function suggestDictionaryColumnMatches(
  definitions: ParsedDictionaryColumn[],
  targets: Pick<ColMeta, "name" | "label">[],
  threshold = 0.78,
): DictionaryColumnMatch[] {
  const candidates: Array<DictionaryColumnMatch & { priority: number }> = [];
  const normalizedTargets = targets.map((target) => ({
    target,
    name: normalizeDictionaryName(target.name),
    label: normalizeDictionaryName(target.label ?? ""),
  }));
  // Fuzzy matching is advisory. Avoid quadratic edit-distance work on very
  // wide datasets; exact normalized name/label matches remain available and
  // unmatched rows can still be mapped manually in the preview.
  const allowFuzzy = definitions.length * targets.length <= 50_000;

  for (const definition of definitions) {
    const source = normalizeDictionaryName(definition.sourceName);
    for (const normalizedTarget of normalizedTargets) {
      const { target, name: targetName, label: targetLabel } = normalizedTarget;
      const exactName = source === targetName;
      const exactLabel = source !== "" && source === targetLabel;
      if (!allowFuzzy && !exactName && !exactLabel) continue;
      const nameScore = exactName ? 1 : similarity(source, targetName);
      const labelScore = exactLabel ? 1 : similarity(source, targetLabel);
      const score = Math.max(nameScore, labelScore);
      if (score < threshold) continue;
      const reason = exactName
        ? "name"
        : exactLabel
          ? "label"
          : "fuzzy";
      candidates.push({
        sourceName: definition.sourceName,
        targetName: target.name,
        score,
        reason,
        priority: reason === "name" ? 2 : reason === "label" ? 1 : 0,
      });
    }
  }

  candidates.sort((a, b) => b.score - a.score || b.priority - a.priority);
  const usedSources = new Set<string>();
  const usedTargets = new Set<string>();
  const matches = new Map<string, DictionaryColumnMatch>();
  for (const candidate of candidates) {
    if (usedSources.has(candidate.sourceName) || usedTargets.has(candidate.targetName)) continue;
    usedSources.add(candidate.sourceName);
    usedTargets.add(candidate.targetName);
    matches.set(candidate.sourceName, {
      sourceName: candidate.sourceName,
      targetName: candidate.targetName,
      score: candidate.score,
      reason: candidate.reason,
    });
  }

  return definitions.map((definition) => matches.get(definition.sourceName) ?? {
    sourceName: definition.sourceName,
    targetName: "",
    score: 0,
    reason: "unmatched",
  });
}
