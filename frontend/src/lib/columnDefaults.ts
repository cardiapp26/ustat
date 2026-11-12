import type { ColMeta, Session } from "../store";

const ID_NAME_TOKENS = new Set(["id", "no", "num", "number", "patient", "subject", "case", "record", "mrn"]);
const EXPLICIT_ID_TOKENS = new Set(["id", "pid", "mrn"]);

function lastNameToken(name: string): string {
  const tokens = name.replace(/([a-z0-9])([A-Z])/g, "$1 $2").toLowerCase().match(/[a-z0-9]+/g);
  return tokens?.at(-1) ?? "";
}

/** Recognize likely grouping labels without requiring many distinct subjects. */
export function hasIdentifierName(name: string): boolean {
  const token = lastNameToken(name);
  return EXPLICIT_ID_TOKENS.has(token) || ID_NAME_TOKENS.has(token);
}

/** Use the full-data classification when provided. Older session snapshots may only have a preview. */
export function isIdLike(column: ColMeta, session: Session): boolean {
  if (column.id_like !== undefined) return column.id_like;

  const values = session.preview.map((row) => row[column.name]).filter((value) => value != null);
  if (values.length < 5) return false;
  const distinct = new Set(values).size;
  if (ID_NAME_TOKENS.has(lastNameToken(column.name)) && distinct >= 10) return true;
  const integer = /^(u?int(8|16|32|64))$/i.test(column.dtype);
  const nonNumeric = !/^(float\d*|complex\d*|bool|boolean)$/i.test(column.dtype) && !integer;
  return session.rows === session.preview.length && (integer || nonNumeric) && distinct / values.length > 0.95;
}

/** Filter automatic selections only. Keep the source lists for manual picks. */
export function analysisDefaultColumns(session: Session | null, names: string[]): string[] {
  if (!session) return names;
  const candidates = new Set(names);
  const identifiers = new Set(session.columns
    .filter((column) => candidates.has(column.name) && (
      column.role === "id" || (column.kind !== "date" && (
        EXPLICIT_ID_TOKENS.has(lastNameToken(column.name))
        || (hasIdentifierName(column.name) && isIdLike(column, session))
      ))
    ))
    .map((column) => column.name));
  return names.filter((name) => !identifiers.has(name));
}
