#!/usr/bin/env node
/**
 * Proves the npm audit gate is a gate.
 *
 * A suppression mechanism that cannot be shown to fail is indistinguishable
 * from `|| true`, so each of the four outcomes is exercised against a fixed
 * audit report: accept the documented exception, reject an undocumented
 * advisory, reject an exception whose advisory has gone, reject an expired
 * exception.
 *
 * Run: node scripts/audit-gate.test.mjs
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const GATE = resolve(HERE, "audit-gate.mjs");
const work = mkdtempSync(join(tmpdir(), "audit-gate-"));

const FUTURE = "2099-01-01";
const PAST = "2000-01-01";

/** The shape npm audit --json produces: one direct advisory plus a package
 *  that is only vulnerable through it. */
const REPORT = {
  vulnerabilities: {
    "maplibre-gl": {
      name: "maplibre-gl",
      severity: "critical",
      via: [{
        source: 1109999, name: "maplibre-gl", severity: "critical",
        title: "XSS Sanitizer Bypass in DOM.sanitize()",
        url: "https://github.com/advisories/GHSA-jrc7-96c5-q579", range: "<=6.4.0",
      }],
    },
    "plotly.js": { name: "plotly.js", severity: "critical", via: ["maplibre-gl"] },
  },
};

const REPORT_WITH_NEW_CRITICAL = {
  vulnerabilities: {
    ...REPORT.vulnerabilities,
    "something-else": {
      name: "something-else", severity: "critical",
      via: [{
        source: 1110000, name: "something-else", severity: "critical",
        title: "Remote code execution",
        url: "https://github.com/advisories/GHSA-aaaa-bbbb-cccc", range: "<1.0.0",
      }],
    },
  },
};

const ENTRY = {
  id: "GHSA-jrc7-96c5-q579", package: "maplibre-gl", severity: "critical",
  title: "XSS Sanitizer Bypass", reviewBy: FUTURE, why: ["test fixture"],
};

function write(name, data) {
  const path = join(work, name);
  writeFileSync(path, JSON.stringify(data, null, 2));
  return path;
}

/** Returns {code, out}. */
function runGate(reportPath, allowlistPath) {
  try {
    const out = execFileSync("node", [GATE, "--audit-json", reportPath, "--allowlist", allowlistPath],
      { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    return { code: 0, out };
  } catch (err) {
    return { code: err.status ?? 1, out: `${err.stdout ?? ""}${err.stderr ?? ""}` };
  }
}

const cases = [
  {
    name: "accepts the documented exception and nothing else",
    report: REPORT, allow: { allow: [ENTRY] },
    expectCode: 0, expectText: "documented exception",
  },
  {
    name: "rejects an advisory that is not allowlisted",
    report: REPORT_WITH_NEW_CRITICAL, allow: { allow: [ENTRY] },
    expectCode: 1, expectText: "unaccepted critical advisory GHSA-aaaa-bbbb-cccc",
  },
  {
    name: "rejects every high+ advisory when the allowlist is empty",
    report: REPORT, allow: { allow: [] },
    expectCode: 1, expectText: "unaccepted critical advisory GHSA-jrc7-96c5-q579",
  },
  {
    name: "rejects an exception whose advisory no longer appears",
    report: { vulnerabilities: {} }, allow: { allow: [ENTRY] },
    expectCode: 1, expectText: "stale exception",
  },
  {
    name: "rejects an exception past its reviewBy date",
    report: REPORT, allow: { allow: [{ ...ENTRY, reviewBy: PAST }] },
    expectCode: 1, expectText: "expired exception",
  },
  {
    name: "rejects an exception with no reviewBy date at all",
    report: REPORT, allow: { allow: [{ ...ENTRY, reviewBy: undefined }] },
    expectCode: 1, expectText: "expired exception",
  },
];

let failed = 0;
for (const [i, c] of cases.entries()) {
  const reportPath = write(`report-${i}.json`, c.report);
  const allowPath = write(`allow-${i}.json`, c.allow);
  const { code, out } = runGate(reportPath, allowPath);
  const ok = code === c.expectCode && out.includes(c.expectText);
  console.log(`${ok ? "ok  " : "FAIL"} ${c.name}`);
  if (!ok) {
    failed++;
    console.log(`     expected exit ${c.expectCode} containing ${JSON.stringify(c.expectText)}`);
    console.log(`     got exit ${code}: ${out.trim()}`);
  }
}

if (failed > 0) {
  console.error(`${failed} gate self-test(s) failed`);
  process.exit(1);
}
console.log(`all ${cases.length} gate self-tests passed`);
