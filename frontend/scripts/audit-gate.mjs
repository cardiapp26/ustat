#!/usr/bin/env node
/**
 * The npm audit gate: fail on any high/critical advisory except the ones
 * audit-allowlist.json argues for by GHSA id.
 *
 * Why this exists rather than `npm audit --audit-level=high`: one advisory
 * (maplibre-gl, bundled by plotly.js) has no fixed version to move to and no
 * reachable code path in this app, so the plain gate can only be satisfied by
 * a downgrade that costs a plotly major and patches nothing. A blanket
 * suppression would have hidden every future critical too, so the exception is
 * pinned to that one id.
 *
 * The gate stays honest in three directions:
 *   - anything high+ that is NOT allowlisted fails;
 *   - an allowlisted id that no longer appears fails, so an exception cannot
 *     outlive the problem it was written for;
 *   - an allowlisted id past its reviewBy date fails, so "accepted" is never
 *     permanent.
 *
 * Usage: node scripts/audit-gate.mjs [--audit-json <file>] [--allowlist <file>]
 * Without --audit-json it runs `npm audit --json` itself, retrying the
 * registry's frequent transient endpoint errors. Both flags exist so the
 * gate's own failure modes can be exercised offline, which is what
 * scripts/audit-gate.selftest.mjs does.
 */
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const FRONTEND = resolve(HERE, "..");
const GATED = new Set(["high", "critical"]);

function flag(name) {
  const at = process.argv.indexOf(name);
  return at === -1 ? null : process.argv[at + 1];
}

const ALLOWLIST = resolve(FRONTEND, flag("--allowlist") ?? "audit-allowlist.json");
const AUDIT_JSON = flag("--audit-json");

/** `npm audit --json`, retrying only the registry's transient failures. */
function runAudit() {
  if (AUDIT_JSON) return JSON.parse(readFileSync(resolve(AUDIT_JSON), "utf8"));
  for (let attempt = 1; attempt <= 3; attempt++) {
    let stdout;
    try {
      stdout = execFileSync("npm", ["audit", "--json"], {
        cwd: FRONTEND, encoding: "utf8", maxBuffer: 64 * 1024 * 1024,
      });
    } catch (err) {
      // A non-zero exit is normal: npm exits 1 whenever it found anything.
      // The report is still on stdout, so only an absent/unparsable body is
      // a real failure here.
      stdout = err.stdout ?? "";
    }
    try {
      const parsed = JSON.parse(stdout);
      if (parsed.error) throw new Error(parsed.error.summary ?? "audit endpoint error");
      return parsed;
    } catch (err) {
      if (attempt === 3) {
        console.error(`::error::npm audit did not return a usable report: ${err.message}`);
        process.exit(1);
      }
      console.error(`::warning::npm audit attempt ${attempt} failed (${err.message}); retrying`);
      execFileSync("sleep", [String(attempt * 20)]);
    }
  }
}

/** GHSA id -> {severity, title, packages} for every gated advisory. */
function gatedFindings(report) {
  const found = new Map();
  for (const [pkg, vuln] of Object.entries(report.vulnerabilities ?? {})) {
    for (const via of vuln.via ?? []) {
      // A string `via` means "vulnerable only through another package"; the
      // advisory itself is recorded on that package, so counting it here
      // would double-report one problem under two names.
      if (typeof via !== "object" || !GATED.has(via.severity)) continue;
      const id = (via.url ?? "").split("/").pop() || `${via.source ?? via.name}`;
      const entry = found.get(id) ?? { id, severity: via.severity, title: via.title, packages: new Set() };
      entry.packages.add(pkg);
      found.set(id, entry);
    }
  }
  return found;
}

const allowlist = JSON.parse(readFileSync(ALLOWLIST, "utf8")).allow ?? [];
const allowed = new Map(allowlist.map((entry) => [entry.id, entry]));
const findings = gatedFindings(runAudit());

const problems = [];

for (const [id, finding] of findings) {
  if (allowed.has(id)) continue;
  problems.push(
    `unaccepted ${finding.severity} advisory ${id} in ${[...finding.packages].join(", ")}: ` +
    `${finding.title}\n    Fix it, or argue for it in frontend/audit-allowlist.json.`
  );
}

const today = new Date().toISOString().slice(0, 10);
for (const entry of allowlist) {
  if (!findings.has(entry.id)) {
    problems.push(
      `stale exception ${entry.id} (${entry.package}): the advisory no longer ` +
      `appears, so remove it from frontend/audit-allowlist.json.`
    );
  }
  if (!entry.reviewBy || entry.reviewBy < today) {
    problems.push(
      `expired exception ${entry.id} (${entry.package}): reviewBy ${entry.reviewBy ?? "missing"} ` +
      `has passed. Re-argue it and set a new date, or fix the advisory.`
    );
  }
}

if (problems.length > 0) {
  for (const problem of problems) console.error(`::error::${problem}`);
  process.exit(1);
}

const accepted = [...allowed.keys()].filter((id) => findings.has(id));
console.log(
  `npm audit gate passed: no unaccepted high/critical advisories` +
  (accepted.length ? `; ${accepted.length} documented exception(s): ${accepted.join(", ")}` : "")
);
