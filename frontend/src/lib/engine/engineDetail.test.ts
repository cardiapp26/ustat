/**
 * A hardcoded version that cannot go stale unnoticed.
 *
 * `PYTHON_ENGINE_DETAIL` names the scipy that produced a number, next to the
 * number. Nothing at runtime reports it -- `/api/engine/identity` fingerprints
 * OUR sources, not the libraries under them -- so it is a constant, and a
 * constant naming someone else's version is a lie waiting to happen. This reads
 * the pin it claims to reflect and fails when the two part company.
 */
/// <reference types="node" />
// Referenced for this file only, the way frontend/src/lib/publicAssets.test.ts
// does it: tsconfig.app.json limits `types` to vite/client so browser code
// cannot reach for node globals by accident, and this check has to read a file
// outside the frontend package.
import { describe, expect, it } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { PYTHON_ENGINE_DETAIL, SCIPY_VERSION } from "./engineDetail";

// vitest runs in the `frontend` package root; the backend's pin is one level up.
const REQUIREMENTS = resolve(process.cwd(), "../backend/requirements.txt");

describe("the Python engine's advertised version", () => {
  it("is checked against a file that exists", () => {
    // A wrong path would make the assertion below vacuous, and a guard that
    // cannot fail is worse than no guard because it is believed.
    expect(existsSync(REQUIREMENTS), `expected backend/requirements.txt at ${REQUIREMENTS}`).toBe(
      true,
    );
  });

  it("matches the scipy pinned in backend/requirements.txt", () => {
    const pin = readFileSync(REQUIREMENTS, "utf8")
      .split("\n")
      .map((line) => line.trim())
      .find((line) => line.startsWith("scipy=="));

    expect(pin, "backend/requirements.txt no longer pins scipy").toBeDefined();
    expect(
      `scipy==${SCIPY_VERSION}`,
      `frontend/src/lib/engine/engineDetail.ts advertises scipy ${SCIPY_VERSION}, ` +
        `but backend/requirements.txt pins "${pin}". Update the constant -- it is ` +
        `printed next to results, so a stale value misattributes a number.`,
    ).toBe(pin);
  });

  it("reads as a sentence a clinician can act on", () => {
    expect(PYTHON_ENGINE_DETAIL).toBe(`Python 3.12 · scipy ${SCIPY_VERSION}`);
  });
});
