/**
 * Shared vocabulary between the main thread and the webR worker.
 *
 * The R counterpart of `../types.ts`, and separate from it on purpose: the two
 * workers boot from different bases, install different things, and prove their
 * identity by different routes, so a single union covering both would be a type
 * whose members were mostly inapplicable at every call site. What IS shared --
 * `EngineKind`, `Runtime`, `LocalUnavailableReason`, `LocalRunFailure` -- is
 * imported from there rather than restated here, because a local run must fail
 * with the same named reasons whichever engine was asked.
 */
import type { DistributiveOmit, LocalUnavailableReason } from "../types";

/** What `ustat_identity_json()` in the bundle answers with. */
export interface REngineIdentity {
  schema: string;
  /** e.g. "4.6.0" -- R's own version, not webR's. */
  r_version: string;
  analyses: string[];
  packages: string[];
}

/** One analysis as the FETCHED manifest declares it (snake_case, as written). */
export interface RManifestEntry {
  id: string;
  needs_frame: boolean;
  packages: string[];
  r_packages: string[];
}

export interface RBundleManifest {
  version: string;
  bundle: string;
  /** sha256 over the bundle's bytes. Checked before the bundle is evaluated. */
  bundle_sha256: string;
  bundle_bytes: number;
  /** sha256 over backend/ustat_engine_r/'s sources, as the SERVER computed it. */
  source_fingerprint: string;
  modules: number;
  analyses: RManifestEntry[];
}

/** What a successful boot tells the main thread. */
export interface RBootInfo {
  identity: REngineIdentity;
  /**
   * `manifest.source_fingerprint`, forwarded verbatim.
   *
   * The browser cannot recompute this: the bundle is a concatenation and the
   * fingerprint is per-file. So the worker proves the bundle matches its
   * manifest (sha256), and the main thread proves the manifest matches the
   * server (this value against GET /api/engine/r/identity). Both links, or no
   * local run.
   */
  sourceFingerprint: string;
  /** The manifest's own declarations, for the build-time copy to be checked against. */
  manifestAnalyses: RManifestEntry[];
  webrVersion: string;
  bundleBytes: number;
  /** Wall-clock milliseconds this boot took, for the honest cost claim. */
  bootMs: number;
}

export type RWorkerRequest =
  | {
      id: number;
      cmd: "init";
      /** Which analysis the boot is for, so the worker installs its packages. */
      analysisId: string;
      /**
       * Packages the BUNDLE needs whatever runs -- jsonlite. Per-analysis
       * packages are deliberately NOT sent from here: the worker reads them off
       * the manifest it fetched, which is the authority, so a spec that gains a
       * dependency does not need a matching edit on this side.
       */
      packages: string[];
      runtimeBase: string;
      bundleBase: string;
      repoUrl: string;
    }
  /** Hand the worker a `ustat.frame/1` envelope to keep resident under `frameKey`. */
  | { id: number; cmd: "frame"; frameKey: string; envelope: unknown }
  | { id: number; cmd: "run"; analysisId: string; params: unknown; frameKey?: string };

export type RWorkerResponse =
  | { id: number; ok: true; result: unknown }
  | { id: number; ok: false; reason: LocalUnavailableReason; detail: string }
  | { id: number; ok: true; kind: "init"; boot: RBootInfo };

export type RWorkerRequestBody = DistributiveOmit<RWorkerRequest, "id">;
