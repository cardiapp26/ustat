# qa/r_reference

A pinned, reproducible R environment for regenerating the R reference
literals used by `qa/models_audit`, `qa/power_audit` and `qa/tests_audit`.

## Why this exists

Those three audits each have a `reference.R` that produces a
`reference.json` of R-computed numbers, which uSTAT's Python (and,
increasingly, in-browser webR) statistics are checked against. Until now,
regenerating that JSON meant running `Rscript` on whatever R + CRAN package
versions happened to be installed on one person's laptop. That is fine for
day-to-day development, but it means "what does R actually say" is not
reproducible, and it silently drifts whenever a package updates.

This directory makes "regenerate the R reference under a known R version"
one command (`./run.sh`) instead of one person's machine state. It builds a
Docker image pinned to a specific R release, installs exactly the packages
the three `reference.R` scripts need, and runs them against the repo's
existing datasets, writing fresh `reference.json` files to an output
directory of your choosing — without modifying anything inside the repo.

## The pin

`Dockerfile` currently pins `FROM r-base:4.6.0`.

**This pin should track whatever R version webR ships**, not the version
on any given contributor's laptop or CI runner. uSTAT is starting to run
its statistics in-browser via webR, and webR bundles its own R + a wasm
CRAN mirror that can lag or lead the versions anyone has installed locally.
The point of pinning here is to be able to ask, concretely, "if we ran the
reference script under the R (and CRAN package set) that webR actually
ships, would any number move?" — see `qa/models_audit/reference.json`'s
`meta.packages` for the versions the *committed* reference.json was built
against, and compare against a run of this image.

When webR moves to a new R release, bump the `FROM` line here to match (or
to whatever release is closest to it that `r-base` publishes) and re-run.

## Usage

```bash
qa/r_reference/run.sh [OUT_DIR]
```

- Builds the image (`docker build`). Expect several minutes the first time —
  `lme4`/`Matrix`/`rms`/`survey`/etc. compile from source on `r-base`, which
  ships no CRAN binaries.
- Runs all three `reference.R` scripts inside a container with the repo
  mounted **read-only** at `/repo` (nothing in the repo is modified). The
  three `qa/*_audit` directories are copied into the container's writable
  layer first, since each script writes `reference.json` next to itself.
- Copies the resulting `reference.json` files out to
  `OUT_DIR/{models_audit,power_audit,tests_audit}/reference.json`.
  `OUT_DIR` defaults to `qa/r_reference/r46-out/` if not given.

Diff the output against the committed `reference.json` files to see what,
if anything, changed under the newer R/package versions. If a package fails
to install, the build log says which one and the rest still install — it is
not a silent skip.

## What's installed

Every package `qa/models_audit/reference.R` touches via
`library()`/`requireNamespace()`/`::` (`survival`, `MASS`, `logistf`,
`ordinal`, `lme4`, `geepack`, `MatchIt`, `survey`, `car`, `pROC`, `rms`),
plus `pwr` and `powerSurvEpi` for `qa/power_audit`, plus `jsonlite`.
`qa/tests_audit/reference.R` uses base R only.
