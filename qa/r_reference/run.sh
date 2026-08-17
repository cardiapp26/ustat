#!/usr/bin/env bash
# Build the pinned-R image and regenerate the three R reference.json files
# under it, without touching anything in the checked-out repo.
#
# Usage:
#   qa/r_reference/run.sh [OUT_DIR]
#
# OUT_DIR defaults to ./r46-out next to this script if not given. The repo
# is mounted read-only at /repo; reference.R et al. write reference.json
# next to themselves, so this script copies qa/models_audit, qa/power_audit
# and qa/tests_audit into the container's writable layer, runs the three
# scripts there (none of the original scripts are modified), and then
# copies the resulting reference.json files out to OUT_DIR.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR/r46-out}"
IMAGE_TAG="ustat-r-reference:4.6.0"

mkdir -p "$OUT_DIR"

echo "== building $IMAGE_TAG (this compiles lme4/Matrix/rms/etc from source and can take several minutes) =="
time docker build -t "$IMAGE_TAG" -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"

echo "== running the three reference scripts inside the container =="
docker run --rm \
  -v "$REPO_ROOT:/repo:ro" \
  -v "$OUT_DIR:/out" \
  "$IMAGE_TAG" \
  bash -c '
    set -euo pipefail
    mkdir -p /work/qa
    for d in models_audit power_audit tests_audit; do
      cp -r "/repo/qa/$d" "/work/qa/$d"
      chmod -R u+w "/work/qa/$d"
    done
    cd /work

    echo "-- models_audit --"
    Rscript qa/models_audit/reference.R

    echo "-- power_audit --"
    Rscript qa/power_audit/reference.R

    echo "-- tests_audit --"
    Rscript qa/tests_audit/reference.R

    mkdir -p /out/models_audit /out/power_audit /out/tests_audit
    cp qa/models_audit/reference.json /out/models_audit/reference.json
    cp qa/power_audit/reference.json  /out/power_audit/reference.json
    cp qa/tests_audit/reference.json  /out/tests_audit/reference.json
    echo "wrote reference.json for all three audits to /out"
  '

echo "== done. R 4.6.0 reference.json files are in: $OUT_DIR =="
