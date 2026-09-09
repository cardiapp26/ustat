# wiz3 Product Maturity Roadmap

Companion to [SPEC.md](SPEC.md), [IMPROVEMENT_OPPORTUNITIES.md](IMPROVEMENT_OPPORTUNITIES.md)
and [ROADMAP_advanced_features.md](ROADMAP_advanced_features.md). Where those
documents enumerate features and bugs, this one sets the product bar: move a
wide feature list to a consistent product standard. The target is a working
environment that combines SPSS's data-management discipline, jamovi's project
experience, and R's reproducibility. Feature parity with all of R is not a
realistic goal; strong R integration is.

## Priority matrix

| Priority | Area | What to do | Done when |
|---|---|---|---|
| **P0** | Statistical validation | Reference result, edge cases and a justified tolerance for every method | No unexplained deviation in critical methods |
| **P0** | Project file | Store data, transformations, analysis settings, results, plots and versions together | Open the project on another machine and reproduce the analyses |
| **P0** | Result integrity | Mark results affected by data/filter/setting changes | A stale result cannot be exported as current |
| **P0** | Data semantics | Preserve missing codes, category order, measurement level and reference group end to end | No loss of meaning across import, save, open, export |
| **P1** | Analysis history and syntax | Generate an executable analysis recipe from GUI actions | Re-run raw data to final report with one command |
| **P1** | Consistent UX | Same settings, warnings and result layout across all analyses | User never re-learns a panel |
| **P1** | Modeling depth | Complete option and diagnostic coverage of existing models | Supported options are explicit and tested |
| **P1** | Performance and resilience | Progress, cancel, recovery and resource limits for long jobs | Measured success at defined data sizes |
| **P2** | Plugin ecosystem | Versioned, documented module interface | Add a validated analysis without touching the core |

## P0.1 Project file (biggest product priority)

The current JSON export carries data, metadata, filters and the audit log,
but not the full set of analysis objects, the result tree, or plot layouts
(see `backend/routers/session.py`). A panel cache is not a portable research
project. A project file must contain:

- Raw data and the transformation steps applied to it.
- Variable dictionary, missing-value rules, category order.
- Multiple saved analyses with their names and settings.
- Results, plot layouts, researcher notes.
- Data digest/hash, engine and package versions, random seed.
- A file schema version and migration from older versions.

A user should be able to keep "Model 1", "Model 2", "Sensitivity analysis"
on the same data and lose none of it when the program closes. jamovi is the
reference here: one file holds data, analyses, options and results.

## P0.2 Validation, managed separately from test counts

An endpoint that runs does not prove it computes the right statistic under
the right assumptions. Every analysis gets a validation card:

- Library, function and version used.
- Assumptions and unsupported cases.
- Behavior for missing data, weights, ties, category coding.
- Reference software output.
- Comparison of coefficients, standard errors, confidence intervals,
  degrees of freedom and p values.
- Acceptance tolerance and explained engine differences.

Edge cases to cover explicitly: tiny samples, constant columns, empty
groups, complete separation, singular model matrices, non-convergence.

Every finding from earlier audits carries a status: **open / fixed /
re-verified**. Old reports must not be treated as currently open by default;
the current tree already contains regression tests for many of them.

Status notes (2026-09-09):

- R reference environment: R version was pinned but packages installed from
  live CRAN; now pinned to a dated Posit Package Manager snapshot
  (`qa/r_reference/install_packages.R`).

## P0.3 Result integrity

Status notes (2026-09-09): first pass landed. Results are stamped with the
data state and provenance that produced them (`frontend/src/lib/resultStamp.ts`,
`useStampedResult`), stale results are marked (`StaleResultNotice`) and the
exporter refuses to export a stale result as current (`ResultExporter`).
Remaining: extend stamping to every panel, and carry stamps into the future
project file.

## P0.4 Data semantics

Keep the existing data dictionary and editing tools, and test them against:

- User-defined missing codes (`99`, `999`, "don't know").
- Separation of numeric code from category label.
- User-defined order for ordinal variables.
- Dates, comma decimals, Turkish characters.
- File merge and key-based matching.
- Long/wide reshaping.
- Persistent visibility of filter, weight and split-file state.

Frequency weights and sampling weights are not the same thing and must not
be processed as if they were. If complex survey support is a goal, strata,
clusters/PSU and design-appropriate uncertainty need separate validation;
SPSS ships this as its own analysis domain.

## P1 Reproducible analysis from the GUI

A code template or an audit log is not enough. User actions should build an
executable recipe: import, define missing codes, filter, transform, fit,
report. The recipe must not depend on invisible panel state and must include
the import and preparation steps.

- An "R code / Python code" view on every analysis.
- Whole-project script export.
- GUI settings and the code generated from one shared analysis definition.
- Compare GUI output and script output in a clean environment.

jamovi's R syntax mode is the model; its docs also note the data-import step
must be reproduced separately, which is exactly the gap to close here.

Status (2026-09-10): the core landed, see
[DESIGN_project_file.md](DESIGN_project_file.md). Every data-mutating
request is recorded as a replayable step including the import
(`prep/steps.json`); whole-project replay scripts export in Python and R
and drive uSTAT's own API, so GUI and script output are identical by
construction; a syntax view shows covered analyses as Python/R code with
an explicit "no translation yet" for the rest. Remaining: widening the
syntax-template allow-list, and a clean-environment CI job that runs a
generated replay script against a fresh server.

## P1 Modeling depth before new methods

Mixed models, SEM and MICE already exist. Next step is a coverage matrix,
not new methods:

- Mixed models: random slopes, nested/crossed structures, convergence
  diagnostics.
- Regression: reference category, contrasts, interactions, estimated
  marginal means.
- Repeated measures: corrections and multiple-comparison options.
- Multiple imputation: supported models, pooling, diagnostics.
- SEM/CFA: scope of current path analysis, fit indices, limits.
- Bayes: explicit scope of supported methods and proper diagnostics.

Unsupported options must be visible. Approximate or two-stage methods must
not share a name with the exact method.

## P1/P2 Release and maintenance standard

Known documentation/CI inconsistencies (all fixed 2026-09-09):

- README described a server-centric architecture while local engines exist
  in the code; it now documents the in-browser engine path.
- SPEC described disk persistence as default behavior; it now states the
  opt-in `SESSION_DISK_CACHE=1` default-off reality.
- SPEC carried stale test counts and a `tsc --noEmit` check; it now points
  at CI for counts and names `tsc -b`.
- CI ran typecheck, tests and build but no lint; the frontend job now runs
  `eslint . --quiet`.

Release criteria: a current method inventory, passing reference tests,
end-to-end tests from browser to export, release notes, known limitations.

## Suggested order

1. **Trustworthy core**: validation inventory, package pinning (done for the
   R reference image), finish the result-tracking work (first pass landed),
   documentation consistency.
2. **Research project**: full project file, saved analyses, re-run and
   recovery.
3. **Shared analysis experience**: model options, data dictionary, result
   tree, editable publication outputs.
4. **Growth**: performance targets, module API, training material,
   independent validation.
