# Validation cards

One card per statistical method family, machine-readable YAML. A card is the
single place that answers, for its method: what computes it, under which
assumptions, how edge inputs behave, what reference output it is checked
against, with what tolerance, and what is still open.

Cards are ENFORCED, not decorative: `backend/tests/test_validation_cards.py`
fails CI when a card is malformed, when a card's reference artifact pointer
does not resolve, when a card names an endpoint the app does not serve, or
when a reference entry in `qa/*/reference.json` / `qa/parity/*.json` is not
claimed by exactly one card. Adding a reference method without writing its
card is a red build.

## Schema

```yaml
method: slug                # unique
title: Human name
panel: models               # UI surface
endpoints: [/api/...]       # must exist in the app
covers: [models:linear]     # reference keys this card owns (see below)
implementation:
  library: statsmodels      # or "custom" -- say so explicitly
  entrypoint: what actually runs, file/function level
  version_source: /api/engine/identity (X-uStat-Packages)
assumptions: [...]
unsupported: [...]          # named, visible gaps
behavior:                   # the four data-semantics questions, always answered
  missing_data: "..."
  weights: "..."
  ties: "..."               # or "not applicable"
  category_coding: "..."
reference:
  software: "R (pinned image qa/r_reference)"
  entrypoint: stats::lm     # the R call
  artifacts:                # where the literals live
    - {file: qa/models_audit/reference.json, key: models.linear}
compared: [coefficients, standard_errors, p_values]
tolerance:
  rel: 1.0e-4
  rationale: why this bound and not a tighter one
edge_cases:                 # status: covered (test named) | open
  - {case: "...", status: covered, test: backend/tests/...}
  - {case: "...", status: open}
findings:                   # audit findings; status: open | fixed | re-verified
  - {id: "...", summary: "...", status: fixed, evidence: "..."}
```

`covers` keys: `models:<key>` for `qa/models_audit/reference.json`'s `models`
map, `tests:<key>` for `qa/tests_audit/reference.json` (minus `meta`),
`parity:<file>` for a whole `qa/parity/<file>.json`.

Old audit findings are not "open" by default: every finding carries an
explicit status, and re-verification cites its evidence (a test, a compare
run). That is the roadmap's open / fixed / re-verified requirement.
