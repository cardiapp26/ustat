# qa/replay_e2e

The clean-environment replay check for the generated analysis script
(`backend/services/script_export.py`, `POST /api/project/{sid}/script`).

```bash
backend/.venv/bin/python qa/replay_e2e/run.py
```

It starts a fresh uSTAT server, drives it the way a user would (upload,
formula column, cell edit, column delete, case filter), asks that session
for its Python replay script and its prepared CSV, stops the server, starts
a **second** fresh process, runs the generated script against it as a
separate OS process, and asserts the two prepared CSVs are identical.

It also asserts the second server does not know the first server's
session, so "fresh" is verified rather than assumed.

Exit code 0 only on exact equality. Runs in CI as the last step of the
backend job; a failure means the recipe recorded something the script
cannot reproduce, or the script no longer drives the API the GUI uses.

Only the Python flavour is executed here; the R flavour is generated from
the same route table and parsed in `backend/tests/test_script_export.py`,
but running it needs an R with `httr`, which the CI image does not carry.
