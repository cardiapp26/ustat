"""Getting a dataset from wherever it lives to wherever the analysis runs.

The engine's statistics already run in two places. The data they run on does
not: on the server it is a DataFrame in `services.store`, and in the browser
there is nothing at all until something puts it there. This subpackage is that
something -- the wire format for one filtered, typed dataset, plus the Select
Cases semantics that decide which rows are in it.

Deliberately NOT imported from `ustat_engine/__init__.py`. Importing this pulls
in pandas, and the browser loads exactly the packages an analysis declares --
`stats.power` asks for numpy/scipy/statsmodels and would fail at
`import ustat_engine` if the package root reached for pandas it had not loaded.
Callers that need a frame import `ustat_engine.frame.envelope` explicitly, by
which point they have paid for pandas on purpose.
"""
from __future__ import annotations
