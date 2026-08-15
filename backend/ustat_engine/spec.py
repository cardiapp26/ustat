"""What the two runtimes need to agree on about a single analysis.

An analysis is not just a function. Before either caller can run one it has to
answer questions the function itself cannot: which Python packages must be
loaded first (the browser pays a download for each), which columns of the
dataset it will touch (so a consented server run can upload those and nothing
else), and roughly what it will cost (so a heavy job can be recognised before
it freezes a tab). Those answers live here, next to the function, because a
copy of them in the frontend would drift from it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class AnalysisSpec:
    """One analysis, described well enough to run it in either runtime.

    `fn` takes `(frame, params)` and returns a JSON-safe dict. `frame` is None
    for the analyses that do not read the dataset at all -- power analysis and
    meta-analysis both compute from numbers typed into the form, which is why
    they are the first two to move.
    """

    id: str
    fn: Callable[..., dict]
    needs_frame: bool = True

    # Package names as Pyodide knows them. The browser loads exactly these
    # rather than everything: a t-test has no reason to pay for scikit-learn.
    deps: Sequence[str] = ("numpy",)

    # Which columns this run will read, given its params. A consented
    # server-side run uploads the answer to this and nothing else, so it has
    # to be derivable without touching the data.
    required_columns: Callable[[dict], list[str]] | None = None

    # Key into cost_model.json. None means "not costed yet", which the caller
    # must treat as heavy rather than as free.
    cost_key: str | None = None

    doc: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def columns_for(self, params: dict) -> list[str]:
        if self.required_columns is None:
            return []
        return [c for c in self.required_columns(params) if c]
