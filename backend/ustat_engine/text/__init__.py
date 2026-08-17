"""The prose an analysis returns, next to the numbers it returns.

A result in this app is not a number; it is a Methods sentence, a Results
sentence and a reproducible R snippet, all of which quote the number and have
to agree with it. `p = 0.035` in the paragraph and `<0.001` in the table is not
a formatting nit -- it is two different claims in one response.

So the generators live in the engine alongside the arithmetic, and
`services.text_generators` / `services.number_format` re-export them. Pure
string formatting, no third-party imports: the browser pays nothing for this.
"""
from __future__ import annotations
