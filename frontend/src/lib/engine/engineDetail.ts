/**
 * How to name the engine that produced a number, in one place.
 *
 * These strings appear next to results, so they are claims about what ran, not
 * decoration. The R side is read off the running engine at boot
 * (`rEngineDetail()`); the Python side cannot be, because neither
 * `/api/engine/identity` nor the Pyodide wheel reports its library versions --
 * they report the fingerprint of OUR code, which is a different question.
 *
 * So the Python detail is a constant, and `engineDetail.test.ts` reads
 * `backend/requirements.txt` and fails if the pin moves without this moving
 * with it. A hardcoded version that cannot go stale unnoticed is honest; one
 * that can is not.
 *
 * Both sides run the same scipy on purpose: `backend/requirements.txt` pins
 * numpy/scipy/pandas to the versions Pyodide ships, so a server answer and a
 * browser Python answer come from the same library build.
 */

/** The scipy the server and the Pyodide wheel both run. Pinned in backend/requirements.txt. */
export const SCIPY_VERSION = "1.14.1";

/** CPython version. Pyodide 0.27.x is a cp312 build -- see the wheel filenames
 *  under `frontend/public/pyodide/wheels/`. */
export const PYTHON_VERSION = "3.12";

export const PYTHON_ENGINE_DETAIL = `Python ${PYTHON_VERSION} · scipy ${SCIPY_VERSION}`;

/**
 * The amber line shown when an R session had to answer with Python.
 *
 * Says three things, because a reader who chose R and got Python deserves all
 * three: that the gap is coverage rather than failure, what actually computed
 * the number, and that the number is not second-best.
 */
export const PYTHON_FALLBACK_NOTICE =
  `Not yet available in R — computed with Python (scipy ${SCIPY_VERSION}). ` +
  `Numbers are unchanged from the Python engine.`;
