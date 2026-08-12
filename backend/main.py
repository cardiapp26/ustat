import os
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from middleware.security_headers import SecurityHeadersMiddleware

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

import logging

from routers import (
    upload, stats, charts, models, session, compute, repeated, advanced_anova, threshold,
    pub_tables, categorical, agreement, reliability, missing_data, decision_curve,
    model_compare, diagnostics, model_diagnostics, pub_export, nomogram,
    survival_advanced, article_parser, code_runner, ml, timeseries, meta,
    multiplicity, factor, bayesian, causal, causal_sem
)

# The agent router is optional: it needs httpx + openai, which a slim
# deployment may omit. Importing it unguarded took the whole app down with a
# 502 when those wheels were missing — one optional feature must never break
# startup for every other endpoint.
try:
    from routers import agent
except ImportError as exc:  # pragma: no cover - depends on the install profile
    agent = None
    _AGENT_IMPORT_ERROR = str(exc)
from services import store

app = FastAPI(title="Wizard Stats API", version="1.0.0")


@app.on_event("startup")
async def _restore_sessions() -> None:
    """Rehydrate any session snapshots the autosave thread wrote before the
    last restart — otherwise a redeploy silently discards in-progress edits."""
    store.load_persisted_sessions()


# ── Global handler: convert FastAPI's "Out of range float values are not JSON
# compliant" crash into a usable 400. This happens when a statistic comes out
# as NaN/Inf (e.g. ANCOVA on a tiny subgroup, Mantel-Haenszel on a degenerate
# stratum) and the response can't be serialised. Returning the original 500
# leaves the user staring at a blank crash.
@app.exception_handler(ValueError)
async def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    msg = str(exc)
    if "JSON compliant" in msg or "Out of range float" in msg:
        return JSONResponse(
            status_code=400,
            content={"detail": (
                "The test produced a non-finite statistic (NaN or Inf), often "
                "caused by a very small subgroup or a degenerate stratum. "
                "Check for outliers, dirty category codes, or empty groups."
            )},
        )
    # Anything else: let FastAPI's default 500 handler take over.
    raise exc


app.add_middleware(SecurityHeadersMiddleware)

# CORS: env-driven origin allow-list. Wildcard ("*") rejected by the OWASP
# semgrep gate, and dangerous in production anyway because it disables
# CSRF protection on cookie-bearing requests. Defaults below cover the
# production domain + local dev ports; override via CORS_ALLOWED_ORIGINS
# (comma-separated) for staging or custom deployments.
_DEFAULT_ORIGINS = (
    "https://ustat.drtr.uk,"
    "http://localhost:5173,http://localhost:5174,http://localhost:5175,"
    "http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:5175"
)
_origins_env = os.environ.get("CORS_ALLOWED_ORIGINS", _DEFAULT_ORIGINS)
_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(stats.router, prefix="/api/stats", tags=["stats"])
app.include_router(charts.router, prefix="/api/charts", tags=["charts"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(session.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(compute.router, prefix="/api/compute", tags=["compute"])
app.include_router(repeated.router, prefix="/api/repeated", tags=["repeated"])
app.include_router(advanced_anova.router, prefix="/api/advanced_anova", tags=["advanced_anova"])
app.include_router(pub_tables.router, prefix="/api/pub_tables", tags=["pub_tables"])
app.include_router(categorical.router, prefix="/api/categorical", tags=["categorical"])
app.include_router(agreement.router, prefix="/api/agreement", tags=["agreement"])
app.include_router(reliability.router, prefix="/api/reliability", tags=["reliability"])
app.include_router(missing_data.router, prefix="/api/missing_data", tags=["missing_data"])
app.include_router(decision_curve.router, prefix="/api/decision_curve", tags=["decision_curve"])
app.include_router(model_compare.router, prefix="/api/model_compare", tags=["model_compare"])
app.include_router(diagnostics.router, prefix="/api/diagnostics", tags=["diagnostics"])
app.include_router(model_diagnostics.router, prefix="/api/model_diagnostics", tags=["model_diagnostics"])
app.include_router(pub_export.router, prefix="/api/pub_export", tags=["pub_export"])
app.include_router(nomogram.router, prefix="/api/nomogram", tags=["nomogram"])
app.include_router(threshold.router, prefix="/api/threshold", tags=["threshold"])
app.include_router(survival_advanced.router, prefix="/api/survival_advanced", tags=["survival_advanced"])
app.include_router(article_parser.router, prefix="/api/article_parser", tags=["article_parser"])
app.include_router(code_runner.router, prefix="/api/code", tags=["code_runner"])
app.include_router(ml.router, prefix="/api/ml", tags=["ml"])
app.include_router(timeseries.router, prefix="/api/timeseries", tags=["timeseries"])
app.include_router(meta.router, prefix="/api/meta", tags=["meta"])
app.include_router(multiplicity.router, prefix="/api/multiplicity", tags=["multiplicity"])
app.include_router(factor.router, prefix="/api/factor", tags=["factor"])
app.include_router(bayesian.router, prefix="/api/bayesian", tags=["bayesian"])
app.include_router(causal.router, prefix="/api/causal", tags=["causal"])
app.include_router(causal_sem.router, prefix="/api/causal", tags=["causal"])
if agent is not None:
    app.include_router(agent.router, prefix="/api/agent", tags=["agent"])
else:
    logging.getLogger(__name__).warning(
        "Agent router disabled - optional dependency missing: %s", _AGENT_IMPORT_ERROR
    )


@app.get("/.well-known/security.txt", response_class=PlainTextResponse)
def security_txt() -> str:
    """RFC 9116 security.txt — vulnerability disclosure contact.

    See: https://www.rfc-editor.org/rfc/rfc9116
    """
    expires = (datetime.now(timezone.utc) + timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
    contact = os.environ.get("SECURITY_CONTACT_EMAIL", "security@drtr.uk")
    return (
        f"Contact: mailto:{contact}\n"
        f"Expires: {expires}\n"
        "Preferred-Languages: en, tr\n"
        "Canonical: https://ustat.drtr.uk/.well-known/security.txt\n"
        "Policy: https://ustat.drtr.uk/security\n"
    )


@app.get("/security")
def security_page_redirect() -> RedirectResponse:
    """Clean URL → static security overview."""
    return RedirectResponse(url="/security.html", status_code=308)


@app.get("/privacy")
def privacy_page_redirect() -> RedirectResponse:
    return RedirectResponse(url="/privacy.html", status_code=308)


@app.get("/terms")
def terms_page_redirect() -> RedirectResponse:
    return RedirectResponse(url="/terms.html", status_code=308)


@app.get("/api/health")
def health():
    """Lightweight health check — no expensive deep memory scan."""
    result: dict = {"status": "ok", "active_sessions": len(store.list_sessions())}

    if psutil:
        process = psutil.Process()
        mem_info = process.memory_info()
        result["memory"] = {
            "process_rss_mb": round(mem_info.rss / (1024 * 1024), 1),
            "process_percent": round(process.memory_percent(), 1),
        }

    return result


# Serve compiled React frontend (production build).
# Must come AFTER all /api routes so API routes are matched first.
_dist = os.environ.get(
    "USTAT_FRONTEND_DIST",
    os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"),
)
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="static")
