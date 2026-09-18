FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install --legacy-peer-deps
COPY frontend/ ./
RUN npx vite build

FROM python:3.11-slim
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY --from=frontend-builder /app/frontend/dist ../frontend/dist

# Run as non-root for defense-in-depth — addresses the Semgrep
# dockerfile.security.missing-user finding. UID 10001 is a high arbitrary
# UID that doesn't clash with common host users when the image is run with
# --userns=keep-id.
#
# The Code Runner executes user-supplied Python as this SAME 'app' user, so the
# application tree must NOT be writable by 'app' — otherwise sandboxed code
# could overwrite the served frontend bundle or the backend source. Keep /app
# root-owned and non-writable; grant write only to the log dir and the /tmp
# scratch the sandbox runs from.
ENV PYTHONDONTWRITEBYTECODE=1
RUN groupadd -r app --gid 10001 \
 && useradd -r -g app --uid 10001 --no-create-home --shell /sbin/nologin app \
 && mkdir -p /tmp/sandbox /app/backend/logs /app/backend/session_cache \
 && chown -R root:root /app \
 && chmod -R go-w /app \
 && chown -R app:app /app/backend/logs /app/backend/session_cache /tmp/sandbox \
 && chmod 0770 /app/backend/logs /app/backend/session_cache /tmp/sandbox
USER app

EXPOSE 8000
# No --limit-max-requests: sessions live in process RAM, so a worker restart
# silently drops every active session. Memory pressure is bounded instead by
# the session TTL and max-session cap enforced in backend/services/store.py.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "30"]
