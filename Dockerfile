# Deep Research Agent
#
# Two stages: the builder compiles a wheel and the runtime installs it, so the
# image never carries build-time cruft (pip, hatchling, the source tree).

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN pip install --no-cache-dir build

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m build --wheel --outdir /dist


FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Deep Research Agent" \
      org.opencontainers.image.description="Multi-agent research pipeline on LlamaIndex Workflows" \
      org.opencontainers.image.source="https://github.com/HassanSayedTesla/deep-research-agent"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RUNS_DIR=/data/runs \
    CACHE_FILE=/data/cache/search.json

# curl is only here for the compose healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /dist/*.whl /tmp/

# Install dependencies from the wheel, then drop the wheel itself.
RUN pip install /tmp/*.whl && rm -f /tmp/*.whl

# Never run as root: a research run makes outbound network calls and writes
# reports, and there is no reason for that to happen as uid 0.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/runs /data/cache \
    && chown -R appuser:appuser /data
USER appuser

WORKDIR /app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "deep_research.server:app", "--host", "0.0.0.0", "--port", "8000"]
