# Multi-stage production build for promptdiff
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY promptdiff ./promptdiff

RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# Production runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Non-root service user for production security
RUN groupadd -g 1001 promptdiff && \
    useradd -u 1001 -g promptdiff -s /bin/bash -m promptdiff

COPY --from=builder --chown=promptdiff:promptdiff /opt/venv /opt/venv

RUN mkdir -p /app/prompts /app/.promptdiff && \
    chown -R promptdiff:promptdiff /app

USER promptdiff

EXPOSE 8000 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || curl -f http://localhost:8765/api/metrics || exit 1

ENTRYPOINT ["promptdiff"]
CMD ["--help"]
