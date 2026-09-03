# Multi-stage production build for PromptDiff
FROM python:3.11-slim as builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY promptdiff ./promptdiff

RUN pip install --no-cache-dir build && python -m build --wheel

# Final minimal production container
FROM python:3.11-slim as runner

WORKDIR /app

# Create non-root user
RUN groupadd -g 1000 promptdiff && \
    useradd -u 1000 -g promptdiff -m -s /bin/bash promptdiff

COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

USER promptdiff

EXPOSE 8765 8000

# Healthcheck checking the studio pricing API
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/pricing')" || exit 1

ENTRYPOINT ["promptdiff"]
CMD ["studio", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
