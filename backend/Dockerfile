# syntax=docker/dockerfile:1.7
# ---------- Stage 1: Builder ----------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY README.md ./README.md

RUN pip install --no-cache-dir --prefix=/install \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu
ENV PYTHONPATH="/install/lib/python3.11/site-packages"
COPY app ./app
RUN pip install --no-cache-dir --prefix=/install .

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:${PATH}" \
    PYTHONPATH="/install/lib/python3.11/site-packages"

RUN groupadd --system --gid 1001 scamguard \
    && useradd --system --uid 1001 --gid scamguard --no-create-home scamguard

WORKDIR /app

COPY --from=builder /install /install
COPY app ./app
COPY models ./models
COPY pyproject.toml ./
COPY alembic ./alembic
COPY alembic.ini ./
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER scamguard

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]