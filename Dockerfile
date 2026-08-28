FROM python:3.11.16-slim-trixie AS builder

WORKDIR /app

# Build wheels separately so compilers are not shipped in the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker-lock.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/wheels --timeout=300 -r requirements-docker-lock.txt

FROM python:3.11.16-slim-trixie AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openssl libssl3t64 libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 securecoating \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin securecoating

RUN --mount=from=builder,source=/wheels,target=/wheels \
    pip install --no-index --find-links=/wheels /wheels/*

# Copy source tree
COPY --chown=securecoating:securecoating . .

# Create necessary directories
RUN mkdir -p outputs logs data && chown -R securecoating:securecoating /app

USER 10001:10001

# Expose ports: FastAPI (8000), Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default process-liveness check. Safety/readiness is exposed separately at /health.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/live', timeout=5).read()"]
