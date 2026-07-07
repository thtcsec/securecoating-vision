FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV and builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and increase timeout for large packages
RUN pip install --upgrade pip

# Install Python requirements (CPU-only for Docker, keeps image small)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements-docker.txt

# Copy source tree
COPY . .

# Create necessary directories
RUN mkdir -p outputs logs data

# Expose ports: FastAPI (8000), Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default healthcheck against FastAPI
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
