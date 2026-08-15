# Multi-stage Python 3.10 image for Nuclear Mass Predictor
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim AS runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade build tools
RUN pip install --no-cache-dir -U pip setuptools wheel

# Copy pyproject.toml first for Docker layer caching
COPY pyproject.toml .

# Copy source code
COPY src/ ./src/

# Install editable project with API dependencies
RUN pip install --no-cache-dir .

# Copy configuration, tests, and data folders
COPY conf/ ./conf/
COPY tests/ ./tests/
COPY README.md .

# Create non-root user for security best practices
RUN addgroup --system kedro_group && adduser --system --group kedro_user
RUN chown -R kedro_user:kedro_group /app

USER kedro_user

EXPOSE 8000

# Default command starts the FastAPI model server
CMD ["nuclear-mass-predictor-api"]
