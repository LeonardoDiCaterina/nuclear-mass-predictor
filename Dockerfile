# Use a slim Python 3.10 image as the base
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim AS runtime

# Install build essentials for compiling some Python C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Upgrade pip and build tools
RUN pip install --no-cache-dir -U pip setuptools wheel

# Copy pyproject.toml first to leverage Docker layer caching for dependencies
COPY pyproject.toml .

# Copy source code — needed before install because the package version
# is read dynamically from src/nuclear_mass_predictor/__init__.py
COPY src/ ./src/

# Install project dependencies
RUN pip install --no-cache-dir .

# Copy the rest of the project configuration
COPY conf/ ./conf/

# Create a non-root user for security best practices
RUN addgroup --system kedro_group && adduser --system --group kedro_user
RUN chown -R kedro_user:kedro_group /app

# Switch to the non-root user
USER kedro_user

# The default command runs the Kedro pipeline
CMD ["kedro", "run"]
