# TT-Forge Compiletron - Docker Container
# Full reference implementation with complete Forge dependencies
#
# This image includes ALL dependencies needed for Forge compilation.
# Image size: ~10GB (includes tensorflow, jax, pytorch, etc.)
# Build time: ~15-20 minutes
#
# NOTE: This is a reference implementation. Users will need to:
# 1. Mount their tt-metal installation
# 2. Mount their tt-forge-fe installation
# 3. Have Tenstorrent hardware available

FROM ubuntu:24.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libhwloc-dev \
    libnuma-dev \
    libboost-all-dev \
    jq \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Create app directory
WORKDIR /app

# Copy requirements first (for Docker cache efficiency)
COPY requirements.txt .
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt

# Copy application files
COPY lib/ ./lib/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY tests/ ./tests/
COPY compiletron.py .
COPY setup.sh .
COPY run_tests.sh .
COPY README.md .
COPY IMPLEMENTATION_COMPLETE.md .
COPY TESTS_ADDED.md .

# Make scripts executable
RUN chmod +x setup.sh run_tests.sh scripts/*.sh

# Create directories for mounted volumes
RUN mkdir -p /cache /results /models /tt-metal /tt-forge-fe

# Environment variables for tt-metal integration
ENV TT_METAL_HOME=/tt-metal
ENV PYTHONPATH=/tt-metal:${PYTHONPATH}
ENV PATH=/tt-forge-fe/env/bin:${PATH}

# Set up entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["help"]
