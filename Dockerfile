# TT-Forge Compiletron - Full Self-Contained Build
# This image builds tt-metal and tt-forge-fe from source
#
# WARNING: This is a LARGE build:
# - Build time: 2-3 hours
# - Final image: ~30GB
# - Requires: 32GB+ RAM, good CPU, fast internet
#
# Use this when you need completely self-contained compilation
# without any host mounts.

FROM ubuntu:24.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    # Compilers (Clang-17 is officially tested for tt-forge)
    clang-17 \
    clang++-17 \
    lld-17 \
    # Python
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    # TT-Metal dependencies
    libhwloc-dev \
    libnuma-dev \
    libboost-all-dev \
    libyaml-cpp-dev \
    libgtest-dev \
    libgmock-dev \
    libcapstone-dev \
    # Additional tools
    jq \
    tmux \
    patchelf \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set up Python and compilers
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-17 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-17 100

WORKDIR /build

# ==============================================================================
# BUILD TT-METAL
# ==============================================================================

# Clone tt-metal (pin to specific commit for reproducibility)
ARG TT_METAL_COMMIT=main
RUN git clone https://github.com/tenstorrent/tt-metal.git && \
    cd tt-metal && \
    git checkout ${TT_METAL_COMMIT} && \
    git submodule update --init --recursive

# Build tt-metal
WORKDIR /build/tt-metal
ENV TT_METAL_HOME=/build/tt-metal
ENV ARCH_NAME=blackhole
ENV PYTHONPATH=/build/tt-metal:${PYTHONPATH}

# Install Python dependencies from pyproject.toml
RUN python3 -m pip install --break-system-packages --no-cache-dir -e .

# Create Clang-17 toolchain file (doesn't exist in this commit)
RUN cat > cmake/x86_64-linux-clang-17-libstdcpp-toolchain.cmake << 'EOF'
set(CMAKE_SYSTEM_PROCESSOR "x86_64")

set(CMAKE_C_COMPILER clang-17 CACHE INTERNAL "C compiler")

set(CMAKE_CXX_COMPILER clang++-17 CACHE INTERNAL "C++ compiler")

# Use for configure time
set(ENABLE_LIBCXX FALSE CACHE INTERNAL "Using clang's libc++")

# Our build is super slow; put a band-aid on it by choosing a linker that can cope better.
# We really need to fix out code, though.
find_program(MOLD ld.mold)
if(MOLD)
    set(CMAKE_LINKER_TYPE MOLD)
else()
    find_program(LLD ld.lld-17)
    if(LLD)
        set(CMAKE_LINKER_TYPE LLD)
    endif()
endif()
EOF

# Build metal (this takes ~45-60 minutes)
RUN ./build_metal.sh --toolchain-path cmake/x86_64-linux-clang-17-libstdcpp-toolchain.cmake

# ==============================================================================
# BUILD TT-FORGE-FE
# ==============================================================================

WORKDIR /build

# Clone tt-forge-fe (pin to specific commit)
# Using 44529413 from 2026-03-01 (more recent, may have debug dialect fixes)
ARG TT_FORGE_COMMIT=44529413
RUN git clone https://github.com/tenstorrent/tt-forge-fe.git && \
    cd tt-forge-fe && \
    git checkout ${TT_FORGE_COMMIT} && \
    git submodule update --init --recursive

# Build forge (this takes ~45-60 minutes)
WORKDIR /build/tt-forge-fe
ENV TTFORGE_TOOLCHAIN_DIR=/opt/ttforge-toolchain
ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain
ENV PATH=${TTFORGE_TOOLCHAIN_DIR}/bin:${PATH}

# Create toolchain directories
RUN mkdir -p ${TTFORGE_TOOLCHAIN_DIR} ${TTMLIR_TOOLCHAIN_DIR}

# Install forge core dependencies
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    -r env/core_requirements.txt

# Build forge environment (toolchain)
RUN bash -c "source env/activate && \
    cmake -B env/build env && \
    cmake --build env/build"

# Build forge
RUN bash -c "source env/activate && \
    cmake -G Ninja -B build -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_C_COMPILER=clang-17 && \
    cmake --build build && \
    cmake --build build -- install_ttforge"

# ==============================================================================
# FINAL IMAGE
# ==============================================================================

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies (smaller subset than build deps)
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libhwloc15 \
    libnuma1 \
    libboost-system1.83.0 \
    libboost-filesystem1.83.0 \
    libboost-thread1.83.0 \
    libyaml-cpp0.8 \
    libncurses6 \
    libopenmpi3t64 \
    openmpi-bin \
    git \
    jq \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Copy built artifacts from builder
COPY --from=builder /build/tt-metal /tt-metal
COPY --from=builder /build/tt-forge-fe /tt-forge-fe
COPY --from=builder /opt/ttforge-toolchain /opt/ttforge-toolchain

# Set environment variables
ENV TT_METAL_HOME=/tt-metal
ENV TTFORGE_TOOLCHAIN_DIR=/opt/ttforge-toolchain
ENV TTFORGE_PYTHON_VERSION=python3.12
ENV TTFORGE_VENV_DIR=/opt/ttforge-toolchain/venv
ENV ARCH_NAME=blackhole
ENV PYTHONPATH=/tt-metal:/tt-forge-fe/forge:${PYTHONPATH}
# Put venv's bin first in PATH so python3 uses venv's python
ENV PATH=/opt/ttforge-toolchain/venv/bin:/opt/ttforge-toolchain/bin:${PATH}
ENV LD_LIBRARY_PATH=/tt-forge-fe/third_party/tt-mlir/build/install/lib:/tt-forge-fe/build/lib:${LD_LIBRARY_PATH}

# Create app directory
WORKDIR /app

# Reinstall tt_tvm in editable mode with correct runtime paths
RUN /opt/ttforge-toolchain/venv/bin/pip install -e /tt-forge-fe/third_party/tvm/python

# Copy compiletron application
COPY requirements.txt .
# Install requirements using toolchain's venv to avoid conflicts
RUN /opt/ttforge-toolchain/venv/bin/pip install --no-cache-dir -r requirements.txt || true

COPY lib/ ./lib/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY tests/ ./tests/
COPY mesh_graph_descriptors/ ./mesh_graph_descriptors/
COPY compiletron.py .
COPY setup.sh .
COPY run_tests.sh .
COPY README.md .
COPY *.md .

# Make scripts executable
RUN chmod +x setup.sh run_tests.sh 2>/dev/null || true && \
    find scripts -name "*.sh" -type f -exec chmod +x {} \;

# Create directories for volumes
RUN mkdir -p /cache /results /models

# Set up entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["help"]
