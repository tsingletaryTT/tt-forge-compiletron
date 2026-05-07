# TT-Forge Compiletron — Self-Contained Build
#
# Builds tt-metal and tt-forge-fe from source, then layers the expedition
# app on top.  Use when you don't have a native Forge install.
#
# WARNING: large build
#   Build time: 2-3 hours
#   Final image: ~25 GB
#   Requires: 32 GB+ RAM, fast internet

FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake ninja-build git wget curl \
    clang-17 clang++-17 lld-17 \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    libhwloc-dev libnuma-dev libboost-all-dev \
    libyaml-cpp-dev libgtest-dev libgmock-dev libcapstone-dev \
    jq patchelf pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-17 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-17 100

WORKDIR /build

# -- tt-metal ----------------------------------------------------------------

ARG TT_METAL_COMMIT=main
RUN git clone https://github.com/tenstorrent/tt-metal.git && \
    cd tt-metal && \
    git checkout ${TT_METAL_COMMIT} && \
    git submodule update --init --recursive

WORKDIR /build/tt-metal
ENV TT_METAL_HOME=/build/tt-metal
ENV ARCH_NAME=blackhole
ENV PYTHONPATH=/build/tt-metal:${PYTHONPATH}

RUN python3 -m pip install --break-system-packages --no-cache-dir -e .

RUN cat > cmake/x86_64-linux-clang-17-libstdcpp-toolchain.cmake << 'EOF'
set(CMAKE_SYSTEM_PROCESSOR "x86_64")
set(CMAKE_C_COMPILER clang-17 CACHE INTERNAL "C compiler")
set(CMAKE_CXX_COMPILER clang++-17 CACHE INTERNAL "C++ compiler")
set(ENABLE_LIBCXX FALSE CACHE INTERNAL "Using clang's libc++")
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

RUN ./build_metal.sh --toolchain-path cmake/x86_64-linux-clang-17-libstdcpp-toolchain.cmake

# -- tt-forge-fe -------------------------------------------------------------

WORKDIR /build

ARG TT_FORGE_COMMIT=main
RUN git clone https://github.com/tenstorrent/tt-forge-fe.git && \
    cd tt-forge-fe && \
    git checkout ${TT_FORGE_COMMIT} && \
    git submodule update --init --recursive

WORKDIR /build/tt-forge-fe
ENV TTFORGE_TOOLCHAIN_DIR=/opt/ttforge-toolchain
ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain
ENV PATH=${TTFORGE_TOOLCHAIN_DIR}/bin:${PATH}

RUN mkdir -p ${TTFORGE_TOOLCHAIN_DIR} ${TTMLIR_TOOLCHAIN_DIR}

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    -r env/core_requirements.txt

RUN bash -c "source env/activate && \
    cmake -B env/build env && cmake --build env/build"

RUN bash -c "source env/activate && \
    cmake -G Ninja -B build -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_C_COMPILER=clang-17 && \
    cmake --build build && cmake --build build -- install_ttforge"

# ============================================================================
# Final image
# ============================================================================

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3-pip \
    libhwloc15 libnuma1 \
    libboost-system1.83.0 libboost-filesystem1.83.0 libboost-thread1.83.0 \
    libyaml-cpp0.8 libncurses6 libopenmpi3t64 openmpi-bin \
    git jq \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

COPY --from=builder /build/tt-metal /tt-metal
COPY --from=builder /build/tt-forge-fe /tt-forge-fe
COPY --from=builder /opt/ttforge-toolchain /opt/ttforge-toolchain

ENV TT_METAL_HOME=/tt-metal
ENV TTFORGE_TOOLCHAIN_DIR=/opt/ttforge-toolchain
ENV TTFORGE_PYTHON_VERSION=python3.12
ENV TTFORGE_VENV_DIR=/opt/ttforge-toolchain/venv
ENV ARCH_NAME=blackhole
ENV PYTHONPATH=/tt-metal:/tt-forge-fe/forge:${PYTHONPATH}
ENV PATH=/opt/ttforge-toolchain/venv/bin:/opt/ttforge-toolchain/bin:${PATH}
ENV LD_LIBRARY_PATH=/tt-forge-fe/third_party/tt-mlir/build/install/lib:/tt-forge-fe/build/lib:${LD_LIBRARY_PATH}

WORKDIR /app

# Reinstall tt_tvm in editable mode with correct runtime paths
RUN /opt/ttforge-toolchain/venv/bin/pip install -e /tt-forge-fe/third_party/tvm/python

# Install compiletron application dependencies
COPY requirements.txt .
RUN /opt/ttforge-toolchain/venv/bin/pip install --no-cache-dir -r requirements.txt || true

# Set up XLA venv for the xla/mixed backends
RUN python3 -m venv /app/xla-venv && \
    /app/xla-venv/bin/pip install --no-cache-dir \
        pjrt-plugin-tt jax==0.7.1 jaxlib==0.7.1 \
        flax==0.8.5 "transformers<5.0" torch \
        --index-url https://pypi.tenstorrent.com/simple/ || true

# Copy application
COPY expedition.py expedition_tui.py .
COPY lib/ ./lib/
COPY tests/ ./tests/
COPY mesh_graph_descriptors/ ./mesh_graph_descriptors/
COPY scripts/ ./scripts/
COPY compiletron.py run_tests.sh README.md INSTALL.md ./

RUN chmod +x run_tests.sh && \
    find scripts -name "*.sh" -type f -exec chmod +x {} \;

# Persistent data volume (bestiary, journals, run artifacts)
RUN mkdir -p /app/data/expeditions /app/data/artifacts /app/data/runs

VOLUME ["/app/data", "/root/.cache/huggingface"]

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["help"]
