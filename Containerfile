# ignis as a linux/amd64 container image: the GPU server (`--features cuda`,
# the real kernel leaf) with the Playground embedded.
#
#   podman build -t ignis:dev -f Containerfile .
#   podman run --rm --device nvidia.com/gpu=all -p 8000:8000 \
#       -v /path/to/models:/models:ro -e IGNIS_ARTIFACT=/models/<name>.ninfer \
#       ignis:dev
#
# (docker: `--gpus all` in place of `--device`.) The image carries the CUDA
# runtime, never a driver: the host's NVIDIA driver is injected by the
# container runtime, which is also where libcuda.so.1 and libnvidia-ml.so.1
# come from at run time -- the build links the toolkit's stubs for them
# (crates/artifact/build.rs).
#
# The kernel leaf is compiled for SM120a (RTX 5090 / Blackwell consumer),
# the one architecture kernel/CMakeLists.txt targets. The image will not run
# on another card.

# --- the Playground frontend (ADR 0026) ------------------------------------
#
# Built first and on its own: crates/server/build.rs embeds whatever is in
# web/dist at compile time, and this stage is the only thing that changes when
# a .tsx does.
FROM docker.io/library/node:24-bookworm-slim AS web

WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- the Rust workspace + the CUDA kernel leaf ------------------------------
FROM docker.io/nvidia/cuda:13.1.1-devel-ubuntu24.04 AS build

# cmake 3.28 is what kernel/CMakeLists.txt asks for and what noble ships.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build pkg-config curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG RUST_VERSION=1.98.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain "${RUST_VERSION}" --profile minimal \
        --target x86_64-unknown-linux-gnu
ENV PATH=/root/.cargo/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /src
COPY . .
COPY --from=web /src/web/dist ./web/dist

# nvcc peaks well above 2 GB of RSS on the W4A4/TMA translation units, so a
# builder with little memory per core needs a narrower width than its core
# count (kernel/build.sh reads this).
ARG KERNEL_BUILD_JOBS=
ENV IGNIS_KERNEL_BUILD_JOBS=${KERNEL_BUILD_JOBS}

# --target explicitly: .cargo/config.toml pins the MSVC triple for every host
# (cargo has no host condition for build.target -- GitHub #167).
RUN cargo build --release --locked --target x86_64-unknown-linux-gnu \
        --workspace --features ignis-server/cuda \
    && strip target/x86_64-unknown-linux-gnu/release/ignis-server \
             target/x86_64-unknown-linux-gnu/release/ignis-bench \
             target/x86_64-unknown-linux-gnu/release/ignis-artifact-inspect

# --- the release payload ----------------------------------------------------
#
# A stage with nothing but the shipped files, so CI can export the exact same
# binaries the image runs:
#   docker buildx build --target artifacts --output type=local,dest=out .
FROM scratch AS artifacts
COPY --from=build /src/target/x86_64-unknown-linux-gnu/release/ignis-server /
COPY --from=build /src/target/x86_64-unknown-linux-gnu/release/ignis-bench /
COPY --from=build /src/target/x86_64-unknown-linux-gnu/release/ignis-artifact-inspect /
# Apache-2.0 asks that a redistribution carry LICENSE and NOTICE; kernel/NOTICE
# is the provenance of the vendored reference ops (ADR 0010), and travels with
# every binary that carries them.
COPY LICENSE NOTICE README.md /
COPY kernel/NOTICE /NOTICE-kernel

# --- the runtime image ------------------------------------------------------
FROM docker.io/nvidia/cuda:13.1.1-base-ubuntu24.04 AS runtime

# ca-certificates: media acquisition (GitHub #179) fetches image URLs over
# https through rustls, which reads the system trust store.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 ignis

COPY --from=build /src/target/x86_64-unknown-linux-gnu/release/ignis-server /usr/local/bin/
COPY --from=build /src/target/x86_64-unknown-linux-gnu/release/ignis-bench /usr/local/bin/
COPY --from=build /src/target/x86_64-unknown-linux-gnu/release/ignis-artifact-inspect /usr/local/bin/

# Apache-2.0 travels with the binaries (LICENSE and the root NOTICE), and so
# does the vendored subtree's own provenance.
COPY LICENSE NOTICE /usr/share/doc/ignis/
COPY kernel/NOTICE /usr/share/doc/ignis/NOTICE-kernel

# The server's own default is 127.0.0.1, which nothing outside the container
# can reach. Everything else stays at the server's default and is set through
# the IGNIS_* environment (crates/server/src/config.rs) or flags after the
# image name -- the Playground included, which is served unless IGNIS_UI=false
# or `--no-ui` says otherwise.
ENV IGNIS_BIND=0.0.0.0:8000

# IGNIS_ARTIFACT is deliberately NOT defaulted. A .ninfer container is tens of
# GB, so it is mounted rather than baked in -- and a default pointing at a path
# that is usually absent would turn "no model given" into a load failure.
# Left unset, the image starts on the deterministic CPU mock (ADR 0006), which
# is what makes `docker run <image>` a useful smoke test on its own; set it to
# the mounted artifact to serve the real model on the GPU.

# The OpenAI-compatible API + the Playground, and the Prometheus listener
# (ADR 0017) for a server started with --metrics --metrics-bind 0.0.0.0:9464.
EXPOSE 8000 9464

USER ignis
WORKDIR /home/ignis

ENTRYPOINT ["/usr/local/bin/ignis-server"]
# No CMD: the server's own defaults are the image's, the Playground among
# them. A CMD here would also be replaced wholesale by the first argument
# anyone passes after the image name, which is how `--ui` used to disappear
# the moment someone added a flag.

LABEL org.opencontainers.image.title="ignis" \
      org.opencontainers.image.description="ignis: OpenAI-compatible inference server for Qwen3.8-27B NVFP4 on Blackwell (SM120a), with the Playground UI" \
      org.opencontainers.image.source="https://github.com/gpillon/ignis" \
      org.opencontainers.image.licenses="Apache-2.0"
