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
# kernel/NOTICE is the provenance of the vendored reference ops (ADR 0010):
# it travels with every binary that carries them.
COPY README.md /
COPY kernel/NOTICE /NOTICE

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

# The server's own default is 127.0.0.1, which nothing outside the container
# can reach. Everything else stays at the server's default and is set through
# the IGNIS_* environment (crates/server/src/config.rs) or flags after the
# image name; `--ui` is a bare switch with no environment variable, which is
# why it is the CMD.
ENV IGNIS_BIND=0.0.0.0:8000
# The model is mounted, never baked in: a .ninfer container is tens of GB.
ENV IGNIS_ARTIFACT=/models/model.ninfer

# The OpenAI-compatible API + the Playground, and the Prometheus listener
# (ADR 0017) for a server started with --metrics --metrics-bind 0.0.0.0:9464.
EXPOSE 8000 9464

USER ignis
WORKDIR /home/ignis

ENTRYPOINT ["/usr/local/bin/ignis-server"]
CMD ["--ui"]

# No org.opencontainers.image.licenses: the repository declares no license of
# its own yet. kernel/NOTICE records the vendored ops' provenance and terms,
# and ships inside the image at /NOTICE.
LABEL org.opencontainers.image.title="ignis" \
      org.opencontainers.image.description="ignis: OpenAI-compatible inference server for Qwen3.8-27B NVFP4 on Blackwell (SM120a), with the Playground UI" \
      org.opencontainers.image.source="https://github.com/gpillon/ignis"
