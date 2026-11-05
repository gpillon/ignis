# ignis as a linux/amd64 container image: the GPU server (`--features cuda`,
# the real kernel leaf) with the Playground embedded.
#
#   podman build -t ignis:dev -f Containerfile .
#   podman run --rm --device nvidia.com/gpu=all -p 8000:8000 \
#       -v /path/to/models:/models:ro -e IGNIS_MODEL_ARTIFACT=/models/<name>.ninfer \
#       ignis:dev
#
# (docker: `--gpus all` in place of `--device`.) The image carries the CUDA
# runtime, never a driver: the host's NVIDIA driver is injected by the
# container runtime, which is also where libcuda.so.1 and libnvidia-ml.so.1
# come from at run time. The binaries link neither (crates/artifact/build.rs,
# GitHub #329) -- the CUDA runtime loads the driver, NVML is opened when a
# load reads it -- so the image also runs where there is no driver at all:
# `ignis-server model download` on a CPU node, or the CPU mock below (with
# `--download-enabled false`, or the start fetches the configured model
# first). A GPU load there is refused naming the missing driver.
#
# The kernel leaf is compiled for 120a (RTX 50 and RTX PRO, Blackwell),
# kernel/CMakeLists.txt's default; CUDA_ARCHITECTURES below adds others, and
# the release image is built with `86;120a`, which adds RTX 30 and RTX 40
# (Ampere, Ada). On a card the build does not cover the server refuses to
# start, naming the card.

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

# The CUDA architectures the kernel leaf is compiled for (GitHub #456), which
# kernel/CMakeLists.txt reads: empty is its default, 120a alone, and
# `--build-arg CUDA_ARCHITECTURES='86;120a'` adds the sm_86 machine code RTX 30
# and RTX 40 cards run, the list the release workflow passes (#482,
# docs/user/contributing/building.md). Each architecture added is another
# device compile of nearly every translation unit. The check after the build
# reads the same variable.
ARG CUDA_ARCHITECTURES=
ENV IGNIS_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}

# What the check requires of the binaries whatever list they were built for:
# the release workflow passes `86;120a`, written apart from
# CUDA_ARCHITECTURES (#482). Empty, the default, it requires nothing beyond
# that list, so a container build from a checkout checks its own.
ARG CUDA_REQUIRED_ARCHITECTURES=

# --target explicitly: .cargo/config.toml pins the MSVC triple for every host
# (cargo has no host condition for build.target -- GitHub #167).
#
# `--features cuda`, never `ignis-server/cuda`: ignis-server has a
# dev-dependency on itself (its integration tests' `fake-speech`), and cargo
# reads `ignis-server/cuda` as a feature of that dependency, which a build of
# the binaries never compiles. The server then came out without CUDA, with no
# error, in v0.9.0 and v0.9.1 (GitHub #408). The check after the build reads
# the binary itself and refuses one that would fall back to the CPU mock, or
# that carries no machine code for an entry of IGNIS_CUDA_ARCHITECTURES or of
# CUDA_REQUIRED_ARCHITECTURES (#482), or whose 8.6 code breaks the full-rate rule
# of mk/full-rate-kernels.txt (cuobjdump comes with this devel image, #512), on
# each of the three binaries the image
# and the tarball ship.
RUN cargo build --release --locked --target x86_64-unknown-linux-gnu \
        --workspace --features cuda \
    && bash mk/cuda-build-check.sh --require "$CUDA_REQUIRED_ARCHITECTURES" target/x86_64-unknown-linux-gnu/release/ignis-server \
             target/x86_64-unknown-linux-gnu/release/ignis-bench \
             target/x86_64-unknown-linux-gnu/release/ignis-artifact-inspect \
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
# image name -- the Playground included, which is served unless IGNIS_SERVER_UI=false
# or `--server-ui false` says otherwise.
ENV IGNIS_SERVER_BIND=0.0.0.0:8000

# IGNIS_MODEL_ARTIFACT is deliberately NOT defaulted. A .ninfer container is tens of
# GB, so it is mounted rather than baked in -- and a default pointing at a path
# that is usually absent would turn "no model given" into a load failure.
# Left unset, the server downloads its default model first (`--download-enabled`
# is on); with `--download-enabled false` it starts on the deterministic CPU
# mock (ADR 0006), which needs no GPU and no driver, so that is the image's
# smoke test. Set IGNIS_MODEL_ARTIFACT to the mounted artifact to serve the real
# model on the GPU.

# The OpenAI-compatible API + the Playground, and the Prometheus listener
# (ADR 0017). Metrics are on by default, but the listener stays on the
# container's own loopback (127.0.0.1:9464): it asks for no key, so it is never
# published unasked. The Playground reads them at /ui/metrics on 8000, behind
# the API key when one is set. To scrape from outside, name it:
# -e IGNIS_SERVER_METRICS_BIND=0.0.0.0:9464 -p 9464:9464.
EXPOSE 8000 9464

USER ignis
WORKDIR /home/ignis

ENTRYPOINT ["/usr/local/bin/ignis-server"]
# No CMD: the server's own defaults are the image's, the Playground among
# them. A CMD here would also be replaced wholesale by the first argument
# anyone passes after the image name, which is how `--server-ui` used to disappear
# the moment someone added a flag.

# The release workflow's metadata step replaces the description on the image
# it publishes, built for `86;120a`; this one is a local build's.
LABEL org.opencontainers.image.title="ignis" \
      org.opencontainers.image.description="ignis: OpenAI-compatible inference server for Qwen3.8 with the Playground UI, on NVIDIA RTX 50 and RTX PRO (Blackwell) cards, and on RTX 30 and RTX 40 (Ampere, Ada) built for 86;120a as the release image is" \
      org.opencontainers.image.source="https://github.com/gpillon/ignis" \
      org.opencontainers.image.licenses="Apache-2.0"
