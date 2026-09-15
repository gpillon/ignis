# The Makefile knobs and their defaults. Every one is `?=`: the command line,
# the environment and local.mk all win over these.

# Backend: 1 = the real GPU model (`--features cuda`, needs the kernel leaf and
# ARTIFACT); 0 = the deterministic CPU mock (ADR 0006).
CUDA ?= 1

# Cargo profile: release | dev (or any custom [profile.*]).
PROFILE ?= release

# 1 = build web/dist before the server (so it is embedded) and pass --ui.
UI ?= 1

# 1 = pass --metrics: Prometheus text at GET /metrics (ADR 0017), behind the
# API key when one is set. Off until #90 measures it on the GPU, so gate runs
# started through make do not carry it by default. make metrics scrapes it.
METRICS ?= 0

# The .ninfer container (used with CUDA=1 only). See README "Models".
ARTIFACT ?= ./models/qwen3_8_27b_nvfp4full-v2.ninfer

# Server settings. Empty = the server's own default.
BIND ?= 127.0.0.1:8000
MODEL ?=
LOG_LEVEL ?=
LOG_FORMAT ?=
# The key /v1 requires (--api-key). Empty = no key; auto = the server
# generates one and prints it when ready: make dev-ui API_KEY=auto
API_KEY ?=
# Expose the server beyond BIND (--expose, ADR 0028). Empty = not exposed;
# cloudflare-quick = a public https://*.trycloudflare.com URL, printed when
# ready. An exposed server always requires a key (auto when API_KEY is empty).
EXPOSE ?=

# The GPU engine configuration (CUDA=1 only; the CPU mock gets none of it).
# Defaults are the G5 gate legs (.scratch/runtime/specs/05, the g5-run driver):
# the full 262144-token envelope, hq-e8-2b KV, DFlash2 speculation with a
# 7-token draft window. Empty = leave the flag off (the server's default);
# SPEC= turns speculation off.
MAX_CONTEXT ?= 262144
KV_FORMAT ?= hq-e8-2b
PREFILL_CHUNK ?= 1024
REQUEST_TIMEOUT ?= 1800
SPEC ?= dflash2
DRAFT_TOKENS ?= 7
KV_POOL_BYTES ?=
# The KV-RAM host tier's budget (--kv-host-pool-bytes, P4-07, GitHub #125):
# 0 disables the host tier entirely (no evict-to-RAM overflow path).
KV_HOST_POOL_BYTES ?= 8G

# Extra ignis-server flags, verbatim: ARGS='--kv-host-pool-bytes 0'
ARGS ?=

# 1 = run the GPU guard before starting a CUDA server.
GPU_CHECK ?= 1
GPU_THRESHOLD_MIB ?= 8192
# Forwarded to the GPU profile script: -SkipKernelBuild, -SkipCargoTests, ...
GPU_PROFILE_ARGS ?=

# Seconds `make start` waits for /v1/models to answer (a cold artifact load
# takes a while).
READY_TIMEOUT ?= 600

# Where `make start` / `run-ui` keep the pid file and server logs (gitignored).
RUNTIME_DIR ?= .scratch/serve

CARGO ?= cargo
NPM ?= npm
CARGO_TARGET_DIR ?= target
