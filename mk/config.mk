# The Makefile knobs and their defaults. Every one is `?=`: the command line,
# the environment and local.mk all win over these.

# Backend: 1 = the real GPU model (`--features cuda`, needs the kernel leaf and
# ARTIFACT); 0 = the deterministic CPU mock (ADR 0006).
CUDA ?= 1

# Cargo profile: release | dev (or any custom [profile.*]).
PROFILE ?= release

# 1 = build web/dist before the server (so it is embedded) and pass --ui.
# Anything else passes --no-ui, because the server serves the Playground
# unless told not to -- make says which it means rather than relying on that.
UI ?= 1

# 1 = pass --metrics (ADR 0017): Prometheus text at GET /metrics on its own
# listener (METRICS_BIND, no key, never exposed) and, with UI=1, at
# /ui/metrics on BIND (behind API_KEY when set). Off until #90 measures it on
# the GPU, so gate runs started through make do not carry it by default.
# make metrics scrapes the metrics listener.
METRICS ?= 0
# The metrics listener (--metrics-bind). Empty = the server's 127.0.0.1:9464.
METRICS_BIND ?=

# The model (ADR 0043, spec flash-next/04): one per process, chosen at start.
# MODEL=flash-next serves Qwen3.8-Flash-Next from its own artifact, as
# qwen3.8-flash-next, with the defaults below (each still a knob). MODEL=27b,
# or empty, is Qwen3.8-27B, the default. Any other value is the 27B served
# under that id (--model), as before.
MODEL ?=
# The family MODEL selects, decided here once: every other place reads this.
# The server still checks it against the artifact's own at start, and refuses
# a served id that names the other model.
MODEL_FAMILY := $(if $(filter flash-next,$(MODEL)),flash-next,27b)
ifeq ($(MODEL_FAMILY),flash-next)
  ARTIFACT ?= F:/ai/models/Qwen3.8-Flash-Next-ignis/qwen3_8_flash_next_trellis_a25-v2.ninfer
  # 262,144 tokens per lane under hq-e8-2b, the checkpoint's whole trained
  # envelope (so no YaRN): the user always keeps that context, and speed
  # never takes VRAM from the KV pool (LANES, below, is the one that trades); 8192-token prefill chunks amortize a
  # chunk's expert transfer (spec flash-next/03); no vision. No speculation
  # by default: on the 5090 the MTP head is PCIe-bound (finding 2026-10-07).
  # SPEC=mtp turns it on, its companion container beside the artifact (spec
  # flash-next/07); DRAFT_TOKENS=k forces at most k drafts per lane;
  # DRAFT_ROWS=r is the row budget that cuts k as lanes join (empty: the
  # decode route's 8; 3 drafts at one lane only).
  MAX_CONTEXT ?= 262144
  # The decode lanes (--decode-lanes, 1..8): the sequences decoded at once,
  # each with a whole context in the KV pool. Fewer lanes leave the expert
  # cache more of the VRAM budget.
  LANES ?= 3
  ROPE_SCALING ?= none
  PREFILL_CHUNK ?= 8192
  # The part of the model's time decoding lanes keep while a prompt prefills
  # (--decode-share, percent, 0-99). Empty = the server's: 25 on both models (0 = one decode round per chunk).
  DECODE_SHARE ?=
  SPEC ?=
  DRAFT_TOKENS ?=
  DRAFT_ROWS ?=
  VISION ?=
  # The KV-RAM arena spec flash-next/05 sizes for Flash-Next: the 38 GB of
  # pinned experts leave no room for the 27B's 8G (the host plan refuses it).
  KV_HOST_POOL_BYTES ?= 2G
  # The n-gram rows held in RAM (--ngram-hot-bytes, GitHub #306): a size (4G),
  # or auto for what the host plan leaves after its other lines and the 6 GiB
  # margin -- the whole ~29 GB table when that fits, and then no step reads
  # the NVMe. Empty = the server's 1G.
  NGRAM_HOT_BYTES ?=
  # The budget is free VRAM minus this headroom, so the process plus the
  # desktop ends at 32.6 GB - 4 GB, under spec flash-next/04 AC10's 29 GB,
  # whatever the desktop holds; the expert cache takes the rest of it. A named
  # VRAM_BUDGET wins: the server refuses a budget and a headroom together.
  VRAM_HEADROOM ?= $(if $(VRAM_BUDGET),,4G)
endif

# The .ninfer container (used with CUDA=1 only). See README "Models".
# UNCENSORED=1 takes the huihui-abliterated twin of the default image from the
# same directory instead: the same container with the 70 matrices the
# abliteration changed re-encoded, and no refusals. The server never fetches
# it; docs/user "The uncensored variant" says where it lives. A named ARTIFACT
# wins over both.
UNCENSORED ?=
ifeq ($(UNCENSORED),1)
  ARTIFACT ?= ./models/qwen3_8_27b_nvfp4full-v2-huihui-abliterated.ninfer
else
  ARTIFACT ?= ./models/qwen3_8_27b_nvfp4full-v2.ninfer
endif

# Server settings. Empty = the server's own default.
BIND ?= 127.0.0.1:8000
LOG_LEVEL ?=
LOG_FORMAT ?=
# The key /v1 requires (--api-key). Empty = no key; auto = the server
# generates one and prints it when ready: make dev-ui API_KEY=auto
API_KEY ?=
# Expose the server beyond BIND (--expose, ADR 0028). Empty = not exposed;
# cloudflare-quick = a public https://*.trycloudflare.com URL, printed when
# ready. An exposed server always requires a key (auto when API_KEY is empty).
EXPOSE ?=
# Where system and developer messages go before the chat template (#209).
# merge = a leading run of system messages (qwen-code's agent prompt + hook
# line) joins the system prompt; strict = 400 for any system message not first.
SYSTEM_MESSAGE_POLICY ?= merge
# inplace (server default), into-system, after-system, one-after-system, reject.
# Pinned rather than left to the default: the Playground's date and time tool,
# set to update every prompt, sends each turn's moment as a developer message
# after the history, so the prompt's head stays byte-identical and a long
# conversation keeps its prefix. into-system and after-system hoist that message
# into the head instead, and every turn then prefills from scratch, silently.
DEVELOPER_MESSAGE_POLICY ?= inplace

# The GPU engine configuration (CUDA=1 only; the CPU mock gets none of it).
# Defaults: a 524288-token context -- twice the checkpoint's trained 262,144
# positions, which is why ROPE_SCALING below defaults to yarn:2 -- then
# hq-e8-2b KV, DFlash2 speculation with a 7-token draft window. The G5 gate
# legs (docs/specs/runtime/05) ran at MAX_CONTEXT=262144 ROPE_SCALING=none.
# Empty = leave the flag off (the server's default); SPEC= turns speculation
# off.
MAX_CONTEXT ?= 524288
KV_FORMAT ?= hq-e8-2b
PREFILL_CHUNK ?= 1024
DECODE_SHARE ?=
REQUEST_TIMEOUT ?= 1800
SPEC ?= dflash2
DRAFT_TOKENS ?= 7
# The drafter's proposal head (--draft-head): empty = the server's default
# (full), shortlist = the artifact's Q4 head over the most frequent tokens.
DRAFT_HEAD ?=
KV_POOL_BYTES ?=
# The VRAM budget (GitHub #210, ADR 0030). Empty = the server's default: the
# memory free at start minus a 1G headroom. VRAM_HEADROOM derives it with
# another headroom; VRAM_BUDGET names it (the whole process, weights
# included) and cannot be combined with VRAM_HEADROOM. ALLOW_VRAM_OVERSUBSCRIPTION=1
# starts an explicit budget above free memory with a warning.
VRAM_HEADROOM ?=
VRAM_BUDGET ?=
ALLOW_VRAM_OVERSUBSCRIPTION ?=
# Retained slots (GitHub #215, #281, ADR 0030): the images of retained prompt
# checkpoints and shared prefixes, reserved at load. RETAINED_DEVICE
# (--retained-device) keeps them in VRAM, handed out first; RETAINED_HOST
# (--retained-host) in one pinned host block, a PCIe copy per capture and
# per claim. Empty = the server's defaults: none in VRAM, two per decode lane
# on the host. A card with VRAM to spare: RETAINED_DEVICE=16 RETAINED_HOST=0.
RETAINED_DEVICE ?=
RETAINED_HOST ?=
# Vision (--vision, GitHub #179): a load that takes image parts. Without it
# every `image_url` part is refused with `vision_disabled`, whatever the
# artifact holds. VISION_MAX_TOKENS caps one request's vision tokens
# (--vision-max-tokens); empty = the server's own envelope.
VISION ?=
VISION_MAX_TOKENS ?=
# RoPE scaling (--rope-scaling, GitHub #227): the text rotary table. `none`
# (or empty) is the linear table the checkpoint was trained with, correct
# through 262,144 positions; `yarn:F` rescales that envelope by F, which is
# what a MAX_CONTEXT past it needs to mean anything. The full spelling is
# `yarn:F[,t=<c>][,bf=<n>][,bs=<n>]`. Defaults to yarn:2, the envelope
# MAX_CONTEXT=524288 needs; it rescales every request's table, short ones
# included, so MAX_CONTEXT=262144 wants ROPE_SCALING=none beside it.
ROPE_SCALING ?= yarn:2
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
