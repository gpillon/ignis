# The Makefile knobs and their defaults. Every one is `?=`: the command line,
# the environment and local.mk all win over these.

# Backend: 1 = the real GPU model (`--features cuda`, needs the kernel leaf and
# ARTIFACT); 0 = the deterministic CPU mock (ADR 0006).
CUDA ?= 1

# Cargo profile: release | dev (or any custom [profile.*]).
PROFILE ?= release

# 1 = build web/dist before the server (so it is embedded) and pass --server-ui.
# Anything else passes --server-ui false, because the server serves the Playground
# unless told not to -- make says which it means rather than relying on that.
UI ?= 1

# Prometheus metrics (ADR 0017): text at GET /metrics on their own listener
# (METRICS_BIND, no key, never exposed) and, with UI=1, at /ui/metrics on BIND
# (behind API_KEY when set). The server's own default since 2026-10-15 (the
# owner's call; #90 proved they cost the inference path nothing), so 1 passes
# nothing; METRICS=0 passes --server-metrics false, as a gate run that must
# not carry them does. A second server on the host finds the default listener
# taken and starts without it, warning; a METRICS_BIND that is taken refuses
# the start. make metrics scrapes the metrics listener.
METRICS ?= 1
# The metrics listener (--server-metrics-bind). Empty = the server's 127.0.0.1:9464.
METRICS_BIND ?=

# The model (ADR 0043, spec flash-next/04): one per process, chosen at start.
# MODEL=flash-next serves Qwen3.8-Flash-Next from its own artifact, as
# qwen3.8-flash-next, with the defaults below (each still a knob). MODEL=27b,
# or empty, is Qwen3.8-27B, the default. Any other value is the 27B served
# under that id (--model-id), as before.
MODEL ?=
# The family MODEL selects, decided here once: every other place reads this.
# The server still checks it against the artifact's own at start, and refuses
# a served id that names the other model.
MODEL_FAMILY := $(if $(filter flash-next,$(MODEL)),flash-next,$(if $(filter image,$(MODEL)),image,27b))
ifeq ($(MODEL_FAMILY),flash-next)
  ARTIFACT ?= F:/ai/models/Qwen3.8-Flash-Next-ignis/qwen3_8_flash_next_trellis_a25-v2.ninfer
  # 262,144 tokens per lane under hq-e8-2b, the checkpoint's whole trained
  # envelope (so no YaRN): the user always keeps that context, and speed
  # never takes VRAM from the KV pool (LANES=1, below, is the one that trades); 8192-token prefill chunks amortize a
  # chunk's expert transfer (spec flash-next/03); no vision. No speculation
  # by default: on the 5090 the MTP head is PCIe-bound (finding 2026-10-07).
  # SPEC=mtp turns it on, its companion container beside the artifact (spec
  # flash-next/07); DRAFT_TOKENS=k forces at most k drafts per lane;
  # DRAFT_ROWS=r is the row budget that cuts k as lanes join (empty: the
  # decode route's 8; 3 drafts at one lane only).
  MAX_CONTEXT ?= 262144
  # The decode lanes (--spec-decode-lanes, 1..8): the sequences decoded at once.
  # They share the KV pool (ADR 0045), reserved first: min(524,288 tokens,
  # LANES x MAX_CONTEXT), never below one MAX_CONTEXT and a page per retained
  # slot; the expert cache takes the rest of the budget. At 262,144 tokens one
  # lane's pool is one context and two lanes or more share 524,288, so one
  # lane leaves the cache ~1 GiB more.
  LANES ?= 3
  ROPE_SCALING ?=
  PREFILL_CHUNK ?= 8192
  # No solo width of its own (GitHub #489): its `auto` is the prefill chunk.
  PREFILL_CHUNK_SOLO ?=
  # The part of the model's time decoding lanes keep while a prompt prefills
  # (--model-decode-share, percent, 0-99). Empty = the server's: 25 on both models (0 = one decode round per chunk).
  DECODE_SHARE ?=
  SPEC ?=
  DRAFT_TOKENS ?=
  DRAFT_ROWS ?=
  VISION ?=
  # The n-gram rows held in RAM (--ngram-hot-bytes, GitHub #306): a size (4G),
  # or auto for what the host plan leaves after its other lines and its
  # margin (HOST_MARGIN_BYTES) -- the whole ~29 GB table when that fits, and
  # then no step reads the NVMe. Empty = the server's 1G.
  NGRAM_HOT_BYTES ?=
  # No Flash-Next headroom of its own: the server's 1536M, as on the 27B (the
  # owner's call, 2026-10-09; it was 4G for spec flash-next/04 AC10's 29 GB
  # with the desktop). The expert cache takes what the budget leaves: 1536M
  # measured +5-13% over 4G, while 1G let WDDM page the model out (finding
  # 2026-10-09-vram-headroom-wddm-paging). VRAM_HEADROOM=4G restores the ceiling.
endif

ifeq ($(MODEL_FAMILY),image)
  # MODEL=image serves the image model (spec image-model/05) as
  # ignis-image-2.1-turbo, from its own artifact (`ignis-server model download
  # ignis-image-2.1-turbo` fetches it). It has no context, lanes, prefill
  # chunk, speculation, KV format or vision, so none of those knobs defaults
  # to a value: they are empty, and a flag the server does not take for this
  # family is never passed. The settings every model shares (BIND, API_KEY,
  # REQUEST_TIMEOUT, IDLE_UNLOAD, ...) are unchanged.
  ARTIFACT ?= F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer
  MAX_CONTEXT ?=
  KV_FORMAT ?=
  PREFILL_CHUNK ?=
  PREFILL_CHUNK_SOLO ?=
  DECODE_SHARE ?=
  SPEC ?=
  DRAFT_TOKENS ?=
  DRAFT_ROWS ?=
  ROPE_SCALING ?=
  VISION ?=
  KV_HOST_POOL_BYTES ?=
  # A switch from here to the 27B takes the 27B's arena (--qwen38-reuse-kv-host-pool-bytes,
  # the scoped spelling: the image model takes no general one), as a
  # switch to Flash-Next takes FN_KV_HOST_POOL_BYTES, never the server's 2G.
  IMAGE_27B_KV_HOST_POOL_BYTES ?= 8G
endif

# The .ninfer container (used with CUDA=1 only). See README "Models".
# UNCENSORED=1 takes the huihui-abliterated twin of the default image from the
# same directory instead: the same container with the 70 matrices the
# abliteration changed re-encoded, and no refusals. It is served as
# qwen3.8-27b-abliterated (the Makefile's SERVED_MODEL), its id in the built-in
# catalog: `ignis-server model download qwen3.8-27b-abliterated` fetches it. A
# named ARTIFACT wins over both.
UNCENSORED ?=
ifeq ($(UNCENSORED),1)
  ARTIFACT ?= ./models/qwen3_8_27b_nvfp4full-v2-huihui-abliterated.ninfer
else
  ARTIFACT ?= ./models/qwen3_8_27b_nvfp4full-v2.ninfer
endif

# The decode lanes (--spec-decode-lanes, 1..8) of whichever model starts, in its
# own section. Empty = the model's own: Flash-Next's 3 (set above), the 27B's 8.
# On the 27B each lane holds ~233 MB of state and its share of the round
# buffers, so LANES=2 frees ~1.6 GB for the KV pool (GitHub #336).
LANES ?=

# Server settings. Empty = the server's own default.
BIND ?= 127.0.0.1:8000
LOG_LEVEL ?=
LOG_FORMAT ?=
# The key /v1 requires (--server-api-key). Empty = no key; auto = the server
# generates one and prints it when ready: make dev-ui API_KEY=auto
API_KEY ?=
# Expose the server beyond BIND (--server-expose, ADR 0028). Empty = not exposed;
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
# The max_tokens of a request that sends none (--model-default-max-tokens, ADR
# 0045), its reasoning included, on both models and on the CPU mock. Empty =
# the server's 38912; 0 = none, a request may generate to the end of the
# context. An explicit max_tokens always wins.
DEFAULT_MAX_TOKENS ?=
# Seconds with no request using the model before it is unloaded, the server
# staying up (--model-idle-unload, ADR 0049); the next request loads it again.
# Empty = the server's 300; 0 = never. A measurement (a bench, a gate leg, an
# A/B leg) passes 0: a cold reload mid-run is not what it measures.
IDLE_UNLOAD ?=
# The loop guard (--model-loop-guard, ADR 0052): rewind, log or off. Empty =
# the server's rewind, which takes a looping reasoning back and nudges the
# model. A measurement (a bench, a gate leg, an A/B leg) passes off: the guard
# changes the output wherever it acts, and the reference has no guard.
LOOP_GUARD ?=

# The GPU engine configuration (CUDA=1 only; the CPU mock gets none of it).
# Defaults: the 27B's context left to the server -- the card class's profile
# (`auto`): 917,504 tokens on the 5090, which the load adapts down to the
# largest the free memory holds rather than refusing or dropping lanes (ADR
# 0063 §3; a context named here is explicit and never adapts) -- then
# hq-e8-2b KV, DFlash2 speculation with a 7-token draft window. MAX_CONTEXT=n
# names a context (524288 was make's own until #336). The G5 gate legs
# (docs/specs/runtime/05) ran at MAX_CONTEXT=262144 ROPE_SCALING=none.
# Empty = leave the flag off (the server's default); SPEC=off turns speculation
# off (SPEC= leaves it to the server, and the card class's profile speculates).
# PREFILL_CHUNK is left to the server too: the card class's profile (`auto`):
# 2048 on the 5090 (GitHub #341; free there, vision's encoder sizes the
# arena), 1024 without it. So is PREFILL_CHUNK_SOLO (--model-prefill-chunk-solo,
# GitHub #489), the chunk while no lane decodes: `auto`, 8192 or 4096 on an
# RTX 30 or RTX 40 card where the VRAM plan holds it with nothing adapted, the
# prefill chunk on the 5090.
MAX_CONTEXT ?=
KV_FORMAT ?= hq-e8-2b
PREFILL_CHUNK ?=
PREFILL_CHUNK_SOLO ?=
DECODE_SHARE ?=
REQUEST_TIMEOUT ?=
STREAM_TIMEOUT ?=
SPEC ?= dflash2
DRAFT_TOKENS ?= 7
# The drafter's proposal head (--spec-draft-head): empty = the server's default
# (full), shortlist = the artifact's Q4 head over the most frequent tokens,
# auto = full on every card class (ADR 0063).
DRAFT_HEAD ?=
# The KV pool (--vram-kv-pool-bytes): a byte count (4G) or tokens (512Ktok), on
# both models. Empty = the policy's (ADR 0045): the rest of the VRAM budget
# when every weight is on the device (the 27B), 524,288 tokens when experts
# stream (Flash-Next on a 32 GB card).
KV_POOL_BYTES ?=
# 1 = start Flash-Next even when its expert cache gets less than the 12 GiB
# floor (--vram-allow-expert-cache-below-floor), with a warning: below it decode
# slows sharply. The 27B refuses it.
ALLOW_EXPERT_CACHE_BELOW_FLOOR ?=
# Flash-Next's pinned hot set (--vram-pinned-expert-bytes, GitHub #329 item 12):
# the calibration-hottest experts held in the expert cache outside its LRU, with
# no copy in the pinned host pool, which takes that much less RAM (on the 5090:
# nothing measurable at 1G, ~1% of three-lane decode at 2G). Empty = the
# server's: 2G under the rtx5090 profile (`auto` on a 5090), else 0 (none). The 27B refuses it.
PINNED_EXPERT_BYTES ?=
# The VRAM budget (GitHub #210, ADR 0030). Empty = the server's default: the
# memory free at start minus a 1536M headroom. VRAM_HEADROOM derives it with
# another headroom; VRAM_BUDGET names it (the whole process, weights
# included) and cannot be combined with VRAM_HEADROOM. ALLOW_VRAM_OVERSUBSCRIPTION=1
# starts an explicit budget above free memory with a warning.
VRAM_HEADROOM ?=
VRAM_BUDGET ?=
ALLOW_VRAM_OVERSUBSCRIPTION ?=
# Retained slots (GitHub #215, #281, ADR 0030): the images of retained prompt
# checkpoints and shared prefixes, reserved at load. RETAINED_DEVICE
# (--reuse-retained-device) keeps them in VRAM, handed out first; RETAINED_HOST
# (--reuse-retained-host) in one pinned host block, a PCIe copy per capture and
# per claim. Empty = the server's defaults: none in VRAM, two per decode lane
# on the host. A card with VRAM to spare: RETAINED_DEVICE=16 RETAINED_HOST=0.
RETAINED_DEVICE ?=
RETAINED_HOST ?=
# Vision (--vision-enabled, GitHub #179): a load that takes image parts. Without it
# every `image_url` part is refused with `vision_disabled`, whatever the
# artifact holds. VISION_MAX_TOKENS caps one request's vision tokens
# (--vision-max-tokens); empty = the server's own envelope.
VISION ?=
VISION_MAX_TOKENS ?=
# RoPE scaling (--model-rope-scaling, GitHub #227): the text rotary table.
# Empty (or `auto`) is the server's default: none up to 262,144 positions, the
# linear table the checkpoint was trained with, else YaRN at the context /
# 262,144 -- MAX_CONTEXT=524288 gets yarn:2 without being told, the 5090's
# 917,504 yarn:3.5, and a context the load adapts its own. `none` forces
# the linear table; `yarn:F` rescales the envelope by F, whatever the context.
# The full spelling is `yarn:F[,t=<c>][,bf=<n>][,bs=<n>]`. A YaRN table
# rescales every request's table, short ones included. Both models take it
# (ADR 0057, ADR 0059).
ROPE_SCALING ?=
# The KV-RAM host tier's budget (--reuse-kv-host-pool-bytes, P4-07, GitHub #125):
# 0 disables the host tier entirely (no evict-to-RAM overflow path). The
# 27B's: Flash-Next takes FN_KV_HOST_POOL_BYTES instead.
KV_HOST_POOL_BYTES ?= 8G
# Flash-Next's own KV-RAM arena (--qwen38flashnext-reuse-kv-host-pool-bytes,
# ADR 0046), passed on every start -- the 27B's too, so a switch to
# Flash-Next gets it rather than the 27B's 8G: beside the 38 GB of pinned
# experts the host plan refuses 8G (spec flash-next/05 sizes it at 2G). 0
# turns Flash-Next's tier off; its evictions then go to KV-disk.
FN_KV_HOST_POOL_BYTES ?= 2G
# What Flash-Next's host plan leaves of the RAM available at load
# (--model-host-margin-bytes): a load whose pinned lines leave less is
# refused, naming the line to shrink. Empty = the server's 4G.
HOST_MARGIN_BYTES ?=
# KV-disk, the tier below KV-RAM (--kv-disk-bytes, spec vram-budget/03):
# evicted sequences and retained prompt checkpoints kept as files and read
# back instead of prefilled again. A ceiling the server cuts at start to the
# volume's free space less 10 GiB; 0 turns it off. Empty = the server's: 4G
# on Flash-Next, 0 (off) on the 27B.
KV_DISK_BYTES ?=
# Where KV-disk's files go (--kv-disk-path): model (beside the artifact),
# auto (the per-user cache directory, under kv-disk) or a directory, each
# process in its own ignis-kv-disk/<pid>-<nonce> there. Empty = model.
KV_DISK_PATH ?=

# Extra ignis-server flags, verbatim: ARGS='--reuse-kv-host-pool-bytes 0'
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
