---
title: "Testing"
description: "The CPU-only default suite, the explicit GPU profile, the Playground's tests and the guide's build: what each covers and when it is due."
sidebar:
  order: 3
---

Every change ships with a test, and a task is not complete until the test
suite is green. The full runbook — fixtures, the GPU profile's script, the
measurement instruments — is [`docs/agents/testing.md`](../../agents/testing.md)
and the per-area files it links; this page is the map.

## The default suite: `cargo test`

```sh
cargo test          # workspace-wide; make test is the same
```

It is CPU-only and never touches the GPU, which is what lets CI run it on
Linux and on Windows runners with no card; the release builds start only when
both legs are green ([ADR 0032](../../adr/0032-release-through-the-container-build.md)).
A machine-local smoke test whose fixture is absent skips, and a skip counts as
green — except for compute work: a test on the forward pass that skips on a
busy GPU or a kernel error proves nothing, so those tests live in the GPU
profile instead.

`cargo test` also holds two things that are not Rust behaviour:

- the vendored kernel subtree's integrity: a vendored file edited without a
  recorded patch turns the suite red
  ([ADR 0010](../../adr/0010-vendored-reference-kernels.md));
- the generated pages of this guide's [reference](../reference/configuration.md):
  a field, profile value, catalog pin, metric or route changed without
  `make docs-reference` turns it red, naming the page.

## The GPU profile

Every test that touches the GPU is `#[ignore]`d and runs only when asked for,
under `IGNIS_GPU_PROFILE=1`. There a busy GPU, a kernel error or a missing
fixture is a failure, never a skip; outside the profile the same conditions
print `SKIP:` and return, for a quick local check
([ADR 0006](../../adr/0006-exclusive-gpu-testing.md)).

```sh
make gpu-status     # VRAM in use, and every process that may hold the card
make gpu-profile    # preflight, the kernel leaf's op tests, the ignored Rust GPU tests
make kernel-test    # the kernel leaf's CTest op tests alone
```

`make gpu-profile` runs `scripts/gpu-profile.ps1` on Windows and
`scripts/gpu-profile.sh` on Linux ([make](make.md) lists their switches).

The card fits one run at a time, in any worktree: check `make gpu-status`
before every GPU run.

### Measurement switches

A few environment variables pick a kernel route only to measure it against
the other. They are no configuration fields: `GET /v1/config` and
`config print` do not show them, and an operator has no reason to set one.

- `IGNIS_GDN_PREFILL=ours|vendored` picks the 27B's gated delta net prefill at
  each model load, on any card: ignis's own full-rate kernels, or the vendored
  chunked kernels they replaced ([ADR 0080](../../adr/0080-on-sm8x-the-trellis-prefill-decodes-each-matrix-once-per-chunk-and-multiplies-on-the-integer-mma.md),
  [ADR 0085](../../adr/0085-on-blackwell-too-the-27bs-gdn-prefill-runs-ignis-own-kernels.md)).
  Unset, every card runs ignis's own. Any other value refuses the load,
  naming the variable. The load logs the route it runs, under
  `ignis.runtime.prefill_routes`, with the trellis prefill and the prompt
  attention band beside it, and its VRAM plan is sized for that route. The
  sm_86 gate's quality leg ([Releases](releases.md)) names `ours` beside the
  other two routes, as its limits were measured.
- `IGNIS_SECOND_TAIL_CUT=1` (from 0.11.0) gives the 27B on an RTX 50 or RTX
  PRO card, or with no card read, its second tail cut back at each model load:
  a reusable prompt is cut at the generation opener's page floor too, where
  its imaged prefix is published, as 0.10.0 cuts it there
  ([ADR 0100](../../adr/0100-the-27b-cuts-a-reusable-prompts-tail-once-at-the-generation-opener-on-every-card.md)),
  the comparison a time-to-first-token measurement needs. Unset, the 27B cuts a
  reusable prompt once, at the opener, on every card. The 27B on an RTX 30 or
  RTX 40 card and Flash-Next cut once whatever the variable says. Any other
  value refuses the load, naming the variable. The load logs what it runs
  under `ignis.model.loaded` (`second_tail_cut`: `true` only where it cuts a
  second time).
- `IGNIS_HQ_DECODE_ATTENTION=ours|vendored` (from 0.11.0) picks the attention
  route of the 27B's decode and verify rounds over `hq-e8-2b` at each model
  load: ignis's warp-specialized route, or the vendored kernel's route it
  replaced ([ADR 0031](../../adr/0031-vendored-kernel-bottleneck-exemption.md),
  [ADR 0093](../../adr/0093-on-blackwell-the-27bs-decode-and-verify-rounds-attend-over-hq-e8-2b-on-ignis-own-route-and-the-vendored-kernel-stays-its-oracle.md)),
  the comparison a speed measurement of the two needs. Unset, or `ours`, an
  RTX 50 or RTX PRO card runs ignis's own; `vendored` runs the vendored one
  there. An RTX 30 or RTX 40 card, a card the build holds no code of the route
  for, and a `bf16` load run the vendored route whatever the variable says.
  Any other value refuses the load, naming the variable. The load logs the
  route it runs under `ignis.runtime.decode_routes` (`hq_decode_attention`),
  and `ignis.runtime.decode_graph_capture` gives the node count of each
  width's decode graph and verify graph (`decode_nodes`, `verify_nodes`),
  which the two routes share, as they share every byte they write and the
  VRAM plan.
- `IGNIS_MTP_HEAD_LINEAR=ours|vendored` (from 0.11.0) picks, at each MTP load
  of the 27B, the route of the head's five W8 matrices: ignis's own, the one
  an RTX 30 or RTX 40 card runs (the vendored SIMT kernel up to 8 tokens, the
  integer MMA above, `kernel/src/w8_imma.h`), or the vendored `ops::linear`
  an RTX 50 or RTX PRO card runs. Unset, or `vendored`, an RTX 50 or RTX PRO
  card runs the vendored one; `ours` runs ignis's there, which is how the
  RTX 5090 checks the route the older cards run
  (`crates/core/tests/mtp_head_gpu.rs` holds it to the head's f64
  recomputation). An RTX 30 or RTX 40 card runs ignis's whatever the variable
  says: the vendored routes there are half-rate BF16 MMA, or have no kernel
  image. Any other value refuses the load, naming the variable. The load logs
  the route under `ignis.runtime.decode_routes` (`mtp_head_linear`: `ours`,
  `vendored`, or `none` without the head), and the VRAM plan is the same on
  both.
- `IGNIS_TMA_DESCRIPTOR_CACHE=0|1` (from 0.11.0) picks, at each model load on
  Windows, how the 27B's NVFP4 prompt products get their tensor maps: from the
  device slots the load keeps (`1`, or unset), or by upstream's allocation,
  copy and free before every product (`0`), the comparison a measurement of
  the two needs.
  Under `0` the plan has no `tma_descriptors` line, so the KV pool takes its
  bytes, and the first request after the load pays the device memory pool's
  first allocation again. The bytes the products write are the same either
  way (`ignis_kernel_tma_descriptor_cache_test`). Linux passes the maps by
  value and has no slots whatever the variable says. Any other value refuses
  the load, naming the variable.
- `IGNIS_PREFILL_SWIGLU=split|fused|quant` (from 0.11.0) picks, at each load
  of the 27B's NVFP4 artifact, how its prefill's MLP multiplies by the gate
  and up matrix and applies SwiGLU. `quant`, or unset: in one kernel at every
  chunk width from 1,024 tokens in whole 256-token tiles, whose epilogue
  quantizes each token's SwiGLU output straight into the down product's NVFP4
  input, so the [17,408 x T] BF16 matrix is never written; in a layer whose
  down product does not run on the NVFP4 W4A4 route (its weight held as an
  exception tensor in FP8, say) the same kernel writes that matrix instead.
  `fused`: the same kernel writing the BF16 matrix in every layer, which the
  down product then quantizes itself. `split`: the vendored route 0.10.0
  runs, the product written out in full and a SiLU-mul pass reading it back,
  and at exactly 1,024 tokens the vendored fused kernel. `quant` and `fused`
  write the split route's bytes at every width: the BF16 output, the down
  product's codes and scales, and the residual after it
  (`ignis_kernel_prefill_swiglu_test`); the vendored fused kernel rounds once
  fewer, so under `split` a 1,024-token chunk computes the MLP with other bits
  than a 2,048-token one. Any other value refuses the load, naming the
  variable. The load logs the route under `ignis.runtime.prefill_routes`
  (`prefill_swiglu`: `quant`, `fused`, `split`, or `none` where its prefill
  runs no W4A4 TMA product, as for `w4a4_activation_layout` below: the trellis
  artifacts, a prefill chunk below 1,024 tokens), and the VRAM plan is the
  same on all three.
- `IGNIS_W4A4_ACTIVATION_LAYOUT=ktile|token` (from 0.11.0) picks, at each
  model load, the layout the 27B's NVFP4 prompt products read their
  activations in on an RTX 50 or RTX PRO card: K tile by K tile (`ktile`, or
  unset), which they read in whole 128-byte lines, or token by token
  (`token`), the layout of earlier releases, the comparison a measurement of
  the two needs. The gate and up product's fused SwiGLU (`IGNIS_PREFILL_SWIGLU`
  above) reads its input in the same layout, and under `quant` writes the
  down product's input in it too: K-tile-major by default, token-major under
  `token`, so both variables combine and every combination writes the split
  route's bytes (`ignis_kernel_prefill_swiglu_test`). The bytes the products
  write and the VRAM plan are the same either way
  (`ignis_kernel_w4a4_activation_layout_test`). The load logs the
  layout under `ignis.runtime.prefill_routes` (`w4a4_activation_layout`:
  `ktile`, `token`, or `none` where its prefill runs no such product: the
  trellis artifacts, Flash-Next, an RTX 30 or RTX 40 card, a prefill chunk
  below 1,024 tokens). Any other value refuses the load, naming the variable.
- `IGNIS_ROUND_TIMING=1` (from 0.11.0) picks no route: it times each decode
  and verify round of the 27B that replays its graph, on the GPU's clock and
  with no tracer, by an event recorded on the stream just before the round's
  graph and one just after it. Every 256 rounds of a kind, the server's
  standard error (`.scratch/serve/ignis-server.log.err` under `make start`)
  gets one line, `ignis leaf: round timing plain n 256: period median … us
  mean …, span median …, gap median …` (`verify` for the verify rounds, and
  `verify extended` for those at the extended round shape of a load past 7
  drafts): the period from one round's start to the next one's, the graph's
  span, and the gap from its end to the next round's start, each the round's
  own whatever shape the next round is, two rounds that begin 50 ms or more
  apart left out (a request or a prefill between them). This is where a
  change to the round is read: under Nsight Systems each graph launch costs
  ~400 us more on the RTX 5090's host, so a launch read off a trace is the
  tracer's. It is read at each capture of the decode graphs, once per load;
  unset, or `0`, nothing is timed and the rounds pay nothing. Any other value
  refuses the load, naming the variable. Flash-Next's rounds are never timed.
- `IGNIS_DEFER_VERIFY=0|1` (from 0.11.0) picks, at each capture of the decode
  graphs, how a verify round of the 27B with DFlash2 waits on the host. `1`, or
  unset: one kernel copies the round's inputs from page-locked host memory the
  GPU maps, another hands its accept back there behind a ticket the server
  reads in place, the drafter's append after the commit replays one graph per
  width, and the round returns with its commit still on the GPU, every later
  call queued behind it. `0`: the round of earlier releases, its inputs and
  its accept copied through pageable memory, a stream synchronize after the
  accept and another after the commit, the comparison a measurement of the two
  needs. The committed tokens are the same either way (the round identity
  test's `sync-verify` arm below). Whatever the variable says, an MTP load's
  rounds, a teacher-forced round, a round where a lane drafts nothing or a
  sampled lane's run is cut (its penalty counts come back to the server), and
  every round before the load's first capture return as earlier releases do.
  The VRAM plan is the same on both. Any other value refuses the load, naming
  the variable.
- `IGNIS_DECODE_GRAPH_COPIES=1` (from 0.11.0) gives the 27B the decode and
  verify rounds of earlier releases, the comparison a measurement of the two
  needs. Unset, an RTX 50 or RTX PRO card runs them copy-free: each layer adds
  its output into its input in place, and each GDN layer of a decode round
  reads its q, k and v where the convolution wrote them, at two or more lanes
  apart for each lane. No decode round's graph holds a copy then (on the NVFP4
  artifact with DFlash2, 677 nodes instead of 885 at one lane, 789 instead of
  997 at two or three, 924 instead of 1,132 at four to eight), and the verify
  round's graph keeps only the drafter's 22 copies of its 86; `1` copies them
  as before. `wide` copies as before past one lane and in the verify
  round and keeps the one-lane round copy-free, so that the share of those
  rounds can be measured apart. The tokens are the same either way (the round
  identity test below, arms `decode-graph-copies` and
  `decode-graph-copies-wide`). An RTX 30 or RTX 40 card copies as before
  whatever the variable says, and Flash-Next's rounds are its own. It is read
  at each capture of the decode graphs, once per load; any other value
  refuses the load, naming the variable. `ignis.runtime.decode_graph_capture`'s
  `decode_nodes` and `verify_nodes` show which round each width captured.
- `IGNIS_FP8_GEMV_PAIRED=0` (from 0.11.0) gives the 27B's grouped q, k, gate
  and v exception projection under `fp8` exception weights its one-row GEMV
  back, one output row a warp, the comparison a speed measurement of the
  paired GEMV needs. Unset, or `1`, a call of 1 to 8 tokens runs the paired
  GEMV, two output rows a warp, whose outputs are the one-row GEMV's byte for
  byte (`ignis_kernel_fp8_linear_test`). The residual-add exception product
  keeps the one-row GEMV either way, as do Flash-Next's FP8 projections, and a
  `bf16` load runs neither. It is read at each call, so a decode graph keeps
  the form its capture read; any other value refuses the load, naming the
  variable.
- `IGNIS_FP8_W8A8_TILE16=0` (from 0.11.0) gives the 27B's exception products
  under `fp8` exception weights their 128-token tile back for a call of 9 to
  16 tokens, the comparison a speed measurement of the 16-token tile needs.
  Unset, or `1`, such a call (a one-lane round at the extended shape, a
  two-lane round at the base shape, a prefill chunk of 9 to 16 tokens) runs
  the product of FP8 activations on a tile of 16 tokens and 64 output rows,
  whose outputs are the 128-token tile's byte for byte
  (`ignis_kernel_fp8_exception_arms_test`), so the tokens are the same (the
  round identity test below, arm `fp8-w8a8-tile128`). A call of 8 tokens or
  fewer keeps its BF16-activation GEMV and one of 17 or more the 128-token
  tile either way, Flash-Next's FP8 projections are their own, and a `bf16`
  load runs no such product. It is read at each call, so a graph keeps the
  tile its capture read; any other value refuses the load, naming the
  variable.
- `IGNIS_DFLASH2_APPEND_KV_ONLY=0` (from 0.11.0) gives the 27B's DFlash2
  drafter its whole q, k and v product back in the append after a verify
  round, the comparison a speed measurement of the append needs. Unset, or
  `1`, an append of 5 to 32 columns (a round at the base shape of 1 to 4
  lanes, one at the extended shape of 1 or 2) projects only the key and value
  rows the drafter keeps, a third of the product, and they are the whole
  product's rows byte for byte (`ignis_kernel_dflash2_small_t_test`). A wider
  append, and the append of a prefill chunk or of a carried anchor, keep the
  whole product either way. It is read at each call, so a graph keeps the
  product its capture read; any other value refuses the load, naming the
  variable.
- `IGNIS_DFLASH2_SMALL_T_FILL=0` (from 0.11.0) gives the 27B's DFlash2
  drafter's conv projections and q, k and v product their vendored launch
  back for a call of 5 to 16 columns, 8 or 16 warps a block, the comparison a
  speed measurement of the launch needs. Unset, or `1`, such a call (a round
  at the base shape of 1 or 2 lanes, one at the extended shape of 1) runs on
  blocks of 4 warps, two or four times as many, whose outputs are the
  vendored launch's byte for byte (`ignis_kernel_dflash2_small_t_test`). A
  wider call keeps the vendored launch either way, as do the drafter's other
  products. It is read at each call, so a graph keeps the launch its capture
  read; any other value refuses the load, naming the variable.
- `IGNIS_GDN_GATING=chain|branch` (from 0.11.0) picks, at each capture of
  the decode graphs, where each GDN layer of the 27B computes its gating (the
  a and b products of the layer's normed input and the decay and update gates
  from them) in the decode and verify rounds. `chain` is the round of earlier
  releases: the gating runs between the convolution and the recurrence, and
  every layer waits on it. `branch`, the value when unset: it runs on a
  branch of the graph forked once the input is normed, beside the input
  projection and the convolution, and joins back before the recurrence. The
  gates are the same bytes in both (`ignis_kernel_gdn_gating_home_test`), so
  the tokens are the same (the round identity test below, arms
  `gdn-gating-chain` and `gdn-gating-branch`), and so are the graphs' node
  counts. An RTX 30 or RTX 40 card keeps `chain` whatever the variable says,
  and Flash-Next's rounds are its own. Any other value refuses the load,
  naming the variable.
- `IGNIS_ROUND_SHAPE=base|extended` (from 0.11.0) forces, at each verify
  round of a DFlash2 load of the 27B past 7 draft tokens, the round's shape
  over the shape rule. Unset: the shape rule and its policy, a round of 1 to 4
  lanes at the extended shape when every lane in it asks for it. `base`: every
  round at the base shape, 8 columns a lane, the round of a load at 7 drafts
  token for token. `extended`: every round of 1 to 4 lanes at the extended
  shape, the window + 1 columns a lane. Under all three a round of 5 to 8
  lanes, or one with a lane held to a grammar, runs at the base shape, and
  each round's outcome still moves its lanes' asks. Both
  shapes' verify and drafter-append graphs are captured at every load, so
  the switch flips between rounds of one load with no capture. A load at 7
  drafts or fewer, an MTP load and Flash-Next never read it. Any other value
  refuses the round, naming the variable. Each shape's graphs, node counts
  and rounds, and how many of them ran eagerly, are in the program's stats;
  `ignis.runtime.decode_graph_capture` lists the shapes (`round_shapes`), and
  `ignis.runtime.verify_graph_capture_incomplete` names a shape's widths whose
  verify graph could not be captured, whose rounds run eagerly.

### The round identity test

`crates/core/tests/round_identity_gpu.rs` holds two arms of the 27B's decode
round, chosen by measurement switches, to the same committed tokens: at every
width the load serves and at its draft window, with greedy and sampled lanes,
a constrained round, a verify round whose runs the host cuts short, a
teacher-forced verify round and a snapshot and restore of each lane right
after a verify round, on the NVFP4 artifact with DFlash2, with its MTP head and with vision, and on
the 16 GB and 12 GB classes' trellis artifacts, each under the exception
weights it has. From 0.11.0 it also loads DFlash2 at 15 and 10 draft tokens
and vision at 15, under `IGNIS_ROUND_SHAPE=extended`, so every arm holds the
extended round at 1 to 4 lanes and the base round past them; DFlash2 at 15
under the shape rule (`nvfp4-dflash2-w15-rule`), every round's shape held to
what its lanes asked and the same lanes taken from the base shape to the
extended one and back on live and teacher-forced rounds at 1 to 4 lanes; a
second test holds a 15-draft load's base rounds to a 7-draft load's tokens;
and a third holds a prefill whose last chunk is 9 to 16 tokens to the same
logits, bit for bit, under `IGNIS_FP8_W8A8_TILE16=0` as unset. Each arm is
held to the graphs' replay with every switch unset; a change behind a switch
of its own adds that switch as an arm, as the file says. The whole run
materializes fourteen loads in turn, so a narrow one is
the usual run:

```sh
IGNIS_ROUND_IDENTITY_LOADS=nvfp4-dflash2 IGNIS_ROUND_IDENTITY_WEIGHTS=bf16 \
  cargo test -p ignis-core --features cuda --test round_identity_gpu -- --ignored --nocapture
```

`IGNIS_ROUND_IDENTITY_ARMS` names the arms the same way. Unset, each of the
three is everything.

### The extended round's numerical class

`crates/core/tests/extended_round_class_gpu.rs` holds the 27B's extended
verify round to its base round's numerical class
([ADR 0104](../../adr/0104-the-27bs-dflash2-verify-round-goes-past-8-columns-a-lane-and-is-held-to-its-base-rounds-numerical-class.md)),
on the NVFP4 artifact with DFlash2 at 15 draft tokens under FP8 exception
weights, on an RTX 5090 (another card skips it). Speculation off writes each
prompt's greedy text; both shapes are then teacher-forced to it at 1 to 4
lanes, once offered the text as their drafts and once replaying their captured
graphs on the drafter's own drafts, and every extended row is held to the base
row at the same text index and lane count, in BF16 ulps, by bounds named in the
file with their measured values. It loads once and takes about two and a half
minutes:

```sh
cargo test -p ignis-core --features cuda --test extended_round_class_gpu -- --ignored --nocapture
```

## The speech executor's CPU legs

The speech executor's kernels and ops are checked against fp64 in the default
suite, and its front end against a recorded Transformers fixture. The checks
on the real model files are `--ignored` CPU legs: every layer against
Transformers FP32, teacher-forced top-1, the WER gates, the token counts and
the kernels on the catalog files' own tensors. They read the study directory
named by `IGNIS_SPEECH_STUDY` (the converted GGUF files, the FLEURS clips, the
dictations) and fail without it, and they run alone on the machine, like a GPU
run. The GPU profile leaves them out:

```sh
cargo test --release -p ignis-speech --test oracle -- --ignored --test-threads=1
```

The fixtures and their bounds are recorded by `tools/speech-oracle/record.py`
([spec speech/01](../../specs/speech/01-cpu-executor.md)).

`server_vad`'s voice activity detection has its own: the default suite checks
the port against Silero's recorded probabilities on the fixture clip, and the
`--ignored` legs check the FLEURS clips and the dictations and run the
`server_vad` WER gate on the dictations, which is CPU-heavy and runs alone
(`IGNIS_SPEECH_LEAD_MS=0,500,1000,1500` adds arms of other leading silences, each
reported).
Its records come from `tools/speech-vad/record.py`
([spec speech/05](../../specs/speech/05-server-vad.md)):

```sh
cargo test --release -p ignis-speech --test vad -- --ignored --test-threads=1
```

## The Playground

The Playground's logic lives in pure modules with vitest tests. A change under
`web/` also needs both green:

```sh
make test-web       # vitest
make web-build      # the TypeScript check and the production build
```

## The connector's skill

From 0.11.0 the [connector's skill](../guides/connector.md#the-skill), the
Node scripts in `connector/ignis`, has `node --test` tests in `connector/`.
They need Node alone, no `npm install`; a change to a script needs them green:

```sh
make test-connector # cd connector && node --test
```

## The guide

A change to `docs/user/` or `docs/site/` builds the site; a link to a missing
page, a missing heading or a missing repository file fails the build.

```sh
make docs-test         # the site's own tests: links, sidebar, changelog
make docs-build        # the site, into docs/site/dist
make docs-dev          # the site with hot reload, at http://localhost:4321/ignis/
make docs-screenshots  # retake the Playground's pictures (SCENES=a,b for some)
```

The build reads the GitHub releases for the Changelog section, so it needs
the network; offline, it writes a placeholder page and says so. The
Playground's pictures under `images/playground/` are taken by a script that
drives the Playground in the installed Chrome: against the dev mock by
default, or a running server with
`PLAYGROUND_URL=http://127.0.0.1:8000/ui/` (the committed pictures come from a
real server). A change that alters a pictured screen retakes its scene.

`make test-all` runs all of the above that need no GPU: `cargo test`, the
Playground's typecheck and tests, the skill's script tests, and the guide's
tests and build.
