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
- `IGNIS_SECOND_TAIL_CUT=1` (from 0.11.0) gives the 27B on an RTX 50 card, or
  with no card read, its second tail cut back at each model load: the imaged
  prefix at the generation opener's page floor, main's behaviour before the
  single cut ([ADR 0100](../../adr/0100-the-27b-cuts-a-reusable-prompts-tail-once-at-the-generation-opener-on-every-card.md)),
  the comparison a time-to-first-token measurement needs. Unset, the 27B cuts a
  reusable prompt once, at the opener, on every card. The 27B on an RTX 30 or
  RTX 40 card and Flash-Next cut once whatever the variable says.
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
