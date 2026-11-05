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
  each model load, on any card: ignis's own full-rate kernels, which an RTX 30
  or RTX 40 card runs, or the vendored chunked kernels an RTX 50 or RTX PRO
  card runs ([ADR 0080](../../adr/0080-on-sm8x-the-trellis-prefill-decodes-each-matrix-once-per-chunk-and-multiplies-on-the-integer-mma.md)).
  Unset, the card's generation decides. Any other value refuses the load,
  naming the variable. The load logs the route it runs, under
  `ignis.runtime.prefill_routes`, with the trellis prefill and the prompt
  attention band beside it, and its VRAM plan is sized for that route. The
  sm_86 gate's quality leg ([Releases](releases.md)) sets it to `ours`, since
  the `span_logits` example it runs reads the variables rather than the
  card.

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
Playground's typecheck and tests, and the guide's tests and build.
