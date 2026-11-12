---
title: "Contributing"
description: "How ignis is worked on: the ground rules, the repository's layout, and where each kind of work is recorded."
sidebar:
  order: 1
---

ignis is built by its owner and by coding agents working from GitHub issues.
A human contributor follows the same rules the agents do; they are written
down in [`AGENTS.md`](../../../AGENTS.md) and the files it points at under
[`docs/agents/`](../../agents/), and this section is the readable tour of
them.

## The ground rules

- **Every code change ships with a test**, and the work is not done until
  `cargo test` passes workspace-wide ([Testing](testing.md)).
- **Every user-visible change updates this guide in the same commit**: a
  flag, a default, a route, a field of a response, a Playground screen, a
  metric ([`docs/agents/user-docs.md`](../../agents/user-docs.md)).
- **The GPU runs one thing at a time.** Before anything touches the card — the
  GPU test profile, a bench run, one `--ignored` test — check that nothing else
  holds it: `make gpu-status`. Two runs on one 32 GB card means one of them
  dies with no diagnostic.
- **Diffs are semantic only.** The repository is not `rustfmt`-clean and is
  never reformatted wholesale; there is deliberately no `fmt` target. This rule
  and the others a review applies are in
  [`CODING_STANDARDS.md`](../../../CODING_STANDARDS.md).
- **A decision that is hard to reverse gets an ADR**, a measured result that
  future work will reuse gets a finding, and the acceptance criteria of a piece
  of work live in its spec's tickets ([How work is tracked](conventions.md)).

## The repository

| Path | What lives there |
|---|---|
| `crates/server` | `ignis-server`: the HTTP surface, configuration, the model catalog and downloads, the Playground's routes, metrics |
| `crates/core` | The scheduler, the paged KV cache, request state, model topology |
| `crates/runtime` | The safe wrapper over the kernel leaf's step ABI, and the scheduler's compute adapter |
| `crates/artifact` | The `.ninfer` container reader; its build script builds the kernel leaf for `--features cuda` |
| `crates/logging` | The structured-logging event model |
| `crates/grammar` | `ignis-grammar`: structured output's grammar, pure Rust: `json_object` and JSON schemas compiled to an automaton over the vocabulary's bytes, which checks each sampled token and gives the permitted set a repair samples from |
| `crates/speech` | `ignis-speech`: the speech model's CPU executor (Qwen3-ASR), pure Rust: the GGUF reader, the quantized AVX2 kernels, the encoder and decoder graphs, the mel front end and the incremental transcription API; and `server_vad`'s voice activity detection, Silero VAD ported, its weights built in |
| `crates/bench` | `ignis-bench`: the trace-replay harness, the canary suite, TTFT and gate measurements |
| `crates/vendor` | The kernel-leaf vendoring tool, and the integrity check of the vendored subtree |
| `kernel/` | The C++/CUDA kernel leaf (SM120a), with its own CTest op tests |
| `web/` | The Playground (React, Vite, vitest), embedded in the server binary |
| `docs/user/` | This guide; `docs/site/` builds it |
| `docs/adr/`, `docs/specs/`, `docs/formats/`, `docs/findings/` | Decisions, the specs written before 2026-10-25 (later ones are GitHub issues), the byte and wire contracts code cites, measured results |
| `docs/agents/` | The conventions agents (and humans) follow |
| `CODING_STANDARDS.md` | The rules a review holds a diff to, and what the review leaves out |
| `tools/`, `scripts/`, `bench/` | Converters and fixtures, local run helpers, bench traces |
| `mk/`, `Makefile` | The make targets and their per-OS hooks |

The domain vocabulary — lane, KV page, retained state, served id, card class —
is defined once in [`GLOSSARY.md`](../../../GLOSSARY.md); the
[glossary](../reference/glossary.md) is its user-facing share.

## Where to go next

1. [Building from source](building.md): the kernel leaf, the Rust workspace,
   the CPU mock.
2. [Testing](testing.md): the default suite, the GPU profile, the Playground's
   and the guide's own checks.
3. [Working with make](make.md): the targets, and how a run is guarded.
4. [How work is tracked](conventions.md): issues, specs, ADRs, contracts,
   findings, the guide, commits.
5. [Cutting a release](releases.md): version, changelog, tag, and what the
   workflows publish.
6. [Working in a fork](forks.md): what to send upstream, and how.
