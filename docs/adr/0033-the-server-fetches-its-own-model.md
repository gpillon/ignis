# ADR 0033 — the server fetches its own model, and asks first only when somebody is there

## Status

Accepted (2026-09-19, owner decision — GitHub #234).

## Context

Until now a server started without `--artifact`/`IGNIS_ARTIFACT` came up on
the placeholder template and `MockCompute` — a server that answers, in text
that is not natural language, and says nothing about where the real model
comes from. The `.ninfer` artifact is published (Hugging Face, the repo the
model card in `models/*.README.md` names), so the one thing standing between
a fresh machine and a working server was a `hf download` line in a README.

The reference stack does not do this either: NInfer uses `isatty` for a load
progress bar and nothing else. This is ours to shape.

## Decision

`ignis-server` fetches a missing model itself, from a registry pinned in the
binary (`crates/server/src/download.rs`).

- **Interactive asks, non-interactive fetches.** With stdin on a terminal the
  operator is asked, on **stderr** (stdout carries the plain lines the
  Makefile helpers parse — the generated API key, the public URL), with the
  size, the source and the destination in the question; `[y/N]`, and EOF
  reads as no, because 19.4 GB is not something to spend on a stray Enter.
  With stdin not a terminal — a container, a daemon, CI — there is nobody to
  ask and it just downloads: a start that blocks forever on a question nobody
  will answer is worse than either answer.
- **Only the unset-`--artifact` case.** A path the operator named is their
  word: missing, it refuses the start as it always has, now with one line
  naming `--model-download-path`. A typo must stay a typo rather than become
  19.4 GB fetched somewhere else.
- **Only a `--features cuda` build.** A binary that cannot run the weights
  has no use for them. This, not a remembered flag, is what keeps `make mock`,
  `cargo test` and every CPU CI job off the network.
- **Flat destination.** `<--model-download-path>/<the repo's file name>`,
  default `./models` — exactly what `hf download <repo> <file> --local-dir
  models` produces, so a model fetched by hand and one fetched by the server
  are the same file, and a machine that already has it downloads nothing.
- **The digest is pinned in the binary,** never read from the repo: the
  published `.sha256` sits in the same trust domain as the artifact it
  describes, so it can attest nothing about it. The transfer streams to
  `<name>.part`, hashes as it writes, and renames only once the length and
  digest match. What becomes of the part file says which failure it was: a
  body that stopped early keeps it, because that is a dropped connection and
  the next start resumes it with `Range`; a body that hashes wrong, one that
  runs past the pinned length, and a partial the source answers `416` to are
  all deleted — they are not this artifact, and keeping them would make every
  later start resume bytes that can never verify.
- **The sidecar first.** The provenance record (ADR 0002) is one small
  request and the loader refuses a load without it, so it is fetched before
  the body: a repo that cannot serve it fails in a second instead of an hour.
- **A failed fetch refuses the start.** Falling back to the mock after the
  operator said yes would be the same silent degradation the loader refuses
  for an unclean checksum.

## Consequences

- A cuda binary run with no `--artifact` and no model on disk now downloads
  instead of coming up on the placeholder. The artifact-less
  `podman run … ghcr.io/gpillon/ignis` smoke test therefore carries
  `--no-model-download` (README), and a container that should keep its model
  gets `-v …:/models --model-download-path /models`.
- Adding a model is a registry row — id, repo, two file names, size, digest —
  and nothing else.
- The pins mean a republished artifact under the same file name fails here —
  by length if it grew or shrank, by digest if it did neither. That is the
  intended direction of the failure: a new image is a new registry row and a
  new binary, not a silent swap.

## Considered Options

- **A `models.json` next to the binary** — rejected: a file an attacker can
  edit is not a trust anchor, and a registry that ships with the engine
  cannot drift from the loader that consumes it.
- **Trusting the repo's `.sha256`** — rejected: a digest served by the same
  repo, over the same connection, as the artifact it describes attests only
  that the transfer was not corrupted — which the length check already tells
  us — and nothing about what was published.
- **Downloading on a non-cuda build too** — rejected: 19.4 GB of weights for
  a backend that will never bind them.
- **Asking even without a TTY** (a timeout, defaulting to no) — rejected: a
  container start that waits, then comes up wrong, is the worst of both.
