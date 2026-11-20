---
status: accepted
amended-by: [ADR-0084]
---

# ADR 0006 — Exclusive GPU testing (no concurrent engines)

## Status

Accepted (2026-09-02, grilling session). Edited 2026-11-20 (#707).

## Context

ignis is developed (and tested) with the same model runner (ninfer, running
Qwen 3.8-27B) that the developer uses for coding — a **self-bootstrapping**
loop. When we run ignis to test it, the RTX 5090 may already be holding ninfer
(both engines need the GPU). The 5090 has 32 GB; a *single* engine's full
footprint (weights ≈16.03 GiB + KV pool ~12–13 GB + graph ~1.9 GB) leaves only
~3.5–4.4 GB free, and two engines' *weights alone* (2 × 16.03 GiB = 32.06 GiB)
exceed 32 GB **before** any KV/graphs. Two full-size engines therefore cannot
coexist on the 5090.

## Decision

Testing is **exclusive**: before running ignis against the GPU, the developer
stops/pauses ninfer so that only one engine holds the GPU. The guard is a
**preflight check in the `bench`/test harness** (detect "another process holds
the GPU → refuse or wait") plus a **runbook step** ("stop ninfer before
testing"). It is a **test-harness concern, not a runtime feature** of the
engine: v1 has no multi-tenant story, and the engine itself does not refuse to
start because another process uses the GPU.

Beside the preflight, a **GPU lock**, one per card, records who holds it: the
holder's pid and process start time, its worktree, its command and when the
hold began. A holder that has exited, or whose pid now names another process,
holds nothing, so a run killed before it could clean up leaves no lock
standing. Every way onto the card takes it: the GPU profile for its whole run
(or for the few tests it is given), a server started through `make`, which
hands the lock to the server until it stops, the kernel leaf's op tests, and
any command run through `make gpu-run`. A server start and the guard refuse at
once while someone else holds it, naming the holder; a test, `make gpu-run`
and the profile wait for it up to a limit, 15 minutes by default, then fail
naming it. The lock is added to the preflight, not put in its place: the
profile's pass marker and the process check stay, since the engine's test
helper reads the marker and a worktree on an older commit knows only those
two. It is still the harness's: the lock lives in the scripts behind `make`,
and `ignis-server` run by hand takes none.

## Consequences

- No simultaneous engines on the 5090 → no VRAM starvation of either engine.
- The preflight makes the foot-gun (accidentally running both) explicit and
  non-silent.
- With the lock, two runs no longer race between the check and the start: the
  one that finds the card taken waits or refuses with the holder's name
  instead of dying with no diagnostic, so an agent can run its own GPU tests
  while others share the card.
- Coexistence would only be possible with *reduced* (partial) engine sizes,
  which is not a v1 goal.