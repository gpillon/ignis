# ADR 0006 — Exclusive GPU testing (no concurrent engines)

## Status

Accepted (2026-09-02, grilling session).

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

## Consequences

- No simultaneous engines on the 5090 → no VRAM starvation of either engine.
- The preflight makes the foot-gun (accidentally running both) explicit and
  non-silent.
- Coexistence would only be possible with *reduced* (partial) engine sizes,
  which is not a v1 goal.