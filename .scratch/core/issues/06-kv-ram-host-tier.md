# 06 — KV-RAM host tier (probation / protected eviction)

Status: resolved (commit 7115251, 2026-09-02; GitHub #17)
GitHub: #17
Blocked by: #13 (core-04)

The **KV-RAM host tier** (`docs/design/ignis-v1.md` §2, `CONTEXT.md`
"KV-RAM"):

- Snapshots GPU lanes to host RAM so sibling requests **restore** instead of
  re-prefilling.
- Two-tier eviction: **probation → protected**.
- Pulled into v1 (not v1.1). Must respect the GDN boundary (`core-02`).

Delivered (commit 7115251): the bounded `HostTier` in `host.rs`
(probation → protected two-tier eviction, GDN-boundary check, LRU
eviction) + 8 unit tests; the `Running→Evicted→Running` (restore) and
`Evicted→Admitted` (re-queue) transitions in `request.rs` — a re-queued
request's GDN state is reset, so a re-prefilled stream cannot accept a
snapshot at a position it never reached; `gdn.rs` `checkpoint` now records
at the current position (>=) + a new `advance` (mid-prefill progress,
non-boundary); `concrete.rs` wiring (`host_capacity_pages` knob,
`try_evict_for_head`, `restore_pass`, decode-phase GDN checkpoints, KV
bookkeeping); `types.rs` `Evicted` / `Restored` / `Requeued` schedule
events; 2 end-to-end scenarios in `crates/core/tests/host_tier.rs`
(evict frees a blocked head, restore skips re-prefill, evictions bounded
under overflow load). Admission tests pin `host_capacity_pages: 0` so the
core-05 scenarios exercise the admission machine in isolation. CPU-tested
(ADR 0006); the full workspace `cargo test` is green.

## Acceptance

- The host tier evicts / restores GPU lanes to host RAM.
- Evictions are bounded under the N=8 + overflow load; sibling requests
  restore instead of re-prefilling.
