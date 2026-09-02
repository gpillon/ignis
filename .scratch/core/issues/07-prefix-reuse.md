# 07 — prefix reuse (sibling prefix caching)

Status: resolved (commit b481df8, 2026-09-02; GitHub #18)
GitHub: #18
Blocked by: #17 (core-06)

**Prefix reuse** (`docs/design/ignis-v1.md` §2, `CONTEXT.md` "Prefix
reuse"):

- Concurrent requests sharing a prefix skip the redundant prefill — siblings
  reuse the shared KV prefix instead of re-prefilling.
- The `sibling_prefix_reused_tok` counter tracks reuse (exposed via
  telemetry, `server-02`).

Delivered (commit b481df8): a new `PrefixCache` in
`crates/core/src/prefix.rs` (register / claim / release, refcounted
page-aligned entries, the core-02 GDN-boundary guard, and the cumulative
`sibling_prefix_reused_tok` counter) + 8 unit tests; the concrete
scheduler (`concrete.rs`) claims the longest cached prefix before the
batched prefill (each claimant's KV reservation shrinks to its tail — the
shared entry's pages are charged to the pool **once**, for every claimant,
not `N` times), registers fresh requests post-prefill (the charge split),
runs the admission machine against `capacity − the cache's pins`
(consistent with `fits`), and wires the reserved-lane candidate flag to the
prefix (ADR 0004); `types.rs` adds `SchedEvent::PrefixReused`,
`request.rs` carries per-request `prefix_entry` / `shared_prefix_tokens`;
the server router (`server/src/engine.rs`) forwards `PrefixReused` to the
request's stream (it carries a request id, unlike the `Protected` batch
marker) + a routing test. Integration tests in
`crates/core/tests/prefix_reuse.rs` (a sibling skips the shared head, a
`1 main + 3 subagents` load accumulates the counter at 3 × 32, the shared
prefix is charged once not per claimant, no reuse for unrelated prompts, a
full-prompt match skips the entire prefill, and a failed-prefill retry does
not double-claim). CPU-tested (ADR 0006); the full workspace `cargo test`
is green.

_Known limitation (v1):_ reuse happens **across** prefill batches — the
first request to prefill a prefix registers it, and later batches'
siblings claim it. Siblings that land in the *same* batched-prefill call
(core-04) cannot reuse each other (a prefix can only be cached once its
prefill has warmed it). The common "1 main, then N subagents" pattern
reuses correctly; a fully concurrent 1 + N load that fits in one
`max_prefill_batch` prefill warms the shared head once per request.
Same-batch dedup is a documented v1.1+ extension.

## Acceptance

- Sibling prefix reuse is active: concurrent requests sharing a prefix skip
  the redundant prefill.
- The `sibling_prefix_reused_tok` counter increments under a "1 main + N
  subagents" load.
