# 01 — Cross-request state reuse: prompt checkpoints, retained prefixes, KV-RAM spill, Tier 2 seam

GitHub: #183 (absorbs #168); slices 0a #184, 0b #185, 1 #186, 2 #187, 3 #188, 4 #190, 5 #189, 6 #191. ADRs: 0029 (this feature), 0024 and 0023
(both amended by it), 0018 (chunk boundaries), 0022 (KV formats), 0028
(expose modes: why reuse is global).

## Problem Statement

ignis reuses state only between *concurrent* requests. A shared prefix is
dropped when its last claimant completes (`crates/core/src/prefix.rs:20-24`),
and KV-RAM holds only live sequences evicted for overflow. The owner's
workload is qwen-code:

- a main agent whose every tool iteration re-sends the whole history;
- bursts of subagents that share ~98% of their prompt (system + tools) but
  rarely overlap in time.

So every turn pays the whole prefill again: seconds of TTFT on a long
agent conversation, where ninfer (`--kv-ram-capacity`) restores in tens of
milliseconds. The Playground shows the same thing (#168).

## Solution

Requests reuse state retained from *earlier, finished* requests, matched
by prompt content:

- Every request leaves a **prompt checkpoint** at its generation opener
  (`<|im_start|>assistant\n`). A later request whose prompt extends it copies
  it and prefills only the rest.
- Every request with a system block of at least one page leaves a
  **retained prefix** at the end of that block. The next subagent of a burst
  claims it as if it were a concurrent sibling.
- Retained state lives on the device while there is room, spills to KV-RAM
  when the device needs the room, and is discarded before any live work is
  touched.
- The blob identity and the match key are shaped so that a later Tier 2
  (KV-disk) adds a tier, not a redesign.

No API change: clients keep re-sending their history. It is on by default.

## User Stories

1. As a coding agent in a tool loop, I want iteration N+1 to prefill only the tool result and the new opener, so that a 60K-token conversation does not pay 60K tokens of prefill per tool call.
2. As a chat user in the Playground, I want the next turn of a conversation to start answering in well under a second, so that multi-turn chat feels like chat (#168).
3. As a chat user, I want "regenerate" and a retry of the same request to hit the same checkpoint, so that asking again costs no prefill.
4. As an agent, I want two forks from the same history (two subagents spawned from one parent context) to both reuse it, so that forking is cheap.
5. As a subagent in a burst that arrives after my sibling finished, I want to skip the shared system + tools block, so that burst TTFT does not depend on timing overlap.
6. As an agent whose conversation was idle long enough to leave the device, I want it restored from KV-RAM instead of re-prefilled, so that coming back to a conversation costs a PCIe copy, not a prefill.
7. As a new user message after a long tool loop, I want to reuse at least the history up to my previous user message, so that dropped thinking does not throw away the whole conversation.
8. As a client whose request was cancelled mid-decode, I want the prompt checkpoint kept, so that the client's retry hits.
9. As a live request, I want retained state never to make me wait or be refused, so that caching someone's past never costs my present.
10. As an evicted live sequence in KV-RAM, I want retained entries discarded before me, so that a bet is lost before certain work.
11. As a multimodal request, I want two prompts that differ only in their images never to share state, so that I am never answered about a picture I did not send.
12. As the owner, I want `request_done` to tell me where reuse came from (`none` / `device` / `kv_ram`), how many tokens it skipped and what the restore cost, so that TTFT is attributable.
13. As the owner, I want Prometheus hit / miss / eviction counters per tier, so that the Monitor shows whether the cache works.
14. As the owner, I want `--prompt-reuse off`, so that cold benches and correctness oracles stay cold.
15. As the owner, I want a byte budget for the device pool of retained images, with a default derived from what is left after load and printed at startup, so that I can see what retention costs.
16. As the owner, I want a blob taken under another artifact, KV format or drafter configuration to be refused, never restored, so that a restart or a config change cannot serve garbage (and so that Tier 2 can exist).
17. As the owner, I want a live/live TTFT comparison with ninfer on a replayed qwen-code trace at phase end, so that a large reuse-path error is detected.
18. As a maintainer, I want the reuse path proven bit-exact against a cold prefill split at the same boundary, so that "reused state is the right state" is a test, not a belief.

## Implementation Decisions

### Rendering prerequisites (slice 0)

Matching requires that a later turn re-render history exactly the way the
earlier turn rendered it, and the way the reference does. Two defects break
this today (verified 2026-09-15):

- **Tool-call arguments come back key-sorted.** `artifact_template.rs:83-97`
  parses `arguments` into a `serde_json` map, and the workspace has no
  `preserve_order`. The fix keeps the model's emitted key order.
- **`preserve_thinking` never reaches the template.** `render_context`
  (`frontend.rs:468-477`) passes only `enable_thinking`, `reasoning_effort`
  and `tools`. The template therefore always keeps a (Rust-emptied) think
  block on every past assistant message, and its strip-before-last-query
  branch never runs. The fix passes it through, so history is rendered as the
  reference renders it.

### Prompt checkpoint on the device (slice 1)

- **Capture.** Every request's prefill is cut at the chunk boundary landing
  on its generation opener: the token offset where the rendered prompt's last
  `<|im_start|>assistant\n` ends. The frontend reports it as a byte offset;
  it is tokenized separately and must be an exact token prefix, otherwise no
  checkpoint is taken.
- **What is captured**, into a device image:
  - the mutable sections: GDN slot, conv taps, drafter window on a DFlash2
    load (its rewrite checkpoint was retired 2026-09-24, spec runtime/05);
  - the penalty-count row — zero at that point, because nothing has been
    sampled yet; verify, and assert it;
  - position and last token;
  - a copy of the partial tail KV page.
  Full KV pages up to the opener are held by refcount.
- **Device image pool.** A byte budget
  (`--retained-pool-bytes`, env `IGNIS_RETAINED_POOL_BYTES`). The default is
  derived from free VRAM after load and reservations; measure it at the
  default load and record it in the ticket. When the budget is full, the
  checkpoint is not taken. The capture never evicts anything.
- **Match.** At admission, find the longest retained prompt checkpoint whose
  token content is a prefix of the request's prompt; ties go to the device.
- **Claim.** Allocate the sequence against the checkpoint: share its full
  pages, clone its image, copy its tail page, and prefill from the opener
  onward. A claim never consumes the entry.
- **First victim.** Retained pages and images are released before admission
  considers any lane (ADR 0023 amendment). Until slice 4, released means
  discarded.
- **Operator surface.**
  - `--prompt-reuse on|off` (default on);
  - `request_done` gains `reuse_source` (`none` / `device` / `kv_ram`),
    `reused_prompt_tokens` and `restore_ms`;
  - the startup capacity event reports the pool budget.
- **Cancellation.** A request cancelled after its checkpoint was captured
  keeps the checkpoint. One cancelled before that point leaves none (no
  partial checkpoints).

### Lineage and non-consuming reuse (slice 2)

- **Chained prefixes** (amended 2026-09-16, #187, from what the leaf turned
  out to be). A capture demands that the whole pages below the opener *be* the
  sequence's shared prefix, and the leaf allowed one prefix per sequence — so
  a sequence that resumed from retained state, or claimed a sibling's head,
  could never capture at its own opener. Both are removed by one change, and
  it is a **composition rather than a relaxed check**: a sequence may now
  publish a prefix *over* the one it holds, owning only the pages it warmed
  itself and taking over the reference it was holding on the head below. The
  capture rule is then satisfied rather than weakened — which matters, because
  a checkpoint standing on pages the capturing sequence owns would outlive its
  own history the moment that request ended. A request publishes at each of
  its boundaries in turn (§Retained prefix), each chained over the last.
- A request that claimed checkpoint C and captures its own checkpoint C'
  supersedes C, unless C is a **turn-opening checkpoint**. C is turn-opening
  when it was the first checkpoint captured after its conversation's last real
  user message — recorded at capture from the frontend's last-real-user-query
  offset, not recomputed later, so a tool-loop iteration does not re-earn the
  role every time it claims.
- A conversation is a **lineage**: the chain of checkpoints linked by "this
  request claimed that entry". There is no session id, so the claim edge is
  the only link there is. A capture joins the lineage of the entry its request
  claimed, or opens one of its own.
- A lineage keeps at most two checkpoints: its latest and its newest
  turn-opening one. Superseded entries are discarded immediately, and the
  discard releases the device image and the hold on the pages below it.
- Regenerate, retry and fork: N claimants of one checkpoint all hit, and the
  entry survives them. **Open** (#187): two forks of one history join one
  lineage, so the later fork's *capture* supersedes the earlier fork's. The
  entry they both claimed is kept either way, and the acceptance criterion
  holds. Splitting a lineage on a fork needs a rule this spec does not have —
  from inside the engine a fork and a new turn are the same shape.

### Retained prefix for bursts (slice 3)

- **Boundary.** The frontend reports the end of the first
  `<|im_start|>system … <|im_end|>\n` block (reasoning instructions + tools +
  system message), floored to whole KV pages. It is 0 (nothing published)
  when the block is under one page.
- **Publish.** The request publishes a shared prefix at that boundary unless
  an identical one is already retained, paying one extra chunk split (ADR
  0024). At refcount 0 the prefix is not dropped: it becomes retained, and is
  a first victim like any retained state.
- **Two boundaries, not one** (amended 2026-09-16, #188 measured it and #187
  fixed it). This spec said "one extra chunk split"; with one prefix per
  sequence the split turned out to be **moved**, not added, and the cost was
  not a chunk but the whole feature: qwen-code sends tools on every request,
  so the system block is always at least a page, so the checkpoint boundary
  always lost and *no prompt checkpoint was ever captured in production*.
  With chained prefixes (§Lineage) both boundaries are published, in prompt
  order, the opener's page chained over the block: two cuts, both real, and
  the pages under each charged to the pool once. The two are always in that
  order — the system block ends before the last `<|im_start|>assistant\n`, and
  flooring is monotone, so the block's page floor never lies past the opener's.
  The list of boundaries a request publishes at is ascending, and the code that
  walks it ("the first boundary past what I already share") depends on that.
- **Claim.** A later request claims it exactly like a concurrent sibling does
  today. A prompt checkpoint match that is longer wins (ADR 0029: longest
  reuse wins). A claimant of a retained prefix chains its own head over it and
  leaves a checkpoint of its own, so a burst leaves one checkpoint per
  sibling rather than one for whoever published first.

### Retained state in KV-RAM (slice 4)

- **Spill.** When the device releases retained state (first-victim path),
  it is snapshotted into KV-RAM if the byte budget can take it — discarding,
  for it, only retained entries that rank strictly below it, planned before
  anything goes — and discarded otherwise. There is no eager copy. (#190, owner
  decision 2026-09-16; ADR 0023 amendment.) A retained prefix that spilled
  comes back to the device once, when a prompt's best reuse is it, and the
  burst shares it there; its blob stays in KV-RAM (ADR 0029 amendment).
- **Materialized blobs.** A snapshot of a sequence or checkpoint holding a
  shared prefix materializes the shared pages (ADR 0024 amendment). The leaf
  refusal code for that case goes away.
- **Discard ordering** (ADR 0023 amendment): retained entries before evicted
  live sequences; then class, probation/protected, LRU. A retained entry
  carries its producer's class. A restore promotes the entry to protected,
  as today, once it lands. An Interactive entry idle past
  `--retained-interactive-ttl` (default 300 s) ranks as an Agent's probation
  entry; a structured retention score is #199.
- **Restore floor.** A KV-RAM match is used only if it reuses at least one
  prefill chunk (1024 tokens) more than the best device match — a device
  prefix included. This is a fixed starting value, tuned by measurement, not
  derived.
- **Lifecycle.** A restored retained entry stays in KV-RAM (non-consuming,
  like the device). Rust never interprets the blob; it keeps
  one host allocation per entry.
- **Metrics.** Prometheus counters per tier for hit, miss, spill, discard and
  restore, following ADR 0017's contract (amend it if the contract requires).
  ignis's values are `device` / `kv_ram`; the gate spec maps them to ninfer's
  `vram_resident` / `host_ram`.

### Identity and the Tier 2 seam (slice 5)

- **Compatibility identity**, carried in the blob header and the device
  entry: artifact content hash, KV format, blob layout version, drafter
  presence and draft window. Operator knobs that do not change state are
  excluded (bind, prefill chunk, concurrency, budgets). A mismatch is refused
  before any byte is written into a sequence.
  - **Departure as built (#189).** The artifact content hash is a digest of
    the container's *directory* — identity, size, payload start, and every
    object's name, kind, numeric format, storage layout, shape, offset and
    length — not of its payload: the v2 container carries no per-tensor
    digest, and hashing ~19 GB at every startup is not a price this check may
    charge. It misses only an in-place re-quantization that preserved the
    whole directory. See ADR 0029 §Consequences; the owner decides whether the
    proxy stands or the container starts carrying a payload digest.
- **Match key**: a hash chain over token ids, with media identity (the
  per-item content digest from the vision processor, #176) mixed in at the
  placeholder spans. It is text-only until vision lands, but the key's shape
  already has the slot, so #180 fills it rather than redesigning it.
- **Keys never contain `RequestId`.** Retained entries are addressed by key.
- **Tier abstraction.** The eviction and restore policy is written over an
  ordered list of tiers, so a third tier adds an entry rather than a branch.
  No disk code in this phase.

### Gate (slice 6)

- **Replay.** A recorded qwen-code trace (main agent tool loop plus one
  subagent burst) through `bench/sim/`. Record TTFT per request with reuse on
  and off, and `reuse_source` / `reused_prompt_tokens` per request.
- **Live/live**, same session, against ninfer with `--kv-ram-capacity` on the
  same trace. An error detector, not an optimization step (owner decision
  2026-09-13: the 99% gate runs once at phase end).
- **Launch discipline.** Needs the GPU exclusively (ninfer-serve stopped;
  standing OK from the owner); `make gpu-status` first.

## Testing Decisions

Good tests check what the next layer observes — tokens generated, the pool
charge, `reuse_source`, the HTTP response — never the order of CUDA calls.

- **CPU, core (mock `Compute`):**
  - match selection (longest wins, device on ties, KV-RAM floor);
  - lineage (latest + turn-opening kept, superseded discarded);
  - first-victim on the device: a retained entry never causes a refusal or
    wait;
  - KV-RAM discard ordering, including *a retained Agent entry discarded
    before an evicted Interactive sequence*;
  - budget exhaustion skips capture;
  - cancellation retention rules;
  - identity mismatch refused.
  Prior art: `crates/core/tests/prefix_reuse.rs`, `host_tier.rs`.
- **CPU, artifact/server:**
  - the rendering fixes: tool-argument key order preserved;
    `preserve_thinking` false strips reasoning before the last query, exactly
    as the reference template does on the same messages;
  - the generation opener offset and the system-block boundary are exact
    token prefixes, on the real frontend, for plain chat, a tool loop and a
    thinking-on render;
  - `--prompt-reuse off` produces no reuse.
  Prior art: `crates/artifact/tests/real_frontend.rs`, `openai_http*.rs`.
- **GPU (fails, never skips, when the card is busy):**
  - **transfer bit-exactness**: a prompt checkpoint restored from the device
    and from KV-RAM continues exactly as the same sequence never moved;
  - **reuse vs split-cold bit-exactness**: turn N+1 reusing turn N's
    checkpoint generates exactly what a cold prefill of turn N+1's prompt
    split at the same opener generates. Also record divergence against the
    unsplit cold prefill as information only (near-ties, cf. #153);
  - **materialized blob**: a claimant of a retained prefix, spilled to KV-RAM
    and restored, continues exactly;
  - all of the above on a DFlash2 load too (drafter sections carried).
  Prior art: `seq_snapshot_gpu.rs`, `prefix_reuse_gpu.rs`,
  `dflash2_window_gpu.rs`.

`cargo test` passes workspace-wide.

## Out of Scope

- Tier 2 KV-disk implementation (seam and identity only).
- Arbitrary longest-common-prefix reuse (GDN state exists only at captured
  points).
- A session or conversation id in the API.
- Per-API-key isolation of reuse (one key exists; revisit when more do).
- Chained / deduplicated blobs.
- A learned retained-prefix boundary.
- Reuse giving queue priority.

## Further Notes

- ninfer reference points: `docs/maintainer/concurrent-inference-architecture.md`
  §6.4–6.5, `docs/maintainer/gpillon-fork-changes.md` ("Host-RAM KV cache
  tier"). Its known defects to avoid:
  - a rewrite-checkpoint restore that served another request's state;
  - a hyperquant side store missing from the RAM record.
  In ignis the section table (ADR 0024) is what prevents the second; the
  slice 5 identity and the bit-exact tests guard the first.
- Measured transfer costs (ADR 0024): device clone ~0.25 ms for 148 MiB;
  snapshot ~45 ms per direction at full context, PCIe Gen 3.
