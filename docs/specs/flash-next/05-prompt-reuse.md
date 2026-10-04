# 05 - prompt reuse for Flash-Next: checkpoints, retained prefixes and KV-RAM on its state

GitHub: #303 (master #298).

ADR 0029 gives the 27B cross-request reuse:
- a prompt checkpoint at every request's generation opener;
- a retained prefix at the end of the system-and-tools block;
- a KV-RAM tier below the device;
- retained slots on the host (#281).

None of it reaches Flash-Next yet. Spec 04 serves the model with every request
prefilled from scratch, and states that this is the follow-up that matters most
for agents. This spec carries ADR 0029 to Flash-Next's state: 36 GDN layers,
12 QSA layers with their indexer, the n-gram embedding's convolution, and
hq-e8-2b KV for 2 KV heads.

ADRs:
- 0029, cross-request state reuse, with its amendments (#188, #190, #193,
  #270);
- 0030, retained slots and every reservation a plan line at load;
- 0024, sequence state transfer;
- 0022, two KV formats with BF16 as the oracle;
- 0043 and 0044 (Accepted 2026-10-04), the second model.

## Decided for the autonomous run (2026-10-04)

The owner approved this spec on 2026-10-04 and vetoed none of the agent's
proposals: every *(proposed)* item below is decided, and ADRs 0043 and 0044 are
accepted. The prerequisites in Further Notes are the ticket's blockers on
GitHub, not open questions.

- `--retained-host 8` and `--retained-device 0` by default for Flash-Next; a
  2 GiB KV-RAM arena.
- The three section details listed in Further Notes are verified against the
  transformers modeling code at the ticket's start; they are checks, not open
  decisions.

## Problem Statement

On Flash-Next a prefill is bound by PCIe, not by compute. Every chunk streams
most of the experts from pinned RAM, measured on the 5090 on 2026-10-04
(`.scratch/flash-next-compression-2026-10-03/review/PREFILL_4K.md`):
- a cold 4K prompt takes 2.65 s, moving 34 GB at 13 GB/s;
- 1K tokens touch 71% of the expert bytes, 4K touch 86%, 8K touch 91%.

An agent turn re-sends its whole history. Take 30K tokens of history plus 1K
new:
- **no reuse:** about 10.4 s, or about 6.5 s with spec 03's scan-resistant
  admission. The 27B pays about 4.5 s for the same turn;
- **with reuse:** an estimated 0.6-1.6 s, against the 27B's ~0.2 s.

Without reuse, Flash-Next is unusable for the agent loops the owner runs, where
each tool call is a turn. With reuse, a turn still pays ~1 s: even 1K new tokens
touch most experts. That floor belongs to the model on this machine, not to
reuse.

## Solution

Flash-Next gets the same reuse as the 27B, matched by content and on by default:
- prompt checkpoints at the generation opener, at most two per conversation,
  claimed without being consumed;
- retained prefixes at the system-and-tools block, chained as in #187;
- KV-RAM as the tier below the device, with the 27B's restore floor;
- retained slots on the host.

What changes is the state, and the memory it has to fit in.

**The mutable image of a Flash-Next sequence** is what a checkpoint copies
beyond its shared KV pages:

| section | size per sequence |
|---|---|
| GDN recurrent state: 36 layers × 48 V heads × 128 × 128, fp32 | 108 MiB |
| GDN conv taps: 36 layers, 3 taps × 10,240 channels | ~2-4 MiB |
| n-gram embedding conv state: the last 9 positions (kernel 4, dilation 3) × 2560 | ~45 KiB |
| penalty-count row (vocab 248,320) | ~1 MiB |
| hq-e8-2b residual window: 12 layers × 2 KV heads, the 27B's window length | ~12.75 MiB |
| **total** | **≈ 124 MiB** |

The 27B's image is 221.8 MiB (187.8 MiB plus a 34 MiB residual window). The
hyper-connection streams carry nothing across tokens, so they add nothing.

**Paged per token** (held by refcount on the device, copied only when
materialized):

| section | bytes per token |
|---|---|
| KV, hq-e8-2b (serving) | 3,456 |
| KV, BF16 (oracle) | 24,576 |
| QSA indexer compressed keys: 12 layers × 128 dims every 4 tokens, BF16 | ~768 |

**A retained 30K-token conversation in hq-e8-2b** costs on the device about
127 MB of pages (104 MB KV and 23 MB indexer) plus its 124 MiB image in a
retained slot. Materialized as a blob in KV-RAM it is about 260 MB.

**Memory.** RAM is the tight side while Flash-Next is loaded:
- ~53 GB available;
- 37.7 GB of pinned experts;
- 1-2 GB of n-gram hot rows;
- staging;
- the remainder is page cache for the NVMe-resident n-gram table.

This spec puts the images on host retained slots and gives KV-RAM a small arena.
Retained state on the device stays the first victim, as ADR 0029 says. The
default device slot count is zero, so the expert cache gives up nothing.

## User Stories

1. As the owner running an agent on Flash-Next, I want iteration N+1 of a tool loop to prefill only the tool result and the new opener, so that a 30K-token conversation does not pay ~10 s per tool call.
2. As the owner, I want a reused agent turn on Flash-Next to start answering in at most about 1.6 s, so that the agent loop stays interactive.
3. As the owner, I want reuse on Flash-Next to match by content exactly as on the 27B, so that no client changes and no session id are needed.
4. As the owner, I want regenerate, retry and forks of the same history to hit the same Flash-Next checkpoint without consuming it, so that asking again costs no prefill.
5. As the owner running two or three agents that share a system and tools block, I want the later ones to skip that block even after the first finished, so that a burst of subagents does not each pay the block's prefill.
6. As the owner, I want a conversation idle long enough to leave the device restored from KV-RAM instead of re-prefilled, so that coming back costs a few hundred milliseconds of PCIe, not seconds of prefill.
7. As the owner, I want a new user message after a long tool loop to reuse at least the history up to my previous user message (the turn-opening checkpoint), so that dropped thinking does not throw the whole conversation away.
8. As the owner, I want retained state never to take VRAM from the expert cache by default, so that decode speed is not traded for reuse unless I choose it.
9. As the owner, I want the host memory reuse costs (retained slots, KV-RAM arena) printed in the host plan beside the experts and the n-gram hot rows, so that I see what it takes from the n-gram page cache.
10. As the owner, I want the load refused, with a message naming what to shrink, when the host plan with reuse leaves less than the safety margin, so that reuse can never push Windows into paging.
11. As the owner, I want `--prompt-reuse off` to work on Flash-Next, so that cold benches and the correctness oracle stay cold.
12. As the owner, I want `request_done` to report the reuse source, the reused token count and the restore time on Flash-Next, so that a slow turn is attributable.
13. As the owner, I want the per-tier hit, miss, spill, discard and restore counters on the Monitor for Flash-Next, so that I can see whether reuse is working.
14. As an engine developer, I want Flash-Next's mutable image defined from its topology (GDN layers and heads, conv taps, n-gram conv state, residual window, penalty row), so that no 27B section size leaks into it.
15. As an engine developer, I want the QSA indexer's compressed keys treated like KV, in pages shared by refcount with the partial tail copied, so that a claim never recomputes the indexer over the history.
16. As an engine developer, I want the n-gram embedding's conv state captured in the image, so that the first new token after a claim sees exactly the embeddings a cold prefill would.
17. As an engine developer, I want the n-gram ids of the first new tokens hashed from the prompt's own preceding tokens, which the match guarantees are the checkpoint's, so that no id context needs storing.
18. As an engine developer, I want a Flash-Next snapshot blob to carry Flash-Next's artifact identity and KV format, so that a 27B blob is never restored under Flash-Next, or the reverse.
19. As an engine developer, I want the reused turn's prefill of its new tokens to go through spec 03's scan-resistant admission, so that a reused turn neither evicts the decode working set nor is slowed by it.
20. As an engine developer, I want reuse proven bit-exact against a cold prefill split at the same boundary, with a cold and a warm expert cache, so that neither the claim nor the expert residency changes what the model computes.
21. As an engine developer, I want the same bit-exactness above 2051 tokens, where QSA attention is sparse, so that a reused long context selects the same key blocks a cold one does.
22. As an engine developer, I want every reuse structure (host slots, KV-RAM arena, tables) owned by the Flash-Next model instance and freed on its drop, so that the phase-2 model switch can reload cleanly.
23. As a reviewer, I want the agent-turn TTFT measured on the 5090 with and without reuse and set against the estimates, so that the spec's claim is a measurement.

## Implementation Decisions

**Owner-made decisions** (2026-10-04):
- reuse for Flash-Next is the follow-up right after spec 04;
- hq-e8-2b KV is in scope (spec 04);
- the model switch is phase 2;
- a switch reloads everything and spends no VRAM on the switch.

*(proposed)* marks the agent's proposals, which the owner may veto.

**ADR 0029 applies unchanged.** The 27B's rules carry over as they are:
- match by content and media identity, longest wins, a tie goes to the tier
  above;
- prompt checkpoints at the generation opener; retained prefixes at the
  system-block boundary, chained per #187;
- at most two checkpoints per lineage (latest and turn-opening); claims never
  consume;
- retained checkpoints are given up before retained prefixes (#188);
- the KV-RAM restore floor and spill rules (#190);
- the reuse boundaries list of #270, of which Flash-Next uses the system block
  and the generation opener: `/v1/decide` and its boundaries are not served on
  Flash-Next (spec 04).

Nothing in this spec changes the scheduler's reuse policy. It adds Flash-Next's
state to the mechanisms the policy drives.

**The Flash-Next mutable image.** The snapshot sections ADR 0024 defines,
derived from the topology (spec 04), for:
- the GDN recurrent state (fp32, as the checkpoint keeps it);
- the GDN conv taps;
- the n-gram embedding conv state;
- the hq-e8-2b residual window;
- the penalty-count row;
- position and last token.

Sizes are computed from the topology and printed at load. The ~124 MiB above is
the expectation, and the ticket replaces it with the measured value. There is no
drafter section, because Flash-Next runs without speculation.

**The n-gram embedding across a claim.**
- The conv state (the last nine positions' conv input) is captured in the image
  *(proposed)*. At ~45 KiB it is cheaper than re-gathering nine positions' rows
  from NVMe, and it keeps the claim bit-exact by construction.
- The n-gram ids of the new tokens are hashed from the prompt's preceding
  tokens. The content match guarantees they equal the checkpoint's, so no id
  context is stored.

**The QSA indexer section is paged like KV.**
- Complete compressed-key blocks live in pages that are shared by refcount
  under a checkpoint or a retained prefix.
- The partial tail (tokens of an incomplete compression block, and the tail of
  its page) is copied with the KV tail page.
- A publish point needs no alignment to the compression block. The copied tail
  carries what the next block needs.

**Retained slots: host by default** *(proposed)*.
- `--retained-host` defaults to 8 for Flash-Next: about 1 GiB pinned, enough
  for three agents (a chain link and a checkpoint each) plus a shared system
  block.
- `--retained-device` defaults to 0. A device slot costs ~124 MiB of expert
  cache, 1:1.
- A claim from a host slot copies ~124 MiB over PCIe, about 10 ms. That is per
  request, never per token, and it is negligible against the turn's ~0.6-1.6 s.

**KV-RAM arena for Flash-Next** *(proposed)*.
- A default of 2 GiB, an operator option.
- It holds materialized blobs of retained state the device gives up, and
  evicted live sequences, as on the 27B.
- At ~260 MB per 30K-token conversation it keeps about seven or eight idle
  conversations.
- Spec 04 left the KV-RAM tier out because the experts take the RAM; this spec
  sizes a small one against the host plan.

**Host plan.** Spec 03's host plan gains two lines: retained host slots and the
KV-RAM arena. With the defaults:

| | GB |
|---|---|
| OS and applications | ~11 |
| pinned experts | 37.7 |
| n-gram hot rows (default) | 1 |
| staging | ~0.5 |
| retained host slots | ~1 |
| KV-RAM arena | 2 |
| **left for page cache and margin** | **~10** |

That is ~3 GB less n-gram page cache than spec 04 alone. The 6 GB refusal
margin of spec 03 still holds. The plan names which reuse line to shrink when
it does not.

**Device side.**
- Retained KV and indexer pages live in the KV pool spec 04 sizes. There is no
  new device reservation, and retained pages are the first victim when a live
  lane needs room.
- A device retained slot, when the operator asks for one, is a plan line taken
  from the expert cache's share and printed as such.

**Identity.**
- A Flash-Next blob's compatibility identity is ADR 0029's: the artifact
  content hash, the KV format and the blob layout version, with drafter
  absent.
- A 27B blob and a Flash-Next blob can never match: their artifact identities
  differ.
- The layout version is per model family.

**Interaction with residency (spec 03).**
- A claimed turn prefills only its new tokens.
- Their expert misses follow the scan-resistant admission: free slots first,
  then the staging ring. The decode working set of other lanes survives.
- Reuse does not warm the expert cache: the experts the new tokens touch are
  streamed as for any prefill. That is why the reused turn costs ~0.6-1.6 s
  and not the 27B's ~0.2 s.

**Determinism is a requirement, not a hope.**
- Bit-exactness against a cold prefill split at the same boundary needs a
  forward that is deterministic run to run for a fixed chunk split.
- The MoE kernels (spec 02) must not let atomic accumulation order or the
  expert-cache slot an expert sits in change any result.
- This spec's GPU test is the one that catches a violation. A non-deterministic
  kernel is fixed in spec 02's code, not tolerated here.

**Ownership.** Host slots, the KV-RAM arena and the retained-entry tables belong
to the Flash-Next model instance and are freed by its drop path. The 27B's
pinned host pool is a process singleton today, as spec 04 notes. This spec must
not reuse that singleton for Flash-Next. It is phase 2's job to remove it, and
Flash-Next's pools are created per model instance.

## Testing Decisions

Good tests check what the next layer observes: the tokens and logits a reused
request produces, where its reuse came from, what the plan charged, and what a
mismatched blob does. They never check copy order or internal table layouts.

- **CPU, in `cargo test`:**
  - Flash-Next's mutable-image and per-token section sizes from the topology,
    against the numbers in this spec;
  - the host plan with retained host slots and the KV-RAM arena, and the
    refusal below the margin with the line to shrink named;
  - blob identity: a 27B blob offered to a Flash-Next model is refused, and the
    reverse;
  - the existing reuse policy tests (match, lineage, first victim, KV-RAM
    ordering, budget exhaustion) run unchanged against a mock compute with
    Flash-Next's section sizes.

  Prior art: the core crate's prefix-reuse and host-tier tests, and the
  retained-host plan tests of #281.
- **GPU** (GPU profile, `--ignored`, fails and never skips when the card is
  busy):
  - **reuse against a split cold prefill:** turn N+1 claiming turn N's
    checkpoint generates the same tokens and logits as a cold prefill of turn
    N+1's prompt split at the same opener. It runs:
    - claimed from the device and restored from KV-RAM;
    - with a cold expert cache and a warm one;
    - in BF16 KV (the oracle format) and in hq-e8-2b;
    - with a history under 2051 tokens (dense QSA) and one over 8K (sparse
      QSA).

    Divergence against an unsplit cold prefill is recorded as information
    (ADR 0029).
  - **retained prefix:** a second request sharing the system block claims it
    after the first finished, and continues exactly;
  - **materialized blob round trip:** a Flash-Next sequence snapshot, spilled
    and restored, continues exactly as the sequence never moved.

  Prior art: the 27B's snapshot, prefix-reuse and transfer GPU tests.
- **Measurement** (GPU, run once at acceptance):
  - the agent-turn scenario, 30K history plus 1K new, on the 5090 with reuse on
    and with `--prompt-reuse off`;
  - a three-agent swarm replay reporting reuse sources, hit rates and
    per-turn TTFT.

  Prior art: the swarm driver and A/B script of #281.

## Acceptance

1. Flash-Next's mutable image (GDN fp32 state for 36 layers × 48 heads, conv taps, n-gram conv state, hq residual window, penalty row, position, last token) and its paged sections (KV in either format, indexer compressed keys) are derived from the topology. Their sizes are printed at load and checked by a CPU test.
2. ADR 0029's reuse works on Flash-Next unchanged: prompt checkpoints at the generation opener, a retained prefix at the system block, lineage with at most two checkpoints, non-consuming claims, first victim on the device, KV-RAM spill and restore with the restore floor. `--prompt-reuse off` disables it.
3. A reused request is bit-exact (tokens and logits) against a cold prefill split at the same boundary: from the device and from KV-RAM, with a cold and a warm expert cache, in BF16 and in hq-e8-2b KV, for a dense-regime and a sparse-regime history. A retained-prefix claim and a blob round trip continue exactly.
4. Retained slots default to host (proposed 8) with device 0. The KV-RAM arena defaults to a stated size (proposed 2 GiB). Both are host-plan lines, and the load refuses with the line to shrink named when the plan leaves less than spec 03's margin. No VRAM is taken from the expert cache unless device slots are asked for.
5. A 27B blob is refused under Flash-Next and a Flash-Next blob under the 27B, by identity.
6. Agent-turn TTFT on the 5090, 30K history plus 1K new, warm expert cache: ≤ 1.6 s with reuse. It is reported against the `--prompt-reuse off` number for the same turn and against the estimates (0.6-1.6 s with reuse; ~6.5 s without, under scan-resistant admission). A three-agent swarm replay reports reuse sources and per-turn TTFT.
7. The reused turn's prefill follows spec 03's scan-resistant admission. The other lanes' decode hit rate right after a reused turn is within 2 points of before it.
8. `request_done` carries the reuse source, reused tokens and restore time on Flash-Next. The per-tier reuse counters are exported and on the Monitor.
9. No process-wide singleton is added. Every reuse structure is owned by the Flash-Next model instance and freed by its drop path.
10. `cargo test` passes workspace-wide, and `cargo check --workspace --features cuda --tests` is clean. The Flash-Next reuse GPU tests and the 27B GPU profile are green on a free 5090.

## Out of Scope

- **The model switch (phase 2),** and anything that keeps retained state
  across it. A switch reloads everything, retained state included.
- **Sharing retained state between the 27B and Flash-Next.** Their identities
  never match.
- **KV-disk (Tier 2) for Flash-Next.** It is the one tier that could outlive a
  model switch, and is a candidate for phase 2's discussion, not for this spec.
- **`/v1/decide` reuse boundaries** (fan-out heads, observed forks, reuse
  markers): `/v1/decide` is not served on Flash-Next.
- **Multimodal reuse rules** (#193 placeholders, `rope_delta`): Flash-Next
  serves no images (spec 04).
- **Drafter sections:** no speculation on Flash-Next.
- **Warming the expert cache from a checkpoint** (prefetching the experts a
  conversation used last time). Possible later; the scan-resistant admission
  already keeps the decode set warm.
- **Changing ADR 0029's policy** for both models.

## Further Notes

- Prerequisites to `ready-for-agent`: ADR 0043 and ADR 0044 accepted or amended,
  spec 04 accepted (Flash-Next serving, hq-e8-2b, topology-driven sections) and
  spec 03's scan-resistant admission.
- Estimates come from `PREFILL_4K.md`, a GPU skeleton benchmark:
  - real pinned-to-device copies of the bytes the routing touches;
  - BF16 expert matmuls as a lower bound for the trellis kernel;
  - a non-expert proxy.

  The trellis kernel, the sparse attention path and the NVMe read rate were not
  measured, so the TTFT target leaves room for them.
- The ~124 MiB image is three fifths of the 27B's, mostly because Flash-Next
  has 36 GDN layers to the 27B's 48 and no drafter window. Host slots therefore
  cost less here than on the 27B, while the RAM they compete with is scarcer.
- Section sizes still to verify against the modeling code when the ticket
  starts:
  - the conv-tap storage dtype;
  - the indexer key dtype and its compression of the tail block;
  - the n-gram convolution's exact receptive field.

  The acceptance tests then check the derived values.
