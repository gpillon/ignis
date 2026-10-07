# A wider prefill chunk buys the 27B at most 4% of cold TTFT and costs decoding lanes up to 5.6x their longest stall

- Kind: experiment
- Status: current
- Observed: 2026-10-07
- Last verified: 2026-10-07
- Scope: serving / 27B prefill chunk width, prefill/decode interleaving (decode share), VRAM plan; kernel / CUDA-graph prefill feasibility
- Related: [#92](https://github.com/gpillon/ignis/issues/92) (criteria 2-4), [Prefill chunk wall time](2026-09-11-prefill-chunk-wall-time.md) (criterion 1), [The prefill "other" class](2026-09-30-prefill-other-class.md), [Decode round anatomy](2026-09-18-decode-round-anatomy.md), [Flash-Next prefill chunk and the decode share](2026-10-07-flash-next-prefill-chunk-and-decode-share.md), [ADR 0018](../adr/0018-chunk-level-prefill-decode-interleaving.md), [ADR 0019](../adr/0019-decode-cuda-graph-slot-indirection.md), [ADR 0020](../adr/0020-batch-wide-decode-round.md), [ADR 0037](../adr/0037-a-vendored-file-may-carry-a-correctness-patch.md), [#272](https://github.com/gpillon/ignis/issues/272), `.scratch/issue-92-2026-10-07/`
- Superseded by: none

## Question

Criterion 1 of #92 ([finding](2026-09-11-prefill-chunk-wall-time.md)) found a
1,024-token chunk 90% compute, 9.3% launch/dispatch and 0.5% synchronization.
It also found per-token cost 1.77x lower at 1,024 tokens per traversal than at
256. The 27B still serves at `PREFILL_CHUNK ?= 1024` (`mk/config.mk`), while
Flash-Next uses 8,192. On current `main`:

- Does a wider chunk pay on the 27B: prefill throughput and TTFT at 8K and 32K?
- What does it cost: the VRAM the wider workspace takes from the KV pool at the
  make default context (the owner's floor is 262,144 tokens), and the pause a
  concurrently decoding lane sees while a chunk runs?
- How does the decode share interact with the width?
- Is a CUDA-graph prefill chunk feasible, and what could it recover?

## Evidence

**Setup.**
- Build: release, `main` at 6722886 (branch `prefill-92`, no source change).
- Server: `make start CUDA=1 UI=0 PREFILL_CHUNK=<W>` with every other make
  default (524,288 context, hq-e8-2b, `yarn:2`, DFlash2 with 7 drafts, prompt
  reuse on, 16 host retained slots, 8 GiB KV-RAM). One load per cell.
- Card: exclusive under the GPU lock. CPU 14% mean, 25% max during the
  measured run (`typeperf`, 141 samples).
- Order: 1024, 2048, 4096, 8192, 1024 again, then 1024 with `--decode-share 50`.
  A later run (same day, `main` at cb6f47d, the stall probe only) added 1024
  at the new default share of 25 (`w1024s25.stall.*`).

**Per load, four readings** (scripts in `.scratch/issue-92-2026-10-07/`, driver `cell.sh`):

1. The `ignis.runtime.vram_plan` and `ignis.runtime.kv_pool` lines.
2. `ignis-bench ttft --cells 8192 --corpus <ninfer bank>`: 5 cold samples after a
   warm-up.
3. `ttft27.py`, 8K and 32K: 5 cold samples each after a warm-up. Every prompt is
   word soup from its own seed, and `cached_tokens` was 0 on every sample.
   - Why not `ignis-bench ttft` at 32K: its corpus prompts share their first
     8,128 tokens between the 8K and the 32K cell, so under prompt reuse 4 of 5
     32K samples came back void.
   - Its own generator takes minutes per 32K prompt.
4. `stall27.py` (#306's `stall.py`, adapted).
   - Two lanes stream long generations with `ignore_eos`. Five seconds after
     both decode, a third request sends a 32.7K prompt. Three reps, plus the
     prompt alone first.
   - Gaps are taken between SSE content events. In these runs the lanes
     committed one token per round: before the prompt, gaps were evenly
     18.0 ms (p95 19.0-19.5). So a token gap here is a round gap.

Traversals per request come from `ignis.request.admitted`'s
`prefill_chunks_consumed`.

**VRAM at the make default context** (`free_at_start` 31.17 GB at 1024-4096
and 31.20 GB at 8192, so its pool is ~29 MB larger than the workspace alone
explains):

| width | workspace (bytes) | vs 1024 | KV pool (bytes) | pages | token capacity | >= 262,144 | >= 524,288 (the plan's floor) |
|---:|---:|---:|---:|---:|---:|---|---|
| 1024 | 1,508,311,552 | | 7,578,320,896 | 12,848 | 822,272 | yes | yes |
| 2048 | 1,920,765,184 | +412 MB | 7,166,033,920 | 12,149 | 777,536 | yes | yes |
| 4096 | 2,459,520,256 | +951 MB | 6,625,165,312 | 11,232 | 718,848 | yes | yes |
| 8192 | 3,359,493,376 | +1,851 MB | 5,756,354,560 | 9,759 | 624,576 | yes | yes, by 100,288 tokens (0.92 GB) |

**The prompt alone** (medians, min-max in brackets, ms; "vs 1024" against the
mean of the two 1024 legs):

| width | traversals 8K / 32K | 8K TTFT (ignis-bench) | 8K tok/s | 32K TTFT (`ttft27`) | 32K tok/s | vs 1024, 8K / 32K |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 (first) | 10 / 34 | 854.2 (853.1-860.8) | 9,590 | 4,045.0 (4,010.3-4,058.9) | 8,091 | |
| 2048 | 6 / 18 | 842.1 (841.0-844.6) | 9,728 | 3,953.0 (3,935.0-3,997.4) | 8,279 | -1.4% / -2.2% |
| 4096 | 4 / 10 | 853.4 (851.0-858.9) | 9,599 | 3,885.4 (3,847.3-3,899.5) | 8,423 | -0.1% / -3.8% |
| 8192 | 3 / 6 | 916.2 (912.6-919.2) | 8,941 | 3,900.4 (3,812.0-3,933.6) | 8,391 | +7.2% / -3.5% |
| 1024 (last) | 10 / 34 | 854.7 (853.7-857.9) | 9,584 | 4,036.6 (4,015.5-4,060.7) | 8,108 | |

- `ttft27`'s 8K cell agrees in shape: 874.1, 879.9, 879.7 and 937.5 ms, and
  894.1 for the last 1024 leg.
- Reuse adds two traversals at every width: the prompt is cut at its publish
  point and at the generation opener.

**A 32.7K prompt while two lanes decode** (lanes at 54.6 tok/s each before the
prompt; times in s and ms):

| width | prompt TTFT alone | with 2 lanes (3 reps) | lanes' tok/s during | ITL p95 during | longest gap |
|---:|---:|---:|---:|---:|---:|
| 1024 | 4.054 / 4.020 | 4.645-4.670 / 4.641-4.696 | 8.9 / 8.9 | 168.9 / 168.5 | 231.3 / 234.3 |
| 2048 | 3.957 | 4.276-4.295 | 5.9 | 315.4 | 387.0 |
| 4096 | 3.835 | 4.030-4.142 | 4.3 | 637.7 | 702.0 |
| 8192 | 3.817 | 3.890-4.046 | 3.4 | 1,159.6 | 1,294.2 |
| 1024, `--decode-share 50` | 4.036 | 7.991-8.098 | 28.5 | 144.5 | 233.3 |
| 1024, `--decode-share 25` (later run, the new default) | 4.057 | 5.544-5.565 | 16.9-17.0 | 159.1-161.4 | 207.6 |

**The decode share on the 27B.**
- The 27B has `--decode-share` (#306); `ModelFamily::default_decode_share_percent`
  was 0 for it (ADR 0018's one decode round per chunk); the owner made it 50
  after this measurement, then 25 on both models by owner decision 2026-10-07
  (a reader reads ~5-8 tok/s), and `--decode-share 0` restores 0.
- After a chunk that took `t`, a share `s` holds the next chunk until the
  decoding lanes have had `t * s / (1 - s)`, and only while a lane decodes
  (`hold_prefill`, `crates/core/src/concrete.rs`).

**One contended leg, discarded.** Another agent's `cargo test --workspace` ran
during a 1024 leg. That leg's 8K median was 999.4 ms (+17%) and its 32K samples
4,130-4,551 ms. It is kept in `.scratch/issue-92-2026-10-07/aborted/run3-contended/`.

## Finding

Observed:

1. **Cold TTFT barely moves with width.**
   - At 32K: 2048 saves 2.2%, 4096 saves 3.8%, 8192 saves 3.5%.
   - At 8K: 2048 saves 1.4%, 4096 nothing, and 8192 costs 7.2%.
   - The two 1024 legs, first and last, agree within 0.1% at 8K and 0.2% at
     32K, so the differences are real but small. 8192's 8K loss shows in the
     server's own admission-to-first-token durations too (890-921 ms against
     831-852 at 1024).
2. **The decoding lanes pay in proportion to the width.**
   - With the 27B's share of 0, each traversal is followed by exactly one
     round, so a lane sees one long gap per traversal: 34 at 1024, 10 at 4096,
     6 at 8192 (one rep, `stall.jsonl`). Each gap grows with the prefix the
     traversal attends to: 101 to 170 ms across 1024's full chunks, 336 to
     570 ms at 4096, 695 to 984 ms at 8192.
   - The longest gap is 231-234 ms at 1024, 387 at 2048, 702 at 4096 and
     1,294 at 8192 (5.6x): about 2x the gap after the prompt's first
     traversal at every width (98-101, 174-185, 336-338 and 690-695 ms over
     all reps and lanes). It follows the traversal
     that ends at the prompt's publish point, ~55 ms above the full chunks'
     trend at 1024. Why that one is longer is not measured (see Inferred).
   - The lanes' rate during the prefill falls from 8.9 tok/s (16% of their
     54.6) to 5.9, 4.3 and 3.4.
3. **The larger gain under load is the lanes' time moving to the prompt.**
   - With two lanes decoding, the prompt's TTFT falls 8% at 2048, 12% at 4096
     and 15% at 8192 against 1024's 4.66 s.
   - Under load the prompt pays about one 18-25 ms round per traversal: 0.62 s
     over 1024's 34 traversals, 0.13-0.15 s over 8192's 6. A wider chunk means fewer
     traversals, so fewer rounds; prefill itself is no faster than in item 1.
4. **The wider workspace comes out of the KV pool.**
   - It grows by 412 MB, 951 MB and 1,851 MB. The pool loses 44,736, 103,424
     and 197,696 tokens.
   - Every width keeps at least 262,144 tokens and the plan's 524,288 floor.
     8192 keeps the floor by only 0.92 GB, so a desktop holding a little more
     VRAM at start would have the make default refused.
5. **The decode share and the width are orthogonal.**
   - At 1024 with a share of 50, the lanes keep 28.5 tok/s (52% of their rate)
     and the prompt's TTFT under load doubles (8.0-8.1 s against 4.04 s alone),
     as the hold's formula predicts.
   - The longest gap does not change (233 ms).
   - The share sets the lanes' rate above a floor of one round per chunk. The
     width sets that floor and the gap.

Inferred:

- **The longest gap's excess.** The publish-point traversal also captures the
  prompt's checkpoint into a host retained slot and, with the slots full,
  evicts one; the 2026-09-30 capture puts those at ~25 and ~30 ms, which
  matches the ~55 ms. Not measured here.
- **Why the curve is flat**, a consistent reading that this sweep cannot
  separate from per-token costs:
  - Fewer traversals save a fixed per-traversal cost of ~5 ms. The 2026-09-30
    capture, taken after the A16 route, attributes ~3.3 ms to launch gaps and
    ~1.5 ms to TMA descriptor uploads per traversal ([finding](2026-09-30-prefill-other-class.md)).
  - A wider traversal costs more per token: criterion 1's isolated-span sweep
    gave 0.0865, 0.0903 and 0.0958 ms per token at 1024, 4096 and 8192.
  - At 8K, 1024 to 4096 saves 6 traversals (~30 ms) and adds ~31 ms of
    per-token cost (8,192 x 3.8 us). At 8192 the per-token rise wins.
- **CPU contention reaches the TTFT.** The eager route enqueues ~1,200 kernels
  per traversal from the host (57 ms of enqueue per 98 ms chunk in September).
  A busy CPU can push that onto the critical path. The contended leg's +17% is
  one observation of it.

### CUDA-graph prefill (criterion 3)

Inferred throughout: a reading of the code named below plus arithmetic on
the 2026-09-30 and 2026-09-18 captures. Nothing here was built or measured.

**The bound.**
- ~4.75 ms of fixed idle per traversal at 1024 (launch gaps plus TMA uploads),
  less a replay's own submission. The decode round finding measures
  `cudaGraphLaunch` at 0.36-0.73 us per node, so ~0.4-0.9 ms for ~1,200 nodes.
- Net, ~4 ms per traversal:
  - **~40 ms of an 8K TTFT (4.5-5%) and ~130-150 ms of a 32K one (3-3.5%)** at
    1024;
  - roughly half that at 2048.
- September's whole 9.2 ms of idle per traversal, including the gaps over
  100 us, is the ceiling: 11% at 8K and 8% at 32K.
- Graphs and width compete for the same per-traversal cost. The wider the
  chunk, the less a graph buys.
- The reference engine (`F:/ai/q38/ninfer`) captures decode, verify and MTP
  rounds (`DecodeGraphDefinition`) and runs its prefill eagerly.

**Feasibility.** Feasible only with work in every layer body. What a capture
would freeze, from `run_program_chunk` (`kernel/src/step.cu`) and the bodies it
calls:

- **Host scalars baked into kernel arguments per chunk:**
  - `fill_i32_positions(positions, start_position)` in the GQA body;
  - the GDN body's state `slot` (the decode graph body reads it from device
    staging instead, `kernel/src/gdn_layer.cu`);
  - the drafter tap scalars uploaded from a host vector;
  - the `chunk_offset == 0` pending append;
  - the multimodal scatter and armed attention readouts.

  Decode solved the same problem with graph-safe bodies reading device staging
  (ADR 0019, ADR 0020).
- **The hq prompt attention grows with the prefix.**
  `max_visible_keys = start_position + tokens` sizes its workspace and the
  history it materializes, so its launch shape changes every chunk. A graph
  needs either per-replay kernel-node updates for the 16 GQA layers, or one
  graph per prefix band.
- **The vendored Windows TMA launcher** (`nvfp4_w4a4_tma.cu`,
  `nvfp4_linear_swiglu_w4a4_tma.cu`) allocates a descriptor block and copies a
  stack-local descriptor per call. A captured copy would replay from a dead
  stack address. Fixing it is a vendored change: a recorded patch under ADR
  0037 if correctness-only, or an optimisation change to vendored code.
- **Not every traversal is full width.** The ragged last chunk and the reuse
  cuts (10 traversals where 8 would do at 8K, #272) have widths of their own,
  so they run eagerly or need more graphs.
- **What does not block it:** the P2-02 / #84 failure detection. A replay is
  still followed by the chunk's one synchronize.

**Verdict: not worth it on current evidence; bound ~5% (8K) / ~3.5% (32K) at 1024.**
- The bound is inferred from the 2026-09-30 capture (source reading plus
  arithmetic), not measured on this `main`; September's 9.2 ms of idle per
  traversal, the 9.3% of criterion 1, would raise it to 11% and 8%.
- The price is graph-safe prefill bodies for 64 layers, per-chunk attention
  updates and a vendored change.
- The TMA part (~1.5 ms per traversal, ~1.5%) needs no graph.
- A graph would also shield prefill from host contention, which matters only if
  the serving host is often busy (one observation, above).

## Implications

- **Keep the 27B's default chunk at 1024** (`PREFILL_CHUNK ?= 1024`,
  `DEFAULT_PREFILL_CHUNK = 1024`): no width is measured-better (Finding items
  1, 2 and 4). This holds for the shape measured: one long prompt against two
  decoding lanes that commit one token per round.
- **The decode share, not the width, is the 27B's knob for prompt-versus-lanes
  priority.** The width moves priority as a side effect (one round per chunk)
  and lengthens the gap. The share moves priority alone, but only toward the
  lanes. Prompt-first beyond one round per chunk is what a wider chunk buys,
  through fewer chunks.

## Limits and unknowns

- **Sample sizes:**
  - one prompt family (word soup);
  - one load per width;
  - 5 samples per TTFT cell and 3 reps of the stall probe.

  Run-to-run spread is known only from the 1024 bracket. The +7.2% at 8192 on
  8K is one load's result (its 5 samples span 912.6-919.2 ms).
- **Two decoding lanes only.** With more lanes a round costs more, so every gap
  grows by the round, and the share of time rounds take under load grows too.
- **Speculation committed one token per round on these lanes.** With real
  acceptance tokens arrive in bursts, but the gap per round, which the chunk
  sets, is the same.
- **The per-traversal costs are not re-captured.** The ~4.75 ms comes from a
  2026-09-30 nsys capture (~5.5K prompt, width 1024). The flat 8K curve agrees
  with ~5 ms, but cannot separate it from per-token costs.
- **Widths not tried:** 512, 1536, 3072.
- **Load shapes not tried:** a mix of long prompts, or more than one
  concurrent prefill.
- **The 8192 margin** depends on the desktop's VRAM at start: 31.17-31.21 GB
  free here.
- **Contention:** one leg.
- **Out of scope here:** criterion 1's 1.77x was 256 against 1,024 tokens per
  traversal, which packing short prompts addresses and width does not; and the
  09-30 capture's retained-slot cost (~75 ms per ~5.5K request) is a larger
  cold-prefill lever than a graph.

## Follow-ups

- At most one implementation ticket, optional at ~1.5%: **the Windows TMA
  GEMM's descriptor without a per-call allocation and pageable upload.**
  - The change: a descriptor per call site and width made once (at load or on
    first use) and kept on the device, or all of a traversal's descriptors
    uploaded in one copy. It is a vendored change: a recorded patch if
    correctness-only (ADR 0037), otherwise an optimisation change.
  - Expected gain: ~1.5 ms per prefill traversal, ~15 ms of an 8K and ~50 ms of
    a 32K cold TTFT at 1024.
  - Blocked by nothing. It would be the first step of any later prefill graph,
    which it unblocks for the TMA GEMMs.
- Owner decision, not a ticket: should the 27B get a non-zero default decode
  share (the share-50 row: lanes 8.9 -> 28.5 tok/s, prompt TTFT x2 under load)?
