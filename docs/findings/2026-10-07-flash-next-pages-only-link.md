# A pages-only link at the opener's page cuts a reused Flash-Next agent turn by 0.34-0.49 s and puts spec 05 acceptance 6 on its threshold

- Kind: experiment
- Status: current
- Observed: 2026-10-07
- Last verified: 2026-10-07
- Scope: serving / Flash-Next prompt reuse, the reused tail's traversals, where the opener's page is published (ADR 0029)
- Related: https://github.com/gpillon/ignis/issues/306 (item 5), [ADR 0029](../adr/0029-cross-request-state-reuse.md) (amendment 2026-10-07), spec [flash-next/05](../specs/flash-next/05-prompt-reuse.md) acceptance 3 and 6, [the agent turn tail](2026-10-07-flash-next-agent-turn-tail.md)
- Superseded by: none

## Question

[The agent turn tail](2026-10-07-flash-next-agent-turn-tail.md) left one lever: a reused turn's tail runs three traversals, and the middle one, from the opener's page floor to the opener, costs ~0.35 s at a median width. ADR 0029 as amended on 2026-10-07 removes it on Flash-Next: the checkpoint capture hands the pages below the opener over as a pages-only link instead of a prefix published at a cut of its own. Does that meet spec 05 acceptance 6 (≤ 1.6 s, 30K history plus 1K new, warm cache), does reuse stay exact, and what does it do to the cold turn?

## Evidence

**Setup.**
- Build: branch `fn-reuse-0029` at 3b1be3b (main c69d336 plus this change), release. Two binaries of the same tree:
  - *after*: Flash-Next's family switch on (`ModelFamily::opener_page_rides_capture`);
  - *before*: the switch off, the 27B's path, which cuts at the floor and publishes an imaged prefix there.

  The two executables differ in 25 bytes, timestamps included.
- Server: the agent-turn-tail finding's flags (`make config MODEL=flash-next`'s, `--kv-host-pool-bytes 1G`, `--no-ui --metrics`), port 8043.
- Client: that finding's `tail.py`, eight seeded conversations (seed 300), thinking off, greedy, `max_tokens` 24.
  - Turn 1 is cold.
  - Turn 2 adds the reply and a ~1.0K-token follow-up: the reused tail.
  - Turn 3 adds ~100 tokens.
- Legs: one load each, back to back, the same prompts in every leg.
  - ~9K history (`W1` 4100 words): before, after, after, before.
  - 30K history (`W1` 14500 words, 29.2-30.1K-token turn 1): after, then before.
- Scripts: `.scratch/ab/ab.sh` in the `fn-reuse-0029` worktree, and the main checkout's `.scratch/flash-next-306-307/harness/shorttail/` (`tail.py`, `summ.py`).
  `tail.py`'s `chunks` field assumes three pieces, so it is wrong for the after legs; the chunk counts below are the server's.

**~9K history** (medians; per leg: before 1.798 and 1.802 s, after 1.460 and 1.466 s):

| leg | turn 2 TTFT (range) | GB per tail | turn 3 TTFT | cold turn 1 TTFT |
|---|---|---:|---:|---:|
| before | 1.80 s (1.46-1.94) | 19.46 | 1.03 s | 4.00 s |
| **after** | **1.46 s (1.37-1.52)** | 15.40 | 0.80 s | 3.76 s |

- After beats before on 8/8 conversations in both pairs. The saving grows with the width of the piece that is gone: 1.456 → 1.411 s at 5 tokens, 1.846 → 1.369 s at 57.
- Turn 2's spread narrows from 0.47 s to 0.15 s.
- `ignis_expert_residency_stall_seconds_total{phase="prefill"}` per tail: 0.24-0.35 s before, 0.22-0.23 s after.
- In every leg every turn 3 claimed turn 2's checkpoint: cached = turn 2's prompt − 4.

**30K history** (acceptance 6):

| leg | turn 2 TTFT (range) | GB per tail | turn 3 TTFT | cold turn 1 TTFT |
|---|---|---:|---:|---:|
| before | 1.94 s (1.76-2.01) | 19.90 | 1.15 s | 10.54 s |
| **after** | **1.58 s (1.46-1.62)** | 15.51 | 0.88 s | 10.16 s |

- Three of the eight after-conversations are over 1.6 s: 1.607, 1.611 and 1.620 s.
- Every turn 3 claimed turn 2's checkpoint.

**Traversals, from the request log** (`ignis.request.admitted`'s `prefill_chunks_consumed`, every conversation of the first leg of each side):

| turn | ~9K before | ~9K after | 30K before | 30K after |
|---|---:|---:|---:|---:|
| 1 (cold) | 3-4 | 3 | 6 | 5 |
| 2 (the reused tail) | 3 | 2 | 3 | 2 |
| 3 | 3 | 2 | 3 | 2 |

A cold turn 1 at ~9K ran 3 chunks before wherever its opener's floor fell on the serving chunk's edge.

**The final code, in a later and noisier window** (21:10-21:30). The review moved the handover to a loan: the capture lends the pages and the link takes them when the sequence is released, so the device work of 3b1be3b's handover (a memset of the tail reservation, a page copy back) is gone. Same harness, one load per leg:

| leg | turn 2 TTFT median (range) | GB per tail | turn 3 TTFT | cold turn 1 TTFT |
|---|---|---:|---:|---:|
| ~9K, final | 1.585 s (1.55-1.67) | 15.70 | 0.88 s | 4.12 s |
| ~9K, 3b1be3b's binary (control) | 1.608 s (1.51-1.66) | 15.71 | 0.88 s | 4.16 s |
| ~9K, final | 1.490 s (1.41-1.52) | 15.71 | 0.82 s | 3.82 s |
| 30K, final | 1.598 s (1.49-1.80) | 15.78 | 0.90 s | 10.40 s |
| 30K, before | 2.092 s (1.84-2.17) | 20.27 | 1.23 s | 11.37 s |

- The same binary ran 1.46 s at ~9K in the first window and 1.61 s in this one, and the cold turn 1, which this change barely touches, moved with it: legs drift by about ±0.1 s between windows.
- At 30K four of the final code's eight conversations are over 1.6 s: 1.621, 1.632, 1.749, 1.804 s.
- Chunks per request in the final code's legs: turn 2 and turn 3 two, a cold turn 1 three, as above.

**Exactness** (GPU profile, `IGNIS_GPU_PROFILE=1`):
- `crates/runtime/tests/flash_next_reuse_gpu.rs`, BF16 and hq-e8-2b, histories of 1,500 and 9,000 tokens: the new shape is bit-exact (last logits and eight greedy tokens) against the two-span split control `[0, opener)`, `[opener, end)` for:
  - the capturing sequence, going on after its pages were handed over;
  - the checkpoint claimed from a host and from a device retained slot;
  - the checkpoint restored from KV-RAM;
  - turn N+1, claiming it and handing its own pages over: it goes on as its split control does, and so does a claimant of its checkpoint, on a link chained over a link;
  - at 1,500 tokens, the reference run first on the fresh load's cold expert cache and again at the end on the warm one: the same bits.
- `crates/server/tests/flash_next_serving_gpu.rs`, the scheduler the server builds for Flash-Next (the switch on): a second turn resumes from the first's checkpoint, 2,500 tokens, on the device.
- Recorded as information (ADR 0029): the two-span control against the three-span one it replaces. Logits differ in all four runs. The first greedy token parts at index 1 (BF16) and 0 (hq-e8-2b) at 1,500 tokens, and never at 9,000. The prompts are synthetic token ids.
- The leaf: `ignis_kernel_seq_checkpoint_test` (a loan from no prefix, from a two-page prefix, at a page-aligned opener; a second capture standing on the loan; both formats; device and host slots; a fault injected at the capture's commit point, for a loan, a capture on a loan and a capture on a prefix, leaves the sequence, its row, its state and the pool unchanged), `seq_flash_next_sections` (the partial page's indexer keys), `seq_prefix`, `seq_snapshot`, `seq_alloc`.
- The 27B, whose scheduler never hands pages over but whose prefix publish now goes through the same handover code: `prompt_checkpoint_gpu`, `prefix_reuse_gpu` and `retained_prefix_gpu` green.

## Finding

Observed:

1. **Dropping the floor's traversal saves 0.34 s** at the median of a ~9K-history reused turn (1.80 → 1.46 s) and **0.36-0.49 s at 30K** (1.94 → 1.58 s in one window, 2.09 → 1.60 s in another). The agent-turn-tail finding had inferred ~0.35 s.
2. **Spec 05 acceptance 6 sits on its threshold.** At 30K the median is 1.58 s in the first window and 1.598 s with the final code in the later one, against 1.6 s; three and four conversations of eight are over it. At ~9K the medians are 1.46-1.61 s.
3. **The tail moves 21% fewer expert bytes**: 19.5 → 15.4 GB.
4. **Turn 3 gains 0.24-0.27 s and the cold turn 1 gains 0.23-0.37 s.** Their prefills were cut at the opener's floor too.
5. **Reuse stays bit-exact** against a cold prefill split at the new boundaries, and the 27B's reuse GPU tests are unchanged.

Inferred:

- **Why the spread narrows.** The piece removed was the one whose cost scaled with the opener's offset in its page. What remains varies with the tail's length (845-1,027 tokens here).
- **Why turn 1 moved.** A cold prompt's last chunk was cut at the floor as well, so it too ran one traversal more than it needed.

## Implications

- **Acceptance 6 holds by 2-20 ms at 30K, inside the ±0.1 s windows drift by.** Anything that adds to the tail's copies breaks it. The levers left are the prefill-width sweep (agent-turn-tail follow-up 2) and the 4-token opener piece (~50-70 ms).
- **A pages-only link takes no retained slot.** A turn now holds one of Flash-Next's eight host slots instead of two, and skips one ~124 MiB image copy. Neither was measured separately.
- **The 27B keeps the imaged prefix** (ADR 0029 amendment). The same switch would save its ~19 ms traversal and a retained slot per turn. Unmeasured: a candidate for the measured-better-is-default rule.

## Limits and unknowns

- One harness: README-word prompts, thinking off, no swarm. At 30K, one leg per side.
- `--prompt-reuse off` was not run. The cold turn 1 (10.2-10.5 s at 29-30K tokens) stands in for the reuse-off number acceptance 6 asks for.
- Acceptance 6's three-agent swarm replay was not run (skipped by the coordinator's decision).
- Windows drift by about ±0.1 s: the same binary measured 1.46 and 1.61 s at ~9K an hour apart. Each comparison above is between legs of one window.
- *Before* is this tree with the family switch off, not main's binary. The first window's numbers are 3b1be3b's binary; the final code's are in the later window only.
- No profile was taken: the traversal counts are the request log's chunk counts, not a trace.

## Follow-ups

1. Measure the switch on the 27B.
2. The prefill lookahead width sweep (6, 8): acceptance 6 has no margin at 30K.
3. Acceptance 6's swarm replay and its `--prompt-reuse off` leg.
