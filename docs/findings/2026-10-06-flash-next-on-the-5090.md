# Flash-Next on the 5090: what the built engine serves

- Kind: experiment
- Status: current
- Observed: 2026-10-06
- Last verified: 2026-10-06
- Scope: serving / Flash-Next quality, decode and prefill speed, prompt reuse, VRAM and host memory, load time
- Related: https://github.com/gpillon/ignis/issues/304, https://github.com/gpillon/ignis/issues/302, https://github.com/gpillon/ignis/issues/303, https://github.com/gpillon/ignis/issues/300, [spec 04](../specs/flash-next/04-the-flash-next-forward-and-serving.md), [spec 05](../specs/flash-next/05-prompt-reuse.md), [spec 06](../specs/flash-next/06-measurements.md), [expert residency replay](2026-10-05-expert-residency-replayed-on-the-study-s-routing.md), [expert miss path](2026-10-05-expert-miss-path-sm-copy-matches-the-copy-engine.md), [MoE decode is structure-bound](2026-10-05-moe-decode-is-structure-bound.md)
- Superseded by: none

## Question

The compression study predicted Flash-Next's behaviour on this machine from
simulations. This finding records what the built engine actually does when
ignis serves it: quality against the checkpoint, decode and time to first
token, prompt reuse, memory and load time. Each number is set against its
estimate or its spec floor.

## Evidence

**Setup.**
- Build: branch `flash-next` at `00a8521`, release profile, CUDA.
- Artifact: `qwen3_8_flash_next_trellis_a25-v2.ninfer`, 71.8 GB.
- Host: RTX 5090 on a PCIe Gen 3 x16 link (~12-13 GB/s, measured in the expert miss path finding). 63.8 GB of RAM, with the desktop and agent sessions running.
- Server flags: the ones `make config MODEL=flash-next METRICS=1 UI=0` prints (hq-e8-2b KV, `--max-context 131072`, `--prefill-chunk 8192`, `--metrics`), with one change: `--kv-host-pool-bytes 2G` in place of make's `8G` (see Memory).
- No other process was on the card. The GPU lock was held for the whole campaign.

**Quality.** These were measured by `flash_next_acceptance_gpu` on BF16 KV, then on hq-e8-2b
(fnval's acceptance run, the same artifact):
- **G1** (teacher-forced argmax against the quantized reference): 102/102 = **100%** (floor 95%).
- **KLD on the 2048-token windows**: all 11 domains are under the limit (engine / quantization-only / limit, nats):

  | domain | engine | quant-only | limit |
  |---|---:|---:|---:|
  | chat | .04798 | .04941 | .05941 |
  | code | .10836 | .10743 | .11817 |
  | de | .14323 | .14081 | .15489 |
  | en | .09891 | .09840 | .10840 |
  | it | .04592 | .04626 | .05626 |
  | ja | .12257 | .12220 | .13442 |
  | math | .04450 | .04554 | .05554 |
  | mmlu | .11381 | .11637 | .12801 |
  | prose | .15344 | .14970 | .16467 |
  | py | .02595 | .02661 | .03661 |
  | zh | .07115 | .07129 | .08129 |

- **KLD on the 8192-token windows** (sparse QSA path): code .09008 / .09200 / limit .10200; prose .11860 / .11895 / limit .13084.
- **hq-e8-2b KV** (reported, not judged): identical to BF16 on every 2048 window, by construction (one prefill chunk reads only fresh rows). On the 8192 windows: code .11862, prose .14742. That is **+0.029 nats** over BF16 KV and above the BF16 limit, from the codec's error on rows read across chunks.
- **MMLU-Pro proxy**: **70.11%** (197/281), floor 71%. This is exactly the converter's quantized-reference figure.
  - Paired against that reference: lost 5, gained 5, p 1.00.
  - Paired against BF16 (73.67%): lost 17, gained 7, p 0.064. The converter itself measured 19/9, p 0.087.

**Decode.**
- Method: a `.scratch` client (`fnperf.py decode`). It sends streaming chat requests with
  `ignore_eos`, 256 generated tokens, at the server's default sampling, after two
  three-lane warm-up passes.
- Per-lane rate: (tokens − 1) / (last − first token).
- The "total" column counts every lane's tokens inside the window where all lanes
  decode at once.
- One token arrived per SSE chunk (chunks = `completion_tokens` in every sample).
- G3's fixed cells (8192 / C=1, C=4) were not used: the spec asks for a ≤ 2K prompt at 1/2/3 lanes.

| context | lanes | total tok/s (2 reps) | floor | simulation |
|---|---:|---:|---:|---:|
| 2.1K | 1 | **65.3 / 65.2** | 70 | 102 |
| 2.1K | 2 | 106.3 / 106.1 | — | — |
| 2.1K | 3 | **131.2 / 129.5** | 130 | 186 |
| 31.6K | 1 | 60.9 | — | — |
| 31.6K | 3 | 131.8 (17.4 s overlap, 2,300 tokens; these lanes resumed retained prefixes, TTFT 0.5-0.7 s) | — | — |
| 122.8K | 1 | 63.3 | — | — |

- Decode-phase expert-cache hit rate: 99.0-99.2% in every cell except the one-lane 32K cell (97.5%).
- PCIe traffic per decode token (`ignis_expert_bytes_moved_total{phase="decode"}` / tokens): **31 MB at one lane** at 2K and 54 MB in the one-lane 32K cell (the cell whose hit rate dipped: long context costs link bandwidth, not decode rate), 19-23 MB at three lanes. The replay predicted 96.5% hits and 23.8 + 39.2 MB/token at one lane.
- The 27B on the README's harness, which is different (speculative decoding with DFlash2, which Flash-Next does not have): 367 tok/s at one lane on coding prompts, 167 tok/s on free prose, 1,065 tok/s aggregate over eight lanes.

**Time to first token.**
- Prompts have no reused prefix, so every prompt is prefilled.
- The expert cache is warm unless the line says otherwise.
- The 2048 and 8192 cells come from the repo's `ignis-bench ttft` (exact lengths; the server's `ignis.request.admitted` duration). The rest come from `fnperf.py ttft`, measured at the client.

| prompt | TTFT | estimate |
|---|---:|---:|
| 1.1K | 2.11, 2.23, 2.71 s | — |
| 2,048 | **3.38 s cold cache** (first request after load); 3.12-3.15 s warm | — |
| 4.5K | 4.36, 4.50 s | 2.65 s for 4K, cold cache (`PREFILL_4K.md`) |
| 8,192 | 6.94, 8.30, 8.36, 8.49 s | — |
| 9.3K | **8.52 s cold cache** (first request after load 2); 8.25 s | — |
| 32.7K | **25.7 s** | — |
| 38.2K | 30.4, 30.5 s | — |

**What a prefill moves.** Each prefill moves about the whole non-resident expert pool host-to-device:
- 1.1K prompt: 17.1 GB;
- 32.7K prompt (four 8192-token chunks): 59.2 GB.

The pool is 37.8 GB and the cache 20.6 GB, so 17.2 GB is not resident. At the link's 12-13 GB/s that is at least 1.3 s per chunk, whatever the chunk's length. The prefill hit rate is 93-94%.

**Agent turn** (spec 05 AC6; `fnperf.py agent`, 2 reps):
- The history is a 33.0K-token tool loop. The new user message adds 1.1K tokens.
- Turn 1 is primed with 16 generated tokens, then turn 2 is timed. Warm cache.

| | turn 2 TTFT | cached tokens | estimate |
|---|---:|---:|---:|
| `--prompt-reuse` on (default) | **2.45 / 2.48 s** | 33,032 / 33,052 | 0.6-1.6 s, target ≤ 1.6 s |
| `--prompt-reuse off` | **28.19 / 28.17 s** | 0 | ~6.5 s |

**Decode hit rate around a reused turn** (spec 05 AC7; `fnperf.py hitrate`):
- Two lanes decode long streams while a third lane runs the reused 34.1K-token turn (33.0K cached, TTFT 2.90 s).
- The decode-phase hit rate is 98.85% in the 5 s before the turn and **99.11%** in the 5 s after it.

**Three-agent swarm** (`fnperf.py swarm`; there is no driver for #281 in the repo):
- Setup: three concurrent agents, five turns each, a shared 1.8K-token system prompt. Each turn adds ~2.4K tokens of tool output and generates up to 128 tokens.
- Per-turn TTFT: 3.3-3.8 s when the turn's prefill runs alone, 6.7-7.8 s when it queues behind another agent's.
- Reuse sources:
  - From turn 3 on, every turn resumed the whole previous prompt (cached = previous prompt − 4).
  - Every agent's second turn resumed only the shared 1,792-token system prefix.
  - The first turn of agents 2 and 3 resumed that prefix from agent 1.
- Per-lane decode was 10-19 tok/s during the swarm, against 44 tok/s with no concurrent prefill.
- Decode hit rate 94.8%, prefill 93.8%.

**Memory.**
- **Default changed after this measurement (9695309):** `make ... MODEL=flash-next` now passes `--vram-headroom-bytes 4G` (process + desktop ≤ 32.6 − 4 GB, under AC10's 29 GB) and `--kv-host-pool-bytes 2G`. The budget below was the old default (free − 1G); at the new one the expert cache is ~3 GB smaller (~17.6 GB instead of 20.6), so the decode hit rate and tok/s in this finding were taken with a larger cache than the default now gives. Not re-measured.
- **VRAM plan** (`ignis.runtime.flash_next_plan`):

  | line | bytes |
  |---|---:|
  | budget | 31,529,590,784 |
  | expert cache | 20,612,945,232 |
  | weights | 5,022,463,744 |
  | program | 1,810,558,704 |
  | residency fixed | 1,577,852,928 |
  | KV pool | 2,053,174,464 |
  | retained device | 0 |

  The plan is for 3 lanes. The sequence state is 4,224 B/token paged (KV 3,456, indexer keys 768) plus a 130,014,272-byte state image.
- **Host plan** (`ignis.runtime.flash_next_host_plan`):

  | line | bytes |
  |---|---:|
  | available | 52,190,724,096 |
  | expert pool | 37,795,446,784 |
  | n-gram hot rows | 1 GiB |
  | retained host (8 slots) | 1,040,115,712 |
  | KV-RAM arena | 2 GiB |
  | planned | 42,056,787,968 |
  | left | 10,133,936,128 |

  Process working set at the end: 40.6 GB.
- **The make default does not start.** With make's default `--kv-host-pool-bytes 8G` the load was refused: "the host plan needs 48,499,238,912 bytes … of the 52,342,587,392 available, and must leave 6 GiB".
- **Peak VRAM, sampled every 500 ms** with `nvidia-smi` across load and every cell:
  - 1,080 MiB with the desktop before load;
  - **31,190 MiB at ready**, and 31,210 MiB at most afterwards.
  - It was flat (±20 MiB) through 1-3 lanes, 32K and 122.8K prompts, and the swarm.
- **N-gram table**:
  - Hot-row hit 33.6% at one lane (12.7K hot rows / 25.1K file rows per 2.1K + 256-token request); 29.2% over the swarm.
  - NVMe: ~20K reads (84 MB) per such request, about 2.7K reads/s and 11.5 MB/s at one lane.

**Load** (`.scratch/fnperf/start.ps1`: process start to the first `200` from `/v1/models`):
- **112.4 s** and **111.9 s**, two back-to-back loads.
- The artifact is verified 1 s after start and the plans are printed at 1.5 s. The round graphs are ready at 111 s.
- An unbuffered read of 3.2 GB of the artifact runs at 1.64 GB/s.

Raw records: `.scratch/fnperf/` in the `flash-next` worktree (untracked):
- `decode-*.jsonl`, `ttft-*.jsonl`, `agent-reuse-{on,off}.jsonl`, `hitrate.jsonl`, `swarm.jsonl`;
- `http-surface.txt`, `metrics-load1-end.txt`, `vram.log`;
- server logs `load{1,2}.out.log`, `load0-refused-8G.err.log`.

## Finding

Observed:

1. **Quality.** G1 and every per-domain KLD on BF16 KV meet spec 04. The MMLU miss belongs to the artifact, not the engine: the engine matches its quantized reference 5/5, p 1.0. The owner accepted it. hq-e8-2b, the serving default, costs +0.03 nats on 8K windows.
2. **Decode** is flat across context: 61-65 tok/s at one lane from 2K to 123K. Three lanes give about 130 tok/s total at 2K and 32K.
   - **Spec 04 AC9 misses the one-lane floor** (65.3 against 70) and only just meets the three-lane floor (129.5-131.8 against 130).
   - The engine reaches 64% / 70% of the simulation's 102 / 186.
3. **Prefill is the bottleneck of the whole serving picture.**
   - About 0.8 ms per token, plus a fixed ~1.5 s per chunk.
   - The fixed part matches the ~17 GB of non-resident experts that each chunk moves over a 12.5 GB/s link.
   - A 4.5K prompt takes 4.4 s against the study's 2.65 s (4K, cold), and 32K takes 25.7 s.
4. **Prompt reuse works and is worth 11.5x on an agent turn** (2.45 s against 28.2 s). Even so, it **misses spec 05 AC6's 1.6 s**: the 1.1K-token tail is still one prefill chunk, and it pays the same expert stream as a cold 1K prompt (2.1-2.7 s).
5. **Spec 05 AC7 passes.** The other lanes' decode hit rate after a reused turn is +0.26 points from before it (bound: within 2).
6. **A prefill chunk stalls the other lanes' decode.** At 32K, two lanes fell to 4.5 and 8.2 tok/s while a third lane prefilled. In the swarm, per-lane decode fell to 10-19 tok/s. An 8192-token chunk runs ~7 s and the decode rounds wait for it. (Since measured, 2026-10-07: ~3.2 s per chunk, one round per chunk, lanes at ~1 tok/s; the decode share now gives them half: [the prefill chunk finding](2026-10-07-flash-next-prefill-chunk-and-decode-share.md).)
7. **Spec 04 AC10 misses the 29 GB bound.**
   - Peak card use is 31.2 GiB with the desktop. The plan gives the expert cache everything left of a 31.5 GB budget (card − 1 GiB headroom).
   - Nothing is allocated while serving.
   - Three lanes at the default context fit: the 2.05 GB KV pool holds 486K tokens, against 3 × 131,072 = 393K.
8. **Spec 04 AC11 passes**, with one wording gap. Each check ran on the real model:
   - `/v1/models` names `qwen3.8-flash-next`.
   - Chat completions work non-streamed and streamed, with thinking (`reasoning_content` deltas) and tool calls (`finish_reason: tool_calls`, arguments `{"city":"Lisbon"}`, streamed `{"city":"Rome"}`).
   - The Responses API works non-streamed with reasoning (answer "42") and streamed with a tool. The stream carries the `reasoning_text` deltas, `function_call_arguments.done` with `{"city":"Oslo"}`, and `response.completed`.
   - An image part gets a 400 `vision_disabled`: "Qwen3.8-Flash-Next takes no images".
   - `/v1/decide` gets a 400 `model_unsupported`: "/v1/decide is not served by Qwen3.8-Flash-Next".
   - The gap: speculation is refused **at start, not with a 400**. `--spec dflash2 --draft-tokens 3` exits 1 before any GPU allocation with "`--spec dflash2`: Qwen3.8-Flash-Next has no speculative decoding". That is the intended design (`config.rs` test `flash_next_refuses_speculation_and_vision_at_start_naming_itself`). AC11's wording ("a 400") does not fit a start flag.
9. **Load to ready takes 112 s** against the estimates of 14-15 s cold and 6-8 s warm.

Inferred (not measured):

- **Why decode is short of the simulation.**
  - What is known about the step: one lane's 15.3 ms ITL holds ~1.2 ms of routed MoE (#300: 25.6 µs/layer × 48, at 33% of the bandwidth roofline). The 31 MB/token of expert traffic is ≤ 2.5 ms of link time, and prefetch should hide part of it.
  - The simulation assumed a 6 ms step plus misses. The remaining ~11 ms is not decomposed.
  - Candidates are the n-gram rows read from NVMe on the decode path (two thirds of the rows come from the file) and the non-MoE ops.
  - So the #300 decode floor (33% against 50%) is real, but it is not what separates 65 from 102 tok/s.
- **Why prefill is short of the study.**
  - The study modelled max(link, compute) with compute at 0.77-1.3 s per 4K.
  - Measured, an 8K chunk spends ~7 s beyond its ~1.3 s link term. Compute is therefore ~3x the study's pessimistic bound, or the link and compute do not overlap, or both.
  - Since measured (2026-10-07, after the decode-round fixes): neither holds. An 8192-token chunk is 1.9 s on the card with 89% of its copies under compute (0.21 ms/token), plus ~1.3 s of host n-gram gather before it; see [the prefill chunk finding](2026-10-07-flash-next-prefill-chunk-and-decode-share.md).
- **Why a reload is slow.**
  - Every load is cold here. The 37.8 GB pinned pool plus the desktop leave the page cache ≤ ~14 GB of the 63.8 GB, so a 71.8 GB artifact cannot stay cached between loads.
  - The disk is not the limit either: 43 GB at 1.64 GB/s is ~26 s. ~85 s of the 112 s is spent in the load path itself (pinning, staging or repacking the pool; not separated here).
- **Why the reuse-off turn is slow.** It takes 28.2 s, not the estimated ~6.5 s, because of the prefill rate above, not because of reuse.

## Implications

- **Agents.** Flash-Next serves three agents at ~43 tok/s each while nobody prefills. Every agent turn is a prefill of at least one chunk, though, and each chunk costs ≥ 2 s. It also stalls the other lanes for its whole duration.
  - Agent-loop latency is set by prefill, not by decode.
  - The two levers are the per-chunk expert stream (17 GB over a Gen 3 link) and the prefill compute rate.
- **Phase-2 model switch.** The switch estimate (14-15 s) does not hold on this build: a switch to Flash-Next costs ~2 minutes. The load path is what to fix before a switch is worth building.
- **Host RAM.** On a 64 GB desktop with apps open, the host plan fits only with a KV-RAM arena of about 2 GiB. `make start MODEL=flash-next` as shipped passes `--kv-host-pool-bytes 8G` and is refused at ~52 GB available.
  - Flash-Next needs **≥ 48.5 GB available** with the reuse defaults: 42.1 GB planned plus the 6 GiB margin.
- **VRAM.** Meeting the 29 GB bound means capping the budget, at a cost of ~2.2 GB of expert cache (~11% of it). How much that costs the hit rate is not measured.

## Limits and unknowns

- **Cold page cache.** It was not produced: no reboot and no 27B session in between. The note above explains why "warm" does not exist on this machine.
- **Unmeasured cells.**
  - Decode at 2 lanes beyond 2K and at 3 lanes at 128K.
  - TTFT at 1K and 4K with a cold expert cache, and every TTFT at 128K.
  - The expert hit rate per domain (all prompts here are prose).
- **Prompts.** The decode and TTFT prompts are synthetic word lists with a story instruction, not agent traffic. The swarm is a simple three-client script, not a recorded trace replay.
- **Comparisons not made here.**
  - MMLU is not paired against the study's recipe (73.0%) or the 27B (68.3%): no per-question records for them were at hand.
  - The 27B's numbers come from the README's harness, which runs speculation, and were not re-measured.
- **Inferences.** The decode-step and prefill decompositions above are inferences from counters and earlier findings. No profile of a decode round, a prefill chunk or the load was taken.

## Update 2026-10-07: lanes at 262K context (GitHub #306)

Flash-Next's default is now 262,144 tokens per lane (the checkpoint's trained positions) and `--decode-lanes N` (`make` knob `LANES`, default 3) sets the lane count. Both runs: release build of `fn-lanes-262k`, hq-e8-2b, `--max-context 262144`, `--prefill-chunk 8192`, default `--vram-headroom-bytes 4G`, MTP off, one lane decoding the greedy prompt (1,800 tokens, `ignore_eos`) back to back for 90 s, 5 requests each, no other process on the card.

| `--decode-lanes` | KV pool | expert cache | decode tok/s (5 requests) | ITL p50 | decode hit rate |
|---:|---:|---:|---:|---:|---:|
| 3 | 3.46 GiB (786K tokens) | 13.38 GiB | 85.3-89.5 (89 typical) | 10.5 ms | 96.25% (8.32M hits, 0.32M misses) |
| 1 | 1.15 GiB (262K tokens) | 15.74 GiB | 98.5-99.3 | 9.5 ms | 97.55% (8.43M hits, 0.21M misses) |

- One lane gives the expert cache 2.36 GiB more (the pool shrinks by 2.31 GiB; the program line moves by 13 MB), and a lone user decodes **11% faster** (99 against 89 tok/s) with a hit rate 1.3 points higher: the misses fall by a third.
- Three lanes still keep the whole 262,144 tokens per lane. Taking lanes away is the only knob that moves VRAM from the pool to the cache, and it does not touch any lane's context.
- The earlier 99.0 tok/s at one lane (#306) is the `--decode-lanes 1` figure, so the lanes=3 default costs a lone user about 10 tok/s.
- Not measured: three lanes decoding at once at 262K (the aggregate), and the load's RAM headroom beyond the 46 GB the host plan needs.

## Follow-ups

The coordinator opens these as issues:
- profile one decode round (step decomposition against the simulation's 6 ms);
- profile one 8192-token prefill chunk (link/compute overlap, compute rate);
- profile the load path (112 s, of which ~85 s is not disk);
- decide the prefill/decode interleave (a chunk stalls the other lanes ~7 s);
- make the Flash-Next Make knob pass a host arena that fits (2 GiB);
- decide the 29 GB VRAM bound against the expert cache;
- fix spec 04 AC11's "400 for speculation" wording to "refused at start".
