# The 27B's 16 GB artifact: first light on ignis

- Kind: experiment
- Status: current
- Observed: 2026-10-23
- Last verified: 2026-10-23
- Scope: serving / the 27B's trellis body (`gpillon/Qwen3.8-27B-trellis-3bpw-ignis`) under a 16 GiB VRAM budget and
  the `rtx50-16gb` profile on the RTX 5090 (PCIe Gen 3): VRAM plan, greedy serving at two lanes with DFlash2, the
  131,072-token leg, a two-window KLD against BF16; the served NVFP4 artifact on the same build
- Topic: artifacts-and-gates
- Related: [GitHub #339](https://github.com/gpillon/ignis/issues/339),
  [GitHub #340](https://github.com/gpillon/ignis/issues/340),
  [spec 27b-16gb/05](../specs/27b-16gb/05-gates-and-simulation.md) (Part A, "As built: acceptance"),
  [spec 27b-16gb/04](../specs/27b-16gb/04-the-runtime-follows-the-card-class.md),
  [ADR 0060](../adr/0060-the-27b-has-a-trellis-body-for-small-cards.md),
  [the dense trellis kernels on the 5090](2026-10-23-dense-trellis-kernels-on-the-5090.md)
- Superseded by: none
- Summary: Under a 16 GiB budget on the 5090 the 27B's trellis artifact plans to its closed forms (a 509,184-token pool, nothing adapted), is KLD 0.0419 against BF16 to the NVFP4's 0.0534, and commits 4.41 tokens per round at 12.83 ms.

## Question

Once the 27B program binds the trellis body (#339), does the 16 GB artifact load under a 16 GiB budget with the plan
the formulas predict, serve coherent greedy text at two lanes including a 128K prompt, and land near M1's weights-only
KLD? Does the served NVFP4 artifact still serve exactly as before?

## Evidence

Branch `339-first-light` at 60f4447b, release build, 2026-10-23; ~2 GB of the card held by the desktop. Raw logs,
JSON records and scripts: `.scratch/339/acceptance/` (untracked).

**The plan**, `make start ARTIFACT=<trellis artifact> VRAM_BUDGET=17179869184 LOOP_GUARD=off IDLE_UNLOAD=0
ARGS="--profile rtx50-16gb"`, the `ignis.runtime.vram_plan` event:

| line | bytes | from |
|---|---:|---|
| weights | 10,907,958,020 | spec 01's trellis text scope, 9,825,087,488, and 1,082,870,532 more: the drafter's ~1.08 GB; the embedding (1,350,864,896) is in host memory |
| cuda_context | 452,595,712 | |
| workspace | 395,898,880 | spec 05a's closed form at chunk 1,024 |
| trellis_workspace | 36,225,280 | the op's layout at `mlp/down`, prefill route, chunk 1,024 |
| decode_graph | 20,621,328 | closed form at two lanes, window 7 |
| verify_round / drafter_round | 38,034,808 / 44,673,544 | |
| lane_state | 393,760,768 | 196,880,384 a lane, the NVFP4 8-lane plan's per-lane cost |
| hq_residual_window | 71,303,296 | 35,651,648 a lane |
| sampling | 1,156,768 | |
| residual | 124,474,800 | |
| KV pool | 4,692,672,512 | 7,956 pages, 509,184 tokens |
| total | 17,179,375,716 | of 17,179,869,184 |

Nothing adapted: 262,144 context, two lanes, `dflash2 draft_tokens=7 draft_head=full`, vision off.

**Serving** (greedy; tokens per round = committed tokens / verify rounds; a round's time = (duration − TTFT) / rounds,
from `ignis.request.done` and `ignis.request.ttft`):

| run | requests | tokens per round | round ms | note |
|---|---:|---:|---:|---|
| canaries, one lane | 8 | 4.41 | 12.83 | 4 of 4 sane, deterministic, coherent |
| canaries, two processes at once, 4 runs | 64 | 4.27 | 16.52 | width-2 rounds, overlap shown in the log |
| word soup, 131,072 tokens exactly, cold | 3 | 4.47 | 19.33 | TTFT 42,984 / 43,136 / 43,216 ms |
| code, 129,992 tokens, cold, thinking off | 1 | 3.56 | 19.31 | TTFT 42.5 s |
| NVFP4 canaries, one lane (8-lane load) | 8 | 4.92 | 14.36 | the same prompts |
| NVFP4 code, 129,992 tokens | 1 | 3.44 | 20.75 | TTFT 17.1 s at chunk 2,048 |

- In the two-lane runs, 56 of 64 outputs equal their sequential text. The 8 that differ are every `math-greedy`
  served on lane 7 of a width-2 round, flipping one reasoning token. At that position the trellis program's two
  candidates are 28.125 and 28.0 logits, one BF16 step apart (p 0.53 / 0.47). The NVFP4 artifact, run the same way,
  differs from its own width-1 text on 48 of 48.
- The code request asks what `nonce(27)` in `crates/bench/src/ttft.rs` returns, ~7 KB into the text. The answer is
  coherent and wrong in one letter: "`a027`", with the right mechanism but index 1 read as `a`. The trellis body at
  23,996 tokens and NVFP4 at both lengths give the same "`a027`".
- `ignis-bench ttft --cells 131072` could not be used: its filler generator re-tokenizes the whole message after
  each word, 0.49 s per 129K-token encode on this host, ~8 h per prompt.

**KLD against BF16** (`span_logits`, chunk 1,024, scored by `.scratch/kld-2026-09-24/ref_kld.py`): windows 0 and 1
of the 2026-09-24 suite, `crates/server/src/api.rs` parts 0 and 1, 4,096 tokens each. KLD / top-1:

| window | trellis, hq-e8-2b | trellis, BF16 KV | M1 (EXL3 weights only, PyTorch) | NVFP4, BF16 KV | NVFP4, hq-e8-2b |
|---|---|---|---|---|---|
| 0 | 0.0534 / 90.80% | 0.0421 / 91.72% | 0.0420 / 91.60% | 0.0553 / 91.70% | 0.0666 / 90.89% |
| 1 | 0.0304 / 93.82% | 0.0270 / 94.53% | 0.0271 / 94.51% | 0.0363 / 93.65% | 0.0401 / 93.43% |

**NVFP4 on the same build**: with main 05ccde09's baseline context (`MAX_CONTEXT=786432`, yarn:3) the canaries are
identical to main's, `ignis-bench oracle compare` agrees on 265 of 265 recorded tokens, and the plan's lines are
main's except the KV pool, which grows by the extra free memory, and `trellis_workspace`, which is 0. With the
baseline's exact command the desktop's 0.9 GB less let 917,504 fit unadapted. Rope `auto` then went to yarn:3.5,
and every canary's text changed.

## Finding

- **Observed: the trellis body is M1 in the engine.** With BF16 KV the engine path's KLD is M1's to within 0.0001 on
  both windows. Everything the engine adds to M1's PyTorch reconstruction (the trellis op's fp16 accumulation and
  activation rotation, the served embedding, the engine's attention and GDN) adds nothing visible at 4K. The hq-e8-2b
  codec adds +0.0113 / +0.0034, the same as it adds to NVFP4 on the same windows (+0.0113 / +0.0038). At serving
  configuration the two windows give 0.0419 against the served NVFP4's 0.0534.
- **Observed: the plan is the formulas, to the byte.** At the 16 GiB budget every line equals spec 05a's closed forms
  or spec 04's per-lane costs. The pool holds 509,184 tokens, so 262,144 context and a spare 65,536 fit with nothing
  adapted. That is 1.94 full sequences, at 493 KB under the budget.
- **Observed: single-stream decode on the 5090 is a wash, prefill is not.** The trellis rounds commit 90% of NVFP4's
  tokens on the canaries, in 11% less time (12.83 against 14.36 ms), so about the same tokens per second. At 130K the
  rounds take 19.3 against 20.8 ms. Prefill at 130K is 2.5x slower: 42.1 against 17.1 s. The chunks differ (1,024
  against NVFP4's 2,048), but the chunk moves NVFP4's cold prefill by ~2% only
  ([the chunk at 2,048](2026-10-21-the-27b-prefill-chunk-at-2048.md)). Inferred: the gap is spec 02's prefill route
  at 51-57% of the tensor peak.
- **Inferred: two-lane divergence is the engine's route dependence at a near-tie, not a trellis defect.** The single
  flip sits on a one-ulp tie, and NVFP4 diverges more under the same load. NVFP4's width dependence is main's:
  runtime's `decode_lanes_gpu`, which holds the 8-lane load's greedy tokens at every width to a fixture recorded on
  main, is green on this branch. Two lanes keep a request's text only up to near-ties. A gate comparing concurrent
  output to sequential needs `near_tie.rs`'s equivalence, not equality.
- **Observed: the 128K leg works.** 131,072-token prompts are cold, exact and coherent, with no error in the log.

## Implications

- Spec 05b's gate 1 (KLD ≤ 1.2x NVFP4) looks comfortable: 0.78x on two code windows. Python and prose are where M1
  was weakest, and they still need measuring.
- On a real 16 GB card the round time is set by bandwidth and SM count, not by the 5090's 12.8 ms. 05b's SM-cap
  simulation and `budget.py` scaling are still the only speed statement.
- On the 5090 the trellis body's gap to NVFP4 is the prefill, not the decode; #345 (spec 06) is that lever.

## Limits and unknowns

- Two windows of 4,096 code tokens: a sanity check that would catch a wiring error, not the gate. No prose, chat,
  Python or long-context KLD.
- The 5090 emulates a 16 GB card's memory only. With no display in the budget, a real card with a desktop loses
  ~2 GB (`PIANO-VRAM.md`).
- Tokens per round come from short canaries (64 tokens) and three long prompts. The replay (gate 3) is the
  measurement.
- The near-tie probe ran on the prefill route (one 124-token chunk), not on the width-2 verify round where the flip
  happens.

## Follow-ups

- #340 (05b): gates 1-4 on the full suite, the capacity simulation at 0 / 1 / 3 GB of desktop, the speed estimate. Its
  Further Notes still say the KLD dumps go to `Y:`; they now go to `H:` (the owner, 2026-10-21: `Y:` is too slow).
- 05b's user bench and any 128K TTFT cell need a prompt generator that is not quadratic. `ignis-bench ttft`'s filler
  path cannot reach 131,072 tokens; `.scratch/339/acceptance/soup_ttft.py` lands exactly through `/v1/tokenize`.
- `crates/runtime/src/cuda_leaf.rs` warns `decode_graph_capture_incomplete` whenever fewer than 8 widths are
  captured. Since #336 a 2-lane load captures 2 of 2, so the warning (empty `error`) fires on every `rtx50-16gb`
  start.
- `crates/core/tests/vision_load_gpu.rs` is red on main 05ccde09, as on this branch: a vision load adds 1,382,400
  bytes more than its weights, its reservation and the rope staging. It is the only red in this branch's GPU profile.
- The HF card of `gpillon/Qwen3.8-27B-trellis-3bpw-ignis` still says there is no command to run it. It needs the
  owner's OK.
