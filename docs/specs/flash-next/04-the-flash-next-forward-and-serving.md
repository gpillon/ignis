# 04 - the Flash-Next forward and serving: topology, GDN and QSA at its geometry, hyper-connections, the n-gram embedding, three lanes

GitHub: #302 (master #298).

This spec turns specs 01-03 into a served model. ignis selects Flash-Next at
start, loads its artifact and runs its forward:
- 48 layers in 12 blocks of (3 GDN + 1 QSA);
- an MoE block in every layer;
- four hyper-connection residual streams;
- an n-gram embedding read from the NVMe-resident table.

It serves up to three lanes through the same OpenAI surface as the 27B, with
decode in CUDA graphs.

**This spec carries the whole feature's acceptance seam:**
- the G1 teacher-forced harness against the quantized reference that spec 01's
  pass recorded: the checkpoint's own modeling code with the artifact's decoded
  weights;
- the span-logits KLD against the BF16 checkpoint's stored top-64
  log-probabilities. It is the single highest seam. Every kernel and residency test below
it only localizes a failure this seam finds.

ADRs:
- 0043 (Accepted 2026-10-04): a second model, the checkpoint as oracle, own ops, new
  geometries as our code;
- 0044 (Accepted 2026-10-04);
- 0014 (teacher-forced agreement, the scoring);
- 0022 (two KV formats: hq-e8-2b serves, BF16 is the oracle);
- 0027 (Make);
- 0030 (plan lines at load).

## Decided for the autonomous run (2026-10-04)

The owner approved this spec on 2026-10-04 and vetoed none of the agent's
proposals: every *(proposed)* item below is decided, and ADRs 0043 and 0044 are
accepted. The prerequisites in Further Notes are the ticket's blockers on
GitHub, not open questions.

- **Model selection:** the model family is read from the artifact (its family
  name in the container), so the server needs no separate family flag; a flag
  that does not fit the loaded family is refused at start. The Make knob is
  `MODEL=flash-next` (default `MODEL=27b`): it sets the default `ARTIFACT` to the
  Flash-Next artifact in `F:/ai/models/Qwen3.8-Flash-Next-ignis/` and the
  Flash-Next defaults below. `make config` prints them.
- **Defaults:** 3 lanes (`--decode-lanes`, 1..8); KV hq-e8-2b; 262,144 tokens of context per lane (the checkpoint's trained positions; owner decision 2026-10-07, GitHub #306); prefill
  chunk 8192; prefetch width 16; n-gram hot rows 1 GB; decode share 50%
  (GitHub #306: decoding lanes keep half the time while a prompt prefills).
- The 8192-token KLD bound is 1.25× the 2048-token one.
- The math of QSA, the indexer, the hyper-connections and the n-gram embedding is
  the transformers modeling code the study ran (installed in
  `F:/ai/ngram-venv`). ExLlamaV3 1.5.3's `*_ref` functions (same venv) record
  kernel fixtures only.

## Problem Statement

The kernels and the expert cache are not a model. ignis's program is written for
one topology:
- 64 layers of hidden 5120;
- GQA with 4 KV heads;
- one residual stream;
- no MoE, no sparse attention, no n-gram embedding.

Its shapes and counts are fixed in many places: KV geometry, sequence sections,
attention wrappers, GDN gating, snapshot layouts. One latent bug works only
because the 27B has 48 GDN layers and 48 GDN value heads. Until the program runs
Flash-Next's topology from its configuration, the owner cannot send it a
request.

## Solution

**One start option selects Flash-Next** (Make knob and CLI). With it, ignis:
- loads the Flash-Next artifact through its binder (spec 01) and builds the
  expert residency (spec 03);
- runs a forward driven by a Flash-Next topology: GDN layers at hidden 2560, QSA
  layers (GQA 24 query / 2 KV heads of 256, partial rotary, sigmoid output gate,
  an indexer that selects key blocks above 2051 tokens), the MoE block (spec 02),
  four hyper-connection streams, and the n-gram embedding at layer 1;
- serves up to three lanes with decode graphs, through chat completions and the
  Responses API, with the checkpoint's tokenizer and chat template.

The 27B stays the default, and its behaviour, tests and gates are unchanged.

The model is accepted when:
- its teacher-forced argmax agreement with the quantized reference (same
  weights, the checkpoint's modeling code) clears the G1 floor;
- its KLD against the BF16 checkpoint is within a stated factor of the
  quantization-only KLD spec 01 measured for the same artifact, so the engine
  adds only kernel numerics;
- it decodes at ≥ 70 tok/s on one lane and ≥ 130 tok/s total on three.

## User Stories

1. As the owner, I want `make start` with a model knob to serve Flash-Next instead of the 27B, so that I choose the model for a session without editing commands.
2. As the owner, I want `make config` to print the exact Flash-Next server command, the VRAM plan (expert cache size included) and the host plan, so that I see what a start will use before it starts.
3. As the owner, I want the 27B to remain the default with every behaviour unchanged, so that nothing I use today regresses.
4. As the owner, I want Flash-Next served through chat completions and the Responses API with streaming, tools, thinking and the same request fields as the 27B, so that Codex, the Playground and my agents work against it unchanged.
5. As the owner, I want `/v1/models` to name the loaded model, so that a client can tell which model answers.
6. As the owner, I want up to three concurrent lanes on Flash-Next, so that two or three agents can work at once.
7. As the owner, I want decode at ≥ 70 tok/s on one lane and ≥ 130 tok/s total on three at short context, so that Flash-Next is interactive and useful for agents.
8. As the owner, I want contexts well beyond 2048 tokens served correctly through the sparse attention path, so that long documents and agent histories work.
9. As the owner, I want a request for a feature Flash-Next does not have (images, `/v1/decide`, speculative decoding) refused with a clear 400, so that I am never served a silent degradation.
10. As the owner, I want the server's metrics and Monitor to show Flash-Next's residency and n-gram counters beside the existing ones, so that I can diagnose a slow turn.
11. As the owner, I want the whole Flash-Next process under ~28-29 GB of VRAM including the desktop, so that the machine stays usable.
12. As an engine developer, I want a Flash-Next `ModelConfig` and topology that every shape, count and layer pattern is derived from, so that no 27B constant leaks into the Flash-Next forward.
13. As an engine developer, I want the GDN layer-count / value-head-count conflation fixed with a test that fails on today's code, so that a model whose GDN layer count differs from its value-head count (Flash-Next: 36 and 48) binds and runs correctly.
14. As an engine developer, I want GDN gating at hidden 2560, so that Flash-Next's GDN layers run with the 27B's head geometry (16 key, 48 value heads of 128) at their own width.
15. As an engine developer, I want GQA at 24 query / 2 KV heads of 256 with partial rotary 0.25 and a sigmoid output gate, written as our own implementation and leaving the vendored wrappers untouched, so that the 27B's vendored path keeps its port claim.
16. As an engine developer, I want the QSA indexer (4 heads of 128, one compressed key head, compression 4, budget 2048) and sparse gathered attention above 2051 tokens, so that long contexts attend to the blocks the checkpoint would attend to.
17. As an engine developer, I want attention dense up to 2051 tokens, exactly as the checkpoint's own threshold, so that short contexts need no indexer at all.
18. As an engine developer, I want the indexer's compressed keys kept as a per-sequence state section of the paged pool, so that sequences, clones and cancellation handle them like KV.
19. As an engine developer, I want four hyper-connection residual streams carried through the layer program, with each sublayer reading a learned mix and writing back through the gated residual, so that the residual is what the checkpoint computes.
20. As an engine developer, I want the hyper-connection change confined to the Flash-Next program path, with the 27B's single-stream layers and the DFlash2 taps unchanged, so that the 27B's numerics do not move.
21. As an engine developer, I want each token's n-gram ids hashed on the host bit-exactly as the checkpoint does (stored hash buffers, per-head sizes, EOS segmentation), so that the embedding reads the right rows.
22. As an engine developer, I want n-gram rows served from a RAM hot-row cache (1-2 GB from the artifact's hot list) and otherwise read from the artifact file on NVMe with aligned unbuffered reads, so that the 28.8 GB table never needs to be in RAM.
23. As an engine developer, I want a prompt's n-gram rows fetched at admission, before its forward reaches layer 1, and a decode step's rows fetched right after the previous token is sampled, so that NVMe latency overlaps with other work.
24. As an engine developer, I want the n-gram embedding's dequantization, dilated convolution (kernel 4) and gate run on the device from the fetched rows, so that the host only gathers bytes.
25. As an engine developer, I want an FP8 per-row-scale linear for all non-expert projections, so that the artifact's FP8 weights (spec 01) run without dequantized copies.
26. As the owner, I want hq-e8-2b KV as Flash-Next's serving default, with BF16 kept as the oracle format (ADR 0022), so that long contexts on three lanes fit without taking VRAM from the expert cache.
27. As an engine developer, I want GDN, conv, KV and indexer state sized per lane from the topology and reserved at load, so that three lanes fit the plan and nothing allocates while serving.
28. As an engine developer, I want decode captured as CUDA graphs per lane count (1, 2, 3), so that host overhead stays out of the 10 ms token.
29. As an engine developer, I want a large default prefill chunk for Flash-Next (proposed 8192 tokens), so that the near-constant expert transfer per chunk (spec 03) is amortized.
30. As an engine developer, I want the span-logits readout to work on Flash-Next, so that its KLD against the checkpoint is measured the way the 27B's was.
31. As an engine developer, I want the G1 teacher-forced harness to run on Flash-Next against the quantized reference's per-position argmax, so that the 95% floor measures my kernels against the same weights and not the compression.
32. As an engine developer, I want every new op (GQA 24/2, GDN gating 2560, QSA indexer and sparse attention, hyper-connection mix, n-gram dequant/conv/gate, FP8 linear) tested at real geometry against fp64 or recorded references, so that the whole-model seam's failures can be localized.
33. As an engine developer, I want no process-wide singleton added and the Flash-Next model's resources freed by its drop path, so that the model switch (phase 2) can reload cleanly.
34. As a reviewer, I want ExLlamaV3's torch reference functions used only to record kernel-level fixtures, never as the model's oracle, so that "correct" means "what the checkpoint computes" (ADR 0043).
35. As a reviewer, I want the measured tok/s, TTFT and KLD reported against the simulation and spec 01's numbers in a finding, so that the gaps are explained.

## Implementation Decisions

**Owner-made decisions:**
- fast version directly;
- ExLlamaV3 as reference only;
- model switch in phase 2.

*(proposed)* marks the agent's proposals, which the owner may veto.

**Model selection.**
- A start-time choice of model family carries the artifact path and Flash-Next
  defaults: lanes 3, KV hq-e8-2b, prefill chunk, prefetch width.
- It is a CLI option of the server and a Make knob (ADR 0027).
- `make config` and the load log print the full plan.
- One model is loaded per process. The 27B is the default.

**Topology.**
- A Flash-Next model configuration describes every count and shape: layers and
  pattern, hidden, heads, head dims, rotary fraction, GDN geometry, MoE geometry,
  hyper-connection count and rank, n-gram parameters, vocab, norm epsilon.
- The configuration crosses the step ABI. The program derives every buffer, view
  and section from it.
- The 27B's configuration is unchanged.
- Wherever a 27B constant is hard-coded in a path Flash-Next uses, it becomes a
  topology field. Paths only the 27B uses (DFlash2, the vendored GQA 24/4 route)
  keep their constants.

**GDN layer/head conflation.** The GDN value-head count becomes its own topology
field, distinct from the GDN layer count, everywhere the binder and the program
size per-head parameters. A test with a topology whose layer count differs from
its value-head count fails first and then passes.

**GDN at hidden 2560.**
- The recurrence is unchanged: any head count with head dim 128.
- Gating and input projections run on the FP8 linear at width 2560.
- The vendored 27B gating plan is not edited (ADR 0043: new geometry is our
  code).

**QSA layers** *(our implementation)*.
- GQA 24/2 with head dim 256, rotary on the first 25% of each head, a sigmoid
  output gate, paged KV in either format (hq-e8-2b serving, BF16 oracle).
- **Indexer:** 4 heads of 128 over one compressed key head. Keys are compressed
  every 4 tokens and stored in a per-sequence indexer section.
- **Scoring:** relu(q·k) summed over heads, scaled by 1/√128.
- **Block selection:** top 512 blocks, plus the incomplete tail block always kept.
- **Attention:** dense up to 2051 tokens; above that, gathered sparse attention
  over the selected blocks.
- The checkpoint's modeling code defines the math. ExLlamaV3's indexer reference
  functions provide recorded kernel fixtures.

**Hyper-connections** *(our implementation)*.
- The Flash-Next program carries four residual streams of 2560 per token.
- Each sublayer (attention or GDN, then MoE) reads its input as a learned mix of
  the streams and writes its output back through the low-rank (320) sigmoid-gated
  residual. The streams are merged before the final norm.
- The mix weights are FP8 or BF16 as the artifact stores them.
- The single-stream layer ABI used by the 27B is not changed. The Flash-Next
  program has its own layer entry points.

**N-gram embedding.**
- **Host:**
  - hashing per token (n-gram size 3, 8 heads per n-gram, 128 parts, stored hash
    buffers, EOS segmentation), bit-exact with the checkpoint;
  - a RAM hot-row cache loaded from the artifact's hot list (size a load option,
    default 1 GB);
  - an NVMe reader issuing aligned unbuffered reads of the artifact's
    host-streamed table range for missing rows, on worker threads;
  - gathered rows land in a pinned staging buffer.
- **Timing:**
  - a prompt's rows are gathered at admission;
  - a decode step's rows are gathered after the previous token is sampled and
    uploaded with the step's inputs, before the graph launch.
- **Device:** INT4 dequantization, the dilated convolution (kernel 4, dilation 3)
  over the sequence's recent rows (a small per-sequence state section), and the
  gate, feeding layer 1.

**FP8 linear** *(our implementation)*. E4M3 weights with an fp32 per-row scale,
BF16 activations, fp32 accumulate, tensor cores in prefill and a GEMV route in
decode. It is used by every non-expert projection and by spec 02's shared expert.

**State and memory.**
- Per lane: GDN recurrent and conv state for 36 GDN layers, paged KV for 12 QSA
  layers (2 KV heads × 256), indexer section, n-gram conv section.
- **KV formats (owner: hq-e8-2b in scope).** Both formats of ADR 0022, a load
  option fixed for the life of the load:
  - **hq-e8-2b is the serving default**: 3,456 bytes per token against 24,576
    for BF16 (factor 7.11, as for the 27B);
  - **BF16 is kept** as the format every correctness oracle runs against.
  - The hq attention routes for 2 KV heads are our own implementation, written
    beside the vendored 4-KV-head routes, which stay untouched (ADR 0043). This
    covers dense prefill, decode and the sparse gathered path over hq pages,
    plus the hq residual window the 27B wires.
  - hq earns its acceptance in-house, as ADR 0022 requires: codec error on real
    Flash-Next KV rows, and the hq route against the BF16 route on identical
    keys and values, with the tolerance derived from the measured codec error.
- Default context per lane: 262,144 tokens with hq-e8-2b, the checkpoint's
  trained positions (about 2.7 GB for 3 lanes; first proposed as 128K, about
  1.4 GB; raised by the owner, GitHub #306). The pool is sized by a byte budget and the token capacity is derived
  from the format in force (ADR 0022).
- The VRAM plan places KV before the expert cache (spec 03) and prints the trade.
- Everything is a plan line at load (ADR 0030).

**Serving surface.**
- Chat completions and the Responses API work unchanged, with the artifact's
  template and tokenizer.
- Requests that need images, `/v1/decide`, or speculation get a 400 naming
  Flash-Next.
- Prompt reuse across requests (ADR 0029) is not wired for Flash-Next in this
  spec. Requests prefill from scratch (Out of Scope).

**Graphs.**
- Decode is captured per lane count 1-3, with residency inside the graph if spec
  03 chose the device-resident miss path, else per layer segment.
- Prefill runs eagerly per chunk.

**Correctness references** (from spec 01's single pass; no BF16 run here).
- **G1:** the canary sequences are fixed text (the 27B canary prompts and
  completions, in Flash-Next's template). The fixture carries the quantized
  reference's argmax per position. The G1 harness gains an optional
  expected-argmax column: when present, ignis's teacher-forced argmax is
  compared with it, while the fed tokens stay the fixture's. Without it the
  harness behaves exactly as today, so the 27B is unchanged. The floor is ADR
  0014's 95%: same weights, so only kernel numerics and routing near-ties can
  disagree.
- **KLD:** the engine's span-logits on the stored windows are scored against
  the BF16 top-64 log-probabilities and log-sum-exp. The 64 entries plus the
  tail mass give the KL; the same scorer runs on spec 01's quantized stream, so
  the comparison is like for like.

**Speed measurement.** The decode bench of G3, run against the Flash-Next server
at short context (≤ 2K prompt, 256 generated), at 1 and 3 lanes, warm cache. TTFT
is reported for cold 2K, 8K and 32K prompts.

## Testing Decisions

A good test here runs the model and checks what it produces against the BF16
checkpoint. Whole-model checks are agreement, KLD and the exam proxy. Op tests at
real geometry are bounded errors. A test never asserts internal buffer layouts or
launch shapes.

- **The acceptance seam (GPU, `--ignored`, GPU profile):**
  - the G1 teacher-forced harness on Flash-Next against the quantized
    reference's expected argmax (prior art: the 27B's teacher-forced G1 test and
    the oracle module; the expected-argmax column is new, with a CPU test that
    the 27B fixtures score exactly as before);
  - the span-logits readout on the stored 2048-token and 8192-token windows,
    scored against the BF16 top-64 references from spec 01 (prior art: the
    2026-09-24 KLD finding's method and its span-logits example);
  - the MMLU-Pro proxy on 281 questions, paired against BF16 and against spec
    01's torch number.
- **Kernel-leaf CTests at Flash-Next geometry,** for every op this spec adds:
  - GQA 24/2 against fp64;
  - GDN gating at 2560 against fp64;
  - indexer scoring and block selection against recorded reference fixtures,
    with selection exact except at documented score ties;
  - sparse attention against fp64 dense attention masked to the selected blocks;
  - hyper-connection mix against fp64 and the recorded reference;
  - n-gram dequant, conv and gate against recorded `forward_reference` outputs;
  - FP8 linear against fp64;
  - hq-e8-2b: codec error on real Flash-Next KV rows, and the hq attention
    routes (dense prefill, decode, sparse gathered) against the BF16 route on
    identical keys and values (prior art: the 27B's hq acceptance, ADR 0022).
- **CPU, in `cargo test`:**
  - the Flash-Next topology and every derived size;
  - the GDN conflation regression;
  - n-gram hashing bit-exact against a fixture of ids recorded from the
    checkpoint's hashing code on fixed token streams;
  - the hot-row cache and reader planning against a small fixture table;
  - the binder on spec 01's fixture artifact;
  - plan arithmetic (KV vs expert cache);
  - server behaviour on a mock engine reporting a Flash-Next identity: model id,
    400s for images, decide and speculation.

  Prior art: the core crate's config and plan tests, the server's mock-compute
  HTTP tests.
- **Speed:** the G3 decode cell against the Flash-Next server, reported against
  the simulation (102 / 186 tok/s at a 21.5 GB cache).
- **The 27B's regression net:** the full existing suite and the 27B GPU profile
  stay green. The model-selection option defaults to the 27B.

## Acceptance

1. A start option and Make knob serve Flash-Next from its artifact. `make config` prints the command, the VRAM plan (expert cache included) and the host plan. The 27B remains the default with all existing tests and gates unchanged.
2. The GDN layer-count / value-head-count conflation is fixed. A regression test with differing counts failed before the fix and passes after.
3. Every op this spec adds (GQA 24/2 with partial rotary and output gate, GDN gating at 2560, QSA indexer and sparse attention, hyper-connection mix, n-gram dequant/conv/gate, FP8 linear) has a kernel-leaf test at real geometry, green on a free 5090. No vendored file is edited.
4. G1: teacher-forced argmax agreement against the quantized reference recorded by spec 01 ≥ 95% overall, every mismatch listed. The G1 harness's expected-argmax column is optional, and the 27B's fixtures score exactly as before.
5. KLD per domain on the stored 2048-token windows (span-logits readout scored against the BF16 top-64 references) ≤ max(1.1 × spec 01's quantization-only KLD for the same artifact and domain, that KLD + 0.01).
6. On the stored 8192-token windows (sparse QSA path), KLD ≤ max(1.1 × spec 01's quantization-only KLD on the same windows, that KLD + 0.01). At kernel level, indexer block selection matches the reference except at documented score ties.
7. hq-e8-2b is the serving default and BF16 the oracle format. The correctness checks of acceptance 4-6 run on BF16 KV. hq-e8-2b's codec error on real Flash-Next KV rows and its routes (dense, decode, sparse) against the BF16 route on identical keys and values are within tolerances derived from the measured codec error. The per-domain KLD with hq-e8-2b on 2048- and 8192-token windows is reported beside the BF16-KV numbers.
8. MMLU-Pro proxy ≥ 71% on the 281 questions, paired against BF16 (73.7%) and spec 01's torch result.
9. Decode ≥ 70 tok/s on one lane and ≥ 130 tok/s total on three lanes (≤ 2K prompt, 256 generated, warm cache, planned cache size), with hq-e8-2b KV, reported against the simulation's 102 / 186. TTFT is reported for cold 2K, 8K and 32K prompts.
10. Peak VRAM with the desktop is ≤ 29 GB, and nothing is allocated while serving. Three lanes at the default context fit the plan.
11. Chat completions and the Responses API (streaming, tools, thinking) work on Flash-Next. Images and `/v1/decide` get a 400 naming the model. Speculation is the MTP head's alone, and off unless `--spec mtp` names it (spec 07): `--spec dflash2` is refused at start naming the model ("Flash-Next drafts with mtp"), and `--spec mtp` without the head's companion container beside the artifact fails the start. `/v1/models` names the loaded model.
12. Residency and n-gram metrics are exported and shown on the Monitor. No process-wide singleton is added, and the model's resources are freed by its drop path.
13. `cargo test` passes workspace-wide, and `cargo check --workspace --features cuda --tests` is clean. The Flash-Next GPU tests and the 27B GPU profile are green on a free 5090.

## Out of Scope

- **The runtime model switch:** phase 2. This spec only refrains from blocking
  it: no new singletons, a full drop path.
- **Prompt reuse across requests for Flash-Next** (ADR 0029: retained slots,
  checkpoints, snapshots of its state sections, a small KV-RAM arena): spec
  `flash-next/05`, right after this one.
- **MTP and speculative decoding for Flash-Next;** the DFlash2 drafter is the
  27B's.
- **Vision, `/v1/decide` readouts and attention-head readouts** on Flash-Next.
- **More than three lanes.**
- **Re-opening uniform versus allocated expert bit widths.**
- **Public benchmark comparisons; ExLlamaV3 as a runtime.**

## Further Notes

- Prerequisites to `ready-for-agent`: ADRs 0043 and 0044, spec 01 (artifact,
  reference recordings), spec 02 (MoE kernels), spec 03 (residency).
- **Prompt reuse matters more for Flash-Next than for the 27B.** A cold prefill
  chunk streams most of the expert set over PCIe (2.65 s measured for a cold
  4K prompt, spec 03), so an
  agent turn that re-prefills a 30K-token history pays several seconds even with
  8192-token chunks. Wiring ADR 0029's reuse to Flash-Next's state sections is
  the first follow-up to propose after this spec's acceptance.
- **Measurements:**
  - The study's simulation gives the decode targets (102 tok/s at 1 lane, 186
    total at 3, with a 21.5 GB cache). The floors in acceptance 9 leave room for
    the non-expert compute and the host steps the simulation did not model.
  - The measured figures go into a finding with the gap explained.
- **The checkpoint is the oracle (ADR 0043).** ExLlamaV3's torch references
  record kernel fixtures only. Its router uses fp16 logits, so its expert choices
  may differ from the checkpoint's and from ours.
- Most hard-coded 27B constants are already listed by file in the
  survey of 2026-10-04 (`.scratch`, untracked). The ticket's first task is to
  turn that list into topology fields.
- **Acceptance 7 as built (GitHub #306 item 8, 2026-10-07).**
  `test_flash_next_sparse_attention_hq` runs the three routes on the real
  layer-3 K/V rows of `kernel/tests/fixtures/hq_kv_rows_flash_next.bin`, hq
  against BF16 on identical rows, codec only. The query rows are synthetic:
  the fixture holds K/V rows only.
  - the decode route: listed rows on the sparse kernel;
  - the dense prefill route: visible rows decoded by position, then
    `qsa::attend_dense`;
  - the sparse prefill route, past `dense_threshold()`: visible rows decoded
    by position, then each row's selection on the sparse kernel.

  Every output is held within the derived fp64 bound over the rows its route
  read. AC7's tolerance, hq within the BF16 output plus the codec's measured
  effect (fp64 over the rows hq read minus fp64 over the BF16 rows) plus both
  bounds, follows from the hq and BF16 outputs' own bound checks by the
  triangle inequality, and is asserted as stated. The check
  that stands on its own is the aggregate one: the route distance minus that
  effect is at most 2^-7 relative RMS.

  The derived bound first took BF16 rounding as 2^-9, true only at the top of
  a binade; round-to-nearest is 2^-8 relative at worst. On real rows the
  attention is peaked and the output's own rounding dominates, so the BF16
  route itself exceeded the 2^-9 bound. With 2^-8 every element of every
  route is within it, with room (the test prints the worst ratio each run,
  and its comment holds the figures). So the bound is the derivation
  corrected, not a margin fitted to the data. The dense kernel normalizes by
  the fp32 sum of its unrounded weights, so its weight rounding is bounded by
  2^-8 Σ w|v|, not the centred form.
