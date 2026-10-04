# 01 - the Flash-Next artifact: experts in trellis with a K per expert, FP8 elsewhere, the n-gram table beside them

GitHub: #299 (master #298).

Qwen3.8-Flash-Next is a 125B mixture-of-experts checkpoint of about 250 GB in
BF16. ignis can load only the `.ninfer` container of the 27B, which ninfer's
converter produced. This spec builds the offline converter that turns the
checkpoint into a `.ninfer` v2 artifact ignis can serve on one RTX 5090. It
also produces, in the same pass, the recordings the rest of the feature is
checked against: the canary references and the stored reference logits.

The recipe is the one the compression study measured
(`.scratch/flash-next-compression-2026-10-03/RISULTATI_3.md`, runs 1-8):

| part | encoding |
|---|---|
| experts | trellis-coded, mean 2.5 bits per weight, K per expert from {2, 2.5, 3, 4} by allocation |
| other linears | FP8 E4M3 with one scale per row |
| n-gram table | INT4 in groups of 32 |
| calibration | broad and balanced |

ADRs:
- 0002, load `.ninfer` directly and consume every object;
- 0043, a second model without a ninfer reference (Accepted 2026-10-04);
- 0044, experts in trellis with a K per expert (Accepted 2026-10-04);
- 0030, device memory reserved at load.

## Decided for the autonomous run (2026-10-04)

The owner approved this spec on 2026-10-04 and vetoed none of the agent's
proposals: every *(proposed)* item below is decided, and ADRs 0043 and 0044 are
accepted. The prerequisites in Further Notes are the ticket's blockers on
GitHub, not open questions.

- **Source:** `Qwen/Qwen3.8-Flash-Next` at revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`, fetched by HTTP range requests
  into RAM with the study's fetcher; no full local copy. The n-gram table's BF16
  source is the study's local cache
  `F:/ai/opencode/inference/.scratch/flash-next-compression-2026-10-03/real/table_cache`
  (96 GB). Do not delete it: the owner decides after the artifact is verified.
- **Calibration corpus, first artifact:** exactly the study's run 4-8
  calibration set, so that acceptance 3 and 4 compare like with like. It is
  selected as in the study's `real/e2e6.py` / `real/e2e8.py`: the first 64
  non-test windows of the 27B KLD windows (repo code, docs, chat;
  `.scratch/kld-2026-09-24/windows*.json`), the calibration chunks of
  `real/ood/run4_chunks.json` (Wikipedia en/it/zh, math, Python) and
  `real/ood/calib_mmlu.json`. The manifest records these sources; they are the
  allowlist. A broader corpus (agent traces, e-mails) is a later re-conversion,
  not this ticket.
- **Study code to port** (all under
  `F:/ai/opencode/inference/.scratch/flash-next-compression-2026-10-03/real/`,
  untracked; worktrees read it by absolute path): `e2e8.py` (run 8: exllamav3
  quantization fed our Hessians, allocation over {2, 2.5, 3, 4}), `e2e.py`,
  `e2e2.py`, `e2e3.py` (layer-streamed forward, Hessians, routing), `quant.py`,
  `fetch.py`, `e2e6.py` (`fp8_rows`, `table_quant`, evaluation), and
  `review/run4_paired.py` (MMLU McNemar, KLD bootstrap).
- **Python environment:** `F:/ai/ngram-venv` (torch 2.13 cu130, exllamav3 1.5.3
  installed and its CUDA extension already built for this card). Pin
  `exllamav3==1.5.3`. Do **not** set `TORCH_CUDA_ARCH_LIST` when importing it:
  that triggers a ~6-minute full rebuild.
- **Who writes the container:** the Python converter writes per-layer work files
  (encoded expert tiles, FP8 tensors, table shards, sidecar data). A Rust packer
  in the artifact crate, built on its existing writer, assembles the `.ninfer`
  v2 container from them. Python never writes the container format itself, so
  reader and writer share one implementation.
- **Output:** `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (artifact, sidecar,
  reference recordings, routing traces), about 71 GB. F: had ~105 GB free on
  2026-10-04; check before starting, and remember each worktree's cargo target
  dir also lives on F:.
- **GPU time:** the full conversion is about 9-10 hours (run 8: ~650 s per
  layer for the four-K sweep) plus the oracle recordings. Hold the shared GPU
  lock (`../.inference-qwen-worktrees/.swarm/gpu-lock.sh`) for the whole run,
  launch it detached (PowerShell `Start-Process` with a
  `cmd /v:on /c ... & echo !ERRORLEVEL! > x.exit` sentinel; background Bash
  watchers get killed), and watch it with a recurring heartbeat. It resumes per
  layer.

## Problem Statement

The owner wants Flash-Next on the 5090 for the work where the 27B falls short:
general knowledge, Italian and other languages, mathematics, long technical
documents, e-mails, slides. The checkpoint does not fit anywhere on the machine
as it is:
- its experts alone are 241 GB in BF16 against 64 GB of RAM;
- no public runtime serves experts at different bit widths;
- the formats that would fit lose too much quality. A plain 2.5-bit integer
  format cost 3 MMLU points and 1.4-1.9× the KLD in the study.

ignis also has no way to know whether a Flash-Next answer is right. There is no
reference engine to record an oracle from, so the oracle has to come from the
checkpoint itself.

## Solution

An offline converter that the owner runs once per checkpoint revision. On this
machine it takes about a night of GPU time and can resume after an
interruption. It reads the BF16 checkpoint layer by layer and writes:

- **one `.ninfer` v2 artifact** with:
  - the experts in a trellis format with a K per expert (≈38 GB);
  - every other linear in FP8 with a per-row scale (≈4.3 GB);
  - the n-gram table in INT4 (≈28.8 GB);
  - the tokenizer and the chat template;
- **a sidecar** recording the K map, the per-layer distortion, the calibration
  corpus manifest, the source revision and the tool versions;
- **the reference recordings** (see "One pass, two references" below):
  - the BF16 checkpoint's top-64 log-probabilities (plus the log-sum-exp) per
    position on fixed 2048- and 8192-token windows;
  - the **quantized reference**'s per-position argmax and top-64
    log-probabilities on the canary sequences and the same windows. The
    quantized reference is the checkpoint's own modeling code run with the
    artifact's decoded weights;
- **routing traces:** the quantized reference's top-10 selection per token and
  layer on the study's test chunks, per domain, which residency (spec 03)
  replays.

The converter proves its own output in the same pass: the MoE error per layer,
and the whole-model KLD and MMLU-Pro proxy of the quantized reference against
the BF16 checkpoint. These are the quantization-only numbers; the engine is
later held to them (spec 04).

## User Stories

1. As the owner, I want one command that turns the Flash-Next checkpoint into an ignis artifact, so that I can serve the model without hand steps.
2. As the owner, I want the converter to resume from the last finished layer after a crash or a reboot, so that a night of GPU time is never lost to one failure.
3. As the owner, I want the converter to refuse to start when the GPU is busy, so that it never kills another run on the card.
4. As the owner, I want the converter to stay under 28 GB of VRAM, so that the desktop keeps working while it runs.
5. As the owner, I want the converter to report its disk need before it starts and to fail early when F: lacks the space, so that a conversion does not die at layer 40.
6. As the owner, I want the experts encoded at a mean of 2.5 bits per weight including scales, so that all of them fit in pinned host RAM (37.7 GB) beside the OS.
7. As the owner, I want each expert to get the bit width its measured distortion earns, so that the experts my work uses most keep the most precision.
8. As the owner, I want every expert to get at least 2 bits, so that an expert my calibration never saw is not destroyed.
9. As the owner, I want the calibration corpus to cover code, agent traces, Italian and English prose, e-mails, technical documents, mathematics, exam-format questions and the other languages I need, in balanced shares, so that allocation does not starve a domain I use.
10. As the owner, I want the corpus manifest (sources, licences, token counts per domain) written into the artifact's sidecar, so that I know what an artifact was calibrated on.
11. As the owner, I want the calibration corpus built only from public data or my own data, so that no contributed or private material ends up shaping or naming anything in the repo.
12. As the owner, I want non-expert linears in FP8 with a per-row scale, so that they cost no measurable quality (the study's MMLU 74.4% with BF16 experts).
13. As the owner, I want the router kept at the checkpoint's precision, so that expert selection, a discrete decision, is not perturbed by weight quantization.
14. As the owner, I want the n-gram table in INT4 with groups of 32, so that it fits on NVMe at 28.8 GB without the quality loss INT2 showed.
15. As the owner, I want the n-gram table's hot rows (the most frequent on the calibration corpus) listed in the artifact, so that the engine can keep 1-2 GB of them in RAM and read the rest from NVMe.
16. As the owner, I want the artifact to carry the checkpoint's own tokenizer and chat template, so that prompts render as the model was trained.
17. As the owner, I want the sidecar to record per layer the MoE distortion in dB against the study's run 6 and run 8 numbers, so that I can see the conversion kept the measured quality.
18. As the owner, I want the converter to measure the whole-model KLD per domain and the MMLU-Pro proxy on the written artifact before I ever load it, so that a bad conversion is caught before the engine is blamed.
19. As the owner, I want a conversion that misses the quality floor to fail loudly and name the fallback (3.0 bits mean), so that I decide instead of serving a degraded model.
20. As an engine developer, I want the artifact to be a `.ninfer` v2 container that ignis's reader parses with no special case, so that the loader, the checksum verification and `inspect` keep working.
21. As an engine developer, I want every expert projection stored bit-compatible with ExLlamaV3's trellis tensor plus its K, so that ExLlamaV3's `reconstruct` is an exact oracle for my decode kernel.
22. As an engine developer, I want each expert projection (the fused gate/up plane, the down plane) stored as one contiguous, aligned range, so that one copy moves it from host to device.
23. As an engine developer, I want an expert projection's byte size fixed by its shape and K alone, so that the cache has eight slot sizes (two shapes × four K) and every slot of a class is interchangeable.
24. As an engine developer, I want the n-gram table stored as a host-streamed object whose rows are fixed-size and directly addressable by row index, so that a row is one aligned read with no index lookup.
25. As an engine developer, I want the hashing buffers the checkpoint uses for the n-gram ids stored in the artifact, so that my host-side hashing is bit-exact with the checkpoint's.
26. As an engine developer, I want every object in the artifact consumed by the Flash-Next binder, and an unconsumed object to fail the load (ADR 0002), so that a converter/binder drift cannot go unnoticed.
27. As an engine developer, I want the sidecar to carry, per K class, the count of expert projections and their routed traffic share measured on calibration, so that residency (spec 03) can size its slot pools from data.
28. As an engine developer, I want the quantized reference's per-position argmax on fixed canary sequences recorded in a G1 fixture, so that G1's teacher-forced agreement measures the engine's kernels against the same weights, not the compression.
29. As an engine developer, I want the BF16 checkpoint's top-64 log-probabilities on fixed 2048- and 8192-token windows stored once, so that the engine's KLD against the checkpoint is measured without re-running 250 GB of BF16.
30. As an engine developer, I want the KLD window set and the MMLU-Pro proxy questions fixed and versioned, so that numbers from different runs are paired, not compared across different texts.
31. As a reviewer, I want the converter's dependency on the exllamav3 Python package pinned to a version and confined to the conversion step, so that no engine code links it.
32. As a reviewer, I want the converter's distortion curves, K map and per-layer errors reproducible from a fixed seed, so that a re-conversion of the same revision gives the same artifact.
33. As a reviewer, I want the converter to print a plain end-of-run report (rate per projection, K histogram per layer, distortion per layer, KLD per domain, MMLU, time), so that a conversion can be judged without opening the artifact.
34. As the owner, I want the BF16 source read either from a local copy or straight from the hub with range requests, so that a full 250 GB local copy is not required.
35. As the owner, I want the n-gram table's 96 GB BF16 source cache usable as the table's source and deletable afterwards, so that disk space is reclaimed after conversion.

## Implementation Decisions

**Owner-made decisions** (2026-10-04): a K per expert by allocation; ExLlamaV3 as
a reference only; the model switch is a later phase; the fast version directly.
**Agent proposals**, which the owner may veto, are marked *(proposed)*.

**The converter is an offline Python tool in the repo's tools tree.** It is
layer-streamed: it never holds more than one decoder layer's BF16 weights on the
GPU. It runs on the same machine as the engine, with the GPU profile's rules:
- one run on the card at a time;
- a preflight that refuses a busy GPU;
- a VRAM cap below 28 GB;
- resumable per layer: each finished layer is written to a work directory, and a
  restart skips it.

**Source.** The checkpoint at a pinned revision, read from a local copy or by
HTTP range requests into RAM. The study's fetcher reached ~42 MB/s with no disk
staging. The n-gram table source can be the study's local BF16 table cache.

**Calibration** *(proposed)*.
- **Corpus:** a manifest of public datasets and the owner's own text, in
  balanced shares across code, agent/tool traces, Italian prose, English prose,
  e-mail and technical documents, mathematics, exam-format questions, Chinese,
  and any language the owner adds. The study's run 4-6 mix is the starting
  point.
- **Split:** a held-out split for evaluation, disjoint by source document.
- **Excluded:** contributed fixtures and partner material; the manifest is
  checked against an allowlist of sources.

**Hessians.** For each expert, the input second moment over the calibration
tokens routed to it, weighted by the square of the routing weight. It is shrunk
5% toward the layer's whole-MoE Hessian, as in the study. An expert with no
routed token falls back to the layer Hessian.

**Allocation.**
- Each expert projection is encoded at every K class: 2, 2.5, 3 and 4.
- The proxy distortion of each encoding gives the expert's curve.
- A Lagrangian picks one K per expert so that the mean rate, scales included, is
  at most 2.5 bits. Gate/up and down are allocated separately.
- The encodings of the chosen K are kept from the sweep, never re-encoded.
- Run 8 measured ~650 s per layer for the four-class sweep, so a full conversion
  is about 9 hours.

**Expert encoding** *(proposed, ADR 0044)*.
- ExLlamaV3's trellis tensor, bit for bit: mul1 codebook, 16×16 tiles, a 128-wide
  Hadamard rotation on both sides, fp16 input and output channel scales.
- Gate and up are one fused plane of 2560 × 1280 and down is 640 × 2560, each
  with its own K.
- The encoder is the exllamav3 package's quantizer (MIT), pinned, fed our
  Hessians and K. It runs only inside the converter.

**Expert layout** *(proposed)*.
- Each **expert projection** (fused gate/up, or down) is one contiguous range,
  tiles and scales together, aligned to 4 KiB.
- Its byte size depends only on its shape and K, so there are eight **K classes**
  (two shapes × four K). The expert projection is the unit residency moves
  (spec 03).
- Projections are ordered by layer, then expert id, then gate/up before down. A
  per-layer index maps (expert id, projection) to offset and K.
- The engine copies one expert projection with one host-to-device transfer.

**Non-experts.**
- FP8 E4M3 with one fp32 scale per output row, using the container's existing
  row-scale layout, for every linear that is not an expert.
  - The study measured FP8 as lossless (MMLU-Pro 74.4% with BF16 experts) on the
    GDN projections, attention q/k/v/o, the indexer projection and the shared
    expert.
  - The hyper-connection mixes, the n-gram projections, the embedding and the
    output head were **not** measured in FP8. FP8 is proposed for them too
    *(proposed)*; the self-check measures their cost (Acceptance 4), and any
    that costs measurably stays BF16.
- Norm weights and small vectors stay BF16.
- The router weight stays BF16 *(proposed)*: 63M parameters, 126 MB in all,
  and it decides a discrete selection.

**The n-gram table.**
- 320M rows × 160 dims, INT4 with one fp16 scale per 32 values: 90 bytes per row
  (study run 6; INT2 rejected).
- It is stored as one object with a new role, **host-streamed**. The binder
  consumes it by handing its file range to the n-gram reader (spec 04), not by
  materializing it.
- Rows are directly addressable: row r is at offset r × row stride. The hash
  buffers and per-head sizes the checkpoint uses to map n-grams to rows are
  stored as small objects.
- A hot-row list, ranked by frequency over the calibration corpus and sized for
  2 GB, is stored as an object. The engine decides how much of it to load.

**Container.**
- `.ninfer` v2, as ignis's reader parses it: prefix, closed JSON directory,
  payload.
- New format and layout codes are added to the directory vocabulary for the
  trellis expert, the per-row FP8 linear and the INT4 table. The Rust reader
  learns them in this spec.
- Object counts and file size are sidecar invariants, as for the 27B (spec
  artifact/03).
- The artifact family is named so that it cannot be mistaken for a 27B artifact.

**Sidecar.** A JSON sidecar beside the artifact records:
- source revision, converter commit, exllamav3 version;
- the corpus manifest;
- the K map and the K histogram per layer and projection;
- per-K-class traffic shares;
- per-layer MoE error in dB on held-out tokens;
- the end-of-run KLD per domain and MMLU-Pro proxy.

**One pass, two references.** The BF16 checkpoint cannot be run
autoregressively here: 250 GB do not fit, and a layer-streamed greedy token
would re-read every layer. So every reference is teacher-forced and comes from
the conversion pass itself, as in the study (run 8's pipeline):
- Per layer, the pass propagates three hidden-state streams over the same
  windows:
  - **BF16**, the checkpoint;
  - **quantized**: decoded trellis experts, FP8 non-experts, INT4 table, which
    is exactly what the artifact holds;
  - **FP8 only**: BF16 experts with FP8 non-experts, for Acceptance 4's second
    variant.
- Windows in the pass:
  - the study's calibration chunks;
  - its 2048-token test chunks;
  - 8 windows of 8192 tokens from the held-out long documents (the 27B KLD
    study's long windows), which exercise the sparse QSA path in transformers.
    At 8192 tokens a layer's states are ~0.1 GB per stream per window; size
    the batch for it;
  - the **canary sequences:** the 27B canary fixture's prompts and recorded
    completions, re-rendered with Flash-Next's chat template and tokenizer and
    fed as fixed text.
- **Stored:**
  - the BF16 stream's top-64 log-probabilities and log-sum-exp per position
    (the KLD reference for spec 04);
  - the quantized stream's argmax and top-64 per position (the engine
    reference for spec 04's G1 and numerics);
  - the quantized stream's routing (spec 03's traces).
- **Measured in the pass and written to the sidecar:**
  - the per-layer MoE error;
  - the quantization-only KLD per domain, computed exactly and with the top-64
    scorer spec 04 uses on the engine, plus top-1 agreement;
  - the MMLU-Pro proxy (281 questions, paired against BF16).
- No second BF16 pass is ever needed.
- **Self-check after writing:** the converter reopens the written artifact and
  decodes a sample of expert projections per K class per layer (all 48 layers)
  through ExLlamaV3's `reconstruct`. They must be bit-identical to the weights
  the pass used.

**Disk.**
- The output takes about 71 GB: experts 38, non-experts 4.3, table 28.8.
- The Rust packer appends each finished layer's work files to the container and
  deletes them. The peak is the container plus about one layer, never two
  copies.
- The converter checks the free space at start and refuses below output + 15 GB.
- The stored references (top-64 on ~0.2M positions) are a few hundred MB.

## Testing Decisions

A good test checks what the artifact is: what the reader sees, what the
decoder reconstructs, what quality the written file gives. It does not check how
the converter got there.

- **CPU, in `cargo test`:**
  - the Rust reader parses a small Flash-Next-shaped fixture artifact (2 layers,
    8 experts in at least three K classes, a 1,000-row table) that the converter's
    writer produces;
  - the Flash-Next binder plan consumes every object, and an added stray object
    fails the bind;
  - the per-layer expert index resolves (expert id, projection) to its offset,
    size and K;
  - a host-streamed object is not materialized.

  Prior art: the artifact crate's `fixture` writer, `real_artifact.rs`, the
  checksum tests of spec artifact/03.
- **Converter unit tests** (pytest, in the tool's directory, run by the tool's
  README command):
  - the allocation meets the rate budget exactly, scales included, and respects
    the K set;
  - the writer's layout round-trips;
  - the hot-row ranking is deterministic;
  - the corpus manifest rejects a source outside the allowlist.

  Prior art: the Python tools under the repo's tools tree.
- **Machine-local, with the real artifact present:**
  - the reader opens the real artifact and every object is consumed;
  - a handful of experts per K class decode through `reconstruct` to the same
    tensor the conversion pass used.
- **GPU, run once per conversion:** the conversion pass's measurements are the
  acceptance measurement. It runs under the GPU profile's rules, never
  concurrently with another run.

## Acceptance

1. A single converter command, resumable per layer, produces the artifact, its sidecar and the reference recordings from the pinned checkpoint revision, in one layer-streamed pass. It refuses a busy GPU, stays under 28 GB of VRAM and checks disk space before starting.
2. Experts: mean rate ≤ 2.50 bits per weight including scales, for gate/up and for down separately. Every expert has a K in {2, 2.5, 3, 4}. The K histogram per layer is in the sidecar.
3. Per-layer MoE error on held-out tokens, measured on the written artifact: mean over layers 1-5 within 0.5 dB of run 8's −16.8 dB or better. Every layer of 0-47 is reported beside run 6's per-layer number. No layer is worse than run 6's by more than 1 dB without being named in the report.
4. Whole-model, quantization-only, on the study's test chunks (2048-token windows, teacher-forced, BF16 reference): KLD per domain ≤ run 6's for the same domain × 1.1 (run 6: code 0.119, prose 0.177, en 0.114, it 0.056, zh 0.089, math 0.044, py 0.028, de 0.224, ja 0.211). MMLU-Pro proxy ≥ 71% on the 281 questions and not significantly below BF16 (paired McNemar, p > 0.05). A second variant with only the non-experts quantized (BF16 experts) is reported beside it, so the cost of the FP8 linears the study did not measure is visible. If either misses, the converter reports FAIL and names the 3.0-bit fallback; nothing is shipped silently.
5. Non-experts are FP8 with a per-row scale and the router BF16. The n-gram table is INT4 g32 at 90 bytes per row, host-streamed, with its hash buffers and a 2 GB hot-row list.
6. ignis's reader parses the artifact. The Flash-Next binder plan consumes every object (ADR 0002), and the CPU fixture tests cover the new format codes, the expert index and the host-streamed role.
7. The pass stores: the BF16 top-64 log-probabilities and log-sum-exp on the 2048- and 8192-token windows; the quantized reference's argmax and top-64 on the canary sequences and the same windows, in a G1 fixture that carries an expected argmax per position; the quantized routing traces per domain for spec 03. Reopened expert projections decode bit-identical to the weights the pass used.
8. The sidecar records the source revision, converter commit, exllamav3 and transformers versions, corpus manifest (allowlisted sources only), K map, per-K-class traffic shares, per-layer errors, KLD per domain and MMLU result.
9. `cargo test` passes workspace-wide. The converter's unit tests pass.

## Out of Scope

- Serving the artifact: the MoE kernels (spec 02), expert residency (spec 03),
  the model forward (spec 04).
- A trellis quantizer of our own: the converter uses exllamav3's.
- Re-opening uniform versus allocated bit widths (ADR 0044). A uniform K map is
  expressible but is not this spec's deliverable.
- The MTP head: it is not converted.
- Publishing the artifact.
- The model switch at runtime: phase 2.

## Further Notes

- Prerequisites to `ready-for-agent`: ADR 0043 and ADR 0044 accepted or amended
  by the owner.
- The study's code is the starting point (layer-streamed forward, fetcher,
  Hessians, allocation, n-gram table quantization, evaluation). It lives in
  `.scratch/flash-next-compression-2026-10-03/real/`, which is untracked. The
  converter moves what it needs into the tools tree; nothing in `.scratch` is
  referenced at run time.
- The ADRs and specs cite the study in `.scratch`, which is untracked but local
  to the clone the owner works in. No finding is required before the ticket
  (owner, 2026-10-04).
- Run 8 measured the trellis only on layers 0-5. Acceptance 3 and 4 are the
  first whole-model measurement of the trellis format, and the 3.0-bit fallback
  (45.3 GB pinned) still fits the memory plan.
- Both references are the checkpoint's own modeling code, never another engine (ADR 0043): the BF16 weights for quality, the artifact's decoded weights for the engine's numerics.
