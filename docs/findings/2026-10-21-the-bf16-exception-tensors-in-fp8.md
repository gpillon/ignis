# The 27B's BF16 exception tensors in FP8: −0.53 GB, +3% cold prefill, KLD unchanged, but two decision fixtures move

- Kind: experiment
- Status: current
- Observed: 2026-10-21
- Last verified: 2026-10-21
- Scope: kernel / the FP8 arms of `attn_input_proj` and `linear_add` (W8A8 grouped GEMM with a residual epilogue, the FP8 linear's GEMV and tensor-core routes), the row quantizer; artifact / FP8 conversion at load (binder, materializer); serving / the 27B's VRAM plan, cold prefill, decode, a decoding lane's pauses, load time, 4K KLD on the RTX 5090 (PCIe Gen 3)
- Topic: artifacts-and-gates
- Related: GitHub #341 (spec [prefill-5090/01](../specs/prefill-5090/01-the-prefill-fixed-part.md), lever 1), [ADR 0065](../adr/0065-the-bf16-exception-tensors-can-be-held-in-fp8.md), [the prefill chunk at 2,048](2026-10-21-the-27b-prefill-chunk-at-2048.md) (the same session's method, the baseline width), [image model kernels](2026-10-13-image-model-kernels-on-the-5090.md) (the W8A8 GEMM), `.scratch/341/`
- Superseded by: none
- Summary: Held in FP8 row-scale and converted at load, the 27B's nine BF16 exception matrices free 534.6 MB of VRAM and speed cold prefill 2.8-4.1% at the same 4K KLD (0.0490 against 0.0492), but two decision fixtures move, so FP8 stays opt-in.

## Question

The artifact keeps nine matrices in BF16 among the NVFP4 body (GQA 3-23 `attention/query_key_gate_value`, GQA 3 and 7
`attention/output`, GDN 4 `gdn/output`; 1.07 GB). Held in FP8 row-scale, converted at load, and multiplied on the FP8
tensor cores in prefill: how much prefill does that buy, at what cost in quality, decode and load time? The owner's
condition: no conversion at load if it lengthens the load.

## Evidence

**The kernels alone** (`ignis_fp8_exception_bench`, synthetic weights at the real shapes, median of 50 launches,
`.scratch/341/bench-fp8-exception-*b.log`):

| T = 2,048 | µs | TFLOPS |
|---|---:|---:|
| BF16 qkgv (vendored) | 1,568.8 | 191.6 |
| BF16 out + residual (vendored) | 709.6 | 181.6 |
| E4M3 quantization of x [T][5120] / [T][6144] | 15.0 / 16.6 | |
| FP8 qkgv, four launches | 721.2 | 416.8 |
| FP8 qkgv, one grouped launch | 583.7 | 515.0 |
| FP8 qkgv, one fused [14336] GEMM (the bound) | 589.2 | 510.3 |
| FP8 out + residual | 240.4 | 535.9 |

The nine per token: BF16 5.636 µs, FP8 2.131 µs (−3.505). At T = 1,024: 5.758 against 2.233; four launches there keep
only 2.14 µs of the 3.53, because k and v fill 64 of the 170 SMs.

**Correctness** (CTests, all green): the quantizer against its contract restated on the host, every code and scale; the
grouped GEMM bit for bit one launch per segment (up to the real 6144 | 1024 | 6144 | 1024 x 5120 at 2,048 tokens); both
residual adds bit for bit BF16(residual + the fp32 product); the FP8 linear against fp64 at [5120][6144] and
[6144][5120]; the dispatch arms' q | k | gate | v each against fp64 of its own rows. On the card
(`fp8_exceptions_gpu`): the nine converted slots are, byte for byte, what `ignis_artifact::fp8` writes from the
artifact's BF16; the leaf's `vram_bytes` shrink by exactly the conversions' saving; decode through the captured graphs
equals eager decode. `vram_plan_gpu`: an FP8 load holds what its plan laid out.

**The real weights quantize evenly** (`.scratch/341/fp8_weight_error.py`): relative Frobenius error 0.0264-0.0265 in
every one of the nine, row amax / row RMS median 4.0-4.2 (max 17-66), 0.01% of codes subnormal, none zero.

**KLD** on the 2026-09-24 4K suite, chunk 2,048, both dumps from one binary on H:, scored in one `ref_kld.py` run:

| | KLD | top-1 |
|---|---:|---:|
| BF16 exception tensors | 0.04916 | 92.58% |
| FP8 | 0.04900 | 92.56% |

Difference −0.00015, 95% CI [−0.00151, +0.00098] (window bootstrap); 8 of 16 windows lower. The BF16 dump reproduces
the chunk lever's 0.04916 exactly.

**A/B**: one binary, `IGNIS_MODEL_EXCEPTION_WEIGHTS` fp8 / bf16 / fp8 / bf16 in one session, every other make default
(524,288 context, chunk 2,048 from the rtx5090 profile, vision, DFlash2/7), `.scratch/341/ab341.py`:

| cell | BF16 tok/s | FP8 tok/s | FP8 vs BF16 | µs per token |
|---|---:|---:|---:|---:|
| 8.2K (the warm-up) | 9,337 (9,242-9,431) | 9,408 (8,793-10,023) | +0.8% | 107.1 → 106.3 |
| 14K | 10,526 (10,493-10,562) | 10,961 (10,905-11,035) | **+4.1%** | 95.0 → 91.2 |
| 46K | 9,901 (9,869-9,924) | 10,250 (10,207-10,285) | **+3.5%** | 101.0 → 97.6 |
| 105K | 8,033 (8,031-8,037) | 8,262 (8,235-8,281) | **+2.8%** | 124.5 → 121.0 |

- Decode: C=1 370.7 → 373.0 tok/s, C=8 1,060.2 → 1,084.2; ms per round 16.70 → 16.76 (C=1), 28.43 → 28.49 (C=8).
  Greedy text identical within each build 8/8, across builds 2/8; τ at C=8 5.309 → 5.477 on the changed texts.
- A 32.7K prompt alone 3.23 → 3.12 s; beside two decoding lanes 4.36 → 4.19 s, the lanes at 17 tok/s both ways,
  longest gap 300 → 293 ms.
- **The load** (`ignis.model.loading` → `loaded`, `.scratch/341/load_times.py`): FP8 11.95 s (the session's first leg)
  and 11.01 s; BF16 11.25 and 11.17 s. Six load-only starts with BF16 first (`load_pairs.py`): BF16 12.93 (the
  first), FP8 11.43, BF16 11.02, FP8 11.04, BF16 11.13, FP8 11.04 s.
- The weights line 17,234,723,840 → 16,700,152,832 B (−534,571,008); the decode-graph line 82,125,920 → 82,441,824 B
  (+315,904: a verify round past 8 rows quantizes its activations too); the KV pool 836-837K → 895K tokens.
- The rtx5090 profile at 917,504 tokens (`profile_probe.py`, budget 28,952,772,608 B today, the desktop holding
  ~3.5 GB): the plan needs 29,698,716,928 B with BF16 (refused by 745.9 MB) and 29,164,461,824 B with FP8 (refused by
  211.7 MB).

**The decision fixtures** (the GPU profile's `decide_entailment_gpu` and `decide_scalar_gpu`, `EngineShape::default()`
at chunk 1,024, `.scratch/341/gputests-server.log`, `expA/expB/expC-*.log`). The entailment test reads p(yes) of 18
fixed cases and fails when a case not marked "watch" loses its label; the scalar test reads six numbers out of their
evidence. Logits of p(yes):

| case | BF16, today's build | FP8 (W8A8 past 8 rows, as built) | FP8 weights, BF16 activations (experiment) | BF16, 2026-10-06 build |
|---|---:|---:|---:|---:|
| 1 Direct field read | +1.750 | **−0.750 wrong** | +5.500 | +3.000 |
| 12 Absence (watch) | −3.125 | −0.375 | +0.625 | +0.875 |
| 16 Date restatement (watch) | −1.500 | −0.625 | +0.250 | +0.250 |
| 17 Proportion | +0.125 | +0.250 | **−0.625 wrong** | +1.625 |
| labels kept | 15/18 | 14/18 | 16/18 | 17/18 |

The scalar `balance` (truth −0.75): BF16 −0.75 at an uncertainty of 0.0001; FP8 0.75 (0.038), and 0.75 (0.148) with
BF16 activations. Every other GPU test of the profile set is green under FP8 as the default (CTest 133/133, core,
runtime, the attention taps).

**Which of the nine** (owner, 2026-10-21: find which one loses the sign; a temporary name filter on the conversion,
both decision tests per subset, `.scratch/341/bisect-round*.txt`). The scalar's −0.75 and the entailment's cases 1 / 17:

| converted | case 1 | case 17 | balance |
|---|---:|---:|---|
| all nine | −0.750 | +0.250 | 0.75 (unc 0.038) |
| the six qkgv | −0.500 | 0.000 | −0.75 |
| attention/output, L3 and L7 | +1.875 | −0.625 | 0.75 (0.123) |
| gdn/output, L4 | +0.375 | +1.750 | −0.75 |
| attention/output L3 | +4.875 | +0.125 | 0.75 (0.093) |
| attention/output L7 | +3.375 | −1.250 | −0.75 |
| qkgv + gdn (seven) | +4.625 | +1.500 | 0.75 (0.038) |
| all but attention/output L3 (eight) | +3.000 | +2.125 | 0.75 (0.213) |
| qkgv 11-23 + attention/output L7 + gdn | +0.750 | +0.375 | −0.75 |
| qkgv 11-23 | +3.625 | +0.250 | 0.75 (0.255) |
| qkgv 11-23 + gdn | +1.125 | −0.125 | 0.75 (0.252) |

No single matrix carries it. The sign is lost by the L3 output alone, by the four late qkgv alone, and by the qkgv
with the GDN output (seven), but kept by each of the six qkgv, the GDN output or the L7 output alone. Adding a
matrix to a passing set can lose it, and adding one to a losing set can restore it. Under any of these perturbations
the read's uncertainty rises from BF16's 0.0001 to 0.04-0.25, and past that the sign behaves like a coin. The one
subset that keeps every fixture (late qkgv + L7 output + GDN output) is that coin, not a criterion.

**A false alarm on the way** (kept for the method). `fp8_exceptions_gpu`'s first acceptance was the two loads'
top-1 agreement over a 362-token prompt (one sentence eight times): 0.635, mean KL 2.28 nats, positions at 17 nats.
Every kernel was exonerated by rerunning with BF16 activations everywhere and then with the GEMV in 8-token slices:
the same numbers. What the prompt shows instead: the BF16 load reads the text's own next token at 30.5% of the
positions and the FP8 load at 29.6%, so both are uncertain there and their argmaxes flip between near-ties; and past 4
tokens every NVFP4 layer quantizes its activations to E2M1 per 16 values, which turns a 2-3% difference upstream into
whole code steps downstream (a single-layer comparison read 10% at every width >= 9 against 0.6% at one token, for
the same reason). The test now checks the next-token rate (within 0.05) and leaves quality to the KLD.

## Finding

Observed:

1. **FP8 holds the KLD**: 0.04900 against 0.04916, inside the noise.
2. **Cold prefill +2.8% to +4.1% from 14K to 105K**, every FP8 sample above every BF16 one; −3.4 to −3.8 µs per token,
   what the kernels alone save (3.5). The 8K cell is within its spread.
3. **Decode is unchanged in time**; its greedy text changes, as expected of other weights. A decode round and a
   one-lane verify round multiply them with BF16 activations; a verify round of two lanes or more (16+ rows, the C=8
   legs) quantizes its activations as prefill does.
4. **The load does not lengthen**: 11.04 / 11.04 s with FP8 against 11.02 / 11.13 s with BF16 on a warm file cache;
   the first load of a series is 1-2 s slower whichever encoding it loads. The upload is the stored bytes; the conversion is a GPU pass over 1.07 GB.
5. **−534.6 MB of VRAM**, which the KV pool takes (+58K tokens at the make default). With vision on the activation
   codes cost no prefill workspace (the encoder sizes the arena; +5.5 MB on a text-only load at 2,048); the decode
   graphs' scratch grows 0.3 MB.

6. **Two decision fixtures read differently**: the entailment's case 1 (+1.75 → −0.75 logits) and the scalar's
   negative balance (its sign lost). The second does with BF16 activations too.

Inferred:

- The fixtures' logits move 1-4 between builds of BF16 alone (today's against 2026-10-06: case 1 +3.0 → +1.75, case 12
  +0.9 → −3.1), so a single case crossing its threshold is within what the engine's numerics do; the scalar's sign,
  read at 0.0001 uncertainty in BF16 and lost under both FP8 forms, is the one that looks like the weights' own error
  (2.6% against BF16's ~0.4%, E4M3's three mantissa bits: no scale choice reduces it).
- Decode rounds read 0.53 GB less but take as long: the nine are ~3% of a round's bytes, and the FP8 GEMV is not
  faster than the vendored BF16 decode kernels at that share. Not profiled.
- Per-token E4M3 activations suffice here (no outlier rows, KLD unchanged); block-scaled activations are not needed.

## Implications

- **FP8 is an option, not the default** (`model.exception_weights`, `bf16` by default; ADR 0065 proposed): it is
  faster, smaller and KLD-equal, but it fails two pre-registered decision fixtures, which is the owner's call, not the
  agent's. The artifact does not change either way.
- **AC 2's fixed part, with the chunk and FP8 levers**: from this session's FP8 legs less #335's attention (~21 / ~45 µs
  at 46K / 105K), ~77 / ~76 µs per token, under the ≤ 85 target at long context; at 8K, the warm-up, ~103.
- **The 917,504 profile is refused today with either encoding**: it depends on ~0.5 GB of desktop VRAM that moved
  between this morning's probe and tonight's (owner's call, the margin should be configurable).
- **Method**: two engines' agreement on a peaked, repetitive prompt is not a quality measure on this model; the KLD
  against the BF16 checkpoint is.

## Limits and unknowns

- One session, two legs per encoding; one prompt family per prefill cell. Both arms are one binary: the `bf16` arm is
  main's numerics (the same KLD to the digit, the same greedy text) but not a main build, so it does not show main's
  speed is unchanged; the BF16 arm ran 1.5-3.2% above the chunk session's legs, the session drift.
- The image model's W8A8 GEMM gained the segments and the residual add as template parameters (the one-segment
  instantiation is the kernel before, instruction for instruction): against a build of 8dc115d6 in the same session
  `ignis_image_gemm_bench` reads 4096 x 4096 at 0.271-0.275 ms against 0.272, 12288 x 4096 at 0.770-0.778 against
  0.763-0.816 (a first version with runtime branches lost 2-4% there). Flash-Next's FP8 linear gained a store branch
  in its epilogue, its speed not re-measured.
- The KLD is the 4K suite's: long-context KLD not measured.
- The decode round's share of the FP8 GEMV against the vendored BF16 kernels is not profiled.
- The spec's levers 3 (the backbone GEMMs), 4 (the GDN prefill) and 5 (a GPTQ encoding) are this ticket's next steps.

## Follow-ups

- Owner: FP8 as the default (and the two fixtures re-registered), or opt-in as built; accept or amend ADR 0065.
- The bisection (above) finds no single matrix: a decision on the default is one on how much this fixture's
  knife-edge weighs against +3% prefill, not a choice of which matrices to convert.
- The 917,504 profile's margin against the desktop (with the chunk lever's finding).
