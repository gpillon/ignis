# Rust quantized matmul matches ggml on the CPU, at Qwen3-ASR's shapes

- Kind: experiment
- Status: current
- Observed: 2026-10-24
- Last verified: 2026-10-24
- Scope: speech / CPU quantized matmul (Q4_K and Q6_K against Q8_K activations, Q8_0 against Q8_0) written in Rust,
  against ggml b11840, at the shapes of Qwen3-ASR-1.7B Q4_K_M and its Q8_0 audio encoder: one decode token, a
  512-token prefill and one encoder window, on the owner's i9-10900K (Windows, 10 threads) and a 4-core podman
  container (Linux, 4 threads)
- Topic: speech
- Related: [ADR 0068](../adr/0068-the-speech-models-cpu-executor-is-ours-in-rust.md),
  [spec speech/01](../specs/speech/01-cpu-executor.md),
  [Qwen3-ASR on the CPU](2026-10-24-qwen3-asr-on-cpu.md)
- Superseded by: none
- Summary: Rust quantized matmul kernels with no C match ggml b11840 at Qwen3-ASR's shapes (decode at 0.954 / 1.014 of ggml's time on the i9 / 4 cores, prefill 0.736 / 0.597), so what the CPU needs is quantized SIMD matmul, not ggml.

## Question

Can kernels we write in Rust, with no C, run the speech model's quantized matmuls about as fast as ggml, the library
the CPU measurement ran on? The criteria were fixed before measuring (the owner's Q61), on the same host and the
same weights:
- decode, one token: Rust time at most 1.111x ggml's (throughput at least 90% of ggml's);
- prefill, M = 512 over the 28 decoder layers: at most 1.5x;
- encoder, M = 104, one window: at most 1.5x.

## Evidence

**What was compared.**
- **The baseline is the measured release itself.** The harness loads the b11840 release's own ggml libraries at
  run time and drives them through ggml's public C API from the same Rust process: on Windows `ggml-base.dll`,
  `ggml.dll` and `ggml-cpu-haswell.dll` from `llama-b11840-win-cpu-x64` (Clang 20.1.8, LLVM OpenMP), in the
  container the ubuntu tarball's `.so` files (GCC 11.4, libgomp). ggml reports
  `features [SSE3 SSSE3 AVX AVX2 F16C FMA BMI2 LLAMAFILE=1 OPENMP=1 REPACK=1]` on both hosts; the Haswell variant
  is pinned, the one llama.cpp picks on this CPU.
- **Repack where llama.cpp uses it.** At b11840 ggml repacks Q4_K on AVX2 (its log shows `q4_K_8x8` for all 168 Q4_K
  tensors), not Q6_K or Q8_0. Q4_K without repack measured within noise of the repacked runs on this CPU (decode
  30.96 ms against 31.4-32.2; prefill 1,572 ms against 1,496-1,544); the baseline uses repack anyway.
- **Same work on both.** The same GGUF bytes, the same f32 activations, the same matmuls in model order with
  distinct per-layer weights. ggml runs one graph per decode token, prefill or window, one `mul_mat` node per
  matmul. Rust dispatches each matmul and quantizes its input itself, as each `mul_mat` does; nothing is fused.
- **Rust build.** rustc 1.98.0, `opt-level=3`, fat LTO, one codegen unit, **no `target-cpu`**. The kernels are
  `#[target_feature(enable = "avx2,fma,f16c")]` functions picked after `is_x86_feature_detected!`. Targets
  `x86_64-pc-windows-msvc` and `x86_64-unknown-linux-gnu`. Only dependency: rayon 1.12.0.

**Shapes** (read from the GGUF headers):
- **Decoder** `Qwen3-ASR-1.7B-Q4_K_M.gguf` (`qwen3vl`: 28 blocks, hidden 2048, FFN 6144, 16/8 heads of 128). Q4_K
  everywhere except `attn_v` and `ffn_down` in 14 layers and `output.weight` (2048 → 151,936), which are Q6_K. One
  decode token is 197 matmuls over 1,101.0 MB of weights. Prefill has no LM head: 845.7 MB, 721.6 GMAC at M = 512.
- **Encoder** `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` (24 blocks, d 1024, FFN 4096) plus the two projector layers: 146
  Q8_0 matmuls, 324.2 MB, 31.7 GMAC per window. M = 104 because the encoder attends inside 8 s windows of 8 chunks of
  13 tokens, and llama.cpp encodes one window per call. The F16 convolutions, the F16 `conv_out` and the F32
  position embedding are not Q8_0 and were not timed.

**Method.**
- Rust and ggml repetitions interleaved A B A B after 2 warm-up passes each; the median of at least 7 repetitions
  (decode 9). A repetition is 20 decode tokens (reported per token), one prefill pass, or 5 encoder windows.
- A 1 s idle gap before each half: 500 ms for idle workers to stop spinning, 500 ms recording system-wide busy %.
- Both engines unpinned. Every run waited until no build, ignis, llama or python process had been seen for 20 s
  and was flagged if one appeared; the two flagged comparison runs were discarded and run again.

**Correctness, before any timing counted.**
- Tensors Q4_K `blk.0.ffn_gate`, `blk.0.attn_k`; Q6_K `blk.0.ffn_down`, `blk.27.attn_v`, `output.weight`; Q8_0
  `a.blk.0.ffn_up`, `a.blk.0.ffn_down`, `mm.a.mlp.2`; M = 1, 7 (remainder rows), 104 and 512; the scalar kernels,
  the AVX2 kernels, every register-tile shape the benches can pick, both Q4_K layouts; ggml on the same input.
- Reference: fp64 dot products of the dequantized weights with the dequantized quantized activations, over every
  output element. Error is |y − ref| / Σ|w·a| for that element. Gates set before running: 1e-5 against fp64, 1e-4
  Rust against ggml.

| Check | Worst observed | Gate |
|---|---|---|
| Rust AVX2 (all tile shapes, both layouts) against fp64 | 1.14e-7 | 1e-5 |
| Rust scalar against fp64 | 9.4e-8 | 1e-5 |
| ggml against fp64 (same activations) | 1.64e-7 | none |
| Rust against ggml | 1.84e-7 (Q8_0 bit-identical) | 1e-4 |
| Q8_K and Q8_0 activation quantizers, scalar against AVX2 | bit-identical in every case | none |

- **Negative controls:** one block scale doubled in the Rust copy only is caught by both checks on both layouts
  (Q4_K 1.24e-1, Q6_K 5.85e-4, Q8_0 2.29e-2).
- **Resolution:** a single Q6_K sub-scale nudged by +7 in a 6,144-long row gives 2.5e-6, under the gate. The gate
  was not tightened after the fact.
- Identical results on Windows and Linux. Each bench run also compares the two engines' first and last op: relative
  L2 difference at most 4e-7.

**Results**, median of the interleaved repetitions; criterion rows in bold.

| Host | Workload | Rust ms | ggml ms | Rust/ggml | Threshold |
|---|---|---|---|---|---|
| i9, 10 thr | **Decode, 1 token** | 30.69 | 32.17 | **0.954** | 1.111 |
| | **Prefill, M = 512** | 1,136.8 | 1,544.0 | **0.736** | 1.5 |
| | **Encoder, M = 104** | 140.6 | 161.4 | **0.871** | 1.5 |
| | Prefill, M = 104 (as llama.cpp prefills audio) | 240.2 | 328.5 | 0.731 | none |
| 4-core, 4 thr | **Decode, 1 token** | 40.81 | 40.23 | **1.014** | 1.111 |
| | **Prefill, M = 512** | 2,339.9 | 3,919.0 | **0.597** | 1.5 |
| | **Encoder, M = 104** | 284.7 | 265.7 | **1.071** | 1.5 |
| | Prefill, M = 104 | 473.0 | 815.2 | 0.580 | none |

**How it got there.**
- **The first straightforward version missed two criteria:** prefill on the i9 (1.53) and the encoder in the
  container (1.54). Its kernel ran 44-49 GMAC/s on one thread: LLVM unpacked a whole superblock to the stack, and
  every 32 multiply-adds paid a scale `madd` and horizontal work.
- **Our own interleaved Q4_K layout decided prefill.** 8 weight rows per group; one 32-byte load holds 4 consecutive
  k of each of the 8 rows; the 6-bit scales and mins are decoded once at load. `maddubs` accumulates in i16 over a
  32-element sub-block, so the scale `madd` runs once per 8 of them. 1,184 B per 8×256 weights against the GGUF's
  1,152, **+2.8%**. Prefill went from 2.16 s to 1.14-1.20 s; decode on the same layout is unchanged within noise,
  so one layout serves both. The repack is done at load and was not timed.
- **The register-tile shape decided the encoder:** Q8_0 at 6×2 ran 138 ms against 189 ms at 4×1.
- **rayon fails decode in the container.** Decode is 197 dispatches a token, the smallest about 30 µs of work.
  rayon ran it at 1.134 (first version) and 1.020 (final) on the i9, and **1.184** in the container. Our own pool,
  about 130 lines (the main thread is worker 0, tasks claimed from an atomic counter, workers spin about 100k
  `pause` and then park), ran at 0.954 and 1.014. For the GEMMs rayon was as fast or faster (prefill 0.684, encoder
  0.792 on the i9).
- **Not tried** once the decision was clear: an interleaved layout for Q6_K (Rust's Q6_K GEMM is about 354 GMAC/s
  against ggml's 467), fused QKV or gate/up dispatch, tiles tuned on the 4-core host.

**Sanity against the end-to-end runs** of the CPU finding: ggml's matmul-only time is 89-92% of llama.cpp's decode
token on the i9 (85-94% in the container), about 80% of a 104-token prefill batch and about 60% of an encoder
window. The rest is attention, norms, RoPE, the convolutions and small ops. Decode streams 34-36 GB/s of weights
on the i9, at the host's dual-channel ceiling: both engines are memory-bound there.

**No C was needed.** The quantizers, the three dot-product families (scalar and AVX2), the interleaved layout and the
pool are `std::arch` intrinsics in Rust; no inline asm, no C shim, no global `target-cpu`. The executor part of the
spike is 1,492 lines (`avx2.rs` 508, `quant.rs` 405, `matmul.rs` 385, `pool.rs` 194); the harness, the ggml FFI
baseline and a GGUF reader are 1,317 more.

**Raw data** stays on `H:\speech-study\rust-spike\` (nothing binary enters the repo):
- `src\`, `Cargo.toml`, `vendor\` (rayon and its dependencies, for the offline container build),
  `container\Containerfile`, `tools\*.sh` (the gated runner and the rounds);
- `results\check_win.txt` and `results\ctr\check_linux.txt`: the correctness runs;
- `results\win\`: the i9 logs (`r1_*` first version, `f_*` and `f2_*` final, `f2_*` replacing the two flagged
  runs), `bench.csv` and `bench_final.csv` per repetition, `sweep\` and `sweep_x8\`, `kbench.log`,
  `profile_prefill.log`;
- `results\ctr\`: the container logs and `bench.csv`.

## Finding

**Observed:**
- Rust kernels written for this, with no C, met all three criteria on both hosts: decode 0.954 and 1.014 of ggml's
  time, prefill 0.736 and 0.597, encoder 0.871 and 1.071.
- Their results are at most 1.14e-7 from an fp64 reference, normalised per element, on every format, tile shape and
  layout; the activation quantizers are bit-identical between scalar and AVX2; doubling one block scale is caught.
- Two levers were needed: our own interleaved Q4_K layout (+2.8% memory) and the register-tile shape. A first
  version without them missed prefill on the i9 and the encoder in the container.
- rayon cannot dispatch decode fast enough on 4 cores (1.184); a spin-then-park pool of our own can.
- Runtime feature detection and `#[target_feature]` reach these speeds with no global `target-cpu`.

**Inferred:**
- What makes the CPU go is quantized SIMD matmul with a layout fit for it, not ggml as such: the CPU measurement
  compared llama.cpp only with Transformers FP32, and this run shows ggml's speed on the matmuls is reachable in
  Rust.
- On decode both engines stream the weights at the memory's ceiling, so neither can be much faster there; the
  margins that matter are prefill and the encoder, which are compute-bound.

## Implications

- ADR 0068: the speech model's CPU executor is ours, in Rust, with these kernels, our Q4_K layout and our own pool.
  No ggml enters the tree.
- Spec speech/01: the correctness method (fp64 reference with this normalisation and gate, negative controls,
  scalar and AVX2 quantizers bit-identical) becomes acceptance criteria.

## Limits and unknowns

- **Only matmuls were timed.** Attention, norms, RoPE, softmax, the encoder's F16 convolutions and sampling were
  not; in llama.cpp they are about 8-15% of decode, about 20% of prefill and about 40-50% of the encoder. Matching
  ggml on matmuls does not show the whole executor matches llama.cpp.
- **ggml ran a matmul-only graph,** through `ggml_backend_graph_compute`, without llama.cpp's scheduler or
  persistent threadpool. With OpenMP its workers are OpenMP's either way.
- **Tuning asymmetry.** Our tile shapes and layout parameters were picked from a few sweeps on the i9, on exactly
  these shapes, and applied unchanged to the container; ggml's kernels are general. The margins (5-40%) exceed any
  plausible overfit except container decode and the container encoder.
- **Noise.** The i9's idle gaps showed 1-34% system-wide busy (median about 10%) from the desktop and other
  sessions; the same configuration drifted ±5-10% between runs. A decode ratio is good to about ±5%, which matters
  most for the container encoder's margin (1.071 against 1.5) and least for container decode.
- **The container is optimistic for a budget 4-core:** it keeps the i9's dual-channel bandwidth and 20 MB L3, and
  WSL's vCPUs are not pinned.
- **One microarchitecture** (Comet Lake: AVX2, no AVX-VNNI, no AVX-512). On newer CPUs ggml loads other variants
  (AVX-VNNI, AVX-512); parity there would need matching kernels of ours, which were neither written nor measured.
- **Memory.** The spike held both the GGUF layout and the interleaved copy of every Q4_K tensor; an executor keeps
  only the interleaved one. The decode numbers already stream the interleaved copy.
- **The prefill criterion is at M = 512,** while llama.cpp prefills audio at M = 104; both were measured, and the
  Rust margin is larger at 104.
- **The decoder Q8_0 GEMV (the `qwen3-asr-1.7b-q8` and `qwen3-asr-0.6b` entries) was not timed;** Q8_0 was timed at
  M = 104 only, on the encoder.

## Follow-ups

- Spec speech/01: the whole executor's own times on the i9 and the 4-core container, per catalog variant.
