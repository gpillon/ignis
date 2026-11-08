---
status: accepted
---

# ADR 0068 — the speech model's CPU executor is ours, in Rust

## Status

Accepted (2026-10-24, owner): decisions Q59-Q63 of the speech grill of
2026-10-23/24 (ADR 0067 has the session), taken when the owner reviewed the
vendoring the grill had first chosen, and the spike Q61 asked for, which
passed every criterion fixed before it ran; Q64-Q68 settled the details the
rewrite left open. Spec
`docs/specs/speech/01-cpu-executor.md`.
- **Replaces** the grill's Q27 and Q41 on the executor (vendored ggml, the
  graphs adapted from llama.cpp, a CMake build behind a cargo feature). Q41's
  pipeline in Rust stands. Q58 (that cargo feature) is moot.
- **Applies ADR 0051**: the executor's kernels are of its first tier, our
  own. Nothing is vendored, so ADR 0010's manifest gets no second upstream.
- **Holds ADR 0032**: the default build is pure Rust, and the speech model is
  in it with no feature flag.
- **Departs from ADR 0009** for the speech model on the CPU only: its forward
  pass runs in Rust, not in the kernel leaf. ADR 0009 puts the forward pass in
  the leaf so that activations and sequence state stay resident on the
  device; a CPU executor has no device. The GPU loan's forward pass is in the
  leaf (spec speech/06).

Sources: the finding
[Rust quantized matmul matches ggml on the CPU](../findings/2026-10-24-rust-quantized-matmul-matches-ggml.md),
with the spike's source and raw data on `H:\speech-study\rust-spike\`; the
finding [Qwen3-ASR on the CPU](../findings/2026-10-24-qwen3-asr-on-cpu.md);
the session's decision log, kept untracked in
`.scratch/speech-grill-2026-10-23/` (local to the owner's clone).

## Context

ADR 0067 makes a CPU executor the speech model's baseline. The CPU
measurement (the first finding) says what it must run:
- The official Transformers path in FP32 is 2.8-2.9x slower than llama.cpp's
  Q8_0 on the same CPU. 1.7B then runs barely faster than real time.
- BF16 is the slowest variant on a CPU without BF16 instructions.
- Quantized SIMD kernels are what make the CPU go. Q4_K_M on AVX2, with its
  weights repacked at load, is 1.3-1.55x faster than Q8_0 at no measurable
  WER cost.

That measurement compared llama.cpp with Transformers FP32 and nothing else.
It shows that quantized SIMD kernels are needed, not that they must be
ggml's.

The grill first chose to vendor ggml (Q27, Q41): its core and x86 CPU
backend, about 70,000 lines of C and C++, built under `kernel-cpu/` by
CMake, behind a cargo feature `speech`, with a Haswell baseline and a CPUID
check before the first ggml call. The owner reviewed that chain the same
day:
- the x86-only cut was ours, made to keep the vendored line count down
  (ggml ships an ARM backend too);
- the Haswell baseline, the CPUID guard against an illegal instruction,
  a CMake build in every `make`, and the feature flag all existed
  only because of the vendored C++;
- the draft's claim that ggml's kernels do not stand alone was false for the
  quantized dot products, which take raw pointers and need only the block
  structs;
- **Q59**: the owner refuses both large foreign code in the tree and a
  platform shape dictated by someone else's code;
- **Q60**: the CPU executor is ours, in Rust (`std::arch` AVX2 now, NEON
  later, a scalar fallback everywhere, our own GGUF reader, the `tokenizers`
  crate, rayon or our own pool). C only if truly necessary, and then only
  ggml's dot products adapted as ADR 0051's adapted ops;
- **Q61**: a spike first, against ggml on the same host, with its criteria
  fixed before it ran.

**The spike passed every criterion, with no C** (the second finding). At the
shapes of Qwen3-ASR-1.7B Q4_K_M and its Q8_0 audio encoder, Rust's time
over ggml b11840's:

| | i9-10900K, 10 threads | 4-core container, 4 threads | criterion |
|---|---|---|---|
| decode, one token | 0.954 | 1.014 | ≤ 1.111 |
| prefill, M = 512 | 0.736 | 0.597 | ≤ 1.5 |
| encoder window, M = 104 | 0.871 | 1.071 | ≤ 1.5 |

- Correct first: at most 1.14e-7 from an fp64 reference, normalised per
  element by Σ|w·a|, against a gate of 1e-5; the activation quantizers
  bit-identical between scalar and AVX2; a doubled block scale caught on
  every format.
- It took three levers: our own interleaved Q4_K layout (+2.8% memory, which
  decided prefill), register-tile shapes (which decided the encoder), and our
  own spin-then-park pool. rayon dispatched decode at 1.184 of ggml's time
  on 4 cores.
- The executor part is about 1,500 lines of Rust. The kernels are
  `#[target_feature]` functions picked at run time, with no global
  `target-cpu`.
- **Its limits** (the finding's section on them):
  - only matmuls were timed. Attention, norms, RoPE, softmax, the encoder's
    convolutions and sampling are about 8-15% of decode, 20% of prefill and
    40-50% of the encoder in llama.cpp;
  - ggml ran a matmul-only graph, without llama.cpp's scheduler;
  - our tiles were tuned on the i9 and applied unchanged to the container;
  - a decode ratio is good to about ±5%, and the container is optimistic for
    a budget 4-core;
  - one microarchitecture: no AVX-VNNI or AVX-512 kernels were written, while
    ggml has variants for them.

## Decision

1. **A pure-Rust executor in the default build** (Q60; Q58 is moot). Its own
   code has no C, no C++, no CMake, no build script and no cargo feature, and
   it adds no `cc` or `cmake` build dependency the workspace does not already
   have. A bare `cargo build` carries the speech model on every target ignis
   builds, and ADR 0032's default build needs no new toolchain.
2. **Our own GGUF reader** (Q60): header, metadata, tensor table and the
   memory-mapped tensor data.
3. **Activation quantizers** f32 → Q8_K and f32 → Q8_0 (the spike).
4. **Matmul kernels** Q4_K × Q8_K, Q6_K × Q8_K and Q8_0 × Q8_0, plus F16 and
   F32 where the GGUF has them: the encoder's convolutional front end and
   `conv_out` (the spike; the GGUF headers). These cover the catalog's three
   entries (Q49).
5. **A scalar reference and AVX2/FMA/F16C kernels** (Q60, the spike), picked
   at run time with `is_x86_feature_detected!` and `#[target_feature]`. No
   global `target-cpu`, so one binary serves every x86-64 CPU. The AVX2
   module is compiled only on x86-64, and the scalar kernels everywhere.
6. **Q4_K is repacked at load into our interleaved 8-row layout** (the
   spike): +2.8% over the file's bytes, and the only copy of those tensors
   kept. ADR 0067's load at first use (Q53) is this repack.
7. **Our own thread pool**, spin then park (Q60, the spike): rayon failed
   decode on 4 cores (1.184). It runs at `speech.threads` (Q38).
8. **Both graphs are written by us, in Rust** (Q60):
   - the AuT audio encoder: the convolutional stem and `conv_out`, attention
     inside 8 s windows of 104 tokens, and the projector;
   - the Qwen3 decoder: 28 layers, QK-norm, GQA, M-RoPE, SwiGLU, and the LM
     head tied to the embedding in the checkpoint.
9. **Provenance: ours** (ADR 0051's first tier). The GGUF block layouts and
   the Q8_K/Q8_0 quantization rules are the file format, data the kernels
   read, not copied code. Code copied later from anywhere follows ADR 0051's
   adapted-op rules. **Amended** (final fixes): that later code came with
   phase 2. `crates/speech/src/vad.rs` ports Silero VAD's graph (MIT) and
   ships its weights, an adapted op under ADR 0051 with provenance in its
   header and `NOTICE`. The executor itself, its kernels and graphs, stays
   ours.
10. **The oracle is the checkpoint's own modelling code**, teacher-forced, as
    in ADR 0043 (Q52). llama.cpp b11840 is an out-of-tree comparison and
    benchmark baseline only, on `H:\speech-study\`, and the offline tool that
    converts the GGUF files (Q47). It is never in the tree or in the build.
11. **AVX2 stays the minimum** (Q28; the spike's kernels need FMA and F16C
    beside it). On a CPU without the three the CPU executor is unavailable,
    with ADR 0067's reason (Q35). Runtime selection removes the
    illegal-instruction risk the CPUID check guarded against: no AVX2
    instruction runs outside a function selected after detection.

One pipeline drives two compute adapters (Q41): this executor, and the CUDA
loan of phase 3 (ADR 0067), which builds on our kernel leaf alone, with no
ggml-cuda base (Q62). Both read the same GGUF files (Q46).

**Licences.** No code is copied into the executor, so ADR 0051's list does
not come into play for it; the VAD of decision 9's amendment is the one
adapted op, and `NOTICE` records it. Two dependencies are not copied code, and
`NOTICE` records the first of them, `symphonia` (the second was not added)
(Q42):
- `symphonia` (MPL-2.0) is an unmodified crate dependency for MP3, AAC and
  the Ogg and WebM containers. MPL-2.0's copyleft is per file and binds only
  its own files, which we do not change.
- `opus`/`opusic-sys` (BSD-3) builds a bundled libopus through the `cmake`
  crate, a new build dependency. The owner accepts it if it does not
  complicate the build or cost too much (Q63): it comes in phase 1 only if a
  verification build passes on both targets, otherwise later (Q42). With the
  executor in Rust it would be the only CMake in a build without `cuda`, and
  the verification weighs that. The C such a build already compiles (`ring`,
  and the oniguruma and esaxx parts of `tokenizers`) goes through the `cc`
  crate, with no CMake.

## Consequences

- No foreign code in the tree for the executor: no manifest entry, no
  `NOTICE` line for ggml, no re-vendor to upgrade. ADR 0010 and ADR 0051 are
  unchanged. The one exception is the VAD's Silero port and weights
  (decision 9, amended), under ADR 0051 and in `NOTICE`.
- CI's CPU test leg runs the executor's tests in the default build. The
  executor adds no toolchain to it: no CMake and no native build step of its
  own, unless libopus joins under Q63.
- The kernels, the layout, the pool and both graphs are ours to maintain.
  ggml's later speed-ups do not arrive by themselves; they are ours to write
  if we want them.
- Parity on CPUs newer than the i9 (AVX-VNNI, AVX-512) needs kernels of ours
  that were neither written nor measured. Nothing in the design is x86-only:
  NEON kernels take their place beside the AVX2 ones by the same runtime
  selection (Q60).
- The spike timed the matmuls alone. Whether the whole executor holds its
  speed is spec speech/01's measurement on the i9 and the 4-core container.

## Considered Options

- **Vendor ggml** (the grill's Q27 and Q41, the first draft of this ADR).
  Rejected by the owner's review (Q59, Q60):
  - about 70,000 lines of foreign C and C++ in the tree;
  - an x86-only cut, made by us to keep that count down;
  - a Haswell baseline that needed a CPUID check before the first call, or
    the process died on an older CPU;
  - a CMake build in every `make` and in CI's CPU leg;
  - a cargo feature `speech`, off in a bare `cargo build`;
  - a second upstream in ADR 0010's manifest.
  The spike showed the kernels it was wanted for can be ours, at parity.
- **Adapt only ggml's dot products, in C** (Q60's fallback): adapted ops
  under ADR 0051, each a function of raw pointers and block structs. Not
  needed: the Rust kernels met every criterion with no C.
- **GPU only, no CPU executor.** Rejected: the loan takes only free KV pages
  and falls back to the CPU when they are not enough (Q30), and on a 16 GB
  card the CPU is the only way (owner, 2026-10-23).
