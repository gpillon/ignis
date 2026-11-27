---
status: accepted
---

# The speech model's CPU executor is ours, in Rust

The speech model's CPU executor, the baseline ADR 0067 asks for, is our own Rust in the default build: its kernels,
both graphs (the AuT audio encoder and the Qwen3 decoder), its GGUF reader and its thread pool are written here, ADR
0051's first tier, and nothing of ggml or llama.cpp is in the tree or the build. The owner refuses both large
foreign code in the tree and a platform shape dictated by someone else's code. The CPU measurement
([finding](../findings/2026-10-24-qwen3-asr-on-cpu.md)) shows that quantized SIMD kernels are what make the CPU go,
not that they must be ggml's, and a spike against ggml on the same host, its criteria fixed before it ran, passed
every one with no C ([finding](../findings/2026-10-24-rust-quantized-matmul-matches-ggml.md)).

- **Pure Rust in the default build.** The executor's own code has no C, no C++, no CMake, no build script and no
  cargo feature, and it adds no `cc` or `cmake` build dependency the workspace does not already have. A bare
  `cargo build` carries the speech model on every target ignis builds, and ADR 0032's default build, pure Rust,
  needs no new toolchain.
- **Kernels are picked at run time**, a scalar reference everywhere and AVX2/FMA/F16C kernels on x86-64, with no
  global `target-cpu`: one binary serves every x86-64 CPU, and no AVX2 instruction runs outside a function selected
  after detection, so no CPUID check has to guard against an illegal instruction. AVX2 stays the minimum, with FMA
  and F16C beside it: on a CPU without the three the CPU executor is unavailable, with ADR 0067's reason.
- **Q4_K is repacked at load into our interleaved layout**, the only copy of those tensors kept. ADR 0067's load at
  first use is this repack.
- **Our own thread pool**, spin then park, at `speech.threads`: rayon failed the spike's decode criterion on 4
  cores.
- **Provenance: ours.** The GGUF block layouts and the Q8_K/Q8_0 quantization rules are the file format, data the
  kernels read, not copied code. Code copied later from anywhere follows ADR 0051's adapted-op rules: the VAD is
  one, Silero VAD's graph (MIT) ported with its weights shipped. The executor itself stays ours.
- **The oracle is the checkpoint's own modelling code**, teacher-forced, as in ADR 0043. llama.cpp is an
  out-of-tree comparison and benchmark baseline only, and the offline tool that converts the GGUF files. It is never
  in the tree or in the build.
- **The forward pass runs in Rust, not in the kernel leaf**: a departure from ADR 0009 for the speech model on the
  CPU only. ADR 0009 puts the forward pass in the leaf so that activations and sequence state stay resident on the
  device; a CPU executor has no device. The GPU loan's forward pass is in the leaf.
- **One pipeline drives two compute adapters**: this executor, and the CUDA loan (ADR 0067), which builds on our
  kernel leaf alone, with no ggml-cuda base. Both read the same GGUF files.
- **Two dependencies are not copied code.** `symphonia` (MPL-2.0) is an unmodified crate dependency for audio
  formats, recorded in `NOTICE`: its copyleft is per file and binds only its own files, which we do not change. A
  bundled libopus (BSD-3) would build through the `cmake` crate, a new build dependency and the only CMake in a
  build without `cuda`, whose C goes through the `cc` crate: the owner accepts it only if it does not complicate the
  build or cost too much, so it comes in only once a verification build passes on both targets.

## Considered Options

- **Vendor ggml** (its core and x86 CPU backend, about 70,000 lines of C and C++), the first choice. Rejected by
  the owner's review: that much foreign code in the tree; an x86-only cut, made by us to keep that count down; a
  Haswell baseline that needed a CPUID check before the first call, or the process died on an older CPU; a CMake
  build in every `make` and in CI's CPU leg; a cargo feature `speech`, off in a bare `cargo build`; and a second
  upstream in ADR 0010's manifest. The spike showed the kernels it was wanted for can be ours, at parity.
- **Adapt only ggml's dot products, in C**, as adapted ops under ADR 0051: the fallback, C only if truly necessary.
  Not needed: the Rust kernels met every criterion with no C.
- **GPU only, no CPU executor.** Rejected: the loan takes only free KV pages and falls back to the CPU when they
  are not enough, and on a 16 GB card the CPU is the only way.

## Consequences

- No foreign code in the tree for the executor, the VAD's port aside: no manifest entry, no `NOTICE` line for ggml,
  no re-vendor to upgrade. CI's CPU test leg runs the executor's tests in the default build, with no toolchain added
  to it, unless libopus joins.
- The kernels, the layout, the pool and both graphs are ours to maintain: ggml's later speed-ups are ours to write
  if we want them.
- Parity on CPUs newer than the one measured (AVX-VNNI, AVX-512) needs kernels of ours that were neither written
  nor measured. Nothing in the design is x86-only: NEON kernels take their place beside the AVX2 ones by the same
  runtime selection.
- The spike timed the matmuls alone, on one microarchitecture: it does not show that the whole executor holds its
  speed.
