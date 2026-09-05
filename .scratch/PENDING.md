# Pending / revisit ledger (ignis)

Cross-cutting items that are intentionally deferred or blocked on an external
dependency. Per-ticket details live in `.scratch/<feature>/specs/`.

## Open

- **bench-03: gate-run — the v1 99% gate end-to-end (GitHub #24, spec 03,
  ADR 0005/0006/0007).**
  PARKED 2026-09-05 (operational — human prerequisites, ADR 0006/0007): the
  gate-run (capture → replay×2 → canary → gate → dogfood) requires ADR 0006
  GPU exclusivity (stopping the user's ninfer model runner — the coding
  agent's own LLM backend — a destructive shared-state operation), a live
  reference (ninfer) stack (`F:\ai\q38`), `ignis-server` + the reference
  running *sequentially* (never both — the 5090 cannot hold two full
  engines), a real "1 main + ~10 subagents" live agent session, and the 99%
  performance-gate verdict (ADR 0007). The **autonomous coding piece is now
  in place**: the `ignis-bench record` capture tool (bench-04, GitHub #34 —
  the capture-proxy subcommand that records a live agent session into a valid
  bench trace, CPU-testable against a mock target per spec 03) has landed on
  `main` as `fdba7ab` and is verified (workspace-wide `cargo test` green).
  What remains is **only the operational run itself** — a human/external
  prerequisite (ADR 0006 / ADR 0007). Owner: bench actor. Blocker: human
  operational run (GPU exclusivity + reference stack + live agent session).

- **bench-02: recorded reference baseline + 99% gate (ADR 0007, GitHub #20).**
  The `ignis-bench` harness is code-complete — `HttpEndpoint` transport,
  per-class metrics, canary self-consistency, 99% gate check (bench-01,
  968a2c1), plus the shipped v1 gate artifact (this bench-02 work:
  `ignis-bench gate` composes the performance report + the divergence
  report into a single shippable JSON artifact; `canary --out` ships the
  divergence report; `CanaryResult` is serializable with the sanity
  reason; the CLI flag parsing is fixed). What remains is the **recorded**
  side: a trace (JSONL) recorded against the reference stack + a reference
  run recorded with the *same harness* — the synthetic
  `main_plus_10.jsonl` fixture is not a reference. Procedure + file
  layout: `bench/traces/README.md`. Then `ignis-bench gate` runs the 99%
  gate. Owner: bench actor. Blocker: GPU + a recorded reference recording.
  PARKED 2026-09-05: this v1 99% gate (ADR 0007) is the acceptance
  milestone achieved by the bench-03 gate-run (GitHub #24) — a
  human/operational prerequisite (GPU exclusivity, reference stack, live
  agent session). The harness is code-complete; what remains is the
  *recorded* side (a real reference trace + a reference run with the same
  harness), which is the gate-run (see the bench-03 entry above).

## Blocked (external)

- **GPU availability (ADR 0006).** All GPU-gated items above require the
  RTX 5090 to be free. Last freed 2026-09-03 (GPU verified free, artifact-01
  GPU test run). Re-check before scheduling GPU work.

## Resolved (pruned weekly)

- **#32: B2 CUDA-graph decode replay (GitHub #32, spec 09, ADR 0008).**
  Resolved 2026-09-05: the decode graph is now the decode hot path (ADR
  0008) — at construction (`CudaCompute::new` + `from_artifact`) the
  representative decode sequence (embed → GQA → GDN → final RMSNorm →
  lm_head GEMV) is captured over persistent fixed-address device staging
  buffers, and each decode step H2Ds the token, replays via
  `ignis_graph_launch`, and D2Hs the logits (bit-identical to the eager
  reference, ADR 0007 self-consistency). A decode step whose batch does not
  match the captured `GraphGeometry` (batch 1) runs the eager sequence; a
  busy/absent GPU (or a VRAM shortfall) leaves the graph `None` (the eager
  fallback, ADR 0003/0006). New: `kernel/src/decode_graph_surface.cu` (the
  decode-graph leaf), the C-ABI `ignis_decode_graph_*` surface (the
  `kernel/include/ignis_kernel.h` + `crates/core/src/ffi.rs` mirror), the
  `CudaCompute` `build_decode_graph` + `graph_logits_{replay,eager}` +
  `uses_graph()` / `graph_launch_count()` observation surfaces + the Drop
  cleanup, and `crates/core/tests/decode_graph_gpu.rs` (the GPU-gated
  acceptance tests — replay==eager bit-exact, captured-at-construction,
  hot-path + eager-fallback; self-skip on a busy GPU, ADR 0006). Landed on
  `main` as `7c5e893` (fast-forward integration); `cargo test --workspace`
  green after rebuilding the kernel `.lib` to pick up the new `.cu` (the
  stale pre-#32 `.lib` was a transient LNK2019 on `ignis_decode_graph_*`).
  Pushed to `origin/main` 2026-09-05 (the user authorized the push;
  `7c5e893`, part of `575f5d5..d724c36`).

- **#34: bench-04 `ignis-bench record` — the capture-proxy harness piece
  (GitHub #34, spec 03, ADR 0005/0006/0007).**
  Resolved 2026-09-05: the `ignis-bench record` subcommand (a transparent
  OpenAI capture proxy in `crates/bench`) now ships — it accepts
  chat-completions from a live agent client, records each request as a
  `TraceLine` (`id`, `class` main|sub, `t_arrive_ms`, `prompt`, `max_tokens`,
  `stream`), forwards it byte-for-byte to `--target`, and finalizes on
  `POST /v1/session/end`. Class policy: `first-is-main` (default) or
  `marker`; a second `main` is demoted to `sub` (the `replay` driver rejects
  >1 main — the "1 main + N subagents" load shape). The recorded file is
  valid bench-trace JSONL — `Trace::from_jsonl` (the shape `replay` consumes)
  round-trips it (asserted in `record_capture.rs`: "the recorded trace loads
  and replays"). CPU-testable: `record_capture.rs` records a live session
  against a mock target (no GPU, no live engine). New: `crates/bench/src/
  record.rs` (the capture-proxy module), `crates/bench/tests/record_capture.rs`
  (the mock-target integration tests), the `record` CLI subcommand in
  `crates/bench/src/main.rs`. Landed on `main` as `fdba7ab` (fast-forward
  integration); `cargo test --workspace` green. Pushed to `origin/main`
  2026-09-05 (the user authorized the push; `fdba7ab`, part of
  `575f5d5..d724c36`). This is the harness piece of the bench-03 gate-run
  (GitHub #24); the operational run itself remains #24, PARKed for human
  execution.

- **#30: A3 full-correct Qwen 3.8-27B forward assembly (GitHub #30, spec 07,
  ADR 0005/0006/0007).**
  Resolved 2026-09-04: the compute-adapter's forward pass now runs the
  *full* layer stack on the real model — the GDN layers' causal conv
  (`ignis_gdn_causal_conv`) + the a / b (gate / beta) projection + the GDN
  step + the state readout (the "for now" host-side `S^T k` GEMV, ADR 0005)
  + the z (output-gate) gating; the GQA layers' QKV projection + the q / k
  RMSNorm (the per-head) + the RoPE (`ignis_rope_qk`, kernel-abi 06) + the
  GQA attention + the output projection; the gated-FFN block; the bf16
  logits GEMM (`ignis_bf16_gemm`, kernel-abi 10, A2b) for the W8-dequantized
  lm_head; and the real `qwen38_27b` topology (the 16 GQA + 48 GDN layers,
  the GDN feature layout — the q / z / a-b widths — the rotary geometry,
  θ = 1e7). The `from_artifact` path routes the real normalized tensors
  (A1 / #27): the NVFP4 fused planes stay device-resident (the `*_device`
  kernels, the #26 fix — the q / k / v slots are row slices of the fused
  `attention/query_key_gate_value` / `mlp/gate_up` planes), the BF16
  tensors are host-copied (the early GQA layers' qkv / output, the layer-4
  `gdn/output` quirk, the `gdn/convolution` + `gdn/a_b_projection` + the
  norms), the W8 endpoints are the A1 host-side dequants (the embedding +
  the lm_head). The decode step threads the actually-generated token (the
  autoregressive decode). New tests: `full_stack_synthetic_composition`
  (CPU — the layer composition + the geometry, spec 07 acceptance 4) +
  `real_model_forward_reproducible` (GPU-gated e2e — the real-model forward
  is in-vocab + reproducible across two fresh backends, spec 07 acceptance
  1; the `IGNIS_ARTIFACT` + `-- --ignored` + `--features cuda` gate,
  self-skip on a busy / OOM GPU, ADR 0006). The server / bench-03 "the
  server runs a real model" acceptance (spec 07 criterion 2) is the gate-
  run flow — the A3 forward pass is its prerequisite (the 99% gate itself
  is #20; the performance material is B1 / #31 + B2 / #32).
  `cargo test --workspace` green.

  **A3 v1 "for now" fidelity notes (ADR 0005 — the ported kernels are the
  "for now" starting point; the floor is a *sane*, reproducible output, not
  a reference match):** the GQA output-gate rows of the fused
  `attention/query_key_gate_value` (the 8 192..14 336 rows) are projected
  but unused (v1 applies no output gate to the attention output); the GDN
  a / b (gate / beta) is reduced to the step's scalar g / beta (the first
  a / first b of the 96-row `gdn/a_b_projection`); the GDN state readout
  is the host-side `S^T k` GEMV (the ported `ignis_gdn_step`'s contract
  updates the state, it does not emit a readout); the NVFP4 `*_device`
  GEMMs read the artifact's fused planes as plain row-major `[m][k/2]` /
  `[m][k/16]` (the container's block-scale plane layout + the
  `weight_divisor` application are the later re-implementation material,
  #20). The fused `mlp/gate_up` plane is read gate-first (rows 0..17 408
  gate, 17 408..34 816 up — the reference's fused gate_up convention).

- **#26: compute-adapter crash fix + VRAM materialization + device-GEMM
  surface (GitHub #26, ADR 0002/0006).**
  Resolved 2026-09-03: `CudaCompute::from_artifact` no longer falls back to
  the synthetic topology (vocab 256) — it uses the real `qwen38_27b()`
  topology, so real-tokenizer ids (up to 248077) never index out of bounds in
  `ignis_embedding` (the `illegal memory access`), and the 19 GB of weights
  materialize to VRAM via `CudaDevice` (`vram_resident()`). Host-side
  `Weights` are a zero-cost `placeholder` (the real weights live in the VRAM
  arena, ADR 0002); the device-resident GEMM surface
  (`ignis_nvfp4_gemm_{decode,prefill}_device`) is compiled into the kernel
  leaf `.lib` (a prerequisite for the broader compute-adapter, kernel-abi
  04/05).
  **The #26 "hang" was a CPU OOM trap, not a GPU deadlock:** the prior build
  ran `Weights::synthetic` at the real topology (~1.6 TiB of generated host
  vectors in a debug build), so the CPU spun for minutes/hours *after* the
  19 GB H2D while the GPU sat at 0 % — which read like a stuck
  `cudaStreamSynchronize`. `Weights::placeholder` (zero-cost) fixes it; the
  E2E (`real_model_e2e`) now passes in ~9 s.
  The numerically-correct real completion (the actual forward pass, the
  CUDA-graph fast path, and the server serving real completions) is the #25
  split — A3 (#30, the forward pass), B1 (#31, batched prefill), B2 (#32, the
  graph replay) — dependency `A1 ∥ A2 ∥ A2b → A3 → (B1 ∥ B2) → #20`.

- **kernel-abi-03: CUDA-graph eager capture at startup (GitHub #10, ADR 0006/0007)** —
  resolved 2026-09-03: `kernel/src/graph_capture.cu` implements the four
  `ignis_graph_*` primitives + `ignis_graph_startup_check` (captures a
  representative GQA-prefill + GDN-step + GQA-decode kernel sequence into a
  CUDA graph, replays it, and verifies replay ≡ eager bit-exactly). The
  kernels are forward-declared in the .cu (defined in the sibling surface .cu
  files) to avoid LNK4006 duplicate definitions. `kernel_abi03_gpu`
  (`graph_primitives_roundtrip_gpu` + `graph_startup_check_gpu` + the CPU
  null-handle pin) passed on 2026-09-03; the startup check confirmed on the
  GPU that replay ≡ eager. The 99% performance gate (ADR 0007) remains
  pending under bench-02 (GitHub #20, a recorded reference baseline is
  required).

- **artifact-01: `CudaDevice` real-artifact VRAM materialization (GitHub #4, ADR 0006)** —
  resolved 2026-09-03: `real_nvfp4full_cuda_device` passed on a free RTX 5090
  (9.42 s, 1,319 tensors H2D, ~19 GB VRAM). Build fix: `CMAKE_MSVC_RUNTIME_LIBRARY`
  set to `MultiThreaded` in `kernel/build.ps1` to force `/MT` static CRT (Rust
  MSVC target requires it; CMake default `/MD` caused LNK2038).

- **server-03: checksum wiring into the artifact loader (GitHub #21)** —
  resolved 2026-09-02 (`loader` module in `crates/server`, verified load
  path, descriptive refusal on a non-clean report or missing sidecar).