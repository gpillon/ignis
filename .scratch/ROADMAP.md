# ignis roadmap — from "no working forward" to "≥ ninfer" (tracked)

Source: `REVIEW-2026-09-05.md` §6. Status and blocking live on GitHub
(AGENTS.md); this file is the map: phases, gates, master tickets, and the
ticket decomposition. Update the ticket numbers here when they are published;
never track status here.

Legend: **G1..G5** = GPU-measured gates. A phase's master ticket is closed only
when its gate is recorded green on a free RTX 5090 (ADR 0006).

| Phase | Gate | Master ticket | Spec |
|---|---|---|---|
| 1 — device-resident correct forward, batch 1, bf16 KV | G1: coherent greedy canary completions; per-layer f64 reference within bf16 tolerance; ≥95% first-32-token agreement with the reference engine; EOS; reproducible | #36 | `runtime/specs/01-device-resident-forward.md` |
| 2 — real prefill (chunked, W4A4, tensor-core attention, GDN chunked) | G2: TTFT @8K/32K ≤ 1.5× reference MTP0 | #63 | to write when G1 lands |
| 3 — serving loop: batched decode rounds, per-width CUDA graphs, sampling, request log | G3: C=1 MTP0 decode ≥ 99% of reference (75–76 tok/s); C=4 aggregate ≥ 99% | #64 | to write when G2 lands |
| 4 — reference feature floor: hq-e8-2b KV, device prefix reuse, KV-RAM tier, tagged lanes | G4: bench-03 99% gate on the recorded "1 main + N subagents" trace | #65 (absorbs #20/#24) | to write when G3 lands |
| 5 — speculative decoding: MTP + ReplaySSM, then DFlash2 | G5: ≥ 99% of reference MTP7-adaptive / DFlash2-7 committed tok/s @24K/98K/196K | #66 | to write when G4 lands |
| 6 — beyond the reference (north star) | per-item gates | — | concurrent prefill / prefill-decode overlap, PDL + fusion, lazy graphs, hot reload, own artifact recipe |

## Phase 1 decomposition (G1, master #36)

Tracer bullets: each ticket ends in something verifiable on the GPU (or on
CPU for docs/tooling), sized for one agent iteration. "Vendor" = copy from the
pinned reference commit via the manifest script (spec: kernel policy).

| # | Ticket | Blocked by | Delivers (verifiable) |
|---|---|---|---|
| P1-01 (#37) | Docs reset: ADR 0009 (step ABI), ADR 0010 (vendoring), revise ADR 0001, README / CONTEXT / design / PENDING status | — | Design record matches reality; no "code-complete" claims |
| P1-02 (#38) | GPU gate profile: explicit GPU test profile that fails (never skips) when the GPU is busy or a kernel errors; preflight script; runbook (stop ninfer → run → restart) | — | `cargo test` stays CPU-only; the GPU profile refuses to run while ninfer holds the GPU and fails on errors |
| P1-03 (#39) | Contract: delete the superseded forward, toy graphs, host-pointer surfaces, scalar kernels, their GPU tests, the parked worktree | — | Workspace green on CPU with the mock; the artifact crate's VRAM materialization still works |
| P1-04 (#40) | Canary oracle tooling: canary prompt set; recorder that queries the reference engine (exact argmax) over HTTP, tokenizes with the artifact tokenizer, writes the fixture; comparer (agreement %, first divergence); CPU tests on a mock | — | Fixture format + tools tested; recording itself is P1-05 |
| P1-05 (#41) | Record the canary oracle fixture against the reference engine (human: needs ninfer running) | P1-04 (#40) | Committed fixture: 3 prompts × 32 greedy tokens + text |
| P1-06 (#42) | Vendor substrate: manifest + copy/verify script; reference core (dtype, tensor, arena, device, layout, weight, PDL, nvtx) + ops common; leaf CMake builds it; kernel test executable (CTest) with the reference op-test harness and one GPU smoke test; Rust workspace still links | — | `ctest` runs a GPU smoke test through the vendored arena/tensor; `cargo build --features cuda` links |
| P1-07 (#43) | Vendor norms + glue: rmsnorm, gated_rmsnorm, l2norm, residual_add, silu_mul, sigmoid_mul + their reference tests | P1-06 (#42) | Op tests green at 27B widths |
| P1-08 (#44) | Vendor embedding (W8G32 gather) + argmax + their tests | P1-06 (#42) | Op tests green at vocab 248320 / hidden 5120 |
| P1-09 (#45) | Vendor NVFP4 linear: codec, format, config, dispatch, GEMV, small-T (W4A4/TMA compiled, untested until G2) + linear tests (a16 profile) | P1-06 (#42) | NVFP4 GEMV test green at [34816,5120], [5120,17408], [16384,5120] with the blockscale layout + divisor |
| P1-10 (#46) | Vendor BF16 + W8G32 linear (GEMV, small-T, MMA) + tests | P1-06 (#42) | BF16 GEMV [14336,5120]; W8 GEMV at the output head [248320,5120] green |
| P1-11 (#47) | Vendor fused attention projections: attn_input_proj (NVFP4 + BF16 arms), linear_add (out-proj + residual) + tests | P1-09, P1-10 | Fused q/k/gate/v split matches the reference row order |
| P1-12 (#48) | Vendor fused GDN projections: gdn_input_proj (qkv + z, NVFP4 + BF16 arms), gdn_gating_proj (a/b) + tests | P1-09, P1-10 | Tests green at [16384,5120] / [96,5120] |
| P1-13 (#49) | Vendor SwiGLU MLP: linear_swiglu (NVFP4 gate_up + silu·mul) + tests | P1-09 (#45) | Test green at [34816,5120] |
| P1-14 (#50) | Vendor GDN family: causal_conv1d_silu, gdn_gating, gated_delta_net (recurrent; chunked kernels compiled + tested) + tests | P1-06, P1-07 | Recurrent step test green at 48 heads × 128 fp32; chunked test green |
| P1-15 (#51) | Vendor GQA attention family: position, rope, qk_norm_rope, paged addressing, gqa_attention (bf16 decode + prefill routes; i8/hq vendored, untested), kv append + tests | P1-06, P1-07 | bf16 decode + prefill attention tests green at 24q/4kv×256 |
| P1-16 (#52) | Vendor sequence-state pools: paged KV pool, linear-attention state pool (+ ring bits) + their tests | P1-06 (#42) | Pool tests green; page geometry matches the reference |
| P1-17 (#53) | Model-load ABI: bound-tensor descriptor (qtype, layout, planes, shapes, divisors) + topology descriptor; artifact crate exports device views for every text-scope tensor (W8 endpoints as device planes, host dequant removed); leaf builds per-layer weights and rejects a missing/mis-shaped object | P1-06 (#42) | GPU test: real artifact → model handle; every expected object bound; VRAM reported |
| P1-18 (#54) | Degenerate program: embedding → final norm → output head → argmax through the step ABI (layers skipped) | P1-07, P1-08, P1-10, P1-17 | Logits for one token match a Rust f64 reference from the artifact |
| P1-19 (#55) | Sequence handle ABI: alloc / release with KV page reservation, GDN slot, conv taps, position; runtime reports page geometry; the core's KV pool sized from it | P1-16, P1-17 | Alloc/exhaust/release/re-alloc tests; zero state on re-alloc |
| P1-20 (#56) | f64 layer references (Rust, CPU): one GQA layer and one GDN layer on real weights for ≤4 tokens, from the artifact's host decoders | P1-17 (#53) | Reference values + CPU test; used by P1-21/22 |
| P1-21 (#57) | GQA layer in the program: input norm → attn_input_proj → qk norm + RoPE → KV append → attention → output gate → out-proj + residual → post norm → SwiGLU → down + residual (T=1) | P1-11, P1-13, P1-15, P1-18, P1-19, P1-20 | Layer output within bf16 tolerance of the f64 reference |
| P1-22 (#58) | GDN layer in the program: input norm → gdn_input_proj → causal conv → gating proj + gating → recurrence → gated norm → out-proj + residual → MLP tail (T=1) | P1-12, P1-13, P1-14, P1-18, P1-19, P1-20 | Layer output within bf16 tolerance of the f64 reference |
| P1-23 (#59) | Full program + prefill/decode ABI: 64 layers, per-token prefill over a span, decode round (batch 1), EOS from artifact defaults, stats | P1-21, P1-22 | Canary prompt → coherent greedy text; reproducible across loads |
| P1-24 (#60) | Rust runtime crate: safe wrapper (model/sequence handles, Drop, error mapping), Compute-trait adapter, EOS / max_tokens stop; mock stays | P1-19 (#55) | CPU tests against a stub leaf; scheduler drives the adapter |
| P1-25 (#61) | Server e2e on the real model: streaming + non-streaming chat completions with `finish_reason: stop`; bench canary against it | P1-23, P1-24 | GPU e2e green |
| P1-26 (#62) | G1 gate run: canary agreement ≥ 95% vs the P1-05 fixture, f64 layer checks, reproducibility; record the verdict in the review; close #36 | P1-05, P1-25 | Gate recorded |

Frontier at start: #37, #38, #39, #40, #42 (five parallel starts).
Critical path: #42 → #45/#46 → #47/#48/#49 → #57/#58 → #59 → #61 → #62.

## Phase 2–5 candidate decomposition (not published; refined when the gate before lands)

- **G2**: W4A4 activation quant + TMA GEMM route on (T ≥ 64) · prefill attention bf16 route in the program · GDN chunked prefill route · chunked span prefill (1024) with KV append · TTFT bench cell · G2 gate.
- **G3**: batched decode round (B ≤ 8) over per-slot views · sampling (temp/top-p/top-k/penalties, seed) · decode graph capture per width + eager fallback · PDL chain where vendored ops support it · request-log JSONL · core KV pool ↔ runtime pages under load · G3 gate (C=1, C=4).
- **G4**: hq-e8-2b codec + attention routes + exact-key side store · device prefix reuse (page refcount, shared system+tools boundary) · KV-RAM tier snapshot/restore (all state sections) · tagged lanes · preserve-thinking / tool-call stream hardening · warmup/readiness · G4 gate (bench-03 trace).
- **G5**: MTP round + pack + adaptive width + ReplaySSM records/fold · DFlash2 drafter load + draft kernels + RAM-tier carry · G5 gate.
