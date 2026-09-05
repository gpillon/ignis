# runtime 01 — device-resident forward pass at the step ABI (gate G1)

GitHub: #36 (supersedes kernel-abi specs 01–10 + ADR 0008; replaces #35)

Supersedes: kernel-abi specs 01–10 (the per-op host-pointer surface, the
compute adapter's forward, the batched-prefill pass, the staging-buffer decode
graph) and ADR 0008. Source: `.scratch/REVIEW-2026-09-05.md` §2, §3, §5, §6
(Phase 0 + Phase 1). ADRs respected: 0002 (the artifact is loaded directly),
0005 (performance-first above a sane-output floor), 0006 (exclusive GPU
testing), 0007 (the 99% gate is a performance gate, correctness is
self-checked). ADRs introduced by this work: 0009 (step-level device-resident
C ABI, revising 0001) and 0010 (kernel policy: verbatim ports with provenance,
clarifying 0005).

## Problem Statement

ignis has never produced a real completion. The engine's forward pass keeps
activations, the paged KV cache and the GDN state in host memory and crosses
the C ABI once per operator with synchronous copies, so even a bug-free
version of it would run seconds per token against the reference's 13 ms. On
top of that the forward computes the wrong function: the NVFP4 scale plane is
read in the wrong storage layout, the per-tensor weight divisor is never
applied, the GDN gating ignores `a_log` / `dt_bias` and uses one scalar for all
48 value heads, the GDN state is one flat bf16 matrix instead of 48 fp32
per-head matrices, RoPE positions advance per layer instead of per token, the
GDN gated norm and the GQA output gate are missing, decode never stops on EOS,
and the "decode hot path" CUDA graph replays a five-kernel toy over zeroed
state. Every kernel behind the ABI is a scalar stand-in, not the proven
reference kernel its header claims. The GPU tests that would have caught all
of this are ignored by default and self-skip on any kernel error, so
`cargo test` is green while nothing works.

The owner needs a forward pass that is correct, device-resident, and shaped so
the next milestones (chunked prefill, batched decode rounds captured in CUDA
graphs, hq-e8-2b KV, prefix reuse, speculative decoding) are additions rather
than rewrites.

## Solution

Move the forward pass into the kernel leaf as a device-resident **program**
(streams, a device arena, per-sequence state, the per-layer op sequence) and
expose it to Rust through a **step-level C ABI**: load a model from the
artifact's device arena, allocate / release / snapshot a sequence, prefill a
token span, run one decode round over a batch of sequences, sample. Rust keeps
everything above the step: HTTP, tokenizer and chat template, the scheduler
and admission state machine, KV page accounting, telemetry, the bench.

Every operator in the program is a **verbatim port** of the corresponding
reference (ninfer) kernel and its launch helper, including the storage-layout
decoders, with per-file provenance — never a scalar re-implementation. Each
ported op brings its reference op test with it into a kernel-leaf test
executable.

Correctness is established by an **oracle**: the reference's per-layer
activation dumps on the same artifact and the same canary prompts. The gate
(G1) is a coherent greedy completion on the canary suite with per-layer
activations within bf16 tolerance of the oracle. GPU tests fail loudly when
the GPU is unavailable; a skip is never green for compute work.

The old forward, the toy graphs and the host-pointer kernel surfaces are
deleted, and the superseded specs / ADR are marked as such so no future
reader builds on them.

## User Stories

1. As the engine owner, I want `ignis-server` with `IGNIS_ARTIFACT` set to return a coherent greedy completion for a chat prompt, so that the engine is finally a testable base.
2. As the engine owner, I want the completion to stop at the model's EOS token (or `max_tokens`), so that responses end where the model ends them.
3. As the engine owner, I want the same prompt with greedy decoding to produce the same tokens on every run, so that the self-consistency check of ADR 0007 is meaningful.
4. As the engine owner, I want per-layer activations of a prefill to match the reference's activation dump within bf16 tolerance, so that "sane output" is measured, not eyeballed.
5. As the engine owner, I want the first 32 greedy tokens on each canary prompt to agree with the reference on at least 95% of positions, so that the correctness floor has a number.
6. As the engine owner, I want activations, KV cache and GDN state to live in VRAM for the lifetime of a sequence, so that no per-op host round trip exists on the hot path.
7. As the engine owner, I want the NVFP4 GEMMs to decode the artifact's `blockscale-k16-m128x4-v1` scale plane and apply the weight divisor, so that every quantized projection multiplies by the right coefficient.
8. As the engine owner, I want the GDN recurrence to run per value head on fp32 128×128 state with the reference's gating (`a_log`, `dt_bias`, softplus, sigmoid), L2-normalized q/k and the 1/√128 readout, so that the linear-attention layers compute the model's recurrence.
9. As the engine owner, I want the GDN gated RMSNorm (`gdn/norm`) and the GQA output gate applied, so that the layer outputs match the model.
10. As the engine owner, I want RoPE to rotate every layer at the token's position, so that attention over the prompt is not scrambled.
11. As the engine owner, I want the W8G32 embedding and output head consumed by device kernels, so that no host-side dequant is needed at load and VRAM is not spent on a bf16 copy.
12. As the engine owner, I want the kernel leaf to have its own test executable that runs each ported op's reference test, so that op correctness is checked without Rust and at real geometry.
13. As the engine owner, I want every ported kernel file to name its source file and revision in the reference, so that a port claim is diffable.
14. As the engine owner, I want GPU tests to fail (not skip) when the GPU is busy or a kernel errors, so that a green run means the forward ran.
15. As the engine owner, I want the default `cargo test` to stay CPU-only and fast, with the GPU suite run explicitly, so that the ADR 0006 self-bootstrapping loop (ninfer holds the GPU while I code) still works.
16. As the engine owner, I want a documented runbook step to stop ninfer, run the GPU suite, and restart ninfer, so that exclusive GPU testing is a checklist, not tribal knowledge.
17. As the engine owner, I want the step ABI to take a batch of sequences for decode from day one, so that batched decode rounds (G3) are a leaf change, not an ABI change.
18. As the engine owner, I want a sequence handle to own its KV pages, GDN slot and conv taps, with snapshot/restore entry points, so that the KV-RAM host tier (G4) has a real object to snapshot.
19. As the engine owner, I want the prefill entry point to take a token span and a starting position, so that chunked prefill and prefix-reuse tail prefill (G2/G4) fit without ABI change.
20. As the engine owner, I want the scheduler's KV pool to count real device pages reported by the runtime, so that admission decisions reflect VRAM.
21. As the engine owner, I want the old forward pass, the toy decode graph and the host-pointer kernel surfaces removed, so that no agent or reviewer mistakes them for the engine.
22. As the engine owner, I want the superseded specs and ADR 0008 marked superseded, and ADR 0009 / 0010 written, so that the design record matches the code.
23. As the engine owner, I want README, CONTEXT.md, the design doc and PENDING to state the true status, so that "v1 code-complete" stops misleading readers.
24. As the engine owner, I want the canary prompts and the oracle dumps to be reproducible from a script, so that G1 can be re-run after every kernel change.
25. As a coding-agent user, I want streaming chat completions to deliver real tokens as they are generated, so that the engine is usable interactively once G1 holds.
26. As the bench actor, I want the runtime to expose per-step timing and kernel/graph counters, so that the harness can report phase rates once performance work starts.
27. As a future kernel author, I want the program layer to isolate op dispatch from the ABI, so that a ported op can later be replaced by our own kernel under the same oracle.
28. As the engine owner, I want the leaf to build with the same CMake + Ninja + nvcc flow and link into Cargo as before, so that the build story does not change.

## Implementation Decisions

**Architecture (ADR 0009).** The C ABI moves from the operator level to the
step level. Rust owns everything above the step; the kernel leaf owns the
step. The leaf gains a *program* layer: a device arena and tensor views,
CUDA streams, a per-layer op sequence for the Qwen 3.8-27B text model, a
sequence-state store (paged KV pages, a slot-indexed fp32 GDN state pool,
conv taps), and the sampling epilogue. ADR 0001 is revised accordingly; ADR
0008 is superseded (graphs will capture the batched decode round over
per-slot state views, not a staging-buffer toy; that is G3 work).

**Step ABI surface (flat C, opaque handles, device-resident, int return
codes).** Model load from the artifact's device arena plus a topology
descriptor; sequence allocate / release with a context reservation; sequence
snapshot to a caller-provided pinned host region and restore from it (entry
points defined now, exercised at G4); prefill of a token span starting at a
position for one sequence, producing the last position's logits or a sampled
id; one decode round over a batch of sequence handles and their current
tokens, producing one id per sequence; a sampling-parameters struct (greedy
this spec; temperature / top-p / top-k / penalties are G3); runtime statistics
(VRAM, per-step time, kernel count). No host activation pointers cross the
boundary. Streams and synchronization are internal to the leaf.

**Model program.** The per-layer sequence follows the reference's text
context: input norm → fused projection → (GQA: per-head q/k norm, RoPE at the
token position, attention over paged bf16 KV, output gate, output projection |
GDN: causal conv + SiLU on q/k/v, gating from the a/b projection with
`a_log`/`dt_bias`, per-head fp32 recurrence with L2-normalized q/k and the
1/√128 readout, gated RMSNorm with z, output projection) → residual →
post-attention norm → gate_up projection → SiLU·mul → down projection →
residual; final norm → output head → argmax. Layer kinds, fused-plane row
orders, head mappings and rotary geometry come from the reference's artifact
document, not from guesses. The fused projection's activation input-scale
divisors are read (the W4A4 path that needs them is G2).

**Kernel policy (ADR 0010) — vendoring.** Every op is taken verbatim from
the reference as a vendored subtree inside the kernel leaf: its kernel, its
launcher, its public wrapper and header, and the reference's own op test.
Provenance is a manifest (the reference's pinned commit, the list of vendored
files with content hashes, and any local patch recorded as a diff) maintained
by a copy-and-verify script — not a hand-written header per file. Vendored
files keep the reference's namespaces, include paths and the feature-major
tensor convention (a token axis as the second dimension), so a diff against
the source stays trivial and a later reference update is a re-run of the
script. The reference's `NOTICE` attribution is kept. The vendored substrate
is the reference's core (dtype, tensor, arena, device, layout, weight
descriptor, PDL helper) and the ops' common headers; on top of it the op
families needed here: NVFP4 / BF16 / W8G32 linear (GEMV and small-T; the
large-T W4A4 path compiles but is gated at G2), the fused projections
(attention input, GDN input, GDN gating, SwiGLU, linear+residual),
embedding, norms (RMSNorm, gated RMSNorm, L2), q/k norm + RoPE, GQA attention
over paged bf16 KV with KV append, causal conv1d + SiLU, GDN gating, the GDN
recurrence (chunked kernels vendored and tested, used at G2), argmax, and the
paged-KV pool + linear-attention state pool. Multi-token prefill in this
spec runs the recurrence and attention per token (what makes the oracle
comparable); the chunked path is G2.

**Program layer (ours).** The per-layer op sequence, the sequence-state
store wiring, the prefill / decode loops and the step ABI are written in the
leaf by us on top of the vendored ops, following the reference's text
context order; the reference's program is not vendored (it is entangled with
MTP, vision, graphs and its engine).

**Prefill in this spec.** A prompt is processed one token at a time through
the same device-resident program (no host activations). This is
deliberately slow and is not gated on time. It must leave the sequence state
(KV pages, GDN state, conv taps, position) exactly where the decode
continues.

**Sequence state.** Per sequence: a block table over device KV pages (bf16, K
and V planes, page geometry chosen to match the reference's paged KV so
hq-e8-2b can drop in later), one GDN slot (48 layers × 48 heads × 128 × 128
fp32) from a slot pool sized by the concurrency, conv taps, position, and the
last token. The scheduler's KV pool is fed the real page count and page bytes
reported by the runtime.

**Rust runtime crate.** A thin safe wrapper over the step ABI replaces the
compute adapter: model handle, sequence handle with Drop, prefill / decode /
sample calls, error mapping. The `Compute` trait implementation for the
scheduler delegates to it; `MockCompute` stays for CPU tests. EOS is read from
the artifact's generation defaults and enforced in the decode loop.

**Artifact crate.** Adds a device-view export: for each bound tensor, the
device pointer, storage layout, format and shape, so the program binds planes
directly (no host dequant of the W8 endpoints, no host copies of BF16
exception tensors beyond what the leaf allocates itself).

**Oracle (two levels).** Op level: each vendored op's reference test (fp64
CPU references and the reference's tolerances) at real 27B geometry. Layer
and program level: (a) a CPU f64 reference of one GQA layer and one GDN
layer on the real weights for a handful of tokens, computed in Rust from the
artifact (the artifact crate already decodes NVFP4 / W8 on the host), against
which the leaf's per-layer output is checked; (b) the reference engine's
greedy completions on the canary suite (recorded once through its HTTP API
with exact argmax, tokenized with the artifact's tokenizer, stored as a
fixture) against which the first 32 greedy tokens are compared. The
reference has no activation taps and its Python reference does not cover
this artifact on this box, so per-layer dumps are not available; the f64
layer reference replaces them.

**Test rule.** A GPU test profile that requires the GPU free and fails on
any kernel error; run explicitly (runbook: stop ninfer, run, restart). The
default `cargo test` remains CPU-only. The kernel leaf gets a test executable
(the reference's op-test style, one test per ported op) built by the existing
CMake flow.

**Deletions.** The compute adapter's forward pass and graph plumbing, the
toy decode graph and graph-capture surfaces, the smoke/vector-sum surface, and
all host-pointer kernel entry points and their scalar kernels. The parked
flat-ABI worktree is removed. Their tests go with them; f32 CPU references
survive only where a ported op test reuses them.

**Documentation.** ADR 0009 and ADR 0010 written; ADR 0008 marked
superseded; kernel-abi specs 01–10 carry a superseded banner pointing here;
README status, CONTEXT.md (GDN head structure, sequence state, step ABI,
oracle, GPU suite), the design doc milestones and PENDING rewritten to the
true status.

**Explicitly deferred but shaped for:** batched decode rounds and per-width
CUDA graphs (batch parameter present, per-slot state views), chunked prefill
and W4A4 GEMM (span + position ABI, divisors read), hq-e8-2b (page geometry),
KV-RAM tier (snapshot/restore entry points), sampling parameters (struct
present, greedy only).

## Testing Decisions

A good test drives the step ABI or the server and checks observable output
(tokens, activations, state after a step, error codes), never a kernel's
internal tiling or the program's dispatch order.

- **Op tests (kernel leaf executable, GPU).** One per ported op, ported with
  the op from the reference's op-test suite, at real 27B geometry, against the
  reference's own CPU/fp64 references and tolerances. Prior art: the
  reference's `tests/ops` layout and its uniform error-record convention.
- **Program tests (GPU, explicit profile).** Through the step ABI: a
  sequence prefilled with a canary prompt has per-layer residual streams and
  final logits within bf16 tolerance of the oracle dump; 32 greedy decode
  steps agree with the oracle on ≥ 95% of positions; two fresh model loads
  produce identical tokens; decode stops at EOS; a sequence released and
  re-allocated starts from zero state; a busy GPU or a kernel error fails the
  test. Prior art: the existing GPU-gated tests' structure (`IGNIS_ARTIFACT`
  gate) minus the self-skip.
- **Runtime crate tests (CPU).** Error mapping, handle lifetimes, EOS and
  `max_tokens` stop logic against a stub leaf; the scheduler's KV pool sized
  from runtime-reported pages. Prior art: the existing scheduler / host-tier
  CPU tests with `MockCompute`.
- **Server e2e (GPU, explicit profile).** `ignis-server` with the artifact
  answers `/v1/chat/completions` (streaming and not) with a coherent
  completion that ends with `finish_reason` `stop` on EOS. Prior art: the
  server's `openai_http` tests.
- **Artifact crate.** The device-view export covers every bound text-scope
  tensor with the right layout tag; the binder's "every object consumed"
  gate is unchanged. Prior art: `real_artifact` and the binding tests.
- **Build.** The leaf test executable and the Rust workspace build from the
  documented commands; `cargo test` (CPU) green; the GPU profile green on a
  free 5090 is the G1 gate and is recorded in the review.

## Out of Scope

- Chunked prefill, W4A4 activation quantization and the tensor-core prefill
  GEMM / attention / GDN chunked kernels (G2).
- Batched decode rounds, per-width CUDA graph capture, PDL, non-greedy
  sampling, request-log JSONL (G3).
- hq-e8-2b KV, device prefix reuse, the KV-RAM host tier's actual
  snapshot/restore, tagged lanes, thinking preservation (G4).
- MTP, DFlash2, ReplaySSM (G5). Vision.
- The 99% performance gate and any timing acceptance: G1 is a correctness
  floor only (ADR 0005/0007 ordering).
- Our own kernel implementations (ADR 0010 defers them to measured
  per-family work after G4).

## Further Notes

- The old specs are outdated by construction: they describe a per-op host
  ABI and a host-resident forward that this spec deletes. Do not read them
  for behavior; read the review and the reference's maintainer docs
  (artifact reference, storage layouts, model doc, paged KV, ReplaySSM).
- Reference numbers to keep in view even though G1 has no timing gate:
  15.85 GB of weights streamed per decode token (9.5 ms floor), 75–76 tok/s
  MTP0 at C=1, ~630 kernels per decode step in one graph.
- The ported kernels are Apache-2.0; attribution stays in `NOTICE`.
- Work splits naturally into agent tickets *per op family under the
  oracle* (GEMV family, attention, GDN, norms/glue, endpoints, program +
  ABI, runtime crate + server, docs/ADRs); each op ticket closes only with
  its GPU op test green at real geometry. Cutting those tickets is the first
  step of executing this spec.
