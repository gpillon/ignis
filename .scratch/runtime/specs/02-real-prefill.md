# runtime 02 — real prefill: chunked spans, W4A4, tensor-core attention, GDN chunked (gate G2)

GitHub: #63 (phase 2 master; blocked by #36, gate G1)

Source: `.scratch/REVIEW-2026-09-05.md` §6 (Phase 2), `.scratch/ROADMAP.md`
(the G2 row and the phase 2–5 candidate decomposition), and the grilling
session of 2026-09-07 that turned them into this spec. Builds directly on
`.scratch/runtime/specs/01-device-resident-forward.md` (G1).

ADRs respected: 0005 (performance-first above a sane-output floor), 0006
(exclusive GPU testing), 0007 (performance gates, correctness self-checked),
0010 (vendored reference kernels), 0014 (teacher-forced canary floor).
ADRs introduced by this work: **0015** (G2 is judged live/live against the
reference, on cold-prefix samples) and **0016** (extensible options structs at
the step ABI, amending ADR 0009's "no ABI change" claim). ADR 0009 is
amended, not superseded: the step ABI's granularity and its span+position /
batch shape are unchanged.

Reference: ninfer at the manifest's pinned commit
(`kernel/vendor/manifest.json`).

## Problem Statement

G1 made ignis compute the model correctly, but it prefills a prompt **one
token at a time**. Every prompt token walks all 64 layers alone: the NVFP4
projections run their GEMV / small-T routes at T=1, GQA attention runs its
small-T decode route, the GDN layers run the recurrent kernel, and the
program synchronizes the stream once per layer — 64 device round trips per
token. The vendored kernels that exist precisely for multi-token work — the
W4A4 MMA/TMA GEMM route, the tensor-core prompt attention route, the GDN
chunked kernels (`prepare_wy_wu`, `state_passing`, `output`) — are compiled
into the leaf and untested, because nothing in the program ever hands them
more than one token.

The result is an engine whose time to first token is dominated by a loop
that the reference does not run. The owner's real workload is agentic
coding: a main agent plus subagents, each opening with a large system +
tools + context prompt of several thousand tokens. TTFT is what that
workload feels first, and at the current shape it is orders of magnitude
away from the reference — an 8K prompt is 8,192 sequential single-token
traversals of a 27B model.

Two secondary problems block measuring any of this. The leaf's default
context reservation is 4,096 tokens per sequence, so a 32K prompt cannot
even be admitted. And `ignis-bench` measures TTFT only as a by-product of
trace replay: there is no way to ask "what is TTFT at exactly 8,192 prompt
tokens", against either engine.

## Solution

The leaf gains a **chunked prefill** path: `ignis_program_prefill` accepts a
token span of any length and processes it in **prefill chunks** (default
1,024 tokens, a multiple of 128, chosen at model load), running each chunk
as one multi-token traversal of the 64 layers. Inside a chunk the program
routes to the multi-token kernels that were vendored for exactly this:

- every NVFP4 projection runs under the **`AllowA4` compute policy**, so the
  vendored per-projection thresholds pick the W4A4 (MMA/TMA) route where
  they are meant to, and the A16 small-T route otherwise;
- GQA layers use the fused append+attend entry point, whose route resolver
  picks the tensor-core prompt route above the small-T width;
- GDN layers use the distinct-state recurrence entry point, which runs the
  chunked tensor-core kernels over whole 64-token chunks and a recurrent
  tail.

The per-token path is not deleted. It becomes a **per-token prefill route**,
selectable per call and used as a **self-oracle**: the same prompt prefilled
both ways must agree, which is what proves the chunk loop, the state carry
across chunk boundaries and the new kernel routes did not change the
function being computed. That check needs no reference engine and no GPU
time from the owner beyond one profile run.

Selecting the route (and the policy) needs one new parameter, so the prefill
entry point takes an **extensible options struct** (`NULL` = defaults, a
leading `size` field for forward compatibility). This is a deliberate,
recorded, one-time ABI extension (ADR 0016), not a silent signature change:
G3's sampling parameters and G4's snapshot controls add *fields*, not
parameters and not `_ex` entry points.

The gate is measured, not asserted. `ignis-bench` gains a **`ttft`
subcommand** that drives any OpenAI-compatible endpoint at a fixed prompt
length and reports the distribution, and a **G2 gate check** that compares
two such records. G2 is judged **live/live** (ADR 0015): ignis and the
reference measured in the same session, on the same machine, on the same
cells, with the ratio ignis/reference ≤ 1.5. Every sample is a **cold
prefix** — a prompt distinct from the first content token, with the engine's
own computed-prefill-token count asserted equal to the prompt length — so no
prefix cache on either side can serve a measurement of prefill work. A
reference record is committed as a fixture for regression and sanity, and
never decides the gate.

Correctness keeps G1's floor unchanged: teacher-forced canary agreement
≥ 95% (ADR 0014), the f64 layer references, determinism across loads. The
policy change touches numerics, so those checks run again on the new path,
and any drop is a ticket, never a waiver.

## User Stories

1. As the engine owner, I want an 8K-token prompt to reach its first token in a time comparable to the reference's, so that the engine is usable for the agentic coding workload it exists for.
2. As the engine owner, I want a 32K-token prompt to be admitted and prefilled at all, so that long-context sessions are measurable rather than rejected.
3. As the engine owner, I want the prefill of a prompt to run as multi-token chunks through the model, so that the GPU does one traversal per thousand tokens instead of one per token.
4. As the engine owner, I want the prefill chunk size to be a model-load option with a documented default of 1,024 tokens and a 128-token alignment rule, so that a chunk-size sweep is a flag change rather than a rebuild.
5. As the engine owner, I want to pass a whole prompt span to the leaf and have the leaf chunk it, so that the caller does not need to know the chunk size and the serving loop (G3) can still choose to hand it smaller spans.
6. As the engine owner, I want the NVFP4 projections to run under the reference's own activation-quantization policy on both prefill and decode, so that the engine uses the routes its vendored kernels were written and measured for.
7. As the engine owner, I want the W4A4 route to be chosen by the vendored per-projection token thresholds rather than by a threshold of our own, so that a route claim is a property of the vendored code and not of our guesswork.
8. As the engine owner, I want GQA prefill to go through the fused append-and-attend entry point, so that a chunk's keys and values are appended and attended in one pass over the paged cache.
9. As the engine owner, I want GQA attention above the small-T width to take the tensor-core prompt route, so that long-prompt attention is not a chunked small-T loop.
10. As the engine owner, I want GDN prefill to use the chunked recurrence kernels for whole chunks with a recurrent tail, so that linear-attention layers stop being a per-token loop on long prompts.
11. As the engine owner, I want a sequence's GDN state, conv taps, KV pages and position after a chunked prefill to be exactly what the per-token path would have left, so that decode continues from the same place regardless of how the prompt was warmed.
12. As the engine owner, I want prefilling a prompt as one span to produce the same result as prefilling it as two consecutive spans split at an arbitrary boundary, so that prefix reuse and tail prefill (G4) rest on a property that is already tested.
13. As the engine owner, I want the per-token prefill route retained and selectable per call, so that the chunked path has an oracle that needs no reference engine.
14. As the engine owner, I want the chunked and per-token routes to agree on a long prompt's next-token predictions, so that a chunk-boundary or state-carry bug is caught by a test rather than by a degraded completion.
15. As the engine owner, I want the teacher-forced canary floor re-measured on the chunked path, so that the activation-quantization policy change is proven not to have broken the forward pass.
16. As the engine owner, I want any correctness regression found at G2 filed as its own ticket, so that no gap is waived to make a gate pass.
17. As the engine owner, I want the prefill scratch sized once at model load from the vendored workspace queries for the configured chunk size, so that no allocation happens on the prefill path and a chunk size that does not fit fails at load with a clear error.
18. As the engine owner, I want the leaf to report the device memory it reserved for that scratch, so that VRAM accounting stays honest as the chunk size grows.
19. As the engine owner, I want the program to synchronize once per chunk rather than once per layer, so that a chunk is one pipelined unit of work instead of 64 stalls.
20. As the engine owner, I want a kernel error during a chunk attributed to that chunk and its sequence, so that a failure is still diagnosable without per-layer synchronization.
21. As the engine owner, I want a failed prefill to leave the sequence in a state the caller can recover from, so that the scheduler's existing retry path stays correct.
22. As the engine owner, I want the maximum per-sequence context to be configurable, so that a 32K-token cell can be admitted without editing a default in code.
23. As the engine owner, I want the scheduler's admission accounting to keep matching the pool the leaf actually built when that context grows, so that admission never promises capacity the GPU does not have.
24. As the engine owner, I want a `ttft` bench subcommand that measures time to first token at an exact prompt length against any OpenAI-compatible endpoint, so that ignis and the reference are measured by the same instrument.
25. As the engine owner, I want each TTFT sample to use its own prompt, distinct from its first content token, so that no prefix cache, host KV tier or checkpoint on either engine can serve a measurement of prefill work.
26. As the engine owner, I want a warmup sample that is itself distinct, so that the warmup cannot populate a reusable prefix for the measured samples.
27. As the engine owner, I want each sample to verify against the engine's own reported computed-prefill-token count that the whole prompt was actually computed, so that a cold-prefix claim is evidence rather than an assumption.
28. As the engine owner, I want a sample whose prefill was partly served from a cache to be void and to fail its cell, so that a contaminated run cannot quietly pass the gate.
29. As the engine owner, I want prompts generated deterministically to an exact post-template token count using the artifact's own tokenizer, so that a cell measures the length it claims.
30. As the engine owner, I want the cell to report the median of five samples after one warmup, so that a single scheduling hiccup does not decide a gate.
31. As the engine owner, I want the bench to record the cell's full sample list, engine build identity, artifact identity, date and profile alongside the median, so that a recorded run can be audited later.
32. As the engine owner, I want the G2 verdict computed from two records measured in the same session, so that the gate compares two engines rather than one engine and a memory.
33. As the engine owner, I want the reference measured in the profile I actually run it in, so that the gate compares against the bar I actually experience.
34. As the engine owner, I want the difference in KV format between the two engines at G2 recorded next to the verdict, so that the comparison is honest about what is not equal yet.
35. As the engine owner, I want a committed reference record used for regression and sanity only, so that a stale fixture can never decide a gate.
36. As the engine owner, I want the gate check to refuse a verdict when the two records were not produced in the same session, so that live/live is enforced by the tool rather than by discipline.
37. As the engine owner, I want the G2 threshold to be one number (ratio ≤ 1.5) applied per cell, so that the verdict is unambiguous.
38. As the engine owner, I want prefill throughput and per-chunk timing reported as diagnostics, so that a failing cell can be attributed without another run.
39. As the engine owner, I want the vendored op tests for the routes this phase turns on to be vendored and run, so that the W4A4, prompt-attention and chunked-GDN claims rest on the reference's own tests at real geometry.
40. As the engine owner, I want the prefill entry point's new options passed as an extensible struct with a size field, so that G3 and G4 add fields instead of breaking the ABI again.
41. As the engine owner, I want a null options pointer to mean the production defaults, so that the common call site stays as simple as it is today.
42. As the engine owner, I want the ABI extension recorded as a decision rather than described as a no-op, so that the design record does not claim an invariant the code broke.
43. As the engine owner, I want `--prefill-chunk` and the context limit exposed as server flags with the existing precedence rules, so that a bench run configures the engine without exporting environment variables.
44. As the engine owner, I want the GPU runbook to cover the G2 run end to end, so that the gate is a checklist rather than tribal knowledge.
45. As a coding-agent user, I want a long prompt's first token to arrive promptly over streaming, so that the engine feels responsive at the start of a session.
46. As a future kernel author, I want the chunked path's route selection to stay inside the program layer, so that replacing a vendored kernel with our own does not touch the ABI or the scheduler.
47. As the engine owner, I want this phase to leave decode's behaviour otherwise unchanged, so that a G2 regression is attributable to prefill work.

## Implementation Decisions

**Scope boundary.** This phase makes *prefill* real. It does not batch
prefill across requests, does not interleave prefill with decode, does not
capture graphs and does not change the KV format. Where a cheap improvement
over the reference presents itself it is recorded as a note or a knob, not
pursued: G2 is a gate, not the optimization phase (ADR 0005's ordering).

**The chunk loop lives in the leaf.** `ignis_program_prefill` keeps its
span + start-position contract and gains an internal loop over prefill
chunks. A caller may pass a span of any length; the leaf decides how it is
cut. The chunk size is a model-load option, defaulting to 1,024 and
validated as a nonzero multiple of 128 (the reference's own alignment rule,
and the alignment the vendored GDN chunked kernels' 64-token chunk divides
evenly). Rust is free to keep passing whole prompts; when G3 wants to
interleave prefill with decode it hands the leaf smaller spans, with no ABI
change.

**The step ABI is extended once, deliberately (ADR 0016).** Prefill takes a
pointer to an options struct whose first field is its own size:

```
struct ignis_prefill_options {
  uint32_t size;            /* sizeof(struct ignis_prefill_options) */
  int32_t  route;           /* chunked (default) | per_token */
  int32_t  compute_policy;  /* engine default | force A16 */
};
```

`NULL` means the production defaults (chunked route, engine policy). The
leaf rejects a size it does not recognize. No `_ex` entry point and no
compatibility wrapper is kept: the only consumer of this ABI is the Rust
binding in the same repository, compiled from the same tree, so a
compatibility shim would be permanently dead code. ADR 0009's claim that
G2's chunked prefill would be "a leaf change, not an ABI change" held for
the span+position shape and did not hold for options; the ADR is amended to
say so.

**Compute policy.** The engine adopts the reference's text-model policy
verbatim: `AllowA4` on every NVFP4 projection, in prefill *and* decode. The
actual route is then chosen by the vendored dispatch's own per-projection
token thresholds — the attention input projection, the GDN input
projection, the SwiGLU gate/up projection and the residual (linear+add)
projections each have their own — so the engine never encodes a threshold
of its own. `A16Only` remains reachable through the options struct, for
tests that want to compare routes on identical inputs. This is a numerics
change on the decode path too, which is why the canary floor is re-measured
rather than assumed.

**GQA layers in a chunk.** The layer keeps its G1 order (input norm → fused
q/k/gate/v projection → q/k norm + RoPE at the token's position → KV append
→ attention with the sigmoid output gate → out-projection + residual → post
norm → SwiGLU → down + residual). Two changes: positions for the chunk are
the token's absolute positions across the whole chunk, and the append and
the attention go through the fused append+attend entry point, whose route
resolver selects the tensor-core prompt route for widths above the small-T
threshold. The per-layer attention workspace is sized from the vendored
capacity query for the configured chunk width and the sequence's execution
envelope, not per call.

**GDN layers in a chunk.** The layer keeps its G1 order (input norm → fused
q/k/v/z projection → causal conv + SiLU on the sequence's rolling taps →
gating projection and gating → recurrence → gated RMSNorm with z →
out-projection + residual → MLP tail). The recurrence moves from the
in-place entry point (which always runs the recurrent kernel) to the
distinct-state entry point (which runs the chunked tensor-core kernels over
whole 64-token chunks and the recurrent kernel over the tail), with the
sequence's slot supplied as both input and output state. The conv taps roll
across chunks as they already do across tokens.

**Sequence state across chunks.** A chunk advances the sequence's KV pages,
GDN slot, conv taps and position exactly as the per-token loop did, so the
state after N tokens is independent of how those N tokens were cut into
chunks and spans. This is the property that makes the self-oracle and the
span-split test meaningful, and it is what G4's prefix reuse will rely on;
no extra boundary hook is added for G4 in this phase.

**Scratch and synchronization.** The prefill scratch (activations for a full
chunk, the NVFP4 W4A4 workspace, the attention partials, the GDN chunked
workspace) is sized once at model load by asking each vendored op its
workspace capacity for the token interval [1, chunk], and reserved in the
model's arena. A chunk size that does not fit the device budget fails at
load with a message naming the shortfall, never at the first long prompt.
The reserved bytes are included in the runtime's reported VRAM. The program
synchronizes once per chunk instead of once per layer; an error is reported
against the chunk (its span offset and its sequence), which is enough to
diagnose without paying 64 stalls per traversal.

**Failure semantics.** A chunk that fails leaves the sequence's position and
pending token as they were before that chunk, and the error names the chunk.
The Rust side keeps today's behaviour of returning the batch's sequences to
a clean state on a prefill error, so the scheduler's retry path is unchanged.

**Context configuration.** The maximum per-sequence context becomes a
configurable value rather than a compiled-in 4,096, with a default large
enough to admit a 32K prompt plus its generation budget. The sequence pool
is still sized from the physical page budget the leaf computes for the
device, and the scheduler's admission accounting continues to be derived
from the page geometry the runtime reports, so growing the context does not
let admission over-promise.

**Server surface.** `--prefill-chunk` and the context limit join the
existing flag/env/default precedence in the server's config module, with the
same fail-fast validation style: an unaligned or zero chunk is a usage error
before any loader work starts.

**Bench: the `ttft` subcommand.** A new `ignis-bench ttft` drives one or more
cells against a single OpenAI-compatible endpoint. Per cell: a prompt
length in tokens, a sample count (default 5) plus one warmup, `max_tokens`
small, greedy, thinking disabled, streaming (TTFT is the arrival of the
first content delta). Prompts are generated deterministically from a seed
and the artifact's tokenizer to an exact **post-template** token count, and
every sample — warmup included — gets a distinct prompt whose divergence
begins at the first content token, so a shared template header cannot
produce a partial prefix hit. Each sample records the engine's reported
computed-prefill-token count (both engines expose it: the reference in its
request log / usage, ignis through its runtime statistics); a sample whose
computed prefill is short of the prompt length is void and fails the cell.
The record holds every sample, the median, the endpoint identity, the engine
build and profile, the artifact identity and a session identifier.

**Bench: the G2 gate check.** A gate subcommand takes two records — ignis
and the reference — and reports, per cell, the ratio of medians against the
threshold 1.5. It refuses a verdict if the records do not share a session
identifier (ADR 0015's live/live requirement), if any cell is missing on
either side, or if any sample was void. The reference is measured in the
owner's production profile, hq-e8-2b KV included; the KV-format difference
is recorded in the verdict as a known inequality rather than corrected for.
A committed reference record is kept as a fixture for regression and sanity
checks only, and the gate check refuses to use it as the live side.

**Vendoring.** The reference's op tests for the routes this phase turns on
are vendored through the manifest script alongside the code they test: the
NVFP4 A4 linear test and the A4 cases of the attention-input and GDN-input
projection tests. The already-vendored `linear_add` and `linear_swiglu`
NVFP4 tests, the GQA attention test and the GDN test already carry
multi-token and A4 coverage; they move from "compiled" to "run" for these
routes.

**Documentation.** ADR 0015 (live/live G2 gate on cold-prefix samples) and
ADR 0016 (extensible options at the step ABI) are written; ADR 0009 gains an
amendment note. `CONTEXT.md` gains the phase's vocabulary (prefill chunk,
chunked prefill, per-token prefill route, compute policy, TTFT cell,
live/live gate, cold prefix). The roadmap's G2 row points here, and the
verdict is recorded in the review and the roadmap when the gate runs.

## Testing Decisions

A good test here drives the step ABI, the server or the bench and checks
observable behaviour: the tokens a sequence predicts, the state a sequence
holds after a step, the error a bad configuration produces, the numbers a
record contains. It never asserts a kernel's tiling, a route's name or the
program's dispatch order — the route a chunk takes is the vendored
dispatch's decision, and pinning it in a test would freeze a choice this
project intends to revisit.

- **Self-oracle, chunked vs per-token (GPU profile).** One long canary
  prompt (8K) prefilled through the chunked route and through the per-token
  route on freshly allocated sequences; the two runs' teacher-forced
  next-token agreement over the scored positions must be ≥ 95%, the same
  floor and the same scoring as ADR 0014. This is the phase's primary
  correctness check and needs no reference engine. Prior art:
  `crates/server/tests/oracle_teacher_forced_gpu.rs` and the scoring helpers
  in the bench crate.
- **Span split (GPU profile).** A prompt prefilled as one span and the same
  prompt prefilled as two consecutive spans split at a boundary that is not
  a chunk multiple must leave the sequence predicting the same next token,
  and must produce the same first decoded tokens. Prior art:
  `crates/core/tests/program_full_gpu.rs`.
- **Teacher-forced canary floor (GPU profile).** The G1 gate re-run on the
  chunked path with the engine's policy: ≥ 95% overall, no waiver. Prior
  art: the existing G1 gate test, unchanged in shape.
- **Determinism (GPU profile).** Two fresh model loads prefilling the same
  long prompt produce identical tokens.
- **Chunk-size invariance (GPU profile).** The same prompt prefilled with
  two different valid chunk sizes leaves the sequence predicting the same
  next token, so the chunk size is a performance knob and not a correctness
  parameter.
- **Failure and recovery (GPU profile).** A prefill that fails mid-span
  leaves the sequence at its pre-chunk position and reports an error naming
  the chunk; the sequence can then be released and re-allocated to zero
  state.
- **Load-time sizing (GPU profile).** A chunk size whose scratch does not
  fit fails at model load with a message naming the shortfall; the reported
  VRAM grows with the configured chunk size.
- **Long-prompt server e2e (GPU profile).** A streaming chat completion with
  a multi-thousand-token prompt returns coherent text and finishes with
  `finish_reason: stop`. Prior art: `crates/server/tests/openai_http_gpu.rs`.
- **Vendored op tests (kernel leaf, CTest, GPU).** The A4 linear, A4
  attention-input and A4 GDN-input tests newly vendored, plus the existing
  multi-token cases of the linear-add, SwiGLU, GQA attention and GDN tests,
  all at real 27B geometry against the reference's own references and
  tolerances. Prior art: the leaf's existing op-test executables.
- **Bench, CPU.** Prompt generation hits the requested post-template token
  count exactly and produces distinct prompts per sample; a record's median
  and ratio arithmetic; the gate check's refusals (records from different
  sessions, a missing cell, a void sample); the void-sample rule when
  computed prefill is short. Prior art: the bench crate's existing
  CPU tests against a mock endpoint (`replay`, `gate`, `canary`).
- **Bench, GPU profile.** One cell measured end to end against a running
  `ignis-server` produces a record whose samples are all cold (computed
  prefill equals the prompt length).
- **Server config, CPU.** The new flags' precedence over their environment
  variables and their defaults; an unaligned or zero chunk size is a usage
  error before any loader work. Prior art: the config module's existing
  precedence tests.
- **Vendor integrity, CPU.** The newly vendored test files verify against
  the manifest, so `cargo test` stays the guard on the subtree (ADR 0010).
- **The gate itself.** The G2 verdict is a GPU-profile run on a free 5090
  with the reference stopped and restarted around it (ADR 0006): measure
  the reference in its production profile, measure ignis, run the gate
  check on the two records from that session, and record the verdict in the
  review and the roadmap.

## Out of Scope

- Batched prefill across requests, and any prefill/decode overlap or
  interleaving (G3, and the north-star item beyond it).
- Batched decode rounds, per-width CUDA graph capture, PDL, non-greedy
  sampling, the request-log JSONL (G3).
- hq-e8-2b KV, device prefix reuse, the KV-RAM tier's actual
  snapshot/restore, tagged lanes (G4). This phase only guarantees the state
  property those features will build on.
- MTP, DFlash2, ReplaySSM (G5). Vision.
- The 99% performance gates (G3's decode gate, G4's trace-replay gate).
- A chunk-size sweep, and any kernel work of our own: the chunk size is
  exposed as a knob and left at the reference's default; replacing a
  vendored kernel stays deferred to measured per-family work after G4
  (ADR 0010).
- Making the two engines' KV formats comparable for the G2 measurement.

## Further Notes

- The gate's ratio is a *floor* ("not meaningfully worse than the
  reference"), consistent with the project's north star of being at least as
  fast as the reference. G3 and G4 tighten to 99%; nothing here licenses
  settling at 1.5×.
- The reference publishes 8K and 64K prefill cells but no 32K cell, which is
  one more reason the gate is measured live rather than read from a table.
- The per-token route survives this phase as a test-only route. Whether it
  survives G3 is a G3 decision: if the self-oracle is superseded by better
  checks, deleting it is a simplification, not a regression.
- The GDN chunked kernels work in fixed 64-token chunks internally; a
  1,024-token prefill chunk contains sixteen of them plus, in general, a
  recurrent tail. Chunk sizes that are multiples of 128 keep that tail empty
  for full chunks.
- Work splits naturally into tracer-bullet tickets: correctness first (the
  chunk loop with the routes as they already are, proven by the self-oracle),
  then one route family at a time (W4A4 policy, prompt attention, GDN
  chunked), then the performance and measurement work (synchronization,
  scratch sizing, context and server flags, the bench cell, the gate run).
  Cutting those tickets is the first step of executing this spec.
