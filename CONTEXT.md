# CONTEXT

Glossary for the `ignis` project. Vocabulary only — not a spec, not implementation notes.
When output names a domain concept, use the term as defined here.

## Engine & process

- **ignis** — the Rust inference engine for Qwen 3.8-27B on a single RTX 5090 (SM120a).
  Deliberately specialized: one model family, one GPU class.
- **Kernel leaf** — the C++/CUDA compute library linked into ignis behind a C ABI.
  The forward pass and all GPU compute live here; Rust owns everything above the
  **step** — scheduling, KV accounting, serving (ADR 0009).
- **Step ABI** — the C ABI between Rust and the leaf, named after its
  granularity: one call per *step* (model load, sequence lifecycle, a prefill
  span, a decode round), not one per operator. Device-resident — no host
  activation pointer crosses it (ADR 0009).
- **Program** — the leaf-side layer that owns the forward pass and the
  sequence state, on top of the vendored ops. Ours, not vendored (ADR 0009).
- **Vendored op** — an operator copied *verbatim* from the reference and
  tracked by a manifest (pinned commit, content hashes, recorded patches).
  Anything hand-written is not one, and carries no port claim (ADR 0010).
- **Lane** — a concurrent decode slot. Requests hold a lane while decoding.
- **Lane tag** — the request's own statement of its class, carried as an ignis
  extension field on the request and echoed in the request log. Two classes,
  `Interactive` and `Agent`; absent or unrecognized means `Interactive`. It is
  what makes a per-class gate cell and the **eviction priority** mean anything.
- **Prefill lane** — the single global prefill slot: exactly one request at a
  time holds device-resident **prefill progress** and consumes chunks, and
  every other prefill queues behind it. It is a slot, not a promise about
  decode: whether decoding continues while it runs is decided by
  **prefill/decode interleaving**, not by this term (ADR 0018).
- **Admission state machine** — the fairness machinery (protection, backfill class,
  temporal credit, frontier distance) deciding which request gets which lane.
- **Hot reload** — in-place model reload without restarting the server; the model
  lifecycle is decoupled from the server lifecycle.
- **Performance-first** — the organizing principle: correctness (a self-check
  of *sane* output) is a non-negotiable floor; above it, performance is the #1
  objective and tie-breaker for every scope, feature, and kernel decision.
- **North-star** — the holistic objective: "the best local coding engine" — max
  performance **and** agent parallelism that saturates the GPU in prefill *and*
  decode.
- **Self-bootstrapping** — the dev/test loop where the very runner used to build
  ignis (ninfer, running Qwen 3.8-27B) is running while we test; two engines
  share the 5090, so GPU testing is exclusive (ADR 0006).
- **Reference** — the ninfer stack: the source of the artifact, the vendored
  ops, the oracle recordings and the speed numbers. A reference, not a target
  to match token-for-token (ADR 0005 / 0007).

## Model

- **Layer kinds** — the model is 64 layers: **16 GQA** (full attention) and
  **48 GDN** (linear attention), hidden 5120, intermediate 17408, vocab 248320.
- **GQA layer** — a grouped-query attention layer: 24 query heads / 4 KV heads
  of 256, over paged KV, rotated by RoPE at the *token's* position.
- **GDN layer** — a gated-delta-net (linear attention) layer: **16 key heads
  and 48 value heads of 128**, whose recurrence carries the sequence's GDN
  slot instead of a KV cache.
- **Fused plane** — a single stored tensor holding several projections stacked
  by row (e.g. `attention/query_key_gate_value`, `mlp/gate_up`). Its row order
  is part of the artifact contract, not a guess.

## Weights & artifact

- **Artifact** — the `.ninfer` container: the native NInfer model artifact
  (base objects + grafted DFlash2 module). Carries NVFP4 tensors, BF16 exception
  tensors, W8G32 endpoints (embedding, output head), and frontend objects
  (tokenizer / chat template).
- **Object** — the unit inside an artifact. The binder must consume every object
  at bind time; an unconsumed object is a load failure.
- **Device view** — the artifact crate's export for a bound tensor: device
  pointer, storage layout, format and shape, so the program binds planes
  directly with no host dequant.
- **NVFP4** — the weight quantization format of most of the model: E2M1 values,
  an E4M3 group scale per 16 stored in a separate **scale plane**
  (`blockscale-k16-m128x4-v1` layout), and a per-tensor **weight divisor**.
- **hq-e8-2b** — the reference's HyperQuant KV cache format, and the profile the
  owner actually runs: a randomized Hadamard rotation, E8-lattice quantization
  and Rice coding packed into a fixed per-row budget of 64 code bytes plus 8
  metadata bytes per (token, KV head). Fixed bytes per token, so page
  addressing and CUDA-graph address stability are unchanged. 9,216 bytes per
  sequence-token against BF16's 65,536 — a factor of 7.11.
- **KV format** — which of the two KV cache formats a load runs: **hq-e8-2b**,
  the serving default from G4, or **BF16**, retained as the format every
  correctness oracle runs against. A model-load option, fixed for the life of
  the load; the pool is sized by a byte budget and its token capacity is
  *derived* from the format rather than configured (ADR 0022). The CLI default
  is `hq-e8-2b` since #123 wired its prefill and decode attention routes and
  captured its decode graphs; a correctness oracle asks for `bf16` by name.

## Sequence state

- **Sequence handle** — the leaf-owned, opaque per-sequence object created by
  the step ABI: its KV pages, its GDN slot, its conv taps, its position, its
  last token, and (P3-03) its penalty-count row. It is what a snapshot
  captures and a restore rebuilds. No RNG state is one of its sections: the
  device-side sampler's RNG is counter-based (keyed by seed, position and
  purpose), so nothing but the handle's own position needs to be carried for
  it.
- **KV page** — a fixed-size device page of the paged KV cache, addressed
  through a per-sequence block table. Page geometry is reported by the runtime
  and is what the scheduler's KV pool counts.
- **GDN slot** — one sequence's linear-attention state: per GDN layer, **48
  value heads × 128 × 128 fp32** (all 48 GDN layers ≈ 144 MiB per sequence),
  drawn from a slot pool sized by the concurrency.
- **Conv taps** — the per-sequence causal-conv1d history of a GDN layer.
- **Penalty-count row** — a sequence's per-vocab-entry occurrence count
  (one int32 per vocab entry), drawn from a slot-indexed device buffer the
  same shape as the KV/GDN pools (one row per slot). Device-side sampling's
  presence/frequency penalties read and atomically update it; zeroed at
  `ignis_seq_alloc` like every other slot section.
- **State section** — one part of what a sequence is made of: its KV pages, its
  GDN slot, its conv taps, its position and last token, its penalty-count row.
  Each is either shareable read-only (KV pages) or must be cloned per sequence
  (everything mutable). The table of them lives inside the leaf, and adding one
  is the act that re-earns the **snapshot point** permission rather than
  inheriting it (ADR 0024).
- **Snapshot blob** — the opaque, versioned byte image of a whole sequence that
  crosses the ABI. Callers ask the leaf for its size and a format version and
  never see the section layout; a restore validates the header and refuses a
  stale or foreign one. Whole-sequence only: GDN state cannot be recomputed
  without re-running the prefix a restore exists to avoid (ADR 0024).
- **KV-RAM** — the host-RAM tier that holds **snapshot blobs** of evicted
  sequences so they resume instead of re-prefilling. Bounded by a byte budget,
  two-tier (probation → protected), and written only at a **chunk boundary**.
  Control plane since v1; it moves real device state from G4.
- **Prefix reuse** — concurrent requests sharing a prefix skip the redundant
  prefill: the shared **KV pages** are refcounted and shared in place on the
  device, and the mutable **state sections** are cloned device-to-device. It
  never round-trips through host RAM, and sharing pages alone would save
  nothing, since prefill must traverse every layer to produce the mutable
  state (ADR 0024). Real since G4.
- **Shared prefix** — the leaf-owned object prefix reuse is built on: the
  physical KV pages of a prompt head plus a device image of the mutable state
  at its end, held by a refcount. Every holder's block-table row addresses the
  same pages, the pool is charged for them once, and they return when the last
  holder releases. A sequence holding one cannot be snapshotted — its history
  is not all its own — so it is released and re-prefilled rather than evicted.
- **Publish point** — the chunk boundary a prefix is published at, always a
  whole number of KV pages in. It is a scheduling decision, not a detail of
  the publish call: what a claimant clones is the mutable state at the
  prefix's *end*, so the publishing request's prefill is cut there, and a
  prompt whose length is not a whole page pays one extra chunk for it.
- **Eviction priority** — the one ordering that decides what loses residency,
  expressed at two levels: leaving the GPU is eligibility and protection, then
  request class, then least-recently-used; leaving the host tier is request
  class, then probation before protected, then least-recently-used. Protection
  outranks class only where something is actively being served (ADR 0023).
- **Chunked prefill** — prefilling a prompt span through the span+position
  prefill call in **prefill chunks** rather than one token at a time. The
  leaf knows how to loop over a span of any length; it is no longer the only
  one that does — under interleaving the scheduler hands it one chunk per
  call and keeps the loop itself.
- **Prefill chunk** — the number of tokens one traversal of the model
  processes during chunked prefill: a model-load option, default 1024, a
  multiple of 128. The unit the prefill scratch is sized for.
- **Per-token prefill route** — the G1 prefill path that runs the program one
  token at a time (recurrent GDN, small-T attention). Retained after G2 as a
  test-only, per-call route: the self-oracle chunked prefill is checked
  against. Never the default.
- **Compute policy** — the per-call activation-precision policy handed to
  every NVFP4 projection (`A16Only` or `AllowA4`). The engine's policy is the
  reference's: `AllowA4` on every NVFP4 text projection, prefill and decode
  alike, with the vendored per-projection token thresholds deciding the
  actual route. Tests may force `A16Only`.
- **Decode round** — one traversal of the model for *all* decode-ready
  sequences in a batch; the unit a decode CUDA graph is captured over, per
  batch width.
- **Prefill batch** — the group of queued requests handed to the backend in
  one prefill call. A call shape, not a traversal: the runtime walks the group
  and runs one model traversal per request. Naming a group does not make it
  one forward pass.
- **Packed prefill** — several requests' prefill tokens in *one* traversal of
  the model (varlen attention, per-sequence GDN state side by side). The thing
  **prefill batch** is often mistaken for. Not built; its phase is decided
  after the per-chunk synchronization investigation reports.
- **Prefill/decode interleaving** — running a decode round between two prefill
  chunks of the same sequence of work, on the one model stream. Nothing is
  concurrent on the GPU: the two take turns, so a long prefill costs the
  decode lanes one chunk of latency instead of the whole span. The scheduler
  drives it by passing one chunk per prefill call.
- **True prefill/decode overlap** — prefill and decode resident on the GPU at
  the same time, on separate streams, contending for SMs. Distinct from
  **prefill/decode interleaving** and deliberately not built: the north-star
  item (roadmap phase 6), excluded from G3 by ADR 0018.
- **Chunk boundary** — the position between two prefill chunks: the sequence's
  KV pages, GDN recurrent slot, conv taps and position have all advanced past
  the same token, and the chunk's synchronization has returned. A property of
  the serving loop, and what **prefill progress** resumes from — resuming is
  nothing more than scheduling the next chunk.
- **Snapshot point** — a position from which the *whole* sequence state may be
  captured to the host tier and later restored. A permission, not a
  consistency claim, and granted to a tier that does not exist before G4.
  Every **chunk boundary** is one for the state that exists today; the two
  terms coincide without being the same property, and a new state section
  (G4's exact-key side store) must earn the permission again.
- **Prefill progress** — how far into its prompt a request has been prefilled.
  It makes `Prefilling` a state a request *lives in* for tens of rounds rather
  than a moment between two, so the admission machinery can see a
  half-prefilled request. Resuming needs no mechanism: it is just the next
  chunk being scheduled. **Cancel is abort, not suspend** — the in-flight
  chunk finishes, then the sequence is released with its KV pages, GDN slot
  and conv taps. Until the KV-RAM tier exists, a half-prefilled request can be
  paused but not evicted without losing the work — a limit of the tier, not of
  what a **chunk boundary** can capture (ADR 0018).
- **Decode round** — one step for every decode-ready lane, run as a single
  batch-wide traversal of the model: the lanes are the batch's rows at one
  token each, so eight lanes stream the weights once rather than eight times.
  Every per-sequence input — token id, absolute position, physical pool slot
  — is read from device staging indexed by row, and that row indirection is
  what keeps the lanes' KV pages, GDN slots and conv taps isolated inside the
  one call (ADR 0020, which amends ADR 0019's per-lane round).
- **Decode graph** — the CUDA graph replayed for a decode round, captured per
  **exact** batch width 1..8. Widths are never padded up to a captured one:
  the GDN slot traffic is per sequence (144 MiB each), so padding a width-1
  round to width 8 moves 1.15 GB instead of 144 MB and spends several times
  the C=1 gate's whole margin. Its staging buffers are a reservation separate
  from the prefill scratch, shared across the widths and sized for the widest.
- **Per-lane sampling** — sampling parameters and RNG state carried per
  sequence, not per decode round: lanes in one round hold different
  temperatures, seeds and penalty histories. Sampling happens device-side in
  the leaf, which returns token ids and never ships logits to the host. What
  a request generates therefore depends on its own seed alone, never on which
  lanes happened to share its round.
- **N-lane concurrency** — 8 resident decode lanes (N=8), with overflow to the
  host KV-RAM tier; sized for a ~10-subagent concurrent coding workload. Real
  at long context only under hq-e8-2b: eight lanes at a 40,960-token context
  are ~20 GiB of BF16 KV against ~3.02 GB of hq (**KV format**).
- **DFlash2** — the 5-layer sliding-window (2048) speculative-decoding drafter
  (hidden 5120, draft tokens 1..7, native acceptance 3.4–3.7 tokens/round).
- **MTP** — the model's native multi-token-prediction heads (draft window 3,
  adaptive verification width).
- **Vision** — multimodal (image/video) input.

## Observability

- **Canonical event** — the one structured representation of a log occurrence
  (severity, event name, body, attributes, trace context). Not a Rust struct
  every call site builds by hand: it's `tracing::Event` + active span
  context, observed by the JSON/pretty `Layer`s. Formatters are presentation,
  never the source of truth (ADR 0011).
- **Event name** — a stable, class-level identifier (`ignis.<subsystem>.<event>`,
  e.g. `ignis.model.loaded`), never occurrence-specific data.
- **`logging` crate** — owns `tracing_subscriber` setup, the JSON/pretty
  `Layer`s, and log-format/log-level config resolution; every other crate
  depends on it only for macros/init, never the reverse. Distinct from the
  pre-existing **server telemetry** (`crates/server/src/telemetry.rs`,
  design §5): the scheduler interval-counter/request-lifecycle JSONL stream
  behind `IGNIS_TELEMETRY`/`--telemetry`. Interval counters are metrics-shaped
  and stay out of the logging system's scope; the request-lifecycle line
  (`admitted`/`ttft`/`done`) migrates onto canonical `ignis.request.*` events.
  The two event models never merge — but since GitHub #108, telemetry's
  sink *implementation* (line-writing only: `StdoutSink`/`FileSink`/
  `MemorySink`/`NullSink`) is `ignis_logging::sink`'s, not a duplicate of
  its own; the interval line's facts, counters, and JSONL shape stay
  telemetry's alone.
- **Logging queue** — the handoff decoupling event *creation* (a `tracing`
  call site, potentially on the prefill/decode path) from physical I/O
  (potentially slow: a full disk, a stalled pipe). Two channels, two
  deliberately different backpressure policies: the priority channel
  (INFO/WARN/ERROR) is bounded and blocks briefly rather than drop — an
  ERROR must never silently vanish; the debug/trace channel (DEBUG/TRACE) is
  a bounded ring buffer that drops its oldest entry on overflow rather than
  ever blocking. A background thread is the only thing that ever performs
  the real write. Flushing at shutdown waits, up to a fixed budget, only on
  the priority channel — a pending DEBUG/TRACE line is exactly what that
  channel already permits losing.
- **Sensitive attribute redaction** — an attribute whose key contains a
  sensitive whole word (`password`, `secret`, `token`, `authorization`,
  `bearer`, `cookie`, `credential`) or word pair (`api_key`, `private_key`)
  is redacted by construction, not by call-site discretion — a call site
  naming a field carelessly is caught before it ships rather than relying on
  every call site remembering to redact by hand.
- **`request.id`** — reused as the OTel `trace_id` for the request's span
  tree (root span at HTTP ingress, children per admission/prefill/**decode
  round**/MTP verify/completion); one identifier for the same causal unit
  across fairness, logs, and spans (ADR 0012). Spans stop at decode-round
  granularity, never per-token — a per-token span is hot-path logging by
  another name.
- **Hot-path logging constraint** — normal INFO operation must not emit one
  record per token/layer/kernel/allocation on the prefill/decode path; any
  logging or tracing change touching that path must pass the G4 performance
  gate (ADR 0007) before merge, not just design review.
- **Hotpath lint** — the grep-based, hand-maintained scan over a fixed list
  of prefill/decode/CUDA-graph-adjacent files that fails `cargo test` on any
  INFO-or-above `tracing` call or raw `println!`/`eprintln!`/`print!` found
  there. What makes the **hot-path logging constraint** checkable in review,
  not only provable later at the G4 gate. A `hotpath-lint-allow:` comment
  marks a reviewed, intentional exception, never a silent one.

## Acceptance

- **Gate** — a milestone acceptance measured **on the GPU**, never on CPU
  tests. G1 correctness floor, G2 prefill, G3 decode + serving loop, G4 the
  reference feature floor, G5 speculative decoding (`.scratch/ROADMAP.md`).
- **GPU profile** — the explicit test profile for GPU work: it requires the
  5090 free and **fails**, never skips, when the GPU is busy or a kernel
  errors. A skip is not green for compute work.
- **Canary suite** — the fixed set of short, high-signal prompts used to detect
  divergences.
- **Canary oracle** — the recorded fixture the canary suite is checked against:
  the reference engine's greedy (exact-argmax) completions on those prompts for
  the same artifact, tokenized with the artifact's tokenizer.
- **Teacher-forced agreement** — how the canary oracle is scored, and the G1
  correctness floor: at each of the first 32 positions the engine is fed the
  *oracle's own* token prefix, and its greedy next-token argmax is compared
  with the oracle's recorded token. Positions are scored independently, so one
  divergence cannot cascade. The floor is ≥ 95% overall (ADR 0014). It is a
  sanity floor for gross implementation errors — wrong layouts, missing ops,
  broken state wiring, wrong positions, corrupted activations — **not** a
  parity requirement: ignis may differ numerically and linguistically from the
  reference (ADR 0007).
- **Free-running agreement** — the older scoring, where the engine generates
  its own continuation and the two token streams are diffed. **Diagnostic
  only** (ADR 0014): after the first divergence the engines no longer share a
  prefix, so it measures continuation similarity, not forward-pass health.
- **TTFT cell** — one fixed prompt length (8K, 32K) at which time to first
  token is measured through the bench, the same way against either engine:
  streaming, greedy, thinking off, a small output budget, deterministic
  synthetic prompts of exactly that many tokens. Every sample is a **cold
  prefix**: each sample (the warmup included) gets its own prompt, distinct
  from the first content token, so no prefix cache or host KV tier on
  either engine can serve it; the engine's own computed-prefill-token count
  must equal the prompt length or the sample is void.
- **Live/live gate** — how G2 is judged: ignis and the reference measured on
  the same TTFT cells in the same session on the same machine, and the
  ratio ignis/reference must be ≤ 1.5. A committed reference record exists
  for regression and sanity only; it never decides the gate.
- **Measurement session** — the identifier both engines' TTFT records carry
  when they were measured back to back in one sitting. It is what makes
  live/live checkable rather than asserted: `ignis-bench g2` refuses a
  verdict when the two records do not share one, which is also what stops a
  committed fixture from ever being the live side.
- **TTFT record** — one engine's measured cells plus the identity a later
  reader needs to audit them: every sample and its computed-prefill count,
  the median, the endpoint, the engine, the artifact, the profile, the date
  and the measurement session (`ignis-bench ttft --out`).
- **f64 layer reference** — a CPU fp64 computation of one GQA layer and one
  GDN layer on the real weights, the tolerance target for the leaf's per-layer
  output at G1.
- **Performance gate (99%)** — the acceptance criterion at G4: ≥ 99% of the
  reference's performance (throughput / latency) on the trace-replay load,
  **not** token-agreement;
  correctness is self-checked (sane output, same model, greedy, fixed seed).
  It is the first of a ladder of performance gates (later gates TBD).
- **Trace replay** — re-sending a recorded "1 main agent + N subagents" load
  trace against both engines to compare scheduler behavior, scored per **lane
  tag**. The trace is the load, not the baseline: the reference's own run is
  measured live in the same session (**live/live gate**), over at least two
  launches per engine (ADR 0021), and the verdict is the pooled cell statistic.
- **Load trace** — the recorded agent session a **trace replay** sends. It holds
  real working content, so it is never committed: its SHA-256 and its shape
  (request count, class split, arrival span, prompt-length distribution) ship
  with the gate artifacts, and both run records carry the hash so a reader can
  prove the two engines replayed the same load.
