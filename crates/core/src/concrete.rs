//! The concrete scheduler — `core-04` (N=8 decode lanes + batched
//! prefill) + `core-05` (the full admission state machine, ADR 0004).
//!
//! The v1 scheduler (`docs/design/ignis-v1.md` §2) driving the [`Compute`]
//! seam (production: the kernel leaf via FFI, tests: [`crate::mock::MockCompute`])
//! in three phases per advance:
//!
//! 1. **Chunked, interleaved prefill** (P3-01, ADR 0018) — at most one
//!    `prefill_step` call per `advance()`, spanning at most
//!    `serving_chunk_tokens` tokens per request, so a long prompt costs the
//!    decode lanes one chunk of latency instead of the whole span (a 32K
//!    prompt no longer inserts a multi-second gap into every other lane's
//!    token stream). Several **queued** requests may still share that one
//!    call — up to `max_prefill_batch` of them, `PrefillJob`s batched
//!    together — but only while each finishes within its own single chunk;
//!    the moment one of them would need more than one chunk, the batch stops
//!    there. **Exactly one request may hold multi-tick (device-resident)
//!    prefill progress at a time**: once a request's chunk leaves it
//!    incomplete, it alone is served on every following `advance()` (`the
//!    rest queue`, `run_admission`'s candidates are unaffected — a request
//!    that already finished prefilling in an earlier tick is still dealt a
//!    lane normally) until it finishes. `RequestState::Prefilling` is
//!    therefore **durable**: [`Request::prefill_progress`] carries how far a
//!    request has gotten, and [`Request::prefill_complete`] is what
//!    `run_admission` gates a lane deal on — a half-prefilled request is
//!    never dealt a lane (never decoded). Each completed chunk — the last
//!    one included — is recorded as a GDN resumable boundary
//!    ([`crate::gdn::GdnState::checkpoint`]); a chunk call that errors
//!    leaves the request's progress untouched, so the retry resends the
//!    same (not-yet-applied) span. **Sibling prefix reuse** (core-07,
//!    device-backed since P4-10 / GitHub #126) runs here too: before a
//!    *fresh* request's first chunk, it claims the longest cached prefix of
//!    its prompt (skipping the redundant prefill — its first job carries
//!    only the tail, and the backend allocates its sequence *against* the
//!    leaf's prefix, sharing the pages in place and cloning the mutable
//!    state device-to-device). A request with no claim publishes the whole
//!    KV pages of its own prompt — and its prefill is **cut at that
//!    boundary**, because the state a claimant clones is the state at the
//!    prefix's end, so a chunk that overshot it would have moved that state
//!    on. A prompt that is not a whole number of pages therefore pays one
//!    extra chunk, and every sibling skips its whole head. The shared
//!    entry's pages are charged to the pool once (the charge split), and the
//!    admission machine runs against the pool minus the cache's pins
//!    (consistent accounting — the cache never over-allocates).
//! 2. **The admission state machine** (core-05) — lane assignment is
//!    driven by the full fairness machinery (`admission.rs`, ported from
//!    the reference stack per ADR 0004): *protection* (a blocked head
//!    freezes the active set and selects the earliest-completion donor
//!    prefix, whose lanes are not evicted), *backfill class* (a
//!    candidate admitted on a donor's future is `Persistent` — it fits
//!    the head's future capacity — or `Temporal` — its own service work
//!    fits the frontier distance and the temporal credit), *temporal
//!    credit* (decays with each temporal backfill admission), and
//!    *frontier distance* (the projected distance to the last still-
//!    active frozen donor). With no protection active the machine
//!    degrades to the class-priority + FIFO deal.
//! 3. **Batched decode** — one compute call spanning every running lane;
//!    a request that reaches its KV reservation (remaining work hits
//!    zero) completes in that step — the pool can never grow past the
//!    reservations (no OOM under the N=8 load, core-01).
//!
//! 4. **KV-RAM host tier** (core-06, real state since P4-07 GitHub #125) —
//!    the overflow path: when a request is blocked by the active set (all
//!    lanes / pages in use), the scheduler evicts a retained lane (driven
//!    by [`crate::admission::choose_retained_lane_victim`]) by snapshotting
//!    it to pinned host memory and releasing its KV pages, GDN slot and
//!    conv taps ([`crate::scheduler::Compute::evict`]), recording the
//!    snapshot in the host-RAM tier ([`crate::host::HostTier`]), bounded by
//!    a **byte budget** (not a page or lane count — a snapshot's fixed GDN
//!    floor is paid regardless of prompt length). The evicted request is
//!    suspended (not done) and **restored** — the blob written back through
//!    [`crate::scheduler::Compute::restore`], resuming generation without
//!    re-prefilling — when a lane frees. The tier's two-tier eviction
//!    (probation → protected) keeps evictions bounded; a snapshot whose
//!    GDN position is mid-prefill is rejected (core-02's boundary).
//!
//! Lane capacity: the scheduler holds [`N_DECODE_LANES`] (8) resident
//! lanes and a KV page capacity (`kv_capacity_pages`, auto-sized from
//! the pool in production; the machine's resource dimension). In-flight
//! admission is capped at `max_in_flight`; beyond N=8 the KV-RAM host
//! tier (core-06) provides overflow — a blocked head is admitted by
//! evicting a retained lane to the host tier, and a request whose KV
//! reservation exceeds the whole pool is rejected with
//! [`SubmitError::Oversized`]. One whose sequence (prompt + `max_tokens`)
//! is longer than `max_sequence_tokens` is rejected with
//! [`SubmitError::ContextExceeded`] (GitHub #166): the leaf never reserves
//! more than its `max_context` for one sequence, however empty the pool.
//! A prefill that the backend keeps failing ends its request with
//! [`FinishReason::Error`] after [`MAX_PREFILL_ATTEMPTS`] tries.
//!
//! **Request-lifecycle spans (GitHub #81, ADR 0012)** — the full hierarchy
//! is documented once in `docs/design/tracing-spans.md`; this crate's own
//! piece is four `tracing` spans, each tagged `request_id = <RequestId>`
//! (the field `ignis_logging::trace_context` reads to derive `trace_id`,
//! never a separately generated id):
//!
//! - `ignis.admission` — [`Self::try_admit`], the lane-deal decision.
//! - `ignis.prefill` — the per-request accounting for one
//!   `Compute::prefill_step` call (the chunked-prefill call boundary,
//!   opened after the call returns, in the loop that already exists to
//!   record each job's progress — never around the call itself, which may
//!   batch several *different* requests' jobs together and so has no
//!   single `request_id` of its own).
//! - `ignis.decode.round` — the per-request accounting for one
//!   `Compute::decode_step` call (one span per request *per round*, opened
//!   in the existing per-lane outcome loop — same reasoning as prefill:
//!   the call itself may be a batch across lanes). This is the unit a
//!   decode CUDA graph is captured over; nothing finer is ever
//!   instrumented here (never per-token, never per-verified-token once MTP
//!   verify lands — see the design doc for that attach point).
//! - `ignis.completion` — [`Self::mark_done`].
//!
//! All four are opened and dropped on the model thread (GitHub #69), never
//! entered from — and never a tracing-parent of — the HTTP-ingress root
//! span `ignis-server` opens per request (a different OS thread); the two
//! sides are correlated by sharing the same `request_id`/`trace_id`, not by
//! `tracing`'s own parent-child span graph. See the design doc for why.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::checkpoint::{
    CheckpointCapture, CheckpointEntry, CheckpointPool, KV_RAM_RESTORE_FLOOR_TOKENS,
    RetainedKind, RetainedStateOperation, ReuseSource, TierList,
};
use crate::admission::{
    ActiveAdmissionSnapshot, AdmissionProtection, AdmissionResources, ProtectionPhase,
    ResidentCandidate, RetainedLaneCandidate, admission_resources_fit,
    choose_resident_candidate_victim, choose_retained_lane_victim, make_admission_protection,
    persistent_backfill_is_safe, protected_head_safe_without_temporal, protection_frontier_distance,
};
use crate::host::{
    DEFAULT_RETAINED_INTERACTIVE_TTL, HostEntry, HostTier, KvRamVictim, ResumePhase, RetainedBlob,
    RetainedKvRamEntry, Tier,
};
use crate::identity::{MediaKey, PromptContent, PromptKeys};
use crate::prefix::{PrefixCache, PrefixId, Retention, SpilledPrefixId};
use crate::request::Request;
use crate::retained_slot::{RetainedHolder, RetainedSkip, RetainedSlotLedger};
use crate::scheduler::{
    CheckpointClaim, Compute, DecodeJob, DecodeOutcome, Occupancy, PrefillJob, PrefillOutcome,
    RetainedAt, Scheduler, SharedPrefixClaim,
};
use crate::types::{
    BackfillClass, ComputeError, DecisionRead, DecodeParams, EngineMode, FinishReason, LaneId,
    N_DECODE_LANES, RequestClass, RequestId, RequestInput, RequestState, SchedEvent, SubmitError,
};

/// Knobs for the concrete scheduler (v1 defaults; the KV-RAM host tier
/// will raise `max_in_flight` beyond N=8 in core-06).
pub struct SchedulerConfig {
    /// The loaded model id (what `Scheduler::model_id` reports; submissions
    /// naming another model are rejected).
    pub model: String,
    /// In-flight cap (Admitted + Prefilling + Running). Defaults to
    /// [`N_DECODE_LANES`]; host-tier overflow (core-06) raises it.
    pub max_in_flight: usize,
    /// Max requests grouped into one prefill batch per step. Defaults to
    /// [`N_DECODE_LANES`] (group everything eligible in one GPU batch).
    pub max_prefill_batch: usize,
    /// The KV block size in tokens (one KV page holds this many tokens):
    /// sets the per-request reservation granularity (core-05).
    pub kv_page_tokens: u32,
    /// The per-sequence token limit, prompt included — the leaf's
    /// `max_context` in production (GitHub #166). A request submitted
    /// without `max_tokens` may generate `this - prompt` tokens, reserves
    /// `ceil(this / kv_page_tokens)` pages and is completed when it reaches
    /// the limit (core-05: the reservation cannot grow mid-generation). A
    /// request whose prompt + `max_tokens` exceeds it is refused with
    /// [`SubmitError::ContextExceeded`].
    pub max_sequence_tokens: u32,
    /// The KV pool capacity in pages (core-05: the admission machine's
    /// resource dimension; production auto-sizes this from the pool,
    /// tests pass small values to drive contention).
    pub kv_capacity_pages: u32,
    /// The leaf's device-resident sequence-slot capacity (P4-07, GitHub
    /// #125; `ignis_seq_pool_spec::slot_count` on the leaf side): a
    /// **separate** resource dimension from decode lanes — a request
    /// occupies a resident slot from its first prefill chunk (`Prefilling`,
    /// before it ever holds a decode lane) until it completes, is
    /// cancelled, or is evicted. Defaults to [`N_DECODE_LANES`], matching
    /// the leaf's own default `slot_count`.
    pub resident_slot_capacity: u32,
    /// The KV-RAM host tier capacity in bytes (core-06, P4-07 GitHub #125:
    /// the host-RAM budget for evicted (suspended) request snapshots — a
    /// byte budget, not a page or lane count, since a snapshot's fixed GDN
    /// floor is paid regardless of prompt length; production exposes this
    /// as an operator flag, tests pass small values to drive contention).
    pub host_capacity_bytes: u64,
    /// The serving prefill chunk width, in tokens (P3-01, ADR 0018): the
    /// scheduler sends at most this many tokens of a request's remaining
    /// prompt per `advance()`, so a long prefill costs the decode lanes one
    /// chunk of latency instead of the whole span. Must be `<=` the width
    /// the program scratch was reserved for at model load (P2-01 / #83) —
    /// resolve against that bound with [`resolve_serving_chunk_tokens`]
    /// before constructing this config; a real load should default this to
    /// the load width. No adaptive policy: fixed for the scheduler's
    /// lifetime.
    pub serving_chunk_tokens: u32,
    /// Cross-request state reuse (GitHub #186, ADR 0029; the operator's
    /// `--prompt-reuse`). On by default. Off means exactly nothing happens:
    /// no request captures a prompt checkpoint and none claims one, so a cold
    /// bench measures a cold engine and a correctness oracle prefills every
    /// prompt it is given.
    pub prompt_reuse: bool,
    /// The load's **retained slots** (GitHub #215, ADR 0030; the operator's
    /// `--retained-slots`): how many images of mutable state retained state
    /// may hold on the device, prompt checkpoints and shared prefixes alike —
    /// the pool's `retained_slot_count` in production. Every publish and every
    /// capture takes one; when none is free, retained state gives one up, and
    /// when nothing can, the publish or capture is skipped. With
    /// `prompt_reuse` off only live siblings' heads take one (the server
    /// resolves that to 0 unless the operator names a count). Tests pass small
    /// values to drive exhaustion.
    pub retained_slots: u32,
    /// How long a retained Interactive checkpoint in KV-RAM keeps its class's
    /// priority after its conversation last used it (GitHub #190; the
    /// operator's `--retained-interactive-ttl`). Past it the entry ranks as
    /// an Agent's would, so conversations nobody is coming back to cannot
    /// hold KV-RAM against every subagent.
    pub retained_interactive_ttl: Duration,
}

/// The wall clock the scheduler reads (GitHub #190): `Instant::now` in
/// production, a hand-driven one in the tests that need idle time to pass.
/// Nothing else in scheduling reads it — ticks order, this only measures how
/// long retained state has sat unused.
pub type Clock = Arc<dyn Fn() -> Instant + Send + Sync>;

/// The serving prefill chunk width's default, in tokens (ADR 0018): the
/// measured 1,024-token chunk from the G2 gate run (`docs/ROADMAP.md`).
/// P3-01 is CPU-only (driven through the `Compute` seam with `MockCompute`),
/// so this is the value CPU tests and this config's [`Default`] use; wiring
/// a real `--features cuda` load's actual reserved width through
/// [`resolve_serving_chunk_tokens`] is `crates/runtime`'s job, not this
/// crate's.
pub const DEFAULT_SERVING_CHUNK_TOKENS: u32 = 1024;

/// Consecutive failed `prefill_step` calls a request survives (GitHub
/// #166): a single fault is retried on the next advance (core-04), but one
/// that keeps repeating is not transient, and retrying it every advance
/// spins the model thread. The attempt that reaches this ends the request
/// with [`FinishReason::Error`].
pub const MAX_PREFILL_ATTEMPTS: u32 = 3;

/// The media identity of `input`'s prompt, for the content match key
/// (GitHub #189, #193).
///
/// Empty — and allocation-free — for a text-only request. The key's media
/// slot is decided by *what* is mixed in, and taking exactly the processor's
/// per-item content digest and grid out of the items is that decision: two
/// encodes of one image name the same item, the same bytes resized to another
/// grid do not.
fn media_keys(input: &RequestInput) -> Vec<MediaKey> {
    match &input.multimodal {
        None => Vec::new(),
        Some(multimodal) => multimodal.media.iter().map(MediaKey::from).collect(),
    }
}

/// The prompt tokens `r`'s next prefill chunk carries: its remaining span up
/// to `serving_chunk` tokens, cut where a second media item would begin
/// (GitHub #178: one media item per chunk).
///
/// And cut short, when it would leave a last chunk narrower than the
/// request's [`RequestInput::prefill_tail`], so that the last chunk keeps
/// that many (GitHub #260): an attention readout is read where the layer's
/// prompt route materialized its keys, and a chunk of eight tokens or fewer
/// takes a route that materializes none. A tail that cannot be kept — a
/// remainder already narrower than it — is left as it is, and the leaf fails
/// that question loudly.
fn chunk_take(r: &Request, serving_chunk: u32) -> u32 {
    let start = r.prefill_progress;
    let remaining = r.input.tokens.len() as u32 - start;
    let take = remaining.min(serving_chunk);
    let take = match &r.input.multimodal {
        Some(multimodal) => multimodal.cap_chunk(start, take),
        None => take,
    };
    let tail = r.input.prefill_tail() as u32;
    let left = remaining - take;
    if left > 0 && left < tail && remaining > tail {
        remaining - tail
    } else {
        take
    }
}

/// `take` tokens from `start`, cut so the chunk ends on `point` when it would
/// run past it; a `point` at or behind `start` (0 included) cuts nothing.
fn cut_at(start: u32, take: u32, point: u32) -> u32 {
    if point > start { take.min(point - start) } else { take }
}

/// Whether a chunk of `take` tokens from `start` reaches `point` — the chunk
/// a publish or a capture at `point` happens on (0 is no point).
fn lands_on(start: u32, take: u32, point: u32) -> bool {
    point > start && point - start <= take
}

/// The tokens `input` may generate (GitHub #166): its `max_tokens`, or —
/// absent that — whatever the per-sequence limit leaves after the prompt.
///
/// Zero for a **decision** (GitHub #238): it generates nothing, so its
/// whole-sequence reservation is its prompt. That is one `if` rather than a
/// second reservation path because every caller of this — `submit`'s context
/// check and [`sequence_tokens`] both — wants the same answer, and a
/// decision admitted on prompt + `max_tokens` would be refused for a budget
/// it was never going to spend.
///
/// The schedule's length for a **constrained decode** (GitHub #242), for the mirror of
/// that reason: it emits at most one token per step and `max_tokens` is
/// meaningless to it, so a constrained decode admitted on some other number would
/// either reserve pages it cannot use or be cut off mid-number.
///
/// It is the budget and, for a schedule with no terminator, also what ends
/// the run — `remaining_work` reaching 0 is the schedule being spent — which
/// is why the two are one number and not two. A schedule that names a
/// terminator (GitHub #255) may stop before it; this still reserves the
/// whole length, because the reservation is made at submit and which round
/// closes the shape is not known then.
fn generation_budget(config: &SchedulerConfig, input: &RequestInput) -> u32 {
    if input.is_decision() {
        return 0;
    }
    if let Some(schedule) = &input.constrained {
        return u32::try_from(schedule.len()).unwrap_or(u32::MAX);
    }
    input.params.max_tokens.unwrap_or_else(|| {
        config
            .max_sequence_tokens
            .saturating_sub(u32::try_from(input.tokens.len()).unwrap_or(u32::MAX))
    })
}

/// The whole-sequence reservation handed to the leaf: prompt + generation
/// budget. `submit` refuses anything over `max_sequence_tokens`, so for an
/// admitted request this always fits.
fn sequence_tokens(config: &SchedulerConfig, input: &RequestInput) -> u32 {
    (input.tokens.len() as u64 + u64::from(generation_budget(config, input))).min(u32::MAX as u64)
        as u32
}

/// Resolve a serving-time chunk width against the width the program scratch
/// was reserved for at model load (P2-01 / #83, ADR 0018): narrowing at
/// serving time is free (the scratch was reserved for at least this many
/// tokens), widening is refused — the scratch was never reserved to serve a
/// wider chunk, and silently widening would run past it.
pub fn resolve_serving_chunk_tokens(requested: u32, load_width: u32) -> Result<u32, String> {
    if requested > load_width {
        return Err(format!(
            "serving chunk width {requested} exceeds the {load_width}-token width reserved at model load"
        ));
    }
    Ok(requested)
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            max_in_flight: N_DECODE_LANES,
            max_prefill_batch: N_DECODE_LANES,
            kv_page_tokens: 16,
            max_sequence_tokens: 8192,
            // Eight full sequences fit by default: the resource dimension
            // of the admission machine is dormant unless the capacity is
            // tightened (or the pool is auto-sized smaller in production).
            kv_capacity_pages: (N_DECODE_LANES * (8192 / 16)) as u32,
            resident_slot_capacity: N_DECODE_LANES as u32,
            // A generous default headroom for CPU tests (`MockCompute`'s
            // `Compute::evict` reports a nominal 1-byte snapshot per
            // request, GitHub #125) — production wires the operator's own
            // byte flag rather than this default.
            host_capacity_bytes: (N_DECODE_LANES * (8192 / 16)) as u64,
            serving_chunk_tokens: DEFAULT_SERVING_CHUNK_TOKENS,
            // GitHub #186: on by default (ADR 0029), with `--retained-slots`'
            // own default (GitHub #215): a slot per lane. A test that wants
            // exhaustion asks for it by setting fewer here.
            prompt_reuse: true,
            retained_slots: N_DECODE_LANES as u32,
            retained_interactive_ttl: DEFAULT_RETAINED_INTERACTIVE_TTL,
        }
    }
}

/// The concrete N=8 resident-lane scheduler (v1). See the module docs
/// for the per-advance phase structure.
pub struct ConcreteScheduler {
    config: SchedulerConfig,
    compute: Arc<dyn Compute>,
    next_id: RequestId,
    requests: Vec<Request>,
    free_lanes: Vec<LaneId>,
    /// The last hard compute error (kernel fault) hit by an advance, if
    /// any.
    last_error: Option<ComputeError>,
    // ── core-05: the admission state machine ─────────────────────────────
    /// The pool capacity the machine admits against: `[N_DECODE_LANES]`
    /// lanes, `kv_capacity_pages` main-pool pages (the speculative
    /// backend pool is 0 until DFlash2 / MTP, v1.2 / v1.3).
    capacity: AdmissionResources,
    /// Main-pool pages reserved by device-resident requests (over-
    /// reservation: charged in full at first materialization — P4-07,
    /// GitHub #125 moved this off the decode-lane deal, since the leaf
    /// reserves the pages the moment it materializes a sequence, at the
    /// first prefill chunk, not when a decode lane is later assigned —
    /// released when the sequence is released (completion, cancellation,
    /// or eviction).
    kv_used_pages: u32,
    /// Device-resident sequence slots in use (P4-07, GitHub #125): every
    /// request currently materialized on the leaf (`Request::resident`),
    /// `Prefilling` or `Running` alike. Charged/released in lockstep with
    /// `kv_used_pages` — see [`Self::materialize`] / [`Self::unmaterialize`].
    resident_slots_used: u32,
    /// The active protection (core-05): `None` while no head is blocked
    /// (or once the protected head has been dealt).
    protection: Option<AdmissionProtection>,
    /// The next protection epoch (protections start at epoch 1).
    protection_epoch: u64,
    // ── core-06: the KV-RAM host tier ──────────────────────────────────
    /// The host-RAM KV tier (core-06): holds evicted (suspended) request
    /// snapshots in two tiers (probation → protected); evictions are
    /// bounded by `host_capacity_bytes`.
    host: HostTier,
    /// The scheduling tick (a per-advance counter; the LRU `use_tick` for
    /// retained-lane victim selection).
    tick: u64,
    // ── core-07: the sibling prefix cache ──────────────────────────────
    /// The sibling prefix cache (core-07): shared KV prefixes of prompt
    /// heads (whole pages, refcounted); concurrent requests sharing a
    /// prefix skip the redundant prefill.
    prefix: PrefixCache,
    // ── GitHub #186: cross-request state reuse (ADR 0029) ───────────────
    /// The ledger of retained **prompt checkpoints**: the state of finished
    /// requests at their generation opener, which a later request whose prompt
    /// extends one resumes from instead of re-prefilling.
    checkpoints: CheckpointPool,
    /// The retained slots and who holds each (GitHub #215, ADR 0030): the one
    /// bound on the images retained state keeps on the device.
    retained: RetainedSlotLedger,
    /// The slot count last reported in a [`SchedEvent::RetainedSlots`] (none
    /// held before the first).
    reported_slots: u32,
    /// The device checkpoints whose partial tail page is a KV page of the
    /// pool (GitHub #215), by publisher: charged to `kv_used_pages` at
    /// capture, given back with the image.
    tail_pages: Vec<RequestId>,
    /// Wall time, for retained state's idle age (GitHub #190).
    clock: Clock,
}

impl ConcreteScheduler {
    /// A scheduler with the v1 defaults: `max_in_flight` and
    /// `max_prefill_batch` both [`N_DECODE_LANES`], default KV knobs.
    pub fn new(model: &str, compute: Arc<dyn Compute>) -> Self {
        Self::with_config(
            SchedulerConfig {
                model: model.into(),
                ..SchedulerConfig::default()
            },
            compute,
        )
    }

    /// A scheduler with explicit knobs (tests; the host tier will set
    /// `max_in_flight > N_DECODE_LANES` in core-06).
    pub fn with_config(config: SchedulerConfig, compute: Arc<dyn Compute>) -> Self {
        assert!(config.max_in_flight > 0, "in-flight cap must be non-zero");
        assert!(
            config.max_prefill_batch > 0,
            "prefill batch size must be non-zero"
        );
        assert!(config.kv_page_tokens > 0, "KV pages must hold tokens");
        assert!(
            config.max_sequence_tokens > 0,
            "the sequence reservation cap must be non-zero"
        );
        assert!(
            config.kv_capacity_pages > 0,
            "the KV pool must hold at least one page"
        );
        assert!(
            config.resident_slot_capacity > 0,
            "the leaf must hold at least one resident sequence slot"
        );
        assert!(
            config.serving_chunk_tokens > 0,
            "the serving prefill chunk width must be non-zero"
        );
        Self {
            capacity: AdmissionResources {
                lanes: N_DECODE_LANES as u32,
                kv_pages: config.kv_capacity_pages,
                backend_pages: 0,
                resident_slots: config.resident_slot_capacity,
            },
            kv_used_pages: 0,
            resident_slots_used: 0,
            protection: None,
            protection_epoch: 1,
            host: {
                let mut host = HostTier::new(config.host_capacity_bytes);
                host.set_retained_interactive_ttl(config.retained_interactive_ttl);
                host
            },
            tick: 0,
            prefix: PrefixCache::new(config.kv_page_tokens),
            // GitHub #189: the pool holds what this backend's state was
            // produced under, so an entry it could not write into a sequence
            // is never offered to one.
            // GitHub #215: no slot means no publish and no capture is ever
            // asked for, without a second flag check at each site. With prompt
            // reuse off the slots given are for live siblings' heads alone.
            retained: RetainedSlotLedger::new(config.retained_slots),
            reported_slots: 0,
            tail_pages: Vec::new(),
            checkpoints: CheckpointPool::with_tiers(
                compute.blob_identity(),
                if config.prompt_reuse && config.host_capacity_bytes > 0 {
                    TierList::device_and_kv_ram()
                } else {
                    TierList::device_only()
                },
            ),
            config,
            compute,
            next_id: 0,
            requests: Vec::new(),
            free_lanes: (0..N_DECODE_LANES).collect(),
            last_error: None,
            clock: Arc::new(Instant::now),
        }
    }

    /// Read wall time from `clock` instead of `Instant::now` (tests that need
    /// retained state to sit idle for minutes without waiting for them).
    pub fn with_clock(mut self, clock: Clock) -> Self {
        self.clock = clock;
        self
    }

    fn now(&self) -> Instant {
        (self.clock)()
    }

    /// The KV-RAM tier, for observation.
    pub fn host(&self) -> &HostTier {
        &self.host
    }

    /// Whether `request` is still being served — what separates a sibling's
    /// prefix from a retained one (GitHub #190): the publisher of a retained
    /// prefix has finished.
    fn is_live(&self, request: RequestId) -> bool {
        self.requests
            .iter()
            .any(|r| r.id == request && r.state != RequestState::Done)
    }

    /// Requests currently in flight (Admitted + Prefilling + Running).
    fn in_flight(&self) -> usize {
        self.requests
            .iter()
            .filter(|r| r.state != RequestState::Done)
            .count()
    }

    /// The active protection (core-05), for telemetry / observability:
    /// `None` while no head is blocked (or once the protected head has
    /// been dealt).
    pub fn protection(&self) -> Option<&AdmissionProtection> {
        self.protection.as_ref()
    }

    /// The main-pool pages currently reserved by running requests
    /// (core-05; telemetry: the `kv_used` dimension).
    pub fn kv_used_pages(&self) -> u32 {
        self.kv_used_pages
    }

    /// The KV-RAM host tier (core-06): the bounded host-RAM budget for
    /// evicted (suspended) request snapshots (telemetry / tests).
    pub fn host_tier(&self) -> &HostTier {
        &self.host
    }

    /// The cumulative `sibling_prefix_reused_tok` counter (core-07, design
    /// §5): every prompt token a sibling skipped through a cached prefix.
    /// The telemetry writer (`server-02`) exposes this (and the per-request
    /// [`SchedEvent::PrefixReused`] events).
    pub fn sibling_prefix_reused_tok(&self) -> u64 {
        self.prefix.reused_tok()
    }

    /// The KV pages the sibling prefix cache pins in the pool (core-07):
    /// the shared prefixes' pages, charged to the pool exactly once (for
    /// every claimant) — a `1 main + N subagents` load pins one shared
    /// prefix, not `N` copies. Exposed for the pool-accounting invariant
    /// (tests) and telemetry.
    pub fn prefix_pinned_pages(&self) -> u32 {
        self.prefix.pinned_pages()
    }

    /// The shared sibling prefix `request` reuses (core-07), if any: the
    /// cached entry's id + the leading prompt tokens it skips.
    ///
    /// Observation only. What actually reaches the leaf is
    /// [`PrefillJob::shared_prefix`](crate::scheduler::PrefillJob), carried
    /// on the request's first job (P4-10, GitHub #126) — the backend cannot
    /// ask the scheduler for it, because it needs it at the moment it builds
    /// the sequence.
    pub fn shared_prefix_of(&self, request: RequestId) -> Option<(u64, u32)> {
        self.requests
            .iter()
            .find(|r| r.id == request)
            .and_then(|r| r.prefix_entry.map(|e| (e, r.shared_prefix_tokens)))
    }

    /// The main-pool pages available to the admission state machine
    /// (core-07): the configured capacity minus the pages the sibling
    /// prefix cache pins (core-07: the cache's pages are in the pool, so
    /// the machine's feasibility arithmetic runs against the remainder —
    /// consistent with [`Self::fits`]'s full accounting, which counts the
    /// cache's charge in `kv_used_pages`). A cache that pins too many
    /// pages cannot starve the machine: an entry drops as soon as its last
    /// claimant is gone (the cache never pins pages no live request needs).
    /// The KV-RAM device pool of retained prompt checkpoints (GitHub #186):
    /// what is retained, what it costs, what it has saved (telemetry / tests).
    pub fn checkpoint_pool(&self) -> &CheckpointPool {
        &self.checkpoints
    }

    /// The retained slots this scheduler hands out (GitHub #215): 0 with
    /// prompt reuse off.
    pub fn retained_slot_count(&self) -> u32 {
        self.retained.capacity()
    }

    /// The retained slots a prefix or checkpoint image holds right now.
    pub fn retained_slots_in_use(&self) -> u32 {
        self.retained.in_use()
    }

    /// The KV pages device checkpoints hold of their own (GitHub #215): one
    /// per checkpoint whose opener ends inside a page, charged in
    /// [`Self::kv_used_pages`].
    pub fn retained_tail_pages(&self) -> u32 {
        self.tail_pages.len() as u32
    }

    fn available_capacity(&self) -> AdmissionResources {
        AdmissionResources {
            lanes: self.capacity.lanes,
            kv_pages: self.capacity.kv_pages.saturating_sub(
                // GitHub #215: each device checkpoint's own tail page too.
                (self.prefix.pinned_pages() + self.tail_pages.len() as u32)
                    // GitHub #186: pages held *only* by retained checkpoints
                    // are not occupied as far as this arithmetic is
                    // concerned. They come back the instant a live request
                    // needs them (`Self::reclaim_retained`), so counting them
                    // here would let retained state open a protection — that
                    // is, make a live request *wait* for a bet that has
                    // already been given up. ADR 0029: retained state never
                    // delays or refuses an admission.
                    //
                    // Only the pages that would *actually* come back, though.
                    // A prefix a live claimant is also standing on is
                    // genuinely occupied, and promising it here would be the
                    // mirror of the mistake `reclaim_retained` used to make.
                    .saturating_sub(self.reclaimable_retained_pages()),
            ),
            backend_pages: self.capacity.backend_pages,
            resident_slots: self.capacity.resident_slots,
        }
    }

    /// The shared prefixes whose pages would come back if every retained
    /// entry standing on them were discarded: the ones no *live* request
    /// holds.
    ///
    /// A prefix's pages return at refcount zero, and its holders are live
    /// requests, retained checkpoints, and — since GitHub #188 — the
    /// prefix's own retention when it is a **retained prefix**. So "nothing
    /// live is standing on it" is exactly "its refcount is the number of
    /// retained holders on it" — which is why the two ledgers are compared
    /// rather than either being read alone.
    ///
    /// Both kinds are listed together because a prefix can be both: a request
    /// whose system block and generation opener land in the same KV page
    /// publishes one head and leaves a checkpoint standing on it. Giving up
    /// either alone would free nothing there, and only listing them together
    /// lets [`Self::reclaim_retained`] give up both.
    fn reclaimable_prefixes(&self) -> Vec<PrefixId> {
        let mut out = self.checkpoints.retained_prefixes();
        for p in self.prefix.reclaimable_retained() {
            if !out.contains(&p) {
                out.push(p);
            }
        }
        out.retain(|&p| self.prefix.refcount_of(p) == self.retained_holders_of(p));
        out
    }

    /// Everything holding `prefix` that is retained state rather than a live
    /// request: the checkpoints standing on it, plus its own retention when it
    /// is a retained prefix (GitHub #186, #188).
    fn retained_holders_of(&self, prefix: PrefixId) -> u32 {
        self.checkpoints.retained_holders(prefix) + u32::from(self.prefix.is_retained(prefix))
    }

    /// KV pages the first-victim path could actually give back right now:
    /// the reclaimable prefixes' own pages, and the tail page of every device
    /// checkpoint that can be given up (GitHub #215) — whatever stands on the
    /// prefix under it, since that page is the checkpoint's alone.
    fn reclaimable_retained_pages(&self) -> u32 {
        let claimed = self.claimed_device_checkpoints();
        let tails = self
            .checkpoints
            .entries()
            .iter()
            .filter(|e| e.tier == ReuseSource::Device && !claimed.contains(&e.id))
            .filter(|e| self.tail_pages.contains(&e.publisher))
            .count() as u32;
        tails
            + self
                .reclaimable_prefixes()
                .into_iter()
                .map(|p| self.prefix.pages_of(p))
                .sum::<u32>()
    }

    /// The device checkpoints an admitted request has claimed and not yet been
    /// built from (GitHub #215): its first chunk copies the image out of the
    /// checkpoint's retained slot, so neither the slot nor the tail page may go
    /// before then.
    fn claimed_device_checkpoints(&self) -> Vec<crate::checkpoint::CheckpointId> {
        self.requests
            .iter()
            .filter(|r| r.state == RequestState::Admitted)
            .filter(|r| r.reuse_source == Some(ReuseSource::Device))
            .filter_map(|r| r.checkpoint_entry)
            .collect()
    }

    /// The KV pages a checkpoint at `tokens` holds of its own: the page its
    /// opener ends inside, or none for an opener on a page boundary (GitHub
    /// #215).
    fn tail_pages_at(&self, tokens: u32) -> u32 {
        u32::from(tokens % self.config.kv_page_tokens != 0)
    }

    /// Drop the backend's handle on the shared prefix `publisher` published at
    /// `tokens`, and give back the retained slot its image held (GitHub #215):
    /// the one place a prefix image stops being reachable.
    fn release_prefix_handle(&mut self, publisher: RequestId, tokens: u32) {
        self.compute.release_prefix(publisher, tokens);
        self.retained
            .give_back(RetainedHolder::Prefix { publisher, tokens });
    }

    /// Drop the backend's handle on `publisher`'s checkpoint, wherever it
    /// lives, and give back what its device image held — a retained slot and
    /// its tail page — if it still held them (GitHub #215).
    fn release_checkpoint_handle(&mut self, publisher: RequestId) {
        self.compute.release_checkpoint(publisher);
        self.forget_checkpoint_image(publisher);
    }

    /// Give back what `publisher`'s device checkpoint image held, once the
    /// backend no longer has it: its retained slot and its tail page. A no-op
    /// for one that held neither.
    fn forget_checkpoint_image(&mut self, publisher: RequestId) {
        self.retained
            .give_back(RetainedHolder::Checkpoint { publisher });
        if let Some(pos) = self.tail_pages.iter().position(|&p| p == publisher) {
            self.tail_pages.swap_remove(pos);
            self.kv_used_pages = self.kv_used_pages.saturating_sub(1);
        }
    }

    /// A retained slot for `holder`, giving retained state up for one when
    /// none is free (GitHub #215, ADR 0030), or `None` when nothing can.
    fn take_retained_slot(
        &mut self,
        holder: RetainedHolder,
        events: &mut Vec<SchedEvent>,
    ) -> Option<u32> {
        loop {
            if let Some(slot) = self.retained.take(holder) {
                return Some(slot);
            }
            if self.retained.capacity() == 0 || !self.give_up_retained_for_slot(events) {
                return None;
            }
        }
    }

    /// Give up the lowest-ranked retained state that holds a slot, in ADR
    /// 0023's order: prompt checkpoints before retained prefixes, `Agent`
    /// before `Interactive`, least recently used — spilling to KV-RAM exactly
    /// as the page path does. Returns whether anything was given up.
    ///
    /// Two things are never given up, and both because their slot is still
    /// needed:
    ///
    /// - a checkpoint a request has claimed and not yet been built from — its
    ///   image is copied when that request's first chunk runs;
    /// - a prefix anything live stands on, or the chain under one. A retained
    ///   prefix is a candidate only while its retention is its sole holder, so
    ///   a claimant, a checkpoint or a chained link above it all keep it.
    ///
    /// Unlike the page path, a checkpoint on a prefix a live request holds is
    /// a candidate: its own slot comes back whatever the prefix does.
    fn give_up_retained_for_slot(&mut self, events: &mut Vec<SchedEvent>) -> bool {
        let claimed = self.claimed_device_checkpoints();
        if let Some(victim) = self.checkpoints.retained_slot_victim(&claimed) {
            if !self.spill_checkpoint(&victim, events) {
                let victim = self
                    .checkpoints
                    .discard(victim.id)
                    .expect("the chosen device victim is still retained");
                self.discard_checkpoint(victim, events);
            }
            return true;
        }
        let Some(prefix) = self.prefix.retained_slot_victim() else {
            return false;
        };
        self.spill_prefix(prefix, events);
        if !self.prefix.unretain(prefix) {
            return false;
        }
        self.release_prefix_claim(Some(prefix));
        true
    }

    /// Tell `request` its publish or capture was not taken (GitHub #215), with
    /// the slots held right then. With prompt reuse off and no slots nothing
    /// was asked for — its heads go unpublished by design — so nothing is said.
    fn skip_retained(&self, request: RequestId, skip: RetainedSkip, events: &mut Vec<SchedEvent>) {
        if self.config.prompt_reuse || self.retained.capacity() > 0 {
            events.push(SchedEvent::RetainedSlotSkipped {
                request,
                skip,
                in_use: self.retained.in_use(),
                capacity: self.retained.capacity(),
            });
        }
    }

    /// Report how many retained slots are held, when that changed since the
    /// last report (GitHub #215).
    fn report_retained_slots(&mut self, events: &mut Vec<SchedEvent>) {
        let in_use = self.retained.in_use();
        if self.reported_slots != in_use {
            self.reported_slots = in_use;
            events.push(SchedEvent::RetainedSlots {
                in_use,
                capacity: self.retained.capacity(),
            });
        }
    }

    /// Discard a retained checkpoint, wherever it lives: release the
    /// backend's handle on it, and let go of what it was holding — the shared
    /// pages under a device image (and the retained slot and tail page the
    /// image held, GitHub #215), or the KV-RAM budget a blob was charged to.
    ///
    /// A KV-RAM blob a request has chosen but not yet restored from outlives
    /// the discard: the entry is already out of the pool, so nothing new can
    /// choose it, and its bytes go when that restore lands or is abandoned
    /// ([`Self::release_kv_ram_claim`]).
    fn discard_checkpoint(&mut self, entry: CheckpointEntry, events: &mut Vec<SchedEvent>) {
        events.push(SchedEvent::RetainedState {
            operation: RetainedStateOperation::Discard,
            source: entry.tier,
            kind: RetainedKind::Checkpoint,
        });
        match entry.tier {
            ReuseSource::Device => {
                self.release_checkpoint_handle(entry.publisher);
                self.release_prefix_claim(Some(entry.prefix));
            }
            ReuseSource::KvRam => {
                if let Some(blob) = self.host.discard_retained(RetainedBlob::Checkpoint(entry.id)) {
                    self.compute.release_checkpoint(blob.publisher);
                }
            }
        }
    }

    /// Let go of `idx`'s claim on a KV-RAM blob, if it holds one. `restored`
    /// says whether the restore landed, which is what promotes the entry.
    fn release_kv_ram_claim(&mut self, idx: usize, restored: bool) {
        let Some(checkpoint) = self.requests[idx].kv_ram_claim.take() else {
            return;
        };
        let restored_at = restored.then(|| self.now());
        if let Some(blob) = self
            .host
            .release_retained_claim(RetainedBlob::Checkpoint(checkpoint), restored_at)
        {
            self.compute.release_checkpoint(blob.publisher);
        }
    }

    /// Give up KV-RAM blob `blob`, whichever kind it is (GitHub #190).
    fn discard_kv_ram_blob(&mut self, blob: RetainedBlob, events: &mut Vec<SchedEvent>) {
        match blob {
            RetainedBlob::Checkpoint(id) => {
                if let Some(entry) = self.checkpoints.discard(id) {
                    self.discard_checkpoint(entry, events);
                }
            }
            RetainedBlob::Prefix(_) => {
                if let Some(entry) = self.host.discard_retained(blob) {
                    self.forget_kv_ram_blob(entry, events);
                }
            }
        }
    }

    /// Make room in KV-RAM for a retained blob of `bytes`, captured by
    /// `owner` and last used at `used_at` (GitHub #190, #213). `true` when
    /// the tier can both hold it and place it.
    ///
    /// Two questions, one order. The budget's is "which entries make `bytes`
    /// free" ([`HostTier::plan_retained_room`]); the arena's is "is there a
    /// span long enough", which free bytes scattered between live blobs do
    /// not answer. So the same victim order keeps running past the byte
    /// plan, one entry at a time, until the backend says the blob fits — and
    /// stops, as the plan does, at the first entry ranking at or above this
    /// one. A bet never displaces something the tier values more, however
    /// badly it is fragmented, and a spill that cannot be placed simply does
    /// not happen.
    ///
    /// Asked before any copy, so a refusal costs no device work. Whatever it
    /// gave up on the way stays given up, exactly as a refused spill's
    /// planned victims always did.
    fn make_kv_ram_room_for_retained(
        &mut self,
        bytes: u64,
        owner: RequestClass,
        used_at: Instant,
        events: &mut Vec<SchedEvent>,
    ) -> bool {
        let now = self.now();
        let Some(victims) = self.host.plan_retained_room(bytes, owner, used_at, now) else {
            return false;
        };
        for victim in victims {
            self.discard_kv_ram_blob(victim, events);
        }
        while !self.compute.host_blob_fits(bytes) {
            let now = self.now();
            let Some(blob) = self.host.next_retained_victim_below(owner, used_at, now) else {
                return false;
            };
            // Out of the tier first, then freed at the backend: the pair
            // `make_host_room_for_bytes` uses for the same job, and what
            // makes this loop shrink the tier on every turn.
            let Some(entry) = self.host.discard_retained(blob) else {
                return false;
            };
            self.forget_kv_ram_blob(entry, events);
        }
        true
    }

    /// A KV-RAM entry already out of the tier's budget: drop what names it,
    /// free its blob, and report the discard.
    fn forget_kv_ram_blob(&mut self, entry: RetainedKvRamEntry, events: &mut Vec<SchedEvent>) {
        let kind = entry.blob.kind();
        match entry.blob {
            RetainedBlob::Checkpoint(id) => {
                self.checkpoints.discard(id);
                self.compute.release_checkpoint(entry.publisher);
            }
            RetainedBlob::Prefix(id) => {
                if let Some(spilled) = self.prefix.forget_spilled(id) {
                    self.compute
                        .discard_spilled_prefix(spilled.publisher, spilled.length_tokens);
                }
            }
        }
        events.push(SchedEvent::RetainedState {
            operation: RetainedStateOperation::Discard,
            source: ReuseSource::KvRam,
            kind,
        });
    }

    /// The retained prefix `entry`, which the first-victim path is giving up,
    /// goes to KV-RAM if the budget takes it (GitHub #190) — before the caller
    /// releases it, while its pages are still on the device to copy. A prefix
    /// that already has a blob there (it was brought back from one) copies
    /// nothing: the blob just ranks by the prefix's latest use.
    fn spill_prefix(&mut self, entry: PrefixId, events: &mut Vec<SchedEvent>) {
        let Some((publisher, tokens, retention)) = self
            .prefix
            .entry(entry)
            .and_then(|e| e.retained.map(|r| (e.publisher, e.length_tokens, r)))
        else {
            return;
        };
        if let Some(copy) = self.prefix.spilled_copy_of(entry) {
            self.host
                .refresh_retained(RetainedBlob::Prefix(copy), retention.used_at);
            return;
        }
        let spilled = self.checkpoints.tiers().below(ReuseSource::Device) == Some(ReuseSource::KvRam)
            && self.write_prefix_blob(entry, publisher, tokens, retention, events);
        if !spilled {
            events.push(SchedEvent::RetainedState {
                operation: RetainedStateOperation::Discard,
                source: ReuseSource::Device,
                kind: RetainedKind::Prefix,
            });
        }
    }

    fn write_prefix_blob(
        &mut self,
        entry: PrefixId,
        publisher: RequestId,
        tokens: u32,
        retention: Retention,
        events: &mut Vec<SchedEvent>,
    ) -> bool {
        let Ok(bytes) = self.compute.prefix_snapshot_size(publisher, tokens) else {
            return false;
        };
        if !self.make_kv_ram_room_for_retained(bytes, retention.class, retention.used_at, events) {
            return false;
        }
        let Ok(bytes) = self.compute.spill_prefix(publisher, tokens) else {
            return false;
        };
        let Some(id) = self.prefix.record_spill(entry) else {
            self.compute.discard_spilled_prefix(publisher, tokens);
            return false;
        };
        let blob =
            RetainedKvRamEntry::new(RetainedBlob::Prefix(id), publisher, retention.class, bytes, retention.used_at);
        if self.host.capture_retained(blob).is_err() {
            self.prefix.forget_spilled(id);
            self.compute.discard_spilled_prefix(publisher, tokens);
            return false;
        }
        events.push(SchedEvent::RetainedState {
            operation: RetainedStateOperation::Spill,
            source: ReuseSource::KvRam,
            kind: RetainedKind::Prefix,
        });
        true
    }

    /// Request `i`'s prompt keyed at every length retained state and shared
    /// prefixes hold (GitHub #193): the one forward pass a claimant's lookups
    /// share, rather than one per pool.
    ///
    /// Its tokens and media — minus the last token for a multimodal request.
    /// A claim reaching the prompt's very end leaves nothing to prefill, and
    /// for a multimodal request the prefill is the only thing that hands the
    /// leaf its `rope_delta`: a claimant is cloned from a publisher's mutable
    /// state and pages, or from a checkpoint's image, and none of them carries
    /// it (`step.cu` sets `seq->rope_delta` from the span options). Without one
    /// tail token its every decode round would rotate at `position + 0`. A
    /// checkpoint is held to it too: its own capture leaves a tail, but a
    /// *claimant's* prompt can end exactly at another request's opener.
    fn reuse_keys(&self, i: usize) -> PromptKeys {
        let input = &self.requests[i].input;
        let media = media_keys(input);
        let prompt = PromptContent::new(&input.tokens, &media);
        // GitHub #238: trimming the *walk* rather than the claim is what
        // makes a decision's reuse limit airtight — no checkpoint, prefix or
        // spilled entry can match past a length the keys were never computed
        // for, so no claim-taking path has to know about decisions at all.
        // `reuse_reach` is where both reasons for stopping a token short
        // now live.
        let reach = input.reuse_reach() as u32;
        prompt
            .head(reach)
            .keys_for(self.checkpoints.match_lengths().chain(self.prefix.match_lengths()))
    }

    /// Bring spilled prefix `id` back onto the device as a retained prefix
    /// (GitHub #190, owner decision): one restore across the bus, after which
    /// the request that matched it — and every later member of its burst —
    /// claims it in place like any sibling. The blob stays in KV-RAM.
    ///
    /// Its pages are taken back from retained state first, like any live
    /// request's, and so is a retained slot for its image (GitHub #215); it
    /// never evicts live work, and it needs a free resident slot for the
    /// moment the restore stands a sequence up. Returns whether the prefix is
    /// on the device now.
    fn return_prefix(&mut self, id: SpilledPrefixId, events: &mut Vec<SchedEvent>) -> bool {
        let Some(spilled) = self.prefix.spilled(id).cloned() else {
            return false;
        };
        if self.resident_slots_used >= self.capacity.resident_slots {
            return false;
        }
        let blob = RetainedBlob::Prefix(id);
        // Held while retained state makes room, so the room is never made out
        // of this blob.
        if !self.host.claim_retained(blob) {
            return false;
        }
        let holder = RetainedHolder::Prefix {
            publisher: spilled.publisher,
            tokens: spilled.length_tokens,
        };
        let pages = spilled.length_tokens / self.config.kv_page_tokens;
        // Pages first: a slot is taken — and retained state given up for
        // one — only once the prefix is known to fit.
        let fits = self
            .reclaim_retained_until(|s| s.kv_used_pages + pages <= s.capacity.kv_pages, events)
            && self.take_retained_slot(holder, events).is_some_and(|slot| {
                self.compute
                    .restore_prefix(spilled.publisher, spilled.length_tokens, slot)
                    .is_ok()
            });
        let now = self.now();
        if let Some(entry) = self.host.release_retained_claim(blob, fits.then_some(now)) {
            self.forget_kv_ram_blob(entry, events);
        }
        if !fits {
            self.retained.give_back(holder);
            return false;
        }
        let Some((entry, own_pages)) = self.prefix.register_returned(&spilled) else {
            self.release_prefix_handle(spilled.publisher, spilled.length_tokens);
            return false;
        };
        // No live request warmed these pages, so nothing else charges them:
        // the entry does, as it does a publisher's once the publisher is gone.
        self.kv_used_pages += own_pages;
        let retention = Retention {
            at: self.tick,
            class: spilled.class,
            used_at: now,
        };
        self.prefix.retain_published(entry, retention);
        // `register` counted a registrant; nobody is one.
        self.prefix.release(entry);
        self.prefix.record_return(id, entry);
        for operation in [RetainedStateOperation::Hit, RetainedStateOperation::Restore] {
            events.push(SchedEvent::RetainedState {
                operation,
                source: ReuseSource::KvRam,
                kind: RetainedKind::Prefix,
            });
        }
        true
    }

    /// Move a device checkpoint the first-victim path is giving up into KV-RAM
    /// (GitHub #190), or report that it cannot go there and has to be
    /// discarded.
    ///
    /// Room is made before a byte moves, and only out of retained entries
    /// ranking below this one ([`Self::make_kv_ram_room_for_retained`]) — in
    /// the tier's byte budget and in the arena the blob has to be placed in.
    /// When those entries are not enough for either, the spill does not
    /// happen.
    ///
    /// A spill can still cost another entry for nothing: the room may be
    /// made and then the leaf fail, or the entries the newcomer outranks may
    /// free the bytes and still leave no span long enough. Both leave what
    /// was given up given up, and neither leaves this one half-spilled.
    fn spill_checkpoint(&mut self, entry: &CheckpointEntry, events: &mut Vec<SchedEvent>) -> bool {
        if self.checkpoints.tiers().below(ReuseSource::Device) != Some(ReuseSource::KvRam) {
            return false;
        }
        let Ok(bytes) = self.compute.checkpoint_snapshot_size(entry.publisher) else {
            return false;
        };
        if !self.make_kv_ram_room_for_retained(bytes, entry.class, entry.used_at, events) {
            return false;
        }
        let Ok(bytes) = self.compute.spill_checkpoint(entry.publisher) else {
            return false;
        };
        // The backend released the device image with the spill: its retained
        // slot and its tail page come back now, whatever happens below. On a
        // failure below the caller's discard forgets them again, which is a
        // no-op — nothing is held twice.
        self.forget_checkpoint_image(entry.publisher);
        let blob = RetainedKvRamEntry::new(
            RetainedBlob::Checkpoint(entry.id),
            entry.publisher,
            entry.class,
            bytes,
            entry.used_at,
        );
        if self.host.capture_retained(blob).is_err()
            || self
                .checkpoints
                .move_to_tier(entry.id, ReuseSource::KvRam)
                .is_err()
        {
            // The leaf wrote a blob of a size other than the one it priced.
            // Its device image is already gone, so the caller's discard is
            // what releases the blob as well.
            return false;
        }
        self.release_prefix_claim(Some(entry.prefix));
        events.push(SchedEvent::RetainedState {
            operation: RetainedStateOperation::Spill,
            source: ReuseSource::KvRam,
            kind: RetainedKind::Checkpoint,
        });
        true
    }

    /// Record the prompt checkpoint the backend just captured for request
    /// `idx` at `at` tokens (GitHub #186, ADR 0029).
    ///
    /// The entry takes a holder's reference on the shared prefix under the
    /// opener, which is what lets the pages outlive the request that warmed
    /// them without ever being charged to the pool twice. Retention starts
    /// here, at the capture, rather than at the request's end: a checkpoint
    /// is immutable history plus an immutable image, so a claimant may stand
    /// on it while its publisher is still decoding — and a request cancelled
    /// after this point keeps its checkpoint without a single extra rule
    /// (spec §Cancellation).
    fn retain_checkpoint(&mut self, idx: usize, at: u32, events: &mut Vec<SchedEvent>) {
        let (publisher, prefix, key, gdn, claimed, turn_opening, class) = {
            let r = &self.requests[idx];
            let Some(prefix) = r.prefix_entry else {
                // The leaf published a prefix and captured against it, and
                // *this* cache then declined to register the head — another
                // request in the same batch took it. Nothing here can name
                // the image, so it is released rather than left to outlive
                // every ledger that knows it exists.
                let publisher = r.id;
                self.release_checkpoint_handle(publisher);
                return;
            };
            let media = media_keys(&r.input);
            (
                r.id,
                prefix,
                PromptContent::new(&r.input.tokens, &media).key_at(at),
                r.gdn.clone(),
                r.checkpoint_entry,
                r.opens_a_turn(),
                r.class,
            )
        };
        let capture = CheckpointCapture {
            publisher,
            key,
            tokens: at,
            prefix,
            // GitHub #187: the whole head below the opener, the prefix's chain
            // included — what a claimant of this entry shares, which is not
            // the same as what any one link would give back.
            pages: self.prefix.total_pages_of(prefix),
            gdn,
            identity: self.checkpoints.identity(),
            tier: ReuseSource::Device,
            claimed,
            turn_opening,
            class,
            captured_at: self.now(),
        };
        match self.checkpoints.retain(capture, self.tick) {
            Ok(retained) => {
                self.requests[idx].checkpoint_captured = true;
                self.prefix.retain(prefix);
                // GitHub #215: the page the opener ends inside is a KV page
                // of the pool now, held for as long as the device image is.
                if self.tail_pages_at(at) > 0 {
                    self.tail_pages.push(publisher);
                    self.kv_used_pages += 1;
                }
                // GitHub #187 — the conversation's superseded checkpoints,
                // released here and now rather than left to the LRU. Their
                // images and their holds on the pages below them are this
                // scheduler's to give back: the pool moved the ledger, and
                // nothing else in the engine knows the device still has them.
                for entry in retained.superseded {
                    self.discard_checkpoint(entry, events);
                }
            }
            // Unreachable on this path — the capture carries the pool's own
            // identity and the device tier, because this load's backend is
            // what produced it — but were it refused, the image the backend
            // took is real and nothing could reach it, so it is released
            // rather than left to outlive every ledger that knows it exists.
            Err(_) => self.release_checkpoint_handle(publisher),
        }
    }

    /// Take retained pages back until `idx` can materialize, or until nothing
    /// is retained (GitHub #186, ADR 0029/0023 as amended).
    ///
    /// This is the **first victim** rule, and it runs *before* the eviction
    /// machinery is asked for anything: a bet is given up before any certain
    /// work is disturbed, so a retained entry can never cause a live request
    /// to be refused, evicted or made to wait. Returns whether `idx` fits now.
    fn reclaim_retained(&mut self, idx: usize, events: &mut Vec<SchedEvent>) -> bool {
        self.reclaim_retained_until(|s| s.fits_for_materialization(&s.requests[idx]), events)
    }

    /// [`Self::reclaim_retained`] until `fits` holds (GitHub #190: a prefix
    /// coming back from KV-RAM needs pages without being a request).
    fn reclaim_retained_until(
        &mut self,
        fits: impl Fn(&Self) -> bool,
        events: &mut Vec<SchedEvent>,
    ) -> bool {
        loop {
            if fits(self) {
                return true;
            }
            // Only entries whose pages would genuinely come back are given
            // up. In the steady state this feature exists for, turn N's
            // prefix is held by its retained entry *and* by the live turn
            // N+1 standing on it — discarding the entry there frees not one
            // page, and a loop that did not know that would give up every
            // checkpoint in the pool and still not fit. When nothing
            // qualifies the answer is "retained state cannot help", and the
            // caller goes to the eviction machinery with the pool intact.
            //
            // GitHub #215: except a checkpoint's own tail page. That page is
            // the checkpoint's alone, so giving the checkpoint up returns it
            // even while a live request stands on the prefix below — unless a
            // request is about to be built from that checkpoint.
            let reclaimable = self.reclaimable_prefixes();
            let claimed = self.claimed_device_checkpoints();
            let tail_pages = &self.tail_pages;
            if let Some(victim) = self.checkpoints.victim_where(ReuseSource::Device, |e| {
                reclaimable.contains(&e.prefix)
                    || (tail_pages.contains(&e.publisher) && !claimed.contains(&e.id))
            }) {
                if !self.spill_checkpoint(&victim, events) {
                    let victim = self
                        .checkpoints
                        .discard(victim.id)
                        .expect("the peeked device victim is still retained");
                    self.discard_checkpoint(victim, events);
                }
                continue;
            }
            // GitHub #188: then the retained prefixes. Checkpoints go first,
            // and not by accident. A checkpoint is the narrow bet — one
            // conversation's next turn — while a retained prefix is the wide
            // one: every future request that opens with that system and tools
            // block, whether or not it belongs to any conversation seen so
            // far. Between two bets, the narrower one is given up first. It
            // also frees strictly more, since a checkpoint's prefix reaches
            // past the block its conversation opened with; and where a prefix
            // carries both, this ordering is what makes the loop able to give
            // up the checkpoints standing on it *and then* the retention,
            // which is the only sequence that returns those pages at all.
            let Some(prefix) = self.prefix.lru_retained() else {
                return false;
            };
            // GitHub #190: to KV-RAM first, while its pages are still there.
            self.spill_prefix(prefix, events);
            if !self.prefix.unretain(prefix) {
                return false;
            }
            self.release_prefix_claim(Some(prefix));
        }
    }

    /// Release one reference to the shared prefix `entry` (core-07), if
    /// the claimant still holds it: when the last claimant releases, the
    /// entry drops and its pages return to the pool (the charge that was
    /// taken once at registration is now returned). `None` while other
    /// claimants still pin the entry (nothing to release yet).
    fn release_prefix_claim(&mut self, entry: Option<PrefixId>) {
        let Some(entry) = entry else {
            return;
        };
        // GitHub #187: a chained entry holds its parent's reference, so one
        // release can drop a whole run of the chain — every link that drops
        // returns its own pages and its own backend handle.
        for (freed, publisher, tokens) in self.prefix.release(entry) {
            self.kv_used_pages = self.kv_used_pages.saturating_sub(freed);
            // P4-10 (GitHub #126): the entry is gone from this cache, so the
            // backend's own handle on the leaf's prefix goes too. The leaf's
            // pages come back when its last *sequence* holder is released,
            // which is why this is a handle drop and not a free — but its
            // image's retained slot comes back now (GitHub #215): nothing can
            // claim the prefix without the handle.
            self.release_prefix_handle(publisher, tokens);
        }
    }

    /// The last hard compute error the most recent advance reported, if
    /// any.
    ///
    /// A failed step is not swallowed: a failed prefill is retryable (until
    /// its bounded attempt limit), while a decode failure emits `Done(Error)`
    /// and releases every affected lane. A successful later advance clears
    /// this field; callers poll it to surface the fault.
    pub fn last_error(&self) -> Option<&ComputeError> {
        self.last_error.as_ref()
    }

    // ── core-05: the admission state machine ─────────────────────────────

    /// `r`'s resource need as far as the *lane-deal* machinery is
    /// concerned: `kv_pages` and `resident_slots` zeroed out (P4-07, GitHub
    /// #125). A `Prefilling`-complete request headed into this machinery
    /// is, by construction, already device-resident — its whole KV/slot
    /// reservation was charged once, at first materialization, and stays
    /// charged (in `self.kv_used_pages` / `self.resident_slots_used`)
    /// until it completes or is evicted; that invariant is what guarantees
    /// those two dimensions never overflow capacity in the first place; see
    /// [`Self::fits_for_materialization`]. Passing `r.resources` unchanged
    /// into `admission.rs`'s protection/backfill arithmetic — which
    /// recomputes its own totals from scratch over the active set rather
    /// than reading the scheduler's running counters — would double-count
    /// that already-settled charge and could reject (or, worse, silently
    /// misjudge) a lane deal on dimensions that were never actually at
    /// stake in it. Only the lane itself (and the still-dormant
    /// speculative-backend reservation) is ever new here.
    fn deal_only(r: &Request) -> AdmissionResources {
        AdmissionResources {
            lanes: r.resources.lanes,
            kv_pages: 0,
            backend_pages: r.resources.backend_pages,
            resident_slots: 0,
        }
    }

    /// Whether `r` can be dealt a decode lane right now (see
    /// [`Self::deal_only`] for why only the lane is new here).
    fn fits(&self, r: &Request) -> bool {
        let running = AdmissionResources {
            lanes: N_DECODE_LANES as u32 - self.free_lanes.len() as u32,
            kv_pages: self.kv_used_pages,
            backend_pages: 0,
            resident_slots: self.resident_slots_used,
        };
        admission_resources_fit(&running.add(&Self::deal_only(r)), &self.capacity)
    }

    /// Whether `r` (still `Admitted`, not yet device-resident) can
    /// materialize a leaf sequence right now: enough resident-slot and
    /// main-pool headroom for its *whole* reservation (P4-07, GitHub #125).
    /// This is the check the prefill-dispatch phase makes *before* sending
    /// a fresh candidate's first chunk — the leaf reserves `r`'s KV pages
    /// and a resident slot at that exact moment, so admission has to agree
    /// beforehand or the two sides' views of capacity drift (a real
    /// materialization failure would otherwise surface as a raw leaf
    /// allocation error instead of an admission refusal the tier can act
    /// on).
    fn fits_for_materialization(&self, r: &Request) -> bool {
        let materialized = AdmissionResources {
            lanes: 0,
            kv_pages: self.kv_used_pages,
            backend_pages: 0,
            resident_slots: self.resident_slots_used,
        };
        let additional = AdmissionResources {
            lanes: 0,
            kv_pages: r.resources.kv_pages,
            backend_pages: 0,
            resident_slots: 1,
        };
        admission_resources_fit(&materialized.add(&additional), &self.capacity)
    }

    /// Charge `r`'s resident-slot and KV-page reservation (P4-07, GitHub
    /// #125): called exactly once per leaf materialization — a fresh
    /// request's first successful chunk, or a continuing request's first
    /// successful chunk *after* a prior chunk's failure released it (see
    /// [`Self::unmaterialize`]) — mirroring `RuntimeCompute` holding a live
    /// `LiveSequence` for it. A no-op if `r` is already materialized (never
    /// double-charged).
    fn materialize(&mut self, idx: usize) {
        if self.requests[idx].resident {
            return;
        }
        self.requests[idx].resident = true;
        self.resident_slots_used += 1;
        self.kv_used_pages += self.requests[idx].resources.kv_pages;
    }

    /// Release `r`'s resident-slot and KV-page charge (the inverse of
    /// [`Self::materialize`]): completion, cancellation, eviction, or a
    /// failed prefill batch unwinding every sequence it touched
    /// (`RuntimeCompute::prefill_step`'s own documented behavior — a
    /// failure releases every job's sequence, continuing ones included, not
    /// only freshly-allocated ones). A no-op if `r` was never materialized.
    fn unmaterialize(&mut self, idx: usize) {
        if !self.requests[idx].resident {
            return;
        }
        self.requests[idx].resident = false;
        self.resident_slots_used = self.resident_slots_used.saturating_sub(1);
        self.kv_used_pages = self
            .kv_used_pages
            .saturating_sub(self.requests[idx].resources.kv_pages);
    }

    /// Deal a lane to `idx` (a `Prefilling`, prefill-complete request).
    /// Charges only the lane (P4-07, GitHub #125: `idx`'s KV pages and
    /// resident slot were already charged at first materialization —
    /// dealing a lane adds no new device state, only decode-round
    /// eligibility). Returns `false` (deals nothing) when no lane is free.
    fn try_admit(
        &mut self,
        idx: usize,
        backfill: BackfillClass,
        events: &mut Vec<SchedEvent>,
    ) -> bool {
        // `free_lanes` is kept sorted descending: pop the smallest free
        // lane for a deterministic deal order.
        let lane = match self.free_lanes.pop() {
            Some(l) => l,
            None => return false,
        };
        let request_id = self.requests[idx].id;
        // GitHub #81 / ADR 0012: the admission-decision span — entry/exit
        // of the state machine's lane deal for this one request, tagged
        // with its own `request_id` (the field `trace_id` is derived
        // from).
        let _span = tracing::info_span!("ignis.admission", request_id).entered();
        if !self.requests[idx].assign_lane(lane) {
            // Not in `Prefilling` (a bug: the machine only deals
            // `Prefilling` requests) — give the lane back, deal nothing.
            self.free_lanes.push(lane);
            return false;
        }
        let backfill_epoch = if backfill == BackfillClass::None {
            0
        } else {
            self.protection.as_ref().map(|p| p.epoch_id).unwrap_or(0)
        };
        self.requests[idx].backfill_class = backfill;
        self.requests[idx].backfill_epoch = backfill_epoch;
        events.push(SchedEvent::Admitted {
            request: request_id,
            lane,
            backfill,
        });
        true
    }

    /// The active set for the protection arithmetic (reference:
    /// `active_admission_set`): every running request with its resources,
    /// remaining service work, and the protection epoch / class it was
    /// admitted under.
    fn active_admission_set(&self) -> Vec<ActiveAdmissionSnapshot> {
        self.requests
            .iter()
            .filter(|r| r.state == RequestState::Running && r.remaining_work > 0)
            .map(|r| ActiveAdmissionSnapshot {
                request_id: r.id,
                // P4-07, GitHub #125: see `Self::deal_only` — a `Running`
                // request's kv_pages/resident_slots are already-settled
                // charges, not something this "what if" recomputation
                // should weigh again.
                resources: Self::deal_only(r),
                remaining_work_quanta: r.remaining_work,
                backfill_epoch: r.backfill_epoch,
                backfill_class: r.backfill_class,
            })
            .collect()
    }

    /// Phase 2 (core-05): the admission state machine drives the lane
    /// deal. The queue is the **completed** `Prefilling` set (P3-01, ADR
    /// 0018: `Prefilling` is durable and may carry only partial progress —
    /// a request that has not sent its whole prompt yet is not a
    /// candidate; a sequence that has not finished prefilling can never be
    /// decoded) in (class priority, FIFO by id) order; the head is dealt
    /// when it fits, and — when the head is blocked by the active set —
    /// the machine opens / maintains the protection and classifies
    /// backfills (persistent / temporal) on the lanes the donors will
    /// free.
    fn run_admission(&mut self, events: &mut Vec<SchedEvent>) {
        let mut queue: Vec<usize> = self
            .requests
            .iter()
            .enumerate()
            .filter(|&(_, r)| r.state == RequestState::Prefilling && r.prefill_complete())
            .map(|(i, _)| i)
            .collect();
        queue.sort_by_key(|&i| (self.requests[i].class, self.requests[i].id));
        if queue.is_empty() {
            return;
        }
        let head = queue[0];
        let active = self.active_admission_set();

        if self.fits(&self.requests[head]) {
            // The head fits: a plain deal (class priority + FIFO).
            self.try_admit(head, BackfillClass::None, events);
            // `clear_protection_if_head`: when the protected head itself is
            // the dealt queue-head, its protection is cleared (the next
            // blocked head opens a fresh epoch). If a higher-priority
            // request overtakes and the protected head is dealt in the
            // trailing loop instead, the stale protection self-heals via
            // `mark_done` when that request completes.
            let protected_head = self.protection.as_ref().map(|p| p.head_request_id);
            if protected_head == Some(self.requests[head].id) {
                self.protection = None;
            }
            // The remaining free lanes go to the queue in order: no
            // overtaking — a candidate that does not fit stays queued
            // (the next advance re-runs the machine; a still-blocked
            // head opens a fresh protection and the backfill path).
            for &c in &queue[1..] {
                if self.fits(&self.requests[c]) {
                    self.try_admit(c, BackfillClass::None, events);
                } else {
                    break;
                }
            }
        } else {
            // The head is blocked by the active set: the protection
            // regime (protection / backfill class / temporal credit /
            // frontier distance — `admission.rs`, ADR 0004).
            if self.protection.is_none() {
                let available = self.available_capacity();
                let protection = match make_admission_protection(
                    self.protection_epoch,
                    self.requests[head].id,
                    Self::deal_only(&self.requests[head]),
                    &active,
                    &available,
                ) {
                    Ok(p) => p,
                    Err(e) => {
                        // A broken invariant here is a caller bug (an
                        // inconsistent active-set snapshot). Debug builds
                        // trap on it; release builds skip this step's
                        // backfill path — the step is retried on the next
                        // advance and no invalid state is written, so the
                        // machine stays consistent.
                        debug_assert!(false, "invalid protection frontier: {e}");
                        return;
                    }
                };
                events.push(SchedEvent::Protected {
                    epoch: protection.epoch_id,
                    head: self.requests[head].id,
                    donors: protection.donor_ids.clone(),
                });
                self.protection_epoch += 1;
                self.protection = Some(protection);
            }
            // Drain phase: once the head fits without the current-epoch
            // temporal borrowers, no new backfills are admitted — the
            // machine waits for the remaining donors / backfills, then
            // deals the head (the next advance's head-fits branch).
            let available = self.available_capacity();
            if protected_head_safe_without_temporal(
                self.protection.as_ref().unwrap(),
                &active,
                &available,
            ) {
                self.protection.as_mut().unwrap().phase = ProtectionPhase::Drain;
            }
            // core-06: try to free a lane by evicting a retained lane to
            // the host tier (the overflow path). When a victim is evicted
            // and the head now fits, deal the head normally (clearing the
            // protection) and skip the backfill classification for this
            // step.
            if self.try_evict_for_head(head, events) {
                self.try_admit(head, BackfillClass::None, events);
                self.protection = None; // the head is dealt: clear it
                return;
            }
            if self.protection.as_ref().unwrap().phase == ProtectionPhase::Open {
                let frontier =
                    protection_frontier_distance(self.protection.as_ref().unwrap(), &active);
                for &c in &queue[1..] {
                    if !self.fits(&self.requests[c]) {
                        continue;
                    }
                    let p = self.protection.as_ref().unwrap();
                    if persistent_backfill_is_safe(
                        p,
                        &active,
                        &Self::deal_only(&self.requests[c]),
                        &self.available_capacity(),
                    ) {
                        self.try_admit(c, BackfillClass::Persistent, events);
                    } else if self.requests[c].remaining_work <= frontier
                        && self.requests[c].remaining_work <= p.temporal_credit
                    {
                        // Credit decay: a temporal backfill spends its
                        // own service work out of the frozen credit.
                        self.protection.as_mut().unwrap().temporal_credit -=
                            self.requests[c].remaining_work;
                        self.try_admit(c, BackfillClass::Temporal, events);
                    }
                }
            }
        }
    }

    /// Release request `idx`'s resources regardless of its lifecycle state
    /// (its lane, if any, its KV reservation, and its shared-prefix claim)
    /// and abort it ([`Request::abort`]). Returns its id and its generated
    /// token count (for the caller's own event, if it emits one). Shared by
    /// [`Self::mark_done`] (a normal completion) and [`Self::cancel`]'s
    /// sweep (P3-01, ADR 0018: cancel is abort, not suspend — no
    /// suspend/resume primitive exists, so cancelling releases exactly what
    /// completing would).
    fn release_request(&mut self, idx: usize) -> (RequestId, u32) {
        let (lane, request_id, tokens, prefix_entry) = {
            let r = &self.requests[idx];
            (r.lane, r.id, r.tokens, r.prefix_entry)
        };
        self.requests[idx].abort();
        self.compute.release(request_id);
        self.release_kv_ram_claim(idx, false);
        if let Some(lane) = lane {
            self.free_lanes.push(lane);
        }
        // P4-07, GitHub #125: the resident-slot + KV-page charge is gated
        // on `resident`, not on holding a lane — a request cancelled while
        // still `Prefilling` (materialized, no lane yet) still held real
        // device state to release; one cancelled while still `Admitted`
        // (never materialized) released nothing and charges nothing back.
        self.unmaterialize(idx);
        // core-07: release the request's shared-prefix claim (its
        // completion frees its reference to the shared pages; the entry
        // drops — and its pages return to the pool — when the last
        // claimant releases).
        self.release_prefix_claim(prefix_entry);
        // A protection exists to let its head in: if the protected head
        // itself completes or is cancelled (rather than being dealt), its
        // protection is stale — the next blocked head opens a fresh epoch.
        let protected_head = self.protection.as_ref().map(|p| p.head_request_id);
        if protected_head == Some(request_id) {
            self.protection = None;
        }
        (request_id, tokens)
    }

    /// Why a request whose generation budget is spent stopped.
    ///
    /// [`FinishReason::Length`] for every ordinary request: it had more to
    /// say and the reservation cap cut it off, which is what `length` means
    /// on the OpenAI surface. [`FinishReason::Stop`] for a **constrained decode**
    /// (GitHub #242), because its budget *is* its schedule — a number that
    /// has read its last digit is finished, not truncated, and reporting
    /// `length` for it would tell a caller their answer may be incomplete
    /// every single time.
    fn budget_spent_reason(&self, idx: usize) -> FinishReason {
        match self.requests[idx].input.is_constrained() {
            true => FinishReason::Stop,
            false => FinishReason::Length,
        }
    }

    /// Complete request `idx` (its lane and KV reservation are released).
    /// `reason` is why it stopped — carried into the emitted
    /// [`SchedEvent::Done`] for the server's `finish_reason` (GitHub #61).
    fn mark_done(&mut self, idx: usize, events: &mut Vec<SchedEvent>, reason: FinishReason) {
        // GitHub #81 / ADR 0012: the completion span — the last stage in
        // this request's lifecycle, tagged with its own `request_id`
        // (read up front: `release_request` below also returns it, but by
        // then the request has already been released).
        let _span =
            tracing::info_span!("ignis.completion", request_id = self.requests[idx].id).entered();
        let spec = self.requests[idx].spec;
        // GitHub #238: a decision's whole answer, moved out rather than
        // cloned — the released request stays in `self.requests` until it is
        // reaped, and an answer left behind on it would be a second copy of
        // the only thing this event exists to carry.
        let readout = self.requests[idx].readout.take();
        // GitHub #260: an attention readout's scores, moved out likewise.
        let attention = self.requests[idx].attention.take();
        // GitHub #242: a run's answer, moved out for the same reason —
        // and `None` rather than an empty vector for a request that never
        // had one, so a reader can tell "this was not a constrained decode" from "this
        // program emitted nothing".
        let drawn = match self.requests[idx].input.is_constrained() {
            true => Some(std::mem::take(&mut self.requests[idx].drawn)),
            false => None,
        };
        let (request_id, tokens) = self.release_request(idx);
        events.push(SchedEvent::Done {
            request: request_id,
            tokens,
            reason,
            spec,
            readout,
            attention,
            drawn,
        });
    }

    /// Cancel `request` (P3-01, ADR 0018): abort, not suspend. Marks the
    /// request; the *next* `advance()` releases it — before running any of
    /// this tick's phases — releasing its lane, KV pages, GDN slot and
    /// shared-prefix claim, and it is never dealt
    /// another chunk or decode round. Because every compute call is
    /// synchronous, whatever chunk was already in flight when this is
    /// called has, by construction, already returned: there is nothing to
    /// interrupt, so "finishes the in-flight chunk, then aborts" holds
    /// without any extra bookkeeping. No `SchedEvent` is emitted (a
    /// cancelled request has no listener left to tell). Returns `false`
    /// when `request` is unknown or already `Done`.
    pub fn cancel(&mut self, request: RequestId) -> bool {
        match self
            .requests
            .iter()
            .position(|r| r.id == request && r.state != RequestState::Done)
        {
            Some(idx) => {
                self.requests[idx].cancelled = true;
                true
            }
            None => false,
        }
    }

    /// The request's lifecycle state (test / telemetry observability).
    /// `None` when `request` is unknown.
    pub fn request_state(&self, request: RequestId) -> Option<RequestState> {
        self.requests
            .iter()
            .find(|r| r.id == request)
            .map(|r| r.state)
    }

    /// The request's prefill progress (P3-01, ADR 0018): prompt tokens
    /// already sent to the compute backend. Durable — this can hold a
    /// partial value across many `advance()` calls while the request sits
    /// in `Prefilling`. `None` when `request` is unknown.
    pub fn prefill_progress(&self, request: RequestId) -> Option<u32> {
        self.requests
            .iter()
            .find(|r| r.id == request)
            .map(|r| r.prefill_progress)
    }

    /// The request's GDN recurrent-state position (core-02, P3-01): a
    /// single absolute counter over the whole sequence (prompt tokens then
    /// generated tokens) — chunk boundaries during prefill and per-token
    /// boundaries during decode both checkpoint into the same counter, so
    /// it never runs backwards. `None` when `request` is unknown.
    pub fn gdn_position(&self, request: RequestId) -> Option<usize> {
        self.requests
            .iter()
            .find(|r| r.id == request)
            .map(|r| r.gdn.position())
    }

    // ── core-06: the KV-RAM host tier ───────────────────────────────────

    /// The retained-lane candidates for victim selection (core-06): every
    /// running request's lane, excluding the protection's donors (donors
    /// are never evicted while the protection is open), reserved lanes (a
    /// lane whose shared prefix is claimed by an earlier-queued interactive
    /// request — core-07). A request holding a shared prefix is a candidate
    /// like any other since GitHub #190: its snapshot materializes the shared
    /// pages, so it resumes as a sequence that owns its whole history.
    fn retained_lane_candidates(&self) -> Vec<RetainedLaneCandidate> {
        let donors: std::collections::HashSet<RequestId> = self
            .protection
            .as_ref()
            .map(|p| p.donor_ids.iter().copied().collect())
            .unwrap_or_default();
        self.requests
            .iter()
            .filter(|r| r.state == RequestState::Running)
            .filter(|r| !donors.contains(&r.id))
            .map(|r| RetainedLaneCandidate {
                lane: r.lane.expect("a running request holds a lane"),
                owner: r.class,
                use_tick: self.tick,
                reserved_for_earlier_interactive: self.reserved_for_earlier_interactive(r),
            })
            .collect()
    }

    /// The lowest-value resident, lane-less `Prefilling` request eligible
    /// for eviction (P4-07, GitHub #125; class-aware priority per ADR
    /// 0023, GitHub #127): device-resident (holds real KV pages, a GDN
    /// slot and conv taps) but holding no decode lane at all — whether it
    /// is the sole half-prefilled request still chunking, or a
    /// fully-prefilled one still queued for a lane deal. Excludes `exclude` (a
    /// request index this call must never pick — the blocked head itself,
    /// when called from [`Self::try_evict_for_head`]: without this, a
    /// `Prefilling`-complete head queued for a lane matches this method's
    /// own filter and would be "evicted" to make room for itself).
    ///
    /// Ordering ([`choose_resident_candidate_victim`]): request class
    /// (Agent before Interactive), then oldest-submitted (request id) as
    /// the LRU proxy — a lane-less candidate has already had eligibility
    /// and protection settled by this method's own filtering, so class and
    /// LRU are all that remain (the GPU-residency half of ADR 0023's
    /// ordering [`RetainedLaneCandidate`] shares). `None` when no eligible
    /// candidate exists.
    fn prefilling_eviction_candidate(&self, exclude: Option<usize>) -> Option<usize> {
        let candidates: Vec<ResidentCandidate> = self
            .requests
            .iter()
            .enumerate()
            .filter(|(i, r)| {
                Some(*i) != exclude
                    && r.state == RequestState::Prefilling
                    && r.resident
            })
            .map(|(_, r)| ResidentCandidate { request_id: r.id, owner: r.class })
            .collect();
        let victim_id = choose_resident_candidate_victim(&candidates)?;
        self.requests.iter().position(|r| r.id == victim_id)
    }

    /// Whether `r`'s lane is reserved for an earlier-queued interactive
    /// request (core-07 wiring, ADR 0004): `r` holds a shared prefix
    /// (its prompt head is a cached sibling prefix) that an *earlier-
    /// queued* (smaller request id — submitted first) *interactive*
    /// request still claims while it waits (Admitted / Prefilling, not
    /// yet on a lane). Evicting `r`'s lane to the host tier would move
    /// the warm shared prefix out of the pool while the earlier
    /// interactive request still needs it — so the lane is not an
    /// eviction victim (reference policy, ported per ADR 0004).
    fn reserved_for_earlier_interactive(&self, r: &Request) -> bool {
        let Some(entry) = r.prefix_entry else {
            return false; // no shared prefix: nothing reserved
        };
        self.requests.iter().any(|c| {
            c.prefix_entry == Some(entry)
                && c.class == RequestClass::Interactive
                && c.id < r.id
                && (c.state == RequestState::Admitted || c.state == RequestState::Prefilling)
        })
    }

    /// Re-queue a discarded (evicted) request for re-prefill (core-06): its
    /// host-tier snapshot was discarded (the tier was full), so the request
    /// goes back to `Admitted` (re-prefills from the start — its warmed KV
    /// is gone) and its service-work counters are reset.
    fn requeue_request(&mut self, idx: usize, events: &mut Vec<SchedEvent>) {
        // GitHub #190: a KV-RAM blob it never restored from is not its to hold
        // across a fresh prefill.
        self.release_kv_ram_claim(idx, false);
        let r = &mut self.requests[idx];
        // core-07: capture the shared-prefix claim (the `requeue()` below
        // resets it; a re-queued request re-prefills from the start and
        // may re-claim a live entry on its fresh prefill).
        let prefix_entry = r.prefix_entry;
        r.requeue(); // Evicted → Admitted, lane released (there is none).
        r.tokens = 0;
        // GitHub #242: and the trace it built, since the re-prefill draws
        // step 0 again — a constrained decode that kept it would answer with twice its
        // own digits.
        r.drawn.clear();
        let effective_max = generation_budget(&self.config, &r.input);
        r.remaining_work = effective_max as u64;
        r.backfill_class = BackfillClass::None;
        r.backfill_epoch = 0;
        // core-07: restore the full (unshrunk) reservation — the re-queued
        // request re-prefills its *entire* prompt (not just its tail), so
        // its pool charge must cover `prompt + max` pages again (the claim
        // loop shrinks it to the tail if a live entry is re-claimed).
        let full_pages = u64::from(sequence_tokens(&self.config, &r.input))
            .div_ceil(self.config.kv_page_tokens as u64) as u32;
        r.resources.kv_pages = full_pages;
        events.push(SchedEvent::Requeued { request: r.id });
        // core-07: release the shared-prefix claim (its pages return to
        // the pool when the last claimant releases).
        self.release_prefix_claim(prefix_entry);
    }

    /// Make room in the host tier for `bytes` (core-06, GitHub #125):
    /// discard the lowest-value entries (probation LRU) while the tier is
    /// over budget, re-queueing each discarded request (its snapshot was
    /// lost — it re-prefills from the start) and freeing its pinned buffer
    /// at the leaf (`Compute::discard_snapshot`). Returns `true` when the
    /// tier can hold `bytes` (there is room, or it was made).
    ///
    /// Room is a *hole*, not a count of free bytes (GitHub #213, ADR 0030):
    /// KV-RAM is one pinned arena, so free bytes scattered between live
    /// blobs are bytes a blob still cannot be placed in.
    /// [`Compute::host_blob_fits`] is what knows the difference, and the same
    /// victim order runs for a fragmented arena as for a full budget — down
    /// to an empty tier, whose `false` is how the evict does not happen.
    fn make_host_room_for_bytes(&mut self, bytes: u64, events: &mut Vec<SchedEvent>) -> bool {
        while self.host.used_bytes() + bytes > self.host.capacity_bytes()
            || !self.compute.host_blob_fits(bytes)
        {
            let now = self.now();
            match self.host.evict_for_live(now) {
                Some(KvRamVictim::Retained(discarded)) => {
                    self.forget_kv_ram_blob(discarded, events);
                }
                Some(KvRamVictim::Live(discarded)) => {
                    self.compute.discard_snapshot(discarded.request);
                    // GitHub #224: the tier's own eviction, emitted before
                    // the requeue it causes — the snapshot left KV-RAM
                    // whether or not the request is still around to be put
                    // back on the queue.
                    events.push(SchedEvent::SnapshotDropped {
                        request: discarded.request,
                    });
                    if let Some(idx) = self.requests.iter().position(|r| r.id == discarded.request) {
                        self.requeue_request(idx, events);
                    }
                }
                None => return false, // the tier is empty (nothing to evict)
            }
        }
        true
    }

    /// Snapshot `v_idx` to the host tier and release its GPU-resident state
    /// (P4-07, GitHub #125): query the real snapshot size, make room for it
    /// in the host tier's byte budget, snapshot to pinned host memory and
    /// release the sequence (`Compute::evict`), record the tier entry (its
    /// resume phase / lane taken from `resume_phase` / `lane`), transition
    /// the request (`Request::evict` for `Running`, `Request::evict_prefilling`
    /// for `Prefilling`), free `lane` if any, and release the resident-slot
    /// + KV-page charge (`Self::unmaterialize`). Returns `true` on success,
    /// `false` on any refusal (an invalid GDN boundary — unreachable by
    /// construction for either caller — a host-tier byte budget that
    /// cannot be freed, or a leaf-level failure) — the caller leaves the
    /// candidate exactly as it was.
    fn snapshot_and_evict(
        &mut self,
        v_idx: usize,
        resume_phase: ResumePhase,
        lane: Option<LaneId>,
        events: &mut Vec<SchedEvent>,
    ) -> bool {
        let (v_id, v_class, v_pages, v_tokens, v_progress, v_work, v_gdn, v_prefix) = {
            let v = &self.requests[v_idx];
            (
                v.id,
                v.class,
                v.resources.kv_pages,
                v.tokens,
                v.prefill_progress,
                v.remaining_work,
                v.gdn.clone(),
                v.prefix_entry,
            )
        };
        // GitHub #190: a request holding a shared prefix is snapshotted with
        // the prefix's pages materialized into its blob, so it comes back as a
        // sequence owning every page of its history — and needs a reservation
        // for all of them, not the tail its claim shrank it to.
        let restore_pages = if v_prefix.is_some() {
            sequence_tokens(&self.config, &self.requests[v_idx].input)
                .div_ceil(self.config.kv_page_tokens)
        } else {
            v_pages
        };
        // core-02: only a chunk-boundary-consistent sequence may be
        // snapshotted. A `Running` candidate is always past its last
        // completed decode round, and a `Prefilling` eviction candidate is
        // only ever selected once `resident` (i.e. its first chunk already
        // landed and checkpointed) — so this is unreachable by
        // construction rather than by timing, the same invariant the
        // leaf's own `NOT_AT_BOUNDARY` refusal exists for.
        if !v_gdn.is_valid_snapshot_point(v_gdn.position()) {
            return false;
        }
        // Query the real snapshot size *before* moving or releasing
        // anything (`Compute::snapshot_size` is a cheap, non-destructive
        // query) so the host tier's byte budget can be checked (and made
        // room for) before the victim's GPU sequence is touched — a tier
        // that cannot make room leaves the victim exactly as it was rather
        // than releasing its GPU state for nothing.
        let bytes = match self.compute.snapshot_size(v_id) {
            Ok(bytes) => bytes,
            Err(_) => return false, // refused (unreachable per the boundary note above)
        };
        // Make room in the host tier (re-queueing any discarded snapshot).
        if !self.make_host_room_for_bytes(bytes, events) {
            return false; // the host tier cannot hold the snapshot
        }
        // Snapshot to pinned host memory and release the GPU sequence (its
        // KV pages, GDN slot and conv taps) — nothing else runs between the
        // size query above and this call on this single-threaded scheduler,
        // so `evict` cannot fail where `snapshot_size` just succeeded.
        let started = Instant::now();
        let Ok(bytes) = self.compute.evict(v_id) else {
            return false;
        };
        let snapshot_micros = started.elapsed().as_micros() as u64;
        let entry = HostEntry {
            request: v_id,
            resume_phase,
            lane,
            owner: v_class,
            pages: restore_pages,
            bytes,
            tokens: v_tokens,
            prefill_progress: v_progress,
            remaining_work: v_work,
            gdn: v_gdn,
            tier: Tier::Probation,
            use_tick: self.tick,
        };
        // Record the entry (`make_host_room_for_bytes` already guaranteed
        // room for `bytes`, so this cannot fail on capacity; the GDN
        // boundary was already checked above).
        self.host
            .capture(entry)
            .expect("room was made for `bytes` and the boundary was already checked");
        match resume_phase {
            ResumePhase::Running => {
                self.requests[v_idx].evict();
            }
            ResumePhase::Prefilling => {
                self.requests[v_idx].evict_prefilling();
            }
        }
        if let Some(lane) = lane {
            self.free_lanes.push(lane);
        }
        // P4-07, GitHub #125: releases the resident-slot + KV-page charge —
        // the leaf just released exactly that state above.
        self.unmaterialize(v_idx);
        // GitHub #190: the leaf dropped the sequence's hold on its prefix with
        // the sequence; this is the ledger's half. What comes back is a
        // standalone sequence, so it holds no claim and owns every page up to
        // where it stopped — no publish point below that is reachable again.
        if v_prefix.is_some() {
            self.release_prefix_claim(v_prefix);
            let r = &mut self.requests[v_idx];
            r.prefix_entry = None;
            r.prefix_publisher = None;
            r.shared_prefix_tokens = 0;
            r.shared_pages = 0;
            r.resources.kv_pages = restore_pages;
            r.standalone_tokens = v_progress;
        }
        // A `Prefilling`-phase eviction (GitHub #125) can target the
        // *protected head itself* — a request that was blocked purely on
        // lanes, with no eligible `Running` victim, falls through to
        // `Self::prefilling_eviction_candidate`, which never selects the
        // head being evaluated (`try_evict_for_head`'s own `exclude`), but
        // *can* select a *different* half-prefilled request that happens
        // to be the head of some other, still-open protection (e.g. a
        // separate fresh candidate's own materialization gate evicting it).
        // Mirrors `Self::release_request`'s own stale-protection clear:
        // without this, the next blocked head would incorrectly reuse a
        // protection frozen around a request that is no longer even
        // queued.
        if self.protection.as_ref().map(|p| p.head_request_id) == Some(v_id) {
            self.protection = None;
        }
        events.push(SchedEvent::Evicted {
            request: v_id,
            snapshot_micros,
        });
        true
    }

    /// Evict the single lowest-value eligible victim (core-06, GitHub
    /// #125): a retained decode lane first (ADR 0004's ported policy,
    /// [`choose_retained_lane_victim`]), a resident lane-less half-prefilled
    /// request otherwise ([`Self::prefilling_eviction_candidate`]).
    /// `exclude` is a request index this call must never pick as its own
    /// victim (a blocked head being evaluated for its own admission — see
    /// [`Self::try_evict_for_head`]; `None` for a fresh candidate's own
    /// materialization gate, which can never self-select since it is still
    /// `Admitted`, never `Prefilling`, at that point). Returns `true` when
    /// a victim was evicted, `false` when none remains eligible.
    fn evict_one_victim(&mut self, exclude: Option<usize>, events: &mut Vec<SchedEvent>) -> bool {
        // `exclude` (when set) is always `Prefilling` (the blocked head
        // `try_evict_for_head` is evaluating), so it can never be a
        // `Running`-lane candidate in the first place — nothing to guard
        // here specifically; `Self::prefilling_eviction_candidate` is
        // where excluding it actually matters.
        let candidates = self.retained_lane_candidates();
        if let Some(victim_lane) = choose_retained_lane_victim(&candidates) {
            let Some(v_idx) = self
                .requests
                .iter()
                .position(|r| r.lane == Some(victim_lane))
            else {
                return false;
            };
            return self.snapshot_and_evict(v_idx, ResumePhase::Running, Some(victim_lane), events);
        }
        if let Some(v_idx) = self.prefilling_eviction_candidate(exclude) {
            return self.snapshot_and_evict(v_idx, ResumePhase::Prefilling, None, events);
        }
        false
    }

    /// Make room for `needed` additional resources on top of what is
    /// currently charged (P4-07, GitHub #125): evicts the lowest-value
    /// eligible victim ([`Self::evict_one_victim`]), repeatedly, until
    /// `needed` fits alongside current usage or no evictable victim
    /// remains. This is the admission-refusal path a fresh candidate's
    /// *materialization* drives too — not only a blocked, already-resident
    /// head's lane deal (see [`Self::fits_for_materialization`]'s own
    /// caller). The caller is never itself a valid victim at this point
    /// (see [`Self::evict_one_victim`]'s own `exclude` doc), so this never
    /// needs one.
    fn make_room(&mut self, needed: &AdmissionResources, events: &mut Vec<SchedEvent>) -> bool {
        loop {
            let used = AdmissionResources {
                lanes: N_DECODE_LANES as u32 - self.free_lanes.len() as u32,
                kv_pages: self.kv_used_pages,
                backend_pages: 0,
                resident_slots: self.resident_slots_used,
            };
            if used.add(needed).fits(&self.capacity) {
                return true;
            }
            if !self.evict_one_victim(None, events) {
                return false;
            }
        }
    }

    /// Try to admit a blocked head by evicting retained state until it fits
    /// (core-06, GitHub #125): the head is always already device-resident
    /// (a `Prefilling`, prefill-complete request), so it needs only a
    /// decode lane — [`Self::fits`] reflects that. Returns `true` once the
    /// head fits (the caller deals it), `false` when no evictable victim
    /// remains (the head is held — the backfill / donor wait path).
    fn try_evict_for_head(&mut self, head_idx: usize, events: &mut Vec<SchedEvent>) -> bool {
        loop {
            if self.fits(&self.requests[head_idx]) {
                return true;
            }
            // `head_idx` is itself `Prefilling` (queued for a lane) and so
            // matches `prefilling_eviction_candidate`'s own filter —
            // excluded here so the head is never evicted to make room for
            // itself.
            if !self.evict_one_victim(Some(head_idx), events) {
                return false; // no evictable victim (all reserved / none)
            }
            // Loop: re-check whether the head now fits.
        }
    }

    /// Restore evicted (suspended) requests (core-06, GitHub #125): a
    /// restored request resumes from where it was evicted (no re-prefill),
    /// taking priority over a fresh prefill. A [`ResumePhase::Running`]
    /// entry needs a free decode lane; a [`ResumePhase::Prefilling`]
    /// (half-prefilled) entry needs none — it resumes chunking from
    /// [`crate::host::HostEntry::prefill_progress`] and re-earns a lane the
    /// normal way once its prefill completes. Either way, restoring
    /// re-materializes the request's resident-slot + KV-page charge
    /// (`Self::materialize`) before anything else, mirroring exactly what
    /// eviction released. Restores as many evicted requests as there is
    /// room for, in the host tier's victim order (the entries closest to
    /// being discarded). A physical restore failure (a corrupt/foreign
    /// blob, or a leaf-level error) discards the snapshot and re-prefills
    /// the request instead — the same fallback the tier's own byte-budget
    /// discard uses.
    fn restore_pass(&mut self, events: &mut Vec<SchedEvent>) {
        loop {
            let victim = match self.host.victim() {
                Some(v) => v.clone(),
                None => break, // no evicted request to restore
            };
            if victim.resume_phase == ResumePhase::Running && self.free_lanes.is_empty() {
                break; // no lane: leave it (retry next advance)
            }
            // The restored request's resident slot + pages must fit the
            // GPU pool either way.
            if self.resident_slots_used + 1 > self.capacity.resident_slots
                || self.kv_used_pages + victim.pages > self.capacity.kv_pages
            {
                break; // no room: leave it (retry next advance)
            }
            let idx = self
                .requests
                .iter()
                .position(|r| r.id == victim.request)
                .expect("a host-tier snapshot always maps to a request");
            let context_tokens = sequence_tokens(&self.config, &self.requests[idx].input);
            let started = Instant::now();
            match self.compute.restore(victim.request, context_tokens) {
                Ok(()) => {
                    let restore_micros = started.elapsed().as_micros() as u64;
                    self.host
                        .restore(victim.request)
                        .expect("the victim is a tier entry");
                    // P4-07, GitHub #125: re-charges resident_slots +
                    // kv_pages — the leaf just re-materialized exactly that
                    // state above.
                    self.materialize(idx);
                    let lane = match victim.resume_phase {
                        ResumePhase::Running => {
                            let lane = self.free_lanes.pop().expect("checked non-empty above");
                            self.requests[idx].restore_lane(lane);
                            Some(lane)
                        }
                        ResumePhase::Prefilling => {
                            // Resume chunking from the snapshotted boundary,
                            // not from zero.
                            self.requests[idx].prefill_progress = victim.prefill_progress;
                            self.requests[idx].restore_prefilling();
                            None
                        }
                    };
                    events.push(SchedEvent::Restored {
                        request: victim.request,
                        lane,
                        restore_micros,
                    });
                }
                Err(_) => {
                    // The blob could not be restored: drop it (never
                    // promoted — it did not actually resume) and free
                    // whatever the leaf still held for it, then re-prefill
                    // from scratch.
                    self.host.discard_request(victim.request);
                    self.compute.discard_snapshot(victim.request);
                    self.requeue_request(idx, events);
                    // Loop: try the next victim.
                }
            }
        }
    }
}

impl Scheduler for ConcreteScheduler {
    fn submit(
        &mut self,
        mut input: RequestInput,
        class: RequestClass,
    ) -> Result<RequestId, SubmitError> {
        if input.model != self.config.model {
            return Err(SubmitError::UnknownModel(input.model));
        }
        // GitHub #242: a **constrained decode**'s budget is its schedule, and nothing
        // else may cut it short. Dropped here, once, rather than checked at
        // each of the places `max_tokens` is read: the backend enforces it
        // too (`RuntimeCompute::decode_step` finishes a lane whose
        // `generated` has reached it *before* the leaf runs, and cannot tell
        // a run's last round from an ordinary one), so a request that
        // carried one would return a number with fewer digits than the
        // caller asked for — which is not a shorter answer but a wrong one.
        if input.is_constrained() {
            input.params.max_tokens = None;
        }
        if self.in_flight() >= self.config.max_in_flight {
            return Err(SubmitError::Full);
        }
        // GitHub #166: the leaf reserves at most `max_sequence_tokens` (its
        // `max_context`) for one sequence, whatever the pool holds. A prompt
        // that already fills the limit, or a prompt + `max_tokens` that
        // overruns it, can never be allocated — refused here rather than
        // failing at the leaf on every advance.
        let prompt_tokens = input.tokens.len() as u64;
        let effective_max = generation_budget(&self.config, &input);
        let reserved_tokens = prompt_tokens + u64::from(effective_max);
        let limit = self.config.max_sequence_tokens;
        if prompt_tokens >= u64::from(limit) || reserved_tokens > u64::from(limit) {
            return Err(SubmitError::ContextExceeded {
                requested: reserved_tokens,
                limit,
            });
        }
        // core-05: compute the request's KV reservation (prompt + the
        // effective token budget, in pages) and reject requests that can
        // never fit — even alone (they would block the queue forever).
        let kv_pages = reserved_tokens.div_ceil(self.config.kv_page_tokens as u64) as u32;
        let resources = AdmissionResources {
            lanes: 1,
            kv_pages,
            backend_pages: 0,
            resident_slots: 1,
        };
        if !admission_resources_fit(&resources, &self.capacity) {
            return Err(SubmitError::Oversized);
        }
        let id = self.next_id;
        self.next_id += 1;
        // P4-10 (GitHub #126): the shareable head of this prompt, decided
        // now rather than after its prefill. A prefix is published at the
        // chunk boundary that lands on it, so the chunk decomposition has to
        // know where that is before it cuts the first chunk.
        // GitHub #238: a decision's shareable head is cut from its *publish
        // reach* — one token less, so the page floor lands a page below a
        // page-multiple prompt rather than on it. Publishing at the prompt's
        // own length would create an entry no decision could ever claim.
        let head = self.prefix.shareable_head_tokens(input.publish_reach());
        // GitHub #186 (ADR 0029): with cross-request reuse on, the
        // head is floored to the **opener's** page rather than the
        // prompt's. That is what puts the generation opener inside
        // the request's own first KV page — the page a later claimant
        // *copies*, since it is still being written — while every
        // page before it is a whole page the claimant *shares*.
        // Without this the two boundaries drift apart whenever the
        // last two or three prompt tokens happen to cross a page, and
        // no checkpoint could be taken on those prompts at all.
        // It never raises the head, so a concurrent sibling loses at
        // most one page of shared prefix.
        //
        // GitHub #188 (ADR 0029): and floored again to the **system
        // block's** page when the frontend reported one of at least a
        // page. That is the retained-prefix boundary, and it is the
        // only head a *burst* can ever share — no subagent's prompt
        // extends its sibling's, so a prefix cut any further in
        // matches nobody but its own conversation. A prompt whose
        // block is under one page, or that has none, keeps #186's
        // opener page and its prompt checkpoint with it.
        //
        // One head is published, not two: the leaf allows one prefix
        // per sequence (`seq_prefix.cu:129`, "already claims a shared
        // prefix"), and a checkpoint demands its whole pages *be* that
        // prefix (`seq_checkpoint.cu:125`). So the extra chunk split
        // the spec asks for is **moved**, not added.
        //
        // The two boundaries share that head when they fall in the
        // same page — a short first turn leaves a retained prefix
        // *and* a checkpoint — and compete for it when they do not,
        // which is every prompt carrying tools.
        //
        // The competition is temporary, and GitHub #187 ends it by
        // **chaining** the publish rather than by weakening the
        // capture. `seq_checkpoint.cu:125`'s
        // `below != seq->shared_pages` stays, and must: a checkpoint
        // holds exactly one copied tail page and its capture takes
        // `seq->kv.page_ids()[0]`, so with the opener pages above the
        // block the copied page would not be the opener's at all and
        // the pages between would have no holder — which is the
        // defect #186 fixed in `565d634`. What #187 removes instead is
        // `seq_prefix.cu:129`'s `seq->prefix != nullptr`, so a
        // sequence standing on the block publishes a *second* prefix
        // over the head it warmed itself, taking over the reference it
        // held. The intermediate pages get their holder from that
        // chained link, and `below == shared_pages` becomes true
        // rather than relaxed.
        let opener_page = |at: u32| (at / self.config.kv_page_tokens) * self.config.kv_page_tokens;
        let publish_tokens = match input.opener_tokens.filter(|_| self.config.prompt_reuse) {
            Some(opener) => head.min(opener_page(opener)),
            None => head,
        };
        // GitHub #193: a multimodal request publishes at that head too — its
        // prefixes are keyed by their images as well as their token ids, so a
        // sibling sending another picture never matches past it. What it must
        // never do is end *inside* an image: the head walks back to the page
        // holding the item's first placeholder. When that page is below the
        // opener's, the checkpoint is lost for this turn (its whole pages are
        // no longer the prefix, `Request::checkpoint_point`), which is the
        // price of an image ending within a page of the generation opener.
        let publish_tokens = input.prefix_floor(publish_tokens, self.config.kv_page_tokens);
        let mut request = Request::new(id, class, input, resources, effective_max as u64);
        // GitHub #187 × #188: the two boundaries **compose** rather than
        // exclude each other. #188 floored this to the system block, because
        // the leaf allowed one prefix per sequence and the block is the only
        // head a burst can share — which cost every request its prompt
        // checkpoint, since qwen-code sends tools and the block is therefore
        // never short. A sequence now holds a prefix *chain*, so the request
        // publishes at the block and then chains the opener's page over it
        // (`Request::publish_point` walks both in prompt order). This stays
        // the opener's page floor, and the block joins it there.
        request.publish_tokens = publish_tokens;

        // And `--prompt-reuse off` publishes at neither boundary. #188 kept
        // that by gating the block where it floored the publish point; with
        // the flooring gone the gate lives here, on the request's own copy of
        // the structural offsets — the same thing the opener filter above
        // does, so that off really is the engine that existed before.
        if !self.config.prompt_reuse {
            request.input.system_block_tokens = None;
        }
        self.requests.push(request);
        Ok(id)
    }

    fn cancel(&mut self, request: RequestId) -> bool {
        ConcreteScheduler::cancel(self, request)
    }

    fn advance(&mut self) -> Vec<SchedEvent> {
        let mut events: Vec<SchedEvent> = Vec::new();
        self.last_error = None;
        // core-06: advance the scheduling tick (the LRU `use_tick` for
        // retained-lane victim selection).
        self.tick += 1;

        // P3-01 / ADR 0018 — cancel is abort, not suspend: release every
        // request marked by `cancel()` since the last `advance()`, before
        // this tick's phases run. Every compute call is synchronous, so
        // whatever chunk was in flight when `cancel()` was called has
        // already returned by now — there is nothing to interrupt, and no
        // suspend/resume primitive is needed.
        let to_cancel: Vec<usize> = self
            .requests
            .iter()
            .enumerate()
            .filter(|&(_, r)| r.cancelled && r.state != RequestState::Done)
            .map(|(i, _)| i)
            .collect();
        for idx in to_cancel {
            self.release_request(idx);
        }

        // Phase 1 — chunked, interleaved prefill (P3-01, ADR 0018): at
        // most one `prefill_step` call this tick. Exactly one request may
        // hold multi-tick (device-resident) prefill progress at a time —
        // find it first; if one exists, it alone is served this tick (the
        // rest queue). Otherwise the batch is built fresh from the
        // `Admitted` queue, same as before chunking existed.
        let active = self
            .requests
            .iter()
            .position(|r| r.state == RequestState::Prefilling && !r.prefill_complete());
        let batch: Vec<usize> = match active {
            Some(idx) => vec![idx],
            None => {
                let mut b: Vec<usize> = self
                    .requests
                    .iter()
                    .enumerate()
                    .filter(|&(_, r)| r.state == RequestState::Admitted)
                    .map(|(i, _)| i)
                    .collect();
                b.sort_by_key(|&i| (self.requests[i].class, self.requests[i].id));
                b.truncate(self.config.max_prefill_batch);

                // core-07 — sibling prefix claim: a *fresh* candidate
                // claims the longest cached prefix of its prompt (skipping
                // the redundant prefill — its first job carries only the
                // tail, and its own reservation shrinks to the tail + max:
                // the shared entry's pages are charged to the pool once,
                // for every claimant). Runs *before* the P4-07 gating loop
                // below, which must see this reduced reservation — not the
                // unclaimed one — or it would charge (and evict for) pages
                // the claimant was never actually going to reserve.
                for &i in &b {
                    // A request that already holds a claim (from a prior
                    // advance, whose prefill failed and is retried) keeps
                    // it: re-claiming would double-count the entry's
                    // refcount and the `sibling_prefix_reused_tok` counter,
                    // and pin the entry forever (the release happens once,
                    // at completion).
                    // GitHub #190: a KV-RAM claimant holds no prefix, so the
                    // checkpoint claim is what says it already chose.
                    if self.requests[i].prefix_entry.is_some()
                        || self.requests[i].checkpoint_publisher.is_some()
                    {
                        continue;
                    }
                    // GitHub #186 (ADR 0029) — longest reuse wins. A retained
                    // prompt checkpoint reaches all the way to a finished
                    // conversation's generation opener; a sibling prefix
                    // stops at a page boundary. The checkpoint is *peeked*
                    // first and used as the floor the prefix claim has to
                    // beat, because claiming a prefix is not free — it pins
                    // the entry and counts a skip — so a prefix that reaches
                    // no further must never be claimed at all.
                    // One walk of the prompt answers all four questions below.
                    let keys = self.reuse_keys(i);
                    let lookup = self.checkpoints.lookup(&keys);
                    self.requests[i].pending_retained_misses =
                        lookup.misses(self.checkpoints.tiers());
                    // A KV-RAM checkpoint has to beat every device reuse by
                    // its restore floor, a prefix's included (GitHub #190).
                    let mut device_prefix = self
                        .prefix
                        .longest_match_tokens(&keys);
                    let mut retained = self.checkpoints.select(&lookup, device_prefix);
                    // GitHub #190: a retained prefix only KV-RAM still holds
                    // comes back to the device when it beats every device
                    // reuse by the restore floor and every KV-RAM checkpoint
                    // outright — and from then on the burst shares it there.
                    if let Some((spilled, length)) = self
                        .prefix
                        .longest_spilled_match(&keys)
                        .map(|s| (s.id, s.length_tokens))
                    {
                        let device_checkpoint =
                            lookup.longest_in(ReuseSource::Device).map_or(0, |m| m.tokens);
                        let kv_ram_checkpoint = retained
                            .as_ref()
                            .filter(|m| m.source == ReuseSource::KvRam)
                            .map_or(0, |m| m.tokens);
                        let best_on_device = device_prefix.max(device_checkpoint);
                        if length >= best_on_device.saturating_add(KV_RAM_RESTORE_FLOOR_TOKENS)
                            && length > kv_ram_checkpoint
                            && self.return_prefix(spilled, &mut events)
                        {
                            device_prefix = length;
                            retained = self.checkpoints.select(&lookup, device_prefix);
                        }
                    }
                    let floor = retained.as_ref().map_or(0, |m| m.tokens);
                    let claimed = self
                        .prefix
                        .claim_longer_than(&keys, floor);
                    if let Some(claim) = claimed {
                        // GitHub #188: a claim moves a *retained* prefix to
                        // the back of the LRU order (a no-op on a live
                        // sibling's). A system block a burst is still arriving
                        // against must not be the first victim merely because
                        // it was published long ago.
                        let now = self.now();
                        self.prefix.touch_retained(claim.id, self.tick, now);
                        let r = &mut self.requests[i];
                        r.prefix_entry = Some(claim.id);
                        r.prefix_publisher = Some(claim.publisher);
                        r.shared_prefix_tokens = claim.tokens;
                        r.shared_pages = claim.pages;
                        r.prefill_progress = claim.tokens; // the shared head is already warm
                        r.gdn = claim.gdn; // core-02: resume at the shared boundary
                        // Shrink the claimant's own reservation by the
                        // shared prefix's pages (the entry now owns them —
                        // charged once, for every claimant). `ceil((prompt
                        // + max) / pt) - shared_pages` equals `ceil((tail +
                        // max) / pt)`: the shared head is page-aligned, so
                        // subtracting its whole pages is exact.
                        r.resources.kv_pages = r.resources.kv_pages.saturating_sub(claim.pages);
                        let request = r.id;
                        events.push(SchedEvent::PrefixReused {
                            request,
                            tokens: claim.tokens,
                            retained: !self.is_live(claim.publisher),
                        });
                    } else if let Some(m) = retained {
                        // Resume from an *earlier, finished* request's state.
                        // The claimant takes a holder's reference on the
                        // shared pages under the checkpoint exactly as a
                        // sibling would, so those pages are charged to the
                        // pool once however many live requests and retained
                        // entries stand on them. It does **not** set
                        // `prefix_publisher`: what its first job carries is
                        // the checkpoint, which the backend claims by copying
                        // the mutable image and the partial tail page on top
                        // of the shared pages — a prefix claim would stop a
                        // whole page short and lose the opener.
                        let now = self.now();
                        self.checkpoints.record_claim(m.id, self.tick, now);
                        let on_device = m.source == ReuseSource::Device;
                        if on_device {
                            self.prefix.retain(m.prefix);
                        } else {
                            // GitHub #190: held until the restore lands, so
                            // nothing discards the only copy in between.
                            let held = self.host.claim_retained(RetainedBlob::Checkpoint(m.id));
                            debug_assert!(held, "a KV-RAM match is a held KV-RAM blob");
                        }
                        events.push(SchedEvent::RetainedState {
                            operation: RetainedStateOperation::Hit,
                            source: m.source,
                            kind: RetainedKind::Checkpoint,
                        });
                        let r = &mut self.requests[i];
                        r.prefix_entry = on_device.then_some(m.prefix);
                        if !on_device {
                            r.kv_ram_claim = Some(m.id);
                            r.standalone_tokens = m.tokens;
                        }
                        r.checkpoint_publisher = Some(m.publisher);
                        // GitHub #187 — the claim edge, kept so this request's
                        // own capture knows which conversation it continues
                        // and which entry it is entitled to supersede.
                        r.checkpoint_entry = Some(m.id);
                        r.checkpoint_tokens = m.tokens;
                        r.reuse_source = Some(m.source);
                        r.shared_pages = if on_device { m.pages } else { 0 };
                        r.prefill_progress = m.tokens; // warm all the way to the opener
                        r.gdn = m.gdn;
                        // A device claimant shares the checkpoint's whole
                        // prefix pages and subtracts their existing charge.
                        // KV-RAM restores a standalone materialized blob into
                        // fresh pages, so it keeps the full reservation.
                        if on_device {
                            r.resources.kv_pages = r.resources.kv_pages.saturating_sub(m.pages);
                        }
                    }
                }

                // A fresh batch (no active carry-over) may still pack
                // several requests into this one call, but only while each
                // finishes within its own single chunk (P3-01: nothing
                // beyond this call may leave more than one request
                // `Prefilling` and incomplete). The moment one candidate's
                // remaining span exceeds the chunk width, it is included
                // (it becomes this tick's chunk) and the batch stops there
                // — whatever queued behind it waits for its turn. Also
                // before the gating loop, for the same reason the claim
                // loop is: a candidate cut here is never materialized this
                // tick at all.
                // GitHub #178: "exceeds" is decided by the chunk the request
                // would actually be dealt, which a second media item can cut
                // short of the serving width.
                if let Some(cut) = b.iter().position(|&i| {
                    let r = &self.requests[i];
                    chunk_take(r, self.config.serving_chunk_tokens)
                        < r.input.tokens.len() as u32 - r.prefill_progress
                }) {
                    b.truncate(cut + 1);
                }

                // P4-07 (GitHub #125): a fresh candidate materializes a
                // real device-resident sequence (KV pages, GDN slot, conv
                // taps) the moment its first chunk lands — admission has to
                // agree there is room *before* that happens, evicting on
                // this same admission-refusal path if not, rather than
                // letting a real leaf allocation fail (which would surface
                // page pressure as a raw kernel error instead of a
                // refusal the tier can act on). `b` is final by now (the
                // claim and the over-wide-chunk cut above already ran), so
                // every candidate's `resources.kv_pages` is exactly what it
                // will actually reserve.
                //
                // `Self::materialize` runs *inside* this loop, immediately
                // once a candidate is confirmed to fit or room was made for
                // it — not deferred to the per-job success loop below. Two
                // fresh candidates batched together both read
                // `fits_for_materialization` against the *same* counters if
                // neither has actually charged yet; charging eagerly here is
                // what makes the second one see the first one's charge
                // (`RuntimeCompute::prefill_step`'s own per-job allocation
                // order is exactly this sequential — a real leaf never
                // reserves both without noticing the first). The later
                // per-job call stays as a no-op safety net (idempotent) for
                // the one path that skips this loop entirely: a continuing
                // (already-`Prefilling`) request re-materializing after an
                // earlier chunk's failure unmaterialized it.
                let mut admitted = Vec::with_capacity(b.len());
                for i in b {
                    // GitHub #186 (ADR 0023 as amended by 0029): retained
                    // state is the first victim, so it goes back *before*
                    // anything else is considered. A live request never waits
                    // for a bet and is never refused because of one.
                    if !self.fits_for_materialization(&self.requests[i]) {
                        self.reclaim_retained(i, &mut events);
                    }
                    if !self.fits_for_materialization(&self.requests[i]) {
                        let needed = AdmissionResources {
                            lanes: 0,
                            kv_pages: self.requests[i].resources.kv_pages,
                            backend_pages: 0,
                            resident_slots: 1,
                        };
                        if !self.make_room(&needed, &mut events) {
                            // No room, and none could be made: stop the
                            // batch here — `i` (and anything sorted after
                            // it) waits for a later tick.
                            break;
                        }
                    }
                    self.materialize(i);
                    admitted.push(i);
                }
                admitted
            }
        };
        // P4-10 (GitHub #126) — one publisher per prompt head per batch.
        //
        // Two requests with the same prompt arriving before either has
        // prefilled both see an empty cache, so neither can claim and both
        // would publish. Only one of them could then register — and the
        // loser's prefix would be a device image nothing ever claims,
        // holding its pages for as long as its sequence lives. So the first
        // candidate in the batch takes the head; the rest prefill it
        // themselves this tick (there is nothing warm to claim yet) and a
        // later sibling claims the one entry that did register.
        //
        // GitHub #193: "the same head" is the same *content* — token ids and
        // the images inside them. Two siblings sending same-size pictures
        // have identical ids, and each is the only publisher of its own head.
        // The key is walked only for a request whose point collides with an
        // earlier one's, which is rare, rather than for every request on
        // every tick its point is nonzero.
        let mut publish_points: Vec<u32> = {
            let head_key = |i: usize, at: u32| {
                let media = media_keys(&self.requests[i].input);
                PromptContent::new(&self.requests[i].input.tokens, &media).key_at(at)
            };
            let mut points: Vec<u32> = Vec::with_capacity(batch.len());
            for (n, &i) in batch.iter().enumerate() {
                let at = self.requests[i].publish_point(self.config.kv_page_tokens);
                let taken = at > 0 && {
                    let mut earlier = batch[..n]
                        .iter()
                        .enumerate()
                        .filter(|&(m, _)| points[m] == at)
                        .peekable();
                    earlier.peek().is_some() && {
                        let head = head_key(i, at);
                        earlier.any(|(_, &j)| head_key(j, at) == head)
                    }
                };
                points.push(if taken { 0 } else { at });
            }
            points
        };
        // GitHub #186 — where each request's prefill is cut for its own
        // prompt checkpoint, and 0 for one that takes none.
        //
        // A capture is a **bet**: it is never taken over state something
        // identical is already retained at, and it is offered only when a
        // retained slot can hold it (below). Declining costs the request
        // nothing at all — the cut simply is not made, and the chunk runs its
        // full width.
        let mut capture_points: Vec<u32> = {
            batch
                .iter()
                .enumerate()
                .map(|(n, &i)| {
                    // `--prompt-reuse off` captures nothing, whatever slots the
                    // load gives live siblings (GitHub #215).
                    if !self.config.prompt_reuse {
                        return 0;
                    }
                    let r = &self.requests[i];
                    // GitHub #238: a decision captures no prompt checkpoint.
                    //
                    // A checkpoint is retained *after* the capturing request
                    // is gone, so it is a bet on a later request that
                    // extends this prompt. Nothing extends a decision: it is
                    // the whole request, answered at its own last position,
                    // and the only thing that could claim its opener is
                    // another copy of itself. The conservative call is to
                    // spend no retained slot on that bet.
                    //
                    // **This may be leaving reuse on the table**, and it is
                    // stated here rather than left to arithmetic because the
                    // arithmetic is not what refuses it. An earlier version
                    // of this comment claimed a decision's prompt ends at
                    // its generation opener, so a capture there would cover
                    // everything and be claimable by nobody. That is false
                    // for the 27B's template, which appends a closed think
                    // block after the opener
                    // (`crates/server/tests/decide_prompt_tail.rs`): the
                    // opener sits four tokens inside the prompt, well within
                    // `reuse_reach`, so an exact repeat *could* claim it and
                    // prefill only the tail. GitHub #240 owns measuring
                    // whether that is worth a retained slot; until it does,
                    // this refuses rather than guesses.
                    if r.input.is_decision() {
                        return 0;
                    }
                    let at = match r.checkpoint_point(self.config.kv_page_tokens) {
                        // A **page-aligned** opener falls exactly on the
                        // publish point, and this is the chunk that creates
                        // the prefix — so `checkpoint_point` cannot see one
                        // yet and would refuse a capture that is in fact
                        // perfectly placed. One prompt in sixty-four lands
                        // here; the capture rides the publish chunk instead,
                        // and the backend publishes then captures in the one
                        // call.
                        0 => {
                            let publish_at = publish_points[n];
                            // GitHub #187: a *claimant* publishing a chained
                            // head lands here too, so this no longer asks the
                            // request to hold no prefix. `publish_at` already
                            // answers that question — `Request::publish_point`
                            // returns 0 for a request not publishing at all.
                            let rides_the_publish = publish_at > 0
                                && r.input.opener_tokens == Some(publish_at)
                                && !r.checkpoint_captured;
                            if rides_the_publish { publish_at } else { 0 }
                        }
                        at => at,
                    };
                    if at == 0 {
                        return 0;
                    }
                    // GitHub #189: what makes two capture points the same is
                    // their content, so the duplicate check is over the key —
                    // and the key is what the entry will carry anyway.
                    //
                    // Known cost, filed as a follow-up rather than fixed here:
                    // the chain is walked to `at` on every advance a capture is
                    // possible on, not only on the chunk that lands on it — a
                    // claimant whose opener floors to `shared_pages` pays it
                    // every tick. The narrowing is to evaluate this only while
                    // `start < at <= start + take`, and the `take` it has to
                    // test is the **pre-cut** one: the cut below is computed
                    // *from* this answer, so testing the cut width would make
                    // the guard true by construction.
                    let media = media_keys(&r.input);
                    let head = PromptContent::new(&r.input.tokens, &media).key_at(at);
                    if self.checkpoints.holds(head) { 0 } else { at }
                })
                .collect()
        };
        // GitHub #215 (ADR 0030) — a publish or a capture that lands on this
        // chunk takes a retained slot before the backend is asked for it, so
        // serving never allocates device memory for retained state. When no
        // slot is free, retained state gives one up; when nothing can, the
        // point is dropped: the request runs, its chunk is not cut there, and
        // it leaves no reuse behind. A capture's tail page is a KV page of the
        // pool, so the pool has to have one spare too.
        //
        // Only the landing chunk asks. A point further along costs nothing
        // until the chunk that reaches it, so an earlier tick never holds a
        // slot for it.
        let mut places: Vec<(Option<RetainedAt>, Option<RetainedAt>)> = vec![(None, None); batch.len()];
        let mut tail_pages_asked = 0;
        for (n, &i) in batch.iter().enumerate() {
            let (request, start, take) = {
                let r = &self.requests[i];
                (r.id, r.prefill_progress, chunk_take(r, self.config.serving_chunk_tokens))
            };
            let publish_at = publish_points[n];
            if lands_on(start, take, publish_at) {
                let holder = RetainedHolder::Prefix {
                    publisher: request,
                    tokens: publish_at,
                };
                match self.take_retained_slot(holder, &mut events) {
                    Some(slot) => places[n].0 = Some(RetainedAt { tokens: publish_at, slot }),
                    None => {
                        publish_points[n] = 0;
                        // A capture riding this publish stands on the prefix
                        // it would have made.
                        if capture_points[n] == publish_at {
                            capture_points[n] = 0;
                        }
                        self.skip_retained(request, RetainedSkip::PublishNoSlot, &mut events);
                    }
                }
            }
            let capture_at = capture_points[n];
            if lands_on(start, cut_at(start, take, publish_points[n]), capture_at) {
                let tail = self.tail_pages_at(capture_at);
                let skip = if self.kv_used_pages + tail_pages_asked + tail > self.capacity.kv_pages {
                    Some(RetainedSkip::CaptureNoPage)
                } else {
                    match self.take_retained_slot(RetainedHolder::Checkpoint { publisher: request }, &mut events) {
                        Some(slot) => {
                            places[n].1 = Some(RetainedAt { tokens: capture_at, slot });
                            tail_pages_asked += tail;
                            None
                        }
                        None => Some(RetainedSkip::CaptureNoSlot),
                    }
                };
                if let Some(skip) = skip {
                    capture_points[n] = 0;
                    self.skip_retained(request, skip, &mut events);
                }
            }
        }
        // Each job carries at most `serving_chunk_tokens` tokens starting
        // at the request's own prefill progress (0 for a fresh request
        // with no shared prefix, `shared_prefix_tokens` for a claimant,
        // or wherever an earlier chunk left off for a continuing request).
        let jobs: Vec<PrefillJob> = batch
            .iter()
            .enumerate()
            .map(|(n, &i)| {
                let r = &self.requests[i];
                let start = r.prefill_progress;
                let remaining = r.input.tokens.len() as u32 - start;
                // P4-10 (GitHub #126): a request that will publish a prefix
                // is cut at its publish point, even mid-prompt. The leaf
                // hands a claimant the mutable state at the prefix's *end*,
                // and a chunk that overshot it would have moved that state
                // on — so the point is a scheduling decision, not a detail of
                // the publish call.
                //
                // GitHub #186: and cut again at the generation opener, for
                // the same reason — the state a claimant of the checkpoint
                // receives is the state *there*. The opener is at most a page
                // past the publish point, so this second cut costs one short
                // chunk (typically a few dozen tokens) and only on the tick
                // that actually takes the checkpoint.
                let take = cut_at(
                    start,
                    cut_at(start, chunk_take(r, self.config.serving_chunk_tokens), publish_points[n]),
                    capture_points[n],
                );
                let (publish_prefix, capture_checkpoint) = places[n];
                debug_assert!(
                    publish_prefix.is_none_or(|p| start + take == p.tokens)
                        && capture_checkpoint.is_none_or(|c| start + take == c.tokens),
                    "a retained slot is taken only for the chunk that lands on its point"
                );
                let tokens = r.input.tokens[start as usize..(start + take) as usize].to_vec();
                // A stochastic prefill samples and updates the sequence's
                // penalty-count row. Only the final chunk's successor is
                // ever emitted; intermediate successors are discarded by
                // the next prompt chunk and therefore must stay greedy (the
                // greedy leaf branch has no sampling-state side effect).
                let params = if take == remaining {
                    r.input.params
                } else {
                    DecodeParams {
                        max_tokens: r.input.params.max_tokens,
                        ..DecodeParams::default()
                    }
                };
                PrefillJob {
                    request: r.id,
                    tokens,
                    context_tokens: sequence_tokens(&self.config, &r.input),
                    start_position: start,
                    params,
                    // Carried on the request's *first* job only — the one
                    // that starts where the prefix ends. That is the job the
                    // backend builds the sequence on, against the leaf's
                    // prefix; every later chunk finds it already built.
                    shared_prefix: r.prefix_publisher.filter(|_| start == r.shared_prefix_tokens).map(
                        |publisher| SharedPrefixClaim {
                            publisher,
                            tokens: r.shared_prefix_tokens,
                        },
                    ),
                    // Exactly "the chunk that lands on the publish point, for a
                    // request that has one and a slot to put it in".
                    publish_prefix,
                    // GitHub #186. Carried on the request's *first* job, the
                    // one that starts where the checkpoint ends — the job the
                    // backend builds the sequence on.
                    checkpoint: r
                        .checkpoint_publisher
                        .filter(|_| start == r.checkpoint_tokens)
                        .map(|publisher| CheckpointClaim {
                            publisher,
                            tokens: r.checkpoint_tokens,
                            source: r.reuse_source.unwrap_or(ReuseSource::Device),
                        }),
                    // Exactly "the chunk that lands on the opener, for a
                    // request that takes a checkpoint".
                    capture_checkpoint,
                    multimodal: r.input.multimodal.clone(),
                    // GitHub #237's seam, asked for by GitHub #238's request
                    // kind. Only on the chunk that ends at the prompt's last
                    // position: that is the position whose next-token
                    // distribution holds the decision, and the chunk is
                    // never empty because `reuse_reach` left it a token.
                    readout: r
                        .input
                        .decision
                        .as_ref()
                        .and_then(DecisionRead::answers)
                        .cloned()
                        .filter(|_| start + take >= r.input.tokens.len() as u32),
                    // GitHub #242: and the run's first step, on the same
                    // chunk and for a related reason — this prefill draws the
                    // token the first decode round emits, so step 0's set
                    // belongs here and nowhere else. `reuse_reach` is what
                    // guarantees the chunk is not empty.
                    permitted: r
                        .input
                        .constrained
                        .as_ref()
                        .filter(|_| start + take >= r.input.tokens.len() as u32)
                        .and_then(|schedule| schedule.step(0)),
                    // GitHub #260: the attention readout, on the same chunk
                    // and for the same reason as the readout — its query is
                    // the prompt's last position. `prefill_tail` is what
                    // keeps that chunk wide enough for the leaf to read.
                    attention: r
                        .input
                        .attention()
                        .filter(|_| start + take >= r.input.tokens.len() as u32)
                        .cloned(),
                }
            })
            .collect();
        if !jobs.is_empty() {
            match self.compute.prefill_step(&jobs) {
                Ok(outcomes) => {
                    // One outcome per job, in order (GitHub #192). A backend
                    // that returns fewer would have its chunks misattributed
                    // by the zip below rather than caught.
                    debug_assert_eq!(outcomes.len(), jobs.len(), "one prefill outcome per job");
                    for ((&i, job), outcome) in batch.iter().zip(&jobs).zip(
                        outcomes.iter().cloned().chain(std::iter::repeat(PrefillOutcome::default())),
                    ) {
                        // GitHub #81 / ADR 0012: the prefill span — one per
                        // request per `prefill_step` call (the chunked-
                        // prefill call boundary), opened here rather than
                        // around the call above, since one call may batch
                        // several different requests' jobs together and so
                        // has no single `request_id` of its own.
                        let request_id = self.requests[i].id;
                        let _span = tracing::info_span!(
                            "ignis.prefill",
                            request_id,
                            chunk_tokens = job.tokens.len() as u64,
                            start_position = job.start_position,
                        )
                        .entered();
                        // P4-07, GitHub #125: charges resident_slots +
                        // kv_pages exactly once per materialization — a
                        // no-op if `i` is already resident (every chunk
                        // after its first, or a restored-to-Prefilling
                        // request continuing from its snapshot boundary).
                        self.materialize(i);
                        let r = &mut self.requests[i];
                        r.prefill_failures = 0;
                        if r.state == RequestState::Admitted {
                            r.advance(RequestState::Prefilling);
                        }
                        r.prefill_progress += job.tokens.len() as u32;
                        // GitHub #238: a decision's answer, off the one
                        // chunk that asked for it. Held on the request
                        // rather than emitted here, because a decision
                        // finishes at the end of this phase and its readout
                        // rides that finish event — there is nothing else
                        // for it to ride.
                        if outcome.readout.is_some() {
                            r.readout = outcome.readout.clone();
                        }
                        if outcome.attention.is_some() {
                            r.attention = outcome.attention.clone();
                        }
                        // P3-01 / ADR 0018: every completed chunk boundary
                        // is a GDN resumable boundary, whether or not it
                        // is this request's last chunk.
                        let new_position = r.prefill_progress as usize;
                        r.checkpoint(new_position);
                        // P3-06: the request log's per-phase fields (prefill
                        // chunks consumed, prefilled tokens) are counted
                        // from this event, not read back off `Request` —
                        // the request struct itself carries no history, only
                        // its current cumulative progress.
                        events.push(SchedEvent::PrefillChunk {
                            request: request_id,
                            chunk_tokens: job.tokens.len() as u32,
                            prefilled_tokens: r.prefill_progress,
                            encode_micros: outcome.encode_micros,
                        });
                        // core-07 — registration, driven by the job that
                        // actually published (P4-10, GitHub #126). The leaf
                        // publishes at the chunk boundary the job named,
                        // because what a claimant clones is the mutable
                        // state at the prefix's *end*; registering at the
                        // end of the prompt instead would cache pages whose
                        // state the registrant had already run past.
                        //
                        // Reading `job.publish_prefix` rather than
                        // re-deriving the condition is what keeps the two
                        // sides of one act from disagreeing — and it is why
                        // the `None` arm below can be sure a leaf prefix
                        // exists to let go of.
                        //
                        // The entry's pages are *not* separately charged
                        // here (P4-07, GitHub #125 changed this): the
                        // publisher's materialization already charged its
                        // *whole* reservation, entry pages included, before
                        // this ever ran (`Self::materialize`, above). This
                        // is bookkeeping only — the charge split, so the
                        // publisher's later release subtracts just its
                        // residual, and the entry's own release
                        // (`Self::release_prefix_claim`) subtracts the
                        // rest when its last claimant is gone.
                        if let Some(published) = job.publish_prefix.map(|p| p.tokens) {
                            let publisher = self.requests[i].id;
                            // Registered over exactly the head the *leaf*
                            // published, not over the whole prompt. The two
                            // were the same number until GitHub #186 floored
                            // the publish point to the generation opener's
                            // page rather than the prompt's; registering the
                            // prompt's head now would cache an entry claiming
                            // more warm history than the leaf's prefix
                            // actually holds, and a claimant would skip
                            // prefill for tokens nothing warmed.
                            // GitHub #193: with the images inside it, so a
                            // sibling sending another picture never matches.
                            let media = media_keys(&self.requests[i].input);
                            let head = PromptContent::new(&self.requests[i].input.tokens, &media)
                                .head(published);
                            // GitHub #187: a request that resumed from
                            // retained state publishes a **chained** entry —
                            // the pages it warmed past what it claimed, over
                            // the claim it was already holding. Its own
                            // reference on the parent becomes the child's, so
                            // `prefix_entry` moves rather than doubling up.
                            let parent = self.requests[i].prefix_entry;
                            let registered = self.prefix.register(
                                publisher,
                                head,
                                &self.requests[i].gdn,
                                parent,
                            );
                            match registered {
                                Some((entry, pages)) => {
                                    let r = &mut self.requests[i];
                                    r.prefix_entry = Some(entry);
                                    r.prefix_publisher = Some(publisher);
                                    // GitHub #186: from here the publisher's
                                    // own first KV page is the one past the
                                    // whole head, so its generation opener —
                                    // which the publish point was floored to
                                    // — falls inside a page it alone writes.
                                    // That is what makes a checkpoint
                                    // capturable at all
                                    // (`Request::checkpoint_point`), and
                                    // GitHub #187 is exactly this line
                                    // reaching a claimant: `pages` counts the
                                    // chain, `shared_pages` grows past what
                                    // the request resumed from, and the
                                    // capture that was refused becomes legal.
                                    //
                                    // `pages` is the entry's *own* pages,
                                    // which is exactly the charge split this
                                    // publish moves: whatever the request was
                                    // already standing on it had already
                                    // handed over when it claimed it.
                                    r.shared_pages += pages;
                                    r.resources.kv_pages =
                                        r.resources.kv_pages.saturating_sub(pages);
                                    // GitHub #188 (ADR 0029): a prefix
                                    // published at or below the system block
                                    // boundary is **retained** — it does not
                                    // drop when its last live claimant goes,
                                    // so the next subagent of the burst claims
                                    // it although its sibling finished. Asked
                                    // of the request rather than re-derived,
                                    // so the head that was published and the
                                    // head that is retained cannot disagree.
                                    //
                                    // `<=`, not `==`: a prompt shorter than
                                    // its own block publishes the whole pages
                                    // it has, which is a prefix *of* the block
                                    // and reusable by the same burst. What
                                    // must never be retained is a head reaching
                                    // *past* the block, since that is the
                                    // request's own conversation and no
                                    // sibling shares it.
                                    let block = self.requests[i]
                                        .retained_prefix_point(self.config.kv_page_tokens);
                                    if self.config.prompt_reuse
                                        && block > 0
                                        && published <= block
                                    {
                                        let retention = Retention {
                                            at: self.tick,
                                            class: self.requests[i].class,
                                            used_at: self.now(),
                                        };
                                        let took = self.prefix.retain_published(entry, retention);
                                        // `retain_published` refuses an entry
                                        // that is gone or already retained,
                                        // and `entry` is neither: `register`
                                        // minted it a line ago, and a fresh
                                        // id carries no retention. Asserted
                                        // rather than dropped, because a
                                        // silent `false` here is a prefix
                                        // that quietly stopped outliving its
                                        // publisher — which no test of a
                                        // *later* request could tell from an
                                        // ordinary miss.
                                        debug_assert!(
                                            took,
                                            "a freshly registered prefix is always retainable"
                                        );
                                    }
                                }
                                // The leaf published and this cache declined
                                // — the head is already registered by
                                // someone else, or the GDN position is not a
                                // reusable boundary. Nothing will ever claim
                                // that prefix, and no entry exists to release
                                // it later, so the backend's handle is
                                // dropped now. Its pages stay out of the pool
                                // only until the publishing sequence itself
                                // is released, which is the same lifetime
                                // they would have had unshared — its retained
                                // slot comes back now.
                                None => self.release_prefix_handle(publisher, published),
                            }
                        }
                        // GitHub #186 — the prompt checkpoint, driven by the
                        // job that actually captured one, for the same reason
                        // the publish above is: reading the job rather than
                        // re-deriving the condition is what keeps the two
                        // sides of one act from disagreeing. `false` here is
                        // a backend that declined the bet; nothing is
                        // recorded, the slot goes back (GitHub #215), and the
                        // request is none the wiser.
                        if let Some(capture) = job.capture_checkpoint {
                            if outcome.checkpoint_captured {
                                self.retain_checkpoint(i, capture.tokens, &mut events);
                            } else {
                                self.retained.give_back(RetainedHolder::Checkpoint {
                                    publisher: request_id,
                                });
                            }
                        }
                        // The reuse this request's prefill actually landed
                        // (GitHub #186): reported here rather than where the
                        // claim was decided, so a claim whose batch failed is
                        // reported when its retry succeeds and never twice,
                        // and so the restore's measured cost can ride along.
                        if let Some(claim) = job.checkpoint {
                            events.push(SchedEvent::StateReused {
                                request: request_id,
                                source: claim.source,
                                tokens: claim.tokens,
                                restore_micros: outcome.restore_micros,
                                // The claim on a prefill job is a prompt
                                // checkpoint's; a shared prefix a request
                                // stands on is reported as `PrefixReused`.
                                kind: RetainedKind::Checkpoint,
                            });
                            events.push(SchedEvent::RetainedState {
                                operation: RetainedStateOperation::Restore,
                                source: claim.source,
                                kind: RetainedKind::Checkpoint,
                            });
                            self.release_kv_ram_claim(i, true);
                        }
                        let misses = std::mem::take(&mut self.requests[i].pending_retained_misses);
                        for source in misses.iter() {
                            // The checkpoint pool's lookup is the only one
                            // that reports a miss (GitHub #216): the prefix
                            // walk beside it records none, so every miss is a
                            // checkpoint miss, and none is invented for the
                            // prefix side to make the two look symmetric.
                            events.push(SchedEvent::RetainedState {
                                operation: RetainedStateOperation::Miss,
                                source,
                                kind: RetainedKind::Checkpoint,
                            });
                        }
                    }
                }
                Err(e) => {
                    // A failed chunk leaves progress untouched: the batch
                    // stays in its pre-call state (`Admitted`, or
                    // `Prefilling` with its prior progress) and is retried
                    // on the next advance with the *same* span — not a
                    // span already applied; the fault is surfaced through
                    // `last_error`.
                    //
                    // P4-07, GitHub #125: `RuntimeCompute::prefill_step`'s
                    // own documented failure behavior releases *every* job's
                    // sequence on any error in the batch, continuing ones
                    // included, not only freshly-allocated ones — so every
                    // request in this batch un-charges its resident-slot +
                    // KV-page reservation here (a no-op for one that was
                    // never materialized, e.g. a fresh candidate whose very
                    // first `allocate_sequence` is what failed). The next
                    // successful chunk re-materializes and re-charges it
                    // (`Self::materialize`, above).
                    for &i in &batch {
                        self.unmaterialize(i);
                        self.requests[i].prefill_failures += 1;
                    }
                    // GitHub #215: the backend unwound the batch's publishes
                    // and took no capture, so every slot this batch was given
                    // comes back; the retry takes them again.
                    for job in &jobs {
                        if let Some(publish) = job.publish_prefix {
                            self.retained.give_back(RetainedHolder::Prefix {
                                publisher: job.request,
                                tokens: publish.tokens,
                            });
                        }
                        if job.capture_checkpoint.is_some() {
                            self.retained.give_back(RetainedHolder::Checkpoint {
                                publisher: job.request,
                            });
                        }
                    }
                    // GitHub #166: a failure that repeats is not transient.
                    // Retried every advance, it spun the model thread and
                    // logged the leaf's error ~100k times a second. After
                    // `MAX_PREFILL_ATTEMPTS` in a row the request ends with
                    // `FinishReason::Error`. Every request in the failed
                    // batch is charged the attempt: the backend's error does
                    // not say which job failed.
                    for &i in &batch {
                        if self.requests[i].prefill_failures >= MAX_PREFILL_ATTEMPTS {
                            self.mark_done(i, &mut events, FinishReason::Error);
                        }
                    }
                    self.last_error = Some(e);
                    self.report_retained_slots(&mut events);
                    return events;
                }
            }
        }

        // GitHub #238, ADR 0034 — a **decision** ends where its prefill
        // ends. It is finished here, between the prefill phase and the lane
        // deal, which is the only place that means what it says: one step
        // later it would be a candidate in `run_admission`'s queue, and
        // being a candidate is the thing it must never be. It never enters
        // `Running`, never holds a decode lane, emits no token, and carries
        // its readout out on the finish event.
        //
        // The placement is load-bearing for ADR 0004, not only for tidiness.
        // A decision's `remaining_work` is 0 — it has no tokens to generate
        // — so it would satisfy `run_admission`'s temporal-backfill test
        // (`remaining_work <= frontier && <= temporal_credit`) trivially and
        // spend no credit doing it, taking a protected lane away from the
        // conversation the protection was opened for. Emptying it from the
        // candidate set here is what makes that unreachable.
        //
        // `FinishReason::Stop` because the decision is *answered* — nothing
        // was cut short. Its `tokens` is 0, honestly: it generated nothing.
        let decided: Vec<usize> = self
            .requests
            .iter()
            .enumerate()
            .filter(|&(_, r)| {
                r.state == RequestState::Prefilling
                    && r.prefill_complete()
                    && r.input.is_decision()
            })
            .map(|(i, _)| i)
            .collect();
        for idx in decided {
            // A decision that finished its prefill with no readout should be
            // impossible: its last chunk always carries at least one token
            // (`RequestInput::reuse_reach`) and always asks for one, and a
            // backend that cannot answer fails the job outright
            // (`scheduler::READOUT_WITHOUT_TOKENS`). If it happens anyway,
            // the one thing this must not do is report `Stop` with an empty
            // answer — that is a decision answered by nothing, dressed as a
            // decision answered. It ends with `Error`, which is what a
            // request that could not be served ends with everywhere else.
            //
            // GitHub #260: a head point is answered by its attention
            // readout, and *that* one can be missing without anything being
            // wrong upstream — a leaf that could not read the keys the
            // layer's attention read returns none, and the question fails
            // here rather than being pointed from some other copy of them.
            let request = &self.requests[idx];
            let answered = match &request.input.decision {
                Some(DecisionRead::Attention(_)) => request.attention.is_some(),
                _ => request.readout.is_some(),
            };
            let reason = match answered {
                true => FinishReason::Stop,
                false => {
                    // hotpath-lint-allow: failure-only path (a decision that cannot be answered ends here), reviewed exception (GitHub #238).
                    tracing::error!(
                        name: "ignis.decision.no_readout",
                        request_id = self.requests[idx].id,
                        prompt_tokens = self.requests[idx].input.tokens.len(),
                        "a decision completed its prefill without reading out"
                    );
                    FinishReason::Error
                }
            };
            self.mark_done(idx, &mut events, reason);
        }

        // Phase 2 — the admission state machine drives the lane deal
        // (core-05; see `run_admission`). Restored (suspended) requests
        // take a free lane before a fresh prefill (core-06: a sibling
        // request restores instead of re-prefilling), so the restore pass
        // runs first.
        self.restore_pass(&mut events);
        self.run_admission(&mut events);

        // Phase 3 — batched decode: one compute call spanning every
        // running lane (lane-ascending order for a deterministic
        // stream). A request that has used up its reservation (its
        // remaining work hit zero) completes in this step: the KV
        // reservation is a hard cap (no OOM under the N=8 load,
        // core-01), so the request stops at the cap even when the
        // backend would keep generating.
        let mut running: Vec<usize> = self
            .requests
            .iter()
            .enumerate()
            .filter_map(|(i, r)| (r.state == RequestState::Running).then_some(i))
            .collect();
        running.sort_by_key(|&i| self.requests[i].lane);
        for &i in &running {
            if self.requests[i].remaining_work == 0 {
                // The reservation cap was reached (core-05): complete
                // now, releasing the lane and the reservation.
                let reason = self.budget_spent_reason(i);
                self.mark_done(i, &mut events, reason);
            }
        }
        let to_decode: Vec<usize> = running
            .into_iter()
            .filter(|&i| self.requests[i].remaining_work > 0)
            .collect();
        if !to_decode.is_empty() {
            let jobs: Vec<DecodeJob> = to_decode
                .iter()
                .map(|&i| DecodeJob {
                    request: self.requests[i].id,
                    lane: self.requests[i].lane.expect("running requests hold a lane"),
                    params: self.requests[i].input.params,
                    remaining_tokens: self.requests[i].remaining_work.min(u32::MAX as u64) as u32,
                    // GitHub #242: the set for the draw this round makes,
                    // which is the step *after* the one it emits — the
                    // prefill drew step 0, so a request that has emitted `n`
                    // tokens draws step `n + 1` here. The last round asks for
                    // a step past the end and gets `None`: its draw is
                    // discarded with the sequence.
                    permitted: self.requests[i].input.constrained.as_ref().and_then(|schedule| {
                        schedule.step(self.requests[i].tokens as usize + 1)
                    }),
                })
                .collect();
            match self.compute.decode_step(&jobs) {
                Ok(results) => {
                    for (&i, res) in to_decode.iter().zip(&results) {
                        // GitHub #81 / ADR 0012: the decode-round span —
                        // one per request per `decode_step` call (the unit
                        // a decode CUDA graph is captured over), opened
                        // here rather than around the call above, since one
                        // call spans every running lane and so has no
                        // single `request_id` of its own. `round = self.tick`
                        // is this `advance()` call's own tick counter — a
                        // stable, monotonic round number. Never subdivided
                        // further: a future MTP verify step attaches its
                        // own child span here, still one per request per
                        // round, never per verified token.
                        let request_id = self.requests[i].id;
                        let _span = tracing::info_span!(
                            "ignis.decode.round",
                            request_id,
                            round = self.tick,
                        )
                        .entered();
                        let DecodeOutcome {
                            tokens,
                            probabilities,
                            finish,
                            spec,
                        } = res;
                        debug_assert!(
                            !tokens.is_empty() || finish.is_some(),
                            "a decode outcome commits a token or finishes the request"
                        );
                        if let Some(spec) = spec {
                            let r = &mut self.requests[i];
                            r.spec = Some(r.spec.unwrap_or_default() + *spec);
                        }
                        // P5-06 (GitHub #154): the round committed a run,
                        // emitted one event per token, in order.
                        let mut run_truncated_by_budget = false;
                        // GitHub #255: a schedule may name a token that ends
                        // its run. Checked here, where the tokens the round
                        // committed are read, because that is the only place
                        // that sees *which* token was drawn — the budget
                        // below sees only how many.
                        let mut terminated = false;
                        for (n, &token) in tokens.iter().enumerate() {
                            // GitHub #242: the probability this token held
                            // inside its own step's set, which only the
                            // backend could pair with it (the round that
                            // drew it is not the round that returned it).
                            // Absent on every unconstrained lane.
                            if let Some(&probability) = probabilities.get(n) {
                                self.requests[i]
                                    .drawn
                                    .push(crate::constrained::Draw { token, probability });
                            }
                            self.requests[i].tokens += 1;
                            // Service-work decay (core-05): one quantum per
                            // generated token.
                            self.requests[i].remaining_work =
                                self.requests[i].remaining_work.saturating_sub(1);
                            events.push(SchedEvent::Token { request: request_id, token });
                            // The terminator is emitted like any other token
                            // — it was generated, and `output_tokens` that
                            // hid it would under-report the round this run
                            // actually cost — and then the run is over.
                            if self.requests[i]
                                .input
                                .constrained
                                .as_ref()
                                .is_some_and(|schedule| schedule.ends_on(token))
                            {
                                terminated = true;
                                break;
                            }
                            // The reservation cap: nothing past the final
                            // reserved token is emitted, whatever the
                            // backend committed.
                            if self.requests[i].remaining_work == 0 {
                                run_truncated_by_budget = n + 1 < tokens.len();
                                break;
                            }
                        }
                        if !tokens.is_empty() {
                            // core-06: record a GDN checkpoint at the new
                            // position (the host tier may snapshot the
                            // request at this boundary — the GDN state is
                            // resumable there). Once per round, at the end
                            // of the run: the positions inside it are not
                            // boundaries the device state ever stood on. The
                            // position is absolute over the whole sequence
                            // (prompt + generated), not just the decode-token
                            // count: P3-01's chunked prefill now checkpoints
                            // at real prompt positions too
                            // (`GdnState::checkpoint` only accepts a position
                            // `>=` the current one), so a decode checkpoint
                            // that restarted counting from 0 would be
                            // silently dropped the instant it fell behind the
                            // prompt length.
                            let new_pos = self.requests[i].input.tokens.len()
                                + self.requests[i].tokens as usize;
                            self.requests[i].checkpoint(new_pos);
                        }
                        match finish {
                            // The run closed its own shape: the schedule may
                            // have steps left and none of them will be
                            // asked for.
                            _ if terminated => {
                                self.mark_done(i, &mut events, FinishReason::Stop)
                            }
                            // Finished (EOS or the backend's own `max_tokens`
                            // enforcement) after the whole run: Done, lane
                            // released.
                            Some(reason) if !run_truncated_by_budget => {
                                self.mark_done(i, &mut events, *reason)
                            }
                            // The request completes on its final reserved
                            // token.
                            _ if self.requests[i].remaining_work == 0 => {
                                let reason = self.budget_spent_reason(i);
                                self.mark_done(i, &mut events, reason)
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => {
                    // A decode or verify failure has no safe retry: every
                    // request in this leaf batch still holds a live lane,
                    // so retrying it on the next tick would spin the model
                    // thread and keep that lane unavailable forever.
                    for &i in &to_decode {
                        tracing::debug!(
                            request_id = self.requests[i].id,
                            error = %e,
                            "ignis.decode.error"
                        );
                        self.mark_done(i, &mut events, FinishReason::Error);
                    }
                    self.last_error = Some(e);
                }
            }
        }

        // Phase 4 — restore (core-06): lanes freed by this step's
        // completions (and the eviction above) go to suspended (host-
        // tier) requests before a fresh prefill — a sibling request
        // restores instead of re-prefilling.
        self.restore_pass(&mut events);

        self.report_retained_slots(&mut events);
        events
    }

    fn is_idle(&self) -> bool {
        self.in_flight() == 0
    }

    fn model_id(&self) -> &str {
        &self.config.model
    }

    fn max_sequence_tokens(&self) -> u32 {
        self.config.max_sequence_tokens
    }

    fn mode(&self) -> EngineMode {
        if self.in_flight() > 0 {
            EngineMode::Serving
        } else {
            EngineMode::Idle
        }
    }

    /// Three field reads (GitHub #216). `kv_used_pages` is the counter the
    /// admission machine keeps as it reserves and releases, so the step that
    /// releases the last request's pages is also the step that reports the
    /// release — there is no later step to report it on.
    fn occupancy(&self) -> Occupancy {
        Occupancy {
            kv_used_pages: self.kv_used_pages,
            kv_pool_pages: self.capacity.kv_pages,
            kv_ram_used_bytes: self.host.used_bytes(),
        }
    }

    /// The backend's own answer (GitHub #260): the artifact its leaf opened.
    fn artifact(&self) -> crate::identity::ArtifactHash {
        self.compute.blob_identity().artifact
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn::GdnState;
    use crate::host::{HostEntry, ResumePhase, RetainedKvRamEntry, Tier};
    use crate::mock::MockCompute;

    /// A live snapshot of `bytes`, captured by `owner` at tick `tick`. The
    /// fields the host tier's own order reads are the only ones that matter
    /// here; the rest is a well-formed resting request.
    fn entry(request: RequestId, bytes: u64, owner: RequestClass, tick: u64) -> HostEntry {
        let mut gdn = GdnState::new();
        gdn.checkpoint(0);
        HostEntry {
            request,
            resume_phase: ResumePhase::Running,
            lane: Some(0),
            owner,
            pages: 1,
            bytes,
            tokens: 0,
            prefill_progress: 0,
            remaining_work: 8,
            gdn,
            tier: Tier::Probation,
            use_tick: tick,
        }
    }

    /// A tier and an arena of four bytes each, holding three one-byte
    /// snapshots placed in order — so the arena is
    /// `[r1][r2][r3][free]` and the ledger reads 3 of 4 used.
    ///
    /// `r2`, in the middle, is the Agent: the tier discards by class first
    /// (ADR 0023), so it is the first victim, and giving it up leaves the
    /// free bytes in two one-byte holes rather than one span of two.
    fn three_blobs() -> (ConcreteScheduler, Arc<MockCompute>) {
        let compute = Arc::new(MockCompute::with_host_arena(4));
        let mut sched = ConcreteScheduler::with_config(
            SchedulerConfig {
                host_capacity_bytes: 4,
                ..SchedulerConfig::default()
            },
            compute.clone(),
        );
        for (request, owner) in [
            (1, RequestClass::Interactive),
            (2, RequestClass::Agent),
            (3, RequestClass::Interactive),
        ] {
            compute.evict(request).expect("the arena has room for a byte");
            sched
                .host
                .capture(entry(request, 1, owner, request))
                .expect("the tier has room for a byte");
        }
        assert_eq!(sched.host.used_bytes(), 3);
        assert_eq!(compute.host_arena_used(), 3);
        (sched, compute)
    }

    #[test]
    fn making_room_keeps_going_until_a_hole_fits_not_until_the_bytes_do() {
        let (mut sched, compute) = three_blobs();
        let mut events = Vec::new();

        assert!(
            sched.make_host_room_for_bytes(2, &mut events),
            "two of the four bytes can be freed"
        );

        // The Agent in the middle goes first and takes the ledger to 2 of 4,
        // which is where a byte budget alone would stop -- and the arena
        // would still have nowhere to put a two-byte blob. Room means a hole.
        assert!(
            compute.host_blob_fits(2),
            "the loop stopped on free bytes, not on a free span"
        );
        assert_eq!(
            sched.host.used_bytes(),
            1,
            "and stopped at the first hole that fits, not by emptying the tier"
        );
        assert_eq!(
            sched.host.used_bytes(),
            compute.host_arena_used(),
            "the ledger and the arena agree after every step"
        );
    }

    /// GitHub #224 — every live snapshot the tier gives up raises its own
    /// `SnapshotDropped`, one per victim. `Requeued` is absent here because
    /// these blobs have no request behind them in this fixture, which is the
    /// point: the tier evicted whether or not anyone was left to re-queue,
    /// so the two facts cannot be the same fact.
    #[test]
    fn every_live_snapshot_the_tier_gives_up_raises_its_own_drop() {
        let (mut sched, _compute) = three_blobs();
        let mut events = Vec::new();

        assert!(sched.make_host_room_for_bytes(2, &mut events));

        let dropped: Vec<RequestId> = events
            .iter()
            .filter_map(|e| match e {
                SchedEvent::SnapshotDropped { request } => Some(*request),
                _ => None,
            })
            .collect();
        assert_eq!(
            dropped.len(),
            2,
            "the two blobs the tier gave up each raised a drop: {events:?}"
        );
        assert_eq!(sched.host.entry_count(), 1, "and only those two left");
    }

    /// The refusal path gives up *everything* and still says no — and still
    /// reports every departure, so a tier that emptied itself for nothing is
    /// visible rather than silent (GitHub #224).
    #[test]
    fn a_refused_blob_still_reports_what_the_tier_gave_up_trying() {
        let (mut sched, _compute) = three_blobs();
        let mut events = Vec::new();

        assert!(!sched.make_host_room_for_bytes(5, &mut events));

        let drops = events
            .iter()
            .filter(|e| matches!(e, SchedEvent::SnapshotDropped { .. }))
            .count();
        assert_eq!(drops, 3, "all three blobs left the tier: {events:?}");
    }

    #[test]
    fn a_blob_no_victim_order_can_fit_leaves_the_tier_empty_and_refuses() {
        let (mut sched, compute) = three_blobs();
        let mut events = Vec::new();

        assert!(
            !sched.make_host_room_for_bytes(5, &mut events),
            "nothing the tier can give up makes a five-byte hole in a four-byte arena"
        );
        assert_eq!(sched.host.entry_count(), 0, "it gave up everything trying");
        assert_eq!(sched.host.used_bytes(), 0);
        assert_eq!(compute.host_arena_used(), 0, "and every blob went back to the arena");
    }

    /// Three spilled checkpoints of a byte each, placed in order in a
    /// four-byte arena, with the middle one the Agent — the first the tier
    /// gives up, and the one whose byte leaves the free bytes in two holes
    /// rather than one span.
    fn three_retained_blobs() -> (ConcreteScheduler, Arc<MockCompute>) {
        let compute = Arc::new(MockCompute::with_host_arena(4));
        let mut sched = ConcreteScheduler::with_config(
            SchedulerConfig {
                host_capacity_bytes: 4,
                ..SchedulerConfig::default()
            },
            compute.clone(),
        );
        let used_at = sched.now();
        for (publisher, owner) in [
            (1, RequestClass::Interactive),
            (2, RequestClass::Agent),
            (3, RequestClass::Interactive),
        ] {
            compute
                .spill_checkpoint(publisher)
                .expect("the arena has room for a byte");
            sched
                .host
                .capture_retained(RetainedKvRamEntry::new(
                    RetainedBlob::Checkpoint(publisher),
                    publisher,
                    owner,
                    1,
                    used_at,
                ))
                .expect("the tier has room for a byte");
        }
        assert_eq!(sched.host.used_bytes(), 3);
        assert_eq!(compute.host_arena_used(), 3);
        (sched, compute)
    }

    #[test]
    fn a_retained_spill_keeps_taking_victims_until_one_of_the_holes_fits() {
        let (mut sched, compute) = three_retained_blobs();
        let used_at = sched.now();
        let mut events = Vec::new();

        assert!(
            sched.make_kv_ram_room_for_retained(2, RequestClass::Interactive, used_at, &mut events),
            "two of the four bytes belong to entries this newcomer outranks"
        );

        // The byte plan needs one byte and takes the Agent in the middle,
        // which leaves two free bytes in two holes. Stopping there is what
        // the budget alone would do, and the blob would have nowhere to go.
        assert!(compute.host_blob_fits(2), "the spill has a span to be placed in");
        assert_eq!(sched.host.used_bytes(), 1, "and it took one more, not everything");
        assert_eq!(sched.host.used_bytes(), compute.host_arena_used());
    }

    #[test]
    fn a_retained_spill_no_entry_below_it_can_place_does_not_happen() {
        let (mut sched, compute) = three_retained_blobs();
        let used_at = sched.now();
        let mut events = Vec::new();

        // An Agent's bet: the two Interactive entries rank above it, so the
        // only byte it may take is the other Agent's — which leaves two
        // one-byte holes and nowhere to put two bytes.
        assert!(
            !sched.make_kv_ram_room_for_retained(2, RequestClass::Agent, used_at, &mut events),
            "a bet does not displace what the tier values more, however fragmented"
        );
        assert_eq!(
            sched.host.retained_count(),
            2,
            "the two it may not take are still there"
        );
        assert_eq!(sched.host.used_bytes(), compute.host_arena_used());
    }
}
