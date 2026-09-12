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
//! [`SubmitError::Oversized`].
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
use std::time::Instant;

use crate::admission::{
    ActiveAdmissionSnapshot, AdmissionProtection, AdmissionResources, ProtectionPhase,
    ResidentCandidate, RetainedLaneCandidate, admission_resources_fit,
    choose_resident_candidate_victim, choose_retained_lane_victim, make_admission_protection,
    persistent_backfill_is_safe, protected_head_safe_without_temporal, protection_frontier_distance,
};
use crate::host::{HostEntry, HostTier, ResumePhase, Tier};
use crate::prefix::{PrefixCache, PrefixId};
use crate::request::Request;
use crate::scheduler::{
    Compute, DecodeJob, DecodeOutcome, PrefillJob, Scheduler, SharedPrefixClaim,
};
use crate::types::{
    BackfillClass, ComputeError, DecodeParams, EngineMode, FinishReason, LaneId, N_DECODE_LANES,
    RequestClass, RequestId, RequestInput, RequestState, SchedEvent, SubmitError,
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
    /// The per-request decode reservation cap (effective max tokens) for
    /// requests submitted without an explicit `max_tokens` (core-05: an
    /// unbounded request reserves `ceil((prompt + this) / kv_page_tokens)`
    /// pages and is completed when it reaches it — the reservation cannot
    /// grow mid-generation).
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
}

/// The serving prefill chunk width's default, in tokens (ADR 0018): the
/// measured 1,024-token chunk from the G2 gate run (`.scratch/ROADMAP.md`).
/// P3-01 is CPU-only (driven through the `Compute` seam with `MockCompute`),
/// so this is the value CPU tests and this config's [`Default`] use; wiring
/// a real `--features cuda` load's actual reserved width through
/// [`resolve_serving_chunk_tokens`] is `crates/runtime`'s job, not this
/// crate's.
pub const DEFAULT_SERVING_CHUNK_TOKENS: u32 = 1024;

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
            host: HostTier::new(config.host_capacity_bytes),
            tick: 0,
            prefix: PrefixCache::new(config.kv_page_tokens),
            config,
            compute,
            next_id: 0,
            requests: Vec::new(),
            free_lanes: (0..N_DECODE_LANES).collect(),
            last_error: None,
        }
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
    fn available_capacity(&self) -> AdmissionResources {
        AdmissionResources {
            lanes: self.capacity.lanes,
            kv_pages: self
                .capacity
                .kv_pages
                .saturating_sub(self.prefix.pinned_pages()),
            backend_pages: self.capacity.backend_pages,
            resident_slots: self.capacity.resident_slots,
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
        let Some((freed, publisher)) = self.prefix.release(entry) else {
            return;
        };
        self.kv_used_pages = self.kv_used_pages.saturating_sub(freed);
        // P4-10 (GitHub #126): the entry is gone from this cache, so the
        // backend's own handle on the leaf's prefix goes too. The leaf's
        // pages come back when its last *sequence* holder is released, which
        // is why this is a handle drop and not a free.
        self.compute.release_prefix(publisher);
    }

    /// The last hard compute error the most recent advance reported, if
    /// any.
    ///
    /// A failed step is not swallowed: it emits no events, the state is
    /// left retryable (a failed prefill leaves the batch in `Admitted`, a
    /// failed decode leaves the lanes running), and the next advance
    /// retries the same step. A successful advance clears it; callers
    /// poll this to surface the fault.
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
        let (request_id, tokens) = self.release_request(idx);
        events.push(SchedEvent::Done {
            request: request_id,
            tokens,
            reason,
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
    /// request — core-07), and any request holding (or holding open) a
    /// shared prefix at all (P4-10, GitHub #126): its sequence cannot be
    /// snapshotted (`IGNIS_SEQ_ERR_SHARED_PREFIX`) since its leading pages
    /// are not its own — excluded here, upstream of victim selection,
    /// rather than picked and then failed at snapshot time.
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
            .filter(|r| r.prefix_entry.is_none())
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
    /// fully-prefilled one still queued for a lane deal. Excludes a
    /// request holding (or holding open) a shared prefix, for the same
    /// reason [`Self::retained_lane_candidates`] does, and `exclude` (a
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
                    && r.prefix_entry.is_none()
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
        let r = &mut self.requests[idx];
        // core-07: capture the shared-prefix claim (the `requeue()` below
        // resets it; a re-queued request re-prefills from the start and
        // may re-claim a live entry on its fresh prefill).
        let prefix_entry = r.prefix_entry;
        r.requeue(); // Evicted → Admitted, lane released (there is none).
        r.tokens = 0;
        let effective_max = r
            .input
            .params
            .max_tokens
            .unwrap_or(self.config.max_sequence_tokens);
        r.remaining_work = effective_max as u64;
        r.backfill_class = BackfillClass::None;
        r.backfill_epoch = 0;
        // core-07: restore the full (unshrunk) reservation — the re-queued
        // request re-prefills its *entire* prompt (not just its tail), so
        // its pool charge must cover `prompt + max` pages again (the claim
        // loop shrinks it to the tail if a live entry is re-claimed).
        let full_pages = ((r.input.tokens.len() as u64) + (effective_max as u64))
            .div_ceil(self.config.kv_page_tokens as u64)
            .min(u32::MAX as u64) as u32;
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
    fn make_host_room_for_bytes(&mut self, bytes: u64, events: &mut Vec<SchedEvent>) -> bool {
        while self.host.used_bytes() + bytes > self.host.capacity_bytes() {
            match self.host.evict_one() {
                Some(discarded) => {
                    self.compute.discard_snapshot(discarded.request);
                    if let Some(idx) = self.requests.iter().position(|r| r.id == discarded.request)
                    {
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
        let (v_id, v_class, v_pages, v_tokens, v_progress, v_work, v_gdn) = {
            let v = &self.requests[v_idx];
            (
                v.id,
                v.class,
                v.resources.kv_pages,
                v.tokens,
                v.prefill_progress,
                v.remaining_work,
                v.gdn.clone(),
            )
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
            pages: v_pages,
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
            let context_tokens = ((self.requests[idx].input.tokens.len() as u64)
                .saturating_add(
                    self.requests[idx]
                        .input
                        .params
                        .max_tokens
                        .unwrap_or(self.config.max_sequence_tokens) as u64,
                )
                .min(u32::MAX as u64)) as u32;
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
        input: RequestInput,
        class: RequestClass,
    ) -> Result<RequestId, SubmitError> {
        if input.model != self.config.model {
            return Err(SubmitError::UnknownModel(input.model));
        }
        if self.in_flight() >= self.config.max_in_flight {
            return Err(SubmitError::Full);
        }
        // core-05: compute the request's KV reservation (prompt + the
        // effective token budget, in pages) and reject requests that can
        // never fit — even alone (they would block the queue forever).
        let effective_max = input
            .params
            .max_tokens
            .unwrap_or(self.config.max_sequence_tokens);
        let reserved_tokens = (input.tokens.len() as u64).saturating_add(effective_max as u64);
        let kv_pages = ((reserved_tokens + self.config.kv_page_tokens as u64 - 1)
            / self.config.kv_page_tokens as u64)
            .min(u32::MAX as u64) as u32;
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
        let publish_tokens = self.prefix.shareable_head_tokens(input.tokens.len());
        let mut request = Request::new(id, class, input, resources, effective_max as u64);
        request.publish_tokens = publish_tokens;
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
                    if self.requests[i].prefix_entry.is_some() {
                        continue;
                    }
                    let claimed = self.prefix.claim(&self.requests[i].input.tokens);
                    if let Some(claim) = claimed {
                        let r = &mut self.requests[i];
                        r.prefix_entry = Some(claim.id);
                        r.prefix_publisher = Some(claim.publisher);
                        r.shared_prefix_tokens = claim.tokens;
                        r.prefill_progress = claim.tokens; // the shared head is already warm
                        r.gdn = claim.gdn; // core-02: resume at the shared boundary
                        // Shrink the claimant's own reservation by the
                        // shared prefix's pages (the entry now owns them —
                        // charged once, for every claimant). `ceil((prompt
                        // + max) / pt) - shared_pages` equals `ceil((tail +
                        // max) / pt)`: the shared head is page-aligned, so
                        // subtracting its whole pages is exact.
                        r.resources.kv_pages = r.resources.kv_pages.saturating_sub(claim.pages);
                        events.push(SchedEvent::PrefixReused {
                            request: r.id,
                            tokens: claim.tokens,
                        });
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
                if let Some(cut) = b.iter().position(|&i| {
                    let r = &self.requests[i];
                    (r.input.tokens.len() as u32 - r.prefill_progress)
                        > self.config.serving_chunk_tokens
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
        let publish_points: Vec<u32> = {
            let mut points: Vec<u32> = Vec::with_capacity(batch.len());
            for (n, &i) in batch.iter().enumerate() {
                let at = self.requests[i].publish_point();
                let head = &self.requests[i].input.tokens[..at as usize];
                let taken = batch[..n].iter().enumerate().any(|(m, &j)| {
                    points[m] == at && self.requests[j].input.tokens[..at as usize] == *head
                });
                points.push(if at > 0 && taken { 0 } else { at });
            }
            points
        };
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
                let mut take = remaining.min(self.config.serving_chunk_tokens);
                // P4-10 (GitHub #126): a request that will publish a prefix
                // is cut at its publish point, even mid-prompt. The leaf
                // hands a claimant the mutable state at the prefix's *end*,
                // and a chunk that overshot it would have moved that state
                // on — so the point is a scheduling decision, not a detail of
                // the publish call.
                let publish_at = publish_points[n];
                if publish_at > start && take > publish_at - start {
                    take = publish_at - start;
                }
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
                    context_tokens: ((r.input.tokens.len() as u64)
                        .saturating_add(
                            r.input
                                .params
                                .max_tokens
                                .unwrap_or(self.config.max_sequence_tokens)
                                as u64,
                        )
                        .min(u32::MAX as u64)) as u32,
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
                    // 0 for a request that publishes nothing, so this is
                    // exactly "the chunk that lands on the publish point,
                    // for a request that has one".
                    publish_prefix_tokens: (publish_at > 0 && start + take == publish_at)
                        .then_some(publish_at),
                }
            })
            .collect();
        if !jobs.is_empty() {
            match self.compute.prefill_step(&jobs) {
                Ok(()) => {
                    for (&i, job) in batch.iter().zip(&jobs) {
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
                        if r.state == RequestState::Admitted {
                            r.advance(RequestState::Prefilling);
                        }
                        r.prefill_progress += job.tokens.len() as u32;
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
                        });
                        // core-07 — registration, driven by the job that
                        // actually published (P4-10, GitHub #126). The leaf
                        // publishes at the chunk boundary the job named,
                        // because what a claimant clones is the mutable
                        // state at the prefix's *end*; registering at the
                        // end of the prompt instead would cache pages whose
                        // state the registrant had already run past.
                        //
                        // Reading `job.publish_prefix_tokens` rather than
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
                        if job.publish_prefix_tokens.is_some() {
                            let publisher = self.requests[i].id;
                            let registered = self.prefix.register(
                                publisher,
                                &self.requests[i].input.tokens,
                                &self.requests[i].gdn,
                            );
                            match registered {
                                Some((entry, pages)) => {
                                    let r = &mut self.requests[i];
                                    r.prefix_entry = Some(entry);
                                    r.prefix_publisher = Some(publisher);
                                    r.resources.kv_pages =
                                        r.resources.kv_pages.saturating_sub(pages);
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
                                // they would have had unshared.
                                None => self.compute.release_prefix(publisher),
                            }
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
                    }
                    self.last_error = Some(e);
                    return events;
                }
            }
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
                self.mark_done(i, &mut events, FinishReason::Length);
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
                        match res {
                            DecodeOutcome::Token(token) => {
                                self.requests[i].tokens += 1;
                                // Service-work decay (core-05): one
                                // quantum per generated token.
                                self.requests[i].remaining_work =
                                    self.requests[i].remaining_work.saturating_sub(1);
                                events.push(SchedEvent::Token {
                                    request: self.requests[i].id,
                                    token: *token,
                                });
                                // core-06: record a GDN checkpoint at the
                                // new position (the host tier may snapshot
                                // the request at this boundary — the GDN
                                // state is resumable there). The position is
                                // absolute over the whole sequence (prompt +
                                // generated), not just the decode-token
                                // count: P3-01's chunked prefill now
                                // checkpoints at real prompt positions too
                                // (`GdnState::checkpoint` only accepts a
                                // position `>=` the current one), so a
                                // decode checkpoint that restarted counting
                                // from 0 would be silently dropped the
                                // instant it fell behind the prompt length.
                                let new_pos = self.requests[i].input.tokens.len()
                                    + self.requests[i].tokens as usize;
                                self.requests[i].checkpoint(new_pos);
                                // The reservation cap: the request
                                // completes on its final reserved token.
                                if self.requests[i].remaining_work == 0 {
                                    self.mark_done(i, &mut events, FinishReason::Length);
                                }
                            }
                            DecodeOutcome::Finished(reason) => {
                                // Finished (EOS or the backend's own
                                // `max_tokens` enforcement): Done, lane
                                // released.
                                self.mark_done(i, &mut events, *reason);
                            }
                        }
                    }
                }
                Err(e) => self.last_error = Some(e),
            }
        }

        // Phase 4 — restore (core-06): lanes freed by this step's
        // completions (and the eviction above) go to suspended (host-
        // tier) requests before a fresh prefill — a sibling request
        // restores instead of re-prefilling.
        self.restore_pass(&mut events);

        events
    }

    fn is_idle(&self) -> bool {
        self.in_flight() == 0
    }

    fn model_id(&self) -> &str {
        &self.config.model
    }

    fn mode(&self) -> EngineMode {
        if self.in_flight() > 0 {
            EngineMode::Serving
        } else {
            EngineMode::Idle
        }
    }
}
