//! The concrete scheduler — `core-04` (N=8 decode lanes + batched
//! prefill) + `core-05` (the full admission state machine, ADR 0004).
//!
//! The v1 scheduler (`docs/design/ignis-v1.md` §2) driving the [`Compute`]
//! seam (production: the kernel leaf via FFI, tests: [`crate::mock::MockCompute`])
//! in three phases per advance:
//!
//! 1. **Batched prefill** — up to `max_prefill_batch` queued requests are
//!    grouped into ONE compute call (not one call per request) to
//!    saturate the GPU in prefill and cut burst TTFT. *An experiment to
//!    verify* — we may be compute-bound, in which case it is useless (a
//!    measure, not a guarantee; the 99% gate of ADR 0007 re-checks it on
//!    the GPU). **Sibling prefix reuse** (core-07) runs here: before the
//!    batch call, each request claims the longest cached prefix of its
//!    prompt (skipping the redundant prefill — its job carries only the
//!    tail); after a successful call, fresh requests register their now-
//!    warm prompt in the prefix cache for siblings. The shared entry's
//!    pages are charged to the pool once (the charge split), and the
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
//! 4. **KV-RAM host tier** (core-06) — the overflow path: when a request
//!    is blocked by the active set (all lanes / pages in use), the
//!    scheduler evicts a retained lane (driven by
//!    [`crate::admission::choose_retained_lane_victim`]) into the
//!    host-RAM KV tier ([`crate::host::HostTier`]), freeing its lane +
//!    pages so the blocked head can be dealt. The evicted request is
//!    suspended (not done) and **restored** to a lane (instead of
//!    re-prefilling) when a lane frees. The tier's two-tier eviction
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

use std::sync::Arc;

use crate::admission::{
    ActiveAdmissionSnapshot, AdmissionProtection, AdmissionResources, ProtectionPhase,
    RetainedLaneCandidate, admission_resources_fit, choose_retained_lane_victim,
    make_admission_protection, persistent_backfill_is_safe, protected_head_safe_without_temporal,
    protection_frontier_distance,
};
use crate::host::{HostEntry, HostTier, Tier};
use crate::prefix::{PrefixCache, PrefixId};
use crate::request::Request;
use crate::scheduler::{Compute, DecodeJob, PrefillJob, Scheduler};
use crate::types::{
    BackfillClass, ComputeError, EngineMode, LaneId, N_DECODE_LANES, RequestClass, RequestId,
    RequestInput, RequestState, SchedEvent, SubmitError,
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
    /// The KV-RAM host tier capacity in pages (core-06: the host-RAM
    /// budget for evicted (suspended) request snapshots; production
    /// auto-sizes this from host RAM, tests pass small values to drive
    /// contention).
    pub host_capacity_pages: u32,
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
            // Eight full sequences fit in host RAM by default: the
            // host-tier overflow budget matches the KV pool (core-06).
            host_capacity_pages: (N_DECODE_LANES * (8192 / 16)) as u32,
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
    /// Main-pool pages reserved by running requests (over-reservation:
    /// charged in full at deal, released at completion).
    kv_used_pages: u32,
    /// The active protection (core-05): `None` while no head is blocked
    /// (or once the protected head has been dealt).
    protection: Option<AdmissionProtection>,
    /// The next protection epoch (protections start at epoch 1).
    protection_epoch: u64,
    // ── core-06: the KV-RAM host tier ──────────────────────────────────
    /// The host-RAM KV tier (core-06): holds evicted (suspended) request
    /// snapshots in two tiers (probation → protected); evictions are
    /// bounded by `host_capacity_pages`.
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
        Self {
            capacity: AdmissionResources {
                lanes: N_DECODE_LANES as u32,
                kv_pages: config.kv_capacity_pages,
                backend_pages: 0,
            },
            kv_used_pages: 0,
            protection: None,
            protection_epoch: 1,
            host: HostTier::new(config.host_capacity_pages),
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
    /// cached entry's id + the leading prompt tokens it skips. The FFI /
    /// kernel leaf uses this to bind the shared prefix's blocks read-only
    /// into the request's block table (the kernel-abi channel; ADR 0001).
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
            kv_pages: self.capacity.kv_pages.saturating_sub(self.prefix.pinned_pages()),
            backend_pages: self.capacity.backend_pages,
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
        let Some(freed) = self.prefix.release(entry) else {
            return;
        };
        self.kv_used_pages = self.kv_used_pages.saturating_sub(freed);
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

    /// Whether `r` can be dealt right now: the running set plus `r`
    /// fits the capacity component-wise (a free lane *and* enough main-
    /// pool pages — the resource dimension that makes protection
    /// meaningful).
    fn fits(&self, r: &Request) -> bool {
        let running = AdmissionResources {
            lanes: N_DECODE_LANES as u32 - self.free_lanes.len() as u32,
            kv_pages: self.kv_used_pages,
            backend_pages: 0,
        };
        admission_resources_fit(&running.add(&r.resources), &self.capacity)
    }

    /// Deal a lane to `idx` (a `Prefilling` request), charging the lane
    /// and its full KV reservation. Returns `false` (deals nothing) when
    /// no lane is free.
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
        if !self.requests[idx].assign_lane(lane) {
            // Not in `Prefilling` (a bug: the machine only deals
            // `Prefilling` requests) — give the lane back, deal nothing.
            self.free_lanes.push(lane);
            return false;
        }
        let (kv_pages, backfill_epoch) = {
            let epoch = if backfill == BackfillClass::None {
                0
            } else {
                self.protection.as_ref().map(|p| p.epoch_id).unwrap_or(0)
            };
            (self.requests[idx].resources.kv_pages, epoch)
        };
        self.kv_used_pages += kv_pages; // over-reservation: charged at deal
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
                resources: r.resources,
                remaining_work_quanta: r.remaining_work,
                backfill_epoch: r.backfill_epoch,
                backfill_class: r.backfill_class,
            })
            .collect()
    }

    /// Phase 2 (core-05): the admission state machine drives the lane
    /// deal. The queue is the `Prefilling` set in (class priority, FIFO
    /// by id) order; the head is dealt when it fits, and — when the head
    /// is blocked by the active set — the machine opens / maintains the
    /// protection and classifies backfills (persistent / temporal) on
    /// the lanes the donors will free.
    fn run_admission(&mut self, events: &mut Vec<SchedEvent>) {
        let mut queue: Vec<usize> = self
            .requests
            .iter()
            .enumerate()
            .filter(|&(_, r)| r.state == RequestState::Prefilling)
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
                    self.requests[head].resources,
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
                        &self.requests[c].resources,
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

    /// Complete request `idx` (its lane and KV reservation are released).
    fn mark_done(&mut self, idx: usize, events: &mut Vec<SchedEvent>) {
        let (release_pages, lane, request_id, tokens, prefix_entry) = {
            let r = &self.requests[idx];
            (r.resources.kv_pages, r.lane, r.id, r.tokens, r.prefix_entry)
        };
        self.requests[idx].advance(RequestState::Done);
        if let Some(lane) = lane {
            self.free_lanes.push(lane);
            self.kv_used_pages = self.kv_used_pages.saturating_sub(release_pages);
        }
        // core-07: release the request's shared-prefix claim (its
        // completion frees its reference to the shared pages; the entry
        // drops — and its pages return to the pool — when the last
        // claimant releases).
        self.release_prefix_claim(prefix_entry);
        // A protection exists to let its head in: if the protected head
        // itself completes (rather than being dealt), its protection is
        // stale — the next blocked head opens a fresh epoch.
        let protected_head = self.protection.as_ref().map(|p| p.head_request_id);
        if protected_head == Some(request_id) {
            self.protection = None;
        }
        events.push(SchedEvent::Done {
            request: request_id,
            tokens,
        });
    }

    // ── core-06: the KV-RAM host tier ───────────────────────────────────

    /// The retained-lane candidates for victim selection (core-06): every
    /// running request's lane, excluding the protection's donors (donors
    /// are never evicted while the protection is open) and reserved lanes
    /// (a lane whose shared prefix is claimed by an earlier-queued
    /// interactive request — core-07).
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

    /// Make room in the host tier for `pages` pages (core-06): discard the
    /// lowest-value entries (probation LRU) while the tier is over budget,
    /// re-queueing each discarded request (its snapshot was lost — it
    /// re-prefills from the start). Returns `true` when the tier can hold
    /// `pages` (there is room, or it was made).
    fn make_room_for(&mut self, pages: u32, events: &mut Vec<SchedEvent>) -> bool {
        while self.host.used_pages() + pages > self.host.capacity_pages() {
            match self.host.evict_one() {
                Some(discarded) => {
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

    /// Try to admit a blocked head by evicting a retained lane to the host
    /// tier (core-06): while the head does not fit, pick the lowest-value
    /// non-donor running lane (`choose_retained_lane_victim`), make room in
    /// the host tier (re-queueing any discarded snapshot), snapshot the
    /// victim into the tier, and release its lane + pages. Returns `true`
    /// once the head fits (the caller deals it), `false` when no evictable
    /// victim remains (the head is held — the backfill / donor wait path).
    fn try_evict_for_head(&mut self, head_idx: usize, events: &mut Vec<SchedEvent>) -> bool {
        loop {
            if self.fits(&self.requests[head_idx]) {
                return true;
            }
            // No free lane / the head still does not fit: try to evict a
            // retained lane (a running request other than the donors).
            let candidates = self.retained_lane_candidates();
            let Some(victim_lane) = choose_retained_lane_victim(&candidates) else {
                return false; // no evictable victim (all reserved / none)
            };
            let Some(v_idx) = self
                .requests
                .iter()
                .position(|r| r.lane == Some(victim_lane))
            else {
                return false;
            };
            // Copy the request's state (avoid a borrow conflict with the
            // mutation below).
            let (v_id, v_class, v_pages, v_tokens, v_work, v_gdn) = {
                let v = &self.requests[v_idx];
                (
                    v.id,
                    v.class,
                    v.resources.kv_pages,
                    v.tokens,
                    v.remaining_work,
                    v.gdn.clone(),
                )
            };
            // Make room in the host tier (re-queueing any discarded
            // snapshot).
            if !self.make_room_for(v_pages, events) {
                return false; // the host tier cannot hold the snapshot
            }
            let entry = HostEntry {
                request: v_id,
                lane: victim_lane,
                owner: v_class,
                pages: v_pages,
                tokens: v_tokens,
                remaining_work: v_work,
                gdn: v_gdn,
                tier: Tier::Probation,
                use_tick: self.tick,
            };
            // Capture the snapshot (rejects a mid-prefill GDN position,
            // core-02).
            if self.host.capture(entry).is_err() {
                return false; // the snapshot is invalid (e.g. mid-prefill)
            }
            // Evict the request (Running → Evicted; the lane is released).
            self.requests[v_idx].evict();
            self.free_lanes.push(victim_lane);
            self.kv_used_pages = self.kv_used_pages.saturating_sub(v_pages);
            events.push(SchedEvent::Evicted { request: v_id });
            // Loop: re-check whether the head now fits.
        }
    }

    /// Restore evicted (suspended) requests to free lanes (core-06): a
    /// restored request resumes from where it was evicted (no re-prefill),
    /// taking priority over a fresh prefill. Restores as many evicted
    /// requests as there are free lanes + page headroom, in the host
    /// tier's victim order (the entries closest to being discarded).
    fn restore_pass(&mut self, events: &mut Vec<SchedEvent>) {
        loop {
            if self.free_lanes.is_empty() {
                break;
            }
            let victim = match self.host.victim() {
                Some(v) => v.clone(),
                None => break, // no evicted request to restore
            };
            // The restored request's pages must fit the GPU pool.
            if self.kv_used_pages + victim.pages > self.capacity.kv_pages {
                break; // no page headroom: leave it (retry next advance)
            }
            let lane = self.free_lanes.pop().expect("checked non-empty above");
            let snap = self
                .host
                .restore(victim.request)
                .expect("the victim is a tier entry");
            let idx = self
                .requests
                .iter()
                .position(|r| r.id == snap.request)
                .expect("a host-tier snapshot always maps to a request");
            self.requests[idx].restore_lane(lane);
            self.kv_used_pages += snap.pages;
            events.push(SchedEvent::Restored {
                request: snap.request,
                lane,
            });
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
        };
        if !admission_resources_fit(&resources, &self.capacity) {
            return Err(SubmitError::Oversized);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.requests.push(Request::new(
            id,
            class,
            input,
            resources,
            effective_max as u64,
        ));
        Ok(id)
    }

    fn advance(&mut self) -> Vec<SchedEvent> {
        let mut events: Vec<SchedEvent> = Vec::new();
        self.last_error = None;
        // core-06: advance the scheduling tick (the LRU `use_tick` for
        // retained-lane victim selection).
        self.tick += 1;

        // Phase 1 — batched prefill: the queued requests go to the
        // compute backend in ONE batched call, not one call per
        // request. The Admitted → Prefilling transition happens only
        // AFTER the call succeeds: a failed prefill leaves the batch in
        // `Admitted` (retryable on the next advance), so a request is
        // never dealt a lane with an unwarmed KV (the `request.rs`
        // invariant: "a request must finish prefill before it holds a
        // lane").
        let mut batch: Vec<usize> = self
            .requests
            .iter()
            .enumerate()
            .filter(|&(_, r)| r.state == RequestState::Admitted)
            .map(|(i, _)| i)
            .collect();
        batch.sort_by_key(|&i| (self.requests[i].class, self.requests[i].id));
        batch.truncate(self.config.max_prefill_batch);
        // core-07 — sibling prefix claim: each batch request without a
        // shared head claims the longest cached prefix of its prompt
        // (skipping the redundant prefill — its job carries only the
        // tail, and its own reservation shrinks to the tail + max: the
        // shared entry's pages are charged to the pool once, for every
        // claimant). The claim is established before the prefill call
        // (the shared prefix is already warm in the pool); a failed
        // prefill is retried next advance with the same claim.
        for &i in &batch {
            // A request that already holds a claim (from a prior advance,
            // whose prefill failed and is retried) keeps it: re-claiming
            // would double-count the entry's refcount and the
            // `sibling_prefix_reused_tok` counter, and pin the entry
            // forever (the release happens once, at completion).
            if self.requests[i].prefix_entry.is_some() {
                continue;
            }
            let claimed = self.prefix.claim(&self.requests[i].input.tokens);
            if let Some(claim) = claimed {
                let r = &mut self.requests[i];
                r.prefix_entry = Some(claim.id);
                r.shared_prefix_tokens = claim.tokens;
                r.gdn = claim.gdn; // core-02: resume at the shared boundary
                // Shrink the claimant's own reservation by the shared
                // prefix's pages (the entry now owns them — charged once,
                // for every claimant). `ceil((prompt + max) / pt) -
                // shared_pages` equals `ceil((tail + max) / pt)`: the
                // shared head is page-aligned, so subtracting its whole
                // pages is exact.
                r.resources.kv_pages = r.resources.kv_pages.saturating_sub(claim.pages);
                events.push(SchedEvent::PrefixReused {
                    request: r.id,
                    tokens: claim.tokens,
                });
            }
        }
        // A claimant carries only its tail (the shared head is already
        // warm in the pool — the kernel leaf binds the shared prefix's
        // blocks read-only); a fresh request carries its full prompt.
        let jobs: Vec<PrefillJob> = batch
            .iter()
            .map(|&i| {
                let r = &self.requests[i];
                let tokens = if r.shared_prefix_tokens > 0 {
                    r.input
                        .tokens
                        .iter()
                        .skip(r.shared_prefix_tokens as usize)
                        .copied()
                        .collect()
                } else {
                    r.input.tokens.clone()
                };
                PrefillJob {
                    request: r.id,
                    tokens,
                    params: r.input.params,
                }
            })
            .collect();
        if !jobs.is_empty() {
            match self.compute.prefill_step(&jobs) {
                Ok(()) => {
                    for &i in &batch {
                        self.requests[i].advance(RequestState::Prefilling);
                        // core-07 — registration: only a fresh request
                        // (no shared head — it claimed nothing) caches
                        // its now-warm prompt for siblings. A claimant's
                        // head is already the cached entry (registering
                        // its full prompt would double-charge the shared
                        // pages), and a same-batch duplicate's prompt is
                        // already cached by the batch's first registrant
                        // (register returns `None` — it keeps its own
                        // charge, no re-registration). The entry's pages
                        // are charged to the pool now (the charge split:
                        // the registrant's own reservation keeps the
                        // residual, the entry holds the shared pages).
                        if self.requests[i].prefix_entry.is_none() {
                            let registered = self.prefix
                                .register(
                                    &self.requests[i].input.tokens,
                                    &self.requests[i].gdn,
                                );
                            if let Some((entry, pages)) = registered {
                                let r = &mut self.requests[i];
                                r.prefix_entry = Some(entry);
                                r.resources.kv_pages =
                                    r.resources.kv_pages.saturating_sub(pages);
                                self.kv_used_pages =
                                    self.kv_used_pages.saturating_add(pages);
                            }
                        }
                    }
                }
                Err(e) => {
                    // Failed prefill: the batch stays `Admitted` and is
                    // retried on the next advance; the fault is
                    // surfaced through `last_error`.
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
                self.mark_done(i, &mut events);
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
                        match res {
                            Some(token) => {
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
                                // state is resumable there).
                                let new_pos = self.requests[i].tokens as usize;
                                self.requests[i].checkpoint(new_pos);
                                // The reservation cap: the request
                                // completes on its final reserved token.
                                if self.requests[i].remaining_work == 0 {
                                    self.mark_done(i, &mut events);
                                }
                            }
                            None => {
                                // Finished (max_tokens / EOS): Done,
                                // lane released.
                                self.mark_done(i, &mut events);
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
