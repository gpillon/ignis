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
//!    the GPU).
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
//! Lane capacity: the scheduler holds [`N_DECODE_LANES`] (8) resident
//! lanes and a KV page capacity (`kv_capacity_pages`, auto-sized from
//! the pool in production; the machine's resource dimension). In-flight
//! admission is capped at `max_in_flight` — host-tier overflow
//! (admitting beyond 8 via the KV-RAM host tier, with retained-lane
//! eviction driven by [`crate::admission::choose_retained_lane_victim`])
//! lands in core-06; until then `submit` fails with
//! [`SubmitError::Full`] at the cap, and a request whose KV reservation
//! exceeds the whole pool is rejected with [`SubmitError::Oversized`].

use std::sync::Arc;

use crate::admission::{
    admission_resources_fit, make_admission_protection, persistent_backfill_is_safe,
    protection_frontier_distance, protected_head_safe_without_temporal,
    ActiveAdmissionSnapshot, AdmissionProtection, AdmissionResources, ProtectionPhase,
};
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
        assert!(
            config.max_in_flight > 0,
            "in-flight cap must be non-zero"
        );
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
            // `clear_protection_if_head`: once the protected head is
            // dealt, its protection is cleared (the next blocked head
            // opens a fresh epoch).
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
                let protection = match make_admission_protection(
                    self.protection_epoch,
                    self.requests[head].id,
                    self.requests[head].resources,
                    &active,
                    &self.capacity,
                ) {
                    Ok(p) => p,
                    Err(e) => {
                        // Broken invariants are a bug, surfaced (not
                        // swallowed): skip this step's backfill path,
                        // the next advance retries the same step.
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
            if protected_head_safe_without_temporal(
                self.protection.as_ref().unwrap(),
                &active,
                &self.capacity,
            ) {
                self.protection
                    .as_mut()
                    .unwrap()
                    .phase = ProtectionPhase::Drain;
            }
            if self
                .protection
                .as_ref()
                .unwrap()
                .phase
                == ProtectionPhase::Open
            {
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
                        &self.capacity,
                    ) {
                        self.try_admit(c, BackfillClass::Persistent, events);
                    } else if self.requests[c].remaining_work <= frontier
                        && self.requests[c].remaining_work <= p.temporal_credit
                    {
                        // Credit decay: a temporal backfill spends its
                        // own service work out of the frozen credit.
                        self.protection
                            .as_mut()
                            .unwrap()
                            .temporal_credit -= self.requests[c].remaining_work;
                        self.try_admit(c, BackfillClass::Temporal, events);
                    }
                }
            }
        }
    }

    /// Complete request `idx` (its lane and KV reservation are released).
    fn mark_done(&mut self, idx: usize, events: &mut Vec<SchedEvent>) {
        let (release_pages, lane, request_id, tokens) = {
            let r = &self.requests[idx];
            (
                r.resources.kv_pages,
                r.lane,
                r.id,
                r.tokens,
            )
        };
        self.requests[idx].advance(RequestState::Done);
        if let Some(lane) = lane {
            self.free_lanes.push(lane);
            self.kv_used_pages = self.kv_used_pages.saturating_sub(release_pages);
        }
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
        let reserved_tokens =
            (input.tokens.len() as u64).saturating_add(effective_max as u64);
        let kv_pages = ((reserved_tokens
            + self.config.kv_page_tokens as u64
            - 1)
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
        let jobs: Vec<PrefillJob> = batch
            .iter()
            .map(|&i| PrefillJob {
                request: self.requests[i].id,
                tokens: self.requests[i].input.tokens.clone(),
                params: self.requests[i].input.params,
            })
            .collect();
        if !jobs.is_empty() {
            match self.compute.prefill_step(&jobs) {
                Ok(()) => {
                    for &i in &batch {
                        self.requests[i].advance(RequestState::Prefilling);
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
        // (core-05; see `run_admission`).
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