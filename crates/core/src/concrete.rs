//! The concrete scheduler — `core-04` (N=8 decode lanes + batched prefill).
//!
//! The v1 scheduler (`docs/design/ignis-v1.md` §2) driving the [`Compute`]
//! seam (production: the kernel leaf via FFI, tests: [`crate::mock::MockCompute`])
//! in three phases per advance:
//!
//! 1. **Batched prefill** — up to `max_prefill_batch` queued requests are
//!    grouped into ONE compute call (not one call per request) to saturate
//!    the GPU in prefill and cut burst TTFT. *An experiment to verify* —
//!    we may be compute-bound, in which case it is useless (a measure, not a
//!    guarantee; the 99% gate of ADR 0007 re-checks it on the GPU).
//! 2. **Lane deal** — free resident lanes are dealt to prefilled requests
//!    (class priority + FIFO; the *full* admission state machine —
//!    protection, backfill, temporal credit, frontier distance — is core-05).
//! 3. **Batched decode** — one compute call spanning every running lane.
//!
//! Lane capacity: the scheduler holds [`N_DECODE_LANES`] (8) resident lanes.
//! In-flight admission is capped at `max_in_flight` (default 8) —
//! host-tier overflow (admitting beyond 8 via the KV-RAM host tier) lands
//! in core-06; until then `submit` fails with [`SubmitError::Full`] at the
//! cap.

use std::sync::Arc;

use crate::request::{Request, admit_candidates};
use crate::scheduler::{Compute, DecodeJob, PrefillJob, Scheduler};
use crate::types::{
    ComputeError, EngineMode, LaneId, N_DECODE_LANES, RequestClass, RequestId, RequestInput,
    RequestState, SchedEvent, SubmitError,
};

/// Knobs for the concrete scheduler (v1 defaults; the KV-RAM host tier will
/// raise `max_in_flight` beyond N=8 in core-06).
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
}

/// The concrete N=8 resident-lane scheduler (v1). See the module docs for
/// the per-advance phase structure.
pub struct ConcreteScheduler {
    config: SchedulerConfig,
    compute: Arc<dyn Compute>,
    next_id: RequestId,
    requests: Vec<Request>,
    free_lanes: Vec<LaneId>,
    /// The last hard compute error (kernel fault) hit by an advance, if any.
    last_error: Option<ComputeError>,
}

impl ConcreteScheduler {
    /// A scheduler with the v1 defaults: `max_in_flight` and
    /// `max_prefill_batch` both [`N_DECODE_LANES`].
    pub fn new(model: &str, compute: Arc<dyn Compute>) -> Self {
        Self::with_config(
            SchedulerConfig {
                model: model.into(),
                max_in_flight: N_DECODE_LANES,
                max_prefill_batch: N_DECODE_LANES,
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
        Self {
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

    /// The hard compute error the most recent advance reported, if any.
    ///
    /// A failed step is not swallowed: it emits no events, the state is
    /// left retryable (a failed prefill leaves the batch in `Admitted`, a
    /// failed decode leaves the lanes running), and the next advance
    /// retries the same step. A successful advance clears it; callers poll
    /// this to surface the fault.
    pub fn last_error(&self) -> Option<&ComputeError> {
        self.last_error.as_ref()
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
        let id = self.next_id;
        self.next_id += 1;
        self.requests.push(Request::new(id, class, input));
        Ok(id)
    }

    fn advance(&mut self) -> Vec<SchedEvent> {
        let mut events: Vec<SchedEvent> = Vec::new();
        self.last_error = None;

        // Phase 1 — batched prefill: the queued requests go to the compute
        // backend in ONE batched call, not one call per request. The
        // Admitted → Prefilling transition happens only AFTER the call
        // succeeds: a failed prefill leaves the batch in `Admitted` (retryable
        // on the next advance), so a request is never dealt a lane with an
        // unwarmed KV (the `request.rs` invariant: "a request must finish
        // prefill before it holds a lane").
        let mut batch: Vec<usize> = (0..self.requests.len())
            .filter(|&i| self.requests[i].state == RequestState::Admitted)
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
                    // retried on the next advance; the fault is surfaced
                    // through `last_error`.
                    self.last_error = Some(e);
                    return events;
                }
            }
        }

        // Phase 2 — lane deal: free resident lanes go to prefilled
        // requests (class priority + FIFO; the full admission machine is
        // core-05). Lanes are dealt smallest-first for a deterministic
        // event stream.
        let mut free: Vec<LaneId> = std::mem::take(&mut self.free_lanes);
        free.sort_by(|a, b| b.cmp(a)); // pop order: 0, 1, 2, ...
        let dealt = admit_candidates(&mut self.requests, &mut free);
        self.free_lanes = free;
        for &(request, lane) in &dealt {
            events.push(SchedEvent::Admitted { request, lane });
        }

        // Phase 3 — batched decode: one compute call spanning every
        // running lane (lane-ascending order for a deterministic stream).
        let mut running: Vec<(usize, LaneId)> = self
            .requests
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                if r.state == RequestState::Running {
                    Some((i, r.lane.expect("Running requests hold a lane")))
                } else {
                    None
                }
            })
            .collect();
        running.sort_by_key(|&(_, lane)| lane);
        if !running.is_empty() {
            let jobs: Vec<DecodeJob> = running
                .iter()
                .map(|&(i, lane)| DecodeJob {
                    request: self.requests[i].id,
                    lane,
                    params: self.requests[i].input.params,
                })
                .collect();
            match self.compute.decode_step(&jobs) {
                Ok(results) => {
                    for ((i, _), res) in running.iter().zip(&results) {
                        match res {
                            Some(token) => {
                                self.requests[*i].tokens += 1;
                                events.push(SchedEvent::Token {
                                    request: self.requests[*i].id,
                                    token: *token,
                                });
                            }
                            None => {
                                // Finished (max_tokens / EOS): Done, lane released.
                                self.requests[*i].advance(RequestState::Done);
                                if let Some(lane) = self.requests[*i].lane.take() {
                                    self.free_lanes.push(lane);
                                }
                                events.push(SchedEvent::Done {
                                    request: self.requests[*i].id,
                                    tokens: self.requests[*i].tokens,
                                });
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
