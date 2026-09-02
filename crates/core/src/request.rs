//! Request state machine + basic admission — `core-03`.
//!
//! The request lifecycle is `admitted → prefilling → running → done`
//! (`CONTEXT.md`). This module models a request in flight: its lifecycle
//! state, its class (for admission / backfill), the decode lane it holds, and
//! its token count, plus — since `core-05` — the admission state
//! machine's bookkeeping: the KV resources the request reserves, its
//! remaining service work (the protection's donor ordering, temporal
//! credit, and frontier distance all run on these quanta), and the
//! backfill class / protection epoch it was admitted under.
//!
//! [`admit_candidates`] / [`basic_admission`] below carry the *basic*
//! lane assignment (class priority + FIFO) that `core-03` shipped; the
//! **full** admission state machine (protection, backfill class, temporal
//! credit, frontier distance — `core-05`, ADR 0004) drives lane
//! assignment in the concrete scheduler (`concrete.rs` + `admission.rs`)
//! and supersedes them there.

use crate::admission::AdmissionResources;
use crate::gdn::GdnState;
use crate::types::{
    BackfillClass, LaneId, RequestClass, RequestId, RequestInput, RequestState, TokenId,
};

/// A request in flight in the engine: its lifecycle state, its class (for
/// admission / backfill), and the decode lane it holds while running.
#[derive(Debug, Clone)]
pub struct Request {
    pub id: RequestId,
    /// Admission / backfill class (drives lane priority and eviction order).
    pub class: RequestClass,
    /// The submitted prompt (tokenized + templated) + generation params.
    pub input: RequestInput,
    /// The lifecycle state (admitted → prefilling → running → done).
    pub state: RequestState,
    /// The resident decode lane held while `Running` (else `None`).
    pub lane: Option<LaneId>,
    /// Tokens generated so far for this request.
    pub tokens: u32,
    /// The resources this request reserves while it holds a lane (core-05:
    /// the admission state machine's KV reservation, charged at deal and
    /// released at completion — the pool never over-allocates).
    pub resources: AdmissionResources,
    /// Remaining service work (quanta; 1 quantum per decode token): drives
    /// the protection's donor ordering, temporal credit, and frontier
    /// distance (core-05).
    pub remaining_work: u64,
    /// The protection epoch this request was admitted under as a backfill
    /// (core-05; 0 = a plain deal, no protection involved).
    pub backfill_epoch: u64,
    /// The class this request was admitted under by the admission state
    /// machine (core-05; [`BackfillClass::None`] for plain deals).
    pub backfill_class: BackfillClass,
    /// The GDN (linear-attention) recurrent-state tracker (core-02): the
    /// checkpoint / frontier boundaries at which the state is resumable.
    /// The host tier (core-06) snapshots this — a snapshot is only valid at
    /// a recorded boundary (a mid-prefill snapshot is invalid for GDN
    /// layers).
    pub gdn: GdnState,
}

impl Request {
    /// A freshly-admitted request: state `Admitted`, no lane yet.
    /// `resources` is the request's KV reservation (the admission state
    /// machine, core-05) and `remaining_work` its service work in quanta.
    pub fn new(
        id: RequestId,
        class: RequestClass,
        input: RequestInput,
        resources: AdmissionResources,
        remaining_work: u64,
    ) -> Self {
        Self {
            id,
            class,
            input,
            state: RequestState::Admitted,
            lane: None,
            tokens: 0,
            resources,
            remaining_work,
            backfill_epoch: 0,
            backfill_class: BackfillClass::None,
            gdn: GdnState::new(),
        }
    }

    /// Whether a transition `from → to` is a valid lifecycle step.
    ///
    /// The lifecycle is a strict pipeline:
    /// `Admitted → Prefilling → Running → Done`, plus — since core-06 — the
    /// host-tier detour `Running → Evicted → Running` (a request evicted to
    /// the host KV-RAM tier is suspended and later restored to a lane
    /// without re-prefilling), and the re-queue `Evicted → Admitted` (a
    /// request whose snapshot was discarded from the host tier is re-queued
    /// for re-prefill). No skipping, no other backwards steps.
    pub fn valid_transition(from: RequestState, to: RequestState) -> bool {
        match from {
            RequestState::Admitted => to == RequestState::Prefilling,
            RequestState::Prefilling => to == RequestState::Running,
            RequestState::Running => to == RequestState::Done || to == RequestState::Evicted,
            RequestState::Evicted => to == RequestState::Running || to == RequestState::Admitted,
            RequestState::Done => false,
        }
    }

    /// Advance the request to `next`, enforcing the valid lifecycle. Returns
    /// `true` (and applies) only when the transition is valid.
    ///
    /// A request only enters `Running` while holding a resident decode
    /// lane — `assign_lane` first (it also transitions to `Running`), so a
    /// lane-less request can never run.
    pub fn advance(&mut self, next: RequestState) -> bool {
        if !Self::valid_transition(self.state, next) {
            return false;
        }
        if next == RequestState::Running && self.lane.is_none() {
            return false;
        }
        self.state = next;
        true
    }

    /// Assign a resident decode lane (only valid from `Prefilling` →
    /// `Running`). Fails (returns `false`) when the request is not in the
    /// `Prefilling` state — a request must finish prefill before it holds a
    /// lane.
    pub fn assign_lane(&mut self, lane: LaneId) -> bool {
        if self.state != RequestState::Prefilling {
            return false;
        }
        self.lane = Some(lane);
        self.state = RequestState::Running;
        true
    }

    /// Evict the request from its decode lane to the host KV-RAM tier
    /// (core-06): transition `Running → Evicted` and release the lane.
    /// Fails (returns `false`) when the request is not `Running` — only a
    /// lane-holding request can be evicted (a `Prefilling` / `Admitted`
    /// request has no warmed KV to snapshot, so it is never a victim).
    pub fn evict(&mut self) -> bool {
        if self.state != RequestState::Running {
            return false;
        }
        self.state = RequestState::Evicted;
        self.lane = None; // the lane is released (the caller reclaims it).
        true
    }

    /// Restore a previously-evicted request onto a decode lane (core-06):
    /// transition `Evicted → Running` and (re-)acquire the lane. Fails
    /// (returns `false`) when the request is not in the `Evicted` state —
    /// only a suspended (host-tier) request can be restored, and it resumes
    /// from where it was evicted (no re-prefill).
    pub fn restore_lane(&mut self, lane: LaneId) -> bool {
        if self.state != RequestState::Evicted {
            return false;
        }
        self.lane = Some(lane);
        self.state = RequestState::Running;
        true
    }

    /// Re-queue a discarded (evicted) request for re-prefill (core-06):
    /// transition `Evicted → Admitted` and release the lane (there is none
    /// — the request is suspended). Fails (returns `false`) when the
    /// request is not in the `Evicted` state. A re-queued request re-
    /// prefills from the start (its snapshot was discarded from the host
    /// tier, so its warmed KV is gone); the caller resets its token /
    /// service-work counters. The GDN recurrent state is also reset to a
    /// fresh (position-0, no-boundary) state: the re-prefilled stream starts
    /// over, so the old (stale) checkpoint boundaries — which sit at
    /// positions the new stream has not reached — would otherwise let a
    /// later snapshot be accepted at a position the stream never reached
    /// (a restore would resume GDN at a point it has not computed).
    pub fn requeue(&mut self) -> bool {
        if self.state != RequestState::Evicted {
            return false;
        }
        self.state = RequestState::Admitted;
        self.lane = None;
        self.gdn = GdnState::new(); // fresh recurrent state (re-prefill starts over)
        true
    }

    /// Record a GDN checkpoint / frontier boundary at `position` (core-02):
    /// the recurrent state becomes resumable there. The host tier (core-06)
    /// may snapshot the request's state only at a recorded boundary.
    pub fn checkpoint(&mut self, position: usize) {
        self.gdn.checkpoint(position);
    }

    /// The prompt tokens this request was submitted with.
    pub fn prompt_tokens(&self) -> &[TokenId] {
        &self.input.tokens
    }
}

/// **Basic admission** (core-03): assign free decode lanes to `Prefilling`
/// requests, returning the (request, lane) pairs dealt, ordered by class
/// priority (Interactive before Agent — the derived `Ord` on
/// [`RequestClass`]) then FIFO by request id. The lane pool is consumed in
/// the order the caller provides it (the concrete scheduler sorts it
/// ascending so the deal order is deterministic).
pub fn admit_candidates(
    requests: &mut [Request],
    free_lanes: &mut Vec<LaneId>,
) -> Vec<(RequestId, LaneId)> {
    // Candidate requests: those that have finished prefill and need a lane.
    let mut pending: Vec<usize> = (0..requests.len())
        .filter(|&i| requests[i].state == RequestState::Prefilling)
        .collect();
    // Class priority first (Interactive < Agent), then FIFO by id.
    pending.sort_by_key(|&i| (requests[i].class, requests[i].id));

    let mut dealt = Vec::new();
    for &i in &pending {
        if let Some(lane) = free_lanes.pop()
            && requests[i].assign_lane(lane)
        {
            dealt.push((requests[i].id, lane));
        }
    }
    dealt
}

/// **Basic admission** (core-03): assign free decode lanes to `Prefilling`
/// requests. Requests are ordered by class priority (Interactive before
/// Agent — the derived `Ord` on [`RequestClass`]) then FIFO by request id.
/// A `Prefilling` request that gets a lane transitions to `Running`.
///
/// Returns the number of lanes assigned. This is the *basic* lane
/// assignment — the full admission state machine (protection, backfill
/// class, temporal credit, frontier distance — core-05, ADR 0004) drives
/// lane assignment in the concrete scheduler and supersedes it there.
pub fn basic_admission(requests: &mut [Request], free_lanes: &mut Vec<LaneId>) -> usize {
    admit_candidates(requests, free_lanes).len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DecodeParams;

    fn req(id: RequestId, class: RequestClass, state: RequestState) -> Request {
        let mut r = Request::new(
            id,
            class,
            RequestInput {
                model: "qwen3.8-27b".into(),
                tokens: vec![1, 2, 3],
                params: DecodeParams::default(),
            },
            AdmissionResources::default(),
            4,
        );
        r.state = state;
        r
    }

    #[test]
    fn lifecycle_is_a_strict_pipeline() {
        let mut r = req(0, RequestClass::Agent, RequestState::Admitted);
        assert!(r.advance(RequestState::Prefilling));
        // Entering Running requires a held lane: assign_lane first (it also
        // transitions to Running), then the pipeline continues to Done.
        assert!(r.assign_lane(3));
        assert_eq!(r.state, RequestState::Running);
        assert!(r.advance(RequestState::Done));
        // No valid transition out of Done.
        assert!(!r.advance(RequestState::Prefilling));
        // A lane-less request cannot enter Running.
        let mut r = req(1, RequestClass::Agent, RequestState::Prefilling);
        assert!(!r.advance(RequestState::Running));
        assert_eq!(r.state, RequestState::Prefilling);
    }

    #[test]
    fn invalid_transitions_are_rejected() {
        // Cannot skip from Admitted straight to Running.
        let mut r = req(0, RequestClass::Agent, RequestState::Admitted);
        assert!(!r.advance(RequestState::Running));
        // Cannot go backwards.
        let mut r = req(0, RequestClass::Agent, RequestState::Prefilling);
        assert!(!r.advance(RequestState::Admitted));
    }

    #[test]
    fn lane_assignment_requires_prefilling_state() {
        // A request still Admitted cannot grab a lane.
        let mut r = req(0, RequestClass::Agent, RequestState::Admitted);
        assert!(!r.assign_lane(3));
        assert_eq!(r.lane, None);
        // From Prefilling it can.
        let mut r = req(0, RequestClass::Agent, RequestState::Prefilling);
        assert!(r.assign_lane(3));
        assert_eq!(r.lane, Some(3));
        assert_eq!(r.state, RequestState::Running);
    }

    #[test]
    fn basic_admission_orders_by_class_then_fifo() {
        let mut requests = vec![
            req(10, RequestClass::Agent, RequestState::Prefilling),
            req(1, RequestClass::Agent, RequestState::Prefilling),
            req(2, RequestClass::Interactive, RequestState::Prefilling),
            req(3, RequestClass::Interactive, RequestState::Prefilling),
        ];
        let mut free = vec![5, 7];
        let n = basic_admission(&mut requests, &mut free);
        assert_eq!(n, 2);
        // Both Interactive requests hold a lane (class priority: Interactive
        // before Agent). Which physical lane each gets is not part of the
        // contract — free lanes are fungible, so assert on the set, not on
        // the deal order.
        assert!(requests[2].lane.is_some());
        assert!(requests[3].lane.is_some());
        // The Agent requests stay unassigned.
        assert!(requests[0].lane.is_none());
        assert!(requests[1].lane.is_none());
        assert!(free.is_empty());

        // FIFO within a class: with a single lane, the lower-id Interactive
        // request is admitted before the higher-id one (and before Agents).
        let mut requests = vec![
            req(9, RequestClass::Interactive, RequestState::Prefilling),
            req(4, RequestClass::Interactive, RequestState::Prefilling),
            req(0, RequestClass::Agent, RequestState::Prefilling),
        ];
        let mut free = vec![3];
        let n = basic_admission(&mut requests, &mut free);
        assert_eq!(n, 1);
        assert!(requests[1].lane.is_some()); // Interactive id 4 won
        assert!(requests[0].lane.is_none()); // Interactive id 9 deferred
        assert!(requests[2].lane.is_none()); // Agent deferred
        assert!(free.is_empty());
    }
}
