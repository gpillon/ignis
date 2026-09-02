//! Request state machine + basic admission — `core-03`.
//!
//! The request lifecycle is `admitted → prefilling → running → done`
//! (`CONTEXT.md`). This module models a request in flight: its lifecycle
//! state, its class (for admission / backfill), the decode lane it holds, and
//! its token count. The **full** admission state machine (protection,
//! backfill class, temporal credit, frontier distance) is core-05; this
//! module carries the *basic* lane assignment it builds on.

use crate::types::{
    LaneId, RequestClass, RequestId, RequestInput, RequestState, TokenId,
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
}

impl Request {
    /// A freshly-admitted request: state `Admitted`, no lane yet.
    pub fn new(id: RequestId, class: RequestClass, input: RequestInput) -> Self {
        Self {
            id,
            class,
            input,
            state: RequestState::Admitted,
            lane: None,
            tokens: 0,
        }
    }

    /// Whether a transition `from → to` is a valid lifecycle step.
    ///
    /// The lifecycle is a strict pipeline:
    /// `Admitted → Prefilling → Running → Done`. No skipping, no backwards.
    pub fn valid_transition(from: RequestState, to: RequestState) -> bool {
        match from {
            RequestState::Admitted => to == RequestState::Prefilling,
            RequestState::Prefilling => to == RequestState::Running,
            RequestState::Running => to == RequestState::Done,
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
/// assignment — the full admission state machine (protection, backfill,
/// temporal credit, frontier distance) is core-05.
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