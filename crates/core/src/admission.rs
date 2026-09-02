//! The admission state machine policy — `core-05` (ADR 0004).
//!
//! The fairness machinery that decides which request gets which lane
//! (`CONTEXT.md`: "the fairness machinery (protection, backfill class,
//! temporal credit, frontier distance) deciding which request gets which
//! lane"). This module is a faithful port of the reference stack's
//! admission policy (ADR 0004: "the port must preserve invariant behavior
//! (protection promotion, credit decay, frontier distance) and each
//! invariant gets a dedicated test"):
//!
//! - **Protection** — when a queued head is blocked by the active set
//!   (no free lane, or its KV reservation does not fit the remaining
//!   pages), the machine *freezes* the active incumbents and selects the
//!   **donors**: the earliest-completion prefix whose release makes the
//!   head feasible. Donors are never evicted while the protection is open
//!   ("resident lanes are not evicted").
//! - **Backfill class** — a candidate admitted *while* a protection is
//!   open (on a lane a donor will free) is classified [`BackfillClass`]:
//!   `Persistent` (fits the head's *future* capacity, never borrows the
//!   donor's reserved pages) or `Temporal` (does not fit the future, but
//!   its own service work fits within the frontier distance and the
//!   temporal credit, so it completes before the donor's capacity is
//!   needed).
//! - **Temporal credit** — the frozen donor's remaining work; each
//!   temporal backfill admission decays it by that backfill's service work
//!   (credit decay).
//! - **Frontier distance** — the projected distance (in work quanta) to
//!   the last still-active frozen donor; a temporal candidate is only
//!   admissible while its own service work stays within it.
//!
//! Everything here is a pure value policy (no scheduler state): the
//! concrete scheduler (`concrete.rs`) drives it per advance. The
//! boundary-capture selection of the KV-RAM host tier (the reference's
//! `choose_boundary_capture` / `source_prefill_capture_frontier`) belongs
//! to core-06 / core-07 and is deliberately not part of this module.
//!
//! *Deviations from the reference, documented per ADR 0004:*
//! - ignis request ids are 0-based (`next_id` starts at 0), so the
//!   reference's `request_id == 0` invalid-state checks are dropped; all
//!   other snapshot validations are ported.
//! - The reference's three request classes (Main / Agents / Classifier)
//!   collapse to ignis's two (`Interactive` / `Agent`): retained-lane
//!   victim priority is Agent before Interactive, LRU within a class.
//! - One quantum of service work = one decode token (the batched prefill
//!   step is a global-lane operation in ignis, not per-request service
//!   work, so it consumes no quanta — unlike the reference's serialized
//!   prefill lane).

use crate::types::{BackfillClass, LaneId, RequestClass, RequestId};

/// The resources a request reserves while it holds a lane (the
/// component-wise admission arithmetic of the reference, all three
/// dimensions carried): the lane itself, its main-pool KV page
/// reservation, and its speculative-backend page reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AdmissionResources {
    /// Decode lanes held (v1: 1 per request).
    pub lanes: u32,
    /// KV pages reserved in the main pool: `ceil((prompt + token budget)
    /// / page_tokens)`. The reservation is the *full* budget (over-
    /// reservation), so the pool can never over-allocate mid-generation.
    pub kv_pages: u32,
    /// KV pages reserved in the speculative-backend pool (0 in v1 — the
    /// backend pool lands with DFlash2 / MTP, v1.2 / v1.3; the dimension
    /// is carried so the arithmetic stays the reference's).
    pub backend_pages: u32,
}

impl AdmissionResources {
    /// Component-wise sum.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            lanes: self.lanes + other.lanes,
            kv_pages: self.kv_pages + other.kv_pages,
            backend_pages: self.backend_pages + other.backend_pages,
        }
    }

    /// Component-wise subtraction. Returns `None` on underflow (a
    /// caller subtracting more than it holds is a bug, surfaced rather
    /// than silently wrapped — the reference's checked `subtract`).
    pub fn sub(&self, other: &Self) -> Option<Self> {
        Some(Self {
            lanes: self.lanes.checked_sub(other.lanes)?,
            kv_pages: self.kv_pages.checked_sub(other.kv_pages)?,
            backend_pages: self.backend_pages.checked_sub(other.backend_pages)?,
        })
    }

    /// Whether `self` (used) fits within `capacity` component-wise
    /// (independent pools: pages in one pool are not borrowable from
    /// another).
    pub fn fits(&self, capacity: &Self) -> bool {
        self.lanes <= capacity.lanes
            && self.kv_pages <= capacity.kv_pages
            && self.backend_pages <= capacity.backend_pages
    }
}

/// Faithful free-function form (reference: `admission_resources_fit`).
#[must_use]
pub fn admission_resources_fit(used: &AdmissionResources, capacity: &AdmissionResources) -> bool {
    used.fits(capacity)
}

/// A snapshot of one active (lane-holding) request, frozen for the
/// protection arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveAdmissionSnapshot {
    /// The request's id.
    pub request_id: RequestId,
    /// The resources it reserves while it holds its lane.
    pub resources: AdmissionResources,
    /// Projected remaining service work (quanta; 1 quantum per decode
    /// token in v1) — the distance to this request's completion.
    pub remaining_work_quanta: u64,
    /// The protection epoch under which the request was admitted as a
    /// backfill (0 = not a backfill of any protection).
    pub backfill_epoch: u64,
    /// The class the request was admitted under (see [`BackfillClass`]).
    pub backfill_class: BackfillClass,
}

/// A protection's phase (reference: `ProtectionPhase`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtectionPhase {
    /// Backfills may still be admitted under this protection.
    Open,
    /// The protected head fits once the current-epoch temporal borrowers
    /// are gone: no new backfills are admitted, the machine waits for the
    /// remaining donors / backfills to complete, then admits the head.
    Drain,
}

/// A frozen protection (reference: `AdmissionProtection`): the head
/// request, the frozen incumbent set, the selected donor prefix, the
/// remaining temporal credit, and the phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionProtection {
    /// The protection's epoch (monotonic; a backfill carries the epoch of
    /// the protection it was admitted under).
    pub epoch_id: u64,
    /// The blocked head this protection exists to let in.
    pub head_request_id: RequestId,
    /// The head's resource reservation.
    pub head_resources: AdmissionResources,
    /// The frozen incumbents (every active request at creation).
    pub incumbent_ids: Vec<RequestId>,
    /// The frozen donor prefix (a subset of the incumbents, in completion
    /// order). Donors are never evicted while the protection is open.
    pub donor_ids: Vec<RequestId>,
    /// The remaining temporal credit: the last (largest) frozen donor's
    /// remaining work at creation, decayed by each temporal backfill
    /// admission. A temporal candidate is admitted only while its own
    /// service work stays within it.
    pub temporal_credit: u64,
    /// Open / Drain (see [`ProtectionPhase`]).
    pub phase: ProtectionPhase,
}

impl AdmissionProtection {
    /// Whether `id` is a frozen incumbent.
    pub fn is_incumbent(&self, id: RequestId) -> bool {
        self.incumbent_ids.contains(&id)
    }

    /// Whether `id` is a frozen donor.
    pub fn is_donor(&self, id: RequestId) -> bool {
        self.donor_ids.contains(&id)
    }
}

/// Errors from freezing a protection (the reference throws; the port
/// returns so the scheduler can surface, not swallow, a broken frontier).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionError {
    /// The head reservation, the active set, or the capacity is invalid
    /// (an empty active set, a head with no lane, a head that alone
    /// exceeds the capacity).
    InvalidFrontier,
    /// The protected head is not actually blocked by the frozen
    /// incumbents (it fits alongside them) — protection is only created
    /// under contention.
    NotBlocked,
    /// An active snapshot has an invalid progress state (zero remaining
    /// work: it is completing, not an incumbent).
    InvalidSnapshot,
    /// No donor prefix releases the head (unreachable under the
    /// validations: releasing every incumbent leaves the head alone,
    /// which fits by construction).
    NoReleasingFrontier,
}

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdmissionError::InvalidFrontier => {
                write!(f, "invalid protected-admission frontier")
            }
            AdmissionError::NotBlocked => {
                write!(f, "protected head is not blocked by the frozen incumbents")
            }
            AdmissionError::InvalidSnapshot => {
                write!(f, "protected incumbent has an invalid progress state")
            }
            AdmissionError::NoReleasingFrontier => {
                write!(
                    f,
                    "exclusive-feasible head has no releasing incumbent frontier"
                )
            }
        }
    }
}

impl std::error::Error for AdmissionError {}

/// Freeze the currently active requests and select the earliest
/// projected-completion prefix whose release makes the protected head
/// feasible (reference: `make_admission_protection`).
///
/// The donors are the incumbents ordered by (remaining work, request id),
/// taken one at a time until the survivors (active + head − donors) fit
/// the capacity; the temporal credit is the last selected donor's
/// remaining work (the frontier distance at creation).
pub fn make_admission_protection(
    epoch_id: u64,
    head_request_id: RequestId,
    head_resources: AdmissionResources,
    active: &[ActiveAdmissionSnapshot],
    capacity: &AdmissionResources,
) -> Result<AdmissionProtection, AdmissionError> {
    // Validation (reference: `invalid protected-admission frontier`).
    // Ignis ids are 0-based, so no `head_request_id == 0` check
    // (documented deviation); the rest of the reference's validations
    // are ported.
    if epoch_id == 0 || head_resources.lanes == 0 || active.is_empty() {
        return Err(AdmissionError::InvalidFrontier);
    }
    if !admission_resources_fit(&head_resources, capacity) {
        return Err(AdmissionError::InvalidFrontier);
    }

    let mut out = AdmissionProtection {
        epoch_id,
        head_request_id,
        head_resources,
        incumbent_ids: Vec::with_capacity(active.len()),
        donor_ids: Vec::new(),
        temporal_credit: 0,
        phase: ProtectionPhase::Open,
    };

    let mut survivors = head_resources;
    for snapshot in active {
        if snapshot.remaining_work_quanta == 0 {
            return Err(AdmissionError::InvalidSnapshot);
        }
        out.incumbent_ids.push(snapshot.request_id);
        survivors = survivors.add(&snapshot.resources);
    }
    if survivors.fits(capacity) {
        return Err(AdmissionError::NotBlocked);
    }

    // Donor selection: the earliest-completion prefix (work ascending,
    // request id as the tie-break).
    let mut order: Vec<usize> = (0..active.len()).collect();
    order.sort_by(|&a, &b| {
        active[a]
            .remaining_work_quanta
            .cmp(&active[b].remaining_work_quanta)
            .then_with(|| active[a].request_id.cmp(&active[b].request_id))
    });

    for &i in &order {
        let donor = &active[i];
        // Survivors after releasing the donor: underflow here would mean
        // the donor holds more than the active set, a bug in the
        // caller's snapshot (the reference throws logic_error; the port
        // surfaces it as an invalid snapshot).
        survivors = match survivors.sub(&donor.resources) {
            Some(s) => s,
            None => return Err(AdmissionError::InvalidSnapshot),
        };
        out.donor_ids.push(donor.request_id);
        out.temporal_credit = donor.remaining_work_quanta;
        if survivors.fits(capacity) {
            return Ok(out);
        }
    }
    Err(AdmissionError::NoReleasingFrontier)
}

/// Tests the cumulative future-frontier invariant (reference:
/// `persistent_backfill_is_safe`): the protected head, every still-active
/// non-donor incumbent, and *every* persistent backfill admitted under
/// this protection's epoch, plus the proposed candidate, must fit the
/// capacity. Persistent backfills never borrow the donor's reserved pages.
#[must_use]
pub fn persistent_backfill_is_safe(
    protection: &AdmissionProtection,
    active: &[ActiveAdmissionSnapshot],
    candidate: &AdmissionResources,
    capacity: &AdmissionResources,
) -> bool {
    let mut future = protection.head_resources;
    for request in active {
        if protection.is_incumbent(request.request_id) {
            if !protection.is_donor(request.request_id) {
                future = future.add(&request.resources);
            }
        } else if request.backfill_epoch == protection.epoch_id
            && request.backfill_class == BackfillClass::Persistent
        {
            future = future.add(&request.resources);
        }
    }
    future = future.add(candidate);
    admission_resources_fit(&future, capacity)
}

/// Projected distance to the last still-active frozen donor (reference:
/// `protection_frontier_distance`). Later admissions never contribute
/// (only the frozen donor set does), so the distance only shrinks as
/// donors complete.
#[must_use]
pub fn protection_frontier_distance(
    protection: &AdmissionProtection,
    active: &[ActiveAdmissionSnapshot],
) -> u64 {
    active
        .iter()
        .filter(|r| protection.is_donor(r.request_id))
        .map(|r| r.remaining_work_quanta)
        .max()
        .unwrap_or(0)
}

/// True once the protected head would fit if the current-epoch temporal
/// borrowers were absent (reference: `protected_head_safe_without_temporal`).
/// This recognizes both the frozen donor frontier and an earlier
/// opportunity created by any incumbent release.
#[must_use]
pub fn protected_head_safe_without_temporal(
    protection: &AdmissionProtection,
    active: &[ActiveAdmissionSnapshot],
    capacity: &AdmissionResources,
) -> bool {
    let mut used = protection.head_resources;
    for request in active {
        if protection.is_incumbent(request.request_id)
            || request.backfill_epoch != protection.epoch_id
            || request.backfill_class != BackfillClass::Temporal
        {
            used = used.add(&request.resources);
        }
    }
    admission_resources_fit(&used, capacity)
}

/// One candidate lane's retained (evictable) state, for the retained-lane
/// victim selection (reference: `RetainedLaneCandidate`). A reservation is
/// *temporal*: it exists only while an earlier-queued Interactive request
/// has exact reusable state on the lane; reserved lanes are never victims,
/// all other retained state remains reclaimable.
///
/// v1 ships this policy (and its unit tests) for ADR 0004 reference
/// fidelity, but the concrete scheduler does **not** invoke it: retained
/// lanes only exist once the KV-RAM host tier (core-06) can snapshot a
/// lane to host RAM. The wiring lands with core-06.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetainedLaneCandidate {
    /// The lane holding the retained state.
    pub lane: LaneId,
    /// The class owning the retained state on the lane (drives the
    /// victim priority: Agent before Interactive).
    pub owner: RequestClass,
    /// The last-used tick (the LRU key within a class).
    pub use_tick: u64,
    /// Reserved for an earlier-queued Interactive request's exact
    /// reusable state — reserved lanes are not victims.
    pub reserved_for_earlier_interactive: bool,
}

/// Lowest-value eligible retained state: Agent before Interactive, LRU
/// (use tick) within a class, lane id as the final tie-break (reference:
/// `retained_lane_is_better_victim`; the reference's three-class priority
/// Classifier < Agents < Main collapses to the two v1 classes).
#[must_use]
pub fn retained_lane_is_better_victim(
    candidate: &RetainedLaneCandidate,
    incumbent: &RetainedLaneCandidate,
) -> bool {
    if candidate.reserved_for_earlier_interactive != incumbent.reserved_for_earlier_interactive {
        return !candidate.reserved_for_earlier_interactive;
    }
    if candidate.reserved_for_earlier_interactive {
        return false;
    }
    let priority = |owner: &RequestClass| match owner {
        // v1 classes: Agent (the lower-priority class) is a better
        // victim than Interactive; the reference's Classifier slot does
        // not exist in v1's class set.
        RequestClass::Agent => 1,
        RequestClass::Interactive => 2,
    };
    let candidate_priority = priority(&candidate.owner);
    let incumbent_priority = priority(&incumbent.owner);
    if candidate_priority != incumbent_priority {
        return candidate_priority < incumbent_priority;
    }
    if candidate.use_tick != incumbent.use_tick {
        return candidate.use_tick < incumbent.use_tick;
    }
    candidate.lane < incumbent.lane
}

/// Pick the lowest-value eligible retained lane to evict (reference:
/// `choose_retained_lane_victim`): `None` when every candidate lane is
/// reserved for an earlier-queued Interactive match.
pub fn choose_retained_lane_victim(
    candidates: &[RetainedLaneCandidate],
) -> Option<LaneId> {
    let mut selected: Option<&RetainedLaneCandidate> = None;
    for candidate in candidates {
        if candidate.reserved_for_earlier_interactive {
            continue;
        }
        match selected {
            None => selected = Some(candidate),
            Some(incumbent) if retained_lane_is_better_victim(candidate, incumbent) => {
                selected = Some(candidate);
            }
            _ => {}
        }
    }
    selected.map(|c| c.lane)
}

#[cfg(test)]
mod tests {
    //! The dedicated invariant tests (ADR 0004: "each invariant gets a
    //! dedicated test"), ported from the reference's
    //! `tests/test_admission_policy.cpp` (the lane / protection / credit /
    //! frontier / victim scenarios; the boundary-capture scenarios belong
    //! to the KV-RAM host tier, core-06 / core-07).

    use super::*;

    fn res(lanes: u32, kv_pages: u32, backend_pages: u32) -> AdmissionResources {
        AdmissionResources {
            lanes,
            kv_pages,
            backend_pages,
        }
    }

    fn snap(
        id: RequestId,
        resources: AdmissionResources,
        work: u64,
        epoch: u64,
        class: BackfillClass,
    ) -> ActiveAdmissionSnapshot {
        ActiveAdmissionSnapshot {
            request_id: id,
            resources,
            remaining_work_quanta: work,
            backfill_epoch: epoch,
            backfill_class: class,
        }
    }

    #[test]
    fn donor_selection_freezes_the_earliest_sufficient_incumbent() {
        // The reference's 4-lane / 160-main / 128-backend pool.
        let capacity = res(4, 160, 128);
        let head = res(1, 64, 48);
        let incumbents = [
            snap(1, res(1, 64, 32), 100, 0, BackfillClass::None),
            snap(2, res(1, 48, 64), 20, 0, BackfillClass::None),
        ];
        let protection = make_admission_protection(7, 10, head, &incumbents, &capacity)
            .expect("the head is blocked by the frozen incumbents");
        assert_eq!(protection.donor_ids, vec![2], "the earliest-completion incumbent (work 20) is the single donor");
        assert_eq!(protection.temporal_credit, 20, "the credit is the frozen donor's remaining work");
        assert_eq!(protection.incumbent_ids, vec![1, 2]);
        assert_eq!(
            protection_frontier_distance(&protection, &incumbents),
            20,
            "frontier distance follows the frozen donor"
        );
    }

    #[test]
    fn persistent_backfill_respects_the_protected_future_capacity() {
        let capacity = res(4, 160, 128);
        let head = res(1, 64, 48);
        let incumbents = [
            snap(1, res(1, 64, 32), 100, 0, BackfillClass::None),
            snap(2, res(1, 48, 64), 20, 0, BackfillClass::None),
        ];
        let protection = make_admission_protection(7, 10, head, &incumbents, &capacity).unwrap();
        // A candidate that fits the *future* (head + non-donor incumbent +
        // candidate = 3 lanes, 152 main, 120 backend ≤ capacity) is safe.
        assert!(
            persistent_backfill_is_safe(&protection, &incumbents, &res(1, 24, 40), &capacity),
            "a future-resource surplus must not reject a persistent-safe backfill"
        );
        // Borrowing the donor's reserved future (40 main pages would push
        // the future past 160) is rejected.
        assert!(
            !persistent_backfill_is_safe(&protection, &incumbents, &res(1, 40, 60), &capacity),
            "a persistent backfill must not borrow protected future capacity"
        );
    }

    #[test]
    fn the_persistent_ledger_accumulates_earlier_backfills() {
        let capacity = res(4, 160, 128);
        let head = res(1, 64, 48);
        let incumbents = [
            snap(1, res(1, 64, 32), 100, 0, BackfillClass::None),
            snap(2, res(1, 48, 64), 20, 0, BackfillClass::None),
        ];
        let protection = make_admission_protection(7, 10, head, &incumbents, &capacity).unwrap();
        // A persistent backfill admitted under epoch 7 (24 main pages)
        // occupies future capacity: a new 9-main candidate no longer fits.
        let with_persistent = [
            incumbents[0],
            incumbents[1],
            snap(3, res(1, 24, 40), 50, 7, BackfillClass::Persistent),
        ];
        assert!(
            !persistent_backfill_is_safe(&protection, &with_persistent, &res(1, 9, 9), &capacity),
            "the persistent ledger must accumulate earlier backfills"
        );
    }

    #[test]
    fn later_temporal_work_does_not_move_the_frozen_frontier() {
        let capacity = res(4, 160, 128);
        let head = res(1, 64, 48);
        let incumbents = [
            snap(1, res(1, 64, 32), 100, 0, BackfillClass::None),
            snap(2, res(1, 48, 64), 20, 0, BackfillClass::None),
        ];
        let protection = make_admission_protection(7, 10, head, &incumbents, &capacity).unwrap();
        // The donor (id 2) has completed; a *later* temporal borrower (id
        // 4, same epoch) must not move the frozen frontier — the distance
        // drops to 0 (no still-active frozen donors).
        let after_donor = [
            incumbents[0],
            snap(4, res(1, 32, 64), 8, 7, BackfillClass::Temporal),
        ];
        assert_eq!(
            protection_frontier_distance(&protection, &after_donor),
            0,
            "later temporal work must not move the frozen frontier"
        );
        assert!(
            protected_head_safe_without_temporal(&protection, &after_donor, &capacity),
            "the head is safe without the current-epoch temporal borrowers"
        );
    }

    #[test]
    fn independent_kv_pools_are_not_interchangeable() {
        let capacity = res(4, 160, 128);
        assert!(
            !res(1, 161, 1).fits(&capacity),
            "main-pool overflow must not fit"
        );
        assert!(
            !res(1, 1, 129).fits(&capacity),
            "backend-pool overflow must not fit"
        );
    }

    // The C=3 analog of the reference's shared-pool scenarios: a 3-lane /
    // 10-page pool, 4-page (large) and 2-page (tiny) requests.
    const POOL: AdmissionResources = AdmissionResources {
        lanes: 3,
        kv_pages: 10,
        backend_pages: 0,
    };
    const LARGE: AdmissionResources = AdmissionResources {
        lanes: 1,
        kv_pages: 4,
        backend_pages: 0,
    };
    const TINY: AdmissionResources = AdmissionResources {
        lanes: 1,
        kv_pages: 2,
        backend_pages: 0,
    };

    fn two_large() -> [ActiveAdmissionSnapshot; 2] {
        [
            snap(11, LARGE, 48, 0, BackfillClass::None),
            snap(12, LARGE, 40, 0, BackfillClass::None),
        ]
    }

    #[test]
    fn fits_now_backfill_into_leftover_capacity() {
        let two_large = two_large();
        let protection = make_admission_protection(3, 13, LARGE, &two_large, &POOL)
            .expect("the head is blocked by the frozen incumbents");
        assert_eq!(
            protection.donor_ids,
            vec![12],
            "the earliest 4-page donor (work 40) is frozen"
        );
        // A 2-page backfill fits the leftover (head 4 + non-donor 4 +
        // candidate 2 = 10 ≤ 10) with a free lane: safe.
        assert!(
            persistent_backfill_is_safe(&protection, &two_large, &TINY, &POOL),
            "a fits-now backfill must be admitted with leftover pages and a free lane"
        );
        // A 4-page candidate would steal the donor's reserved future.
        assert!(
            !persistent_backfill_is_safe(&protection, &two_large, &LARGE, &POOL),
            "a candidate may not backfill into the protected head's leftover"
        );
        assert!(
            !persistent_backfill_is_safe(&protection, &two_large, &res(1, 3, 0), &POOL),
            "a 3-page candidate steals a leftover page from the protected head"
        );
    }

    #[test]
    fn a_second_backfill_sees_the_first_persistent_occupant() {
        let two_large = two_large();
        let protection = make_admission_protection(3, 13, LARGE, &two_large, &POOL).unwrap();
        // The first tiny backfill (epoch 3, persistent) now occupies 2 of
        // the 10 pages: a second tiny candidate no longer fits.
        let after_tiny = [
            two_large[0],
            two_large[1],
            snap(14, TINY, 2, 3, BackfillClass::Persistent),
        ];
        assert!(
            !persistent_backfill_is_safe(&protection, &after_tiny, &TINY, &POOL),
            "the second backfill must see the first persistent occupant"
        );
    }

    #[test]
    fn a_leftover_zero_occupancy_cannot_spare_pages() {
        let two_five = [
            snap(21, res(1, 5, 0), 10, 0, BackfillClass::None),
            snap(22, res(1, 5, 0), 8, 0, BackfillClass::None),
        ];
        let protection = make_admission_protection(4, 23, res(1, 5, 0), &two_five, &POOL).unwrap();
        assert_eq!(protection.donor_ids, vec![22], "the work-8 donor is frozen");
        // 5 + 5 occupancy spares no pages: even the 2-page backfill
        // borrows the donor's reserved future.
        assert!(
            !persistent_backfill_is_safe(&protection, &two_five, &TINY, &POOL),
            "a leftover-0 occupancy cannot spare pages for a backfill"
        );
    }

    #[test]
    fn occupied_lanes_block_the_page_fit() {
        let two_lanes = res(2, 10, 0);
        let two_large = two_large();
        let protection =
            make_admission_protection(5, 33, LARGE, &two_large, &two_lanes).unwrap();
        assert!(
            !persistent_backfill_is_safe(&protection, &two_large, &TINY, &two_lanes),
            "the 2-page leftover fit must still respect the lane dimension"
        );
    }

    #[test]
    fn retained_lane_victims_prefer_agents_then_lru() {
        // The reference's 4-candidate scenario, mapped onto the v1 class
        // set (Agent = the lower-priority class, Interactive = the
        // protected foreground class):
        let mut retained = [
            RetainedLaneCandidate {
                lane: 0,
                owner: RequestClass::Interactive,
                use_tick: 1,
                reserved_for_earlier_interactive: false,
            },
            RetainedLaneCandidate {
                lane: 1,
                owner: RequestClass::Agent,
                use_tick: 9,
                reserved_for_earlier_interactive: false,
            },
            RetainedLaneCandidate {
                lane: 2,
                owner: RequestClass::Agent,
                use_tick: 4,
                reserved_for_earlier_interactive: false,
            },
            RetainedLaneCandidate {
                lane: 3,
                owner: RequestClass::Agent,
                use_tick: 6,
                reserved_for_earlier_interactive: false,
            },
        ];
        assert_eq!(
            choose_retained_lane_victim(&retained),
            Some(2),
            "the Agent lane with the lowest use tick (LRU) is the first victim"
        );
        // Reserving a lane pins it: the victim moves to the next-eligible
        // Agent lane, in LRU order.
        retained[2].reserved_for_earlier_interactive = true;
        assert_eq!(
            choose_retained_lane_victim(&retained),
            Some(3),
            "a reserved lane is not a victim"
        );
        retained[3].reserved_for_earlier_interactive = true;
        assert_eq!(
            choose_retained_lane_victim(&retained),
            Some(1),
            "after the Agent reservations the last Agent lane is the victim"
        );
        retained[1].reserved_for_earlier_interactive = true;
        retained[0].reserved_for_earlier_interactive = true;
        assert_eq!(
            choose_retained_lane_victim(&retained),
            None,
            "pending-interactive reservations pin every matching lane"
        );
    }

    #[test]
    fn retained_lane_ties_break_on_lane_id() {
        let candidates = [
            RetainedLaneCandidate {
                lane: 5,
                owner: RequestClass::Agent,
                use_tick: 7,
                reserved_for_earlier_interactive: false,
            },
            RetainedLaneCandidate {
                lane: 3,
                owner: RequestClass::Agent,
                use_tick: 7,
                reserved_for_earlier_interactive: false,
            },
        ];
        assert_eq!(
            choose_retained_lane_victim(&candidates),
            Some(3),
            "same class and tick: the lower lane id wins"
        );
    }
}