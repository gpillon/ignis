//! The KV-RAM host tier — `core-06`.
//!
//! The host-RAM KV cache tier (`CONTEXT.md`: "KV-RAM"): snapshots GPU lanes
//! so sibling requests **restore** instead of re-prefilling, with a
//! **two-tier eviction** (probation → protected). This is the CPU-side state
//! model the concrete scheduler drives: the actual H2D / D2H copies live in
//! the kernel leaf (the GPU-coupled part behind the
//! [`crate::scheduler::Compute`] seam, ADR 0006), so the whole tier — and the
//! scheduler's evict / restore policy on top of it — is CPU-testable without
//! a GPU.
//!
//! The directions a snapshot travels:
//! - **GPU → host (capture / evict-to-tier)** — the scheduler snapshots a
//!   running lane into the tier to free its lane + pages (the overflow path:
//!   admitting beyond N=8 resident lanes).
//! - **host → GPU (restore)** — a suspended (evicted) request is brought
//!   back onto a lane; it resumes from where it was evicted (no re-prefill).
//! - **host → discard (tier eviction)** — when the tier itself fills, it
//!   evicts its lowest-value entries to make room; a discarded snapshot's
//!   warmed KV is lost (the request re-prefills later).
//!
//! **Discard ordering (ADR 0023, GitHub #127):** request class first (Agent
//! before Interactive — [`crate::types::RequestClass::eviction_rank`], the
//! same ranking GPU residency uses), then probation before protected, then
//! least-recently-used. This is deliberately the *opposite* primary key
//! from the GPU ([`crate::admission`]'s eligibility-and-protection-first
//! ordering): nothing on the host tier is actively being served, so class —
//! whose work costs most to lose — is the only thing left to lead with. An
//! `Interactive` snapshot in probation outlives an `Agent` snapshot in
//! protected.
//!
//! Two-tier eviction (probation → protected) is still the *second* key: a
//! freshly captured snapshot starts in **probation** (discarded before any
//! same-class protected entry, like a fresh page on an LRU "inactive"
//! list). A snapshot that has been **restored** at least once has proven
//! its value and is placed in the **protected** tier (discarded only after
//! every same-class probation entry, like an "active" page).
//!
//! The tier honors the GDN boundary (core-02): a snapshot is only valid at a
//! recorded checkpoint / frontier boundary, so [`HostTier::capture`] rejects
//! a mid-prefill (non-boundary) GDN position.
//!
//! **The tier is bounded by a byte budget, not a page or lane count**
//! (P4-07, GitHub #125, spec `04-reference-feature-floor.md`): a snapshot's
//! floor is the GDN slot + conv taps + penalty row (a fixed ~145 MiB,
//! regardless of prompt length) while its KV plane scales with pages, so a
//! page count alone misprices a short sequence against a full-context one.
//! [`HostEntry::bytes`] is what [`Compute::evict`](crate::scheduler::Compute::evict)
//! actually wrote to pinned host memory; this module never allocates or
//! moves those bytes itself — it only accounts for them, keeping the tier
//! CPU-testable without a GPU (see the module-level note above on the
//! `Compute` seam).

use std::collections::HashSet;

use crate::gdn::GdnState;
use crate::types::{LaneId, RequestClass, RequestId};

/// The two-tier eviction tiers of the host tier (`CONTEXT.md`: "two-tier
/// eviction (probation → protected)"). Request class (ADR 0023, GitHub
/// #127) is the *primary* discard key; this tier is the secondary one — see
/// the module-level doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// A freshly captured snapshot: discarded before any same-class
    /// protected entry (like a fresh page on an LRU "inactive" list).
    Probation,
    /// A snapshot proven worth retaining (restored at least once):
    /// discarded only after every same-class probation entry (like an
    /// "active" page, evicted last).
    Protected,
}

/// Which lifecycle phase a snapshot resumes into (P4-07, GitHub #125): the
/// two shapes [`Request::evict`](crate::request::Request::evict) /
/// [`Request::evict_prefilling`](crate::request::Request::evict_prefilling)
/// suspend, and the two
/// [`Request::restore_lane`](crate::request::Request::restore_lane) /
/// [`Request::restore_prefilling`](crate::request::Request::restore_prefilling)
/// resume. A restricted two-variant discriminant rather than the general
/// `RequestState` on purpose: a host-tier entry can only ever have been
/// evicted from one of these two states, so this makes the other three
/// states unrepresentable here instead of merely unreachable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumePhase {
    /// Evicted mid-prefill (half-prefilled, GitHub #125): held no decode
    /// lane, and restores back into `Prefilling` — [`HostEntry::prefill_progress`]
    /// is where it resumes chunking from, not from zero. It re-earns a
    /// decode lane the normal way once prefill completes.
    Prefilling,
    /// Evicted from a decode lane: restores straight back onto one and
    /// resumes generation with no re-prefill at all.
    Running,
}

/// A GPU-resident snapshot captured into the host tier: everything needed
/// to **restore** the request without re-prefilling (its KV page
/// reservation, its generation and prefill progress, and the GDN state at
/// a valid boundary).
#[derive(Debug, Clone)]
pub struct HostEntry {
    /// The suspended (evicted) request this snapshot belongs to.
    pub request: RequestId,
    /// Which phase this snapshot resumes into (P4-07, GitHub #125): decides
    /// whether restore hands the request a decode lane or puts it back into
    /// `Prefilling` with no lane at all.
    pub resume_phase: ResumePhase,
    /// The decode lane it was evicted from (KV block mapping / telemetry),
    /// or `None` for a half-prefilled ([`ResumePhase::Prefilling`]) entry,
    /// which held no lane to begin with.
    pub lane: Option<LaneId>,
    /// The class owning the request (retained-lane victim priority: Agent
    /// before Interactive — see [`crate::admission`]).
    pub owner: RequestClass,
    /// The main-pool KV pages the request holds (re-charged on the GPU pool
    /// at restore — unrelated to this tier's own budget, which is bytes,
    /// not pages).
    pub pages: u32,
    /// The snapshot's real size in pinned host memory (P4-07, GitHub #125):
    /// what [`Compute::evict`](crate::scheduler::Compute::evict) actually
    /// wrote. This is the tier's own budget unit ([`HostTier::capacity_bytes`]).
    pub bytes: u64,
    /// Tokens generated so far (the request resumes from here — no
    /// re-prefill).
    pub tokens: u32,
    /// Prompt tokens already sent to the compute backend at the moment of
    /// eviction (P4-07, GitHub #125; mirrors
    /// [`crate::request::Request::prefill_progress`]): a half-prefilled
    /// request resumes chunking from here, not from zero. Meaningless (and
    /// unused) for a [`ResumePhase::Running`] entry, whose prefill was
    /// already complete when it was evicted.
    pub prefill_progress: u32,
    /// Remaining service work (quanta; 1 quantum per decode token) — frozen
    /// while suspended.
    pub remaining_work: u64,
    /// The GDN recurrent state at the snapshot point (core-02): the
    /// boundary set + position. The snapshot is only valid at a recorded
    /// boundary (a mid-prefill position is invalid for GDN layers).
    pub gdn: GdnState,
    /// The eviction tier this snapshot sits in.
    pub tier: Tier,
    /// The last-use tick (the LRU key within a tier: the oldest capture is
    /// evicted / restored first).
    pub use_tick: u64,
}

/// Errors from capturing a lane into the host tier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostError {
    /// The snapshot's GDN position is not a resumable boundary (core-02: GDN
    /// state is resumable only at a checkpoint / frontier boundary — a
    /// mid-prefill snapshot is invalid for GDN layers).
    InvalidSnapshotPoint,
    /// The snapshot alone exceeds the tier's host-RAM byte budget (it can
    /// never be held, even alone).
    Oversized,
}

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HostError::InvalidSnapshotPoint => {
                write!(f, "the snapshot's GDN position is not a resumable boundary")
            }
            HostError::Oversized => {
                write!(f, "the snapshot alone exceeds the host tier's byte budget")
            }
        }
    }
}

impl std::error::Error for HostError {}

/// `Probation` ranks before `Protected` in the discard ordering (ADR 0023,
/// GitHub #127's secondary key — see the module-level doc).
fn tier_discard_rank(tier: Tier) -> u8 {
    match tier {
        Tier::Probation => 0,
        Tier::Protected => 1,
    }
}

/// True when `candidate` is the lower discard-value entry between
/// `candidate` and `incumbent` under the host tier's ordering (ADR 0023,
/// GitHub #127): request class first (Agent before Interactive — the same
/// [`RequestClass::eviction_rank`] GPU residency uses), then probation
/// before protected, then least-recently-used (`use_tick` ascending).
fn is_better_discard_victim(candidate: &HostEntry, incumbent: &HostEntry) -> bool {
    let candidate_rank = candidate.owner.eviction_rank();
    let incumbent_rank = incumbent.owner.eviction_rank();
    if candidate_rank != incumbent_rank {
        return candidate_rank < incumbent_rank;
    }
    let candidate_tier = tier_discard_rank(candidate.tier);
    let incumbent_tier = tier_discard_rank(incumbent.tier);
    if candidate_tier != incumbent_tier {
        return candidate_tier < incumbent_tier;
    }
    candidate.use_tick < incumbent.use_tick
}

/// The bounded host-RAM KV tier: holds evicted (suspended) request
/// snapshots in two tiers (probation → protected), discards its lowest-
/// value entries when full (class first, then probation before protected,
/// then LRU — ADR 0023, GitHub #127), and restores a snapshot to the GPU on
/// request.
pub struct HostTier {
    /// The host-RAM budget in bytes (P4-07, GitHub #125): the tier never
    /// holds more than this many bytes of snapshots — entries are evicted
    /// to keep it bounded.
    capacity_bytes: u64,
    /// Probation entries (LRU order: oldest capture at the front, evicted
    /// first).
    probation: Vec<HostEntry>,
    /// Protected entries (LRU order: oldest capture at the front, evicted
    /// after every probation entry).
    protected: Vec<HostEntry>,
    /// The tier's current host-RAM usage, in bytes (the sum of every held
    /// entry's [`HostEntry::bytes`]).
    used_bytes: u64,
    /// Requests proven worth retaining (restored at least once): their next
    /// capture lands directly in the protected tier.
    promoted: HashSet<RequestId>,
}

impl HostTier {
    /// A host tier holding `capacity_bytes` of host-RAM snapshot budget
    /// (P4-07, GitHub #125). The tier never holds more than `capacity_bytes`
    /// bytes: entries are evicted (lowest discard value first — class, then
    /// tier, then LRU; ADR 0023, GitHub #127) to make room (bounded
    /// eviction — the tier does not grow without bound).
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            probation: Vec::new(),
            protected: Vec::new(),
            used_bytes: 0,
            promoted: HashSet::new(),
        }
    }

    /// The tier's host-RAM budget, in bytes.
    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    /// The tier's current host-RAM usage, in bytes.
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// The number of snapshots currently held (both tiers).
    pub fn entry_count(&self) -> usize {
        self.probation.len() + self.protected.len()
    }

    /// Whether the tier holds a snapshot for `request`.
    pub fn contains(&self, request: RequestId) -> bool {
        self.probation.iter().any(|e| e.request == request)
            || self.protected.iter().any(|e| e.request == request)
    }

    /// The lowest-value entry (the first to be discarded from host RAM):
    /// request class first (Agent before Interactive), then probation
    /// before protected, then least-recently-used (ADR 0023, GitHub #127).
    /// Peeks without removing.
    pub fn victim(&self) -> Option<&HostEntry> {
        let mut selected: Option<&HostEntry> = None;
        for entry in self.probation.iter().chain(self.protected.iter()) {
            selected = match selected {
                None => Some(entry),
                Some(incumbent) if is_better_discard_victim(entry, incumbent) => Some(entry),
                _ => selected,
            };
        }
        selected
    }

    /// Capture a lane's state into the tier (GPU → host). Fails with
    /// [`HostError::InvalidSnapshotPoint`] when the snapshot's GDN position
    /// is not a resumable boundary (core-02), and
    /// [`HostError::Oversized`] when the snapshot alone exceeds the byte
    /// budget. On success the entry is placed in the **protected** tier
    /// when the request has been proven worth retaining (restored before),
    /// else in **probation**; entries are evicted (lowest discard value
    /// first — [`Self::victim`]'s ordering) to make room, keeping the tier
    /// within its budget.
    pub fn capture(&mut self, mut entry: HostEntry) -> Result<(), HostError> {
        // core-02: the GDN boundary invariant — a mid-prefill (non-
        // boundary) snapshot is invalid for GDN layers.
        if !entry.gdn.is_valid_snapshot_point(entry.gdn.position()) {
            return Err(HostError::InvalidSnapshotPoint);
        }
        // A snapshot that alone exceeds the tier can never be held.
        if entry.bytes > self.capacity_bytes {
            return Err(HostError::Oversized);
        }
        // A previously-restored request re-enters as protected (proven); a
        // fresh capture starts in probation.
        let tier = if self.promoted.contains(&entry.request) {
            Tier::Protected
        } else {
            Tier::Probation
        };
        // Make room: evict the lowest discard-value entries ([`Self::victim`]'s
        // ordering) until the new snapshot fits. An empty tier fits any
        // snapshot ≤ the budget, so the loop always terminates with room
        // available.
        while self.used_bytes + entry.bytes > self.capacity_bytes {
            // Unreachable (an empty tier fits a snapshot ≤ the budget);
            // guard against an infinite loop regardless.
            if self.evict_one().is_none() {
                return Err(HostError::Oversized);
            }
        }
        entry.tier = tier;
        let bytes = entry.bytes;
        match tier {
            Tier::Protected => self.protected.push(entry),
            Tier::Probation => self.probation.push(entry),
        }
        self.used_bytes += bytes;
        Ok(())
    }

    /// Discard the lowest-value entry (host RAM → discard): [`Self::victim`]'s
    /// ordering (class, then probation before protected, then LRU). The
    /// discarded entry's warmed KV is lost (the request re-prefills later).
    /// Returns the discarded entry, or `None` when the tier is empty.
    pub fn evict_one(&mut self) -> Option<HostEntry> {
        let request = self.victim()?.request;
        self.remove(request)
    }

    /// Restore a request from the tier (host RAM → GPU). Removes its
    /// snapshot, records it as proven worth retaining (its next capture
    /// lands in the protected tier), and returns the snapshot (the caller
    /// re-charges the GPU pool and re-acquires a lane). Returns `None` when
    /// the request is not in the tier.
    pub fn restore(&mut self, request: RequestId) -> Option<HostEntry> {
        let entry = self.remove(request)?;
        // A restored request has proven its value: its next snapshot is
        // protected (evicted last).
        self.promoted.insert(request);
        Some(entry)
    }

    /// Discard a specific request's snapshot without promoting it (P4-07,
    /// GitHub #125): the scheduler's fallback when a physical restore
    /// attempt fails (a corrupt/foreign blob, or a leaf-level error) — the
    /// snapshot is dropped from the tier exactly as [`HostTier::evict_one`]
    /// would drop it, and the request falls back to re-prefilling. Returns
    /// the discarded entry, or `None` when `request` is not in the tier.
    pub fn discard_request(&mut self, request: RequestId) -> Option<HostEntry> {
        self.remove(request)
    }

    /// Remove a request's snapshot from whichever tier holds it, updating
    /// the usage accounting. Returns the removed entry, or `None` when the
    /// request is not in the tier.
    fn remove(&mut self, request: RequestId) -> Option<HostEntry> {
        if let Some(i) = self.probation.iter().position(|e| e.request == request) {
            let entry = self.probation.remove(i);
            self.used_bytes = self.used_bytes.saturating_sub(entry.bytes);
            return Some(entry);
        }
        if let Some(i) = self.protected.iter().position(|e| e.request == request) {
            let entry = self.protected.remove(i);
            self.used_bytes = self.used_bytes.saturating_sub(entry.bytes);
            return Some(entry);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn::GdnState;

    /// A GDN state resumable at `position` (a recorded boundary): a valid
    /// snapshot point. Position 0 is always a boundary.
    fn gdn_boundary(position: usize) -> GdnState {
        let mut gdn = GdnState::new();
        gdn.checkpoint(position);
        gdn
    }

    /// A GDN state mid-prefill (position not a boundary): an *invalid*
    /// snapshot point (a mid-prefill snapshot is invalid for GDN layers).
    fn gdn_mid_prefill(position: usize) -> GdnState {
        let mut gdn = GdnState::new();
        gdn.advance(position); // moves forward without recording a boundary
        gdn
    }

    fn entry(request: u64, bytes: u64, gdn: GdnState, tick: u64) -> HostEntry {
        HostEntry {
            request,
            resume_phase: ResumePhase::Running,
            lane: Some(0),
            owner: RequestClass::Agent,
            pages: 1, // GPU-pool accounting, unrelated to this tier's byte budget
            bytes,
            tokens: 0,
            prefill_progress: 0,
            remaining_work: 8,
            gdn,
            tier: Tier::Probation,
            use_tick: tick,
        }
    }

    #[test]
    fn fresh_snapshots_start_in_probation() {
        let mut tier = HostTier::new(100);
        tier.capture(entry(1, 10, gdn_boundary(0), 0)).unwrap();
        tier.capture(entry(2, 10, gdn_boundary(0), 1)).unwrap();
        assert_eq!(tier.entry_count(), 2);
        // Both are probation (never restored): the victim is the oldest
        // capture (LRU).
        assert_eq!(tier.victim().unwrap().request, 1);
        assert_eq!(tier.used_bytes(), 20);
    }

    #[test]
    fn gdn_mid_prefill_snapshot_is_rejected() {
        let mut tier = HostTier::new(100);
        // A mid-prefill snapshot (position not a boundary) must be rejected
        // (core-02: a mid-prefill snapshot is invalid for GDN layers).
        let e = entry(1, 10, gdn_mid_prefill(128), 0);
        assert_eq!(tier.capture(e), Err(HostError::InvalidSnapshotPoint));
        assert_eq!(
            tier.entry_count(),
            0,
            "a mid-prefill snapshot is not stored"
        );
    }

    #[test]
    fn gdn_boundary_snapshot_is_accepted() {
        let mut tier = HostTier::new(100);
        // A snapshot at a recorded boundary is valid.
        tier.capture(entry(1, 10, gdn_boundary(512), 0)).unwrap();
        assert_eq!(tier.entry_count(), 1);
        assert!(tier.contains(1));
    }

    #[test]
    fn two_tier_eviction_probation_before_protected() {
        // A 30-byte tier with 20-byte snapshots: only one fits at a time.
        let mut tier = HostTier::new(30);
        tier.capture(entry(1, 20, gdn_boundary(0), 0)).unwrap(); // probation
        tier.capture(entry(2, 20, gdn_boundary(0), 1)).unwrap(); // evicts 1 (probation LRU)
        assert_eq!(
            tier.entry_count(),
            1,
            "only one 20-byte snapshot fits a 30-byte tier"
        );
        assert_eq!(tier.victim().unwrap().request, 2);
        // Promote request 2 (restore, so its next capture is protected).
        tier.restore(2).unwrap();
        tier.capture(entry(2, 20, gdn_boundary(0), 2)).unwrap(); // protected (proven)
        // A new probation 20-byte snapshot cannot coexist with the
        // protected 20-byte snapshot (30 < 40): the protected entry is
        // evicted as a last resort.
        tier.capture(entry(3, 20, gdn_boundary(0), 3)).unwrap();
        assert_eq!(tier.entry_count(), 1);
        assert_eq!(
            tier.victim().unwrap().request,
            3,
            "the probation entry is the victim (the protected entry was evicted last)"
        );
    }

    #[test]
    fn evict_one_drops_probation_before_protected() {
        let mut tier = HostTier::new(100);
        tier.capture(entry(1, 10, gdn_boundary(0), 0)).unwrap(); // probation
        tier.restore(1).unwrap(); // promote: next capture is protected
        tier.capture(entry(1, 10, gdn_boundary(0), 1)).unwrap(); // protected
        tier.capture(entry(2, 10, gdn_boundary(0), 2)).unwrap(); // probation
        // Two entries: 1 (protected) + 2 (probation). Evicting one drops
        // the probation entry (2) first.
        let evicted = tier.evict_one().unwrap();
        assert_eq!(evicted.request, 2, "probation is evicted before protected");
        assert_eq!(tier.entry_count(), 1);
        assert_eq!(
            tier.victim().unwrap().request,
            1,
            "the protected entry remains"
        );
    }

    #[test]
    fn usage_is_bounded_by_capacity() {
        // A 40-byte tier with 10-byte snapshots: at most 4 fit. The tier
        // evicts (probation LRU) to stay within the budget — it never
        // exceeds `capacity_bytes`.
        let mut tier = HostTier::new(40);
        for i in 0..10u64 {
            tier.capture(entry(i, 10, gdn_boundary(0), i)).unwrap();
            assert!(
                tier.used_bytes() <= tier.capacity_bytes(),
                "the tier must never exceed its budget (used {} > {})",
                tier.used_bytes(),
                tier.capacity_bytes()
            );
        }
        assert_eq!(
            tier.used_bytes(),
            40,
            "the tier holds exactly its budget worth of bytes"
        );
        assert_eq!(
            tier.entry_count(),
            4,
            "only four 10-byte snapshots fit a 40-byte tier"
        );
    }

    #[test]
    fn an_oversized_snapshot_is_rejected() {
        let mut tier = HostTier::new(10);
        // A 20-byte snapshot exceeds the 10-byte budget: it can never be
        // held.
        assert_eq!(
            tier.capture(entry(1, 20, gdn_boundary(0), 0)),
            Err(HostError::Oversized)
        );
        assert_eq!(tier.entry_count(), 0);
    }

    #[test]
    fn restore_returns_the_snapshot_and_promotes() {
        let mut tier = HostTier::new(100);
        tier.capture(entry(1, 10, gdn_boundary(0), 0)).unwrap();
        let snap = tier.restore(1).unwrap();
        assert_eq!(snap.request, 1);
        assert!(!tier.contains(1), "a restored request leaves the tier");
        // A restored request is promoted: its next capture is protected.
        tier.capture(entry(1, 10, gdn_boundary(0), 1)).unwrap();
        assert_eq!(
            tier.victim().unwrap().tier,
            Tier::Protected,
            "a proven request's snapshot is protected"
        );
        // Restoring a request not in the tier returns None.
        assert!(tier.restore(99).is_none());
    }

    #[test]
    fn a_half_prefilled_entry_carries_its_resume_phase_and_progress() {
        // P4-07, GitHub #125: a half-prefilled (Prefilling) eviction holds
        // no lane and must resume chunking from its snapshotted progress,
        // not from zero.
        let mut tier = HostTier::new(100);
        let half_prefilled = HostEntry {
            resume_phase: ResumePhase::Prefilling,
            lane: None,
            prefill_progress: 384,
            ..entry(1, 10, gdn_boundary(384), 0)
        };
        tier.capture(half_prefilled).unwrap();
        let snap = tier.restore(1).unwrap();
        assert_eq!(snap.resume_phase, ResumePhase::Prefilling);
        assert_eq!(snap.lane, None);
        assert_eq!(
            snap.prefill_progress, 384,
            "resumes from the snapshotted chunk boundary, not from zero"
        );
    }

    #[test]
    fn discard_request_removes_without_promoting() {
        // P4-07, GitHub #125: the scheduler's fallback when a physical
        // restore fails — the snapshot leaves the tier but the request is
        // never marked "proven" (it did not actually resume).
        let mut tier = HostTier::new(100);
        tier.capture(entry(1, 10, gdn_boundary(0), 0)).unwrap();
        let discarded = tier.discard_request(1).unwrap();
        assert_eq!(discarded.request, 1);
        assert!(!tier.contains(1));
        assert_eq!(tier.used_bytes(), 0);
        // Not promoted: a fresh capture lands back in probation.
        tier.capture(entry(1, 10, gdn_boundary(0), 1)).unwrap();
        assert_eq!(tier.victim().unwrap().tier, Tier::Probation);
        // Discarding a request not in the tier returns None.
        assert!(tier.discard_request(99).is_none());
    }

    // ── ADR 0023 / GitHub #127: class-first discard ordering ───────────

    #[test]
    fn an_interactive_snapshot_in_probation_outlives_an_agent_snapshot_in_protected() {
        // The asymmetry test ADR 0023 calls out by name: on the host tier
        // nothing is being served, so class outranks tier — the opposite
        // of the GPU, where protection outranks class.
        let mut tier = HostTier::new(100);
        // Agent request 1: capture, restore (promotes it), re-capture —
        // lands in protected.
        tier.capture(entry(1, 10, gdn_boundary(0), 0)).unwrap();
        tier.restore(1).unwrap();
        tier.capture(entry(1, 10, gdn_boundary(0), 1)).unwrap();
        assert_eq!(
            tier.victim().unwrap().tier,
            Tier::Protected,
            "sanity: request 1 (Agent) is protected"
        );

        // Interactive request 2: a fresh capture — lands in probation, and
        // is captured *after* request 1, so it is also the more-recently-
        // used entry (never mind LRU, class alone must decide this).
        tier.capture(HostEntry {
            owner: RequestClass::Interactive,
            ..entry(2, 10, gdn_boundary(0), 2)
        })
        .unwrap();

        assert_eq!(
            tier.victim().unwrap().request,
            1,
            "the protected Agent snapshot is discarded first: class outranks tier"
        );
    }

    #[test]
    fn a_protected_lane_is_not_evicted_ahead_of_an_unprotected_lower_class() {
        // AC: a protected lane mid-stream is not evicted ahead of an
        // unprotected one of a lower class — the mirror of the asymmetry
        // test, phrased from the protected entry's side.
        let mut tier = HostTier::new(100);
        tier.capture(HostEntry {
            owner: RequestClass::Interactive,
            ..entry(1, 10, gdn_boundary(0), 0)
        })
        .unwrap();
        tier.restore(1).unwrap();
        tier.capture(HostEntry {
            owner: RequestClass::Interactive,
            ..entry(1, 10, gdn_boundary(0), 1)
        })
        .unwrap(); // protected Interactive
        tier.capture(entry(2, 10, gdn_boundary(0), 2)).unwrap(); // probation Agent (default)

        assert_eq!(
            tier.victim().unwrap().request,
            2,
            "the probation Agent snapshot is the victim; the protected Interactive one is not evicted ahead of it"
        );
    }

    #[test]
    fn a_single_class_workload_reproduces_probation_before_protected() {
        // AC: with every request defaulting to `Interactive` (no tags in
        // play), the ordering reproduces today's (pre-#127) behavior
        // exactly — class ties collapse to tier, then LRU. Exercised here
        // with `Interactive` (the wire default) rather than the `entry()`
        // helper's `Agent`, mirroring `two_tier_eviction_probation_before_protected`.
        let mut tier = HostTier::new(30);
        let interactive = |id, tick| HostEntry {
            owner: RequestClass::Interactive,
            ..entry(id, 20, gdn_boundary(0), tick)
        };
        tier.capture(interactive(1, 0)).unwrap();
        tier.capture(interactive(2, 1)).unwrap(); // evicts 1 (probation LRU)
        assert_eq!(tier.entry_count(), 1);
        assert_eq!(tier.victim().unwrap().request, 2);
        tier.restore(2).unwrap();
        tier.capture(interactive(2, 2)).unwrap(); // protected (proven)
        tier.capture(interactive(3, 3)).unwrap(); // probation
        assert_eq!(tier.entry_count(), 1);
        assert_eq!(
            tier.victim().unwrap().request,
            3,
            "same class throughout: the probation entry is the victim, exactly as before #127"
        );
    }
}
