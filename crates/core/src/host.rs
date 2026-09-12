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
//!   evicts its lowest-value entries (probation first, then protected) to
//!   make room; a discarded snapshot's warmed KV is lost (the request
//!   re-prefills later).
//!
//! Two-tier eviction (probation → protected): a freshly captured snapshot
//! starts in **probation** (first to be discarded when the tier fills, like
//! a fresh page on an LRU "inactive" list). A snapshot that has been
//! **restored** at least once has proven its value and is placed in the
//! **protected** tier (evicted last, like an "active" page).
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
/// eviction (probation → protected)").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// A freshly captured snapshot: first in line to be evicted from host
    /// RAM when the tier fills (like a new page on the LRU "inactive"
    /// list).
    Probation,
    /// A snapshot proven worth retaining (restored at least once): evicted
    /// only after every probation entry (like an "active" page, evicted
    /// last).
    Protected,
}

/// A GPU-lane snapshot captured into the host tier: everything needed to
/// **restore** the request without re-prefilling (its KV page reservation,
/// its generation progress, and the GDN state at a valid boundary).
#[derive(Debug, Clone)]
pub struct HostEntry {
    /// The suspended (evicted) request this snapshot belongs to.
    pub request: RequestId,
    /// The decode lane it was evicted from (KV block mapping / telemetry).
    pub lane: LaneId,
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

/// The bounded host-RAM KV tier: holds evicted (suspended) request
/// snapshots in two tiers (probation → protected), evicts probation entries
/// first when full, and restores a snapshot to the GPU on request.
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
    /// bytes: entries are evicted (probation first, then protected) to make
    /// room (bounded eviction — the tier does not grow without bound).
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
    /// the oldest probation entry, then the oldest protected entry. Peeks
    /// without removing.
    pub fn victim(&self) -> Option<&HostEntry> {
        self.probation.first().or_else(|| self.protected.first())
    }

    /// Capture a lane's state into the tier (GPU → host). Fails with
    /// [`HostError::InvalidSnapshotPoint`] when the snapshot's GDN position
    /// is not a resumable boundary (core-02), and
    /// [`HostError::Oversized`] when the snapshot alone exceeds the byte
    /// budget. On success the entry is placed in the **protected** tier
    /// when the request has been proven worth retaining (restored before),
    /// else in **probation**; entries are evicted (probation first) to make
    /// room, keeping the tier within its budget.
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
        // Make room: evict the lowest-value entries (probation LRU first)
        // until the new snapshot fits. An empty tier fits any snapshot ≤
        // the budget, so the loop always terminates with room available.
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

    /// Discard the lowest-value entry (host RAM → discard): the oldest
    /// probation entry first, then the oldest protected entry. The
    /// discarded entry's warmed KV is lost (the request re-prefills later).
    /// Returns the discarded entry, or `None` when the tier is empty.
    pub fn evict_one(&mut self) -> Option<HostEntry> {
        let entry = if !self.probation.is_empty() {
            self.probation.remove(0)
        } else if !self.protected.is_empty() {
            self.protected.remove(0)
        } else {
            return None; // the tier is empty (nothing to evict)
        };
        self.used_bytes = self.used_bytes.saturating_sub(entry.bytes);
        Some(entry)
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
            lane: 0,
            owner: RequestClass::Agent,
            pages: 1, // GPU-pool accounting, unrelated to this tier's byte budget
            bytes,
            tokens: 0,
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
}
