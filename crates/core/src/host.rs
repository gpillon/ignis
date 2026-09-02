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
    /// The main-pool KV pages the request holds (held in host RAM while
    /// suspended; re-charged on the GPU pool at restore).
    pub pages: u32,
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
    /// The snapshot alone exceeds the tier's host-RAM capacity (it can never
    /// be held, even alone).
    Oversized,
}

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HostError::InvalidSnapshotPoint => {
                write!(f, "the snapshot's GDN position is not a resumable boundary")
            }
            HostError::Oversized => {
                write!(f, "the snapshot alone exceeds the host tier's capacity")
            }
        }
    }
}

impl std::error::Error for HostError {}

/// The bounded host-RAM KV tier: holds evicted (suspended) request
/// snapshots in two tiers (probation → protected), evicts probation entries
/// first when full, and restores a snapshot to the GPU on request.
pub struct HostTier {
    /// The host-RAM budget in KV pages (the tier never holds more than this
    /// many pages — entries are evicted to keep it bounded).
    capacity_pages: u32,
    /// Probation entries (LRU order: oldest capture at the front, evicted
    /// first).
    probation: Vec<HostEntry>,
    /// Protected entries (LRU order: oldest capture at the front, evicted
    /// after every probation entry).
    protected: Vec<HostEntry>,
    /// The tier's current host-RAM usage (pages held by all entries).
    used_pages: u32,
    /// Requests proven worth retaining (restored at least once): their next
    /// capture lands directly in the protected tier.
    promoted: HashSet<RequestId>,
}

impl HostTier {
    /// A host tier holding `capacity_pages` pages of host-RAM KV budget.
    /// The tier never holds more than `capacity_pages` pages: entries are
    /// evicted (probation first, then protected) to make room (bounded
    /// eviction — the tier does not grow without bound).
    pub fn new(capacity_pages: u32) -> Self {
        Self {
            capacity_pages,
            probation: Vec::new(),
            protected: Vec::new(),
            used_pages: 0,
            promoted: HashSet::new(),
        }
    }

    /// The tier's host-RAM budget (pages).
    pub fn capacity_pages(&self) -> u32 {
        self.capacity_pages
    }

    /// The tier's current host-RAM usage (pages held by all entries).
    pub fn used_pages(&self) -> u32 {
        self.used_pages
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
    /// [`HostError::Oversized`] when the snapshot alone exceeds the
    /// capacity. On success the entry is placed in the **protected** tier
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
        if entry.pages > self.capacity_pages {
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
        // the capacity, so the loop always terminates with room available.
        while self.used_pages + entry.pages > self.capacity_pages {
            // Unreachable (an empty tier fits a snapshot ≤ capacity); guard
            // against an infinite loop regardless.
            if self.evict_one().is_none() {
                return Err(HostError::Oversized);
            }
        }
        entry.tier = tier;
        let pages = entry.pages;
        match tier {
            Tier::Protected => self.protected.push(entry),
            Tier::Probation => self.probation.push(entry),
        }
        self.used_pages += pages;
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
        self.used_pages = self.used_pages.saturating_sub(entry.pages);
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

    /// Remove a request's snapshot from whichever tier holds it, updating
    /// the usage accounting. Returns the removed entry, or `None` when the
    /// request is not in the tier.
    fn remove(&mut self, request: RequestId) -> Option<HostEntry> {
        if let Some(i) = self.probation.iter().position(|e| e.request == request) {
            let entry = self.probation.remove(i);
            self.used_pages = self.used_pages.saturating_sub(entry.pages);
            return Some(entry);
        }
        if let Some(i) = self.protected.iter().position(|e| e.request == request) {
            let entry = self.protected.remove(i);
            self.used_pages = self.used_pages.saturating_sub(entry.pages);
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

    fn entry(request: u64, pages: u32, gdn: GdnState, tick: u64) -> HostEntry {
        HostEntry {
            request,
            lane: 0,
            owner: RequestClass::Agent,
            pages,
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
        assert_eq!(tier.used_pages(), 20);
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
        // A 30-page tier with 20-page snapshots: only one fits at a time.
        let mut tier = HostTier::new(30);
        tier.capture(entry(1, 20, gdn_boundary(0), 0)).unwrap(); // probation
        tier.capture(entry(2, 20, gdn_boundary(0), 1)).unwrap(); // evicts 1 (probation LRU)
        assert_eq!(
            tier.entry_count(),
            1,
            "only one 20-page snapshot fits a 30-page tier"
        );
        assert_eq!(tier.victim().unwrap().request, 2);
        // Promote request 2 (restore, so its next capture is protected).
        tier.restore(2).unwrap();
        tier.capture(entry(2, 20, gdn_boundary(0), 2)).unwrap(); // protected (proven)
        // A new probation 20-page snapshot cannot coexist with the
        // protected 20-page snapshot (30 < 40): the protected entry is
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
        // A 40-page tier with 10-page snapshots: at most 4 fit. The tier
        // evicts (probation LRU) to stay within the budget — it never
        // exceeds `capacity_pages`.
        let mut tier = HostTier::new(40);
        for i in 0..10u64 {
            tier.capture(HostEntry {
                request: i,
                lane: 0,
                owner: RequestClass::Agent,
                pages: 10,
                tokens: 0,
                remaining_work: 8,
                gdn: gdn_boundary(0),
                tier: Tier::Probation,
                use_tick: i,
            })
            .unwrap();
            assert!(
                tier.used_pages() <= tier.capacity_pages(),
                "the tier must never exceed its capacity (used {} > {})",
                tier.used_pages(),
                tier.capacity_pages()
            );
        }
        assert_eq!(
            tier.used_pages(),
            40,
            "the tier holds exactly its budget worth of pages"
        );
        assert_eq!(
            tier.entry_count(),
            4,
            "only four 10-page snapshots fit a 40-page tier"
        );
    }

    #[test]
    fn an_oversized_snapshot_is_rejected() {
        let mut tier = HostTier::new(10);
        // A 20-page snapshot exceeds the 10-page capacity: it can never be
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
}
