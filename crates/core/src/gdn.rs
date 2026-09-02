//! GDN (linear-attention) state management — `core-02`.
//!
//! The recurrent state of the linear-attention (GDN) layers is resumable
//! **only** at checkpoint / frontier boundaries (`CONTEXT.md` "GDN state").
//! The actual state matrices live in the GPU kernel leaf; this module tracks
//! the CPU-side invariant: which positions are resumable boundaries. The
//! KV-RAM host tier (core-06) must honor it — a snapshot taken mid-prefill is
//! invalid for GDN layers, so the host tier may only snapshot a lane at a
//! recorded boundary.

/// The GDN recurrent-state resumability tracker.
///
/// Holds the set of checkpoint / frontier boundaries at which the state is
/// resumable, and the state's current position. This is the CPU-side model of
/// the invariant "GDN state is resumable only at checkpoint / frontier
/// boundaries" — the host tier (core-06) consults
/// [`GdnState::is_valid_snapshot_point`] before snapshotting a lane to host
/// RAM.
#[derive(Debug, Clone, Default)]
pub struct GdnState {
    /// Token positions at which the state is resumable (checkpoint /
    /// frontier boundaries). A snapshot at any other position is invalid.
    boundaries: Vec<usize>,
    /// The current position (token index) of the state.
    position: usize,
}

impl GdnState {
    /// A fresh state: position 0 is the initial boundary (the start of the
    /// sequence is always resumable).
    pub fn new() -> Self {
        Self {
            boundaries: vec![0],
            position: 0,
        }
    }

    /// Record a checkpoint / frontier boundary: the state becomes resumable
    /// at `position`. The frontier only moves forward — a checkpoint at or
    /// past the current position is recorded (checking the current position
    /// makes it resumable); backwards checkpoints are ignored.
    pub fn checkpoint(&mut self, position: usize) {
        if position >= self.position {
            self.boundaries.push(position);
            self.position = position;
        }
    }

    /// Advance the state's position without recording a boundary (mid-
    /// prefill / mid-decode progress): the state moves forward but is **not
    /// resumable** at the new position — a snapshot taken mid-prefill (at an
    /// un-checkpointed position) is invalid for GDN layers (core-06: the
    /// host tier may only snapshot at a recorded boundary). The frontier
    /// only moves forward — duplicate or backwards advances are ignored.
    pub fn advance(&mut self, position: usize) {
        if position > self.position {
            self.position = position;
        }
    }

    /// The state's current position (token index).
    pub fn position(&self) -> usize {
        self.position
    }

    /// Whether the state can be resumed at `position` — true only for a
    /// recorded checkpoint / frontier boundary.
    pub fn can_resume_at(&self, position: usize) -> bool {
        self.boundaries.contains(&position)
    }

    /// Whether `position` is exactly at a recorded boundary and thus a
    /// valid snapshot point for the host tier (core-06). A snapshot mid-
    /// prefill (between boundaries) is **not** valid — GDN state is only
    /// resumable at a recorded checkpoint / frontier boundary.
    pub fn is_valid_snapshot_point(&self, position: usize) -> bool {
        self.can_resume_at(position)
    }

    /// Restore the state to a boundary position. Fails (returns `false`) when
    /// `position` is not a recorded boundary (e.g. mid-prefill) — a
    /// mid-prefill snapshot is invalid for GDN layers.
    pub fn restore(&mut self, position: usize) -> bool {
        if !self.can_resume_at(position) {
            return false;
        }
        self.position = position;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_position_is_a_boundary() {
        let state = GdnState::new();
        assert_eq!(state.position(), 0);
        assert!(state.can_resume_at(0));
        // Restoring to the start is valid.
        let mut state = GdnState::new();
        assert!(state.restore(0));
    }

    #[test]
    fn resumable_only_at_boundaries() {
        let mut state = GdnState::new();
        // Checkpoint at a frontier boundary (e.g. end of a prefill window).
        state.checkpoint(512);
        assert!(state.can_resume_at(512));
        // A mid-prefill position (between boundaries) is NOT resumable.
        assert!(!state.can_resume_at(511));
        assert!(!state.can_resume_at(513));
        assert!(!state.restore(511), "mid-prefill restore must fail");
        // A boundary restore succeeds.
        assert!(state.restore(512));
        assert_eq!(state.position(), 512);
    }

    #[test]
    fn host_tier_respects_the_gdn_boundary() {
        // The host tier (core-06) may snapshot a lane only at a valid GDN
        // snapshot point. A mid-prefill snapshot must be rejected.
        let mut state = GdnState::new();
        state.checkpoint(256);
        // A snapshot at a boundary is valid.
        assert!(state.is_valid_snapshot_point(256));
        // A snapshot mid-prefill (not a boundary) is invalid.
        assert!(!state.is_valid_snapshot_point(128));
    }

    #[test]
    fn checkpoint_at_current_position_records_a_boundary() {
        // core-06: `checkpoint` uses `>=` (not `>`), so recording a
        // checkpoint at the *current* position makes that position resumable
        // (a decode step lands exactly on the current position and records a
        // boundary there, so the host tier can snapshot a lane at it).
        let mut state = GdnState::new();
        state.checkpoint(64); // position 0 -> 64 (a boundary at 64)
        assert_eq!(state.position(), 64);
        // A checkpoint at the current position (64) is accepted (>=, not >):
        // it records 64 as a boundary and leaves the position unchanged.
        state.checkpoint(64);
        assert!(state.is_valid_snapshot_point(64), "the current position is a boundary");
        assert_eq!(state.position(), 64);
        // A checkpoint *ahead* of the current position also records a
        // boundary and moves the position forward.
        state.checkpoint(128);
        assert!(state.is_valid_snapshot_point(128));
        assert_eq!(state.position(), 128);
    }

    #[test]
    fn advance_moves_without_recording_a_boundary() {
        // core-02: `advance` moves the position forward without recording a
        // boundary (mid-prefill / mid-decode progress). The new position is
        // *not* resumable — a snapshot taken there is invalid for GDN layers.
        let mut state = GdnState::new();
        state.advance(128); // mid-prefill: position 0 -> 128 (no boundary at 128)
        assert_eq!(state.position(), 128);
        assert!(
            !state.is_valid_snapshot_point(128),
            "a mid-prefill (advanced) position is not a snapshot point"
        );
        // The initial boundary (0) is still resumable.
        assert!(state.is_valid_snapshot_point(0));
        // A checkpoint *after* the advance records a new boundary.
        state.checkpoint(256);
        assert!(state.is_valid_snapshot_point(256));
        assert_eq!(state.position(), 256);
    }

    #[test]
    fn advance_is_monotonic() {
        // The frontier only moves forward: duplicate or backwards advances
        // are ignored (the position never moves backwards).
        let mut state = GdnState::new();
        state.advance(64);
        assert_eq!(state.position(), 64);
        // A duplicate (64) or backwards (32) advance is ignored.
        state.advance(64);
        assert_eq!(state.position(), 64);
        state.advance(32);
        assert_eq!(state.position(), 64, "a backwards advance is ignored");
        // A forward advance (128) is applied.
        state.advance(128);
        assert_eq!(state.position(), 128);
    }
}
