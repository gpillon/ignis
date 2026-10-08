//! KV-disk, Tier 2 — the ledger (spec vram-budget/03, ADR 0045).
//!
//! What the disk holds, what it costs, and what goes first. The bytes live in
//! files behind the [`Compute`](crate::scheduler::Compute) seam; this module
//! only accounts for them, so the policy is CPU-testable exactly as KV-RAM's
//! is ([`crate::host`]).
//!
//! **The order is KV-RAM's, extended — and it is the same code.** The held
//! entries are a [`HostTier`]: retained state before evicted live sequences,
//! then class, then probation before protected, then least recently used,
//! with the Interactive TTL (ADR 0023 as amended by 0045). The spec asks for
//! the victim order to be shared rather than copied so the two tiers cannot
//! drift, and the plainest way to share it is to hold the same type twice.
//!
//! Two rules make the disk's use of it differ from KV-RAM's:
//!
//! - **No live work is discarded for room.** A live blob is given room only
//!   out of retained entries ([`HostTier::plan_live_room`]); with none left
//!   to give up, the spill does not happen and the request that wanted the
//!   room waits. KV-RAM's `capture`, which discards live entries, is never
//!   called on a disk ledger without the room already made.
//! - **A file is charged from the moment its spill starts** (it is
//!   *landing*): the volume's bytes are being written, and a second spill
//!   must not count them as free. A spill that fails or is cancelled gives
//!   them back, so the ledger's used bytes return to what they were before
//!   the transfer (spec AC 21).
//!
//! The byte figure is the file's, not the blob's: a 4 KiB header page and the
//! blob padded to the unbuffered-IO alignment ([`disk_file_bytes`]).

use std::time::{Duration, Instant};

use crate::host::{HostEntry, HostError, HostTier, RetainedBlob, RetainedKvRamEntry};
use crate::scheduler::DiskBlob;
use crate::types::{RequestClass, RequestId};

/// A KV-disk file's header page (spec vram-budget/03): written last, as the
/// commit.
pub const DISK_HEADER_BYTES: u64 = 4096;

/// The unbuffered-IO alignment a file's body is padded to — the sector
/// multiple `ignis_artifact::DIRECT_IO_ALIGNMENT` names.
pub const DISK_IO_ALIGNMENT: u64 = 4096;

/// The volume bytes a file holding a blob of `blob_bytes` takes: the header
/// page, then the blob, its last window padded to the alignment.
pub fn disk_file_bytes(blob_bytes: u64) -> u64 {
    DISK_HEADER_BYTES + blob_bytes.div_ceil(DISK_IO_ALIGNMENT) * DISK_IO_ALIGNMENT
}

/// The KV-disk ledger: committed files in KV-RAM's order, and the files
/// being written.
pub struct DiskTier {
    held: HostTier,
    /// Files being written, by blob, and the bytes each will hold.
    landing: Vec<(DiskBlob, u64)>,
}

impl DiskTier {
    /// A ledger of `capacity_bytes` (the load's effective budget), whose
    /// Interactive entries keep their class's rank for `interactive_ttl`
    /// after their last use.
    pub fn new(capacity_bytes: u64, interactive_ttl: Duration) -> Self {
        let mut held = HostTier::new(capacity_bytes);
        held.set_retained_interactive_ttl(interactive_ttl);
        Self {
            held,
            landing: Vec::new(),
        }
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.held.capacity_bytes()
    }

    /// Bytes in use: committed files and files being written.
    pub fn used_bytes(&self) -> u64 {
        self.held.used_bytes() + self.landing_bytes()
    }

    fn landing_bytes(&self) -> u64 {
        self.landing.iter().map(|&(_, bytes)| bytes).sum()
    }

    /// The committed files, in the order they go.
    pub fn held(&self) -> &HostTier {
        &self.held
    }

    pub fn held_mut(&mut self) -> &mut HostTier {
        &mut self.held
    }

    /// Which retained files to delete so a live blob's file of `bytes` fits
    /// beside everything held and landing, or `None` when deleting every
    /// retained file would not be enough: live files are never candidates.
    pub fn plan_live_room(&self, bytes: u64, now: Instant) -> Option<Vec<RetainedBlob>> {
        self.held.plan_live_room(bytes.checked_add(self.landing_bytes())?, now)
    }

    /// Which retained files to delete so a retained blob's file of `bytes`,
    /// of `owner` last used at `used_at`, fits: only files ranking strictly
    /// below it (KV-RAM's rule, [`HostTier::plan_retained_room`]).
    pub fn plan_retained_room(
        &self,
        bytes: u64,
        owner: RequestClass,
        used_at: Instant,
        now: Instant,
    ) -> Option<Vec<RetainedBlob>> {
        self.held
            .plan_retained_room(bytes.checked_add(self.landing_bytes())?, owner, used_at, now)
    }

    /// A spill of `blob` started: its file's `bytes` are charged now.
    pub fn begin_landing(&mut self, blob: DiskBlob, bytes: u64) {
        debug_assert!(!self.is_landing(blob), "one spill of a blob at a time");
        self.landing.push((blob, bytes));
    }

    /// The spill of `blob` ended, whichever way it went: its charge is
    /// lifted, and returned. A blob not landing is `None`.
    pub fn end_landing(&mut self, blob: DiskBlob) -> Option<u64> {
        let pos = self.landing.iter().position(|&(b, _)| b == blob)?;
        Some(self.landing.swap_remove(pos).1)
    }

    pub fn is_landing(&self, blob: DiskBlob) -> bool {
        self.landing.iter().any(|&(b, _)| b == blob)
    }

    /// A live blob's file committed: `entry` (its bytes the file's) joins
    /// the held files. Its landing charge must already have been lifted, so
    /// the room was made when the spill started and nothing is discarded
    /// here.
    pub fn commit_live(&mut self, entry: HostEntry) -> Result<(), HostError> {
        if self.used_bytes().saturating_add(entry.bytes) > self.capacity_bytes() {
            return Err(HostError::Oversized);
        }
        self.held.capture(entry)
    }

    /// A retained blob's file committed.
    pub fn commit_retained(&mut self, entry: RetainedKvRamEntry) -> Result<(), HostError> {
        if self.used_bytes().saturating_add(entry.bytes) > self.capacity_bytes() {
            return Err(HostError::Oversized);
        }
        self.held.capture_retained(entry)
    }

    /// Whether `request`'s live blob has a committed file here.
    pub fn holds_live(&self, request: RequestId) -> bool {
        self.held.contains(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn::GdnState;
    use crate::host::{ResumePhase, Tier};

    fn live(request: RequestId, bytes: u64, owner: RequestClass) -> HostEntry {
        HostEntry {
            request,
            resume_phase: ResumePhase::Running,
            lane: Some(0),
            owner,
            pages: 1,
            bytes,
            tokens: 0,
            prefill_progress: 0,
            remaining_work: 8,
            gdn: GdnState::new(),
            tier: Tier::Probation,
            use_tick: request,
        }
    }

    fn retained(id: u64, owner: RequestClass, bytes: u64, at: Instant) -> RetainedKvRamEntry {
        RetainedKvRamEntry::new(RetainedBlob::Checkpoint(id), id * 10, owner, bytes, at)
    }

    #[test]
    fn a_file_is_its_header_page_and_its_blob_padded_to_a_sector() {
        assert_eq!(disk_file_bytes(1), 4096 + 4096);
        assert_eq!(disk_file_bytes(4096), 4096 + 4096);
        assert_eq!(disk_file_bytes(4097), 4096 + 8192);
        assert_eq!(disk_file_bytes(32 << 20), 4096 + (32 << 20));
    }

    #[test]
    fn a_landing_file_is_charged_until_it_commits_or_fails() {
        let mut disk = DiskTier::new(100, Duration::from_secs(300));
        disk.begin_landing(DiskBlob::Live(1), 60);
        assert_eq!(disk.used_bytes(), 60, "a file being written is charged");
        assert!(disk.plan_live_room(50, Instant::now()).is_none(), "and a second spill cannot count its bytes free");
        assert_eq!(disk.end_landing(DiskBlob::Live(1)), Some(60));
        assert_eq!(disk.used_bytes(), 0, "a failed or cancelled spill gives the bytes back (AC 21)");
        assert_eq!(disk.end_landing(DiskBlob::Live(1)), None);

        disk.begin_landing(DiskBlob::Live(2), 60);
        disk.end_landing(DiskBlob::Live(2));
        disk.commit_live(live(2, 60, RequestClass::Agent)).unwrap();
        assert_eq!(disk.used_bytes(), 60);
        assert!(disk.holds_live(2));
    }

    #[test]
    fn a_live_blob_takes_room_from_retained_files_only() {
        let now = Instant::now();
        let mut disk = DiskTier::new(100, Duration::from_secs(300));
        disk.commit_live(live(1, 40, RequestClass::Agent)).unwrap();
        disk.commit_retained(retained(7, RequestClass::Interactive, 30, now)).unwrap();
        disk.commit_retained(retained(8, RequestClass::Agent, 20, now)).unwrap();
        // 10 free: a 30-byte live blob needs the Agent file, a 40-byte one
        // the Interactive one after it -- retained state of any class goes
        // before live work.
        assert_eq!(disk.plan_live_room(30, now), Some(vec![RetainedBlob::Checkpoint(8)]));
        assert_eq!(
            disk.plan_live_room(40, now),
            Some(vec![RetainedBlob::Checkpoint(8), RetainedBlob::Checkpoint(7)])
        );
        // 70 would need the live file too: no live work is discarded for room.
        assert_eq!(disk.plan_live_room(70, now), None);
        assert_eq!(disk.used_bytes(), 90, "planning changes nothing");
    }

    #[test]
    fn a_retained_blob_displaces_only_what_ranks_below_it() {
        let now = Instant::now();
        let later = now + Duration::from_secs(1);
        let mut disk = DiskTier::new(30, Duration::from_secs(300));
        disk.commit_retained(retained(1, RequestClass::Agent, 10, now)).unwrap();
        disk.commit_retained(retained(2, RequestClass::Interactive, 10, now)).unwrap();
        disk.commit_live(live(3, 10, RequestClass::Agent)).unwrap();
        assert_eq!(
            disk.plan_retained_room(10, RequestClass::Agent, later, later),
            Some(vec![RetainedBlob::Checkpoint(1)]),
            "an Agent newcomer displaces the older Agent file"
        );
        assert_eq!(
            disk.plan_retained_room(20, RequestClass::Agent, later, later),
            None,
            "never the Interactive one above it, nor the live file"
        );
    }

    #[test]
    fn the_ledger_never_discards_a_live_file_to_commit_one() {
        let mut disk = DiskTier::new(50, Duration::from_secs(300));
        disk.commit_live(live(1, 40, RequestClass::Agent)).unwrap();
        assert_eq!(
            disk.commit_live(live(2, 40, RequestClass::Interactive)),
            Err(HostError::Oversized),
            "a commit with no room refuses rather than discarding live work"
        );
        assert!(disk.holds_live(1));
    }
}
