//! KV-disk transfers -- [`RuntimeCompute`]'s half of the `Compute` seam's disk
//! calls (spec vram-budget/03, ADR 0045).
//!
//! A transfer is a blob moved a window at a time between the device (or a
//! KV-RAM span) and its file, pumped by [`Compute::disk_advance`] between
//! steps: at most one new window per transfer per call, each window a copy
//! on the leaf's transfer stream and an IO request on the store's threads,
//! each polled -- never waited for -- on the model thread.
//!
//! - **A spill from the device** copies window `i` into a staging slot (D2H),
//!   and once its fence has passed hands the slot to the writer, which takes
//!   the window's CRC and writes it; the other slot meanwhile takes the next
//!   window. The sequence is untouched until the header page -- written last,
//!   with every window's CRC -- commits the file; only then is it released.
//! - **A spill from KV-RAM** writes straight from the blob's span (aligned,
//!   since the arena places blobs on 4 KiB boundaries), and the span is the
//!   transfer's until the commit, which frees it.
//! - **A restore** allocates its sequence at once (the scheduler charged it),
//!   reads and checks the header page, then reads each window into a slot,
//!   checks its CRC and feeds it to the leaf's windowed restore. A refused
//!   header or window ends it before that window's bytes reach the device.
//!
//! A transfer that ends any other way than committing is **drained** before
//! anything it touches is let go: its copies' fences and its IO requests'
//! tickets are polled to completion, then its slots come back, its sequence
//! is released and its file deleted. The one wait the model thread ever pays
//! is on a cancelled restore's copies already issued into the sequence it is
//! about to release -- PCIe, never the disk.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use ignis_artifact::{AlignedBuffer, DirectReader, DirectWriter};
use ignis_core::scheduler::{DiskBlob, DiskBlobMeta, DiskEvent, DiskOp, DiskOutcome, DiskSource, DiskTarget};
use ignis_core::{ComputeError, RequestId};

use crate::kv_disk::{self, BlobKind, Bytes, DiskStore, FileHeader, Job, Ticket, HEADER_BYTES};
use crate::{LiveSequence, RuntimeCompute, RuntimeError, StepLeaf, TransferWindow};

/// A pinned staging window: [`kv_disk::WINDOW_BYTES`] of host memory on an
/// unbuffered-IO boundary, which a device copy lands in and the disk reads or
/// writes from.
pub type StagingWindow = Box<dyn AsMut<[u8]> + Send>;

/// One of the staging windows, and whether a window of some transfer is in
/// it.
struct Slot {
    buffer: StagingWindow,
    busy: bool,
}

/// A committed file: where it is, what its header says, and -- for a live
/// sequence -- how far it had generated.
pub(crate) struct DiskFile {
    path: PathBuf,
    header: FileHeader,
    generated: u32,
}

/// One window of a transfer, in flight.
enum Op {
    /// A spill window copied from the device into `slot`, until `fence`.
    Copying { index: usize, slot: usize, fence: u64 },
    /// A spill window being written (from `slot`, or straight from the
    /// transfer's own KV-RAM span).
    Writing { index: usize, slot: Option<usize>, ticket: Ticket },
    /// A restore window being read into `slot`.
    Reading { index: usize, slot: usize, ticket: Ticket },
    /// A restore window copied from `slot` to the device, until `fence`.
    Feeding { slot: usize, fence: u64 },
}

enum Kind<L: StepLeaf> {
    Spill {
        from: DiskSource,
        writer: Option<Arc<DirectWriter>>,
        /// A KV-RAM blob's span, the transfer's until it ends.
        source: Option<L::SnapshotBuf>,
        generated: u32,
        meta: DiskBlobMeta,
        crcs: Vec<u32>,
        header: Option<(AlignedBuffer, Ticket)>,
    },
    Restore {
        into: DiskTarget,
        reader: Arc<DirectReader>,
        expected: FileHeader,
        header: Option<(AlignedBuffer, Ticket)>,
        validated: bool,
        sequence: Option<L::Sequence>,
        generated: u32,
    },
}

struct Transfer<L: StepLeaf> {
    blob: DiskBlob,
    path: PathBuf,
    blob_bytes: u64,
    windows: usize,
    issued: usize,
    landed: usize,
    ops: VecDeque<Op>,
    started: Instant,
    kind: Kind<L>,
    /// Whether a drained transfer's file goes too: not for an abandoned
    /// checkpoint claim, whose file is a retained one.
    delete_file: bool,
}

impl<L: StepLeaf> Transfer<L> {
    fn window(&self, index: usize, window_bytes: u64) -> TransferWindow {
        let offset = index as u64 * window_bytes;
        TransferWindow {
            offset,
            bytes: window_bytes.min(self.blob_bytes - offset),
            blob_bytes: self.blob_bytes,
        }
    }
}

/// The disk tier's state in a [`RuntimeCompute`]: the store, its staging, the
/// files it has committed, the transfers under way and the ones draining.
pub(crate) struct DiskTier<L: StepLeaf> {
    pub(crate) store: DiskStore,
    slots: Vec<Slot>,
    pub(crate) files: HashMap<DiskBlob, DiskFile>,
    transfers: Vec<Transfer<L>>,
    draining: Vec<Transfer<L>>,
}

impl<L: StepLeaf> DiskTier<L> {
    pub(crate) fn new(store: DiskStore, staging: Vec<StagingWindow>) -> Self {
        Self {
            store,
            slots: staging.into_iter().map(|buffer| Slot { buffer, busy: false }).collect(),
            files: HashMap::new(),
            transfers: Vec::new(),
            draining: Vec::new(),
        }
    }

    fn free_slot(&mut self) -> Option<usize> {
        let at = self.slots.iter().position(|s| !s.busy)?;
        self.slots[at].busy = true;
        Some(at)
    }

    fn slot_bytes(&mut self, slot: usize) -> Bytes {
        let buffer = self.slots[slot].buffer.as_mut().as_mut();
        Bytes {
            ptr: buffer.as_mut_ptr(),
            len: buffer.len(),
        }
    }

    /// The file bytes the spills under way will hold.
    fn landing_bytes(&self) -> u64 {
        self.transfers
            .iter()
            .filter(|t| matches!(t.kind, Kind::Spill { .. }))
            .map(|t| ignis_core::disk::disk_file_bytes(t.blob_bytes))
            .sum()
    }
}

fn leaf_err(code: i32) -> ComputeError {
    RuntimeError::Leaf(code).into()
}

impl<L: StepLeaf> RuntimeCompute<L> {
    pub(crate) fn tier_fits(&self, bytes: u64) -> bool {
        let tier = self.disk.lock().unwrap();
        let Some(tier) = tier.as_ref() else {
            return false;
        };
        tier.store.fits(ignis_core::disk::disk_file_bytes(bytes), tier.landing_bytes())
    }

    pub(crate) fn tier_spill(&self, blob: DiskBlob, from: DiskSource, meta: DiskBlobMeta) -> Result<u64, ComputeError> {
        let mut guard = self.disk.lock().unwrap();
        let tier = guard.as_mut().ok_or(ComputeError::Kernel(-1))?;
        if tier.transfers.iter().any(|t| t.blob == blob) || tier.files.contains_key(&blob) {
            return Err(ComputeError::Kernel(-1));
        }
        // What the blob is, and -- from KV-RAM -- its span, which the
        // transfer owns until it ends.
        let (blob_bytes, source, generated) = match (blob, from) {
            (DiskBlob::Live(request), DiskSource::Device) => {
                let sequences = self.sequences.lock().unwrap();
                let live = sequences.get(&request).ok_or(ComputeError::Kernel(-1))?;
                let bytes = self
                    .model
                    .leaf
                    .snapshot_bytes(self.model.handle(), &live.handle)
                    .map_err(leaf_err)?;
                (bytes, None, live.generated)
            }
            (DiskBlob::Live(request), DiskSource::KvRam) => {
                let evicted = self.evicted.lock().unwrap().remove(&request).ok_or(ComputeError::Kernel(-1))?;
                (evicted.buf.as_ref().len() as u64, Some(evicted.buf), evicted.generated)
            }
            (DiskBlob::Checkpoint(publisher), DiskSource::KvRam) => {
                let buf = self.retained.lock().unwrap().remove(&publisher).ok_or(ComputeError::Kernel(-1))?;
                (buf.as_ref().len() as u64, Some(buf), 0)
            }
            (DiskBlob::Prefix(publisher, tokens), DiskSource::KvRam) => {
                let buf = self
                    .spilled_prefixes
                    .lock()
                    .unwrap()
                    .remove(&(publisher, tokens))
                    .ok_or(ComputeError::Kernel(-1))?;
                (buf.as_ref().len() as u64, Some(buf), 0)
            }
            // Retained state never goes to the disk straight from the device
            // (it would hold pages a live request needs while it moved).
            _ => return Err(ComputeError::Kernel(-1)),
        };
        let window_bytes = tier.store.window_bytes;
        let windows = kv_disk::window_count(blob_bytes, window_bytes);
        let path = tier.store.dir.file(blob);
        let created = (windows <= kv_disk::MAX_WINDOWS && blob_bytes > 0)
            .then(|| DirectWriter::create(&path))
            .ok_or_else(|| format!("a {blob_bytes}-byte blob does not fit a file's header"))
            .and_then(|w| w.map_err(|e| e.to_string()));
        let writer = match created {
            Ok(writer) => writer,
            Err(error) => {
                // The span goes back where it came from: nothing moved.
                if let Some(buf) = source {
                    self.put_back(blob, buf, generated);
                }
                tracing::warn!(name: "ignis.kv_disk.create_failed", %error, path = %path.display(), "a KV-disk file could not be created");
                return Err(ComputeError::Kernel(-1));
            }
        };
        tier.transfers.push(Transfer {
            blob,
            path,
            blob_bytes,
            windows,
            issued: 0,
            landed: 0,
            ops: VecDeque::new(),
            started: Instant::now(),
            kind: Kind::Spill {
                from,
                writer: Some(Arc::new(writer)),
                source,
                generated,
                meta,
                crcs: vec![0; windows],
                header: None,
            },
            delete_file: true,
        });
        Ok(blob_bytes)
    }

    /// A KV-RAM blob back into its map: a spill that did not happen.
    fn put_back(&self, blob: DiskBlob, buf: L::SnapshotBuf, generated: u32) {
        match blob {
            DiskBlob::Live(request) => {
                self.evicted
                    .lock()
                    .unwrap()
                    .insert(request, crate::EvictedSequence { buf, generated });
            }
            DiskBlob::Checkpoint(publisher) => {
                self.retained.lock().unwrap().insert(publisher, buf);
            }
            DiskBlob::Prefix(publisher, tokens) => {
                self.spilled_prefixes.lock().unwrap().insert((publisher, tokens), buf);
            }
        }
    }

    pub(crate) fn tier_restore(&self, blob: DiskBlob, into: DiskTarget) -> Result<(), ComputeError> {
        let mut guard = self.disk.lock().unwrap();
        let tier = guard.as_mut().ok_or(ComputeError::Kernel(-1))?;
        let file = tier.files.get(&blob).ok_or(ComputeError::Kernel(-1))?;
        let DiskTarget::Sequence { context_tokens, .. } = into else {
            // A prefix comes back only from KV-RAM in this version.
            return Err(ComputeError::Kernel(-1));
        };
        let reader = DirectReader::open(&file.path).map_err(|_| ComputeError::Kernel(-1))?;
        let sequence = self
            .model
            .leaf
            .allocate_sequence(self.model.handle(), context_tokens)
            .map_err(leaf_err)?;
        let generated = match blob {
            DiskBlob::Live(_) => file.generated,
            _ => 0,
        };
        let (path, expected) = (file.path.clone(), file.header.clone());
        let windows = expected.crcs.len();
        let blob_bytes = expected.blob_bytes;
        let mut page = AlignedBuffer::new(HEADER_BYTES).map_err(|_| ComputeError::Kernel(-1))?;
        let reader = Arc::new(reader);
        let bytes = Bytes {
            ptr: page.as_mut_slice().as_mut_ptr(),
            len: HEADER_BYTES,
        };
        let ticket = tier.store.io.submit(Job::Read {
            file: Arc::clone(&reader),
            offset: 0,
            bytes,
            crc_len: 0,
        });
        tier.transfers.push(Transfer {
            blob,
            path,
            blob_bytes,
            windows,
            issued: 0,
            landed: 0,
            ops: VecDeque::new(),
            started: Instant::now(),
            kind: Kind::Restore {
                into,
                reader,
                expected,
                header: Some((page, ticket)),
                validated: false,
                sequence: Some(sequence),
                generated,
            },
            // A live blob's file goes once it has landed, or failed; a
            // retained one's stays (a claim never consumes).
            delete_file: matches!(blob, DiskBlob::Live(_)),
        });
        Ok(())
    }

    pub(crate) fn tier_advance(&self) -> Vec<DiskEvent> {
        let mut guard = self.disk.lock().unwrap();
        let Some(tier) = guard.as_mut() else {
            return Vec::new();
        };
        self.drain(tier);
        let mut events = Vec::new();
        let mut i = 0;
        while i < tier.transfers.len() {
            match self.pump(tier, i) {
                Some(outcome) => {
                    let transfer = tier.transfers.remove(i);
                    let blob = transfer.blob;
                    if !matches!(outcome, DiskOutcome::Spilled { .. } | DiskOutcome::Restored { .. }) {
                        tier.draining.push(transfer);
                    }
                    events.push(DiskEvent { blob, outcome });
                }
                None => i += 1,
            }
        }
        // A failure's resources are let go as soon as its ops have drained,
        // which for most is now.
        self.drain(tier);
        events
    }

    pub(crate) fn tier_discard(&self, blob: DiskBlob) {
        let mut guard = self.disk.lock().unwrap();
        let Some(tier) = guard.as_mut() else {
            return;
        };
        if let Some(at) = tier.transfers.iter().position(|t| t.blob == blob) {
            let transfer = tier.transfers.remove(at);
            self.abandon(tier, transfer, true);
        }
        if let Some(file) = tier.files.remove(&blob) {
            tier.store.io.submit(Job::Delete { path: file.path });
        }
    }

    pub(crate) fn tier_abandon_restore(&self, request: RequestId) {
        let mut guard = self.disk.lock().unwrap();
        let Some(tier) = guard.as_mut() else {
            return;
        };
        let at = tier.transfers.iter().position(|t| {
            matches!(t.kind, Kind::Restore { into: DiskTarget::Sequence { request: r, .. }, .. } if r == request)
        });
        if let Some(at) = at {
            let transfer = tier.transfers.remove(at);
            let keep = !matches!(transfer.blob, DiskBlob::Live(_));
            self.abandon(tier, transfer, !keep);
        }
    }

    /// A transfer ended before its time (a cancel): it drains like a failed
    /// one. A restore's copies into the sequence it was building are waited
    /// for first -- PCIe, not the disk -- so the sequence can go now, before
    /// its pages are anyone else's.
    fn abandon(&self, tier: &mut DiskTier<L>, mut transfer: Transfer<L>, delete_file: bool) {
        transfer.delete_file = delete_file;
        if let Kind::Restore { sequence, .. } = &mut transfer.kind {
            for op in transfer.ops.iter() {
                if let Op::Feeding { fence, .. } = op {
                    let _ = self.model.leaf.transfer_wait(self.model.handle(), *fence);
                }
            }
            transfer.ops.retain(|op| !matches!(op, Op::Feeding { .. }));
            if let Some(sequence) = sequence.take() {
                self.release_sequence(sequence);
            }
        }
        tier.draining.push(transfer);
        self.drain(tier);
    }

    /// Let go of what drained transfers held once their ops are done: their
    /// slots, a spill's KV-RAM span (back where it came from), a restore's
    /// sequence, and the file.
    fn drain(&self, tier: &mut DiskTier<L>) {
        let mut i = 0;
        while i < tier.draining.len() {
            let done = {
                let transfer = &mut tier.draining[i];
                let slots = &mut tier.slots;
                transfer.ops.retain(|op| {
                    let finished = match op {
                        Op::Copying { fence, .. } | Op::Feeding { fence, .. } => self
                            .model
                            .leaf
                            .transfer_passed(self.model.handle(), *fence)
                            .unwrap_or(true),
                        Op::Writing { ticket, .. } | Op::Reading { ticket, .. } => ticket.poll().is_some(),
                    };
                    if finished {
                        let slot = match op {
                            Op::Copying { slot, .. } | Op::Reading { slot, .. } | Op::Feeding { slot, .. } => {
                                Some(*slot)
                            }
                            Op::Writing { slot, .. } => *slot,
                        };
                        if let Some(slot) = slot {
                            slots[slot].busy = false;
                        }
                    }
                    !finished
                });
                let header_done = match &transfer.kind {
                    Kind::Spill { header, .. } | Kind::Restore { header, .. } => {
                        header.as_ref().is_none_or(|(_, ticket)| ticket.poll().is_some())
                    }
                };
                transfer.ops.is_empty() && header_done
            };
            if !done {
                i += 1;
                continue;
            }
            let transfer = tier.draining.swap_remove(i);
            let delete = transfer.delete_file;
            match transfer.kind {
                Kind::Spill { writer, source, generated, .. } => {
                    // The handle goes before the delete is queued behind it.
                    drop(writer);
                    if let Some(buf) = source {
                        self.put_back(transfer.blob, buf, generated);
                    }
                }
                Kind::Restore { sequence, reader, .. } => {
                    drop(reader);
                    if let Some(sequence) = sequence {
                        self.release_sequence(sequence);
                    }
                }
            }
            if delete {
                tier.files.remove(&transfer.blob);
                tier.store.io.submit(Job::Delete { path: transfer.path });
            }
        }
    }

    /// Advance transfer `i` by what has landed and at most one new window;
    /// its outcome when it ended.
    fn pump(&self, tier: &mut DiskTier<L>, i: usize) -> Option<DiskOutcome> {
        match tier.transfers[i].kind {
            Kind::Spill { .. } => self.pump_spill(tier, i),
            Kind::Restore { .. } => self.pump_restore(tier, i),
        }
    }

    fn pump_spill(&self, tier: &mut DiskTier<L>, i: usize) -> Option<DiskOutcome> {
        let window_bytes = tier.store.window_bytes;
        // What has landed: copies out to the writer, writes into the CRC
        // table.
        let mut failed = false;
        let mut keep = VecDeque::new();
        let ops = std::mem::take(&mut tier.transfers[i].ops);
        for op in ops {
            match op {
                Op::Copying { index, slot, fence } => {
                    match self.model.leaf.transfer_passed(self.model.handle(), fence) {
                        Ok(false) => keep.push_back(Op::Copying { index, slot, fence }),
                        Ok(true) => {
                            let window = tier.transfers[i].window(index, window_bytes);
                            let bytes = Bytes {
                                len: kv_disk::aligned(window.bytes) as usize,
                                ..tier.slot_bytes(slot)
                            };
                            let Kind::Spill { writer: Some(writer), .. } = &tier.transfers[i].kind else {
                                unreachable!("a spill holds its writer until it ends");
                            };
                            let ticket = tier.store.io.submit(Job::Write {
                                file: Arc::clone(writer),
                                offset: HEADER_BYTES as u64 + window.offset,
                                bytes,
                                crc_len: window.bytes as usize,
                                copy_from: None,
                            });
                            keep.push_back(Op::Writing { index, slot: Some(slot), ticket });
                        }
                        Err(_) => {
                            failed = true;
                            keep.push_back(Op::Copying { index, slot, fence });
                        }
                    }
                }
                Op::Writing { index, slot, ticket } => match ticket.poll() {
                    None => keep.push_back(Op::Writing { index, slot, ticket }),
                    Some(Ok(crc)) => {
                        if let Kind::Spill { crcs, .. } = &mut tier.transfers[i].kind {
                            crcs[index] = crc;
                        }
                        if let Some(slot) = slot {
                            tier.slots[slot].busy = false;
                        }
                        tier.transfers[i].landed += 1;
                    }
                    Some(Err(error)) => {
                        tracing::warn!(name: "ignis.kv_disk.write_failed", %error, "a KV-disk window write failed");
                        if let Some(slot) = slot {
                            tier.slots[slot].busy = false;
                        }
                        failed = true;
                    }
                },
                other => keep.push_back(other),
            }
        }
        tier.transfers[i].ops = keep;
        if failed {
            return Some(DiskOutcome::Failed { op: DiskOp::Write });
        }

        // The header, last: the commit.
        if let Kind::Spill { header: Some((_, ticket)), .. } = &tier.transfers[i].kind {
            return match ticket.poll() {
                None => None,
                Some(Err(error)) => {
                    tracing::warn!(name: "ignis.kv_disk.write_failed", %error, "a KV-disk header write failed");
                    Some(DiskOutcome::Failed { op: DiskOp::Write })
                }
                Some(Ok(_)) => Some(self.commit_spill(tier, i)),
            };
        }
        let transfer = &tier.transfers[i];
        if transfer.landed == transfer.windows {
            return self.write_header(tier, i);
        }

        // One new window.
        if transfer.issued < transfer.windows {
            let index = transfer.issued;
            let window = transfer.window(index, window_bytes);
            let from = match &transfer.kind {
                Kind::Spill { from, .. } => *from,
                Kind::Restore { .. } => unreachable!("a spill"),
            };
            match from {
                DiskSource::Device => {
                    let Some(slot) = tier.free_slot() else {
                        return None;
                    };
                    let DiskBlob::Live(request) = tier.transfers[i].blob else {
                        unreachable!("only a live sequence spills from the device");
                    };
                    let dst = &mut tier.slots[slot].buffer.as_mut().as_mut()[..window.bytes as usize];
                    let issued = {
                        let sequences = self.sequences.lock().unwrap();
                        match sequences.get(&request) {
                            Some(live) => self
                                .model
                                .leaf
                                .snapshot_window(self.model.handle(), &live.handle, window, dst)
                                .and_then(|()| self.model.leaf.transfer_fence(self.model.handle())),
                            None => Err(-1),
                        }
                    };
                    match issued {
                        Ok(fence) => {
                            tier.transfers[i].ops.push_back(Op::Copying { index, slot, fence });
                            tier.transfers[i].issued += 1;
                        }
                        Err(code) => {
                            tier.slots[slot].busy = false;
                            tracing::warn!(name: "ignis.kv_disk.copy_failed", code, "a KV-disk window copy could not be issued");
                            return Some(DiskOutcome::Failed { op: DiskOp::Write });
                        }
                    }
                }
                DiskSource::KvRam => {
                    let (src_ptr, aligned_src) = {
                        let Kind::Spill { source: Some(buf), .. } = &tier.transfers[i].kind else {
                            unreachable!("a KV-RAM spill owns its span");
                        };
                        let ptr = buf.as_ref()[window.offset as usize..].as_ptr();
                        (ptr, (ptr as usize) % kv_disk::ALIGNMENT as usize == 0)
                    };
                    let whole = window.bytes % kv_disk::ALIGNMENT == 0;
                    let padded = kv_disk::aligned(window.bytes) as usize;
                    // Straight from the span when it is on the boundary and
                    // the window whole sectors; else through a slot the
                    // writer itself copies into.
                    let (slot, bytes, copy_from) = if aligned_src && whole {
                        (None, Bytes { ptr: src_ptr as *mut u8, len: padded }, None)
                    } else {
                        let Some(slot) = tier.free_slot() else {
                            return None;
                        };
                        let bytes = Bytes { len: padded, ..tier.slot_bytes(slot) };
                        (Some(slot), bytes, Some(Bytes { ptr: src_ptr as *mut u8, len: window.bytes as usize }))
                    };
                    let Kind::Spill { writer: Some(writer), .. } = &tier.transfers[i].kind else {
                        unreachable!("a spill holds its writer until it ends");
                    };
                    let ticket = tier.store.io.submit(Job::Write {
                        file: Arc::clone(writer),
                        offset: HEADER_BYTES as u64 + window.offset,
                        bytes,
                        crc_len: window.bytes as usize,
                        copy_from,
                    });
                    tier.transfers[i].ops.push_back(Op::Writing { index, slot, ticket });
                    tier.transfers[i].issued += 1;
                }
            }
        }
        None
    }

    /// Every window landed: queue the header page, with every CRC.
    fn write_header(&self, tier: &mut DiskTier<L>, i: usize) -> Option<DiskOutcome> {
        let transfer = &mut tier.transfers[i];
        let Kind::Spill { writer: Some(writer), meta, crcs, header, .. } = &mut transfer.kind else {
            unreachable!("a spill holds its writer until it ends");
        };
        let page = FileHeader {
            identity: tier.store.identity.clone(),
            kind: BlobKind::of(transfer.blob),
            key: meta.key.to_bytes(),
            tokens: meta.tokens,
            blob_bytes: transfer.blob_bytes,
            window_bytes: tier.store.window_bytes,
            crcs: crcs.clone(),
        };
        let encoded = match page.encode() {
            Ok(encoded) => encoded,
            Err(error) => {
                tracing::warn!(name: "ignis.kv_disk.write_failed", %error, "a KV-disk header could not be encoded");
                return Some(DiskOutcome::Failed { op: DiskOp::Write });
            }
        };
        let Ok(mut buffer) = AlignedBuffer::new(HEADER_BYTES) else {
            return Some(DiskOutcome::Failed { op: DiskOp::Write });
        };
        buffer.as_mut_slice().copy_from_slice(&encoded);
        let bytes = Bytes {
            ptr: buffer.as_mut_slice().as_mut_ptr(),
            len: HEADER_BYTES,
        };
        let ticket = tier.store.io.submit(Job::Write {
            file: Arc::clone(writer),
            offset: 0,
            bytes,
            crc_len: 0,
            copy_from: None,
        });
        *header = Some((buffer, ticket));
        None
    }

    /// The header landed: the file is committed, and its source gives its
    /// bytes up -- the device sequence released, the KV-RAM span freed.
    fn commit_spill(&self, tier: &mut DiskTier<L>, i: usize) -> DiskOutcome {
        let identity = tier.store.identity.clone();
        let window_bytes = tier.store.window_bytes;
        let transfer = &mut tier.transfers[i];
        let Kind::Spill { from, writer, source, generated, meta, crcs, .. } = &mut transfer.kind else {
            unreachable!("a spill");
        };
        let from = *from;
        drop(writer.take());
        drop(source.take());
        let mut generated = *generated;
        if let (DiskBlob::Live(request), DiskSource::Device) = (transfer.blob, from) {
            let live = self.sequences.lock().unwrap().remove(&request);
            if let Some(live) = live {
                generated = live.generated;
                self.release_sequence(live.handle);
            }
            // Vision state is not part of the blob, as for a KV-RAM evict
            // (GitHub #194).
            self.release_media_of(request);
        }
        let header = FileHeader {
            identity,
            kind: BlobKind::of(transfer.blob),
            key: meta.key.to_bytes(),
            tokens: meta.tokens,
            blob_bytes: transfer.blob_bytes,
            window_bytes,
            crcs: std::mem::take(crcs),
        };
        let bytes = ignis_core::disk::disk_file_bytes(transfer.blob_bytes);
        tracing::debug!(
            name: "ignis.kv_disk.spilled",
            from = from.as_str(),
            bytes,
            micros = transfer.started.elapsed().as_micros() as u64,
            "a KV-disk file committed"
        );
        let (blob, path) = (transfer.blob, transfer.path.clone());
        tier.files.insert(blob, DiskFile { path, header, generated });
        DiskOutcome::Spilled { bytes }
    }

    fn pump_restore(&self, tier: &mut DiskTier<L>, i: usize) -> Option<DiskOutcome> {
        let window_bytes = tier.store.window_bytes;
        // The header page first: refused before any byte moves.
        {
            let Kind::Restore { header, validated, expected, .. } = &mut tier.transfers[i].kind else {
                unreachable!("a restore");
            };
            if !*validated {
                let (page, ticket) = header.as_ref().expect("a restore reads its header first");
                match ticket.poll() {
                    None => return None,
                    Some(Err(error)) => return Some(self.refuse(kv_disk::Refusal::Io(error))),
                    Some(Ok(_)) => {
                        let checked = FileHeader::decode(page.as_slice())
                            .and_then(|read| read.check(&tier.store.identity, expected));
                        *header = None;
                        if let Err(refusal) = checked {
                            return Some(self.refuse(refusal));
                        }
                        *validated = true;
                    }
                }
            }
        }

        // What has landed: reads into checked windows fed to the device,
        // copies into freed slots.
        let mut keep = VecDeque::new();
        let ops = std::mem::take(&mut tier.transfers[i].ops);
        let mut outcome = None;
        for op in ops {
            if outcome.is_some() {
                keep.push_back(op);
                continue;
            }
            match op {
                Op::Reading { index, slot, ticket } => match ticket.poll() {
                    None => keep.push_back(Op::Reading { index, slot, ticket }),
                    Some(Err(error)) => {
                        tier.slots[slot].busy = false;
                        outcome = Some(self.refuse(kv_disk::Refusal::Io(error)));
                    }
                    Some(Ok(crc)) => {
                        let window = tier.transfers[i].window(index, window_bytes);
                        let (expected_crc, sequence) = match &mut tier.transfers[i].kind {
                            Kind::Restore { expected, sequence, .. } => (expected.crcs[index], sequence),
                            Kind::Spill { .. } => unreachable!("a restore"),
                        };
                        if crc != expected_crc {
                            tier.slots[slot].busy = false;
                            outcome = Some(self.refuse(kv_disk::Refusal::CorruptWindow(index)));
                            continue;
                        }
                        let src = &tier.slots[slot].buffer.as_mut().as_mut()[..window.bytes as usize];
                        let sequence = sequence.as_mut().expect("a restore holds its sequence until it ends");
                        let fed = self
                            .model
                            .leaf
                            .restore_window(self.model.handle(), sequence, window, src)
                            .and_then(|()| self.model.leaf.transfer_fence(self.model.handle()));
                        match fed {
                            Ok(fence) => keep.push_back(Op::Feeding { slot, fence }),
                            Err(code) => {
                                tier.slots[slot].busy = false;
                                tracing::warn!(name: "ignis.kv_disk.restore_refused", code, "the leaf refused a KV-disk window");
                                outcome = Some(DiskOutcome::Failed { op: DiskOp::Read });
                            }
                        }
                    }
                },
                Op::Feeding { slot, fence } => match self.model.leaf.transfer_passed(self.model.handle(), fence) {
                    Ok(false) => keep.push_back(Op::Feeding { slot, fence }),
                    Ok(true) => {
                        tier.slots[slot].busy = false;
                        tier.transfers[i].landed += 1;
                    }
                    Err(_) => {
                        keep.push_back(Op::Feeding { slot, fence });
                        outcome = Some(DiskOutcome::Failed { op: DiskOp::Read });
                    }
                },
                other => keep.push_back(other),
            }
        }
        tier.transfers[i].ops = keep;
        if outcome.is_some() {
            return outcome;
        }
        let transfer = &tier.transfers[i];
        if transfer.landed == transfer.windows {
            return Some(self.land_restore(tier, i));
        }
        // One new window: read into a free slot.
        if transfer.issued < transfer.windows {
            let index = transfer.issued;
            let window = transfer.window(index, window_bytes);
            let Some(slot) = tier.free_slot() else {
                return None;
            };
            let bytes = Bytes {
                len: kv_disk::aligned(window.bytes) as usize,
                ..tier.slot_bytes(slot)
            };
            let Kind::Restore { reader, .. } = &tier.transfers[i].kind else {
                unreachable!("a restore");
            };
            let ticket = tier.store.io.submit(Job::Read {
                file: Arc::clone(reader),
                offset: HEADER_BYTES as u64 + window.offset,
                bytes,
                crc_len: window.bytes as usize,
            });
            tier.transfers[i].ops.push_back(Op::Reading { index, slot, ticket });
            tier.transfers[i].issued += 1;
        }
        None
    }

    /// A file the restore will not take: never restored (spec AC 15).
    fn refuse(&self, refusal: kv_disk::Refusal) -> DiskOutcome {
        tracing::warn!(name: "ignis.kv_disk.file_refused", reason = %refusal, "a KV-disk file failed its check");
        DiskOutcome::Failed { op: DiskOp::Read }
    }

    /// Every window landed: the sequence is the request's.
    fn land_restore(&self, tier: &mut DiskTier<L>, i: usize) -> DiskOutcome {
        let transfer = &mut tier.transfers[i];
        let micros = transfer.started.elapsed().as_micros() as u64;
        let Kind::Restore { into, sequence, generated, .. } = &mut transfer.kind else {
            unreachable!("a restore");
        };
        let DiskTarget::Sequence { request, .. } = *into else {
            unreachable!("only sequences restore from the disk in this version");
        };
        if let Some(handle) = sequence.take() {
            self.sequences.lock().unwrap().insert(
                request,
                LiveSequence {
                    handle,
                    generated: *generated,
                },
            );
        }
        if transfer.delete_file {
            tier.files.remove(&transfer.blob);
            tier.store.io.submit(Job::Delete { path: transfer.path.clone() });
        }
        DiskOutcome::Restored { micros }
    }

    /// Everything the tier holds, let go of at shutdown: waits are fine here.
    pub(crate) fn tier_shutdown(&self) {
        let Some(mut tier) = self.disk.lock().unwrap().take() else {
            return;
        };
        for transfer in std::mem::take(&mut tier.transfers) {
            self.abandon(&mut tier, transfer, true);
        }
        while !tier.draining.is_empty() {
            self.drain(&mut tier);
            std::thread::yield_now();
        }
    }
}
