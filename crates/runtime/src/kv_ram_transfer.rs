//! Live moves through KV-RAM a window at a time -- [`RuntimeCompute`]'s half
//! of the `Compute` seam's KV-RAM moves (spec vram-budget/03 AC 37, GitHub
//! #309).
//!
//! A synchronous move held the model thread -- and so every decoding lane --
//! for the whole copy of a blob: ~0.32 s for a 236K-token Flash-Next
//! sequence. Here a move is a blob copied a window at a time between a live
//! sequence and its span of the pinned KV-RAM arena, on the leaf's transfer
//! stream, pumped by [`Compute::kv_ram_advance`] between steps: KV-disk's
//! transfers (`disk_transfer`) without the file, and without staging, since
//! the span is pinned and the copies land in it and leave from it directly.
//!
//! - **A move out** takes its span at the start and copies the sequence into
//!   it. The sequence stays the request's, stepped by nobody, until the last
//!   window's fence has passed; only then is it released and the span filed
//!   as the request's snapshot.
//! - **A move in** draws its sequence at the start (the scheduler charged it)
//!   and feeds it from the snapshot's span, windows in order. The sequence is
//!   the request's once the last window's fence has passed; the span then
//!   goes back to the arena.
//! - **The pace** ([`TransferPace`]): one window in flight each way, a new one
//!   issued only once the last has landed and at most once a call -- so a
//!   decode round shares the link with at most one window of a move.
//! - **Ending early** -- a copy that failed, or a cancel -- waits for the
//!   window in flight before anything it touches is let go: PCIe, one
//!   window, never the whole blob.

use std::time::Instant;

use ignis_core::scheduler::{KvRamEvent, KvRamOutcome};
use ignis_core::{ComputeError, RequestId};

use crate::{EvictedSequence, LiveSequence, RuntimeCompute, RuntimeError, StepLeaf, TransferWindow};

/// What a move holds, by direction.
enum Kind<L: StepLeaf> {
    /// The span the sequence's blob is landing in.
    Out { span: L::SnapshotBuf },
    /// The snapshot's span, and the sequence it is feeding.
    In {
        span: L::SnapshotBuf,
        sequence: L::Sequence,
        generated: u32,
    },
}

/// One move under way.
struct Move<L: StepLeaf> {
    request: RequestId,
    blob_bytes: u64,
    /// Bytes issued so far, from the blob's start.
    issued: u64,
    /// The fence after the window in flight, if one is.
    in_flight: Option<u64>,
    /// A window failed as the move started: it ends, failed, at the next
    /// advance, as one failing later would.
    failed: bool,
    started: Instant,
    kind: Kind<L>,
}

/// The windowed KV-RAM moves under way in a [`RuntimeCompute`]: at most one
/// each way, as the scheduler starts them.
pub(crate) struct KvRamMoves<L: StepLeaf> {
    under_way: Vec<Move<L>>,
}

impl<L: StepLeaf> KvRamMoves<L> {
    pub(crate) fn new() -> Self {
        Self { under_way: Vec::new() }
    }

    fn has(&self, request: RequestId) -> bool {
        self.under_way.iter().any(|m| m.request == request)
    }
}

/// How a pump of one move went.
enum Pumped {
    Going,
    Landed,
    Failed,
}

fn leaf_err(code: i32) -> ComputeError {
    RuntimeError::Leaf(code).into()
}

impl<L: StepLeaf> RuntimeCompute<L> {
    /// Start moving live `request` into KV-RAM: its span taken now, its
    /// first window issued. Returns the blob's bytes.
    pub(crate) fn ram_move_out(&self, request: RequestId) -> Result<u64, ComputeError> {
        let mut moves = self.kv_ram.lock().unwrap();
        if moves.has(request) {
            return Err(ComputeError::Kernel(-1));
        }
        let blob_bytes = {
            let sequences = self.sequences.lock().unwrap();
            let live = sequences.get(&request).ok_or(ComputeError::Kernel(-1))?;
            self.model
                .leaf
                .snapshot_bytes(self.model.handle(), &live.handle)
                .map_err(leaf_err)?
        };
        let span = self.model.leaf.alloc_snapshot_buf(blob_bytes).map_err(leaf_err)?;
        self.begin_ram_move(&mut moves, request, blob_bytes, Kind::Out { span });
        Ok(blob_bytes)
    }

    /// Start bringing `request`'s snapshot back into a sequence reserving
    /// `context_tokens`, drawn now; its first window issued. The snapshot
    /// stays where it was when this fails.
    pub(crate) fn ram_move_in(&self, request: RequestId, context_tokens: u32) -> Result<(), ComputeError> {
        let mut moves = self.kv_ram.lock().unwrap();
        if moves.has(request) {
            return Err(ComputeError::Kernel(-1));
        }
        let evicted = self.evicted.lock().unwrap().remove(&request).ok_or(ComputeError::Kernel(-1))?;
        let sequence = match self.model.leaf.allocate_sequence(self.model.handle(), context_tokens) {
            Ok(sequence) => sequence,
            Err(code) => {
                self.evicted.lock().unwrap().insert(request, evicted);
                return Err(leaf_err(code));
            }
        };
        let blob_bytes = evicted.buf.as_ref().len() as u64;
        let kind = Kind::In {
            span: evicted.buf,
            sequence,
            generated: evicted.generated,
        };
        self.begin_ram_move(&mut moves, request, blob_bytes, kind);
        Ok(())
    }

    /// File a move and issue its first window now, beside this advance's
    /// round. A first window that fails ends the move at the next advance,
    /// as any other failed window does.
    fn begin_ram_move(&self, moves: &mut KvRamMoves<L>, request: RequestId, blob_bytes: u64, kind: Kind<L>) {
        let mut m = Move {
            request,
            blob_bytes,
            issued: 0,
            in_flight: None,
            failed: false,
            started: Instant::now(),
            kind,
        };
        m.failed = matches!(self.pump_ram(&mut m), Pumped::Failed);
        moves.under_way.push(m);
    }

    /// Advance every move by what has landed and at most one new window, and
    /// report the ones that ended.
    pub(crate) fn ram_advance(&self) -> Vec<KvRamEvent> {
        let mut moves = self.kv_ram.lock().unwrap();
        let mut events = Vec::new();
        let mut i = 0;
        while i < moves.under_way.len() {
            let landed = match self.pump_ram(&mut moves.under_way[i]) {
                Pumped::Going => {
                    i += 1;
                    continue;
                }
                Pumped::Landed => true,
                Pumped::Failed => false,
            };
            let ended = moves.under_way.remove(i);
            let request = ended.request;
            let micros = ended.started.elapsed().as_micros() as u64;
            let outcome = if self.end_ram_move(ended, landed) {
                KvRamOutcome::Landed { micros }
            } else {
                KvRamOutcome::Failed
            };
            events.push(KvRamEvent { request, outcome });
        }
        events
    }

    /// Abandon `request`'s move once its window in flight has landed.
    pub(crate) fn ram_abandon(&self, request: RequestId) {
        let mut moves = self.kv_ram.lock().unwrap();
        if let Some(at) = moves.under_way.iter().position(|m| m.request == request) {
            let abandoned = moves.under_way.remove(at);
            self.end_ram_move(abandoned, false);
        }
    }

    /// Every move let go of at shutdown.
    pub(crate) fn ram_shutdown(&self) {
        // From `Drop`: a lock a panic poisoned is still the moves'.
        let mut moves = self.kv_ram.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for abandoned in std::mem::take(&mut moves.under_way) {
            self.end_ram_move(abandoned, false);
        }
    }

    /// Settle what has landed of `m`'s window in flight, and issue its next
    /// window when there is none.
    fn pump_ram(&self, m: &mut Move<L>) -> Pumped {
        if m.failed {
            return Pumped::Failed;
        }
        if let Some(fence) = m.in_flight {
            match self.model.leaf.transfer_passed(self.model.handle(), fence) {
                Ok(false) => return Pumped::Going,
                Ok(true) => m.in_flight = None,
                Err(code) => {
                    tracing::warn!(name: "ignis.kv_ram.copy_failed", code, request_id = m.request, "a KV-RAM move's window failed");
                    return Pumped::Failed;
                }
            }
        }
        if m.issued == m.blob_bytes {
            return Pumped::Landed;
        }
        let window_bytes = match m.kind {
            Kind::Out { .. } => self.pace.move_out_bytes,
            Kind::In { .. } => self.pace.move_in_bytes,
        };
        let window = TransferWindow {
            offset: m.issued,
            bytes: window_bytes.min(m.blob_bytes - m.issued),
            blob_bytes: m.blob_bytes,
        };
        let range = window.offset as usize..(window.offset + window.bytes) as usize;
        let issued = match &mut m.kind {
            Kind::Out { span } => {
                let sequences = self.sequences.lock().unwrap();
                match sequences.get(&m.request) {
                    Some(live) => {
                        self.model
                            .leaf
                            .snapshot_window(self.model.handle(), &live.handle, window, &mut span.as_mut()[range])
                    }
                    None => Err(-1),
                }
            }
            Kind::In { span, sequence, .. } => {
                self.model
                    .leaf
                    .restore_window(self.model.handle(), sequence, window, &span.as_ref()[range])
            }
        };
        let (fence, issued) = self.fence_window(issued);
        m.in_flight = fence;
        match issued {
            Ok(()) => {
                m.issued += window.bytes;
                Pumped::Going
            }
            Err(code) => {
                tracing::warn!(name: "ignis.kv_ram.copy_failed", code, request_id = m.request, "a KV-RAM move's window could not be issued");
                Pumped::Failed
            }
        }
    }

    /// End `m`, after its window in flight: `landed` files a move out's span
    /// as the request's snapshot and hands a move in's sequence to the
    /// request; otherwise a move out's span goes back to the arena and a move
    /// in's sequence is released, its snapshot left where it was. Returns
    /// whether it landed.
    fn end_ram_move(&self, m: Move<L>, landed: bool) -> bool {
        if let Some(fence) = m.in_flight {
            // PCIe, one window at most: the copies still read or write what
            // is about to be let go.
            let _ = self.model.leaf.transfer_wait(self.model.handle(), fence);
        }
        match m.kind {
            Kind::Out { span } => {
                if !landed {
                    return false;
                }
                let live = self.sequences.lock().unwrap().remove(&m.request);
                let Some(live) = live else {
                    return false;
                };
                self.release_sequence(live.handle);
                // Vision state is not part of the blob (GitHub #194).
                self.release_media_of(m.request);
                self.evicted.lock().unwrap().insert(
                    m.request,
                    EvictedSequence {
                        buf: span,
                        generated: live.generated,
                    },
                );
                true
            }
            Kind::In { span, sequence, generated } => {
                if landed {
                    self.sequences
                        .lock()
                        .unwrap()
                        .insert(m.request, LiveSequence { handle: sequence, generated });
                } else {
                    self.release_sequence(sequence);
                    self.evicted
                        .lock()
                        .unwrap()
                        .insert(m.request, EvictedSequence { buf: span, generated });
                }
                landed
            }
        }
    }
}
