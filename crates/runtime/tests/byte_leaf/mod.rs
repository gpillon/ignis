//! A leaf whose sequences are byte vectors, windowed like a real one, for the
//! CPU tests of `RuntimeCompute`'s moves: KV-disk (`kv_disk_transfers.rs`)
//! and KV-RAM a window at a time (`kv_ram_moves.rs`, GitHub #309).
//!
//! Its transfer stream is a counter of fences, which pass at once unless the
//! test holds them -- a copy still on the link -- and a window can be made
//! to fail, as a copy that errored.

#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use ignis_core::{Compute, DecodeJob, DecodeParams, PrefillJob, RequestId};
use ignis_runtime::{DecodeLane, LaneRun, RuntimeCompute, RuntimeStats, StepLeaf, TransferWindow};

/// The stub's blob: long enough for several 8 KiB windows, and not a whole
/// number of them.
pub const BLOB_BYTES: usize = 5 * 8192 + 1000;
pub const WINDOW: u64 = 8192;

#[derive(Default)]
pub struct Calls {
    pub released: u32,
    pub snapshot_windows: Vec<TransferWindow>,
    pub restore_windows: Vec<TransferWindow>,
    /// Fences waited for, blocking.
    pub waits: u32,
}

#[derive(Default)]
struct Fences {
    /// The last fence taken.
    taken: u64,
    /// Every fence up to this one has passed.
    passed: u64,
    holding: bool,
}

/// A leaf whose sequence is its bytes: prefill fills them from the prompt,
/// decode flips one and emits a token that hashes them all -- so two
/// sequences that decode the same tokens hold the same bytes.
pub struct ByteLeaf {
    pub calls: Mutex<Calls>,
    fences: Mutex<Fences>,
    failing: AtomicBool,
    windowed: bool,
}

impl ByteLeaf {
    pub fn new() -> Self {
        Self {
            calls: Mutex::default(),
            fences: Mutex::default(),
            failing: AtomicBool::new(false),
            windowed: true,
        }
    }

    /// A leaf with no transfer stream: it moves a blob in one call.
    pub fn without_transfer_stream() -> Self {
        Self { windowed: false, ..Self::new() }
    }

    /// Copies issued from now on stay on the link until
    /// [`ByteLeaf::release_fences`].
    pub fn hold_fences(&self) {
        self.fences.lock().unwrap().holding = true;
    }

    /// Every copy issued so far lands, and later ones land at once.
    pub fn release_fences(&self) {
        let mut fences = self.fences.lock().unwrap();
        fences.holding = false;
        fences.passed = fences.taken;
    }

    /// Every window from now on fails to issue (or issues again, with
    /// `false`).
    pub fn fail_windows(&self, failing: bool) {
        self.failing.store(failing, Ordering::SeqCst);
    }
}

#[derive(Debug, Default)]
pub struct ByteSeq {
    bytes: Vec<u8>,
}

impl StepLeaf for ByteLeaf {
    type Model = ();
    type Sequence = ByteSeq;
    type Prefix = ();
    type SnapshotBuf = Vec<u8>;
    type Media = ();
    type Checkpoint = ();

    fn load_model(&self) -> Result<(), i32> {
        Ok(())
    }
    fn release_model(&self, _model: ()) {}
    fn stats(&self, _model: &()) -> Result<RuntimeStats, i32> {
        Ok(RuntimeStats {
            vram_bytes: 0,
            kv_page_tokens: 64,
            kv_page_bytes: 0,
            kv_page_count: 0,
            last_step_micros: 0,
            kernel_count: 0,
            graph_launches: 0,
            free_vram_bytes: 0,
            reserved: Default::default(),
        })
    }
    fn vocab(&self, _model: &()) -> u32 {
        16
    }
    fn allocate_sequence(&self, _model: &(), _context_tokens: u32) -> Result<ByteSeq, i32> {
        Ok(ByteSeq::default())
    }
    fn release_sequence(&self, _model: &(), _sequence: ByteSeq) {
        self.calls.lock().unwrap().released += 1;
    }
    fn allocate_sequence_shared(&self, _model: &(), _context_tokens: u32, _prefix: &()) -> Result<ByteSeq, i32> {
        Err(-1)
    }
    fn publish_prefix(&self, _model: &(), _sequence: &mut ByteSeq, _tokens: u32, _slot: u32) -> Result<(), i32> {
        Err(-1)
    }
    fn release_prefix(&self, _model: &(), _prefix: ()) {}
    fn prefill(
        &self,
        _model: &(),
        sequence: &mut ByteSeq,
        tokens: &[u32],
        _start_position: u32,
        _params: DecodeParams,
        _permitted: &[u32],
        _out_logits: Option<&mut [f32]>,
        _attention: Option<&mut ignis_runtime::AttentionRead>,
    ) -> Result<f32, i32> {
        let seed = tokens.iter().fold(17u32, |h, &t| h.wrapping_mul(31).wrapping_add(t));
        sequence.bytes = (0..BLOB_BYTES).map(|i| (seed as usize + i * 7 + i / 977) as u8).collect();
        Ok(0.0)
    }
    fn decode(&self, _model: &(), sequences: &mut [&mut ByteSeq], _lanes: &[DecodeLane<'_>]) -> Result<Vec<LaneRun>, i32> {
        Ok(sequences
            .iter_mut()
            .map(|s| {
                // Every bit of the state is in the bytes, as a real blob
                // holds all of a sequence: which byte flips is a function of
                // them alone.
                let at = crc32fast::hash(&s.bytes) as usize % s.bytes.len();
                s.bytes[at] ^= 0x5A;
                LaneRun::token(crc32fast::hash(&s.bytes) % 1000)
            })
            .collect())
    }
    fn alloc_snapshot_buf(&self, bytes: u64) -> Result<Vec<u8>, i32> {
        Ok(vec![0; bytes as usize])
    }
    fn snapshot_bytes(&self, _model: &(), sequence: &ByteSeq) -> Result<u64, i32> {
        Ok(sequence.bytes.len() as u64)
    }
    fn snapshot_into(&self, _model: &(), sequence: &ByteSeq, dst: &mut [u8]) -> Result<(), i32> {
        dst[..sequence.bytes.len()].copy_from_slice(&sequence.bytes);
        Ok(())
    }
    fn restore_sequence(&self, _model: &(), sequence: &mut ByteSeq, src: &[u8]) -> Result<(), i32> {
        sequence.bytes = src.to_vec();
        Ok(())
    }
    fn windowed_transfer(&self) -> bool {
        self.windowed
    }
    fn snapshot_window(&self, _model: &(), sequence: &ByteSeq, window: TransferWindow, dst: &mut [u8]) -> Result<(), i32> {
        if window.blob_bytes != sequence.bytes.len() as u64 || self.failing.load(Ordering::SeqCst) {
            return Err(-1);
        }
        let from = window.offset as usize;
        dst.copy_from_slice(&sequence.bytes[from..from + window.bytes as usize]);
        self.calls.lock().unwrap().snapshot_windows.push(window);
        Ok(())
    }
    fn restore_window(&self, _model: &(), sequence: &mut ByteSeq, window: TransferWindow, src: &[u8]) -> Result<(), i32> {
        if window.offset != sequence.bytes.len() as u64 || self.failing.load(Ordering::SeqCst) {
            return Err(-1);
        }
        sequence.bytes.extend_from_slice(src);
        self.calls.lock().unwrap().restore_windows.push(window);
        Ok(())
    }
    fn transfer_fence(&self, _model: &()) -> Result<u64, i32> {
        let mut fences = self.fences.lock().unwrap();
        fences.taken += 1;
        if !fences.holding {
            fences.passed = fences.taken;
        }
        Ok(fences.taken)
    }
    fn transfer_passed(&self, _model: &(), fence: u64) -> Result<bool, i32> {
        Ok(fence <= self.fences.lock().unwrap().passed)
    }
    fn transfer_wait(&self, _model: &(), fence: u64) -> Result<(), i32> {
        self.calls.lock().unwrap().waits += 1;
        let mut fences = self.fences.lock().unwrap();
        fences.passed = fences.passed.max(fence);
        Ok(())
    }
}

/// Prefill `request` on `compute`: its sequence holds [`BLOB_BYTES`].
pub fn prefill(compute: &RuntimeCompute<ByteLeaf>, request: RequestId) {
    let job = PrefillJob {
        request,
        tokens: vec![1, 2, 3, 4, request as u32],
        context_tokens: 64,
        start_position: 0,
        params: DecodeParams::default(),
        shared_prefix: None,
        publish_prefix: None,
        checkpoint: None,
        capture_checkpoint: None,
        multimodal: None,
        readout: None,
        permitted: None,
        attention: None,
    };
    compute.prefill_step(&[job]).unwrap();
}

/// `rounds` decode rounds of `request` alone, and the tokens they emitted.
pub fn decode(compute: &RuntimeCompute<ByteLeaf>, request: RequestId, rounds: u32) -> Vec<u32> {
    let job = DecodeJob {
        request,
        lane: 0,
        params: DecodeParams {
            max_tokens: Some(10_000),
            ..DecodeParams::default()
        },
        remaining_tokens: 1,
        permitted: None,
    };
    (0..rounds)
        .flat_map(|_| compute.decode_step(std::slice::from_ref(&job)).unwrap().remove(0).tokens)
        .collect()
}
