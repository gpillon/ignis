//! Spec vram-budget/03 (ADR 0045) -- KV-disk through `RuntimeCompute` on a
//! CPU: a stub leaf whose sequences are byte vectors, windowed like a real
//! one, and the real store (a temp directory, real files, real IO threads).
//!
//! What this holds the adapter to: a blob spilled a window at a time and
//! read back is the same blob, its source given up only when the header
//! commits the file; a file torn, corrupt or foreign is never restored; a
//! spill's source keeps its bytes when the spill fails or is abandoned; and
//! the n-gram table's prefill gathers go before the tier's IO.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ignis_core::identity::MatchKey;
use ignis_core::ngram_table::GatherGate;
use ignis_core::scheduler::{DiskBlob, DiskBlobMeta, DiskOp, DiskOutcome, DiskSource, DiskTarget};
use ignis_core::{Compute, DecodeJob, DecodeParams, PrefillJob, RequestId};
use ignis_runtime::kv_disk::{self, DiskIdentity, DiskStore, FileHeader, HEADER_BYTES};
use ignis_runtime::{DecodeLane, LaneRun, Model, RuntimeCompute, RuntimeStats, StepLeaf, TransferWindow};

/// The stub's blob: long enough for several 8 KiB windows, and not a whole
/// number of them.
const BLOB_BYTES: usize = 5 * 8192 + 1000;
const WINDOW: u64 = 8192;

#[derive(Default)]
struct Calls {
    released: u32,
    snapshot_windows: Vec<TransferWindow>,
    restore_windows: Vec<TransferWindow>,
}

/// A leaf whose sequence is its bytes: prefill fills them from the prompt,
/// decode flips one and emits a token that hashes them all -- so two
/// sequences that decode the same tokens hold the same bytes.
struct ByteLeaf {
    calls: Mutex<Calls>,
}

#[derive(Debug, Default)]
struct ByteSeq {
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
    fn snapshot_window(&self, _model: &(), sequence: &ByteSeq, window: TransferWindow, dst: &mut [u8]) -> Result<(), i32> {
        if window.blob_bytes != sequence.bytes.len() as u64 {
            return Err(-1);
        }
        let from = window.offset as usize;
        dst.copy_from_slice(&sequence.bytes[from..from + window.bytes as usize]);
        self.calls.lock().unwrap().snapshot_windows.push(window);
        Ok(())
    }
    fn restore_window(&self, _model: &(), sequence: &mut ByteSeq, window: TransferWindow, src: &[u8]) -> Result<(), i32> {
        if window.offset != sequence.bytes.len() as u64 {
            return Err(-1);
        }
        sequence.bytes.extend_from_slice(src);
        self.calls.lock().unwrap().restore_windows.push(window);
        Ok(())
    }
}

fn identity() -> DiskIdentity {
    DiskIdentity {
        model_id: "stub".to_string(),
        artifact: [3; 32],
        kv_format: 1,
        layout_version: 7,
        drafter: String::new(),
        sidecar_sha256: None,
        rope_scaling: [0; 4],
    }
}

fn temp_location(tag: &str) -> std::path::PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "ignis-kv-disk-transfers-{tag}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&path);
    path
}

/// A temp location, removed when it drops -- after the adapter, which is
/// declared before it in [`Rig`].
struct TempDir(std::path::PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Rig {
    compute: RuntimeCompute<ByteLeaf>,
    leaf: Arc<ByteLeaf>,
    gate: GatherGate,
    location: std::path::PathBuf,
    _cleanup: TempDir,
}

fn new_rig(tag: &str) -> Rig {
    let location = temp_location(tag);
    let leaf = Arc::new(ByteLeaf { calls: Mutex::default() });
    let model = Arc::new(Model::load(Arc::clone(&leaf)).unwrap());
    let gate = GatherGate::default();
    let store = DiskStore::open_with(
        &location,
        1 << 30,
        identity(),
        gate.clone(),
        Box::new(|| Ok(100 << 30)),
        WINDOW,
        10 << 30,
    )
    .unwrap()
    .unwrap();
    let staging: Vec<ignis_runtime::StagingWindow> = (0..kv_disk::STAGING_WINDOWS)
        .map(|_| Box::new(ignis_artifact::AlignedBuffer::new(WINDOW as usize).unwrap()) as ignis_runtime::StagingWindow)
        .collect();
    let compute = RuntimeCompute::new(model, 0).with_kv_disk(store, staging);
    let cleanup = TempDir(location.clone());
    Rig { compute, leaf, gate, location, _cleanup: cleanup }
}

fn prefill(compute: &RuntimeCompute<ByteLeaf>, request: RequestId) {
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

fn decode(compute: &RuntimeCompute<ByteLeaf>, request: RequestId, rounds: u32) -> Vec<u32> {
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

fn live_meta() -> DiskBlobMeta {
    DiskBlobMeta {
        key: MatchKey::empty(),
        tokens: 5,
    }
}

/// Advance until `blob`'s transfer ends; its outcome, and how many advances.
fn run(compute: &RuntimeCompute<ByteLeaf>, blob: DiskBlob) -> (DiskOutcome, usize) {
    for step in 1..10_000 {
        if let Some(event) = compute.disk_advance().into_iter().find(|e| e.blob == blob) {
            return (event.outcome, step);
        }
        std::thread::sleep(Duration::from_micros(50));
    }
    panic!("the transfer of {blob:?} never ended");
}

fn files_in(rig: &Rig) -> Vec<String> {
    let root = rig.location.join(kv_disk::DIR_NAME);
    let mut names = Vec::new();
    for dir in std::fs::read_dir(root).unwrap().flatten() {
        for file in std::fs::read_dir(dir.path()).unwrap().flatten() {
            names.push(file.file_name().to_string_lossy().into_owned());
        }
    }
    names.retain(|n| n != "lock");
    names.sort();
    names
}

fn file_of(rig: &Rig, name: &str) -> std::path::PathBuf {
    let root = rig.location.join(kv_disk::DIR_NAME);
    let dir = std::fs::read_dir(root).unwrap().flatten().next().unwrap().path();
    dir.join(name)
}

// ── the round trip ──────────────────────────────────────────────────────────

#[test]
fn a_live_sequence_spilled_and_restored_a_window_at_a_time_continues_as_if_it_never_moved() {
    let rig = new_rig("roundtrip");
    let (a, b) = (1, 2);
    prefill(&rig.compute, a);
    prefill(&rig.compute, b);
    decode(&rig.compute, a, 3);
    decode(&rig.compute, b, 3);

    let blob = DiskBlob::Live(a);
    assert_eq!(rig.compute.disk_spill(blob, DiskSource::Device, live_meta()).unwrap(), BLOB_BYTES as u64);
    let released_before = rig.leaf.calls.lock().unwrap().released;
    let (outcome, steps) = run(&rig.compute, blob);
    let file_bytes = ignis_core::disk::disk_file_bytes(BLOB_BYTES as u64);
    assert_eq!(outcome, DiskOutcome::Spilled { bytes: file_bytes });
    assert!(steps >= 6, "six windows, at most one new one an advance: {steps}");
    assert_eq!(
        rig.leaf.calls.lock().unwrap().released,
        released_before + 1,
        "the device sequence is released with the commit, not before"
    );
    assert_eq!(files_in(&rig), vec!["live-1.kv"]);
    let on_disk = std::fs::metadata(file_of(&rig, "live-1.kv")).unwrap().len();
    assert_eq!(on_disk, file_bytes, "a header page, then the blob padded to a sector");
    let windows = rig.leaf.calls.lock().unwrap().snapshot_windows.clone();
    assert_eq!(windows.len(), 6);
    assert_eq!(windows[5], TransferWindow { offset: 5 * WINDOW, bytes: 1000, blob_bytes: BLOB_BYTES as u64 });

    rig.compute
        .disk_restore(blob, DiskTarget::Sequence { request: a, context_tokens: 64 })
        .unwrap();
    let (outcome, _) = run(&rig.compute, blob);
    assert!(matches!(outcome, DiskOutcome::Restored { .. }), "{outcome:?}");
    assert_eq!(rig.leaf.calls.lock().unwrap().restore_windows.len(), 6);
    assert_eq!(decode(&rig.compute, a, 5).len(), 5, "A decodes again on its restored sequence");
    assert!(files_in(&rig).is_empty(), "a live file goes once it has landed");
}

#[test]
fn the_restored_bytes_are_the_spilled_bytes() {
    // The stub's token hashes the whole sequence, so a restored sequence
    // that decodes what an unmoved copy of it decodes holds its bytes.
    let rig = new_rig("bytes");
    let twin = new_rig("bytes-twin");
    for r in [&rig, &twin] {
        prefill(&r.compute, 1);
        decode(&r.compute, 1, 4);
    }
    let blob = DiskBlob::Live(1);
    rig.compute.disk_spill(blob, DiskSource::Device, live_meta()).unwrap();
    assert!(matches!(run(&rig.compute, blob).0, DiskOutcome::Spilled { .. }));
    rig.compute
        .disk_restore(blob, DiskTarget::Sequence { request: 1, context_tokens: 64 })
        .unwrap();
    assert!(matches!(run(&rig.compute, blob).0, DiskOutcome::Restored { .. }));
    assert_eq!(decode(&rig.compute, 1, 6), decode(&twin.compute, 1, 6));
}

#[test]
fn a_kv_ram_blob_spills_from_its_span_and_comes_back_onto_the_device() {
    let rig = new_rig("kvram");
    let twin = new_rig("kvram-twin");
    for r in [&rig, &twin] {
        prefill(&r.compute, 4);
        decode(&r.compute, 4, 2);
    }
    rig.compute.evict(4).unwrap();
    let blob = DiskBlob::Live(4);
    rig.compute.disk_spill(blob, DiskSource::KvRam, live_meta()).unwrap();
    assert!(matches!(run(&rig.compute, blob).0, DiskOutcome::Spilled { .. }));
    assert!(rig.compute.restore(4, 64).is_err(), "KV-RAM gave its blob up with the commit");
    rig.compute
        .disk_restore(blob, DiskTarget::Sequence { request: 4, context_tokens: 64 })
        .unwrap();
    assert!(matches!(run(&rig.compute, blob).0, DiskOutcome::Restored { .. }));
    assert_eq!(decode(&rig.compute, 4, 5), decode(&twin.compute, 4, 5));
}

// ── AC 15: torn, corrupt and foreign files are never restored ───────────────

fn spilled(tag: &str) -> Rig {
    let rig = new_rig(tag);
    prefill(&rig.compute, 1);
    let blob = DiskBlob::Live(1);
    rig.compute.disk_spill(blob, DiskSource::Device, live_meta()).unwrap();
    assert!(matches!(run(&rig.compute, blob).0, DiskOutcome::Spilled { .. }));
    rig
}

fn refused(rig: &Rig) {
    let blob = DiskBlob::Live(1);
    rig.compute
        .disk_restore(blob, DiskTarget::Sequence { request: 1, context_tokens: 64 })
        .unwrap();
    let (outcome, _) = run(&rig.compute, blob);
    assert_eq!(outcome, DiskOutcome::Failed { op: DiskOp::Read });
    assert!(
        rig.leaf.calls.lock().unwrap().restore_windows.len() < 6,
        "the restore never completed"
    );
    assert!(
        rig.compute.decode_step(&[DecodeJob {
            request: 1,
            lane: 0,
            params: DecodeParams::default(),
            remaining_tokens: 1,
            permitted: None,
        }])
        .is_err(),
        "and no sequence was handed back for the request"
    );
}

#[test]
fn a_torn_file_with_no_header_is_never_restored() {
    let rig = spilled("torn");
    let path = file_of(&rig, "live-1.kv");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[..HEADER_BYTES].fill(0);
    std::fs::write(&path, bytes).unwrap();
    refused(&rig);
}

#[test]
fn a_flipped_byte_in_a_window_is_never_restored() {
    for window in [0usize, 3, 5] {
        let rig = spilled(&format!("flip-{window}"));
        let path = file_of(&rig, "live-1.kv");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[HEADER_BYTES + window * WINDOW as usize + 17] ^= 0x40;
        std::fs::write(&path, bytes).unwrap();
        refused(&rig);
        let fed = rig.leaf.calls.lock().unwrap().restore_windows.len();
        assert!(fed <= window, "window {window} failed its CRC before it was fed (fed {fed})");
    }
}

#[test]
fn a_header_naming_another_load_is_never_restored() {
    let rig = spilled("foreign");
    let path = file_of(&rig, "live-1.kv");
    let mut bytes = std::fs::read(&path).unwrap();
    let mut header = FileHeader::decode(&bytes[..HEADER_BYTES]).unwrap();
    header.identity.model_id = "another-model".to_string();
    bytes[..HEADER_BYTES].copy_from_slice(&header.encode().unwrap());
    std::fs::write(&path, bytes).unwrap();
    refused(&rig);
    assert!(rig.leaf.calls.lock().unwrap().restore_windows.is_empty(), "refused before any byte moved");
}

// ── a spill that does not happen leaves its source as it was ────────────────

/// AC 21: a spill the scheduler discards mid-write (its request was
/// cancelled) deletes its file and frees the KV-RAM span it was writing
/// from: nothing is put back under a request that is gone.
#[test]
fn a_discarded_spill_deletes_its_file_and_frees_its_source() {
    let rig = new_rig("abandon");
    prefill(&rig.compute, 1);
    decode(&rig.compute, 1, 2);
    rig.compute.evict(1).unwrap();
    let blob = DiskBlob::Live(1);
    rig.compute.disk_spill(blob, DiskSource::KvRam, live_meta()).unwrap();
    rig.compute.disk_advance();
    rig.compute.disk_discard(blob);
    for _ in 0..200 {
        rig.compute.disk_advance();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(files_in(&rig).is_empty(), "its file is deleted once its writes drained");
    assert!(rig.compute.restore(1, 64).is_err(), "the KV-RAM span went with the discard, not back to the map");
}

// ── AC 22: the n-gram table goes first ──────────────────────────────────────

#[test]
fn a_spill_issues_no_io_while_a_prefill_gather_is_pending() {
    let rig = new_rig("gate");
    prefill(&rig.compute, 1);
    let held = rig.gate.enter();
    let blob = DiskBlob::Live(1);
    rig.compute.disk_spill(blob, DiskSource::Device, live_meta()).unwrap();
    for _ in 0..50 {
        assert!(rig.compute.disk_advance().is_empty(), "nothing lands while the gather is pending");
        std::thread::sleep(Duration::from_millis(1));
    }
    let copied = rig.leaf.calls.lock().unwrap().snapshot_windows.len();
    assert!(copied <= kv_disk::STAGING_WINDOWS, "the copies wait on the writes the gate holds: {copied}");
    drop(held);
    assert!(matches!(run(&rig.compute, blob).0, DiskOutcome::Spilled { .. }));
}

#[test]
fn the_directory_goes_with_the_adapter() {
    let rig = new_rig("shutdown");
    prefill(&rig.compute, 1);
    let blob = DiskBlob::Live(1);
    rig.compute.disk_spill(blob, DiskSource::Device, live_meta()).unwrap();
    rig.compute.disk_advance();
    let root = rig.location.join(kv_disk::DIR_NAME);
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
    let Rig { compute, _cleanup, .. } = rig;
    drop(compute);
    assert_eq!(
        std::fs::read_dir(&root).unwrap().count(),
        0,
        "a clean shutdown removes the process's directory, a spill under way included"
    );
}
