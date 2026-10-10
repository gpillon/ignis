//! Spec vram-budget/03 (ADR 0045) -- KV-disk through `RuntimeCompute` on a
//! CPU: a stub leaf whose sequences are byte vectors, windowed like a real
//! one, and the real store (a temp directory, real files, real IO threads).
//!
//! What this holds the adapter to: a blob spilled a window at a time and
//! read back is the same blob, its source given up only when the header
//! commits the file; a file torn, corrupt or foreign is never restored; a
//! spill's source keeps its bytes when the spill fails or is abandoned; the
//! n-gram table's prefill gathers go before the tier's IO; and a restore is
//! fed onto the device in order, a slice of the move-in pace at a time
//! (GitHub #309).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use ignis_core::identity::MatchKey;
use ignis_core::ngram_table::GatherGate;
use ignis_core::scheduler::{DiskBlob, DiskBlobMeta, DiskOp, DiskOutcome, DiskSource, DiskTarget};
use ignis_core::{Compute, DecodeJob, DecodeParams};
use ignis_runtime::kv_disk::{self, DiskIdentity, DiskStore, FileHeader, HEADER_BYTES};
use ignis_runtime::{Model, RuntimeCompute, TransferPace, TransferWindow};

mod byte_leaf;
use byte_leaf::{decode, prefill, ByteLeaf, BLOB_BYTES, WINDOW};

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
    new_rig_paced(tag, TransferPace::default())
}

/// A rig whose moves onto the device are fed at `pace` (GitHub #309).
fn new_rig_paced(tag: &str, pace: TransferPace) -> Rig {
    let location = temp_location(tag);
    let leaf = Arc::new(ByteLeaf::new());
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
    let compute = RuntimeCompute::new(model, 0).with_kv_disk(store, staging).with_transfer_pace(pace);
    let cleanup = TempDir(location.clone());
    Rig { compute, leaf, gate, location, _cleanup: cleanup }
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

/// The files left once the deletes already queued have run: a transfer that
/// ends queues its file's delete on the IO thread, which runs it after.
fn files_once_deleted(rig: &Rig) -> Vec<String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let names = files_in(rig);
        if names.is_empty() || std::time::Instant::now() > deadline {
            return names;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
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
    assert!(files_once_deleted(&rig).is_empty(), "a live file goes once it has landed");
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
            reasoning_close: None,
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

// ── GitHub #309: a restore's feed is paced ──────────────────────────────────

/// Advance until `rig`'s leaf has been fed `windows` restore windows.
fn until_fed(rig: &Rig, windows: usize) {
    for _ in 0..10_000 {
        if rig.leaf.calls.lock().unwrap().restore_windows.len() >= windows {
            return;
        }
        rig.compute.disk_advance();
        std::thread::sleep(Duration::from_micros(50));
    }
    panic!("the restore never fed {windows} window(s)");
}

#[test]
fn a_restore_is_fed_in_order_a_slice_of_the_move_in_pace_at_a_time() {
    let pace = TransferPace { move_in_bytes: 4096, move_out_bytes: 4096 };
    let rig = new_rig_paced("paced", pace);
    let twin = new_rig("paced-twin");
    for r in [&rig, &twin] {
        prefill(&r.compute, 1);
        decode(&r.compute, 1, 3);
    }
    let blob = DiskBlob::Live(1);
    rig.compute.disk_spill(blob, DiskSource::Device, live_meta()).unwrap();
    assert!(matches!(run(&rig.compute, blob).0, DiskOutcome::Spilled { .. }));
    rig.compute
        .disk_restore(blob, DiskTarget::Sequence { request: 1, context_tokens: 64 })
        .unwrap();
    assert!(matches!(run(&rig.compute, blob).0, DiskOutcome::Restored { .. }));
    let fed = rig.leaf.calls.lock().unwrap().restore_windows.clone();
    let mut next = 0;
    for slice in &fed {
        assert_eq!(slice.offset, next, "slices in the blob's order");
        assert!(slice.bytes <= 4096, "no slice over the pace: {slice:?}");
        next += slice.bytes;
    }
    assert_eq!(next, BLOB_BYTES as u64, "the whole blob, once");
    assert_eq!(fed.len(), 11, "each 8 KiB window in two slices, the last 1,000 bytes in one");
    assert_eq!(decode(&rig.compute, 1, 6), decode(&twin.compute, 1, 6), "and it is the same sequence");
}

#[test]
fn one_slice_of_a_restore_is_on_the_link_at_a_time() {
    let rig = new_rig_paced("one-slice", TransferPace { move_in_bytes: 4096, move_out_bytes: 4096 });
    prefill(&rig.compute, 1);
    let blob = DiskBlob::Live(1);
    rig.compute.disk_spill(blob, DiskSource::Device, live_meta()).unwrap();
    assert!(matches!(run(&rig.compute, blob).0, DiskOutcome::Spilled { .. }));
    rig.leaf.hold_fences();
    rig.compute
        .disk_restore(blob, DiskTarget::Sequence { request: 1, context_tokens: 64 })
        .unwrap();
    until_fed(&rig, 1);
    for _ in 0..50 {
        assert!(rig.compute.disk_advance().is_empty());
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        rig.leaf.calls.lock().unwrap().restore_windows.len(),
        1,
        "the next slice waits for the one on the link to land, both windows read or not"
    );
    rig.leaf.release_fences();
    assert!(matches!(run(&rig.compute, blob).0, DiskOutcome::Restored { .. }));
}

// ── GitHub #310: a spill's device copies are paced ─────────────────────────

#[test]
fn a_spill_from_the_device_is_copied_in_order_a_slice_of_the_move_out_pace_at_a_time() {
    let rig = new_rig_paced("spill-paced", TransferPace { move_in_bytes: 4096, move_out_bytes: 4096 });
    let twin = new_rig("spill-paced-twin");
    for r in [&rig, &twin] {
        prefill(&r.compute, 1);
        decode(&r.compute, 1, 3);
    }
    let blob = DiskBlob::Live(1);
    rig.compute.disk_spill(blob, DiskSource::Device, live_meta()).unwrap();
    let (outcome, advances) = run(&rig.compute, blob);
    assert!(matches!(outcome, DiskOutcome::Spilled { .. }));
    let copied = rig.leaf.calls.lock().unwrap().snapshot_windows.clone();
    let mut next = 0;
    for slice in &copied {
        assert_eq!(slice.offset, next, "slices in the blob's order");
        assert!(slice.bytes <= 4096, "no slice over the pace: {slice:?}");
        next += slice.bytes;
    }
    assert_eq!(next, BLOB_BYTES as u64, "the whole blob, once");
    assert_eq!(copied.len(), 11, "each 8 KiB window in two slices, the last 1,000 bytes in one");
    assert!(advances >= copied.len(), "at most one new slice an advance: {advances} advances");
    rig.compute
        .disk_restore(blob, DiskTarget::Sequence { request: 1, context_tokens: 64 })
        .unwrap();
    assert!(matches!(run(&rig.compute, blob).0, DiskOutcome::Restored { .. }));
    assert_eq!(decode(&rig.compute, 1, 6), decode(&twin.compute, 1, 6), "and the file holds the same sequence");
}

#[test]
fn one_slice_of_a_spill_is_on_the_link_at_a_time() {
    let rig = new_rig_paced("spill-one-slice", TransferPace { move_in_bytes: 4096, move_out_bytes: 4096 });
    prefill(&rig.compute, 1);
    rig.leaf.hold_fences();
    let blob = DiskBlob::Live(1);
    rig.compute.disk_spill(blob, DiskSource::Device, live_meta()).unwrap();
    let mut ended = Vec::new();
    for _ in 0..50 {
        ended.extend(rig.compute.disk_advance());
        std::thread::sleep(Duration::from_millis(1));
    }
    let copied = rig.leaf.calls.lock().unwrap().snapshot_windows.len();
    // Released before anything is asserted: an adapter dropped with a copy
    // still on the link waits for it.
    rig.leaf.release_fences();
    assert!(ended.is_empty(), "nothing ends while its copy is on the link");
    assert_eq!(copied, 1, "the next slice waits for the one on the link to land, a staging slot free or not");
    assert!(matches!(run(&rig.compute, blob).0, DiskOutcome::Spilled { .. }));
}

/// Advance until `rig`'s leaf has copied `slices` spill slices off the device.
fn until_copied(rig: &Rig, slices: usize) {
    for _ in 0..10_000 {
        if rig.leaf.calls.lock().unwrap().snapshot_windows.len() >= slices {
            return;
        }
        rig.compute.disk_advance();
        std::thread::sleep(Duration::from_micros(50));
    }
    panic!("the spill never copied {slices} slice(s)");
}

#[test]
fn a_spill_abandoned_mid_window_gives_its_staging_back() {
    let rig = new_rig_paced("spill-abandon", TransferPace { move_in_bytes: 4096, move_out_bytes: 4096 });
    prefill(&rig.compute, 1);
    let blob = DiskBlob::Live(1);
    rig.compute.disk_spill(blob, DiskSource::Device, live_meta()).unwrap();
    // The window's first slice lands at once; its second stays on the link.
    until_copied(&rig, 1);
    rig.leaf.hold_fences();
    until_copied(&rig, 2);
    let busy = rig.compute.disk_staging_busy();
    rig.compute.disk_discard(blob);
    let waited = rig.leaf.calls.lock().unwrap().waits;
    rig.leaf.release_fences();
    for _ in 0..200 {
        rig.compute.disk_advance();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(busy > 0, "a window was half copied");
    assert_eq!(waited, 0, "the model thread did not wait for the slice on the link");
    assert_eq!(rig.compute.disk_staging_busy(), 0, "every staging window came back once its slice landed");
    assert!(files_in(&rig).is_empty(), "and the file went");
}

#[test]
fn a_spill_failing_mid_window_gives_its_staging_back() {
    let rig = new_rig_paced("spill-fail", TransferPace { move_in_bytes: 4096, move_out_bytes: 4096 });
    prefill(&rig.compute, 1);
    let blob = DiskBlob::Live(1);
    rig.compute.disk_spill(blob, DiskSource::Device, live_meta()).unwrap();
    until_copied(&rig, 1);
    rig.leaf.fail_windows(true);
    let (outcome, _) = run(&rig.compute, blob);
    rig.leaf.fail_windows(false);
    assert!(matches!(outcome, DiskOutcome::Failed { op: DiskOp::Write }), "{outcome:?}");
    for _ in 0..200 {
        rig.compute.disk_advance();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(rig.compute.disk_staging_busy(), 0, "the half-copied window's slot came back");
    assert_eq!(decode(&rig.compute, 1, 1).len(), 1, "and the sequence is still the request's");
}

#[test]
fn a_restore_abandoned_mid_feed_gives_its_staging_back() {
    let rig = new_rig("abandon-feed");
    prefill(&rig.compute, 1);
    let blob = DiskBlob::Live(1);
    rig.compute.disk_spill(blob, DiskSource::Device, live_meta()).unwrap();
    assert!(matches!(run(&rig.compute, blob).0, DiskOutcome::Spilled { .. }));
    rig.leaf.hold_fences();
    rig.compute
        .disk_restore(blob, DiskTarget::Sequence { request: 1, context_tokens: 64 })
        .unwrap();
    until_fed(&rig, 1);
    assert!(rig.compute.disk_staging_busy() > 0, "a window is on its way");
    rig.compute.disk_discard(blob);
    assert!(rig.leaf.calls.lock().unwrap().waits >= 1, "the slice on the link was waited for");
    for _ in 0..200 {
        rig.compute.disk_advance();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(rig.compute.disk_staging_busy(), 0, "every staging window came back");
    assert!(files_in(&rig).is_empty(), "and the file went");
}
