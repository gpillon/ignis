//! GitHub #309 (spec vram-budget/03 AC 37) -- live moves through KV-RAM a
//! window at a time, through `RuntimeCompute` on a CPU: a stub leaf whose
//! sequences are byte vectors and whose transfer stream is a counter of
//! fences the test can hold.
//!
//! What this holds the adapter to: a move out copies its sequence into its
//! span a window of the pace at a time, one on the link, and releases the
//! sequence only once the last has landed; a move in feeds a fresh sequence
//! from the span in order and hands it to the request only once the last has
//! landed; a move that ends early waits for its window on the link -- and no
//! more -- before it lets anything go, leaving the sequence live (out) or the
//! snapshot where it was (in); and a leaf with no transfer stream moves in
//! one call.

use std::sync::Arc;

use ignis_core::scheduler::{KvRamMove, KvRamOutcome};
use ignis_core::{Compute, RequestId};
use ignis_core::compute::ModelFamily;
use ignis_runtime::{Model, RuntimeCompute, TransferPace, TransferWindow, MOVE_IN_WINDOW_BYTES, MOVE_OUT_WINDOW_BYTES};

mod byte_leaf;
use byte_leaf::{decode, prefill, ByteLeaf, BLOB_BYTES, WINDOW};

/// A window of [`WINDOW`] each way: the stub's blob is six of them.
const PACE: TransferPace = TransferPace { move_in_bytes: WINDOW, move_out_bytes: WINDOW };

fn rig(leaf: ByteLeaf) -> (RuntimeCompute<ByteLeaf>, Arc<ByteLeaf>) {
    let leaf = Arc::new(leaf);
    let model = Arc::new(Model::load(Arc::clone(&leaf)).unwrap());
    (RuntimeCompute::new(model, 0).with_transfer_pace(PACE), leaf)
}

/// A rig and its never-moved twin, `request` prefilled and decoded alike in
/// both.
fn twins(request: RequestId) -> ((RuntimeCompute<ByteLeaf>, Arc<ByteLeaf>), RuntimeCompute<ByteLeaf>) {
    let moved = rig(ByteLeaf::new());
    let (twin, _) = rig(ByteLeaf::new());
    for compute in [&moved.0, &twin] {
        prefill(compute, request);
        decode(compute, request, 3);
    }
    (moved, twin)
}

/// Advance until `request`'s move ends: its outcome, and how many advances.
fn run(compute: &RuntimeCompute<ByteLeaf>, request: RequestId) -> (KvRamOutcome, usize) {
    for step in 1..1_000 {
        if let Some(event) = compute.kv_ram_advance().into_iter().find(|e| e.request == request) {
            return (event.outcome, step);
        }
    }
    panic!("the move of {request} never ended");
}

fn windows(leaf: &ByteLeaf, out: bool) -> Vec<TransferWindow> {
    let calls = leaf.calls.lock().unwrap();
    if out { calls.snapshot_windows.clone() } else { calls.restore_windows.clone() }
}

/// `windows` are the blob, in order, one of the pace at a time.
fn assert_tiles_the_blob(windows: &[TransferWindow]) {
    let mut next = 0;
    for w in windows {
        assert_eq!(w.offset, next, "windows in the blob's order");
        assert!(w.bytes <= WINDOW, "no window over the pace: {w:?}");
        next += w.bytes;
    }
    assert_eq!(next, BLOB_BYTES as u64, "the whole blob, once");
}

#[test]
fn a_move_out_copies_a_window_at_a_time_and_lets_the_sequence_go_only_once_it_has_landed() {
    let ((compute, leaf), _) = twins(1);
    leaf.hold_fences();
    let (bytes, how) = compute.kv_ram_move_out(1).unwrap();
    assert_eq!((bytes, how), (BLOB_BYTES as u64, KvRamMove::Started));
    assert_eq!(windows(&leaf, true).len(), 1, "the first window went out at the start");
    for _ in 0..10 {
        assert!(compute.kv_ram_advance().is_empty());
    }
    assert_eq!(windows(&leaf, true).len(), 1, "no second window while the first is on the link");
    assert_eq!(compute.live_sequences(), 1, "the sequence is the request's until the move lands");
    assert_eq!(leaf.calls.lock().unwrap().released, 0);

    // The link moves: one new window an advance, then the landing.
    leaf.release_fences();
    let (outcome, steps) = run(&compute, 1);
    assert!(matches!(outcome, KvRamOutcome::Landed { .. }), "{outcome:?}");
    assert_eq!(steps, 6, "five more windows, one an advance, then the landing");
    assert_tiles_the_blob(&windows(&leaf, true));
    assert_eq!(leaf.calls.lock().unwrap().released, 1, "released with the landing, not before");
    assert_eq!(compute.live_sequences(), 0);
    assert_eq!(compute.evicted_sequences(), 1, "the span is the request's snapshot");
}

#[test]
fn a_move_in_feeds_its_sequence_in_order_and_hands_it_over_only_once_it_has_landed() {
    let ((compute, leaf), twin) = twins(1);
    compute.kv_ram_move_out(1).unwrap();
    assert!(matches!(run(&compute, 1).0, KvRamOutcome::Landed { .. }));
    leaf.hold_fences();
    assert_eq!(compute.kv_ram_move_in(1, 64).unwrap(), KvRamMove::Started);
    for _ in 0..10 {
        assert!(compute.kv_ram_advance().is_empty());
    }
    assert_eq!(windows(&leaf, false).len(), 1, "one window on the link at a time");
    assert_eq!(compute.live_sequences(), 0, "not the request's before its last window lands");
    leaf.release_fences();
    assert!(matches!(run(&compute, 1).0, KvRamOutcome::Landed { .. }));
    assert_tiles_the_blob(&windows(&leaf, false));
    assert_eq!(compute.live_sequences(), 1);
    assert_eq!(compute.evicted_sequences(), 0, "the span went back with the landing");
    assert_eq!(decode(&compute, 1, 6), decode(&twin, 1, 6), "and the sequence is the one that left");
}

#[test]
fn a_move_out_abandoned_mid_copy_waits_for_its_window_and_leaves_the_sequence_live() {
    let ((compute, leaf), twin) = twins(1);
    leaf.hold_fences();
    compute.kv_ram_move_out(1).unwrap();
    compute.kv_ram_abandon(1);
    assert_eq!(leaf.calls.lock().unwrap().waits, 1, "the window on the link was waited for, and no more");
    assert_eq!(windows(&leaf, true).len(), 1);
    assert_eq!(compute.evicted_sequences(), 0, "its span went");
    assert_eq!(compute.live_sequences(), 1);
    assert_eq!(leaf.calls.lock().unwrap().released, 0, "the sequence is the request's to release");
    leaf.release_fences();
    assert!(compute.kv_ram_advance().is_empty(), "an abandoned move reports nothing");
    assert_eq!(decode(&compute, 1, 4), decode(&twin, 1, 4));
}

#[test]
fn a_move_in_abandoned_mid_feed_releases_its_sequence_and_keeps_the_snapshot() {
    let ((compute, leaf), twin) = twins(1);
    compute.kv_ram_move_out(1).unwrap();
    assert!(matches!(run(&compute, 1).0, KvRamOutcome::Landed { .. }));
    let released = leaf.calls.lock().unwrap().released;
    leaf.hold_fences();
    compute.kv_ram_move_in(1, 64).unwrap();
    compute.kv_ram_abandon(1);
    assert_eq!(leaf.calls.lock().unwrap().waits, 1, "the window on the link was waited for");
    assert_eq!(leaf.calls.lock().unwrap().released, released + 1, "the half-built sequence went");
    assert_eq!(compute.live_sequences(), 0);
    assert_eq!(compute.evicted_sequences(), 1, "the snapshot is where it was");
    // ... and comes back whole on a later move.
    leaf.release_fences();
    compute.kv_ram_move_in(1, 64).unwrap();
    assert!(matches!(run(&compute, 1).0, KvRamOutcome::Landed { .. }));
    assert_eq!(decode(&compute, 1, 4), decode(&twin, 1, 4));
}

#[test]
fn a_window_that_fails_ends_the_move_and_loses_nothing() {
    // Out: the sequence stays live, the span goes.
    let ((compute, leaf), twin) = twins(1);
    compute.kv_ram_move_out(1).unwrap();
    leaf.fail_windows(true);
    assert_eq!(run(&compute, 1).0, KvRamOutcome::Failed);
    assert_eq!(compute.live_sequences(), 1);
    assert_eq!(compute.evicted_sequences(), 0);
    leaf.fail_windows(false);
    assert_eq!(decode(&compute, 1, 4), decode(&twin, 1, 4), "the sequence was never touched");

    // In: the half-built sequence goes, the snapshot stays.
    compute.kv_ram_move_out(1).unwrap();
    assert!(matches!(run(&compute, 1).0, KvRamOutcome::Landed { .. }));
    compute.kv_ram_move_in(1, 64).unwrap();
    leaf.fail_windows(true);
    assert_eq!(run(&compute, 1).0, KvRamOutcome::Failed);
    assert_eq!(compute.live_sequences(), 0);
    assert_eq!(compute.evicted_sequences(), 1, "the snapshot is where it was");
}

#[test]
fn a_first_window_that_fails_ends_the_move_at_the_next_advance() {
    // As any failed window does, so the scheduler's handling of a failure is
    // one path.
    let ((compute, leaf), _) = twins(1);
    leaf.fail_windows(true);
    assert_eq!(compute.kv_ram_move_out(1).unwrap().1, KvRamMove::Started);
    assert_eq!(run(&compute, 1), (KvRamOutcome::Failed, 1));
    assert_eq!((compute.live_sequences(), compute.evicted_sequences()), (1, 0), "nothing moved");
    leaf.fail_windows(false);
    compute.kv_ram_move_out(1).unwrap();
    assert!(matches!(run(&compute, 1).0, KvRamOutcome::Landed { .. }));
    leaf.fail_windows(true);
    assert_eq!(compute.kv_ram_move_in(1, 64).unwrap(), KvRamMove::Started);
    assert_eq!(run(&compute, 1), (KvRamOutcome::Failed, 1));
    assert_eq!((compute.live_sequences(), compute.evicted_sequences()), (0, 1), "the snapshot stayed");
}

#[test]
fn a_leaf_without_a_transfer_stream_moves_in_one_call() {
    let (compute, leaf) = rig(ByteLeaf::without_transfer_stream());
    prefill(&compute, 1);
    assert_eq!(compute.kv_ram_move_out(1).unwrap(), (BLOB_BYTES as u64, KvRamMove::Done));
    assert_eq!(compute.evicted_sequences(), 1);
    assert_eq!(compute.kv_ram_move_in(1, 64).unwrap(), KvRamMove::Done);
    assert_eq!(compute.live_sequences(), 1);
    assert!(windows(&leaf, true).is_empty() && windows(&leaf, false).is_empty(), "no window was asked for");
}

#[test]
fn a_pace_is_at_least_a_sector() {
    let (compute, _) = rig(ByteLeaf::new());
    let compute = compute.with_transfer_pace(TransferPace { move_in_bytes: 1, move_out_bytes: 0 });
    assert_eq!(
        compute.transfer_pace(),
        TransferPace {
            move_in_bytes: TransferPace::MIN_WINDOW_BYTES,
            move_out_bytes: TransferPace::MIN_WINDOW_BYTES,
        }
    );
}

#[test]
fn moves_under_way_are_let_go_with_the_adapter() {
    let ((compute, leaf), _) = twins(1);
    prefill(&compute, 2);
    compute.kv_ram_move_out(2).unwrap();
    assert!(matches!(run(&compute, 2).0, KvRamOutcome::Landed { .. }));
    leaf.hold_fences();
    compute.kv_ram_move_out(1).unwrap();
    compute.kv_ram_move_in(2, 64).unwrap();
    let released = leaf.calls.lock().unwrap().released;
    drop(compute);
    let calls = leaf.calls.lock().unwrap();
    assert_eq!(calls.waits, 2, "each move's window on the link was waited for");
    assert_eq!(calls.released, released + 2, "1's live sequence and 2's half-built one");
}

#[test]
fn flash_next_moves_are_paced_and_the_27bs_go_in_one_window() {
    assert_eq!(
        TransferPace::for_family(ModelFamily::FlashNext),
        TransferPace { move_in_bytes: MOVE_IN_WINDOW_BYTES, move_out_bytes: MOVE_OUT_WINDOW_BYTES }
    );
    assert_eq!(TransferPace::for_family(ModelFamily::Qwen38_27b), TransferPace::UNPACED);

    let (compute, leaf) = rig(ByteLeaf::new());
    let compute = compute.with_transfer_pace(TransferPace::UNPACED);
    prefill(&compute, 1);
    compute.kv_ram_move_out(1).unwrap();
    assert!(matches!(run(&compute, 1).0, KvRamOutcome::Landed { .. }));
    compute.kv_ram_move_in(1, 64).unwrap();
    assert!(matches!(run(&compute, 1).0, KvRamOutcome::Landed { .. }));
    let whole = TransferWindow { offset: 0, bytes: BLOB_BYTES as u64, blob_bytes: BLOB_BYTES as u64 };
    assert_eq!(windows(&leaf, true), vec![whole], "out in one window");
    assert_eq!(windows(&leaf, false), vec![whole], "and back in one");
}
