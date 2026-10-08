//! The deterministic [`Compute`] mock — the kernel-leaf stand-in for CPU
//! tests (ADR 0006: a mock stands in for the kernel leaf, which keeps the
//! whole scheduler CPU-testable without a GPU).
//!
//! The mock is *deterministic by construction*: the token a request
//! generates at step `i` is a pure function of (mock seed, request id,
//! request seed, `i`) — no RNG, no clocks. That mirrors the production
//! floor (greedy + fixed seed, ADR 0007) and lets tests pin exact event
//! streams. It also *records* the batch shape of every call, so tests can
//! assert the scheduler's batching behavior (batched prefill groups N
//! requests into one call, not N calls) without a GPU.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::decision::{Readout, argmax, log_sum_exp};
use crate::scheduler::{
    Compute, DecodeJob, DecodeOutcome, DiskBlob, DiskBlobMeta, DiskEvent, DiskOp, DiskOutcome, DiskSource, DiskTarget,
    KvRamEvent, KvRamMove, KvRamOutcome, PrefillJob, PrefillOutcome, NO_HOST_ROOM,
};
use crate::types::{ComputeError, FinishReason, RequestId, SpecCounters, TokenId};

/// Recording handle onto the mock's call history (shared through the
/// `Arc<dyn Compute>` the scheduler holds, so tests can assert the batch
/// shape after driving the scheduler).
#[derive(Default)]
struct Inner {
    /// Max tokens per request, learned from the prefill jobs' params.
    limits: HashMap<RequestId, Option<u32>>,
    /// Per-request generation seeds, learned from the prefill jobs' params.
    seeds: HashMap<RequestId, u64>,
    /// Explicit stop points (`stop_after`), overriding the learned limit.
    stops: HashMap<RequestId, u32>,
    /// Tokens generated so far, per request.
    generated: HashMap<RequestId, u32>,
    /// Run lengths to commit per decode round (P5-06, GitHub #154), cycled
    /// per request; empty is today's round of one.
    run_lengths: Vec<u32>,
    /// Decode rounds served so far, per request (indexes `run_lengths`).
    rounds: HashMap<RequestId, usize>,
    /// The generation step whose token is EOS (`eos_after`), per request.
    eos: HashMap<RequestId, u32>,
    /// Every prefill batch the mock received (batch shape for assertions).
    prefill_batches: Vec<Vec<PrefillJob>>,
    /// Every decode batch the mock received (batch shape for assertions).
    decode_batches: Vec<Vec<DecodeJob>>,
    /// Shared prefixes the scheduler told the backend to let go of (P4-10,
    /// GitHub #126), in order: the publishing request of each. The real
    /// adapter drops its leaf handle here, so a test that never sees the
    /// call is looking at a prefix the engine would have pinned forever.
    prefixes_released: Vec<RequestId>,
    /// Requests whose device state the scheduler released, in order.
    released: Vec<RequestId>,
    /// Retained prompt checkpoints the scheduler told the backend to let go
    /// of (GitHub #186), in order: the capturing request of each. A retained
    /// entry the scheduler drops without this call is a device image nothing
    /// will ever free.
    checkpoints_released: Vec<RequestId>,
    /// Retained checkpoints the scheduler moved to KV-RAM (GitHub #190), in
    /// order — each one a device-to-host copy. Kept apart from
    /// `checkpoints_released` so a test can tell a spill from a discard.
    checkpoints_spilled: Vec<RequestId>,
    /// Requests whose next checkpoint spill the backend will fail.
    spill_failures: std::collections::HashSet<RequestId>,
    /// Retained prefixes written to KV-RAM, brought back, and freed there
    /// (GitHub #190), as (publisher, tokens).
    prefixes_spilled: Vec<(RequestId, u32)>,
    prefixes_returned: Vec<(RequestId, u32)>,
    spilled_prefixes_discarded: Vec<(RequestId, u32)>,
    /// Requests whose next prefill batch the backend will fail (GitHub #190).
    prefill_failures: std::collections::HashSet<RequestId>,
    /// Requests whose next asked-for checkpoint capture the backend will
    /// decline (`refuse_capture`) — a leaf with no room in its own image
    /// pool, or a sequence it will not capture.
    capture_refusals: std::collections::HashSet<RequestId>,
    /// Pages the leaf reports lending at a request's next capture
    /// (`lend_at_capture`, GitHub #306); 0 for any request not named.
    capture_lent_pages: HashMap<RequestId, u32>,
    /// Requests whose next attention readout the leaf "could not read"
    /// (GitHub #260): the chunk lands and the outcome carries no scores.
    attention_refusals: std::collections::HashSet<RequestId>,
    /// The leaf's KV-RAM arena, modelled (GitHub #213), or `None` for a
    /// backend whose blobs are ordinary allocations — today's default, and
    /// what every scenario that is not about placement wants.
    arena: Option<MockHostArena>,
    /// The draw a **constrained decode** request has made and not yet emitted (GitHub
    /// #242), per request: the token the *next* decode round returns.
    ///
    /// This is the leaf's one-round lag, modelled on purpose. A mock that
    /// emitted the set it was handed in the same call would let a scheduler
    /// that constrained only its rounds pass every CPU test and read one
    /// free token in the middle of its forced text on the card — which is
    /// exactly the bug `permitted_decode_gpu.rs` caught, and exactly the
    /// bug a mock exists to catch first (ADR 0006).
    pending: HashMap<RequestId, crate::constrained::Draw>,
    /// Constrained steps served so far, per request: what varies the mock's
    /// pick from one step to the next.
    drawn: HashMap<RequestId, u32>,
    /// Where each request's last prefill chunk ended: with `generated`, the
    /// tokens a live snapshot of it covers.
    prefilled: HashMap<RequestId, u32>,
    /// The tokens each captured checkpoint covers, by publisher: what its
    /// materialized blob is sized from.
    checkpoint_tokens: HashMap<RequestId, u32>,
    /// The fake KV-disk (spec vram-budget/03), or `None`: a backend with no
    /// disk, which takes nothing -- every scenario that is not about the
    /// tier.
    disk: Option<FakeDiskState>,
    /// Live moves through KV-RAM a window at a time (GitHub #309), or `None`:
    /// moves in one call, the default.
    ram: Option<FakeRamState>,
}

/// How the mock moves a live sequence through KV-RAM a window at a time
/// (GitHub #309): each move ends `advances` calls to
/// [`Compute::kv_ram_advance`] after it started.
#[derive(Debug, Default)]
struct FakeRamState {
    advances: u32,
    moves: Vec<FakeRamMove>,
    /// Every move from now on fails when it would have landed
    /// (`fail_kv_ram_moves`).
    failing: bool,
    /// Every move started, in order: the request and whether it went out.
    started: Vec<(RequestId, bool)>,
}

#[derive(Debug, Clone, Copy)]
struct FakeRamMove {
    request: RequestId,
    out: bool,
    /// Advances left before it ends.
    left: u32,
}

/// A fake KV-disk's shape (spec vram-budget/03): the volume's room, and how
/// long a transfer takes. A blob moves in `ceil(bytes / window_bytes)`
/// windows, one every `advances_per_window` calls to
/// [`Compute::disk_advance`] -- "a second per window" at one advance a
/// second -- so a scenario can watch the other lanes decode while one moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FakeDisk {
    /// The volume's room for files, in bytes ([`Compute::disk_fits`]).
    pub room_bytes: u64,
    pub window_bytes: u64,
    pub advances_per_window: u32,
}

impl FakeDisk {
    /// A disk with `room_bytes` of room, every blob one window, each window
    /// one advance: the fastest a transfer can be and still be a transfer.
    pub fn with_room(room_bytes: u64) -> Self {
        Self {
            room_bytes,
            window_bytes: u64::MAX,
            advances_per_window: 1,
        }
    }
}

/// One transfer the fake disk has under way.
#[derive(Debug, Clone)]
struct FakeTransfer {
    blob: DiskBlob,
    /// `Some(from)` for a spill, `None` for a restore.
    from: Option<DiskSource>,
    /// A restore's target.
    into: Option<DiskTarget>,
    /// The file's bytes (a spill's, once committed), and the blob's.
    file_bytes: u64,
    blob_bytes: u64,
    windows_left: u64,
    /// Advances before the next window moves.
    wait: u32,
}

#[derive(Debug)]
struct FakeDiskState {
    shape: FakeDisk,
    /// Committed files: their bytes on the volume, and the blob's own.
    files: HashMap<DiskBlob, (u64, u64)>,
    transfers: Vec<FakeTransfer>,
    /// Every write fails once its first window moved (`fail_disk_writes`).
    failing_writes: bool,
    /// Files whose next read fails its check (`corrupt_disk_file`).
    corrupt: std::collections::HashSet<DiskBlob>,
    spills: Vec<(DiskBlob, DiskSource)>,
    restores: Vec<DiskBlob>,
    discards: Vec<DiskBlob>,
    /// For each `disk_advance` call, the blobs that moved a window.
    windows: Vec<Vec<DiskBlob>>,
}

/// What the mock charges a KV-RAM blob: a mutable image plus the paged bytes
/// of every token it covers (spec flash-next/05).
///
/// [`MockSections::NOMINAL`], one byte a blob whatever its length, is what a
/// scenario that counts entries wants, and the mock's default.
/// [`MockSections::of`] is a model's real sizes, under which the same
/// scenarios run with blobs of the size and the spread a load sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MockSections {
    pub image_bytes: u64,
    pub bytes_per_token: u64,
}

impl MockSections {
    /// One nominal byte a blob.
    pub const NOMINAL: Self = Self {
        image_bytes: 1,
        bytes_per_token: 0,
    };

    /// `cfg`'s sizes under `kv_format`: its state image as a slot holds it,
    /// and its paged sections per token.
    pub fn of(cfg: &crate::compute::ModelConfig, kv_format: crate::kv_format::KvFormat) -> Self {
        Self {
            image_bytes: cfg.state_image(kv_format).slot_bytes(),
            bytes_per_token: cfg.paged_sections(kv_format).bytes_per_token(),
        }
    }

    /// A blob covering `tokens`.
    pub fn blob_bytes(&self, tokens: u32) -> u64 {
        self.image_bytes + u64::from(tokens) * self.bytes_per_token
    }

    /// KV-RAM bytes for `blobs` blobs of at most `max_tokens` each: the
    /// largest blob, `blobs` times. That holds exactly `blobs` of them,
    /// whatever their lengths, while the image outweighs `blobs` blobs'
    /// paged bytes -- as it does for every scenario's few blobs -- and at
    /// least `blobs` otherwise. [`MockSections::NOMINAL`]'s is `blobs`.
    pub fn capacity_for(&self, blobs: u64, max_tokens: u32) -> u64 {
        blobs.saturating_mul(self.blob_bytes(max_tokens))
    }

    /// `config` with its KV-RAM capacity, which a scenario states in
    /// nominal blobs, in bytes of these: [`MockSections::capacity_for`] at
    /// the config's own longest sequence. [`MockSections::NOMINAL`] leaves
    /// it as it is.
    pub fn scale(&self, config: crate::concrete::SchedulerConfig) -> crate::concrete::SchedulerConfig {
        crate::concrete::SchedulerConfig {
            host_capacity_bytes: self.capacity_for(config.host_capacity_bytes, config.max_sequence_tokens),
            ..config
        }
    }
}

/// Which blob a span of [`MockHostArena`] holds: the three kinds the host
/// tier places there (`CONTEXT.md`: "Snapshot blob").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MockBlob {
    /// An evicted live sequence, named by the request it suspends.
    Live(RequestId),
    /// A spilled prompt checkpoint, named by its publisher.
    Checkpoint(RequestId),
    /// A spilled retained prefix, named by publisher and length.
    Prefix(RequestId, u32),
}

/// A first-fit model of the leaf's one pinned KV-RAM arena (GitHub #213,
/// ADR 0030), so the scheduler's behaviour under *fragmentation* is testable
/// on a CPU.
///
/// It is not a second implementation of the vendored `HostPinnedArena` — it
/// is a way to make holes appear on demand. What it shares with the real one
/// is the only thing the scheduler can observe: first fit, bytes counted as
/// asked for, and a blob that finds no long-enough span refused even while
/// the tier's byte ledger says the bytes are free. The real arena's own
/// placement is pinned in `kernel/tests/test_seq_snapshot.cpp`.
///
/// It packs with no alignment, where the real one rounds each blob's start
/// up and gives the padding back to the free list. That is where the two
/// first-fits can diverge, and it is deliberate: a scenario here says how
/// many blobs of what size, never where the bytes land.
#[derive(Debug)]
struct MockHostArena {
    capacity: u64,
    /// The live blobs as (blob, offset, length), sorted by offset, so the
    /// gaps between them are the free spans.
    live: Vec<(MockBlob, u64, u64)>,
}

impl MockHostArena {
    fn new(capacity: u64) -> Self {
        Self {
            capacity,
            live: Vec::new(),
        }
    }

    /// The lowest offset a blob of `bytes` fits at, or `None` when no span
    /// is long enough.
    fn first_fit(&self, bytes: u64) -> Option<u64> {
        let mut at = 0;
        for &(_, offset, length) in &self.live {
            if offset - at >= bytes {
                return Some(at);
            }
            at = offset + length;
        }
        (self.capacity - at >= bytes).then_some(at)
    }

    fn place(&mut self, blob: MockBlob, bytes: u64) -> bool {
        let Some(offset) = self.first_fit(bytes) else {
            return false;
        };
        let at = self.live.partition_point(|&(_, o, _)| o < offset);
        self.live.insert(at, (blob, offset, bytes));
        true
    }

    /// Give `blob`'s span back. A blob the arena never placed is a no-op:
    /// the scheduler frees what it believes it spilled, and a scenario that
    /// spilled nothing has nothing to return.
    fn free(&mut self, blob: MockBlob) {
        self.live.retain(|&(held, _, _)| held != blob);
    }

    fn used(&self) -> u64 {
        self.live.iter().map(|&(_, _, length)| length).sum()
    }
}

/// A deterministic, recording [`Compute`] implementation for tests.
///
/// Token generation: the token a request emits at decode step `i` mixes the
/// mock's seed, the request id, the request's (learned) seed, and `i` — a
/// fixed, side-effect-free function, so identical runs produce identical
/// streams and different request seeds produce different streams.
pub struct MockCompute {
    seed: u64,
    identity: crate::identity::BlobIdentity,
    sections: MockSections,
    inner: Mutex<Inner>,
}

impl MockCompute {
    /// A mock with the default (zero) seed.
    pub fn new() -> Self {
        Self::with_seed(0)
    }

    /// A mock whose token streams are mixed with `seed` (test variation
    /// knob; the request's own seed always takes part in the mix too).
    pub fn with_seed(seed: u64) -> Self {
        Self {
            seed,
            identity: crate::identity::BlobIdentity::UNSET,
            sections: MockSections::NOMINAL,
            inner: Mutex::new(Inner::default()),
        }
    }

    /// A mock whose KV-RAM blobs are sized by `sections` (spec
    /// flash-next/05): a model's real image and per-token bytes instead of
    /// one nominal byte each.
    pub fn with_sections(sections: MockSections) -> Self {
        Self {
            sections,
            ..Self::new()
        }
    }

    /// A mock whose state is produced under `identity` (GitHub #189): what a
    /// test uses to stand in for a real load, or for a *different* load than
    /// the one some blob came from.
    pub fn with_blob_identity(identity: crate::identity::BlobIdentity) -> Self {
        Self {
            identity,
            ..Self::new()
        }
    }

    /// A mock that commits runs like a speculative round (P5-06, GitHub
    /// #154): round `r` of a request commits up to `lengths[r % len]` tokens,
    /// cut short by the request's limit or its EOS, and reports the round's
    /// counters — `length - 1` drafted, `committed - 1` accepted.
    pub fn with_runs(lengths: &[u32]) -> Self {
        assert!(lengths.iter().all(|&n| n >= 1), "a run commits at least its anchor");
        let mock = Self::new();
        mock.inner.lock().unwrap().run_lengths = lengths.to_vec();
        mock
    }

    /// A mock whose KV-RAM blobs are placed first-fit in one arena of
    /// `capacity_bytes` (GitHub #213), the way the leaf's pinned arena
    /// places them.
    ///
    /// Give it the same figure as `SchedulerConfig::host_capacity_bytes` and
    /// the two start out saying the same thing. Every blob here is the
    /// nominal byte, so what fragments the arena is asking it for a blob
    /// longer than any one hole — two free bytes in two holes are not two
    /// bytes of room.
    pub fn with_host_arena(capacity_bytes: u64) -> Self {
        let mock = Self::new();
        mock.inner.lock().unwrap().arena = Some(MockHostArena::new(capacity_bytes));
        mock
    }

    /// A mock with a fake KV-disk of `disk`'s shape (spec vram-budget/03).
    pub fn with_disk(self, disk: FakeDisk) -> Self {
        self.inner.lock().unwrap().disk = Some(FakeDiskState {
            shape: disk,
            files: HashMap::new(),
            transfers: Vec::new(),
            failing_writes: false,
            corrupt: std::collections::HashSet::new(),
            spills: Vec::new(),
            restores: Vec::new(),
            discards: Vec::new(),
            windows: Vec::new(),
        });
        self
    }

    /// A mock that moves live sequences through KV-RAM a window at a time
    /// (GitHub #309), as the real adapter does: a move ends `advances` calls
    /// to [`Compute::kv_ram_advance`] after it started, at least one.
    pub fn with_windowed_kv_ram(self, advances: u32) -> Self {
        self.inner.lock().unwrap().ram = Some(FakeRamState {
            advances: advances.max(1),
            ..FakeRamState::default()
        });
        self
    }

    /// Make every windowed KV-RAM move from now on fail when it would have
    /// landed (or land again, with `false`): a copy that errored.
    pub fn fail_kv_ram_moves(&self, failing: bool) {
        self.with_fake_ram(|ram| ram.failing = failing);
    }

    /// The windowed KV-RAM moves under way.
    pub fn kv_ram_moves(&self) -> usize {
        self.with_fake_ram(|ram| ram.moves.len())
    }

    /// Every windowed KV-RAM move started so far, in order: the request, and
    /// whether it moved off the device.
    pub fn kv_ram_moves_started(&self) -> Vec<(RequestId, bool)> {
        self.with_fake_ram(|ram| ram.started.clone())
    }

    fn with_fake_ram<T>(&self, f: impl FnOnce(&mut FakeRamState) -> T) -> T {
        f(self
            .inner
            .lock()
            .unwrap()
            .ram
            .as_mut()
            .expect("windowed KV-RAM moves need MockCompute::with_windowed_kv_ram"))
    }

    /// Make every disk write from now on fail once its first window moved
    /// (or succeed again, with `false`): a volume that errors mid-file.
    pub fn fail_disk_writes(&self, failing: bool) {
        self.with_fake_disk(|disk| disk.failing_writes = failing);
    }

    /// Make the next read of `blob`'s file fail its check: a torn or corrupt
    /// file, never restored.
    pub fn corrupt_disk_file(&self, blob: DiskBlob) {
        self.with_fake_disk(|disk| {
            disk.corrupt.insert(blob);
        });
    }

    /// The fake disk's committed files and their bytes on the volume.
    pub fn disk_files(&self) -> HashMap<DiskBlob, u64> {
        self.with_fake_disk(|disk| disk.files.iter().map(|(&blob, &(file, _))| (blob, file)).collect())
    }

    /// The spills the fake disk was asked for, in order.
    pub fn disk_spills(&self) -> Vec<(DiskBlob, DiskSource)> {
        self.with_fake_disk(|disk| disk.spills.clone())
    }

    /// The restores the fake disk was asked for, in order.
    pub fn disk_restores(&self) -> Vec<DiskBlob> {
        self.with_fake_disk(|disk| disk.restores.clone())
    }

    /// The files the fake disk was told to delete, in order.
    pub fn disk_discards(&self) -> Vec<DiskBlob> {
        self.with_fake_disk(|disk| disk.discards.clone())
    }

    /// For each `disk_advance` call so far, the blobs that moved a window.
    pub fn disk_windows(&self) -> Vec<Vec<DiskBlob>> {
        self.with_fake_disk(|disk| disk.windows.clone())
    }

    /// Transfers the fake disk has under way.
    pub fn disk_transfers(&self) -> usize {
        self.with_fake_disk(|disk| disk.transfers.len())
    }

    fn with_fake_disk<T>(&self, f: impl FnOnce(&mut FakeDiskState) -> T) -> T {
        f(self
            .inner
            .lock()
            .unwrap()
            .disk
            .as_mut()
            .expect("the fake disk needs MockCompute::with_disk"))
    }

    /// A blob's bytes, as the mock prices them everywhere else.
    fn disk_blob_bytes(&self, blob: DiskBlob) -> u64 {
        match blob {
            DiskBlob::Live(request) => self.live_bytes(request),
            DiskBlob::Checkpoint(publisher) => self.checkpoint_bytes(publisher),
            DiskBlob::Prefix(_, tokens) => self.sections.blob_bytes(tokens),
        }
    }

    /// The bytes the modelled arena's live blobs hold — the figure
    /// `HostTier::used_bytes` must agree with after every step.
    ///
    /// Panics when the mock has no arena: a test that asks this without
    /// [`Self::with_host_arena`] is asserting against nothing.
    pub fn host_arena_used(&self) -> u64 {
        self.inner
            .lock()
            .unwrap()
            .arena
            .as_ref()
            .expect("host_arena_used needs MockCompute::with_host_arena")
            .used()
    }

    /// Make `request`'s token at generation step `step` its EOS: the round
    /// that commits it finishes the request with `Stop`, emitting only the
    /// tokens before it.
    pub fn eos_after(&self, request: RequestId, step: u32) {
        self.inner.lock().unwrap().eos.insert(request, step);
    }

    /// The prefill batches the mock received, in order (each entry is one
    /// `prefill_step` call; its jobs are the batched prefill group).
    pub fn prefill_calls(&self) -> Vec<Vec<PrefillJob>> {
        self.inner.lock().unwrap().prefill_batches.clone()
    }

    /// The decode batches the mock received, in order.
    pub fn decode_calls(&self) -> Vec<Vec<DecodeJob>> {
        self.inner.lock().unwrap().decode_batches.clone()
    }

    /// The publishers whose shared prefix the scheduler released (P4-10,
    /// GitHub #126), in order.
    pub fn released_prefixes(&self) -> Vec<RequestId> {
        self.inner.lock().unwrap().prefixes_released.clone()
    }

    /// The requests the scheduler released through [`Compute::release`], in
    /// order.
    pub fn released_requests(&self) -> Vec<RequestId> {
        self.inner.lock().unwrap().released.clone()
    }

    /// The capturing requests whose retained prompt checkpoint the scheduler
    /// released (GitHub #186), in order.
    pub fn released_checkpoints(&self) -> Vec<RequestId> {
        self.inner.lock().unwrap().checkpoints_released.clone()
    }

    /// The capturing requests whose retained checkpoint the scheduler spilled
    /// to KV-RAM (GitHub #190), in order.
    pub fn spilled_checkpoints(&self) -> Vec<RequestId> {
        self.inner.lock().unwrap().checkpoints_spilled.clone()
    }

    /// Make the backend fail the next spill of `publisher`'s checkpoint, the
    /// way a leaf that could not write the blob does (GitHub #190).
    pub fn fail_spill(&self, publisher: RequestId) {
        self.inner.lock().unwrap().spill_failures.insert(publisher);
    }

    /// The retained prefixes spilled to KV-RAM (GitHub #190), in order.
    pub fn spilled_prefixes(&self) -> Vec<(RequestId, u32)> {
        self.inner.lock().unwrap().prefixes_spilled.clone()
    }

    /// The spilled prefixes brought back onto the device, in order.
    pub fn returned_prefixes(&self) -> Vec<(RequestId, u32)> {
        self.inner.lock().unwrap().prefixes_returned.clone()
    }

    /// The spilled prefixes whose blob was freed, in order.
    pub fn discarded_spilled_prefixes(&self) -> Vec<(RequestId, u32)> {
        self.inner.lock().unwrap().spilled_prefixes_discarded.clone()
    }

    /// Fail the next prefill batch carrying a job for `request`, the way a
    /// leaf error fails the whole call (GitHub #190).
    pub fn fail_prefill(&self, request: RequestId) {
        self.inner.lock().unwrap().prefill_failures.insert(request);
    }

    /// Make the backend decline `request`'s next checkpoint capture (GitHub
    /// #186): the chunk lands normally and reports that nothing was captured,
    /// which is how a real leaf refuses a bet it cannot afford.
    pub fn refuse_capture(&self, request: RequestId) {
        self.inner.lock().unwrap().capture_refusals.insert(request);
    }

    /// Make the backend report that `request`'s next capture lent `pages`
    /// pages to a pages-only link (GitHub #306) -- whatever the scheduler
    /// expected, which is how a leaf whose view of the sequence drifted from
    /// the scheduler's would report it.
    pub fn lend_at_capture(&self, request: RequestId, pages: u32) {
        self.inner.lock().unwrap().capture_lent_pages.insert(request, pages);
    }

    /// Make the backend fail to read `request`'s next attention readout
    /// (GitHub #260): the chunk lands normally and carries no scores, which
    /// is how a real leaf reports keys the layer's attention never
    /// materialized for it to read.
    pub fn refuse_attention(&self, request: RequestId) {
        self.inner.lock().unwrap().attention_refusals.insert(request);
    }

    /// Where the mock's attention map peaks, among `count` keys: the key a
    /// third of the way into the span. Public so a test can say where the
    /// point must land without restating the mock — and a third, rather than
    /// the middle, so a non-square grid tells a row from a column.
    pub fn attention_peak(count: u32) -> usize {
        (count / 3) as usize
    }

    /// Where the mock's head set peaks (GitHub #263): head `i` of the set on
    /// the key `attention_peak(count) + i % 3` — the pointing head's peak
    /// and the two keys after it — moved forward past any `excluded` key,
    /// wrapping at the span's end. Public for the same reason as
    /// [`MockCompute::attention_peak`]: a test says where the extent must
    /// land without restating the mock.
    pub fn attention_set_argmax(count: u32, heads: usize, excluded: &[u32]) -> Vec<u32> {
        let peak = Self::attention_peak(count) as u32;
        (0..heads as u32)
            .map(|i| {
                let mut key = (peak + i % 3) % count;
                while excluded.contains(&key) {
                    key = (key + 1) % count;
                }
                key
            })
            .collect()
    }

    /// The pre-softmax score the mock gives the peak key, and every other
    /// key's: eight nats apart, so the map is as peaked as a real pointing
    /// head's (the region is one cell) without being a delta — its share is
    /// just under one, not one.
    pub const ATTENTION_PEAK_SCORE: f32 = 6.0;
    /// See [`MockCompute::ATTENTION_PEAK_SCORE`].
    pub const ATTENTION_BACKGROUND_SCORE: f32 = -2.0;

    /// The sub-cell offset every head of the mock's set reads (GitHub #264),
    /// in cells. The neighbour scores below are deliberately lopsided, and
    /// lopsided by the same amount for every head, so a test can say where
    /// the extent lands without restating the parabola.
    pub const ATTENTION_SUB_CELL: (f64, f64) = (0.25, -0.25);

    /// The four scores around each head's peak (GitHub #264): the argmax's
    /// left, right, up and down neighbours in the image grid, `None` where
    /// the peak sits on the grid's border and there is no cell beyond it.
    ///
    /// One nat below the peak on one side and three on the other, which the
    /// parabola reads as [`MockCompute::ATTENTION_SUB_CELL`].
    pub fn attention_set_neighbours(argmax: &[u32], count: u32, cols: u32) -> Vec<Option<f32>> {
        let (near, far) = (
            Self::ATTENTION_PEAK_SCORE - 1.0,
            Self::ATTENTION_PEAK_SCORE - 3.0,
        );
        let cols = cols.max(1);
        let rows = count.div_ceil(cols);
        argmax
            .iter()
            .flat_map(|&key| {
                let (col, row) = (key % cols, key / cols);
                [
                    (col > 0).then_some(far),         // left
                    (col + 1 < cols).then_some(near), // right
                    (row > 0).then_some(near),        // up
                    (row + 1 < rows).then_some(far),  // down
                ]
            })
            .collect()
    }

    /// Force `request` to stop after `n` generated tokens, regardless of
    /// its learned `max_tokens` (for driving streams of requests submitted
    /// without a token cap).
    pub fn stop_after(&self, request: RequestId, n: u32) {
        self.inner.lock().unwrap().stops.insert(request, n);
    }

    /// The token this mock would emit for `request` at decode step `step`
    /// (pure function of the seeds — exposed so tests can pin a stream
    /// without running the scheduler).
    pub fn token_for(&self, request: RequestId, step: u32) -> TokenId {
        Self::mix(self.seed, request, 0, step)
    }
}

impl Default for MockCompute {
    fn default() -> Self {
        Self::new()
    }
}

impl Compute for MockCompute {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        let mut g = self.inner.lock().unwrap();
        if jobs.iter().any(|job| g.prefill_failures.remove(&job.request)) {
            return Err(ComputeError::Kernel(-1));
        }
        for job in jobs {
            // Learn the request's limits / seed from its params.
            g.limits.insert(job.request, job.params.max_tokens);
            g.seeds.insert(job.request, job.params.seed);
            g.prefilled.insert(job.request, job.start_position + job.tokens.len() as u32);
        }
        for job in jobs {
            // GitHub #242: a prefill *draws*, and a run's first token is
            // the one it draws. Held until a decode round asks for it.
            if let Some(permitted) = &job.permitted {
                let draw = Self::draw(self.seed, &mut g, job.request, permitted);
                g.pending.insert(job.request, draw);
            }
        }
        g.prefill_batches.push(jobs.to_vec());
        Ok(jobs
            .iter()
            .map(|job| PrefillOutcome {
                encode_micros: 0,
                // GitHub #186: a nominal, deterministic restore cost on the
                // job that claimed a checkpoint, so the request log's
                // `restore_ms` is observable on CPU without inventing a
                // clock. 0 on every other job, as a real backend reports.
                restore_micros: if job.checkpoint.is_some() { 1 } else { 0 },
                // The mock holds no device image, but it *does* answer the
                // question the scheduler is really asking — "did the capture
                // you asked for happen?" — so the retained ledger is
                // exercisable. `capture_failures` makes a backend that
                // declines the bet testable too.
                checkpoint_captured: {
                    let captured = job.capture_checkpoint.is_some()
                        && !g.capture_refusals.remove(&job.request);
                    if captured {
                        let tokens = job.start_position + job.tokens.len() as u32;
                        g.checkpoint_tokens.insert(job.request, tokens);
                    }
                    captured
                },
                checkpoint_lent_pages: match job.capture_checkpoint {
                    Some(_) => g.capture_lent_pages.remove(&job.request).unwrap_or(0),
                    None => 0,
                },
                // GitHub #237 / ADR 0034: the `Compute` seam now carries a
                // second kind of answer, and every CPU-only implementation
                // of it has to produce one or the scheduler's tests stop
                // covering the path (ADR 0006). A job that asked for no
                // readout gets none, exactly as a real backend reports.
                readout: job
                    .readout
                    .as_deref()
                    .map(|answers| Self::readout(self.seed, job.request, answers)),
                // GitHub #260 / ADR 0038: the seam's third answer, which a
                // mock must produce for the same reason — deterministic, one
                // score per key of the span, peaked at a documented key.
                attention: job
                    .attention
                    .as_ref()
                    .filter(|_| !g.attention_refusals.remove(&job.request))
                    .map(|query| Self::attention(query, job.start_position + job.tokens.len() as u32)),
            })
            .collect())
    }

    fn release_prefix(&self, publisher: RequestId, _tokens: u32) {
        self.inner.lock().unwrap().prefixes_released.push(publisher);
    }

    fn blob_identity(&self) -> crate::identity::BlobIdentity {
        self.identity
    }

    fn release_checkpoint(&self, publisher: RequestId) {
        // The device image and, if the checkpoint had been spilled, its
        // KV-RAM blob: the real adapter drops both here.
        self.free_blob(MockBlob::Checkpoint(publisher));
        self.inner
            .lock()
            .unwrap()
            .checkpoints_released
            .push(publisher);
    }

    // GitHub #190: a materialized checkpoint blob is one nominal byte, so
    // `host_capacity_bytes` counts how many spilled checkpoints KV-RAM holds
    // -- or, with `with_sections`, the image and the tokens it covers.
    fn checkpoint_snapshot_size(&self, publisher: RequestId) -> Result<u64, ComputeError> {
        Ok(self.checkpoint_bytes(publisher))
    }

    fn spill_checkpoint(&self, publisher: RequestId) -> Result<u64, ComputeError> {
        if self.inner.lock().unwrap().spill_failures.remove(&publisher) {
            return Err(ComputeError::Kernel(-1));
        }
        // Placed before it is recorded: a spill the arena turns away is one
        // that did not happen, and a test reading `spilled_checkpoints()`
        // must not see it.
        let bytes = self.checkpoint_bytes(publisher);
        if !self.place_blob(MockBlob::Checkpoint(publisher), bytes) {
            return Err(ComputeError::Kernel(NO_HOST_ROOM));
        }
        self.inner.lock().unwrap().checkpoints_spilled.push(publisher);
        Ok(bytes)
    }

    // GitHub #190: a retained prefix's blob is sized the same way.
    fn prefix_snapshot_size(&self, _publisher: RequestId, tokens: u32) -> Result<u64, ComputeError> {
        Ok(self.sections.blob_bytes(tokens))
    }

    fn spill_prefix(&self, publisher: RequestId, tokens: u32) -> Result<u64, ComputeError> {
        // Placed before it is recorded, as in `spill_checkpoint`.
        let bytes = self.sections.blob_bytes(tokens);
        if !self.place_blob(MockBlob::Prefix(publisher, tokens), bytes) {
            return Err(ComputeError::Kernel(NO_HOST_ROOM));
        }
        self.inner.lock().unwrap().prefixes_spilled.push((publisher, tokens));
        Ok(bytes)
    }

    fn restore_prefix(&self, publisher: RequestId, tokens: u32, _slot: u32) -> Result<u64, ComputeError> {
        self.inner.lock().unwrap().prefixes_returned.push((publisher, tokens));
        Ok(1)
    }

    fn discard_spilled_prefix(&self, publisher: RequestId, tokens: u32) {
        self.free_blob(MockBlob::Prefix(publisher, tokens));
        self.inner
            .lock()
            .unwrap()
            .spilled_prefixes_discarded
            .push((publisher, tokens));
    }

    fn release(&self, request: RequestId) {
        self.inner.lock().unwrap().released.push(request);
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        let mut g = self.inner.lock().unwrap();
        g.decode_batches.push(jobs.to_vec());
        Ok(jobs
            .iter()
            .map(|job| {
                // GitHub #242 — a constrained lane, which is a different round
                // entirely: exactly one token, the one drawn a round ago,
                // and no speculation, no EOS and no token cap (the schedule
                // is the budget, and the scheduler owns it).
                if let Some(emitted) = g.pending.remove(&job.request) {
                    if let Some(permitted) = &job.permitted {
                        let next = Self::draw(self.seed, &mut g, job.request, permitted);
                        g.pending.insert(job.request, next);
                    }
                    *g.generated.entry(job.request).or_insert(0) += 1;
                    return DecodeOutcome::constrained_run(
                        vec![emitted.token],
                        vec![emitted.probability],
                    );
                }
                let round = {
                    let rounds = g.rounds.entry(job.request).or_insert(0);
                    *rounds += 1;
                    *rounds - 1
                };
                // A free lane handed a set mid-run (the thinking budget's
                // forced close): the leaf runs it as a plain round — one
                // token, the free draw it already holds — and the set's draw
                // is what the *next* round returns, the same lag as above.
                let length = match (&job.permitted, g.run_lengths.len()) {
                    (Some(_), _) | (None, 0) => 1,
                    (None, n) => g.run_lengths[round % n],
                };
                // An explicit stop_after overrides the learned limit; an
                // unlearned request (no prefill seen) is treated as
                // unbounded.
                let limit = g
                    .stops
                    .get(&job.request)
                    .copied()
                    .or_else(|| g.limits.get(&job.request).copied().flatten());
                let eos = g.eos.get(&job.request).copied();
                let seed = g.seeds.get(&job.request).copied().unwrap_or(0);
                let mut run = Vec::new();
                let mut committed = 0;
                let mut finish = None;
                for _ in 0..length {
                    let step = *g.generated.entry(job.request).or_insert(0);
                    if limit.is_some_and(|n| step >= n) {
                        break;
                    }
                    *g.generated.get_mut(&job.request).unwrap() += 1;
                    committed += 1;
                    if eos == Some(step) {
                        finish = Some(FinishReason::Stop);
                        break;
                    }
                    run.push(Self::mix(self.seed, job.request, seed, step));
                }
                if let Some(permitted) = &job.permitted {
                    if committed > 0 && finish.is_none() {
                        let next = Self::draw(self.seed, &mut g, job.request, permitted);
                        g.pending.insert(job.request, next);
                    }
                }
                if committed == 0 {
                    // Without `eos_after` the mock has no real EOS token —
                    // its stop condition is a token-count cap, reached
                    // before this round committed anything: `Length`.
                    return DecodeOutcome::finished(FinishReason::Length);
                }
                DecodeOutcome {
                    tokens: run,
                    finish,
                    probabilities: Vec::new(),
                    spec: (!g.run_lengths.is_empty())
                        .then(|| SpecCounters::round(length - 1, committed - 1)),
                }
            })
            .collect())
    }

    // core-06 (P4-07, GitHub #125): the mock holds no real GPU sequence, so
    // it has no real snapshot to move — but a host-tier eviction scenario
    // still needs *some* nonzero, deterministic byte cost per request for
    // the byte-budget bookkeeping to be exercisable at all (a `Compute`
    // that always reports 0, the trait's own default, would make a tier of
    // any size always "fit," and no eviction test could ever force a
    // discard). One nominal byte per snapshot, uniform across every
    // request, is exactly what the existing page-based scenarios already
    // assumed before this ticket's byte-budget rewrite: a fixed per-entry
    // cost that scales purely with entry *count*. `with_sections` sizes it
    // by the tokens the sequence holds instead.
    fn snapshot_size(&self, request: RequestId) -> Result<u64, ComputeError> {
        Ok(self.live_bytes(request))
    }

    fn host_blob_fits(&self, bytes: u64) -> bool {
        let g = self.inner.lock().unwrap();
        match g.arena.as_ref() {
            Some(arena) => arena.first_fit(bytes).is_some(),
            None => true,
        }
    }

    fn evict(&self, request: RequestId) -> Result<u64, ComputeError> {
        let bytes = self.live_bytes(request);
        let mut g = self.inner.lock().unwrap();
        if let Some(arena) = g.arena.as_mut() {
            if !arena.place(MockBlob::Live(request), bytes) {
                // What the leaf reports when no span is long enough
                // (`crate::seq::NO_HOST_ROOM`). The scheduler probes with
                // `host_blob_fits` first, so reaching this means a test
                // drove the call directly.
                return Err(ComputeError::Kernel(NO_HOST_ROOM));
            }
        }
        Ok(bytes)
    }

    fn restore(&self, request: RequestId, _context_tokens: u32) -> Result<(), ComputeError> {
        self.free_blob(MockBlob::Live(request));
        Ok(())
    }

    fn discard_snapshot(&self, request: RequestId) {
        self.free_blob(MockBlob::Live(request));
    }

    // GitHub #309: a windowed move takes its span at the start -- the blob
    // is placed then -- and gives it back when a move in lands or a move out
    // fails. Nothing is copied: the mock's streams are pure functions of the
    // request, wherever its state has been.

    fn kv_ram_move_out(&self, request: RequestId) -> Result<(u64, KvRamMove), ComputeError> {
        let windowed = self.inner.lock().unwrap().ram.as_ref().map(|ram| ram.advances);
        let bytes = self.evict(request)?;
        let Some(advances) = windowed else {
            return Ok((bytes, KvRamMove::Done));
        };
        self.with_fake_ram(|ram| {
            ram.moves.push(FakeRamMove { request, out: true, left: advances });
            ram.started.push((request, true));
        });
        Ok((bytes, KvRamMove::Started))
    }

    fn kv_ram_move_in(&self, request: RequestId, context_tokens: u32) -> Result<KvRamMove, ComputeError> {
        let windowed = self.inner.lock().unwrap().ram.as_ref().map(|ram| ram.advances);
        let Some(advances) = windowed else {
            return self.restore(request, context_tokens).map(|()| KvRamMove::Done);
        };
        self.with_fake_ram(|ram| {
            ram.moves.push(FakeRamMove { request, out: false, left: advances });
            ram.started.push((request, false));
        });
        Ok(KvRamMove::Started)
    }

    fn kv_ram_advance(&self) -> Vec<KvRamEvent> {
        let ended: Vec<(FakeRamMove, bool)> = {
            let mut g = self.inner.lock().unwrap();
            let Some(ram) = g.ram.as_mut() else {
                return Vec::new();
            };
            let failing = ram.failing;
            let mut ended = Vec::new();
            ram.moves.retain_mut(|m| {
                m.left -= 1;
                if m.left == 0 {
                    ended.push((*m, failing));
                }
                m.left > 0
            });
            ended
        };
        ended
            .into_iter()
            .map(|(m, failed)| {
                // A move in that landed gives its span back; a move out that
                // failed never filled its own.
                if m.out == failed {
                    self.free_blob(MockBlob::Live(m.request));
                }
                KvRamEvent {
                    request: m.request,
                    outcome: if failed { KvRamOutcome::Failed } else { KvRamOutcome::Landed { micros: 0 } },
                }
            })
            .collect()
    }

    fn kv_ram_abandon(&self, request: RequestId) {
        let abandoned = {
            let mut g = self.inner.lock().unwrap();
            let Some(ram) = g.ram.as_mut() else {
                return;
            };
            let at = ram.moves.iter().position(|m| m.request == request);
            at.map(|at| ram.moves.remove(at))
        };
        // A move out's span goes with it; a move in leaves the snapshot for
        // the caller to discard.
        if abandoned.is_some_and(|m| m.out) {
            self.free_blob(MockBlob::Live(request));
        }
    }

    // Spec vram-budget/03: the fake disk. Files are a byte count apiece, a
    // transfer a countdown of windows; nothing is copied, and nothing about
    // a request's tokens depends on where its state has been -- the mock's
    // streams are pure functions of the request, which is exactly what lets
    // a scenario check that a moved request's output continues unbroken.

    fn disk_fits(&self, bytes: u64) -> bool {
        let g = self.inner.lock().unwrap();
        let Some(disk) = g.disk.as_ref() else {
            return false;
        };
        let held: u64 = disk.files.values().map(|&(file, _)| file).sum::<u64>()
            + disk.transfers.iter().filter(|t| t.from.is_some()).map(|t| t.file_bytes).sum::<u64>();
        held.saturating_add(crate::disk::disk_file_bytes(bytes)) <= disk.shape.room_bytes
    }

    fn disk_spill(&self, blob: DiskBlob, from: DiskSource, _meta: DiskBlobMeta) -> Result<u64, ComputeError> {
        let bytes = self.disk_blob_bytes(blob);
        let mut g = self.inner.lock().unwrap();
        let Some(disk) = g.disk.as_mut() else {
            return Err(ComputeError::Kernel(-1));
        };
        let windows = bytes.div_ceil(disk.shape.window_bytes).max(1);
        disk.transfers.push(FakeTransfer {
            blob,
            from: Some(from),
            into: None,
            file_bytes: crate::disk::disk_file_bytes(bytes),
            blob_bytes: bytes,
            windows_left: windows,
            wait: disk.shape.advances_per_window.saturating_sub(1),
        });
        disk.spills.push((blob, from));
        Ok(bytes)
    }

    fn disk_restore(&self, blob: DiskBlob, into: DiskTarget) -> Result<(), ComputeError> {
        let mut g = self.inner.lock().unwrap();
        let Some(disk) = g.disk.as_mut() else {
            return Err(ComputeError::Kernel(-1));
        };
        let Some(&(file_bytes, blob_bytes)) = disk.files.get(&blob) else {
            return Err(ComputeError::Kernel(-1));
        };
        let windows = blob_bytes.div_ceil(disk.shape.window_bytes).max(1);
        disk.transfers.push(FakeTransfer {
            blob,
            from: None,
            into: Some(into),
            file_bytes,
            blob_bytes,
            windows_left: windows,
            wait: disk.shape.advances_per_window.saturating_sub(1),
        });
        disk.restores.push(blob);
        Ok(())
    }

    fn disk_advance(&self) -> Vec<DiskEvent> {
        let mut g = self.inner.lock().unwrap();
        let Some(disk) = g.disk.as_mut() else {
            return Vec::new();
        };
        let mut moved = Vec::new();
        let mut ended = Vec::new();
        let mut kv_ram_freed = Vec::new();
        let per_window = disk.shape.advances_per_window.saturating_sub(1);
        let mut keep = Vec::with_capacity(disk.transfers.len());
        for mut t in std::mem::take(&mut disk.transfers) {
            if t.wait > 0 {
                t.wait -= 1;
                keep.push(t);
                continue;
            }
            moved.push(t.blob);
            t.windows_left -= 1;
            // A failing volume errors on the first window it is handed.
            if t.from.is_some() && disk.failing_writes {
                ended.push(DiskEvent {
                    blob: t.blob,
                    outcome: DiskOutcome::Failed { op: DiskOp::Write },
                });
                continue;
            }
            if t.windows_left > 0 {
                t.wait = per_window;
                keep.push(t);
                continue;
            }
            let outcome = match t.from {
                Some(from) => {
                    disk.files.insert(t.blob, (t.file_bytes, t.blob_bytes));
                    if from == DiskSource::KvRam {
                        kv_ram_freed.push(t.blob);
                    }
                    DiskOutcome::Spilled { bytes: t.file_bytes }
                }
                None if disk.corrupt.remove(&t.blob) => {
                    disk.files.remove(&t.blob);
                    DiskOutcome::Failed { op: DiskOp::Read }
                }
                None => {
                    // A live blob's file goes once it has landed; a retained
                    // one stays, since a claim never consumes.
                    if matches!(t.blob, DiskBlob::Live(_)) {
                        disk.files.remove(&t.blob);
                    }
                    DiskOutcome::Restored { micros: 1 }
                }
            };
            ended.push(DiskEvent { blob: t.blob, outcome });
        }
        disk.transfers = keep;
        disk.windows.push(moved);
        // The KV-RAM spans of the blobs written from them come back with the
        // commit, as the real tier's do.
        if let Some(arena) = g.arena.as_mut() {
            for blob in kv_ram_freed {
                arena.free(match blob {
                    DiskBlob::Live(request) => MockBlob::Live(request),
                    DiskBlob::Checkpoint(publisher) => MockBlob::Checkpoint(publisher),
                    DiskBlob::Prefix(publisher, tokens) => MockBlob::Prefix(publisher, tokens),
                });
            }
        }
        ended
    }

    fn disk_discard(&self, blob: DiskBlob) {
        let mut g = self.inner.lock().unwrap();
        if let Some(disk) = g.disk.as_mut() {
            disk.transfers.retain(|t| t.blob != blob);
            disk.files.remove(&blob);
            disk.discards.push(blob);
        }
    }

    fn disk_abandon_restore(&self, request: RequestId) {
        let mut g = self.inner.lock().unwrap();
        if let Some(disk) = g.disk.as_mut() {
            disk.transfers.retain(|t| {
                !matches!(t.into, Some(DiskTarget::Sequence { request: r, .. }) if r == request)
            });
        }
    }
}

impl MockCompute {
    /// A spilled checkpoint's blob: the image and the tokens it covers.
    fn checkpoint_bytes(&self, publisher: RequestId) -> u64 {
        let tokens = self.inner.lock().unwrap().checkpoint_tokens.get(&publisher).copied();
        self.sections.blob_bytes(tokens.unwrap_or(0))
    }

    /// A live sequence's snapshot: the image, its prompt and what it
    /// generated.
    fn live_bytes(&self, request: RequestId) -> u64 {
        let g = self.inner.lock().unwrap();
        let tokens = g.prefilled.get(&request).copied().unwrap_or(0)
            + g.generated.get(&request).copied().unwrap_or(0);
        self.sections.blob_bytes(tokens)
    }

    /// Place `blob` of `bytes` in the modelled arena, if there is one.
    /// `false` when it is there and has no span long enough.
    fn place_blob(&self, blob: MockBlob, bytes: u64) -> bool {
        let mut g = self.inner.lock().unwrap();
        match g.arena.as_mut() {
            Some(arena) => arena.place(blob, bytes),
            None => true,
        }
    }

    fn free_blob(&self, blob: MockBlob) {
        if let Some(arena) = self.inner.lock().unwrap().arena.as_mut() {
            arena.free(blob);
        }
    }

    /// How much of the mock's modelled distribution sits *outside* the
    /// answer tokens, in nats. Small and nonzero on purpose: a readout
    /// whose answer mass were exactly 1 would let a caller that forgot to
    /// check the mass pass every CPU test and fail on the card, and the
    /// real measurement is a median 99.8% held by the declared options
    /// (`docs/findings/2026-09-19-typed-option-logit-readout.md`).
    const OUTSIDE_THE_ANSWERS: f64 = 0.002;

    /// The deterministic readout (GitHub #237): a pure function of (mock
    /// seed, request id, answer token id, slot), shaped like a real one
    /// rather than uniform — separated logits with one clear winner, an
    /// answer mass just under 1, and an unrestricted argmax that *is* a
    /// declared answer, which is what the served model does on every row
    /// the finding scored.
    fn readout(seed: u64, request: RequestId, answers: &[TokenId]) -> Readout {
        let logits: Vec<f32> = answers
            .iter()
            .enumerate()
            .map(|(slot, &id)| {
                let mixed = Self::mix(seed, request, u64::from(id), slot as u32);
                // -4 ..= +4 in thousandths: wide enough to separate slots,
                // fine enough that two of one batch practically never tie.
                (f64::from(mixed % 8_000) / 1000.0 - 4.0) as f32
            })
            .collect();
        let answer_lse = log_sum_exp(&logits);
        Readout {
            full_log_sum_exp: if answer_lse.is_finite() {
                answer_lse + Self::OUTSIDE_THE_ANSWERS
            } else {
                0.0
            },
            full_argmax: argmax(&logits).map_or(0, |slot| answers[slot]),
            logits,
        }
    }

    /// Every head's row of a set read in rows (GitHub #275), `[heads]
    /// [count]`: the background everywhere and one peak at
    /// [`MockCompute::attention_peak`], standing `ln(1 + prompt_tokens)`
    /// nats above it — so the rows **sharpen with the prompt** they are read
    /// at the end of. A question whose instruction is longer than its
    /// content-free twin's `N/A` therefore lifts its peak's segment above
    /// the baseline, as a real question does, and every head votes for it.
    pub fn attention_rows(count: u32, heads: usize, prompt_tokens: u32) -> Vec<f32> {
        let peak = Self::attention_peak(count);
        let top = Self::ATTENTION_BACKGROUND_SCORE + (1.0 + prompt_tokens as f32).ln();
        let row: Vec<f32> = (0..count as usize)
            .map(|key| match key == peak {
                true => top,
                false => Self::ATTENTION_BACKGROUND_SCORE,
            })
            .collect();
        (0..heads).flat_map(|_| row.iter().copied()).collect()
    }

    /// The deterministic attention readout (GitHub #260): one score per key,
    /// the peak at [`MockCompute::attention_peak`] — and, for a query naming
    /// an image's head set (GitHub #263), one key per head at
    /// [`MockCompute::attention_set_argmax`], each with the four scores
    /// around it (GitHub #264, [`MockCompute::attention_set_neighbours`]); for
    /// a set read in rows (GitHub #275), every head's row
    /// ([`MockCompute::attention_rows`], read at the end of a
    /// `prompt_tokens`-long prompt) with its argmax and peak.
    fn attention(query: &crate::pointing::AttentionQuery, prompt_tokens: u32) -> crate::pointing::AttentionScores {
        use crate::pointing::SetRead;
        let count = query.key_count;
        let peak = Self::attention_peak(count);
        let mut scores = crate::pointing::AttentionScores::pointing(
            (0..count as usize)
                .map(|key| match key == peak {
                    true => Self::ATTENTION_PEAK_SCORE,
                    false => Self::ATTENTION_BACKGROUND_SCORE,
                })
                .collect::<Vec<f32>>(),
        );
        match query.set.as_ref().map(|set| (set.heads.len(), &set.read)) {
            None => {}
            Some((heads, SetRead::Peaks { excluded, grid_cols })) => {
                let argmax = Self::attention_set_argmax(count, heads, excluded);
                scores.set_neighbours = Some(Self::attention_set_neighbours(&argmax, count, *grid_cols).into());
                scores.set_peak = Some(vec![Self::ATTENTION_PEAK_SCORE; heads].into());
                scores.set_argmax = Some(argmax.into());
            }
            Some((heads, SetRead::Rows)) => {
                let rows = Self::attention_rows(count, heads, prompt_tokens);
                scores.set_argmax = Some(vec![peak as u32; heads].into());
                scores.set_peak = Some(vec![rows[peak]; heads].into());
                scores.set_rows = Some(rows.into());
            }
        }
        scores
    }

    /// The mock's **constrained** draw (GitHub #242): a member of
    /// `permitted`, picked deterministically, with a probability inside it.
    ///
    /// The probability is never 0 and never 1 for a set of more than one
    /// token, for the reason [`MockCompute::OUTSIDE_THE_ANSWERS`] exists: a
    /// mock that reported a perfect draw would let a caller that never looks
    /// at the trace report zero uncertainty and pass, then meet a units
    /// digit at 0.149 on the card
    /// (`docs/findings/2026-09-19-constrained-digit-readout-points.md`). A
    /// set of **one** does report exactly 1, which is not a courtesy but the
    /// arithmetic: a softmax over one logit is 1, and a forced literal
    /// therefore adds nothing to an answer's uncertainty.
    fn draw(
        seed: u64,
        state: &mut Inner,
        request: RequestId,
        permitted: &[TokenId],
    ) -> crate::constrained::Draw {
        let step = {
            let drawn = state.drawn.entry(request).or_insert(0);
            *drawn += 1;
            *drawn - 1
        };
        let request_seed = state.seeds.get(&request).copied().unwrap_or(0);
        let mixed = Self::mix(seed, request, request_seed, step);
        let token = permitted[mixed as usize % permitted.len()];
        let probability = match permitted.len() {
            1 => 1.0,
            // 0.50 ..= 0.999, well clear of both ends.
            _ => 0.5 + (mixed % 500) as f32 / 1000.0,
        };
        crate::constrained::Draw { token, probability }
    }

    /// The deterministic token mix: a pure function of (mock seed, request
    /// id, request seed, step).
    fn mix(seed: u64, request: RequestId, request_seed: u64, step: u32) -> TokenId {
        let mut h = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        h ^= request.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        h ^= request_seed.wrapping_mul(0x1656_67B1_9E37_79F9);
        h ^= (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (h % (u32::MAX as u64)) as u32
    }
}

/// A test-only `Compute` decorator (GitHub #69) that lets a test hold one
/// `decode_step` call open deterministically — a stand-in for the real
/// backend's per-step GPU latency, without a single `sleep()` anywhere
/// (ADR 0006). Wraps any `Compute` (in practice, always a [`MockCompute`]).
pub struct GatedCompute {
    inner: Arc<dyn Compute>,
    armed: AtomicBool,
    entered_tx: SyncSender<()>,
    release_rx: Mutex<Receiver<()>>,
}

/// The test's handle onto a [`GatedCompute`]'s gate.
pub struct GateController {
    entered_rx: Receiver<()>,
    release_tx: SyncSender<()>,
}

/// How long [`GateController::wait_entered`] waits for the armed call. A
/// gated step enters within microseconds; a request that never reaches
/// decode (cancelled first) would otherwise block its CI job until the job
/// itself is killed — one Windows run sat four hours that way.
const GATE_ENTRY_LIMIT: Duration = Duration::from_secs(60);

impl GateController {
    /// Blocks the calling thread until the gated `decode_step` call has
    /// entered the gate — proof that whatever is driving `Compute` (in
    /// production, the model thread) is now stuck inside this call and
    /// cannot do anything else until [`GateController::release`] is called.
    /// Panics after a minute (`GATE_ENTRY_LIMIT`): a gate nobody enters is a
    /// failed test, not a hung one.
    pub fn wait_entered(&self) {
        self.wait_entered_within(GATE_ENTRY_LIMIT);
    }

    fn wait_entered_within(&self, limit: Duration) {
        self.entered_rx.recv_timeout(limit).unwrap_or_else(|e| {
            panic!("the armed decode_step must enter the gate within {limit:?}: {e}")
        });
    }

    /// Releases the held `decode_step` call, letting it complete.
    pub fn release(&self) {
        self.release_tx
            .send(())
            .expect("the armed decode_step must still be waiting to be released");
    }
}

impl GatedCompute {
    /// Wrap `inner`; the gate starts disarmed (`decode_step` passes through
    /// untouched until [`GatedCompute::arm`] is called).
    pub fn new(inner: Arc<dyn Compute>) -> (Arc<Self>, GateController) {
        let (entered_tx, entered_rx) = sync_channel(0);
        let (release_tx, release_rx) = sync_channel(0);
        (
            Arc::new(Self {
                inner,
                armed: AtomicBool::new(false),
                entered_tx,
                release_rx: Mutex::new(release_rx),
            }),
            GateController {
                entered_rx,
                release_tx,
            },
        )
    }

    /// Arms the gate: the next `decode_step` call blocks until
    /// [`GateController::release`] is called. Consumed on entry (one arm =
    /// one held call) — a test re-arms explicitly for each hold it wants.
    pub fn arm(&self) {
        self.armed.store(true, Ordering::SeqCst);
    }
}

impl Compute for GatedCompute {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        self.inner.prefill_step(jobs)
    }

    fn blob_identity(&self) -> crate::identity::BlobIdentity {
        self.inner.blob_identity()
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        if self.armed.swap(false, Ordering::SeqCst) {
            // A rendezvous pair: `send` blocks until `wait_entered`'s `recv`
            // is there to receive it, then this call blocks again until
            // `release` sends the go-ahead.
            let _ = self.entered_tx.send(());
            let _ = self.release_rx.lock().unwrap().recv();
        }
        self.inner.decode_step(jobs)
    }

    fn release(&self, request: RequestId) {
        self.inner.release(request);
    }

    fn snapshot_size(&self, request: RequestId) -> Result<u64, ComputeError> {
        self.inner.snapshot_size(request)
    }

    fn evict(&self, request: RequestId) -> Result<u64, ComputeError> {
        self.inner.evict(request)
    }

    fn restore(&self, request: RequestId, context_tokens: u32) -> Result<(), ComputeError> {
        self.inner.restore(request, context_tokens)
    }

    fn discard_snapshot(&self, request: RequestId) {
        self.inner.discard_snapshot(request);
    }

    fn kv_ram_move_out(&self, request: RequestId) -> Result<(u64, KvRamMove), ComputeError> {
        self.inner.kv_ram_move_out(request)
    }

    fn kv_ram_move_in(&self, request: RequestId, context_tokens: u32) -> Result<KvRamMove, ComputeError> {
        self.inner.kv_ram_move_in(request, context_tokens)
    }

    fn kv_ram_advance(&self) -> Vec<KvRamEvent> {
        self.inner.kv_ram_advance()
    }

    fn kv_ram_abandon(&self, request: RequestId) {
        self.inner.kv_ram_abandon(request);
    }

    fn release_checkpoint(&self, publisher: RequestId) {
        self.inner.release_checkpoint(publisher);
    }

    fn checkpoint_snapshot_size(&self, publisher: RequestId) -> Result<u64, ComputeError> {
        self.inner.checkpoint_snapshot_size(publisher)
    }

    fn spill_checkpoint(&self, publisher: RequestId) -> Result<u64, ComputeError> {
        self.inner.spill_checkpoint(publisher)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A gate nobody enters fails the test instead of hanging it: a request
    /// cancelled before its first decode step never reaches the gate.
    #[test]
    #[should_panic(expected = "must enter the gate within")]
    fn a_gate_nobody_enters_fails_instead_of_hanging() {
        let (gated, controller) = GatedCompute::new(Arc::new(MockCompute::new()));
        gated.arm();
        controller.wait_entered_within(Duration::from_millis(10));
    }
}
