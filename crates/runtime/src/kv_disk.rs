//! KV-disk, Tier 2 — the store (spec vram-budget/03, ADR 0045).
//!
//! The files behind the scheduler's disk ledger (`ignis_core::disk`): where
//! they live, what each one says about itself, and the two threads that
//! write and read them. The transfers that move a blob through them a window
//! at a time are [`crate::RuntimeCompute`]'s; this module knows nothing of
//! sequences or the device.
//!
//! - **One directory a process.** A load writes under
//!   `<location>/ignis-kv-disk/<pid>-<nonce>/` and holds a lock file there,
//!   open and locked, for its life ([`DiskDir`]). At start every directory
//!   whose lock can be taken — its owner is gone — is removed; at a clean
//!   shutdown the process removes its own. Nothing is reused across a
//!   restart (v1), and no file another process wrote is ever read: the store
//!   reads only files it made, by the names it gave them.
//! - **One file a blob**: a 4 KiB header page, written last as the commit,
//!   then the blob, its last window padded to the unbuffered-IO alignment
//!   ([`FileHeader`]). The header names the load (#205: the served model, the
//!   blob identity, the sidecar's payload hash when there is one, the RoPE
//!   scaling) and the blob (its kind, match key, tokens and length), with a
//!   CRC32 per window and one of its own. A file with no header (a torn
//!   write), a byte flipped anywhere, or a header naming another load is
//!   refused ([`Refusal`]) and never restored.
//! - **Two IO threads**, a writer and a reader, each running one positional
//!   unbuffered request at a time ([`IoThreads`]). Neither starts a request
//!   while a prefill's n-gram gather is pending ([`GatherGate`]); a request
//!   already started finishes its window.
//! - **The budget is a ceiling, not a reservation**: `min(--kv-disk-bytes,
//!   volume free - 10 GiB)` at start ([`effective_budget`]), and a write that
//!   would cross the margin is refused ([`DiskStore::fits`]).

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::Duration;

use ignis_artifact::{DirectReader, DirectWriter};
use ignis_core::BlobIdentity;
use ignis_core::compute::ModelFamily;
use ignis_core::ngram_table::GatherGate;
use ignis_core::scheduler::DiskBlob;

/// A window: what one copy crosses PCIe with, and one IO request moves
/// (spec vram-budget/03).
pub const WINDOW_BYTES: u64 = 32 << 20;

/// The pinned staging a device transfer crosses through: two windows, one on
/// the bus while the other is written or read.
pub const STAGING_WINDOWS: usize = 2;

/// The staging's bytes: a host-plan line on Flash-Next, part of the tier's
/// open on the 27B.
pub const STAGING_BYTES: u64 = WINDOW_BYTES * STAGING_WINDOWS as u64;

/// What the tier leaves free on its volume (spec vram-budget/03): a write
/// that would cross it is refused, so a full volume degrades reuse instead of
/// breaking everything else on it.
pub const VOLUME_MARGIN_BYTES: u64 = 10 << 30;

/// A file's header page, written last as the commit (`ignis_core::disk`'s).
pub const HEADER_BYTES: usize = ignis_core::disk::DISK_HEADER_BYTES as usize;

/// The unbuffered-IO alignment every offset, length and buffer obeys.
pub const ALIGNMENT: u64 = ignis_artifact::DIRECT_IO_ALIGNMENT;

/// The directory under a location every process's own directory sits in.
pub const DIR_NAME: &str = "ignis-kv-disk";

const LOCK_NAME: &str = "lock";
const MAGIC: [u8; 8] = *b"IGNKVDK1";
const FORMAT_VERSION: u32 = 1;
/// Where the per-window CRCs start in the header page; the header's own CRC
/// is its last four bytes.
const CRC_TABLE_AT: usize = 512;
const MODEL_ID_MAX: usize = 255;
/// The most windows a header page has room for: 895 x 32 MiB, ~28 GiB.
pub const MAX_WINDOWS: usize = (HEADER_BYTES - CRC_TABLE_AT - 4) / 4;

/// The tier's budget when `--kv-disk-bytes` is unnamed (spec vram-budget/03):
/// 16 GiB on Flash-Next, whose pool is smallest while its experts stream;
/// off on the 27B, whose KV-RAM arena is its last tier unless named.
pub fn default_bytes(family: ModelFamily) -> u64 {
    match family {
        ModelFamily::FlashNext => 16 << 30,
        ModelFamily::Qwen38_27b => 0,
    }
}

/// The fewest tokens a disk restore must save over the tier above it to be
/// taken (spec vram-budget/03): a read of a short prefix costs more than its
/// prefill. Flash-Next's prefill is the slower, so its floor is the lower.
pub fn restore_floor_tokens(family: ModelFamily) -> u32 {
    match family {
        ModelFamily::FlashNext => 8_192,
        ModelFamily::Qwen38_27b => 16_384,
    }
}

/// The tier's effective budget (spec vram-budget/03 AC 14): the flag, or
/// what the volume has free above its margin, whichever is smaller. Zero is
/// no tier.
pub fn effective_budget(flag_bytes: u64, volume_free_bytes: u64, margin_bytes: u64) -> u64 {
    flag_bytes.min(volume_free_bytes.saturating_sub(margin_bytes))
}

/// `bytes` rounded up to the unbuffered-IO alignment.
pub fn aligned(bytes: u64) -> u64 {
    bytes.div_ceil(ALIGNMENT) * ALIGNMENT
}

// ── the load a file is taken under (GitHub #205) ────────────────────────────

/// What every file of a load says about the load, and what a file must say
/// to be restored by it: the served model, the blob identity (ADR 0029), the
/// sidecar's payload SHA-256 when the artifact has one, and the RoPE scaling
/// the KV was rotated with (#205's comment).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskIdentity {
    pub model_id: String,
    pub artifact: [u8; 32],
    /// The KV format's ABI code.
    pub kv_format: i32,
    pub layout_version: u32,
    /// The drafter bound at load, as `backend:draft_tokens:head`, or empty.
    pub drafter: String,
    pub sidecar_sha256: Option<[u8; 32]>,
    /// The RoPE scaling's four parameters, as bits.
    pub rope_scaling: [u32; 4],
}

impl DiskIdentity {
    /// The identity of a load serving `model_id` with `blob`'s state under
    /// `rope_scaling`, whose artifact's sidecar hashed to `sidecar_sha256`.
    pub fn of_load(
        model_id: &str,
        blob: &BlobIdentity,
        rope_scaling: ignis_core::RopeScaling,
        sidecar_sha256: Option<[u8; 32]>,
    ) -> Self {
        Self {
            model_id: model_id.chars().take(MODEL_ID_MAX).collect(),
            artifact: *blob.artifact.as_bytes(),
            kv_format: blob.kv_format.abi_code(),
            layout_version: blob.layout_version,
            drafter: blob.drafter.map_or_else(String::new, |d| {
                format!("{}:{}:{}", d.backend().as_str(), d.draft_tokens(), d.proposal_head().as_str())
            }),
            sidecar_sha256,
            rope_scaling: [
                rope_scaling.factor().to_bits(),
                rope_scaling.temperature().to_bits(),
                rope_scaling.beta_fast().to_bits(),
                rope_scaling.beta_slow().to_bits(),
            ],
        }
    }

    /// The first field in which `self` and `load` differ, or `None`.
    fn differs_from(&self, load: &Self) -> Option<&'static str> {
        if self.model_id != load.model_id {
            Some("model id")
        } else if self.artifact != load.artifact {
            Some("artifact hash")
        } else if self.kv_format != load.kv_format {
            Some("kv format")
        } else if self.layout_version != load.layout_version {
            Some("layout version")
        } else if self.drafter != load.drafter {
            Some("drafter")
        } else if self.sidecar_sha256 != load.sidecar_sha256 {
            Some("sidecar payload hash")
        } else if self.rope_scaling != load.rope_scaling {
            Some("rope scaling")
        } else {
            None
        }
    }
}

/// The payload SHA-256 `<artifact>.sha256` records (GitHub #205: written as
/// `<64 hex> *<file name>`), or `None` when there is no such sidecar or it
/// does not hold one. Reading it costs nothing at start; hashing the
/// artifact would take many seconds.
pub fn sidecar_sha256(artifact: &Path) -> Option<[u8; 32]> {
    let mut name = artifact.as_os_str().to_owned();
    name.push(".sha256");
    let text = std::fs::read_to_string(PathBuf::from(name)).ok()?;
    let hex = text.split_whitespace().next()?;
    if hex.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

// ── the file format ─────────────────────────────────────────────────────────

/// What kind of blob a file holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobKind {
    Live,
    Checkpoint,
    Prefix,
}

impl BlobKind {
    pub fn of(blob: DiskBlob) -> Self {
        match blob {
            DiskBlob::Live(_) => Self::Live,
            DiskBlob::Checkpoint(_) => Self::Checkpoint,
            DiskBlob::Prefix(..) => Self::Prefix,
        }
    }

    fn code(self) -> u8 {
        match self {
            Self::Live => 1,
            Self::Checkpoint => 2,
            Self::Prefix => 3,
        }
    }

    fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Live),
            2 => Some(Self::Checkpoint),
            3 => Some(Self::Prefix),
            _ => None,
        }
    }
}

/// A file's header page: the load it was taken under, the blob it holds, and
/// a CRC32 per window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileHeader {
    pub identity: DiskIdentity,
    pub kind: BlobKind,
    /// The blob's match key (`MatchKey::to_bytes`); the empty key's for a
    /// live sequence.
    pub key: [u8; 16],
    pub tokens: u32,
    /// The blob's true length; the file's body is it padded to the alignment.
    pub blob_bytes: u64,
    pub window_bytes: u64,
    pub crcs: Vec<u32>,
}

/// Why a file is not restored (spec vram-budget/03 AC 15). Never a partial
/// restore: a refusal is found before any byte reaches the device, or, for a
/// window's CRC, before that window does -- and the restore is abandoned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// No header: the write that would have committed it never happened.
    NoHeader,
    /// The header fails its own CRC, or its layout.
    CorruptHeader,
    /// Window `n` fails its CRC.
    CorruptWindow(usize),
    /// The header names another load: this field differs.
    Foreign(&'static str),
    /// The header names another blob than the one asked for.
    NotTheBlob(&'static str),
    /// The volume failed the read.
    Io(String),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::NoHeader => write!(f, "the file has no header (a torn write)"),
            Refusal::CorruptHeader => write!(f, "the file's header fails its check"),
            Refusal::CorruptWindow(n) => write!(f, "window {n} fails its CRC"),
            Refusal::Foreign(field) => write!(f, "the file was taken under another load: its {field} differs"),
            Refusal::NotTheBlob(field) => write!(f, "the file holds another blob: its {field} differs"),
            Refusal::Io(e) => write!(f, "the read failed: {e}"),
        }
    }
}

/// The windows a blob of `blob_bytes` moves in, at `window_bytes` each.
pub fn window_count(blob_bytes: u64, window_bytes: u64) -> usize {
    blob_bytes.div_ceil(window_bytes).max(1) as usize
}

fn put(page: &mut [u8], at: usize, bytes: &[u8]) {
    page[at..at + bytes.len()].copy_from_slice(bytes);
}

fn u32_at(page: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(page[at..at + 4].try_into().expect("four bytes"))
}

fn u64_at(page: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(page[at..at + 8].try_into().expect("eight bytes"))
}

impl FileHeader {
    /// The page: fixed fields at fixed offsets, little-endian, the CRC table
    /// at [`CRC_TABLE_AT`], and the page's own CRC32 in its last four bytes.
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        if self.crcs.len() > MAX_WINDOWS {
            return Err(format!("{} windows do not fit a header page ({MAX_WINDOWS} do)", self.crcs.len()));
        }
        if self.crcs.len() != window_count(self.blob_bytes, self.window_bytes) {
            return Err("a header carries one CRC per window".to_string());
        }
        let id = &self.identity;
        let model = id.model_id.as_bytes();
        if model.len() > MODEL_ID_MAX {
            return Err(format!("a model id of {} bytes does not fit a header page", model.len()));
        }
        let mut page = vec![0u8; HEADER_BYTES];
        put(&mut page, 0, &MAGIC);
        put(&mut page, 8, &FORMAT_VERSION.to_le_bytes());
        put(&mut page, 12, &(HEADER_BYTES as u32).to_le_bytes());
        page[16] = self.kind.code();
        put(&mut page, 20, &self.tokens.to_le_bytes());
        put(&mut page, 24, &self.blob_bytes.to_le_bytes());
        put(&mut page, 32, &self.window_bytes.to_le_bytes());
        put(&mut page, 40, &(self.crcs.len() as u32).to_le_bytes());
        put(&mut page, 48, &self.key);
        put(&mut page, 64, &id.artifact);
        put(&mut page, 96, &id.kv_format.to_le_bytes());
        put(&mut page, 100, &id.layout_version.to_le_bytes());
        page[104] = u8::from(id.sidecar_sha256.is_some());
        put(&mut page, 108, &id.sidecar_sha256.unwrap_or_default());
        for (i, bits) in id.rope_scaling.iter().enumerate() {
            put(&mut page, 140 + 4 * i, &bits.to_le_bytes());
        }
        let drafter = id.drafter.as_bytes();
        if drafter.len() > 63 {
            return Err("a drafter spelling of more than 63 bytes does not fit a header page".to_string());
        }
        page[156] = drafter.len() as u8;
        put(&mut page, 157, drafter);
        page[220] = model.len() as u8;
        put(&mut page, 221, model);
        for (i, crc) in self.crcs.iter().enumerate() {
            put(&mut page, CRC_TABLE_AT + 4 * i, &crc.to_le_bytes());
        }
        let crc = crc32fast::hash(&page[..HEADER_BYTES - 4]);
        put(&mut page, HEADER_BYTES - 4, &crc.to_le_bytes());
        Ok(page)
    }

    /// Read a header page back, refusing a missing or corrupt one.
    pub fn decode(page: &[u8]) -> Result<Self, Refusal> {
        if page.len() < HEADER_BYTES || page[..8] != MAGIC {
            return Err(if page.iter().all(|&b| b == 0) { Refusal::NoHeader } else { Refusal::CorruptHeader });
        }
        if crc32fast::hash(&page[..HEADER_BYTES - 4]) != u32_at(page, HEADER_BYTES - 4)
            || u32_at(page, 8) != FORMAT_VERSION
            || u32_at(page, 12) != HEADER_BYTES as u32
        {
            return Err(Refusal::CorruptHeader);
        }
        let kind = BlobKind::from_code(page[16]).ok_or(Refusal::CorruptHeader)?;
        let blob_bytes = u64_at(page, 24);
        let window_bytes = u64_at(page, 32);
        let windows = u32_at(page, 40) as usize;
        if window_bytes == 0 || windows > MAX_WINDOWS || windows != window_count(blob_bytes, window_bytes) {
            return Err(Refusal::CorruptHeader);
        }
        let text = |at: usize| -> Result<String, Refusal> {
            let len = page[at] as usize;
            String::from_utf8(page[at + 1..at + 1 + len].to_vec()).map_err(|_| Refusal::CorruptHeader)
        };
        let mut rope_scaling = [0u32; 4];
        for (i, bits) in rope_scaling.iter_mut().enumerate() {
            *bits = u32_at(page, 140 + 4 * i);
        }
        Ok(Self {
            identity: DiskIdentity {
                model_id: text(220)?,
                artifact: page[64..96].try_into().expect("32 bytes"),
                kv_format: u32_at(page, 96) as i32,
                layout_version: u32_at(page, 100),
                drafter: text(156)?,
                sidecar_sha256: (page[104] != 0).then(|| page[108..140].try_into().expect("32 bytes")),
                rope_scaling,
            },
            kind,
            key: page[48..64].try_into().expect("16 bytes"),
            tokens: u32_at(page, 20),
            blob_bytes,
            window_bytes,
            crcs: (0..windows).map(|i| u32_at(page, CRC_TABLE_AT + 4 * i)).collect(),
        })
    }

    /// Whether this load may restore the file as `expected`: the same load,
    /// the same blob.
    pub fn check(&self, load: &DiskIdentity, expected: &FileHeader) -> Result<(), Refusal> {
        if let Some(field) = self.identity.differs_from(load) {
            return Err(Refusal::Foreign(field));
        }
        if self.kind != expected.kind {
            return Err(Refusal::NotTheBlob("kind"));
        }
        if self.key != expected.key {
            return Err(Refusal::NotTheBlob("match key"));
        }
        if self.tokens != expected.tokens {
            return Err(Refusal::NotTheBlob("token count"));
        }
        if self.blob_bytes != expected.blob_bytes || self.window_bytes != expected.window_bytes {
            return Err(Refusal::NotTheBlob("length"));
        }
        if self.crcs != expected.crcs {
            return Err(Refusal::NotTheBlob("window CRCs"));
        }
        Ok(())
    }
}

// ── the directory ───────────────────────────────────────────────────────────

/// A load's own directory under a location, locked for the load's life and
/// removed at a clean shutdown (spec vram-budget/03 AC 13).
#[derive(Debug)]
pub struct DiskDir {
    path: PathBuf,
    /// Open and locked for as long as the directory is this process's.
    lock: Option<File>,
}

impl DiskDir {
    /// Make this process's directory under `location`, removing first every
    /// directory there whose lock can be taken -- its owner is gone.
    pub fn open(location: &Path) -> io::Result<Self> {
        let root = location.join(DIR_NAME);
        std::fs::create_dir_all(&root)?;
        let removed = sweep(&root);
        if removed > 0 {
            tracing::info!(
                name: "ignis.kv_disk.stale_removed",
                directories = removed,
                root = %root.display(),
                "removed KV-disk directories whose process is gone"
            );
        }
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
            ^ NEXT.fetch_add(1, Ordering::Relaxed).rotate_left(48);
        let path = root.join(format!("{}-{nonce:016x}", std::process::id()));
        std::fs::create_dir(&path)?;
        let lock = OpenOptions::new().read(true).write(true).create_new(true).open(path.join(LOCK_NAME))?;
        lock.try_lock().map_err(|e| io::Error::other(format!("lock {}: {e}", path.display())))?;
        Ok(Self { path, lock: Some(lock) })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The file `blob` lives in.
    pub fn file(&self, blob: DiskBlob) -> PathBuf {
        self.path.join(match blob {
            DiskBlob::Live(request) => format!("live-{request}.kv"),
            DiskBlob::Checkpoint(publisher) => format!("checkpoint-{publisher}.kv"),
            DiskBlob::Prefix(publisher, tokens) => format!("prefix-{publisher}-{tokens}.kv"),
        })
    }
}

impl Drop for DiskDir {
    fn drop(&mut self) {
        // The lock goes first: a handle still open keeps Windows from
        // removing the directory it sits in.
        drop(self.lock.take());
        if let Err(error) = std::fs::remove_dir_all(&self.path) {
            tracing::warn!(
                name: "ignis.kv_disk.dir_not_removed",
                %error,
                path = %self.path.display(),
                "the KV-disk directory was not removed; the next start removes it"
            );
        }
    }
}

/// Remove every directory under `root` whose lock can be taken, and every
/// one with no lock at all; leave the ones another process holds. Returns how
/// many went.
fn sweep(root: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(root) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let free = match OpenOptions::new().read(true).write(true).open(path.join(LOCK_NAME)) {
            Ok(lock) => lock.try_lock().is_ok(),
            Err(e) => e.kind() == io::ErrorKind::NotFound,
        };
        // The probe's handle is dropped before the removal above, in the
        // match arm, so nothing of this process holds the directory open.
        if free && std::fs::remove_dir_all(&path).is_ok() {
            removed += 1;
        }
    }
    removed
}

// ── the IO threads ──────────────────────────────────────────────────────────

/// Bytes a job reads into or writes from, which the transfer that issued the
/// job keeps alive and untouched until the job is done.
#[derive(Debug, Clone, Copy)]
pub struct Bytes {
    pub ptr: *mut u8,
    pub len: usize,
}

// SAFETY: the issuer keeps the bytes alive and does not touch them until the
// job's ticket reports done; the worker is the only other party.
unsafe impl Send for Bytes {}

/// One request for an IO thread.
pub enum Job {
    /// Write `bytes` at `offset`, after a CRC32 of their first `crc_len` --
    /// copied first from `copy_from` when one is named: a blob whose own
    /// bytes are not on an unbuffered-IO boundary, staged by this thread so
    /// the model thread copies nothing.
    Write { file: Arc<DirectWriter>, offset: u64, bytes: Bytes, crc_len: usize, copy_from: Option<Bytes> },
    /// Read `bytes` at `offset`, then a CRC32 of their first `crc_len`.
    Read { file: Arc<DirectReader>, offset: u64, bytes: Bytes, crc_len: usize },
    /// Delete `path`, after every request queued before it on its thread.
    Delete { path: PathBuf },
}

/// How a job ended: the CRC32 it took, or why it failed. Polled by the model
/// thread, which never waits on it except to let memory go on a cancel.
#[derive(Debug, Clone, Default)]
pub struct Ticket(Arc<(Mutex<Option<Result<u32, String>>>, Condvar)>);

impl Ticket {
    /// `None` while the job is queued or running.
    pub fn poll(&self) -> Option<Result<u32, String>> {
        self.0.0.lock().unwrap().clone()
    }

    /// Block until the job is done.
    pub fn wait(&self) -> Result<u32, String> {
        let (result, done) = &*self.0;
        let mut guard = result.lock().unwrap();
        while guard.is_none() {
            guard = done.wait(guard).unwrap();
        }
        guard.clone().expect("checked above")
    }

    fn set(&self, outcome: Result<u32, String>) {
        let (result, done) = &*self.0;
        *result.lock().unwrap() = Some(outcome);
        done.notify_all();
    }
}

/// The tier's own two threads (spec vram-budget/03: one pool per purpose, so
/// a slow write never holds a gather's thread): a writer and a reader, each
/// one request at a time, in the order they were queued.
pub struct IoThreads {
    writer: Worker,
    reader: Worker,
}

struct Worker {
    jobs: Option<mpsc::Sender<(Job, Ticket)>>,
    handle: Option<JoinHandle<()>>,
}

impl IoThreads {
    /// Start both threads; neither starts a request while `gate` says a
    /// prefill gather is pending.
    pub fn start(gate: GatherGate) -> io::Result<Self> {
        Ok(Self {
            writer: Worker::start("ignis-kv-disk-writer", gate.clone())?,
            reader: Worker::start("ignis-kv-disk-reader", gate)?,
        })
    }

    /// Queue a write or a delete on the writer, a read on the reader.
    pub fn submit(&self, job: Job) -> Ticket {
        let worker = match job {
            Job::Read { .. } => &self.reader,
            Job::Write { .. } | Job::Delete { .. } => &self.writer,
        };
        let ticket = Ticket::default();
        let sent = worker
            .jobs
            .as_ref()
            .is_some_and(|jobs| jobs.send((job, ticket.clone())).is_ok());
        if !sent {
            ticket.set(Err("the KV-disk IO thread has stopped".to_string()));
        }
        ticket
    }
}

impl Worker {
    fn start(name: &str, gate: GatherGate) -> io::Result<Self> {
        let (jobs, queue) = mpsc::channel::<(Job, Ticket)>();
        let handle = std::thread::Builder::new().name(name.to_string()).spawn(move || {
            for (job, ticket) in queue {
                // AC 22: the n-gram table's prefill gathers go first.
                while gate.pending() {
                    std::thread::sleep(Duration::from_micros(100));
                }
                ticket.set(run(job));
            }
        })?;
        Ok(Self {
            jobs: Some(jobs),
            handle: Some(handle),
        })
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Closing the queue ends the thread once it has drained it.
        drop(self.jobs.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn run(job: Job) -> Result<u32, String> {
    match job {
        Job::Write {
            file,
            offset,
            bytes,
            crc_len,
            copy_from,
        } => {
            if let Some(from) = copy_from {
                // SAFETY: see `Bytes`; the two never overlap.
                unsafe { std::ptr::copy_nonoverlapping(from.ptr, bytes.ptr, crc_len.min(from.len)) };
            }
            // SAFETY: see `Bytes`.
            let slice = unsafe { std::slice::from_raw_parts(bytes.ptr, bytes.len) };
            let crc = crc32fast::hash(&slice[..crc_len]);
            file.write_at(offset, slice).map_err(|e| e.to_string())?;
            Ok(crc)
        }
        Job::Read {
            file,
            offset,
            bytes,
            crc_len,
        } => {
            // SAFETY: see `Bytes`.
            let slice = unsafe { std::slice::from_raw_parts_mut(bytes.ptr, bytes.len) };
            let got = file.read_at(offset, slice).map_err(|e| e.to_string())?;
            if got < crc_len {
                return Err(format!("the file ends {got} bytes into a {crc_len}-byte window"));
            }
            Ok(crc32fast::hash(&slice[..crc_len]))
        }
        Job::Delete { path } => match std::fs::remove_file(&path) {
            Ok(()) => Ok(0),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(0),
            Err(e) => Err(format!("delete {}: {e}", path.display())),
        },
    }
}

// ── the store ───────────────────────────────────────────────────────────────

/// A volume's free bytes, as the store asks for them: the real volume's in
/// serving, an injected figure in a test.
pub type FreeBytes = Box<dyn Fn() -> io::Result<u64> + Send + Sync>;

/// The store a load's disk tier writes through: its directory, the load's
/// identity, the IO threads, and the volume's margin.
pub struct DiskStore {
    // The threads before the directory: they drain and stop first, so no
    // request still runs against a file the directory's removal deletes.
    pub io: IoThreads,
    pub dir: DiskDir,
    pub identity: DiskIdentity,
    pub window_bytes: u64,
    budget_bytes: u64,
    margin_bytes: u64,
    free: FreeBytes,
}

impl DiskStore {
    /// Open the tier under `location`: its directory (sweeping the dead
    /// ones'), its threads, and its effective budget, `min(flag, free -
    /// margin)`. `None` with a WARN when that leaves nothing -- the start
    /// goes on without the tier, which is a cache.
    pub fn open(
        location: &Path,
        flag_bytes: u64,
        identity: DiskIdentity,
        gate: GatherGate,
        free: FreeBytes,
    ) -> io::Result<Option<Self>> {
        Self::open_with(location, flag_bytes, identity, gate, free, WINDOW_BYTES, VOLUME_MARGIN_BYTES)
    }

    /// [`DiskStore::open`] for a load: the volume's free space read through
    /// `location` each time it is asked.
    pub fn open_on_volume(
        location: &Path,
        flag_bytes: u64,
        identity: DiskIdentity,
        gate: GatherGate,
    ) -> io::Result<Option<Self>> {
        let at = location.to_path_buf();
        let free: FreeBytes =
            Box::new(move || ignis_artifact::volume_free_bytes(&at).map_err(|e| io::Error::other(e.to_string())));
        Self::open(location, flag_bytes, identity, gate, free)
    }

    /// [`DiskStore::open`] with the window and the margin named (tests).
    pub fn open_with(
        location: &Path,
        flag_bytes: u64,
        identity: DiskIdentity,
        gate: GatherGate,
        free: FreeBytes,
        window_bytes: u64,
        margin_bytes: u64,
    ) -> io::Result<Option<Self>> {
        assert!(window_bytes > 0 && window_bytes % ALIGNMENT == 0, "a window is whole sectors");
        std::fs::create_dir_all(location)?;
        let volume_free = free()?;
        let budget_bytes = effective_budget(flag_bytes, volume_free, margin_bytes);
        if budget_bytes < flag_bytes {
            tracing::warn!(
                name: "ignis.kv_disk.budget_cut",
                flag_bytes,
                budget_bytes,
                volume_free_bytes = volume_free,
                margin_bytes,
                location = %location.display(),
                "the KV-disk budget is cut to the volume's free space above its margin"
            );
        }
        if budget_bytes == 0 {
            tracing::warn!(
                name: "ignis.kv_disk.off",
                location = %location.display(),
                volume_free_bytes = volume_free,
                margin_bytes,
                "the volume has no room above its margin: the KV-disk tier is off"
            );
            return Ok(None);
        }
        let dir = DiskDir::open(location)?;
        let io = IoThreads::start(gate)?;
        tracing::info!(
            name: "ignis.kv_disk.ready",
            directory = %dir.path().display(),
            budget_bytes,
            flag_bytes,
            window_bytes,
            "KV-disk tier ready"
        );
        Ok(Some(Self {
            io,
            dir,
            identity,
            window_bytes,
            budget_bytes,
            margin_bytes,
            free,
        }))
    }

    /// The effective budget the scheduler's ledger is sized to.
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Whether the volume can take a file of `file_bytes` more and keep its
    /// margin, with `in_flight` bytes of files already being written.
    pub fn fits(&self, file_bytes: u64, in_flight: u64) -> bool {
        match (self.free)() {
            Ok(free) => free.saturating_sub(in_flight) >= file_bytes.saturating_add(self.margin_bytes),
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_artifact::AlignedBuffer;

    fn temp_location(tag: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ignis-kv-disk-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    fn identity() -> DiskIdentity {
        DiskIdentity {
            model_id: "qwen3.8-flash-next".to_string(),
            artifact: [7; 32],
            kv_format: 1,
            layout_version: 0x101,
            drafter: String::new(),
            sidecar_sha256: None,
            rope_scaling: [0, 1, 2, 3],
        }
    }

    fn header(blob_bytes: u64, window_bytes: u64) -> FileHeader {
        FileHeader {
            identity: identity(),
            kind: BlobKind::Checkpoint,
            key: [9; 16],
            tokens: 1150,
            blob_bytes,
            window_bytes,
            crcs: (0..window_count(blob_bytes, window_bytes) as u32).map(|i| 0xC0DE_0000 + i).collect(),
        }
    }

    // ── AC 14: the budget ───────────────────────────────────────────────────

    #[test]
    fn the_budget_is_the_flag_or_the_room_above_the_margin_whichever_is_smaller() {
        let gib = 1u64 << 30;
        assert_eq!(effective_budget(16 * gib, 100 * gib, 10 * gib), 16 * gib);
        assert_eq!(effective_budget(16 * gib, 20 * gib, 10 * gib), 10 * gib, "cut to the volume");
        assert_eq!(effective_budget(16 * gib, 10 * gib, 10 * gib), 0, "nothing above the margin");
        assert_eq!(effective_budget(16 * gib, 4 * gib, 10 * gib), 0, "and never negative");
    }

    #[test]
    fn a_volume_without_room_above_its_margin_leaves_the_tier_off_and_the_start_going() {
        let location = temp_location("off");
        let store =
            DiskStore::open(&location, 16 << 30, identity(), GatherGate::default(), Box::new(|| Ok(9 << 30)))
                .unwrap();
        assert!(store.is_none(), "the tier is off, and nothing refused the start");
        assert!(!location.join(DIR_NAME).exists(), "and nothing was created");
        let _ = std::fs::remove_dir_all(&location);
    }

    #[test]
    fn a_write_that_would_cross_the_margin_does_not_fit() {
        let location = temp_location("margin");
        let free = Arc::new(AtomicU64::new(100 << 30));
        let read = Arc::clone(&free);
        let store = DiskStore::open(
            &location,
            16 << 30,
            identity(),
            GatherGate::default(),
            Box::new(move || Ok(read.load(Ordering::Relaxed))),
        )
        .unwrap()
        .unwrap();
        assert_eq!(store.budget_bytes(), 16 << 30);
        assert!(store.fits(1 << 30, 0));
        free.store(11 << 30, Ordering::Relaxed);
        assert!(store.fits(1 << 30, 0), "exactly at the margin still fits");
        assert!(!store.fits(1 << 30, 1), "a byte past it, landing files counted, does not");
        assert!(!store.fits(2 << 30, 0));
        drop(store);
        let _ = std::fs::remove_dir_all(&location);
    }

    // ── AC 15: the file format ──────────────────────────────────────────────

    #[test]
    fn a_header_round_trips_and_carries_its_identity_and_its_blob() {
        let mut h = header(100 << 20, 32 << 20);
        h.identity.sidecar_sha256 = Some([0xAB; 32]);
        h.identity.drafter = "dflash2:3:full".to_string();
        let page = h.encode().unwrap();
        assert_eq!(page.len(), HEADER_BYTES);
        assert_eq!(FileHeader::decode(&page).unwrap(), h);
        assert_eq!(h.crcs.len(), 4, "100 MiB in 32 MiB windows: the last one short");
    }

    #[test]
    fn a_file_with_no_header_is_a_torn_write_and_is_refused() {
        assert_eq!(FileHeader::decode(&vec![0u8; HEADER_BYTES]), Err(Refusal::NoHeader));
    }

    #[test]
    fn a_single_flipped_byte_anywhere_in_the_header_is_refused() {
        let page = header(5 * 4096 + 1, 4096).encode().unwrap();
        for at in 0..HEADER_BYTES {
            let mut flipped = page.clone();
            flipped[at] ^= 0x01;
            assert!(FileHeader::decode(&flipped).is_err(), "byte {at} flipped and accepted");
        }
    }

    #[test]
    fn a_header_naming_another_load_is_refused_field_by_field() {
        let load = identity();
        let expected = header(8192, 4096);
        let ok = FileHeader::decode(&expected.encode().unwrap()).unwrap();
        assert_eq!(ok.check(&load, &expected), Ok(()));
        let variants: Vec<(&str, Box<dyn Fn(&mut DiskIdentity)>)> = vec![
            ("model id", Box::new(|i: &mut DiskIdentity| i.model_id = "qwen3.8-27b".to_string())),
            ("artifact hash", Box::new(|i: &mut DiskIdentity| i.artifact[31] ^= 1)),
            ("kv format", Box::new(|i: &mut DiskIdentity| i.kv_format = 0)),
            ("layout version", Box::new(|i: &mut DiskIdentity| i.layout_version = 5)),
            ("drafter", Box::new(|i: &mut DiskIdentity| i.drafter = "mtp:3:full".to_string())),
            ("sidecar payload hash", Box::new(|i: &mut DiskIdentity| i.sidecar_sha256 = Some([1; 32]))),
            ("rope scaling", Box::new(|i: &mut DiskIdentity| i.rope_scaling[0] = 4.0f32.to_bits())),
        ];
        for (field, change) in variants {
            let mut foreign = expected.clone();
            change(&mut foreign.identity);
            let read = FileHeader::decode(&foreign.encode().unwrap()).unwrap();
            assert_eq!(read.check(&load, &expected), Err(Refusal::Foreign(field)), "{field}");
        }
        let mut other = expected.clone();
        other.key[0] ^= 1;
        let read = FileHeader::decode(&other.encode().unwrap()).unwrap();
        assert_eq!(read.check(&load, &expected), Err(Refusal::NotTheBlob("match key")));
    }

    #[test]
    fn the_sidecar_hash_is_read_from_the_artifacts_sha256_file() {
        let location = temp_location("sidecar");
        std::fs::create_dir_all(&location).unwrap();
        let artifact = location.join("model.ninfer");
        assert_eq!(sidecar_sha256(&artifact), None, "no sidecar: structural identity only");
        let hex = "abb1e120".repeat(8);
        std::fs::write(location.join("model.ninfer.sha256"), format!("{hex} *model.ninfer\n")).unwrap();
        let hash = sidecar_sha256(&artifact).unwrap();
        assert_eq!(&hash[..4], &[0xab, 0xb1, 0xe1, 0x20]);
        std::fs::write(location.join("model.ninfer.sha256"), "not a hash").unwrap();
        assert_eq!(sidecar_sha256(&artifact), None);
        let _ = std::fs::remove_dir_all(&location);
    }

    // ── AC 13: the directory ────────────────────────────────────────────────

    #[test]
    fn a_load_writes_under_its_own_locked_directory_and_removes_it_at_shutdown() {
        let location = temp_location("dir");
        let dir = DiskDir::open(&location).unwrap();
        assert!(dir.path().starts_with(location.join(DIR_NAME)));
        let name = dir.path().file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with(&format!("{}-", std::process::id())), "<pid>-<nonce>: {name}");
        assert!(dir.path().join(LOCK_NAME).exists());
        assert_eq!(dir.file(DiskBlob::Live(7)), dir.path().join("live-7.kv"));
        let path = dir.path().to_path_buf();
        std::fs::write(dir.file(DiskBlob::Checkpoint(3)), b"blob").unwrap();
        drop(dir);
        assert!(!path.exists(), "a clean shutdown removes the process's own directory");
        let _ = std::fs::remove_dir_all(&location);
    }

    #[test]
    fn a_second_process_leaves_the_first_ones_directory_alone_and_removes_a_dead_ones() {
        let location = temp_location("two");
        let first = DiskDir::open(&location).unwrap();
        std::fs::write(first.file(DiskBlob::Live(1)), b"first's").unwrap();
        // A directory whose process died: its lock file is there, unlocked.
        let dead = location.join(DIR_NAME).join("999999-dead");
        std::fs::create_dir_all(&dead).unwrap();
        std::fs::write(dead.join(LOCK_NAME), b"").unwrap();
        std::fs::write(dead.join("live-3.kv"), b"dead's").unwrap();
        // One that never got as far as its lock.
        let unlocked = location.join(DIR_NAME).join("999998-nolock");
        std::fs::create_dir_all(&unlocked).unwrap();

        let second = DiskDir::open(&location).unwrap();
        assert!(first.path().exists(), "the first's directory is untouched");
        assert_eq!(std::fs::read(first.file(DiskBlob::Live(1))).unwrap(), b"first's");
        assert!(!dead.exists(), "a directory whose lock can be taken is removed");
        assert!(!unlocked.exists(), "and so is one with no lock at all");
        assert_ne!(first.path(), second.path());
        drop(second);
        drop(first);
        let _ = std::fs::remove_dir_all(&location);
    }

    // ── the IO threads ──────────────────────────────────────────────────────

    #[test]
    fn the_threads_write_read_and_delete_in_order_with_a_crc_per_request() {
        let location = temp_location("io");
        let dir = DiskDir::open(&location).unwrap();
        let io = IoThreads::start(GatherGate::default()).unwrap();
        let path = dir.file(DiskBlob::Live(1));
        let writer = Arc::new(DirectWriter::create(&path).unwrap());
        let mut window = AlignedBuffer::new(8192).unwrap();
        for (i, b) in window.as_mut_slice().iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let expected = crc32fast::hash(&window.as_slice()[..5000]);
        let bytes = Bytes { ptr: window.as_mut_slice().as_mut_ptr(), len: 8192 };
        let wrote = io.submit(Job::Write { file: Arc::clone(&writer), offset: 4096, bytes, crc_len: 5000, copy_from: None });
        assert_eq!(wrote.wait(), Ok(expected), "the writer takes the CRC of the window's own bytes");
        drop(writer);

        let reader = Arc::new(DirectReader::open(&path).unwrap());
        let mut back = AlignedBuffer::new(8192).unwrap();
        let bytes = Bytes { ptr: back.as_mut_slice().as_mut_ptr(), len: 8192 };
        let read = io.submit(Job::Read { file: reader, offset: 4096, bytes, crc_len: 5000 });
        assert_eq!(read.wait(), Ok(expected));
        assert_eq!(&back.as_slice()[..5000], &window.as_slice()[..5000]);

        let deleted = io.submit(Job::Delete { path: path.clone() });
        assert_eq!(deleted.wait(), Ok(0));
        assert!(!path.exists());
        drop(io);
        drop(dir);
        let _ = std::fs::remove_dir_all(&location);
    }

    #[test]
    fn no_new_request_starts_while_a_prefill_gather_is_pending() {
        // AC 22: the gate is checked before each request; one held keeps
        // every queued request waiting, and lifting it lets them run.
        let location = temp_location("gate");
        std::fs::create_dir_all(&location).unwrap();
        let gate = GatherGate::default();
        let io = IoThreads::start(gate.clone()).unwrap();
        let held = gate.enter();
        let path = location.join("gated.kv");
        let writer = Arc::new(DirectWriter::create(&path).unwrap());
        let mut window = AlignedBuffer::new(4096).unwrap();
        let bytes = Bytes { ptr: window.as_mut_slice().as_mut_ptr(), len: 4096 };
        let ticket = io.submit(Job::Write { file: writer, offset: 0, bytes, crc_len: 4096, copy_from: None });
        std::thread::sleep(Duration::from_millis(50));
        assert!(ticket.poll().is_none(), "the write waits while the gather is pending");
        drop(held);
        assert!(ticket.wait().is_ok(), "and runs once it is not");
        drop(io);
        let _ = std::fs::remove_dir_all(&location);
    }
}
