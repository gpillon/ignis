//! One writer per entry (GitHub #379, spec #374): the server — its Server
//! view's Download and its start — and `ignis-server model download` never
//! fetch the same catalog entry into the same folder at once. Two writers on
//! one `.part` would truncate and append each other's bytes, and each would
//! check only the digest of what it sent itself.
//!
//! The lock is the operating system's, on a file of its own in the models
//! folder, `<artifact>.lock` beside the entry's files: `flock` on Linux and
//! macOS, through [`std::fs::File::try_lock`], and `LockFileEx` on Windows.
//! The kernel holds it for the open file and drops it when the file is
//! closed — when the fetch ends, whichever way, and when the process holding
//! it dies, killed or crashed — so a lock never outlives its holder. A fetch
//! that ends releases it by name before the close: on Linux and macOS a
//! process spawned meanwhile shares the open file until its exec, and the
//! close alone would leave the lock held that long (GitHub #492). The file
//! itself stays, empty, for the next fetch of the entry to lock again:
//! deleting it on the way out would let a taker that opened it just before
//! lock a file no longer in the folder while a third locks a new one under
//! the same name.
//!
//! While it holds the lock, the holder's note is the file's content: who it
//! is and its process id ([`Holder`]), so a fetch refused meanwhile says who
//! holds the entry (story 17). `flock` locks no bytes, so the note reads as
//! any file does; on Windows a locked range cannot be read through another
//! handle, so the lock there is one byte far past the note, never the whole
//! file as std's `try_lock` takes it.

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use super::catalog::CatalogEntry;
use super::transfer::DownloadError;

/// An entry's lock, held until it is dropped.
#[derive(Debug)]
pub struct EntryLock {
    /// The open lock file: closing it releases the lock.
    file: File,
}

/// Who takes an entry's lock: written into the lock file while it holds it,
/// for a fetch refused meanwhile to name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Holder {
    /// A server: its Server view's Download, its start-time fetch, or a
    /// removal.
    Server,
    /// `ignis-server model download`.
    Command,
}

impl Holder {
    /// How a refusal names it: what it is, and its process id.
    fn note(self) -> String {
        let pid = std::process::id();
        match self {
            Self::Server => format!("a server (process {pid})"),
            Self::Command => format!("`ignis-server model download` (process {pid})"),
        }
    }
}

/// How a refusal names a holder whose note it cannot read: one that wrote
/// none yet, an older binary's lock over the whole file, or a file that is
/// not a note.
const UNKNOWN_HOLDER: &str = "a server or `ignis-server model download`";

/// The most of a lock file a refusal reads: a note is one line of a few
/// dozen bytes.
const NOTE_MAX: u64 = 256;

/// Why [`EntryLock::take`] took no lock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotLocked {
    /// Another fetch of the entry into the folder holds it: the entry is
    /// being downloaded.
    Held(DownloadError),
    /// The folder or the lock file could not be opened, for this reason.
    Failed(DownloadError),
}

impl From<NotLocked> for DownloadError {
    fn from(refused: NotLocked) -> Self {
        match refused {
            NotLocked::Held(err) | NotLocked::Failed(err) => err,
        }
    }
}

impl EntryLock {
    /// The file `entry`'s lock is held on under `dir`.
    pub fn path(entry: &CatalogEntry, dir: &Path) -> PathBuf {
        let mut name = entry.artifact_path(dir).into_os_string();
        name.push(".lock");
        PathBuf::from(name)
    }

    /// Take `entry`'s lock under `dir` as `holder`, creating the folder when
    /// it is not there yet; refused at once, never waited for, when another
    /// fetch holds it, naming that one as its note says.
    pub fn take(entry: &CatalogEntry, dir: &Path, holder: Holder) -> Result<Self, NotLocked> {
        let failed = |what: String| NotLocked::Failed(DownloadError(what));
        std::fs::create_dir_all(dir).map_err(|err| failed(format!("cannot create {}: {err}", dir.display())))?;
        let path = Self::path(entry, dir);
        // Read and write: `LockFileEx` locks only a handle opened for one of them.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|err| failed(format!("cannot open the lock {}: {err}", path.display())))?;
        match try_lock(&file) {
            Ok(()) => {
                let mut lock = Self { file };
                // Best effort: a lock whose note could not be written still
                // locks, and its refusals name no holder.
                let _ = lock.write_note(&holder.note());
                Ok(lock)
            }
            Err(TryLockError::WouldBlock) => {
                let who = read_note(&file).unwrap_or_else(|| UNKNOWN_HOLDER.to_owned());
                Err(NotLocked::Held(DownloadError(format!(
                    "{} is already being downloaded into {}, by {who}, which holds {}: one writer per entry, so this one fetched nothing; try again once that download ends",
                    entry.id,
                    dir.display(),
                    path.display()
                ))))
            }
            Err(TryLockError::Error(err)) => Err(failed(format!("cannot lock {}: {err}", path.display()))),
        }
    }

    /// Make `note` the lock file's whole content.
    fn write_note(&mut self, note: &str) -> std::io::Result<()> {
        self.file.set_len(0)?;
        self.file.write_all(note.as_bytes())
    }
}

impl Drop for EntryLock {
    /// The note goes before the lock does: the file stays empty for the next.
    /// Then the lock goes by name, not with the file's close (GitHub #492):
    /// a process spawned meanwhile shares the open description `flock`
    /// belongs to until its exec, so a close alone would leave the entry held
    /// for that while, and a fetch right after refused.
    fn drop(&mut self) {
        let _ = self.file.set_len(0);
        unlock(&self.file);
    }
}

/// Release `file`'s lock on its open description, whoever else holds that
/// description: `flock(LOCK_UN)` through std.
#[cfg(not(windows))]
fn unlock(file: &File) {
    let _ = file.unlock();
}

/// Windows needs no unlock by name: `LockFileEx` locks belong to the handle,
/// which a spawned process never inherits (std opens none inheritable), and
/// closing it releases them.
#[cfg(windows)]
fn unlock(_: &File) {}

/// The holder's note in an open lock file, its first line: `None` when
/// there is none to read.
fn read_note(file: &File) -> Option<String> {
    let mut text = String::new();
    file.take(NOTE_MAX).read_to_string(&mut text).ok()?;
    let note = text.lines().next()?.trim();
    (!note.is_empty()).then(|| note.to_owned())
}

/// Lock `file`, refused at once when another open file holds it: `flock`
/// through std, which locks no bytes a reader needs.
#[cfg(not(windows))]
fn try_lock(file: &File) -> Result<(), TryLockError> {
    file.try_lock()
}

/// The byte a lock file is locked on, on Windows: far past any note, which a
/// refused fetch then reads through its own handle.
#[cfg(windows)]
const LOCKED_BYTE: u64 = 1 << 40;

/// Lock `file`, refused at once when another open file holds it:
/// `LockFileEx` on [`LOCKED_BYTE`] alone, exclusive, failing immediately.
#[cfg(windows)]
fn try_lock(file: &File) -> Result<(), TryLockError> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::ERROR_LOCK_VIOLATION;
    use windows_sys::Win32::Storage::FileSystem::{LockFileEx, LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY};
    use windows_sys::Win32::System::IO::{OVERLAPPED, OVERLAPPED_0, OVERLAPPED_0_0};

    let at = OVERLAPPED_0_0 { Offset: LOCKED_BYTE as u32, OffsetHigh: (LOCKED_BYTE >> 32) as u32 };
    let mut overlapped = OVERLAPPED { Anonymous: OVERLAPPED_0 { Anonymous: at }, ..Default::default() };
    // SAFETY: the handle is `file`'s, open for the length of the call, and
    // `overlapped` outlives it: on a handle opened without
    // FILE_FLAG_OVERLAPPED, as std opens one, the call returns only once it
    // has locked or refused.
    let locked = unsafe { LockFileEx(file.as_raw_handle(), LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &mut overlapped) };
    if locked != 0 {
        return Ok(());
    }
    let err = std::io::Error::last_os_error();
    if err.raw_os_error() == Some(ERROR_LOCK_VIOLATION as i32) {
        return Err(TryLockError::WouldBlock);
    }
    Err(TryLockError::Error(err))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry() -> CatalogEntry {
        crate::download::Catalog::built_in().entry("qwen3-asr-0.6b").expect("a built-in entry").clone()
    }

    fn dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ignis-entry-lock-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// Story 17: a fetch refused for the lock names who holds it — the
    /// command or a server, and its process id — from the note the holder
    /// wrote; the note goes with the lock, and the next taker writes its own.
    #[test]
    fn a_refused_taker_names_the_holder_and_its_process() {
        let (entry, dir) = (entry(), dir("holder"));
        let pid = std::process::id();
        let held = EntryLock::take(&entry, &dir, Holder::Command).expect("free");
        let Err(NotLocked::Held(err)) = EntryLock::take(&entry, &dir, Holder::Server) else { panic!("held by the command") };
        assert!(
            err.0.starts_with(&format!("qwen3-asr-0.6b is already being downloaded into {}, by `ignis-server model download` (process {pid}), which holds", dir.display())),
            "{err}"
        );
        drop(held);
        assert_eq!(std::fs::read(EntryLock::path(&entry, &dir)).unwrap(), b"", "the note went with the lock");

        let held = EntryLock::take(&entry, &dir, Holder::Server).expect("free again");
        let Err(NotLocked::Held(err)) = EntryLock::take(&entry, &dir, Holder::Command) else { panic!("held by the server") };
        assert!(err.0.contains(&format!("by a server (process {pid}), which holds")), "{err}");
        drop(held);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A holder that wrote no note (an older binary's, or one between its
    /// lock and its note) is named as either.
    #[test]
    fn a_holder_with_no_note_is_named_as_either() {
        let (entry, dir) = (entry(), dir("no-note"));
        let held = EntryLock::take(&entry, &dir, Holder::Server).expect("free");
        held.file.set_len(0).unwrap();
        let Err(NotLocked::Held(err)) = EntryLock::take(&entry, &dir, Holder::Command) else { panic!("held") };
        assert!(err.0.contains(&format!("by {UNKNOWN_HOLDER}, which holds")), "{err}");
        drop(held);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
