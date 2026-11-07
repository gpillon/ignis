//! An entry's lock across a spawn (GitHub #492), in a test binary of its own:
//! the child here holds every descriptor this process has open for a second,
//! which in a binary of other tests would hold theirs too.
//!
//! A process spawned while the lock is held shares the lock file's open
//! description from its fork until its exec closes its copy (std opens every
//! file close-on-exec), and `flock` belongs to that description. A lock
//! dropped in that window, as a fetch ending while another thread spawns, is
//! free at once all the same: the drop unlocks it rather than leaving it to
//! the last descriptor's close. Before it did, the fetch right after it was
//! refused as the entry being downloaded, and `the_token_never_appears_in_a_
//! progress_line` failed so whenever `model_download.rs`'s cross-process test
//! spawned `ignis-server` at the moment its first fetch ended. Windows locks a
//! handle no child inherits, so the window is Linux's and macOS's alone.
#![cfg(unix)]

use std::fs::File;
use std::io::{Read, Write};
use std::mem::ManuallyDrop;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::time::Duration;

use ignis_server::download::{Catalog, EntryLock, Holder};

#[test]
fn a_dropped_lock_is_free_while_a_child_spawned_under_it_has_yet_to_exec() {
    let entry = Catalog::built_in().entry("qwen3-asr-0.6b").expect("a built-in entry").clone();
    let dir: PathBuf = std::env::temp_dir().join(format!("ignis-entry-lock-spawn-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let held = EntryLock::take(&entry, &dir, Holder::Server).expect("free");
    // The child says it has forked, one byte down a pipe, then waits in the
    // window: forked, holding the lock's description, not yet exec'd.
    let (mut forked, signal) = std::io::pipe().expect("a pipe");
    let signal_fd = signal.as_raw_fd();
    // `spawn` returns once the child has exec'd, so the window is waited out
    // on a thread of its own while this one drops and takes the lock.
    let spawner = std::thread::spawn(move || {
        let mut command = std::process::Command::new("true");
        // SAFETY: the closure runs in the forked child before its exec. It
        // writes one byte to the pipe's write end, open in the child until
        // its exec, through a `File` it never closes, then sleeps
        // (write and nanosleep, async-signal-safe); it allocates nothing.
        unsafe {
            command.pre_exec(move || {
                let mut signal = ManuallyDrop::new(File::from_raw_fd(signal_fd));
                signal.write_all(b"f")?;
                std::thread::sleep(Duration::from_millis(1000));
                Ok(())
            });
        }
        command.spawn().expect("spawn").wait().expect("wait");
        drop(signal);
    });
    forked.read_exact(&mut [0u8; 1]).expect("the child forked");
    drop(held);
    let taken = EntryLock::take(&entry, &dir, Holder::Command);
    spawner.join().expect("the spawner");
    assert!(taken.is_ok(), "the dropped lock is still held: {:?}", taken.as_ref().err());
    drop(taken);
    let _ = std::fs::remove_dir_all(&dir);
}
