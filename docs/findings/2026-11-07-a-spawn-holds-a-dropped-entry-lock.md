# A process spawned while an entry's lock is held keeps it past its drop, until the child's exec

- Kind: experiment
- Status: current
- Observed: 2026-11-07
- Last verified: 2026-11-07
- Scope: server / model download, the entry lock (`crates/server/src/download/lock.rs`); test harness / Linux
  flakes in `model_download.rs` and `responses_socket.rs`
- Topic: harness-and-build
- Related: [GitHub #492](https://github.com/gpillon/ignis/issues/492), [GitHub #379](https://github.com/gpillon/ignis/issues/379)
- Superseded by: none
- Summary: On Linux an entry lock released only by closing its file stays held after its drop while a process forked meanwhile has not yet exec'd, as `flock`'s semantics imply, and an explicit unlock frees it at once (`model_download` 100 of 100).

## Question

On Linux, `the_token_never_appears_in_a_progress_line` (`crates/server/tests/model_download.rs`) failed once in a
full workspace run: its resuming fetch found the entry's lock held right after the interrupted fetch before it had
returned and dropped its `EntryLock`. The guess in #492: a child process spawned by another test of the same binary
(`the_lock_holds_across_processes_and_dies_with_the_process_holding_it` spawns `ignis-server`) shares the lock file's
open description from its fork to its exec, and `flock` belongs to that description, so a lock dropped in that
window outlives its drop. Is that what holds it?

## Evidence

A Linux container (`rust:1-bookworm`, rustc 1.99.1, 12 of 20 cores), the tree of `main` at `a5132fb5` with the test
below. The test takes an `EntryLock`; a second thread spawns `true` with a `pre_exec` hook that sleeps 1 s, so the
child sits forked and not yet exec'd; 250 ms in, the first thread drops the lock and takes it again at once.

- **Before the fix** (the lock released by the file's close alone): the take is refused, `qwen3-asr-0.6b is already
  being downloaded into ..., by a server or ``ignis-server model download``, which holds ...`, in its own test binary
  and in the library's.
- In the library's test binary the same run also failed a sibling, `a_refused_taker_names_the_holder_and_its_process`,
  at `free again`: its lock was dropped on another test thread while this test's child was forked, so it stayed
  held for it. That is #492's flake, one test's spawn holding another test's lock, reproduced on demand.
- **After the fix** (`EntryLock`'s drop calls `File::unlock`, `flock(LOCK_UN)`, before the close): the take
  succeeds; `model_download` passed 100 of 100 runs and `pings_do_not_keep_an_idle_socket_open` 100 of 100, each
  pinned to 4 cores with `taskset -c 0-3`, the CI runner's width.

Windows needs nothing: `LockFileEx` locks belong to the handle, std opens no handle inheritable, and the flake was
never seen there.

## Finding

The guess holds. Observed, on Linux: a lock released only by closing its file stays held after its holder's drop
while a process spawned meanwhile, by another thread, has forked and not yet exec'd; released with `flock(LOCK_UN)`
first, it is free at once in the same window.

Inferred from `flock`'s documented semantics, not measured: the lock belongs to the open file description, which is
released only by an unlock or by the close of its last descriptor, and every process forked while the lock is held
carries a descriptor of it until its exec closes it (std opens files close-on-exec). The window is therefore as long
as the spawn takes to exec: short for `posix_spawn`, which std uses without a `pre_exec` hook, and as long as the
hook with one. macOS documents the same semantics for `flock`.

## Implications

- Any lock or file whose release the code reads from a close (a lock file, a pipe's end, a socket) can be held past
  its drop by a concurrent spawn on Unix. Release by name where the next holder may come at once.
- A test that spawns a process holds every descriptor of its test binary open for the spawn's window. A test whose
  spawn waits in that window (a `pre_exec` hook) belongs in a binary of its own: `crates/server/tests/entry_lock_spawn.rs`
  is one for that reason.

## Limits and unknowns

- The original flake's own rate before the fix was not measured: it needs a spawn to land inside the microseconds
  between a fetch's last write and its drop, and the reproduction above settles the mechanism without it.
- macOS was not run; its `flock` has the same open-description semantics.

## Follow-ups

None.
