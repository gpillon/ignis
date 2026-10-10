//! An owned KV-RAM arena (GitHub #303, spec flash-next/05): the arena a
//! Flash-Next load pins for its host tier, held by the model instance rather
//! than by the process.
//!
//! What the process-wide arena's tests (`crates/runtime/tests/
//! kv_ram_arena_gpu.rs`) pin for the 27B, plus what only an owned one has:
//! two of them live at once beside the process's, and a blob keeps its arena
//! alive past the last handle the owner held.

#![cfg(feature = "cuda")]

use ignis_core::seq::{AllocCount, AllocKind, HostArena, PinnedAllocError, alloc_count, host_pool_stats};

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn an_owned_arena_places_blobs_and_refuses_what_it_cannot_hold() {
    let capacity = 64 * 1024 * 1024;
    let before = alloc_count(AllocKind::KvRamArena);
    let arena = HostArena::create(capacity).expect("64 MiB of pinned RAM");
    assert_eq!(arena.stats(), (capacity, 0));
    assert_eq!(host_pool_stats(), (0, 0), "the process-wide arena is not this one");

    let blob = arena.alloc(1_000_003).expect("a fresh arena has room");
    assert_eq!(blob.len(), 1_000_003);
    assert_eq!(arena.stats(), (capacity, 1_000_003), "used is the bytes asked for");
    assert!(!arena.fits(capacity));
    assert!(matches!(arena.alloc(capacity), Err(PinnedAllocError::NoRoom)), "full, not broken");
    assert!(!arena.fits(0), "nothing of no bytes");

    drop(blob);
    assert_eq!(arena.stats(), (capacity, 0), "a dropped blob goes back");
    assert!(arena.fits(capacity), "and the arena is whole again");
    drop(arena);
    assert_eq!(
        alloc_count(AllocKind::KvRamArena).since(&before),
        AllocCount { allocs: 1, frees: 1, alloc_bytes: capacity },
        "one pinned region, freed with its owner"
    );
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn two_owned_arenas_live_at_once_and_a_blob_outlives_its_owners_handle() {
    let before = alloc_count(AllocKind::KvRamArena);
    let first = HostArena::create(4 * 1024 * 1024).expect("4 MiB");
    let second = HostArena::create(8 * 1024 * 1024).expect("8 MiB beside it: no singleton");
    let mut blob = first.alloc(4096).expect("room");
    let other = second.alloc(8192).expect("room");
    blob.fill(0x5a);
    assert_eq!((first.stats().1, second.stats().1), (4096, 8192), "each counts its own blobs");

    // The owner lets go first: the blob still holds the region it sits in.
    drop(first);
    assert!(blob.iter().all(|&b| b == 0x5a), "the region is still pinned under the blob");
    assert_eq!(alloc_count(AllocKind::KvRamArena).since(&before).frees, 0, "nothing freed yet");
    drop(blob);
    assert_eq!(alloc_count(AllocKind::KvRamArena).since(&before).frees, 1, "the last holder freed it");
    drop(other);
    drop(second);
    assert_eq!(
        alloc_count(AllocKind::KvRamArena).since(&before),
        AllocCount { allocs: 2, frees: 2, alloc_bytes: 12 * 1024 * 1024 }
    );
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn an_arena_the_host_cannot_page_lock_names_the_size_and_the_flag() {
    let bytes = u64::MAX / 2;
    let error = HostArena::create(bytes).err().expect("no host page-locks 8 EiB");
    assert!(error.contains(&bytes.to_string()), "{error}");
    assert!(error.contains("--reuse-kv-host-pool-bytes"), "{error}");
    assert!(HostArena::create(0).is_err(), "a disabled tier creates no arena");
}
