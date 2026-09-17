//! The KV-RAM arena at the Rust seam (GitHub #213, ADR 0030): what the load
//! does with `--kv-host-pool-bytes` before it serves anything.
//!
//! The arena's own placement — first fit, freeing, the "no hole fits" path —
//! is pinned against the vendored allocator in
//! `kernel/tests/test_seq_snapshot.cpp`, and what the scheduler does with a
//! refusal is pinned on a CPU in `ignis_core::concrete`'s own tests. What is
//! left, and only testable here, is the pair of answers a start can get:
//! a region that pins, and one that does not.
//!
//! Its own test binary rather than a leg of the KV-RAM legs: it pins and
//! releases arenas of its own, and the arena is process-wide.

#![cfg(feature = "cuda")]

use ignis_core::seq::{HostPinnedPool, PinnedAllocError, PinnedBuffer, host_blob_fits, host_pool_stats};

/// Enough to be refused by any host, and not so large that the request
/// overflows before the driver sees it: 8 EiB of page-locked RAM does not
/// exist.
const MORE_THAN_ANY_HOST_HAS: u64 = u64::MAX / 2;

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_region_the_host_cannot_page_lock_refuses_the_start_naming_the_size_and_the_flag() {
    let error = HostPinnedPool::create(MORE_THAN_ANY_HOST_HAS)
        .err()
        .expect("no host page-locks 8 EiB");

    // An operator reading this has to know both what was asked for and what
    // to turn down, or the message is only an apology.
    assert!(
        error.contains(&MORE_THAN_ANY_HOST_HAS.to_string()),
        "the refusal must name the size it could not pin: {error}"
    );
    assert!(
        error.contains("--kv-host-pool-bytes"),
        "the refusal must name the knob that sets it: {error}"
    );

    // And nothing was left behind: the next start is free to pin its own.
    assert_eq!(host_pool_stats(), (0, 0), "a failed pin holds nothing");
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_zero_pool_reserves_nothing_and_a_pinned_one_reports_what_its_blobs_hold() {
    {
        let pool = HostPinnedPool::create(0).expect("0 pins nothing, which cannot fail");
        assert_eq!(pool.capacity_bytes(), 0);
        assert_eq!(host_pool_stats(), (0, 0), "--kv-host-pool-bytes 0 reserves nothing");
        // With the tier off nothing fits, and a blob asked for anyway is
        // refused as a failure rather than as a full arena: there is no
        // arena to give blobs back to, so no victim order would help.
        assert!(!host_blob_fits(4096));
        assert!(matches!(
            PinnedBuffer::new(4096),
            Err(PinnedAllocError::Failed(_))
        ));
    }

    let capacity = 64 * 1024 * 1024;
    let _pool = HostPinnedPool::create(capacity).expect("64 MiB of pinned RAM");
    assert_eq!(host_pool_stats(), (capacity, 0));

    // `used` is what the blobs asked for, not what first fit padded them to:
    // it is the figure `HostTier::used_bytes` is asserted against.
    let blob = PinnedBuffer::new(1_000_003).expect("a fresh arena has room");
    assert_eq!(host_pool_stats(), (capacity, 1_000_003), "used is the bytes asked for");
    assert_eq!(blob.len(), 1_000_003);

    // A blob longer than the whole arena is refused as a full arena would
    // be, not as a broken call -- the caller gives blobs back and asks again.
    assert!(!host_blob_fits(capacity));
    assert!(matches!(PinnedBuffer::new(capacity), Err(PinnedAllocError::NoRoom)));

    drop(blob);
    assert_eq!(host_pool_stats(), (capacity, 0), "a dropped blob goes back to the arena");
    assert!(host_blob_fits(capacity), "and the arena is whole again");
}
