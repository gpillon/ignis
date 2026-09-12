//! The sequence handle step ABI (ADR 0009, GitHub #55, P1-19).
//!
//! A request owns real device state: KV pages from the leaf's paged KV pool
//! and a GDN slot (recurrent state + conv taps) from its linear-attention
//! state pool (`kernel/include/ignis_seq.h`). [`SeqPool`] builds both pools
//! once; [`SeqPool::alloc`] draws a zeroed [`Seq`] from them and
//! [`Drop`] returns it.
//!
//! A sequence can also leave the GPU and come back (P4-06, GitHub #124, ADR
//! 0024): [`Seq::snapshot`] returns an opaque blob of the size
//! [`Seq::snapshot_bytes`] reports, and [`Seq::restore`] validates that
//! blob's own header before writing any of it back. The layout inside the
//! blob is the leaf's alone — this side asks for a size, holds bytes it does
//! not interpret, and hands them back.
//!
//! `Seq<'a>` borrows the [`SeqPool`] it came from: the borrow checker
//! rejects a pool drop while any sequence drawn from it is still alive,
//! which the flat C ABI cannot enforce on its own (the leaf's pools are
//! freed by `ignis_seq_pool_free` regardless of outstanding sequences).

#![cfg(feature = "cuda")]

use std::ffi::CStr;
use std::marker::PhantomData;
use std::os::raw::c_void;

use crate::compute::ModelConfig;
use crate::kv_format::KvFormat;

pub(crate) mod ffi {
    use std::os::raw::{c_char, c_void};

    /// Opaque sequence-state pool handle (`kernel/include/ignis_seq.h`).
    #[repr(C)]
    pub struct IgnisSeqPool([u8; 1]);

    /// Opaque sequence handle.
    #[repr(C)]
    pub struct IgnisSeq([u8; 1]);

    /// 1:1 with `struct ignis_seq_pool_spec`.
    #[repr(C)]
    pub struct IgnisSeqPoolSpec {
        pub num_kv_heads: u32,
        pub head_dim: u32,
        /// `enum ignis_kv_format` (GitHub #122): what the pool's planes
        /// store, fixed for the life of the load.
        pub kv_format: i32,
        pub kv_page_group_count: u32,
        pub max_context_tokens: u32,
        pub slot_count: u32,
        pub gdn_num_layers: u32,
        pub gdn_conv_channels: u32,
        pub gdn_value_heads: u32,
        pub gdn_head_dim: u32,
        /// The model's vocabulary size (P3-03, GitHub #99): sizes each
        /// slot's presence/frequency penalty count buffer (one int32 per
        /// vocab entry).
        pub vocab: u32,
    }

    /// 1:1 with `struct ignis_seq_pool_stats`.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct IgnisSeqPoolStats {
        pub kv_page_group_count: u32,
        pub kv_entitled_pages: u32,
        pub kv_free_pages: u32,
        pub kv_page_bytes: u64,
        pub logical_page_capacity: u32,
        pub slot_count: u32,
        pub free_slot_count: u32,
        /// `enum ignis_kv_format` — the format the pool actually built
        /// with, read back rather than assumed.
        pub kv_format: i32,
        /// Bytes one sequence-token costs in that format.
        pub kv_bytes_per_token: u64,
        /// Resident sequence-tokens the pool holds — the derived capacity
        /// the byte budget bought (GitHub #122).
        pub kv_token_capacity: u64,
    }

    /// 1:1 with `struct ignis_seq_stats`.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct IgnisSeqStats {
        pub slot: i32,
        pub page_entitlement: u32,
        pub mapped_pages: u32,
        pub token_capacity: u64,
    }

    unsafe extern "C" {
        pub fn ignis_seq_pool_create(
            spec: *const IgnisSeqPoolSpec,
            out_pool: *mut *mut IgnisSeqPool,
        ) -> i32;

        pub fn ignis_seq_pool_stats(
            pool: *const IgnisSeqPool,
            out_stats: *mut IgnisSeqPoolStats,
        ) -> i32;

        pub fn ignis_seq_pool_free(pool: *mut IgnisSeqPool);

        pub fn ignis_seq_alloc(
            pool: *mut IgnisSeqPool,
            context_tokens: u32,
            out_seq: *mut *mut IgnisSeq,
        ) -> i32;

        pub fn ignis_seq_release(pool: *mut IgnisSeqPool, seq: *mut IgnisSeq);

        pub fn ignis_seq_stats(seq: *const IgnisSeq, out_stats: *mut IgnisSeqStats) -> i32;

        pub fn ignis_seq_snapshot_format_version() -> u32;

        pub fn ignis_seq_snapshot_size(
            pool: *const IgnisSeqPool,
            seq: *const IgnisSeq,
            out_bytes: *mut u64,
        ) -> i32;

        pub fn ignis_seq_snapshot(
            pool: *const IgnisSeqPool,
            seq: *const IgnisSeq,
            dst: *mut c_void,
            dst_bytes: u64,
        ) -> i32;

        pub fn ignis_seq_restore(
            pool: *mut IgnisSeqPool,
            seq: *mut IgnisSeq,
            src: *const c_void,
            src_bytes: u64,
        ) -> i32;

        pub fn ignis_seq_last_error() -> *const c_char;
    }

    // Test-only diagnostic seam (`kernel/include/ignis_kv_capture.h`,
    // GitHub #119) -- not part of the public flat C ABI above, and gated
    // behind the non-default `kv-capture` feature (this crate's
    // Cargo.toml): the .cu file does compile into ignis_kernel.lib
    // regardless (kernel/CMakeLists.txt globs every kernel/src/*.cu), but
    // nothing in a production build ever references these symbols unless
    // this feature is on, and it never is in one. See
    // [`super::Seq::capture_kv_rows_for_test`].
    #[cfg(feature = "kv-capture")]
    unsafe extern "C" {
        pub fn ignis_kv_capture_rows(
            pool: *const IgnisSeqPool,
            seq: *const IgnisSeq,
            gqa_layer_ordinal: i32,
            role: i32,
            kv_head: i32,
            first_position: i32,
            row_count: i32,
            head_dim: i32,
            out_rows: *mut u16,
        ) -> i32;

        pub fn ignis_kv_capture_last_error() -> *const c_char;
    }
}

pub use ffi::{IgnisSeqPoolStats, IgnisSeqStats};

/// `IGNIS_SEQ_ERR_NOT_IMPLEMENTED` (`kernel/include/ignis_seq.h`): the
/// leaf's return code for an entry point that is declared but not built
/// yet. No entry point returns it today — P4-06 (GitHub #124) implemented
/// the last two that did.
pub const NOT_IMPLEMENTED: i32 = -2;

/// `IGNIS_SEQ_ERR_NOT_AT_BOUNDARY`: the sequence is mid-chunk, so its state
/// sections are not consistent with one another and there is nothing
/// coherent to capture. The same call succeeds once the in-flight chunk
/// completes, which is what an evicting scheduler waits for.
pub const NOT_AT_BOUNDARY: i32 = -3;

/// `IGNIS_SEQ_ERR_BAD_SNAPSHOT`: the blob is not one this leaf can restore
/// (foreign buffer, stale format version, disagreeing size, or a geometry
/// the target does not match). The target sequence is left untouched, so
/// the right response is to discard the blob and re-prefill.
pub const BAD_SNAPSHOT: i32 = -4;

/// The snapshot blob format version this leaf writes and accepts (ADR
/// 0024). A caller that persists blobs beyond one process records this
/// beside them: a blob written under a different version is refused, never
/// reinterpreted.
pub fn snapshot_format_version() -> u32 {
    unsafe { ffi::ignis_seq_snapshot_format_version() }
}

/// A refused or failed state transfer: the leaf's return code plus the
/// message it set.
///
/// The code is what a caller branches on — [`NOT_AT_BOUNDARY`] means "try
/// again after this chunk", [`BAD_SNAPSHOT`] means "discard this blob" —
/// while the message is what a log or a panic prints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqTransferError {
    /// The leaf's return code: `-1`, [`NOT_AT_BOUNDARY`] or
    /// [`BAD_SNAPSHOT`].
    pub code: i32,
    /// `ignis_seq_last_error` at the moment of the failure.
    pub message: String,
}

impl SeqTransferError {
    /// The sequence was mid-chunk: retry once the chunk completes.
    pub fn is_not_at_boundary(&self) -> bool {
        self.code == NOT_AT_BOUNDARY
    }

    /// The blob was refused: discard it and re-prefill.
    pub fn is_bad_snapshot(&self) -> bool {
        self.code == BAD_SNAPSHOT
    }
}

impl std::fmt::Display for SeqTransferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (rc {})", self.message, self.code)
    }
}

impl std::error::Error for SeqTransferError {}

fn transfer_error(code: i32) -> SeqTransferError {
    SeqTransferError {
        code,
        message: last_error(),
    }
}

fn last_error() -> String {
    let msg = unsafe { CStr::from_ptr(ffi::ignis_seq_last_error()) };
    msg.to_string_lossy().into_owned()
}

#[cfg(feature = "kv-capture")]
fn kv_capture_last_error() -> String {
    let msg = unsafe { CStr::from_ptr(ffi::ignis_kv_capture_last_error()) };
    msg.to_string_lossy().into_owned()
}

/// The geometry a [`SeqPool`] is built from — everything
/// `ignis_seq_pool_create` needs beyond what [`ModelConfig`] already
/// carries.
pub struct SeqPoolBudget {
    /// The KV storage format the pool's planes hold (ADR 0022, GitHub
    /// #122). Fixed for the life of the load: it decides the planes, and so
    /// how many tokens the page count below is worth.
    pub kv_format: KvFormat,
    /// The physical KV page count this pool holds — what a byte budget
    /// bought under `kv_format` ([`crate::kv_format::plan_kv_pool`], whose
    /// arithmetic is the leaf's own `ignis_paged_kv_page_budget`).
    pub kv_page_group_count: u32,
    /// The largest single sequence's KV reservation, in tokens.
    pub max_context_tokens: u32,
    /// Max concurrent sequences (KV block-table rows == GDN slots).
    pub slot_count: u32,
}

/// A device-resident pool of sequence state (paged KV pages + GDN slots).
///
/// Not `Send`/`Sync` (the default for a raw-pointer field): the leaf does
/// no internal locking, so `alloc`/`release` from more than one thread at a
/// time is not defined. A single scheduler thread drives it, matching every
/// other handle in this crate (`Model`, `CudaDevice`).
pub struct SeqPool {
    handle: *mut ffi::IgnisSeqPool,
}

impl SeqPool {
    /// Build the pool from a model's GDN geometry and a caller-sized
    /// budget (P1-19).
    pub fn create(cfg: &ModelConfig, budget: &SeqPoolBudget) -> Result<Self, String> {
        let spec = ffi::IgnisSeqPoolSpec {
            num_kv_heads: cfg.num_kv_heads as u32,
            head_dim: cfg.head_dim as u32,
            kv_format: budget.kv_format.abi_code(),
            kv_page_group_count: budget.kv_page_group_count,
            max_context_tokens: budget.max_context_tokens,
            slot_count: budget.slot_count,
            gdn_num_layers: cfg.gdn_num_layers as u32,
            gdn_conv_channels: cfg.gdn_conv_channels() as u32,
            gdn_value_heads: cfg.gdn_value_heads as u32,
            gdn_head_dim: cfg.gdn_head_dim as u32,
            vocab: cfg.vocab as u32,
        };
        let mut handle: *mut ffi::IgnisSeqPool = std::ptr::null_mut();
        let rc = unsafe { ffi::ignis_seq_pool_create(&spec, &mut handle) };
        if rc != 0 || handle.is_null() {
            return Err(last_error());
        }
        Ok(Self { handle })
    }

    /// Pool-wide geometry + live usage.
    pub fn stats(&self) -> IgnisSeqPoolStats {
        let mut stats = IgnisSeqPoolStats::default();
        let rc = unsafe { ffi::ignis_seq_pool_stats(self.handle, &mut stats) };
        assert_eq!(rc, 0, "ignis_seq_pool_stats: null handle (unreachable — SeqPool always holds one)");
        stats
    }

    /// Reserve a slot: KV pages for `context_tokens` plus its GDN state,
    /// zeroed before return. `Err` (the pool left unchanged) on a bad
    /// argument or exhaustion (no free slot, or not enough free KV pages).
    pub fn alloc(&self, context_tokens: u32) -> Result<Seq<'_>, String> {
        let mut handle: *mut ffi::IgnisSeq = std::ptr::null_mut();
        let rc = unsafe { ffi::ignis_seq_alloc(self.handle, context_tokens, &mut handle) };
        if rc != 0 || handle.is_null() {
            return Err(last_error());
        }
        Ok(Seq {
            handle,
            pool: self.handle,
            _pool: PhantomData,
        })
    }

    /// The raw handle (for the GDN layer ABI, GitHub #58). `pub(crate)`: never
    /// exposed outside this crate (mirrors the handle's C-ABI opacity).
    pub(crate) fn handle(&self) -> *mut ffi::IgnisSeqPool {
        self.handle
    }
}

impl Drop for SeqPool {
    fn drop(&mut self) {
        unsafe { ffi::ignis_seq_pool_free(self.handle) };
    }
}

/// A live sequence: one slot's KV allocation + GDN state, borrowed from the
/// [`SeqPool`] it was allocated from (the lifetime prevents the pool from
/// being dropped first).
pub struct Seq<'a> {
    handle: *mut ffi::IgnisSeq,
    pool: *mut ffi::IgnisSeqPool,
    _pool: PhantomData<&'a SeqPool>,
}

impl Seq<'_> {
    /// This sequence's slot, KV page entitlement/mapping and token
    /// capacity.
    pub fn stats(&self) -> IgnisSeqStats {
        let mut stats = IgnisSeqStats::default();
        let rc = unsafe { ffi::ignis_seq_stats(self.handle, &mut stats) };
        assert_eq!(rc, 0, "ignis_seq_stats: null handle (unreachable — Seq always holds one)");
        stats
    }

    /// Bytes a snapshot of this sequence costs **as it stands now** (ADR
    /// 0024): its written KV history, its GDN recurrent state and conv
    /// taps, its penalty-count row and its progress scalars, plus the
    /// blob's own header.
    ///
    /// It grows with the sequence, so a host tier sizes its buffer per
    /// snapshot rather than once per pool. `Err` with
    /// [`NOT_AT_BOUNDARY`](SeqTransferError::is_not_at_boundary) while the
    /// sequence is mid-chunk.
    pub fn snapshot_bytes(&self) -> Result<u64, SeqTransferError> {
        let mut bytes: u64 = 0;
        let rc = unsafe { ffi::ignis_seq_snapshot_size(self.pool, self.handle, &mut bytes) };
        if rc == 0 {
            Ok(bytes)
        } else {
            Err(transfer_error(rc))
        }
    }

    /// Write this sequence's whole device state into `dst` as an opaque
    /// blob, which must be at least [`Seq::snapshot_bytes`] long.
    ///
    /// Synchronous: it returns with every byte already in `dst`. Pinned
    /// host memory is faster and is what the host tier uses (GitHub #125),
    /// but any host slice works. A snapshot is taken only at a chunk
    /// boundary — mid-chunk the call is refused with [`NOT_AT_BOUNDARY`]
    /// rather than capturing state that would restore into a subtly wrong
    /// sequence.
    pub fn snapshot_into(&self, dst: &mut [u8]) -> Result<(), SeqTransferError> {
        let rc = unsafe {
            ffi::ignis_seq_snapshot(
                self.pool,
                self.handle,
                dst.as_mut_ptr() as *mut c_void,
                dst.len() as u64,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(transfer_error(rc))
        }
    }

    /// [`Seq::snapshot_bytes`] then [`Seq::snapshot_into`] over a freshly
    /// allocated buffer — the convenience shape for a test or a caller that
    /// does not pool its host memory.
    pub fn snapshot(&self) -> Result<Vec<u8>, SeqTransferError> {
        let bytes = self.snapshot_bytes()?;
        let mut blob = vec![0u8; bytes as usize];
        self.snapshot_into(&mut blob)?;
        Ok(blob)
    }

    /// Restore this sequence from a blob [`Seq::snapshot`] produced, whose
    /// length must be exactly the blob's own recorded size.
    ///
    /// Whole-sequence, one call: partial restore is not offered, because
    /// GDN state cannot be recomputed without re-running the prefix a
    /// restore exists to avoid (ADR 0024). Every check runs before any byte
    /// is written, so a refusal
    /// ([`BAD_SNAPSHOT`](SeqTransferError::is_bad_snapshot)) leaves this
    /// sequence exactly as it was.
    pub fn restore(&mut self, src: &[u8]) -> Result<(), SeqTransferError> {
        let rc = unsafe {
            ffi::ignis_seq_restore(
                self.pool,
                self.handle,
                src.as_ptr() as *const c_void,
                src.len() as u64,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(transfer_error(rc))
        }
    }

    /// The raw handle (for the GDN layer ABI, GitHub #58). `pub(crate)`: never
    /// exposed outside this crate (mirrors the handle's C-ABI opacity).
    pub(crate) fn handle(&self) -> *mut ffi::IgnisSeq {
        self.handle
    }

    /// Test-only diagnostic for the #119 hq-e8-2b fixture capture
    /// (`crates/core/tests/hq_kv_fixture_capture_gpu.rs`): reads back
    /// `row_count` consecutive (position, kv_head) rows of one GQA layer's
    /// K or V plane, starting at `first_position`, as their raw BF16 bit
    /// patterns (`kernel/include/ignis_kv_capture.h`).
    ///
    /// Gated behind the non-default `kv-capture` feature (never enabled in
    /// a production build, see this crate's `Cargo.toml`) -- that gate,
    /// not the `pub` visibility below, is what keeps this off the engine's
    /// forward-pass ABI. `pub` rather than `pub(crate)` only because it is
    /// exercised from an integration test in `crates/core/tests/` (outside
    /// this crate's privacy boundary); no production caller uses it, and
    /// it exists only so the committed fixture can be re-captured if the
    /// artifact or the capture prompt ever changes. `gqa_layer_ordinal` is
    /// `Seq`/`ignis_seq_internal.h`'s own 0..kIgnisGqaLayerCount-1 index,
    /// not an absolute backbone layer; `role` is 0 for K, 1 for V;
    /// `head_dim` must match the pool's actual geometry (the leaf
    /// validates it rather than trusting it silently).
    #[cfg(feature = "kv-capture")]
    pub fn capture_kv_rows_for_test(
        &self,
        gqa_layer_ordinal: i32,
        role: i32,
        kv_head: i32,
        first_position: i32,
        row_count: i32,
        head_dim: i32,
    ) -> Result<Vec<u16>, String> {
        let mut out = vec![0u16; (row_count.max(0) as usize) * (head_dim.max(0) as usize)];
        let rc = unsafe {
            ffi::ignis_kv_capture_rows(
                self.pool,
                self.handle,
                gqa_layer_ordinal,
                role,
                kv_head,
                first_position,
                row_count,
                head_dim,
                out.as_mut_ptr(),
            )
        };
        if rc == 0 {
            Ok(out)
        } else {
            Err(kv_capture_last_error())
        }
    }

    /// Detach the compile-time borrow tying this sequence to its pool
    /// (GitHub #61 / P1-25): the production leaf keeps a sequence alive for
    /// a whole request, well past the stack frame that called
    /// [`SeqPool::alloc`], so a lifetime parameter cannot express the
    /// relationship across the `ignis-runtime` crate boundary. The pool's C
    /// handle is a raw pointer either way — `into_static` only removes the
    /// Rust-side borrow-checker tie, not a real one.
    ///
    /// # Safety
    /// The caller must keep the originating [`SeqPool`] alive (not
    /// dropped) for as long as the returned handle exists or is dropped.
    pub unsafe fn into_static(self) -> Seq<'static> {
        let handle = self.handle;
        let pool = self.pool;
        std::mem::forget(self);
        Seq {
            handle,
            pool,
            _pool: PhantomData,
        }
    }
}

impl Drop for Seq<'_> {
    fn drop(&mut self) {
        unsafe { ffi::ignis_seq_release(self.pool, self.handle) };
    }
}

// A `Seq` is moved into the scheduler's `Mutex`-guarded live-sequence map
// (never accessed from more than one thread at a time — the engine drives
// every `Compute` call under its own single-owner lock, mirroring
// `SeqPool`'s documented single-thread-driver contract above) but never
// shared by reference across threads, so only `Send` is asserted.
unsafe impl Send for Seq<'_> {}
