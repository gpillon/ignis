//! The sequence handle step ABI (ADR 0009, GitHub #55, P1-19).
//!
//! A request owns real device state: KV pages from the leaf's paged KV pool
//! and a GDN slot (recurrent state + conv taps) from its linear-attention
//! state pool (`kernel/include/ignis_seq.h`). [`SeqPool`] builds both pools
//! once; [`SeqPool::alloc`] draws a zeroed [`Seq`] from them and
//! [`Drop`] returns it. Snapshot/restore are declared on [`Seq`] but return
//! [`NOT_IMPLEMENTED`] until the KV-RAM host tier (G4).
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

mod ffi {
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
        pub kv_page_group_count: u32,
        pub max_context_tokens: u32,
        pub slot_count: u32,
        pub gdn_num_layers: u32,
        pub gdn_conv_channels: u32,
        pub gdn_value_heads: u32,
        pub gdn_head_dim: u32,
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

        pub fn ignis_seq_snapshot(seq: *const IgnisSeq, dst: *mut c_void, dst_bytes: u64) -> i32;

        pub fn ignis_seq_restore(seq: *mut IgnisSeq, src: *const c_void, src_bytes: u64) -> i32;

        pub fn ignis_seq_last_error() -> *const c_char;
    }
}

pub use ffi::{IgnisSeqPoolStats, IgnisSeqStats};

/// `IGNIS_SEQ_ERR_NOT_IMPLEMENTED` (`kernel/include/ignis_seq.h`): the
/// snapshot/restore entry points' return code until the KV-RAM host tier
/// (G4).
pub const NOT_IMPLEMENTED: i32 = -2;

fn last_error() -> String {
    let msg = unsafe { CStr::from_ptr(ffi::ignis_seq_last_error()) };
    msg.to_string_lossy().into_owned()
}

/// The geometry a [`SeqPool`] is built from — everything
/// `ignis_seq_pool_create` needs beyond what [`ModelConfig`] already
/// carries.
pub struct SeqPoolBudget {
    /// The physical KV page count this pool holds (typically from
    /// [`ignis_artifact::paged_kv_page_budget`] against the VRAM left after
    /// weights).
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
            kv_page_group_count: budget.kv_page_group_count,
            max_context_tokens: budget.max_context_tokens,
            slot_count: budget.slot_count,
            gdn_num_layers: cfg.gdn_num_layers as u32,
            gdn_conv_channels: cfg.gdn_conv_channels() as u32,
            gdn_value_heads: cfg.gdn_value_heads as u32,
            gdn_head_dim: cfg.gdn_head_dim as u32,
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

    /// Not implemented until the KV-RAM host tier (G4): always
    /// `Err(NOT_IMPLEMENTED)`.
    pub fn snapshot(&self, dst: &mut [u8]) -> Result<(), i32> {
        let rc = unsafe {
            ffi::ignis_seq_snapshot(self.handle, dst.as_mut_ptr() as *mut c_void, dst.len() as u64)
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(rc)
        }
    }

    /// Not implemented until the KV-RAM host tier (G4): always
    /// `Err(NOT_IMPLEMENTED)`.
    pub fn restore(&mut self, src: &[u8]) -> Result<(), i32> {
        let rc = unsafe {
            ffi::ignis_seq_restore(self.handle, src.as_ptr() as *const c_void, src.len() as u64)
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(rc)
        }
    }
}

impl Drop for Seq<'_> {
    fn drop(&mut self) {
        unsafe { ffi::ignis_seq_release(self.pool, self.handle) };
    }
}
