//! GPU legs of GitHub #190 on a plain load: retained state spilled to KV-RAM
//! comes back bit-exactly — a prompt checkpoint restored into a fresh
//! sequence, and a retained-prefix claimant snapshotted mid-decode with the
//! prefix's pages materialized into its blob. The shared body is
//! `kv_ram_gpu_common`; `cuda_leaf_kv_ram_dflash2_gpu.rs` runs it with the
//! drafter's sections carried too.
//!
//! Its own test binary: it materializes the artifact, and the card fits one
//! at a time.

#![cfg(feature = "cuda")]

mod kv_ram_gpu_common;

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn retained_state_restored_from_kv_ram_continues_bit_exactly() {
    let Some(loaded) = kv_ram_gpu_common::load(None) else {
        return;
    };
    kv_ram_gpu_common::an_idle_conversation_resumes_from_kv_ram_exactly(&loaded);
    kv_ram_gpu_common::a_retained_prefix_claimant_evicted_mid_decode_continues_exactly(&loaded);
}
