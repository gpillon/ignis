//! GitHub #190's KV-RAM legs on a DFlash2 load: the drafter's window and
//! checkpoint sections travel in the blob too, so a restored sequence drafts
//! from the same window it would have had on the device and generates the
//! same text. Same body as `cuda_leaf_kv_ram_gpu.rs`.
//!
//! Its own test binary: it materializes the artifact, and the card fits one
//! at a time.

#![cfg(feature = "cuda")]

mod kv_ram_gpu_common;

use ignis_core::{Speculation, SpeculativeBackend};

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn retained_state_restored_from_kv_ram_continues_bit_exactly_on_a_dflash2_load() {
    let speculation = Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap();
    let Some(loaded) = kv_ram_gpu_common::load(Some(speculation)) else {
        return;
    };
    kv_ram_gpu_common::an_idle_conversation_resumes_from_kv_ram_exactly(&loaded);
    kv_ram_gpu_common::a_retained_prefix_claimant_evicted_mid_decode_continues_exactly(&loaded);
}
