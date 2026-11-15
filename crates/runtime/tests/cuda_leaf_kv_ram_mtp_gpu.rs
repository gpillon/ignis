//! GitHub #190's KV-RAM legs on a 27B load that drafts with its MTP head
//! (GitHub #622, spec #618): the head's state -- its layer in every KV page,
//! under hq-e8-2b its residual window and ring words, and the continuation
//! hidden -- travels in the blob, and every holder of a shared prefix's or a
//! checkpoint's last page writes its own copy of it, so a sequence evicted to
//! KV-RAM and brought back, a retained prefix's claimant evicted mid-decode,
//! and a turn resumed from a retained slot each go on drafting and generate
//! the text of a run whose state never moved. Same body as
//! `cuda_leaf_kv_ram_gpu.rs`, driven through the scheduler and the runtime's
//! compute seam; the server still refuses the MTP backend on the 27B (#623).
//!
//! Its own test binary: it materializes the artifact, and the card fits one
//! at a time.

#![cfg(feature = "cuda")]

mod kv_ram_gpu_common;

use ignis_core::{Speculation, SpeculativeBackend};

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn retained_state_restored_from_kv_ram_continues_bit_exactly_on_an_mtp_load() {
    let speculation = Speculation::new(SpeculativeBackend::Mtp, 7).unwrap();
    let Some(loaded) = kv_ram_gpu_common::load(Some(speculation)) else {
        return;
    };
    kv_ram_gpu_common::an_idle_conversation_resumes_from_kv_ram_exactly(&loaded);
    kv_ram_gpu_common::a_retained_prefix_claimant_evicted_mid_decode_continues_exactly(&loaded);
    kv_ram_gpu_common::a_burst_block_brought_back_from_kv_ram_serves_exactly(&loaded);
    kv_ram_gpu_common::a_turn_from_a_retained_slot_generates_what_a_split_cold_prefill_generates(&loaded);
    kv_ram_gpu_common::a_checkpoint_given_up_for_a_slot_resumes_from_kv_ram_exactly(&loaded);
}
