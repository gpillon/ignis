//! The KV pools the VRAM-plan GPU tests pin (GitHub #607), on the CPU: each
//! pinned load's shape, beside the card's lines, gives the pool its test
//! pinned before the context was measured (GitHub #604) when the context is
//! charged at `CUDA_CONTEXT_BYTES`, and the pool main's plans give at the
//! context the RTX 5090 measured under Windows on 2026-11-14, 452,161,536
//! bytes. The trellis workspaces are the RTX 5090's, as its plans logged them.

#[path = "support/kv_pool_pins.rs"]
mod kv_pool_pins;

use ignis_runtime::CUDA_CONTEXT_BYTES;
use kv_pool_pins::{CardLines, RTX50_16GB, RTX50_16GB_LESS_1_GIB, RTX50_16GB_LESS_3_GIB, RTX50_16GB_LESS_3_GIB_131072_SPARE, RTX_16GB_AT_THE_CHUNK, RTX_24GB_AT_THE_CHUNK};

/// The context the RTX 5090 measured under Windows on 2026-11-14.
const MEASURED: u64 = 452_161_536;

fn on_the_5090(cuda_context: u64, trellis_workspace: u64) -> CardLines {
    CardLines { cuda_context, trellis_workspace }
}

#[test]
fn rtx_24gb_at_the_chunk_keeps_508s_pool_at_the_constant_and_one_page_more_at_the_measured_context() {
    let trellis = 64_252_160;
    assert_eq!(RTX_24GB_AT_THE_CHUNK.pool_tokens(on_the_5090(CUDA_CONTEXT_BYTES, trellis)), 1_139_776, "#508's pool");
    assert_eq!(RTX_24GB_AT_THE_CHUNK.pool_tokens(on_the_5090(MEASURED, trellis)), 1_139_840, "main's plan");
}

#[test]
fn rtx_16gb_at_the_chunk_keeps_its_profile_notes_pool_at_both_contexts() {
    let trellis = 32_134_400;
    for context in [CUDA_CONTEXT_BYTES, MEASURED] {
        assert_eq!(RTX_16GB_AT_THE_CHUNK.pool_tokens(on_the_5090(context, trellis)), 386_048, "{context}");
    }
}

#[test]
fn rtx50_16gb_keeps_339s_pool_at_the_constant_and_one_page_more_at_the_measured_context() {
    let trellis = 36_225_280;
    assert_eq!(RTX50_16GB.pool_tokens(on_the_5090(CUDA_CONTEXT_BYTES, trellis)), 509_184, "#339 A1's 7,956 pages");
    assert_eq!(RTX50_16GB.pool_tokens(on_the_5090(MEASURED, trellis)), 509_248, "main's plan");
}

#[test]
fn rtx50_16gb_less_1_gib_keeps_340s_pool_at_the_constant_and_one_page_more_at_the_measured_context() {
    let trellis = 36_225_280;
    assert_eq!(RTX50_16GB_LESS_1_GIB.pool_tokens(on_the_5090(CUDA_CONTEXT_BYTES, trellis)), 392_704, "#340's pool");
    assert_eq!(RTX50_16GB_LESS_1_GIB.pool_tokens(on_the_5090(MEASURED, trellis)), 392_768, "main's plan");
}

#[test]
fn rtx50_16gb_less_3_gib_adapts_to_65536_on_340s_pool_at_the_constant_and_one_page_more_at_the_measured_context() {
    let trellis = 36_225_280;
    assert_eq!(RTX50_16GB_LESS_3_GIB.pool_tokens(on_the_5090(CUDA_CONTEXT_BYTES, trellis)), 159_680, "#340's pool");
    assert_eq!(RTX50_16GB_LESS_3_GIB.pool_tokens(on_the_5090(MEASURED, trellis)), 159_744, "main's plan");
}

#[test]
fn rtx50_16gb_less_3_gib_misses_131072_and_the_spare_reservation_by_577_pages_at_the_constant_and_576_at_the_measured_context() {
    let trellis = 36_225_280;
    assert_eq!(RTX50_16GB_LESS_3_GIB_131072_SPARE.pages_short(on_the_5090(CUDA_CONTEXT_BYTES, trellis)), 577, "#340's");
    assert_eq!(RTX50_16GB_LESS_3_GIB_131072_SPARE.pages_short(on_the_5090(MEASURED, trellis)), 576, "main's refusal");
}

/// A context above the constant, as the WSL podman machine measured one
/// (519,270,400 bytes, #604): 66,674,688 bytes more, 113.04 pages, so every
/// pool is 113 or 114 pages smaller than at the constant.
#[test]
fn a_context_above_the_constant_costs_its_whole_pages() {
    const WSL: u64 = 519_270_400;
    let loads = [
        (RTX_24GB_AT_THE_CHUNK, 64_252_160),
        (RTX_16GB_AT_THE_CHUNK, 32_134_400),
        (RTX50_16GB, 36_225_280),
        (RTX50_16GB_LESS_1_GIB, 36_225_280),
        (RTX50_16GB_LESS_3_GIB, 36_225_280),
    ];
    for (load, trellis) in loads {
        let fewer = load.pool_pages(on_the_5090(CUDA_CONTEXT_BYTES, trellis)) - load.pool_pages(on_the_5090(WSL, trellis));
        assert!((113..=114).contains(&fewer), "{load:?}: {fewer} pages");
    }
    let refusal = RTX50_16GB_LESS_3_GIB_131072_SPARE;
    let more = refusal.pages_short(on_the_5090(WSL, 36_225_280)) - refusal.pages_short(on_the_5090(CUDA_CONTEXT_BYTES, 36_225_280));
    assert!((113..=114).contains(&more), "{more} pages");
}
