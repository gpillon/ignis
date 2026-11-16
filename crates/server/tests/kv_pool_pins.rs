//! The KV pools the VRAM-plan GPU tests pin (GitHub #607), on the CPU: each
//! pinned load's shape, beside the card's lines, gives the pool its test
//! pinned before the context was measured (GitHub #604) when the context is
//! charged at `CUDA_CONTEXT_BYTES`, and the pool main's plans give at the
//! context the RTX 5090 measured under Windows on 2026-11-14, 452,161,536
//! bytes. The trellis workspaces are the RTX 5090's, as its plans logged them.
//!
//! The 12 GB class's (spec #630, GitHub #637) is the 12 GB artifact's load
//! under `rtx-12gb`, read off its plan on the RTX 5090 in 12 GiB on
//! 2026-11-16 at that measured context: its pool, the pool less 1 GiB, where
//! the ladder steps to 98,304, and the same load on the routes an RTX 30 or
//! 40 card runs, whose dense buffer takes 142 pages of it.

#[path = "support/kv_pool_pins.rs"]
mod kv_pool_pins;

use ignis_runtime::CUDA_CONTEXT_BYTES;
use ignis_server::adaptation::{AdaptStep, Room, steps_text};
use kv_pool_pins::{
    Pool, RTX_12GB, RTX_12GB_8X, RTX_12GB_8X_LESS_1_GIB, RTX_12GB_8X_LESS_1_GIB_98304, RTX_12GB_8X_LESS_1_GIB_ADAPTED,
    RTX_12GB_LESS_1_GIB, RTX_12GB_LESS_1_GIB_131072, RTX_12GB_LESS_1_GIB_ADAPTED, RTX_16GB_AT_THE_CHUNK,
    RTX_24GB_AT_THE_CHUNK, RTX50_16GB, RTX50_16GB_LESS_1_GIB, RTX50_16GB_LESS_3_GIB, RTX50_16GB_LESS_3_GIB_131072_SPARE,
};

/// The context the RTX 5090 measured under Windows on 2026-11-14.
const MEASURED: u64 = 452_161_536;

#[test]
fn rtx_24gb_at_the_chunk_keeps_508s_pool_at_the_constant_and_one_page_more_at_the_measured_context() {
    let load = RTX_24GB_AT_THE_CHUNK;
    assert_eq!(load.pool_tokens(load.on_the_rtx_5090(CUDA_CONTEXT_BYTES)), 1_139_776, "#508's pool");
    assert_eq!(load.pool_tokens(load.on_the_rtx_5090(MEASURED)), 1_139_840, "main's plan");
}

#[test]
fn rtx_16gb_at_the_chunk_keeps_its_profile_notes_pool_at_both_contexts() {
    let load = RTX_16GB_AT_THE_CHUNK;
    for context in [CUDA_CONTEXT_BYTES, MEASURED] {
        assert_eq!(load.pool_tokens(load.on_the_rtx_5090(context)), 386_048, "{context}");
    }
}

#[test]
fn rtx50_16gb_keeps_339s_pool_at_the_constant_and_one_page_more_at_the_measured_context() {
    let load = RTX50_16GB;
    assert_eq!(load.pool_tokens(load.on_the_rtx_5090(CUDA_CONTEXT_BYTES)), 509_184, "#339 A1's 7,956 pages");
    assert_eq!(load.pool_tokens(load.on_the_rtx_5090(MEASURED)), 509_248, "main's plan");
}

#[test]
fn rtx50_16gb_less_1_gib_keeps_340s_pool_at_the_constant_and_one_page_more_at_the_measured_context() {
    let load = RTX50_16GB_LESS_1_GIB;
    assert_eq!(load.pool_tokens(load.on_the_rtx_5090(CUDA_CONTEXT_BYTES)), 392_704, "#340's pool");
    assert_eq!(load.pool_tokens(load.on_the_rtx_5090(MEASURED)), 392_768, "main's plan");
}

#[test]
fn rtx50_16gb_less_3_gib_adapts_to_65536_on_340s_pool_at_the_constant_and_one_page_more_at_the_measured_context() {
    let load = RTX50_16GB_LESS_3_GIB;
    assert_eq!(load.pool_tokens(load.on_the_rtx_5090(CUDA_CONTEXT_BYTES)), 159_680, "#340's pool");
    assert_eq!(load.pool_tokens(load.on_the_rtx_5090(MEASURED)), 159_744, "main's plan");
}

#[test]
fn rtx50_16gb_less_3_gib_misses_131072_and_the_spare_reservation_by_577_pages_at_the_constant_and_576_at_the_measured_context() {
    let refusal = RTX50_16GB_LESS_3_GIB_131072_SPARE;
    assert_eq!(refusal.pages_short(refusal.on_the_rtx_5090(CUDA_CONTEXT_BYTES)), 577, "#340's");
    assert_eq!(refusal.pages_short(refusal.on_the_rtx_5090(MEASURED)), 576, "main's refusal");
}

/// A context above the constant, as the WSL podman machine measured one on
/// the same RTX 5090 (519,270,400 bytes, #604): 66,674,688 bytes more, 113.04
/// pages, so every pool is 113 or 114 pages smaller than at the constant; on
/// the MTP head's pool, whose pages are larger, 106.39, so 106 or 107.
#[test]
fn a_context_above_the_constant_costs_its_whole_pages() {
    const WSL: u64 = 519_270_400;
    for load in [RTX_24GB_AT_THE_CHUNK, RTX_16GB_AT_THE_CHUNK, RTX50_16GB, RTX50_16GB_LESS_1_GIB, RTX50_16GB_LESS_3_GIB] {
        let fewer = load.pool_pages(load.on_the_rtx_5090(CUDA_CONTEXT_BYTES)) - load.pool_pages(load.on_the_rtx_5090(WSL));
        assert!((113..=114).contains(&fewer), "{load:?}: {fewer} pages");
    }
    for load in [RTX_12GB, RTX_12GB_LESS_1_GIB, RTX_12GB_8X, RTX_12GB_8X_LESS_1_GIB] {
        let fewer = load.pool_pages(load.on_the_rtx_5090(CUDA_CONTEXT_BYTES)) - load.pool_pages(load.on_the_rtx_5090(WSL));
        assert!((106..=107).contains(&fewer), "{load:?}: {fewer} pages");
    }
    let refusal = RTX50_16GB_LESS_3_GIB_131072_SPARE;
    let more = refusal.pages_short(refusal.on_the_rtx_5090(WSL)) - refusal.pages_short(refusal.on_the_rtx_5090(CUDA_CONTEXT_BYTES));
    assert!((113..=114).contains(&more), "{more} pages");
}

/// The MTP head's pool cuts its pages for 17 attention layers, 9,792 bytes a
/// token as the 12 GB load's `ignis.runtime.kv_pool` line logs them; every
/// other pool for the 27B's 16, 9,216.
#[test]
fn the_mtp_heads_pages_hold_its_own_layer_beside_the_27bs_sixteen() {
    assert_eq!((Pool::Body.page_bytes(), Pool::Mtp.page_bytes()), (9_216 * 64, 9_792 * 64));
}

/// Spec #630 (GitHub #637): the 12 GB artifact's load under `rtx-12gb` in
/// 12 GiB keeps the pool its plan logged at the measured context, 3,346 pages
/// (214,144 tokens, the three the MTP head's holders keep back among them),
/// one page fewer at the constant, and its 131,072 tokens beside it.
#[test]
fn rtx_12gb_keeps_its_measured_pool_and_one_page_fewer_at_the_constant() {
    let load = RTX_12GB;
    assert_eq!(load.pool_tokens(load.on_the_rtx_5090(MEASURED)), 214_144, "the plan of #635's load");
    assert_eq!(load.pool_tokens(load.on_the_rtx_5090(CUDA_CONTEXT_BYTES)), 214_080);
}

/// Less 1 GiB the 131,072-token step misses its floor of 2,053 pages by 420
/// at the measured context (421 at the constant), 114,688 its 1,797 by 164,
/// and 98,304 holds its 1,541 in a pool of 1,633 pages, 104,512 tokens: the
/// load adapts, and its log says so.
#[test]
fn rtx_12gb_less_1_gib_adapts_to_98304_on_a_pool_of_1633_pages() {
    let refusal = RTX_12GB_LESS_1_GIB_131072;
    assert_eq!(refusal.pages_short(refusal.on_the_rtx_5090(MEASURED)), 420);
    assert_eq!(refusal.pages_short(refusal.on_the_rtx_5090(CUDA_CONTEXT_BYTES)), 421);
    let load = RTX_12GB_LESS_1_GIB;
    assert_eq!(load.pool_tokens(load.on_the_rtx_5090(MEASURED)), 104_512);
    assert_eq!(load.pool_tokens(load.on_the_rtx_5090(CUDA_CONTEXT_BYTES)), 104_448);
    let step = AdaptStep::MaxContext { from: 131_072, to: 98_304, context_step: 16_384, room: Room::OneLane };
    assert_eq!(steps_text(&[step]), RTX_12GB_LESS_1_GIB_ADAPTED);
}

/// The same load on the routes an RTX 30 or 40 card runs: the dense buffer
/// takes 142 pages, 3,204 left in 12 GiB (205,056 tokens against Blackwell's
/// 214,144), and 131,072 tokens still load. Less 1 GiB, with the RTX 5090's
/// context and trellis workspace, 98,304 misses its floor by 50 pages (51 at
/// the constant), and the load adapts to 81,920 on a pool of 1,491 pages.
#[test]
fn on_an_rtx_30_or_40_cards_routes_rtx_12gb_keeps_131072_in_12_gib_and_adapts_to_81920_less_1_gib() {
    let load = RTX_12GB_8X;
    assert_eq!(load.pool_tokens(load.on_the_rtx_5090(MEASURED)), 205_056);
    assert_eq!(load.pool_tokens(load.on_the_rtx_5090(CUDA_CONTEXT_BYTES)), 204_992);
    let blackwell = RTX_12GB.pool_pages(RTX_12GB.on_the_rtx_5090(MEASURED));
    assert_eq!(blackwell - load.pool_pages(load.on_the_rtx_5090(MEASURED)), 142, "the dense buffer's pages");
    let refusal = RTX_12GB_8X_LESS_1_GIB_98304;
    assert_eq!(refusal.pages_short(refusal.on_the_rtx_5090(MEASURED)), 50);
    assert_eq!(refusal.pages_short(refusal.on_the_rtx_5090(CUDA_CONTEXT_BYTES)), 51);
    let load = RTX_12GB_8X_LESS_1_GIB;
    assert_eq!(load.pool_tokens(load.on_the_rtx_5090(MEASURED)), 95_424);
    assert_eq!(load.pool_tokens(load.on_the_rtx_5090(CUDA_CONTEXT_BYTES)), 95_360);
    let step = AdaptStep::MaxContext { from: 131_072, to: 81_920, context_step: 16_384, room: Room::OneLane };
    assert_eq!(steps_text(&[step]), RTX_12GB_8X_LESS_1_GIB_ADAPTED);
}
