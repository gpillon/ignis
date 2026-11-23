//! `struct ignis_decode_options` and `struct ignis_prefill_options`
//! (`kernel/include/ignis_step.h`) as plain `repr(C)` mirrors, and (GitHub
//! #711) `struct ignis_program_stats` with its round shapes': no kernel is
//! needed to lay them out, so their layout pins run in the CPU suite (GitHub
//! #340, #520) while [`crate::step`], which fills and reads them, is
//! `cuda`-only.

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

/// 1:1 with `struct ignis_decode_options` (ADR 0016; P5-04, GitHub
/// #153). `size` must be `sizeof(struct ignis_decode_options)`; a null
/// options pointer is today's round.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IgnisDecodeOptions {
    pub size: u32,
    pub speculative_window: u32,
    pub drafts: *const i32,
    pub draft_counts: *const u32,
    pub out_committed_counts: *mut i32,
    /// P5-05 (GitHub #155): each lane's extent this round, or null.
    pub out_extents: *mut u32,
    /// P6-06 (GitHub #242): each lane's committed token's probability
    /// within its own permitted set, or 0 for a lane that declared
    /// none. Null asks for nothing. One float per lane rather than a
    /// logits row — the per-digit confidence a number's uncertainty is
    /// summed from has to come from somewhere, and this is the cheapest
    /// thing that is not the host reading logits (ADR 0034).
    pub out_permitted_probs: *mut f32,
    /// GitHub #302: each lane's n-gram table rows on a Flash-Next load,
    /// host `[batch][16][90]` (a verify round: `[batch][window + 1]
    /// [16][90]`); null on the 27B.
    pub ngram_rows: *const u8,
    /// GitHub #307: a Flash-Next MTP load's drafts for each lane's next
    /// round, `[batch][window]`; null asks for nothing.
    pub out_drafts: *mut i32,
    /// GitHub #315 (ADR 0048): per lane, 1 when the round redirected the
    /// lane's successor to its `reasoning_close_id`, else 0; null asks
    /// for nothing.
    pub out_reasoning_redirected: *mut i32,
    /// GitHub #329 item 5 (ADR 0056): a Flash-Next round's BF16 logits,
    /// every row its head produced (`[batch][vocab]`, a verify round
    /// `[batch][window + 1][vocab]`); null asks for nothing.
    pub out_logits: *mut u16,
    /// GitHub #267, test-only: a verify round's teacher forcing, host
    /// `[batch][window + 1]`, or null -- see
    /// `step::decode_program_verify_forced`.
    pub forced_tokens: *const i32,
    /// GitHub #590: per lane, 1 when the verify commit's cut is the host's
    /// (`host_cut`), else 0; null with `host_cut` for none.
    pub host_cut_lanes: *const u8,
    /// GitHub #590: the host's cut of a lane's verified run -- `run` is its
    /// anchor and accepted drafts, `run_count` of them -- returning how many
    /// of them the commit keeps (1..=run_count), negative on a failure. See
    /// `step::decode_program_verify_cut`.
    pub host_cut: Option<unsafe extern "C" fn(*mut std::ffi::c_void, u64, *const i32, i32) -> i32>,
    pub host_cut_context: *mut std::ffi::c_void,
    /// GitHub #692, test-only: a teacher-forced round's drafts in place of
    /// its drafter's, host `[batch][window]`, or null -- see
    /// `step::decode_program_verify_forced_with`.
    pub forced_drafts: *const i32,
    /// GitHub #692, test-only: the window of the round shape a
    /// teacher-forced round runs at, or 0 for the shape any round's choice
    /// gives.
    pub forced_shape_window: u32,
    /// GitHub #689: the verify round's shape, 0 the base shape or 1 the
    /// extended one. GitHub #712: each lane's, `[batch]`; null asks for
    /// nothing.
    pub out_lane_shapes: *mut u32,
    /// GitHub #690: how many of the verify round's lanes asked the extended
    /// shape going into it; null asks for nothing.
    pub out_shape_asks: *mut u32,
    /// GitHub #713, test-only: each lane's forced shape, `[batch]`, 1 the
    /// extended shape and 0 the base one, so that lanes of both run a mixed
    /// round; null for the shape any round's choice gives -- see
    /// `step::decode_program_verify_mixed`.
    pub forced_lane_shapes: *const u8,
}

/// 1:1 with `struct ignis_prefill_options` (ADR 0016, P2-02, GitHub
/// #84). `size` must be `sizeof(struct ignis_prefill_options)`; the
/// leaf rejects a size it does not recognize.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IgnisPrefillOptions {
    pub size: u32,
    pub route: i32,
    pub compute_policy: i32,
    /// GitHub #178: the span's axis-major `[3, T]` positions, or null
    /// for a text span (whose media fields are then empty).
    pub mrope_positions: *const i32,
    pub rope_delta: i32,
    pub media_column_count: u32,
    pub media: *const IgnisMediaEmbedding,
    pub media_scatter_indices: *const i32,
    pub media_first_column: u32,
    /// P6-06 (GitHub #242): the span's own draw's probability within its
    /// permitted set, or null. A constrained run's **first** token is
    /// drawn here — a decode round returns the successor the previous
    /// call made ready — so this is where its confidence comes back.
    pub out_permitted_prob: *mut f32,
    /// GitHub #260 (ADR 0038): the attention readout — none when the
    /// ordinal is negative. One query head of one GQA layer at the
    /// span's last position, over the keys of
    /// `[attention_key_begin, + attention_key_count)` as that layer's
    /// attention read them; one pre-softmax score per key into
    /// `out_attention_scores`, and `*out_attention_read` 1 when the leaf
    /// could read them, 0 when it could not.
    pub attention_gqa_ordinal: i32,
    pub attention_query_head: i32,
    pub attention_key_begin: i64,
    pub attention_key_count: i64,
    pub out_attention_scores: *mut f32,
    pub out_attention_read: *mut i32,
    /// GitHub #263 (ADR 0039): the head set read beside the attention
    /// readout, or none when `attention_set_count` is 0. Head `i` is
    /// query head `attention_set_query_heads[i]` of GQA layer
    /// `attention_set_gqa_ordinals[i]`; its argmax over the span minus
    /// the `attention_excluded` span-relative keys lands in
    /// `out_attention_set_argmax[i]`. Every armed layer must read, or
    /// `*out_attention_read` is 0.
    pub attention_set_count: u32,
    pub attention_set_gqa_ordinals: *const i32,
    pub attention_set_query_heads: *const i32,
    pub attention_excluded_count: u32,
    pub attention_excluded: *const i32,
    pub out_attention_set_argmax: *mut i32,
    /// GitHub #264 (ADR 0040): with the set, head `i`'s own score at its
    /// argmax, and the four scores around it — the argmax's left, right,
    /// up and down neighbours in an image grid of `attention_grid_cols`
    /// columns, `NaN` for one off the grid.
    pub attention_grid_cols: u32,
    pub out_attention_set_peak: *mut f32,
    pub out_attention_set_neighbours: *mut f32,
    /// The measurement readout: every position's BF16 logits, host,
    /// `[num_tokens][vocab]` ([`super::prefill_program_span_logits`]).
    pub out_span_logits: *mut u16,
    /// GitHub #275 (ADR 0041): with the set, every head's whole row,
    /// host, `[attention_set_count][attention_key_count]`, or null. With
    /// rows the grid's columns may be 0 and the neighbours null.
    pub out_attention_set_rows: *mut f32,
    /// GitHub #302: the span's n-gram table rows on a Flash-Next load,
    /// host `[num_tokens][16][90]`; null on the 27B.
    pub ngram_rows: *const u8,
    /// GitHub #315 (ADR 0048): 1 when the span's draw was redirected to
    /// its `reasoning_close_id`, else 0; null asks for nothing.
    pub out_reasoning_redirected: *mut i32,
    /// GitHub #506: the first absolute position whose feature taps a
    /// later drafter window reads. The span taps from the later of its
    /// last 2048 positions and this; 0 is that span rule alone.
    pub drafter_floor: u64,
    /// GitHub #519: the prompt token after the span, or -1 on the
    /// prompt's last span. With a token the leaf runs no output head and
    /// draws nothing, unless `keeps_draw`; Flash-Next's MTP head builds
    /// its entry at the span's last position from it (GitHub #520).
    pub next_prompt_token: i32,
    /// GitHub #520: nonzero when a prefix image or a prompt checkpoint
    /// retains the state after the span, which then draws with a next
    /// prompt token too, as the image carries the draw.
    pub keeps_draw: i32,
    /// GitHub #287: nonzero when the span's draw keeps its scores for a
    /// repair -- a request held to a grammar; 0 keeps none.
    pub keeps_scores: i32,
    /// GitHub #620: a measurement readout, on a load with the 27B's MTP head:
    /// the target's post-final-norm hidden of every position of the span
    /// (host BF16 `[tokens][hidden]`), what the head's rows read. Null in
    /// every serving call.
    pub out_span_hidden: *mut u16,
    /// GitHub #624: the three axes (temporal, height, width) the position
    /// before a multimodal span rotated at, or null. A 27B load built for the
    /// MTP head needs them on a multimodal span that continues a sequence: the
    /// head's row there rotates where the target's hidden did.
    pub mrope_previous: *const i32,
}

/// Opaque `struct ignis_media_embedding` (GitHub #178).
#[repr(C)]
pub struct IgnisMediaEmbedding {
    _private: [u8; 0],
}

/// `IGNIS_ROUND_SHAPES_MAX` (`kernel/include/ignis_step.h`): the most
/// round shapes a load builds (GitHub #686): GitHub #687, a DFlash2
/// load's base and extended shapes.
pub const IGNIS_ROUND_SHAPES_MAX: usize = 2;

/// `IGNIS_ROUND_MIXES` (`kernel/include/ignis_step.h`, GitHub #713): the
/// mixes a mixed verify round takes.
pub const IGNIS_ROUND_MIXES: usize = 12;

/// `IGNIS_MIXED_FALLBACK_REASONS` (GitHub #714): why a round at a mix
/// with no graph ran at the uniform choice (`enum ignis_mixed_fallback`).
pub const IGNIS_MIXED_FALLBACK_REASONS: usize = 2;

/// 1:1 with `struct ignis_round_shape_stats` (GitHub #686): one round
/// shape's verify round.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IgnisRoundShapeStats {
    pub window: u32,
    pub verify_graph_ready_mask: u32,
    pub vram_bytes: u64,
    pub verify_graph_nodes: [u32; 8],
    pub verify_graph_copy_nodes: [u32; 8],
    pub drafter_append_graph_ready_mask: u32,
    /// GitHub #687.
    pub lanes: u32,
    pub verify_folds_in_flight: u64,
    /// GitHub #687.
    pub verify_rounds: u64,
    pub eager_verify_rounds: u64,
    /// GitHub #712.
    pub drafter_append_graph_nodes: [u32; 8],
}

/// 1:1 with `struct ignis_program_stats`: the program's counters, read back
/// by `step::program_stats`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IgnisProgramStats {
    pub vram_bytes: u64,
    pub last_step_micros: u64,
    pub kernel_count: u64,
    /// P3-05 (GitHub #102): `cudaGraphLaunch` calls the most recent
    /// `ignis_program_decode` call made -- 1 if it replayed a decode
    /// graph, 0 if it ran the eager per-lane loop.
    pub graph_launches: u64,
    /// P3-05 (GitHub #102): bit (w-1) set when a decode graph for exact
    /// width w (1..the load's decode lanes, GitHub #336) is captured and
    /// replayable.
    pub decode_graph_ready_mask: u32,
    /// P5-04 (GitHub #153): the same for the verify graphs at the load's
    /// draft window; 0 on a load without one.
    pub verify_graph_ready_mask: u32,
    /// GitHub #572: entry w-1 the node count of the 27B's decode graph
    /// and of its verify graph for exact width w; 0 for a width with no
    /// graph and on Flash-Next.
    pub decode_graph_nodes: [u32; 8],
    pub verify_graph_nodes: [u32; 8],
    /// GitHub #677: entry w-1 how many of the decode graph's nodes for
    /// exact width w are copies.
    pub decode_graph_copy_nodes: [u32; 8],
    /// GitHub #676: the verify rounds of this load that returned with
    /// their fold in flight.
    pub verify_folds_in_flight: u64,
    /// GitHub #676: bit (w-1) set when the DFlash2 drafter's append after
    /// a verify round's commit is a captured graph at exact width w.
    pub drafter_append_graph_ready_mask: u32,
    /// GitHub #680: the same for the verify graphs.
    pub verify_graph_copy_nodes: [u32; 8],
    /// GitHub #681: the GDN gating's home, 0 chain, 1 branch.
    pub gdn_gating: u32,
    /// GitHub #686: the round shapes the load built, and each one's
    /// verify round, the base shape first.
    pub round_shapes: u32,
    pub shapes: [IgnisRoundShapeStats; IGNIS_ROUND_SHAPES_MAX],
    /// GitHub #713: each mix's verify and drafter append graph nodes.
    pub mixed_verify_graph_nodes: [u32; IGNIS_ROUND_MIXES],
    pub mixed_append_graph_nodes: [u32; IGNIS_ROUND_MIXES],
    /// GitHub #714: whether the latest capture asked for the mixed
    /// graphs, the mixes it could not capture, each mix's graphs' bytes
    /// and the rounds at a mix with no graph run at the uniform choice,
    /// by reason.
    pub mixed_graphs_asked: u32,
    pub mixed_graph_failed_mask: u32,
    pub mixed_graph_bytes: [u64; IGNIS_ROUND_MIXES],
    pub mixed_fallback_rounds: [u64; IGNIS_MIXED_FALLBACK_REASONS],
    /// GitHub #718: each mix's verify and drafter append graph's root
    /// nodes and memset nodes.
    pub mixed_verify_graph_roots: [u32; IGNIS_ROUND_MIXES],
    pub mixed_verify_graph_memset_nodes: [u32; IGNIS_ROUND_MIXES],
    pub mixed_append_graph_roots: [u32; IGNIS_ROUND_MIXES],
    pub mixed_append_graph_memset_nodes: [u32; IGNIS_ROUND_MIXES],
    /// GitHub #719: the mixed rounds run at each mix.
    pub mixed_rounds: [u64; IGNIS_ROUND_MIXES],
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GitHub #267: `ignis_decode_options` with the teacher forcing appended
    /// last, so every field the serving path sets keeps its offset. uint32
    /// size and window, then thirteen pointers, then GitHub #692's uint32
    /// and its padding, then GitHub #689's, #690's and #713's pointers; the leaf
    /// pins the same numbers
    /// (kernel/include/ignis_step.h) and refuses any other `size`.
    #[test]
    fn the_decode_options_mirror_is_the_leaf_structs_layout() {
        assert_eq!(std::mem::size_of::<IgnisDecodeOptions>(), 152);
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, out_logits), 72);
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, forced_tokens), 80);
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, host_cut_lanes), 88);
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, host_cut), 96);
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, host_cut_context), 104);
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, forced_drafts), 112);
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, forced_shape_window), 120);
        // GitHub #689: the round's shape, after the forced shape (#712: each lane's).
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, out_lane_shapes), 128);
        // GitHub #690: how many lanes asked it.
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, out_shape_asks), 136);
        // GitHub #713: the forced lane shapes, last.
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, forced_lane_shapes), 144);
    }

    /// GitHub #711: `ignis_program_stats` has no `size` field, so both sides
    /// pin its layout and its round shapes': GitHub #712's append graph nodes
    /// at each shape's tail, and the mixes' fields of GitHub #713, #714, #718
    /// and #719 at the stats' tail. The leaf pins the same numbers
    /// (kernel/include/ignis_step.h).
    #[test]
    fn the_program_stats_mirror_is_the_leaf_structs_layout() {
        assert_eq!(std::mem::size_of::<IgnisRoundShapeStats>(), 144);
        assert_eq!(std::mem::offset_of!(IgnisRoundShapeStats, verify_folds_in_flight), 88);
        assert_eq!(std::mem::offset_of!(IgnisRoundShapeStats, drafter_append_graph_nodes), 112);
        assert_eq!(std::mem::size_of::<IgnisProgramStats>(), 984);
        assert_eq!(std::mem::offset_of!(IgnisProgramStats, shapes), 192);
        assert_eq!(std::mem::offset_of!(IgnisProgramStats, mixed_verify_graph_nodes), 480);
        assert_eq!(std::mem::offset_of!(IgnisProgramStats, mixed_graphs_asked), 576);
        assert_eq!(std::mem::offset_of!(IgnisProgramStats, mixed_graph_bytes), 584);
        assert_eq!(std::mem::offset_of!(IgnisProgramStats, mixed_fallback_rounds), 680);
        assert_eq!(std::mem::offset_of!(IgnisProgramStats, mixed_verify_graph_roots), 696);
        assert_eq!(std::mem::offset_of!(IgnisProgramStats, mixed_append_graph_memset_nodes), 840);
        assert_eq!(std::mem::offset_of!(IgnisProgramStats, mixed_rounds), 888);
    }

    /// GitHub #520: `keeps_draw` filled the tail padding after
    /// `next_prompt_token`, so `size` alone no longer tells the layout before
    /// it from the layout after; GitHub #287's `keeps_scores` follows it,
    /// GitHub #620's `out_span_hidden` them and GitHub #624's
    /// `mrope_previous` that. The leaf pins the same numbers
    /// (kernel/include/ignis_step.h), as for the decode options.
    #[test]
    fn the_prefill_options_mirror_is_the_leaf_structs_layout() {
        assert_eq!(std::mem::size_of::<IgnisPrefillOptions>(), 248);
        assert_eq!(std::mem::offset_of!(IgnisPrefillOptions, mrope_previous), 240);
        assert_eq!(std::mem::offset_of!(IgnisPrefillOptions, out_reasoning_redirected), 200);
        assert_eq!(std::mem::offset_of!(IgnisPrefillOptions, drafter_floor), 208);
        assert_eq!(std::mem::offset_of!(IgnisPrefillOptions, next_prompt_token), 216);
        assert_eq!(std::mem::offset_of!(IgnisPrefillOptions, keeps_draw), 220);
        assert_eq!(std::mem::offset_of!(IgnisPrefillOptions, keeps_scores), 224);
        assert_eq!(std::mem::offset_of!(IgnisPrefillOptions, out_span_hidden), 232);
    }
}
