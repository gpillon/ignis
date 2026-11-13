//! `struct ignis_decode_options` and `struct ignis_prefill_options`
//! (`kernel/include/ignis_step.h`) as plain `repr(C)` mirrors: no kernel is
//! needed to lay them out, so their layout pins run in the CPU suite (GitHub
//! #340, #520) while [`crate::step`], which fills them, is `cuda`-only.

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
}

/// Opaque `struct ignis_media_embedding` (GitHub #178).
#[repr(C)]
pub struct IgnisMediaEmbedding {
    _private: [u8; 0],
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GitHub #267: `ignis_decode_options` with the teacher forcing appended
    /// last, so every field the serving path sets keeps its offset. uint32
    /// size and window, then ten pointers; the leaf pins the same numbers
    /// (kernel/include/ignis_step.h) and refuses any other `size`.
    #[test]
    fn the_decode_options_mirror_is_the_leaf_structs_layout() {
        assert_eq!(std::mem::size_of::<IgnisDecodeOptions>(), 112);
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, out_logits), 72);
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, forced_tokens), 80);
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, host_cut_lanes), 88);
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, host_cut), 96);
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, host_cut_context), 104);
    }

    /// GitHub #520: `keeps_draw` filled the tail padding after
    /// `next_prompt_token`, so `size` alone no longer tells the layout before
    /// it from the layout after; GitHub #287's `keeps_scores` follows it. The
    /// leaf pins the same numbers (kernel/include/ignis_step.h), as for the
    /// decode options.
    #[test]
    fn the_prefill_options_mirror_is_the_leaf_structs_layout() {
        assert_eq!(std::mem::size_of::<IgnisPrefillOptions>(), 232);
        assert_eq!(std::mem::offset_of!(IgnisPrefillOptions, out_reasoning_redirected), 200);
        assert_eq!(std::mem::offset_of!(IgnisPrefillOptions, drafter_floor), 208);
        assert_eq!(std::mem::offset_of!(IgnisPrefillOptions, next_prompt_token), 216);
        assert_eq!(std::mem::offset_of!(IgnisPrefillOptions, keeps_draw), 220);
        assert_eq!(std::mem::offset_of!(IgnisPrefillOptions, keeps_scores), 224);
    }
}
