//! Vision as a load option (GitHub #177, spec
//! `.scratch/vision/specs/01-image-input.md`).
//!
//! Like speculation, vision is engine residency, chosen at load and frozen for
//! the life of that load: with a [`Vision`], the `vision/*` objects are bound
//! in their stored formats and the leaf reserves the encoder workspace (inside
//! the prefill scratch, GitHub #212) and one item's output transient for the
//! envelope, before the sequence pool exists.
//! `None` is today's engine — nothing vision-related is bound or allocated.
//!
//! GitHub #178 adds what a multimodal request carries through the scheduler
//! ([`Multimodal`]), the chunk rule over its media items, and the per-item
//! encoder control the leaf's media encode step takes
//! ([`VisionItemControl`], the reference's `build_vision_control`).

pub use ignis_artifact::vision::{Grid, MediaItem, PreparedPrompt, TokenSpan};

use crate::types::TokenId;

/// The default vision envelope, in merged vision tokens per request (131,072
/// raw patches): the reference's own frontend limit.
pub const DEFAULT_VISION_MAX_TOKENS: u32 = 32_768;

/// The widest envelope a load accepts (`IGNIS_VISION_MAX_TOKENS_LIMIT`,
/// `kernel/include/ignis_model.h`).
pub const VISION_MAX_TOKENS_LIMIT: u32 = 1 << 20;

/// The artifact's `vision/*` object count: three patch/position globals, 27
/// blocks of twelve, and the merger's six.
pub const VISION_OBJECTS: usize = 3 + 27 * 12 + 6;

/// The target's hidden width, which the encoder's merged output carries.
const HIDDEN: u64 = 5120;

/// The leaf's vision allocation alignment (`kVisionWorkspaceAlignment`).
const VISION_ALIGN: u64 = 256;

/// The vision envelope a load reserves for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vision {
    max_tokens: u32,
}

/// A vision envelope outside `1..=VISION_MAX_TOKENS_LIMIT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionEnvelopeOutOfRange(pub u32);

impl std::fmt::Display for VisionEnvelopeOutOfRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "vision max tokens must be in 1..={VISION_MAX_TOKENS_LIMIT}, got {}",
            self.0
        )
    }
}

impl std::error::Error for VisionEnvelopeOutOfRange {}

impl Default for Vision {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_VISION_MAX_TOKENS,
        }
    }
}

impl Vision {
    /// An envelope of `max_tokens` merged vision tokens per request.
    pub fn new(max_tokens: u32) -> Result<Self, VisionEnvelopeOutOfRange> {
        if max_tokens == 0 || max_tokens > VISION_MAX_TOKENS_LIMIT {
            return Err(VisionEnvelopeOutOfRange(max_tokens));
        }
        Ok(Self { max_tokens })
    }

    /// The configured envelope, in merged vision tokens.
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// The envelope the leaf actually reserves for: no request can carry more
    /// vision tokens than its context holds.
    pub fn envelope_tokens(&self, max_context_tokens: u32) -> u32 {
        self.max_tokens.min(max_context_tokens)
    }

    /// One item's `[5120, V]` BF16 encoder output at the envelope, rounded to
    /// the leaf's alignment (the output transient half of the reservation).
    pub fn output_transient_bytes(&self, max_context_tokens: u32) -> u64 {
        let bytes = HIDDEN * u64::from(self.envelope_tokens(max_context_tokens)) * 2;
        bytes.div_ceil(VISION_ALIGN) * VISION_ALIGN
    }
}

/// The multimodal part of a request (GitHub #178): the prompt's three-axis
/// rope positions, its `rope_delta` and its media items, as the processor
/// prepared them. A text-only request carries none and takes today's path.
///
/// Invariant: `positions` is `3 * prompt tokens` long and every media item's
/// token span lies inside the prompt — [`Multimodal::from_prepared`] checks
/// it at the seam where a prepared prompt becomes a request, so the scheduler
/// and the leaf can index it without asking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Multimodal {
    /// Axis-major `[3, T]`: temporal, height, width, over the whole prompt.
    pub positions: Vec<i32>,
    /// Decode rotates at `position + rope_delta`.
    pub rope_delta: i32,
    /// The media items in prompt order, their token spans ascending.
    pub media: Vec<MediaItem>,
}

/// The part of one media item a prefill chunk covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkMedia {
    /// The item's index in [`Multimodal::media`].
    pub item: usize,
    /// The encoder output column of the first covered placeholder.
    pub first_column: u32,
    /// Chunk-relative positions of the covered placeholder columns.
    pub scatter_indices: Vec<i32>,
    /// Whether this chunk covers the item's last placeholder.
    pub completes_item: bool,
}

impl Multimodal {
    /// Split a prepared prompt into the tokens the scheduler submits and the
    /// multimodal part the request carries beside them.
    ///
    /// Panics when the prompt breaks the type's invariant (positions three
    /// per token, every span inside the prompt) — a logic error in whoever
    /// built it, caught here rather than indexing out of bounds on the model
    /// thread. The processor's own output always satisfies it.
    pub fn from_prepared(prompt: PreparedPrompt) -> (Vec<TokenId>, Self) {
        let tokens = prompt.token_ids.len();
        assert_eq!(
            prompt.positions.len(),
            3 * tokens,
            "a prepared prompt carries three positions per token"
        );
        assert!(
            prompt
                .media
                .iter()
                .all(|item| item.token_span.begin + item.token_span.count <= tokens),
            "a prepared prompt's media spans lie inside it"
        );
        let multimodal = Self {
            positions: prompt.positions,
            rope_delta: prompt.rope_delta,
            media: prompt.media,
        };
        (prompt.token_ids, multimodal)
    }

    /// Prompt tokens the positions cover.
    pub fn prompt_tokens(&self) -> usize {
        self.positions.len() / 3
    }

    /// Axis-major `[3, len]` positions of the span `[start, start + len)`.
    pub fn span_positions(&self, start: usize, len: usize) -> Vec<i32> {
        let tokens = self.prompt_tokens();
        (0..3)
            .flat_map(|axis| &self.positions[axis * tokens + start..axis * tokens + start + len])
            .copied()
            .collect()
    }

    fn item_end(item: &MediaItem) -> usize {
        item.token_span.begin + item.token_span.count
    }

    /// How many of `take` tokens from `start` one prefill chunk may carry: a
    /// chunk holds at most one media item's placeholders, so it stops where
    /// a second item would begin. A media item may still span several
    /// chunks.
    pub fn cap_chunk(&self, start: u32, take: u32) -> u32 {
        let (start, end) = (start as usize, start as usize + take as usize);
        let mut overlapping = self
            .media
            .iter()
            .filter(|item| Self::item_end(item) > start && item.token_span.begin < end);
        match (overlapping.next(), overlapping.next()) {
            (Some(_), Some(second)) => (second.token_span.begin - start) as u32,
            _ => take,
        }
    }

    /// The last whole-page boundary at or before `at` that does not fall
    /// **inside** a media item's placeholder span (GitHub #193), or 0.
    ///
    /// Retained state — a shared prefix, a retained prefix — is cut at whole
    /// KV pages, and a page floor lands mid-image whenever an image straddles
    /// a page boundary, which a picture hundreds of tokens long almost always
    /// does. The boundary walks back to the page holding the item's first
    /// placeholder rather than being refused: the head before the image is
    /// still the history every sibling shares, whatever picture it sent. A
    /// boundary exactly at an item's first placeholder is outside it.
    pub fn floor_outside_media(&self, at: u32, page_tokens: u32) -> u32 {
        if page_tokens == 0 {
            return 0;
        }
        let mut floor = (at / page_tokens) * page_tokens;
        // Walking back past one item can land inside the one before it; the
        // items are ascending, so one pass from the last item down settles it.
        for item in self.media.iter().rev() {
            let begin = item.token_span.begin as u32;
            if begin < floor && floor < begin + item.token_span.count as u32 {
                floor = (begin / page_tokens) * page_tokens;
            }
        }
        floor
    }

    /// The media item the chunk `[start, start + len)` covers, if any. The
    /// chunk must already be capped by [`Self::cap_chunk`].
    pub fn chunk_media(&self, start: u32, len: u32) -> Option<ChunkMedia> {
        let (start, end) = (start as usize, start as usize + len as usize);
        let item = self
            .media
            .iter()
            .position(|item| Self::item_end(item) > start && item.token_span.begin < end)?;
        let span = self.media[item].token_span;
        let (from, to) = (span.begin.max(start), Self::item_end(&self.media[item]).min(end));
        Some(ChunkMedia {
            item,
            first_column: (from - span.begin) as u32,
            scatter_indices: (from - start..to - start).map(|i| i as i32).collect(),
            completes_item: to == Self::item_end(&self.media[item]),
        })
    }
}

/// The side of the 2-D position-embedding table the encoder interpolates
/// (48 x 48).
const POSITION_SIDE: i32 = 48;

/// One media item's encoder control (the reference's `VisionItemControl`,
/// `impl/vision/control.cpp`): computed on the host and handed to the media
/// encode step with the item's patch rows.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionItemControl {
    /// Raw patches (patch rows) the item carries.
    pub patches: u32,
    /// Merged tokens the encoder emits (`patches / 4`).
    pub merged_tokens: u32,
    /// Attention segments: one per temporal frame.
    pub segment_count: u32,
    /// Axis-major `[P, 2]` vision RoPE positions: every patch's row, then
    /// every patch's column, in 2x2 merge-block order.
    pub position_ids: Vec<i32>,
    /// `[segments + 1]` segment bounds over the patches.
    pub cu_seqlens: Vec<i32>,
    /// `[4, P]` bilinear position-table corners, four per patch.
    pub position_table_indices: Vec<i32>,
    /// `[4, P]` their weights, four per patch.
    pub position_table_weights: Vec<f32>,
}

fn table_coordinate(index: i32, size: i32) -> f32 {
    if size <= 1 {
        0.0
    } else {
        index as f32 * (POSITION_SIDE - 1) as f32 / (size - 1) as f32
    }
}

/// The encoder control of an item on `grid`, step for step the reference's.
/// The grid's height and width are merge-aligned by construction (the
/// processor's smart resize).
pub fn vision_item_control(grid: Grid) -> VisionItemControl {
    const MERGE: i32 = 2;
    let (t, h, w) = (grid.t as i32, grid.h as i32, grid.w as i32);
    let patches = (t * h * w) as usize;
    let mut control = VisionItemControl {
        patches: patches as u32,
        merged_tokens: (patches / 4) as u32,
        segment_count: t as u32,
        position_ids: vec![0; patches * 2],
        cu_seqlens: vec![0],
        position_table_indices: Vec::with_capacity(patches * 4),
        position_table_weights: Vec::with_capacity(patches * 4),
    };
    let mut cursor = 0;
    for _ in 0..t {
        control.cu_seqlens.push(control.cu_seqlens.last().unwrap() + h * w);
        for block_y in 0..h / MERGE {
            for block_x in 0..w / MERGE {
                for inner_y in 0..MERGE {
                    for inner_x in 0..MERGE {
                        let (y, x) = (block_y * MERGE + inner_y, block_x * MERGE + inner_x);
                        control.position_ids[cursor] = y;
                        control.position_ids[patches + cursor] = x;
                        cursor += 1;
                        let (yf, xf) = (table_coordinate(y, h), table_coordinate(x, w));
                        let (y0, x0) = (yf as i32, xf as i32);
                        let (y1, x1) = ((y0 + 1).min(POSITION_SIDE - 1), (x0 + 1).min(POSITION_SIDE - 1));
                        let (wy, wx) = (yf - y0 as f32, xf - x0 as f32);
                        control.position_table_indices.extend([
                            y0 * POSITION_SIDE + x0,
                            y0 * POSITION_SIDE + x1,
                            y1 * POSITION_SIDE + x0,
                            y1 * POSITION_SIDE + x1,
                        ]);
                        control.position_table_weights.extend([
                            (1.0 - wy) * (1.0 - wx),
                            (1.0 - wy) * wx,
                            wy * (1.0 - wx),
                            wy * wx,
                        ]);
                    }
                }
            }
        }
    }
    control
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(begin: usize, count: usize) -> MediaItem {
        // A grid of `count` merged tokens laid out in one row.
        MediaItem {
            grid: Grid { t: 1, h: 2, w: 2 * count as u32 },
            token_span: TokenSpan { begin, count },
            patches: Vec::new(),
            content_digest: [0; 32],
        }
    }

    fn multimodal(tokens: usize, media: Vec<MediaItem>) -> Multimodal {
        let positions = (0..3).flat_map(|axis| (0..tokens).map(move |t| (axis * 1000 + t) as i32)).collect();
        Multimodal { positions, rope_delta: -3, media }
    }

    #[test]
    fn a_chunk_holds_at_most_one_media_item() {
        let prompt = multimodal(100, vec![item(10, 20), item(40, 20), item(70, 5)]);
        // From the start, the chunk runs up to the second item's first
        // placeholder.
        assert_eq!(prompt.cap_chunk(0, 64), 40);
        // A chunk inside one item and ending before the next is untouched.
        assert_eq!(prompt.cap_chunk(15, 20), 20);
        // Starting inside the second item, it stops where the third begins.
        assert_eq!(prompt.cap_chunk(45, 55), 25);
        // Text alone, or one item, is never cut.
        assert_eq!(prompt.cap_chunk(75, 25), 25);
        assert_eq!(multimodal(10, Vec::new()).cap_chunk(0, 10), 10);
    }

    #[test]
    fn a_chunk_names_the_item_columns_it_covers() {
        let prompt = multimodal(100, vec![item(10, 20), item(40, 20)]);
        assert_eq!(prompt.chunk_media(0, 5), None);
        // The item's first 8 placeholders, at chunk columns 10..18.
        let head = prompt.chunk_media(0, 18).unwrap();
        assert_eq!((head.item, head.first_column, head.completes_item), (0, 0, false));
        assert_eq!(head.scatter_indices, (10..18).collect::<Vec<i32>>());
        // The rest of the item from column 8, and text after it.
        let tail = prompt.chunk_media(18, 22).unwrap();
        assert_eq!((tail.item, tail.first_column, tail.completes_item), (0, 8, true));
        assert_eq!(tail.scatter_indices, (0..12).collect::<Vec<i32>>());
        let second = prompt.chunk_media(40, 60).unwrap();
        assert_eq!((second.item, second.first_column, second.completes_item), (1, 0, true));
    }

    #[test]
    fn a_page_floor_never_lands_inside_an_image() {
        let prompt = multimodal(200, vec![item(20, 30), item(60, 50)]);
        // Text before any image, and a floor exactly at an image's first
        // placeholder, stand.
        assert_eq!(prompt.floor_outside_media(19, 16), 16);
        assert_eq!(prompt.floor_outside_media(20, 4), 20);
        // 48 is inside 20..50: back to the page holding placeholder 20.
        assert_eq!(prompt.floor_outside_media(50, 16), 16);
        // 96 is inside 60..110, and its first placeholder's page (48) is
        // inside 20..50 in turn: both walk back.
        assert_eq!(prompt.floor_outside_media(100, 16), 16);
        // Past both images, the plain floor.
        assert_eq!(prompt.floor_outside_media(130, 16), 128);
        assert_eq!(prompt.floor_outside_media(130, 0), 0);
        assert_eq!(multimodal(64, Vec::new()).floor_outside_media(63, 16), 48);
    }

    #[test]
    fn span_positions_are_axis_major_slices() {
        let prompt = multimodal(5, Vec::new());
        assert_eq!(prompt.span_positions(1, 2), [1, 2, 1001, 1002, 2001, 2002]);
    }

    #[test]
    fn a_prepared_prompt_splits_into_tokens_and_its_multimodal_part() {
        let prompt = PreparedPrompt {
            token_ids: vec![7, 8],
            token_types: vec![0, 0],
            positions: vec![0, 1, 0, 1, 0, 1],
            rope_delta: 0,
            media: vec![item(0, 1)],
            frontiers: Vec::new(),
        };
        let (tokens, multimodal) = Multimodal::from_prepared(prompt);
        assert_eq!(tokens, [7, 8]);
        assert_eq!(multimodal.prompt_tokens(), 2);
        assert_eq!(multimodal.media.len(), 1);
    }

    #[test]
    fn item_control_follows_the_merge_block_order() {
        let control = vision_item_control(Grid { t: 1, h: 4, w: 4 });
        assert_eq!((control.patches, control.merged_tokens, control.segment_count), (16, 4, 1));
        assert_eq!(control.cu_seqlens, [0, 16]);
        // Block (0,0) then block (0,1): rows 0,0,1,1 and columns 0,1,0,1 / 2,3,2,3.
        assert_eq!(&control.position_ids[..8], [0, 0, 1, 1, 0, 0, 1, 1]);
        assert_eq!(&control.position_ids[16..24], [0, 1, 0, 1, 2, 3, 2, 3]);
        // Patch 1 is (y 0, x 1): x lands 47/3 of the way along the table.
        let xf = 47.0f32 / 3.0;
        let wx = xf - 15.0;
        assert_eq!(&control.position_table_indices[4..8], [15, 16, 48 + 15, 48 + 16]);
        assert_eq!(&control.position_table_weights[4..8], [1.0 - wx, wx, 0.0, 0.0]);
        // The last patch sits on the table's far corner.
        assert_eq!(&control.position_table_indices[60..64], [47 * 48 + 47; 4]);
        assert_eq!(control.position_table_weights[60], 1.0);
    }

    #[test]
    fn a_video_grid_gets_one_segment_per_frame() {
        let control = vision_item_control(Grid { t: 2, h: 2, w: 2 });
        assert_eq!(control.cu_seqlens, [0, 4, 8]);
        assert_eq!(control.segment_count, 2);
        // A two-patch axis spans the table end to end: patch (0,0) reads the
        // first corner cell, patch (1,1) the last.
        assert_eq!(&control.position_table_indices[..4], [0, 1, 48, 49]);
        assert_eq!(&control.position_table_weights[..4], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(&control.position_table_indices[12..16], [47 * 48 + 47; 4]);
    }

    #[test]
    fn the_default_envelope_is_the_references() {
        assert_eq!(Vision::default().max_tokens(), 32_768);
    }

    #[test]
    fn an_envelope_outside_the_range_is_refused() {
        assert_eq!(Vision::new(0), Err(VisionEnvelopeOutOfRange(0)));
        assert!(Vision::new(VISION_MAX_TOKENS_LIMIT + 1).is_err());
        assert_eq!(Vision::new(1).unwrap().max_tokens(), 1);
        assert_eq!(
            Vision::new(VISION_MAX_TOKENS_LIMIT).unwrap().max_tokens(),
            VISION_MAX_TOKENS_LIMIT
        );
    }

    #[test]
    fn the_envelope_is_capped_by_the_context() {
        let vision = Vision::default();
        assert_eq!(vision.envelope_tokens(40_960), 32_768);
        assert_eq!(vision.envelope_tokens(1024), 1024);
    }

    #[test]
    fn the_output_transient_is_one_items_bf16_hidden_columns() {
        // 5120 x 32768 x 2 bytes, already a multiple of 256.
        assert_eq!(Vision::default().output_transient_bytes(262_144), 335_544_320);
        assert_eq!(Vision::new(1).unwrap().output_transient_bytes(4096), 10_240);
    }

    #[test]
    fn the_vision_object_count_is_the_artifacts() {
        assert_eq!(VISION_OBJECTS, 333);
    }
}
