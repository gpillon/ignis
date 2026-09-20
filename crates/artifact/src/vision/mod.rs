//! The vision processor (GitHub #176, spec `.scratch/vision/specs/01-image-input.md`
//! §Processor): message text rendered by the chat template plus image bytes
//! become a [`PreparedPrompt`] — token ids, per-token modality, three-axis
//! positions, `rope_delta`, and one [`MediaItem`] per image (grid, token
//! span, BF16 patch rows, SHA-256 content digest).
//!
//! The output is exactly the reference processor's (ninfer
//! `targets/qwen3_6/impl/frontend/processor.cpp`) on the same input, checked
//! against fixtures recorded from it (`tests/vision_processor.rs`,
//! `tools/vision-fixtures`). The algorithm is the reference's step for step:
//! decode to RGB8 ([`decode`]), smart resize and antialiased bicubic
//! ([`resize`]), BF16 patch packing ([`patches`]), placeholder expansion and
//! MRoPE positions ([`layout`]).
//!
//! The processor is split where a media cache will sit: [`VisionProcessor::prepare_media`]
//! is the per-image unit keyed by content digest, [`VisionProcessor::prepare_prompt`]
//! lays the prepared items out in the rendered text. [`VisionProcessor::prepare`]
//! runs both in the reference's order.

mod decode;
mod jpeg;
pub mod layout;
mod patches;
mod resize;

use std::fmt;

use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};

use crate::frontend::{ChatMessage, ChatRenderOptions, FrontendSet, Tokenizer};

pub use decode::{decode_image, DecodeFailure};
pub use resize::{smart_resize, Size};

/// Side of one vision patch, in pixels.
pub const PATCH: usize = 16;
/// Frames per temporal patch (an image repeats its one frame).
pub const TEMPORAL: usize = 2;
/// Patches per merged token along each spatial axis.
pub const MERGE: usize = 2;
/// Resize factor: a merged token covers `FACTOR x FACTOR` pixels.
pub const FACTOR: usize = PATCH * MERGE;
/// Width of one patch row, `3 x TEMPORAL x PATCH x PATCH`.
pub const PATCH_FEATURES: usize = 3 * TEMPORAL * PATCH * PATCH;
/// The rendered image placeholder.
pub const IMAGE_PAD: &str = "<|image_pad|>";
/// The rendered video placeholder (no video input yet; refused on sight).
pub const VIDEO_PAD: &str = "<|video_pad|>";
/// The model contract's image placeholder token id.
pub const IMAGE_PAD_ID: u32 = 248_056;
/// The model contract's video placeholder token id.
pub const VIDEO_PAD_ID: u32 = 248_057;
/// The reference's per-request vision envelope, in merged tokens.
pub const MAX_VISION_TOKENS: u64 = 32_768;

/// A decoded image: `width x height` RGB8 pixels, row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rgb8Image {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

/// A media item's patch grid (temporal, height, width), in patches — not
/// merged tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub t: u32,
    pub h: u32,
    pub w: u32,
}

impl Grid {
    /// Raw patches (patch rows) the grid holds.
    pub fn raw_patches(&self) -> u64 {
        self.t as u64 * self.h as u64 * self.w as u64
    }

    /// Merged vision tokens the grid expands to.
    pub fn vision_tokens(&self) -> u64 {
        self.raw_patches() / (MERGE * MERGE) as u64
    }
}

/// A run of prompt tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenSpan {
    pub begin: usize,
    pub count: usize,
}

/// One image after decode, resize and packing — the unit a media cache
/// keys by content digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedMedia {
    pub grid: Grid,
    /// Row-major `[raw_patches, PATCH_FEATURES]` BF16 bits.
    pub patches: Vec<u16>,
    /// SHA-256 of the acquired (encoded) bytes.
    pub content_digest: [u8; 32],
    /// Size of the acquired bytes.
    pub encoded_bytes: usize,
    /// The **submitted** image's pixel size, `(width, height)`, before
    /// `smart_resize` (GitHub #242).
    ///
    /// Not recoverable from `grid`, which describes the resized image in
    /// patches and has lost both the original scale and, to rounding, its
    /// exact aspect ratio. A **point** answers in pixels of the image the
    /// caller sent — that is the whole reason the endpoint does the
    /// rescaling rather than handing a caller a 0-999 pair and letting them
    /// get a non-square image wrong once.
    pub source_pixels: (u32, u32),
}

/// A prepared image placed in the prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaItem {
    pub grid: Grid,
    pub token_span: TokenSpan,
    /// Row-major `[raw_patches, PATCH_FEATURES]` BF16 bits.
    pub patches: Vec<u16>,
    pub content_digest: [u8; 32],
}

/// What the model is fed for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedPrompt {
    pub token_ids: Vec<u32>,
    /// Per token: [`layout::TEXT`], [`layout::IMAGE`] or [`layout::VIDEO`].
    pub token_types: Vec<u8>,
    /// Axis-major `[3, T]`: temporal, height, width.
    pub positions: Vec<i32>,
    /// `max_position + 1 - T`; decode rotates at `position + rope_delta`.
    pub rope_delta: i32,
    pub media: Vec<MediaItem>,
    /// For each byte boundary the caller passed: the token frontier it
    /// lands on, or `None` when the rendered prefix up to it does not
    /// tokenize to an exact prefix of `token_ids`.
    pub frontiers: Vec<Option<u32>>,
}

impl PreparedPrompt {
    /// One position axis (0 temporal, 1 height, 2 width).
    pub fn position_axis(&self, axis: usize) -> &[i32] {
        let length = self.token_ids.len();
        &self.positions[axis * length..(axis + 1) * length]
    }

    /// Merged vision tokens across every media item.
    pub fn vision_tokens(&self) -> u64 {
        self.media.iter().map(|m| m.grid.vision_tokens()).sum()
    }
}

/// Per-request limits, defaulting to the reference's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessorOptions {
    /// Lower pixel bound of a resized image (`size.shortest_edge`).
    pub min_pixels: u64,
    /// Upper pixel bound of a resized image (`size.longest_edge`).
    pub max_pixels: u64,
    /// Encoded bytes across a request's media.
    pub max_encoded_media_bytes: u64,
    /// Decoded pixels of one image, before resize.
    pub max_decoded_pixels: u64,
    /// Raw patches across a request.
    pub max_raw_patches: u64,
    /// Merged vision tokens across a request.
    pub max_vision_tokens: u64,
}

impl ProcessorOptions {
    /// The reference's limits with the pixel bounds read from the artifact's
    /// `preprocessor_config.json` (`size.shortest_edge` / `size.longest_edge`
    /// are pixel counts for this processor, not edge lengths).
    pub fn from_preprocessor_config(config: &[u8]) -> Result<Self, ProcessorError> {
        let invalid = |what: &str| ProcessorError::InvalidConfig(format!("preprocessor_config.json: {what}"));
        let json: JsonValue = serde_json::from_slice(config).map_err(|e| invalid(&e.to_string()))?;
        for (key, expected) in [("patch_size", PATCH), ("temporal_patch_size", TEMPORAL), ("merge_size", MERGE)] {
            if json.get(key).and_then(JsonValue::as_u64) != Some(expected as u64) {
                return Err(invalid(&format!("{key} must be {expected}")));
            }
        }
        let bound = |key: &str| {
            json.get("size")
                .and_then(|size| size.get(key))
                .and_then(JsonValue::as_u64)
                .filter(|&v| v > 0)
                .ok_or_else(|| invalid(&format!("size.{key} must be a positive integer")))
        };
        let (min_pixels, max_pixels) = (bound("shortest_edge")?, bound("longest_edge")?);
        if max_pixels < min_pixels {
            return Err(invalid("size.longest_edge is below size.shortest_edge"));
        }
        Ok(Self {
            min_pixels,
            max_pixels,
            max_encoded_media_bytes: 256 << 20,
            max_decoded_pixels: 64 * 1024 * 1024,
            max_raw_patches: MAX_VISION_TOKENS * (MERGE * MERGE) as u64,
            max_vision_tokens: MAX_VISION_TOKENS,
        })
    }
}

/// Why an image could not be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidMedia {
    /// The longer side is more than 200 times the shorter one.
    AspectRatio { width: u32, height: u32 },
    /// The bytes are not a well-formed image of a recognised format.
    Undecodable(String),
    /// A recognised container the processor has no exact decoder for.
    Unsupported(String),
}

/// A per-request media limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Budget {
    /// Media items: more than even the smallest grids could fit.
    MediaItems,
    EncodedBytes,
    DecodedPixels,
    RawPatches,
    VisionTokens,
}

impl Budget {
    /// What the limit counts, as the error message names it.
    pub fn name(self) -> &'static str {
        match self {
            Self::MediaItems => "media items",
            Self::EncodedBytes => "media bytes",
            Self::DecodedPixels => "decoded pixels",
            Self::RawPatches => "vision raw patches",
            Self::VisionTokens => "vision tokens",
        }
    }
}

/// A typed processor failure. Every variant is a client-visible 400; see
/// [`ProcessorError::code`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessorError {
    /// Media item `item` (0-based, in prompt order) is unusable.
    InvalidMedia { item: usize, reason: InvalidMedia },
    /// A request exceeds one of its media limits.
    BudgetExceeded { budget: Budget, limit: u64, requested: u64 },
    /// The rendered placeholders and the media items do not line up.
    PlaceholderMismatch(&'static str),
    /// A tracked byte boundary falls inside a placeholder.
    BoundaryInsidePlaceholder,
    /// The tokenizer does not map a vision placeholder to the model
    /// contract's id (a load failure).
    PadTokenMismatch { token: &'static str, expected: u32, actual: Vec<u32> },
    /// The artifact's `preprocessor_config.json` is unusable (a load
    /// failure).
    InvalidConfig(String),
    /// The tokenizer failed on the expanded text.
    Tokenize(String),
    /// The chat template refused the conversation (e.g. an image in a
    /// system message).
    Render(String),
}

impl ProcessorError {
    /// The wire error code: `media_budget_exceeded` for a limit, and
    /// `invalid_media` for everything else.
    pub fn code(&self) -> &'static str {
        match self {
            Self::BudgetExceeded { .. } => "media_budget_exceeded",
            _ => "invalid_media",
        }
    }
}

impl fmt::Display for ProcessorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMedia { item, reason } => match reason {
                InvalidMedia::AspectRatio { width, height } => {
                    write!(f, "media item {item}: image aspect ratio must be at most 200 ({width}x{height})")
                }
                InvalidMedia::Undecodable(why) => write!(f, "media item {item}: cannot decode image: {why}"),
                InvalidMedia::Unsupported(why) => write!(f, "media item {item}: unsupported image: {why}"),
            },
            Self::BudgetExceeded { budget, limit, requested } => {
                write!(f, "{} exceed the request media budget ({requested} > {limit})", budget.name())
            }
            Self::PlaceholderMismatch(why) => write!(f, "{why}"),
            Self::BoundaryInsidePlaceholder => write!(f, "a prompt boundary intersects a media placeholder"),
            Self::PadTokenMismatch { token, expected, actual } => {
                write!(f, "tokenizer maps {token} to {actual:?}, the model contract requires [{expected}]")
            }
            Self::InvalidConfig(why) => write!(f, "{why}"),
            Self::Tokenize(why) => write!(f, "tokenize: {why}"),
            Self::Render(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for ProcessorError {}

fn check_budget(budget: Budget, limit: u64, requested: u64) -> Result<(), ProcessorError> {
    if requested > limit {
        return Err(ProcessorError::BudgetExceeded { budget, limit, requested });
    }
    Ok(())
}

/// Prepares image prompts for one artifact frontend. Built once at load:
/// construction validates the tokenizer's placeholder ids and reads the
/// pixel bounds.
#[derive(Debug, Clone)]
pub struct VisionProcessor {
    options: ProcessorOptions,
}

impl FrontendSet {
    /// The vision processor for this frontend, with the reference's limits
    /// (GitHub #176). Fails when the tokenizer's placeholder ids are not the
    /// model contract's or the preprocessor config is unusable.
    pub fn vision_processor(&self) -> Result<VisionProcessor, ProcessorError> {
        VisionProcessor::new(self.tokenizer(), ProcessorOptions::from_preprocessor_config(self.preprocessor_config())?)
    }

    /// Messages plus image bytes to a prepared prompt: render through the
    /// chat template (each image part becomes one placeholder), then
    /// [`VisionProcessor::prepare`]. `images` are the bytes of the image
    /// parts in prompt order.
    pub fn prepare_prompt(
        &self,
        processor: &VisionProcessor,
        messages: &[ChatMessage],
        images: &[&[u8]],
        options: ChatRenderOptions,
        tools: Option<&[JsonValue]>,
    ) -> Result<PreparedPrompt, ProcessorError> {
        let rendered = self
            .chat_template()
            .render_with_thinking_and_tools(messages, options, tools)
            .map_err(|e| ProcessorError::Render(e.to_string()))?;
        processor.prepare(self.tokenizer(), &rendered, images, &[])
    }
}

impl VisionProcessor {
    /// A processor over `tokenizer` with explicit limits.
    pub fn new(tokenizer: &Tokenizer, options: ProcessorOptions) -> Result<Self, ProcessorError> {
        for (token, expected) in [(IMAGE_PAD, IMAGE_PAD_ID), (VIDEO_PAD, VIDEO_PAD_ID)] {
            let actual = tokenizer.encode(token).map_err(|e| ProcessorError::Tokenize(e.to_string()))?;
            if actual != [expected] {
                return Err(ProcessorError::PadTokenMismatch { token, expected, actual });
            }
        }
        Ok(Self { options })
    }

    pub fn options(&self) -> &ProcessorOptions {
        &self.options
    }

    /// Decode, resize and pack media item `item`.
    pub fn prepare_media(&self, item: usize, bytes: &[u8]) -> Result<PreparedMedia, ProcessorError> {
        let options = &self.options;
        check_budget(Budget::EncodedBytes, options.max_encoded_media_bytes, bytes.len() as u64)?;
        let content_digest: [u8; 32] = Sha256::digest(bytes).into();
        let image = decode::decode_image(bytes, options.max_decoded_pixels).map_err(|failure| match failure {
            decode::DecodeFailure::Invalid(reason) => ProcessorError::InvalidMedia { item, reason },
            decode::DecodeFailure::Pixels(requested) => ProcessorError::BudgetExceeded {
                budget: Budget::DecodedPixels,
                limit: options.max_decoded_pixels,
                requested,
            },
        })?;
        let (source_width, source_height) = (image.width, image.height);
        let size = resize::smart_resize(image.height, image.width, options.min_pixels, options.max_pixels)
            .map_err(|error| match error {
                resize::ResizeError::AspectRatio => ProcessorError::InvalidMedia {
                    item,
                    reason: InvalidMedia::AspectRatio { width: image.width, height: image.height },
                },
                resize::ResizeError::InvalidConfiguration => {
                    ProcessorError::InvalidConfig("invalid image resize configuration".to_owned())
                }
            })?;
        let grid = Grid { t: 1, h: size.height / PATCH as u32, w: size.width / PATCH as u32 };
        check_budget(Budget::RawPatches, options.max_raw_patches, grid.raw_patches())?;
        check_budget(Budget::VisionTokens, options.max_vision_tokens, grid.vision_tokens())?;
        let resized = resize::resize_bicubic(image, size);
        Ok(PreparedMedia {
            grid,
            source_pixels: (source_width, source_height),
            patches: patches::pack_patches(&resized),
            content_digest,
            encoded_bytes: bytes.len(),
        })
    }

    /// Lay prepared media out in `rendered` (the chat template's output,
    /// one `<|image_pad|>` per image in order): expand placeholders,
    /// tokenize, assign positions. `boundaries` are byte offsets into
    /// `rendered` whose token frontiers the caller wants back.
    pub fn prepare_prompt(
        &self,
        tokenizer: &Tokenizer,
        rendered: &str,
        media: Vec<PreparedMedia>,
        boundaries: &[usize],
    ) -> Result<PreparedPrompt, ProcessorError> {
        let options = &self.options;
        check_budget(
            Budget::EncodedBytes,
            options.max_encoded_media_bytes,
            media.iter().map(|m| m.encoded_bytes as u64).sum(),
        )?;
        check_budget(Budget::RawPatches, options.max_raw_patches, media.iter().map(|m| m.grid.raw_patches()).sum())?;
        check_budget(
            Budget::VisionTokens,
            options.max_vision_tokens,
            media.iter().map(|m| m.grid.vision_tokens()).sum(),
        )?;
        let grids: Vec<Grid> = media.iter().map(|m| m.grid).collect();
        let expanded = layout::expand_placeholders(rendered, &grids, boundaries)?;
        let encode = |text: &str| tokenizer.encode(text).map_err(|e| ProcessorError::Tokenize(e.to_string()));
        let token_ids = encode(&expanded.text)?;
        let frontiers = expanded
            .boundaries
            .iter()
            .map(|&boundary| {
                if boundary > expanded.text.len() {
                    return Ok(None);
                }
                let prefix = encode(&expanded.text[..boundary])?;
                let exact = !prefix.is_empty() && token_ids.starts_with(&prefix);
                Ok(exact.then_some(prefix.len() as u32))
            })
            .collect::<Result<_, ProcessorError>>()?;
        let token_types = layout::token_types(&token_ids);
        let (positions, spans, rope_delta) = layout::assign_positions(&token_types, &grids)?;
        let media = media
            .into_iter()
            .zip(spans)
            .map(|(m, token_span)| MediaItem {
                grid: m.grid,
                token_span,
                patches: m.patches,
                content_digest: m.content_digest,
            })
            .collect();
        Ok(PreparedPrompt { token_ids, token_types, positions, rope_delta, media, frontiers })
    }

    /// [`Self::prepare_media`] for every image in prompt order, then
    /// [`Self::prepare_prompt`], with the reference's request-level checks
    /// first: an item count no minimum grid could fit, and the aggregate
    /// encoded bytes.
    pub fn prepare(
        &self,
        tokenizer: &Tokenizer,
        rendered: &str,
        images: &[&[u8]],
        boundaries: &[usize],
    ) -> Result<PreparedPrompt, ProcessorError> {
        let options = &self.options;
        let max_items = (options.max_raw_patches / (MERGE * MERGE) as u64).min(options.max_vision_tokens);
        check_budget(Budget::MediaItems, max_items, images.len() as u64)?;
        check_budget(
            Budget::EncodedBytes,
            options.max_encoded_media_bytes,
            images.iter().map(|b| b.len() as u64).sum(),
        )?;
        let media = images
            .iter()
            .enumerate()
            .map(|(item, bytes)| self.prepare_media(item, bytes))
            .collect::<Result<Vec<_>, _>>()?;
        self.prepare_prompt(tokenizer, rendered, media, boundaries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A word-level tokenizer whose placeholders map to `image` / `video`.
    fn tokenizer(image: u32, video: u32) -> Tokenizer {
        let added = |id: u32, content: &str| {
            serde_json::json!({"id": id, "content": content, "single_word": false, "lstrip": false,
                               "rstrip": false, "normalized": false, "special": true})
        };
        let json = serde_json::json!({
            "version": "1.0",
            "added_tokens": [added(image, IMAGE_PAD), added(video, VIDEO_PAD)],
            "pre_tokenizer": {"type": "Whitespace"},
            "model": {"type": "WordLevel", "vocab": {"x": 0, IMAGE_PAD: image, VIDEO_PAD: video}, "unk_token": "x"},
        });
        Tokenizer::from_bytes(json.to_string().as_bytes()).unwrap()
    }

    fn options() -> ProcessorOptions {
        ProcessorOptions {
            min_pixels: 32 * 32,
            max_pixels: 1 << 20,
            max_encoded_media_bytes: 1 << 20,
            max_decoded_pixels: 1 << 20,
            max_raw_patches: 1 << 16,
            max_vision_tokens: 1 << 14,
        }
    }

    fn processor(options: ProcessorOptions) -> (Tokenizer, VisionProcessor) {
        let tokenizer = tokenizer(IMAGE_PAD_ID, VIDEO_PAD_ID);
        let processor = VisionProcessor::new(&tokenizer, options).unwrap();
        (tokenizer, processor)
    }

    /// A flat RGB PNG.
    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, width, height);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&vec![100u8; (width * height * 3) as usize]).unwrap();
        }
        out
    }

    #[test]
    fn a_tokenizer_with_the_wrong_pad_id_is_refused_at_construction() {
        let error = VisionProcessor::new(&tokenizer(7, VIDEO_PAD_ID), options()).unwrap_err();
        assert_eq!(error, ProcessorError::PadTokenMismatch { token: IMAGE_PAD, expected: IMAGE_PAD_ID, actual: vec![7] });
        assert_eq!(error.code(), "invalid_media");
        let error = VisionProcessor::new(&tokenizer(IMAGE_PAD_ID, 9), options()).unwrap_err();
        assert!(matches!(error, ProcessorError::PadTokenMismatch { token: VIDEO_PAD, .. }), "{error}");
    }

    #[test]
    fn an_image_prompt_expands_tokenizes_and_positions() {
        let (tokenizer, processor) = processor(options());
        // 64x64 is on the grid: a 4x4 patch grid, 2x2 merged = 4 tokens.
        let image = png(64, 64);
        let prompt = processor.prepare(&tokenizer, &format!("x {IMAGE_PAD} x"), &[&image], &[]).unwrap();
        assert_eq!(prompt.token_ids, [0, IMAGE_PAD_ID, IMAGE_PAD_ID, IMAGE_PAD_ID, IMAGE_PAD_ID, 0]);
        assert_eq!(prompt.media.len(), 1);
        assert_eq!(prompt.media[0].grid, Grid { t: 1, h: 4, w: 4 });
        assert_eq!(prompt.media[0].token_span, TokenSpan { begin: 1, count: 4 });
        assert_eq!(prompt.media[0].patches.len(), 16 * PATCH_FEATURES);
        assert_eq!(prompt.media[0].content_digest, <[u8; 32]>::from(Sha256::digest(&image)));
        assert_eq!(prompt.position_axis(1), [0, 1, 1, 2, 2, 3]);
        assert_eq!(prompt.position_axis(2), [0, 1, 2, 1, 2, 3]);
        assert_eq!(prompt.rope_delta, 3 + 1 - 6);
        assert_eq!(prompt.vision_tokens(), 4);
    }

    #[test]
    fn placeholder_count_and_order_mismatches_are_typed_errors() {
        let (tokenizer, processor) = processor(options());
        let image = png(64, 64);
        for rendered in [format!("{IMAGE_PAD} {IMAGE_PAD}"), "x".to_owned(), format!("{VIDEO_PAD} {IMAGE_PAD}")] {
            let error = processor.prepare(&tokenizer, &rendered, &[&image], &[]).unwrap_err();
            assert!(matches!(error, ProcessorError::PlaceholderMismatch(_)), "{rendered}: {error}");
            assert_eq!(error.code(), "invalid_media");
        }
    }

    #[test]
    fn bad_aspect_and_undecodable_bytes_name_their_item() {
        let (tokenizer, processor) = processor(options());
        let (good, strip) = (png(64, 64), png(201, 1));
        let rendered = format!("{IMAGE_PAD} {IMAGE_PAD}");
        let error = processor.prepare(&tokenizer, &rendered, &[&good, &strip], &[]).unwrap_err();
        assert_eq!(
            error,
            ProcessorError::InvalidMedia { item: 1, reason: InvalidMedia::AspectRatio { width: 201, height: 1 } }
        );
        let error = processor.prepare(&tokenizer, &rendered, &[b"\x89PNG\r\n\x1a\ngarbage", &good], &[]).unwrap_err();
        assert!(
            matches!(error, ProcessorError::InvalidMedia { item: 0, reason: InvalidMedia::Undecodable(_) }),
            "{error}"
        );
        let error = processor.prepare(&tokenizer, &rendered, &[&good, b"BM not supported"], &[]).unwrap_err();
        assert!(
            matches!(error, ProcessorError::InvalidMedia { item: 1, reason: InvalidMedia::Unsupported(_) }),
            "{error}"
        );
        assert_eq!(error.code(), "invalid_media");
    }

    #[test]
    fn every_budget_is_a_media_budget_exceeded_error() {
        let image = png(64, 64);
        let rendered = format!("{IMAGE_PAD}");
        let cases: [(fn(&mut ProcessorOptions), Budget); 4] = [
            (|o| o.max_encoded_media_bytes = 10, Budget::EncodedBytes),
            (|o| o.max_decoded_pixels = 64 * 64 - 1, Budget::DecodedPixels),
            (|o| o.max_raw_patches = 15, Budget::RawPatches),
            (|o| o.max_vision_tokens = 3, Budget::VisionTokens),
        ];
        for (shrink, budget) in cases {
            let mut limits = options();
            shrink(&mut limits);
            let (tokenizer, processor) = processor(limits);
            let error = processor.prepare(&tokenizer, &rendered, &[&image], &[]).unwrap_err();
            assert!(matches!(error, ProcessorError::BudgetExceeded { budget: b, .. } if b == budget), "{error}");
            assert_eq!(error.code(), "media_budget_exceeded");
        }
    }

    #[test]
    fn budgets_are_aggregated_across_the_request() {
        let mut limits = options();
        limits.max_vision_tokens = 6;
        let (tokenizer, processor) = processor(limits);
        let image = png(64, 64);
        let rendered = format!("{IMAGE_PAD} {IMAGE_PAD}");
        let error = processor.prepare(&tokenizer, &rendered, &[&image, &image], &[]).unwrap_err();
        assert_eq!(error, ProcessorError::BudgetExceeded { budget: Budget::VisionTokens, limit: 6, requested: 8 });
    }

    #[test]
    fn boundaries_come_back_as_token_frontiers_across_the_expansion() {
        let (tokenizer, processor) = processor(options());
        let image = png(64, 64);
        let rendered = format!("x {IMAGE_PAD} x x");
        let after_image = rendered.find(" x x").unwrap();
        let prompt = processor.prepare(&tokenizer, &rendered, &[&image], &[1, after_image]).unwrap();
        assert_eq!(prompt.frontiers, [Some(1), Some(5)]);
        let inside = rendered.find(IMAGE_PAD).unwrap() + 2;
        let error = processor.prepare(&tokenizer, &rendered, &[&image], &[inside]).unwrap_err();
        assert_eq!(error, ProcessorError::BoundaryInsidePlaceholder);
    }

    #[test]
    fn preprocessor_config_gives_the_pixel_bounds_and_rejects_another_geometry() {
        let config = br#"{"size": {"longest_edge": 16777216, "shortest_edge": 65536},
                          "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2}"#;
        let options = ProcessorOptions::from_preprocessor_config(config).unwrap();
        assert_eq!((options.min_pixels, options.max_pixels), (65_536, 16_777_216));
        assert_eq!(options.max_vision_tokens, MAX_VISION_TOKENS);
        let other = br#"{"size": {"longest_edge": 100, "shortest_edge": 10}, "patch_size": 14,
                         "temporal_patch_size": 2, "merge_size": 2}"#;
        assert!(matches!(ProcessorOptions::from_preprocessor_config(other), Err(ProcessorError::InvalidConfig(_))));
        let missing = br#"{"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2}"#;
        assert!(matches!(ProcessorOptions::from_preprocessor_config(missing), Err(ProcessorError::InvalidConfig(_))));
    }
}
