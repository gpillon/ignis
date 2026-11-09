//! Prompt reuse with an image on Flash-Next, at the leaf seam (GitHub #559,
//! spec #555; ADR 0029 as amended for multimodal prompts, ADR 0087): a
//! sequence holding an image carries what its image put in its state through
//! a checkpoint claim, a KV-RAM restore and a spill, so what it computes
//! afterwards is what a cold prefill split at the same boundary computes, bit
//! for bit.
//!
//! What has to travel, beside the pages:
//!
//! - **the rope delta**, in the progress image a snapshot and a spill carry.
//!   A checkpoint's capture zeroes it and its claimant takes its own from the
//!   span it prefills, as on the 27B (`ignis_seq_capture_state`);
//! - **the n-gram context**, in the block ahead of every blob and beside every
//!   checkpoint;
//! - **an incomplete indexer block's first token's three axes**, in that
//!   block's key slot (GitHub #558, `indexer.h` `AxisRecord`): a frontier
//!   inside a block begun on an image token leaves the block's key to the call
//!   after it, which rotates the key at that token's axes only if they came
//!   along with the page.
//!
//! One vision load (the `host` residency, the RTX 5090 profile's; hq-e8-2b,
//! the serving format), one image (a 16 x 16 merged grid of synthetic patches,
//! encoded once by the vision companion's tower and placed by every span over
//! it), one prompt: 2,302 text tokens, the image's 256 placeholders, 300 text
//! tokens. Two frontiers, each three tokens or one into a block begun on an
//! image token -- the `image_across_blocks` shape of
//! `kernel/tests/fixtures/flash_next_mrope/` -- past the indexer's dense range,
//! so every row after it selects blocks by their keys, and inside a KV page,
//! so a checkpoint carries a partial tail page:
//!
//! - 2,401, inside the image: the block 2,400..2,404 is image tokens only;
//! - 2,559, at its edge: the block 2,556..2,560 begins on the image's
//!   next-to-last placeholder and ends in text.
//!
//! Per frontier, against the prompt prefilled cold in two spans cut there --
//! its last logits, eight greedy rounds, and the logits of the token after
//! them prefilled as text, which rotates at its position plus the rope delta:
//!
//! - the sequence that captured the checkpoint, going on;
//! - the checkpoint claimed from a host and a device retained slot;
//! - the checkpoint through KV-RAM;
//! - a sequence spilled at the frontier and restored;
//! - a sequence spilled mid-decode and restored, whose rounds after it rotate
//!   by the rope delta its blob carried and nothing else.
//!
//! The reference runs again at the end, on the expert cache all of that
//! warmed: the two agree too.
//!
//! ```text
//! cargo test -p ignis-runtime --features cuda --test flash_next_vision_reuse_gpu -- --ignored --nocapture
//! ```
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` with its vision
//! companion (or `IGNIS_FLASH_NEXT_DIR`). Explicit GPU profile (ADR 0006):
//! outside `IGNIS_GPU_PROFILE=1` a missing artifact or GPU is a skip, under it
//! a failure. Needs ~40 GB of free RAM: the pinned expert pool, the tower and
//! the embedding pool, two host retained slots and a 1 GiB arena.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::vision::layout::{assign_positions, token_types};
use ignis_artifact::vision::{IMAGE_PAD_ID, PATCH_FEATURES};
use ignis_core::compute::ModelConfig;
use ignis_core::flash_next::EngineOptions;
use ignis_core::gpu_profile;
use ignis_core::vision::{Grid, MediaItem, Multimodal};
use ignis_core::{DecodeParams, KvFormat, Vision, VisionResidency};
use ignis_runtime::{
    DecodeLane, FlashNextLeaf, FlashNextModel, FlashNextSequence, MultimodalSpan, SpanMedia, StepLeaf, KV_PAGE_TOKENS,
};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MAX_CONTEXT: u32 = 8192;
/// Text before the image: past the dense range, and two tokens into a block,
/// so the image's own blocks begin two placeholders in.
const HEAD: usize = 2_302;
/// The image's merged grid side: 256 placeholders.
const SIDE: u32 = 16;
/// Text after the image.
const TAIL: usize = 300;
/// The frontiers, each in a block begun on an image token.
const FRONTIERS: [(usize, &str); 2] = [(2_401, "inside the image"), (2_559, "at the image's edge")];
/// Greedy rounds after the prompt.
const DECODED: usize = 8;
/// Retained slots: 0 on the device, 1 and 2 on the host.
const DEVICE_SLOT: u32 = 0;
const HOST_SLOT: u32 = 1;

type Media = <FlashNextLeaf as StepLeaf>::Media;

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

/// A vision load with prompt reuse's slots and arena, or `None` through
/// `skip_or_fail` when the artifact, its companion or the GPU is not there.
fn leaf() -> Option<FlashNextLeaf> {
    let path = model_dir().join(ARTIFACT_FILE_NAME);
    if !path.exists() {
        gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display()));
        return None;
    }
    let vision = Vision::new(MAX_CONTEXT).expect("an envelope").with_residency(VisionResidency::Host);
    let options = EngineOptions {
        max_context_tokens: MAX_CONTEXT,
        kv_format: KvFormat::HqE8_2b,
        retained_device_slots: 1,
        retained_host_slots: 2,
        kv_ram_arena_bytes: 1 << 30,
        vision: Some(vision),
        ..EngineOptions::default()
    };
    match FlashNextLeaf::open(&path, options) {
        Ok(leaf) => Some(leaf),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("open the Flash-Next leaf with vision: {e}"));
            None
        }
    }
}

/// Deterministic ordinary token ids (the low vocabulary is text).
fn text(len: usize, salt: u32) -> Vec<u32> {
    (0..len as u32).map(|i| 1000 + (i * 7919 + salt * 104_729) % 60_000).collect()
}

/// The prompt's ids and its multimodal part: the image's placeholders at
/// [`HEAD`], laid out by the processor's own rule (`assign_positions`: text on
/// all three axes, the image at its start plus its row and column, the text
/// after it past the longer side), and patches of synthetic pixels in the
/// tower's value range.
fn prompt() -> (Vec<u32>, Multimodal) {
    let count = (SIDE * SIDE) as usize;
    let tokens: Vec<u32> = [text(HEAD, 1), vec![IMAGE_PAD_ID; count], text(TAIL, 2)].concat();
    let grid = Grid { t: 1, h: 2 * SIDE, w: 2 * SIDE };
    let (positions, spans, rope_delta) = assign_positions(&token_types(&tokens), &[grid]).expect("the layout");
    let values = grid.raw_patches() as usize * PATCH_FEATURES;
    let patches = (0..values as u32)
        .map(|i| {
            let unit = (i.wrapping_mul(2_654_435_761) >> 8) as f32 / (1u32 << 24) as f32;
            ((unit * 2.0 - 1.0).to_bits() >> 16) as u16
        })
        .collect();
    let media = vec![MediaItem { grid, token_span: spans[0], patches, content_digest: [0x59; 32] }];
    (tokens, Multimodal { positions, rope_delta, media })
}

/// What a request ends with: the last prefill's logits, the tokens greedy
/// rounds commit after it, and the logits of the token after those prefilled
/// as text -- bit patterns.
#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    logits: Vec<u32>,
    tokens: Vec<u32>,
    probe: Vec<u32>,
}

struct Run<'a> {
    leaf: &'a FlashNextLeaf,
    model: &'a FlashNextModel,
    tokens: &'a [u32],
    multimodal: &'a Multimodal,
    media: &'a Media,
}

impl Run<'_> {
    fn fresh(&self) -> FlashNextSequence {
        self.leaf.allocate_sequence(self.model, MAX_CONTEXT).expect("a free lane")
    }

    fn logits(&self) -> Vec<f32> {
        vec![0f32; self.leaf.vocab(self.model) as usize]
    }

    /// Prefill the prompt's `[start, end)` as a span of a multimodal prompt
    /// -- its three axes, the rope delta, and the image's columns over the
    /// placeholders it covers -- returning its last logits.
    fn prefill(&self, seq: &mut FlashNextSequence, start: usize, end: usize) -> Vec<u32> {
        let len = end - start;
        let positions = self.multimodal.span_positions(start, len);
        let chunk = self.multimodal.chunk_media(start as u32, len as u32);
        let media = chunk.as_ref().map(|chunk| SpanMedia {
            embedding: self.media,
            first_column: chunk.first_column,
            scatter_indices: &chunk.scatter_indices,
        });
        let span = MultimodalSpan { positions: &positions, rope_delta: self.multimodal.rope_delta, media };
        let mut logits = self.logits();
        self.leaf
            .prefill_multimodal(
                self.model,
                seq,
                &self.tokens[start..end],
                start as u32,
                0,
                None,
                false,
                DecodeParams::default(),
                &[],
                None,
                span,
                Some(&mut logits),
                None,
            )
            .unwrap_or_else(|code| panic!("prefill of {start}..{end}: leaf code {code}"));
        logits.iter().map(|v| v.to_bits()).collect()
    }

    fn decode(&self, seq: &mut FlashNextSequence, rounds: usize) -> Vec<u32> {
        let lane = DecodeLane {
            params: DecodeParams::default(),
            remaining_tokens: 1,
            stop_ids: &[],
            permitted: &[],
            reasoning_close: None,
        };
        (0..rounds)
            .map(|round| {
                let runs = self
                    .leaf
                    .decode(self.model, &mut [&mut *seq], std::slice::from_ref(&lane))
                    .unwrap_or_else(|code| panic!("decode round {round}: leaf code {code}"));
                runs[0].tokens[0]
            })
            .collect()
    }

    /// The token the last round drew, prefilled as text at the frontier past
    /// the prompt and [`DECODED`] rounds: it rotates at its position plus the
    /// sequence's rope delta, so its logits read the delta the sequence holds.
    fn probe(&self, seq: &mut FlashNextSequence) -> Vec<u32> {
        let pending = self.leaf.pending_token(seq).expect("a pending token");
        let position = (self.tokens.len() + DECODED) as u32;
        let mut logits = self.logits();
        self.leaf
            .prefill(self.model, seq, &[pending], position, 0, None, false, DecodeParams::default(), &[], None, Some(&mut logits), None)
            .unwrap_or_else(|code| panic!("the probe at {position}: leaf code {code}"));
        logits.iter().map(|v| v.to_bits()).collect()
    }

    /// The prompt from `frontier` on, the rounds and the probe, on a
    /// sequence standing at `frontier`.
    fn finish(&self, mut seq: FlashNextSequence, frontier: usize) -> Outcome {
        let logits = self.prefill(&mut seq, frontier, self.tokens.len());
        let tokens = self.decode(&mut seq, DECODED);
        let probe = self.probe(&mut seq);
        self.leaf.release_sequence(self.model, seq);
        Outcome { logits, tokens, probe }
    }

    /// Put a blob of `bytes` in the arena and fill it with `write`.
    fn blob(&self, bytes: u64, write: impl FnOnce(&mut [u8]) -> Result<(), i32>) -> ignis_core::seq::ArenaBuffer {
        assert!(self.leaf.host_blob_fits(bytes), "a {bytes}-byte blob fits the arena");
        let mut buf = self.leaf.alloc_snapshot_buf(bytes).unwrap_or_else(|code| panic!("arena span: {code}"));
        write(buf.as_mut()).unwrap_or_else(|code| panic!("blob write: leaf code {code}"));
        buf
    }

    /// `seq` spilled to KV-RAM, released, and restored into a fresh
    /// sequence: what a request evicted to the host tier comes back as.
    fn spilled_and_restored(&self, seq: FlashNextSequence) -> FlashNextSequence {
        let bytes = self.leaf.snapshot_bytes(self.model, &seq).expect("snapshot size");
        let blob = self.blob(bytes, |dst| self.leaf.snapshot_into(self.model, &seq, dst));
        self.leaf.release_sequence(self.model, seq);
        let mut restored = self.fresh();
        self.leaf.restore_sequence(self.model, &mut restored, &blob).expect("restore the spilled sequence");
        restored
    }
}

/// Every way a sequence's state moves, against the cold prefill split at
/// `frontier`.
fn the_state_at_an_image_frontier_travels(run: &Run<'_>, frontier: usize, label: &str) {
    let (leaf, model) = (run.leaf, run.model);
    let reference = |run: &Run<'_>| {
        let mut seq = run.fresh();
        run.prefill(&mut seq, 0, frontier);
        run.finish(seq, frontier)
    };
    let cold = reference(run);
    assert!(cold.logits.iter().chain(&cold.probe).all(|&b| f32::from_bits(b).is_finite()), "{label}: finite logits");
    println!("{label}: the reference decodes {:?}", cold.tokens);

    // The turn: the prompt to the frontier, its checkpoint captured into a
    // host and a device slot, then on as if it had captured nothing.
    let mut turn = run.fresh();
    run.prefill(&mut turn, 0, frontier);
    let on_host = leaf.capture_checkpoint(model, &mut turn, frontier as u32, HOST_SLOT).expect("capture (host)");
    let on_device = leaf.capture_checkpoint(model, &mut turn, frontier as u32, DEVICE_SLOT).expect("capture (device)");
    assert_eq!(run.finish(turn, frontier), cold, "{label}: the sequence that captured goes on as if it had not");

    for (slot, checkpoint) in [("host", &on_host), ("device", &on_device)] {
        let (seq, micros) = leaf.allocate_sequence_from_checkpoint(model, MAX_CONTEXT, checkpoint).expect("claim");
        assert_eq!(run.finish(seq, frontier), cold, "{label}: the checkpoint claimed from a {slot} slot");
        println!("{label}: the checkpoint claimed from a {slot} slot ({micros} us) computes the reference");
    }

    // The checkpoint through KV-RAM: materialized, then restored into a fresh
    // sequence, as a KV-RAM claim does.
    let bytes = leaf.checkpoint_snapshot_bytes(model, &on_host).expect("checkpoint blob size");
    let blob = run.blob(bytes, |dst| leaf.checkpoint_snapshot_into(model, &on_host, dst));
    drop((on_host, on_device));
    let mut seq = run.fresh();
    leaf.restore_sequence(model, &mut seq, &blob).expect("restore the checkpoint blob");
    drop(blob);
    assert_eq!(run.finish(seq, frontier), cold, "{label}: the checkpoint restored from KV-RAM ({bytes} bytes)");
    println!("{label}: the checkpoint restored from KV-RAM ({bytes} bytes) computes the reference");

    // A sequence spilled at the frontier and restored.
    let mut seq = run.fresh();
    run.prefill(&mut seq, 0, frontier);
    let seq = run.spilled_and_restored(seq);
    assert_eq!(run.finish(seq, frontier), cold, "{label}: a sequence spilled at the frontier and restored");
    println!("{label}: a sequence spilled at the frontier and restored computes the reference");

    // A sequence spilled mid-decode: every round after its restore rotates at
    // its position plus the rope delta its blob carried.
    let mut seq = run.fresh();
    run.prefill(&mut seq, 0, frontier);
    let logits = run.prefill(&mut seq, frontier, run.tokens.len());
    let mut tokens = run.decode(&mut seq, DECODED / 2);
    let mut seq = run.spilled_and_restored(seq);
    tokens.extend(run.decode(&mut seq, DECODED - DECODED / 2));
    let probe = run.probe(&mut seq);
    leaf.release_sequence(model, seq);
    assert_eq!(Outcome { logits, tokens, probe }, cold, "{label}: a sequence spilled mid-decode and restored");
    println!("{label}: a sequence spilled mid-decode and restored computes the reference");

    assert_eq!(reference(run), cold, "{label}: a warm expert cache computes what a cold one did");
    let (capacity, used) = leaf.kv_ram_arena_stats();
    assert_eq!(used, 0, "{label}: every blob went back to the {capacity}-byte arena");
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact and its vision companion"]
fn an_image_s_state_travels_through_checkpoints_and_spills_bit_for_bit() {
    let (tokens, multimodal) = prompt();
    let item = multimodal.media[0].token_span;
    assert_eq!((item.begin, item.count), (HEAD, (SIDE * SIDE) as usize), "the image's placeholders");
    let indexer = ModelConfig::qwen38_flash_next().indexer.expect("Flash-Next's indexer");
    let block = indexer.compress_ratio as usize;
    // The geometry this rests on, said rather than hoped for.
    for (frontier, label) in FRONTIERS {
        let begun = frontier / block * block;
        assert_ne!(frontier % block, 0, "{label}: the frontier is inside a block");
        assert!((item.begin..item.begin + item.count).contains(&begun), "{label}: the block begins on an image token");
        // A block whose first token's axes did not travel rotates at that
        // token's KV position on all three: these are not those.
        let axes: Vec<i32> = (0..3).map(|axis| multimodal.positions[axis * tokens.len() + begun]).collect();
        assert_ne!(axes, [begun as i32; 3], "{label}: its first token's axes are not its KV position");
        assert!(frontier as u64 > indexer.dense_threshold(), "{label}: every row past it selects blocks");
        assert_ne!(frontier % KV_PAGE_TOKENS as usize, 0, "{label}: a checkpoint carries a partial tail page");
        println!("{label}: frontier {frontier}, the block from {begun} begins at axes {axes:?}");
    }

    let Some(leaf) = leaf() else { return };
    let model = leaf.load_model().expect("load the model");
    let media = leaf.encode_media(&model, &multimodal.media[0]).expect("encode the image");
    let run = Run { leaf: &leaf, model: &model, tokens: &tokens, multimodal: &multimodal, media: &media };
    for (frontier, label) in FRONTIERS {
        the_state_at_an_image_frontier_travels(&run, frontier, label);
    }
    leaf.release_media(&model, media);
    leaf.release_model(model);
}
