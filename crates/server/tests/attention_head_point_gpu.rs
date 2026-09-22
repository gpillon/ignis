//! Does one attention head point, in the engine?
//!
//! `docs/findings/2026-09-21-one-attention-head-points.md` found, in a
//! PyTorch vehicle, that one head — **L39.h10**, GQA ordinal 9, query head 10
//! — read at the position after the forced `{"x":` lands inside the target
//! button on 236 and 233 of 240 scenes, against the ten-round digit chain's
//! 218 and 213, and that as a guard on the chain it gives 239 of 240. That
//! vehicle is not this engine: its weights were bitsandbytes NF4 where these
//! are NVFP4, its KV was BF16 where serving is hq-e8-2b, and its images were
//! 1024 px where the endpoint takes up to 4096. This measures the same head
//! here.
//!
//! **How the attention is read.** The fused attention kernels never
//! materialize weights, so the test-only tap (`ignis_core::attn_tap`,
//! `kernel/include/ignis_attn_tap.h`) captures the rotated query and key
//! rows `run_gqa_layer` hands to the attention op, and the host forms
//! `q . k / 16` for every (GQA layer, query head) over the image positions.
//!
//! **Under hq-e8-2b there are two readings, and only one is what attention
//! computes.** The hq prompt route materializes the visible history into
//! rotated-frame BF16 scratch planes and attends over those. The vendored
//! kernel *can* fill them from three sources — the current chunk exact, the
//! first 32 and the 512 rows before the chunk exact from a residual window,
//! everything older decoded (`gqa_attention_prefill_hq.cuh`) — but the
//! residual window is a feature a caller opts into by handing the KV view
//! `residual_k`/`residual_v`/`ring_valid` (`core/paged_kv_cache.h`: "empty
//! tensors mean the feature is off"), and **ignis hands it none**. So the
//! launcher's `has_fresh` is false, no side row ever fires, and **every key
//! this engine's hq attention reads is decoded by the codec** — measured on
//! the first 4096 px capture, where the rows the three-source rule would
//! have kept exact came back at the codec's own median 0.370 and max 0.762.
//!
//! - `IGNIS_POINT_KV=hq` scores against those scratch planes — the keys
//!   attention *consumed*, captured after the op returns (`with_attn_tap_hq`).
//!   The rotation is orthonormal, so rotating the query instead of
//!   un-rotating the keys leaves `q . k` unchanged. This is the hq number.
//! - `IGNIS_POINT_KV=hq-precodec` scores against the keys *given to* the
//!   codec, which is what this test measured under hq before the consumed
//!   capture existed (commit 94b9224's hq rows). Kept so those numbers can
//!   be reproduced; it is not what attention reads.
//!
//! The consumed capture verifies itself, and a run that fails the check is
//! not a measurement. Every prompt row on the armed head's KV head is
//! classified by `ignis_core::hq_ring::prompt_source` — the vendored prompt
//! route's rule with the residual window wired (GitHub #257), replayed over
//! this prompt's own chunks — and compared with the rotated pre-codec key: a
//! fresh, sink or ring row must be exact (relative L2 under
//! [`EXACT_ROW_REL_ERR`]), a clobbered one exact to the key whose append
//! rewrote its ring slot, a codec row not exact, and the codec rows' median
//! must sit in the codec's own band, [`CODEC_ROW_MIN_MEDIAN`] to
//! [`CODEC_ROW_MAX_MEDIAN`], which a mis-offset or mis-rotated capture
//! (uncorrelated, ~1.4) cannot.
//!
//! **Geometry.** Before #257 every key was decoded, so the codec fraction was
//! 100% at every image size and every hq number above was measured that way.
//! With the window the image's codec fraction depends on where the query's
//! chunk starts — 40.1% on a 1146-token 1024 px prompt, 96.3% on the 4096 px
//! fixture — and the JSON records it per scene. Set C4096 still exists for
//! the resolution.
//!
//! **Two renders, because the vehicle did not measure the served prompt.**
//! The vehicle rendered with `enable_thinking` undefined, which in this
//! artifact's template means three things `/v1/decide` never sends: a
//! "Reasoning effort is set to xhigh …" paragraph at the head of the system
//! block, a think block left *open* before the forced `{"x":`, and the
//! instruction as plain text instead of `{"instruction":…}`.
//!
//! - `IGNIS_POINT_RENDER=vehicle` (default) reproduces that render —
//!   `enable_thinking: true`, `reasoning_effort` left to the template's own
//!   default — and asserts both ends of it. It is the only arm that isolates
//!   weights and KV from the prompt, so **the pre-registered criterion is
//!   asserted here and only here**.
//! - `IGNIS_POINT_RENDER=served` renders what `/v1/decide` sends —
//!   `enable_thinking: false`, `{"instruction":…}` built by the endpoint's
//!   own `OrderedValue` — and asserts that too. It is a new measurement,
//!   reported and not held to a criterion nobody pre-registered for it.
//!
//! **Pre-registered** before the first engine run (2026-09-21, agreed between
//! the two sessions that wrote this and the tap), vehicle render, 1024 px,
//! the same 480 scenes, L39.h10 read by TAG's region rule:
//! set A (target always blue) at least [`CRITERION_A`] of 240, set B (colour
//! and label instructions) at least [`CRITERION_B`] of 240 — within about six
//! of the vehicle — and the guard at [`GUARD_DISTANCE`] at least
//! [`CRITERION_GUARD`] on both. Same criterion under BF16 KV first, then
//! hq-e8-2b. Whether cross-validation still picks L39.h10 in the engine is
//! reported by the scorer that reads this test's dump, not asserted here.
//!
//! **Pre-registered for set C** (2026-09-22, before any run on it, agreed
//! between the same two sessions): set C is 240 new varied scenes at 1024 px
//! (`scenes.py --varied --seed 20260923`), measured with the **served**
//! render and **consumed hq** keys — the production configuration at this
//! size. L39.h10 at least [`CRITERION_C_HEAD`] of 240, the guard at
//! [`GUARD_DISTANCE`] (unchanged, not re-tuned) at least
//! [`CRITERION_C_GUARD`]. These are **"not broken" floors, not "as good as
//! before"**: the worst hq arm observed (230 head, 235 guard, keys before
//! the codec) minus the same slack as the first criterion, because the
//! codec on the armed layer may cost something and nobody has measured how
//! much. Set C4096 (`--side 4096 --seed 20260924`, 240 scenes, same render
//! and KV) is **reported and not evaluated**: it is the regime production
//! serves, and nothing about it was known when these floors were written.
//!
//! **Scenes.** `IGNIS_POINT_SCENES=<dir with manifest.json>` runs a generated
//! set (`.scratch/latent-probe/scenes` or `scenes-varied`, 1024 px, per-scene
//! `instruction` and `kind` in the varied one). Without it, the committed
//! three-scene fixture at **4096 px**, where C5 has never been measured: there
//! the test asserts structure and the chain (which `decide_point_gpu.rs`
//! measured inside on all three) and only *prints* L39.h10, because asserting
//! an unmeasured regime would turn a guess into a guarantee.
//!
//! `IGNIS_POINT_KV=bf16|hq|hq-precodec` (default bf16),
//! `IGNIS_POINT_CHUNK=<tokens>` (default 1024, the serving default; a
//! multiple of 128 — at 4096 px the image crosses ~16 prefill chunks, and
//! a wider chunk is how a chunk-boundary effect is told from the model's
//! own; a non-default width is appended to the dump's name),
//! `IGNIS_POINT_LAYERS=all|head` (default: all GQA layers at 1024 px and
//! below, only the head's layer above — at 4096 px sixteen layers of keys
//! are ~0.5 GB of host memory per scene and the head is already fixed),
//! `IGNIS_POINT_LIMIT=<n>` for a smoke run (a criterion is asserted only on
//! a full 240), and
//! `IGNIS_POINT_OUT=<dir>` for the dump (default: the OS temp dir). The dump
//! is `<set>-<render>-<kv>.bin` — every armed head's scores as
//! little-endian f16, `[scene][armed GQA layer][query head 0..24][image
//! position]` — and a
//! `.json` beside it with everything a scorer needs, including the full
//! rendered text of the first scene so it can be compared byte for byte with
//! the vehicle's own render.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38), and the `attn-tap` feature.

#![cfg(all(feature = "cuda", feature = "attn-tap"))]

#[path = "support/mod.rs"]
mod support;

use std::path::{Path, PathBuf};

use ignis_artifact::{
    ChatMessage, ChatRenderOptions, ContentPart, CudaDevice, FrontendSet, MessageContent,
    ModelScope, Reader, Role, bind_model_scope_27b_with, materialize,
};
use ignis_core::attn_tap::{GQA_LAYERS, Q_HEADS, with_attn_tap, with_attn_tap_hq};
use ignis_core::hq_ring::{PromptSource, prompt_source, ring_after_prefill};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b_with_options;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step;
use ignis_core::vision::{Multimodal, vision_item_control};
use ignis_core::{KvFormat, RopeScaling, Vision};
use ignis_server::decide::OrderedValue;
use ignis_server::numbers::point_system;
use serde::Deserialize;

use support::vision_canary::prefill_prompt;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// Sized for the fixture's 4096x4096 images (~16.5K prompt tokens); the
/// 1024 px sets need ~1.2K.
const MAX_CONTEXT: u32 = 20_480;
/// The serving default (`DEFAULT_PREFILL_CHUNK`); `IGNIS_POINT_CHUNK`
/// overrides it for the chunk-boundary control.
const DEFAULT_PREFILL_CHUNK: u32 = 1024;

fn prefill_chunk() -> u32 {
    match std::env::var("IGNIS_POINT_CHUNK") {
        Err(_) => DEFAULT_PREFILL_CHUNK,
        Ok(v) => {
            let chunk: u32 = v.parse().unwrap_or_else(|e| panic!("IGNIS_POINT_CHUNK={v}: {e}"));
            assert!(
                chunk > 0 && chunk % 128 == 0,
                "IGNIS_POINT_CHUNK={chunk}: a nonzero multiple of 128"
            );
            chunk
        }
    }
}

const DIGITS: usize = 3;
const SCALE: f64 = 999.0;

/// `<|image_pad|>` in this artifact's tokenizer (ADR 0034's own correction
/// names it at 248,056). Checked against every position of the media item's
/// span, so the span and the token ids are two sources for one fact.
const IMAGE_PAD: u32 = 248_056;

/// The head under test: backbone layer 39 is GQA ordinal 9 (`4 * 9 + 3`).
const HEAD_ORDINAL: usize = 9;
const HEAD_Q: usize = 10;

/// TAG's region rule, as the vehicle's scorer applies it.
const REGION_THRESHOLD: f64 = 0.5;
/// The guard: a chain point farther than this from the head's point, in
/// 0-999 units of the side, is taken to have picked the wrong element.
const GUARD_DISTANCE: f64 = 60.0;

/// The generated sets, named by the seed their manifest records — so a
/// criterion is tied to the exact scenes it was written for, not to a
/// property another set could share (`varied` is true of B and C alike).
const SEED_A: u64 = 20_260_921;
const SEED_B: u64 = 20_260_922;
const SEED_C: u64 = 20_260_923;
const SEED_C4096: u64 = 20_260_924;

/// Pre-registered 2026-09-21: sets A and B, vehicle render, BF16 keys and
/// the keys given to the hq codec (module docs).
const CRITERION_SCENES: usize = 240;
const CRITERION_A: usize = 230;
const CRITERION_B: usize = 227;
const CRITERION_GUARD: usize = 237;

/// Pre-registered 2026-09-22, before any run on set C: served render,
/// consumed hq keys, 1024 px. "Not broken" floors (module docs).
const CRITERION_C_HEAD: usize = 224;
const CRITERION_C_GUARD: usize = 233;

/// Under this relative L2 against the rotated pre-codec key a row was read
/// exact, i.e. not through the codec. The same bound as
/// `attn_tap_hq_consumed_gpu.rs`: exact rows sit near 0.002-0.004, the
/// codec's lowest row on a 4096 px prompt at 0.333, so 0.1 is far from both.
const EXACT_ROW_REL_ERR: f64 = 0.1;
/// The codec's own per-row error band on real rows: median ~0.37, max
/// ~0.77 (`docs/findings/2026-09-12-hq-attention-route-agreement.md`). A
/// median below the floor was not quantized; above the ceiling the capture
/// is not aligned with the keys it claims to be (uncorrelated rows sit ~1.4).
const CODEC_ROW_MIN_MEDIAN: f64 = 0.2;
const CODEC_ROW_MAX_MEDIAN: f64 = 0.6;

const DEFAULT_INSTRUCTION: &str = "click the blue button";

/// The head of the system block the vehicle's render carries and
/// `/v1/decide` never does (the template's `reasoning_effort` default).
const XHIGH_HEAD: &str = "<|im_start|>system\nReasoning effort is set to xhigh.";
const VEHICLE_TAIL: &str = "<|im_start|>assistant\n<think>\n";
const SERVED_TAIL: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

#[derive(Deserialize)]
struct Manifest {
    side: u32,
    #[serde(default)]
    seed: Option<u64>,
    scenes: Vec<Scene>,
}

#[derive(Deserialize)]
struct Scene {
    id: String,
    image: String,
    /// The target's box and centre, in pixels of the scene's own side. The
    /// generated sets keep the fixture's `blue_*` names even when the target
    /// is another colour.
    blue_box: [i64; 4],
    blue_centre: [i64; 2],
    #[serde(default)]
    instruction: Option<String>,
    #[serde(default)]
    kind: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Render {
    Vehicle,
    Served,
}

impl Render {
    fn from_env() -> Self {
        match std::env::var("IGNIS_POINT_RENDER").as_deref() {
            Err(_) | Ok("vehicle") => Self::Vehicle,
            Ok("served") => Self::Served,
            Ok(other) => panic!("IGNIS_POINT_RENDER must be vehicle or served, not {other:?}"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Vehicle => "vehicle",
            Self::Served => "served",
        }
    }

    /// A fresh value per call: the options are consumed by the render and by
    /// the prompt preparation, and both must see the same ones.
    fn options(self) -> ChatRenderOptions {
        match self {
            // `reasoning_effort` left `None` on purpose: that is the
            // template's own default, which is what the vehicle got by not
            // passing it at all.
            Self::Vehicle => ChatRenderOptions { enable_thinking: true, ..Default::default() },
            Self::Served => ChatRenderOptions { enable_thinking: false, ..Default::default() },
        }
    }

    fn user_text(self, instruction: &str) -> String {
        match self {
            Self::Vehicle => instruction.to_owned(),
            // The endpoint's own serializer, so the bytes are the endpoint's.
            Self::Served => format!(
                "{{\"instruction\":{}}}",
                OrderedValue::String(instruction.to_owned()).to_text()
            ),
        }
    }

    /// Both ends of the render, asserted — a render this test did not intend
    /// would otherwise be measured as if it were one of the two.
    fn check(self, rendered: &str, scene: &str) {
        match self {
            Self::Vehicle => {
                assert!(
                    rendered.starts_with(XHIGH_HEAD),
                    "{scene}: the vehicle render must open with the template's xhigh reasoning \
                     paragraph; it opens with {:?}",
                    &rendered[..rendered.len().min(120)]
                );
                assert!(
                    rendered.ends_with(VEHICLE_TAIL),
                    "{scene}: the vehicle render must end with an open think block; it ends with {:?}",
                    &rendered[rendered.len().saturating_sub(60)..]
                );
            }
            Self::Served => {
                assert!(
                    !rendered.contains("Reasoning effort"),
                    "{scene}: the served render must carry no reasoning paragraph"
                );
                assert!(
                    rendered.ends_with(SERVED_TAIL),
                    "{scene}: the served render must end with a closed, empty think block; it ends \
                     with {:?}",
                    &rendered[rendered.len().saturating_sub(60)..]
                );
            }
        }
    }
}

/// Which keys the scores are formed against (module docs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum KvMode {
    Bf16,
    /// hq-e8-2b, the keys attention consumed: fresh, side and decoded rows.
    HqConsumed,
    /// hq-e8-2b, the keys given to the codec (the pre-consumed-capture hq).
    HqPrecodec,
}

impl KvMode {
    fn from_env() -> Self {
        match std::env::var("IGNIS_POINT_KV").as_deref() {
            Err(_) | Ok("bf16") => Self::Bf16,
            Ok("hq") => Self::HqConsumed,
            Ok("hq-precodec") => Self::HqPrecodec,
            Ok(other) => panic!("IGNIS_POINT_KV must be bf16, hq or hq-precodec, not {other:?}"),
        }
    }

    fn format(self) -> KvFormat {
        match self {
            Self::Bf16 => KvFormat::Bf16,
            Self::HqConsumed | Self::HqPrecodec => KvFormat::HqE8_2b,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::HqConsumed => "hq",
            Self::HqPrecodec => "hq-precodec",
        }
    }
}

/// The start of the prefill chunk holding `position`, cut exactly as
/// `support::vision_canary::prefill_prompt` cuts it (`cap_chunk`, then the
/// chunk width). The hq route's fresh/side/codec rule is relative to it.
fn chunk_start_of(prompt: &Multimodal, total: u32, chunk: u32, position: u32) -> u32 {
    let mut start = 0u32;
    while start < total {
        let len = prompt.cap_chunk(start, chunk.min(total - start));
        if position < start + len {
            return start;
        }
        start += len;
    }
    panic!("position {position} is past the prompt's {total} tokens")
}

/// The `(start, len)` chunks up to and including the one holding `position`,
/// cut as `chunk_start_of` cuts them: what the hq ring saw appended by the
/// time that chunk's attention read it.
fn chunks_through(prompt: &Multimodal, total: u32, chunk: u32, position: u32) -> Vec<(u64, u64)> {
    let mut chunks = Vec::new();
    let mut start = 0u32;
    while start < total {
        let len = prompt.cap_chunk(start, chunk.min(total - start));
        chunks.push((u64::from(start), u64::from(len)));
        if position < start + len {
            return chunks;
        }
        start += len;
    }
    panic!("position {position} is past the prompt's {total} tokens")
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    Some(values[values.len() / 2])
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("pointing")
}

fn messages_with(system: &str, text: String) -> [ChatMessage; 2] {
    [
        ChatMessage {
            role: Role::System,
            content: MessageContent::Text(system.to_owned()),
            tool_calls: Vec::new(),
            reasoning_content: None,
        },
        ChatMessage {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::Image { url: None },
                ContentPart::Text(text),
            ]),
            tool_calls: Vec::new(),
            reasoning_content: None,
        },
    ]
}

fn logsumexp(values: impl Iterator<Item = f32> + Clone) -> f64 {
    let maximum = values
        .clone()
        .fold(f64::NEG_INFINITY, |acc, v| acc.max(f64::from(v)));
    if !maximum.is_finite() {
        return maximum;
    }
    maximum + values.map(|v| (f64::from(v) - maximum).exp()).sum::<f64>().ln()
}

/// One constrained step, as `classify_pointing_gpu.rs` takes it: the winning
/// slot and its probability within the slots.
fn constrained_pick(logits: &[f32], slots: &[u32]) -> (usize, f64) {
    let restricted: Vec<f32> = slots.iter().map(|&slot| logits[slot as usize]).collect();
    let total = logsumexp(restricted.iter().copied());
    let (index, _) = restricted
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| if v > best.1 { (i, v) } else { best });
    (index, (f64::from(restricted[index]) - total).exp())
}

/// TAG's region rule over one head's map, exactly as the vehicle's scorer
/// (`.scratch/latent-probe/c5_score.py::region_tag`) applies it: min-max
/// normalize, keep cells at or above the threshold, take the 4-connected
/// region with the highest mean, and return its weighted centre in pixels.
///
/// The map is `exp(s - max s)` over the image positions. The vehicle's maps
/// were softmax weights over the whole row restricted to the image columns,
/// which differ from these by one positive factor — and min-max
/// normalization removes it, so the point is the same.
fn region_point(scores: &[f32], gh: usize, gw: usize, side: f64) -> (f64, f64) {
    let top = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let weights: Vec<f64> = scores.iter().map(|&s| f64::from(s - top).exp()).collect();
    let centre = |k: usize| -> (f64, f64) {
        (
            ((k % gw) as f64 + 0.5) * side / gw as f64,
            ((k / gw) as f64 + 0.5) * side / gh as f64,
        )
    };
    let lo = weights.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if hi <= lo {
        let k = weights
            .iter()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
            .0;
        return centre(k);
    }
    let r: Vec<f64> = weights.iter().map(|&w| (w - lo) / (hi - lo)).collect();
    let on: Vec<bool> = r.iter().map(|&v| v >= REGION_THRESHOLD).collect();
    let mut seen = vec![false; r.len()];
    let mut best: Option<(f64, Vec<usize>)> = None;
    for start in 0..r.len() {
        if !on[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        let mut stack = vec![start];
        let mut component = Vec::new();
        while let Some(k) = stack.pop() {
            component.push(k);
            let (row, col) = (k / gw, k % gw);
            let mut visit = |row: usize, col: usize| {
                let n = row * gw + col;
                if on[n] && !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            };
            if row + 1 < gh {
                visit(row + 1, col);
            }
            if row > 0 {
                visit(row - 1, col);
            }
            if col + 1 < gw {
                visit(row, col + 1);
            }
            if col > 0 {
                visit(row, col - 1);
            }
        }
        let mean = component.iter().map(|&k| r[k]).sum::<f64>() / component.len() as f64;
        // Strictly greater, so the first region found wins a tie — the
        // vehicle's scorer does the same.
        if best.as_ref().is_none_or(|(m, _)| mean > *m) {
            best = Some((mean, component));
        }
    }
    let (_, component) = best.expect("the maximum cell is always on");
    let total: f64 = component.iter().map(|&k| r[k]).sum();
    let (mut x, mut y) = (0.0, 0.0);
    for &k in &component {
        let (cx, cy) = centre(k);
        x += cx * r[k] / total;
        y += cy * r[k] / total;
    }
    (x, y)
}

fn inside(point: (f64, f64), b: [i64; 4]) -> bool {
    b[0] as f64 <= point.0 && point.0 <= b[2] as f64 && b[1] as f64 <= point.1 && point.1 <= b[3] as f64
}

/// f32 to IEEE half, round to nearest even. Written out rather than taking a
/// dependency for one conversion; attention logits sit well inside the
/// normal range, and the edges are handled anyway.
fn f32_to_f16(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | (if mant != 0 { 0x0200 } else { 0 });
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let rounded = if rem > halfway || (rem == halfway && half & 1 == 1) { half + 1 } else { half };
        return sign | rounded as u16;
    }
    let half = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    let rounded = if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) { half + 1 } else { half };
    sign | rounded as u16
}

#[test]
#[ignore = "GPU profile only, with --features attn-tap"]
fn one_attention_head_points_in_the_engine() {
    let render = Render::from_env();
    let kv_mode = KvMode::from_env();
    let kv_format = kv_mode.format();
    let chunk = prefill_chunk();
    let (dir, generated) = match std::env::var("IGNIS_POINT_SCENES") {
        Ok(dir) => (PathBuf::from(dir), true),
        Err(_) => (fixture_dir(), false),
    };
    let Ok(manifest_text) = std::fs::read_to_string(dir.join("manifest.json")) else {
        if gpu_profile::skip_or_fail(&format!("no scene manifest in {}", dir.display())) {
            return;
        }
        unreachable!("skip_or_fail panics under the profile");
    };
    let manifest: Manifest =
        serde_json::from_str(&manifest_text).unwrap_or_else(|e| panic!("parse the manifest: {e}"));
    let limit = std::env::var("IGNIS_POINT_LIMIT")
        .ok()
        .map(|v| v.parse::<usize>().unwrap_or_else(|e| panic!("IGNIS_POINT_LIMIT: {e}")));
    let scenes: Vec<&Scene> = manifest
        .scenes
        .iter()
        .take(limit.unwrap_or(usize::MAX))
        .collect();
    let set_name = if generated {
        dir.file_name().and_then(|n| n.to_str()).unwrap_or("scenes").to_owned()
    } else {
        "fixture".to_owned()
    };
    let varied = scenes.iter().any(|s| s.kind.is_some());

    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let processor = frontend
        .vision_processor()
        .unwrap_or_else(|e| panic!("vision processor: {e}"));
    let tokenizer = frontend.tokenizer();
    let encode = |text: &str| -> Vec<u32> {
        tokenizer.encode(text).unwrap_or_else(|e| panic!("encode {text:?}: {e}"))
    };
    let digit_slots: Vec<u32> = (0..10)
        .map(|digit| {
            let ids = encode(&digit.to_string());
            assert_eq!(ids.len(), 1, "digit {digit} must be one token");
            ids[0]
        })
        .collect();
    let x_prefix = encode("{\"x\":");
    let y_literal: Vec<i32> = encode(",\"y\":").into_iter().map(|t| t as i32).collect();
    let system = point_system(DIGITS as u32);

    let (plan, handles) = bind_model_scope_27b_with(&reader, ModelScope { draft: None, vision: true })
        .unwrap_or_else(|e| panic!("bind with vision: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(d) => d,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(a) => a,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize text + vision: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let model = load_qwen38_27b_with_options(
        &reader,
        &artifact,
        &handles,
        chunk,
        MAX_CONTEXT,
        kv_format,
        None,
        Some(Vision::default()),
        RopeScaling::NONE,
    )
    .unwrap_or_else(|e| panic!("ignis_model_load with vision: {e}"));
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format,
            kv_page_group_count: MAX_CONTEXT / 64,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 1,
            retained_slot_count: 0,
        },
    )
    .unwrap_or_else(|e| panic!("seq pool create: {e}"));
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let mut logits = vec![0f32; vocab];
    let head_only = match std::env::var("IGNIS_POINT_LAYERS").as_deref() {
        Err(_) => manifest.side > 1024,
        Ok("all") => false,
        Ok("head") => true,
        Ok(other) => panic!("IGNIS_POINT_LAYERS must be all or head, not {other:?}"),
    };
    let ordinals: Vec<i32> = if head_only {
        vec![HEAD_ORDINAL as i32]
    } else {
        (0..GQA_LAYERS as i32).collect()
    };
    // Where the head's layer sits among the armed ones: the capture indexes
    // by that, not by ordinal.
    let head_layer = ordinals
        .iter()
        .position(|&o| o as usize == HEAD_ORDINAL)
        .expect("the head's layer is always armed");
    let side = f64::from(manifest.side);

    eprintln!(
        "attention head point: set {set_name} ({} scenes{}), render {}, KV {}, head L{}.h{}",
        scenes.len(),
        if varied { ", varied" } else { "" },
        render.name(),
        kv_mode.name(),
        4 * HEAD_ORDINAL + 3,
        HEAD_Q
    );

    let mut dump: Vec<u8> = Vec::new();
    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut n_image: Option<usize> = None;
    let mut grid: Option<(usize, usize)> = None;
    let mut first_render: Option<String> = None;
    let (mut head_hits, mut chain_hits, mut guard_hits) = (0usize, 0usize, 0usize);
    // Consumed-capture self-check failures, collected so the dump is written
    // before any of them fails the run.
    let mut verify_failures: Vec<String> = Vec::new();

    for scene in &scenes {
        let bytes = std::fs::read(dir.join(&scene.image))
            .unwrap_or_else(|e| panic!("{}: read image: {e}", scene.id));
        let instruction = scene.instruction.as_deref().unwrap_or(DEFAULT_INSTRUCTION);
        let messages = messages_with(&system, render.user_text(instruction));

        // ── the render, checked at both ends ─────────────────────────────
        let rendered = frontend
            .chat_template()
            .render_with_thinking_and_tools(&messages, render.options(), None)
            .unwrap_or_else(|e| panic!("{}: render: {e}", scene.id));
        render.check(&rendered, &scene.id);
        if first_render.is_none() {
            first_render = Some(rendered.clone());
        }

        let prepared = frontend
            .prepare_prompt(&processor, &messages, &[&bytes], render.options(), None)
            .unwrap_or_else(|e| panic!("{}: prepare: {e}", scene.id));
        let (token_ids, mut prompt) = Multimodal::from_prepared(prepared);

        // ── the image span, from two sources ─────────────────────────────
        assert_eq!(prompt.media.len(), 1, "{}: one image", scene.id);
        let (begin, count, item_grid) = {
            let item = &prompt.media[0];
            (item.token_span.begin as usize, item.token_span.count as usize, item.grid)
        };
        assert_eq!(item_grid.t, 1, "{}: a still image has one temporal patch", scene.id);
        let (gh, gw) = (item_grid.h as usize / 2, item_grid.w as usize / 2);
        assert_eq!(count, gh * gw, "{}: span {count} != merged grid {gh}x{gw}", scene.id);
        assert!(
            token_ids[begin..begin + count].iter().all(|&t| t == IMAGE_PAD),
            "{}: the media span holds a token that is not <|image_pad|>",
            scene.id
        );
        match (n_image, grid) {
            (None, None) => {
                n_image = Some(count);
                grid = Some((gh, gw));
            }
            (Some(n), Some(g)) => assert!(
                n == count && g == (gh, gw),
                "{}: every scene of a dump must share one grid ({g:?}, {n}); this one is ({gh}, {gw})",
                scene.id
            ),
            _ => unreachable!(),
        }
        let image_positions: Vec<usize> = (begin..begin + count).collect();

        // Encoded before the prompt is extended: the embedding borrows only
        // the model, and `append_text` needs the prompt mutably.
        let control = vision_item_control(item_grid);
        let embedding = step::encode_media(&model, item_grid, &prompt.media[0].patches, &control)
            .unwrap_or_else(|(_, e)| panic!("{}: encode: {e}", scene.id));

        // ── the forced prefix, appended the way the served path does ─────
        assert!(
            prompt.append_text(x_prefix.len()),
            "{}: the prompt cannot be extended by the forced prefix",
            scene.id
        );
        let mut tokens: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        tokens.extend(x_prefix.iter().map(|&t| t as i32));
        assert!(
            tokens.len() + DIGITS * 2 + 16 < MAX_CONTEXT as usize,
            "{}: {} prompt tokens do not fit",
            scene.id,
            tokens.len()
        );
        // The query row is the last prompt position: the one after `{"x":`,
        // where the chain reads x's first digit and a one-pass point reads.
        let query = tokens.len() - 1;
        let max_positions = tokens.len() as i64 + 8;

        let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
        logits.fill(0.0);

        // The chunk the query sits in: the hq route's fresh/side/codec rule is
        // relative to its start.
        let query_chunk_start =
            chunk_start_of(&prompt, tokens.len() as u32, chunk, query as u32) as usize;

        // ── one armed prefill ────────────────────────────────────────────
        let run = || {
            prefill_prompt(
                &model,
                &pool,
                &mut sequence,
                &tokens,
                &prompt,
                &embedding,
                chunk,
                &mut logits,
            )
        };
        let (prefilled, capture) = match kv_mode {
            KvMode::HqConsumed => with_attn_tap_hq(&ordinals, &[query as i64], max_positions, run),
            KvMode::Bf16 | KvMode::HqPrecodec => {
                with_attn_tap(&ordinals, &[query as i64], max_positions, run)
            }
        }
        .unwrap_or_else(|e| panic!("{}: attention tap: {e}", scene.id));
        prefilled.unwrap_or_else(|e| panic!("{}: prefill: {e}", scene.id));
        assert_eq!(capture.queries_seen, vec![1], "{}: the query row was not captured", scene.id);
        assert!(
            capture.rows_written.iter().all(|&r| r == tokens.len() as i64),
            "{}: every armed layer must write one key row per prompt position ({}), wrote {:?}",
            scene.id,
            tokens.len(),
            capture.rows_written
        );

        // ── the consumed capture checks itself before it is used ─────────
        let mut hq_stats = serde_json::Value::Null;
        if kv_mode == KvMode::HqConsumed {
            let consumed = capture.consumed_rows[head_layer] as usize;
            if consumed != tokens.len() {
                verify_failures.push(format!(
                    "{}: the head's layer consumed {consumed} rows, expected {} (not captured?)",
                    scene.id,
                    tokens.len()
                ));
            }
            // The chunk the capture came from, as the kernel reports it, and as
            // this test computes it from the prompt's own chunking: two sources
            // for the boundary the whole fresh/side/codec rule hangs on.
            let captured_start = capture.consumed_chunk_start[head_layer];
            if captured_start != query_chunk_start as i64 {
                verify_failures.push(format!(
                    "{}: the capture's chunk starts at {captured_start}, the prompt's chunking \
                     puts the query's chunk at {query_chunk_start}",
                    scene.id
                ));
            }
            let query_chunk_start = captured_start.max(0) as usize;
            // Every row of the prompt on the head's own KV head, classified by
            // the kernel's rule (GitHub #257: the ring as the chunks up to the
            // query's appended it, the query chunk's own append included) and
            // compared with the rotated pre-codec key.
            let ring = ring_after_prefill(&chunks_through(
                &prompt,
                tokens.len() as u32,
                chunk,
                query as u32,
            ));
            let kv_head = HEAD_Q / (Q_HEADS / 4);
            let (mut exact_rows, mut clobbered, mut codec) = (Vec::new(), Vec::new(), Vec::new());
            let mut off_rule = 0usize;
            let (mut image_codec_rule, mut image_codec_measured) = (0usize, 0usize);
            for position in 0..tokens.len() {
                let err = f64::from(capture.consumed_key_rel_err(head_layer, position, kv_head));
                let source = prompt_source(position as u64, query_chunk_start as u64, &ring);
                match source {
                    PromptSource::Fresh | PromptSource::Sink | PromptSource::Ring => {
                        off_rule += usize::from(err >= EXACT_ROW_REL_ERR);
                        exact_rows.push(err);
                    }
                    PromptSource::Clobbered { by } => {
                        let to_by = f64::from(capture.consumed_key_rel_err_to(
                            head_layer,
                            position,
                            kv_head,
                            by as usize,
                        ));
                        off_rule += usize::from(to_by >= EXACT_ROW_REL_ERR);
                        clobbered.push(to_by);
                    }
                    PromptSource::Codec => {
                        off_rule += usize::from(err < EXACT_ROW_REL_ERR);
                        codec.push(err);
                    }
                }
                if (begin..begin + count).contains(&position) {
                    image_codec_rule += usize::from(source == PromptSource::Codec);
                    image_codec_measured += usize::from(err >= EXACT_ROW_REL_ERR);
                }
            }
            if off_rule > 0 {
                verify_failures.push(format!(
                    "{}: {off_rule} of {} rows are not what the hq prompt route's rule says (exact \
                     where it keeps a row, decoded where it does not)",
                    scene.id,
                    tokens.len()
                ));
            }
            let codec_median = median(&mut codec.clone()).unwrap_or(f64::NAN);
            if !(CODEC_ROW_MIN_MEDIAN..=CODEC_ROW_MAX_MEDIAN).contains(&codec_median) {
                verify_failures.push(format!(
                    "{}: the decoded keys sit at median rel L2 {codec_median:.4} from the rotated \
                     pre-codec keys, outside the codec's band [{CODEC_ROW_MIN_MEDIAN}, \
                     {CODEC_ROW_MAX_MEDIAN}] — the capture is not the keys attention consumed",
                    scene.id
                ));
            }
            hq_stats = serde_json::json!({
                "query_chunk_start": query_chunk_start,
                // What the rule puts through the codec, beside what was
                // measured decoded (a clobbered row is exact to another key,
                // so it counts as measured-decoded against its own).
                "codec_fraction_rule": image_codec_rule as f64 / count as f64,
                "codec_fraction_measured": image_codec_measured as f64 / count as f64,
                "rows": {"exact": exact_rows.len(), "clobbered": clobbered.len(), "codec": codec.len()},
                "median_rel_err_by_rule_class": {
                    "exact": median(&mut exact_rows),
                    "clobbered_to_rewriting_key": median(&mut clobbered),
                    "codec": median(&mut codec),
                },
            });
        }

        // ── every armed head's scores, into the dump ─────────────────────
        let mut head_scores: Vec<f32> = Vec::new();
        for layer in 0..ordinals.len() {
            for q_head in 0..Q_HEADS {
                let scores = match kv_mode {
                    KvMode::HqConsumed => capture.consumed_scores(layer, 0, q_head, &image_positions),
                    KvMode::Bf16 | KvMode::HqPrecodec => capture.scores(layer, 0, q_head, &image_positions),
                };
                assert!(
                    scores.iter().all(|s| s.is_finite()),
                    "{}: non-finite score at ordinal {} head {q_head}",
                    scene.id,
                    ordinals[layer]
                );
                for &s in &scores {
                    dump.extend_from_slice(&f32_to_f16(s).to_le_bytes());
                }
                if layer == head_layer && q_head == HEAD_Q {
                    head_scores = scores;
                }
            }
        }
        drop(capture);
        let head_point = region_point(&head_scores, gh, gw, side);

        // ── the chain, continuing from the same prefill ──────────────────
        let mut position = tokens.len() as u64;
        let read_digits = |sequence: &mut _, position: &mut u64, logits: &mut Vec<f32>| {
            let mut value = 0i64;
            let mut first_p = 0.0;
            for place in 0..DIGITS {
                let (digit, probability) = constrained_pick(logits, &digit_slots);
                if place == 0 {
                    first_p = probability;
                }
                value = value * 10 + digit as i64;
                let token = [digit_slots[digit] as i32];
                step::prefill_program(&model, &pool, sequence, &token, *position, Some(logits.as_mut_slice()))
                    .unwrap_or_else(|e| panic!("{}: digit {place}: {e}", scene.id));
                *position += 1;
            }
            (value, first_p)
        };
        let (x, x_p) = read_digits(&mut sequence, &mut position, &mut logits);
        step::prefill_program(&model, &pool, &mut sequence, &y_literal, position, Some(logits.as_mut_slice()))
            .unwrap_or_else(|e| panic!("{}: force ,\"y\": {e}", scene.id));
        position += y_literal.len() as u64;
        let (y, y_p) = read_digits(&mut sequence, &mut position, &mut logits);
        drop(sequence);
        drop(embedding);

        let chain_point = (x as f64 / SCALE * side, y as f64 / SCALE * side);
        let distance = ((chain_point.0 - head_point.0).powi(2) + (chain_point.1 - head_point.1).powi(2))
            .sqrt()
            / side
            * SCALE;
        let guarded = if distance <= GUARD_DISTANCE { chain_point } else { head_point };
        let (head_in, chain_in, guard_in) = (
            inside(head_point, scene.blue_box),
            inside(chain_point, scene.blue_box),
            inside(guarded, scene.blue_box),
        );
        head_hits += usize::from(head_in);
        chain_hits += usize::from(chain_in);
        guard_hits += usize::from(guard_in);
        eprintln!(
            "  {} {:>7}: head ({:.0},{:.0}) {}  chain ({x},{y}) p1 {x_p:.3}/{y_p:.3} {}  d {distance:.0} guard {}",
            scene.id,
            scene.kind.as_deref().unwrap_or("-"),
            head_point.0,
            head_point.1,
            if head_in { "in " } else { "OUT" },
            if chain_in { "in " } else { "OUT" },
            if guard_in { "in" } else { "OUT" },
        );
        rows.push(serde_json::json!({
            "id": scene.id,
            "kind": scene.kind,
            "instruction": instruction,
            "box": scene.blue_box,
            "centre": scene.blue_centre,
            "prompt_tokens": tokens.len(),
            "chain": [x, y],
            "first_digit_p": [x_p, y_p],
            "head_point": [head_point.0, head_point.1],
            "head_inside": head_in,
            "chain_inside": chain_in,
            "guard_distance": distance,
            "guard_inside": guard_in,
            "hq": hq_stats,
        }));
    }

    // ── the dump, written before anything is asserted ────────────────────
    let out_dir = std::env::var("IGNIS_POINT_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("ignis-attention-head-point"));
    std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| panic!("create {}: {e}", out_dir.display()));
    let stem = match chunk {
        DEFAULT_PREFILL_CHUNK => format!("{set_name}-{}-{}", render.name(), kv_mode.name()),
        _ => format!("{set_name}-{}-{}-chunk{chunk}", render.name(), kv_mode.name()),
    };
    let (gh, gw) = grid.expect("at least one scene");
    let meta = serde_json::json!({
        "set": set_name,
        "render": render.name(),
        "kv": kv_mode.name(),
        "kv_format": kv_format.as_str(),
        "prefill_chunk": chunk,
        "seed": manifest.seed,
        "side": manifest.side,
        "grid": [gh, gw],
        "n_image": n_image,
        "layers": ordinals.iter().map(|o| 4 * o + 3).collect::<Vec<_>>(),
        "q_heads": Q_HEADS,
        "dtype": "f16-le",
        "shape": [rows.len(), ordinals.len(), Q_HEADS, n_image],
        "scores": "q . k / 16 over the image positions, before softmax",
        "keys": match kv_mode {
            KvMode::Bf16 => "the BF16 keys attention reads",
            KvMode::HqConsumed => "the rotated-frame scratch rows the hq route consumed (query rotated to match)",
            KvMode::HqPrecodec => "the keys given to the hq codec, NOT what attention reads",
        },
        "verify_failures": verify_failures,
        "head": {"ordinal": HEAD_ORDINAL, "layer": 4 * HEAD_ORDINAL + 3, "q_head": HEAD_Q},
        "region_threshold": REGION_THRESHOLD,
        "guard_distance": GUARD_DISTANCE,
        "first_render": first_render,
        "rows": rows,
    });
    std::fs::write(out_dir.join(format!("{stem}.bin")), &dump)
        .unwrap_or_else(|e| panic!("write the dump: {e}"));
    std::fs::write(
        out_dir.join(format!("{stem}.json")),
        serde_json::to_string_pretty(&meta).expect("serialize"),
    )
    .unwrap_or_else(|e| panic!("write the dump's metadata: {e}"));

    let n = scenes.len();
    eprintln!(
        "attention head point: {set_name} {} {}: head L{}.h{} {head_hits}/{n}, chain {chain_hits}/{n}, \
         guard {guard_hits}/{n}; dump {}",
        render.name(),
        kv_mode.name(),
        4 * HEAD_ORDINAL + 3,
        HEAD_Q,
        out_dir.join(&stem).display()
    );

    // A consumed capture that failed its own check is not a measurement,
    // whatever it scored — asserted before any criterion, after the dump.
    assert!(
        verify_failures.is_empty(),
        "the consumed-hq capture failed its self-check on {} scene(s):\n{}",
        verify_failures.len(),
        verify_failures.join("\n")
    );

    // ── what is asserted, and where ──────────────────────────────────────
    if !generated {
        // The committed fixture at 4096 px. The head is not measured at this
        // size, so it is printed above and not held to anything. The chain is
        // measured inside on all three by `decide_point_gpu.rs` -- through the
        // endpoint, so on the *served* render only; the vehicle render at
        // 4096 in the engine is unmeasured, and asserting it would be the
        // same mistake as asserting the head.
        if render == Render::Served {
            assert_eq!(chain_hits, n, "the chain must land inside on every fixture scene");
        }
        return;
    }
    if n != CRITERION_SCENES {
        return;
    }
    // Each criterion names its set by seed, its render and its keys: the
    // arms it was written for, and no other.
    let criterion = match (manifest.seed, render, kv_mode) {
        (Some(SEED_A), Render::Vehicle, KvMode::Bf16 | KvMode::HqPrecodec) => {
            Some(("A (2026-09-21)", CRITERION_A, CRITERION_GUARD))
        }
        (Some(SEED_B), Render::Vehicle, KvMode::Bf16 | KvMode::HqPrecodec) => {
            Some(("B (2026-09-21)", CRITERION_B, CRITERION_GUARD))
        }
        (Some(SEED_C), Render::Served, KvMode::HqConsumed) if manifest.side == 1024 => {
            Some(("C (2026-09-22)", CRITERION_C_HEAD, CRITERION_C_GUARD))
        }
        // Set C4096 is the production regime and was written down as
        // reported, not evaluated; so is every arm not named above.
        (Some(SEED_C4096), _, _) => None,
        _ => None,
    };
    if let Some((name, head_floor, guard_floor)) = criterion {
        assert!(
            head_hits >= head_floor,
            "pre-registered criterion {name}: L39.h10 must land inside on at least {head_floor}/{n} \
             ({} render, {} keys); it did on {head_hits}",
            render.name(),
            kv_mode.name()
        );
        assert!(
            guard_hits >= guard_floor,
            "pre-registered criterion {name}: the guard at d={GUARD_DISTANCE} must land inside on at \
             least {guard_floor}/{n} ({} render, {} keys); it did on {guard_hits}",
            render.name(),
            kv_mode.name()
        );
    }
}
