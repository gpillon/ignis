//! Spec 13 acceptance 1 (GitHub #260, ADR 0038) and spec 14 acceptance 1
//! (GitHub #263, ADR 0039): **the leaf's reads are the attention's.**
//!
//! Each head point here names the served calibration's **head set** too, as
//! `/v1/decide` does: 96 heads over nine armed GQA layers, the tap armed on
//! every one of them. For every head of the set the leaf's device argmax must
//! be the argmax of the tap's host-side `q . k / 16` over the span minus the
//! fallback cells — a different key only where the two scores tie to float
//! accumulation — and the pointing head's row, now written in the fused
//! launch, must still equal the tap's.
//!
//! The attention readout is production code: the scheduler asks for it on a
//! head point's last chunk, `RuntimeCompute` hands it to the leaf, and the
//! GQA layer scores the image span right after its attention, from the keys
//! that attention read. The test-only attention tap
//! (`kernel/include/ignis_attn_tap.h`) is the oracle, and an independent one:
//! it copies the layer's rotated query and keys (under hq-e8-2b, the keys the
//! prompt route consumed) to the host and forms `q . k / 16` there, in its own
//! code, never touching the new path. The spec's accuracy numbers were
//! measured on exactly the tap's scores.
//!
//! Each scene goes through the **production path**: a `ConcreteScheduler`
//! over the real `RuntimeCompute` and `CudaLeaf`, a `RequestInput` built the
//! way `/v1/decide` builds a head point — the served render, the forced
//! `{"x":` — so the chunk cut, the tail rule and the reuse trim are the
//! served ones. The tap is armed on the same layer at the same position, and
//! the `Done` event's scores are compared with it, then read into a point by
//! the same region rule both ways.
//!
//! Scenes: the committed 4096 px pointing fixture (three) and a handful of
//! 1024 px scenes (`fixtures/pointing/1024`, the first five of set C). Under
//! BF16 and under hq-e8-2b with its residual window, one load each.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38), and the `attn-tap` feature,
//! which `scripts/gpu-profile.ps1` does not turn on — run it by hand under
//! the profile's preflight, one load at a time:
//! `cargo test -p ignis-server --features cuda,attn-tap --test attention_readout_gpu
//! -- --ignored --test-threads=1 --nocapture`.

#![cfg(all(feature = "cuda", feature = "attn-tap"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ignis_artifact::{ChatMessage, ChatRenderOptions, ContentPart, FrontendSet, MessageContent, Reader, Role};
use ignis_core::attn_tap::{with_attn_tap, with_attn_tap_hq};
use ignis_core::pointing::{AttentionQuery, PointingHead, SetQuery, calibrated_artifacts, calibration, read_head_map};
use ignis_core::types::{DecodeParams, RequestClass, RequestInput, SchedEvent};
use ignis_core::vision::Multimodal;
use ignis_core::{ConcreteScheduler, DecisionRead, KvFormat, Scheduler, Vision, gpu_profile};
use ignis_server::decide::OrderedValue;
use ignis_server::numbers::{DEFAULT_DIGITS, point_system};
use ignis_server::runtime::{EngineShape, cuda_scheduler};
use serde::Deserialize;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
const HEAD: PointingHead = PointingHead {
    gqa_ordinal: 9,
    query_head: 10,
};
/// "Within float accumulation": the leaf and the tap multiply the same BF16
/// query and key values in f32 and differ only in the order they add 256
/// products (and, under hq, 256-point butterflies). Scores run to a few
/// tens; a difference past this is a different key, a different frame or a
/// different position, not arithmetic.
const SCORE_TOLERANCE: f32 = 2e-3;
/// The two points, read by the same rule from maps this close, agree to far
/// better than a cell: a region picked differently would move a whole cell.
const POINT_TOLERANCE_CELLS: f64 = 1e-3;

#[derive(Deserialize)]
struct Manifest {
    scenes: Vec<Scene>,
}

#[derive(Deserialize)]
struct Scene {
    id: String,
    image: String,
    #[serde(default)]
    instruction: Option<String>,
}

fn fixture_dir(name: &str) -> PathBuf {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("pointing");
    match name {
        "4096" => base,
        other => base.join(other),
    }
}

/// The served render's two messages: `point_system`, then the image and the
/// instruction as `/v1/decide` serializes it.
fn messages(instruction: &str) -> [ChatMessage; 2] {
    [
        ChatMessage {
            role: Role::System,
            content: MessageContent::Text(point_system(DEFAULT_DIGITS)),
            tool_calls: Vec::new(),
            reasoning_content: None,
        },
        ChatMessage {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::Image { url: None },
                ContentPart::Text(format!(
                    "{{\"instruction\":{}}}",
                    OrderedValue::String(instruction.to_owned()).to_text()
                )),
            ]),
            tool_calls: Vec::new(),
            reasoning_content: None,
        },
    ]
}

fn served_options() -> ChatRenderOptions {
    ChatRenderOptions {
        enable_thinking: false,
        ..Default::default()
    }
}

/// Every scene the test compares on, as (label, image bytes, instruction).
fn scenes() -> Option<Vec<(String, Vec<u8>, String)>> {
    let mut out = Vec::new();
    for set in ["4096", "1024"] {
        let dir = fixture_dir(set);
        let Ok(text) = std::fs::read_to_string(dir.join("manifest.json")) else {
            gpu_profile::skip_or_fail(&format!("no scene manifest in {}", dir.display()));
            return None;
        };
        let manifest: Manifest = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{set}: {e}"));
        for scene in manifest.scenes {
            let bytes = std::fs::read(dir.join(&scene.image))
                .unwrap_or_else(|e| panic!("{set}/{}: {e}", scene.id));
            let instruction = scene.instruction.unwrap_or_else(|| "click the blue button".to_owned());
            out.push((format!("{set}/{}", scene.id), bytes, instruction));
        }
    }
    Some(out)
}

fn the_leaf_reads_what_the_tap_sees(kv_format: KvFormat) {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let Some(scenes) = scenes() else { return };
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().expect("an eos token");
    let processor = frontend.vision_processor().unwrap_or_else(|e| panic!("vision processor: {e}"));
    let x_prefix = frontend.tokenizer().encode("{\"x\":").unwrap_or_else(|e| panic!("encode: {e}"));
    let shape = EngineShape {
        vision: Some(Vision::default()),
        kv_format,
        ..EngineShape::default()
    };
    let mut scheduler: ConcreteScheduler = match cuda_scheduler(path, MODEL.into(), eos, shape) {
        Ok((scheduler, _)) => scheduler,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("cuda_scheduler: {e}"));
            return;
        }
    };
    let hq = kv_format == KvFormat::HqE8_2b;

    for (label, bytes, instruction) in &scenes {
        let prepared = frontend
            .prepare_prompt(&processor, &messages(instruction), &[bytes.as_slice()], served_options(), None)
            .unwrap_or_else(|e| panic!("{label}: prepare: {e}"));
        let (mut tokens, mut prompt) = Multimodal::from_prepared(prepared);
        assert!(prompt.append_text(x_prefix.len()), "{label}: the forced prefix extends the prompt");
        tokens.extend_from_slice(&x_prefix);
        let item = &prompt.media[0];
        let (begin, count) = (item.token_span.begin, item.token_span.count);
        let (rows, cols) = ((item.grid.h / 2) as usize, (item.grid.w / 2) as usize);
        assert_eq!(rows * cols, count, "{label}: the span is the merged grid");
        let query_position = tokens.len() - 1;
        let set = calibrated_artifacts()
            .next()
            .and_then(calibration)
            .and_then(|calibration| calibration.set)
            .expect("the served calibration has a head set");
        let set_query = SetQuery::for_grid(set, rows as u32, cols as u32);
        let input = RequestInput {
            model: MODEL.into(),
            tokens: tokens.clone(),
            params: DecodeParams::default(),
            multimodal: Some(Arc::new(prompt)),
            opener_tokens: None,
            user_turn_tokens: None,
            system_block_tokens: None,
            decision: Some(DecisionRead::Attention(AttentionQuery {
                head: HEAD,
                key_begin: begin as u32,
                key_count: count as u32,
                set: Some(set_query.clone()),
            })),
            constrained: None,
        };

        // One armed prefill, driven by the scheduler to its completion.
        let run = || {
            let id = scheduler.submit(input, RequestClass::Agent).expect("admitted");
            let mut ticks = 0;
            loop {
                for event in scheduler.advance() {
                    if let SchedEvent::Done { request, attention, reason, .. } = event
                        && request == id
                    {
                        return (attention, reason);
                    }
                }
                ticks += 1;
                assert!(ticks < 10_000, "the head point never finished");
            }
        };
        // Every armed layer: the pointing head's and each holding a head of
        // the set.
        let mut ordinals: Vec<i32> = set.heads.iter().map(|head| head.gqa_ordinal as i32).collect();
        ordinals.push(HEAD.gqa_ordinal as i32);
        ordinals.sort_unstable();
        ordinals.dedup();
        let layer_of = |ordinal: u32| ordinals.iter().position(|&o| o == ordinal as i32).expect("armed");
        let queries = [query_position as i64];
        let max_positions = tokens.len() as i64 + 8;
        let ((attention, reason), capture) = match hq {
            true => with_attn_tap_hq(&ordinals, &queries, max_positions, run),
            false => with_attn_tap(&ordinals, &queries, max_positions, run),
        }
        .unwrap_or_else(|e| panic!("{label}: attention tap: {e}"));
        let leaf = attention.unwrap_or_else(|| panic!("{label}: the leaf read no scores ({reason:?})"));
        assert_eq!(capture.queries_seen, vec![1], "{label}: the tap saw the query row");
        if hq {
            for layer in 0..ordinals.len() {
                assert_eq!(
                    capture.consumed_rows[layer],
                    tokens.len() as i64,
                    "{label}: the tap captured the keys the prompt route consumed at ordinal {}",
                    ordinals[layer]
                );
            }
        }

        let positions: Vec<usize> = (begin..begin + count).collect();
        let tap_scores = |head: PointingHead| match hq {
            true => capture.consumed_scores(layer_of(head.gqa_ordinal), 0, head.query_head as usize, &positions),
            false => capture.scores(layer_of(head.gqa_ordinal), 0, head.query_head as usize, &positions),
        };

        // ── the set: every head's argmax against the tap's ───────────────
        let argmax = leaf.set_argmax.as_deref().unwrap_or_else(|| panic!("{label}: no head set came back"));
        assert_eq!(argmax.len(), set.heads.len(), "{label}: one key per head of the set");
        let mut ties = 0usize;
        for (head, &key) in set.heads.iter().zip(argmax) {
            let oracle = tap_scores(*head);
            let (best, top) = oracle
                .iter()
                .enumerate()
                .filter(|(k, _)| !set_query.excluded.contains(&(*k as u32)))
                .fold((0usize, f32::NEG_INFINITY), |(bk, bs), (k, &s)| if s >= bs { (k, s) } else { (bk, bs) });
            assert!(
                !set_query.excluded.contains(&key),
                "{label}: {head} landed on excluded key {key}"
            );
            if key as usize != best {
                let gap = (top - oracle[key as usize]).abs() / top.abs().max(1.0);
                assert!(
                    gap <= SCORE_TOLERANCE,
                    "{label}: {head} peaks at key {key} (tap score {}), the tap at key {best} ({top})",
                    oracle[key as usize]
                );
                ties += 1;
            }
        }

        // ── the pointing head's row, written in the fused launch ─────────
        let leaf = leaf.scores;
        let oracle = tap_scores(HEAD);
        assert_eq!(leaf.len(), oracle.len(), "{label}: one score per key of the span");
        let (mut worst, mut at) = (0f32, 0usize);
        for (k, (&a, &b)) in leaf.iter().zip(&oracle).enumerate() {
            let diff = (a - b).abs() / b.abs().max(1.0);
            if diff > worst {
                (worst, at) = (diff, k);
            }
        }
        let from_leaf = read_head_map(&leaf, rows, cols).expect("the leaf's map reads");
        let from_tap = read_head_map(&oracle, rows, cols).expect("the tap's map reads");
        eprintln!(
            "attention readout [{kv_format:?}] {label}: {count} keys ({rows}x{cols}), worst diff \
             {worst:.2e} at key {at} (leaf {} tap {}); point leaf ({:.3}, {:.3}) tap ({:.3}, {:.3}) \
             cells, share {:.3}; {} set heads on {} layers, {ties} argmax ties",
            leaf[at],
            oracle[at],
            from_leaf.x,
            from_leaf.y,
            from_tap.x,
            from_tap.y,
            from_leaf.share,
            set.heads.len(),
            ordinals.len()
        );
        assert!(
            worst <= SCORE_TOLERANCE,
            "{label}: the leaf's score at key {at} is {} and the tap's {} — past float accumulation",
            leaf[at],
            oracle[at]
        );
        assert!(
            (from_leaf.x - from_tap.x).abs() < POINT_TOLERANCE_CELLS
                && (from_leaf.y - from_tap.y).abs() < POINT_TOLERANCE_CELLS
                && from_leaf.cells == from_tap.cells,
            "{label}: the region rule reads ({}, {}) from the leaf and ({}, {}) from the tap",
            from_leaf.x,
            from_leaf.y,
            from_tap.x,
            from_tap.y
        );
    }
}

#[test]
#[ignore = "GPU profile only (attn-tap): scripts/gpu-profile.ps1, --test-threads=1"]
fn under_bf16_the_leaf_reads_what_the_tap_sees() {
    the_leaf_reads_what_the_tap_sees(KvFormat::Bf16);
}

#[test]
#[ignore = "GPU profile only (attn-tap): scripts/gpu-profile.ps1, --test-threads=1"]
fn under_hq_the_leaf_reads_what_the_tap_sees() {
    the_leaf_reads_what_the_tap_sees(KvFormat::HqE8_2b);
}
