//! The attention tap's hq capture reads the keys hq-e8-2b attention consumed
//! — and in this engine, attention consumes **every** key from the codec.
//!
//! `ignis_attn_tap_arm_consumed` copies the hq prompt route's own scratch
//! plane after the attention op, found where the vendored op puts it — the
//! first allocation of the workspace `run_gqa_layer` hands it — and in the
//! codec's rotated frame. Both are facts about vendored code this repo does
//! not own, so this test holds them to account on a real prefill.
//!
//! **What the engine actually does.** The vendored prompt route can keep the
//! current chunk, the 32 sink keys and the 512-key ring exact, from residual
//! side planes — but only when the cache view carries them, and ignis never
//! sets `residual_k` / `residual_v` / `ring_valid` (`paged_kv_cache.h`: empty
//! tensors mean the feature is off). With them empty, `has_fresh` and every
//! `side_row` are false and each row goes through `hq_decode_row_group`. The
//! first run of this test expected the three-source rule and found every row,
//! of 66,024 (16,506 positions x 4 KV heads), at the codec's own error — 0.333
//! to 0.784 relative, median 0.3696, which is the per-row median
//! `docs/findings/2026-09-12-hq-attention-route-agreement.md` measured for the
//! codec — and not one row anywhere near an exact row's ~0.004.
//!
//! So the test asserts the two things that separate a right capture from a
//! wrong one, given that:
//!
//! - **correlated at codec level**: the codec's rotation of the key the layer
//!   produced (captured before the cache) sits a median ~0.37 from the
//!   consumed row. A wrong offset, sign vector or Hadamard order puts it past
//!   ~1.3 — uncorrelated vectors of similar norm, which is what the other KV
//!   heads' rows measured at when the first run was taken apart;
//! - **no exact row**: every row is off by at least [`EXACT_ROW_BOUND`]. If
//!   one is not, the residual window has been switched on and the
//!   three-source rule (`ignis_core::attn_tap::hq_source`) now applies —
//!   worth knowing loudly, because it changes what an hq number through this
//!   tap means.
//!
//! Explicit GPU profile (ADR 0006), and the `attn-tap` feature.

#![cfg(all(feature = "cuda", feature = "attn-tap"))]

#[path = "support/mod.rs"]
mod support;

use std::path::{Path, PathBuf};

use ignis_artifact::{
    ChatMessage, ChatRenderOptions, ContentPart, CudaDevice, FrontendSet, MessageContent,
    ModelScope, Reader, Role, bind_model_scope_27b_with, materialize,
};
use ignis_core::attn_tap::{HqSource, KV_HEADS, hq_source, with_attn_tap_hq};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b_with_options;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step;
use ignis_core::vision::{Multimodal, vision_item_control};
use ignis_core::{KvFormat, RopeScaling, Vision};
use ignis_server::decide::OrderedValue;
use ignis_server::numbers::point_system;

use support::vision_canary::prefill_prompt;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 20_480;
const PREFILL_CHUNK: u32 = 1024;
/// L39 is GQA ordinal 9; the head the pointing findings are about.
const ORDINAL: i32 = 9;
const Q_HEAD: usize = 10;
/// Below this, a row reproduces its rotated key: exact, not decoded. Exact
/// rows sit near 0.004 and the codec's rows at 0.33 and above.
const EXACT_ROW_BOUND: f32 = 0.1;
/// The codec's per-row error median sits near 0.37; a capture reading the
/// wrong rows or the wrong frame sits past ~1.3.
const CODEC_MEDIAN_BAND: (f32, f32) = (0.2, 0.6);

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    values[values.len() / 2]
}

#[test]
#[ignore = "GPU profile only, with --features attn-tap"]
fn the_consumed_capture_reads_what_hq_attention_read() {
    let image = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("pointing")
        .join("small.png");
    let Ok(bytes) = std::fs::read(&image) else {
        if gpu_profile::skip_or_fail(&format!("the pointing fixture is absent: {}", image.display())) {
            return;
        }
        unreachable!("skip_or_fail panics under the profile");
    };
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let processor = frontend.vision_processor().unwrap_or_else(|e| panic!("vision processor: {e}"));
    let tokenizer = frontend.tokenizer();
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
    let artifact = materialize(&reader, &plan, &mut device, None)
        .unwrap_or_else(|e| panic!("materialize text + vision: {e}"));
    let model = load_qwen38_27b_with_options(
        &reader,
        &artifact,
        &handles,
        PREFILL_CHUNK,
        MAX_CONTEXT,
        KvFormat::HqE8_2b,
        None,
        Some(Vision::default()),
        RopeScaling::NONE,
    )
    .unwrap_or_else(|e| panic!("ignis_model_load with vision: {e}"));
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: KvFormat::HqE8_2b,
            kv_page_group_count: MAX_CONTEXT / 64,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 1,
            retained_slot_count: 0,
        },
    )
    .unwrap_or_else(|e| panic!("seq pool create: {e}"));

    // The served render: what `/v1/decide` sends.
    let instruction = format!(
        "{{\"instruction\":{}}}",
        OrderedValue::String("click the blue button".to_owned()).to_text()
    );
    let messages = [
        ChatMessage {
            role: Role::System,
            content: MessageContent::Text(point_system(3)),
            tool_calls: Vec::new(),
            reasoning_content: None,
        },
        ChatMessage {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::Image { url: None },
                ContentPart::Text(instruction),
            ]),
            tool_calls: Vec::new(),
            reasoning_content: None,
        },
    ];
    let options = ChatRenderOptions { enable_thinking: false, ..Default::default() };
    let prepared = frontend
        .prepare_prompt(&processor, &messages, &[&bytes], options, None)
        .unwrap_or_else(|e| panic!("prepare: {e}"));
    let (token_ids, mut prompt) = Multimodal::from_prepared(prepared);
    let (begin, count, grid) = {
        let item = &prompt.media[0];
        (item.token_span.begin as usize, item.token_span.count as usize, item.grid)
    };
    let control = vision_item_control(grid);
    let embedding = step::encode_media(&model, grid, &prompt.media[0].patches, &control)
        .unwrap_or_else(|(_, e)| panic!("encode: {e}"));
    let prefix = tokenizer.encode("{\"x\":").unwrap_or_else(|e| panic!("encode prefix: {e}"));
    assert!(prompt.append_text(prefix.len()), "the prompt cannot take the forced prefix");
    let mut tokens: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
    tokens.extend(prefix.iter().map(|&t| t as i32));
    let query = tokens.len() - 1;

    let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
    let mut logits = vec![0f32; ModelConfig::qwen38_27b().vocab as usize];
    let (prefilled, capture) = with_attn_tap_hq(&[ORDINAL], &[query as i64], tokens.len() as i64 + 8, || {
        prefill_prompt(&model, &pool, &mut sequence, &tokens, &prompt, &embedding, PREFILL_CHUNK, &mut logits)
    })
    .unwrap_or_else(|e| panic!("attention tap: {e}"));
    prefilled.unwrap_or_else(|e| panic!("prefill: {e}"));

    assert_eq!(capture.queries_seen, vec![1], "the query row was not captured");
    assert_eq!(
        capture.consumed_rows,
        vec![tokens.len() as i64],
        "the query chunk's attention consumed every prompt key, and the capture must have them all"
    );
    let chunk_start = usize::try_from(capture.consumed_chunk_start[0])
        .unwrap_or_else(|_| panic!("no chunk start was recorded"));

    // Every row against the rotation of the key the layer produced. The
    // three-source rule is reported as what the vendored route *would* keep
    // exact if the residual window were on.
    let mut errors = Vec::with_capacity(tokens.len() * KV_HEADS);
    let (mut rule_exact, mut exact_rows) = (0usize, 0usize);
    for position in 0..tokens.len() {
        if hq_source(position, chunk_start) != HqSource::Codec {
            rule_exact += 1;
        }
        for kv_head in 0..KV_HEADS {
            let err = capture.consumed_key_rel_err(0, position, kv_head);
            if err < EXACT_ROW_BOUND {
                exact_rows += 1;
            }
            errors.push(err);
        }
    }
    let lowest = errors.iter().copied().fold(f32::INFINITY, f32::min);
    let highest = errors.iter().copied().fold(0.0f32, f32::max);
    let codec_median = median(&mut errors.clone());
    let rule_image_codec = (begin..begin + count)
        .filter(|&p| hq_source(p, chunk_start) == HqSource::Codec)
        .count();
    eprintln!(
        "hq consumed keys, L39, 4096 px served: {} prompt tokens, query chunk from {chunk_start}; \
         {} rows x {KV_HEADS} heads, rel err min {lowest:.4} median {codec_median:.4} max {highest:.4}; \
         exact rows {exact_rows} (the three-source rule would keep {rule_exact} positions exact and \
         put {:.1}% of the image keys through the codec)",
        tokens.len(),
        tokens.len(),
        100.0 * rule_image_codec as f64 / count as f64
    );

    // Debug aid: IGNIS_TAP_DUMP=<dir> writes the armed layer's pre-cache and
    // consumed key rows and the query row, raw BF16, for offline inspection.
    if let Ok(dir) = std::env::var("IGNIS_TAP_DUMP") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create dump dir: {e}"));
        let rows = tokens.len() * KV_HEADS * 256;
        let bytes = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        std::fs::write(dir.join("k.bin"), bytes(&capture.k[..rows])).expect("write k");
        std::fs::write(dir.join("kc.bin"), bytes(&capture.kc[..rows])).expect("write kc");
        std::fs::write(dir.join("q.bin"), bytes(&capture.q)).expect("write q");
        std::fs::write(
            dir.join("meta.json"),
            format!(
                "{{\"tokens\":{},\"chunk_start\":{chunk_start},\"begin\":{begin},\"count\":{count}}}",
                tokens.len()
            ),
        )
        .expect("write meta");
    }

    assert!(
        (CODEC_MEDIAN_BAND.0..CODEC_MEDIAN_BAND.1).contains(&codec_median),
        "the consumed rows sit a median {codec_median:.4} from their rotated keys, outside the \
         codec's band {CODEC_MEDIAN_BAND:?}: the capture is not reading the scratch plane the layout \
         says, or the rotation is not the codec's"
    );
    assert_eq!(
        exact_rows, 0,
        "{exact_rows} consumed rows reproduce their rotated key exactly: the hq residual window is on, \
         and the three-source rule (`hq_source`) now decides which keys the codec touched"
    );

    // And the head is scoreable on what attention consumed.
    let positions: Vec<usize> = (begin..begin + count).collect();
    let scores = capture.consumed_scores(0, 0, Q_HEAD, &positions);
    assert!(scores.iter().all(|s| s.is_finite()), "non-finite consumed score");
}
