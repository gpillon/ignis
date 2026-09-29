//! Spec 18 acceptance 3 (GitHub #275, ADR 0041): **the leaf's rows over a
//! text span are the attention's.**
//!
//! A `locate` names the served vote's 32 heads over eight GQA layers, read in
//! whole rows over the state's key span at the copy scaffold's last token.
//! For every head the leaf's row must equal the test-only attention tap's
//! host-side `q . k / 16` over the same span, to float accumulation — the
//! tap is the independent oracle: it copies the layer's rotated query and
//! keys (under hq-e8-2b, the keys the prompt route consumed) to the host and
//! forms the scores in its own code, never touching the new path.
//!
//! Each state goes through the **production path**, a `ConcreteScheduler`
//! over the real `RuntimeCompute` and `CudaLeaf` of a text load — no vision,
//! which is the load the text room must be reserved on — with the prompt
//! rendered the way `/v1/decide` renders a `locate`: layout L1, the kind
//! text, `{"quote":"` forced, the system block reported so the state is
//! retained. Then its **content-free twin** is asked over the same state and
//! claims that retained prefix, as a `locate`'s baseline does in a fan-out:
//! its keys were written by the earlier request, and the leaf must read them
//! where the layer's attention did — the cache's pages under BF16, the
//! prompt route's plane (decoded, with the residual window exact) under
//! hq-e8-2b. The BF16 oracle for a claimed key is the key the first request
//! wrote; the hq oracle is the key the twin's attention consumed.
//!
//! States: a short log, a log near the vote's measured ceiling
//! (`LOCATE_MAX_KEYS`), and a record array.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38), and the `attn-tap` feature:
//! `cargo test -p ignis-server --features cuda,attn-tap --test attention_readout_text_gpu
//! -- --ignored --test-threads=1 --nocapture`, one load at a time.

#![cfg(all(feature = "cuda", feature = "attn-tap"))]

use std::path::Path;

use ignis_artifact::{FrontendSet, Reader};
use ignis_core::attn_tap::{AttnTapCapture, HEAD_DIM, KV_HEADS, Q_HEADS, bf16_to_f32, with_attn_tap, with_attn_tap_hq};
use ignis_core::locate::{LocateCalibration, calibrated_artifacts, calibration};
use ignis_core::pointing::{AttentionQuery, AttentionScores, PointingHead, SetQuery};
use ignis_core::types::{DecodeParams, RequestClass, RequestInput, SchedEvent};
use ignis_core::{ConcreteScheduler, DecisionRead, KvFormat, Scheduler, gpu_profile};
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::decide::OrderedValue;
use ignis_server::locate::{COPY_OPENING, CONTENT_FREE, evidence_within, map_segments, user_text};
use ignis_server::runtime::{EngineShape, cuda_scheduler};
use ignis_server::template::{ChatMessage, TemplateProvider};
use ignis_server::thinking::ThinkingOptions;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
/// "Within float accumulation", as `attention_readout_gpu.rs` has it: the
/// leaf and the tap multiply the same BF16 values in f32 and differ only in
/// the order they add them.
const SCORE_TOLERANCE: f32 = 2e-3;

/// A deterministic service log of `lines` lines, the shape of
/// `tools/locate-sets/logs.py`'s. `salt` makes two logs differ from their
/// first line, so neither claims the other's retained state.
fn log(lines: usize, salt: usize) -> OrderedValue {
    let services = ["billing", "gateway", "ledger", "orders", "search", "mailer"];
    let levels = ["INFO", "DEBUG", "WARN", "ERROR"];
    let text: Vec<String> = (0..lines)
        .map(|i| {
            format!(
                "{:02}:{:02} {} {}: request r_{} finished in {} ms",
                9 + i / 60,
                i % 60,
                levels[(i * 7) % levels.len()],
                services[(i * 5 + salt) % services.len()],
                1000 + i * 37 % 900,
                (i * 13) % 400
            )
        })
        .collect();
    OrderedValue::String(text.join("\n"))
}

fn records(count: usize) -> OrderedValue {
    let cities = ["Canberra", "Oslo", "Lima", "Porto", "Kyoto", "Accra"];
    let text: Vec<String> = (0..count)
        .map(|i| {
            format!(
                r#"{{"id":"E-{:04}","name":"Employee {i}","city":"{}","since":{}}}"#,
                1000 + i,
                cities[(i * 11) % cities.len()],
                2010 + i % 14
            )
        })
        .collect();
    serde_json::from_str(&format!("[{}]", text.join(","))).expect("records parse")
}

/// A `locate`'s request over `state` with `instruction`, as `/v1/decide`
/// builds one, and its key span.
fn locate_input(
    provider: &ArtifactTemplateProvider,
    opening: &[u32],
    calibration: LocateCalibration,
    state: &OrderedValue,
    instruction: &str,
) -> (RequestInput, std::ops::Range<usize>) {
    let evidence = evidence_within(state, "").expect("segmentable");
    let messages = [
        ChatMessage::text("system", evidence.system.clone()),
        ChatMessage::text("user", user_text(evidence.unit, &OrderedValue::String(instruction.to_owned()))),
    ];
    let thinking = ThinkingOptions { enable_thinking: false, ..ThinkingOptions::default() };
    let rendered = provider.apply_chat_template_with_text(&messages, &thinking, &[]).expect("render");
    let text = rendered.text.clone().expect("the render's text");
    let (span, _keys) = map_segments(&text.text, &text.offsets, &evidence).expect("the segments map");
    assert!(span.len() <= calibration.max_keys as usize, "{} keys, past the vote's ceiling", span.len());
    let mut tokens = rendered.tokens.clone();
    tokens.extend_from_slice(opening);
    let input = RequestInput {
        model: MODEL.into(),
        tokens,
        params: DecodeParams::default(),
        multimodal: None,
        opener_tokens: rendered.opener_tokens,
        user_turn_tokens: rendered.user_turn_tokens,
        system_block_tokens: rendered.system_block_tokens,
        reuse_boundaries: Vec::new(),
        decision: Some(DecisionRead::Attention(AttentionQuery {
            head: calibration.heads[0],
            key_begin: span.start as u32,
            key_count: span.len() as u32,
            set: Some(SetQuery::rows(calibration.heads)),
        })),
        constrained: None,
        warm_up: false,
    };
    (input, span)
}

/// Drive one request to its end, returning its attention and how many prompt
/// tokens it claimed from a retained prefix.
fn run(scheduler: &mut ConcreteScheduler, input: RequestInput) -> (Option<AttentionScores>, u32) {
    let id = scheduler.submit(input, RequestClass::Agent).expect("admitted");
    let mut claimed = 0;
    for _ in 0..10_000 {
        for event in scheduler.advance() {
            match event {
                SchedEvent::PrefixReused { request, tokens, .. } if request == id => claimed = tokens,
                SchedEvent::Done { request, attention, reason, .. } if request == id => {
                    assert!(attention.is_some(), "the leaf read no rows ({reason:?})");
                    return (attention, claimed);
                }
                _ => {}
            }
        }
    }
    panic!("the locate never finished");
}

/// `q . k / 16` for `head`, its query from `asked` and each key from `keys`
/// at the same armed layer.
fn dot(asked: &AttnTapCapture, keys: &AttnTapCapture, layer: usize, head: usize, position: usize) -> f32 {
    let q = asked.query(layer, 0, head);
    let k = keys.key(layer, position, head / (Q_HEADS / KV_HEADS));
    q.iter().zip(k).map(|(&a, &b)| bf16_to_f32(a) * bf16_to_f32(b)).sum::<f32>() / (HEAD_DIM as f32).sqrt()
}

/// Every head's row against `oracle(layer, head, key)`, the worst relative
/// difference reported.
fn check_rows(
    label: &str,
    heads: &[PointingHead],
    ordinals: &[i32],
    rows: &[f32],
    count: usize,
    oracle: impl Fn(usize, usize, usize) -> f32,
) {
    assert_eq!(rows.len(), heads.len() * count, "{label}: heads x span");
    let mut worst = 0f32;
    for (h, head) in heads.iter().enumerate() {
        let layer = ordinals.iter().position(|&o| o == head.gqa_ordinal as i32).expect("armed");
        for key in 0..count {
            let (leaf, tap) = (rows[h * count + key], oracle(layer, head.query_head as usize, key));
            let diff = (leaf - tap).abs() / tap.abs().max(1.0);
            worst = worst.max(diff);
            assert!(diff <= SCORE_TOLERANCE, "{label}: {head} key {key}: leaf {leaf}, tap {tap}");
        }
    }
    eprintln!("  {label}: {} heads x {count} keys, worst relative difference {worst:.2e}", heads.len());
}

fn the_leaf_reads_a_text_span_as_the_tap_sees_it(kv_format: KvFormat) {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().expect("an eos token");
    let opening = frontend.tokenizer().encode(COPY_OPENING).expect("encode the scaffold");
    let provider = ArtifactTemplateProvider::new(FrontendSet::from_reader(&reader).expect("frontend"));
    let calibration = calibrated_artifacts().next().and_then(calibration).expect("the served vote");
    let shape = EngineShape { kv_format, ..EngineShape::default() };
    assert!(shape.vision.is_none(), "a text load: the room is reserved without vision");
    let mut scheduler = match cuda_scheduler(path, MODEL.into(), eos, shape) {
        Ok((scheduler, _)) => scheduler,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("cuda_scheduler: {e}"));
            return;
        }
    };
    let hq = kv_format == KvFormat::HqE8_2b;
    let mut ordinals: Vec<i32> = calibration.heads.iter().map(|head| head.gqa_ordinal as i32).collect();
    ordinals.sort_unstable();
    ordinals.dedup();

    let states = [
        ("log-60", log(60, 0), "Which line reports the slowest billing request?"),
        ("log-198", log(198, 1), "Which line says the gateway request r_1500 failed?"),
        ("records-40", records(40), "Which employee works from Kyoto and joined in 2015?"),
    ];
    for (label, state, instruction) in &states {
        let (asked, span) = locate_input(&provider, &opening, calibration, state, instruction);
        let (twin, twin_span) = locate_input(&provider, &opening, calibration, state, CONTENT_FREE);
        assert_eq!(span, twin_span, "{label}: the twin maps the state onto the same keys");
        let count = span.len();
        let positions: Vec<usize> = span.clone().collect();

        // The question, prefilled whole.
        let (asked_tokens, twin_tokens) = (asked.tokens.len(), twin.tokens.len());
        let tap = |tokens: usize, f: &mut dyn FnMut() -> (Option<AttentionScores>, u32)| {
            let queries = [tokens as i64 - 1];
            match hq {
                true => with_attn_tap_hq(&ordinals, &queries, tokens as i64 + 8, f),
                false => with_attn_tap(&ordinals, &queries, tokens as i64 + 8, f),
            }
            .unwrap_or_else(|e| panic!("{label}: attention tap: {e}"))
        };
        let ((first, claimed_first), first_capture) = tap(asked_tokens, &mut || run(&mut scheduler, asked.clone()));
        assert_eq!(claimed_first, 0, "{label}: the first request claims nothing");
        let first = first.expect("read");
        let rows = first.set_rows.as_deref().expect("the rows");
        check_rows(&format!("{label} [{kv_format:?}] question"), calibration.heads, &ordinals, rows, count, |layer, head, key| {
            match hq {
                true => first_capture.consumed_scores(layer, 0, head, &positions[key..key + 1])[0],
                false => dot(&first_capture, &first_capture, layer, head, positions[key]),
            }
        });

        // Its content-free twin, over the state the question retained.
        let ((second, claimed), twin_capture) = tap(twin_tokens, &mut || run(&mut scheduler, twin.clone()));
        assert!(
            claimed as usize > span.start,
            "{label}: the twin claims the retained state ({claimed} tokens; the span starts at {})",
            span.start
        );
        let second = second.expect("read");
        let rows = second.set_rows.as_deref().expect("the rows");
        check_rows(&format!("{label} [{kv_format:?}] twin, {claimed} tokens claimed"), calibration.heads, &ordinals, rows, count, |layer, head, key| {
            let position = positions[key];
            match (hq, position < claimed as usize) {
                (true, _) => twin_capture.consumed_scores(layer, 0, head, &[position])[0],
                // A claimed key is the one the first request wrote.
                (false, true) => dot(&twin_capture, &first_capture, layer, head, position),
                (false, false) => dot(&twin_capture, &twin_capture, layer, head, position),
            }
        });
    }
}

#[test]
#[ignore = "GPU profile only (attn-tap): --test-threads=1"]
fn under_bf16_the_leaf_reads_a_text_span_as_the_tap_sees_it() {
    the_leaf_reads_a_text_span_as_the_tap_sees_it(KvFormat::Bf16);
}

#[test]
#[ignore = "GPU profile only (attn-tap): --test-threads=1"]
fn under_hq_the_leaf_reads_a_text_span_as_the_tap_sees_it() {
    the_leaf_reads_a_text_span_as_the_tap_sees_it(KvFormat::HqE8_2b);
}
