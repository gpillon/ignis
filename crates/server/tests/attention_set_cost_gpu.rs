//! Spec 14 acceptance 5 (GitHub #263, ADR 0039): **what reading the head set
//! costs, inside the prefill.**
//!
//! One prompt — the served render of a head point, forced `{"x":` and all —
//! is prefilled over and over through the production path (a
//! `ConcreteScheduler` over the real `RuntimeCompute` and `CudaLeaf`) in three
//! shapes, interleaved so a clock that drifts drifts under all three: a
//! decision that asks for **no attention readout** (a one-token logits
//! readout, which every decision pays anyway), the **pointing head alone**
//! (spec 13: one armed layer), and the **head set** (spec 14: nine armed
//! layers, 96 heads, one fused launch each). The reading chunk's GPU time comes
//! from the leaf's own chunk profile (`IGNIS_CHUNK_PROFILE`, GitHub #92), with
//! prompt reuse off so every run prefills the same chunks.
//!
//! Since spec 15 (GitHub #264, ADR 0040) the set's shape is two launches a
//! layer: the fused reduction, and the gather that scores the four keys
//! around each of that layer's argmaxes. What this measures is both together
//! — there is no build with one and not the other — so the bound is spec
//! 14's plus spec 15's allowance for the gather: at most **0.8 ms** at
//! 1024 px and **1.6 ms** at 4096 px over the pointing head alone. The
//! gather's own cost is the difference from spec 14's recorded measurement
//! on the same path (0.15 ms and 0.21 ms), printed beside the total.
//! Reported: the pointing head alone over no readout, and the same deltas
//! over the nine armed layers' own spans (`IGNIS_CHUNK_PROFILE_LAYERS`).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38). The leaf reads the profile
//! variables through the C runtime, whose copy of the environment is taken at
//! process start — a `set_var` from Rust never reaches it — so each test runs
//! itself again as a child process with both set, and only the child loads
//! the model: `cargo test -p ignis-server --features cuda --test
//! attention_set_cost_gpu -- --ignored --test-threads=1 --nocapture`.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ignis_artifact::{ChatMessage, ChatRenderOptions, ContentPart, FrontendSet, MessageContent, Reader, Role};
use ignis_core::pointing::{AttentionQuery, SetQuery, calibrated_artifacts, calibration};
use ignis_core::types::{DecodeParams, RequestClass, RequestInput, SchedEvent};
use ignis_core::vision::Multimodal;
use ignis_core::{ConcreteScheduler, DecisionRead, Scheduler, Vision, gpu_profile};
use ignis_server::decide::OrderedValue;
use ignis_server::numbers::{DEFAULT_DIGITS, point_system};
use ignis_server::runtime::{EngineShape, cuda_scheduler};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
/// Runs per shape discarded while clocks and caches settle, then measured.
const WARMUP: usize = 3;
const REPS: usize = 25;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    NoReadout,
    PointingHead,
    HeadSet,
}

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

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("pointing").join(name)
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

/// The last chunk of the most recent span in `lines`, as (its GPU span in
/// ms, the summed spans of the GQA layers the set arms).
fn reading_chunk(lines: &[serde_json::Value], armed: &[usize]) -> (f64, f64) {
    let chunk = lines
        .iter()
        .rev()
        .find(|line| line["last"] == 1)
        .expect("the span's last chunk was profiled");
    let (span, offset) = (&chunk["span"], &chunk["chunk_offset"]);
    let layers: f64 = lines
        .iter()
        .filter(|line| &line["span"] == span && &line["chunk_offset"] == offset)
        .filter(|line| line["layer"].as_u64().is_some_and(|l| armed.contains(&(l as usize))))
        .map(|line| line["layer_ms"].as_f64().expect("a layer span"))
        .sum();
    (chunk["gpu_span_ms"].as_f64().expect("a GPU span"), layers)
}

/// Run `test` again in a child process with the chunk profile turned on,
/// and pass or fail with it. `None` in the child itself, which then measures.
fn in_a_profiled_child(test: &str) -> Option<()> {
    if std::env::var_os("IGNIS_CHUNK_PROFILE").is_some() {
        return None;
    }
    let profile = std::env::temp_dir().join(format!("ignis-set-cost-{}-{test}.jsonl", std::process::id()));
    std::fs::remove_file(&profile).ok();
    let status = std::process::Command::new(std::env::current_exe().expect("this test binary"))
        .args([test, "--exact", "--ignored", "--nocapture", "--test-threads=1"])
        .env("IGNIS_CHUNK_PROFILE", &profile)
        .env("IGNIS_CHUNK_PROFILE_LAYERS", "1")
        .status()
        .expect("the profiled child runs");
    std::fs::remove_file(&profile).ok();
    assert!(status.success(), "the profiled run of {test} failed: {status}");
    Some(())
}

fn the_set_costs_under(test: &str, image: &str, instruction: &str, bound_ms: f64, spec14_ms: f64) {
    if in_a_profiled_child(test).is_some() {
        return;
    }
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let profile = PathBuf::from(std::env::var("IGNIS_CHUNK_PROFILE").expect("the child's profile"));
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().expect("an eos token");
    let processor = frontend.vision_processor().unwrap_or_else(|e| panic!("vision processor: {e}"));
    let x_prefix = frontend.tokenizer().encode("{\"x\":").unwrap_or_else(|e| panic!("encode: {e}"));
    let shape = EngineShape {
        vision: Some(Vision::default()),
        prompt_reuse: false,
        retained_slots: 0,
        ..EngineShape::default()
    };
    let mut scheduler: ConcreteScheduler = match cuda_scheduler(path, MODEL.into(), eos, shape) {
        Ok((scheduler, _)) => scheduler,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("cuda_scheduler: {e}"));
            return;
        }
    };
    let bytes = std::fs::read(fixture(image)).unwrap_or_else(|e| panic!("{image}: {e}"));
    let prepared = frontend
        .prepare_prompt(
            &processor,
            &messages(instruction),
            &[bytes.as_slice()],
            ChatRenderOptions { enable_thinking: false, ..Default::default() },
            None,
        )
        .unwrap_or_else(|e| panic!("prepare: {e}"));
    let (mut tokens, mut prompt) = Multimodal::from_prepared(prepared);
    assert!(prompt.append_text(x_prefix.len()));
    tokens.extend_from_slice(&x_prefix);
    let item = &prompt.media[0];
    let (rows, cols) = ((item.grid.h / 2) as u32, (item.grid.w / 2) as u32);
    let (begin, count) = (item.token_span.begin as u32, item.token_span.count as u32);
    let calibrated = calibrated_artifacts().next().and_then(calibration).expect("the served calibration");
    let set = calibrated.set.expect("with a head set");
    // The backbone layers the set arms: GQA ordinal k is layer 4k + 3.
    let mut armed: Vec<usize> = set.heads.iter().map(|h| 4 * h.gqa_ordinal as usize + 3).collect();
    armed.sort_unstable();
    armed.dedup();
    let prompt = Arc::new(prompt);
    let input = |shape: Shape| RequestInput {
        model: MODEL.into(),
        tokens: tokens.clone(),
        params: DecodeParams::default(),
        multimodal: Some(prompt.clone()),
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        decision: Some(match shape {
            Shape::NoReadout => DecisionRead::Answers(Arc::from(vec![tokens[tokens.len() - 1]])),
            Shape::PointingHead | Shape::HeadSet => DecisionRead::Attention(AttentionQuery {
                head: calibrated.head,
                key_begin: begin,
                key_count: count,
                set: (shape == Shape::HeadSet).then(|| SetQuery::for_grid(set, rows, cols)),
            }),
        }),
        constrained: None,
    };

    let mut gpu: [Vec<f64>; 3] = Default::default();
    let mut layers: [Vec<f64>; 3] = Default::default();
    for rep in 0..WARMUP + REPS {
        for (slot, shape) in [Shape::NoReadout, Shape::PointingHead, Shape::HeadSet].into_iter().enumerate() {
            let id = scheduler.submit(input(shape), RequestClass::Agent).expect("admitted");
            let mut ticks = 0;
            'run: loop {
                for event in scheduler.advance() {
                    if let SchedEvent::Done { request, attention, reason, .. } = event
                        && request == id
                    {
                        assert_eq!(
                            attention.is_some(),
                            shape != Shape::NoReadout,
                            "{shape:?}: read what it asked for ({reason:?})"
                        );
                        break 'run;
                    }
                }
                ticks += 1;
                assert!(ticks < 10_000, "{shape:?} never finished");
            }
            let text = std::fs::read_to_string(&profile).expect("the chunk profile");
            let lines: Vec<serde_json::Value> =
                text.lines().map(|line| serde_json::from_str(line).expect("a JSON line")).collect();
            let (span_ms, armed_ms) = reading_chunk(&lines, &armed);
            if rep >= WARMUP {
                gpu[slot].push(span_ms);
                layers[slot].push(armed_ms);
            }
        }
    }
    let [none, head, with_set] = [median(&gpu[0]), median(&gpu[1]), median(&gpu[2])];
    let [l_none, l_head, l_set] = [median(&layers[0]), median(&layers[1]), median(&layers[2])];
    // Paired: each repetition's head-set run against the pointing-head run
    // just before it, so a clock that moved between repetitions cancels —
    // the whole chunk's median alone moves by more than a millisecond at
    // 4096 px from run to run.
    let paired = |a: &[f64], b: &[f64]| median(&a.iter().zip(b).map(|(x, y)| x - y).collect::<Vec<_>>());
    let added = paired(&gpu[2], &gpu[1]);
    eprintln!(
        "set cost [{image}, {rows}x{cols}, {} prompt tokens]: reading chunk GPU median {none:.3} ms \
         no readout (a logits readout), {head:.3} ms pointing head, {with_set:.3} ms head set; paired, \
         the set adds {added:+.3} ms to the chunk and {:+.3} ms to its {} armed layers ({l_none:.3} / \
         {l_head:.3} / {l_set:.3} ms), the pointing head over no readout {:+.3} ms ({REPS} runs each, \
         interleaved)",
        tokens.len(),
        paired(&layers[2], &layers[1]),
        armed.len(),
        paired(&gpu[1], &gpu[0]),
    );
    eprintln!(
        "set cost [{image}]: spec 14 measured {spec14_ms:.2} ms for the fused launches alone on          this path, so the gather is about {:+.3} ms of the {added:+.3} ms total",
        added - spec14_ms,
    );
    assert!(
        added <= bound_ms,
        "[{image}] arming the head set added {added:.3} ms to the reading chunk; the bound is {bound_ms} ms"
    );
}

#[test]
#[ignore = "GPU profile only: run alone, --test-threads=1"]
fn the_head_set_costs_under_half_a_millisecond_at_1024_px() {
    the_set_costs_under(
        "the_head_set_costs_under_half_a_millisecond_at_1024_px",
        "1024/scene0000.png",
        "click the blue button",
        0.8,
        0.15,
    );
}

#[test]
#[ignore = "GPU profile only: run alone, --test-threads=1"]
fn the_head_set_costs_under_a_millisecond_at_4096_px() {
    the_set_costs_under(
        "the_head_set_costs_under_a_millisecond_at_4096_px",
        "large.png",
        "click the blue button",
        1.6,
        0.21,
    );
}
