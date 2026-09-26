//! Where does the model's attention point in a **text** state? The
//! calibration harness of spec 18 phase A
//! (`docs/specs/decide/18-locate-by-attention.md`, GitHub #274).
//!
//! For every question of a set written by `tools/locate-sets/generate.py`,
//! the state is rendered the way a `locate` would be — layout L1: the system
//! message `{"evidence": …}` alone, the user message the kind text and
//! `{"instruction": …}`, the assistant turn opening with the forced scaffold
//! (`support/locate.rs`) — and prefilled four times:
//!
//! - `s1`: the index-shaped scaffold (`{"line":` / `{"item":`);
//! - `s2`: the copy-shaped scaffold (`{"quote":"`);
//! - `s1-na` and `s2-na`: the same with the instruction replaced by `N/A`,
//!   the **content-free baseline** a reading may subtract.
//!
//! The four share the state's bytes, and the segment map is asserted
//! identical across them. Each prefill arms the test-only attention tap on
//! every GQA layer and dumps, for every (layer, query head), the scores
//! `q . k / 16` at the last position — the scaffold's last token — over the
//! state's **key span**: from the first segment-owned token to the last.
//! `tools/locate-sets/score.py` turns the dump into the readings R1, R2 and
//! R3 and applies the spec's pre-registered rules.
//!
//! **The keys are the ones attention read.** Under hq-e8-2b (the default,
//! `IGNIS_LOCATE_KV=hq`) they are the rotated-frame scratch rows the hq
//! prompt route consumed, captured after the op (`with_attn_tap_hq`), with
//! the residual window wired: fresh, sink and ring rows exact, the rest
//! decoded by the codec. The capture checks itself on every armed layer — one
//! KV head per layer, a different one each — as
//! `attention_head_point_gpu.rs` checks its one layer: each row classified by
//! `ignis_core::hq_ring::prompt_source` must be exact where the rule keeps it
//! and decoded where it does not, and the decoded rows' median must sit in
//! the codec's own band. **A capture that fails the check is not a
//! measurement**: the dump is written, then the test fails.
//! `IGNIS_LOCATE_KV=bf16` measures a BF16 pool instead.
//!
//! **Chunks** are cut as `ConcreteScheduler::chunk_take` cuts a decision's
//! text prompt with an attention readout: `IGNIS_LOCATE_CHUNK` wide (default
//! 1024, the serving default), the last kept at least
//! `ATTENTION_MIN_CHUNK_TOKENS` wide so that the hq route materializes the
//! keys. A fresh prefill, with no retained prefix: a served fan-out that
//! claims the state from an earlier request reads the same keys from the
//! cache, where all but the residual window is decoded.
//!
//! **The render is the endpoint's machinery**: `ArtifactTemplateProvider`
//! with thinking off, as `/v1/decide` renders, and the prompt tokens are held
//! to the provider's own tokenization of it — two sources for the bytes
//! every head was chosen on. The first render of each (scaffold, unit) is in
//! the dump, for phase B's prompt-pinning test.
//!
//! `IGNIS_LOCATE_SET=<dir with manifest.json>` (required),
//! `IGNIS_LOCATE_OUT=<dir>` for the dump (default: the OS temp dir),
//! `IGNIS_LOCATE_LIMIT=<n>` for a smoke run, and `IGNIS_LOCATE_RESUME=1` to
//! continue a dump a crash cut short. The dump is `<set>-<kv>.bin` — f16
//! little-endian, per question and per variant in the order above,
//! `[GQA layer 0..16][query head 0..24][span key]` — rows in `<set>-<kv>.jsonl`
//! as each question finishes, and `<set>-<kv>.json` with the rest at the end.
//!
//! Nothing is asserted about where the heads point: phase A's rules are
//! applied by the scorer, on sets chosen for it. What is asserted is that the
//! measurement is one: the render, the token map, the capture.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38), and the `attn-tap` feature.

#![cfg(all(feature = "cuda", feature = "attn-tap"))]

#[path = "support/locate.rs"]
mod locate;

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use ignis_artifact::{CudaDevice, FrontendSet, ModelScope, Reader, bind_model_scope_27b_with, materialize};
use ignis_core::attn_tap::{AttnTapCapture, GQA_LAYERS, KV_HEADS, Q_HEADS, with_attn_tap, with_attn_tap_hq};
use ignis_core::compute::ModelConfig;
use ignis_core::decision::AnswerAlphabet;
use ignis_core::gpu_profile;
use ignis_core::hq_ring::{PromptSource, prompt_source, ring_before_chunk};
use ignis_core::model_load::load_qwen38_27b_with_options;
use ignis_core::pointing::ATTENTION_MIN_CHUNK_TOKENS;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step;
use ignis_core::{KvFormat, RopeScaling};
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::decide::OrderedValue;
use ignis_server::template::{ChatMessage, TemplateProvider};
use ignis_server::thinking::ThinkingOptions;
use serde::Deserialize;

use locate::{CONTENT_FREE, Scaffold, Segment, chunks, evidence, key_span, owners, segment_keys, user_text};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// The longest set state, a 1,000-line log, renders about 18K tokens.
const MAX_CONTEXT: u32 = 20_480;
const DEFAULT_PREFILL_CHUNK: u32 = 1024;
const SERVED_TAIL: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// The pointing head (L39.h10), scored on text as one more R1 candidate by
/// the scorer and printed here as a live signal.
const POINTING_ORDINAL: usize = 9;
const POINTING_Q: usize = 10;

/// The consumed capture's self-check bounds, as `attention_head_point_gpu.rs`
/// sets them: an exact row sits near 0.002-0.004 relative L2 from the rotated
/// pre-codec key, the codec's rows at a median ~0.37.
const EXACT_ROW_REL_ERR: f64 = 0.1;
const CODEC_ROW_MIN_MEDIAN: f64 = 0.2;
const CODEC_ROW_MAX_MEDIAN: f64 = 0.6;

#[derive(Deserialize)]
struct Manifest {
    #[serde(default)]
    seed: Option<u64>,
    questions: Vec<Question>,
}

#[derive(Deserialize)]
struct Question {
    id: String,
    family: String,
    #[serde(default)]
    kind: Option<String>,
    split: String,
    absent: bool,
    state: OrderedValue,
    instruction: String,
    targets: Vec<usize>,
    #[serde(default)]
    distractors: Vec<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum KvMode {
    Bf16,
    HqConsumed,
}

impl KvMode {
    fn from_env() -> Self {
        match std::env::var("IGNIS_LOCATE_KV").as_deref() {
            Err(_) | Ok("hq") => Self::HqConsumed,
            Ok("bf16") => Self::Bf16,
            Ok(other) => panic!("IGNIS_LOCATE_KV must be hq or bf16, not {other:?}"),
        }
    }

    fn format(self) -> KvFormat {
        match self {
            Self::Bf16 => KvFormat::Bf16,
            Self::HqConsumed => KvFormat::HqE8_2b,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::HqConsumed => "hq",
        }
    }
}

/// The four prefills of one question, in dump order.
const VARIANTS: [(Scaffold, bool); 4] = [
    (Scaffold::Index, false),
    (Scaffold::Index, true),
    (Scaffold::Quote, false),
    (Scaffold::Quote, true),
];

fn variant_name(scaffold: Scaffold, content_free: bool) -> String {
    match content_free {
        false => scaffold.name().to_owned(),
        true => format!("{}-na", scaffold.name()),
    }
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    Some(values[values.len() / 2])
}

/// f32 to IEEE half, round to nearest even (as `attention_head_point_gpu.rs`
/// writes its dump).
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

/// The consumed capture's self-check on one armed layer and one KV head:
/// every prompt row classified by the hq route's rule and compared with the
/// rotated pre-codec key (`attention_head_point_gpu.rs`'s check, per layer).
/// The failures, and the layer's row counts by class.
fn check_layer(
    capture: &AttnTapCapture,
    layer: usize,
    kv_head: usize,
    total: usize,
    query_chunk_start: usize,
    ring: &ignis_core::hq_ring::HqRing,
) -> (Vec<String>, serde_json::Value) {
    let mut failures = Vec::new();
    let consumed = capture.consumed_rows[layer] as usize;
    if consumed != total {
        failures.push(format!("layer {layer} consumed {consumed} rows, expected {total} (not captured?)"));
        return (failures, serde_json::Value::Null);
    }
    let captured_start = capture.consumed_chunk_start[layer];
    if captured_start != query_chunk_start as i64 {
        failures.push(format!(
            "layer {layer}: the capture's chunk starts at {captured_start}, the prompt's chunking puts \
             the query's chunk at {query_chunk_start}"
        ));
    }
    let (mut exact, mut codec) = (Vec::new(), Vec::new());
    let mut off_rule = 0usize;
    for position in 0..total {
        let err = f64::from(capture.consumed_key_rel_err(layer, position, kv_head));
        match prompt_source(position as u64, query_chunk_start as u64, ring) {
            PromptSource::Fresh | PromptSource::Sink | PromptSource::Ring => {
                off_rule += usize::from(err >= EXACT_ROW_REL_ERR);
                exact.push(err);
            }
            // The reference's order, never ignis's: off the rule whatever it
            // measures.
            PromptSource::Clobbered { .. } => off_rule += 1,
            PromptSource::Codec => {
                off_rule += usize::from(err < EXACT_ROW_REL_ERR);
                codec.push(err);
            }
        }
    }
    if off_rule > 0 {
        failures.push(format!(
            "layer {layer}, KV head {kv_head}: {off_rule} of {total} rows are not what the hq prompt \
             route's rule says (exact where it keeps a row, decoded where it does not)"
        ));
    }
    // A prompt the residual window covers has no codec row, which is nothing
    // for the band to check rather than a capture that failed it.
    let codec_median = median(&mut codec.clone());
    if let Some(m) = codec_median {
        if !(CODEC_ROW_MIN_MEDIAN..=CODEC_ROW_MAX_MEDIAN).contains(&m) {
            failures.push(format!(
                "layer {layer}, KV head {kv_head}: the decoded keys sit at median rel L2 {m:.4}, outside \
                 the codec's band [{CODEC_ROW_MIN_MEDIAN}, {CODEC_ROW_MAX_MEDIAN}]"
            ));
        }
    }
    let stats = serde_json::json!({
        "exact": exact.len(),
        "codec": codec.len(),
        "exact_median": median(&mut exact),
        "codec_median": codec_median,
    });
    (failures, stats)
}

/// Every armed head's scores over `span`, `[layer][q_head][key]`, one thread
/// per layer.
fn span_scores(capture: &AttnTapCapture, kv_mode: KvMode, span: &std::ops::Range<usize>) -> Vec<Vec<f32>> {
    let positions: Vec<usize> = span.clone().collect();
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..capture.ordinals.len())
            .map(|layer| {
                let positions = &positions;
                scope.spawn(move || {
                    let mut out = Vec::with_capacity(Q_HEADS * positions.len());
                    for q_head in 0..Q_HEADS {
                        let scores = match kv_mode {
                            KvMode::HqConsumed => capture.consumed_scores(layer, 0, q_head, positions),
                            KvMode::Bf16 => capture.scores(layer, 0, q_head, positions),
                        };
                        out.extend(scores);
                    }
                    out
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().expect("a scoring thread")).collect()
    })
}

/// The segment that holds the most softmax mass of one head, over the span:
/// R1's reading, printed as a live signal for the pointing head.
fn r1_winner(scores: &[f32], keys: &[Option<std::ops::Range<usize>>]) -> Option<usize> {
    let top = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mass = |range: &std::ops::Range<usize>| scores[range.clone()].iter().map(|&s| f64::from(s - top).exp()).sum::<f64>();
    keys.iter()
        .enumerate()
        .filter_map(|(index, range)| range.as_ref().map(|r| (index, mass(r))))
        .fold(None, |best: Option<(usize, f64)>, (index, m)| match best {
            Some((_, most)) if m <= most => best,
            _ => Some((index, m)),
        })
        .map(|(index, _)| index)
}

#[test]
#[ignore = "GPU profile only, with --features attn-tap"]
fn attention_over_a_text_state_is_dumped_for_calibration() {
    let kv_mode = KvMode::from_env();
    let kv_format = kv_mode.format();
    let chunk: u32 = std::env::var("IGNIS_LOCATE_CHUNK")
        .map(|v| v.parse().unwrap_or_else(|e| panic!("IGNIS_LOCATE_CHUNK={v}: {e}")))
        .unwrap_or(DEFAULT_PREFILL_CHUNK);
    let Ok(dir) = std::env::var("IGNIS_LOCATE_SET").map(PathBuf::from) else {
        if gpu_profile::skip_or_fail("IGNIS_LOCATE_SET names no question set") {
            return;
        }
        unreachable!("skip_or_fail panics under the profile");
    };
    let manifest_text = std::fs::read_to_string(dir.join("manifest.json"))
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.join("manifest.json").display()));
    let manifest: Manifest = serde_json::from_str(&manifest_text).unwrap_or_else(|e| panic!("parse the manifest: {e}"));
    let limit = std::env::var("IGNIS_LOCATE_LIMIT")
        .ok()
        .map(|v| v.parse::<usize>().unwrap_or_else(|e| panic!("IGNIS_LOCATE_LIMIT: {e}")));
    let questions: Vec<&Question> = manifest.questions.iter().take(limit.unwrap_or(usize::MAX)).collect();
    let set_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("set").to_owned();

    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let provider = ArtifactTemplateProvider::new(FrontendSet::from_reader(&reader).expect("frontend"));
    let tokenizer_set = FrontendSet::from_reader(&reader).expect("frontend");
    let tokenizer = tokenizer_set.tokenizer();
    // What the labelled baseline labels its segments with: the endpoint's own
    // answer alphabet, in the order it assigns them (`tools/locate-sets/labelled.py`).
    let alphabet: Vec<String> =
        AnswerAlphabet::from_tokenizer(tokenizer).tokens().iter().map(|t| t.label.clone()).collect();
    let thinking = ThinkingOptions { enable_thinking: false, ..ThinkingOptions::default() };

    let (plan, handles) = bind_model_scope_27b_with(&reader, ModelScope { draft: None, vision: false })
        .unwrap_or_else(|e| panic!("bind: {e}"));
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
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
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
        None,
        RopeScaling::NONE,
    )
    .unwrap_or_else(|e| panic!("ignis_model_load: {e}"));
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
    let ordinals: Vec<i32> = (0..GQA_LAYERS as i32).collect();

    // ── the dump's files, continued on resume ────────────────────────────
    let out_dir = std::env::var("IGNIS_LOCATE_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("ignis-attention-head-locate"));
    std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| panic!("create {}: {e}", out_dir.display()));
    let stem = match chunk {
        DEFAULT_PREFILL_CHUNK => format!("{set_name}-{}", kv_mode.name()),
        _ => format!("{set_name}-{}-chunk{chunk}", kv_mode.name()),
    };
    let (bin_path, rows_path) = (out_dir.join(format!("{stem}.bin")), out_dir.join(format!("{stem}.jsonl")));
    let resume = std::env::var("IGNIS_LOCATE_RESUME").is_ok_and(|v| v == "1");
    let mut done: Vec<serde_json::Value> = Vec::new();
    if resume && rows_path.exists() {
        let file = std::fs::File::open(&rows_path).expect("open the rows");
        for line in std::io::BufReader::new(file).lines() {
            done.push(serde_json::from_str(&line.expect("a row")).expect("a JSON row"));
        }
        // The rows say where the scores end; a longer bin holds a question
        // the crash cut short, which is dropped.
        let end = done.last().map_or(0, |row| row["end"].as_u64().expect("end") * 2);
        let bin = std::fs::OpenOptions::new().write(true).open(&bin_path).expect("open the bin");
        bin.set_len(end).expect("truncate the bin to its last whole question");
        eprintln!("resume: {} questions already dumped", done.len());
    } else {
        std::fs::write(&bin_path, []).expect("create the bin");
        std::fs::write(&rows_path, []).expect("create the rows");
    }
    let mut bin = std::fs::OpenOptions::new().append(true).open(&bin_path).expect("open the bin");
    let mut rows_file = std::fs::OpenOptions::new().append(true).open(&rows_path).expect("open the rows");
    let mut written: u64 = done.last().map_or(0, |row| row["end"].as_u64().expect("end"));

    eprintln!(
        "attention head locate: set {set_name} ({} questions), KV {}, chunk {chunk}",
        questions.len(),
        kv_mode.name()
    );
    let mut renders = serde_json::Map::new();
    let mut verify_failures: Vec<String> = Vec::new();
    let (mut l39_hits, mut l39_present) = ([0usize; 2], 0usize);
    let started = Instant::now();

    for question in questions.iter().skip(done.len()) {
        let ev = evidence(&question.state).unwrap_or_else(|e| panic!("{}: {e}", question.id));
        let owning = ev.segments.iter().filter(|s| s.owns).count();
        assert!(owning >= 2, "{}: {owning} segments own keys; a locate needs two", question.id);
        let mut segment_map: Option<(std::ops::Range<usize>, Vec<Option<std::ops::Range<usize>>>)> = None;
        let mut variants = serde_json::Map::new();
        let mut l39 = serde_json::Map::new();

        for (scaffold, content_free) in VARIANTS {
            let name = variant_name(scaffold, content_free);
            let instruction = if content_free { CONTENT_FREE } else { question.instruction.as_str() };
            let messages = [
                ChatMessage::text("system", ev.system.clone()),
                ChatMessage::text("user", user_text(scaffold, ev.unit, instruction)),
            ];
            // ── the render, from the endpoint's machinery ────────────────
            let rendered = provider
                .render_text(&messages, &thinking, &[])
                .unwrap_or_else(|e| panic!("{}: render: {e:?}", question.id));
            assert!(!rendered.contains("Reasoning effort"), "{}: a reasoning paragraph", question.id);
            assert!(
                rendered.ends_with(SERVED_TAIL),
                "{}: the render must end with a closed, empty think block; it ends with {:?}",
                question.id,
                &rendered[rendered.len().saturating_sub(60)..]
            );
            let served = provider
                .apply_chat_template(&messages, &thinking, &[])
                .unwrap_or_else(|e| panic!("{}: tokens: {e:?}", question.id))
                .tokens;
            let (ids, offsets) = tokenizer
                .encode_with_offsets(&rendered)
                .unwrap_or_else(|e| panic!("{}: encode: {e}", question.id));
            assert_eq!(ids, served, "{}: the prompt is not the provider's tokenization", question.id);
            let render_key = format!("{}-{}", scaffold.name(), ev.unit.name());
            renders.entry(render_key).or_insert_with(|| serde_json::Value::String(rendered.clone()));

            // ── the segment map, the same for all four ───────────────────
            assert_eq!(rendered.matches(&ev.system).count(), 1, "{}: the evidence appears once", question.id);
            let base = rendered.find(&ev.system).expect("the evidence is rendered");
            let shifted: Vec<Segment> = ev
                .segments
                .iter()
                .map(|s| Segment { bytes: s.bytes.start + base..s.bytes.end + base, owns: s.owns })
                .collect();
            let owned = owners(&offsets, &shifted);
            let span = key_span(&owned).expect("the evidence owns keys");
            let keys = segment_keys(&owned, &span, shifted.len());
            match &segment_map {
                None => segment_map = Some((span.clone(), keys.clone())),
                Some(first) => assert!(
                    first.0 == span && first.1 == keys,
                    "{}: {name} maps the state to other keys than the first variant",
                    question.id
                ),
            }

            // ── the scaffold, forced, and one armed prefill ──────────────
            let opening = tokenizer
                .encode(scaffold.opening(ev.unit))
                .unwrap_or_else(|e| panic!("encode the opening: {e}"));
            let mut tokens: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
            tokens.extend(opening.iter().map(|&t| t as i32));
            assert!(tokens.len() + 8 < MAX_CONTEXT as usize, "{}: {} tokens do not fit", question.id, tokens.len());
            let total = tokens.len();
            let query = total - 1;
            let cut = chunks(total as u32, chunk, ATTENTION_MIN_CHUNK_TOKENS);
            let (query_chunk_start, query_chunk_len) = *cut.last().expect("a chunk");
            let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
            let run = || -> Result<(), String> {
                for &(start, len) in &cut {
                    let (start, len) = (start as usize, len as usize);
                    step::prefill_program(&model, &pool, &mut sequence, &tokens[start..start + len], start as u64, None)?;
                }
                Ok(())
            };
            let prefill_started = Instant::now();
            let max_positions = total as i64 + 8;
            let (prefilled, capture) = match kv_mode {
                KvMode::HqConsumed => with_attn_tap_hq(&ordinals, &[query as i64], max_positions, run),
                KvMode::Bf16 => with_attn_tap(&ordinals, &[query as i64], max_positions, run),
            }
            .unwrap_or_else(|e| panic!("{}: attention tap: {e}", question.id));
            let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1e3;
            prefilled.unwrap_or_else(|e| panic!("{}: prefill: {e}", question.id));
            drop(sequence);
            assert_eq!(capture.queries_seen, vec![1], "{}: the query row was not captured", question.id);
            assert!(
                capture.rows_written.iter().all(|&r| r == total as i64),
                "{}: every armed layer must write one key row per position ({total}), wrote {:?}",
                question.id,
                capture.rows_written
            );

            // ── the consumed capture checks itself, every layer ──────────
            let mut hq = serde_json::Value::Null;
            if kv_mode == KvMode::HqConsumed {
                let ring = ring_before_chunk(&cut, query_chunk_start);
                let checked: Vec<(Vec<String>, serde_json::Value)> = std::thread::scope(|scope| {
                    let workers: Vec<_> = (0..ordinals.len())
                        .map(|layer| {
                            let (capture, ring) = (&capture, &ring);
                            scope.spawn(move || {
                                check_layer(capture, layer, layer % KV_HEADS, total, query_chunk_start as usize, ring)
                            })
                        })
                        .collect();
                    workers.into_iter().map(|w| w.join().expect("a check thread")).collect()
                });
                let mut layers = Vec::new();
                for (failures, stats) in checked {
                    verify_failures.extend(failures.into_iter().map(|f| format!("{} {name}: {f}", question.id)));
                    layers.push(stats);
                }
                hq = serde_json::json!({"query_chunk_start": query_chunk_start, "layers": layers});
            }

            // ── every head's scores over the span, into the dump ─────────
            let scores = span_scores(&capture, kv_mode, &span);
            drop(capture);
            let mut bytes = Vec::with_capacity(GQA_LAYERS * Q_HEADS * span.len() * 2);
            for layer in &scores {
                assert!(layer.iter().all(|s| s.is_finite()), "{}: a non-finite score", question.id);
                for &s in layer {
                    bytes.extend_from_slice(&f32_to_f16(s).to_le_bytes());
                }
            }
            bin.write_all(&bytes).expect("write the scores");
            let offset = written;
            written += (bytes.len() / 2) as u64;
            let pointing = &scores[POINTING_ORDINAL][POINTING_Q * span.len()..(POINTING_Q + 1) * span.len()];
            let winner = r1_winner(pointing, &segment_map.as_ref().expect("mapped").1);
            l39.insert(name.clone(), serde_json::json!(winner));
            variants.insert(
                name,
                serde_json::json!({
                    "offset": offset,
                    "prompt_tokens": total,
                    "last_chunk": [query_chunk_start, query_chunk_len],
                    "prefill_ms": prefill_ms,
                    "hq": hq,
                }),
            );
        }

        let (span, keys) = segment_map.expect("four variants mapped");
        let present = !question.absent;
        if present {
            l39_present += 1;
            for (slot, name) in ["s1", "s2"].iter().enumerate() {
                let winner = l39[*name].as_u64().map(|w| w as usize);
                l39_hits[slot] += usize::from(winner.is_some_and(|w| question.targets.contains(&w)));
            }
        }
        eprintln!(
            "  {} {:7} {:10} {:7} span {:>5}  L39.h10 s1 {} s2 {}  targets {:?}  [{:.0}s]",
            question.id,
            question.family,
            question.split,
            if question.absent { "absent" } else { "present" },
            span.len(),
            l39["s1"],
            l39["s2"],
            question.targets,
            started.elapsed().as_secs_f64()
        );
        let row = serde_json::json!({
            "id": question.id,
            "family": question.family,
            "kind": question.kind,
            "split": question.split,
            "absent": question.absent,
            "segments": ev.segments.len(),
            "owning": owning,
            "unit": ev.unit.name(),
            "targets": question.targets,
            "distractors": question.distractors,
            "span": [span.start, span.len()],
            "keys": keys.iter().map(|k| k.as_ref().map(|r| [r.start, r.end])).collect::<Vec<_>>(),
            "variants": variants,
            "l39_h10": l39,
            "end": written,
        });
        writeln!(rows_file, "{row}").expect("write a row");
        rows_file.flush().expect("flush the rows");
        bin.flush().expect("flush the scores");
    }

    // ── the rest of the dump, written before anything is asserted ────────
    let meta = serde_json::json!({
        "set": set_name,
        "seed": manifest.seed,
        "kv": kv_mode.name(),
        "kv_format": kv_format.as_str(),
        "prefill_chunk": chunk,
        "layers": ordinals.iter().map(|o| 4 * o + 3).collect::<Vec<_>>(),
        "q_heads": Q_HEADS,
        "dtype": "f16-le",
        "layout": "per question, per variant (s1, s1-na, s2, s2-na): [GQA layer][query head][span key]",
        "variants": VARIANTS.iter().map(|&(s, na)| variant_name(s, na)).collect::<Vec<_>>(),
        "scores": "q . k / 16 at the scaffold's last position over the state's key span, before softmax",
        "keys": match kv_mode {
            KvMode::Bf16 => "the BF16 keys attention reads",
            KvMode::HqConsumed => "the rotated-frame scratch rows the hq route consumed (query rotated to match)",
        },
        "rows_file": rows_path.file_name().and_then(|n| n.to_str()),
        "bin_file": bin_path.file_name().and_then(|n| n.to_str()),
        "alphabet": alphabet,
        "renders": renders,
        "verify_failures": verify_failures,
    });
    std::fs::write(out_dir.join(format!("{stem}.json")), serde_json::to_string_pretty(&meta).expect("serialize"))
        .unwrap_or_else(|e| panic!("write the dump's metadata: {e}"));
    eprintln!(
        "attention head locate: {set_name} {}: L39.h10 R1 s1 {}/{l39_present}, s2 {}/{l39_present} this run; \
         dump {} ({:.0}s)",
        kv_mode.name(),
        l39_hits[0],
        l39_hits[1],
        out_dir.join(&stem).display(),
        started.elapsed().as_secs_f64()
    );
    assert!(
        verify_failures.is_empty(),
        "the consumed-hq capture failed its self-check {} time(s):\n{}",
        verify_failures.len(),
        verify_failures.join("\n")
    );
}
