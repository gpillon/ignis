//! Where in a text does the attention point, to the token? The instrument
//! of spec 19 phase 1 (`docs/specs/decide/19-a-span-read-from-attention.md`,
//! GitHub #276), over the span sets `tools/locate-sets/spans.py` writes.
//!
//! Spec 18's harness (`attention_head_locate_gpu.rs`) dumped every head at
//! one position over the state's keys, for two scaffolds. Phase 0 read those
//! dumps and kept the copy scaffold and its content-free prefill; this dumps
//! what phase 0 could not see. Per question, up to three prefills of the L1
//! render, with the span kind text (`support/locate.rs::SPAN_KIND`) and the
//! copy scaffold `{"quote":"` forced:
//!
//! - `q`: the instruction. At the scaffold's last token, every head's scores
//!   over the **full row** — every key of the prompt, template, kind text
//!   and instruction included, each token labelled with its region — and,
//!   from up to [`INSTRUCTION_QUERIES`] of the instruction's own tokens
//!   (ICR's direction), every head's softmax weights averaged over those
//!   queries, over the keys up to the instruction's end;
//! - `q-na`: the instruction replaced by `N/A`, the full row at the
//!   scaffold's last token (the content-free baseline);
//! - `q-forced` (present questions): the first gold's text forced after the
//!   scaffold, teacher-style, and every head's scores over the state's key
//!   span at the scaffold's last token and at the quote's first, middle and
//!   last tokens — where a copy head goes as the quote is written, and
//!   where it looks when the quote is about to close.
//!
//! The row of each question carries the key span, each segment's keys, the
//! byte range of every span key relative to the evidence (so a scorer maps a
//! gold character span onto keys without tokenizing), and the regions.
//!
//! **The keys are the ones attention read** (hq-e8-2b consumed, the default;
//! `IGNIS_LOCATE_KV=bf16` for a BF16 pool), checked on every armed layer as
//! spec 18's harness checks them (`support/locate_tap.rs`). Every query sits
//! in the prompt's last chunk — cut to hold the whole user turn and scaffold
//! — because the consumed capture covers the first query's chunk only.
//!
//! The dump is `<set>-<kv>.bin`, f16 little-endian, per question in the
//! order: `q` scores `[16][24][T]`, `q` instruction weights `[16][24][I]`,
//! `q-na` scores `[16][24][T_na]`, then `q-forced` scores
//! `[queries][16][24][span]`; offsets and shapes in `<set>-<kv>.jsonl`, the
//! rest in `<set>-<kv>.json`.
//!
//! `IGNIS_LOCATE_SET` (required), `IGNIS_LOCATE_OUT`, `IGNIS_LOCATE_LIMIT`,
//! `IGNIS_LOCATE_RESUME=1`, `IGNIS_LOCATE_KV`, `IGNIS_LOCATE_CHUNK`, as the
//! spec 18 harness reads them.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38), and the `attn-tap` feature.

#![cfg(all(feature = "cuda", feature = "attn-tap"))]

#[path = "support/locate.rs"]
mod locate;
#[path = "support/locate_tap.rs"]
mod locate_tap;

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use ignis_artifact::{CudaDevice, FrontendSet, ModelScope, Reader, bind_model_scope_27b_with, materialize};
use ignis_core::attn_tap::{AttnTapCapture, GQA_LAYERS, KV_HEADS, Q_HEADS, with_attn_tap, with_attn_tap_hq};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::hq_ring::ring_before_chunk;
use ignis_core::model_load::load_qwen38_27b_with_options;
use ignis_core::pointing::ATTENTION_MIN_CHUNK_TOKENS;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step;
use ignis_core::RopeScaling;
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::decide::OrderedValue;
use ignis_server::template::{ChatMessage, TemplateProvider};
use ignis_server::thinking::ThinkingOptions;
use serde::Deserialize;

use locate::{CONTENT_FREE, Region, SPAN_KIND, chunks, evidence, key_bytes, map_segments, regions, span_user_text};
use locate_tap::{KvMode, check_layer, f16_bytes, f32_to_f16, head_scores};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// The longest set state, a 1,000-line log, renders about 18K tokens.
const MAX_CONTEXT: u32 = 20_480;
const DEFAULT_PREFILL_CHUNK: u32 = 1024;
const SERVED_TAIL: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
const OPENING: &str = "{\"quote\":\"";
/// The instruction's tokens read as queries, evenly spaced over it.
const INSTRUCTION_QUERIES: usize = 8;

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
    split: String,
    absent: bool,
    state: OrderedValue,
    instruction: String,
    targets: Vec<usize>,
    #[serde(default)]
    spans: Vec<serde_json::Value>,
    #[serde(default)]
    quote: Option<String>,
}

/// One armed prefill: the capture, and where its last chunk started.
struct Prefilled {
    capture: AttnTapCapture,
    last_chunk: (u64, u64),
    prefill_ms: f64,
    hq: serde_json::Value,
    failures: Vec<String>,
}

/// `positions` evenly spaced over `range`, at most `n`, the last included.
fn spaced(range: &std::ops::Range<usize>, n: usize) -> Vec<usize> {
    let len = range.len();
    if len <= n {
        return range.clone().collect();
    }
    let mut out: Vec<usize> = (0..n).map(|i| range.start + (i * (len - 1)) / (n - 1)).collect();
    out.dedup();
    out
}

/// Softmax weights of `[layer][q_head][key]` scores, averaged over the
/// queries of `rows` (each `[layer][q_head * keys]`, keys past a query's own
/// position already cut to -inf by the caller), as `[layer][q_head][key]`.
fn mean_weights(rows: &[Vec<Vec<f32>>], keys: usize) -> Vec<Vec<f32>> {
    let layers = rows[0].len();
    let mut out = vec![vec![0.0f32; Q_HEADS * keys]; layers];
    for row in rows {
        for (layer, scores) in row.iter().enumerate() {
            for head in 0..Q_HEADS {
                let s = &scores[head * keys..(head + 1) * keys];
                let top = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f64> = s.iter().map(|&x| f64::from(x - top).exp()).collect();
                let total: f64 = e.iter().sum();
                for (k, v) in e.iter().enumerate() {
                    out[layer][head * keys + k] += (v / total / rows.len() as f64) as f32;
                }
            }
        }
    }
    out
}

/// Append f16 bytes to the dump; their offset, in f16 elements.
fn append(bin: &mut std::fs::File, written: &mut u64, bytes: &[u8]) -> u64 {
    bin.write_all(bytes).expect("write the scores");
    let offset = *written;
    *written += (bytes.len() / 2) as u64;
    offset
}

#[test]
#[ignore = "GPU profile only, with --features attn-tap"]
fn attention_over_a_span_set_is_dumped_with_its_full_row() {
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
        .unwrap_or_else(|_| std::env::temp_dir().join("ignis-attention-span-locate"));
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
        let end = done.last().map_or(0, |row| row["end"].as_u64().expect("end") * 2);
        let bin = std::fs::OpenOptions::new().write(true).open(&bin_path).expect("open the bin");
        bin.set_len(end).expect("truncate the bin to its last whole question");
        for (row, question) in done.iter().zip(&questions) {
            assert_eq!(row["id"], question.id.as_str(), "resume: the dump is not this set's, in this order");
        }
        eprintln!("resume: {} questions already dumped", done.len());
    } else {
        std::fs::write(&bin_path, []).expect("create the bin");
        std::fs::write(&rows_path, []).expect("create the rows");
    }
    let mut bin = std::fs::OpenOptions::new().append(true).open(&bin_path).expect("open the bin");
    let mut rows_file = std::fs::OpenOptions::new().append(true).open(&rows_path).expect("open the rows");
    let mut written: u64 = done.last().map_or(0, |row| row["end"].as_u64().expect("end"));

    eprintln!("attention span locate: set {set_name} ({} questions), KV {}, chunk {chunk}", questions.len(), kv_mode.name());
    let render = |id: &str, ev: &locate::Evidence, user: &str| {
        let messages = [ChatMessage::text("system", ev.system.clone()), ChatMessage::text("user", user.to_owned())];
        let rendered = provider
            .render_text(&messages, &thinking, &[])
            .unwrap_or_else(|e| panic!("{id}: render: {e:?}"));
        assert!(!rendered.contains("Reasoning effort"), "{id}: a reasoning paragraph");
        assert!(rendered.ends_with(SERVED_TAIL), "{id}: the render must end with a closed, empty think block");
        let served = provider
            .apply_chat_template(&messages, &thinking, &[])
            .unwrap_or_else(|e| panic!("{id}: tokens: {e:?}"))
            .tokens;
        let (ids, offsets) = tokenizer.encode_with_offsets(&rendered).unwrap_or_else(|e| panic!("{id}: encode: {e}"));
        assert_eq!(ids, served, "{id}: the prompt is not the provider's tokenization");
        (rendered, ids, offsets)
    };
    let opening: Vec<i32> =
        tokenizer.encode(OPENING).unwrap_or_else(|e| panic!("encode the opening: {e}")).iter().map(|&t| t as i32).collect();

    // One armed prefill of `tokens`, its last chunk holding `tail` tokens or
    // more, the queries at `queries` (the first one decides which chunk the
    // consumed capture copies).
    let prefill = |id: &str, name: &str, tokens: &[i32], tail: usize, queries: &[usize]| -> Prefilled {
        let total = tokens.len();
        assert!(total + 8 < MAX_CONTEXT as usize, "{id}: {total} tokens do not fit");
        let tail = tail.max(ATTENTION_MIN_CHUNK_TOKENS as usize) as u32;
        let cut = chunks(total as u32, chunk, tail);
        let (query_chunk_start, query_chunk_len) = *cut.last().expect("a chunk");
        assert!(
            queries.iter().all(|&q| q as u64 >= query_chunk_start),
            "{id} {name}: a query before the last chunk ({query_chunk_start})"
        );
        let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
        let run = || -> Result<(), String> {
            for &(start, len) in &cut {
                let (start, len) = (start as usize, len as usize);
                step::prefill_program(&model, &pool, &mut sequence, &tokens[start..start + len], start as u64, None)?;
            }
            Ok(())
        };
        let started = Instant::now();
        let positions: Vec<i64> = queries.iter().map(|&q| q as i64).collect();
        let max_positions = total as i64 + 8;
        let (prefilled, capture) = match kv_mode {
            KvMode::HqConsumed => with_attn_tap_hq(&ordinals, &positions, max_positions, run),
            KvMode::Bf16 => with_attn_tap(&ordinals, &positions, max_positions, run),
        }
        .unwrap_or_else(|e| panic!("{id} {name}: attention tap: {e}"));
        let prefill_ms = started.elapsed().as_secs_f64() * 1e3;
        prefilled.unwrap_or_else(|e| panic!("{id} {name}: prefill: {e}"));
        drop(sequence);
        assert!(capture.queries_seen.iter().all(|&s| s == 1), "{id} {name}: a query row was not captured");
        assert!(capture.rows_written.iter().all(|&r| r == total as i64), "{id} {name}: missing key rows");
        let mut failures = Vec::new();
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
            for (f, stats) in checked {
                failures.extend(f.into_iter().map(|f| format!("{id} {name}: {f}")));
                layers.push(stats);
            }
            hq = serde_json::json!({"query_chunk_start": query_chunk_start, "layers": layers});
        }
        Prefilled {
            capture,
            last_chunk: (query_chunk_start, query_chunk_len),
            prefill_ms,
            hq,
            failures,
        }
    };

    let mut verify_failures: Vec<String> = done
        .iter()
        .flat_map(|row| row["verify_failures"].as_array().cloned().unwrap_or_default())
        .map(|f| f.as_str().expect("a failure").to_owned())
        .collect();
    let mut renders = serde_json::Map::new();
    let started = Instant::now();

    for question in questions.iter().skip(done.len()) {
        let id = question.id.as_str();
        let ev = evidence(&question.state).unwrap_or_else(|e| panic!("{id}: {e}"));
        let mut failures: Vec<String> = Vec::new();

        // ── q: the instruction, full row and the instruction's own queries ─
        let user = span_user_text(&question.instruction);
        let (rendered, ids, offsets) = render(id, &ev, &user);
        renders.entry(ev.unit.name()).or_insert_with(|| serde_json::Value::String(rendered.clone()));
        let (span, keys) = map_segments(&rendered, &offsets, &ev).unwrap_or_else(|e| panic!("{id}: {e}"));
        let runs = regions(&rendered, &offsets, &ev, SPAN_KIND, &user).unwrap_or_else(|e| panic!("{id}: {e}"));
        let bytes_of_keys = key_bytes(&rendered, &offsets, &ev, &span).unwrap_or_else(|e| panic!("{id}: {e}"));
        let instruction = runs
            .iter()
            .find(|(_, r)| *r == Region::Instruction)
            .map(|(range, _)| range.clone())
            .unwrap_or_else(|| panic!("{id}: no instruction token"));
        let mut tokens: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
        tokens.extend(&opening);
        let total = tokens.len();
        let last = total - 1;
        let instruction_queries = spaced(&instruction, INSTRUCTION_QUERIES);
        let mut queries = vec![last];
        queries.extend(&instruction_queries);
        let q = prefill(id, "q", &tokens, total - instruction.start, &queries);
        failures.extend(q.failures.iter().cloned());
        let full: Vec<usize> = (0..total).collect();
        let q_scores = head_scores(&q.capture, kv_mode, 0, &full);
        let q_offset = append(&mut bin, &mut written, &f16_bytes(&q_scores, id));
        // Each instruction query sees the keys up to itself; the weights are
        // over the keys up to the instruction's last token.
        let upto = instruction.end;
        let rows: Vec<Vec<Vec<f32>>> = instruction_queries
            .iter()
            .enumerate()
            .map(|(i, &at)| {
                let mut row = head_scores(&q.capture, kv_mode, i + 1, &full[..upto]);
                for layer in &mut row {
                    for head in 0..Q_HEADS {
                        for s in &mut layer[head * upto + at + 1..(head + 1) * upto] {
                            *s = f32::NEG_INFINITY;
                        }
                    }
                }
                row
            })
            .collect();
        let weights = mean_weights(&rows, upto);
        let mut weight_bytes = Vec::with_capacity(GQA_LAYERS * Q_HEADS * upto * 2);
        for layer in &weights {
            for &w in layer {
                weight_bytes.extend_from_slice(&f32_to_f16(w).to_le_bytes());
            }
        }
        let w_offset = append(&mut bin, &mut written, &weight_bytes);
        let q_row = serde_json::json!({
            "scores_offset": q_offset, "keys": total, "weights_offset": w_offset, "weight_keys": upto,
            "instruction_queries": instruction_queries, "prompt_tokens": total,
            "last_chunk": [q.last_chunk.0, q.last_chunk.1], "prefill_ms": q.prefill_ms, "hq": q.hq,
        });
        drop(q);

        // ── q-na: the content-free baseline, full row ─────────────────────
        let user_na = span_user_text(CONTENT_FREE);
        let (rendered_na, ids_na, offsets_na) = render(id, &ev, &user_na);
        let (span_na, keys_na) = map_segments(&rendered_na, &offsets_na, &ev).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert!(span_na == span && keys_na == keys, "{id}: q-na maps the state to other keys than q");
        let mut tokens_na: Vec<i32> = ids_na.iter().map(|&t| t as i32).collect();
        tokens_na.extend(&opening);
        let total_na = tokens_na.len();
        let na = prefill(id, "q-na", &tokens_na, ATTENTION_MIN_CHUNK_TOKENS as usize, &[total_na - 1]);
        failures.extend(na.failures.iter().cloned());
        let na_full: Vec<usize> = (0..total_na).collect();
        let na_offset = append(&mut bin, &mut written, &f16_bytes(&head_scores(&na.capture, kv_mode, 0, &na_full), id));
        let runs_na = regions(&rendered_na, &offsets_na, &ev, SPAN_KIND, &user_na).unwrap_or_else(|e| panic!("{id}: {e}"));
        let na_row = serde_json::json!({
            "scores_offset": na_offset, "keys": total_na, "prompt_tokens": total_na,
            "regions": runs_na.iter().map(|(r, g)| serde_json::json!([r.start, r.end, g.name()])).collect::<Vec<_>>(),
            "last_chunk": [na.last_chunk.0, na.last_chunk.1], "prefill_ms": na.prefill_ms, "hq": na.hq,
        });
        drop(na);

        // ── q-forced: the first gold written after the scaffold ───────────
        let mut forced_row = serde_json::Value::Null;
        if let (false, Some(quote)) = (question.absent, question.quote.as_deref()) {
            let escaped = serde_json::to_string(quote).expect("a string serializes");
            let quote_tokens: Vec<i32> = tokenizer
                .encode(&escaped[1..escaped.len() - 1])
                .unwrap_or_else(|e| panic!("{id}: encode the quote: {e}"))
                .iter()
                .map(|&t| t as i32)
                .collect();
            let mut forced = tokens.clone();
            forced.extend(&quote_tokens);
            let n = quote_tokens.len();
            let mut fq = vec![last, last + 1, last + n.div_ceil(2), last + n];
            fq.dedup();
            let f = prefill(id, "q-forced", &forced, forced.len() - last, &fq);
            failures.extend(f.failures.iter().cloned());
            let span_positions: Vec<usize> = span.clone().collect();
            let mut offsets_f = Vec::new();
            for (i, _) in fq.iter().enumerate() {
                let scores = head_scores(&f.capture, kv_mode, i, &span_positions);
                offsets_f.push(append(&mut bin, &mut written, &f16_bytes(&scores, id)));
            }
            forced_row = serde_json::json!({
                "queries": fq, "scores_offsets": offsets_f, "keys": span.len(), "quote_tokens": n,
                "prompt_tokens": forced.len(), "last_chunk": [f.last_chunk.0, f.last_chunk.1],
                "prefill_ms": f.prefill_ms, "hq": f.hq,
            });
            drop(f);
        }

        eprintln!(
            "  {id:14} {:9} {:10} {:7} keys {:>5} span {:>5}  [{:.0}s]",
            question.family,
            question.split,
            if question.absent { "absent" } else { "present" },
            total,
            span.len(),
            started.elapsed().as_secs_f64()
        );
        let row = serde_json::json!({
            "id": id,
            "family": question.family,
            "split": question.split,
            "absent": question.absent,
            "unit": ev.unit.name(),
            "segments": ev.segments.len(),
            "targets": question.targets,
            "spans": question.spans,
            "quote": question.quote,
            "span": [span.start, span.len()],
            "keys": keys.iter().map(|k| k.as_ref().map(|r| [r.start, r.end])).collect::<Vec<_>>(),
            "key_bytes": bytes_of_keys,
            "regions": runs.iter().map(|(r, g)| serde_json::json!([r.start, r.end, g.name()])).collect::<Vec<_>>(),
            "scaffold": [total - opening.len(), total],
            "q": q_row,
            "q-na": na_row,
            "q-forced": forced_row,
            "verify_failures": failures,
            "end": written,
        });
        writeln!(rows_file, "{row}").expect("write a row");
        rows_file.flush().expect("flush the rows");
        verify_failures.extend(failures);
    }
    bin.flush().expect("flush the scores");

    let meta = serde_json::json!({
        "set": set_name,
        "seed": manifest.seed,
        "kv": kv_mode.name(),
        "kv_format": kv_format.as_str(),
        "prefill_chunk": chunk,
        "layers": ordinals.iter().map(|o| 4 * o + 3).collect::<Vec<_>>(),
        "q_heads": Q_HEADS,
        "dtype": "f16-le",
        "layout": "per question: q scores [16][24][T], q instruction weights [16][24][I], q-na scores [16][24][T_na], \
                   q-forced scores [queries][16][24][span]",
        "scores": "q . k / 16 before softmax; the instruction weights are softmax weights averaged over its queries",
        "span_kind": SPAN_KIND,
        "opening": OPENING,
        "rows_file": rows_path.file_name().and_then(|n| n.to_str()),
        "bin_file": bin_path.file_name().and_then(|n| n.to_str()),
        "renders": renders,
        "verify_failures": verify_failures,
    });
    std::fs::write(out_dir.join(format!("{stem}.json")), serde_json::to_string_pretty(&meta).expect("serialize"))
        .unwrap_or_else(|e| panic!("write the dump's metadata: {e}"));
    eprintln!("attention span locate: {set_name} dumped to {} ({:.0}s)", out_dir.join(&stem).display(), started.elapsed().as_secs_f64());
    assert!(
        verify_failures.is_empty(),
        "the consumed-hq capture failed its self-check {} time(s):\n{}",
        verify_failures.len(),
        verify_failures.join("\n")
    );
}
