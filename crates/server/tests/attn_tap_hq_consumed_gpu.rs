//! The attention tap's hq capture reads the keys hq-e8-2b attention consumed
//! — and with the residual window wired (GitHub #257, spec runtime/06), which
//! of them came back exact is exactly what the vendored route's rule says.
//!
//! `ignis_attn_tap_arm_consumed` copies the hq prompt route's own scratch
//! plane after the attention op, found where the vendored op puts it — the
//! first allocation of the workspace `run_gqa_layer` hands it — and in the
//! codec's rotated frame. Both are facts about vendored code this repo does
//! not own, so this test holds them to account on a real prefill.
//!
//! **Before the window was wired**, every row — of 66,024 at 4096 px — sat at
//! the codec's own error (0.333 to 0.784 relative, median 0.3696, the per-row
//! median `docs/findings/2026-09-12-hq-attention-route-agreement.md`
//! measured) and not one near an exact row's ~0.004.
//!
//! **Now** each consumed row is classified by [`prompt_source`], the rule of
//! `gqa_attention_prefill_hq_scratch_kernel` replayed over the prompt's own
//! chunks, and held to it:
//!
//! - **fresh, sink, ring**: exact — within [`EXACT_ROW_BOUND`] of the
//!   rotation of the key the layer produced;
//! - **clobbered**: a key before the chunk whose ring slot the chunk's own
//!   append rewrote (the vendored prompt launch appends *before* it decodes
//!   the scratch) — exact, but to the *rewriting* key's row, not its own;
//! - **codec**: at the codec's error, and — on GQA ordinal 0, where nothing
//!   upstream reads the KV cache, so the key the codec is given is the same
//!   bit for bit as before — byte-identical to the capture of a build without
//!   the window (`IGNIS_TAP_BASELINE`). The codec is deterministic (its dither
//!   seed is `(kv_head, position, role)`), so a byte that moved there means the
//!   write path changed. On ordinal 9 (L39) the earlier GQA layers now attend
//!   over exact rows, so its keys — and their codes — legitimately move.
//!
//! The measured exact set must be the predicted one exactly: a row the rule
//! calls codec that comes back exact, or the reverse, fails.
//!
//! `IGNIS_TAP_DUMP=<dir>` writes each armed layer's pre-cache and consumed
//! rows under `<dir>/<size>/l<ordinal>/`, the files `IGNIS_TAP_BASELINE`
//! reads. `IGNIS_TAP_EXPECT_WINDOW=off` runs the pre-#257 expectation (every
//! row decoded) — only to record that baseline from a build without the
//! window.
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
use ignis_core::attn_tap::{AttnTapCapture, KV_HEADS, with_attn_tap_hq};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::hq_ring::{PromptSource, prompt_source, ring_after_prefill};
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
/// GQA ordinal 0 (backbone layer 3): only GDN layers before it, so its keys
/// do not depend on what attention reads. Ordinal 9 (L39): the head the
/// pointing findings are about.
const ORDINALS: [i32; 2] = [0, 9];
const Q_HEAD: usize = 10;
/// Below this, a row reproduces its rotated key: exact, not decoded. Exact
/// rows sit near 0.004 and the codec's rows at 0.33 and above.
const EXACT_ROW_BOUND: f32 = 0.1;
/// The codec's per-row error median sits near 0.37; a capture reading the
/// wrong rows or the wrong frame sits past ~1.3.
const CODEC_MEDIAN_BAND: (f32, f32) = (0.2, 0.6);
const ROW: usize = KV_HEADS * 256;

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    values[values.len() / 2]
}

/// `bytes` (an 8-bit RGB or RGBA PNG) box-filtered over `factor` x `factor`
/// blocks, as an RGB PNG: the committed 4096 px scene at a smaller size, so
/// one fixture measures both sizes spec runtime/06's acceptance names.
fn downscale_png(bytes: &[u8], factor: u32) -> Vec<u8> {
    let mut reader = png::Decoder::new(std::io::Cursor::new(bytes))
        .read_info()
        .unwrap_or_else(|e| panic!("decode the fixture: {e}"));
    let mut buf = vec![0u8; reader.output_buffer_size().expect("a bounded frame")];
    let info = reader.next_frame(&mut buf).unwrap_or_else(|e| panic!("decode the frame: {e}"));
    assert_eq!(info.bit_depth, png::BitDepth::Eight, "the fixture is 8-bit");
    let channels = match info.color_type {
        png::ColorType::Rgb => 3usize,
        png::ColorType::Rgba => 4usize,
        other => panic!("the fixture is {other:?}, expected RGB or RGBA"),
    };
    let (width, height) = (info.width / factor, info.height / factor);
    let mut out = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let mut sum = 0u32;
                for dy in 0..factor {
                    for dx in 0..factor {
                        let (sx, sy) = ((x * factor + dx) as usize, (y * factor + dy) as usize);
                        sum += u32::from(buf[sy * info.line_size + sx * channels + c]);
                    }
                }
                out[((y * width + x) * 3) as usize + c] = (sum / (factor * factor)) as u8;
            }
        }
    }
    let mut encoded = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut encoded, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap_or_else(|e| panic!("encode: {e}"));
        writer.write_image_data(&out).unwrap_or_else(|e| panic!("encode: {e}"));
    }
    encoded
}

/// The `(start, len)` chunks `prefill_prompt` runs `prompt` in: the same
/// capped loop, so the ring model replays the appends the device saw.
fn prefill_chunks(prompt: &Multimodal, total: u32, chunk: u32) -> Vec<(u64, u64)> {
    let mut chunks = Vec::new();
    let mut start = 0u32;
    while start < total {
        let len = prompt.cap_chunk(start, chunk.min(total - start));
        chunks.push((u64::from(start), u64::from(len)));
        start += len;
    }
    chunks
}

fn le_bytes(rows: &[u16]) -> Vec<u8> {
    rows.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn read_u16s(path: &Path) -> Option<Vec<u16>> {
    let bytes = std::fs::read(path).ok()?;
    Some(bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
}

/// One armed layer's rows, `[position][kv_head][256]`, for the prompt.
fn layer_rows(rows: &[u16], capture: &AttnTapCapture, layer: usize, tokens: usize) -> Vec<u16> {
    let at = layer * capture.max_positions as usize * ROW;
    rows[at..at + tokens * ROW].to_vec()
}

#[test]
#[ignore = "GPU profile only, with --features attn-tap"]
fn the_consumed_capture_reads_what_hq_attention_read() {
    let window_on = std::env::var("IGNIS_TAP_EXPECT_WINDOW").as_deref() != Ok("off");
    let image = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("pointing")
        .join("small.png");
    let Ok(fixture) = std::fs::read(&image) else {
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

    let mut failures: Vec<String> = Vec::new();
    for (size, bytes) in [("4096", fixture.clone()), ("1024", downscale_png(&fixture, 4))] {
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
            .unwrap_or_else(|e| panic!("{size} px: prepare: {e}"));
        let (token_ids, mut prompt) = Multimodal::from_prepared(prepared);
        let (begin, count, grid) = {
            let item = &prompt.media[0];
            (item.token_span.begin as usize, item.token_span.count as usize, item.grid)
        };
        let control = vision_item_control(grid);
        let embedding = step::encode_media(&model, grid, &prompt.media[0].patches, &control)
            .unwrap_or_else(|(_, e)| panic!("{size} px: encode: {e}"));
        let prefix = tokenizer.encode("{\"x\":").unwrap_or_else(|e| panic!("encode prefix: {e}"));
        assert!(prompt.append_text(prefix.len()), "the prompt cannot take the forced prefix");
        let mut tokens: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        tokens.extend(prefix.iter().map(|&t| t as i32));
        let n = tokens.len();
        let query = n - 1;

        let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
        let mut logits = vec![0f32; ModelConfig::qwen38_27b().vocab as usize];
        let (prefilled, capture) = with_attn_tap_hq(&ORDINALS, &[query as i64], n as i64 + 8, || {
            prefill_prompt(&model, &pool, &mut sequence, &tokens, &prompt, &embedding, PREFILL_CHUNK, &mut logits)
        })
        .unwrap_or_else(|e| panic!("{size} px: attention tap: {e}"));
        prefilled.unwrap_or_else(|e| panic!("{size} px: prefill: {e}"));

        assert_eq!(capture.queries_seen, vec![1], "{size} px: the query row was not captured");
        assert_eq!(
            capture.consumed_rows,
            vec![n as i64; ORDINALS.len()],
            "{size} px: the query chunk's attention consumed every prompt key, and the capture must \
             have them all"
        );

        // The ring the query chunk's scratch decode read: every chunk's
        // append up to and including the query chunk's own.
        let chunks = prefill_chunks(&prompt, n as u32, PREFILL_CHUNK);
        let chunk_start = capture.consumed_chunk_start[0] as u64;
        let upto = chunks.iter().position(|&(start, _)| start == chunk_start).unwrap_or_else(|| {
            panic!("{size} px: the capture's chunk start {chunk_start} is not a chunk of {chunks:?}")
        });
        let ring = ring_after_prefill(&chunks[..=upto]);
        let sources: Vec<PromptSource> = (0..n as u64)
            .map(|p| if window_on { prompt_source(p, chunk_start, &ring) } else { PromptSource::Codec })
            .collect();
        let tally = |f: fn(&PromptSource) -> bool, range: std::ops::Range<usize>| {
            sources[range].iter().filter(|s| f(s)).count()
        };
        let image_codec = tally(|s| *s == PromptSource::Codec, begin..begin + count);
        eprintln!(
            "{size} px served: {n} prompt tokens in {} chunks, query chunk from {chunk_start}; rule: \
             fresh {} sink {} ring {} clobbered {} codec {} ({:.1}% of the image through the codec)",
            chunks.len(),
            tally(|s| *s == PromptSource::Fresh, 0..n),
            tally(|s| *s == PromptSource::Sink, 0..n),
            tally(|s| *s == PromptSource::Ring, 0..n),
            tally(|s| matches!(s, PromptSource::Clobbered { .. }), 0..n),
            tally(|s| *s == PromptSource::Codec, 0..n),
            100.0 * image_codec as f64 / count as f64
        );

        for (layer, &ordinal) in ORDINALS.iter().enumerate() {
            let label = format!("{size} px, GQA ordinal {ordinal} (L{})", 4 * ordinal + 3);
            let (mut codec_errors, mut mismatched, mut clobber_worst, mut exact_worst) =
                (Vec::new(), Vec::new(), 0.0f32, 0.0f32);
            for position in 0..n {
                for kv_head in 0..KV_HEADS {
                    let own = capture.consumed_key_rel_err(layer, position, kv_head);
                    match sources[position] {
                        PromptSource::Fresh | PromptSource::Sink | PromptSource::Ring => {
                            exact_worst = exact_worst.max(own);
                            if own >= EXACT_ROW_BOUND {
                                mismatched.push(format!("{position}/{kv_head} {:?} {own:.4}", sources[position]));
                            }
                        }
                        PromptSource::Clobbered { by } => {
                            let to_by = capture.consumed_key_rel_err_to(layer, position, kv_head, by as usize);
                            clobber_worst = clobber_worst.max(to_by);
                            if to_by >= EXACT_ROW_BOUND || own < EXACT_ROW_BOUND {
                                mismatched.push(format!(
                                    "{position}/{kv_head} clobbered by {by}: {to_by:.4} to it, {own:.4} to its own"
                                ));
                            }
                        }
                        PromptSource::Codec => {
                            codec_errors.push(own);
                            if own < EXACT_ROW_BOUND {
                                mismatched.push(format!("{position}/{kv_head} codec but exact {own:.4}"));
                            }
                        }
                    }
                }
            }
            let codec_median = if codec_errors.is_empty() { 0.0 } else { median(&mut codec_errors) };
            eprintln!(
                "  {label}: exact rows worst {exact_worst:.4}, clobbered rows worst {clobber_worst:.4} to \
                 the rewriting key, codec rows median {codec_median:.4} ({} rows); {} rows off the rule",
                codec_errors.len(),
                mismatched.len()
            );
            if !mismatched.is_empty() {
                failures.push(format!(
                    "{label}: {} consumed rows are not what the rule says, first {:?}",
                    mismatched.len(),
                    &mismatched[..mismatched.len().min(8)]
                ));
            }
            if !(CODEC_MEDIAN_BAND.0..CODEC_MEDIAN_BAND.1).contains(&codec_median) {
                failures.push(format!(
                    "{label}: the codec rows sit a median {codec_median:.4} from their rotated keys, \
                     outside the codec's band {CODEC_MEDIAN_BAND:?}: the capture is not reading the \
                     scratch plane the layout says, or the rotation is not the codec's"
                ));
            }

            let keys = layer_rows(&capture.k, &capture, layer, n);
            let consumed = layer_rows(&capture.kc, &capture, layer, n);
            if let Ok(dir) = std::env::var("IGNIS_TAP_DUMP") {
                let dir = PathBuf::from(dir).join(size).join(format!("l{ordinal}"));
                std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create dump dir: {e}"));
                std::fs::write(dir.join("k.bin"), le_bytes(&keys)).expect("write k");
                std::fs::write(dir.join("kc.bin"), le_bytes(&consumed)).expect("write kc");
                std::fs::write(
                    dir.join("meta.json"),
                    format!("{{\"tokens\":{n},\"chunk_start\":{chunk_start},\"begin\":{begin},\"count\":{count}}}"),
                )
                .expect("write meta");
            }
            // The byte-identical check, where it is meaningful: the key the
            // codec was given must be the baseline's bit for bit first.
            if let Ok(dir) = std::env::var("IGNIS_TAP_BASELINE") {
                let dir = PathBuf::from(dir).join(size).join(format!("l{ordinal}"));
                let (Some(base_k), Some(base_kc)) = (read_u16s(&dir.join("k.bin")), read_u16s(&dir.join("kc.bin")))
                else {
                    panic!("{label}: no baseline capture under {}", dir.display());
                };
                assert_eq!(base_k.len(), keys.len(), "{label}: the baseline is another prompt");
                let same_key = |p: usize| keys[p * ROW..(p + 1) * ROW] == base_k[p * ROW..(p + 1) * ROW];
                let same_code = |p: usize| consumed[p * ROW..(p + 1) * ROW] == base_kc[p * ROW..(p + 1) * ROW];
                let codec_positions: Vec<usize> = (0..n).filter(|&p| sources[p] == PromptSource::Codec).collect();
                let keyed = codec_positions.iter().filter(|&&p| same_key(p)).count();
                let identical = codec_positions.iter().filter(|&&p| same_key(p) && same_code(p)).count();
                eprintln!(
                    "  {label}: against the pre-#257 build, {keyed} of {} codec positions have the same key \
                     bit for bit, and {identical} of those decode to the same bytes",
                    codec_positions.len()
                );
                if ordinal == 0 {
                    if keyed != codec_positions.len() {
                        failures.push(format!(
                            "{label}: {} codec positions' keys moved, but nothing before this layer reads \
                             the KV cache",
                            codec_positions.len() - keyed
                        ));
                    }
                    if identical != keyed {
                        failures.push(format!(
                            "{label}: {} codec rows decode to other bytes from the same key: the write path \
                             changed",
                            keyed - identical
                        ));
                    }
                }
            }
        }

        // And the head is scoreable on what attention consumed.
        let positions: Vec<usize> = (begin..begin + count).collect();
        let scores = capture.consumed_scores(1, 0, Q_HEAD, &positions);
        assert!(scores.iter().all(|s| s.is_finite()), "{size} px: non-finite consumed score");
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
