//! PROTOTYPE #409 (throwaway, branch `prototype/409-vision-host`): the
//! capture harness for the vision residency prototype. It loads the 27B with
//! vision under whatever `IGNIS_VISION_RESIDENCY` / `IGNIS_VISION_PROTO_*` the
//! process carries, and writes, under `PROTO409_OUT/PROTO409_LABEL/`:
//!
//! - every media embedding it encodes, as raw BF16 bits (`emb_<name>.bin`) and
//!   an FNV-1a hash in `summary.txt`;
//! - for the vision canary (`PROTO409_CANARY=1`), each prompt's teacher-forced
//!   predictions and a hash of every scored position's logits, whole and in
//!   48-token spans, so two modes compare bit for bit through the prefill's
//!   scatter as well as the encoder;
//! - for `PROTO409_IMAGE`, timed encodes (`PROTO409_ENCODE_RUNS`, after
//!   `PROTO409_WARMUP` warm-ups) and timed prefills of that image's prompt
//!   (`PROTO409_PREFILL_RUNS`).
//!
//! `proto_409_compare` (no GPU) then compares two labels' outputs. Every
//! error panics: this is a measurement, not a profile test that may skip.

#![cfg(feature = "cuda")]

#[path = "support/mod.rs"]
mod support;

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ignis_artifact::{
    bind_model_scope_27b_with, materialize, ChatMessage, ChatRenderOptions, ContentPart, CudaDevice,
    FrontendSet, MessageContent, ModelScope, Reader, Role,
};
use ignis_core::compute::ModelConfig;
use ignis_core::model_load::load_qwen38_27b_with_options;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step;
use ignis_core::vision::{vision_item_control, Multimodal};
use ignis_core::{KvFormat, RopeScaling, Vision};

use support::vision_canary::{argmax_lowest_id, load_canaries, prefill_prompt};

const DEFAULT_ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const FIRST_N: usize = 32;
const SPANNING_CHUNK: u32 = 48;

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn num(name: &str, default: u32) -> u32 {
    var(name).map_or(default, |v| v.parse().unwrap_or_else(|e| panic!("{name}={v}: {e}")))
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn bits_of(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect()
}

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

struct Summary {
    file: std::fs::File,
}

impl Summary {
    fn note(&mut self, line: &str) {
        eprintln!("proto409: {line}");
        writeln!(self.file, "{line}").expect("write summary");
        self.file.flush().expect("flush summary");
    }
}

fn image_prompt(frontend: &FrontendSet, image: &[u8], question: &str) -> (Vec<i32>, Multimodal) {
    let processor = frontend.vision_processor().expect("vision processor");
    let messages = [ChatMessage {
        role: Role::User,
        content: MessageContent::Parts(vec![ContentPart::Image { url: None }, ContentPart::Text(question.into())]),
        tool_calls: Vec::new(),
        reasoning_content: None,
    }];
    let prepared = frontend
        .prepare_prompt(&processor, &messages, &[image], ChatRenderOptions { enable_thinking: false, ..Default::default() }, None)
        .expect("prepare prompt");
    let (token_ids, prompt) = Multimodal::from_prepared(prepared);
    (token_ids.iter().map(|&t| t as i32).collect(), prompt)
}

#[test]
#[ignore = "PROTOTYPE #409: run by hand with PROTO409_* set"]
fn proto_409_capture() {
    let out = PathBuf::from(var("PROTO409_OUT").expect("PROTO409_OUT"));
    let label = var("PROTO409_LABEL").expect("PROTO409_LABEL");
    let dir = out.join(&label);
    std::fs::create_dir_all(&dir).expect("create the label's dir");
    let mut summary = Summary { file: std::fs::File::create(dir.join("summary.txt")).expect("summary.txt") };

    let artifact_path = var("PROTO409_ARTIFACT").unwrap_or_else(|| DEFAULT_ARTIFACT.to_string());
    let chunk = num("PROTO409_CHUNK", 1024);
    let context = num("PROTO409_CONTEXT", 4096);
    let envelope = num("PROTO409_ENVELOPE", 1024);
    let kv = match var("PROTO409_KV").as_deref() {
        None | Some("bf16") => KvFormat::Bf16,
        Some("hq") => KvFormat::HqE8_2b,
        Some(other) => panic!("PROTO409_KV={other}"),
    };
    summary.note(&format!(
        "label={label} IGNIS_VISION_RESIDENCY={:?} IGNIS_VISION_PROTO_STAGING={:?} IGNIS_VISION_PROTO_POOL={:?} \
         artifact={artifact_path} chunk={chunk} context={context} envelope={envelope} kv={kv:?}",
        var("IGNIS_VISION_RESIDENCY"),
        var("IGNIS_VISION_PROTO_STAGING"),
        var("IGNIS_VISION_PROTO_POOL")
    ));

    let reader = Reader::open(Path::new(&artifact_path)).expect("open the artifact");
    let frontend = FrontendSet::from_reader(&reader).expect("frontend");
    let (plan, handles) = bind_model_scope_27b_with(&reader, ModelScope { draft: None, vision: true, fp8_exceptions: false })
        .expect("bind with vision");
    summary.note(&format!(
        "plan device_capacity_bytes={} host_mapped_capacity_bytes={} device_objects={} host_mapped_objects={}",
        plan.device_capacity_bytes,
        plan.host_mapped_capacity_bytes,
        plan.device_objects.len(),
        plan.host_mapped_objects.len()
    ));
    let mut device = CudaDevice::create(0).expect("CUDA device");
    let started = Instant::now();
    let mut artifact = materialize(&reader, &plan, &mut device, None).expect("materialize");
    summary.note(&format!("materialize_seconds={:.1}", started.elapsed().as_secs_f64()));
    let vision = Vision::new(envelope).expect("envelope");
    let model = load_qwen38_27b_with_options(&reader, &artifact, &handles, chunk, context, kv, None, Some(vision), RopeScaling::NONE)
        .expect("ignis_model_load with vision");
    let stats = model.stats();
    summary.note(&format!(
        "model vram_bytes={} host_mapped_bytes={} vision_reserved_bytes={} workspace_bytes={} media_embedding_bytes={}",
        stats.vram_bytes,
        stats.host_mapped_bytes,
        stats.vision_reserved_bytes,
        stats.reserved.workspace_bytes,
        stats.reserved.media_embedding_bytes
    ));
    let config = ModelConfig::qwen38_27b();
    let hidden = config.hidden as usize;
    let vocab = config.vocab as usize;
    let pool = SeqPool::create(
        &config,
        &SeqPoolBudget {
            kv_format: kv,
            kv_page_group_count: context / 64 + 8,
            max_context_tokens: context,
            slot_count: 1,
            retained_slot_count: 0,
            retained_host_slot_count: 0,
        },
    )
    .expect("seq pool create");
    let program = step::program_stats(&model, &pool).expect("program stats");
    summary.note(&format!("program vram_bytes={}", program.vram_bytes));

    if var("PROTO409_CANARY").as_deref() == Some("1") {
        for canary in load_canaries(&frontend) {
            let (tokens, prompt) = image_prompt(&frontend, &canary.image, &canary.question);
            let item = &prompt.media[0];
            let embedding = step::encode_media(&model, item.grid, &item.patches, &vision_item_control(item.grid))
                .unwrap_or_else(|(rc, e)| panic!("{}: encode {rc}: {e}", canary.id));
            let bits = u16_bytes(&embedding.proto_read(hidden).expect("read the embedding"));
            std::fs::write(dir.join(format!("emb_canary_{}.bin", canary.id)), &bits).expect("write embedding");
            summary.note(&format!(
                "canary {} columns={} embedding_fnv={:016x}",
                canary.id,
                embedding.columns(),
                fnv64(&bits)
            ));
            for (mode, span) in [("whole", chunk), ("spanning", SPANNING_CHUNK)] {
                let mut sequence = pool.alloc(context).expect("seq alloc");
                let mut logits = vec![0f32; vocab];
                prefill_prompt(&model, &pool, &mut sequence, &tokens, &prompt, &embedding, span, &mut logits)
                    .unwrap_or_else(|e| panic!("{} {mode}: prefill: {e}", canary.id));
                let mut position = tokens.len() as u64;
                let mut predictions = Vec::new();
                let mut hashes = String::new();
                for i in 0..FIRST_N.min(canary.expected.len()) {
                    predictions.push(argmax_lowest_id(&logits));
                    let _ = write!(hashes, "{:016x},", fnv64(&bits_of(&logits)));
                    let forced = [canary.expected[i] as i32];
                    step::prefill_program(&model, &pool, &mut sequence, &forced, position, Some(&mut logits))
                        .unwrap_or_else(|e| panic!("{} {mode}: forced {i}: {e}", canary.id));
                    position += 1;
                }
                let agree = predictions.iter().zip(&canary.expected).filter(|(a, b)| a == b).count();
                summary.note(&format!(
                    "canary {} {mode} agree={agree}/{} predictions={predictions:?} logits_fnv={hashes}",
                    canary.id,
                    predictions.len()
                ));
            }
        }
    }

    if let Some(image_path) = var("PROTO409_IMAGE") {
        let image = std::fs::read(&image_path).expect("read PROTO409_IMAGE");
        let name = Path::new(&image_path).file_stem().unwrap().to_string_lossy().into_owned();
        let (tokens, prompt) = image_prompt(&frontend, &image, "Describe this image in one sentence.");
        let item = &prompt.media[0];
        let control = vision_item_control(item.grid);
        summary.note(&format!(
            "image {name}: grid {}x{}x{} -> {} merged tokens, {} prompt tokens",
            item.grid.t,
            item.grid.h,
            item.grid.w,
            item.token_span.count,
            tokens.len()
        ));
        let encode = || {
            step::encode_media(&model, item.grid, &item.patches, &control)
                .unwrap_or_else(|(rc, e)| panic!("{name}: encode {rc}: {e}"))
        };
        let warmup = num("PROTO409_WARMUP", 3);
        let runs = num("PROTO409_ENCODE_RUNS", 0);
        for _ in 0..warmup {
            drop(encode());
        }
        let mut encode_ms = Vec::new();
        for _ in 0..runs {
            let started = Instant::now();
            let embedding = encode();
            encode_ms.push(started.elapsed().as_secs_f64() * 1e3);
            drop(embedding);
        }
        if !encode_ms.is_empty() {
            let raw = encode_ms.iter().map(|ms| format!("{ms:.3}")).collect::<Vec<_>>().join(",");
            let mut sorted = encode_ms.clone();
            sorted.sort_by(f64::total_cmp);
            summary.note(&format!(
                "encode {name}: runs={} median_ms={:.3} p10_ms={:.3} p90_ms={:.3} min_ms={:.3} raw_ms={raw}",
                sorted.len(),
                percentile(&sorted, 0.5),
                percentile(&sorted, 0.1),
                percentile(&sorted, 0.9),
                sorted[0]
            ));
        }
        let embedding = encode();
        let bits = u16_bytes(&embedding.proto_read(hidden).expect("read the embedding"));
        if var("PROTO409_WRITE_EMBEDDING").as_deref() != Some("0") {
            std::fs::write(dir.join(format!("emb_{name}.bin")), &bits).expect("write embedding");
        }
        summary.note(&format!("image {name} columns={} embedding_fnv={:016x}", embedding.columns(), fnv64(&bits)));

        let prefill_runs = num("PROTO409_PREFILL_RUNS", 0);
        let mut prefill_ms = Vec::new();
        let mut logits_hashes = Vec::new();
        for _ in 0..prefill_runs {
            let mut sequence = pool.alloc(context).expect("seq alloc");
            let mut logits = vec![0f32; vocab];
            let started = Instant::now();
            prefill_prompt(&model, &pool, &mut sequence, &tokens, &prompt, &embedding, chunk, &mut logits)
                .unwrap_or_else(|e| panic!("{name}: prefill: {e}"));
            prefill_ms.push(started.elapsed().as_secs_f64() * 1e3);
            logits_hashes.push(format!("{:016x}", fnv64(&bits_of(&logits))));
            drop(sequence);
        }
        if !prefill_ms.is_empty() {
            let raw = prefill_ms.iter().map(|ms| format!("{ms:.3}")).collect::<Vec<_>>().join(",");
            let mut sorted = prefill_ms.clone();
            sorted.sort_by(f64::total_cmp);
            summary.note(&format!(
                "prefill {name}: tokens={} chunk={chunk} runs={} median_ms={:.3} p10_ms={:.3} p90_ms={:.3} min_ms={:.3} raw_ms={raw} last_logits_fnv={}",
                tokens.len(),
                sorted.len(),
                percentile(&sorted, 0.5),
                percentile(&sorted, 0.1),
                percentile(&sorted, 0.9),
                sorted[0],
                logits_hashes.join(",")
            ));
        }
        drop(embedding);
    }

    drop(pool);
    drop(model);
    let _ = artifact.release_arena(&mut device);
    summary.note("done");
}

/// No GPU: compares two labels' embeddings byte for byte and their canary
/// lines' predictions and logits hashes, `PROTO409_A` against `PROTO409_B`.
#[test]
#[ignore = "PROTOTYPE #409: run by hand with PROTO409_* set"]
fn proto_409_compare() {
    let out = PathBuf::from(var("PROTO409_OUT").expect("PROTO409_OUT"));
    let a = out.join(var("PROTO409_A").expect("PROTO409_A"));
    let b = out.join(var("PROTO409_B").expect("PROTO409_B"));
    let mut report = String::new();
    let mut failures = 0;
    let mut names: Vec<_> = std::fs::read_dir(&a)
        .expect("read A")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("emb_") && n.ends_with(".bin"))
        .collect();
    names.sort();
    for name in names {
        let left = std::fs::read(a.join(&name)).expect("read A embedding");
        let Ok(right) = std::fs::read(b.join(&name)) else {
            let _ = writeln!(report, "{name}: absent in B");
            continue;
        };
        let differing = left.chunks(2).zip(right.chunks(2)).filter(|(x, y)| x != y).count();
        let same = left == right;
        failures += usize::from(!same);
        let _ = writeln!(
            report,
            "{name}: {} bytes, {} ({differing} of {} BF16 values differ)",
            left.len(),
            if same { "IDENTICAL" } else { "DIFFERENT" },
            left.len() / 2
        );
    }
    let canary_lines = |dir: &Path| -> Vec<String> {
        std::fs::read_to_string(dir.join("summary.txt"))
            .unwrap_or_default()
            .lines()
            .filter(|l| l.starts_with("canary ") && l.contains(" agree="))
            .map(str::to_owned)
            .collect()
    };
    let (left, right) = (canary_lines(&a), canary_lines(&b));
    for (l, r) in left.iter().zip(&right) {
        let same = l == r;
        failures += usize::from(!same);
        let head: String = l.split(" predictions=").next().unwrap_or("").to_owned();
        let _ = writeln!(report, "{head}: predictions+logits {}", if same { "IDENTICAL" } else { "DIFFERENT" });
    }
    if left.len() != right.len() {
        failures += 1;
        let _ = writeln!(report, "canary lines: {} in A, {} in B", left.len(), right.len());
    }
    eprintln!("proto409 compare {} vs {}:\n{report}", a.display(), b.display());
    std::fs::write(out.join(format!("compare_{}_vs_{}.txt", var("PROTO409_A").unwrap(), var("PROTO409_B").unwrap())), &report)
        .expect("write the comparison");
    assert_eq!(failures, 0, "{report}");
}
