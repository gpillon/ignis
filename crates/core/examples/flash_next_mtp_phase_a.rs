//! Spec flash-next/07 phase A, the engine side: the trunk states the MTP
//! prototype is fed, and what an extra row costs a decode round.
//!
//! ```text
//! flash_next_mtp_phase_a corpus <out-dir>
//! flash_next_mtp_phase_a cost <out-dir>
//! ```
//!
//! - **corpus.** Prompts cut from the converter's reference windows (the
//!   allowlisted corpus, spec 01): twelve of 1,536 tokens (code, Python,
//!   prose, English, chat) and four long ones (the code and the prose
//!   document at 8,192 and 24,576 tokens). Each is continued by 512 greedy
//!   tokens through decode rounds (three lanes at a time), then the prompt
//!   and its continuation are prefilled once more with the residual tap
//!   armed: `<name>.tokens.u32`, `.stacks.bf16` (`[tokens][10240]`),
//!   `.argmax.u32` and `.margin.f32` (the engine's own pick after every
//!   position) and `manifest.json`.
//! - **cost.** Decode rounds of 1-8 rows on a load of eight lanes, timed per
//!   round (staging included), in two shapes: `consecutive` -- L groups of w
//!   lanes, lane j of a group holding its text up to token P + j, so a round
//!   runs w consecutive tokens of one text per group, as a verify of w
//!   columns does -- and `distinct`, every row another text (the three-lane
//!   proxy's regime). `cost.json`.
//!
//! The load matches the served one where it matters for a round's cost: the
//! hq-e8-2b KV, captured graphs and a 17.0 GB expert cache (the served plan
//! at the 4G headroom default gives 16.2-17.2 GB). Machine-local:
//! `F:/ai/models/Qwen3.8-Flash-Next-ignis/` or `IGNIS_FLASH_NEXT_DIR`. Needs
//! the GPU lock and ~38 GB of free RAM for the pinned expert pool.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use ignis_core::flash_next::{EngineOptions, FlashNextEngine};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const EXPERT_CACHE_BYTES: u64 = 17_000_000_000;
const GENERATED: usize = 512;
const SHORT_PROMPT: usize = 1536;
const ROUNDS: usize = 40;
const WARMUP_ROUNDS: usize = 8;

/// (name, kind, reference set, window, prompt tokens). test2048 windows are
/// stored padded to 2,048; long8192's windows 0-3 are one code document and
/// 4-7 one prose document, consecutive.
const PROMPTS: &[(&str, &str, &str, usize, usize)] = &[
    ("code0", "code", "test2048", 0, SHORT_PROMPT),
    ("code1", "code", "test2048", 1, SHORT_PROMPT),
    ("code6", "code", "test2048", 6, SHORT_PROMPT),
    ("code7", "code", "test2048", 7, SHORT_PROMPT),
    ("py32", "code", "test2048", 32, SHORT_PROMPT),
    ("py33", "code", "test2048", 33, SHORT_PROMPT),
    ("prose2", "prose", "test2048", 2, SHORT_PROMPT),
    ("prose3", "prose", "test2048", 3, SHORT_PROMPT),
    ("en8", "prose", "test2048", 8, SHORT_PROMPT),
    ("en9", "prose", "test2048", 9, SHORT_PROMPT),
    ("chat4", "prose", "test2048", 4, SHORT_PROMPT),
    ("chat5", "prose", "test2048", 5, SHORT_PROMPT),
    ("code-8k", "code", "long8192", 0, 8192),
    ("prose-8k", "prose", "long8192", 4, 8192),
    ("code-24k", "code", "long8192", 0, 24576),
    ("prose-24k", "prose", "long8192", 4, 24576),
];

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

fn read_u32s(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

/// A prompt's tokens: `len` tokens from the start of window `window` of a
/// reference set, running on into the next windows (long8192's are
/// consecutive slices of one document).
fn prompt_tokens(set: &str, window: usize, len: usize) -> Result<Vec<u32>, String> {
    let tokens = read_u32s(&model_dir().join("references").join(set).join("tokens.u32"))?;
    let stride = if set == "test2048" { 2048 } else { 8192 };
    let start = window * stride;
    if set == "test2048" && len > stride {
        return Err(format!("{set} windows are {stride} tokens, not {len}"));
    }
    tokens.get(start..start + len).map(<[u32]>::to_vec).ok_or_else(|| format!("{set}[{window}] has no {len} tokens"))
}

fn load(decode_lanes: u32, max_context_tokens: u32) -> Result<FlashNextEngine, String> {
    let options = EngineOptions {
        decode_lanes,
        max_context_tokens,
        expert_cache_bytes: EXPERT_CACHE_BYTES,
        ..EngineOptions::default()
    };
    let engine = FlashNextEngine::load(&model_dir(), options)?;
    if engine.graphs_ready() != (1u32 << decode_lanes) - 1 {
        return Err(format!("round graphs {:#b} of {decode_lanes} lanes: {:?}", engine.graphs_ready(), engine.graph_error()));
    }
    Ok(engine)
}

fn corpus(out: &Path) -> Result<(), String> {
    let mut engine = load(3, 32 * 1024)?;
    let prompts: Vec<Vec<u32>> =
        PROMPTS.iter().map(|&(_, _, set, window, len)| prompt_tokens(set, window, len)).collect::<Result<_, _>>()?;
    let mut texts: Vec<Vec<u32>> = Vec::new();
    for batch in prompts.chunks(3) {
        let generated = engine.generate(batch, GENERATED)?;
        texts.extend(batch.iter().zip(generated).map(|(p, g)| [p.as_slice(), &g].concat()));
        eprintln!("generated {} of {}", texts.len(), prompts.len());
    }
    let mut manifest = Vec::new();
    for (&(name, kind, set, window, len), text) in PROMPTS.iter().zip(&texts) {
        let span = engine.tapped_span(text)?;
        let base = out.join(name);
        write_bytes(&base.with_extension("tokens.u32"), &text.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>())?;
        write_bytes(&base.with_extension("stacks.bf16"), &span.stacks.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
        write_bytes(&base.with_extension("argmax.u32"), &span.argmax.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>())?;
        write_bytes(&base.with_extension("margin.f32"), &span.margin.iter().flat_map(|m| m.to_le_bytes()).collect::<Vec<_>>())?;
        // How often the prefill's pick after a generated token is the
        // decode's next token (a near-tie flip between the routes otherwise).
        let agree = (len..text.len() - 1).filter(|&p| span.argmax[p] == text[p + 1]).count();
        eprintln!("{name}: {} tokens tapped, prefill pick = decode token at {agree}/{}", text.len(), text.len() - 1 - len);
        manifest.push(serde_json::json!({
            "name": name, "kind": kind, "source": format!("{set}[{window}]"),
            "prompt_tokens": len, "generated_tokens": text.len() - len, "width": span.width,
            "prefill_pick_equals_decode": agree,
        }));
    }
    let json = serde_json::json!({ "texts": manifest, "expert_cache_bytes": EXPERT_CACHE_BYTES });
    write_bytes(&out.join("manifest.json"), serde_json::to_string_pretty(&json).unwrap().as_bytes())
}

fn summary(times: &[Duration]) -> serde_json::Value {
    let mut ms: Vec<f64> = times[WARMUP_ROUNDS..].iter().map(|d| d.as_secs_f64() * 1e3).collect();
    ms.sort_by(f64::total_cmp);
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    serde_json::json!({ "median_ms": ms[ms.len() / 2], "mean_ms": mean, "rounds": ms.len() })
}

fn cost(out: &Path) -> Result<(), String> {
    let short: Vec<Vec<u32>> = PROMPTS
        .iter()
        .filter(|p| p.4 == SHORT_PROMPT)
        .map(|&(name, ..)| read_u32s(&out.join(name).with_extension("tokens.u32")))
        .collect::<Result<_, _>>()?;
    let mut engine = load(8, 4096)?;
    let mut cells = Vec::new();
    // Warm the expert cache on decode before the first timed cell.
    engine.generate_timed(&[short[0][..SHORT_PROMPT].to_vec()], 64)?;
    for (lanes, widths) in [(1usize, 1..=8usize), (2, 1..=4), (3, 1..=2)] {
        for width in widths {
            let prompts: Vec<Vec<u32>> = (0..lanes)
                .flat_map(|g| (0..width).map(move |j| (g, j)))
                .map(|(g, j)| short[g][..SHORT_PROMPT + j].to_vec())
                .collect();
            let (_, times) = engine.generate_timed(&prompts, ROUNDS)?;
            let cell = serde_json::json!({ "shape": "consecutive", "lanes": lanes, "width": width, "rows": lanes * width, "time": summary(&times) });
            eprintln!("{cell}");
            cells.push(cell);
        }
    }
    for rows in 1..=8usize {
        let prompts: Vec<Vec<u32>> = short.iter().take(rows).map(|t| t[..SHORT_PROMPT].to_vec()).collect();
        let (_, times) = engine.generate_timed(&prompts, ROUNDS)?;
        let cell = serde_json::json!({ "shape": "distinct", "lanes": rows, "width": 1, "rows": rows, "time": summary(&times) });
        eprintln!("{cell}");
        cells.push(cell);
    }
    let json = serde_json::json!({ "cells": cells, "expert_cache_bytes": EXPERT_CACHE_BYTES, "rounds": ROUNDS, "warmup_rounds": WARMUP_ROUNDS });
    write_bytes(&out.join("cost.json"), serde_json::to_string_pretty(&json).unwrap().as_bytes())
}

/// The consecutive shape again, widths interleaved per text group and the
/// width-1 round measured first and last, so a cell's cost is read against
/// its own group's spec-off round and the drift between them shows:
/// `cost_repeat.json`.
fn cost_repeat(out: &Path) -> Result<(), String> {
    let short: Vec<Vec<u32>> = PROMPTS
        .iter()
        .filter(|p| p.4 == SHORT_PROMPT)
        .map(|&(name, ..)| read_u32s(&out.join(name).with_extension("tokens.u32")))
        .collect::<Result<_, _>>()?;
    let mut engine = load(8, 4096)?;
    let groups: [(&[usize], &[usize]); 6] = [
        (&[0], &[1, 2, 3, 4, 5, 1]),
        (&[6], &[1, 2, 3, 4, 5, 1]),
        (&[8], &[1, 2, 3, 4, 5, 1]),
        (&[2], &[1, 2, 3, 4, 5, 1]),
        (&[0, 6], &[1, 2, 3, 4, 1]),
        (&[8, 2, 4], &[1, 2, 1]),
    ];
    let mut cells = Vec::new();
    for (texts, widths) in groups {
        engine.generate_timed(&[short[texts[0]][..SHORT_PROMPT].to_vec()], 32)?;
        for (order, &width) in widths.iter().enumerate() {
            let prompts: Vec<Vec<u32>> =
                texts.iter().flat_map(|&t| (0..width).map(move |j| (t, j))).map(|(t, j)| short[t][..SHORT_PROMPT + j].to_vec()).collect();
            let (_, times) = engine.generate_timed(&prompts, ROUNDS)?;
            let cell = serde_json::json!({ "texts": texts, "order": order, "lanes": texts.len(), "width": width, "rows": texts.len() * width, "time": summary(&times) });
            eprintln!("{cell}");
            cells.push(cell);
        }
    }
    let json = serde_json::json!({ "cells": cells, "expert_cache_bytes": EXPERT_CACHE_BYTES, "rounds": ROUNDS, "warmup_rounds": WARMUP_ROUNDS });
    write_bytes(&out.join("cost_repeat.json"), serde_json::to_string_pretty(&json).unwrap().as_bytes())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mode, out) = match args.as_slice() {
        [_, mode, out] => (mode.as_str(), PathBuf::from(out)),
        _ => {
            eprintln!("usage: flash_next_mtp_phase_a corpus|cost|cost-repeat <out-dir>");
            std::process::exit(2);
        }
    };
    let result = fs::create_dir_all(&out).map_err(|e| e.to_string()).and_then(|()| match mode {
        "corpus" => corpus(&out),
        "cost" => cost(&out),
        "cost-repeat" => cost_repeat(&out),
        other => Err(format!("unknown mode {other}")),
    });
    if let Err(e) = result {
        eprintln!("flash_next_mtp_phase_a: {e}");
        std::process::exit(1);
    }
}
