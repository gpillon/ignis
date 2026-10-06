//! Spec flash-next/07 phase B, the engine side of the MTP head's
//! calibration (layout.md §13.4): the trunk's states over the trunk's own
//! calibration corpus, as the served engine computes them.
//!
//! ```text
//! flash_next_mtp_calibration <calib-dir>
//! ```
//!
//! `<calib-dir>` holds what `tools/flash-next-mtp/convert_head.py chunks`
//! wrote: `chunks.json` (`{"chunks": [{"name", "tokens"}, ...]}`) and one
//! `<name>.tokens.u32` per chunk. Each chunk is prefilled once with the
//! residual tap armed and its final pre-mixer stacks written to
//! `<name>.stacks.bf16` (`[tokens][10240]` BF16), through a temporary file
//! and a rename, so a rerun skips the chunks already written and redoes a
//! torn one. `tap.json` records what the states came from (the container,
//! its directory, the KV format), for the converter's record.
//!
//! The load is the served one where it matters for the states: the main
//! container, hq-e8-2b KV, 2,048-token prefill chunks. Machine-local:
//! `F:/ai/models/Qwen3.8-Flash-Next-ignis/` or `IGNIS_FLASH_NEXT_DIR`. Needs
//! the GPU lock and ~38 GB of free RAM for the pinned expert pool.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ignis_core::flash_next::{EngineOptions, FlashNextEngine};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const EXPERT_CACHE_BYTES: u64 = 17_000_000_000;
/// One position's stack: four streams of 2,560.
const WIDTH: usize = 4 * 2560;

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

fn read_u32s(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn calibrate(dir: &Path) -> Result<(), String> {
    let manifest = dir.join("chunks.json");
    let text = fs::read_to_string(&manifest).map_err(|e| format!("read {}: {e}", manifest.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", manifest.display()))?;
    let chunks = json["chunks"].as_array().ok_or_else(|| format!("{} has no chunks", manifest.display()))?;
    let options = EngineOptions {
        decode_lanes: 1,
        max_context_tokens: 4096,
        expert_cache_bytes: EXPERT_CACHE_BYTES,
        capture_graphs: false,
        ..EngineOptions::default()
    };
    let kv_format = options.kv_format.as_str();
    let mut engine = FlashNextEngine::load(&model_dir(), options)?;
    let started = Instant::now();
    for (at, chunk) in chunks.iter().enumerate() {
        let name = chunk["name"].as_str().ok_or("a chunk has no name")?;
        let tokens = read_u32s(&dir.join(format!("{name}.tokens.u32")))?;
        if Some(tokens.len() as u64) != chunk["tokens"].as_u64() {
            return Err(format!("{name}: {} tokens, chunks.json says {}", tokens.len(), chunk["tokens"]));
        }
        let out = dir.join(format!("{name}.stacks.bf16"));
        if fs::metadata(&out).ok().map(|m| m.len() as usize) == Some(tokens.len() * WIDTH * 2) {
            continue;
        }
        let span = engine.tapped_span(&tokens)?;
        if span.width != WIDTH || span.stacks.len() != tokens.len() * WIDTH {
            return Err(format!("{name}: the tap wrote {} values for {} tokens", span.stacks.len(), tokens.len()));
        }
        let tmp = dir.join(format!("{name}.stacks.bf16.tmp"));
        let bytes: Vec<u8> = span.stacks.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(&tmp, &bytes).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        fs::rename(&tmp, &out).map_err(|e| format!("rename {}: {e}", tmp.display()))?;
        if at % 16 == 15 || at + 1 == chunks.len() {
            eprintln!("tapped {} of {} chunks, {:.0} s", at + 1, chunks.len(), started.elapsed().as_secs_f64());
        }
    }
    let tap = serde_json::json!({
        "model_dir": model_dir().display().to_string(),
        "artifact": ignis_artifact::packer::ARTIFACT_FILE_NAME,
        "kv_format": kv_format,
        "chunks": chunks.len(),
    });
    let path = dir.join("tap.json");
    fs::write(&path, serde_json::to_string_pretty(&tap).unwrap()).map_err(|e| format!("write {}: {e}", path.display()))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, dir] = args.as_slice() else {
        eprintln!("usage: flash_next_mtp_calibration <calib-dir>");
        std::process::exit(2);
    };
    if let Err(e) = calibrate(Path::new(dir)) {
        eprintln!("flash_next_mtp_calibration: {e}");
        std::process::exit(1);
    }
}
