//! One-shot capture of real BF16 K/V rows from a real Flash-Next prefill,
//! committed as `kernel/tests/fixtures/hq_kv_rows_flash_next.bin`
//! (+ `.provenance.json`): spec flash-next/04 acceptance 7 measures the
//! hq-e8-2b codec's error on real Flash-Next KV rows and derives its
//! tolerances from that measurement, without loading the 72 GB artifact on
//! every CTest run. The file format and row order are the 27B fixture's
//! (`hq_kv_fixture_capture_gpu.rs`), so `test_hq_codec_kv_rows.cu` reads
//! both; only the geometry in the header differs (2 KV heads, 12 KV layers).
//!
//! A *recording* test, not a regression test: running it overwrites the
//! committed fixture. `hq_kv_fixture_integrity.rs` holds the committed files
//! to their recorded SHA-256 on every plain `cargo test`. Behind the
//! non-default `kv-capture` feature, so the GPU profile never re-runs it:
//!
//! ```text
//! cargo test -p ignis-core --features cuda,kv-capture --test flash_next_kv_fixture_capture_gpu -- --ignored --nocapture
//! ```
//!
//! The prompt is the first 1024 tokens of the converter's `test2048` prose
//! window (`references/test2048/tokens.u32`, window 2: the allowlisted
//! public corpus of spec flash-next/01).

#![cfg(all(feature = "cuda", feature = "kv-capture"))]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use ignis_artifact::flash_next::FlashNextGeometry;
use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::flash_next::{EngineOptions, FlashNextEngine};
use ignis_core::{gpu_profile, KvFormat};
use sha2::{Digest, Sha256};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MODEL_ID: &str = "qwen3.8-flash-next";
/// The `test2048` window the prompt is cut from (manifest kind "prose") and
/// the tokens taken from its start.
const WINDOW: usize = 2;
const WINDOW_TOKENS: usize = 2048;
const PROMPT_TOKENS: usize = 1024;
/// KV layer ordinals over the 12 QSA layers (`seq->gqa_positions` order):
/// 0 is model layer 3, the first QSA layer, and the last is layer 47.
const KV_LAYER_ORDINALS: [i32; 4] = [0, 4, 8, 11];
const ROLES: [i32; 2] = [0, 1];
/// Positions [0, ROWS_PER_BLOCK) per (layer, role, kv_head): four pages.
const ROWS_PER_BLOCK: i32 = 256;

const FIXTURE_MAGIC: &[u8; 8] = b"IGNHQKV1";
const FIXTURE_FORMAT_VERSION: u32 = 1;

/// FNV-1a 64 over the row payload, as the 27B fixture's header carries it.
fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernel/tests/fixtures")
}

#[test]
#[ignore = "GPU, kv-capture feature: one-shot fixture capture, spec flash-next/04 acceptance 7"]
fn capture_flash_next_kv_fixture_from_a_real_prefill() {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from);
    let tokens_path = dir.join("references").join("test2048").join("tokens.u32");
    if !tokens_path.exists() && gpu_profile::skip_or_fail(&format!("{} does not exist", tokens_path.display())) {
        return;
    }
    let bytes = fs::read(&tokens_path).unwrap_or_else(|e| panic!("read {}: {e}", tokens_path.display()));
    let all: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().expect("four bytes"))).collect();
    let start = WINDOW * WINDOW_TOKENS;
    let prompt = &all[start..start + PROMPT_TOKENS];

    let options = EngineOptions { max_context_tokens: 2048, kv_format: KvFormat::Bf16, capture_graphs: false, ..EngineOptions::default() };
    let mut engine = match FlashNextEngine::load(&dir, options) {
        Ok(engine) => engine,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next engine: {e}"));
            return;
        }
    };
    let geometry = FlashNextGeometry::qwen38_flash_next();
    let (head_dim, kv_heads) = (geometry.head_dim as i32, geometry.kv_heads as i32);
    let sequence = engine.prefill_for_kv_capture(prompt).unwrap_or_else(|e| panic!("prefill: {e}"));

    let mut payload: Vec<u16> = Vec::new();
    let mut total_rows = 0usize;
    let mut first_k: Vec<Vec<u16>> = Vec::new();
    let mut abs_mean = Vec::new();
    for &layer in &KV_LAYER_ORDINALS {
        for &role in &ROLES {
            for kv_head in 0..kv_heads {
                let rows = sequence
                    .capture_kv_rows_for_test(layer, role, kv_head, 0, ROWS_PER_BLOCK, head_dim)
                    .unwrap_or_else(|e| panic!("capture layer={layer} role={role} kv_head={kv_head}: {e}"));
                assert_eq!(rows.len(), (ROWS_PER_BLOCK * head_dim) as usize);
                assert!(rows.iter().any(|&v| v != 0), "layer {layer} role {role} head {kv_head} is all zero");
                assert!(rows.iter().all(|&v| bf16_to_f32(v).is_finite()), "layer {layer} role {role} head {kv_head} holds a non-finite value");
                abs_mean.push((layer, role, kv_head, rows.iter().map(|&v| f64::from(bf16_to_f32(v).abs())).sum::<f64>() / rows.len() as f64));
                if layer == KV_LAYER_ORDINALS[0] && role == 0 {
                    first_k.push(rows.clone());
                }
                total_rows += ROWS_PER_BLOCK as usize;
                payload.extend_from_slice(&rows);
            }
        }
    }
    assert!(first_k.windows(2).any(|pair| pair[0] != pair[1]), "different kv_heads captured bit-identical rows");
    drop(sequence);

    let row_bytes: Vec<u8> = payload.iter().flat_map(|v| v.to_le_bytes()).collect();
    // The 27B fixture's header: magic, version, head_dim, kv_heads,
    // role_count, layer_count, ordinals, first_position, rows_per_block,
    // total_rows, checksum.
    let mut bin = Vec::new();
    bin.extend_from_slice(FIXTURE_MAGIC);
    for word in [FIXTURE_FORMAT_VERSION, head_dim as u32, kv_heads as u32, ROLES.len() as u32, KV_LAYER_ORDINALS.len() as u32] {
        bin.extend_from_slice(&word.to_le_bytes());
    }
    for &layer in &KV_LAYER_ORDINALS {
        bin.extend_from_slice(&(layer as u32).to_le_bytes());
    }
    for word in [0u32, ROWS_PER_BLOCK as u32, total_rows as u32] {
        bin.extend_from_slice(&word.to_le_bytes());
    }
    bin.extend_from_slice(&fnv1a64(&row_bytes).to_le_bytes());
    bin.extend_from_slice(&row_bytes);

    let out = fixtures_dir();
    let bin_path = out.join("hq_kv_rows_flash_next.bin");
    fs::write(&bin_path, &bin).unwrap_or_else(|e| panic!("write {}: {e}", bin_path.display()));
    let bin_sha256 = format!("{:x}", Sha256::digest(&bin));
    let unix = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock after epoch").as_secs();
    let provenance = serde_json::json!({
        "artifact_path": dir.join(ARTIFACT_FILE_NAME).display().to_string(),
        "model_id": MODEL_ID,
        "prompt": format!("references/test2048/tokens.u32, window {WINDOW} (prose), tokens [0, {PROMPT_TOKENS})"),
        "prompt_token_ids": prompt,
        "kv_layer_ordinals": KV_LAYER_ORDINALS,
        "model_layers": KV_LAYER_ORDINALS.map(|o| 3 + 4 * o),
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "roles": {"0": "K", "1": "V"},
        "first_position": 0,
        "rows_per_block": ROWS_PER_BLOCK,
        "total_rows": total_rows,
        "row_order": "for layer in kv_layer_ordinals { for role in [K, V] { for kv_head in 0..kv_heads { for position in first_position..first_position+rows_per_block { head_dim x bf16-bit-pattern u16 LE } } } }",
        "fixture_format": {"magic": "IGNHQKV1", "format_version": FIXTURE_FORMAT_VERSION, "checksum": "FNV-1a 64 over the row payload bytes only"},
        "capture_unix_seconds": unix,
        "bin_sha256": bin_sha256,
        "bin_bytes": bin.len(),
        "captured_by": "crates/core/tests/flash_next_kv_fixture_capture_gpu.rs",
    });
    let provenance_path = out.join("hq_kv_rows_flash_next.provenance.json");
    fs::write(&provenance_path, serde_json::to_string_pretty(&provenance).expect("serialize") + "\n")
        .unwrap_or_else(|e| panic!("write {}: {e}", provenance_path.display()));
    eprintln!("captured {total_rows} rows ({} bytes) -> {}", bin.len(), bin_path.display());
    eprintln!("per (layer, role, head) |mean|: {abs_mean:?}");
}
