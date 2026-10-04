//! Flash-Next's n-gram row ids, bit for bit against the checkpoint's own
//! hashing code (spec flash-next/04, GitHub #302).
//!
//! The fixture (`fixtures/flash_next/ngram_ids.json`) was recorded by
//! `tools/flash-next-fixtures/record_ngram_ids.py`: the checkpoint's modeling
//! code (`Qwen4ExpTextNGramEmbedding`) with the hash buffers the checkpoint
//! stores, on fixed token streams with and without EOS, each hashed whole and
//! chunk by chunk through transformers' own cache.

use std::path::PathBuf;

use ignis_core::compute::ModelConfig;
use ignis_core::ngram::{NgramContext, NgramHashBuffers, NgramHasher};
use serde_json::Value;

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("flash_next")
        .join("ngram_ids.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("parse the n-gram fixture")
}

fn i64s(value: &Value) -> Vec<i64> {
    value.as_array().expect("an array").iter().map(|v| v.as_i64().expect("an i64")).collect()
}

fn u32s(value: &Value) -> Vec<u32> {
    value.as_array().expect("an array").iter().map(|v| v.as_u64().expect("a u64") as u32).collect()
}

fn stored_buffers(fixture: &Value) -> NgramHashBuffers {
    let stored = &fixture["stored"];
    NgramHashBuffers {
        layer_multipliers: i64s(&stored["layer_multipliers"]),
        head_vocab_sizes: i64s(&stored["ngram_heads_vocab_sizes"]),
        head_offsets: i64s(&stored["ngram_heads_offsets"]),
    }
}

fn hasher(fixture: &Value) -> NgramHasher {
    let geometry = ModelConfig::qwen38_flash_next().ngram.expect("Flash-Next's n-gram geometry");
    NgramHasher::new(geometry, stored_buffers(fixture)).expect("the stored buffers fit the geometry")
}

/// Every stream's ids, one sequence from its first token: the checkpoint's
/// ids for every token, heads in the checkpoint's order.
#[test]
fn whole_streams_hash_to_the_checkpoints_ids() {
    let fixture = fixture();
    let hasher = hasher(&fixture);
    let streams = fixture["streams"].as_array().expect("streams");
    assert!(streams.len() >= 10, "the fixture's streams");
    for stream in streams {
        let name = stream["name"].as_str().expect("name");
        let tokens = u32s(&stream["tokens"]);
        let mut context = NgramContext::new(&hasher);
        let mut ids = Vec::new();
        hasher.hash(&mut context, &tokens, &mut ids);
        let expected: Vec<u64> = stream["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .flat_map(|row| row.as_array().expect("a token's ids").iter().map(|v| v.as_u64().expect("an id")))
            .collect();
        assert_eq!(ids.len(), tokens.len() * hasher.heads(), "{name}: 16 ids per token");
        assert_eq!(ids, expected, "{name}");
    }
}

/// The same streams fed in the chunk sizes the recorder fed transformers'
/// cache with: a sequence's ids do not depend on how its tokens arrive, so a
/// decode step (one token) hashes as the prompt would have.
#[test]
fn chunked_streams_hash_as_the_whole_stream_does() {
    let fixture = fixture();
    let hasher = hasher(&fixture);
    for stream in fixture["streams"].as_array().expect("streams") {
        let name = stream["name"].as_str().expect("name");
        let tokens = u32s(&stream["tokens"]);
        let mut whole = Vec::new();
        hasher.hash(&mut NgramContext::new(&hasher), &tokens, &mut whole);
        let sizes = stream["chunk_sizes_equal_to_whole"].as_array().expect("chunk sizes");
        assert!(!sizes.is_empty(), "{name}: the recorder checked at least one chunking");
        for size in sizes {
            let size = size.as_u64().expect("a size") as usize;
            let mut context = NgramContext::new(&hasher);
            let mut ids = Vec::new();
            for chunk in tokens.chunks(size) {
                hasher.hash(&mut context, chunk, &mut ids);
            }
            assert_eq!(ids, whole, "{name}: chunks of {size}");
        }
    }
}

/// The buffers the config rebuilds are the ones the checkpoint stores (the
/// recorder found them equal), so a mis-stored artifact buffer is caught.
#[test]
fn the_config_derives_the_stored_buffers() {
    let fixture = fixture();
    assert_eq!(fixture["stored_equals_computed"], Value::Bool(true));
    let cfg = ModelConfig::qwen38_flash_next();
    let derived = NgramHashBuffers::derive(&cfg.ngram.expect("n-gram geometry"), cfg.vocab);
    assert_eq!(derived, stored_buffers(&fixture));
    assert_eq!(derived.table_rows(), fixture["table_rows"].as_u64().expect("table_rows"));
    assert_eq!(
        NgramHasher::new(cfg.ngram.unwrap(), derived).unwrap().padded_table_rows(),
        fixture["padded_table_rows"].as_u64().expect("padded_table_rows"),
    );
}
