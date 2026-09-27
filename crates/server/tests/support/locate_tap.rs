//! What the `locate` GPU harnesses share once the attention tap has captured
//! a prefill: the KV mode they read, the consumed capture's self-check, the
//! scoring of every armed head, and the f16 the dumps are written in.
//! Spec 18 phase A (`attention_head_locate_gpu.rs`, GitHub #274) and spec 19
//! phase 1 (`attention_span_locate_gpu.rs`, GitHub #276).
//!
//! GPU-only, like the harnesses: `ignis_core::attn_tap` exists under the
//! `attn-tap` feature alone.

#![allow(dead_code)]

use ignis_core::KvFormat;
use ignis_core::attn_tap::{AttnTapCapture, Q_HEADS};
use ignis_core::hq_ring::{HqRing, PromptSource, prompt_source};

/// The consumed capture's self-check bounds, as `attention_head_point_gpu.rs`
/// sets them: an exact row sits near 0.002-0.004 relative L2 from the rotated
/// pre-codec key, the codec's rows at a median ~0.37.
pub const EXACT_ROW_REL_ERR: f64 = 0.1;
pub const CODEC_ROW_MIN_MEDIAN: f64 = 0.2;
pub const CODEC_ROW_MAX_MEDIAN: f64 = 0.6;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvMode {
    Bf16,
    HqConsumed,
}

impl KvMode {
    /// `IGNIS_LOCATE_KV`: `hq` (the default, the consumed keys) or `bf16`.
    pub fn from_env() -> Self {
        match std::env::var("IGNIS_LOCATE_KV").as_deref() {
            Err(_) | Ok("hq") => Self::HqConsumed,
            Ok("bf16") => Self::Bf16,
            Ok(other) => panic!("IGNIS_LOCATE_KV must be hq or bf16, not {other:?}"),
        }
    }

    pub fn format(self) -> KvFormat {
        match self {
            Self::Bf16 => KvFormat::Bf16,
            Self::HqConsumed => KvFormat::HqE8_2b,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::HqConsumed => "hq",
        }
    }
}

pub fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    Some(values[values.len() / 2])
}

/// f32 to IEEE half, round to nearest even (as `attention_head_point_gpu.rs`
/// writes its dump).
pub fn f32_to_f16(value: f32) -> u16 {
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
pub fn check_layer(
    capture: &AttnTapCapture,
    layer: usize,
    kv_head: usize,
    total: usize,
    query_chunk_start: usize,
    ring: &HqRing,
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

/// Every armed head's scores from query index `query` over `positions`,
/// `[layer][q_head][position]` (one Vec per layer), one thread per layer.
pub fn head_scores(capture: &AttnTapCapture, kv_mode: KvMode, query: usize, positions: &[usize]) -> Vec<Vec<f32>> {
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..capture.ordinals.len())
            .map(|layer| {
                scope.spawn(move || {
                    let mut out = Vec::with_capacity(Q_HEADS * positions.len());
                    for q_head in 0..Q_HEADS {
                        let scores = match kv_mode {
                            KvMode::HqConsumed => capture.consumed_scores(layer, query, q_head, positions),
                            KvMode::Bf16 => capture.scores(layer, query, q_head, positions),
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

/// `[layer][q_head][position]` scores as the dump's f16 bytes, checked
/// finite.
pub fn f16_bytes(scores: &[Vec<f32>], what: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(scores.iter().map(Vec::len).sum::<usize>() * 2);
    for layer in scores {
        assert!(layer.iter().all(|s| s.is_finite()), "{what}: a non-finite score");
        for &s in layer {
            bytes.extend_from_slice(&f32_to_f16(s).to_le_bytes());
        }
    }
    bytes
}
