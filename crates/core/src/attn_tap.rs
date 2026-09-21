//! Test-only attention-input tap (`kernel/include/ignis_attn_tap.h`).
//!
//! Captures, during one prefill, the rotated query rows of chosen positions
//! and the rotated key rows of every position for chosen GQA layers — what
//! the fused attention kernels compute their weights from, since they never
//! materialize the weights themselves. The host does the scoring. It exists
//! to measure in the engine what
//! `docs/findings/2026-09-21-one-attention-head-points.md` measured in a
//! PyTorch vehicle: whether one head's attention from the answer's first
//! position onto the image tokens points at the target.
//!
//! Gated behind the non-default `attn-tap` feature, like `kv-capture`: no
//! production build turns it on, so nothing in production can arm it. The
//! disarmed check it adds to every GQA prefill call is one flag load, and it
//! is the only part of this that production runs.
//!
//! **The keys are the ones before the KV cache.** Under a BF16 pool they are
//! exactly what attention reads; under hq-e8-2b they are what the codec was
//! given, not what it decodes. A number taken through this tap under hq says
//! so.

use std::ffi::{CStr, c_char};

/// Query heads per GQA layer.
pub const Q_HEADS: usize = 24;
/// KV heads per GQA layer; query head `h` reads KV head `h / (Q_HEADS / KV_HEADS)`.
pub const KV_HEADS: usize = 4;
/// Head dimension.
pub const HEAD_DIM: usize = 256;
/// GQA layers in the model; ordinal `o` is backbone layer `4 * o + 3`.
pub const GQA_LAYERS: usize = 16;

mod ffi {
    use super::c_char;

    unsafe extern "C" {
        pub fn ignis_attn_tap_arm(
            gqa_ordinals: *const i32,
            n_layers: i32,
            query_positions: *const i64,
            n_queries: i32,
            max_positions: i64,
            q_out: *mut u16,
            k_out: *mut u16,
        ) -> i32;
        pub fn ignis_attn_tap_disarm(rows_written: *mut i64, queries_seen: *mut i32) -> i32;
        pub fn ignis_attn_tap_last_error() -> *const c_char;
    }
}

fn last_error() -> String {
    let msg = unsafe { CStr::from_ptr(ffi::ignis_attn_tap_last_error()) };
    msg.to_string_lossy().into_owned()
}

/// What one armed prefill captured.
#[derive(Debug, Clone)]
pub struct AttnTapCapture {
    /// The armed GQA ordinals, in the order given.
    pub ordinals: Vec<i32>,
    /// The requested query positions, in the order given.
    pub query_positions: Vec<i64>,
    /// Key rows per layer the buffer was sized for.
    pub max_positions: i64,
    /// `[layer][query][Q_HEADS][HEAD_DIM]`, BF16 bit patterns.
    pub q: Vec<u16>,
    /// `[layer][max_positions][KV_HEADS][HEAD_DIM]`, BF16 bit patterns.
    pub k: Vec<u16>,
    /// Key rows each armed layer actually wrote.
    pub rows_written: Vec<i64>,
    /// Per query position: 1 when every armed layer saw it, else 0.
    pub queries_seen: Vec<i32>,
}

impl AttnTapCapture {
    /// The query row of `q_head` at query index `query` in armed layer `layer`.
    pub fn query(&self, layer: usize, query: usize, q_head: usize) -> &[u16] {
        let n_queries = self.query_positions.len();
        let at = ((layer * n_queries + query) * Q_HEADS + q_head) * HEAD_DIM;
        &self.q[at..at + HEAD_DIM]
    }

    /// The key row of `kv_head` at cache position `position` in armed layer `layer`.
    pub fn key(&self, layer: usize, position: usize, kv_head: usize) -> &[u16] {
        let at = ((layer * self.max_positions as usize + position) * KV_HEADS + kv_head) * HEAD_DIM;
        &self.k[at..at + HEAD_DIM]
    }

    /// `q . k / sqrt(HEAD_DIM)` for one head of one armed layer, from query
    /// index `query` to each of `positions` — the pre-softmax attention
    /// logits the fused kernel would have formed, in f32.
    pub fn scores(&self, layer: usize, query: usize, q_head: usize, positions: &[usize]) -> Vec<f32> {
        let kv_head = q_head / (Q_HEADS / KV_HEADS);
        let q: Vec<f32> = self.query(layer, query, q_head).iter().map(|&b| bf16_to_f32(b)).collect();
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        positions
            .iter()
            .map(|&p| {
                let k = self.key(layer, p, kv_head);
                q.iter().zip(k).map(|(a, &b)| a * bf16_to_f32(b)).sum::<f32>() * scale
            })
            .collect()
    }
}

/// Arm the tap, run `f` (one sequence's prefill), and disarm — always, even
/// when `f` panics — returning `f`'s value and what was captured.
///
/// The arm is process-wide: nothing else may prefill while `f` runs.
pub fn with_attn_tap<T>(
    ordinals: &[i32],
    query_positions: &[i64],
    max_positions: i64,
    f: impl FnOnce() -> T,
) -> Result<(T, AttnTapCapture), String> {
    let n_layers = ordinals.len();
    let n_queries = query_positions.len();
    let mut q = vec![0u16; n_layers * n_queries * Q_HEADS * HEAD_DIM];
    let mut k = vec![0u16; n_layers * max_positions.max(0) as usize * KV_HEADS * HEAD_DIM];
    let rc = unsafe {
        ffi::ignis_attn_tap_arm(
            ordinals.as_ptr(),
            n_layers as i32,
            query_positions.as_ptr(),
            n_queries as i32,
            max_positions,
            q.as_mut_ptr(),
            k.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(last_error());
    }

    struct Disarm {
        rows_written: Vec<i64>,
        queries_seen: Vec<i32>,
        done: bool,
    }
    impl Drop for Disarm {
        fn drop(&mut self) {
            if !self.done {
                unsafe {
                    ffi::ignis_attn_tap_disarm(std::ptr::null_mut(), std::ptr::null_mut());
                }
            }
        }
    }
    let mut guard = Disarm {
        rows_written: vec![0; n_layers],
        queries_seen: vec![0; n_queries],
        done: false,
    };
    let value = f();
    let rc = unsafe {
        ffi::ignis_attn_tap_disarm(guard.rows_written.as_mut_ptr(), guard.queries_seen.as_mut_ptr())
    };
    guard.done = true;
    if rc != 0 {
        return Err(last_error());
    }
    Ok((
        value,
        AttnTapCapture {
            ordinals: ordinals.to_vec(),
            query_positions: query_positions.to_vec(),
            max_positions,
            q,
            k,
            rows_written: std::mem::take(&mut guard.rows_written),
            queries_seen: std::mem::take(&mut guard.queries_seen),
        },
    ))
}

/// A BF16 bit pattern as f32 (exact: BF16 is f32's top half).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_widening_is_exact() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xbf80), -1.0);
        assert_eq!(bf16_to_f32(0x4049), 3.140625);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn a_query_head_reads_its_own_kv_group() {
        // 24 query heads over 4 KV heads: six per group, in order.
        let group = |h: usize| h / (Q_HEADS / KV_HEADS);
        assert_eq!((group(0), group(5), group(6), group(10), group(23)), (0, 0, 1, 1, 3));
    }

    #[test]
    fn scores_index_the_right_rows() {
        // One layer, one query, three positions; head 10 lives in KV group 1.
        let mut cap = AttnTapCapture {
            ordinals: vec![9],
            query_positions: vec![2],
            max_positions: 3,
            q: vec![0; Q_HEADS * HEAD_DIM],
            k: vec![0; 3 * KV_HEADS * HEAD_DIM],
            rows_written: vec![3],
            queries_seen: vec![1],
        };
        let one = 0x3f80u16;
        cap.q[10 * HEAD_DIM] = one; // head 10, dim 0 = 1
        for (p, v) in [(0usize, 0x3f80u16), (1, 0x4000), (2, 0x4040)] {
            cap.k[(p * KV_HEADS + 1) * HEAD_DIM] = v; // KV head 1, dim 0 = 1, 2, 3
            cap.k[(p * KV_HEADS) * HEAD_DIM] = 0x4120; // KV head 0 must not be read
        }
        let s = cap.scores(0, 0, 10, &[0, 1, 2]);
        let scale = 1.0 / 16.0;
        assert_eq!(s, vec![1.0 * scale, 2.0 * scale, 3.0 * scale]);
    }
}
