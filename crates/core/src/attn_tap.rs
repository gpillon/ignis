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
//! given, not what it decodes. [`with_attn_tap_hq`] adds the other half: the
//! keys the hq prompt attention actually consumed, in the codec's rotated
//! frame, scored with [`AttnTapCapture::consumed_scores`]. Since GitHub #257
//! the cache view carries the residual window, so the route keeps the
//! current chunk, the 32 sink keys and the ring exact and decodes the rest —
//! which key came from where is `crate::hq_ring::prompt_source` over the ring
//! as it stood before the query's chunk was appended (GitHub #258), and
//! `attn_tap_hq_consumed_gpu.rs` holds the capture to it row by row.

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
        pub fn ignis_attn_tap_arm_consumed(kc_out: *mut u16) -> i32;
        pub fn ignis_attn_tap_disarm(
            rows_written: *mut i64,
            queries_seen: *mut i32,
            consumed_rows: *mut i64,
            consumed_chunk_start: *mut i64,
        ) -> i32;
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
    /// The keys hq-e8-2b attention consumed for the first query's chunk,
    /// `[layer][max_positions][KV_HEADS][HEAD_DIM]` BF16, **rotated frame**;
    /// empty unless taken through [`with_attn_tap_hq`].
    pub kc: Vec<u16>,
    /// Consumed rows each armed layer copied (0 = not captured).
    pub consumed_rows: Vec<i64>,
    /// Where each armed layer's captured chunk began (-1 = not captured):
    /// the `chunk_start` `crate::hq_ring::prompt_source` needs.
    pub consumed_chunk_start: Vec<i64>,
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

    /// The consumed (rotated-frame) key row of `kv_head` at `position` in
    /// armed layer `layer`.
    pub fn consumed_key(&self, layer: usize, position: usize, kv_head: usize) -> &[u16] {
        let at = ((layer * self.max_positions as usize + position) * KV_HEADS + kv_head) * HEAD_DIM;
        &self.kc[at..at + HEAD_DIM]
    }

    /// The same logits as [`Self::scores`], but against the keys hq
    /// attention consumed: the query is rotated into the codec's frame, which
    /// is orthonormal, so the dot product is the one the kernel formed.
    pub fn consumed_scores(
        &self,
        layer: usize,
        query: usize,
        q_head: usize,
        positions: &[usize],
    ) -> Vec<f32> {
        let kv_head = q_head / (Q_HEADS / KV_HEADS);
        let q: Vec<f32> = self.query(layer, query, q_head).iter().map(|&b| bf16_to_f32(b)).collect();
        let rq = hq_rotate(&q);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        positions
            .iter()
            .map(|&p| {
                let k = self.consumed_key(layer, p, kv_head);
                rq.iter().zip(k).map(|(a, &b)| a * bf16_to_f32(b)).sum::<f32>() * scale
            })
            .collect()
    }

    /// `|R k - kc| / |R k|` for one row: how far the consumed key is from the
    /// rotation of the key the layer produced. About one BF16 rounding for
    /// the exact sources, the codec's error for the rest.
    pub fn consumed_key_rel_err(&self, layer: usize, position: usize, kv_head: usize) -> f32 {
        self.consumed_key_rel_err_to(layer, position, kv_head, position)
    }

    /// `|R k - kc| / |R k|` with `k` the key the layer produced at
    /// `key_position` and `kc` the row attention consumed at `position`:
    /// which key a consumed row actually is. On the reference's launch order
    /// the hq ring serves one key's row for another's
    /// (`crate::hq_ring::PromptSource::Clobbered`, GitHub #258), and this is
    /// how a test tells that apart from a decoded row.
    pub fn consumed_key_rel_err_to(
        &self,
        layer: usize,
        position: usize,
        kv_head: usize,
        key_position: usize,
    ) -> f32 {
        let k: Vec<f32> = self.key(layer, key_position, kv_head).iter().map(|&b| bf16_to_f32(b)).collect();
        let rk = hq_rotate(&k);
        let kc = self.consumed_key(layer, position, kv_head);
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for (a, &b) in rk.iter().zip(kc) {
            let d = a - bf16_to_f32(b);
            num += d * d;
            den += a * a;
        }
        (num / den.max(f32::MIN_POSITIVE)).sqrt()
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
    armed(ordinals, query_positions, max_positions, false, f)
}

/// [`with_attn_tap`], plus the keys hq-e8-2b attention consumed for the
/// chunk holding `query_positions[0]` ([`AttnTapCapture::kc`]). On a cache or
/// route that does not materialize them, `consumed_rows` comes back 0.
pub fn with_attn_tap_hq<T>(
    ordinals: &[i32],
    query_positions: &[i64],
    max_positions: i64,
    f: impl FnOnce() -> T,
) -> Result<(T, AttnTapCapture), String> {
    armed(ordinals, query_positions, max_positions, true, f)
}

fn armed<T>(
    ordinals: &[i32],
    query_positions: &[i64],
    max_positions: i64,
    consumed: bool,
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
    let mut kc = if consumed { vec![0u16; k.len()] } else { Vec::new() };
    if consumed {
        let rc = unsafe { ffi::ignis_attn_tap_arm_consumed(kc.as_mut_ptr()) };
        if rc != 0 {
            let err = last_error();
            unsafe {
                ffi::ignis_attn_tap_disarm(
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                );
            }
            return Err(err);
        }
    }

    struct Disarm {
        rows_written: Vec<i64>,
        queries_seen: Vec<i32>,
        consumed_rows: Vec<i64>,
        consumed_start: Vec<i64>,
        done: bool,
    }
    impl Drop for Disarm {
        fn drop(&mut self) {
            if !self.done {
                unsafe {
                    ffi::ignis_attn_tap_disarm(
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                    );
                }
            }
        }
    }
    let mut guard = Disarm {
        rows_written: vec![0; n_layers],
        queries_seen: vec![0; n_queries],
        consumed_rows: vec![0; n_layers],
        consumed_start: vec![-1; n_layers],
        done: false,
    };
    let value = f();
    let rc = unsafe {
        ffi::ignis_attn_tap_disarm(
            guard.rows_written.as_mut_ptr(),
            guard.queries_seen.as_mut_ptr(),
            guard.consumed_rows.as_mut_ptr(),
            guard.consumed_start.as_mut_ptr(),
        )
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
            kc,
            consumed_rows: std::mem::take(&mut guard.consumed_rows),
            consumed_chunk_start: std::mem::take(&mut guard.consumed_start),
        },
    ))
}

/// The engine-wide sign of coordinate `d` (`hq_engine_sign`,
/// kernel/vendor/src/ops/kernel/hq_codec.cuh), bit for bit.
pub fn hq_engine_sign(d: u32) -> f32 {
    let mut x = 0x005E_ED01u32 ^ d.wrapping_mul(0x9E37_79B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    if x & 1 == 1 { 1.0 } else { -1.0 }
}

/// The codec's rotation, `R = H * diag(signs) / 16`: signs, then a natural-order
/// Walsh-Hadamard transform over 256 coordinates, then `1 / sqrt(256)` — the
/// `hq_fwht256_sign` + `hq_store_rotated_row_warp` pair. Orthonormal, so
/// `R q . R k == q . k`.
pub fn hq_rotate(v: &[f32]) -> Vec<f32> {
    assert_eq!(v.len(), HEAD_DIM, "hq_rotate: a row is {HEAD_DIM} coordinates");
    let mut x: Vec<f32> = v.iter().enumerate().map(|(d, &a)| a * hq_engine_sign(d as u32)).collect();
    let mut len = 1;
    while len < HEAD_DIM {
        for base in (0..HEAD_DIM).step_by(2 * len) {
            for i in base..base + len {
                let (a, b) = (x[i], x[i + len]);
                x[i] = a + b;
                x[i + len] = a - b;
            }
        }
        len <<= 1;
    }
    let inv = 1.0 / (HEAD_DIM as f32).sqrt();
    x.iter_mut().for_each(|a| *a *= inv);
    x
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
            kc: Vec::new(),
            consumed_rows: vec![0],
            consumed_chunk_start: vec![-1],
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

    #[test]
    fn the_hq_rotation_is_orthonormal() {
        // Two unlike rows: norms and the dot product survive the rotation,
        // which is the whole reason a rotated query can score rotated keys.
        let a: Vec<f32> = (0..HEAD_DIM).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.25).collect();
        let b: Vec<f32> = (0..HEAD_DIM).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.5).collect();
        let (ra, rb) = (hq_rotate(&a), hq_rotate(&b));
        let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
        assert!((dot(&a, &a) - dot(&ra, &ra)).abs() < 1e-3 * dot(&a, &a));
        assert!((dot(&a, &b) - dot(&ra, &rb)).abs() < 1e-3 * dot(&a, &a).max(dot(&b, &b)));
        // and it is a rotation, not the identity
        assert!(a.iter().zip(&ra).any(|(x, y)| (x - y).abs() > 1e-3));
    }

    #[test]
    fn the_engine_signs_are_a_balanced_deterministic_diagonal() {
        let signs: Vec<f32> = (0..HEAD_DIM as u32).map(hq_engine_sign).collect();
        assert!(signs.iter().all(|&s| s == 1.0 || s == -1.0));
        let plus = signs.iter().filter(|&&s| s == 1.0).count();
        assert!((96..=160).contains(&plus), "{plus} of 256 positive");
        assert_eq!(signs, (0..HEAD_DIM as u32).map(hq_engine_sign).collect::<Vec<_>>());
    }
}
