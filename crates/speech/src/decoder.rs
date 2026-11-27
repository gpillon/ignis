//! The Qwen3 decoder's graph, ours (spec speech/01 §2, ADR 0068):
//! per layer RMSNorm, the Q/K/V projections, QK-norm, M-RoPE, GQA attention
//! over our KV cache, the output projection, RMSNorm and SwiGLU; then the
//! final norm and the LM head.
//!
//! The KV cache is one contiguous per-transcription buffer of f16 keys and
//! values, 112 KiB per token for both sizes (28 layers × 2 × 8 heads × 128 ×
//! 2 B). It is sized for the longest clip but grows to it on demand,
//! doubling, so a short transcription commits memory for what it holds.

use crate::cpu::Isa;
use crate::encoder::Tap;
use crate::matmul::matmul;
use crate::model::Decoder;
use crate::ops::{self, AttnShape, Kv, Mask, Rope};
use crate::pool::Pool;

/// K and V for every layer: `[layer][position][kv head · head dim]`, f16.
pub struct Cache {
    k: Vec<u16>,
    v: Vec<u16>,
    kvs: usize,
    layers: usize,
    /// Positions allocated, and the most it may grow to.
    cap: usize,
    max: usize,
    /// Positions filled.
    pub len: usize,
}

impl Cache {
    /// A cache of exactly `cap` positions.
    pub fn new(dec: &Decoder, cap: usize) -> Cache {
        Cache::growing(dec, cap, cap)
    }

    /// A cache of `initial` positions that grows on demand up to `max`.
    pub fn growing(dec: &Decoder, initial: usize, max: usize) -> Cache {
        let kvs = dec.kv_heads * dec.head_dim;
        let layers = dec.layers.len();
        let cap = initial.min(max);
        Cache { k: vec![0; layers * cap * kvs], v: vec![0; layers * cap * kvs], kvs, layers, cap, max, len: 0 }
    }

    /// The most positions this cache may hold.
    pub fn max_positions(&self) -> usize {
        self.max
    }

    /// Makes room for `positions` in all, doubling the allocation; false past the maximum.
    pub fn reserve(&mut self, positions: usize) -> bool {
        if positions <= self.cap {
            return true;
        }
        if positions > self.max {
            return false;
        }
        let cap = positions.max(2 * self.cap).min(self.max);
        let (mut k, mut v) = (vec![0u16; self.layers * cap * self.kvs], vec![0u16; self.layers * cap * self.kvs]);
        let used = self.len * self.kvs;
        for l in 0..self.layers {
            let (d, s) = (l * cap * self.kvs, l * self.cap * self.kvs);
            k[d..d + used].copy_from_slice(&self.k[s..s + used]);
            v[d..d + used].copy_from_slice(&self.v[s..s + used]);
        }
        (self.k, self.v, self.cap) = (k, v, cap);
        true
    }

    /// Rolls the cache back to its first `len` positions.
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    /// Copies positions `0..len` of every layer from `other` (the prefix cache's entry).
    pub fn copy_prefix(&mut self, other: &Cache, len: usize) {
        assert!(self.kvs == other.kvs && len <= other.len && self.reserve(len));
        for l in 0..self.layers {
            let (d, s) = (l * self.cap * self.kvs, l * other.cap * other.kvs);
            self.k[d..d + len * self.kvs].copy_from_slice(&other.k[s..s + len * self.kvs]);
            self.v[d..d + len * self.kvs].copy_from_slice(&other.v[s..s + len * self.kvs]);
        }
        self.len = len;
    }

    fn layer(&self, l: usize) -> (&[u16], &[u16]) {
        let s = l * self.cap * self.kvs;
        (&self.k[s..s + self.cap * self.kvs], &self.v[s..s + self.cap * self.kvs])
    }
}

/// Which rows' logits a forward pass returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Logits {
    None,
    Last,
    /// The last n rows.
    Tail(usize),
    All,
}

/// The rows of `ids` from the embedding table.
pub fn embed(dec: &Decoder, ids: &[u32]) -> Vec<f32> {
    let h = dec.hidden;
    let mut x = vec![0f32; ids.len() * h];
    for (r, &id) in ids.iter().enumerate() {
        dec.embed.row(id, &mut x[r * h..(r + 1) * h]);
    }
    x
}

/// Runs `x` (rows of `hidden`: token embeddings or projected audio) through the decoder at the
/// cache's next positions, appending their K and V, and returns the requested logits.
pub fn forward(dec: &Decoder, pool: &Pool, isa: Isa, cache: &mut Cache, mut x: Vec<f32>, want: Logits, mut tap: Option<Tap>) -> Vec<f32> {
    let hd = dec.head_dim;
    let n = x.len() / dec.hidden;
    let past = cache.len;
    assert!(n > 0 && cache.reserve(past + n), "the KV cache holds {} positions", cache.max);
    let (hq, hkv) = (dec.heads, dec.kv_heads);
    let shape = AttnShape { hq, hkv, hd };
    let rope = Rope::text(past, n, dec.sections, dec.theta, hd);
    let kvs = cache.kvs;
    let mut att = vec![0f32; n * hq * hd];
    for (li, l) in dec.layers.iter().enumerate() {
        let mut h = x.clone();
        ops::rms_norm(isa, &mut h, &l.attn_norm, dec.eps);
        let mut q = vec![0f32; n * hq * hd];
        let mut k = vec![0f32; n * kvs];
        let mut v = vec![0f32; n * kvs];
        matmul(pool, isa, &l.q, &h, n, &mut q);
        matmul(pool, isa, &l.k, &h, n, &mut k);
        matmul(pool, isa, &l.v, &h, n, &mut v);
        ops::rms_norm(isa, &mut q, &l.q_norm, dec.eps);
        ops::rms_norm(isa, &mut k, &l.k_norm, dec.eps);
        rope.apply(isa, &mut q, hq);
        rope.apply(isa, &mut k, hkv);
        let s = li * cache.cap * kvs + past * kvs;
        ops::to_f16(isa, &k, &mut cache.k[s..s + n * kvs]);
        ops::to_f16(isa, &v, &mut cache.v[s..s + n * kvs]);
        let (kc, vc) = cache.layer(li);
        ops::attention(pool, isa, shape, &q, n, Kv::F16 { k: kc, v: vc }, past + n, Mask::Causal { past }, &mut att);
        let mut o = vec![0f32; n * dec.hidden];
        matmul(pool, isa, &l.o, &att, n, &mut o);
        ops::add(isa, &mut x, &o);
        let mut h = x.clone();
        ops::rms_norm(isa, &mut h, &l.ffn_norm, dec.eps);
        let mut g = vec![0f32; n * l.gate.n];
        let mut u = vec![0f32; n * l.up.n];
        matmul(pool, isa, &l.gate, &h, n, &mut g);
        matmul(pool, isa, &l.up, &h, n, &mut u);
        ops::silu_mul(isa, &mut g, &u);
        matmul(pool, isa, &l.down, &g, n, &mut o);
        ops::add(isa, &mut x, &o);
        if let Some(f) = tap.as_mut() {
            f(&format!("dec.layer.{li}"), &x);
        }
    }
    cache.len = past + n;
    let rows = match want {
        Logits::None => return Vec::new(),
        Logits::Last => n - 1..n,
        Logits::Tail(t) => n - t.min(n)..n,
        Logits::All => 0..n,
    };
    let hdim = dec.hidden;
    let mut y = x[rows.start * hdim..rows.end * hdim].to_vec();
    ops::rms_norm(isa, &mut y, &dec.norm, dec.eps);
    if let Some(f) = tap.as_mut() {
        f("dec.norm", &y);
    }
    let m = rows.len();
    let mut logits = vec![0f32; m * dec.vocab];
    matmul(pool, isa, &dec.head, &y, m, &mut logits);
    logits
}

/// The index of the largest logit (the first, on a tie).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}
