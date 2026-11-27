//! The AuT audio encoder's graph, ours (spec speech/01 §2, ADR 0068):
//! per 1 s chunk the three stride-2 convolutions with GELU,
//! `conv_out` and the chunk's sinusoidal positions; then, per 8 s window of up
//! to 104 tokens, the pre-LN layers with bidirectional attention inside the
//! window; the final norm and the two-layer projector.
//!
//! The checkpoint's encoder attends only inside a window and runs the
//! convolutions per chunk, so encoding a clip window by window gives the
//! whole-clip encode.

use crate::cpu::Isa;
use crate::matmul::matmul;
use crate::mel::{chunk_tokens, CHUNK_FRAMES, CHUNK_TOKENS, N_MELS};
use crate::model::{Encoder, Linear};
use crate::ops::{self, AttnShape, Kv, Mask};
use crate::pool::Pool;

/// Called with a tensor's name and its rows: the oracle's taps.
pub type Tap<'a> = &'a mut dyn FnMut(&str, &[f32]);

/// `y = x · wᵀ + b` for `m` rows.
pub(crate) fn linear(pool: &Pool, isa: Isa, l: &Linear, x: &[f32], m: usize) -> Vec<f32> {
    let mut y = vec![0f32; m * l.w.n];
    matmul(pool, isa, &l.w, x, m, &mut y);
    if let Some(b) = &l.b {
        ops::add_bias(isa, &mut y, b);
    }
    y
}

/// Encodes one window: `mel` holds `frames` normalised frames as rows of 128 (at most 800),
/// the last chunk possibly partial. Returns the projected tokens, rows of `out_dim`.
pub fn encode_window(e: &Encoder, pool: &Pool, isa: Isa, mel: &[f32], frames: usize, mut tap: Option<Tap>) -> Vec<f32> {
    assert!(frames > 0 && mel.len() >= frames * N_MELS);
    let ch = e.conv_channels;
    let bins3 = N_MELS.div_ceil(8);
    let mut stem_in: Vec<f32> = Vec::new();
    let mut positions: Vec<usize> = Vec::new();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for c in 0..frames.div_ceil(CHUNK_FRAMES) {
        let valid = (frames - c * CHUNK_FRAMES).min(CHUNK_FRAMES);
        // [mel bin][frame], zero past the chunk's valid frames (the processor's padding).
        let mut plane = vec![0f32; N_MELS * CHUNK_FRAMES];
        for f in 0..valid {
            for (bin, v) in mel[(c * CHUNK_FRAMES + f) * N_MELS..][..N_MELS].iter().enumerate() {
                plane[bin * CHUNK_FRAMES + f] = *v;
            }
        }
        let conv = &e.conv;
        ops::conv3x3s2_gelu(pool, isa, &conv[0].w, conv[0].b.as_deref().unwrap_or(&[]), &plane, 1, N_MELS, CHUNK_FRAMES, &mut a);
        ops::conv3x3s2_gelu(pool, isa, &conv[1].w, conv[1].b.as_deref().unwrap_or(&[]), &a, ch, N_MELS.div_ceil(2), CHUNK_FRAMES.div_ceil(2), &mut b);
        ops::conv3x3s2_gelu(pool, isa, &conv[2].w, conv[2].b.as_deref().unwrap_or(&[]), &b, ch, N_MELS.div_ceil(4), CHUNK_FRAMES.div_ceil(4), &mut a);
        // a is [channel][bin][13]: each kept time step becomes a row of [channel][bin].
        for t in 0..chunk_tokens(valid) {
            for k in 0..ch * bins3 {
                stem_in.push(a[k * CHUNK_TOKENS + t]);
            }
            positions.push(t);
        }
    }
    let n = positions.len();
    let d = e.d;
    let mut h = vec![0f32; n * d];
    matmul(pool, isa, &e.conv_out, &stem_in, n, &mut h);
    for (r, &t) in positions.iter().enumerate() {
        ops::add(isa, &mut h[r * d..(r + 1) * d], &e.pos[t * d..(t + 1) * d]);
    }
    if let Some(f) = tap.as_mut() {
        f("enc.stem", &h);
    }
    let shape = AttnShape { hq: e.heads, hkv: e.heads, hd: d / e.heads };
    let mut att = vec![0f32; n * d];
    for (i, l) in e.layers.iter().enumerate() {
        let mut x = h.clone();
        ops::layer_norm(isa, &mut x, &l.ln1.0, &l.ln1.1, e.eps);
        let q = linear(pool, isa, &l.q, &x, n);
        let k = linear(pool, isa, &l.k, &x, n);
        let v = linear(pool, isa, &l.v, &x, n);
        ops::attention(pool, isa, shape, &q, n, Kv::F32 { k: &k, v: &v }, n, Mask::Full, &mut att);
        let o = linear(pool, isa, &l.o, &att, n);
        ops::add(isa, &mut h, &o);
        let mut x = h.clone();
        ops::layer_norm(isa, &mut x, &l.ln2.0, &l.ln2.1, e.eps);
        let mut f = linear(pool, isa, &l.fc1, &x, n);
        ops::gelu(isa, &mut f);
        let y = linear(pool, isa, &l.fc2, &f, n);
        ops::add(isa, &mut h, &y);
        if let Some(f) = tap.as_mut() {
            f(&format!("enc.layer.{i}"), &h);
        }
    }
    ops::layer_norm(isa, &mut h, &e.post_ln.0, &e.post_ln.1, e.eps);
    if let Some(f) = tap.as_mut() {
        f("enc.post_ln", &h);
    }
    let mut p = linear(pool, isa, &e.proj1, &h, n);
    ops::gelu(isa, &mut p);
    let out = linear(pool, isa, &e.proj2, &p, n);
    if let Some(f) = tap.as_mut() {
        f("proj", &out);
    }
    out
}
