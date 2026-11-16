//! The 27B's MTP head held to its host reference (#620): BF16's unit
//! roundoff, the roundings on the device's path to each quantity a GPU test
//! reads, and the relative L2 a quantity after them stays within. One copy for
//! the head's GPU tests (`mtp_head_gpu.rs`, `mtp_round_gpu.rs`,
//! `mtp_vision_gpu.rs`).

// Each including test takes the bounds it needs.
#![allow(dead_code)]

/// BF16's unit roundoff: an 8-bit significand rounds to the nearest value,
/// within half an ulp, at most 2^-8 of the value.
pub const BF16_UNIT: f64 = 1.0 / 256.0;

/// The BF16 roundings on the device's path from the head's inputs to each
/// quantity read (kernel/src/mtp_head.cu): the stem's normed embedding and
/// hidden, its projection, the layer's input norm and fused projection, then
/// the key's norm and rotation; the value is the projection's. The head's
/// hidden adds the query's norm and rotation, the attention, the output
/// projection and its residual add, the MLP's input norm, gate and up, the
/// SwiGLU, down and its residual add, and the final norm; the scores add the
/// output head's.
pub const KEY_ROUNDINGS: f64 = 6.0;
pub const VALUE_ROUNDINGS: f64 = 5.0;
pub const HIDDEN_ROUNDINGS: f64 = 16.0;
pub const SCORE_ROUNDINGS: f64 = 17.0;

/// The relative L2 a quantity after `roundings` BF16 roundings stays within:
/// independent roundings, each at most the unit roundoff, compound in
/// quadrature, and the bound takes each at its largest (the RMS of one is a
/// third of that squared), which leaves the norms and the softmax, whose
/// gains are near one, room. A wrong row, position, token, concatenation
/// order or gain lands far above it: those move the result by its own size.
pub fn tolerance(roundings: f64) -> f64 {
    BF16_UNIT * roundings.sqrt()
}
