//! The production [`Compute`] backend (kernel-abi 04, the compute-adapter).
//!
//! A topology-driven forward pass that composes the kernel-leaf C-ABI
//! primitives (ADR 0001) into the engine's `prefill_step` / `decode_step`
//! (the [`Compute`] seam, `scheduler.rs`). The heavy ops run on the GPU via
//! the FFI (NVFP4 GEMM/GEMV, GQA attention, GDN step, the GDN causal conv,
//! the GQA RoPE / q / k RMSNorm, the bf16 GEMM, RMSNorm, embedding, greedy
//! sample, the CUDA-graph primitives); the pointwise glue (residual add,
//! the gated-SiLU activation, the gate·up multiply, the GDN state readout)
//! runs on the host as the **correctness floor** (ADR 0005: correctness is
//! the non-negotiable floor; the fused-SiLU / fused-residual / fused-
//! readout kernels are the later 99%-gate performance material, ADR 0007 /
//! bench-03).
//!
//! **Topology-driven:** the forward pass is parameterized by a
//! [`ModelConfig`] (layer count, per-layer kind (GQA / GDN), head geometry,
//! GDN state dims, the GDN feature layout (q / z / a-b widths), the rotary
//! geometry, FFN width, vocab, block geometry). The [`Weights`] hold the
//! model's weights in the kernel-expected formats (bf16 activations, NVFP4
//! E2M1 codes + E4M3 scales for the GEMM weights, the bf16 exception
//! tensors, the device-resident NVFP4 planes for the `from_artifact`
//! routing), so the same code serves a *synthetic* (test) model and a real
//! (artifact) model.
//!
//! **The full-correct forward assembly (A3 / #30, spec 07):** the real
//! (artifact) model's forward pass runs the *full* layer stack — the GDN
//! layers' causal conv (`ignis_gdn_causal_conv`, kernel-abi 06, A2 / #28)
//! + the GDN a / b (gate / beta) projection + the GDN step + the state
//! readout (the "for now" host-side `S^T k` GEMV, ADR 0005) + the z
//! (output-gate) gating; the GQA layers' QKV projection + the q / k
//! RMSNorm (`ignis_rmsnorm`, the per-head) + the RoPE (`ignis_rope_qk`,
//! kernel-abi 06) + the GQA attention + the output projection; the
//! gated-FFN block (the NVFP4 GEMMs + the host gated-SiLU); the bf16
//! logits GEMM (`ignis_bf16_gemm`, kernel-abi 10, A2b / #29) for the
//! W8-dequantized lm_head; and the real `qwen38_27b` topology + the
//! device-resident NVFP4 routing (the artifact's normalized tensors, A1 /
//! #27 — the NVFP4 fused planes stay in VRAM (the `*_device` kernels, no
//! per-call H2D, the #26 fix), the BF16 tensors are host-copied (the
//! bounded text-scope copy), the W8 endpoints are the A1 host-side
//! dequants).
//!
//! **The CUDA-graph fast path (kernel-abi 03):** at construction the
//! kernel-leaf startup check (`ignis_graph_startup_check` — a few KB of
//! VRAM, runs even with a model loaded, ADR 0006 nuance) runs, and a
//! representative decode graph is captured (the `ignis_graph_begin_capture`
//! / `ignis_graph_end_capture` primitives) as the eager-sequence warm-up.
//! The graph **launch** (`ignis_graph_launch` per decode step) is the
//! performance material (the 99% gate, ADR 0007 / bench-03) — **not
//! implemented in this ticket** (the eager sequence is always used; the
//! graph is captured but never launched, the documented follow-up, B2 /
//! #32).
//!
//! **The decode query (the autoregressive decode, A3 / #30):** the decode
//! step threads the actually-generated token back into the next step (the
//! prefill's last prompt token on the first decode, the previous decode's
//! token thereafter) — the real-model autoregressive wiring (a fresh
//! request without a prefill uses token 0).
//!
//! **The batched prefill (the multi-token forward path, B1 / #31):** the
//! `prefill` seam's dispatch — a `seq > 1` fresh-prompt prefill runs the
//! layer stack in one multi-token pass (`prefill_batched` +
//! `forward_layers_multi`: the multi-token GEMM, kernel-abi 05, + the
//! multi-token attention, kernel-abi 01, + the per-token GDN recurrence
//! / RoPE / KV writeback); `seq == 1` (the GEMV special case, ADR 0001)
//! and a warm-KV (prefix-reuse tail) prefill keep the per-token loop
//! (the ADR 0003 eager fallback — a busy/absent multi-token kernel also
//! falls back to the per-token loop after the fresh state is restored).
//! The batched path's accumulation order differs from the per-token loop
//! (spec 08's design §7 caveat), so the acceptance is a *sane*,
//! reproducible output (the correctness floor, ADR 0005 / 0007) — not a
//! bit-exact agreement with the per-token loop (the 99% gate, #20, is
//! the re-check).
//!
//! **Documented gaps (the 99%-gate performance material, ADR 0005 /
//! 0007 / bench-03 — not the correctness floor):** the CUDA-graph
//! **launch** (the per-decode-step replay, B2 / #32), and the
//! *re-implementation* of the "for now" ported kernels (the tensor-core
//! NVFP4 / bf16 GEMMs, the fused qk-norm+RoPE, the fused readout, the
//! fused multi-token norms — the later performance gate, the 99%
//! material, #20). The correctness floor (the full-correct forward pass —
//! a *sane*, reproducible output, ADR 0005 / 0007) is complete in this
//! module.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::ffi;
use crate::scheduler::{Compute, DecodeJob, PrefillJob};
use crate::types::{ComputeError, RequestId, TokenId};

// The A1 normalization's host-side dequant endpoint (the `from_artifact`'s
// W8 -> bf16, the `Weights::with_w8_endpoints` seam; the artifact crate is a
// non-optional dep, so `W8Endpoints` is available in the CPU build too).
use ignis_artifact::W8Endpoints;

// The artifact's device context types (the `from_artifact` path, the #26
// fix). `CudaDevice` is `cuda`-feature-gated (the VRAM arena owner, ADR 0002);
// all are only referenced by the `cuda`-gated `DeviceCtx::Cuda` variant
// + `from_artifact`, so the imports are `cuda`-gated (unused in the CPU build).
#[cfg(feature = "cuda")]
use ignis_artifact::{dequant_w8_endpoints, CudaDevice, MaterializedArtifact, Reader};

// ---------------------------------------------------------------------------
// bf16 helpers (round-to-nearest-even, matching the kernel's __float2bfloat16_rn)
// ---------------------------------------------------------------------------

/// Encode an f32 as a bf16 (16-bit) value (round-to-nearest-even into 16 bits).
#[inline]
fn f32_to_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    let lsb = ((b >> 16) & 1) as u32;
    ((b + 0x7fff + lsb) >> 16) as u16
}

/// Decode a bf16 (16-bit) value to f32.
#[inline]
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// Build a bf16 host buffer from f32 values.
fn to_bf16(values: &[f32]) -> Vec<u16> {
    values.iter().map(|&v| f32_to_bf16(v)).collect()
}

/// A bf16 buffer -> an f32 buffer (the logits GEMM output -> the greedy
/// sample's f32 logits, the kernel-abi 02 contract).
fn bf16_to_f32s(values: &[u16]) -> Vec<f32> {
    values.iter().map(|&v| bf16_to_f32(v)).collect()
}

/// The RoPE (split-half NeoX) inverse-frequency table (kernel-abi 06,
/// GitHub #28): `inv_freq[pair] = θ^(-2·pair/rotary_dim)` (pair ∈
/// `[0, rotary_dim/2)`), computed in f64 and rounded to f32 (the
/// reference's `rope_linear_frequencies` table, `ops/wrapper/rope.cpp`).
///
/// Host-side, deterministic, and computed **once at construction** (the
/// v1 table: θ = 1e7, rotary_dim = 64 — the Qwen 3.8-27B GQA geometry);
/// it is uploaded once per `ignis_rope_qk` call (a non-goal is the
/// per-step table recompute). The caller (the A3 forward assembly) builds
/// it and passes it to the kernel (the kernel consumes the f32 table and
/// never recomputes it).
pub fn rope_inv_frequencies(theta: f64, rotary_dim: i64) -> Vec<f32> {
    assert!(
        theta > 0.0 && theta.is_finite(),
        "rope_inv_frequencies: theta must be positive and finite"
    );
    assert!(
        rotary_dim > 0 && rotary_dim % 2 == 0,
        "rope_inv_frequencies: rotary_dim must be a positive even value"
    );
    let pairs = (rotary_dim / 2) as usize;
    (0..pairs)
        .map(|pair| {
            theta
                .powf(-2.0 * (pair as f64) / (rotary_dim as f64))
                as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Topology (ADR 0001: the model config the forward pass is parameterized by)
// ---------------------------------------------------------------------------

/// The kind of attention a decoder layer uses (the Qwen 3.8-27B hybrid is a
/// GQA + GDN (Gated DeltaNet linear-attention) mix).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// A standard GQA (grouped-query attention) layer.
    Gqa,
    /// A GDN (Gated DeltaNet linear-attention, recurrent-state) layer.
    Gdn,
}

/// The model topology the forward pass is parameterized by (ADR 0001).
///
/// A real (artifact) model's config is derived from the container's tensor
/// directory (the per-layer shapes); a synthetic (test) model uses
/// [`ModelConfig::synthetic`].
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// The number of decoder layers (in order).
    pub num_layers: usize,
    /// Each layer's kind (GQA / GDN). Must have length `num_layers`.
    pub layer_kinds: Vec<LayerKind>,
    /// The residual-stream width (hidden dim).
    pub hidden: u64,
    /// The vocabulary size (the `lm_head` output dim; token ids in
    /// `[0, vocab)`).
    pub vocab: u64,
    /// GQA query heads.
    pub num_q_heads: u64,
    /// GQA KV heads (`num_q_heads` is a multiple of this — the GQA group).
    pub num_kv_heads: u64,
    /// Per-head dim.
    pub head_dim: u64,
    /// GDN recurrent-state row dim (d_v).
    pub gdn_state_rows: u64,
    /// GDN recurrent-state column dim (d_k). The GDN step's feature dim is
    /// `state_cols + state_rows + 2` (k, v, gate, beta — `gdn_step.cuh`).
    pub gdn_state_cols: u64,
    /// The GDN step's `num_gdn_layers` argument (per-layer GDN state count).
    pub gdn_num_layers: u64,
    /// The GDN input-projection `q` rows (the GDN feature's query part,
    /// before the k / v parts — `0` for a model without a separate GDN q
    /// part; the Qwen 3.8-27B real model's `gdn/query_key_value_z` is
    /// q 2048 + k 2048 + v 6144 + z 6144 = 16 384 rows, A3 / #30).
    pub gdn_q_width: u64,
    /// The GDN input-projection `z` (output-gate) rows (the rows that
    /// bypass the causal conv and gate the state readout; `0` for a model
    /// without a z part).
    pub gdn_z_width: u64,
    /// The GDN a/b (gate / beta) projection width (`gdn/a_b_projection`,
    /// a bf16 GEMM: the first half is the gate `a`, the second half the
    /// beta `b` — `0` = no a/b projection, the step's g / beta are 0;
    /// the Qwen 3.8-27B real model is 96 = 48 a + 48 b, A3 / #30).
    pub gdn_ab_width: u64,
    /// The GQA RoPE rotary dim (of `head_dim` — the first `rotary_dim`
    /// dims of each q / k head are rotated; `rotary_dim / 2` pairs,
    /// kernel-abi 06, GitHub #28).
    pub rotary_dim: u64,
    /// The RoPE base θ (the `inv_freq[pair] = θ^(-2·pair/rotary_dim)`
    /// table, kernel-abi 06 — the Qwen 3.8-27B GQA geometry θ = 1e7).
    pub rope_theta: f64,
    /// The FFN (gated-SiLU) intermediate width.
    pub ffn_intermediate: u64,
    /// The paged KV block size (keys per block).
    pub block_size: u64,
    /// The paged KV block count per request (capacity = block_size *
    /// num_blocks keys).
    pub num_blocks: u64,
}

impl ModelConfig {
    /// The GQA query/output width (`num_q_heads * head_dim`).
    pub fn gqa_width(&self) -> u64 {
        self.num_q_heads * self.head_dim
    }

    /// The GQA key/value width (`num_kv_heads * head_dim`).
    pub fn gqa_kv_width(&self) -> u64 {
        self.num_kv_heads * self.head_dim
    }

    /// The GDN step's feature dim (`state_cols + state_rows + 2`).
    pub fn gdn_state_dim(&self) -> u64 {
        self.gdn_state_cols + self.gdn_state_rows + 2
    }

    /// The GDN state matrix width (`state_rows * state_cols`).
    pub fn gdn_state_mat(&self) -> u64 {
        self.gdn_state_rows * self.gdn_state_cols
    }

    /// The GDN input-projection GEMM width (`m` = the GDN feature rows:
    /// the q / k / v / z parts — `gdn_q_width + state_cols + state_rows +
    /// gdn_z_width`; the artifact's `gdn/query_key_value_z` tensor is
    /// exactly this wide, A3 / #30).
    pub fn gdn_in_proj_m(&self) -> u64 {
        self.gdn_q_width + self.gdn_state_cols + self.gdn_state_rows + self.gdn_z_width
    }

    /// The GDN causal-conv channel count (the conv'd q / k / v part of the
    /// input projection — `gdn_q_width + state_cols + state_rows`; the
    /// z rows bypass the conv, the `ignis_gdn_causal_conv` contract,
    /// kernel-abi 06, A3 / #30).
    pub fn gdn_conv_channels(&self) -> u64 {
        self.gdn_q_width + self.gdn_state_cols + self.gdn_state_rows
    }

    /// The GDN state readout GEMM input dim (`k` = the per-token readout
    /// width `state_rows` — the artifact's `gdn/output` tensor is
    /// `[hidden][state_rows]`, A3 / #30).
    pub fn gdn_readout_k(&self) -> u64 {
        self.gdn_state_rows
    }

    /// The GQA RoPE inverse-frequency pair count (`rotary_dim / 2`).
    pub fn rope_pairs(&self) -> u64 {
        self.rotary_dim / 2
    }

    /// A small, fast synthetic topology for the self-consistency GPU test
    /// (one GDN + one GQA layer, small dims, a small paged KV) — exercises
    /// every kernel primitive (embedding, GEMM/GEMV, GQA, GDN step, norms,
    /// sample, the CUDA-graph primitives) with a deterministic synthetic
    /// model. All NVFP4 GEMM shapes satisfy the kernel's `k % 16 == 0`
    /// group-scale validation (kernel-abi 05: the GDN readout GEMM's
    /// `k` = the readout width `state_rows` is a multiple of 16 — a
    /// `k` that violates it is rejected by the kernel *before* any CUDA
    /// call, so the synthetic forward pass would fault on it, B1 / #31).
    pub fn synthetic() -> Self {
        Self {
            num_layers: 2,
            layer_kinds: vec![LayerKind::Gdn, LayerKind::Gqa],
            hidden: 64,
            vocab: 256,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            // A multiple of 16 (the NVFP4 GEMM's `k` group-scale
            // validation — the GDN readout GEMM's `k` = `state_rows`,
            // kernel-abi 05 — the synthetic forward pass's kernels all
            // pass validation, B1 / #31).
            gdn_state_rows: 16,
            gdn_state_cols: 8,
            gdn_num_layers: 1,
            // The synthetic model has no separate GDN q / z / a-b parts
            // (the input projection is the k / v / g / beta feature
            // directly — the `gdn_state_dim` layout, A3 / #30).
            gdn_q_width: 0,
            gdn_z_width: 0,
            gdn_ab_width: 0,
            // The RoPE geometry: `rotary_dim` 8 of `head_dim` 16 (4
            // pairs), θ = 1e7 (the real-model base, a deterministic
            // synthetic table, A3 / #30).
            rotary_dim: 8,
            rope_theta: 1e7,
            ffn_intermediate: 32,
            block_size: 4,
            num_blocks: 8,
        }
    }

    /// The real Qwen 3.8-27B topology (the v1 specialization, CONTEXT.md: one
    /// model family, one GPU class). This is the #26 crash fix — the
    /// `from_artifact` path uses it instead of [`ModelConfig::synthetic`],
    /// so the embedding table has the real vocab (248 320) and a real
    /// tokenizer's ids (up to 248 077) never index out of bounds (the
    /// `ignis_embedding` OOB that produced the `illegal memory access`).
    /// The layer pattern + head geometry are the model constants (ignis is
    /// specialized for Qwen 3.8-27B); the full-correct device routing
    /// (the W8 / BF16-exception dequants, the GDN causal conv / RoPE /
    /// q / k RMSNorm ops, the bf16 logits GEMM, the per-layer tensor
    /// routing) is the A3 / #30 assembly (spec 07) — the forward pass
    /// runs the *real* model (the correctness floor, ADR 0005: a *sane*,
    /// reproducible output — the 99%-gate performance material is #20,
    /// not this ticket).
    pub fn qwen38_27b() -> Self {
        let num_layers: usize = 64;
        // Layer `i` is GQA (full attention) exactly when `(i + 1) % 4 == 0`
        // (16 GQA + 48 GDN linear-attention layers).
        let layer_kinds: Vec<LayerKind> = (0..num_layers)
            .map(|i| if (i + 1) % 4 == 0 { LayerKind::Gqa } else { LayerKind::Gdn })
            .collect();
        Self {
            num_layers,
            layer_kinds,
            hidden: 5120,
            vocab: 248_320,
            num_q_heads: 24,
            num_kv_heads: 4,
            head_dim: 256,
            // GDN recurrent state: 48 V heads x 128 (rows) by 16 Q/K heads x 128
            // (cols). `gdn_num_layers` = the 48 GDN layers.
            gdn_state_rows: 6144,
            gdn_state_cols: 2048,
            gdn_num_layers: 48,
            // The GDN input projection's `gdn/query_key_value_z` layout
            // (the artifact's directory, A1 inventory, A3 / #30): q 2048
            // + k 2048 + v 6144 + z 6144 = 16 384 rows — the causal conv
            // covers the first 10 240 channels (q / k / v), the z rows
            // bypass it. The a/b (gate / beta) projection is 96 rows
            // (`gdn/a_b_projection`: 48 gate + 48 beta, one per v-head).
            gdn_q_width: 2048,
            gdn_z_width: 6144,
            gdn_ab_width: 96,
            // The GQA RoPE geometry (kernel-abi 06, GitHub #28): the
            // split-half NeoX rotary of `rotary_dim` = 64 of
            // `head_dim` = 256 (32 pairs), base θ = 1e7 (the reference's
            // `rope_linear_frequencies` table).
            rotary_dim: 64,
            rope_theta: 1e7,
            ffn_intermediate: 17_408,
            // Paged KV: 64-token pages (the reference P=64 granularity); 4096
            // blocks per request (the 262k context envelope, design §2).
            block_size: 64,
            num_blocks: 4096,
        }
    }
}

// ---------------------------------------------------------------------------
// Weights (host buffers in the kernel-expected formats)
// ---------------------------------------------------------------------------

/// A single NVFP4 GEMM weight: `out = act @ W^T`, where `W` is NVFP4-
/// quantized (E2M1 codes, 2 packed per byte; E4M3 scale per 16-element
/// group). `m` = output rows, `k` = input dim (a multiple of 16).
#[derive(Debug, Clone)]
pub struct Nvfp4Weight {
    /// E2M1 (FP4) weight codes, `[m][k/2]` bytes (2 codes per byte).
    pub codes: Vec<u8>,
    /// E4M3 (FP8) per-group scales, `[m][k/16]` bytes.
    pub scales: Vec<u8>,
    /// The output dim `m` (rows of W).
    pub m: u64,
    /// The input dim `k` (columns of W; a multiple of 16).
    pub k: u64,
}

impl Nvfp4Weight {
    /// A zero-sized (unused) weight (a GDN layer's unused GQA projections).
    pub fn empty() -> Self {
        Self {
            m: 0,
            k: 0,
            codes: Vec::new(),
            scales: Vec::new(),
        }
    }

    /// A geometry-only NVFP4 GEMM weight (the #27 A1 normalization seam):
    /// the real (m, k) with empty code/scale planes — the content is the
    /// artifact's normalized NVFP4 buffers (the device-resident
    /// materialization, ADR 0002), consumed by the forward pass (A3).
    pub fn geometry_only(m: u64, k: u64) -> Self {
        Self {
            m,
            k,
            codes: Vec::new(),
            scales: Vec::new(),
        }
    }
}

/// The logits GEMM weight (the model's `lm_head`): either a preserved NVFP4
/// GEMM weight (the synthetic path, the `ignis_nvfp4_gemm_*` kernels — the
/// content is the device-resident materialization, ADR 0002) or a dequantized
/// bf16 weight (the artifact path: the `text/output_head` W8G32 dequantized to
/// bf16, ADR 0005 host-side dequant — the correctness floor; the fused bf16
/// GEMM kernel is kernel-abi 10, the later performance material). The
/// (m, k) follows the kernel's convention (m = output dim = vocab, k = input
/// dim = hidden), per the `Nvfp4Weight` doc and the `ignis_nvfp4_gemm_*` FFI
/// (out[tokens][m] = sum_k act[tokens][k] * W[m][k]).
#[derive(Debug, Clone)]
pub enum HeadWeight {
    /// A preserved NVFP4 GEMM weight (the synthetic / device-resident path;
    /// the `ignis_nvfp4_gemm_*` kernels consume the codes + scales).
    Nvfp4(Nvfp4Weight),
    /// A dequantized bf16 GEMM weight (the artifact's W8 `text/output_head`,
    /// ADR 0005 host-side dequant). `data` is `[m][k]` bf16 (row-major),
    /// `m` = output dim (vocab), `k` = input dim (hidden).
    DequantBf16 {
        data: Vec<u16>,
        m: u64,
        k: u64,
    },
}

impl HeadWeight {
    /// The GEMM weight's (m, k) (the output rows, the input dim), for every
    /// variant (`(0, 0)` is unused here — both variants carry the real (m, k)).
    pub fn dims(&self) -> (u64, u64) {
        match self {
            HeadWeight::Nvfp4(w) => (w.m, w.k),
            HeadWeight::DequantBf16 { m, k, .. } => (*m, *k),
        }
    }

    /// The NVFP4 weight, if this is the (preserved / device-resident) NVFP4
    /// variant (the synthetic / geometry-only path).
    pub fn as_nvfp4(&self) -> Option<&Nvfp4Weight> {
        match self {
            HeadWeight::Nvfp4(w) => Some(w),
            HeadWeight::DequantBf16 { .. } => None,
        }
    }
}

/// A deterministic synthetic NVFP4 weight of shape `[m][k]` (a fixed pattern
/// of E2M1 codes + E4M3 scales from `seed`). `k` is a multiple of 16.
fn nvfp4_weight(m: u64, k: u64, seed: u64) -> Nvfp4Weight {
    if m == 0 || k == 0 {
        return Nvfp4Weight::empty();
    }
    let m = m as usize;
    let k = k as usize;
    let code_row = k / 2;
    let scale_row = k / 16;
    let codes: Vec<u8> = (0..m * code_row)
        .map(|i| {
            let mi = i / code_row;
            let b = i % code_row;
            // A fixed, seed-deterministic E2M1 code pattern (values in
            // [0, 8) map to the E2M1 magnitude set {0,.5,1,1.5,2,3,4,6}).
            let lo = ((mi.wrapping_mul(3) + b + (seed as usize) % 7) % 8) as u8;
            let hi = ((mi.wrapping_mul(5) + b + (seed as usize) % 5) % 8) as u8;
            lo | (hi << 4)
        })
        .collect();
    let scales: Vec<u8> = (0..m * scale_row).map(|_| 0x38u8).collect(); // E4M3 1.0
    Nvfp4Weight {
        m: m as u64,
        k: k as u64,
        codes,
        scales,
    }
}

/// A bf16 GEMM / pointwise weight: row-major `[m][k]` bf16 (m output rows,
/// k input dim). A1 preserves the artifact's BF16 tensors as-is (spec 04:
/// the `gdn/convolution`, the norms, the BF16-exception projections stay
/// contiguous bf16), so the host tier carries them as plain bf16 buffers
/// the `ignis_bf16_gemm` kernel (kernel-abi 10) + the
/// `ignis_gdn_causal_conv` kernel (kernel-abi 06) consume directly
/// (A3 / #30): the artifact's BF16 tensors (the early GQA layers'
/// `attention/query_key_gate_value` + `attention/output`, the layer-4
/// `gdn/output`, every GDN layer's `gdn/a_b_projection` +
/// `gdn/convolution`) are host-copied on the `from_artifact` path (the
/// bounded text-scope copy — the #26 lesson is the unbounded NVFP4 host
/// generation, not the bf16 endpoints).
#[derive(Debug, Clone)]
pub struct Bf16Weight {
    /// The bf16 content, `[m][k]` row-major.
    pub data: Vec<u16>,
    /// The output dim `m` (rows of W).
    pub m: u64,
    /// The input dim `k` (columns of W).
    pub k: u64,
}

impl Bf16Weight {
    /// A zero-sized (unused) bf16 weight.
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            m: 0,
            k: 0,
        }
    }

    /// A geometry-only bf16 weight (the `from_geometry`'s real (m, k) with
    /// empty content — the content is the artifact's bf16 buffer, consumed
    /// by the forward pass, A3 / #30).
    pub fn geometry_only(m: u64, k: u64) -> Self {
        Self {
            data: Vec::new(),
            m,
            k,
        }
    }

    /// A deterministic synthetic bf16 weight (a pure function of `seed` +
    /// the geometry). Values are bf16-exact multiples of 1/8 within
    /// `[-1, 1)` (a bounded, numerically-sane pattern, like the
    /// `nvfp4_weight` helper).
    fn bf16_weight(m: u64, k: u64, seed: u64) -> Self {
        if m == 0 || k == 0 {
            return Self::empty();
        }
        let m = m as usize;
        let k = k as usize;
        let data: Vec<u16> = (0..m * k)
            .map(|i| {
                let r = i / k;
                let c = i % k;
                // A bounded deterministic value in [-1.0, 1.0)
                // (bf16-exact multiples of 1/8).
                let v = ((r as u64)
                    .wrapping_mul(7)
                    .wrapping_add((c as u64).wrapping_mul(3))
                    .wrapping_add(seed))
                    % 16;
                f32_to_bf16(v as f32 / 8.0 - 1.0)
            })
            .collect();
        Self {
            data,
            m: m as u64,
            k: k as u64,
        }
    }
}

/// A device-resident NVFP4 GEMM plane (the artifact's materialized arena,
/// ADR 0002): the raw pointers into the `CudaDevice` arena the
/// `ignis_nvfp4_gemm_*_device` kernels (ticket 26, GitHub #26) consume
/// (no per-call weight H2D, the #26 fix). Set on the `from_artifact` path
/// (A3 / #30); empty on the synthetic / dev path (the host `Nvfp4Weight`
/// carries the content there). The plane is the artifact's *fused* tensor
/// (`attention/query_key_gate_value`, `gdn/query_key_value_z`,
/// `mlp/gate_up`, …) — the per-slot GEMM (q / k / v / gate / up) is a row
/// slice of it (the GEMM dispatch's `row_off` argument; the slot's row
/// start within the fused tensor).
///
/// The plane's code / scale layout is the artifact's (the materialized
/// payload's planes, ADR 0002); the ported GEMM kernels read them as
/// plain row-major `[m][k/2]` / `[m][k/16]` planes (the "for now"
/// starting point, ADR 0005 — the container plane layout + the
/// `weight_divisor` application are the later re-implementation
/// material, not this assembly's).
#[derive(Debug, Clone, Copy)]
pub struct Nvfp4DevicePlane {
    /// The E2M1 codes plane (device, `[m][k/2]` bytes, 2 codes per byte).
    pub codes: *const u8,
    /// The E4M3 group-scales plane (device, `[m][k/16]` bytes).
    pub scales: *const u8,
    /// The plane's output rows `m` (the fused tensor's rows).
    pub m: u64,
    /// The plane's input dim `k`.
    pub k: u64,
}

// SAFETY: an `Nvfp4DevicePlane` is a pair of raw pointers into the
// artifact's VRAM arena (the `CudaDevice` owner, the `DeviceCtx` —
// dropped last in the `CudaCompute`, after the `Weights` that hold the
// planes). Sharing the planes across threads is safe: the forward pass
// reads the planes (read-only); the producing arena outlives the views
// (the `DeviceCtx` drop order, ADR 0002).
unsafe impl Send for Nvfp4DevicePlane {}
unsafe impl Sync for Nvfp4DevicePlane {}

/// The per-layer device-resident NVFP4 planes (the `from_artifact` path's
/// routing table, A3 / #30): which artifact tensor each GEMM slot
/// consumes (a plane + the slot's row slice within it). A `None` slot
/// falls back to the host `Nvfp4Weight` (the synthetic / dev path) or
/// the layer's bf16 exception weight (the artifact's BF16-exception
/// layers).
#[derive(Debug, Clone, Copy, Default)]
pub struct LayerDeviceSlots {
    /// The GQA fused qkvz projection (`attention/query_key_gate_value`;
    /// the q / k / v slots are row slices of it).
    pub qkv: Option<Nvfp4DevicePlane>,
    /// The GQA output projection (`attention/output`).
    pub attn_out: Option<Nvfp4DevicePlane>,
    /// The GDN input projection (`gdn/query_key_value_z`).
    pub gdn_in: Option<Nvfp4DevicePlane>,
    /// The GDN state readout (`gdn/output`).
    pub gdn_out: Option<Nvfp4DevicePlane>,
    /// The fused FFN gate+up projection (`mlp/gate_up`; the gate / up
    /// slots are row slices of it).
    pub mlp_gate_up: Option<Nvfp4DevicePlane>,
    /// The FFN down projection (`mlp/down`).
    pub mlp_down: Option<Nvfp4DevicePlane>,
}

/// A decoder layer's GEMM weights (the projections + the gated-FFN weights).
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// The layer's kind (GQA / GDN).
    pub kind: LayerKind,
    /// The attention/GDN projections (`[4]` NVFP4 GEMM weights):
    /// - GQA: `[0]` q, `[1]` k, `[2]` v, `[3]` attention-output.
    /// - GDN: `[0]` input (hidden -> GDN feature), the rest unused.
    pub projection: [Nvfp4Weight; 4],
    /// GDN only: the recurrent-state -> output projection
    /// (`[state_rows*state_cols] -> [hidden]`); unused for a GQA layer.
    pub gdn_output: Nvfp4Weight,
    /// The gated-FFN weights (the same for every layer kind): the gate + up
    /// projections (`[hidden] -> [ffn]` each) + the down projection
    /// (`[ffn] -> [hidden]`).
    pub ffn_gate: Nvfp4Weight,
    pub ffn_up: Nvfp4Weight,
    pub ffn_down: Nvfp4Weight,
    /// The pre-attention + post-attention RMSNorm weights: bf16 `[hidden]`.
    pub norm_in: Vec<u16>,
    pub norm_post: Vec<u16>,
    /// The GDN causal-conv weight (A3 / #30, kernel-abi 06): bf16
    /// `[4][conv_channels]` (the 4 taps w0..w3, tap-major — the
    /// artifact's `gdn/convolution` tensor, the GDN layers only; empty
    /// for a GQA layer / a model without a conv).
    pub gdn_conv: Bf16Weight,
    /// The GDN a/b (gate / beta) projection (A3 / #30): bf16
    /// `[gdn_ab_width][hidden]` (the artifact's `gdn/a_b_projection`;
    /// the first half is the gate `a`, the second the beta `b` — empty
    /// when the model has no a/b projection, `gdn_ab_width` = 0).
    pub gdn_ab: Bf16Weight,
    /// The GQA q / k RMSNorm weights (A3 / #30): bf16 `[head_dim]` each
    /// (the artifact's `attention/query_norm` + `attention/key_norm`,
    /// per-head RMSNorm weights — empty = a parameter-free RMSNorm, the
    /// synthetic / no-weight convention).
    pub qk_norm: [Vec<u16>; 2],
    /// The GQA fused qkvz projection's BF16-exception content (A3 / #30):
    /// the artifact's early GQA layers store `attention/query_key_gate_value`
    /// in bf16 (the A1 inventory's `QKGV_BF16_LAYERS`) — the q / k / v
    /// slots are row slices of this buffer (the `ignis_bf16_gemm`
    /// kernel, A2b). Empty on the synthetic / NVFP4 paths.
    pub qkv_bf16: Bf16Weight,
    /// The GQA output projection's BF16-exception content (the artifact's
    /// early GQA layers store `attention/output` in bf16 — the
    /// A1 inventory's `ATTENTION_OUT_BF16_LAYERS`). Empty otherwise.
    pub attn_out_bf16: Bf16Weight,
    /// The GDN state readout's BF16-exception content (the artifact's
    /// layer-4 `gdn/output` quirk, the A1 inventory's
    /// `GDN_OUT_BF16_LAYERS`). Empty otherwise.
    pub gdn_out_bf16: Bf16Weight,
    /// The device-resident NVFP4 planes (the `from_artifact` routing,
    /// A3 / #30 — set on the cuda path; `Default` on the synthetic / dev
    /// path, the host `Nvfp4Weight`s carry the content there).
    pub dev: LayerDeviceSlots,
    /// This layer's index within the model's GDN layers (the GDN
    /// recurrent-state + causal-conv-state slice, the request state's
    /// per-layer planes; `0` for a GQA layer).
    pub gdn_index: usize,
}

/// The model's weights, in the kernel-expected host formats. A synthetic
/// (test) model's weights are deterministic (a pure function of `seed` + the
/// geometry); a real (artifact) model's are dequantized from the container
/// (the `from_artifact` path).
#[derive(Debug, Clone)]
pub struct Weights {
    /// The token-embedding table: bf16 `[vocab][hidden]`
    /// (`ignis_embedding` expects bf16).
    pub embedding: Vec<u16>,
    /// Per-layer GEMM weights (in the `layer_kinds` order).
    pub per_layer: Vec<LayerWeights>,
    /// The final-norm weight: bf16 `[hidden]` (`ignis_rmsnorm`).
    pub final_norm: Vec<u16>,
    /// The lm_head weight (the logits GEMM): a preserved NVFP4 GEMM weight
    /// (the synthetic / device-resident path) or a dequantized bf16 weight
    /// (the artifact's W8 `text/output_head`, ADR 0005 host-side dequant).
    pub lm_head: HeadWeight,
}

/// A decoder layer's GEMM weight geometries (the (m, k) pairs the forward
/// pass's `nvfp4_gemm` consumes — the same derivation
/// [`Weights::synthetic`] uses for its content, minus the content).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerGeometry {
    /// The layer's kind (GQA / GDN).
    pub kind: LayerKind,
    /// The attention/GDN projections' (m, k) pairs (`[4]`: GQA's
    /// q / k / v / output projections; GDN's input projection + the
    /// three unused slots, (0, 0)).
    pub projection: [(u64, u64); 4],
    /// The GDN state readout (m, k) (the GDN layers only; (0, 0) for
    /// GQA layers).
    pub gdn_output: (u64, u64),
    /// The gated-FFN (m, k) pairs (the same for every layer kind): the
    /// gate + up projections (m = ffn, k = hidden) + the down projection
    /// (m = hidden, k = ffn).
    pub ffn_gate: (u64, u64),
    pub ffn_up: (u64, u64),
    pub ffn_down: (u64, u64),
    /// The pre-attention + post-attention norm sizes (bf16 `[hidden]`).
    pub norm_in: u64,
    pub norm_post: u64,
    /// The GDN causal-conv geometry (m = the 4 taps, k = the conv
    /// channels) — the GDN layers only; (0, 0) for a GQA layer
    /// (A3 / #30, kernel-abi 06).
    pub gdn_conv: (u64, u64),
    /// The GDN a/b (gate / beta) projection (m, k) — the GDN layers
    /// with a `gdn_ab_width` > 0; (0, 0) otherwise (A3 / #30).
    pub gdn_ab: (u64, u64),
    /// The GQA q / k RMSNorm weight width (bf16 `[head_dim]` each) — the
    /// GQA layers only; 0 for a GDN layer (A3 / #30).
    pub qk_norm: u64,
    /// The GDN causal-conv channel count (the conv'd q / k / v part —
    /// `gdn_q_width + state_cols + state_rows`, the z rows bypass it;
    /// 0 when there is no conv, A3 / #30).
    pub gdn_conv_channels: u64,
    /// The GDN input-projection (m, k) (the `gdn_in_proj_m` rows: the
    /// q / k / v / z parts — the artifact's `gdn/query_key_value_z`
    /// width, A3 / #30).
    pub gdn_in_proj: (u64, u64),
}

/// The kernel-expected weight geometry (sizes only, no content): the
/// `Weights` layout a [`ModelConfig`] demands — pure, config-derived,
/// CPU-testable (the #27 A1 normalization seam, spec 04).
///
/// It replaces the zero geometry of [`Weights::placeholder`] on the
/// `from_artifact` path (via [`Weights::from_geometry`]): the GEMM
/// `Nvfp4Weight`s carry the real (m, k) (the code/scale planes are the
/// artifact's normalized NVFP4 buffers — the device-resident
/// materialization, ADR 0002), and every field's size matches the 27B
/// topology (spec 04's acceptance: the `Weights` geometry matches
/// `ModelConfig::qwen38_27b`). The forward-pass *consumption* of the
/// normalized content (the W8 -> bf16 embedding / lm_head dequants, the
/// GEMM routing) is A3 (#30), not A1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightsGeometry {
    /// The token-embedding table: (rows, cols) = (vocab, hidden) bf16.
    pub embedding: (u64, u64),
    /// The per-layer GEMM geometries (in the `layer_kinds` order).
    pub per_layer: Vec<LayerGeometry>,
    /// The final-norm size (bf16 `[hidden]`).
    pub final_norm: u64,
    /// The lm_head GEMM (m, k) (the logits GEMM; the artifact's
    /// `text/output_head` is W8G32 — dequantized to bf16 by the A1
    /// normalize step, spec 04).
    pub lm_head: (u64, u64),
}

impl WeightsGeometry {
    /// Derive the geometry from a topology (the [`Weights::synthetic`]
    /// (m, k) derivation, minus the content — pure, config-derived).
    pub fn from_config(config: &ModelConfig) -> Self {
        // The per-layer GEMM (m, k) pairs follow the kernel's convention
        // (m = output rows, k = input dim — the `Nvfp4Weight` doc + the
        // `ignis_nvfp4_gemm_*` FFI, GitHub #33): a projection that maps
        // `input_dim -> output_dim` is (m = output_dim, k = input_dim).
        let per_layer = config
            .layer_kinds
            .iter()
            .map(|kind| match kind {
                LayerKind::Gqa => {
                    let gqa_w = config.gqa_width();
                    let gqa_kv_w = config.gqa_kv_width();
                    LayerGeometry {
                        kind: LayerKind::Gqa,
                        projection: [
                            (gqa_w, config.hidden),
                            (gqa_kv_w, config.hidden),
                            (gqa_kv_w, config.hidden),
                            (config.hidden, gqa_w),
                        ],
                        gdn_output: (0, 0),
                        ffn_gate: (config.ffn_intermediate, config.hidden),
                        ffn_up: (config.ffn_intermediate, config.hidden),
                        ffn_down: (config.hidden, config.ffn_intermediate),
                        norm_in: config.hidden,
                        norm_post: config.hidden,
                        // The GQA layer has no GDN conv / a-b / input
                        // projection (the GDN-only geometry fields,
                        // A3 / #30).
                        gdn_conv: (0, 0),
                        gdn_ab: (0, 0),
                        // The q / k RMSNorm weights are `[head_dim]`
                        // each (the per-head RMSNorm, A3 / #30).
                        qk_norm: config.head_dim,
                        gdn_conv_channels: 0,
                        gdn_in_proj: (0, 0),
                    }
                }
                LayerKind::Gdn => {
                    let conv_ch = config.gdn_conv_channels();
                    LayerGeometry {
                        kind: LayerKind::Gdn,
                        // The GDN input projection's (m, k): the q / k /
                        // v / z parts (the `gdn_in_proj_m` rows — the
                        // artifact's `gdn/query_key_value_z` width,
                        // A3 / #30).
                        projection: [
                            (config.gdn_in_proj_m(), config.hidden),
                            (0, 0),
                            (0, 0),
                            (0, 0),
                        ],
                        // The state readout (m = hidden, k = the
                        // per-token readout width `state_rows` — the
                        // artifact's `gdn/output` tensor is
                        // `[hidden][state_rows]`, A3 / #30).
                        gdn_output: (config.hidden, config.gdn_readout_k()),
                        ffn_gate: (config.ffn_intermediate, config.hidden),
                        ffn_up: (config.ffn_intermediate, config.hidden),
                        ffn_down: (config.hidden, config.ffn_intermediate),
                        norm_in: config.hidden,
                        norm_post: config.hidden,
                        // The GDN causal-conv geometry (the 4 taps ×
                        // the conv channels, A3 / #30).
                        gdn_conv: if conv_ch > 0 {
                            (4, conv_ch)
                        } else {
                            (0, 0)
                        },
                        // The GDN a/b (gate / beta) projection (the
                        // `gdn_ab_width` rows — `gdn/a_b_projection`,
                        // A3 / #30).
                        gdn_ab: if config.gdn_ab_width > 0 {
                            (config.gdn_ab_width, config.hidden)
                        } else {
                            (0, 0)
                        },
                        qk_norm: 0,
                        gdn_conv_channels: conv_ch,
                        gdn_in_proj: (config.gdn_in_proj_m(), config.hidden),
                    }
                }
            })
            .collect();
        Self {
            embedding: (config.vocab, config.hidden),
            per_layer,
            final_norm: config.hidden,
            // The logits GEMM (m, k): m = output dim (vocab), k = input dim
            // (hidden). Derived from the artifact descriptor's
            // `text/output_head` shape `[vocab, hidden]` (the W8G32 lm_head,
            // A1 normalization, spec 04) — the kernel's (m, k) convention is
            // (m = output rows, k = input dim) per the `Nvfp4Weight` doc and
            // the `ignis_nvfp4_gemm_*` FFI (out[tokens][m] = sum_k
            // act[tokens][k] * W[m][k]).
            lm_head: (config.vocab, config.hidden),
        }
    }
}

impl Weights {
    /// A deterministic synthetic model's weights (a pure function of `seed`
    /// + the geometry). Every GEMM weight is a fixed pattern of E2M1 codes +
    /// E4M3 scales, so identical inputs produce identical outputs (the
    /// self-consistency invariant, ADR 0007: greedy + fixed seed).
    pub fn synthetic(config: &ModelConfig, seed: u64) -> Self {
        let ones = to_bf16(&vec![1.0f32; config.hidden as usize]);
        // The synthetic q / k RMSNorm weights (the identity convention —
        // the synthetic model's per-head RMSNorm is a plain RMS, A3 / #30).
        let head_ones = to_bf16(&vec![1.0f32; config.head_dim as usize]);
        let mut gdn_index = 0usize;
        let per_layer = config
            .layer_kinds
            .iter()
            .map(|kind| match kind {
                LayerKind::Gqa => {
                    let gqa_w = config.gqa_width();
                    let gqa_kv_w = config.gqa_kv_width();
                    LayerWeights {
                        kind: LayerKind::Gqa,
                        // (m, k) = (output dim, input dim) — the kernel's
                        // GEMM convention (the `Nvfp4Weight` doc + the
                        // `ignis_nvfp4_gemm_*` FFI, GitHub #33).
                        projection: [
                            nvfp4_weight(gqa_w, config.hidden, seed),
                            nvfp4_weight(
                                gqa_kv_w,
                                config.hidden,
                                seed.wrapping_add(1),
                            ),
                            nvfp4_weight(
                                gqa_kv_w,
                                config.hidden,
                                seed.wrapping_add(2),
                            ),
                            nvfp4_weight(
                                config.hidden,
                                gqa_w,
                                seed.wrapping_add(3),
                            ),
                        ],
                        gdn_output: Nvfp4Weight::empty(),
                        ffn_gate: nvfp4_weight(
                            config.ffn_intermediate,
                            config.hidden,
                            seed.wrapping_add(4),
                        ),
                        ffn_up: nvfp4_weight(
                            config.ffn_intermediate,
                            config.hidden,
                            seed.wrapping_add(5),
                        ),
                        ffn_down: nvfp4_weight(
                            config.hidden,
                            config.ffn_intermediate,
                            seed.wrapping_add(6),
                        ),
                        norm_in: ones.clone(),
                        norm_post: ones.clone(),
                        // The GQA layer has no GDN conv / a-b (the GDN-
                        // only fields, A3 / #30); the q / k RMSNorm
                        // weights are the identity (the synthetic
                        // convention — a plain per-head RMS).
                        gdn_conv: Bf16Weight::empty(),
                        gdn_ab: Bf16Weight::empty(),
                        qk_norm: [head_ones.clone(), head_ones.clone()],
                        // The synthetic path's GEMM content is the host
                        // NVFP4 plane (the BF16-exception + the device
                        // slots are the `from_artifact` path's, A3 / #30).
                        qkv_bf16: Bf16Weight::empty(),
                        attn_out_bf16: Bf16Weight::empty(),
                        gdn_out_bf16: Bf16Weight::empty(),
                        dev: LayerDeviceSlots::default(),
                        gdn_index: 0,
                    }
                }
                LayerKind::Gdn => {
                    let in_proj_m = config.gdn_in_proj_m();
                    let conv_ch = config.gdn_conv_channels();
                    let gdn_idx = gdn_index;
                    gdn_index += 1;
                    LayerWeights {
                        kind: LayerKind::Gdn,
                        // The GDN input projection (m = the GDN feature
                        // rows — the q / k / v / z parts, `gdn_in_proj_m` —
                        // k = hidden) + the state readout (m = hidden,
                        // k = the per-token readout width `state_rows`,
                        // A3 / #30) — the kernel's (m, k) convention
                        // (GitHub #33).
                        projection: [
                            nvfp4_weight(in_proj_m, config.hidden, seed),
                            Nvfp4Weight::empty(),
                            Nvfp4Weight::empty(),
                            Nvfp4Weight::empty(),
                        ],
                        gdn_output: nvfp4_weight(
                            config.hidden,
                            config.gdn_readout_k(),
                            seed.wrapping_add(1),
                        ),
                        ffn_gate: nvfp4_weight(
                            config.ffn_intermediate,
                            config.hidden,
                            seed.wrapping_add(2),
                        ),
                        ffn_up: nvfp4_weight(
                            config.ffn_intermediate,
                            config.hidden,
                            seed.wrapping_add(3),
                        ),
                        ffn_down: nvfp4_weight(
                            config.hidden,
                            config.ffn_intermediate,
                            seed.wrapping_add(4),
                        ),
                        norm_in: ones.clone(),
                        norm_post: ones.clone(),
                        // The GDN causal-conv weight (the 4 taps × the
                        // conv channels — the synthetic pattern, A3 / #30).
                        gdn_conv: Bf16Weight::bf16_weight(4, conv_ch, seed.wrapping_add(8)),
                        // The a/b projection (the model's `gdn_ab_width`
                        // rows — 0 for the synthetic model: the step's
                        // g / beta are 0, A3 / #30).
                        gdn_ab: Bf16Weight::bf16_weight(config.gdn_ab_width, config.hidden, seed.wrapping_add(9)),
                        qk_norm: [Vec::new(), Vec::new()],
                        qkv_bf16: Bf16Weight::empty(),
                        attn_out_bf16: Bf16Weight::empty(),
                        gdn_out_bf16: Bf16Weight::empty(),
                        dev: LayerDeviceSlots::default(),
                        gdn_index: gdn_idx,
                    }
                }
            })
            .collect();
        Self {
            embedding: to_bf16(
                &(0..(config.vocab * config.hidden) as usize)
                    .map(|i| (((i as u64).wrapping_mul(seed).wrapping_add(1) % 31) as f32) / 31.0)
                    .collect::<Vec<_>>(),
            ),
            final_norm: ones,
            // The logits GEMM (m, k): m = vocab (output rows), k = hidden
            // (input dim) — the kernel's convention (the `Nvfp4Weight` doc
            // + the `ignis_nvfp4_gemm_*` FFI), the same (vocab, hidden)
            // the `from_config` derivation uses (GitHub #27 / #33).
            lm_head: HeadWeight::Nvfp4(nvfp4_weight(
                config.vocab,
                config.hidden,
                seed.wrapping_add(7),
            )),
            per_layer,
        }
    }

    /// A zero-cost weight placeholder for a real (artifact) topology (#26):
    /// the real weights live in VRAM (the materialized artifact, ADR 0002)
    /// and the host-side `Weights` stay empty. Superseded on the
    /// `from_artifact` path by #27's [`Weights::from_geometry`] (the real
    /// (m, k) geometry, not the zero geometry); retained as the dev-path
    /// zero-cost construction.
    ///
    /// **Why not [`Weights::synthetic`] here:** the real Qwen 3.8-27B
    /// topology's synthetic weights are ~1.6 TiB of generated host vectors
    /// (48 GDN layers × ~30 GiB of `gdn_output` E2M1 codes alone, 32
    /// billion-loop iterations per layer in a debug build), so the E2E test
    /// stalled after the 19 GB H2D (the #26 hang — the stall looked like a
    /// GPU deadlock, but the GPU was idle: the stall was this host-side
    /// weight generation).
    pub fn placeholder(config: &ModelConfig) -> Self {
        let mut gdn_index = 0usize;
        let per_layer = config
            .layer_kinds
            .iter()
            .map(|kind| {
                let gdn_idx = if matches!(kind, LayerKind::Gdn) {
                    let idx = gdn_index;
                    gdn_index += 1;
                    idx
                } else {
                    0
                };
                LayerWeights {
                    kind: *kind,
                    projection: [
                        Nvfp4Weight::empty(),
                        Nvfp4Weight::empty(),
                        Nvfp4Weight::empty(),
                        Nvfp4Weight::empty(),
                    ],
                    gdn_output: Nvfp4Weight::empty(),
                    ffn_gate: Nvfp4Weight::empty(),
                    ffn_up: Nvfp4Weight::empty(),
                    ffn_down: Nvfp4Weight::empty(),
                    norm_in: Vec::new(),
                    norm_post: Vec::new(),
                    // The A3 / #30 fields: all empty on the zero-cost
                    // placeholder (the dev path — the content is the
                    // synthetic / `from_geometry` construction's).
                    gdn_conv: Bf16Weight::empty(),
                    gdn_ab: Bf16Weight::empty(),
                    qk_norm: [Vec::new(), Vec::new()],
                    qkv_bf16: Bf16Weight::empty(),
                    attn_out_bf16: Bf16Weight::empty(),
                    gdn_out_bf16: Bf16Weight::empty(),
                    dev: LayerDeviceSlots::default(),
                    gdn_index: gdn_idx,
                }
            })
            .collect();
        Self {
            embedding: Vec::new(),
            per_layer,
            final_norm: Vec::new(),
            lm_head: HeadWeight::Nvfp4(Nvfp4Weight::empty()),
        }
    }

    /// A `Weights` with the real (non-zero) geometry for a real (artifact)
    /// topology (the #27 A1 normalization seam, spec 04 — the replacement
    /// of [`Weights::placeholder`] on the `from_artifact` path):
    ///
    /// - The GEMM `Nvfp4Weight`s carry the real (m, k) (the
    ///   [`WeightsGeometry`] derivation, the `synthetic` convention) with
    ///   **empty** code/scale planes — the content is the artifact's
    ///   normalized NVFP4 buffers (the device-resident materialization,
    ///   ADR 0002), consumed by the forward pass (A3).
    /// - The norm vectors are sized (identity 1.0, the `synthetic`
    ///   convention — a neutral, numerically-sane placeholder value).
    /// - The `embedding` content is the `text/token_embedding` W8 -> bf16
    ///   dequant buffer (the A1 normalize step's output, the
    ///   `text/output_head` lm_head dequant the same), routed by the
    ///   forward pass (A3) — not zero-filled here (the #26 lesson: no
    ///   host-side weight explosion on the load path).
    pub fn from_geometry(config: &ModelConfig) -> Self {
        let geometry = WeightsGeometry::from_config(config);
        let ones = to_bf16(&vec![1.0f32; config.hidden as usize]);
        let head_ones = to_bf16(&vec![1.0f32; config.head_dim as usize]);
        let mut gdn_index = 0usize;
        let per_layer = geometry
            .per_layer
            .iter()
            .map(|lg| {
                let gdn_idx = if lg.kind == LayerKind::Gdn {
                    let idx = gdn_index;
                    gdn_index += 1;
                    idx
                } else {
                    0
                };
                LayerWeights {
                    kind: lg.kind,
                    projection: [
                        Nvfp4Weight::geometry_only(lg.projection[0].0, lg.projection[0].1),
                        Nvfp4Weight::geometry_only(lg.projection[1].0, lg.projection[1].1),
                        Nvfp4Weight::geometry_only(lg.projection[2].0, lg.projection[2].1),
                        Nvfp4Weight::geometry_only(lg.projection[3].0, lg.projection[3].1),
                    ],
                    gdn_output: Nvfp4Weight::geometry_only(lg.gdn_output.0, lg.gdn_output.1),
                    ffn_gate: Nvfp4Weight::geometry_only(lg.ffn_gate.0, lg.ffn_gate.1),
                    ffn_up: Nvfp4Weight::geometry_only(lg.ffn_up.0, lg.ffn_up.1),
                    ffn_down: Nvfp4Weight::geometry_only(lg.ffn_down.0, lg.ffn_down.1),
                    norm_in: ones.clone(),
                    norm_post: ones.clone(),
                    // The A3 / #30 fields (the real (m, k) geometry, the
                    // content is the artifact's normalized buffers — the
                    // GEMM planes are device-resident (the `dev` slots,
                    // the `from_artifact` routing), the bf16 buffers are
                    // host-copied, the norms are identity (the
                    // `synthetic` convention — a neutral, numerically-
                    // sane placeholder value)).
                    gdn_conv: Bf16Weight::geometry_only(lg.gdn_conv.0, lg.gdn_conv.1),
                    gdn_ab: Bf16Weight::geometry_only(lg.gdn_ab.0, lg.gdn_ab.1),
                    qk_norm: if lg.qk_norm > 0 {
                        [head_ones.clone(), head_ones.clone()]
                    } else {
                        [Vec::new(), Vec::new()]
                    },
                    qkv_bf16: Bf16Weight::empty(),
                    attn_out_bf16: Bf16Weight::empty(),
                    gdn_out_bf16: Bf16Weight::empty(),
                    dev: LayerDeviceSlots::default(),
                    gdn_index: gdn_idx,
                }
            })
            .collect();
        Self {
            embedding: Vec::new(),
            per_layer,
            final_norm: ones,
            lm_head: HeadWeight::Nvfp4(Nvfp4Weight::geometry_only(
                geometry.lm_head.0,
                geometry.lm_head.1,
            )),
        }
    }

    /// Populate the `Weights` with the A1 host-side dequantized W8 endpoints
    /// (the `text/token_embedding` + the `text/output_head`, ADR 0005): the
    /// embedding table + the lm_head carry the real dequantized bf16 content
    /// (the "real normalized buffers", spec 04 criterion 3 — the artifact's
    /// two W8 endpoints, dequantized host-side). The NVFP4 GEMM planes stay
    /// device-resident (the `from_geometry`'s geometry-only content — not
    /// host-copied, the #26 lesson: no host weight explosion on the load
    /// path; the whole-text-scope copy is A3's, not A1's).
    pub fn with_w8_endpoints(mut self, endpoints: W8Endpoints) -> Self {
        self.embedding = endpoints.embedding;
        self.lm_head = HeadWeight::DequantBf16 {
            data: endpoints.lm_head,
            m: endpoints.lm_head_shape.0,
            k: endpoints.lm_head_shape.1,
        };
        self
    }
}

// ---------------------------------------------------------------------------
// Per-request state (host-side; the kernel-leaf H2D/D2H's these each call)
// ---------------------------------------------------------------------------

/// One request's compute state: the paged KV cache (GQA), the GDN
/// recurrent state (GDN layers), the block table, and the generated-token
/// bookkeeping (the `max_tokens` / EOS soft-stop, ADR 0007).
#[derive(Debug)]
struct RequestState {
    /// The paged GQA KV cache: two planes (K then V), each
    /// `[num_blocks][num_kv_heads][block_size][head_dim]` bf16.
    kv_cache: Vec<u16>,
    /// The GDN recurrent state (per GDN layer): `[num_gdn_layers]
    /// [state_rows][state_cols]` bf16.
    gdn_state: Vec<u16>,
    /// The GDN causal-conv rolling state (per GDN layer, A3 / #30,
    /// kernel-abi 06): `[num_gdn_layers][conv_channels][3]` bf16 (the
    /// 3-tap rolling state s0, s1, s2 per channel — zero until the
    /// first token).
    gdn_conv_state: Vec<u16>,
    /// The logical block -> physical page table (`[num_blocks]` i32).
    block_table: Vec<i32>,
    /// The current paged-KV fill (keys placed so far; the GQA seq_len).
    kv_len: u64,
    /// The sequence position of the next token to place (the RoPE `pos`
    /// contract, kernel-abi 06: every (batch, seq) token rotates at
    /// `pos` — the per-token prefill / decode position, A3 / #30).
    seq_pos: u64,
    /// The last generated token (the autoregressive decode query — the
    /// real model threads the actually-generated token back into the next
    /// step, A3 / #30; `None` until the first decode, a fresh request
    /// prefills then decodes from the last prompt token).
    last_generated: Option<TokenId>,
    /// Tokens generated so far (the `max_tokens` / EOS soft-stop counter).
    generated: u32,
    /// The request's `max_tokens` cap (from its prefill `params`).
    max_tokens: Option<u32>,
}

impl RequestState {
    fn new(config: &ModelConfig, max_tokens: Option<u32>) -> Self {
        let kv_plane = (config.num_blocks
            * config.num_kv_heads
            * config.block_size
            * config.head_dim) as usize;
        let gdn_mat = (config.gdn_num_layers
            * config.gdn_state_rows
            * config.gdn_state_cols) as usize;
        // The GDN causal-conv rolling state (A3 / #30): per GDN layer,
        // the 3-tap rolling state per conv channel (`[channels][3]`,
        // channel-major — the `ignis_gdn_causal_conv` contract).
        let conv_state = (config.gdn_num_layers
            * config.gdn_conv_channels()
            * 3) as usize;
        Self {
            kv_cache: vec![0u16; kv_plane * 2],
            gdn_state: vec![0u16; gdn_mat],
            gdn_conv_state: vec![0u16; conv_state],
            block_table: (0..config.num_blocks).map(|b| b as i32).collect(),
            kv_len: 0,
            seq_pos: 0,
            last_generated: None,
            generated: 0,
            max_tokens,
        }
    }
}

// ---------------------------------------------------------------------------
// The compute backend
// ---------------------------------------------------------------------------

/// A captured decode graph's representative geometry (the decode-batch dims
/// the graph was captured for; a decode step of a different size replays the
/// eager sequence instead, ADR 0003 eager-fallback).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphGeometry {
    /// The decode batch size the graph was captured for.
    pub batch: i64,
}

/// A thread-safe handle to the captured CUDA graph (the kernel-abi 03 empty-
/// capture warm-up, superseded by the B2 / #32 decode graph — retained for
/// the kernel-abi 03 surface's handle type). The raw `*mut IgnisGraph` is
/// not `Send`, but the handle is captured once at startup and launched via
/// the leaf's thread-safe FFI primitives (ADR 0003), so a wrapper asserts
/// the thread-safety (the `Compute` trait's `Send + Sync` bound,
/// `scheduler.rs`).
#[allow(dead_code)]
struct SendGraph(*mut ffi::IgnisGraph);

#[allow(dead_code)]
unsafe impl Send for SendGraph {}

/// A thread-safe handle to the decode graph (the CUDA-graph decode replay,
/// B2 / #32, ADR 0008). The raw `*mut IgnisDecodeGraph` is not `Send`, but
/// the handle is constructed once at startup (the leaf-owned staging buffers
/// + capture stream are stable) and replayed / eager-referenced via the
/// leaf's thread-safe FFI primitives (ADR 0003), so a wrapper asserts the
/// thread-safety (the `Compute` trait's `Send + Sync` bound, `scheduler.rs`).
struct SendDecodeGraph(*mut ffi::IgnisDecodeGraph);

unsafe impl Send for SendDecodeGraph {}

/// The artifact's device context (the #26 fix): holds the `CudaDevice` (the
/// VRAM arena owner), the `MaterializedArtifact` (the materialized tensor
/// views), and the `Reader` (name resolution) so the 19 GB of weights stay
/// in VRAM for the lifetime of the backend (ADR 0002: the producing device
/// must outlive the materialized artifact and its typed views). Always
/// present on the struct (so the construction is uniform); `None` on the
/// synthetic / dev path (or when the `cuda` feature is off — the enum has no
/// variants, so it can only be `None`). The `device` field is declared last
/// so it is dropped last (the arena outlives the materialized views).
enum DeviceCtx {
    #[cfg(feature = "cuda")]
    Cuda {
        /// The artifact reader (name resolution; dropped before the device).
        /// Held (not read in the scoped #26) so the device outlives the
        /// forward pass; #25's device routing reads it.
        #[allow(dead_code)]
        reader: Reader,
        /// The materialized artifact (the tensor views point into the
        /// device's arena; dropped before the device). Held (not read in the
        /// scoped #26); #25's device routing reads it.
        #[allow(dead_code)]
        artifact: MaterializedArtifact,
        /// The CUDA device (the VRAM arena owner; dropped last — the arena
        /// outlives the materialized views). Held so the 19 GB of weights
        /// stay in VRAM for the lifetime of the backend (the #26 fix); #25's
        /// device routing reads it.
        #[allow(dead_code)]
        device: CudaDevice,
    },
}

// SAFETY: `DeviceCtx` holds the `CudaDevice` (the VRAM arena owner), the
// `MaterializedArtifact` (raw pointers into the arena), and the `Reader`
// (the mmap'd file reader). The raw pointers point into the device arena,
// which is stable for the lifetime of the `CudaDevice` (declared last in
// `DeviceCtx`, so dropped last — the arena outlives the views). Sharing
// `DeviceCtx` across threads (the `Compute` trait's `Send + Sync` bound,
// `scheduler.rs`) is safe: the tensor views are read-only (the forward pass
// reads the weights, never mutates them), and the producing device (the
// arena owner) is dropped last. In the non-`cuda` build `DeviceCtx` has no
// variants (an empty enum), so this is a no-op.
unsafe impl Send for DeviceCtx {}
unsafe impl Sync for DeviceCtx {}

/// The production [`Compute`] backend: the kernel-leaf C-ABI forward pass
/// (the compute-adapter, kernel-abi 04).
///
/// Construct with a [`ModelConfig`] + [`Weights`] (a synthetic model for the
/// self-consistency test, or a dequantized artifact model via `from_artifact`,
/// feature `cuda`).
pub struct CudaCompute {
    config: ModelConfig,
    weights: Weights,
    state: Mutex<HashMap<RequestId, RequestState>>,
    /// The decode graph (the CUDA-graph decode replay, the decode hot path,
    /// B2 / #32, ADR 0008): the leaf's `ignis_decode_graph` handle (the
    /// captured decode DAG + the fixed-address device staging buffers).
    /// `Some` when the construction-time capture succeeded (a free GPU, ADR
    /// 0006), `None` on a busy/absent GPU or a VRAM shortfall (the eager
    /// fallback path, ADR 0003 / ADR 0006).
    graph: Mutex<Option<SendDecodeGraph>>,
    /// The captured decode graph's representative geometry (`None` on a
    /// busy/absent GPU or a VRAM shortfall — the eager fallback path, ADR
    /// 0006). A decode step of a different batch runs the eager sequence
    /// (ADR 0003).
    graph_geom: Option<GraphGeometry>,
    /// The GQA RoPE inverse-frequency table (kernel-abi 06, A3 / #30):
    /// `inv_freq[pair] = θ^(-2·pair/rotary_dim)` (θ = `rope_theta`,
    /// `rotary_dim/2` pairs) — computed once at construction (host-side,
    /// a deterministic table; a non-goal is the per-step table recompute),
    /// consumed by the `ignis_rope_qk` kernel (the GQA layers' q / k,
    /// the forward assembly, A3 / #30).
    rope_inv_freq: Vec<f32>,
    /// The artifact's device context (the #26 fix: the 19 GB of weights stay
    /// in VRAM for the lifetime of the backend). `None` on the synthetic /
    /// dev path.
    device_ctx: Option<DeviceCtx>,
    /// The prefill jobs completed through the multi-token (batched) forward
    /// path (B1 / #31): incremented once per `prefill` that runs the
    /// multi-token layer-stack pass (`prefill_batched`). The eager per-token
    /// loop (the `seq == 1` GEMV special case, the prefix-reuse tail
    /// prefill, the ADR 0003 busy/absent-kernel fallback) never increments —
    /// the observation surface for the `prefill_step` dispatch (spec 08
    /// acceptance: the GPU test asserts a multi-token prefill took the
    /// multi-token path, a single-token one did not).
    batched_prefills: AtomicU64,
    /// The decode-step jobs replayed through the CUDA-graph hot path (B2 /
    /// #32, ADR 0008): incremented once per `decode_step` that runs the
    /// graph replay (the single-token, representative-batch case — a
    /// `jobs.len() == 1` step while the graph is active). The eager fallback
    /// (a batch that does not match the captured `GraphGeometry`, or a
    /// busy/absent GPU that left the graph `None`, ADR 0003 / ADR 0006)
    /// never increments — the observation surface for the `decode_step`
    /// dispatch (the spec 09 acceptance: a test asserts the hot path used
    /// the graph — the counter > 0 after a single-token step — and the eager
    /// fallback engaged on a batch mismatch / a no-graph host).
    graph_launches: AtomicU64,
}

impl CudaCompute {
    /// Construct a compute backend over a synthetic (or dequantized) model.
    ///
    /// B2 / #32: constructs the decode graph (the CUDA-graph decode replay,
    /// the decode hot path, ADR 0008) — the fixed-address device staging
    /// buffers + the captured representative decode sequence (the graph's
    /// DAG). On a busy/absent GPU (or a VRAM shortfall) the decode graph
    /// self-skips (ADR 0006) and the backend falls back to the eager
    /// sequence (ADR 0003) — a no-GPU host still gets a (correct, eager)
    /// backend, so the scheduler never faults on a busy GPU.
    pub fn new(config: ModelConfig, weights: Weights) -> Self {
        // B2 (#32): the decode graph (the decode hot path, ADR 0008). The
        // construction-time capture (a free GPU, a VRAM that fits the
        // staging) leaves `graph` Some + the `GraphGeometry` set; a
        // busy/absent GPU or a VRAM shortfall self-skips (ADR 0006) and
        // leaves the eager fallback (ADR 0003).
        let (graph, graph_geom) = match Self::build_decode_graph(&config, &weights) {
            Some(handle) => (
                Some(SendDecodeGraph(handle)),
                Some(GraphGeometry { batch: 1 }),
            ),
            None => (None, None),
        };
        // The GQA RoPE inverse-frequency table (kernel-abi 06, A3 / #30):
        // computed once at construction (host-side, a deterministic table;
        // a non-goal is the per-step table recompute), consumed by the
        // `ignis_rope_qk` kernel (the GQA layers' q / k, the forward
        // assembly).
        let rope_inv_freq = rope_inv_frequencies(config.rope_theta, config.rotary_dim as i64);
        Self {
            config,
            weights,
            state: Mutex::new(HashMap::new()),
            graph: Mutex::new(graph),
            graph_geom,
            rope_inv_freq,
            device_ctx: None,
            batched_prefills: AtomicU64::new(0),
            graph_launches: AtomicU64::new(0),
        }
    }

    /// Construct the compute backend on the EAGER path (no kernel-leaf
    /// startup check, no CUDA-graph capture) — the #26 fix. `from_artifact`
    /// uses this (not [`Self::new`]) to keep the scoped #26 (materialize +
    /// `vram_resident`) free of the graph fast path (the CUDA-graph launch is
    /// #25 material, not needed to land 19 GB in VRAM). Note: the earlier
    /// "hang" (a `cudaStreamSynchronize` that would not complete, the GPU
    /// idle) was NOT the graph check — it was a CPU-side `Weights::synthetic`
    /// OOM trap at the real topology (see [`Weights::placeholder`]).
    pub fn new_eager(config: ModelConfig, weights: Weights) -> Self {
        // The GQA RoPE inverse-frequency table (kernel-abi 06, A3 / #30):
        // computed once at construction (host-side, a deterministic table;
        // a non-goal is the per-step table recompute), consumed by the
        // `ignis_rope_qk` kernel (the GQA layers' q / k, the forward
        // assembly).
        let rope_inv_freq = rope_inv_frequencies(config.rope_theta, config.rotary_dim as i64);
        Self {
            config,
            weights,
            state: Mutex::new(HashMap::new()),
            graph: Mutex::new(None),
            graph_geom: None,
            rope_inv_freq,
            device_ctx: None,
            batched_prefills: AtomicU64::new(0),
            graph_launches: AtomicU64::new(0),
        }
    }

    /// B2 (#32): build the decode graph (ADR 0008) — the representative
    /// decode geometry (the decode-batch dims, derived from the topology) +
    /// the read-only weights (host — the leaf H2D's them once, the
    /// synthetic / dev path; the leaf-allocated + zeroed mutable decode
    /// state, the paged KV + GDN state, the ADR 0003 eager geometry) — then
    /// the leaf's `ignis_decode_graph_new` (the construction-time capture
    /// + the staging buffers). The decode graph self-skips on a busy /
    /// absent GPU or a VRAM shortfall (ADR 0006 — the eager fallback).
    /// Returns the leaf handle (`Some`) or `None` (the self-skip — the
    /// eager fallback, ADR 0003).
    fn build_decode_graph(
        config: &ModelConfig,
        weights: &Weights,
    ) -> Option<*mut ffi::IgnisDecodeGraph> {
        // The representative decode geometry (the decode-batch dims, ADR 0008).
        let geom = ffi::IgnisDecodeGraphGeom {
            hidden: config.hidden as i64,
            vocab: config.vocab as i64,
            num_q_heads: config.num_q_heads as i64,
            num_kv_heads: config.num_kv_heads as i64,
            head_dim: config.head_dim as i64,
            block_size: config.block_size as i64,
            num_blocks: config.num_blocks as i64,
            gdn_state_rows: config.gdn_state_rows as i64,
            gdn_state_cols: config.gdn_state_cols as i64,
            gdn_state_dim: config.gdn_state_dim() as i64,
            gdn_num_layers: config.gdn_num_layers as i64,
        };
        // The lm_head's bf16 buffer (the representative decode step's logits
        // GEMM consumes a plain bf16 weight): the dequantized bf16 variant
        // (the artifact's W8 endpoint) is used as-is (the host buffer — the
        // leaf H2D's it during the call); the NVFP4 variant (the synthetic
        // path) is dequantized to bf16 (a temporary the leaf H2D's during
        // the call, ADR 0005).
        let (lm_head_ptr, lm_head_tmp): (*const c_void, Option<Vec<u16>>) =
            match &weights.lm_head {
                HeadWeight::DequantBf16 { data, .. } => {
                    (data.as_ptr() as *const c_void, None)
                }
                HeadWeight::Nvfp4(nv) => {
                    let buf = Self::dequant_nvfp4_to_bf16(nv);
                    (buf.as_ptr() as *const c_void, Some(buf))
                }
            };
        // The weights descriptor (host — the leaf H2D's the read-only
        // weights once; `weights_on_device` = 0 selects the host case, ADR
        // 0008; the device-resident artifact case is the `from_artifact`
        // path's `weights_on_device` = 1, ADR 0002).
        let wts = ffi::IgnisDecodeGraphWeights {
            weights_on_device: 0,
            embedding: weights.embedding.as_ptr() as *const c_void,
            embedding_bytes: (config.vocab * config.hidden * 2) as i64,
            final_norm: weights.final_norm.as_ptr() as *const c_void,
            final_norm_bytes: (config.hidden * 2) as i64,
            lm_head: lm_head_ptr,
            lm_head_bytes: (config.vocab * config.hidden * 2) as i64,
        };
        // Construct the decode graph (the self-skip — a -1 on a busy /
        // absent GPU, or a VRAM shortfall, ADR 0006 — the eager fallback).
        // `lm_head_tmp` (the NVFP4 dequant temporary) lives for the call
        // (the leaf H2D's the weights synchronously inside it).
        let mut out: *mut ffi::IgnisDecodeGraph = std::ptr::null_mut();
        let rc = unsafe { ffi::ignis_decode_graph_new(&geom, &wts, &mut out) };
        drop(lm_head_tmp);
        if rc == 0 {
            Some(out)
        } else {
            None
        }
    }

    /// Dequantize an NVFP4 GEMM weight to a plain bf16 `[m][k]` row-major
    /// buffer (the deterministic host dequant, B2 / #32 — the
    /// representative decode step's lm_head GEMM consumes a plain bf16
    /// weight; the NVFP4 variant (the synthetic path) is dequantized here,
    /// the leaf H2D's the result once, a temporary). Mirrors the nvfp4
    /// codec (the E2M1 / E4M3 dequant, kernel-abi 01).
    fn dequant_nvfp4_to_bf16(w: &Nvfp4Weight) -> Vec<u16> {
        let m = w.m as usize;
        let k = w.k as usize;
        if m == 0 || k == 0 {
            return Vec::new();
        }
        let mut out = vec![0u16; m * k];
        for r in 0..m {
            for c in 0..k {
                // The E2M1 code (two packed per byte: the low 4 bits = the
                // even k, the high 4 bits = the odd k).
                let byte = w.codes[r * (k / 2) + c / 2];
                let code = if c % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
                let e2m1 = Self::nvfp4_e2m1(code);
                // The E4M3 group scale (one per 16-element group).
                let scale = Self::nvfp4_e4m3(w.scales[r * (k / 16) + c / 16]);
                out[r * k + c] = f32_to_bf16(e2m1 * scale);
            }
        }
        out
    }

    /// The E2M1 (FP4) decode (1 sign bit, 3 magnitude bits ->
    /// {0, .5, 1, 1.5, 2, 3, 4, 6}), 1:1 with the nvfp4 codec (kernel-abi
    /// 01, the `decode_nvfp4_e2m1` helper).
    fn nvfp4_e2m1(code: u8) -> f32 {
        let mag = match code & 0x7 {
            0 => 0.0,
            1 => 0.5,
            2 => 1.0,
            3 => 1.5,
            4 => 2.0,
            5 => 3.0,
            6 => 4.0,
            _ => 6.0,
        };
        if code & 0x8 != 0 {
            -mag
        } else {
            mag
        }
    }

    /// The E4M3 (OCP FP8, bias 7, no inf) decode (1 sign, 4 exponent, 3
    /// mantissa; subnormals (exp == 0) use (m/8) * 2^-6), 1:1 with the
    /// nvfp4 codec (kernel-abi 01, the `decode_nvfp4_e4m3` helper).
    fn nvfp4_e4m3(code: u8) -> f32 {
        let sign = if code & 0x80 != 0 { -1.0f32 } else { 1.0 };
        let exp = ((code >> 3) & 0x0F) as i32;
        let man = (code & 0x07) as f32;
        let mag = if exp == 0 {
            (man / 8.0) * 0.015625 // subnormal: (m/8) * 2^-6
        } else {
            (1.0 + man / 8.0) * 2.0f32.powi(exp - 7)
        };
        sign * mag
    }

    /// Whether the CUDA-graph fast path is active (the construction-time
    /// capture succeeded; the decode step may launch the graph, ADR 0003).
    pub fn uses_graph(&self) -> bool {
        self.graph_geom.is_some()
    }

    /// The captured decode graph's representative geometry (`None` when the
    /// graph is not active — a busy/absent GPU, ADR 0006).
    pub fn graph_geometry(&self) -> Option<GraphGeometry> {
        self.graph_geom
    }

    /// Whether this backend holds the artifact's device context (the #26 fix):
    /// the 19 GB of weights are materialized in VRAM and held for the
    /// lifetime of the backend (the `from_artifact` path). The CPU /
    /// synthetic path returns `false` (no materialization, ADR 0006).
    pub fn vram_resident(&self) -> bool {
        self.device_ctx.is_some()
    }

    /// The number of prefill jobs completed through the multi-token
    /// (batched) forward path (B1 / #31 — the `prefill_step` dispatch's
    /// observation surface, spec 08): a prefill runs the multi-token path
    /// only when `seq > 1` (the `seq == 1` GEMV special case stays on the
    /// single-token path, ADR 0001) on a fresh (empty-KV) request (the
    /// multi-token attention's fresh-prompt causal mask — base_pos = 0 —
    /// a warm-KV tail prefill stays on the per-token loop), and the
    /// multi-token kernels were available (a busy/absent kernel falls back
    /// to the per-token eager loop, ADR 0003 — a fallback that ran never
    /// increments this counter). The GPU tests (the `batched_prefill_gpu`
    /// integration) assert the dispatch through this counter.
    pub fn batched_prefill_count(&self) -> u64 {
        self.batched_prefills.load(Ordering::Relaxed)
    }

    /// The number of decode-step jobs replayed through the CUDA-graph hot
    /// path (B2 / #32, ADR 0008 — the `decode_step` dispatch's observation
    /// surface, spec 09): a `decode_step` increments this counter once per
    /// single-token (representative-batch) step that runs the graph replay
    /// (the `ignis_graph_launch`, ADR 0008). The eager fallback (a batch
    /// that does not match the captured `GraphGeometry`, or a busy/absent
    /// GPU that left the graph `None`, ADR 0003 / ADR 0006) never
    /// increments — the GPU test asserts the hot path used the graph
    /// (the counter > 0 after a single-token step) and the eager fallback
    /// engaged on a batch mismatch (the counter unchanged after a
    /// multi-token step).
    pub fn graph_launch_count(&self) -> u64 {
        self.graph_launches.load(Ordering::Relaxed)
    }

    /// B2 (#32, ADR 0008): the decode graph's replay (the per-step hot
    /// path): H2D the token id (the per-step input), launch the graph (the
    /// whole decode DAG runs on the fixed staging buffers), D2H the logits
    /// (bf16 `[vocab]`). Returns the logits (the greedy sample's input —
    /// the caller runs `ignis_greedy_sample`, the A3 / #30 convention). A
    /// no-graph host (a busy/absent GPU, a VRAM shortfall — ADR 0006)
    /// returns an error (the caller runs the eager fallback, ADR 0003).
    pub fn graph_logits_replay(&self, token_id: i32) -> Result<Vec<u16>, ComputeError> {
        // The decode graph's leaf handle (the extracted raw pointer — a no-
        // graph host (a busy/absent GPU, a VRAM shortfall — ADR 0006)
        // returns an error, the caller runs the eager fallback, ADR 0003).
        let g = self
            .graph
            .lock()
            .unwrap()
            .as_ref()
            .map(|s| s.0)
            .ok_or(ComputeError::Kernel(-1))?;
        let mut logits = vec![0u16; self.config.vocab as usize];
        let rc = unsafe {
            ffi::ignis_decode_graph_replay(
                g,
                token_id,
                logits.as_mut_ptr() as *mut c_void,
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(ComputeError::Kernel(rc));
        }
        Ok(logits)
    }

    /// B2 (#32, the kernel-abi 03 "replay == eager" invariant, ADR 0007):
    /// run the representative decode sequence *eagerly* (no graph) over the
    /// same fixed staging buffers — the eager reference for the bit-exact
    /// check (the graph replay's verification). Same inputs / buffers /
    /// kernels as [`Self::graph_logits_replay`], so the logits must be
    /// bit-identical. A no-graph host (a busy/absent GPU, a VRAM
    /// shortfall — ADR 0006) returns an error (the eager fallback, ADR
    /// 0003).
    pub fn graph_logits_eager(&self, token_id: i32) -> Result<Vec<u16>, ComputeError> {
        // The decode graph's leaf handle (the extracted raw pointer — a no-
        // graph host (a busy/absent GPU, a VRAM shortfall — ADR 0006)
        // returns an error, the caller runs the eager fallback, ADR 0003).
        let g = self
            .graph
            .lock()
            .unwrap()
            .as_ref()
            .map(|s| s.0)
            .ok_or(ComputeError::Kernel(-1))?;
        let mut logits = vec![0u16; self.config.vocab as usize];
        let rc = unsafe {
            ffi::ignis_decode_graph_eager(
                g,
                token_id,
                logits.as_mut_ptr() as *mut c_void,
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(ComputeError::Kernel(rc));
        }
        Ok(logits)
    }

    /// Get-or-create a request's compute state (the paged KV cache, the GDN
    /// state, the block table, the `max_tokens` bookkeeping).
    fn ensure_state(&self, request: RequestId, max_tokens: Option<u32>) {
        let mut st = self.state.lock().unwrap();
        if !st.contains_key(&request) {
            st.insert(request, RequestState::new(&self.config, max_tokens));
        }
    }

    // ── pointwise glue (the correctness floor, ADR 0005) ──────────────────

    /// Host-side residual add: `acc += delta` (both bf16, elementwise).
    fn residual(acc: &mut [u16], delta: &[u16]) {
        for (a, d) in acc.iter_mut().zip(delta.iter()) {
            *a = f32_to_bf16(bf16_to_f32(*a) + bf16_to_f32(*d));
        }
    }

    /// Host-side gated-FFN activation: `silu(gate) * up` (bf16, elementwise;
    /// the fused-SiLU kernel is the later performance material, ADR 0005).
    fn silu_mul(up: &[u16], gate: &[u16]) -> Vec<u16> {
        up.iter()
            .zip(gate.iter())
            .map(|(u, g)| {
                let uf = bf16_to_f32(*u);
                let gf = bf16_to_f32(*g);
                let silu = gf / (1.0 + (-gf).exp());
                f32_to_bf16(uf * silu)
            })
            .collect()
    }

    // ── the kernel-leaf ops (the FFI, the heavy ops on the GPU) ──────────

    /// The NVFP4 GEMM: `out = act @ W^T`. For a single-token decode the
    /// kernel-leaf `ignis_nvfp4_gemm_decode` (the GEMV, the correct kernel
    /// for the single-token case, ADR 0001); for a multi-token prefill
    /// `ignis_nvfp4_gemm_prefill` (the multi-token NVFP4 GEMM, kernel-abi
    /// 05). `act` is a bf16 `[tokens][k]` buffer, the output a bf16
    /// `[tokens][m]`.
    fn nvfp4_gemm(
        &self,
        w: &Nvfp4Weight,
        act: &[u16],
        tokens: u64,
    ) -> Result<Vec<u16>, ComputeError> {
        let m = w.m as usize;
        let k = w.k as usize;
        debug_assert_eq!(
            act.len(),
            (tokens as usize) * k,
            "activation width must match the weight's input dim (a caller bug)"
        );
        let mut out = vec![0u16; (tokens as usize) * m];
        let rc = if tokens == 1 {
            // The single-token GEMV (the decode case, ADR 0001): the
            // kernel-leaf `ignis_nvfp4_gemm_decode`.
            unsafe {
                ffi::ignis_nvfp4_gemm_decode(
                    act.as_ptr() as *const c_void,
                    w.codes.as_ptr() as *const c_void,
                    w.scales.as_ptr() as *const c_void,
                    std::ptr::null(),
                    out.as_mut_ptr() as *mut c_void,
                    w.m as i64,
                    w.k as i64,
                    std::ptr::null_mut(),
                )
            }
        } else {
            // The multi-token GEMM (the prefill case, kernel-abi 05): the
            // kernel-leaf `ignis_nvfp4_gemm_prefill`.
            unsafe {
                ffi::ignis_nvfp4_gemm_prefill(
                    act.as_ptr() as *const c_void,
                    w.codes.as_ptr() as *const c_void,
                    w.scales.as_ptr() as *const c_void,
                    std::ptr::null(),
                    out.as_mut_ptr() as *mut c_void,
                    tokens as i64,
                    w.m as i64,
                    w.k as i64,
                    std::ptr::null_mut(),
                )
            }
        };
        if rc != 0 {
            return Err(ComputeError::Kernel(rc));
        }
        Ok(out)
    }

    /// The logits GEMM (the mixed-quant, A1 / #27 + A2b / #29):
    /// `out = act @ W^T`, dispatching on the weight's kernel-expected
    /// format (the [`HeadWeight`]). The NVFP4 variant (the synthetic /
    /// device-resident path) uses the kernel-leaf `ignis_nvfp4_gemm_*`;
    /// the dequantized bf16 variant (the W8 `text/output_head`, ADR 0005
    /// host-side dequant) uses the kernel-leaf `ignis_bf16_gemm` kernel
    /// (kernel-abi 10, A2b / #29 — the third 27B-fidelity kernel; the
    /// host-side scalar fallback is the pre-kernel seam, replaced on the
    /// real path, A3 / #30). `act` is a bf16 `[tokens][k]` buffer, the
    /// output a bf16 `[tokens][m]`.
    fn head_gemm(
        &self,
        w: &HeadWeight,
        act: &[u16],
        tokens: u64,
    ) -> Result<Vec<u16>, ComputeError> {
        match w {
            // The preserved NVFP4 GEMM weight (the synthetic / device-resident
            // path): the kernel-leaf `ignis_nvfp4_gemm_*`.
            HeadWeight::Nvfp4(nv) => self.nvfp4_gemm(nv, act, tokens),
            // The dequantized bf16 GEMM weight (the W8 `text/output_head`,
            // ADR 0005 host-side dequant): the kernel-leaf `ignis_bf16_gemm`
            // (kernel-abi 10, A2b / #29 — the third 27B-fidelity kernel;
            // the logits path on the real model, A3 / #30).
            HeadWeight::DequantBf16 { data, m, k } => {
                self.bf16_gemm_rows(
                    data.as_ptr(),
                    *m,
                    *k,
                    0,
                    *m,
                    act,
                    tokens,
                )
            }
        }
    }

    /// The bf16 GEMM (`ignis_bf16_gemm`, kernel-abi 10, A2b / #29 — the
    /// logits path for the W8-dequantized lm_head + the artifact's BF16
    /// tensors): `out[tokens][m] = sum_k act[tokens][k] * W[m][k]`.
    /// `wt` is the row-major `[m][k]` bf16 plane (row 0), the GEMM
    /// consumes the row slice `[row_off, row_off + m_slice)` (the fused
    /// tensor's per-slot slices, A3 / #30 — the q / k / v slots of the
    /// fused qkvz plane, the gate / up slots of the fused `mlp/gate_up`).
    fn bf16_gemm_rows(
        &self,
        wt: *const u16,
        m: u64,
        k: u64,
        row_off: u64,
        m_slice: u64,
        act: &[u16],
        tokens: u64,
    ) -> Result<Vec<u16>, ComputeError> {
        debug_assert_eq!(
            act.len(),
            (tokens as usize) * (k as usize),
            "activation width must match the weight's input dim (a caller bug)"
        );
        debug_assert!(
            row_off + m_slice <= m,
            "the slot slice must fit within the plane"
        );
        // The row slice's plane offset (row-major `[m][k]` bf16: a row is
        // `k` bf16 words).
        let wt_ptr = unsafe { wt.add((row_off as usize) * (k as usize)) };
        let mut out = vec![0u16; (tokens as usize) * (m_slice as usize)];
        let rc = unsafe {
            ffi::ignis_bf16_gemm(
                act.as_ptr() as *const c_void,
                wt_ptr as *const c_void,
                std::ptr::null(),
                out.as_mut_ptr() as *mut c_void,
                tokens as i64,
                m_slice as i64,
                k as i64,
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(ComputeError::Kernel(rc));
        }
        Ok(out)
    }

    /// The bf16 GEMM over a [`Bf16Weight`] (the artifact's preserved BF16
    /// tensor — the `gdn/a_b_projection` / the BF16-exception projections,
    /// A3 / #30).
    fn bf16_gemm(
        &self,
        w: &Bf16Weight,
        row_off: u64,
        m_slice: u64,
        act: &[u16],
        tokens: u64,
    ) -> Result<Vec<u16>, ComputeError> {
        self.bf16_gemm_rows(
            w.data.as_ptr(),
            w.m,
            w.k,
            row_off,
            m_slice,
            act,
            tokens,
        )
    }

    /// The NVFP4 GEMM with a DEVICE-RESIDENT plane (the artifact's
    /// materialized arena, ADR 0002 — the `ignis_nvfp4_gemm_*_device`
    /// kernels, ticket 26 / GitHub #26: no per-call weight H2D, the #26
    /// fix). `plane` is the artifact's fused tensor's planes (the
    /// `Nvfp4DevicePlane`, the `from_artifact` routing, A3 / #30); the
    /// GEMM consumes the row slice `[row_off, row_off + m_slice)` (the
    /// slot's rows within the fused plane — the q / k / v slots of the
    /// qkvz plane, the gate / up slots of the `mlp/gate_up` plane).
    /// `act` is a bf16 `[tokens][k]` buffer (host, H2D'd), the output a
    /// bf16 `[tokens][m_slice]` (D2H'd).
    fn nvfp4_gemm_device(
        &self,
        plane: &Nvfp4DevicePlane,
        row_off: u64,
        m_slice: u64,
        act: &[u16],
        tokens: u64,
    ) -> Result<Vec<u16>, ComputeError> {
        let k = plane.k as usize;
        debug_assert_eq!(
            act.len(),
            (tokens as usize) * k,
            "activation width must match the plane's input dim (a caller bug)"
        );
        debug_assert!(
            row_off + m_slice <= plane.m,
            "the slot slice must fit within the plane"
        );
        // The row slice's plane offsets (the planes are the artifact's
        // `[m][k/2]` / `[m][k/16]` layout — the "for now" row-major
        // starting point, ADR 0005).
        let codes_ptr = unsafe { plane.codes.add((row_off as usize) * (k / 2)) };
        let scales_ptr = unsafe { plane.scales.add((row_off as usize) * (k / 16)) };
        let mut out = vec![0u16; (tokens as usize) * (m_slice as usize)];
        let rc = if tokens == 1 {
            // The single-token GEMV (the decode case, ADR 0001): the
            // kernel-leaf `ignis_nvfp4_gemm_decode_device`.
            unsafe {
                ffi::ignis_nvfp4_gemm_decode_device(
                    act.as_ptr() as *const c_void,
                    codes_ptr as *const c_void,
                    scales_ptr as *const c_void,
                    std::ptr::null(),
                    out.as_mut_ptr() as *mut c_void,
                    m_slice as i64,
                    plane.k as i64,
                    std::ptr::null_mut(),
                )
            }
        } else {
            // The multi-token GEMM (kernel-abi 05): the kernel-leaf
            // `ignis_nvfp4_gemm_prefill_device`.
            unsafe {
                ffi::ignis_nvfp4_gemm_prefill_device(
                    act.as_ptr() as *const c_void,
                    codes_ptr as *const c_void,
                    scales_ptr as *const c_void,
                    std::ptr::null(),
                    out.as_mut_ptr() as *mut c_void,
                    tokens as i64,
                    m_slice as i64,
                    plane.k as i64,
                    std::ptr::null_mut(),
                )
            }
        };
        if rc != 0 {
            return Err(ComputeError::Kernel(rc));
        }
        Ok(out)
    }

    /// The GDN causal conv (`ignis_gdn_causal_conv`, kernel-abi 06, A2 /
    /// #28 — the GDN layer's input, A3 / #30): the 4-tap depthwise causal
    /// conv + SiLU over the projected q / k / v rows (the conv'd part of
    /// the GDN input projection — the z rows bypass it). `w` is the
    /// `[4][channels]` tap-major conv weight (the artifact's
    /// `gdn/convolution`); `state_in` / `state_out` are the per-layer
    /// rolling 3-tap state (`[channels][3]`, channel-major — the
    /// `state_out` receives the updated state — the last 3 consumed taps;
    /// `state_in` may alias `state_out`). `projected` is the conv'd part
    /// of the input projection (`[tokens][channels]` bf16), the output
    /// the conv'd + SiLU'd rows. The `tokens` param serves the single-
    /// token step (`tokens` = 1) and the multi-token batched prefill
    /// (the B1 / #31 pass — the rolling conv state advances over the
    /// whole chunk in one call).
    fn gdn_causal_conv(
        &self,
        w: &Bf16Weight,
        projected: &[u16],
        state_in: &[u16],
        state_out: &mut [u16],
        tokens: u64,
    ) -> Result<Vec<u16>, ComputeError> {
        let ch = w.k as usize;
        debug_assert_eq!(
            w.m,
            4,
            "the GDN causal conv has 4 taps (the artifact's gdn/convolution)"
        );
        debug_assert_eq!(
            projected.len(),
            (tokens as usize) * ch,
            "the conv'd part must be [tokens][channels]"
        );
        debug_assert_eq!(
            state_in.len(),
            ch * 3,
            "the rolling conv state is [channels][3]"
        );
        debug_assert_eq!(
            state_out.len(),
            ch * 3,
            "the rolling conv state is [channels][3]"
        );
        debug_assert!(
            !w.data.is_empty(),
            "the GDN layer needs a causal-conv weight (the artifact's gdn/convolution)"
        );
        let mut out = vec![0u16; (tokens as usize) * ch];
        let rc = unsafe {
            ffi::ignis_gdn_causal_conv(
                projected.as_ptr() as *const c_void,
                w.data.as_ptr() as *const c_void,
                state_in.as_ptr() as *const c_void,
                state_out.as_mut_ptr() as *mut c_void,
                out.as_mut_ptr() as *mut c_void,
                tokens as i64,
                ch as i64,
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(ComputeError::Kernel(rc));
        }
        Ok(out)
    }

    /// The GQA RoPE (`ignis_rope_qk`, kernel-abi 06, A2 / #28 — the GQA
    /// layers' q / k, A3 / #30): the split-half NeoX rotation of the first
    /// `rotary_dim` dims of each q / k head (in-place on `q` / `k`), at
    /// the token's sequence position `pos` (the per-token `pos` contract —
    /// every (batch, seq) token rotates at `pos`, the multi-token prefill
    /// is per-token, kernel-abi 06). `q` is bf16 `[num_q_heads][head_dim]`
    /// (batch = 1, seq = 1), `k` is bf16 `[num_kv_heads][head_dim]`.
    fn rope_qk(&self, q: &mut [u16], k: &mut [u16], pos: u64) -> Result<(), ComputeError> {
        let cfg = &self.config;
        debug_assert_eq!(
            q.len(),
            (cfg.num_q_heads as usize) * (cfg.head_dim as usize),
            "the q plane must be [num_q_heads][head_dim]"
        );
        debug_assert_eq!(
            k.len(),
            (cfg.num_kv_heads as usize) * (cfg.head_dim as usize),
            "the k plane must be [num_kv_heads][head_dim]"
        );
        let rc = unsafe {
            ffi::ignis_rope_qk(
                q.as_mut_ptr() as *mut c_void,
                k.as_mut_ptr() as *mut c_void,
                self.rope_inv_freq.as_ptr() as *const c_void,
                1, // batch = 1
                1, // seq = 1 (the single-token step)
                cfg.num_q_heads as i64,
                cfg.num_kv_heads as i64,
                cfg.head_dim as i64,
                cfg.rotary_dim as i64,
                pos as i32,
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(ComputeError::Kernel(rc));
        }
        Ok(())
    }

    /// The q / k RMSNorm (the `ignis_rmsnorm`, kernel-abi 02 — the
    /// per-head RMSNorm, the reference's `qk_norm` step, A3 / #30):
    /// `out = (x / rms(x)) * weight` over each head's `head_dim` slice.
    /// `weight` is the `[head_dim]` per-head RMSNorm weight (the
    /// artifact's `attention/query_norm` / `key_norm`); an empty
    /// `weight` is a parameter-free RMS (a null weight, the synthetic /
    /// no-weight convention).
    fn per_head_rmsnorm(
        &self,
        x: &[u16],
        num_heads: u64,
        weight: &[u16],
    ) -> Result<Vec<u16>, ComputeError> {
        let hd = self.config.head_dim as usize;
        debug_assert_eq!(
            x.len(),
            (num_heads as usize) * hd,
            "the q / k plane must be [num_heads][head_dim]"
        );
        let weight_ptr = if weight.is_empty() {
            std::ptr::null()
        } else {
            debug_assert_eq!(
                weight.len(),
                hd,
                "the q / k RMSNorm weight is [head_dim] (the per-head norm)"
            );
            weight.as_ptr() as *const c_void
        };
        let mut out = x.to_vec();
        for h in 0..num_heads as usize {
            let xs = &x[h * hd..(h + 1) * hd];
            let os = &mut out[h * hd..(h + 1) * hd];
            let rc = unsafe {
                ffi::ignis_rmsnorm(
                    xs.as_ptr() as *const c_void,
                    weight_ptr,
                    std::ptr::null(),
                    os.as_mut_ptr() as *mut c_void,
                    hd as i64,
                    0.0,
                    std::ptr::null_mut(),
                )
            };
            if rc != 0 {
                return Err(ComputeError::Kernel(rc));
            }
        }
        Ok(out)
    }

    /// The RMSNorm (`ignis_rmsnorm`, the kernel-leaf pointwise op):
    /// `out = (x / rms(x)) * weight` (bf16, `[n]`).
    fn rmsnorm(&self, x: &[u16], weight: &[u16]) -> Result<Vec<u16>, ComputeError> {
        let mut out = vec![0u16; x.len()];
        let rc = unsafe {
            ffi::ignis_rmsnorm(
                x.as_ptr() as *const c_void,
                weight.as_ptr() as *const c_void,
                std::ptr::null(),
                out.as_mut_ptr() as *mut c_void,
                x.len() as i64,
                0.0,
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(ComputeError::Kernel(rc));
        }
        Ok(out)
    }

    /// The embedding lookup (`ignis_embedding`, the kernel-leaf pointwise
    /// op): `out[row] = table[id[row]]` (bf16, `[batch][hidden]`).
    fn embed(&self, ids: &[i32]) -> Result<Vec<u16>, ComputeError> {
        let hid = self.config.hidden as usize;
        let mut out = vec![0u16; ids.len() * hid];
        let rc = unsafe {
            ffi::ignis_embedding(
                self.weights.embedding.as_ptr() as *const c_void,
                ids.as_ptr() as *const c_void,
                out.as_mut_ptr() as *mut c_void,
                ids.len() as i64,
                self.config.vocab as i64,
                self.config.hidden as i64,
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(ComputeError::Kernel(rc));
        }
        Ok(out)
    }

    /// The greedy sampler (`ignis_greedy_sample`, the kernel-leaf argmax):
    /// `out[i] = argmax(logits[i])` (deterministic, ties to the lowest
    /// index — the v1 correctness floor, ADR 0007: greedy + fixed seed).
    fn sample(&self, logits: &[f32]) -> Result<TokenId, ComputeError> {
        let mut out = [0i32; 1];
        let rc = unsafe {
            ffi::ignis_greedy_sample(
                logits.as_ptr() as *const c_void,
                out.as_mut_ptr() as *mut c_void,
                1,
                self.config.vocab as i64,
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(ComputeError::Kernel(rc));
        }
        let tok = out[0] as u32;
        if tok as u64 >= self.config.vocab {
            return Err(ComputeError::Kernel(-1));
        }
        Ok(tok)
    }

    /// Place a token's k/v into the paged KV cache (the GQA block-table
    /// addressing, ADR 0001). `k`/`v` are bf16 `[num_kv_heads*head_dim]`
    /// (the GQA key/value planes). A caller holding the state lock calls
    /// this (it mutates the state in place).
    fn store_kv(&self, req_state: &mut RequestState, k: &[u16], v: &[u16]) {
        let cfg = &self.config;
        let kv_w = cfg.gqa_kv_width() as usize;
        if k.len() != kv_w || v.len() != kv_w {
            // A geometry mismatch (a caller bug): the projection produced
            // the wrong width. A no-op (not a CUDA fault).
            return;
        }
        let head = cfg.head_dim as usize;
        let kv_heads = cfg.num_kv_heads as usize;
        let block = cfg.block_size as usize;
        let plane = (cfg.num_blocks * kv_heads as u64 * block as u64 * head as u64) as usize;
        // The paged KV is kv_head-major within a page (head_dim fastest);
        // the synthetic model places k/v linearly (the self-consistency
        // invariant holds regardless of the exact addressing).
        let page = (req_state.kv_len as usize) / kv_heads;
        let slot = (req_state.kv_len as usize) % kv_heads;
        for h in 0..kv_heads {
            for d in 0..head {
                let idx = (h * block + slot) * head + d;
                if idx < plane {
                    req_state.kv_cache[idx] = k[h * head + d];
                    req_state.kv_cache[plane + idx] = v[h * head + d];
                }
            }
        }
        req_state.kv_len += 1;
        let _ = page; // (the page index is the block-table's job, ADR 0001)
    }

    // ── the forward pass ───────────────────────────────────────────────────

    /// The batched (multi-token) prefill dispatch rule (B1 / #31 — spec
    /// 08's acceptance criterion 1 + 3): a prefill runs the multi-token
    /// path only when the prompt is `seq > 1` tokens (the `seq == 1` GEMV
    /// special case stays on the single-token path, ADR 0001) and the
    /// request's KV state is fresh (`kv_len == 0` — the multi-token
    /// attention's fresh-prompt causal mask (base_pos = 0) is only valid
    /// on an empty cache; a prefix-reuse tail prefill (a warm KV) keeps
    /// the per-token loop). Pure + CPU-testable (the `batched_prefill`
    /// integration's dispatch pins); the `prefill` dispatch consults it.
    pub fn batched_prefill_eligible(seq: u64, kv_len: u64) -> bool {
        seq > 1 && kv_len == 0
    }

    /// The compute-adapter's prefill step: warm a request's KV cache + GDN
    /// state (the `prefill_step` seam, `scheduler.rs`). Composes the
    /// kernel-leaf primitives (embedding, the NVFP4 GEMM, the GQA/GDN
    /// attention, the norms) over the prompt tokens.
    ///
    /// The dispatch (B1 / #31 — the multi-token forward path, spec 08):
    /// a `seq > 1` fresh (empty-KV) prompt runs the layer stack in **one
    /// multi-token pass** (`prefill_batched` — the multi-token GEMM,
    /// kernel-abi 05 + the multi-token attention, kernel-abi 01);
    /// `seq == 1` (the GEMV special case, ADR 0001) and a non-fresh
    /// (prefix-reuse tail) prefill keep the **per-token loop**
    /// (`prefill_eager` — the ADR 0003 eager fallback); a busy/absent
    /// multi-token kernel (a kernel-rc error mid-pass) falls back to the
    /// per-token loop too (the fresh state is restored first — the
    /// correctness floor is unchanged, a non-GPU host still gets a
    /// (correct, eager) prefill).
    fn prefill(&self, job: &PrefillJob) -> Result<(), ComputeError> {
        self.ensure_state(job.request, job.params.max_tokens);
        let seq = job.tokens.len() as u64;
        // The dispatch decision (the `batched_prefill_eligible` rule — the
        // request's KV state before this prefill; a fresh (never-prefilled)
        // request has `kv_len == 0` by construction).
        let kv_len = self
            .state
            .lock()
            .unwrap()
            .get(&job.request)
            .map(|s| s.kv_len)
            .unwrap_or(0);
        if Self::batched_prefill_eligible(seq, kv_len) {
            if let Err(e) = self.prefill_batched(job, seq) {
                // A busy/absent multi-token kernel (the ADR 0003 eager
                // fallback — spec 08's acceptance criterion 3): restore
                // the fresh state invariant (a failed batched pass may
                // have left partial state) and run the per-token loop
                // (the correctness floor is unchanged).
                if matches!(e, ComputeError::Kernel(_)) {
                    self.reset_fresh_state(job.request);
                    return self.prefill_eager(job);
                }
                return Err(e);
            }
            return Ok(());
        }
        self.prefill_eager(job)
    }

    /// The per-token (eager) prefill loop (the ADR 0003 eager fallback —
    /// the `seq == 1` GEMV special case, the prefix-reuse tail prefill,
    /// and the busy/absent-kernel fallback of the batched pass, B1 /
    /// #31): run the layer stack over the prompt one token at a time
    /// (the single-token GEMV `ignis_nvfp4_gemm_decode` + the single-
    /// token `ignis_gqa_attention_decode`, ADR 0001).
    fn prefill_eager(&self, job: &PrefillJob) -> Result<(), ComputeError> {
        let cfg = &self.config;
        let ids: Vec<i32> = job.tokens.iter().map(|&t| t as i32).collect();
        let emb = self.embed(&ids)?;
        let hid = cfg.hidden as usize;
        let seq = job.tokens.len();
        // Run the layer stack over the prompt (the GQA layers warm the KV
        // cache — the rotated k / v; the GDN layers warm the GDN state +
        // the rolling conv state). Per-token (the RoPE `pos` contract —
        // the multi-token prefill is B1, A3 / #30; this is the per-token
        // loop it falls back to).
        let mut acc = vec![0u16; hid];
        for pos in 0..seq {
            let h_in = &emb[pos * hid..(pos + 1) * hid];
            acc = self.forward_layers(job.request, h_in)?;
        }
        // The prefill is complete (the KV + GDN state + the conv state
        // are warm); the decode query is the last prompt token (the
        // autoregressive decode — the real model threads the actually-
        // generated token back into the next step, A3 / #30).
        self.state
            .lock()
            .unwrap()
            .get_mut(&job.request)
            .map(|s| {
                s.kv_len = (seq as u64).min(cfg.num_blocks * cfg.block_size);
                if let Some(last) = job.tokens.last() {
                    s.last_generated = Some(*last);
                }
            });
        let _ = acc; // the decode query is the lm-head GEMM input (below).
        Ok(())
    }

    /// The multi-token (batched) prefill pass (B1 / #31 — spec 08's
    /// performance path): a `seq`-token prompt through the layer stack in
    /// one pass — the embedding (the `ignis_embedding` batch, the
    /// `[seq][hidden]` activation), the multi-token layer-stack forward
    /// (`forward_layers_multi` — the multi-token GEMM, kernel-abi 05, +
    /// the multi-token attention, kernel-abi 01, + the per-token GDN
    /// recurrence + the per-token RoPE / KV writeback, spec 08), the
    /// final RMSNorm (the per-token pointwise glue, ADR 0005) + the
    /// lm_head multi-token GEMM (`ignis_nvfp4_gemm_prefill` — the
    /// synthetic path — / the `ignis_bf16_gemm` logits path, A2b / #29)
    /// + the greedy sample (`ignis_greedy_sample`) over the last token's
    /// logits (the sane-output self-check, ADR 0007). The prefill itself
    /// emits no token to the scheduler (the `prefill_step` contract) —
    /// the decode query stays the last prompt token (the A3 / #30 decode
    /// convention).
    fn prefill_batched(&self, job: &PrefillJob, seq: u64) -> Result<(), ComputeError> {
        let cfg = &self.config;
        // The embedding lookup (the kernel-leaf `ignis_embedding` — the
        // `[seq][hidden]` activation, the multi-token pass's input).
        let ids: Vec<i32> = job.tokens.iter().map(|&t| t as i32).collect();
        let emb = self.embed(&ids)?;
        // The multi-token layer-stack forward (the batched prefill pass,
        // B1 / #31 — the KV + GDN state + the rolling conv state are warm
        // after it; the GQA layers' KV writeback filled the paged cache
        // with all `seq` tokens' K / V, spec 08).
        let acc = self.forward_layers_multi(job.request, &emb, seq)?;
        // The final RMSNorm (the per-token pointwise glue — the
        // `ignis_rmsnorm` over the `[seq][hidden]` plane, ADR 0005) +
        // the lm_head multi-token GEMM (kernel-abi 05 / the A2b / #29
        // bf16 logits path) + the greedy sample (`ignis_greedy_sample`)
        // over the last token's logits (the sane-output self-check, ADR
        // 0007 — the prefill's emitted logits are in-vocabulary).
        let hid = cfg.hidden as usize;
        let seq_us = seq as usize;
        let mut final_in = vec![0u16; seq_us * hid];
        for i in 0..seq_us {
            let n = self.rmsnorm(&acc[i * hid..(i + 1) * hid], &self.weights.final_norm)?;
            final_in[i * hid..(i + 1) * hid].copy_from_slice(&n);
        }
        let logits = self.head_gemm(&self.weights.lm_head, &final_in, seq)?;
        let vocab = cfg.vocab as usize;
        let last_logits = &logits[(seq_us - 1) * vocab..seq_us * vocab];
        let _sampled = self.sample(&bf16_to_f32s(last_logits))?;
        // The prefill is complete (the KV + GDN state + the conv state
        // are warm); the decode query is the last prompt token (the
        // autoregressive decode — the real model threads the actually-
        // generated token back into the next step, A3 / #30).
        self.state
            .lock()
            .unwrap()
            .get_mut(&job.request)
            .map(|s| {
                s.kv_len = seq.min(cfg.num_blocks * cfg.block_size);
                if let Some(last) = job.tokens.last() {
                    s.last_generated = Some(*last);
                }
            });
        // The batched prefill completed (the `prefill_step` dispatch's
        // observation surface, spec 08 — the GPU tests assert the multi-
        // token path ran through this counter).
        self.batched_prefills.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Restore a fresh (never-prefilled) request's state invariant after a
    /// failed multi-token (batched) prefill (B1 / #31 — the ADR 0003
    /// eager fallback's precondition): the failed pass may have left
    /// partial state (K / V written into some paged slots, the GDN state
    /// / the rolling conv state advanced mid-chunk, `kv_len` / `seq_pos`
    /// advanced), so the per-token fallback loop runs from a clean fresh
    /// state (the `RequestState::new` values — the zeroed GDN / conv
    /// state, the zero `kv_len` / `seq_pos` / `generated`, no `last_
    /// generated`). The K / V cache slots the fallback writes are a
    /// superset of the failed pass's write set (both use the same
    /// `store_kv` placement from `kv_len = 0` — the fallback's full
    /// per-token store set subsumes the failed pass's partial one), so
    /// the cache planes need no zeroing (the stale partial values are
    /// overwritten by the fallback's deterministic placement).
    fn reset_fresh_state(&self, request: RequestId) {
        let mut st = self.state.lock().unwrap();
        if let Some(s) = st.get_mut(&request) {
            s.kv_len = 0;
            s.seq_pos = 0;
            s.generated = 0;
            s.last_generated = None;
            s.gdn_state.fill(0);
            s.gdn_conv_state.fill(0);
        }
    }

    /// The compute-adapter's decode step: generate one token per lane (the
    /// `decode_step` seam, `scheduler.rs`). A request that reaches
    /// `max_tokens` / EOS soft-stops (a per-job `None`, not a fault). The
    /// decode query is the autoregressive one (A3 / #30): the last
    /// generated token (the prefill's last prompt token on the first
    /// decode, the previous decode's token thereafter) — the real model
    /// threads the actually-generated token back into the next step.
    fn decode(&self, job: &DecodeJob) -> Result<Option<TokenId>, ComputeError> {
        let cfg = &self.config;
        // Ensure the request's state (the scheduler always prefills before
        // decoding; a missing state is a caller bug).
        self.ensure_state(job.request, job.params.max_tokens);
        // The decode query: the last generated token (the autoregressive
        // decode, A3 / #30 — the prefill's last prompt token on the first
        // decode, the previous decode's token thereafter; a fresh
        // request without a prefill uses 0).
        let cur = self
            .state
            .lock()
            .unwrap()
            .get(&job.request)
            .and_then(|s| s.last_generated)
            .unwrap_or(0);
        // Embed the current token (the decode query; the single-token case).
        let ids = vec![cur as i32];
        let emb = self.embed(&ids)?;
        let h_in = &emb[..cfg.hidden as usize];
        // The layer stack over the current token.
        let mut acc = self.forward_layers(job.request, h_in)?;
        // The final RMSNorm + the lm-head GEMM (the logits — the real
        // path is the `ignis_bf16_gemm` kernel, A2b / #29) + the greedy
        // sample (the deterministic token, ADR 0007).
        acc = self.rmsnorm(&acc, &self.weights.final_norm)?;
        let logits = self.head_gemm(&self.weights.lm_head, &acc, 1)?;
        let token = self.sample(&bf16_to_f32s(&logits))?;
        // The soft-stop: the request reached `max_tokens` / EOS.
        // The autoregressive bookkeeping: the generated token threads
        // into the next decode step's query (A3 / #30).
        let stop = {
            let mut st = self.state.lock().unwrap();
            let s = st.get_mut(&job.request).ok_or(ComputeError::Kernel(-1))?;
            s.last_generated = Some(token);
            let mt = s.max_tokens.or(job.params.max_tokens);
            s.generated + 1 >= mt.unwrap_or(u32::MAX)
        };
        if !stop {
            self.state
                .lock()
                .unwrap()
                .get_mut(&job.request)
                .map(|s| s.generated += 1);
        }
        Ok(if stop { None } else { Some(token) })
    }

    /// The per-token GDN state readout (the host-side GEMV, the "for now"
    /// readout, ADR 0005): `y[dv] = sum_d S[dv][d] · k[d]` (the current
    /// key's `S^T k` readout, f32 precision — the ported `ignis_gdn_step`
    /// contract updates the state, it does not emit a readout; the caller
    /// assembles it, A3 / #30). `state` is the updated per-layer state
    /// (`[state_rows][state_cols]` bf16), `k` is the conv'd key part
    /// (`[state_cols]` bf16).
    fn state_readout(state: &[u16], k: &[u16]) -> Vec<f32> {
        let cols = k.len();
        let rows = state.len() / cols;
        let k_f: Vec<f32> = k.iter().map(|&v| bf16_to_f32(v)).collect();
        (0..rows)
            .map(|dv| {
                let row = &state[dv * cols..dv * cols + cols];
                row.iter().zip(k_f.iter()).map(|(s, kf)| bf16_to_f32(*s) * kf).sum()
            })
            .collect()
    }

    /// The GQA q / k / v projection GEMMs (A3 / #30 — the GEMM dispatch;
    /// the `tokens` param serves the single-token decode (`tokens` = 1)
    /// and the multi-token batched prefill (the B1 / #31 pass)):
    /// the fused NVFP4 device plane (the `from_artifact` routing — the
    /// artifact's `attention/query_key_gate_value`, the slot's row slice) /
    /// the BF16-exception plane (the artifact's early GQA layers, A1's
    /// `QKGV_BF16_LAYERS`) / the synthetic host `Nvfp4Weight` (the
    /// separate per-slot weights). `slot` is 0 (q), 1 (k), 2 (v) — the
    /// slot's row slice within the fused plane (q: 0, k: `gqa_width`,
    /// v: `gqa_width + gqa_kv_width`).
    fn gqa_proj(
        &self,
        lw: &LayerWeights,
        slot: u64,
        pre: &[u16],
        tokens: u64,
    ) -> Result<Vec<u16>, ComputeError> {
        let cfg = &self.config;
        let (row_off, m) = match slot {
            0 => (0, cfg.gqa_width()),
            1 => (cfg.gqa_width(), cfg.gqa_kv_width()),
            _ => (cfg.gqa_width() + cfg.gqa_kv_width(), cfg.gqa_kv_width()),
        };
        if let Some(plane) = lw.dev.qkv {
            return self.nvfp4_gemm_device(&plane, row_off, m, pre, tokens);
        }
        if !lw.qkv_bf16.data.is_empty() {
            return self.bf16_gemm(&lw.qkv_bf16, row_off, m, pre, tokens);
        }
        // The synthetic / dev path: the separate per-slot host weights
        // (no slice — the synthetic q / k / v are individual weights).
        self.nvfp4_gemm(&lw.projection[slot as usize], pre, tokens)
    }

    /// The GQA output projection GEMM (A3 / #30 — the GEMM dispatch;
    /// the `tokens` param serves the single-token decode (`tokens` = 1)
    /// and the multi-token batched prefill (the B1 / #31 pass)):
    /// the NVFP4 device plane (the artifact's `attention/output`, m =
    /// `hidden`, k = `gqa_width`) / the BF16-exception plane (the early
    /// GQA layers, A1's `ATTENTION_OUT_BF16_LAYERS`) / the synthetic host
    /// `Nvfp4Weight`.
    fn gqa_out_gemm(
        &self,
        lw: &LayerWeights,
        attn_out: &[u16],
        tokens: u64,
    ) -> Result<Vec<u16>, ComputeError> {
        let cfg = &self.config;
        if let Some(plane) = lw.dev.attn_out {
            return self.nvfp4_gemm_device(&plane, 0, cfg.hidden, attn_out, tokens);
        }
        if !lw.attn_out_bf16.data.is_empty() {
            return self.bf16_gemm(&lw.attn_out_bf16, 0, cfg.hidden, attn_out, tokens);
        }
        self.nvfp4_gemm(&lw.projection[3], attn_out, tokens)
    }

    /// The GDN input projection GEMM (A3 / #30 — the GEMM dispatch;
    /// the `tokens` param serves the single-token decode (`tokens` = 1)
    /// and the multi-token batched prefill (the B1 / #31 pass)):
    /// the NVFP4 device plane (the artifact's `gdn/query_key_value_z`,
    /// the q / k / v / z rows, m = `gdn_in_proj_m`) / the synthetic host
    /// `Nvfp4Weight`.
    fn gdn_in_gemm(
        &self,
        lw: &LayerWeights,
        pre: &[u16],
        tokens: u64,
    ) -> Result<Vec<u16>, ComputeError> {
        let cfg = &self.config;
        if let Some(plane) = lw.dev.gdn_in {
            return self.nvfp4_gemm_device(&plane, 0, cfg.gdn_in_proj_m(), pre, tokens);
        }
        self.nvfp4_gemm(&lw.projection[0], pre, tokens)
    }

    /// The GDN state readout GEMM (A3 / #30 — the GEMM dispatch;
    /// the `tokens` param serves the single-token decode (`tokens` = 1)
    /// and the multi-token batched prefill (the B1 / #31 pass)):
    /// the NVFP4 device plane (the artifact's `gdn/output`, m = `hidden`, k =
    /// the readout width `state_rows`) / the BF16-exception plane (the
    /// layer-4 quirk, A1's `GDN_OUT_BF16_LAYERS`) / the synthetic host
    /// `Nvfp4Weight`.
    fn gdn_out_gemm(
        &self,
        lw: &LayerWeights,
        gated: &[u16],
        tokens: u64,
    ) -> Result<Vec<u16>, ComputeError> {
        let cfg = &self.config;
        if let Some(plane) = lw.dev.gdn_out {
            return self.nvfp4_gemm_device(&plane, 0, cfg.hidden, gated, tokens);
        }
        if !lw.gdn_out_bf16.data.is_empty() {
            return self.bf16_gemm(&lw.gdn_out_bf16, 0, cfg.hidden, gated, tokens);
        }
        self.nvfp4_gemm(&lw.gdn_output, gated, tokens)
    }

    /// The gated-FFN gate / up projection GEMMs (A3 / #30 — the GEMM
    /// dispatch; the `tokens` param serves the single-token decode
    /// (`tokens` = 1) and the multi-token batched prefill (the B1 / #31
    /// pass)): the fused NVFP4 device plane (the artifact's
    /// `mlp/gate_up` — the gate slot is row 0, the up slot is
    /// `ffn_intermediate` within the fused plane) / the synthetic host
    /// `Nvfp4Weight`s (the separate gate / up weights).
    fn ffn_gemm(
        &self,
        lw: &LayerWeights,
        is_up: bool,
        post: &[u16],
        tokens: u64,
    ) -> Result<Vec<u16>, ComputeError> {
        let cfg = &self.config;
        if let Some(plane) = lw.dev.mlp_gate_up {
            let row_off = if is_up { cfg.ffn_intermediate } else { 0 };
            return self.nvfp4_gemm_device(&plane, row_off, cfg.ffn_intermediate, post, tokens);
        }
        if is_up {
            self.nvfp4_gemm(&lw.ffn_up, post, tokens)
        } else {
            self.nvfp4_gemm(&lw.ffn_gate, post, tokens)
        }
    }

    /// The gated-FFN down projection GEMM (A3 / #30 — the GEMM dispatch;
    /// the `tokens` param serves the single-token decode (`tokens` = 1)
    /// and the multi-token batched prefill (the B1 / #31 pass)):
    /// the NVFP4 device plane (the artifact's `mlp/down`, m = `hidden`,
    /// k = `ffn_intermediate`) / the synthetic host `Nvfp4Weight`.
    fn ffn_down_gemm(
        &self,
        lw: &LayerWeights,
        act: &[u16],
        tokens: u64,
    ) -> Result<Vec<u16>, ComputeError> {
        let cfg = &self.config;
        if let Some(plane) = lw.dev.mlp_down {
            return self.nvfp4_gemm_device(&plane, 0, cfg.hidden, act, tokens);
        }
        self.nvfp4_gemm(&lw.ffn_down, act, tokens)
    }

    /// Run the layer stack (GQA / GDN + gated-FFN, the compute-adapter's
    /// core) over one token's hidden state `h_in` (bf16), returning the
    /// residual stream after the layer stack (bf16 — the next layer's /
    /// the lm-head's input; the final RMSNorm is the caller's, the
    /// decode step's lm-head input, A3 / #30). Composes the kernel-leaf
    /// primitives (GEMM, GQA, GDN, the GDN causal conv, the GQA RoPE,
    /// the norms — A3 / #30) + the pointwise glue (residual, gated-SiLU,
    /// the correctness floor, ADR 0005).
    ///
    /// The per-layer stack (the full-correct assembly, spec 07):
    /// - **GQA layer:** the QKV projection (the NVFP4 GEMM) → the q / k
    ///   RMSNorm (`ignis_rmsnorm`, the per-head, the
    ///   `attention/query_norm` / `key_norm` weights) + the RoPE
    ///   (`ignis_rope_qk`, the GQA layers' q / k at the token's sequence
    ///   position) → the GQA attention (`ignis_gqa_attention_decode`,
    ///   the paged KV) → the output projection (the NVFP4 GEMM) → the
    ///   residual.
    /// - **GDN layer:** the input projection (the NVFP4 GEMM, the
    ///   `gdn/query_key_value_z` q / k / v / z rows) → the GDN causal
    ///   conv (`ignis_gdn_causal_conv`, the conv'd q / k / v part — the
    ///   z rows bypass it) → the GDN a / b (gate / beta) projection (the
    ///   bf16 GEMM, the `gdn/a_b_projection`) → the GDN step
    ///   (`ignis_gdn_step`, the Gated-DeltaNet recurrence — the
    ///   per-layer state update) → the state readout (the host-side `S^T
    ///   k` GEMV, the "for now" readout, ADR 0005) gated by the z part →
    ///   the output projection (the NVFP4 GEMM, the `gdn/output`) → the
    ///   residual.
    /// - **FFN (every layer):** the gate / up projections (the NVFP4
    ///   GEMM, the fused `mlp/gate_up`) → the gated SiLU (the host
    ///   pointwise glue, ADR 0005) → the down projection (the NVFP4
    ///   GEMM, `mlp/down`) → the residual.
    fn forward_layers(&self, request: RequestId, h_in: &[u16]) -> Result<Vec<u16>, ComputeError> {
        let cfg = &self.config;
        let mut acc: Vec<u16> = h_in.to_vec();
        // Hold the state lock for the layer loop (the GDN state, the GDN
        // conv state, + the KV cache are mutated in place; no
        // re-locking, no deadlock).
        let mut st = self.state.lock().unwrap();
        let req_state = st.get_mut(&request).ok_or(ComputeError::Kernel(-1))?;
        for lw in self.weights.per_layer.iter() {
            // ── attention block ─────────────────────────────────
            // The pre-attention RMSNorm (Qwen pre-norm: the attention /
            // FFN blocks operate on the normalized residual stream —
            // the real model's `input_norm` weight, the identity in the
            // synthetic convention, A3 / #30).
            let pre = if lw.norm_in.is_empty() {
                acc.clone()
            } else {
                self.rmsnorm(&acc, &lw.norm_in)?
            };
            let attn: Vec<u16> = match lw.kind {
                LayerKind::Gqa => {
                    // The QKV projection (the NVFP4 GEMM — the fused
                    // device plane / the BF16-exception plane / the
                    // synthetic host plane, A3 / #30) + the q / k RMSNorm
                    // (the per-head, A3 / #30) + the RoPE (the GQA
                    // layer's q / k at the token's sequence position,
                    // A3 / #30) + the GQA attention (the kernel-leaf
                    // `ignis_gqa_attention_decode`, the paged KV) + the
                    // output projection.
                    let q = self.gqa_proj(lw, 0, &pre, 1)?;
                    let k = self.gqa_proj(lw, 1, &pre, 1)?;
                    let v = self.gqa_proj(lw, 2, &pre, 1)?;
                    // The q / k RMSNorm (the per-head — the real model's
                    // `attention/query_norm` / `key_norm` weights, a
                    // parameter-free RMS when empty, A3 / #30) + the
                    // RoPE (the GQA layer's q / k, the split-half NeoX
                    // rotation at the token's sequence position, A3 /
                    // #30).
                    let q = self.per_head_rmsnorm(&q, cfg.num_q_heads, &lw.qk_norm[0])?;
                    let k = self.per_head_rmsnorm(&k, cfg.num_kv_heads, &lw.qk_norm[1])?;
                    // The RoPE (the GQA layer's q / k — the split-half NeoX
                    // rotation at the token's sequence position, in-place
                    // on q / k, A3 / #30).
                    let mut q = q;
                    let mut k = k;
                    let pos = req_state.seq_pos;
                    self.rope_qk(&mut q, &mut k, pos)?;
                    // Store k / v into the paged KV cache (the GQA
                    // layer's block-table addressing, ADR 0001) — the
                    // rotated k (the cache holds the rotated keys; the
                    // attention's queries are rotated at the same `pos`).
                    self.store_kv(req_state, &k, &v);
                    let qk = cfg.gqa_width() as usize;
                    let mut attn_out = vec![0u16; qk];
                    let rc = unsafe {
                        ffi::ignis_gqa_attention_decode(
                            q.as_ptr() as *const c_void,
                            req_state.kv_cache.as_ptr() as *const c_void,
                            req_state.block_table.as_ptr() as *const c_void,
                            attn_out.as_mut_ptr() as *mut c_void,
                            cfg.num_q_heads as i64,
                            cfg.num_kv_heads as i64,
                            cfg.head_dim as i64,
                            (cfg.num_blocks * cfg.block_size) as i64,
                            cfg.block_size as i64,
                            cfg.num_blocks as i64,
                            0.0, // default 1/sqrt(head_dim)
                            std::ptr::null_mut(),
                        )
                    };
                    if rc != 0 {
                        return Err(ComputeError::Kernel(rc));
                    }
                    // The attention output projection (GQA_width -> hidden).
                    self.gqa_out_gemm(lw, &attn_out, 1)?
                }
                LayerKind::Gdn => {
                    // GDN: the input projection (the NVFP4 GEMM — the
                    // `gdn/query_key_value_z` q / k / v / z rows, A3 /
                    // #30) + the GDN causal conv (the kernel-leaf
                    // `ignis_gdn_causal_conv`, the conv'd q / k / v part
                    // — the z rows bypass it, A3 / #30) + the GDN a / b
                    // (gate / beta) projection (the bf16 GEMM, the
                    // `gdn/a_b_projection`, A3 / #30) + the GDN step
                    // (the kernel-leaf `ignis_gdn_step`, the
                    // Gated-DeltaNet recurrence, the per-layer state
                    // update) + the state readout (the host-side `S^T k`
                    // GEMV, the "for now" readout, ADR 0005) gated by
                    // the z part + the state -> output projection (the
                    // recurrent-state readout GEMM, the `gdn/output`).
                    let feat = self.gdn_in_gemm(lw, &pre, 1)?;
                    // The GDN causal conv (the kernel-leaf
                    // `ignis_gdn_causal_conv`, kernel-abi 06, A2 / #28 —
                    // the GDN layer's input, A3 / #30): the 4-tap
                    // depthwise causal conv + SiLU over the conv'd q / k
                    // / v part (the z rows bypass it, the kernel's
                    // contract), the rolling 3-tap state per layer.
                    let conv_ch = cfg.gdn_conv_channels() as usize;
                    let conv_base = (lw.gdn_index * conv_ch * 3) as usize;
                    let conv_state_in =
                        &req_state.gdn_conv_state[conv_base..conv_base + conv_ch * 3];
                    let mut conv_state_out = vec![0u16; conv_ch * 3];
                    let conv_out = self.gdn_causal_conv(
                        &lw.gdn_conv,
                        &feat[..conv_ch],
                        conv_state_in,
                        &mut conv_state_out,
                        1, // the single-token step (the per-token eager loop)
                    )?;
                    // Commit the updated rolling conv state (in-place;
                    // the next token's conv reads the updated state).
                    req_state
                        .gdn_conv_state[conv_base..conv_base + conv_ch * 3]
                        .copy_from_slice(&conv_state_out);
                    // The GDN a / b (gate / beta) projection (the bf16
                    // GEMM — the artifact's `gdn/a_b_projection`, the
                    // first half is the gate `a`, the second the beta
                    // `b`; 0 when the model has no a / b projection, the
                    // step's g / beta are 0, A3 / #30).
                    let ab = if lw.gdn_ab.m > 0 {
                        self.bf16_gemm(&lw.gdn_ab, 0, lw.gdn_ab.m, &pre, 1)?
                    } else {
                        Vec::new()
                    };
                    // The GDN step's feature x = [k, v, g, beta] (the
                    // conv'd k / v parts + the a / b's gate / beta — the
                    // `ignis_gdn_step` contract, kernel-abi 01).
                    let cols = cfg.gdn_state_cols as usize;
                    let rows = cfg.gdn_state_rows as usize;
                    let q_w = cfg.gdn_q_width as usize;
                    let mut x = vec![0u16; cfg.gdn_state_dim() as usize];
                    x[..cols].copy_from_slice(&conv_out[q_w..q_w + cols]);
                    x[cols..cols + rows]
                        .copy_from_slice(&conv_out[q_w + cols..q_w + cols + rows]);
                    if !ab.is_empty() {
                        x[cols + rows] = ab[0]; // the gate (a) — the first half.
                        x[cols + rows + 1] = ab[ab.len() / 2]; // the beta (b) — the second half.
                    }
                    // The GDN step (the kernel-leaf `ignis_gdn_step`, the
                    // Gated-DeltaNet recurrence — the per-layer state
                    // update, A3 / #30: this layer's state slice, the
                    // flat-ABI per-layer semantics).
                    let state_mat = rows * cols;
                    let state_base = (lw.gdn_index * state_mat) as usize;
                    let state_in = &req_state.gdn_state[state_base..state_base + state_mat];
                    let mut state_out = vec![0u16; state_mat];
                    let rc = unsafe {
                        ffi::ignis_gdn_step(
                            x.as_ptr() as *const c_void,
                            state_in.as_ptr() as *const c_void,
                            state_out.as_mut_ptr() as *mut c_void,
                            1, // batch = 1
                            1, // num_gdn_layers = 1 (this layer's state slice)
                            cfg.gdn_state_rows as i64,
                            cfg.gdn_state_cols as i64,
                            cfg.gdn_state_dim() as i64,
                            std::ptr::null_mut(),
                        )
                    };
                    if rc != 0 {
                        return Err(ComputeError::Kernel(rc));
                    }
                    // Commit the updated GDN state (in-place; the next
                    // step reads the updated state).
                    req_state
                        .gdn_state[state_base..state_base + state_mat]
                        .copy_from_slice(&state_out);
                    // The state readout (the per-token readout `y[dv] =
                    // sum_d S[dv][d] · k[d]` — the host-side GEMV, the
                    // "for now" readout, ADR 0005: the ported step's
                    // contract updates the state, it does not emit a
                    // readout) + the z (output-gate) part of the input
                    // projection (the z rows bypass the conv, they gate
                    // the readout, A3 / #30).
                    let k_part = &conv_out[q_w..q_w + cols];
                    let readout = Self::state_readout(&state_out, k_part);
                    let gated: Vec<u16> = if cfg.gdn_z_width > 0 {
                        let z_w = cfg.gdn_z_width as usize;
                        let z = &feat[conv_ch..conv_ch + z_w];
                        readout
                            .iter()
                            .zip(z.iter())
                            .map(|(y, z)| f32_to_bf16(*y * bf16_to_f32(*z)))
                            .collect()
                    } else {
                        readout.iter().map(|&y| f32_to_bf16(y)).collect()
                    };
                    // The GDN state -> output projection (the readout;
                    // the fused readout kernel is the later performance
                    // material, ADR 0005, the GEMM dispatch, A3 / #30).
                    self.gdn_out_gemm(lw, &gated, 1)?
                }
            };
            // The residual (host pointwise glue, the correctness floor).
            Self::residual(&mut acc, &attn);
            // The post-attention RMSNorm (`ignis_rmsnorm` — the FFN's
            // input, the pre-norm convention, A3 / #30).
            let post = if lw.norm_post.is_empty() {
                acc.clone()
            } else {
                self.rmsnorm(&acc, &lw.norm_post)?
            };
            // ── the gated-FFN block (gate / up GEMV + the gated-SiLU
            // activation (host pointwise, ADR 0005) + the down GEMV) ───
            let gate = self.ffn_gemm(lw, false, &post, 1)?;
            let up = self.ffn_gemm(lw, true, &post, 1)?;
            let act = Self::silu_mul(&up, &gate);
            let ffn_out = self.ffn_down_gemm(lw, &act, 1)?;
            Self::residual(&mut acc, &ffn_out);
            // This token has been consumed: the sequence position
            // advances (the RoPE `pos` contract, A3 / #30).
            req_state.seq_pos += 1;
        }
        // The final RMSNorm is the caller's (the decode step's lm-head
        // input, A3 / #30).
        drop(st);
        Ok(acc)
    }

    /// Run the layer stack (GQA / GDN + gated-FFN, the compute-adapter's
    /// core) over a `seq`-token hidden state `h_in` (bf16
    /// `[seq][hidden]`) in **one pass** (the multi-token batched-prefill
    /// forward, B1 / #31 — spec 08's performance path): every
    /// projection is a multi-token GEMM (the `ignis_nvfp4_gemm_prefill`
    /// family, kernel-abi 05 — the eager per-token loop's single-token
    /// GEMV, ADR 0001, is the `seq == 1` special case), every GQA layer
    /// is the multi-token `ignis_gqa_attention_prefill` (kernel-abi 01 —
    /// the batched query attends over the whole sequence, its K / V
    /// written back into the paged cache first, spec 08), and the GDN
    /// layers' projections are multi-token while the GDN recurrence is
    /// per-token (the Gated-DeltaNet state update is a per-token
    /// recurrence — the GDN step runs per token within the chunk, the
    /// kernel-leaf `ignis_gdn_step`, kernel-abi 01; the projections are
    /// batched, the recurrence is not (inherently sequential) — the
    /// fused GDN-prefill kernel is a later optimization, spec 08's
    /// non-goal).
    ///
    /// The per-token sub-loops (the GDN recurrence, the GQA RoPE / q / k
    /// RMSNorm, the KV writeback, the norms) are the "for now" per-token
    /// composition of the multi-token pass (the pointwise glue is the
    /// correctness floor, ADR 0005: the fused-SiLU / fused qk-norm+RoPE
    /// / fused readout kernels are the later 99%-gate performance
    /// material, ADR 0007).
    ///
    /// The per-token GQA RoPE positions mirror the eager per-token loop's
    /// `pos` contract (kernel-abi 06 — token `i` of layer `l` rotates at
    /// `i * num_layers + l`): the multi-token pass leaves the request
    /// state (the KV cache's rotated K / V, the GDN state, the rolling
    /// conv state, `seq_pos`) exactly where the per-token loop would
    /// have left it, so the subsequent decode step continues the same
    /// position sequence (the self-consistency invariant is unchanged —
    /// the batched path's *accumulation order* differs from the per-
    /// token loop (the multi-token GEMM / attention, spec 08's design §7
    /// caveat), but the state layout + the decode query are identical).
    ///
    /// Returns the residual stream after the layer stack (bf16
    /// `[seq][hidden]` — the caller's final-RMSNorm / lm-head input, the
    /// prefill's multi-token logits, B1 / #31).
    fn forward_layers_multi(
        &self,
        request: RequestId,
        h_in: &[u16],
        tokens: u64,
    ) -> Result<Vec<u16>, ComputeError> {
        let cfg = &self.config;
        let seq = tokens as usize;
        let hid = cfg.hidden as usize;
        let layers = self.weights.per_layer.len();
        debug_assert_eq!(
            h_in.len(),
            seq * hid,
            "the multi-token input must be [seq][hidden]"
        );
        let mut acc: Vec<u16> = h_in.to_vec();
        // Hold the state lock for the layer loop (the GDN state, the GDN
        // conv state, + the KV cache are mutated in place; no
        // re-locking, no deadlock).
        let mut st = self.state.lock().unwrap();
        let req_state = st.get_mut(&request).ok_or(ComputeError::Kernel(-1))?;
        for (l, lw) in self.weights.per_layer.iter().enumerate() {
            // ── attention block (the per-token RMSNorm glue, ADR 0005) ─
            // The pre-attention RMSNorm (Qwen pre-norm: the attention /
            // FFN blocks operate on the normalized residual stream — the
            // real model's `input_norm` weight, the identity in the
            // synthetic convention, A3 / #30): the per-token
            // `ignis_rmsnorm` over the `[seq][hidden]` plane (the fused
            // multi-token norm is the later 99%-gate material, ADR 0007).
            let pre: Vec<u16> = if lw.norm_in.is_empty() {
                acc.clone()
            } else {
                let mut pre = vec![0u16; seq * hid];
                for i in 0..seq {
                    let n = self.rmsnorm(&acc[i * hid..(i + 1) * hid], &lw.norm_in)?;
                    pre[i * hid..(i + 1) * hid].copy_from_slice(&n);
                }
                pre
            };
            let attn: Vec<u16> = match lw.kind {
                LayerKind::Gqa => {
                    // The QKV projection (the multi-token NVFP4 GEMM —
                    // kernel-abi 05 — the fused device plane / the
                    // BF16-exception plane / the synthetic host plane,
                    // A3 / #30) + the per-token q / k RMSNorm (the
                    // per-head, A3 / #30) + the per-token RoPE (the GQA
                    // layer's q / k at the token's sequence position,
                    // A3 / #30) + the KV writeback (all `seq` tokens'
                    // K / V into the paged cache — the multi-token
                    // attention reads its K / V from the cache, spec
                    // 08) + the multi-token attention (`ignis_gqa_
                    // attention_prefill`, kernel-abi 01 — the batched
                    // query attends over the whole sequence, the fresh-
                    // prompt causal mask) + the output projection (the
                    // multi-token GEMM).
                    let q = self.gqa_proj(lw, 0, &pre, tokens)?;
                    let k = self.gqa_proj(lw, 1, &pre, tokens)?;
                    let v = self.gqa_proj(lw, 2, &pre, tokens)?;
                    let qw = cfg.gqa_width() as usize;
                    let kw = cfg.gqa_kv_width() as usize;
                    let mut q = q;
                    let mut k = k;
                    for i in 0..seq {
                        // The per-token q / k RMSNorm (the per-head — the
                        // real model's `attention/query_norm` / `key_
                        // norm` weights, a parameter-free RMS when empty,
                        // A3 / #30).
                        let qn = self.per_head_rmsnorm(
                            &q[i * qw..(i + 1) * qw],
                            cfg.num_q_heads,
                            &lw.qk_norm[0],
                        )?;
                        q[i * qw..(i + 1) * qw].copy_from_slice(&qn);
                        let kn = self.per_head_rmsnorm(
                            &k[i * kw..(i + 1) * kw],
                            cfg.num_kv_heads,
                            &lw.qk_norm[1],
                        )?;
                        k[i * kw..(i + 1) * kw].copy_from_slice(&kn);
                        // The RoPE (the GQA layer's q / k — the split-
                        // half NeoX rotation at the token's sequence
                        // position, in-place on q / k, A3 / #30 — the
                        // eager per-token `pos` contract: token `i` of
                        // layer `l` rotates at `i * num_layers + l`,
                        // kernel-abi 06).
                        let pos = (i as u64) * (layers as u64) + (l as u64);
                        self.rope_qk(
                            &mut q[i * qw..(i + 1) * qw],
                            &mut k[i * kw..(i + 1) * kw],
                            pos,
                        )?;
                        // The KV writeback (the GQA layer's paged-KV
                        // store, ADR 0001) — the rotated k (the cache
                        // holds the rotated keys; the attention's
                        // queries are rotated at the same `pos`).
                        self.store_kv(
                            req_state,
                            &k[i * kw..(i + 1) * kw],
                            &v[i * kw..(i + 1) * kw],
                        );
                    }
                    // The multi-token attention (`ignis_gqa_attention_
                    // prefill`, kernel-abi 01 — the batched query
                    // attends over the whole sequence: `q` is
                    // `[batch = 1][seq][num_q_heads][head_dim]`, the K /
                    // V read from the paged cache's two planes, the
                    // fresh-prompt causal mask — query `i` attends to
                    // keys `[0..=i]`).
                    let qk = cfg.gqa_width() as usize;
                    let mut attn_out = vec![0u16; seq * qk];
                    let rc = unsafe {
                        ffi::ignis_gqa_attention_prefill(
                            q.as_ptr() as *const c_void,
                            req_state.kv_cache.as_ptr() as *const c_void,
                            req_state.block_table.as_ptr() as *const c_void,
                            attn_out.as_mut_ptr() as *mut c_void,
                            1, // batch = 1 (a single request's prompt)
                            seq as i64,
                            cfg.num_q_heads as i64,
                            cfg.num_kv_heads as i64,
                            cfg.head_dim as i64,
                            cfg.block_size as i64,
                            cfg.num_blocks as i64,
                            0.0, // default 1/sqrt(head_dim)
                            std::ptr::null_mut(),
                        )
                    };
                    if rc != 0 {
                        return Err(ComputeError::Kernel(rc));
                    }
                    // The attention output projection (the multi-token
                    // GEMM, `gqa_width` -> hidden).
                    self.gqa_out_gemm(lw, &attn_out, tokens)?
                }
                LayerKind::Gdn => {
                    // GDN: the input projection (the multi-token NVFP4
                    // GEMM — the `gdn/query_key_value_z` q / k / v / z
                    // rows, A3 / #30) + the GDN causal conv (the kernel-
                    // leaf `ignis_gdn_causal_conv`, the conv'd q / k / v
                    // part over the whole chunk — the rolling 3-tap
                    // state advances across the `seq` tokens in one
                    // call, A2 / #28 — the z rows bypass it) + the GDN a
                    // / b (gate / beta) projection (the multi-token bf16
                    // GEMM, the `gdn/a_b_projection`, A3 / #30) + the
                    // GDN step (the kernel-leaf `ignis_gdn_step`, the
                    // Gated-DeltaNet recurrence — the per-token state
                    // update within the chunk, spec 08) + the state
                    // readout (the host-side `S^T k` GEMV, the "for now"
                    // readout, ADR 0005) gated by the z part + the
                    // state -> output projection (the multi-token NVFP4
                    // GEMM, the `gdn/output`).
                    let feat = self.gdn_in_gemm(lw, &pre, tokens)?;
                    // The GDN causal conv (the kernel-leaf
                    // `ignis_gdn_causal_conv`, kernel-abi 06, A2 / #28 —
                    // the GDN layer's input, A3 / #30): the 4-tap
                    // depthwise causal conv + SiLU over the conv'd q / k
                    // / v part (the z rows bypass it, the kernel's
                    // contract), the rolling 3-tap state over the whole
                    // chunk (one call — the `seq`-token `projected`).
                    let conv_ch = cfg.gdn_conv_channels() as usize;
                    let m_in = cfg.gdn_in_proj_m() as usize;
                    let conv_base = (lw.gdn_index * conv_ch * 3) as usize;
                    let conv_state_in =
                        &req_state.gdn_conv_state[conv_base..conv_base + conv_ch * 3];
                    let mut conv_state_out = vec![0u16; conv_ch * 3];
                    let conv_out = self.gdn_causal_conv(
                        &lw.gdn_conv,
                        &feat[..seq * conv_ch],
                        conv_state_in,
                        &mut conv_state_out,
                        tokens,
                    )?;
                    // Commit the updated rolling conv state (in-place —
                    // the state after the whole chunk; the next layer's /
                    // request's conv reads it).
                    req_state
                        .gdn_conv_state[conv_base..conv_base + conv_ch * 3]
                        .copy_from_slice(&conv_state_out);
                    // The GDN a / b (gate / beta) projection (the
                    // multi-token bf16 GEMM — the artifact's
                    // `gdn/a_b_projection`, the first half is the gate
                    // `a`, the second the beta `b`; 0 when the model has
                    // no a / b projection, the step's g / beta are 0, A3
                    // / #30).
                    let ab = if lw.gdn_ab.m > 0 {
                        self.bf16_gemm(&lw.gdn_ab, 0, lw.gdn_ab.m, &pre, tokens)?
                    } else {
                        Vec::new()
                    };
                    // The GDN step's feature x = [k, v, g, beta] (the
                    // conv'd k / v parts + the a / b's gate / beta — the
                    // `ignis_gdn_step` contract, kernel-abi 01). The
                    // Gated-DeltaNet recurrence is per-token (the state
                    // update `S <- αS + δk^T` is a per-token recurrence —
                    // within a prefill chunk the GDN step runs per token,
                    // the `ignis_gdn_step` kernel, kernel-abi 01 — the
                    // projections are batched, the recurrence is not,
                    // spec 08).
                    let cols = cfg.gdn_state_cols as usize;
                    let rows = cfg.gdn_state_rows as usize;
                    let q_w = cfg.gdn_q_width as usize;
                    let z_w = cfg.gdn_z_width as usize;
                    let ab_w = lw.gdn_ab.m as usize;
                    let state_mat = rows * cols;
                    let state_base = (lw.gdn_index * state_mat) as usize;
                    let state_slice =
                        &mut req_state.gdn_state[state_base..state_base + state_mat];
                    let mut state_out = vec![0u16; state_mat];
                    let mut gated = vec![0u16; seq * rows];
                    for i in 0..seq {
                        // The token's GDN step feature x = [k_i, v_i,
                        // g_i, beta_i] (the conv'd k / v parts + the a /
                        // b's gate / beta — the `ignis_gdn_step`
                        // contract, kernel-abi 01).
                        let conv_row = &conv_out[i * conv_ch..(i + 1) * conv_ch];
                        let k_part = &conv_row[q_w..q_w + cols];
                        let mut x = vec![0u16; cfg.gdn_state_dim() as usize];
                        x[..cols].copy_from_slice(k_part);
                        x[cols..cols + rows]
                            .copy_from_slice(&conv_row[q_w + cols..q_w + cols + rows]);
                        if !ab.is_empty() {
                            let ab_row = &ab[i * ab_w..(i + 1) * ab_w];
                            x[cols + rows] = ab_row[0]; // the gate (a) — the first half.
                            x[cols + rows + 1] = ab_row[ab_row.len() / 2]; // the beta (b) — the second half.
                        }
                        // The GDN step (the kernel-leaf `ignis_gdn_step`,
                        // the Gated-DeltaNet recurrence — the per-token
                        // state update, A3 / #30: this layer's state
                        // slice, the flat-ABI per-layer semantics).
                        let rc = unsafe {
                            ffi::ignis_gdn_step(
                                x.as_ptr() as *const c_void,
                                state_slice.as_ptr() as *const c_void,
                                state_out.as_mut_ptr() as *mut c_void,
                                1, // batch = 1 (the per-token recurrence)
                                1, // num_gdn_layers = 1 (this layer's state slice)
                                cfg.gdn_state_rows as i64,
                                cfg.gdn_state_cols as i64,
                                cfg.gdn_state_dim() as i64,
                                std::ptr::null_mut(),
                            )
                        };
                        if rc != 0 {
                            return Err(ComputeError::Kernel(rc));
                        }
                        // Commit the updated GDN state (in-place — the
                        // next token's step reads the updated state, the
                        // per-token recurrence within the chunk).
                        state_slice.copy_from_slice(&state_out);
                        // The token's state readout (the per-token
                        // readout `y[dv] = sum_d S[dv][d] · k[d]` — the
                        // host-side GEMV, the "for now" readout, ADR
                        // 0005) + the z (output-gate) part of the input
                        // projection (the z rows bypass the conv, they
                        // gate the readout, A3 / #30).
                        let readout = Self::state_readout(&state_out, k_part);
                        if z_w > 0 {
                            let z = &feat[i * m_in + conv_ch..i * m_in + conv_ch + z_w];
                            for (j, (y, zv)) in readout.iter().zip(z.iter()).enumerate() {
                                gated[i * rows + j] = f32_to_bf16(*y * bf16_to_f32(*zv));
                            }
                        } else {
                            for (j, y) in readout.iter().enumerate() {
                                gated[i * rows + j] = f32_to_bf16(*y);
                            }
                        }
                    }
                    // The GDN state -> output projection (the multi-
                    // token readout GEMM — the fused readout kernel is
                    // the later performance material, ADR 0005, the GEMM
                    // dispatch, A3 / #30).
                    self.gdn_out_gemm(lw, &gated, tokens)?
                }
            };
            // The residual (host pointwise glue, the correctness floor —
            // the elementwise `[seq][hidden]` add).
            Self::residual(&mut acc, &attn);
            // The post-attention RMSNorm (the `ignis_rmsnorm` — the FFN's
            // input, the pre-norm convention, A3 / #30): the per-token
            // pointwise glue over the `[seq][hidden]` plane (ADR 0005).
            let post = if lw.norm_post.is_empty() {
                acc.clone()
            } else {
                let mut post = vec![0u16; seq * hid];
                for i in 0..seq {
                    let n = self.rmsnorm(&acc[i * hid..(i + 1) * hid], &lw.norm_post)?;
                    post[i * hid..(i + 1) * hid].copy_from_slice(&n);
                }
                post
            };
            // ── the gated-FFN block (the multi-token gate / up GEMM +
            // the gated-SiLU activation (host pointwise glue, ADR 0005)
            // + the multi-token down GEMM) ─
            let gate = self.ffn_gemm(lw, false, &post, tokens)?;
            let up = self.ffn_gemm(lw, true, &post, tokens)?;
            let act = Self::silu_mul(&up, &gate);
            let ffn_out = self.ffn_down_gemm(lw, &act, tokens)?;
            Self::residual(&mut acc, &ffn_out);
            // These `seq` tokens have been consumed at this layer: the
            // sequence position advances by `seq` (the eager per-token
            // loop advances it by one per token-per-layer — the same
            // total after the pass, the RoPE `pos` contract, A3 / #30).
            req_state.seq_pos += tokens;
        }
        // The final RMSNorm is the caller's (the prefill's multi-token
        // logits, B1 / #31).
        drop(st);
        Ok(acc)
    }
}

/// The compute-adapter's prefill / decode steps (the [`Compute`] seam the
/// `ConcreteScheduler` drives, `scheduler.rs`).
impl Compute for CudaCompute {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<(), ComputeError> {
        for job in jobs {
            self.prefill(job)?;
        }
        Ok(())
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<Option<TokenId>>, ComputeError> {
        // B2 (#32, ADR 0008): the decode graph hot path — a single-token,
        // representative-batch (batch == 1) decode step replays the decode
        // graph (the whole decode DAG runs on the fixed staging buffers, the
        // `ignis_graph_launch`, no per-step capture / node update). A decode
        // step whose batch does not match the captured `GraphGeometry`
        // (a `jobs.len() != 1` step) runs the eager sequence (ADR 0003); a
        // busy/absent GPU (the decode graph `None`, ADR 0006) also runs the
        // eager sequence (the eager fallback).
        if jobs.len() == 1 && self.uses_graph() {
            let job = &jobs[0];
            let token = self.decode_graph_step(job)?;
            // The hot path used the graph (the `decode_step` dispatch's
            // observation surface, spec 09 — the GPU test asserts the
            // counter > 0 after a single-token step).
            self.graph_launches.fetch_add(1, Ordering::Relaxed);
            return Ok(vec![token]);
        }
        // The eager fallback (a batch that does not match the captured
        // `GraphGeometry`, or a busy/absent GPU that left the graph `None`,
        // ADR 0003 / ADR 0006): the per-lane hybrid `decode()` (the full-
        // correct model decode, the correctness floor, ADR 0005).
        let mut out = Vec::with_capacity(jobs.len());
        for job in jobs {
            out.push(self.decode(job)?);
        }
        Ok(out)
    }
}

impl CudaCompute {
    /// B2 (#32, ADR 0008): the decode graph's per-step decode (the hot
    /// path): H2D the decode query (the autoregressive token — the last
    /// generated token, A3 / #30), launch the graph (the whole decode DAG
    /// runs on the fixed staging buffers), D2H the logits, the greedy
    /// sample (the deterministic token, ADR 0007), and the soft-stop
    /// bookkeeping (the `max_tokens` / EOS counter, the A3 / #30
    /// convention). Returns the generated token (or `None` on a soft-stop).
    /// This is the *representative* decode (the mechanism this ticket
    /// delivers — the full per-layer stack + the host pointwise glue as
    /// device kernels is the 99%-gate performance material, ADR 0005 /
    /// 0007, ticket 20); the full-correct model decode remains the eager
    /// hybrid `decode()` (the correctness floor).
    fn decode_graph_step(&self, job: &DecodeJob) -> Result<Option<TokenId>, ComputeError> {
        // Ensure the request's state (the scheduler always prefills before
        // decoding; a missing state is a caller bug).
        self.ensure_state(job.request, job.params.max_tokens);
        // The decode query (the autoregressive one — the last generated
        // token, A3 / #30): the prefill's last prompt token on the first
        // decode, the previous decode's token thereafter (a fresh request
        // without a prefill uses token 0).
        let cur = self
            .state
            .lock()
            .unwrap()
            .get(&job.request)
            .and_then(|s| s.last_generated)
            .unwrap_or(0);
        // The decode graph replay (the hot path, ADR 0008): H2D the decode
        // query, launch the graph, D2H the logits (the representative
        // decode's logits).
        let logits = self.graph_logits_replay(cur as i32)?;
        // The greedy sample (the deterministic token, ADR 0007) — the bf16
        // logits -> the f32 logits (the `ignis_greedy_sample` contract,
        // kernel-abi 02), the A3 / #30 convention.
        let token = self.sample(&bf16_to_f32s(&logits))?;
        // The soft-stop (the `max_tokens` / EOS) + the autoregressive
        // bookkeeping (the generated token threads into the next step's
        // query, A3 / #30).
        let stop = {
            let mut st = self.state.lock().unwrap();
            let s = st.get_mut(&job.request).ok_or(ComputeError::Kernel(-1))?;
            s.last_generated = Some(token);
            let mt = s.max_tokens.or(job.params.max_tokens);
            s.generated + 1 >= mt.unwrap_or(u32::MAX)
        };
        if !stop {
            self.state
                .lock()
                .unwrap()
                .get_mut(&job.request)
                .map(|s| s.generated += 1);
        }
        Ok(if stop { None } else { Some(token) })
    }
}

impl Drop for CudaCompute {
    fn drop(&mut self) {
        // Destroy the decode graph (the captured graph + the leaf-owned
        // capture stream + the device staging buffers + the H2D'd weight
        // copies, the host case); NULL is a no-op (the eager fallback,
        // ADR 0006).
        if let Some(SendDecodeGraph(g)) = *self.graph.lock().unwrap() {
            unsafe { ffi::ignis_decode_graph_free(g) };
        }
    }
}

impl std::fmt::Debug for CudaCompute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaCompute")
            .field("config", &self.config)
            .field("graph", &"opaque")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// The artifact path (feature `cuda`): a real (dequantized) model from a
// `.ninfer` container (ADR 0002) — the compute-adapter's production
// constructor.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
impl CudaCompute {
    /// Construct a compute backend from a `.ninfer` artifact (the
    /// production path, A3 / #30): open the container (ADR 0002),
    /// materialize the model to the `CudaDevice` (VRAM), route the real
    /// normalized tensors into the `Weights` (the full-correct forward
    /// assembly — the NVFP4 GEMM planes stay device-resident (the
    /// `*_device` kernels, no per-call H2D, the #26 fix), the BF16
    /// tensors are host-copied (the bounded text-scope copy — the
    /// artifact's `gdn/convolution` + `gdn/a_b_projection` + the
    /// norms + the BF16-exception projections), the W8 endpoints are
    /// the A1 host-side dequants (the embedding table + the lm_head)),
    /// and build the backend eagerly (`new_eager` — no startup graph
    /// check / CUDA-graph capture; the graph fast path is #25's 99%-gate
    /// material).
    ///
    /// The forward pass (the `forward_layers` assembly, A3 / #30) runs
    /// the *real* model (the `qwen38_27b` topology — the 16 GQA + 48
    /// GDN layers, the real head geometry + the GDN state dims + the
    /// rotary geometry): the GDN layers' causal conv + the GQA layers'
    /// q / k RMSNorm + RoPE (kernel-abi 06, A2 / #28), the bf16 logits
    /// GEMM (kernel-abi 10, A2b / #29), and the device-resident NVFP4
    /// routing (the #26 `_device` surface) — the numerically-correct
    /// forward pass (the correctness floor, ADR 0005: a *sane*,
    /// reproducible output — the "for now" ported kernels, re-implemented
    /// later, ADR 0005 / 0007).
    pub fn from_artifact(
        path: &std::path::Path,
        model: &str,
    ) -> Result<Self, ComputeError> {
        use ignis_artifact::{Binding, Binder, CudaDevice, Object, materialize};
        // Open + read the container (ADR 0002: the binder must consume every
        // object — an unconsumed object is a load failure).
        let reader =
            Reader::open(path).map_err(|_| ComputeError::Kernel(-1))?;
        // Bind every object (ADR 0002: the binder must consume every object
        // — an unconsumed object is a load failure) -> the materialization
        // plan (the per-object device placement).
        let mut binder = Binder::new(&reader);
        for object in reader.objects() {
            match object {
                Object::Tensor(t) => {
                    let handle = binder
                        .require_tensor(&t.name, t.format, t.layout, &t.shape)
                        .map_err(|_| ComputeError::Kernel(-1))?;
                    binder
                        .materialize_on_device(handle)
                        .map_err(|_| ComputeError::Kernel(-1))?;
                }
                Object::Resource(r) => {
                    let handle = binder
                        .require_resource(&r.name, r.encoding)
                        .map_err(|_| ComputeError::Kernel(-1))?;
                    binder
                        .retain_on_host(handle)
                        .map_err(|_| ComputeError::Kernel(-1))?;
                }
            }
        }
        let plan = binder.finish().map_err(|_| ComputeError::Kernel(-1))?;

        // The #26 fix: materialize the artifact on a `CudaDevice` (the 19 GB
        // of weights land in VRAM — ADR 0002: `.ninfer` artifacts load
        // directly, no converter). The device + the materialized artifact +
        // the reader are held in `device_ctx` so the VRAM arena outlives the
        // forward pass (ADR 0002: the producing device must outlive the
        // materialized artifact and its typed views).
        let mut device =
            CudaDevice::create(0).map_err(|_| ComputeError::Kernel(-1))?;
        let artifact = materialize(&reader, &plan, &mut device, None)
            .map_err(|_| ComputeError::Kernel(-1))?;

        // The crash fix: the real model's topology (no synthetic fallback,
        // so the embedding table has the real vocab and a real tokenizer's
        // ids never index out of bounds — the `illegal memory access`).
        // The #27 A1 normalization seam: the `from_artifact` `Weights` are
        // the real normalized buffers (spec 04 criterion 3) — the two W8
        // text-scope endpoints (`text/token_embedding` + the
        // `text/output_head`) are host-dequantized to bf16 (ADR 0005,
        // ~5 GB) and routed into the `Weights` (the embedding table + the
        // lm_head), while the NVFP4 GEMM planes stay device-resident
        // (the `from_geometry`'s geometry-only content — not host-copied,
        // the #26 lesson: no host weight explosion on the load path).
        // The A3 / #30 assembly: the full-correct forward routing — the
        // per-layer GEMM slots are routed to the artifact's real tensors
        // (the NVFP4 fused tensors' device planes via the `*_device`
        // kernels, the BF16 tensors host-copied — the artifact's
        // directory facts, A1's inventory: the early GQA layers'
        // `attention/query_key_gate_value` + `attention/output` are
        // BF16, the layer-4 `gdn/output` quirk, the GDN layers'
        // `gdn/convolution` + `gdn/a_b_projection` + the norms).
        let config = ModelConfig::qwen38_27b();
        let endpoints =
            dequant_w8_endpoints(&reader).map_err(|_| ComputeError::Kernel(-1))?;
        let mut weights = Weights::from_geometry(&config).with_w8_endpoints(endpoints);
        // The artifact's device views (the NVFP4 planes' raw pointers,
        // the `Nvfp4DevicePlane`'s routing, A3 / #30) + the host-side
        // reads (the BF16 tensors' bounded copies — the container's
        // mmap'd bytes, the `Reader`'s payload spans, ADR 0002).
        let binding = Binding::new(&reader, &artifact);
        // The per-layer routing (the A1 inventory's directory facts —
        // layer `i` is GQA iff `(i + 1) % 4 == 0`, the model constant).
        for (i, lw) in weights.per_layer.iter_mut().enumerate() {
            let prefix = format!("text/layers/{i}/");
            if (i + 1) % 4 == 0 {
                // ── the GQA layer: the fused qkvz + the output
                // projections (the NVFP4 device plane / the BF16
                // exception — the A1 inventory's directory facts,
                // A3 / #30).
                let qkv_name = format!("{prefix}attention/query_key_gate_value");
                match binding.nvfp4(&qkv_name) {
                    Ok(v) => lw.dev.qkv = Some(Nvfp4DevicePlane {
                        codes: v.code,
                        scales: v.scale,
                        m: v.rows,
                        k: v.cols,
                    }),
                    // The BF16-exception (the early GQA layers — the
                    // A1 inventory's `QKGV_BF16_LAYERS`): the host copy
                    // (the `ignis_bf16_gemm` kernel, A2b / #29).
                    Err(_) => {
                        lw.qkv_bf16 =
                            Self::host_bf16_matrix(&reader, &qkv_name)?;
                    }
                }
                let out_name = format!("{prefix}attention/output");
                match binding.nvfp4(&out_name) {
                    Ok(v) => lw.dev.attn_out = Some(Nvfp4DevicePlane {
                        codes: v.code,
                        scales: v.scale,
                        m: v.rows,
                        k: v.cols,
                    }),
                    // The BF16-exception (the early GQA layers — the
                    // A1 inventory's `ATTENTION_OUT_BF16_LAYERS`).
                    Err(_) => {
                        lw.attn_out_bf16 =
                            Self::host_bf16_matrix(&reader, &out_name)?;
                    }
                }
                // The q / k RMSNorm weights (the artifact's
                // `attention/query_norm` + `attention/key_norm`, the
                // per-head `[head_dim]` bf16, A3 / #30).
                let qn = format!("{prefix}attention/query_norm");
                let kn = format!("{prefix}attention/key_norm");
                let q_norm = Self::host_bf16_vector(&reader, &qn)?;
                let k_norm = Self::host_bf16_vector(&reader, &kn)?;
                lw.qk_norm = [q_norm, k_norm];
            } else {
                // ── the GDN layer: the input projection (the NVFP4
                // `gdn/query_key_value_z`) + the causal conv + the a / b
                // (gate / beta) projection + the state readout (the
                // NVFP4 / the layer-4 BF16 quirk, A3 / #30).
                let in_name = format!("{prefix}gdn/query_key_value_z");
                match binding.nvfp4(&in_name) {
                    Ok(v) => lw.dev.gdn_in = Some(Nvfp4DevicePlane {
                        codes: v.code,
                        scales: v.scale,
                        m: v.rows,
                        k: v.cols,
                    }),
                    Err(_) => {
                        return Err(ComputeError::Kernel(-1));
                    }
                }
                let out_name = format!("{prefix}gdn/output");
                match binding.nvfp4(&out_name) {
                    Ok(v) => lw.dev.gdn_out = Some(Nvfp4DevicePlane {
                        codes: v.code,
                        scales: v.scale,
                        m: v.rows,
                        k: v.cols,
                    }),
                    // The layer-4 quirk (the A1 inventory's
                    // `GDN_OUT_BF16_LAYERS`): the host copy.
                    Err(_) => {
                        lw.gdn_out_bf16 =
                            Self::host_bf16_matrix(&reader, &out_name)?;
                    }
                }
                // The GDN causal-conv weight (the artifact's
                // `gdn/convolution`, bf16 `[4][channels]` tap-major,
                // A3 / #30).
                let conv_name = format!("{prefix}gdn/convolution");
                lw.gdn_conv  = Self::host_bf16_matrix(&reader, &conv_name)?;
                // The GDN a / b (gate / beta) projection (the artifact's
                // `gdn/a_b_projection`, bf16 `[ab][hidden]`, A3 / #30).
                let ab_name = format!("{prefix}gdn/a_b_projection");
                lw.gdn_ab  = Self::host_bf16_matrix(&reader, &ab_name)?;
            }
            // The per-layer RMSNorm weights (the artifact's
            // `input_norm` / `post_attention_norm`, bf16 `[hidden]`,
            // A3 / #30).
            lw.norm_in = Self::host_bf16_vector(&reader, &format!("{prefix}input_norm"))?;
            lw.norm_post =
                Self::host_bf16_vector(&reader, &format!("{prefix}post_attention_norm"))?;
            // The fused FFN gate+up projection (the NVFP4 device plane —
            // the gate / up slots are row slices, A3 / #30).
            let gate_up = format!("{prefix}mlp/gate_up");
            match binding.nvfp4(&gate_up) {
                Ok(v) => lw.dev.mlp_gate_up = Some(Nvfp4DevicePlane {
                    codes: v.code,
                    scales: v.scale,
                    m: v.rows,
                    k: v.cols,
                }),
                Err(_) => {
                    return Err(ComputeError::Kernel(-1));
                }
            }
            // The FFN down projection (the NVFP4 device plane, A3 / #30).
            let down = format!("{prefix}mlp/down");
            match binding.nvfp4(&down) {
                Ok(v) => lw.dev.mlp_down = Some(Nvfp4DevicePlane {
                    codes: v.code,
                    scales: v.scale,
                    m: v.rows,
                    k: v.cols,
                }),
                Err(_) => {
                    return Err(ComputeError::Kernel(-1));
                }
            }
        }
        // The final-norm weight (the artifact's `text/final_norm`, bf16
        // `[hidden]`, A3 / #30).
        weights.final_norm = Self::host_bf16_vector(&reader, "text/final_norm")?;
        // B2 (#32): the decode graph (the production path — the same
        // mechanism as the synthetic path, ADR 0008: the fixed-address
        // device staging buffers + the captured representative decode
        // sequence). The construction-time capture self-skips on a VRAM
        // shortfall (a 27B decode graph's staging — the paged KV + the GDN
        // state + the H2D'd bf16 weight copies — does not fit alongside the
        // 19 GB of weights on a VRAM-constrained GPU, the eager fallback,
        // ADR 0006) or a busy/absent GPU; the eager hybrid `decode()` is
        // the correctness floor (ADR 0005).
        let decode_graph = Self::build_decode_graph(&config, &weights);
        // The #26 fix: the eager construction (the 19 GB of weights land in
        // VRAM, ADR 0002) — the decode graph (B2 / #32) is layered on top
        // (the construction-time capture, self-skipping on a VRAM shortfall).
        let mut compute = Self::new_eager(config, weights);
        // B2 (#32): the decode graph (the decode hot path, ADR 0008) — the
        // `GraphGeometry` is set (batch 1, the representative decode
        // geometry); a self-skip (a VRAM shortfall, ADR 0006) leaves the
        // eager fallback (the `graph_geom` stays `None`).
        if let Some(handle) = decode_graph {
            *compute.graph.lock().unwrap() = Some(SendDecodeGraph(handle));
            compute.graph_geom = Some(GraphGeometry { batch: 1 });
        }
        // Hold the device context (the 19 GB of weights stay in VRAM for the
        // lifetime of the backend, ADR 0002).
        compute.device_ctx = Some(DeviceCtx::Cuda {
            reader,
            artifact,
            device,
        });
        let _ = model;
        Ok(compute)
    }

    /// The artifact's host-side bf16 copy of a rank-2 tensor (A3 / #30):
    /// the container's raw bf16 bytes (the `contiguous-le-v1` layout,
    /// row-major `[m][k]` bf16 words) copied to the host (the bounded
    /// text-scope copy — the BF16-exception projections + the GDN conv /
    /// a-b tensors + the norms; the `ignis_bf16_gemm` kernel, A2b / #29,
    /// consumes the host buffers). `name`'s shape is `[m][k]` (the
    /// artifact's directory, ADR 0002).
    fn host_bf16_matrix(reader: &Reader, name: &str) -> Result<Bf16Weight, ComputeError> {
        use ignis_artifact::{NumericFormat, Object};
        let tensor = reader
            .find(name)
            .ok_or(ComputeError::Kernel(-1))?;
        let tensor = match tensor {
            Object::Tensor(t) => t,
            Object::Resource(_) => return Err(ComputeError::Kernel(-1)),
        };
        if tensor.format != NumericFormat::Bf16 || tensor.shape.len() != 2 {
            return Err(ComputeError::Kernel(-1));
        }
        let (m, k) = (tensor.shape[0], tensor.shape[1]);
        let span = reader.payload(name).map_err(|_| ComputeError::Kernel(-1))?;
        if span.data.len() != (m * k * 2) as usize {
            return Err(ComputeError::Kernel(-1));
        }
        let mut data = vec![0u16; (m * k) as usize];
        for i in 0..data.len() {
            data[i] = u16::from_le_bytes([span.data[2 * i], span.data[2 * i + 1]]);
        }
        Ok(Bf16Weight { data, m, k })
    }

    /// The artifact's host-side bf16 copy of a rank-1 vector (A3 / #30):
    /// the norm weights (`input_norm` / `post_attention_norm` /
    /// `text/final_norm` / the q / k RMSNorm weights — the artifact's
    /// bf16 `[width]` vectors, the `contiguous-le-v1` layout).
    fn host_bf16_vector(reader: &Reader, name: &str) -> Result<Vec<u16>, ComputeError> {
        use ignis_artifact::{NumericFormat, Object};
        let tensor = reader
            .find(name)
            .ok_or(ComputeError::Kernel(-1))?;
        let tensor = match tensor {
            Object::Tensor(t) => t,
            Object::Resource(_) => return Err(ComputeError::Kernel(-1)),
        };
        if tensor.format != NumericFormat::Bf16 || tensor.shape.len() != 1 {
            return Err(ComputeError::Kernel(-1));
        }
        let n = tensor.shape[0] as usize;
        let span = reader.payload(name).map_err(|_| ComputeError::Kernel(-1))?;
        if span.data.len() != n * 2 {
            return Err(ComputeError::Kernel(-1));
        }
        let mut data = vec![0u16; n];
        for i in 0..n {
            data[i] = u16::from_le_bytes([span.data[2 * i], span.data[2 * i + 1]]);
        }
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic (deterministic) backend: the self-consistency floor
    /// (greedy + fixed seed, ADR 0007). A non-GPU host still constructs a
    /// (correct, eager) backend (the startup capture self-skips, ADR 0006).
    #[test]
    fn synthetic_backend_constructs() {
        let cfg = ModelConfig::synthetic();
        let _ = CudaCompute::new(cfg.clone(), Weights::synthetic(&cfg, 0));
    }

    #[test]
    fn synthetic_weights_are_deterministic() {
        let cfg = ModelConfig::synthetic();
        let a = Weights::synthetic(&cfg, 7);
        let b = Weights::synthetic(&cfg, 7);
        assert_eq!(a.lm_head.dims(), b.lm_head.dims());
        assert_eq!(
            a.lm_head.as_nvfp4().unwrap().codes.len(),
            b.lm_head.as_nvfp4().unwrap().codes.len()
        );
        assert_eq!(a.embedding.len(), b.embedding.len());
        assert_eq!(a.embedding.len(), (cfg.vocab * cfg.hidden) as usize);
        assert_eq!(a.per_layer.len(), cfg.num_layers);
    }

    /// The pointwise glue (the correctness floor, ADR 0005): the residual
    /// add + the gated-SiLU activation are deterministic (fixed-seed).
    #[test]
    fn pointwise_glue_is_deterministic() {
        let mut acc = to_bf16(&[1.0, 2.0, 3.0]);
        let delta = to_bf16(&[0.5, -1.0, 4.0]);
        CudaCompute::residual(&mut acc, &delta);
        let want = to_bf16(&[1.5, 1.0, 7.0]);
        for (a, w) in acc.iter().zip(want.iter()) {
            assert!((bf16_to_f32(*a) - bf16_to_f32(*w)).abs() < 0.02);
        }
        let up = to_bf16(&[1.0, -2.0]);
        let gate = to_bf16(&[3.0, -1.0]);
        let act = CudaCompute::silu_mul(&up, &gate);
        assert_eq!(act.len(), 2);
    }

    /// The #26 crash fix (the red-capable CPU signal): the artifact path's
    /// config is the real Qwen 3.8-27B topology (not the synthetic fallback).
    /// This pins the fix — the embedding table has the real vocab (248 320),
    /// so a real tokenizer's ids (up to 248 077) never index out of bounds
    /// (the `ignis_embedding` OOB that produced the `illegal memory access`).
    #[test]
    fn qwen38_27b_config_is_the_real_topology() {
        let cfg = ModelConfig::qwen38_27b();
        assert_eq!(cfg.vocab, 248_320, "the real vocab (not the synthetic 256)");
        assert_eq!(cfg.hidden, 5120);
        assert_eq!(cfg.num_layers, 64);
        assert_eq!(cfg.num_q_heads, 24);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.ffn_intermediate, 17_408);
        assert_eq!(cfg.gdn_state_rows, 6144);
        assert_eq!(cfg.gdn_state_cols, 2048);
        assert_eq!(cfg.gdn_num_layers, 48);
        assert_eq!(cfg.block_size, 64);
        // The A3 / #30 real-model geometry: the GDN input projection's
        // q / z / a-b parts (the artifact's `gdn/query_key_value_z`
        // layout — q 2048 + k 2048 + v 6144 + z 6144 = 16 384 rows, the
        // `gdn/a_b_projection` is 96 rows), the rotary geometry (the
        // GQA RoPE — `rotary_dim` 64 of `head_dim` 256, θ = 1e7,
        // kernel-abi 06, A2 / #28).
        assert_eq!(cfg.gdn_q_width, 2048, "the GDN q part (the GDN q rows)");
        assert_eq!(cfg.gdn_z_width, 6144, "the GDN z (output-gate) part");
        assert_eq!(cfg.gdn_ab_width, 96, "the GDN a / b (gate / beta) projection (48 a + 48 b)");
        assert_eq!(cfg.rotary_dim, 64, "the GQA rotary dim (64 of 256)");
        assert_eq!(cfg.rope_theta, 1e7, "the RoPE base θ (1e7)");
        // The derived A3 / #30 geometry: the GDN in-projection rows
        // (q + k + v + z), the conv channels (q + k + v), the state
        // readout width (state_rows), the RoPE pair count.
        assert_eq!(
            cfg.gdn_in_proj_m(),
            16_384,
            "the GDN in-projection (16 384 rows = q + k + v + z)"
        );
        assert_eq!(
            cfg.gdn_conv_channels(),
            10_240,
            "the GDN conv channels (10 240 = q + k + v)"
        );
        assert_eq!(cfg.gdn_readout_k(), 6144, "the GDN readout width (state_rows)");
        assert_eq!(cfg.rope_pairs(), 32, "the RoPE pair count (rotary_dim / 2)");
        // 16 GQA + 48 GDN layers (layer i is GQA iff (i + 1) % 4 == 0).
        let gqa = cfg.layer_kinds.iter().filter(|k| **k == LayerKind::Gqa).count();
        let gdn = cfg.layer_kinds.iter().filter(|k| **k == LayerKind::Gdn).count();
        assert_eq!(gqa, 16, "16 GQA (full-attention) layers");
        assert_eq!(gdn, 48, "48 GDN (linear-attention) layers");
        assert_eq!(cfg.layer_kinds.len(), cfg.num_layers);
    }

    /// The synthetic fallback is NOT the artifact path's topology (the #26
    /// crash fix): `from_artifact` uses `qwen38_27b` (the real geometry), so
    /// the embedding table is the real vocab (248 320), never the synthetic
    /// 256 (which made a real tokenizer's ids index out of bounds).
    #[test]
    fn synthetic_config_is_not_the_artifact_topology() {
        let real = ModelConfig::qwen38_27b();
        let synth = ModelConfig::synthetic();
        assert_ne!(
            real.vocab, synth.vocab,
            "the artifact path must use the real vocab (248 320), not the synthetic 256"
        );
        assert_ne!(real.num_layers, synth.num_layers);
    }

    /// (A3 / #30, spec 07 acceptance criterion 4) the full layer-stack
    /// composition: a synthetic-model CPU test verifying the GQA + GDN +
    /// FFN layer composition (the causal conv + the RoPE + the q / k
    /// RMSNorm + the a / b projection are on the *right* layers — the
    /// GDN layers' conv / a-b, the GQA layers' q / k norms + RoPE — and
    /// the per-layer GEMM geometry matches the config). The runtime
    /// *call* of the conv / RoPE / qk-norm primitives is exercised by
    /// the GPU e2e test (`kernel_abi04_gpu`, ADR 0006 — the CPU cannot
    /// launch kernels); this pins the composition + the geometry at the
    /// weights level (CPU-runnable, the "conv + RoPE are called for the
    /// right layers, the geometry matches" acceptance).
    #[test]
    fn full_stack_synthetic_composition() {
        let cfg = ModelConfig::synthetic(); // [Gdn, Gqa] — both layer kinds
        let w = Weights::synthetic(&cfg, 1);
        let (gdn, gqa) = (&w.per_layer[0], &w.per_layer[1]);
        assert_eq!(gdn.kind, LayerKind::Gdn);
        assert_eq!(gqa.kind, LayerKind::Gqa);
        // ── the GDN layer: the causal conv + the a / b projection are
        // GDN-only (the conv'd q / k / v part — the z rows bypass it).
        assert!(
            !gdn.gdn_conv.data.is_empty(),
            "the GDN layer carries the causal-conv weight (4 taps × the conv channels)"
        );
        assert_eq!(gdn.gdn_conv.m, 4, "the conv has 4 taps (the artifact's gdn/convolution)");
        assert_eq!(
            gdn.gdn_conv.k,
            cfg.gdn_conv_channels(),
            "the conv channels = the conv'd q / k / v part (the z rows bypass it)"
        );
        assert_eq!(
            gdn.gdn_ab.m,
            cfg.gdn_ab_width,
            "the a / b projection width (0 for the synthetic model — the step's g / beta are 0)"
        );
        // The GDN layer has no q / k RMSNorm (the GQA-only op, A3 / #30).
        assert!(
            gdn.qk_norm[0].is_empty() && gdn.qk_norm[1].is_empty(),
            "the GDN layer has no q / k RMSNorm"
        );
        // The GDN layer's per-layer index (the state / conv-state slice,
        // the request state's per-layer planes).
        assert_eq!(gdn.gdn_index, 0, "the synthetic model's single GDN layer is index 0");
        // ── the GQA layer: the q / k RMSNorm + the RoPE geometry are
        // GQA-only (the causal conv / the a / b are GDN-only).
        assert!(
            gqa.gdn_conv.data.is_empty(),
            "the GQA layer has no causal conv (the GDN-only op)"
        );
        assert_eq!(gqa.gdn_ab.m, 0, "the GQA layer has no a / b projection");
        assert_eq!(
            gqa.qk_norm[0].len(),
            cfg.head_dim as usize,
            "the GQA q RMSNorm weight is [head_dim] (the per-head norm)"
        );
        assert_eq!(
            gqa.qk_norm[1].len(),
            cfg.head_dim as usize,
            "the GQA k RMSNorm weight is [head_dim] (the per-head norm)"
        );
        // ── the RoPE geometry: the inv-freq table = `rotary_dim / 2`
        // pairs (the kernel's `ignis_rope_qk` contract, A3 / #30).
        let freqs = rope_inv_frequencies(cfg.rope_theta, cfg.rotary_dim as i64);
        assert_eq!(
            freqs.len(),
            cfg.rope_pairs() as usize,
            "the RoPE inv-freq table has `rotary_dim / 2` pairs"
        );
        // ── the per-layer GEMM geometry (config-driven, A3 / #30):
        // GDN: the in-projection (m = the q/k/v/z feature rows, k =
        // hidden) + the state readout (m = hidden, k = the readout
        // width `state_rows`); GQA: q / k / v + the output projection.
        assert_eq!(
            (gdn.projection[0].m, gdn.projection[0].k),
            (cfg.gdn_in_proj_m(), cfg.hidden),
            "the GDN in-projection geometry (m = the feature rows, k = hidden)"
        );
        assert_eq!(
            (gdn.gdn_output.m, gdn.gdn_output.k),
            (cfg.hidden, cfg.gdn_readout_k()),
            "the GDN state readout geometry (m = hidden, k = the readout width)"
        );
        assert_eq!(
            (gqa.projection[0].m, gqa.projection[0].k),
            (cfg.gqa_width(), cfg.hidden),
            "the GQA q projection geometry (m = the q width, k = hidden)"
        );
        assert_eq!(
            (gqa.projection[1].m, gqa.projection[1].k),
            (cfg.gqa_kv_width(), cfg.hidden),
            "the GQA k projection geometry (m = the kv width, k = hidden)"
        );
        assert_eq!(
            (gqa.projection[3].m, gqa.projection[3].k),
            (cfg.hidden, cfg.gqa_width()),
            "the GQA output projection geometry (m = hidden, k = the q width)"
        );
        // The real-model geometry matches `qwen38_27b` (the 27B
        // topology's per-layer A3 geometry — the `weights_geometry_
        // matches_27b_topology` test pins it; here: the synthetic and
        // the real share the *same* per-layer-kind geometry *rule* —
        // the conv / a-b / qk-norm are present exactly for the right
        // layer kind).
        let real = ModelConfig::qwen38_27b();
        let g = WeightsGeometry::from_config(&real);
        let real_gdn = g.per_layer.iter().find(|l| l.kind == LayerKind::Gdn).unwrap();
        let real_gqa = g.per_layer.iter().find(|l| l.kind == LayerKind::Gqa).unwrap();
        assert_eq!(
            real_gdn.gdn_conv,
            (4, real.gdn_conv_channels()),
            "the 27B GDN conv geometry (4 taps × 10 240 channels)"
        );
        assert_eq!(
            real_gdn.gdn_ab,
            (real.gdn_ab_width, real.hidden),
            "the 27B GDN a / b projection geometry (96 × 5 120)"
        );
        assert_eq!(real_gqa.qk_norm, real.head_dim, "the 27B GQA q / k RMSNorm width (256)");
        assert_eq!(
            real_gdn.gdn_in_proj,
            (real.gdn_in_proj_m(), real.hidden),
            "the 27B GDN in-projection (16 384 × 5 120)"
        );
        assert_eq!(
            real_gdn.gdn_output,
            (real.hidden, real.gdn_readout_k()),
            "the 27B GDN state readout (5 120 × 6 144)"
        );
    }

    /// (GitHub #33) The kernel's GEMM convention (m = output rows, k = input
    /// dim — per the `Nvfp4Weight` doc, the `ignis_nvfp4_gemm_*` FFI, and
    /// the artifact's shape table) is pinned on an **asymmetric** topology:
    /// every GEMM (m, k) pair is (output dim, input dim), never
    /// (input, output). The synthetic config's coincident dimensions
    /// (hidden = 64 = gqa_width) hid the #33 transposition (the tests
    /// asserted determinism, not semantics); here every width is distinct,
    /// so any transposed construction is caught. The pin covers both
    /// derivations: `Weights::synthetic` (the content) and
    /// `WeightsGeometry::from_config` (the artifact path's geometry),
    /// which must agree.
    #[test]
    fn gemm_pairs_follow_the_kernel_output_by_input_convention() {
        // An asymmetric topology: distinct GEMM widths (hidden 96, q 48,
        // kv 16, ffn 32, the GDN feature 42 / state matrix 384, vocab
        // 256) — a transposed (input, output) pair always mismatches.
        let cfg = ModelConfig {
            num_layers: 2,
            layer_kinds: vec![LayerKind::Gdn, LayerKind::Gqa],
            hidden: 96,
            vocab: 256,
            num_q_heads: 3,
            num_kv_heads: 1,
            head_dim: 16,
            gdn_state_rows: 24,
            gdn_state_cols: 16,
            gdn_num_layers: 1,
            // The A3 / #30 geometry: no GDN q / z / a-b parts (the
            // synthetic in-projection is the k / v / g / beta feature
            // directly), the RoPE geometry (rotary_dim 8 of 16, θ = 1e7).
            gdn_q_width: 0,
            gdn_z_width: 0,
            gdn_ab_width: 0,
            rotary_dim: 8,
            rope_theta: 1e7,
            ffn_intermediate: 32,
            block_size: 4,
            num_blocks: 8,
        };
        // ── the synthetic weights (the content + the (m, k) geometry) ──
        let w = Weights::synthetic(&cfg, 1);
        let (gdn, gqa) = (&w.per_layer[0], &w.per_layer[1]);
        // GDN: the input projection (m = the GDN feature rows — the
        // q / k / v / z parts, `gdn_in_proj_m` — k = hidden) + the state
        // readout (m = hidden, k = the per-token readout width
        // `state_rows`, A3 / #30).
        assert_eq!(
            (gdn.projection[0].m, gdn.projection[0].k),
            (cfg.gdn_in_proj_m(), cfg.hidden),
            "the GDN input projection (m = output, k = input)"
        );
        assert_eq!(
            (gdn.gdn_output.m, gdn.gdn_output.k),
            (cfg.hidden, cfg.gdn_readout_k()),
            "the GDN state readout (m = output, k = input)"
        );
        // GQA: q / k / v map hidden -> the head widths (m = output,
        // k = input); the output projection maps gqa_width -> hidden.
        assert_eq!(
            (gqa.projection[0].m, gqa.projection[0].k),
            (cfg.gqa_width(), cfg.hidden),
            "the q projection (m = output, k = input)"
        );
        assert_eq!(
            (gqa.projection[1].m, gqa.projection[1].k),
            (cfg.gqa_kv_width(), cfg.hidden),
            "the k projection (m = output, k = input)"
        );
        assert_eq!(
            (gqa.projection[2].m, gqa.projection[2].k),
            (cfg.gqa_kv_width(), cfg.hidden),
            "the v projection (m = output, k = input)"
        );
        assert_eq!(
            (gqa.projection[3].m, gqa.projection[3].k),
            (cfg.hidden, cfg.gqa_width()),
            "the attention-output projection (m = output, k = input)"
        );
        // The gated-FFN: gate / up map hidden -> ffn, down maps
        // ffn -> hidden (both layer kinds).
        for lw in [gdn, gqa] {
            assert_eq!(
                (lw.ffn_gate.m, lw.ffn_gate.k),
                (cfg.ffn_intermediate, cfg.hidden),
                "the ffn gate projection (m = output, k = input)"
            );
            assert_eq!(
                (lw.ffn_up.m, lw.ffn_up.k),
                (cfg.ffn_intermediate, cfg.hidden),
                "the ffn up projection (m = output, k = input)"
            );
            assert_eq!(
                (lw.ffn_down.m, lw.ffn_down.k),
                (cfg.hidden, cfg.ffn_intermediate),
                "the ffn down projection (m = output, k = input)"
            );
        }
        // The logits GEMM (the lm_head): m = vocab (output), k = hidden
        // (input) — the same (vocab, hidden) the #27 `from_config` fix
        // established (the artifact's `text/output_head` shape).
        assert_eq!(
            w.lm_head.dims(),
            (cfg.vocab, cfg.hidden),
            "the lm_head (m = output, k = input)"
        );
        // The code / scale planes are sized to the weight's own (m, k)
        // (rows = m; a code row = k/2 bytes, a scale row = k/16) — these
        // check the planes agree with the geometry; the convention pin is
        // the (m, k) field asserts above (a plane-size product is
        // orientation-symmetric, so it alone would not catch a
        // transposition).
        assert_eq!(
            gqa.projection[0].codes.len(),
            gqa.projection[0].m as usize * (gqa.projection[0].k as usize) / 2,
            "the codes plane is sized [m][k/2] to the weight's geometry"
        );
        assert_eq!(
            gqa.projection[0].scales.len(),
            gqa.projection[0].m as usize * (gqa.projection[0].k as usize) / 16,
            "the scales plane is sized [m][k/16] to the weight's geometry"
        );
        // ── the geometry derivation (the artifact path, `from_config`) ──
        let g = WeightsGeometry::from_config(&cfg);
        let (gl, gw) = (&g.per_layer[0], &g.per_layer[1]);
        // The GDN state readout (m = hidden, k = the per-token readout
        // width `state_rows`) + the input projection (m = the GDN feature
        // rows — the q / k / v / z parts, `gdn_in_proj_m` — k = hidden),
        // A3 / #30.
        assert_eq!(gl.gdn_output, (cfg.hidden, cfg.gdn_readout_k()));
        assert_eq!(gl.projection[0], (cfg.gdn_in_proj_m(), cfg.hidden));
        // The A3 / #30 geometry: the GDN causal-conv (m = 4 taps,
        // k = the conv channels), the a / b projection (0 when the
        // model has no a-b), the GQA q / k RMSNorm width (`head_dim`),
        // the conv channel count + the in-projection (m, k).
        assert_eq!(
            gl.gdn_conv,
            (4, cfg.gdn_conv_channels()),
            "the GDN causal-conv geometry (4 taps × the conv channels)"
        );
        assert_eq!(
            gl.gdn_ab,
            (0, 0),
            "no a / b projection (gdn_ab_width = 0) — (0, 0)"
        );
        assert_eq!(gl.qk_norm, 0, "the GDN layer has no q / k RMSNorm");
        assert_eq!(
            gl.gdn_conv_channels,
            cfg.gdn_conv_channels(),
            "the GDN conv channel count"
        );
        assert_eq!(
            gl.gdn_in_proj,
            (cfg.gdn_in_proj_m(), cfg.hidden),
            "the GDN input-projection (m, k)"
        );
        assert_eq!(
            gw.qk_norm,
            cfg.head_dim,
            "the GQA q / k RMSNorm weights are [head_dim] each"
        );
        assert_eq!(gw.gdn_conv, (0, 0), "the GQA layer has no GDN conv");
        assert_eq!(gw.gdn_conv_channels, 0, "the GQA layer has no conv");
        assert_eq!(gw.gdn_in_proj, (0, 0), "the GQA layer has no GDN in-proj");
        assert_eq!(
            gw.projection,
            [
                (cfg.gqa_width(), cfg.hidden),
                (cfg.gqa_kv_width(), cfg.hidden),
                (cfg.gqa_kv_width(), cfg.hidden),
                (cfg.hidden, cfg.gqa_width()),
            ],
            "the GQA projections (m = output, k = input)"
        );
        for lg in [gl, gw] {
            assert_eq!(
                (lg.ffn_gate, lg.ffn_up),
                (
                    (cfg.ffn_intermediate, cfg.hidden),
                    (cfg.ffn_intermediate, cfg.hidden)
                ),
                "the ffn gate / up (m = output, k = input)"
            );
            assert_eq!(
                lg.ffn_down,
                (cfg.hidden, cfg.ffn_intermediate),
                "the ffn down (m = output, k = input)"
            );
        }
        assert_eq!(g.lm_head, (cfg.vocab, cfg.hidden));
        // The geometry agrees with the synthetic content's (m, k) pairs
        // (the `from_config` doc: the same derivation, minus the content).
        assert_eq!(
            (gw.projection[0].0, gw.projection[0].1),
            (gqa.projection[0].m, gqa.projection[0].k),
            "the geometry + the synthetic content agree"
        );
        assert_eq!(
            (gl.gdn_output.0, gl.gdn_output.1),
            (gdn.gdn_output.m, gdn.gdn_output.k),
            "the geometry + the synthetic content agree"
        );
    }

    /// (kernel-abi 06, GitHub #28) The RoPE inverse-frequency table (θ = 1e7,
    /// rotary_dim = 64 — the Qwen 3.8-27B GQA geometry) is pinned to the
    /// deterministic values the `ignis_rope_qk` kernel consumes: pair `p` is
    /// θ^(-2p/64) (the reference's `rope_linear_frequencies` table, computed
    /// in f64 and rounded to f32). The table is computed once at construction
    /// (host-side, a deterministic table — a non-goal is the per-step
    /// recompute), so its values are pinned here by exact f32 bits (the
    /// independently computed literals: θ=1e7, R=64 → 32 pairs).
    #[test]
    fn rope_inv_frequencies_pins_the_theta_1e7_table() {
        let freqs = rope_inv_frequencies(1e7, 64);
        assert_eq!(freqs.len(), 32, "rotary_dim 64 -> 32 pairs");
        assert_eq!(freqs[0], 1.0f32, "pair 0: θ^0 = 1 (the exact 1.0f32 bits)");
        // The table endpoints + a mid-table pin (independent f32 literals).
        assert_eq!(freqs[1].to_bits(), 0x3f1ab32b, "pair 1: θ^(-1/32)");
        assert_eq!(freqs[15].to_bits(), 0x3a092e02, "pair 15: θ^(-15/32)");
        assert_eq!(freqs[31].to_bits(), 0x3431af44, "pair 31: θ^(-31/32)");
        // The table is strictly decreasing (θ^(-2p/R), p increasing).
        assert!(
            freqs.windows(2).all(|w| w[0] > w[1]),
            "the inv_freq table must be strictly decreasing"
        );
    }
    /// The #27 A1 normalization seam (spec 04's acceptance: "the
    /// `from_artifact` `Weights` are the real normalized buffers — a
    /// non-GPU / CPU test verifies the `Weights` geometry matches the 27B
    /// topology"): the geometry is pure, config-derived, and matches
    /// `ModelConfig::qwen38_27b` — the embedding / lm_head dims are the
    /// real vocab × hidden, the 16 GQA + 48 GDN layer-kind order is
    /// preserved, and the GEMM (m, k) pairs follow the `synthetic`
    /// derivation (the forward pass's established convention).
    #[test]
    fn weights_geometry_matches_27b_topology() {
        let cfg = ModelConfig::qwen38_27b();
        let g = WeightsGeometry::from_config(&cfg);
        // The 27B topology's endpoints (spec 04: the W8 embedding /
        // lm_head dequants are `[vocab][hidden]` = [248 320][5 120]).
        assert_eq!(g.embedding, (248_320, 5_120), "the real embedding table dims");
        assert_eq!(g.lm_head, (cfg.vocab, cfg.hidden), "the logits GEMM (m, k) = the artifact's text/output_head shape [vocab, hidden] (m = output dim, k = input dim)");
        assert_eq!(g.final_norm, cfg.hidden);
        assert_eq!(g.per_layer.len(), cfg.num_layers, "one geometry per layer");
        // The 27B layer-kind mix (16 GQA + 48 GDN, the `(i + 1) % 4` rule).
        let gqa = g.per_layer.iter().filter(|l| l.kind == LayerKind::Gqa).count();
        let gdn = g.per_layer.iter().filter(|l| l.kind == LayerKind::Gdn).count();
        assert_eq!((gqa, gdn), (16, 48), "the 27B layer-kind mix");
        // Every GDN layer's state readout (m = hidden, k = the per-token readout
        // width `state_rows` = 6144 — the artifact's `gdn/output` shape
        // `[hidden][state_rows]`, A3 / #30) + the input projection
        // (m = the GDN feature rows — the q / k / v / z parts,
        // `gdn_in_proj_m` = 16 384 — k = hidden — the artifact's
        // `gdn/query_key_value_z` shape, A3 / #30) — the forward pass's
        // (m, k) contract (m = output rows, k = input dim, the kernel
        // convention, GitHub #33).
        for lg in g.per_layer.iter().filter(|l| l.kind == LayerKind::Gdn) {
            assert_eq!(
                lg.gdn_output,
                (cfg.hidden, cfg.gdn_readout_k()),
                "the GDN state readout (m = hidden, k = the readout width state_rows)"
            );
            assert_eq!(
                lg.projection[0],
                (cfg.gdn_in_proj_m(), cfg.hidden),
                "the GDN input projection (m = the q/k/v/z feature rows, k = hidden)"
            );
            assert_eq!(lg.projection[1], (0, 0), "the unused GDN slots are (0, 0)");
            // The A3 / #30 GDN geometry: the causal-conv (4 taps × the
            // conv channels — the artifact's `gdn/convolution`), the a / b
            // (gate / beta) projection (the `gdn/a_b_projection`), the
            // conv channel count, the in-projection (m, k).
            assert_eq!(
                lg.gdn_conv,
                (4, cfg.gdn_conv_channels()),
                "the GDN causal-conv geometry (4 taps × the conv channels)"
            );
            assert_eq!(
                lg.gdn_ab,
                (cfg.gdn_ab_width, cfg.hidden),
                "the GDN a / b (gate / beta) projection (m = ab rows, k = hidden)"
            );
            assert_eq!(
                lg.gdn_conv_channels,
                cfg.gdn_conv_channels(),
                "the GDN conv channel count"
            );
            assert_eq!(
                lg.gdn_in_proj,
                (cfg.gdn_in_proj_m(), cfg.hidden),
                "the GDN in-projection (m, k)"
            );
        }
        // The GQA layers' projections (the `synthetic`'s (m, k) convention:
        // q / k / v map hidden -> the head widths (m = output, k = input);
        // the output projection maps gqa_width -> hidden (m = hidden,
        // k = gqa_width) — the kernel's (m, k) orientation, GitHub #33).
        // + the A3 / #30 GQA geometry (the q / k RMSNorm width =
        // `head_dim`, the artifact's `attention/query_norm` /
        // `attention/key_norm` shape).
        for lg in g.per_layer.iter().filter(|l| l.kind == LayerKind::Gqa) {
            assert_eq!(lg.projection[3], (cfg.hidden, cfg.gqa_width()));
            assert_eq!(lg.ffn_gate, (cfg.ffn_intermediate, cfg.hidden));
            assert_eq!(lg.ffn_down, (cfg.hidden, cfg.ffn_intermediate));
            assert_eq!(lg.norm_in, cfg.hidden);
            assert_eq!(lg.qk_norm, cfg.head_dim, "the GQA q / k RMSNorm width (head_dim)");
            assert_eq!(lg.gdn_conv, (0, 0), "the GQA layer has no GDN conv");
            assert_eq!(lg.gdn_in_proj, (0, 0), "the GQA layer has no GDN in-proj");
        }
        // The `from_artifact` path's `Weights` (via `from_geometry`) carry
        // this same geometry (the non-zero (m, k), the sized norms) — the
        // zero-geometry `placeholder` is no longer the artifact path's
        // construction.
        let w = Weights::from_geometry(&cfg);
        assert_eq!(w.lm_head.dims(), (cfg.vocab, cfg.hidden));
        assert!(
            w.lm_head.as_nvfp4().unwrap().codes.is_empty(),
            "the GEMM planes are device-resident (A3)"
        );
        for lw in &w.per_layer {
            assert_eq!(lw.norm_in.len(), cfg.hidden as usize);
            assert_eq!(lw.norm_post.len(), cfg.hidden as usize);
        }
        assert_eq!(w.final_norm.len(), cfg.hidden as usize);
        // The norms are identity (1.0, the `synthetic` convention).
        assert!(w.final_norm.iter().all(|&w| w == 0x3F80), "the identity norms");
    }

    /// `from_geometry` on a small topology: the `Weights` carry the real
    /// (m, k) geometry (not the `placeholder`'s zeros), the GEMM planes
    /// are empty (device-resident content, A3), and the embedding content
    /// is the W8 -> bf16 dequant buffer (A3's routing — not zero-filled
    /// here, the #26 lesson).
    #[test]
    fn from_geometry_weights_carry_real_geometry() {
        let cfg = ModelConfig::synthetic();
        let w = Weights::from_geometry(&cfg);
        assert_eq!(w.per_layer.len(), cfg.num_layers);
        for lw in &w.per_layer {
            match lw.kind {
                LayerKind::Gdn => {
                    // The state readout (m = hidden, k = the per-token
                    // readout width `state_rows` — A3 / #30) + the
                    // input projection (m = the GDN feature rows,
                    // `gdn_in_proj_m`, k = hidden) — the kernel's
                    // (m, k) orientation (GitHub #33).
                    assert_eq!(
                        (lw.gdn_output.m, lw.gdn_output.k),
                        (cfg.hidden, cfg.gdn_readout_k())
                    );
                    assert_eq!(
                        (lw.projection[0].m, lw.projection[0].k),
                        (cfg.gdn_in_proj_m(), cfg.hidden),
                        "the GDN input projection (m = output, k = input)"
                    );
                    assert!(
                        lw.gdn_output.codes.is_empty(),
                        "the GEMM planes are device-resident (A3)"
                    );
                }
                LayerKind::Gqa => {
                    assert_eq!(
                        lw.projection[0].m,
                        cfg.gqa_width(),
                        "the q projection's m (the output rows)"
                    );
                    assert!(lw.projection[3].m > 0, "the output projection's m");
                    assert!(lw.projection[3].codes.is_empty());
                }
            }
            assert_eq!(lw.ffn_gate.m, cfg.ffn_intermediate, "the ffn gate's m (output rows)");
            assert_eq!(lw.ffn_gate.k, cfg.hidden, "the ffn gate's k (input dim)");
            assert_eq!(lw.ffn_down.m, cfg.hidden, "the ffn down's m (output rows)");
            assert_eq!(
                lw.ffn_down.k,
                cfg.ffn_intermediate,
                "the ffn down's k (input dim)"
            );
            assert_eq!(lw.norm_in.len(), cfg.hidden as usize);
        }
        assert_eq!(w.lm_head.dims(), (cfg.vocab, cfg.hidden));
        // The embedding content is A3's dequant buffer (not zero-filled).
        assert!(w.embedding.is_empty());
    }

    /// The A1 / #27 normalization (spec 04 criterion 3): the `from_artifact`
    /// `Weights` are the real normalized buffers — the two W8 endpoints are
    /// host-dequantized to bf16 (the [`ignis_artifact::W8Endpoints`], ADR
    /// 0005) and routed into the `Weights` (the embedding table + the
    /// lm_head). The lm_head's (m, k) is the artifact's `text/output_head`
    /// shape (m = rows = vocab, k = cols = hidden, the kernel's
    /// (m = output, k = input) convention — not transposed). The NVFP4 GEMM
    /// planes stay device-resident (the `from_geometry`'s geometry-only
    /// content — not host-copied, the #26 lesson: no host weight explosion
    /// on the load path).
    #[test]
    fn weights_with_w8_endpoints_carry_the_dequant_content() {
        // A small `W8Endpoints` (the A1 host-side dequant of the two W8
        // endpoints, ADR 0005): the embedding + the lm_head content are
        // deterministic (the dequant of the W8 payloads).
        let endpoints = ignis_artifact::W8Endpoints {
            embedding: vec![0x3F80; 4 * 8], // bf16 1.0, the embedding table [4][8]
            embedding_shape: (4, 8),
            lm_head: vec![0x40C0; 8 * 16],  // bf16 6.0, the lm_head [8][16]
            lm_head_shape: (8, 16),
        };
        let cfg = ModelConfig::synthetic();
        let w = Weights::from_geometry(&cfg).with_w8_endpoints(endpoints);
        // The embedding carries the W8 dequant content (not empty —
        // criterion 3's "the real normalized buffers, not a placeholder").
        assert_eq!(w.embedding.len(), 4 * 8);
        assert!(
            w.embedding.iter().all(|&v| v == 0x3F80),
            "the W8 embedding content is carried (not a placeholder)"
        );
        // The lm_head is the dequantized bf16 (the W8 `text/output_head`),
        // with the artifact's `text/output_head` shape (m = rows, k = cols,
        // the kernel's (m = output, k = input) convention — not transposed).
        match &w.lm_head {
            HeadWeight::DequantBf16 { data, m, k } => {
                assert_eq!(
                    (*m, *k),
                    (8, 16),
                    "the lm_head (m, k) = the text/output_head shape [rows][cols]"
                );
                assert_eq!(data.len(), 8 * 16);
                assert!(
                    data.iter().all(|&v| v == 0x40C0),
                    "the W8 lm_head content is carried"
                );
            }
            other => panic!(
                "the lm_head must be the dequantized bf16 (the W8 endpoint), got {other:?}"
            ),
        }
    }
}
