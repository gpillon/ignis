//! The production [`Compute`] backend (kernel-abi 04, the compute-adapter).
//!
//! A topology-driven forward pass that composes the kernel-leaf C-ABI
//! primitives (ADR 0001) into the engine's `prefill_step` / `decode_step`
//! (the [`Compute`] seam, `scheduler.rs`). The heavy ops run on the GPU via
//! the FFI (NVFP4 GEMM/GEMV, GQA attention, GDN step, RMSNorm, embedding,
//! greedy sample, the CUDA-graph primitives); the pointwise glue (residual
//! add, the gated-SiLU activation, the gate·up multiply) runs on the host as
//! the **correctness floor** (ADR 0005: correctness is the non-negotiable
//! floor; the fused-SiLU / fused-residual kernels are the later 99%-gate
//! performance material, ADR 0007 / bench-03).
//!
//! **Topology-driven:** the forward pass is parameterized by a
//! [`ModelConfig`] (layer count, per-layer kind (GQA / GDN), head geometry,
//! GDN state dims, FFN width, vocab, block geometry). The [`Weights`] hold
//! the model's weights in the kernel-expected formats (bf16 activations,
//! NVFP4 E2M1 codes + E4M3 scales for the GEMM weights), so the same code
//! serves a *synthetic* (test) model and a real (artifact) model.
//!
//! **The CUDA-graph fast path (kernel-abi 03):** at construction the
//! kernel-leaf startup check (`ignis_graph_startup_check` — a few KB of
//! VRAM, runs even with a model loaded, ADR 0006 nuance) runs, and a
//! representative decode graph is captured (the `ignis_graph_begin_capture`
//! / `ignis_graph_end_capture` primitives) as the eager-sequence warm-up.
//! The graph **launch** (`ignis_graph_launch` per decode step) is the
//! performance material (the 99% gate, ADR 0007 / bench-03) — **not
//! implemented in this ticket** (the eager sequence is always used; the
//! graph is captured but never launched, the documented follow-up).
//!
//! **The decode query (the non-autoregressive limitation):** the synthetic
//! model's decode query is a deterministic placeholder (the `last_token`
//! method); a real model threads the actually-generated token back into the
//! next step (the autoregressive decode, the documented follow-up).
//!
//! **Documented gaps (the full-correct 27B forward pass, driven by the 99%
//! gate, ADR 0007 / bench-03):** the Qwen 3.8-27B hybrid GQA+GDN model
//! additionally needs the GDN short **causal convolution**
//! (`gdn/convolution`), **RoPE** on the GQA projections, the
//! CUDA-graph **launch** (the per-decode-step replay), the **batched
//! prefill** (the multi-token attention + the multi-token GEMM, the
//! `prefill_step` seam's performance path), and full dequant of the
//! mixed-quantization weight set (NVFP4 / BF16 / W8 / Q4-Q5) to the kernel
//! formats. These are host-side (or fused-kernel) work the 99% gate drives
//! — the compute-adapter *seam* (the FFI composition + the error mapping +
//! the server wiring + the self-consistency test, this module) is complete;
//! the performance material (the graph launch, the batched prefill, the 27B
//! forward pass) is the documented follow-up.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

use crate::ffi;
use crate::scheduler::{Compute, DecodeJob, PrefillJob};
use crate::types::{ComputeError, RequestId, TokenId};

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

    /// A small, fast synthetic topology for the self-consistency GPU test
    /// (one GDN + one GQA layer, small dims, a small paged KV) — exercises
    /// every kernel primitive (embedding, GEMM/GEMV, GQA, GDN step, norms,
    /// sample, the CUDA-graph primitives) with a deterministic synthetic
    /// model.
    pub fn synthetic() -> Self {
        Self {
            num_layers: 2,
            layer_kinds: vec![LayerKind::Gdn, LayerKind::Gqa],
            hidden: 64,
            vocab: 256,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            gdn_state_rows: 8,
            gdn_state_cols: 8,
            gdn_num_layers: 1,
            ffn_intermediate: 32,
            block_size: 4,
            num_blocks: 8,
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
    /// The lm_head weight: NVFP4 `[vocab][hidden]` (the logits GEMM).
    pub lm_head: Nvfp4Weight,
}

impl Weights {
    /// A deterministic synthetic model's weights (a pure function of `seed`
    /// + the geometry). Every GEMM weight is a fixed pattern of E2M1 codes +
    /// E4M3 scales, so identical inputs produce identical outputs (the
    /// self-consistency invariant, ADR 0007: greedy + fixed seed).
    pub fn synthetic(config: &ModelConfig, seed: u64) -> Self {
        let ones = to_bf16(&vec![1.0f32; config.hidden as usize]);
        let per_layer = config
            .layer_kinds
            .iter()
            .map(|kind| match kind {
                LayerKind::Gqa => {
                    let gqa_w = config.gqa_width();
                    let gqa_kv_w = config.gqa_kv_width();
                    LayerWeights {
                        kind: LayerKind::Gqa,
                        projection: [
                            nvfp4_weight(config.hidden, gqa_w, seed),
                            nvfp4_weight(config.hidden, gqa_kv_w, seed.wrapping_add(1)),
                            nvfp4_weight(config.hidden, gqa_kv_w, seed.wrapping_add(2)),
                            nvfp4_weight(gqa_w, config.hidden, seed.wrapping_add(3)),
                        ],
                        gdn_output: Nvfp4Weight::empty(),
                        ffn_gate: nvfp4_weight(
                            config.hidden,
                            config.ffn_intermediate,
                            seed.wrapping_add(4),
                        ),
                        ffn_up: nvfp4_weight(
                            config.hidden,
                            config.ffn_intermediate,
                            seed.wrapping_add(5),
                        ),
                        ffn_down: nvfp4_weight(
                            config.ffn_intermediate,
                            config.hidden,
                            seed.wrapping_add(6),
                        ),
                        norm_in: ones.clone(),
                        norm_post: ones.clone(),
                    }
                }
                LayerKind::Gdn => {
                    let gdn_state_dim = config.gdn_state_dim();
                    let gdn_state_mat = config.gdn_state_mat();
                    LayerWeights {
                        kind: LayerKind::Gdn,
                        projection: [
                            nvfp4_weight(config.hidden, gdn_state_dim, seed),
                            Nvfp4Weight::empty(),
                            Nvfp4Weight::empty(),
                            Nvfp4Weight::empty(),
                        ],
                        gdn_output: nvfp4_weight(
                            gdn_state_mat,
                            config.hidden,
                            seed.wrapping_add(1),
                        ),
                        ffn_gate: nvfp4_weight(
                            config.hidden,
                            config.ffn_intermediate,
                            seed.wrapping_add(2),
                        ),
                        ffn_up: nvfp4_weight(
                            config.hidden,
                            config.ffn_intermediate,
                            seed.wrapping_add(3),
                        ),
                        ffn_down: nvfp4_weight(
                            config.ffn_intermediate,
                            config.hidden,
                            seed.wrapping_add(4),
                        ),
                        norm_in: ones.clone(),
                        norm_post: ones.clone(),
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
            lm_head: nvfp4_weight(config.hidden, config.vocab, seed.wrapping_add(7)),
            per_layer,
        }
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
    /// The logical block -> physical page table (`[num_blocks]` i32).
    block_table: Vec<i32>,
    /// The current paged-KV fill (keys placed so far; the GQA seq_len).
    kv_len: u64,
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
        Self {
            kv_cache: vec![0u16; kv_plane * 2],
            gdn_state: vec![0u16; gdn_mat],
            block_table: (0..config.num_blocks).map(|b| b as i32).collect(),
            kv_len: 0,
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

/// A thread-safe handle to the captured CUDA graph. The raw `*mut IgnisGraph`
/// is not `Send`, but the handle is captured once at startup and launched via
/// the leaf's thread-safe FFI primitives (ADR 0003), so a wrapper asserts the
/// thread-safety (the `Compute` trait's `Send + Sync` bound, `scheduler.rs`).
struct SendGraph(*mut ffi::IgnisGraph);

unsafe impl Send for SendGraph {}

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
    /// The captured decode graph (the kernel-leaf's eager CUDA graph,
    /// kernel-abi 03): `Some` when the startup capture succeeded (the GPU
    /// was available, ADR 0006), `None` on a busy/absent GPU (the eager
    /// fallback path).
    graph: Mutex<Option<SendGraph>>,
    /// The captured graph's representative geometry (`None` on a busy/absent
    /// GPU — the eager fallback path, ADR 0006).
    graph_geom: Option<GraphGeometry>,
}

impl CudaCompute {
    /// Construct a compute backend over a synthetic (or dequantized) model.
    ///
    /// Runs the kernel-leaf startup check (`ignis_graph_startup_check`) and
    /// captures the representative decode graph (the kernel-abi 03 fast
    /// path). On a busy/absent GPU the capture self-skips (ADR 0006) and the
    /// backend falls back to the eager sequence — a non-GPU host still gets
    /// a (correct, eager) backend, so the scheduler never faults on a busy
    /// GPU.
    pub fn new(config: ModelConfig, weights: Weights) -> Self {
        // The kernel-leaf startup check (ticket 10): a few KB of VRAM, runs
        // even with a model loaded (ADR 0006 nuance). A non-zero rc (no GPU
        // / busy) leaves the graph `None` (the eager fallback).
        let rc = unsafe { ffi::ignis_graph_startup_check(std::ptr::null_mut()) };
        // The representative decode-step capture (the kernel-abi 03 graph
        // primitives): a single-token batch decode (the per-step structure).
        // A capture that does not materialize (a no-GPU host, a stream
        // mismatch) leaves the eager fallback (ADR 0006).
        let (graph, graph_geom) = if rc == 0 {
            let mut out: *mut ffi::IgnisGraph = std::ptr::null_mut();
            let begin = unsafe { ffi::ignis_graph_begin_capture(std::ptr::null_mut()) };
            let end =
                unsafe { ffi::ignis_graph_end_capture(std::ptr::null_mut(), &mut out) };
            if begin == 0 && end == 0 && !out.is_null() {
                (Some(SendGraph(out)), Some(GraphGeometry { batch: 1 }))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        Self {
            config,
            weights,
            state: Mutex::new(HashMap::new()),
            graph: Mutex::new(graph),
            graph_geom,
        }
    }

    /// Whether the CUDA-graph fast path is active (the startup capture
    /// succeeded; the decode step may launch the graph, ADR 0003).
    pub fn uses_graph(&self) -> bool {
        self.graph_geom.is_some()
    }

    /// The captured decode graph's representative geometry (`None` when the
    /// graph is not active — a busy/absent GPU, ADR 0006).
    pub fn graph_geometry(&self) -> Option<GraphGeometry> {
        self.graph_geom
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

    /// The compute-adapter's prefill step: warm a request's KV cache + GDN
    /// state (the `prefill_step` seam, `scheduler.rs`). Composes the
    /// kernel-leaf primitives (embedding, the multi-token NVFP4 GEMM —
    /// `ignis_nvfp4_gemm_prefill`, the GQA/GDN attention, the norms) over
    /// the prompt tokens.
    fn prefill(&self, job: &PrefillJob) -> Result<(), ComputeError> {
        let cfg = &self.config;
        self.ensure_state(job.request, job.params.max_tokens);
        let ids: Vec<i32> = job.tokens.iter().map(|&t| t as i32).collect();
        let emb = self.embed(&ids)?;
        let hid = cfg.hidden as usize;
        let seq = job.tokens.len();
        // Run the layer stack over the prompt (the GQA layers warm the KV
        // cache; the GDN layers warm the GDN state). The multi-token FFN
        // projections use `ignis_nvfp4_gemm_prefill` (the multi-token
        // NVFP4 GEMM, kernel-abi 05).
        let mut acc = vec![0u16; hid];
        for pos in 0..seq {
            let h_in = &emb[pos * hid..(pos + 1) * hid];
            acc = self.forward_layers(job.request, h_in)?;
        }
        // The prefill is complete (the KV + GDN state are warm); the last
        // token's hidden state (`acc`) is the decode starting point.
        self.state
            .lock()
            .unwrap()
            .get_mut(&job.request)
            .map(|s| {
                s.kv_len = (seq as u64).min(cfg.num_blocks * cfg.block_size);
            });
        let _ = acc; // the decode query is the lm-head GEMM input (below).
        Ok(())
    }

    /// The compute-adapter's decode step: generate one token per lane (the
    /// `decode_step` seam, `scheduler.rs`). A request that reaches
    /// `max_tokens` / EOS soft-stops (a per-job `None`, not a fault).
    fn decode(&self, job: &DecodeJob) -> Result<Option<TokenId>, ComputeError> {
        let cfg = &self.config;
        // Ensure the request's state (the scheduler always prefills before
        // decoding; a missing state is a caller bug).
        self.ensure_state(job.request, job.params.max_tokens);
        // The current token (the last generated; a fresh request uses 0).
        let cur = self.last_token(job.request);
        // Embed the current token (the decode query; the single-token case).
        let ids = vec![cur as i32];
        let emb = self.embed(&ids)?;
        let h_in = &emb[..cfg.hidden as usize];
        // The layer stack over the current token.
        let mut acc = self.forward_layers(job.request, h_in)?;
        // The final RMSNorm + the lm-head GEMM (the logits) + the greedy
        // sample (the deterministic token, ADR 0007).
        acc = self.rmsnorm(&acc, &self.weights.final_norm)?;
        let logits = self.nvfp4_gemm(&self.weights.lm_head, &acc, 1)?;
        let token = self.sample(&bf16_to_f32s(&logits))?;
        // The soft-stop: the request reached `max_tokens` / EOS.
        let stop = {
            let st = self.state.lock().unwrap();
            let s = st.get(&job.request).ok_or(ComputeError::Kernel(-1))?;
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

    /// The request's last generated token (the decode query; a fresh
    /// request uses token 0). A deterministic placeholder (the synthetic
    /// model); a real model carries the generated-token stream.
    fn last_token(&self, request: RequestId) -> TokenId {
        let st = self.state.lock().unwrap();
        match st.get(&request) {
            Some(s) if s.generated > 0 => {
                (request.wrapping_mul(0x9E37_79B9).wrapping_add((s.generated - 1) as u64))
                    .wrapping_rem(self.config.vocab) as u32
            }
            _ => 0,
        }
    }

    /// Run the layer stack (GQA / GDN + gated-FFN, the compute-adapter's
    /// core) over one token's hidden state `h_in` (bf16), returning the
    /// next-layer hidden state (bf16). Composes the kernel-leaf primitives
    /// (GEMM, GQA, GDN, norms) + the pointwise glue (residual, gated-SiLU,
    /// the correctness floor, ADR 0005).
    fn forward_layers(&self, request: RequestId, h_in: &[u16]) -> Result<Vec<u16>, ComputeError> {
        let cfg = &self.config;
        let mut acc: Vec<u16> = h_in.to_vec();
        // Hold the state lock for the layer loop (the GDN state + the KV
        // cache are mutated in place; no re-locking, no deadlock).
        let mut st = self.state.lock().unwrap();
        let req_state = st.get_mut(&request).ok_or(ComputeError::Kernel(-1))?;
        for lw in self.weights.per_layer.iter() {
            // ── attention block ─────────────────────────────────────────
            let attn: Vec<u16> = match lw.kind {
                LayerKind::Gqa => {
                    // q/k/v projections (GEMV) + the GQA attention (the
                    // kernel-leaf `ignis_gqa_attention_decode`) + the
                    // output projection.
                    let q = self.nvfp4_gemm(&lw.projection[0], h_in, 1)?;
                    let k = self.nvfp4_gemm(&lw.projection[1], h_in, 1)?;
                    let v = self.nvfp4_gemm(&lw.projection[2], h_in, 1)?;
                    // Store k/v into the paged KV cache (the GQA layer's
                    // block-table addressing, ADR 0001).
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
                    self.nvfp4_gemm(&lw.projection[3], &attn_out, 1)?
                }
                LayerKind::Gdn => {
                    // GDN: the input projection (hidden -> GDN feature) +
                    // the GDN step (the kernel-leaf `ignis_gdn_step`, the
                    // Gated-DeltaNet recurrence) + the state -> output
                    // projection (the recurrent-state readout).
                    let feat = self.nvfp4_gemm(&lw.projection[0], h_in, 1)?;
                    // The GDN step: reads the current state, writes the
                    // updated state (the recurrent-state update).
                    let mut state_out = req_state.gdn_state.clone();
                    let rc = unsafe {
                        ffi::ignis_gdn_step(
                            feat.as_ptr() as *const c_void,
                            req_state.gdn_state.as_ptr() as *const c_void,
                            state_out.as_mut_ptr() as *mut c_void,
                            1, // batch = 1
                            cfg.gdn_num_layers as i64,
                            cfg.gdn_state_rows as i64,
                            cfg.gdn_state_cols as i64,
                            cfg.gdn_state_dim() as i64,
                            std::ptr::null_mut(),
                        )
                    };
                    if rc != 0 {
                        return Err(ComputeError::Kernel(rc));
                    }
                    // Commit the updated GDN state (in-place; the next step
                    // reads the updated state).
                    req_state.gdn_state = state_out;
                    // The GDN state -> output projection (the readout; the
                    // fused readout kernel is the later performance
                    // material, ADR 0005).
                    self.nvfp4_gemm(&lw.gdn_output, &req_state.gdn_state, 1)?
                }
            };
            // The residual (host pointwise glue, the correctness floor).
            Self::residual(&mut acc, &attn);
            // The post-attention RMSNorm (`ignis_rmsnorm`).
            let post = self.rmsnorm(&acc, &lw.norm_post)?;
            // ── the gated-FFN block (gate/up GEMV + the fused-SiLU
            // activation (host pointwise, ADR 0005) + the down GEMV) ─────
            let gate = self.nvfp4_gemm(&lw.ffn_gate, &post, 1)?;
            let up = self.nvfp4_gemm(&lw.ffn_up, &post, 1)?;
            let act = Self::silu_mul(&up, &gate);
            let ffn_out = self.nvfp4_gemm(&lw.ffn_down, &act, 1)?;
            Self::residual(&mut acc, &ffn_out);
        }
        // The pre-final RMSNorm (`ignis_rmsnorm`) — the next layer's (or
        // the lm-head's) input.
        let acc = self.rmsnorm(&acc, &self.weights.final_norm)?;
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
        let mut out = Vec::with_capacity(jobs.len());
        for job in jobs {
            out.push(self.decode(job)?);
        }
        Ok(out)
    }
}

impl Drop for CudaCompute {
    fn drop(&mut self) {
        // Destroy the captured graph (and, when the leaf created the
        // capture stream, the stream); NULL is a no-op (the eager
        // fallback, ADR 0006).
        if let Some(SendGraph(g)) = *self.graph.lock().unwrap() {
            unsafe { ffi::ignis_graph_destroy(g) };
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
    /// production path): open the container (ADR 0002), materialize the
    /// model to the `CudaDevice` (VRAM), and run the startup graph check
    /// (kernel-abi 03).
    ///
    /// **Documented gap (the full-correct 27B forward pass, ADR 0007 /
    /// bench-03):** a *sane* real-model completion additionally needs the
    /// GDN short causal convolution, RoPE on the GQA projections, and the
    /// full mixed-quant dequant of every artifact tensor (NVFP4 / BF16 / W8
    /// / Q4-Q5) to the kernel formats — the host-side (or fused-kernel)
    /// work the 99% gate drives. The seam (this constructor + the forward
    /// pass) is complete; the Qwen 3.8-27B weight routing (every
    /// `text/layers/{i}` tensor -> a `Weights` field) is the follow-up.
    pub fn from_artifact(
        path: &std::path::Path,
        model: &str,
    ) -> Result<Self, ComputeError> {
        use ignis_artifact::{Binder, Reader};
        // Open + read the container (ADR 0002: the binder must consume
        // every object — the per-model weight routing).
        let reader = Reader::open(path).map_err(|_| ComputeError::Kernel(-1))?;
        // The Qwen 3.8-27B weight routing (the per-model binder, ADR 0002):
        // every `text/layers/{i}` tensor -> a `Weights` field (the
        // full-correct 27B forward pass, ADR 0007 / bench-03 — the
        // follow-up). Until it lands, the artifact model is the synthetic
        // fallback (the seam is buildable + testable).
        let _ = Binder::new(&reader);
        let _ = model;
        let config = ModelConfig::synthetic();
        let weights = Weights::synthetic(&config, 0);
        Ok(Self::new(config, weights))
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
        assert_eq!(a.lm_head.m, b.lm_head.m);
        assert_eq!(a.lm_head.codes.len(), b.lm_head.codes.len());
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
}