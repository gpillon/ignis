//! The model topology config (ADR 0001).
//!
//! [`ModelConfig`] describes the shape the forward pass runs against (layer
//! count, per-layer kind (GQA / GDN), head geometry, GDN state dims, the GDN
//! feature layout (q / z / a-b widths), the rotary geometry, FFN width,
//! vocab, block geometry) — a real (artifact) model's config is derived from
//! the container's tensor directory; a synthetic (test) model uses
//! [`ModelConfig::synthetic`].
//!
//! The superseded compute adapter (the flat-C-ABI forward pass, its
//! `Weights`/`HeadWeight`/`Nvfp4Weight` host formats, the CUDA-graph
//! plumbing, and the `CudaCompute` production backend) was deleted by
//! GitHub #39 (ADR 0010): the vendored op-by-op replacement lands under the
//! Phase 1 decomposition (`.scratch/ROADMAP.md`), starting with the
//! [`crate::scheduler::Compute`] adapter at P1-24 (#60). Until then, the
//! engine drives [`crate::mock::MockCompute`] (ADR 0006).

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
    /// `state_cols + state_rows + 2` (k, v, gate, beta).
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
    /// The GDN recurrence's value-head count (`gdn_state_rows /
    /// gdn_head_dim`; the sequence-state pool's per-layer slot is this many
    /// `gdn_head_dim x gdn_head_dim` fp32 matrices, GitHub #55).
    pub gdn_value_heads: u64,
    /// The GDN recurrence's per-head state dimension — square (the
    /// reference's state matrix is `gdn_head_dim x gdn_head_dim` per value
    /// head: `value_head_dim == key_head_dim`, GitHub #55).
    pub gdn_head_dim: u64,
    /// The GQA RoPE rotary dim (of `head_dim` — the first `rotary_dim`
    /// dims of each q / k head are rotated; `rotary_dim / 2` pairs,
    /// GitHub #28).
    pub rotary_dim: u64,
    /// The RoPE base θ (the `inv_freq[pair] = θ^(-2·pair/rotary_dim)`
    /// table — the Qwen 3.8-27B GQA geometry θ = 1e7).
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
    /// z rows bypass the conv, A3 / #30).
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

    /// A small, fast synthetic topology for CPU tests (one GDN + one GQA
    /// layer, small dims, a small paged KV) — exercises every geometry
    /// derivation with a deterministic synthetic model.
    pub fn synthetic() -> Self {
        Self {
            num_layers: 2,
            layer_kinds: vec![LayerKind::Gdn, LayerKind::Gqa],
            hidden: 64,
            vocab: 256,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            gdn_state_rows: 16,
            gdn_state_cols: 8,
            gdn_num_layers: 1,
            // The synthetic model has no separate GDN q / z / a-b parts
            // (the input projection is the k / v / g / beta feature
            // directly — the `gdn_state_dim` layout, A3 / #30).
            gdn_q_width: 0,
            gdn_z_width: 0,
            gdn_ab_width: 0,
            // 2 value heads x 8-wide state == gdn_state_rows (16); square
            // state, so key_head_dim is the same 8.
            gdn_value_heads: 2,
            gdn_head_dim: 8,
            rotary_dim: 8,
            rope_theta: 1e7,
            ffn_intermediate: 32,
            block_size: 4,
            num_blocks: 8,
        }
    }

    /// The real Qwen 3.8-27B topology (the v1 specialization, CONTEXT.md: one
    /// model family, one GPU class). The layer pattern + head geometry are
    /// the model constants (ignis is specialized for Qwen 3.8-27B).
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
            // 48 value heads x 128-wide state == gdn_state_rows (6144);
            // square state (the reference's recurrence is 128x128 per
            // head, A3 / #30 / GitHub #55), so key_head_dim is the same
            // 128.
            gdn_value_heads: 48,
            gdn_head_dim: 128,
            // The GQA RoPE geometry (GitHub #28): the split-half NeoX
            // rotary of `rotary_dim` = 64 of `head_dim` = 256 (32 pairs),
            // base θ = 1e7 (the reference's `rope_linear_frequencies`
            // table).
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `gdn_value_heads * gdn_head_dim` must equal `gdn_state_rows` (the
    /// sequence-state pool's per-layer slot is `gdn_value_heads` square
    /// `gdn_head_dim x gdn_head_dim` matrices, GitHub #55) — pins the
    /// relationship so the two representations cannot silently drift.
    #[test]
    fn gdn_value_heads_and_head_dim_agree_with_state_rows() {
        for cfg in [ModelConfig::synthetic(), ModelConfig::qwen38_27b()] {
            assert_eq!(
                cfg.gdn_value_heads * cfg.gdn_head_dim,
                cfg.gdn_state_rows,
                "gdn_value_heads * gdn_head_dim must equal gdn_state_rows"
            );
        }
    }
}
