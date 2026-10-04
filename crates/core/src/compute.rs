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
//! Phase 1 decomposition (`docs/ROADMAP.md`), starting with the
//! [`crate::scheduler::Compute`] adapter at P1-24 (#60). Until then, the
//! engine drives [`crate::mock::MockCompute`] (ADR 0006).

use crate::kv_format::KvGeometry;

// ---------------------------------------------------------------------------
// Topology (ADR 0001: the model config the forward pass is parameterized by)
// ---------------------------------------------------------------------------

/// The kind of attention a decoder layer uses (the Qwen 3.8-27B hybrid is a
/// GQA + GDN (Gated DeltaNet linear-attention) mix).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// A standard GQA (grouped-query attention) layer. On Flash-Next it is a
    /// QSA layer: the same GQA with the model's [`IndexerGeometry`] in front.
    Gqa,
    /// A GDN (Gated DeltaNet linear-attention, recurrent-state) layer.
    Gdn,
}

/// Which model a topology describes (ADR 0043: two models, one loaded at a
/// time). It names the program the forward runs; every count and shape is
/// still read off the topology's own fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Qwen 3.8-27B: one residual stream and a dense MLP in every layer. The
    /// synthetic test topology is this program at small dims.
    Qwen38_27b,
    /// Qwen3.8-Flash-Next: four hyper-connection streams, an MoE block in
    /// every layer, QSA layers with an indexer, and the n-gram embedding.
    FlashNext,
}

/// Flash-Next's mixture-of-experts block, one per layer: the router picks
/// `experts_per_token` of `num_experts` routed SwiGLU experts, and the shared
/// expert runs for every token behind a sigmoid gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeGeometry {
    pub num_experts: u64,
    pub experts_per_token: u64,
    /// A routed expert's intermediate width (`moe_intermediate_size`).
    pub expert_intermediate: u64,
    /// The shared expert's intermediate width
    /// (`shared_expert_intermediate_size`).
    pub shared_expert_intermediate: u64,
}

impl MoeGeometry {
    /// The rows of an expert's fused gate/up projection (gate, then up).
    pub fn expert_gate_up_rows(&self) -> u64 {
        2 * self.expert_intermediate
    }
}

/// Flash-Next's hyper-connections: `streams` residual streams of `hidden`
/// per token. Each sublayer reads a learned mix of them through a rank-`rank`
/// sigmoid gate and writes back through a gated residual (`hc_count`,
/// `hc_lowrank`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperConnections {
    pub streams: u64,
    pub rank: u64,
}

/// The QSA indexer in front of every Flash-Next attention layer: `heads`
/// query heads of `head_dim` score blocks of `compress_ratio` keys of one
/// compressed key head, and the top `budget / compress_ratio` blocks plus the
/// incomplete tail are what the attention reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexerGeometry {
    pub heads: u64,
    pub head_dim: u64,
    pub kv_heads: u64,
    pub compress_ratio: u64,
    /// The most tokens selected from complete blocks per query.
    pub budget: u64,
}

impl IndexerGeometry {
    /// Blocks selected per query (`block_topk` in the modeling code).
    pub fn block_top_k(&self) -> u64 {
        self.budget / self.compress_ratio
    }

    /// The most visible tokens for which attention is still dense: every
    /// complete block is selected while `floor(V / compress_ratio) <=
    /// block_top_k`, and the incomplete tail always is.
    pub fn dense_threshold(&self) -> u64 {
        self.block_top_k() * self.compress_ratio + self.compress_ratio - 1
    }

    /// The rows of the indexer's fused query/key projection
    /// (`index_qk_proj`): the query heads, then the key head.
    pub fn qk_projection_rows(&self) -> u64 {
        (self.heads + self.kv_heads) * self.head_dim
    }
}

/// Flash-Next's hashed n-gram embedding (the checkpoint's PLE layer): each
/// token's 2-grams and 3-grams hash to `heads()` rows of `head_dim()` in the
/// n-gram table, concatenated to `embed_dim`, gated per stream and passed
/// through a dilated causal conv before `layer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramGeometry {
    /// The longest n-gram hashed (`ngram_size`).
    pub ngram_size: u64,
    pub heads_per_ngram: u64,
    /// The concatenated embedding width (`ple_embed_dim`).
    pub embed_dim: u64,
    /// The dilated conv's kernel width (`ple_conv_kernel_size`).
    pub conv_kernel: u64,
    /// The decoder layer (0-based) whose input the embedding is added to.
    pub layer: usize,
    /// Each head's table size is a prime above this (`ngram_vocab_size_base`).
    pub vocab_size_base: u64,
    /// The table's row count is padded to a multiple of this
    /// (`make_ngram_vocab_size_divisible_by`).
    pub vocab_divisor: u64,
    /// The checkpoint stores the table in this many parts
    /// (`split_ngram_parts`).
    pub split_parts: u64,
    /// The seed the hash multipliers are drawn from (`seed`).
    pub seed: u64,
    /// The token that ends a segment: no n-gram reaches back across it.
    pub eos_token_id: u32,
}

impl NgramGeometry {
    /// Hashed heads per token: `heads_per_ngram` for each n-gram size from 2
    /// to `ngram_size`.
    pub fn heads(&self) -> u64 {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// One head's row width: the embedding split evenly over the heads.
    pub fn head_dim(&self) -> u64 {
        self.embed_dim / self.heads()
    }

    /// The previous tokens a token's n-grams read.
    pub fn context_tokens(&self) -> u64 {
        self.ngram_size - 1
    }

    /// The conv's dilation, which the checkpoint ties to the n-gram size.
    pub fn conv_dilation(&self) -> u64 {
        self.ngram_size
    }

    /// The past positions the dilated conv reads: `(kernel - 1) * dilation`.
    pub fn conv_state_tokens(&self) -> u64 {
        (self.conv_kernel - 1) * self.conv_dilation()
    }
}

/// The model topology the forward pass is parameterized by (ADR 0001).
///
/// A real (artifact) model's config is derived from the container's tensor
/// directory (the per-layer shapes); a synthetic (test) model uses
/// [`ModelConfig::synthetic`].
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// The model this topology describes (ADR 0043).
    pub family: ModelFamily,
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
    /// The GDN layer count: one conv + recurrent state per GDN layer. Not
    /// the GDN value-head count ([`ModelConfig::gdn_value_heads`]); the two
    /// are equal on the 27B (48) and differ on Flash-Next (36 and 48).
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
    /// The FFN (gated-SiLU) intermediate width. 0 for a model with no dense
    /// MLP (Flash-Next: its MLP is [`ModelConfig::moe`]).
    pub ffn_intermediate: u64,
    /// The paged KV block size (keys per block).
    pub block_size: u64,
    /// The paged KV block count per request (capacity = block_size *
    /// num_blocks keys).
    pub num_blocks: u64,
    /// The RMSNorm epsilon of every norm in the text model.
    pub rms_norm_eps: f32,
    /// The MoE block of every layer; `None` for a dense MLP.
    pub moe: Option<MoeGeometry>,
    /// The hyper-connection streams; `None` for a single residual stream.
    pub hyper_connections: Option<HyperConnections>,
    /// The QSA indexer of every attention layer; `None` for plain GQA.
    pub indexer: Option<IndexerGeometry>,
    /// The hashed n-gram embedding; `None` for a model without one.
    pub ngram: Option<NgramGeometry>,
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

    /// The attention layers (GQA, or QSA on Flash-Next): each keeps its own
    /// paged K/V.
    pub fn attention_layer_count(&self) -> usize {
        self.layer_kinds.iter().filter(|&&kind| kind == LayerKind::Gqa).count()
    }

    /// The GDN layers: each keeps its own conv taps and recurrent state.
    pub fn gdn_layer_count(&self) -> usize {
        self.layer_kinds.iter().filter(|&&kind| kind == LayerKind::Gdn).count()
    }

    /// The paged-KV geometry this topology's attention layers store.
    pub fn kv_geometry(&self) -> KvGeometry {
        KvGeometry {
            gqa_layers: self.attention_layer_count() as u32,
            num_kv_heads: self.num_kv_heads as u32,
            head_dim: self.head_dim as u32,
        }
    }

    /// Residual streams per token: the hyper-connection count, or 1.
    pub fn residual_streams(&self) -> u64 {
        self.hyper_connections.map_or(1, |hyper| hyper.streams)
    }

    /// The residual's features per token across every stream.
    pub fn residual_width(&self) -> u64 {
        self.residual_streams() * self.hidden
    }

    /// The BF16 residual's bytes per token.
    pub fn residual_bytes_per_token(&self) -> u64 {
        self.residual_width() * 2
    }

    /// The indexer's compressed BF16 keys per sequence-token across every
    /// attention layer: one key of `kv_heads * head_dim` per
    /// `compress_ratio` tokens. 0 without an indexer.
    pub fn indexer_bytes_per_token(&self) -> u64 {
        self.indexer.map_or(0, |indexer| {
            self.attention_layer_count() as u64 * indexer.kv_heads * indexer.head_dim * 2
                / indexer.compress_ratio
        })
    }

    /// The n-gram embedding's dilated-conv state per lane: its past
    /// positions of the BF16 PLE output, one channel per residual feature.
    /// 0 without an n-gram embedding.
    pub fn ngram_conv_state_bytes(&self) -> u64 {
        self.ngram
            .map_or(0, |ngram| ngram.conv_state_tokens() * self.residual_width() * 2)
    }

    /// A small, fast synthetic topology for CPU tests (one GDN + one GQA
    /// layer, small dims, a small paged KV) — exercises every geometry
    /// derivation with a deterministic synthetic model.
    pub fn synthetic() -> Self {
        Self {
            family: ModelFamily::Qwen38_27b,
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
            rms_norm_eps: 1e-6,
            moe: None,
            hyper_connections: None,
            indexer: None,
            ngram: None,
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
            family: ModelFamily::Qwen38_27b,
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
            // The Qwen 3.8-27B text config's RMSNorm epsilon (the reference's
            // `TextConfig::rms_epsilon`, `qwen3_6_27b/impl/config.h`).
            rms_norm_eps: 1e-6,
            moe: None,
            hyper_connections: None,
            indexer: None,
            ngram: None,
        }
    }

    /// The Qwen3.8-Flash-Next topology (ADR 0043), every number from the
    /// checkpoint's text config (`Qwen/Qwen3.8-Flash-Next` revision
    /// `de4b8e4d43b917e7706784d8bb445c9af86a3540`, `config.json`
    /// `text_config`).
    pub fn qwen38_flash_next() -> Self {
        let num_layers: usize = 48;
        // `layer_types`: `full_attention` exactly when `(i + 1) %
        // full_attention_interval == 0` (interval 4): 12 QSA + 36 GDN layers.
        let layer_kinds: Vec<LayerKind> = (0..num_layers)
            .map(|i| if (i + 1) % 4 == 0 { LayerKind::Gqa } else { LayerKind::Gdn })
            .collect();
        Self {
            family: ModelFamily::FlashNext,
            num_layers,
            layer_kinds,
            hidden: 2560,
            vocab: 248_320,
            num_q_heads: 24,
            num_kv_heads: 2,
            head_dim: 256,
            // GDN: 48 value heads x 128 (rows) by 16 key heads x 128 (cols),
            // the 27B's head geometry at hidden 2560 (`linear_num_value_heads`,
            // `linear_num_key_heads`, `linear_*_head_dim`).
            gdn_state_rows: 6144,
            gdn_state_cols: 2048,
            gdn_num_layers: 36,
            // `in_proj_qkv` is q 2048 + k 2048 + v 6144 (the conv'd part),
            // `in_proj_z` 6144, `in_proj_a` and `in_proj_b` 48 each.
            gdn_q_width: 2048,
            gdn_z_width: 6144,
            gdn_ab_width: 96,
            gdn_value_heads: 48,
            gdn_head_dim: 128,
            // `partial_rotary_factor` 0.25 of `head_dim` 256. The config's
            // interleaved M-RoPE sections (11, 11, 10 pairs) rotate one text
            // position on all three axes, which is the 1-D table.
            rotary_dim: 64,
            rope_theta: 1e7,
            ffn_intermediate: 0,
            // Paged KV: 64-token pages, 4096 per request for the checkpoint's
            // 262,144 `max_position_embeddings`.
            block_size: 64,
            num_blocks: 4096,
            rms_norm_eps: 1e-6,
            moe: Some(MoeGeometry {
                num_experts: 512,
                experts_per_token: 10,
                expert_intermediate: 640,
                shared_expert_intermediate: 640,
            }),
            hyper_connections: Some(HyperConnections { streams: 4, rank: 320 }),
            indexer: Some(IndexerGeometry {
                heads: 4,
                head_dim: 128,
                kv_heads: 1,
                compress_ratio: 4,
                budget: 2048,
            }),
            ngram: Some(NgramGeometry {
                ngram_size: 3,
                heads_per_ngram: 8,
                embed_dim: 2560,
                conv_kernel: 4,
                // `ple_layer_ids` [2] is one-indexed: the modeling code puts
                // the PLE layer in decoder layer `i` when `i + 1` is listed.
                layer: 2 - 1,
                vocab_size_base: 20_000_000,
                vocab_divisor: 128,
                split_parts: 128,
                // Not in `text_config`: the config class's default.
                seed: 1234,
                eos_token_id: 248_044,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// The topology at the step ABI (`struct ignis_topology`, ADR 0009)
// ---------------------------------------------------------------------------

impl LayerKind {
    /// `enum ignis_layer_kind`.
    pub fn abi_code(self) -> i32 {
        match self {
            LayerKind::Gdn => 0,
            LayerKind::Gqa => 1,
        }
    }
}

impl ModelFamily {
    /// The model's name, as a refusal names it to a client.
    pub fn name(self) -> &'static str {
        match self {
            ModelFamily::Qwen38_27b => "Qwen3.8-27B",
            ModelFamily::FlashNext => "Qwen3.8-Flash-Next",
        }
    }

    /// `enum ignis_model_family`.
    pub fn abi_code(self) -> i32 {
        match self {
            ModelFamily::Qwen38_27b => 0,
            ModelFamily::FlashNext => 1,
        }
    }
}

/// 1:1 with `struct ignis_moe_topology`; all zero for a dense MLP.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IgnisMoeTopology {
    pub num_experts: u64,
    pub experts_per_token: u64,
    pub expert_intermediate: u64,
    pub shared_expert_intermediate: u64,
}

/// 1:1 with `struct ignis_hyper_topology`; all zero for one residual stream.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IgnisHyperTopology {
    pub streams: u64,
    pub rank: u64,
}

/// 1:1 with `struct ignis_indexer_topology`; all zero for plain GQA.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IgnisIndexerTopology {
    pub heads: u64,
    pub head_dim: u64,
    pub kv_heads: u64,
    pub compress_ratio: u64,
    pub budget: u64,
}

/// 1:1 with `struct ignis_ngram_topology`: what the device side of the
/// n-gram embedding is shaped by (the hashing is the host's, `crate::ngram`).
/// All zero without an n-gram embedding.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IgnisNgramTopology {
    pub ngram_size: u64,
    pub heads_per_ngram: u64,
    pub embed_dim: u64,
    pub conv_kernel: u64,
    pub layer: u64,
}

/// 1:1 with `struct ignis_topology` (`kernel/include/ignis_model.h`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IgnisTopology {
    pub num_layers: u32,
    pub layer_kinds: *const i32,
    pub hidden: u64,
    pub vocab: u64,
    pub num_q_heads: u64,
    pub num_kv_heads: u64,
    pub head_dim: u64,
    pub rotary_dim: u64,
    pub rope_theta: f64,
    pub gdn_state_rows: u64,
    pub gdn_state_cols: u64,
    pub gdn_num_layers: u64,
    pub gdn_q_width: u64,
    pub gdn_z_width: u64,
    pub gdn_ab_width: u64,
    pub ffn_intermediate: u64,
    pub rms_norm_eps: f32,
    pub family: i32,
    pub gdn_value_heads: u64,
    pub moe: IgnisMoeTopology,
    pub hyper: IgnisHyperTopology,
    pub indexer: IgnisIndexerTopology,
    pub ngram: IgnisNgramTopology,
}

/// A topology descriptor and the layer-kind array its pointer reads, owned
/// together so the pointer cannot outlive the array.
#[derive(Debug)]
pub struct TopologyAbi {
    /// Read only through `raw.layer_kinds`: held so that pointer stays valid.
    #[allow(dead_code)]
    layer_kinds: Vec<i32>,
    raw: IgnisTopology,
}

impl TopologyAbi {
    /// The descriptor `ignis_model_load` reads, valid while `self` lives.
    pub fn raw(&self) -> &IgnisTopology {
        &self.raw
    }
}

impl ModelConfig {
    /// This topology as it crosses the step ABI.
    pub fn topology_abi(&self) -> TopologyAbi {
        let layer_kinds: Vec<i32> = self.layer_kinds.iter().map(|kind| kind.abi_code()).collect();
        let moe = self.moe.map_or_else(IgnisMoeTopology::default, |moe| IgnisMoeTopology {
            num_experts: moe.num_experts,
            experts_per_token: moe.experts_per_token,
            expert_intermediate: moe.expert_intermediate,
            shared_expert_intermediate: moe.shared_expert_intermediate,
        });
        let hyper = self
            .hyper_connections
            .map_or_else(IgnisHyperTopology::default, |hyper| IgnisHyperTopology { streams: hyper.streams, rank: hyper.rank });
        let indexer = self.indexer.map_or_else(IgnisIndexerTopology::default, |indexer| IgnisIndexerTopology {
            heads: indexer.heads,
            head_dim: indexer.head_dim,
            kv_heads: indexer.kv_heads,
            compress_ratio: indexer.compress_ratio,
            budget: indexer.budget,
        });
        let ngram = self.ngram.map_or_else(IgnisNgramTopology::default, |ngram| IgnisNgramTopology {
            ngram_size: ngram.ngram_size,
            heads_per_ngram: ngram.heads_per_ngram,
            embed_dim: ngram.embed_dim,
            conv_kernel: ngram.conv_kernel,
            layer: ngram.layer as u64,
        });
        let raw = IgnisTopology {
            num_layers: self.num_layers as u32,
            // The Vec's heap buffer does not move when the Vec does.
            layer_kinds: layer_kinds.as_ptr(),
            hidden: self.hidden,
            vocab: self.vocab,
            num_q_heads: self.num_q_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            rotary_dim: self.rotary_dim,
            rope_theta: self.rope_theta,
            gdn_state_rows: self.gdn_state_rows,
            gdn_state_cols: self.gdn_state_cols,
            gdn_num_layers: self.gdn_num_layers,
            gdn_q_width: self.gdn_q_width,
            gdn_z_width: self.gdn_z_width,
            gdn_ab_width: self.gdn_ab_width,
            ffn_intermediate: self.ffn_intermediate,
            rms_norm_eps: self.rms_norm_eps,
            family: self.family.abi_code(),
            gdn_value_heads: self.gdn_value_heads,
            moe,
            hyper,
            indexer,
            ngram,
        };
        TopologyAbi { layer_kinds, raw }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_format::{KvFormat, KvGeometry};

    /// The layer pattern of `fn_config.json` (the checkpoint's text config):
    /// `full_attention` at every fourth layer, 3, 7, ..., 47, and
    /// `linear_attention` (GDN) everywhere else.
    #[test]
    fn flash_next_is_twelve_blocks_of_three_gdn_and_one_qsa() {
        let cfg = ModelConfig::qwen38_flash_next();
        assert_eq!(cfg.family, ModelFamily::FlashNext);
        assert_eq!(cfg.num_layers, 48);
        assert_eq!(cfg.layer_kinds.len(), 48);
        let attention: Vec<usize> = (0..cfg.num_layers)
            .filter(|&i| cfg.layer_kinds[i] == LayerKind::Gqa)
            .collect();
        assert_eq!(attention, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]);
        assert_eq!(cfg.attention_layer_count(), 12);
        assert_eq!(cfg.gdn_layer_count(), 36);
    }

    /// The GDN layer count and the GDN value-head count are different
    /// numbers on Flash-Next (36 and 48); on the 27B both are 48, which is
    /// what let the two be conflated.
    #[test]
    fn the_gdn_layer_count_and_value_head_count_are_separate_fields() {
        let flash = ModelConfig::qwen38_flash_next();
        assert_eq!(flash.gdn_num_layers, 36);
        assert_eq!(flash.gdn_value_heads, 48);
        let dense = ModelConfig::qwen38_27b();
        assert_eq!((dense.gdn_num_layers, dense.gdn_value_heads), (48, 48));
        for cfg in [ModelConfig::synthetic(), dense, flash] {
            assert_eq!(cfg.gdn_num_layers, cfg.gdn_layer_count() as u64, "{:?}", cfg.family);
        }
    }

    /// `fn_config.json`: hidden 2560, 24 query / 2 KV heads of 256, partial
    /// rotary 0.25 (64 of 256), θ 1e7, eps 1e-6; GDN 16 key / 48 value heads
    /// of 128, conv kernel 4; no dense MLP.
    #[test]
    fn flash_next_head_and_gdn_geometry_is_the_checkpoints() {
        let cfg = ModelConfig::qwen38_flash_next();
        assert_eq!(cfg.hidden, 2560);
        assert_eq!(cfg.vocab, 248_320);
        assert_eq!((cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim), (24, 2, 256));
        assert_eq!(cfg.rotary_dim, 64);
        assert_eq!(cfg.rope_theta, 1e7);
        assert_eq!(cfg.rms_norm_eps, 1e-6);
        // in_proj_qkv = q 2048 + k 2048 + v 6144; in_proj_z 6144; in_proj_a
        // and in_proj_b 48 each.
        assert_eq!(cfg.gdn_q_width, 2048);
        assert_eq!(cfg.gdn_state_cols, 2048);
        assert_eq!(cfg.gdn_state_rows, 6144);
        assert_eq!(cfg.gdn_z_width, 6144);
        assert_eq!(cfg.gdn_ab_width, 96);
        assert_eq!(cfg.gdn_head_dim, 128);
        assert_eq!(cfg.gdn_conv_channels(), 10_240);
        assert_eq!(cfg.gdn_in_proj_m(), 16_384);
        assert_eq!(cfg.gqa_width(), 6144);
        assert_eq!(cfg.gqa_kv_width(), 512);
        assert_eq!(cfg.ffn_intermediate, 0, "every Flash-Next layer is MoE");
        assert_eq!(cfg.block_size, 64);
        assert_eq!(cfg.block_size * cfg.num_blocks, 262_144, "max_position_embeddings");
    }

    /// `fn_config.json`: 512 experts, 10 per token, intermediate 640, a
    /// shared expert of 640.
    #[test]
    fn flash_next_moe_geometry() {
        let moe = ModelConfig::qwen38_flash_next().moe.expect("Flash-Next is MoE");
        assert_eq!(moe.num_experts, 512);
        assert_eq!(moe.experts_per_token, 10);
        assert_eq!(moe.expert_intermediate, 640);
        assert_eq!(moe.shared_expert_intermediate, 640);
        // gate_up_proj [512, 1280, 2560], down_proj [512, 2560, 640].
        assert_eq!(moe.expert_gate_up_rows(), 1280);
    }

    /// `fn_config.json`: hc_count 4, hc_lowrank 320. The residual is four
    /// streams of 2560 BF16 per token: 20,480 bytes, the 20 KB/token the
    /// study's RAM estimate used.
    #[test]
    fn flash_next_carries_four_hyper_connection_streams() {
        let cfg = ModelConfig::qwen38_flash_next();
        let hyper = cfg.hyper_connections.expect("Flash-Next has hyper-connections");
        assert_eq!((hyper.streams, hyper.rank), (4, 320));
        assert_eq!(cfg.residual_streams(), 4);
        assert_eq!(cfg.residual_width(), 10_240);
        assert_eq!(cfg.residual_bytes_per_token(), 20_480);
    }

    /// `fn_config.json`: indexer 4 heads of 128 over 1 key head, compress
    /// ratio 4, budget 2048. Attention stays dense while every complete
    /// block fits the budget: floor(V / 4) <= 512, so up to V = 2051 visible
    /// tokens (the checkpoint's own threshold, spec 04).
    #[test]
    fn flash_next_indexer_geometry_and_dense_threshold() {
        let cfg = ModelConfig::qwen38_flash_next();
        let indexer = cfg.indexer.expect("Flash-Next's QSA layers have an indexer");
        assert_eq!((indexer.heads, indexer.head_dim, indexer.kv_heads), (4, 128, 1));
        assert_eq!((indexer.compress_ratio, indexer.budget), (4, 2048));
        assert_eq!(indexer.block_top_k(), 512);
        assert_eq!(indexer.dense_threshold(), 2051);
        // index_qk_proj: (4 + 1) x 128 rows.
        assert_eq!(indexer.qk_projection_rows(), 640);
        // One compressed BF16 key of 128 per 4 tokens, in each of the 12 QSA
        // layers: 64 bytes per layer per token.
        assert_eq!(cfg.indexer_bytes_per_token(), 12 * 64);
    }

    /// `fn_config.json`: n-gram size 3, 8 heads per n-gram (16 heads in all,
    /// the 2-grams then the 3-grams), embedding 2560 (160 per head), conv
    /// kernel 4 at dilation 3 (the n-gram size), 128 table parts, base vocab
    /// 20,000,000 padded to a multiple of 128, seed 1234 (the config's
    /// default), EOS 248044. `ple_layer_ids` is one-indexed: `[2]` is the
    /// second decoder layer, index 1.
    #[test]
    fn flash_next_ngram_geometry() {
        let cfg = ModelConfig::qwen38_flash_next();
        let ngram = cfg.ngram.expect("Flash-Next has the n-gram embedding");
        assert_eq!((ngram.ngram_size, ngram.heads_per_ngram), (3, 8));
        assert_eq!(ngram.heads(), 16);
        assert_eq!(ngram.embed_dim, 2560);
        assert_eq!(ngram.head_dim(), 160);
        assert_eq!(ngram.context_tokens(), 2);
        assert_eq!((ngram.conv_kernel, ngram.conv_dilation()), (4, 3));
        assert_eq!(ngram.conv_state_tokens(), 9);
        assert_eq!(ngram.layer, 1);
        assert_eq!(ngram.split_parts, 128);
        assert_eq!((ngram.vocab_size_base, ngram.vocab_divisor), (20_000_000, 128));
        assert_eq!(ngram.seed, 1234);
        assert_eq!(ngram.eos_token_id, 248_044);
        // The dilated conv runs over the PLE's output, one channel per
        // stream feature: 9 past columns of 4 x 2560 BF16 per lane.
        assert_eq!(cfg.ngram_conv_state_bytes(), 9 * 10_240 * 2);
    }

    /// Spec 04's KV numbers: 24,576 bytes per token under BF16 and 3,456
    /// under hq-e8-2b (factor 7.11, as for the 27B's 65,536 / 9,216).
    #[test]
    fn kv_bytes_per_token_derive_from_the_topology() {
        let flash = ModelConfig::qwen38_flash_next().kv_geometry();
        assert_eq!(flash, KvGeometry { gqa_layers: 12, num_kv_heads: 2, head_dim: 256 });
        assert_eq!(KvFormat::Bf16.bytes_per_token(flash), 24_576);
        assert_eq!(KvFormat::HqE8_2b.bytes_per_token(flash), 3_456);
        let dense = ModelConfig::qwen38_27b().kv_geometry();
        assert_eq!(dense, KvGeometry::qwen38_27b());
        assert_eq!(KvFormat::Bf16.bytes_per_token(dense), 65_536);
        assert_eq!(KvFormat::HqE8_2b.bytes_per_token(dense), 9_216);
    }

    /// The 27B's topology is exactly what it was before Flash-Next: every
    /// field, and none of Flash-Next's blocks.
    #[test]
    fn the_27b_topology_is_unchanged() {
        let cfg = ModelConfig::qwen38_27b();
        assert_eq!(cfg.family, ModelFamily::Qwen38_27b);
        assert_eq!(cfg.num_layers, 64);
        assert_eq!(cfg.attention_layer_count(), 16);
        assert_eq!(cfg.gdn_layer_count(), 48);
        assert_eq!((cfg.hidden, cfg.vocab), (5120, 248_320));
        assert_eq!((cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim), (24, 4, 256));
        assert_eq!((cfg.gdn_state_rows, cfg.gdn_state_cols), (6144, 2048));
        assert_eq!((cfg.gdn_q_width, cfg.gdn_z_width, cfg.gdn_ab_width), (2048, 6144, 96));
        assert_eq!((cfg.gdn_value_heads, cfg.gdn_head_dim), (48, 128));
        assert_eq!((cfg.rotary_dim, cfg.rope_theta), (64, 1e7));
        assert_eq!(cfg.ffn_intermediate, 17_408);
        assert_eq!((cfg.block_size, cfg.num_blocks), (64, 4096));
        assert_eq!(cfg.rms_norm_eps, 1e-6);
        assert!(cfg.moe.is_none() && cfg.hyper_connections.is_none());
        assert!(cfg.indexer.is_none() && cfg.ngram.is_none());
        assert_eq!(cfg.residual_streams(), 1);
        assert_eq!(cfg.residual_bytes_per_token(), 10_240);
        assert_eq!(cfg.indexer_bytes_per_token(), 0);
        assert_eq!(cfg.ngram_conv_state_bytes(), 0);
    }

    fn kinds_of(abi: &TopologyAbi) -> Vec<i32> {
        let raw = abi.raw();
        unsafe { std::slice::from_raw_parts(raw.layer_kinds, raw.num_layers as usize) }.to_vec()
    }

    /// The 27B's descriptor is the one `model_load` built before Flash-Next
    /// (every field below is its old literal), with the value heads that
    /// field held implicitly and none of Flash-Next's blocks.
    #[test]
    fn the_27b_topology_crosses_the_abi_field_for_field() {
        let abi = ModelConfig::qwen38_27b().topology_abi();
        let raw = abi.raw();
        assert_eq!(raw.num_layers, 64);
        let kinds = kinds_of(&abi);
        assert_eq!(kinds.len(), 64);
        for (i, &kind) in kinds.iter().enumerate() {
            assert_eq!(kind, if (i + 1) % 4 == 0 { 1 } else { 0 }, "layer {i}: GQA = 1, GDN = 0");
        }
        assert_eq!((raw.hidden, raw.vocab), (5120, 248_320));
        assert_eq!((raw.num_q_heads, raw.num_kv_heads, raw.head_dim), (24, 4, 256));
        assert_eq!((raw.rotary_dim, raw.rope_theta), (64, 1e7));
        assert_eq!((raw.gdn_state_rows, raw.gdn_state_cols, raw.gdn_num_layers), (6144, 2048, 48));
        assert_eq!((raw.gdn_q_width, raw.gdn_z_width, raw.gdn_ab_width), (2048, 6144, 96));
        assert_eq!(raw.ffn_intermediate, 17_408);
        assert_eq!(raw.rms_norm_eps, 1.0e-6);
        assert_eq!(raw.family, 0, "IGNIS_MODEL_FAMILY_QWEN38_27B");
        assert_eq!(raw.gdn_value_heads, 48);
        assert_eq!(raw.moe, IgnisMoeTopology::default());
        assert_eq!(raw.hyper, IgnisHyperTopology::default());
        assert_eq!(raw.indexer, IgnisIndexerTopology::default());
        assert_eq!(raw.ngram, IgnisNgramTopology::default());
    }

    /// Flash-Next's descriptor carries its 36 GDN layers and its 48 GDN value
    /// heads as two fields, and every block the leaf will derive its buffers
    /// from.
    #[test]
    fn flash_next_crosses_the_abi_with_its_gdn_layers_and_value_heads_apart() {
        let abi = ModelConfig::qwen38_flash_next().topology_abi();
        let raw = abi.raw();
        assert_eq!(raw.family, 1, "IGNIS_MODEL_FAMILY_FLASH_NEXT");
        assert_eq!(raw.num_layers, 48);
        assert_eq!(kinds_of(&abi).iter().filter(|&&kind| kind == 1).count(), 12);
        assert_eq!((raw.gdn_num_layers, raw.gdn_value_heads), (36, 48));
        assert_eq!((raw.hidden, raw.num_kv_heads, raw.ffn_intermediate), (2560, 2, 0));
        assert_eq!(
            raw.moe,
            IgnisMoeTopology { num_experts: 512, experts_per_token: 10, expert_intermediate: 640, shared_expert_intermediate: 640 }
        );
        assert_eq!(raw.hyper, IgnisHyperTopology { streams: 4, rank: 320 });
        assert_eq!(
            raw.indexer,
            IgnisIndexerTopology { heads: 4, head_dim: 128, kv_heads: 1, compress_ratio: 4, budget: 2048 }
        );
        assert_eq!(
            raw.ngram,
            IgnisNgramTopology { ngram_size: 3, heads_per_ngram: 8, embed_dim: 2560, conv_kernel: 4, layer: 1 }
        );
    }

    /// `struct ignis_topology` has no `size` field, so both sides pin its
    /// layout: kernel/include/ignis_model.h static_asserts the same size and
    /// offsets. The fields before `family` keep the offsets they had.
    #[test]
    fn the_topology_descriptor_is_the_leafs_layout() {
        use std::mem::{offset_of, size_of};
        assert_eq!(offset_of!(IgnisTopology, rms_norm_eps), 128);
        assert_eq!(offset_of!(IgnisTopology, family), 132);
        assert_eq!(offset_of!(IgnisTopology, gdn_value_heads), 136);
        assert_eq!(offset_of!(IgnisTopology, moe), 144);
        assert_eq!(offset_of!(IgnisTopology, hyper), 176);
        assert_eq!(offset_of!(IgnisTopology, indexer), 192);
        assert_eq!(offset_of!(IgnisTopology, ngram), 232);
        assert_eq!(size_of::<IgnisTopology>(), 272);
    }

    /// `gdn_value_heads * gdn_head_dim` must equal `gdn_state_rows` (the
    /// sequence-state pool's per-layer slot is `gdn_value_heads` square
    /// `gdn_head_dim x gdn_head_dim` matrices, GitHub #55) — pins the
    /// relationship so the two representations cannot silently drift.
    #[test]
    fn gdn_value_heads_and_head_dim_agree_with_state_rows() {
        for cfg in [ModelConfig::synthetic(), ModelConfig::qwen38_27b(), ModelConfig::qwen38_flash_next()] {
            assert_eq!(
                cfg.gdn_value_heads * cfg.gdn_head_dim,
                cfg.gdn_state_rows,
                "gdn_value_heads * gdn_head_dim must equal gdn_state_rows"
            );
        }
    }
}
