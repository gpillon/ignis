//! The KV cache storage format and the byte-budget pool plan it implies
//! (ADR 0022, GitHub #122, P4-04).
//!
//! Two formats exist, and which one a model load runs on is a load option
//! fixed for the life of that load: [`KvFormat::Bf16`] (what every
//! correctness oracle loads) and [`KvFormat::HqE8_2b`] (the reference's own
//! serving format). Both keep the paged-KV contract's fixed-bytes-per-token
//! property, so the only thing the format changes about the pool is how
//! many bytes one (token, KV head) row costs.
//!
//! Because of that, the pool is described in **bytes**, never in tokens: a
//! byte budget buys a whole number of physical pages, and the token
//! capacity those pages hold is *derived* from the format in force
//! ([`plan_kv_pool`]). No token target is compiled in anywhere — change the
//! format and the same budget reports a different capacity, which is the
//! entire reason the format is in phase 4 (`docs/findings/2026-09-11-hq-e8-2b-kv-capacity.md`).
//!
//! The per-page byte formula here is the vendored planner's own
//! (`plan_paged_kv_pool`: `dtype_size * leading_extent * kPagedKVPageSize *
//! head_extent`, summed over the planes, as `kernel/src/paged_kv_budget.cu`
//! records). This module states the *planes* a format is made of, not a
//! byte count; `crates/core/tests/kv_format_leaf_agreement.rs` cross-checks
//! the arithmetic against the leaf's own `ignis_paged_kv_page_budget` so a
//! change to the reference's layout math cannot drift past it.

/// Tokens held by one physical KV page (`kPagedKVPageSize`,
/// `kernel/vendor/src/core/paged_kv_cache.h`). Fixed by the vendored
/// header, identical in both formats.
pub const KV_PAGE_TOKENS: u32 = 64;

/// The element type of a KV storage plane. Only two occur — the formats
/// store either the BF16 values themselves or the codec's packed bytes —
/// and naming them as a closed set is what keeps [`KvPlaneSpec::page_bytes`]
/// total rather than fallible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPlaneDtype {
    Bf16,
    U8,
}

impl KvPlaneDtype {
    /// `ninfer::DType`'s ordinal, as `struct ignis_paged_kv_plane` carries
    /// it (`kernel/include/ignis_paged_kv_budget.h`): 0 BF16, 3 U8.
    pub fn abi_code(&self) -> i32 {
        match self {
            Self::Bf16 => 0,
            Self::U8 => 3,
        }
    }

    pub fn bytes(&self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::U8 => 1,
        }
    }
}

/// The hq-e8-2b per-row code budget, in bytes (`kHqRowBudgetBytes`,
/// `kernel/vendor/src/ops/kernel/hq_codec.cuh`): every (token, KV head)
/// row occupies exactly this much code plane, whatever it contains.
pub const HQ_ROW_CODE_BYTES: u32 = 64;
/// The hq-e8-2b per-row metadata budget, in bytes (`kHqMetaBytes`): the
/// row norm, the Rice parameter, the used-bit count and the group-decode
/// segment offsets.
pub const HQ_ROW_META_BYTES: u32 = 8;

/// The default paged-KV pool byte budget when the operator names none.
///
/// 4 GiB — chosen so the BF16 pool is byte-for-byte the one this engine
/// already shipped (65,536 sequence-tokens at 65,536 bytes each, the old
/// `DEFAULT_KV_POOL_TOKENS`), while the same budget under hq-e8-2b buys
/// 7.11x the tokens with nothing else changed. The default is a budget, not
/// a token target: the capacity it reports moves with the format.
pub const DEFAULT_KV_POOL_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// One storage plane's per-page geometry — what a format is *made of*,
/// rather than a byte count derived from it. 1:1 with
/// `struct ignis_paged_kv_plane` / `ninfer::PagedKVPlaneSpec`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPlaneSpec {
    pub dtype: KvPlaneDtype,
    /// The plane's per-row extent (head_dim for BF16 values, the row byte
    /// budget for an hq code or metadata plane).
    pub leading_extent: u32,
    /// KV heads carried by the plane.
    pub head_extent: u32,
}

impl KvPlaneSpec {
    /// Bytes this plane occupies in one physical page — the vendored
    /// planner's own per-plane storage
    /// (`dtype_size * leading_extent * kPagedKVPageSize * head_extent`).
    pub fn page_bytes(&self) -> u64 {
        self.dtype.bytes()
            * u64::from(self.leading_extent)
            * u64::from(KV_PAGE_TOKENS)
            * u64::from(self.head_extent)
    }
}

/// The paged-KV geometry a pool is planned against: how many full-attention
/// layers store K and V, and each layer's head shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvGeometry {
    /// Full-attention (GQA) layers, each storing its own K/V pair
    /// (`kIgnisGqaLayerCount`).
    pub gqa_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
}

impl KvGeometry {
    /// The Qwen 3.8-27B geometry (`CONTEXT.md`): 16 GQA layers, 4 KV heads
    /// of 256.
    pub fn qwen38_27b() -> Self {
        Self {
            gqa_layers: 16,
            num_kv_heads: 4,
            head_dim: 256,
        }
    }
}

/// The KV cache storage format, fixed for the life of a model load
/// (ADR 0022).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvFormat {
    /// Unquantized BF16 K/V rows. The format every correctness oracle
    /// loads, and — until the hq attention routes land (GitHub #123) — the
    /// only one that serves a token.
    #[default]
    Bf16,
    /// The reference's HyperQuant KV format: a fixed 64-byte code row plus
    /// an 8-byte metadata row per (token, KV head).
    HqE8_2b,
}

impl KvFormat {
    /// The spelling the CLI, the logs and the load report use.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::HqE8_2b => "hq-e8-2b",
        }
    }

    /// `enum ignis_kv_format`'s value (`kernel/include/ignis_seq.h`) —
    /// keep 1:1.
    pub fn abi_code(&self) -> i32 {
        match self {
            Self::Bf16 => 0,
            Self::HqE8_2b => 1,
        }
    }

    /// Parse an operator-supplied spelling (`--kv-format`,
    /// `IGNIS_KV_FORMAT`). Case-insensitive, and `hq` is accepted as a
    /// short form of the only hq profile that exists.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "bf16" => Ok(Self::Bf16),
            "hq-e8-2b" | "hq_e8_2b" | "hq" => Ok(Self::HqE8_2b),
            other => Err(format!(
                "unknown KV format `{other}` (expected `bf16` or `hq-e8-2b`)"
            )),
        }
    }

    /// The storage planes one GQA layer's K/V history occupies, in the
    /// order the pool allocates them.
    ///
    /// BF16 is one plane per role. hq-e8-2b is two per role — a code plane
    /// and a metadata plane — which is the *only* structural difference
    /// between the formats: the page addressing over them is identical
    /// (`paged_kv_element_offset` with the plane's own leading extent).
    pub fn planes_per_gqa_layer(&self, geometry: KvGeometry) -> Vec<KvPlaneSpec> {
        let heads = geometry.num_kv_heads;
        match self {
            Self::Bf16 => vec![
                // K, then V.
                KvPlaneSpec {
                    dtype: KvPlaneDtype::Bf16,
                    leading_extent: geometry.head_dim,
                    head_extent: heads,
                },
                KvPlaneSpec {
                    dtype: KvPlaneDtype::Bf16,
                    leading_extent: geometry.head_dim,
                    head_extent: heads,
                },
            ],
            Self::HqE8_2b => vec![
                // K codes, K metadata, V codes, V metadata.
                KvPlaneSpec {
                    dtype: KvPlaneDtype::U8,
                    leading_extent: HQ_ROW_CODE_BYTES,
                    head_extent: heads,
                },
                KvPlaneSpec {
                    dtype: KvPlaneDtype::U8,
                    leading_extent: HQ_ROW_META_BYTES,
                    head_extent: heads,
                },
                KvPlaneSpec {
                    dtype: KvPlaneDtype::U8,
                    leading_extent: HQ_ROW_CODE_BYTES,
                    head_extent: heads,
                },
                KvPlaneSpec {
                    dtype: KvPlaneDtype::U8,
                    leading_extent: HQ_ROW_META_BYTES,
                    head_extent: heads,
                },
            ],
        }
    }

    /// Every plane the pool allocates, layer-major.
    pub fn pool_planes(&self, geometry: KvGeometry) -> Vec<KvPlaneSpec> {
        let per_layer = self.planes_per_gqa_layer(geometry);
        let mut planes = Vec::with_capacity(per_layer.len() * geometry.gqa_layers as usize);
        for _ in 0..geometry.gqa_layers {
            planes.extend_from_slice(&per_layer);
        }
        planes
    }

    /// Bytes one physical page costs across every plane of every GQA layer.
    pub fn page_bytes(&self, geometry: KvGeometry) -> u64 {
        self.planes_per_gqa_layer(geometry)
            .iter()
            .map(KvPlaneSpec::page_bytes)
            .sum::<u64>()
            * u64::from(geometry.gqa_layers)
    }

    /// Bytes one sequence-token costs across every GQA layer and both
    /// roles. 65,536 under BF16 and 9,216 under hq-e8-2b at this model's
    /// geometry.
    pub fn bytes_per_token(&self, geometry: KvGeometry) -> u64 {
        self.page_bytes(geometry) / u64::from(KV_PAGE_TOKENS)
    }
}

impl std::fmt::Display for KvFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a byte budget bought, under one format: the pool the leaf will
/// build and the token capacity it holds. Every field is derived — nothing
/// here is configured, and nothing is a constant the allocator enforces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPoolPlan {
    pub format: KvFormat,
    /// The budget this plan was asked to fit inside.
    pub budget_bytes: u64,
    /// Physical pages the budget buys (`kv_page_group_count`).
    pub page_count: u32,
    /// Bytes of one physical page across every plane.
    pub page_bytes: u64,
    /// Bytes one sequence-token costs in this format.
    pub bytes_per_token: u64,
    /// Resident sequence-tokens the pool holds — the derived number the
    /// load reports.
    pub token_capacity: u64,
    /// Bytes the pool actually occupies (`<= budget_bytes`).
    pub pool_bytes: u64,
}

/// Plan a pool from a byte budget under `format`: how many whole pages fit,
/// and the token capacity they hold.
///
/// The page count saturates at `u32::MAX` the same way the leaf's own
/// budget query does; a budget smaller than one page plans an empty pool
/// (which [`plan_kv_pool_for_context`] then refuses, naming the numbers).
pub fn plan_kv_pool(format: KvFormat, geometry: KvGeometry, budget_bytes: u64) -> KvPoolPlan {
    let page_bytes = format.page_bytes(geometry);
    let page_count = (budget_bytes / page_bytes).min(u64::from(u32::MAX)) as u32;
    KvPoolPlan {
        format,
        budget_bytes,
        page_count,
        page_bytes,
        bytes_per_token: format.bytes_per_token(geometry),
        token_capacity: u64::from(page_count) * u64::from(KV_PAGE_TOKENS),
        pool_bytes: u64::from(page_count) * page_bytes,
    }
}

/// A byte budget that cannot hold one full-context sequence under the
/// format in force (GitHub #122): a **load** failure, not a later admission
/// refusal — a pool the per-sequence cap does not fit inside would admit
/// requests the leaf can never allocate pages for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvBudgetTooSmall {
    pub format: KvFormat,
    pub budget_bytes: u64,
    /// The capacity the budget actually bought.
    pub token_capacity: u64,
    /// The per-sequence context the pool has to be able to hold.
    pub required_tokens: u64,
    /// The budget that would have held `required_tokens`.
    pub required_bytes: u64,
}

impl std::fmt::Display for KvBudgetTooSmall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KV pool budget of {} bytes under {} KV buys {} resident tokens, \
             short of the {}-token context this load is configured for \
             (it needs at least {} bytes)",
            self.budget_bytes,
            self.format,
            self.token_capacity,
            self.required_tokens,
            self.required_bytes,
        )
    }
}

impl std::error::Error for KvBudgetTooSmall {}

/// Plan a pool that must be able to hold one `max_context_tokens`
/// sequence, or refuse the load naming the budget, the format and the
/// capacity it bought.
pub fn plan_kv_pool_for_context(
    format: KvFormat,
    geometry: KvGeometry,
    budget_bytes: u64,
    max_context_tokens: u32,
) -> Result<KvPoolPlan, KvBudgetTooSmall> {
    let plan = plan_kv_pool(format, geometry, budget_bytes);
    let required = u64::from(max_context_tokens);
    if plan.token_capacity < required {
        return Err(KvBudgetTooSmall {
            format,
            budget_bytes,
            token_capacity: plan.token_capacity,
            required_tokens: required,
            required_bytes: context_bytes(format, geometry, max_context_tokens),
        });
    }
    Ok(plan)
}

/// The smallest byte budget whose whole pages hold `tokens` — the bound the
/// auto default and the too-small message are both stated in.
fn context_bytes(format: KvFormat, geometry: KvGeometry, tokens: u32) -> u64 {
    u64::from(tokens.div_ceil(KV_PAGE_TOKENS)) * format.page_bytes(geometry)
}

/// The pool budget to use when the operator named none: the
/// [`DEFAULT_KV_POOL_BYTES`] floor, raised if a single configured context
/// would not fit inside it under this format.
///
/// Format-aware rather than a token target: the same call returns 4 GiB for
/// a 40,960-token BF16 load and 4 GiB for the hq load that gets 7.11x the
/// tokens out of it.
pub fn auto_kv_pool_bytes(format: KvFormat, geometry: KvGeometry, max_context_tokens: u32) -> u64 {
    DEFAULT_KV_POOL_BYTES.max(context_bytes(format, geometry, max_context_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN: KvGeometry = KvGeometry {
        gqa_layers: 16,
        num_kv_heads: 4,
        head_dim: 256,
    };

    #[test]
    fn a_sequence_token_costs_what_the_capacity_finding_measured() {
        // `docs/findings/2026-09-11-hq-e8-2b-kv-capacity.md`: 65,536 bytes
        // under BF16, 9,216 under hq — derived here from the planes, not
        // restated as a constant.
        assert_eq!(KvFormat::Bf16.bytes_per_token(QWEN), 65_536);
        assert_eq!(KvFormat::HqE8_2b.bytes_per_token(QWEN), 9_216);
    }

    #[test]
    fn hq_is_denser_by_the_factor_the_finding_recorded() {
        let bf16 = KvFormat::Bf16.bytes_per_token(QWEN) as f64;
        let hq = KvFormat::HqE8_2b.bytes_per_token(QWEN) as f64;
        // 7.11, not the 8x the "2 bits per dimension" headline suggests:
        // the 8-byte metadata row costs 12.5% on top of the code budget.
        assert!((bf16 / hq - 7.111).abs() < 0.001, "ratio {}", bf16 / hq);
    }

    #[test]
    fn the_pool_allocates_two_planes_per_layer_under_bf16_and_four_under_hq() {
        assert_eq!(KvFormat::Bf16.pool_planes(QWEN).len(), 32);
        assert_eq!(KvFormat::HqE8_2b.pool_planes(QWEN).len(), 64);
    }

    #[test]
    fn the_hq_planes_carry_the_codecs_fixed_row_budget() {
        let planes = KvFormat::HqE8_2b.planes_per_gqa_layer(QWEN);
        let extents: Vec<u32> = planes.iter().map(|p| p.leading_extent).collect();
        // K codes, K meta, V codes, V meta.
        assert_eq!(extents, vec![64, 8, 64, 8]);
        assert!(planes.iter().all(|p| p.dtype == KvPlaneDtype::U8));
    }

    #[test]
    fn token_capacity_is_derived_from_the_budget_and_the_format() {
        // One budget, two formats, two capacities — the whole point of
        // describing the pool in bytes.
        let budget = DEFAULT_KV_POOL_BYTES;
        let bf16 = plan_kv_pool(KvFormat::Bf16, QWEN, budget);
        let hq = plan_kv_pool(KvFormat::HqE8_2b, QWEN, budget);
        assert_eq!(bf16.token_capacity, 65_536);
        assert_eq!(hq.token_capacity, 465_984);
        assert_eq!(bf16.page_count, 1_024);
        assert_eq!(hq.page_count, 7_281);
        assert!(bf16.pool_bytes <= budget && hq.pool_bytes <= budget);
    }

    #[test]
    fn the_default_budget_reproduces_the_engines_previous_bf16_pool() {
        // The pool this engine shipped before the format was an option:
        // 65,536 sequence-tokens (the old `DEFAULT_KV_POOL_TOKENS`).
        let plan = plan_kv_pool(KvFormat::Bf16, QWEN, DEFAULT_KV_POOL_BYTES);
        assert_eq!(plan.token_capacity, 65_536);
    }

    #[test]
    fn the_standard_target_profile_holds_eight_full_contexts_under_hq() {
        // The gate's requirement (GitHub #122): at least 8 x 40,960 =
        // 327,680 resident tokens, read off the derived capacity rather
        // than from any constant in the allocator.
        let budget = auto_kv_pool_bytes(KvFormat::HqE8_2b, QWEN, 40_960);
        let plan = plan_kv_pool(KvFormat::HqE8_2b, QWEN, budget);
        assert!(
            plan.token_capacity >= 8 * 40_960,
            "hq capacity {} is short of 8 x 40,960",
            plan.token_capacity
        );
        // And the same profile under BF16 does not — which is the
        // inequality every gate so far recorded.
        let bf16_budget = auto_kv_pool_bytes(KvFormat::Bf16, QWEN, 40_960);
        assert!(plan_kv_pool(KvFormat::Bf16, QWEN, bf16_budget).token_capacity < 8 * 40_960);
    }

    #[test]
    fn a_bigger_budget_buys_more_tokens_in_either_format() {
        for format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
            let small = plan_kv_pool(format, QWEN, DEFAULT_KV_POOL_BYTES);
            let big = plan_kv_pool(format, QWEN, 2 * DEFAULT_KV_POOL_BYTES);
            // Doubling the budget doubles the pages, up to the partial page
            // a budget that is not a whole multiple of the page size leaves
            // on the table.
            assert!(
                big.page_count == 2 * small.page_count
                    || big.page_count == 2 * small.page_count + 1,
                "{format}: {} pages then {}",
                small.page_count,
                big.page_count
            );
            assert!(big.pool_bytes <= 2 * DEFAULT_KV_POOL_BYTES, "{format}");
        }
    }

    #[test]
    fn the_auto_default_always_holds_one_configured_context() {
        for format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
            for context in [1_024u32, 40_960, 400_000, 1_000_000] {
                let budget = auto_kv_pool_bytes(format, QWEN, context);
                let plan = plan_kv_pool_for_context(format, QWEN, budget, context)
                    .unwrap_or_else(|e| panic!("{format} at {context}: {e}"));
                assert!(plan.token_capacity >= u64::from(context));
            }
        }
    }

    #[test]
    fn a_budget_too_small_for_the_context_fails_naming_the_numbers() {
        // One 40,960-token sequence needs 640 hq pages; this budget buys 2.
        let budget = 2 * KvFormat::HqE8_2b.page_bytes(QWEN);
        let err = plan_kv_pool_for_context(KvFormat::HqE8_2b, QWEN, budget, 40_960)
            .expect_err("a budget this small cannot hold the configured context");
        assert_eq!(err.token_capacity, 128);
        assert_eq!(err.required_tokens, 40_960);
        let message = err.to_string();
        assert!(message.contains(&budget.to_string()), "{message}");
        assert!(message.contains("hq-e8-2b"), "{message}");
        assert!(message.contains("128"), "{message}");
    }

    #[test]
    fn a_budget_that_is_only_big_enough_under_hq_fails_under_bf16() {
        // The same budget, the same context: the format decides whether the
        // load starts. This is the option being a real option.
        let budget = 8 * 1024 * 1024 * 1024u64;
        let context = 400_000u32;
        assert!(plan_kv_pool_for_context(KvFormat::HqE8_2b, QWEN, budget, context).is_ok());
        assert!(plan_kv_pool_for_context(KvFormat::Bf16, QWEN, budget, context).is_err());
    }

    #[test]
    fn format_spellings_round_trip() {
        assert_eq!(KvFormat::parse("bf16"), Ok(KvFormat::Bf16));
        assert_eq!(KvFormat::parse("BF16"), Ok(KvFormat::Bf16));
        assert_eq!(KvFormat::parse("hq-e8-2b"), Ok(KvFormat::HqE8_2b));
        assert_eq!(KvFormat::parse(" hq "), Ok(KvFormat::HqE8_2b));
        assert_eq!(KvFormat::Bf16.as_str(), "bf16");
        assert_eq!(KvFormat::HqE8_2b.as_str(), "hq-e8-2b");
        let err = KvFormat::parse("int8").expect_err("int8 KV is not a format this engine stores");
        assert!(err.contains("int8") && err.contains("hq-e8-2b"), "{err}");
    }

    #[test]
    fn the_abi_codes_match_the_leaf_enum() {
        // 1:1 with `enum ignis_kv_format` (kernel/include/ignis_seq.h).
        assert_eq!(KvFormat::Bf16.abi_code(), 0);
        assert_eq!(KvFormat::HqE8_2b.abi_code(), 1);
    }

    #[test]
    fn bf16_is_the_default_format_until_the_hq_routes_land() {
        // GitHub #123 owns the attention routes; until then a load that
        // names no format must be one that can serve a token.
        assert_eq!(KvFormat::default(), KvFormat::Bf16);
    }
}
