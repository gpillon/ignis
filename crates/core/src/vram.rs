//! The load's device-memory plan within a **VRAM budget** (GitHub #210,
//! ADR 0030, `CONTEXT.md`).
//!
//! Pure arithmetic over numbers the loader has already measured or asked the
//! leaf for: what was free at start, what each fixed reservation costs, and
//! what one KV pool of `n` pages occupies. It decides the budget, sizes the
//! KV pool by the **KV pool policy** (ADR 0045: the rest of the budget when
//! every weight is on the device, a default reserved first when experts
//! stream), and refuses a start that cannot hold one sequence at the maximum
//! context beside every retained slot's tail page — before a byte of the plan
//! is allocated. Nothing here touches a device, so every branch, refusal and
//! warning is pinned on the CPU.

use crate::kv_format::{KV_PAGE_TOKENS, KvFormat, KvGeometry};

/// The offloaded KV pool's default (ADR 0045, owner): the tokens a load whose
/// experts stream reserves before its expert cache takes the rest.
pub const OFFLOADED_KV_POOL_TOKENS: u32 = 524_288;

/// How a load's KV pool is sized (ADR 0045): by whether every weight is on
/// the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPoolPolicy {
    /// Every weight is on the device (the 27B; Flash-Next on a card that
    /// holds every expert): the KV pool takes the rest of the budget.
    Resident,
    /// Experts stream over PCIe (Flash-Next on the 5090): the KV pool is
    /// reserved first, [`OFFLOADED_KV_POOL_TOKENS`] by default, and the
    /// expert cache takes the rest.
    Offloaded,
}

impl KvPoolPolicy {
    /// The `kv_pool_policy` field of the plan events.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::Offloaded => "offloaded",
        }
    }
}

/// A KV pool the operator named (`--kv-pool-bytes`, ADR 0045): a byte count,
/// or a token count. The plan turns either into pages, since only it knows
/// what a page costs on the model it loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPoolSize {
    Bytes(u64),
    Tokens(u64),
}

impl KvPoolSize {
    /// The pages this size names on a load whose page costs `page_bytes`:
    /// the whole pages a byte count buys, as the leaf's pool cuts them
    /// ([`crate::plan_kv_pool`]), or enough pages to hold a token count.
    pub fn pages(&self, page_bytes: u64) -> u32 {
        let pages = match *self {
            Self::Bytes(bytes) => bytes.checked_div(page_bytes).unwrap_or(0),
            Self::Tokens(tokens) => tokens.div_ceil(u64::from(KV_PAGE_TOKENS)),
        };
        pages.min(u64::from(u32::MAX)) as u32
    }
}

impl std::fmt::Display for KvPoolSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bytes(bytes) => write!(f, "{bytes} bytes"),
            Self::Tokens(tokens) => write!(f, "{tokens} tokens"),
        }
    }
}

/// Where a load's weights live, which decides its [`KvPoolPolicy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// Every weight is a line of the plan (the 27B): **Resident** by
    /// construction.
    Resident,
    /// A model whose experts the device may or may not hold (Flash-Next).
    Experts(ExpertResidency),
}

/// What decides whether a load with experts is resident, and what its
/// offloaded pool is capped at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertResidency {
    /// Every expert projection, as the expert cache's class pools hold them
    /// all ([`crate::residency::ExpertCatalog::total_bytes`]).
    pub expert_pool_bytes: u64,
    /// Residency's own VRAM lines, the prefill staging ring and the tables:
    /// placed on either branch.
    pub fixed_bytes: u64,
    /// The decode lanes: the offloaded default never exceeds their whole
    /// contexts, since a larger pool could only hold retained pages.
    pub decode_lanes: u32,
}

/// How the budget is chosen: `--vram-headroom-bytes` (derived, the default)
/// or `--vram-budget-bytes` (explicit), never both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VramMode {
    /// Free memory at start minus the **VRAM headroom**.
    Derived { headroom_bytes: u64 },
    /// The operator's figure, the whole process included. More than is free
    /// refuses the start unless `allow_oversubscription`
    /// (`--allow-vram-oversubscription`) turns that into a warning.
    Explicit {
        budget_bytes: u64,
        allow_oversubscription: bool,
    },
}

impl VramMode {
    /// The `mode` field of `ignis.runtime.vram_plan`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Derived { .. } => "derived",
            Self::Explicit { .. } => "explicit",
        }
    }

    fn allows_oversubscription(&self) -> bool {
        matches!(
            self,
            Self::Explicit {
                allow_oversubscription: true,
                ..
            }
        )
    }
}

/// Every reservation the plan places before the KV pool, in bytes, in plan
/// order (weights, workspaces, lanes, retained state). The KV pool takes
/// what they leave.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VramLines {
    /// The artifact's device arena.
    pub weights: u64,
    /// The CUDA context: the budget, like Task Manager, counts it.
    pub cuda_context: u64,
    /// The one scratch arena prefill chunks and media encode share (GitHub
    /// #212): one `--prefill-chunk` span's scratch, or with `--vision` the
    /// encoder's workspace when that is larger. The two are never live at
    /// once.
    pub workspace: u64,
    /// One media item's encoder output (0 without `--vision`).
    pub media_embedding: u64,
    /// Device sampling's staging buffers and workspace.
    pub sampling: u64,
    /// The decode round's scratch and staging.
    pub decode_graph: u64,
    /// The verify round (0 without `--spec`).
    pub verify_round: u64,
    /// The drafter's round buffers (0 without the DFlash2 drafter).
    pub drafter_round: u64,
    /// Every lane's mutable state in the sequence pool: GDN recurrent and
    /// conv state, penalty counts, the drafter's window.
    pub lane_state: u64,
    /// The sequence pool's device retained slots (GitHub #211, #215,
    /// `--retained-device`): a lane's state each, reserved at load beside the
    /// lanes, holding retained prompt checkpoints' and shared prefixes'
    /// images. The host slots (`--retained-host`, GitHub #281) live in
    /// pinned host memory and are no line of this plan.
    pub retained_slots: u64,
    /// The hq-e8-2b residual window (GitHub #257, spec runtime/06): every
    /// slot's exact sink and recent-ring K/V rows and its ring validity
    /// words, lanes and retained slots alike -- about 34 MiB per slot. 0 on a
    /// BF16 load, which keeps no window.
    pub hq_residual_window: u64,
    /// What a load holds beyond every line above: allocator rounding, the
    /// decode graph captures, the kernel's lazily created handles. Measured,
    /// not derived.
    pub residual: u64,
}

impl VramLines {
    /// How many lines a plan has. The arity of [`VramLines::entries`], named
    /// so a reader of the plan does not have to write the number out again.
    pub const LINES: usize = 12;

    /// `(name, bytes)` in plan order; the names are the `*_bytes` fields of
    /// `ignis.runtime.vram_plan` without the suffix.
    pub fn entries(&self) -> [(&'static str, u64); Self::LINES] {
        [
            ("weights", self.weights),
            ("cuda_context", self.cuda_context),
            ("workspace", self.workspace),
            ("media_embedding", self.media_embedding),
            ("sampling", self.sampling),
            ("decode_graph", self.decode_graph),
            ("verify_round", self.verify_round),
            ("drafter_round", self.drafter_round),
            ("lane_state", self.lane_state),
            ("retained_slots", self.retained_slots),
            ("hq_residual_window", self.hq_residual_window),
            ("residual", self.residual),
        ]
    }

    /// Every line added up.
    pub fn total(&self) -> u64 {
        self.entries()
            .iter()
            .fold(0u64, |sum, (_, bytes)| sum.saturating_add(*bytes))
    }
}

/// What the loader knows before it allocates anything.
pub struct VramRequest<'a> {
    pub mode: VramMode,
    /// Device memory free before the first reservation.
    pub free_at_start_bytes: u64,
    pub lines: VramLines,
    /// One KV page's payload bytes on this load: what a named byte count is
    /// divided by and what the rest is cut into (the format's page on the
    /// 27B, every paged section's -- KV and indexer keys -- on Flash-Next).
    pub kv_page_bytes: u64,
    /// `--max-context`: the plan must hold one sequence this long.
    pub max_context_tokens: u32,
    /// Every retained slot, `--retained-device` and `--retained-host`
    /// together (GitHub #215, #281). The KV pool must also hold one page per
    /// slot, whichever kind: a prompt checkpoint keeps the page its opener
    /// ends inside, and a request claiming it cannot take that page back
    /// while it stands on the checkpoint — so a lone `--max-context` sequence
    /// has to fit beside every one.
    pub retained_slots: u32,
    /// `--kv-pool-bytes`, when the operator named it, in bytes or tokens: it
    /// replaces the policy's size (ADR 0045). `None` takes the policy's.
    pub kv_pool: Option<KvPoolSize>,
    /// Whether every weight is a line ([`Residency::Resident`]) or the load has
    /// experts the budget may not hold.
    pub residency: Residency,
    /// Whether the operator named `--vision-embedding-pool-mib` (GitHub
    /// #243). Only so a refusal can name the knob they just set: telling
    /// someone who asked for a large embedding pool to run "without
    /// --vision" is the one remedy they did not mean.
    pub embedding_pool_named: bool,
    /// The device bytes a KV pool of this many pages occupies: its planes
    /// and its block tables, as the leaf lays them out.
    pub kv_arena_bytes: &'a dyn Fn(u32) -> u64,
    /// Whether an oversubscribed allocation pages (Windows WDDM) or fails.
    pub can_page: bool,
}

/// The plan the load carries out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VramPlan {
    pub mode: VramMode,
    pub free_at_start_bytes: u64,
    pub budget_bytes: u64,
    pub lines: VramLines,
    /// The branch the pool was sized by (ADR 0045).
    pub kv_pool_policy: KvPoolPolicy,
    /// Pages the KV pool holds.
    pub kv_page_count: u32,
    /// The KV pool's device bytes (planes and block tables).
    pub kv_pool_bytes: u64,
    /// Residency's fixed lines ([`ExpertResidency::fixed_bytes`]); 0 on a
    /// [`Residency::Resident`] load.
    pub residency_bytes: u64,
    /// The VRAM expert cache: every expert projection on the resident
    /// branch, the rest of the budget on the offloaded one; 0 on a
    /// [`Residency::Resident`] load.
    pub expert_cache_bytes: u64,
    /// Every line, the KV pool, and residency's lines with the expert cache:
    /// what the process holds after load.
    pub total_bytes: u64,
    /// The plan holds more than was free at start.
    pub oversubscribed: bool,
    /// Why the start proceeds anyway, one message per reason.
    pub warnings: Vec<String>,
}

impl VramPlan {
    /// The payload budget that buys exactly [`VramPlan::kv_page_count`]
    /// pages under `format` — what the leaf's pool is built from.
    pub fn kv_budget_bytes(&self, format: KvFormat, geometry: KvGeometry) -> u64 {
        u64::from(self.kv_page_count) * format.page_bytes(geometry)
    }

    /// Tokens the KV pool holds.
    pub fn kv_token_capacity(&self) -> u64 {
        u64::from(self.kv_page_count) * u64::from(KV_PAGE_TOKENS)
    }
}

/// A start the plan refuses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VramPlanError {
    /// An explicit budget above the memory free at start, without
    /// `--allow-vram-oversubscription`.
    BudgetAboveFree { budget_bytes: u64, free_bytes: u64 },
    /// The budget cannot hold every reservation and one sequence at
    /// `--max-context` (resident), or every reservation and the KV pool
    /// before the expert cache (offloaded).
    BelowMinimum {
        mode: VramMode,
        policy: KvPoolPolicy,
        budget_bytes: u64,
        needed_bytes: u64,
        max_context_tokens: u32,
        kv_pool_named: bool,
        embedding_pool_named: bool,
    },
    /// A named KV pool smaller than one `--max-context` sequence plus a page
    /// per retained slot (ADR 0045's invariant), whatever the budget.
    PoolBelowFloor {
        pool: KvPoolSize,
        pool_pages: u32,
        floor_pages: u32,
        max_context_tokens: u32,
        retained_slots: u32,
        /// The model's own bytes per token, so a byte count can be redone.
        bytes_per_token: u64,
    },
}

impl std::fmt::Display for VramPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::BudgetAboveFree {
                budget_bytes,
                free_bytes,
            } => write!(
                f,
                "the VRAM budget of {budget_bytes} bytes (--vram-budget-bytes) is more than the \
                 {free_bytes} bytes free at start; lower it, or pass \
                 --allow-vram-oversubscription to start anyway"
            ),
            Self::PoolBelowFloor {
                pool,
                pool_pages,
                floor_pages,
                max_context_tokens,
                retained_slots,
                bytes_per_token,
            } => write!(
                f,
                "the KV pool of {pool} (--kv-pool-bytes) holds {pool_pages} pages of {KV_PAGE_TOKENS} tokens \
                 ({bytes_per_token} bytes per token on this model), fewer than the {floor_pages} it must hold: \
                 one --max-context sequence ({max_context_tokens} tokens) and a page per retained slot \
                 ({retained_slots}); name a larger --kv-pool-bytes, or a smaller --max-context, \
                 --retained-device or --retained-host"
            ),
            Self::BelowMinimum {
                mode,
                policy: KvPoolPolicy::Offloaded,
                budget_bytes,
                needed_bytes,
                max_context_tokens,
                ..
            } => {
                write!(
                    f,
                    "the VRAM plan needs {needed_bytes} bytes for the weights, workspaces, lanes, residency's \
                     staging ring and tables and the KV pool, {} bytes more than the {budget_bytes}-byte VRAM \
                     budget before any of it goes to the expert cache; shrink it with a smaller --kv-pool-bytes, \
                     a smaller --max-context (now {max_context_tokens}) or --prefill-chunk, or with ",
                    needed_bytes.saturating_sub(budget_bytes),
                )?;
                match mode {
                    VramMode::Derived { .. } => f.write_str("a smaller --vram-headroom-bytes"),
                    VramMode::Explicit { .. } => {
                        f.write_str("a larger --vram-budget-bytes (or --allow-vram-oversubscription)")
                    }
                }
            }
            Self::BelowMinimum {
                mode,
                policy: KvPoolPolicy::Resident,
                budget_bytes,
                needed_bytes,
                max_context_tokens,
                kv_pool_named,
                embedding_pool_named,
            } => {
                write!(
                    f,
                    "the VRAM plan needs {needed_bytes} bytes for the weights, workspaces, lanes, \
                     retained state and {} KV, {} bytes more than the {budget_bytes}-byte VRAM \
                     budget; shrink it with a smaller --max-context (now {max_context_tokens}), \
                     without --vision, with fewer --retained-device, or with ",
                    if kv_pool_named {
                        "the --kv-pool-bytes"
                    } else {
                        "one --max-context sequence of"
                    },
                    needed_bytes.saturating_sub(budget_bytes),
                )?;
                match mode {
                    VramMode::Derived { .. } => {
                        f.write_str("a smaller --vram-headroom-bytes")?
                    }
                    VramMode::Explicit { .. } => f.write_str(
                        "a larger --vram-budget-bytes (or --allow-vram-oversubscription)",
                    )?,
                }
                if kv_pool_named {
                    f.write_str(", or name a smaller --kv-pool-bytes")?;
                }
                if embedding_pool_named {
                    f.write_str(", or name a smaller --vision-embedding-pool-mib")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for VramPlanError {}

/// Lay the plan out, or refuse the start.
pub fn plan_vram(request: &VramRequest<'_>) -> Result<VramPlan, VramPlanError> {
    let free = request.free_at_start_bytes;
    let mut warnings = Vec::new();
    let budget = match request.mode {
        VramMode::Derived { headroom_bytes } => free.saturating_sub(headroom_bytes),
        VramMode::Explicit {
            budget_bytes,
            allow_oversubscription,
        } => {
            if budget_bytes > free {
                if !allow_oversubscription {
                    return Err(VramPlanError::BudgetAboveFree {
                        budget_bytes,
                        free_bytes: free,
                    });
                }
                warnings.push(format!(
                    "the VRAM budget of {budget_bytes} bytes is more than the {free} bytes free at \
                     start (--allow-vram-oversubscription): {}",
                    oversubscription_consequence(request.can_page)
                ));
            }
            budget_bytes
        }
    };

    let page_bytes = request.kv_page_bytes;
    // ADR 0030's floor, on both models since ADR 0045: one `--max-context`
    // sequence and a page per retained slot.
    let min_pages = request
        .max_context_tokens
        .div_ceil(KV_PAGE_TOKENS)
        .saturating_add(request.retained_slots);
    let fixed = request.lines.total();
    let arena = request.kv_arena_bytes;

    let named_pages = request.kv_pool.map(|pool| pool.pages(page_bytes));
    if let (Some(pool), Some(pool_pages)) = (request.kv_pool, named_pages) {
        if pool_pages < min_pages {
            return Err(VramPlanError::PoolBelowFloor {
                pool,
                pool_pages,
                floor_pages: min_pages,
                max_context_tokens: request.max_context_tokens,
                retained_slots: request.retained_slots,
                bytes_per_token: page_bytes / u64::from(KV_PAGE_TOKENS),
            });
        }
    }

    // The policy (ADR 0045): resident when the budget holds every line,
    // residency's own, every expert projection and the pool a named size or
    // the floor asks for -- the 27B by construction -- else offloaded.
    // Offloaded, the pool's default comes with the branch.
    let (policy, residency_bytes, experts_bytes, offloaded_pages) = match request.residency {
        Residency::Resident => (KvPoolPolicy::Resident, 0, 0, None),
        Residency::Experts(experts) => {
            let pool = arena(named_pages.unwrap_or(min_pages));
            let resident = fixed
                .saturating_add(experts.fixed_bytes)
                .saturating_add(experts.expert_pool_bytes)
                .saturating_add(pool)
                <= budget;
            if resident {
                (KvPoolPolicy::Resident, experts.fixed_bytes, experts.expert_pool_bytes, None)
            } else {
                let pages = offloaded_default_pages(request.max_context_tokens, experts.decode_lanes, min_pages);
                (KvPoolPolicy::Offloaded, experts.fixed_bytes, 0, Some(pages))
            }
        }
    };
    // Everything placed beside the pool before it: on a whole load, the lines.
    let held = fixed.saturating_add(residency_bytes).saturating_add(experts_bytes);

    // A named size, else the offloaded default, else the rest.
    let wanted_pages = named_pages
        .or(offloaded_pages)
        .unwrap_or_else(|| pages_fitting(budget.saturating_sub(held), arena, page_bytes));
    let fits = wanted_pages >= min_pages
        && held.saturating_add(arena(wanted_pages)) <= budget;
    let kv_page_count = if fits {
        wanted_pages
    } else {
        let pages = wanted_pages.max(min_pages);
        let needed_bytes = held.saturating_add(arena(pages));
        let refusal = VramPlanError::BelowMinimum {
            mode: request.mode,
            policy,
            budget_bytes: budget,
            needed_bytes,
            max_context_tokens: request.max_context_tokens,
            kv_pool_named: request.kv_pool.is_some(),
            embedding_pool_named: request.embedding_pool_named,
        };
        if !request.mode.allows_oversubscription() {
            return Err(refusal);
        }
        warnings.push(format!(
            "{refusal}; starting anyway (--allow-vram-oversubscription): {}",
            oversubscription_consequence(request.can_page)
        ));
        pages
    };

    let kv_pool_bytes = arena(kv_page_count);
    // Resident, the cache holds every projection (a named pool leaves the
    // rest of the budget unused, as on the 27B); offloaded, it takes the rest.
    let expert_cache_bytes = match (request.residency, policy) {
        (Residency::Resident, _) => 0,
        (Residency::Experts(_), KvPoolPolicy::Resident) => experts_bytes,
        (Residency::Experts(_), KvPoolPolicy::Offloaded) => {
            budget.saturating_sub(held.saturating_add(kv_pool_bytes))
        }
    };
    let total_bytes = fixed
        .saturating_add(kv_pool_bytes)
        .saturating_add(residency_bytes)
        .saturating_add(expert_cache_bytes);
    Ok(VramPlan {
        mode: request.mode,
        free_at_start_bytes: free,
        budget_bytes: budget,
        lines: request.lines,
        kv_pool_policy: policy,
        kv_page_count,
        kv_pool_bytes,
        residency_bytes,
        expert_cache_bytes,
        total_bytes,
        oversubscribed: total_bytes > free,
        warnings,
    })
}

/// The offloaded pool's default pages (ADR 0045): 524,288 tokens, capped at
/// every lane's whole context -- a larger pool could hold only retained
/// pages, and one lane keeps the pool it had before the policy -- and never
/// below the floor, so a `--max-context` past 524,288 starts.
fn offloaded_default_pages(max_context_tokens: u32, decode_lanes: u32, floor_pages: u32) -> u32 {
    let lanes_pages = decode_lanes.saturating_mul(max_context_tokens.div_ceil(KV_PAGE_TOKENS));
    (OFFLOADED_KV_POOL_TOKENS / KV_PAGE_TOKENS).min(lanes_pages).max(floor_pages)
}

fn oversubscription_consequence(can_page: bool) -> &'static str {
    if can_page {
        "this system pages device memory to system RAM, so every step past free memory is slow \
         and erratic"
    } else {
        "this system cannot page device memory, so an allocation past free memory will fail"
    }
}

/// The most pages whose arena fits in `bytes`.
fn pages_fitting(bytes: u64, arena: &dyn Fn(u32) -> u64, page_bytes: u64) -> u32 {
    if page_bytes == 0 || arena(1) > bytes {
        return 0;
    }
    let overhead = arena(1).saturating_sub(page_bytes);
    let mut pages = ((bytes - overhead) / page_bytes).min(u64::from(u32::MAX)) as u32;
    while pages > 1 && arena(pages) > bytes {
        pages -= 1;
    }
    pages
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;
    const QWEN: KvGeometry = KvGeometry {
        gqa_layers: 16,
        num_kv_heads: 4,
        head_dim: 256,
    };
    const FORMAT: KvFormat = KvFormat::HqE8_2b;
    /// A block table that is not a whole number of pages, so a plan that
    /// forgot it would overrun the budget.
    const TABLE_BYTES: u64 = 3 * MIB + 17;

    fn page_bytes() -> u64 {
        FORMAT.page_bytes(QWEN)
    }

    fn arena(pages: u32) -> u64 {
        TABLE_BYTES + u64::from(pages) * page_bytes()
    }

    fn lines() -> VramLines {
        VramLines {
            weights: 17 * GIB,
            cuda_context: 300 * MIB,
            // max(1300 MiB of prefill scratch, 2 GiB of vision encoder).
            workspace: 2 * GIB,
            media_embedding: 320 * MIB,
            sampling: 10 * MIB,
            decode_graph: 200 * MIB,
            verify_round: 100 * MIB,
            drafter_round: 150 * MIB,
            lane_state: 1800 * MIB,
            retained_slots: 1800 * MIB,
            hq_residual_window: 272 * MIB,
            residual: 280 * MIB,
        }
    }

    fn request(mode: VramMode, free: u64) -> VramRequest<'static> {
        VramRequest {
            mode,
            free_at_start_bytes: free,
            lines: lines(),
            kv_page_bytes: page_bytes(),
            max_context_tokens: 262_144,
            retained_slots: SLOTS,
            kv_pool: None,
            residency: Residency::Resident,
            embedding_pool_named: false,
            kv_arena_bytes: &arena,
            can_page: true,
        }
    }

    const DERIVED: VramMode = VramMode::Derived {
        headroom_bytes: GIB,
    };

    /// Retained slots, each one tail page on top of the full context.
    const SLOTS: u32 = 8;

    #[test]
    fn a_derived_budget_is_free_memory_less_the_headroom_and_kv_takes_the_rest() {
        let free = 30 * GIB;
        let plan = plan_vram(&request(DERIVED, free)).expect("fits");
        assert_eq!(plan.mode, DERIVED);
        // ADR 0045 (AC 1): the 27B is resident -- every weight is a line --
        // and nothing beyond the lines and the pool is placed.
        assert_eq!(plan.kv_pool_policy, KvPoolPolicy::Resident);
        assert_eq!((plan.residency_bytes, plan.expert_cache_bytes), (0, 0));
        assert_eq!(plan.budget_bytes, 29 * GIB);
        assert_eq!(plan.kv_pool_bytes, arena(plan.kv_page_count));
        assert_eq!(plan.total_bytes, lines().total() + plan.kv_pool_bytes);
        // The rest, to the page: one more would not fit.
        assert!(plan.total_bytes <= plan.budget_bytes);
        assert!(lines().total() + arena(plan.kv_page_count + 1) > plan.budget_bytes);
        assert!(!plan.oversubscribed);
        assert!(plan.warnings.is_empty());
        assert_eq!(plan.kv_token_capacity(), u64::from(plan.kv_page_count) * 64);
        assert_eq!(
            plan.kv_budget_bytes(FORMAT, QWEN),
            u64::from(plan.kv_page_count) * page_bytes()
        );
        assert_eq!(
            crate::kv_format::plan_kv_pool(FORMAT, QWEN, plan.kv_budget_bytes(FORMAT, QWEN)).page_count,
            plan.kv_page_count,
            "the leaf's pool, built from that budget, holds the planned pages"
        );
    }

    #[test]
    fn an_explicit_budget_is_the_figure_itself() {
        let mode = VramMode::Explicit {
            budget_bytes: 28 * GIB,
            allow_oversubscription: false,
        };
        let plan = plan_vram(&request(mode, 31 * GIB)).expect("fits");
        assert_eq!(plan.budget_bytes, 28 * GIB);
        assert!(plan.total_bytes <= 28 * GIB);
        assert!(lines().total() + arena(plan.kv_page_count + 1) > 28 * GIB);
        assert!(!plan.oversubscribed);
    }

    #[test]
    fn an_explicit_budget_above_free_memory_refuses_naming_both_and_the_flag() {
        let mode = VramMode::Explicit {
            budget_bytes: 30 * GIB,
            allow_oversubscription: false,
        };
        let err = plan_vram(&request(mode, 29 * GIB)).expect_err("more than is free");
        assert_eq!(
            err,
            VramPlanError::BudgetAboveFree {
                budget_bytes: 30 * GIB,
                free_bytes: 29 * GIB
            }
        );
        let message = err.to_string();
        assert!(message.contains(&(30 * GIB).to_string()), "{message}");
        assert!(message.contains(&(29 * GIB).to_string()), "{message}");
        assert!(message.contains("--allow-vram-oversubscription"), "{message}");
    }

    #[test]
    fn oversubscription_turns_the_refusal_into_a_warning_that_says_what_happens() {
        let mode = VramMode::Explicit {
            budget_bytes: 30 * GIB,
            allow_oversubscription: true,
        };
        let plan = plan_vram(&request(mode, 29 * GIB)).expect("allowed");
        assert_eq!(plan.budget_bytes, 30 * GIB);
        assert!(plan.oversubscribed);
        assert_eq!(plan.warnings.len(), 1);
        assert!(plan.warnings[0].contains("pages device memory"), "{:?}", plan.warnings);

        let failing = VramRequest {
            can_page: false,
            ..request(mode, 29 * GIB)
        };
        let plan = plan_vram(&failing).expect("allowed");
        assert!(plan.warnings[0].contains("will fail"), "{:?}", plan.warnings);
    }

    #[test]
    fn a_plan_that_cannot_hold_one_full_context_refuses_naming_the_shortfall_and_the_knobs() {
        // Enough for every fixed line, a few pages short of a 262K sequence
        // and a tail page per retained slot.
        let min_pages = 262_144u32.div_ceil(64) + SLOTS;
        let free = GIB + lines().total() + arena(min_pages) - 5 * page_bytes();
        let err = plan_vram(&request(DERIVED, free)).expect_err("below the minimum");
        let VramPlanError::BelowMinimum {
            budget_bytes,
            needed_bytes,
            ..
        } = err
        else {
            panic!("{err:?}");
        };
        assert_eq!(budget_bytes, free - GIB);
        assert_eq!(needed_bytes, lines().total() + arena(min_pages));
        let message = err.to_string();
        assert!(message.contains(&(5 * page_bytes()).to_string()), "{message}");
        for knob in ["--max-context", "--vision", "--retained-device", "--vram-headroom-bytes"] {
            assert!(message.contains(knob), "{knob} missing: {message}");
        }
    }

    #[test]
    fn the_minimum_holds_a_tail_page_for_every_retained_slot() {
        // Exactly one full context: enough before GitHub #215, one page per
        // retained slot short of it now.
        let full_context = 262_144u32.div_ceil(64);
        let free = GIB + lines().total() + arena(full_context);
        assert!(
            plan_vram(&VramRequest {
                retained_slots: 0,
                ..request(DERIVED, free)
            })
            .is_ok(),
            "no slot, no tail page"
        );
        let err = plan_vram(&request(DERIVED, free)).expect_err("a page short per slot");
        let VramPlanError::BelowMinimum { needed_bytes, .. } = err else {
            panic!("{err:?}");
        };
        assert_eq!(needed_bytes, lines().total() + arena(full_context + SLOTS));
    }

    #[test]
    fn headroom_above_free_memory_refuses_rather_than_underflowing() {
        let err = plan_vram(&request(DERIVED, GIB / 2)).expect_err("nothing left");
        assert!(matches!(err, VramPlanError::BelowMinimum { budget_bytes: 0, .. }));
    }

    #[test]
    fn below_the_minimum_with_oversubscription_warns_and_plans_one_full_context() {
        let min_pages = 262_144u32.div_ceil(64) + SLOTS;
        let budget = lines().total() + arena(min_pages) - page_bytes();
        let mode = VramMode::Explicit {
            budget_bytes: budget,
            allow_oversubscription: true,
        };
        let plan = plan_vram(&request(mode, budget)).expect("allowed");
        assert_eq!(plan.kv_page_count, min_pages);
        assert!(plan.oversubscribed);
        assert_eq!(plan.warnings.len(), 1);
        assert!(plan.warnings[0].contains("--max-context"), "{:?}", plan.warnings);
        assert!(plan.warnings[0].contains("--vram-budget-bytes"), "{:?}", plan.warnings);
    }

    #[test]
    fn a_named_kv_pool_is_planned_as_named_and_leaves_the_rest_free() {
        let named = 4 * GIB;
        let plan = plan_vram(&VramRequest {
            kv_pool: Some(KvPoolSize::Bytes(named)),
            ..request(DERIVED, 31 * GIB)
        })
        .expect("fits");
        // A byte count buys what the leaf's own pool arithmetic buys.
        let pages = crate::kv_format::plan_kv_pool(FORMAT, QWEN, named).page_count;
        assert_eq!(plan.kv_page_count, pages);
        assert_eq!(plan.kv_pool_bytes, arena(pages));
        assert!(plan.total_bytes < plan.budget_bytes);
    }

    #[test]
    fn a_named_kv_pool_past_the_budget_refuses_and_names_the_flag() {
        let err = plan_vram(&VramRequest {
            kv_pool: Some(KvPoolSize::Bytes(10 * GIB)),
            ..request(DERIVED, 30 * GIB)
        })
        .expect_err("past the budget");
        let message = err.to_string();
        assert!(message.contains("--kv-pool-bytes"), "{message}");
    }

    #[test]
    fn a_named_embedding_pool_past_the_budget_refuses_and_names_that_flag_too() {
        // GitHub #243: the remedies an operator is offered have to include
        // the one they just reached for. Telling someone who asked for a
        // large embedding pool to run "without --vision" is the single
        // remedy they did not mean.
        let huge = || {
            let mut l = lines();
            l.media_embedding = 20 * GIB;
            l
        };
        let message = plan_vram(&VramRequest {
            lines: huge(),
            embedding_pool_named: true,
            ..request(DERIVED, 30 * GIB)
        })
        .expect_err("past the budget")
        .to_string();
        assert!(message.contains("--vision-embedding-pool-mib"), "{message}");

        // And it stays quiet when the operator did not name one: the pool is
        // then whatever the envelope implies, and shrinking it is not a knob
        // they turned.
        let quiet = plan_vram(&VramRequest { lines: huge(), ..request(DERIVED, 30 * GIB) })
            .expect_err("past the budget")
            .to_string();
        assert!(!quiet.contains("--vision-embedding-pool-mib"), "{quiet}");
    }

    // ── the KV pool policy (ADR 0045, GitHub #309) ───────────────────────

    /// Flash-Next's paged bytes per token under hq-e8-2b (KV 3,456, indexer
    /// keys 768), and a page of them.
    const FN_TOKEN_BYTES: u64 = 4_224;
    const FN_PAGE_BYTES: u64 = FN_TOKEN_BYTES * 64;
    const FN_CONTEXT: u32 = 262_144;
    /// Flash-Next's default host retained slots.
    const FN_SLOTS: u32 = 8;
    const FN_FLOOR: u32 = FN_CONTEXT / 64 + FN_SLOTS;
    /// The 5090 finding's `ignis.runtime.flash_next_plan` (2026-10-06): the
    /// budget, residency's fixed lines, and every expert projection.
    const FN_BUDGET: u64 = 31_529_590_784;
    const FN_RESIDENCY: u64 = 1_577_852_928;
    const FN_EXPERTS: u64 = 37_795_446_784;

    /// The pool's arena: its pages, beside block tables and the indexer's
    /// per-slot tails, which are not.
    fn fn_arena(pages: u32) -> u64 {
        5 * MIB + 3 + u64::from(pages) * FN_PAGE_BYTES
    }

    /// The finding's weights and program, and the lanes' state. Held fixed
    /// across lane counts: only the pool is at stake here.
    fn fn_lines() -> VramLines {
        VramLines {
            weights: 5_022_463_744,
            cuda_context: 300 * MIB,
            workspace: 1_810_558_704,
            lane_state: 400 * MIB,
            hq_residual_window: 102 * MIB,
            ..VramLines::default()
        }
    }

    fn flash_next(lanes: u32) -> VramRequest<'static> {
        VramRequest {
            mode: VramMode::Explicit {
                budget_bytes: FN_BUDGET,
                allow_oversubscription: false,
            },
            free_at_start_bytes: FN_BUDGET + GIB,
            lines: fn_lines(),
            kv_page_bytes: FN_PAGE_BYTES,
            max_context_tokens: FN_CONTEXT,
            retained_slots: FN_SLOTS,
            kv_pool: None,
            residency: Residency::Experts(ExpertResidency {
                expert_pool_bytes: FN_EXPERTS,
                fixed_bytes: FN_RESIDENCY,
                decode_lanes: lanes,
            }),
            embedding_pool_named: false,
            kv_arena_bytes: &fn_arena,
            can_page: true,
        }
    }

    /// [`flash_next`] on an explicit budget of `budget`, all of it free.
    fn flash_next_at(budget: u64, lanes: u32) -> VramRequest<'static> {
        VramRequest {
            mode: VramMode::Explicit {
                budget_bytes: budget,
                allow_oversubscription: false,
            },
            free_at_start_bytes: budget,
            ..flash_next(lanes)
        }
    }

    /// The cache before ADR 0045: every lane's whole context and a page per
    /// retained slot in the pool, the cache taking the rest.
    fn whole_context_cache(lanes: u32) -> u64 {
        FN_BUDGET - fn_lines().total() - fn_arena(lanes * (FN_CONTEXT / 64) + FN_SLOTS) - FN_RESIDENCY
    }

    #[test]
    fn flash_next_on_the_5090_is_offloaded_and_reserves_524288_tokens_first() {
        for lanes in 1..=8 {
            let plan = plan_vram(&flash_next(lanes)).expect("fits");
            assert_eq!(plan.kv_pool_policy, KvPoolPolicy::Offloaded, "{lanes} lanes");
            // AC 2: one lane keeps its pool; two lanes and more share 524,288
            // tokens.
            let pages = if lanes == 1 { 4_104 } else { 8_192 };
            assert_eq!(plan.kv_page_count, pages, "{lanes} lanes");
            assert_eq!(plan.kv_pool_bytes, fn_arena(pages));
            assert_eq!(plan.residency_bytes, FN_RESIDENCY);
            assert_eq!(
                plan.expert_cache_bytes,
                FN_BUDGET - fn_lines().total() - fn_arena(pages) - FN_RESIDENCY,
                "the budget less every line, the pool and residency's fixed bytes"
            );
            assert_eq!(plan.total_bytes, FN_BUDGET, "the cache takes the whole rest");
        }
        assert_eq!(plan_vram(&flash_next(1)).unwrap().expert_cache_bytes, whole_context_cache(1), "byte for byte");
        let two = plan_vram(&flash_next(2)).unwrap();
        assert_eq!(two.expert_cache_bytes - whole_context_cache(2), 8 * FN_PAGE_BYTES, "within 8 pages");
        // Three lanes: the cache gains what two more whole contexts and the
        // retained pages held, 4,104 pages -- ~1.03 GiB.
        let gain = plan_vram(&flash_next(3)).unwrap().expert_cache_bytes - whole_context_cache(3);
        assert_eq!(gain, 4_104 * FN_PAGE_BYTES);
        assert!((gain as f64 / GIB as f64 - 1.033).abs() < 0.001, "{gain}");
    }

    #[test]
    fn a_budget_that_holds_every_expert_is_resident_and_the_pool_takes_the_rest() {
        // AC 3: on the threshold -- every line, residency's, every expert and
        // the floor pool -- the load is resident; one byte less, offloaded.
        let threshold = fn_lines().total() + FN_RESIDENCY + FN_EXPERTS + fn_arena(FN_FLOOR);
        let plan = plan_vram(&flash_next_at(threshold, 3)).expect("fits");
        assert_eq!(plan.kv_pool_policy, KvPoolPolicy::Resident);
        assert_eq!(plan.expert_cache_bytes, FN_EXPERTS, "the whole expert pool");
        assert_eq!(plan.kv_page_count, FN_FLOOR, "the rest, which is the floor");
        let below = plan_vram(&flash_next_at(threshold - 1, 3)).expect("fits");
        assert_eq!(below.kv_pool_policy, KvPoolPolicy::Offloaded);
        assert_eq!(below.kv_page_count, 8_192);

        // A 96 GB card: the pool takes the rest, to the page.
        let card = 96_000_000_000;
        let plan = plan_vram(&flash_next_at(card, 3)).expect("fits");
        assert_eq!(plan.kv_pool_policy, KvPoolPolicy::Resident);
        assert_eq!(plan.expert_cache_bytes, FN_EXPERTS);
        let held = fn_lines().total() + FN_RESIDENCY + FN_EXPERTS;
        assert_eq!(plan.total_bytes, held + plan.kv_pool_bytes);
        assert!(plan.total_bytes <= card);
        assert!(held + fn_arena(plan.kv_page_count + 1) > card);
    }

    #[test]
    fn a_named_pool_below_one_context_refuses_naming_the_knobs_and_the_bytes_per_token() {
        // AC 4, on either model: a whole context, but no page for the
        // retained slots.
        for residency in [flash_next(3).residency, Residency::Resident] {
            let request = VramRequest {
                kv_pool: Some(KvPoolSize::Tokens(262_144)),
                residency,
                ..flash_next(3)
            };
            let err = plan_vram(&request).expect_err("below the floor");
            assert_eq!(
                err,
                VramPlanError::PoolBelowFloor {
                    pool: KvPoolSize::Tokens(262_144),
                    pool_pages: 4_096,
                    floor_pages: FN_FLOOR,
                    max_context_tokens: FN_CONTEXT,
                    retained_slots: FN_SLOTS,
                    bytes_per_token: FN_TOKEN_BYTES,
                }
            );
            let message = err.to_string();
            for needle in ["--kv-pool-bytes", "--max-context", "--retained-device", "--retained-host", "4224 bytes per token", "4104"] {
                assert!(message.contains(needle), "{needle} missing: {message}");
            }
        }
    }

    #[test]
    fn a_context_past_the_default_takes_the_floor_rather_than_refusing() {
        let plan = plan_vram(&VramRequest {
            max_context_tokens: 524_288,
            ..flash_next(3)
        })
        .expect("fits");
        assert_eq!(plan.kv_pool_policy, KvPoolPolicy::Offloaded);
        assert_eq!(plan.kv_page_count, 8_200);
    }

    #[test]
    fn a_named_pool_replaces_the_policy_s_size_in_bytes_or_tokens() {
        // AC 5: 512Ktok is 8,192 pages on either model; a token count rounds
        // up to whole pages, a byte count buys whole pages.
        assert_eq!(KvPoolSize::Tokens(512 * 1024).pages(FN_PAGE_BYTES), 8_192);
        assert_eq!(KvPoolSize::Tokens(512 * 1024).pages(page_bytes()), 8_192);
        assert_eq!(KvPoolSize::Tokens(100).pages(FN_PAGE_BYTES), 2);
        assert_eq!(KvPoolSize::Bytes(3 * FN_PAGE_BYTES - 1).pages(FN_PAGE_BYTES), 2);

        // Offloaded, a named pool replaces the default, past the lanes' cap
        // too; and a byte count is read at Flash-Next's own per-token cost.
        let named = |pool, lanes| {
            plan_vram(&VramRequest {
                kv_pool: Some(pool),
                ..flash_next(lanes)
            })
            .expect("fits")
        };
        let plan = named(KvPoolSize::Tokens(512 * 1024), 1);
        assert_eq!((plan.kv_pool_policy, plan.kv_page_count), (KvPoolPolicy::Offloaded, 8_192));
        let plan = named(KvPoolSize::Bytes(3 * GIB), 3);
        assert_eq!(plan.kv_page_count, (3 * GIB / FN_PAGE_BYTES) as u32);
        assert_eq!(plan.expert_cache_bytes, FN_BUDGET - fn_lines().total() - plan.kv_pool_bytes - FN_RESIDENCY);

        // The 27B, the same token count: its pool, the rest left unused.
        let plan = plan_vram(&VramRequest {
            kv_pool: Some(KvPoolSize::Tokens(512 * 1024)),
            ..request(DERIVED, 31 * GIB)
        })
        .expect("fits");
        assert_eq!((plan.kv_pool_policy, plan.kv_page_count), (KvPoolPolicy::Resident, 8_192));
        assert!(plan.total_bytes < plan.budget_bytes);
    }

    #[test]
    fn a_named_pool_on_a_resident_load_replaces_the_minimum_and_leaves_the_rest_unused() {
        let named = |budget| VramRequest {
            kv_pool: Some(KvPoolSize::Tokens(1 << 20)),
            ..flash_next_at(budget, 3)
        };
        let card = 96_000_000_000;
        let plan = plan_vram(&named(card)).expect("fits");
        assert_eq!(plan.kv_pool_policy, KvPoolPolicy::Resident);
        assert_eq!(plan.kv_page_count, 16_384);
        assert_eq!(plan.expert_cache_bytes, FN_EXPERTS);
        assert!(plan.total_bytes < card, "the rest of the budget stays unused");
        // The residency test charges the named pool, not the floor.
        let tight = fn_lines().total() + FN_RESIDENCY + FN_EXPERTS + fn_arena(16_384);
        assert_eq!(plan_vram(&named(tight)).unwrap().kv_pool_policy, KvPoolPolicy::Resident);
        assert_eq!(plan_vram(&named(tight - 1)).unwrap().kv_pool_policy, KvPoolPolicy::Offloaded);
    }

    #[test]
    fn an_explicit_flash_next_budget_above_free_memory_refuses_as_on_the_27b() {
        // AC 7: Flash-Next's plan goes through the same refusal.
        let above = VramRequest {
            free_at_start_bytes: FN_BUDGET - 1,
            ..flash_next(3)
        };
        assert!(matches!(plan_vram(&above), Err(VramPlanError::BudgetAboveFree { .. })));
        let allowed = VramRequest {
            mode: VramMode::Explicit {
                budget_bytes: FN_BUDGET,
                allow_oversubscription: true,
            },
            ..above
        };
        let plan = plan_vram(&allowed).expect("allowed");
        assert!(plan.oversubscribed);
        assert_eq!(plan.warnings.len(), 1, "{:?}", plan.warnings);
    }

    #[test]
    fn an_offloaded_pool_the_budget_cannot_hold_refuses_naming_its_own_knobs() {
        let budget = fn_lines().total() + FN_RESIDENCY + fn_arena(8_192) - 1;
        let err = plan_vram(&flash_next_at(budget, 3)).expect_err("short of the pool");
        assert!(
            matches!(err, VramPlanError::BelowMinimum { policy: KvPoolPolicy::Offloaded, .. }),
            "{err:?}"
        );
        let message = err.to_string();
        for needle in ["--kv-pool-bytes", "--max-context", "--prefill-chunk", "--vram-budget-bytes", "expert cache", "1 bytes more"] {
            assert!(message.contains(needle), "{needle} missing: {message}");
        }
        assert!(!message.contains("--vision"), "{message}");
    }

    #[test]
    fn the_lines_keep_plan_order_and_add_up() {
        let names: Vec<_> = lines().entries().iter().map(|(name, _)| *name).collect();
        assert_eq!(
            names,
            [
                "weights",
                "cuda_context",
                "workspace",
                "media_embedding",
                "sampling",
                "decode_graph",
                "verify_round",
                "drafter_round",
                "lane_state",
                "retained_slots",
                "hq_residual_window",
                "residual",
            ]
        );
        assert_eq!(
            lines().total(),
            lines().entries().iter().map(|(_, b)| b).sum::<u64>()
        );
    }
}
