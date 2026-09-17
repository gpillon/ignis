//! The load's device-memory plan within a **VRAM budget** (GitHub #210,
//! ADR 0030, `CONTEXT.md`).
//!
//! Pure arithmetic over numbers the loader has already measured or asked the
//! leaf for: what was free at start, what each fixed reservation costs, and
//! what one KV pool of `n` pages occupies. It decides the budget, gives the
//! KV pool the rest, and refuses a start that cannot hold one sequence at
//! the maximum context — before a byte of the plan is allocated. Nothing
//! here touches a device, so every refusal and warning is pinned on the CPU.

use crate::kv_format::{KV_PAGE_TOKENS, KvFormat, KvGeometry, plan_kv_pool};

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
    /// The prefill scratch arena, sized for one `--prefill-chunk` span.
    pub prefill_scratch: u64,
    /// The vision encoder workspace (0 without `--vision`).
    pub vision_workspace: u64,
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
    /// conv state, penalty counts, the drafter's window and checkpoint.
    pub lane_state: u64,
    /// The sequence pool's retained slots (GitHub #211): a lane's state each,
    /// reserved at load beside the lanes.
    pub retained_slots: u64,
    /// The retained checkpoint budget (0 with `--prompt-reuse off`). A ledger
    /// the scheduler charges checkpoint images to, reserved here so the KV
    /// pool never takes the memory those images are allocated from.
    pub retained: u64,
    /// What a load holds beyond every line above: allocator rounding, the
    /// decode graph captures, the kernel's lazily created handles. Measured,
    /// not derived.
    pub residual: u64,
}

impl VramLines {
    /// `(name, bytes)` in plan order; the names are the `*_bytes` fields of
    /// `ignis.runtime.vram_plan` without the suffix.
    pub fn entries(&self) -> [(&'static str, u64); 13] {
        [
            ("weights", self.weights),
            ("cuda_context", self.cuda_context),
            ("prefill_scratch", self.prefill_scratch),
            ("vision_workspace", self.vision_workspace),
            ("media_embedding", self.media_embedding),
            ("sampling", self.sampling),
            ("decode_graph", self.decode_graph),
            ("verify_round", self.verify_round),
            ("drafter_round", self.drafter_round),
            ("lane_state", self.lane_state),
            ("retained_slots", self.retained_slots),
            ("retained", self.retained),
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
    pub kv_format: KvFormat,
    pub kv_geometry: KvGeometry,
    /// `--max-context`: the plan must hold one sequence this long.
    pub max_context_tokens: u32,
    /// `--kv-pool-bytes`, when the operator named it: the pool's payload
    /// budget, as [`plan_kv_pool`] reads it. `None` gives the pool the rest.
    pub kv_pool_bytes: Option<u64>,
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
    /// Pages the KV pool holds.
    pub kv_page_count: u32,
    /// The KV pool's device bytes (planes and block tables).
    pub kv_pool_bytes: u64,
    /// Every line plus the KV pool: what the process holds after load.
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
    /// `--max-context`.
    BelowMinimum {
        mode: VramMode,
        budget_bytes: u64,
        needed_bytes: u64,
        max_context_tokens: u32,
        kv_pool_named: bool,
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
            Self::BelowMinimum {
                mode,
                budget_bytes,
                needed_bytes,
                max_context_tokens,
                kv_pool_named,
            } => {
                write!(
                    f,
                    "the VRAM plan needs {needed_bytes} bytes for the weights, workspaces, lanes, \
                     retained state and {} KV, {} bytes more than the {budget_bytes}-byte VRAM \
                     budget; shrink it with a smaller --max-context (now {max_context_tokens}), \
                     without --vision, or with ",
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

    let page_bytes = request.kv_format.page_bytes(request.kv_geometry);
    let min_pages = request.max_context_tokens.div_ceil(KV_PAGE_TOKENS);
    let fixed = request.lines.total();
    let arena = request.kv_arena_bytes;

    let wanted_pages = match request.kv_pool_bytes {
        Some(bytes) => plan_kv_pool(request.kv_format, request.kv_geometry, bytes).page_count,
        None => pages_fitting(budget.saturating_sub(fixed), arena, page_bytes),
    };
    let fits = wanted_pages >= min_pages
        && fixed.saturating_add(arena(wanted_pages)) <= budget;
    let kv_page_count = if fits {
        wanted_pages
    } else {
        let pages = wanted_pages.max(min_pages);
        let needed_bytes = fixed.saturating_add(arena(pages));
        let refusal = VramPlanError::BelowMinimum {
            mode: request.mode,
            budget_bytes: budget,
            needed_bytes,
            max_context_tokens: request.max_context_tokens,
            kv_pool_named: request.kv_pool_bytes.is_some(),
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
    let total_bytes = fixed.saturating_add(kv_pool_bytes);
    Ok(VramPlan {
        mode: request.mode,
        free_at_start_bytes: free,
        budget_bytes: budget,
        lines: request.lines,
        kv_page_count,
        kv_pool_bytes,
        total_bytes,
        oversubscribed: total_bytes > free,
        warnings,
    })
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
            prefill_scratch: 1300 * MIB,
            vision_workspace: 2 * GIB,
            media_embedding: 320 * MIB,
            sampling: 10 * MIB,
            decode_graph: 200 * MIB,
            verify_round: 100 * MIB,
            drafter_round: 150 * MIB,
            lane_state: 1800 * MIB,
            retained_slots: 450 * MIB,
            retained: 1800 * MIB,
            residual: 280 * MIB,
        }
    }

    fn request(mode: VramMode, free: u64) -> VramRequest<'static> {
        VramRequest {
            mode,
            free_at_start_bytes: free,
            lines: lines(),
            kv_format: FORMAT,
            kv_geometry: QWEN,
            max_context_tokens: 262_144,
            kv_pool_bytes: None,
            kv_arena_bytes: &arena,
            can_page: true,
        }
    }

    const DERIVED: VramMode = VramMode::Derived {
        headroom_bytes: GIB,
    };

    #[test]
    fn a_derived_budget_is_free_memory_less_the_headroom_and_kv_takes_the_rest() {
        let free = 30 * GIB;
        let plan = plan_vram(&request(DERIVED, free)).expect("fits");
        assert_eq!(plan.mode, DERIVED);
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
            plan_kv_pool(FORMAT, QWEN, plan.kv_budget_bytes(FORMAT, QWEN)).page_count,
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
        // Enough for every fixed line, a few pages short of a 262K sequence.
        let min_pages = 262_144u32.div_ceil(64);
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
        for knob in ["--max-context", "--vision", "--vram-headroom-bytes"] {
            assert!(message.contains(knob), "{knob} missing: {message}");
        }
    }

    #[test]
    fn headroom_above_free_memory_refuses_rather_than_underflowing() {
        let err = plan_vram(&request(DERIVED, GIB / 2)).expect_err("nothing left");
        assert!(matches!(err, VramPlanError::BelowMinimum { budget_bytes: 0, .. }));
    }

    #[test]
    fn below_the_minimum_with_oversubscription_warns_and_plans_one_full_context() {
        let min_pages = 262_144u32.div_ceil(64);
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
            kv_pool_bytes: Some(named),
            ..request(DERIVED, 31 * GIB)
        })
        .expect("fits");
        let pages = plan_kv_pool(FORMAT, QWEN, named).page_count;
        assert_eq!(plan.kv_page_count, pages);
        assert_eq!(plan.kv_pool_bytes, arena(pages));
        assert!(plan.total_bytes < plan.budget_bytes);
    }

    #[test]
    fn a_named_kv_pool_past_the_budget_refuses_and_names_the_flag() {
        let err = plan_vram(&VramRequest {
            kv_pool_bytes: Some(10 * GIB),
            ..request(DERIVED, 30 * GIB)
        })
        .expect_err("past the budget");
        let message = err.to_string();
        assert!(message.contains("--kv-pool-bytes"), "{message}");
    }

    #[test]
    fn the_lines_keep_plan_order_and_add_up() {
        let names: Vec<_> = lines().entries().iter().map(|(name, _)| *name).collect();
        assert_eq!(
            names,
            [
                "weights",
                "cuda_context",
                "prefill_scratch",
                "vision_workspace",
                "media_embedding",
                "sampling",
                "decode_graph",
                "verify_round",
                "drafter_round",
                "lane_state",
                "retained_slots",
                "retained",
                "residual",
            ]
        );
        assert_eq!(
            lines().total(),
            lines().entries().iter().map(|(_, b)| b).sum::<u64>()
        );
    }
}
