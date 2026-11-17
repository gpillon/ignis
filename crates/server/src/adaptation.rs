//! **Adaptation** (ADR 0063 §3, spec 27b-16gb/04 solution 4, `GLOSSARY.md`):
//! a 27B load whose VRAM plan refuses its profile's values lowers the ones
//! the operator left to the profile or the hardcoded default, in a fixed
//! order, until the plan fits -- and says which it lowered.
//!
//! The **ladder** is a pure function of the values it starts from, which of
//! them may adapt, its two config fields ([`Ladder`]) and a closure that
//! plans at given values. The planner ([`ignis_core::plan_vram`]) stays a
//! pure function of its inputs, called again with the next attempt's; the
//! load (`runtime::qwen27b_load`) supplies the closure, which re-plans the
//! leaf's lines at each attempt's context, lanes and head. So every rule
//! below is a CPU test with injected budgets.
//!
//! Before any of them, the **solo width** (GitHub #489, ADR 0063 §3 as
//! amended; [`adapt_solo`]): at `auto`, each width of the card's ladder is
//! planned at the start's values, and the first the plan fits is kept, with
//! nothing else adapted for it and, at more than one lane, the KV pool still
//! holding one full sequence and the spare reservation (the criterion an
//! adapted context is kept by); otherwise the solo width is the prefill
//! chunk and the order below runs. A width the operator named is the one
//! explicit value that adapts: when the plan refuses it below its minimum,
//! its drop to the prefill chunk is the first step.
//!
//! The order, each adaptation taken only while the plan is still refused:
//! 1. the drafter's proposal head: a `shortlist` from the profile falls to
//!    `full`, freeing the shortlist's ~357 MB (it is bound beside the full
//!    head). `auto` resolves to `full` already, so on today's profiles this
//!    never fires;
//! 2. `max_context`, from its value down every multiple of the ladder's
//!    context step (`model.adapt_context_step`) to one step, keeping the
//!    largest at which the plan fits and the KV pool holds one full sequence
//!    and a spare reservation of `vram.adapt_spare_tokens` tokens (at one
//!    lane, where a spare reservation has no lane to serve, the plan's own
//!    floor alone);
//! 3. `decode_lanes` to 1, after which the context ladder runs again from the
//!    value it started from. Lanes that cannot drop leave the largest context
//!    the plan itself holds, the spare reservation unmet, and the log says
//!    so.
//!
//! A start the plan fits as configured adapts nothing, whatever room its
//! pool leaves. A plan refused for any other reason than its minimum (a
//! budget above free memory, a named KV pool below its floor) is not adapted
//! either: no adaptation lowers what that refusal names. When not even one
//! step's context fits, the start is refused with the last attempt's deficit
//! and every adaptation tried, in order.

use ignis_core::{ProposalHead, VramPlanError};

/// The ladder's two config fields (ADR 0063 §3), both 65,536 tokens unless
/// named: the context step, which is also the floor
/// (`model.adapt_context_step`, a whole number of KV pages), and the spare
/// reservation a context keeps room for beside one full sequence at more
/// than one lane (`vram.adapt_spare_tokens`; 0, one sequence is enough).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ladder {
    pub context_step: u32,
    pub spare_tokens: u32,
}

impl Ladder {
    /// Whether a KV pool of `kv_tokens` keeps what a kept context needs: one
    /// full sequence of `context` tokens and, at more than one lane, the spare
    /// reservation beside it; at one lane, where a spare reservation has no
    /// lane to serve, the plan's own floor, which a plan that fits holds.
    fn keeps(&self, context: u32, lanes: u32, kv_tokens: u64) -> bool {
        lanes <= 1 || kv_tokens >= u64::from(context) + u64::from(self.spare_tokens)
    }
}

impl Default for Ladder {
    /// The fields' defaults ([`crate::config::DEFAULT_ADAPT_CONTEXT_STEP`],
    /// [`crate::config::DEFAULT_ADAPT_SPARE_TOKENS`]).
    fn default() -> Self {
        Self {
            context_step: crate::config::DEFAULT_ADAPT_CONTEXT_STEP,
            spare_tokens: crate::config::DEFAULT_ADAPT_SPARE_TOKENS,
        }
    }
}

/// The values a 27B load's VRAM plan is made at that adaptation may lower.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attempt {
    pub max_context: u32,
    pub decode_lanes: u32,
    /// The drafter's proposal head; `None` without a drafter.
    pub draft_head: Option<ProposalHead>,
    /// The solo width the prefill scratch is planned for (GitHub #489): the
    /// prefill chunk, unless [`adapt_solo`] tries a wider one.
    pub prefill_chunk_solo: u32,
}

/// Which of an [`Attempt`]'s values the operator left to the profile or the
/// hardcoded default ([`crate::config::Config::adaptable`]): only those
/// adapt. An explicit value never does, and is refused as it always was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Adaptable {
    pub max_context: bool,
    pub decode_lanes: bool,
    pub draft_head: bool,
}

impl Adaptable {
    /// Every value adapts: a start that names none of them.
    pub const ALL: Self = Self { max_context: true, decode_lanes: true, draft_head: true };
    /// No value adapts: a start that names all three, or a shape built by
    /// hand, which is refused as it was before adaptation existed.
    pub const NONE: Self = Self { max_context: false, decode_lanes: false, draft_head: false };
}

/// What a context the ladder kept leaves beside one full sequence: the
/// criterion it was kept by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Room {
    /// A spare reservation of this many tokens, at more than one lane.
    Spare(u32),
    /// One lane: the plan's own floor, a spare reservation having no lane to
    /// serve.
    OneLane,
    /// Not the spare reservation of this many tokens: the lanes could not
    /// drop, so the largest context the plan itself holds.
    Short(u32),
}

/// One adaptation a load took, in the words of `ignis.runtime.vram_plan`'s
/// `adapted` list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptStep {
    DraftHead { from: ProposalHead, to: ProposalHead },
    /// Down the ladder's `context_step` at a time, kept by `room`.
    MaxContext { from: u32, to: u32, context_step: u32, room: Room },
    DecodeLanes { from: u32, to: u32 },
    /// A solo width the operator named, dropped to the prefill chunk (GitHub
    /// #489).
    PrefillChunkSolo { from: u32, to: u32 },
}

impl std::fmt::Display for AdaptStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::DraftHead { from, to } => write!(f, "draft_head {} -> {}", from.as_str(), to.as_str()),
            Self::MaxContext { from, to, context_step, room } => {
                write!(f, "max_context {from} -> {to} (adapt_context_step {context_step}, ")?;
                match room {
                    Room::Spare(tokens) => write!(f, "adapt_spare_tokens {tokens})"),
                    Room::OneLane => write!(f, "one lane)"),
                    Room::Short(tokens) => write!(
                        f,
                        "one sequence only: no room for adapt_spare_tokens {tokens}, and decode_lanes is explicit)"
                    ),
                }
            }
            Self::DecodeLanes { from, to } => write!(f, "decode_lanes {from} -> {to}"),
            Self::PrefillChunkSolo { from, to } => write!(f, "prefill_chunk_solo {from} -> {to}"),
        }
    }
}

/// `steps` as the `adapted` field of `ignis.runtime.vram_plan` spells them:
/// in the order taken, comma-separated; empty when the load adapted nothing.
pub fn steps_text(steps: &[AdaptStep]) -> String {
    steps.iter().map(AdaptStep::to_string).collect::<Vec<_>>().join(", ")
}

/// What one plan at an [`Attempt`] came to.
#[derive(Debug)]
pub enum Planned<T> {
    /// The plan fits; its KV pool holds `kv_tokens` tokens.
    Fits { plan: T, kv_tokens: u64 },
    /// The plan does not fit the attempt: the planner's refusal.
    Unfit(VramPlanError),
}

/// The values a load runs at, the plan made at them, and the steps that got
/// there (none when the configured values fit).
#[derive(Debug)]
pub struct Adapted<T> {
    pub attempt: Attempt,
    pub plan: T,
    pub steps: Vec<AdaptStep>,
}

/// One adaptation a refused start tried: a step, or a walk down the context
/// ladder (every context planned, the first being where it started).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tried {
    Step(AdaptStep),
    Contexts(Vec<u32>),
}

impl std::fmt::Display for Tried {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Step(step) => step.fmt(f),
            Self::Contexts(contexts) => write!(
                f,
                "max_context {}",
                contexts.iter().map(u32::to_string).collect::<Vec<_>>().join(" -> ")
            ),
        }
    }
}

/// A start adaptation could not fit: the plan's refusal at the last attempt
/// -- its deficit, as the plan words it -- and every adaptation tried before
/// it, in order. With nothing tried it reads exactly as the plan's refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartRefusal {
    pub error: VramPlanError,
    pub tried: Vec<Tried>,
}

impl std::fmt::Display for StartRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)?;
        if !self.tried.is_empty() {
            let tried = self.tried.iter().map(Tried::to_string).collect::<Vec<_>>().join(", ");
            write!(f, "; adaptation tried, in order: {tried}")?;
        }
        Ok(())
    }
}

/// Why [`adapt`] returned no plan: the start is refused, or the closure
/// itself failed.
#[derive(Debug)]
pub enum AdaptError<E> {
    Refused(StartRefusal),
    Failed(E),
}

/// The contexts the ladder plans from `from` at `context_step` tokens a step
/// (ADR 0063 §3): `from` itself, then every multiple of the step below it,
/// down to one step -- 262,144, 196,608, 131,072, 65,536 on the 16 GB
/// profile at the default step; 917,504 down 65,536 at a time on the
/// 5090's. A context at or below one step is its own only candidate. The
/// step is nonzero (`model.adapt_context_step` is validated so).
pub fn contexts(from: u32, context_step: u32) -> Vec<u32> {
    assert!(context_step > 0, "a context step of no tokens");
    let below = (from.saturating_sub(1) / context_step) * context_step;
    let mut contexts = vec![from];
    contexts.extend((1..=below / context_step).rev().map(|n| n * context_step));
    contexts
}

/// The solo widths a 27B load tries before any adaptation (GitHub #489).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SoloCandidates {
    /// `auto`: these widths, widest first, each kept only if the plan fits
    /// it at the start's values with the room an adapted context keeps;
    /// then the prefill chunk. Empty on a card with no ladder
    /// ([`crate::config::arch::solo_ladder`]).
    Auto(Vec<u32>),
    /// A width the operator or a profile named.
    Named(u32),
}

/// [`adapt`] with the solo width first (GitHub #489, ADR 0063 §3 as
/// amended): `start` is at the prefill chunk (its `prefill_chunk_solo`), and
/// `solo` says what is tried above it. An `auto` width is kept where the plan
/// fits it and, at more than one lane, its KV pool holds the start's context
/// and `ladder`'s spare reservation; one that does not is no step and leaves
/// no trace. A named width the plan refuses below its minimum
/// is dropped to the prefill chunk as the first step, before the head, the
/// context and the lanes. Any other refusal of a named width is the start's.
pub fn adapt_solo<T, E>(
    start: Attempt,
    solo: &SoloCandidates,
    adaptable: Adaptable,
    ladder: Ladder,
    mut plan: impl FnMut(Attempt) -> Result<Planned<T>, E>,
) -> Result<Adapted<T>, AdaptError<E>> {
    let named = match solo {
        SoloCandidates::Auto(widths) => {
            for &width in widths.iter().filter(|&&width| width > start.prefill_chunk_solo) {
                let attempt = Attempt { prefill_chunk_solo: width, ..start };
                match plan(attempt).map_err(AdaptError::Failed)? {
                    Planned::Fits { plan, kv_tokens } if ladder.keeps(start.max_context, start.decode_lanes, kv_tokens) => {
                        return Ok(Adapted { attempt, plan, steps: Vec::new() });
                    }
                    Planned::Fits { .. } | Planned::Unfit(_) => {}
                }
            }
            return adapt(start, adaptable, ladder, plan);
        }
        &SoloCandidates::Named(width) if width > start.prefill_chunk_solo => width,
        SoloCandidates::Named(_) => return adapt(start, adaptable, ladder, plan),
    };
    let attempt = Attempt { prefill_chunk_solo: named, ..start };
    match plan(attempt).map_err(AdaptError::Failed)? {
        Planned::Fits { plan, .. } => return Ok(Adapted { attempt, plan, steps: Vec::new() }),
        Planned::Unfit(VramPlanError::BelowMinimum { .. }) => {}
        Planned::Unfit(error) => return Err(AdaptError::Refused(StartRefusal { error, tried: Vec::new() })),
    }
    let step = AdaptStep::PrefillChunkSolo { from: named, to: start.prefill_chunk_solo };
    match adapt(start, adaptable, ladder, plan) {
        Ok(mut adapted) => {
            adapted.steps.insert(0, step);
            Ok(adapted)
        }
        Err(AdaptError::Refused(mut refusal)) => {
            refusal.tried.insert(0, Tried::Step(step));
            Err(AdaptError::Refused(refusal))
        }
        Err(failed) => Err(failed),
    }
}

/// Plan at `start`, and when the plan refuses it below its minimum, adapt
/// what `adaptable` allows, in the order the module doc gives, down
/// `ladder`'s contexts and against its spare reservation, until a plan
/// fits. `plan` plans at an attempt: [`Planned::Unfit`] carries the
/// planner's refusal, an `Err` a failure of its own, which ends the ladder.
pub fn adapt<T, E>(
    start: Attempt,
    adaptable: Adaptable,
    ladder: Ladder,
    mut plan: impl FnMut(Attempt) -> Result<Planned<T>, E>,
) -> Result<Adapted<T>, AdaptError<E>> {
    let mut plan = |attempt| plan(attempt).map_err(AdaptError::Failed);
    let refused = |error, tried| AdaptError::Refused(StartRefusal { error, tried });
    let mut last = match plan(start)? {
        Planned::Fits { plan, .. } => return Ok(Adapted { attempt: start, plan, steps: Vec::new() }),
        Planned::Unfit(error @ VramPlanError::BelowMinimum { .. }) => error,
        Planned::Unfit(error) => return Err(refused(error, Vec::new())),
    };
    let mut attempt = start;
    let mut steps = Vec::new();
    let mut tried = Vec::new();

    // 1. The drafter's head: a shortlist the operator did not name.
    if adaptable.draft_head && attempt.draft_head == Some(ProposalHead::Shortlist) {
        let step = AdaptStep::DraftHead { from: ProposalHead::Shortlist, to: ProposalHead::Full };
        attempt.draft_head = Some(ProposalHead::Full);
        steps.push(step);
        tried.push(Tried::Step(step));
        match plan(attempt)? {
            Planned::Fits { plan, .. } => return Ok(Adapted { attempt, plan, steps }),
            Planned::Unfit(error @ VramPlanError::BelowMinimum { .. }) => last = error,
            Planned::Unfit(error) => return Err(refused(error, tried)),
        }
    }

    // 2. The context, at these lanes, then 3. at one lane. The first pass
    // skips the context just refused at them.
    let candidates =
        if adaptable.max_context { contexts(start.max_context, ladder.context_step) } else { vec![start.max_context] };
    let context_step = |to, room| AdaptStep::MaxContext { from: start.max_context, to, context_step: ladder.context_step, room };
    let mut skip = 1;
    loop {
        let room = if attempt.decode_lanes <= 1 { Room::OneLane } else { Room::Spare(ladder.spare_tokens) };
        let mut fallback = None;
        for &context in &candidates[skip..] {
            let candidate = Attempt { max_context: context, ..attempt };
            match plan(candidate)? {
                Planned::Fits { plan, kv_tokens } if ladder.keeps(context, attempt.decode_lanes, kv_tokens) => {
                    if context != start.max_context {
                        steps.push(context_step(context, room));
                    }
                    return Ok(Adapted { attempt: candidate, plan, steps });
                }
                // The largest context the plan holds, the spare reservation
                // unmet: kept only if the lanes cannot drop.
                Planned::Fits { plan, .. } => {
                    if fallback.is_none() {
                        fallback = Some((candidate, plan));
                    }
                }
                Planned::Unfit(error @ VramPlanError::BelowMinimum { .. }) => last = error,
                Planned::Unfit(error) => return Err(refused(error, tried)),
            }
        }
        if candidates.len() > 1 {
            tried.push(Tried::Contexts(candidates.clone()));
        }
        if !adaptable.decode_lanes || room == Room::OneLane {
            return match fallback {
                Some((kept, plan)) => {
                    steps.push(context_step(kept.max_context, Room::Short(ladder.spare_tokens)));
                    Ok(Adapted { attempt: kept, plan, steps })
                }
                None => Err(refused(last, tried)),
            };
        }
        let step = AdaptStep::DecodeLanes { from: attempt.decode_lanes, to: 1 };
        attempt.decode_lanes = 1;
        steps.push(step);
        tried.push(Tried::Step(step));
        skip = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_core::{KvFormat, KvGeometry, Residency, VramLines, VramMode, VramPlan};
    use std::convert::Infallible;

    /// One hq-e8-2b page: 64 tokens at 9,216 bytes each.
    fn page_bytes() -> u64 {
        KvFormat::HqE8_2b.page_bytes(KvGeometry::qwen38_27b())
    }

    /// The shortlist head's bytes beside the full head (plans of 2026-10-20).
    const SHORTLIST_BYTES: u64 = 357_040_380;

    /// The defaults: 65,536-token steps and a 65,536-token spare reservation.
    const DEFAULT: Ladder = Ladder { context_step: 65_536, spare_tokens: 65_536 };

    /// A card the tests plan on: the real planner over lines shaped like
    /// the 27B's -- weights, the drafter head's, a workspace whose hq planes
    /// follow the context up to 262,144 keys, a per-lane cost, and two host
    /// retained slots a lane, a KV page each -- inside an injected budget.
    #[derive(Clone, Copy)]
    struct Card {
        budget: u64,
        weights: u64,
        /// The workspace below its 4,096 bytes per context token.
        workspace: u64,
        /// What a lane costs: its state, hq window and round buffers.
        lane: u64,
        /// The prefill chunk its workspace is planned at; a wider solo width
        /// costs `SOLO_SCRATCH_PER_TOKEN` a token above it.
        chunk: u32,
    }

    /// The 5090's 27B under `rtx5090`: its plan's lines at 8 lanes, from
    /// `vram_plan` events (`.scratch/27b-16gb-study-2026-10-20`).
    const RTX5090: Card = Card {
        budget: 0,
        weights: 18_585_584_640 + 452_595_712 + 335_544_320 + 124_474_800,
        workspace: 1_508_311_552 - 4_096 * 262_144,
        lane: 196_880_384 + 35_651_648 + (4_623_344 + 82_125_920 + 152_139_488 + 78_027_552) / 8,
        chunk: CHUNK,
    };

    /// A 16 GB card's 27B under `rtx-16gb`: a stand-in for the 16 GB
    /// body's weights (#338), text only.
    const GB16: Card = Card { budget: 0, weights: 12_500_000_000, workspace: 342_290_432, lane: 268_000_000, chunk: CHUNK };

    /// The 12 GB artifact's load under `rtx-12gb` (spec #630, GitHub #637),
    /// planned as the server plans it: the lines its plan logged on the RTX
    /// 5090 in a 12 GiB budget on 2026-11-16 (#635, on `ig28a`, whose objects
    /// the admitted artifact carries), one lane drafting with MTP at 7, the
    /// 512-token chunk, vision on in host RAM at 4,096 tokens; a pool whose
    /// pages hold the MTP head's own attention layer beside the 27B's, the
    /// pages admission keeps back for the head's holders, and the pool's block
    /// tables, 4 bytes a context page a lane. No line follows the context on
    /// the trellis body (ADR 0061), and these are one lane's.
    #[derive(Clone, Copy)]
    struct Rtx12gbLoad {
        budget: u64,
        /// The dense prefill's buffer (GitHub #509): 0 on Blackwell's routes,
        /// the INT8 copy of the body's largest matrix on the routes an RTX 30
        /// or 40 card runs, the one line those routes move at this chunk.
        trellis_dense: u64,
        /// The CUDA context the card measures: the RTX 5090's
        /// (`CUDA_CONTEXT_5090`) but where a real card read its own.
        cuda_context: u64,
    }

    /// A 12 GB card's whole memory.
    const GIB_12: u64 = 12 << 30;

    /// The dense prefill's one buffer on the 27B: 5120 x 17408.
    const TRELLIS_DENSE: u64 = 89_128_960;

    /// The CUDA context the RTX 5090 measured under Windows, #635's load.
    const CUDA_CONTEXT_5090: u64 = 452_161_536;

    /// The CUDA context an RTX 3060 12 GB measured under Linux with nothing
    /// else on it, the first real 12 GB card (GitHub #639, 2026-11-17).
    const CUDA_CONTEXT_3060: u64 = 114_425_856;

    impl Rtx12gbLoad {
        /// Blackwell's routes, in `budget` bytes.
        fn in_budget(budget: u64) -> Self {
            Self { budget, trellis_dense: 0, cuda_context: CUDA_CONTEXT_5090 }
        }

        /// The plan's lines but the pool's, as #635's load logged them.
        fn lines(&self) -> VramLines {
            VramLines {
                weights: 9_651_141_632,
                cuda_context: self.cuda_context,
                workspace: 302_433_536,
                trellis_workspace: 32_134_400,
                trellis_dense: self.trellis_dense,
                sampling: 579_048,
                decode_graph: 12_802_060,
                verify_round: 19_017_436,
                lane_state: 154_947_584,
                hq_residual_window: 37_879_936,
                residual: 124_474_800,
                ..VramLines::default()
            }
        }

        /// One page of the MTP head's pool: the 27B's 16 attention layers and
        /// the head's own.
        fn page() -> u64 {
            let body = KvGeometry::qwen38_27b();
            KvFormat::HqE8_2b.page_bytes(KvGeometry { gqa_layers: body.gqa_layers + 1, ..body })
        }

        /// The pool's block tables at `attempt`.
        fn block_tables(attempt: Attempt) -> u64 {
            4 * u64::from(attempt.max_context.div_ceil(64)) * u64::from(attempt.decode_lanes)
        }

        /// The pages of the plan's floor at `attempt`: one sequence, a page
        /// for each of the lane's two retained slots, and one for each holder
        /// of a copy the MTP head keeps back, the lane and its two slots
        /// (GitHub #623).
        fn floor_pages(attempt: Attempt) -> u32 {
            attempt.max_context.div_ceil(64) + 2 + 3
        }

        /// The budget at which `attempt` just holds its floor.
        fn floor_budget(attempt: Attempt) -> u64 {
            Self::in_budget(0).lines().total() + u64::from(Self::floor_pages(attempt)) * Self::page() + Self::block_tables(attempt)
        }

        fn plan(&self, attempt: Attempt) -> Result<VramPlan, VramPlanError> {
            assert_eq!(attempt.decode_lanes, 1, "the lines are one lane's");
            let page = Self::page();
            let arena = move |pages: u32| u64::from(pages) * page + Self::block_tables(attempt);
            ignis_core::plan_vram(&ignis_core::VramRequest {
                mode: VramMode::Explicit { budget_bytes: self.budget, allow_oversubscription: false },
                free_at_start_bytes: u64::MAX,
                lines: self.lines(),
                kv_page_bytes: page,
                max_context_tokens: attempt.max_context,
                retained_slots: 2,
                holder_pages: 3,
                kv_pool: None,
                residency: Residency::Resident,
                embedding_pool_named: false,
                kv_arena_bytes: &arena,
                can_page: true,
            })
        }

        /// `rtx-12gb`'s ladder from its 131,072 tokens, with every attempt it
        /// planned.
        fn adapt(&self, ladder: Ladder) -> (Result<Adapted<VramPlan>, AdaptError<Infallible>>, Vec<Attempt>) {
            adapt_planning(rtx_12gb_at(131_072), Adaptable::ALL, ladder, |attempt| self.plan(attempt))
        }
    }

    /// The ladder from `start` over the plans `plan` makes, with every
    /// attempt it planned.
    fn adapt_planning(
        start: Attempt,
        adaptable: Adaptable,
        ladder: Ladder,
        plan: impl Fn(Attempt) -> Result<VramPlan, VramPlanError>,
    ) -> (Result<Adapted<VramPlan>, AdaptError<Infallible>>, Vec<Attempt>) {
        let mut attempts = Vec::new();
        let adapted = adapt(start, adaptable, ladder, |attempt| {
            attempts.push(attempt);
            Ok::<_, Infallible>(match plan(attempt) {
                Ok(plan) => Planned::Fits { kv_tokens: plan.kv_token_capacity(), plan },
                Err(error) => Planned::Unfit(error),
            })
        });
        (adapted, attempts)
    }

    /// The ladder `rtx-12gb` gives the 27B: 16,384-token steps (the owner's
    /// call of 2026-11-16), the default spare reservation (which one lane
    /// never asks for).
    const RTX_12GB_LADDER: Ladder = Ladder { context_step: 16_384, spare_tokens: 65_536 };

    /// `rtx-12gb`'s values at `max_context`: one lane, the MTP head (the
    /// full head), its scratch at the 512-token chunk.
    fn rtx_12gb_at(max_context: u32) -> Attempt {
        Attempt { max_context, decode_lanes: 1, draft_head: Some(ProposalHead::Full), prefill_chunk_solo: 512 }
    }

    impl Card {
        fn with_budget(self, budget: u64) -> Self {
            Self { budget, ..self }
        }

        /// Every line but the pool at `attempt`.
        fn held(&self, attempt: Attempt) -> u64 {
            let head = if attempt.draft_head == Some(ProposalHead::Shortlist) { SHORTLIST_BYTES } else { 0 };
            self.weights + head + self.workspace + 4_096 * u64::from(attempt.max_context.min(262_144))
                + self.lane * u64::from(attempt.decode_lanes)
                + SOLO_SCRATCH_PER_TOKEN * u64::from(attempt.prefill_chunk_solo - self.chunk)
        }

        /// The budget at which `attempt` keeps its context with exactly the
        /// pool `tokens` asks for.
        fn budget_holding(&self, attempt: Attempt, tokens: u32) -> u64 {
            self.held(attempt) + u64::from(tokens.div_ceil(64)) * page_bytes()
        }

        fn plan(&self, attempt: Attempt) -> Result<VramPlan, VramPlanError> {
            let page = page_bytes();
            let arena = move |pages: u32| u64::from(pages) * page;
            ignis_core::plan_vram(&ignis_core::VramRequest {
                mode: VramMode::Explicit { budget_bytes: self.budget, allow_oversubscription: false },
                free_at_start_bytes: u64::MAX,
                lines: VramLines { weights: self.held(attempt), ..VramLines::default() },
                kv_page_bytes: page,
                max_context_tokens: attempt.max_context,
                retained_slots: 2 * attempt.decode_lanes,
                holder_pages: 0,
                kv_pool: None,
                residency: Residency::Resident,
                embedding_pool_named: false,
                kv_arena_bytes: &arena,
                can_page: true,
            })
        }

        /// The ladder on this card, with every attempt it planned.
        fn adapt_with(
            &self,
            start: Attempt,
            adaptable: Adaptable,
            ladder: Ladder,
        ) -> (Result<Adapted<VramPlan>, AdaptError<Infallible>>, Vec<Attempt>) {
            adapt_planning(start, adaptable, ladder, |attempt| self.plan(attempt))
        }

        /// The ladder at its default fields.
        fn adapt(&self, start: Attempt, adaptable: Adaptable) -> (Result<Adapted<VramPlan>, AdaptError<Infallible>>, Vec<Attempt>) {
            self.adapt_with(start, adaptable, DEFAULT)
        }

        /// The ladder with the solo width first (GitHub #489), at its default
        /// fields.
        fn adapt_solo(
            &self,
            start: Attempt,
            solo: &SoloCandidates,
            adaptable: Adaptable,
        ) -> (Result<Adapted<VramPlan>, AdaptError<Infallible>>, Vec<Attempt>) {
            let mut attempts = Vec::new();
            let adapted = adapt_solo(start, solo, adaptable, DEFAULT, |attempt| {
                attempts.push(attempt);
                Ok::<_, Infallible>(match self.plan(attempt) {
                    Ok(plan) => Planned::Fits { kv_tokens: plan.kv_token_capacity(), plan },
                    Err(error) => Planned::Unfit(error),
                })
            });
            (adapted, attempts)
        }
    }

    /// The prefill chunk the tests' starts take: the profiles' 2,048.
    const CHUNK: u32 = 2048;

    /// What a solo width costs the tests' cards past the prefill chunk's
    /// scratch, a token: 8,192 takes ~0.6 GB more, about what the 24 GB
    /// class's routes are estimated to (spec #503).
    const SOLO_SCRATCH_PER_TOKEN: u64 = 100_000;

    fn values(max_context: u32, decode_lanes: u32) -> Attempt {
        Attempt { max_context, decode_lanes, draft_head: Some(ProposalHead::Full), prefill_chunk_solo: CHUNK }
    }

    /// `start` with its scratch planned for `solo` tokens.
    fn solo_at(start: Attempt, solo: u32) -> Attempt {
        Attempt { prefill_chunk_solo: solo, ..start }
    }

    /// The `auto` ladder of an RTX 30 or RTX 40 card, above a 1,024-token
    /// chunk as above the tests' 2,048 (the ladder is filtered to the widths
    /// above the chunk).
    fn auto_8x() -> SoloCandidates {
        SoloCandidates::Auto(vec![8192, 4096, 2048])
    }

    /// The budget at which the 24 GB class's start holds its whole context
    /// and the default spare reservation beside it, at `solo`: the room
    /// `auto` keeps a width by.
    fn rtx24_holding(start: Attempt, solo: u32) -> u64 {
        RTX5090.budget_holding(solo_at(start, solo), 1_048_576 + 65_536)
    }

    /// The budget at which the 24 GB class's start just holds its whole
    /// context and its eight lanes' retained slots' tail pages, at `solo`:
    /// the plan's own floor, below which a start adapts.
    fn rtx24_floor(start: Attempt, solo: u32) -> u64 {
        RTX5090.budget_holding(solo_at(start, solo), 1_048_576 + 16 * 64)
    }

    /// GitHub #489: `auto` keeps 8,192 where the plan fits it at the
    /// profile's values with the spare reservation: one plan, nothing
    /// adapted, no step named.
    #[test]
    fn auto_keeps_the_widest_solo_width_the_plan_holds_with_nothing_adapted() {
        let start = values(1_048_576, 8);
        let card = RTX5090.with_budget(rtx24_holding(start, 8192));
        let (result, attempts) = card.adapt_solo(start, &auto_8x(), Adaptable::ALL);
        let adapted = kept(result);
        assert_eq!((adapted.attempt, adapted.steps.as_slice()), (solo_at(start, 8192), &[][..]));
        assert_eq!(attempts, [solo_at(start, 8192)]);
    }

    /// A byte short of 8,192's room, 4,096; a byte short of 4,096's, the
    /// prefill chunk -- each at the full context, never a smaller context
    /// for a wider chunk, and never named in `adapted`.
    #[test]
    fn auto_walks_8192_then_4096_then_the_prefill_chunk_at_the_profiles_values() {
        let start = values(1_048_576, 8);
        let card = RTX5090.with_budget(rtx24_holding(start, 8192) - 1);
        let (result, attempts) = card.adapt_solo(start, &auto_8x(), Adaptable::ALL);
        assert_eq!(kept(result).attempt, solo_at(start, 4096));
        assert_eq!(attempts, [solo_at(start, 8192), solo_at(start, 4096)]);

        let card = RTX5090.with_budget(rtx24_holding(start, 4096) - 1);
        let (result, attempts) = card.adapt_solo(start, &auto_8x(), Adaptable::ALL);
        let adapted = kept(result);
        assert_eq!((adapted.attempt, adapted.steps.as_slice()), (start, &[][..]));
        assert_eq!(attempts, [solo_at(start, 8192), solo_at(start, 4096), start]);
    }

    /// A width the plan fits whose pool keeps one sequence but not the spare
    /// reservation beside it is passed over, as a context is (ADR 0063 §3):
    /// the room for a second request comes before a wider chunk. At one
    /// lane, where the reservation has no lane to serve, the plan's own
    /// floor is enough.
    #[test]
    fn auto_keeps_the_spare_reservation_at_more_than_one_lane() {
        let start = values(1_048_576, 8);
        let card = RTX5090.with_budget(rtx24_holding(start, 4096));
        assert!(card.plan(solo_at(start, 8192)).is_ok(), "8,192 fits, its pool short of the spare reservation");
        let (result, attempts) = card.adapt_solo(start, &auto_8x(), Adaptable::ALL);
        assert_eq!(kept(result).attempt, solo_at(start, 4096), "{attempts:?}");
        let one = values(1_048_576, 1);
        let card = RTX5090.with_budget(RTX5090.budget_holding(solo_at(one, 8192), 1_048_576 + 2 * 64));
        assert_eq!(kept(card.adapt_solo(one, &auto_8x(), Adaptable::ALL).0).attempt, solo_at(one, 8192));
    }

    /// The ladder steps down to 2,048 where the chunk is narrower: a
    /// 1,024-token chunk whose 4,096 leaves no spare reservation keeps 2,048.
    #[test]
    fn auto_steps_down_to_2048_above_a_1024_token_chunk() {
        let start = Attempt { prefill_chunk_solo: 1024, ..values(262_144, 2) };
        let card = Card { chunk: 1024, ..GB16 };
        let card = card.with_budget(card.budget_holding(solo_at(start, 2048), 262_144 + 65_536));
        let (result, attempts) = card.adapt_solo(start, &auto_8x(), Adaptable::ALL);
        assert_eq!(kept(result).attempt, solo_at(start, 2048), "{attempts:?}");
        assert_eq!(attempts.iter().map(|a| a.prefill_chunk_solo).collect::<Vec<_>>(), [8192, 4096, 2048]);
    }

    /// Where even the prefill chunk needs the context lowered, the context
    /// adapts at the prefill chunk, as before the solo width existed.
    #[test]
    fn auto_never_trades_context_for_a_wider_chunk() {
        let start = values(1_048_576, 8);
        let card = RTX5090.with_budget(rtx24_floor(start, CHUNK) - 100_000_000);
        let (result, _) = card.adapt_solo(start, &auto_8x(), Adaptable::ALL);
        let adapted = kept(result);
        assert_eq!(adapted.attempt.prefill_chunk_solo, CHUNK);
        assert!(adapted.attempt.max_context < 1_048_576, "{:?}", adapted.attempt);
        assert!(matches!(adapted.steps[..], [AdaptStep::MaxContext { .. }]), "{:?}", adapted.steps);
    }

    /// Blackwell's ladder is empty, and a width at or below the chunk is no
    /// candidate: one plan at the prefill chunk.
    #[test]
    fn an_empty_ladder_plans_the_prefill_chunk_alone() {
        let start = values(1_048_576, 8);
        let card = RTX5090.with_budget(rtx24_holding(start, 8192));
        for solo in [SoloCandidates::Auto(Vec::new()), SoloCandidates::Auto(vec![CHUNK, 1024]), SoloCandidates::Named(CHUNK)] {
            let (result, attempts) = card.adapt_solo(start, &solo, Adaptable::ALL);
            assert_eq!(kept(result).attempt, start, "{solo:?}");
            assert_eq!(attempts, [start], "{solo:?}");
        }
    }

    /// A named width the plan holds is kept; one it refuses below its
    /// minimum falls to the prefill chunk first, named as the first step,
    /// even with nothing else adaptable -- then the context adapts after it.
    #[test]
    fn a_named_solo_width_is_dropped_to_the_prefill_chunk_as_the_first_step() {
        let start = values(1_048_576, 8);
        let named = SoloCandidates::Named(8192);
        let card = RTX5090.with_budget(rtx24_floor(start, 8192));
        assert_eq!(kept(card.adapt_solo(start, &named, Adaptable::NONE).0).attempt, solo_at(start, 8192), "no spare reservation asked");

        let card = RTX5090.with_budget(rtx24_floor(start, 8192) - 1);
        let (result, attempts) = card.adapt_solo(start, &named, Adaptable::NONE);
        let adapted = kept(result);
        assert_eq!(adapted.attempt, start);
        assert_eq!(steps_text(&adapted.steps), "prefill_chunk_solo 8192 -> 2048");
        assert_eq!(attempts, [solo_at(start, 8192), start]);

        let card = RTX5090.with_budget(rtx24_floor(start, CHUNK) - 100_000_000);
        let adapted = kept(card.adapt_solo(start, &named, Adaptable::ALL).0);
        let text = steps_text(&adapted.steps);
        assert!(text.starts_with("prefill_chunk_solo 8192 -> 2048, max_context 1048576 -> "), "{text}");
    }

    /// A start that fits at no width is refused with the drop among the
    /// adaptations tried, first.
    #[test]
    fn a_refused_start_names_the_solo_drop_it_tried_first() {
        let start = values(1_048_576, 8);
        let card = RTX5090.with_budget(rtx24_floor(start, CHUNK) - 1);
        let refusal = refused(card.adapt_solo(start, &SoloCandidates::Named(8192), Adaptable::NONE).0);
        assert_eq!(refusal.tried, [Tried::Step(AdaptStep::PrefillChunkSolo { from: 8192, to: CHUNK })]);
        let text = refusal.to_string();
        assert!(text.ends_with("; adaptation tried, in order: prefill_chunk_solo 8192 -> 2048"), "{text}");
    }

    fn kept(result: Result<Adapted<VramPlan>, AdaptError<Infallible>>) -> Adapted<VramPlan> {
        match result {
            Ok(adapted) => adapted,
            Err(AdaptError::Refused(refused)) => panic!("refused: {refused}"),
            Err(AdaptError::Failed(never)) => match never {},
        }
    }

    fn refused(result: Result<Adapted<VramPlan>, AdaptError<Infallible>>) -> StartRefusal {
        match result {
            Ok(adapted) => panic!("adapted to {:?}: {}", adapted.attempt, steps_text(&adapted.steps)),
            Err(AdaptError::Refused(refused)) => refused,
            Err(AdaptError::Failed(never)) => match never {},
        }
    }

    /// A context step at the default fields, kept by the spare reservation.
    fn context_step(from: u32, to: u32) -> AdaptStep {
        AdaptStep::MaxContext { from, to, context_step: 65_536, room: Room::Spare(65_536) }
    }

    #[test]
    fn the_ladder_goes_down_one_step_at_a_time_from_any_value_to_one_step() {
        assert_eq!(contexts(262_144, 65_536), [262_144, 196_608, 131_072, 65_536]);
        assert_eq!(contexts(917_504, 65_536).len(), 14);
        assert_eq!(contexts(917_504, 65_536)[..3], [917_504, 851_968, 786_432]);
        assert_eq!(contexts(1_048_576, 65_536).len(), 16);
        assert_eq!(contexts(100_000, 65_536), [100_000, 65_536], "a value between steps, then the multiples below it");
        assert_eq!(contexts(65_536, 65_536), [65_536]);
        assert_eq!(contexts(40_960, 65_536), [40_960], "below one step: itself alone");
        // Another step: its own multiples, down to one of it.
        assert_eq!(contexts(917_504, 131_072)[..3], [917_504, 786_432, 655_360]);
        assert_eq!(contexts(917_504, 131_072).len(), 7);
        assert_eq!(contexts(262_144, 32_768), [262_144, 229_376, 196_608, 163_840, 131_072, 98_304, 65_536, 32_768]);
        assert_eq!(contexts(100, 64), [100, 64]);
    }

    /// Decision 5: on the 5090, a start that fits today -- 917,504 by a
    /// hair, its pool short of a spare reservation -- plans once and adapts
    /// nothing.
    #[test]
    fn a_5090_plan_that_fits_by_a_hair_adapts_nothing() {
        let start = values(917_504, 8);
        let card = RTX5090.with_budget(RTX5090.budget_holding(start, 917_504 + 16 * 64));
        let (result, attempts) = card.adapt(start, Adaptable::ALL);
        let adapted = kept(result);
        assert_eq!((adapted.attempt, adapted.steps.as_slice()), (start, &[][..]));
        assert_eq!(attempts, [start], "one plan, at the configured values");
        assert_eq!(steps_text(&adapted.steps), "", "the event's `adapted` is empty");
        assert!(
            adapted.plan.kv_token_capacity() < 917_504 + 65_536,
            "the pool holds no spare reservation, and the context is kept: {}",
            adapted.plan.kv_token_capacity()
        );
    }

    /// 100 MB short of 917,504, 851,968 fits but its pool cannot hold a
    /// spare 65,536-token reservation beside it: the load runs 786,432, as
    /// the 5090 did by hand with a 3.3 GB desktop (release 0.7.3).
    #[test]
    fn a_5090_plan_100_mb_short_keeps_the_largest_context_with_a_spare_reservation() {
        let start = values(917_504, 8);
        let card = RTX5090.with_budget(RTX5090.budget_holding(start, 917_504 + 16 * 64) - 100_000_000);
        let (result, attempts) = card.adapt(start, Adaptable::ALL);
        let adapted = kept(result);
        assert_eq!(adapted.attempt, values(786_432, 8));
        assert_eq!(adapted.steps, [context_step(917_504, 786_432)]);
        assert_eq!(
            steps_text(&adapted.steps),
            "max_context 917504 -> 786432 (adapt_context_step 65536, adapt_spare_tokens 65536)"
        );
        assert!(adapted.plan.kv_token_capacity() >= 786_432 + 65_536);
        assert_eq!(attempts.iter().map(|a| a.max_context).collect::<Vec<_>>(), [917_504, 851_968, 786_432]);
    }

    /// The two fields decide the ladder: on the same card, a 131,072-token
    /// step never plans 851,968, a spare reservation of 0 keeps the largest
    /// context the plan holds, and a larger spare one goes lower.
    #[test]
    fn the_step_and_the_spare_reservation_are_the_ladders_fields() {
        let start = values(917_504, 8);
        let card = RTX5090.with_budget(RTX5090.budget_holding(start, 917_504 + 16 * 64) - 100_000_000);

        let ladder = Ladder { context_step: 131_072, spare_tokens: 0 };
        let (result, attempts) = card.adapt_with(start, Adaptable::ALL, ladder);
        let adapted = kept(result);
        assert_eq!(adapted.attempt, values(786_432, 8));
        assert_eq!(attempts.iter().map(|a| a.max_context).collect::<Vec<_>>(), [917_504, 786_432]);
        assert_eq!(
            steps_text(&adapted.steps),
            "max_context 917504 -> 786432 (adapt_context_step 131072, adapt_spare_tokens 0)"
        );

        let ladder = Ladder { context_step: 65_536, spare_tokens: 0 };
        let adapted = kept(card.adapt_with(start, Adaptable::ALL, ladder).0);
        assert_eq!(adapted.attempt, values(851_968, 8), "one sequence is enough: the first context that fits");
        assert!(adapted.plan.kv_token_capacity() < 851_968 + 65_536);

        let ladder = Ladder { context_step: 65_536, spare_tokens: 131_072 };
        let adapted = kept(card.adapt_with(start, Adaptable::ALL, ladder).0);
        assert_eq!(adapted.attempt, values(720_896, 8));
        assert!(adapted.plan.kv_token_capacity() >= 720_896 + 131_072);
        assert_eq!(
            adapted.steps,
            [AdaptStep::MaxContext { from: 917_504, to: 720_896, context_step: 65_536, room: Room::Spare(131_072) }]
        );
    }

    /// The 16 GB profile's ladder: each budget keeps one step lower, never
    /// touching the lanes while a context at two lanes holds a spare
    /// reservation.
    #[test]
    fn the_16gb_ladder_steps_262144_196608_131072_65536() {
        let start = values(262_144, 2);
        let fits = GB16.with_budget(GB16.budget_holding(start, 262_144 + 4 * 64));
        assert!(kept(fits.adapt(start, Adaptable::ALL).0).steps.is_empty(), "262,144 fits as configured");
        for context in [196_608, 131_072, 65_536] {
            let card = GB16.with_budget(GB16.budget_holding(values(context, 2), context + 65_536));
            let adapted = kept(card.adapt(start, Adaptable::ALL).0);
            assert_eq!(adapted.attempt, values(context, 2), "{context}");
            assert_eq!(adapted.steps, [context_step(262_144, context)]);
        }
    }

    /// Spec #630 (GitHub #637), at the step the owner set on 2026-11-16: the
    /// 12 GB profile's ladder, at one lane, walks 131,072 down 16,384 at a
    /// time, each budget keeping one step lower, and names the step as every
    /// adaptation is named. A budget that holds 114,688 keeps it, where a
    /// 32,768-token step would drop it to 98,304 and the default step to
    /// 65,536: the context a coarser step wastes on a real card.
    #[test]
    fn the_12gb_ladder_steps_131072_down_16384_at_a_time_at_one_lane() {
        let fits = Rtx12gbLoad::in_budget(Rtx12gbLoad::floor_budget(rtx_12gb_at(131_072)));
        assert!(kept(fits.adapt(RTX_12GB_LADDER).0).steps.is_empty(), "131,072 fits as configured");
        for context in [114_688, 98_304, 81_920, 65_536, 49_152, 32_768, 16_384] {
            let card = Rtx12gbLoad::in_budget(Rtx12gbLoad::floor_budget(rtx_12gb_at(context)));
            let (result, attempts) = card.adapt(RTX_12GB_LADDER);
            let adapted = kept(result);
            assert_eq!(adapted.attempt, rtx_12gb_at(context), "{context}");
            assert_eq!(adapted.steps, [AdaptStep::MaxContext { from: 131_072, to: context, context_step: 16_384, room: Room::OneLane }]);
            assert!(attempts.iter().all(|a| a.decode_lanes == 1), "one lane has no lane to drop: {attempts:?}");
        }
        let card = Rtx12gbLoad::in_budget(Rtx12gbLoad::floor_budget(rtx_12gb_at(114_688)));
        assert_eq!(
            steps_text(&kept(card.adapt(RTX_12GB_LADDER).0).steps),
            "max_context 131072 -> 114688 (adapt_context_step 16384, one lane)"
        );
        // At 32,768 the same budget falls to 98,304, at the default step to 65,536.
        assert_eq!(kept(card.adapt(Ladder { context_step: 32_768, ..RTX_12GB_LADDER }).0).attempt, rtx_12gb_at(98_304));
        assert_eq!(kept(card.adapt(DEFAULT).0).attempt, rtx_12gb_at(65_536));
    }

    /// Spec #630 (GitHub #637): a 10 GiB card is read as the 12 GB class, and
    /// its plan refuses it once the ladder has tried every context, naming the
    /// bytes the last, smallest attempt misses; no lane step is tried at one
    /// lane.
    #[test]
    fn a_10gib_card_under_rtx_12gb_is_refused_naming_the_bytes_it_misses() {
        let budget = (10u64 << 30) - (512 << 20);
        let refused = refused(Rtx12gbLoad::in_budget(budget).adapt(RTX_12GB_LADDER).0);
        let needed = Rtx12gbLoad::floor_budget(rtx_12gb_at(16_384));
        assert!(matches!(refused.error, VramPlanError::BelowMinimum { max_context_tokens: 16_384, .. }), "{:?}", refused.error);
        let message = refused.to_string();
        assert!(
            message.contains(&format!("needs {needed} bytes")) && message.contains(&format!("{} bytes more than the {budget}-byte VRAM budget", needed - budget)),
            "{message}"
        );
        assert!(
            message.ends_with("; adaptation tried, in order: max_context 131072 -> 114688 -> 98304 -> 81920 -> 65536 -> 49152 -> 32768 -> 16384"),
            "{message}"
        );
    }

    /// Spec #630 (GitHub #637): in a 12 GiB budget the 12 GB artifact's plan
    /// is its load's, to the byte -- a KV pool of 3,346 pages, 2,096,906,240
    /// bytes with its block tables, the three the head's holders keep back
    /// among them, and 12,884,478,208 bytes in all -- at 131,072 tokens, with
    /// nothing adapted.
    #[test]
    fn in_12_gib_the_12gb_artifacts_plan_is_its_loads_to_the_byte_and_adapts_nothing() {
        let (result, attempts) = Rtx12gbLoad::in_budget(GIB_12).adapt(RTX_12GB_LADDER);
        let adapted = kept(result);
        assert_eq!((adapted.attempt, adapted.steps.as_slice()), (rtx_12gb_at(131_072), &[][..]));
        assert_eq!(attempts, [rtx_12gb_at(131_072)], "one plan");
        let plan = adapted.plan;
        assert_eq!((plan.kv_page_count, plan.holder_pages), (3_346, 3));
        assert_eq!((plan.kv_pool_bytes, plan.total_bytes), (2_096_906_240, 12_884_478_208));
    }

    /// Spec #630 (GitHub #637): less 1 GiB, 131,072 tokens miss their floor of
    /// 2,053 pages, 114,688 theirs of 1,797, and the ladder's next context,
    /// 98,304, holds its 1,541 in a pool of 1,633: the load adapts and says
    /// so.
    #[test]
    fn less_1_gib_the_12gb_artifacts_plan_adapts_to_98304_and_names_the_step() {
        let card = Rtx12gbLoad::in_budget(GIB_12 - (1 << 30));
        let VramPlanError::BelowMinimum { needed_bytes, .. } = card.plan(rtx_12gb_at(131_072)).unwrap_err() else {
            panic!("refused below its minimum");
        };
        assert_eq!(needed_bytes, 12_074_170_624, "2,053 pages beside the lines");
        let (result, attempts) = card.adapt(RTX_12GB_LADDER);
        let adapted = kept(result);
        assert_eq!(adapted.attempt, rtx_12gb_at(98_304));
        assert_eq!(steps_text(&adapted.steps), "max_context 131072 -> 98304 (adapt_context_step 16384, one lane)");
        assert_eq!(attempts.iter().map(|a| a.max_context).collect::<Vec<_>>(), [131_072, 114_688, 98_304]);
        assert_eq!(adapted.plan.kv_page_count, 1_633);
    }

    /// Spec #630 (GitHub #637): on the routes an RTX 30 or 40 card runs, the
    /// dense prefill's buffer is the plan's one line more: in 12 GiB it leaves
    /// 3,204 pages, 142 fewer, and 131,072 tokens still load; less 1 GiB, at
    /// the RTX 5090's context and trellis workspace, 98,304 misses its floor
    /// by 50 pages and the load adapts one step further, to 81,920.
    #[test]
    fn on_an_rtx_30_or_40_cards_routes_the_dense_buffer_keeps_131072_in_12_gib_and_81920_less_1_gib() {
        let routes_8x = |budget| Rtx12gbLoad { budget, trellis_dense: TRELLIS_DENSE, cuda_context: CUDA_CONTEXT_5090 };
        let adapted = kept(routes_8x(GIB_12).adapt(RTX_12GB_LADDER).0);
        assert_eq!((adapted.attempt, adapted.steps.as_slice()), (rtx_12gb_at(131_072), &[][..]));
        assert_eq!(adapted.plan.kv_page_count, 3_204);

        let card = routes_8x(GIB_12 - (1 << 30));
        let VramPlanError::BelowMinimum { needed_bytes, budget_bytes, .. } = card.plan(rtx_12gb_at(98_304)).unwrap_err() else {
            panic!("refused below its minimum");
        };
        assert_eq!((needed_bytes - budget_bytes).div_ceil(Rtx12gbLoad::page()), 50);
        let (result, attempts) = card.adapt(RTX_12GB_LADDER);
        let adapted = kept(result);
        assert_eq!(steps_text(&adapted.steps), "max_context 131072 -> 81920 (adapt_context_step 16384, one lane)");
        assert_eq!(attempts.iter().map(|a| a.max_context).collect::<Vec<_>>(), [131_072, 114_688, 98_304, 81_920]);
        assert_eq!(adapted.plan.kv_page_count, 1_491);
    }

    /// Spec #630 (GitHub #637), the figures the owner asked the guide for on
    /// 2026-11-16: on a real 12 GB card the budget is derived, the memory
    /// NVML reads free before the start less `rtx-12gb`'s 512 MiB of headroom,
    /// and the context is the plan's on the RTX 5090's lines (a real card
    /// confirms them, #639). From a card with nothing else on it, about 11.75
    /// of a 12 GB card's 11.9 GiB free, to one whose desktop holds about
    /// 1.3 GB, on Blackwell's routes and on an RTX 30 or 40 card's; then the
    /// free memory 131,072 tokens need on each, a MiB less taking one step.
    #[test]
    fn on_a_real_12gb_card_the_derived_budget_gives_each_desktop_its_context() {
        const HEADROOM: u64 = 512 << 20;
        let derived = |free_mib: u64, trellis_dense: u64| Rtx12gbLoad { budget: (free_mib << 20) - HEADROOM, trellis_dense, cuda_context: CUDA_CONTEXT_5090 };
        let context = |card: Rtx12gbLoad| card.adapt(RTX_12GB_LADDER).0.ok().map(|adapted| adapted.attempt.max_context);
        // Free MiB at the start; the context on Blackwell's routes, then on an RTX 30 or 40 card's.
        let cases = [
            (12_032, Some(131_072), Some(114_688)), // nothing else on the card
            (11_878, Some(114_688), Some(98_304)),  // a light desktop, ~0.3 GB
            (11_704, Some(81_920), Some(81_920)),   // ~0.5 GB
            (11_468, Some(65_536), Some(49_152)),   // ~0.75 GB
            (11_233, Some(32_768), Some(32_768)),   // ~1 GB
            (10_946, None, None),                   // ~1.3 GB: refused
        ];
        for (free_mib, blackwell, routes_8x) in cases {
            assert_eq!(context(derived(free_mib, 0)), blackwell, "{free_mib} MiB free, Blackwell's routes");
            assert_eq!(context(derived(free_mib, TRELLIS_DENSE)), routes_8x, "{free_mib} MiB free, an RTX 30 or 40 card's routes");
        }
        // 131,072 tokens need 12,027 MiB free on Blackwell's routes and 12,112
        // on an RTX 30 or 40 card's; each step below 256 pages less, ~153 MiB.
        for (free_mib, trellis_dense) in [(12_027, 0), (12_112, TRELLIS_DENSE)] {
            assert_eq!(context(derived(free_mib, trellis_dense)), Some(131_072), "{free_mib}");
            assert_eq!(context(derived(free_mib - 1, trellis_dense)), Some(114_688), "{free_mib} less 1 MiB");
        }
        let step = Rtx12gbLoad::floor_budget(rtx_12gb_at(131_072)) - Rtx12gbLoad::floor_budget(rtx_12gb_at(114_688));
        assert_eq!(step, 160_433_152, "256 pages and their block tables");
    }

    /// GitHub #639, the first real 12 GB card: an RTX 3060 12 GB under Linux
    /// with nothing else on it read 12,486,508,544 bytes free at the start
    /// (11,908 MiB) and a CUDA context of 114,425,856 bytes, the RTX 5090's
    /// less 337,735,680; its every other line was the RTX 5090's on the routes
    /// an RTX 30 or 40 card runs. The plan's arithmetic on those lines is the
    /// plan the card logged, byte for byte: 2,250 pages, 144,000 tokens,
    /// 131,072 kept, nothing adapted, where at the RTX 5090's context the same
    /// free memory keeps 98,304. At that context 131,072 tokens need 11,790
    /// MiB free, not 12,112, and each row of the guide's table keeps more: the
    /// first two 131,072, then 114,688, 81,920, 65,536, and 32,768 where the
    /// RTX 5090's context is refused.
    #[test]
    fn an_rtx_3060s_own_cuda_context_gives_the_plan_it_logged() {
        const HEADROOM: u64 = 512 << 20;
        let card = Rtx12gbLoad {
            budget: 12_486_508_544 - HEADROOM,
            trellis_dense: TRELLIS_DENSE,
            cuda_context: CUDA_CONTEXT_3060,
        };
        assert_eq!(card.budget, 11_949_637_632, "the budget the card logged");
        let (result, attempts) = card.adapt(RTX_12GB_LADDER);
        let adapted = kept(result);
        assert_eq!((adapted.attempt, adapted.steps.as_slice()), (rtx_12gb_at(131_072), &[][..]));
        assert_eq!(attempts, [rtx_12gb_at(131_072)], "one plan");
        let plan = adapted.plan;
        assert_eq!((plan.kv_page_count, plan.kv_token_capacity()), (2_250, 144_000));
        assert_eq!((plan.kv_pool_bytes, plan.total_bytes), (1_410_056_192, 11_949_021_440));
        let at_the_5090s = Rtx12gbLoad { cuda_context: CUDA_CONTEXT_5090, ..card };
        assert_eq!(kept(at_the_5090s.adapt(RTX_12GB_LADDER).0).attempt, rtx_12gb_at(98_304));

        let context = |free_mib: u64| {
            let card = Rtx12gbLoad { budget: (free_mib << 20) - HEADROOM, ..card };
            card.adapt(RTX_12GB_LADDER).0.ok().map(|adapted| adapted.attempt.max_context)
        };
        assert_eq!(context(11_790), Some(131_072));
        assert_eq!(context(11_789), Some(114_688), "11,790 less 1 MiB");
        let rows = [
            (12_032, 131_072),
            (11_878, 131_072),
            (11_704, 114_688),
            (11_468, 81_920),
            (11_233, 65_536),
            (10_946, 32_768),
        ];
        for (free_mib, max_context) in rows {
            assert_eq!(context(free_mib), Some(max_context), "{free_mib} MiB free at the RTX 3060's context");
        }
    }

    /// Decision 6: a context the plan holds at two lanes without a spare
    /// reservation is passed over; the lanes drop to one and the ladder runs
    /// again from the profile's value, the one-lane criterion being the
    /// plan's own floor.
    #[test]
    fn lanes_drop_to_one_then_the_ladder_runs_again_from_the_profiles_value() {
        let start = values(262_144, 2);
        // A pool one page short of 65,536 + 65,536 at two lanes: no context
        // there keeps a spare reservation. A lane's bytes back hold no more
        // than 65,536 at one lane: 65,536 tokens more of context take more
        // workspace than a lane frees.
        let card = GB16.with_budget(GB16.budget_holding(values(65_536, 2), 131_072) - 1);
        let (result, attempts) = card.adapt(start, Adaptable::ALL);
        let adapted = kept(result);
        assert_eq!(adapted.attempt, values(65_536, 1));
        assert_eq!(
            adapted.steps,
            [
                AdaptStep::DecodeLanes { from: 2, to: 1 },
                AdaptStep::MaxContext { from: 262_144, to: 65_536, context_step: 65_536, room: Room::OneLane },
            ]
        );
        assert!(card.plan(values(131_072, 1)).is_err(), "65,536 is the largest one lane holds");
        let two_lane_contexts: Vec<u32> = attempts.iter().filter(|a| a.decode_lanes == 2).map(|a| a.max_context).collect();
        assert_eq!(two_lane_contexts, [262_144, 196_608, 131_072, 65_536], "every two-lane context first");
        assert_eq!(attempts.iter().find(|a| a.decode_lanes == 1).unwrap().max_context, 262_144, "then from the top");

        // A costlier lane, whose bytes hold 196,608 at one lane where no
        // context fits two: both adaptations, the lanes' first.
        let card = Card { lane: 2_000_000_000, ..GB16 };
        let card = card.with_budget(card.budget_holding(values(196_608, 1), 196_608 + 2 * 64));
        let adapted = kept(card.adapt(start, Adaptable::ALL).0);
        assert_eq!(adapted.attempt, values(196_608, 1));
        assert_eq!(
            steps_text(&adapted.steps),
            "decode_lanes 2 -> 1, max_context 262144 -> 196608 (adapt_context_step 65536, one lane)"
        );
    }

    /// Step 1 (decision 2): a profile's `shortlist` falls to `full` first,
    /// and a plan that fits then adapts nothing else.
    #[test]
    fn a_profiles_shortlist_head_falls_to_full_before_anything_else() {
        let start = Attempt { draft_head: Some(ProposalHead::Shortlist), ..values(262_144, 2) };
        let card = GB16.with_budget(GB16.budget_holding(values(262_144, 2), 262_144 + 4 * 64));
        assert!(card.plan(start).is_err(), "the shortlist's bytes do not fit");
        let (result, attempts) = card.adapt(start, Adaptable::ALL);
        let adapted = kept(result);
        assert_eq!(adapted.attempt, values(262_144, 2));
        assert_eq!(adapted.steps, [AdaptStep::DraftHead { from: ProposalHead::Shortlist, to: ProposalHead::Full }]);
        assert_eq!(steps_text(&adapted.steps), "draft_head shortlist -> full");
        assert_eq!(attempts, [start, values(262_144, 2)]);
    }

    /// An explicit value never adapts: each of the three, alone, keeps the
    /// value it was given in every plan the ladder makes.
    #[test]
    fn an_explicit_value_never_adapts() {
        let start = Attempt { draft_head: Some(ProposalHead::Shortlist), ..values(262_144, 2) };
        let card = GB16.with_budget(GB16.budget_holding(values(65_536, 1), 65_536 + 2 * 64));

        let (_, attempts) = card.adapt(start, Adaptable { draft_head: false, ..Adaptable::ALL });
        assert!(attempts.iter().all(|a| a.draft_head == Some(ProposalHead::Shortlist)), "{attempts:?}");

        let (result, attempts) = card.adapt(start, Adaptable { max_context: false, ..Adaptable::ALL });
        assert!(attempts.iter().all(|a| a.max_context == 262_144), "{attempts:?}");
        assert!(matches!(result, Err(AdaptError::Refused(_))), "262,144 fits on no lane count here");

        let (result, attempts) = card.adapt(start, Adaptable { decode_lanes: false, ..Adaptable::ALL });
        assert!(attempts.iter().all(|a| a.decode_lanes == 2), "{attempts:?}");
        assert!(matches!(result, Err(AdaptError::Refused(_))), "65,536 at one lane is all this card holds");
    }

    /// Lanes that cannot drop leave the largest context the plan holds --
    /// here 851,968, well above one step, since no context leaves room for a
    /// spare reservation larger than the pool -- and the log says the spare
    /// reservation is not there.
    #[test]
    fn explicit_lanes_keep_the_largest_context_the_plan_holds_and_say_so() {
        let start = values(917_504, 8);
        let card = RTX5090.with_budget(RTX5090.budget_holding(start, 917_504 + 16 * 64) - 100_000_000);
        let ladder = Ladder { context_step: 65_536, spare_tokens: 2_000_000 };
        let (result, attempts) = card.adapt_with(start, Adaptable { decode_lanes: false, ..Adaptable::ALL }, ladder);
        let adapted = kept(result);
        assert_eq!(adapted.attempt, values(851_968, 8));
        assert_eq!(
            adapted.steps,
            [AdaptStep::MaxContext { from: 917_504, to: 851_968, context_step: 65_536, room: Room::Short(2_000_000) }]
        );
        assert_eq!(
            steps_text(&adapted.steps),
            "max_context 917504 -> 851968 (adapt_context_step 65536, one sequence only: \
             no room for adapt_spare_tokens 2000000, and decode_lanes is explicit)"
        );
        assert!(attempts.iter().all(|a| a.decode_lanes == 8));
        assert_eq!(attempts.len(), 14, "every context, looking for the spare reservation");
    }

    /// At one lane already, the plan's floor is the criterion: the largest
    /// context it holds is kept, nothing unmet.
    #[test]
    fn at_one_lane_the_largest_context_the_plan_holds_is_kept() {
        let card = GB16.with_budget(GB16.budget_holding(values(131_072, 1), 131_072 + 2 * 64));
        let adapted = kept(card.adapt(values(262_144, 1), Adaptable::ALL).0);
        assert_eq!(adapted.attempt, values(131_072, 1));
        assert_eq!(
            adapted.steps,
            [AdaptStep::MaxContext { from: 262_144, to: 131_072, context_step: 65_536, room: Room::OneLane }]
        );
    }

    /// Decision 7: when not even 65,536 at one lane fits, the start is
    /// refused with that last attempt's deficit and every adaptation tried.
    #[test]
    fn a_budget_below_the_last_step_is_refused_with_its_deficit_and_the_steps_tried() {
        let start = Attempt { draft_head: Some(ProposalHead::Shortlist), ..values(262_144, 2) };
        let card = GB16.with_budget(GB16.budget_holding(values(65_536, 1), 65_536 + 2 * 64) - 1);
        let refused = refused(card.adapt(start, Adaptable::ALL).0);
        let last = card.plan(values(65_536, 1)).expect_err("one byte short");
        assert_eq!(refused.error, last, "the deficit of the last, smallest attempt");
        assert!(matches!(refused.error, VramPlanError::BelowMinimum { max_context_tokens: 65_536, .. }));
        let message = refused.to_string();
        assert!(message.starts_with(&last.to_string()), "{message}");
        assert!(
            message.ends_with(
                "; adaptation tried, in order: draft_head shortlist -> full, \
                 max_context 262144 -> 196608 -> 131072 -> 65536, decode_lanes 2 -> 1, \
                 max_context 262144 -> 196608 -> 131072 -> 65536"
            ),
            "{message}"
        );
    }

    /// Nothing adaptable: the refusal is the plan's own, word for word, as
    /// before adaptation existed.
    #[test]
    fn with_nothing_adaptable_the_refusal_is_the_plans_own() {
        let start = values(262_144, 2);
        let card = GB16.with_budget(GB16.budget_holding(start, 262_144));
        let (result, attempts) = card.adapt(start, Adaptable::NONE);
        let refused = refused(result);
        assert_eq!(attempts, [start]);
        assert!(refused.tried.is_empty());
        assert_eq!(refused.to_string(), card.plan(start).unwrap_err().to_string());
    }

    /// Decision 5: only a plan refused below its minimum adapts -- a budget
    /// above free memory is refused at once, nothing tried.
    #[test]
    fn a_refusal_other_than_the_minimum_is_not_adapted() {
        let start = values(262_144, 2);
        let mut attempts = 0;
        let result = adapt(start, Adaptable::ALL, DEFAULT, |_| {
            attempts += 1;
            Ok::<Planned<()>, Infallible>(Planned::Unfit(VramPlanError::BudgetAboveFree { budget_bytes: 2, free_bytes: 1 }))
        });
        let Err(AdaptError::Refused(refused)) = result else { panic!("refused") };
        assert_eq!((attempts, refused.tried.len()), (1, 0));
        assert!(matches!(refused.error, VramPlanError::BudgetAboveFree { .. }));
    }

    /// The closure's own failure ends the ladder where it happened.
    #[test]
    fn a_planning_failure_ends_the_ladder() {
        let result = adapt(values(262_144, 2), Adaptable::ALL, DEFAULT, |attempt| {
            if attempt.max_context < 262_144 {
                return Err("the leaf cannot plan that");
            }
            Ok::<Planned<()>, _>(Planned::Unfit(VramPlanError::BelowMinimum {
                mode: VramMode::Derived { headroom_bytes: 0 },
                policy: ignis_core::KvPoolPolicy::Resident,
                budget_bytes: 1,
                needed_bytes: 2,
                max_context_tokens: attempt.max_context,
                kv_pool_named: false,
                embedding_pool_named: false,
            }))
        });
        assert!(matches!(result, Err(AdaptError::Failed("the leaf cannot plan that"))));
    }
}
