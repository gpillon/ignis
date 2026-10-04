//! The **policy model**: expert residency's replacement policy as a pure,
//! trace-driven CPU model. Fed one layer step at a time — which experts the
//! router selected, and optionally the next layer's lookahead ranking — it
//! decides, exactly and deterministically, what hits, what misses, what is
//! evicted and what is prefetched. The GPU implementation is tested against
//! it: on the same trace, its hit/miss sequence must equal this model's
//! (spec flash-next/03, acceptance 4).
//!
//! # The contract, step by step
//!
//! A **layer step** is one layer's MoE for one decode round or one prefill
//! chunk. The model's **logical clock** advances by one per step, and every
//! projection the step touches is stamped with it. A step:
//!
//! 1. drops whatever the staging ring held for any layer but this one;
//! 2. takes the union of its selected experts, both projections of each;
//! 3. **hits** every selected projection resident in a class pool (stamping
//!    it, which is the LRU refresh) or staged for this layer by the previous
//!    step's prefill lookahead;
//! 4. admits every other selected projection — a **miss** — in canonical key
//!    order ([`ProjectionId`]'s derived order):
//!    - *decode*: a free slot of its class, else the class's **victim**;
//!    - *prefill* (scan-resistant, owner 2026-10-04): a free slot of its
//!      class, else the **staging ring**. A staged projection is never an
//!      entry of the pool, so a prefill evicts nothing;
//! 5. with a lookahead, takes the first `prefetch_width` experts of each
//!    lane's ranking for the next layer, both projections of each — the
//!    **candidates**, in **rank order**: every lane's first expert, then
//!    every lane's second, …, gate/up before down, a repeat kept where it
//!    first appears — skips what is resident or staged, and admits the rest
//!    in that order the same way, except that a decode prefetch is
//!    **dropped**, not an error, when no victim is available, when step 4
//!    had to evict it, or once the step's prefetches would pass
//!    `prefetch_budget_bytes` (decode only: a prefill streams its lookahead
//!    whole);
//! 6. releases this layer's staged projections: the expert kernel has run.
//!
//! **Victim:** of a class's resident projections not stamped by the current
//! step, the one with the smallest `(stamp, key)` that is not **protected**;
//! a candidate already resident at the start of the step is protected, so a
//! step does not evict what it predicts the next layer reads. Only a miss,
//! and only when nothing unprotected is left, takes the smallest protected
//! one; a prefetch never does. Everything the current step stamped — its
//! hits, its misses, its prefetches — is **pinned** until the step ends, so a
//! step never evicts what its own kernel reads. That makes a step's victims
//! order-independent: the `k` victims of a class are its first `k` unpinned
//! projections ordered by `(protected, stamp, key)`, however the misses are
//! walked.
//!
//! A decode step whose class cannot hold its selection — more misses than
//! free plus unpinned slots — is refused whole with
//! [`StepError::NoEvictableSlot`], before anything changes. The plan sizes
//! every pool so that this cannot happen (see `plan`).
//!
//! **A prefetched projection is an ordinary entry** stamped at its arrival
//! (the step that issued it). Its first use is a hit and counts as
//! *prefetch used*; a wrong prefetch costs one eviction and nothing more.
//!
//! **Prefetch width is in experts**, each bringing both projections: the
//! router ranks experts, and the study's recall figures (62/77/81% at
//! 10/16/20) are per expert.
//!
//! **Warm start** ([`ResidencyModel::warm_start`]) fills each pool from a
//! hottest-first list before the first step; the hottest gets the most
//! recent stamp, so the least hot is evicted first.

use std::collections::BTreeSet;

use super::class::{ExpertCatalog, KClass, Projection, ProjectionId};
use super::metrics::ResidencyCounters;

/// The router's lookahead width the spec sets by default (a load option).
pub const DEFAULT_PREFETCH_WIDTH: usize = 16;

/// What kind of step a layer step is: one decode round, or one prefill chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    Decode,
    Prefill,
}

impl Phase {
    pub const ALL: [Phase; 2] = [Phase::Decode, Phase::Prefill];

    pub fn index(self) -> usize {
        match self {
            Phase::Decode => 0,
            Phase::Prefill => 1,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Decode => "decode",
            Phase::Prefill => "prefill",
        }
    }
}

/// Where a copied projection lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Admission {
    /// A slot of its class's pool: a cache entry from now on.
    Slot,
    /// The prefill staging ring: read by this (or the next) layer's kernel
    /// and released, never a cache entry.
    Staging,
}

/// The pools' shape: slots per class and the lookahead width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyConfig {
    /// Slots per class, indexed like [`KClass::ALL`].
    pub capacity: [u32; KClass::COUNT],
    /// Experts taken from the top of each lane's lookahead ranking.
    pub prefetch_width: usize,
    /// The most a decode step may prefetch, in bytes; `None`: no limit. The
    /// link is shared with the step's own misses, so a prefetch only pays
    /// while it fits beside the step's compute (the study's window: one
    /// layer's compute time at the link's bandwidth).
    pub prefetch_budget_bytes: Option<u64>,
}

/// One layer step of a routing trace.
#[derive(Debug, Clone, Copy)]
pub struct LayerStep<'a> {
    pub layer: u16,
    pub phase: Phase,
    /// The experts the router selected for this layer, over every token of
    /// the step (lanes or chunk); repeats are one selection.
    pub selected: &'a [u16],
    /// Per lane (decode) or per token (prefill), the next layer's router
    /// applied to this layer's MoE input, ranked best first. Empty: no
    /// prefetch.
    pub lookahead: &'a [&'a [u16]],
}

impl<'a> LayerStep<'a> {
    pub fn decode(layer: u16, selected: &'a [u16]) -> Self {
        Self {
            layer,
            phase: Phase::Decode,
            selected,
            lookahead: &[],
        }
    }

    pub fn prefill(layer: u16, selected: &'a [u16]) -> Self {
        Self {
            layer,
            phase: Phase::Prefill,
            selected,
            lookahead: &[],
        }
    }

    pub fn lookahead(self, lookahead: &'a [&'a [u16]]) -> Self {
        Self { lookahead, ..self }
    }
}

/// What one step did. Every list is in canonical key order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StepOutcome {
    /// Selected and already resident (or staged for this layer): no copy.
    pub hits: Vec<ProjectionId>,
    /// Of [`StepOutcome::hits`], the first uses of a prefetch.
    pub prefetch_hits: Vec<ProjectionId>,
    /// Selected and copied in by this step.
    pub misses: Vec<(ProjectionId, Admission)>,
    /// Given up to make room, by misses and prefetches alike.
    pub evictions: Vec<ProjectionId>,
    /// Copied ahead for the next layer.
    pub prefetches: Vec<(ProjectionId, Admission)>,
    /// Lookahead candidates that found no slot (decode only).
    pub prefetch_dropped: Vec<ProjectionId>,
    /// Bytes this step copied host-to-device: its misses and prefetches.
    pub bytes_moved: u64,
}

/// A step the model refuses, before changing anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepError {
    /// A layer or an expert the catalog does not have.
    OutOfCatalog { layer: u16, expert: u16 },
    /// A decode step's misses in `class` outnumber its free and unpinned
    /// slots: the pool is smaller than one step's selection.
    NoEvictableSlot { class: KClass, layer: u16 },
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::OutOfCatalog { layer, expert } => {
                write!(f, "expert {expert} of layer {layer} is not in the catalog")
            }
            Self::NoEvictableSlot { class, layer } => write!(
                f,
                "layer {layer}'s selection needs more {} slots than are free or unpinned",
                class.as_str()
            ),
        }
    }
}

impl std::error::Error for StepError {}

#[derive(Debug, Clone, Copy)]
struct Entry {
    stamp: u64,
    /// Admitted by a prefetch and not used since.
    prefetched: bool,
}

/// Every projection's entry, if resident, indexed densely by (layer,
/// expert, plane): the model's hottest lookup.
#[derive(Debug, Clone)]
struct Resident {
    experts: usize,
    entries: Vec<Option<Entry>>,
}

impl Resident {
    fn new(catalog: &ExpertCatalog) -> Self {
        let experts = usize::from(catalog.experts());
        Self {
            experts,
            entries: vec![None; usize::from(catalog.layers()) * experts * 2],
        }
    }

    fn at(&self, id: &ProjectionId) -> usize {
        (usize::from(id.layer) * self.experts + usize::from(id.expert)) * 2
            + usize::from(id.projection == Projection::Down)
    }

    fn contains_key(&self, id: &ProjectionId) -> bool {
        self.entries[self.at(id)].is_some()
    }

    fn get(&self, id: &ProjectionId) -> Option<&Entry> {
        self.entries[self.at(id)].as_ref()
    }

    fn insert(&mut self, id: ProjectionId, entry: Entry) {
        let at = self.at(&id);
        self.entries[at] = Some(entry);
    }

    fn remove(&mut self, id: &ProjectionId) {
        let at = self.at(id);
        self.entries[at] = None;
    }
}

/// The policy model. See the module documentation for its contract.
#[derive(Debug, Clone)]
pub struct ResidencyModel {
    catalog: ExpertCatalog,
    config: PolicyConfig,
    clock: u64,
    resident: Resident,
    /// Per class, its resident projections by `(stamp, key)`: the first
    /// unpinned one is the victim.
    lru: [BTreeSet<(u64, ProjectionId)>; KClass::COUNT],
    /// The staging ring's contents: this layer's and the next one's.
    staged: BTreeSet<ProjectionId>,
    counters: ResidencyCounters,
}

impl ResidencyModel {
    /// A cold model: every pool empty.
    pub fn new(catalog: ExpertCatalog, config: PolicyConfig) -> Self {
        Self {
            resident: Resident::new(&catalog),
            catalog,
            config,
            clock: 0,
            lru: Default::default(),
            staged: BTreeSet::new(),
            counters: ResidencyCounters::default(),
        }
    }

    pub fn catalog(&self) -> &ExpertCatalog {
        &self.catalog
    }

    pub fn config(&self) -> &PolicyConfig {
        &self.config
    }

    /// Resident projections per class, indexed like [`KClass::ALL`].
    pub fn occupancy(&self) -> [u32; KClass::COUNT] {
        std::array::from_fn(|i| self.lru[i].len() as u32)
    }

    /// Whether a projection is in its class's pool now.
    pub fn is_resident(&self, id: ProjectionId) -> bool {
        self.catalog.contains(id) && self.resident.contains_key(&id)
    }

    /// Everything counted since the model was made.
    pub fn counters(&self) -> &ResidencyCounters {
        &self.counters
    }

    /// Fill each pool from `hottest_first`, up to its capacity, skipping a
    /// projection whose class is full. The hottest admitted projection gets
    /// the most recent stamp. Returns how many were admitted. Meant for a
    /// cold model, before the first step; nothing it admits counts as moved.
    pub fn warm_start(&mut self, hottest_first: &[ProjectionId]) -> usize {
        let mut admitted = Vec::new();
        let mut room: [u32; KClass::COUNT] =
            std::array::from_fn(|i| self.config.capacity[i].saturating_sub(self.lru[i].len() as u32));
        let mut seen = BTreeSet::new();
        for &id in hottest_first {
            if !self.catalog.contains(id) || self.resident.contains_key(&id) || !seen.insert(id) {
                continue;
            }
            let class = self.catalog.class_of(id).index();
            if room[class] > 0 {
                room[class] -= 1;
                admitted.push(id);
            }
        }
        for &id in admitted.iter().rev() {
            self.clock += 1;
            self.insert(id, self.clock, false);
        }
        admitted.len()
    }

    /// Run one layer step. See the module documentation for what it does.
    pub fn step(&mut self, step: &LayerStep<'_>) -> Result<StepOutcome, StepError> {
        let layer = step.layer;
        let check = |expert: u16| {
            if layer < self.catalog.layers() && expert < self.catalog.experts() {
                Ok(())
            } else {
                Err(StepError::OutOfCatalog { layer, expert })
            }
        };
        if step.selected.is_empty() {
            check(0)?;
        }
        for &expert in step.selected {
            check(expert)?;
        }
        let next_layer = (layer + 1 < self.catalog.layers()).then_some(layer + 1);
        if next_layer.is_some() {
            for lane in step.lookahead {
                for &expert in lane.iter().take(self.config.prefetch_width) {
                    check(expert)?;
                }
            }
        }

        let selected = projections_of(layer, step.selected.iter().copied());
        let candidates = match next_layer {
            Some(next) => ranked_candidates(next, step.lookahead, self.config.prefetch_width),
            None => Vec::new(),
        };
        // What the next layer is predicted to read and already holds: this
        // step's misses take it only when nothing else can go.
        let protected: BTreeSet<ProjectionId> = candidates
            .iter()
            .copied()
            .filter(|id| self.resident.contains_key(id))
            .collect();
        let (mut hits, mut misses) = (Vec::new(), Vec::new());
        for &id in &selected {
            if self.resident.contains_key(&id) || self.staged.contains(&id) {
                hits.push(id);
            } else {
                misses.push(id);
            }
        }
        if step.phase == Phase::Decode {
            self.check_room(layer, &hits, &misses)?;
        }

        // From here the step is committed.
        self.staged.retain(|id| id.layer == layer);
        self.clock += 1;
        let now = self.clock;
        let phase = step.phase.index();
        let mut out = StepOutcome::default();

        for &id in &hits {
            let class = self.catalog.class_of(id).index();
            self.counters.hits[class] += 1;
            if let Some(entry) = self.resident.get(&id).copied() {
                if entry.prefetched {
                    out.prefetch_hits.push(id);
                }
                self.lru[class].remove(&(entry.stamp, id));
                self.insert(id, now, false);
            } else {
                // Staged for this layer: only a prefill lookahead stages.
                out.prefetch_hits.push(id);
            }
        }
        self.counters.prefetch_used += out.prefetch_hits.len() as u64;
        out.hits = hits;

        for id in misses {
            let class = self.catalog.class_of(id).index();
            let admission = match step.phase {
                Phase::Decode => {
                    let room = self.make_room(class, now, &protected, true, &mut out.evictions);
                    debug_assert!(room, "check_room promised a slot");
                    self.insert(id, now, false);
                    Admission::Slot
                }
                Phase::Prefill => self.admit_without_evicting(id, now, false),
            };
            let bytes = self.catalog.bytes(id);
            self.counters.misses[class][phase] += 1;
            self.counters.bytes_moved[phase] += bytes;
            out.bytes_moved += bytes;
            out.misses.push((id, admission));
        }

        let budget = match step.phase {
            Phase::Decode => self.config.prefetch_budget_bytes,
            Phase::Prefill => None,
        };
        let mut spent = 0u64;
        let mut spent_out = false;
        for id in candidates {
            if self.resident.contains_key(&id) || self.staged.contains(&id) {
                continue;
            }
            // A protected candidate this step's misses had to take is not
            // fetched back by the same step.
            if out.evictions.contains(&id) {
                out.prefetch_dropped.push(id);
                continue;
            }
            let bytes = self.catalog.bytes(id);
            spent_out |= budget.is_some_and(|b| spent + bytes > b);
            if spent_out {
                out.prefetch_dropped.push(id);
                continue;
            }
            let class = self.catalog.class_of(id).index();
            let admission = match step.phase {
                Phase::Decode => {
                    if !self.make_room(class, now, &protected, false, &mut out.evictions) {
                        out.prefetch_dropped.push(id);
                        continue;
                    }
                    self.insert(id, now, true);
                    Admission::Slot
                }
                Phase::Prefill => self.admit_without_evicting(id, now, true),
            };
            spent += bytes;
            self.counters.prefetch_issued += 1;
            self.counters.bytes_moved[phase] += bytes;
            out.bytes_moved += bytes;
            out.prefetches.push((id, admission));
        }

        self.staged.retain(|id| id.layer != layer);
        out.evictions.sort();
        out.prefetches.sort();
        out.prefetch_dropped.sort();
        Ok(out)
    }

    /// Refuse a decode step whose misses some class cannot place.
    fn check_room(
        &self,
        layer: u16,
        hits: &[ProjectionId],
        misses: &[ProjectionId],
    ) -> Result<(), StepError> {
        let mut need = [0u32; KClass::COUNT];
        let mut pinned = [0u32; KClass::COUNT];
        for &id in misses {
            need[self.catalog.class_of(id).index()] += 1;
        }
        for &id in hits {
            if self.resident.contains_key(&id) {
                pinned[self.catalog.class_of(id).index()] += 1;
            }
        }
        for class in KClass::ALL {
            let i = class.index();
            let usable = self.config.capacity[i].saturating_sub(pinned[i]);
            if need[i] > usable {
                return Err(StepError::NoEvictableSlot { class, layer });
            }
        }
        Ok(())
    }

    /// A free slot in `class`, evicting its victim if it has none: the
    /// smallest `(stamp, key)` not pinned by this step and not `protected`,
    /// or — only when `may_take_protected` and nothing else can go — the
    /// smallest protected one. False when no victim may be taken.
    fn make_room(
        &mut self,
        class: usize,
        now: u64,
        protected: &BTreeSet<ProjectionId>,
        may_take_protected: bool,
        evictions: &mut Vec<ProjectionId>,
    ) -> bool {
        if (self.lru[class].len() as u32) < self.config.capacity[class] {
            return true;
        }
        // Pinned entries carry `now`, the largest stamp, so they sit at the
        // end and every walk below stops before them.
        let unpinned = || self.lru[class].iter().take_while(|(stamp, _)| *stamp < now);
        let victim = unpinned()
            .find(|(_, id)| !protected.contains(id))
            .or_else(|| unpinned().next().filter(|_| may_take_protected))
            .copied();
        match victim {
            Some((stamp, victim)) => {
                self.lru[class].remove(&(stamp, victim));
                self.resident.remove(&victim);
                evictions.push(victim);
                true
            }
            None => false,
        }
    }

    /// Prefill admission: a free slot, else the staging ring.
    fn admit_without_evicting(&mut self, id: ProjectionId, now: u64, prefetched: bool) -> Admission {
        let class = self.catalog.class_of(id).index();
        if (self.lru[class].len() as u32) < self.config.capacity[class] {
            self.insert(id, now, prefetched);
            Admission::Slot
        } else {
            self.staged.insert(id);
            Admission::Staging
        }
    }

    fn insert(&mut self, id: ProjectionId, stamp: u64, prefetched: bool) {
        let class = self.catalog.class_of(id).index();
        self.lru[class].insert((stamp, id));
        self.resident.insert(id, Entry { stamp, prefetched });
    }
}

/// Both projections of every listed expert of `layer`, deduplicated, in
/// canonical key order.
fn projections_of(layer: u16, experts: impl Iterator<Item = u16>) -> Vec<ProjectionId> {
    let mut set = BTreeSet::new();
    for expert in experts {
        for projection in Projection::ALL {
            set.insert(ProjectionId::new(layer, expert, projection));
        }
    }
    set.into_iter().collect()
}

/// The lookahead's candidates for `layer` in rank order: every lane's
/// first expert, then every lane's second, up to `width`, both projections
/// of each, a repeat kept where it first appears.
fn ranked_candidates(layer: u16, lanes: &[&[u16]], width: usize) -> Vec<ProjectionId> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for rank in 0..width {
        for lane in lanes {
            let Some(&expert) = lane.get(rank) else { continue };
            for projection in Projection::ALL {
                let id = ProjectionId::new(layer, expert, projection);
                if seen.insert(id) {
                    out.push(id);
                }
            }
        }
    }
    out
}
