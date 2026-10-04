//! Expert residency's reservations as **plan lines** computed at load (ADR
//! 0030): the host plan (the pinned expert pool, the n-gram hot rows, staging
//! buffers) with its refusal margin, and the VRAM lines — the expert cache,
//! split into eight K-class pools, the prefill staging ring and residency's
//! tables.
//!
//! The VRAM lines compose with [`crate::vram::plan_vram`] rather than
//! extending its line set: the Flash-Next load names its KV pool
//! (`kv_pool_bytes`), so the plan leaves `budget − total` free, and the
//! expert cache takes that rest after the staging ring and the tables. Pure
//! arithmetic over numbers the loader measured or read from the artifact;
//! every refusal is pinned on the CPU.

use super::class::{ExpertCatalog, KClass, Projection, ProjectionId};

const GIB: u64 = 1024 * 1024 * 1024;

/// What the host plan must leave of the available physical memory (spec
/// flash-next/03: "6 GB", taken as GiB): below it, Windows starts paging.
pub const HOST_MARGIN_BYTES: u64 = 6 * GIB;

/// The smallest VRAM expert cache a load accepts (spec flash-next/03: "12
/// GB", taken as GiB). Below it the hit rate, and with it decode speed,
/// collapses.
pub const EXPERT_CACHE_FLOOR_BYTES: u64 = 12 * GIB;

/// What the host plan is made from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostPlanRequest {
    /// Physical memory available at load (`MEMORYSTATUSEX::ullAvailPhys`),
    /// measured before the first pinned allocation.
    pub available_physical_bytes: u64,
    /// Every expert projection, pinned for the life of the load: the
    /// [`ExpertCatalog::total_bytes`] of the artifact.
    pub expert_pool_bytes: u64,
    /// The n-gram table's hot rows held in RAM (spec flash-next/04).
    pub ngram_hot_rows_bytes: u64,
    /// Host staging buffers: the artifact read buffers and the like.
    pub staging_bytes: u64,
}

/// The host plan the load carries out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostPlan {
    pub request: HostPlanRequest,
    pub total_bytes: u64,
    /// What stays available after the plan: at least [`HOST_MARGIN_BYTES`].
    pub left_bytes: u64,
}

impl HostPlan {
    /// `(name, bytes)` in plan order.
    pub fn entries(&self) -> [(&'static str, u64); 3] {
        host_lines(&self.request)
    }
}

fn host_lines(request: &HostPlanRequest) -> [(&'static str, u64); 3] {
    [
        ("expert_pool", request.expert_pool_bytes),
        ("ngram_hot_rows", request.ngram_hot_rows_bytes),
        ("staging", request.staging_bytes),
    ]
}

/// A load the host plan refuses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostPlanError {
    /// The plan leaves less than [`HOST_MARGIN_BYTES`] of the available
    /// physical memory.
    BelowMargin {
        available_bytes: u64,
        planned_bytes: u64,
        /// The first line, in plan order, whose running total passes what
        /// the margin leaves usable.
        crossing_line: &'static str,
        lines: [(&'static str, u64); 3],
    },
}

impl std::fmt::Display for HostPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self::BelowMargin {
            available_bytes,
            planned_bytes,
            crossing_line,
            lines,
        } = self;
        write!(
            f,
            "the host plan needs {planned_bytes} bytes ("
        )?;
        for (i, (name, bytes)) in lines.iter().enumerate() {
            write!(f, "{}{name} {bytes}", if i == 0 { "" } else { ", " })?;
        }
        write!(
            f,
            ") of the {available_bytes} bytes of physical memory available, and must leave \
             {HOST_MARGIN_BYTES} (6 GiB) so Windows does not page; the {crossing_line} line \
             crosses that margin: close other applications"
        )?;
        if *crossing_line != "expert_pool" {
            f.write_str(", or load fewer n-gram hot rows")?;
        }
        Ok(())
    }
}

impl std::error::Error for HostPlanError {}

/// Lay the host plan out, or refuse the load.
pub fn plan_host(request: &HostPlanRequest) -> Result<HostPlan, HostPlanError> {
    let lines = host_lines(request);
    let total = lines.iter().fold(0u64, |sum, (_, b)| sum.saturating_add(*b));
    let usable = request
        .available_physical_bytes
        .saturating_sub(HOST_MARGIN_BYTES);
    if total > usable {
        let mut running = 0u64;
        let crossing_line = lines
            .iter()
            .find(|(_, bytes)| {
                running = running.saturating_add(*bytes);
                running > usable
            })
            .map_or(lines[0].0, |(name, _)| *name);
        return Err(HostPlanError::BelowMargin {
            available_bytes: request.available_physical_bytes,
            planned_bytes: total,
            crossing_line,
            lines,
        });
    }
    Ok(HostPlan {
        request: *request,
        total_bytes: total,
        left_bytes: request.available_physical_bytes - total,
    })
}

/// What the expert cache line is made from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertCacheRequest {
    /// The VRAM plan's budget ([`crate::vram::VramPlan::budget_bytes`]).
    pub budget_bytes: u64,
    /// Everything the VRAM plan placed, the named KV pool included
    /// ([`crate::vram::VramPlan::total_bytes`]): non-expert weights, the
    /// CUDA context, workspaces, graphs, lane state, KV.
    pub planned_bytes: u64,
    /// [`prefill_staging_ring_bytes`].
    pub staging_ring_bytes: u64,
    /// [`residency_table_bytes`].
    pub table_bytes: u64,
    /// [`EXPERT_CACHE_FLOOR_BYTES`] for a real load.
    pub floor_bytes: u64,
    /// One slot of each class, indexed like [`KClass::ALL`].
    pub slot_bytes: [u64; KClass::COUNT],
    /// Projections per class ([`ExpertCatalog::class_counts`]): a pool never
    /// holds more slots than its class has projections.
    pub projections: [u64; KClass::COUNT],
    /// Calibration selections per class ([`class_selections`]).
    pub selections: [u64; KClass::COUNT],
    /// [`min_slots_per_class`]: what every populated class gets first.
    pub min_slots: u32,
}

/// One class's pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ClassPool {
    pub slots: u32,
    pub bytes: u64,
}

/// The expert cache line and its pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertCachePlan {
    /// The line: budget − planned − staging ring − tables.
    pub cache_bytes: u64,
    /// Indexed like [`KClass::ALL`]. Their sum is at most `cache_bytes`;
    /// flooring to whole slots leaves under one slot per class unused.
    pub pools: [ClassPool; KClass::COUNT],
    pub staging_ring_bytes: u64,
    pub table_bytes: u64,
}

impl ExpertCachePlan {
    /// Slots per class: the policy model's
    /// [`PolicyConfig::capacity`](super::policy::PolicyConfig::capacity).
    pub fn capacity(&self) -> [u32; KClass::COUNT] {
        std::array::from_fn(|i| self.pools[i].slots)
    }

    /// What the pools hold.
    pub fn pooled_bytes(&self) -> u64 {
        self.pools.iter().map(|p| p.bytes).sum()
    }

    /// `(name, bytes)` of residency's VRAM lines, in plan order, beside the
    /// ones [`crate::vram::VramLines::entries`] prints.
    pub fn entries(&self) -> [(&'static str, u64); 3] {
        [
            ("expert_cache", self.cache_bytes),
            ("prefill_staging", self.staging_ring_bytes),
            ("residency_tables", self.table_bytes),
        ]
    }
}

/// A load the expert cache line refuses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertCachePlanError {
    /// What the plan leaves is below the floor.
    BelowFloor {
        cache_bytes: u64,
        floor_bytes: u64,
        budget_bytes: u64,
        planned_bytes: u64,
        staging_ring_bytes: u64,
        table_bytes: u64,
    },
}

impl std::fmt::Display for ExpertCachePlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self::BelowFloor {
            cache_bytes,
            floor_bytes,
            budget_bytes,
            planned_bytes,
            staging_ring_bytes,
            table_bytes,
        } = *self;
        write!(
            f,
            "the VRAM expert cache would get {cache_bytes} bytes, below its {floor_bytes}-byte \
             floor: the {budget_bytes}-byte VRAM budget holds {planned_bytes} for the weights, \
             workspaces, graphs and KV pool, {staging_ring_bytes} for the prefill staging ring \
             and {table_bytes} for residency's tables; shrink the KV pool (a shorter \
             --max-context or fewer lanes), the prefill chunk, or --vram-headroom-bytes, or \
             close what holds VRAM on the desktop"
        )
    }
}

impl std::error::Error for ExpertCachePlanError {}

/// The expert cache line and its split, or a refusal below the floor.
///
/// Every class with projections first gets `min(min_slots, projections)`
/// slots. What is left is shared by **byte traffic** — calibration
/// selections × slot bytes, so a pool's share of the cache is its class's
/// share of the bytes routing asks for — by water-filling: a class whose
/// share would exceed its projections is capped there and the rest is
/// shared again among the others.
pub fn plan_expert_cache(
    request: &ExpertCacheRequest,
) -> Result<ExpertCachePlan, ExpertCachePlanError> {
    let cache = request
        .budget_bytes
        .saturating_sub(request.planned_bytes)
        .saturating_sub(request.staging_ring_bytes)
        .saturating_sub(request.table_bytes);
    if cache < request.floor_bytes {
        return Err(ExpertCachePlanError::BelowFloor {
            cache_bytes: cache,
            floor_bytes: request.floor_bytes,
            budget_bytes: request.budget_bytes,
            planned_bytes: request.planned_bytes,
            staging_ring_bytes: request.staging_ring_bytes,
            table_bytes: request.table_bytes,
        });
    }

    let mut slots = [0u64; KClass::COUNT];
    let mut left = cache;
    for i in 0..KClass::COUNT {
        let first = u64::from(request.min_slots).min(request.projections[i]);
        let bytes = first * request.slot_bytes[i];
        slots[i] = first;
        left = left.saturating_sub(bytes);
    }

    let room = |i: usize, slots: &[u64; KClass::COUNT]| request.projections[i] - slots[i];
    let weight = |i: usize| u128::from(request.selections[i]) * u128::from(request.slot_bytes[i]);
    let mut active: Vec<usize> = (0..KClass::COUNT)
        .filter(|&i| room(i, &slots) > 0 && weight(i) > 0 && request.slot_bytes[i] > 0)
        .collect();
    while !active.is_empty() {
        let total: u128 = active.iter().map(|&i| weight(i)).sum();
        let share = |i: usize| (u128::from(left) * weight(i) / total) as u64 / request.slot_bytes[i];
        let capped: Vec<usize> = active
            .iter()
            .copied()
            .filter(|&i| share(i) >= room(i, &slots))
            .collect();
        if capped.is_empty() {
            for &i in &active {
                slots[i] += share(i);
            }
            break;
        }
        for &i in &capped {
            let more = room(i, &slots);
            slots[i] += more;
            left -= more * request.slot_bytes[i];
        }
        active.retain(|i| !capped.contains(i));
    }

    let pools = std::array::from_fn(|i| ClassPool {
        slots: slots[i] as u32,
        bytes: slots[i] * request.slot_bytes[i],
    });
    Ok(ExpertCachePlan {
        cache_bytes: cache,
        pools,
        staging_ring_bytes: request.staging_ring_bytes,
        table_bytes: request.table_bytes,
    })
}

/// The slots a class needs so that no decode step can be refused and its
/// prefetches still find room: every lane's selection and every lane's
/// lookahead, as if all of them fell in that one class.
pub fn min_slots_per_class(lanes: u32, top_k: u32, prefetch_width: u32) -> u32 {
    lanes * (top_k + prefetch_width)
}

/// The prefill staging ring: two layers, each as large as the heaviest
/// layer's projections — the most one chunk can touch in a layer — so the
/// layer being computed and the layer being filled never overflow it.
pub fn prefill_staging_ring_bytes(catalog: &ExpertCatalog) -> u64 {
    let heaviest = (0..catalog.layers())
        .map(|layer| catalog.layer_bytes(layer))
        .max()
        .unwrap_or(0);
    2 * heaviest
}

/// One slot-table entry (a device address and the K) per projection, as the
/// expert kernels read it (spec flash-next/02).
pub const SLOT_TABLE_ENTRY_BYTES: u64 = 16;

/// One device LRU entry (a stamp and a key) per slot.
pub const LRU_ENTRY_BYTES: u64 = 16;

/// Residency's device tables: a slot-table entry per projection and an LRU
/// entry per slot, counted for as many slots as there are projections — an
/// upper bound that does not depend on the cache split it is subtracted
/// before.
pub fn residency_table_bytes(layers: u64, experts: u64) -> u64 {
    layers * experts * 2 * (SLOT_TABLE_ENTRY_BYTES + LRU_ENTRY_BYTES)
}

/// Calibration selections per class, from the sidecar's per-expert traffic
/// (`expert_traffic`: selections per (layer, expert), layer-major). A
/// selected expert reads both its projections.
pub fn class_selections(catalog: &ExpertCatalog, counts: &[u64]) -> [u64; KClass::COUNT] {
    let mut out = [0u64; KClass::COUNT];
    for (layer, expert, count) in per_expert(catalog, counts) {
        for projection in Projection::ALL {
            let class = catalog.class_of(ProjectionId::new(layer, expert, projection));
            out[class.index()] += count;
        }
    }
    out
}

/// The warm start's order: both projections of every selected expert,
/// hottest first by calibration selections, ties in canonical key order.
/// Never-selected experts are not warmed.
pub fn warm_start_order(catalog: &ExpertCatalog, counts: &[u64]) -> Vec<ProjectionId> {
    let mut experts: Vec<(u64, u16, u16)> = per_expert(catalog, counts)
        .filter(|&(_, _, count)| count > 0)
        .map(|(layer, expert, count)| (count, layer, expert))
        .collect();
    experts.sort_by(|a, b| b.0.cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2))));
    experts
        .into_iter()
        .flat_map(|(_, layer, expert)| {
            Projection::ALL.map(|projection| ProjectionId::new(layer, expert, projection))
        })
        .collect()
}

fn per_expert<'a>(
    catalog: &ExpertCatalog,
    counts: &'a [u64],
) -> impl Iterator<Item = (u16, u16, u64)> + 'a {
    let experts = catalog.experts();
    let total = usize::from(catalog.layers()) * usize::from(experts);
    counts.iter().take(total).enumerate().map(move |(i, &count)| {
        (
            (i / usize::from(experts)) as u16,
            (i % usize::from(experts)) as u16,
            count,
        )
    })
}
