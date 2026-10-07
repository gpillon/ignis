//! Expert residency's device side (`kernel/include/ignis_residency.h`,
//! `crates/core/src/residency/device.rs`) against the CPU policy model it
//! implements (spec flash-next/03, acceptance 4, GitHub #301).
//!
//! - The leaf's own plan line for the tables stays within the CPU plan's
//!   upper bound (host arithmetic, no GPU: runs with `--features cuda`).
//! - On random traces -- decode of one to three lanes, prefill chunks, a
//!   lookahead with holes, a budget or none, a warm start, refused steps,
//!   forwards restarted at a layer --
//!   every step's outcome on the device equals `ResidencyModel::step`'s, and
//!   the counters agree at the end. Half the steps hand the lookahead over as
//!   the next router's logits, so the device's own ranking (BF16-rounded,
//!   ties to the lower id) is held to the CPU's.
//!
//! The kernel leaf's residency CTest replays the committed fixture with the
//! copy and slot-table checks; this test covers what one fixed trace cannot.
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing GPU is a skip, under it a hard failure (`ignis_core::gpu_profile`).

#![cfg(feature = "cuda")]

use std::os::raw::c_void;

use ignis_artifact::{CudaDevice, Device, DeviceBuffer};
use ignis_core::gpu_profile;
use ignis_core::residency::device::{self, DeviceResidency, NO_BUDGET, ResidencyDesc};
use ignis_core::residency::{
    ExpertCatalog, KBits, KClass, LayerStep, Phase, PolicyConfig, ProjectionId, Projection,
    ResidencyModel, StepError, residency_table_bytes,
};

const TOP_K: usize = 10;

#[test]
fn the_leaf_s_tables_line_stays_within_the_cpu_plan_s_upper_bound() {
    // Flash-Next's geometry, every projection given a slot (the bound's own
    // worst case), an 8192-token chunk and W = 16.
    let record_bytes = [827_392, 1_032_192, 1_236_992, 1_646_592, 417_792, 520_192, 622_592, 827_392];
    let desc = ResidencyDesc {
        layers: 48,
        experts: 512,
        capacity: [6144; 8],
        record_bytes,
        max_tokens: 8192,
        lookahead_width: 16,
        prefill_lookahead_width: 10,
        prefetch_budget_bytes: NO_BUDGET,
        staging_half_bytes: 800_000_000,
        host_pool_bytes: 38_000_000_000,
        copy_blocks: 16,
        report: 0,
    };
    let plan = device::plan_bytes(&desc).expect("plan");
    let bound = residency_table_bytes(48, 512, 8192, 16);
    assert!(plan.tables <= bound, "leaf {} B, CPU bound {bound} B", plan.tables);
    assert!(plan.tables * 10 >= bound * 8, "a loose bound: leaf {} B, CPU {bound} B", plan.tables);
    let round = |b: u64| b.div_ceil(256) * 256;
    let pools: u64 = record_bytes.iter().map(|&b| round(6144 * b)).sum();
    assert_eq!(plan.pools, pools);
    assert_eq!(plan.staging, round(1_600_000_000));
    assert_eq!(plan.total, plan.pools + plan.staging + plan.tables);
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// f32 rounded to BF16 (nearest, ties to even) and back, as the router ranks.
fn bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

/// The router's ranking of one row of logits: BF16-rounded, ties to the
/// lower id, best first.
fn rank(logits: &[f32], width: usize) -> Vec<u16> {
    let mut order: Vec<u16> = (0..logits.len() as u16).collect();
    order.sort_by(|&a, &b| {
        bf16(logits[usize::from(b)])
            .total_cmp(&bf16(logits[usize::from(a)]))
            .then(a.cmp(&b))
    });
    order.truncate(width);
    order
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn ptr(buf: &DeviceBuffer) -> *mut c_void {
    buf.base_ptr() as *mut c_void
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn the_device_steps_equal_the_policy_model_on_random_traces() {
    let mut dev = match CudaDevice::create(0) {
        Ok(d) => d,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("no CUDA device: {e}"));
            return;
        }
    };
    const LAYERS: u16 = 3;
    const EXPERTS: u16 = 32;
    const WIDTH: usize = 3;
    // A chunk's own, narrower width: the device must take it from the same rankings.
    const PREFILL_WIDTH: usize = 2;
    const MAX_ROWS: usize = 8;
    let record_bytes = [4096, 8192, 8192, 12288, 4096, 4096, 8192, 8192];
    let ids_buf = dev.allocate((MAX_ROWS * TOP_K * 4) as u64).expect("ids");
    let look_buf = dev.allocate((MAX_ROWS * usize::from(EXPERTS) * 4) as u64).expect("lookahead");
    let mut steps_run = 0;

    for seed in 0..6u64 {
        let mut rng = Rng(1000 + seed);
        let map = (0..usize::from(LAYERS) * usize::from(EXPERTS))
            .map(|_| (KBits::ALL[rng.below(4) as usize], KBits::ALL[rng.below(4) as usize]))
            .collect();
        let catalog = ExpertCatalog::new(LAYERS, EXPERTS, map, record_bytes).expect("catalog");
        let capacity: [u32; 8] = std::array::from_fn(|_| 6 + rng.below(20) as u32);
        let budget = if seed % 2 == 0 { Some(10_000 + rng.below(30_000)) } else { None };
        let mut model = ResidencyModel::new(
            catalog.clone(),
            PolicyConfig { capacity, prefetch_width: WIDTH, prefill_prefetch_width: PREFILL_WIDTH, prefetch_budget_bytes: budget },
        );
        let (k2, offsets, pool_bytes) = device::packed_layout(&catalog);
        let heaviest = (0..LAYERS).map(|l| catalog.layer_bytes(l)).max().unwrap();
        let desc = ResidencyDesc {
            layers: u32::from(LAYERS),
            experts: u32::from(EXPERTS),
            capacity,
            record_bytes,
            max_tokens: MAX_ROWS as u32,
            lookahead_width: WIDTH as u32,
            prefill_lookahead_width: PREFILL_WIDTH as u32,
            prefetch_budget_bytes: budget.unwrap_or(NO_BUDGET),
            staging_half_bytes: heaviest,
            host_pool_bytes: pool_bytes,
            copy_blocks: 4,
            report: 1,
        };
        let mut gpu = match DeviceResidency::new(&desc, &k2, &offsets) {
            Ok(g) => g,
            Err(e) => {
            gpu_profile::skip_or_fail(&format!("ignis_residency_create: {e}"));
            return;
        }
        };
        for (i, b) in gpu.host_pool_mut().iter_mut().enumerate() {
            *b = (i * 7 + 3) as u8;
        }
        let warm: Vec<ProjectionId> = (0..6u16)
            .flat_map(|e| (0..LAYERS).map(move |l| (l, e)))
            .flat_map(|(l, e)| Projection::ALL.map(|p| ProjectionId::new(l, e, p)))
            .collect();
        assert_eq!(gpu.warm_start(&warm).expect("warm start"), model.warm_start(&warm), "seed {seed}");
        assert_eq!(gpu.slots_in_use().expect("occupancy"), model.occupancy(), "seed {seed}: warm occupancy");

        for round in 0..30 {
            let prefill = round % 4 == 3;
            let rows = if prefill { 4 + rng.below(5) as usize } else { 1 + rng.below(3) as usize };
            // Now and then a forward restarted at a layer: the same layer twice in a row.
            let mut order = Vec::new();
            for layer in 0..LAYERS {
                order.push(layer);
                if rng.below(8) == 0 {
                    order.push(layer);
                }
            }
            for layer in order {
                let mut ids = Vec::with_capacity(rows * TOP_K);
                for _ in 0..rows {
                    let mut row: Vec<u16> = Vec::new();
                    while row.len() < TOP_K {
                        let e = if rng.below(3) == 0 { rng.below(u64::from(EXPERTS)) } else { rng.below(12) } as u16;
                        if !row.contains(&e) {
                            row.push(e);
                        }
                    }
                    ids.extend(row.iter().map(|&e| i32::from(e)));
                }
                dev.copy_h2d(&ids_buf, 0, as_bytes(&ids)).expect("ids upload");
                let selected: Vec<u16> = ids.iter().map(|&e| e as u16).collect();
                let last = layer + 1 == LAYERS;
                let by_logits = rng.below(2) == 0;
                // The CPU model's lanes, and what the device is given.
                let lanes: Vec<Vec<u16>> = if last {
                    Vec::new()
                } else if by_logits {
                    // A few distinct values, so BF16 ties are common.
                    let logits: Vec<f32> = (0..rows * usize::from(EXPERTS))
                        .map(|_| (rng.below(9) as f32 - 4.0) * 0.375 + (rng.below(3) as f32) * 1e-4)
                        .collect();
                    dev.copy_h2d(&look_buf, 0, as_bytes(&logits)).expect("logits upload");
                    logits.chunks(usize::from(EXPERTS)).map(|row| rank(row, WIDTH)).collect()
                } else {
                    let stride = WIDTH + 1;
                    let mut raw = vec![-1i32; rows * stride];
                    let mut lanes = Vec::new();
                    for r in 0..rows {
                        let mut lane = Vec::new();
                        for j in 0..stride {
                            if rng.below(6) == 0 {
                                continue; // a hole
                            }
                            let e = rng.below(u64::from(EXPERTS)) as u16;
                            raw[r * stride + j] = i32::from(e);
                            lane.push(e);
                        }
                        lanes.push(lane);
                    }
                    dev.copy_h2d(&look_buf, 0, as_bytes(&raw)).expect("lookahead upload");
                    lanes
                };
                dev.synchronize().expect("uploads");
                let phase = if prefill { Phase::Prefill } else { Phase::Decode };
                let result = unsafe {
                    if last {
                        gpu.step(u32::from(layer), phase, ptr(&ids_buf) as *const i32, rows as u32, std::ptr::null(), std::ptr::null_mut())
                    } else if by_logits {
                        gpu.step(
                            u32::from(layer),
                            phase,
                            ptr(&ids_buf) as *const i32,
                            rows as u32,
                            ptr(&look_buf) as *const f32,
                            std::ptr::null_mut(),
                        )
                    } else {
                        gpu.step_ranked(
                            u32::from(layer),
                            phase,
                            ptr(&ids_buf) as *const i32,
                            rows as u32,
                            ptr(&look_buf) as *const i32,
                            rows as u32,
                            (WIDTH + 1) as u32,
                            std::ptr::null_mut(),
                        )
                    }
                };
                if let Err(e) = result {
                    gpu_profile::skip_or_fail(&format!("ignis_residency_step: {e}"));
                    return;
                }
                let report = gpu.last_report(u32::from(layer)).expect("report");
                let lane_refs: Vec<&[u16]> = lanes.iter().map(Vec::as_slice).collect();
                let step = if prefill { LayerStep::prefill(layer, &selected) } else { LayerStep::decode(layer, &selected) };
                let at = format!("seed {seed} round {round} layer {layer}");
                match model.step(&step.lookahead(&lane_refs)) {
                    Ok(o) => {
                        assert_eq!(report.status, 0, "{at}");
                        assert_eq!(report.hits, o.hits, "{at}: hits");
                        assert_eq!(report.prefetch_hits, o.prefetch_hits, "{at}: prefetch hits");
                        assert_eq!(report.misses, o.misses, "{at}: misses");
                        assert_eq!(report.evictions, o.evictions, "{at}: evictions");
                        assert_eq!(report.prefetches, o.prefetches, "{at}: prefetches");
                        assert_eq!(report.prefetch_dropped, o.prefetch_dropped, "{at}: dropped");
                        assert_eq!(report.bytes_moved, o.bytes_moved, "{at}: bytes");
                    }
                    Err(StepError::NoEvictableSlot { class, .. }) => {
                        assert_eq!(report.status, 1 + class.index() as u32, "{at}: refused");
                    }
                    Err(e) => panic!("{at}: {e}"),
                }
                assert_eq!(gpu.slots_in_use().expect("occupancy"), model.occupancy(), "{at}: occupancy");
                steps_run += 1;
            }
        }
        let mut counters = gpu.counters().expect("counters");
        // The device times its demand copies; the model has no clock. A phase
        // that copied a miss in waited for it, one that copied none did not.
        for phase in [Phase::Decode, Phase::Prefill] {
            let copied = KClass::ALL.iter().any(|c| model.counters().misses[c.index()][phase.index()] > 0);
            assert_eq!(counters.stall_nanos[phase.index()] > 0, copied, "seed {seed}: {phase:?} stall");
        }
        counters.stall_nanos = [0, 0];
        assert_eq!(&counters, model.counters(), "seed {seed}: counters");
        let _ = KClass::ALL;
    }
    println!("{steps_run} steps equal on the device and in the policy model");
}
