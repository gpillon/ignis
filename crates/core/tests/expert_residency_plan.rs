//! Expert residency's plan lines (spec flash-next/03, GitHub #301, ADR 0030):
//! the host plan and its refusal, the VRAM expert cache line and its split
//! into eight K-class pools, the prefill staging ring. Pure arithmetic over
//! numbers the loader has measured or read from the artifact.

use ignis_core::residency::{
    ExpertCachePlanError, ExpertCacheRequest, ExpertCatalog, ExpertTraffic, HOST_MARGIN_BYTES,
    HostPlanError, HostPlanRequest, KBits, Projection, ProjectionId, default_prefetch_budget_bytes,
    min_slots_per_class, plan_expert_cache, plan_host, prefill_staging_ring_bytes,
    residency_table_bytes, warm_start_order,
};

const GIB: u64 = 1024 * 1024 * 1024;

// ---- the host plan -----------------------------------------------------------

fn host(available: u64) -> HostPlanRequest {
    HostPlanRequest {
        available_physical_bytes: available,
        expert_pool_bytes: 38 * GIB,
        ngram_hot_rows_bytes: 2 * GIB,
        staging_bytes: GIB,
        retained_host_slots_bytes: 0,
        kv_ram_arena_bytes: 0,
    }
}

#[test]
fn the_host_plan_adds_its_lines_and_says_what_it_leaves() {
    let plan = plan_host(&host(52 * GIB)).expect("fits");
    let names: Vec<_> = plan.entries().iter().map(|(name, _)| *name).collect();
    assert_eq!(
        names,
        ["expert_pool", "ngram_hot_rows", "staging", "retained_host_slots", "kv_ram_arena"]
    );
    assert_eq!(plan.total_bytes, 41 * GIB);
    assert_eq!(plan.left_bytes, 11 * GIB);
}

#[test]
fn the_host_plan_holds_exactly_at_the_six_gib_margin() {
    assert_eq!(HOST_MARGIN_BYTES, 6 * GIB);
    assert!(plan_host(&host(47 * GIB)).is_ok());
    assert!(plan_host(&host(47 * GIB - 1)).is_err());
}

#[test]
fn a_host_plan_below_the_margin_refuses_naming_the_line_that_crosses_it() {
    // 45 GiB available, 39 usable: the pool fits (38), the hot rows cross.
    let err = plan_host(&host(45 * GIB)).expect_err("below the margin");
    let HostPlanError::BelowMargin {
        available_bytes,
        planned_bytes,
        crossing_line,
        ..
    } = &err;
    assert_eq!(*available_bytes, 45 * GIB);
    assert_eq!(*planned_bytes, 41 * GIB);
    assert_eq!(*crossing_line, "ngram_hot_rows");
    let message = err.to_string();
    for needle in [
        "ngram_hot_rows",
        &(45 * GIB).to_string(),
        &(41 * GIB).to_string(),
        &(2 * GIB).to_string(),
        "6 GiB",
    ] {
        assert!(message.contains(needle), "{needle} missing: {message}");
    }
}

#[test]
fn this_machine_reports_its_available_physical_memory() {
    // The host plan's one measured input. Every machine this suite runs on
    // (the 5090 box, the CI legs) has some, and none has a petabyte.
    let available = ignis_core::residency::available_physical_bytes().expect("measurable here");
    assert!(available > 64 * 1024 * 1024, "{available}");
    assert!(available < 1 << 50, "{available}");
}

#[test]
fn a_machine_short_of_the_pool_itself_names_the_pool() {
    let err = plan_host(&host(40 * GIB)).expect_err("below the margin");
    let message = err.to_string();
    let HostPlanError::BelowMargin { crossing_line, .. } = err;
    assert_eq!(crossing_line, "expert_pool");
    assert!(message.contains("close other applications"), "{message}");
    assert!(!message.contains("hot rows"), "{message}");
}

#[test]
fn the_staging_line_crossing_names_staging_and_its_own_remedy() {
    // 41.5 GiB available, 35.5 usable: the pool and the hot rows fit (40),
    // the staging buffers cross.
    let request = HostPlanRequest {
        expert_pool_bytes: 33 * GIB,
        ..host(41 * GIB + GIB / 2)
    };
    let err = plan_host(&request).expect_err("below the margin");
    let message = err.to_string();
    let HostPlanError::BelowMargin { crossing_line, .. } = err;
    assert_eq!(crossing_line, "staging");
    assert!(message.contains("smaller staging buffers"), "{message}");
    assert!(!message.contains("hot rows,"), "{message}");
}

// ---- the VRAM expert cache -------------------------------------------------

const GU2: usize = 0;
const GU4: usize = 3;
const DN2: usize = 4;
const DN4: usize = 7;

/// One layer of twenty experts: ten hot ones at K = 4 (nine selections
/// each), ten cold ones at K = 2 (one each); every slot 100 bytes.
fn hot_and_cold() -> (ExpertCatalog, ExpertTraffic) {
    hot_and_cold_with(1)
}

fn hot_and_cold_with(cold_selections: u64) -> (ExpertCatalog, ExpertTraffic) {
    let map = (0..20)
        .map(|e| if e < 10 { (KBits::K4, KBits::K4) } else { (KBits::K2, KBits::K2) })
        .collect();
    let catalog = ExpertCatalog::new(1, 20, map, [100; 8]).expect("catalog");
    let counts = (0..20).map(|e| if e < 10 { 9 } else { cold_selections }).collect();
    let traffic = ExpertTraffic::new(&catalog, counts).expect("traffic");
    (catalog, traffic)
}

fn cache_request<'a>(catalog: &'a ExpertCatalog, traffic: &'a ExpertTraffic) -> ExpertCacheRequest<'a> {
    ExpertCacheRequest {
        budget_bytes: 3_800,
        planned_bytes: 1_000,
        staging_ring_bytes: 500,
        table_bytes: 300,
        floor_bytes: 1_500,
        catalog,
        traffic,
        min_slots: 0,
    }
}

#[test]
fn the_cache_takes_what_the_plan_leaves_and_splits_it_as_one_lru_would_hold_it() {
    let (catalog, traffic) = hot_and_cold();
    let plan = plan_expert_cache(&cache_request(&catalog, &traffic)).expect("fits");
    // 3,800 - 1,000 planned - 500 ring - 300 tables.
    assert_eq!(plan.cache_bytes, 2_000);
    // One LRU of 20 slots over 20 hot projections (rate 9) and 20 cold ones
    // (rate 1) keeps a fraction 1 - u^9 of each hot one and 1 - u of each
    // cold one, with u^9 + u = 1: u = 0.8243. So 8.24 slots per hot class,
    // 1.76 per cold one; flooring leaves two slots, which go to the largest
    // remainders. Raw traffic (9:1) would have given the cold classes one.
    assert_eq!(plan.capacity(), [2, 0, 0, 8, 2, 0, 0, 8]);
    assert_eq!(plan.pooled_bytes(), 2_000);
    assert_eq!(plan.pools[GU4].bytes, 800);
    // Each class one LRU over equally hot projections: 8 of 10 hot ones and
    // 2 of 10 cold ones resident, weighted by selections (180 hot, 20 cold).
    assert!((plan.expected_hit_rate - (180.0 * 0.8 + 20.0 * 0.2) / 200.0).abs() < 1e-6);
    let printed = plan.to_string();
    for needle in ["expert_cache 2000 bytes", "expected hit rate 74.0%", "before locality", "gate_up_k4 8", "down_k2 2"] {
        assert!(printed.contains(needle), "{needle} missing: {printed}");
    }
}

#[test]
fn a_cache_larger_than_every_projection_holds_them_all() {
    let (catalog, traffic) = hot_and_cold();
    let request = ExpertCacheRequest {
        budget_bytes: 10_000,
        ..cache_request(&catalog, &traffic)
    };
    let plan = plan_expert_cache(&request).expect("fits");
    assert_eq!(plan.capacity(), [10, 0, 0, 10, 10, 0, 0, 10]);
}

#[test]
fn every_populated_class_gets_its_minimum_even_without_traffic() {
    let (catalog, traffic) = hot_and_cold_with(0);
    let request = ExpertCacheRequest {
        min_slots: 3,
        ..cache_request(&catalog, &traffic)
    };
    let plan = plan_expert_cache(&request).expect("fits");
    // Three slots each first (1,200 B); the 800 B left go to the classes
    // with traffic, equally hot: four more each.
    assert_eq!(plan.capacity()[GU2], 3);
    assert_eq!(plan.capacity()[DN2], 3);
    assert_eq!(plan.capacity()[GU4], 7);
    assert_eq!(plan.capacity()[DN4], 7);
}

#[test]
fn class_minimums_the_cache_cannot_hold_refuse_rather_than_overrun_it() {
    let (catalog, traffic) = hot_and_cold();
    // No floor, and 400 B of cache against four classes of 3 x 100 B.
    let request = ExpertCacheRequest {
        planned_bytes: 2_600,
        floor_bytes: 0,
        min_slots: 3,
        ..cache_request(&catalog, &traffic)
    };
    let err = plan_expert_cache(&request).expect_err("below the minimums");
    assert_eq!(
        err,
        ExpertCachePlanError::BelowClassMinimum {
            cache_bytes: 400,
            needed_bytes: 1_200,
            min_slots: 3
        }
    );
    assert!(err.to_string().contains("1200"), "{err}");
}

#[test]
fn traffic_that_does_not_cover_the_catalog_is_refused() {
    let (catalog, _) = hot_and_cold();
    let err = ExpertTraffic::new(&catalog, vec![1; 19]).expect_err("one expert short");
    assert_eq!((err.expected, err.got), (20, 19));
}

#[test]
fn a_cache_below_its_floor_refuses_naming_what_to_shrink() {
    let (catalog, traffic) = hot_and_cold();
    let request = ExpertCacheRequest {
        planned_bytes: 2_000,
        ..cache_request(&catalog, &traffic)
    };
    let err = plan_expert_cache(&request).expect_err("below the floor");
    let message = err.to_string();
    for needle in [
        "1000", // the cache it would get
        "1500", // the floor
        "--max-context",
        "--vram-headroom-bytes",
        "prefill chunk",
        "desktop",
    ] {
        assert!(message.contains(needle), "{needle} missing: {message}");
    }
}

#[test]
fn the_spec_s_floor_is_twelve_gib() {
    assert_eq!(ignis_core::residency::EXPERT_CACHE_FLOOR_BYTES, 12 * GIB);
}

#[test]
fn the_default_prefetch_budget_is_one_layer_s_share_of_the_round_at_the_link_s_speed() {
    // 6 ms a round at one lane, 7 at three, over 48 layers at 12 GB/s.
    let small = |largest: u64| {
        let mut slot_bytes = [100u64; 8];
        slot_bytes[3] = largest;
        ExpertCatalog::new(48, 1, vec![(KBits::K2, KBits::K2); 48], slot_bytes).expect("catalog")
    };
    assert_eq!(default_prefetch_budget_bytes(1, &small(100)), 1_500_000);
    assert_eq!(default_prefetch_budget_bytes(3, &small(100)), 1_750_000);
    // Never below one projection of the largest class: spec 01's gate/up at
    // K = 4 (1,646,592 B) is larger than one lane's window.
    assert_eq!(default_prefetch_budget_bytes(1, &small(1_646_592)), 1_646_592);
}

#[test]
fn the_class_minimum_holds_every_lane_s_selection_and_prefetch() {
    // Three lanes, top-10, W = 16: a step's own projections of one class
    // plus its prefetches, if every one fell in that class.
    assert_eq!(min_slots_per_class(3, 10, 16), 78);
}

// ---- the staging ring, the tables, the sidecar's traffic -------------------

/// Two layers of three experts; layer 1 is the heavier.
fn catalog() -> ExpertCatalog {
    let map = vec![
        (KBits::K2, KBits::K2),
        (KBits::K2, KBits::K2),
        (KBits::K2, KBits::K2),
        (KBits::K4, KBits::K2),
        (KBits::K3, KBits::K2_5),
        (KBits::K2, KBits::K2),
    ];
    ExpertCatalog::new(2, 3, map, [200, 250, 300, 400, 100, 125, 150, 200]).expect("catalog")
}

#[test]
fn the_staging_ring_holds_two_of_the_heaviest_layer() {
    // Layer 0: 3 x (200 + 100) = 900. Layer 1: 400+100 + 300+125 + 200+100 = 1,225.
    assert_eq!(prefill_staging_ring_bytes(&catalog()), 2 * 1_225);
}

#[test]
fn the_tables_line_is_an_upper_bound_counted_per_projection_and_per_chunk() {
    // 48 layers x 512 experts x 2 projections, 32 bytes each.
    // 48 bytes a projection, two job lists and two ring key lists of one
    // layer, the lookahead ranking of an 8192-token chunk at W = 16, 16 KiB
    // of scalars and rounding.
    assert_eq!(
        residency_table_bytes(48, 512, 8192, 16),
        48 * 512 * 2 * 48 + 2 * 1024 * 28 + 8192 * 16 * 4 + 16 * 1024
    );
}

#[test]
fn the_sidecar_s_per_expert_traffic_gives_the_warm_start_order() {
    // Selections per (layer, expert), layer-major, as converter.json's
    // expert_traffic records them.
    let traffic = ExpertTraffic::new(&catalog(), vec![5, 0, 9, 7, 9, 1]).expect("traffic");
    let order = warm_start_order(&traffic);
    let p = |layer, expert, projection| ProjectionId::new(layer, expert, projection);
    // Hottest first; a tie keeps the canonical key order; an expert never
    // selected is not warmed.
    assert_eq!(
        order,
        vec![
            p(0, 2, Projection::GateUp),
            p(0, 2, Projection::Down),
            p(1, 1, Projection::GateUp),
            p(1, 1, Projection::Down),
            p(1, 0, Projection::GateUp),
            p(1, 0, Projection::Down),
            p(0, 0, Projection::GateUp),
            p(0, 0, Projection::Down),
            p(1, 2, Projection::GateUp),
            p(1, 2, Projection::Down),
        ]
    );
}
