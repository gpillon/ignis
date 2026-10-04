//! Expert residency's plan lines (spec flash-next/03, GitHub #301, ADR 0030):
//! the host plan and its refusal, the VRAM expert cache line and its split
//! into eight K-class pools, the prefill staging ring. Pure arithmetic over
//! numbers the loader has measured or read from the artifact.

use ignis_core::residency::{
    ExpertCacheRequest, ExpertCatalog, HOST_MARGIN_BYTES, HostPlanError, HostPlanRequest, KBits,
    KClass, Projection, ProjectionId, class_selections, min_slots_per_class, plan_expert_cache,
    plan_host, prefill_staging_ring_bytes, residency_table_bytes, warm_start_order,
};

const GIB: u64 = 1024 * 1024 * 1024;

// ---- the host plan -----------------------------------------------------------

fn host(available: u64) -> HostPlanRequest {
    HostPlanRequest {
        available_physical_bytes: available,
        expert_pool_bytes: 38 * GIB,
        ngram_hot_rows_bytes: 2 * GIB,
        staging_bytes: GIB,
    }
}

#[test]
fn the_host_plan_adds_its_lines_and_says_what_it_leaves() {
    let plan = plan_host(&host(52 * GIB)).expect("fits");
    let names: Vec<_> = plan.entries().iter().map(|(name, _)| *name).collect();
    assert_eq!(names, ["expert_pool", "ngram_hot_rows", "staging"]);
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
    let HostPlanError::BelowMargin { crossing_line, .. } = err;
    assert_eq!(crossing_line, "expert_pool");
}

// ---- the VRAM expert cache -------------------------------------------------

const GU2: usize = 0;
const DN2: usize = 4;

/// Two populated classes, gate/up K2 (100 B slots) and down K2 (50 B).
fn cache_request() -> ExpertCacheRequest {
    let mut slot_bytes = [1u64; 8];
    slot_bytes[GU2] = 100;
    slot_bytes[DN2] = 50;
    let mut projections = [0u64; 8];
    projections[GU2] = 1000;
    projections[DN2] = 1000;
    let mut selections = [0u64; 8];
    selections[GU2] = 30;
    selections[DN2] = 40;
    ExpertCacheRequest {
        budget_bytes: 11_800,
        planned_bytes: 1_000,
        staging_ring_bytes: 500,
        table_bytes: 300,
        floor_bytes: 5_000,
        slot_bytes,
        projections,
        selections,
        min_slots: 2,
    }
}

#[test]
fn the_cache_takes_what_the_plan_leaves_and_splits_it_by_byte_traffic() {
    let plan = plan_expert_cache(&cache_request()).expect("fits");
    // 11,800 - 1,000 planned - 500 ring - 300 tables.
    assert_eq!(plan.cache_bytes, 10_000);
    // Two slots each first (300 B), then 9,700 B split 3:2 by selections x
    // slot bytes (30 x 100 against 40 x 50): 5,820 B -> 58 more gate/up
    // slots, 3,880 B -> 77 more down slots.
    let capacity = plan.capacity();
    assert_eq!(capacity[GU2], 60);
    assert_eq!(capacity[DN2], 79);
    for (i, slots) in capacity.iter().enumerate() {
        if i != GU2 && i != DN2 {
            assert_eq!(*slots, 0, "class {i} has no projections");
        }
    }
    assert_eq!(plan.pools[GU2].bytes, 6_000);
    assert_eq!(plan.pools[DN2].bytes, 3_950);
    assert!(plan.pooled_bytes() <= plan.cache_bytes);
}

#[test]
fn a_class_never_gets_more_slots_than_it_has_projections_and_its_share_goes_to_the_rest() {
    let mut request = cache_request();
    request.projections[GU2] = 20;
    let plan = plan_expert_cache(&request).expect("fits");
    let capacity = plan.capacity();
    assert_eq!(capacity[GU2], 20);
    // 10,000 - 2,000 for the whole gate/up class = 8,000 B of downs.
    assert_eq!(capacity[DN2], 160);
    assert_eq!(plan.pooled_bytes(), 10_000);
}

#[test]
fn every_populated_class_gets_its_minimum_even_without_traffic() {
    let mut request = cache_request();
    request.selections[DN2] = 0;
    let plan = plan_expert_cache(&request).expect("fits");
    assert_eq!(plan.capacity()[DN2], 2);
    // Everything else goes to the class that has traffic.
    assert_eq!(plan.capacity()[GU2], (10_000 - 100) / 100);
}

#[test]
fn a_cache_below_its_floor_refuses_naming_what_to_shrink() {
    let mut request = cache_request();
    request.planned_bytes = 7_000;
    let err = plan_expert_cache(&request).expect_err("below the floor");
    let message = err.to_string();
    for needle in [
        "4000", // the cache it would get
        "5000", // the floor
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
fn the_tables_cost_a_slot_table_entry_and_an_lru_entry_per_projection() {
    // 48 layers x 512 experts x 2 projections, 32 bytes each.
    assert_eq!(residency_table_bytes(48, 512), 48 * 512 * 2 * 32);
}

#[test]
fn the_sidecar_s_per_expert_traffic_gives_class_selections_and_the_warm_start_order() {
    // Selections per (layer, expert), layer-major, as converter.json's
    // expert_traffic records them.
    let counts = [5, 0, 9, 7, 9, 1];
    let selections = class_selections(&catalog(), &counts);
    let gu = |k| KClass::new(Projection::GateUp, k).index();
    let dn = |k| KClass::new(Projection::Down, k).index();
    assert_eq!(selections[gu(KBits::K2)], 5 + 9 + 1);
    assert_eq!(selections[gu(KBits::K3)], 9);
    assert_eq!(selections[gu(KBits::K4)], 7);
    assert_eq!(selections[dn(KBits::K2)], 5 + 9 + 7 + 1);
    assert_eq!(selections[dn(KBits::K2_5)], 9);

    let order = warm_start_order(&catalog(), &counts);
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
