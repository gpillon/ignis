//! Expert residency's plan lines (spec flash-next/03, GitHub #301, ADR 0030):
//! the host plan and its refusal, the VRAM expert cache line and its split
//! into eight K-class pools, the prefill staging ring. Pure arithmetic over
//! numbers the loader has measured or read from the artifact.

use ignis_core::residency::{
    ExpertCachePlanError, ExpertCacheRequest, ExpertCatalog, ExpertTraffic, DEFAULT_HOST_MARGIN_BYTES,
    HostPlanError, HostPlanRequest, KBits, MEASURED_ONE_ROW_PREFETCH_BUDGET_BYTES,
    MEASURED_THREE_ROW_PREFETCH_BUDGET_BYTES, Projection, ProjectionId, default_prefetch_budget, human_bytes,
    holds_every_expert, min_slots_per_class, pin_hot_experts, plan_expert_cache, plan_host, prefill_staging_ring_bytes,
    residency_table_bytes, staging_ring_bytes_for, warm_start_order,
};

const GIB: u64 = 1024 * 1024 * 1024;

// ---- the host plan -----------------------------------------------------------

fn host(available: u64) -> HostPlanRequest {
    HostPlanRequest {
        available_physical_bytes: available,
        scope: "qwen38flashnext",
        token_embedding_bytes: 0,
        vision_tower_bytes: 0,
        vision_pool_bytes: 0,
        expert_pool_bytes: 38 * GIB,
        load_staging_bytes: 0,
        ngram_hot_rows_bytes: 2 * GIB,
        staging_bytes: GIB,
        retained_host_slots_bytes: 0,
        kv_ram_arena_bytes: 0,
        margin_bytes: DEFAULT_HOST_MARGIN_BYTES,
    }
}

#[test]
fn the_host_plan_adds_its_lines_and_says_what_it_leaves() {
    let plan = plan_host(&host(52 * GIB)).expect("fits");
    let names: Vec<_> = plan.entries().iter().map(|(name, _)| *name).collect();
    assert_eq!(
        names,
        [
            "token_embedding",
            "vision_tower",
            "vision_pool",
            "expert_pool",
            "load_staging",
            "ngram_hot_rows",
            "staging",
            "retained_host_slots",
            "kv_ram_arena"
        ]
    );
    assert_eq!(plan.total_bytes, 41 * GIB);
    assert_eq!(plan.left_bytes, 11 * GIB);
}

#[test]
fn the_host_plan_holds_exactly_at_the_four_gib_default_margin() {
    assert_eq!(DEFAULT_HOST_MARGIN_BYTES, 4 * GIB);
    assert!(plan_host(&host(45 * GIB)).is_ok());
    assert!(plan_host(&host(45 * GIB - 1)).is_err());
}

#[test]
fn a_named_margin_is_the_one_the_plan_leaves() {
    // The spec's old 6 GiB: 47 GiB holds the 41 GiB plan, a byte less does not.
    let six = |available| HostPlanRequest { margin_bytes: 6 * GIB, ..host(available) };
    assert!(plan_host(&six(47 * GIB)).is_ok());
    assert!(plan_host(&six(47 * GIB - 1)).is_err());
    // None at all: the lines may take every available byte.
    let none = |available| HostPlanRequest { margin_bytes: 0, ..host(available) };
    assert_eq!(plan_host(&none(41 * GIB)).expect("fits exactly").left_bytes, 0);
    // A margin that is not whole GiB prints with its decimal: 44.5 GiB
    // needed, 44 available.
    let half = HostPlanRequest { margin_bytes: 3 * GIB + GIB / 2, ..host(44 * GIB) };
    let err = plan_host(&half).expect_err("below the margin");
    assert_eq!((err.needed_bytes(), err.short_bytes()), (44 * GIB + GIB / 2, GIB / 2));
    let message = err.to_string();
    assert!(message.contains("keeps 3.5 GiB free") && message.contains("(512 MiB short)"), "{message}");
}

#[test]
fn bytes_read_as_binary_units() {
    for (bytes, text) in [
        (4 * GIB, "4 GiB"),
        (37_795_446_784, "35.2 GiB"),
        (1_040_115_712, "992 MiB"),
        (64 << 20, "64 MiB"),
        (512 << 10, "512 KiB"),
        (100, "100 bytes"),
        (0, "0 bytes"),
    ] {
        assert_eq!(human_bytes(bytes), text, "{bytes}");
    }
}

#[test]
fn a_host_plan_below_the_margin_refuses_naming_the_line_that_crosses_it() {
    // 43 GiB available, 39 usable: the pool fits (38), the hot rows cross.
    let err = plan_host(&host(43 * GIB)).expect_err("below the margin");
    let HostPlanError::BelowMargin {
        available_bytes,
        planned_bytes,
        crossing_line,
        ..
    } = &err;
    assert_eq!(*available_bytes, 43 * GIB);
    assert_eq!(*planned_bytes, 41 * GIB);
    assert_eq!(*crossing_line, "ngram_hot_rows");
    assert_eq!((err.needed_bytes(), err.short_bytes()), (45 * GIB, 2 * GIB));
    // A person reads it in GiB, with how far short the load fell and what
    // would make room; the exact bytes are the error's fields.
    let message = err.to_string();
    for needle in [
        "not enough RAM: the load holds 41 GiB and keeps 4 GiB free for the system",
        "needs 45 GiB available, but only 43 GiB is (2 GiB short)",
        "the n-gram hot rows do not: hold fewer (--ngram-hot-bytes)",
        "Plan: experts 38 GiB, n-gram hot rows 2 GiB, staging 1 GiB",
        "margin 4 GiB (--qwen38flashnext-model-host-margin-bytes)",
    ] {
        assert!(message.contains(needle), "{needle} missing: {message}");
    }
    assert!(!message.contains(&(41 * GIB).to_string()), "no raw byte counts: {message}");
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
    assert!(message.contains("Even the experts alone (38 GiB) do not fit: close other applications"), "{message}");
    assert!(!message.contains("--ngram-hot-bytes"), "{message}");
}

#[test]
fn the_staging_line_crossing_names_staging_and_its_own_remedy() {
    // 39.5 GiB available, 35.5 usable: the pool and the hot rows fit (35),
    // the staging buffers cross.
    let request = HostPlanRequest {
        expert_pool_bytes: 33 * GIB,
        ..host(39 * GIB + GIB / 2)
    };
    let err = plan_host(&request).expect_err("below the margin");
    let message = err.to_string();
    let HostPlanError::BelowMargin { crossing_line, .. } = err;
    assert_eq!(crossing_line, "staging");
    // Flash-Next's own flag, as every other remedy names: a 27B start's
    // general `--kv-disk-bytes` cannot shadow it.
    assert!(message.contains("KV-disk's staging buffers do not: --qwen38flashnext-kv-disk-bytes 0 drops them"), "{message}");
    assert!(!message.contains("--ngram-hot-bytes"), "{message}");
}

/// A device-only load (ADR 0054): no expert pool, its experts read through
/// 5 GiB of layer buffers; 1 GiB of n-gram hot rows, KV-disk's 64 MiB, a
/// GiB of retained host slots and the 2 GiB arena held while serving.
fn device_only(available: u64) -> HostPlanRequest {
    HostPlanRequest {
        available_physical_bytes: available,
        scope: "qwen38flashnext",
        token_embedding_bytes: 0,
        vision_tower_bytes: 0,
        vision_pool_bytes: 0,
        expert_pool_bytes: 0,
        load_staging_bytes: 5 * GIB,
        ngram_hot_rows_bytes: GIB,
        staging_bytes: 64 << 20,
        retained_host_slots_bytes: GIB,
        kv_ram_arena_bytes: 2 * GIB,
        margin_bytes: DEFAULT_HOST_MARGIN_BYTES,
    }
}

#[test]
fn the_layer_buffers_are_charged_only_while_the_experts_load() {
    // Freed before the n-gram rows are read (GitHub #329 item 3): the lines
    // held while serving may take what the buffers held.
    let serving = HostPlanRequest { ngram_hot_rows_bytes: 26 * GIB, ..device_only(0) };
    let plan = plan_host(&HostPlanRequest { available_physical_bytes: 33 * GIB + (64 << 20), ..serving })
        .expect("serving's 29 GiB and 64 MiB fit beside the margin");
    assert_eq!(plan.total_bytes, 29 * GIB + (64 << 20), "the peak, not the lines' 34 GiB sum");
    assert_eq!(plan.left_bytes, DEFAULT_HOST_MARGIN_BYTES);
    let err = plan_host(&HostPlanRequest { available_physical_bytes: 33 * GIB + (64 << 20) - 1, ..serving })
        .expect_err("a byte short");
    let HostPlanError::BelowMargin { crossing_line, planned_bytes, .. } = err;
    assert_eq!((crossing_line, planned_bytes), ("kv_ram_arena", 29 * GIB + (64 << 20)));
}

#[test]
fn a_residency_phase_short_of_room_names_the_layer_buffers() {
    // 8.5 GiB available, 4.5 usable: serving's 4 GiB and 64 MiB fit, the
    // 5 GiB of layer buffers do not.
    let err = plan_host(&device_only(8 * GIB + GIB / 2)).expect_err("below the margin");
    let HostPlanError::BelowMargin { planned_bytes, crossing_line, lines, .. } = &err;
    assert_eq!((*crossing_line, *planned_bytes), ("load_staging", 5 * GIB));
    assert_eq!((err.needed_bytes(), err.short_bytes()), (9 * GIB, GIB / 2));
    assert_eq!(lines[4], ("load_staging", 5 * GIB), "the exact bytes name it");
    let message = err.to_string();
    for needle in [
        "the load holds 5 GiB and keeps 4 GiB free for the system, so it needs 9 GiB",
        "The experts' layer buffers do not fit while the load reads them into VRAM: close other applications",
        "Plan: expert layer buffers 5 GiB (while the experts load), n-gram hot rows 1 GiB, \
         staging 64 MiB, retained host slots 1 GiB, KV-RAM arena 2 GiB; margin 4 GiB",
    ] {
        assert!(message.contains(needle), "{needle} missing: {message}");
    }
    let plan = plan_host(&device_only(9 * GIB)).expect("fits exactly");
    assert_eq!((plan.total_bytes, plan.left_bytes), (5 * GIB, 4 * GIB));
}

#[test]
fn a_residency_phase_short_of_room_is_named_before_a_reuse_line() {
    // 7 GiB available, 3 usable: the arena crosses while serving, but
    // shrinking it would still leave the layer buffers short.
    let err = plan_host(&device_only(7 * GIB)).expect_err("below the margin");
    let message = err.to_string();
    let HostPlanError::BelowMargin { crossing_line, .. } = err;
    assert_eq!(crossing_line, "load_staging");
    assert!(!message.contains("-reuse-"), "{message}");
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
        allow_below_floor: false,
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

/// The resident load's cache is exactly the catalog's bytes. The LRU split
/// holds no never-selected projection and floors each class's share, so it
/// left projections slotless there with bytes unused (244 at one lane, 1,944
/// at eight on the real artifact); every projection now gets its slot.
#[test]
fn a_cache_of_exactly_the_catalog_gives_every_projection_a_slot_selected_or_not() {
    let (catalog, traffic) = hot_and_cold_with(0);
    assert_eq!(catalog.total_bytes(), 4_000);
    let request = ExpertCacheRequest {
        budget_bytes: 4_000 + 1_000 + 500 + 300,
        ..cache_request(&catalog, &traffic)
    };
    let plan = plan_expert_cache(&request).expect("fits");
    assert_eq!(plan.cache_bytes, 4_000);
    assert_eq!(plan.capacity(), [10, 0, 0, 10, 10, 0, 0, 10]);
    assert_eq!(plan.pooled_bytes(), 4_000, "no byte of the cache left unassigned");
    assert_eq!(plan.expected_hit_rate, 1.0);
    // One slot's bytes short of the catalog the LRU split runs as before: one
    // projection without a slot, every byte assigned.
    let short = plan_expert_cache(&ExpertCacheRequest { budget_bytes: request.budget_bytes - 100, ..request })
        .expect("fits");
    assert_eq!(short.capacity().iter().sum::<u32>(), 39);
    assert_eq!(short.pooled_bytes(), 3_900);
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
        "--vram-kv-pool-bytes",
        "--model-max-context",
        "--vram-headroom-bytes",
        "prefill chunk",
        "--reuse-retained-device",
        "desktop",
        // ADR 0045: and the way past it.
        "--vram-allow-expert-cache-below-floor",
    ] {
        assert!(message.contains(needle), "{needle} missing: {message}");
    }
}

#[test]
fn the_floor_opt_in_plans_the_cache_below_it_and_says_so() {
    // ADR 0045 (AC 6): `--vram-allow-expert-cache-below-floor` turns the refusal
    // into a plan flagged for the load's warning -- the same split the cache
    // would get at any size.
    let (catalog, traffic) = hot_and_cold();
    let request = ExpertCacheRequest {
        planned_bytes: 2_000,
        allow_below_floor: true,
        ..cache_request(&catalog, &traffic)
    };
    let plan = plan_expert_cache(&request).expect("allowed below the floor");
    assert!(plan.below_floor);
    assert_eq!(plan.cache_bytes, 1_000);
    assert_eq!(plan.pooled_bytes(), 1_000);
    // Above the floor nothing is flagged, override or not.
    for allow_below_floor in [false, true] {
        let plan = plan_expert_cache(&ExpertCacheRequest { allow_below_floor, ..cache_request(&catalog, &traffic) })
            .expect("fits");
        assert!(!plan.below_floor);
    }
}

#[test]
fn the_class_minimum_still_refuses_with_the_floor_opt_in() {
    let (catalog, traffic) = hot_and_cold();
    let request = ExpertCacheRequest {
        planned_bytes: 2_600,
        min_slots: 3,
        allow_below_floor: true,
        ..cache_request(&catalog, &traffic)
    };
    assert!(matches!(
        plan_expert_cache(&request),
        Err(ExpertCachePlanError::BelowClassMinimum { cache_bytes: 400, needed_bytes: 1_200, min_slots: 3 })
    ));
}

#[test]
fn the_spec_s_floor_is_twelve_gib() {
    assert_eq!(ignis_core::residency::EXPERT_CACHE_FLOOR_BYTES, 12 * GIB);
}

#[test]
fn the_default_prefetch_budget_is_the_line_through_the_budgets_that_served_best() {
    // GitHub #306: what served Flash-Next best at one lane and at three, a
    // step's budget following its rows in between and past them.
    let budget = default_prefetch_budget();
    assert_eq!(budget.bytes_for(1), MEASURED_ONE_ROW_PREFETCH_BUDGET_BYTES);
    assert_eq!(budget.bytes_for(3), MEASURED_THREE_ROW_PREFETCH_BUDGET_BYTES);
    assert_eq!((budget.one_row_bytes, budget.per_row_bytes), (1_172_500, 726_250));
    assert_eq!(budget.bytes_for(2), 1_898_750);
    // A verify round of one lane and three drafts decodes four rows.
    assert_eq!(budget.bytes_for(4), 3_351_250);
    // No floor: one row's budget is below one gate/up projection at K = 4
    // (spec 01's 1,646,592 B), which then is never prefetched at one lane.
    assert!(budget.bytes_for(1) < 1_646_592);
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

/// GitHub #329 item 2: a cache that holds the whole catalog makes the
/// residency device-only (ADR 0054): no step misses, so nothing stages and it
/// reserves no ring. One byte short, a prefill miss can stage, and it needs
/// the ring.
#[test]
fn a_cache_that_holds_every_expert_reserves_no_staging_ring() {
    let catalog = catalog();
    assert_eq!(catalog.total_bytes(), 900 + 1_225);
    let at = |cache_bytes| (holds_every_expert(&catalog, cache_bytes), staging_ring_bytes_for(&catalog, cache_bytes));
    assert_eq!(at(2_125), (true, 0));
    assert_eq!(at(12 << 30), (true, 0));
    assert_eq!(at(2_124), (false, 2 * 1_225));
    assert_eq!(at(0), (false, 2 * 1_225));
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

// ---- the pinned hot set (GitHub #329 item 12) -------------------------------

/// Slots per class for `catalog()`: gate/up K2 4, K3 1, K4 2, down K2 6, K2.5 2.
const POOLS: [u32; 8] = [4, 0, 1, 2, 6, 2, 0, 0];

#[test]
fn nothing_is_pinned_at_a_budget_of_zero() {
    let traffic = ExpertTraffic::new(&catalog(), vec![5, 0, 9, 7, 9, 1]).expect("traffic");
    let pinned = pin_hot_experts(&catalog(), &traffic, POOLS, 1, 0);
    assert!(pinned.is_empty());
    assert_eq!((pinned.bytes, pinned.experts(), pinned.per_class), (0, 0, [0; 8]));
    assert_eq!(pinned.lru_capacity(POOLS), POOLS, "the LRU keeps every slot");
    assert_eq!(pinned.by_key(&catalog()), vec![false; 12]);
}

#[test]
fn the_pinned_set_is_the_hottest_whole_experts_until_their_bytes_reach_the_budget() {
    let catalog = catalog();
    let traffic = ExpertTraffic::new(&catalog, vec![5, 0, 9, 7, 9, 1]).expect("traffic");
    let p = |layer, expert, projection| ProjectionId::new(layer, expert, projection);
    // The hottest, (0, 2), is 200 + 100 bytes: a budget it reaches pins it alone.
    let one = pin_hot_experts(&catalog, &traffic, [4, 0, 2, 2, 6, 2, 0, 0], 1, 300);
    assert_eq!(one.projections, vec![p(0, 2, Projection::GateUp), p(0, 2, Projection::Down)]);
    assert_eq!((one.bytes, one.per_class), (300, [1, 0, 0, 0, 1, 0, 0, 0]));
    // One byte more takes the next expert whole, (1, 1) at 300 + 125.
    let two = pin_hot_experts(&catalog, &traffic, [4, 0, 2, 2, 6, 2, 0, 0], 1, 301);
    assert_eq!(two.experts(), 2);
    assert_eq!((two.bytes, two.per_class), (725, [1, 0, 1, 0, 1, 1, 0, 0]));
    let keys = two.by_key(&catalog);
    // Keys (layer · 3 + expert) · 2 + projection: (0, 2) is 4 and 5, (1, 1) is 8 and 9.
    let pinned_keys: Vec<usize> = keys.iter().enumerate().filter(|&(_, &pinned)| pinned).map(|(key, _)| key).collect();
    assert_eq!(pinned_keys, vec![4, 5, 8, 9]);
}

#[test]
fn an_expert_that_would_take_a_class_below_its_lru_minimum_is_skipped() {
    let catalog = catalog();
    let traffic = ExpertTraffic::new(&catalog, vec![5, 0, 9, 7, 9, 1]).expect("traffic");
    let p = |layer, expert| [ProjectionId::new(layer, expert, Projection::GateUp), ProjectionId::new(layer, expert, Projection::Down)];
    // gate/up K3 has one slot, its minimum: (1, 1) is skipped, the walk goes on.
    // (0, 1) is never selected, so never pinned.
    let pinned = pin_hot_experts(&catalog, &traffic, POOLS, 1, 10_000);
    assert_eq!(pinned.projections, [p(0, 2), p(1, 0), p(0, 0), p(1, 2)].concat());
    assert_eq!(pinned.bytes, 300 + 500 + 300 + 300);
    assert_eq!(pinned.per_class, [3, 0, 0, 1, 4, 0, 0, 0]);
    assert_eq!(pinned.lru_capacity(POOLS), [1, 0, 1, 1, 2, 2, 0, 0]);
}
