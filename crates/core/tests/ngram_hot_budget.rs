//! The n-gram hot-row budget (`--ngram-hot-bytes`, GitHub #306): a named
//! size, the 1 GiB default, or `auto` -- what the host plan leaves the line
//! once its other lines and the 6 GiB margin are placed, capped at the whole
//! table. Pure arithmetic over the table's shape and the host plan's lines;
//! the shapes below are the real artifact's (layout.md 7: 320,001,536 rows of
//! 90 bytes, a hot list of 15,642,665 rows) and the host plan measured on the
//! 5090 box (`docs/findings/2026-10-06-flash-next-on-the-5090.md`).

use ignis_core::ngram_table::{HotBudget, TableFootprint, DEFAULT_HOT_BYTES};
use ignis_core::residency::{ngram_hot_rows_room, plan_host, HostPlanRequest, HOST_MARGIN_BYTES};

const GIB: u64 = 1 << 30;

const FLASH_NEXT: TableFootprint = TableFootprint { table_rows: 320_001_536, row_bytes: 90, ranked_rows: 15_642_665 };

/// The host plan of a Flash-Next start at `available` bytes, but for its
/// hot-row line.
fn host(available: u64) -> HostPlanRequest {
    HostPlanRequest {
        available_physical_bytes: available,
        expert_pool_bytes: 37_795_446_784,
        ngram_hot_rows_bytes: 0,
        staging_bytes: 0,
        retained_host_slots_bytes: 1_040_115_712,
        kv_ram_arena_bytes: 2 * GIB,
    }
}

#[test]
fn the_table_costs_its_rows_whole_and_its_rows_and_index_ranked() {
    assert_eq!(FLASH_NEXT.whole_table_bytes(), 28_800_138_240);
    assert_eq!(FLASH_NEXT.ranked_bytes(), 15_642_665 * 94);
    // The default holds the hot list's first 11,422,785 rows, as it always
    // has (the cache file beside the model is 80 + 11,422,785 * 90 bytes).
    assert_eq!(FLASH_NEXT.held_bytes(DEFAULT_HOT_BYTES), 11_422_785 * 94);
    // Past the hot list and short of the whole table, a budget holds the list.
    assert_eq!(FLASH_NEXT.held_bytes(4 * GIB), FLASH_NEXT.ranked_bytes());
    assert_eq!(FLASH_NEXT.held_bytes(FLASH_NEXT.whole_table_bytes() - 1), FLASH_NEXT.ranked_bytes());
    assert_eq!(FLASH_NEXT.held_bytes(FLASH_NEXT.whole_table_bytes()), FLASH_NEXT.whole_table_bytes());
    assert_eq!(FLASH_NEXT.held_bytes(u64::MAX), FLASH_NEXT.whole_table_bytes(), "capped at the whole table");
    assert_eq!(FLASH_NEXT.held_bytes(0), 0);
}

#[test]
fn auto_takes_the_whole_table_else_the_hot_list_else_whole_gib_never_below_the_default() {
    let whole = FLASH_NEXT.whole_table_bytes();
    let ranked = FLASH_NEXT.ranked_bytes();
    for (room, budget) in [
        (0, DEFAULT_HOT_BYTES),
        (GIB - 1, DEFAULT_HOT_BYTES),
        (GIB, GIB),
        (ranked - 1, GIB),
        (ranked, ranked),
        (20 * GIB, ranked),
        (whole - 1, ranked),
        (whole, whole),
        (u64::MAX, whole),
    ] {
        assert_eq!(FLASH_NEXT.auto_budget(room), budget, "room {room}");
    }
    // A hot list at the converter's 2 GiB cap: the whole GiB below it count.
    let long = TableFootprint { ranked_rows: (2 * GIB) / 90, ..FLASH_NEXT };
    assert_eq!(long.auto_budget(2 * GIB + 5), 2 * GIB);
    assert_eq!(long.auto_budget(2 * GIB - 1), GIB);
    assert_eq!(long.auto_budget(long.ranked_bytes()), long.ranked_bytes());
}

#[test]
fn a_named_budget_is_used_as_named_and_an_unmeasured_auto_is_the_default() {
    assert_eq!(HotBudget::default(), HotBudget::Bytes(DEFAULT_HOT_BYTES));
    assert_eq!(HotBudget::Bytes(4 * GIB).resolve(&FLASH_NEXT, Some(0)), 4 * GIB);
    assert_eq!(HotBudget::Bytes(0).resolve(&FLASH_NEXT, Some(64 * GIB)), 0);
    assert_eq!(HotBudget::Auto.resolve(&FLASH_NEXT, None), DEFAULT_HOT_BYTES);
    assert_eq!(HotBudget::Auto.resolve(&FLASH_NEXT, Some(64 * GIB)), FLASH_NEXT.whole_table_bytes());
    assert_eq!((HotBudget::Auto.to_string(), HotBudget::Bytes(GIB).to_string()), ("auto".into(), GIB.to_string()));
}

#[test]
fn the_room_is_what_the_other_lines_and_the_margin_leave() {
    let available = 52_190_724_096;
    let others = 37_795_446_784 + 1_040_115_712 + 2 * GIB;
    assert_eq!(ngram_hot_rows_room(&host(available)), available - HOST_MARGIN_BYTES - others);
    // The line's own request does not shrink its room.
    let asked = HostPlanRequest { ngram_hot_rows_bytes: 9 * GIB, ..host(available) };
    assert_eq!(ngram_hot_rows_room(&asked), ngram_hot_rows_room(&host(available)));
    assert_eq!(ngram_hot_rows_room(&host(40 * GIB)), 0, "nothing left saturates at 0");
    // A line of exactly the room fits the plan; a byte more crosses the margin.
    let room = ngram_hot_rows_room(&host(available));
    assert!(plan_host(&HostPlanRequest { ngram_hot_rows_bytes: room, ..host(available) }).is_ok());
    assert!(plan_host(&HostPlanRequest { ngram_hot_rows_bytes: room + 1, ..host(available) }).is_err());
}

#[test]
fn auto_never_refuses_a_start_the_default_would_make() {
    let default_line = FLASH_NEXT.held_bytes(DEFAULT_HOT_BYTES);
    let mut starts = (0, 0);
    let mut available = 40 * GIB;
    while available <= 80 * GIB {
        let room = ngram_hot_rows_room(&host(available));
        let auto_line = FLASH_NEXT.held_bytes(HotBudget::Auto.resolve(&FLASH_NEXT, Some(room)));
        let by_default = plan_host(&HostPlanRequest { ngram_hot_rows_bytes: default_line, ..host(available) });
        let by_auto = plan_host(&HostPlanRequest { ngram_hot_rows_bytes: auto_line, ..host(available) });
        starts.0 += u32::from(by_default.is_ok());
        starts.1 += u32::from(by_auto.is_ok());
        if by_default.is_ok() {
            assert!(by_auto.is_ok(), "available {available}: auto takes {auto_line} of a {room}-byte room");
            assert!(auto_line >= default_line, "available {available}: auto holds less than the default");
        } else {
            // The default's line did not fit: auto falls back to it, and the
            // plan refuses the same way, naming the same line.
            assert_eq!(auto_line, default_line, "available {available}");
            assert_eq!(by_auto.unwrap_err(), by_default.unwrap_err());
        }
        available += 7 * (1 << 20);
    }
    assert_eq!(starts.0, starts.1);
    assert!(starts.0 > 0, "the sweep covers starts as well as refusals");
}

#[test]
fn auto_selects_at_most_three_row_sets_however_free_ram_moves() {
    // The persistent cache file is keyed by the rows the budget selects, so
    // auto must not select new ones whenever free RAM moves by a few MB: it
    // selects the default's rows, the whole hot list, or the whole table
    // (which is never written to a cache file).
    let mut held = std::collections::BTreeSet::new();
    let mut room = 0;
    while room <= 40 * GIB {
        held.insert(FLASH_NEXT.held_bytes(FLASH_NEXT.auto_budget(room)));
        room += 3 * (1 << 20);
    }
    let expected: std::collections::BTreeSet<u64> = [
        FLASH_NEXT.held_bytes(DEFAULT_HOT_BYTES),
        FLASH_NEXT.ranked_bytes(),
        FLASH_NEXT.whole_table_bytes(),
    ]
    .into();
    assert_eq!(held, expected);
}
