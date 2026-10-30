//! The 27B's host plan (ADR 0062): its token embedding lives in mapped pinned
//! host memory, so its load plans host memory as Flash-Next's does -- the
//! embedding, KV-disk's staging, the host retained slots and the KV-RAM
//! arena against `model.host_margin_bytes` -- and refuses a start that cannot
//! pin them with the host plan's message, naming the 27B's own flags.

use ignis_core::residency::{plan_host, HostPlanError, HostPlanRequest, DEFAULT_HOST_MARGIN_BYTES};

const GIB: u64 = 1024 * 1024 * 1024;
/// `text/token_embedding`, W8G32 `[248320, 5120]`: 5,440 bytes a row.
const TABLE_BYTES: u64 = 248_320 * 5_440;

/// The make default's 27B: the table, 16 host retained slots' 2 GiB, the
/// 8 GiB KV-RAM arena, KV-disk off.
fn qwen27b(available: u64) -> HostPlanRequest {
    HostPlanRequest {
        available_physical_bytes: available,
        scope: "qwen38",
        token_embedding_bytes: TABLE_BYTES,
        vision_tower_bytes: 0,
        expert_pool_bytes: 0,
        load_staging_bytes: 0,
        ngram_hot_rows_bytes: 0,
        staging_bytes: 0,
        retained_host_slots_bytes: 2 * GIB,
        kv_ram_arena_bytes: 8 * GIB,
        margin_bytes: DEFAULT_HOST_MARGIN_BYTES,
    }
}

#[test]
fn the_27b_plan_lists_the_token_embedding_first_and_holds_every_line() {
    let plan = plan_host(&qwen27b(64 * GIB)).expect("fits");
    assert_eq!(plan.entries()[0], ("token_embedding", TABLE_BYTES), "the model's line before reuse's");
    assert_eq!(plan.total_bytes, TABLE_BYTES + 10 * GIB);
    assert_eq!(plan.left_bytes, 64 * GIB - plan.total_bytes);
    // Exactly at the margin, and a byte short of it.
    let exact = plan.total_bytes + DEFAULT_HOST_MARGIN_BYTES;
    assert!(plan_host(&qwen27b(exact)).is_ok());
    assert!(plan_host(&qwen27b(exact - 1)).is_err());
}

#[test]
fn a_host_that_cannot_pin_the_token_embedding_refuses_the_start_naming_it() {
    // 5 GiB available, 1 GiB usable: not even the 1.3 GiB table.
    let err = plan_host(&qwen27b(5 * GIB)).expect_err("below the margin");
    let HostPlanError::BelowMargin { crossing_line, planned_bytes, .. } = &err;
    assert_eq!((*crossing_line, *planned_bytes), ("token_embedding", TABLE_BYTES + 10 * GIB));
    let message = err.to_string();
    for needle in [
        "not enough RAM: the load holds 11.3 GiB and keeps 4 GiB free for the system",
        "Even the token embedding alone (1.3 GiB) does not fit in pinned memory: close other applications",
        "Plan: token embedding 1.3 GiB, retained host slots 2 GiB, KV-RAM arena 8 GiB; \
         margin 4 GiB (--qwen38-model-host-margin-bytes).",
    ] {
        assert!(message.contains(needle), "{needle} missing: {message}");
    }
    assert!(!message.contains("qwen38flashnext"), "the 27B's own flags: {message}");
    assert!(!message.contains("experts"), "the 27B has none: {message}");
}

#[test]
fn a_27b_plan_short_of_its_reuse_lines_names_the_27bs_reuse_flags() {
    // 10 GiB usable: the table and the slots fit, the arena does not.
    let err = plan_host(&qwen27b(14 * GIB)).expect_err("below the margin");
    let message = err.to_string();
    let HostPlanError::BelowMargin { crossing_line, .. } = err;
    assert_eq!(crossing_line, "kv_ram_arena");
    assert!(message.contains("--qwen38-reuse-kv-host-pool-bytes, 0 turns it off"), "{message}");

    // 2.5 GiB usable: the table fits, the slots do not.
    let err = plan_host(&qwen27b(6 * GIB + GIB / 2)).expect_err("below the margin");
    let message = err.to_string();
    let HostPlanError::BelowMargin { crossing_line, .. } = err;
    assert_eq!(crossing_line, "retained_host_slots");
    assert!(message.contains("(--qwen38-reuse-retained-host)"), "{message}");

    // KV-disk on: 32 MiB past the table, its 64 MiB of staging windows cross
    // before reuse's lines.
    let staged =
        HostPlanRequest { staging_bytes: 64 << 20, ..qwen27b(DEFAULT_HOST_MARGIN_BYTES + TABLE_BYTES + (32 << 20)) };
    let err = plan_host(&staged).expect_err("below the margin");
    let message = err.to_string();
    let HostPlanError::BelowMargin { crossing_line, .. } = err;
    assert_eq!(crossing_line, "staging");
    assert!(
        message.contains("The token embedding fits, KV-disk's staging buffers do not: --qwen38-kv-disk-bytes 0 drops them"),
        "{message}"
    );
}

/// ADR 0075 (GitHub #414): the vision tower under the `host` residency, as the
/// binder lays it out in the mapped allocation: 333 spans, each 4 KiB aligned
/// at both ends (the prototype's +297,000,960 bytes on its host line, #409).
const TOWER_SPAN_BYTES: u64 = 297_000_960;

#[test]
fn the_host_residencys_vision_tower_is_a_line_of_its_own_beside_the_token_embedding() {
    let request = HostPlanRequest { vision_tower_bytes: TOWER_SPAN_BYTES, ..qwen27b(64 * GIB) };
    let plan = plan_host(&request).expect("fits");
    assert_eq!(plan.entries()[0], ("token_embedding", TABLE_BYTES), "not charged to the table's line");
    assert_eq!(plan.entries()[1], ("vision_tower", TOWER_SPAN_BYTES));
    assert_eq!(plan.total_bytes, TABLE_BYTES + TOWER_SPAN_BYTES + 10 * GIB);
    // A load whose tower is on the device, or that has none, plans no such bytes.
    assert_eq!(plan_host(&qwen27b(64 * GIB)).unwrap().entries()[1], ("vision_tower", 0));
}

#[test]
fn a_host_that_cannot_pin_the_vision_tower_refuses_the_start_naming_the_bytes_and_both_ways_out() {
    // The table and 100 MiB fit beside the margin; the 283 MiB tower does not.
    let available = DEFAULT_HOST_MARGIN_BYTES + TABLE_BYTES + (100 << 20);
    let request = HostPlanRequest {
        vision_tower_bytes: TOWER_SPAN_BYTES,
        retained_host_slots_bytes: 0,
        kv_ram_arena_bytes: 0,
        ..qwen27b(available)
    };
    let err = plan_host(&request).expect_err("below the margin");
    assert_eq!(err.short_bytes(), TOWER_SPAN_BYTES - (100 << 20));
    let message = err.to_string();
    let HostPlanError::BelowMargin { crossing_line, .. } = err;
    assert_eq!(crossing_line, "vision_tower");
    for needle in [
        "(183 MiB short)",
        "The token embedding fits, the vision tower (283 MiB) does not",
        "vision.residency: device",
        "--vision-residency device",
        "vision off",
        "Plan: token embedding 1.3 GiB, vision tower 283 MiB; margin 4 GiB (--qwen38-model-host-margin-bytes).",
    ] {
        assert!(message.contains(needle), "{needle} missing: {message}");
    }
    // Short of the reuse lines instead, it names those: the tower fits.
    let err = plan_host(&HostPlanRequest { vision_tower_bytes: TOWER_SPAN_BYTES, ..qwen27b(14 * GIB) }).expect_err("short");
    let HostPlanError::BelowMargin { crossing_line, .. } = err;
    assert_eq!(crossing_line, "kv_ram_arena");
}
