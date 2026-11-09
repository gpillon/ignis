//! Spec flash-next/05 (GitHub #303): prompt reuse on Flash-Next, on a CPU.
//!
//! ADR 0029's policy does not change for Flash-Next; its state does. So the
//! reuse-policy suites -- match, lineage, first victim, KV-RAM order, budget
//! exhaustion -- are mounted here unchanged, with one difference from their
//! own runs: every KV-RAM blob the mock writes is Flash-Next's size (its
//! state image as a slot holds it, plus 4,224 paged bytes per token in
//! hq-e8-2b) instead of one nominal byte, and every KV-RAM capacity the
//! suites state in blobs is that many of those. A policy that leaned on the
//! 27B's sizes, or on every blob being the same size, fails here.
//!
//! Below them, blob identity across the two models, the host plan's two
//! reuse lines, and an image conversation on Flash-Next's own capture shape
//! (GitHub #559).

use std::sync::Arc;

use ignis_core::compute::{ModelConfig, ModelFamily};
use ignis_core::residency::{DEFAULT_HOST_MARGIN_BYTES, HostPlanError, HostPlanRequest, plan_host};
use ignis_core::checkpoint::ReuseSource;
use ignis_core::types::{DecodeParams, RequestClass, RequestId, RequestInput, SchedEvent};
use ignis_core::vision::{Grid, MediaItem, Multimodal, TokenSpan};
use ignis_core::{
    ArtifactHash, BlobIdentity, ConcreteScheduler, IdentityField, KvFormat, MockCompute,
    MockSections, Scheduler, SchedulerConfig, Speculation, SpeculativeBackend,
};

/// Flash-Next's blob sizes under hq-e8-2b, the serving format: what every
/// mounted suite's mock charges.
fn sections() -> MockSections {
    MockSections::of(&ModelConfig::qwen38_flash_next(), KvFormat::HqE8_2b)
}

#[path = "prompt_checkpoint.rs"]
mod prompt_checkpoint;

#[path = "kv_ram_spill.rs"]
mod kv_ram_spill;

#[path = "prefix_reuse.rs"]
mod prefix_reuse;

#[path = "retained_prefix.rs"]
mod retained_prefix;

#[path = "retained_slots.rs"]
mod retained_slots;

#[path = "reuse_boundaries.rs"]
mod reuse_boundaries;

// GitHub #559: ADR 0029's rules for multimodal prompts -- an image's identity
// in the match, no publish point inside its placeholders, a claim leaving a
// tail to prefill, an evicted image request restored from its blob.
#[path = "multimodal_scheduling.rs"]
mod multimodal_scheduling;

/// The mounted suites really run at Flash-Next's sizes: a blob is ~124 MiB
/// of image plus its tokens, not a byte.
#[test]
fn the_mounted_suites_charge_flash_next_blobs() {
    let sections = sections();
    assert_eq!(sections.image_bytes, 130_014_464);
    assert_eq!(sections.bytes_per_token, 4_224);
    assert_eq!(sections.blob_bytes(30_000), 130_014_464 + 126_720_000);
}

// ── Blob identity (spec flash-next/05 acceptance 5) ─────────────────────

/// Two artifacts' content hashes. Content hashes of two models' containers
/// differ by construction (`Reader::content_hash` covers every object's
/// name, format and offset), so any two distinct values stand in for them.
const QWEN38_27B: [u8; 32] = [27; 32];
const FLASH_NEXT: [u8; 32] = [38; 32];

const MODEL: &str = "qwen3.8-flash-next";

/// The blob layout version: one leaf, one number today. It is the same on
/// both sides here on purpose, so that only the artifact can refuse.
const LAYOUT: u32 = 5;

fn flash_next_load() -> BlobIdentity {
    // No drafter: Flash-Next has no speculative decoding (spec 04).
    BlobIdentity::of_load(ArtifactHash::from_bytes(FLASH_NEXT), KvFormat::HqE8_2b, None, LAYOUT)
}

/// The 27B as close to Flash-Next as a load of it can be: same KV format,
/// same layout, no drafter.
fn plain_27b_load() -> BlobIdentity {
    BlobIdentity::of_load(ArtifactHash::from_bytes(QWEN38_27B), KvFormat::HqE8_2b, None, LAYOUT)
}

/// The 27B as it serves by default, with DFlash2.
fn speculative_27b_load() -> BlobIdentity {
    BlobIdentity::of_load(
        ArtifactHash::from_bytes(QWEN38_27B),
        KvFormat::HqE8_2b,
        Some(Speculation::new(SpeculativeBackend::Dflash2, 4).unwrap()),
        LAYOUT,
    )
}

/// A conversation's turn whose opener leaves a prompt checkpoint.
fn turn() -> RequestInput {
    RequestInput {
        decision: None,
        model: MODEL.into(),
        tokens: (1..41).collect(),
        params: DecodeParams {
            max_tokens: Some(4),
            ..DecodeParams::default()
        },
        multimodal: None,
        opener_tokens: Some(37),
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        constrained: None,
        forced_literal: None,
        warm_up: false,
    }
}

/// A scheduler on a load of `identity`, after one turn: its pool's identity
/// and the header of the checkpoint the turn left.
fn after_one_turn(identity: BlobIdentity) -> ConcreteScheduler {
    let compute = Arc::new(MockCompute::with_blob_identity(identity));
    let config = SchedulerConfig {
        model: MODEL.into(),
        ..SchedulerConfig::default()
    };
    let mut sched = ConcreteScheduler::with_config(config, compute);
    sched.submit(turn(), RequestClass::Interactive).unwrap();
    while !sched.is_idle() {
        sched.advance();
    }
    sched
}

#[test]
fn a_27b_blob_is_refused_under_flash_next_by_its_artifact() {
    let flash_next = after_one_turn(flash_next_load());
    for dense in [plain_27b_load(), speculative_27b_load()] {
        let header = after_one_turn(dense).checkpoint_pool().entries()[0].header();
        assert_eq!(header.identity, dense, "the entry names the load that made it");
        let refusal = flash_next
            .checkpoint_pool()
            .accepts(&header)
            .expect_err("a 27B blob under Flash-Next");
        assert_eq!(refusal.field, IdentityField::Artifact, "{refusal}");
        assert!(refusal.to_string().contains("artifact"), "{refusal}");
    }
}

#[test]
fn a_flash_next_blob_is_refused_under_the_27b_by_its_artifact() {
    let header = after_one_turn(flash_next_load()).checkpoint_pool().entries()[0].header();
    for dense in [plain_27b_load(), speculative_27b_load()] {
        let refusal = after_one_turn(dense)
            .checkpoint_pool()
            .accepts(&header)
            .expect_err("a Flash-Next blob under the 27B");
        assert_eq!(refusal.field, IdentityField::Artifact, "{refusal}");
    }
    // And Flash-Next takes back what it produced.
    assert_eq!(after_one_turn(flash_next_load()).checkpoint_pool().accepts(&header), Ok(()));
}

// ── The host plan's reuse lines (spec flash-next/05 acceptance 4) ───────

const GIB: u64 = 1024 * 1024 * 1024;

/// Retained host slots by default: 8 on Flash-Next (three agents' chain link
/// and checkpoint each, and a shared system block) whatever its lanes,
/// against the 27B's two per decode lane of its load (GitHub #336).
#[test]
fn flash_next_defaults_to_eight_host_retained_slots() {
    for lanes in [1, 3, 8] {
        assert_eq!(ModelFamily::FlashNext.default_retained_host_slots(lanes), 8);
        assert_eq!(ModelFamily::Qwen38_27b.default_retained_host_slots(lanes), 2 * lanes);
    }
    assert_eq!(
        ModelFamily::Qwen38_27b.default_retained_host_slots(ModelFamily::Qwen38_27b.default_decode_lanes()),
        2 * ignis_core::N_DECODE_LANES as u32
    );
}

/// Spec 05's host plan with the defaults: ~53 GB available, 37.7 GB of
/// pinned experts, 1 GiB of n-gram hot rows, ~0.5 GiB of staging, eight
/// host slots of one image each (~1 GiB) and a 2 GiB KV-RAM arena.
fn spec_host_plan(available: u64) -> HostPlanRequest {
    let image = ModelConfig::qwen38_flash_next().state_image(KvFormat::HqE8_2b).slot_bytes();
    HostPlanRequest {
        available_physical_bytes: available,
        scope: "qwen38flashnext",
        token_embedding_bytes: 0,
        vision_tower_bytes: 0,
        vision_pool_bytes: 0,
        expert_pool_bytes: 37_700_000_000,
        load_staging_bytes: 0,
        ngram_hot_rows_bytes: GIB,
        staging_bytes: GIB / 2,
        retained_host_slots_bytes: u64::from(ModelFamily::FlashNext.default_retained_host_slots(3)) * image,
        kv_ram_arena_bytes: 2 * GIB,
        margin_bytes: DEFAULT_HOST_MARGIN_BYTES,
    }
}

#[test]
fn the_host_plan_charges_the_retained_slots_and_the_kv_ram_arena() {
    let plan = plan_host(&spec_host_plan(53_000_000_000)).expect("the spec's defaults fit");
    let lines = plan.entries();
    let names: Vec<_> = lines.iter().map(|(name, _)| *name).collect();
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
    assert_eq!(lines[7].1, 8 * 130_014_464, "eight images, ~1 GiB");
    assert_eq!(lines[8].1, 2 * GIB);
    // The spec's table: ~10 GB left for the n-gram page cache and the margin.
    assert!((10_000_000_000..11_000_000_000).contains(&plan.left_bytes), "{}", plan.left_bytes);
}

#[test]
fn a_host_plan_short_at_the_retained_slots_names_them() {
    // 44 GB available, 39.71 GB usable past the 4 GiB margin: the experts,
    // hot rows and staging fit (39.31 GB), the eight slots cross (40.35 GB).
    let err = plan_host(&spec_host_plan(44_000_000_000)).expect_err("below the margin");
    let message = err.to_string();
    let HostPlanError::BelowMargin { crossing_line, .. } = err;
    assert_eq!(crossing_line, "retained_host_slots");
    assert!(message.contains("retained host slots do not: fewer of them (--qwen38flashnext-reuse-retained-host)"), "{message}");
    assert!(!message.contains("-reuse-kv-host-pool-bytes"), "{message}");
}

#[test]
fn a_host_plan_short_at_the_kv_ram_arena_names_it() {
    // 46 GB available, 41.71 GB usable: everything but the arena fits.
    let err = plan_host(&spec_host_plan(46_000_000_000)).expect_err("below the margin");
    let message = err.to_string();
    let HostPlanError::BelowMargin { crossing_line, .. } = err;
    assert_eq!(crossing_line, "kv_ram_arena");
    assert!(message.contains("a smaller one (--qwen38flashnext-reuse-kv-host-pool-bytes, 0 turns it off)"), "{message}");
    assert!(message.contains("KV-RAM arena 2 GiB"), "{message}");
}

/// The switch to Flash-Next refused on 2026-10-12, its lines as the log
/// printed them, at the lower of two attempts' available memory (46.7 GB,
/// the desktop up): the 27B's 8 GiB arena crosses the margin, Flash-Next's
/// own 2 GiB fits under the 4 GiB one.
#[test]
fn the_logged_switch_fits_with_flash_nexts_own_arena() {
    let logged = |arena| HostPlanRequest {
        available_physical_bytes: 46_708_031_488,
        scope: "qwen38flashnext",
        token_embedding_bytes: 0,
        vision_tower_bytes: 0,
        vision_pool_bytes: 0,
        expert_pool_bytes: 37_795_446_784,
        load_staging_bytes: 0,
        ngram_hot_rows_bytes: GIB,
        staging_bytes: 64 << 20,
        retained_host_slots_bytes: 1_040_115_712,
        kv_ram_arena_bytes: arena,
        margin_bytes: DEFAULT_HOST_MARGIN_BYTES,
    };
    let HostPlanError::BelowMargin { crossing_line, .. } = plan_host(&logged(8 * GIB)).expect_err("the 27B's arena");
    assert_eq!(crossing_line, "kv_ram_arena");
    let plan = plan_host(&logged(2 * GIB)).expect("Flash-Next's own arena");
    assert!(plan.left_bytes >= DEFAULT_HOST_MARGIN_BYTES, "{}", plan.left_bytes);
}

/// The switch refused on 2026-10-13, as the owner read it: 41.6 GB
/// available with the desktop at ~25 GiB, too little for the experts
/// alone and the margin. The whole message, as a person reads it.
#[test]
fn the_refusal_reads_in_binary_units_with_the_shortfall() {
    let request = HostPlanRequest {
        available_physical_bytes: 41_632_530_432,
        scope: "qwen38flashnext",
        token_embedding_bytes: 0,
        vision_tower_bytes: 0,
        vision_pool_bytes: 0,
        expert_pool_bytes: 37_795_446_784,
        load_staging_bytes: 0,
        ngram_hot_rows_bytes: GIB,
        staging_bytes: 64 << 20,
        retained_host_slots_bytes: 1_040_115_712,
        kv_ram_arena_bytes: 2 * GIB,
        margin_bytes: DEFAULT_HOST_MARGIN_BYTES,
    };
    let err = plan_host(&request).expect_err("the experts alone cross the margin");
    assert_eq!((err.needed_bytes(), err.short_bytes()), (46_418_864_128, 4_786_333_696));
    assert_eq!(
        err.to_string(),
        "not enough RAM: the load holds 39.2 GiB and keeps 4 GiB free for the system, so it needs 43.2 GiB \
         available, but only 38.8 GiB is (4.5 GiB short). Even the experts alone (35.2 GiB) do not fit: close \
         other applications. Plan: experts 35.2 GiB, n-gram hot rows 1 GiB, staging 64 MiB, retained host slots \
         992 MiB, KV-RAM arena 2 GiB; margin 4 GiB (--qwen38flashnext-model-host-margin-bytes)."
    );
}

#[test]
fn prompt_reuse_off_charges_the_host_plan_nothing() {
    let off = HostPlanRequest {
        retained_host_slots_bytes: 0,
        kv_ram_arena_bytes: 0,
        ..spec_host_plan(53_000_000_000)
    };
    let on = plan_host(&spec_host_plan(53_000_000_000)).unwrap();
    let plan = plan_host(&off).unwrap();
    assert_eq!(on.total_bytes - plan.total_bytes, 8 * 130_014_464 + 2 * GIB);
}

// ── An image conversation on Flash-Next (GitHub #559) ───────────────────

/// Flash-Next's capture shape: 64-token pages, and the whole pages below a
/// request's opener handed over by its capture as a pages-only link (GitHub
/// #306) -- where the 27B on Blackwell publishes an imaged prefix at the
/// opener's page floor whenever that floor lies outside the picture.
fn flash_next_config() -> SchedulerConfig {
    SchedulerConfig {
        model: MODEL.into(),
        kv_page_tokens: 64,
        max_sequence_tokens: 40_960,
        kv_capacity_pages: 8 * 40_960 / 64,
        opener_page_rides_capture: true,
        ..sections().scale(SchedulerConfig::default())
    }
}

/// A single-turn image request as the server renders one: four opening
/// tokens, `count` placeholders of the picture `digest`, `<|vision_end|>`, a
/// `question`-token question and the five tokens that close the turn and
/// open the assistant's -- the generation opener -- then `after` tokens past
/// it (four with thinking off, the empty think block).
fn image_turn(count: u32, digest: u8, question: u32, after: u32) -> RequestInput {
    let opener = 4 + count + 1 + question + 5;
    let len = opener + after;
    RequestInput {
        tokens: (1..=len).collect(),
        opener_tokens: Some(opener),
        user_turn_tokens: Some(0),
        multimodal: Some(Arc::new(Multimodal {
            positions: (0..3).flat_map(|_| 0..len as i32).collect(),
            rope_delta: -(count as i32),
            media: vec![MediaItem {
                grid: Grid { t: 1, h: 2, w: 2 * count },
                token_span: TokenSpan { begin: 4, count: count as usize },
                patches: Vec::new(),
                content_digest: [digest; 32],
            }],
        })),
        params: DecodeParams { max_tokens: Some(1), ..DecodeParams::default() },
        ..turn()
    }
}

/// `turn`'s next turn: its prompt up to its opener, a 20-token answer and the
/// two tokens closing it, a new user message of eight tokens with its own
/// opener, and the empty think block past that.
fn next_image_turn(turn: &RequestInput) -> RequestInput {
    let mut next = turn.clone();
    next.tokens.truncate(turn.opener_tokens.unwrap() as usize);
    next.tokens.extend(900_000..900_022);
    next.user_turn_tokens = Some(next.tokens.len() as u32);
    next.tokens.extend(910_000..910_013);
    next.opener_tokens = Some(next.tokens.len() as u32);
    next.tokens.extend(920_000..920_004);
    let len = next.tokens.len();
    Arc::make_mut(next.multimodal.as_mut().unwrap()).positions = (0..3).flat_map(|_| 0..len as i32).collect();
    next
}

/// What one request's prefill was: its chunk widths, and the retained state
/// it resumed from.
#[derive(Debug, PartialEq)]
struct Served {
    widths: Vec<usize>,
    reused: Vec<(ReuseSource, u32)>,
}

/// Submit `inputs` one after another on one scheduler of Flash-Next's capture
/// shape and blob sizes, each run to completion before the next arrives.
fn served_on_flash_next<const N: usize>(inputs: [RequestInput; N]) -> [Served; N] {
    let compute = Arc::new(MockCompute::with_sections(sections()));
    let mut sched = ConcreteScheduler::with_config(flash_next_config(), compute.clone());
    inputs.map(|input| {
        let id: RequestId = sched.submit(input, RequestClass::Interactive).unwrap();
        let mut reused = Vec::new();
        while !sched.is_idle() {
            for event in sched.advance() {
                if let SchedEvent::StateReused { request, source, tokens, .. } = event {
                    if request == id {
                        reused.push((source, tokens));
                    }
                }
            }
        }
        let widths =
            compute.prefill_calls().iter().flatten().filter(|job| job.request == id).map(|job| job.tokens.len()).collect();
        Served { widths, reused }
    })
}

#[test]
fn an_image_turn_is_cut_once_at_its_opener_and_its_repeat_and_next_turn_resume_there() {
    // A 1,024-token picture whose run crosses the opener's page floor, 1,024,
    // and one that ends below it (a 70-token question puts the floor at
    // 1,088): either way the prefill is cut once, at the opener, the pages
    // below it ride the capture, and the identical repeat and the
    // conversation's next turn resume from the checkpoint there, image and
    // all, each leaving the empty think block to prefill.
    for (question, opener) in [(6, 1_040), (70, 1_104)] {
        let turn = image_turn(1_024, 0xAA, question, 4);
        let next = next_image_turn(&turn);
        let [first, repeat, following] = served_on_flash_next([turn.clone(), turn, next]);
        let label = format!("a {question}-token question");
        assert_eq!(first.widths, [1024, opener as usize - 1024, 4], "{label}: one cut, at the opener");
        assert!(first.reused.is_empty(), "{label}");
        assert_eq!(repeat.reused, [(ReuseSource::Device, opener)], "{label}: the repeat resumes at the opener");
        assert_eq!(repeat.widths, [4], "{label}: and prefills the think block alone");
        assert_eq!(following.reused, [(ReuseSource::Device, opener)], "{label}: the next turn resumes there too");
        assert_eq!(following.widths, [35, 4], "{label}: from the first opener to its own, then the think block");
    }
}

#[test]
fn another_picture_of_the_same_size_resumes_nothing_of_an_image_turn() {
    // The same token ids, another picture: the checkpoint is keyed by its
    // image as well, and the link below it is never claimed on its own.
    let [first, other] = served_on_flash_next([image_turn(1_024, 0xAA, 6, 4), image_turn(1_024, 0xBB, 6, 4)]);
    assert_eq!(first.widths, [1024, 16, 4]);
    assert!(other.reused.is_empty(), "{:?}", other.reused);
    assert_eq!(other.widths, first.widths, "it prefills its whole prompt, cut where the first was");
}

#[test]
fn an_image_prompt_ending_at_a_checkpoint_opener_re_prefills_on_flash_next() {
    // A multimodal claim leaves a tail to prefill (ADR 0029 as amended
    // 2026-09-16): only a prefill hands the leaf its rope delta. A prompt
    // ending exactly where a longer one left its checkpoint claims nothing.
    let [first, shorter] = served_on_flash_next([image_turn(1_024, 0xAA, 6, 4), image_turn(1_024, 0xAA, 6, 0)]);
    assert_eq!(first.widths, [1024, 16, 4], "the longer prompt captured at 1,040");
    assert!(shorter.reused.is_empty(), "{:?}", shorter.reused);
    assert_eq!(shorter.widths.iter().sum::<usize>(), 1_040);
}
