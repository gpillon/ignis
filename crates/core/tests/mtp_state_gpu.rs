//! GPU coverage for the 27B MTP head's state travelling with its sequence
//! (GitHub #622, spec #618, ADR 0095): snapshot and restore, prefix clone and
//! prompt checkpoint, on a load built for the head.
//!
//! Every comparison is against a sequence that never moved, over the same
//! prompt cut at the same points, decoding alone through the same verify
//! rounds at the same budgets -- so a difference can only come from the move:
//!
//! - **The snapshot points.** A sequence snapshotted at a completed chunk
//!   boundary, after a committed round with rejected drafts, and after a round
//!   at extent 0, is restored into a fresh slot and goes on: every round after
//!   the restore drafts at the uninterrupted run's extent and commits its run,
//!   token for token, under BF16 KV and under hq-e8-2b (whose ring words, the
//!   target's and the head's, come back with it).
//! - **The prefix's last entry.** A claimant whose next token differs from its
//!   publisher's builds the head's entry at the prefix's last position from
//!   its own: bit for bit the entry of a sequence that prefilled the same
//!   tokens cold, and not the publisher's, which the publisher keeps. Publisher
//!   and claimant draft from their first round exactly as their cold twins do,
//!   and end on their twins' bytes; a second claimant with the publisher's
//!   next token builds the publisher's entry.
//! - **The checkpoint.** A checkpoint captured at an opener on a page boundary
//!   and one inside a page, claimed after its capturer wrote its own entry at
//!   the opener's last position, and restored from its spilled blob: each
//!   claimant drafts as the cold twin does.
//!
//! The head is reached here through the core load API, below the server's
//! MTP load (#623), which `ignis-server`'s `mtp_serving_gpu.rs` serves.
//!
//! Explicit GPU profile (ADR 0006): outside `IGNIS_GPU_PROFILE=1` a missing
//! GPU or artifact is a skip, under it a failure.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{bind_model_scope_27b, materialize, CudaDevice, DraftModule, FrontendSet, MaterializedArtifact, ObjectHandle, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b_with_speculation, Model};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{decode_program_verify_runs, prefill_program_sampled, SamplingParams, VerifyLane};
use ignis_core::{mtp_readout, KvFormat, Speculation, SpeculativeBackend};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// The 27B MTP load's chunk: a multiple of 128 (`mtp_head::PREFILL_ROWS`).
const CHUNK: u32 = 128;
const CONTEXT: u32 = 640;
const WINDOW: u32 = 7;
/// Tokens each sequence emits: several full windows.
const TOTAL: usize = 40;
/// The round every run takes at a budget of one token: extent 0.
const EXTENT_ZERO_ROUND: usize = 2;
const PAGE: usize = 64;
/// Where the shared prefix is published: four whole pages.
const PREFIX: usize = 256;
/// A device retained slot for the prefix, the host one for the checkpoint.
const PREFIX_SLOT: u32 = 0;
const CHECKPOINT_SLOT: u32 = 2;

/// One verify round of one lane: its extent and its committed run.
type Round = (u32, Vec<i32>);

struct OnDevice {
    artifact: MaterializedArtifact,
    device: CudaDevice,
}

impl Drop for OnDevice {
    fn drop(&mut self) {
        let _ = self.artifact.release_arena(&mut self.device);
    }
}

/// The artifact, its text + MTP scope's handles and the scope on the card, or
/// `None` when the test skips.
fn materialized() -> Option<(Reader, Vec<ObjectHandle>, OnDevice)> {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let (plan, handles) =
        bind_model_scope_27b(&reader, Some(DraftModule::Mtp)).unwrap_or_else(|e| panic!("bind with the MTP head: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return None;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => Some((reader, handles, OnDevice { artifact, device })),
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize text + mtp: {e}")) {
                return None;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    }
}

fn load(reader: &Reader, artifact: &MaterializedArtifact, handles: &[ObjectHandle], kv_format: KvFormat) -> Model {
    let spec = Speculation::new(SpeculativeBackend::Mtp, WINDOW).unwrap();
    load_qwen38_27b_with_speculation(reader, artifact, handles, CHUNK, CONTEXT, kv_format, Some(spec))
        .unwrap_or_else(|e| panic!("model load (MTP, {kv_format:?}): {e}"))
}

/// A pool built for the head: five lanes, two device retained slots and a
/// host one, and pages for every lane's whole context and the copies its
/// holders keep.
fn pool(kv_format: KvFormat) -> SeqPool {
    let slots = 5;
    SeqPool::create_with_speculation(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format,
            kv_page_group_count: CONTEXT.div_ceil(64) * slots + 8,
            max_context_tokens: CONTEXT,
            slot_count: slots,
            retained_slot_count: 2,
            retained_host_slot_count: 1,
        },
        Some(SpeculativeBackend::Mtp),
    )
    .unwrap_or_else(|e| panic!("MTP pool create: {e}"))
}

fn encoder(reader: &Reader) -> impl Fn(&str) -> Vec<i32> + '_ {
    let frontend = FrontendSet::from_reader(reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    move |text: &str| -> Vec<i32> {
        frontend
            .tokenizer()
            .encode(text)
            .unwrap_or_else(|e| panic!("tokenize: {e}"))
            .into_iter()
            .map(|id| i32::try_from(id).expect("token id fits i32"))
            .collect()
    }
}

/// A system block of at least `tokens` tokens.
fn block(encode: &dyn Fn(&str) -> Vec<i32>, tokens: usize) -> Vec<i32> {
    let mut block = encode("<|im_start|>system\n");
    while block.len() < tokens {
        block.extend(encode(
            "You are a careful coding assistant. You answer in short, precise sentences and you never invent \
             APIs. When a question is about Rust, you name the crate and the function you rely on. ",
        ));
    }
    block
}

fn user_turn(encode: &dyn Fn(&str) -> Vec<i32>, text: &str) -> Vec<i32> {
    encode(&format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))
}

fn stop_ids(encode: &dyn Fn(&str) -> Vec<i32>) -> Vec<i32> {
    ["<|im_end|>", "<|endoftext|>"].iter().flat_map(|marker| encode(marker)).collect()
}

/// Prefill `tokens[from..to]` at `from`, greedy.
fn prefill(model: &Model, pool: &SeqPool, seq: &mut Seq<'_>, tokens: &[i32], from: usize, to: usize) {
    prefill_program_sampled(model, pool, seq, &tokens[from..to], from as u64, SamplingParams::greedy(), None)
        .unwrap_or_else(|e| panic!("prefill {from}..{to}: {e}"));
}

/// Prefill `tokens` cut at `cuts` (each a position the next call starts at).
fn prefill_cut(model: &Model, pool: &SeqPool, seq: &mut Seq<'_>, tokens: &[i32], cuts: &[usize]) {
    let mut from = 0;
    for &to in cuts.iter().chain(std::iter::once(&tokens.len())) {
        prefill(model, pool, seq, tokens, from, to);
        from = to;
    }
}

/// The rounds of `seq` from round `first`, having emitted `emitted`, until it
/// has emitted [`TOTAL`] tokens or a stop id: each round budgeted to what is
/// left, but round [`EXTENT_ZERO_ROUND`] to one token (extent 0).
fn rounds_from(
    model: &Model,
    pool: &SeqPool,
    seq: &mut Seq<'_>,
    stop_ids: &[i32],
    first: usize,
    mut emitted: usize,
) -> Vec<Round> {
    let mut rounds = Vec::new();
    let mut index = first;
    while emitted < TOTAL {
        let budget = if index == EXTENT_ZERO_ROUND { 1 } else { (TOTAL - emitted) as u32 };
        let lane = VerifyLane { sampling: SamplingParams::greedy(), remaining_tokens: budget, stop_ids, ..VerifyLane::greedy(&[]) };
        let run = decode_program_verify_runs(model, pool, &mut [&mut *seq], &[lane], WINDOW)
            .unwrap_or_else(|e| panic!("verify round {index}: {e}"))
            .remove(0);
        assert_eq!(run.extent, WINDOW.min(budget - 1), "round {index}: the extent its budget allows");
        emitted += run.tokens.len();
        let stopped = run.tokens.iter().any(|t| stop_ids.contains(t));
        rounds.push((run.extent, run.tokens));
        index += 1;
        if stopped {
            break;
        }
    }
    rounds
}

fn emitted(rounds: &[Round]) -> usize {
    rounds.iter().map(|(_, tokens)| tokens.len()).sum()
}

fn accepted(rounds: &[Round]) -> usize {
    rounds.iter().map(|(_, tokens)| tokens.len() - 1).sum()
}

/// `seq`'s ring words, the target's and the head's, under hq-e8-2b; `None`
/// under BF16.
fn rings(pool: &SeqPool, seq: &Seq<'_>, kv_format: KvFormat) -> Option<(Vec<u32>, Vec<u32>)> {
    (kv_format == KvFormat::HqE8_2b).then(|| {
        let (target, head) = mtp_readout::ring_words(pool, seq).unwrap_or_else(|e| panic!("ring words: {e}"));
        (target.to_vec(), head.to_vec())
    })
}

/// A restored sequence's rounds against the uninterrupted run's from the
/// same round: the same extents and runs, the first drafting.
fn assert_goes_on(what: &str, restored: &[Round], uninterrupted: &[Round]) {
    assert!(!restored.is_empty(), "{what}: no round after the restore");
    assert_eq!(restored, uninterrupted, "{what}: the restored sequence's rounds are the uninterrupted run's");
    println!("{what}: {} rounds after the restore, {} drafts accepted", restored.len(), accepted(restored));
}

/// AC 1 and AC 4: the snapshot points, under `kv_format`.
fn the_snapshot_points(kv_format: KvFormat) {
    let Some((reader, handles, on_device)) = materialized() else {
        return;
    };
    let model = load(&reader, &on_device.artifact, &handles, kv_format);
    let pool = pool(kv_format);
    let encode = encoder(&reader);
    let stop = stop_ids(&encode);
    let mut prompt = block(&encode, 160);
    prompt.extend(encode("<|im_end|>\n"));
    prompt.extend(user_turn(&encode, "List five Rust crates for parsing command-line arguments, one per line."));
    assert!(prompt.len() > CHUNK as usize, "a prompt of two chunks");
    let cut = [CHUNK as usize];
    let fmt = format!("{kv_format:?}");

    // The uninterrupted run.
    let mut never = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    prefill_cut(&model, &pool, &mut never, &prompt, &cut);
    let uninterrupted = rounds_from(&model, &pool, &mut never, &stop, 0, 0);
    drop(never);
    assert!(uninterrupted.len() > EXTENT_ZERO_ROUND + 1, "{fmt}: the run outlives its extent-0 round");
    assert!(accepted(&uninterrupted) > 0, "{fmt}: the head landed no draft");

    // At a completed chunk boundary, the prompt's first chunk behind it.
    {
        let mut seq = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        prefill(&model, &pool, &mut seq, &prompt, 0, CHUNK as usize);
        let blob = seq.snapshot().unwrap_or_else(|e| panic!("{fmt}: snapshot at the chunk boundary: {e}"));
        let at_snapshot = rings(&pool, &seq, kv_format);
        drop(seq);
        let mut restored = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        restored.restore(&blob).unwrap_or_else(|e| panic!("{fmt}: restore: {e}"));
        assert_eq!(rings(&pool, &restored, kv_format), at_snapshot, "{fmt}: the ring words come back");
        prefill(&model, &pool, &mut restored, &prompt, CHUNK as usize, prompt.len());
        let rounds = rounds_from(&model, &pool, &mut restored, &stop, 0, 0);
        assert!(rounds[0].0 > 0, "{fmt}: the restored sequence drafts in its first round");
        assert_goes_on(&format!("{fmt}: a snapshot at a chunk boundary"), &rounds, &uninterrupted);
    }

    // After a committed round: one with rejected drafts, and the extent-0 one.
    // A round the next of which drafts: not the one before the extent-0 round.
    let rejected = uninterrupted
        .iter()
        .enumerate()
        .position(|(index, (extent, tokens))| {
            index + 1 != EXTENT_ZERO_ROUND && index + 1 < uninterrupted.len() && *extent > 0 && tokens.len() < *extent as usize + 1
        })
        .unwrap_or_else(|| panic!("{fmt}: no round rejected a draft: {uninterrupted:?}"));
    for (after, label) in [(rejected, "a round with rejected drafts"), (EXTENT_ZERO_ROUND, "a round at extent 0")] {
        let mut seq = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        prefill_cut(&model, &pool, &mut seq, &prompt, &cut);
        let before = rounds_from_to(&model, &pool, &mut seq, &stop, after + 1);
        assert_eq!(before, uninterrupted[..=after], "{fmt}: the rounds before the snapshot");
        let blob = seq.snapshot().unwrap_or_else(|e| panic!("{fmt}: snapshot after {label}: {e}"));
        let at_snapshot = rings(&pool, &seq, kv_format);
        drop(seq);
        let mut restored = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        restored.restore(&blob).unwrap_or_else(|e| panic!("{fmt}: restore: {e}"));
        assert_eq!(rings(&pool, &restored, kv_format), at_snapshot, "{fmt}: the ring words come back after {label}");
        let rounds = rounds_from(&model, &pool, &mut restored, &stop, after + 1, emitted(&before));
        assert_goes_on(&format!("{fmt}: a snapshot after {label} (round {after})"), &rounds, &uninterrupted[after + 1..]);
        assert!(rounds[0].0 > 0, "{fmt}: the restored sequence drafts in its first round after {label}");
    }
}

/// The first `count` rounds of [`rounds_from`] from round 0.
fn rounds_from_to(model: &Model, pool: &SeqPool, seq: &mut Seq<'_>, stop_ids: &[i32], count: usize) -> Vec<Round> {
    let mut rounds = Vec::new();
    let mut emitted = 0;
    for index in 0..count {
        let budget = if index == EXTENT_ZERO_ROUND { 1 } else { (TOTAL - emitted) as u32 };
        let lane = VerifyLane { sampling: SamplingParams::greedy(), remaining_tokens: budget, stop_ids, ..VerifyLane::greedy(&[]) };
        let run = decode_program_verify_runs(model, pool, &mut [&mut *seq], &[lane], WINDOW)
            .unwrap_or_else(|e| panic!("verify round {index}: {e}"))
            .remove(0);
        emitted += run.tokens.len();
        rounds.push((run.extent, run.tokens));
    }
    rounds
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_sequence_restored_at_a_chunk_boundary_or_after_a_committed_round_drafts_on_as_if_never_moved_bf16() {
    the_snapshot_points(KvFormat::Bf16);
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_sequence_restored_at_a_chunk_boundary_or_after_a_committed_round_drafts_on_as_if_never_moved_hq() {
    the_snapshot_points(KvFormat::HqE8_2b);
}

/// The head's row at `position` of `seq`, key then value.
fn head_row(pool: &SeqPool, seq: &Seq<'_>, position: usize) -> Vec<u16> {
    let (mut keys, values) =
        mtp_readout::head_state(pool, seq, position as u32, 1).unwrap_or_else(|e| panic!("head row {position}: {e}"));
    keys.extend(values);
    keys
}

/// AC 2: the claimant's entry at the prefix's last position is its own.
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_claimant_builds_the_heads_last_prefix_entry_from_its_own_next_token_and_drafts_at_once() {
    let Some((reader, handles, on_device)) = materialized() else {
        return;
    };
    let model = load(&reader, &on_device.artifact, &handles, KvFormat::Bf16);
    let pool = pool(KvFormat::Bf16);
    let encode = encoder(&reader);
    let stop = stop_ids(&encode);
    let head: Vec<i32> = block(&encode, PREFIX)[..PREFIX].to_vec();
    // Two tails whose first tokens differ: the token after the prefix's last
    // position, from which the head's entry there is built.
    let with_tail = |tail: &str, question: &str| {
        let mut prompt = head.clone();
        prompt.extend(encode(tail));
        prompt.extend(user_turn(&encode, question));
        prompt
    };
    let publisher_prompt = with_tail("<|im_end|>\n", "Name one Rust crate for HTTP clients and say why.");
    let claimant_prompt = with_tail(" Keep every answer brief.<|im_end|>\n", "Name one Rust crate for JSON and say why.");
    assert_ne!(publisher_prompt[PREFIX], claimant_prompt[PREFIX], "the two next tokens differ");
    let last = PREFIX - 1;

    // The publisher: publishes the prefix, prefills its tail and drafts.
    let mut publisher = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    prefill(&model, &pool, &mut publisher, &publisher_prompt, 0, PREFIX);
    let prefix = publisher.publish_prefix(PREFIX as u32, PREFIX_SLOT).unwrap_or_else(|e| panic!("publish: {e}"));
    prefill(&model, &pool, &mut publisher, &publisher_prompt, PREFIX, publisher_prompt.len());
    let publisher_row = head_row(&pool, &publisher, last);
    let publisher_rounds = rounds_from(&model, &pool, &mut publisher, &stop, 0, 0);

    // Its cold twin.
    let mut twin = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    prefill_cut(&model, &pool, &mut twin, &publisher_prompt, &[PREFIX]);
    assert_eq!(head_row(&pool, &twin, last), publisher_row, "the publisher's entry at {last} is its cold twin's");
    let twin_rounds = rounds_from(&model, &pool, &mut twin, &stop, 0, 0);
    assert_eq!(publisher_rounds, twin_rounds, "publishing leaves the publisher's rounds as they were");
    assert!(publisher_rounds[0].0 > 0 && accepted(&publisher_rounds) > 0, "the publisher drafts");
    assert!(
        publisher.snapshot().unwrap_or_else(|e| panic!("snapshot: {e}")) == twin.snapshot().unwrap_or_else(|e| panic!("snapshot: {e}")),
        "the publisher ends on its cold twin's bytes"
    );
    drop(twin);

    // The claimant, after the publisher wrote its own entry at the last position.
    let mut claimant = pool.alloc_shared(CONTEXT, &prefix).unwrap_or_else(|e| panic!("claim: {e}"));
    assert_eq!(claimant.stats().shared_pages, (PREFIX / PAGE - 1) as u32, "every page but the prefix's last is shared");
    prefill(&model, &pool, &mut claimant, &claimant_prompt, PREFIX, claimant_prompt.len());
    let claimant_row = head_row(&pool, &claimant, last);
    let claimant_rounds = rounds_from(&model, &pool, &mut claimant, &stop, 0, 0);

    let mut cold = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    prefill_cut(&model, &pool, &mut cold, &claimant_prompt, &[PREFIX]);
    assert_eq!(head_row(&pool, &cold, last), claimant_row, "the claimant's entry at {last} is its cold twin's, bit for bit");
    assert_ne!(claimant_row, publisher_row, "and not the publisher's, built from another token");
    assert_eq!(head_row(&pool, &publisher, last), publisher_row, "the publisher keeps its own entry");
    let cold_rounds = rounds_from(&model, &pool, &mut cold, &stop, 0, 0);
    assert!(claimant_rounds[0].0 > 0, "the claimant drafts in its first round");
    assert_eq!(claimant_rounds, cold_rounds, "the claimant's rounds are its cold twin's");
    assert!(accepted(&claimant_rounds) > 0, "the claimant lands drafts");
    assert!(
        claimant.snapshot().unwrap_or_else(|e| panic!("snapshot: {e}")) == cold.snapshot().unwrap_or_else(|e| panic!("snapshot: {e}")),
        "the claimant ends on its cold twin's bytes"
    );
    println!(
        "publisher: {} rounds, {} accepted; claimant: {} rounds, {} accepted",
        publisher_rounds.len(),
        accepted(&publisher_rounds),
        claimant_rounds.len(),
        accepted(&claimant_rounds)
    );
    drop(cold);
    drop(claimant);

    // A second claimant with the publisher's next token builds the
    // publisher's entry: the entry is the token's, whoever wrote before.
    let mut second = pool.alloc_shared(CONTEXT, &prefix).unwrap_or_else(|e| panic!("second claim: {e}"));
    prefill(&model, &pool, &mut second, &publisher_prompt, PREFIX, publisher_prompt.len());
    assert_eq!(head_row(&pool, &second, last), publisher_row, "a claimant with the publisher's token builds its entry");
    assert_eq!(rounds_from(&model, &pool, &mut second, &stop, 0, 0), publisher_rounds, "and drafts as the publisher did");
    drop(second);
    drop(publisher);
    drop(prefix);
}

/// AC 3, the checkpoint: captured at an opener on a page boundary and inside
/// a page, claimed after its capturer wrote its own entry at the opener's
/// last position, and restored from its spilled blob.
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_checkpoints_claimant_and_its_spilled_blob_draft_on_as_a_cold_prefill_does() {
    let Some((reader, handles, on_device)) = materialized() else {
        return;
    };
    let model = load(&reader, &on_device.artifact, &handles, KvFormat::Bf16);
    let pool = pool(KvFormat::Bf16);
    let encode = encoder(&reader);
    let stop = stop_ids(&encode);
    let turn = user_turn(&encode, "Which crate parses TOML?");
    for opener in [320usize, 300] {
        let mut turn_n = block(&encode, opener)[..opener - turn.len()].to_vec();
        turn_n.extend(&turn);
        assert_eq!(turn_n.len(), opener);
        let mut turn_n1 = turn_n.clone();
        turn_n1.extend(encode("The `toml` crate.<|im_end|>\n"));
        turn_n1.extend(user_turn(&encode, "And YAML?"));
        let at = format!("opener {opener}");

        let mut capturer = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        prefill(&model, &pool, &mut capturer, &turn_n, 0, opener);
        let checkpoint =
            capturer.capture_checkpoint(opener as u32, CHECKPOINT_SLOT).unwrap_or_else(|e| panic!("{at}: capture: {e}"));
        // The capturer goes on and writes its own entry at the opener's last
        // position, which no claimant may see.
        let capturer_rounds = rounds_from(&model, &pool, &mut capturer, &stop, 0, 0);
        assert!(accepted(&capturer_rounds) > 0, "{at}: the capturer drafts");

        let mut cold = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        prefill_cut(&model, &pool, &mut cold, &turn_n1, &[opener]);
        let cold_rounds = rounds_from(&model, &pool, &mut cold, &stop, 0, 0);
        let cold_blob = cold.snapshot().unwrap_or_else(|e| panic!("snapshot: {e}"));
        drop(cold);

        let mut claimant =
            pool.alloc_from_checkpoint(CONTEXT, &checkpoint).unwrap_or_else(|e| panic!("{at}: claim: {e}"));
        prefill(&model, &pool, &mut claimant, &turn_n1, opener, turn_n1.len());
        let rounds = rounds_from(&model, &pool, &mut claimant, &stop, 0, 0);
        assert!(rounds[0].0 > 0, "{at}: the claimant drafts in its first round");
        assert_goes_on(&format!("{at}: the checkpoint's claimant"), &rounds, &cold_rounds);
        assert!(
            claimant.snapshot().unwrap_or_else(|e| panic!("snapshot: {e}")) == cold_blob,
            "{at}: the claimant ends on its cold twin's bytes"
        );
        drop(claimant);

        // The checkpoint spilled, and restored into a fresh sequence.
        let mut blob = vec![0u8; checkpoint.snapshot_bytes().unwrap_or_else(|e| panic!("{at}: blob size: {e}")) as usize];
        checkpoint.snapshot_into(&mut blob).unwrap_or_else(|e| panic!("{at}: spill: {e}"));
        let mut restored = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        restored.restore(&blob).unwrap_or_else(|e| panic!("{at}: restore the spilled checkpoint: {e}"));
        prefill(&model, &pool, &mut restored, &turn_n1, opener, turn_n1.len());
        let rounds = rounds_from(&model, &pool, &mut restored, &stop, 0, 0);
        assert_goes_on(&format!("{at}: the spilled checkpoint"), &rounds, &cold_rounds);
        drop(restored);
        drop(checkpoint);
        drop(capturer);
    }
}
