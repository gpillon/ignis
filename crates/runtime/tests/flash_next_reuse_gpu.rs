//! Prompt reuse on Flash-Next, at the leaf seam (spec flash-next/05, GitHub
//! #303): what a reused request computes is what a cold prefill split at the
//! same boundary computes, bit for bit.
//!
//! One leaf per KV format (BF16, the oracle, and hq-e8-2b), and in each a
//! history under the dense threshold (2051 visible tokens) and one past 8K
//! (sparse QSA). Per history, the request a scheduler would run:
//!
//! - **turn N** prefills the history, publishes the whole pages below its
//!   opener as a prefix (the chunk boundary there), prefills the rest and
//!   captures a checkpoint at the opener -- once into a host retained slot,
//!   once into a device one;
//! - **the reference** is a fresh sequence prefilled with the same three
//!   calls plus turn N+1's new tokens, and decoded greedily: its last-position
//!   logits and its tokens are what every reuse below must reproduce;
//! - **reuse**: the checkpoint claimed from either slot; the checkpoint
//!   materialized into the KV-RAM arena and restored; the prefix claimed and
//!   the rest of the history prefilled on it; the prefix materialized and
//!   republished from its blob (the KV-RAM path back to the device); and a
//!   live sequence snapshotted mid-decode, restored, and decoded on.
//!
//! Then the shape Flash-Next serves since GitHub #306: no prefix published at
//! the opener's page floor, the capture handing those pages over as a
//! pages-only link (`the_opener_s_page_rides_the_capture`).
//!
//! The reference runs twice, first on a cold expert cache and last on a warm
//! one: the two must agree too, so neither the claim nor where an expert
//! sits in the cache changes a result (spec flash-next/05, "Determinism").
//! The n-gram context is part of what is checked: a claim or a restore that
//! hashed its first tokens from the wrong context would read other rows.
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`). Explicit GPU profile (ADR 0006): outside
//! `IGNIS_GPU_PROFILE=1` a missing artifact or GPU is a skip, under it a
//! failure. Needs ~41 GB of free RAM: the pinned expert pool, the two host
//! retained slots and the 2 GiB arena.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::flash_next::EngineOptions;
use ignis_core::gpu_profile;
use ignis_artifact::Reader;
use ignis_core::{ArtifactHash, BlobIdentity, DecodeParams, KvFormat};
use ignis_runtime::{DecodeLane, FlashNextLeaf, FlashNextModel, FlashNextSequence, StepLeaf, KV_PAGE_TOKENS};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MAX_CONTEXT: u32 = 16 * 1024;
/// Turn N+1's new tokens, and the tokens decoded after them.
const NEW_TOKENS: usize = 300;
const DECODED: usize = 8;
/// Retained slots: 0 on the device, 1 and 2 on the host.
const DEVICE_SLOT: u32 = 0;
const HOST_SLOTS: [u32; 2] = [1, 2];

fn artifact_path() -> PathBuf {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from);
    dir.join(ARTIFACT_FILE_NAME)
}

fn leaf(kv_format: KvFormat) -> Option<FlashNextLeaf> {
    let path = artifact_path();
    if !path.exists() {
        gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display()));
        return None;
    }
    let options = EngineOptions {
        max_context_tokens: MAX_CONTEXT,
        kv_format,
        retained_device_slots: 1,
        retained_host_slots: 2,
        kv_ram_arena_bytes: 2 << 30,
        ..EngineOptions::default()
    };
    match FlashNextLeaf::open(&path, options) {
        Ok(leaf) => Some(leaf),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("open the Flash-Next leaf: {e}"));
            None
        }
    }
}

/// A deterministic prompt of ordinary token ids (the low vocab is text).
fn prompt(len: usize, salt: u32) -> Vec<u32> {
    (0..len as u32).map(|i| 1000 + (i * 7919 + salt * 104_729) % 60_000).collect()
}

/// What a request ends with: the last prefill's logits, bit patterns, and
/// the tokens greedy decode rounds return after it.
#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    logits: Vec<u32>,
    tokens: Vec<u32>,
}

struct Run<'a> {
    leaf: &'a FlashNextLeaf,
    model: &'a FlashNextModel,
}

impl Run<'_> {
    fn fresh(&self) -> FlashNextSequence {
        self.leaf.allocate_sequence(self.model, MAX_CONTEXT).expect("a free lane")
    }

    /// Prefill `tokens` at `start`, returning the span's last logits.
    fn prefill(&self, seq: &mut FlashNextSequence, tokens: &[u32], start: usize) -> Vec<u32> {
        let mut logits = vec![0f32; self.leaf.vocab(self.model) as usize];
        self.leaf
            .prefill(self.model, seq, tokens, start as u32, DecodeParams::default(), &[], Some(&mut logits), None)
            .unwrap_or_else(|code| panic!("prefill of {} at {start}: leaf code {code}", tokens.len()));
        logits.iter().map(|v| v.to_bits()).collect()
    }

    fn decode(&self, seq: &mut FlashNextSequence, rounds: usize) -> Vec<u32> {
        let lane = DecodeLane { params: DecodeParams::default(), remaining_tokens: 1, stop_ids: &[], permitted: &[] };
        (0..rounds)
            .map(|round| {
                let runs = self
                    .leaf
                    .decode(self.model, &mut [&mut *seq], std::slice::from_ref(&lane))
                    .unwrap_or_else(|code| panic!("decode round {round}: leaf code {code}"));
                runs[0].tokens[0]
            })
            .collect()
    }

    /// Turn N+1's tail on a sequence standing at the opener: its new tokens,
    /// then the decode rounds.
    fn finish(&self, mut seq: FlashNextSequence, new: &[u32], opener: usize) -> Outcome {
        let logits = self.prefill(&mut seq, new, opener);
        let tokens = self.decode(&mut seq, DECODED);
        self.leaf.release_sequence(self.model, seq);
        Outcome { logits, tokens }
    }

    /// Put a blob of `bytes` in the arena and fill it with `write`.
    fn blob(&self, bytes: u64, write: impl FnOnce(&mut [u8]) -> Result<(), i32>) -> ignis_core::seq::ArenaBuffer {
        assert!(self.leaf.host_blob_fits(bytes), "a {bytes}-byte blob fits the arena");
        let mut buf = self.leaf.alloc_snapshot_buf(bytes).unwrap_or_else(|code| panic!("arena span: {code}"));
        write(buf.as_mut()).unwrap_or_else(|code| panic!("blob write: leaf code {code}"));
        buf
    }
}

/// Every reuse path against the cold split prefill, for a history of
/// `history` tokens (not a whole number of pages, so a checkpoint carries a
/// partial tail page).
fn history_reuses_exactly(run: &Run<'_>, history: usize, salt: u32) {
    assert_ne!(history % KV_PAGE_TOKENS as usize, 0, "the opener ends inside a page");
    let (leaf, model) = (run.leaf, run.model);
    let tokens = prompt(history, salt);
    let new = prompt(NEW_TOKENS, salt + 1);
    let whole = history / KV_PAGE_TOKENS as usize * KV_PAGE_TOKENS as usize;
    let label = format!("history {history}");

    // The cold split prefill, on whatever the cache holds now.
    let reference = |run: &Run<'_>| {
        let mut seq = run.fresh();
        run.prefill(&mut seq, &tokens[..whole], 0);
        run.prefill(&mut seq, &tokens[whole..], whole);
        run.finish(seq, &new, history)
    };
    let cold = reference(run);
    assert!(cold.logits.iter().all(|&b| f32::from_bits(b).is_finite()), "{label}: finite logits");

    // Turn N: the prefix at the whole pages, checkpoints at the opener.
    let mut turn = run.fresh();
    run.prefill(&mut turn, &tokens[..whole], 0);
    let prefix = leaf.publish_prefix(model, &mut turn, whole as u32, HOST_SLOTS[0]).expect("publish the prefix");
    run.prefill(&mut turn, &tokens[whole..], whole);
    let on_host = leaf.capture_checkpoint(model, &mut turn, history as u32, HOST_SLOTS[1]).expect("capture (host)");
    let on_device = leaf.capture_checkpoint(model, &mut turn, history as u32, DEVICE_SLOT).expect("capture (device)");
    leaf.release_sequence(model, turn);

    for (slot, checkpoint) in [("host", &on_host), ("device", &on_device)] {
        let (seq, micros) = leaf.allocate_sequence_from_checkpoint(model, MAX_CONTEXT, checkpoint).expect("claim");
        assert_eq!(run.finish(seq, &new, history), cold, "{label}: the checkpoint claimed from a {slot} slot");
        println!("{label}: {slot}-slot claim {micros} us");
    }

    // The checkpoint through KV-RAM: materialized, then restored into a
    // fresh sequence (what a KV-RAM claim does).
    let bytes = leaf.checkpoint_snapshot_bytes(model, &on_host).expect("checkpoint blob size");
    let blob = run.blob(bytes, |dst| leaf.checkpoint_snapshot_into(model, &on_host, dst));
    drop((on_host, on_device));
    let mut seq = run.fresh();
    leaf.restore_sequence(model, &mut seq, &blob).expect("restore the checkpoint blob");
    drop(blob);
    assert_eq!(run.finish(seq, &new, history), cold, "{label}: the checkpoint restored from KV-RAM ({bytes} bytes)");

    // The prefix claimed: the rest of the history prefilled on it.
    let mut seq = leaf.allocate_sequence_shared(model, MAX_CONTEXT, &prefix).expect("claim the prefix");
    run.prefill(&mut seq, &tokens[whole..], whole);
    assert_eq!(run.finish(seq, &new, history), cold, "{label}: the prefix claimed");

    // The prefix through KV-RAM and back to the device: restored into a
    // carrier one page longer and published again from it, as the runtime's
    // `restore_prefix` does, then claimed.
    let bytes = leaf.prefix_snapshot_bytes(model, &prefix).expect("prefix blob size");
    let blob = run.blob(bytes, |dst| leaf.prefix_snapshot_into(model, &prefix, dst));
    leaf.release_prefix(model, prefix);
    let mut carrier = leaf.allocate_sequence(model, whole as u32 + KV_PAGE_TOKENS).expect("a carrier");
    leaf.restore_sequence(model, &mut carrier, &blob).expect("restore the prefix blob");
    drop(blob);
    let prefix = leaf.publish_prefix(model, &mut carrier, whole as u32, HOST_SLOTS[0]).expect("republish");
    leaf.release_sequence(model, carrier);
    let mut seq = leaf.allocate_sequence_shared(model, MAX_CONTEXT, &prefix).expect("claim the republished prefix");
    leaf.release_prefix(model, prefix);
    run.prefill(&mut seq, &tokens[whole..], whole);
    assert_eq!(run.finish(seq, &new, history), cold, "{label}: the prefix back from KV-RAM");

    // A live sequence evicted mid-decode: snapshot, release, restore, decode
    // on. Its tokens continue the reference's.
    let mut seq = run.fresh();
    run.prefill(&mut seq, &tokens[..whole], 0);
    run.prefill(&mut seq, &tokens[whole..], whole);
    run.prefill(&mut seq, &new, history);
    let mut decoded = run.decode(&mut seq, DECODED / 2);
    let bytes = leaf.snapshot_bytes(model, &seq).expect("snapshot size");
    let blob = run.blob(bytes, |dst| leaf.snapshot_into(model, &seq, dst));
    leaf.release_sequence(model, seq);
    let mut seq = run.fresh();
    leaf.restore_sequence(model, &mut seq, &blob).expect("restore the live sequence");
    drop(blob);
    decoded.extend(run.decode(&mut seq, DECODED - DECODED / 2));
    leaf.release_sequence(model, seq);
    assert_eq!(decoded, cold.tokens, "{label}: a sequence restored mid-decode continues as it would have");

    // And the reference again, on the cache all of the above warmed.
    assert_eq!(reference(run), cold, "{label}: a warm expert cache computes what a cold one did");
    let (capacity, used) = leaf.kv_ram_arena_stats();
    assert_eq!(used, 0, "{label}: every blob went back to the {capacity}-byte arena");
}

/// GitHub #306 (ADR 0029 as amended 2026-10-07): the shape Flash-Next now
/// serves. Turn N prefills its history to the opener in one call -- no cut at
/// the opener's page floor -- and the capture hands the whole pages below the
/// opener over as a pages-only link. The split control is two spans,
/// `[0, opener)` and `[opener, end)`, and against it:
///
/// - the capturing sequence goes on exactly as if it had captured nothing:
///   the handover moves who owns its pages, never what they hold;
/// - the checkpoint claimed from a host and a device slot (the second capture
///   stands on the link the first one made), and restored from KV-RAM;
/// - turn N+1, claiming it, captures at its own opener -- a link chained over
///   the link -- and both it and a claimant of *that* continue as its own
///   split control does, `[0, opener)`, `[opener, opener')`, `[opener', end)`.
///
/// The reference runs first and again at the end: at 1,500 tokens it is the
/// first thing a fresh load computes, so the two are a cold and a warm expert
/// cache (spec flash-next/05 acceptance 3). Where the two-span control and the
/// three-span one above part company is printed, not asserted: that is the
/// chunking effect (ADR 0029).
fn the_opener_s_page_rides_the_capture(run: &Run<'_>, history: usize, salt: u32) {
    assert_ne!(history % KV_PAGE_TOKENS as usize, 0, "the opener ends inside a page");
    let (leaf, model) = (run.leaf, run.model);
    let tokens = prompt(history, salt);
    let new = prompt(NEW_TOKENS, salt + 1);
    let label = format!("history {history}, the page rides the capture");

    let reference = |run: &Run<'_>| {
        let mut seq = run.fresh();
        run.prefill(&mut seq, &tokens, 0);
        run.finish(seq, &new, history)
    };
    let cold = reference(run);
    assert!(cold.logits.iter().all(|&b| f32::from_bits(b).is_finite()), "{label}: finite logits");

    let mut turn = run.fresh();
    run.prefill(&mut turn, &tokens, 0);
    let on_host = leaf.capture_checkpoint(model, &mut turn, history as u32, HOST_SLOTS[1]).expect("capture (host)");
    let on_device = leaf.capture_checkpoint(model, &mut turn, history as u32, DEVICE_SLOT).expect("capture (device)");
    assert_eq!(run.finish(turn, &new, history), cold, "{label}: the capturing sequence goes on as if it had not captured");

    for (slot, checkpoint) in [("host", &on_host), ("device", &on_device)] {
        let (seq, _) = leaf.allocate_sequence_from_checkpoint(model, MAX_CONTEXT, checkpoint).expect("claim");
        assert_eq!(run.finish(seq, &new, history), cold, "{label}: the checkpoint claimed from a {slot} slot");
    }
    let bytes = leaf.checkpoint_snapshot_bytes(model, &on_host).expect("checkpoint blob size");
    let blob = run.blob(bytes, |dst| leaf.checkpoint_snapshot_into(model, &on_host, dst));
    let mut seq = run.fresh();
    leaf.restore_sequence(model, &mut seq, &blob).expect("restore the checkpoint blob");
    drop(blob);
    assert_eq!(run.finish(seq, &new, history), cold, "{label}: the checkpoint restored from KV-RAM");

    // Turn N+1: its own opener 10 tokens before the end of `new`, so its
    // capture ends inside a page and hands over the pages it warmed.
    let opener = history + NEW_TOKENS - 10;
    let next = prompt(NEW_TOKENS, salt + 2);
    let mut control = run.fresh();
    run.prefill(&mut control, &tokens, 0);
    run.prefill(&mut control, &new[..NEW_TOKENS - 10], history);
    let control = run.finish(control, &next, opener);
    let (mut turn, _) = leaf.allocate_sequence_from_checkpoint(model, MAX_CONTEXT, &on_device).expect("turn N+1 claims");
    drop((on_host, on_device));
    run.prefill(&mut turn, &new[..NEW_TOKENS - 10], history);
    let second = leaf.capture_checkpoint(model, &mut turn, opener as u32, HOST_SLOTS[1]).expect("turn N+1 captures");
    assert_eq!(run.finish(turn, &next, opener), control, "{label}: turn N+1 goes on as if it had not captured");
    let (seq, _) = leaf.allocate_sequence_from_checkpoint(model, MAX_CONTEXT, &second).expect("turn N+2 claims");
    drop(second);
    assert_eq!(run.finish(seq, &next, opener), control, "{label}: a checkpoint on a link over a link");

    // Information, not a failure (ADR 0029): the three-span split this shape
    // replaced, against the two-span one it serves.
    let whole = history / KV_PAGE_TOKENS as usize * KV_PAGE_TOKENS as usize;
    let mut three = run.fresh();
    run.prefill(&mut three, &tokens[..whole], 0);
    run.prefill(&mut three, &tokens[whole..], whole);
    let three = run.finish(three, &new, history);
    let first_token_apart = three.tokens.iter().zip(&cold.tokens).position(|(a, b)| a != b);
    println!(
        "{label}: two spans against three -- logits {}, first decoded token apart: {first_token_apart:?}",
        if three.logits == cold.logits { "identical" } else { "differ" }
    );
    // And the reference again, on the cache all of the above warmed.
    assert_eq!(reference(run), cold, "{label}: a warm expert cache computes what a cold one did");
    let (capacity, used) = leaf.kv_ram_arena_stats();
    assert_eq!(used, 0, "{label}: every blob went back to the {capacity}-byte arena");
}

fn reuse_is_bit_exact(kv_format: KvFormat) {
    let Some(leaf) = leaf(kv_format) else { return };
    let model = leaf.load_model().expect("load the model");

    // ADR 0029's identity, read off the load: this artifact, this format, no
    // drafter, Flash-Next's own blob layout -- which the 27B's refuses.
    let identity = leaf.blob_identity();
    assert_eq!((identity.kv_format, identity.drafter, identity.layout_version), (kv_format, None, 0x101));
    // The leaf drops its map of the file after open (GitHub #306) and keeps
    // the hash it read then.
    let reader = Reader::open(&artifact_path()).expect("open the artifact");
    assert_eq!(identity.artifact, ArtifactHash::from_bytes(reader.content_hash()), "the artifact's own hash");
    drop(reader);
    let a_27b_blob = BlobIdentity { layout_version: 5, ..identity };
    assert!(identity.accepts(&a_27b_blob).is_err(), "a 27B-layout blob is refused");

    let run = Run { leaf: &leaf, model: &model };
    // First, on the cold expert cache a fresh load has: the shape Flash-Next
    // serves checks its reference there and again on the cache it warmed.
    the_opener_s_page_rides_the_capture(&run, 1500, 31);
    history_reuses_exactly(&run, 1500, 11);
    history_reuses_exactly(&run, 9000, 23);
    the_opener_s_page_rides_the_capture(&run, 9000, 41);
    leaf.release_model(model);
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn reuse_is_bit_exact_against_a_split_cold_prefill_hq_e8_2b() {
    reuse_is_bit_exact(KvFormat::HqE8_2b);
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn reuse_is_bit_exact_against_a_split_cold_prefill_bf16() {
    reuse_is_bit_exact(KvFormat::Bf16);
}
