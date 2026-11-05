//! GitHub #508 (spec #503, ADR 0029 as amended 2026-11-05): prompt
//! checkpoints with the generation opener's page **riding the capture**, the
//! 27B's policy on Ampere and Ada, against the real model.
//!
//! `prompt_checkpoint_gpu.rs` holds the Blackwell policy: turn N publishes an
//! imaged prefix at its opener's page floor and captures at the opener, two
//! cuts. Here turn N publishes nothing: it prefills to the opener in one span
//! and the capture lends the whole pages below the opener to a pages-only link
//! (GitHub #306), so the prefill is cut once. What must then hold is the same
//! claim (ADR 0029): a reused prompt resumes at the opener and generates
//! exactly what a cold prefill **split at the same boundaries** generates, now
//! `[0, opener)` and `[opener, end)`:
//!
//!   * a **reuser** standing on turn N's checkpoint, on the device, a second
//!     claimant of it, and a KV-RAM restore of it, each against the one-cut
//!     split control, the state after the tail byte for byte;
//!   * one turn on, turn N+1 standing on that checkpoint takes no prefix of its
//!     own either: it prefills to its own opener in one span and its capture
//!     lends the pages it warmed, chained over the first link; turn N+2,
//!     standing on that, against a cold prefill cut at both openers;
//!   * the capturing request itself, lent from or not, decodes the same tokens.
//!
//! An unsplit cold prefill is printed beside them: where it parts company is
//! the chunking effect, recorded and not asserted.
//!
//! The 16 GB trellis artifact, the 27B an RTX 30 or RTX 40 card loads, in the
//! FP16 prompt band, so the test runs on every card the policy serves; BF16,
//! the oracle format, then hq-e8-2b, whose residual window rides every image.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing GPU or artifact is a **skip**; under the profile it is a **hard
//! failure**. Its own test binary: the card holds one 27B load at a time.

#![cfg(feature = "cuda")]

#[path = "support/snapshot_blob.rs"]
mod snapshot_blob;

use std::path::Path;

use ignis_artifact::{
    bind_text_scope_27b, materialize, CudaDevice, FrontendSet, MaterializedArtifact, ObjectHandle, Reader,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b_with_prefill_attention;
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{decode_program_batch, prefill_program};
use ignis_core::{KvFormat, PrefillAttention, RopeScaling};

use snapshot_blob::hq_window;

const TRELLIS: &str = "F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/qwen3_8_27b_trellis_sc30-v2.ninfer";
const MAX_CONTEXT: u32 = 1024;
/// The leaf's KV page, in tokens.
const PAGE: u32 = 64;
/// Decoded from each sequence. Enough that a stale GDN slot, a missing
/// penalty-count row or a mis-copied tail page would have shown by the end.
const GENERATED: usize = 8;

fn decode_n(
    model: &ignis_core::model_load::Model,
    pool: &SeqPool,
    sequence: &mut Seq<'_>,
    count: usize,
    label: &str,
) -> Vec<i32> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.extend(
            decode_program_batch(model, pool, &mut [&mut *sequence]).unwrap_or_else(|e| panic!("{label}: decode: {e}")),
        );
    }
    out
}

/// Where two token streams first part company, or `None` when they agree.
fn first_divergence(a: &[i32], b: &[i32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// BF16, then hq-e8-2b, over one materialized artifact.
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_reused_prompt_resumes_at_the_opener_with_one_cut_as_a_split_cold_prefill_does() {
    let path = Path::new(TRELLIS);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {TRELLIS}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    for kv_format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
        one_cut_turns(&reader, &frontend, &artifact, &handles, kv_format);
    }
}

/// Three turns of one conversation, rendered through the real chat template,
/// so each opener is the real thing and each turn's prompt extends the head
/// of the one before (the conversation `prompt_checkpoint_gpu.rs` runs).
fn conversation() -> [Vec<ignis_artifact::ChatMessage>; 3] {
    use ignis_artifact::{ChatMessage, Role};
    let turn_n = vec![
        ChatMessage::text(Role::System, "You are a careful assistant. Answer precisely and at length."),
        ChatMessage::text(
            Role::User,
            "Explain what a paged KV cache buys an inference engine that serves many requests at once, and what it \
             costs.",
        ),
    ];
    let mut turn_n_plus_1 = turn_n.clone();
    turn_n_plus_1.push(ChatMessage::text(
        Role::Assistant,
        "It lets a sequence's history be non-contiguous, which is what makes many concurrent sequences fit one pool.",
    ));
    turn_n_plus_1.push(ChatMessage::text(Role::User, "Now explain the block table itself, in as much detail as you can."));
    let mut turn_n_plus_2 = turn_n_plus_1.clone();
    turn_n_plus_2.push(ChatMessage::text(
        Role::Assistant,
        "The block table is one row per sequence: logical page index in, physical page id out, read by every \
         attention kernel.",
    ));
    turn_n_plus_2.push(ChatMessage::text(Role::User, "And what happens to that row when two sequences share a prompt head?"));
    [turn_n, turn_n_plus_1, turn_n_plus_2]
}

/// One format's turns against the one-cut split controls.
fn one_cut_turns(
    reader: &Reader,
    frontend: &FrontendSet,
    artifact: &MaterializedArtifact,
    handles: &[ObjectHandle],
    kv_format: KvFormat,
) {
    let hq = kv_format == KvFormat::HqE8_2b;
    println!("prompt_checkpoint_one_cut_gpu: {kv_format:?}");
    let render = |messages: &[ignis_artifact::ChatMessage]| {
        frontend
            .chat_template()
            .render_with_thinking_and_tools(messages, ignis_artifact::ChatRenderOptions::default(), Some(&[]))
            .unwrap_or_else(|e| panic!("render: {e}"))
    };
    let encode = |text: &str| -> Vec<i32> {
        frontend
            .tokenizer()
            .encode(text)
            .unwrap_or_else(|e| panic!("tokenize: {e}"))
            .into_iter()
            .map(|id| i32::try_from(id).expect("token id fits i32"))
            .collect()
    };
    let opener_of = |messages: &[ignis_artifact::ChatMessage]| -> u32 {
        let rendered = render(messages);
        let at = ignis_artifact::ChatTemplate::generation_opener_offset(&rendered).expect("the render has a generation opener");
        encode(&rendered[..at]).len() as u32
    };
    let [turn_n, turn_n_plus_1, turn_n_plus_2] = conversation();
    let opener = opener_of(&turn_n);
    let opener_2 = opener_of(&turn_n_plus_1);
    let prompt = encode(&render(&turn_n_plus_1));
    let prompt3 = encode(&render(&turn_n_plus_2));
    let floor = (opener / PAGE) * PAGE;
    let floor_2 = (opener_2 / PAGE) * PAGE;
    assert!(floor > 0, "the conversation must be at least one page long");
    assert!(floor_2 > floor, "turn N+1 must own whole pages of its own below its opener: floors {floor} and {floor_2}");
    assert!(prompt3.starts_with(&prompt[..opener_2 as usize]), "turn N+1's head must be a token prefix of turn N+2's prompt");
    assert!(prompt3.len() < (MAX_CONTEXT as usize) - GENERATED, "turn N+2 must fit the reservation: {} tokens", prompt3.len());
    println!(
        "prompt_checkpoint_one_cut_gpu: openers {opener} (floor {floor}) and {opener_2} (floor {floor_2}); prompts {} \
         and {} tokens",
        prompt.len(),
        prompt3.len()
    );

    let model = load_qwen38_27b_with_prefill_attention(
        reader,
        artifact,
        handles,
        MAX_CONTEXT,
        MAX_CONTEXT,
        kv_format,
        None,
        None,
        RopeScaling::NONE,
        PrefillAttention::Fp16,
    )
    .unwrap_or_else(|e| panic!("load model: {e}"));
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format,
            kv_page_group_count: 80,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 4,
            // Turn N's checkpoint, turn N+1's, and the captured twin's: no
            // prefix image takes one, since none is published.
            retained_slot_count: 3,
            retained_host_slot_count: 0,
        },
    )
    .unwrap_or_else(|e| panic!("ignis_seq_pool_create: {e}"));
    let free_at_rest = pool.stats().kv_free_pages;
    // A checkpoint keeps the page its opener ends inside as a page of the pool.
    let tail_page = |opener: u32| u32::from(opener % PAGE != 0);

    // ---- turn N: one span to the opener, capture. Nothing is published.
    let mut publisher = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc publisher: {e}"));
    prefill_program(&model, &pool, &mut publisher, &prompt[..opener as usize], 0, None)
        .unwrap_or_else(|e| panic!("publisher prefill to the opener: {e}"));
    let window_at_opener = hq.then(|| {
        let blob = publisher.snapshot().unwrap_or_else(|e| panic!("publisher snapshot: {e:?}"));
        hq_window(&blob).to_vec()
    });
    let checkpoint = publisher.capture_checkpoint(opener, 0).unwrap_or_else(|e| panic!("capture: {e}"));
    assert_eq!(checkpoint.lent_by(&publisher), floor / PAGE, "the capture lends every whole page below the opener");
    assert_eq!(checkpoint.stats().tokens, opener, "the checkpoint reaches the opener");
    assert_eq!(checkpoint.stats().pages, floor / PAGE, "over the pages below it");
    let mut kv_ram_blob = vec![
        0u8;
        usize::try_from(checkpoint.snapshot_bytes().unwrap_or_else(|e| panic!("checkpoint snapshot size: {e}")))
            .expect("snapshot size fits usize")
    ];
    checkpoint.snapshot_into(&mut kv_ram_blob).unwrap_or_else(|e| panic!("checkpoint snapshot: {e}"));
    // Turn N goes; the link it lent to keeps its pages for the checkpoint.
    drop(publisher);
    assert_eq!(
        free_at_rest - pool.stats().kv_free_pages,
        floor / PAGE + tail_page(opener),
        "the retained checkpoint holds the lent pages and its own tail page, and nothing else"
    );

    // ---- the split control: turn N+1's prompt, cold, cut where turn N cut it.
    let mut control = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc split control: {e}"));
    for (span, start) in [(&prompt[..opener as usize], 0u64), (&prompt[opener as usize..], u64::from(opener))] {
        prefill_program(&model, &pool, &mut control, span, start, None)
            .unwrap_or_else(|e| panic!("split control prefill at {start}: {e}"));
    }
    let control_after_tail = control.snapshot().unwrap_or_else(|e| panic!("control snapshot: {e:?}"));
    let expected = decode_n(&model, &pool, &mut control, GENERATED, "split control");
    drop(control);

    // ---- the reuser, a second claimant and a KV-RAM restore: each resumes at
    //      the opener and prefills only the tail.
    let tail = |sequence: &mut Seq<'_>, label: &str| -> Vec<i32> {
        assert_eq!(sequence.stats().position, u64::from(opener), "{label} stands at the opener before it prefills anything");
        prefill_program(&model, &pool, sequence, &prompt[opener as usize..], u64::from(opener), None)
            .unwrap_or_else(|e| panic!("{label}: tail prefill: {e}"));
        assert!(
            sequence.snapshot().unwrap_or_else(|e| panic!("{label}: snapshot: {e:?}")) == control_after_tail,
            "{label} after its tail is the split control after the same tail, byte for byte"
        );
        decode_n(&model, &pool, sequence, GENERATED, label)
    };
    let mut reuser = pool.alloc_from_checkpoint(MAX_CONTEXT, &checkpoint).unwrap_or_else(|e| panic!("alloc from checkpoint: {e}"));
    if let Some(window) = &window_at_opener {
        let blob = reuser.snapshot().unwrap_or_else(|e| panic!("reuser snapshot: {e:?}"));
        assert!(hq_window(&blob) == window.as_slice(), "the claimant's slot holds the capture's window at the opener");
    }
    assert_eq!(tail(&mut reuser, "reuser"), expected, "reuse generates exactly what the one-cut split cold prefill does");
    drop(reuser);
    let mut fork = pool.alloc_from_checkpoint(MAX_CONTEXT, &checkpoint).unwrap_or_else(|e| panic!("alloc fork: {e}"));
    assert_eq!(tail(&mut fork, "fork"), expected, "a second claimant hits the same checkpoint");
    drop(fork);
    let mut restored = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc KV-RAM restore target: {e}"));
    restored.restore(&kv_ram_blob).unwrap_or_else(|e| panic!("restore the materialized checkpoint: {e}"));
    assert_eq!(tail(&mut restored, "KV-RAM restore"), expected, "a KV-RAM restore generates what the device claim does");
    drop(restored);

    // ---- the unsplit control: information, not a verdict (ADR 0029).
    let mut unsplit = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc unsplit control: {e}"));
    prefill_program(&model, &pool, &mut unsplit, &prompt, 0, None).unwrap_or_else(|e| panic!("unsplit prefill: {e}"));
    let whole = decode_n(&model, &pool, &mut unsplit, GENERATED, "unsplit control");
    drop(unsplit);
    match first_divergence(&whole, &expected) {
        None => println!("prompt_checkpoint_one_cut_gpu: the unsplit cold prefill agrees for all {GENERATED} tokens"),
        Some(at) => println!(
            "prompt_checkpoint_one_cut_gpu: the unsplit cold prefill parts company at token {at} ({} vs {}), the \
             chunk-decomposition effect, recorded not asserted",
            whole[at], expected[at]
        ),
    }

    // ---- turn N+1 on the checkpoint: one span to its own opener, capture;
    //      the pages it warmed ride its capture, chained over turn N's link.
    let mut turn2 = pool.alloc_from_checkpoint(MAX_CONTEXT, &checkpoint).unwrap_or_else(|e| panic!("alloc turn N+1: {e}"));
    prefill_program(&model, &pool, &mut turn2, &prompt[opener as usize..opener_2 as usize], u64::from(opener), None)
        .unwrap_or_else(|e| panic!("turn N+1 prefill to its opener: {e}"));
    let checkpoint_2 = turn2.capture_checkpoint(opener_2, 1).unwrap_or_else(|e| panic!("turn N+1 capture: {e}"));
    assert_eq!(checkpoint_2.stats().tokens, opener_2);
    assert_eq!(checkpoint_2.stats().pages, floor_2 / PAGE, "the second checkpoint reaches over both links");
    assert_eq!(checkpoint_2.lent_by(&turn2), (floor_2 - floor) / PAGE, "it lends only the pages turn N+1 warmed itself");
    prefill_program(&model, &pool, &mut turn2, &prompt[opener_2 as usize..], u64::from(opener_2), None)
        .unwrap_or_else(|e| panic!("turn N+1 tail prefill: {e}"));
    drop(turn2);
    assert_eq!(
        free_at_rest - pool.stats().kv_free_pages,
        floor_2 / PAGE + tail_page(opener) + tail_page(opener_2),
        "two retained checkpoints over one chain of links hold each page once, and a tail page each"
    );

    // ---- turn N+2 on the second checkpoint, against a cold prefill cut at
    //      both openers.
    let mut control3 = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc turn N+2 split control: {e}"));
    for (span, start) in [
        (&prompt3[..opener as usize], 0u64),
        (&prompt3[opener as usize..opener_2 as usize], u64::from(opener)),
        (&prompt3[opener_2 as usize..], u64::from(opener_2)),
    ] {
        prefill_program(&model, &pool, &mut control3, span, start, None)
            .unwrap_or_else(|e| panic!("turn N+2 split control prefill at {start}: {e}"));
    }
    let expected3 = decode_n(&model, &pool, &mut control3, GENERATED, "turn N+2 split control");
    drop(control3);
    let mut turn3 =
        pool.alloc_from_checkpoint(MAX_CONTEXT, &checkpoint_2).unwrap_or_else(|e| panic!("alloc turn N+2 from the checkpoint: {e}"));
    assert_eq!(turn3.stats().position, u64::from(opener_2), "turn N+2 stands at turn N+1's opener");
    prefill_program(&model, &pool, &mut turn3, &prompt3[opener_2 as usize..], u64::from(opener_2), None)
        .unwrap_or_else(|e| panic!("turn N+2 tail prefill: {e}"));
    let reused3 = decode_n(&model, &pool, &mut turn3, GENERATED, "turn N+2 reuser");
    assert_eq!(reused3, expected3, "a claimant of the chained checkpoint generates what the split cold prefill does");
    drop(turn3);
    drop(checkpoint_2);
    drop(checkpoint);
    assert_eq!(pool.stats().kv_free_pages, free_at_rest, "releasing the last holder returns every page");

    // ---- the capturing request: lent from or not, it decodes the same.
    let run = |capture: bool, label: &str| -> Vec<i32> {
        let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("{label}: alloc: {e}"));
        prefill_program(&model, &pool, &mut seq, &prompt[..opener as usize], 0, None)
            .unwrap_or_else(|e| panic!("{label}: prefill to the opener: {e}"));
        let taken = capture.then(|| seq.capture_checkpoint(opener, 2).unwrap_or_else(|e| panic!("{label}: capture: {e}")));
        prefill_program(&model, &pool, &mut seq, &prompt[opener as usize..], u64::from(opener), None)
            .unwrap_or_else(|e| panic!("{label}: tail prefill: {e}"));
        let out = decode_n(&model, &pool, &mut seq, GENERATED, label);
        drop(seq);
        drop(taken);
        out
    };
    let uncaptured = run(false, "uncaptured");
    assert_eq!(run(true, "captured"), uncaptured, "lending its pages at the capture costs the request nothing, not a token");
    assert_eq!(pool.stats().kv_free_pages, free_at_rest, "and both twins gave every page back");
}
