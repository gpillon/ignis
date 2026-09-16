//! GPU coverage for prompt checkpoints against the real model (GitHub #186,
//! ADR 0029).
//!
//! The leaf's own test (`kernel/tests/test_seq_checkpoint.cpp`) proves that
//! the bytes move: a claimant's mutable state is the capture's byte for byte,
//! its head addresses the shared physical pages, its own first page carries
//! the copied partial tail, and its penalty-count row is zero. What it cannot
//! say is whether a sequence built that way **generates the same thing** as
//! one that prefilled the whole conversation itself — for that the model has
//! to run and the tokens have to match.
//!
//! The correctness claim (ADR 0029) is bit-exactness against a **cold prefill
//! split at the same boundaries**, not against an unsplit one. The two differ
//! in how their chunks are cut, and a different chunk decomposition can move
//! a near-tie (the #153 effect), which is a property of chunking and not of
//! reuse. So this runs three sequences over turn N+1's prompt:
//!
//!   * a **split control** that prefills it in the three spans the publisher's
//!     own decomposition uses — `[0, page)`, `[page, opener)`, `[opener, end)`
//!     — with no sharing at all. This is what the reuse must match exactly.
//!   * a **reuser** that stands up on turn N's checkpoint and prefills only
//!     `[opener, end)`.
//!   * an **unsplit control** that prefills the whole prompt in one span.
//!     Its divergence from the other two is *recorded, not asserted* — it is
//!     the chunking effect, and the run prints where it first parts company.
//!
//! BF16 is asked for by name: it is the oracle format (ADR 0022), and every
//! correctness check in the profile runs against it.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing GPU or artifact is a **skip**; under the profile it is a **hard
//! failure**. Run via `scripts/gpu-profile.ps1` (stops the reference
//! `ninfer-serve` first — the RTX 5090 is exclusive).

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{CudaDevice, FrontendSet, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b;
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{decode_program_batch, prefill_program};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
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
            decode_program_batch(model, pool, &mut [&mut *sequence])
                .unwrap_or_else(|e| panic!("{label}: decode: {e}")),
        );
    }
    out
}

/// Where two token streams first part company, or `None` when they agree.
fn first_divergence(a: &[i32], b: &[i32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn turn_n_plus_1_reusing_a_checkpoint_generates_what_a_split_cold_prefill_generates() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));

    // Two turns of one conversation, rendered through the real chat template
    // so the opener is the real thing and turn N+1's prompt really is an
    // extension of turn N's head (`crates/artifact/tests/generation_opener.rs`
    // pins that property on CPU; here it is the input).
    let turn_n = [
        ignis_artifact::ChatMessage::text(
            ignis_artifact::Role::System,
            "You are a careful assistant. Answer precisely and at length.",
        ),
        ignis_artifact::ChatMessage::text(
            ignis_artifact::Role::User,
            "Explain what a paged KV cache buys an inference engine that serves many \
             requests at once, and what it costs.",
        ),
    ];
    let mut turn_n_plus_1 = turn_n.to_vec();
    turn_n_plus_1.push(ignis_artifact::ChatMessage::text(
        ignis_artifact::Role::Assistant,
        "It lets a sequence's history be non-contiguous, which is what makes many \
         concurrent sequences fit one pool.",
    ));
    turn_n_plus_1.push(ignis_artifact::ChatMessage::text(
        ignis_artifact::Role::User,
        "Now explain the block table itself, in as much detail as you can.",
    ));

    let render = |messages: &[ignis_artifact::ChatMessage]| {
        frontend
            .chat_template()
            .render_with_thinking_and_tools(
                messages,
                ignis_artifact::ChatRenderOptions::default(),
                Some(&[]),
            )
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

    let prompt_n = render(&turn_n);
    let opener_at = ignis_artifact::ChatTemplate::generation_opener_offset(&prompt_n)
        .expect("turn N's render has a generation opener");
    let head = encode(&prompt_n[..opener_at]);
    let prompt = encode(&render(&turn_n_plus_1));
    assert!(
        prompt.starts_with(&head),
        "turn N's head must be a token prefix of turn N+1's prompt — the whole feature"
    );
    let opener = head.len() as u32;
    let publish_at = (opener / PAGE) * PAGE;
    assert!(publish_at > 0, "the conversation must be at least one page long");
    assert!(
        prompt.len() < (MAX_CONTEXT as usize) - GENERATED,
        "the prompt plus its generation must fit the reservation: {} tokens",
        prompt.len()
    );
    println!(
        "prompt_checkpoint_gpu: turn N head {opener} tokens (publish at {publish_at}), \
         turn N+1 prompt {} tokens",
        prompt.len()
    );

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
    // BF16 by name: the oracle format (ADR 0022).
    let model = load_qwen38_27b(
        &reader,
        &artifact,
        &handles,
        MAX_CONTEXT,
        MAX_CONTEXT,
        ignis_core::KvFormat::Bf16,
    )
    .unwrap_or_else(|e| panic!("load model: {e}"));
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: ignis_core::KvFormat::Bf16,
            kv_page_group_count: 80,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 4,
        },
    )
    .unwrap_or_else(|e| panic!("ignis_seq_pool_create: {e}"));
    let free_at_rest = pool.stats().kv_free_pages;
    let image_bytes = pool
        .checkpoint_image_bytes()
        .unwrap_or_else(|e| panic!("checkpoint image bytes: {e}"));
    println!("prompt_checkpoint_gpu: one checkpoint image is {image_bytes} bytes");
    assert!(image_bytes > 0, "a real pool prices a checkpoint above zero");

    // ---- turn N: prefill to the publish point, publish, walk to the opener,
    //      capture. Exactly the decomposition the scheduler cuts.
    let mut publisher = pool
        .alloc(MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("alloc publisher: {e}"));
    prefill_program(&model, &pool, &mut publisher, &head[..publish_at as usize], 0, None)
        .unwrap_or_else(|e| panic!("publisher head prefill: {e}"));
    let prefix = publisher
        .publish_prefix(publish_at)
        .unwrap_or_else(|e| panic!("publish: {e}"));
    prefill_program(
        &model,
        &pool,
        &mut publisher,
        &head[publish_at as usize..],
        u64::from(publish_at),
        None,
    )
    .unwrap_or_else(|e| panic!("publisher opener prefill: {e}"));
    let checkpoint = publisher
        .capture_checkpoint(opener)
        .unwrap_or_else(|e| panic!("capture: {e}"));
    let stats = checkpoint.stats();
    assert_eq!(stats.tokens, opener, "the checkpoint reaches the opener");
    assert_eq!(stats.pages, publish_at / PAGE, "over the pages below it");
    // Turn N finishes and goes. Only the checkpoint keeps its history alive.
    drop(publisher);
    drop(prefix);
    let free_retained = pool.stats().kv_free_pages;
    assert_eq!(
        free_at_rest - free_retained,
        publish_at / PAGE,
        "the retained checkpoint holds exactly the shared pages, and nothing else"
    );

    // ---- the split control: turn N+1's prompt, cold, cut where turn N cut it.
    let mut control = pool
        .alloc(MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("alloc split control: {e}"));
    for (span, start) in [
        (&prompt[..publish_at as usize], 0u64),
        (&prompt[publish_at as usize..opener as usize], u64::from(publish_at)),
        (&prompt[opener as usize..], u64::from(opener)),
    ] {
        prefill_program(&model, &pool, &mut control, span, start, None)
            .unwrap_or_else(|e| panic!("split control prefill at {start}: {e}"));
    }
    let expected = decode_n(&model, &pool, &mut control, GENERATED, "split control");

    // ---- the reuser: stand up on the checkpoint, prefill only the tail.
    let mut reuser = pool
        .alloc_from_checkpoint(MAX_CONTEXT, &checkpoint)
        .unwrap_or_else(|e| panic!("alloc from checkpoint: {e}"));
    assert_eq!(
        reuser.stats().position,
        u64::from(opener),
        "the reuser stands at the opener before it prefills anything"
    );
    prefill_program(&model, &pool, &mut reuser, &prompt[opener as usize..], u64::from(opener), None)
        .unwrap_or_else(|e| panic!("reuser tail prefill: {e}"));
    let reused = decode_n(&model, &pool, &mut reuser, GENERATED, "reuser");

    assert_eq!(
        reused, expected,
        "reuse must generate exactly what a cold prefill split at the same boundaries does"
    );

    // ---- non-consuming: a second claimant hits the same entry.
    let mut fork = pool
        .alloc_from_checkpoint(MAX_CONTEXT, &checkpoint)
        .unwrap_or_else(|e| panic!("alloc fork from checkpoint: {e}"));
    prefill_program(&model, &pool, &mut fork, &prompt[opener as usize..], u64::from(opener), None)
        .unwrap_or_else(|e| panic!("fork tail prefill: {e}"));
    let forked = decode_n(&model, &pool, &mut fork, GENERATED, "fork");
    assert_eq!(forked, expected, "a second claimant hits the same checkpoint");
    println!(
        "prompt_checkpoint_gpu: claims {}, last claim {:.3} ms",
        checkpoint.stats().claim_count,
        checkpoint.stats().last_claim_micros / 1000.0
    );

    // ---- the unsplit control: information, not a verdict (ADR 0029).
    let mut unsplit = pool
        .alloc(MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("alloc unsplit control: {e}"));
    prefill_program(&model, &pool, &mut unsplit, &prompt, 0, None)
        .unwrap_or_else(|e| panic!("unsplit control prefill: {e}"));
    let whole = decode_n(&model, &pool, &mut unsplit, GENERATED, "unsplit control");
    match first_divergence(&whole, &expected) {
        None => println!(
            "prompt_checkpoint_gpu: the unsplit cold prefill agrees with both split runs \
             for all {GENERATED} tokens"
        ),
        Some(at) => println!(
            "prompt_checkpoint_gpu: the unsplit cold prefill parts company at token {at} \
             ({} vs {}) — the chunk-decomposition effect (#153), recorded not asserted",
            whole[at], expected[at]
        ),
    }

    drop(fork);
    drop(reuser);
    drop(control);
    drop(unsplit);
    drop(checkpoint);
    assert_eq!(
        pool.stats().kv_free_pages,
        free_at_rest,
        "releasing the last holder returns every page"
    );
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_capture_leaves_the_capturing_sequence_generating_what_it_would_have() {
    // The other half of "a capture is a pure read": the request that pays for
    // the chunk must be exactly as well off as one that was never asked to
    // capture. Two identical sequences, one captured against and one not,
    // decode the same tokens.
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let text = "A block table maps a sequence's logical KV pages to physical ones, \
                which is what lets a paged cache place a long history wherever there \
                is room. Describe the trade-offs in as much detail as you can manage. "
        .repeat(3);
    let prompt: Vec<i32> = frontend
        .tokenizer()
        .encode(&text)
        .unwrap_or_else(|e| panic!("tokenize: {e}"))
        .into_iter()
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();
    // An opener 40 tokens into the sequence's own first page: not a page
    // boundary, which is the case that matters.
    let publish_at = 2 * PAGE;
    let opener = publish_at + 40;
    assert!(prompt.len() > opener as usize + 8, "the prompt must reach past the opener");
    assert!(prompt.len() < MAX_CONTEXT as usize - GENERATED);

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
    let model = load_qwen38_27b(
        &reader,
        &artifact,
        &handles,
        MAX_CONTEXT,
        MAX_CONTEXT,
        ignis_core::KvFormat::Bf16,
    )
    .unwrap_or_else(|e| panic!("load model: {e}"));
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: ignis_core::KvFormat::Bf16,
            kv_page_group_count: 80,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 4,
        },
    )
    .unwrap_or_else(|e| panic!("ignis_seq_pool_create: {e}"));

    // One sequence publishes, walks to the opener, captures, and finishes its
    // prompt. The other does exactly the same without the capture.
    let run = |capture: bool, label: &str| -> Vec<i32> {
        let mut seq = pool
            .alloc(MAX_CONTEXT)
            .unwrap_or_else(|e| panic!("{label}: alloc: {e}"));
        prefill_program(&model, &pool, &mut seq, &prompt[..publish_at as usize], 0, None)
            .unwrap_or_else(|e| panic!("{label}: head prefill: {e}"));
        let prefix = seq
            .publish_prefix(publish_at)
            .unwrap_or_else(|e| panic!("{label}: publish: {e}"));
        prefill_program(
            &model,
            &pool,
            &mut seq,
            &prompt[publish_at as usize..opener as usize],
            u64::from(publish_at),
            None,
        )
        .unwrap_or_else(|e| panic!("{label}: opener prefill: {e}"));
        let checkpoint = capture.then(|| {
            seq.capture_checkpoint(opener)
                .unwrap_or_else(|e| panic!("{label}: capture: {e}"))
        });
        prefill_program(&model, &pool, &mut seq, &prompt[opener as usize..], u64::from(opener), None)
            .unwrap_or_else(|e| panic!("{label}: tail prefill: {e}"));
        let out = decode_n(&model, &pool, &mut seq, GENERATED, label);
        drop(checkpoint);
        drop(seq);
        drop(prefix);
        out
    };

    let uncaptured = run(false, "uncaptured");
    let captured = run(true, "captured");
    assert_eq!(
        captured, uncaptured,
        "being captured against must cost the capturing request nothing, not even a token"
    );
}
