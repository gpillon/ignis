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
//! It then does it again one turn on (GitHub #187). Turn N+1 stands at turn
//! N's opener with its history shared, so its *own* opener's whole pages are
//! not the ones it shares and the leaf refuses to capture there — until it
//! publishes a **chained** prefix at its own opener's page floor, the pages it
//! warmed itself over the ones it claimed. The same claim then has to hold for
//! turn N+2 standing on *that*: what it generates is what a cold prefill split
//! at the same boundaries generates.
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
    let mut kv_ram_blob = vec![
        0u8;
        usize::try_from(
            checkpoint
                .snapshot_bytes()
                .unwrap_or_else(|e| panic!("checkpoint snapshot size: {e}"))
        )
        .expect("snapshot size fits usize")
    ];
    checkpoint
        .snapshot_into(&mut kv_ram_blob)
        .unwrap_or_else(|e| panic!("checkpoint snapshot: {e}"));
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

    // ---- KV-RAM: restore the materialized host blob after all it shares
    //      was copied into it, then take exactly the same tail as the device
    //      checkpoint claim. It must be the same state, not merely a prompt
    //      that happens to produce plausible text.
    let mut host_restored = pool
        .alloc(MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("alloc KV-RAM restore target: {e}"));
    host_restored
        .restore(&kv_ram_blob)
        .unwrap_or_else(|e| panic!("restore materialized checkpoint: {e}"));
    assert_eq!(host_restored.stats().position, u64::from(opener));
    prefill_program(
        &model,
        &pool,
        &mut host_restored,
        &prompt[opener as usize..],
        u64::from(opener),
        None,
    )
    .unwrap_or_else(|e| panic!("KV-RAM reuser tail prefill: {e}"));
    let host_reused = decode_n(&model, &pool, &mut host_restored, GENERATED, "KV-RAM reuser");
    assert_eq!(
        host_reused, expected,
        "KV-RAM restore must generate exactly what the device checkpoint and split-cold control do"
    );
    drop(host_restored);

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

    // ---- GitHub #187: turn N+1 takes a checkpoint of its own ---------------
    //
    // The reuser above stands at turn N's opener with `publish_at / PAGE`
    // pages shared, and its *own* opener is further along — so the whole pages
    // below it are not the ones it shares, and the leaf refuses to capture
    // there. It publishes a **chained** prefix at its own opener's page floor
    // first: the pages it warmed itself, over the ones it claimed. From there
    // the capture is the same capture as any other.
    //
    // What must then be true is the same claim one turn on: turn N+2, standing
    // on that chained checkpoint, generates exactly what a cold prefill of its
    // prompt split at the same boundaries generates.
    let prompt_n_plus_1 = render(&turn_n_plus_1);
    let opener_2 = encode(
        &prompt_n_plus_1[..ignis_artifact::ChatTemplate::generation_opener_offset(&prompt_n_plus_1)
            .expect("turn N+1's render has a generation opener")],
    )
    .len() as u32;
    let publish_at_2 = (opener_2 / PAGE) * PAGE;
    assert!(
        publish_at_2 > publish_at,
        "the fixture must grow past a whole KV page between turns, or the chained publish \
         this section exists to exercise never happens (turn N+1's opener page floor is \
         {publish_at_2}, turn N's is {publish_at}) — lengthen the messages"
    );
    let mut turn_n_plus_2 = turn_n_plus_1.clone();
    turn_n_plus_2.push(ignis_artifact::ChatMessage::text(
        ignis_artifact::Role::Assistant,
        "The block table is one row per sequence: logical page index in, physical page id \
         out, read by every attention kernel.",
    ));
    turn_n_plus_2.push(ignis_artifact::ChatMessage::text(
        ignis_artifact::Role::User,
        "And what happens to that row when two sequences share a prompt head?",
    ));
    let prompt3 = encode(&render(&turn_n_plus_2));
    assert!(
        prompt3.starts_with(&prompt[..opener_2 as usize]),
        "turn N+1's head must be a token prefix of turn N+2's prompt"
    );
    assert!(
        prompt3.len() < (MAX_CONTEXT as usize) - GENERATED,
        "turn N+2's prompt plus its generation must fit the reservation: {} tokens",
        prompt3.len()
    );
    println!(
        "prompt_checkpoint_gpu: turn N+1 opener {opener_2} (chained publish at {publish_at_2}), \
         turn N+2 prompt {} tokens",
        prompt3.len()
    );

    // Turn N+1, decomposed the way the scheduler decomposes it: claim, prefill
    // to its own opener's page floor, publish the chain, prefill to the opener,
    // capture, then the rest.
    let mut turn2 = pool
        .alloc_from_checkpoint(MAX_CONTEXT, &checkpoint)
        .unwrap_or_else(|e| panic!("alloc turn N+1 from checkpoint: {e}"));
    prefill_program(
        &model,
        &pool,
        &mut turn2,
        &prompt[opener as usize..publish_at_2 as usize],
        u64::from(opener),
        None,
    )
    .unwrap_or_else(|e| panic!("turn N+1 prefill to the chained publish point: {e}"));
    let chained = turn2
        .publish_prefix(publish_at_2)
        .unwrap_or_else(|e| panic!("chained publish: {e}"));
    assert_eq!(
        chained.stats().pages,
        (publish_at_2 - publish_at) / PAGE,
        "the chained entry owns only the pages turn N+1 warmed itself"
    );
    assert_eq!(
        chained.stats().tokens,
        publish_at_2,
        "and covers the whole head, its parent's pages included"
    );
    prefill_program(
        &model,
        &pool,
        &mut turn2,
        &prompt[publish_at_2 as usize..opener_2 as usize],
        u64::from(publish_at_2),
        None,
    )
    .unwrap_or_else(|e| panic!("turn N+1 prefill to its own opener: {e}"));
    let checkpoint_2 = turn2
        .capture_checkpoint(opener_2)
        .unwrap_or_else(|e| panic!("turn N+1 capture (the refusal #187 removed): {e}"));
    assert_eq!(checkpoint_2.stats().tokens, opener_2);
    assert_eq!(
        checkpoint_2.stats().pages,
        publish_at_2 / PAGE,
        "the second checkpoint reaches over the whole chain below its opener"
    );
    prefill_program(
        &model,
        &pool,
        &mut turn2,
        &prompt[opener_2 as usize..],
        u64::from(opener_2),
        None,
    )
    .unwrap_or_else(|e| panic!("turn N+1 tail prefill: {e}"));
    drop(turn2);
    assert_eq!(
        free_at_rest - pool.stats().kv_free_pages,
        publish_at_2 / PAGE,
        "two retained checkpoints over one chain hold each page once"
    );

    // The split control for turn N+2: its prompt, cold, cut at every boundary
    // the reuse path cut it at.
    let mut control3 = pool
        .alloc(MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("alloc turn N+2 split control: {e}"));
    for (span, start) in [
        (&prompt3[..publish_at as usize], 0u64),
        (&prompt3[publish_at as usize..opener as usize], u64::from(publish_at)),
        (&prompt3[opener as usize..publish_at_2 as usize], u64::from(opener)),
        (&prompt3[publish_at_2 as usize..opener_2 as usize], u64::from(publish_at_2)),
        (&prompt3[opener_2 as usize..], u64::from(opener_2)),
    ] {
        prefill_program(&model, &pool, &mut control3, span, start, None)
            .unwrap_or_else(|e| panic!("turn N+2 split control prefill at {start}: {e}"));
    }
    let expected3 = decode_n(&model, &pool, &mut control3, GENERATED, "turn N+2 split control");

    let mut turn3 = pool
        .alloc_from_checkpoint(MAX_CONTEXT, &checkpoint_2)
        .unwrap_or_else(|e| panic!("alloc turn N+2 from the chained checkpoint: {e}"));
    assert_eq!(
        turn3.stats().position,
        u64::from(opener_2),
        "turn N+2 stands at turn N+1's opener before it prefills anything"
    );
    prefill_program(
        &model,
        &pool,
        &mut turn3,
        &prompt3[opener_2 as usize..],
        u64::from(opener_2),
        None,
    )
    .unwrap_or_else(|e| panic!("turn N+2 tail prefill: {e}"));
    let reused3 = decode_n(&model, &pool, &mut turn3, GENERATED, "turn N+2 reuser");
    assert_eq!(
        reused3, expected3,
        "a claimant of a chained checkpoint must generate exactly what a cold prefill split \
         at the same boundaries does"
    );
    drop(turn3);
    drop(control3);
    drop(checkpoint_2);
    drop(chained);

    drop(checkpoint);
    assert_eq!(
        pool.stats().kv_free_pages,
        free_at_rest,
        "releasing the last holder returns every page"
    );

    // ---- the other half of "a capture is a pure read" ----------------------
    //
    // Everything above is about what a *claimant* receives. This is about what
    // the capturing request is left with: it paid for the chunk, and being
    // captured against must cost it nothing — not a page, not a token. Two
    // identical sequences over the same conversation, one captured against and
    // one not, must decode the same tokens.
    //
    // It shares this test's model and pool deliberately. One `#[test]` per
    // binary that loads the 27B artifact is not a style choice: two loads in
    // one process do not fit on the card, and the second fails at
    // `ignis_model_load` with a scratch reservation it cannot make.
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
        let taken = capture.then(|| {
            seq.capture_checkpoint(opener)
                .unwrap_or_else(|e| panic!("{label}: capture: {e}"))
        });
        prefill_program(
            &model,
            &pool,
            &mut seq,
            &prompt[opener as usize..],
            u64::from(opener),
            None,
        )
        .unwrap_or_else(|e| panic!("{label}: tail prefill: {e}"));
        let out = decode_n(&model, &pool, &mut seq, GENERATED, label);
        drop(taken);
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
    assert_eq!(
        pool.stats().kv_free_pages,
        free_at_rest,
        "and both twins gave every page back"
    );
}
