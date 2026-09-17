//! GPU coverage for **retained prefixes** against the real model (GitHub
//! #188, ADR 0029).
//!
//! `prefix_reuse_gpu.rs` proves that a claimant decodes what a sibling that
//! prefilled the head decodes — with the publisher *still alive* beside it.
//! What #188 adds is that the publisher is **gone**: a burst's members do not
//! overlap in time, so the prefix a second subagent claims is one whose last
//! live holder released long ago and which only the retention keeps on the
//! card. The pages, the block-table rows and the cloned mutable state have to
//! be as good then as they were while their publisher was serving; if a page
//! were returned to the pool and handed out again, or the clone taken from a
//! sequence that had already been torn down, the reuse would decode plausible
//! nonsense and nothing else in the stack would say so.
//!
//! The correctness claim (ADR 0029) is bit-exactness against a **cold prefill
//! split at the same boundary**, not against an unsplit one: the two cut their
//! chunks differently, and a different decomposition can move a near-tie (the
//! #153 effect), which is a property of chunking and not of reuse. So this
//! runs, over one burst of two subagents that share a system block:
//!
//!   * a **split control** that prefills subagent 2's whole prompt cold, in
//!     the two spans the publisher's own decomposition uses — `[0, block)` and
//!     `[block, end)` — with no sharing at all. This is what the reuse must
//!     match exactly.
//!   * the **reuser**, allocated against the retained prefix after its
//!     publisher's sequence has been dropped, prefilling only `[block, end)`.
//!   * an **unsplit control** over the same prompt in one span. Its divergence
//!     is *recorded, not asserted* — the chunking effect.
//!
//! It also drives the accounting the retention rests on: the prefix's pages
//! stay out of the pool while nothing live holds it, and come back in full
//! when the retention is finally released.
//!
//! BF16 by name: the oracle format (ADR 0022).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing GPU or artifact is a **skip**; under the profile it is a **hard
//! failure**. Run via `scripts/gpu-profile.ps1` (stops the reference
//! `ninfer-serve` first — the RTX 5090 is exclusive).

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{
    ChatMessage, ChatRenderOptions, ChatTemplate, CudaDevice, FrontendSet, Reader, Role,
    bind_text_scope_27b, materialize,
};
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
/// penalty-count row or a mis-addressed page would have shown by the end.
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
fn a_claimant_of_a_retained_prefix_generates_what_a_split_cold_prefill_generates() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));

    // Two subagents of one burst, rendered through the real chat template so
    // the boundary is the real thing and the block really is the whole of what
    // they share (`crates/artifact/tests/system_block.rs` pins that property on
    // CPU; here it is the input). The instructions are long enough that the
    // block floors to more than one KV page.
    let instructions = "You are a subagent working inside a single git worktree. Read only the \
         files you are given, never touch a sibling worktree, and report what you found in at \
         most ten lines. Explain your reasoning before your conclusion, and say plainly when a \
         question cannot be answered from the files you were shown. Prefer quoting the source \
         over paraphrasing it, and name every file you read by its full path.";
    let render = |query: &str| {
        frontend
            .chat_template()
            .render_with_thinking_and_tools(
                &[
                    ChatMessage::text(Role::System, instructions),
                    ChatMessage::text(Role::User, query),
                ],
                ChatRenderOptions::default(),
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

    let prompt_1 = render("Summarise what crates/core/src/prefix.rs is responsible for.");
    let prompt_2 = render("List every ADR that mentions eviction, and say what each decides.");
    let block_at = ChatTemplate::system_block_offset(&prompt_1)
        .expect("the render opens with a system block");
    assert_eq!(
        ChatTemplate::system_block_offset(&prompt_2),
        Some(block_at),
        "both subagents render the same block"
    );
    let head = encode(&prompt_1[..block_at]);
    let first = encode(&prompt_1);
    let second = encode(&prompt_2);
    assert!(
        first.starts_with(&head) && second.starts_with(&head),
        "the block must be an exact token prefix of both prompts — the whole feature"
    );
    // The published head is the block floored to whole KV pages: what the
    // scheduler cuts the publishing chunk at (`Request::retained_prefix_point`).
    let publish_at = (head.len() as u32 / PAGE) * PAGE;
    assert!(
        publish_at > 0,
        "the system block must be at least one page long: {} tokens",
        head.len()
    );
    assert!(
        second.len() < (MAX_CONTEXT as usize) - GENERATED,
        "the prompt plus its generation must fit the reservation: {} tokens",
        second.len()
    );
    println!(
        "retained_prefix_gpu: block {} tokens (publish at {publish_at}), subagent prompts {} and \
         {} tokens",
        head.len(),
        first.len(),
        second.len()
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
            // GitHub #215: the block's image goes into a retained slot.
            retained_slot_count: 1,
        },
    )
    .unwrap_or_else(|e| panic!("ignis_seq_pool_create: {e}"));
    let free_at_rest = pool.stats().kv_free_pages;

    // ---- subagent 1: prefill to the block, publish it, prefill its own
    //      question, then **finish and go**. Exactly the decomposition the
    //      scheduler cuts, followed by the release that #186's checkpoint path
    //      never exercises for a bare prefix.
    let mut publisher = pool
        .alloc(MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("alloc publisher: {e}"));
    prefill_program(&model, &pool, &mut publisher, &first[..publish_at as usize], 0, None)
        .unwrap_or_else(|e| panic!("publisher head prefill: {e}"));
    let prefix = publisher
        .publish_prefix(publish_at, 0)
        .unwrap_or_else(|e| panic!("publish: {e}"));
    assert_eq!(prefix.stats().tokens, publish_at, "the prefix covers the block");
    assert_eq!(prefix.stats().pages, publish_at / PAGE);
    prefill_program(
        &model,
        &pool,
        &mut publisher,
        &first[publish_at as usize..],
        u64::from(publish_at),
        None,
    )
    .unwrap_or_else(|e| panic!("publisher tail prefill: {e}"));
    let _ = decode_n(&model, &pool, &mut publisher, GENERATED, "publisher");
    // Subagent 1 completes. Only the retention keeps the block alive now —
    // which is the whole of what #188 changed, and the state every assertion
    // below runs against.
    drop(publisher);
    let free_retained = pool.stats().kv_free_pages;
    assert_eq!(
        free_at_rest - free_retained,
        publish_at / PAGE,
        "the retained prefix holds exactly the block's pages, and nothing else"
    );

    // ---- the split control: subagent 2's prompt, cold, cut where subagent 1
    //      cut it.
    let mut control = pool
        .alloc(MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("alloc split control: {e}"));
    for (span, start) in [
        (&second[..publish_at as usize], 0u64),
        (&second[publish_at as usize..], u64::from(publish_at)),
    ] {
        prefill_program(&model, &pool, &mut control, span, start, None)
            .unwrap_or_else(|e| panic!("split control prefill at {start}: {e}"));
    }
    let expected = decode_n(&model, &pool, &mut control, GENERATED, "split control");

    // ---- subagent 2: stand up on the retained prefix, prefill only its own
    //      question.
    let mut reuser = pool
        .alloc_shared(MAX_CONTEXT, &prefix)
        .unwrap_or_else(|e| panic!("alloc against the retained prefix: {e}"));
    // The claimant's own account of what it holds, read off the leaf rather
    // than assumed, so the two views are compared with each other instead of
    // each with a literal (the shape `prefix_reuse_gpu.rs` uses).
    assert_eq!(
        reuser.stats().shared_pages,
        prefix.stats().pages,
        "the claimant's shared pages are exactly the retained block's"
    );
    prefill_program(
        &model,
        &pool,
        &mut reuser,
        &second[publish_at as usize..],
        u64::from(publish_at),
        None,
    )
    .unwrap_or_else(|e| panic!("reuser tail prefill: {e}"));
    let reused = decode_n(&model, &pool, &mut reuser, GENERATED, "reuser");

    assert_eq!(
        reused, expected,
        "a claimant of a retained prefix must generate exactly what a cold prefill split at \
         the same boundary does"
    );

    // ---- non-consuming: the rest of the burst hits the same entry, and the
    //      block's pages are charged once however many stand on it.
    let free_before_third = pool.stats().kv_free_pages;
    let mut third = pool
        .alloc_shared(MAX_CONTEXT, &prefix)
        .unwrap_or_else(|e| panic!("alloc third claimant: {e}"));
    let claimant = third.stats();
    assert_eq!(
        free_before_third - pool.stats().kv_free_pages,
        claimant.mapped_pages - claimant.shared_pages,
        "a second claimant reserves only its own tail: the block is charged once, \
         however many of the burst stand on it"
    );
    prefill_program(
        &model,
        &pool,
        &mut third,
        &second[publish_at as usize..],
        u64::from(publish_at),
        None,
    )
    .unwrap_or_else(|e| panic!("third claimant tail prefill: {e}"));
    let again = decode_n(&model, &pool, &mut third, GENERATED, "third claimant");
    assert_eq!(again, expected, "a later burst member hits the same entry");

    // ---- the unsplit control: information, not a verdict (ADR 0029).
    let mut unsplit = pool
        .alloc(MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("alloc unsplit control: {e}"));
    prefill_program(&model, &pool, &mut unsplit, &second, 0, None)
        .unwrap_or_else(|e| panic!("unsplit control prefill: {e}"));
    let whole = decode_n(&model, &pool, &mut unsplit, GENERATED, "unsplit control");
    match first_divergence(&whole, &expected) {
        None => println!(
            "retained_prefix_gpu: the unsplit cold prefill agrees with both split runs for all \
             {GENERATED} tokens"
        ),
        Some(at) => println!(
            "retained_prefix_gpu: the unsplit cold prefill parts company at token {at} ({} vs \
             {}) — the chunk-decomposition effect (#153), recorded not asserted",
            whole[at], expected[at]
        ),
    }

    // ---- giving up the retention returns every page ------------------------
    //
    // The first-victim path's end state: `PrefixCache::unretain` drops the
    // scheduler's ledger entry and the handle below is the reference it was
    // holding. Nothing may be left behind.
    drop(third);
    drop(reuser);
    drop(control);
    drop(unsplit);
    assert_eq!(
        free_at_rest - pool.stats().kv_free_pages,
        publish_at / PAGE,
        "with every claimant gone the retention alone still holds the block"
    );
    drop(prefix);
    assert_eq!(
        pool.stats().kv_free_pages,
        free_at_rest,
        "and releasing the retention returns every page"
    );
}
