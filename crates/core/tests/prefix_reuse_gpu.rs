//! GPU integration coverage for device prefix reuse against the real model
//! (P4-10, GitHub #126, ADR 0024).
//!
//! The leaf's own test (`kernel/tests/test_seq_prefix.cpp`) proves that the
//! pages are shared and the mutable state is cloned: a claimant's
//! block-table row addresses the publisher's own physical pages, and its GDN
//! slot, conv taps and penalty counts come out byte-identical. What it cannot
//! say is whether a sequence built that way **generates the same thing** as
//! one that prefilled the prefix itself — for that the model has to run, and
//! the tokens have to match.
//!
//! So this test runs three sequences over the same prompt:
//!
//!   * a **control** that prefills the whole prompt in one span, the way a
//!     request with no sibling does;
//!   * a **publisher** that prefills the shared head, publishes it, then
//!     prefills its own tail;
//!   * a **claimant** that is allocated against that prefix — skipping the
//!     head entirely — and prefills only the tail.
//!
//! All three must decode to the same tokens. If the clone missed a section,
//! or the shared pages were zeroed, or the claimant's row addressed the wrong
//! physical pages, the tails diverge — and nothing else in the stack would
//! have said so.
//!
//! It also drives the two claims the accounting rests on: the shared pages
//! are charged to the pool **once** however many claimants hold them, and
//! releasing one claimant leaves the others' output alone.
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
const MAX_CONTEXT: u32 = 512;
/// The shared head, in tokens. A prefix is whole KV pages and the leaf's page
/// is 64 tokens, so this is two pages — enough that the head is real history
/// rather than a rounding artifact, and short enough to leave room for a tail
/// inside `MAX_CONTEXT`.
const SHARED_TOKENS: u32 = 128;
/// Decoded from each sequence. Enough that a stale GDN slot, a missing
/// penalty-count row or a mis-addressed page would have shown up by the end.
const GENERATED: usize = 6;

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

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_claimant_decodes_what_a_sibling_that_prefilled_the_prefix_decodes() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    // A prompt long enough that its first 128 tokens are a real shared head.
    // The text is repeated rather than padded with a single token: a
    // degenerate prefix would make a wrong clone easy to survive.
    let text = "A paged KV cache stores attention keys and values in fixed-size \
                pages so that a sequence's history need not be contiguous in \
                memory. Each sequence owns a block table mapping its logical \
                pages to physical ones. Explain, carefully and at length, what \
                that buys an inference engine serving many requests at once, \
                and what it costs. "
        .repeat(4);
    let prompt: Vec<i32> = frontend
        .tokenizer()
        .encode(&text)
        .unwrap_or_else(|e| panic!("tokenize prompt: {e}"))
        .into_iter()
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();
    assert!(
        prompt.len() > SHARED_TOKENS as usize + 8,
        "the prompt must carry a shared head and a tail: got {} tokens",
        prompt.len()
    );
    assert!(
        prompt.len() < MAX_CONTEXT as usize - GENERATED,
        "the prompt plus its generation must fit the reservation"
    );
    let head = SHARED_TOKENS as usize;

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
    // Four slots and enough pages for the control, the publisher and two
    // claimants at 8 pages each — deliberately more than the 32 an unshared
    // four-way load would need, so the accounting check below is measuring
    // the sharing and not a pool that happened to be tight.
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: ignis_core::KvFormat::Bf16,
            kv_page_group_count: 48,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 4,
        },
    )
    .unwrap_or_else(|e| panic!("ignis_seq_pool_create: {e}"));
    let free_at_rest = pool.stats().kv_free_pages;

    // ---- the control: one sequence, one span, no sharing -------------------
    let mut control = pool
        .alloc(MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("alloc control: {e}"));
    prefill_program(&model, &pool, &mut control, &prompt, 0, None)
        .unwrap_or_else(|e| panic!("control prefill: {e}"));
    let expected = decode_n(&model, &pool, &mut control, GENERATED, "control");

    // ---- the publisher: head, publish, tail --------------------------------
    //
    // The head is prefilled as its own span so the sequence stands exactly on
    // the prefix when it publishes — which is the leaf's precondition, and
    // the reason the scheduler cuts a chunk there.
    let mut publisher = pool
        .alloc(MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("alloc publisher: {e}"));
    prefill_program(&model, &pool, &mut publisher, &prompt[..head], 0, None)
        .unwrap_or_else(|e| panic!("publisher head prefill: {e}"));
    let free_before_publish = pool.stats().kv_free_pages;
    let prefix = publisher
        .publish_prefix(SHARED_TOKENS)
        .unwrap_or_else(|e| panic!("publish: {e}"));
    let stats = prefix.stats();
    assert_eq!(stats.tokens, SHARED_TOKENS, "the prefix covers the head");
    assert_eq!(stats.pages, 2, "128 tokens over 64-token pages is 2 pages");
    assert_eq!(
        pool.stats().kv_free_pages,
        free_before_publish,
        "publishing takes over pages the publisher already held — it costs none"
    );

    prefill_program(
        &model,
        &pool,
        &mut publisher,
        &prompt[head..],
        u64::from(SHARED_TOKENS),
        None,
    )
    .unwrap_or_else(|e| panic!("publisher tail prefill: {e}"));
    assert_eq!(
        decode_n(&model, &pool, &mut publisher, GENERATED, "publisher"),
        expected,
        "publishing a prefix must not change what the publisher itself generates"
    );

    // ---- two claimants: skip the head entirely -----------------------------
    let free_before_claims = pool.stats().kv_free_pages;
    let mut first = pool
        .alloc_shared(MAX_CONTEXT, &prefix)
        .unwrap_or_else(|e| panic!("first claim: {e}"));
    let mut second = pool
        .alloc_shared(MAX_CONTEXT, &prefix)
        .unwrap_or_else(|e| panic!("second claim: {e}"));
    // The claimant's own account of what it holds, which is what admission
    // charges against: a tail of its own plus a shared head it is not billed
    // for. Read off the leaf rather than assumed, so the two views are
    // compared with each other instead of each with a literal.
    let claimant = first.stats();
    let shared = prefix.stats().pages;
    assert_eq!(
        claimant.shared_pages, shared,
        "a claimant's shared pages are exactly the prefix's"
    );
    let tail = claimant.mapped_pages - claimant.shared_pages;
    assert_eq!(
        free_before_claims - pool.stats().kv_free_pages,
        2 * tail,
        "two claimants cost two tails: the shared head is charged to the pool once"
    );
    // And the same arithmetic the admission machine runs. `ConcreteScheduler`
    // charges `prefix.pages + sum(tails)` for a published prefix and its
    // holders (crates/core/src/concrete.rs — the charge split at registration
    // and at each claim); this is that expression evaluated from the leaf's
    // own numbers and compared against the leaf's own pool.
    let holders = 3; // the publisher and its two claimants
    assert_eq!(
        free_at_rest - pool.stats().kv_free_pages,
        claimant.mapped_pages + shared + holders * tail,
        "the leaf's pool spends the unshared control sequence plus the prefix once          and one tail per holder — the admission machine's own charge for the same set"
    );

    // A claimant's history is not all its own, so it cannot be moved as one
    // blob (P4-10 against P4-06). The right response is to release it and
    // re-prefill, which is what the distinct code is for.
    let refusal = first
        .snapshot_bytes()
        .expect_err("a sequence sharing a prefix has no whole-sequence snapshot");
    assert!(
        refusal.is_shared_prefix(),
        "the sequence, not the call, is what cannot be transferred: {refusal}"
    );

    let claim_stats = prefix.stats();
    assert_eq!(claim_stats.clone_count, 2, "two claims were served");
    assert!(
        claim_stats.last_clone_micros > 0.0,
        "the clone's cost is measured, not assumed"
    );

    // Each claimant carries only its tail — the head is already warm in the
    // pages it shares, and the clone put its recurrent state where the
    // publisher's was.
    for (sequence, label) in [(&mut first, "first claimant"), (&mut second, "second claimant")] {
        prefill_program(&model, &pool, sequence, &prompt[head..], u64::from(SHARED_TOKENS), None)
            .unwrap_or_else(|e| panic!("{label} tail prefill: {e}"));
    }
    let first_head = decode_n(&model, &pool, &mut first, GENERATED / 2, "first claimant");
    assert_eq!(
        first_head,
        expected[..GENERATED / 2],
        "a claimant that skipped the prefill decodes what the control decoded"
    );

    // ---- releasing one claimant leaves the other alone ---------------------
    //
    // The shared pages are refcounted by the leaf; a release that returned
    // them early would hand them to the next allocation and quietly corrupt
    // whoever still reads them.
    drop(first);
    let mut filler = pool
        .alloc(MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("alloc filler: {e}"));
    prefill_program(&model, &pool, &mut filler, &prompt[..head], 0, None)
        .unwrap_or_else(|e| panic!("filler prefill: {e}"));
    let _ = decode_n(&model, &pool, &mut filler, 2, "filler");
    assert_eq!(
        decode_n(&model, &pool, &mut second, GENERATED, "second claimant"),
        expected,
        "one claimant's release, and another sequence's allocation, leave the \
         surviving claimant's output untouched"
    );
    drop(filler);

    // ---- the last holder returns the pages ---------------------------------
    drop(second);
    drop(publisher);
    drop(prefix);
    drop(control);
    assert_eq!(
        pool.stats().kv_free_pages,
        free_at_rest,
        "every page is back once the last holder releases"
    );
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn publishing_is_refused_unless_the_sequence_stands_on_the_prefix() {
    // The precondition the whole mechanism rests on: what a claimant clones
    // is the mutable state at the prefix's end, so a sequence standing
    // anywhere else has nothing to give. Checked here at the binding level,
    // where the failure is a typed error a scheduler branches on.
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let _ = &mut device;

    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: ignis_core::KvFormat::Bf16,
            kv_page_group_count: 24,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 2,
        },
    )
    .unwrap_or_else(|e| panic!("ignis_seq_pool_create: {e}"));

    // No model needed: the refusal is decided before any device work, and
    // the sequence's frontier is what publish checks. Prefill would only
    // make the test slower and its failure mode less clear.
    let mut publisher = pool
        .alloc(MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("alloc publisher: {e}"));
    let fresh = publisher
        .snapshot_bytes()
        .expect("a sequence with no prefix has a snapshot size");
    assert!(fresh > 0);
    // A fresh sequence stands at 0, and 0 is not a publishable prefix — the
    // refusal names the boundary rather than silently publishing nothing.
    let refusal = publisher
        .publish_prefix(SHARED_TOKENS)
        .expect_err("a sequence that has written nothing cannot publish a 128-token head");
    assert!(
        refusal.is_not_at_boundary(),
        "the frontier, not the arguments, is what is wrong: {refusal}"
    );
}
