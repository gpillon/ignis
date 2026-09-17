//! GitHub #188 (ADR 0029) — the **retained prefix**: the shared prefix a
//! request publishes at its system-and-tools block, kept alive after its last
//! claimant is gone so the next subagent of a burst claims it.
//!
//! The prompt checkpoint (#186) serves a *conversation*: turn N+1's prompt
//! extends turn N's, so the state at turn N's generation opener is the state
//! turn N+1 would have prefilled. A **burst** has no such relation. Two
//! subagents spawned from one parent send two different questions, and neither
//! prompt extends the other's — so no checkpoint can ever match between them,
//! and what they do share, the whole system and tools block, is all there is
//! to reuse. That block is what this publishes, and keeping the entry alive at
//! refcount 0 is what makes "after the first finished" the same as "while the
//! first is still running".
//!
//! What these tests can prove, and what they cannot. "The reused pages are the
//! *right* pages" needs the model to run and lives in `retained_prefix_gpu.rs`.
//! What lives here is what the next layer up observes without a card: which
//! prompt tokens are prefilled at all, where each request's prefill is cut,
//! what the pool is charged, and — the load-bearing one — that a retained
//! prefix never makes a live request wait.
//!
//! **One prefix per sequence, today.** The leaf refuses a second publish from
//! a sequence that already holds one (`kernel/src/seq_prefix.cu:129`), and a
//! prompt checkpoint demands that the whole pages below its opener *be* that
//! prefix (`kernel/src/seq_checkpoint.cu:125`, whose own message names #187 as
//! the ticket that makes such a request qualify). So a request publishes
//! **one** head, and the two
//! boundaries compete for it whenever they fall in different KV pages: the
//! extra chunk split the spec asks for is moved, not added.
//!
//! They are not exclusive in general, only per published head. When the block
//! and the opener land in the *same* page, the one prefix below it serves
//! both, and the request leaves a retained prefix *and* a checkpoint —
//! `a_block_and_an_opener_in_one_page_leave_both_a_retained_prefix_and_a_checkpoint`
//! below, and what #186's live GPU check observed.
//!
//! **This is a state that ends, and not by weakening the capture.** #187
//! chains the *publish*: it removes `seq_prefix.cu:129`'s
//! `seq->prefix != nullptr`, so a sequence standing on the block publishes a
//! second prefix over the head it warmed itself, taking over the reference it
//! held — which is what gives the pages between the block and the opener a
//! holder. `seq_checkpoint.cu:125`'s `below != seq->shared_pages` stays, and
//! must: a checkpoint holds exactly one copied tail page, taken from
//! `seq->kv.page_ids()[0]`, so relaxing it would copy a page that is not the
//! opener's and hand a claimant intermediate pages nothing warmed — the defect
//! #186 fixed in `565d634`. After #187 `below == shared_pages` becomes *true*
//! for such a request rather than waived, and the two compose instead of
//! competing.
//!
//! Until then a prompt with tools — every qwen-code request —
//! leaves the block and no checkpoint, which is what
//! `the_prefill_is_cut_at_the_system_block_rather_than_at_the_opener` pins.
//! A prompt whose block is under one page keeps #186's behaviour unchanged,
//! and that is asserted below rather than left to be discovered.
//!
//! Seams (ADR 0006): the `Scheduler` trait driven with a `MockCompute` that
//! records the prefill call shape behind the `Compute` seam, exactly as
//! `prefix_reuse.rs` and `prompt_checkpoint.rs` do.

use std::sync::Arc;

use ignis_core::checkpoint::ReuseSource;
use ignis_core::types::{DecodeParams, RequestClass, RequestId, RequestInput, SchedEvent};
use ignis_core::vision::{Grid, MediaItem, Multimodal, TokenSpan};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig};

const MODEL: &str = "qwen3.8-27b";
/// The default scheduler's KV page, in tokens.
const PAGE: u32 = 16;
/// The shared system-and-tools block, in tokens: 35, so it floors to two whole
/// pages and its last three tokens are the partial page nobody may share.
const BLOCK: u32 = 35;

/// `n` distinct tokens starting at `start`.
fn tokens(start: u32, n: u32) -> Vec<u32> {
    (start..start + n).collect()
}

/// A request whose rendered prompt reports `block` as the end of its first
/// system-and-tools block and `opener` as its generation opener.
fn input(prompt: Vec<u32>, block: Option<u32>, opener: Option<u32>, max: u32) -> RequestInput {
    RequestInput {
        model: MODEL.into(),
        tokens: prompt,
        params: DecodeParams {
            max_tokens: Some(max),
            ..DecodeParams::default()
        },
        multimodal: None,
        opener_tokens: opener,
        user_turn_tokens: None,
        system_block_tokens: block,
    }
}

/// One subagent of a burst: the shared 35-token block, then its own 25-token
/// question, with the generation opener three tokens from the end (where a
/// real render puts it — the primed `<think>` follows).
fn subagent(query: u32) -> RequestInput {
    input(
        [tokens(1, BLOCK), tokens(query, 25)].concat(),
        Some(BLOCK),
        Some(57),
        4,
    )
}

fn scheduler(compute: Arc<MockCompute>, config: SchedulerConfig) -> ConcreteScheduler {
    ConcreteScheduler::with_config(config, compute)
}

fn config() -> SchedulerConfig {
    SchedulerConfig {
        model: MODEL.into(),
        ..SchedulerConfig::default()
    }
}

fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    while !sched.is_idle() {
        events.extend(sched.advance());
    }
    events
}

/// The leading prompt tokens `request` skipped through a shared prefix.
fn prefix_reuses(events: &[SchedEvent], request: RequestId) -> Vec<u32> {
    events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::PrefixReused {
                request: r, tokens, ..
            } if *r == request => Some(*tokens),
            _ => None,
        })
        .collect()
}

/// The retained-state reuses (`source`, skipped tokens) recorded for
/// `request` — a prompt checkpoint claim, never a prefix one.
fn state_reuses(events: &[SchedEvent], request: RequestId) -> Vec<(ReuseSource, u32)> {
    events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::StateReused {
                request: r,
                source,
                tokens,
                ..
            } if *r == request => Some((*source, *tokens)),
            _ => None,
        })
        .collect()
}

/// Every prefill chunk width dealt to `request`, in order.
fn chunk_widths(compute: &MockCompute, request: RequestId) -> Vec<usize> {
    compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|j| j.request == request)
        .map(|j| j.tokens.len())
        .collect()
}

/// `request`'s first prefill job — the one that stands the sequence up.
fn first_job(compute: &MockCompute, request: RequestId) -> ignis_core::scheduler::PrefillJob {
    compute
        .prefill_calls()
        .into_iter()
        .flatten()
        .find(|j| j.request == request)
        .expect("the request was prefilled")
}

// ── The burst ───────────────────────────────────────────────────────────

#[test]
fn a_second_subagent_arriving_after_the_first_finished_skips_the_system_block() {
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());

    // The first subagent runs and **finishes**. Nothing of it is live when the
    // second arrives, which is exactly the case sibling prefix reuse (core-07)
    // cannot serve: its entry would have dropped with its last claimant.
    let first = sched.submit(subagent(500), RequestClass::Agent).unwrap();
    run_to_idle(&mut sched);
    assert!(sched.is_idle(), "the first subagent is gone");
    assert_eq!(
        sched.prefix_pinned_pages(),
        BLOCK / PAGE + 1,
        "its block is still held — two whole pages, charged once — plus the one page          it chained over them for its own prompt checkpoint (#187)"
    );

    let second = sched.submit(subagent(900), RequestClass::Agent).unwrap();
    let events = run_to_idle(&mut sched);

    assert_eq!(
        prefix_reuses(&events, second),
        vec![32],
        "the second subagent skipped the whole system and tools block"
    );
    // It prefilled its own 28 tokens and nothing else: no chunk of it ever
    // carried a token the first subagent had already warmed.
    assert_eq!(
        chunk_widths(&compute, second).iter().sum::<usize>(),
        28,
        "only its own question"
    );
    let job = first_job(&compute, second);
    assert_eq!(job.start_position, 32, "it starts where the block ends");
    assert_eq!(
        job.shared_prefix.map(|c| (c.publisher, c.tokens)),
        Some((first, 32)),
        "its first job claims the first subagent's prefix by name"
    );
    assert_eq!(
        sched.sibling_prefix_reused_tok(),
        32,
        "the skip is counted once"
    );
    assert_eq!(
        sched.prefix_pinned_pages(),
        BLOCK / PAGE + 2,
        "and the block is still one set of pages, not two — what grew is one chained          page per subagent, each holding that subagent's own checkpoint (#187)"
    );
}

#[test]
fn a_whole_burst_of_subagents_claims_the_one_retained_block() {
    // The shape the feature exists for: one parent spawns a burst, and the
    // members neither overlap in time nor extend one another's prompts.
    //
    // Five members keep eleven images on the device — the block, and each
    // member's chained link and checkpoint — so the load has room for them
    // (GitHub #215: every one takes a retained slot).
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(
        compute.clone(),
        SchedulerConfig {
            retained_slots: 11,
            ..config()
        },
    );
    sched.submit(subagent(500), RequestClass::Agent).unwrap();
    run_to_idle(&mut sched);

    for query in [600u32, 700, 800, 900] {
        let id = sched.submit(subagent(query), RequestClass::Agent).unwrap();
        let events = run_to_idle(&mut sched);
        assert_eq!(
            prefix_reuses(&events, id),
            vec![32],
            "subagent {query} skipped the block although every sibling had finished"
        );
    }
    assert_eq!(sched.sibling_prefix_reused_tok(), 4 * 32);
    assert_eq!(
        sched.prefix_pinned_pages(),
        BLOCK / PAGE + 5,
        "one entry, one charge, however many claimed it — and one chained page each"
    );
    // GitHub #187 closed #186's departure here: a burst used to leave one
    // checkpoint, whoever published first. Each member now chains its own
    // head over the shared block and captures at its own opener, so a retry
    // of any one of these five questions hits rather than re-prefills.
    assert_eq!(
        sched.checkpoint_pool().entry_count(),
        5,
        "one checkpoint per sibling, not one for the publisher"
    );
}

#[test]
fn a_subagent_with_a_different_block_shares_nothing() {
    // Matched by content, never by shape: a burst under different
    // instructions renders a different block, and must reuse none of it.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    sched.submit(subagent(500), RequestClass::Agent).unwrap();
    run_to_idle(&mut sched);

    let other = sched
        .submit(
            input(
                [tokens(1000, BLOCK), tokens(500, 25)].concat(),
                Some(BLOCK),
                Some(57),
                4,
            ),
            RequestClass::Agent,
        )
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert!(prefix_reuses(&events, other).is_empty(), "nothing matched");
    assert_eq!(
        chunk_widths(&compute, other).iter().sum::<usize>(),
        60,
        "it prefilled its whole prompt"
    );
}

// ── Where the prefill is cut ────────────────────────────────────────────

#[test]
fn the_prefill_is_cut_at_the_system_block_and_again_at_the_opener() {
    // The composition proof for the whole wave, and the reason this file's
    // assertion used to read the other way round.
    //
    // #186 cut this prompt at its opener's page. #188 cut it at the block's
    // page instead, because that is the only head a *burst* can share — and
    // while the leaf allowed one prefix per sequence that was publishing
    // *instead of* the opener's page, so the capture's `below ==
    // shared_pages` could not hold and the checkpoint was lost. Here the two
    // boundaries fall in different pages (the block's ends page 2, the opener
    // is in page 3), which is exactly the shape qwen-code sends on every
    // request, so that was every request.
    //
    // #187 makes the two compose: a sequence holds a prefix *chain*, so this
    // request publishes the block, chains its own opener's page over it, and
    // captures there. Three cuts, one retained prefix, one prompt checkpoint
    // — and the capture's precondition is satisfied rather than weakened
    // (see this file's header).
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    let id = sched.submit(subagent(500), RequestClass::Agent).unwrap();
    run_to_idle(&mut sched);

    assert_eq!(
        chunk_widths(&compute, id),
        vec![32, 16, 9, 3],
        "cut at the two-page block boundary, then at the opener's page, then at the opener"
    );
    let published: Vec<Option<u32>> = compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|j| j.request == id)
        .map(|j| j.publish_prefix.map(|p| p.tokens))
        .collect();
    assert_eq!(
        published,
        vec![Some(32), Some(48), None, None],
        "the block, then the chained head this request warmed past it"
    );
    let captured: Vec<Option<u32>> = compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|j| j.request == id)
        .map(|j| j.capture_checkpoint.map(|c| c.tokens))
        .collect();
    assert_eq!(
        captured,
        vec![None, None, Some(57), None],
        "and exactly the chunk that ends on the opener asks for the capture"
    );
    assert_eq!(
        sched.checkpoint_pool().entry_count(),
        1,
        "and it still leaves a prompt checkpoint: the block prefix and the opener's page are a chain, not a choice (#187)"
    );
}

#[test]
fn a_block_and_an_opener_in_one_page_leave_both_a_retained_prefix_and_a_checkpoint() {
    // The two are not exclusive in general — they are exclusive *per published
    // head*. A short first turn puts the block's end and the generation opener
    // inside the same KV page, so the one prefix published below that page is
    // both the burst's retained block and the whole pages the conversation's
    // own checkpoint stands on. #186's live GPU check observed exactly this.
    //
    // A 35-token block (two whole pages) with a 45-token opener: 45 / 16 is 2,
    // which is the publish point's page count, so `checkpoint_point` is
    // satisfied on the very prefix the block published.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    let id = sched
        .submit(
            input(
                [tokens(1, BLOCK), tokens(500, 15)].concat(),
                Some(BLOCK),
                Some(45),
                4,
            ),
            RequestClass::Interactive,
        )
        .unwrap();
    run_to_idle(&mut sched);

    assert_eq!(
        chunk_widths(&compute, id),
        vec![32, 13, 5],
        "cut at the block's page, then again at the opener inside that page"
    );
    assert_eq!(
        sched.checkpoint_pool().entry_count(),
        1,
        "the conversation kept its checkpoint"
    );
    assert_eq!(
        sched.prefix_pinned_pages(),
        BLOCK / PAGE,
        "standing on the same two pages the burst's block retained"
    );

    // Both are then claimable, each by the request it exists for: the next
    // turn of this conversation through the checkpoint, an unrelated burst
    // member through the block under it.
    // Turn N+1 is turn N's prompt *up to its opener* — the block, then the
    // ten tokens of the first question that preceded the opener — and then
    // its own continuation.
    let next_turn = sched
        .submit(
            input(
                [tokens(1, BLOCK), tokens(500, 10), tokens(700, 15)].concat(),
                Some(BLOCK),
                Some(57),
                4,
            ),
            RequestClass::Interactive,
        )
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(
        state_reuses(&events, next_turn),
        vec![(ReuseSource::Device, 45)],
        "the next turn resumed at the opener"
    );

    let sibling = sched.submit(subagent(900), RequestClass::Agent).unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(
        prefix_reuses(&events, sibling),
        vec![32],
        "and a burst member that shares only the block still claims it"
    );
}

#[test]
fn a_block_under_one_page_publishes_nothing_and_leaves_the_checkpoint_alone() {
    // "Under one page nothing is published" means nothing is published *at
    // that boundary* — a shared prefix is whole KV pages, and half a page
    // cannot be split between two requests. The request keeps #186's opener
    // page and its prompt checkpoint with it, which is strictly better than
    // giving up both.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    let id = sched
        .submit(
            input(tokens(1, 40), Some(PAGE - 1), Some(37), 4),
            RequestClass::Interactive,
        )
        .unwrap();
    run_to_idle(&mut sched);

    assert_eq!(
        chunk_widths(&compute, id),
        vec![32, 5, 3],
        "cut at the opener's page and again at the opener, exactly as #186 cuts it"
    );
    assert_eq!(
        sched.checkpoint_pool().entry_count(),
        1,
        "the prompt checkpoint is untouched by a sub-page block"
    );
}

#[test]
fn a_render_with_no_system_block_behaves_exactly_as_it_did_before() {
    // Three of the seven reference-recorded renders carry neither a system
    // message nor tools. They report no boundary, and nothing about them
    // changes.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    let id = sched
        .submit(input(tokens(1, 40), None, Some(37), 4), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(chunk_widths(&compute, id), vec![32, 5, 3]);
    assert_eq!(sched.checkpoint_pool().entry_count(), 1);
}

// ── Longest reuse wins ──────────────────────────────────────────────────

#[test]
fn a_longer_prompt_checkpoint_match_wins_over_the_retained_prefix() {
    // ADR 0029: longest reuse wins. A retained prefix stops at a page
    // boundary inside the system block; a prompt checkpoint reaches all the
    // way to a finished conversation's generation opener. When both match a
    // prompt the checkpoint takes it — and the losing prefix must not have
    // been claimed at all, because claiming is not free: it pins the entry and
    // counts a skip that never happened.
    //
    // Both entries exist only because the two requests that left them were
    // **concurrent**: whoever arrives second claims what is already there and
    // therefore publishes nothing. A burst and a conversation landing in one
    // batch is exactly that case.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    // The conversation: no system block reported, so it publishes at its
    // opener's page (32) and captures a checkpoint at 37.
    let chat = sched
        .submit(input(tokens(1, 40), None, Some(37), 4), RequestClass::Interactive)
        .unwrap();
    // The burst member, submitted before the conversation prefills: a
    // one-page block, so it publishes at 16 — a different head, so the
    // one-publisher-per-head rule does not silence it. No opener reported, so
    // the block is all it leaves: what this test weighs is one checkpoint
    // against one retained prefix, and a member that chained and captured too
    // (#187) would be weighing three.
    let burst = sched
        .submit(
            input([tokens(1, 20), tokens(700, 30)].concat(), Some(20), None, 4),
            RequestClass::Interactive,
        )
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(sched.checkpoint_pool().entry_count(), 1, "the chat checkpoint");
    assert_eq!(
        sched.prefix_pinned_pages(),
        2 + 1,
        "the chat's two pages under its checkpoint, and the burst's one retained page"
    );
    let _ = (chat, burst);

    // A third request that starts with both: the checkpoint's 37 tokens and,
    // inside them, the retained 16.
    let baseline = sched.sibling_prefix_reused_tok();
    let later = sched
        .submit(
            input([tokens(1, 37), tokens(900, 20)].concat(), Some(20), Some(54), 4),
            RequestClass::Interactive,
        )
        .unwrap();
    let events = run_to_idle(&mut sched);

    assert_eq!(
        state_reuses(&events, later),
        vec![(ReuseSource::Device, 37)],
        "the checkpoint reaches further, so it wins"
    );
    assert!(
        prefix_reuses(&events, later).is_empty(),
        "and the shorter retained prefix was never claimed"
    );
    assert_eq!(
        sched.sibling_prefix_reused_tok(),
        baseline,
        "a losing match counts no skip"
    );
    assert_eq!(
        chunk_widths(&compute, later).iter().sum::<usize>(),
        20,
        "it prefilled only what nothing had warmed"
    );
}

#[test]
fn a_retained_prefix_wins_when_no_checkpoint_reaches_further() {
    // The mirror, so the comparison above is not passing for the wrong
    // reason: a checkpoint that matches nothing leaves the prefix to win.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    sched
        .submit(input(tokens(1, 40), None, Some(37), 4), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);
    sched.submit(subagent(500), RequestClass::Agent).unwrap();
    run_to_idle(&mut sched);

    // Its prompt shares the block but parts from the conversation at token 35.
    let later = sched.submit(subagent(900), RequestClass::Agent).unwrap();
    let events = run_to_idle(&mut sched);
    assert!(
        state_reuses(&events, later).is_empty(),
        "no checkpoint is a prefix of this prompt"
    );
    assert_eq!(prefix_reuses(&events, later), vec![32], "the block is");
}

// ── Retained state never costs a live request ───────────────────────────

#[test]
fn a_live_request_takes_a_retained_prefix_back_rather_than_waiting_for_it() {
    // ADR 0023 as amended by 0029: on the device, retained state is the first
    // victim — always, and before the eviction machinery is asked for
    // anything. A retained prefix must never be the reason a live request is
    // refused, evicted or made to wait.
    let compute = Arc::new(MockCompute::new());
    // Room for six pages: the burst's two, and four for whoever comes next.
    let mut sched = scheduler(
        compute.clone(),
        SchedulerConfig {
            kv_capacity_pages: 6,
            max_sequence_tokens: 96,
            ..config()
        },
    );
    sched.submit(subagent(500), RequestClass::Agent).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        sched.prefix_pinned_pages(),
        3,
        "the block is retained, and the page the subagent chained over it (#187)"
    );

    // A request that cannot fit beside it: 90 tokens + 4 generated is six
    // pages, and two of them are under the retained block.
    let big = sched
        .submit(input(tokens(2000, 90), None, None, 4), RequestClass::Interactive)
        .unwrap();
    let events = run_to_idle(&mut sched);

    assert_eq!(
        chunk_widths(&compute, big).iter().sum::<usize>(),
        90,
        "it ran, and prefilled its whole prompt"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, SchedEvent::Evicted { .. })),
        "nothing live was evicted for it — the bet was given up first"
    );
    assert_eq!(
        sched.prefix_pinned_pages(),
        0,
        "the retained block went back to the pool"
    );
    // And the burst's reuse is simply gone, not half-there: a later subagent
    // re-prefills its block rather than claiming pages nothing holds.
    let after = sched.submit(subagent(900), RequestClass::Agent).unwrap();
    let events = run_to_idle(&mut sched);
    assert!(prefix_reuses(&events, after).is_empty());
    assert_eq!(chunk_widths(&compute, after).iter().sum::<usize>(), 60);
}

#[test]
fn the_narrower_bet_is_given_up_first_when_a_pool_holds_both_kinds() {
    // ADR 0029's amendment (2026-09-16, #188): retained checkpoints go before
    // retained prefixes, LRU within each. A checkpoint serves one
    // conversation's next turn; a retained prefix serves every future request
    // that opens with that block. Between two bets the narrower one goes
    // first — and it frees more pages doing it.
    //
    // The two entries exist together only because the requests that left them
    // were concurrent (whoever arrives second claims and therefore publishes
    // nothing), which is a burst and a conversation landing in one batch.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(
        compute.clone(),
        SchedulerConfig {
            kv_capacity_pages: 6,
            max_sequence_tokens: 96,
            ..config()
        },
    );
    // The conversation: no block reported, so it publishes at its opener's
    // page (32 = two pages) and captures a checkpoint there. The opener sits
    // on the page boundary, so the checkpoint holds no page of its own
    // (GitHub #215) and the arithmetic below is the prefixes' alone.
    sched
        .submit(input(tokens(1, 40), None, Some(32), 4), RequestClass::Interactive)
        .unwrap();
    // The burst member: a one-page block, so it publishes at 16 — a different
    // head, so the one-publisher-per-head rule does not silence it. It reports
    // no opener, so the block is *all* it publishes: a burst member with one
    // would chain its own head over the block and capture there too (#187),
    // which is a second bet and would make "which kind goes first" a question
    // about three entries instead of two.
    sched
        .submit(
            input([tokens(1, 20), tokens(700, 30)].concat(), Some(20), None, 4),
            RequestClass::Interactive,
        )
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(sched.checkpoint_pool().entry_count(), 1, "one checkpoint");
    assert_eq!(sched.prefix_pinned_pages(), 3, "its two pages, and the block's one");

    // A live request needing four pages, with only three free. Exactly one
    // discard is required, so which kind goes is observable: giving up the
    // checkpoint returns two pages and is enough, giving up the block would
    // return one and would also have been enough — so a pool that chose the
    // block would pass every other assertion in this file.
    let live = sched
        .submit(input(tokens(2000, 60), None, None, 4), RequestClass::Interactive)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(
        chunk_widths(&compute, live).iter().sum::<usize>(),
        60,
        "the live request ran"
    );
    assert!(
        !events.iter().any(|e| matches!(e, SchedEvent::Evicted { .. })),
        "and nothing live was evicted for it"
    );
    // Went from the *device*, which is the ordering under test. With a
    // KV-RAM tier below it (GitHub #190) the checkpoint is spilled there
    // rather than lost, so it is still in the pool — just not on the card.
    assert_eq!(
        compute.spilled_checkpoints().len(),
        1,
        "the narrow bet went"
    );
    assert!(
        sched
            .checkpoint_pool()
            .entries()
            .iter()
            .all(|e| e.tier == ignis_core::checkpoint::ReuseSource::KvRam),
        "no checkpoint is left on the device"
    );
    assert_eq!(
        sched.prefix_pinned_pages(),
        1,
        "and the block survived it — the wide bet is given up last"
    );

    // Not an exemption, an ordering: enough pressure and the block goes too.
    let bigger = sched
        .submit(input(tokens(3000, 92), None, None, 4), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        chunk_widths(&compute, bigger).iter().sum::<usize>(),
        92,
        "the larger request ran too"
    );
    assert_eq!(sched.prefix_pinned_pages(), 0, "the block went when it had to");
}

#[test]
fn a_retained_prefix_a_live_request_stands_on_is_not_given_up_for_nothing() {
    // Its pages come back only at refcount zero, so discarding a retention a
    // live claimant is also holding frees not one page — and would cost every
    // later subagent its reuse. The first-victim path has to know that.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(
        compute.clone(),
        SchedulerConfig {
            kv_capacity_pages: 6,
            max_sequence_tokens: 96,
            ..config()
        },
    );
    sched.submit(subagent(500), RequestClass::Agent).unwrap();
    run_to_idle(&mut sched);

    // A live subagent claims the block and stays in flight while a second
    // request needs room. Advancing once leaves it prefilling.
    sched.submit(subagent(900), RequestClass::Agent).unwrap();
    sched.advance();
    let big = sched
        .submit(input(tokens(2000, 60), None, None, 4), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);

    assert_eq!(
        chunk_widths(&compute, big).iter().sum::<usize>(),
        60,
        "the live request still ran"
    );
    assert_eq!(
        sched.prefix_pinned_pages(),
        2,
        "and the block survived: giving it up while a claimant held it would \
         have freed nothing"
    );
}

// ── The operator's switch ───────────────────────────────────────────────

#[test]
fn prompt_reuse_off_retains_no_prefix() {
    // A cold bench measures a cold engine: nothing is retained, so the head
    // goes back to the whole prompt's pages and a later subagent pays its
    // block again.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(
        compute.clone(),
        SchedulerConfig {
            prompt_reuse: false,
            ..config()
        },
    );
    let id = sched.submit(subagent(500), RequestClass::Agent).unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        chunk_widths(&compute, id),
        vec![48, 12],
        "cut at the whole prompt's pages, the way it was before #186"
    );
    assert_eq!(
        sched.prefix_pinned_pages(),
        0,
        "nothing is retained past the publisher"
    );

    let later = sched.submit(subagent(900), RequestClass::Agent).unwrap();
    let events = run_to_idle(&mut sched);
    assert!(prefix_reuses(&events, later).is_empty());
    assert_eq!(chunk_widths(&compute, later).iter().sum::<usize>(), 60);
}

/// [`subagent`] with a four-placeholder image of content `digest` in its
/// question, at prompt tokens 38..42 — past the block, inside the page the
/// opener's head ends on.
fn subagent_with_image(query: u32, digest: u8) -> RequestInput {
    let mut input = subagent(query);
    let count = input.tokens.len();
    input.multimodal = Some(Arc::new(Multimodal {
        positions: (0..3).flat_map(|_| 0..count as i32).collect(),
        rope_delta: -2,
        media: vec![MediaItem {
            grid: Grid { t: 1, h: 2, w: 8 },
            token_span: TokenSpan { begin: 38, count: 4 },
            patches: vec![0; 4 * 4 * 1536],
            content_digest: [digest; 32],
        }],
    }));
    input
}

#[test]
fn a_multimodal_prompt_ending_at_a_checkpoint_opener_does_not_resume_from_it() {
    // GitHub #193: a claim that reaches the prompt's end prefills nothing, and
    // for an image prompt the prefill is what hands the leaf its rope delta.
    // A checkpoint's own capture always leaves a tail, but a claimant's
    // prompt can stop exactly at that opener: it takes the whole-page head
    // below instead, and prefills the rest.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    sched
        .submit(subagent_with_image(500, 0xAA), RequestClass::Agent)
        .unwrap();
    run_to_idle(&mut sched);

    let mut short = subagent_with_image(500, 0xAA);
    short.tokens.truncate(57);
    short.opener_tokens = None;
    let multimodal = Arc::get_mut(short.multimodal.as_mut().unwrap()).unwrap();
    multimodal.positions = (0..3).flat_map(|_| 0..57).collect();
    let short = sched.submit(short, RequestClass::Agent).unwrap();
    let events = run_to_idle(&mut sched);
    assert!(state_reuses(&events, short).is_empty(), "{events:?}");
    assert_eq!(prefix_reuses(&events, short), [48]);
    assert_eq!(chunk_widths(&compute, short), [9]);
}

#[test]
fn a_multimodal_burst_shares_the_block_whatever_image_each_subagent_sends() {
    // GitHub #193: a prompt carrying an image publishes its retained prefix
    // like any other, because the prefix is keyed by its images too. The
    // block holds no image, so a subagent sending another picture still skips
    // it — and nothing past it, where the pictures differ.
    let compute = Arc::new(MockCompute::new());
    let mut sched = scheduler(compute.clone(), config());
    sched
        .submit(subagent_with_image(500, 0xAA), RequestClass::Agent)
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(
        sched.prefix_pinned_pages(),
        BLOCK / PAGE + 1,
        "the block, plus the opener's page chained over it (#187)"
    );

    let other = sched
        .submit(subagent_with_image(500, 0xBB), RequestClass::Agent)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(
        prefix_reuses(&events, other),
        [BLOCK / PAGE * PAGE],
        "the block and no further: the chained head holds the other picture"
    );
    assert!(state_reuses(&events, other).is_empty(), "nor its checkpoint: {events:?}");

    // The same picture and question is the first subagent's own history, all
    // the way to its generation opener.
    let same = sched
        .submit(subagent_with_image(500, 0xAA), RequestClass::Agent)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(
        state_reuses(&events, same),
        [(ReuseSource::Device, 57)],
        "{events:?}"
    );
    assert_eq!(first_job(&compute, same).start_position, 57);
}
