//! Shared body of the KV-RAM GPU legs (GitHub #190): a real scheduler over a
//! real `CudaLeaf`, so what is compared is what a client would receive.
//!
//! Every comparison is **token for token** against a run in which the state
//! never moved, and every run decodes alone, so the batch a round runs in is
//! the same shape on both sides and a difference can only come from the
//! transfer.

use std::path::Path;
use std::sync::Arc;

use ignis_artifact::{bind_model_scope_27b, materialize, CudaDevice, DraftModule, FrontendSet, Reader};
use ignis_core::checkpoint::{RetainedStateOperation, ReuseSource};
use ignis_core::gpu_profile;
use ignis_core::scheduler::CheckpointClaim;
use ignis_core::{
    Compute, ConcreteScheduler, DecodeJob, DecodeParams, KvFormat, N_DECODE_LANES, PrefillJob,
    RequestClass, RequestId, RequestInput, RetainedAt, SchedEvent, Scheduler, SchedulerConfig,
    Speculation, TokenId,
};
use ignis_runtime::{auto_kv_pool_bytes, CudaLeaf, CudaLeafConfig, Model, RuntimeCompute, KV_PAGE_TOKENS};

pub const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 2048;
const CHUNK: u32 = 1024;
const GENERATED: u32 = 24;
const GIB: u64 = 1024 * 1024 * 1024;
/// The KV-RAM tier these legs run on: the leaf pins exactly this as its
/// arena (GitHub #213) and the scheduler budgets exactly this in bytes, so
/// the ledger and the arena are two views of one region — which is what lets
/// a leg assert they agree.
///
/// Four gibibytes rather than the sixteen this was before the arena: the
/// budget is now RAM actually page-locked for the run, and no leg here holds
/// more than a handful of blobs of a 2048-token geometry.
pub const HOST_POOL_BYTES: u64 = 4 * GIB;

pub struct Loaded {
    model: Arc<Model<CudaLeaf>>,
    eos: TokenId,
    frontend: FrontendSet,
}

/// Load the artifact onto the card, with the drafter when `speculation` asks
/// for it. `None` when the profile allows the leg to skip (it fails otherwise).
pub fn load(speculation: Option<Speculation>) -> Option<Loaded> {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend
        .eos_token_id()
        .unwrap_or_else(|| panic!("qwen3.8-27b generation config must carry eos_token_id"));
    let draft = speculation.map(|_| DraftModule::Dflash2);
    let (plan, handles) =
        bind_model_scope_27b(&reader, draft).unwrap_or_else(|e| panic!("bind: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return None;
            }
            unreachable!();
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return None;
            }
            unreachable!();
        }
    };
    let config = CudaLeafConfig {
        max_context_tokens: MAX_CONTEXT,
        kv_format: KvFormat::default(),
        kv_pool_bytes: auto_kv_pool_bytes(KvFormat::default(), MAX_CONTEXT),
        prefill_chunk_tokens: CHUNK,
        speculation,
        ..CudaLeafConfig::default()
    };
    let leaf = CudaLeaf::new(device, reader, artifact, handles, config)
        .with_kv_ram_arena(HOST_POOL_BYTES)
        .unwrap_or_else(|e| panic!("pin KV-RAM: {e}"));
    let model = Arc::new(Model::load(Arc::new(leaf)).unwrap_or_else(|e| panic!("model load: {e:?}")));
    Some(Loaded { model, eos, frontend })
}

impl Loaded {
    fn tokens(&self, text: &str) -> Vec<TokenId> {
        self.frontend
            .tokenizer()
            .encode(text)
            .unwrap_or_else(|e| panic!("tokenize: {e}"))
    }

    /// A fresh scheduler over its own adapter. Its device pool is one full
    /// sequence (`MAX_CONTEXT`) however much the leaf holds, so a request
    /// needing all of it takes every retained page back; `resident_slots`
    /// bounds how many sequences may be on the card at once.
    fn scheduler(&self, resident_slots: u32) -> ConcreteScheduler {
        self.scheduler_with_retained_slots(resident_slots, N_DECODE_LANES as u32)
    }

    /// [`Loaded::scheduler`] handing out `retained_slots` of the leaf's
    /// retained slots (GitHub #215): fewer than the leaf holds is fine, more is
    /// not.
    fn scheduler_with_retained_slots(&self, resident_slots: u32, retained_slots: u32) -> ConcreteScheduler {
        let compute = Arc::new(RuntimeCompute::new(Arc::clone(&self.model), self.eos));
        ConcreteScheduler::with_config(
            SchedulerConfig {
                model: "kv-ram-gpu".into(),
                kv_page_tokens: KV_PAGE_TOKENS,
                max_sequence_tokens: MAX_CONTEXT,
                // And the page a claimed checkpoint's opener ends inside,
                // which its claimant cannot take back (GitHub #215).
                kv_capacity_pages: MAX_CONTEXT / KV_PAGE_TOKENS + 1,
                resident_slot_capacity: resident_slots,
                serving_chunk_tokens: CHUNK,
                host_capacity_bytes: HOST_POOL_BYTES,
                retained_slots,
                ..SchedulerConfig::default()
            },
            compute,
        )
    }

    /// A system-and-tools block several pages long, which every request here
    /// opens with — the retained prefix a qwen-code request always carries.
    fn block(&self) -> Vec<TokenId> {
        self.block_of(3 * KV_PAGE_TOKENS as usize)
    }

    /// A system block of at least `tokens` tokens.
    fn block_of(&self, tokens: usize) -> Vec<TokenId> {
        self.block_saying(
            tokens,
            "You are a careful coding assistant. You answer in short, precise sentences and you \
             never invent APIs. When a question is about Rust, you name the crate and the \
             function you rely on. ",
        )
    }

    /// A system block of at least `tokens` tokens repeating `text`: another
    /// burst's block when `text` differs.
    fn block_saying(&self, tokens: usize, text: &str) -> Vec<TokenId> {
        let mut block = self.tokens("<|im_start|>system\n");
        while block.len() < tokens {
            block.extend(self.tokens(text));
        }
        block.extend(self.tokens("<|im_end|>\n"));
        block
    }

    /// A short user turn after `block`, closed with the generation opener: its
    /// opener lands at least a page past the block's last whole page and never
    /// on a page boundary, so a request publishes the block, chains a link at
    /// the opener's page and keeps a tail page with its checkpoint. Returns the
    /// tokens and where the opener ends.
    fn short_turn(&self, block: &[TokenId], question: &str) -> (Vec<TokenId>, u32) {
        let page = KV_PAGE_TOKENS as usize;
        let mut prompt = block.to_vec();
        prompt.extend(self.tokens(&format!("<|im_start|>user\n{question}")));
        loop {
            let opener = prompt.len() + self.tokens("<|im_end|>\n<|im_start|>assistant\n").len();
            if opener / page > block.len() / page && opener % page != 0 {
                break;
            }
            prompt.extend(self.tokens(" Please."));
        }
        prompt.extend(self.tokens("<|im_end|>\n<|im_start|>assistant\n"));
        let opener = prompt.len() as u32;
        prompt.extend(self.tokens("<think>\n\n</think>\n\n"));
        (prompt, opener)
    }

    /// A user turn padded past KV-RAM's restore floor, closed with the
    /// generation opener. Returns the tokens and where the opener ends.
    fn long_turn(&self, block: &[TokenId]) -> (Vec<TokenId>, u32) {
        let mut prompt = block.to_vec();
        let paragraph = "The scheduler keeps retained state on the device while there is room, \
                         and gives it up to KV-RAM when a live request needs the pages. ";
        prompt.extend(self.tokens("<|im_start|>user\nRead this log and summarise it.\n"));
        while prompt.len() < 1_300 {
            prompt.extend(self.tokens(paragraph));
        }
        prompt.extend(self.tokens("<|im_end|>\n<|im_start|>assistant\n"));
        let opener = prompt.len() as u32;
        prompt.extend(self.tokens("<think>\n\n</think>\n\n"));
        (prompt, opener)
    }

    fn request(&self, tokens: Vec<TokenId>, opener: Option<u32>, block: u32, max: u32) -> RequestInput {
        RequestInput {
            decision: None,
            model: "kv-ram-gpu".into(),
            tokens,
            params: DecodeParams {
                max_tokens: Some(max),
                ..DecodeParams::default()
            },
            multimodal: None,
            opener_tokens: opener,
            user_turn_tokens: None,
            system_block_tokens: Some(block),
            constrained: None,
        }
    }
}

fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    while !sched.is_idle() {
        events.extend(sched.advance());
        if let Some(error) = sched.last_error() {
            panic!("the leaf failed a step: {error}");
        }
    }
    events
}

fn generated(events: &[SchedEvent], request: RequestId) -> Vec<TokenId> {
    events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::Token { request: r, token } if *r == request => Some(*token),
            _ => None,
        })
        .collect()
}

fn reuse(events: &[SchedEvent], request: RequestId) -> Option<(ReuseSource, u32)> {
    events.iter().find_map(|e| match e {
        SchedEvent::StateReused {
            request: r,
            source,
            tokens,
            ..
        } if *r == request => Some((*source, *tokens)),
        _ => None,
    })
}

/// A request that needs the whole device pool — and so every retained page.
fn whole_pool(loaded: &Loaded) -> RequestInput {
    let mut prompt = Vec::new();
    while prompt.len() < (MAX_CONTEXT - 16) as usize {
        prompt.extend(loaded.tokens("Unrelated filler that shares nothing with the conversation. "));
    }
    prompt.truncate((MAX_CONTEXT - 16) as usize);
    RequestInput {
        system_block_tokens: None,
        ..loaded.request(prompt, None, 0, 8)
    }
}

/// AC1 (GitHub #190): an idle conversation pushed off the device is restored
/// from KV-RAM and continues exactly as it does from the device checkpoint
/// that never moved.
pub fn an_idle_conversation_resumes_from_kv_ram_exactly(loaded: &Loaded) {
    let mut sched = loaded.scheduler(8);
    let block = loaded.block();
    let block_tokens = block.len() as u32;
    let (turn_n, opener) = loaded.long_turn(&block);

    // Turn N+1 has no opener of its own, so it leaves no checkpoint that a
    // replay of it could match instead of turn N's.
    let mut turn_n1 = turn_n[..opener as usize].to_vec();
    turn_n1.extend(loaded.tokens("Sure.<|im_end|>\n<|im_start|>user\nNow in one line.<|im_end|>\n"));
    turn_n1.extend(loaded.tokens("<|im_start|>assistant\n<think>\n\n</think>\n\n"));

    sched
        .submit(loaded.request(turn_n, Some(opener), block_tokens, 4), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);
    assert!(
        sched
            .checkpoint_pool()
            .entries()
            .iter()
            .any(|e| e.tier == ReuseSource::Device && e.tokens == opener),
        "turn N left a device checkpoint at its opener"
    );

    // Control: turn N+1 from the checkpoint that never left the device.
    let control = sched
        .submit(
            loaded.request(turn_n1.clone(), None, block_tokens, GENERATED),
            RequestClass::Interactive,
        )
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuse(&events, control), Some((ReuseSource::Device, opener)));
    let expected = generated(&events, control);
    assert!(expected.len() > 1, "the control generated past its first token: {expected:?}");

    // Push it off the device.
    sched.submit(whole_pool(loaded), RequestClass::Interactive).unwrap();
    run_to_idle(&mut sched);
    assert!(
        sched
            .checkpoint_pool()
            .entries()
            .iter()
            .any(|e| e.tier == ReuseSource::KvRam && e.tokens == opener),
        "turn N's checkpoint was spilled, not discarded: {:?}",
        sched.checkpoint_pool().entries()
    );

    // The same turn N+1, from KV-RAM.
    let restored = sched
        .submit(loaded.request(turn_n1, None, block_tokens, GENERATED), RequestClass::Interactive)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuse(&events, restored), Some((ReuseSource::KvRam, opener)));
    assert_eq!(
        generated(&events, restored),
        expected,
        "a KV-RAM restore must generate exactly what the device checkpoint does"
    );
}

/// AC2 (GitHub #190): a request standing on a retained prefix, snapshotted to
/// KV-RAM mid-decode with that prefix's pages materialized into its blob,
/// continues exactly as it does when nothing evicts it.
pub fn a_retained_prefix_claimant_evicted_mid_decode_continues_exactly(loaded: &Loaded) {
    let block = loaded.block();
    let block_tokens = block.len() as u32;
    let question = |text: &str| {
        let mut prompt = block.clone();
        prompt.extend(loaded.tokens(&format!(
            "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )));
        prompt
    };
    let burst_first = question("Name one Rust crate for HTTP clients.");
    // A long answer: a drafting load lands several tokens a round, and the
    // intruder has to arrive while the claimant is still decoding.
    let claimant_prompt = question(
        "List ten Rust crates for parsing command-line arguments, one per line, each with a \
         sentence on what sets it apart.",
    );
    let intruder = RequestInput {
        system_block_tokens: None,
        ..loaded.request(loaded.tokens("Count from one to five."), None, 0, 8)
    };

    // One resident sequence at a time: every request decodes alone, and the
    // intruder can only run by snapshotting the claimant.
    let run = |with_intruder: bool| -> (Vec<TokenId>, Vec<SchedEvent>, RequestId) {
        let mut sched = loaded.scheduler(1);
        sched
            .submit(
                loaded.request(burst_first.clone(), None, block_tokens, 4),
                RequestClass::Interactive,
            )
            .unwrap();
        run_to_idle(&mut sched);
        let claimant = sched
            .submit(
                loaded.request(claimant_prompt.clone(), None, block_tokens, 96),
                RequestClass::Agent,
            )
            .unwrap();
        let mut events = Vec::new();
        while generated(&events, claimant).len() < 2 {
            events.extend(sched.advance());
            assert!(!sched.is_idle(), "the claimant finished before it could be evicted");
        }
        if with_intruder {
            sched.submit(intruder.clone(), RequestClass::Interactive).unwrap();
        }
        events.extend(run_to_idle(&mut sched));
        (generated(&events, claimant), events, claimant)
    };

    let (expected, control_events, claimant) = run(false);
    assert!(
        control_events.iter().any(|e| matches!(
            e,
            SchedEvent::PrefixReused { request, retained: true, .. } if *request == claimant
        )),
        "the claimant stands on the retained block"
    );
    assert!(expected.len() > 16, "the control decoded well past the point the intruder arrives");

    let (actual, events, claimant) = run(true);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, SchedEvent::Evicted { request, .. } if *request == claimant)),
        "the claimant was snapshotted while holding the prefix"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == claimant)),
        "and restored from its materialized blob"
    );
    assert!(!events.iter().any(|e| matches!(e, SchedEvent::Requeued { .. })));
    assert_eq!(
        actual, expected,
        "a claimant restored from a materialized blob must continue exactly"
    );
}

/// GitHub #190: a burst's system block, given up by the device into KV-RAM
/// and brought back by the next subagent, serves it exactly as the block that
/// never left does.
pub fn a_burst_block_brought_back_from_kv_ram_serves_exactly(loaded: &Loaded) {
    // Past the restore floor, so bringing it back is worth the crossing.
    let block = loaded.block_of(1_100);
    let block_tokens = block.len() as u32;
    let subagent = |text: &str, max| {
        let mut prompt = block.clone();
        prompt.extend(loaded.tokens(&format!(
            "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )));
        loaded.request(prompt, None, block_tokens, max)
    };

    let run = |pressure: bool| -> (Vec<TokenId>, Vec<SchedEvent>, RequestId) {
        let mut sched = loaded.scheduler(8);
        sched
            .submit(subagent("Name one Rust crate for HTTP clients.", 4), RequestClass::Agent)
            .unwrap();
        run_to_idle(&mut sched);
        let mut events = Vec::new();
        if pressure {
            sched.submit(whole_pool(loaded), RequestClass::Interactive).unwrap();
            events.extend(run_to_idle(&mut sched));
        }
        let second = sched
            .submit(
                subagent("Name one Rust crate for serializing to JSON.", GENERATED),
                RequestClass::Agent,
            )
            .unwrap();
        events.extend(run_to_idle(&mut sched));
        // GitHub #213: the tier's byte ledger and the arena the blobs are
        // placed in are two views of one region, checked here rather than at
        // an empty end state -- under pressure the block's blob is still in
        // KV-RAM at this point.
        let (capacity, used) = ignis_core::seq::host_pool_stats();
        assert_eq!(capacity, HOST_POOL_BYTES, "the arena is the tier's whole budget");
        assert_eq!(
            sched.host().used_bytes(),
            used,
            "the ledger and the arena disagree about what KV-RAM holds"
        );
        assert_eq!(used > 0, pressure, "only the pressured run spills a blob");
        (generated(&events, second), events, second)
    };

    let (expected, control, second) = run(false);
    assert!(
        control.iter().any(|e| matches!(
            e,
            SchedEvent::PrefixReused { request, retained: true, .. } if *request == second
        )),
        "the control's second subagent claims the block on the device"
    );
    assert!(expected.len() > 1, "the control generated past its first token");

    let (actual, events, second) = run(true);
    let kv_ram = |operation: RetainedStateOperation| {
        events.iter().any(|e| {
            matches!(
                e,
                // GitHub #216 gave the fact a `kind`; what this asserts is
                // the operation and the tier it happened on, as it always
                // did -- either kind spilling and coming back is the claim.
                SchedEvent::RetainedState { operation: o, source: ReuseSource::KvRam, kind: _ }
                    if *o == operation
            )
        })
    };
    assert!(kv_ram(RetainedStateOperation::Spill), "the block was spilled");
    assert!(kv_ram(RetainedStateOperation::Restore), "and brought back");
    assert!(
        events.iter().any(|e| matches!(
            e,
            SchedEvent::PrefixReused { request, retained: true, tokens, .. }
                if *request == second && *tokens > 0
        )),
        "the second subagent claimed the block that came back"
    );
    assert_eq!(actual, expected, "a block brought back from KV-RAM must serve exactly");
}

/// Greedy decode of `request` on `compute` until `count` tokens are committed
/// or it finishes: what a client would receive, round by round.
fn decode_alone(compute: &RuntimeCompute<CudaLeaf>, request: RequestId, count: u32) -> Vec<TokenId> {
    let params = DecodeParams {
        max_tokens: Some(count),
        ..DecodeParams::default()
    };
    let mut tokens = Vec::new();
    while (tokens.len() as u32) < count {
        let outcome = compute
            .decode_step(&[DecodeJob {
                request,
                lane: 0,
                params,
                remaining_tokens: count - tokens.len() as u32,
                permitted: None,
}])
            .unwrap_or_else(|e| panic!("decode {request}: {e:?}"))
            .remove(0);
        tokens.extend(outcome.tokens);
        if outcome.finish.is_some() {
            break;
        }
    }
    tokens
}

/// GitHub #215, AC "bit-exact reuse": turn N+1 standing on turn N's
/// checkpoint — its image in a retained slot, its tail page a page of the
/// pool, the prefixes under it in slots of their own — generates exactly what
/// a cold prefill of turn N+1 split at the same boundaries generates.
///
/// Driven through the compute adapter with the jobs the scheduler would build,
/// so both sides are cut at exactly the same points: the block's page floor,
/// the opener's page floor, the opener.
#[allow(dead_code)] // not every binary that shares this module runs every leg
pub fn a_turn_from_a_retained_slot_generates_what_a_split_cold_prefill_generates(loaded: &Loaded) {
    let compute = RuntimeCompute::new(Arc::clone(&loaded.model), loaded.eos);
    let page = KV_PAGE_TOKENS;
    let block = loaded.block();
    let block_page = block.len() as u32 / page * page;
    let (turn_n, opener) = loaded.short_turn(&block, "Which crate parses TOML?");
    let opener_page = opener / page * page;
    let mut turn_n1 = turn_n[..opener as usize].to_vec();
    turn_n1.extend(loaded.tokens("The `toml` crate.<|im_end|>\n<|im_start|>user\nAnd YAML?<|im_end|>\n"));
    turn_n1.extend(loaded.tokens("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    let job = |request, tokens: &[TokenId], from: u32, to: u32| PrefillJob {
        request,
        tokens: tokens[from as usize..to as usize].to_vec(),
        context_tokens: MAX_CONTEXT,
        start_position: from,
        params: DecodeParams {
            max_tokens: Some(GENERATED),
            ..DecodeParams::default()
        },
        shared_prefix: None,
        publish_prefix: None,
        checkpoint: None,
        capture_checkpoint: None,
        multimodal: None,
        readout: None,
        permitted: None,
};

    // Turn N: publish the block and the chained link, capture at the opener,
    // each image in the retained slot the scheduler would name.
    let turn_n_len = turn_n.len() as u32;
    for (from, to, publish, capture) in [
        (0, block_page, Some(RetainedAt { tokens: block_page, slot: 0 }), None),
        (block_page, opener_page, Some(RetainedAt { tokens: opener_page, slot: 1 }), None),
        (opener_page, opener, None, Some(RetainedAt { tokens: opener, slot: 2 })),
        (opener, turn_n_len, None, None),
    ] {
        let outcome = compute
            .prefill_step(&[PrefillJob {
                publish_prefix: publish,
                capture_checkpoint: capture,
                ..job(1, &turn_n, from, to)
            }])
            .unwrap_or_else(|e| panic!("turn N prefill {from}..{to}: {e:?}"));
        assert_eq!(outcome[0].checkpoint_captured, capture.is_some(), "turn N capture at {to}");
    }
    compute.release(1);

    // Turn N+1 from the checkpoint.
    let turn_n1_len = turn_n1.len() as u32;
    compute
        .prefill_step(&[PrefillJob {
            checkpoint: Some(CheckpointClaim {
                publisher: 1,
                tokens: opener,
                source: ReuseSource::Device,
            }),
            ..job(2, &turn_n1, opener, turn_n1_len)
        }])
        .unwrap_or_else(|e| panic!("turn N+1 from the checkpoint: {e:?}"));
    let reused = decode_alone(&compute, 2, GENERATED);
    compute.release(2);

    // The control: turn N+1 cold, cut where turn N and the claim cut it.
    for (from, to) in [
        (0, block_page),
        (block_page, opener_page),
        (opener_page, opener),
        (opener, turn_n1_len),
    ] {
        compute
            .prefill_step(&[job(3, &turn_n1, from, to)])
            .unwrap_or_else(|e| panic!("cold split prefill {from}..{to}: {e:?}"));
    }
    let cold = decode_alone(&compute, 3, GENERATED);
    compute.release(3);

    assert!(cold.len() > 1, "the control generated past its first token: {cold:?}");
    assert_eq!(
        reused, cold,
        "a turn standing on retained slots must generate what the split cold prefill generates"
    );
    compute.release_checkpoint(1);
    compute.release_prefix(1, opener_page);
    compute.release_prefix(1, block_page);
}

/// GitHub #215, AC "spill": a conversation whose checkpoint loses its retained
/// slot to another request — not its pages — is spilled to KV-RAM, and its
/// next turn restored from there generates exactly what it generates from the
/// device checkpoint that was never given up.
#[allow(dead_code)] // not every binary that shares this module runs every leg
pub fn a_checkpoint_given_up_for_a_slot_resumes_from_kv_ram_exactly(loaded: &Loaded) {
    let block = loaded.block();
    let block_tokens = block.len() as u32;
    let (turn_n, opener) = loaded.long_turn(&block);
    let mut turn_n1 = turn_n[..opener as usize].to_vec();
    turn_n1.extend(loaded.tokens("Sure.<|im_end|>\n<|im_start|>user\nNow in one line.<|im_end|>\n"));
    turn_n1.extend(loaded.tokens("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    // Another burst's short turn: few pages, so it needs no page turn N holds —
    // only retained slots.
    let other_block = loaded.block_saying(
        3 * KV_PAGE_TOKENS as usize,
        "You are a terse shell assistant. You answer with one command and nothing else. ",
    );
    let other_block_tokens = other_block.len() as u32;
    let (other, other_opener) = loaded.short_turn(&other_block, "List hidden files.");

    // The control: plenty of slots, turn N+1 from the device.
    let mut sched = loaded.scheduler(8);
    sched
        .submit(loaded.request(turn_n.clone(), Some(opener), block_tokens, 4), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);
    let control = sched
        .submit(loaded.request(turn_n1.clone(), None, block_tokens, GENERATED), RequestClass::Interactive)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuse(&events, control), Some((ReuseSource::Device, opener)));
    let expected = generated(&events, control);
    assert!(expected.len() > 1, "the control generated past its first token: {expected:?}");
    drop(sched);

    // Three slots: exactly turn N's block, link and checkpoint.
    let mut sched = loaded.scheduler_with_retained_slots(8, 3);
    sched
        .submit(loaded.request(turn_n, Some(opener), block_tokens, 4), RequestClass::Interactive)
        .unwrap();
    run_to_idle(&mut sched);
    assert_eq!(sched.retained_slots_in_use(), 3, "block, link and checkpoint");
    let other = sched
        .submit(loaded.request(other, Some(other_opener), other_block_tokens, 4), RequestClass::Agent)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert!(
        !events.iter().any(|e| matches!(e, SchedEvent::RetainedSlotSkipped { request, .. } if *request == other)),
        "the other burst found its slots by giving retained state up: {events:?}"
    );
    assert!(
        sched
            .checkpoint_pool()
            .entries()
            .iter()
            .any(|e| e.tier == ReuseSource::KvRam && e.tokens == opener),
        "turn N's checkpoint gave its slot up to KV-RAM: {:?}",
        sched.checkpoint_pool().entries()
    );

    let restored = sched
        .submit(loaded.request(turn_n1, None, block_tokens, GENERATED), RequestClass::Interactive)
        .unwrap();
    let events = run_to_idle(&mut sched);
    assert_eq!(reuse(&events, restored), Some((ReuseSource::KvRam, opener)));
    assert_eq!(
        generated(&events, restored),
        expected,
        "a checkpoint spilled for a slot must restore exactly"
    );
}
