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
use ignis_core::checkpoint::ReuseSource;
use ignis_core::gpu_profile;
use ignis_core::{
    ConcreteScheduler, DecodeParams, KvFormat, RequestClass, RequestId, RequestInput, SchedEvent,
    Scheduler, SchedulerConfig, Speculation, TokenId,
};
use ignis_runtime::{auto_kv_pool_bytes, CudaLeaf, CudaLeafConfig, Model, RuntimeCompute, KV_PAGE_TOKENS};

pub const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 2048;
const CHUNK: u32 = 1024;
const GENERATED: u32 = 24;
const GIB: u64 = 1024 * 1024 * 1024;

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
    let leaf = CudaLeaf::new(device, reader, artifact, handles, config);
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
        let compute = Arc::new(RuntimeCompute::new(Arc::clone(&self.model), self.eos));
        ConcreteScheduler::with_config(
            SchedulerConfig {
                model: "kv-ram-gpu".into(),
                kv_page_tokens: KV_PAGE_TOKENS,
                max_sequence_tokens: MAX_CONTEXT,
                kv_capacity_pages: MAX_CONTEXT / KV_PAGE_TOKENS,
                resident_slot_capacity: resident_slots,
                serving_chunk_tokens: CHUNK,
                host_capacity_bytes: 16 * GIB,
                retained_pool_bytes: 4 * GIB,
                ..SchedulerConfig::default()
            },
            compute,
        )
    }

    /// A system-and-tools block several pages long, which every request here
    /// opens with — the retained prefix a qwen-code request always carries.
    fn block(&self) -> Vec<TokenId> {
        let mut block = self.tokens("<|im_start|>system\n");
        while block.len() < 3 * KV_PAGE_TOKENS as usize {
            block.extend(self.tokens(
                "You are a careful coding assistant. You answer in short, precise sentences and \
                 you never invent APIs. When a question is about Rust, you name the crate and \
                 the function you rely on. ",
            ));
        }
        block.extend(self.tokens("<|im_end|>\n"));
        block
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
