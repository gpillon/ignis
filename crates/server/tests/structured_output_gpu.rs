//! Structured output's exactness on the 27B (GitHub #587, spec #287, ADR
//! 0089): with speculation off and greedy decoding, a `json_object` answer
//! served check-first -- each draw checked against the grammar before the
//! next round consumes it, a miss repaired from the scores its call kept --
//! is the answer masking every step gives.
//!
//! Two halves, the model loaded once for each:
//!
//! 1. **Served.** The production scheduler over the real leaf
//!    (`cuda_scheduler`), its grammar compiled against the artifact's own
//!    tokenizer as the server compiles it, each prompt submitted with the
//!    grammar and its tokens read off the event stream.
//! 2. **Reference.** The spike's method (`crates/core/examples/
//!    structured_harness.rs` on `spike/287-structured-output`): the served
//!    tokens teacher-forced one at a time through the prefill route, whose
//!    readout gives each position's full logits; the expected token is the
//!    argmax over the grammar's permitted set there.
//!
//! The served answer must be the reference's token for token. Where the
//! two differ, the served decode route and the reference's prefill route
//! must call the two candidates a near-tie (`support/near_tie.rs`, the
//! tolerance the speculative tests use): a route's argmax is not invariant
//! there, and the comparison goes on from the served token. Every served
//! token is permitted where it stands, the answer ends on a stop the
//! grammar allows, and its text parses as a JSON object.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile it is a failure.
//!
//! ```text
//! cargo test -p ignis-server --features cuda --test structured_output_gpu -- --ignored --nocapture
//! ```

#![cfg(feature = "cuda")]

#[allow(dead_code)]
#[path = "../../core/tests/support/near_tie.rs"]
mod near_tie;

use std::path::Path;
use std::sync::Arc;

use ignis_artifact::{bind_text_scope_27b, materialize, CudaDevice, FrontendSet, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::prefill_program;
use ignis_core::structured::{AnswerGrammar, StructuredKind};
use ignis_core::{DecodeParams, FinishReason, KvFormat, RequestClass, RequestInput, SchedEvent, TokenId};
use ignis_grammar::{Grammar, Matcher, TokenSet};
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::engine::Engine;
use ignis_server::runtime::{cuda_scheduler, EngineShape};
use ignis_server::telemetry::SystemClock;
use ignis_server::template::{ChatMessage, TemplateProvider};
use ignis_server::thinking::ThinkingOptions;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
/// Enough for each prompt's object; a cut answer is compared as far as it
/// went.
const MAX_TOKENS: u32 = 192;
/// The reference's sequences: the longest prompt and its answer, with room.
const CONTEXT: u32 = 1024;

/// Prompts the model answers in prose, behind a fence, or as an object:
/// misses at the first token, on keys and at the end.
const PROMPTS: &[&str] = &[
    "Describe a cat as a JSON object with its name, its age and its three favourite toys.",
    "Name three rivers of Europe and say how long each one is.",
    "What is the capital of Italy? Answer in one word.",
];

fn vocab() -> usize {
    ModelConfig::qwen38_27b().vocab as usize
}

/// The argmax over `set` of `logits`, the lower id on a tie: the vendored
/// sampler's greedy rule.
fn masked_argmax(logits: &[f32], set: &TokenSet) -> TokenId {
    let mut best: Option<TokenId> = None;
    for id in set.iter() {
        if best.is_none_or(|b| logits[id as usize] > logits[b as usize]) {
            best = Some(id);
        }
    }
    best.expect("a grammar never leaves an empty permitted set")
}

/// What the served half returns for one prompt.
struct Served {
    prompt: Vec<TokenId>,
    tokens: Vec<TokenId>,
    reason: FinishReason,
}

/// The served half: every prompt through the production scheduler, held to
/// `grammar`. `None` on a skip.
async fn serve(
    frontend: &FrontendSet,
    provider: &ArtifactTemplateProvider,
    grammar: &Arc<Grammar>,
    eos: TokenId,
) -> Option<Vec<Served>> {
    let shape = EngineShape { kv_format: KvFormat::Bf16, speculation: None, speculation_off: true, ..EngineShape::default() };
    let scheduler = match cuda_scheduler(Path::new(ARTIFACT), MODEL.into(), eos, shape) {
        Ok((scheduler, _reservations)) => scheduler,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("cuda_scheduler: {e}")) {
                return None;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let (engine, driver) = Engine::with_clock_and_driver(Box::new(scheduler), Arc::new(SystemClock));
    let off = ThinkingOptions { enable_thinking: false, ..ThinkingOptions::default() };
    let mut served = Vec::new();
    for text in PROMPTS {
        let message: ChatMessage = serde_json::from_value(serde_json::json!({ "role": "user", "content": text })).unwrap();
        let rendered = provider
            .apply_chat_template(std::slice::from_ref(&message), &off, &[])
            .unwrap_or_else(|e| panic!("render: {}", e.message));
        let input = RequestInput {
            model: MODEL.into(),
            tokens: rendered.tokens.clone(),
            params: DecodeParams { max_tokens: Some(MAX_TOKENS), ..DecodeParams::default() },
            multimodal: None,
            opener_tokens: None,
            user_turn_tokens: None,
            system_block_tokens: None,
            reuse_boundaries: Vec::new(),
            decision: None,
            constrained: None,
            forced_literal: None,
            grammar: Some(AnswerGrammar::new(grammar.clone(), StructuredKind::JsonObject)),
            warm_up: false,
        };
        let (_, mut stream) = engine.submit(input, RequestClass::Interactive).await.expect("admitted");
        let mut tokens = Vec::new();
        let reason = loop {
            match stream.recv().await.expect("the request finishes") {
                SchedEvent::Token { token, .. } => tokens.push(token),
                SchedEvent::Done { reason, .. } => break reason,
                _ => {}
            }
        };
        eprintln!(
            "served {text:?}: {} tokens, {reason:?}: {:?}",
            tokens.len(),
            frontend.tokenizer().decode(&tokens).unwrap_or_default()
        );
        served.push(Served { prompt: rendered.tokens, tokens, reason });
    }
    // The model thread owns the leaf: joined before the reference loads its own.
    drop(engine);
    let _ = driver.join();
    Some(served)
}

#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn json_object_answers_equal_the_per_token_masked_argmax() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().expect("the 27B's generation config names its EOS");
    // The grammar as the server compiles it: the provider's own cache over
    // the logits row, the EOS ending a complete answer.
    let provider = ArtifactTemplateProvider::new(
        FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}")),
    )
    .with_grammar_vocabulary(vocab(), eos);
    let grammar = provider.grammars().expect("the 27B's vocabulary reads as bytes").json_object();
    let Some(served) = serve(&frontend, &provider, &grammar, eos).await else { return };

    // ── the reference: the prefill route's readouts, teacher-forced ──────
    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind: {e}"));
    let mut device = CudaDevice::create(0).unwrap_or_else(|e| panic!("CUDA: {e}"));
    let artifact = materialize(&reader, &plan, &mut device, None).unwrap_or_else(|e| panic!("materialize: {e}"));
    let model = load_qwen38_27b(&reader, &artifact, &handles, CONTEXT, CONTEXT, KvFormat::Bf16)
        .unwrap_or_else(|e| panic!("model load: {e}"));
    let new_pool = || {
        SeqPool::create(
            &ModelConfig::qwen38_27b(),
            &SeqPoolBudget {
                kv_format: KvFormat::Bf16,
                kv_page_group_count: CONTEXT.div_ceil(64),
                max_context_tokens: CONTEXT,
                slot_count: 1,
                retained_slot_count: 0,
                retained_host_slot_count: 0,
            },
        )
        .unwrap_or_else(|e| panic!("seq pool: {e}"))
    };
    let mut repaired_positions = 0;
    for (text, answer) in PROMPTS.iter().zip(&served) {
        let pool = new_pool();
        let mut sequence = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        let mut logits = vec![0f32; vocab()];
        let prompt: Vec<i32> = answer.prompt.iter().map(|&t| t as i32).collect();
        prefill_program(&model, &pool, &mut sequence, &prompt, 0, Some(&mut logits))
            .unwrap_or_else(|e| panic!("{text}: prompt prefill: {e}"));
        let mut matcher = Matcher::new(grammar.clone());
        let mut set = TokenSet::new(vocab());
        let mut position = prompt.len() as u64;
        // The served tokens, then the stop a finished answer ended on.
        let stop = (answer.reason == FinishReason::Stop).then_some(eos);
        for (index, &token) in answer.tokens.iter().chain(stop.iter()).enumerate() {
            matcher.mask(&mut set);
            assert!(set.contains(token), "{text}: served token {index} ({token}) is not permitted where it stands");
            let expected = masked_argmax(&logits, &set);
            if !matcher.check(logits_argmax(&logits)) {
                repaired_positions += 1;
            }
            if expected != token {
                let agreed: Vec<i32> = answer.tokens[..index].iter().map(|&t| t as i32).collect();
                near_tie::assert_near_tie(
                    &model,
                    &new_pool,
                    CONTEXT,
                    &prompt,
                    &agreed,
                    expected as i32,
                    token as i32,
                    &format!("{text}: token {index}"),
                );
                eprintln!("{text}: token {index}: served {token}, reference {expected}, a near-tie");
            }
            assert!(matcher.accept(token), "{text}: token {index} accepted");
            if Some(token) == stop {
                break;
            }
            prefill_program(&model, &pool, &mut sequence, &[token as i32], position, Some(&mut logits))
                .unwrap_or_else(|e| panic!("{text}: token {index}: {e}"));
            position += 1;
        }
        let content = frontend.tokenizer().decode(&answer.tokens).unwrap_or_default();
        match answer.reason {
            FinishReason::Stop => {
                assert!(matcher.is_finished(), "{text}: the answer ended on a stop the grammar allows");
                let value: serde_json::Value =
                    serde_json::from_str(&content).unwrap_or_else(|e| panic!("{text}: not JSON ({e}): {content:?}"));
                assert!(value.is_object(), "{text}: not an object: {content:?}");
            }
            FinishReason::Length => assert_eq!(answer.tokens.len() as u32, MAX_TOKENS, "{text}: cut at its cap"),
            other => panic!("{text}: ended {other:?}"),
        }
    }
    eprintln!("structured output gpu: {repaired_positions} positions where the free argmax missed");
    assert!(repaired_positions > 0, "the prompts exercised at least one repair");
}

/// The unrestricted argmax: what a free greedy draw would have been.
fn logits_argmax(logits: &[f32]) -> TokenId {
    let mut best = 0usize;
    for (id, &value) in logits.iter().enumerate() {
        if value > logits[best] {
            best = id;
        }
    }
    best as TokenId
}
