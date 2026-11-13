//! Structured output's exactness on the 27B (GitHub #587 and #590, spec
//! #287, ADR 0089): with greedy decoding, an answer served check-first --
//! each draw checked against the grammar before the next round consumes it,
//! a miss repaired from the scores its call kept -- is the answer masking
//! every step gives, and speculation never changes it.
//!
//! Three halves, the model loaded once for each:
//!
//! 1. **Served, speculation off.** The production scheduler over the real
//!    leaf (`cuda_scheduler`), its grammars compiled against the artifact's
//!    own tokenizer as the server compiles them -- `json_object`, and each
//!    schema of the corpus (`crates/grammar/tests/corpus/cases.json`)
//!    straight through `ignis-grammar` -- each prompt submitted with its
//!    grammar and its tokens read off the event stream.
//! 2. **Served, DFlash2 on** (#590): the same, every round a verify round in
//!    which the host cuts the constrained lane's run at its first miss and
//!    repairs it from the verify pass's scores.
//! 3. **Reference.** The spike's method (`crates/core/examples/
//!    structured_harness.rs` on `spike/287-structured-output`): the served
//!    tokens teacher-forced one at a time through the prefill route, whose
//!    readout gives each position's full logits; the expected token is the
//!    argmax over the grammar's permitted set there.
//!
//! Each answer with DFlash2 on is the answer with it off, token for token;
//! where the two part, the engine's two prefill routes must call the two
//! candidates a near-tie (`support/near_tie.rs`, the tolerance the
//! speculative tests use), and both answers then go on to the reference on
//! their own. Each served answer is the reference's token for token, up to
//! a near-tie the same way: a route's argmax is not invariant there, and the
//! comparison goes on from the served token. Every served token is
//! permitted where it stands, an answer that stopped ends on a stop the
//! grammar allows, and its text parses as JSON (an object for
//! `json_object`). The DFlash2 half drafted.
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
use ignis_core::model_load::{load_qwen38_27b, Model};
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::prefill_program;
use ignis_core::structured::{AnswerGrammar, StructuredKind};
use ignis_core::{
    DecodeParams, FinishReason, KvFormat, RequestClass, RequestInput, SchedEvent, Speculation, SpeculativeBackend,
    TokenId,
};
use ignis_grammar::{Grammar, Matcher, TokenSet};
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::engine::Engine;
use ignis_server::runtime::{cuda_scheduler, EngineShape};
use ignis_server::telemetry::SystemClock;
use ignis_server::template::{ChatMessage, TemplateProvider};
use ignis_server::thinking::ThinkingOptions;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
/// The schemas of the grammar's corpus, and its two `json_object` cases.
const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../grammar/tests/corpus/cases.json");
/// Enough for each prompt's object; a cut answer is compared as far as it
/// went.
const MAX_TOKENS: u32 = 192;
/// The reference's sequences: the longest prompt and its answer, with room.
const CONTEXT: u32 = 1024;
/// DFlash2-7, every built-in profile's window.
const WINDOW: u32 = 7;

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

/// One constrained prompt: its name, its prompt rendered as the server
/// renders it, and the grammar its answer is held to.
struct Case {
    name: String,
    prompt: Vec<TokenId>,
    grammar: Arc<Grammar>,
    kind: StructuredKind,
}

/// The three prompts under `json_object`, and every case of the corpus under
/// its own grammar: `json_object`, or its schema compiled strict.
fn cases(provider: &ArtifactTemplateProvider) -> Vec<Case> {
    let grammars = provider.grammars().expect("the 27B's vocabulary reads as bytes");
    let off = ThinkingOptions { enable_thinking: false, ..ThinkingOptions::default() };
    let render = |text: &str| -> Vec<TokenId> {
        let message: ChatMessage = serde_json::from_value(serde_json::json!({ "role": "user", "content": text })).unwrap();
        provider
            .apply_chat_template(std::slice::from_ref(&message), &off, &[])
            .unwrap_or_else(|e| panic!("render: {}", e.message))
            .tokens
    };
    let mut cases: Vec<Case> = PROMPTS
        .iter()
        .map(|text| Case {
            name: (*text).into(),
            prompt: render(text),
            grammar: grammars.json_object(),
            kind: StructuredKind::JsonObject,
        })
        .collect();
    let corpus: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(CORPUS).unwrap_or_else(|e| panic!("{CORPUS}: {e}"))).unwrap();
    for case in corpus.as_array().expect("the corpus is a list of cases") {
        let name = case["name"].as_str().expect("a case's name").to_owned();
        let (grammar, kind) = match case.get("schema") {
            None => (grammars.json_object(), StructuredKind::JsonObject),
            Some(schema) => (
                grammars.json_schema(&schema.to_string(), true).unwrap_or_else(|e| panic!("{name}: {e}")),
                StructuredKind::JsonSchema,
            ),
        };
        let prompt = render(case["prompt"].as_str().expect("a case's prompt"));
        cases.push(Case { name, prompt, grammar, kind });
    }
    cases
}

/// What one served answer was.
struct Served {
    tokens: Vec<TokenId>,
    reason: FinishReason,
    /// Drafts proposed to its verify rounds; 0 with speculation off.
    drafted: u32,
}

/// The served half: every case through the production scheduler, held to
/// its grammar, with DFlash2 on or off. `None` on a skip.
async fn serve(frontend: &FrontendSet, cases: &[Case], eos: TokenId, dflash2: bool) -> Option<Vec<Served>> {
    let shape = match dflash2 {
        true => EngineShape {
            kv_format: KvFormat::Bf16,
            speculation: Some(Speculation::new(SpeculativeBackend::Dflash2, WINDOW).expect("dflash2-7")),
            ..EngineShape::default()
        },
        false => EngineShape { kv_format: KvFormat::Bf16, speculation: None, speculation_off: true, ..EngineShape::default() },
    };
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
    let mut served = Vec::new();
    for case in cases {
        let input = RequestInput {
            model: MODEL.into(),
            tokens: case.prompt.clone(),
            params: DecodeParams { max_tokens: Some(MAX_TOKENS), ..DecodeParams::default() },
            multimodal: None,
            opener_tokens: None,
            user_turn_tokens: None,
            system_block_tokens: None,
            reuse_boundaries: Vec::new(),
            decision: None,
            constrained: None,
            forced_literal: None,
            grammar: Some(AnswerGrammar::new(case.grammar.clone(), case.kind)),
            warm_up: false,
        };
        let (_, mut stream) = engine.submit(input, RequestClass::Interactive).await.expect("admitted");
        let mut tokens = Vec::new();
        let (reason, spec) = loop {
            match stream.recv().await.expect("the request finishes") {
                SchedEvent::Token { token, .. } => tokens.push(token),
                SchedEvent::Done { reason, spec, .. } => break (reason, spec),
                _ => {}
            }
        };
        let drafted = spec.map_or(0, |spec| spec.drafted);
        eprintln!(
            "served {} (dflash2 {dflash2}): {} tokens, {reason:?}, {drafted} drafted: {:?}",
            case.name,
            tokens.len(),
            frontend.tokenizer().decode(&tokens).unwrap_or_default()
        );
        served.push(Served { tokens, reason, drafted });
    }
    // The model thread owns the leaf: joined before the next load.
    drop(engine);
    let _ = driver.join();
    Some(served)
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

fn as_i32(tokens: &[TokenId]) -> Vec<i32> {
    tokens.iter().map(|&t| t as i32).collect()
}

/// The served token at `index`, the EOS once an answer that stopped ended.
fn token_at(answer: &Served, index: usize, eos: TokenId) -> TokenId {
    answer.tokens.get(index).copied().unwrap_or(eos)
}

/// Holds `answer` to the reference: the per-token masked argmax of the
/// prefill route's readouts, up to a near-tie. Returns the positions where
/// the free argmax missed.
fn hold_to_reference(
    model: &Model,
    new_pool: &dyn Fn() -> SeqPool,
    case: &Case,
    answer: &Served,
    eos: TokenId,
    what: &str,
) -> usize {
    let pool = new_pool();
    let mut sequence = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    let mut logits = vec![0f32; vocab()];
    let prompt = as_i32(&case.prompt);
    prefill_program(model, &pool, &mut sequence, &prompt, 0, Some(&mut logits))
        .unwrap_or_else(|e| panic!("{what}: prompt prefill: {e}"));
    let mut matcher = Matcher::new(case.grammar.clone());
    let mut set = TokenSet::new(vocab());
    let mut position = prompt.len() as u64;
    let mut missed = 0;
    // The served tokens, then the stop a finished answer ended on.
    let stop = (answer.reason == FinishReason::Stop).then_some(eos);
    for (index, &token) in answer.tokens.iter().chain(stop.iter()).enumerate() {
        matcher.mask(&mut set);
        assert!(set.contains(token), "{what}: served token {index} ({token}) is not permitted where it stands");
        let expected = masked_argmax(&logits, &set);
        if !matcher.check(logits_argmax(&logits)) {
            missed += 1;
        }
        if expected != token {
            near_tie::assert_near_tie(
                model,
                new_pool,
                CONTEXT,
                &prompt,
                &as_i32(&answer.tokens[..index]),
                expected as i32,
                token as i32,
                &format!("{what}: token {index}"),
            );
            eprintln!("{what}: token {index}: served {token}, reference {expected}, a near-tie");
        }
        assert!(matcher.accept(token), "{what}: token {index} accepted");
        if Some(token) == stop {
            break;
        }
        prefill_program(model, &pool, &mut sequence, &[token as i32], position, Some(&mut logits))
            .unwrap_or_else(|e| panic!("{what}: token {index}: {e}"));
        position += 1;
    }
    match answer.reason {
        FinishReason::Stop => assert!(matcher.is_finished(), "{what}: the answer ended on a stop the grammar allows"),
        FinishReason::Length => assert_eq!(answer.tokens.len() as u32, MAX_TOKENS, "{what}: cut at its cap"),
        other => panic!("{what}: ended {other:?}"),
    }
    missed
}

#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn constrained_answers_equal_the_per_token_masked_argmax_with_dflash2_off_and_on() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().expect("the 27B's generation config names its EOS");
    // The grammars as the server compiles them: the provider's own cache
    // over the logits row, the EOS ending a complete answer.
    let provider = ArtifactTemplateProvider::new(
        FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}")),
    )
    .with_grammar_vocabulary(vocab(), eos);
    let cases = cases(&provider);
    let Some(off) = serve(&frontend, &cases, eos, false).await else { return };
    let Some(on) = serve(&frontend, &cases, eos, true).await else { return };

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
    let (mut missed, mut parted) = (0, 0);
    for ((case, off), on) in cases.iter().zip(&off).zip(&on) {
        let what = &case.name;
        assert_eq!(off.drafted, 0, "{what}: speculation off drafted nothing");
        missed += hold_to_reference(&model, &new_pool, case, off, eos, &format!("{what} (dflash2 off)"));
        // DFlash2 on: the same answer, or the same up to a near-tie.
        if off.tokens != on.tokens || off.reason != on.reason {
            let at = near_tie::first_divergence(&as_i32(&off.tokens), &as_i32(&on.tokens))
                .unwrap_or(off.tokens.len().min(on.tokens.len()));
            near_tie::assert_near_tie(
                &model,
                &new_pool,
                CONTEXT,
                &as_i32(&case.prompt),
                &as_i32(&off.tokens[..at]),
                token_at(off, at, eos) as i32,
                token_at(on, at, eos) as i32,
                &format!("{what}: dflash2 on against off, token {at}"),
            );
            eprintln!("{what}: dflash2 on and off part at token {at}, a near-tie");
            parted += 1;
            hold_to_reference(&model, &new_pool, case, on, eos, &format!("{what} (dflash2 on)"));
        }
        for (answer, label) in [(off, "off"), (on, "on")] {
            if answer.reason != FinishReason::Stop {
                continue;
            }
            let content = frontend.tokenizer().decode(&answer.tokens).unwrap_or_default();
            let value: serde_json::Value = serde_json::from_str(&content)
                .unwrap_or_else(|e| panic!("{what} (dflash2 {label}): not JSON ({e}): {content:?}"));
            if case.kind == StructuredKind::JsonObject {
                assert!(value.is_object(), "{what} (dflash2 {label}): not an object: {content:?}");
            }
        }
    }
    let drafted: u32 = on.iter().map(|answer| answer.drafted).sum();
    eprintln!(
        "structured output gpu: {} cases, {missed} positions where the free argmax missed, {parted} answers parted at a \
         near-tie, {drafted} drafts proposed with dflash2 on",
        cases.len()
    );
    assert!(missed > 0, "the prompts exercised at least one repair");
    assert!(drafted > 0, "speculation stayed on for the constrained requests");
}
