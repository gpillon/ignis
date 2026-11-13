//! Structured output's exactness on the 27B (GitHub #587, #588 and #590,
//! spec #287, ADR 0089): with greedy decoding, an answer served check-first
//! -- each draw checked against the grammar before the next round consumes
//! it, a miss repaired from the scores its call kept -- is the answer
//! masking every step gives, and speculation never changes it. The answers
//! are held to `json_object` (three prompts the model answers in prose,
//! behind a fence, or as an object) and to each case of the grammar's corpus
//! (`crates/grammar/tests/corpus/cases.json`: its schemas compiled strict as
//! a request's are, its prompts as written, the schema not described).
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
//! 3. **References**, the model loaded with DFlash2:
//!    - **the prefill route's**, the spike's method (`crates/core/examples/
//!      structured_harness.rs` on `spike/287-structured-output`): the served
//!      tokens teacher-forced one at a time through the prefill route, whose
//!      readout gives each position's full logits; the expected token is the
//!      argmax over the grammar's permitted set there.
//!    - **the verify route's**, for a DFlash2 answer: its tokens
//!      teacher-forced through verify rounds at width 1, as served
//!      (`step::decode_program_verify_forced`), each round's logits read
//!      (`dflash2_readout::verify_logits`); the expected token is the argmax
//!      over the permitted set of the verify column that predicted it.
//!
//! Each answer with DFlash2 off is the prefill route's reference token for
//! token, up to a near-tie: where the two part, the engine's two prefill
//! routes must call the two candidates one (`support/near_tie.rs`, the
//! tolerance the speculative tests use), a route's argmax not being
//! invariant there, and the comparison goes on from the served token. Each
//! answer with DFlash2 on is the answer with it off, token for token, or
//! parts from it at such a near-tie; and every token of it is the verify
//! route's reference, exactly. The verify traversal is a route of its own: a
//! column's scores are the same bit for bit whichever round it falls in, and
//! whatever columns follow it (measured at width 1: one run of eight cut
//! after three, the columns compared), but they part from both prefill
//! routes' by more than those part from each other -- on this test's second
//! prompt the DFlash2 answer's token 188 is 8944, which its verify column
//! ranks over 1628 by 1.25 logits where both prefill routes rank 1628 over it
//! by 1.5 and 1.375, and the same answer comes from a request with no grammar
//! at all, its first token forced as a literal. So a speculative answer is
//! held to the scores it was drawn from, with no tolerance: a draft the cut
//! kept that the grammar refuses, a repair from the wrong column or a state
//! the cut left wrong is a token that is not its column's masked argmax.
//! Every served token is permitted where it stands, an answer that stopped
//! ends on a stop the grammar allows, and its text parses as JSON (an object
//! for `json_object`). The DFlash2 half drafted, and repaired at least one
//! miss from a verify pass's scores.
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

use ignis_artifact::{bind_model_scope_27b, materialize, CudaDevice, DraftModule, FrontendSet, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::constrained::PermittedMask;
use ignis_core::dflash2_readout::verify_logits;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b_with_speculation, Model};
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::{
    capture_decode_graphs, decode_program_verify_forced, prefill_program, prefill_program_sampled, repair_program,
    SamplingParams, VerifyLane,
};
use ignis_core::structured::{AnswerGrammar, StructuredKind};
use ignis_core::{
    DecodeParams, FinishReason, KvFormat, RequestClass, RequestInput, SchedEvent, Speculation, SpeculativeBackend,
    TokenId,
};
use ignis_grammar::json::Json;
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
    // Read with the grammar's own parser: a serde_json value would sort a
    // schema's keys, and the grammar writes them in the schema's order.
    let corpus = Json::parse(&std::fs::read_to_string(CORPUS).unwrap_or_else(|e| panic!("{CORPUS}: {e}")))
        .unwrap_or_else(|e| panic!("{CORPUS}: {e}"));
    let Json::Arr(entries) = corpus else { panic!("the corpus is a list of cases") };
    for entry in &entries {
        let name = entry.get("name").and_then(Json::as_str).expect("a case's name").to_owned();
        let (grammar, kind) = match entry.get("schema") {
            None => (grammars.json_object(), StructuredKind::JsonObject),
            Some(schema) => (
                grammars.json_schema(&schema.compact(), true).unwrap_or_else(|e| panic!("{name}: {e}")),
                StructuredKind::JsonSchema,
            ),
        };
        let prompt = render(entry.get("prompt").and_then(Json::as_str).expect("a case's prompt"));
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

/// A row of the verify logits, BF16 widened.
fn widened(row: &[u16]) -> Vec<f32> {
    row.iter().map(|&bits| f32::from_bits(u32::from(bits) << 16)).collect()
}

/// Holds a DFlash2 `answer` to the verify route's own scores: its tokens
/// teacher-forced through verify rounds at width 1, as it was served, each
/// token the argmax over the grammar's permitted set of the column that
/// predicted it -- exactly, since that column is bit for bit the one it was
/// drawn from, whichever round it fell in. The first token is the prefill's
/// draw, held to the prefill readout the same way. Returns the positions
/// past the first where the column's own argmax missed: the misses a verify
/// pass's scores repaired.
fn hold_to_verify_route(
    model: &Model,
    new_pool: &dyn Fn() -> SeqPool,
    case: &Case,
    answer: &Served,
    eos: TokenId,
    what: &str,
) -> usize {
    let pool = new_pool();
    let _graphs = capture_decode_graphs(model, &pool).unwrap_or_else(|e| panic!("{what}: capture: {e}"));
    let mut sequence = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    let mut readout = vec![0f32; vocab()];
    prefill_program_sampled(model, &pool, &mut sequence, &as_i32(&case.prompt), 0, SamplingParams::greedy(), Some(&mut readout))
        .unwrap_or_else(|e| panic!("{what}: prompt prefill: {e}"));
    // The recording: the served tokens, then the stop a finished answer
    // ended on, or -- after an answer cut at its cap -- a last token only
    // for the leaf to take as the final run's successor, never checked.
    let checked = answer.tokens.len() + usize::from(answer.reason == FinishReason::Stop);
    let recording: Vec<i32> = as_i32(&answer.tokens).into_iter().chain([eos as i32]).collect();
    let mut matcher = Matcher::new(case.grammar.clone());
    let mut set = TokenSet::new(vocab());
    let mut hold = |index: usize, logits: &[f32]| -> bool {
        let token = recording[index] as TokenId;
        matcher.mask(&mut set);
        assert!(set.contains(token), "{what}: served token {index} ({token}) is not permitted where it stands");
        assert_eq!(
            token,
            masked_argmax(logits, &set),
            "{what}: token {index} is not the argmax over the permitted set of the scores it was drawn from"
        );
        assert!(token == eos || matcher.accept(token), "{what}: token {index} accepted");
        logits_argmax(logits) != token
    };
    if checked == 0 {
        return 0;
    }
    hold(0, &readout);
    // The prefill drew the model's own token: the served answer's first is
    // the repair of it, when it missed.
    if sequence.pending_token() != Some(recording[0]) {
        let only = PermittedMask::from_ids(vocab() as u32, &[recording[0] as u32]).unwrap_or_else(|e| panic!("mask: {e}"));
        repair_program(model, &pool, &mut sequence, SamplingParams::greedy(), &only)
            .unwrap_or_else(|e| panic!("{what}: the first token: {e}"));
    }
    let columns = WINDOW as usize + 1;
    let (mut anchor, mut repaired) = (0usize, 0usize);
    while anchor + 1 < checked {
        let mut forced: Vec<i32> = recording[anchor + 1..].iter().copied().take(columns).collect();
        forced.resize(columns, -1);
        // Never a draft past the recording's last token, which is then the
        // round's successor at most.
        let lane = VerifyLane { remaining_tokens: (recording.len() - 1 - anchor) as u32, ..VerifyLane::greedy(&[]) };
        let runs = decode_program_verify_forced(model, &pool, &mut [&mut sequence], &[lane], WINDOW, &[&forced])
            .unwrap_or_else(|e| panic!("{what}: the round at token {anchor}: {e}"));
        let logits = verify_logits(model, 1, WINDOW).unwrap_or_else(|e| panic!("{what}: verify logits: {e}"));
        let committed = runs[0].tokens.len();
        assert_eq!(runs[0].tokens, recording[anchor..anchor + committed], "{what}: the round at token {anchor} replayed the answer");
        for column in 0..committed {
            let index = anchor + column + 1;
            if index < checked && hold(index, &widened(&logits[column * vocab()..(column + 1) * vocab()])) {
                repaired += 1;
            }
        }
        anchor += committed;
    }
    repaired
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
    let provider = ArtifactTemplateProvider::serving(
        FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}")),
        ignis_core::compute::ModelFamily::Qwen38_27b,
        ignis_grammar::GrammarLimits::default(),
    );
    let cases = cases(&provider);
    let Some(off) = serve(&frontend, &cases, eos, false).await else { return };
    let Some(on) = serve(&frontend, &cases, eos, true).await else { return };

    // ── the references: the prefill route's readouts and the verify route's,
    // teacher-forced, on one load with DFlash2 ──────────────────────────────
    let (plan, handles) =
        bind_model_scope_27b(&reader, Some(DraftModule::Dflash2)).unwrap_or_else(|e| panic!("bind with dflash2: {e}"));
    let mut device = CudaDevice::create(0).unwrap_or_else(|e| panic!("CUDA: {e}"));
    let artifact = materialize(&reader, &plan, &mut device, None).unwrap_or_else(|e| panic!("materialize: {e}"));
    let speculation = Speculation::new(SpeculativeBackend::Dflash2, WINDOW).expect("dflash2-7");
    let model = load_qwen38_27b_with_speculation(&reader, &artifact, &handles, CONTEXT, CONTEXT, KvFormat::Bf16, Some(speculation))
        .unwrap_or_else(|e| panic!("model load (dflash2, window {WINDOW}): {e}"));
    let new_pool = || {
        SeqPool::create_with_speculation(
            &ModelConfig::qwen38_27b(),
            &SeqPoolBudget {
                kv_format: KvFormat::Bf16,
                kv_page_group_count: CONTEXT.div_ceil(64),
                max_context_tokens: CONTEXT,
                slot_count: 1,
                retained_slot_count: 0,
                retained_host_slot_count: 0,
            },
            Some(SpeculativeBackend::Dflash2),
        )
        .unwrap_or_else(|e| panic!("seq pool: {e}"))
    };
    let (mut missed, mut parted, mut repaired) = (0, 0, 0);
    for ((case, off), on) in cases.iter().zip(&off).zip(&on) {
        let what = &case.name;
        assert_eq!(off.drafted, 0, "{what}: speculation off drafted nothing");
        missed += hold_to_reference(&model, &new_pool, case, off, eos, &format!("{what} (dflash2 off)"));
        // DFlash2 on: the verify route's own masked argmax, exactly ...
        repaired += hold_to_verify_route(&model, &new_pool, case, on, eos, &format!("{what} (dflash2 on)"));
        // ... and the answer with it off, or the same up to a near-tie.
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
         near-tie, {drafted} drafts proposed with dflash2 on, {repaired} misses repaired from a verify pass's scores",
        cases.len()
    );
    assert!(missed > 0, "the prompts exercised at least one repair");
    assert!(drafted > 0, "speculation stayed on for the constrained requests");
    assert!(repaired > 0, "no miss was repaired from a verify pass's scores: the DFlash2 half exercised no cut");
}
