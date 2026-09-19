//! GPU experiment: can the served Qwen 3.8 27B NVFP4 answer a typed
//! decision by its **option-letter logits alone** -- one prefill, zero
//! decoded tokens?
//!
//! This is the go/no-go probe for a `/v1/classify` endpoint in the shape of
//! TypeSafe's Jev `POST /v1/systemone` (one `state`, a map of `questions`,
//! an answer with `probabilities` per option). The readout itself is the
//! technique SemIf (github.com/TheoLeeCJ/SemIf) measured on a 4B: build a
//! prompt that names each option by an uppercase letter, prefill it, and
//! read the last position's logits restricted to those letters' token ids.
//! `prefill_program`'s `out_logits` (GitHub #72) already hands those over,
//! so nothing below the kernel leaf needs to change to *measure* this.
//!
//! **What decides the answer is not accuracy.** The 27B is a different model
//! from anything SemIf scored, so its accuracy is a reference, not a target.
//! The number that says whether a restricted softmax is signal or noise is
//! where the probability mass actually goes:
//!
//!   * `argmax_in_slots` -- is the model's own unrestricted winner one of the
//!     declared letters, or is it `\n` / `{` / a thinking token?
//!   * `allowed_mass` -- exp(logsumexp(slot logits) - logsumexp(all logits)),
//!     how much of the distribution the declared options hold at all.
//!
//! If the mass sits outside the slots, the restricted softmax is renormalized
//! noise however plausible its argmax looks, and the plumbing above the leaf
//! (a readout through the `Compute` seam, a prefill-only request kind, the
//! endpoint) would be built on nothing.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, fixture or kernel error is a **skip**; under the
//! profile the same condition is a **hard failure**.

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ignis_artifact::{CudaDevice, FrontendSet, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::prefill_program;
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};

use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::template::{ChatMessage, TemplateProvider};
use ignis_server::thinking::ThinkingOptions;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

/// Long enough for every authored row's rendered prompt with room to spare;
/// the longest in the fixture is a few hundred tokens.
const MAX_CONTEXT: u32 = 2048;

/// SemIf's answer alphabet, verbatim -- 16 options is its ceiling. Jev's own
/// is 255, which no single-token letter alphabet reaches; that is a design
/// question for the endpoint, not for this probe.
const LETTERS: &str = "ABCDEFGHIJKLMNOP";

/// SemIf's `DIRECT_SYSTEM`, verbatim (`src/semif_phase1/core.py`): the probe
/// measures their prompt on our model, so a prompt difference cannot be
/// mistaken for a model difference.
const DIRECT_SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";

/// One row of SemIf's `benchmarks/data/authored144.jsonl`. `label` is the
/// **index into `options`** of the authored answer.
#[derive(Deserialize)]
struct Row {
    id: String,
    family: String,
    state: JsonValue,
    question: String,
    options: Vec<Opt>,
    label: usize,
}

#[derive(Deserialize)]
struct Opt {
    id: String,
    description: String,
}

/// What one row produced, or why it produced nothing.
struct Readout {
    id: String,
    family: String,
    label: usize,
    predicted: usize,
    probabilities: Vec<f64>,
    option_logits: Vec<f32>,
    allowed_mass: f64,
    argmax_in_slots: bool,
    full_vocab_argmax: u32,
    prompt_tokens: usize,
    /// The declared options' own ids, in slot order -- what an answer is
    /// keyed by, rather than the letter that carried it.
    option_ids: Vec<String>,
    /// Wall time for this row's prefill alone, in microseconds -- the whole
    /// GPU cost of one decision, since no token is ever decoded.
    prefill_micros: u128,
}

/// SemIf's authored decision fixture, committed beside this test.
///
/// Committed rather than fetched because this is a GPU-profile test: under
/// `IGNIS_GPU_PROFILE=1` a missing fixture is a hard failure, so the
/// serialized sweep must not depend on a file somebody downloaded by hand.
/// See `fixtures/semif/NOTICE.md` for its provenance and licence.
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("semif")
        .join("authored144.jsonl")
}

/// `.scratch/jev-classify/` at the worktree root -- raw per-row output, never
/// committed (`docs/agents/findings.md`: raw material stays in `.scratch/`,
/// the durable synthesis is promoted to `docs/findings/`).
fn scratch_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(".scratch")
        .join("jev-classify")
}

/// SemIf's `direct_messages`: a system instruction plus one user message
/// carrying the whole decision as JSON, evidence first.
///
/// Evidence first is not cosmetic -- it is what makes one `state` shared
/// across many questions a shared *token prefix*, which is the reuse ignis
/// already has (GitHub #191/#193) and the shape Jev's own API is built
/// around (one `state`, a map of `questions`).
fn decision_messages(row: &Row) -> Vec<ChatMessage> {
    let options: Vec<JsonValue> = row
        .options
        .iter()
        .enumerate()
        .map(|(index, option)| {
            json!({
                "letter": &LETTERS[index..index + 1],
                "description": option.description,
            })
        })
        .collect();
    let payload = json!({
        "evidence": row.state,
        "criterion": row.question,
        "options": options,
    });
    vec![
        ChatMessage::text("system", DIRECT_SYSTEM),
        ChatMessage::text("user", payload.to_string()),
    ]
}

/// The answer slots for `count` options: each letter's token id, verified the
/// way SemIf verifies them (`_slot_ids` + `encode_prompt`'s boundary check).
///
/// Three conditions, all of which must hold or the readout is meaningless:
/// the letter is exactly one token; it round-trips through decode; and
/// appending it to the rendered prompt extends the tokenization by exactly
/// that one token rather than re-tokenizing the prompt's tail.
fn slot_ids(
    tokenizer: &ignis_artifact::Tokenizer,
    prompt_text: &str,
    prompt_ids: &[u32],
    count: usize,
) -> Result<Vec<u32>, String> {
    let mut slots = Vec::with_capacity(count);
    for index in 0..count {
        let letter = &LETTERS[index..index + 1];
        let encoded = tokenizer.encode(letter).map_err(|e| format!("encode {letter}: {e}"))?;
        if encoded.len() != 1 {
            return Err(format!("slot {letter} is {} tokens, not one", encoded.len()));
        }
        let decoded = tokenizer
            .decode(&encoded)
            .map_err(|e| format!("decode slot {letter}: {e}"))?;
        if decoded != letter {
            return Err(format!("slot {letter} does not round-trip (decoded {decoded:?})"));
        }
        let with_letter = tokenizer
            .encode(&format!("{prompt_text}{letter}"))
            .map_err(|e| format!("encode the prompt with {letter}: {e}"))?;
        if with_letter.len() != prompt_ids.len() + 1
            || with_letter[..prompt_ids.len()] != *prompt_ids
            || with_letter[prompt_ids.len()] != encoded[0]
        {
            return Err(format!("the answer boundary re-tokenizes the prompt tail at slot {letter}"));
        }
        slots.push(encoded[0]);
    }
    if slots.iter().collect::<std::collections::BTreeSet<_>>().len() != slots.len() {
        return Err("answer-slot tokens collide".to_owned());
    }
    Ok(slots)
}

/// log-sum-exp in f64 over f32 logits.
fn logsumexp(values: impl Iterator<Item = f32> + Clone) -> f64 {
    let maximum = values
        .clone()
        .fold(f64::NEG_INFINITY, |acc, v| acc.max(f64::from(v)));
    if !maximum.is_finite() {
        return maximum;
    }
    maximum + values.map(|v| (f64::from(v) - maximum).exp()).sum::<f64>().ln()
}

fn softmax(values: &[f32]) -> Vec<f64> {
    let total = logsumexp(values.iter().copied());
    values.iter().map(|&v| (f64::from(v) - total).exp()).collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        })
        .0
}

/// Mean of the per-class recalls, over the authored `label` index -- the
/// metric SemIf reports for this fixture.
fn balanced_accuracy(rows: &[Readout]) -> f64 {
    let mut per_class: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    for row in rows {
        let entry = per_class.entry(row.label).or_insert((0, 0));
        entry.1 += 1;
        if row.predicted == row.label {
            entry.0 += 1;
        }
    }
    if per_class.is_empty() {
        return f64::NAN;
    }
    let sum: f64 = per_class
        .values()
        .map(|&(hit, total)| hit as f64 / total as f64)
        .sum();
    sum / per_class.len() as f64
}

fn median(values: &mut Vec<f64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn typed_option_logits_are_readable_from_one_prefill() {
    let fixture = fixture_path();
    let Ok(fixture_text) = std::fs::read_to_string(&fixture) else {
        if gpu_profile::skip_or_fail(&format!("the fixture is absent: {}", fixture.display())) {
            return;
        }
        unreachable!("skip_or_fail panics under the profile");
    };
    let rows: Vec<Row> = fixture_text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap_or_else(|e| panic!("parse a fixture row: {e}")))
        .collect();
    assert!(!rows.is_empty(), "the fixture must carry rows");

    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    // Two sets: `ArtifactTemplateProvider` takes ownership of one, and the
    // slot verification needs the tokenizer beside it.
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let tokenizer_set = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let provider = ArtifactTemplateProvider::new(frontend);
    // Thinking off, always: with it on, the prompt's last position sits
    // inside a `<think>` block and the next token is reasoning, not an
    // answer letter. A classify readout has no reasoning phase to wait out.
    let thinking = ThinkingOptions {
        enable_thinking: false,
        ..ThinkingOptions::default()
    };

    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind text scope: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize the text scope on the device: {e}")) {
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
    .unwrap_or_else(|e| panic!("ignis_model_load: {e}"));
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: ignis_core::KvFormat::Bf16,
            kv_page_group_count: 64,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 1,
            retained_slot_count: 0,
        },
    )
    .unwrap_or_else(|e| panic!("seq pool create: {e}"));

    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let mut logits = vec![0f32; vocab];
    let mut readouts: Vec<Readout> = Vec::new();
    let mut excluded: Vec<(String, String)> = Vec::new();

    for row in &rows {
        if row.options.len() < 2 || row.options.len() > LETTERS.len() {
            excluded.push((row.id.clone(), format!("{} options", row.options.len())));
            continue;
        }
        if row.label >= row.options.len() {
            excluded.push((row.id.clone(), format!("label {} is out of range", row.label)));
            continue;
        }
        let messages = decision_messages(row);
        let prompt_text = match provider.render_text(&messages, &thinking, &[]) {
            Ok(text) => text,
            Err(rejection) => {
                excluded.push((row.id.clone(), format!("render: {}", rejection.message)));
                continue;
            }
        };
        let prompt_ids: Vec<u32> = match provider.apply_chat_template(&messages, &thinking, &[]) {
            Ok(rendered) => rendered.tokens,
            Err(rejection) => {
                excluded.push((row.id.clone(), format!("template: {}", rejection.message)));
                continue;
            }
        };
        if prompt_ids.is_empty() || prompt_ids.len() > MAX_CONTEXT as usize {
            excluded.push((row.id.clone(), format!("{} prompt tokens", prompt_ids.len())));
            continue;
        }
        let slots = match slot_ids(tokenizer_set.tokenizer(), &prompt_text, &prompt_ids, row.options.len()) {
            Ok(slots) => slots,
            Err(reason) => {
                excluded.push((row.id.clone(), reason));
                continue;
            }
        };

        let token_ids: Vec<i32> = prompt_ids
            .iter()
            .map(|&id| i32::try_from(id).expect("token id fits i32"))
            .collect();
        let mut sequence = pool
            .alloc(MAX_CONTEXT)
            .unwrap_or_else(|e| panic!("{}: seq alloc: {e}", row.id));
        logits.fill(0.0);
        let started = std::time::Instant::now();
        let prefilled = prefill_program(&model, &pool, &mut sequence, &token_ids, 0, Some(&mut logits));
        let prefill_micros = started.elapsed().as_micros();
        if let Err(e) = prefilled {
            if gpu_profile::skip_or_fail(&format!("{}: prefill_program: {e}", row.id)) {
                excluded.push((row.id.clone(), format!("prefill: {e}")));
                continue;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
        drop(sequence);

        assert!(
            logits.iter().all(|v| v.is_finite()),
            "{}: the readout's logits must be finite",
            row.id
        );
        let option_logits: Vec<f32> = slots.iter().map(|&slot| logits[slot as usize]).collect();
        let full_vocab_argmax = argmax(&logits) as u32;
        let allowed_mass =
            (logsumexp(option_logits.iter().copied()) - logsumexp(logits.iter().copied())).exp();
        readouts.push(Readout {
            id: row.id.clone(),
            family: row.family.clone(),
            label: row.label,
            predicted: argmax(&option_logits),
            probabilities: softmax(&option_logits),
            option_logits,
            allowed_mass,
            argmax_in_slots: slots.contains(&full_vocab_argmax),
            full_vocab_argmax,
            prompt_tokens: prompt_ids.len(),
            option_ids: row.options.iter().map(|o| o.id.clone()).collect(),
            prefill_micros,
        });
    }

    // ── raw rows, then the summary the go/no-go is read off ──────────────
    let scratch = scratch_dir();
    std::fs::create_dir_all(&scratch).unwrap_or_else(|e| panic!("create {}: {e}", scratch.display()));
    let mut lines = String::new();
    for readout in &readouts {
        let record = json!({
            "id": readout.id,
            "family": readout.family,
            "option_ids": readout.option_ids,
            "label": readout.label,
            "predicted": readout.predicted,
            "correct": readout.predicted == readout.label,
            "probabilities": readout.probabilities,
            "option_logits": readout.option_logits,
            "allowed_mass": readout.allowed_mass,
            "argmax_in_slots": readout.argmax_in_slots,
            "full_vocab_argmax": readout.full_vocab_argmax,
            "prompt_tokens": readout.prompt_tokens,
            "prefill_micros": readout.prefill_micros,
        });
        lines.push_str(&record.to_string());
        lines.push('\n');
    }
    let out = scratch.join("readout.jsonl");
    std::fs::write(&out, lines).unwrap_or_else(|e| panic!("write {}: {e}", out.display()));

    assert!(
        !readouts.is_empty(),
        "every one of the {} rows was excluded before the readout: {excluded:?}",
        rows.len()
    );
    let scored = readouts.len();
    let hits = readouts.iter().filter(|r| r.predicted == r.label).count();
    let in_slots = readouts.iter().filter(|r| r.argmax_in_slots).count();
    let mut masses: Vec<f64> = readouts.iter().map(|r| r.allowed_mass).collect();
    let min_mass = masses.iter().copied().fold(f64::INFINITY, f64::min);
    let median_mass = median(&mut masses);

    eprintln!("ignis classify: rows={} scored={scored} excluded={}", rows.len(), excluded.len());
    for (id, reason) in &excluded {
        eprintln!("ignis classify: excluded {id}: {reason}");
    }
    eprintln!(
        "ignis classify: accuracy={:.3} balanced_accuracy={:.3} (SemIf reference on a 4B: 0.813)",
        hits as f64 / scored as f64,
        balanced_accuracy(&readouts)
    );
    eprintln!(
        "ignis classify: argmax_in_slots={in_slots}/{scored} ({:.1}%) allowed_mass median={median_mass:.4} min={min_mass:.4}",
        100.0 * in_slots as f64 / scored as f64
    );
    let mut latencies: Vec<f64> = readouts.iter().map(|r| r.prefill_micros as f64 / 1000.0).collect();
    let total_ms: f64 = latencies.iter().sum();
    let median_ms = median(&mut latencies);
    let mut prompt_lengths: Vec<f64> = readouts.iter().map(|r| r.prompt_tokens as f64).collect();
    eprintln!(
        "ignis classify: prefill median={median_ms:.1} ms  total={total_ms:.0} ms over {scored} decisions ({:.1} decisions/s, serial, no prefix reuse)",
        1000.0 * scored as f64 / total_ms
    );
    eprintln!(
        "ignis classify: prompt tokens median={:.0} max={:.0}",
        median(&mut prompt_lengths),
        prompt_lengths.iter().copied().fold(0.0, f64::max)
    );
    let mut families: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for readout in &readouts {
        let entry = families.entry(readout.family.as_str()).or_insert((0, 0));
        entry.1 += 1;
        if readout.predicted == readout.label {
            entry.0 += 1;
        }
    }
    for (family, (hit, total)) in &families {
        eprintln!(
            "ignis classify: family {family}: {hit}/{total} ({:.3})",
            *hit as f64 / *total as f64
        );
    }
    // The full-vocabulary winners on the rows where the model did not pick a
    // declared letter: what it wanted to say instead is the whole diagnosis
    // when the mass sits outside the slots.
    let strays: Vec<String> = readouts
        .iter()
        .filter(|r| !r.argmax_in_slots)
        .take(8)
        .map(|r| {
            let text = tokenizer_set
                .tokenizer()
                .decode(&[r.full_vocab_argmax])
                .unwrap_or_else(|_| "<undecodable>".to_owned());
            format!("{}={:?}(mass {:.4})", r.id, text, r.allowed_mass)
        })
        .collect();
    if !strays.is_empty() {
        eprintln!("ignis classify: off-slot winners: {}", strays.join(" "));
    }
    eprintln!("ignis classify: rows written to {}", out.display());
}
