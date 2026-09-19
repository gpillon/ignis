//! Where does the answer alphabet stop working?
//!
//! `slot_alphabet.rs` counts 1,255 labels that are exactly one token in this
//! tokenizer. That is a fact about the **tokenizer**. The endpoint's real
//! ceiling is a fact about the **model**: at some number of declared options
//! it stops putting its next-token mass on the answer tokens at all, and past
//! that point the restricted softmax is renormalized noise however plausible
//! its argmax looks.
//!
//! This walks one fixed set of decisions up through 2, 8, 16, 32, 64 and 128
//! declared options and reports, at each width, how much of the distribution
//! the answer tokens hold and how often the model's own unrestricted winner
//! is one of them.
//!
//! The fixture is a support-desk taxonomy generated here rather than SemIf's
//! authored decisions: those 144 rows carry only **nine** distinct option
//! descriptions between them (three families of three), so there is nothing
//! to widen them with. Routing is a domain where 128 categories are ordinary,
//! and the taxonomy is built as `area - action` pairs so that every option is
//! a plausible destination and no two are the same phrase.
//!
//! The tickets name their area and action almost literally, so the right
//! answer is not in doubt. That makes accuracy here a **sanity check** — a
//! model that cannot route "I was charged twice for the same invoice" to
//! `Billing - duplicate charges` is not being tested on width — and never a
//! quality claim: the fixture is ours and it is easy. Mass is what this
//! measures, and mass is a property of the model's distribution that does not
//! care whether the task is easy.
//!
//! The alphabet is drawn in the order the endpoint would draw it: `A`-`Z`,
//! then `a`-`z`, then `0`-`9`, then uppercase bigrams — so a width of 128
//! exercises labels like `AA` and `BJ` that no multiple-choice question the
//! model was trained on ever used.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38).

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{CudaDevice, FrontendSet, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::prefill_program;
use serde_json::json;

use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::template::{ChatMessage, TemplateProvider};
use ignis_server::thinking::ThinkingOptions;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// 256 options of prose is a long prompt.
const MAX_CONTEXT: u32 = 8192;
const WIDTHS: &[usize] = &[8, 32, 64, 128, 192, 256];

const DIRECT_SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";
const CRITERION: &str = "Which queue should handle this support ticket?";

/// One routing decision: a ticket, and the index of the category that should
/// take it.
struct Ticket {
    id: &'static str,
    text: &'static str,
    /// Index into [`categories`].
    label: usize,
}

/// 16 actions x 16 areas = 256 plausible destinations in a fixed order, so a
/// width of N is always the first N of the same list and every ticket's right
/// answer keeps its index at every width it survives.
fn categories() -> Vec<String> {
    const AREAS: [&str; 16] = [
        "Billing", "Payments", "Account access", "Subscriptions", "Refunds", "Invoicing",
        "Data export", "Integrations", "API access", "Notifications", "Permissions", "Security",
        "Performance", "Mobile app", "Reporting", "Onboarding",
    ];
    const ACTIONS: [&str; 16] = [
        "duplicate charges", "failed transactions", "cancellation requests", "setup help",
        "unexpected behaviour", "missing records", "configuration changes", "upgrade requests",
        "downgrade requests", "access reviews", "outage reports", "data corrections",
        "billing disputes", "renewal questions", "migration help", "compliance requests",
    ];
    let mut out = Vec::with_capacity(AREAS.len() * ACTIONS.len());
    for action in ACTIONS {
        for area in AREAS {
            out.push(format!("{area} - {action}"));
        }
    }
    out
}

/// Tickets whose destination is unambiguous, spread across the taxonomy so the
/// right answer is not always among the first few options. A ticket is scored
/// at a width only if its answer fits inside it.
const TICKETS: &[Ticket] = &[
    Ticket { id: "t01", text: "I was charged twice for the same invoice this month.", label: 0 },
    Ticket { id: "t02", text: "My card payment keeps getting declined at checkout.", label: 17 },
    Ticket { id: "t03", text: "I cannot sign in since I reset my password yesterday.", label: 34 },
    Ticket { id: "t04", text: "Please help me set up the annual plan for my team.", label: 51 },
    Ticket { id: "t05", text: "The refund you issued shows as pending forever, which is not right.", label: 68 },
    Ticket { id: "t06", text: "Three invoices from March are missing from my history.", label: 85 },
    Ticket { id: "t07", text: "I need to change which columns the CSV export includes.", label: 102 },
    Ticket { id: "t08", text: "We would like to upgrade our Salesforce integration tier.", label: 119 },
    Ticket { id: "t09", text: "Two API keys were created for the same service by mistake.", label: 8 },
    Ticket { id: "t10", text: "Notification emails are failing to send to our domain.", label: 25 },
    Ticket { id: "t11", text: "Please cancel the extra admin permission we requested.", label: 42 },
    Ticket { id: "t12", text: "We need help configuring SSO for our security team.", label: 59 },
    Ticket { id: "t13", text: "The mobile app crashes whenever I open the reports tab.", label: 77 },
    Ticket { id: "t14", text: "Dashboards take over a minute to load since this morning.", label: 12 },
    Ticket { id: "t15", text: "Walk me through connecting our first workspace, please.", label: 63 },
    Ticket { id: "t16", text: "Two subscription renewals were billed on the same day.", label: 3 },
];

/// The endpoint's answer alphabet, in the order it would be drawn from:
/// single characters first, then uppercase bigrams.
///
/// **Filtered against the loaded tokenizer**, which is not an optimization but
/// the whole correctness of the thing. 114 of the 676 uppercase bigrams are
/// two tokens in this one (`slot_alphabet.rs`) — `BQ`, `CJ`, `DQ` and the rest
/// of the letter pairs English never writes — and a label that is two tokens
/// reads the logit of its first, which another label shares. The alphabet is
/// therefore a property of the loaded model, computed at load, never a
/// constant compiled in.
fn answer_alphabet(tokenizer: &ignis_artifact::Tokenizer, count: usize) -> Vec<String> {
    let clean = |label: &str| {
        tokenizer
            .encode(label)
            .ok()
            .filter(|ids| ids.len() == 1)
            .filter(|ids| tokenizer.decode(ids).ok().as_deref() == Some(label))
            .map(|ids| ids[0])
    };
    let mut labels = Vec::with_capacity(count);
    let mut taken = std::collections::BTreeSet::new();
    let singles = ('A'..='Z').chain('a'..='z').chain('0'..='9').map(|c| c.to_string());
    let bigrams = ('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}")));
    for label in singles.chain(bigrams) {
        if labels.len() >= count {
            break;
        }
        if let Some(id) = clean(&label) {
            if taken.insert(id) {
                labels.push(label);
            }
        }
    }
    assert_eq!(labels.len(), count, "the tokenizer offers fewer than {count} clean answer tokens");
    labels
}

fn logsumexp(values: impl Iterator<Item = f32> + Clone) -> f64 {
    let maximum = values
        .clone()
        .fold(f64::NEG_INFINITY, |acc, v| acc.max(f64::from(v)));
    if !maximum.is_finite() {
        return maximum;
    }
    maximum + values.map(|v| (f64::from(v) - maximum).exp()).sum::<f64>().ln()
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

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_answer_alphabet_holds_its_mass_up_to_some_width() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let all = categories();
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let tokenizer_set = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let provider = ArtifactTemplateProvider::new(frontend);
    let thinking = ThinkingOptions { enable_thinking: false, ..ThinkingOptions::default() };

    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind text scope: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(d) => d,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(a) => a,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize the text scope: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT, ignis_core::KvFormat::Bf16)
        .unwrap_or_else(|e| panic!("ignis_model_load: {e}"));
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: ignis_core::KvFormat::Bf16,
            kv_page_group_count: MAX_CONTEXT / 64,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 1,
            retained_slot_count: 0,
        },
    )
    .unwrap_or_else(|e| panic!("seq pool create: {e}"));
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let mut logits = vec![0f32; vocab];

    eprintln!(
        "{:>6} {:>7} {:>7} {:>9} {:>9} {:>9} {:>8} {:>8}",
        "width", "scored", "tokens", "mass p50", "mass min", "top p50", "in-slot", "correct"
    );
    for &width in WIDTHS {
        let labels = answer_alphabet(tokenizer_set.tokenizer(), width);
        // Every label must still be one clean token at this width, or the
        // number below would be about tokenization rather than about the model.
        let slots: Vec<u32> = labels
            .iter()
            .map(|label| {
                let ids = tokenizer_set
                    .tokenizer()
                    .encode(label)
                    .unwrap_or_else(|e| panic!("encode {label}: {e}"));
                assert_eq!(ids.len(), 1, "label {label} is not one token at width {width}");
                ids[0]
            })
            .collect();
        let options: Vec<serde_json::Value> = all[..width]
            .iter()
            .enumerate()
            .map(|(i, description)| json!({ "letter": labels[i], "description": description }))
            .collect();

        let mut masses: Vec<f64> = Vec::new();
        let mut tops: Vec<f64> = Vec::new();
        let (mut in_slots, mut hits, mut scored) = (0usize, 0usize, 0usize);
        let mut prompt_tokens = 0usize;
        for ticket in TICKETS {
            if ticket.label >= width {
                continue;
            }
            let payload = json!({
                "evidence": ticket.text,
                "criterion": CRITERION,
                "options": options,
            });
            let messages = [
                ChatMessage::text("system", DIRECT_SYSTEM),
                ChatMessage::text("user", payload.to_string()),
            ];
            let rendered = provider
                .apply_chat_template(&messages, &thinking, &[])
                .unwrap_or_else(|e| panic!("{}: template at width {width}: {}", ticket.id, e.message));
            assert!(
                !rendered.tokens.is_empty() && rendered.tokens.len() <= MAX_CONTEXT as usize,
                "{}: {} prompt tokens at width {width}",
                ticket.id,
                rendered.tokens.len()
            );
            prompt_tokens = rendered.tokens.len();
            let token_ids: Vec<i32> = rendered
                .tokens
                .iter()
                .map(|&id| i32::try_from(id).expect("token id fits i32"))
                .collect();
            let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
            logits.fill(0.0);
            if let Err(e) = prefill_program(&model, &pool, &mut sequence, &token_ids, 0, Some(&mut logits)) {
                if gpu_profile::skip_or_fail(&format!("{}: prefill at width {width}: {e}", ticket.id)) {
                    continue;
                }
                unreachable!("skip_or_fail panics under the profile");
            }
            drop(sequence);
            let option_logits: Vec<f32> = slots.iter().map(|&s| logits[s as usize]).collect();
            let restricted_total = logsumexp(option_logits.iter().copied());
            let mass = (restricted_total - logsumexp(logits.iter().copied())).exp();
            let predicted = argmax(&option_logits);
            let top = (f64::from(option_logits[predicted]) - restricted_total).exp();
            let winner = argmax(&logits) as u32;
            masses.push(mass);
            tops.push(top);
            in_slots += usize::from(slots.contains(&winner));
            hits += usize::from(predicted == ticket.label);
            scored += 1;
        }
        assert!(scored > 0, "no ticket's answer fits inside width {width}");
        masses.sort_by(f64::total_cmp);
        tops.sort_by(f64::total_cmp);
        eprintln!(
            "{width:>6} {scored:>7} {prompt_tokens:>7} {:>9.4} {:>9.4} {:>9.3} {:>7.0}% {:>7.0}%",
            masses[masses.len() / 2],
            masses[0],
            tops[tops.len() / 2],
            100.0 * in_slots as f64 / scored as f64,
            100.0 * hits as f64 / scored as f64,
        );
    }
}
