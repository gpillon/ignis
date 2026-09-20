//! `POST /v1/decide` — the decision endpoint (GitHub #239, ADR 0034).
//!
//! One `state` and a map of typed `questions`, answered from the logits of
//! named **answer tokens** at one position. Nothing is generated: a decision
//! costs one prefill and `usage.output_tokens` is 0, honestly.
//!
//! The wire shape is TypeSafe's Jev (`POST /v1/systemone`) copied rather
//! than invented, so an unmodified Jev client reaches this by changing the
//! URL. Their vocabulary — `noul`, `criteria`, `instructions` — is therefore
//! imported, not ours, and is spelled their way. Where we would have said
//! something different the name is accepted as an alias and nothing more.
//!
//! What is *not* copied is the confidence arithmetic: Jev documents theirs
//! as "derived from the answer's probability distribution" without saying
//! how, so ours is defined here and stated in full ([`confidence_of`] and
//! [`score_confidence`]).
//!
//! The prompt is the one the measurements were taken on
//! (`docs/findings/2026-09-19-typed-option-logit-readout.md`): SemIf's
//! `DIRECT_SYSTEM` verbatim, plus one user message carrying the decision as
//! JSON with the evidence first. Evidence first is not cosmetic — it is what
//! makes one `state` across many questions a shared token *prefix*.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use axum::response::IntoResponse;
use serde_json::{Value as JsonValue, json};

use ignis_core::decision::{AnswerAlphabet, AnswerToken, Readout};

use crate::template::{ChatMessage, ContentPart, MessageContent};

/// SemIf's `DIRECT_SYSTEM`, verbatim (`src/semif_phase1/core.py`).
///
/// Verbatim because every number the endpoint rests on was measured with
/// exactly this text: 0.934 balanced accuracy over 144 authored decisions,
/// the declared options holding a median 99.8% of the distribution, and the
/// unrestricted winner inside the declared set at 100% of rows out to 256
/// options. A reworded instruction is an unmeasured one.
///
/// It says "uppercase letter" although the alphabet reaches past `Z` into
/// bigrams. That is not an oversight left in: `classify_option_ceiling_gpu.rs`
/// served 256 options — labels like `AA` and `BJ` — under this very text and
/// found the model's own winner among the declared labels on every row at
/// every width. The instruction that was measured is the instruction that
/// ships.
pub const DIRECT_SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";

/// Options one question may declare.
///
/// **Measured this far, not a property of the model.** 256 is where
/// `classify_option_ceiling_gpu.rs` stopped walking, and it found no decay:
/// answer mass p50 ≥ 0.996 and the unrestricted winner in the declared set
/// at 100% of rows at every width from 8 to 256. What grows with width is
/// the prompt, not the noise. The ceiling is here because an endpoint should
/// not serve a shape nobody has measured, and it moves when somebody
/// measures further.
pub const MAX_OPTIONS: usize = 256;

/// Levels a `score` question must declare at least — Jev's own rule, and
/// arithmetic besides: an expected value over one level is that level.
pub const MIN_SCORE_LEVELS: usize = 2;

// ---------------------------------------------------------------------------
// An order-preserving map
// ---------------------------------------------------------------------------

/// A JSON object read **in order of appearance**.
///
/// `serde_json::Map` is a `BTreeMap` in this build, so a plain deserialize
/// would sort the keys — and for `criteria` that is not a presentation
/// detail but a different prompt. The options are written into the prompt in
/// the order the caller declared them, each against the answer token at that
/// position, so re-sorting them silently re-labels every option and asks the
/// model a question nobody wrote.
///
/// Enabling `serde_json/preserve_order` would do the same job and also
/// reorder the keys of every other JSON this server emits, which is a larger
/// promise than this needs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Ordered<T>(pub Vec<(String, T)>);

impl<T> Ordered<T> {
    /// The entries, in the order they were written.
    pub fn entries(&self) -> &[(String, T)] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The first key that appears more than once, if any. A duplicate key is
    /// a request whose meaning depends on which copy a parser kept.
    pub fn duplicate(&self) -> Option<&str> {
        let mut seen = std::collections::BTreeSet::new();
        self.0
            .iter()
            .find(|(key, _)| !seen.insert(key.as_str()))
            .map(|(key, _)| key.as_str())
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Ordered<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct InOrder<T>(std::marker::PhantomData<T>);

        impl<'de, T: Deserialize<'de>> Visitor<'de> for InOrder<T> {
            type Value = Ordered<T>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a JSON object")
            }

            fn visit_map<M: MapAccess<'de>>(self, mut access: M) -> Result<Ordered<T>, M::Error> {
                let mut entries = Vec::with_capacity(access.size_hint().unwrap_or(0));
                while let Some((key, value)) = access.next_entry::<String, T>()? {
                    entries.push((key, value));
                }
                Ok(Ordered(entries))
            }
        }

        deserializer.deserialize_map(InOrder(std::marker::PhantomData))
    }
}

// ---------------------------------------------------------------------------
// The request
// ---------------------------------------------------------------------------

/// `POST /v1/decide` — Jev's `POST /v1/systemone` body.
#[derive(Debug, Deserialize)]
pub struct DecideRequest {
    /// The content to evaluate: a string, a JSON object or array, **or**
    /// OpenAI content parts, so the evidence may be an image. The content
    /// parts are ours; Jev's `state` is `string | object | array`, which
    /// makes this a superset rather than a clone.
    pub state: JsonValue,
    /// The model to route to; the loaded model when absent or blank.
    #[serde(default)]
    pub model: Option<String>,
    /// The typed questions, keyed by ids the caller chooses. Answers come
    /// back under the same ids.
    pub questions: Ordered<Question>,
    /// Refused when true (GitHub #239): a decision's prompt ends where the
    /// answer is read, and a thinking prompt puts a reasoning block there
    /// instead. Accepted only so the refusal can name it.
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    /// The `thinking` object some clients carry `enable_thinking` inside.
    #[serde(default)]
    pub thinking: Option<JsonValue>,
}

/// One typed question.
///
/// `instructions` and `criteria` are Jev's field names. `question` and
/// `options` are what we would have called them, and are accepted as
/// aliases for exactly that reason — they are not a second shape.
#[derive(Debug, Deserialize)]
pub struct Question {
    /// `noul`, `choice` or `score`.
    #[serde(rename = "type")]
    pub kind: QuestionKind,
    /// What the model should decide. A string, object or array — anything
    /// but a string is serialized into the prompt as JSON.
    #[serde(alias = "question")]
    pub instructions: JsonValue,
    /// The type's own options: absent for a bare `noul`, a map for a
    /// `choice`, an ordered array for a `score`.
    #[serde(default, alias = "options")]
    pub criteria: Option<Criteria>,
}

/// A question's `criteria`, read in the order it was written.
///
/// It cannot be a `JsonValue`: `serde_json::Value::Object` is a sorted map
/// in this build, so by the time a `choice`'s options reached validation
/// their declared order would already be gone — and a different option order
/// is a different prompt. Parsing straight into an [`Ordered`] is what keeps
/// the order from ever being lost rather than trying to recover it.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Criteria {
    /// A `score`'s ordered levels.
    Levels(Vec<JsonValue>),
    /// A `noul`'s two descriptions, or a `choice`'s options in declared
    /// order.
    Map(Ordered<JsonValue>),
}

/// The three one-position primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionKind {
    /// A yes/no question, answered with the probability of yes.
    #[serde(alias = "boolean")]
    Noul,
    /// One option from a declared set.
    Choice,
    /// A probability-weighted value across ordered levels.
    Score,
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// A refused request: the whole request, before any prefill is spent.
///
/// All-or-nothing, and *early*, because the alternative is charging a caller
/// a GPU prefill for nineteen good questions and then failing on the
/// twentieth. Jev answers 422 for a body that fails validation, so this
/// does too.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    /// A stable, machine-readable code.
    pub code: &'static str,
    /// What is wrong, naming the offending field.
    pub message: String,
}

impl Refusal {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self { code, message: message.into() }
    }
}

/// One question, validated and bound to the answer tokens that will carry
/// it: everything the prompt builder and the answer shaper need, and
/// nothing a caller could still get wrong.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedQuestion {
    /// The id the caller chose; the answer comes back under it.
    pub id: String,
    pub kind: QuestionKind,
    /// The instructions, as the prompt will carry them.
    pub instructions: JsonValue,
    /// The options in declared order: what the caller called each one, and
    /// the description the prompt gives it.
    pub options: Vec<PreparedOption>,
    /// The answer token per option, parallel to `options`.
    pub answers: Vec<AnswerToken>,
}

/// One option of a prepared question.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedOption {
    /// How the answer names it: `"true"`/`"false"` for a `noul`, the
    /// caller's key for a `choice`, the level index for a `score`.
    pub name: String,
    /// What the prompt says it means.
    pub description: String,
}

/// The descriptions a bare `noul` uses when the caller supplied none.
const NOUL_DEFAULT: [(&str, &str); 2] = [("true", "Yes"), ("false", "No")];

/// Validate and bind every question, or refuse the whole request.
///
/// `alphabet` is the loaded model's (GitHub #237): the labels it can name
/// are a property of its tokenizer, so a request asking for more options
/// than this model can label is refused here rather than served with two
/// options sharing a logit.
pub fn prepare(
    questions: &Ordered<Question>,
    alphabet: &AnswerAlphabet,
) -> Result<Vec<PreparedQuestion>, Refusal> {
    if questions.is_empty() {
        return Err(Refusal::new("no_questions", "`questions` must carry at least one question"));
    }
    if let Some(duplicate) = questions.duplicate() {
        return Err(Refusal::new(
            "duplicate_question",
            format!("question id {duplicate:?} appears more than once"),
        ));
    }
    questions
        .entries()
        .iter()
        .map(|(id, question)| prepare_one(id, question, alphabet))
        .collect()
}

fn prepare_one(
    id: &str,
    question: &Question,
    alphabet: &AnswerAlphabet,
) -> Result<PreparedQuestion, Refusal> {
    if instructions_are_empty(&question.instructions) {
        return Err(Refusal::new(
            "empty_instructions",
            format!("question {id:?} has empty `instructions`"),
        ));
    }
    let options = match question.kind {
        QuestionKind::Noul => noul_options(id, question.criteria.as_ref())?,
        QuestionKind::Choice => choice_options(id, question.criteria.as_ref())?,
        QuestionKind::Score => score_options(id, question.criteria.as_ref())?,
    };
    if options.len() > MAX_OPTIONS {
        return Err(Refusal::new(
            "too_many_options",
            format!(
                "question {id:?} declares {} options; {MAX_OPTIONS} is the measured ceiling",
                options.len()
            ),
        ));
    }
    let answers = alphabet.take(options.len()).ok_or_else(|| {
        Refusal::new(
            "alphabet_exhausted",
            format!(
                "question {id:?} declares {} options, and this model's tokenizer can name only {}",
                options.len(),
                alphabet.len()
            ),
        )
    })?;
    Ok(PreparedQuestion {
        id: id.to_owned(),
        kind: question.kind,
        instructions: question.instructions.clone(),
        options,
        answers: answers.to_vec(),
    })
}

/// Whether `instructions` says nothing at all — an empty string, an empty
/// object or array, or `null`. A question with no instructions has no
/// criterion to apply, and the model would answer the options alone.
fn instructions_are_empty(instructions: &JsonValue) -> bool {
    match instructions {
        JsonValue::Null => true,
        JsonValue::String(text) => text.trim().is_empty(),
        JsonValue::Array(items) => items.is_empty(),
        JsonValue::Object(fields) => fields.is_empty(),
        _ => false,
    }
}

fn noul_options(id: &str, criteria: Option<&Criteria>) -> Result<Vec<PreparedOption>, Refusal> {
    let fields = match criteria {
        None => None,
        Some(Criteria::Map(map)) => Some(map),
        Some(Criteria::Levels(_)) => {
            return Err(Refusal::new(
                "malformed_criteria",
                format!("question {id:?} is a noul, so `criteria` must be an object of `true` and `false`"),
            ));
        }
    };
    // Order is fixed here, not taken from the caller: a noul's answer is
    // "the probability of the true option", so which slot is which is part
    // of the primitive rather than part of the request.
    NOUL_DEFAULT
        .iter()
        .map(|(name, fallback)| {
            let stated = fields.and_then(|map| {
                map.entries().iter().find(|(key, _)| key == name).map(|(_, value)| value)
            });
            let description = match stated {
                None | Some(JsonValue::Null) => (*fallback).to_owned(),
                Some(JsonValue::String(text)) if !text.trim().is_empty() => text.clone(),
                Some(_) => {
                    return Err(Refusal::new(
                        "malformed_criteria",
                        format!("question {id:?}: `criteria.{name}` must be a non-empty string"),
                    ));
                }
            };
            Ok(PreparedOption { name: (*name).to_owned(), description })
        })
        .collect()
}

fn choice_options(id: &str, criteria: Option<&Criteria>) -> Result<Vec<PreparedOption>, Refusal> {
    let Some(criteria) = criteria else {
        return Err(Refusal::new(
            "missing_criteria",
            format!("question {id:?} is a choice, so `criteria` is required"),
        ));
    };
    let Criteria::Map(ordered) = criteria else {
        return Err(Refusal::new(
            "malformed_criteria",
            format!("question {id:?} is a choice, so `criteria` must be an object of option to description"),
        ));
    };
    if ordered.is_empty() {
        return Err(Refusal::new(
            "no_options",
            format!("question {id:?} declares no options"),
        ));
    }
    if let Some(duplicate) = ordered.duplicate() {
        return Err(Refusal::new(
            "duplicate_option",
            format!("question {id:?}: option {duplicate:?} appears more than once"),
        ));
    }
    ordered
        .entries()
        .iter()
        .map(|(name, description)| {
            // `null` means the option needs no extra detail, so the id is
            // its own description — Jev's own rule.
            let description = match description {
                JsonValue::Null => name.clone(),
                JsonValue::String(text) if !text.trim().is_empty() => text.clone(),
                _ => {
                    return Err(Refusal::new(
                        "malformed_criteria",
                        format!("question {id:?}: option {name:?} must describe itself with a non-empty string or null"),
                    ));
                }
            };
            Ok(PreparedOption { name: name.clone(), description })
        })
        .collect()
}

fn score_options(id: &str, criteria: Option<&Criteria>) -> Result<Vec<PreparedOption>, Refusal> {
    let Some(Criteria::Levels(levels)) = criteria else {
        return Err(Refusal::new(
            "malformed_criteria",
            format!("question {id:?} is a score, so `criteria` must be an ordered array of levels"),
        ));
    };
    if levels.len() < MIN_SCORE_LEVELS {
        return Err(Refusal::new(
            "too_few_levels",
            format!(
                "question {id:?} declares {} score levels; at least {MIN_SCORE_LEVELS} are needed",
                levels.len()
            ),
        ));
    }
    levels
        .iter()
        .enumerate()
        .map(|(index, level)| match level {
            JsonValue::String(text) if !text.trim().is_empty() => Ok(PreparedOption {
                // The level's *index* is its name: a score answers with a
                // number across the levels, and `legend` maps each index
                // back to what the caller wrote.
                name: index.to_string(),
                description: text.clone(),
            }),
            _ => Err(Refusal::new(
                "malformed_criteria",
                format!("question {id:?}: score level {index} must be a non-empty string"),
            )),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The prompt
// ---------------------------------------------------------------------------

/// The messages one prepared question is asked as.
///
/// SemIf's `direct_messages`: the system instruction, plus one user message
/// carrying `{evidence, criterion, options: [{letter, description}]}` as
/// JSON with the evidence first.
///
/// When `state` is content parts the evidence is the parts themselves — the
/// image first, then the decision's JSON without an `evidence` field, which
/// is the shape `classify_vision_readout_gpu.rs` measured. Either way the
/// evidence leads, so one `state` across many questions is one shared token
/// prefix.
pub fn messages_for(state: &Evidence, question: &PreparedQuestion) -> Vec<ChatMessage> {
    let options: Vec<JsonValue> = question
        .options
        .iter()
        .zip(&question.answers)
        .map(|(option, answer)| {
            json!({ "letter": answer.label, "description": option.description })
        })
        .collect();
    let system = ChatMessage::text("system", DIRECT_SYSTEM);
    match state {
        Evidence::Json(value) => {
            let payload = payload_text(&[
                ("evidence", value),
                ("criterion", &question.instructions),
                ("options", &JsonValue::Array(options)),
            ]);
            vec![system, ChatMessage::text("user", payload)]
        }
        Evidence::Parts(parts) => {
            let payload = payload_text(&[
                ("criterion", &question.instructions),
                ("options", &JsonValue::Array(options)),
            ]);
            let mut content = parts.clone();
            content.push(ContentPart {
                kind: Some("text".to_owned()),
                text: Some(payload),
                url: None,
            });
            vec![
                system,
                ChatMessage {
                    role: "user".to_owned(),
                    content: MessageContent::Parts(content),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
            ]
        }
    }
}

/// Serialize `fields` as a JSON object **in the order given**.
///
/// Built by hand rather than with `json!` because `serde_json::Map` sorts
/// its keys in this build, and the order of these three is load-bearing.
/// `{"criterion": ...}` sorts before `{"evidence": ...}`, so a `json!`
/// payload puts the *question* in front of the evidence — and then two
/// questions over one `state` share nothing but the nine characters of
/// `{"criterion`, which is the opposite of what one shared evidence is
/// supposed to buy (spec 04's fan-out, and the reuse the finding argues
/// for).
///
/// See `docs/findings/2026-09-20-evidence-first-needs-explicit-key-order.md`:
/// the measured prompts had this same sorting, so "evidence first" was a
/// description of the intent and not of the bytes.
fn payload_text(fields: &[(&str, &JsonValue)]) -> String {
    let mut out = String::from("{");
    for (index, (name, value)) in fields.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(name);
        out.push_str("\":");
        out.push_str(&serde_json::to_string(value).unwrap_or_else(|_| "null".to_owned()));
    }
    out.push('}');
    out
}

/// The evidence a decision is put to.
#[derive(Debug, Clone, PartialEq)]
pub enum Evidence {
    /// A string, object or array — Jev's own `state`, carried into the
    /// prompt's JSON payload.
    Json(JsonValue),
    /// OpenAI content parts, so the evidence may be an image. Ours, not
    /// Jev's.
    Parts(Vec<ContentPart>),
}

impl Evidence {
    /// Read a `state` field.
    ///
    /// Content parts and a JSON array are the same JSON shape, so the rule
    /// has to be stated rather than guessed: an array is content parts only
    /// when **every** element is an object carrying a `type` string, which
    /// is what a content part always has and what evidence like
    /// `[{"id": 1, "title": "…"}]` does not. Anything else is evidence.
    pub fn read(state: &JsonValue) -> Self {
        let JsonValue::Array(items) = state else {
            return Self::Json(state.clone());
        };
        let parts_shaped = !items.is_empty()
            && items
                .iter()
                .all(|item| item.get("type").and_then(JsonValue::as_str).is_some());
        if !parts_shaped {
            return Self::Json(state.clone());
        }
        match serde_json::from_value::<Vec<ContentPart>>(state.clone()) {
            Ok(parts) => Self::Parts(parts),
            Err(_) => Self::Json(state.clone()),
        }
    }

    /// Whether this evidence carries a media part — what decides whether the
    /// request needs the media path at all.
    pub fn has_media(&self) -> bool {
        match self {
            Self::Json(_) => false,
            Self::Parts(parts) => parts
                .iter()
                .any(|part| part.kind.as_deref().is_some_and(|kind| kind != "text")),
        }
    }
}

// ---------------------------------------------------------------------------
// The answer
// ---------------------------------------------------------------------------

/// The response body.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct DecideResponse {
    /// The model that performed the evaluation.
    pub model: String,
    /// One answer per question, under the ids the caller chose.
    pub answers: BTreeMap<String, Answer>,
    /// Token usage. `output_tokens` is 0 — a decision generates nothing,
    /// and saying otherwise would be the one lie this endpoint could tell
    /// that nothing downstream would catch.
    pub usage: Usage,
}

/// The prompt's cost, and the absence of any other.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// One question's answer, or the error that stands in its place.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    /// A yes/no answer: the probability of the true option, 0 (no) to 1
    /// (yes). Jev's `noul` answer carries no confidence, and neither does
    /// this — the number *is* the confidence.
    Noul { noul: f64 },
    /// The chosen option, the distribution over all of them, and how certain
    /// the distribution is.
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    /// The probability-weighted value across the levels — which can land
    /// between them — with each level mapped back to what the caller wrote.
    Score {
        score: f64,
        legend: BTreeMap<String, String>,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    /// This question alone failed after the GPU was already spent on its
    /// siblings (spec 04): validation is all-or-nothing and happens earlier,
    /// so nothing here is a caller's mistake.
    Error { code: String, message: String },
}

/// Shape `readout` into `question`'s answer.
pub fn answer_for(question: &PreparedQuestion, readout: &Readout) -> Answer {
    let probabilities = readout.probabilities();
    match question.kind {
        QuestionKind::Noul => Answer::Noul {
            // The true option is slot 0 by construction (`noul_options`).
            noul: probabilities.first().copied().unwrap_or(0.0),
        },
        QuestionKind::Choice => {
            let winner = readout.winner().unwrap_or(0);
            Answer::Choice {
                choice: question.options[winner].name.clone(),
                probabilities: named(question, &probabilities),
                confidence: confidence_of(&probabilities),
            }
        }
        QuestionKind::Score => Answer::Score {
            score: expected_level(&probabilities),
            legend: question
                .options
                .iter()
                .map(|option| (option.name.clone(), option.description.clone()))
                .collect(),
            probabilities: named(question, &probabilities),
            confidence: score_confidence(&probabilities),
        },
    }
}

fn named(question: &PreparedQuestion, probabilities: &[f64]) -> BTreeMap<String, f64> {
    question
        .options
        .iter()
        .zip(probabilities)
        .map(|(option, &p)| (option.name.clone(), p))
        .collect()
}

/// The expected value of the distribution over level *indices* — Jev's own
/// `1.6` for `{0: 0.05, 1: 0.3, 2: 0.65}`.
///
/// A score is not a class. The model's uncertainty between two adjacent
/// levels is information about where the answer sits, not noise to be
/// argmaxed away: a state the model splits evenly between "Frustrated" and
/// "Very angry" is a 1.5, and reporting either level alone would throw that
/// away.
pub fn expected_level(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .enumerate()
        .map(|(index, p)| index as f64 * p)
        .sum()
}

/// A categorical answer's confidence: the **top probability**.
///
/// Documented as exactly that, because it is the quantity that was measured
/// to separate right answers from wrong ones: every one of the 118 rows
/// above 0.9 was correct, and the nine errors clustered at a median 0.634
/// (`docs/findings/2026-09-19-typed-option-logit-readout.md`). It is an
/// abstention signal with a known threshold, not a feeling.
pub fn confidence_of(probabilities: &[f64]) -> f64 {
    probabilities.iter().copied().fold(0.0, f64::max).clamp(0.0, 1.0)
}

/// A score's confidence: `1 - (standard deviation / half the level range)`.
///
/// The top probability is the wrong measure here and would be actively
/// misleading. A distribution split evenly between two *adjacent* levels
/// knows exactly where the answer is — halfway between them — and its top
/// probability is 0.5. One split evenly between the lowest and the highest
/// knows nothing, and its top probability is also 0.5. Spread, not height,
/// is what a score is uncertain about.
///
/// Normalized by half the range so a two-level score behaves: the most
/// spread a `n`-level distribution can have is `(n-1)/2`, reached by putting
/// half the mass at each end, which scores 0.
pub fn score_confidence(probabilities: &[f64]) -> f64 {
    if probabilities.len() < 2 {
        return 1.0;
    }
    let mean = expected_level(probabilities);
    let variance: f64 = probabilities
        .iter()
        .enumerate()
        .map(|(index, p)| p * (index as f64 - mean).powi(2))
        .sum();
    let half_range = (probabilities.len() - 1) as f64 / 2.0;
    (1.0 - variance.sqrt() / half_range).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// The handler
// ---------------------------------------------------------------------------

/// `POST /v1/decide`, and `POST /v1/systemone` under its Jev name.
pub async fn decide(
    axum::extract::State(server): axum::extract::State<std::sync::Arc<crate::Server>>,
    body: Result<axum::Json<DecideRequest>, axum::extract::rejection::JsonRejection>,
) -> axum::response::Response {
    // A body that will not parse is a malformed request, which is the same
    // 422 a body that parses into nonsense gets. Jev answers 422 for "a
    // missing required field or a malformed question" and does not
    // distinguish the two, so neither does this.
    let request = match body {
        Ok(axum::Json(request)) => request,
        Err(rejection) => {
            return refused(&Refusal::new("malformed_request", rejection.body_text()));
        }
    };
    // Thinking is refused, never ignored. A decision's prompt ends exactly
    // where its answer is read; a thinking prompt puts an open reasoning
    // block there instead, so the position whose logits this reads would
    // hold the first token of a reasoning trace. Serving that silently
    // would return a well-formed answer computed from the wrong position.
    if asks_for_thinking(&request) {
        return refused(&Refusal::new(
            "thinking_unsupported",
            "a decision reads one position and generates nothing, so `enable_thinking` cannot be honoured; omit it or set it to false",
        ));
    }
    // Everything validates before the GPU is touched: a caller must never
    // pay a prefill for nineteen good questions and a refusal on the
    // twentieth.
    let prepared = match prepare(&request.questions, &server.alphabet) {
        Ok(prepared) => prepared,
        Err(refusal) => return refused(&refusal),
    };
    let evidence = Evidence::read(&request.state);
    let class = ignis_core::types::RequestClass::for_decision(None);

    let mut answers = BTreeMap::new();
    let mut input_tokens = 0u32;
    for question in &prepared {
        match ask(&server, &evidence, question, class).await {
            Ok((answer, tokens)) => {
                input_tokens = input_tokens.saturating_add(tokens);
                answers.insert(question.id.clone(), answer);
            }
            Err(error) => {
                answers.insert(question.id.clone(), error);
            }
        }
    }
    let model = request
        .model
        .filter(|model| !model.is_empty())
        .unwrap_or_else(|| server.engine.model_id());
    axum::Json(DecideResponse {
        model,
        answers,
        usage: Usage {
            input_tokens,
            // Zero, honestly. A decision reads one position's logits and
            // samples nothing, so `ignis_decoded_tokens_total` does not move
            // for it either — which is correct, and makes decisions
            // invisible to the existing throughput panels (ADR 0034).
            output_tokens: 0,
        },
    })
    .into_response()
}

/// Whether the request asked for thinking, on either of the two fields a
/// client may carry it on.
fn asks_for_thinking(request: &DecideRequest) -> bool {
    request.enable_thinking == Some(true)
        || request
            .thinking
            .as_ref()
            .and_then(|thinking| thinking.get("enable_thinking"))
            .and_then(JsonValue::as_bool)
            == Some(true)
        || request
            .thinking
            .as_ref()
            .and_then(JsonValue::as_bool)
            == Some(true)
}

/// Put one question to the model and shape its answer.
///
/// Returns the answer and the prompt tokens it cost, or — when the engine
/// could not answer *this* question — an [`Answer::Error`] to stand in its
/// slot. A failure here is per-question by design (spec 04): the other
/// questions' answers are already paid for, and throwing them away to
/// report one failure helps nobody.
async fn ask(
    server: &crate::Server,
    evidence: &Evidence,
    question: &PreparedQuestion,
    class: ignis_core::types::RequestClass,
) -> Result<(Answer, u32), Answer> {
    let messages = messages_for(evidence, question);
    let thinking = crate::thinking::ThinkingOptions {
        enable_thinking: false,
        ..crate::thinking::ThinkingOptions::default()
    };
    let params = ignis_core::types::DecodeParams::default();
    let (mut input, _model, prompt_tokens, media) =
        crate::api::prepare_decision_request(server, &messages, params, &thinking)
            .await
            .map_err(|error| failed("render_failed", error))?;
    input.decision = Some(std::sync::Arc::from(
        question.answers.iter().map(|answer| answer.id).collect::<Vec<_>>(),
    ));

    let (id, mut events) = server
        .engine
        .submit_with_media(input, class, media)
        .await
        .map_err(|error| failed("submit_failed", format!("{error:?}")))?;
    // The engine keeps working on a request whose caller has gone until it
    // is told otherwise, and a fan-out is twenty of them.
    let mut guard = crate::api::CancelOnDrop::new(server.engine.clone(), id);
    let readout = crate::engine::collect_readout(&mut events, server.request_timeout)
        .await
        .map_err(|_| {
            failed(
                "not_completed",
                "the engine did not answer this question in time".to_owned(),
            )
        })?;
    guard.completed();
    Ok((answer_for(question, &readout), prompt_tokens))
}

fn failed(code: &str, message: String) -> Answer {
    Answer::Error { code: code.to_owned(), message }
}

/// A refusal, as the 422 Jev documents for a body that fails validation.
fn refused(refusal: &Refusal) -> axum::response::Response {
    (
        axum::http::StatusCode::UNPROCESSABLE_ENTITY,
        axum::Json(json!({
            "error": {
                "type": "invalid_request_error",
                "code": refusal.code,
                "message": refusal.message,
            }
        })),
    )
        .into_response()
}

