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
//! **What this does not carry: the answer mass.** A decision's one silent
//! failure is the declared options holding none of the distribution — the
//! answers are then well-formed noise with a plausible argmax, and neither
//! `confidence` nor `probabilities` can show it, because both are computed
//! *after* the restriction throws the rest away (`CONTEXT.md`, **answer
//! mass**; ADR 0034 calls it "the only failure of this endpoint that nothing
//! else would show"). It is deliberately not a response field: Jev's answer
//! shape has none, and an aggregate is the useful form of it. GitHub #241
//! owns exposing it, as a histogram on the metrics listener beside a
//! decision counter, and `Readout::answer_mass` is what it reads.
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

    /// How many entries were written, duplicates counted separately —
    /// [`Ordered::duplicate`] is what decides whether that matters.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the object was written empty, which for `questions` means a
    /// request with nothing to decide.
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
    /// The thinking controls, in every shape this server documents them
    /// (GitHub #68): the top-level field, the effort level that implies
    /// one, and the `chat_template_kwargs` object the template reads.
    ///
    /// Accepted only so that asking for thinking can be *refused*. A
    /// decision's prompt ends exactly where its answer is read, and a
    /// thinking prompt puts an open reasoning block at that position, so a
    /// readout would report the first token of a reasoning trace as the
    /// answer. Dropping these fields silently would serve that.
    #[serde(default)]
    pub enable_thinking: Option<JsonValue>,
    #[serde(default)]
    pub reasoning_effort: Option<JsonValue>,
    #[serde(default)]
    pub preserve_thinking: Option<JsonValue>,
    #[serde(default)]
    pub chat_template_kwargs: Option<JsonValue>,
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
#[derive(Debug)]
pub enum Criteria {
    /// A `score`'s ordered levels.
    Levels(Vec<JsonValue>),
    /// A `noul`'s two descriptions, or a `choice`'s options in declared
    /// order.
    Map(Ordered<JsonValue>),
    /// Neither — a string, a number, a bool. Kept rather than rejected at
    /// the parse, so the refusal can come from validation and name the
    /// question it belongs to.
    Other,
}

impl<'de> Deserialize<'de> for Criteria {
    /// Written out rather than `#[serde(untagged)]` for the sake of the
    /// *failure*. Untagged reports "data did not match any variant", at the
    /// `Json` extractor, before the question it belongs to has a name — so
    /// `"criteria": "high"` would refuse the whole request without saying
    /// which question or which field, while every other criteria fault
    /// names both.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct EitherShape;

        impl<'de> Visitor<'de> for EitherShape {
            type Value = Criteria;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("`criteria` as an object of options or an array of score levels")
            }

            fn visit_map<M: MapAccess<'de>>(self, mut access: M) -> Result<Criteria, M::Error> {
                let mut entries = Vec::with_capacity(access.size_hint().unwrap_or(0));
                while let Some(entry) = access.next_entry::<String, JsonValue>()? {
                    entries.push(entry);
                }
                Ok(Criteria::Map(Ordered(entries)))
            }

            fn visit_seq<S: serde::de::SeqAccess<'de>>(
                self,
                mut access: S,
            ) -> Result<Criteria, S::Error> {
                let mut levels = Vec::with_capacity(access.size_hint().unwrap_or(0));
                while let Some(level) = access.next_element::<JsonValue>()? {
                    levels.push(level);
                }
                Ok(Criteria::Levels(levels))
            }

            fn visit_unit<E: serde::de::Error>(self) -> Result<Criteria, E> {
                // `"criteria": null` is a noul that supplied none.
                Ok(Criteria::Map(Ordered(Vec::new())))
            }

            fn visit_none<E: serde::de::Error>(self) -> Result<Criteria, E> {
                self.visit_unit()
            }

            // Everything else parses into `Other` and is refused by name a
            // moment later. Refusing here instead would produce serde's own
            // message at the `Json` extractor, which knows neither which
            // question nor which field it was reading.
            fn visit_str<E: serde::de::Error>(self, _value: &str) -> Result<Criteria, E> {
                Ok(Criteria::Other)
            }

            fn visit_bool<E: serde::de::Error>(self, _value: bool) -> Result<Criteria, E> {
                Ok(Criteria::Other)
            }

            fn visit_i64<E: serde::de::Error>(self, _value: i64) -> Result<Criteria, E> {
                Ok(Criteria::Other)
            }

            fn visit_u64<E: serde::de::Error>(self, _value: u64) -> Result<Criteria, E> {
                Ok(Criteria::Other)
            }

            fn visit_f64<E: serde::de::Error>(self, _value: f64) -> Result<Criteria, E> {
                Ok(Criteria::Other)
            }
        }

        deserializer.deserialize_any(EitherShape)
    }
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
        Some(Criteria::Levels(_) | Criteria::Other) => {
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
            // An option's key is what the answer comes back under, and —
            // when it describes itself — what the prompt says it means. A
            // blank one is neither, and would reach the model as an
            // option with no text at all.
            if name.trim().is_empty() {
                return Err(Refusal::new(
                    "unclean_option",
                    format!("question {id:?} declares an option with a blank name"),
                ));
            }
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
/// SemIf's `direct_messages` with one departure, which GitHub #240 forced:
/// the evidence rides in the **system message**, not at the head of the
/// user payload. The system message is `DIRECT_SYSTEM` followed by
/// `{"evidence": …}`; the user message carries
/// `{criterion, options: [{letter, description}]}`.
///
/// **Why the system block and not just "first".** A fan-out of N questions
/// over one `state` is only cheap if the state is prefilled once, and the
/// tier that can do that is decided by *where* the shared text sits, not by
/// its being shared:
///
/// - A **live shared prefix** needs a publisher that is still running when
///   the claimant arrives. A decision terminates in the same tick its
///   prefill does (GitHub #238), so it is never a live publisher for a
///   sibling — the very property that makes it cheap denies it this tier.
/// - A **prompt checkpoint** is refused outright for a decision
///   (`ConcreteScheduler`'s capture point, GitHub #238).
/// - A **retained prefix** outlives its publisher, which is exactly what a
///   sequenced first question needs. It is cut at
///   `Request::retained_prefix_point` — the page floor of
///   `system_block_tokens` — and nothing outside the system block is ever
///   part of it.
///
/// So evidence in the user turn can be first, shared, and still re-prefilled
/// N times. Spec 04 asked for "evidence first"; what it needed was "evidence
/// in the block".
///
/// **An image `state` does not follow it, and spec 04's acceptance 2 is
/// unmet.** The parts stay in the user turn — image first, then the
/// decision's JSON without an `evidence` field, the shape
/// `classify_vision_readout_gpu.rs` measured — so a fan-out over an image
/// shares nothing and re-encodes it per question.
///
/// Why, precisely, because an earlier version of this comment got it wrong:
/// `check_content_parts` does refuse media in a system message (GitHub
/// #175), but it is called only from the chat and responses routes and
/// **never on this path** (`api.rs::prepare_request` does not call it), so
/// it is not what stops this. What stops it is that putting an image in a
/// system message would be a policy this server enforces everywhere else,
/// reversed here, on a render nobody has checked the real chat template
/// does at all — and that even inside the block an image is normally
/// excluded anyway, since `prefix_floor` walks the page floor back out of
/// any media item it lands inside (GitHub #193). It would have to be the
/// image *first* and then a whole page of text, which is a third prompt
/// layout to measure. That is a design fork for #240's owner or GitHub
/// #235, not something to decide in a prompt builder.
pub fn messages_for(state: &Evidence, question: &PreparedQuestion) -> Vec<ChatMessage> {
    let options: Vec<JsonValue> = question
        .options
        .iter()
        .zip(&question.answers)
        .map(|(option, answer)| {
            json!({ "letter": answer.label, "description": option.description })
        })
        .collect();
    let ask = payload_text(&[
        ("criterion", &question.instructions),
        ("options", &JsonValue::Array(options)),
    ]);
    match state {
        Evidence::Json(value) => {
            // One blank line between the instruction and the evidence: the
            // instruction is the same bytes for every question over every
            // state, so a reader — and a retained prefix — meets it first.
            let system = format!("{DIRECT_SYSTEM}\n\n{}", payload_text(&[("evidence", value)]));
            vec![
                ChatMessage::text("system", system),
                ChatMessage::text("user", ask),
            ]
        }
        Evidence::Parts(parts) => {
            let mut content = parts.clone();
            content.push(ContentPart {
                kind: Some("text".to_owned()),
                text: Some(ask),
                url: None,
            });
            vec![
                ChatMessage::text("system", DIRECT_SYSTEM),
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
/// its keys in this build, so `json!` emits an order nobody wrote and
/// nobody can read off the call site.
///
/// The sorting used to be load-bearing here: with the evidence in the user
/// payload, `{"criterion"` sorted in front of `{"evidence"`, and two
/// questions over one `state` shared nine characters instead of the whole
/// state (`docs/findings/2026-09-20-evidence-first-needs-explicit-key-order.md`
/// — the measured prompts had that sorting, so "evidence first" described
/// the intent and not the bytes). GitHub #240 moved the evidence into the
/// system block, where the reuse actually lives, and the two keys left
/// happen to sort into the order they are written in.
///
/// Kept, and kept explicit, because "happens to sort right" is not a
/// property anyone should have to re-derive: the next field added here
/// would silently reorder a prompt that has been measured. The order in
/// the call site is the order the model sees.
fn payload_text(fields: &[(&str, &JsonValue)]) -> String {
    let mut out = String::from("{");
    for (index, (name, value)) in fields.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(name);
        out.push_str("\":");
        out.push_str(
            &serde_json::to_string(value).expect("a `JsonValue` always serializes"),
        );
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

    /// Whether this evidence carries a media part — what decides whether a
    /// load without `--vision` can evaluate it at all.
    ///
    /// The same rule `crate::media::has_media` applies to a conversation,
    /// asked of a `state` instead: an `image_url` part that actually has a
    /// url. Two predicates that disagreed about what media *is* would send
    /// a request down the media path that the media path then refused, or
    /// worse, the other way round.
    pub fn has_media(&self) -> bool {
        match self {
            Self::Json(_) => false,
            Self::Parts(parts) => parts
                .iter()
                .any(|part| part.kind.as_deref() == Some("image_url") && part.url.is_some()),
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
    /// This question alone failed at *runtime*, after the GPU was already
    /// spent on its siblings (spec 04).
    ///
    /// Everything a caller could have got wrong — a malformed question, an
    /// unnameable option, an image a text-only load cannot take, a prompt
    /// past the context — refuses the whole request before the first
    /// submit, so what lands here is the engine: a refused admission, or a
    /// question the engine did not answer within `--request-timeout`. The
    /// siblings' answers are already paid for, and discarding them to
    /// report one fault helps nobody.
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
    match serve(&server, request).await {
        Ok(response) => axum::Json(response).into_response(),
        Err(refusal) => refused(&refusal),
    }
}

/// Everything the endpoint refuses, and then everything it answers.
///
/// The split is the acceptance: **nothing reaches the engine until every
/// question has been validated and rendered**. A caller must never pay a
/// prefill for nineteen good questions and a refusal on the twentieth, and
/// "before the GPU" has to mean before the *first* submit, not before each
/// one.
async fn serve(
    server: &crate::Server,
    request: DecideRequest,
) -> Result<DecideResponse, Refusal> {
    refuse_thinking(server, &request)?;
    let prepared = prepare(&request.questions, &server.alphabet)?;
    let evidence = Evidence::read(&request.state);
    // An image `state` on a load that cannot take images is a refusal, not
    // an error in an answer slot: the request was never servable, and it is
    // the caller's to fix.
    if evidence.has_media() && server.media.is_none() {
        return Err(Refusal::new(
            "media_unsupported",
            "this server was loaded without `--vision`, so a `state` carrying an image cannot be evaluated",
        ));
    }
    // The **Lane tag**, exactly as every other route reads it — off the
    // `model` field's `@<lane>` suffix (`CONTEXT.md`). What is left is the
    // model the caller named, which goes to the scheduler rather than
    // being echoed back unexamined: a decision addressed to a model this
    // engine does not load is refused like any other request.
    let (model, lane) = crate::api::split_model_lane(request.model.clone());
    let class = ignis_core::types::RequestClass::for_decision(lane);
    // A model this engine does not load can never be served, so it is a
    // refusal of the whole request rather than N identical errors in N
    // answer slots — and it is checked here, before the first submit, for
    // the same reason everything else is.
    //
    // 422 where `/v1/chat/completions` answers 404 `model_not_found`: Jev's
    // error table has no 404, and its 422 is "the request body failed
    // validation … The body details the offending field", which is what a
    // wrong `model` is. A Jev client meets the status its own docs told it
    // to expect.
    let loaded = server.engine.model_id();
    if let Some(named) = model.as_deref().filter(|named| !named.is_empty() && *named != loaded) {
        return Err(Refusal::new(
            "model_not_found",
            format!("`model` names {named:?}, and this engine serves {loaded:?}"),
        ));
    }

    // Render every question first. This is the last thing that can refuse
    // the whole request, and it is all CPU: the chat template, the
    // instruction policy and — for an image — the media acquisition and
    // the placeholder expansion.
    let mut rendered = Vec::with_capacity(prepared.len());
    for question in &prepared {
        rendered.push(render(server, &evidence, question, model.clone()).await?);
    }

    let input_tokens = rendered
        .iter()
        .fold(0u32, |total, ready| total.saturating_add(ready.prompt_tokens));
    let resolved = rendered.first().map(|ready| ready.model.clone());

    // The fan-out (GitHub #240). **One question goes first, alone**, and the
    // rest go together once it is answered.
    //
    // The sequencing is not politeness, it is the whole saving. Handed N
    // questions at once the scheduler sees N requests with no published
    // prefix between them, and every one of them prefills the whole state —
    // for an image, N x 16K tokens. The first question prefills it once and
    // leaves a **retained prefix** behind (`messages_for` says why it is
    // retained and not shared); the followers claim it and prefill only
    // their own tail.
    //
    // After that there is nothing left to serialize, so the followers run
    // together, [`FAN_OUT_WIDTH`] of them at a time.
    let mut questions = prepared.iter().zip(rendered);
    let mut answers = BTreeMap::new();
    if let Some((question, ready)) = questions.next() {
        // A first question the engine has no room for leaves the fan-out
        // with no prefix to share, but it is still one question's failure
        // and not the request's — the same slot an engine-full follower
        // gets, and the same one the sequential loop before #240 gave it.
        let answer = match ask(server, question, ready, class).await {
            Attempt::Answered(answer) => answer,
            Attempt::Full(_) => engine_full(),
        };
        answers.insert(question.id.clone(), answer);
    }

    // The followers, in waves, with whatever the engine turned away put
    // back for the next one.
    //
    // A wave is `FAN_OUT_WIDTH` wide *because* the engine admits about that
    // many, but it is not the only client: two of its lanes may already be
    // somebody else's, and then two of this wave's questions come back
    // `Full`. Answering those with an error would be a failure the
    // one-at-a-time loop before #240 never produced, so they are re-queued
    // instead — a wave that answers even one question makes room for them.
    // A wave where *nothing* got in is an engine with no room at all, and
    // that is reported rather than spun on.
    let mut pending: std::collections::VecDeque<_> = questions.collect();
    while !pending.is_empty() {
        let wave: Vec<_> = (0..FAN_OUT_WIDTH.min(pending.len()))
            .filter_map(|_| pending.pop_front())
            .collect();
        let asked: Vec<&PreparedQuestion> = wave.iter().map(|(question, _)| *question).collect();
        let run = wave
            .into_iter()
            .map(|(question, ready)| ask(server, question, ready, class))
            .collect();
        let mut answered_one = false;
        for (question, attempt) in asked.into_iter().zip(concurrently(run).await) {
            match attempt {
                Attempt::Answered(answer) => {
                    answers.insert(question.id.clone(), answer);
                    answered_one = true;
                }
                Attempt::Full(ready) => pending.push_back((question, ready)),
            }
        }
        if !answered_one {
            for (question, _) in pending.drain(..) {
                answers.insert(question.id.clone(), engine_full());
            }
        }
    }
    Ok(DecideResponse {
        // The model that *performed* the evaluation, which is the one the
        // engine resolved — not the string the caller sent.
        model: resolved.unwrap_or_else(|| server.engine.model_id()),
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
}

/// Refuse a request that asked for thinking, in any of the shapes this
/// server reads it in.
///
/// Resolved rather than pattern-matched, so `reasoning_effort` — which
/// *implies* thinking (`thinking.rs`) — is caught by the same rule as the
/// field that says so outright. The defaults handed in are the decision's
/// own, not the server's: a request that mentioned nothing must resolve to
/// thinking-off whatever `--enable-thinking` an operator set for chat.
fn refuse_thinking(server: &crate::Server, request: &DecideRequest) -> Result<(), Refusal> {
    let fields = crate::thinking::ThinkingRequestFields {
        enable_thinking: request.enable_thinking.as_ref(),
        reasoning_effort: request.reasoning_effort.as_ref(),
        preserve_thinking: request.preserve_thinking.as_ref(),
        chat_template_kwargs: request.chat_template_kwargs.as_ref(),
    };
    let defaults = crate::thinking::ThinkingDefaults {
        enable_thinking: false,
        reasoning_effort: None,
    };
    let resolved = crate::thinking::resolve(fields, &defaults, &server.template.thinking_capabilities())
        .map_err(|error| match error {
            crate::thinking::ThinkingError::Validation(message)
            | crate::thinking::ThinkingError::Capability(message) => {
                Refusal::new("malformed_request", message)
            }
        })?;
    if resolved.enable_thinking {
        return Err(Refusal::new(
            "thinking_unsupported",
            "a decision reads one position and generates nothing, so thinking cannot be honoured; omit `enable_thinking` and `reasoning_effort` or turn them off",
        ));
    }
    Ok(())
}

/// One question's prompt, rendered and ready to submit.
struct Rendered {
    input: ignis_core::types::RequestInput,
    model: String,
    prompt_tokens: u32,
    media: Option<crate::media::MediaStats>,
}

/// Build one question's prompt. Refuses the whole request on failure: a
/// prompt that cannot be rendered, or one longer than the engine's context,
/// is the caller's mistake and every sibling shares it.
async fn render(
    server: &crate::Server,
    evidence: &Evidence,
    question: &PreparedQuestion,
    model: Option<String>,
) -> Result<Rendered, Refusal> {
    let messages = messages_for(evidence, question);
    let thinking = crate::thinking::ThinkingOptions {
        enable_thinking: false,
        ..crate::thinking::ThinkingOptions::default()
    };
    let params = ignis_core::types::DecodeParams::default();
    let (mut input, model, prompt_tokens, media) =
        crate::api::prepare_decision_request(server, model, &messages, params, &thinking)
            .await
            .map_err(|(code, message)| {
                Refusal::new(code, format!("question {:?}: {message}", question.id))
            })?;
    if prompt_tokens > server.engine.max_model_len() {
        return Err(Refusal::new(
            "context_exceeded",
            format!(
                "question {:?} renders {prompt_tokens} prompt tokens, past this engine's {} context",
                question.id,
                server.engine.max_model_len()
            ),
        ));
    }
    input.decision = Some(std::sync::Arc::from(
        question.answers.iter().map(|answer| answer.id).collect::<Vec<_>>(),
    ));
    Ok(Rendered { input, model, prompt_tokens, media })
}

/// Put one rendered question to the model and shape its answer.
///
/// A failure here is per-question by design (spec 04): the engine already
/// answered this question's siblings, and throwing their paid-for answers
/// away to report one runtime fault helps nobody. Everything a *caller*
/// could have got wrong was refused before any of this ran.
async fn ask(
    server: &crate::Server,
    question: &PreparedQuestion,
    ready: Rendered,
    class: ignis_core::types::RequestClass,
) -> Attempt {
    // Cloned because `submit_with_media` consumes what it takes and a
    // `Full` has to be retriable: a prompt's worth of token ids beside a
    // prefill is nothing.
    let submitted = server
        .engine
        .submit_with_media(ready.input.clone(), class, ready.media)
        .await;
    let (id, mut events) = match submitted {
        Ok(pair) => pair,
        // Not an answer. The engine is saying "not now", and a fan-out's
        // own siblings are the likeliest reason.
        Err(ignis_core::SubmitError::Full) => return Attempt::Full(ready),
        Err(error) => return Attempt::Answered(failed("submit_failed", format!("{error:?}"))),
    };
    // The engine keeps working on a request whose caller has gone until it
    // is told otherwise, and a fan-out is twenty of them.
    let mut guard = crate::api::CancelOnDrop::new(server.engine.clone(), id);
    Attempt::Answered(
        match crate::engine::collect_readout(&mut events, server.request_timeout).await {
            Ok(readout) => {
                guard.completed();
                answer_for(question, &readout)
            }
            Err(_) => failed(
                "not_completed",
                "the engine did not answer this question in time".to_owned(),
            ),
        },
    )
}

/// What one attempt at a question produced: an answer, or the unsubmitted
/// question back because the engine had no room for it.
enum Attempt {
    Answered(Answer),
    Full(Rendered),
}

fn engine_full() -> Answer {
    failed(
        "engine_full",
        "the engine could not admit this question (all lanes in use); retry".to_owned(),
    )
}

/// How many follower questions are put to the engine at once (GitHub #240).
///
/// Not a throughput knob — a **floor under the failure mode parallelism
/// introduces**. The engine admits `SchedulerConfig::max_in_flight` requests
/// and turns the rest away with `SubmitError::Full`: a 503 "retry" on
/// `/v1/chat/completions`, and an error in that question's slot here.
/// Twenty questions fired at once would answer eight and fail twelve, which
/// is worse than answering them one at a time — so they go in waves, and an
/// idle engine never refuses one.
///
/// **What it is tied to, and what it is not.** The number that matters is
/// `max_in_flight`, which no server flag sets: `runtime.rs` builds its
/// `SchedulerConfig` without it, so it is the default, and the default is
/// `N_DECODE_LANES`. That is the whole justification for reading it from
/// there — *not* that a decision occupies a decode lane, which it never does
/// (`CONTEXT.md`: a decision takes only the single global prefill lane). The
/// two numbers are equal today and mean different things; if `max_in_flight`
/// ever becomes configurable from the server, or core-06 raises it as its
/// doc promises, this has to be read from the engine instead of assumed.
/// Being too small is a slower fan-out, being too large is a failed one.
///
/// A fan-out sharing the engine with other clients can still meet a full
/// engine, exactly as any single request can. That is the pre-existing
/// behaviour and not something waves promise to fix; what they fix is a
/// fan-out competing with *itself*.
///
/// The cost is one `--request-timeout` per wave rather than per question:
/// twenty questions bound at four timeouts — the sequenced first, then
/// `ceil(19 / 8)` waves — instead of twenty. A wave also ends no sooner
/// than its slowest question, so one question that times out holds the next
/// wave for that long; a sliding window would not, and is not worth the
/// machinery until a fan-out is wider than two waves.
///
/// The other cost is honest to state: a leaf error fails the whole prefill
/// batch it was in (`MockCompute::fail_prefill`'s own contract), so one
/// kernel fault now takes up to a wave's worth of questions down where
/// one-at-a-time took one. Spec 04's "leaves the others' paid-for answers
/// alone" holds for a fault in one question's own readout, which is the
/// per-question failure it describes; a fault in the batch is not one.
const FAN_OUT_WIDTH: usize = ignis_core::N_DECODE_LANES;

/// Run every future to completion **at the same time**, in place, and
/// collect their outputs in order.
///
/// Fifteen lines rather than a `futures-util` dependency, but that is not
/// the reason it is hand-written. The two obvious tools are both wrong
/// here:
///
/// - `tokio::spawn` / `JoinSet` need `'static` futures, and these borrow
///   the server and the prepared question. That alone settles it, and the
///   shape it would force — detaching the work from the handler — is
///   precisely spec 04's warning: "twenty independent requests is exactly
///   the shape in which one forgets to cancel nineteen".
/// - Awaiting them in sequence is what this replaces.
///
/// `futures_util::future::join_all` does have the right semantics and would
/// be the answer if it were already here; it is not a dependency of this
/// crate (`futures-core` is, and carries no combinators), and a dependency
/// edge is a larger thing to add than the fifteen lines it saves.
///
/// Held in a `Vec` and polled in place, the futures are owned by the
/// handler's own future. A client that disconnects drops that, which drops
/// these, which drops each `CancelOnDrop` — nineteen cancelled engine
/// requests, with nobody having to remember them.
async fn concurrently<F: std::future::Future>(futures: Vec<F>) -> Vec<F::Output> {
    let mut futures: Vec<_> = futures.into_iter().map(Box::pin).collect();
    let mut done: Vec<Option<F::Output>> = futures.iter().map(|_| None).collect();
    std::future::poll_fn(move |cx| {
        let mut waiting = false;
        for (slot, future) in done.iter_mut().zip(futures.iter_mut()) {
            if slot.is_some() {
                continue;
            }
            match future.as_mut().poll(cx) {
                std::task::Poll::Ready(output) => *slot = Some(output),
                std::task::Poll::Pending => waiting = true,
            }
        }
        if waiting {
            return std::task::Poll::Pending;
        }
        // Every slot was filled above, and the future is never polled again
        // after it returns `Ready`.
        std::task::Poll::Ready(done.iter_mut().filter_map(Option::take).collect())
    })
    .await
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
