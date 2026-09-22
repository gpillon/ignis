//! `POST /v1/decide` — the decision endpoint (GitHub #239, ADR 0034).
//!
//! One `state` and a map of typed `questions`. Jev's three primitives are
//! answered from the logits of named **answer tokens** at one position:
//! nothing is generated, they cost one prefill, and `usage.output_tokens` is
//! 0 for a request of them, honestly.
//!
//! GitHub #242 adds three that *do* generate — `number`, `point` and `box` —
//! by restricting each step to a declared alphabet instead of reading one
//! position ([`crate::numbers`]). They cost one prefill and a round per
//! digit, they move `usage.output_tokens` and the throughput panels, and
//! they are counted in `ignis_decisions_total` beside the readouts. What
//! they do not have is an **answer mass**, which is why the histogram beside
//! that counter is readout-only (ADR 0017).
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
use ignis_core::types::TokenId;

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

/// A whole JSON value that keeps the order its object keys were written in.
///
/// [`Ordered`] does this for a map whose values all share one type, which is
/// what `criteria` is. A `state` is any JSON at all, and `instructions` may be
/// an object too, so those need the same promise over the whole shape — the
/// same reason, one level up.
///
/// **Without it a JSON evidence reaches the model alphabetised.**
/// `serde_json::Map` is a `BTreeMap` in this build, so `{"order":…,"note":…}`
/// serializes as `{"note":…,"order":…}`, and the record the caller wrote is
/// not the record the model reads. Measured on the loaded model: one order
/// record answered `gift` 0.32 as an object and 0.82 as the identical text,
/// and the object answered *bit-identically* to the same object with its keys
/// already sorted — which is what named the cause.
///
/// `serde_json::Number` is kept rather than an `f64`, so `149.0` is still
/// `149.0` in the prompt. The caller's own spelling of a number is part of the
/// bytes the model reads, and normalizing it moved an answer too.
#[derive(Debug, Clone, PartialEq)]
pub enum OrderedValue {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<OrderedValue>),
    /// Entries in the order they were written. Duplicates are kept, for the
    /// reason [`Ordered`] keeps them: which copy wins is a parser's choice.
    Object(Vec<(String, OrderedValue)>),
}

impl OrderedValue {
    /// A `serde_json::Value` as one of these — for a value this server built
    /// itself, where there is no caller's order to lose.
    pub fn from_json(value: &JsonValue) -> Self {
        match value {
            JsonValue::Null => Self::Null,
            JsonValue::Bool(flag) => Self::Bool(*flag),
            JsonValue::Number(number) => Self::Number(number.clone()),
            JsonValue::String(text) => Self::String(text.clone()),
            JsonValue::Array(items) => Self::Array(items.iter().map(Self::from_json).collect()),
            JsonValue::Object(fields) => {
                Self::Object(fields.iter().map(|(key, value)| (key.clone(), Self::from_json(value))).collect())
            }
        }
    }

    /// The same value as `serde_json`'s, for a path that does not care about
    /// order — reading content parts, whose shape is fixed and named.
    pub fn to_json(&self) -> JsonValue {
        match self {
            Self::Null => JsonValue::Null,
            Self::Bool(flag) => JsonValue::Bool(*flag),
            Self::Number(number) => JsonValue::Number(number.clone()),
            Self::String(text) => JsonValue::String(text.clone()),
            Self::Array(items) => JsonValue::Array(items.iter().map(Self::to_json).collect()),
            Self::Object(entries) => {
                JsonValue::Object(entries.iter().map(|(key, value)| (key.clone(), value.to_json())).collect())
            }
        }
    }

    /// Write this value as JSON, every object's entries in their own order.
    ///
    /// Hand-written for the reason [`payload_text`] is hand-written:
    /// `serde_json` would sort exactly what this exists to keep. Strings and
    /// keys still go through `serde_json` for their escaping, which is the one
    /// part of this nobody should write a second time.
    pub fn write(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
            Self::Number(number) => out.push_str(&number.to_string()),
            Self::String(text) => out.push_str(&quoted(text)),
            Self::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Self::Object(entries) => {
                out.push('{');
                for (index, (key, value)) in entries.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    out.push_str(&quoted(key));
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }

    /// This value as JSON text, every object's entries in their own order.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    /// The text this value carries, if it is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }
}

/// A JSON string literal, escaped the way `serde_json` escapes one.
fn quoted(text: &str) -> String {
    serde_json::to_string(text).expect("a string always serializes")
}

impl fmt::Display for OrderedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
    }
}

impl<'de> Deserialize<'de> for OrderedValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct AnyValue;

        impl<'de> Visitor<'de> for AnyValue {
            type Value = OrderedValue;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("any JSON value")
            }

            fn visit_unit<E: serde::de::Error>(self) -> Result<OrderedValue, E> {
                Ok(OrderedValue::Null)
            }

            fn visit_none<E: serde::de::Error>(self) -> Result<OrderedValue, E> {
                Ok(OrderedValue::Null)
            }

            fn visit_some<D: Deserializer<'de>>(self, inner: D) -> Result<OrderedValue, D::Error> {
                OrderedValue::deserialize(inner)
            }

            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<OrderedValue, E> {
                Ok(OrderedValue::Bool(value))
            }

            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<OrderedValue, E> {
                Ok(OrderedValue::Number(value.into()))
            }

            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<OrderedValue, E> {
                Ok(OrderedValue::Number(value.into()))
            }

            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<OrderedValue, E> {
                // A non-finite float is not JSON, and `null` is what
                // `serde_json` itself writes in place of one.
                Ok(serde_json::Number::from_f64(value).map_or(OrderedValue::Null, OrderedValue::Number))
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<OrderedValue, E> {
                Ok(OrderedValue::String(value.to_owned()))
            }

            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<OrderedValue, E> {
                Ok(OrderedValue::String(value))
            }

            fn visit_seq<S: serde::de::SeqAccess<'de>>(self, mut access: S) -> Result<OrderedValue, S::Error> {
                let mut items = Vec::with_capacity(access.size_hint().unwrap_or(0));
                while let Some(item) = access.next_element()? {
                    items.push(item);
                }
                Ok(OrderedValue::Array(items))
            }

            fn visit_map<M: MapAccess<'de>>(self, mut access: M) -> Result<OrderedValue, M::Error> {
                let mut entries = Vec::with_capacity(access.size_hint().unwrap_or(0));
                while let Some(entry) = access.next_entry::<String, OrderedValue>()? {
                    entries.push(entry);
                }
                Ok(OrderedValue::Object(entries))
            }
        }

        deserializer.deserialize_any(AnyValue)
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
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct DecideRequest {
    /// The content to evaluate: a string, a JSON object or array, **or**
    /// OpenAI content parts, so the evidence may be an image. The content
    /// parts are ours; Jev's `state` is `string | object | array`, which
    /// makes this a superset rather than a clone.
    ///
    /// [`OrderedValue`] and not `JsonValue`, because an object's key order is
    /// part of what the model reads and `serde_json::Map` would sort it away.
    /// That order is a real property of this endpoint and one JSON Schema
    /// cannot express: the document can say "any JSON", not "read in the
    /// order you wrote it".
    #[schema(value_type = serde_json::Value)]
    pub state: OrderedValue,
    /// The model to route to; the loaded model when absent or blank.
    #[serde(default)]
    pub model: Option<String>,
    /// The typed questions, keyed by ids the caller chooses. Answers come
    /// back under the same ids, and the questions are asked in the order
    /// they were written (which the schema below cannot state).
    #[schema(value_type = std::collections::BTreeMap<String, Question>)]
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
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct Question {
    /// `noul`, `choice` or `score`.
    #[serde(rename = "type")]
    pub kind: QuestionKind,
    /// What the model should decide. A string, object or array — anything
    /// but a string is serialized into the prompt as JSON, in the order it
    /// was written ([`OrderedValue`]). Also accepted as `question`.
    #[serde(alias = "question")]
    #[schema(value_type = serde_json::Value)]
    pub instructions: OrderedValue,
    /// The type's own options: absent for a bare `noul`, a map for a
    /// `choice`, an ordered array for a `score`. A `choice`'s map is read in
    /// declared order — a different option order is a different prompt — and
    /// that, too, is outside what a schema can say. Also accepted as
    /// `options`.
    #[serde(default, alias = "options")]
    #[schema(value_type = Option<serde_json::Value>)]
    pub criteria: Option<Criteria>,
    /// Digits per axis for a `number`, `point` or `box` (GitHub #242);
    /// [`crate::numbers::DEFAULT_DIGITS`] when absent, and refused outside
    /// [`crate::numbers::DIGITS`].
    ///
    /// For a `scalar` (GitHub #255) it is a **ceiling** rather than a width,
    /// which is a different quantity with its own range and its own default:
    /// [`crate::scalar::DIGITS`] and [`crate::scalar::DEFAULT_DIGITS`]. A
    /// field must be filled and a ceiling need not, so the two cannot share
    /// a bound.
    ///
    /// Refused rather than ignored on a readout question, like every other
    /// field this endpoint cannot honour: a caller who wrote `digits` on a
    /// `choice` meant something by it.
    #[serde(default)]
    pub digits: Option<u32>,
    /// How a `point` is answered (GitHub #260): `"head"` reads the loaded
    /// artifact's calibrated **pointing head** in one pass — one prefill, no
    /// decode round — and `"chain"` writes the digits one decode round at a
    /// time. Absent, it is `head` when the load has a calibrated head and
    /// `chain` otherwise; every point answer says which ran.
    ///
    /// The chain stays for what it is still better at: a point precise to
    /// less than one image token (the head's resolution is one token, 32 px
    /// of an unresized image), and a per-digit trace. Refused on every other
    /// type — a `box` cannot come out of the head, whose region is not a box
    /// — and an unknown value is refused naming the two, never read as the
    /// default.
    #[serde(default)]
    #[schema(value_type = Option<PointMethod>)]
    pub method: Option<String>,
}

/// How a `point` is answered (GitHub #260, spec 13).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum PointMethod {
    /// One pass: the calibrated pointing head's attention over the image,
    /// read at the forced `{"x":` and turned into a point on the host.
    /// Coarse — one image token — and on labelled targets it marks where the
    /// label begins, not the target's centre.
    Head,
    /// The digit chain: the forced `{"x":` and then the digits, one decode
    /// round each, under a constrained decode (GitHub #242).
    Chain,
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

/// The seven primitives: Jev's three, which read one position, and the four
/// that **generate** (GitHub #242, #255).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum QuestionKind {
    /// A yes/no question, answered with the probability of yes.
    #[serde(alias = "boolean")]
    Noul,
    /// One option from a declared set.
    Choice,
    /// A probability-weighted value across ordered levels.
    Score,
    /// A whole number, read digit by digit into a field of fixed width.
    Number,
    /// A number that decides its own width, and may have a decimal part
    /// (GitHub #255): the same mechanism with the closing brace in the
    /// alphabet, so the run ends when the number is complete.
    Scalar,
    /// Two numbers: a position on the submitted image.
    Point,
    /// Four numbers: a bounding box on the submitted image.
    #[serde(rename = "box")]
    Box,
}

impl QuestionKind {
    /// The label this primitive is counted under (GitHub #241).
    ///
    /// Stated here, at the wire type, rather than in the projection: the
    /// projection owns the label set and this owns which of its values a
    /// question maps to, so a primitive added to the wire fails to compile
    /// until somebody decides what it is counted as.
    fn primitive(self) -> crate::metrics::Primitive {
        match self {
            Self::Noul => crate::metrics::Primitive::Noul,
            Self::Choice => crate::metrics::Primitive::Choice,
            Self::Score => crate::metrics::Primitive::Score,
            Self::Number => crate::metrics::Primitive::Number,
            Self::Scalar => crate::metrics::Primitive::Scalar,
            Self::Point => crate::metrics::Primitive::Point,
            Self::Box => crate::metrics::Primitive::Box,
        }
    }

    /// The axis layout this primitive generates under, or `None` for a
    /// readout (GitHub #242).
    fn layout(self) -> Option<crate::numbers::Layout> {
        match self {
            // A scalar generates, but not over axes: its plan is
            // `crate::scalar`'s, so it has no layout here and
            // `is_constrained` cannot be `layout().is_some()` any more.
            Self::Noul | Self::Choice | Self::Score | Self::Scalar => None,
            Self::Number => Some(crate::numbers::NUMBER_LAYOUT),
            Self::Point => Some(crate::numbers::POINT_LAYOUT),
            Self::Box => Some(crate::numbers::BOX_LAYOUT),
        }
    }

    /// Whether this primitive's answer is in **pixels of the submitted
    /// image** (GitHub #242), and therefore needs one.
    fn is_spatial(self) -> bool {
        matches!(self, Self::Point | Self::Box)
    }

    /// Whether this primitive answers with a **constrained decode**
    /// (`CONTEXT.md`) rather than a readout of one position.
    pub fn is_constrained(self) -> bool {
        self.layout().is_some() || self == Self::Scalar
    }

    /// The system text a question of this primitive is put under:
    /// [`DIRECT_SYSTEM`] for a readout, and the shape-declaring text of each
    /// constrained one (GitHub #242, [`crate::numbers`]).
    ///
    /// A constrained decode's is not `DIRECT_SYSTEM` with a clause bolted
    /// on. It has to declare the **scale**, which is what the finding
    /// established its accuracy against, and `DIRECT_SYSTEM` says "choose
    /// exactly one listed option" — which a number does not do.
    ///
    /// On the kind rather than beside the prompt builder: the builder reads
    /// nothing else of the question to decide this, and a primitive added to
    /// the wire should fail to compile until somebody writes its
    /// instruction.
    pub fn system_text(self, digits: u32) -> String {
        match self {
            Self::Number => crate::numbers::number_system(digits),
            Self::Scalar => crate::scalar::scalar_system(digits),
            Self::Point => crate::numbers::point_system(digits),
            Self::Box => crate::numbers::box_system(digits),
            Self::Noul | Self::Choice | Self::Score => DIRECT_SYSTEM.to_owned(),
        }
    }
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
    pub instructions: OrderedValue,
    /// The options in declared order: what the caller called each one, and
    /// the description the prompt gives it.
    pub options: Vec<PreparedOption>,
    /// The answer token per option, parallel to `options`.
    pub answers: Vec<AnswerToken>,
    /// The schedule a **constrained decode** question generates under (GitHub #242),
    /// and `None` for a readout. `options` and `answers` are then empty: a
    /// program names no options, it forces an alphabet.
    pub plan: Option<std::sync::Arc<crate::numbers::Plan>>,
    /// The plan a `scalar` generates under (GitHub #255), and `None` for
    /// every other primitive.
    ///
    /// Beside `plan` rather than inside it: a scalar has one axis, no
    /// separators and a terminator, so an `Axis` list would be a shape with
    /// two of its three fields unused and `numbers::read` would grow a
    /// branch for a run it cannot weight — a digit's place there is not
    /// known until the decimal point has been seen.
    pub scalar: Option<std::sync::Arc<crate::scalar::Plan>>,
    /// Digits per axis for a constrained question, or the **maximum** for a
    /// scalar.
    pub digits: u32,
    /// The pointing head a `point` is answered from in one pass (GitHub
    /// #260), or `None` for a point answered by the chain and for every
    /// other primitive.
    ///
    /// A head point keeps its `plan`: the prompt is the chain's byte for
    /// byte — same system text, same forced `{"x":` — so head and chain
    /// questions over one image share their prefix, and the head's query
    /// sits inside the answer's scaffold where it was measured. Only the
    /// schedule goes unused.
    pub head: Option<ignis_core::pointing::PointingHead>,
}

impl PreparedQuestion {
    /// How this question is answered, if it is a `point`.
    pub fn point_method(&self) -> Option<PointMethod> {
        (self.kind == QuestionKind::Point).then_some(match self.head {
            Some(_) => PointMethod::Head,
            None => PointMethod::Chain,
        })
    }
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
///
/// `pointing` is the load's calibrated pointing head (GitHub #260), `None`
/// on a load nobody calibrated one for: it is what a `point` with no
/// `method` is answered from, and what a `point` asking for `head` is
/// refused without.
pub fn prepare(
    questions: &Ordered<Question>,
    alphabet: &AnswerAlphabet,
    encode: Encoder<'_>,
    pointing: Option<ignis_core::pointing::PointingHead>,
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
        .map(|(id, question)| prepare_one(id, question, alphabet, encode, pointing))
        .collect()
}

/// The loaded tokenizer, as a run's plan needs it: text to token ids
/// (GitHub #242). [`crate::template::TemplateProvider::encode_literal`] is
/// what supplies one.
pub type Encoder<'a> = &'a dyn Fn(&str) -> Option<Vec<TokenId>>;

fn prepare_one(
    id: &str,
    question: &Question,
    alphabet: &AnswerAlphabet,
    encode: Encoder<'_>,
    pointing: Option<ignis_core::pointing::PointingHead>,
) -> Result<PreparedQuestion, Refusal> {
    if instructions_are_empty(&question.instructions) {
        return Err(Refusal::new(
            "empty_instructions",
            format!("question {id:?} has empty `instructions`"),
        ));
    }
    let head = point_head(id, question, pointing)?;
    if question.kind == QuestionKind::Scalar {
        return prepare_scalar(id, question, encode);
    }
    if let Some(layout) = question.kind.layout() {
        return prepare_program(id, question, layout, encode)
            .map(|prepared| PreparedQuestion { head, ..prepared });
    }
    if question.digits.is_some() {
        return Err(Refusal::new(
            "digits_unsupported",
            format!(
                "question {id:?} is a {} and reads one position, so `digits` cannot be honoured",
                question.kind.primitive().label()
            ),
        ));
    }
    let options = match question.kind {
        QuestionKind::Noul => noul_options(id, question.criteria.as_ref())?,
        QuestionKind::Choice => choice_options(id, question.criteria.as_ref())?,
        QuestionKind::Score => score_options(id, question.criteria.as_ref())?,
        // Unreachable: `prepare_one` returns above for the scalar and for
        // every kind with a layout, which is exactly these four.
        QuestionKind::Scalar
        | QuestionKind::Number
        | QuestionKind::Point
        | QuestionKind::Box => Vec::new(),
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
        plan: None,
        scalar: None,
        digits: 0,
        head: None,
    })
}

/// Resolve a question's `method` (GitHub #260): the pointing head a `point`
/// is answered from, `None` for a chain point and for every other primitive
/// — or a refusal, before any prefill.
///
/// Omitted, the measured-better method runs: the head when the load has a
/// calibrated one, the chain otherwise. Asked for by name, the head is
/// refused on a load without one rather than silently served by the chain
/// the caller explicitly did not ask for.
fn point_head(
    id: &str,
    question: &Question,
    pointing: Option<ignis_core::pointing::PointingHead>,
) -> Result<Option<ignis_core::pointing::PointingHead>, Refusal> {
    let Some(method) = question.method.as_deref() else {
        return Ok(pointing.filter(|_| question.kind == QuestionKind::Point));
    };
    if question.kind != QuestionKind::Point {
        let why = match question.kind {
            QuestionKind::Box => {
                "a box cannot come out of the pointing head (its region is not a box), so a box is always the chain's"
            }
            _ => "`method` chooses how a `point` is answered, and this primitive has one way",
        };
        return Err(Refusal::new(
            "method_unsupported",
            format!(
                "question {id:?} is a {}: {why}",
                question.kind.primitive().label()
            ),
        ));
    }
    match method {
        "chain" => Ok(None),
        "head" => pointing.map(Some).ok_or_else(|| {
            Refusal::new(
                "pointing_head_unavailable",
                format!(
                    "question {id:?} asks for `\"method\": \"head\"`, and the loaded artifact has no calibrated pointing head; ask for \"chain\", or omit `method` to be answered by the chain"
                ),
            )
        }),
        other => Err(Refusal::new(
            "method_unknown",
            format!(
                "question {id:?} asks for method {other:?}; the accepted values are \"head\" and \"chain\""
            ),
        )),
    }
}

/// Validate and plan a `scalar` question (GitHub #255, spec 10).
///
/// `digits` is a **maximum** here and not a width, so the default is the
/// ceiling rather than a guess: a caller who does not know the magnitude is
/// exactly the caller this primitive exists for, and one who writes nothing
/// should get the widest answer the run can close early out of.
fn prepare_scalar(
    id: &str,
    question: &Question,
    encode: Encoder<'_>,
) -> Result<PreparedQuestion, Refusal> {
    if question.criteria.is_some() {
        return Err(Refusal::new(
            "criteria_unsupported",
            format!(
                "question {id:?} is a scalar and declares no options, so `criteria` cannot be honoured"
            ),
        ));
    }
    let digits = question.digits.unwrap_or(crate::scalar::DEFAULT_DIGITS);
    if !crate::scalar::DIGITS.contains(&digits) {
        return Err(Refusal::new(
            "digits_out_of_range",
            format!(
                "question {id:?} allows {digits} digits; {}..={} is the range a scalar serves — and this is a ceiling, not a width, so a caller who does not know the magnitude should leave it out and get {}",
                crate::scalar::DIGITS.start,
                crate::scalar::DIGITS.end - 1,
                crate::scalar::DEFAULT_DIGITS
            ),
        ));
    }
    let plan = crate::scalar::plan(digits, encode).map_err(|error| {
        // A property of the load and not of the request: this tokenizer
        // cannot spell a digit, a point, a sign or a brace as one token, so
        // this model cannot answer a scalar at all.
        Refusal::new("digits_unnameable", format!("question {id:?}: {error}"))
    })?;
    Ok(PreparedQuestion {
        id: id.to_owned(),
        kind: question.kind,
        instructions: question.instructions.clone(),
        options: Vec::new(),
        answers: Vec::new(),
        plan: None,
        scalar: Some(std::sync::Arc::new(plan)),
        digits,
        head: None,
    })
}

/// Validate and plan a **constrained decode** question (GitHub #242): a `number`,
/// `point` or `box`.
///
/// It declares no options — it forces the digit alphabet — so nothing here
/// touches the answer alphabet, and `criteria` is refused rather than
/// ignored: a caller who wrote one meant this to be a choice.
fn prepare_program(
    id: &str,
    question: &Question,
    layout: crate::numbers::Layout,
    encode: Encoder<'_>,
) -> Result<PreparedQuestion, Refusal> {
    if question.criteria.is_some() {
        return Err(Refusal::new(
            "criteria_unsupported",
            format!(
                "question {id:?} is a {} and declares no options, so `criteria` cannot be honoured",
                question.kind.primitive().label()
            ),
        ));
    }
    let digits = question.digits.unwrap_or(crate::numbers::DEFAULT_DIGITS);
    if !crate::numbers::DIGITS.contains(&digits) {
        return Err(Refusal::new(
            "digits_out_of_range",
            format!(
                "question {id:?} asks for {digits} digits; {}..={} is the range this endpoint serves — one digit is a choice between ten answers, and six is already far past the resolution the model reports for itself",
                crate::numbers::DIGITS.start,
                crate::numbers::DIGITS.end - 1
            ),
        ));
    }
    let plan = crate::numbers::plan(layout, digits, encode).map_err(|error| {
        // Every one of these is a property of the load, not of the request:
        // this tokenizer cannot spell a digit as one token, so this model
        // cannot answer a number at all.
        Refusal::new("digits_unnameable", format!("question {id:?}: {error}"))
    })?;
    Ok(PreparedQuestion {
        id: id.to_owned(),
        kind: question.kind,
        instructions: question.instructions.clone(),
        options: Vec::new(),
        answers: Vec::new(),
        plan: Some(std::sync::Arc::new(plan)),
        scalar: None,
        digits,
        head: None,
    })
}

/// Whether `instructions` says nothing at all — an empty string, an empty
/// object or array, or `null`. A question with no instructions has no
/// criterion to apply, and the model would answer the options alone.
fn instructions_are_empty(instructions: &OrderedValue) -> bool {
    match instructions {
        OrderedValue::Null => true,
        OrderedValue::String(text) => text.trim().is_empty(),
        OrderedValue::Array(items) => items.is_empty(),
        OrderedValue::Object(entries) => entries.is_empty(),
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
    // GitHub #242: a run's user turn is the instruction alone — it
    // declares no options, and the shape its answer must take lives in the
    // system text, which is the shape the finding measured.
    let ask = match question.kind.is_constrained() {
        true => payload_text(&[("instruction", &question.instructions)]),
        false => {
            // `description` before `letter`, which is **not** the order this
            // reads in. It is the order `json!` used to emit, because
            // `serde_json::Map` sorted these two keys — and every number this
            // endpoint rests on was measured against those bytes
            // (`classify_option_ceiling_gpu.rs` and
            // `classify_readout_gpu.rs` still build their options that way).
            // Writing them in the order a reader would choose would be a
            // prompt nobody has measured, so the accident is kept and named.
            let options: Vec<OrderedValue> = question
                .options
                .iter()
                .zip(&question.answers)
                .map(|(option, answer)| {
                    OrderedValue::Object(vec![
                        ("description".to_owned(), OrderedValue::String(option.description.clone())),
                        ("letter".to_owned(), OrderedValue::String(answer.label.clone())),
                    ])
                })
                .collect();
            payload_text(&[
                ("criterion", &question.instructions),
                ("options", &OrderedValue::Array(options)),
            ])
        }
    };
    let instruction = question.kind.system_text(question.digits);
    match state {
        Evidence::Json(value) => {
            // One blank line between the instruction and the evidence: the
            // instruction is the same bytes for every question over every
            // state, so a reader — and a retained prefix — meets it first.
            let system = format!("{instruction}\n\n{}", payload_text(&[("evidence", value)]));
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
                ChatMessage::text("system", instruction),
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
fn payload_text(fields: &[(&str, &OrderedValue)]) -> String {
    let mut out = String::from("{");
    for (index, (name, value)) in fields.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(name);
        out.push_str("\":");
        value.write(&mut out);
    }
    out.push('}');
    out
}

/// The evidence a decision is put to.
#[derive(Debug, Clone, PartialEq)]
pub enum Evidence {
    /// A string, object or array — Jev's own `state`, carried into the
    /// prompt's JSON payload in the order it was written.
    Json(OrderedValue),
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
    pub fn read(state: &OrderedValue) -> Self {
        let OrderedValue::Array(items) = state else {
            return Self::Json(state.clone());
        };
        let parts_shaped = !items.is_empty()
            && items.iter().all(|item| match item {
                OrderedValue::Object(entries) => entries
                    .iter()
                    .any(|(key, value)| key == "type" && value.as_str().is_some()),
                _ => false,
            });
        if !parts_shaped {
            return Self::Json(state.clone());
        }
        // Content parts are a named shape with no order to keep, so they read
        // through `serde_json` like every other typed body on this server.
        match serde_json::from_value::<Vec<ContentPart>>(state.to_json()) {
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
#[derive(Debug, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
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
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, utoipa::ToSchema)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// One question's answer, or the error that stands in its place.
#[derive(Debug, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
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
    /// A number that chose its own width (GitHub #255).
    ///
    /// `value` is an `f64` and `text` is what the model actually wrote,
    /// because a caller checking a reading against its trace wants the
    /// spelling that produced it — `3` and `3.0` are the same number and
    /// not the same answer.
    ///
    /// `uncertainty` is in units of the value, as [`Answer::Number`]'s is,
    /// but it cannot be computed the same way: with a decimal point a
    /// digit's place is not known until the point has been seen, so the run
    /// is parsed before it is weighted.
    Scalar {
        value: f64,
        text: String,
        uncertainty: f64,
        digits: Vec<crate::numbers::DigitDraw>,
    },
    /// A whole number, read digit by digit (GitHub #242).
    ///
    /// `uncertainty` is in units of the number itself — not a 0-1 score —
    /// and `digits` is the trace it is computed from, so a caller can see
    /// *which* place the model is unsure about rather than only how much.
    Number {
        number: u64,
        uncertainty: f64,
        digits: Vec<crate::numbers::DigitDraw>,
    },
    /// A position on the submitted image, in **its** pixels.
    ///
    /// `normalized` is the model's own 0-`scale` reading beside it, and
    /// `uncertainty` is in pixels on each axis. The server does the
    /// rescaling because per-axis normalization on a non-square image is the
    /// mistake everyone makes once.
    ///
    /// `method` says which answered it (GitHub #260), and the rest follows
    /// from it:
    ///
    /// - **`head`** — one pass over the calibrated pointing head's attention.
    ///   `uncertainty` is the map's resolution, one image token per axis, and
    ///   not a spread. `region` is the cells the point was read from and
    ///   their `share` of the head's attention over the image: the
    ///   confidence to act on (it separates hits from misses on the measured
    ///   scenes), not a calibrated probability. On labelled targets the point
    ///   sits where the label **begins** (about 30% across a button), not at
    ///   the target's centre; on unlabelled targets it is unmeasured. No
    ///   `digits`.
    /// - **`chain`** — the digit chain: `uncertainty` from the digits'
    ///   distributions, and `digits` the trace it came from.
    Point {
        method: PointMethod,
        pixels: BTreeMap<String, i64>,
        normalized: BTreeMap<String, u64>,
        uncertainty: BTreeMap<String, f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        region: Option<HeadRegion>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        digits: Option<BTreeMap<String, Vec<crate::numbers::DigitDraw>>>,
    },
    /// A bounding box on the submitted image: [`Answer::Point`]'s shape over
    /// `x0`, `y0`, `x1`, `y1`.
    #[serde(rename = "box")]
    Box {
        pixels: BTreeMap<String, i64>,
        normalized: BTreeMap<String, u64>,
        uncertainty: BTreeMap<String, f64>,
        digits: BTreeMap<String, Vec<crate::numbers::DigitDraw>>,
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

/// The region a head point was read from (GitHub #260).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
pub struct HeadRegion {
    /// The image-token cells the point is the weighted centre of — one on
    /// most answers: the head's map is that peaked.
    pub cells: u32,
    /// Those cells' share of the head's attention over the image, 0 to 1.
    /// A diffuse map is a weaker answer; not a calibrated probability.
    pub share: f64,
}

/// Shape a head point's **attention readout** into its answer (GitHub #260).
///
/// `grid` is the image's merged token grid, `(rows, cols)`; `pixels` the
/// submitted image's `(width, height)`. The point maps to each through its
/// own side on each axis, which is the processor's own scale: it resizes the
/// whole image onto the grid.
pub fn head_answer_for(
    question: &PreparedQuestion,
    scores: &[f32],
    grid: (usize, usize),
    pixels: (u32, u32),
) -> Answer {
    let Some(reading) = ignis_core::pointing::read_head_map(scores, grid.0, grid.1) else {
        return failed(
            "attention_malformed",
            format!(
                "the pointing head's map carried {} scores for a {}x{} image grid, or a score that is not finite",
                scores.len(),
                grid.0,
                grid.1
            ),
        );
    };
    let (width, height) = pixels;
    let (x, y) = reading.pixels(width, height);
    let (cell_w, cell_h) = reading.cell_pixels(width, height);
    let (nx, ny) = reading.normalized(crate::numbers::scale(question.digits));
    fn axes<T>(x: T, y: T) -> BTreeMap<String, T> {
        BTreeMap::from([("x".to_owned(), x), ("y".to_owned(), y)])
    }
    Answer::Point {
        method: PointMethod::Head,
        pixels: axes(x.round() as i64, y.round() as i64),
        normalized: axes(nx, ny),
        uncertainty: axes(cell_w, cell_h),
        region: Some(HeadRegion {
            cells: u32::try_from(reading.cells).unwrap_or(u32::MAX),
            share: reading.share,
        }),
        digits: None,
    }
}

/// Shape a **constrained decode**'s finished run into `question`'s answer (GitHub
/// #242).
///
/// `pixels` is the submitted image's `(width, height)`, needed by `point`
/// and `box` and ignored by `number`.
///
/// A run shorter than the schedule is an **error**, not a smaller
/// uncertainty: `SchedEvent::Done`'s trace says how far a cut-off run got,
/// and a place-weighted sum over a partial number is a plausible-looking
/// wrong answer rather than a missing one — the exact failure mode this
/// endpoint exists to remove.
pub fn constrained_answer_for(
    question: &PreparedQuestion,
    drawn: &[ignis_core::constrained::Draw],
    pixels: Option<(u32, u32)>,
) -> Answer {
    if question.kind == QuestionKind::Scalar {
        let Some(plan) = &question.scalar else {
            return failed("not_a_program", "this scalar question carries no plan".to_owned());
        };
        // The length is not the signal here (spec 10): a run that closed its
        // own object is complete with steps to spare, and only a run that
        // stopped without closing *and* short of the cap was cut off. The
        // reader owns that distinction because the reader is what sees the
        // last token.
        return match crate::scalar::read(plan, drawn) {
            Ok(reading) => Answer::Scalar {
                value: reading.value,
                text: reading.text,
                uncertainty: reading.uncertainty,
                digits: reading.digits,
            },
            Err(crate::scalar::ReadError::OffAlphabet) => {
                failed("run_off_alphabet", crate::scalar::ReadError::OffAlphabet.to_string())
            }
            Err(crate::scalar::ReadError::CutShort) => {
                failed("run_cut_short", crate::scalar::ReadError::CutShort.to_string())
            }
            Err(error @ crate::scalar::ReadError::Malformed(_)) => {
                failed("malformed_scalar", error.to_string())
            }
            // Its own code: the run *is* a number, and a caller told it was
            // malformed would look for the fault in the wrong place.
            Err(error @ crate::scalar::ReadError::TooManyDigits { .. }) => {
                failed("too_many_digits", error.to_string())
            }
        };
    }
    let Some(plan) = &question.plan else {
        return failed("not_a_program", "this question generates nothing".to_owned());
    };
    if drawn.len() != plan.schedule.len() {
        return failed(
            "run_cut_short",
            format!(
                "the engine committed {} of this question's {} forced tokens, so its answer would be a number with digits missing from the middle",
                drawn.len(),
                plan.schedule.len()
            ),
        );
    }
    let Some(readings) = crate::numbers::read(plan, drawn) else {
        return failed(
            "run_off_alphabet",
            "a forced step committed a token outside its own permitted set".to_owned(),
        );
    };
    let digits: BTreeMap<String, Vec<crate::numbers::DigitDraw>> = readings
        .iter()
        .map(|(axis, reading)| (axis.clone(), reading.digits.clone()))
        .collect();
    if question.kind == QuestionKind::Number {
        let reading = &readings["value"];
        return Answer::Number {
            number: reading.value,
            uncertainty: reading.sigma,
            digits: reading.digits.clone(),
        };
    }
    debug_assert!(
        question.kind.is_spatial(),
        "every constrained decode that is not a number answers on an image"
    );
    // Spatial: the answer is in pixels of the image the caller submitted,
    // and each axis is scaled by its own side.
    let Some((width, height)) = pixels else {
        return no_image();
    };
    let mut in_pixels = BTreeMap::new();
    let mut normalized = BTreeMap::new();
    let mut uncertainty = BTreeMap::new();
    for (axis, reading) in &readings {
        let side = if axis.starts_with('x') { width } else { height };
        let (value, sigma) = reading.to_pixels(question.digits, side);
        in_pixels.insert(axis.clone(), value);
        normalized.insert(axis.clone(), reading.value);
        uncertainty.insert(axis.clone(), sigma);
    }
    match question.kind {
        QuestionKind::Box => Answer::Box {
            pixels: in_pixels,
            normalized,
            uncertainty,
            digits,
        },
        _ => Answer::Point {
            method: PointMethod::Chain,
            pixels: in_pixels,
            normalized,
            uncertainty,
            region: None,
            digits: Some(digits),
        },
    }
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
        // A constrained decode never reaches here: `ask` routes it to
        // `constrained_answer_for`, which is the only function that has a run to
        // shape. Answered rather than `unreachable!()` because this is the
        // request path and a wrong route is a bug to report, not a panic on
        // the model's thread.
        QuestionKind::Scalar
        | QuestionKind::Number
        | QuestionKind::Point
        | QuestionKind::Box => failed(
            "not_a_readout",
            "this question generates its answer and reads no position".to_owned(),
        ),
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
#[utoipa::path(
    post,
    path = "/v1/decide",
    tag = "decide",
    operation_id = "decide",
    summary = "A typed decision, read rather than generated",
    description = "Evaluates `state` against typed `questions` and answers each one from the model's own readout at a single position (ADR 0034): the decision is read out of the forward pass, not generated, so `usage.output_tokens` is 0 for the readout kinds.

Seven primitives. Read at one position: `noul` (yes/no, answered with the probability of yes), `choice` (one option from a declared set, with the distribution over all of them) and `score` (a probability-weighted value across ordered levels, which can land between them). Generated a digit at a time under a constrained decode: `number`, `point` and `box` -- the last two in the submitted image's own pixels -- and `scalar`, which closes its own object as soon as the number is complete, so `digits` is a ceiling the caller can leave out and the answer may have a decimal part.

A `point` is answered **in one pass** by default (ADR 0038): the prefill that forces its `{\"x\":` reads the loaded artifact's calibrated pointing head over the image and the server turns that map into a point, with no decode round (`method: head`). Its `uncertainty` is one image token per axis, its `region.share` is the confidence to act on, and on a labelled target it sits where the label begins rather than at the target's centre. `\"method\": \"chain\"` asks for the digit chain instead -- finer than one token, with a per-digit trace -- and a load with no calibrated head answers every `point` by chain. Every point answer names its `method`.

Every fault a caller can commit refuses the whole request with a 422 before the first submit: a caller never pays a prefill for nineteen good questions and a refusal on the twentieth. Only an engine fault lands per-answer, as an `error` answer beside its siblings.

Thinking is refused rather than ignored: a decision's prompt ends exactly where its answer is read, and a thinking prompt would put an open reasoning block at that position.

`POST /v1/systemone` is the same handler under Jev's name.",
    request_body = DecideRequest,
    responses(
        (status = 200, description = "One answer per question, under the ids the caller chose.", body = DecideResponse),
        (status = 401, description = "The server was started with `--api-key` and the request carried no matching bearer token.", body = crate::api::ApiError),
        (status = 422, description = "The body does not parse, or a question is malformed, or the request asked for something this endpoint cannot honour (thinking, an unnameable option, an image on a text-only load, a `method` on anything but a `point`, `head` on a load with no calibrated pointing head). Nothing reached the engine.",
            body = crate::api::ApiError),
        (status = 503, description = "The engine is at capacity and the request was not admitted.", body = crate::api::ApiError),
    ),
)]
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
    let started = std::time::Instant::now();
    refuse_thinking(server, &request)?;
    // The tokenizer, for any constrained question's forced alphabet (GitHub
    // #242). A load with no real tokenizer answers `None` and the question
    // is refused, rather than forcing ids this server invented.
    let encode = |text: &str| server.template.encode_literal(text);
    let prepared = prepare(&request.questions, &server.alphabet, &encode, server.pointing_head)?;
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
    log_decision(&prepared, &answers, class, input_tokens, started);
    let output_tokens = generated(&prepared, &answers);
    Ok(DecideResponse {
        // The model that *performed* the evaluation, which is the one the
        // engine resolved — not the string the caller sent.
        model: resolved.unwrap_or_else(|| server.engine.model_id()),
        answers,
        usage: Usage {
            input_tokens,
            // Zero for a request of readouts, honestly: a decision reads one
            // position's logits and samples nothing, so
            // `ignis_decoded_tokens_total` does not move for it either.
            //
            // A **constrained decode** does generate (GitHub #242), and this counts
            // what it generated. Saying 0 for a `point` would be the one lie
            // this endpoint could tell that nothing downstream would catch —
            // and the throughput panels, which a constrained decode *does* move, would
            // disagree with the usage a caller was billed by.
            output_tokens,
        },
    })
}

/// The tokens this request's **constrained decodes** generated (GitHub #242): each
/// answered constrained question's whole schedule, and nothing for a readout.
///
/// The schedule's length for a `number`, a `point` and a `box`, because
/// there they are the same number by construction — such a run ends when its
/// schedule is spent — and a question that did *not* end that way is an
/// error in its slot, with nothing to bill for.
///
/// **Not for a `scalar`** (GitHub #255), whose schedule is a ceiling it
/// usually stops short of: billing its length would charge a caller six
/// rounds for the two that answered `3`. Its run is exactly the characters
/// it wrote — digits, a point, a sign — plus the brace that closed it, and
/// the brace is there unless the run reached the cap instead.
fn generated(prepared: &[PreparedQuestion], answers: &BTreeMap<String, Answer>) -> u32 {
    prepared
        .iter()
        .filter_map(|question| match answers.get(&question.id) {
            None | Some(Answer::Error { .. }) => None,
            // No fallback when the plan is missing: that state cannot
            // happen — a scalar answer comes from a scalar plan — and a
            // fallback would make it bill like an at-cap run instead of
            // being visible.
            Some(Answer::Scalar { text, .. }) => question.scalar.as_ref().map(|plan| {
                let written = text.chars().count();
                written + usize::from(written < plan.schedule.len())
            }),
            // GitHub #260: a head point is answered by its prefill and
            // generates nothing, however long the chain's schedule would be.
            Some(Answer::Point { method: PointMethod::Head, .. }) => None,
            _ => question.plan.as_ref().map(|plan| plan.schedule.len()),
        })
        .fold(0u32, |total, tokens| {
            total.saturating_add(u32::try_from(tokens).unwrap_or(u32::MAX))
        })
}

/// The request log's line for one served `POST /v1/decide` (GitHub #241,
/// ADR 0011).
///
/// **`ignis.decide.done`, not `ignis.decision.done`.** A *decision* is the
/// engine's unit — one readout, one internal request, one question
/// (GitHub #238, and what `ignis_decisions_total` counts). A *decide
/// request* is what a caller sent, and twenty `ignis.request.admitted` /
/// `done` pairs is not a reading of it. This is that reading.
///
/// It **summarises** the fan-out; it does not tie it together. No request
/// id crosses the endpoint boundary — the ids are minted inside the engine
/// and the `ignis.request.*` events are emitted on the model thread's own
/// side — so a reader can see that a decide request asked three questions
/// and cannot join this line to the three it caused. Carrying up to 256
/// ids in a log field would not be that join either. Stated rather than
/// implied, because the obvious reading of a summary line is that it has
/// one.
///
/// `types` is the distinct primitives in declared order, spelled by
/// [`crate::metrics::Primitive::label`] so a log field and a metric label
/// value that mean the same thing *are* the same string. It is one of seven
/// values and never unbounded: the log is not a metric, but a field
/// somebody will eventually group by should be groupable.
fn log_decision(
    prepared: &[PreparedQuestion],
    answers: &BTreeMap<String, Answer>,
    class: ignis_core::types::RequestClass,
    input_tokens: u32,
    started: std::time::Instant,
) {
    let mut types: Vec<&str> = Vec::with_capacity(3);
    for question in prepared {
        // The projection's own word for the primitive, borrowed rather than
        // spelled again here: a log field and a label value that are
        // supposed to be the same string should not be two `match`es that
        // happen to agree.
        let label = question.kind.primitive().label();
        if !types.contains(&label) {
            types.push(label);
        }
    }
    let errors = answers
        .values()
        .filter(|answer| matches!(answer, Answer::Error { .. }))
        .count();
    tracing::info!(
        name: "ignis.decide.done",
        questions = prepared.len(),
        types = types.join(","),
        // The same spelling the `ignis.request.*` events give it, since a
        // decision defaults to `Agent` where every other route defaults to
        // `Interactive` (`CONTEXT.md`, *Lane tag*) and a reader comparing
        // the two should not have to know that.
        class = class.as_extension_str(),
        answered = prepared.len() - errors,
        errors,
        input_tokens,
        output_tokens = 0,
        duration_ms = started.elapsed().as_millis() as u64,
        "decide request done"
    );
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
    /// A head point's image grid, `(rows, cols)` of merged tokens (GitHub
    /// #260), and `None` for every other question.
    grid: Option<(usize, usize)>,
    /// The answer a question already has without being submitted — a head
    /// point with no image to point on, or one whose image is not a grid the
    /// region rule can read (GitHub #260). `None` for every question the
    /// engine is asked.
    answered: Option<Answer>,
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
    let (mut input, model, mut prompt_tokens, media) =
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
    // A scalar's prefix and schedule are its own module's (GitHub #255), and
    // otherwise it is a constrained decode like any other: same forced
    // opening, same context check, same MRoPE extension.
    let constrained = match (&question.plan, &question.scalar) {
        (Some(plan), _) => Some((plan.prefix.as_slice(), plan.schedule.clone())),
        (None, Some(plan)) => Some((plan.prefix.as_slice(), plan.schedule.clone())),
        (None, None) => None,
    };
    match constrained {
        // GitHub #237/#238: a readout names its answer tokens and ends where
        // its prefill ends.
        None => {
            input.decision = Some(ignis_core::DecisionRead::Answers(std::sync::Arc::from(
                question.answers.iter().map(|answer| answer.id).collect::<Vec<_>>(),
            )));
        }
        // GitHub #242: a constrained decode appends its opening literal to the prompt —
        // forced text the model appears to have written, costing prefill
        // rather than a decode round each — and carries the schedule for
        // everything after it.
        Some((prefix, schedule)) => {
            prompt_tokens = prompt_tokens.saturating_add(prefix.len() as u32);
            if prompt_tokens > server.engine.max_model_len() {
                return Err(Refusal::new(
                    "context_exceeded",
                    format!(
                        "question {:?} renders {prompt_tokens} prompt tokens with its forced prefix, past this engine's {} context",
                        question.id,
                        server.engine.max_model_len()
                    ),
                ));
            }
            input.tokens.extend_from_slice(prefix);
            if let Some(multimodal) = &mut input.multimodal {
                // The MRoPE positions have to grow with the tokens or the
                // leaf refuses the chunk outright. A prompt that cannot be
                // extended is one ending in a placeholder, which no template
                // renders — refused rather than sent with positions nobody
                // can check.
                if !std::sync::Arc::make_mut(multimodal).append_text(prefix.len()) {
                    return Err(Refusal::new(
                        "prompt_not_extendable",
                        format!(
                            "question {:?} renders a prompt ending inside an image, which leaves no position for its forced prefix to continue from",
                            question.id
                        ),
                    ));
                }
            }
            match question.head {
                None => input.constrained = Some(std::sync::Arc::new(schedule)),
                // GitHub #260: a head point is the chain's prompt, forced
                // opening and all, read at its last position instead of
                // decoded from: a decision over the pointing head's
                // attention across the image's placeholder span.
                Some(head) => {
                    // An image the head cannot point on is answered here, and
                    // nothing is submitted for it.
                    let (grid, answered) = match head_query(&input, head) {
                        Ok((query, grid)) => {
                            input.decision = Some(ignis_core::DecisionRead::Attention(query));
                            (Some(grid), None)
                        }
                        Err(answer) => (None, Some(answer)),
                    };
                    return Ok(Rendered { input, model, prompt_tokens, media, grid, answered });
                }
            }
        }
    }
    Ok(Rendered { input, model, prompt_tokens, media, grid: None, answered: None })
}

/// The attention readout a head point asks for over `input`'s image, and
/// that image's merged token grid `(rows, cols)` (GitHub #260) — or the
/// answer it gets instead: `state_carries_no_image` when there is no image,
/// the chain's own failure for the same state, and `image_not_a_grid` when
/// the item's placeholders are not one frame's merged grid, row by row, which
/// is the only map the region rule reads.
///
/// The **first** image, as the chain's pixels are the first image's: a
/// `point` answers about the submitted image, and a multi-image state is
/// outside what either method was measured on.
fn head_query(
    input: &ignis_core::types::RequestInput,
    head: ignis_core::pointing::PointingHead,
) -> Result<(ignis_core::pointing::AttentionQuery, (usize, usize)), Answer> {
    let item = input
        .multimodal
        .as_ref()
        .and_then(|multimodal| multimodal.media.first())
        .ok_or_else(no_image)?;
    let merge = ignis_artifact::vision::MERGE as u32;
    let (rows, cols) = ((item.grid.h / merge) as usize, (item.grid.w / merge) as usize);
    let span = (u32::try_from(item.token_span.begin), u32::try_from(item.token_span.count));
    match span {
        (Ok(key_begin), Ok(key_count)) if item.grid.t == 1 && rows * cols == item.token_span.count => {
            Ok((ignis_core::pointing::AttentionQuery { head, key_begin, key_count }, (rows, cols)))
        }
        _ => Err(failed(
            "image_not_a_grid",
            format!(
                "the image's {} placeholders are not one frame's {rows}x{cols} merged grid, so the pointing head's map cannot be read over it",
                item.token_span.count
            ),
        )),
    }
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
    mut ready: Rendered,
    class: ignis_core::types::RequestClass,
) -> Attempt {
    // GitHub #260: a head point over a state with no image fails as the
    // chain's does, and costs nothing — there is no span to read.
    if let Some(answer) = ready.answered.take() {
        return Attempt::Answered(answer);
    }
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
    // GitHub #260: a head point's completion carries the attention readout.
    if let (Some(_), Some(grid)) = (question.head, ready.grid) {
        let pixels = ready.media.and_then(|stats| stats.source_pixels);
        return Attempt::Answered(
            match crate::engine::collect_attention(&mut events, server.request_timeout).await {
                Ok(Some(scores)) => {
                    guard.completed();
                    if let Some(metrics) = &server.metrics {
                        // A `point` like the chain's (spec 13 leaves a
                        // `method` label to ADR 0017), with no answer mass.
                        metrics.record_decision(question.kind.primitive(), None);
                    }
                    match pixels {
                        Some(pixels) => head_answer_for(question, &scores, grid, pixels),
                        None => no_image(),
                    }
                }
                Ok(None) => {
                    guard.completed();
                    failed(
                        "attention_unread",
                        "the engine finished this question without the pointing head's attention over the image: the keys were not where the layer's attention materialized them, or the prefill itself failed".to_owned(),
                    )
                }
                Err(_) => failed(
                    "not_completed",
                    "the engine did not answer this question in time".to_owned(),
                ),
            },
        );
    }
    // GitHub #242: a run's completion carries a trace, not a readout,
    // so `collect_readout` would report every one of them as never
    // completed.
    if question.kind.is_constrained() {
        let pixels = ready.media.and_then(|stats| stats.source_pixels);
        return Attempt::Answered(
            match crate::engine::collect_draws(&mut events, server.request_timeout).await {
                Ok(drawn) => {
                    guard.completed();
                    if let Some(metrics) = &server.metrics {
                        // Counted like any other decision and observing no
                        // answer mass, because it has none: its answer is a
                        // run of sampled tokens, not a restricted softmax
                        // over one position (ADR 0017's row says so).
                        metrics.record_decision(question.kind.primitive(), None);
                    }
                    constrained_answer_for(question, &drawn, pixels)
                }
                Err(_) => failed(
                    "not_completed",
                    "the engine did not answer this question in time".to_owned(),
                ),
            },
        );
    }
    Attempt::Answered(
        match crate::engine::collect_readout(&mut events, server.request_timeout).await {
            Ok(readout) => {
                guard.completed();
                // GitHub #241. Here rather than in `serve` because this is
                // where the `Readout` is, and the answer built from it
                // keeps no trace of how much of the distribution it stood
                // on — a `choice` of 0.03 answer mass and one of 0.999 are
                // the same JSON.
                if let Some(metrics) = &server.metrics {
                    metrics.record_decision(question.kind.primitive(), Some(readout.answer_mass()));
                }
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

/// The answer a spatial question gets over a `state` with no image —
/// whichever method it asked for.
fn no_image() -> Answer {
    failed(
        "state_carries_no_image",
        "a point is a position on an image, and this `state` carried none".to_owned(),
    )
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
