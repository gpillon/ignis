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
//! GitHub #275 adds `locate`: which **segment** of a text `state` — a line,
//! an array element — the instruction names, read from a vote of calibrated
//! attention heads in one prefill and one content-free baseline, with
//! nothing written into the state and nothing generated ([`crate::locate`],
//! `ignis_core::locate`, ADR 0041).
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

mod shortlist;

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

/// Written back **in order** (GitHub #275): a `locate` answers with the
/// segment exactly as the caller sent it, and an element that is an object
/// would otherwise come back with its keys sorted by `serde_json::Map`.
impl Serialize for OrderedValue {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{SerializeMap, SerializeSeq};
        match self {
            Self::Null => serializer.serialize_unit(),
            Self::Bool(flag) => serializer.serialize_bool(*flag),
            Self::Number(number) => number.serialize(serializer),
            Self::String(text) => serializer.serialize_str(text),
            Self::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Self::Object(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (key, value) in entries {
                    map.serialize_entry(key, value)?;
                }
                map.end()
            }
        }
    }
}

/// A JSON string literal, escaped the way `serde_json` escapes one.
pub(crate) fn quoted(text: &str) -> String {
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
    /// A content part may carry `"cache_control": {"type": "ephemeral"}`, a
    /// **reuse marker** (GitHub #270): the server keeps the state up to the
    /// end of that part, so a later request whose state begins the same way
    /// resumes after it instead of prefilling it again. It changes no byte of
    /// the prompt and never what a request may reuse — reuse is matched by
    /// content — only where state is kept. At most four per `state`; one
    /// marker makes the request explicit-only, and the server then keeps no
    /// part end it would otherwise have found repeated across requests.
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
    /// How a `point` or a `box` is answered (GitHub #260, #263): `"head"`
    /// reads the loaded artifact's calibrated heads in one pass — one
    /// prefill, no decode round — and `"chain"` writes the digits one decode
    /// round at a time.
    ///
    /// Absent, a `point` is `head` on a load calibrated for it and `chain`
    /// otherwise, and a `box` is [`BOX_DEFAULT_METHOD`]; every point and box
    /// answer says which ran. A head `point` is read off the **head set**
    /// where the load has one — the object's centre, with its extent — and
    /// off the pointing head alone where it has only that. A head `box` needs
    /// the head set, and is refused without one rather than answered by the
    /// chain the caller did not ask for.
    ///
    /// The chain stays for what it is still better at: a coordinate finer
    /// than one image token (the heads' resolution is one token, 32 px of an
    /// unresized image), and a per-digit trace. Refused on every other type,
    /// and an unknown value is refused naming the two, never read as the
    /// default.
    ///
    /// On a `locate` (GitHub #278, spec 22): `"shortlist"` (the default) — the
    /// calibrated heads narrow the text to a few candidates and a labelled
    /// `choice` decides among them — or `"vote"`, the head vote of GitHub
    /// #275, unchanged.
    #[serde(default)]
    #[schema(value_type = Option<QuestionMethod>)]
    pub method: Option<String>,
    /// The part of the `state` a `locate` searches (GitHub #275): an RFC 6901
    /// JSON Pointer, the whole state when absent. It must name a string —
    /// whose segments are its lines, split on `\n` exactly — or a non-empty
    /// array — whose segments are its elements. The whole state is still
    /// written into the prompt. Refused on every other type.
    #[serde(default)]
    pub within: Option<String>,
    /// Which reading a `locate`'s text gets (GitHub #278, spec 22): `"auto"`
    /// (the default) — `records` for an array of JSON objects, else `log`
    /// when the fold of its first 2,000 segments with content puts at least
    /// half of them in shared templates, `prose` otherwise — or `"log"`,
    /// `"prose"`, `"records"` by name. Refused on every other type.
    #[serde(default, rename = "kind")]
    #[schema(value_type = Option<LocateKindField>)]
    pub text_kind: Option<String>,
    /// What a `locate`'s text is read as (GitHub #278, spec 22):
    /// `"template_fold"` — a log folded into templates and their values
    /// first, no long prefill — or `"none"`. Absent, a log is folded and
    /// prose and records are not. A fold of prose, and a fold under
    /// `"vote"`, are refused. Refused on every other type.
    #[serde(default)]
    #[schema(value_type = Option<LocateCompression>)]
    pub compression: Option<String>,
}

/// The `method` values a question may name: a `point`'s or a `box`'s
/// (`head`, `chain`), or a `locate`'s (`shortlist`, `vote`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum QuestionMethod {
    Head,
    Chain,
    Shortlist,
    Vote,
}

/// A `locate`'s `kind` as a caller may send it (GitHub #278).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum LocateKindField {
    Auto,
    Log,
    Prose,
    Records,
}

/// The reading a `locate`'s text got (GitHub #278): the kind a caller named,
/// or the one `auto` told. `auto` never appears in an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum LocateKind {
    Log,
    Prose,
    Records,
}

/// How a `locate` was answered (GitHub #278).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum LocateMethod {
    /// The heads narrow, a labelled `choice` decides (ADR 0042).
    Shortlist,
    /// The head vote (ADR 0041).
    Vote,
}

/// What a `locate`'s text was read as (GitHub #278).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum LocateCompression {
    TemplateFold,
    None,
}

impl From<ignis_core::locate::Kind> for LocateKind {
    fn from(kind: ignis_core::locate::Kind) -> Self {
        match kind {
            ignis_core::locate::Kind::Log => Self::Log,
            ignis_core::locate::Kind::Prose => Self::Prose,
            ignis_core::locate::Kind::Records => Self::Records,
        }
    }
}

impl From<ignis_core::locate::Method> for LocateMethod {
    fn from(method: ignis_core::locate::Method) -> Self {
        match method {
            ignis_core::locate::Method::Shortlist => Self::Shortlist,
            ignis_core::locate::Method::Vote => Self::Vote,
        }
    }
}

impl From<ignis_core::locate::Compression> for LocateCompression {
    fn from(compression: ignis_core::locate::Compression) -> Self {
        match compression {
            ignis_core::locate::Compression::TemplateFold => Self::TemplateFold,
            ignis_core::locate::Compression::None => Self::None,
        }
    }
}

impl From<LocateKind> for ignis_core::locate::Kind {
    fn from(kind: LocateKind) -> Self {
        match kind {
            LocateKind::Log => Self::Log,
            LocateKind::Prose => Self::Prose,
            LocateKind::Records => Self::Records,
        }
    }
}

impl From<LocateMethod> for ignis_core::locate::Method {
    fn from(method: LocateMethod) -> Self {
        match method {
            LocateMethod::Shortlist => Self::Shortlist,
            LocateMethod::Vote => Self::Vote,
        }
    }
}

impl From<LocateCompression> for ignis_core::locate::Compression {
    fn from(compression: LocateCompression) -> Self {
        match compression {
            LocateCompression::TemplateFold => Self::TemplateFold,
            LocateCompression::None => Self::None,
        }
    }
}

/// What a `locate` asked for (GitHub #278), validated: the kind it named
/// (`None` for `auto`), its method, and the compression it named (`None`
/// for its kind's default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocateAsk {
    pub kind: Option<ignis_core::locate::Kind>,
    pub method: ignis_core::locate::Method,
    pub compression: Option<ignis_core::locate::Compression>,
}

/// How a `point` or a `box` is answered (GitHub #260, #263; specs 13 and 14).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum SpatialMethod {
    /// One pass: the calibrated heads' attention over the image, read at the
    /// forced `{"x":` and turned into an answer on the host — the head set's
    /// **extent** and its centre where the load has a head set, the pointing
    /// head's own point where it has only that. Coarse: one image token.
    Head,
    /// The digit chain: the forced opening and then the digits, one decode
    /// round each, under a constrained decode (GitHub #242).
    Chain,
}

impl SpatialMethod {
    /// The wire's spelling.
    pub fn label(self) -> &'static str {
        match self {
            Self::Head => "head",
            Self::Chain => "chain",
        }
    }
}

/// What a load says, once, about how `/v1/decide` will answer `point` and
/// `box` (GitHub #260, #263): the `ignis.decide.pointing_head` event's
/// fields, decided here so that every branch is testable without a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadMethods {
    /// How a `point` with no `method` is answered: `head_set`, `head`,
    /// `chain`, or `none` on a load that takes no image.
    pub point: &'static str,
    /// The `method`s a `box` accepts: `chain,head`, `chain`, or `none`.
    pub box_methods: &'static str,
    /// How a `box` with no `method` is answered.
    pub box_default: &'static str,
    /// The sentence the event says it with.
    pub summary: &'static str,
}

/// What the `ignis.decide.locate` event says at load (GitHub #275, #278):
/// whether a `locate` will be answered, and why not when it will not.
pub fn locate_summary(calibration: Option<ignis_core::locate::LocateCalibration>) -> &'static str {
    match calibration {
        Some(_) => "`locate` answers by a shortlist: the calibrated end or sum heads narrow the text window by window, a labelled `choice` decides; the head vote stays one field away",
        None => "no calibrated `locate` heads for this artifact: `/v1/decide` refuses a `locate`",
    }
}

/// [`LoadMethods`] for a load with `calibration`, with or without vision.
pub fn load_methods(calibration: Option<ignis_core::pointing::Calibration>, vision: bool) -> LoadMethods {
    let set = calibration.is_some_and(|calibration| calibration.set.is_some());
    let box_default = match set {
        true => BOX_DEFAULT_METHOD.label(),
        false => SpatialMethod::Chain.label(),
    };
    match (vision, calibration, set) {
        (false, _, _) => LoadMethods {
            point: "none",
            box_methods: "none",
            box_default: "none",
            summary: "loaded without --vision: `point` and `box` have no image to answer on",
        },
        (true, Some(_), true) => LoadMethods {
            point: "head_set",
            box_methods: "chain,head",
            box_default,
            summary: "`point` answers in one pass off the calibrated head set, anchored on the pointing head; `box` accepts `head`",
        },
        (true, Some(_), false) => LoadMethods {
            point: "head",
            box_methods: "chain",
            box_default,
            summary: "`point` answers in one pass off the calibrated pointing head; no head set, so `box` answers with the chain",
        },
        (true, None, _) => LoadMethods {
            point: "chain",
            box_methods: "chain",
            box_default,
            summary: "no calibrated pointing head for this artifact: `point` and `box` answer with the digit chain",
        },
    }
}

/// Say how `/v1/decide` will answer on `model`, once per load (the start,
/// and every model switch — spec model-switch/01): the `point` and `box`
/// methods (GitHub #260, #263, `ignis.decide.pointing_head`) and whether a
/// `locate` can be answered, with which heads and window (GitHub #275, #278,
/// `ignis.decide.locate`). The heads are keyed to the artifact's content
/// hash, so a load nobody calibrated says "chain" or "refused" here instead
/// of being found out from its answers.
pub fn log_load_heads(model: &crate::ActiveModel) {
    let methods = load_methods(model.calibration, model.media.is_some());
    tracing::info!(
        name: "ignis.decide.pointing_head",
        point_method = methods.point,
        box_methods = methods.box_methods,
        box_default = methods.box_default,
        head = model.calibration.map(|calibration| calibration.head.to_string()),
        set_heads = model.calibration.and_then(|calibration| calibration.set).map(|set| set.heads.len()),
        artifact = %model.engine.artifact(),
        "{}",
        methods.summary
    );
    let names = |heads: &[ignis_core::pointing::PointingHead]| {
        heads.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
    };
    tracing::info!(
        name: "ignis.decide.locate",
        available = model.locate.is_some(),
        heads = model.locate.map(|calibration| calibration.heads.len()),
        max_keys = model.locate.map(|calibration| calibration.max_keys),
        sum_heads = model.locate.map(|calibration| names(calibration.heads)),
        end_heads = model.locate.map(|calibration| names(calibration.end_heads)),
        window_keys = model.locate.map(|calibration| calibration.window_keys),
        artifact = %model.engine.artifact(),
        "{}",
        locate_summary(model.locate)
    );
}

/// How a `box` with no `method` is answered (GitHub #263).
///
/// Spec 14 § Acceptance 6 decided it by a rule written before the
/// measurement: `head` if and only if the head box's IoU >= 0.5 rate is at
/// least the chain box's on every acceptance set. It was on three of four and
/// not on 1024 px buttons (43% against 79%: a box spanned on 32 px cells
/// frames a target under two tokens tall loosely), so the chain
/// (`docs/findings/2026-09-23-the-head-set-holds-through-decide.md`). `head`
/// stays the better box for anything larger and at 4096 px, for the asking.
pub const BOX_DEFAULT_METHOD: SpatialMethod = SpatialMethod::Chain;

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

/// The eight primitives: Jev's three, which read one position, the four that
/// **generate** (GitHub #242, #255), and `locate`, which reads attention
/// (GitHub #275).
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
    /// The segment of a text `state` the instruction names — a line of a
    /// string or an element of an array, `within` a JSON Pointer — read from
    /// the calibrated heads' attention in one prefill (GitHub #275).
    Locate,
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
            Self::Locate => crate::metrics::Primitive::Locate,
        }
    }

    /// The axis layout this primitive generates under, or `None` for a
    /// readout (GitHub #242).
    fn layout(self) -> Option<crate::numbers::Layout> {
        match self {
            // A scalar generates, but not over axes: its plan is
            // `crate::scalar`'s, so it has no layout here and
            // `is_constrained` cannot be `layout().is_some()` any more.
            Self::Noul | Self::Choice | Self::Score | Self::Scalar | Self::Locate => None,
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
            // Layout L1 (spec 17): a `locate`'s system message is the
            // evidence alone, and its kind text leads the user turn
            // (`crate::locate::user_text`).
            Self::Locate => String::new(),
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
    /// The heads a `point` or a `box` is answered from in one pass (GitHub
    /// #260, #263) — the pointing head, and the head set where the load has
    /// one (always, for a box) — or `None` for a question answered by the
    /// chain and for every other primitive.
    ///
    /// A head question keeps its `plan`, and it is the **point's**: the
    /// prompt is the chain point's byte for byte — same system text, same
    /// forced `{"x":` — so head and chain questions over one image share
    /// their prefix, and the heads' query sits inside the answer's scaffold
    /// where every head number was measured. A head `box` is read in that
    /// same pass. Only the schedule goes unused.
    pub head: Option<ignis_core::pointing::Calibration>,
    /// The JSON Pointer a `locate` searches (GitHub #275) — empty for the
    /// whole state — and `None` for every other primitive.
    pub within: Option<String>,
    /// A `locate`'s kind, method and compression (GitHub #278), and `None`
    /// for every other primitive.
    pub locate: Option<LocateAsk>,
}

impl PreparedQuestion {
    /// How this question is answered, if it is a `point` or a `box`.
    pub fn method(&self) -> Option<SpatialMethod> {
        self.kind.is_spatial().then_some(match self.head {
            Some(_) => SpatialMethod::Head,
            None => SpatialMethod::Chain,
        })
    }

    /// The primitive whose prompt this question is put under: its own,
    /// except a head `box` (GitHub #263), which is read in a point's pass —
    /// the prompt the head set was chosen and measured on.
    pub fn prompt_kind(&self) -> QuestionKind {
        match (self.kind, self.head) {
            (QuestionKind::Box, Some(_)) => QuestionKind::Point,
            (kind, _) => kind,
        }
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
/// `pointing` is the load's calibration (GitHub #260, #263) — its pointing
/// head, and its head set when it has one — `None` on a load nobody
/// calibrated: it is what a `point` with no `method` is answered from, and
/// what a `head` question is refused without.
pub fn prepare(
    questions: &Ordered<Question>,
    alphabet: &AnswerAlphabet,
    encode: Encoder<'_>,
    pointing: Option<ignis_core::pointing::Calibration>,
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
    pointing: Option<ignis_core::pointing::Calibration>,
) -> Result<PreparedQuestion, Refusal> {
    if instructions_are_empty(&question.instructions) {
        return Err(Refusal::new(
            "empty_instructions",
            format!("question {id:?} has empty `instructions`"),
        ));
    }
    if question.kind == QuestionKind::Locate {
        return prepare_locate(id, question);
    }
    let head = head_reading(id, question, pointing)?;
    let primitive = question.kind.primitive().label();
    if question.within.is_some() {
        return Err(Refusal::new(
            "within_unsupported",
            format!("question {id:?} is a {primitive}: `within` names the part of the state a `locate` searches"),
        ));
    }
    // GitHub #278: a field a caller wrote is never ignored.
    if question.text_kind.is_some() {
        return Err(Refusal::new(
            "kind_unsupported",
            format!("question {id:?} is a {primitive}: `kind` names the reading a `locate`'s text gets"),
        ));
    }
    if question.compression.is_some() {
        return Err(Refusal::new(
            "compression_unsupported",
            format!("question {id:?} is a {primitive}: `compression` names what a `locate`'s text is read as"),
        ));
    }
    if question.kind == QuestionKind::Scalar {
        return prepare_scalar(id, question, encode);
    }
    if let Some(layout) = question.kind.layout() {
        // GitHub #263: a head box is planned as the point it is read in.
        let layout = match head {
            Some(_) => crate::numbers::POINT_LAYOUT,
            None => layout,
        };
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
        // Unreachable: `prepare_one` returns above for `locate`, the scalar
        // and every kind with a layout, which is exactly these five.
        QuestionKind::Scalar
        | QuestionKind::Number
        | QuestionKind::Point
        | QuestionKind::Box
        | QuestionKind::Locate => Vec::new(),
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
        within: None,
        locate: None,
    })
}

/// Validate a `locate` question (GitHub #275): it declares no options and
/// generates no digits, so `criteria` and `digits` are refused rather than
/// ignored. Where it searches, and whether this load can answer it at all,
/// are the state's and the load's to say, before any prefill ([`serve`]).
///
/// GitHub #278 (spec 22): its `method`, `kind` and `compression`, each an
/// enum whose unknown value is refused naming the accepted ones, and the two
/// combinations that never apply — a fold under `vote` (measured 28 of R2's
/// 58) and a fold of prose (prose does not fold) — refused by name. Whether
/// a kind contradicts the state is the state's to say ([`serve`]).
fn prepare_locate(id: &str, question: &Question) -> Result<PreparedQuestion, Refusal> {
    use ignis_core::locate::{Compression, Kind, Method};
    if question.criteria.is_some() {
        return Err(Refusal::new(
            "criteria_unsupported",
            format!("question {id:?} is a locate: its options are the state's own segments, so `criteria` cannot be honoured"),
        ));
    }
    if question.digits.is_some() {
        return Err(Refusal::new(
            "digits_unsupported",
            format!("question {id:?} is a locate and generates nothing, so `digits` cannot be honoured"),
        ));
    }
    let method = match question.method.as_deref() {
        None | Some("shortlist") => Method::Shortlist,
        Some("vote") => Method::Vote,
        Some(other) => {
            return Err(Refusal::new(
                "method_unknown",
                format!("question {id:?} asks for method {other:?}; a locate's accepted values are \"shortlist\" and \"vote\""),
            ));
        }
    };
    let kind = match question.text_kind.as_deref() {
        None | Some("auto") => None,
        Some("log") => Some(Kind::Log),
        Some("prose") => Some(Kind::Prose),
        Some("records") => Some(Kind::Records),
        Some(other) => {
            return Err(Refusal::new(
                "kind_unknown",
                format!("question {id:?} asks for kind {other:?}; the accepted values are \"auto\", \"log\", \"prose\" and \"records\""),
            ));
        }
    };
    let compression = match question.compression.as_deref() {
        None => None,
        Some("template_fold") => Some(Compression::TemplateFold),
        Some("none") => Some(Compression::None),
        Some(other) => {
            return Err(Refusal::new(
                "compression_unknown",
                format!("question {id:?} asks for compression {other:?}; the accepted values are \"template_fold\" and \"none\""),
            ));
        }
    };
    if compression == Some(Compression::TemplateFold) {
        if method == Method::Vote {
            return Err(Refusal::new(
                "compression_unsupported",
                format!("question {id:?}: the vote reads the text as it is — over a fold it read 28 of 58 real-log questions — so \"template_fold\" is refused with \"vote\"; ask for \"shortlist\", or for \"none\""),
            ));
        }
        if kind == Some(Kind::Prose) {
            return Err(Refusal::new(
                "compression_unsupported",
                format!("question {id:?}: prose does not fold into templates, so \"template_fold\" is refused with \"prose\"; ask for \"none\", or omit `compression`"),
            ));
        }
    }
    Ok(PreparedQuestion {
        id: id.to_owned(),
        kind: question.kind,
        instructions: question.instructions.clone(),
        options: Vec::new(),
        answers: Vec::new(),
        plan: None,
        scalar: None,
        digits: 0,
        head: None,
        within: Some(question.within.clone().unwrap_or_default()),
        locate: Some(LocateAsk { kind, method, compression }),
    })
}

/// Resolve a question's `method` (GitHub #260, #263): the calibration a
/// `point` or a `box` is answered from, `None` for a chain answer and for
/// every other primitive — or a refusal, before any prefill.
///
/// Omitted, the measured-better method runs: for a `point`, the heads when
/// the load has a calibration and the chain otherwise; for a `box`,
/// [`BOX_DEFAULT_METHOD`] where the load has a head set. Asked for by name,
/// `head` is refused on a load that cannot answer it — a `point` needs a
/// pointing head, a `box` a head set — rather than silently served by the
/// chain the caller explicitly did not ask for.
fn head_reading(
    id: &str,
    question: &Question,
    pointing: Option<ignis_core::pointing::Calibration>,
) -> Result<Option<ignis_core::pointing::Calibration>, Refusal> {
    let with_set = pointing.filter(|calibration| calibration.set.is_some());
    let Some(method) = question.method.as_deref() else {
        return Ok(match question.kind {
            QuestionKind::Point => pointing,
            QuestionKind::Box => with_set.filter(|_| BOX_DEFAULT_METHOD == SpatialMethod::Head),
            _ => None,
        });
    };
    if !question.kind.is_spatial() {
        return Err(Refusal::new(
            "method_unsupported",
            format!(
                "question {id:?} is a {}: `method` chooses how a `point`, a `box` or a `locate` is answered, and this primitive has one way",
                question.kind.primitive().label()
            ),
        ));
    }
    match method {
        "chain" => Ok(None),
        "head" if question.kind == QuestionKind::Box => with_set.map(Some).ok_or_else(|| {
            Refusal::new(
                "pointing_head_unavailable",
                format!(
                    "question {id:?} asks for a box with `\"method\": \"head\"`, and the loaded artifact has no calibrated head set to read a box off; ask for \"chain\", or omit `method` to be answered by the chain"
                ),
            )
        }),
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
        within: None,
        locate: None,
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
        within: None,
        locate: None,
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
/// **An image `state` does not follow it.** The parts stay in the user turn —
/// image first, then the decision's JSON without an `evidence` field, the
/// shape `classify_vision_readout_gpu.rs` measured. What shares such a state
/// is not its place in the prompt but the **reuse boundaries** GitHub #270
/// cuts inside the user turn without moving a byte: a fan-out's head, an
/// observed fork, a caller's reuse marker (`place_reuse_boundaries`, spec
/// 16). They are cut from where each part ends, and each floors to a whole
/// KV page walked out of any image (GitHub #193), so a head reaches past an
/// image only when the text shared after it crosses the next page boundary.
///
/// Why the parts are not moved into the system block instead, because an
/// earlier version of this comment got it wrong:
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
    let instruction = question.prompt_kind().system_text(question.digits);
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
                cache_control: None,
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
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
    /// `method` says which answered it (GitHub #260, #263), and the rest
    /// follows from it:
    ///
    /// - **`head`** — one pass over the calibrated heads' attention.
    ///   `uncertainty` is the maps' resolution and not a spread: **half an
    ///   image token** per axis on a load with a head set, whose reading
    ///   resolves a peak inside its token (GitHub #264), and one whole token
    ///   for the pointing head alone. `region` is the pointing head's: the cells its point
    ///   was read from and their `share` of its attention over the image —
    ///   the confidence to act on (it separates hits from misses on the
    ///   measured scenes, and a near-flat map, about 0.03, is nothing found),
    ///   not a calibrated probability. On a load with a **head set** the
    ///   point is the centre of the object's **`extent`** — the box the set's
    ///   heads outline around the pointing head's point — and `extent` is
    ///   that box, `x0`, `y0`, `x1`, `y1` in pixels of the submitted image.
    ///   On a load with the pointing head alone there is no `extent`, and the
    ///   point sits on the part of the object that head reads — where a
    ///   button's label begins, a large object's bottom-right corner. No
    ///   `digits`.
    /// - **`chain`** — the digit chain: `uncertainty` from the digits'
    ///   distributions, and `digits` the trace it came from.
    Point {
        method: SpatialMethod,
        pixels: BTreeMap<String, i64>,
        normalized: BTreeMap<String, u64>,
        uncertainty: BTreeMap<String, f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        region: Option<HeadRegion>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extent: Option<BTreeMap<String, i64>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        digits: Option<BTreeMap<String, Vec<crate::numbers::DigitDraw>>>,
    },
    /// A bounding box on the submitted image: [`Answer::Point`]'s shape over
    /// `x0`, `y0`, `x1`, `y1`, with the `method` that produced it (GitHub
    /// #263).
    ///
    /// - **`head`** — the head set's **extent**, read in one pass, in the
    ///   same pixels and on the same 0-`scale` normalization as the chain's.
    ///   `uncertainty` is half an image token per edge (half a token's width
    ///   for `x0` and `x1`, half its height for `y0` and `y1`) — the
    ///   reading's resolution inside the token (GitHub #264) — `region` the
    ///   pointing head's, and no `digits`.
    /// - **`chain`** — the digit chain, with its `digits` trace.
    #[serde(rename = "box")]
    Box {
        method: SpatialMethod,
        pixels: BTreeMap<String, i64>,
        normalized: BTreeMap<String, u64>,
        uncertainty: BTreeMap<String, f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        region: Option<HeadRegion>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        digits: Option<BTreeMap<String, Vec<crate::numbers::DigitDraw>>>,
    },
    /// The segment of the `state` the instruction names (GitHub #275, #278):
    /// its index — the 0-based line of `split("\n")` for a string, the
    /// element's index for an array, always into the state as sent, whatever
    /// was folded or windowed — and `value`, the segment exactly as the
    /// caller sent it. `kind`, `method` and `compression` name what produced
    /// it, defaults included.
    ///
    /// **`shortlist`** (the default, ADR 0042): the calibrated heads narrow
    /// the text to a few candidates and a labelled `choice` decides among
    /// them. `confidence` is that `choice`'s probability of the pick — times
    /// the pick's probability at a fold's first level — and `ranking` its
    /// candidates by probability, at most five, on the same scale, the pick
    /// first. `pointers` is every candidate whose share is at least 0.05,
    /// best first, the pick always among them: one answer or several.
    /// `found` (on `log` + `template_fold`, `prose` + `none` and `records` +
    /// `none`) says whether the text answers at all: below 0.5 `segment`,
    /// `value` and `confidence` are `null`, `pointers` is empty, and
    /// `ranking` still lists the candidates. Neither is a calibrated
    /// probability.
    ///
    /// **`vote`** (ADR 0041): read in one prefill from a vote of the
    /// calibrated heads; `confidence` is the winner's share of the votes and
    /// `ranking` the voted segments by votes, at most five — how much the
    /// heads agree (`docs/findings/2026-09-27-locate-through-decide.md`).
    /// No `found`; `pointers` holds the winner alone.
    Locate {
        kind: LocateKind,
        method: LocateMethod,
        compression: LocateCompression,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        found: Option<f64>,
        segment: Option<usize>,
        #[schema(value_type = serde_json::Value)]
        value: Option<OrderedValue>,
        confidence: Option<f64>,
        ranking: Vec<LocateRank>,
        pointers: Vec<LocatePointer>,
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

/// One segment of a `locate`'s ranking (GitHub #275): its index and its share
/// — of the heads' votes, or of the labelled `choice` (GitHub #278).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
pub struct LocateRank {
    pub segment: usize,
    pub share: f64,
}

/// A segment a `locate` points at (GitHub #278): its index, its value as the
/// caller sent it, and its share.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
pub struct LocatePointer {
    pub segment: usize,
    #[schema(value_type = serde_json::Value)]
    pub value: OrderedValue,
    pub share: f64,
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

/// Shape a head question's **attention readout** into its answer (GitHub
/// #260, #263).
///
/// `grid` is the image's merged token grid, `(rows, cols)`; `pixels` the
/// submitted image's `(width, height)`. A point or a box maps to each through
/// its own side on each axis, which is the processor's own scale: it resizes
/// the whole image onto the grid.
///
/// With the head set's argmax beside the pointing head's scores, a `point`
/// is the anchored reading's centre with its `extent`, and a `box` is the
/// extent ([`ignis_core::pointing::read_anchored`]); without it, a `point` is
/// the pointing head's own reading (spec 13). A question that was asked with
/// a set and came back without one — or a box with none — is malformed, not
/// answered by half the reading.
pub fn head_answer_for(
    question: &PreparedQuestion,
    attention: &ignis_core::pointing::AttentionScores,
    grid: (usize, usize),
    pixels: (u32, u32),
) -> Answer {
    let scores = &attention.scores;
    let malformed = || {
        failed(
            "attention_malformed",
            format!(
                "the heads' readout carried {} scores for a {}x{} image grid, a score that is not finite, or a head set that is not the one asked for",
                scores.len(),
                grid.0,
                grid.1
            ),
        )
    };
    let Some(reading) = ignis_core::pointing::read_head_map(scores, grid.0, grid.1) else {
        return malformed();
    };
    let (width, height) = pixels;
    let (cell_w, cell_h) = reading.cell_pixels(width, height);
    let scale = crate::numbers::scale(question.digits);
    let region = Some(HeadRegion {
        cells: u32::try_from(reading.cells).unwrap_or(u32::MAX),
        share: reading.share,
    });
    let asked_set = question.head.is_some_and(|calibration| calibration.set.is_some());
    let set = (
        attention.set_argmax.as_deref(),
        attention.set_peak.as_deref(),
        attention.set_neighbours.as_deref(),
    );
    let anchored = match (set, asked_set) {
        ((Some(argmax), Some(peak), Some(around)), true) => {
            match ignis_core::pointing::read_anchored(
                scores, argmax, peak, around, grid.0, grid.1, width, height,
            ) {
                Some(anchored) => Some(anchored),
                None => return malformed(),
            }
        }
        ((None, None, None), false) => None,
        _ => return malformed(),
    };
    let on_scale = |value: f64, side: u32| {
        ((value / f64::from(side)) * scale as f64).round().clamp(0.0, scale as f64) as u64
    };
    let in_pixels = |value: f64, side: u32| value.round().clamp(0.0, f64::from(side)) as i64;
    fn axes<T>(pairs: [(&str, T); 2]) -> BTreeMap<String, T> {
        pairs.into_iter().map(|(axis, value)| (axis.to_owned(), value)).collect()
    }
    fn corners<T>(values: [T; 4]) -> BTreeMap<String, T> {
        ["x0", "y0", "x1", "y1"].into_iter().map(str::to_owned).zip(values).collect()
    }
    // Spec 15: an anchored reading resolves a peak *inside* its cell, so it
    // answers to half a cell; the pointing head alone still answers to one.
    let (sub_w, sub_h) = (cell_w / 2.0, cell_h / 2.0);
    match (question.kind, anchored) {
        (QuestionKind::Box, Some(anchored)) => {
            let e = anchored.extent;
            Answer::Box {
                method: SpatialMethod::Head,
                pixels: corners([
                    in_pixels(e.x0, width),
                    in_pixels(e.y0, height),
                    in_pixels(e.x1, width),
                    in_pixels(e.y1, height),
                ]),
                normalized: corners([
                    on_scale(e.x0, width),
                    on_scale(e.y0, height),
                    on_scale(e.x1, width),
                    on_scale(e.y1, height),
                ]),
                uncertainty: corners([sub_w, sub_h, sub_w, sub_h]),
                region,
                digits: None,
            }
        }
        (QuestionKind::Box, None) => malformed(),
        (_, Some(anchored)) => {
            let e = anchored.extent;
            let (x, y) = anchored.point();
            Answer::Point {
                method: SpatialMethod::Head,
                pixels: axes([("x", in_pixels(x, width)), ("y", in_pixels(y, height))]),
                normalized: axes([("x", on_scale(x, width)), ("y", on_scale(y, height))]),
                uncertainty: axes([("x", sub_w), ("y", sub_h)]),
                region,
                extent: Some(corners([
                    in_pixels(e.x0, width),
                    in_pixels(e.y0, height),
                    in_pixels(e.x1, width),
                    in_pixels(e.y1, height),
                ])),
                digits: None,
            }
        }
        (_, None) => {
            let (x, y) = reading.pixels(width, height);
            let (nx, ny) = reading.normalized(scale);
            Answer::Point {
                method: SpatialMethod::Head,
                pixels: axes([("x", x.round() as i64), ("y", y.round() as i64)]),
                normalized: axes([("x", nx), ("y", ny)]),
                uncertainty: axes([("x", cell_w), ("y", cell_h)]),
                region,
                extent: None,
                digits: None,
            }
        }
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
            method: SpatialMethod::Chain,
            pixels: in_pixels,
            normalized,
            uncertainty,
            region: None,
            digits: Some(digits),
        },
        _ => Answer::Point {
            method: SpatialMethod::Chain,
            pixels: in_pixels,
            normalized,
            uncertainty,
            region: None,
            extent: None,
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
        // A `locate` is read off attention (`locate_answer_for`); reaching
        // here is a wrong route, answered rather than panicked on.
        QuestionKind::Locate => failed(
            "not_a_readout",
            "this question reads attention, not an answer position".to_owned(),
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

Eight primitives. Read at one position: `noul` (yes/no, answered with the probability of yes), `choice` (one option from a declared set, with the distribution over all of them) and `score` (a probability-weighted value across ordered levels, which can land between them). Generated a digit at a time under a constrained decode: `number`, `point` and `box` -- the last two in the submitted image's own pixels -- and `scalar`, which closes its own object as soon as the number is complete, so `digits` is a ceiling the caller can leave out and the answer may have a decimal part. Read from attention: `locate`.

A `locate` names the **segment** of a JSON `state` the instruction asks for -- a line of a string, split on `\\n`, or an element of an array, `within` an optional JSON Pointer -- without writing anything into the state and without generating. By default (`method: shortlist`, ADR 0042) the loaded artifact's calibrated attention heads narrow the text to a few candidates and a labelled `choice` decides among them, and the text may be far longer than the context: prose and record arrays are read in windows of at most 200,000 keys, each with a content-free baseline. `kind` names the reading -- `auto` (default: `records` for an array of JSON objects, else `log` when the fold of its first 2,000 segments with content puts at least half of them in shared templates, else `prose`), `log`, `prose` or `records` -- and `compression` whether a log is folded into templates first (`template_fold`, the default for `log`: no long prefill) or read as it is (`none`, the default for prose and records). `method: vote` is the head vote of ADR 0041, unchanged, over at most the span it was measured on (`locate_too_long`). The answer names the resolved `kind`, `method` and `compression`; the segment's index into the state as sent, its `value`, a `confidence` and a `ranking` (the `choice`'s probabilities -- not calibrated), and `pointers`: every candidate at a share of 0.05 or more, the pick first. `found` (on `log` + `template_fold`, `prose` + `none`, `records` + `none`) says whether the text answers at all: below 0.5 the answer names no segment and keeps its ranking. Every prefill -- windows, their content-free baselines, a fold's levels, the `choice`s -- is counted in `usage.input_tokens`.

A `point` is answered **in one pass** by default (ADR 0038, ADR 0039): the prefill that forces its `{\"x\":` reads the loaded artifact's calibrated heads over the image and the server turns their maps into an answer, with no decode round (`method: head`). The pointing head names which object -- its `region.share` is the confidence to act on, and a near-flat map (about 0.03) is nothing found -- and the head set outlines it: the point is the centre of the object's `extent`, which the answer carries as `x0`, `y0`, `x1`, `y1` in pixels of the submitted image. Its `uncertainty` is one image token per axis. A load calibrated with the pointing head alone answers without `extent`, at the part of the object that head reads. `\"method\": \"chain\"` asks for the digit chain instead -- finer than one token, with a per-digit trace -- and a load with no calibrated head answers every `point` by chain.

A `box` answers with the digit chain unless it asks for `\"method\": \"head\"`: the same extent, read in the same one pass, in the same pixels and 0-999 scale as the chain's, with one image token of `uncertainty` per edge and no `digits`. A load without a head set refuses a head `box`. Every point and box answer names its `method`.

Every fault a caller can commit refuses the whole request with a 422 before the first submit: a caller never pays a prefill for nineteen good questions and a refusal on the twentieth. Only an engine fault lands per-answer, as an `error` answer beside its siblings.

Thinking is refused rather than ignored: a decision's prompt ends exactly where its answer is read, and a thinking prompt would put an open reasoning block at that position.

`model` is read as on chat completions: another model the server lists (`--known-model`) switches the server to it before the decision is evaluated, unless `--allow-model-switch false` -- so naming the 27B on a Qwen3.8-Flash-Next load, which serves no `/v1/decide`, moves the server back to the model that does. During the switch every other request is refused `503 model_switching`; a switch that does not land refuses the decision `model_switch_failed`.

`POST /v1/systemone` is the same handler under Jev's name.",
    request_body = DecideRequest,
    responses(
        (status = 200, description = "One answer per question, under the ids the caller chose.", body = DecideResponse),
        (status = 400, description = "The loaded model serves no `/v1/decide` (`model_unsupported`): Qwen3.8-Flash-Next has no readouts.", body = crate::api::ApiError),
        (status = 401, description = "The server was started with `--api-key` and the request carried no matching bearer token.", body = crate::api::ApiError),
        (status = 422, description = "The body does not parse, or a question is malformed, or the request asked for something this endpoint cannot honour (thinking, an unnameable option, an image on a text-only load, a `method` on anything but a `point` or a `box`, `head` on a load with no calibrated pointing head, or a head `box` on a load with no head set), or a `state` part's reuse marker is not exactly `{\"type\": \"ephemeral\"}` (`malformed_reuse_marker`: retention is by eviction, never by time, so a `ttl` is refused) or there are more than four of them (`too_many_reuse_markers`), or a `locate` cannot be served: the load has no calibrated heads for it (`locate_uncalibrated`), the `state` is content parts (`locate_needs_json_state`), `within` is not a pointer, names nothing, or names a key written twice (`locate_within_malformed`, `locate_within_not_found`, `locate_within_ambiguous`), the target is not a string or a non-empty array (`locate_target_unsegmentable`), fewer than two of its segments own a token (`locate_too_few_segments`), a vote's target is longer than the vote was measured on (`locate_too_long`), a single segment is longer than a window (`locate_segment_too_long`), or the loaded template cannot say where its tokens sit (`locate_unsupported`); a `kind`, `method` or `compression` a locate does not know (`kind_unknown`, `method_unknown`, `compression_unknown`), a fold under `vote` or of prose (`compression_unsupported`), `kind: records` on a state that is not an array of JSON objects or `log`/`prose` on one (`kind_mismatch`); `criteria` and `digits` on a `locate`, and `within`, `kind` and `compression` on anything else (`kind_unsupported`, `compression_unsupported`), are refused too, as is a fold's level-1 text past the context (`context_exceeded`). Nothing reached the engine. (A shortlist step rendered from an earlier step's answer -- a fold's level 2, every `choice` -- can only fault after a prefill, and then answers its own question with an `error`.) A `model` the server neither loads nor may switch to is refused `model_not_found`; one it began switching to that did not load, `model_switch_failed`.",
            body = crate::api::ApiError),
        (status = 503, description = "The engine is at capacity and the request was not admitted; or a model switch is under way (`model_switching`, with `Retry-After`).", body = crate::api::ApiError),
    ),
)]
pub async fn decide(
    axum::extract::State(server): axum::extract::State<std::sync::Arc<crate::Server>>,
    body: Result<axum::Json<DecideRequest>, axum::extract::rejection::JsonRejection>,
) -> axum::response::Response {
    // A `model` naming another known model switches to it first (spec
    // model-switch/01 §Implicit switch) — ahead of the Flash-Next refusal
    // below, so a decision naming the 27B on a Flash-Next load moves the
    // server to the model that serves it. A model it may not switch to meets
    // the `model_not_found` refusal in `serve`, as before.
    let named = body.as_ref().ok().and_then(|request| request.0.model.clone());
    match crate::model_switch::implicit_switch(&server, crate::api::split_model_lane(named).0.as_deref()).await {
        Ok(()) => {}
        Err(refusal @ crate::model_switch::ImplicitRefusal::Switching(_)) => {
            return crate::api::implicit_switch_refused(refusal);
        }
        Err(crate::model_switch::ImplicitRefusal::Failed { to, reason }) => {
            return refused(&Refusal::new(
                "model_switch_failed",
                format!("`model` names {to:?}, and switching to it failed: {reason}"),
            ));
        }
    }
    // One model for the whole request, every round of it (spec
    // model-switch/01).
    let server = std::sync::Arc::new(server.pinned());
    // Flash-Next has no readouts (spec flash-next/04): the endpoint is not
    // served at all, whatever the body says, and the 400 names the model.
    if !server.active().family.serves_readouts() {
        return crate::api::error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "model_unsupported",
            format!("/v1/decide is not served by {}: it has no readouts", server.active().family.name()),
        );
    }
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
    let encode = |text: &str| server.active().template.encode_literal(text);
    let prepared = prepare(&request.questions, &server.active().alphabet, &encode, server.active().calibration)?;
    let evidence = Evidence::read(&request.state);
    let markers = reuse_markers(&evidence)?;
    // An image `state` on a load that cannot take images is a refusal, not
    // an error in an answer slot: the request was never servable, and it is
    // the caller's to fix.
    if evidence.has_media() && server.active().media.is_none() {
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
    let loaded = server.active().engine.model_id();
    if let Some(named) = model.as_deref().filter(|named| !named.is_empty() && *named != loaded) {
        return Err(Refusal::new(
            "model_not_found",
            format!("`model` names {named:?}, and this engine serves {loaded:?}"),
        ));
    }

    // GitHub #275: a `locate` reads the segments of a JSON state, and only on
    // a load calibrated for it. Both are the request's to know before any
    // prefill, whatever else it asks.
    let locates = prepared.iter().any(|question| question.kind == QuestionKind::Locate);
    let locate = match (locates, server.active().locate) {
        (false, _) => None,
        (true, None) => {
            return Err(Refusal::new(
                "locate_uncalibrated",
                format!(
                    "the loaded artifact ({}) has no calibrated `locate` heads, and a `locate` is never answered by heads chosen for another model",
                    server.active().engine.artifact()
                ),
            ));
        }
        (true, Some(calibration)) => Some(calibration),
    };
    if locates && matches!(evidence, Evidence::Parts(_)) {
        return Err(Refusal::new(
            "locate_needs_json_state",
            "a `locate` reads the lines of a string or the elements of an array, and this `state` is content parts; send the text as a JSON string or array",
        ));
    }

    // GitHub #278: each `locate`'s target, the kind it resolves to and the
    // compression that kind gets — and a kind that contradicts the state
    // refused — before anything is rendered.
    let mut resolved_kinds: Vec<Option<(ignis_core::locate::Kind, ignis_core::locate::Compression)>> =
        Vec::with_capacity(prepared.len());
    // `auto` is told once per target (spec 22 § `auto`), and its host time
    // goes on the request log beside the question that paid it.
    let mut told: BTreeMap<String, ignis_core::locate::Kind> = BTreeMap::new();
    let mut auto_ms: Vec<Option<f64>> = Vec::with_capacity(prepared.len());
    for question in &prepared {
        let (resolved, spent) = match (&evidence, question.locate) {
            (Evidence::Json(state), Some(ask)) => {
                let (resolved, spent) = resolve_locate(state, question, ask, &mut told)?;
                (Some(resolved), spent)
            }
            _ => (None, None),
        };
        resolved_kinds.push(resolved);
        auto_ms.push(spent);
    }

    // Render every question first. This is the last thing that can refuse
    // the whole request, and it is all CPU: the chat template, the
    // instruction policy and — for an image — the media acquisition and
    // the placeholder expansion.
    //
    // GitHub #278: a `shortlist` locate is planned here — its target, its
    // fold, every reading it is asked before it has read anything — and
    // answered after the others ([`shortlist::answer_all`]): each of its
    // later steps is rendered from an earlier one's answer.
    let mut main: Vec<(usize, Rendered)> = Vec::with_capacity(prepared.len());
    let mut plans: Vec<Option<LocatePlan>> = Vec::with_capacity(prepared.len());
    let mut shortlists: Vec<shortlist::Planned> = Vec::new();
    let mut shared = shortlist::Shared::default();
    // GitHub #275: one content-free baseline per `locate` target, shared by
    // every `locate` over it — rendered with the questions, asked after them.
    let mut baselines: Vec<Baseline> = Vec::new();
    // GitHub #270: a parts `state`'s reuse boundaries are cut where its parts
    // end, and each end costs a tokenization of everything before it. Only
    // the first question under each system text needs them: the rest share
    // its prompt up to the end of the state, so they claim the fan-out's head,
    // which covers every one of those ends, and their run ends are its keys.
    let mut systems = std::collections::BTreeSet::new();
    for (slot, question) in prepared.iter().enumerate() {
        match (locate, &evidence, resolved_kinds[slot], question.locate) {
            (Some(calibration), Evidence::Json(state), Some((kind, compression)), Some(ask)) => match ask.method {
                ignis_core::locate::Method::Vote => {
                    let (ready, plan) =
                        render_locate(server, state, question, model.clone(), calibration, kind, &mut baselines).await?;
                    main.push((slot, ready));
                    plans.push(Some(plan));
                }
                ignis_core::locate::Method::Shortlist => {
                    let planned = shortlist::plan(
                        server,
                        state,
                        question,
                        slot,
                        kind,
                        compression,
                        calibration,
                        model.clone(),
                        &mut shared,
                    )
                    .await?;
                    shortlists.push(planned);
                    plans.push(None);
                }
            },
            _ => {
                let first_of_kind = systems.insert(question.prompt_kind().system_text(question.digits));
                let part_ends = first_of_kind && matches!(evidence, Evidence::Parts(_));
                main.push((slot, render(server, &evidence, question, model.clone(), part_ends).await?));
                plans.push(None);
            }
        }
    }
    let asked_questions = prepared.len();
    main.extend(baselines.into_iter().enumerate().map(|(b, baseline)| (asked_questions + b, baseline.ready)));
    let (main_slots, mut main_rendered): (Vec<usize>, Vec<Rendered>) = main.into_iter().unzip();
    // GitHub #270: where each question's state is kept for a later one. Held
    // until the handler is done, answered or dropped, so a fan-out's head
    // never outlives the fan-out.
    let _fan_out = place_reuse_boundaries(server, &evidence, &markers, &mut main_rendered);

    let mut input_tokens = main_rendered
        .iter()
        .fold(0u32, |total, ready| total.saturating_add(ready.prompt_tokens));
    let resolved = main_rendered.first().map(|ready| ready.model.clone());

    // The fan-out (GitHub #240). **One question goes first, alone**, and the
    // rest go together once it is answered.
    //
    // The sequencing is not politeness, it is the whole saving. Handed N
    // questions at once the scheduler sees N requests with no published
    // prefix between them, and every one of them prefills the whole state —
    // for an image, N x 16K tokens. The first question prefills it once and
    // leaves a **retained prefix** behind (`messages_for` says why it is
    // retained and not shared) — the system block for a JSON state, the
    // fan-out's head for any state (GitHub #270) — and the followers claim
    // it and prefill only their own tail.
    //
    // After that there is nothing left to serialize, so the followers run
    // together, [`FAN_OUT_WIDTH`] of them at a time. A `locate`'s baseline
    // (GitHub #275) is one more follower: it claims the state like any
    // sibling, and is answered with the questions that read it.
    //
    // A `locate` keeps its state under another layout (L1, spec 17) than
    // the other kinds, so one behind a `noul` would find nothing to claim.
    // The first `locate` therefore goes alone too, right after the first
    // question, and leads its own kind (GitHub #275).
    let collect = |slot: usize| match prepared.get(slot) {
        Some(question) if question.kind != QuestionKind::Locate => Collect::Answer(question),
        _ => Collect::Rows,
    };
    let first_asked = main_slots.first().copied();
    let first_vote = main_slots
        .iter()
        .copied()
        .find(|&slot| prepared.get(slot).is_some_and(|question| question.kind == QuestionKind::Locate));
    let leads = |slot: usize| Some(slot) == first_asked || Some(slot) == first_vote;
    let mut replies: Vec<Option<Reply>> = (0..asked_questions + main_slots.len()).map(|_| None).collect();
    let (leaders, followers): (Vec<_>, Vec<_>) =
        main_slots.into_iter().zip(main_rendered).partition(|(slot, _)| leads(*slot));
    for (slot, ready) in leaders {
        // A leader the engine has no room for leaves the fan-out with no
        // prefix to share, but it is still one question's failure and not
        // the request's — the same slot an engine-full follower gets, and
        // the same one the sequential loop before #240 gave it.
        replies[slot] = Some(alone(server, collect(slot), ready, class).await);
    }
    // The followers, in waves ([`in_waves`]).
    let followers = followers.into_iter().map(|(slot, ready)| (slot, ready, collect(slot))).collect();
    for (slot, reply) in in_waves(server, followers, class).await {
        replies[slot] = Some(reply);
    }

    // GitHub #278: every shortlist, step by step.
    let mut shortlisted: BTreeMap<usize, (Answer, shortlist::Summary)> = BTreeMap::new();
    let resolved = match (resolved, shortlists.is_empty()) {
        (Some(resolved), _) => Some(resolved),
        (None, false) => Some(model.clone().filter(|named| !named.is_empty()).unwrap_or_else(|| server.active().engine.model_id())),
        (None, true) => None,
    };
    if let (Some(calibration), false) = (locate, shortlists.is_empty()) {
        let (answered, spent) = shortlist::answer_all(server, shortlists, shared, calibration, model.clone(), class).await;
        input_tokens = input_tokens.saturating_add(spent);
        for (slot, answer, summary) in answered {
            shortlisted.insert(slot, (answer, summary));
        }
    }

    // Every question's answer: its own reply, or — for a `locate` — its rows
    // read against its baseline's (GitHub #275), or its shortlist's (#278).
    let mut answers = BTreeMap::new();
    for (slot, question) in prepared.iter().enumerate() {
        if let Some((answer, summary)) = shortlisted.remove(&slot) {
            record_locate(server, question, &answer, Some(&summary), auto_ms[slot]);
            answers.insert(question.id.clone(), answer);
            continue;
        }
        let reply = replies[slot].take().expect("every question was asked");
        let answer = match (reply, &plans[slot], locate) {
            (Reply::Answer(answer), _, _) => answer,
            (Reply::Rows(asked), Some(plan), Some(calibration)) => {
                let baseline = match replies.get(asked_questions + plan.baseline) {
                    Some(Some(Reply::Rows(rows))) => rows.clone(),
                    _ => Err(failed("not_completed", "this `locate`'s content-free baseline was never asked".to_owned())),
                };
                let answer = match (asked, baseline) {
                    (Ok(asked), Ok(baseline)) => locate_answer_for(
                        plan.kind,
                        &plan.keys,
                        &plan.values,
                        calibration.heads.len(),
                        &asked,
                        &baseline,
                    ),
                    (Err(failure), _) => failure,
                    (Ok(_), Err(Answer::Error { code, message })) => Answer::Error {
                        code,
                        message: format!("its content-free baseline: {message}"),
                    },
                    (Ok(_), Err(other)) => other,
                };
                record_locate(server, question, &answer, None, auto_ms[slot]);
                answer
            }
            (Reply::Rows(_), _, _) => failed("not_a_readout", "this question read attention it has no plan for".to_owned()),
        };
        answers.insert(question.id.clone(), answer);
    }
    log_decision(&prepared, &answers, class, input_tokens, started);
    let output_tokens = generated(&prepared, &answers);
    Ok(DecideResponse {
        // The model that *performed* the evaluation, which is the one the
        // engine resolved — not the string the caller sent.
        model: resolved.unwrap_or_else(|| server.active().engine.model_id()),
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

/// A `locate`'s target, the kind it resolves to and its compression (GitHub
/// #278, spec 22): `auto` told by the target's shape and its fold, a named
/// kind that contradicts the state refused (`kind_mismatch`), and a fold of
/// what `auto` told is prose refused — all before any prefill.
fn resolve_locate(
    state: &OrderedValue,
    question: &PreparedQuestion,
    ask: LocateAsk,
    told: &mut BTreeMap<String, ignis_core::locate::Kind>,
) -> Result<((ignis_core::locate::Kind, ignis_core::locate::Compression), Option<f64>), Refusal> {
    use ignis_core::locate::{Compression, Kind};
    let id = question.id.as_str();
    let within = question.within.as_deref().unwrap_or("");
    let target = crate::locate::segmentable_target(state, within)
        .map_err(|error| Refusal::new(error.code(), format!("question {id:?}: {error}")))?;
    let records = crate::locate::is_records_array(target);
    let mut spent = None;
    let kind = match ask.kind {
        None => match told.get(within) {
            Some(&kind) => kind,
            None => {
                let started = std::time::Instant::now();
                let kind = crate::locate::auto_kind(target).0;
                spent = Some(started.elapsed().as_secs_f64() * 1e3);
                told.insert(within.to_owned(), kind);
                kind
            }
        },
        Some(Kind::Records) if !records => {
            return Err(Refusal::new(
                "kind_mismatch",
                format!("question {id:?} names kind \"records\", and its target is not an array of two or more JSON objects; ask for \"log\" or \"prose\", or omit `kind`"),
            ));
        }
        Some(kind @ (Kind::Log | Kind::Prose)) if records => {
            return Err(Refusal::new(
                "kind_mismatch",
                format!(
                    "question {id:?} names kind {:?}, and its target is an array of JSON objects, which is read as records; to fold it, ask for kind \"records\" with compression \"template_fold\"",
                    kind.label()
                ),
            ));
        }
        Some(kind) => kind,
    };
    let compression = ask.compression.unwrap_or(kind.default_compression());
    if (kind, compression) == (Kind::Prose, Compression::TemplateFold) {
        return Err(Refusal::new(
            "compression_unsupported",
            format!("question {id:?}: its target reads as prose, which does not fold into templates, so \"template_fold\" is refused; ask for \"none\", or omit `compression`"),
        ));
    }
    Ok(((kind, compression), spent))
}

/// Count and log one answered `locate` (GitHub #275, #278): a decision with
/// no answer mass (ADR 0017), a series of `ignis_locates_total`, and a line
/// of the request log with what produced it.
fn record_locate(
    server: &crate::Server,
    question: &PreparedQuestion,
    answer: &Answer,
    summary: Option<&shortlist::Summary>,
    auto_ms: Option<f64>,
) {
    let found_of = |found: Option<f64>| match found {
        None => crate::metrics::LocateFound::Unmeasured,
        Some(found) if found >= ignis_core::locate::FOUND_THRESHOLD => crate::metrics::LocateFound::True,
        Some(_) => crate::metrics::LocateFound::False,
    };
    match answer {
        Answer::Locate { kind, method, compression, found, segment, .. } => {
            let kind = ignis_core::locate::Kind::from(*kind);
            let method = ignis_core::locate::Method::from(*method);
            let compression = ignis_core::locate::Compression::from(*compression);
            if let Some(metrics) = &server.metrics {
                // Counted like a head point: a decision with no answer mass
                // (ADR 0017). Its baseline and its steps are not decisions.
                metrics.record_decision(question.kind.primitive(), None);
                metrics.record_locate(kind, method, compression, found_of(*found));
            }
            tracing::info!(
                name: "ignis.decide.located",
                question = question.id.as_str(),
                kind = kind.label(),
                method = method.label(),
                compression = compression.label(),
                found = *found,
                segment = *segment,
                p_none = summary.and_then(|s| s.stats.p_none),
                p_yes = summary.and_then(|s| s.stats.p_yes),
                windows = summary.map(|s| s.stats.windows),
                templates = summary.and_then(|s| s.stats.templates),
                template = summary.and_then(|s| s.stats.template),
                rows = summary.and_then(|s| s.stats.rows),
                candidates = summary.map(|s| candidates_text(&s.stats.candidates)).unwrap_or_default(),
                input_tokens = summary.map(|s| s.tokens),
                plan_ms = summary.map(|s| s.stats.plan_ms),
                auto_ms,
                "locate answered"
            );
        }
        Answer::Error { code, .. } => {
            tracing::info!(
                name: "ignis.decide.located",
                question = question.id.as_str(),
                kind = summary.map(|s| s.kind.label()),
                compression = summary.map(|s| s.compression.label()),
                error = code.as_str(),
                "locate failed"
            );
        }
        _ => {}
    }
}

/// Each `choice`'s candidates, as the request log carries them: segments or
/// templates comma-separated, one `choice` after another `;`-separated.
fn candidates_text(candidates: &[Vec<usize>]) -> String {
    candidates
        .iter()
        .map(|list| list.iter().map(ToString::to_string).collect::<Vec<_>>().join(","))
        .collect::<Vec<_>>()
        .join(";")
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
            // GitHub #260, #263: a head point or box is answered by its
            // prefill and generates nothing, however long the chain's
            // schedule would be.
            Some(Answer::Point { method: SpatialMethod::Head, .. })
            | Some(Answer::Box { method: SpatialMethod::Head, .. }) => None,
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
    let resolved = crate::thinking::resolve(fields, &defaults, &server.active().template.thinking_capabilities())
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
    /// A head question's image grid, `(rows, cols)` of merged tokens (GitHub
    /// #260), and `None` for every other question.
    grid: Option<(usize, usize)>,
    /// Where each of the `state`'s content parts ends in the prompt, in tokens
    /// (GitHub #270). Empty for a JSON `state`, whose evidence is in the
    /// system block, and for a question that did not ask.
    part_ends: Vec<Option<u32>>,
    /// The answer a question already has without being submitted — a head
    /// question with no image to read, or one whose image is not a grid the
    /// heads' maps can be read over (GitHub #260). `None` for every question
    /// the engine is asked.
    answered: Option<Answer>,
}

/// Build one question's prompt. Refuses the whole request on failure: a
/// prompt that cannot be rendered, or one longer than the engine's context,
/// is the caller's mistake and every sibling shares it.
///
/// `part_ends` asks the render where the `state`'s parts end (GitHub #270),
/// which only a question whose reuse boundaries are cut from them needs.
async fn render(
    server: &crate::Server,
    evidence: &Evidence,
    question: &PreparedQuestion,
    model: Option<String>,
    part_ends: bool,
) -> Result<Rendered, Refusal> {
    let messages = messages_for(evidence, question);
    let thinking = crate::thinking::ThinkingOptions {
        enable_thinking: false,
        ..crate::thinking::ThinkingOptions::default()
    };
    let params = ignis_core::types::DecodeParams::default();
    let structure = match part_ends {
        true => crate::api::Structure::PartEnds,
        false => crate::api::Structure::Tokens,
    };
    let crate::api::PreparedRequest { mut input, model, mut prompt_tokens, media, part_ends, .. } =
        crate::api::prepare_decision_request(server, model, &messages, params, &thinking, structure)
            .await
            .map_err(|(code, message)| {
                Refusal::new(code, format!("question {:?}: {message}", question.id))
            })?;
    if prompt_tokens > server.active().engine.max_model_len() {
        return Err(Refusal::new(
            "context_exceeded",
            format!(
                "question {:?} renders {prompt_tokens} prompt tokens, past this engine's {} context",
                question.id,
                server.active().engine.max_model_len()
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
            if prompt_tokens > server.active().engine.max_model_len() {
                return Err(Refusal::new(
                    "context_exceeded",
                    format!(
                        "question {:?} renders {prompt_tokens} prompt tokens with its forced prefix, past this engine's {} context",
                        question.id,
                        server.active().engine.max_model_len()
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
                // GitHub #260, #263: a head question is the chain point's
                // prompt, forced opening and all, read at its last position
                // instead of decoded from: a decision over the calibrated
                // heads' attention across the image's placeholder span.
                Some(calibration) => {
                    // An image the heads cannot read is answered here, and
                    // nothing is submitted for it.
                    let (grid, answered) = match head_query(&input, calibration) {
                        Ok((query, grid)) => {
                            input.decision = Some(ignis_core::DecisionRead::Attention(query));
                            (Some(grid), None)
                        }
                        Err(answer) => (None, Some(answer)),
                    };
                    return Ok(Rendered { input, model, prompt_tokens, media, grid, part_ends, answered });
                }
            }
        }
    }
    Ok(Rendered { input, model, prompt_tokens, media, grid: None, part_ends, answered: None })
}

/// The most **reuse markers** one `state` may carry (GitHub #270): the hosted
/// APIs' own limit, and each costs a retained slot and a chunk split.
pub const MAX_REUSE_MARKERS: usize = 4;

/// The `state` parts carrying a **reuse marker** (GitHub #270), by index —
/// or the refusal of a malformed one, or of more than [`MAX_REUSE_MARKERS`].
///
/// Exactly `{"type": "ephemeral"}`, the shape Alibaba Model Studio and
/// Anthropic use. Retention here is by eviction, not by time, so a `ttl` is a
/// promise this server would not keep, and is refused rather than dropped. A
/// JSON `state` has no parts and so no markers: a `cache_control` key inside
/// evidence is evidence.
fn reuse_markers(evidence: &Evidence) -> Result<Vec<usize>, Refusal> {
    let Evidence::Parts(parts) = evidence else {
        return Ok(Vec::new());
    };
    let ephemeral = json!({"type": "ephemeral"});
    let mut marked = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        match &part.cache_control {
            None => {}
            Some(marker) if *marker == ephemeral => marked.push(index),
            Some(marker) => {
                return Err(Refusal::new(
                    "malformed_reuse_marker",
                    format!(
                        "`state` part {index} carries `cache_control: {marker}`; the only reuse marker is `{ephemeral}`, and it is kept until the device needs the room, never for a time"
                    ),
                ));
            }
        }
    }
    if marked.len() > MAX_REUSE_MARKERS {
        return Err(Refusal::new(
            "too_many_reuse_markers",
            format!(
                "`state` carries {} reuse markers, and at most {MAX_REUSE_MARKERS} are kept: each costs a retained slot",
                marked.len()
            ),
        ));
    }
    Ok(marked)
}

/// Give every rendered question the **reuse boundaries** its `state` earns
/// (GitHub #270, spec 16), and return what ends the fan-out's head, if it has
/// one.
///
/// For a `state` of content parts, the caller's **reuse markers** — or, when
/// there are none, an **observed fork**. Then, for a fan-out of two or more,
/// the **fan-out head**, on its sequenced first question only: the followers
/// claim it rather than publish it.
///
/// None of it changes a byte of any prompt, and none of it decides what a
/// question may claim — only where state is kept.
fn place_reuse_boundaries(
    server: &crate::Server,
    evidence: &Evidence,
    markers: &[usize],
    rendered: &mut [Rendered],
) -> Option<FanOutEnd> {
    use ignis_core::types::ReuseBoundary;
    if let Evidence::Parts(parts) = evidence {
        if markers.is_empty() {
            observe_forks(server, parts.len(), rendered);
        } else {
            for ready in rendered.iter_mut() {
                let ends = markers.iter().filter_map(|&part| ready.part_ends.get(part).copied().flatten());
                ready.input.reuse_boundaries.extend(ends.map(ReuseBoundary::retained));
            }
        }
    }
    // Only questions the engine is asked: one answered here is never
    // prefilled, so it neither publishes a head nor claims one.
    let asked: Vec<usize> = (0..rendered.len()).filter(|&i| rendered[i].answered.is_none()).collect();
    if asked.len() < 2 || asked[0] != 0 {
        return None;
    }
    let prompts: Vec<&ignis_core::types::RequestInput> = asked.iter().map(|&i| &rendered[i].input).collect();
    let head = crate::reuse::common_head(&prompts);
    if head == 0 {
        return None;
    }
    let owner = server.next_fan_out.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    rendered[0].input.reuse_boundaries.push(ReuseBoundary::fan_out(head, owner));
    Some(FanOutEnd { engine: server.active().engine.clone(), owner })
}

/// The **observed fork** (GitHub #270): each question whose `state` begins
/// with a run of parts a recent request's also began with gets a retained
/// boundary at the end of the longest such run. Then every run end of this
/// request enters the history — after all its questions have read it, so the
/// first request over a `state` publishes no fork, whatever its width.
///
/// A run end is where `parts[0..=i]` ends, keyed by the whole prompt up to
/// there: the same parts under another question kind's instruction are
/// another run.
fn observe_forks(server: &crate::Server, state_parts: usize, rendered: &mut [Rendered]) {
    use ignis_core::types::ReuseBoundary;
    let runs: Vec<Vec<crate::reuse::RunEnd>> = rendered
        .iter()
        .map(|ready| {
            if ready.answered.is_some() {
                return Vec::new();
            }
            // A run end at or below the system block adds nothing: the block
            // is kept there already.
            let block = ready.input.system_block_tokens.unwrap_or(0);
            let ends: Vec<u32> = ready
                .part_ends
                .iter()
                .take(state_parts)
                .flatten()
                .copied()
                .filter(|&end| end > block)
                .collect();
            crate::reuse::run_ends(&ready.input, &ends)
        })
        .collect();
    let mut history = server.fork_history.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    for (ready, runs) in rendered.iter_mut().zip(&runs) {
        if let Some(at) = history.longest_seen(runs) {
            ready.input.reuse_boundaries.push(ReuseBoundary::retained(at));
        }
    }
    for runs in &runs {
        history.record(runs);
    }
}

/// Tells the engine a fan-out is over when dropped (GitHub #270): its head
/// goes whether every question was answered, one failed, or the client left
/// and the handler's future with it.
struct FanOutEnd {
    engine: crate::engine::Engine,
    owner: ignis_core::types::FanOutId,
}

impl Drop for FanOutEnd {
    fn drop(&mut self) {
        self.engine.end_fan_out(self.owner);
    }
}

/// The attention readout a head question asks for over `input`'s image, and
/// that image's merged token grid `(rows, cols)` (GitHub #260) — or the
/// answer it gets instead: `state_carries_no_image` when there is no image,
/// the chain's own failure for the same state, and `image_not_a_grid` when
/// the item's placeholders are not one frame's merged grid, row by row, which
/// is the only map the heads are read over.
///
/// With a head set (GitHub #263) the readout names it too, with that grid's
/// fallback cells as the keys its argmax skips.
///
/// The **first** image, as the chain's pixels are the first image's: a
/// `point` answers about the submitted image, and a multi-image state is
/// outside what either method was measured on.
fn head_query(
    input: &ignis_core::types::RequestInput,
    calibration: ignis_core::pointing::Calibration,
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
            let set = calibration
                .set
                .map(|set| ignis_core::pointing::SetQuery::for_grid(set, rows as u32, cols as u32));
            let query = ignis_core::pointing::AttentionQuery {
                head: calibration.head,
                key_begin,
                key_count,
                set,
            };
            Ok((query, (rows, cols)))
        }
        _ => Err(failed(
            "image_not_a_grid",
            format!(
                "the image's {} placeholders are not one frame's {rows}x{cols} merged grid, so the heads' maps cannot be read over it",
                item.token_span.count
            ),
        )),
    }
}

/// What a `locate` is answered from once its rows are back (GitHub #275):
/// each segment's keys over the span the heads read, each segment's value,
/// and which baseline is its content-free baseline.
struct LocatePlan {
    kind: ignis_core::locate::Kind,
    keys: Vec<Option<std::ops::Range<usize>>>,
    values: Vec<OrderedValue>,
    baseline: usize,
}

/// A `locate` target's **content-free** prefill (GitHub #275): the same
/// state, kind text and scaffold with the instruction `N/A`, rendered once
/// per `within` and read by every `locate` over it. What the heads do with
/// the state when nothing is asked, which the vote subtracts.
struct Baseline {
    within: String,
    span: std::ops::Range<usize>,
    keys: Vec<Option<std::ops::Range<usize>>>,
    ready: Rendered,
}

/// One `locate` prompt, rendered as the vote was calibrated.
struct LocatePrompt {
    ready: Rendered,
    span: std::ops::Range<usize>,
    keys: Vec<Option<std::ops::Range<usize>>>,
    values: Vec<OrderedValue>,
}

/// Render a `locate` and, the first time its target is met, the target's
/// content-free baseline (GitHub #275). A question whose keys are not its
/// baseline's — the two prompts share every byte up to the instruction, so
/// they never should — is answered here with that error and not submitted.
async fn render_locate(
    server: &crate::Server,
    state: &OrderedValue,
    question: &PreparedQuestion,
    model: Option<String>,
    calibration: ignis_core::locate::LocateCalibration,
    kind: ignis_core::locate::Kind,
    baselines: &mut Vec<Baseline>,
) -> Result<(Rendered, LocatePlan), Refusal> {
    let within = question.within.as_deref().unwrap_or("");
    let mut prompt =
        locate_prompt(server, state, within, &question.instructions, &question.id, model.clone(), calibration).await?;
    let baseline = match baselines.iter().position(|baseline| baseline.within == within) {
        Some(index) => index,
        None => {
            let content_free = OrderedValue::String(crate::locate::CONTENT_FREE.to_owned());
            let twin = locate_prompt(server, state, within, &content_free, &question.id, model, calibration).await?;
            baselines.push(Baseline { within: within.to_owned(), span: twin.span, keys: twin.keys, ready: twin.ready });
            baselines.len() - 1
        }
    };
    let twin = &baselines[baseline];
    if (&twin.span, &twin.keys) != (&prompt.span, &prompt.keys) {
        prompt.ready.answered = Some(failed(
            "locate_baseline_misaligned",
            "the content-free baseline's prompt maps the state onto other keys than this question's, so the two cannot be read against each other".to_owned(),
        ));
    }
    Ok((prompt.ready, LocatePlan { kind, keys: prompt.keys, values: prompt.values, baseline }))
}

/// One `locate` prompt over `state` with `instruction`, as the vote was
/// calibrated (spec 18, GitHub #275): layout L1 — the evidence alone in the
/// system message, the kind text and the instruction in the user turn — the
/// copy scaffold forced, and the readout naming the vote's heads, in whole
/// rows, over the state's key span.
///
/// Everything a caller could have got wrong is refused here, before any
/// prefill: a `within` that names no segmentable target, fewer than two
/// segments that own a key, a span past the ceiling the vote was measured
/// to hold at, a prompt past the context.
async fn locate_prompt(
    server: &crate::Server,
    state: &OrderedValue,
    within: &str,
    instruction: &OrderedValue,
    id: &str,
    model: Option<String>,
    calibration: ignis_core::locate::LocateCalibration,
) -> Result<LocatePrompt, Refusal> {
    let prompt = render_reading(server, state, within, instruction, id, model, calibration.heads).await?;
    too_few_segments(id, &prompt.keys, 2)?;
    if prompt.span.len() > calibration.max_keys as usize {
        return Err(Refusal::new(
            "locate_too_long",
            format!(
                "question {id:?}: the target spans {} tokens, past the {} a `locate` was measured to hold at on this artifact",
                prompt.span.len(),
                calibration.max_keys
            ),
        ));
    }
    within_context(server, id, &prompt)?;
    Ok(prompt)
}

/// Refuse a reading with fewer than `least` segments that own a key: a
/// `locate` chooses between at least two.
fn too_few_segments(id: &str, keys: &[Option<std::ops::Range<usize>>], least: usize) -> Result<(), Refusal> {
    let owning = keys.iter().flatten().count();
    match owning < least {
        true => Err(Refusal::new(
            "locate_too_few_segments",
            format!(
                "question {id:?}: {owning} of the target's {} segments own a token of the prompt, and a `locate` chooses between at least two",
                keys.len()
            ),
        )),
        false => Ok(()),
    }
}

/// Refuse a reading prompt past the engine's context.
fn within_context(server: &crate::Server, id: &str, prompt: &LocatePrompt) -> Result<(), Refusal> {
    match prompt.ready.prompt_tokens > server.active().engine.max_model_len() {
        true => Err(Refusal::new(
            "context_exceeded",
            format!(
                "question {id:?} renders {} prompt tokens with its forced scaffold, past this engine's {} context",
                prompt.ready.prompt_tokens,
                server.active().engine.max_model_len()
            ),
        )),
        false => Ok(()),
    }
}

/// A **reading** prompt (GitHub #275, #278): the vote's render — layout L1,
/// the copy scaffold forced — with the readout naming `heads` in whole rows
/// over `within`'s key span, and nothing bounded yet: the vote bounds it by
/// its measured ceiling ([`locate_prompt`]), the shortlist by its window.
async fn render_reading(
    server: &crate::Server,
    state: &OrderedValue,
    within: &str,
    instruction: &OrderedValue,
    id: &str,
    model: Option<String>,
    heads: &'static [ignis_core::pointing::PointingHead],
) -> Result<LocatePrompt, Refusal> {
    let target = crate::locate::evidence_within(state, within)
        .map_err(|error| Refusal::new(error.code(), format!("question {id:?}: {error}")))?;
    let messages = vec![
        ChatMessage::text("system", target.system.clone()),
        ChatMessage::text("user", crate::locate::user_text(target.unit, instruction)),
    ];
    let thinking = crate::thinking::ThinkingOptions {
        enable_thinking: false,
        ..crate::thinking::ThinkingOptions::default()
    };
    let params = ignis_core::types::DecodeParams::default();
    let crate::api::PreparedRequest { mut input, model, mut prompt_tokens, media, text, .. } =
        crate::api::prepare_decision_request(server, model, &messages, params, &thinking, crate::api::Structure::Text)
            .await
            .map_err(|(code, message)| Refusal::new(code, format!("question {id:?}: {message}")))?;
    let unsupported = || {
        Refusal::new(
            "locate_unsupported",
            format!("question {id:?}: this load's template cannot say where its tokens sit, so the state's segments cannot be mapped onto keys"),
        )
    };
    let text = text.filter(|text| text.offsets.len() == input.tokens.len()).ok_or_else(unsupported)?;
    let opening = server.active().template.encode_literal(crate::locate::COPY_OPENING).ok_or_else(unsupported)?;
    let (span, keys) = crate::locate::map_segments(&text.text, &text.offsets, &target)
        .map_err(|message| Refusal::new("render_failed", format!("question {id:?}: {message}")))?;
    prompt_tokens = prompt_tokens.saturating_add(opening.len() as u32);
    // The scaffold the heads are read at, forced as text the model appears
    // to have written; the readout sits at its last token.
    input.tokens.extend_from_slice(&opening);
    input.decision = Some(ignis_core::DecisionRead::Attention(ignis_core::pointing::AttentionQuery {
        head: heads[0],
        key_begin: span.start as u32,
        key_count: span.len() as u32,
        set: Some(ignis_core::pointing::SetQuery::rows(heads)),
    }));
    Ok(LocatePrompt {
        ready: Rendered { input, model, prompt_tokens, media, grid: None, part_ends: Vec::new(), answered: None },
        span,
        keys,
        values: target.values,
    })
}

/// Shape a `locate`'s rows, and its content-free baseline's, into its answer
/// (GitHub #275): the head vote ([`ignis_core::locate::read_vote`]) over each
/// segment's `keys`, with the winner's `value` as the caller sent it. Rows
/// that did not come back whole — either prefill's — are the question's
/// failure, never a segment.
///
/// GitHub #278: the answer names the `kind` the target resolved to (unused
/// by the vote), `vote` and `none`, and points at the winner alone.
pub fn locate_answer_for(
    kind: ignis_core::locate::Kind,
    keys: &[Option<std::ops::Range<usize>>],
    values: &[OrderedValue],
    heads: usize,
    question: &ignis_core::pointing::AttentionScores,
    baseline: &ignis_core::pointing::AttentionScores,
) -> Answer {
    let reading = match (question.set_rows.as_deref(), baseline.set_rows.as_deref()) {
        (Some(asked), Some(content_free)) => ignis_core::locate::read_vote(asked, content_free, heads, keys),
        _ => None,
    };
    match reading {
        Some(reading) if reading.winner < values.len() => Answer::Locate {
            kind: kind.into(),
            method: LocateMethod::Vote,
            compression: LocateCompression::None,
            found: None,
            segment: Some(reading.winner),
            value: Some(values[reading.winner].clone()),
            confidence: Some(reading.confidence),
            pointers: vec![LocatePointer {
                segment: reading.winner,
                value: values[reading.winner].clone(),
                share: reading.confidence,
            }],
            ranking: reading
                .ranking
                .into_iter()
                .map(|(segment, share)| LocateRank { segment, share })
                .collect(),
        },
        _ => failed(
            "attention_malformed",
            format!(
                "the heads' rows were not {heads} whole rows over the state's key span in both the question's prefill and its content-free baseline's"
            ),
        ),
    }
}

/// How one asked prompt's completion is collected (GitHub #275): into its
/// question's answer, or — for a `locate` and its content-free baseline —
/// into the heads' rows, which only the two together answer.
#[derive(Clone, Copy)]
enum Collect<'a> {
    Answer(&'a PreparedQuestion),
    /// GitHub #278: a readout a `locate`'s shortlist asks on its way to its
    /// answer — a labelled `choice`, its "none" variant, its yes/no. Shaped as
    /// any readout, and never counted as a decision of its own: the caller
    /// asked one `locate`.
    Step(&'a PreparedQuestion),
    Rows,
}

/// What one asked prompt came back with.
enum Reply {
    Answer(Answer),
    Rows(Result<ignis_core::pointing::AttentionScores, Answer>),
}

impl Reply {
    /// The reply of a prompt that failed before it read anything, `failure`
    /// in the slot its collection would have filled.
    fn failed(collect: Collect<'_>, failure: Answer) -> Self {
        match collect {
            Collect::Answer(_) | Collect::Step(_) => Self::Answer(failure),
            Collect::Rows => Self::Rows(Err(failure)),
        }
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
    collect: Collect<'_>,
    mut ready: Rendered,
    class: ignis_core::types::RequestClass,
) -> Attempt {
    // GitHub #260: a head question over a state with no image fails as the
    // chain's does, and costs nothing — there is no span to read.
    if let Some(answer) = ready.answered.take() {
        return Attempt::Answered(Reply::Answer(answer));
    }
    // Cloned because `submit_with_media` consumes what it takes and a
    // `Full` has to be retriable: a prompt's worth of token ids beside a
    // prefill is nothing.
    // The engine the question is submitted to is the one its cancel guard
    // must reach, whatever a model switch does meanwhile.
    let engine = server.active().engine.clone();
    let submitted = engine.submit_with_media(ready.input.clone(), class, ready.media).await;
    let (id, mut events) = match submitted {
        Ok(pair) => pair,
        // Not an answer. The engine is saying "not now", and a fan-out's
        // own siblings are the likeliest reason.
        Err(ignis_core::SubmitError::Full) => return Attempt::Full(ready),
        Err(error) => {
            return Attempt::Answered(Reply::failed(collect, failed("submit_failed", format!("{error:?}"))));
        }
    };
    // The engine keeps working on a request whose caller has gone until it
    // is told otherwise, and a fan-out is twenty of them.
    let mut guard = crate::api::CancelOnDrop::new(engine, id);
    // GitHub #275: a `locate`'s completion, or its baseline's, carries the
    // heads' rows, which are an answer only beside the other's.
    // GitHub #278: a shortlist's step is a readout the caller did not ask
    // for, and is counted with the `locate` it answers, not beside it.
    let counted = matches!(collect, Collect::Answer(_));
    let question = match collect {
        Collect::Answer(question) | Collect::Step(question) => question,
        Collect::Rows => {
            return Attempt::Answered(Reply::Rows(
                match crate::engine::collect_attention(&mut events, server.live().request_timeout).await {
                    Ok(Some(attention)) => {
                        guard.completed();
                        Ok(attention)
                    }
                    Ok(None) => {
                        guard.completed();
                        Err(failed(
                            "attention_unread",
                            "the engine finished this `locate` without the heads' rows over the state: the keys were not where an armed layer's attention materialized them, or the prefill itself failed".to_owned(),
                        ))
                    }
                    Err(_) => Err(failed(
                        "not_completed",
                        "the engine did not answer this question in time".to_owned(),
                    )),
                },
            ));
        }
    };
    // GitHub #260, #263: a head question's completion carries the attention
    // readout.
    if let (Some(_), Some(grid)) = (question.head, ready.grid) {
        let pixels = ready.media.and_then(|stats| stats.source_pixels);
        return Attempt::Answered(Reply::Answer(
            match crate::engine::collect_attention(&mut events, server.live().request_timeout).await {
                Ok(Some(attention)) => {
                    guard.completed();
                    if let Some(metrics) = &server.metrics {
                        // A `point` or a `box` like the chain's (specs 13 and
                        // 14 leave a `method` label to ADR 0017), with no
                        // answer mass.
                        metrics.record_decision(question.kind.primitive(), None);
                    }
                    match pixels {
                        Some(pixels) => head_answer_for(question, &attention, grid, pixels),
                        None => no_image(),
                    }
                }
                Ok(None) => {
                    guard.completed();
                    failed(
                        "attention_unread",
                        "the engine finished this question without the heads' attention over the image: the keys were not where an armed layer's attention materialized them, or the prefill itself failed".to_owned(),
                    )
                }
                Err(_) => failed(
                    "not_completed",
                    "the engine did not answer this question in time".to_owned(),
                ),
            },
        ));
    }
    // GitHub #242: a run's completion carries a trace, not a readout,
    // so `collect_readout` would report every one of them as never
    // completed.
    if question.kind.is_constrained() {
        let pixels = ready.media.and_then(|stats| stats.source_pixels);
        return Attempt::Answered(Reply::Answer(
            match crate::engine::collect_draws(&mut events, server.live().request_timeout).await {
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
        ));
    }
    Attempt::Answered(Reply::Answer(
        match crate::engine::collect_readout(&mut events, server.live().request_timeout).await {
            Ok(readout) => {
                guard.completed();
                // GitHub #241. Here rather than in `serve` because this is
                // where the `Readout` is, and the answer built from it
                // keeps no trace of how much of the distribution it stood
                // on — a `choice` of 0.03 answer mass and one of 0.999 are
                // the same JSON.
                if let (Some(metrics), true) = (&server.metrics, counted) {
                    metrics.record_decision(question.kind.primitive(), Some(readout.answer_mass()));
                }
                answer_for(question, &readout)
            }
            Err(_) => failed(
                "not_completed",
                "the engine did not answer this question in time".to_owned(),
            ),
        },
    ))
}

/// Put `items` to the engine together, [`FAN_OUT_WIDTH`] at a time, with
/// whatever the engine turned away put back for the next wave.
///
/// A wave is `FAN_OUT_WIDTH` wide *because* the engine admits about that
/// many, but it is not the only client: two of its lanes may already be
/// somebody else's, and then two of this wave's questions come back
/// `Full`. Answering those with an error would be a failure the
/// one-at-a-time loop before #240 never produced, so they are re-queued
/// instead — a wave that answers even one question makes room for them. A
/// wave where *nothing* got in is an engine with no room at all, and that is
/// reported rather than spun on.
async fn in_waves<'a>(
    server: &crate::Server,
    items: Vec<(usize, Rendered, Collect<'a>)>,
    class: ignis_core::types::RequestClass,
) -> Vec<(usize, Reply)> {
    let mut out = Vec::with_capacity(items.len());
    let mut pending: std::collections::VecDeque<_> = items.into();
    while !pending.is_empty() {
        let wave: Vec<_> = (0..FAN_OUT_WIDTH.min(pending.len()))
            .filter_map(|_| pending.pop_front())
            .collect();
        let asked: Vec<(usize, Collect<'a>)> = wave.iter().map(|(slot, _, collect)| (*slot, *collect)).collect();
        let run = wave
            .into_iter()
            .map(|(_, ready, collect)| ask(server, collect, ready, class))
            .collect();
        let mut answered_one = false;
        for ((slot, collect), attempt) in asked.into_iter().zip(concurrently(run).await) {
            match attempt {
                Attempt::Answered(reply) => {
                    out.push((slot, reply));
                    answered_one = true;
                }
                Attempt::Full(ready) => pending.push_back((slot, ready, collect)),
            }
        }
        if !answered_one {
            for (slot, _, collect) in pending.drain(..) {
                out.push((slot, Reply::failed(collect, engine_full())));
            }
        }
    }
    out
}

/// The first of `items` alone, then the rest [`in_waves`] (GitHub #278): the
/// first leaves the prefix they share retained, and they claim it.
async fn led<'a>(
    server: &crate::Server,
    mut items: Vec<(usize, Rendered, Collect<'a>)>,
    class: ignis_core::types::RequestClass,
) -> Vec<(usize, Reply)> {
    if items.is_empty() {
        return Vec::new();
    }
    let (slot, ready, collect) = items.remove(0);
    let mut out = vec![(slot, alone(server, collect, ready, class).await)];
    out.extend(in_waves(server, items, class).await);
    out
}

/// Ask one prompt alone: its reply, or — when the engine had no room for it —
/// its failure in the slot its collection would have filled.
async fn alone(
    server: &crate::Server,
    collect: Collect<'_>,
    ready: Rendered,
    class: ignis_core::types::RequestClass,
) -> Reply {
    match ask(server, collect, ready, class).await {
        Attempt::Answered(reply) => reply,
        Attempt::Full(_) => Reply::failed(collect, engine_full()),
    }
}

/// What one attempt at a question produced: its reply, or the unsubmitted
/// question back because the engine had no room for it.
enum Attempt {
    Answered(Reply),
    Full(Rendered),
}

/// The answer a spatial question gets over a `state` with no image —
/// whichever method it asked for.
fn no_image() -> Answer {
    failed(
        "state_carries_no_image",
        "a point or a box is a place on an image, and this `state` carried none".to_owned(),
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
