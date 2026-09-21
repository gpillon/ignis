//! GitHub #239 — the `/v1/decide` wire shape, its validation and its
//! arithmetic, none of which needs an engine.
//!
//! The shape is TypeSafe's Jev (`POST /v1/systemone`), copied so an
//! unmodified Jev client reaches this server by changing the URL. Their own
//! documented examples are reproduced here verbatim: if one of them stops
//! parsing, or stops producing an answer of their documented shape, this
//! endpoint has stopped being reachable by the clients it exists for.

use ignis_core::decision::{AnswerAlphabet, LabelTokenizer};
use ignis_core::types::TokenId;
use serde_json::{Value as JsonValue, json};

use ignis_server::decide::{
    Answer, DecideRequest, Evidence, MAX_OPTIONS, Ordered, OrderedValue, PreparedQuestion, Question,
    QuestionKind, Refusal, answer_for, constrained_answer_for, expected_level, messages_for,
    prepare, score_confidence,
};

/// A tokenizer that names every single character and every uppercase bigram
/// — 738 labels, enough to reach the 256-option ceiling and past it.
///
/// The endpoint's real alphabet is the loaded model's (GitHub #237); this is
/// what lets the *wire* rules be tested without one.
struct WideTokenizer {
    entries: Vec<String>,
}

impl WideTokenizer {
    fn new() -> Self {
        let mut entries: Vec<String> = ('A'..='Z')
            .chain('a'..='z')
            .chain('0'..='9')
            .map(|c| c.to_string())
            .collect();
        for a in 'A'..='Z' {
            for b in 'A'..='Z' {
                entries.push(format!("{a}{b}"));
            }
        }
        Self { entries }
    }
}

impl LabelTokenizer for WideTokenizer {
    fn encode(&self, text: &str) -> Option<Vec<TokenId>> {
        self.entries
            .iter()
            .position(|entry| entry == text)
            .map(|index| vec![index as TokenId])
    }

    fn decode(&self, ids: &[TokenId]) -> Option<String> {
        ids.iter()
            .map(|&id| self.entries.get(id as usize).cloned())
            .collect::<Option<Vec<String>>>()
            .map(|pieces| pieces.concat())
    }
}

fn alphabet() -> AnswerAlphabet {
    AnswerAlphabet::from_tokenizer(&WideTokenizer::new())
}

/// The literal encoder a **constrained decode** question's plan needs (GitHub #242):
/// one token per character, so every digit is a single vocabulary entry and
/// an arbitrary forced literal has an encoding.
///
/// Separate from [`WideTokenizer`], which only knows the labels it was built
/// from and would refuse `{"x":` — the real tokenizer encodes anything.
fn encoder(text: &str) -> Option<Vec<TokenId>> {
    Some(text.chars().map(|c| c as TokenId).collect())
}

/// Parse a request from **raw JSON text**, the way axum's extractor will.
///
/// Never from a `serde_json::Value`: this build's `Value::Object` is a
/// sorted map, so a `json!` literal has already lost the declared option
/// order by the time it is deserialized. A test that went through one could
/// not see the order rule at all -- which is exactly the mistake the rule
/// exists to prevent, made one layer up.
fn parse(body: &str) -> DecideRequest {
    serde_json::from_str(body).expect("a well-formed request parses")
}

fn prepared(body: &str) -> Vec<PreparedQuestion> {
    prepare(&parse(body).questions, &alphabet(), &encoder).expect("a valid request prepares")
}

/// [`prepared`] for a body built with `json!`, where no declared order is
/// under test.
fn prepared_value(body: JsonValue) -> Vec<PreparedQuestion> {
    prepared(&body.to_string())
}

/// A readout standing in for the model's: the named logits, and everything
/// else far below them.
fn readout(logits: &[f32]) -> ignis_core::decision::Readout {
    let mut vocab = vec![-60.0f32; 1024];
    for (slot, &logit) in logits.iter().enumerate() {
        vocab[slot] = logit;
    }
    let ids: Vec<TokenId> = (0..logits.len() as TokenId).collect();
    ignis_core::decision::Readout::gather(&vocab, &ids)
}

/// A readout that lands the given probabilities on the answer tokens
/// exactly — logits are `ln(p)`, which renormalizes back to `p`.
fn readout_of(probabilities: &[f64]) -> ignis_core::decision::Readout {
    let logits: Vec<f32> = probabilities.iter().map(|p| p.ln() as f32).collect();
    readout(&logits)
}

// ── Jev's own documented examples (acceptance 1) ─────────────────────────

/// Jev's `noul` example, verbatim from `docs.typesafe.ai/api`.
const JEV_NOUL: &str = r#"{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this convey urgency?",
      "criteria": {
        "true": "Explicitly time-sensitive",
        "false": "No urgency expressed"
      }
    }
  }
}"#;

/// Jev's `choice` example, verbatim.
const JEV_CHOICE: &str = r#"{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"
      }
    }
  }
}"#;

/// Jev's `score` example, verbatim.
const JEV_SCORE: &str = r#"{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}"#;

/// Jev's first example: a noul with no `criteria` at all.
const JEV_BARE_NOUL: &str = r#"{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this convey urgency?"
    }
  }
}"#;

#[test]
fn jevs_noul_example_answers_in_jevs_noul_shape() {
    let questions = prepared(JEV_NOUL);
    assert_eq!(questions.len(), 1);
    let question = &questions[0];
    assert_eq!(question.id, "is_urgent");
    assert_eq!(question.kind, QuestionKind::Noul);
    assert_eq!(
        question.options.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(),
        vec!["true", "false"],
        "the true option is slot 0, because the answer is the probability of yes"
    );

    let answer = answer_for(question, &readout_of(&[0.92, 0.08]));
    let json = serde_json::to_value(&answer).expect("serializes");
    assert_eq!(json["type"], "noul");
    let noul = json["noul"].as_f64().expect("a number");
    assert!((noul - 0.92).abs() < 1e-6, "{noul}");
    assert_eq!(
        json.as_object().expect("an object").len(),
        2,
        "Jev's noul answer carries `type` and `noul` and nothing else: {json}"
    );
}

#[test]
fn jevs_choice_example_answers_in_jevs_choice_shape() {
    let questions = prepared(JEV_CHOICE);
    let question = &questions[0];
    assert_eq!(question.id, "department");
    assert_eq!(
        question.options.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(),
        vec!["billing", "technical", "sales"],
        "in the order the caller declared them, not sorted"
    );

    let answer = answer_for(question, &readout_of(&[0.08, 0.85, 0.07]));
    let json = serde_json::to_value(&answer).expect("serializes");
    assert_eq!(json["type"], "choice");
    assert_eq!(json["choice"], "technical", "the highest-probability option");
    let probabilities = json["probabilities"].as_object().expect("a map");
    assert_eq!(probabilities.len(), 3);
    assert!(
        (probabilities["technical"].as_f64().unwrap() - 0.85).abs() < 1e-6,
        "{probabilities:?}"
    );
    let sum: f64 = probabilities.values().map(|p| p.as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-9, "the distribution sums to one: {sum}");
    let confidence = json["confidence"].as_f64().expect("a number");
    assert!(
        (confidence - 0.85).abs() < 1e-6,
        "our confidence is the top probability, documented as such: {confidence}"
    );
}

#[test]
fn jevs_score_example_answers_in_jevs_score_shape() {
    let questions = prepared(JEV_SCORE);
    let question = &questions[0];
    assert_eq!(question.id, "frustration");
    assert_eq!(
        question.options.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(),
        vec!["0", "1", "2"],
        "a level is named by its index, which is what the score is over"
    );

    let answer = answer_for(question, &readout_of(&[0.05, 0.3, 0.65]));
    let json = serde_json::to_value(&answer).expect("serializes");
    assert_eq!(json["type"], "score");
    // Acceptance 2, and Jev's own number for this distribution.
    let score = json["score"].as_f64().expect("a number");
    assert!((score - 1.6).abs() < 1e-6, "expected 1.6, got {score}");
    assert_eq!(
        json["legend"],
        json!({ "0": "Calm", "1": "Frustrated", "2": "Very angry" }),
        "each level number mapped back to its description"
    );
    let probabilities = json["probabilities"].as_object().expect("a map");
    assert_eq!(probabilities.len(), 3);
    assert!((probabilities["2"].as_f64().unwrap() - 0.65).abs() < 1e-6);
    assert!(json["confidence"].is_number());
}

#[test]
fn jevs_bare_noul_example_needs_no_criteria() {
    // Jev's first example carries no `criteria` at all: for a noul they are
    // optional descriptions of what a yes and a no mean.
    let questions = prepared(JEV_BARE_NOUL);
    assert_eq!(questions[0].options.len(), 2);
    assert_eq!(
        questions[0].options.iter().map(|o| o.description.as_str()).collect::<Vec<_>>(),
        vec!["Yes", "No"]
    );
}

// ── the arithmetic (acceptance 2) ────────────────────────────────────────

#[test]
fn a_score_is_the_expected_value_over_the_level_indices() {
    assert!((expected_level(&[0.05, 0.3, 0.65]) - 1.6).abs() < 1e-12);
    assert!((expected_level(&[1.0, 0.0, 0.0]) - 0.0).abs() < 1e-12);
    assert!((expected_level(&[0.0, 0.0, 1.0]) - 2.0).abs() < 1e-12);
    assert!(
        (expected_level(&[0.0, 0.5, 0.5]) - 1.5).abs() < 1e-12,
        "a state the model splits between two adjacent levels lands between them"
    );
}

#[test]
fn a_score_straddling_two_adjacent_levels_is_confident() {
    // The reason a score's confidence is not its top probability. Both of
    // these have a top probability of 0.5; only one of them knows anything.
    let adjacent = score_confidence(&[0.0, 0.5, 0.5, 0.0]);
    let opposite = score_confidence(&[0.5, 0.0, 0.0, 0.5]);
    assert!(
        adjacent > 0.6,
        "half-way between two neighbouring levels is a confident answer: {adjacent}"
    );
    assert!(
        opposite < 0.05,
        "half-way between the extremes knows nothing: {opposite}"
    );
    assert!(adjacent > opposite);
}

#[test]
fn a_certain_score_is_fully_confident_and_never_leaves_the_unit_interval() {
    assert!((score_confidence(&[0.0, 1.0, 0.0]) - 1.0).abs() < 1e-12);
    for distribution in [
        vec![0.5, 0.5],
        vec![1.0, 0.0],
        vec![0.25, 0.25, 0.25, 0.25],
        vec![0.05, 0.3, 0.65],
    ] {
        let confidence = score_confidence(&distribution);
        assert!(
            (0.0..=1.0).contains(&confidence),
            "{distribution:?} gave {confidence}"
        );
    }
    assert!(
        score_confidence(&[0.5, 0.5]) < 1e-12,
        "the widest a two-level score can be spread scores zero"
    );
}

// ── criteria (acceptance 3) ──────────────────────────────────────────────

#[test]
fn a_null_criterion_makes_the_option_id_its_own_description() {
    let questions = prepared(
        r#"{"state":"s","questions":{"q":{"type":"choice","instructions":"Which?","criteria":{"billing":null,"technical":"Bugs and outages"}}}}"#,
    );
    let options = &questions[0].options;
    assert_eq!(options[0].name, "billing");
    assert_eq!(
        options[0].description, "billing",
        "null means the option needs no extra detail, so the id describes it"
    );
    assert_eq!(options[1].description, "Bugs and outages");
}

#[test]
fn our_own_names_for_jevs_fields_are_read_as_aliases() {
    let questions = prepared_value(json!({
        "state": "s",
        "questions": {
            "q": { "type": "boolean", "question": "Urgent?", "options": { "true": "yes", "false": "no" } }
        }
    }));
    assert_eq!(questions[0].kind, QuestionKind::Noul);
    assert_eq!(questions[0].instructions, OrderedValue::String("Urgent?".to_owned()));
    assert_eq!(questions[0].options[0].description, "yes");
}

// ── option order is an input (acceptance 4) ──────────────────────────────

#[test]
fn the_same_options_in_two_orders_are_two_different_prompts() {
    let of = |criteria: &str| -> String {
        let body = format!(
            r#"{{"state":"s","questions":{{"department":{{"type":"choice","instructions":"Which team?","criteria":{criteria}}}}}}}"#
        );
        let questions = prepared(&body);
        let messages = messages_for(&Evidence::Json(OrderedValue::from_json(&json!("evidence"))), &questions[0]);
        messages[1].content.text()
    };
    let forward = of(r#"{"billing":"Payments","technical":"Bugs"}"#);
    let reversed = of(r#"{"technical":"Bugs","billing":"Payments"}"#);

    assert_ne!(
        forward, reversed,
        "a different option order is a different prompt, so it must reach the model as one"
    );
    assert!(
        forward.find("Payments").unwrap() < forward.find("Bugs").unwrap(),
        "declared order survives into the prompt: {forward}"
    );
    assert!(
        reversed.find("Bugs").unwrap() < reversed.find("Payments").unwrap(),
        "and so does the other one: {reversed}"
    );
}

#[test]
fn the_answer_tokens_follow_the_declared_order() {
    let questions = prepared(
        r#"{"state":"s","questions":{"q":{"type":"choice","instructions":"Which?","criteria":{"zebra":null,"aardvark":null}}}}"#,
    );
    let question = &questions[0];
    assert_eq!(question.answers[0].label, "A", "the first declared option is A");
    assert_eq!(question.answers[1].label, "B");
    assert_eq!(question.options[0].name, "zebra", "even when sorting would say otherwise");

    // And the winning label maps back to the option that carried it.
    let answer = answer_for(question, &readout_of(&[0.1, 0.9]));
    assert!(matches!(&answer, Answer::Choice { choice, .. } if choice == "aardvark"));
}

// ── an evidence's own order is an input too ─────────────────────────────

/// The system block one question is asked under, for a `state` written exactly
/// as the caller wrote it.
fn system_block_for(state: &str) -> String {
    let body = format!(
        r#"{{"state":{state},"questions":{{"q":{{"type":"noul","instructions":"Is it?"}}}}}}"#
    );
    let request: DecideRequest = serde_json::from_str(&body).expect("a valid request");
    let questions = prepare(&request.questions, &alphabet(), &encoder).expect("a valid decision");
    messages_for(&Evidence::read(&request.state), &questions[0])[0].content.text()
}

#[test]
fn a_json_evidence_reaches_the_model_in_the_order_it_was_written() {
    // The same argument option order gets, one level up. `serde_json::Map` is
    // a `BTreeMap` in this build, so without `OrderedValue` this record would
    // arrive alphabetised — and it is not a presentation detail: on the loaded
    // model the authored order and the sorted order answered 0.82 and 0.32 to
    // the same question.
    let authored = system_block_for(r#"{"order":"A-4471","note":"it is a gift","amount":12}"#);
    assert!(
        authored.find(r#""order""#).unwrap() < authored.find(r#""note""#).unwrap(),
        "the caller wrote `order` first: {authored}"
    );
    assert!(
        authored.find(r#""note""#).unwrap() < authored.find(r#""amount""#).unwrap(),
        "and `note` before `amount`: {authored}"
    );

    // Written the other way round it is a different prompt, which is the whole
    // claim: a sorted-away order would make these two identical.
    let reversed = system_block_for(r#"{"amount":12,"note":"it is a gift","order":"A-4471"}"#);
    assert_ne!(authored, reversed);
}

#[test]
fn a_nested_object_keeps_its_order_too() {
    let block = system_block_for(r#"{"shipping":{"method":"standard","country":"IT"}}"#);
    assert!(
        block.find(r#""method""#).unwrap() < block.find(r#""country""#).unwrap(),
        "the order holds all the way down: {block}"
    );
}

#[test]
fn a_number_keeps_the_spelling_the_caller_gave_it() {
    // `149.0` and `149` are different bytes in the prompt, and the model reads
    // bytes. Parsing through an `f64` would have made them the same.
    assert!(system_block_for(r#"{"price":149.0}"#).contains("149.0"));
    assert!(!system_block_for(r#"{"price":149}"#).contains("149.0"));
}

#[test]
fn an_object_instruction_keeps_its_order() {
    let body = r#"{"state":"s","questions":{"q":{"type":"noul","instructions":{"ask":"is it?","about":"the order"}}}}"#;
    let request: DecideRequest = serde_json::from_str(body).expect("a valid request");
    let questions = prepare(&request.questions, &alphabet(), &encoder).expect("a valid decision");
    let user = messages_for(&Evidence::read(&request.state), &questions[0])[1].content.text();
    assert!(
        user.find(r#""ask""#).unwrap() < user.find(r#""about""#).unwrap(),
        "`instructions` is a `state` in miniature: {user}"
    );
}

#[test]
fn the_options_keep_the_two_keys_in_the_order_they_were_measured_with() {
    // `description` before `letter`. Not the order anyone would choose, but
    // the order `serde_json` used to emit, and the order every number this
    // endpoint rests on was measured against.
    let questions = prepared(
        r#"{"state":"s","questions":{"q":{"type":"choice","instructions":"Which?","criteria":{"billing":"Payments"}}}}"#,
    );
    let user = messages_for(&Evidence::Json(OrderedValue::from_json(&json!("s"))), &questions[0])[1]
        .content
        .text();
    assert!(
        user.contains(r#"{"description":"Payments","letter":"A"}"#),
        "the measured option shape is kept: {user}"
    );
}

// ── the ceiling (acceptance 6) ───────────────────────────────────────────

fn wide_choice(count: usize) -> String {
    let criteria: Vec<String> = (0..count)
        .map(|index| format!("\"option{index}\":null"))
        .collect();
    format!(
        r#"{{"state":"s","questions":{{"q":{{"type":"choice","instructions":"Which?","criteria":{{{}}}}}}}}}"#,
        criteria.join(",")
    )
}

#[test]
fn the_measured_ceiling_is_served_and_one_past_it_is_refused() {
    let at_ceiling = prepare(&parse(&wide_choice(MAX_OPTIONS)).questions, &alphabet(), &encoder);
    assert!(
        at_ceiling.is_ok(),
        "{MAX_OPTIONS} options is measured and served: {:?}",
        at_ceiling.err()
    );
    let past = prepare(&parse(&wide_choice(MAX_OPTIONS + 1)).questions, &alphabet(), &encoder)
        .expect_err("one past the ceiling is refused");
    assert_eq!(past.code, "too_many_options");
    assert!(
        past.message.contains("257") && past.message.contains("256"),
        "the refusal names both numbers: {}",
        past.message
    );
}

#[test]
fn a_model_whose_tokenizer_cannot_name_the_options_refuses_rather_than_collides() {
    // The alphabet is the loaded model's. A narrow one does not silently
    // give two options the same logit.
    struct OnlyTwo;
    impl LabelTokenizer for OnlyTwo {
        fn encode(&self, text: &str) -> Option<Vec<TokenId>> {
            match text {
                "A" => Some(vec![0]),
                "B" => Some(vec![1]),
                _ => None,
            }
        }
        fn decode(&self, ids: &[TokenId]) -> Option<String> {
            match ids {
                [0] => Some("A".to_owned()),
                [1] => Some("B".to_owned()),
                _ => None,
            }
        }
    }
    let narrow = AnswerAlphabet::from_tokenizer(&OnlyTwo);
    assert_eq!(narrow.len(), 2);
    let refusal = prepare(&parse(&wide_choice(3)).questions, &narrow, &encoder)
        .expect_err("three options do not fit a two-label alphabet");
    assert_eq!(refusal.code, "alphabet_exhausted");
}

// ── refusals are all-or-nothing (acceptance 7) ───────────────────────────

/// Each of these refuses the **whole** request, naming its offender.
#[test]
fn a_malformed_question_refuses_the_whole_request() {
    let cases: Vec<(&str, JsonValue)> = vec![
        (
            "empty_instructions",
            json!({ "bad": { "type": "noul", "instructions": " " } }),
        ),
        (
            "missing_criteria",
            json!({ "bad": { "type": "choice", "instructions": "Which?" } }),
        ),
        (
            "malformed_criteria",
            json!({ "bad": { "type": "choice", "instructions": "Which?", "criteria": ["a", "b"] } }),
        ),
        (
            "malformed_criteria",
            json!({ "bad": { "type": "score", "instructions": "How?", "criteria": { "a": "b" } } }),
        ),
        (
            "too_few_levels",
            json!({ "bad": { "type": "score", "instructions": "How?", "criteria": ["only"] } }),
        ),
        (
            "malformed_criteria",
            json!({ "bad": { "type": "score", "instructions": "How?", "criteria": ["ok", 7] } }),
        ),
        (
            "no_options",
            json!({ "bad": { "type": "choice", "instructions": "Which?", "criteria": {} } }),
        ),
        (
            "malformed_criteria",
            json!({ "bad": { "type": "choice", "instructions": "Which?", "criteria": { "a": 7 } } }),
        ),
    ];
    for (code, questions) in cases {
        // A good question beside the bad one, so the refusal is visibly
        // about the request and not about the only thing in it.
        let mut all = serde_json::Map::new();
        all.insert(
            "good".to_owned(),
            json!({ "type": "noul", "instructions": "Fine?" }),
        );
        for (id, question) in questions.as_object().expect("an object") {
            all.insert(id.clone(), question.clone());
        }
        let body = json!({ "state": "s", "questions": all }).to_string();
        let refusal = prepare(&parse(&body).questions, &alphabet(), &encoder)
            .err()
            .unwrap_or_else(|| panic!("{code}: expected a refusal"));
        assert_eq!(refusal.code, code, "{}", refusal.message);
        assert!(
            refusal.message.contains("bad"),
            "the refusal names the offending question: {}",
            refusal.message
        );
    }
}

#[test]
fn a_request_with_no_questions_is_refused() {
    let refusal = prepare(&Ordered(Vec::<(String, Question)>::new()), &alphabet(), &encoder)
        .expect_err("nothing to decide");
    assert_eq!(refusal.code, "no_questions");
}

#[test]
fn a_duplicate_question_id_is_refused_rather_than_silently_halved() {
    // `serde_json::json!` cannot express a duplicate key, so this is raw.
    let body: DecideRequest = serde_json::from_str(
        r#"{"state":"s","questions":{"q":{"type":"noul","instructions":"a"},"q":{"type":"noul","instructions":"b"}}}"#,
    )
    .expect("parses");
    assert_eq!(body.questions.len(), 2, "both copies survive the parse");
    let refusal = prepare(&body.questions, &alphabet(), &encoder).expect_err("ambiguous");
    assert_eq!(refusal.code, "duplicate_question");
}

// ── the evidence (acceptance 8's wire half) ──────────────────────────────

#[test]
fn a_state_that_is_content_parts_is_read_as_content_parts() {
    let state = json!([
        { "type": "image_url", "image_url": { "url": "https://example.test/a.png" } },
        { "type": "text", "text": "the receipt above" }
    ]);
    let evidence = Evidence::read(&OrderedValue::from_json(&state));
    let Evidence::Parts(parts) = &evidence else {
        panic!("expected content parts, got {evidence:?}");
    };
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].kind.as_deref(), Some("image_url"));
    assert!(evidence.has_media(), "an image is media");
}

#[test]
fn a_json_array_of_records_is_evidence_not_content_parts() {
    // The ambiguity worth getting right: Jev's `state` may be an array, and
    // an array of records is not a parts array. The rule is that every
    // element must carry a `type` string.
    for state in [
        json!([{ "id": 1, "title": "Payout failed" }, { "id": 2, "title": "Retry" }]),
        json!(["a", "b"]),
        json!([]),
        json!({ "ticket": 12 }),
        json!("a plain string"),
    ] {
        let evidence = Evidence::read(&OrderedValue::from_json(&state));
        assert!(
            matches!(evidence, Evidence::Json(_)),
            "{state} should be evidence, got {evidence:?}"
        );
        assert!(!evidence.has_media());
    }
}

#[test]
fn image_evidence_leads_the_prompt_and_drops_the_evidence_field() {
    // The shape `classify_vision_readout_gpu.rs` measured: the image is the
    // evidence, so it comes first as a content part and the JSON payload
    // carries only the criterion and the options.
    let questions = prepared_value(json!({
        "state": "s",
        "questions": { "q": { "type": "choice", "instructions": "Which number?", "criteria": { "42": null, "47": null } } }
    }));
    let state = json!([{ "type": "image_url", "image_url": { "url": "https://example.test/a.png" } }]);
    let messages = messages_for(&Evidence::read(&OrderedValue::from_json(&state)), &questions[0]);

    assert_eq!(messages[0].content.text(), ignis_server::decide::DIRECT_SYSTEM);
    let ignis_server::template::MessageContent::Parts(parts) = &messages[1].content else {
        panic!("the user message carries parts");
    };
    assert_eq!(parts[0].kind.as_deref(), Some("image_url"), "the image leads");
    let payload: JsonValue = serde_json::from_str(
        parts.last().expect("a text part").text.as_deref().expect("text"),
    )
    .expect("the payload is JSON");
    assert!(
        payload.get("evidence").is_none(),
        "the image is the evidence, so there is no evidence field: {payload}"
    );
    assert_eq!(payload["criterion"], "Which number?");
    assert_eq!(payload["options"][0]["letter"], "A");
}

#[test]
fn text_evidence_rides_in_the_system_block() {
    // GitHub #240. The evidence used to lead the user payload, which made it
    // *first* without making it *shared*: the only tier a decision's sibling
    // can claim is a retained prefix, and a retained prefix is cut inside the
    // system block and nowhere else (`decide::messages_for` says why).
    let questions = prepared(JEV_CHOICE);
    let messages = messages_for(
        &Evidence::read(&OrderedValue::from_json(&json!("Help! My payouts have been failing for 3 days."))),
        &questions[0],
    );
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "system");

    let system = messages[0].content.text();
    assert!(
        system.starts_with(ignis_server::decide::DIRECT_SYSTEM),
        "the instruction leads, so it is the same bytes for every state: {system}"
    );
    let evidence: JsonValue = serde_json::from_str(
        system.strip_prefix(ignis_server::decide::DIRECT_SYSTEM).unwrap().trim(),
    )
    .expect("the evidence is the block's JSON tail");
    assert_eq!(evidence["evidence"], "Help! My payouts have been failing for 3 days.");

    // And the question is the user turn, which is this request's alone.
    let payload: JsonValue =
        serde_json::from_str(&messages[1].content.text()).expect("the payload is JSON");
    assert!(payload.get("evidence").is_none(), "not repeated: {payload}");
    assert_eq!(payload["criterion"], "Which team should handle this?");
    assert_eq!(payload["options"][1]["letter"], "B");
    assert_eq!(payload["options"][1]["description"], "Bugs, outages, integrations");
}

#[test]
fn two_questions_over_one_state_have_the_same_system_block() {
    // The property the fan-out rests on, at the seam that decides it: what
    // two questions share has to be a whole *prefix*, byte for byte, or the
    // page floor of the block is a page of two different prompts.
    let body = r#"{
      "state": "Help! My payouts have been failing for 3 days.",
      "questions": {
        "is_urgent": { "type": "noul", "instructions": "Urgent?" },
        "department": { "type": "choice", "instructions": "Which team?", "criteria": { "billing": null, "technical": null } }
      }
    }"#;
    let questions = prepared(body);
    let evidence = Evidence::read(&OrderedValue::from_json(&json!("Help! My payouts have been failing for 3 days.")));
    let first = messages_for(&evidence, &questions[0]);
    let second = messages_for(&evidence, &questions[1]);

    assert_eq!(
        first[0].content.text(),
        second[0].content.text(),
        "one state is one system block"
    );
    assert_ne!(
        first[1].content.text(),
        second[1].content.text(),
        "and the questions are still two different prompts"
    );
}

#[test]
fn an_image_state_stays_in_the_user_turn() {
    // Spec 04's acceptance 2 is **not met**, and this is where that is
    // recorded rather than forgotten: an image `state` does not go where
    // the retained prefix is cut, so a fan-out over one re-encodes it per
    // question.
    //
    // Not because it *cannot*. `check_content_parts` refuses media in a
    // system message (GitHub #175) but never runs on the decide path, so
    // what keeps the image in the user turn is a choice — see
    // `decide::messages_for` for the two things that argue for it. This
    // test pins the choice; it does not pretend to pin a law.
    let questions = prepared_value(json!({
        "state": "s",
        "questions": { "q": { "type": "choice", "instructions": "Which number?", "criteria": { "42": null, "47": null } } }
    }));
    let state = json!([{ "type": "image_url", "image_url": { "url": "https://example.test/a.png" } }]);
    let messages = messages_for(&Evidence::read(&OrderedValue::from_json(&state)), &questions[0]);

    assert_eq!(
        messages[0].content.text(),
        ignis_server::decide::DIRECT_SYSTEM,
        "the system block carries the instruction and nothing else"
    );
    let ignis_server::template::MessageContent::Parts(parts) = &messages[1].content else {
        panic!("the image rides in the user turn");
    };
    assert_eq!(parts[0].kind.as_deref(), Some("image_url"), "and leads it");

    // What the rest of the server would say about the other arrangement,
    // recorded so the choice is visible next to its cost: the chat and
    // responses routes refuse exactly this.
    let mut moved = messages.clone();
    moved[0].content = messages[1].content.clone();
    let rejection = ignis_server::template::check_content_parts(&moved, true)
        .expect_err("media in a system message is refused on the routes that check");
    assert_eq!(rejection.code, "invalid_media");
}

#[test]
fn a_structured_instruction_reaches_the_prompt_whole() {
    // Jev's `instructions` is `string | object | array`.
    let questions = prepared_value(json!({
        "state": "s",
        "questions": {
            "q": {
                "type": "noul",
                "instructions": { "ask": "Is it urgent?", "note": "consider the tone" }
            }
        }
    }));
    let text = messages_for(&Evidence::Json(OrderedValue::from_json(&json!("s"))), &questions[0]).remove(1).content.text();
    let payload: JsonValue = serde_json::from_str(&text).expect("JSON");
    assert_eq!(payload["criterion"]["ask"], "Is it urgent?");
}

// ── number, point and box: the wire rules (GitHub #242) ─────────────────

/// Prepare one question from raw JSON text, or the refusal it earned.
fn prepare_one_wire(body: &str) -> Result<Vec<PreparedQuestion>, Refusal> {
    prepare(&parse(body).questions, &alphabet(), &encoder)
}

fn program_body(kind: &str, extra: &str) -> String {
    format!(
        r#"{{"state":"s","questions":{{"q":{{"type":"{kind}","instructions":"where?"{extra}}}}}}}"#
    )
}

/// Acceptance 6: `digits` outside 1..=6 is refused, and the refusal names
/// the field.
#[test]
fn digits_outside_the_served_range_is_refused() {
    for digits in ["0", "7", "64"] {
        let refusal = prepare_one_wire(&program_body("number", &format!(r#","digits":{digits}"#)))
            .expect_err("a width this endpoint does not serve");
        assert_eq!(refusal.code, "digits_out_of_range", "{digits}: {}", refusal.message);
        assert!(
            refusal.message.contains("digits"),
            "and names the field: {}",
            refusal.message
        );
    }
    for digits in ["1", "3", "6"] {
        prepare_one_wire(&program_body("number", &format!(r#","digits":{digits}"#)))
            .unwrap_or_else(|e| panic!("{digits} digits is served: {}", e.message));
    }
}

/// The width the caller did not name is the width that was measured.
#[test]
fn a_program_without_digits_takes_the_measured_width() {
    let prepared = prepare_one_wire(&program_body("point", "")).expect("a point");
    assert_eq!(prepared[0].digits, ignis_server::numbers::DEFAULT_DIGITS);
    let plan = prepared[0].plan.as_ref().expect("a point carries a schedule");
    // Two axes of three digits, with the separator's tokens between them.
    assert_eq!(plan.axes.len(), 2);
    assert_eq!(plan.axes[0].digits.len(), 3);
    assert_eq!(plan.axes[1].digits.len(), 3);
}

/// A field this primitive cannot honour is refused, never dropped.
#[test]
fn a_field_the_primitive_cannot_honour_is_refused_not_ignored() {
    let on_a_readout = prepare_one_wire(
        r#"{"state":"s","questions":{"q":{"type":"noul","instructions":"urgent?","digits":3}}}"#,
    )
    .expect_err("a readout has no digits");
    assert_eq!(on_a_readout.code, "digits_unsupported", "{}", on_a_readout.message);

    let on_a_program = prepare_one_wire(&program_body("number", r#","criteria":{"a":"b"}"#))
        .expect_err("a constrained decode declares no options");
    assert_eq!(on_a_program.code, "criteria_unsupported", "{}", on_a_program.message);
}

/// The system text a constrained decode is put under declares the shape and the scale,
/// and its user turn carries the instruction alone.
#[test]
fn a_constrained_questions_prompt_declares_its_shape_and_asks_the_instruction_alone() {
    let prepared = prepare_one_wire(&program_body("point", "")).expect("a point");
    let messages = messages_for(&Evidence::Json(OrderedValue::from_json(&json!("s"))), &prepared[0]);
    let system = messages[0].content.text();
    assert!(
        system.starts_with(&ignis_server::numbers::point_system(3)),
        "the measured point instruction leads the system block: {system}"
    );
    assert!(
        system.contains("\"evidence\""),
        "with the evidence after it, as every decision's is (GitHub #240)"
    );
    let ask: JsonValue =
        serde_json::from_str(&messages[1].content.text()).expect("the user turn is JSON");
    assert_eq!(ask["instruction"], "where?");
    assert!(
        ask.get("options").is_none(),
        "a constrained decode declares no options: the alphabet is forced, not listed"
    );
}

/// Each primitive's forced layout, spelled out where a reader can check it
/// against the finding.
#[test]
fn each_primitive_forces_the_json_shape_its_prompt_declared() {
    let ids = |text: &str| encoder(text).expect("the stand-in tokenizer encodes anything");
    for (kind, prefix, separators) in [
        ("number", "{\"value\":", vec![]),
        ("point", "{\"x\":", vec![",\"y\":"]),
        ("box", "{\"x0\":", vec![",\"y0\":", ",\"x1\":", ",\"y1\":"]),
    ] {
        let prepared = prepare_one_wire(&program_body(kind, r#","digits":2"#))
            .unwrap_or_else(|e| panic!("{kind}: {}", e.message));
        let plan = prepared[0].plan.as_ref().expect("a schedule");
        assert_eq!(plan.prefix, ids(prefix), "{kind}'s opening literal is prompt");
        let forced: usize = separators.iter().map(|s| ids(s).len()).sum();
        let axes = separators.len() + 1;
        assert_eq!(
            plan.schedule.len(),
            axes * 2 + forced,
            "{kind}: two digits per axis plus every separator, one step per token"
        );
        for step in plan.schedule.steps() {
            assert!(
                step.len() == 10 || step.len() == 1,
                "{kind}: a step is the ten digits or one forced token"
            );
        }
    }
}

// ── the scalar (GitHub #255, spec 10) ────────────────────────────────────

fn scalar_body(extra: &str) -> String {
    format!(
        r#"{{"state":"s","questions":{{"q":{{"type":"scalar","instructions":"how much?"{extra}}}}}}}"#
    )
}

/// A caller who does not know the magnitude writes nothing, and gets the
/// ceiling rather than a guess (spec 10).
#[test]
fn a_scalar_needs_no_width_from_its_caller() {
    let prepared = prepare_one_wire(&scalar_body("")).expect("a bare scalar is legal");
    let question = &prepared[0];
    assert_eq!(question.digits, ignis_server::scalar::MAX_DIGITS);
    assert!(question.plan.is_none(), "a scalar has no axis layout");
    let plan = question.scalar.as_ref().expect("its own plan");
    assert_eq!(plan.prefix, encoder(ignis_server::scalar::PREFIX).expect("the opening"));
    assert!(plan.schedule.terminator().is_some(), "the run can close itself");
    assert!(question.options.is_empty(), "a scalar names no options");
}

/// The first step opens a number and the rest may close it — the difference
/// between `{"value":-3.5}` and `{"value":.}`.
#[test]
fn the_terminator_and_the_point_are_not_permitted_where_they_make_nonsense() {
    let prepared = prepare_one_wire(&scalar_body(r#","digits":3"#)).expect("a scalar");
    let plan = prepared[0].scalar.as_ref().expect("a plan");
    let end = plan.schedule.terminator().expect("a terminator");
    let opening = plan.schedule.step(0).expect("a first step");
    assert!(!opening.contains(&end), "a number cannot be nothing at all");
    let rest = plan.schedule.step(1).expect("a second step");
    assert!(rest.contains(&end), "and it can end after one digit");
    assert_eq!(
        opening.len(),
        11,
        "ten digits and a sign: {opening:?}"
    );
    assert_eq!(rest.len(), 12, "ten digits, a point and the brace: {rest:?}");
}

/// A field this primitive cannot honour is refused, never dropped — the
/// same policy `digits` has everywhere else.
#[test]
fn a_scalar_declares_no_options_and_says_so() {
    let refusal = prepare_one_wire(&scalar_body(r#","criteria":{"a":"b"}"#))
        .expect_err("a scalar forces an alphabet");
    assert_eq!(refusal.code, "criteria_unsupported", "{}", refusal.message);

    let wide = prepare_one_wire(&scalar_body(r#","digits":9"#))
        .expect_err("nine digits is past the range this endpoint serves");
    assert_eq!(wide.code, "digits_out_of_range", "{}", wide.message);
    assert!(wide.message.contains("ceiling"), "{}", wide.message);
}

/// The prompt is the scalar's own: it declares a maximum and asks the model
/// to close, and it must not carry #254's padding clause, which would tell
/// the model to fill the very field the terminator exists to avoid.
#[test]
fn a_scalar_is_put_under_its_own_system_text() {
    let prepared = prepare_one_wire(&scalar_body(r#","digits":4"#)).expect("a scalar");
    let state = Evidence::Json(OrderedValue::String("s".to_owned()));
    let messages = messages_for(&state, &prepared[0]);
    let system = match &messages[0].content {
        ignis_server::template::MessageContent::Text(text) => text.clone(),
        other => panic!("a system message is text: {other:?}"),
    };
    assert!(system.contains("at most 4 digits"), "{system}");
    assert!(system.contains("close the object"), "{system}");
    assert!(!system.contains("padded on the left"), "{system}");
}

/// A terminated run is the number before the brace, and the brace is not a
/// digit (acceptance 1 and 5).
#[test]
fn a_scalar_answers_with_what_it_wrote_and_how_sure_it_was() {
    let prepared = prepare_one_wire(&scalar_body(r#","digits":4"#)).expect("a scalar");
    let plan = prepared[0].scalar.as_ref().expect("a plan");
    let ids = |text: &str| encoder(text).expect("the stand-in tokenizer encodes anything")[0];
    let draw = |text: &str, probability: f32| ignis_core::constrained::Draw {
        token: ids(text),
        probability,
    };
    let drawn = [draw("3", 1.0), draw(".", 1.0), draw("5", 0.5), draw("}", 1.0)];
    let Answer::Scalar { value, text, uncertainty, digits } =
        constrained_answer_for(&prepared[0], &drawn, None)
    else {
        panic!("a scalar answers with a scalar");
    };
    assert_eq!(value, 3.5);
    assert_eq!(text, "3.5");
    assert_eq!(digits.len(), 2, "the point and the brace are not digits");
    // Half a unit of doubt one place after the point is 0.05, not 0.5.
    assert!((uncertainty - 0.05).abs() < 1e-6, "{uncertainty}");
    let _ = plan;
}

/// The three ways a run ends, and the one that is an error (spec 10).
#[test]
fn only_a_run_the_engine_cut_short_is_an_error() {
    let prepared = prepare_one_wire(&scalar_body(r#","digits":1"#)).expect("a scalar");
    let ids = |text: &str| encoder(text).expect("encodable")[0];
    let draw = |text: &str| ignis_core::constrained::Draw { token: ids(text), probability: 1.0 };

    // Spent its schedule without closing. A well-formed answer could have
    // closed itself — the schedule has room for the sign, the point and the
    // brace — so this one wrote past its ceiling, and it is named as that
    // rather than as a number or as a cut-off run.
    let over = [draw("1"), draw("2"), draw("3"), draw("4")];
    let Answer::Error { code, .. } = constrained_answer_for(&prepared[0], &over, None) else {
        panic!("a one-digit question cannot answer 1234");
    };
    assert_eq!(code, "too_many_digits");

    // Stopped without closing and short: the engine cut it off.
    let Answer::Error { code, .. } = constrained_answer_for(&prepared[0], &[draw("1")], None)
    else {
        panic!("a cut-off run is an error");
    };
    assert_eq!(code, "run_cut_short");

    // Allowed by the schedule, not a number: refused and never repaired.
    let nonsense = [draw("3"), draw("."), draw("}")];
    let Answer::Error { code, .. } = constrained_answer_for(&prepared[0], &nonsense, None) else {
        panic!("`3.` is not a number");
    };
    assert_eq!(code, "malformed_scalar");
}
