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
    Answer, DecideRequest, Evidence, MAX_OPTIONS, Ordered, PreparedQuestion, Question,
    QuestionKind, answer_for, expected_level, messages_for, prepare, score_confidence,
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
    prepare(&parse(body).questions, &alphabet()).expect("a valid request prepares")
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
    assert_eq!(questions[0].instructions, json!("Urgent?"));
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
        let messages = messages_for(&Evidence::Json(json!("evidence")), &questions[0]);
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
    let at_ceiling = prepare(&parse(&wide_choice(MAX_OPTIONS)).questions, &alphabet());
    assert!(
        at_ceiling.is_ok(),
        "{MAX_OPTIONS} options is measured and served: {:?}",
        at_ceiling.err()
    );
    let past = prepare(&parse(&wide_choice(MAX_OPTIONS + 1)).questions, &alphabet())
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
    let refusal = prepare(&parse(&wide_choice(3)).questions, &narrow)
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
            json!({ "bad": { "type": "noul", "instructions": "   " } }),
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
        let refusal = prepare(&parse(&body).questions, &alphabet())
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
    let refusal = prepare(&Ordered(Vec::<(String, Question)>::new()), &alphabet())
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
    let refusal = prepare(&body.questions, &alphabet()).expect_err("ambiguous");
    assert_eq!(refusal.code, "duplicate_question");
}

// ── the evidence (acceptance 8's wire half) ──────────────────────────────

#[test]
fn a_state_that_is_content_parts_is_read_as_content_parts() {
    let state = json!([
        { "type": "image_url", "image_url": { "url": "https://example.test/a.png" } },
        { "type": "text", "text": "the receipt above" }
    ]);
    let evidence = Evidence::read(&state);
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
        let evidence = Evidence::read(&state);
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
    let messages = messages_for(&Evidence::read(&state), &questions[0]);

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
fn text_evidence_leads_the_payload() {
    let questions = prepared(JEV_CHOICE);
    let messages = messages_for(
        &Evidence::read(&json!("Help! My payouts have been failing for 3 days.")),
        &questions[0],
    );
    let text = messages[1].content.text();
    let payload: JsonValue = serde_json::from_str(&text).expect("the payload is JSON");
    assert_eq!(payload["evidence"], "Help! My payouts have been failing for 3 days.");
    assert_eq!(payload["criterion"], "Which team should handle this?");
    assert_eq!(payload["options"][1]["letter"], "B");
    assert_eq!(payload["options"][1]["description"], "Bugs, outages, integrations");
    assert!(
        text.find("evidence").unwrap() < text.find("criterion").unwrap(),
        "evidence first, which is what makes one state a shared prefix"
    );
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
    let text = messages_for(&Evidence::Json(json!("s")), &questions[0]).remove(1).content.text();
    let payload: JsonValue = serde_json::from_str(&text).expect("JSON");
    assert_eq!(payload["criterion"]["ask"], "Is it urgent?");
}
