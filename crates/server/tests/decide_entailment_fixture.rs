//! The shape of the contributed entailment set (`fixtures/entailment/`),
//! checked on CPU so a broken edit to the fixture fails the default suite
//! rather than the GPU profile. The model's answers are
//! `decide_entailment_gpu.rs`'s.

use std::collections::BTreeMap;

use ignis_server::decide::{DecideRequest, QuestionKind};

#[path = "support/entailment.rs"]
mod entailment;

use entailment::Tier;

#[test]
fn the_set_is_the_contributed_eighteen_cases_with_their_labels() {
    let fixture = entailment::load();
    let ids: Vec<u32> = fixture.cases.iter().map(|case| case.id).collect();
    assert_eq!(ids, (1..=18).collect::<Vec<_>>());
    let expected_true: Vec<u32> =
        fixture.cases.iter().filter(|case| case.expected).map(|case| case.id).collect();
    assert_eq!(expected_true, [1, 3, 10, 12, 13, 16, 17, 18]);
    let starred: Vec<u32> =
        fixture.cases.iter().filter(|case| case.starred).map(|case| case.id).collect();
    assert_eq!(starred, [5, 7, 9, 11, 15]);
}

#[test]
fn every_case_is_a_well_formed_noul_decision() {
    let fixture = entailment::load();
    for case in &fixture.cases {
        let body = fixture.request(case, None);
        let request: DecideRequest = serde_json::from_value(body)
            .unwrap_or_else(|e| panic!("case {}: not a decide request: {e}", case.id));
        let questions = request.questions.entries();
        assert_eq!(questions.len(), 1, "case {}", case.id);
        assert_eq!(questions[0].0, entailment::QUESTION_ID, "case {}", case.id);
        assert_eq!(questions[0].1.kind, QuestionKind::Noul, "case {}", case.id);
        assert!(case.state.starts_with("Claim: "), "case {}", case.id);
        assert!(case.state.contains("\n\nCited evidence: "), "case {}", case.id);
    }
    // The criteria are load-bearing: without them the labels do not hold.
    assert!(fixture.question["criteria"]["true"].is_string());
    assert!(fixture.question["criteria"]["false"].is_string());
}

/// 1/2, 3/4 and 13/14 are one claim against opposite evidence: a model that
/// answers from the claim's wording alone fails them. The correction to 13
/// was applied to 14 as well, so the pair still differs only in the date.
#[test]
fn the_inversion_pairs_share_a_claim_and_split_the_label() {
    let fixture = entailment::load();
    let by_id: BTreeMap<u32, _> = fixture.cases.iter().map(|case| (case.id, case)).collect();
    for (a, b) in [(1, 2), (3, 4), (13, 14)] {
        let (a, b) = (by_id[&a], by_id[&b]);
        let claim = |state: &str| state.lines().next().unwrap_or_default().to_owned();
        assert_eq!(claim(&a.state), claim(&b.state), "pair {}/{}", a.id, b.id);
        assert_ne!(a.expected, b.expected, "pair {}/{}", a.id, b.id);
    }
    assert!(by_id[&13].state.contains("as_of_date = 2026-09-20"));
    assert!(by_id[&14].state.contains("as_of_date = 2026-09-20"));
}

/// The tiers are what the GPU run asserts; case 13 is a known failure of
/// the model, kept at its correct label rather than relabelled.
#[test]
fn only_gate_cases_are_asserted_and_thirteen_is_a_known_failure() {
    let fixture = entailment::load();
    let tier = |tier| -> Vec<u32> {
        fixture.cases.iter().filter(|case| case.tier == tier).map(|case| case.id).collect()
    };
    assert_eq!(tier(Tier::KnownFailure), [13]);
    assert_eq!(tier(Tier::Watch), [4, 12, 16, 18]);
    assert_eq!(tier(Tier::Gate).len(), 13);
    assert!(fixture.cases.iter().all(|case| !case.starred || case.tier == Tier::Gate));
}

/// Every mart, key and field a case cites is one of the public names. The
/// public side is the one enumerated, so a name nobody listed fails rather
/// than slipping through. A line mixing vocabularies also reads differently
/// to the model (case 10 went from p(yes) 0.996 to 0.119), so a stray name
/// is a broken case, not a style slip.
#[test]
fn every_cited_name_is_a_public_one() {
    const PUBLIC: &[&str] = &[
        "plan_health", "stakeholder_map", "whitespace_opportunity", "plan_header",
        "plan_id", "account_name", "initiative_name",
        "high_value_potentials_count", "potentials_count", "linked_opportunities_count",
        "tasks_open_count", "tasks_in_progress_count", "tasks_total_count",
        "days_since_scorecard_update", "days_since_whitespace_update", "days_since_map_update",
        "stakeholder_map_last_modified_date", "as_of_date", "renewal_date", "plan_status",
        "scorecard_questions_answered", "scorecard_questions_total",
        "champion_identified", "exec_sponsor_engaged",
    ];
    let fixture = entailment::load();
    for case in &fixture.cases {
        let (_, evidence) = case.state.split_once("\n\n").expect("a claim, then the evidence");
        let words = evidence.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'));
        for word in words.filter(|word| !word.is_empty()) {
            if word.contains('_') {
                assert!(PUBLIC.contains(&word), "case {}: {word} is not a public name", case.id);
            }
            if word.starts_with(|c: char| c.is_ascii_uppercase()) && word.contains('-') {
                assert!(word.starts_with("PLAN-"), "case {}: {word} is not a public id", case.id);
            }
        }
    }
}
