//! The contributed evidence-to-claim entailment set
//! (`fixtures/entailment/`), shared by its CPU shape check
//! (`decide_entailment_fixture.rs`) and its GPU run
//! (`decide_entailment_gpu.rs`). Included via `#[path]`, like `media.rs`.

// Each binary reads a different subset (only the GPU run prints `name`).
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::{Value as JsonValue, json};

/// What a case's label means to the GPU run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// The label is asserted: right by at least one logit on every arm
    /// the fixture's `measurement` lists.
    Gate,
    /// Not clear of the 0.5 line on every arm, where a change of the
    /// numerics can flip it. Printed, not asserted.
    Watch,
    /// The label is right and the model gets it wrong on every arm.
    /// Printed, not asserted; one that starts passing is a candidate for
    /// `gate`.
    KnownFailure,
}

#[derive(Debug, Deserialize)]
pub struct Case {
    pub id: u32,
    pub name: String,
    pub expected: bool,
    pub starred: bool,
    pub tier: Tier,
    /// The case's logit on each of the measurement's arms, by arm name.
    pub logits: BTreeMap<String, f64>,
    pub state: String,
    pub note: String,
}

/// One route the tiers were measured on.
#[derive(Debug, Deserialize)]
pub struct Arm {
    pub name: String,
    pub surface: String,
    pub gdn_prefill: String,
    pub prefill_attention: String,
    /// What to run to redo the arm.
    pub redo: String,
}

/// The one measurement the tiers come from (see `NOTICE.md`).
#[derive(Debug, Deserialize)]
pub struct Measurement {
    pub date: String,
    pub build: String,
    pub card: String,
    pub rule: String,
    pub logit: String,
    pub arms: Vec<Arm>,
}

/// A logit this far on the right side of the line makes a gate case.
pub const GATE_MARGIN: f64 = 1.0;

#[derive(Debug, Deserialize)]
pub struct Fixture {
    pub measurement: Measurement,
    /// The one `noul` question every case asks, `criteria` included: the
    /// labels are defined relative to its definition of "support".
    pub question: JsonValue,
    pub cases: Vec<Case>,
}

/// The id the question is asked under, in the request and the answer.
pub const QUESTION_ID: &str = "supported";

/// The tier the rule gives a case from its logits: a gate when the minimum
/// signed margin over the arms is at least `GATE_MARGIN`, a known failure
/// when it is wrong on every arm, a watch case otherwise.
pub fn tier_by_rule(expected: bool, logits: &[f64]) -> Tier {
    let margins: Vec<f64> =
        logits.iter().map(|logit| if expected { *logit } else { -*logit }).collect();
    if margins.iter().all(|margin| *margin >= GATE_MARGIN) {
        Tier::Gate
    } else if margins.iter().all(|margin| *margin <= 0.0) {
        Tier::KnownFailure
    } else {
        Tier::Watch
    }
}

/// Parse the fixture and check it: every case has a tier and a logit on
/// every arm, its tier is the one the rule gives from those logits, and
/// the gate holds a case of each label.
pub fn parse(text: &str) -> Result<Fixture, String> {
    let fixture: Fixture = serde_json::from_str(text).map_err(|e| format!("parse: {e}"))?;
    let arms: Vec<&str> = fixture.measurement.arms.iter().map(|arm| arm.name.as_str()).collect();
    for case in &fixture.cases {
        let mut logits = Vec::new();
        for arm in &arms {
            let logit = case.logits.get(*arm);
            logits.push(*logit.ok_or_else(|| format!("case {}: no logit on arm {arm}", case.id))?);
        }
        if case.logits.len() != arms.len() {
            return Err(format!("case {}: a logit on an arm the measurement lacks", case.id));
        }
        let by_rule = tier_by_rule(case.expected, &logits);
        if case.tier != by_rule {
            return Err(format!(
                "case {}: tier {:?}, but the rule gives {by_rule:?} from its logits",
                case.id, case.tier
            ));
        }
    }
    for label in [true, false] {
        if !fixture.cases.iter().any(|c| c.tier == Tier::Gate && c.expected == label) {
            return Err(format!("no gate case has the label {label}"));
        }
    }
    Ok(fixture)
}

pub fn path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("entailment")
        .join("cases.json")
}

pub fn load() -> Fixture {
    let path = path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    parse(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

impl Fixture {
    /// The `/v1/decide` body for one case.
    pub fn request(&self, case: &Case, model: Option<&str>) -> JsonValue {
        let mut body = json!({
            "state": case.state,
            "questions": { QUESTION_ID: self.question },
        });
        if let Some(model) = model {
            body["model"] = json!(model);
        }
        body
    }
}
