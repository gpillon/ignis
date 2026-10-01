//! The contributed evidence-to-claim entailment set
//! (`fixtures/entailment/`), shared by its CPU shape check
//! (`decide_entailment_fixture.rs`) and its GPU run
//! (`decide_entailment_gpu.rs`). Included via `#[path]`, like `media.rs`.

// Each binary reads a different subset (only the GPU run prints `name`).
#![allow(dead_code)]

use std::path::PathBuf;

use serde::Deserialize;
use serde_json::{Value as JsonValue, json};

/// What a case's label means to the GPU run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// The label is asserted: measured at least one logit from the 0.5 line.
    Gate,
    /// Right when measured, but within a logit of the line, where a
    /// quantisation or artifact change can flip it. Printed, not asserted.
    Watch,
    /// The label is right and the model gets it wrong. Printed, not
    /// asserted; one that starts passing is a candidate for `gate`.
    KnownFailure,
}

#[derive(Debug, Deserialize)]
pub struct Case {
    pub id: u32,
    pub name: String,
    pub expected: bool,
    pub starred: bool,
    pub tier: Tier,
    pub state: String,
    pub note: String,
}

#[derive(Debug, Deserialize)]
pub struct Fixture {
    /// The one `noul` question every case asks, `criteria` included: the
    /// labels are defined relative to its definition of "support".
    pub question: JsonValue,
    pub cases: Vec<Case>,
}

/// The id the question is asked under, in the request and the answer.
pub const QUESTION_ID: &str = "supported";

pub fn load() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("entailment")
        .join("cases.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
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
