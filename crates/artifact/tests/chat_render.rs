//! The rendering prerequisites of cross-request state reuse (GitHub #184 and
//! #185, spec 01 §Rendering prerequisites, ADR 0029): ignis's minijinja
//! render of the real artifact's chat template equals the reference's render
//! of the same messages, byte for byte — tool-call parameters in the order
//! the model emitted them, and history thinking kept or stripped exactly
//! where the reference keeps or strips it.
//!
//! Fixtures (`tests/fixtures/chat_render/*.json`) were recorded from the
//! reference by `tools/chat-render-fixtures` (see its README). CPU-only;
//! skips when the artifact is not at its machine-local path (same
//! convention as `real_frontend.rs` and `vision_processor.rs`).

use std::path::{Path, PathBuf};

use ignis_artifact::{ChatMessage, FrontendSet, MessageContent, Reader, Role, ToolCall};
use serde_json::Value;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chat_render")
}

fn frontend_or_skip() -> Option<FrontendSet> {
    if !Path::new(ARTIFACT).exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return None;
    }
    let reader = Reader::open(Path::new(ARTIFACT)).expect("open artifact");
    Some(FrontendSet::from_reader(&reader).expect("frontend set"))
}

/// The case's messages as the server hands them to the template. A tool
/// call's `arguments` is the fixture's own wire string, never re-encoded —
/// re-encoding it through `serde_json` is the very defect #184 fixes.
fn messages(case: &Value) -> Vec<ChatMessage> {
    case["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| ChatMessage {
            role: Role::parse(m["role"].as_str().unwrap()).unwrap(),
            content: MessageContent::Text(m["content"].as_str().unwrap().to_owned()),
            tool_calls: m["tool_calls"]
                .as_array()
                .map(|calls| {
                    calls
                        .iter()
                        .map(|c| ToolCall {
                            id: c["id"].as_str().map(str::to_owned),
                            name: c["name"].as_str().unwrap().to_owned(),
                            arguments: c["arguments"].as_str().unwrap().to_owned(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            // Handed over whole; the template decides what survives
            // (GitHub #185).
            reasoning_content: m["reasoning_content"].as_str().map(str::to_owned),
        })
        .collect()
}

#[test]
fn rendered_prompts_match_the_reference_fixtures() {
    let Some(frontend) = frontend_or_skip() else { return };
    let mut dir: Vec<_> = std::fs::read_dir(fixtures())
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    dir.sort();
    assert_eq!(dir.len(), 7, "every recorded fixture is present");
    // Every case is checked and every mismatch reported, so one divergence
    // does not hide another.
    let failures: Vec<String> = dir
        .iter()
        .filter_map(|path| check_fixture(&frontend, path).err())
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} fixtures differ:\n{}",
        failures.len(),
        dir.len(),
        failures.join("\n")
    );
}

fn check_fixture(frontend: &FrontendSet, path: &Path) -> Result<(), String> {
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let case = &fixture["case"];
    let name = case["name"].as_str().unwrap();
    let tools: Vec<Value> = case["tools"].as_array().cloned().unwrap_or_default();
    let thinking = case["enable_thinking"].as_bool().unwrap_or(false);
    // Defaulted the way a request leaves it, which is also the way the
    // recorder engages the reference's option (GitHub #185).
    let preserve = case["preserve_thinking"].as_bool().unwrap_or(false);
    let rendered = frontend
        .chat_template()
        .render_with_thinking_and_tools(&messages(case), thinking, None, preserve, Some(&tools))
        .map_err(|err| format!("{name}: render failed: {err}"))?;
    let expected = fixture["expected"]["text"].as_str().unwrap();
    if rendered == expected {
        return Ok(());
    }
    Err(format!("{name}: {}", first_difference(expected, &rendered)))
}

/// Where two renders part company, with the bytes around it — a diff of
/// two 1.6 KB prompts is unreadable, the byte that differs is not.
fn first_difference(expected: &str, actual: &str) -> String {
    let at = expected
        .bytes()
        .zip(actual.bytes())
        .position(|(e, a)| e != a)
        .unwrap_or_else(|| expected.len().min(actual.len()));
    let from = at.saturating_sub(40);
    let window = |text: &str| {
        let to = (at + 60).min(text.len());
        // A window that starts or ends mid-character is reported by its
        // bytes rather than panicking on a char boundary.
        text.get(from..to).map_or_else(
            || format!("{:?}", &text.as_bytes()[from.min(text.len())..to]),
            str::to_owned,
        )
    };
    format!(
        "diverges at byte {at} (expected {} bytes, got {})\n  reference: {:?}\n  ignis:     {:?}",
        expected.len(),
        actual.len(),
        window(expected),
        window(actual)
    )
}
