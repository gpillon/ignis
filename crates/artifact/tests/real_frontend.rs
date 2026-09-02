//! Integration test against the real Qwen 3.8-27B `nvfp4full` artifact
//! (the 1,325-object container, ~19 GB): frontend object extraction
//! (artifact-02, GitHub #7). Skips gracefully when the file is not at its
//! machine-local path (same convention as `real_artifact.rs`).
//!
//! Cheap by design: a directory walk + six resource spans (no full
//! materialization, no GPU) — safe to run while the reference runner
//! holds the RTX 5090 (ADR 0006).

use std::path::Path;

use ignis_artifact::{ChatMessage, FrontendSet, Object, Reader, Role, FRONTEND_RESOURCES};

/// The fork-local model cache (the artifact the running `ninfer-serve`
/// loads).
const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

fn open_or_skip() -> Option<Reader> {
    let path = Path::new(ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return None;
    }
    Some(Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}")))
}

#[test]
fn real_nvfp4full_frontend_extraction() {
    let reader = match open_or_skip() {
        Some(r) => r,
        None => return,
    };

    // Acceptance: all six frontend resources are present in the container
    // (a missing resource is a load failure, so `from_reader` succeeding
    // is the round-trip proof).
    for base in FRONTEND_RESOURCES {
        let found = reader
            .objects()
            .iter()
            .any(|object| matches!(object, Object::Resource(r) if r.name.contains(base)));
        assert!(found, "missing frontend resource: {base}");
    }

    let set = FrontendSet::from_reader(&reader).expect("frontend set");

    // The four config resources round-trip non-empty raw bytes.
    assert!(!set.tokenizer_config().is_empty(), "tokenizer_config empty");
    assert!(
        !set.generation_config().is_empty(),
        "generation_config empty"
    );
    assert!(
        !set.preprocessor_config().is_empty(),
        "preprocessor_config empty"
    );
    assert!(
        !set.video_preprocessor_config().is_empty(),
        "video_preprocessor_config empty"
    );

    // Tokenizer: the real BPE tokenizer parses, and a known string
    // round-trips exactly (encode -> decode).
    let ids = set.tokenizer().encode("hello world").expect("encode");
    assert!(!ids.is_empty(), "BPE tokenizer must emit ids");
    let text = set.tokenizer().decode(&ids).expect("decode");
    assert_eq!(text, "hello world", "BPE round-trip must be exact");

    // Chat template: the real Qwen template compiles once and renders a
    // two-message conversation (property assertions, not an exact
    // snapshot: role markers + contents + the closing generation prompt).
    let messages = [
        ChatMessage::text(Role::User, "hello world"),
        ChatMessage::text(Role::Assistant, "hi there"),
    ];
    let prompt = set.chat_template().render(&messages).expect("render");
    assert!(
        prompt.contains("hello world"),
        "user content missing from prompt"
    );
    assert!(
        prompt.contains("hi there"),
        "assistant content missing from prompt"
    );
    assert!(
        prompt.contains(
            "
"
        ),
        "
 role marker missing from prompt"
    );
    assert!(
        prompt.ends_with(
            "
"
        ),
        "closing generation prompt (
 marker) missing"
    );

    // The template's own error path: an empty message list raises
    // (a clean abort — not a panic, not a silent empty render).
    let err = set
        .chat_template()
        .render(&[])
        .expect_err("the real template must raise on an empty message list");
    assert!(err.to_string().contains("No messages provided."), "{err}");
}
