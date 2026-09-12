//! The SSE reader against a **recorded** engine stream (GitHub #137).
//!
//! `tests/fixtures/thinking_tool_call.sse` is not synthetic: it is the raw
//! response body of one tool-calling request against a running
//! `ignis-server` with its default `enable_thinking`, captured during the G4
//! gate run of GitHub #128 (session `g4-20260912T162610Z`). Every token in
//! it arrives on `delta.reasoning_content`, a single `tool_calls` delta
//! follows, and the stream closes on `finish_reason: "tool_calls"` and
//! `[DONE]` — a completely well-formed stream that the reader used to score
//! as a request which generated nothing at all.

use std::io::Cursor;
use std::path::PathBuf;
use std::time::Instant;

use ignis_bench::client::{read_sse_stream, FinishReason};

fn recorded() -> String {
    let path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/thinking_tool_call.sse");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn the_recorded_thinking_turn_is_measured_rather_than_read_as_empty() {
    let sse = recorded();
    let out = read_sse_stream(
        Cursor::new(sse),
        "http://recorded/v1/chat/completions",
        Instant::now(),
        &mut |_| true,
    )
    .expect("the recorded stream parses");

    // 28 thinking tokens went past the old reader unseen, leaving the G4
    // per-class cell to report 0.0 tok/s and `ttft_ms == total_ms`.
    assert_eq!(out.n_tokens, 28, "every recorded token is counted");
    assert_eq!(out.reasoning_tokens, Some(28), "all of them on the thinking channel");
    assert_eq!(out.token_times_ms.len(), 28, "and every one of them is timed");
    assert_eq!(
        out.ttft_ms, out.token_times_ms[0],
        "ttft is the first generated token, not the 'nothing to measure' fallback"
    );

    // The turn never reached the answer: it thought, then called a tool.
    assert!(out.output.is_empty(), "the recording carries no content chunk");
    assert!(
        out.reasoning_output.starts_with("The user is asking"),
        "the thinking text is read back: {:?}",
        out.reasoning_output
    );
    assert_eq!(
        out.finish_reason,
        Some(FinishReason::Engine("tool_calls".into())),
        "the engine's own reason survives the tool-call chunk"
    );
}
