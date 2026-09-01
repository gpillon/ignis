//! ignis-server: OpenAI-compatible HTTP entrypoint (localhost, no auth).
//!
//! v1 endpoints: `/v1/models`, `/v1/chat/completions`, `/v1/responses`
//! (docs/design/ignis-v1.md §4). Telemetry: JSONL events + interval lines.

fn main() {
    // v1: model load (ignis-artifact), engine init (ignis-core), HTTP + telemetry.
    eprintln!("ignis-server: not implemented yet (see docs/design/ignis-v1.md)");
}