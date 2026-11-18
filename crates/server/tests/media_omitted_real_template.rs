//! A dropped image on the **real frontend** (GitHub #673, ADR 0102): the
//! part it was, rewritten as the omission text, renders through the 27B's own
//! chat template as that text where the image was, in a user message and in a
//! tool result alike, and the kept image still renders as its placeholder.
//!
//! The rule and the prefix it keeps are pinned over the placeholder template
//! in `media_budget_reuse.rs`; what only the artifact's template can show is
//! that it renders a text part wherever it renders an image part.
//!
//! CPU-only. Skips when the artifact is not at its machine-local path, the
//! convention `checkpoint_lineage.rs` follows.

use std::path::Path;

use serde_json::json;

use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::media::{AcquiredMedia, MediaStats, OMITTED_IMAGE};
use ignis_server::template::ChatMessage;
use ignis_server::thinking::ThinkingOptions;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

fn provider_or_skip() -> Option<ArtifactTemplateProvider> {
    if !Path::new(ARTIFACT).exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return None;
    }
    let reader = ignis_artifact::Reader::open(Path::new(ARTIFACT)).expect("open artifact");
    let frontend = ignis_artifact::FrontendSet::from_reader(&reader).expect("frontend set");
    Some(ArtifactTemplateProvider::new(frontend))
}

#[test]
fn a_dropped_image_renders_as_its_text_in_a_user_message_and_in_a_tool_result() {
    let Some(provider) = provider_or_skip() else {
        return;
    };
    let image = |name: &str| json!({"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{name}")}});
    let messages: Vec<ChatMessage> = serde_json::from_value(json!([
        {"role": "system", "content": "You read screenshots."},
        {"role": "user", "content": [{"type": "text", "text": "Before:"}, image("AAAA"), {"type": "text", "text": "After."}]},
        {"role": "assistant", "content": null, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "crop", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "Cropped:"}, image("BBBB")]},
        {"role": "user", "content": [{"type": "text", "text": "And now?"}, image("CCCC")]},
    ]))
    .unwrap();
    let options = ThinkingOptions { enable_thinking: false, ..ThinkingOptions::default() };
    let whole = provider.render_text(&messages, &options, &[]).expect("render");
    assert_eq!(whole.matches("<|image_pad|>").count(), 3, "{whole}");

    // The two oldest dropped, the newest kept.
    let acquired = AcquiredMedia { media: Vec::new(), stats: MediaStats::default(), dropped: 2 };
    let text = provider.render_text(&acquired.omit_dropped(&messages), &options, &[]).expect("render");
    assert_eq!(text.matches("<|image_pad|>").count(), 1, "{text}");
    assert_eq!(text.matches(OMITTED_IMAGE).count(), 2, "{text}");
    // Each where its image was. The template puts a text part on a line of
    // its own, so the prompt is unchanged up to the first dropped image and
    // goes on from there with the text on its line.
    let block = "<|vision_start|><|image_pad|><|vision_end|>";
    let user = whole.find(block).expect("the user's image");
    assert_eq!(text[..user], whole[..user], "unchanged before the first dropped image");
    assert!(text[user..].starts_with(&format!("\n{OMITTED_IMAGE}\nAfter.")), "{text}");
    let tool = text.find("Cropped:").expect("the tool result") + "Cropped:".len();
    assert!(text[tool..].starts_with(&format!("\n{OMITTED_IMAGE}\n</tool_response>")), "{text}");
    assert!(text.ends_with(&whole[whole.rfind("And now?").unwrap()..]), "the kept image renders as before:\n{text}");
}
