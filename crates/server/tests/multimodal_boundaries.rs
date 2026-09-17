//! GitHub #193 — a prompt carrying images reports the structural boundaries
//! cross-request reuse is cut at, exactly as a text prompt does, counted over
//! its **expanded** tokens: the system block before the image where the text
//! render puts it, the generation opener and the last user query moved past
//! the image by its placeholder run.
//!
//! Real artifact frontend and vision processor, CPU only; skips when the
//! artifact is not at its machine-local path (the `real_frontend.rs`
//! convention).

use std::path::Path;

use ignis_artifact::{FrontendSet, Reader};
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::template::{ChatMessage, TemplateProvider};
use ignis_server::thinking::ThinkingOptions;

#[path = "support/media.rs"]
#[allow(dead_code)]
mod media;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

fn frontend_or_skip() -> Option<FrontendSet> {
    if !Path::new(ARTIFACT).exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return None;
    }
    let reader = Reader::open(Path::new(ARTIFACT)).expect("open artifact");
    Some(FrontendSet::from_reader(&reader).expect("frontend set"))
}

fn conversation() -> Vec<ChatMessage> {
    serde_json::from_value(serde_json::json!([
        {"role": "system", "content": "You are a careful assistant that describes pictures."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
            {"type": "text", "text": "What colour is this image?"}
        ]}
    ]))
    .expect("messages")
}

#[test]
fn a_multimodal_prompt_reports_its_boundaries_over_the_expanded_tokens() {
    let Some(frontend) = frontend_or_skip() else {
        return;
    };
    let processor = frontend.vision_processor().expect("vision processor");
    let text = ArtifactTemplateProvider::new(FrontendSet::from_reader(&Reader::open(Path::new(ARTIFACT)).unwrap()).unwrap());
    let provider = ArtifactTemplateProvider::new(frontend).with_vision(processor.clone());
    let messages = conversation();
    let options = ThinkingOptions::default();

    // The same conversation on the text path renders the image as its one
    // placeholder token: the reference layout the expansion moves.
    let unexpanded = text
        .apply_chat_template(&messages, &options, &[])
        .expect("render the text prompt");
    let media = processor.prepare_media(0, &media::png(256, 256)).expect("prepare");
    let (rendered, multimodal) = provider
        .prepare_multimodal(&messages, &options, &[], vec![media])
        .expect("prepare the multimodal prompt");
    let item = multimodal.media[0].token_span;
    let growth = item.count as u32 - 1;

    let block = rendered.system_block_tokens.expect("a system block");
    let opener = rendered.opener_tokens.expect("a generation opener");
    let user = rendered.user_turn_tokens.expect("a user query");
    assert!(block as usize <= item.begin, "the block ends before the image");
    assert!(user as usize <= item.begin, "the query begins before the image");
    assert!(opener as usize > item.begin + item.count, "the opener is after it");
    assert_eq!(Some(block), unexpanded.system_block_tokens, "nothing before it moves");
    assert_eq!(Some(user), unexpanded.user_turn_tokens);
    assert_eq!(Some(opener - growth), unexpanded.opener_tokens, "the opener moves by the run");
    assert_eq!(
        rendered.tokens[..block as usize],
        unexpanded.tokens[..block as usize],
        "and the block is the same ids a text request's is"
    );
}
