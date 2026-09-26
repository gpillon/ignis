//! GitHub #270 (spec 16) — a decision's reuse boundaries are cut where its
//! `state`'s content parts end, and the prompt the model reads does not change
//! by a byte for it.
//!
//! Only the real chat template can say either: `SimpleTemplateProvider`'s
//! "template" is a word hash. Real artifact frontend and vision processor, CPU
//! only; skips when the artifact is not at its machine-local path (the
//! `real_frontend.rs` convention).

use std::path::Path;

use ignis_artifact::{FrontendSet, Reader};
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::decide::{DIRECT_SYSTEM, DecideRequest, Evidence, messages_for, prepare};
use ignis_server::template::TemplateProvider;
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

const RULES: &str = "Rule 1: when you see a door, consider what it means for reaching the exit.";
const SITUATION: &str = "NOW: health 80, ammo 12.";
const QUESTION: &str = "Is there a monster in the view?";

/// A one-question decision over `state`, as the endpoint would read it.
fn request(state: &str) -> DecideRequest {
    serde_json::from_str(&format!(
        r#"{{"state":{state},"questions":{{"q":{{"type":"noul","instructions":"{QUESTION}"}}}}}}"#
    ))
    .expect("the request parses")
}

/// `[rules][situation]`, the rules part marked or not.
fn text_state(marked: bool) -> String {
    let marker = if marked { r#","cache_control":{"type":"ephemeral"}"# } else { "" };
    format!(
        r#"[{{"type":"text","text":"{RULES}"{marker}}},{{"type":"text","text":"{SITUATION}"}}]"#
    )
}

fn thinking_off() -> ThinkingOptions {
    ThinkingOptions {
        enable_thinking: false,
        ..ThinkingOptions::default()
    }
}

#[test]
fn a_marker_changes_no_byte_of_the_prompt_and_the_prompt_is_the_one_measured() {
    let Some(frontend) = frontend_or_skip() else {
        return;
    };
    let provider = ArtifactTemplateProvider::new(frontend);
    let alphabet = provider.answer_alphabet();
    let encode = |text: &str| provider.encode_literal(text);
    let render = |state: &str| {
        let request = request(state);
        let prepared = prepare(&request.questions, &alphabet, &encode, None).expect("a valid decision");
        let messages = messages_for(&Evidence::read(&request.state), &prepared[0]);
        provider.render_text(&messages, &thinking_off(), &[]).expect("render")
    };

    let plain = render(&text_state(false));
    assert_eq!(render(&text_state(true)), plain, "a marker is not rendered");
    // Nor on an image part, which renders as its placeholder either way.
    let with_image = |marker: &str| {
        format!(
            r#"[{{"type":"text","text":"{RULES}"}},{{"type":"image_url","image_url":{{"url":"data:image/png;base64,"}}{marker}}}]"#
        )
    };
    assert_eq!(
        render(&with_image(r#","cache_control":{"type":"ephemeral"}"#)),
        render(&with_image("")),
        "a marked image renders as an unmarked one"
    );
    // The layout #240 measured for a parts state: the instruction alone in
    // the system block, the parts in the user turn joined by a line break,
    // the question after them — written out by hand, not re-derived.
    let ask = r#"{"criterion":"Is there a monster in the view?","options":[{"description":"Yes","letter":"A"},{"description":"No","letter":"B"}]}"#;
    let expected = format!(
        "<|im_start|>system\n{DIRECT_SYSTEM}<|im_end|>\n<|im_start|>user\n{RULES}\n{SITUATION}\n{ask}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    assert_eq!(plain, expected);
}

#[test]
fn each_parts_end_is_an_exact_token_prefix_where_the_part_ends() {
    let Some(frontend) = frontend_or_skip() else {
        return;
    };
    let tokenizer_set = FrontendSet::from_reader(&Reader::open(Path::new(ARTIFACT)).unwrap()).unwrap();
    let tokenizer = tokenizer_set.tokenizer();
    let provider = ArtifactTemplateProvider::new(frontend);
    let alphabet = provider.answer_alphabet();
    let encode = |text: &str| provider.encode_literal(text);
    let request = request(&text_state(false));
    let prepared = prepare(&request.questions, &alphabet, &encode, None).expect("a valid decision");
    let messages = messages_for(&Evidence::read(&request.state), &prepared[0]);

    let rendered = provider
        .apply_chat_template_with_part_ends(&messages, &thinking_off(), &[])
        .expect("render");
    assert_eq!(rendered.part_ends.len(), 2, "the two state parts; the question ends the message");
    let head = format!("<|im_start|>system\n{DIRECT_SYSTEM}<|im_end|>\n<|im_start|>user\n");
    let expected = [format!("{head}{RULES}"), format!("{head}{RULES}\n{SITUATION}")]
        .map(|text| tokenizer.encode(&text).expect("encode").len() as u32);
    assert_eq!(rendered.part_ends[..2], expected.map(Some), "each part's own end, in tokens");

    // A plain render reports none: only a decision pays for them.
    let plain = provider.apply_chat_template(&messages, &thinking_off(), &[]).expect("render");
    assert!(plain.part_ends.is_empty());
    assert_eq!(plain.tokens, rendered.tokens, "and the prompt is the same");
}

#[test]
fn an_image_parts_end_is_past_its_whole_placeholder_run() {
    let Some(frontend) = frontend_or_skip() else {
        return;
    };
    let processor = frontend.vision_processor().expect("vision processor");
    let provider = ArtifactTemplateProvider::new(frontend).with_vision(processor.clone());
    let alphabet = provider.answer_alphabet();
    let encode = |text: &str| provider.encode_literal(text);
    let state = format!(
        r#"[{{"type":"text","text":"{RULES}"}},{{"type":"image_url","image_url":{{"url":"data:image/png;base64,"}}}},{{"type":"text","text":"{SITUATION}"}}]"#
    );
    let request = request(&state);
    let prepared = prepare(&request.questions, &alphabet, &encode, None).expect("a valid decision");
    let messages = messages_for(&Evidence::read(&request.state), &prepared[0]);
    let media = processor.prepare_media(0, &media::png(256, 256)).expect("prepare");

    let (rendered, multimodal) = provider
        .prepare_multimodal_with_part_ends(&messages, &thinking_off(), &[], vec![media])
        .expect("prepare the multimodal prompt");
    let item = multimodal.media[0].token_span;
    let ends: Vec<u32> = rendered.part_ends.iter().map(|end| end.expect("every end is exact")).collect();
    assert_eq!(ends.len(), 3, "the three state parts; the question ends the message");
    assert!(ends[0] as usize <= item.begin, "the text before the image ends before it");
    assert_eq!(
        ends[1] as usize,
        item.begin + item.count + 1,
        "the image part ends past its run and the `<|vision_end|>` closing it"
    );
    assert!(ends[1] < ends[2], "and the part after it follows");
}
