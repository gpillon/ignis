//! Seam A of the vision spec (GitHub #176): the prepared prompt of the real
//! artifact's frontend equals the reference processor's, exactly, on the
//! committed image set — token ids, token types, positions, `rope_delta`,
//! grids, token spans, the SHA-256 of the packed BF16 patch rows, and the
//! rewrite-checkpoint frontier carried across the expansion.
//!
//! Fixtures (`tests/fixtures/vision/expected/*.json`) were recorded from the
//! reference by `tools/vision-fixtures` (see its README). CPU-only; skips
//! when the artifact is not at its machine-local path (same convention as
//! `real_frontend.rs`).

use std::path::{Path, PathBuf};

use ignis_artifact::vision::{layout, ProcessorError};
use ignis_artifact::{ChatMessage, ContentPart, FrontendSet, MessageContent, Reader, Role, ToolCall};
use serde_json::Value;
use sha2::{Digest, Sha256};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision")
}

fn frontend_or_skip() -> Option<FrontendSet> {
    if !Path::new(ARTIFACT).exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return None;
    }
    let reader = Reader::open(Path::new(ARTIFACT)).expect("open artifact");
    Some(FrontendSet::from_reader(&reader).expect("frontend set"))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The case's messages as the server hands them to the template, plus the
/// image bytes in prompt order.
fn messages(case: &Value) -> (Vec<ChatMessage>, Vec<Vec<u8>>) {
    let mut images = Vec::new();
    let messages = case["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            let role = Role::parse(m["role"].as_str().unwrap()).unwrap();
            let content = match &m["content"] {
                Value::String(text) => MessageContent::Text(text.clone()),
                Value::Array(parts) => MessageContent::Parts(
                    parts
                        .iter()
                        .map(|p| match p["type"].as_str().unwrap() {
                            "text" => ContentPart::Text(p["text"].as_str().unwrap().to_owned()),
                            "image" => {
                                let file = p["file"].as_str().unwrap();
                                images.push(std::fs::read(fixtures().join("images").join(file)).unwrap());
                                ContentPart::Image { url: Some(file.to_owned()) }
                            }
                            other => panic!("unknown part {other}"),
                        })
                        .collect(),
                ),
                other => panic!("bad content {other}"),
            };
            let tool_calls = m["tool_calls"]
                .as_array()
                .map(|calls| {
                    calls
                        .iter()
                        .map(|c| ToolCall {
                            id: c["id"].as_str().map(str::to_owned),
                            name: c["name"].as_str().unwrap().to_owned(),
                            arguments: c["arguments"].clone(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            // The server drops history reasoning unless preserve_thinking.
            let reasoning_content = case["preserve_thinking"]
                .as_bool()
                .unwrap_or(false)
                .then(|| m["reasoning_content"].as_str().map(str::to_owned))
                .flatten();
            ChatMessage { role, content, tool_calls, reasoning_content }
        })
        .collect();
    (messages, images)
}

/// The reference's turn-closure rewrite checkpoint (`chat_template.cpp`):
/// just after the header of the first assistant turn following the last
/// real user query, else after the generation prompt's header.
fn turn_closure_offset(rendered: &str) -> usize {
    const USER: &str = "<|im_start|>user\n";
    const ASSISTANT: &str = "<|im_start|>assistant\n";
    let last_query = rendered
        .match_indices(USER)
        .map(|(at, _)| at)
        .filter(|&at| !rendered[at + USER.len()..].starts_with("<tool_response>"))
        .last()
        .expect("a user query");
    last_query + rendered[last_query..].find(ASSISTANT).expect("an assistant header") + ASSISTANT.len()
}

fn as_vec<T: TryFrom<i64>>(value: &Value) -> Vec<T> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| T::try_from(v.as_i64().unwrap()).ok().expect("value in range"))
        .collect()
}

#[test]
fn prepared_prompts_match_the_reference_fixtures() {
    let Some(frontend) = frontend_or_skip() else { return };
    let processor = frontend.vision_processor().expect("vision processor");
    let mut dir: Vec<_> = std::fs::read_dir(fixtures().join("expected")).unwrap().map(|e| e.unwrap().path()).collect();
    dir.sort();
    assert_eq!(dir.len(), 18, "every recorded fixture is present");
    // Every case is checked and every mismatch reported, so one divergence
    // does not hide another.
    let failures: Vec<String> = dir.iter().filter_map(|path| check_fixture(&frontend, &processor, path).err()).collect();
    assert!(failures.is_empty(), "{} of {} fixtures differ:\n{}", failures.len(), dir.len(), failures.join("\n"));
}

fn check_fixture(
    frontend: &FrontendSet,
    processor: &ignis_artifact::vision::VisionProcessor,
    path: &Path,
) -> Result<(), String> {
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let case = &fixture["case"];
    let name = case["name"].as_str().unwrap();
    let (messages, images) = messages(case);
    let thinking = case["enable_thinking"].as_bool().unwrap_or(false);
    let rendered = frontend.chat_template().render_with_thinking(&messages, thinking, None).expect("render");
    let refs: Vec<&[u8]> = images.iter().map(Vec::as_slice).collect();
    let boundary = turn_closure_offset(&rendered);
    let result = processor.prepare(frontend.tokenizer(), &rendered, &refs, &[boundary]);
    let check = |ok: bool, what: String| if ok { Ok(()) } else { Err(format!("{name}: {what}")) };

    if let Some(reference) = fixture.get("error") {
        return match result {
            Ok(_) => Err(format!("{name}: accepted, the reference refused it: {reference}")),
            Err(error) => check(error.code() == "invalid_media", format!("{error} is not invalid_media")),
        };
    }
    let prompt = result.map_err(|e| format!("{name}: refused, the reference accepted it: {e}"))?;
    let expected = &fixture["expected"];
    let ids = as_vec::<u32>(&expected["token_ids"]);
    if prompt.token_ids != ids {
        let first = prompt.token_ids.iter().zip(&ids).position(|(a, b)| a != b);
        return Err(format!(
            "{name}: token ids differ (ours {}, ref {}, first difference at {first:?})",
            prompt.token_ids.len(),
            ids.len()
        ));
    }
    check(prompt.token_types == as_vec::<u8>(&expected["token_types"]), "token types".into())?;
    for axis in 0..3 {
        let want = as_vec::<i32>(&expected["positions"][axis]);
        check(prompt.position_axis(axis) == want.as_slice(), format!("position axis {axis}"))?;
    }
    let delta = expected["rope_delta"].as_i64().unwrap() as i32;
    check(prompt.rope_delta == delta, format!("rope_delta {} vs {delta}", prompt.rope_delta))?;
    let items = expected["items"].as_array().unwrap();
    check(prompt.media.len() == items.len(), "media item count".into())?;
    let decoded = fixture["decoded"].as_array().unwrap();
    for (i, (item, want)) in prompt.media.iter().zip(items).enumerate() {
        let grid = as_vec::<u32>(&want["grid"]);
        check([item.grid.t, item.grid.h, item.grid.w] == grid.as_slice(), format!("[{i}] grid {:?}", item.grid))?;
        let span = as_vec::<usize>(&want["token_spans"][0]);
        check([item.token_span.begin, item.token_span.count] == span.as_slice(), format!("[{i}] span"))?;
        check(hex(&item.content_digest) == want["content_sha256"], format!("[{i}] content digest"))?;
        let le: Vec<u8> = item.patches.iter().flat_map(|v| v.to_le_bytes()).collect();
        if hex(&Sha256::digest(&le)) != want["patch_sha256"] {
            // Attribute the difference: decode, or resize/pack.
            let bytes = &images[i];
            let rgb = ignis_artifact::vision::decode_image(bytes, u64::MAX).unwrap();
            let decode_matches = hex(&Sha256::digest(&rgb.rgb)) == decoded[i]["rgb_sha256"];
            return Err(format!("{name}: [{i}] patch sha256 differs (decoded RGB matches: {decode_matches})"));
        }
    }
    let checkpoint = &expected["rewrite_checkpoint"];
    let frontier = checkpoint["frontier"].as_u64().unwrap() as u32;
    check(
        checkpoint["kind"] == "turn_closure" && prompt.frontiers == [Some(frontier)],
        format!("rewrite checkpoint frontier {:?} vs {frontier}", prompt.frontiers),
    )
}

#[test]
fn a_text_only_prompt_is_the_degenerate_case() {
    let Some(frontend) = frontend_or_skip() else { return };
    let processor = frontend.vision_processor().expect("vision processor");
    let messages = [ChatMessage::text(Role::User, "hello world")];
    let rendered = frontend.chat_template().render_with_thinking(&messages, true, None).unwrap();
    let prompt = processor.prepare(frontend.tokenizer(), &rendered, &[], &[]).unwrap();
    assert_eq!(prompt.token_ids, frontend.tokenizer().encode(&rendered).unwrap());
    assert!(prompt.token_types.iter().all(|&t| t == layout::TEXT));
    let expected: Vec<i32> = (0..prompt.token_ids.len() as i32).collect();
    for axis in 0..3 {
        assert_eq!(prompt.position_axis(axis), expected.as_slice());
    }
    assert_eq!(prompt.rope_delta, 0);
    assert!(prompt.media.is_empty());
}

#[test]
fn messages_with_more_image_parts_than_images_are_a_placeholder_mismatch() {
    let Some(frontend) = frontend_or_skip() else { return };
    let processor = frontend.vision_processor().expect("vision processor");
    let image = std::fs::read(fixtures().join("images/on_grid.png")).unwrap();
    let messages = [ChatMessage {
        role: Role::User,
        content: MessageContent::Parts(vec![
            ContentPart::Image { url: None },
            ContentPart::Image { url: None },
        ]),
        tool_calls: Vec::new(),
        reasoning_content: None,
    }];
    let error = frontend.prepare_prompt(&processor, &messages, &[&image], false, None, false, None).unwrap_err();
    assert!(matches!(error, ProcessorError::PlaceholderMismatch(_)), "{error}");
    assert_eq!(error.code(), "invalid_media");
    // One image for one part prepares.
    let one = [ChatMessage { content: MessageContent::Parts(vec![ContentPart::Image { url: None }]), ..messages[0].clone() }];
    let prompt = frontend.prepare_prompt(&processor, &one, &[&image], false, None, false, None).unwrap();
    assert_eq!(prompt.vision_tokens(), 300);
}

#[test]
fn an_image_in_a_system_message_is_refused_by_the_template() {
    let Some(frontend) = frontend_or_skip() else { return };
    let processor = frontend.vision_processor().expect("vision processor");
    let image = std::fs::read(fixtures().join("images/on_grid.png")).unwrap();
    let messages = [
        ChatMessage {
            role: Role::System,
            content: MessageContent::Parts(vec![ContentPart::Image { url: None }]),
            tool_calls: Vec::new(),
            reasoning_content: None,
        },
        ChatMessage::text(Role::User, "hi"),
    ];
    let error = frontend.prepare_prompt(&processor, &messages, &[&image], false, None, false, None).unwrap_err();
    assert!(matches!(error, ProcessorError::Render(_)), "{error}");
}

#[test]
fn the_real_tokenizer_carries_the_contract_pad_ids() {
    let Some(frontend) = frontend_or_skip() else { return };
    assert!(frontend.vision_processor().is_ok());
}
