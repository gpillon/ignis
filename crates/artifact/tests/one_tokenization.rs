//! One tokenization of a chat prompt (GitHub #647, spec #646): wherever an
//! added token starts at a cut, the head the server counts is read off the
//! prompt's own encoding rather than encoded again, and it must be exactly the
//! head's own encoding, ids and count. Checked at every such cut of the
//! reference-recorded renders, a generated prompt of about 16K tokens, and texts
//! that put an added token right after a multi-byte character; and at the three
//! boundaries the server reads (the generation opener, the last user query and
//! the system block's end), each of which sits on an added token in these
//! renders.
//!
//! CPU-only; skips when the artifact is not at its machine-local path (the
//! convention `real_frontend.rs` and `chat_render.rs` follow).

use std::path::{Path, PathBuf};

use ignis_artifact::{ChatMessage, ChatRenderOptions, ChatTemplate, FrontendSet, Reader, Role};
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

/// A user message of plain English prose about `chars` characters long (about
/// 6.6 characters a token), the same text on every run: a nonce, a filler body
/// and a closing question.
fn long_text(chars: usize) -> String {
    const WORDS: [&str; 24] = [
        "river", "stone", "ledger", "kernel", "window", "harbor", "signal", "copper", "meadow", "lantern", "orbit",
        "cipher", "garden", "vector", "thread", "anchor", "prism", "falcon", "summit", "glacier", "market", "engine",
        "silver", "timber",
    ];
    let mut out = String::from("1234 2345 3456 4567\n");
    let mut k = 0usize;
    while out.len() < chars {
        out.push_str(WORDS[(k * 7 + k / 5) % WORDS.len()]);
        out.push(if k % 17 == 16 { '.' } else { ' ' });
        if k % 170 == 169 {
            out.push('\n');
        }
        k += 1;
    }
    out.push_str("\nIn one sentence, what is this document?");
    out
}

#[test]
fn every_head_read_off_one_encoding_is_the_heads_own_encoding() {
    let Some(frontend) = frontend_or_skip() else {
        return;
    };
    let tokenizer = frontend.tokenizer();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(fixtures()).unwrap().map(|e| e.unwrap().path()).collect();
    paths.sort();
    assert_eq!(paths.len(), 7, "every recorded fixture is present");
    let mut texts: Vec<String> = paths
        .iter()
        .map(|path| {
            let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            fixture["expected"]["text"].as_str().unwrap().to_owned()
        })
        .collect();
    // An added token right after a two-, three- and four-byte character.
    texts.push("<|im_start|>user\nnaïve café 日本語🙂<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".into());
    texts.push("<|im_start|>system\n日本語<|im_end|>\n<|im_start|>user\né<|im_end|>\n<|im_start|>assistant\n<think>\n🙂</think>\n\n".into());
    // And right after runs of spaces and of line breaks.
    texts.push("<|im_start|>system\n  trailing spaces   <|im_end|>\n<|im_start|>user\n\n\n<|im_end|>\n<|im_start|>assistant\n".into());
    let options = ChatRenderOptions { enable_thinking: false, ..Default::default() };
    let long = frontend
        .chat_template()
        .render_with_thinking_and_tools(&[ChatMessage::text(Role::User, long_text(110_000))], options, Some(&[]))
        .expect("render");
    let n = tokenizer.encode(&long).expect("encode").len();
    assert!(n > 16_000, "the generated prompt is a 16K one: {n} tokens");
    texts.push(long);
    let mut checked = 0;
    for text in &texts {
        let (ids, offsets) = tokenizer.encode_with_offsets(text).expect("encode");
        assert_eq!(tokenizer.prefix_at_added_token(text, text.len(), &ids, &offsets), Some(ids.len()));
        for (k, &(start, _)) in offsets.iter().enumerate() {
            if let Some(n) = tokenizer.prefix_at_added_token(text, start, &ids, &offsets) {
                let head = tokenizer.encode(&text[..start]).expect("encode head");
                assert_eq!(n, k, "the head ends where the added token at byte {start} starts");
                assert_eq!(head, ids[..n], "the head read off the encoding at byte {start} is the head's own encoding");
                checked += 1;
            }
        }
        // The boundaries the server reads: the opener, the last user query
        // and the system block's end.
        for at in [
            ChatTemplate::generation_opener_offset(text),
            ChatTemplate::last_user_query_offset(text),
            ChatTemplate::system_block_offset(text),
        ]
        .into_iter()
        .flatten()
        {
            let n = tokenizer.prefix_at_added_token(text, at, &ids, &offsets).expect("each boundary sits on an added token");
            assert_eq!(tokenizer.encode(&text[..at]).expect("encode head"), ids[..n], "the boundary at byte {at}");
        }
    }
    eprintln!("{checked} heads checked over {} texts", texts.len());
    assert!(checked > 100, "{checked} heads checked");
}
