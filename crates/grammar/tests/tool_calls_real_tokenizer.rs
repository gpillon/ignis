//! The tool calls' grammar (GitHub #611, spec #582) on the 27B's real
//! tokenizer: the call's two tags are one added token each and the grammar
//! reads each added token as the text the tokenizer decodes it to; no
//! ordinary token holds the call's opener, so the first draw cannot miss;
//! the permitted set equals the token-by-token reference on the states of
//! held calls as the tokenizer splits them (`=` merged into a name, the
//! newlines merged with whitespace, the tags in several tokens); and spike
//! S1's two miss points land on the real ids. Machine-local: skips when the
//! artifact is not at its path (the convention of
//! `crates/artifact/tests/real_artifact.rs`). CPU-only.

use std::path::Path;

use ignis_artifact::{FrontendSet, Reader, TokenBytes, Tokenizer};
use ignis_grammar::{DeclaredTool, GrammarCache, TokenSet, ToolCallMatcher, ToolHold, Vocab, VocabEntry, DEFAULT_CACHE_BYTES};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// The 27B family's logits row (crates/core/src/compute.rs).
const WIDTH: usize = 248_320;

fn open_or_skip() -> Option<Reader> {
    let path = Path::new(ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return None;
    }
    Some(Reader::open(path).expect("open artifact"))
}

/// The grammars as the server builds them: every column as bytes, the
/// added tokens as control tokens read by their decoded text, the chat
/// end-of-turn token as the stop.
fn grammars(tok: &Tokenizer) -> (GrammarCache, Vec<(u32, Vec<u8>)>) {
    let table = tok.token_bytes_table(WIDTH).expect("byte table");
    let stop = tok.encode("<|im_end|>").unwrap()[0];
    let mut added = Vec::new();
    let entries = table
        .into_iter()
        .enumerate()
        .map(|(id, e)| match e {
            TokenBytes::Text(b) => VocabEntry::Text(b),
            TokenBytes::Added { .. } => {
                if id as u32 != stop {
                    added.push((id as u32, tok.decode(&[id as u32]).unwrap().into_bytes()));
                }
                VocabEntry::Control
            }
            TokenBytes::Absent => VocabEntry::Absent,
        })
        .collect();
    let cache = GrammarCache::new(Vocab::new(entries, &[stop]), DEFAULT_CACHE_BYTES).with_added_texts(added.clone());
    (cache, added)
}

const WRITE_FILE: &str = r#"{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string","enum":["write","append"]}},"required":["path","content","mode"],"additionalProperties":false}"#;
const GET_TIME: &str = r#"{"type":"object","properties":{"zone":{"type":"string","pattern":"^[A-Z][a-z]+/[A-Z][a-z_]+$"},"day":{"type":"string","format":"date"}},"required":["zone","day"],"additionalProperties":false}"#;

fn tools() -> Vec<DeclaredTool> {
    vec![
        DeclaredTool { name: "write_file".into(), hold: ToolHold::Strict(WRITE_FILE.into()) },
        DeclaredTool { name: "get_time".into(), hold: ToolHold::Strict(GET_TIME.into()) },
        DeclaredTool { name: "read".into(), hold: ToolHold::Free },
    ]
}

/// A call in the template's form.
fn call(name: &str, args: &[(&str, &str)]) -> String {
    let mut out = format!("<tool_call>\n<function={name}>\n");
    for (k, v) in args {
        out.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
    }
    out.push_str("</function>\n</tool_call>");
    out
}

#[test]
fn the_calls_tags_are_one_added_token_each_and_read_as_the_text_they_decode_to() {
    let Some(reader) = open_or_skip() else { return };
    let frontend = FrontendSet::from_reader(&reader).expect("frontend set");
    let tok = frontend.tokenizer();
    let (cache, added) = grammars(tok);
    // Spike S1's ids and splits.
    assert_eq!(tok.encode("<tool_call>").unwrap(), [248_058]);
    assert_eq!(tok.encode("</tool_call>").unwrap(), [248_059]);
    assert_eq!(tok.encode("</function>").unwrap(), [510, 1628, 29]);
    assert_eq!(tok.encode("</parameter>").unwrap(), [510, 15704, 29]);
    let texts: std::collections::HashMap<u32, Vec<u8>> = added.into_iter().collect();
    assert_eq!(texts.get(&248_058).map(Vec::as_slice), Some(&b"<tool_call>"[..]));
    assert_eq!(texts.get(&248_059).map(Vec::as_slice), Some(&b"</tool_call>"[..]));
    // What the decoder writes for a call is what the grammar reads: the
    // tags' text in place, between the other tokens' bytes.
    let text = format!("ok\n{}", call("read", &[("path", "a.txt")]));
    let ids = tok.encode(&text).unwrap();
    assert!(ids.contains(&248_058) && ids.contains(&248_059));
    assert_eq!(tok.decode(&ids).unwrap(), text);
    let mut m = ToolCallMatcher::new(cache.tool_calls(&tools()).unwrap());
    for &id in &ids {
        assert!(m.accept(id), "token {id} of {text:?}");
    }
    assert!(!m.in_call() && m.is_accepting());
}

#[test]
fn no_ordinary_token_holds_the_opener_so_the_first_draw_cannot_miss() {
    let Some(reader) = open_or_skip() else { return };
    let frontend = FrontendSet::from_reader(&reader).expect("frontend set");
    let (cache, _) = grammars(frontend.tokenizer());
    assert!(!cache.tool_calls(&tools()).unwrap().first_draw_may_miss());
}

#[test]
fn the_permitted_set_equals_the_reference_on_held_calls_as_the_tokenizer_splits_them() {
    let Some(reader) = open_or_skip() else { return };
    let frontend = FrontendSet::from_reader(&reader).expect("frontend set");
    let tok = frontend.tokenizer();
    let (cache, _) = grammars(tok);
    let grammar = cache.tool_calls(&tools()).unwrap();
    let file = "<div>\n<parameter=path>\nb.txt\n</parameter>\n</div>\n\n  indented\r\nlast line";
    let texts = [
        format!("I will write it.\n\n{}", call("write_file", &[("path", "notes/a.txt"), ("mode", "append"), ("content", file)])),
        format!("{}\n{}", call("get_time", &[("day", "2026-11-15"), ("zone", "Europe/Rome")]), call("read", &[("path", "x")])),
        // A forgotten closer, and the whitespace around the tags merged.
        "<tool_call>\n<function=write_file>\n<parameter=content>\nx  \n\n<parameter=mode>\nwrite\n</parameter>  \n<parameter=path>\np\n</function>\n</tool_call>".to_string(),
    ];
    let (mut mask, mut reference) = (TokenSet::new(WIDTH), TokenSet::new(WIDTH));
    let mut held = 0;
    for text in &texts {
        let ids = tok.encode(text).unwrap();
        let mut m = ToolCallMatcher::new(grammar.clone());
        let every = (ids.len() / 16).max(1);
        for (i, &id) in ids.iter().enumerate() {
            if i % every == 0 || m.in_call() && i % 3 == 0 {
                m.mask(&mut mask);
                m.mask_reference(&mut reference);
                assert_eq!(mask, reference, "token {i} of {text:?}");
                held += 1;
            }
            assert!(m.check(id), "token {i} ({id}) of {text:?} refused");
            assert!(m.accept(id));
        }
        assert!(!m.in_call(), "{text:?} ends in free text");
    }
    assert!(held >= 40, "{held}");
}

#[test]
fn spike_s1s_two_miss_points_land_on_the_real_ids() {
    let Some(reader) = open_or_skip() else { return };
    let frontend = FrontendSet::from_reader(&reader).expect("frontend set");
    let tok = frontend.tokenizer();
    let (cache, _) = grammars(tok);
    let grammar = cache.tool_calls(&tools()).unwrap();
    let id = |text: &str| {
        let ids = tok.encode(text).unwrap();
        assert_eq!(ids.len(), 1, "{text:?} is one token");
        ids[0]
    };
    let (closing, function, parameter, open) = (id("</"), id("function"), id("parameter"), id("<"));
    assert_eq!((closing, function, parameter), (510, 1628, 15704));
    // `path` written, `content` and `mode` still missing.
    let value = tok.encode("<tool_call>\n<function=write_file>\n<parameter=path>\na.txt").unwrap();
    let fed = |extra: &[u32]| {
        let mut m = ToolCallMatcher::new(grammar.clone());
        for &t in value.iter().chain(extra) {
            assert!(m.accept(t), "token {t}");
        }
        m
    };
    // After a parameter's closer and its newline, `</` is a miss; `<` is not.
    let after_closer: Vec<u32> = tok.encode("\n</parameter>\n").unwrap();
    let mut m = fed(&after_closer);
    assert!(!m.check(closing) && m.check(open));
    // Inside the value, after a line-start `</`, `function` is a miss and
    // `parameter`, the forgotten closer, is not.
    let line_start: Vec<u32> = [tok.encode("\n").unwrap(), vec![closing]].concat();
    let mut m = fed(&line_start);
    assert!(!m.check(function) && m.check(parameter));
}
