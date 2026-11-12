//! The grammar on the 27B's real tokenizer (GitHub #585, spec #287): the
//! per-token byte table equals the tokenizer's own decode, and on every
//! schema family of the corpus (`tests/corpus/cases.json`: `json_object`,
//! nested objects, arrays of objects, enums, null unions, `anyOf`, `$defs`
//! and root recursion, bounded numbers, `multipleOf`, the nine formats,
//! patterns) the permitted set equals the token-by-token reference on many
//! states, each document's own next token is in it, valid documents are
//! accepted token by token and violations refused. Machine-local: skips when
//! the artifact is not at its path (the convention of
//! `crates/artifact/tests/real_artifact.rs`). CPU-only.

use std::path::Path;
use std::sync::Arc;

use ignis_artifact::{FrontendSet, Reader, TokenBytes, Tokenizer};
use ignis_grammar::json::Json;
use ignis_grammar::{random_document, Grammar, Matcher, TokenSet, Vocab, VocabEntry};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// The 27B family's logits row (crates/core/src/compute.rs).
const WIDTH: usize = 248_320;
/// Reference states sampled per replayed document.
const STATES_PER_DOC: usize = 12;

/// The vocabulary as the server builds it: every column as bytes, the chat
/// end-of-turn token as the stop.
fn vocab(tok: &Tokenizer) -> (Arc<Vocab>, u32) {
    let table = tok.token_bytes_table(WIDTH).expect("byte table");
    let mut text_tokens = 0;
    for (id, e) in table.iter().enumerate() {
        if let TokenBytes::Text(b) = e {
            text_tokens += 1;
            if let Ok(s) = std::str::from_utf8(b) {
                assert_eq!(tok.decode(&[id as u32]).unwrap(), s, "token {id}");
            }
        }
    }
    assert!(text_tokens > 240_000, "{text_tokens}");
    let stop = tok.encode("<|im_end|>").unwrap()[0];
    let entries = table
        .into_iter()
        .map(|e| match e {
            TokenBytes::Text(b) => VocabEntry::Text(b),
            TokenBytes::Added { .. } => VocabEntry::Control,
            TokenBytes::Absent => VocabEntry::Absent,
        })
        .collect();
    (Vocab::new(entries, &[stop]), stop)
}

/// Replay `ids` then the stop, holding the mask to the reference on sampled
/// states and to the next token on every one; the number of states held.
fn replay(g: &Arc<Grammar>, ids: &[u32], stop: u32, what: &str) -> usize {
    let n = g.vocab().len();
    let (mut mask, mut reference) = (TokenSet::new(n), TokenSet::new(n));
    let mut m = Matcher::new(g.clone());
    let every = (ids.len() / STATES_PER_DOC).max(1);
    let mut held = 0;
    for (i, &id) in ids.iter().chain([&stop]).enumerate() {
        m.mask(&mut mask);
        assert!(mask.contains(id), "{what}: token {i} ({id}) is not in its permitted set");
        if i % every == 0 || id == stop {
            m.mask_reference(&mut reference);
            assert_eq!(mask, reference, "{what}: permitted set != reference at token {i}");
            held += 1;
        }
        assert!(m.accept(id), "{what}: token {i} of a valid document refused");
    }
    assert!(m.is_finished());
    held
}

fn open_or_skip() -> Option<Reader> {
    let path = Path::new(ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return None;
    }
    Some(Reader::open(path).expect("open artifact"))
}

#[test]
fn the_permitted_set_equals_the_reference_on_every_schema_family() {
    let Some(reader) = open_or_skip() else { return };
    let frontend = FrontendSet::from_reader(&reader).expect("frontend set");
    let tok = frontend.tokenizer();
    let (vocab, stop) = vocab(tok);
    let corpus = Json::parse(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/corpus/cases.json")).unwrap()).unwrap();
    let Json::Arr(cases) = corpus else { panic!("the corpus is an array") };
    assert_eq!(cases.len(), 19);
    // One thread per case: each holds a few dozen states to a reference that
    // checks all 248,320 columns.
    let held: Vec<(String, usize)> = std::thread::scope(|s| {
        let handles: Vec<_> = cases
            .iter()
            .map(|case| {
                let vocab = vocab.clone();
                s.spawn(move || {
                    let name = case.get("name").and_then(Json::as_str).unwrap().to_string();
                    let g = match case.get("schema") {
                        None => Grammar::json_object(vocab.clone()),
                        Some(schema) => Grammar::json_schema(vocab.clone(), &schema.compact(), true).unwrap_or_else(|e| panic!("{name}: {e}")),
                    };
                    let mut held = 0;
                    let Some(Json::Arr(valid)) = case.get("valid") else { panic!("{name}: no valid documents") };
                    for doc in valid {
                        for text in [doc.compact(), doc.pretty()] {
                            let ids = tok.encode(&text).unwrap();
                            held += replay(&g, &ids, stop, &format!("{name}: {text}"));
                        }
                    }
                    for seed in 0..4u64 {
                        let Some(doc) = random_document(&g, seed + 1, 120, 20_000).unwrap() else { continue };
                        let text = String::from_utf8(doc).unwrap();
                        held += replay(&g, &tok.encode(&text).unwrap(), stop, &format!("{name} random {seed}: {text}"));
                    }
                    let mut invalid: Vec<String> = match case.get("invalid") {
                        Some(Json::Arr(xs)) => xs.iter().map(Json::compact).collect(),
                        _ => Vec::new(),
                    };
                    if let Some(Json::Arr(xs)) = case.get("invalid_text") {
                        invalid.extend(xs.iter().map(|x| x.as_str().unwrap().to_string()));
                    }
                    for text in invalid {
                        let mut m = Matcher::new(g.clone());
                        let ids = tok.encode(&text).unwrap();
                        let accepted = m.accept_run(&ids) == ids.len() && m.accept(stop);
                        assert!(!accepted, "{name}: a violation was accepted: {text}");
                    }
                    (name, held)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (name, n) in &held {
        assert!(*n >= 40, "{name}: only {n} reference states");
    }
    eprintln!("reference states per case: {held:?}, {} in all", held.iter().map(|h| h.1).sum::<usize>());
}

#[test]
fn the_rare_string_states_are_masked_exactly_on_the_real_vocabulary() {
    let Some(reader) = open_or_skip() else { return };
    let frontend = FrontendSet::from_reader(&reader).expect("frontend set");
    let (vocab, _) = vocab(frontend.tokenizer());
    let unanchored = r#"{"type":"object","properties":{"code":{"type":"string","pattern":"[0-9]{2}"}},"required":["code"],"additionalProperties":false}"#;
    let cases = [
        // Inside a \u escape of a free string, and after a matched
        // unanchored pattern, where no \u escape may follow.
        (Grammar::json_object(vocab.clone()), &br#"{"note": "caf\u00"#[..]),
        (Grammar::json_object(vocab.clone()), br#"{"note": "\ud83d\ude"#),
        (Grammar::json_schema(vocab.clone(), unanchored, true).unwrap(), br#"{"code": "ab12c"#),
    ];
    let (mut mask, mut reference) = (TokenSet::new(WIDTH), TokenSet::new(WIDTH));
    for (g, prefix) in cases {
        let mut m = Matcher::new(g);
        assert!(m.accept_bytes(prefix));
        m.mask(&mut mask);
        m.mask_reference(&mut reference);
        assert_eq!(mask, reference, "{}", String::from_utf8_lossy(prefix));
        assert!(!mask.is_empty());
    }
}
