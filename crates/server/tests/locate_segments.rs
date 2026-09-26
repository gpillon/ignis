//! The `locate` evidence's segments and their keys (spec 18 phase A, GitHub
//! #274): the pure host functions `attention_head_locate_gpu.rs` measures
//! with, held to tables here.
//!
//! The last test renders a real prompt with the served artifact's own
//! template and tokenizer and checks that every line owns its own tokens.
//! It is machine-local and CPU-only: it skips when the artifact is absent
//! (`docs/agents/testing.md`).

use std::path::Path;

use ignis_server::decide::OrderedValue;

#[path = "support/locate.rs"]
mod locate;

use locate::{
    Region, SPAN_KIND, Scaffold, Segment, Unit, chunks, evidence, key_bytes, key_span, map_segments, owners, regions,
    segment_keys, span_user_text, user_text,
};

fn state(json: &str) -> OrderedValue {
    serde_json::from_str(json).expect("a JSON state")
}

/// Each segment's text as the system message carries it, and whether it owns keys.
fn spelled(json: &str) -> (Unit, Vec<(String, bool)>) {
    let evidence = evidence(&state(json)).expect("segmentable");
    let segments = evidence
        .segments
        .iter()
        .map(|segment| (evidence.system[segment.bytes.clone()].to_owned(), segment.owns))
        .collect();
    (evidence.unit, segments)
}

#[test]
fn a_strings_segments_are_its_lines_as_written() {
    let cases: &[(&str, &[(&str, bool)])] = &[
        (r#""a\nb""#, &[("a", true), ("b", true)]),
        // Empty and all-whitespace lines keep their index and own nothing.
        (r#""a\n\n  \nb""#, &[("a", true), ("", false), ("  ", false), ("b", true)]),
        // Split on `\n` exactly: a `\r` stays part of its line.
        (r#""a\r\nb""#, &[("a\\r", true), ("b", true)]),
        // Two identical lines are two segments.
        (r#""x\nx""#, &[("x", true), ("x", true)]),
        (r#""é\n日本""#, &[("é", true), ("日本", true)]),
        // A line's range is its escaped text.
        (r#""say \"hi\"\ttab""#, &[("say \\\"hi\\\"\\ttab", true)]),
        (r#""one line""#, &[("one line", true)]),
        (r#""trailing\n""#, &[("trailing", true), ("", false)]),
    ];
    for (json, expected) in cases {
        let (unit, segments) = spelled(json);
        assert_eq!(unit, Unit::Line, "{json}");
        let expected: Vec<(String, bool)> = expected.iter().map(|&(t, o)| (t.to_owned(), o)).collect();
        assert_eq!(segments, expected, "{json}");
    }
}

#[test]
fn an_arrays_segments_are_its_elements_without_separators() {
    let cases: &[(&str, &[(&str, bool)])] = &[
        (r#"["a", " ", "b"]"#, &[("a", true), (" ", false), ("b", true)]),
        (r#"[{"k":1},{"k":"two","z":null}]"#, &[(r#"{"k":1}"#, true), (r#"{"k":"two","z":null}"#, true)]),
        (r#"[1, [2, 3], true]"#, &[("1", true), ("[2,3]", true), ("true", true)]),
        // An object's own key order is kept, as the endpoint keeps it.
        (r#"[{"z":1,"a":2}]"#, &[(r#"{"z":1,"a":2}"#, true)]),
    ];
    for (json, expected) in cases {
        let (unit, segments) = spelled(json);
        assert_eq!(unit, Unit::Item, "{json}");
        let expected: Vec<(String, bool)> = expected.iter().map(|&(t, o)| (t.to_owned(), o)).collect();
        assert_eq!(segments, expected, "{json}");
    }
}

#[test]
fn the_evidence_is_the_bytes_the_endpoint_writes() {
    // `1.50` is written `1.5`: whatever the endpoint writes, the evidence is.
    let evidence = evidence(&state(r#"[{"b":1.50,"a":"x"}, "y"]"#)).expect("segmentable");
    assert_eq!(evidence.system, r#"{"evidence":[{"b":1.5,"a":"x"},"y"]}"#);
    let evidence = locate::evidence(&state(r#""l0\nl1""#)).expect("segmentable");
    assert_eq!(evidence.system, r#"{"evidence":"l0\nl1"}"#);
}

#[test]
fn only_a_string_or_a_non_empty_array_is_segmented() {
    for json in [r#"{"log": "a\nb"}"#, "[]", "17", "null", "true"] {
        assert!(evidence(&state(json)).is_err(), "{json} must not segment");
    }
}

fn segments(ranges: &[(usize, usize, bool)]) -> Vec<Segment> {
    ranges.iter().map(|&(a, b, owns)| Segment { bytes: a..b, owns }).collect()
}

#[test]
fn a_token_belongs_to_the_segment_holding_most_of_its_bytes() {
    // Segments at 0..3 and 5..8, a keyless one at 10..12, then 14..16.
    let layout = segments(&[(0, 3, true), (5, 8, true), (10, 12, false), (14, 16, true)]);
    let cases: &[((usize, usize), Option<u32>)] = &[
        ((0, 3), Some(0)),
        // Separator bytes belong to no segment.
        ((3, 5), None),
        // One byte each side: the tie goes to the earlier segment.
        ((2, 6), Some(0)),
        // `\nERROR` counts for the line it begins ...
        ((3, 8), Some(1)),
        // ... and `x\n` for the line it ends.
        ((7, 9), Some(1)),
        // A segment that owns no key never wins, even when it holds the most.
        ((9, 15), Some(3)),
        ((10, 12), None),
        ((16, 20), None),
    ];
    let offsets: Vec<(usize, usize)> = cases.iter().map(|&(range, _)| range).collect();
    let expected: Vec<Option<u32>> = cases.iter().map(|&(_, owner)| owner).collect();
    assert_eq!(owners(&offsets, &layout), expected);
}

#[test]
fn the_span_runs_from_the_first_owned_key_to_the_last() {
    assert_eq!(key_span(&[None, Some(0), None, Some(1), None]), Some(1..4));
    assert_eq!(key_span(&[Some(0)]), Some(0..1));
    assert_eq!(key_span(&[None, None]), None);
}

#[test]
fn a_segments_keys_are_a_range_of_the_span() {
    let owned = [None, Some(0), Some(0), None, Some(2), None];
    let span = key_span(&owned).expect("some key is owned");
    assert_eq!(span, 1..5);
    assert_eq!(segment_keys(&owned, &span, 3), vec![Some(0..2), None, Some(3..4)]);
}

#[test]
fn the_last_chunk_keeps_the_readouts_tail() {
    let cases: &[(u32, &[(u64, u64)])] = &[
        (2050, &[(0, 1024), (1024, 1017), (2041, 9)]),
        (2060, &[(0, 1024), (1024, 1024), (2048, 12)]),
        (2053, &[(0, 1024), (1024, 1020), (2044, 9)]),
        (1030, &[(0, 1021), (1021, 9)]),
        (1024, &[(0, 1024)]),
        // A prompt shorter than the tail is one chunk.
        (5, &[(0, 5)]),
    ];
    for &(total, expected) in cases {
        assert_eq!(chunks(total, 1024, 9), expected.to_vec(), "{total} tokens");
    }
}

#[test]
fn the_user_turn_is_the_kind_text_then_the_instruction() {
    assert_eq!(
        user_text(Scaffold::Quote, Unit::Line, "Which line says \"hi\"?"),
        "Find the one line of the evidence that the instruction asks for. Answer with only a JSON \
         object {\"quote\": \"<that line, copied exactly>\"}.\n\n{\"instruction\":\"Which line says \\\"hi\\\"?\"}"
    );
    assert_eq!(Scaffold::Index.opening(Unit::Line), "{\"line\":");
    assert_eq!(Scaffold::Index.opening(Unit::Item), "{\"item\":");
    assert_eq!(Scaffold::Quote.opening(Unit::Item), "{\"quote\":\"");
    assert!(Scaffold::Index.kind_text(Unit::Item).contains("{\"item\": <its number>}"));
}

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

/// With the served artifact's own template and tokenizer: the render opens
/// the evidence where the system message was written, and every line owns
/// tokens that spell it.
#[test]
fn every_line_of_a_served_render_owns_its_own_tokens() {
    use ignis_artifact::{FrontendSet, Reader};
    use ignis_server::artifact_template::ArtifactTemplateProvider;
    use ignis_server::template::{ChatMessage, TemplateProvider};
    use ignis_server::thinking::ThinkingOptions;

    let path = Path::new(ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let provider = ArtifactTemplateProvider::new(FrontendSet::from_reader(&reader).expect("frontend"));
    let set = FrontendSet::from_reader(&reader).expect("frontend");
    let tokenizer = set.tokenizer();

    // Identical lines, a line that starts with a digit, a line that is one
    // letter, an empty line, non-ASCII, quotes and a `\r`.
    let log = "09:21 ERROR billing: provider returned 503\n09:21 ERROR billing: provider returned 503\n\
               x\n\nÉtat: prêt à démarrer\nsaid \"no\"\r\nDEBUG done";
    let evidence = evidence(&OrderedValue::String(log.to_owned())).expect("segmentable");
    let thinking = ThinkingOptions { enable_thinking: false, ..ThinkingOptions::default() };
    let messages = [
        ChatMessage::text("system", evidence.system.clone()),
        ChatMessage::text("user", user_text(Scaffold::Quote, evidence.unit, "Which line says no?")),
    ];
    let rendered = provider.render_text(&messages, &thinking, &[]).expect("render");
    let served = provider.apply_chat_template(&messages, &thinking, &[]).expect("tokens").tokens;
    let (ids, offsets) = tokenizer.encode_with_offsets(&rendered).expect("encode");
    assert_eq!(ids, served, "the served prompt is the rendered text's tokenization");

    let (span, keys) = map_segments(&rendered, &offsets, &evidence).expect("the evidence maps to keys");
    for (index, (segment, keys)) in evidence.segments.iter().zip(&keys).enumerate() {
        match keys {
            None => assert!(!segment.owns, "segment {index} owns content and no token"),
            Some(range) => {
                assert!(segment.owns, "segment {index} owns no content and got tokens");
                let tokens = &ids[span.start + range.start..span.start + range.end];
                let text = tokenizer.decode(tokens).expect("decode");
                // A line's first token may carry the `\n` escape before it,
                // which belongs to no segment; the rest is the line exactly.
                let spelled = text.strip_prefix("\\n").unwrap_or(&text);
                let wanted = &evidence.system[segment.bytes.clone()];
                assert_eq!(spelled, wanted, "segment {index}: its tokens spell {text:?}");
            }
        }
    }
    // The two identical lines are two places.
    assert_ne!(keys[0], keys[1]);
}

#[test]
fn a_span_questions_user_turn_is_the_span_kind_text_then_the_instruction() {
    assert_eq!(
        span_user_text("Which job?"),
        format!("{SPAN_KIND}\n\n{{\"instruction\":\"Which job?\"}}")
    );
}

#[test]
fn every_prompt_token_is_labelled_by_where_its_first_byte_sits() {
    // One token per byte: a character-level tokenization makes every
    // boundary visible.
    let ev = evidence(&state(r#""ab\ncd""#)).expect("segmented");
    let user = "K.\n\n{\"instruction\":\"go\"}".to_owned();
    let rendered = format!("<s>{}</s><u>{user}</u>", ev.system);
    let offsets: Vec<(usize, usize)> = (0..rendered.len()).map(|i| (i, i + 1)).collect();
    let runs = regions(&rendered, &offsets, &ev, "K.", &user).expect("regions");
    let label = |byte: usize| runs.iter().find(|(r, _)| r.contains(&byte)).expect("covered").1;
    let ev_at = rendered.find(&ev.system).expect("evidence");
    let us = rendered.find(&user).expect("user");
    assert_eq!(label(0), Region::Template);
    assert_eq!(label(ev_at), Region::Evidence);
    assert_eq!(label(ev_at + ev.system.len() - 1), Region::Evidence);
    assert_eq!(label(ev_at + ev.system.len()), Region::Template);
    assert_eq!(label(us), Region::Kind);
    assert_eq!(label(us + 2), Region::Template, "the blank line after the kind text");
    let go = us + user.find("go").expect("instruction");
    assert_eq!((label(go), label(go + 1)), (Region::Instruction, Region::Instruction));
    assert_eq!(label(go + 2), Region::Template, "the closing quote");
    // runs are contiguous and cover every token
    assert_eq!(runs.first().expect("a run").0.start, 0);
    assert!(runs.windows(2).all(|w| w[0].0.end == w[1].0.start));
    assert_eq!(runs.last().expect("a run").0.end, offsets.len());
}

#[test]
fn a_keys_bytes_are_relative_to_the_evidence() {
    let ev = evidence(&state(r#""ab\ncd""#)).expect("segmented");
    let rendered = format!("xyz{}", ev.system);
    let offsets: Vec<(usize, usize)> = (0..rendered.len()).map(|i| (i, i + 1)).collect();
    let bytes = key_bytes(&rendered, &offsets, &ev, &(3..6)).expect("key bytes");
    assert_eq!(bytes, vec![[0, 1], [1, 2], [2, 3]]);
}
