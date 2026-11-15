//! The tool calls' grammar (GitHub #611) on a small vocabulary: every byte
//! its own token, a few multi-byte tokens the way a real tokenizer splits
//! the dialect's tags, the call's two tags as added tokens with their text,
//! an added token with text of its own, a stop, and a column with no token.

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use super::*;
use crate::{GrammarCache, GrammarLimits, VocabEntry, XorShift};

/// Multi-byte tokens, after the 256 single bytes.
const PIECES: &[&str] = &["</", "parameter", "function", ">\n", "\n\n", "<parameter=", "path", "content", "=path", " \n", "abc", "\n</"];
const OPEN: u32 = 256 + PIECES.len() as u32;
const CLOSE: u32 = OPEN + 1;
const THINK: u32 = OPEN + 2;
const STOP: u32 = OPEN + 3;
const NOTHING: u32 = OPEN + 4;
const VOCAB: usize = NOTHING as usize + 1;
/// The ids of `</` and `function`.
const CLOSING: u32 = 256;
const FUNCTION: u32 = 258;

fn vocab() -> Arc<Vocab> {
    let mut entries: Vec<VocabEntry> = (0..=255u8).map(|b| VocabEntry::Text(vec![b])).collect();
    entries.extend(PIECES.iter().map(|p| VocabEntry::Text(p.as_bytes().to_vec())));
    entries.extend([VocabEntry::Control, VocabEntry::Control, VocabEntry::Control, VocabEntry::Control, VocabEntry::Absent]);
    Vocab::new(entries, &[STOP])
}

fn added() -> Vec<(u32, Vec<u8>)> {
    vec![(OPEN, b"<tool_call>".to_vec()), (CLOSE, b"</tool_call>".to_vec()), (THINK, b"<think>".to_vec())]
}

fn cache_within(limits: GrammarLimits) -> GrammarCache {
    GrammarCache::within(vocab(), limits).with_added_texts(added())
}

fn cache() -> GrammarCache {
    cache_within(GrammarLimits::default())
}

const WRITE_FILE: &str = r#"{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string","enum":["w","a"]}},"required":["path","content","mode"],"additionalProperties":false}"#;
const GET_TIME: &str = r#"{"type":"object","properties":{"zone":{"type":"string","pattern":"^[A-Z][a-z]+/[A-Z][a-z]+$"},"day":{"type":"string","format":"date"}},"required":["zone","day"],"additionalProperties":false}"#;

fn tool(name: &str, strict: Option<&str>) -> DeclaredTool {
    DeclaredTool { name: name.into(), strict: strict.map(str::to_string) }
}

/// `write_file` and `get_time` strict, `search` free.
fn tools() -> Vec<DeclaredTool> {
    vec![tool("write_file", Some(WRITE_FILE)), tool("get_time", Some(GET_TIME)), tool("search", None)]
}

fn matcher_of(cache: &GrammarCache, tools: &[DeclaredTool]) -> ToolCallMatcher {
    ToolCallMatcher::new(cache.tool_calls(tools).unwrap_or_else(|r| panic!("{r:?}")))
}

fn matcher() -> ToolCallMatcher {
    matcher_of(&cache(), &tools())
}

/// A call in the template's form (`artifact_template.rs`'s dialect).
fn call(name: &str, args: &[(&str, &str)]) -> String {
    let mut out = format!("<tool_call>\n<function={name}>\n");
    for (k, v) in args {
        out.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
    }
    out.push_str("</function>\n</tool_call>");
    out
}

/// How many leading bytes of `text` the matcher takes, byte by byte.
fn taken(m: &mut ToolCallMatcher, text: &str) -> usize {
    text.bytes().take_while(|&b| m.accept_bytes(&[b])).count()
}

/// Whether the whole of `text`, fed byte by byte from a fresh matcher,
/// is accepted and leaves the matcher in free text.
fn accepted(text: &str) -> bool {
    let mut m = matcher();
    taken(&mut m, text) == text.len() && !m.in_call()
}

/// `prefix` is accepted whole; then the first byte of `next` is refused.
fn refused_after(prefix: &str, next: &str) -> bool {
    let mut m = matcher();
    assert_eq!(taken(&mut m, prefix), prefix.len(), "{prefix:?} is accepted");
    !m.accept_bytes(&next.as_bytes()[..1])
}

#[test]
fn free_text_passes_every_token_and_a_stop_ends_it() {
    let mut m = matcher();
    for id in [b'h' as u32, CLOSING, THINK, NOTHING, b'<' as u32, CLOSE] {
        assert!(m.accept(id), "token {id}");
    }
    assert!(m.is_accepting() && !m.in_call());
    assert!(m.accept(STOP) && m.is_finished());
    assert!(!m.check(b'x' as u32));
}

#[test]
fn the_opener_arms_by_its_token_or_its_bytes_and_the_name_is_held() {
    for opener in [Some(OPEN), None] {
        let mut m = matcher();
        match opener {
            Some(id) => assert!(m.accept(id)),
            None => assert!(m.accept_bytes(b"text <tool_call>")),
        }
        assert!(m.in_call() && !m.is_accepting(), "a stop is a miss once a call is open");
        assert!(!m.check(STOP) && !m.check(NOTHING), "nor does a column with no text pass");
        let mut undeclared = m.clone();
        assert_eq!(taken(&mut undeclared, "\n<function=delete>"), "\n<function=".len(), "an undeclared name");
        let mut prefix = m.clone();
        assert_eq!(taken(&mut prefix, "\n<function=write>"), "\n<function=write".len(), "a declared name's prefix is not a name");
        assert!(m.accept_bytes(b"\n<function=write_file>"));
    }
}

#[test]
fn a_tool_that_is_not_held_runs_free_to_its_first_closer() {
    let mut m = matcher();
    assert!(m.accept_bytes(b"<tool_call>\n<function=search>\n<parameter=q>anything</function> at all"));
    assert!(m.in_call() && m.is_accepting(), "a stop cuts a free call, as today");
    assert!(m.accept(NOTHING) && m.accept(THINK));
    assert!(m.accept(CLOSE) && !m.in_call(), "its first `</tool_call>` ends it");
    // And a further call opens the same way.
    assert!(m.accept_bytes(call("search", &[("q", "x")]).as_bytes()) && !m.in_call());
}

#[test]
fn a_held_call_in_the_templates_form_is_accepted_in_any_parameter_order_and_again() {
    let args = [("path", "a.txt"), ("content", "line one\n  line two"), ("mode", "w")];
    for order in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
        let written: Vec<(&str, &str)> = order.iter().map(|&i| args[i]).collect();
        let text = format!("Writing it.\n{}\n{}", call("write_file", &written), call("get_time", &[("zone", "Europe/Rome"), ("day", "2026-11-15")]));
        assert!(accepted(&text), "{text}");
    }
    // The call's tags as the added tokens, the rest as bytes.
    let mut m = matcher();
    assert!(m.accept(OPEN));
    let body = call("write_file", &[("path", "a"), ("content", "b"), ("mode", "a")]);
    let body = body.strip_prefix("<tool_call>").unwrap().strip_suffix("</tool_call>").unwrap();
    assert!(m.accept_bytes(body.as_bytes()) && m.in_call());
    assert!(m.accept(CLOSE) && !m.in_call() && m.is_accepting());
}

#[test]
fn a_forgotten_parameter_closer_is_tolerated() {
    let text = "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n<parameter=mode>\nw\n</parameter>\n<parameter=content>\nx\n</function>\n</tool_call>";
    assert!(accepted(text));
}

#[test]
fn a_parameter_is_opened_once_and_an_undeclared_one_never() {
    // Before the first parameter only whitespace and a step.
    assert!(refused_after("<tool_call>\n<function=write_file>\n<parameter=", "force>"));
    assert!(refused_after("<tool_call>\n<function=write_file>\n", "x"));
    // After `path`, its opener at a line start is the value's own text.
    let mut m = matcher();
    let head = "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\n";
    assert!(m.accept_bytes(head.as_bytes()));
    assert!(m.accept_bytes(b"<parameter=path>\nb.txt\n"), "text of `content`");
    assert!(m.accept_bytes(b"</parameter>\n<parameter=mode>\na\n</parameter>\n</function>\n</tool_call>") && !m.in_call());
    // A held string's next line must be a step: there, a written or
    // undeclared parameter's opener is a miss.
    assert!(refused_after("<tool_call>\n<function=write_file>\n<parameter=mode>\nw\n</parameter>\n<parameter=", "mode>"));
    assert!(refused_after("<tool_call>\n<function=write_file>\n<parameter=mode>\nw\n</parameter>\n<parameter=", "x"));
}

#[test]
fn the_function_closes_only_once_every_required_parameter_is_written() {
    // Before any parameter, a close is a miss at its `f`.
    assert!(refused_after("<tool_call>\n<function=write_file>\n</", "function>"));
    // After a held string, at the line after it: `</` after its closer's
    // line, `function` after a line-start `</`.
    assert!(refused_after("<tool_call>\n<function=write_file>\n<parameter=mode>\nw\n</parameter>\n<", "/"));
    assert!(refused_after("<tool_call>\n<function=write_file>\n<parameter=mode>\nw\n</", "function>"));
    // Every one written: the close is the call's.
    assert!(accepted(&call("write_file", &[("mode", "w"), ("path", "p"), ("content", "c")])));
}

#[test]
fn a_free_value_with_a_required_parameter_missing_misses_at_the_two_points_of_the_spike() {
    let value = "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n";
    // `</` after a parameter's closer and its newline.
    assert!(refused_after(&format!("{value}</parameter>\n<"), "/"));
    assert!(refused_after(&format!("{value}</parameter>\n\n <"), "/"));
    // `function` after a line-start `</`, the forgotten closer's place.
    assert!(refused_after(&format!("{value}</functio"), "n"));
    let mut m = matcher();
    assert!(m.accept_bytes(value.as_bytes()));
    assert!(m.clone().accept(CLOSING) && !m.clone().accept_bytes(b"</function"), "`</` passes; `</function` is the miss");
    let mut after_closer = m.clone();
    assert!(after_closer.accept_bytes(b"</parameter>\n") && !after_closer.check(CLOSING), "the token `</` after a closer's line");
    let mut line_start = m.clone();
    assert!(line_start.accept(CLOSING) && !line_start.check(FUNCTION), "the token `function` after a line-start `</`");
    // Lines that only look like it are the value's text.
    assert!(m.accept_bytes(b"</footer>\n</parameter>x\n</p>\n"));
    // With every required parameter written, the value may hold the same lines.
    let all = "<tool_call>\n<function=write_file>\n<parameter=mode>\nw\n</parameter>\n<parameter=path>\na\n</parameter>\n<parameter=content>\n";
    let mut m = matcher();
    assert!(m.accept_bytes(format!("{all}</functional>\n</parameter>\n</div>\n</parameter>\n</function>\n</tool_call>").as_bytes()));
    assert!(!m.in_call());
}

#[test]
fn a_held_string_is_held_to_its_enum_pattern_and_format_on_one_line() {
    let mode = "<tool_call>\n<function=write_file>\n<parameter=mode>\n";
    assert!(refused_after(mode, "x"));
    assert!(refused_after(&format!("{mode}w"), "a"));
    assert!(refused_after(&format!("{mode}w\n"), "x"), "one line, then the call's next step");
    assert!(refused_after(mode, "\n"), "the empty string is not one of its values");
    let time = "<tool_call>\n<function=get_time>\n<parameter=zone>\n";
    assert!(refused_after(time, "e"));
    assert!(refused_after(&format!("{time}Europe"), "\n"), "a value the pattern does not match yet");
    assert!(refused_after(&format!("{time}Europe/Rome\n</parameter>\n<parameter=day>\n2026-1"), "3"));
    assert!(accepted(&call("get_time", &[("day", "2026-02-28"), ("zone", "America/Lima")])));
    // A held string's value is its line: no leading newline but the one
    // trimmed, which the parser also trims.
    assert!(accepted("<tool_call>\n<function=write_file>\n<parameter=mode>w\n</parameter>\n<parameter=path>\n\n</parameter>\n<parameter=content>\n</parameter>\n</function>\n</tool_call>"));
}

#[test]
fn a_value_cannot_hold_its_own_calls_confirmed_close() {
    let mut m = matcher();
    let head = "<tool_call>\n<function=write_file>\n<parameter=mode>\nw\n</parameter>\n<parameter=path>\na\n</parameter>\n<parameter=content>\nan example:\n";
    assert!(m.accept_bytes(head.as_bytes()));
    assert!(m.accept_bytes(b"</function>\n") && m.in_call());
    assert!(m.accept(CLOSE) && !m.in_call(), "the documented residue: the call ends there");
}

#[test]
fn whitespace_between_tags_is_bounded_at_its_limit_and_the_byte_past_it_is_a_miss() {
    let limit = 4;
    let cache = cache_within(GrammarLimits { max_whitespace: limit, ..GrammarLimits::default() });
    let fresh = || matcher_of(&cache, &tools());
    // Line breaks: before the first parameter a step starts a line.
    let run = |n: usize| "\n".repeat(n);
    let places = [
        ("<tool_call>".to_string(), "<function=write_file>"),
        ("<tool_call><function=write_file>".to_string(), "<parameter=path>"),
        ("<tool_call><function=write_file><parameter=path>\na\n</parameter>".to_string(), "<parameter=mode>"),
        ("<tool_call><function=write_file><parameter=path>\na\n</parameter><parameter=content>c\n</parameter><parameter=mode>w\n</parameter></function>".to_string(), "</tool_call>"),
    ];
    for (before, after) in places {
        let mut m = fresh();
        assert!(m.accept_bytes(before.as_bytes()), "{before:?}");
        assert!(m.clone().accept_bytes(format!("{}{after}", run(limit as usize)).as_bytes()), "{before:?} + {limit} spaces");
        let mut past = m.clone();
        assert!(past.accept_bytes(run(limit as usize).as_bytes()) && !past.accept_bytes(b"\n") && !past.accept_bytes(b" "), "{before:?} + {} spaces", limit + 1);
    }
    // In a free value, past the bound after a closer's line the text goes on
    // with something else.
    let mut m = fresh();
    assert!(m.accept_bytes(b"<tool_call><function=write_file><parameter=path>\na\n</parameter> \t\r\n"));
    assert!(!m.clone().accept_bytes(b" ") && m.accept_bytes(b"x\n"));
}

#[test]
fn the_refusals_name_the_tool_and_its_constraint() {
    let object = |inner: &str| format!(r#"{{"type":"object","properties":{{"a":{inner}}},"required":["a"],"additionalProperties":false}}"#);
    let cases: Vec<(Vec<DeclaredTool>, usize, ToolField, &str, &str)> = vec![
        (vec![tool("f", None), tool("g", Some(&object(r#"{"type":"string","minLength":3}"#)))], 1, ToolField::Parameters, "unsupported_keyword", "minLength"),
        (vec![tool("g", Some(r#"{"type":"object","properties":{}}"#))], 0, ToolField::Parameters, "additional_properties_not_false", "additionalProperties"),
        (vec![tool("a>b", Some(&object(r#"{"type":"string"}"#)))], 0, ToolField::Name, "tool_name_unwritable", "name"),
        (vec![tool("x</tool_call", Some(&object(r#"{"type":"string"}"#)))], 0, ToolField::Name, "tool_name_unwritable", "name"),
        (vec![tool("g", None), tool("g", Some(&object(r#"{"type":"string"}"#)))], 1, ToolField::Name, "duplicate_tool_name", "name"),
        (vec![tool("g", Some(r#"{"type":"object","properties":{"a>b":{"type":"string"}},"required":["a>b"],"additionalProperties":false}"#))], 0, ToolField::Parameters, "parameter_name_unwritable", "properties"),
        (vec![tool("g", Some(&object(r#"{"type":"string","enum":["a\nb","<parameter=x>"]}"#)))], 0, ToolField::Parameters, "no_writable_value", "enum"),
        (vec![tool("g", Some(&object(r#"{"type":"string","pattern":"^\\n$"}"#)))], 0, ToolField::Parameters, "no_writable_value", "pattern"),
        (vec![tool("g", Some(&object(r#"{"type":"string","pattern":"^</function>$"}"#)))], 0, ToolField::Parameters, "no_writable_value", "pattern"),
    ];
    for (tools, at, field, code, keyword) in cases {
        let refused = cache().tool_calls(&tools).expect_err(code);
        assert_eq!((refused.tool, refused.field, refused.refusal.code, refused.refusal.keyword.as_str()), (at, field, code, keyword), "{tools:?}");
    }
    // The same names on tools that are not held change nothing; such a name
    // is just not callable.
    assert!(cache().tool_calls(&[tool("a>b", None), tool("g", None), tool("g", None)]).is_ok());
}

#[test]
fn every_refusal_code_of_the_dialect_is_pinned() {
    let source = include_str!("../tool_calls.rs");
    let source = &source[..source.find("#[cfg(test)]").unwrap()];
    let codes: Vec<&str> = ["tool_name_unwritable", "duplicate_tool_name", "parameter_name_unwritable", "no_writable_value"]
        .into_iter()
        .filter(|code| source.contains(&format!("\"{code}\"")))
        .collect();
    assert_eq!(codes.len(), 4, "{codes:?}");
    // Every quoted snake_case code the module gives is one of them.
    let given = source.matches("refusal(").count();
    assert!(given >= 4, "{given}");
    let pinned = include_str!("tests.rs");
    for code in codes {
        assert!(pinned.matches(&format!("\"{code}\"")).count() >= 2, "refusal `{code}` has no test");
    }
}

#[test]
fn a_tool_set_sent_again_is_compiled_once_and_its_strict_schemas_with_it() {
    let cache = cache();
    let first = cache.tool_calls(&tools()).unwrap();
    let compiles = cache.compiles();
    let again = cache.tool_calls(&tools()).unwrap();
    assert!(Arc::ptr_eq(&first, &again));
    assert_eq!(cache.compiles(), compiles);
    // The strict schemas are the response_format cache's own entries.
    let before = cache.compiles();
    cache.json_schema(WRITE_FILE, true).unwrap();
    assert_eq!(cache.compiles(), before);
    let held: Vec<(&str, Vec<String>)> = first.held().map(|(name, p)| (name, p.declared().to_vec())).collect();
    assert_eq!(held, vec![("write_file", vec!["path".into(), "content".into(), "mode".into()]), ("get_time", vec!["zone".into(), "day".into()])]);
}

#[test]
fn a_typed_parameter_is_free_text_until_its_values_are_held() {
    let schema = r#"{"type":"object","properties":{"n":{"type":"integer"},"s":{"type":["string","null"]}},"required":["n","s"],"additionalProperties":false}"#;
    let mut m = matcher_of(&cache(), &[tool("count", Some(schema))]);
    assert!(m.accept_bytes(call("count", &[("n", "twelve"), ("s", "null")]).as_bytes()) && !m.in_call());
}

#[test]
fn the_first_draw_can_miss_only_on_a_token_holding_the_opener_and_more() {
    assert!(!cache().tool_calls(&tools()).unwrap().first_draw_may_miss());
    let mut entries: Vec<VocabEntry> = (0..=255u8).map(|b| VocabEntry::Text(vec![b])).collect();
    entries.push(VocabEntry::Text(b"<tool_call>x".to_vec()));
    let cache = GrammarCache::new(Vocab::new(entries, &[]), 1 << 24);
    assert!(cache.tool_calls(&tools()).unwrap().first_draw_may_miss());
    // No tool callable: the opener itself is a miss.
    assert!(GrammarCache::new(vocab(), 1 << 24).tool_calls(&[tool("a>b", None)]).unwrap().first_draw_may_miss());
}

// ── generated texts ─────────────────────────────────────────────────────

/// The dialect's pieces, and text: what a generated reading is made of.
const SOUP: &[&str] = &[
    "<tool_call>", "\n", "<function=write_file>", "<function=get_time>", "<function=search>", "<parameter=path>",
    "<parameter=content>", "<parameter=mode>", "<parameter=zone>", "<parameter=day>", "<parameter=x>", "</parameter>",
    "</function>", "</tool_call>", "a.txt", "w", "a", "x", " ", "\t", "<", "</", "Europe/Rome", "2026-11-15", "é", "hello\n",
    "</div>", "<think>",
];

/// A text the grammar accepts, grown from `SOUP`'s pieces in random order
/// (a piece refused is skipped), with the matcher at each piece's end.
fn generated(seed: u64, pieces: usize) -> (Vec<u8>, Vec<ToolCallMatcher>) {
    let mut rng = XorShift(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut m = matcher();
    let mut text = Vec::new();
    let mut states = Vec::new();
    for _ in 0..pieces * 8 {
        if states.len() >= pieces {
            break;
        }
        let piece = SOUP[(rng.next_u64() % SOUP.len() as u64) as usize];
        if m.accept_bytes(piece.as_bytes()) {
            text.extend_from_slice(piece.as_bytes());
            states.push(m.clone());
        }
    }
    (text, states)
}

/// Whether some bytes lead `m` back to free text: a breadth-first search
/// over states, printable bytes and the line break (no space: a run of
/// mixed whitespace makes as many states as it has spellings).
fn completes(m: &ToolCallMatcher) -> bool {
    let g = &m.grammar;
    let alphabet: Vec<u8> = (0x21..0x7F).chain([b'\n']).collect();
    let mut seen: HashSet<CallState> = HashSet::new();
    let mut queue = VecDeque::from([m.state.clone()]);
    while let Some(s) = queue.pop_front() {
        if ToolCallGrammar::outside_held(&s) {
            return true;
        }
        assert!(seen.len() <= 200_000, "search budget spent");
        for &b in &alphabet {
            let mut next = s.clone();
            if g.step(&mut next, b) && seen.insert(next.clone()) {
                queue.push_back(next);
            }
        }
    }
    false
}

#[test]
fn no_state_is_a_dead_end() {
    let mut held = 0;
    for seed in 0..16 {
        let (text, states) = generated(seed, 40);
        for m in states.iter().filter(|m| m.in_call()).step_by(2) {
            assert!(completes(m), "a dead end after {:?}", String::from_utf8_lossy(&text));
            held += 1;
        }
    }
    assert!(held > 50, "{held}");
}

/// `text` as the vocabulary's tokens: the call's tags as their added
/// tokens, else the longest piece, else the byte.
fn tokenize(text: &[u8]) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut at = 0;
    while at < text.len() {
        let rest = &text[at..];
        let tagged = [(OPEN, "<tool_call>"), (CLOSE, "</tool_call>"), (THINK, "<think>")].into_iter().find(|(_, t)| rest.starts_with(t.as_bytes()));
        if let Some((id, t)) = tagged {
            ids.push(id);
            at += t.len();
            continue;
        }
        let piece = PIECES.iter().enumerate().filter(|(_, p)| rest.starts_with(p.as_bytes())).max_by_key(|(_, p)| p.len());
        match piece {
            Some((i, p)) => {
                ids.push(256 + i as u32);
                at += p.len();
            }
            None => {
                ids.push(rest[0] as u32);
                at += 1;
            }
        }
    }
    ids
}

#[test]
fn the_permitted_set_equals_the_reference_on_every_state_of_generated_texts() {
    let (mut mask, mut reference) = (TokenSet::new(VOCAB), TokenSet::new(VOCAB));
    let mut states = 0;
    for seed in 0..40 {
        let (text, _) = generated(seed, 30);
        let mut m = matcher();
        for id in tokenize(&text) {
            m.mask(&mut mask);
            m.mask_reference(&mut reference);
            assert_eq!(mask, reference, "after {:?}", String::from_utf8_lossy(&text));
            assert!(mask.contains(id), "the text's own next token {id}");
            assert!(m.accept(id), "{:?}", String::from_utf8_lossy(&text));
            states += 1;
        }
    }
    assert!(states > 1_000, "{states}");
}
