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
    DeclaredTool { name: name.into(), hold: strict.map_or(ToolHold::Free, |schema| ToolHold::Strict(schema.into())) }
}

/// A tool sent without `strict` where the surface normalizes it (#614).
fn normalized(name: &str, schema: &str) -> DeclaredTool {
    DeclaredTool { name: name.into(), hold: ToolHold::Normalized(schema.into()) }
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
fn free_text_passes_every_token_a_stop_included() {
    let mut m = matcher();
    for id in [b'h' as u32, CLOSING, THINK, NOTHING, b'<' as u32, CLOSE] {
        assert!(m.accept(id), "token {id}");
    }
    assert!(m.is_accepting() && !m.in_call());
    // A stop passes and changes nothing: a lane that decodes past its EOS
    // goes on in free text, a call it opens held as any.
    assert!(m.accept(STOP) && !m.is_finished());
    assert!(m.accept(OPEN) && !m.check(STOP));
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
    // Before the first parameter, spaces leave room for the line break a
    // step needs: a run of them up to the bound is no dead end.
    let mut m = fresh();
    assert!(m.accept_bytes(format!("<tool_call><function=write_file>{}", " ".repeat(limit as usize - 1)).as_bytes()));
    assert!(!m.clone().accept_bytes(b" ") && m.accept_bytes(b"\n<parameter=path>"));
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

// ── typed and nullable values (#612) ─────────────────────────────────────

/// `edit`, strict and typed: an integer with bounds, a boolean, an object
/// holding a list of edits, a nested array of objects, a recursive tree,
/// and two nullable parameters, a string and a bounded integer.
const EDIT: &str = r##"{"type":"object","properties":{"line":{"type":"integer","minimum":1,"maximum":1000},"dry_run":{"type":"boolean"},"change":{"type":"object","properties":{"edits":{"type":"array","items":{"type":"object","properties":{"old":{"type":"string"},"new":{"type":"string"}},"required":["old","new"],"additionalProperties":false}}},"required":["edits"],"additionalProperties":false},"rows":{"type":"array","items":{"type":"array","items":{"type":"object","properties":{"k":{"type":"string"},"v":{"type":"number"}},"required":["k","v"],"additionalProperties":false}}},"tree":{"$ref":"#/$defs/node"},"note":{"type":["string","null"]},"limit":{"anyOf":[{"type":"integer","minimum":0,"maximum":9},{"type":"null"}]}},"required":["line","dry_run","change","rows","tree","note","limit"],"additionalProperties":false,"$defs":{"node":{"type":"object","properties":{"name":{"type":"string"},"kids":{"type":"array","items":{"$ref":"#/$defs/node"}}},"required":["name","kids"],"additionalProperties":false}}}"##;

/// `edit`'s values in the template's form (`tojson`'s spelling for a typed
/// one), in the schema's order.
const EDIT_VALUES: [(&str, &str); 7] = [
    ("line", "12"),
    ("dry_run", "false"),
    ("change", r#"{"edits": [{"old": "x", "new": "y"}, {"old": "a\nb", "new": ""}]}"#),
    ("rows", r#"[[{"k": "a", "v": 1.5}], [], [{"k": "b", "v": -2e3}, {"k": "c", "v": 0}]]"#),
    ("tree", r#"{"name": "root", "kids": [{"name": "leaf", "kids": []}, {"name": "twig", "kids": [{"name": "bud", "kids": []}]}]}"#),
    ("note", "hello"),
    ("limit", "3"),
];

fn edit_matcher() -> ToolCallMatcher {
    matcher_of(&cache(), &[tool("edit", Some(EDIT))])
}

/// A call to `edit`: its values, each replaced by `with`'s where named, in
/// the schema's order, those `without` names left out.
fn edit_call(with: &[(&str, &str)], without: &[&str]) -> String {
    let args: Vec<(&str, &str)> = EDIT_VALUES
        .iter()
        .filter(|(k, _)| !without.contains(k))
        .map(|&(k, v)| with.iter().find(|(name, _)| *name == k).copied().unwrap_or((k, v)))
        .collect();
    call("edit", &args)
}

/// Whether `m` takes the whole of `text` and is back in free text.
fn takes_whole(mut m: ToolCallMatcher, text: &str) -> bool {
    taken(&mut m, text) == text.len() && !m.in_call()
}

/// The bytes of `text` an `edit` matcher takes before its first miss.
fn edit_taken(text: &str) -> usize {
    taken(&mut edit_matcher(), text)
}

/// The text of `edit`'s call up to the value of `name` (its opener and
/// newline included), every value before it as `EDIT_VALUES` has it.
fn edit_head(name: &str) -> String {
    let call = edit_call(&[], &[]);
    let at = call.find(&format!("<parameter={name}>\n")).unwrap();
    call[..at + format!("<parameter={name}>\n").len()].to_owned()
}

#[test]
fn a_typed_value_is_json_held_to_its_subschema() {
    assert!(takes_whole(edit_matcher(), &edit_call(&[], &[])));
    // Compact, as `tojson` spells it, and pretty-printed.
    let compact: Vec<(&str, String)> = EDIT_VALUES.iter().map(|&(k, v)| (k, if k == "note" { v.to_owned() } else { Json::parse(v).unwrap().compact() })).collect();
    let pretty: Vec<(&str, String)> = EDIT_VALUES.iter().map(|&(k, v)| (k, if k == "note" { v.to_owned() } else { Json::parse(v).unwrap().pretty() })).collect();
    for args in [compact, pretty] {
        let args: Vec<(&str, &str)> = args.iter().map(|(k, v)| (*k, v.as_str())).collect();
        assert!(takes_whole(edit_matcher(), &call("edit", &args)), "{args:?}");
    }
    // Each miss at the byte that makes it one.
    let misses: [(&str, &str, &str); 11] = [
        ("line", "twelve", ""),
        ("line", "2000", "200"),
        ("line", "0", ""),
        ("line", "1.5", "1"),
        ("dry_run", "\"no\"", ""),
        ("dry_run", "True", ""),
        ("change", r#"{"edits": [{"old": "x"}]}"#, r#"{"edits": [{"old": "x""#),
        ("change", r#"{"edits": [{"old": "x", "path": "p"}]}"#, r#"{"edits": [{"old": "x", ""#),
        ("rows", r#"[{"k": "a", "v": 1}]"#, "["),
        ("tree", r#"{"name": "root", "kids": [{"name": "leaf"}]}"#, r#"{"name": "root", "kids": [{"name": "leaf""#),
        ("limit", "10", "1"),
    ];
    for (name, value, kept) in misses {
        let text = edit_call(&[(name, value)], &[]);
        let head = edit_head(name);
        assert_eq!(edit_taken(&text), head.len() + kept.len(), "{name}: {value}");
    }
}

#[test]
fn a_typed_value_starts_at_its_first_byte_and_whitespace_may_follow_it() {
    let head = edit_head("line");
    // No whitespace before it but the newline the parser trims, which it
    // may also leave out.
    for value in [" 12", "\n12", "\t12"] {
        assert_eq!(edit_taken(&edit_call(&[("line", value)], &[])), head.len(), "{value:?}");
    }
    let bare = edit_call(&[], &[]).replace("<parameter=line>\n12", "<parameter=line>12");
    assert!(takes_whole(edit_matcher(), &bare));
    // After it, whitespace and blank lines, then the call's next step at a
    // line's start, and nothing else.
    assert!(takes_whole(edit_matcher(), &edit_call(&[("line", "12 \t\n\n ")], &[])));
    for (value, kept) in [("12 x", "12 "), ("12\nx", "12\n"), ("12\n </parameter>", "12\n "), ("12</parameter>", "12")] {
        assert_eq!(edit_taken(&edit_call(&[("line", value)], &[])), head.len() + kept.len(), "{value:?}");
    }
    // With a later parameter still to write, the close after a typed value
    // is a miss at spike S1's two points, by the rule itself.
    for (after, kept) in [("</parameter>\n</function>", "</parameter>\n<"), ("</function>", "</")] {
        let text = format!("{head}12\n{after}");
        assert_eq!(edit_taken(&text), head.len() + 3 + kept.len(), "{after:?}");
    }
    // The whitespace after it is bounded, and a line break always left.
    let cache = cache_within(GrammarLimits { max_whitespace: 4, ..GrammarLimits::default() });
    let mut m = matcher_of(&cache, &[tool("edit", Some(EDIT))]);
    assert!(m.accept_bytes(format!("{head}12   ").as_bytes()));
    assert!(!m.clone().accept_bytes(b" ") && m.accept_bytes(b"\n"));
    assert!(!m.clone().accept_bytes(b"\n") && m.accept_bytes(b"</parameter>\n<parameter=note>\nx\n"));
}

#[test]
fn a_nullable_parameter_may_be_left_out_or_written_null() {
    for without in [&["note"][..], &["limit"], &["note", "limit"]] {
        assert!(takes_whole(edit_matcher(), &edit_call(&[], without)), "{without:?}");
    }
    for (name, value) in [("note", "null"), ("limit", "null"), ("note", "a note\nof two lines"), ("limit", "9")] {
        assert!(takes_whole(edit_matcher(), &edit_call(&[(name, value)], &[])), "{name}: {value}");
    }
    // A parameter that may not be null is still required.
    let head = "<tool_call>\n<function=edit>\n<parameter=note>\nhi\n</parameter>\n<";
    assert_eq!(edit_taken(&format!("{head}/function>")), head.len());
    // What the parser is told: the nullable ones are not required, and are
    // delivered `null` when left out; how each value reads.
    let grammar = cache().tool_calls(&[tool("edit", Some(EDIT))]).unwrap();
    let (_, parameters) = grammar.held().next().unwrap();
    let readings: Vec<(&str, Reading, bool)> =
        parameters.declared().iter().enumerate().map(|(i, name)| (name.as_str(), parameters.reading_of(i), parameters.nulled_when_omitted(i))).collect();
    assert_eq!(
        readings,
        [
            ("line", Reading::Json, false),
            ("dry_run", Reading::Json, false),
            ("change", Reading::Json, false),
            ("rows", Reading::Json, false),
            ("tree", Reading::Json, false),
            ("note", Reading::TextOrNull, true),
            ("limit", Reading::Json, true),
        ]
    );
    let all_but_nullable = vec![true, true, true, true, true, false, false];
    assert!(parameters.may_close(&all_but_nullable));
    assert!(!parameters.may_close(&[false, true, true, true, true, true, true]));
}

/// A strict tool of one required parameter `a` of this schema.
fn one(schema: &str) -> ToolCallMatcher {
    let parameters = format!(r#"{{"type":"object","properties":{{"a":{schema}}},"required":["a"],"additionalProperties":false}}"#);
    matcher_of(&cache(), &[tool("one", Some(&parameters))])
}

/// Whether a call of `one(schema)` with `value` is taken whole.
fn one_takes(schema: &str, value: &str) -> bool {
    takes_whole(one(schema), &call("one", &[("a", value)]))
}

#[test]
fn a_string_that_may_be_null_is_raw_text_and_its_text_null_is_null() {
    // A free one: any text, `null` the parser's null.
    let free = r#"{"type":["string","null"]}"#;
    for value in ["null", "\"quoted\"", "anything at all", "12"] {
        assert!(one_takes(free, value), "{value}");
    }
    // Held to its pattern or format, or `null`.
    let date = r#"{"type":["string","null"],"format":"date"}"#;
    for (value, ok) in [("2026-11-15", true), ("null", true), ("nul", false), ("nulls", false), ("null\"", false), ("2026-13-01", false)] {
        assert_eq!(one_takes(date, value), ok, "{value}");
    }
    let pattern_or_null = r#"{"anyOf":[{"type":"string","pattern":"^[a-z]+$"},{"type":"null"}]}"#;
    assert!(one_takes(pattern_or_null, "null") && one_takes(pattern_or_null, "abc") && !one_takes(pattern_or_null, "ABC"));
    // An enum: its strings and, where it holds null, `null`.
    let with_null = r#"{"type":["string","null"],"enum":["a",null]}"#;
    assert!(one_takes(with_null, "a") && one_takes(with_null, "null") && !one_takes(with_null, "b"));
    // The string "null" with no null beside it is that string.
    let the_word = r#"{"type":["string","null"],"enum":["null","a"]}"#;
    assert!(one_takes(the_word, "null") && one_takes(the_word, "a"));
    let readings = |schema: &str| {
        let parameters = format!(r#"{{"type":"object","properties":{{"a":{schema}}},"required":["a"],"additionalProperties":false}}"#);
        let grammar = cache().tool_calls(&[tool("one", Some(&parameters))]).unwrap();
        let (_, p) = grammar.held().next().unwrap();
        (p.reading_of(0), p.nulled_when_omitted(0))
    };
    for schema in [free, date, pattern_or_null, with_null] {
        assert_eq!(readings(schema), (Reading::TextOrNull, true), "{schema}");
    }
    assert_eq!(readings(the_word), (Reading::Text, false));
    assert_eq!(readings(r#"{"type":"string"}"#), (Reading::Text, false));
}

#[test]
fn a_union_no_raw_text_holds_is_json_a_string_in_it_quoted() {
    let cases: [(&str, &[&str], &[&str]); 4] = [
        (r#"{"anyOf":[{"type":"string"},{"type":"integer"}]}"#, &["\"abc\"", "5", "\"5\""], &["abc", "5.5"]),
        (r#"{"enum":["a",1,true]}"#, &["\"a\"", "1", "true"], &["a", "2"]),
        (r#"{"anyOf":[{"type":"string","pattern":"^a+$"},{"type":"string","format":"date"}]}"#, &["\"aa\"", "\"2026-11-15\""], &["aa", "\"b\""]),
        (r#"{"type":"null"}"#, &["null"], &["nil", "\"null\""]),
    ];
    for (schema, taken, missed) in cases {
        for value in taken {
            assert!(one_takes(schema, value), "{schema}: {value}");
        }
        for value in missed {
            assert!(!one_takes(schema, value), "{schema}: {value}");
        }
    }
}

// ── normalized tools (#614) ──────────────────────────────────────────────

/// `find`, as a client writes it without `strict`: its objects open, `path`
/// and `tag` required, `tag` and `note` nullable, `depth` and `opts`
/// optional, `opts` an object of one property.
const FIND: &str = r#"{"type":"object","properties":{"path":{"type":"string"},"depth":{"type":"integer"},"note":{"type":["string","null"]},"tag":{"type":["string","null"]},"opts":{"type":"object","properties":{"hidden":{"type":"boolean"}}}},"required":["path","tag"]}"#;

fn find_matcher() -> ToolCallMatcher {
    matcher_of(&cache(), &[normalized("find", FIND)])
}

#[test]
fn a_normalized_tool_is_held_its_objects_closed_and_its_optional_parameters_left_out() {
    let grammar = cache().tool_calls(&[normalized("find", FIND), tool("search", None)]).unwrap();
    assert!(grammar.holds(0) && !grammar.holds(1));
    // What the parser is told: a call may close with `path` alone, `tag`
    // (required, nullable) delivered `null` when left out, the optional ones
    // left out of the arguments.
    let (_, parameters) = grammar.held().next().unwrap();
    let nulled: Vec<(&str, bool)> = parameters.declared().iter().enumerate().map(|(i, name)| (name.as_str(), parameters.nulled_when_omitted(i))).collect();
    assert_eq!(nulled, [("path", false), ("depth", false), ("note", false), ("tag", true), ("opts", false)]);
    assert!(parameters.may_close(&[true, false, false, false, false]));
    assert!(!parameters.may_close(&[false, true, true, true, true]));
    assert!(takes_whole(find_matcher(), &call("find", &[("path", "src")])));
    let every = [("opts", r#"{"hidden": true}"#), ("depth", "2"), ("path", "src"), ("note", "null"), ("tag", "x")];
    assert!(takes_whole(find_matcher(), &call("find", &every)));
    // Closed: a parameter the schema does not declare is a miss at its name,
    // a key the nested object does not declare at its comma.
    let head = "<tool_call>\n<function=find>\n<parameter=";
    assert_eq!(taken(&mut find_matcher(), &format!("{head}force>")), head.len());
    let nested = "<tool_call>\n<function=find>\n<parameter=opts>\n{\"hidden\": true";
    assert_eq!(taken(&mut find_matcher(), &format!("{nested}, \"x\": 1}}")), nested.len());
    // Still typed: a word for the integer is a miss at its first letter.
    let depth = "<tool_call>\n<function=find>\n<parameter=depth>\n";
    assert_eq!(taken(&mut find_matcher(), &format!("{depth}two")), depth.len());
}

#[test]
fn a_normalized_tool_without_parameters_is_held_to_none() {
    for parameters in [r#"{"type":"object","properties":{}}"#, r#"{"type":"object"}"#] {
        let m = matcher_of(&cache(), &[normalized("now", parameters)]);
        assert!(m.grammar().holds(0), "{parameters}");
        assert!(takes_whole(m.clone(), &call("now", &[])), "{parameters}");
        // No parameter is declared, so none opens: the miss is at its `p`.
        let head = "<tool_call>\n<function=now>\n<";
        assert_eq!(taken(&mut m.clone(), &format!("{head}parameter=zone>")), head.len(), "{parameters}");
    }
}

#[test]
fn a_tool_that_cannot_be_normalized_is_left_free_and_never_refused() {
    let object = |inner: &str| format!(r#"{{"type":"object","properties":{{"a":{inner}}}}}"#);
    let cases: Vec<(&str, Vec<DeclaredTool>)> = vec![
        ("a keyword outside the strict subset", vec![normalized("g", &object(r#"{"type":"string","minLength":3}"#))]),
        ("an object open by name", vec![normalized("g", r#"{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":true}"#)]),
        ("a map", vec![normalized("g", &object(r#"{"type":"object","additionalProperties":{"type":"string"}}"#))]),
        ("a nested object that declares no property", vec![normalized("g", &object(r#"{"type":"object"}"#))]),
        ("a required key the object does not declare", vec![normalized("g", r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["b"]}"#)]),
        ("a root that is not an object", vec![normalized("g", r#"{"type":"string"}"#)]),
        ("an untyped schema", vec![normalized("g", r#"{"properties":{"a":{"type":"string"}}}"#)]),
        ("a name the dialect cannot write", vec![normalized("a>b", &object(r#"{"type":"string"}"#))]),
        ("a name another tool has", vec![tool("g", None), normalized("g", &object(r#"{"type":"string"}"#))]),
        ("a parameter name the dialect cannot write", vec![normalized("g", r#"{"type":"object","properties":{"a>b":{"type":"string"}}}"#)]),
        ("a value no line can hold", vec![normalized("g", &object(r#"{"type":"string","enum":["a\nb"]}"#))]),
    ];
    for (what, tools) in cases {
        let grammar = cache().tool_calls(&tools).unwrap_or_else(|refused| panic!("{what}: {refused:?}"));
        assert!(grammar.held().next().is_none(), "{what}");
        // A strict tool beside it is held still.
        let mut beside = tools.clone();
        beside.push(tool("write_file", Some(WRITE_FILE)));
        let grammar = cache().tool_calls(&beside).unwrap_or_else(|refused| panic!("{what}: {refused:?}"));
        assert!((0..tools.len()).all(|i| !grammar.holds(i)) && grammar.holds(tools.len()), "{what}");
    }
}

#[test]
fn a_schema_sent_strict_and_normalized_is_two_grammars() {
    let open = r#"{"type":"object","properties":{"a":{"type":"string"}}}"#;
    let cache = cache();
    assert!(cache.tool_calls(&[normalized("g", open)]).unwrap().holds(0));
    assert_eq!(cache.tool_calls(&[tool("g", Some(open))]).unwrap_err().refusal.code, "additional_properties_not_false");
    assert!(cache.tool_calls(&[normalized("g", open)]).unwrap().holds(0));
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

/// `edit`'s pieces beside a free tool's (#612): the dialect's, JSON's, and
/// text.
const TYPED_SOUP: &[&str] = &[
    "<tool_call>", "\n", "<function=edit>", "<function=search>", "<parameter=line>", "<parameter=dry_run>",
    "<parameter=change>", "<parameter=rows>", "<parameter=tree>", "<parameter=note>", "<parameter=limit>", "</parameter>",
    "</function>", "</tool_call>", "{", "}", "[", "]", "\"", ",", ":", " ", "\"edits\"", "\"old\"", "\"new\"", "\"k\"", "\"v\"",
    "\"name\"", "\"kids\"", "1", "2", "0", "-", ".", "e", "true", "false", "null", "x", "é", "<", "</",
    "12\n</parameter>\n", "\"edits\": [", "{\"old\": \"a\", \"new\": \"b\"}", "{\"name\": \"n\", \"kids\": []}", "{\"k\": \"a\", \"v\": 1}",
];

/// `edit`, held, and `search`, free, with whitespace runs of at most two
/// bytes (fewer spellings of the same JSON for a search to go through).
fn typed_matcher() -> ToolCallMatcher {
    let cache = cache_within(GrammarLimits { max_whitespace: 2, ..GrammarLimits::default() });
    matcher_of(&cache, &[tool("edit", Some(EDIT)), tool("search", None)])
}

/// The two generated readings: `SOUP`'s over the string tools, and
/// `TYPED_SOUP`'s over the typed one, half its steps a byte the grammar
/// takes there.
const READINGS: [(fn() -> ToolCallMatcher, &[&str], bool); 2] = [(matcher, SOUP, false), (typed_matcher, TYPED_SOUP, true)];

/// A text the grammar accepts, grown from `soup`'s pieces in random order
/// (a piece refused is skipped), and with `bytes`, half the time from a byte
/// the grammar takes (printable or a line break, a backslash aside: an
/// escape's hex digits make thousands of states for a search), with the
/// matcher at each piece's end.
fn generated(fresh: fn() -> ToolCallMatcher, soup: &[&str], bytes: bool, seed: u64, pieces: usize) -> (Vec<u8>, Vec<ToolCallMatcher>) {
    let mut rng = XorShift(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut m = fresh();
    let mut text = Vec::new();
    let mut states = Vec::new();
    for _ in 0..pieces * 8 {
        if states.len() >= pieces {
            break;
        }
        if bytes && rng.next_u64() % 2 == 0 {
            let allowed: Vec<u8> = m.allowed_bytes().into_iter().filter(|&b| (b == b'\n' || (0x20..0x7F).contains(&b)) && b != b'\\').collect();
            if let Some(&b) = allowed.get((rng.next_u64() % allowed.len().max(1) as u64) as usize) {
                assert!(m.accept_bytes(&[b]));
                text.push(b);
                states.push(m.clone());
            }
            continue;
        }
        let piece = soup[(rng.next_u64() % soup.len() as u64) as usize];
        if m.accept_bytes(piece.as_bytes()) {
            text.extend_from_slice(piece.as_bytes());
            states.push(m.clone());
        }
    }
    (text, states)
}

/// Whether some bytes lead `m` back to free text: a breadth-first search
/// over states, printable bytes and the line break (no space: a run of
/// mixed whitespace makes as many states as it has spellings), one value
/// at a time: from each state it reaches, to one with another parameter
/// written, or free text (a whole call's search spends its budget on the
/// orders of a typed tool's other parameters and the spellings of its JSON).
fn completes(m: &ToolCallMatcher) -> bool {
    let g = &m.grammar;
    let alphabet: Vec<u8> = (0x21..0x7F).filter(|&b| b != b'\\').chain([b'\n']).collect();
    let written = |s: &CallState| s.written.iter().filter(|&&w| w).count();
    let mut from = m.state.clone();
    'value: loop {
        if ToolCallGrammar::outside_held(&from) {
            return true;
        }
        let (mut seen, mut queue) = (HashSet::new(), VecDeque::from([from.clone()]));
        while let Some(s) = queue.pop_front() {
            if ToolCallGrammar::outside_held(&s) || written(&s) > written(&from) {
                from = s;
                continue 'value;
            }
            assert!(seen.len() <= 200_000, "search budget spent");
            for &b in &alphabet {
                let mut next = s.clone();
                if g.step(&mut next, b) && seen.insert(next.clone()) {
                    queue.push_back(next);
                }
            }
        }
        return false;
    }
}

#[test]
fn no_state_is_a_dead_end() {
    for (fresh, soup, bytes) in READINGS {
        let mut held = 0;
        for seed in 0..16 {
            let (text, states) = generated(fresh, soup, bytes, seed, 40);
            for m in states.iter().filter(|m| m.in_call()).step_by(2) {
                assert!(completes(m), "a dead end after {:?}", String::from_utf8_lossy(&text));
                held += 1;
            }
        }
        assert!(held > 50, "{held}");
    }
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
    for (fresh, soup, bytes) in READINGS {
        let mut states = 0;
        for seed in 0..40 {
            let (text, _) = generated(fresh, soup, bytes, seed, 30);
            let mut m = fresh();
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
}
