//! The automaton against hand-written documents, the mask against the
//! per-token reference, and random documents of the language against JSON
//! itself, on a toy byte-level vocabulary.

use std::sync::Arc;

use super::*;

const STOP: u32 = 400;

/// Every byte as a token (as in a byte-level BPE vocabulary), the multi-byte
/// shapes a real vocabulary has for JSON, one stop id, one absent column.
fn vocab() -> Arc<Vocab> {
    let mut entries: Vec<VocabEntry> = (0..=255u8).map(|b| VocabEntry::Text(vec![b])).collect();
    let multi: &[&str] = &[
        "{\"", "\":", "\",", "\"}", "\":\"", "\",\"", " \"", "\n  ", "\n", "  ", "true", "false", "null", "name", "river",
        "\"river", "Danube", "12", ".5", "},", "}]", "\"]", "[{\"", "abc\"", "\"abc\"", "-0", "2024-", "\": ", "\":[", "\":{\"",
        "\"}\n", "\"},", ",\n", "\n}", "\n]", "{\n", "[\n", "    ", " {", " [", "\\n", "\\u00", "é", "河流", "@example.com",
        "type", "\"type\":\"", "items", "value", "children", "\":[]", "]}", "}}", "0.", "00", "1e", "e+", "\":1", ",\"",
    ];
    for m in multi {
        entries.push(VocabEntry::Text(m.as_bytes().to_vec()));
    }
    while entries.len() < STOP as usize {
        entries.push(VocabEntry::Absent);
    }
    entries.push(VocabEntry::Control); // STOP
    entries.push(VocabEntry::Control); // a control token that is not a stop
    Vocab::new(entries, &[STOP])
}

fn tokenize(v: &Vocab, text: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let mut best = (0usize, 0u32);
        for id in 0..v.len() as u32 {
            if let Some(b) = v.token_bytes(id)
                && b.len() > best.0
                && text[i..].starts_with(b)
            {
                best = (b.len(), id);
            }
        }
        assert!(best.0 > 0);
        out.push(best.1);
        i += best.0;
    }
    out
}

fn accepts(g: &Arc<Grammar>, text: &str) -> bool {
    let mut m = Matcher::new(g.clone());
    for t in tokenize(&g.vocab, text.as_bytes()) {
        if !m.accept(t) {
            return false;
        }
    }
    m.accept(STOP)
}

fn schema(v: &Arc<Vocab>, s: &str) -> Arc<Grammar> {
    Grammar::json_schema(v.clone(), s, true).unwrap_or_else(|e| panic!("{e}"))
}

const TRAIN: &str = r#"{"type":"object","properties":{
  "train":{"type":"string","pattern":"^[A-Z]{2}\\d{3,4}$"},
  "stops":{"type":"array","minItems":1,"items":{"type":"object","properties":{
     "station":{"type":"string"},"platform":{"type":"integer","minimum":1},"wait_min":{"type":"number"}},
     "required":["station","platform","wait_min"],"additionalProperties":false}},
  "operator":{"type":"string"}},
  "required":["train","stops","operator"],"additionalProperties":false}"#;

const TREE: &str = r##"{"type":"object","properties":{"root":{"$ref":"#/$defs/node"}},"required":["root"],"additionalProperties":false,
  "$defs":{"node":{"type":"object","properties":{"value":{"type":"string"},"children":{"type":"array","items":{"$ref":"#/$defs/node"}}},
  "required":["value","children"],"additionalProperties":false}}}"##;

const UNION: &str = r#"{"type":"object","properties":{
  "item":{"anyOf":[
     {"type":"object","properties":{"type":{"type":"string","enum":["book"]},"title":{"type":"string"}},"required":["type","title"],"additionalProperties":false},
     {"type":"object","properties":{"type":{"type":"string","enum":["movie"]},"year":{"type":"integer","minimum":1888,"maximum":2100}},"required":["type","year"],"additionalProperties":false}]},
  "note":{"type":["string","null"]},
  "score":{"type":"number","minimum":0,"maximum":1},
  "email":{"type":"string","format":"email"}},
  "required":["item","note","score","email"],"additionalProperties":false}"#;

#[test]
fn json_object_accepts_objects_only() {
    let v = vocab();
    let g = Grammar::json_object(v.clone());
    assert!(accepts(&g, r#"{"river": "Danube", "length_km": 2850.5, "mouth": {"delta": true}, "bridges": ["a", "b", {"x": null}]}"#));
    assert!(accepts(&g, "{\n  \"a\": [1, -0.5e+3, false]\n}\n"));
    assert!(accepts(&g, "{}"));
    for bad in ["[1]", "\"x\"", "{\"a\" 1}", "{\"a\":01}", "{\"a\":[1,]}", "{\"a\":tru}", "{'a':1}", "{\"a\":1}}", "{\"a\":\"\n\"}"] {
        assert!(!accepts(&g, bad), "{bad}");
    }
}

#[test]
fn whitespace_runs_are_bounded() {
    let v = vocab();
    let g = Grammar::json_object(v.clone());
    let ok = format!("{{{}}}", " ".repeat(DEFAULT_MAX_WS_RUN as usize));
    let long = format!("{{{}}}", " ".repeat(DEFAULT_MAX_WS_RUN as usize + 1));
    assert!(accepts(&g, &ok));
    assert!(!accepts(&g, &long));
    // A stop token is valid only once the root object is complete.
    let mut m = Matcher::new(g.clone());
    assert!(!m.check(STOP));
    assert!(m.accept_bytes(b"{}"));
    assert!(m.check(STOP));
    assert!(!m.check(STOP + 1), "a control token that is not a stop is never text");
    assert!(!m.check(300), "an absent column is never valid");
}

#[test]
fn timetable_schema() {
    let v = vocab();
    let g = schema(&v, TRAIN);
    assert!(accepts(&g, r#"{"train":"RE4521","stops":[{"station":"Verona","platform":3,"wait_min":2.5}],"operator":"Regional Rail"}"#));
    assert!(accepts(&g, "{\n  \"train\": \"IC612\",\n  \"stops\": [\n    {\"station\": \"Trento\", \"platform\": 12, \"wait_min\": 5e-1}\n  ],\n  \"operator\": \"Regional Rail\"\n}"));
    for bad in [
        r#"{"stops":[],"train":"RE4521","operator":"x"}"#,                                                  // key order
        r#"{"train":"RE4521","stops":[],"operator":"x"}"#,                                                  // minItems 1
        r#"{"train":"RE4521","stops":[{"station":"s","platform":0,"wait_min":1}],"operator":"x"}"#,         // minimum 1
        r#"{"train":"RE4521","stops":[{"station":"s","platform":1.5,"wait_min":1}],"operator":"x"}"#,       // integer
        r#"{"train":"re4521","stops":[{"station":"s","platform":1,"wait_min":1}],"operator":"x"}"#,         // pattern
        r#"{"train":"RE4521","stops":[{"station":"s","platform":1,"wait_min":1}],"operator":"x","x":1}"#,   // extra key
        r#"{"train":"RE4521","stops":[{"station":"s","platform":1,"wait_min":1}]}"#,                        // missing key
    ] {
        assert!(!accepts(&g, bad), "{bad}");
    }
}

#[test]
fn recursion_and_unions() {
    let v = vocab();
    let t = schema(&v, TREE);
    assert!(accepts(&t, r#"{"root":{"value":"a","children":[{"value":"b","children":[]},{"value":"c","children":[{"value":"d","children":[]}]}]}}"#));
    assert!(!accepts(&t, r#"{"root":{"value":"a","children":[{"value":"b"}]}}"#));
    let u = schema(&v, UNION);
    assert!(accepts(&u, r#"{"item":{"type":"book","title":"Dune"},"note":null,"score":0.5,"email":"a@example.com"}"#));
    assert!(accepts(&u, r#"{"item":{"type":"movie","year":1999},"note":"x","score":1,"email":"x.y@b.co"}"#));
    assert!(!accepts(&u, r#"{"item":{"type":"movie","title":"Dune"},"note":null,"score":0.5,"email":"a@example.com"}"#));
    assert!(!accepts(&u, r#"{"item":{"type":"movie","year":1887},"note":null,"score":0.5,"email":"a@example.com"}"#));
    assert!(!accepts(&u, r#"{"item":{"type":"book","title":"Dune"},"note":null,"score":1.5,"email":"a@example.com"}"#));
    assert!(!accepts(&u, r#"{"item":{"type":"book","title":"Dune"},"note":null,"score":0.5,"email":"not-an-email"}"#));
}

#[test]
fn mask_inside_a_matched_unanchored_pattern_equals_the_reference() {
    // Once an unanchored pattern has matched, the rest of the string is
    // free, but it is still a pattern string: no `\u` escape is written in
    // it, so the mask must not offer `\u00`.
    let v = vocab();
    let g = schema(&v, r#"{"type":"object","properties":{"code":{"type":"string","pattern":"\\d"}},"required":["code"],"additionalProperties":false}"#);
    let mut m = Matcher::new(g);
    assert!(m.accept_bytes(br#"{"code":"x1y"#));
    let mut out = TokenSet::new(v.len());
    let mut reference = TokenSet::new(v.len());
    m.mask(&mut out);
    m.mask_reference(&mut reference);
    assert_eq!(out, reference);
}

#[test]
fn a_unicode_escape_in_a_free_string_is_masked_exactly_and_grows_no_shared_table() {
    let v = vocab();
    let g = Grammar::json_object(v.clone());
    let before = v.heap_bytes();
    let mut out = TokenSet::new(v.len());
    let mut reference = TokenSet::new(v.len());
    // Every partial \u escape names a different lexer state; none of them
    // may leave a table behind in the vocabulary all grammars share.
    for prefix in [&br#"{"a":"\u"#[..], br#"{"a":"\u0"#, br#"{"a":"\u00e"#, br#"{"a":"\ud83d"#, br#"{"a":"\ud83d\u"#, br#"{"a":"\ud83d\ude0"#, br#"{"a":"\u12"#] {
        let mut m = Matcher::new(g.clone());
        assert!(m.accept_bytes(prefix), "{}", String::from_utf8_lossy(prefix));
        m.mask(&mut out);
        m.mask_reference(&mut reference);
        assert_eq!(out, reference, "{}", String::from_utf8_lossy(prefix));
        assert!(!out.is_empty());
    }
    assert_eq!(v.heap_bytes(), before);
}

/// A linked list whose every cell must hold a `next` item: the recursion
/// ends only through the `null` alternative.
const LIST: &str = r##"{"type":"object","properties":{"head":{"$ref":"#/$defs/cell"}},"required":["head"],"additionalProperties":false,
  "$defs":{"cell":{"type":"object","properties":{"v":{"type":"integer"},"next":{"type":"array","minItems":1,"maxItems":1,
  "items":{"anyOf":[{"$ref":"#/$defs/cell"},{"type":"null"}]}}},"required":["v","next"],"additionalProperties":false}}}"##;

#[test]
fn a_schema_with_no_finite_value_is_refused_by_name() {
    let v = vocab();
    let endless = r##"{"type":"object","properties":{"child":{"$ref":"#"}},"required":["child"],"additionalProperties":false}"##;
    let r = Grammar::json_schema(v.clone(), endless, true).expect_err("refused");
    assert_eq!((r.code, r.keyword.as_str()), ("no_finite_value", "$ref"));
}

#[test]
fn an_alternative_with_no_finite_value_is_never_entered() {
    let v = vocab();
    let g = schema(&v, r##"{"type":"object","properties":{"a":{"anyOf":[{"$ref":"#/$defs/loop"},{"type":"null"}]}},"required":["a"],"additionalProperties":false,
      "$defs":{"loop":{"type":"object","properties":{"x":{"$ref":"#/$defs/loop"}},"required":["x"],"additionalProperties":false}}}"##);
    assert!(accepts(&g, r#"{"a":null}"#));
    let mut m = Matcher::new(g);
    assert!(!m.accept_bytes(br#"{"a":{"#), "the object alternative can never be finished");
}

/// A completion of the matcher's state: a steered random walk that must
/// reach a complete document; the prefix and the completion parse as JSON.
fn assert_completes(m: &Matcher, prefix: &[u8], seed: u64) {
    let rest = random_completion(m, seed, 200_000)
        .unwrap_or_else(|e| panic!("{e}"))
        .unwrap_or_else(|| panic!("no completion within the budget after {:?}", String::from_utf8_lossy(prefix)));
    let mut done = m.clone();
    assert!(done.accept_bytes(&rest) && done.is_accepting());
    let text = [prefix, &rest].concat();
    let text = String::from_utf8(text).expect("UTF-8");
    serde_json::from_str::<serde_json::Value>(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
}

#[test]
fn the_deepest_nesting_a_grammar_allows_never_strands_the_model() {
    let v = vocab();
    for (gi, g) in [Grammar::json_object(v.clone()), schema(&v, TREE), schema(&v, LIST)].iter().enumerate() {
        // Open a container whenever one may open, close strings at once:
        // the walk runs into the automaton's stack bound and stays there.
        let mut m = Matcher::new(g.clone());
        let mut rng = XorShift(0x5EED + gi as u64);
        let mut out = Vec::new();
        let (mut depth, mut deepest) = (0i32, 0i32);
        for step in 0..4000 {
            let allowed = m.allowed_bytes();
            if allowed.is_empty() {
                // Only a list ends: its innermost cell must be `null`.
                assert!(m.is_accepting(), "grammar {gi}: dead end after {:?}", String::from_utf8_lossy(&out));
                break;
            }
            let pick = b"[{\":,".iter().copied().find(|b| allowed.contains(b))
                .unwrap_or_else(|| allowed[(rng.next_u64() % allowed.len() as u64) as usize]);
            assert!(m.accept_bytes(&[pick]));
            out.push(pick);
            depth += match pick { b'[' | b'{' => 1, b']' | b'}' => -1, _ => 0 };
            deepest = deepest.max(depth);
            if step % 97 == 0 {
                assert_completes(&m, &out, step);
            }
        }
        assert!(deepest >= 50, "grammar {gi}: the walk only reached depth {deepest}");
        assert_completes(&m, &out, 1);
    }
}

#[test]
fn from_every_state_a_random_walk_reaches_a_valid_completion_exists() {
    let v = vocab();
    let mut states = 0;
    for (gi, g) in grammars(&v).iter().enumerate() {
        for seed in 0..6u64 {
            let Some(doc) = random_document(g, 7000 + seed * 13 + gi as u64, 60, 6000).unwrap() else { continue };
            let mut m = Matcher::new(g.clone());
            for (i, &b) in doc.iter().enumerate() {
                assert_completes(&m, &doc[..i], seed * 1000 + i as u64);
                assert!(m.accept_bytes(&[b]));
                states += 1;
            }
        }
    }
    assert!(states > 600, "{states}");
}

fn grammars(v: &Arc<Vocab>) -> Vec<Arc<Grammar>> {
    vec![Grammar::json_object(v.clone()), schema(v, TRAIN), schema(v, TREE), schema(v, UNION), schema(v, LIST)]
}

#[test]
fn random_documents_are_json_and_never_dead_end() {
    let v = vocab();
    for (gi, g) in grammars(&v).iter().enumerate() {
        let mut made = 0;
        for seed in 0..150u64 {
            let doc = random_document(g, seed * 31 + gi as u64, 60, 6000).unwrap_or_else(|e| panic!("grammar {gi}: {e}"));
            let Some(doc) = doc else { continue };
            let text = String::from_utf8(doc).expect("valid UTF-8");
            let parsed: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("grammar {gi}: {e}: {text}"));
            assert!(parsed.is_object());
            assert!(accepts(g, &text), "token-level replay of {text}");
            made += 1;
        }
        assert!(made > 100, "grammar {gi}: only {made} documents finished");
    }
}

#[test]
fn mask_equals_the_reference_on_every_state_of_random_documents() {
    let v = vocab();
    let mut out = TokenSet::new(v.len());
    let mut reference = TokenSet::new(v.len());
    let mut states = 0;
    for (gi, g) in grammars(&v).iter().enumerate() {
        for seed in 0..12u64 {
            let Some(doc) = random_document(g, 1000 + seed, 40, 4000).unwrap() else { continue };
            let mut m = Matcher::new(g.clone());
            for t in tokenize(&v, &doc).into_iter().chain([STOP]) {
                m.mask(&mut out);
                m.mask_reference(&mut reference);
                assert_eq!(out, reference, "grammar {gi} at {:?}", m.position());
                assert!(out.contains(t), "the document's own next token is in the mask");
                // Second visit: from the cache, identical.
                assert!(m.mask(&mut out));
                assert_eq!(out, reference);
                states += 1;
                assert!(m.accept(t));
            }
        }
    }
    assert!(states > 500, "{states}");
}

#[test]
fn a_schema_sent_again_is_compiled_once() {
    let cache = GrammarCache::new(vocab(), DEFAULT_CACHE_BYTES);
    let a = cache.json_schema(TRAIN, true).unwrap();
    let b = cache.json_schema(&TRAIN.replace(':', " : "), true).unwrap();
    assert!(Arc::ptr_eq(&a, &b), "the same schema, spaced differently");
    assert_eq!(cache.compiles(), 1);
    cache.json_schema(TRAIN, false).unwrap();
    assert_eq!(cache.compiles(), 2, "strict and not strict are two grammars");
    for _ in 0..2 {
        assert_eq!(cache.json_schema(r#"{"type":"string"}"#, true).unwrap_err().code, "root_not_object");
        cache.json_object();
    }
    assert_eq!(cache.compiles(), 4, "a refusal and json_object are kept too");
    assert_eq!(cache.len(), 3);
}

#[test]
fn the_cache_stays_within_its_memory_by_dropping_the_least_recently_used() {
    let v = vocab();
    let variant = |i: usize| TRAIN.replace("operator", &format!("operator{i}"));
    let one = Grammar::json_schema(v.clone(), &variant(0), true).unwrap().heap_bytes();
    let capacity = one * 7 / 2;
    let cache = GrammarCache::new(v.clone(), capacity);
    let first = cache.json_schema(&variant(0), true).unwrap();
    for i in 1..10 {
        cache.json_schema(&variant(i), true).unwrap();
        // The first schema, used again each time, is never the oldest.
        assert!(Arc::ptr_eq(&first, &cache.json_schema(&variant(0), true).unwrap()));
        assert!(cache.heap_bytes() <= capacity, "{} > {capacity}", cache.heap_bytes());
    }
    assert_eq!(cache.len(), 3);
    assert_eq!(cache.compiles(), 10);
}

#[test]
fn a_grammar_whose_masks_grew_is_brought_back_within_the_bound_at_the_next_lookup() {
    let v = vocab();
    let variant = |i: usize| TRAIN.replace("operator", &format!("operator{i}"));
    let doc = br#"{"train":"AB123","stops":[{"station":"x","platform":1,"wait_min":2}],"operator0":"y"}"#;
    let use_it = |g: &Arc<Grammar>| {
        let mut m = Matcher::new(g.clone());
        let mut out = TokenSet::new(v.len());
        for b in doc {
            m.mask(&mut out);
            assert!(m.accept_bytes(&[*b]));
        }
    };
    let one = Grammar::json_schema(v.clone(), &variant(0), true).unwrap().heap_bytes();
    let grown = {
        let g = Grammar::json_schema(v.clone(), &variant(0), true).unwrap();
        use_it(&g);
        g.heap_bytes()
    };
    assert!(grown > one);
    // Three fresh grammars fit; one grown beside two fresh ones does not.
    let capacity = 2 * one + grown - 1;
    let cache = GrammarCache::new(v.clone(), capacity);
    let first = cache.json_schema(&variant(0), true).unwrap();
    cache.json_schema(&variant(1), true).unwrap();
    cache.json_schema(&variant(2), true).unwrap();
    assert_eq!(cache.len(), 3);
    use_it(&first);
    cache.json_schema(&variant(0), true).unwrap();
    assert_eq!(cache.len(), 2, "the least recently used schema is dropped");
    assert!(cache.heap_bytes() <= capacity);
}

#[test]
fn a_run_is_accepted_up_to_its_first_miss_and_the_miss_is_repaired_from_there() {
    let v = vocab();
    let g = schema(&v, TRAIN);
    let good = tokenize(&v, br#"{"train":"RE4521","#);
    // The next key is "stops": its opening quote passes, the `r` misses.
    let run: Vec<u32> = good.iter().copied().chain(tokenize(&v, br#""route":1"#)).collect();
    let mut m = Matcher::new(g.clone());
    let n = m.accept_run(&run);
    assert_eq!(n, good.len() + 1);
    let mut mask = TokenSet::new(v.len());
    m.mask(&mut mask);
    assert!(!mask.contains(run[n]) && mask.contains(b's' as u32), "the repair writes the schema's key");
    assert!(m.accept(b's' as u32));
    // A whole valid answer and its stop: every token, and the matcher ends.
    let doc = tokenize(&v, br#"{"train":"RE4521","stops":[{"station":"V","platform":3,"wait_min":2.5}],"operator":"R"}"#);
    let mut m = Matcher::new(g);
    let all: Vec<u32> = doc.iter().copied().chain([STOP, b' ' as u32]).collect();
    assert_eq!(m.accept_run(&all), doc.len() + 1, "nothing follows the stop");
    assert!(m.is_finished());
}

#[test]
fn the_answer_starts_at_its_first_byte_past_leading_whitespace() {
    let v = vocab();
    for g in [Grammar::json_object(v.clone()), schema(&v, TRAIN)] {
        let mut m = Matcher::new(g);
        assert!(!m.answer_started());
        assert!(m.accept_bytes(b"\n\n "));
        assert!(!m.answer_started(), "after leading whitespace another opener may still come");
        assert!(m.accept_bytes(b"{"));
        assert!(m.answer_started());
    }
}

// ── The grammar's own limits are the caller's (`model.grammar_*`) ─────────

fn within(max: impl FnOnce(&mut GrammarLimits)) -> GrammarLimits {
    let mut limits = GrammarLimits::default();
    max(&mut limits);
    limits
}

#[test]
fn a_lowered_whitespace_bound_holds_the_model_to_it() {
    let v = vocab();
    let g = Grammar::json_object_within(v.clone(), &within(|l| l.max_whitespace = 4));
    assert!(accepts(&g, "{    }"));
    assert!(!accepts(&g, "{     }"), "five is past the bound of four");
    let g = Grammar::json_schema_within(v, TRAIN, true, &within(|l| l.max_whitespace = 1)).unwrap();
    let mut m = Matcher::new(g);
    assert!(m.accept_bytes(b"{ "));
    assert!(!m.accept_bytes(b" "), "a schema's grammar holds the same bound");
}

#[test]
fn a_pattern_past_the_state_bound_is_refused_under_strict_and_inert_without() {
    let v = vocab();
    let wide = r#"{"type":"object","properties":{"code":{"type":"string","pattern":"^[a-z]{40}$"}},"required":["code"],"additionalProperties":false}"#;
    let date = r#"{"type":"object","properties":{"day":{"type":"string","format":"date"}},"required":["day"],"additionalProperties":false}"#;
    let tight = within(|l| l.max_pattern_states = 8);
    for (schema, keyword) in [(wide, "pattern"), (date, "format")] {
        let refusal = Grammar::json_schema_within(v.clone(), schema, true, &tight).unwrap_err();
        assert_eq!((refusal.code, refusal.keyword.as_str()), ("pattern_too_large", keyword), "{refusal}");
        assert!(refusal.detail.contains("more than 8 states"), "{refusal}");
        let g = Grammar::json_schema_within(v.clone(), schema, false, &tight).unwrap();
        assert_eq!(g.inert().iter().map(|i| i.keyword.as_str()).collect::<Vec<_>>(), [keyword]);
        // Within the default bound both are held.
        assert!(Grammar::json_schema(v.clone(), schema, true).is_ok());
    }
}

#[test]
fn a_schema_deeper_than_the_depth_bound_is_refused_by_its_own_code() {
    let v = vocab();
    let nested = |levels: usize| {
        (0..levels).fold(r#"{"type":"string"}"#.to_owned(), |inner, _| {
            format!(r#"{{"type":"object","properties":{{"a":{inner}}},"required":["a"],"additionalProperties":false}}"#)
        })
    };
    let shallow = within(|l| l.max_depth = 6);
    let refusal = Grammar::json_schema_within(v.clone(), &nested(8), true, &shallow).unwrap_err();
    assert_eq!((refusal.code, refusal.keyword.as_str()), ("exceeds_max_depth", "depth"), "{refusal}");
    assert!(refusal.detail.contains("6 levels"), "{refusal}");
    assert!(Grammar::json_schema_within(v.clone(), &nested(2), true, &shallow).is_ok());
    assert!(Grammar::json_schema(v.clone(), &nested(8), true).is_ok(), "the default bound holds it");
    // A schema no finite answer satisfies keeps its own code.
    let endless = r##"{"type":"object","properties":{"child":{"$ref":"#"}},"required":["child"],"additionalProperties":false}"##;
    assert_eq!(Grammar::json_schema_within(v.clone(), endless, true, &shallow).unwrap_err().code, "no_finite_value");
    // And the automaton opens no value it has no room to finish.
    let g = Grammar::json_object_within(v, &within(|l| l.max_depth = 6));
    let mut m = Matcher::new(g);
    let mut opened = 0;
    while m.accept_bytes(b"{\"a\":") {
        opened += 1;
        assert!(opened < 10, "the bound of 6 never closed in");
    }
    assert!((2..6).contains(&opened), "{opened} objects open under a bound of 6");
    assert_completes(&m, &b"{\"a\":".repeat(opened), 7);
}

#[test]
fn the_compilers_recursion_is_bounded_by_the_ref_chain_limit() {
    let v = vocab();
    let nested = (0..4).fold(r#"{"type":"string"}"#.to_owned(), |inner, _| {
        format!(r#"{{"type":"object","properties":{{"a":{inner}}},"required":["a"],"additionalProperties":false}}"#)
    });
    let refusal = Grammar::json_schema_within(v.clone(), &nested, true, &within(|l| l.max_ref_chain = 3)).unwrap_err();
    assert_eq!(refusal.code, "references_too_deep", "{refusal}");
    assert!(refusal.detail.contains("deeper than 3"), "{refusal}");
    assert!(Grammar::json_schema_within(v, &nested, true, &within(|l| l.max_ref_chain = 16)).is_ok());
}

/// The server's widest bounds (`model.grammar_max_ref_chain` 1024,
/// `model.grammar_max_depth` 4096) compile a schema at the edge of both on a
/// thread's default stack, as a request's handler does.
#[test]
fn the_widest_bounds_the_server_allows_compile_on_a_default_stack() {
    let v = vocab();
    let levels = 500;
    let defs: Vec<String> = (0..levels)
        .map(|i| format!(r##""d{i}":{{"type":"object","properties":{{"x":{{"$ref":"#/$defs/d{}"}}}},"required":["x"],"additionalProperties":false}}"##, i + 1))
        .collect();
    let schema = format!(
        r##"{{"type":"object","properties":{{"a":{{"$ref":"#/$defs/d0"}}}},"required":["a"],"additionalProperties":false,"$defs":{{{},"d{levels}":{{"type":"null"}}}}}}"##,
        defs.join(",")
    );
    let widest = within(|l| {
        l.max_ref_chain = 1024;
        l.max_depth = 4096;
    });
    let compiled = std::thread::spawn(move || Grammar::json_schema_within(v, &schema, true, &widest).map(|_| ()))
        .join()
        .expect("no stack overflow");
    assert_eq!(compiled, Ok(()));
}

#[test]
fn a_grammars_own_caches_stop_growing_at_the_cache_bytes() {
    let v = vocab();
    let doc = br#"{"train":"AB123","stops":[{"station":"x","platform":1,"wait_min":2}],"operator":"y"}"#;
    let grow = |limits: &GrammarLimits| {
        let g = Grammar::json_schema_within(v.clone(), TRAIN, true, limits).unwrap();
        let mut m = Matcher::new(g.clone());
        let mut out = TokenSet::new(v.len());
        for b in doc {
            m.mask(&mut out);
            assert!(m.accept_bytes(&[*b]));
        }
        g.cache_bytes()
    };
    let grown = grow(&GrammarLimits::default());
    assert!(grown > 0, "masks and tables are kept by default");
    assert_eq!(grow(&within(|l| l.cache_bytes = 0)), 0, "nothing is kept in no room");
    let half = grown / 2;
    let kept = grow(&within(|l| l.cache_bytes = half));
    assert!(kept > 0 && kept <= half, "{kept} of {half}");
}
