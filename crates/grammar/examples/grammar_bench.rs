//! The grammar measured on the real 27B tokenizer (GitHub #585, spec #287;
//! the method of the #287 spike's part A).
//!
//! ```text
//! cargo run --release -p ignis-grammar --example grammar_bench -- \
//!     ARTIFACT CORPUS_JSON OUT_DIR [REFERENCE_STATES_PER_DOC]
//! python -I tools/grammar-check/validate.py OUT_DIR/crosscheck.jsonl
//! ```
//!
//! Loads the tokenizer the way the server does (the artifact's frontend set),
//! decodes every logits column to its bytes once, builds the vocabulary trie
//! and the free-string tables, then for every case of the corpus: compiles the
//! grammar (timed), replays each valid document as the tokenizer encodes it
//! (pretty and compact), timing CHECK on every token and MASK on every state,
//! first visit and cached, holding the mask to the per-token reference on
//! sampled states and to the document's own next token on every state;
//! replays the schema violations (each must be refused); generates random
//! documents of the grammar's language. It also times compiling an
//! SDK-sized schema and one at OpenAI's limits, and prints each figure beside
//! #585's bound. `OUT_DIR/crosscheck.jsonl` carries every document and the
//! automaton's verdict for an independent validator
//! (`tools/grammar-check/validate.py`); `OUT_DIR/summary.json` the numbers.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use ignis_artifact::{FrontendSet, Reader, TokenBytes};
use ignis_grammar::json::Json;
use ignis_grammar::{random_document, Grammar, Matcher, TokenSet, Vocab, VocabEntry};

/// The 27B family's logits row (crates/core/src/compute.rs).
const WIDTH: usize = 248_320;

/// `{"type":"object","properties":{...},"required":[every key],"additionalProperties":false}`.
fn object(fields: &[(String, String)]) -> String {
    let props: Vec<String> = fields.iter().map(|(k, v)| format!(r#""{k}":{v}"#)).collect();
    let names: Vec<String> = fields.iter().map(|(k, _)| format!(r#""{k}""#)).collect();
    format!(r#"{{"type":"object","properties":{{{}}},"required":[{}],"additionalProperties":false}}"#, props.join(","), names.join(","))
}

/// An agent SDK's output model at its larger end: nine models under `$defs`
/// (enums, null unions, five formats, bounded numbers, `multipleOf`,
/// patterns, arrays of references, a reference cycle), 99 properties.
fn sdk_schema() -> String {
    let models = ["task", "person", "address", "invoice", "line", "event", "ticket", "note", "step"];
    let formats = ["date-time", "email", "uuid", "date", "hostname"];
    let field = |name: &str, i: usize| -> (String, String) {
        let schema = match i % 9 {
            0 => r#"{"type":"string","description":"free text"}"#.to_string(),
            1 => r#"{"type":["string","null"]}"#.to_string(),
            2 => r#"{"type":"string","enum":["low","medium","high","critical"]}"#.to_string(),
            3 => r#"{"type":"integer","minimum":0,"maximum":10000}"#.to_string(),
            4 => r#"{"type":"number","minimum":0,"multipleOf":0.01}"#.to_string(),
            5 => format!(r#"{{"type":"string","format":"{}"}}"#, formats[i % formats.len()]),
            6 => r#"{"type":"array","items":{"type":"string"},"maxItems":8}"#.to_string(),
            7 => r#"{"type":"boolean"}"#.to_string(),
            _ => r#"{"type":"string","pattern":"^[A-Z]{2,4}-[0-9]{1,6}$"}"#.to_string(),
        };
        (format!("{name}_{i}"), schema)
    };
    let defs: Vec<String> = models
        .iter()
        .enumerate()
        .map(|(m, name)| {
            let mut fields: Vec<(String, String)> = (0..10).map(|i| field(name, m + i)).collect();
            let next = models[(m + 1) % models.len()];
            fields.push((format!("{name}_next"), format!(r##"{{"anyOf":[{{"$ref":"#/$defs/{next}"}},{{"type":"null"}}]}}"##)));
            format!(r#""{name}":{}"#, object(&fields))
        })
        .collect();
    let root: Vec<(String, String)> =
        models.iter().map(|m| (format!("{m}s"), format!(r##"{{"type":"array","items":{{"$ref":"#/$defs/{m}"}}}}"##))).collect();
    let root = object(&root);
    format!(r#"{},"$defs":{{{}}}}}"#, &root[..root.len() - 1], defs.join(","))
}

/// A schema at OpenAI's limits: 5,000 properties over 50 objects, 1,000 enum
/// values.
fn limits_schema() -> String {
    let objects: Vec<(String, String)> = (0..50)
        .map(|o| {
            let fields: Vec<(String, String)> = (0..99)
                .map(|p| {
                    let schema = match p % 3 {
                        0 if p < 30 => format!(r#"{{"type":"string","enum":["a{o}x{p}","b{o}x{p}"]}}"#),
                        1 => r#"{"type":["integer","null"],"minimum":0}"#.to_string(),
                        _ => r#"{"type":"string"}"#.to_string(),
                    };
                    (format!("f{p}"), schema)
                })
                .collect();
            (format!("o{o}"), object(&fields))
        })
        .collect();
    object(&objects)
}

/// The first compile and the median of `n` more, in nanoseconds.
fn time_compile(n: usize, compile: impl Fn() -> Arc<Grammar>) -> (u64, u64) {
    let t = Instant::now();
    let _ = compile();
    let first = t.elapsed().as_nanos() as u64;
    let mut times: Vec<u64> = (0..n)
        .map(|_| {
            let t = Instant::now();
            let _ = compile();
            t.elapsed().as_nanos() as u64
        })
        .collect();
    times.sort_unstable();
    (first, times[n / 2])
}

fn compact(j: &Json) -> String {
    j.compact()
}

/// Microsecond percentiles of a sample of nanosecond timings.
fn pct(v: &mut [u64]) -> serde_json::Value {
    if v.is_empty() {
        return serde_json::json!(null);
    }
    v.sort_unstable();
    let at = |p: f64| v[((v.len() as f64 - 1.0) * p).round() as usize] as f64 / 1000.0;
    serde_json::json!({"n": v.len(), "p50_us": at(0.5), "p99_us": at(0.99), "max_us": *v.last().unwrap() as f64 / 1000.0,
        "mean_us": v.iter().sum::<u64>() as f64 / v.len() as f64 / 1000.0})
}

fn line(name: &str, v: &serde_json::Value) {
    if v.is_null() {
        println!("  {name:<40} -");
        return;
    }
    println!(
        "  {name:<40} n={:<7} p50={:>9.2}us p99={:>9.2}us max={:>9.2}us",
        v["n"],
        v["p50_us"].as_f64().unwrap(),
        v["p99_us"].as_f64().unwrap(),
        v["max_us"].as_f64().unwrap()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: grammar_bench ARTIFACT CORPUS_JSON OUT_DIR [REFERENCE_STATES_PER_DOC]");
        std::process::exit(2);
    }
    let ref_per_doc: usize = args.get(4).map_or(24, |v| v.parse().unwrap());
    let out_dir = Path::new(&args[3]);
    std::fs::create_dir_all(out_dir).unwrap();

    // --- The vocabulary --------------------------------------------------
    let t = Instant::now();
    let reader = Reader::open(Path::new(&args[1])).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let tok = frontend.tokenizer();
    let frontend_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let table = tok.token_bytes_table(WIDTH).unwrap_or_else(|e| panic!("token bytes: {e}"));
    let table_ms = t.elapsed().as_secs_f64() * 1e3;
    // The table against the tokenizer's own decode, on every token whose
    // bytes are a whole UTF-8 string.
    let (mut decoded_same, mut decoded_partial) = (0usize, 0usize);
    for (id, e) in table.iter().enumerate() {
        if let TokenBytes::Text(b) = e {
            match std::str::from_utf8(b) {
                Ok(s) => {
                    let d = tok.decode(&[id as u32]).unwrap();
                    assert_eq!(d, s, "token {id}: table {s:?} vs decode {d:?}");
                    decoded_same += 1;
                }
                Err(_) => decoded_partial += 1,
            }
        }
    }
    let stops: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .map(|s| {
            let ids = tok.encode(s).unwrap();
            assert_eq!(ids.len(), 1, "{s} is one token");
            ids[0]
        })
        .collect();
    let (mut n_text, mut n_added, mut n_special, mut n_absent) = (0, 0, 0, 0);
    let entries: Vec<VocabEntry> = table
        .into_iter()
        .map(|e| match e {
            TokenBytes::Text(b) => {
                n_text += 1;
                VocabEntry::Text(b)
            }
            TokenBytes::Added { special } => {
                n_added += 1;
                n_special += special as usize;
                VocabEntry::Control
            }
            TokenBytes::Absent => {
                n_absent += 1;
                VocabEntry::Absent
            }
        })
        .collect();
    let t = Instant::now();
    let vocab = Vocab::new(entries, &stops);
    let vocab_ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "vocabulary: {} columns, {n_text} text tokens ({decoded_same} equal to the tokenizer's decode, {decoded_partial} partial UTF-8), {n_added} added ({n_special} special), {n_absent} absent; stops {stops:?}",
        vocab.len()
    );
    println!(
        "  frontend {frontend_ms:.0} ms, byte table {table_ms:.0} ms, trie + {} free-string tables {vocab_ms:.0} ms; trie {} nodes; vocabulary memory {:.1} MiB",
        vocab.free_table_count(),
        vocab.trie_node_count(),
        vocab.heap_bytes() as f64 / (1 << 20) as f64
    );

    // --- The corpus --------------------------------------------------------
    let corpus = Json::parse(&std::fs::read_to_string(&args[2]).unwrap()).unwrap();
    let Json::Arr(cases) = corpus else { panic!("corpus is an array") };
    let mut jsonl = std::io::BufWriter::new(std::fs::File::create(out_dir.join("crosscheck.jsonl")).unwrap());
    let (mut all_check, mut all_first, mut all_cached, mut all_check_random) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut first_by_pos: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    let mut compile_rows = Vec::new();
    let (mut ref_states, mut ref_mismatch) = (0usize, 0usize);
    let mut mask_sizes: Vec<usize> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    let mut slow_checks: Vec<String> = Vec::new();
    let mut slow_masks: Vec<String> = Vec::new();
    let mut out = TokenSet::new(vocab.len());
    let mut reference = TokenSet::new(vocab.len());
    let mut total_docs = 0;

    for case in &cases {
        let name = case.get("name").and_then(Json::as_str).unwrap().to_string();
        let kind = case.get("kind").and_then(Json::as_str).unwrap().to_string();
        let schema_text = case.get("schema").map(compact);
        let compile = |v: &Arc<Vocab>| -> Arc<Grammar> {
            match &schema_text {
                None => Grammar::json_object(v.clone()),
                Some(s) => Grammar::json_schema(v.clone(), s, true).unwrap_or_else(|e| panic!("{name}: {e}")),
            }
        };
        // Compile time: the first compile, then the median of 20.
        let (first_compile, compile_median) = time_compile(20, || compile(&vocab));
        let g0 = compile(&vocab);

        let mut valid_texts = Vec::new();
        if let Some(Json::Arr(vs)) = case.get("valid") {
            for v in vs {
                valid_texts.push(compact(v));
                valid_texts.push(v.pretty());
            }
        }
        let mut invalid_texts = Vec::new();
        if let Some(Json::Arr(vs)) = case.get("invalid") {
            for v in vs {
                invalid_texts.push(compact(v));
            }
        }
        if let Some(Json::Arr(vs)) = case.get("invalid_text") {
            for v in vs {
                invalid_texts.push(v.as_str().unwrap().to_string());
            }
        }
        // Outside the generation subset by design: the first valid object
        // with its top-level keys reversed (a strict grammar writes keys in
        // schema order).
        let mut outside = Vec::new();
        if kind == "json_schema"
            && let Some(Json::Arr(vs)) = case.get("valid")
            && let Some(Json::Obj(e)) = vs.first()
        {
            let mut r = e.clone();
            r.reverse();
            if r != *e {
                outside.push(compact(&Json::Obj(r)));
            }
        }

        let replay = |g: &Arc<Grammar>, ids: &[u32]| -> bool {
            let mut m = Matcher::new(g.clone());
            ids.iter().chain(&stops[..1]).all(|&id| m.accept(id))
        };
        let row = |source: &str, text: &str, ours: bool| {
            serde_json::json!({"case": name, "kind": kind, "schema": schema_text, "source": source, "text": text,
                "ours": if ours { "valid" } else { "invalid" }})
        };

        let mut case_states = 0;
        for text in &valid_texts {
            total_docs += 1;
            let ids = tok.encode(text).unwrap();
            assert_eq!(tok.decode(&ids).unwrap(), *text, "{name}: tokenization round trip");
            // A fresh grammar: every state's first visit is a real first
            // visit (the vocabulary's free-string tables were built at load).
            let g = compile(&vocab);
            let mut m = Matcher::new(g.clone());
            let mut ok = true;
            let sample_every = (ids.len() / ref_per_doc.max(1)).max(1);
            for (i, &id) in ids.iter().chain(&stops[..1]).enumerate() {
                let pos = format!("{:?}", m.position());
                let t = Instant::now();
                let cached = m.mask(&mut out);
                let dt = t.elapsed().as_nanos() as u64;
                if cached {
                    all_cached.push(dt);
                } else {
                    all_first.push(dt);
                    if dt > 300_000 {
                        let prefix = String::from_utf8_lossy(&ids[..i].iter().flat_map(|&t| vocab.token_bytes(t).unwrap_or(&[]).to_vec()).collect::<Vec<u8>>()).into_owned();
                        let tail: String = prefix.chars().rev().take(40).collect::<Vec<_>>().into_iter().rev().collect();
                        slow_masks.push(format!("{name}: {:.0} us at {pos} after ...{tail:?}", dt as f64 / 1e3));
                    }
                    first_by_pos.entry(pos).or_default().push(dt);
                }
                let t = Instant::now();
                m.mask(&mut out);
                all_cached.push(t.elapsed().as_nanos() as u64);
                mask_sizes.push(out.count());
                if !out.contains(id) {
                    failures.push(format!("{name}: token {i} ({id}) of a valid document is not in its mask"));
                }
                if i % sample_every == 0 || i == ids.len() {
                    m.mask_reference(&mut reference);
                    ref_states += 1;
                    case_states += 1;
                    if out != reference {
                        ref_mismatch += 1;
                        failures.push(format!("{name}: mask != reference at token {i}"));
                    }
                }
                let t = Instant::now();
                let c = m.check(id);
                all_check.push(t.elapsed().as_nanos() as u64);
                if !c || !m.accept(id) {
                    ok = false;
                    failures.push(format!("{name}: valid document refused at token {i}: {text}"));
                    break;
                }
            }
            writeln!(jsonl, "{}", row("corpus_valid", text, ok)).unwrap();
        }
        let mut invalid_refused = 0;
        for text in &invalid_texts {
            total_docs += 1;
            let accepted = replay(&g0, &tok.encode(text).unwrap());
            if accepted {
                failures.push(format!("{name}: invalid document accepted: {text}"));
            } else {
                invalid_refused += 1;
            }
            writeln!(jsonl, "{}", row("corpus_invalid", text, accepted)).unwrap();
        }
        for text in &outside {
            let accepted = replay(&g0, &tok.encode(text).unwrap());
            writeln!(jsonl, "{}", row("outside_subset", text, accepted)).unwrap();
        }
        // Random documents of the language.
        let mut random_ok = 0;
        for seed in 0..40u64 {
            let doc = match random_document(&g0, seed + 1, 160, 20_000) {
                Ok(Some(d)) => d,
                Ok(None) => continue,
                Err(e) => {
                    failures.push(format!("{name}: {e}"));
                    continue;
                }
            };
            let text = String::from_utf8(doc).unwrap();
            let ids = tok.encode(&text).unwrap();
            let mut m = Matcher::new(g0.clone());
            let mut ok = true;
            for (i, &id) in ids.iter().chain(&stops[..1]).enumerate() {
                if i % 50 == 7 {
                    m.mask(&mut out);
                    m.mask_reference(&mut reference);
                    ref_states += 1;
                    case_states += 1;
                    if out != reference {
                        ref_mismatch += 1;
                        failures.push(format!("{name}: mask != reference on a random document at token {i}"));
                    }
                }
                let t = Instant::now();
                let c = m.check(id);
                let dt = t.elapsed().as_nanos() as u64;
                all_check_random.push(dt);
                if dt > 20_000 {
                    // Time it again: a cold cache line or a preemption does
                    // not repeat, the automaton's own cost does.
                    let t = Instant::now();
                    m.check(id);
                    let again = t.elapsed().as_nanos() as u64;
                    slow_checks.push(format!(
                        "{name}: check {:.1} us (again {:.1} us), token of {} bytes, {} configurations",
                        dt as f64 / 1e3,
                        again as f64 / 1e3,
                        vocab.token_bytes(id).map_or(0, |b| b.len()),
                        m.alternatives()
                    ));
                }
                if !c || !m.accept(id) {
                    ok = false;
                    failures.push(format!("{name}: random document refused at token {i}"));
                    break;
                }
            }
            random_ok += ok as usize;
            writeln!(jsonl, "{}", row("random", &text, ok)).unwrap();
        }
        println!(
            "{name:<18} {kind:<11} compile first {:>7.2} ms median {:>7.3} ms; nodes {:>3}, DFA states {:>4}; {} valid docs, {}/{} invalid refused, {} random docs, {} reference states; grammar memory {:.2} MiB",
            first_compile as f64 / 1e6,
            compile_median as f64 / 1e6,
            g0.node_count(),
            g0.dfa_states(),
            valid_texts.len(),
            invalid_refused,
            invalid_texts.len(),
            random_ok,
            case_states,
            g0.heap_bytes() as f64 / (1 << 20) as f64
        );
        compile_rows.push(serde_json::json!({"case": name, "first_ms": first_compile as f64 / 1e6, "median_ms": compile_median as f64 / 1e6,
            "nodes": g0.node_count(), "dfa_states": g0.dfa_states(), "memory_bytes": g0.heap_bytes(), "reference_states": case_states}));
    }
    jsonl.flush().unwrap();

    // Compiling schemas larger than the corpus's.
    let mut large = BTreeMap::new();
    for (name, schema) in [("sdk_sized", sdk_schema()), ("openai_limits", limits_schema())] {
        let g = Grammar::json_schema(vocab.clone(), &schema, true).unwrap_or_else(|e| panic!("{name}: {e}"));
        let (first, median) = time_compile(10, || Grammar::json_schema(vocab.clone(), &schema, true).unwrap());
        // A random document of it, held to the reference at a few states.
        let doc = random_document(&g, 1, 400, 200_000).unwrap().expect("a document");
        let text = String::from_utf8(doc).unwrap();
        writeln!(jsonl, "{}", serde_json::json!({"case": name, "kind": "json_schema", "schema": schema, "source": "random", "text": text, "ours": "valid"})).unwrap();
        let ids = tok.encode(&text).unwrap();
        let mut m = Matcher::new(g.clone());
        for (i, &id) in ids.iter().enumerate() {
            if i % (ids.len() / 8).max(1) == 0 {
                m.mask(&mut out);
                m.mask_reference(&mut reference);
                ref_states += 1;
                if out != reference {
                    ref_mismatch += 1;
                    failures.push(format!("{name}: mask != reference at token {i}"));
                }
            }
            assert!(m.accept(id), "{name}: its own random document refused at token {i}");
        }
        println!(
            "{name:<18} {} schema characters, {} nodes: compile first {:.2} ms, median {:.2} ms",
            schema.len(),
            g.node_count(),
            first as f64 / 1e6,
            median as f64 / 1e6
        );
        large.insert(name, serde_json::json!({"schema_chars": schema.len(), "nodes": g.node_count(), "first_ms": first as f64 / 1e6, "median_ms": median as f64 / 1e6}));
    }
    // Schemas compiled without `strict` and with no inert keyword: what the
    // grammar adds there (an untyped schema's implied types, an open object,
    // a required key `properties` does not list, a map, a root that is not
    // an object) must keep every document valid too.
    let loose = [
        r#"{"type":"object","properties":{"a":{"maximum":3,"multipleOf":0.5},"b":{"type":"string"}},"required":["a","c"]}"#,
        r#"{"type":"object","additionalProperties":{"type":"integer","minimum":0,"maximum":99}}"#,
        r#"{"type":"array","items":{"anyOf":[{"type":"string","format":"date"},{"type":"null"}]},"minItems":2,"maxItems":4}"#,
        r#"{"properties":{"tags":{"items":{"enum":["x","y",1.5]}},"when":{"format":"time"}}}"#,
    ];
    for (i, schema) in loose.iter().enumerate() {
        let name = format!("not_strict_{i}");
        let g = Grammar::json_schema(vocab.clone(), schema, false).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(g.inert().is_empty(), "{name}: {:?}", g.inert());
        let mut made = 0;
        for seed in 0..40u64 {
            let Some(doc) = random_document(&g, seed + 1, 120, 20_000).unwrap_or_else(|e| panic!("{name}: {e}")) else { continue };
            let text = String::from_utf8(doc).unwrap();
            writeln!(jsonl, "{}", serde_json::json!({"case": name, "kind": "json_schema", "schema": schema, "source": "random", "text": text, "ours": "valid"})).unwrap();
            made += 1;
        }
        println!("{name:<18} not strict: {made} random documents");
    }
    jsonl.flush().unwrap();

    mask_sizes.sort_unstable();
    let by_pos: BTreeMap<String, serde_json::Value> = first_by_pos.iter_mut().map(|(k, v)| (k.clone(), pct(v))).collect();
    let summary = serde_json::json!({
        "vocabulary": {"columns": vocab.len(), "text_tokens": n_text, "added": n_added, "special": n_special, "absent": n_absent,
            "trie_nodes": vocab.trie_node_count(), "free_string_tables": vocab.free_table_count(), "memory_bytes": vocab.heap_bytes(),
            "frontend_ms": frontend_ms, "byte_table_ms": table_ms, "trie_and_tables_ms": vocab_ms, "stops": stops},
        "documents": total_docs,
        "check_corpus": pct(&mut all_check),
        "check_random": pct(&mut all_check_random),
        "mask_first_visit": pct(&mut all_first),
        "mask_cached": pct(&mut all_cached),
        "mask_first_visit_by_position": by_pos,
        "mask_size": {"p50": mask_sizes[mask_sizes.len() / 2], "max": mask_sizes.last()},
        "compile": compile_rows,
        "compile_large": large,
        "reference_states": ref_states,
        "reference_mismatches": ref_mismatch,
        "failures": failures,
        "slow_checks": slow_checks,
        "slow_first_visit_masks": slow_masks,
    });
    std::fs::write(out_dir.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    println!("\ntimings over {total_docs} corpus documents:");
    line("check (corpus tokens)", &summary["check_corpus"]);
    line("check (random-document tokens)", &summary["check_random"]);
    line("mask, first visit", &summary["mask_first_visit"]);
    line("mask, cached", &summary["mask_cached"]);
    for (k, v) in summary["mask_first_visit_by_position"].as_object().unwrap() {
        line(&format!("mask, first visit, {k}"), v);
    }
    println!("mask size p50 {} max {}", summary["mask_size"]["p50"], summary["mask_size"]["max"]);
    println!("reference states {ref_states}, mismatches {ref_mismatch}");
    for s in summary["slow_checks"].as_array().unwrap() {
        println!("slow check: {s}");
    }
    for s in summary["slow_first_visit_masks"].as_array().unwrap() {
        println!("slow first-visit mask: {s}");
    }
    // #585's bounds.
    let p99 = |k: &str| summary[k]["p99_us"].as_f64().unwrap();
    let corpus_compile = summary["compile"].as_array().unwrap().iter().map(|r| r["first_ms"].as_f64().unwrap()).fold(0.0, f64::max);
    let sdk_compile = summary["compile_large"]["sdk_sized"]["first_ms"].as_f64().unwrap();
    println!("\nbounds (#585):");
    for (what, got, bound, unit) in [
        ("check, p99 (corpus and random tokens)", p99("check_corpus").max(p99("check_random")), 10.0, "us"),
        ("permitted set, first visit, p99", p99("mask_first_visit"), 1000.0, "us"),
        ("compile, an SDK-sized schema (first)", sdk_compile, 50.0, "ms"),
        ("compile, the corpus's slowest (first)", corpus_compile, 50.0, "ms"),
    ] {
        println!("  {what:<40} {got:>9.2} {unit} (bound {bound} {unit}): {}", if got <= bound { "met" } else { "MISSED" });
    }
    let failures = summary["failures"].as_array().unwrap();
    println!("failures: {}", failures.len());
    for f in failures.iter().take(20) {
        println!("  {f}");
    }
}
