//! The validator (GitHub #608, spec #582) on a corpus of tool arguments,
//! valid and invalid, each invalid one labelled by hand with the kinds of
//! violation it has: the verdicts and the kinds are held to those labels, and
//! the verdicts to an independent JSON Schema validator, `jsonschema` (Draft
//! 2020-12, formats checked, numbers as exact decimals) through
//! `tools/grammar-check/validate.py`.
//!
//! The independent check needs a Python with
//! `tools/grammar-check/requirements.txt`: `IGNIS_GRAMMAR_CHECK_PYTHON`, else
//! the venv `make grammar-venv` makes under `.scratch/grammar-venv`. Without
//! one it is skipped with a line saying so, and fails on CI (`CI` set), as
//! the served structured-output tests do.

use std::path::{Path, PathBuf};

use ignis_grammar::json::{write_string, Json};
use ignis_grammar::validate::violations;

/// The repository's root.
fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// One argument object of the corpus, and the kinds it violates (none for a
/// valid one).
struct Row {
    case: String,
    schema: Json,
    arguments: Json,
    kinds: Vec<String>,
}

fn corpus() -> Vec<Row> {
    let text = std::fs::read_to_string(repo().join("crates/grammar/tests/corpus/tool_arguments.json")).unwrap();
    let Json::Arr(cases) = Json::parse(&text).unwrap() else { panic!("the corpus is an array of cases") };
    let mut rows = Vec::new();
    for case in &cases {
        let name = case.get("case").and_then(Json::as_str).unwrap().to_owned();
        let schema = case.get("schema").unwrap().clone();
        let Some(Json::Arr(valid)) = case.get("valid") else { panic!("{name}: valid is a list") };
        for arguments in valid {
            rows.push(Row { case: name.clone(), schema: schema.clone(), arguments: arguments.clone(), kinds: Vec::new() });
        }
        let Some(Json::Arr(invalid)) = case.get("invalid") else { panic!("{name}: invalid is a list") };
        for entry in invalid {
            let Some(Json::Arr(kinds)) = entry.get("kinds") else { panic!("{name}: an invalid entry names its kinds") };
            let kinds: Vec<String> = kinds.iter().map(|k| k.as_str().unwrap().to_owned()).collect();
            assert!(!kinds.is_empty(), "{name}: an invalid entry names at least one kind");
            rows.push(Row { case: name.clone(), schema: schema.clone(), arguments: entry.get("arguments").unwrap().clone(), kinds });
        }
    }
    rows
}

/// `value` as JSON text with every number as the corpus wrote it (the
/// canonical form would write `1.0` as `1`, which is another argument).
fn text(value: &Json) -> String {
    fn go(value: &Json, out: &mut Vec<u8>) {
        match value {
            Json::Null => out.extend_from_slice(b"null"),
            Json::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
            Json::Num(n) => out.extend_from_slice(n.as_bytes()),
            Json::Str(s) => write_string(s, out),
            Json::Arr(items) => {
                out.push(b'[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    go(item, out);
                }
                out.push(b']');
            }
            Json::Obj(entries) => {
                out.push(b'{');
                for (i, (key, item)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    write_string(key, out);
                    out.push(b':');
                    go(item, out);
                }
                out.push(b'}');
            }
        }
    }
    let mut out = Vec::new();
    go(value, &mut out);
    String::from_utf8(out).unwrap()
}

#[test]
fn every_verdict_and_kind_is_the_one_labelled_by_hand() {
    let rows = corpus();
    let mut wrong = Vec::new();
    for row in &rows {
        let found = violations(&row.schema, &row.arguments);
        let mut kinds: Vec<&str> = found.iter().map(|v| v.kind.label()).collect();
        kinds.sort_unstable();
        kinds.dedup();
        let mut expected: Vec<&str> = row.kinds.iter().map(String::as_str).collect();
        expected.sort_unstable();
        if kinds != expected {
            wrong.push(format!("{}: {} -> {kinds:?} ({found:?}), labelled {expected:?}", row.case, text(&row.arguments)));
        }
    }
    assert!(wrong.is_empty(), "{} of {} rows disagree with their labels:\n{}", wrong.len(), rows.len(), wrong.join("\n"));
    let invalid = rows.iter().filter(|r| !r.kinds.is_empty()).count();
    assert!(rows.len() - invalid >= 25 && invalid >= 80, "a corpus of both: {} valid, {invalid} invalid", rows.len() - invalid);
}

/// The Python that runs `tools/grammar-check/validate.py` with its pinned
/// `jsonschema`: `IGNIS_GRAMMAR_CHECK_PYTHON` (what CI sets), else the venv
/// `make grammar-venv` makes under `.scratch/grammar-venv`.
fn validator_python() -> Option<PathBuf> {
    if let Some(python) = std::env::var_os("IGNIS_GRAMMAR_CHECK_PYTHON") {
        return Some(PathBuf::from(python));
    }
    let venv = repo().join(".scratch/grammar-venv");
    [venv.join("Scripts/python.exe"), venv.join("bin/python")].into_iter().find(|python| python.exists())
}

#[test]
fn every_verdict_agrees_with_an_independent_validator() {
    let Some(python) = validator_python() else {
        let missing = "no Python with tools/grammar-check/requirements.txt (IGNIS_GRAMMAR_CHECK_PYTHON, or `make grammar-venv`)";
        assert!(std::env::var_os("CI").is_none(), "{missing}: on CI the independent validator must run");
        eprintln!("skip: {missing}");
        return;
    };
    let rows = corpus();
    let mut lines = String::new();
    let (mut valid, mut invalid) = (0, 0);
    for row in &rows {
        let ours = if violations(&row.schema, &row.arguments).is_empty() { "valid" } else { "invalid" };
        match ours {
            "valid" => valid += 1,
            _ => invalid += 1,
        }
        let line = format!(
            "{{\"case\":{},\"source\":\"tool_arguments\",\"ours\":\"{ours}\",\"schema\":{},\"text\":{}}}\n",
            text(&Json::Str(row.case.clone())),
            text(&Json::Str(text(&row.schema))),
            text(&Json::Str(text(&row.arguments))),
        );
        lines.push_str(&line);
    }
    let file = std::env::temp_dir().join(format!("ignis-608-tool-arguments-{}.jsonl", std::process::id()));
    std::fs::write(&file, lines).unwrap();
    let output = std::process::Command::new(&python)
        .arg("-I")
        .arg(repo().join("tools/grammar-check/validate.py"))
        .arg(&file)
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", python.display()));
    let _ = std::fs::remove_file(&file);
    let report = String::from_utf8_lossy(&output.stdout);
    eprintln!("{report}");
    assert!(output.status.success(), "the independent validator disagrees:\n{report}\n{}", String::from_utf8_lossy(&output.stderr));
    for both in [format!("ours valid, theirs valid: {valid}"), format!("ours invalid, theirs invalid: {invalid}")] {
        assert!(report.contains(&both), "every row was validated ({both}): {report}");
    }
}
