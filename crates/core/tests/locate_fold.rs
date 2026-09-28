//! **`template_fold`** (spec 22, GitHub #278): the fold a `locate` over a log
//! reads first, held to the Python that measured it.
//!
//! Every case is written by `tools/locate-sets/golden22.py` from
//! `tools/locate-sets/compress.py` itself — labels, times in every shape,
//! masked variables, the SIM boundary, the 600-character budget, long
//! values, non-ASCII, repeats, affixes, whitespace Python and Rust disagree
//! on, carriage returns, a bracket-opened line, records as spaced JSON,
//! string elements. Synthetic inputs only.

use ignis_core::locate::fold::{Level2, fold};
use serde_json::Value;

fn fixture() -> Value {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/locate_fold.json"))
        .expect("the fold's golden cases");
    serde_json::from_str(&text).expect("golden JSON")
}

fn strings(value: &Value) -> Vec<String> {
    value.as_array().expect("an array").iter().map(|v| v.as_str().expect("a string").to_owned()).collect()
}

fn indices(value: &Value) -> Vec<usize> {
    value.as_array().expect("an array").iter().map(|v| v.as_u64().expect("an index") as usize).collect()
}

#[test]
fn the_fold_reproduces_every_golden_case() {
    let golden = fixture();
    assert_eq!(golden["sim"], 0.5, "the settings R2 was judged with");
    let cases = golden["cases"].as_array().expect("cases");
    assert!(cases.len() >= 12, "{} cases", cases.len());
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let lines = strings(&case["lines"]);
        let folded = fold(&lines);
        let clusters = case["clusters"].as_array().expect("clusters");
        assert_eq!(folded.clusters.len(), clusters.len(), "{name}: clusters");
        for (got, want) in folded.clusters.iter().zip(clusters) {
            assert_eq!(got.label, want["label"].as_str().expect("label"), "{name}");
            assert_eq!(got.template, strings(&want["template"]), "{name}");
            assert_eq!(got.members, indices(&want["members"]), "{name}");
        }
        assert_eq!(folded.level1, strings(&case["level1"]), "{name}: level 1");
        for (index, want) in case["level2"].as_array().expect("level2").iter().enumerate() {
            let Level2 { texts, members } = folded.level2(&lines, index);
            assert_eq!(texts, strings(&want["texts"]), "{name}: level 2 of cluster {index}");
            let want_members: Vec<Vec<usize>> =
                want["members"].as_array().expect("members").iter().map(indices).collect();
            assert_eq!(members, want_members, "{name}: level 2 members of cluster {index}");
        }
    }
}

/// A folded line maps back to the lines it stands for, every folded line
/// exactly once — the map a `locate` answers through.
#[test]
fn every_line_is_in_exactly_one_template_and_one_row() {
    let lines: Vec<String> = (0..300).map(|i| format!("2026-09-28T10:00:{:02}Z pod-{} ready in {}ms", i % 60, i % 7, i)).collect();
    let folded = fold(&lines);
    let mut seen = vec![0u32; lines.len()];
    for index in 0..folded.clusters.len() {
        for row in folded.level2(&lines, index).members {
            for line in row {
                seen[line] += 1;
            }
        }
    }
    assert!(seen.iter().all(|&n| n == 1), "{seen:?}");
    assert!(folded.templated_share() >= 0.5);
}
