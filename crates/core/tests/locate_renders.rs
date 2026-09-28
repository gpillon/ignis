//! What a `locate`'s labelled `choice` is shown, and the `found` rules (spec
//! 22, GitHub #278), held to the Python that measured them
//! (`tools/locate-sets/golden22.py`: `zd_prose.render`, `zd_notfound.ask`'s
//! labelled lines, spec 22 § Not found).

use ignis_core::locate::fold::fold;
use ignis_core::locate::render::{in_paragraphs, labelled, rows_first_lines};
use ignis_core::locate::{FOUND_THRESHOLD, found_by_none, found_log};
use serde_json::Value;

fn fixture() -> Value {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/locate_renders.json"))
        .expect("the renders' golden cases");
    serde_json::from_str(&text).expect("golden JSON")
}

fn indices(value: &Value) -> Vec<usize> {
    value.as_array().expect("an array").iter().map(|v| v.as_u64().expect("an index") as usize).collect()
}

fn strings(value: &Value) -> Vec<String> {
    value.as_array().expect("an array").iter().map(|v| v.as_str().expect("a string").to_owned()).collect()
}

#[test]
fn every_render_reproduces_the_reference() {
    let golden = fixture();
    let labels = strings(&golden["labels"]);
    for case in golden["renders"].as_array().expect("renders") {
        let name = case["name"].as_str().expect("name");
        let lines = strings(&case["lines"]);
        match case["kind"].as_str().expect("kind") {
            "prose" => {
                let candidates: Vec<usize> =
                    case["candidates"].as_array().expect("candidates").iter().map(|c| c.as_u64().expect("index") as usize).collect();
                let (text, named) = in_paragraphs(&lines, &candidates, &labels);
                assert_eq!(text, case["text"].as_str().expect("text"), "{name}");
                let want = case["labels"].as_object().expect("labels");
                assert_eq!(named.len(), want.len(), "{name}");
                for (label, line) in labels.iter().zip(&named) {
                    assert_eq!(want[label].as_u64(), Some(*line as u64), "{name}: {label}");
                }
            }
            "labelled" => assert_eq!(labelled(&lines, &labels), case["text"].as_str().expect("text"), "{name}"),
            // A fold's level 1: the kept templates' level-1 lines.
            "level1" => {
                let folded = fold(&lines);
                let kept: Vec<String> = indices(&case["kept"]).iter().map(|&t| folded.level1[t].clone()).collect();
                assert_eq!(labelled(&kept, &labels), case["text"].as_str().expect("text"), "{name}");
            }
            // A fold's last `choice`: the kept rows' first original lines.
            "raw_final" => {
                let folded = fold(&lines);
                let cluster = case["cluster"].as_u64().expect("cluster") as usize;
                let members = folded.level2(&lines, cluster).members;
                let raw = rows_first_lines(&members, &indices(&case["rows"]));
                assert_eq!(raw, indices(&case["raw"]), "{name}");
                let shown: Vec<&String> = raw.iter().map(|&i| &lines[i]).collect();
                assert_eq!(labelled(&shown, &labels), case["text"].as_str().expect("text"), "{name}");
            }
            other => panic!("{other}"),
        }
    }
}

#[test]
fn found_follows_spec_22s_rules() {
    let golden = fixture();
    for case in golden["found"].as_array().expect("found") {
        let p_none = case["p_none"].as_f64().expect("p_none");
        let found = match case["route"].as_str().expect("route") {
            "log" => found_log(p_none, case["p_yes"].as_f64().expect("p_yes")),
            _ => found_by_none(p_none),
        };
        let want = case["found"].as_f64().expect("found");
        assert!((found - want).abs() < 1e-12, "{case}");
        assert_eq!(found >= FOUND_THRESHOLD, case["found_is"].as_bool().expect("found_is"), "{case}");
    }
}
