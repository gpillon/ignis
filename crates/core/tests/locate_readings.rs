//! The heads' **readings** of a long text, its **windows**, their **merge**
//! and the **shortlist** (spec 22, GitHub #278), held to the Python that
//! measured them.
//!
//! Every case is written by `tools/locate-sets/golden22.py` from the
//! reference functions themselves (`zd_cache.key_features` in f64,
//! `zd_offline.zsum`, `zd_windows.sub_windows`, `zd_records.cut`,
//! `zd_prose.py rank`, `zd_records.py rank-windows`). Synthetic rows only.

use std::ops::Range;

use ignis_core::locate::reading::{Merge, Reading, cut_windows, merge, shortlist, window_scores};
use serde_json::Value;

fn fixture() -> Value {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/locate_readings.json"))
        .expect("the readings' golden cases");
    serde_json::from_str(&text).expect("golden JSON")
}

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().expect("an array").iter().map(|v| v.as_f64().expect("a number")).collect()
}

fn keys(value: &Value) -> Vec<Option<Range<usize>>> {
    value
        .as_array()
        .expect("keys")
        .iter()
        .map(|k| {
            k.as_array().map(|pair| pair[0].as_u64().expect("start") as usize..pair[1].as_u64().expect("end") as usize)
        })
        .collect()
}

/// Close enough to be the same arithmetic: numpy sums pairwise where the
/// port sums in order, which moves the last bits and nothing else.
fn close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-9 * w.abs().max(1.0), "{what}[{i}]: {g} vs {w}");
    }
}

#[test]
fn each_windows_reading_reproduces_the_reference() {
    let golden = fixture();
    let cases = golden["readings"].as_array().expect("readings");
    assert!(cases.len() >= 6);
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let heads = case["heads"].as_u64().expect("heads") as usize;
        let span = case["span"].as_u64().expect("span") as usize;
        // The rows are f32 on the wire, as the leaf reads them.
        let q: Vec<f32> = floats(&case["q"]).into_iter().map(|x| x as f32).collect();
        let na: Vec<f32> = floats(&case["na"]).into_iter().map(|x| x as f32).collect();
        assert_eq!(q.len(), heads * span, "{name}");
        let keys = keys(&case["keys"]);
        for (reading, field) in [(Reading::End, "end"), (Reading::Sum, "sum")] {
            let got = window_scores(&q, &na, heads, &keys, reading).expect("whole rows");
            close(&got, &floats(&case[field]), &format!("{name} ({field})"));
        }
    }
}

#[test]
fn windows_are_cut_as_the_references_cut_them() {
    let golden = fixture();
    for case in golden["windows"].as_array().expect("windows") {
        let name = case["name"].as_str().expect("name");
        let costs: Vec<u64> = case["costs"].as_array().expect("costs").iter().map(|c| c.as_u64().expect("cost")).collect();
        let empty: Vec<bool> = case["empty"].as_array().expect("empty").iter().map(|e| e.as_bool().expect("bool")).collect();
        let budget = case["budget"].as_u64().expect("budget");
        let want: Vec<Range<usize>> = case["windows"]
            .as_array()
            .expect("windows")
            .iter()
            .map(|w| w[0].as_u64().expect("first") as usize..w[1].as_u64().expect("end") as usize)
            .collect();
        assert_eq!(cut_windows(&costs, &empty, budget), want, "{name}");
    }
}

#[test]
fn windows_merge_and_shortlist_as_the_references_do() {
    let golden = fixture();
    for case in golden["merges"].as_array().expect("merges") {
        let name = case["name"].as_str().expect("name");
        let rule = match case["rule"].as_str().expect("rule") {
            "prose" => Merge::Prose,
            "records" => Merge::Records,
            other => panic!("{other}"),
        };
        let segments = case["segments"].as_u64().expect("segments") as usize;
        let windows: Vec<(usize, Vec<f64>)> = case["windows"]
            .as_array()
            .expect("windows")
            .iter()
            .map(|w| (w[0].as_u64().expect("first") as usize, floats(&w[1])))
            .collect();
        let merged = merge(rule, segments, &windows);
        close(&merged, &floats(&case["merged"]), name);
        let candidate: Vec<bool> =
            case["candidate"].as_array().expect("candidate").iter().map(|c| c.as_bool().expect("bool")).collect();
        let k = case["k"].as_u64().expect("k") as usize;
        let want: Vec<usize> =
            case["shortlist"].as_array().expect("shortlist").iter().map(|s| s.as_u64().expect("index") as usize).collect();
        assert_eq!(shortlist(&merged, &candidate, k), want, "{name}");
    }
}

/// Rows the leaf did not bring back whole are no reading.
#[test]
fn rows_that_are_not_whole_read_nothing() {
    let keys = vec![Some(0..2), Some(3..4)];
    let rows = vec![0.0f32; 8];
    assert!(window_scores(&rows, &rows, 2, &keys, Reading::End).is_some());
    assert!(window_scores(&rows, &rows[..6], 2, &keys, Reading::End).is_none(), "the baseline disagrees");
    assert!(window_scores(&rows, &rows, 0, &keys, Reading::End).is_none(), "no heads");
    assert!(window_scores(&rows, &rows, 3, &keys, Reading::Sum).is_none(), "not heads by one span");
    assert!(window_scores(&rows, &rows, 2, &[Some(0..5)], Reading::Sum).is_none(), "past the span");
    assert!(window_scores(&rows, &rows, 2, &[None, None], Reading::Sum).is_none(), "nothing owned");
    let mut bad = rows.clone();
    bad[1] = f32::NAN;
    assert!(window_scores(&bad, &rows, 2, &keys, Reading::Sum).is_none(), "not finite");
}
