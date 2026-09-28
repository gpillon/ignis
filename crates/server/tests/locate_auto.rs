//! `auto` and the **spaced JSON** a record is folded and shown as (spec 22,
//! GitHub #278), held to the Python that measured them: the golden cases
//! `tools/locate-sets/golden22.py` writes into `crates/core/tests/fixtures/`
//! (records arrays, content-parts-shaped arrays, arrays of strings and of
//! numbers, text; `json.dumps(ensure_ascii=False)`).

use ignis_core::locate::Kind;
use ignis_server::decide::OrderedValue;
use ignis_server::locate::{auto_kind, is_records_array, segment_texts, spaced_json};
use serde_json::Value;

fn fixture(name: &str) -> Value {
    let path = format!("{}/../core/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).expect("golden cases")).expect("golden JSON")
}

/// A golden state, read as the endpoint reads a `state`: in the order it
/// was written.
fn ordered(value: &Value) -> OrderedValue {
    serde_json::from_str(&value.to_string()).expect("an ordered value")
}

#[test]
fn auto_tells_every_golden_state_as_the_reference_does() {
    let golden = fixture("locate_auto.json");
    assert_eq!(golden["segments"], 2000);
    assert_eq!(golden["threshold"], 0.5);
    for case in golden["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let state = ordered(&case["state"]);
        let (kind, share) = auto_kind(&state);
        let want = match case["kind"].as_str().expect("kind") {
            "log" => Kind::Log,
            "prose" => Kind::Prose,
            "records" => Kind::Records,
            other => panic!("{other}"),
        };
        assert_eq!(kind, want, "{name}");
        match case["share"].as_f64() {
            Some(expected) => {
                let share = share.expect("a folded state has a share");
                assert!((share - expected).abs() < 1e-12, "{name}: {share} vs {expected}");
            }
            None => assert_eq!(share, None, "{name}"),
        }
    }
}

#[test]
fn a_records_array_is_told_by_its_shape() {
    let read = |text: &str| is_records_array(&serde_json::from_str::<OrderedValue>(text).unwrap());
    assert!(read(r#"[{"id":1},{"id":2}]"#));
    assert!(read(r#"[{"type":"a"},{"type":2}]"#), "a `type` that is not a string on every one is a record's field");
    assert!(!read(r#"[{"type":"text","text":"a"},{"type":"text","text":"b"}]"#), "content parts");
    assert!(!read(r#"[{"id":1}]"#), "one element");
    assert!(!read(r#"[{"id":1},"x"]"#), "mixed");
    assert!(!read(r#""a\nb""#), "text");
}

#[test]
fn spaced_json_is_pythons_json_dumps() {
    let golden = fixture("locate_renders.json");
    for case in golden["spaced_json"].as_array().expect("spaced_json") {
        let text = case["json"].as_str().expect("json");
        let value: OrderedValue = serde_json::from_str(text).expect("a value");
        assert_eq!(spaced_json(&value), case["spaced"].as_str().expect("spaced"), "{text}");
    }
    // A record array's last `choice` (`zd_records.py ask`): the candidates
    // in array order, each as its spaced JSON, labelled.
    let case = &golden["records"];
    let records: Vec<OrderedValue> = case["records"]
        .as_array()
        .expect("records")
        .iter()
        .map(|text| serde_json::from_str(text.as_str().expect("a record's JSON")).expect("a record"))
        .collect();
    let mut candidates: Vec<usize> =
        case["candidates"].as_array().expect("candidates").iter().map(|c| c.as_u64().expect("index") as usize).collect();
    candidates.sort_unstable();
    let lines: Vec<String> = candidates.iter().map(|&i| spaced_json(&records[i])).collect();
    let labels: Vec<String> = golden["labels"].as_array().expect("labels").iter().map(|l| l.as_str().unwrap().to_owned()).collect();
    assert_eq!(ignis_core::locate::render::labelled(&lines, &labels), case["text"].as_str().expect("text"));
    // Segments as text: a string's lines, an array's strings as they are and
    // everything else spaced.
    let array: OrderedValue = serde_json::from_str(r#"["a b",{"k":[1,2]},3]"#).unwrap();
    assert_eq!(segment_texts(&array), ["a b", r#"{"k": [1, 2]}"#, "3"]);
    let text: OrderedValue = serde_json::from_str(r#""one\n\ntwo""#).unwrap();
    assert_eq!(segment_texts(&text), ["one", "", "two"]);
}
