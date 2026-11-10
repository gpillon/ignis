//! Flash-Next's three-axis positions against the reference implementation
//! (GitHub #558, spec #555, ADR 0087): the positions `assign_positions` gives
//! a prompt with images, and the rotation a decode step takes after it, equal
//! bit for bit the checkpoint's own transformers code (`Qwen4ExpModel`'s
//! `get_rope_index` for a prompt, `compute_3d_position_ids` for a decode step;
//! its multimodal rope index is Qwen3.5's).
//!
//! The fixture `tests/fixtures/flash_next_mrope/positions.json` was recorded by
//! `kernel/tests/fixtures/flash_next_mrope/record_mrope.py`, beside the rotations
//! the kernel's op test (`kernel/tests/test_flash_next_mrope.cu`) checks over
//! the same prompts: a text-only prompt, one image, two images, an image across
//! a block boundary, and decode after an image. CPU-only.

use std::path::Path;

use ignis_artifact::vision::layout::{assign_positions, IMAGE, TEXT};
use ignis_artifact::vision::{Grid, TokenSpan, IMAGE_PAD_ID, MERGE};
use serde_json::Value;

fn fixture() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/flash_next_mrope/positions.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("positions.json parses")
}

fn ints(value: &Value) -> Vec<i64> {
    value.as_array().expect("an array").iter().map(|v| v.as_i64().expect("an integer")).collect()
}

/// The recorded `[3][n]` rows, axis-major as `assign_positions` lays them out.
fn axis_major(value: &Value) -> Vec<i32> {
    let axes = value.as_array().expect("three axes");
    assert_eq!(axes.len(), 3, "three axes");
    axes.iter().flat_map(|axis| ints(axis).into_iter().map(|p| p as i32)).collect()
}

#[test]
fn the_checkpoint_merges_and_marks_images_as_the_processor_does() {
    let fx = fixture();
    assert_eq!(fx["spatial_merge_size"].as_u64(), Some(MERGE as u64));
    assert_eq!(fx["image_token_id"].as_u64(), Some(u64::from(IMAGE_PAD_ID)));
}

#[test]
fn prompt_positions_and_rope_delta_are_the_references_bit_for_bit() {
    let fx = fixture();
    let cases = fx["cases"].as_array().expect("cases");
    let names: Vec<&str> = cases.iter().map(|c| c["name"].as_str().expect("a name")).collect();
    assert_eq!(names, ["text", "one_image", "two_images", "image_across_blocks", "decode_after_image"]);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let types: Vec<u8> = ints(&case["types"]).into_iter().map(|t| t as u8).collect();
        let grids: Vec<Grid> = case["grids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| {
                let thw = ints(g);
                Grid { t: thw[0] as u32, h: thw[1] as u32, w: thw[2] as u32 }
            })
            .collect();
        let (positions, spans, delta) = assign_positions(&types, &grids).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(positions, axis_major(&case["positions"]), "{name}: positions");
        assert_eq!(i64::from(delta), case["rope_delta"].as_i64().unwrap(), "{name}: rope delta");
        // Each image's placeholders are one span, where the token types put them.
        let mut want = Vec::new();
        let mut begin = 0;
        while begin < types.len() {
            let end = begin + types[begin..].iter().take_while(|&&t| t == types[begin]).count();
            if types[begin] == IMAGE {
                want.push(TokenSpan { begin, count: end - begin });
            } else {
                assert_eq!(types[begin], TEXT);
            }
            begin = end;
        }
        assert_eq!(spans, want, "{name}: spans");
    }
}

/// A decode step after the prompt rotates at its KV position plus the
/// prompt's rope delta on all three axes -- what the kernel adds
/// (`ignis_seq::rope_delta`, staged as the batch's rope deltas).
#[test]
fn decode_after_an_image_rotates_at_the_position_plus_the_rope_delta() {
    let fx = fixture();
    let mut decoded = 0;
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let delta = case["rope_delta"].as_i64().unwrap();
        let first = case["decode"]["first"].as_i64().unwrap();
        assert_eq!(first as usize, case["types"].as_array().unwrap().len(), "{name}: decode starts after the prompt");
        let axes = case["decode"]["positions"].as_array().unwrap();
        for axis in axes {
            for (step, &p) in ints(axis).iter().enumerate() {
                assert_eq!(p, first + step as i64 + delta, "{name}: decode step {step}");
                decoded += 1;
            }
        }
    }
    assert_eq!(decoded, 3 * 6, "the decode case's six steps on three axes");
    // The decode case's prompt ends on an image, and its delta is not 0: a
    // rule without the delta, or one that continued from an image axis,
    // would differ.
    let case = &fx["cases"][4];
    assert_ne!(case["rope_delta"].as_i64(), Some(0));
}
