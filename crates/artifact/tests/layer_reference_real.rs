//! Explicit real-artifact acceptance for P1-20 (GitHub #56).
//!
//! It is ignored in the normal CPU gate because an f64 traversal of two real
//! layers is deliberately slow; run it in release with:
//! `cargo test -p ignis-artifact --release --test layer_reference_real -- --ignored`.
//! The only environmental skip is a missing locally cached artifact.

use std::path::Path;

use ignis_artifact::{
    f64_reference::{evaluate_layer, LayerFixture, LayerInput, HIDDEN},
    Reader,
};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

fn input(tokens: usize) -> LayerInput {
    let residual = (0..tokens * HIDDEN)
        .map(|i| ((i % HIDDEN) as f64 * 0.001 + (i / HIDDEN) as f64 * 0.125).sin())
        .collect();
    LayerInput::new(residual, (0..tokens as i32).collect()).expect("valid layer input")
}

#[test]
#[ignore = "real-artifact f64 layer oracle; run explicitly in release"]
fn gqa_and_gdn_references_cover_two_tokens() {
    let path = Path::new(ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return;
    }
    let reader = Reader::open(path).expect("open real artifact");
    let two_tokens = input(2);

    // Layer 3 exercises the BF16 GQA exception. Layer 4 is GDN's BF16
    // output-projection exception; its fused input remains NVFP4. The second
    // column is a causal/recurrent continuation rather than an independent
    // width-one calculation.
    let first_only = LayerInput::new(two_tokens.residual[..HIDDEN].to_vec(), vec![0])
        .expect("one-token prefix input");
    let second_only = LayerInput::new(two_tokens.residual[HIDDEN..].to_vec(), vec![1])
        .expect("one-token suffix input");
    let mut outputs = Vec::new();
    for layer in [3, 27, 4] {
        let output = evaluate_layer(&reader, layer, &two_tokens)
            .unwrap_or_else(|error| panic!("layer {layer} f64 reference: {error}"));
        assert_eq!(output.residual.len(), 2 * HIDDEN);
        assert!(output.residual.iter().all(|value| value.is_finite()));
        let isolated_first = evaluate_layer(&reader, layer, &first_only)
            .unwrap_or_else(|error| panic!("layer {layer} first-token reference: {error}"));
        assert_eq!(
            &output.residual[..HIDDEN],
            isolated_first.residual.as_slice(),
            "layer {layer}: token one must not observe a future token"
        );
        let isolated_second = evaluate_layer(&reader, layer, &second_only)
            .unwrap_or_else(|error| panic!("layer {layer} second-token reference: {error}"));
        assert!(
            output.residual[HIDDEN..]
                .iter()
                .zip(&isolated_second.residual)
                .any(|(continued, isolated)| continued != isolated),
            "layer {layer}: the second token must observe its prefix"
        );
        outputs.push(output);
    }

    // The checked-in values are the hand-off oracle for the device-side
    // tickets. Recompute them from the real artifact here so a host-decoder
    // or layer-formula regression cannot silently replace the reference with
    // merely finite, causal values.
    for (name, gqa_output) in [
        ("layer_reference_gqa3_gdn4.bin", &outputs[0]),
        ("layer_reference_gqa27_gdn4.bin", &outputs[1]),
    ] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join(name);
        let fixture = LayerFixture::read_from(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert_eq!(fixture.gqa_input, two_tokens, "{name}: recorded GQA input");
        assert_eq!(
            fixture.gqa_output, *gqa_output,
            "{name}: recorded GQA output"
        );
        assert_eq!(fixture.gdn_input, two_tokens, "{name}: recorded GDN input");
        assert_eq!(
            fixture.gdn_output, outputs[2],
            "{name}: recorded GDN output"
        );
    }

    if let Some(path) = std::env::var_os("IGNIS_RECORD_LAYER_FIXTURE") {
        let path = Path::new(&path);
        LayerFixture {
            gqa_layer: 3,
            gdn_layer: 4,
            gqa_input: two_tokens.clone(),
            gqa_output: outputs[0].clone(),
            gdn_input: two_tokens.clone(),
            gdn_output: outputs[2].clone(),
        }
        .write_to(path)
        .unwrap_or_else(|error| {
            panic!("write layer-reference fixture {}: {error}", path.display())
        });
        let nvfp4_path = path.with_file_name("layer_reference_gqa27_gdn4.bin");
        LayerFixture {
            gqa_layer: 27,
            gdn_layer: 4,
            gqa_input: two_tokens.clone(),
            gqa_output: outputs[1].clone(),
            gdn_input: two_tokens,
            gdn_output: outputs[2].clone(),
        }
        .write_to(&nvfp4_path)
        .unwrap_or_else(|error| {
            panic!(
                "write layer-reference fixture {}: {error}",
                nvfp4_path.display()
            )
        });
    }
}

#[test]
fn shipped_layer_fixtures_are_readable_by_the_public_oracle_api() {
    let fixtures = [
        ("layer_reference_gqa3_gdn4.bin", 3),
        ("layer_reference_gqa27_gdn4.bin", 27),
    ];
    for (name, gqa_layer) in fixtures {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join(name);
        let fixture = LayerFixture::read_from(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert_eq!(fixture.gqa_layer, gqa_layer);
        assert_eq!(fixture.gdn_layer, 4);
        assert_eq!(fixture.gqa_input.tokens(), 2);
        assert_eq!(fixture.gdn_input.tokens(), 2);
        assert_eq!(fixture.gqa_output.residual.len(), 2 * HIDDEN);
        assert_eq!(fixture.gdn_output.residual.len(), 2 * HIDDEN);
    }
}
