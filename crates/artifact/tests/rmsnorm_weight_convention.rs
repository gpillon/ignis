//! The artifact's RMSNorm weight convention, pinned against the real weights
//! (GitHub #67).
//!
//! Qwen 3.8-27B stores every hidden-width RMSNorm gain in *offset* form: the
//! tensor holds `w`, and the gain the forward pass must apply is `1 + w`
//! (`unit_offset=true`). Reading those tensors as a direct gain silently
//! multiplies the residual stream by a near-zero, mostly *negative* factor —
//! the model still produces finite activations and plausible-looking token
//! IDs, so nothing crashes and no NaN appears; the decoded text is simply
//! garbage (issue #67: whole-sequence decoding full of U+FFFD).
//!
//! That failure mode is invisible to a GPU-vs-oracle layer test when the
//! oracle shares the assumption, which is exactly what happened. This test
//! closes that gap at the only place the two cannot both be wrong: the stored
//! weights themselves. It is a data invariant, not an implementation echo.
//!
//! The GDN output norm and the GQA q/k norms are *not* covered by the same
//! rule: they are consumed by the vendored fused ops (`gated_rmsnorm`, which
//! applies no offset, and `qk_norm_rope`, which applies the offset itself),
//! so their conventions are fixed by the vendored contract rather than by an
//! ignis call-site flag. They are asserted here too, so a future change that
//! swaps one convention for the other is caught.

use std::path::Path;

use ignis_artifact::Reader;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

fn weights(reader: &Reader, name: &str) -> Vec<f64> {
    reader
        .payload(name)
        .unwrap_or_else(|error| panic!("{name}: {error}"))
        .data
        .chunks_exact(2)
        .map(|word| f32::from_bits((u16::from_le_bytes(word.try_into().unwrap()) as u32) << 16) as f64)
        .collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

#[test]
fn hidden_width_norms_are_stored_in_unit_offset_form() {
    let path = Path::new(ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return;
    }
    let reader = Reader::open(path).expect("open real artifact");

    // `post_attention_norm` is the sharpest witness: in the early layers every
    // one of the 5120 stored values is negative and the minimum sits just
    // above -1. Read as a direct gain that is a whole-stream sign flip; read
    // as `1 + w` it is a well-behaved attenuation in (0, 1].
    for layer in [0usize, 1, 3, 4, 27, 63] {
        for site in ["input_norm", "post_attention_norm"] {
            let name = format!("text/layers/{layer}/{site}");
            let w = weights(&reader, &name);
            assert_eq!(w.len(), 5120, "{name}: hidden width");

            assert!(
                w.iter().all(|v| *v > -1.0),
                "{name}: a stored value <= -1 would make the unit-offset gain \
                 non-positive, so the tensor is not in `1 + w` form"
            );
            assert!(
                mean(&w).abs() < 0.5,
                "{name}: mean {:+.4} is not centered near zero — the tensor \
                 does not look like an offset (`w`) gain",
                mean(&w)
            );
        }
    }

    // The specific value that makes `unit_offset=false` indefensible.
    let post0 = weights(&reader, "text/layers/0/post_attention_norm");
    let negatives = post0.iter().filter(|v| **v < 0.0).count();
    assert!(
        negatives * 100 >= post0.len() * 99,
        "text/layers/0/post_attention_norm: {negatives}/{} stored values are \
         negative; virtually all of them must be, because a direct-gain \
         reading would negate the residual stream",
        post0.len()
    );

    // The final norm shares the convention (the reference applies the offset
    // here too) even though its stored values sit near 1 rather than near 0.
    let final_norm = weights(&reader, "text/final_norm");
    assert_eq!(final_norm.len(), 5120);
    assert!(final_norm.iter().all(|v| *v > -1.0), "text/final_norm gain");
}

#[test]
fn fused_op_norms_keep_their_vendored_conventions() {
    let path = Path::new(ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return;
    }
    let reader = Reader::open(path).expect("open real artifact");

    // `gated_rmsnorm` applies no unit offset, so the stored GDN output-norm
    // gain must already be centered on 1.
    let gdn = weights(&reader, "text/layers/4/gdn/norm");
    assert_eq!(gdn.len(), 128);
    assert!(
        (mean(&gdn) - 1.0).abs() < 0.25,
        "text/layers/4/gdn/norm: mean {:+.4} must be near 1 — `gated_rmsnorm` \
         applies no unit offset",
        mean(&gdn)
    );

    // `qk_norm_rope` applies the offset itself, so these are stored as `w`.
    for site in ["query_norm", "key_norm"] {
        let name = format!("text/layers/3/attention/{site}");
        let w = weights(&reader, &name);
        assert_eq!(w.len(), 256, "{name}: head width");
        assert!(
            mean(&w).abs() < 0.5,
            "{name}: mean {:+.4} must be near 0 — `qk_norm_rope` adds the \
             unit offset itself",
            mean(&w)
        );
    }
}
