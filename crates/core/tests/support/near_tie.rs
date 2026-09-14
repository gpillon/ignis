//! Spec-on against spec-off, up to a near-tie (P5-04, GitHub #153; shared
//! with P5-05, GitHub #155). Included by path from the GPU tests that hold
//! greedy speculation to the spec-off text.
//!
//! The engine's argmax is not route-invariant at a near-tie: the verify
//! traversal's k+1 columns per lane take different kernel tilings than a
//! one-column decode, and the engine's own two prefill routes already
//! disagree about the winner at such positions (measured and recorded in
//! `speculative_round_gpu.rs`'s header). So a divergence passes only where
//! those two routes call the two candidates a near-tie, or where both side
//! with spec-on's pick -- the spec-off stream is a route too, and a batched
//! one-column decode can drift from both prefill routes (#155 measured it at
//! width 4: 0.75 logits past them). The one divergence refused is the one both
//! routes lay at spec-on's door. The tolerance is never a constant.

use ignis_core::compute::ModelConfig;
use ignis_core::model_load::Model;
use ignis_core::seq::SeqPool;
use ignis_core::step::{prefill_program_with_route, PrefillRoute};

pub fn first_divergence(a: &[i32], b: &[i32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then(|| a.len().min(b.len())))
}

/// `a` (the spec-off pick) against `b` (the spec-on pick) after `prompt` +
/// `agreed`, by the engine's own two prefill routes, each run on a sequence of
/// `context` tokens from a fresh `probe_pool()` (a pool of the model's own
/// speculative backend): a near-tie when the routes disagree about which wins,
/// or rate the gap no wider than the spread between them; spec-off's own drift
/// when both prefer `b`. Panics naming both gaps otherwise -- both routes prefer
/// `a` beyond their spread.
pub fn assert_near_tie(
    model: &Model,
    probe_pool: &dyn Fn() -> SeqPool,
    context: u32,
    prompt: &[i32],
    agreed: &[i32],
    a: i32,
    b: i32,
    what: &str,
) {
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let mut tokens = prompt.to_vec();
    tokens.extend_from_slice(agreed);
    let gap = |route: PrefillRoute| {
        let pool = probe_pool();
        let mut seq = pool.alloc(context).unwrap_or_else(|e| panic!("alloc: {e}"));
        let mut logits = vec![0f32; vocab];
        prefill_program_with_route(model, &pool, &mut seq, &tokens, 0, route, Some(&mut logits))
            .unwrap_or_else(|e| panic!("near-tie probe prefill: {e}"));
        logits[a as usize] - logits[b as usize]
    };
    let chunked = gap(PrefillRoute::Chunked);
    let per_token = gap(PrefillRoute::PerToken);
    let routes_disagree = (chunked > 0.0) != (per_token > 0.0) || chunked == 0.0 || per_token == 0.0;
    let within_spread = chunked.abs().min(per_token.abs()) <= (chunked - per_token).abs();
    let routes_side_with_spec_on = chunked < 0.0 && per_token < 0.0;
    assert!(
        routes_disagree || within_spread || routes_side_with_spec_on,
        "{what}: spec-on picked {b} where spec-off picked {a} after {} tokens, it is not a near-tie, and both \
         prefill routes side with spec-off: logit[{a}] - logit[{b}] = {chunked} (chunked route), {per_token} \
         (per-token route)",
        agreed.len()
    );
}

/// Spec-on `on` against spec-off `off`, lane by lane: identical, or
/// identical up to a near-tie divergence ([`assert_near_tie`]). Returns each
/// lane's agreeing prefix length.
pub fn assert_equivalent(
    model: &Model,
    probe_pool: &dyn Fn() -> SeqPool,
    context: u32,
    prompts: &[Vec<i32>],
    off: &[Vec<i32>],
    on: &[Vec<i32>],
    what: &str,
) -> Vec<usize> {
    assert_eq!(off.len(), on.len(), "{what}: lane count");
    prompts
        .iter()
        .zip(off.iter().zip(on))
        .enumerate()
        .map(|(lane, (prompt, (off, on)))| match first_divergence(off, on) {
            None => off.len(),
            Some(d) => {
                assert!(d < off.len() && d < on.len(), "{what} lane {lane}: one stream ended early at {d}");
                assert_near_tie(model, probe_pool, context, prompt, &off[..d], off[d], on[d], &format!("{what} lane {lane}"));
                d
            }
        })
        .collect()
}
