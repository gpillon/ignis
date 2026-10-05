//! A Flash-Next artifact served (spec flash-next/04, GitHub #302): the
//! scheduler the server builds for it (`runtime::flash_next_scheduler`, the
//! leaf `ignis_runtime::FlashNextLeaf`) generates for three requests at once
//! and finishes each -- prefill chunks, decode rounds over three lanes and
//! the n-gram rows staged on the host for every step -- with prompt reuse
//! and KV-RAM off, as the load serves them.
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`). Explicit GPU profile (ADR 0006): a missing
//! artifact or GPU is a skip outside it, a failure under it. Pins the ~38 GB
//! expert pool.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::types::{DecodeParams, FinishReason, RequestClass, RequestInput, SchedEvent};
use ignis_core::{gpu_profile, Scheduler};
use ignis_server::runtime::{EngineShape, flash_next_scheduler};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MODEL: &str = "qwen3.8-flash-next";
/// Flash-Next's EOS (generation_config.json).
const EOS: u32 = 248_044;

fn input(prompt: Vec<u32>) -> RequestInput {
    RequestInput {
        decision: None,
        model: MODEL.into(),
        tokens: prompt,
        params: DecodeParams {
            max_tokens: Some(16),
            ..DecodeParams::default()
        },
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        constrained: None,
        warm_up: false,
    }
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn three_requests_generate_and_finish_on_three_lanes() {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from);
    let path = dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display())) {
        return;
    }
    let shape = EngineShape { max_context: 8192, prefill_chunk: 2048, ..EngineShape::default() };
    let (mut sched, reserved) = match flash_next_scheduler(&path, MODEL.into(), EOS, shape, None) {
        Ok(loaded) => loaded,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next scheduler: {e}"));
            return;
        }
    };
    println!("plan: {:?}", reserved.lines);
    let requests: Vec<_> = (0..3u32)
        .map(|r| {
            let prompt: Vec<u32> = (0..200 + 50 * r).map(|i| 1000 + (i * 7919 + r * 104_729) % 60_000).collect();
            sched.submit(input(prompt), RequestClass::Interactive).expect("submit")
        })
        .collect();
    let mut tokens = vec![0usize; requests.len()];
    let mut finished = vec![None; requests.len()];
    while !sched.is_idle() {
        for event in sched.advance() {
            match event {
                SchedEvent::Token { request, .. } => {
                    if let Some(i) = requests.iter().position(|&r| r == request) {
                        tokens[i] += 1;
                    }
                }
                SchedEvent::Done { request, reason, .. } => {
                    if let Some(i) = requests.iter().position(|&r| r == request) {
                        finished[i] = Some(reason);
                    }
                }
                _ => {}
            }
        }
    }
    for (i, reason) in finished.iter().enumerate() {
        println!("request {i}: {} tokens, {reason:?}", tokens[i]);
        assert!(tokens[i] > 0, "request {i} generated nothing");
        assert!(
            matches!(reason, Some(FinishReason::Length) | Some(FinishReason::Stop)),
            "request {i} finished as {reason:?}"
        );
    }
}
