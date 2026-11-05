//! GPU coverage for the 27B's decode lanes as a load option (GitHub #336,
//! spec 27b-16gb/04 solution 1): a load of two lanes with the DFlash2
//! drafter captures its decode and verify graphs at widths 1 and 2 and no
//! wider, serves a round of two, and refuses a round of three -- verify and
//! decode alike -- by name, over a pool that holds three sequences: its
//! round buffers are two lanes wide. A load past the leaf's eight is refused
//! before anything is reserved.
//!
//! BF16 KV, as `dflash2_round_gpu.rs` runs the drafter. The prompts are the
//! canary suite's (`crates/bench/src/canary.rs::CANARIES`).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{bind_model_scope_27b, materialize, CudaDevice, DraftModule, FrontendSet, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b_with_decode_lanes;
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    capture_decode_graphs, decode_program_batch_sampled, decode_program_verify_runs, prefill_program_sampled,
    program_stats, SamplingParams, VerifyLane,
};
use ignis_core::{KvFormat, PrefillAttention, PrefillLinear, RopeScaling, Speculation, SpeculativeBackend};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 128;
const WINDOW: u32 = 7;
const LANES: u32 = 2;

const CANARY_PROMPTS: [&str; 3] = [
    "In one sentence, what does `fn main() { println!(\"hi\"); }` do?",
    "What does `let v = vec![3,1,2]; v.sort();` set `v` to, after the call?",
    "Explain in one sentence what `x.reverse()` does to a `Vec<i32>` named `x`.",
];

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_two_lane_load_serves_two_lanes_and_refuses_a_third() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let prompts: Vec<Vec<i32>> = CANARY_PROMPTS
        .iter()
        .map(|prompt| {
            frontend
                .tokenizer()
                .encode(&format!(
                    "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                ))
                .unwrap_or_else(|e| panic!("tokenize: {e}"))
                .into_iter()
                .map(|t| t as i32)
                .collect()
        })
        .collect();
    let (plan, handles) =
        bind_model_scope_27b(&reader, Some(DraftModule::Dflash2)).unwrap_or_else(|e| panic!("bind: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!();
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize text + dflash2: {e}")) {
                return;
            }
            unreachable!();
        }
    };
    let spec = Speculation::new(SpeculativeBackend::Dflash2, WINDOW).unwrap();
    let load = |lanes| {
        load_qwen38_27b_with_decode_lanes(
            &reader,
            &artifact,
            &handles,
            MAX_CONTEXT,
            MAX_CONTEXT,
            KvFormat::Bf16,
            Some(spec),
            None,
            RopeScaling::NONE,
            PrefillAttention::default(),
            PrefillLinear::default(),
            lanes,
        )
    };
    let err = load(9).err().expect("nine lanes are past the leaf's eight");
    assert!(err.contains("decode_lanes 9"), "{err}");
    let model = load(LANES).unwrap_or_else(|e| panic!("model load at {LANES} lanes: {e}"));

    // A pool of three: the third sequence is real, so only the width refuses
    // the round.
    let slots = LANES + 1;
    let pool = SeqPool::create_with_speculation(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: KvFormat::Bf16,
            kv_page_group_count: MAX_CONTEXT.div_ceil(64) * slots,
            max_context_tokens: MAX_CONTEXT,
            slot_count: slots,
            retained_slot_count: 0,
            retained_host_slot_count: 0,
        },
        Some(SpeculativeBackend::Dflash2),
    )
    .unwrap_or_else(|e| panic!("seq pool create: {e}"));
    let capture = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
    let widths = (1u32 << LANES) - 1;
    assert_eq!(capture.ready_mask, widths, "decode graphs at widths 1..{LANES} and no wider");
    let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
    assert_eq!(
        (stats.decode_graph_ready_mask, stats.verify_graph_ready_mask),
        (widths, widths),
        "verify graphs at the same widths"
    );

    let mut sequences: Vec<Seq<'_>> =
        prompts.iter().map(|_| pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"))).collect();
    for (seq, prompt) in sequences.iter_mut().zip(&prompts) {
        prefill_program_sampled(&model, &pool, seq, prompt, 0, SamplingParams::greedy(), None)
            .unwrap_or_else(|e| panic!("prefill: {e}"));
    }
    let lanes = |n: usize| vec![VerifyLane { remaining_tokens: 16, ..VerifyLane::greedy(&[]) }; n];

    // Three lanes: refused, both rounds, by the load's lanes; nothing moved.
    let positions: Vec<u64> = sequences.iter().map(|s| s.stats().position).collect();
    let mut all: Vec<&mut Seq<'_>> = sequences.iter_mut().collect();
    let err = decode_program_verify_runs(&model, &pool, &mut all, &lanes(3), WINDOW).err().expect("a verify round of three");
    assert!(err.contains("exceeds the load's decode lanes (2)"), "{err}");
    let err = decode_program_batch_sampled(&model, &pool, &mut all, &[SamplingParams::greedy(); 3])
        .err()
        .expect("a decode round of three");
    assert!(err.contains("exceeds the load's decode lanes (2)"), "{err}");
    assert_eq!(all.iter().map(|s| s.stats().position).collect::<Vec<_>>(), positions, "a refused round moves no lane");

    // Two lanes: served from the captured graph, each lane committing a run.
    let mut two: Vec<&mut Seq<'_>> = sequences.iter_mut().take(LANES as usize).collect();
    let runs = decode_program_verify_runs(&model, &pool, &mut two, &lanes(2), WINDOW)
        .unwrap_or_else(|e| panic!("a verify round of two: {e}"));
    assert_eq!(runs.len(), 2);
    assert!(runs.iter().all(|run| !run.tokens.is_empty()), "{runs:?}");
    let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
    assert_eq!(stats.graph_launches, 1, "the width-2 verify graph replayed");
}
