//! GPU integration coverage for P1-23's real program path.  It drives the
//! public step seam: a human-readable prompt is tokenized from the artifact,
//! prefilled one token at a time, then decoded greedily.  Two independent
//! model+sequence loads must yield identical ids.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{CudaDevice, FrontendSet, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::{decode_program_batch, prefill_program, program_stats};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 512;
const GENERATED: usize = 8;

fn run_once(
    reader: &Reader,
    artifact: &ignis_artifact::MaterializedArtifact,
    handles: &[ignis_artifact::ObjectHandle],
    prompt: &[i32],
) -> Result<Vec<i32>, String> {
    let model = load_qwen38_27b(reader, artifact, handles)?;
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_page_group_count: 8,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 1,
        },
    )?;
    let mut sequence = pool.alloc(MAX_CONTEXT)?;
    prefill_program(&model, &pool, &mut sequence, prompt, 0, None)?;
    let mut output = Vec::with_capacity(GENERATED);
    for _ in 0..GENERATED {
        output.extend(decode_program_batch(&model, &pool, &mut [&mut sequence])?);
    }
    let stats = program_stats(&model, &pool)?;
    assert!(stats.vram_bytes > 0, "program reports its allocated VRAM");
    assert!(
        stats.last_step_micros > 0,
        "program reports per-step duration"
    );
    assert_eq!(
        stats.kernel_count, 64,
        "one batch-1 round dispatches all layers"
    );
    Ok(output)
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn full_program_prefill_and_greedy_decode_are_deterministic() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let prompt: Vec<i32> = frontend
        .tokenizer()
        .encode("In one sentence, explain what Rust's println! macro does.")
        .unwrap_or_else(|e| panic!("tokenize prompt: {e}"))
        .into_iter()
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();
    assert!(!prompt.is_empty());

    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind: {e}"));
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
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!();
        }
    };
    let first = run_once(&reader, &artifact, &handles, &prompt)
        .unwrap_or_else(|e| panic!("first program run: {e}"));
    let second = run_once(&reader, &artifact, &handles, &prompt)
        .unwrap_or_else(|e| panic!("second program run: {e}"));
    assert_eq!(
        first, second,
        "two fresh model loads produce identical greedy ids"
    );
    assert!(
        first.iter().all(|&id| id >= 0),
        "generated ids are valid token ids"
    );
}
