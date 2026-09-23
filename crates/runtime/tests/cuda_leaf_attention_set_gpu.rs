//! Spec 14 acceptance 1, its last clause (GitHub #263, ADR 0039): **if any
//! armed layer cannot read, the question fails** — through the production
//! `CudaLeaf` and `RuntimeCompute`, on an hq-e8-2b load with vision.
//!
//! A head point naming the served head set arms nine GQA layers. Under
//! hq-e8-2b the leaf reads each layer's keys from the prompt route's plane,
//! which only a chunk wider than eight tokens materializes (the scheduler's
//! tail rule keeps a head point's last chunk at nine or more). Here the jobs
//! are cut by hand, so the rule can be broken on purpose:
//!
//! - a last chunk of 16 tokens reads: the pointing head's scores and one key
//!   per head of the set come back, every key inside the span and none of
//!   them a fallback cell;
//! - a last chunk of 4 tokens takes the small-T route on every armed layer,
//!   and nothing comes back — not the scores, not part of the set — while the
//!   prefill itself stands.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside it a missing artifact
//! or GPU is a skip; under the profile, a hard failure.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ignis_artifact::{
    ChatMessage, ChatRenderOptions, ContentPart, CudaDevice, FrontendSet, MessageContent, ModelScope, Reader, Role,
    bind_model_scope_27b_with, materialize,
};
use ignis_core::pointing::{AttentionQuery, SetQuery, calibrated_artifacts, calibration};
use ignis_core::vision::Multimodal;
use ignis_core::{Compute, DecodeParams, KvFormat, PrefillJob, TokenId, Vision, gpu_profile};
use ignis_runtime::{CudaLeaf, CudaLeafConfig, Model, RuntimeCompute};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 2048;

fn image() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("server")
        .join("tests")
        .join("fixtures")
        .join("vision_canary")
        .join("number.png");
    std::fs::read(path).expect("the canary's number image")
}

/// The hq-e8-2b vision load, or `None` where the profile allows a skip.
fn load() -> Option<(RuntimeCompute<CudaLeaf>, FrontendSet)> {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().expect("eos");
    let (plan, handles) = bind_model_scope_27b_with(&reader, ModelScope { draft: None, vision: true })
        .unwrap_or_else(|e| panic!("bind: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}"));
            return None;
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("materialize: {e}"));
            return None;
        }
    };
    let config = CudaLeafConfig {
        max_context_tokens: MAX_CONTEXT,
        kv_format: KvFormat::HqE8_2b,
        kv_pool_bytes: ignis_runtime::auto_kv_pool_bytes(KvFormat::HqE8_2b, MAX_CONTEXT),
        vision: Some(Vision::default()),
        ..CudaLeafConfig::default()
    };
    let leaf = CudaLeaf::new(device, reader, artifact, handles, config);
    let model = Arc::new(Model::load(Arc::new(leaf)).unwrap_or_else(|e| panic!("model load: {e:?}")));
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).try_init();
    Some((RuntimeCompute::new(model, eos), frontend))
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_head_set_no_armed_layer_can_read_comes_back_unread_and_a_readable_one_whole() {
    let Some((compute, frontend)) = load() else { return };
    let processor = frontend.vision_processor().expect("vision processor");
    let messages = [ChatMessage {
        role: Role::User,
        content: MessageContent::Parts(vec![
            ContentPart::Image { url: None },
            // Long enough that the text after the image fills the 16-token
            // reading chunk below on its own.
            ContentPart::Text(
                "Where on the screen is the number printed, and which number is it? Answer in one short sentence."
                    .into(),
            ),
        ]),
        tool_calls: Vec::new(),
        reasoning_content: None,
    }];
    let prepared = frontend
        .prepare_prompt(
            &processor,
            &messages,
            &[image().as_slice()],
            ChatRenderOptions { enable_thinking: false, ..Default::default() },
            None,
        )
        .expect("prepare");
    let (tokens, prompt): (Vec<TokenId>, Multimodal) = Multimodal::from_prepared(prepared);
    let item = &prompt.media[0];
    let (rows, cols) = ((item.grid.h / 2) as u32, (item.grid.w / 2) as u32);
    let (begin, count) = (item.token_span.begin as u32, item.token_span.count as u32);
    let calibrated = calibrated_artifacts().next().and_then(calibration).expect("the served calibration");
    let set = SetQuery::for_grid(calibrated.set.expect("a head set"), rows, cols);
    let query = AttentionQuery { head: calibrated.head, key_begin: begin, key_count: count, set: Some(set.clone()) };
    let prompt = Arc::new(prompt);
    let n = tokens.len() as u32;
    let job = |request: u64, start: u32, len: u32, attention: Option<AttentionQuery>| PrefillJob {
        request,
        tokens: tokens[start as usize..(start + len) as usize].to_vec(),
        context_tokens: n,
        start_position: start,
        params: DecodeParams::default(),
        shared_prefix: None,
        publish_prefix: None,
        checkpoint: None,
        capture_checkpoint: None,
        multimodal: Some(prompt.clone()),
        readout: None,
        permitted: None,
        attention,
    };

    for (request, tail) in [(1u64, 16u32), (2, 4)] {
        assert!(begin + count + tail <= n, "the tail is text after the image");
        compute.prefill_step(&[job(request, 0, n - tail, None)]).expect("the head of the prompt prefills");
        let outcomes = compute
            .prefill_step(&[job(request, n - tail, tail, Some(query.clone()))])
            .expect("the reading chunk prefills, read or not");
        let read = &outcomes[0].attention;
        eprintln!(
            "head set over {count} keys ({rows}x{cols}), last chunk {tail} tokens: {}",
            match read {
                Some(scores) => format!("read, {} set keys", scores.set_argmax.as_ref().map_or(0, |a| a.len())),
                None => "unread".to_owned(),
            }
        );
        match tail {
            16 => {
                let scores = read.as_ref().expect("a chunk on the prompt route reads every armed layer");
                assert_eq!(scores.scores.len(), count as usize);
                let argmax = scores.set_argmax.as_ref().expect("the set comes back");
                assert_eq!(argmax.len(), set.heads.len(), "one key per head of the set");
                assert!(argmax.iter().all(|&k| k < count && !set.excluded.contains(&k)), "{argmax:?}");
            }
            _ => assert!(
                read.is_none(),
                "a small-T chunk materializes no plane on any armed layer: nothing — no scores, no part of the set — may come back"
            ),
        }
    }
    assert_eq!(compute.live_sequences(), 2, "both prefills stand, read or not");
}
