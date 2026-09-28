//! The served `locate` prompt is the one the vote was calibrated on (spec 18
//! acceptance 6, GitHub #275).
//!
//! The heads were chosen, and checked on set D, on the bytes spec 18's
//! harness rendered (`attention_head_locate_gpu.rs`): layout L1, the copy
//! scaffold's kind text, `{"instruction": …}`, and `{"quote":"` forced after
//! the generation opener. A reworded prompt is an unmeasured one (ADR 0034).
//! So this test puts two of D's own questions — the first log and the first
//! record array — through `/v1/decide` itself, with the served artifact's
//! template and tokenizer in front of a mock engine, and holds the tokens
//! the engine was handed to the renders D's dump recorded
//! (`fixtures/locate/renders.json`): the question's, and its content-free
//! baseline's, which is the same render with the instruction `N/A`.
//!
//! Machine-local: skips when the artifact is absent (`docs/agents/testing.md`
//! — CPU-only, nowhere near the forward pass, so a skip is green).

use std::path::Path;
use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::Request;
use ignis_artifact::{FrontendSet, Reader};
use ignis_core::mock::MockCompute;
use ignis_core::{ConcreteScheduler, SchedulerConfig};
use ignis_server::Server;
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::engine::Engine;
use serde_json::{Value as JsonValue, json};
use tower::ServiceExt;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

/// Every prompt token the engine was handed for `request`, laid out at its
/// positions — a claimed prefix left as `None`.
fn prompt_of(compute: &MockCompute, request: u64) -> Vec<Option<u32>> {
    let mut prompt = Vec::new();
    for job in compute.prefill_calls().into_iter().flatten().filter(|job| job.request == request) {
        let start = job.start_position as usize;
        if prompt.len() < start + job.tokens.len() {
            prompt.resize(start + job.tokens.len(), None);
        }
        for (at, &token) in job.tokens.iter().enumerate() {
            prompt[start + at] = Some(token);
        }
    }
    prompt
}

#[tokio::test]
async fn the_served_locate_prompt_is_the_one_the_vote_was_calibrated_on() {
    let path = Path::new(ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let tokenizer_set = FrontendSet::from_reader(&reader).expect("frontend");
    let tokenizer = tokenizer_set.tokenizer();
    let fixture: JsonValue =
        serde_json::from_str(include_str!("fixtures/locate/renders.json")).expect("the fixture parses");
    let opening = fixture["opening"].as_str().expect("the opening");
    let opening_ids = tokenizer.encode(opening).expect("encode the opening");

    for case in fixture["cases"].as_array().expect("cases") {
        let id = case["id"].as_str().expect("an id");
        let render = case["render"].as_str().expect("a render");
        let instruction = case["instruction"].as_str().expect("an instruction");

        // A mock engine behind the served artifact's frontend, reporting the
        // served artifact's hash so the load is calibrated for `locate`.
        let served = ignis_core::locate::calibrated_artifacts().next().expect("a calibrated artifact");
        let compute = Arc::new(MockCompute::with_blob_identity(ignis_core::BlobIdentity {
            artifact: served,
            ..ignis_core::BlobIdentity::UNSET
        }));
        let scheduler = ConcreteScheduler::with_config(
            SchedulerConfig { model: "test-model".into(), ..SchedulerConfig::default() },
            compute.clone(),
        );
        let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(
            ArtifactTemplateProvider::new(FrontendSet::from_reader(&reader).expect("frontend")),
        ));
        // Raw text, never a `serde_json::Value`: its objects are sorted in
        // this build, and a record's key order is part of the prompt.
        let state = case["state"].as_str().expect("the state's JSON text");
        let body = format!(
            r#"{{"state":{state},"questions":{{"q":{{"type":"locate","method":"vote","compression":"none","instructions":{}}}}}}}"#,
            json!(instruction)
        );
        let request = Request::builder()
            .method("POST")
            .uri("/v1/decide")
            .header("content-type", "application/json")
            .body(Body::from(body.into_bytes()))
            .unwrap();
        let response = server.app().oneshot(request).await.unwrap();
        let status = response.status().as_u16();
        let text = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap();
        assert_eq!(status, 200, "{id}: {text}");

        // The question, asked first: every token, byte for byte.
        let mut expected = tokenizer.encode(render).expect("encode the recorded render");
        expected.extend(&opening_ids);
        let asked = prompt_of(&compute, 0);
        assert_eq!(
            asked.iter().map(|t| t.expect("the first question claims nothing")).collect::<Vec<_>>(),
            expected,
            "{id}: the served prompt is not the one D was dumped with"
        );

        // Its baseline: the same render with the instruction `N/A`, from
        // wherever the claimed state ends.
        let quoted = serde_json::to_string(instruction).expect("a JSON string");
        assert_eq!(render.matches(&quoted).count(), 1, "{id}: the instruction appears once in the render");
        let mut twin = tokenizer.encode(&render.replace(&quoted, "\"N/A\"")).expect("encode the twin");
        twin.extend(&opening_ids);
        let baseline = prompt_of(&compute, 1);
        assert_eq!(baseline.len(), twin.len(), "{id}: the baseline's length");
        for (at, token) in baseline.iter().enumerate() {
            if let Some(token) = token {
                assert_eq!(*token, twin[at], "{id}: the baseline differs at position {at}");
            }
        }
    }
}
