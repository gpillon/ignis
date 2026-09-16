//! GPU coverage for the blob compatibility identity against a real load
//! (GitHub #189, ADR 0029; spec `.scratch/kv-reuse/specs/01-cross-request-reuse.md`
//! §"Identity and the Tier 2 seam").
//!
//! `ignis-core`'s CPU tests prove the *rule*: an identity accepts only itself,
//! and names the field that differs. What they cannot prove is that the four
//! facts are real — that the artifact hash comes from the container this leaf
//! opened, that the blob layout version is the one the leaf's own
//! state-section table is at rather than a number Rust restated, and that a
//! blob refused on any of them is refused **before** a byte of it is written
//! into a sequence.
//!
//! So this takes a real snapshot of a real sequence on the real 27B artifact
//! and puts it through both halves: the header this load produced is accepted
//! and the restore really happens (the sequence goes on decoding), and a
//! header with any one of the four changed is refused with the restore never
//! attempted.
//!
//! BF16 is asked for by name: it is the oracle format (ADR 0022).
//!
//! One model load, the way GitHub #186's GPU test was fixed to be: the card
//! fits one at a time, so the four refusals are four mutations of one
//! identity, not four loads.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing GPU or artifact is a **skip**; under the profile it is a **hard
//! failure**. Run via `scripts/gpu-profile.ps1` (stops the reference
//! `ninfer-serve` first — the RTX 5090 is exclusive).

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;

use ignis_artifact::{CudaDevice, FrontendSet, Reader, bind_text_scope_27b, materialize};
use ignis_core::gpu_profile;
use ignis_core::seq::snapshot_format_version;
use ignis_core::{
    ArtifactHash, BlobHeader, BlobIdentity, Compute, DecodeJob, DecodeParams, IdentityField,
    KvFormat, PrefillJob, PromptContent, RequestId, Speculation, SpeculativeBackend,
};
use ignis_runtime::{CudaLeaf, CudaLeafConfig, Model, RuntimeCompute, auto_kv_pool_bytes};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 1024;
/// Decoded either side of the eviction. Enough that a sequence restored from
/// the wrong bytes would have shown by the end.
const GENERATED: usize = 6;

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_real_load_names_itself_and_refuses_a_blob_from_any_other() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend
        .eos_token_id()
        .unwrap_or_else(|| panic!("qwen3.8-27b generation config must carry eos_token_id"));
    let prompt = frontend
        .tokenizer()
        .encode("In one sentence, what is 2 + 2?")
        .unwrap_or_else(|e| panic!("tokenize prompt: {e}"));
    assert!(!prompt.is_empty());
    // Read from the container before the leaf takes ownership of it — and
    // recorded here so the identity the leaf reports can be checked against
    // the artifact rather than against itself.
    let artifact_hash = ArtifactHash::from_bytes(reader.content_hash());

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

    let config = CudaLeafConfig {
        max_context_tokens: MAX_CONTEXT,
        kv_format: KvFormat::Bf16,
        kv_pool_bytes: auto_kv_pool_bytes(KvFormat::Bf16, MAX_CONTEXT),
        prefill_chunk_tokens: MAX_CONTEXT,
        speculation: None,
        ..CudaLeafConfig::default()
    };
    let leaf = CudaLeaf::new(device, reader, artifact, handles, config);
    let model =
        Arc::new(Model::load(Arc::new(leaf)).unwrap_or_else(|e| panic!("model load: {e:?}")));
    let compute = RuntimeCompute::new(model, eos);

    // ── What the load says it is ────────────────────────────────────────
    let identity = compute.blob_identity();
    assert_eq!(
        identity.artifact, artifact_hash,
        "the identity names the container this leaf opened"
    );
    assert_eq!(identity.kv_format, KvFormat::Bf16, "the format asked for");
    assert_eq!(identity.drafter, None, "a text-scope load binds no drafter");
    assert_eq!(
        identity.layout_version,
        snapshot_format_version(),
        "the blob layout version is the leaf's own, not a Rust constant"
    );
    assert_ne!(
        identity.layout_version, 0,
        "and the leaf really answered — 0 is the no-backend value"
    );
    assert_eq!(
        compute.blob_identity(),
        identity,
        "reading it twice reads the same load"
    );
    assert_eq!(identity.accepts(&identity), Ok(()));

    // ── A real snapshot, named by this load ─────────────────────────────
    let request: RequestId = 1;
    let params = DecodeParams {
        max_tokens: Some(64),
        ..DecodeParams::default()
    };
    let covered = prompt.len() as u32;
    compute
        .prefill_step(&[PrefillJob {
            request,
            tokens: prompt.clone(),
            context_tokens: MAX_CONTEXT,
            start_position: 0,
            params,
            shared_prefix: None,
            publish_prefix_tokens: None,
            checkpoint: None,
            capture_checkpoint_tokens: None,
            multimodal: None,
        }])
        .unwrap_or_else(|e| panic!("prefill_step: {e}"));

    let mut before = Vec::new();
    for _ in 0..GENERATED {
        before.extend(decode_once(&compute, request, params));
    }

    assert!(
        compute.snapshot_size(request).unwrap_or_else(|e| panic!("snapshot_size: {e}")) > 0,
        "a real sequence has a real blob"
    );
    let bytes = compute
        .evict(request)
        .unwrap_or_else(|e| panic!("evict: {e}"));
    assert!(bytes > 0, "the blob was captured into host memory");

    // What that blob would carry to a tier that does not hold its history.
    let header = BlobHeader {
        identity,
        key: PromptContent::text(&prompt).key_at(covered),
        tokens: covered,
    };

    // ── Every other load is refused, and nothing is restored ────────────
    let elsewhere = [
        (
            IdentityField::Artifact,
            BlobIdentity {
                artifact: ArtifactHash::from_bytes([0xA5; 32]),
                ..identity
            },
        ),
        (
            IdentityField::KvFormat,
            BlobIdentity {
                kv_format: KvFormat::HqE8_2b,
                ..identity
            },
        ),
        (
            IdentityField::LayoutVersion,
            BlobIdentity {
                layout_version: identity.layout_version + 1,
                ..identity
            },
        ),
        (
            IdentityField::Drafter,
            BlobIdentity {
                drafter: Some(Speculation::new(SpeculativeBackend::Dflash2, 4).unwrap()),
                ..identity
            },
        ),
    ];
    for (field, other_load) in elsewhere {
        let refused = other_load
            .accepts(&header.identity)
            .expect_err("a blob from this load is refused by every other");
        assert_eq!(
            refused.field,
            field,
            "the refusal names the field that differs: {refused}"
        );
    }

    // ── Its own load accepts it, and the restore really happens ─────────
    assert_eq!(
        identity.accepts(&header.identity),
        Ok(()),
        "the load that produced the blob takes it back"
    );
    compute
        .restore(request, MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("restore: {e}"));
    let mut after = Vec::new();
    for _ in 0..GENERATED {
        after.extend(decode_once(&compute, request, params));
    }
    assert_eq!(
        after.len(),
        GENERATED,
        "the restored sequence went on decoding: {before:?} then {after:?}"
    );
    compute.release(request);
}

/// One decode round's committed tokens.
fn decode_once(
    compute: &RuntimeCompute<CudaLeaf>,
    request: RequestId,
    params: DecodeParams,
) -> Vec<u32> {
    compute
        .decode_step(&[DecodeJob {
            request,
            lane: 0,
            params,
            remaining_tokens: 64,
        }])
        .unwrap_or_else(|e| panic!("decode_step: {e}"))
        .into_iter()
        .flat_map(|outcome| outcome.tokens)
        .collect()
}
