//! GPU coverage for the blob compatibility identity against a real load
//! (GitHub #189, ADR 0029; spec `.scratch/kv-reuse/specs/01-cross-request-reuse.md`
//! §"Identity and the Tier 2 seam").
//!
//! `ignis-core`'s CPU tests prove the *rule*: an identity accepts only itself,
//! and names the field that differs. Three things they cannot prove, and this
//! one does:
//!
//!   * the four facts are **real** — the artifact hash is of the container
//!     this leaf opened (not a value the test handed it), the blob layout
//!     version comes back live across the FFI from the leaf's own
//!     state-section table and is nonzero, and the format and drafter are the
//!     pool that was actually built;
//!   * a blob refused on its identity is refused **before a byte of it is
//!     written**: the four mismatches go through the same guarded restore the
//!     accepted one does, and after each one the sequence is still gone from
//!     the leaf — a decode round for it errors, which is the only way to ask —
//!     and the blob is still whole, which the legitimate restore proves at the
//!     end by succeeding and landing on the right state;
//!   * the restored sequence is the **right** one. Its continuation is
//!     compared token for token against a control sequence over the same
//!     prompt that never moved.
//!
//! The control is a *second* sequence, not the snapshotted one, so — unlike
//! `seq_snapshot_gpu.rs`, where the control is the source — its tail is not
//! itself evidence about the refusals: a refused restore would have written
//! into a freshly acquired slot for the subject, never into the control's
//! pages. It is still generated after the refusals, which costs nothing and
//! keeps the comparison out of any state they might have left behind.
//!
//! BF16 is asked for by name: it is the oracle format (ADR 0022).
//!
//! One model load, the way GitHub #186's GPU test was fixed to be: the card
//! fits one at a time, so the four refusals are four mutations of one
//! identity, not four loads, and the control is a second sequence in the same
//! pool.
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
    IdentityMismatch, KvFormat, PrefillJob, PromptContent, RequestId, Speculation,
    SpeculativeBackend,
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
    // GitHub #213: this leg evicts, so the leaf needs the arena its blob is
    // placed in. A gibibyte is many times one snapshot of this geometry.
    let leaf = CudaLeaf::new(device, reader, artifact, handles, config)
        .with_kv_ram_arena(1024 * 1024 * 1024)
        .unwrap_or_else(|e| panic!("pin KV-RAM: {e}"));
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

    // ── Two identical sequences: one is evicted, one never moves ────────
    let subject: RequestId = 1;
    let control: RequestId = 2;
    let params = DecodeParams {
        max_tokens: Some(64),
        ..DecodeParams::default()
    };
    let covered = prompt.len() as u32;
    for request in [subject, control] {
        compute
            .prefill_step(&[PrefillJob {
                request,
                tokens: prompt.clone(),
                context_tokens: MAX_CONTEXT,
                start_position: 0,
                params,
                shared_prefix: None,
                publish_prefix: None,
                checkpoint: None,
                capture_checkpoint: None,
                multimodal: None,
            }])
            .unwrap_or_else(|e| panic!("prefill_step {request}: {e}"));
    }
    // The shared head, generated by both before either moves. Greedy and
    // seeded alike (ADR 0007), so if these already differ the comparison
    // below would be meaningless and the test says so here rather than at
    // the end.
    let mut subject_head = Vec::new();
    let mut control_head = Vec::new();
    for _ in 0..GENERATED {
        subject_head.extend(decode_once(&compute, subject, params, "subject head"));
        control_head.extend(decode_once(&compute, control, params, "control head"));
    }
    assert_eq!(
        subject_head, control_head,
        "two sequences over one prompt agree before either is touched"
    );

    // ── A real snapshot, named by this load ─────────────────────────────
    assert!(
        compute
            .snapshot_size(subject)
            .unwrap_or_else(|e| panic!("snapshot_size: {e}"))
            > 0,
        "a real sequence has a real blob"
    );
    let bytes = compute
        .evict(subject)
        .unwrap_or_else(|e| panic!("evict: {e}"));
    assert!(bytes > 0, "the blob was captured into host memory");

    // What that blob would carry to a tier that does not hold its history.
    let header = BlobHeader {
        identity,
        key: PromptContent::text(&prompt).key_at(covered),
        tokens: covered,
    };

    // ── Every other load is refused, and nothing is written ─────────────
    //
    // Each mismatch goes through the same guarded restore the accepted one
    // does below, so what is exercised is the path, not an `accepts` call the
    // CPU tests already cover. The blob has to survive all four: it is
    // restored afterwards, and the restored sequence has to continue exactly
    // as the control does.
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
        let refused = restore_if_accepted(&compute, &other_load, &header, subject)
            .expect_err("a blob from this load is refused by every other");
        assert_eq!(
            refused.field, field,
            "the refusal names the field that differs: {refused}"
        );
        // Nothing was written, because nothing was restored: the sequence is
        // still gone from the leaf, which a decode round is the only way to
        // ask about.
        assert!(
            compute
                .decode_step(&[DecodeJob {
                    request: subject,
                    lane: 0,
                    params,
                    remaining_tokens: 64,
                }])
                .is_err(),
            "{}: the refused restore left the sequence unrestored",
            field.as_str()
        );
    }

    // ── The control's tail, from the sequence that never moved ──────────
    //
    // After the refusals rather than before — defensively, not as the proof.
    // A refused restore would have written into a slot freshly acquired for
    // the *subject*, which cannot be the control's; what shows that nothing
    // was written is the failing decode above and the bit-exact tail below.
    let mut control_tail = Vec::new();
    for _ in 0..GENERATED {
        control_tail.extend(decode_once(&compute, control, params, "control tail"));
    }

    // ── Its own load accepts it, and the restore is the right state ─────
    restore_if_accepted(&compute, &identity, &header, subject)
        .unwrap_or_else(|e| panic!("the load that produced the blob refused it: {e}"));
    let mut subject_tail = Vec::new();
    for _ in 0..GENERATED {
        subject_tail.extend(decode_once(&compute, subject, params, "subject tail"));
    }
    assert_eq!(
        subject_tail.len(),
        GENERATED,
        "the restored sequence decoded a full tail"
    );
    assert_eq!(
        subject_tail, control_tail,
        "the restored sequence continued exactly as one that never moved: \
         head {subject_head:?}, tail {subject_tail:?} against {control_tail:?}"
    );
    compute.release(subject);
    compute.release(control);
}

/// Restore `request` from its pending blob **only if** `load` accepts the
/// blob's identity — the guard every restore path has to stand behind (ADR
/// 0029), written once here so the refused and the accepted case go through
/// exactly the same code.
fn restore_if_accepted(
    compute: &RuntimeCompute<CudaLeaf>,
    load: &BlobIdentity,
    header: &BlobHeader,
    request: RequestId,
) -> Result<(), IdentityMismatch> {
    load.accepts(&header.identity)?;
    compute
        .restore(request, MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("restore: {e}"));
    Ok(())
}

/// One decode round's committed tokens.
fn decode_once(
    compute: &RuntimeCompute<CudaLeaf>,
    request: RequestId,
    params: DecodeParams,
    label: &str,
) -> Vec<u32> {
    compute
        .decode_step(&[DecodeJob {
            request,
            lane: 0,
            params,
            remaining_tokens: 64,
        }])
        .unwrap_or_else(|e| panic!("{label}: decode_step: {e}"))
        .into_iter()
        .flat_map(|outcome| outcome.tokens)
        .collect()
}
