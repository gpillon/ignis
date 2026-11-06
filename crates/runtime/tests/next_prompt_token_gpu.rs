//! The head runs only on a prompt's last prefill job (GitHub #519, spec #502
//! part 4), on each leaf through the `StepLeaf` seam `RuntimeCompute` calls.
//!
//! One prompt is cut into several jobs and prefilled twice on the same load:
//! once as today, every job handed no next prompt token (every job draws),
//! and once with each job handed the prompt's token after its span, the last
//! none (`PrefillJob::next_prompt_token`). Then:
//!
//! - **the intermediate jobs run no head**: on the 27B and on Flash-Next
//!   without MTP the sequence holds no pending token after each of them --
//!   the draw is the head's only product, and today's run holds one there;
//! - **the last job's draw, the sequence's state and the greedy text are
//!   today's**: the pending token after the last job, the whole snapshot blob
//!   at the end of the prefill (every section, the drafter's window
//!   included), and the tokens a few greedy rounds then commit;
//! - **Flash-Next with MTP is unchanged, bit for bit**: its head's entry at a
//!   job's last position still reads that job's draw, so every job still
//!   draws (#520 removes that), and its pending tokens after every job, its
//!   blob and its text are today's.
//!
//! Which job gets a token, and that a job whose state a prefix or a checkpoint
//! retains gets none, is the scheduler's and `RuntimeCompute`'s
//! (`crates/core/tests/next_prompt_token.rs`, `runtime_compute.rs`).
//!
//! Explicit GPU profile (ADR 0006): outside `IGNIS_GPU_PROFILE=1` a missing
//! artifact or GPU is a skip, under it a failure. Run one test at a time: each
//! loads a whole model, and Flash-Next's two need ~46 GB of free host RAM.

#![cfg(feature = "cuda")]

#[path = "../../core/tests/support/snapshot_blob.rs"]
mod snapshot_blob;

use std::path::{Path, PathBuf};

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::{bind_model_scope_27b, materialize, CudaDevice, DraftModule, Reader};
use ignis_core::flash_next::EngineOptions;
use ignis_core::gpu_profile;
use ignis_core::seq::snapshot_format_version;
use ignis_core::speculation::FlashNextSpeculation;
use ignis_core::{DecodeParams, KvFormat, Speculation, SpeculativeBackend, TokenId};
use ignis_runtime::{auto_kv_pool_bytes, CudaLeaf, CudaLeafConfig, DecodeLane, FlashNextLeaf, StepLeaf};

use snapshot_blob::{read_i32, read_u32, read_u64, section, HEADER_BYTES, RECORD_BYTES, SECTION_PROGRESS};

const ARTIFACT_27B: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const FLASH_NEXT_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MAX_CONTEXT: u32 = 2048;
/// The 27B load's prefill chunk: most jobs below span several.
const PREFILL_CHUNK: u32 = 256;
/// The prompt's jobs, by where each ends: two long ones, a prefix-like cut, a
/// short opener-like one, then the rest -- the shapes the scheduler deals.
const CUTS: [usize; 5] = [300, 640, 650, 1000, 1200];
const GENERATED: usize = 16;
/// Where the progress image keeps the pending token: after the position.
const PROGRESS_PENDING_TOKEN: usize = 8;
/// The context block a Flash-Next leaf's blob opens with
/// (`flash_next_leaf.rs`'s `CONTEXT_BLOCK_BYTES`), before the pool's blob.
const FLASH_NEXT_CONTEXT_BLOCK: usize = 256;

/// What one prefill of the prompt leaves: the pending token after each job,
/// the snapshot at the prompt's end, and the greedy text after it.
struct Run {
    pending: Vec<Option<i32>>,
    blob: Vec<u8>,
    text: Vec<TokenId>,
}

fn prompt() -> Vec<TokenId> {
    (0..CUTS[CUTS.len() - 1] as u32).map(|i| 1000 + (i * 7919) % 60_000).collect()
}

fn snapshot<L: StepLeaf>(leaf: &L, model: &L::Model, seq: &L::Sequence, label: &str) -> Vec<u8> {
    let bytes = leaf.snapshot_bytes(model, seq).unwrap_or_else(|code| panic!("{label}: snapshot size: {code}"));
    let mut blob = vec![0u8; usize::try_from(bytes).expect("a blob fits usize")];
    leaf.snapshot_into(model, seq, &mut blob).unwrap_or_else(|code| panic!("{label}: snapshot: {code}"));
    blob
}

/// The sequence's pending token, read off its snapshot's progress section;
/// `None` for the -1 of a sequence that holds none.
fn pending_of(blob: &[u8], header: usize) -> Option<i32> {
    let pool_blob = &blob[header..];
    let (at, _) = section(pool_blob, SECTION_PROGRESS);
    Some(read_i32(pool_blob, at + PROGRESS_PENDING_TOKEN)).filter(|&token| token >= 0)
}

/// `GENERATED` tokens of greedy rounds on `seq`.
fn greedy<L: StepLeaf>(leaf: &L, model: &L::Model, seq: &mut L::Sequence, label: &str) -> Vec<TokenId> {
    let mut text = Vec::new();
    while text.len() < GENERATED {
        let lane = DecodeLane {
            params: DecodeParams::default(),
            remaining_tokens: (GENERATED - text.len()) as u32,
            stop_ids: &[],
            permitted: &[],
            reasoning_close: None,
        };
        let runs = leaf
            .decode(model, &mut [&mut *seq], std::slice::from_ref(&lane))
            .unwrap_or_else(|code| panic!("{label}: decode: leaf code {code}"));
        text.extend_from_slice(&runs[0].tokens);
    }
    text
}

/// The prompt prefilled as [`CUTS`]' jobs on a fresh sequence, each handed
/// the prompt's next token when `hands_next`, none as today otherwise.
fn run<L: StepLeaf>(leaf: &L, model: &L::Model, prompt: &[TokenId], hands_next: bool, header: usize, label: &str) -> Run {
    let mut seq = leaf.allocate_sequence(model, MAX_CONTEXT).unwrap_or_else(|code| panic!("{label}: alloc: {code}"));
    let mut pending = Vec::with_capacity(CUTS.len());
    let mut start = 0;
    for end in CUTS {
        let next = if hands_next { prompt.get(end).copied() } else { None };
        // Greedy, as the scheduler deals every job but the prompt's last.
        leaf.prefill(model, &mut seq, &prompt[start..end], start as u32, 0, next, DecodeParams::default(), &[], None, None, None)
            .unwrap_or_else(|code| panic!("{label}: prefill [{start}, {end}): leaf code {code}"));
        pending.push(pending_of(&snapshot(leaf, model, &seq, label), header));
        start = end;
    }
    let blob = snapshot(leaf, model, &seq, label);
    let text = greedy(leaf, model, &mut seq, label);
    leaf.release_sequence(model, seq);
    Run { pending, blob, text }
}

/// Today's run and the run with next tokens, each after one that decoded the
/// same text. A snapshot copies whole pages, and the rows of the last page
/// past the prompt's end are whatever the previous sequence on those pages
/// decoded there -- after a fresh load, nothing: Flash-Next's indexer keys
/// differ there between a load's first run and its second, today's path both
/// times. So a first run warms the pages and is compared with nothing.
fn today_and_cut<L: StepLeaf>(leaf: &L, model: &L::Model, header: usize, label: &str) -> (Run, Run) {
    let prompt = prompt();
    let warm = run(leaf, model, &prompt, false, header, &format!("{label}, warming the pages"));
    let today = run(leaf, model, &prompt, false, header, &format!("{label}, today"));
    let cut = run(leaf, model, &prompt, true, header, &format!("{label}, next tokens"));
    assert_eq!(today.text, warm.text, "{label}: today's text reproduces");
    (today, cut)
}

/// Where two blobs of one layout differ: the leaf's own block before the
/// pool's blob, and each section of it, by differing bytes.
fn differences(a: &[u8], b: &[u8], header: usize) -> String {
    if a.len() != b.len() {
        return format!("{} bytes against {}", a.len(), b.len());
    }
    let differing = |x: &[u8], y: &[u8]| x.iter().zip(y).filter(|(p, q)| p != q).count();
    let mut out = vec![format!("leaf block {} of {header}", differing(&a[..header], &b[..header]))];
    let (a, b) = (&a[header..], &b[header..]);
    let count = read_u32(a, 20) as usize;
    for record in (0..count).map(|i| HEADER_BYTES + i * RECORD_BYTES) {
        let (kind, at, bytes) = (read_u32(a, record), read_u64(a, record + 8) as usize, read_u64(a, record + 16) as usize);
        out.push(format!("section {kind} {} of {bytes}", differing(&a[at..at + bytes], &b[at..at + bytes])));
    }
    out.join(", ")
}

/// The head ran only on the last job, and left what today's run leaves.
fn the_head_runs_only_on_the_last_job(today: &Run, cut: &Run, header: usize, label: &str) {
    assert!(today.pending.iter().all(Option::is_some), "{label}: today every job draws: {:?}", today.pending);
    let (last, intermediate) = cut.pending.split_last().expect("jobs");
    assert!(intermediate.iter().all(Option::is_none), "{label}: an intermediate job drew: {:?}", cut.pending);
    assert_eq!(*last, today.pending[CUTS.len() - 1], "{label}: the last job's draw");
    assert!(
        cut.blob == today.blob,
        "{label}: the sequence's state at the prompt's end differs from today's: {}",
        differences(&today.blob, &cut.blob, header)
    );
    assert_eq!(cut.text, today.text, "{label}: the greedy text");
    println!("{label}: last draw {last:?}, text {:?}", cut.text);
}

fn check_format() {
    assert_eq!(snapshot_format_version(), 5, "this test reads the version-5 blob layout");
}

fn flash_next_leaf(speculation: Option<FlashNextSpeculation>) -> Option<FlashNextLeaf> {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(FLASH_NEXT_DIR), PathBuf::from);
    let path = dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() {
        gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display()));
        return None;
    }
    let options = EngineOptions { max_context_tokens: MAX_CONTEXT, speculation, ..EngineOptions::default() };
    match FlashNextLeaf::open(&path, options) {
        Ok(leaf) => Some(leaf),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("open the Flash-Next leaf: {e}"));
            None
        }
    }
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_27b_runs_the_head_only_on_the_prompts_last_job() {
    check_format();
    let path = Path::new(ARTIFACT_27B);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT_27B}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let (plan, handles) =
        bind_model_scope_27b(&reader, Some(DraftModule::Dflash2)).unwrap_or_else(|e| panic!("bind with dflash2: {e}"));
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
    // The serving shape: hq-e8-2b, the DFlash2 drafter, whose window every
    // job taps.
    let config = CudaLeafConfig {
        max_context_tokens: MAX_CONTEXT,
        kv_format: KvFormat::default(),
        kv_pool_bytes: auto_kv_pool_bytes(KvFormat::default(), MAX_CONTEXT),
        prefill_chunk_tokens: PREFILL_CHUNK,
        retained_device_slots: 0,
        retained_host_slots: 0,
        speculation: Some(Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap()),
        ..CudaLeafConfig::default()
    };
    let leaf = CudaLeaf::new(device, reader, artifact, handles, config);
    let model = leaf.load_model().unwrap_or_else(|code| panic!("model load: leaf code {code}"));
    let (today, cut) = today_and_cut(&leaf, &model, 0, "27B");
    the_head_runs_only_on_the_last_job(&today, &cut, 0, "27B");

    // What only the prompt's last span reads is refused beside a next token,
    // before any work: the sequence stays where it was.
    let prompt = prompt();
    let mut seq = leaf.allocate_sequence(&model, MAX_CONTEXT).expect("a free lane");
    let span = &prompt[..CUTS[0]];
    let next = Some(prompt[CUTS[0]]);
    let mut logits = vec![0f32; leaf.vocab(&model) as usize];
    assert!(
        leaf.prefill(&model, &mut seq, span, 0, 0, next, DecodeParams::default(), &[], None, Some(&mut logits), None).is_err(),
        "a logits readout beside a next token"
    );
    assert!(
        leaf.prefill(&model, &mut seq, span, 0, 0, next, DecodeParams::default(), &[prompt[0]], None, None, None).is_err(),
        "a permitted set beside a next token"
    );
    leaf.prefill(&model, &mut seq, span, 0, 0, next, DecodeParams::default(), &[], None, None, None)
        .expect("the refusals left the sequence at position 0");
    leaf.release_sequence(&model, seq);
    leaf.release_model(model);
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn flash_next_without_mtp_runs_the_head_only_on_the_prompts_last_job() {
    check_format();
    let Some(leaf) = flash_next_leaf(None) else { return };
    let model = leaf.load_model().unwrap_or_else(|code| panic!("model load: leaf code {code}"));
    let (today, cut) = today_and_cut(&leaf, &model, FLASH_NEXT_CONTEXT_BLOCK, "Flash-Next");
    the_head_runs_only_on_the_last_job(&today, &cut, FLASH_NEXT_CONTEXT_BLOCK, "Flash-Next");
    leaf.release_model(model);
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn flash_next_with_mtp_draws_on_every_job_as_today() {
    check_format();
    let mtp = FlashNextSpeculation::new(SpeculativeBackend::Mtp, 3, 0).unwrap();
    let Some(leaf) = flash_next_leaf(Some(mtp)) else { return };
    let model = leaf.load_model().unwrap_or_else(|code| panic!("model load: leaf code {code}"));
    let (today, cut) = today_and_cut(&leaf, &model, FLASH_NEXT_CONTEXT_BLOCK, "Flash-Next MTP");
    assert!(today.pending.iter().all(Option::is_some), "today every job draws: {:?}", today.pending);
    assert_eq!(cut.pending, today.pending, "Flash-Next with MTP: every job still draws, today's draw");
    assert!(
        cut.blob == today.blob,
        "Flash-Next with MTP: the state at the prompt's end differs from today's: {}",
        differences(&today.blob, &cut.blob, FLASH_NEXT_CONTEXT_BLOCK)
    );
    assert_eq!(cut.text, today.text, "Flash-Next with MTP: the greedy text");
    println!("Flash-Next MTP: pending {:?}, text {:?}", cut.pending, cut.text);
    leaf.release_model(model);
}
