//! The head runs only on a prompt's last prefill job (GitHub #519, #520, spec
//! #502 part 4), on each leaf through the `StepLeaf` seam `RuntimeCompute`
//! calls.
//!
//! One prompt is cut into several jobs and prefilled twice on the same load:
//! once as before #519, every job handed no next prompt token (every job
//! draws), and once with each job handed the prompt's token after its span,
//! the last none (`PrefillJob::next_prompt_token`), one job keeping its draw
//! as a job whose state a prefix or a checkpoint retains does. Then:
//!
//! - **the intermediate jobs run no head**: the sequence holds no pending
//!   token after each of them -- the draw is the head's only product, and
//!   the first run holds one there; the job that keeps its draw holds the
//!   first run's;
//! - **the last job's draw, the sequence's state and the greedy text are the
//!   first run's**: the pending token after the last job, the whole snapshot
//!   blob at the end of the prefill (every section, the drafter's window
//!   included), and the tokens a few greedy rounds then commit;
//! - **Flash-Next with MTP builds its head's entry at a job's last position
//!   from the next prompt token** (#520), not from the job's draw: the whole
//!   state at the prompt's end, the head's entries at every job boundary
//!   included, is a one-span prefill's of the same prompt, while the first
//!   run's differs from it; the text is the first run's. And on the shapes
//!   the scheduler deals, cuts off the chunk grid, partial chunks and a job
//!   that keeps its draw, a prompt whose token after each cut is the first
//!   run's draw there leaves the two runs in one state, byte for byte.
//!
//! Which job gets a token, and that a job whose state a prefix or a checkpoint
//! retains keeps its draw, is the scheduler's and `RuntimeCompute`'s
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
/// The job that keeps its draw, as one whose state a prefix image retains
/// (GitHub #520): the prefix-like cut.
const KEPT: usize = 1;
/// The Flash-Next MTP load's prefill chunk, and its jobs: each a whole chunk
/// of a one-span prefill of the prompt, so a job runs exactly the chunk that
/// span runs and the two states compare bit for bit.
const MTP_CHUNK: u32 = 256;
const MTP_CUTS: [usize; 5] = [256, 512, 768, 1024, 1200];
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
            checked: false,
        };
        let runs = leaf
            .decode(model, &mut [&mut *seq], std::slice::from_ref(&lane))
            .unwrap_or_else(|code| panic!("{label}: decode: leaf code {code}"));
        text.extend_from_slice(&runs[0].tokens);
    }
    text
}

/// How a run deals the prompt's jobs.
#[derive(Clone, Copy)]
enum Jobs<'a> {
    /// Before #519: every job handed no next prompt token, so every one draws.
    Drawing(&'a [usize]),
    /// Each job handed the prompt's token after its span, the last none, and
    /// the `.1`th keeping its draw, as a job a prefix image retains.
    Next(&'a [usize], usize),
}

/// The prompt prefilled as `jobs` deals it on a fresh sequence.
fn run<L: StepLeaf>(leaf: &L, model: &L::Model, prompt: &[TokenId], jobs: Jobs<'_>, header: usize, label: &str) -> Run {
    let (cuts, hands_next, kept) = match jobs {
        Jobs::Drawing(cuts) => (cuts, false, None),
        Jobs::Next(cuts, kept) => (cuts, true, Some(kept)),
    };
    let mut seq = leaf.allocate_sequence(model, MAX_CONTEXT).unwrap_or_else(|code| panic!("{label}: alloc: {code}"));
    let mut pending = Vec::with_capacity(cuts.len());
    let mut start = 0;
    for (job, &end) in cuts.iter().enumerate() {
        let next = if hands_next { prompt.get(end).copied() } else { None };
        let keeps_draw = kept == Some(job);
        // Greedy, as the scheduler deals every job but the prompt's last.
        leaf.prefill(model, &mut seq, &prompt[start..end], start as u32, 0, next, keeps_draw, DecodeParams::default(), &[], None, None, None)
            .unwrap_or_else(|code| panic!("{label}: prefill [{start}, {end}): leaf code {code}"));
        pending.push(pending_of(&snapshot(leaf, model, &seq, label), header));
        start = end;
    }
    let blob = snapshot(leaf, model, &seq, label);
    let text = greedy(leaf, model, &mut seq, label);
    leaf.release_sequence(model, seq);
    Run { pending, blob, text }
}

/// The prompt with its token after each of `cuts` but the last replaced by
/// the greedy draw a drawing run of `cuts` makes there, built one job at a
/// time, since a job's draw reads nothing past its span. A run handed next
/// prompt tokens over it then builds each job boundary's entry from the token
/// the drawing run built it from (GitHub #520).
fn drawn_at_cuts<L: StepLeaf>(leaf: &L, model: &L::Model, cuts: &[usize], header: usize, label: &str) -> Vec<TokenId> {
    let mut prompt = prompt();
    let mut seq = leaf.allocate_sequence(model, MAX_CONTEXT).unwrap_or_else(|code| panic!("{label}: alloc: {code}"));
    let mut start = 0;
    for &end in cuts {
        leaf.prefill(model, &mut seq, &prompt[start..end], start as u32, 0, None, false, DecodeParams::default(), &[], None, None, None)
            .unwrap_or_else(|code| panic!("{label}: prefill [{start}, {end}): leaf code {code}"));
        if end < prompt.len() {
            let drawn = pending_of(&snapshot(leaf, model, &seq, label), header).expect("a drawing job draws");
            prompt[end] = drawn as TokenId;
        }
        start = end;
    }
    leaf.release_sequence(model, seq);
    prompt
}

/// The drawing run and the run with next tokens over [`CUTS`], each after one
/// that decoded the same text. A snapshot copies whole pages, and the rows of
/// the last page past the prompt's end are whatever the previous sequence on
/// those pages decoded there -- after a fresh load, nothing: Flash-Next's
/// indexer keys differ there between a load's first run and its second, the
/// same path both times. So a first run warms the pages and is compared with
/// nothing.
fn today_and_cut<L: StepLeaf>(leaf: &L, model: &L::Model, header: usize, label: &str) -> (Run, Run) {
    let prompt = prompt();
    let warm = run(leaf, model, &prompt, Jobs::Drawing(&CUTS), header, &format!("{label}, warming the pages"));
    let today = run(leaf, model, &prompt, Jobs::Drawing(&CUTS), header, &format!("{label}, today"));
    let cut = run(leaf, model, &prompt, Jobs::Next(&CUTS, KEPT), header, &format!("{label}, next tokens"));
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

/// The head ran only on the last job and the one that keeps its draw, and
/// each of them drew what the drawing run drew there.
fn the_head_runs_only_where_a_draw_is_read(today: &Run, cut: &Run, label: &str) {
    assert!(today.pending.iter().all(Option::is_some), "{label}: today every job draws: {:?}", today.pending);
    let (last, intermediate) = cut.pending.split_last().expect("jobs");
    for (job, pending) in intermediate.iter().enumerate() {
        match job == KEPT {
            true => assert_eq!(*pending, today.pending[job], "{label}: the kept job's draw: {:?}", cut.pending),
            false => assert!(pending.is_none(), "{label}: intermediate job {job} drew: {:?}", cut.pending),
        }
    }
    assert_eq!(*last, today.pending[cut.pending.len() - 1], "{label}: the last job's draw");
}

/// The head ran only where a draw is read, and the run left what the drawing
/// run leaves.
fn the_head_runs_only_where_read_and_leaves_todays_state(today: &Run, cut: &Run, header: usize, label: &str) {
    the_head_runs_only_where_a_draw_is_read(today, cut, label);
    assert!(
        cut.blob == today.blob,
        "{label}: the sequence's state at the prompt's end differs from today's: {}",
        differences(&today.blob, &cut.blob, header)
    );
    assert_eq!(cut.text, today.text, "{label}: the greedy text");
    println!("{label}: pending {:?}, text {:?}", cut.pending, cut.text);
}

fn check_format() {
    assert_eq!(snapshot_format_version(), 5, "this test reads the version-5 blob layout");
}

fn flash_next_leaf(speculation: Option<FlashNextSpeculation>, prefill_chunk_tokens: u32) -> Option<FlashNextLeaf> {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(FLASH_NEXT_DIR), PathBuf::from);
    let path = dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() {
        gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display()));
        return None;
    }
    let options = EngineOptions { max_context_tokens: MAX_CONTEXT, prefill_chunk_tokens, speculation, ..EngineOptions::default() };
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
    the_head_runs_only_where_read_and_leaves_todays_state(&today, &cut, 0, "27B");

    // What only the prompt's last span reads is refused beside a next token,
    // before any work: the sequence stays where it was.
    let prompt = prompt();
    let mut seq = leaf.allocate_sequence(&model, MAX_CONTEXT).expect("a free lane");
    let span = &prompt[..CUTS[0]];
    let next = Some(prompt[CUTS[0]]);
    let mut logits = vec![0f32; leaf.vocab(&model) as usize];
    assert!(
        leaf.prefill(&model, &mut seq, span, 0, 0, next, false, DecodeParams::default(), &[], None, Some(&mut logits), None).is_err(),
        "a logits readout beside a next token"
    );
    assert!(
        leaf.prefill(&model, &mut seq, span, 0, 0, next, false, DecodeParams::default(), &[prompt[0]], None, None, None).is_err(),
        "a permitted set beside a next token"
    );
    leaf.prefill(&model, &mut seq, span, 0, 0, next, false, DecodeParams::default(), &[], None, None, None)
        .expect("the refusals left the sequence at position 0");
    leaf.release_sequence(&model, seq);
    leaf.release_model(model);
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn flash_next_without_mtp_runs_the_head_only_on_the_prompts_last_job() {
    check_format();
    let Some(leaf) = flash_next_leaf(None, EngineOptions::default().prefill_chunk_tokens) else { return };
    let model = leaf.load_model().unwrap_or_else(|code| panic!("model load: leaf code {code}"));
    let (today, cut) = today_and_cut(&leaf, &model, FLASH_NEXT_CONTEXT_BLOCK, "Flash-Next");
    the_head_runs_only_where_read_and_leaves_todays_state(&today, &cut, FLASH_NEXT_CONTEXT_BLOCK, "Flash-Next");
    leaf.release_model(model);
}

/// GitHub #520: on an MTP load the head's entry at a job's last position is
/// built from the prompt's next token, a kept draw's job included, so the
/// state after the prompt's jobs is a one-span prefill's, the head's entries
/// at every job boundary with it; before, each was built from the job's
/// greedy draw. The jobs are whole chunks of that span ([`MTP_CUTS`]), so the
/// two run the same chunks and compare bit for bit.
#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn flash_next_with_mtp_builds_each_job_boundarys_entry_from_the_next_prompt_token() {
    check_format();
    let mtp = FlashNextSpeculation::new(SpeculativeBackend::Mtp, 3, 0).unwrap();
    let Some(leaf) = flash_next_leaf(Some(mtp), MTP_CHUNK) else { return };
    let model = leaf.load_model().unwrap_or_else(|code| panic!("model load: leaf code {code}"));
    let prompt = prompt();
    let header = FLASH_NEXT_CONTEXT_BLOCK;
    let label = "Flash-Next MTP";
    let one_span = [prompt.len()];
    // Each compared run follows one that dealt its jobs the same way, so the
    // rows past the prompt's end hold the same decode (see `today_and_cut`):
    // the run with next tokens and the one-span run reach one state, and the
    // drawing run is compared with the one-span run after another of them.
    let warm = run(&leaf, &model, &prompt, Jobs::Drawing(&MTP_CUTS), header, &format!("{label}, warming the pages"));
    let today = run(&leaf, &model, &prompt, Jobs::Drawing(&MTP_CUTS), header, &format!("{label}, drawing"));
    assert_eq!(today.text, warm.text, "{label}: the drawing run's text reproduces");
    run(&leaf, &model, &prompt, Jobs::Next(&MTP_CUTS, KEPT), header, &format!("{label}, warming for next tokens"));
    let cut = run(&leaf, &model, &prompt, Jobs::Next(&MTP_CUTS, KEPT), header, &format!("{label}, next tokens"));
    let single = run(&leaf, &model, &prompt, Jobs::Drawing(&one_span), header, &format!("{label}, one span"));
    let drawing = run(&leaf, &model, &prompt, Jobs::Drawing(&MTP_CUTS), header, &format!("{label}, drawing again"));
    assert_eq!(drawing.text, today.text, "{label}: the drawing run's text reproduces after the others");

    the_head_runs_only_where_a_draw_is_read(&today, &cut, label);
    assert!(
        cut.blob == single.blob,
        "{label}: the state after the jobs differs from a one-span prefill's: {}",
        differences(&single.blob, &cut.blob, header)
    );
    assert!(
        drawing.blob != single.blob,
        "{label}: the drawing run's entries at the job boundaries equal the one-span run's: the test reads nothing"
    );
    assert_eq!(cut.pending.last(), single.pending.last(), "{label}: the last draw is the one-span run's");
    assert_eq!(cut.text, today.text, "{label}: the greedy text is the drawing run's");
    assert_eq!(single.text, cut.text, "{label}: the one-span run's text");
    println!(
        "{label}: pending {:?}, text {:?}; the drawing run against one span: {}",
        cut.pending,
        cut.text,
        differences(&single.blob, &drawing.blob, header)
    );

    // The shapes the scheduler deals ([`CUTS`]): cuts off the chunk grid,
    // partial chunks and the kept draw's job. With each cut's next prompt
    // token the drawing run's draw there, the entries built from the next
    // token are the ones built from the draw, so the two runs reach one state
    // byte for byte -- the aligned check above shows the state holds the
    // entries, and that they move with the token they are built from.
    let label = "Flash-Next MTP, unaligned cuts";
    let drawn = drawn_at_cuts(&leaf, &model, &CUTS, header, label);
    let warm = run(&leaf, &model, &drawn, Jobs::Drawing(&CUTS), header, &format!("{label}, warming the pages"));
    let today = run(&leaf, &model, &drawn, Jobs::Drawing(&CUTS), header, &format!("{label}, drawing"));
    assert_eq!(today.text, warm.text, "{label}: the drawing run's text reproduces");
    let cut = run(&leaf, &model, &drawn, Jobs::Next(&CUTS, KEPT), header, &format!("{label}, next tokens"));
    the_head_runs_only_where_read_and_leaves_todays_state(&today, &cut, header, label);
    leaf.release_model(model);
}
