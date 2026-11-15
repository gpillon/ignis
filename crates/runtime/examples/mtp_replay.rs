//! Teacher-forced replay of recorded agent turns under the 27B's MTP head
//! (GitHub #626, spec #618): per verify round, the drafts the head's chain
//! proposed, the head's probability of each, its leading candidates, and how
//! much of the chain the recorded text accepted -- the records the
//! measurement (#628) and the comparison of adaptive extents (#629) read. The
//! DFlash2 replay (`dflash2_replay.rs`) is its sibling and is unchanged.
//!
//! ```text
//! cargo run -p ignis-runtime --features cuda --example mtp_replay -- \
//!     [--window N] [--lanes N] [--prefill-chunk N] \
//!     ARTIFACT MANIFEST_JSON INPUT_DIR OUT_DIR [FIRST_JOB [LAST_JOB]]
//! ```
//!
//! The manifest and the token streams are the DFlash2 replay's
//! (`tools/dflash2-replay/select.py`): a token stream per job
//! (`INPUT_DIR/<job>.tok`, int32 LE) and the recorded assistant turns in it.
//! The load is the DFlash2 replay's shape (`support/replay_shape.rs`), the MTP
//! head drafting in place of DFlash2 at `--window` (7 unless named): on the
//! trellis body `rtx50-16gb`'s 16 GiB plan, on the NVFP4 one the same engine
//! shape on the card's derived budget -- 262,144 context, hq-e8-2b KV,
//! 1024-token prefill chunks, rope none, verify graphs captured. The head
//! drafts under the fixed draft only, so the lane is greedy and the recording
//! decides. For each job, on one sequence:
//!
//! - everything before a turn is prefilled, its last draw forced to the turn's
//!   first recorded token (a permitted set of one), which becomes the pending
//!   anchor;
//! - the turn is walked in verify rounds: the chain readout
//!   (`mtp_readout::chain`) reads the drafts the next round proposes, each
//!   with the head's probability of it (the softmax at temperature 1 over the
//!   whole vocabulary of the head's scores) and its 16 leading candidates,
//!   then a **teacher-forced** round (`step::decode_program_verify_forced`)
//!   runs the ordinary pass, the head's chain in it, and commits the recorded
//!   text -- the leading drafts it agrees with plus the recorded token after
//!   them -- cut at `<|im_end|>`. A greedy lane accepts a draft exactly when
//!   the text has it there, so the rounds are the ones a live run that
//!   produced this text would have made.
//!
//! One fixed-size little-endian record per round lands in
//! `OUT_DIR/<job>.rounds` (layout: [`Record`], read by
//! `tools/mtp-replay/records.py`); a summary line per turn in
//! `OUT_DIR/turns.jsonl`; a job with `OUT_DIR/<job>.done` is skipped, so a run
//! resumes. `OUT_DIR/replay.json` names the drafter and the window, and one
//! directory keeps to one: a directory of another window's replay, or of the
//! DFlash2 replay's (its `rule.json`), is refused.
//!
//! The run is its own check: every round's committed run is the recording's,
//! every turn is walked to its end, and the readout's drafts must be the
//! round's -- the forced run is their leading agreement with the recording,
//! cut at a stop. A round where it is not is counted on the turn's line and
//! the run exits with status 1 once its records are written.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ignis_artifact::{body_27b, FrontendSet, Reader};
use ignis_core::mtp_readout::{self, MtpChain, TOP_K};
use ignis_core::step::{decode_program_verify_forced, prefill_program_permitted, SamplingParams, VerifyLane};
use ignis_core::DraftRule;

#[path = "support/replay_shape.rs"]
mod replay_shape;
#[path = "support/replay_load.rs"]
mod replay_load;

use replay_shape::{LoadShape, ShapeFlags};

/// The record's draft positions: the widest window (`MAX_DRAFT_TOKENS`).
const POSITIONS: usize = ignis_core::MAX_DRAFT_TOKENS as usize;
const COLUMNS: usize = POSITIONS + 1;
/// One record's bytes, `tools/mtp-replay/records.py`'s `RECORD` (pinned there
/// by `test_records.py`).
const RECORD_BYTES: usize = 1_016;

/// One round's record, every field little-endian, in this order:
///
/// | field | type | count |
/// |---|---|---|
/// | job, turn (dataset index), anchor position, window, extent, accepted, committed | u32 | 7 |
/// | committed run, the anchor first (-1 past it) | i32 | 8 |
/// | drafts (-1 past the extent) | i32 | 7 |
/// | the head's probability of each draft (NaN past the extent) | f32 | 7 |
/// | the head's candidate ids `[7][16]`, descending (-1 past the extent) | i32 | 112 |
/// | their probabilities `[7][16]` (NaN past the extent) | f32 | 112 |
/// | readout agreed with the round (1) or not (0) | u32 | 1 |
///
/// `accepted` is the drafts the recording took (their leading agreement with
/// it, at most the extent); `committed` the run's length, `accepted + 1` cut
/// at a stop id. A probability is the draft's share of the softmax at
/// temperature 1 over the head's scores, the whole vocabulary.
struct Record {
    header: [u32; 7],
    run: [i32; COLUMNS],
    drafts: [i32; POSITIONS],
    probabilities: [f32; POSITIONS],
    candidate_ids: Vec<i32>,
    candidate_probabilities: Vec<f32>,
    agreed: u32,
}

impl Record {
    /// The round's record: its header, the chain read before it and the run
    /// it committed.
    fn new(header: [u32; 7], chain: &MtpChain, run: &[i32], agreed: bool) -> Self {
        let mut record = Self {
            header,
            run: [-1; COLUMNS],
            drafts: [-1; POSITIONS],
            probabilities: [f32::NAN; POSITIONS],
            candidate_ids: vec![-1; POSITIONS * TOP_K],
            candidate_probabilities: vec![f32::NAN; POSITIONS * TOP_K],
            agreed: u32::from(agreed),
        };
        record.run[..run.len()].copy_from_slice(run);
        for (s, step) in chain.steps.iter().enumerate() {
            record.drafts[s] = step.draft;
            record.probabilities[s] = step.probability;
            record.candidate_ids[s * TOP_K..(s + 1) * TOP_K].copy_from_slice(&step.candidate_ids);
            record.candidate_probabilities[s * TOP_K..(s + 1) * TOP_K].copy_from_slice(&step.candidate_probabilities);
        }
        record
    }

    fn write(&self, out: &mut impl Write) -> std::io::Result<()> {
        let mut bytes = Vec::with_capacity(RECORD_BYTES);
        self.header.iter().for_each(|v| bytes.extend(v.to_le_bytes()));
        self.run.iter().chain(&self.drafts).for_each(|v| bytes.extend(v.to_le_bytes()));
        self.probabilities.iter().for_each(|v| bytes.extend(v.to_le_bytes()));
        self.candidate_ids.iter().for_each(|v| bytes.extend(v.to_le_bytes()));
        self.candidate_probabilities.iter().for_each(|v| bytes.extend(v.to_le_bytes()));
        bytes.extend(self.agreed.to_le_bytes());
        // A table of the wrong length would shift every later record the
        // reader slices at this size.
        assert_eq!(bytes.len(), RECORD_BYTES, "a round's record is not the reader's layout");
        out.write_all(&bytes)
    }
}

/// `--window N` out of the command line, the head's draft window (7 unless
/// named); every other argument stays, in order.
fn take_window(args: &mut Vec<String>) -> u32 {
    let Some(at) = args.iter().position(|a| a == "--window") else {
        return POSITIONS as u32;
    };
    args.remove(at);
    assert!(at < args.len(), "--window needs a value");
    let raw = args.remove(at);
    raw.parse().unwrap_or_else(|e| panic!("--window `{raw}`: {e}"))
}

/// `OUT_DIR/replay.json`: what the directory's records were replayed under.
fn replay_document(window: u32) -> serde_json::Value {
    serde_json::json!({
        "drafter": "mtp",
        "window": window,
        "draft_rule": "fixed",
        "lane": "greedy (the recording decides)",
    })
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    let window = take_window(&mut args);
    let flags = ShapeFlags::take(&mut args).unwrap_or_else(|e| panic!("{e}"));
    if !(5..=7).contains(&args.len()) {
        eprintln!(
            "usage: mtp_replay [--window N] [--lanes N] [--prefill-chunk N] ARTIFACT MANIFEST_JSON INPUT_DIR OUT_DIR \
             [FIRST_JOB [LAST_JOB]]"
        );
        std::process::exit(2);
    }
    assert!(
        flags.draft_rule.is_none_or(|rule| rule == DraftRule::Fixed),
        "the MTP head drafts under the fixed draft only"
    );
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&args[2]).unwrap_or_else(|e| panic!("read {}: {e}", args[2])),
    )
    .unwrap_or_else(|e| panic!("parse manifest: {e}"));
    let input = PathBuf::from(&args[3]);
    let out_dir = PathBuf::from(&args[4]);
    std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| panic!("create {}: {e}", out_dir.display()));
    let jobs = manifest["jobs"].as_array().expect("manifest jobs");
    let first: usize = args.get(5).map_or(0, |v| v.parse().expect("first job"));
    let last: usize = args.get(6).map_or(jobs.len() - 1, |v| v.parse().expect("last job"));
    let jobs = &jobs[first..=last.min(jobs.len() - 1)];
    let longest = jobs.iter().map(|j| j["tokens"].as_u64().unwrap()).max().unwrap_or(0) as u32;

    let reader = Reader::open(Path::new(&args[1])).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let shape = LoadShape::new(body_27b(&reader), flags)
        .and_then(|shape| shape.with_mtp(window))
        .unwrap_or_else(|e| panic!("load shape: {e}"));
    // One directory, one replay: records of two drafters or two windows in
    // one directory would read as one.
    assert!(
        !out_dir.join("rule.json").exists(),
        "{} holds a DFlash2 replay: replay MTP into another directory",
        out_dir.display()
    );
    let replay_path = out_dir.join("replay.json");
    let document = replay_document(window);
    match std::fs::read_to_string(&replay_path) {
        Ok(existing) => {
            let existing: serde_json::Value =
                serde_json::from_str(&existing).unwrap_or_else(|e| panic!("parse {}: {e}", replay_path.display()));
            assert_eq!(existing, document, "{} holds another replay", out_dir.display());
        }
        Err(_) => {
            let started = std::fs::read_dir(&out_dir)
                .unwrap_or_else(|e| panic!("read {}: {e}", out_dir.display()))
                .any(|e| e.unwrap().path().extension().is_some_and(|x| x == "rounds"));
            assert!(!started, "{} holds records of an unnamed replay", out_dir.display());
            std::fs::write(&replay_path, serde_json::to_string_pretty(&document).unwrap() + "\n")
                .unwrap_or_else(|e| panic!("write {}: {e}", replay_path.display()));
        }
    }
    // A round at the turn's end still runs its whole window.
    let max_context = shape.max_context;
    assert!(
        longest + COLUMNS as u32 <= max_context,
        "a job's {longest} tokens and a round's {COLUMNS} columns outgrow the load's {max_context}-token context"
    );
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let stop_ids: Vec<i32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .flat_map(|m| frontend.tokenizer().encode(m).unwrap_or_else(|e| panic!("tokenize: {e}")))
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();
    assert_eq!(stop_ids.len(), 2, "the stop markers must tokenize as special tokens");
    let loaded = replay_load::load(&reader, shape).unwrap_or_else(|e| panic!("load: {e}"));
    let (model, pool) = (&loaded.model, &loaded.pool);
    println!("replaying {} jobs at max context {max_context} under the MTP head at window {window}", jobs.len());

    let mut summary = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_dir.join("turns.jsonl"))
        .unwrap_or_else(|e| panic!("open summary: {e}"));
    let mut disagreements_total = 0usize;
    for (offset, job) in jobs.iter().enumerate() {
        let job_index = first + offset;
        let name = job["job"].as_str().unwrap();
        if out_dir.join(format!("{name}.done")).exists() {
            continue;
        }
        let bytes = std::fs::read(input.join(format!("{name}.tok"))).unwrap_or_else(|e| panic!("read {name}: {e}"));
        let tokens: Vec<i32> = bytes.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let mut records = std::io::BufWriter::new(
            std::fs::File::create(out_dir.join(format!("{name}.rounds"))).unwrap_or_else(|e| panic!("create: {e}")),
        );
        let mut seq = pool.alloc(max_context).unwrap_or_else(|e| panic!("alloc: {e}"));
        let mut cursor = 0usize;
        for turn in job["turns"].as_array().unwrap() {
            let (start, end) = (turn["start"].as_u64().unwrap() as usize, turn["end"].as_u64().unwrap() as usize);
            let turn_id = turn["turn"].as_u64().unwrap() as u32;
            let began = Instant::now();
            prefill_program_permitted(
                model,
                pool,
                &mut seq,
                &tokens[cursor..start],
                cursor as u64,
                SamplingParams::greedy(),
                &tokens[start..=start],
                None,
            )
            .unwrap_or_else(|e| panic!("{name} turn {turn_id}: prefill: {e}"));
            let prefill_s = began.elapsed().as_secs_f64();
            let began = Instant::now();
            let (mut rounds, mut disagreements, mut round_s) = (0usize, 0usize, 0f64);
            let mut at = start;
            while at < end {
                let chain = mtp_readout::chain(model, pool, &mut seq, window, window, false)
                    .unwrap_or_else(|e| panic!("{name} turn {turn_id} at {at}: readout: {e}"));
                let mut forced = vec![-1i32; window as usize + 1];
                for (j, slot) in forced.iter_mut().enumerate() {
                    if at + 1 + j < end {
                        *slot = tokens[at + 1 + j];
                    }
                }
                let round_began = Instant::now();
                let run = decode_program_verify_forced(
                    model,
                    pool,
                    &mut [&mut seq],
                    &[VerifyLane { stop_ids: &stop_ids, ..VerifyLane::greedy(&[]) }],
                    window,
                    &[&forced],
                )
                .unwrap_or_else(|e| panic!("{name} turn {turn_id} at {at}: forced round: {e}"))
                .remove(0);
                round_s += round_began.elapsed().as_secs_f64();
                let committed = run.tokens.len();
                assert_eq!(run.tokens[..], tokens[at..at + committed], "{name} turn {turn_id} at {at}: the committed run");
                // The forced run is the read drafts' agreement with the
                // recording, at most the round's extent, cut at a stop.
                let drafts = chain.drafts();
                let accepted = drafts.iter().zip(&forced).take_while(|(d, f)| d == f).count().min(run.extent as usize);
                let cut = run.tokens.iter().position(|t| stop_ids.contains(t)).map_or(usize::MAX, |p| p + 1);
                let agreed = run.extent == window && committed == (accepted + 1).min(cut);
                disagreements += usize::from(!agreed);
                let header =
                    [job_index as u32, turn_id, at as u32, window, run.extent, accepted as u32, committed as u32];
                Record::new(header, &chain, &run.tokens, agreed)
                    .write(&mut records)
                    .unwrap_or_else(|e| panic!("write record: {e}"));
                rounds += 1;
                at += committed;
            }
            assert_eq!(at, end, "{name} turn {turn_id}: the rounds did not walk the turn to its end");
            cursor = end;
            disagreements_total += disagreements;
            let walk_s = began.elapsed().as_secs_f64();
            let line = serde_json::json!({
                "job": name, "turn": turn_id, "source": turn["source"], "kind": turn["kind"], "drafter": "mtp",
                "window": window, "context": start, "output": end - start, "rounds": rounds,
                "tok_per_round": (end - start) as f64 / rounds.max(1) as f64,
                "readout_disagreements": disagreements,
                "prefill_s": prefill_s, "walk_s": walk_s, "round_s": round_s,
            });
            writeln!(summary, "{line}").unwrap();
            summary.flush().unwrap();
            println!(
                "{name} turn {turn_id} ({} {}): ctx {start} out {} -> {rounds} rounds, {:.2} tok/round, prefill \
                 {prefill_s:.1}s walk {walk_s:.1}s, {disagreements} readout disagreements",
                turn["source"].as_str().unwrap_or("?"),
                turn["kind"].as_str().unwrap_or("?"),
                end - start,
                (end - start) as f64 / rounds.max(1) as f64,
            );
        }
        records.flush().unwrap();
        drop(records);
        std::fs::write(out_dir.join(format!("{name}.done")), b"").unwrap();
    }
    loaded.release();
    if disagreements_total > 0 {
        eprintln!("{disagreements_total} rounds where the readout's drafts were not the round's");
        std::process::exit(1);
    }
}
