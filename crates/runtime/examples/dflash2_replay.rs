//! Teacher-forced DFlash2 replay of recorded agent turns (GitHub #267,
//! docs/specs/runtime/08-dflash2-sampled-acceptance.md): per verify round, what
//! the drafter proposed, what the target thought of it, and how much of it the
//! recorded text accepted. GitHub #340 runs it on both 27B bodies.
//!
//! ```text
//! cargo run --release -p ignis-runtime --features cuda --example dflash2_replay -- \
//!     [--lanes N] [--prefill-chunk N] [--draft-rule fixed|sampled] \
//!     ARTIFACT MANIFEST_JSON INPUT_DIR OUT_DIR [FIRST_JOB [LAST_JOB]]
//! ```
//!
//! The manifest (`tools/dflash2-replay/select.py`) lists jobs: a token stream
//! (`INPUT_DIR/<job>.tok`, int32 LE) and the recorded assistant turns in it.
//! The load follows the artifact's body (`support/replay_shape.rs`), planned
//! with the server's plan pieces (`support/replay_load.rs`): the trellis one in
//! `rtx50-16gb`'s 16 GiB plan, the NVFP4 one at the same engine shape on the
//! card's derived budget -- 262,144 context, DFlash2-7 with the full head,
//! hq-e8-2b KV, 1024-token prefill chunks, rope none, verify graphs captured.
//! On the NVFP4 body `--prefill-chunk` picks another chunk (its served 2048)
//! and `--lanes` another lane count than its eight; the trellis body refuses
//! both. For each job, on one sequence:
//!
//! - everything before a turn is prefilled, its last draw forced to the turn's
//!   first recorded token (a permitted set of one), which becomes the pending
//!   anchor;
//! - the turn is walked in verify rounds: the proposal readout reads the
//!   drafter's drafts, candidates and lattice, then a **teacher-forced** round
//!   (`step::decode_program_verify_forced`) runs the ordinary pass and commits
//!   the recorded text -- the leading drafts it agrees with plus the recorded
//!   token after them -- cut at `<|im_end|>`. Because a sampled text accepts a
//!   one-hot draft exactly when it equals it (the vendored accept's rule), the
//!   rounds are the ones a live run that sampled this text would have made;
//! - the round's target logits are read (`dflash2_readout::verify_logits`) and
//!   reduced to each column's top 20 (the sampler's cap), its full-vocabulary
//!   log-sum-exp and the forced token's logit.
//!
//! GitHub #577: `--draft-rule sampled` replays the same turns under the
//! sampled draft (spec #574, ADR 0090) at its default drafter settings. Each
//! turn's lane then samples as the recorded traffic did -- temperature 1,
//! top-k 20, top-p 0.95 -- under a seed of the turn's own, the readout reads
//! the drafts that lane would draw (`dflash2_readout::propose_sampled`), and
//! the forced round drafts along the recording and takes each recorded token
//! with the probability a live round under the rule takes it, min(p, q) / p
//! (`sampled_draft::forced_verdict`): its rounds are distributed as live
//! rounds under the rule are, over the same text. The record is the same; its
//! `drafts` are the lane's draws, and its `agreed` says the forced run took
//! only recorded tokens among their column's candidates. The rule and its
//! settings land in `OUT_DIR/rule.json`, which one directory keeps to one
//! rule; a directory without it is today's rule. The readers
//! (`tools/dflash2-replay/draft_rule.py`) print tokens per round, the survival
//! per draft column and the split per kind of output for each directory's
//! rule, beside each other.
//!
//! One fixed-size little-endian record per round lands in
//! `OUT_DIR/<job>.rounds` (layout: [`Record`]); a summary line per turn in
//! `OUT_DIR/turns.jsonl`; a job with `OUT_DIR/<job>.done` is skipped, so a run
//! resumes. Rows `0..=c-1` of a round are the recorded chain; rows past the
//! committed run are conditioned on the drafts, the target's view of paths the
//! recording did not take.

use std::collections::BinaryHeap;
use std::cmp::Reverse;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ignis_artifact::{body_27b, FrontendSet, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::dflash2_readout;
use ignis_core::step::{decode_program_verify_forced, prefill_program_permitted, SamplingParams, VerifyLane};
use ignis_core::{DraftRule, DrafterSampling};

#[path = "support/replay_shape.rs"]
mod replay_shape;
#[path = "support/replay_load.rs"]
mod replay_load;

use replay_shape::{LoadShape, ShapeFlags};

/// The record's draft window: the load's (`LoadShape::window`).
const WINDOW: usize = 7;
const COLUMNS: usize = WINDOW + 1;
/// The vendored sampler's candidate cap (`kSamplerCandidateCap`).
const TARGET_TOP: usize = 20;
/// One record's bytes, `tools/dflash2-replay/records.py`'s `RECORD` (pinned
/// there by `test_records.py`).
const RECORD_BYTES: usize = 9_520;

/// One round's record, every field little-endian, in this order:
///
/// | field | type | count |
/// |---|---|---|
/// | job, turn (dataset index), anchor position, committed, extent | u32 | 5 |
/// | forced tokens after the anchor (-1 past the turn) | i32 | 8 |
/// | drafts | i32 | 7 |
/// | candidate ids `[7][16]` | i32 | 112 |
/// | candidate logits `[7][16]` | f32 | 112 |
/// | selector scores `[7][16 pred][16 succ]` | f32 | 1792 |
/// | drafter head log-sum-exp per column | f32 | 7 |
/// | target top-20 ids `[8][20]`, descending | i32 | 160 |
/// | target top-20 logits `[8][20]` | f32 | 160 |
/// | target log-sum-exp per column (whole vocabulary, T = 1) | f32 | 8 |
/// | target logit of the forced token per column (NaN when -1) | f32 | 8 |
/// | readout agreed with the round (1) or not (0) | u32 | 1 |
struct Record {
    header: [u32; 5],
    forced: [i32; COLUMNS],
    drafts: Vec<i32>,
    candidate_ids: Vec<i32>,
    candidate_logits: Vec<f32>,
    scores: Vec<f32>,
    drafter_lse: Vec<f32>,
    top_ids: Vec<i32>,
    top_logits: Vec<f32>,
    target_lse: Vec<f32>,
    forced_logit: Vec<f32>,
    agreed: u32,
}

impl Record {
    fn write(&self, out: &mut impl Write) -> std::io::Result<()> {
        let mut bytes = Vec::with_capacity(RECORD_BYTES);
        self.header.iter().for_each(|v| bytes.extend(v.to_le_bytes()));
        self.forced.iter().for_each(|v| bytes.extend(v.to_le_bytes()));
        for ints in [&self.drafts, &self.candidate_ids] {
            ints.iter().for_each(|v| bytes.extend(v.to_le_bytes()));
        }
        for floats in [&self.candidate_logits, &self.scores, &self.drafter_lse] {
            floats.iter().for_each(|v| bytes.extend(v.to_le_bytes()));
        }
        self.top_ids.iter().for_each(|v| bytes.extend(v.to_le_bytes()));
        for floats in [&self.top_logits, &self.target_lse, &self.forced_logit] {
            floats.iter().for_each(|v| bytes.extend(v.to_le_bytes()));
        }
        bytes.extend(self.agreed.to_le_bytes());
        // A table of the wrong length would shift every later record the
        // reader slices at this size.
        assert_eq!(bytes.len(), RECORD_BYTES, "a round's record is not the reader's layout");
        out.write_all(&bytes)
    }
}

fn bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

/// A total order on logits for the top-k heap: larger value first, then the
/// smaller id (the sampler's own tie rule).
#[derive(PartialEq)]
struct Entry(f32, i32);
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then_with(|| other.1.cmp(&self.1))
    }
}

/// Logits more than this far below a row's maximum are left out of its
/// log-sum-exp: at most 248,320 * e^-18 ~ 0.4% of the partition function, and
/// the replay's host time is what keeps a round off the GPU.
const LSE_SPAN: f32 = 18.0;

/// A row's top `k` (descending) and its log-sum-exp.
fn reduce_row(row: &[u16], k: usize) -> (Vec<(i32, f32)>, f32) {
    let mut heap: BinaryHeap<Reverse<Entry>> = BinaryHeap::with_capacity(k + 1);
    // The heap's smallest value once full: most of the vocabulary is below it
    // and costs one comparison.
    let mut floor = f32::NEG_INFINITY;
    for (id, &bits) in row.iter().enumerate() {
        let v = bf16(bits);
        if v < floor {
            continue;
        }
        if heap.len() < k {
            heap.push(Reverse(Entry(v, id as i32)));
        } else if Entry(v, id as i32) > heap.peek().unwrap().0 {
            heap.pop();
            heap.push(Reverse(Entry(v, id as i32)));
        }
        if heap.len() == k {
            floor = heap.peek().unwrap().0 .0;
        }
    }
    let mut top: Vec<(i32, f32)> = heap.into_iter().map(|Reverse(Entry(v, id))| (id, v)).collect();
    top.sort_by(|a, b| Entry(b.1, b.0).cmp(&Entry(a.1, a.0)));
    let lse = row_lse(row, top[0].1);
    (top, lse)
}

/// A row's log-sum-exp given its maximum (the terms within [`LSE_SPAN`] of it).
fn row_lse(row: &[u16], max: f32) -> f32 {
    let cut = max - LSE_SPAN;
    let sum: f64 = row
        .iter()
        .map(|&bits| bf16(bits))
        .filter(|&v| v > cut)
        .map(|v| f64::from((v - max).exp()))
        .sum();
    max + sum.ln() as f32
}

/// GitHub #577: the recorded traffic's sampling (the agent harnesses':
/// temperature 1, top-k 20, top-p 0.95), the lane a turn replays as under the
/// sampled draft.
fn recorded_sampling(seed: u64) -> SamplingParams {
    SamplingParams { temperature: 1.0, top_k: 20, top_p: 0.95, seed, ..SamplingParams::greedy() }
}

/// A turn's own seed: fixed, so a replay reproduces, and the turn's alone.
fn turn_seed(job_index: usize, turn: u32) -> u64 {
    0x577_0000_0000 ^ ((job_index as u64) << 20) ^ u64::from(turn)
}

/// `OUT_DIR/rule.json`: what the directory's records were replayed under.
fn rule_document(rule: DraftRule) -> serde_json::Value {
    let drafter = DrafterSampling::default();
    let lane = recorded_sampling(0);
    match rule {
        DraftRule::Fixed => serde_json::json!({ "draft_rule": "fixed", "lane": "greedy (the recording decides)" }),
        DraftRule::Sampled => serde_json::json!({
            "draft_rule": "sampled",
            "drafter_temperature_factor": drafter.temperature_factor(),
            "drafter_top_p": drafter.top_p(),
            "temperature": lane.temperature, "top_k": lane.top_k, "top_p": lane.top_p,
            "seed": "turn_seed(job, turn) in crates/runtime/examples/dflash2_replay.rs",
        }),
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    let flags = ShapeFlags::take(&mut args).unwrap_or_else(|e| panic!("{e}"));
    if !(5..=7).contains(&args.len()) {
        eprintln!(
            "usage: dflash2_replay [--lanes N] [--prefill-chunk N] [--draft-rule fixed|sampled] ARTIFACT MANIFEST_JSON \
             INPUT_DIR OUT_DIR [FIRST_JOB [LAST_JOB]]"
        );
        std::process::exit(2);
    }
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
    let shape = LoadShape::new(body_27b(&reader), flags).unwrap_or_else(|e| panic!("load shape: {e}"));
    assert_eq!(shape.window as usize, WINDOW, "the record's window is the load's");
    // One directory, one rule: records of two rules in one directory would
    // read as one replay.
    let rule = shape.draft_rule;
    let rule_path = out_dir.join("rule.json");
    let document = rule_document(rule);
    match std::fs::read_to_string(&rule_path) {
        Ok(existing) => {
            let existing: serde_json::Value =
                serde_json::from_str(&existing).unwrap_or_else(|e| panic!("parse {}: {e}", rule_path.display()));
            assert_eq!(existing, document, "{} holds another rule's replay", out_dir.display());
        }
        Err(_) => {
            let finished = std::fs::read_dir(&out_dir)
                .unwrap_or_else(|e| panic!("read {}: {e}", out_dir.display()))
                .any(|e| e.unwrap().path().extension().is_some_and(|x| x == "done"));
            assert!(
                !finished || rule == DraftRule::Fixed,
                "{} holds a replay of today's rule (no rule.json): replay the sampled draft into another directory",
                out_dir.display()
            );
            std::fs::write(&rule_path, serde_json::to_string_pretty(&document).unwrap() + "\n")
                .unwrap_or_else(|e| panic!("write {}: {e}", rule_path.display()));
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
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    println!("replaying {} jobs at max context {max_context} under the {} draft", jobs.len(), rule.as_str());

    let mut summary = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_dir.join("turns.jsonl"))
        .unwrap_or_else(|e| panic!("open summary: {e}"));
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
            // Today's rule: a greedy lane, whose forced verdict is the
            // agreement. The sampled draft: the recorded traffic's lane.
            let lane_sampling = match rule {
                DraftRule::Fixed => SamplingParams::greedy(),
                DraftRule::Sampled => recorded_sampling(turn_seed(job_index, turn_id)),
            };
            let mut at = start;
            while at < end {
                let proposal = match rule {
                    DraftRule::Fixed => dflash2_readout::propose(model, pool, &seq, WINDOW as u32, WINDOW as u32, true),
                    DraftRule::Sampled => {
                        dflash2_readout::propose_sampled(model, pool, &seq, WINDOW as u32, WINDOW as u32, true, lane_sampling)
                    }
                }
                .unwrap_or_else(|e| panic!("{name} turn {turn_id} at {at}: readout: {e}"));
                let mut forced = [-1i32; COLUMNS];
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
                    &[VerifyLane { sampling: lane_sampling, stop_ids: &stop_ids, ..VerifyLane::greedy(&[]) }],
                    WINDOW as u32,
                    &[&forced],
                )
                .unwrap_or_else(|e| panic!("{name} turn {turn_id} at {at}: forced round: {e}"))
                .remove(0);
                round_s += round_began.elapsed().as_secs_f64();
                let committed = run.tokens.len();
                assert_eq!(run.tokens[..], tokens[at..at + committed], "{name} turn {turn_id} at {at}: the committed run");
                let cut = run.tokens.iter().position(|t| stop_ids.contains(t)).map_or(usize::MAX, |p| p + 1);
                let agreed = match rule {
                    // The forced run is the read drafts' agreement with the
                    // recording.
                    DraftRule::Fixed => {
                        let agreement = proposal.drafts.iter().zip(&forced).take_while(|(d, f)| d == f).count();
                        committed == (agreement.min(run.extent as usize) + 1).min(cut)
                    }
                    // The forced run took recorded tokens among their
                    // column's candidates only.
                    DraftRule::Sampled => {
                        let reachable =
                            (0..WINDOW).take_while(|&s| proposal.candidates(s).contains(&forced[s])).count();
                        committed <= (reachable.min(run.extent as usize) + 1).min(cut)
                    }
                };
                disagreements += usize::from(!agreed);

                let logits = dflash2_readout::verify_logits(model, 1, WINDOW as u32)
                    .unwrap_or_else(|e| panic!("{name}: verify logits: {e}"));
                let drafter = proposal.drafter_logits.as_ref().unwrap();
                // Eight host threads over the round's 15 rows: under a
                // loaded CPU this is what the GPU waits on.
                let (target_rows, drafter_rows) = std::thread::scope(|scope| {
                    let target: Vec<_> = (0..COLUMNS)
                        .map(|j| {
                            let row = &logits[j * vocab..(j + 1) * vocab];
                            scope.spawn(move || reduce_row(row, TARGET_TOP))
                        })
                        .collect();
                    let head_max = &proposal.candidate_logits;
                    let drafts: Vec<f32> = (0..WINDOW)
                        .map(|s| row_lse(&drafter[s * vocab..(s + 1) * vocab], head_max[s * dflash2_readout::TOP_K]))
                        .collect();
                    (target.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>(), drafts)
                });
                let mut top_ids = Vec::with_capacity(COLUMNS * TARGET_TOP);
                let mut top_logits = Vec::with_capacity(COLUMNS * TARGET_TOP);
                let mut target_lse = Vec::with_capacity(COLUMNS);
                let mut forced_logit = Vec::with_capacity(COLUMNS);
                for (j, (top, lse)) in target_rows.iter().enumerate() {
                    top.iter().for_each(|&(id, v)| {
                        top_ids.push(id);
                        top_logits.push(v);
                    });
                    target_lse.push(*lse);
                    forced_logit.push(if forced[j] >= 0 { bf16(logits[j * vocab + forced[j] as usize]) } else { f32::NAN });
                }
                Record {
                    header: [job_index as u32, turn_id, at as u32, committed as u32, run.extent],
                    forced,
                    drafts: proposal.drafts.clone(),
                    candidate_ids: proposal.candidate_ids.clone(),
                    candidate_logits: proposal.candidate_logits.clone(),
                    scores: proposal.scores.clone(),
                    drafter_lse: drafter_rows,
                    top_ids,
                    top_logits,
                    target_lse,
                    forced_logit,
                    agreed: u32::from(agreed),
                }
                .write(&mut records)
                .unwrap_or_else(|e| panic!("write record: {e}"));
                rounds += 1;
                at += committed;
            }
            cursor = end;
            let walk_s = began.elapsed().as_secs_f64();
            let line = serde_json::json!({
                "job": name, "turn": turn_id, "source": turn["source"], "kind": turn["kind"], "draft_rule": rule.as_str(),
                "context": start, "output": end - start, "rounds": rounds,
                "tok_per_round": (end - start) as f64 / rounds.max(1) as f64,
                "readout_disagreements": disagreements,
                "prefill_s": prefill_s, "walk_s": walk_s, "round_s": round_s,
                "ninfer": turn["dflash2"],
            });
            writeln!(summary, "{line}").unwrap();
            summary.flush().unwrap();
            println!(
                "{name} turn {turn_id} ({} {}): ctx {start} out {} -> {rounds} rounds, {:.2} tok/round \
                 (ninfer {:.2}), prefill {prefill_s:.1}s walk {walk_s:.1}s, {disagreements} readout disagreements",
                turn["source"].as_str().unwrap(),
                turn["kind"].as_str().unwrap(),
                end - start,
                (end - start) as f64 / rounds.max(1) as f64,
                turn["dflash2"]["completion_tokens"].as_f64().unwrap() / turn["dflash2"]["rounds"].as_f64().unwrap(),
            );
        }
        records.flush().unwrap();
        drop(records);
        std::fs::write(out_dir.join(format!("{name}.done")), b"").unwrap();
    }
    loaded.release();
}
