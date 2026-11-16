//! What a round costs under the 27B's MTP head at each draft window (GitHub
//! #628, spec #618): wall time of one verify round with the head drafting at
//! windows 1..7, of the same round with no drafter (the verify substrate fed
//! its drafts, `SpeculativeBackend::VerifyOnly`), and of a plain decode round
//! (window 0), at one lane and at the load's lanes, each lane standing at the
//! same context; and what each load's prefill took to bring the lanes there.
//! The MTP sibling of `verify_width_cost.rs`, which is unchanged.
//!
//! ```text
//! cargo run --release -p ignis-runtime --features cuda --example mtp_round_cost -- \
//!     [--lanes N] [--prefill-chunk N] ARTIFACT TOKENS_BIN CONTEXTS ROUNDS OUT_JSONL [WINDOWS]
//! ```
//!
//! `TOKENS_BIN` is int32 LE text (any real stream). `CONTEXTS` is an ascending
//! comma list: lane `i` reads the text from `i * last context`, so the lanes
//! are distinct, and is prefilled to the first context, measured, extended by
//! prefill to the next, measured again. Loads, each at the artifact's body's
//! replay shape (`support/replay_shape.rs`: the trellis one in `rtx50-16gb`'s
//! 16 GiB plan at its two lanes, the NVFP4 one at eight or `--lanes`) planned
//! with the server's plan pieces (`support/replay_load.rs`), graphs captured:
//! plain, then for each window `k` in `WINDOWS` (default 1..7) the MTP head
//! at `k` and the verify substrate at `k`. Per context and lane count `n`
//! (1 and the load's lanes), `ROUNDS` greedy rounds over the first `n` lanes
//! after 3 warm-up rounds, each timed on the host around the call (the call
//! returns after its device work). One JSON line per (load, context, lanes):
//! median, mean and p10/p90 round milliseconds; and one per (load, context)
//! for the prefill: each lane's milliseconds to bring it from the previous
//! context to this one.
//!
//! A round's cost does not depend on what it accepts: the head's chain runs
//! its `k` steps and the verify round computes every lane's `k + 1` columns
//! whatever the accept keeps. The substrate's rounds verify the lane's next
//! `k` tokens of the text as drafts, which the lane mostly rejects; what the
//! head adds to a round is the MTP round's time less the substrate's at the
//! same window, lanes and context: its chain and the commit of its rows.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use ignis_artifact::{body_27b, Reader};
use ignis_core::seq::Seq;
use ignis_core::step::{
    decode_program_batch_sampled, decode_program_verify_runs, prefill_program_sampled, SamplingParams, VerifyLane,
};
use ignis_core::SpeculativeBackend;

#[path = "support/replay_shape.rs"]
mod replay_shape;
#[path = "support/replay_load.rs"]
mod replay_load;

use replay_shape::{LoadShape, ShapeFlags};

const WARMUP: usize = 3;

fn stats(mut samples: Vec<f64>) -> serde_json::Value {
    samples.sort_by(f64::total_cmp);
    let at = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize];
    serde_json::json!({
        "median_ms": at(0.5), "mean_ms": samples.iter().sum::<f64>() / samples.len() as f64,
        "p10_ms": at(0.1), "p90_ms": at(0.9), "samples": samples.len(),
    })
}

/// Lane `i`'s text: `stride` tokens from `i * stride`.
fn lane_text(text: &[i32], lane: usize, stride: usize) -> &[i32] {
    &text[lane * stride..(lane + 1) * stride]
}

/// What a load drafts with: nothing, the head, or the substrate fed drafts.
#[derive(Clone, Copy)]
enum Drafter {
    Plain,
    Mtp(u32),
    Substrate(u32),
}

impl Drafter {
    fn label(self) -> String {
        match self {
            Drafter::Plain => "plain".into(),
            Drafter::Mtp(k) => format!("mtp-{k}"),
            Drafter::Substrate(k) => format!("verify-only-{k}"),
        }
    }

    fn window(self) -> u32 {
        match self {
            Drafter::Plain => 0,
            Drafter::Mtp(k) | Drafter::Substrate(k) => k,
        }
    }

    fn shape(self, base: LoadShape) -> LoadShape {
        match self {
            Drafter::Plain => LoadShape { window: 0, ..base },
            Drafter::Mtp(k) => base.with_mtp(k).unwrap_or_else(|e| panic!("mtp shape: {e}")),
            Drafter::Substrate(k) => LoadShape { drafter: SpeculativeBackend::VerifyOnly, window: k, ..base },
        }
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    let flags = ShapeFlags::take(&mut args).unwrap_or_else(|e| panic!("{e}"));
    if !(6..=7).contains(&args.len()) {
        eprintln!(
            "usage: mtp_round_cost [--lanes N] [--prefill-chunk N] ARTIFACT TOKENS_BIN CONTEXTS ROUNDS OUT_JSONL [WINDOWS]"
        );
        std::process::exit(2);
    }
    let bytes = std::fs::read(&args[2]).unwrap_or_else(|e| panic!("read {}: {e}", args[2]));
    let text: Vec<i32> = bytes.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let contexts: Vec<usize> = args[3].split(',').map(|c| c.parse().expect("context")).collect();
    assert!(contexts.windows(2).all(|w| w[0] < w[1]), "contexts must ascend");
    let stride = *contexts.last().unwrap();
    let rounds: usize = args[4].parse().expect("rounds");
    let windows: Vec<u32> = args
        .get(6)
        .map_or_else(|| (1..=7).collect(), |v| v.split(',').map(|w| w.parse().expect("window")).collect());
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&args[5]).expect("open out");

    let reader = Reader::open(Path::new(&args[1])).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let body = body_27b(&reader);
    let base = LoadShape::new(body, flags).unwrap_or_else(|e| panic!("load shape: {e}"));
    let lanes = base.lanes as usize;
    let lane_counts: Vec<usize> = if lanes == 1 { vec![1] } else { vec![1, lanes] };
    assert!(text.len() >= lanes * stride + 8, "the text holds {} tokens, not {}", text.len(), lanes * stride + 8);
    // Room for the prompt plus every timed round's full commit at every context.
    let committed_max = contexts.len() * lane_counts.len() * (WARMUP + rounds) * 8;
    let lane_context = ((stride + committed_max + 64) as u32).div_ceil(64) * 64;
    assert!(lane_context <= base.max_context, "a lane's {lane_context} tokens outgrow the load's {}", base.max_context);
    let mut record = |line: serde_json::Value| {
        println!("{line}");
        writeln!(out, "{line}").unwrap();
    };

    let drafters: Vec<Drafter> = std::iter::once(Drafter::Plain)
        .chain(windows.iter().flat_map(|&k| [Drafter::Mtp(k), Drafter::Substrate(k)]))
        .collect();
    for drafter in drafters {
        let window = drafter.window();
        let loaded = replay_load::load(&reader, drafter.shape(base))
            .unwrap_or_else(|e| panic!("load ({}): {e}", drafter.label()));
        {
            let (model, pool) = (&loaded.model, &loaded.pool);
            let mut seqs: Vec<Seq<'_>> =
                (0..lanes).map(|_| pool.alloc(lane_context).unwrap_or_else(|e| panic!("alloc: {e}"))).collect();
            let label = drafter.label();
            let mut prefilled = 0;
            for &context in &contexts {
                let mut prefill_ms = Vec::with_capacity(lanes);
                for (i, seq) in seqs.iter_mut().enumerate() {
                    let position = seq.stats().position;
                    let began = Instant::now();
                    prefill_program_sampled(
                        model,
                        pool,
                        seq,
                        &lane_text(&text, i, stride)[prefilled..context],
                        position,
                        SamplingParams::greedy(),
                        None,
                    )
                    .unwrap_or_else(|e| panic!("prefill: {e}"));
                    prefill_ms.push(began.elapsed().as_secs_f64() * 1e3);
                }
                record(serde_json::json!({
                    "load": label, "body": format!("{body:?}"), "window": window, "context": context,
                    "prefill": { "from": prefilled, "tokens": context - prefilled, "lane_ms": prefill_ms },
                }));
                prefilled = context;
                for &n in &lane_counts {
                    let mut samples = Vec::with_capacity(rounds);
                    for index in 0..WARMUP + rounds {
                        // The substrate's drafts: each lane's next `k` tokens of
                        // the text past where the lane stands.
                        let drafts: Vec<Vec<i32>> = match drafter {
                            Drafter::Substrate(k) => (0..n)
                                .map(|i| {
                                    let from = i * stride + (seqs[i].stats().position as usize % stride);
                                    text[from..from + k as usize].to_vec()
                                })
                                .collect(),
                            _ => vec![Vec::new(); n],
                        };
                        let mut refs: Vec<&mut Seq<'_>> = seqs.iter_mut().take(n).collect();
                        let verify_lanes: Vec<VerifyLane<'_>> = drafts.iter().map(|d| VerifyLane::greedy(d)).collect();
                        let began = Instant::now();
                        match drafter {
                            Drafter::Plain => {
                                decode_program_batch_sampled(model, pool, &mut refs, &vec![SamplingParams::greedy(); n])
                                    .unwrap_or_else(|e| panic!("decode: {e}"));
                            }
                            _ => {
                                decode_program_verify_runs(model, pool, &mut refs, &verify_lanes, window)
                                    .unwrap_or_else(|e| panic!("round: {e}"));
                            }
                        }
                        if index >= WARMUP {
                            samples.push(began.elapsed().as_secs_f64() * 1e3);
                        }
                    }
                    record(serde_json::json!({
                        "load": label, "body": format!("{body:?}"), "window": window, "verify_width": window + 1,
                        "lanes": n, "context": context, "round": stats(samples),
                    }));
                }
            }
        }
        loaded.release();
    }
}
