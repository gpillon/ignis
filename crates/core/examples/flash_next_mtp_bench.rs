//! Spec flash-next/07's measurement (acceptance 6): greedy decode tok/s at
//! one, two and three lanes, speculation off against the MTP head, with each
//! round's verified drafts and acceptance per draft position; and the
//! verify round's marginal column cost c(w) with a fake drafter.
//!
//! ```text
//! flash_next_mtp_bench off|mtp|columns [--kv-format hq-e8-2b|bf16] [--draft-tokens k] [--draft-rows r] [--sets n] [--expert-cache-gb g]
//! ```
//!
//! - **off**: today's one-token rounds (no speculation bound).
//! - **mtp**: the head (its companion beside the artifact), at the load's
//!   draft tokens and row budget.
//! - **columns**: a verify-only load at `--draft-tokens k`, one lane, whose
//!   oracle drafter proposes the lane's own spec-off text, so every round
//!   verifies k + 1 columns and accepts them all: the round time per column
//!   against `off`'s is c(w).
//!
//! Prompts: the converter's reference windows (the allowlisted corpus, spec
//! 01) -- two code and two prose windows of 1,536 tokens, and the code and
//! prose documents at 24,576 (one lane each); `TOKENS` greedy tokens each (`IGNIS_BENCH_TOKENS`).
//! The rounds' wall time includes the host's n-gram staging, as a serving
//! loop pays it; the first `WARMUP` rounds are dropped. Machine-local
//! (`IGNIS_FLASH_NEXT_DIR`), needs the GPU lock and ~38 GB of free RAM.

use std::path::{Path, PathBuf};
use std::time::Duration;

use ignis_core::flash_next::{EngineOptions, FlashNextEngine, LaneRound};
use ignis_core::kv_format::KvFormat;
use ignis_core::speculation::{FlashNextSpeculation, SpeculativeBackend};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const EXPERT_CACHE_BYTES: u64 = 16_000_000_000;
const TOKENS: usize = 256;
const WARMUP: usize = 4;

fn tokens() -> usize {
    std::env::var("IGNIS_BENCH_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(TOKENS)
}

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

fn read_u32s(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// `len` tokens from window `window` of reference set `set` (test2048's
/// windows are 2,048 apart, long8192's 8,192 and consecutive).
fn prompt(set: &str, window: usize, len: usize) -> Result<Vec<u32>, String> {
    let tokens = read_u32s(&model_dir().join("references").join(set).join("tokens.u32"))?;
    let start = window * if set == "test2048" { 2048 } else { 8192 };
    tokens.get(start..start + len).map(<[u32]>::to_vec).ok_or_else(|| format!("{set}[{window}] has no {len} tokens"))
}

struct Measured {
    tokens: Vec<Vec<u32>>,
    rounds: Vec<Vec<LaneRound>>,
    times: Vec<Duration>,
}

impl Measured {
    /// Tokens per second over the rounds past the warm-up, all lanes.
    fn rate(&self) -> f64 {
        let committed: u32 = self.rounds.iter().skip(WARMUP).flatten().map(|l| l.committed).sum();
        let time: f64 = self.times.iter().skip(WARMUP).map(Duration::as_secs_f64).sum();
        f64::from(committed) / time
    }

    fn mean_round_ms(&self) -> f64 {
        let n = self.times.len().saturating_sub(WARMUP).max(1);
        self.times.iter().skip(WARMUP).map(Duration::as_secs_f64).sum::<f64>() * 1e3 / n as f64
    }

    /// Each draft position's (accepted, reached), and the histogram of the
    /// rounds' extents.
    fn acceptance(&self) -> (Vec<(u32, u32)>, Vec<u32>) {
        let mut at = vec![(0u32, 0u32); 8];
        let mut extents = vec![0u32; 8];
        for lane in self.rounds.iter().flatten() {
            extents[lane.extent as usize] += 1;
            let accepted = lane.committed - 1;
            for j in 0..lane.extent.min(accepted + 1) {
                at[j as usize].1 += 1;
                if j < accepted {
                    at[j as usize].0 += 1;
                }
            }
        }
        (at, extents)
    }
}

fn run(engine: &mut FlashNextEngine, prompts: &[Vec<u32>], oracle: Option<&[Vec<u32>]>) -> Result<Measured, String> {
    if engine.options().speculation.is_none() {
        let (tokens, times) = engine.generate_timed(prompts, tokens())?;
        let rounds = times.iter().map(|_| vec![LaneRound { extent: 0, committed: 1 }; prompts.len()]).collect();
        return Ok(Measured { tokens, rounds, times });
    }
    let mut drafter = |lane: usize, emitted: &[u32], window: u32| -> Vec<u32> {
        let text = &oracle.expect("an oracle")[lane];
        let next = emitted.len() + 1;
        text[next.min(text.len())..(next + window as usize).min(text.len())].to_vec()
    };
    let run = engine.generate_speculative(
        prompts,
        tokens(),
        oracle.map(|_| &mut drafter as &mut dyn FnMut(usize, &[u32], u32) -> Vec<u32>),
    )?;
    Ok(Measured { tokens: run.tokens, rounds: run.rounds, times: run.times })
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().cloned().ok_or("usage: flash_next_mtp_bench off|mtp|columns [options]")?;
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let kv_format = match flag("--kv-format").as_deref() {
        None | Some("hq-e8-2b") => KvFormat::HqE8_2b,
        Some("bf16") => KvFormat::Bf16,
        Some(other) => return Err(format!("--kv-format {other}")),
    };
    let draft_tokens = flag("--draft-tokens").map_or(Ok(2), |v| v.parse::<u32>().map_err(|e| e.to_string()))?;
    let draft_rows = flag("--draft-rows").map_or(Ok(0), |v| v.parse::<u32>().map_err(|e| e.to_string()))?;
    let speculation = match mode.as_str() {
        "off" => None,
        "mtp" => Some(FlashNextSpeculation::new(SpeculativeBackend::Mtp, draft_tokens, draft_rows)?),
        "columns" => Some(FlashNextSpeculation::new(SpeculativeBackend::VerifyOnly, draft_tokens, draft_rows)?),
        other => return Err(format!("unknown mode {other}")),
    };
    let options = EngineOptions {
        max_context_tokens: 32 * 1024,
        kv_format,
        expert_cache_bytes: flag("--expert-cache-gb")
            .and_then(|v| v.parse::<f64>().ok())
            .map_or(EXPERT_CACHE_BYTES, |gb| (gb * 1e9) as u64),
        speculation,
        ..EngineOptions::default()
    };
    let mut engine = FlashNextEngine::load(&model_dir(), options)?;
    println!(
        "mode {mode} kv {kv_format:?} draft tokens {draft_tokens} rows {draft_rows} graphs {:#b} verify graphs {:#b} {:?}",
        engine.graphs_ready(),
        engine.verify_graphs_ready()?,
        ignis_core::step::last_decode_graph_error()
    );
    let short = [
        ("code", prompt("test2048", 0, 1536)?),
        ("code", prompt("test2048", 32, 1536)?),
        ("prose", prompt("test2048", 2, 1536)?),
        ("prose", prompt("test2048", 8, 1536)?),
    ];
    let long = [("code-24k", prompt("long8192", 0, 24576)?), ("prose-24k", prompt("long8192", 4, 24576)?)];
    let mut sets: Vec<(String, Vec<Vec<u32>>)> = Vec::new();
    for (name, p) in &short {
        sets.push((format!("1 lane {name}"), vec![p.clone()]));
    }
    sets.push(("2 lanes code+prose".into(), vec![short[0].1.clone(), short[2].1.clone()]));
    sets.push(("2 lanes code+code".into(), vec![short[0].1.clone(), short[1].1.clone()]));
    sets.push(("3 lanes".into(), vec![short[0].1.clone(), short[2].1.clone(), short[1].1.clone()]));
    for (name, p) in &long {
        sets.push((format!("1 lane {name}"), vec![p.clone()]));
    }
    if let Some(n) = flag("--sets").and_then(|v| v.parse::<usize>().ok()) {
        sets.truncate(n);
    }
    for (name, prompts) in &sets {
        // The oracle drafter needs the spec-off text: a draft-free verify run of the same load.
        let oracle =
            if mode == "columns" { Some(run(&mut engine, prompts, Some(&vec![Vec::new(); prompts.len()]))?.tokens) } else { None };
        let before = engine.residency().counters()?;
        let m = run(&mut engine, prompts, oracle.as_deref())?;
        let after = engine.residency().counters()?;
        let decode = |c: &ignis_core::residency::ResidencyCounters| -> (u64, u64) {
            (c.hits.iter().map(|h| h[0]).sum(), c.misses.iter().map(|m| m[0]).sum())
        };
        let ((h0, m0), (h1, m1)) = (decode(&before), decode(&after));
        let rounds = m.rounds.len() as f64;
        println!(
            "  decode residency: {:.1} hits, {:.1} misses, {:.2} MB moved per round",
            (h1 - h0) as f64 / rounds,
            (m1 - m0) as f64 / rounds,
            (after.bytes_moved[0] - before.bytes_moved[0]) as f64 / rounds / 1e6
        );
        let (alpha, extents) = m.acceptance();
        let alpha: Vec<String> = alpha
            .iter()
            .take_while(|&&(_, reached)| reached > 0)
            .map(|&(hit, reached)| format!("{:.3}", f64::from(hit) / f64::from(reached)))
            .collect();
        println!(
            "{name}: {:.1} tok/s, {:.2} ms/round, {} rounds, extents {:?}, acceptance per position [{}]",
            m.rate(),
            m.mean_round_ms(),
            m.rounds.len(),
            extents.iter().enumerate().filter(|(_, n)| **n > 0).collect::<Vec<_>>(),
            alpha.join(", ")
        );
        let _ = m.tokens;
    }
    Ok(())
}
