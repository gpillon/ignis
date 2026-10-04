//! The residency policy model replayed over real routing (spec flash-next/03,
//! GitHub #301, acceptance 5 and 6 on the CPU side).
//!
//! **Machine-local.** It reads a trace directory shaped like the artifact's
//! (`docs/specs/flash-next/layout.md` §9 and §12: `work/converter.json` with
//! the K map, the class record bytes and the calibration traffic, and
//! `traces/<domain>/`), from `IGNIS_FLASH_NEXT_TRACES` or, by default, the
//! study proxy `expert_residency_study_proxy.py` writes. Absent, it skips:
//! the default suite stays CPU-only and data-free.
//!
//! What it measures, at the study's 21.5 GB expert cache split into the
//! eight K-class pools by byte traffic, warm-started from calibration:
//! - per-domain decode hit rates at one lane and the residency cost per
//!   round at one and three lanes, against the simulation of `PLACEMENT.md`
//!   (a single byte-LRU over whole experts, no prefetch: 93-96%, 3.8 / 9.1 ms);
//! - the same with the router lookahead (W = 16);
//! - scan resistance: decode, then a 4096-token prefill of other text, then
//!   decode again — the post-prefill hit rate must stay within 2 points of
//!   the pre-prefill one (acceptance 6), and plain LRU admission of the same
//!   prefill is shown beside it.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use ignis_core::residency::{
    DEFAULT_PREFETCH_WIDTH, ExpertCacheRequest, ExpertCatalog, KBits, KClass, LayerStep,
    PolicyConfig, Projection, ResidencyModel, min_slots_per_class,
    plan_expert_cache, prefill_staging_ring_bytes, residency_table_bytes, warm_start_order,
};

const LAYERS: usize = 48;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const LOOKAHEAD: usize = 20;
/// The simulation's assumptions (`PLACEMENT.md`): 12 GB/s, 20 us per fetch.
const BANDWIDTH: f64 = 12e9;
const FETCH_LATENCY: f64 = 20e-6;
const CACHE_BYTES: u64 = 21_500_000_000;
const DECODE_STEPS: usize = 256;
/// A decode step's prefetch budget at one and three lanes: one layer's share
/// of the simulation's 6 / 7 ms step at 12 GB/s, the window `sim_bytes.py`
/// gave a prefetch to land in.
const WINDOW_BUDGET: [u64; 2] = [1_500_000, 1_750_000];

fn root() -> Option<PathBuf> {
    let path = std::env::var_os("IGNIS_FLASH_NEXT_TRACES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("F:/ai/models/flash-next-residency-study-proxy"));
    if path.join("work/converter.json").is_file() && path.join("traces").is_dir() {
        Some(path)
    } else {
        eprintln!("SKIP: no Flash-Next routing traces at {}", path.display());
        None
    }
}

struct Chunk {
    id: u64,
    domain: String,
    /// Token-major: `[token][layer][rank]`.
    experts: Vec<i16>,
    lookahead: Vec<i16>,
    tokens: usize,
}

impl Chunk {
    fn selected(&self, token: usize, layer: usize) -> Vec<u16> {
        let at = (token * LAYERS + layer) * TOP_K;
        self.experts[at..at + TOP_K].iter().map(|&e| e as u16).collect()
    }

    fn lookahead(&self, token: usize, layer: usize, width: usize) -> Vec<u16> {
        let at = (token * LAYERS + layer) * LOOKAHEAD;
        self.lookahead[at..at + width.min(LOOKAHEAD)]
            .iter()
            .filter(|&&e| e >= 0)
            .map(|&e| e as u16)
            .collect()
    }
}

struct Study {
    catalog: ExpertCatalog,
    counts: Vec<u64>,
    chunks: Vec<Chunk>,
}

fn read_i16(path: &Path) -> Vec<i16> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect()
}

fn k_of(k2: u64) -> KBits {
    match k2 {
        4 => KBits::K2,
        5 => KBits::K2_5,
        6 => KBits::K3,
        8 => KBits::K4,
        other => panic!("k2 {other} is not a K class"),
    }
}

fn load(root: &Path) -> Study {
    let text = std::fs::read_to_string(root.join("work/converter.json")).expect("converter.json");
    let json: serde_json::Value = serde_json::from_str(&text).expect("json");
    let mut map = Vec::with_capacity(LAYERS * EXPERTS);
    for layer in json["k_map"]["layers"].as_array().expect("k_map") {
        let gu = layer["gu"].as_array().expect("gu");
        let dn = layer["dn"].as_array().expect("dn");
        for (g, d) in gu.iter().zip(dn) {
            map.push((k_of(g.as_u64().unwrap()), k_of(d.as_u64().unwrap())));
        }
    }
    let mut slot_bytes = [0u64; KClass::COUNT];
    for class in json["k_classes"].as_array().expect("k_classes") {
        let name = class["class"].as_str().unwrap();
        let (proj, k) = name.split_once('-').unwrap();
        let projection = if proj == "gu" { Projection::GateUp } else { Projection::Down };
        let k = match k {
            "2" => KBits::K2,
            "2.5" => KBits::K2_5,
            "3" => KBits::K3,
            "4" => KBits::K4,
            other => panic!("class {other}"),
        };
        slot_bytes[KClass::new(projection, k).index()] = class["record_bytes"].as_u64().unwrap();
    }
    let catalog = ExpertCatalog::new(LAYERS as u16, EXPERTS as u16, map, slot_bytes).expect("catalog");
    let counts = json["expert_traffic"]["layers"]
        .as_array()
        .expect("expert_traffic")
        .iter()
        .flat_map(|l| l["counts"].as_array().unwrap().iter().map(|c| c.as_u64().unwrap()))
        .collect();

    let mut chunks = Vec::new();
    for entry in std::fs::read_dir(root.join("traces")).expect("traces") {
        let dir = entry.expect("entry").path();
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("manifest.json")).expect("manifest"),
        )
        .expect("manifest json");
        let domain = dir.file_name().unwrap().to_string_lossy().into_owned();
        let experts = read_i16(&dir.join("experts.i16"));
        let lookahead = read_i16(&dir.join("lookahead.i16"));
        let mut start = 0usize;
        for chunk in manifest["chunks"].as_array().expect("chunks") {
            let tokens = chunk["valid"].as_u64().unwrap() as usize;
            let span = |per: usize, all: &[i16]| {
                all[start * LAYERS * per..(start + tokens) * LAYERS * per].to_vec()
            };
            chunks.push(Chunk {
                id: chunk["chunk"].as_u64().unwrap(),
                domain: domain.clone(),
                experts: span(TOP_K, &experts),
                lookahead: span(LOOKAHEAD, &lookahead),
                tokens,
            });
            start += tokens;
        }
    }
    chunks.sort_by_key(|c| c.id);
    Study {
        catalog,
        counts,
        chunks,
    }
}

/// A warm-started model at the study's cache, pools split by byte traffic.
fn warm_model(study: &Study, prefetch_width: usize, prefetch_budget_bytes: Option<u64>) -> ResidencyModel {
    let catalog = &study.catalog;
    let ring = prefill_staging_ring_bytes(catalog);
    let tables = residency_table_bytes(LAYERS as u64, EXPERTS as u64);
    let plan = plan_expert_cache(&ExpertCacheRequest {
        budget_bytes: CACHE_BYTES + ring + tables,
        planned_bytes: 0,
        staging_ring_bytes: ring,
        table_bytes: tables,
        floor_bytes: 0,
        catalog,
        traffic: &study.counts,
        min_slots: min_slots_per_class(3, TOP_K as u32, DEFAULT_PREFETCH_WIDTH as u32),
    })
    .expect("plan");
    let mut model = ResidencyModel::new(
        catalog.clone(),
        PolicyConfig {
            capacity: plan.capacity(),
            prefetch_width,
            prefetch_budget_bytes,
        },
    );
    model.warm_start(&warm_start_order(catalog, &study.counts));
    model
}

#[derive(Default, Clone, Copy)]
struct Tally {
    uses: u64,
    hits: u64,
    demand_bytes: u64,
    prefetch_bytes: u64,
    /// The simulation's per-round residency cost: demand bytes at 12 GB/s
    /// plus 20 us per expert with a missed projection.
    seconds: f64,
    rounds: u64,
}

impl Tally {
    fn rate(&self) -> f64 {
        self.hits as f64 / self.uses as f64
    }

    fn cost(&self) -> f64 {
        self.seconds / self.rounds as f64
    }

    fn moved_per_round(&self) -> f64 {
        (self.demand_bytes + self.prefetch_bytes) as f64 / self.rounds as f64
    }

    fn add(&mut self, other: &Tally) {
        self.uses += other.uses;
        self.hits += other.hits;
        self.demand_bytes += other.demand_bytes;
        self.prefetch_bytes += other.prefetch_bytes;
        self.seconds += other.seconds;
        self.rounds += other.rounds;
    }
}

/// `rounds` decode rounds of `lanes` (each a chunk and a token range), one
/// step per layer per round, tallied per lane.
fn decode(
    model: &mut ResidencyModel,
    lanes: &[(&Chunk, std::ops::Range<usize>)],
    prefetch_width: usize,
    per_lane: &mut [Tally],
) {
    let rounds = lanes.iter().map(|(_, r)| r.len()).min().unwrap();
    for round in 0..rounds {
        for layer in 0..LAYERS {
            let picks: Vec<Vec<u16>> = lanes
                .iter()
                .map(|(c, r)| c.selected(r.start + round, layer))
                .collect();
            let ahead: Vec<Vec<u16>> = if prefetch_width > 0 && layer + 1 < LAYERS {
                lanes
                    .iter()
                    .map(|(c, r)| c.lookahead(r.start + round, layer, prefetch_width))
                    .collect()
            } else {
                Vec::new()
            };
            let ahead_refs: Vec<&[u16]> = ahead.iter().map(|v| v.as_slice()).collect();
            let union: Vec<u16> = picks.iter().flatten().copied().collect();
            let out = model
                .step(&LayerStep::decode(layer as u16, &union).lookahead(&ahead_refs))
                .expect("step");
            let hit: BTreeSet<_> = out.hits.iter().map(|p| (p.expert, p.projection)).collect();
            let demand: u64 = out.misses.iter().map(|(p, _)| model.catalog().bytes(*p)).sum();
            let missed_experts: BTreeSet<u16> = out.misses.iter().map(|(p, _)| p.expert).collect();
            let prefetched: u64 = out.prefetches.iter().map(|(p, _)| model.catalog().bytes(*p)).sum();
            for (lane, pick) in picks.iter().enumerate() {
                let t = &mut per_lane[lane];
                for e in pick.iter().copied().collect::<BTreeSet<u16>>() {
                    for projection in Projection::ALL {
                        t.uses += 1;
                        t.hits += u64::from(hit.contains(&(e, projection)));
                    }
                }
            }
            // The round's cost is shared: book it on lane 0.
            let t = &mut per_lane[0];
            t.demand_bytes += demand;
            t.prefetch_bytes += prefetched;
            t.seconds += demand as f64 / BANDWIDTH + FETCH_LATENCY * missed_experts.len() as f64;
        }
        per_lane[0].rounds += 1;
    }
}

/// One prefill chunk over `tokens` of `chunks`, one step per layer, the
/// lookahead (top-W of every token) included. `scan` false admits it with
/// plain LRU instead, as a decode step would.
fn prefill(model: &mut ResidencyModel, tokens: &[(&Chunk, usize)], scan: bool) -> u64 {
    let union = |pick: &dyn Fn(&Chunk, usize) -> Vec<u16>| {
        let mut seen = [false; EXPERTS];
        for (c, t) in tokens {
            for e in pick(c, *t) {
                seen[usize::from(e)] = true;
            }
        }
        (0..EXPERTS as u16).filter(|&e| seen[usize::from(e)]).collect::<Vec<u16>>()
    };
    let mut moved = 0;
    for layer in 0..LAYERS {
        let selected = union(&|c, t| c.selected(t, layer));
        let out = if scan {
            // Every token's top-W, as one-expert "lanes": the same union the
            // model would take from one ranking per token, built once.
            let ahead = if layer + 1 < LAYERS {
                union(&|c, t| c.lookahead(t, layer, DEFAULT_PREFETCH_WIDTH))
            } else {
                Vec::new()
            };
            let lanes: Vec<&[u16]> = ahead.chunks(1).collect();
            model.step(&LayerStep::prefill(layer as u16, &selected).lookahead(&lanes))
        } else {
            model.step(&LayerStep::decode(layer as u16, &selected))
        }
        .expect("step");
        moved += out.bytes_moved;
    }
    moved
}

/// `PLACEMENT.md`'s dynamic row: 1-lane hit rate per domain.
const SIMULATED: [(&str, f64); 9] = [
    ("code", 0.96),
    ("prose", 0.95),
    ("chat", 0.95),
    ("en", 0.94),
    ("it", 0.94),
    ("zh", 0.95),
    ("math", 0.93),
    ("py", 0.95),
    ("mmlu", 0.94),
];

/// `PLACEMENT.md`'s dynamic row: residency cost per round, 1 and 3 lanes.
const SIMULATED_COST: [f64; 2] = [3.8e-3, 9.1e-3];

/// The traces, read once and shared by every test of this file (they run in
/// parallel); `None` when absent.
fn study() -> Option<&'static Study> {
    static STUDY: std::sync::OnceLock<Option<Study>> = std::sync::OnceLock::new();
    STUDY
        .get_or_init(|| {
            let root = root()?;
            let study = load(&root);
            println!("traces: {} ({} chunks)", root.display(), study.chunks.len());
            Some(study)
        })
        .as_ref()
}

/// Decode tallies at one lane (each chunk alone from the warm start, with
/// hit rates per domain) and at three (consecutive chunks in id order, as
/// the simulation grouped them).
fn decode_replay(
    study: &Study,
    width: usize,
    budgets: [Option<u64>; 2],
) -> (Vec<(String, f64)>, Tally, Tally) {
    let warm = warm_model(study, width, budgets[0]);
    let mut by_domain: Vec<(String, Vec<f64>)> = Vec::new();
    let mut one = Tally::default();
    for chunk in &study.chunks {
        let mut model = warm.clone();
        let mut lane = [Tally::default()];
        decode(&mut model, &[(chunk, 0..DECODE_STEPS.min(chunk.tokens))], width, &mut lane);
        match by_domain.iter_mut().find(|(d, _)| *d == chunk.domain) {
            Some((_, rates)) => rates.push(lane[0].rate()),
            None => by_domain.push((chunk.domain.clone(), vec![lane[0].rate()])),
        }
        one.add(&lane[0]);
    }
    let warm = warm_model(study, width, budgets[1]);
    let mut lanes3 = [Tally::default(); 3];
    for group in study.chunks.chunks(3).filter(|g| g.len() == 3) {
        let mut model = warm.clone();
        let lanes: Vec<_> = group.iter().map(|c| (c, 0..DECODE_STEPS.min(c.tokens))).collect();
        decode(&mut model, &lanes, width, &mut lanes3);
    }
    // A round's bytes and seconds are booked on lane 0 alone.
    let mut three = lanes3[0];
    for lane in &lanes3[1..] {
        three.uses += lane.uses;
        three.hits += lane.hits;
    }
    let rates = by_domain
        .into_iter()
        .map(|(d, r)| {
            let mean = r.iter().sum::<f64>() / r.len() as f64;
            (d, mean)
        })
        .collect();
    (rates, one, three)
}

fn report(label: &str, rates: &[(String, f64)], one: &Tally, three: &Tally) {
    println!("{label}, 1 lane: hit rate by domain (simulation: one byte-LRU, no prefetch)");
    for (domain, simulated) in SIMULATED {
        if let Some((_, mean)) = rates.iter().find(|(d, _)| d == domain) {
            println!("  {domain:6} {:5.1}%  ({:.0}%)", mean * 100.0, simulated * 100.0);
        }
    }
    for (lanes, t, simulated) in [(1, one, SIMULATED_COST[0]), (3, three, SIMULATED_COST[1])] {
        println!(
            "  {lanes} lane(s): hit {:.1}% | demand {:.1} MB + prefetch {:.1} MB per round, {:.2} ms of link at 12 GB/s | demand cost {:.2} ms per round (simulation {:.1} ms)",
            t.rate() * 100.0,
            t.demand_bytes as f64 / t.rounds as f64 / 1e6,
            t.prefetch_bytes as f64 / t.rounds as f64 / 1e6,
            t.moved_per_round() / BANDWIDTH * 1e3,
            t.cost() * 1e3,
            simulated * 1e3
        );
    }
}

/// Acceptance 5's comparison, on the CPU model: without prefetch, as the
/// simulation ran, hit rates per domain within 3 points and the residency
/// cost per round within 1.3x, at one lane and at three.
#[test]
fn decode_on_real_routing_matches_the_simulation() {
    let Some(study) = study() else { return };
    let (rates, one, three) = decode_replay(study, 0, [None; 2]);
    report("W = 0", &rates, &one, &three);
    for (domain, simulated) in SIMULATED {
        let (_, mean) = rates.iter().find(|(d, _)| d == domain).expect("every domain traced");
        assert!(
            (mean - simulated).abs() <= 0.03,
            "{domain}: {mean:.3} against the simulation's {simulated}"
        );
    }
    assert!(one.cost() <= 1.3 * SIMULATED_COST[0], "1 lane: {:.2} ms", one.cost() * 1e3);
    assert!(three.cost() <= 1.3 * SIMULATED_COST[1], "3 lanes: {:.2} ms", three.cost() * 1e3);
}

/// The router lookahead at W = 16 turns most of the remaining misses into
/// prefetch hits. Unbudgeted it moves more bytes than the link carries in a
/// step (reported, not asserted); held to the window budget, the link fits.
/// The cost printed is the demand misses alone, prefetches taken as hidden
/// behind compute: whether they are is a timing question for the GPU replay.
#[test]
fn the_lookahead_turns_most_remaining_misses_into_hits() {
    let Some(study) = study() else { return };
    let (rates, one, three) = decode_replay(study, DEFAULT_PREFETCH_WIDTH, [None; 2]);
    report("W = 16, no budget", &rates, &one, &three);
    let budgets = WINDOW_BUDGET.map(Some);
    let (rates, one, three) = decode_replay(study, DEFAULT_PREFETCH_WIDTH, budgets);
    report("W = 16, window budget", &rates, &one, &three);
    // Without the lookahead the same replay hits ~94.5% at ~3.8 ms a round.
    assert!(one.rate() >= 0.96, "{:.3}", one.rate());
    assert!(one.cost() <= SIMULATED_COST[0] * 0.75, "{:.2} ms", one.cost() * 1e3);
    // Within the budget, the link carries a round's bytes in the time of
    // the simulation's step plus the demand misses' own wait.
    for (t, step) in [(&one, 6e-3), (&three, 7e-3)] {
        assert!(t.moved_per_round() / BANDWIDTH <= step + t.cost(), "{:.2} ms", t.moved_per_round() / BANDWIDTH * 1e3);
    }
}

/// Per chunk (every `every`-th): decode 256 tokens to settle, 256 measured
/// (pre), a 4096-token prefill of the next two chunks, 256 measured (post).
/// Returns the pre and post tallies and the bytes the prefills moved.
fn scan_replay(study: &Study, width: usize, scan: bool, every: usize) -> (Tally, Tally, u64) {
    let budget = (width > 0).then_some(WINDOW_BUDGET[0]);
    let warm = warm_model(study, width, budget);
    let n = study.chunks.len();
    let (mut pre, mut post, mut moved, mut runs) = (Tally::default(), Tally::default(), 0u64, 0u64);
    for (i, a) in study.chunks.iter().enumerate().step_by(every) {
        if a.tokens < 3 * DECODE_STEPS {
            continue;
        }
        let others = [&study.chunks[(i + 1) % n], &study.chunks[(i + 2) % n]];
        let tokens: Vec<(&Chunk, usize)> = others
            .iter()
            .flat_map(|c| (0..c.tokens.min(2048)).map(move |t| (*c, t)))
            .collect();
        let mut model = warm.clone();
        let mut settle = [Tally::default()];
        decode(&mut model, &[(a, 0..DECODE_STEPS)], width, &mut settle);
        decode(&mut model, &[(a, DECODE_STEPS..2 * DECODE_STEPS)], width, std::slice::from_mut(&mut pre));
        moved += prefill(&mut model, &tokens, scan);
        decode(&mut model, &[(a, 2 * DECODE_STEPS..3 * DECODE_STEPS)], width, std::slice::from_mut(&mut post));
        runs += 1;
    }
    println!(
        "W = {width}, {:14} pre {:.1}% ({:.1} MB/token) | post {:.1}% ({:.1} MB/token) | drop {:.1} points | a warm 4096-token prefill moved {:.1} GB ({runs} runs)",
        if scan { "scan-resistant" } else { "plain LRU" },
        pre.rate() * 100.0,
        pre.moved_per_round() / 1e6,
        post.rate() * 100.0,
        post.moved_per_round() / 1e6,
        (pre.rate() - post.rate()) * 100.0,
        moved as f64 / runs as f64 / 1e9
    );
    (pre, post, moved)
}

/// Acceptance 6: decode, a 4096-token prefill of other text, decode again —
/// with the lookahead, the post-prefill hit rate within 2 points of the
/// pre-prefill one.
#[test]
fn a_scan_resistant_prefill_keeps_the_decode_hit_rate() {
    let Some(study) = study() else { return };
    let (pre, post, _) = scan_replay(study, DEFAULT_PREFETCH_WIDTH, true, 1);
    assert!(
        pre.rate() - post.rate() <= 0.02,
        "a prefill costs decode {:.1} points",
        (pre.rate() - post.rate()) * 100.0
    );
}

/// The check discriminates: without the lookahead to hide misses, plain LRU
/// admission of the same prefill evicts the decode working set, which
/// scan-resistant admission keeps (every second chunk).
#[test]
fn plain_lru_admission_of_a_prefill_would_cost_decode_what_scan_resistance_saves() {
    let Some(study) = study() else { return };
    let (scan_pre, scan_post, scan_moved) = scan_replay(study, 0, true, 2);
    let (lru_pre, lru_post, lru_moved) = scan_replay(study, 0, false, 2);
    let scan_drop = scan_pre.rate() - scan_post.rate();
    let lru_drop = lru_pre.rate() - lru_post.rate();
    assert!(
        lru_drop > scan_drop + 0.01,
        "plain LRU {:.1} points against scan-resistant {:.1}",
        lru_drop * 100.0,
        scan_drop * 100.0
    );
    assert!(scan_moved < lru_moved);
}
