//! A live move's PCIe contention, measured on Flash-Next (spec vram-budget/03
//! AC 37, the fixed branch; ADR 0045, GitHub #309).
//!
//! C (`agent`, a 236,000-token prompt: a blob of more than 1 GB) decodes
//! beside two `interactive` lanes, B1 and B2 (short prompts, explicit
//! `max_tokens` of different lengths, `ignore_eos`). An `interactive` arrival
//! E whose reservation does not fit beside them moves C off the device -- on
//! the fixed branch only an admission moves anything -- and when E ends, C
//! comes back while B1 and B2 still decode. Two legs:
//!
//! - **KV-RAM**: an arena that holds C's blob, the synchronous snapshot and
//!   restore;
//! - **KV-disk**: no arena, the windowed spill and read back.
//!
//! Measured for each move: its bytes, duration and GB/s, and B1's and B2's
//! inter-token latency (p50 and max) over the steps it spanned, each against a
//! baseline window of the same number of steps at the same decode width with
//! no transfer in flight, from the same run:
//!
//! - a disk move runs between steps while B1 and B2 decode at width 2 (C,
//!   mid-transfer, is in no round): its baseline is B1' and B2' decoding alone
//!   at width 2 before C arrives;
//! - a KV-RAM move out happens inside the step that also runs E's first
//!   prefill chunk: its baseline is the step that ran the first chunk -- the
//!   same 8,192 tokens -- of E0, an `interactive` arrival that fits beside C
//!   and moves nothing;
//! - a KV-RAM move in happens at the end of the step E ends in: its baseline
//!   is the width-3 decode of C, B1 and B2 before any arrival.
//!
//! Starting thresholds, for the owner to confirm (printed, not asserted): move
//! out ITL p50 within +10 %, move in within +25 %, either one's max within the
//! baseline's max + 150 ms. Asserted: each move happened through its tier, and
//! no work was lost -- every request generates its full `max_tokens`, no
//! `Requeued`, no dropped snapshot, no disk failure. `IGNIS_KV_P3_RAW` names a
//! directory the raw samples are written to, one JSON file a leg.
//!
//! The pool is cut to one context (`--kv-pool-bytes`'s token form, #309 P1)
//! so that E cannot fit beside C. Machine-local: the Flash-Next artifact
//! (`IGNIS_FLASH_NEXT_DIR`), the tier's files under this checkout's
//! `.scratch/kv-disk-gpu/` (or `IGNIS_KV_DISK_TEST_DIR`).

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::ngram_cache::CacheLocation;
use ignis_core::scheduler::DiskSource;
use ignis_core::types::{DecodeParams, FinishReason, RequestClass, RequestId, RequestInput, SchedEvent};
use ignis_core::{gpu_profile, ConcreteScheduler, Scheduler};
use ignis_server::runtime::{flash_next_scheduler_with_ngram_cache, EngineShape};

const FLASH_NEXT_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MODEL: &str = "qwen3.8-flash-next";
const EOS: u32 = 248_044;
const CONTEXT: u32 = 262_144;
const CHUNK: u32 = 8192;
const PAGE_TOKENS: u32 = 64;
/// At least 236K tokens: a blob of at least 1 GB (a 130 MB image and 4,224
/// bytes a token).
const C_PROMPT: u32 = 236_000;
const C_TOKENS: u32 = 2_000;
const B_PROMPT: u32 = 2_000;
const B1_TOKENS: u32 = 1_200;
const B2_TOKENS: u32 = 1_600;
/// The width-2 baseline's lanes.
const B0_TOKENS: u32 = 300;
/// E0 fits beside C, B1 and B2 (8,256 of the 17,344 tokens they leave); its
/// one chunk is E's first.
const E0_PROMPT: u32 = CHUNK;
/// E does not fit (18,064): C has to go.
const E_PROMPT: u32 = 18_000;
const E_TOKENS: u32 = 64;
/// B1 and B2 decode this many tokens at width 3 before any arrival: the
/// KV-RAM move in's baseline.
const WIDTH3_TOKENS: usize = 160;
/// The KV-RAM leg's arena: C's blob is ~1.13 GB. On this host's ~45.9 GB of
/// available RAM the plan takes it only without the n-gram hot rows (37.8 GB
/// of pinned experts, the arena, and the 6 GiB margin), so both legs run
/// without them, as AC 25's harness does: a gather reads its rows from the
/// artifact.
const ARENA_BYTES: u64 = 1200 << 20;

fn prompt(seed: u32, n: u32) -> Vec<u32> {
    (0..n).map(|i| 1000 + (i * 7919 + seed * 104_729) % 60_000).collect()
}

fn input(tokens: Vec<u32>, max_tokens: u32) -> RequestInput {
    RequestInput {
        decision: None,
        model: MODEL.into(),
        tokens,
        params: DecodeParams { max_tokens: Some(max_tokens), ignore_eos: true, ..DecodeParams::default() },
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        constrained: None,
        forced_literal: None,
        warm_up: false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    KvRam,
    KvDisk,
}

/// One advance: its wall time, when it returned, its events, and whether a
/// disk transfer was in flight when it returned.
struct Step {
    wall: Duration,
    at: Instant,
    events: Vec<SchedEvent>,
    busy_after: bool,
}

struct Rig {
    sched: ConcreteScheduler,
    steps: Vec<Step>,
}

impl Rig {
    fn step(&mut self) {
        let started = Instant::now();
        let events = self.sched.advance();
        let wall = started.elapsed();
        if let Some(error) = self.sched.last_error() {
            panic!("the leaf failed a step: {error}");
        }
        let busy_after = self.sched.disk_busy();
        self.steps.push(Step { wall, at: Instant::now(), events, busy_after });
    }

    fn to_idle(&mut self) {
        while !self.sched.is_idle() {
            self.step();
        }
    }

    fn mark(&self) -> usize {
        self.steps.len()
    }

    fn events(&self) -> impl Iterator<Item = &SchedEvent> {
        self.steps.iter().flat_map(|s| &s.events)
    }

    fn tokens_of(&self, request: RequestId) -> usize {
        self.events().filter(|e| matches!(e, SchedEvent::Token { request: r, .. } if *r == request)).count()
    }

    fn until(&mut self, done: impl Fn(&Self) -> bool) {
        while !done(self) {
            assert!(!self.sched.is_idle(), "the run went idle first");
            self.step();
        }
    }

    fn until_tokens(&mut self, request: RequestId, n: usize) {
        self.until(|rig| rig.tokens_of(request) >= n);
    }

    /// The first step at or after `from` with an event `f` holds of.
    fn find(&self, from: usize, f: impl Fn(&SchedEvent) -> bool) -> Option<usize> {
        self.steps[from..].iter().position(|s| s.events.iter().any(&f)).map(|i| from + i)
    }

    fn has(&self, f: impl Fn(&SchedEvent) -> bool) -> bool {
        self.events().any(f)
    }

    fn finished_at_length(&self, request: RequestId) -> bool {
        self.has(|e| matches!(e, SchedEvent::Done { request: r, reason: FinishReason::Length, .. } if *r == request))
    }

    /// The gaps between consecutive tokens of `requests` that land in the
    /// steps `first..=last`; a gap belongs to the step its later token lands
    /// in, so a step's own work is inside it.
    fn itl(&self, requests: &[RequestId], first: usize, last: usize) -> Vec<Duration> {
        let mut previous: HashMap<RequestId, Instant> = HashMap::new();
        let mut gaps = Vec::new();
        for (i, step) in self.steps.iter().enumerate().take(last + 1) {
            for event in &step.events {
                if let SchedEvent::Token { request, .. } = event {
                    if requests.contains(request) {
                        if let Some(before) = previous.insert(*request, step.at) {
                            if i >= first {
                                gaps.push(step.at - before);
                            }
                        }
                    }
                }
            }
        }
        gaps
    }

    /// The steps in `from..to` that ran no prefill chunk and had no transfer
    /// in flight: steady decode.
    fn steady(&self, from: usize, to: usize) -> Vec<usize> {
        (from..to)
            .filter(|&i| {
                let s = &self.steps[i];
                !s.events.iter().any(|e| matches!(e, SchedEvent::PrefillChunk { .. })) && !s.busy_after
            })
            .collect()
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// p50 and max of `gaps`, in ms.
fn stats(gaps: &[Duration]) -> (f64, f64) {
    let mut v: Vec<f64> = gaps.iter().map(|d| ms(*d)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    (v[(v.len() - 1) / 2], v[v.len() - 1])
}

/// What one move measured.
struct Move {
    name: &'static str,
    bytes: u64,
    duration: Duration,
    steps: usize,
    itl: Vec<Duration>,
    baseline: Vec<Duration>,
    baseline_label: &'static str,
    /// The width-2 baseline taken again at the end of the run, for a disk
    /// move.
    late_baseline: Option<Vec<Duration>>,
    p50_bound: f64,
}

impl Move {
    fn gbps(&self) -> f64 {
        self.bytes as f64 / self.duration.as_secs_f64() / 1e9
    }

    fn print(&self, leg: Tier) {
        let (p50, max) = stats(&self.itl);
        let (b50, bmax) = stats(&self.baseline);
        let p50_ok = p50 <= b50 * (1.0 + self.p50_bound);
        let max_ok = max <= bmax + 150.0;
        println!(
            "{leg:?} {}: {} bytes in {:.1} ms ({:.2} GB/s) over {} step(s); ITL p50 {p50:.2} ms vs {b50:.2} ms ({:+.1} %, bound +{:.0} %: {}), \
             max {max:.2} ms vs {bmax:.2} ms ({:+.1} ms, bound +150 ms: {}); {} gaps against {} ({})",
            self.name,
            self.bytes,
            ms(self.duration),
            self.gbps(),
            self.steps,
            (p50 / b50 - 1.0) * 100.0,
            self.p50_bound * 100.0,
            if p50_ok { "within" } else { "OVER" },
            max - bmax,
            if max_ok { "within" } else { "OVER" },
            self.itl.len(),
            self.baseline.len(),
            self.baseline_label,
        );
        if let Some(late) = &self.late_baseline {
            let (l50, lmax) = stats(late);
            println!(
                "{leg:?} {}: against the width-2 baseline taken last, ITL p50 {:+.1} % (vs {l50:.2} ms), max {:+.1} ms (vs {lmax:.2} ms)",
                self.name,
                (p50 / l50 - 1.0) * 100.0,
                max - lmax,
            );
        }
    }

    fn json(&self) -> String {
        let list = |v: &[Duration]| v.iter().map(|d| format!("{:.3}", ms(*d))).collect::<Vec<_>>().join(",");
        format!(
            "{{\"move\":\"{}\",\"bytes\":{},\"duration_ms\":{:.3},\"gb_per_s\":{:.3},\"steps\":{},\"itl_ms\":[{}],\"baseline\":\"{}\",\"baseline_itl_ms\":[{}],\"late_baseline_itl_ms\":[{}]}}",
            self.name,
            self.bytes,
            ms(self.duration),
            self.gbps(),
            self.steps,
            list(&self.itl),
            self.baseline_label,
            list(&self.baseline),
            self.late_baseline.as_deref().map_or_else(String::new, list)
        )
    }
}

/// `n` consecutive steady steps out of `steady`, taken from its middle: a
/// baseline window as long as the move's.
fn window(steady: &[usize], n: usize) -> (usize, usize) {
    let n = n.max(1);
    assert!(steady.len() >= n, "the baseline has {} steady steps, the move {n}", steady.len());
    let start = (steady.len() - n) / 2;
    (steady[start], steady[start + n - 1])
}

/// The leg's blob directory, emptied before the leg and when it drops, a
/// failed leg included: a spilled blob is more than a gigabyte.
struct BlobDir(PathBuf);

impl Drop for BlobDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn leg(tier: Tier) -> Option<Vec<Move>> {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(FLASH_NEXT_DIR), PathBuf::from);
    let path = dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display())) {
        return None;
    }
    let blobs = BlobDir(
        std::env::var_os("IGNIS_KV_DISK_TEST_DIR")
            .map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.scratch/kv-disk-gpu"), PathBuf::from)
            .join(format!("contention-{tier:?}")),
    );
    let _ = std::fs::remove_dir_all(&blobs.0);
    std::fs::create_dir_all(&blobs.0).unwrap();
    let (host_pool_bytes, kv_disk_bytes) = match tier {
        Tier::KvRam => (ARENA_BYTES, 0),
        Tier::KvDisk => (0, 8 << 30),
    };
    let shape = EngineShape {
        max_context: CONTEXT,
        prefill_chunk: CHUNK,
        // Four in flight: C, B1, B2 and the arrival.
        decode_lanes: 4,
        host_pool_bytes,
        prompt_reuse: false,
        retained_device_slots: 0,
        retained_host_slots: 0,
        retained_host_named: true,
        kv_disk_bytes: Some(kv_disk_bytes),
        kv_pool: Some(ignis_core::KvPoolSize::Tokens(u64::from(CONTEXT))),
        ngram_hot_bytes: ignis_core::ngram_table::HotBudget::Bytes(0),
        ..EngineShape::default()
    };
    let (sched, reserved) = match flash_next_scheduler_with_ngram_cache(
        &path,
        MODEL.into(),
        EOS,
        shape,
        None,
        ignis_core::ngram_cache::PersistenceOptions { enabled: false, ..Default::default() },
        &CacheLocation::Directory(blobs.0.clone()),
    ) {
        Ok(loaded) => loaded,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next scheduler for the {tier:?} leg: {e}"));
            return None;
        }
    };
    let pool = reserved.kv_pool_pages * PAGE_TOKENS;
    let beside = (C_PROMPT + C_TOKENS) + (B_PROMPT + B1_TOKENS) + (B_PROMPT + B2_TOKENS);
    if pool < beside + E0_PROMPT + E_TOKENS || pool >= beside + E_PROMPT + E_TOKENS {
        gpu_profile::skip_or_fail(&format!(
            "the pool holds {pool} tokens: E0 must fit beside C, B1 and B2 ({beside}) and E must not"
        ));
        return None;
    }
    let mut rig = Rig { sched, steps: Vec::new() };
    let interactive = RequestClass::Interactive;

    // ── the width-2 baseline: two lanes alone ──────────────────────────────
    let b0 = [
        rig.sched.submit(input(prompt(20, B_PROMPT), B0_TOKENS), interactive).unwrap(),
        rig.sched.submit(input(prompt(21, B_PROMPT), B0_TOKENS), interactive).unwrap(),
    ];
    let from = rig.mark();
    rig.to_idle();
    let width2 = rig.steady(from, rig.mark());

    // ── C, then B1 and B2 beside it ────────────────────────────────────────
    let c = rig.sched.submit(input(prompt(1, C_PROMPT), C_TOKENS), RequestClass::Agent).unwrap();
    rig.until_tokens(c, 4);
    let b = [
        rig.sched.submit(input(prompt(2, B_PROMPT), B1_TOKENS), interactive).unwrap(),
        rig.sched.submit(input(prompt(3, B_PROMPT), B2_TOKENS), interactive).unwrap(),
    ];
    rig.until_tokens(b[0], 16);
    rig.until_tokens(b[1], 16);
    let from = rig.mark();
    rig.until_tokens(b[0], 16 + WIDTH3_TOKENS);
    let width3 = rig.steady(from, rig.mark());

    // ── E0: an arrival that fits, its first chunk the baseline of a move in a
    //    chunk's step ───────────────────────────────────────────────────────
    let from = rig.mark();
    let e0 = rig.sched.submit(input(prompt(4, E0_PROMPT), E_TOKENS), interactive).unwrap();
    rig.until(|rig| rig.finished_at_length(e0));
    let e0_chunk = rig
        .find(from, |e| matches!(e, SchedEvent::PrefillChunk { request, .. } if *request == e0))
        .expect("E0 prefilled");
    assert!(
        !rig.has(|e| matches!(e, SchedEvent::Evicted { .. } | SchedEvent::DiskSpilled { .. })),
        "E0 fits beside C: nothing moved for it"
    );

    // ── E: C out ───────────────────────────────────────────────────────────
    let from = rig.mark();
    let e = rig.sched.submit(input(prompt(5, E_PROMPT), E_TOKENS), interactive).unwrap();
    let mut moves = Vec::new();
    match tier {
        Tier::KvRam => {
            rig.until(|rig| rig.has(|ev| matches!(ev, SchedEvent::Evicted { request, .. } if *request == c)));
            let at = rig.find(from, |ev| matches!(ev, SchedEvent::Evicted { request, .. } if *request == c)).unwrap();
            let micros = rig.steps[at]
                .events
                .iter()
                .find_map(|ev| match ev {
                    SchedEvent::Evicted { request, snapshot_micros } if *request == c => Some(*snapshot_micros),
                    _ => None,
                })
                .unwrap();
            assert!(
                rig.steps[at].events.iter().any(|ev| matches!(ev, SchedEvent::PrefillChunk { request, .. } if *request == e)),
                "the step that moved C ran E's first chunk"
            );
            let bytes = rig.sched.host_tier().entry(c).expect("C's snapshot is in KV-RAM").bytes;
            moves.push(Move {
                name: "move out (device -> KV-RAM)",
                bytes,
                duration: Duration::from_micros(micros),
                steps: 1,
                itl: rig.itl(&b, at, at),
                baseline: rig.itl(&b, e0_chunk, e0_chunk),
                baseline_label: "E0's first-chunk step, no move",
                late_baseline: None,
                p50_bound: 0.10,
            });
        }
        Tier::KvDisk => {
            rig.until(|rig| {
                rig.has(|ev| matches!(ev, SchedEvent::DiskSpilled { request, from: DiskSource::Device } if *request == c))
            });
            let start = (from..rig.mark()).find(|&i| rig.steps[i].busy_after).expect("the spill started");
            let end = rig
                .find(from, |ev| matches!(ev, SchedEvent::DiskSpilled { request, .. } if *request == c))
                .unwrap();
            let bytes = rig.sched.disk_tier().expect("the tier").used_bytes();
            let started_at = rig.steps[start].at - rig.steps[start].wall;
            // The steps after the one that started it, up to the one before
            // its commit was seen: that one also runs E's first chunk, whose
            // seconds are E's, not the move's.
            let n = (end - 1 - start).max(1);
            let (b_first, b_last) = window(&width2, n);
            moves.push(Move {
                name: "move out (device -> KV-disk)",
                bytes,
                duration: rig.steps[end].at - rig.steps[end].wall - started_at,
                steps: n,
                itl: rig.itl(&b, start + 1, start + n),
                baseline: rig.itl(&b0, b_first, b_last),
                baseline_label: "B1' and B2' alone at width 2",
                late_baseline: None,
                p50_bound: 0.10,
            });
        }
    }

    // ── E ends: C in ───────────────────────────────────────────────────────
    rig.until(|rig| rig.finished_at_length(e));
    let e_done = rig.find(from, |ev| matches!(ev, SchedEvent::Done { request, .. } if *request == e)).unwrap();
    rig.until(|rig| rig.has(|ev| matches!(ev, SchedEvent::Restored { request, .. } if *request == c)));
    let restored = rig.find(e_done, |ev| matches!(ev, SchedEvent::Restored { request, .. } if *request == c)).unwrap();
    let still_decoding = b.iter().filter(|&&r| !rig.finished_at_length(r)).count();
    assert!(still_decoding >= 1, "at least one of B1 and B2 still decodes when C comes back");
    match tier {
        Tier::KvRam => {
            let micros = rig.steps[restored]
                .events
                .iter()
                .find_map(|ev| match ev {
                    SchedEvent::Restored { request, restore_micros, .. } if *request == c => Some(*restore_micros),
                    _ => None,
                })
                .unwrap();
            let (b_first, b_last) = window(&width3, 1);
            moves.push(Move {
                name: "move in (KV-RAM -> device)",
                bytes: moves[0].bytes,
                duration: Duration::from_micros(micros),
                steps: 1,
                itl: rig.itl(&b, restored, restored),
                baseline: rig.itl(&b, b_first, b_last),
                baseline_label: "C, B1 and B2 at width 3",
                late_baseline: None,
                p50_bound: 0.25,
            });
        }
        Tier::KvDisk => {
            // Started at the end of the step E ended in, seen landed at the
            // top of the step that restored C.
            let start = (e_done..=restored).find(|&i| rig.steps[i].busy_after).expect("the restore started");
            let n = (restored - start).max(1);
            let (b_first, b_last) = window(&width2, n);
            moves.push(Move {
                name: "move in (KV-disk -> device)",
                bytes: moves[0].bytes,
                duration: rig.steps[restored].at - rig.steps[restored].wall - rig.steps[start].at,
                steps: n,
                itl: rig.itl(&b, start + 1, start + n),
                baseline: rig.itl(&b0, b_first, b_last),
                baseline_label: "B1' and B2' alone at width 2",
                late_baseline: None,
                p50_bound: 0.25,
            });
        }
    }
    rig.to_idle();

    // ── the width-2 baseline again, last: whether the first one, taken
    //    before C's prefill, stands for the card's state during the moves ──
    let b9 = [
        rig.sched.submit(input(prompt(22, B_PROMPT), B0_TOKENS), interactive).unwrap(),
        rig.sched.submit(input(prompt(23, B_PROMPT), B0_TOKENS), interactive).unwrap(),
    ];
    let from = rig.mark();
    rig.to_idle();
    let late = rig.steady(from, rig.mark());
    let late_itl = rig.itl(&b9, late[0], *late.last().unwrap());
    let (late50, late_max) = stats(&late_itl);
    let early_itl = rig.itl(&b0, width2[0], *width2.last().unwrap());
    let (early50, early_max) = stats(&early_itl);
    println!(
        "{tier:?} width-2 baselines: first ITL p50 {early50:.2} ms max {early_max:.2} ms ({} gaps), \
         last p50 {late50:.2} ms max {late_max:.2} ms ({} gaps)",
        early_itl.len(),
        late_itl.len()
    );
    for m in moves.iter_mut().filter(|m| m.baseline_label.contains("width 2")) {
        m.late_baseline = Some(late_itl.clone());
    }

    // ── no work lost ───────────────────────────────────────────────────────
    for (r, n) in [(c, C_TOKENS), (b[0], B1_TOKENS), (b[1], B2_TOKENS), (e0, E_TOKENS), (e, E_TOKENS), (b9[0], B0_TOKENS)] {
        assert!(rig.finished_at_length(r), "{r} generated its full max_tokens");
        assert_eq!(rig.tokens_of(r), n as usize);
    }
    assert!(!rig.has(|ev| matches!(ev, SchedEvent::Requeued { .. })), "no Requeued");
    assert!(!rig.has(|ev| matches!(ev, SchedEvent::SnapshotDropped { .. })), "no live snapshot dropped");
    assert!(!rig.has(|ev| matches!(ev, SchedEvent::DiskFailure { .. })), "no disk failure");
    let outs = rig.events().filter(|ev| matches!(ev, SchedEvent::Evicted { .. } | SchedEvent::DiskSpilled { .. })).count();
    assert_eq!(outs, 1, "C moved out once, and nothing else moved");
    drop(rig);
    drop(blobs);
    Some(moves)
}

fn measure(tier: Tier) {
    let Some(moves) = leg(tier) else {
        return;
    };
    for m in &moves {
        m.print(tier);
    }
    if let Some(dir) = std::env::var_os("IGNIS_KV_P3_RAW").map(PathBuf::from) {
        std::fs::create_dir_all(&dir).expect("the raw samples' directory");
        let json = format!("[{}]\n", moves.iter().map(Move::json).collect::<Vec<_>>().join(","));
        std::fs::write(dir.join(format!("ac37-{tier:?}.json")), json).expect("write the raw samples");
    }
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact, minutes"]
fn a_live_move_through_kv_ram_beside_decoding_lanes() {
    measure(Tier::KvRam);
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact, minutes"]
fn a_live_move_through_kv_disk_beside_decoding_lanes() {
    measure(Tier::KvDisk);
}
