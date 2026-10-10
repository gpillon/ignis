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
//! - **KV-RAM**: an arena that holds C's blob, moved a window at a time
//!   between steps (GitHub #309; synchronous before it);
//! - **KV-disk**: no arena, the windowed spill and read back.
//!
//! Measured for each move: its bytes, duration and GB/s, and B1's and B2's
//! inter-token latency (p50 and max) over the steps its windows were on the
//! link -- B1 and B2 decoding at width 2, C mid-move in no round, E waiting
//! for the room or ended. The baseline (GitHub #309, the same for all four
//! moves): **B1' and B2', two fresh `interactive` lanes decoding alone at
//! width 2 after everything else has ended**, a window of as many steady
//! steps as the move's from the middle of their run -- the card's state after
//! C's 236K-token prefill, which a baseline taken before it does not show
//! (the 2026-10-08 finding). The width-2 pair taken before C arrives is
//! printed beside it.
//!
//! Starting thresholds, for the owner to confirm (printed, not asserted): move
//! out ITL p50 within +10 %; move in within +25 %, read since GitHub #310 on a
//! step's time outside its expert stall and transfer passes -- what the
//! transfer adds (owner, 2026-10-08), the raw ITL printed beside it; either
//! one's max within the baseline's max + 150 ms. Asserted: each move happened through its tier, and
//! no work was lost -- every request generates its full `max_tokens`, no
//! `Requeued`, no dropped snapshot, no disk failure. `IGNIS_KV_P3_RAW` names a
//! directory the raw samples are written to, one JSON file a leg.
//! `IGNIS_KV_MOVE_PACE=<in MiB>,<out MiB>` runs the load at another pace than
//! `ignis_runtime::TransferPace::default()`.
//!
//! The pool is cut to one context (`--vram-kv-pool-bytes`'s token form, #309 P1)
//! so that E cannot fit beside C. Machine-local: the Flash-Next artifact
//! (`IGNIS_FLASH_NEXT_DIR`), the tier's files under this checkout's
//! `.scratch/kv-disk-gpu/` (or `IGNIS_KV_DISK_TEST_DIR`).
//!
//! The text decides the expert misses (GitHub #310): a round's misses are
//! those of what its lanes generate, and greedy decode is not
//! batch-invariant, so two runs whose batches differ -- a moved run and the
//! control -- generate different text once they part. With `IGNIS_KV_P3_RAW`
//! each run also writes what every request generated, by its prompt's seed
//! (`ac37-<leg>-tokens.json`), and `IGNIS_AC37_FORCE=<such a file>` makes
//! every request generate exactly that (a forced literal, one token a
//! round): the same text whatever moves. A forced round is never a captured
//! graph, so a forced run's ITL is not comparable with a free run's; its
//! misses are. AC 43's test forces both of its runs to a committed text
//! (`tests/fixtures/ac37_text.json`). `IGNIS_AC37_NO_ARRIVALS=1` runs the
//! control without E0 and E.

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
/// Long enough that both still decode through C's move back in.
const B1_TOKENS: u32 = 2_000;
const B2_TOKENS: u32 = 2_400;
/// The width-2 baselines' lanes.
const B0_TOKENS: u32 = 300;
/// The baseline taken last runs longer: the moves' windows are drawn from its
/// middle.
const B9_TOKENS: u32 = 800;
/// E0 fits beside C, B1 and B2 (8,256 of the 15,744 tokens they leave).
const E0_PROMPT: u32 = CHUNK;
/// E does not fit (18,064): C has to go.
const E_PROMPT: u32 = 18_000;
const E_TOKENS: u32 = 64;
/// B1 and B2 decode this many tokens beside C before any arrival.
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

fn input(tokens: Vec<u32>, max_tokens: u32, forced: Option<Vec<u32>>) -> RequestInput {
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
        forced_literal: forced.map(|tokens| {
            std::sync::Arc::new(ignis_core::forced_literal::ForcedLiteral::at_generation(tokens).expect("a forced text"))
        }),
        warm_up: false,
    }
}

/// What every request generates, by its prompt's seed (GitHub #310).
type Texts = std::collections::BTreeMap<u32, Vec<u32>>;

/// `{seed: [tokens]}`, as [`Rig::write_texts`] writes it.
fn read_texts(path: &Path) -> Texts {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let by_seed: HashMap<String, Vec<u32>> = serde_json::from_str(&text).expect("a text file is {seed: [tokens]}");
    by_seed.into_iter().map(|(seed, tokens)| (seed.parse().expect("a seed"), tokens)).collect()
}

/// What `IGNIS_AC37_FORCE` names every request to generate: a run's own
/// `ac37-<leg>-tokens.json`. None named, the requests generate freely.
fn forced_from_env() -> Texts {
    std::env::var_os("IGNIS_AC37_FORCE").map_or_else(Texts::new, |path| read_texts(Path::new(&path)))
}

/// The text AC 43 forces both of its runs to (GitHub #310): what the KV-RAM
/// leg generated freely on 2026-10-08. Any fixed text serves -- the two runs
/// only have to generate the same one.
const AC43_TEXT: &str = "tests/fixtures/ac37_text.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    KvRam,
    KvDisk,
}

/// One advance: its wall time, when it returned, its events, and whether a
/// move was in flight when it returned.
struct Step {
    wall: Duration,
    at: Instant,
    events: Vec<SchedEvent>,
    busy_after: bool,
    /// The model thread's time in the advance's transfer passes.
    pump: Duration,
    /// The load's expert residency and n-gram counts after the step.
    counters: Option<ignis_core::flash_next_counters::FlashNextCounters>,
}

struct Rig {
    sched: ConcreteScheduler,
    steps: Vec<Step>,
    counters: Option<std::sync::Arc<ignis_core::flash_next_counters::FlashNextCounterSource>>,
    /// Each request's prompt seed: what names it across runs, whose request
    /// ids differ (GitHub #310).
    seeds: HashMap<RequestId, u32>,
    /// What each seed is made to generate, if anything.
    forced: Texts,
}

impl Rig {
    fn new(
        sched: ConcreteScheduler,
        counters: Option<std::sync::Arc<ignis_core::flash_next_counters::FlashNextCounterSource>>,
        forced: Texts,
    ) -> Self {
        Self { sched, steps: Vec::new(), counters, seeds: HashMap::new(), forced }
    }

    /// Submit the prompt of `seed`, `prompt_tokens` long, generating
    /// `max_tokens` -- the forced text of `seed` when one is named: as much
    /// of it as there is, the rest generated freely.
    fn submit(&mut self, seed: u32, prompt_tokens: u32, max_tokens: u32, class: RequestClass) -> RequestId {
        let forced = self
            .forced
            .get(&seed)
            .map(|tokens| tokens[..tokens.len().min(max_tokens as usize)].to_vec());
        let id = self.sched.submit(input(prompt(seed, prompt_tokens), max_tokens, forced), class).unwrap();
        self.seeds.insert(id, seed);
        id
    }

    /// What every request generated, by its prompt's seed.
    fn texts(&self) -> Texts {
        let mut by_seed = Texts::new();
        for event in self.events() {
            if let SchedEvent::Token { request, token } = event {
                by_seed.entry(self.seeds[request]).or_default().push(*token);
            }
        }
        by_seed
    }

    /// [`Self::texts`] as JSON: what `IGNIS_AC37_FORCE` reads (GitHub #310).
    fn write_texts(&self, path: &Path) {
        let rows: Vec<String> = self
            .texts()
            .iter()
            .map(|(seed, tokens)| format!("\"{seed}\":{tokens:?}"))
            .collect();
        std::fs::write(path, format!("{{{}}}\n", rows.join(",\n"))).expect("write the generated texts");
    }

    /// Step `i`'s decode expert misses and stall: differences of the load's
    /// totals, `None` without the counters.
    fn residency(&self, i: usize) -> Option<(u64, Duration)> {
        let totals = |c: &ignis_core::flash_next_counters::FlashNextCounters| {
            (c.residency.misses.iter().map(|p| p[0]).sum::<u64>(), c.residency.stall_nanos[0])
        };
        let now = totals(self.steps[i].counters.as_ref()?);
        let before = if i == 0 { (0, 0) } else { totals(self.steps[i - 1].counters.as_ref()?) };
        Some((now.0 - before.0, Duration::from_nanos(now.1 - before.1)))
    }

    /// Step `i`'s wall time outside its decode expert stall and its transfer
    /// passes: what a move's copies could add to a round (GitHub #309), and
    /// what AC 37's move-in bound reads since #310.
    fn outside_stall(&self, i: usize) -> Duration {
        let stall = self.residency(i).map_or(Duration::ZERO, |(_, stall)| stall);
        self.steps[i].wall.saturating_sub(stall).saturating_sub(self.steps[i].pump)
    }

    /// The steady steps of `from..to` in which `three` decoded and nothing
    /// else did, each with how many tokens `b1` had generated by its end and its decode
    /// expert misses (GitHub #310): the rounds AC 43 compares, by B1's token.
    fn rounds(&self, three: &[RequestId], b1: RequestId, from: usize, to: usize) -> Vec<Round> {
        let mut b1_tokens = 0;
        let mut rounds = Vec::new();
        for i in 0..to {
            b1_tokens += self.steps[i]
                .events
                .iter()
                .filter(|e| matches!(e, SchedEvent::Token { request, .. } if *request == b1))
                .count();
            let steady = !self.steps[i].events.iter().any(|e| matches!(e, SchedEvent::PrefillChunk { .. }))
                && !self.steps[i].busy_after;
            let decoded = self.steps[i].events.iter().filter(|e| matches!(e, SchedEvent::Token { .. })).count();
            if i >= from && steady && decoded == three.len() && self.all_decoded(three, i) {
                if let Some((misses, _)) = self.residency(i) {
                    rounds.push(Round { b1_tokens, misses });
                }
            }
        }
        rounds
    }

    /// Every step as one JSON row -- when it ended (Unix ms), its wall time,
    /// the transfer passes' time, how many requests decoded in it, whether it
    /// ran a prefill chunk or had a move in flight, and its move and end
    /// facts -- to set beside a GPU sampler's log (GitHub #309).
    fn write_timeline(&self, path: &Path) {
        let (now_instant, now_unix) = (Instant::now(), std::time::SystemTime::now());
        let unix_ms = |at: Instant| {
            let ago = now_instant - at;
            (now_unix - ago).duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64() * 1e3)
        };
        // Decode's expert hits, misses and demand-copy stall, and the
        // n-gram file reads, in the step: each a difference of totals.
        let decode = |c: &ignis_core::flash_next_counters::FlashNextCounters| {
            let r = &c.residency;
            (
                r.hits.iter().map(|p| p[0]).sum::<u64>(),
                r.misses.iter().map(|p| p[0]).sum::<u64>(),
                r.stall_nanos[0],
                c.ngram.file_rows,
            )
        };
        let mut previous = (0, 0, 0, 0);
        let rows: Vec<String> = self
            .steps
            .iter()
            .map(|s| {
                let now = s.counters.as_ref().map_or(previous, decode);
                let (hits, misses, stall, ngram) =
                    (now.0 - previous.0, now.1 - previous.1, now.2 - previous.2, now.3 - previous.3);
                previous = now;
                let decoded: std::collections::BTreeSet<RequestId> = s
                    .events
                    .iter()
                    .filter_map(|e| match e {
                        SchedEvent::Token { request, .. } => Some(*request),
                        _ => None,
                    })
                    .collect();
                // GitHub #310: the step's tokens, by prompt seed.
                let tokens: Vec<[u32; 2]> = s
                    .events
                    .iter()
                    .filter_map(|e| match e {
                        SchedEvent::Token { request, token } => Some([self.seeds[request], *token]),
                        _ => None,
                    })
                    .collect();
                let facts: Vec<String> = s
                    .events
                    .iter()
                    .filter_map(|e| match e {
                        SchedEvent::Evicted { request, .. } => Some(format!("evicted {request}")),
                        SchedEvent::Restored { request, .. } => Some(format!("restored {request}")),
                        SchedEvent::DiskSpilled { request, .. } => Some(format!("spilled {request}")),
                        SchedEvent::Done { request, .. } => Some(format!("done {request}")),
                        _ => None,
                    })
                    .collect();
                // The cache's occupancy and the prefetches' totals, as they
                // stand after the step.
                let (in_use, capacity, prefetch_issued, prefetch_used) =
                    s.counters.as_ref().map_or((0, 0, 0, 0), |c| {
                        (
                            c.slots_in_use.iter().map(|&n| u64::from(n)).sum::<u64>(),
                            c.slots_capacity.iter().map(|&n| u64::from(n)).sum::<u64>(),
                            c.residency.prefetch_issued,
                            c.residency.prefetch_used,
                        )
                    });
                format!(
                    "{{\"unix_ms\":{:.1},\"wall_ms\":{:.3},\"pump_ms\":{:.3},\"decoded\":{:?},\"prefill\":{},\"busy\":{},\"facts\":{:?},\"decode_hits\":{hits},\"decode_misses\":{misses},\"decode_stall_ms\":{:.3},\"ngram_file_rows\":{ngram},\"slots_in_use\":{in_use},\"slots_capacity\":{capacity},\"prefetch_issued\":{prefetch_issued},\"prefetch_used\":{prefetch_used},\"tokens\":{tokens:?}}}",
                    unix_ms(s.at),
                    ms(s.wall),
                    ms(s.pump),
                    decoded.iter().collect::<Vec<_>>(),
                    s.events.iter().any(|e| matches!(e, SchedEvent::PrefillChunk { .. })),
                    s.busy_after,
                    facts,
                    stall as f64 / 1e6
                )
            })
            .collect();
        std::fs::write(path, format!("[\n{}\n]\n", rows.join(",\n"))).expect("write the step timeline");
    }

    fn step(&mut self) {
        let started = Instant::now();
        let events = self.sched.advance();
        let wall = started.elapsed();
        if let Some(error) = self.sched.last_error() {
            panic!("the leaf failed a step: {error}");
        }
        let busy_after = self.sched.transfer_busy();
        let pump = Duration::from_micros(self.sched.transfer_pass_micros());
        let counters = self.counters.as_ref().map(|source| source.read());
        self.steps.push(Step { wall, at: Instant::now(), events, busy_after, pump, counters });
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

    /// The gaps of `requests` whose later token lands in one of `steps`.
    fn itl_in(&self, requests: &[RequestId], steps: &[usize]) -> Vec<Duration> {
        steps.iter().flat_map(|&i| self.itl(requests, i, i)).collect()
    }

    /// Whether every one of `requests` decoded in step `i`: a round of that
    /// width at least.
    fn all_decoded(&self, requests: &[RequestId], i: usize) -> bool {
        requests
            .iter()
            .all(|r| self.steps[i].events.iter().any(|e| matches!(e, SchedEvent::Token { request, .. } if request == r)))
    }

    /// The span a KV-RAM move of `request` reported landing in step `at`,
    /// start to landing; a disk move's is read off the steps it spanned.
    fn kv_ram_span(&self, tier: Tier, request: RequestId, at: usize) -> Option<Duration> {
        if tier != Tier::KvRam {
            return None;
        }
        self.steps[at].events.iter().find_map(|ev| match ev {
            SchedEvent::Evicted { request: r, snapshot_micros } if *r == request => Some(Duration::from_micros(*snapshot_micros)),
            SchedEvent::Restored { request: r, restore_micros, .. } if *r == request => {
                Some(Duration::from_micros(*restore_micros))
            }
            _ => None,
        })
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
    /// B1' and B2' alone at width 2 after everything else ended, as many
    /// steps as the move's: the baseline.
    baseline: Vec<Duration>,
    /// The same pair's window before C arrived, for reference.
    early_baseline: Vec<Duration>,
    /// The model thread's time in the transfer passes, each of the move's
    /// steps: the host side of what the move costs a round.
    pump: Vec<Duration>,
    /// Each of the move's steps outside its expert stall and its transfer
    /// passes ([`Rig::outside_stall`]), and the baseline's steps alike.
    outside: Vec<Duration>,
    baseline_outside: Vec<Duration>,
    /// The expert stall a decode miss cost over the move's steps and over
    /// the baseline's, in µs: where a transfer sharing the link with the
    /// expert copies would show, which `outside` leaves out.
    stall_per_miss: (f64, f64),
    /// What the p50 bound reads (GitHub #310): the ITL, or -- for a move in,
    /// whose rounds' expert misses are the text's (AC 43) -- the step time
    /// outside the expert stall, which is what the transfer adds.
    bound_on: BoundOn,
    p50_bound: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundOn {
    Itl,
    OutsideStall,
}

impl Move {
    fn gbps(&self) -> f64 {
        self.bytes as f64 / self.duration.as_secs_f64() / 1e9
    }

    fn print(&self, leg: Tier) {
        let (p50, max) = stats(&self.itl);
        let (b50, bmax) = stats(&self.baseline);
        let (e50, emax) = stats(&self.early_baseline);
        let (o50, _) = stats(&self.outside);
        let (bo50, _) = stats(&self.baseline_outside);
        let verdict = |ok: bool, on: BoundOn| match (on == self.bound_on, ok) {
            (false, _) => format!("bound on {:?}", self.bound_on),
            (true, true) => format!("bound +{:.0} %: within", self.p50_bound * 100.0),
            (true, false) => format!("bound +{:.0} %: OVER", self.p50_bound * 100.0),
        };
        let max_ok = max <= bmax + 150.0;
        println!(
            "{leg:?} {}: {} bytes in {:.1} ms ({:.2} GB/s) over {} step(s); ITL p50 {p50:.2} ms vs {b50:.2} ms ({:+.1} %, {}), \
             max {max:.2} ms vs {bmax:.2} ms ({:+.1} ms, bound +150 ms: {}); {} gaps against {} (B1' and B2' at width 2, taken last); \
             against the pair taken first: p50 {:+.1} % (vs {e50:.2} ms), max {:+.1} ms (vs {emax:.2} ms)",
            self.name,
            self.bytes,
            ms(self.duration),
            self.gbps(),
            self.steps,
            (p50 / b50 - 1.0) * 100.0,
            verdict(p50 <= b50 * (1.0 + self.p50_bound), BoundOn::Itl),
            max - bmax,
            if max_ok { "within" } else { "OVER" },
            self.itl.len(),
            self.baseline.len(),
            (p50 / e50 - 1.0) * 100.0,
            max - emax,
        );
        println!(
            "{leg:?} {}: a step outside its expert stall and transfer passes, p50 {o50:.2} ms vs {bo50:.2} ms ({:+.1} %, {}); \
             the expert stall a miss cost: {:.1} µs vs {:.1} µs",
            self.name,
            (o50 / bo50 - 1.0) * 100.0,
            verdict(o50 <= bo50 * (1.0 + self.p50_bound), BoundOn::OutsideStall),
            self.stall_per_miss.0,
            self.stall_per_miss.1,
        );
        let (pump50, pump_max) = stats(&self.pump);
        println!(
            "{leg:?} {}: the model thread's own time in the transfer passes, a step: p50 {:.3} ms, max {:.3} ms",
            self.name, pump50, pump_max
        );
    }

    fn json(&self, pace: ignis_runtime::TransferPace) -> String {
        let list = |v: &[Duration]| v.iter().map(|d| format!("{:.3}", ms(*d))).collect::<Vec<_>>().join(",");
        format!(
            "{{\"move\":\"{}\",\"pace_in_bytes\":{},\"pace_out_bytes\":{},\"bytes\":{},\"duration_ms\":{:.3},\"gb_per_s\":{:.3},\"steps\":{},\"itl_ms\":[{}],\"baseline\":\"B1' and B2' alone at width 2, taken last\",\"baseline_itl_ms\":[{}],\"early_baseline_itl_ms\":[{}],\"pump_ms\":[{}],\"outside_stall_ms\":[{}],\"baseline_outside_stall_ms\":[{}],\"stall_per_miss_us\":[{:.2},{:.2}]}}",
            self.name,
            pace.move_in_bytes,
            pace.move_out_bytes,
            self.bytes,
            ms(self.duration),
            self.gbps(),
            self.steps,
            list(&self.itl),
            list(&self.baseline),
            list(&self.early_baseline),
            list(&self.pump),
            list(&self.outside),
            list(&self.baseline_outside),
            self.stall_per_miss.0,
            self.stall_per_miss.1,
        )
    }
}

/// The pace the legs run at: `IGNIS_KV_MOVE_PACE=<in MiB>,<out MiB>`, or
/// Flash-Next's own.
fn pace() -> ignis_runtime::TransferPace {
    let Ok(named) = std::env::var("IGNIS_KV_MOVE_PACE") else {
        return ignis_runtime::TransferPace::for_family(ignis_core::compute::ModelFamily::FlashNext);
    };
    let mib: Vec<u64> = named
        .split(',')
        .map(|v| v.trim().parse::<u64>().expect("IGNIS_KV_MOVE_PACE is <in MiB>,<out MiB>") << 20)
        .collect();
    assert_eq!(mib.len(), 2, "IGNIS_KV_MOVE_PACE is <in MiB>,<out MiB>");
    ignis_runtime::TransferPace { move_in_bytes: mib[0], move_out_bytes: mib[1] }
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

/// One width-3 round (C, B1 and B2 decoding): how many tokens B1 had
/// generated by its end, and its decode expert misses (GitHub #310).
#[derive(Debug, Clone, Copy)]
struct Round {
    b1_tokens: usize,
    misses: u64,
}

/// What a leg measured: its moves, its width-3 rounds after C came back, and
/// the text every request generated.
struct Leg {
    moves: Vec<Move>,
    after_restore: Vec<Round>,
    texts: Texts,
    expert_cache_bytes: u64,
}

fn leg(tier: Tier, pace: ignis_runtime::TransferPace, forced: Texts) -> Option<Leg> {
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
        transfer_pace: Some(pace),
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
    println!("expert cache: {} bytes", reserved.expert_cache_bytes);
    let expert_cache_bytes = reserved.expert_cache_bytes;
    let mut rig = Rig::new(sched, reserved.flash_next.clone(), forced);
    let interactive = RequestClass::Interactive;

    // ── the width-2 baseline: two lanes alone ──────────────────────────────
    let b0 = [
        rig.submit(20, B_PROMPT, B0_TOKENS, interactive),
        rig.submit(21, B_PROMPT, B0_TOKENS, interactive),
    ];
    let from = rig.mark();
    rig.to_idle();
    let width2 = rig.steady(from, rig.mark());

    // ── C, then B1 and B2 beside it ────────────────────────────────────────
    let c = rig.submit(1, C_PROMPT, C_TOKENS, RequestClass::Agent);
    rig.until_tokens(c, 4);
    let b = [
        rig.submit(2, B_PROMPT, B1_TOKENS, interactive),
        rig.submit(3, B_PROMPT, B2_TOKENS, interactive),
    ];
    rig.until_tokens(b[0], 16);
    rig.until_tokens(b[1], 16);
    let width3_from = rig.mark();
    rig.until_tokens(b[0], 16 + WIDTH3_TOKENS);
    let width3_to = rig.mark();

    // ── E0: an arrival that fits beside C moves nothing ────────────────────
    let e0 = rig.submit(4, E0_PROMPT, E_TOKENS, interactive);
    rig.until(|rig| rig.finished_at_length(e0));
    assert!(
        !rig.has(|e| matches!(e, SchedEvent::Evicted { .. } | SchedEvent::DiskSpilled { .. })),
        "E0 fits beside C: nothing moved for it"
    );

    // ── E: C out ───────────────────────────────────────────────────────────
    // Both tiers alike: started in the step E's admission ran in, and seen
    // landed at the top of the step that also runs E's first chunk -- whose
    // seconds are E's, not the move's. The steps between are the move's: B1
    // and B2 at width 2, C in no round, a window on the link. A KV-RAM move
    // issues its first window as it starts, beside that step's round; the
    // disk's first is its file's.
    let from = rig.mark();
    let e = rig.submit(5, E_PROMPT, E_TOKENS, interactive);
    let landed_out = move |ev: &SchedEvent| match tier {
        Tier::KvRam => matches!(ev, SchedEvent::Evicted { request, .. } if *request == c),
        Tier::KvDisk => matches!(ev, SchedEvent::DiskSpilled { request, from: DiskSource::Device } if *request == c),
    };
    rig.until(|rig| rig.has(landed_out));
    let start = (from..rig.mark()).find(|&i| rig.steps[i].busy_after).expect("the move out started");
    let end = rig.find(from, landed_out).unwrap();
    assert!(
        rig.steps[end].events.iter().any(|ev| matches!(ev, SchedEvent::PrefillChunk { request, .. } if *request == e)),
        "E took the room in the step C's move landed in"
    );
    let bytes = match tier {
        Tier::KvRam => rig.sched.host_tier().entry(c).expect("C's snapshot is in KV-RAM").bytes,
        Tier::KvDisk => rig.sched.disk_tier().expect("the tier").used_bytes(),
    };
    let first = match tier {
        Tier::KvRam => start,
        Tier::KvDisk => start + 1,
    };
    // The steps of the move at width 2: B1 and B2 both decoded in them.
    let out_steps = (first..end).filter(|&i| rig.all_decoded(&b, i)).collect::<Vec<_>>();
    assert!(!out_steps.is_empty(), "B1 and B2 decoded beside C's move out");
    println!("{tier:?} move out: {} of its {} steps at width 2", out_steps.len(), end - first);
    let out_duration = rig
        .kv_ram_span(tier, c, end)
        .unwrap_or(rig.steps[end].at - rig.steps[end].wall - (rig.steps[start].at - rig.steps[start].wall));

    // ── E ends: C in ───────────────────────────────────────────────────────
    // Started at the end of the step E ended in, seen landed at the top of
    // the step that restored C, which then decodes in it at width 3: the
    // steps between are the move's.
    rig.until(|rig| rig.finished_at_length(e));
    let e_done = rig.find(from, |ev| matches!(ev, SchedEvent::Done { request, .. } if *request == e)).unwrap();
    rig.until(|rig| rig.has(|ev| matches!(ev, SchedEvent::Restored { request, .. } if *request == c)));
    let restored = rig.find(e_done, |ev| matches!(ev, SchedEvent::Restored { request, .. } if *request == c)).unwrap();
    assert!(b.iter().all(|&r| !rig.finished_at_length(r)), "B1 and B2 both still decode when C comes back");
    let in_start = (e_done..=restored).find(|&i| rig.steps[i].busy_after).expect("the move in started");
    let in_steps = (in_start + 1..restored).filter(|&i| rig.all_decoded(&b, i)).collect::<Vec<_>>();
    assert!(!in_steps.is_empty(), "B1 and B2 decoded beside C's move in");
    println!("{tier:?} move in: {} of its {} steps at width 2", in_steps.len(), restored - in_start - 1);
    let in_duration = rig
        .kv_ram_span(tier, c, restored)
        .unwrap_or(rig.steps[restored].at - rig.steps[restored].wall - rig.steps[in_start].at);
    rig.to_idle();

    // ── width 3 before any arrival and after C is back, for the record:
    //    the two are at other tokens, and what the lanes generate decides
    //    their expert misses (GitHub #310) -- AC 43 compares the rounds after
    //    the restore with the same text unmoved ─────────────────────────────
    let three = [c, b[0], b[1]];
    let width3_steps = |from: usize, to: usize| {
        rig.steady(from, to).into_iter().filter(|&i| rig.all_decoded(&three, i)).collect::<Vec<_>>()
    };
    let before = width3_steps(width3_from, width3_to);
    let after = width3_steps(restored + 1, rig.mark());
    let (before50, _) = stats(&rig.itl_in(&b, &before));
    let (after50, _) = stats(&rig.itl_in(&b, &after));
    println!(
        "{tier:?} width 3 (C, B1, B2), B1's and B2's ITL p50: {before50:.2} ms over {} steps before any arrival, \
         {after50:.2} ms over {} steps after C came back -- other tokens, not the move's cost (AC 43)",
        before.len(),
        after.len(),
    );
    let after_restore = rig.rounds(&three, b[0], restored + 1, rig.mark());

    // ── the baseline: two fresh lanes alone at width 2, last ───────────────
    let b9 = [
        rig.submit(22, B_PROMPT, B9_TOKENS, interactive),
        rig.submit(23, B_PROMPT, B9_TOKENS, interactive),
    ];
    let from = rig.mark();
    rig.to_idle();
    let late = rig.steady(from, rig.mark());
    let late_itl = rig.itl(&b9, late[0], *late.last().unwrap());
    let (late50, late_max) = stats(&late_itl);
    let early_itl = rig.itl(&b0, width2[0], *width2.last().unwrap());
    let (early50, early_max) = stats(&early_itl);
    println!(
        "{tier:?} at pace in {} MiB, out {} MiB; width-2 baselines: first ITL p50 {early50:.2} ms max {early_max:.2} ms ({} gaps), \
         last p50 {late50:.2} ms max {late_max:.2} ms ({} gaps)",
        pace.move_in_bytes >> 20,
        pace.move_out_bytes >> 20,
        early_itl.len(),
        late_itl.len()
    );
    let per_miss = |steps: &mut dyn Iterator<Item = usize>| {
        let (misses, stall) = steps
            .filter_map(|i| rig.residency(i))
            .fold((0u64, Duration::ZERO), |(m, s), (misses, stall)| (m + misses, s + stall));
        stall.as_secs_f64() * 1e6 / misses.max(1) as f64
    };
    let measured = |name, bytes, duration, steps: &[usize], bound_on, p50_bound| {
        let n = steps.len().max(1);
        let (late_first, late_last) = window(&late, n);
        let (early_first, early_last) = window(&width2, n.min(width2.len()));
        let in_baseline = |i: &usize| (late_first..=late_last).contains(i);
        Move {
            name,
            bytes,
            duration,
            steps: steps.len(),
            itl: rig.itl_in(&b, steps),
            baseline: rig.itl(&b9, late_first, late_last),
            early_baseline: rig.itl(&b0, early_first, early_last),
            pump: steps.iter().map(|&i| rig.steps[i].pump).collect(),
            outside: steps.iter().map(|&i| rig.outside_stall(i)).collect(),
            baseline_outside: late.iter().filter(|i| in_baseline(i)).map(|&i| rig.outside_stall(i)).collect(),
            stall_per_miss: (
                per_miss(&mut steps.iter().copied()),
                per_miss(&mut late.iter().copied().filter(|i| in_baseline(i))),
            ),
            bound_on,
            p50_bound,
        }
    };
    let (out_name, in_name) = match tier {
        Tier::KvRam => ("move out (device -> KV-RAM)", "move in (KV-RAM -> device)"),
        Tier::KvDisk => ("move out (device -> KV-disk)", "move in (KV-disk -> device)"),
    };
    // GitHub #310 (owner, 2026-10-08): the move-in bound is the transfer's,
    // what its copies add to a round outside the expert stall; the rounds'
    // misses are the text's (AC 43).
    let moves = vec![
        measured(out_name, bytes, out_duration, &out_steps, BoundOn::Itl, 0.10),
        measured(in_name, bytes, in_duration, &in_steps, BoundOn::OutsideStall, 0.25),
    ];

    // ── no work lost ───────────────────────────────────────────────────────
    for (r, n) in [(c, C_TOKENS), (b[0], B1_TOKENS), (b[1], B2_TOKENS), (e0, E_TOKENS), (e, E_TOKENS), (b9[0], B9_TOKENS)] {
        assert!(rig.finished_at_length(r), "{r} generated its full max_tokens");
        assert_eq!(rig.tokens_of(r), n as usize);
    }
    assert!(!rig.has(|ev| matches!(ev, SchedEvent::Requeued { .. })), "no Requeued");
    assert!(!rig.has(|ev| matches!(ev, SchedEvent::SnapshotDropped { .. })), "no live snapshot dropped");
    assert!(!rig.has(|ev| matches!(ev, SchedEvent::DiskFailure { .. })), "no disk failure");
    let outs = rig.events().filter(|ev| matches!(ev, SchedEvent::Evicted { .. } | SchedEvent::DiskSpilled { .. })).count();
    assert_eq!(outs, 1, "C moved out once, and nothing else moved");
    if let Some(dir) = std::env::var_os("IGNIS_KV_P3_RAW").map(PathBuf::from) {
        std::fs::create_dir_all(&dir).expect("the raw samples' directory");
        rig.write_timeline(&dir.join(format!("ac37-{tier:?}-steps.json")));
        rig.write_texts(&dir.join(format!("ac37-{tier:?}-tokens.json")));
    }
    let texts = rig.texts();
    drop(rig);
    drop(blobs);
    Some(Leg { moves, after_restore, texts, expert_cache_bytes })
}

fn measure(tier: Tier) {
    let pace = pace();
    let Some(Leg { moves, .. }) = leg(tier, pace, forced_from_env()) else {
        return;
    };
    for m in &moves {
        m.print(tier);
    }
    if let Some(dir) = std::env::var_os("IGNIS_KV_P3_RAW").map(PathBuf::from) {
        std::fs::create_dir_all(&dir).expect("the raw samples' directory");
        let json = format!("[{}]\n", moves.iter().map(|m| m.json(pace)).collect::<Vec<_>>().join(","));
        let name = format!(
            "ac37-{tier:?}-in{}-out{}.json",
            pace.move_in_bytes >> 20,
            pace.move_out_bytes >> 20
        );
        std::fs::write(dir.join(name), json).expect("write the raw samples");
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

/// What the control measured: its width-3 rounds, all of them, and the text
/// every request generated (GitHub #310).
struct Control {
    rounds: Vec<Round>,
    texts: Texts,
    expert_cache_bytes: u64,
}

/// The control for the rounds after a move (GitHub #309): the same requests
/// on a pool with room for E beside C, so that nothing moves. Width-3 rounds
/// (C, B1, B2) before any arrival and after E has ended are compared, with
/// the expert cache's misses and its stall per round. Without `arrivals`
/// neither E0 nor E is sent (GitHub #310): what the rounds after them cost
/// without the long arrival's prefill.
fn control(forced: Texts, arrivals: bool) -> Option<Control> {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(FLASH_NEXT_DIR), PathBuf::from);
    let path = dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display())) {
        return None;
    }
    let blobs = BlobDir(
        std::env::var_os("IGNIS_KV_DISK_TEST_DIR")
            .map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.scratch/kv-disk-gpu"), PathBuf::from)
            .join("contention-control"),
    );
    let _ = std::fs::remove_dir_all(&blobs.0);
    std::fs::create_dir_all(&blobs.0).unwrap();
    let shape = EngineShape {
        max_context: CONTEXT,
        prefill_chunk: CHUNK,
        decode_lanes: 4,
        host_pool_bytes: ARENA_BYTES,
        prompt_reuse: false,
        retained_device_slots: 0,
        retained_host_slots: 0,
        retained_host_named: true,
        kv_disk_bytes: Some(0),
        // A chunk more than one context: E fits beside C, B1 and B2.
        kv_pool: Some(ignis_core::KvPoolSize::Tokens(u64::from(CONTEXT + CHUNK))),
        // And a chunk more budget for it, so that the expert cache -- what
        // a round's misses depend on -- is the leg's (GitHub #310).
        vram: ignis_core::VramMode::Derived {
            headroom_bytes: ignis_server::config::DEFAULT_VRAM_HEADROOM_BYTES
                - u64::from(CHUNK)
                    * ignis_core::ModelConfig::qwen38_flash_next()
                        .paged_sections(ignis_core::KvFormat::default())
                        .bytes_per_token(),
        },
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
            gpu_profile::skip_or_fail(&format!("load the Flash-Next scheduler for the control: {e}"));
            return None;
        }
    };
    println!("expert cache: {} bytes", reserved.expert_cache_bytes);
    let expert_cache_bytes = reserved.expert_cache_bytes;
    let mut rig = Rig::new(sched, reserved.flash_next.clone(), forced);
    let interactive = RequestClass::Interactive;
    let c = rig.submit(1, C_PROMPT, C_TOKENS, RequestClass::Agent);
    rig.until_tokens(c, 4);
    let b = [
        rig.submit(2, B_PROMPT, B1_TOKENS, interactive),
        rig.submit(3, B_PROMPT, B2_TOKENS, interactive),
    ];
    rig.until_tokens(b[0], 16);
    rig.until_tokens(b[1], 16);
    let width3_from = rig.mark();
    rig.until_tokens(b[0], 16 + WIDTH3_TOKENS);
    let width3_to = rig.mark();
    if arrivals {
        let e0 = rig.submit(4, E0_PROMPT, E_TOKENS, interactive);
        rig.until(|rig| rig.finished_at_length(e0));
        let e = rig.submit(5, E_PROMPT, E_TOKENS, interactive);
        rig.until(|rig| rig.finished_at_length(e));
    }
    let e_done = rig.mark();
    rig.until(|rig| rig.finished_at_length(b[0]));
    let after_to = rig.mark();
    assert!(
        !rig.has(|ev| matches!(ev, SchedEvent::Evicted { .. } | SchedEvent::DiskSpilled { .. })),
        "the pool holds E beside C: nothing moved"
    );
    let three = [c, b[0], b[1]];
    let phase = |from: usize, to: usize| {
        let steps: Vec<usize> =
            rig.steady(from, to).into_iter().filter(|&i| rig.all_decoded(&three, i)).collect();
        let (p50, _) = stats(&rig.itl_in(&b, &steps));
        let per_step = |f: &dyn Fn(&ignis_core::flash_next_counters::FlashNextCounters) -> u64| {
            let total: u64 = steps
                .iter()
                .filter_map(|&i| Some(f(rig.steps[i].counters.as_ref()?) - f(rig.steps[i - 1].counters.as_ref()?)))
                .sum();
            total as f64 / steps.len().max(1) as f64
        };
        let misses = per_step(&|c| c.residency.misses.iter().map(|p| p[0]).sum());
        let stall_ms = per_step(&|c| c.residency.stall_nanos[0]) / 1e6;
        (steps.len(), p50, misses, stall_ms)
    };
    let before = phase(width3_from, width3_to);
    let after = phase(e_done, after_to);
    println!(
        "control, nothing moved{}: width 3 (C, B1, B2) before any arrival: {} steps, B1's and B2's ITL p50 {:.2} ms, \
         {:.1} decode misses and {:.2} ms of expert stall a step; after E ended: {} steps, {:.2} ms, {:.1} misses, {:.2} ms",
        if arrivals { "" } else { ", no arrivals (\"after E\": the same rounds on)" },
        before.0,
        before.1,
        before.2,
        before.3,
        after.0,
        after.1,
        after.2,
        after.3
    );
    if let Some(dir) = std::env::var_os("IGNIS_KV_P3_RAW").map(PathBuf::from) {
        std::fs::create_dir_all(&dir).expect("the raw samples' directory");
        rig.write_timeline(&dir.join("ac37-control-steps.json"));
        rig.write_texts(&dir.join("ac37-control-tokens.json"));
    }
    let rounds = rig.rounds(&three, b[0], 0, after_to);
    let texts = rig.texts();
    drop(rig);
    drop(blobs);
    Some(Control { rounds, texts, expert_cache_bytes })
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact, minutes"]
fn a_long_arrival_beside_decoding_lanes_with_nothing_moved() {
    // `IGNIS_AC37_NO_ARRIVALS=1`: the same run without E0 and E (GitHub #310).
    let arrivals = std::env::var_os("IGNIS_AC37_NO_ARRIVALS").is_none();
    let _ = control(forced_from_env(), arrivals);
}

/// The median misses of `rounds`, `NaN` when there are none.
fn median_misses(misses: &[u64]) -> f64 {
    let mut v = misses.to_vec();
    v.sort_unstable();
    if v.is_empty() {
        return f64::NAN;
    }
    v[(v.len() - 1) / 2] as f64
}

/// AC 43 (spec vram-budget/03, GitHub #310): the rounds after a restore miss
/// what the same text misses unmoved.
///
/// A round's expert misses are those of what its lanes generate, and the
/// moved leg and its control generate different text once their batches
/// differ (greedy decode is not batch-invariant). So both are made to
/// generate one recorded text ([`AC43_TEXT`], or `IGNIS_AC37_FORCE`): the
/// KV-RAM leg of AC 37, and the control with E0 and E arriving and nothing
/// moved. The width-3 rounds after C's restore are compared with the
/// control's at the same tokens -- aligned by B1's token index; C is some
/// 260 tokens further on in the control, which its text (a token never
/// repeated) does not make a different cost. Starting bounds, for the owner
/// to confirm (printed, not asserted): C's experts come back into the cache
/// over its first rounds, so from the 20th round after the restore (N = 20)
/// the mean misses of rounds 20-49 are within +10% of the control's over the
/// same tokens (ten rounds are too few to judge: they spread +-15% either
/// way); and from the first round, each full window of 200 rounds'
/// median misses is within +10%. Asserted: the two runs generated the
/// same text, and no work was lost (AC 37's leg asserts it). A forced round
/// is never a captured graph, so the runs' ITL are not compared.
#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact, ~8 minutes"]
fn the_rounds_after_a_restore_miss_what_the_same_text_misses_unmoved() {
    let text = match std::env::var_os("IGNIS_AC37_FORCE") {
        Some(_) => forced_from_env(),
        None => read_texts(&Path::new(env!("CARGO_MANIFEST_DIR")).join(AC43_TEXT)),
    };
    for (seed, max_tokens) in [(1, C_TOKENS), (2, B1_TOKENS), (3, B2_TOKENS)] {
        assert!(
            text.get(&seed).is_some_and(|t| t.len() >= max_tokens as usize),
            "the text covers every compared lane's max_tokens (seed {seed})"
        );
    }
    // The two loads share one CUDA context, held from before the first:
    // each reads the device's free memory with it already counted, so the
    // second's expert cache is not the context's size smaller than the
    // first's (the control's misses were 6% higher for it before this).
    let _context = match ignis_artifact::CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("CUDA device: {e}"));
            return;
        }
    };
    let Some(moved) = leg(Tier::KvRam, pace(), text.clone()) else {
        return;
    };
    let Some(unmoved) = control(text, true) else {
        return;
    };
    for m in &moved.moves {
        m.print(Tier::KvRam);
    }
    let cache_gap = unmoved.expert_cache_bytes as f64 / moved.expert_cache_bytes as f64 - 1.0;
    println!(
        "AC 43: expert cache {} bytes moved, {} unmoved ({:+.2} %)",
        moved.expert_cache_bytes,
        unmoved.expert_cache_bytes,
        cache_gap * 100.0
    );
    // A smaller cache misses more: 2.3% less cache read 6% more misses.
    assert!(
        cache_gap.abs() < 0.01,
        "the two runs' expert caches are within 1% of each other, or their misses are not comparable"
    );
    for seed in [1, 2, 3] {
        let (a, b) = (&moved.texts[&seed], &unmoved.texts[&seed]);
        let n = a.len().min(b.len());
        assert_eq!(a[..n], b[..n], "seed {seed} generated the same text in both runs");
    }
    let unmoved_at: HashMap<usize, u64> = unmoved.rounds.iter().map(|r| (r.b1_tokens, r.misses)).collect();
    let pairs: Vec<(u64, u64)> = moved
        .after_restore
        .iter()
        .filter_map(|r| Some((r.misses, *unmoved_at.get(&r.b1_tokens)?)))
        .collect();
    assert!(pairs.len() >= 200, "{} width-3 rounds after the restore met the control's tokens", pairs.len());
    let windows = |size: usize| {
        pairs
            .chunks(size)
            .filter(|w| w.len() == size)
            .map(|w| {
                let moved = median_misses(&w.iter().map(|p| p.0).collect::<Vec<_>>());
                let unmoved = median_misses(&w.iter().map(|p| p.1).collect::<Vec<_>>());
                (moved, unmoved, (moved / unmoved - 1.0) * 100.0)
            })
            .collect::<Vec<_>>()
    };
    let mean_gap = |rounds: &[(u64, u64)]| {
        let (moved, unmoved) = rounds.iter().fold((0u64, 0u64), |(m, u), p| (m + p.0, u + p.1));
        (moved as f64 / unmoved as f64 - 1.0) * 100.0
    };
    let settled = mean_gap(&pairs[20..50]);
    println!(
        "AC 43: the rounds after the restore, mean misses moved against the same text unmoved: rounds 0-4 {:+.1} %, \
         5-9 {:+.1} %, 10-19 {:+.1} %, 20-49 {settled:+.1} % (N = 20, bound +10 %: {}); the first 10, each: {:?}",
        mean_gap(&pairs[..5]),
        mean_gap(&pairs[5..10]),
        mean_gap(&pairs[10..20]),
        if settled <= 10.0 { "within" } else { "OVER" },
        &pairs[..10]
    );
    for (size, bound) in [(50, None), (200, Some(10.0))] {
        let w = windows(size);
        let worst = w.iter().map(|x| x.2).fold(f64::NEG_INFINITY, f64::max);
        let listed: Vec<String> = w.iter().map(|(m, u, d)| format!("{m:.0}/{u:.0} ({d:+.1} %)")).collect();
        println!(
            "AC 43: {}-round windows from the restore on, median misses moved/unmoved: {}; worst {worst:+.1} %{}",
            size,
            listed.join(", "),
            bound.map_or(String::new(), |b| format!(
                ", bound +{b:.0} % from the first window: {}",
                if worst <= b { "within" } else { "OVER" }
            )),
        );
    }
    let all_moved = median_misses(&pairs.iter().map(|p| p.0).collect::<Vec<_>>());
    let all_unmoved = median_misses(&pairs.iter().map(|p| p.1).collect::<Vec<_>>());
    println!(
        "AC 43: all {} rounds after the restore: median misses {all_moved:.0} moved, {all_unmoved:.0} unmoved ({:+.1} %)",
        pairs.len(),
        (all_moved / all_unmoved - 1.0) * 100.0
    );
}
