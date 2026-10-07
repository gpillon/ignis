//! Prometheus metrics (GitHub #89, ADR 0017): the opt-in exposition
//! `ignis-server` serves with `--metrics` — at `GET /metrics` on its own
//! listener, and at `GET /ui/metrics` on the API listener for the Playground.
//!
//! [`Metrics`] is an aggregate projection of facts the model thread already
//! sends to the asynchronous telemetry consumer (`engine.rs`'s
//! `telemetry_task`). That consumer is its only writer; the HTTP task only
//! reads it, and all text encoding happens there, at scrape time. Nothing in
//! the scheduler, the model thread, the runtime or the kernel leaf knows it
//! exists, and without `--metrics` it is never built.
//!
//! Fixed atomics rather than a lock: a scrape can never make the consumer
//! wait, and the consumer can never make a scrape wait. A scrape therefore
//! reads each series on its own, not one consistent cut across all of them —
//! which Prometheus does not assume either.

use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::Router;
use axum::http::header;
use axum::routing::get;
use ignis_core::checkpoint::{RetainedKind, RetainedStateOperation, ReuseSource};
use ignis_core::scheduler::{DiskOp, DiskSource};
use ignis_core::flash_next_counters::{FlashNextCounterSource, FlashNextCounters};
use ignis_core::ngram_table::NgramCounters;
use ignis_core::residency::{KClass, Phase, ResidencyCounters};
use ignis_core::{RetainedSkip, SubmitError};

/// What one load reserved, and the shapes the reservations bound (GitHub
/// #216, ADR 0030 §Observability). Computed once, during the load that
/// already builds the plan, and never read again — so exporting it adds no
/// serving work of any kind.
///
/// A load that has no plan (the placeholder path, which loads no model) has
/// no reservations either, and every series below stays at its zero.
#[derive(Debug, Clone)]
pub struct LoadReservations {
    /// The plan's lines, as the plan itself holds them — so the exposition
    /// names them with the plan's own spellings rather than a second set
    /// kept in step by hand.
    pub lines: ignis_core::VramLines,
    /// The budget the plan was laid out inside, derived or explicit.
    pub budget_bytes: u64,
    /// The KV pool's page count, as the leaf verified it at load.
    pub kv_pool_pages: u32,
    /// One KV page's bytes.
    pub kv_page_bytes: u64,
    /// A Flash-Next load's two lines beyond [`ignis_core::VramLines`] (GitHub
    /// #306): the expert residency's fixed bytes (ring and tables), and the
    /// VRAM expert cache that takes what the plan leaves. 0 on a 27B load.
    pub residency_bytes: u64,
    pub expert_cache_bytes: u64,
    /// `--kv-host-pool-bytes`, pinned whole at start.
    pub kv_ram_arena_bytes: u64,
    /// The retained slots this load hands out — the effective count, which
    /// is 0 with prompt reuse off and neither `--retained-device` nor
    /// `--retained-host` named.
    pub retained_slots: u32,
    /// Of those, the host ones (GitHub #281), and their pinned block: host
    /// memory, beside the plan's lines rather than one of them.
    pub retained_host_slots: u32,
    pub retained_host_bytes: u64,
    /// The KV-disk tier's effective budget (spec vram-budget/03), or `None`
    /// on a load without the tier -- which then never renders a disk series,
    /// as a 27B load never renders the expert residency's.
    pub kv_disk_bytes: Option<u64>,
    /// Where a Flash-Next load's counters are read (GitHub #301, #302);
    /// `None` on a 27B load. Not a reservation: it rides here to reach the
    /// telemetry consumer, which reads it at every tick
    /// ([`crate::Server::with_load_reservations`]).
    pub flash_next: Option<Arc<FlashNextCounterSource>>,
}

/// The exposition's content type: Prometheus text format 0.0.4.
pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// `ignis_request_ttft_seconds`' bucket boundaries (ADR 0017), in
/// milliseconds — the telemetry clock's unit.
const TTFT_BOUNDS_MS: [u64; 12] =
    [50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000, 120_000, 300_000];

/// `ignis_request_duration_seconds`' bucket boundaries (ADR 0017), in
/// milliseconds.
const DURATION_BOUNDS_MS: [u64; 12] =
    [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000, 120_000, 300_000, 600_000];

/// Why a submission was rejected: `ignis_requests_rejected_total`'s fixed
/// `reason` set (ADR 0017).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejection {
    /// The engine could not admit it right now.
    Full,
    /// It named a model the engine does not load.
    UnknownModel,
    /// It can never fit, however empty the engine is.
    Oversized,
}

impl Rejection {
    /// The reason a submit error counts under. A request longer than the
    /// per-sequence context (GitHub #166, after ADR 0017's table) is a
    /// request that can never fit, like one larger than the KV pool.
    pub fn of(err: &SubmitError) -> Self {
        match err {
            SubmitError::Full => Self::Full,
            SubmitError::UnknownModel(_) => Self::UnknownModel,
            SubmitError::Oversized | SubmitError::ContextExceeded { .. } => Self::Oversized,
        }
    }

    const ALL: [Rejection; 3] = [Self::Full, Self::UnknownModel, Self::Oversized];

    fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::UnknownModel => "unknown_model",
            Self::Oversized => "oversized",
        }
    }
}

/// Which of the two counting routes a call reached:
/// `ignis_tokenize_requests_total`'s fixed `route` label (GitHub #285).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizeRoute {
    /// `POST /v1/tokenize`.
    Tokenize,
    /// `POST /v1/detokenize`.
    Detokenize,
}

impl TokenizeRoute {
    const ALL: [TokenizeRoute; 2] = [Self::Tokenize, Self::Detokenize];

    fn label(self) -> &'static str {
        match self {
            Self::Tokenize => "tokenize",
            Self::Detokenize => "detokenize",
        }
    }
}

/// Which typed primitive a question asked for: `ignis_decisions_total`'s
/// fixed `type` label (GitHub #241, ADR 0034).
///
/// Its own enum rather than `decide::QuestionKind` so the projection keeps
/// owning its label sets — the same separation `Rejection` keeps from
/// `SubmitError`. The wire vocabulary is Jev's and may grow a primitive
/// without the contract growing a label value in the same release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primitive {
    Noul,
    Choice,
    Score,
    Scalar,
    Number,
    Point,
    Box,
    Locate,
}

impl Primitive {
    /// Every primitive, in the order their series are rendered.
    pub const ALL: [Primitive; 8] = [
        Self::Noul,
        Self::Choice,
        Self::Score,
        Self::Scalar,
        Self::Number,
        Self::Point,
        Self::Box,
        Self::Locate,
    ];

    /// Whether this primitive's answer has an **answer mass** to observe
    /// (GitHub #242).
    ///
    /// The three readouts do: their answer is a restricted softmax over
    /// declared option tokens, and the mass is how much of the real
    /// distribution those options held — the one silent failure the family
    /// exists to show. The four **constrained decodes** do not: their answer is a run
    /// of sampled tokens, each drawn from a permitted set, and there is no
    /// single position whose distribution the answer stands on. A number
    /// reported as mass 1 would be a lie, and one reported as 0 would put a
    /// false alarm in the bucket a real one lands in.
    pub(crate) fn has_answer_mass(self) -> bool {
        matches!(self, Self::Noul | Self::Choice | Self::Score)
    }

    /// The label value this primitive is exported under — and the word the
    /// request log calls it by, so the two are the same string by
    /// construction rather than by coincidence.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Scalar => "scalar",
            Self::Number => "number",
            Self::Point => "point",
            Self::Box => "box",
            Self::Locate => "locate",
        }
    }
}

use ignis_core::locate::{Compression, Kind, Method};

/// Whether a served `locate` found its answer (GitHub #278):
/// `ignis_locates_total`'s `found` label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocateFound {
    /// `found` at or above its threshold: the answer names a segment.
    True,
    /// `found` below its threshold: "not found".
    False,
    /// A route with no measured `found` rule, which names a segment always.
    Unmeasured,
}

impl LocateFound {
    fn label(self) -> &'static str {
        match self {
            Self::True => "true",
            Self::False => "false",
            Self::Unmeasured => "unmeasured",
        }
    }
}

/// Every `ignis_locates_total` series that can move (GitHub #278, spec 22):
/// the routes the endpoint serves — a fold of prose and a fold under the
/// vote are refused — each with the `found` values it can report.
const LOCATE_SERIES: [(Kind, Method, Compression, LocateFound); 11] = {
    use Compression::{None as Whole, TemplateFold as Fold};
    use Kind::{Log, Prose, Records};
    use LocateFound::{False, True, Unmeasured};
    use Method::{Shortlist, Vote};
    [
        (Log, Shortlist, Fold, True),
        (Log, Shortlist, Fold, False),
        (Log, Shortlist, Whole, Unmeasured),
        (Prose, Shortlist, Whole, True),
        (Prose, Shortlist, Whole, False),
        (Records, Shortlist, Whole, True),
        (Records, Shortlist, Whole, False),
        (Records, Shortlist, Fold, Unmeasured),
        (Log, Vote, Whole, Unmeasured),
        (Prose, Vote, Whole, Unmeasured),
        (Records, Vote, Whole, Unmeasured),
    ]
};

/// `ignis_decision_answer_mass`' bucket boundaries (ADR 0017 as amended by
/// #241), in millionths.
///
/// Not a latency scale reused for a ratio. The measured baseline is a median
/// of 0.996 and above from 8 to 256 options
/// (`docs/findings/2026-09-19-typed-option-logit-readout.md`), so every
/// healthy observation would land in one bucket under any evenly spaced
/// scale and the panel would show a flat line whatever happened. The
/// resolution is where the signal is — between 0.99 and 1 — and the two
/// coarse buckets below exist to make a collapse unmissable rather than to
/// resolve it.
const ANSWER_MASS_BOUNDS: [u64; 10] =
    [500_000, 900_000, 950_000, 980_000, 990_000, 995_000, 998_000, 999_000, 999_500, 1_000_000];

/// A fixed-bucket histogram over observations in `[0, 1]`, counted in
/// millionths.
///
/// Millionths rather than `f64` for the same reason the latency histograms
/// count milliseconds: a sum of integers is exact and a sum of floats is
/// whatever the order of arrival made it, and two scrapes of the same
/// observations must agree.
#[derive(Debug)]
struct Ratio {
    bounds: &'static [u64; 10],
    /// Observations per bucket, not cumulative; the last is `+Inf`'s own.
    buckets: [AtomicU64; 11],
    sum_millionths: AtomicU64,
}

impl Ratio {
    fn new(bounds: &'static [u64; 10]) -> Self {
        Self { bounds, buckets: Default::default(), sum_millionths: AtomicU64::new(0) }
    }

    /// Observe `value`, a probability.
    ///
    /// Anything that is **not** a probability — negative, above one, or not
    /// a number — is counted in `+Inf` and nowhere else. It is not clamped:
    /// the last bound is `1`, so `le="1"` equalling `_count` is the
    /// exposition's own statement that every reading was in range, and
    /// clamping would make that statement true by construction and
    /// therefore worth nothing. `_count - le="1"` is the number of
    /// readings that were not probabilities.
    ///
    /// It is not dropped either, because a readout producing a NaN is a
    /// failure of exactly the kind this histogram exists to show, and a
    /// clamp to zero would file it under "the options collapsed" — a
    /// different diagnosis with a different cause.
    ///
    /// `_sum` takes the value when it is one a sum can hold, so a mass of
    /// 1.5 inflates the mean the way it should; a NaN or a negative
    /// contributes nothing to it, and the bucket count is where they are
    /// visible.
    fn observe(&self, value: f64) {
        let millionths = match value.is_finite() && value >= 0.0 {
            true => (value * 1e6).round() as u64,
            false => 0,
        };
        let in_range = value.is_finite() && (0.0..=1.0).contains(&value);
        let bucket = match in_range {
            true => self
                .bounds
                .iter()
                .position(|&bound| millionths <= bound)
                .unwrap_or(self.bounds.len()),
            false => self.bounds.len(),
        };
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        self.sum_millionths.fetch_add(millionths, Ordering::Relaxed);
    }

    fn render(&self, out: &mut String, name: &str, help: &str) {
        render_buckets(out, name, help, self.bounds, &self.buckets, &self.sum_millionths, MILLIONTHS);
    }
}

/// The scale an observation is counted in: thousandths of a second for the
/// latency histograms, millionths for a ratio.
const THOUSANDTHS: u64 = 1_000;
const MILLIONTHS: u64 = 1_000_000;

/// Cumulative `_bucket` lines, then `_sum` and `_count`, for a histogram
/// whose observations are integers of `1 / scale`.
///
/// One function for both shapes because the exposition format is the
/// format whatever the unit is — the two histogram types differ in their
/// bucket count, their scale and what they accept, and not in a single
/// character of what they print. The count is the `+Inf` bucket as read
/// here, so `_count` and the last bucket always agree.
fn render_buckets(
    out: &mut String,
    name: &str,
    help: &str,
    bounds: &[u64],
    buckets: &[AtomicU64],
    sum: &AtomicU64,
    scale: u64,
) {
    declare(out, name, "histogram", help);
    let mut cumulative = 0;
    for (bucket, count) in buckets.iter().enumerate() {
        cumulative += count.load(Ordering::Relaxed);
        let le = match bounds.get(bucket) {
            Some(&bound) => decimal(bound, scale),
            None => "+Inf".to_owned(),
        };
        let _ = writeln!(out, "{name}_bucket{{le=\"{le}\"}} {cumulative}");
    }
    let _ = writeln!(out, "{name}_sum {}", decimal(sum.load(Ordering::Relaxed), scale));
    let _ = writeln!(out, "{name}_count {cumulative}");
}

/// `value` units of `1 / scale`, as a decimal without trailing zeros.
fn decimal(value: u64, scale: u64) -> String {
    let (whole, frac) = (value / scale, value % scale);
    if frac == 0 {
        return whole.to_string();
    }
    let digits = scale.ilog10() as usize;
    let frac = format!("{frac:0digits$}");
    format!("{whole}.{}", frac.trim_end_matches('0'))
}

/// A fixed-bucket histogram over millisecond observations.
#[derive(Debug)]
struct Histogram {
    bounds_ms: &'static [u64; 12],
    /// Observations per bucket, not cumulative; the last is `+Inf`'s own.
    buckets: [AtomicU64; 13],
    sum_ms: AtomicU64,
}

impl Histogram {
    fn new(bounds_ms: &'static [u64; 12]) -> Self {
        Self { bounds_ms, buckets: Default::default(), sum_ms: AtomicU64::new(0) }
    }

    fn observe(&self, ms: u64) {
        let bucket = self.bounds_ms.iter().position(|&bound| ms <= bound).unwrap_or(self.bounds_ms.len());
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        self.sum_ms.fetch_add(ms, Ordering::Relaxed);
    }

    fn render(&self, out: &mut String, name: &str, help: &str) {
        render_buckets(out, name, help, self.bounds_ms, &self.buckets, &self.sum_ms, THOUSANDTHS);
    }
}

/// The aggregate the telemetry consumer maintains while metrics are on —
/// except the rejections, which the HTTP handler records after the submit
/// call returns its error (ADR 0017).
#[derive(Debug)]
pub struct Metrics {
    waiting: AtomicU64,
    running: AtomicU64,
    accepted: AtomicU64,
    completed: AtomicU64,
    cancelled: AtomicU64,
    rejected: [AtomicU64; 3],
    generated_tokens: AtomicU64,
    decoded_tokens: AtomicU64,
    kv_evictions: AtomicU64,
    kv_ram_evictions: AtomicU64,
    prefix_reused_tokens: AtomicU64,
    /// Per [`ReuseSource::index`], for each of the families below (#190).
    retained_reused_tokens: RetainedFamily,
    retained_state_hits: RetainedFamily,
    retained_state_misses: RetainedFamily,
    retained_state_spills: RetainedFamily,
    retained_state_discards: RetainedFamily,
    retained_state_restores: RetainedFamily,
    /// Per [`RetainedSkip::index`] (GitHub #216).
    retained_slot_skips: [AtomicU64; RetainedSkip::ALL.len()],
    /// The retained slots held right now, and the count this load hands out.
    retained_slots_in_use: AtomicU64,
    retained_slots_capacity: AtomicU64,
    /// Of the retained slots, the host ones and their pinned block (GitHub
    /// #281): written once at load.
    retained_host_slots: AtomicU64,
    retained_host_bytes: AtomicU64,
    /// The load's reservations (GitHub #216): the plan's lines in
    /// `VramLines::entries()` order, then the shapes they bound. Written once
    /// at load, and zero on a load that built no plan.
    vram_reserved: [AtomicU64; VRAM_LINE_COUNT],
    /// Flash-Next's `residency` and `expert_cache` lines (GitHub #306), in
    /// [`FLASH_NEXT_VRAM_LINES`] order: exported beside the plan's own.
    flash_next_vram_reserved: [AtomicU64; 2],
    vram_budget_bytes: AtomicU64,
    kv_pool_pages: AtomicU64,
    kv_page_bytes: AtomicU64,
    kv_ram_arena_capacity_bytes: AtomicU64,
    /// What the scheduler has occupied, republished on every tick.
    kv_pool_used_pages: AtomicU64,
    kv_ram_arena_used_bytes: AtomicU64,
    /// KV-disk (spec vram-budget/03): 1 while the load has the tier, its
    /// budget and what it holds, its live spills per [`DiskSource`] and its
    /// failures per [`DiskOp`] (in their declaration order).
    kv_disk_loaded: AtomicU64,
    kv_disk_capacity_bytes: AtomicU64,
    kv_disk_used_bytes: AtomicU64,
    kv_disk_spills: [AtomicU64; 2],
    kv_disk_failures: [AtomicU64; 2],
    ttft: Histogram,
    duration: Histogram,
    /// Per [`Primitive::ALL`] (GitHub #241). Absent from the exposition
    /// until one of them moves — see [`Metrics::render`].
    decisions: [AtomicU64; Primitive::ALL.len()],
    answer_mass: Ratio,
    /// Requests whose reasoning block the thinking budget closed (spec
    /// server/08). Absent from the exposition until the first one, like the
    /// decision family.
    thinking_forced_closes: AtomicU64,
    /// Speculative decoding (P5-06's request counters, GitHub #307): verify
    /// rounds, drafts verified and drafts committed, and per draft position
    /// how many rounds proposed it and committed it.
    spec_rounds: AtomicU64,
    spec_drafted: AtomicU64,
    spec_accepted: AtomicU64,
    spec_drafted_at: [AtomicU64; ignis_core::types::DRAFT_POSITIONS],
    spec_accepted_at: [AtomicU64; ignis_core::types::DRAFT_POSITIONS],
    /// Per [`TokenizeRoute::ALL`] (GitHub #285). Absent from the exposition
    /// until one of them moves. Counts calls, not requests of the lifecycle
    /// above: a counting route never enters the scheduler.
    tokenize_calls: [AtomicU64; TokenizeRoute::ALL.len()],
    /// Per [`LOCATE_SERIES`] (GitHub #278). Absent from the exposition until
    /// the first `locate`, like the decision family.
    locates: [AtomicU64; LOCATE_SERIES.len()],
    /// Open Responses API WebSockets, and their requests waiting in the
    /// server-wide admission queue (GitHub #282).
    responses_sockets: AtomicU64,
    responses_queued: AtomicU64,
    /// A Flash-Next load's expert residency and n-gram counters (GitHub
    /// #301, #302), as the telemetry consumer last read them, and 1 in
    /// `flash_next_loaded` while the load has them. Never set on a 27B load,
    /// whose exposition then ends where it always did.
    flash_next: FlashNextSeries,
    flash_next_loaded: AtomicU64,
}

/// [`FlashNextCounters`] as fixed atomics, each series stored and read on
/// its own.
#[derive(Debug, Default)]
struct FlashNextSeries {
    hits: [[AtomicU64; 2]; KClass::COUNT],
    misses: [[AtomicU64; 2]; KClass::COUNT],
    prefetch_issued: AtomicU64,
    prefetch_used: AtomicU64,
    bytes_moved: [AtomicU64; 2],
    stall_nanos: [AtomicU64; 2],
    slots_capacity: [AtomicU64; KClass::COUNT],
    slots_in_use: [AtomicU64; KClass::COUNT],
    ngram_hot_rows: AtomicU64,
    ngram_file_rows: AtomicU64,
    ngram_reads: AtomicU64,
    ngram_read_bytes: AtomicU64,
}

impl FlashNextSeries {
    fn store(&self, reading: &FlashNextCounters) {
        let store = |series: &AtomicU64, value: u64| series.store(value, Ordering::Relaxed);
        let r = &reading.residency;
        for class in 0..KClass::COUNT {
            for phase in 0..2 {
                store(&self.hits[class][phase], r.hits[class][phase]);
                store(&self.misses[class][phase], r.misses[class][phase]);
            }
            store(&self.slots_capacity[class], u64::from(reading.slots_capacity[class]));
            store(&self.slots_in_use[class], u64::from(reading.slots_in_use[class]));
        }
        store(&self.prefetch_issued, r.prefetch_issued);
        store(&self.prefetch_used, r.prefetch_used);
        for phase in 0..2 {
            store(&self.bytes_moved[phase], r.bytes_moved[phase]);
            store(&self.stall_nanos[phase], r.stall_nanos[phase]);
        }
        let n = &reading.ngram;
        store(&self.ngram_hot_rows, n.hot_rows);
        store(&self.ngram_file_rows, n.file_rows);
        store(&self.ngram_reads, n.reads);
        store(&self.ngram_read_bytes, n.read_bytes);
    }

    fn load(&self) -> FlashNextCounters {
        let load = |series: &AtomicU64| series.load(Ordering::Relaxed);
        let hot_rows = load(&self.ngram_hot_rows);
        let file_rows = load(&self.ngram_file_rows);
        FlashNextCounters {
            residency: ResidencyCounters {
                hits: self.hits.each_ref().map(|phases| phases.each_ref().map(load)),
                misses: self.misses.each_ref().map(|phases| phases.each_ref().map(load)),
                prefetch_issued: load(&self.prefetch_issued),
                prefetch_used: load(&self.prefetch_used),
                bytes_moved: self.bytes_moved.each_ref().map(load),
                stall_nanos: self.stall_nanos.each_ref().map(load),
            },
            slots_capacity: self.slots_capacity.each_ref().map(|c| load(c) as u32),
            slots_in_use: self.slots_in_use.each_ref().map(|c| load(c) as u32),
            ngram: NgramCounters {
                rows: hot_rows + file_rows,
                hot_rows,
                file_rows,
                reads: load(&self.ngram_reads),
                read_bytes: load(&self.ngram_read_bytes),
            },
        }
    }
}

/// One retained-state family: a count per residency tier and per kind of
/// retained state (GitHub #216) — `[tier][kind]`, so a prompt checkpoint and
/// a shared prefix are never summed into one figure.
type RetainedFamily = [[AtomicU64; RetainedKind::ALL.len()]; ReuseSource::ALL.len()];

/// How many lines a VRAM plan has, taken from the plan's own shape rather
/// than written out again here.
const VRAM_LINE_COUNT: usize = ignis_core::VramLines::LINES;

/// The two lines a Flash-Next plan has beyond [`ignis_core::VramLines`]
/// (GitHub #306): the residency's fixed bytes, then the expert cache. Always
/// exported, 0 on the 27B.
const FLASH_NEXT_VRAM_LINES: [&str; 2] = ["residency", "expert_cache"];

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    /// An all-zero projection.
    pub fn new() -> Self {
        Self {
            waiting: AtomicU64::new(0),
            running: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            rejected: Default::default(),
            generated_tokens: AtomicU64::new(0),
            decoded_tokens: AtomicU64::new(0),
            kv_evictions: AtomicU64::new(0),
            kv_ram_evictions: AtomicU64::new(0),
            prefix_reused_tokens: AtomicU64::new(0),
            retained_reused_tokens: Default::default(),
            retained_state_hits: Default::default(),
            retained_state_misses: Default::default(),
            retained_state_spills: Default::default(),
            retained_state_discards: Default::default(),
            retained_state_restores: Default::default(),
            retained_slot_skips: Default::default(),
            retained_slots_in_use: AtomicU64::new(0),
            retained_slots_capacity: AtomicU64::new(0),
            retained_host_slots: AtomicU64::new(0),
            retained_host_bytes: AtomicU64::new(0),
            vram_reserved: Default::default(),
            flash_next_vram_reserved: Default::default(),
            vram_budget_bytes: AtomicU64::new(0),
            kv_pool_pages: AtomicU64::new(0),
            kv_page_bytes: AtomicU64::new(0),
            kv_ram_arena_capacity_bytes: AtomicU64::new(0),
            kv_pool_used_pages: AtomicU64::new(0),
            kv_ram_arena_used_bytes: AtomicU64::new(0),
            kv_disk_loaded: AtomicU64::new(0),
            kv_disk_capacity_bytes: AtomicU64::new(0),
            kv_disk_used_bytes: AtomicU64::new(0),
            kv_disk_spills: Default::default(),
            kv_disk_failures: Default::default(),
            ttft: Histogram::new(&TTFT_BOUNDS_MS),
            duration: Histogram::new(&DURATION_BOUNDS_MS),
            decisions: Default::default(),
            answer_mass: Ratio::new(&ANSWER_MASS_BOUNDS),
            thinking_forced_closes: AtomicU64::new(0),
            spec_rounds: AtomicU64::new(0),
            spec_drafted: AtomicU64::new(0),
            spec_accepted: AtomicU64::new(0),
            spec_drafted_at: std::array::from_fn(|_| AtomicU64::new(0)),
            spec_accepted_at: std::array::from_fn(|_| AtomicU64::new(0)),
            tokenize_calls: Default::default(),
            locates: Default::default(),
            responses_sockets: AtomicU64::new(0),
            responses_queued: AtomicU64::new(0),
            flash_next: FlashNextSeries::default(),
            flash_next_loaded: AtomicU64::new(0),
        }
    }

    /// A Flash-Next load's counters as the telemetry consumer read them
    /// (GitHub #301, #302), rendered after every other family; `None` for a
    /// load that has none, which takes the families out again.
    pub(crate) fn set_flash_next(&self, reading: Option<&FlashNextCounters>) {
        if let Some(reading) = reading {
            self.flash_next.store(reading);
        }
        self.flash_next_loaded.store(u64::from(reading.is_some()), Ordering::Relaxed);
    }

    /// A `locate` was answered (GitHub #278, spec 22): by the kind it
    /// resolved to, its method, its compression, and whether it found an
    /// answer. Counted beside `ignis_decisions_total{type="locate"}`, which
    /// still counts it once as a decision. A combination the endpoint never
    /// serves is not counted — it has no series.
    pub fn record_locate(&self, kind: Kind, method: Method, compression: Compression, found: LocateFound) {
        let at = LOCATE_SERIES
            .iter()
            .position(|&series| series == (kind, method, compression, found));
        debug_assert!(at.is_some(), "{kind:?} {method:?} {compression:?} {found:?} is a route the endpoint serves");
        if let Some(at) = at {
            self.locates[at].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// One question of a decision was answered (GitHub #241, ADR 0034):
    /// count it under its primitive and observe the **answer mass** its
    /// readout carried.
    ///
    /// Recorded by the HTTP handler, like [`Metrics::record_rejected`] and
    /// for the same reason: there is no fact for it on the model thread's
    /// stream and inventing one would put a decision's arithmetic on the
    /// inference path to observe something the handler already holds.
    ///
    /// Called once per *question*, not once per request: twenty questions
    /// over one `state` are twenty decisions, and the mass of each is a
    /// separate reading of the same failure.
    ///
    /// `answer_mass` is `None` for a **constrained decode** (GitHub #242): a `number`,
    /// `point` or `box` is counted like any other decision and observes no
    /// mass, because it has none to observe
    /// ([`Primitive::has_answer_mass`]). The histogram is therefore
    /// deliberately **readout-only**, and its `_count` sits below the
    /// counter's sum by exactly the number of constrained decodes served — stated in
    /// ADR 0017's row rather than left for a reader to infer from a graph
    /// that does not add up.
    pub fn record_decision(&self, primitive: Primitive, answer_mass: Option<f64>) {
        // Two atomics, so a scrape can land between them. The mass goes
        // first on purpose: the histogram may then be one ahead of the
        // counter for a moment, which reads as "a decision whose count has
        // not arrived yet", where the other order reads as "a decision with
        // no mass" — the shape of the failure this whole family exists to
        // show.
        debug_assert_eq!(
            answer_mass.is_some(),
            primitive.has_answer_mass(),
            "a primitive observes a mass exactly when it has one to observe"
        );
        if let Some(answer_mass) = answer_mass {
            self.answer_mass.observe(answer_mass);
        }
        // `ALL` lists the primitives in declaration order, so a primitive's
        // discriminant is its slot.
        self.decisions[primitive as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A request that ran verify rounds ended: its speculative counters
    /// (GitHub #307), the request log's `spec.*` summed across requests.
    pub(crate) fn record_speculation(&self, spec: &ignis_core::types::SpecCounters) {
        self.spec_rounds.fetch_add(u64::from(spec.rounds), Ordering::Relaxed);
        self.spec_drafted.fetch_add(u64::from(spec.drafted), Ordering::Relaxed);
        self.spec_accepted.fetch_add(u64::from(spec.accepted), Ordering::Relaxed);
        for (series, &n) in self.spec_drafted_at.iter().zip(&spec.drafted_at) {
            series.fetch_add(u64::from(n), Ordering::Relaxed);
        }
        for (series, &n) in self.spec_accepted_at.iter().zip(&spec.accepted_at) {
            series.fetch_add(u64::from(n), Ordering::Relaxed);
        }
    }

    /// A completed request's thinking budget forced the model's close (spec
    /// server/08).
    pub(crate) fn record_thinking_forced_close(&self) {
        self.thinking_forced_closes.fetch_add(1, Ordering::Relaxed);
    }

    /// A call reached `/v1/tokenize` or `/v1/detokenize` (GitHub #285),
    /// counted whatever it answered: the series says the route is used, and
    /// a client polling it is what an operator looks for. Recorded by the
    /// handler, like [`Metrics::record_rejected`], since the call has no fact
    /// on the model thread's stream — it never reaches it.
    pub(crate) fn record_tokenize(&self, route: TokenizeRoute) {
        // `ALL` lists the routes in declaration order.
        self.tokenize_calls[route as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A submission was rejected, for `reason`.
    pub fn record_rejected(&self, reason: Rejection) {
        // `ALL` lists the reasons in declaration order, so a reason's
        // discriminant is its slot.
        self.rejected[reason as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A request was evicted to the host KV-RAM tier.
    pub(crate) fn record_eviction(&self) {
        self.kv_evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// A live snapshot was dropped out of the KV-RAM tier to make room for
    /// another (GitHub #224) — the tier's own eviction, the mirror of
    /// [`Metrics::record_eviction`]'s departure from the device.
    pub(crate) fn record_kv_ram_eviction(&self) {
        self.kv_ram_evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// A live snapshot was written to KV-disk from `from` (spec
    /// vram-budget/03): straight from the device, or demoted from KV-RAM.
    pub(crate) fn record_kv_disk_spill(&self, from: DiskSource) {
        let at = match from {
            DiskSource::Device => 0,
            DiskSource::KvRam => 1,
        };
        self.kv_disk_spills[at].fetch_add(1, Ordering::Relaxed);
    }

    /// A KV-disk transfer failed (spec vram-budget/03).
    pub(crate) fn record_kv_disk_failure(&self, op: DiskOp) {
        let at = match op {
            DiskOp::Write => 0,
            DiskOp::Read => 1,
        };
        self.kv_disk_failures[at].fetch_add(1, Ordering::Relaxed);
    }

    /// A request's prefill skipped `tokens` through a sibling's prefix.
    pub(crate) fn record_prefix_reused(&self, tokens: u32) {
        self.prefix_reused_tokens.fetch_add(u64::from(tokens), Ordering::Relaxed);
    }

    /// A request's prefill skipped `tokens` through retained state in
    /// `source` — a retained prefix, or a prompt checkpoint (GitHub #190,
    /// #216).
    pub(crate) fn record_retained_reused(&self, source: ReuseSource, kind: RetainedKind, tokens: u32) {
        self.retained_reused_tokens[source.index()][kind.index()]
            .fetch_add(u64::from(tokens), Ordering::Relaxed);
    }

    /// One retained-state lifecycle operation, in its residency tier and on
    /// the kind of state it moved (GitHub #190, #216). The telemetry consumer
    /// is the only writer.
    pub(crate) fn record_retained_state(
        &self,
        operation: RetainedStateOperation,
        source: ReuseSource,
        kind: RetainedKind,
    ) {
        let series = match operation {
            RetainedStateOperation::Hit => &self.retained_state_hits,
            RetainedStateOperation::Miss => &self.retained_state_misses,
            RetainedStateOperation::Spill => &self.retained_state_spills,
            RetainedStateOperation::Discard => &self.retained_state_discards,
            RetainedStateOperation::Restore => &self.retained_state_restores,
        };
        series[source.index()][kind.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// A publish or a capture found no retained slot, or no tail page
    /// (GitHub #216): the signal that a load has run out of room to leave
    /// reuse behind and is running on without it.
    pub(crate) fn record_retained_slot_skip(&self, skip: RetainedSkip) {
        self.retained_slot_skips[skip.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// The retained slots a prefix or checkpoint image holds right now.
    pub(crate) fn set_retained_slots_in_use(&self, in_use: u32) {
        self.retained_slots_in_use.store(u64::from(in_use), Ordering::Relaxed);
    }

    /// What the scheduler had occupied when the last step ended (GitHub
    /// #216): the main pool's pages, and the KV-RAM arena's bytes. Both are
    /// republished on every tick, including the tick of the step that
    /// released the last request — which is the last tick there is, so a
    /// projection that skipped it would read a stale figure for as long as
    /// the load stayed idle.
    pub(crate) fn set_occupancy(&self, occupancy: ignis_core::Occupancy) {
        self.kv_pool_used_pages.store(u64::from(occupancy.kv_used_pages), Ordering::Relaxed);
        self.kv_ram_arena_used_bytes.store(occupancy.kv_ram_used_bytes, Ordering::Relaxed);
        self.kv_disk_used_bytes.store(occupancy.kv_disk_used_bytes, Ordering::Relaxed);
    }

    /// What this load reserved (GitHub #216, ADR 0030): written once, by the
    /// load that built the plan, before the first request is served.
    pub(crate) fn set_load_reservations(&self, reserved: LoadReservations) {
        for (slot, (_, bytes)) in self.vram_reserved.iter().zip(reserved.lines.entries()) {
            slot.store(bytes, Ordering::Relaxed);
        }
        self.flash_next_vram_reserved[0].store(reserved.residency_bytes, Ordering::Relaxed);
        self.flash_next_vram_reserved[1].store(reserved.expert_cache_bytes, Ordering::Relaxed);
        self.vram_budget_bytes.store(reserved.budget_bytes, Ordering::Relaxed);
        self.kv_pool_pages.store(u64::from(reserved.kv_pool_pages), Ordering::Relaxed);
        self.kv_page_bytes.store(reserved.kv_page_bytes, Ordering::Relaxed);
        self.kv_ram_arena_capacity_bytes.store(reserved.kv_ram_arena_bytes, Ordering::Relaxed);
        self.retained_slots_capacity.store(u64::from(reserved.retained_slots), Ordering::Relaxed);
        self.retained_host_slots.store(u64::from(reserved.retained_host_slots), Ordering::Relaxed);
        self.retained_host_bytes.store(reserved.retained_host_bytes, Ordering::Relaxed);
        self.kv_disk_loaded.store(u64::from(reserved.kv_disk_bytes.is_some()), Ordering::Relaxed);
        self.kv_disk_capacity_bytes.store(reserved.kv_disk_bytes.unwrap_or(0), Ordering::Relaxed);
    }

    /// A request's first token came `ms` after its submission.
    pub(crate) fn observe_ttft_ms(&self, ms: u64) {
        self.ttft.observe(ms);
    }

    /// A request completed `ms` after its submission.
    pub(crate) fn observe_duration_ms(&self, ms: u64) {
        self.duration.observe(ms);
    }

    /// A submission was accepted by the scheduler.
    pub(crate) fn record_accepted(&self) {
        self.accepted.fetch_add(1, Ordering::Relaxed);
    }

    /// A request completed, having generated `tokens`.
    pub(crate) fn record_completed(&self, tokens: u32) {
        self.completed.fetch_add(1, Ordering::Relaxed);
        self.generated_tokens.fetch_add(u64::from(tokens), Ordering::Relaxed);
    }

    /// A request in flight was dealt one generated token (GitHub #165): the
    /// live counterpart of the tokens `record_completed` adds only at the end.
    pub(crate) fn record_decoded_token(&self) {
        self.decoded_tokens.fetch_add(1, Ordering::Relaxed);
    }

    /// An accepted request was cancelled before it completed (its client
    /// went away).
    pub(crate) fn record_cancelled(&self) {
        self.cancelled.fetch_add(1, Ordering::Relaxed);
    }

    /// The Responses WebSockets open now (GitHub #282).
    pub(crate) fn set_responses_sockets(&self, open: u64) {
        self.responses_sockets.store(open, Ordering::Relaxed);
    }

    /// The socket requests waiting in the admission queue now (GitHub #282).
    pub(crate) fn set_responses_queued(&self, queued: u64) {
        self.responses_queued.store(queued, Ordering::Relaxed);
    }

    /// The scheduler's current request counts by observable state.
    pub(crate) fn set_scheduler_requests(&self, waiting: u32, running: u32) {
        self.waiting.store(u64::from(waiting), Ordering::Relaxed);
        self.running.store(u64::from(running), Ordering::Relaxed);
    }

    /// The Prometheus text exposition of the latest projection.
    pub fn render(&self) -> String {
        let read = |series: &AtomicU64| series.load(Ordering::Relaxed);
        let mut out = String::with_capacity(1024);
        declare(&mut out, "ignis_build_info", "gauge", "Constant build identity with value 1.");
        let _ = writeln!(
            out,
            "ignis_build_info{{version=\"{}\"}} 1",
            escape_label_value(env!("CARGO_PKG_VERSION"))
        );
        declare(
            &mut out,
            "ignis_scheduler_requests",
            "gauge",
            "Current requests by observable scheduler state.",
        );
        let _ = writeln!(out, "ignis_scheduler_requests{{state=\"waiting\"}} {}", read(&self.waiting));
        let _ = writeln!(out, "ignis_scheduler_requests{{state=\"running\"}} {}", read(&self.running));
        let counters = [
            ("ignis_requests_accepted_total", "Accepted submissions.", &self.accepted),
            ("ignis_requests_completed_total", "Completed requests.", &self.completed),
            (
                "ignis_requests_cancelled_total",
                "Accepted requests cancelled before completion.",
                &self.cancelled,
            ),
            (
                "ignis_generated_tokens_total",
                "Generated tokens on completed requests.",
                &self.generated_tokens,
            ),
            (
                "ignis_decoded_tokens_total",
                "Tokens generated so far, counted as each one is emitted.",
                &self.decoded_tokens,
            ),
            ("ignis_kv_cache_evictions_total", "Cumulative host-tier evictions.", &self.kv_evictions),
            // The other end of the same tier (#224): what KV-RAM itself
            // gave up. A live snapshot dropped here is the costliest
            // departure in the system — the request re-prefills from zero —
            // so it is its own series and never folded into the row above,
            // which counts arrivals at the tier.
            (
                "ignis_kv_ram_evictions_total",
                "Live host-tier snapshots dropped from KV-RAM to make room; the request re-prefills from the start.",
                &self.kv_ram_evictions,
            ),
            // Sibling-prefix reuse only, as ADR 0017's table row says. A
            // retained prefix is claimed through the same path (#188), and
            // `SchedEvent::PrefixReused::retained` is what keeps its tokens
            // out of here and in `ignis_retained_reused_tokens_total` (#190).
            (
                "ignis_prefix_reused_tokens_total",
                "Cumulative tokens skipped through sibling-prefix reuse.",
                &self.prefix_reused_tokens,
            ),
        ];
        for (name, help, series) in counters {
            declare(&mut out, name, "counter", help);
            let _ = writeln!(out, "{name} {}", read(series));
        }
        for (name, help, series) in [
            (
                "ignis_retained_reused_tokens_total",
                "Cumulative tokens skipped through retained state, by residency tier.",
                &self.retained_reused_tokens,
            ),
            (
                "ignis_retained_state_hits_total",
                "Retained state chosen to resume from or brought back, by residency tier.",
                &self.retained_state_hits,
            ),
            // `kind` is always `checkpoint` here, and that is the honest
            // shape: the checkpoint pool's lookup is the only one that
            // reports a miss, so the prefix series sits at zero rather than
            // carrying one invented to match it. ADR 0017 records this under
            // its metric table; emitting a prefix miss would be a change to
            // the fact stream, not to this projection (GitHub #216, #222).
            (
                "ignis_retained_state_misses_total",
                "First prefill chunks with no retained checkpoint matching in the tier.",
                &self.retained_state_misses,
            ),
            (
                "ignis_retained_state_spills_total",
                "Retained checkpoints and prefixes spilled into the tier.",
                &self.retained_state_spills,
            ),
            (
                "ignis_retained_state_discards_total",
                "Retained checkpoints and prefixes discarded from the tier.",
                &self.retained_state_discards,
            ),
            (
                "ignis_retained_state_restores_total",
                "Retained state restored from the tier.",
                &self.retained_state_restores,
            ),
        ] {
            declare(&mut out, name, "counter", help);
            // Spec vram-budget/03: the disk's rows only on a load that has
            // the tier, as its own families below.
            let disk = read(&self.kv_disk_loaded) != 0;
            for (source, tier) in ReuseSource::ALL.iter().zip(series) {
                if *source == ReuseSource::Disk && !disk {
                    continue;
                }
                for (kind, slot) in RetainedKind::ALL.iter().zip(tier) {
                    let _ = writeln!(
                        out,
                        "{name}{{tier=\"{}\",kind=\"{}\"}} {}",
                        source.as_str(),
                        kind.as_str(),
                        read(slot)
                    );
                }
            }
        }
        declare(
            &mut out,
            "ignis_retained_slot_skips_total",
            "counter",
            "Publishes and captures that found no retained slot or no tail page.",
        );
        for (skip, series) in RetainedSkip::ALL.iter().zip(&self.retained_slot_skips) {
            let _ = writeln!(
                out,
                "ignis_retained_slot_skips_total{{reason=\"{}\"}} {}",
                skip.as_str(),
                read(series)
            );
        }
        // What the load reserved, and what is occupied of it (GitHub #216,
        // ADR 0030 §Observability). Bytes, pages and slots — never a
        // percentage, which would hide which of its two terms moved.
        declare(
            &mut out,
            "ignis_vram_reserved_bytes",
            "gauge",
            "Device bytes this load reserved, by the plan line that reserved them.",
        );
        let names = ignis_core::VramLines::default().entries();
        for ((line, _), series) in names.iter().zip(&self.vram_reserved) {
            let _ = writeln!(out, "ignis_vram_reserved_bytes{{line=\"{line}\"}} {}", read(series));
        }
        for (line, series) in FLASH_NEXT_VRAM_LINES.iter().zip(&self.flash_next_vram_reserved) {
            let _ = writeln!(out, "ignis_vram_reserved_bytes{{line=\"{line}\"}} {}", read(series));
        }
        let plain_gauges = [
            (
                "ignis_vram_budget_bytes",
                "The device budget the plan was laid out inside.",
                &self.vram_budget_bytes,
            ),
            ("ignis_kv_pool_pages", "Pages the KV pool holds.", &self.kv_pool_pages),
            ("ignis_kv_page_bytes", "One KV page's bytes.", &self.kv_page_bytes),
            (
                "ignis_kv_pool_used_pages",
                "KV pool pages reserved by running requests and retained state.",
                &self.kv_pool_used_pages,
            ),
            (
                "ignis_retained_host_slots",
                "Of the retained slots, those whose images live in the pinned host block.",
                &self.retained_host_slots,
            ),
            (
                "ignis_retained_host_bytes",
                "The pinned host block holding the host retained slots' images.",
                &self.retained_host_bytes,
            ),
            (
                "ignis_responses_sockets",
                "Open Responses API WebSocket connections.",
                &self.responses_sockets,
            ),
            (
                "ignis_responses_queued_requests",
                "Responses WebSocket requests waiting in the server-wide admission queue.",
                &self.responses_queued,
            ),
        ];
        for (name, help, series) in plain_gauges {
            declare(&mut out, name, "gauge", help);
            let _ = writeln!(out, "{name} {}", read(series));
        }
        // Two states of one family: what bounds it, and what is in it. The
        // arena is bytes and says `used`; the slots are counted and say
        // `in_use` (ADR 0030 §Observability spells each).
        for (name, help, used_state, capacity, used) in [
            (
                "ignis_kv_ram_arena_bytes",
                "The pinned host KV-RAM arena: what it holds, and what is used of it.",
                "used",
                &self.kv_ram_arena_capacity_bytes,
                &self.kv_ram_arena_used_bytes,
            ),
            (
                "ignis_retained_slots",
                "Retained slots this load hands out, and how many hold an image.",
                "in_use",
                &self.retained_slots_capacity,
                &self.retained_slots_in_use,
            ),
        ] {
            declare(&mut out, name, "gauge", help);
            let _ = writeln!(out, "{name}{{state=\"capacity\"}} {}", read(capacity));
            let _ = writeln!(out, "{name}{{state=\"{used_state}\"}} {}", read(used));
        }
        // KV-disk (spec vram-budget/03, ADR 0045): every series from the
        // first scrape, zeros included, on a load with the tier -- and none
        // on a load without it.
        if read(&self.kv_disk_loaded) != 0 {
            declare(
                &mut out,
                "ignis_kv_disk_bytes",
                "gauge",
                "The KV-disk tier: its effective budget, and the bytes its files hold.",
            );
            let _ = writeln!(out, "ignis_kv_disk_bytes{{state=\"capacity\"}} {}", read(&self.kv_disk_capacity_bytes));
            let _ = writeln!(out, "ignis_kv_disk_bytes{{state=\"used\"}} {}", read(&self.kv_disk_used_bytes));
            declare(
                &mut out,
                "ignis_kv_disk_spills_total",
                "counter",
                "Live snapshots written to KV-disk, by the tier they came from.",
            );
            for (from, series) in [DiskSource::Device, DiskSource::KvRam].iter().zip(&self.kv_disk_spills) {
                let _ = writeln!(out, "ignis_kv_disk_spills_total{{from=\"{}\"}} {}", from.as_str(), read(series));
            }
            declare(
                &mut out,
                "ignis_kv_disk_failures_total",
                "counter",
                "KV-disk writes refused or failed (nothing lost), and files that failed their check on the way back.",
            );
            for (op, series) in [DiskOp::Write, DiskOp::Read].iter().zip(&self.kv_disk_failures) {
                let _ = writeln!(out, "ignis_kv_disk_failures_total{{op=\"{}\"}} {}", op.as_str(), read(series));
            }
        }
        declare(
            &mut out,
            "ignis_requests_rejected_total",
            "counter",
            "Rejected submissions by fixed reason.",
        );
        for (reason, series) in Rejection::ALL.iter().zip(&self.rejected) {
            let _ = writeln!(out, "ignis_requests_rejected_total{{reason=\"{}\"}} {}", reason.label(), read(series));
        }
        self.ttft.render(&mut out, "ignis_request_ttft_seconds", "Submission-to-first-token latency.");
        self.duration.render(&mut out, "ignis_request_duration_seconds", "Submission-to-completion latency.");
        // The decision family, **only once there has been a decision**
        // (GitHub #241). ADR 0017 exports zeros for everything else; this
        // follows its other rule instead — "only authoritative values are
        // exported" — because a load that serves no decisions is the normal
        // one, and three permanently-zero series plus an eleven-bucket
        // histogram on every scrape of every server would be clutter that
        // says nothing. A `rate()` over a series that appears mid-window is
        // handled by Prometheus the same way a new target is.
        if self.decisions.iter().map(read).sum::<u64>() > 0 {
            declare(
                &mut out,
                "ignis_decisions_total",
                "counter",
                "Questions answered by a readout, by typed primitive.",
            );
            for (primitive, series) in Primitive::ALL.iter().zip(&self.decisions) {
                let _ = writeln!(
                    out,
                    "ignis_decisions_total{{type=\"{}\"}} {}",
                    primitive.label(),
                    read(series)
                );
            }
            self.answer_mass.render(
                &mut out,
                "ignis_decision_answer_mass",
                "Share of the next-token distribution held by a decision's declared options.",
            );
        }
        // GitHub #278: absent until the first `locate`, for the decision
        // family's reason.
        if self.locates.iter().map(read).sum::<u64>() > 0 {
            declare(
                &mut out,
                "ignis_locates_total",
                "counter",
                "Locate questions answered, by resolved kind, method, compression and whether they found an answer.",
            );
            for ((kind, method, compression, found), series) in LOCATE_SERIES.iter().zip(&self.locates) {
                let _ = writeln!(
                    out,
                    "ignis_locates_total{{kind=\"{}\",method=\"{}\",compression=\"{}\",found=\"{}\"}} {}",
                    kind.label(),
                    method.label(),
                    compression.label(),
                    found.label(),
                    read(series)
                );
            }
        }
        // Spec server/08: absent until the thinking budget has forced a
        // close, for the decision family's reason — a load whose traffic
        // never reaches its budget is a normal one, and says nothing by
        // exporting a zero.
        let forced = read(&self.thinking_forced_closes);
        if forced > 0 {
            declare(
                &mut out,
                "ignis_thinking_forced_closes_total",
                "counter",
                "Completed requests whose reasoning block the thinking budget closed.",
            );
            let _ = writeln!(out, "ignis_thinking_forced_closes_total {forced}");
        }
        // GitHub #307: speculative decoding, absent until a request that ran
        // verify rounds ends (a load without speculation exports nothing of
        // it). Acceptance is accepted / drafted; per position, a draft at
        // position p counts in `drafted` when a round proposed it and in
        // `accepted` when the round committed it too.
        let rounds = read(&self.spec_rounds);
        if rounds > 0 {
            for (name, help, series) in [
                ("ignis_speculative_rounds_total", "Verify rounds run.", &self.spec_rounds),
                ("ignis_speculative_drafted_tokens_total", "Draft tokens verified.", &self.spec_drafted),
                ("ignis_speculative_accepted_tokens_total", "Draft tokens committed.", &self.spec_accepted),
            ] {
                declare(&mut out, name, "counter", help);
                let _ = writeln!(out, "{name} {}", read(series));
            }
            for (name, help, family) in [
                (
                    "ignis_speculative_position_drafted_total",
                    "Verify rounds that proposed a draft at the position.",
                    &self.spec_drafted_at,
                ),
                (
                    "ignis_speculative_position_accepted_total",
                    "Verify rounds that committed the draft at the position.",
                    &self.spec_accepted_at,
                ),
            ] {
                declare(&mut out, name, "counter", help);
                for (j, series) in family.iter().enumerate() {
                    let _ = writeln!(out, "{name}{{position=\"{}\"}} {}", j + 1, read(series));
                }
            }
        }
        // GitHub #285: absent until the first call, for the decision
        // family's reason.
        if self.tokenize_calls.iter().map(read).sum::<u64>() > 0 {
            declare(
                &mut out,
                "ignis_tokenize_requests_total",
                "counter",
                "Calls to the counting routes, which never enter the scheduler, by route.",
            );
            for (route, series) in TokenizeRoute::ALL.iter().zip(&self.tokenize_calls) {
                let _ = writeln!(out, "ignis_tokenize_requests_total{{route=\"{}\"}} {}", route.label(), read(series));
            }
        }
        // GitHub #301, #302: last, so a 27B load's exposition is the same
        // bytes it was before Flash-Next.
        if read(&self.flash_next_loaded) == 1 {
            let reading = self.flash_next.load();
            render_expert_residency(&mut out, &reading.residency, &reading.slots_capacity, &reading.slots_in_use);
            render_ngram(&mut out, &reading.ngram);
        }
        out
    }
}

/// Flash-Next's expert residency (spec flash-next/03, GitHub #301): the
/// families a Flash-Next load exports from the facts residency keeps
/// ([`ResidencyCounters`]) and its K-class pools' slot capacity and
/// occupancy (indexed like [`KClass::ALL`]). Fixed cardinality -- `class`
/// takes the eight K-class spellings, `phase` `decode|prefill`, `state`
/// `capacity|in_use` -- with zeros exported. Counters and gauges in their
/// own units, no ratio: a decode hit rate is `hits` over `hits + misses` of
/// `phase="decode"`. A 27B load has no expert cache and never renders these;
/// [`Metrics::render`] renders them for a Flash-Next load, from what the
/// telemetry consumer read. The kernels' wait is the device's own timing of
/// its demand copies (`stall_nanos`), rendered exactly in seconds.
pub fn render_expert_residency(
    out: &mut String,
    counters: &ResidencyCounters,
    capacity: &[u32; KClass::COUNT],
    occupancy: &[u32; KClass::COUNT],
) {
    for (name, help, series) in [
        (
            "ignis_expert_cache_hits_total",
            "Selected expert projections already resident in the VRAM expert cache, or staged for their layer, by K class and phase.",
            &counters.hits,
        ),
        (
            "ignis_expert_cache_misses_total",
            "Selected expert projections copied in from the pinned host pool by their own step, by K class and phase.",
            &counters.misses,
        ),
    ] {
        declare(out, name, "counter", help);
        for class in KClass::ALL {
            for phase in Phase::ALL {
                let _ = writeln!(
                    out,
                    "{name}{{class=\"{}\",phase=\"{}\"}} {}",
                    class.as_str(),
                    phase.as_str(),
                    series[class.index()][phase.index()]
                );
            }
        }
    }
    for (name, help, value) in [
        (
            "ignis_expert_prefetches_issued_total",
            "Expert projections copied ahead for the next layer by the router lookahead.",
            counters.prefetch_issued,
        ),
        (
            "ignis_expert_prefetches_used_total",
            "Prefetched expert projections at their first use, whenever it comes.",
            counters.prefetch_used,
        ),
    ] {
        declare(out, name, "counter", help);
        let _ = writeln!(out, "{name} {value}");
    }
    declare(
        out,
        "ignis_expert_bytes_moved_total",
        "counter",
        "Bytes of expert projections copied host-to-device, misses and prefetches, by phase.",
    );
    for phase in Phase::ALL {
        let _ = writeln!(
            out,
            "ignis_expert_bytes_moved_total{{phase=\"{}\"}} {}",
            phase.as_str(),
            counters.bytes_moved[phase.index()]
        );
    }
    declare(
        out,
        "ignis_expert_residency_stall_seconds_total",
        "counter",
        "Device time the expert kernels waited on their step's demand copies (the misses), by phase; a step with no miss adds nothing.",
    );
    for phase in Phase::ALL {
        let _ = writeln!(
            out,
            "ignis_expert_residency_stall_seconds_total{{phase=\"{}\"}} {}",
            phase.as_str(),
            decimal(counters.stall_nanos[phase.index()], 1_000_000_000)
        );
    }
    declare(
        out,
        "ignis_expert_cache_slots",
        "gauge",
        "VRAM expert cache slots per K class: capacity reserved at load, and in use.",
    );
    for class in KClass::ALL {
        for (state, slots) in [("capacity", capacity), ("in_use", occupancy)] {
            let _ = writeln!(
                out,
                "ignis_expert_cache_slots{{class=\"{}\",state=\"{state}\"}} {}",
                class.as_str(),
                slots[class.index()]
            );
        }
    }
}

/// Flash-Next's n-gram rows (spec flash-next/04, GitHub #302): the rows its
/// steps staged, by where they came from -- the RAM hot-row cache, or the
/// artifact file -- and the reads issued to the file with their bytes. Fixed
/// cardinality, zeros exported; a hot-row hit rate is `source="hot"` over the
/// sum. A 27B load has no n-gram table and never renders these.
pub fn render_ngram(out: &mut String, counters: &NgramCounters) {
    declare(
        out,
        "ignis_ngram_rows_total",
        "counter",
        "N-gram embedding rows staged for prefill spans and decode rounds, by source: the RAM hot-row cache or the artifact file.",
    );
    for (source, rows) in [("hot", counters.hot_rows), ("file", counters.file_rows)] {
        let _ = writeln!(out, "ignis_ngram_rows_total{{source=\"{source}\"}} {rows}");
    }
    for (name, help, value) in [
        (
            "ignis_ngram_reads_total",
            "Reads issued to the artifact file for n-gram rows the hot-row cache did not hold.",
            counters.reads,
        ),
        (
            "ignis_ngram_read_bytes_total",
            "Bytes of the n-gram reads issued to the artifact file.",
            counters.read_bytes,
        ),
    ] {
        declare(out, name, "counter", help);
        let _ = writeln!(out, "{name} {value}");
    }
}

/// A metric's `HELP` and `TYPE` lines.
fn declare(out: &mut String, name: &str, kind: &str, help: &str) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {kind}");
}

/// A label value escaped for the text format: backslash, double quote and
/// line feed.
fn escape_label_value(value: &str) -> String {
    value.replace('\\', r"\\").replace('"', "\\\"").replace('\n', r"\n")
}

/// `GET path` over `metrics`: `/metrics` for the metrics listener,
/// `/ui/metrics` for the Playground's copy on the API listener.
pub fn router<S>(path: &str, metrics: Arc<Metrics>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new().route(
        path,
        get(move || async move { ([(header::CONTENT_TYPE, CONTENT_TYPE)], metrics.render()) }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sample lines of `text`: `(name, labels, value)`, comments skipped.
    fn samples(text: &str) -> Vec<(String, String, String)> {
        text.lines()
            .filter(|line| !line.starts_with('#') && !line.is_empty())
            .map(|line| {
                let (series, value) = line.rsplit_once(' ').expect("`series value`");
                let (name, labels) = match series.split_once('{') {
                    Some((name, labels)) => (name, labels.trim_end_matches('}')),
                    None => (series, ""),
                };
                (name.to_owned(), labels.to_owned(), value.to_owned())
            })
            .collect()
    }

    fn value(text: &str, name: &str, labels: &str) -> String {
        samples(text)
            .into_iter()
            .find(|(n, l, _)| n == name && l == labels)
            .unwrap_or_else(|| panic!("no `{name}{{{labels}}}` in:\n{text}"))
            .2
    }

    #[test]
    fn every_metric_is_declared_once_with_help_and_type_before_its_samples() {
        declared_once(&Metrics::new().render(), &[]);
        // And the decision family, which only exists once a decision has
        // been served (GitHub #241) — its declarations have to be as
        // well-formed as the ones that are always there.
        let served = Metrics::new();
        served.record_decision(Primitive::Choice, Some(0.998));
        declared_once(
            &served.render(),
            &[("ignis_decisions_total", "counter"), ("ignis_decision_answer_mass", "histogram")],
        );
        // And the locate family (GitHub #278), on the same terms.
        let located = Metrics::new();
        located.record_locate(Kind::Prose, Method::Shortlist, Compression::None, LocateFound::False);
        declared_once(&located.render(), &[("ignis_locates_total", "counter")]);
        // And the forced-close counter (spec server/08), on the same terms.
        let forced = Metrics::new();
        forced.record_thinking_forced_close();
        declared_once(&forced.render(), &[("ignis_thinking_forced_closes_total", "counter")]);
        // And the counting routes' counter (GitHub #285).
        let counted = Metrics::new();
        counted.record_tokenize(TokenizeRoute::Tokenize);
        declared_once(&counted.render(), &[("ignis_tokenize_requests_total", "counter")]);
    }

    /// GitHub #285: absent until a counting route is called, then both routes
    /// have a series and a call moves exactly its own.
    #[test]
    fn the_tokenize_counter_is_absent_until_a_call_and_counts_by_route() {
        let metrics = Metrics::new();
        assert!(!metrics.render().contains("ignis_tokenize_requests_total"));
        metrics.record_tokenize(TokenizeRoute::Tokenize);
        metrics.record_tokenize(TokenizeRoute::Tokenize);
        let text = metrics.render();
        assert_eq!(value(&text, "ignis_tokenize_requests_total", "route=\"tokenize\""), "2", "{text}");
        assert_eq!(value(&text, "ignis_tokenize_requests_total", "route=\"detokenize\""), "0", "{text}");
    }

    /// GitHub #278: `ignis_locates_total` is absent until the first `locate`,
    /// then every route the endpoint serves has its series, and a locate
    /// moves exactly its own.
    #[test]
    fn the_locate_counter_is_absent_until_a_locate_and_counts_by_route() {
        let metrics = Metrics::new();
        assert!(!metrics.render().contains("ignis_locates_total"));
        metrics.record_locate(Kind::Log, Method::Shortlist, Compression::TemplateFold, LocateFound::True);
        metrics.record_locate(Kind::Log, Method::Shortlist, Compression::TemplateFold, LocateFound::True);
        metrics.record_locate(Kind::Records, Method::Vote, Compression::None, LocateFound::Unmeasured);
        let text = metrics.render();
        let series: Vec<&str> = text.lines().filter(|line| line.starts_with("ignis_locates_total{")).collect();
        assert_eq!(series.len(), LOCATE_SERIES.len(), "{text}");
        assert!(text.contains(
            "ignis_locates_total{kind=\"log\",method=\"shortlist\",compression=\"template_fold\",found=\"true\"} 2"
        ));
        assert!(text.contains("ignis_locates_total{kind=\"records\",method=\"vote\",compression=\"none\",found=\"unmeasured\"} 1"));
        assert!(text.contains("ignis_locates_total{kind=\"prose\",method=\"shortlist\",compression=\"none\",found=\"false\"} 0"));
    }

    /// GitHub #307: the speculative family is absent until a request that
    /// ran verify rounds ends, then sums the requests' counters.
    #[test]
    fn the_speculative_counters_sum_the_requests_verify_rounds() {
        let metrics = Metrics::new();
        assert!(!metrics.render().contains("ignis_speculative_rounds_total"));
        let spec = ignis_core::types::SpecCounters::round(3, 2) + ignis_core::types::SpecCounters::round(3, 0);
        metrics.record_speculation(&spec);
        metrics.record_speculation(&ignis_core::types::SpecCounters::round(1, 1));
        let text = metrics.render();
        declared_once(
            &text,
            &[
                ("ignis_speculative_rounds_total", "counter"),
                ("ignis_speculative_drafted_tokens_total", "counter"),
                ("ignis_speculative_accepted_tokens_total", "counter"),
                ("ignis_speculative_position_drafted_total", "counter"),
                ("ignis_speculative_position_accepted_total", "counter"),
            ],
        );
        assert_eq!(value(&text, "ignis_speculative_rounds_total", ""), "3");
        assert_eq!(value(&text, "ignis_speculative_drafted_tokens_total", ""), "7");
        assert_eq!(value(&text, "ignis_speculative_accepted_tokens_total", ""), "3");
        assert_eq!(value(&text, "ignis_speculative_position_drafted_total", "position=\"1\""), "3");
        assert_eq!(value(&text, "ignis_speculative_position_accepted_total", "position=\"1\""), "2");
        assert_eq!(value(&text, "ignis_speculative_position_accepted_total", "position=\"2\""), "1");
        assert_eq!(value(&text, "ignis_speculative_position_drafted_total", "position=\"3\""), "2");
    }

    /// A load's reservations with nothing in them but KV-disk's budget.
    fn with_disk(kv_disk_bytes: Option<u64>) -> LoadReservations {
        LoadReservations {
            lines: ignis_core::VramLines::default(),
            budget_bytes: 0,
            kv_pool_pages: 0,
            kv_page_bytes: 0,
            residency_bytes: 0,
            expert_cache_bytes: 0,
            kv_ram_arena_bytes: 0,
            retained_slots: 0,
            retained_host_slots: 0,
            retained_host_bytes: 0,
            kv_disk_bytes,
            flash_next: None,
        }
    }

    /// Spec vram-budget/03 AC 38: on a load with KV-disk, its three families
    /// and the retained families' `tier="disk"` rows render from the first
    /// scrape, zeros included; on a load without it, none of them ever does.
    #[test]
    fn the_disk_tier_renders_from_the_first_scrape_only_on_a_load_that_has_it() {
        let without = Metrics::new();
        without.set_load_reservations(with_disk(None));
        let text = without.render();
        assert!(!text.contains("ignis_kv_disk_"), "{text}");
        assert!(!text.contains("tier=\"disk\""), "{text}");

        let with = Metrics::new();
        with.set_load_reservations(with_disk(Some(16 << 30)));
        let text = with.render();
        declared_once(
            &text,
            &[
                ("ignis_kv_disk_bytes", "gauge"),
                ("ignis_kv_disk_spills_total", "counter"),
                ("ignis_kv_disk_failures_total", "counter"),
            ],
        );
        assert_eq!(value(&text, "ignis_kv_disk_bytes", "state=\"capacity\""), "17179869184");
        assert_eq!(value(&text, "ignis_kv_disk_bytes", "state=\"used\""), "0");
        for from in ["device", "kv_ram"] {
            assert_eq!(value(&text, "ignis_kv_disk_spills_total", &format!("from=\"{from}\"")), "0");
        }
        for op in ["write", "read"] {
            assert_eq!(value(&text, "ignis_kv_disk_failures_total", &format!("op=\"{op}\"")), "0");
        }
        for name in [
            "ignis_retained_reused_tokens_total",
            "ignis_retained_state_hits_total",
            "ignis_retained_state_misses_total",
            "ignis_retained_state_spills_total",
            "ignis_retained_state_discards_total",
            "ignis_retained_state_restores_total",
        ] {
            for kind in ["checkpoint", "prefix"] {
                assert_eq!(value(&text, name, &format!("tier=\"disk\",kind=\"{kind}\"")), "0", "{name}");
            }
        }

        with.record_retained_state(RetainedStateOperation::Spill, ReuseSource::Disk, RetainedKind::Checkpoint);
        with.record_retained_reused(ReuseSource::Disk, RetainedKind::Checkpoint, 9000);
        with.set_occupancy(ignis_core::Occupancy {
            kv_used_pages: 0,
            kv_pool_pages: 0,
            kv_ram_used_bytes: 0,
            kv_disk_used_bytes: 5 << 30,
        });
        let text = with.render();
        assert_eq!(value(&text, "ignis_retained_state_spills_total", "tier=\"disk\",kind=\"checkpoint\""), "1");
        assert_eq!(value(&text, "ignis_retained_reused_tokens_total", "tier=\"disk\",kind=\"checkpoint\""), "9000");
        assert_eq!(value(&text, "ignis_kv_disk_bytes", "state=\"used\""), "5368709120");
    }

    #[test]
    fn the_forced_close_counter_is_absent_until_a_close_is_forced() {
        let metrics = Metrics::new();
        assert!(!metrics.render().contains("ignis_thinking_forced_closes_total"));
        metrics.record_thinking_forced_close();
        metrics.record_thinking_forced_close();
        assert_eq!(value(&metrics.render(), "ignis_thinking_forced_closes_total", ""), "2");
    }

    /// Every metric in the base contract, plus `also`, declared once with
    /// `HELP` and `TYPE` before its first sample.
    fn declared_once(text: &str, also: &[(&str, &str)]) {
        let expected = [
            ("ignis_build_info", "gauge"),
            ("ignis_scheduler_requests", "gauge"),
            ("ignis_requests_accepted_total", "counter"),
            ("ignis_requests_completed_total", "counter"),
            ("ignis_requests_cancelled_total", "counter"),
            ("ignis_generated_tokens_total", "counter"),
            ("ignis_decoded_tokens_total", "counter"),
            ("ignis_kv_cache_evictions_total", "counter"),
            ("ignis_kv_ram_evictions_total", "counter"),
            ("ignis_prefix_reused_tokens_total", "counter"),
            ("ignis_retained_reused_tokens_total", "counter"),
            ("ignis_retained_state_hits_total", "counter"),
            ("ignis_retained_state_misses_total", "counter"),
            ("ignis_retained_state_spills_total", "counter"),
            ("ignis_retained_state_discards_total", "counter"),
            ("ignis_retained_state_restores_total", "counter"),
            ("ignis_retained_slot_skips_total", "counter"),
            ("ignis_vram_reserved_bytes", "gauge"),
            ("ignis_vram_budget_bytes", "gauge"),
            ("ignis_kv_pool_pages", "gauge"),
            ("ignis_kv_page_bytes", "gauge"),
            ("ignis_kv_pool_used_pages", "gauge"),
            ("ignis_kv_ram_arena_bytes", "gauge"),
            ("ignis_retained_slots", "gauge"),
            ("ignis_retained_host_slots", "gauge"),
            ("ignis_retained_host_bytes", "gauge"),
            ("ignis_responses_sockets", "gauge"),
            ("ignis_responses_queued_requests", "gauge"),
            ("ignis_requests_rejected_total", "counter"),
            ("ignis_request_ttft_seconds", "histogram"),
            ("ignis_request_duration_seconds", "histogram"),
        ];
        let lines: Vec<&str> = text.lines().collect();
        for (name, kind) in expected.iter().chain(also).copied() {
            let help = lines
                .iter()
                .position(|l| l.starts_with(&format!("# HELP {name} ")))
                .unwrap_or_else(|| panic!("no HELP for {name}:\n{text}"));
            assert_eq!(lines[help + 1], format!("# TYPE {name} {kind}"), "{text}");
            let first_sample = lines
                .iter()
                .position(|l| {
                    l.starts_with(&format!("{name} "))
                        || l.starts_with(&format!("{name}{{"))
                        || (kind == "histogram" && l.starts_with(&format!("{name}_bucket{{")))
                })
                .unwrap_or_else(|| panic!("no sample for {name}:\n{text}"));
            assert!(first_sample > help + 1, "{name}'s samples follow its TYPE:\n{text}");
            assert_eq!(
                lines.iter().filter(|l| l.starts_with(&format!("# TYPE {name} "))).count(),
                1,
                "{text}"
            );
        }
        assert!(text.ends_with('\n'), "the exposition ends with a line feed");
    }

    #[test]
    fn a_fresh_projection_reports_build_identity_and_zeros() {
        let text = Metrics::new().render();
        let version = format!("version=\"{}\"", env!("CARGO_PKG_VERSION"));
        assert_eq!(value(&text, "ignis_build_info", &version), "1");
        assert_eq!(value(&text, "ignis_scheduler_requests", "state=\"waiting\""), "0");
        assert_eq!(value(&text, "ignis_scheduler_requests", "state=\"running\""), "0");
        for name in [
            "ignis_requests_accepted_total",
            "ignis_requests_completed_total",
            "ignis_requests_cancelled_total",
            "ignis_generated_tokens_total",
            "ignis_decoded_tokens_total",
            "ignis_kv_cache_evictions_total",
            "ignis_kv_ram_evictions_total",
            "ignis_prefix_reused_tokens_total",
        ] {
            assert_eq!(value(&text, name, ""), "0", "{name}");
        }
        for reason in ["full", "unknown_model", "oversized"] {
            assert_eq!(value(&text, "ignis_requests_rejected_total", &format!("reason=\"{reason}\"")), "0");
        }
    }

    #[test]
    fn a_submit_error_counts_under_its_fixed_reason() {
        assert_eq!(Rejection::of(&SubmitError::Full), Rejection::Full);
        assert_eq!(Rejection::of(&SubmitError::UnknownModel("x".into())), Rejection::UnknownModel);
        assert_eq!(Rejection::of(&SubmitError::Oversized), Rejection::Oversized);
        assert_eq!(
            Rejection::of(&SubmitError::ContextExceeded { requested: 9000, limit: 8192 }),
            Rejection::Oversized
        );
    }

    #[test]
    fn a_scaled_integer_is_rendered_without_trailing_zeros() {
        // Milliseconds as seconds, the latency histograms' unit.
        assert_eq!(decimal(0, THOUSANDTHS), "0");
        assert_eq!(decimal(50, THOUSANDTHS), "0.05");
        assert_eq!(decimal(2_500, THOUSANDTHS), "2.5");
        assert_eq!(decimal(600_000, THOUSANDTHS), "600");
        assert_eq!(decimal(1_001, THOUSANDTHS), "1.001");
        // And millionths as a ratio, which needs the width to come from the
        // scale rather than from a literal: `{frac:03}` would render
        // 998,002 millionths as "0.998002" by luck and 2 as "0.2".
        assert_eq!(decimal(1_000_000, MILLIONTHS), "1");
        assert_eq!(decimal(998_002, MILLIONTHS), "0.998002");
        assert_eq!(decimal(999_500, MILLIONTHS), "0.9995");
        assert_eq!(decimal(2, MILLIONTHS), "0.000002");
        assert_eq!(decimal(3_992_008, MILLIONTHS), "3.992008");
    }

    #[test]
    fn recorded_facts_move_their_series() {
        let metrics = Metrics::new();
        metrics.record_accepted();
        metrics.record_accepted();
        metrics.record_completed(7);
        metrics.record_completed(5);
        metrics.record_cancelled();
        metrics.record_decoded_token();
        metrics.record_decoded_token();
        metrics.set_scheduler_requests(3, 4);
        metrics.set_scheduler_requests(1, 2);

        let text = metrics.render();
        assert_eq!(value(&text, "ignis_requests_accepted_total", ""), "2");
        assert_eq!(value(&text, "ignis_requests_completed_total", ""), "2");
        assert_eq!(value(&text, "ignis_requests_cancelled_total", ""), "1");
        assert_eq!(value(&text, "ignis_generated_tokens_total", ""), "12");
        // Decoded tokens are their own series, not derived from completions.
        assert_eq!(value(&text, "ignis_decoded_tokens_total", ""), "2");
        // Gauges are the latest state, not a sum.
        assert_eq!(value(&text, "ignis_scheduler_requests", "state=\"waiting\""), "1");
        assert_eq!(value(&text, "ignis_scheduler_requests", "state=\"running\""), "2");
    }

    #[test]
    fn the_operational_counters_move_with_their_facts() {
        let metrics = Metrics::new();
        metrics.record_eviction();
        metrics.record_eviction();
        metrics.record_kv_ram_eviction();
        metrics.record_prefix_reused(32);
        metrics.record_prefix_reused(64);
        metrics.record_rejected(Rejection::Full);
        metrics.record_rejected(Rejection::Oversized);
        metrics.record_rejected(Rejection::Oversized);

        let text = metrics.render();
        assert_eq!(value(&text, "ignis_kv_cache_evictions_total", ""), "2");
        // The two tiers' evictions never bleed into each other: two
        // snapshots arrived at KV-RAM, one live snapshot left it (#224).
        assert_eq!(value(&text, "ignis_kv_ram_evictions_total", ""), "1");
        assert_eq!(value(&text, "ignis_prefix_reused_tokens_total", ""), "96");
        assert_eq!(value(&text, "ignis_requests_rejected_total", "reason=\"full\""), "1");
        assert_eq!(value(&text, "ignis_requests_rejected_total", "reason=\"unknown_model\""), "0");
        assert_eq!(value(&text, "ignis_requests_rejected_total", "reason=\"oversized\""), "2");
    }

    #[test]
    fn retained_state_operations_are_counted_per_residency_tier_and_kind() {
        let metrics = Metrics::new();
        let checkpoint = RetainedKind::Checkpoint;
        metrics.record_retained_state(RetainedStateOperation::Hit, ReuseSource::Device, checkpoint);
        metrics.record_retained_state(RetainedStateOperation::Miss, ReuseSource::Device, checkpoint);
        metrics.record_retained_state(RetainedStateOperation::Spill, ReuseSource::KvRam, checkpoint);
        metrics.record_retained_state(RetainedStateOperation::Discard, ReuseSource::KvRam, checkpoint);
        metrics.record_retained_state(RetainedStateOperation::Restore, ReuseSource::KvRam, checkpoint);

        let text = metrics.render();
        for (name, tier) in [
            ("ignis_retained_state_hits_total", "device"),
            ("ignis_retained_state_misses_total", "device"),
            ("ignis_retained_state_spills_total", "kv_ram"),
            ("ignis_retained_state_discards_total", "kv_ram"),
            ("ignis_retained_state_restores_total", "kv_ram"),
        ] {
            assert_eq!(
                value(&text, name, &format!("tier=\"{tier}\",kind=\"checkpoint\"")),
                "1",
                "{name}"
            );
            // The other kind is its own series and did not move with it —
            // the whole point of the split (GitHub #216).
            assert_eq!(
                value(&text, name, &format!("tier=\"{tier}\",kind=\"prefix\"")),
                "0",
                "{name}"
            );
        }
        assert_eq!(
            value(&text, "ignis_retained_state_hits_total", "tier=\"kv_ram\",kind=\"checkpoint\""),
            "0"
        );
    }

    #[test]
    fn a_prefix_and_a_checkpoint_in_the_same_tier_are_never_summed_into_one_series() {
        let metrics = Metrics::new();
        metrics.record_retained_state(
            RetainedStateOperation::Spill,
            ReuseSource::KvRam,
            RetainedKind::Checkpoint,
        );
        for _ in 0..3 {
            metrics.record_retained_state(
                RetainedStateOperation::Spill,
                ReuseSource::KvRam,
                RetainedKind::Prefix,
            );
        }
        metrics.record_retained_reused(ReuseSource::Device, RetainedKind::Prefix, 64);
        metrics.record_retained_reused(ReuseSource::KvRam, RetainedKind::Checkpoint, 1536);

        let text = metrics.render();
        let spills = |kind: &str| {
            value(&text, "ignis_retained_state_spills_total", &format!("tier=\"kv_ram\",kind=\"{kind}\""))
        };
        assert_eq!(spills("checkpoint"), "1");
        assert_eq!(spills("prefix"), "3", "four spills, and the load shed prefixes");
        assert_eq!(
            value(&text, "ignis_retained_reused_tokens_total", "tier=\"device\",kind=\"prefix\""),
            "64"
        );
        assert_eq!(
            value(&text, "ignis_retained_reused_tokens_total", "tier=\"kv_ram\",kind=\"checkpoint\""),
            "1536"
        );
        // Sibling-prefix reuse keeps its own meaning and takes no kind.
        assert_eq!(value(&text, "ignis_prefix_reused_tokens_total", ""), "0");
    }

    #[test]
    fn the_reservations_are_exported_with_the_plan_own_line_spellings() {
        let metrics = Metrics::new();
        // A fresh projection reserves nothing and holds nothing.
        let empty = metrics.render();
        for (line, _) in ignis_core::VramLines::default().entries() {
            assert_eq!(value(&empty, "ignis_vram_reserved_bytes", &format!("line=\"{line}\"")), "0");
        }
        assert_eq!(value(&empty, "ignis_kv_pool_pages", ""), "0");

        let mut lines = ignis_core::VramLines::default();
        lines.weights = 21_000_000_000;
        lines.cuda_context = 600_000_000;
        lines.retained_slots = 1_000_000_000;
        metrics.set_load_reservations(LoadReservations {
            lines,
            budget_bytes: 30_000_000_000,
            kv_pool_pages: 5_000,
            kv_page_bytes: 1_048_576,
            residency_bytes: 2_000_000_000,
            expert_cache_bytes: 15_000_000_000,
            kv_ram_arena_bytes: 8 << 30,
            retained_slots: 9,
            retained_host_slots: 7,
            retained_host_bytes: 7 * 232_532_224,
            kv_disk_bytes: None,
            flash_next: None,
        });

        let text = metrics.render();
        // The twelve lines are the plan's, in the plan's order and spelling,
        // then Flash-Next's two (GitHub #306).
        let exported: Vec<String> = samples(&text)
            .into_iter()
            .filter(|(name, _, _)| name == "ignis_vram_reserved_bytes")
            .map(|(_, labels, _)| labels)
            .collect();
        let expected: Vec<String> = ignis_core::VramLines::default()
            .entries()
            .iter()
            .map(|(line, _)| format!("line=\"{line}\""))
            .chain(["line=\"residency\"".to_owned(), "line=\"expert_cache\"".to_owned()])
            .collect();
        assert_eq!(exported, expected, "{text}");
        assert_eq!(value(&text, "ignis_vram_reserved_bytes", "line=\"residency\""), "2000000000");
        assert_eq!(value(&text, "ignis_vram_reserved_bytes", "line=\"expert_cache\""), "15000000000");
        assert_eq!(value(&empty, "ignis_vram_reserved_bytes", "line=\"expert_cache\""), "0");
        assert_eq!(value(&text, "ignis_vram_reserved_bytes", "line=\"weights\""), "21000000000");
        assert_eq!(value(&text, "ignis_vram_reserved_bytes", "line=\"retained_slots\""), "1000000000");
        assert_eq!(value(&text, "ignis_vram_budget_bytes", ""), "30000000000");
        assert_eq!(value(&text, "ignis_kv_pool_pages", ""), "5000");
        assert_eq!(value(&text, "ignis_kv_page_bytes", ""), "1048576");
        assert_eq!(value(&text, "ignis_kv_ram_arena_bytes", "state=\"capacity\""), "8589934592");
        assert_eq!(value(&text, "ignis_retained_slots", "state=\"capacity\""), "9");
        // GitHub #281: of the nine, the host ones and their pinned block --
        // host memory, beside the VRAM lines rather than one of them.
        assert_eq!(value(&text, "ignis_retained_host_slots", ""), "7");
        assert_eq!(value(&text, "ignis_retained_host_bytes", ""), "1627725568");
        // Nothing is occupied until a step reports occupancy.
        assert_eq!(value(&text, "ignis_kv_pool_used_pages", ""), "0");
        assert_eq!(value(&text, "ignis_kv_ram_arena_bytes", "state=\"used\""), "0");
        assert_eq!(value(&text, "ignis_retained_slots", "state=\"in_use\""), "0");
    }

    #[test]
    fn occupancy_and_slot_skips_follow_the_facts_and_come_back_to_zero() {
        let metrics = Metrics::new();
        metrics.set_occupancy(ignis_core::Occupancy {
            kv_used_pages: 412,
            kv_pool_pages: 1_000,
            kv_ram_used_bytes: 3 << 30,
                kv_disk_used_bytes: 0,
});
        metrics.set_retained_slots_in_use(4);
        metrics.record_retained_slot_skip(RetainedSkip::PublishNoSlot);
        metrics.record_retained_slot_skip(RetainedSkip::CaptureNoPage);
        metrics.record_retained_slot_skip(RetainedSkip::CaptureNoPage);

        let text = metrics.render();
        assert_eq!(value(&text, "ignis_kv_pool_used_pages", ""), "412");
        assert_eq!(value(&text, "ignis_kv_ram_arena_bytes", "state=\"used\""), "3221225472");
        assert_eq!(value(&text, "ignis_retained_slots", "state=\"in_use\""), "4");
        for (reason, count) in [
            ("publish_skipped_no_slot", "1"),
            ("capture_skipped_no_slot", "0"),
            ("capture_skipped_no_page", "2"),
        ] {
            assert_eq!(
                value(&text, "ignis_retained_slot_skips_total", &format!("reason=\"{reason}\"")),
                count
            );
        }

        // The last step of a load releases everything, and reports it: these
        // are gauges of the latest reading, not high-water marks.
        metrics.set_occupancy(ignis_core::Occupancy { kv_pool_pages: 1_000, ..Default::default() });
        metrics.set_retained_slots_in_use(0);
        let text = metrics.render();
        assert_eq!(value(&text, "ignis_kv_pool_used_pages", ""), "0");
        assert_eq!(value(&text, "ignis_kv_ram_arena_bytes", "state=\"used\""), "0");
        assert_eq!(value(&text, "ignis_retained_slots", "state=\"in_use\""), "0");
        // A counter does not come back: the skips already happened.
        assert_eq!(
            value(&text, "ignis_retained_slot_skips_total", "reason=\"capture_skipped_no_page\""),
            "2"
        );
    }

    #[test]
    fn no_exported_series_is_a_percentage() {
        // ADR 0030: every memory series is bytes, pages or slots, so a
        // reader always has both terms and never a ratio that hides which
        // of them moved.
        //
        // Both shapes of exposition, because the decision family is absent
        // from a fresh one (GitHub #241) and it is the one series here that
        // *is* a ratio — answer mass has no second term, which is the ADR's
        // own distinction and worth having under a test.
        let served = Metrics::new();
        served.record_decision(Primitive::Score, Some(0.996));
        for text in [Metrics::new().render(), served.render()] {
            for line in text.lines().filter(|l| l.starts_with("# HELP ignis_")) {
                let name = line.split_whitespace().nth(2).expect("# HELP <name> <help>");
                assert!(
                    !name.contains("_pct") && !name.contains("percent") && !name.contains("_ratio"),
                    "{name} is a percentage"
                );
            }
        }
    }

    #[test]
    fn the_decision_family_is_absent_until_a_decision_is_served() {
        // Spec 05's acceptance 3: no zero-valued clutter for a server that
        // never sees a decision, which is most of them.
        let metrics = Metrics::new();
        let text = metrics.render();
        assert!(!text.contains("ignis_decisions_total"), "{text}");
        assert!(!text.contains("ignis_decision_answer_mass"), "{text}");

        metrics.record_decision(Primitive::Noul, Some(0.9983));
        let text = metrics.render();
        assert_eq!(value(&text, "ignis_decisions_total", "type=\"noul\""), "1");
        // The other two primitives appear at zero *now*, because the family
        // exists: a label value that vanishes when its count is zero is a
        // series that breaks `sum by (type)` the moment traffic shifts.
        assert_eq!(value(&text, "ignis_decisions_total", "type=\"choice\""), "0");
        assert_eq!(value(&text, "ignis_decisions_total", "type=\"score\""), "0");
        assert_eq!(value(&text, "ignis_decision_answer_mass_count", ""), "1");
    }

    #[test]
    fn a_decision_counts_under_its_own_primitive_and_leaves_the_token_counters_alone() {
        // Spec 05's acceptance 1 and 2, at the projection: a readout
        // generates nothing, so the series that count tokens must not move
        // for one. Asserted together, because `ignis_decoded_tokens_total 0`
        // beside no decisions at all would prove nothing.
        let metrics = Metrics::new();
        metrics.record_decision(Primitive::Choice, Some(0.997));
        metrics.record_decision(Primitive::Choice, Some(0.999));
        metrics.record_decision(Primitive::Score, Some(0.9));

        let text = metrics.render();
        assert_eq!(value(&text, "ignis_decisions_total", "type=\"choice\""), "2");
        assert_eq!(value(&text, "ignis_decisions_total", "type=\"score\""), "1");
        assert_eq!(value(&text, "ignis_decisions_total", "type=\"noul\""), "0");
        assert_eq!(value(&text, "ignis_decision_answer_mass_count", ""), "3");
        assert_eq!(value(&text, "ignis_decoded_tokens_total", ""), "0");
        assert_eq!(value(&text, "ignis_generated_tokens_total", ""), "0");
    }

    /// GitHub #242: a constrained decode is counted and observes no mass, so the
    /// histogram's `_count` is deliberately below the counter's sum.
    ///
    /// The owner's call, and an asymmetry a reader would otherwise have to
    /// infer from a graph that does not add up — ADR 0017's row says it in
    /// words and this says it in numbers.
    #[test]
    fn a_program_is_counted_and_observes_no_answer_mass() {
        let metrics = Metrics::new();
        metrics.record_decision(Primitive::Noul, Some(0.998));
        metrics.record_decision(Primitive::Point, None);
        metrics.record_decision(Primitive::Number, None);
        metrics.record_decision(Primitive::Box, None);

        let text = metrics.render();
        for (primitive, expected) in [("noul", "1"), ("number", "1"), ("point", "1"), ("box", "1")]
        {
            assert_eq!(
                value(&text, "ignis_decisions_total", &format!("type=\"{primitive}\"")),
                expected,
                "every primitive is counted, whatever answered it"
            );
        }
        assert_eq!(
            value(&text, "ignis_decision_answer_mass_count", ""),
            "1",
            "and only the readout observed a mass: a run's answer is a run of sampled tokens, with no single position's distribution to be a share of"
        );
        assert_eq!(
            value(&text, "ignis_decision_answer_mass_sum", ""),
            "0.998",
            "a constrained decode contributing 0 would put a false alarm in the bucket a real collapse lands in"
        );
    }

    #[test]
    fn a_mass_observation_lands_in_every_bucket_at_or_above_it() {
        let metrics = Metrics::new();
        // 0.998 sits *on* a boundary (`le` is inclusive), 0.9 on the second,
        // and a collapse below the coarsest bound lands only above 0.5.
        metrics.record_decision(Primitive::Noul, Some(0.9));
        metrics.record_decision(Primitive::Noul, Some(0.998));
        metrics.record_decision(Primitive::Noul, Some(0.03));

        let text = metrics.render();
        let at = |le: &str| {
            value(&text, "ignis_decision_answer_mass_bucket", &format!("le=\"{le}\""))
        };
        assert_eq!(at("0.5"), "1", "the collapsed one, and only it");
        assert_eq!(at("0.9"), "2");
        assert_eq!(at("0.995"), "2");
        assert_eq!(at("0.998"), "3", "a boundary is inclusive");
        assert_eq!(at("1"), "3");
        assert_eq!(at("+Inf"), "3");
        assert_eq!(value(&text, "ignis_decision_answer_mass_count", ""), "3");
        assert_eq!(value(&text, "ignis_decision_answer_mass_sum", ""), "1.928");
    }

    #[test]
    fn a_mass_outside_the_unit_interval_is_visible_rather_than_clamped() {
        // `le="1"` equalling `_count` is the exposition's own statement
        // that every reading was a probability, and it is only worth
        // exporting if it can be false. Clamping would make it true by
        // construction; so would filing a NaN under 0, which would also
        // report a collapse that did not happen.
        let metrics = Metrics::new();
        metrics.record_decision(Primitive::Choice, Some(0.998));
        metrics.record_decision(Primitive::Choice, Some(1.5));
        metrics.record_decision(Primitive::Choice, Some(-0.2));
        metrics.record_decision(Primitive::Choice, Some(f64::NAN));

        let text = metrics.render();
        let at = |le: &str| {
            value(&text, "ignis_decision_answer_mass_bucket", &format!("le=\"{le}\""))
        };
        assert_eq!(at("0.5"), "0", "nothing collapsed, whatever went wrong");
        assert_eq!(at("1"), "1", "one reading was a probability");
        assert_eq!(at("+Inf"), "4", "and three were not");
        assert_eq!(value(&text, "ignis_decision_answer_mass_count", ""), "4");
        // 0.998 + 1.5; the NaN and the negative are not numbers a sum can
        // carry, and the bucket count is where they show.
        assert_eq!(value(&text, "ignis_decision_answer_mass_sum", ""), "2.498");
        assert_eq!(value(&text, "ignis_decisions_total", "type=\"choice\""), "4");
    }

    #[test]
    fn confidence_is_never_exported() {
        // Spec 05 states it as a rule rather than an omission: confidence is
        // per-caller and per-domain, and a histogram over callers who each
        // mean something different by it means nothing. Pinned, so the
        // omission cannot be undone by somebody adding "the other number
        // the endpoint already has".
        let metrics = Metrics::new();
        metrics.record_decision(Primitive::Score, Some(0.9));
        let text = metrics.render();
        assert!(!text.contains("confidence"), "{text}");
    }

    /// ADR 0017's fixed boundaries, in seconds, `+Inf` implied.
    const TTFT_BOUNDS: [&str; 12] =
        ["0.05", "0.1", "0.25", "0.5", "1", "2.5", "5", "10", "30", "60", "120", "300"];
    const DURATION_BOUNDS: [&str; 12] =
        ["0.1", "0.25", "0.5", "1", "2.5", "5", "10", "30", "60", "120", "300", "600"];

    /// The `le` values of `name`'s buckets, in exposition order.
    fn bucket_bounds(text: &str, name: &str) -> Vec<String> {
        samples(text)
            .into_iter()
            .filter(|(n, _, _)| *n == format!("{name}_bucket"))
            .map(|(_, labels, _)| {
                labels.strip_prefix("le=\"").and_then(|l| l.strip_suffix('"')).expect("only `le`").to_owned()
            })
            .collect()
    }

    /// The amended contract's mass boundaries, `+Inf` implied.
    const MASS_BOUNDS: [&str; 10] =
        ["0.5", "0.9", "0.95", "0.98", "0.99", "0.995", "0.998", "0.999", "0.9995", "1"];

    #[test]
    fn the_histograms_use_exactly_the_adr_s_buckets_and_positive_infinity() {
        let text = Metrics::new().render();
        for (name, bounds) in [
            ("ignis_request_ttft_seconds", TTFT_BOUNDS),
            ("ignis_request_duration_seconds", DURATION_BOUNDS),
        ] {
            let mut expected: Vec<String> = bounds.iter().map(|b| (*b).to_owned()).collect();
            expected.push("+Inf".to_owned());
            assert_eq!(bucket_bounds(&text, name), expected, "{text}");
            assert_eq!(value(&text, &format!("{name}_sum"), ""), "0");
            assert_eq!(value(&text, &format!("{name}_count"), ""), "0");
        }
        // The mass histogram exists only after a decision, so it is read off
        // a projection that has seen one.
        let metrics = Metrics::new();
        metrics.record_decision(Primitive::Noul, Some(1.0));
        let text = metrics.render();
        let mut expected: Vec<String> = MASS_BOUNDS.iter().map(|b| (*b).to_owned()).collect();
        expected.push("+Inf".to_owned());
        assert_eq!(bucket_bounds(&text, "ignis_decision_answer_mass"), expected, "{text}");
    }

    #[test]
    fn an_observation_lands_in_every_bucket_at_or_above_it() {
        let metrics = Metrics::new();
        // 50 ms sits on the first TTFT boundary (`le` is inclusive); 250 ms
        // on the third; 400 s is past the last one, so only `+Inf` has it.
        metrics.observe_ttft_ms(50);
        metrics.observe_ttft_ms(250);
        metrics.observe_ttft_ms(400_000);
        metrics.observe_duration_ms(700);

        let text = metrics.render();
        let ttft = |le: &str| value(&text, "ignis_request_ttft_seconds_bucket", &format!("le=\"{le}\""));
        assert_eq!(ttft("0.05"), "1");
        assert_eq!(ttft("0.1"), "1");
        assert_eq!(ttft("0.25"), "2");
        assert_eq!(ttft("300"), "2");
        assert_eq!(ttft("+Inf"), "3");
        assert_eq!(value(&text, "ignis_request_ttft_seconds_count", ""), "3");
        assert_eq!(value(&text, "ignis_request_ttft_seconds_sum", ""), "400.3");

        let duration = |le: &str| value(&text, "ignis_request_duration_seconds_bucket", &format!("le=\"{le}\""));
        assert_eq!(duration("0.5"), "0");
        assert_eq!(duration("1"), "1");
        assert_eq!(duration("+Inf"), "1");
        assert_eq!(value(&text, "ignis_request_duration_seconds_sum", ""), "0.7");
    }

    #[test]
    fn a_label_value_is_escaped_per_the_text_format() {
        assert_eq!(escape_label_value(r#"a"b\c"#), r#"a\"b\\c"#);
        assert_eq!(escape_label_value("a\nb"), r"a\nb");
    }

    #[test]
    fn expert_residency_renders_fixed_families_from_its_counts() {
        use ignis_core::residency::{KBits, Projection};
        let mut counters = ResidencyCounters::default();
        let down_k2 = KClass::new(Projection::Down, KBits::K2);
        counters.hits[down_k2.index()][Phase::Decode.index()] = 7;
        counters.misses[down_k2.index()][Phase::Prefill.index()] = 3;
        counters.prefetch_issued = 5;
        counters.prefetch_used = 4;
        counters.bytes_moved = [1_000, 2_000];
        counters.stall_nanos = [1_500_000, 2_000_000_007];
        let mut capacity = [0u32; KClass::COUNT];
        capacity[down_k2.index()] = 40;
        let mut occupancy = [0u32; KClass::COUNT];
        occupancy[down_k2.index()] = 12;
        let mut text = String::new();
        render_expert_residency(&mut text, &counters, &capacity, &occupancy);

        // Help and type once per family; 8 x 2 hits, 8 x 2 misses, 2
        // prefetch, 2 bytes, 8 x 2 slots, zeros included.
        for (name, kind) in [
            ("ignis_expert_cache_hits_total", "counter"),
            ("ignis_expert_cache_misses_total", "counter"),
            ("ignis_expert_prefetches_issued_total", "counter"),
            ("ignis_expert_prefetches_used_total", "counter"),
            ("ignis_expert_bytes_moved_total", "counter"),
            ("ignis_expert_residency_stall_seconds_total", "counter"),
            ("ignis_expert_cache_slots", "gauge"),
        ] {
            assert_eq!(text.matches(&format!("# TYPE {name} {kind}\n")).count(), 1, "{name}");
            assert_eq!(text.matches(&format!("# HELP {name} ")).count(), 1, "{name}");
        }
        assert_eq!(samples(&text).len(), 54);
        // Nanoseconds as exact seconds.
        let stall = |phase: &str| value(&text, "ignis_expert_residency_stall_seconds_total", &format!("phase=\"{phase}\""));
        assert_eq!(stall("decode"), "0.0015");
        assert_eq!(stall("prefill"), "2.000000007");
        let hits = |labels: &str| value(&text, "ignis_expert_cache_hits_total", labels);
        assert_eq!(hits("class=\"down_k2\",phase=\"decode\""), "7");
        assert_eq!(hits("class=\"gate_up_k2_5\",phase=\"prefill\""), "0");
        assert_eq!(
            value(&text, "ignis_expert_cache_misses_total", "class=\"down_k2\",phase=\"prefill\""),
            "3"
        );
        assert_eq!(value(&text, "ignis_expert_prefetches_issued_total", ""), "5");
        assert_eq!(value(&text, "ignis_expert_prefetches_used_total", ""), "4");
        assert_eq!(value(&text, "ignis_expert_bytes_moved_total", "phase=\"prefill\""), "2000");
        assert_eq!(
            value(&text, "ignis_expert_cache_slots", "class=\"down_k2\",state=\"capacity\""),
            "40"
        );
        assert_eq!(
            value(&text, "ignis_expert_cache_slots", "class=\"down_k2\",state=\"in_use\""),
            "12"
        );
    }

    #[test]
    fn a_load_without_flash_next_counters_renders_none_of_their_families() {
        let text = Metrics::new().render();
        assert!(!text.contains("ignis_expert_"), "{text}");
        assert!(!text.contains("ignis_ngram_"), "{text}");
    }

    /// A Flash-Next reading, with figures in every family.
    fn flash_next_reading() -> FlashNextCounters {
        use ignis_core::residency::{KBits, Projection};
        let gate_up_k4 = KClass::new(Projection::GateUp, KBits::K4).index();
        let mut reading = FlashNextCounters::default();
        reading.residency.hits[gate_up_k4][Phase::Decode.index()] = 21;
        reading.residency.misses[gate_up_k4][Phase::Decode.index()] = 4;
        reading.residency.bytes_moved = [3_000, 0];
        reading.residency.stall_nanos = [9, 0];
        reading.slots_capacity[gate_up_k4] = 64;
        reading.slots_in_use[gate_up_k4] = 33;
        reading.ngram = NgramCounters { rows: 320, hot_rows: 300, file_rows: 20, reads: 7, read_bytes: 28_672 };
        reading
    }

    #[test]
    fn a_flash_next_load_appends_its_counters_after_the_whole_27b_exposition() {
        let metrics = Metrics::new();
        metrics.record_accepted();
        let without = metrics.render();
        metrics.set_flash_next(Some(&flash_next_reading()));
        let with = metrics.render();

        // Byte-identical up to the end of what a 27B load renders.
        let suffix = with.strip_prefix(without.as_str()).expect("the 27B exposition is a prefix");
        assert!(suffix.starts_with("# HELP ignis_expert_cache_hits_total "), "{suffix}");
        declared_once(
            &with,
            &[
                ("ignis_expert_cache_hits_total", "counter"),
                ("ignis_expert_cache_misses_total", "counter"),
                ("ignis_expert_prefetches_issued_total", "counter"),
                ("ignis_expert_prefetches_used_total", "counter"),
                ("ignis_expert_bytes_moved_total", "counter"),
                ("ignis_expert_residency_stall_seconds_total", "counter"),
                ("ignis_expert_cache_slots", "gauge"),
                ("ignis_ngram_rows_total", "counter"),
                ("ignis_ngram_reads_total", "counter"),
                ("ignis_ngram_read_bytes_total", "counter"),
            ],
        );
        let hits = value(&with, "ignis_expert_cache_hits_total", "class=\"gate_up_k4\",phase=\"decode\"");
        assert_eq!(hits, "21");
        assert_eq!(
            value(&with, "ignis_expert_cache_misses_total", "class=\"gate_up_k4\",phase=\"decode\""),
            "4"
        );
        assert_eq!(value(&with, "ignis_expert_bytes_moved_total", "phase=\"decode\""), "3000");
        assert_eq!(
            value(&with, "ignis_expert_cache_slots", "class=\"gate_up_k4\",state=\"capacity\""),
            "64"
        );
        assert_eq!(
            value(&with, "ignis_expert_cache_slots", "class=\"gate_up_k4\",state=\"in_use\""),
            "33"
        );
        assert_eq!(value(&with, "ignis_ngram_rows_total", "source=\"hot\""), "300");
        assert_eq!(value(&with, "ignis_ngram_rows_total", "source=\"file\""), "20");
        assert_eq!(value(&with, "ignis_ngram_reads_total", ""), "7");
        assert_eq!(value(&with, "ignis_ngram_read_bytes_total", ""), "28672");
        assert_eq!(
            value(&with, "ignis_expert_residency_stall_seconds_total", "phase=\"decode\""),
            "0.000000009"
        );
        // 54 residency series and 4 n-gram ones, nothing else added.
        assert_eq!(samples(&with).len(), samples(&without).len() + 58);
    }

    #[test]
    fn a_load_that_has_no_counters_any_more_renders_the_27b_exposition_again() {
        let metrics = Metrics::new();
        let without = metrics.render();
        metrics.set_flash_next(Some(&flash_next_reading()));
        assert_ne!(metrics.render(), without);
        // A reload onto a model without them: no dead model's families.
        metrics.set_flash_next(None);
        assert_eq!(metrics.render(), without);
        // And a new Flash-Next load replaces every figure.
        metrics.set_flash_next(Some(&FlashNextCounters::default()));
        let text = metrics.render();
        assert_eq!(value(&text, "ignis_expert_cache_hits_total", "class=\"gate_up_k4\",phase=\"decode\""), "0");
        assert_eq!(value(&text, "ignis_ngram_reads_total", ""), "0");
    }
}
