//! The G3 measurement instrument (P3-07, GitHub #100): C=1 / C=4 / ITL,
//! measured over HTTP/SSE against any OpenAI-compatible endpoint.
//!
//! Spec: `.scratch/runtime/specs/03-serving-loop.md`; ADR 0015 (live/live
//! gate, cold samples). The verdict half lives in [`crate::g3_gate`].
//!
//! Reuses [`crate::ttft`]'s prompt generator and cold-prefix rule verbatim
//! (ADR 0015: "later gates should follow this method... unless a specific
//! gate has a reason not to" — G3 has none) and [`crate::metrics`]'s
//! throughput-weighted aggregation and percentile math, so a G3 cell is
//! built from the same audited pieces G2's cell was.
//!
//! ## Three cells
//!
//! - **C=1**: one sequence, prompt 8,192 / cap 256. The verdict is a
//!   percentage of the live reference's decode tok/s.
//! - **C=4**: four sequences, the same fixture, fired concurrently. The
//!   verdict is the *aggregate* throughput (throughput-weighted, the way
//!   [`crate::metrics::class_stats`] already reduces a class) as a
//!   percentage of the live reference's aggregate.
//! - **ITL**: four decode lanes (prompt 4,096 / safety cap 4,032) sampled
//!   *continuously* while ten 32,768-token prefillers run **sequentially**
//!   against the pool — each allocated, prefilled and released before the
//!   next is allocated (ADR 0015: every prefiller prompt is its own, cold,
//!   verified). p50/p95/p99/max are all recorded; p95 decides.
//!
//! Every cell's prompts are the artifact's own tokenizer, exact post-
//! template length, generated and proven distinct by [`crate::ttft`] —
//! this module never re-derives that proof.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::client::{Endpoint, Request};
use crate::metrics::{class_stats, percentile, RequestMetrics};
use crate::time::{unix_now, utc_timestamp};
use crate::trace::RequestClass;
use crate::ttft::{
    self, generate_cell_prompts, generate_cell_prompts_from_corpus,
    generate_cell_prompts_from_corpus_with_suffix, generate_cell_prompts_with_suffix, load_corpus,
    PromptSet, PromptTemplate,
};

// ── fixtures (spec 03, the G3 gate table) ───────────────────────────────

/// C=1 / C=4 share this fixture: prompt 8,192, cap 256 (reserved
/// `context_tokens` 8,448).
pub const THROUGHPUT_PROMPT_TOKENS: u32 = 8_192;
pub const THROUGHPUT_MAX_TOKENS: u32 = 256;
pub const C1_CONCURRENCY: usize = 1;
pub const C4_CONCURRENCY: usize = 4;

/// The ITL cell's decode lanes: prompt 4,096, safety cap 4,032.
///
/// The cap exists so a lane the engine will not keep alive still ends
/// somewhere defined. It is meant to be *spare*: the measurement boundary
/// should close every lane first, and a leg whose lanes end on the cap has
/// had its length decided by the fixture rather than by what it is
/// measuring. #110's reference leg ended that way, clearing the final
/// prefill window by only 3.4 to 7.0 s out of a 101 s run, so a reference
/// some 7% faster would have exhausted the cap first (GitHub #114).
///
/// Since #139 that is a *shorter* measurement, not a refused one: the cell
/// measures the span all lanes shared and reports how much of the
/// prefiller series it covered ([`ItlCell::window_warning`]). The cap
/// being load-bearing therefore costs distribution rather than the leg —
/// which matters because the engine this gate exists to make faster is the
/// one that hits the cap first.
///
/// 4,032 is the largest cap the pool admits, so it is the most runway this
/// fixture can buy. Admission reserves `ceil((prompt + token budget) / 64)`
/// pages up front and never over-allocates mid-generation
/// (`ignis_core::admission`), and the engine's pool is 65,536 tokens =
/// 1,024 pages (`ignis_runtime::auto_kv_pool_bytes`'s 4 GiB default budget
/// under BF16 KV, at the default 40,960-token `--max-context`; under
/// hq-e8-2b the same budget buys 7.11x that, GitHub #122). Peak concurrent demand is the four lanes
/// plus the one in-flight prefiller:
///
///   lane      ceil((4,096 + 4,032) / 64) = 127 pages,  x4 = 508
///   prefiller ceil((32,768 +    64) / 64) = 513 pages
///   total                                             1,021 of 1,024
///
/// One more page per lane (a cap of 4,096) would need 1,025 and refuse a
/// lane. The three spare pages are not what protects the sequential
/// prefiller invariant, and never were: two overlapping prefillers need
/// 1,026 pages at any cap, so the invariant rests on the harness sending
/// the next prefiller only after the previous request returned, not on
/// headroom.
pub const ITL_DECODE_PROMPT_TOKENS: u32 = 4_096;
pub const ITL_DECODE_MAX_TOKENS: u32 = 4_032;
pub const ITL_DECODE_LANES: usize = 4;
const ITL_DECODE_INSTRUCTION: &str =
    "Produce at least 4032 tokens. Do not stop, conclude, or emit EOS earlier.";

/// The fewest prefill windows the measurement window may cover before the
/// ITL cell reports an anecdote rather than a distribution (GitHub #139).
///
/// The window is bounded by the *shortest-lived* decode lane, and three
/// different things can end one: the harness cancelling it at the end of
/// the prefiller series (what the fixture intends), the engine's own EOS
/// when the endpoint honours neither `ignore_eos` nor a `logit_bias`
/// exclusion, or [`ITL_DECODE_MAX_TOKENS`] when the engine is fast enough
/// to exhaust it first. The cell measures whatever span all four lanes
/// shared, so none of the three voids it — but spec 03 asks for ten
/// prefillers so the percentile has a distribution behind it, and a window
/// covering one has none at all.
pub const ITL_MIN_COVERED_PREFILLERS: usize = 2;

/// The ITL cell's prefillers: prompt 32,768, cap 64, run ten times,
/// sequentially, each released before the next is allocated.
pub const ITL_PREFILL_PROMPT_TOKENS: u32 = 32_768;
pub const ITL_PREFILL_MAX_TOKENS: u32 = 64;
pub const ITL_PREFILL_COUNT: usize = 10;

/// The request one sample sends: streaming (both throughput and ITL need
/// per-token timing), the trailing usage chunk (the cold-prefix evidence).
fn sample_request(id: String, prompt: String, max_tokens: u32) -> Request {
    Request {
        id,
        class: RequestClass::Main,
        prompt,
        max_tokens,
        stream: true,
        include_usage: true,
        enable_thinking: Some(false),
    }
}

// ── C=1 / C=4: throughput cells ─────────────────────────────────────────

/// One sequence's throughput sample: whether it completed, whether its
/// prefix was provably cold, and the raw timing a `RequestMetrics` is built
/// from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThroughputSample {
    /// This sample's request id (for the audit trail).
    pub id: String,
    pub ttft_ms: f64,
    pub total_ms: f64,
    pub n_tokens: u32,
    /// `false` when the request itself failed (never sent, or a mid-stream
    /// error) — distinct from [`ThroughputSample::void`], which is a
    /// request that *completed* but whose prefix was not provably cold.
    pub ok: bool,
    pub computed_prefill_tokens: Option<u32>,
    pub void: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub void_reason: Option<String>,
}

impl ThroughputSample {
    /// This sample as a [`RequestMetrics`] (the shared aggregation seam).
    fn as_metrics(&self) -> RequestMetrics {
        RequestMetrics {
            id: self.id.clone(),
            class: RequestClass::Main,
            ttft_ms: self.ttft_ms,
            n_tokens: self.n_tokens,
            total_ms: self.total_ms,
            ok: self.ok,
        }
    }
}

/// A throughput cell (C=1 or C=4): its fixture, its samples, and the
/// throughput-weighted aggregate over them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThroughputCell {
    pub prompt_tokens: u32,
    pub max_tokens: u32,
    /// How many sequences ran concurrently (1 for C=1, 4 for C=4).
    pub concurrency: usize,
    pub samples: Vec<ThroughputSample>,
    /// The throughput-weighted decode speed (tokens/s) over `samples`
    /// ([`crate::metrics::class_stats`]'s reduction — total decoded tokens
    /// over total decode time, so no single lane's outlier can inflate the
    /// aggregate). `0.0` when there is nothing to measure.
    pub aggregate_tok_s: f64,
    /// Set when the cell failed before any sample could be sent (prompt
    /// generation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ThroughputCell {
    /// True when every sample completed and was provably cold — the
    /// property a gate verdict may be computed over.
    pub fn all_cold(&self) -> bool {
        self.error.is_none()
            && !self.samples.is_empty()
            && self.samples.iter().all(|s| s.ok && !s.void)
    }

    /// The samples that are not usable (failed or not provably cold), with
    /// their reasons.
    pub fn bad_samples(&self) -> Vec<&ThroughputSample> {
        self.samples.iter().filter(|s| !s.ok || s.void).collect()
    }

    fn failed(prompt_tokens: u32, max_tokens: u32, concurrency: usize, error: String) -> Self {
        Self {
            prompt_tokens,
            max_tokens,
            concurrency,
            samples: Vec::new(),
            aggregate_tok_s: 0.0,
            error: Some(error),
        }
    }
}

/// Measure a throughput cell: `concurrency` sequences of `prompt_tokens` /
/// `max_tokens`, fired at once (a shared thread per sequence — [`std::thread::scope`]
/// so the borrowed `ep` / `template` need not be `'static`), each proving
/// its own cold prefix. Prompts are generated by the filler word-growth
/// path ([`generate_cell_prompts`]); [`measure_throughput_cell_from_corpus`]
/// is the bounded-generation alternative at production scale.
pub fn measure_throughput_cell(
    ep: &dyn Endpoint,
    template: &dyn PromptTemplate,
    prompt_tokens: u32,
    max_tokens: u32,
    concurrency: usize,
) -> ThroughputCell {
    measure_throughput_cell_with(ep, prompt_tokens, max_tokens, concurrency, || {
        generate_cell_prompts(template, prompt_tokens as usize, concurrency)
    })
}

/// Measure a throughput cell cut from a pre-tokenized corpus (the
/// `--corpus` path): bounded generation instead of the filler path's O(n²)
/// growth — the same tradeoff [`ttft::measure_cell_from_corpus`] makes for
/// G2, needed here because C=1/C=4's 8,192-token fixture pays the same
/// cost.
pub fn measure_throughput_cell_from_corpus(
    ep: &dyn Endpoint,
    template: &dyn PromptTemplate,
    prompt_tokens: u32,
    max_tokens: u32,
    concurrency: usize,
    corpus: &[u32],
) -> ThroughputCell {
    measure_throughput_cell_with(ep, prompt_tokens, max_tokens, concurrency, || {
        generate_cell_prompts_from_corpus(template, corpus, prompt_tokens as usize, concurrency)
    })
}

/// [`measure_throughput_cell_from_corpus`] with a fixed instruction appended
/// to every prompt (the window shortens to absorb it, so the prompt still
/// lands on `prompt_tokens`) — how G5 asks an engine that cannot be told to
/// ignore EOS for a long answer (#159).
pub fn measure_throughput_cell_from_corpus_with_suffix(
    ep: &dyn Endpoint,
    template: &dyn PromptTemplate,
    prompt_tokens: u32,
    max_tokens: u32,
    concurrency: usize,
    corpus: &[u32],
    suffix: &str,
) -> ThroughputCell {
    measure_throughput_cell_with(ep, prompt_tokens, max_tokens, concurrency, || {
        ttft::generate_cell_prompts_from_corpus_with_suffix(
            template,
            corpus,
            prompt_tokens as usize,
            concurrency,
            suffix,
        )
    })
}

/// The shared measurement: one prompt set (generated however the caller
/// says), fired at `concurrency`, each sample proving its own cold prefix.
fn measure_throughput_cell_with(
    ep: &dyn Endpoint,
    prompt_tokens: u32,
    max_tokens: u32,
    concurrency: usize,
    prompts: impl FnOnce() -> Result<PromptSet, String>,
) -> ThroughputCell {
    if concurrency == 0 {
        return ThroughputCell::failed(
            prompt_tokens,
            max_tokens,
            concurrency,
            "a throughput cell needs at least one sequence".to_string(),
        );
    }
    let set = match prompts() {
        Ok(set) => set,
        Err(err) => return ThroughputCell::failed(prompt_tokens, max_tokens, concurrency, err),
    };

    let samples: Vec<ThroughputSample> = std::thread::scope(|scope| {
        let handles: Vec<_> = set
            .prompts
            .iter()
            .enumerate()
            .map(|(index, prompt)| {
                let prompt = prompt.clone();
                scope.spawn(move || {
                    let id = format!("thr-{prompt_tokens}-{concurrency}-{index}");
                    let req = sample_request(id.clone(), prompt, max_tokens);
                    match ep.complete(&req) {
                        Ok(outcome) => {
                            let computed = outcome.computed_prefill_tokens();
                            let void_reason = ttft::coldness_failure(computed, prompt_tokens);
                            ThroughputSample {
                                id,
                                ttft_ms: outcome.ttft_ms,
                                total_ms: outcome.total_ms,
                                n_tokens: outcome.n_tokens,
                                ok: true,
                                computed_prefill_tokens: computed,
                                void: void_reason.is_some(),
                                void_reason,
                            }
                        }
                        Err(err) => ThroughputSample {
                            id,
                            ttft_ms: 0.0,
                            total_ms: 0.0,
                            n_tokens: 0,
                            ok: false,
                            computed_prefill_tokens: None,
                            void: true,
                            void_reason: Some(format!("the request failed: {err}")),
                        },
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a throughput worker thread panicked"))
            .collect()
    });

    let metrics: Vec<RequestMetrics> = samples.iter().map(ThroughputSample::as_metrics).collect();
    let aggregate_tok_s = class_stats(RequestClass::Main, &metrics).tok_s;

    ThroughputCell {
        prompt_tokens,
        max_tokens,
        concurrency,
        samples,
        aggregate_tok_s,
        error: None,
    }
}

// ── ITL: inter-token latency under a concurrent prefill ─────────────────

/// One prefiller in the ITL series: whether its prefix was provably cold.
/// Mirrors [`ttft::Sample`] in shape — same rule, different cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrefillerSample {
    pub index: usize,
    /// Request start on the ITL cell's shared monotonic timeline.
    #[serde(default)]
    pub started_ms: f64,
    /// First content token on that same timeline (the end of prefill).
    #[serde(default)]
    pub first_token_ms: f64,
    pub ttft_ms: f64,
    pub computed_prefill_tokens: Option<u32>,
    pub void: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub void_reason: Option<String>,
}

/// How a decode lane's stream ended (GitHub #114).
///
/// [`LaneFinish::Window`] is the one the fixture intends: the lane outlived
/// the whole prefiller series and the harness closed it at the measurement
/// boundary. Every other value means something else decided the lane's
/// length, which a reader of the record has to be able to see rather than
/// infer from a token count that happens to equal the cap.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneFinish {
    /// The harness closed the lane at the measurement boundary.
    Window,
    /// The engine stopped on its own end-of-sequence: the EOS suppression
    /// the lane asked for was not granted (only ignis honours
    /// `ignore_eos`; a `logit_bias` exclusion has already been recorded as
    /// insufficient against the reference).
    StopToken,
    /// The engine reached the fixture's safety cap
    /// ([`ITL_DECODE_MAX_TOKENS`]).
    Cap,
    /// The engine named a reason this instrument does not model.
    Other,
    /// The stream ended with neither a finish reason nor a cancellation —
    /// including every record written before this field existed, which is
    /// why it is the default.
    #[default]
    Unknown,
}

impl LaneFinish {
    /// What the transport saw, in the record's own vocabulary. `"length"`
    /// and `"stop"` are the OpenAI finish reasons both engines speak.
    fn from_transport(reason: Option<&crate::client::FinishReason>) -> Self {
        match reason {
            Some(crate::client::FinishReason::Cancelled) => Self::Window,
            Some(crate::client::FinishReason::Engine(reason)) => match reason.as_str() {
                "length" => Self::Cap,
                "stop" => Self::StopToken,
                _ => Self::Other,
            },
            None => Self::Unknown,
        }
    }

    /// Whether this is the fixture deciding the lane's length rather than
    /// the measurement boundary — what the cell warns about.
    pub fn is_cap(self) -> bool {
        matches!(self, Self::Cap)
    }

    /// A short phrase for the rendered record.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Window => "the measurement boundary",
            Self::StopToken => "its own EOS",
            Self::Cap => "the safety cap",
            Self::Other => "a reason this instrument does not model",
            Self::Unknown => "unknown",
        }
    }
}

/// One decode lane's trace across the whole ITL series: every content
/// token's arrival time (ms since the lane's own request started). Only
/// intervals overlapping a prefiller window enter the pooled ITL sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeLaneTrace {
    pub id: String,
    /// Request start on the ITL cell's shared monotonic timeline.
    #[serde(default)]
    pub started_ms: f64,
    pub n_tokens: u32,
    pub token_times_ms: Vec<f64>,
    /// Why this lane's stream ended (GitHub #114).
    #[serde(default)]
    pub finish: LaneFinish,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl DecodeLaneTrace {
    /// When this lane's last content token arrived, on the cell's shared
    /// monotonic timeline. A lane with no tokens never decoded, so this is
    /// its own request start.
    pub fn end_ms(&self) -> f64 {
        self.started_ms + self.token_times_ms.last().copied().unwrap_or(0.0)
    }
}

/// The ITL cell: the ten sequential prefillers, the four continuous decode
/// lanes, and the pooled inter-token-interval percentiles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ItlCell {
    pub prefill_prompt_tokens: u32,
    pub prefill_max_tokens: u32,
    pub prefill_count: usize,
    pub decode_prompt_tokens: u32,
    pub decode_max_tokens: u32,
    pub decode_lanes: usize,
    pub prefillers: Vec<PrefillerSample>,
    pub lanes: Vec<DecodeLaneTrace>,
    /// The **measurement window**: the span of the cell's shared monotonic
    /// timeline in which every decode lane was alive. It opens at the last
    /// lane's first token and closes with the first lane to end, so a lane
    /// the engine ended early — at its own EOS, or on the safety cap —
    /// shortens the window instead of voiding the cell (GitHub #139).
    ///
    /// Where a lane *ends* depends on who ended it. One the harness
    /// cancelled was generating up to the cancellation, so that is its end;
    /// only a lane the engine stopped ends at its last observed token.
    /// Otherwise the window's close would be a race between a lane's final
    /// SSE chunk and the harness's own store, which is a property of
    /// neither engine.
    ///
    /// The opening is guaranteed rather than merely measured: the harness
    /// does not send the first prefiller until every lane has produced a
    /// token, so `window_start_ms` always precedes the series. It is
    /// recorded so a reader can see it instead of trusting it.
    #[serde(default)]
    pub window_start_ms: f64,
    /// Where the measurement window closed (see
    /// [`ItlCell::window_start_ms`]).
    #[serde(default)]
    pub window_end_ms: f64,
    /// How many prefillers reached their first token inside the
    /// measurement window. Only those are pooled against, so this — not
    /// [`ItlCell::prefill_count`] — is the size of the distribution behind
    /// the percentiles.
    #[serde(default)]
    pub prefillers_covered: usize,
    /// Every decode lane's inter-token intervals that lie inside the
    /// measurement window and overlap a covered prefiller's request-start
    /// -> first-token window, pooled across the series.
    pub intervals_ms: Vec<f64>,
    pub p50_ms: Option<f64>,
    pub p95_ms: Option<f64>,
    pub p99_ms: Option<f64>,
    pub max_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ItlCell {
    /// True when every prefiller was provably cold, every decode lane
    /// completed, the measurement window covered enough of the series to
    /// be a distribution ([`ITL_MIN_COVERED_PREFILLERS`]), and the cell has
    /// intervals to report — the property a gate verdict may be computed
    /// over.
    pub fn all_cold(&self) -> bool {
        self.error.is_none()
            && !self.prefillers.is_empty()
            && self
                .prefillers
                .iter()
                .all(|p| !p.void && p.first_token_ms > p.started_ms)
            && !self.lanes.is_empty()
            && self.lanes.iter().all(|l| l.error.is_none())
            && self.covers_enough()
            && !self.intervals_ms.is_empty()
    }

    /// Whether the measurement window covered enough of the prefiller
    /// series to stand behind a percentile ([`ITL_MIN_COVERED_PREFILLERS`]).
    pub fn covers_enough(&self) -> bool {
        self.prefillers_covered >= ITL_MIN_COVERED_PREFILLERS
    }

    /// The prefillers that are not provably cold, with their reasons.
    pub fn void_prefillers(&self) -> Vec<&PrefillerSample> {
        self.prefillers.iter().filter(|p| p.void).collect()
    }

    /// The lanes the fixture's safety cap ended rather than the measurement
    /// boundary (GitHub #114). Not an error: the cell measures the span
    /// every lane shared, so a capped lane shortens the measurement window
    /// rather than contaminating it. It does mean the cap was
    /// load-bearing for this leg, which is worth saying out loud rather
    /// than leaving to a reader who notices a token count equal to the cap.
    pub fn lanes_on_cap(&self) -> Vec<&DecodeLaneTrace> {
        self.lanes.iter().filter(|lane| lane.finish.is_cap()).collect()
    }

    /// The lane that closed the measurement window early — the first one
    /// the *engine* ended, and therefore the one that decided how much of
    /// the prefiller series this cell could measure.
    ///
    /// `None` when the harness closed every lane itself, which is the case
    /// in which nothing closed the window early: a cancelled lane was
    /// generating until the boundary, so it bounds nothing.
    pub fn window_closed_by(&self) -> Option<&DecodeLaneTrace> {
        self.lanes
            .iter()
            .filter(|lane| {
                lane.error.is_none()
                    && !lane.token_times_ms.is_empty()
                    && lane.finish != LaneFinish::Window
            })
            .min_by(|a, b| a.end_ms().total_cmp(&b.end_ms()))
    }

    /// A one-line warning when the measurement window covered less than the
    /// whole prefiller series, naming the lane that closed it and how that
    /// lane ended. `None` when every prefiller was covered.
    ///
    /// Not an error — [`ItlCell::all_cold`] still holds down to
    /// [`ITL_MIN_COVERED_PREFILLERS`]. What it costs is distribution: the
    /// percentiles stand on the prefill windows actually covered, not on
    /// spec 03's ten.
    pub fn window_warning(&self) -> Option<String> {
        if self.prefillers.is_empty() || self.prefillers_covered >= self.prefillers.len() {
            return None;
        }
        let closed_by = self.window_closed_by();
        Some(format!(
            "the measurement window covered {} of {} prefill window(s): lane {} ended first, on \
             {} -- the pooled percentiles have that much of the series behind them, not all of it",
            self.prefillers_covered,
            self.prefillers.len(),
            closed_by.map_or("(none)", |lane| lane.id.as_str()),
            closed_by.map_or("an unrecorded reason", |lane| lane.finish.describe()),
        ))
    }

    /// A one-line warning when the safety cap ended any lane (GitHub #114),
    /// or `None` when no lane reached it.
    ///
    /// Independent of [`ItlCell::window_warning`], and needed beside it: a
    /// lane that exhausts the cap *after* the last prefill window closed
    /// costs this leg no coverage at all and would otherwise go unsaid --
    /// which is #110's exact shape, and the reason #114 asked for it. The
    /// cap being load-bearing is a fact about how close this fixture came
    /// to deciding the cell's length itself, whatever the coverage was.
    pub fn cap_warning(&self) -> Option<String> {
        let on_cap = self.lanes_on_cap();
        if on_cap.is_empty() {
            return None;
        }
        let names: Vec<&str> = on_cap.iter().map(|lane| lane.id.as_str()).collect();
        Some(format!(
            "{} of {} decode lane(s) ended on the {}-token safety cap rather than at the \
             measurement boundary ({}) -- the cap is load-bearing for this leg, not spare",
            on_cap.len(),
            self.lanes.len(),
            self.decode_max_tokens,
            names.join(", "),
        ))
    }

    /// Everything about this cell a reader has to be told rather than left
    /// to infer: how much of the series the window covered, and whether the
    /// fixture's own cap ended a lane. Either, both, or neither.
    pub fn warnings(&self) -> Vec<String> {
        [self.window_warning(), self.cap_warning()].into_iter().flatten().collect()
    }

    fn failed(cfg: &ItlConfig, error: String) -> Self {
        Self {
            prefill_prompt_tokens: cfg.prefill_prompt_tokens,
            prefill_max_tokens: cfg.prefill_max_tokens,
            prefill_count: cfg.prefill_count,
            decode_prompt_tokens: cfg.decode_prompt_tokens,
            decode_max_tokens: cfg.decode_max_tokens,
            decode_lanes: cfg.decode_lanes,
            prefillers: Vec::new(),
            lanes: Vec::new(),
            window_start_ms: 0.0,
            window_end_ms: 0.0,
            prefillers_covered: 0,
            intervals_ms: Vec::new(),
            p50_ms: None,
            p95_ms: None,
            p99_ms: None,
            max_ms: None,
            error: Some(error),
        }
    }
}

/// What an ITL run measures: the two fixtures (spec 03's table) and how
/// many of each.
#[derive(Debug, Clone)]
pub struct ItlConfig {
    pub prefill_prompt_tokens: u32,
    pub prefill_max_tokens: u32,
    pub prefill_count: usize,
    pub decode_prompt_tokens: u32,
    pub decode_max_tokens: u32,
    pub decode_lanes: usize,
}

impl Default for ItlConfig {
    fn default() -> Self {
        Self {
            prefill_prompt_tokens: ITL_PREFILL_PROMPT_TOKENS,
            prefill_max_tokens: ITL_PREFILL_MAX_TOKENS,
            prefill_count: ITL_PREFILL_COUNT,
            decode_prompt_tokens: ITL_DECODE_PROMPT_TOKENS,
            decode_max_tokens: ITL_DECODE_MAX_TOKENS,
            decode_lanes: ITL_DECODE_LANES,
        }
    }
}

/// Measure the ITL cell: start `cfg.decode_lanes` decode lanes (each its
/// own streaming request, observed for the whole series), then run
/// `cfg.prefill_count` prefillers **sequentially** on the calling thread,
/// each sent only once the previous request's `ep.complete` call has
/// returned. This instrument watches only the HTTP/SSE side, so it cannot
/// *observe* the engine's own release of a prefiller's pages — it can only
/// guarantee the weaker property an HTTP client is able to guarantee: no
/// two prefiller requests are ever in flight at once. [`std::thread::scope`]
/// joins the decode lanes before returning, so `ep` / `template` need not
/// be `'static`. Prompts are generated by the filler word-growth path
/// ([`generate_cell_prompts`]); [`measure_itl_from_corpus`] is the
/// bounded-generation alternative the 32,768-token prefiller fixture needs
/// at production scale.
pub fn measure_itl(ep: &dyn Endpoint, template: &dyn PromptTemplate, cfg: &ItlConfig) -> ItlCell {
    measure_itl_with(
        ep,
        cfg,
        template.eos_token_ids(),
        || {
            generate_cell_prompts_with_suffix(
                template,
                cfg.decode_prompt_tokens as usize,
                cfg.decode_lanes,
                ITL_DECODE_INSTRUCTION,
            )
        },
        || generate_cell_prompts(template, cfg.prefill_prompt_tokens as usize, cfg.prefill_count),
    )
}

/// Measure the ITL cell with both fixtures cut from a pre-tokenized corpus
/// (the `--corpus` path): bounded generation instead of the filler path's
/// O(n²) growth, which the 32,768-token prefiller fixture pays for badly
/// (the same tradeoff [`ttft::measure_cell_from_corpus`] makes for G2's
/// own 32K cell).
pub fn measure_itl_from_corpus(
    ep: &dyn Endpoint,
    template: &dyn PromptTemplate,
    cfg: &ItlConfig,
    corpus: &[u32],
) -> ItlCell {
    measure_itl_with(
        ep,
        cfg,
        template.eos_token_ids(),
        || {
            generate_cell_prompts_from_corpus_with_suffix(
                template,
                corpus,
                cfg.decode_prompt_tokens as usize,
                cfg.decode_lanes,
                ITL_DECODE_INSTRUCTION,
            )
        },
        || {
            generate_cell_prompts_from_corpus(
                template,
                corpus,
                cfg.prefill_prompt_tokens as usize,
                cfg.prefill_count,
            )
        },
    )
}

/// The shared measurement: both fixtures generated however the caller
/// says, then the decode-lane / sequential-prefiller series described on
/// [`measure_itl`].
fn measure_itl_with(
    ep: &dyn Endpoint,
    cfg: &ItlConfig,
    eos_tokens: Vec<u32>,
    decode_prompts: impl FnOnce() -> Result<PromptSet, String>,
    prefill_prompts: impl FnOnce() -> Result<PromptSet, String>,
) -> ItlCell {
    if cfg.decode_lanes == 0 || cfg.prefill_count == 0 {
        return ItlCell::failed(
            cfg,
            "the ITL cell needs at least one decode lane and one prefiller".to_string(),
        );
    }
    let decode_set = match decode_prompts() {
        Ok(set) => set,
        Err(err) => return ItlCell::failed(cfg, format!("generating decode-lane prompts: {err}")),
    };
    let prefill_set = match prefill_prompts() {
        Ok(set) => set,
        Err(err) => return ItlCell::failed(cfg, format!("generating prefiller prompts: {err}")),
    };

    let epoch = std::time::Instant::now();
    let (lane_ready_tx, lane_ready_rx) = std::sync::mpsc::channel();
    let lanes_running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let eos_tokens = std::sync::Arc::new(eos_tokens);
    let (lanes, prefillers, cancelled_at_ms) = std::thread::scope(|scope| {
        // The decode lanes: one streaming request each, held open until the
        // sequential prefiller series ends, then cancelled.
        //
        // Cancellation is the *intended* terminator, not the guaranteed
        // one. It only gets the chance if the engine kept the lane alive
        // that long, which needs the endpoint to have honoured the EOS
        // suppression; an engine that honours neither `ignore_eos` nor
        // `logit_bias` stops at its own EOS instead, and one that is fast
        // enough reaches `decode_max_tokens` first. Whichever of the three
        // ends the shortest lane, the pooled intervals stay honest: the
        // measurement window below is the span every lane shared, and
        // nothing outside it is pooled.
        let lane_handles: Vec<_> = decode_set
            .prompts
            .iter()
            .enumerate()
            .map(|(index, prompt)| {
                let prompt = prompt.clone();
                let lane_ready_tx = lane_ready_tx.clone();
                let lanes_running = std::sync::Arc::clone(&lanes_running);
                let eos_tokens = std::sync::Arc::clone(&eos_tokens);
                scope.spawn(move || {
                    let id = format!("itl-decode-{index}");
                    let req = sample_request(id.clone(), prompt, cfg.decode_max_tokens);
                    let started_ms = epoch.elapsed().as_secs_f64() * 1_000.0;
                    let mut ready_sent = false;
                    let mut observe = |_| {
                        if !ready_sent {
                            let _ = lane_ready_tx.send(());
                            ready_sent = true;
                        }
                        lanes_running.load(std::sync::atomic::Ordering::Acquire)
                    };
                    let result = ep.complete_observed_while(&req, &mut observe, eos_tokens.as_slice());
                    if !ready_sent {
                        let _ = lane_ready_tx.send(());
                    }
                    match result {
                        Ok(outcome) => DecodeLaneTrace {
                            id,
                            started_ms,
                            n_tokens: outcome.n_tokens,
                            finish: LaneFinish::from_transport(outcome.finish_reason.as_ref()),
                            token_times_ms: outcome.token_times_ms,
                            error: None,
                        },
                        Err(err) => DecodeLaneTrace {
                            id,
                            started_ms,
                            n_tokens: 0,
                            token_times_ms: Vec::new(),
                            finish: LaneFinish::Unknown,
                            error: Some(err),
                        },
                    }
                })
            })
            .collect();

        for _ in 0..cfg.decode_lanes {
            lane_ready_rx.recv().expect("a decode-lane thread exited before signalling readiness");
        }

        // The prefillers: sequential, on this (the scope's spawning)
        // thread. The next is only sent once `ep.complete` returns for the
        // previous — the strongest release guarantee an HTTP/SSE-only
        // instrument can make (see `measure_itl`'s doc comment).
        let mut prefillers = Vec::with_capacity(prefill_set.prompts.len());
        for (index, prompt) in prefill_set.prompts.iter().enumerate() {
            let id = format!("itl-prefill-{index}");
            let req = sample_request(id, prompt.clone(), cfg.prefill_max_tokens);
            let started_ms = epoch.elapsed().as_secs_f64() * 1_000.0;
            let sample = match ep.complete(&req) {
                Ok(outcome) => {
                    let computed = outcome.computed_prefill_tokens();
                    let void_reason = ttft::coldness_failure(computed, cfg.prefill_prompt_tokens);
                    PrefillerSample {
                        index,
                        started_ms,
                        first_token_ms: started_ms + outcome.ttft_ms,
                        ttft_ms: outcome.ttft_ms,
                        computed_prefill_tokens: computed,
                        void: void_reason.is_some(),
                        void_reason,
                    }
                }
                Err(err) => PrefillerSample {
                    index,
                    started_ms,
                    first_token_ms: started_ms,
                    ttft_ms: 0.0,
                    computed_prefill_tokens: None,
                    void: true,
                    void_reason: Some(format!("the request failed: {err}")),
                },
            };
            prefillers.push(sample);
        }

        // End all lane streams together at the actual measurement boundary.
        lanes_running.store(false, std::sync::atomic::Ordering::Release);
        let cancelled_at_ms = epoch.elapsed().as_secs_f64() * 1_000.0;

        let lanes: Vec<DecodeLaneTrace> = lane_handles
            .into_iter()
            .map(|h| h.join().expect("a decode-lane thread panicked"))
            .collect();
        (lanes, prefillers, cancelled_at_ms)
    });

    // The measurement window: the span in which *every* lane was producing
    // tokens. Cancellation at the end of the series is the terminator the
    // fixture intends, but never a guaranteed one — a lane the endpoint
    // stopped at its own EOS, or one the safety cap ended, closes the
    // window earlier (GitHub #139). Measuring inside the window keeps
    // exactly the property the cell claims, and keeps it whichever of the
    // three ended the shortest lane: every pooled interval was measured
    // with all `decode_lanes` lanes decoding at once. A lane that failed on
    // the wire decoded no span at all, so it leaves no window to measure
    // in.
    //
    // A lane the *harness* closed was alive and generating right up to
    // that moment: its last token is merely the last one that arrived
    // before the stream was dropped, not where the lane stopped. So a
    // cancelled lane's span ends at the cancellation, and only a lane the
    // engine ended — on its own EOS, on the cap — ends at its last token.
    // Without that distinction the window's close would be a race between
    // a lane's final token and the harness's store, which is not a
    // property of anything being measured.
    let spans: Option<Vec<(f64, f64)>> = lanes
        .iter()
        .map(|lane| {
            if lane.error.is_some() {
                return None;
            }
            let first = lane.token_times_ms.first()?;
            let end = if lane.finish == LaneFinish::Window {
                cancelled_at_ms
            } else {
                lane.end_ms()
            };
            Some((lane.started_ms + first, end))
        })
        .collect();
    let (window_start_ms, window_end_ms) = match spans {
        Some(spans) if !spans.is_empty() => (
            spans.iter().map(|span| span.0).fold(f64::NEG_INFINITY, f64::max),
            spans.iter().map(|span| span.1).fold(f64::INFINITY, f64::min),
        ),
        _ => (0.0, 0.0),
    };

    // The prefill windows that closed inside it. Their *opening* needs no
    // test: the harness holds the first prefiller until every lane has
    // produced a token, so the series begins after `window_start_ms` by
    // construction.
    let covered: Vec<(f64, f64)> = prefillers
        .iter()
        .filter(|prefill| prefill.first_token_ms <= window_end_ms)
        .map(|prefill| (prefill.started_ms, prefill.first_token_ms))
        .collect();
    let prefillers_covered = covered.len();

    // Pool only intervals that lie inside the measurement window and
    // overlap one of those prefill windows. Shift each lane's
    // request-relative SSE timestamps onto the shared monotonic timeline
    // before testing either.
    let mut intervals_ms: Vec<f64> = Vec::new();
    if window_end_ms > window_start_ms {
        for lane in &lanes {
            intervals_ms.extend(lane.token_times_ms.windows(2).filter_map(|pair| {
                let interval_start = lane.started_ms + pair[0];
                let interval_end = lane.started_ms + pair[1];
                if interval_start < window_start_ms || interval_end > window_end_ms {
                    return None;
                }
                covered
                    .iter()
                    .any(|(prefill_start, prefill_first_token)| {
                        interval_start < *prefill_first_token && interval_end > *prefill_start
                    })
                    .then_some(pair[1] - pair[0])
            }));
        }
    }
    let mut sorted = intervals_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let (p50_ms, p95_ms, p99_ms, max_ms) = if sorted.is_empty() {
        (None, None, None, None)
    } else {
        (
            Some(percentile(&sorted, 50.0)),
            Some(percentile(&sorted, 95.0)),
            Some(percentile(&sorted, 99.0)),
            sorted.last().copied(),
        )
    };

    ItlCell {
        prefill_prompt_tokens: cfg.prefill_prompt_tokens,
        prefill_max_tokens: cfg.prefill_max_tokens,
        prefill_count: cfg.prefill_count,
        decode_prompt_tokens: cfg.decode_prompt_tokens,
        decode_max_tokens: cfg.decode_max_tokens,
        decode_lanes: cfg.decode_lanes,
        prefillers,
        lanes,
        window_start_ms,
        window_end_ms,
        prefillers_covered,
        intervals_ms,
        p50_ms,
        p95_ms,
        p99_ms,
        max_ms,
        error: None,
    }
}

// ── the record ───────────────────────────────────────────────────────────

/// A G3 record: what one engine measured, on which cells, in which
/// session. Mirrors [`ttft::Record`]'s identity fields, so a G3 gate check
/// enforces the same live/live discipline (ADR 0015).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// The measurement session both engines' records must share.
    pub session: String,
    /// Which engine this is ("ignis", "reference", ...).
    pub label: String,
    /// The endpoint measured.
    pub endpoint: String,
    /// The engine's own identity: the model id it reports at
    /// `GET /v1/models`.
    pub engine: String,
    /// The artifact this engine was serving, as the operator named it.
    pub artifact: String,
    /// The profile the engine was running in.
    pub profile: String,
    /// When the record was made (UTC, RFC 3339 seconds).
    pub date: String,
    pub c1: ThroughputCell,
    pub c4: ThroughputCell,
    pub itl: ItlCell,
}

impl Record {
    /// Serialize to pretty JSON (the on-disk record format).
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("serialize the record: {e}"))
    }

    /// Parse a record from JSON.
    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| format!("parse the record: {e}"))
    }

    /// Write the record to `path` as pretty JSON.
    pub fn write(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.to_json()?).map_err(|e| format!("write {}: {e}", path.display()))
    }

    /// Read a record previously written by [`Record::write`].
    pub fn read(path: &Path) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::from_json(&text)
    }

    /// A human-readable rendering for the terminal.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "g3 record  session={}  label={}  engine={}  profile={}\n",
            self.session, self.label, self.engine, self.profile
        ));
        out.push_str(&format!(
            "  endpoint={}  artifact={}  date={}\n",
            self.endpoint, self.artifact, self.date
        ));
        for (name, cell) in [("C=1", &self.c1), ("C=4", &self.c4)] {
            match &cell.error {
                Some(err) => out.push_str(&format!("  {name:<4} FAILED: {err}\n")),
                None => {
                    let bad = cell.bad_samples().len();
                    out.push_str(&format!(
                        "  {name:<4} aggregate {:>9.1} tok/s  over {} sequence(s){}\n",
                        cell.aggregate_tok_s,
                        cell.samples.len(),
                        if bad == 0 {
                            "  (all cold)".to_string()
                        } else {
                            format!("  ({bad} BAD)")
                        },
                    ));
                }
            }
        }
        match &self.itl.error {
            Some(err) => out.push_str(&format!("  ITL  FAILED: {err}\n")),
            None => {
                let void = self.itl.void_prefillers().len();
                out.push_str(&format!(
                    "  ITL  p50 {:>7.2} ms  p95 {:>7.2} ms  p99 {:>7.2} ms  max {:>7.2} ms  \
                     over {} intervals ({} prefillers{})\n",
                    self.itl.p50_ms.unwrap_or(0.0),
                    self.itl.p95_ms.unwrap_or(0.0),
                    self.itl.p99_ms.unwrap_or(0.0),
                    self.itl.max_ms.unwrap_or(0.0),
                    self.itl.intervals_ms.len(),
                    self.itl.prefillers.len(),
                    if void == 0 { String::new() } else { format!(", {void} VOID") },
                ));
                // GitHub #139: the span every lane shared, and how much of
                // the series it covered -- the size of the distribution the
                // percentiles above stand on.
                out.push_str(&format!(
                    "       window {:.3} -> {:.3} ms  covering {} of {} prefill window(s)\n",
                    self.itl.window_start_ms,
                    self.itl.window_end_ms,
                    self.itl.prefillers_covered,
                    self.itl.prefillers.len(),
                ));
                // GitHub #114: how each lane ended, always -- a reader
                // should never have to infer it from a token count.
                for lane in &self.itl.lanes {
                    out.push_str(&format!(
                        "       lane {:<16} {:>6} tokens  ended: {}
",
                        lane.id,
                        lane.n_tokens,
                        lane.finish.describe(),
                    ));
                }
                for warning in self.itl.warnings() {
                    out.push_str(&format!("  ITL  WARNING: {warning}
"));
                }
            }
        }
        out
    }
}

/// The C=1 / C=4 fixture (spec 03's table): both cells share it, differing
/// only in concurrency ([`C1_CONCURRENCY`] / [`C4_CONCURRENCY`]).
#[derive(Debug, Clone, Copy)]
pub struct ThroughputSpec {
    pub prompt_tokens: u32,
    pub max_tokens: u32,
}

impl Default for ThroughputSpec {
    fn default() -> Self {
        Self {
            prompt_tokens: THROUGHPUT_PROMPT_TOKENS,
            max_tokens: THROUGHPUT_MAX_TOKENS,
        }
    }
}

/// What a `g3` run measures and how the record identifies it. The fixtures
/// default to spec 03's table ([`ThroughputSpec::default`],
/// [`ItlConfig::default`]) — a test wanting a fast run overrides them with
/// smaller sizes, the same way [`crate::ttft::TtftConfig`] lets a caller
/// pick its cells rather than this crate hard-coding the production sizes
/// into every measurement.
#[derive(Debug, Clone)]
pub struct G3Config {
    /// Which engine this run is measuring ("ignis", "reference", ...).
    pub label: String,
    /// The profile the engine is running in.
    pub profile: String,
    /// The artifact the engine is serving, as the operator names it.
    pub artifact: String,
    /// The measurement session both engines' records must share.
    pub session: String,
    pub throughput: ThroughputSpec,
    pub itl: ItlConfig,
    /// A pre-tokenized prompt bank (whitespace-separated ids) to cut every
    /// cell's prompts from, when set (the `--corpus` flag): detokenized
    /// rotated windows instead of the filler generator's word growth —
    /// needed at the ITL cell's 32,768-token prefiller scale
    /// ([`ttft::generate_prompt`]'s doc comment: O(n²) at 32K). Absent, the
    /// filler generator is used (`ttft::TtftConfig`'s same default).
    pub corpus: Option<PathBuf>,
}

/// Measure the three G3 cells against one endpoint and return the record.
///
/// A configured corpus (`cfg.corpus`) is loaded once and every cell's
/// prompts are cut from it; a corpus that cannot be read fails all three
/// cells with that error rather than measuring some of them with a
/// different prompt source than the others.
pub fn measure(
    ep: &dyn Endpoint,
    template: &dyn PromptTemplate,
    engine: String,
    endpoint: String,
    cfg: &G3Config,
) -> Record {
    let corpus = cfg.corpus.as_ref().map(|path| load_corpus(path));
    let (c1, c4, itl) = match &corpus {
        Some(Ok(ids)) => (
            measure_throughput_cell_from_corpus(
                ep,
                template,
                cfg.throughput.prompt_tokens,
                cfg.throughput.max_tokens,
                C1_CONCURRENCY,
                ids,
            ),
            measure_throughput_cell_from_corpus(
                ep,
                template,
                cfg.throughput.prompt_tokens,
                cfg.throughput.max_tokens,
                C4_CONCURRENCY,
                ids,
            ),
            measure_itl_from_corpus(ep, template, &cfg.itl, ids),
        ),
        Some(Err(error)) => (
            ThroughputCell::failed(
                cfg.throughput.prompt_tokens,
                cfg.throughput.max_tokens,
                C1_CONCURRENCY,
                error.clone(),
            ),
            ThroughputCell::failed(
                cfg.throughput.prompt_tokens,
                cfg.throughput.max_tokens,
                C4_CONCURRENCY,
                error.clone(),
            ),
            ItlCell::failed(&cfg.itl, error.clone()),
        ),
        None => (
            measure_throughput_cell(
                ep,
                template,
                cfg.throughput.prompt_tokens,
                cfg.throughput.max_tokens,
                C1_CONCURRENCY,
            ),
            measure_throughput_cell(
                ep,
                template,
                cfg.throughput.prompt_tokens,
                cfg.throughput.max_tokens,
                C4_CONCURRENCY,
            ),
            measure_itl(ep, template, &cfg.itl),
        ),
    };
    Record {
        session: cfg.session.clone(),
        label: cfg.label.clone(),
        endpoint,
        engine,
        artifact: cfg.artifact.clone(),
        profile: cfg.profile.clone(),
        date: utc_timestamp(unix_now()),
        c1,
        c4,
        itl,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock template: one token per whitespace-separated word, no
    /// header/footer overhead — so it matches [`MockEndpoint`]'s own
    /// prompt-token count exactly (a whitespace word count), the way
    /// `tests/ttft_cell.rs`'s `WordTemplate` matches the in-process mock
    /// engine. Enough to exercise exact-length generation and the void
    /// rule without an artifact (ADR 0006).
    struct MockTemplate {
        header: usize,
        footer: usize,
    }

    impl MockTemplate {
        fn new() -> Self {
            Self { header: 0, footer: 0 }
        }
    }

    impl PromptTemplate for MockTemplate {
        fn encode_user_message(&self, content: &str) -> Result<Vec<u32>, String> {
            let mut ids: Vec<u32> = (0..self.header).map(|i| 1_000 + i as u32).collect();
            for word in content.split_whitespace() {
                ids.push(word.bytes().fold(7u32, |a, b| a.wrapping_mul(131).wrapping_add(b as u32)));
            }
            ids.extend((0..self.footer).map(|i| 2_000 + i as u32));
            Ok(ids)
        }
        fn decode(&self, ids: &[u32]) -> Result<String, String> {
            Ok(ids.iter().map(|id| format!("t{id}")).collect::<Vec<_>>().join(" "))
        }
    }

    /// A mock endpoint standing in for the wire: deterministic per-request
    /// timing (fixed ttft, fixed decode interval) plus the cold-prefix
    /// evidence the void rule reads. `cached` lets a test simulate a warm
    /// prefix (a cache hit) on every request.
    struct MockEndpoint {
        cached: std::sync::atomic::AtomicU32,
        fail: std::sync::atomic::AtomicBool,
    }

    impl MockEndpoint {
        fn new() -> Self {
            Self {
                cached: std::sync::atomic::AtomicU32::new(0),
                fail: std::sync::atomic::AtomicBool::new(false),
            }
        }
        fn set_cached(&self, cached: u32) {
            self.cached.store(cached, std::sync::atomic::Ordering::Relaxed);
        }
        fn set_fail(&self, fail: bool) {
            self.fail.store(fail, std::sync::atomic::Ordering::Relaxed);
        }
    }

    impl Endpoint for MockEndpoint {
        fn complete(&self, req: &Request) -> Result<crate::client::Outcome, String> {
            if self.fail.load(std::sync::atomic::Ordering::Relaxed) {
                return Err("mock endpoint down".to_string());
            }
            let prompt_tokens = req.prompt.split_whitespace().count() as u32;
            let cached = self.cached.load(std::sync::atomic::Ordering::Relaxed);
            let n = req.max_tokens;
            let ttft = 50.0_f64;
            let interval = 5.0_f64;
            let token_times_ms: Vec<f64> = (0..n).map(|i| ttft + i as f64 * interval).collect();
            let total_ms = token_times_ms.last().copied().unwrap_or(ttft);
            Ok(crate::client::Outcome {
                ttft_ms: ttft,
                total_ms,
                n_tokens: n,
                output: String::new(),
                reasoning_output: String::new(),
                reasoning_tokens: Some(0),
                prompt_tokens: Some(prompt_tokens),
                cached_prompt_tokens: if cached > 0 { Some(cached) } else { None },
                token_times_ms,
                finish_reason: Some(crate::client::FinishReason::Engine("length".into())),
            })
        }
    }

    #[test]
    fn a_throughput_cell_measures_every_sequence_cold() {
        let ep = MockEndpoint::new();
        let template = MockTemplate::new();
        let cell = measure_throughput_cell(&ep, &template, 32, 8, 4);
        assert_eq!(cell.samples.len(), 4);
        assert!(cell.all_cold(), "bad samples: {:?}", cell.bad_samples());
        assert!(cell.aggregate_tok_s > 0.0, "a throughput figure must be measured");
        // Every sequence got its own prompt (the divergence proof already
        // covers this — this just checks the cell actually used it).
        let prompts: std::collections::HashSet<String> =
            cell.samples.iter().map(|s| s.id.clone()).collect();
        assert_eq!(prompts.len(), 4, "four distinct request ids");
    }

    #[test]
    fn c1_is_concurrency_one_and_c4_is_four() {
        let ep = MockEndpoint::new();
        let template = MockTemplate::new();
        let c1 = measure_throughput_cell(&ep, &template, 32, 8, C1_CONCURRENCY);
        let c4 = measure_throughput_cell(&ep, &template, 32, 8, C4_CONCURRENCY);
        assert_eq!(c1.samples.len(), 1);
        assert_eq!(c4.samples.len(), 4);
    }

    #[test]
    fn a_cache_hit_makes_every_throughput_sample_void() {
        let ep = MockEndpoint::new();
        ep.set_cached(10);
        let template = MockTemplate::new();
        let cell = measure_throughput_cell(&ep, &template, 32, 8, 2);
        assert!(!cell.all_cold());
        assert_eq!(cell.bad_samples().len(), 2);
        for sample in &cell.samples {
            assert!(sample.void);
            assert!(sample.void_reason.as_deref().unwrap().contains("cache"));
        }
    }

    #[test]
    fn a_failed_request_is_recorded_as_a_bad_sample_not_a_panic() {
        let ep = MockEndpoint::new();
        ep.set_fail(true);
        let template = MockTemplate::new();
        let cell = measure_throughput_cell(&ep, &template, 32, 8, 2);
        assert!(!cell.all_cold());
        for sample in &cell.samples {
            assert!(!sample.ok);
            assert!(sample.void_reason.as_deref().unwrap().contains("failed"));
        }
    }

    #[test]
    fn zero_concurrency_is_refused() {
        let ep = MockEndpoint::new();
        let template = MockTemplate::new();
        let cell = measure_throughput_cell(&ep, &template, 32, 8, 0);
        assert!(cell.error.is_some());
    }

    #[test]
    fn itl_runs_prefillers_sequentially_and_pools_only_overlapping_intervals() {
        let template = MockTemplate::new();
        let cfg = ItlConfig {
            prefill_prompt_tokens: 40,
            prefill_max_tokens: 4,
            prefill_count: 3,
            decode_prompt_tokens: 24,
            decode_max_tokens: 6,
            decode_lanes: 2,
        };
        // Paced against the series rather than against the scheduler: the
        // cell's timeline spans two kinds of request, so a fixture whose
        // canned token times bear no relation to when the harness actually
        // sent anything cannot say which intervals overlap (GitHub #146).
        let ep = PacedSeries::until_cancelled(cfg.decode_lanes, cfg.prefill_count, 2);
        let cell = measure_itl(&ep, &template, &cfg);
        assert!(cell.error.is_none(), "{:?}", cell.error);
        assert_eq!(cell.prefillers.len(), 3);
        assert_eq!(cell.lanes.len(), 2);
        assert!(cell.all_cold(), "void prefillers: {:?}", cell.void_prefillers());
        let measured: usize =
            cell.lanes.iter().map(|lane| lane.token_times_ms.len().saturating_sub(1)).sum();
        assert!(
            cell.intervals_ms.len() < measured,
            "intervals outside the prefill windows must be excluded: pooled {} of {measured}",
            cell.intervals_ms.len()
        );
        assert!(cell.p50_ms.is_some() && cell.p95_ms.is_some() && cell.p99_ms.is_some());
        let expected_max = cell.intervals_ms.iter().cloned().fold(f64::MIN, f64::max);
        assert_eq!(cell.max_ms, Some(expected_max));
    }

    /// How far apart [`PacedSeries`] spaces a lane's tokens (see its emit
    /// loop for why a fixture needs real spacing at all).
    const TOKEN_PACE: std::time::Duration = std::time::Duration::from_millis(2);

    /// A fixture whose decode lanes are paced by the prefiller series
    /// rather than by the OS scheduler (GitHub #146).
    ///
    /// Each lane may emit `tokens_per_prefill` tokens before the series
    /// starts, and `tokens_per_prefill` more for each prefiller that has
    /// begun; each prefiller returns only once every lane has emitted its
    /// allowance (or stopped itself at its own `lane_caps` entry). Which
    /// terminator ends a lane is therefore decided by those two numbers,
    /// never by how the machine happened to schedule the lane threads —
    /// which is the whole point of #146: the previous fixtures emitted
    /// their canned tokens with no pacing at all, so a loaded machine could
    /// close a lane at the measurement boundary where an idle one let it
    /// reach the cap.
    struct PacedSeries {
        state: std::sync::Mutex<PacedState>,
        ready: std::sync::Condvar,
        tokens_per_prefill: u32,
        /// Per lane: the tokens after which it stops itself and reports
        /// `"length"` (the engine reaching the safety cap), or `None` for a
        /// lane that only ever ends when the harness cancels it. Lanes may
        /// differ, which is the shape #139 found live — one lane spent well
        /// before the others.
        lane_caps: Vec<Option<u32>>,
        prefill_count: usize,
    }

    struct PacedState {
        /// Prefillers begun so far — what the lanes' allowance is measured
        /// in.
        round: usize,
        /// Tokens emitted so far, per lane index.
        emitted: Vec<u32>,
        /// Every prefiller has been served: the lanes may now run freely
        /// until the harness closes them.
        series_over: bool,
    }

    impl PacedSeries {
        /// A lane that runs until the harness cancels it — the terminator
        /// the fixture intends.
        fn until_cancelled(lanes: usize, prefill_count: usize, tokens_per_prefill: u32) -> Self {
            Self::new(prefill_count, tokens_per_prefill, vec![None; lanes])
        }

        /// A lane that stops itself after `cap` tokens, the way an engine
        /// fast enough to exhaust [`ITL_DECODE_MAX_TOKENS`] does.
        fn capped_at(
            lanes: usize,
            prefill_count: usize,
            tokens_per_prefill: u32,
            cap: u32,
        ) -> Self {
            Self::new(prefill_count, tokens_per_prefill, vec![Some(cap); lanes])
        }

        /// Lanes spent at different points in the series — one stops well
        /// before the others, so the window it closes provably leaves the
        /// survivors' later intervals outside it.
        fn capped_per_lane(prefill_count: usize, tokens_per_prefill: u32, caps: &[u32]) -> Self {
            Self::new(
                prefill_count,
                tokens_per_prefill,
                caps.iter().map(|cap| Some(*cap)).collect(),
            )
        }

        fn new(
            prefill_count: usize,
            tokens_per_prefill: u32,
            lane_caps: Vec<Option<u32>>,
        ) -> Self {
            Self {
                state: std::sync::Mutex::new(PacedState {
                    round: 0,
                    emitted: vec![0; lane_caps.len()],
                    series_over: false,
                }),
                ready: std::sync::Condvar::new(),
                tokens_per_prefill,
                lane_caps,
                prefill_count,
            }
        }

        /// The tokens lane `index` must have emitted before the prefiller of
        /// round `round` may return — never more than that lane's own cap
        /// will ever produce, which would deadlock the series.
        fn target(&self, round: usize, index: usize) -> u32 {
            let owed = (round as u32 + 1) * self.tokens_per_prefill;
            self.lane_caps[index].map_or(owed, |cap| owed.min(cap))
        }
    }

    impl Endpoint for PacedSeries {
        fn complete(&self, req: &Request) -> Result<crate::client::Outcome, String> {
            if !req.id.starts_with("itl-prefill") {
                return Err(format!("this fixture only serves prefillers here, got `{}`", req.id));
            }
            let start = std::time::Instant::now();
            let round = {
                let mut state = self.state.lock().unwrap();
                state.round += 1;
                self.ready.notify_all();
                state.round
            };
            {
                let mut state = self.state.lock().unwrap();
                while state
                    .emitted
                    .iter()
                    .enumerate()
                    .any(|(index, &emitted)| emitted < self.target(round, index))
                {
                    state = self.ready.wait(state).unwrap();
                }
                if round == self.prefill_count {
                    state.series_over = true;
                    self.ready.notify_all();
                }
            }
            // A prefiller that has nothing left to wait for still costs
            // something: in production this is a 32,768-token prefill. A
            // zero-width request-start -> first-token window is not a
            // window at all, and the cell rightly refuses one.
            std::thread::sleep(TOKEN_PACE);
            let ttft_ms = start.elapsed().as_secs_f64() * 1_000.0;
            Ok(crate::client::Outcome {
                ttft_ms,
                total_ms: ttft_ms,
                n_tokens: 1,
                output: String::new(),
                reasoning_output: String::new(),
                reasoning_tokens: Some(0),
                prompt_tokens: Some(req.prompt.split_whitespace().count() as u32),
                cached_prompt_tokens: None,
                token_times_ms: vec![ttft_ms],
                finish_reason: Some(crate::client::FinishReason::Engine("length".into())),
            })
        }

        fn complete_observed_while(
            &self,
            req: &Request,
            observer: &mut dyn FnMut(f64) -> bool,
            _suppress_eos: &[u32],
        ) -> Result<crate::client::Outcome, String> {
            // A lane that never stops on its own. If cancellation failed to
            // reach it, it runs to this ceiling instead of blocking
            // forever — which is why the tests below check both ends.
            const NEVER_CANCELLED: usize = 100_000;

            let index: usize = req
                .id
                .rsplit('-')
                .next()
                .and_then(|tail| tail.parse().ok())
                .ok_or_else(|| format!("not a decode-lane id: `{}`", req.id))?;
            let start = std::time::Instant::now();
            let mut token_times_ms: Vec<f64> = Vec::new();
            let mut cancelled = false;
            while token_times_ms.len() < NEVER_CANCELLED {
                // Wait for the allowance this lane is owed. Once the series
                // is over there is nothing left to pace against, so the
                // lane runs free until the harness closes it.
                {
                    let mut state = self.state.lock().unwrap();
                    loop {
                        let allowance = (state.round as u32 + 1) * self.tokens_per_prefill;
                        if state.series_over || state.emitted[index] < allowance {
                            break;
                        }
                        state = self.ready.wait(state).unwrap();
                    }
                }
                // Real spacing between tokens. Both halves of this cell's
                // timeline are assembled from two clocks -- the harness
                // stamps a request's start, the endpoint times the tokens
                // within it -- so events microseconds apart cannot be
                // ordered against each other under load. Production tokens
                // are tens of milliseconds apart and the question never
                // arises; a fixture has to buy the same margin.
                std::thread::sleep(TOKEN_PACE);
                let time_ms = start.elapsed().as_secs_f64() * 1_000.0;
                token_times_ms.push(time_ms);
                {
                    let mut state = self.state.lock().unwrap();
                    state.emitted[index] += 1;
                    self.ready.notify_all();
                }
                // The cap is the *engine* stopping, so it is decided
                // before the client is asked for more: a lane whose cap
                // falls on the same round the harness cancels must still
                // record `Cap`, or the fixture has handed #146's coin flip
                // back to the scheduler.
                if self.lane_caps[index].is_some_and(|cap| token_times_ms.len() as u32 >= cap) {
                    break;
                }
                if !observer(time_ms) {
                    cancelled = true;
                    break;
                }
            }
            Ok(crate::client::Outcome {
                ttft_ms: token_times_ms.first().copied().unwrap_or(0.0),
                total_ms: token_times_ms.last().copied().unwrap_or(0.0),
                n_tokens: token_times_ms.len() as u32,
                output: String::new(),
                reasoning_output: String::new(),
                reasoning_tokens: Some(0),
                prompt_tokens: None,
                cached_prompt_tokens: None,
                token_times_ms,
                finish_reason: Some(if cancelled {
                    crate::client::FinishReason::Cancelled
                } else {
                    // The lane stopped itself at the cap (GitHub #114).
                    crate::client::FinishReason::Engine("length".into())
                }),
            })
        }
    }

    /// GitHub #146: the lanes outlive the series by the fixture's own
    /// arithmetic — two tokens before the first prefiller and two more per
    /// prefiller after it — so `Window` is what ends them on an idle
    /// machine and on a loaded one alike.
    #[test]
    fn itl_decode_lanes_are_stopped_after_the_final_prefill_window() {
        const NEVER_CANCELLED: u32 = 100_000;
        let cfg = ItlConfig {
            prefill_prompt_tokens: 40,
            prefill_max_tokens: 1,
            prefill_count: 3,
            decode_prompt_tokens: 24,
            decode_max_tokens: 2,
            decode_lanes: 2,
        };
        let ep = PacedSeries::until_cancelled(cfg.decode_lanes, cfg.prefill_count, 2);
        let cell = measure_itl(&ep, &MockTemplate::new(), &cfg);

        assert!(cell.error.is_none(), "{:?}", cell.error);
        assert!(cell.lanes.iter().all(|lane| lane.error.is_none()), "{:?}", cell.lanes);
        assert!(
            cell.lanes.iter().all(|lane| lane.n_tokens > cfg.decode_max_tokens),
            "the measured window, not the fixture's old token cap, must end each lane: {:?}",
            cell.lanes.iter().map(|lane| lane.n_tokens).collect::<Vec<_>>()
        );
        assert!(
            cell.lanes.iter().all(|lane| lane.n_tokens < NEVER_CANCELLED),
            "each lane must be *stopped* at the window boundary, not merely run out"
        );
        // GitHub #114: the record has to *say* that, rather than leave a
        // reader to infer it from the token counts above.
        assert!(
            cell.lanes.iter().all(|lane| lane.finish == LaneFinish::Window),
            "the boundary closed every lane, so every lane must record it: {:?}",
            cell.lanes.iter().map(|lane| lane.finish).collect::<Vec<_>>()
        );
        assert!(cell.lanes_on_cap().is_empty());
        // GitHub #139: lanes that outlived the series cover all of it.
        assert_eq!(cell.prefillers_covered, cfg.prefill_count);
        assert_eq!(cell.window_warning(), None, "the window covered the whole series");
        assert!(cell.all_cold(), "void: {:?}", cell.void_prefillers());
    }

    /// GitHub #146 + #139. The lanes stop themselves at the cap partway
    /// through the series — deterministically, because the cap is reached
    /// on the fixture's own token allowance rather than on a race. That
    /// shortens the measurement window instead of voiding the cell
    /// (GitHub #139), and the record says by how much.
    #[test]
    fn lanes_that_end_on_the_cap_shorten_the_window_and_are_named_in_a_warning() {
        const LANE_CAP: u32 = 10;
        let cfg = ItlConfig {
            prefill_prompt_tokens: 40,
            prefill_max_tokens: 1,
            prefill_count: 6,
            decode_prompt_tokens: 24,
            decode_max_tokens: LANE_CAP,
            decode_lanes: 2,
        };
        let ep = PacedSeries::capped_at(cfg.decode_lanes, cfg.prefill_count, 2, LANE_CAP);
        let cell = measure_itl(&ep, &MockTemplate::new(), &cfg);

        assert!(cell.error.is_none(), "{:?}", cell.error);
        assert_eq!(cell.lanes_on_cap().len(), 2, "{:?}", cell.lanes);
        assert!(cell.lanes.iter().all(|lane| lane.n_tokens == LANE_CAP), "{:?}", cell.lanes);
        // Two tokens before the series and two per prefiller: the lanes are
        // spent partway through, so some prefill windows closed inside the
        // measurement window and the rest did not. How many is a property
        // of the run rather than of the contract — what the contract says
        // is that the window shortened and the cell survived it.
        assert!(
            cell.prefillers_covered < cfg.prefill_count,
            "the spent lanes must shorten the window: covered {} of {}",
            cell.prefillers_covered,
            cfg.prefill_count
        );
        assert!(
            cell.prefillers_covered >= ITL_MIN_COVERED_PREFILLERS,
            "the lanes outlived enough of the series to keep the cell: covered {}",
            cell.prefillers_covered
        );
        assert!(
            cell.window_end_ms > cell.window_start_ms,
            "the lanes shared a span: {} -> {}",
            cell.window_start_ms,
            cell.window_end_ms
        );
        // A shorter measurement, not a refused one (GitHub #139).
        assert!(
            cell.all_cold(),
            "void: {:?}; window {} -> {}; covered {}; intervals {}; lanes {:?}; prefillers {:?}",
            cell.void_prefillers(),
            cell.window_start_ms,
            cell.window_end_ms,
            cell.prefillers_covered,
            cell.intervals_ms.len(),
            cell.lanes
                .iter()
                .map(|lane| (
                    lane.started_ms,
                    lane.token_times_ms.first().copied(),
                    lane.token_times_ms.last().copied()
                ))
                .collect::<Vec<_>>(),
            cell.prefillers
                .iter()
                .map(|p| (p.started_ms, p.first_token_ms))
                .collect::<Vec<_>>(),
        );
        let warning = cell.window_warning().expect("a shortened window must warn");
        assert!(
            warning.contains(&format!(
                "{} of {}",
                cell.prefillers_covered, cfg.prefill_count
            )),
            "the warning states the coverage: {warning}"
        );
        assert!(warning.contains("itl-decode-"), "the warning names the lane: {warning}");
        assert!(warning.contains("safety cap"), "the warning says how it ended: {warning}");
    }

    /// GitHub #139: a window too short to cover
    /// [`ITL_MIN_COVERED_PREFILLERS`] prefill windows is an anecdote rather
    /// than a distribution, so the cell refuses it instead of reporting a
    /// percentile over one prefill.
    #[test]
    fn a_window_covering_too_little_of_the_series_is_not_all_cold() {
        const LANE_CAP: u32 = 2;
        let cfg = ItlConfig {
            prefill_prompt_tokens: 40,
            prefill_max_tokens: 1,
            prefill_count: 4,
            decode_prompt_tokens: 24,
            decode_max_tokens: LANE_CAP,
            decode_lanes: 2,
        };
        let ep = PacedSeries::capped_at(cfg.decode_lanes, cfg.prefill_count, 2, LANE_CAP);
        let cell = measure_itl(&ep, &MockTemplate::new(), &cfg);

        // The lanes spend their whole budget on the allowance they get
        // before the first prefiller is even sent, so the measurement
        // window closes before the series has a distribution in it.
        assert!(
            cell.prefillers_covered < ITL_MIN_COVERED_PREFILLERS,
            "covered {} of {}",
            cell.prefillers_covered,
            cfg.prefill_count
        );
        assert!(!cell.all_cold(), "a window this short cannot decide a gate");
        assert!(cell.window_warning().is_some());
    }

    /// GitHub #139: an interval measured after the shortest lane died had
    /// fewer lanes decoding beside it than the cell claims, so it stays out
    /// of the pool even when it overlaps a prefill window.
    #[test]
    fn intervals_outside_the_shared_window_are_not_pooled() {
        let cfg = ItlConfig {
            prefill_prompt_tokens: 40,
            prefill_max_tokens: 1,
            prefill_count: 8,
            decode_prompt_tokens: 24,
            decode_max_tokens: 14,
            decode_lanes: 2,
        };
        // Lane 0 is spent four tokens before lane 1, the way #139's lane-0
        // was spent well before the rest. Lane 1's last four intervals are
        // therefore measured with only one lane decoding — outside the
        // window by construction, so this test cannot pass vacuously.
        let ep = PacedSeries::capped_per_lane(cfg.prefill_count, 2, &[10, 14]);
        let cell = measure_itl(&ep, &MockTemplate::new(), &cfg);

        assert_eq!(cell.lanes[0].n_tokens, 10, "{:?}", cell.lanes);
        assert_eq!(cell.lanes[1].n_tokens, 14, "{:?}", cell.lanes);
        assert!(!cell.intervals_ms.is_empty(), "the covered windows pooled something");

        let mut inside = 0usize;
        let mut outside = 0usize;
        for lane in &cell.lanes {
            for pair in lane.token_times_ms.windows(2) {
                let within = lane.started_ms + pair[0] >= cell.window_start_ms
                    && lane.started_ms + pair[1] <= cell.window_end_ms;
                if within {
                    inside += 1;
                } else {
                    outside += 1;
                }
            }
        }
        assert!(
            outside > 0,
            "lane 1 outlived lane 0, so some of its intervals must fall outside the window"
        );
        assert!(
            cell.intervals_ms.len() <= inside,
            "pooled {} intervals but only {inside} lie inside the shared window",
            cell.intervals_ms.len()
        );
    }

    /// GitHub #114 + #139. A lane that exhausts the safety cap *after* the
    /// last prefill window closed costs the leg no coverage, so the window
    /// warning rightly stays silent — and the cap warning must not, or the
    /// very fact #114 exists to surface goes unsaid on #110's own shape.
    #[test]
    fn a_cap_that_cost_no_coverage_is_still_warned_about() {
        let lane = |id: &str, finish: LaneFinish| DecodeLaneTrace {
            id: id.to_string(),
            started_ms: 0.0,
            n_tokens: 2,
            token_times_ms: vec![1.0, 900.0],
            finish,
            error: None,
        };
        let cell = ItlCell {
            prefill_prompt_tokens: 32_768,
            prefill_max_tokens: 64,
            prefill_count: 2,
            decode_prompt_tokens: 4_096,
            decode_max_tokens: ITL_DECODE_MAX_TOKENS,
            decode_lanes: 2,
            prefillers: (0..2)
                .map(|index| PrefillerSample {
                    index,
                    started_ms: index as f64 * 100.0,
                    first_token_ms: index as f64 * 100.0 + 90.0,
                    ttft_ms: 90.0,
                    computed_prefill_tokens: Some(32_768),
                    void: false,
                    void_reason: None,
                })
                .collect(),
            lanes: vec![lane("itl-decode-0", LaneFinish::Cap), lane("itl-decode-1", LaneFinish::Window)],
            window_start_ms: 1.0,
            window_end_ms: 900.0,
            prefillers_covered: 2,
            intervals_ms: vec![5.0, 6.0],
            p50_ms: Some(5.5),
            p95_ms: Some(6.0),
            p99_ms: Some(6.0),
            max_ms: Some(6.0),
            error: None,
        };

        assert!(cell.all_cold(), "full coverage: the cap cost this leg nothing");
        assert_eq!(cell.window_warning(), None, "nothing was lost, so nothing to say about coverage");
        let warning = cell.cap_warning().expect("a load-bearing cap must still be named");
        assert!(warning.contains("itl-decode-0"), "the warning names the lane: {warning}");
        assert!(
            warning.contains(&ITL_DECODE_MAX_TOKENS.to_string()),
            "the warning states the cap: {warning}"
        );
        assert!(!warning.contains("itl-decode-1"), "only the capped lane: {warning}");
        assert_eq!(cell.warnings(), vec![warning], "the reader gets exactly this one");
    }

    /// GitHub #114: every transport outcome the instrument can see maps to
    /// exactly one recorded reason, so a record never has to be read by
    /// guessing.
    #[test]
    fn every_transport_outcome_maps_to_one_recorded_reason() {
        use crate::client::FinishReason;
        let cases = [
            (Some(FinishReason::Cancelled), LaneFinish::Window),
            (Some(FinishReason::Engine("length".into())), LaneFinish::Cap),
            (Some(FinishReason::Engine("stop".into())), LaneFinish::StopToken),
            (Some(FinishReason::Engine("content_filter".into())), LaneFinish::Other),
            (None, LaneFinish::Unknown),
        ];
        for (transport, expected) in cases {
            assert_eq!(
                LaneFinish::from_transport(transport.as_ref()),
                expected,
                "{transport:?}"
            );
        }
        assert!(LaneFinish::Cap.is_cap());
        assert!(!LaneFinish::Window.is_cap());
    }

    /// GitHub #114: the cap is the largest the engine's KV pool admits, and
    /// the arithmetic that says so is checked rather than asserted in a
    /// comment. Admission reserves the full `ceil((prompt + budget) /
    /// page)` up front (`ignis_core::admission::AdmissionResources`), and
    /// the pool is the 65,536 tokens `ignis_runtime::auto_kv_pool_bytes`'s
    /// 4 GiB default buys under BF16 KV at the server's default
    /// `--max-context`. Those two numbers are mirrored
    /// here rather than imported: this crate measures any OpenAI-compatible
    /// engine and does not depend on ignis's own runtime.
    #[test]
    fn the_decode_cap_is_the_largest_the_kv_pool_admits() {
        const PAGE_TOKENS: u32 = 64;
        const POOL_PAGES: u32 = 65_536 / PAGE_TOKENS;
        let pages = |tokens: u32| tokens.div_ceil(PAGE_TOKENS);
        let peak = |cap: u32| {
            ITL_DECODE_LANES as u32 * pages(ITL_DECODE_PROMPT_TOKENS + cap)
                + pages(ITL_PREFILL_PROMPT_TOKENS + ITL_PREFILL_MAX_TOKENS)
        };
        assert!(
            peak(ITL_DECODE_MAX_TOKENS) <= POOL_PAGES,
            "the fixture's peak reservation ({} pages) must fit the pool ({POOL_PAGES})",
            peak(ITL_DECODE_MAX_TOKENS)
        );
        // And it is the *largest* such cap: one more page per lane does not
        // fit, so nothing is being left on the table.
        assert!(
            peak(ITL_DECODE_MAX_TOKENS + PAGE_TOKENS) > POOL_PAGES,
            "a cap {} tokens higher would still fit -- raise it",
            PAGE_TOKENS
        );
    }

    #[test]
    fn a_warm_prefiller_is_void_and_the_cell_is_not_all_cold() {
        let ep = MockEndpoint::new();
        ep.set_cached(8);
        let template = MockTemplate::new();
        let cfg = ItlConfig {
            prefill_prompt_tokens: 40,
            prefill_max_tokens: 4,
            prefill_count: 2,
            decode_prompt_tokens: 24,
            decode_max_tokens: 4,
            decode_lanes: 1,
        };
        let cell = measure_itl(&ep, &template, &cfg);
        assert_eq!(cell.void_prefillers().len(), 2);
        assert!(!cell.all_cold());
    }

    #[test]
    fn a_decode_lane_failure_is_recorded_and_the_cell_is_not_all_cold() {
        struct FlakyDecode(MockEndpoint);
        impl Endpoint for FlakyDecode {
            fn complete(&self, req: &Request) -> Result<crate::client::Outcome, String> {
                if req.id.starts_with("itl-decode") {
                    return Err("decode lane dropped".to_string());
                }
                self.0.complete(req)
            }
        }
        let ep = FlakyDecode(MockEndpoint::new());
        let template = MockTemplate::new();
        let cfg = ItlConfig {
            prefill_prompt_tokens: 40,
            prefill_max_tokens: 4,
            prefill_count: 2,
            decode_prompt_tokens: 24,
            decode_max_tokens: 4,
            decode_lanes: 1,
        };
        let cell = measure_itl(&ep, &template, &cfg);
        assert!(!cell.all_cold());
        assert!(cell.lanes[0].error.is_some());
        assert!(cell.intervals_ms.is_empty(), "a failed lane contributes no intervals");
    }

    #[test]
    fn zero_lanes_or_zero_prefillers_is_refused() {
        let ep = MockEndpoint::new();
        let template = MockTemplate::new();
        let cfg = ItlConfig { decode_lanes: 0, ..ItlConfig::default() };
        assert!(measure_itl(&ep, &template, &cfg).error.is_some());
        let cfg = ItlConfig { prefill_count: 0, ..ItlConfig::default() };
        assert!(measure_itl(&ep, &template, &cfg).error.is_some());
    }

    #[test]
    fn a_record_round_trips_through_json() {
        let ep = MockEndpoint::new();
        let template = MockTemplate::new();
        let cfg = G3Config {
            label: "ignis".into(),
            profile: "test-profile".into(),
            artifact: "mock.ninfer".into(),
            session: "S1".into(),
            // Small fixtures: the production sizes (spec 03) exercise the
            // filler generator's O(n^2) growth (`ttft::generate_prompt`'s
            // doc comment) — fine for a real gate run, wasteful for a unit
            // test that only checks the record's shape round-trips.
            throughput: ThroughputSpec { prompt_tokens: 32, max_tokens: 8 },
            itl: ItlConfig {
                prefill_prompt_tokens: 40,
                prefill_max_tokens: 4,
                prefill_count: 2,
                decode_prompt_tokens: 24,
                decode_max_tokens: 4,
                decode_lanes: 1,
            },
            corpus: None,
        };
        let record = measure(&ep, &template, "mock-engine".into(), "http://mock".into(), &cfg);
        let json = record.to_json().expect("serialize");
        assert_eq!(Record::from_json(&json).expect("parse"), record);
        let text = record.render();
        assert!(text.contains("C=1") && text.contains("C=4") && text.contains("ITL"));
    }

    // ── the corpus path ────────────────────────────────────────────────────

    /// A template whose decode is the exact inverse of its encode (each id
    /// decodes to a `t<id>` atom that encodes back to the same id) — the
    /// same shape as `ttft.rs`'s own `CorpusMock`, needed here because
    /// [`MockTemplate`] never implements `decode` meaningfully.
    ///
    /// The one non-atom it accepts is [`ITL_DECODE_INSTRUCTION`], the
    /// natural-language suffix the ITL decode lanes carry: each of its
    /// words takes a reserved id above the corpus bank's range. Every
    /// *other* non-atom stays an error, so a corpus window that fails to
    /// round-trip is still caught rather than silently hashed.
    struct CorpusMock {
        header: usize,
        footer: usize,
    }

    impl CorpusMock {
        /// The id range reserved for the instruction suffix's words. Far
        /// above `corpus_bank`'s ids, so an instruction word can never be
        /// mistaken for a corpus atom.
        const INSTRUCTION_ID_BASE: u32 = 800_000;

        /// The reserved id for `word` when it belongs to the instruction
        /// suffix, or `None` when it does not. Read off the production
        /// constant, so a change to the suffix reaches the mock rather than
        /// silently bypassing it.
        fn instruction_id(word: &str) -> Option<u32> {
            ITL_DECODE_INSTRUCTION
                .split_whitespace()
                .position(|w| w == word)
                .map(|i| Self::INSTRUCTION_ID_BASE + i as u32)
        }
    }

    impl PromptTemplate for CorpusMock {
        fn encode_user_message(&self, content: &str) -> Result<Vec<u32>, String> {
            let mut ids: Vec<u32> = (0..self.header).map(|i| 1_000 + i as u32).collect();
            for word in content.split_whitespace() {
                let id = word
                    .strip_prefix('t')
                    .and_then(|s| s.parse::<u32>().ok())
                    .or_else(|| Self::instruction_id(word))
                    .ok_or_else(|| format!("the corpus mock only decodes 't<id>' atoms, got `{word}`"))?;
                ids.push(id);
            }
            ids.extend((0..self.footer).map(|i| 2_000 + i as u32));
            Ok(ids)
        }
        fn decode(&self, ids: &[u32]) -> Result<String, String> {
            Ok(ids.iter().map(|id| format!("t{id}")).collect::<Vec<_>>().join(" "))
        }
    }

    /// The bank the corpus tests cut from: distinct-ish ids, large enough
    /// for both the throughput and the ITL fixtures used below.
    fn corpus_bank() -> Vec<u32> {
        (0..2_000).map(|i| ((i * 37) % 900) as u32 + 100).collect()
    }

    #[test]
    fn a_throughput_cell_measures_cold_samples_from_a_corpus() {
        let ep = MockEndpoint::new();
        let template = CorpusMock { header: 0, footer: 0 };
        let bank = corpus_bank();
        let cell = measure_throughput_cell_from_corpus(&ep, &template, 40, 8, 3, &bank);
        assert!(cell.error.is_none(), "{:?}", cell.error);
        assert_eq!(cell.samples.len(), 3);
        assert!(cell.all_cold(), "bad samples: {:?}", cell.bad_samples());
    }

    #[test]
    fn an_itl_cell_measures_cold_samples_from_a_corpus() {
        let template = CorpusMock { header: 0, footer: 0 };
        let bank = corpus_bank();
        let cfg = ItlConfig {
            prefill_prompt_tokens: 60,
            prefill_max_tokens: 4,
            prefill_count: 3,
            decode_prompt_tokens: 24,
            decode_max_tokens: 4,
            decode_lanes: 2,
        };
        let ep = PacedSeries::until_cancelled(cfg.decode_lanes, cfg.prefill_count, 2);
        let cell = measure_itl_from_corpus(&ep, &template, &cfg, &bank);
        assert!(cell.error.is_none(), "{:?}", cell.error);
        assert!(cell.all_cold(), "void prefillers: {:?}", cell.void_prefillers());
        assert_eq!(cell.prefillers.len(), 3);
        assert_eq!(cell.lanes.len(), 2);
    }

    #[test]
    fn a_missing_corpus_file_fails_all_three_cells_with_one_error() {
        let ep = MockEndpoint::new();
        let template = MockTemplate::new();
        let cfg = G3Config {
            label: "ignis".into(),
            profile: "test-profile".into(),
            artifact: "mock.ninfer".into(),
            session: "S1".into(),
            throughput: ThroughputSpec { prompt_tokens: 32, max_tokens: 8 },
            itl: ItlConfig::default(),
            corpus: Some(PathBuf::from("this/path/does/not/exist.ids")),
        };
        let record = measure(&ep, &template, "mock-engine".into(), "http://mock".into(), &cfg);
        assert!(record.c1.error.is_some(), "C=1 must fail rather than fall back to the filler path");
        assert!(record.c4.error.is_some());
        assert!(record.itl.error.is_some());
    }
}
