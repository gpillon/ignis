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
//! - **ITL**: four decode lanes (prompt 4,096 / safety cap 3,072) sampled
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

/// The ITL cell's decode lanes: prompt 4,096, safety cap 3,072. Live
/// measurement found that the original 512-token estimate ended reference
/// lanes after 15-19 s while the prefiller series lasted about 62 s. The
/// series boundary now cancels first; 3,072 keeps enough runway while the
/// peak 61,504-token reservation still fits the specified 65,536-token pool.
pub const ITL_DECODE_PROMPT_TOKENS: u32 = 4_096;
pub const ITL_DECODE_MAX_TOKENS: u32 = 3_072;
pub const ITL_DECODE_LANES: usize = 4;
const ITL_DECODE_INSTRUCTION: &str =
    "Produce at least 3072 tokens. Do not stop, conclude, or emit EOS earlier.";

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
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
    /// Every decode lane's inter-token intervals that overlap a prefiller's
    /// request-start -> first-token window, pooled across the series.
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
    /// completed, and the cell has intervals to report — the property a
    /// gate verdict may be computed over.
    pub fn all_cold(&self) -> bool {
        self.error.is_none()
            && !self.prefillers.is_empty()
            && self
                .prefillers
                .iter()
                .all(|p| !p.void && p.first_token_ms > p.started_ms)
            && !self.lanes.is_empty()
            && self.lanes.iter().all(|l| l.error.is_none())
            && !self.intervals_ms.is_empty()
    }

    /// The prefillers that are not provably cold, with their reasons.
    pub fn void_prefillers(&self) -> Vec<&PrefillerSample> {
        self.prefillers.iter().filter(|p| p.void).collect()
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
    let (mut lanes, prefillers) = std::thread::scope(|scope| {
        // The decode lanes: one streaming request each, held open until the
        // sequential prefiller series ends, then cancelled.
        //
        // Cancellation is the *intended* terminator, not the guaranteed
        // one. It only gets the chance if the engine kept the lane alive
        // that long, which needs the endpoint to have honoured the EOS
        // suppression; an engine that honours neither `ignore_eos` nor
        // `logit_bias` stops at its own EOS instead, and one that is fast
        // enough reaches `decode_max_tokens` first. Either way the pooled
        // intervals stay honest, because the guard below refuses any lane
        // that ended before the final prefill window closed.
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
                            token_times_ms: outcome.token_times_ms,
                            error: None,
                        },
                        Err(err) => DecodeLaneTrace {
                            id,
                            started_ms,
                            n_tokens: 0,
                            token_times_ms: Vec::new(),
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

        let lanes: Vec<DecodeLaneTrace> = lane_handles
            .into_iter()
            .map(|h| h.join().expect("a decode-lane thread panicked"))
            .collect();
        (lanes, prefillers)
    });

    // Pool only intervals overlapping a prefiller's request-start -> first-
    // token window. Shift each lane's request-relative SSE timestamps onto
    // the shared monotonic timeline before testing the intersection.
    let last_prefill_end = prefillers.iter().map(|p| p.first_token_ms).max_by(f64::total_cmp);
    let mut intervals_ms: Vec<f64> = Vec::new();
    for lane in &mut lanes {
        if lane.error.is_some() {
            continue;
        }
        let last_lane_token = lane.token_times_ms.last().map(|time| lane.started_ms + time);
        if last_prefill_end.is_some_and(|end| last_lane_token.is_none_or(|last| last < end)) {
            lane.error = Some(format!(
                "decode lane ended at {:.3} ms before the final prefill window ended at {:.3} ms",
                last_lane_token.unwrap_or(lane.started_ms),
                last_prefill_end.unwrap_or_default()
            ));
            continue;
        }
        intervals_ms.extend(lane.token_times_ms.windows(2).filter_map(|window| {
            let interval_start = lane.started_ms + window[0];
            let interval_end = lane.started_ms + window[1];
            prefillers
                .iter()
                .any(|prefill| {
                    interval_start < prefill.first_token_ms && interval_end > prefill.started_ms
                })
                .then_some(window[1] - window[0])
        }));
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
                prompt_tokens: Some(prompt_tokens),
                cached_prompt_tokens: if cached > 0 { Some(cached) } else { None },
                token_times_ms,
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
        let ep = MockEndpoint::new();
        let template = MockTemplate::new();
        let cfg = ItlConfig {
            prefill_prompt_tokens: 40,
            prefill_max_tokens: 4,
            prefill_count: 3,
            decode_prompt_tokens: 24,
            decode_max_tokens: 6,
            decode_lanes: 2,
        };
        let cell = measure_itl(&ep, &template, &cfg);
        assert!(cell.error.is_none(), "{:?}", cell.error);
        assert_eq!(cell.prefillers.len(), 3);
        assert_eq!(cell.lanes.len(), 2);
        assert!(cell.all_cold(), "void prefillers: {:?}", cell.void_prefillers());
        assert!(
            cell.intervals_ms.len() < 10,
            "intervals outside the prefill windows must be excluded"
        );
        assert!(cell.p50_ms.is_some() && cell.p95_ms.is_some() && cell.p99_ms.is_some());
        let expected_max = cell.intervals_ms.iter().cloned().fold(f64::MIN, f64::max);
        assert_eq!(cell.max_ms, Some(expected_max));
    }

    #[test]
    fn itl_decode_lanes_are_stopped_after_the_final_prefill_window() {
        // A lane that never stops on its own. If cancellation failed to
        // reach it, it runs to this ceiling instead of blocking forever —
        // which is why the assertions below check *both* ends: past the
        // fixture's token cap, and short of the ceiling.
        const NEVER_CANCELLED: u32 = 100_000;

        struct UntilCancelled;

        impl Endpoint for UntilCancelled {
            fn complete(&self, req: &Request) -> Result<crate::client::Outcome, String> {
                if req.id.starts_with("itl-decode") {
                    return Err("decode lane must use the cancellable streaming path".into());
                }
                Ok(crate::client::Outcome {
                    ttft_ms: 1.0,
                    total_ms: 1.0,
                    n_tokens: 1,
                    output: String::new(),
                    prompt_tokens: Some(req.prompt.split_whitespace().count() as u32),
                    cached_prompt_tokens: None,
                    token_times_ms: vec![1.0],
                })
            }

            fn complete_observed_while(
                &self,
                _req: &Request,
                observer: &mut dyn FnMut(f64) -> bool,
                _suppress_eos: &[u32],
            ) -> Result<crate::client::Outcome, String> {
                let mut token_times_ms = Vec::new();
                for tick in 1..=NEVER_CANCELLED {
                    let time_ms = tick as f64;
                    token_times_ms.push(time_ms);
                    if !observer(time_ms) {
                        break;
                    }
                    std::thread::yield_now();
                }
                Ok(crate::client::Outcome {
                    ttft_ms: 1.0,
                    total_ms: *token_times_ms.last().unwrap(),
                    n_tokens: token_times_ms.len() as u32,
                    output: String::new(),
                    prompt_tokens: None,
                    cached_prompt_tokens: None,
                    token_times_ms,
                })
            }
        }

        let cfg = ItlConfig {
            prefill_prompt_tokens: 40,
            prefill_max_tokens: 1,
            prefill_count: 3,
            decode_prompt_tokens: 24,
            decode_max_tokens: 2,
            decode_lanes: 2,
        };
        let cell = measure_itl(&UntilCancelled, &MockTemplate::new(), &cfg);
        assert!(cell.error.is_none(), "{:?}", cell.error);
        assert!(cell.lanes.iter().all(|lane| lane.error.is_none()), "{:?}", cell.lanes);
        assert!(
            cell.lanes.iter().all(|lane| lane.n_tokens > cfg.decode_max_tokens),
            "the measured window, not the fixture's old token cap, must end each lane"
        );
        assert!(
            cell.lanes.iter().all(|lane| lane.n_tokens < NEVER_CANCELLED),
            "each lane must be *stopped* at the window boundary, not merely run out"
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
        let ep = MockEndpoint::new();
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
