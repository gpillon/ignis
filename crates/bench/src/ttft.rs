//! The G2 measurement instrument (P2-05, GitHub #87): time to first token
//! at an **exact prompt length**, on **cold prefixes**, against any
//! OpenAI-compatible endpoint.
//!
//! Spec: `.scratch/runtime/specs/02-real-prefill.md`; ADR 0015 (G2 is
//! judged live/live on cold-prefix samples). The verdict half lives in
//! [`crate::g2`].
//!
//! ## Why "cold prefix" is the whole design
//!
//! The quantity G2 gates is **prefill**. With the reference in its
//! production profile, repeating one prompt across samples turns samples
//! 2..N into prefix-cache hits, and what the stopwatch then measures is a
//! cache lookup. So every sample here — the warmup included — gets its
//! **own** prompt, differing from the *first content token*, and every
//! sample **verifies** its coldness against the engine's own reported
//! computed-prefill-token count ([`crate::client::Outcome::computed_prefill_tokens`]).
//! A sample whose computed prefill is not the prompt's own length is
//! **void** and fails its cell. Coldness is evidence here, never an
//! assumption.
//!
//! ## The instrument is one instrument
//!
//! ignis and the reference are driven through the same
//! [`crate::client::Endpoint`], with the same request shape (streaming,
//! greedy, thinking disabled, a small output budget) and the same prompt
//! generator. A cell's statistic is the **median of five samples after one
//! warmup**, so a single scheduling hiccup does not decide a gate.
//!
//! ## Prompts of an exact length
//!
//! A cell claims a prompt length in tokens, so it has to hit that length
//! exactly — *post-template*, since the template's own header and footer
//! are part of what the engine prefills. [`PromptTemplate`] is that seam:
//! render one user message through the artifact's chat template (thinking
//! off, the way the samples are sent) and encode it. The real
//! implementation is [`ignis_artifact::FrontendSet`]; the tests use a
//! trivial mock, so the generator, the median, the void rule and every
//! refusal are CPU-testable with no artifact and no GPU (ADR 0006).

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::client::{Endpoint, Request};
use crate::time::{unix_now, utc_timestamp};
use crate::trace::RequestClass;

/// Samples per cell, after one warmup (spec 02: "the median of five
/// samples after one warmup").
pub const DEFAULT_SAMPLES: usize = 5;

/// The generation budget a TTFT sample asks for. TTFT is the arrival of
/// the first content delta, so the engine only has to get *started*;
/// anything beyond a couple of tokens is time spent not measuring.
pub const DEFAULT_MAX_TOKENS: u32 = 8;

// ── the template seam ────────────────────────────────────────────────────

/// Renders one user message the way a sample is sent and encodes it: the
/// exact token sequence the engine will prefill.
///
/// A seam (like [`crate::oracle::Tokenize`]) so prompt generation is
/// unit-testable without a `.ninfer` artifact. The production
/// implementation is on [`ignis_artifact::FrontendSet`] below: the
/// artifact's own chat template with thinking disabled, then the
/// artifact's own tokenizer — which is what makes a cell's claimed length
/// the length the engine actually sees.
pub trait PromptTemplate: Send + Sync {
    /// The post-template token ids for a single user message carrying
    /// `content`. A human-readable error on failure (a template that
    /// raises is a failed cell, not a panic).
    fn encode_user_message(&self, content: &str) -> Result<Vec<u32>, String>;
}

impl PromptTemplate for ignis_artifact::FrontendSet {
    fn encode_user_message(&self, content: &str) -> Result<Vec<u32>, String> {
        let messages = [ignis_artifact::ChatMessage::text(
            ignis_artifact::Role::User,
            content.to_string(),
        )];
        // `enable_thinking: false` matches what the samples send, so the
        // count a cell claims is the count the engine prefills.
        let prompt = self
            .chat_template()
            .render_with_thinking(&messages, false, None)
            .map_err(|e| format!("render the chat template: {e}"))?;
        self.tokenizer()
            .encode(&prompt)
            .map_err(|e| format!("tokenize the rendered prompt: {e}"))
    }
}

// ── deterministic prompts of an exact length ─────────────────────────────

/// The generator's word pool. Ordinary lowercase words: nothing here
/// should tokenize into anything exotic, and the pool is fixed so a seed
/// reproduces a prompt exactly.
const VOCAB: &[&str] = &[
    "system", "buffer", "kernel", "matrix", "window", "thread", "packet", "vector", "record",
    "handle", "stream", "socket", "module", "branch", "cursor", "device", "logging", "session",
    "target", "column", "bucket", "anchor", "digest", "marker", "region", "signal", "policy",
    "latency", "context", "message", "counter", "adapter",
];

/// The single-token filler the generator lands the last few tokens with.
/// A short common word preceded by a space is one BPE token in every
/// tokenizer this project meets; the generator re-measures after every
/// append regardless, so a tokenizer where it is not simply fails the cell
/// with a clear message instead of silently missing the target.
const FILLER: &str = "a";

/// The first character of sample `index`'s prompt content. Distinct first
/// characters are what make the samples differ **from the first content
/// token**: whatever token the tokenizer forms there, two samples' tokens
/// cannot be equal if their first characters differ.
fn nonce(index: usize) -> String {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    let letter = ALPHABET[index % ALPHABET.len()] as char;
    format!("{letter}{index:03}")
}

/// A tiny deterministic PRNG (an LCG): the same seed yields the same
/// prompt, so a recorded cell can be reproduced exactly.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// Generate a prompt whose **post-template** token count is exactly
/// `target_tokens`, beginning with `nonce`.
///
/// Grow with pseudorandom words until the target is reached or passed, drop
/// back under it, then land on it with single-token filler — re-measuring
/// through `template` at every step, so the count is what the tokenizer
/// says rather than what an estimate hoped.
pub fn generate_prompt(
    template: &dyn PromptTemplate,
    target_tokens: usize,
    nonce: &str,
    seed: u64,
) -> Result<String, String> {
    let mut rng = Lcg(seed);
    let mut words: Vec<String> = vec![nonce.to_string()];
    let mut count = template.encode_user_message(&words.join(" "))?.len();
    if count > target_tokens {
        return Err(format!(
            "the chat template's own overhead is {count} tokens: a {target_tokens}-token cell is \
             not reachable"
        ));
    }
    // Grow. Each word is at least one token, so this terminates.
    while count < target_tokens {
        words.push(VOCAB[(rng.next() as usize) % VOCAB.len()].to_string());
        count = template.encode_user_message(&words.join(" "))?.len();
    }
    // Drop back under (the last word may have overshot by several tokens).
    while count > target_tokens && words.len() > 1 {
        words.pop();
        count = template.encode_user_message(&words.join(" "))?.len();
    }
    // Land exactly, one filler token at a time.
    while count < target_tokens {
        words.push(FILLER.to_string());
        let grown = template.encode_user_message(&words.join(" "))?.len();
        if grown > target_tokens {
            return Err(format!(
                "cannot land on {target_tokens} tokens: appending `{FILLER}` moved the count from \
                 {count} to {grown}"
            ));
        }
        if grown == count {
            return Err(format!(
                "cannot land on {target_tokens} tokens: appending `{FILLER}` did not change the \
                 count ({count})"
            ));
        }
        count = grown;
    }
    Ok(words.join(" "))
}

/// A cell's prompts: one per sample, plus the warmup's, each exactly
/// `target_tokens` post-template tokens and each diverging from the others
/// at the first content token.
#[derive(Debug, Clone)]
pub struct PromptSet {
    /// `[0]` is the warmup's prompt; `[1..]` are the samples', in order.
    pub prompts: Vec<String>,
    /// The post-template token ids of each prompt, same order.
    pub tokens: Vec<Vec<u32>>,
}

/// Generate `count` prompts of exactly `target_tokens` post-template
/// tokens (the warmup's first), and **prove** they are distinct where it
/// matters before any of them is sent.
///
/// The proof has three parts, and a cell that fails any of them is a
/// failed cell, not a measured one:
///
/// 1. every prompt hits the claimed length exactly;
/// 2. every pair diverges, and every pair diverges at the *same* index —
///    so all the prompts share one prefix and then part ways at once,
///    rather than some pair sharing content the others do not;
/// 3. that shared prefix is shorter than the template's own overhead, so
///    it cannot contain a content token. A shared header is unavoidable
///    (it *is* the template); a shared content token is exactly the
///    partial prefix hit this instrument exists to rule out.
pub fn generate_cell_prompts(
    template: &dyn PromptTemplate,
    target_tokens: usize,
    count: usize,
) -> Result<PromptSet, String> {
    if count == 0 {
        return Err("a cell needs at least one prompt".to_string());
    }
    let mut prompts = Vec::with_capacity(count);
    let mut tokens = Vec::with_capacity(count);
    for index in 0..count {
        let prompt = generate_prompt(
            template,
            target_tokens,
            &nonce(index),
            0x9E3779B97F4A7C15 ^ (index as u64).wrapping_mul(0x100000001B3),
        )?;
        let ids = template.encode_user_message(&prompt)?;
        if ids.len() != target_tokens {
            return Err(format!(
                "prompt {index} came out {} tokens, not the {target_tokens} the cell claims",
                ids.len()
            ));
        }
        prompts.push(prompt);
        tokens.push(ids);
    }
    // The template's own overhead, as an upper bound on how much two
    // samples may legitimately share.
    let overhead = template.encode_user_message("")?.len();
    let mut divergence: Option<usize> = None;
    for i in 0..count {
        for j in (i + 1)..count {
            let d = common_prefix_len(&tokens[i], &tokens[j]);
            if d == tokens[i].len() {
                return Err(format!("prompts {i} and {j} are identical"));
            }
            match divergence {
                None => divergence = Some(d),
                Some(expected) if expected != d => {
                    return Err(format!(
                        "prompts {i} and {j} share {d} leading tokens, but another pair shares \
                         {expected}: the samples do not all diverge at the same token"
                    ));
                }
                Some(_) => {}
            }
        }
    }
    if let Some(d) = divergence {
        if d > overhead {
            return Err(format!(
                "the samples share {d} leading tokens but the chat template's overhead is only \
                 {overhead}: they share content, so a partial prefix hit is possible"
            ));
        }
    }
    Ok(PromptSet { prompts, tokens })
}

/// The number of leading elements `a` and `b` have in common.
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

// ── the record ───────────────────────────────────────────────────────────

/// One measured sample: its time to first token, and the evidence that it
/// was a cold prefix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    /// The sample's index within its cell (0-based; the warmup is not a
    /// sample).
    pub index: usize,
    /// Time to first token, in ms: the arrival of the first content delta.
    pub ttft_ms: f64,
    /// The engine's own computed-prefill-token count for this sample
    /// (reported prompt tokens minus whatever it served from a cache), or
    /// `None` when the engine reported no usage at all.
    pub computed_prefill_tokens: Option<u32>,
    /// True when this sample cannot be used: its prefix was not provably
    /// cold, or the request failed. A void sample fails its cell, and the
    /// gate check refuses a verdict over it.
    pub void: bool,
    /// Why the sample is void (absent when it is not).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub void_reason: Option<String>,
}

/// One measured cell: a prompt length, its samples, and their median.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cell {
    /// The cell's prompt length, in post-template tokens.
    pub prompt_tokens: u32,
    /// The warmup's TTFT, in ms. Recorded for diagnostics and never part
    /// of the statistic — its own prompt is distinct too, so the warmup
    /// cannot populate a prefix the samples then reuse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup_ttft_ms: Option<f64>,
    /// Every sample, in order.
    pub samples: Vec<Sample>,
    /// The cell's statistic: the median of `samples`' TTFTs, or `None`
    /// when the cell has no samples at all.
    pub median_ttft_ms: Option<f64>,
    /// Set when the cell failed before it could be measured (prompt
    /// generation, or the warmup request).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Cell {
    /// True when every sample in this cell was provably cold — the
    /// property a gate verdict may be computed over.
    pub fn all_cold(&self) -> bool {
        self.error.is_none() && !self.samples.is_empty() && self.samples.iter().all(|s| !s.void)
    }

    /// The samples that are not usable, with their reasons.
    pub fn void_samples(&self) -> Vec<&Sample> {
        self.samples.iter().filter(|s| s.void).collect()
    }
}

/// A TTFT record: what one engine measured, on which cells, in which
/// session.
///
/// The session identifier is what makes a **live/live** verdict possible
/// (ADR 0015): two records carrying the same session were produced in the
/// same measurement session, on the same machine, by the same operator
/// run. The gate check refuses to compare records that do not.
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
    /// The profile the engine was running in (the reference's production
    /// profile, ignis's configured shape) — recorded because the gate
    /// compares engines *as run*, not as they might be configured.
    pub profile: String,
    /// When the record was made (UTC, RFC 3339 seconds).
    pub date: String,
    /// The generation budget every sample asked for.
    pub max_tokens: u32,
    /// The measured cells, in the order they were measured.
    pub cells: Vec<Cell>,
}

impl Record {
    /// This record's cell at `prompt_tokens`, if it has one.
    pub fn cell(&self, prompt_tokens: u32) -> Option<&Cell> {
        self.cells.iter().find(|c| c.prompt_tokens == prompt_tokens)
    }

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

    /// A one-line-per-cell rendering for the terminal.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "ttft record  session={}  label={}  engine={}  profile={}\n",
            self.session, self.label, self.engine, self.profile
        ));
        out.push_str(&format!(
            "  endpoint={}  artifact={}  date={}\n",
            self.endpoint, self.artifact, self.date
        ));
        for cell in &self.cells {
            match (&cell.error, cell.median_ttft_ms) {
                (Some(err), _) => {
                    out.push_str(&format!("  {:>7} tokens  FAILED: {err}\n", cell.prompt_tokens));
                }
                (None, Some(median)) => {
                    let voids = cell.void_samples().len();
                    out.push_str(&format!(
                        "  {:>7} tokens  median {median:>9.1} ms  over {} samples{}\n",
                        cell.prompt_tokens,
                        cell.samples.len(),
                        if voids == 0 {
                            "  (all cold)".to_string()
                        } else {
                            format!("  ({voids} VOID)")
                        },
                    ));
                    for sample in cell.void_samples() {
                        out.push_str(&format!(
                            "            sample {}: void — {}\n",
                            sample.index,
                            sample.void_reason.as_deref().unwrap_or("(no reason)"),
                        ));
                    }
                }
                (None, None) => {
                    out.push_str(&format!("  {:>7} tokens  no samples\n", cell.prompt_tokens));
                }
            }
        }
        out
    }
}

/// The median of `values` (the mean of the two middle values when the
/// count is even). `None` for an empty slice.
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    Some(if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    })
}

// ── measuring ────────────────────────────────────────────────────────────

/// One cell to measure: a prompt length and how many samples to take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellSpec {
    /// The prompt length in post-template tokens.
    pub prompt_tokens: u32,
    /// Samples after the warmup.
    pub samples: usize,
}

/// What a `ttft` run measures and how the record identifies it.
#[derive(Debug, Clone)]
pub struct TtftConfig {
    /// The cells, in the order they are measured.
    pub cells: Vec<CellSpec>,
    /// The generation budget every sample asks for.
    pub max_tokens: u32,
    /// Which engine this run is measuring ("ignis", "reference", ...).
    pub label: String,
    /// The profile the engine is running in.
    pub profile: String,
    /// The artifact the engine is serving, as the operator names it.
    pub artifact: String,
    /// The measurement session both engines' records must share.
    pub session: String,
}

/// The request one sample sends: streaming (TTFT is the first content
/// delta), greedy and fixed-seed (`HttpEndpoint` pins `temperature: 0` /
/// `seed: 0`), thinking disabled (a thinking preamble would move the first
/// *content* delta behind a reasoning stream), a small output budget, and
/// the trailing usage chunk (the cold-prefix evidence).
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

/// Measure one cell: generate its prompts, send the warmup, then take the
/// samples and reduce them to a median.
pub fn measure_cell(
    ep: &dyn Endpoint,
    template: &dyn PromptTemplate,
    spec: &CellSpec,
    max_tokens: u32,
) -> Cell {
    let failed = |error: String| Cell {
        prompt_tokens: spec.prompt_tokens,
        warmup_ttft_ms: None,
        samples: Vec::new(),
        median_ttft_ms: None,
        error: Some(error),
    };
    if spec.samples == 0 {
        return failed("a cell needs at least one sample".to_string());
    }
    // One prompt per sample plus the warmup's — all distinct from the first
    // content token, proven before anything is sent.
    let set = match generate_cell_prompts(template, spec.prompt_tokens as usize, spec.samples + 1) {
        Ok(set) => set,
        Err(err) => return failed(err),
    };

    // The warmup: its own distinct prompt, so it cannot leave a prefix the
    // measured samples reuse. Its timing is recorded but never counted.
    let warmup = ep.complete(&sample_request(
        format!("ttft-{}-warmup", spec.prompt_tokens),
        set.prompts[0].clone(),
        max_tokens,
    ));
    let warmup_ttft_ms = match warmup {
        Ok(outcome) => Some(outcome.ttft_ms),
        Err(err) => return failed(format!("the warmup request failed: {err}")),
    };

    let mut samples = Vec::with_capacity(spec.samples);
    for index in 0..spec.samples {
        let request = sample_request(
            format!("ttft-{}-{index}", spec.prompt_tokens),
            set.prompts[index + 1].clone(),
            max_tokens,
        );
        samples.push(match ep.complete(&request) {
            Ok(outcome) => {
                let computed = outcome.computed_prefill_tokens();
                let void_reason = coldness_failure(computed, spec.prompt_tokens);
                Sample {
                    index,
                    ttft_ms: outcome.ttft_ms,
                    computed_prefill_tokens: computed,
                    void: void_reason.is_some(),
                    void_reason,
                }
            }
            Err(err) => Sample {
                index,
                ttft_ms: 0.0,
                computed_prefill_tokens: None,
                void: true,
                void_reason: Some(format!("the request failed: {err}")),
            },
        });
    }
    let ttfts: Vec<f64> = samples.iter().map(|s| s.ttft_ms).collect();
    Cell {
        prompt_tokens: spec.prompt_tokens,
        warmup_ttft_ms,
        median_ttft_ms: median(&ttfts),
        samples,
        error: None,
    }
}

/// Why a sample's prefix cannot be called cold, or `None` when the engine
/// computed exactly the prompt it was given.
///
/// Short means part of the prompt came from a cache — the contamination
/// this instrument exists to catch. Long means our post-template count and
/// the engine's disagree, which makes the cell's claimed length a fiction;
/// either way the sample is not evidence, so it is void (ADR 0015: the
/// tool enforces live/live, not the operator's discipline).
fn coldness_failure(computed: Option<u32>, prompt_tokens: u32) -> Option<String> {
    match computed {
        None => Some(
            "the engine reported no usage, so its computed prefill cannot be read back".to_string(),
        ),
        Some(computed) if computed < prompt_tokens => Some(format!(
            "computed prefill {computed} is short of the {prompt_tokens}-token prompt: part of it \
             was served from a cache"
        )),
        Some(computed) if computed > prompt_tokens => Some(format!(
            "the engine counted {computed} prompt tokens where the cell generated {prompt_tokens}: \
             the cell is not measuring the length it claims"
        )),
        Some(_) => None,
    }
}

/// Measure every cell in `cfg` against one endpoint and return the record.
pub fn measure(
    ep: &dyn Endpoint,
    template: &dyn PromptTemplate,
    engine: String,
    endpoint: String,
    cfg: &TtftConfig,
) -> Record {
    let cells = cfg
        .cells
        .iter()
        .map(|spec| measure_cell(ep, template, spec, cfg.max_tokens))
        .collect();
    Record {
        session: cfg.session.clone(),
        label: cfg.label.clone(),
        endpoint,
        engine,
        artifact: cfg.artifact.clone(),
        profile: cfg.profile.clone(),
        date: utc_timestamp(unix_now()),
        max_tokens: cfg.max_tokens,
        cells,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock template: a fixed header and footer around one token per
    /// whitespace-separated word, each word hashed to an id. Enough to
    /// exercise exact-length generation, the distinctness proof and the
    /// void rule without an artifact (ADR 0006).
    struct MockTemplate {
        header: usize,
        footer: usize,
    }

    impl MockTemplate {
        fn new() -> Self {
            Self { header: 6, footer: 3 }
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
    }

    #[test]
    fn a_generated_prompt_hits_the_requested_post_template_length_exactly() {
        let template = MockTemplate::new();
        for target in [16usize, 17, 40, 128] {
            let prompt = generate_prompt(&template, target, "a000", 1).expect("generated");
            assert_eq!(
                template.encode_user_message(&prompt).unwrap().len(),
                target,
                "target {target}"
            );
        }
    }

    #[test]
    fn generation_is_deterministic() {
        let template = MockTemplate::new();
        let a = generate_prompt(&template, 64, "a000", 42).expect("generated");
        let b = generate_prompt(&template, 64, "a000", 42).expect("generated");
        assert_eq!(a, b, "the same seed must reproduce the same prompt");
    }

    #[test]
    fn a_target_below_the_template_overhead_is_refused() {
        // The header + footer alone are 9 tokens; a 4-token cell cannot be
        // measured, and saying so beats generating a prompt of the wrong
        // length.
        let err = generate_prompt(&MockTemplate::new(), 4, "a000", 1).expect_err("must refuse");
        assert!(err.contains("overhead"), "{err}");
    }

    #[test]
    fn every_sample_gets_its_own_prompt_diverging_at_the_first_content_token() {
        let template = MockTemplate::new();
        let set = generate_cell_prompts(&template, 64, 6).expect("prompts");
        assert_eq!(set.prompts.len(), 6, "one warmup + five samples");
        for ids in &set.tokens {
            assert_eq!(ids.len(), 64);
        }
        // Every pair diverges, and at the header boundary — not later.
        for i in 0..set.tokens.len() {
            for j in (i + 1)..set.tokens.len() {
                assert_eq!(
                    common_prefix_len(&set.tokens[i], &set.tokens[j]),
                    template.header,
                    "prompts {i} and {j} must share the template header and nothing more"
                );
            }
        }
    }

    #[test]
    fn prompts_that_share_a_content_token_are_refused() {
        /// A template whose "tokens" ignore the content's leading word, so
        /// every prompt of a given length shares its first content token —
        /// exactly the partial-prefix-hit hazard the proof exists to catch.
        struct SharedHeadTemplate;
        impl PromptTemplate for SharedHeadTemplate {
            fn encode_user_message(&self, content: &str) -> Result<Vec<u32>, String> {
                let mut ids = vec![1u32, 2, 3];
                // The first content word always tokenizes to the same id.
                for (i, word) in content.split_whitespace().enumerate() {
                    ids.push(if i == 0 {
                        99
                    } else {
                        word.bytes().fold(7u32, |a, b| a.wrapping_mul(131).wrapping_add(b as u32))
                    });
                }
                Ok(ids)
            }
        }
        let err = generate_cell_prompts(&SharedHeadTemplate, 40, 6).expect_err("must refuse");
        assert!(err.contains("share content"), "{err}");
    }

    #[test]
    fn the_median_is_the_middle_sample() {
        assert_eq!(median(&[5.0, 1.0, 3.0, 2.0, 4.0]), Some(3.0));
        // Even counts average the two middles.
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
        assert_eq!(median(&[]), None);
    }

    #[test]
    fn a_short_computed_prefill_is_void() {
        let reason = coldness_failure(Some(4_096), 8_192).expect("void");
        assert!(reason.contains("cache"), "{reason}");
    }

    #[test]
    fn a_missing_usage_report_is_void() {
        let reason = coldness_failure(None, 8_192).expect("void");
        assert!(reason.contains("no usage"), "{reason}");
    }

    #[test]
    fn a_computed_prefill_over_the_prompt_length_is_void_too() {
        let reason = coldness_failure(Some(8_200), 8_192).expect("void");
        assert!(reason.contains("not measuring the length it claims"), "{reason}");
    }

    #[test]
    fn an_exactly_computed_prefill_is_cold() {
        assert_eq!(coldness_failure(Some(8_192), 8_192), None);
    }
}
