//! The G4 measurement instrument (P4-01, GitHub #117): a trace-replay run
//! record that carries what a live/live verdict needs to trust it — the
//! measurement session and the SHA-256 of the load trace it replayed — plus
//! the needle-retrieval correctness floor. The verdict half lives in
//! [`crate::g4_gate`].
//!
//! Spec: `.scratch/runtime/specs/04-reference-feature-floor.md` ("Gate G4");
//! ADR 0015 (live/live), ADR 0021 (launch pooling). Before this ticket,
//! `crates/bench/src/gate.rs` / `report.rs` compared two in-memory [`Run`]s
//! with no session or trace identity at all — exactly the "record one
//! reference run, commit it, compare later" shape ADR 0015 and ADR 0021 now
//! refuse for every other gate. [`Record`] is the trace-replay run's
//! counterpart to [`crate::g3::Record`] / [`crate::ttft::Record`]: same
//! identity fields, so [`crate::g4_gate::check`] can enforce the same
//! live/live discipline.
//!
//! ## The needle-retrieval cell
//!
//! A planted fact ("verification code") is buried in the middle of a
//! filler haystack landed at an exact post-template token count (64K /
//! 128K by default — [`NEEDLE_CONTEXT_64K`], [`NEEDLE_CONTEXT_128K`]), and
//! the model is asked to recall it. This is a **correctness floor**, not a
//! ratio (spec 04's Gate G4 table): the fact is retrieved or it is not,
//! never averaged or compared against the reference's own number.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::client::{replay, Endpoint, Request, ReplayConfig};
use crate::metrics::Run;
use crate::time::{unix_now, utc_timestamp};
use crate::trace::{RequestClass, Trace};
use crate::ttft::{load_corpus, PromptTemplate};

// ── the needle-retrieval cell ────────────────────────────────────────────

/// The two context lengths spec 04's Gate G4 table requires.
pub const NEEDLE_CONTEXT_64K: u32 = 65_536;
pub const NEEDLE_CONTEXT_128K: u32 = 131_072;

/// The needle-retrieval answer's token budget: a handful of digits, not a
/// generation cell.
pub const NEEDLE_MAX_TOKENS: u32 = 16;

/// One needle-retrieval sample: whether the planted fact came back.
///
/// A correctness floor (spec 04): [`NeedleResult::passed`] is a bare
/// boolean, never a ratio — a pass/fail cell folds into nothing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeedleResult {
    /// The haystack's post-template token count (64K / 128K).
    pub context_tokens: u32,
    /// The planted fact this sample asked for (deterministic per
    /// `context_tokens`, so a re-run asks the identical question).
    pub secret: String,
    /// Whether the model's answer contained the planted fact.
    pub retrieved: bool,
    /// Set when the haystack could not be built or the request failed —
    /// distinct from a haystack that built fine but whose answer missed the
    /// fact (`retrieved: false`, `error: None`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl NeedleResult {
    /// The floor: no error, and the fact came back.
    pub fn passed(&self) -> bool {
        self.error.is_none() && self.retrieved
    }
}

/// The filler vocabulary the haystack is grown from. Distinct from
/// [`crate::ttft`]'s pool (that module's is private) — ordinary words that
/// tokenize predictably in every tokenizer this project meets.
const HAYSTACK_VOCAB: &[&str] = &[
    "forest", "harbor", "meadow", "canyon", "orchard", "quarry", "tunnel", "glacier", "prairie",
    "valley", "island", "desert", "plateau", "wetland", "coastline", "riverbed", "hillside",
    "woodland", "marshland", "grassland",
];

/// The single-token filler the haystack lands on (mirrors
/// [`crate::ttft::generate_prompt`]'s landing discipline).
const HAYSTACK_FILLER: &str = "a";

/// A tiny deterministic PRNG (an LCG, the same shape as `ttft.rs`'s
/// private one): the same seed grows the same haystack.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// A deterministic six-digit code for `context_tokens`, so a re-run at the
/// same cell asks the identical question (reproducible, per
/// [`crate::ttft::generate_prompt`]'s own rationale for a seeded PRNG).
fn needle_secret(context_tokens: u32) -> String {
    let n = 100_000 + ((context_tokens as u64).wrapping_mul(2_654_435_761) % 900_000);
    format!("{n:06}")
}

fn needle_sentence(secret: &str) -> String {
    format!("The verification code you must remember for later is {secret}. Do not forget it.")
}

const NEEDLE_QUESTION: &str =
    "What is the six-digit verification code stated earlier in this text? Respond with only the digits, nothing else.";

/// Build a haystack landed at exactly `context_tokens` post-template
/// tokens, with the needle sentence spliced into the middle of the filler
/// and the question appended at the end.
///
/// Grows one word at a time, re-measuring through `template` at every step
/// (the same landing discipline as [`crate::ttft::generate_prompt`]): a
/// tokenizer that cannot land exactly fails the cell with a clear message
/// rather than silently measuring a different length than it claims.
fn build_haystack(
    template: &dyn PromptTemplate,
    context_tokens: usize,
    secret: &str,
) -> Result<String, String> {
    let needle = needle_sentence(secret);
    let render = |words: &[String]| -> String {
        let insert_at = words.len() / 2;
        let mut body = String::new();
        for (i, w) in words.iter().enumerate() {
            if i == insert_at {
                body.push_str(&needle);
                body.push(' ');
            }
            body.push_str(w);
            body.push(' ');
        }
        if words.is_empty() {
            body.push_str(&needle);
            body.push(' ');
        }
        format!("{body}{NEEDLE_QUESTION}")
    };

    let mut rng = Lcg(0xC0FFEE_u64 ^ context_tokens as u64);
    let mut words: Vec<String> = Vec::new();
    let mut count = template.encode_user_message(&render(&words))?.len();
    if count > context_tokens {
        return Err(format!(
            "the needle sentence + question alone is {count} tokens: a {context_tokens}-token \
             needle cell is not reachable"
        ));
    }
    while count < context_tokens {
        words.push(HAYSTACK_VOCAB[(rng.next() as usize) % HAYSTACK_VOCAB.len()].to_string());
        count = template.encode_user_message(&render(&words))?.len();
    }
    while count > context_tokens && !words.is_empty() {
        words.pop();
        count = template.encode_user_message(&render(&words))?.len();
    }
    while count < context_tokens {
        words.push(HAYSTACK_FILLER.to_string());
        let grown = template.encode_user_message(&render(&words))?.len();
        if grown <= count {
            return Err(format!(
                "cannot land the needle haystack on {context_tokens} tokens: appending filler \
                 did not grow the count past {count}"
            ));
        }
        count = grown;
    }
    Ok(render(&words))
}

/// Splice `needle` into the middle of `body` (on the nearest word boundary
/// after the midpoint) and append the question — the corpus path's render,
/// sharing [`build_haystack`]'s shape (needle buried mid-text, question
/// last) without its word-list representation.
fn render_haystack_body(body: &str, needle: &str) -> String {
    if body.is_empty() {
        return format!("{needle} {NEEDLE_QUESTION}");
    }
    // The corpus decodes to real text and may carry multi-byte UTF-8 (a
    // real tokenizer's vocabulary is not ASCII-only): `body.len() / 2` is a
    // byte offset that can land inside a character, so walk forward to the
    // next char boundary before slicing on it.
    let mut mid = body.len() / 2;
    while mid < body.len() && !body.is_char_boundary(mid) {
        mid += 1;
    }
    let split_at = body[mid..].find(' ').map(|o| mid + o).unwrap_or(body.len());
    let (head, tail) = body.split_at(split_at);
    format!("{head} {needle}{tail} {NEEDLE_QUESTION}")
}

/// Build a haystack from a pre-tokenized corpus (the `--corpus` path):
/// bounded generation instead of [`build_haystack`]'s word-by-word growth,
/// which the 64K/128K needle cell pays for badly against a real tokenizer
/// (the same tradeoff [`crate::ttft::generate_cell_prompts_from_corpus`]
/// makes for G2/G3's own production-scale fixtures — a binary search over
/// window *length*, since the re-encoded count is monotone in it, instead
/// of one re-encode per appended word).
///
/// A real tokenizer's decode/encode round-trip can drift a few tokens
/// (`ttft::land_corpus_window`'s doc comment), so after the search narrows
/// down to a window length a small neighbourhood is scanned before this
/// gives up and refuses rather than measuring a length it does not claim.
fn build_haystack_from_corpus(
    template: &dyn PromptTemplate,
    corpus: &[u32],
    context_tokens: usize,
    secret: &str,
) -> Result<String, String> {
    if corpus.is_empty() {
        return Err("the corpus is empty: no window to cut".to_string());
    }
    let needle = needle_sentence(secret);
    let overhead = template.encode_user_message(&render_haystack_body("", &needle))?.len();
    if overhead >= context_tokens {
        return Err(format!(
            "the needle sentence + question alone is {overhead} tokens: a {context_tokens}-token \
             needle cell is not reachable"
        ));
    }
    // The corpus repeats (ttft.rs's doc comment: "a tiled rotation of a
    // curated bank"), so a window longer than the bank itself is still a
    // meaningful, if repetitive, haystack -- wrap rather than cap at
    // `corpus.len()`.
    let cut = |n: usize| -> Result<(String, usize), String> {
        let n = n.max(1);
        let ids: Vec<u32> = (0..n).map(|k| corpus[k % corpus.len()]).collect();
        let body = template.decode(&ids)?;
        let text = render_haystack_body(&body, &needle);
        let count = template.encode_user_message(&text)?.len();
        Ok((text, count))
    };

    let mut lo = 1usize;
    let mut hi = context_tokens.saturating_mul(4).max(corpus.len()).max(2);
    let mut closest: Option<usize> = None;
    for _ in 0..64 {
        if lo >= hi {
            break;
        }
        let mid = lo + (hi - lo) / 2;
        let (text, count) = cut(mid)?;
        if count == context_tokens {
            return Ok(text);
        }
        closest = Some(mid);
        if count < context_tokens {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    // The round-trip drift window: a handful of ids either side of where
    // the search settled.
    const SCAN_RADIUS: usize = 64;
    let center = closest.unwrap_or(lo);
    for delta in 0..SCAN_RADIUS {
        for n in [center.saturating_sub(delta), center + delta] {
            let (text, count) = cut(n)?;
            if count == context_tokens {
                return Ok(text);
            }
        }
    }
    Err(format!(
        "cannot land the needle haystack on {context_tokens} tokens from this corpus after a \
         bounded search"
    ))
}

/// The needle-retrieval request for a haystack already built at
/// `context_tokens`, and the [`NeedleResult`] its outcome (or a build
/// failure) produces — shared by [`measure_needle`] and
/// [`measure_needle_from_corpus`].
fn send_needle_request(
    ep: &dyn Endpoint,
    context_tokens: u32,
    secret: String,
    haystack: Result<String, String>,
) -> NeedleResult {
    let prompt = match haystack {
        Ok(p) => p,
        Err(err) => {
            return NeedleResult {
                context_tokens,
                secret,
                retrieved: false,
                error: Some(err),
            }
        }
    };
    // Streaming, like every other measurement request in this crate
    // (g2/g3/replay): a non-streaming request leaves the connection
    // carrying zero bytes for the whole prefill — tens of seconds at
    // 64K/128K — which a live smoke test found closes the connection on at
    // least one transport before the response ever arrives, even though
    // the engine finishes the request normally on its own side.
    let req = Request {
        id: format!("needle-{context_tokens}"),
        class: RequestClass::Main,
        prompt,
        max_tokens: NEEDLE_MAX_TOKENS,
        stream: true,
        include_usage: false,
        enable_thinking: Some(false),
    };
    match ep.complete(&req) {
        Ok(outcome) => NeedleResult {
            context_tokens,
            retrieved: outcome.output.contains(&secret),
            secret,
            error: None,
        },
        Err(err) => NeedleResult {
            context_tokens,
            secret,
            retrieved: false,
            error: Some(format!("the request failed: {err}")),
        },
    }
}

/// Measure one needle-retrieval cell: build the haystack (the filler
/// word-growth path), ask the question, and check whether the planted
/// fact came back.
pub fn measure_needle(
    ep: &dyn Endpoint,
    template: &dyn PromptTemplate,
    context_tokens: u32,
) -> NeedleResult {
    let secret = needle_secret(context_tokens);
    let haystack = build_haystack(template, context_tokens as usize, &secret);
    send_needle_request(ep, context_tokens, secret, haystack)
}

/// [`measure_needle`] cut from a pre-tokenized corpus instead of grown word
/// by word — the path a real tokenizer needs at 64K/128K (see
/// [`build_haystack_from_corpus`]'s doc comment).
pub fn measure_needle_from_corpus(
    ep: &dyn Endpoint,
    template: &dyn PromptTemplate,
    corpus: &[u32],
    context_tokens: u32,
) -> NeedleResult {
    let secret = needle_secret(context_tokens);
    let haystack = build_haystack_from_corpus(template, corpus, context_tokens as usize, &secret);
    send_needle_request(ep, context_tokens, secret, haystack)
}

// ── trace hashing ────────────────────────────────────────────────────────

/// The SHA-256 of `bytes`, lowercase hex — the load trace's identity a G4
/// record carries (spec 04: "both run records carry the hash, so a later
/// reader can prove both engines replayed the same load").
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

// ── the record ───────────────────────────────────────────────────────────

/// A G4 record: what one engine measured replaying the load trace, in
/// which session, against which trace — plus the needle-retrieval floor.
///
/// Mirrors [`crate::g3::Record`]'s identity fields (`session`, `label`,
/// `endpoint`, `engine`, `artifact`, `profile`, `date`); `trace_sha256` is
/// new (spec 04's first acceptance criterion): the gate refuses a verdict
/// when two records do not share it, exactly as it already refuses one
/// when they do not share a session (ADR 0015).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// The measurement session every launch of both engines must share.
    pub session: String,
    /// The SHA-256 (lowercase hex) of the load trace this run replayed.
    pub trace_sha256: String,
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
    /// The trace-replay run: one [`crate::metrics::RequestMetrics`] per
    /// trace line (main + subagent classes, ADR 0007's per-class gate).
    pub run: Run,
    /// The needle-retrieval cells (64K / 128K by default).
    pub needles: Vec<NeedleResult>,
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
            "g4 record  session={}  label={}  engine={}  profile={}\n",
            self.session, self.label, self.engine, self.profile
        ));
        out.push_str(&format!(
            "  endpoint={}  artifact={}  date={}  trace={}\n",
            self.endpoint, self.artifact, self.date, self.trace_sha256
        ));
        out.push_str(&format!("  requests: {}\n", self.run.metrics.len()));
        for class in self.run.classes() {
            if let Some(stats) = self.run.stats_for(class) {
                out.push_str(&format!(
                    "    {:<5} {:>4} requests  {:>9.1} tok/s\n",
                    match class {
                        RequestClass::Main => "main",
                        RequestClass::Sub => "sub",
                    },
                    stats.n_requests,
                    stats.tok_s,
                ));
            }
        }
        for needle in &self.needles {
            match &needle.error {
                Some(err) => out.push_str(&format!(
                    "  needle@{} FAILED: {err}\n",
                    needle.context_tokens
                )),
                None => out.push_str(&format!(
                    "  needle@{} {}\n",
                    needle.context_tokens,
                    if needle.retrieved { "RETRIEVED" } else { "MISSED" },
                )),
            }
        }
        out
    }
}

/// What a `g4` run measures and how the record identifies it.
#[derive(Debug, Clone)]
pub struct G4Config {
    /// Which engine this run is measuring ("ignis", "reference", ...).
    pub label: String,
    /// The profile the engine is running in.
    pub profile: String,
    /// The artifact the engine is serving, as the operator names it.
    pub artifact: String,
    /// The measurement session every launch of both engines must share.
    pub session: String,
    /// The context lengths the needle-retrieval cell measures (defaults to
    /// spec 04's 64K / 128K).
    pub needle_context_tokens: Vec<u32>,
    pub replay: ReplayConfig,
    /// A pre-tokenized prompt bank (whitespace-separated ids) to cut the
    /// needle haystacks from, when set (the `--corpus` flag): bounded
    /// generation instead of the filler path's word-by-word growth, needed
    /// at the needle cell's 64K/128K scale (`build_haystack_from_corpus`'s
    /// doc comment — the same tradeoff `g3::G3Config::corpus` makes for its
    /// 32,768-token prefiller fixture). Absent, the filler generator is
    /// used (fine at the small lengths a unit test measures).
    pub corpus: Option<PathBuf>,
}

impl Default for G4Config {
    fn default() -> Self {
        Self {
            label: "ignis".into(),
            profile: "unrecorded".into(),
            artifact: String::new(),
            session: String::new(),
            needle_context_tokens: vec![NEEDLE_CONTEXT_64K, NEEDLE_CONTEXT_128K],
            replay: ReplayConfig::default(),
            corpus: None,
        }
    }
}

/// Measure a G4 record: replay `trace` against `ep`, then the
/// needle-retrieval cells. `trace_bytes` is the trace file's raw content —
/// hashed as-is, so the recorded identity is the bytes that were actually
/// replayed, not a re-serialization of the parsed [`Trace`]. A configured
/// corpus (`cfg.corpus`) is loaded once and every needle cell is cut from
/// it; a corpus that cannot be read fails every needle cell with that
/// error rather than falling back to the (impractically slow, at this
/// scale) filler generator.
pub fn measure(
    ep: Arc<dyn Endpoint>,
    template: &dyn PromptTemplate,
    engine: String,
    endpoint: String,
    trace: &Trace,
    trace_bytes: &[u8],
    cfg: &G4Config,
) -> Record {
    let trace_sha256 = sha256_hex(trace_bytes);
    let metrics = replay(Arc::clone(&ep), trace, &cfg.replay);
    let run = Run::new(cfg.label.clone(), metrics);
    let corpus = cfg.corpus.as_ref().map(|path| load_corpus(path));
    let needles = cfg
        .needle_context_tokens
        .iter()
        .map(|&ctx| match &corpus {
            Some(Ok(ids)) => measure_needle_from_corpus(ep.as_ref(), template, ids, ctx),
            Some(Err(error)) => NeedleResult {
                context_tokens: ctx,
                secret: needle_secret(ctx),
                retrieved: false,
                error: Some(error.clone()),
            },
            None => measure_needle(ep.as_ref(), template, ctx),
        })
        .collect();
    Record {
        session: cfg.session.clone(),
        trace_sha256,
        label: cfg.label.clone(),
        endpoint,
        engine,
        artifact: cfg.artifact.clone(),
        profile: cfg.profile.clone(),
        date: utc_timestamp(unix_now()),
        run,
        needles,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{FinishReason, MockEndpoint, Outcome};

    /// A mock template: one token per whitespace-separated word — matches
    /// `g3.rs`'s own `MockTemplate` (a mock engine with no artifact,
    /// ADR 0006).
    struct MockTemplate;

    impl PromptTemplate for MockTemplate {
        fn encode_user_message(&self, content: &str) -> Result<Vec<u32>, String> {
            Ok(content
                .split_whitespace()
                .map(|w| w.bytes().fold(7u32, |a, b| a.wrapping_mul(131).wrapping_add(b as u32)))
                .collect())
        }
        fn decode(&self, ids: &[u32]) -> Result<String, String> {
            Ok(ids.iter().map(|id| format!("t{id}")).collect::<Vec<_>>().join(" "))
        }
    }

    #[test]
    fn sha256_of_known_bytes_matches_a_known_digest() {
        // The empty string's SHA-256, a standard test vector.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_is_stable_and_sensitive_to_every_byte() {
        let a = sha256_hex(b"the recorded load trace");
        let b = sha256_hex(b"the recorded load trace");
        let c = sha256_hex(b"the recorded load traces");
        assert_eq!(a, b, "the same bytes hash the same way");
        assert_ne!(a, c, "one changed byte changes the digest");
        assert_eq!(a.len(), 64, "lowercase hex of 32 bytes");
    }

    #[test]
    fn a_haystack_lands_on_the_exact_target_and_carries_the_needle_and_question() {
        let template = MockTemplate;
        let secret = needle_secret(200);
        let prompt = build_haystack(&template, 200, &secret).expect("a reachable haystack");
        let count = template.encode_user_message(&prompt).unwrap().len();
        assert_eq!(count, 200, "the haystack must land on the exact target length");
        assert!(prompt.contains(&secret), "the planted fact must be in the haystack");
        assert!(prompt.ends_with(NEEDLE_QUESTION), "the question is asked last");
    }

    #[test]
    fn an_unreachable_target_fails_with_a_clear_message() {
        let template = MockTemplate;
        let secret = needle_secret(1);
        let err = build_haystack(&template, 1, &secret).expect_err("too small to fit the question");
        assert!(err.contains("not reachable"), "{err}");
    }

    #[test]
    fn a_needle_is_retrieved_when_the_answer_contains_the_secret() {
        let secret = needle_secret(200);
        let ep = MockEndpoint::new(vec![Outcome {
            ttft_ms: 5.0,
            total_ms: 5.0,
            n_tokens: 1,
            output: format!("The code is {secret}."),
            prompt_tokens: Some(200),
            cached_prompt_tokens: None,
            token_times_ms: Vec::new(),
            finish_reason: Some(FinishReason::Engine("stop".into())),
        }]);
        let result = measure_needle(&ep, &MockTemplate, 200);
        assert!(result.passed(), "{result:?}");
        assert_eq!(result.context_tokens, 200);
    }

    #[test]
    fn a_needle_is_missed_when_the_answer_lacks_the_secret() {
        let ep = MockEndpoint::new(vec![Outcome {
            ttft_ms: 5.0,
            total_ms: 5.0,
            n_tokens: 1,
            output: "I don't recall any such code.".into(),
            prompt_tokens: Some(200),
            cached_prompt_tokens: None,
            token_times_ms: Vec::new(),
            finish_reason: Some(FinishReason::Engine("stop".into())),
        }]);
        let result = measure_needle(&ep, &MockTemplate, 200);
        assert!(!result.passed());
        assert!(!result.retrieved);
        assert!(result.error.is_none(), "a wrong answer is not an error");
    }

    #[test]
    fn a_failed_request_is_recorded_as_an_error_not_a_panic() {
        struct Failing;
        impl Endpoint for Failing {
            fn complete(&self, _req: &Request) -> Result<Outcome, String> {
                Err("endpoint down".into())
            }
        }
        let result = measure_needle(&Failing, &MockTemplate, 200);
        assert!(!result.passed());
        assert!(result.error.as_deref().unwrap().contains("failed"));
    }

    /// The bank the corpus tests cut from: distinct-ish ids, large enough
    /// to be rotated for a few different target lengths (mirrors
    /// `g3.rs::corpus_bank`).
    fn corpus_bank() -> Vec<u32> {
        (0..2_000).map(|i| ((i * 37) % 900) as u32 + 100).collect()
    }

    #[test]
    fn a_corpus_haystack_lands_on_the_exact_target_and_carries_the_needle_and_question() {
        let template = MockTemplate;
        let bank = corpus_bank();
        let secret = needle_secret(500);
        let prompt =
            build_haystack_from_corpus(&template, &bank, 500, &secret).expect("a reachable haystack");
        let count = template.encode_user_message(&prompt).unwrap().len();
        assert_eq!(count, 500, "the corpus haystack must land on the exact target length");
        assert!(prompt.contains(&secret), "the planted fact must be in the haystack");
        assert!(prompt.ends_with(NEEDLE_QUESTION), "the question is asked last");
    }

    #[test]
    fn a_corpus_haystack_reaches_needle_scale_without_the_ohn2_filler_path() {
        // The regression this path exists to fix: at 64K the filler
        // word-growth path re-encodes the whole (growing) prompt on every
        // appended word against a real tokenizer, which is what made a
        // live smoke test of this cell hang. The corpus path's binary
        // search does a bounded number of encodes regardless of the
        // target, so this must land quickly even at 64K/128K scale.
        let template = MockTemplate;
        let bank = corpus_bank();
        for &target in &[NEEDLE_CONTEXT_64K, NEEDLE_CONTEXT_128K] {
            let secret = needle_secret(target);
            let prompt = build_haystack_from_corpus(&template, &bank, target as usize, &secret)
                .unwrap_or_else(|e| panic!("{target}: {e}"));
            let count = template.encode_user_message(&prompt).unwrap().len();
            assert_eq!(count, target as usize, "{target}");
        }
    }

    #[test]
    fn an_empty_corpus_is_refused() {
        let template = MockTemplate;
        let err = build_haystack_from_corpus(&template, &[], 500, "123456").expect_err("no corpus");
        assert!(err.contains("empty"), "{err}");
    }

    /// A template whose corpus decodes to multi-byte UTF-8 text (a real
    /// tokenizer's vocabulary is not ASCII-only) — the regression a live
    /// smoke test against the real corpus fixture caught: splitting the
    /// decoded body on a byte offset that lands mid-character panics.
    struct MultiByteTemplate;

    impl PromptTemplate for MultiByteTemplate {
        fn encode_user_message(&self, content: &str) -> Result<Vec<u32>, String> {
            Ok(content.split_whitespace().map(|w| w.chars().count() as u32).collect())
        }
        fn decode(&self, ids: &[u32]) -> Result<String, String> {
            // Each id decodes to one multi-byte word ("本" is 3 bytes),
            // repeated `id % 3 + 1` times, so the decoded body's midpoint
            // byte offset lands inside a character for most `n`.
            Ok(ids.iter().map(|id| "本".repeat((*id as usize % 3) + 1)).collect::<Vec<_>>().join(" "))
        }
    }

    #[test]
    fn a_multi_byte_corpus_body_does_not_panic_on_the_split() {
        let template = MultiByteTemplate;
        let bank: Vec<u32> = (0..500).collect();
        let secret = needle_secret(120);
        // The assertion is reaching this line at all: the char-boundary
        // regression panicked before returning any `Result`. Landing
        // exactly is secondary here (encode/decode are not exact inverses
        // for a repeat-based mock).
        let _ = build_haystack_from_corpus(&template, &bank, 120, &secret);
    }

    #[test]
    fn measure_needle_from_corpus_retrieves_a_planted_fact() {
        let bank = corpus_bank();
        let secret = needle_secret(500);
        let ep = MockEndpoint::new(vec![Outcome {
            ttft_ms: 5.0,
            total_ms: 5.0,
            n_tokens: 1,
            output: format!("The code is {secret}."),
            prompt_tokens: Some(500),
            cached_prompt_tokens: None,
            token_times_ms: Vec::new(),
            finish_reason: Some(FinishReason::Engine("stop".into())),
        }]);
        let result = measure_needle_from_corpus(&ep, &MockTemplate, &bank, 500);
        assert!(result.passed(), "{result:?}");
    }

    #[test]
    fn two_different_context_lengths_ask_different_questions() {
        let a = needle_secret(NEEDLE_CONTEXT_64K);
        let b = needle_secret(NEEDLE_CONTEXT_128K);
        assert_ne!(a, b, "each cell plants its own fact");
    }

    #[test]
    fn a_record_round_trips_through_json() {
        let ep: Arc<dyn Endpoint> = Arc::new(MockEndpoint::deterministic());
        let trace = Trace::from_jsonl(
            r#"{"id":"main","class":"main","t_arrive_ms":0,"prompt":"P P P","max_tokens":4,"stream":false}
{"id":"s1","class":"sub","t_arrive_ms":0,"prompt":"Q Q","max_tokens":4,"stream":false}"#,
        )
        .expect("valid trace");
        let cfg = G4Config {
            label: "ignis".into(),
            profile: "test-profile".into(),
            artifact: "mock.ninfer".into(),
            session: "S1".into(),
            needle_context_tokens: vec![80, 160],
            replay: ReplayConfig { max_concurrency: 2, time_scale: 0.0 },
            corpus: None,
        };
        let trace_bytes = b"irrelevant for this test's assertions, only the hash matters";
        let record = measure(ep, &MockTemplate, "mock-engine".into(), "http://mock".into(), &trace, trace_bytes, &cfg);
        assert_eq!(record.trace_sha256, sha256_hex(trace_bytes));
        assert_eq!(record.run.metrics.len(), 2);
        assert_eq!(record.needles.len(), 2);
        let json = record.to_json().expect("serialize");
        assert_eq!(Record::from_json(&json).expect("parse"), record);
        let text = record.render();
        assert!(text.contains("session=S1") && text.contains(&record.trace_sha256));
    }
}
