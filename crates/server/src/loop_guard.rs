//! The **loop guard** (ADR 0052): a task between a request's engine stream
//! and its consumer that watches the reasoning for a loop and, in `rewind`
//! mode, takes the generation back to
//! before it. It covers every request that starts inside its reasoning, on
//! chat completions and on `/v1/responses` over HTTP and the WebSocket.
//! `rewind` is the default; `log` only logs; benches and gates run `off`,
//! for reference parity. A server built without a config ([`crate::Live`]'s
//! default) runs `off`.
//!
//! Why: `docs/findings/2026-10-12-reasoning-loops-are-the-models.md` — about a
//! third of long-enumeration turns loop in their reasoning, on ignis and on
//! ninfer alike, and only the thinking budget's close ends them.
//!
//! **Detection.** Every `check_every` characters of new reasoning, the
//! zlib ratio of the last `window` characters of the reasoning as
//! [`Normalized`] scores it — code taken out, every run of digits read as one
//! `0` — under `ratio` is a loop, when the last `window` characters as
//! written are under `raw_gate` too. Measured offline on 2026-10-13 with this
//! detector (`docs/findings/2026-10-14-reasoning-loop-guard.md`): no real
//! reasoning part of 1,420 under 0.27 at a 1,000-character window, and 12 of
//! the 17 template and counting loops the raw text at 3,000 and 0.12 missed
//! under 0.20. The gate is for tables of numbers (Fibonacci, hex, primes), a
//! template once their digits fold, whose digits carry the information as
//! written: on 47M characters of qwen-code reasoning it takes 5 of 7 such
//! fires away and no harmful loop (the same finding).
//!
//! **Persistence** (ADR 0053). The guard acts on the `persist_checks`-th loop
//! score in a row: a loop persists, a list or a log the reasoning transcribes
//! ends with its content. And it acts only when the cut would keep at least
//! `min_kept_chars` of the reasoning: a reasoning that opens in a loop, or in
//! an enumeration scored as one, has no clean window to go back to, so that
//! run of loop scores is only logged and held ([`Watch::hold`]).
//!
//! **Repeats** (GitHub #727, ADR 0053). A loop whose block is longer than the
//! window never shows two copies in it and scores like prose. Each time a
//! sentence of the scored reasoning ends, the guard also looks for the same
//! sentences back to back ([`Normalized::repeat`]): `repeat_copies` copies,
//! each allowed `repeat_varying` sentences the one before it lacks, compared
//! as written, with at least `repeat_min` characters of prose repeated
//! verbatim per copy and a period up to `repeat_max`. A repeat is acted on
//! when its last copy ends, and its cut is the start of the second copy, which
//! keeps the first ([`Watch::cut`]). Measured offline on 2026-11-27
//! (`docs/findings/2026-11-27-a-loop-longer-than-the-window-repeats-verbatim.md`):
//! verbatim loops of 250 to 8,000 characters caught at their second copy, 21
//! looping agent parts the score never saw, no fire on the legitimate ones.
//!
//! **Rewind.** The engine request is cancelled and a **continuation**
//! submitted: the same input, its prompt extended by the generated tokens
//! kept from before the loop ([`Watch::cut_point`]: back to the start of the
//! last window that scored clean) and a nudge sentence (`nudge`, then
//! `nudge_again`: [`LoopGuardParams::nudge_text`]; after a kept text that
//! ends in a line break, without its own: [`sentence_after`]), with
//! what is left of `max_tokens` and of the thinking budget. Only the kept
//! tokens and the nudge prefill: the continuation's prompt starts with the
//! request's own, whose checkpoint the cancelled request published at its
//! generation opener (ADR 0029), and the engine restores it. Measured at
//! 79k-138k tokens of prompt: 0.08-0.10 s from the tag to the next delta,
//! 1.28 s with about 3,645 tokens kept (the finding). The client keeps what
//! it has already received and is told what was taken back
//! by a tag in the reasoning ([`tag`]): `[LOOP DETECTED, REWINDING n TOKENS /
//! m CHARS · h]`, `m` the characters it received since the cut as it shows
//! them (a line break its decoder still held included) and `h` their
//! [`struck_hash`]. The Playground strikes them through. After `retries`
//! nudges, the next loop is closed with the thinking budget's own close
//! ([`crate::thinking::THINKING_CLOSE_TEXT`]).
//!
//! **Holdback.** With `holdback_chars` (0, off, by default) the client is
//! kept that many characters of reasoning behind the model. A cut in what it
//! has not received yet sends no tag: the struck tokens are dropped and the
//! client gets the kept text, the nudge and the continuation, the model's
//! view token for token. A cut in what it has received strikes only that;
//! the rest is dropped. Everything held goes out at the reasoning's close, at
//! the `Done` (an error's too) and when the engine's stream ends. Tokens
//! dropped unsent still count in `completion_tokens` ([`RewoundTokens`]).
//!
//! **History repair** ([`repair`], [`repair_history`]): on every request,
//! guard on or off, each assistant message's reasoning loses every tagged
//! rewind whose struck text hashes to its tag, so the next turn's prompt is
//! what the model saw. Applied where every prompt is built
//! (`api::conversation`): chat completions, `/v1/responses` (its `reasoning`
//! items are that reasoning) and `/v1/tokenize`.
//!
//! **Cancel by response.** A rewind replaces the engine request: [`Running`]
//! names the one generating now, and on `/v1/responses` what cancels a
//! response (`response.cancel`, a socket closing, an HTTP client hanging up,
//! through `api::CancelOnDrop`) cancels that one. On every surface a
//! consumer that leaves stops it from the guard's side as well, even when
//! the holdback has nothing to send it, and takes a continuation out of its
//! wait for room. A socket's continuation waits in the socket's own
//! admission queue (`responses::queue::QueuedOps`).
//!
//! **On `/v1/responses`** the count is `output_tokens_details.rewound_tokens`;
//! a WebSocket stream continues from the model's view of a rewound response,
//! and `include: ["reasoning.encrypted_content"]` carries that view to the
//! client (`responses::events::reasoning_blob`), which may send it back.
//!
//! Not covered by the guard: a request with images (logged, never rewound);
//! a loop in the answer. A continuation's penalty counts start from zero, and
//! a forced tool call (`tool_choice`) is not forced after a close the guard
//! made.

use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use ignis_core::{RequestClass, RequestId, RequestInput, SchedEvent, SubmitError, TokenId};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::sync::watch;

use crate::decoder::{Channel, OutputDecoder};
use crate::engine::{Engine, EventStream, RequestNotes};
use crate::metrics::{LoopOutcome, Metrics};
use crate::template::{ChatMessage, TemplateProvider};

/// What the guard does with a request (`--model-loop-guard`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoopGuardMode {
    /// No guard: the engine's stream goes to the consumer untouched.
    Off,
    /// Detect and log, never act.
    Log,
    /// Detect, take the loop back and nudge; close after the retries. The
    /// default (the owner, 2026-10-14): benches and gates set `off`.
    #[default]
    Rewind,
}

impl LoopGuardMode {
    /// The modes' written names, in order.
    pub const NAMES: &'static [&'static str] = &["off", "log", "rewind"];

    /// The mode a written name names.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "off" => Some(Self::Off),
            "log" => Some(Self::Log),
            "rewind" => Some(Self::Rewind),
            _ => None,
        }
    }

    /// The mode's written name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Log => "log",
            Self::Rewind => "rewind",
        }
    }
}

/// `--model-loop-guard-ratio`'s default, in thousandths: 0.265. With 4
/// checks in a row ([`DEFAULT_PERSIST_CHECKS`]) the harmful Rome loops fire
/// under 0.257 at the latest, and no legitimate agentic reasoning under 0.274
/// once [`DEFAULT_MIN_KEPT_CHARS`] has taken out what opens in an enumeration
/// (the Rust scorer, `docs/findings/2026-10-16-a-loop-persists.md`); 0.20 was
/// the threshold of a guard that acted at the first check.
pub const DEFAULT_RATIO_PERMILLE: u32 = 265;
/// `--model-loop-guard-window`'s default, in characters of scored reasoning.
pub const DEFAULT_WINDOW_CHARS: u32 = 1000;
/// `--model-loop-guard-normalize`'s default: the reasoning is scored
/// [`Normalized`].
pub const DEFAULT_NORMALIZE: bool = true;
/// `--model-loop-guard-raw-gate`'s default, in thousandths: 0.30.
pub const DEFAULT_RAW_GATE_PERMILLE: u32 = 300;
/// `--model-loop-guard-clean`'s default, in thousandths: 0.40. The longest
/// run of 1,000-character windows under it in 47 clean Rome runs spans
/// 2,750 characters; going back to the last window over it, a rewind cuts
/// before the onset of each of the 33 labelled loops the detector catches.
pub const DEFAULT_CLEAN_PERMILLE: u32 = 400;
/// `--model-loop-guard-margin`'s default, in characters before the clean
/// window: none.
pub const DEFAULT_MARGIN_CHARS: u32 = 0;
/// `--model-loop-guard-retries`'s default: nudges before the close.
pub const DEFAULT_RETRIES: u32 = 2;
/// `--model-loop-guard-holdback`'s default, in characters of reasoning held
/// back from the client: none.
pub const DEFAULT_HOLDBACK_CHARS: u32 = 0;
/// `--model-loop-guard-check-every`'s default, in characters of new
/// reasoning between two scores.
pub const DEFAULT_CHECK_EVERY_CHARS: u32 = 250;
/// `--model-loop-guard-snap`'s default, in characters: how far back from the
/// cut a line break pulls it.
pub const DEFAULT_SNAP_CHARS: u32 = 300;
/// `--model-loop-guard-min-continuation`'s default, in tokens of `max_tokens`
/// a continuation must have left.
pub const DEFAULT_MIN_CONTINUATION_TOKENS: u32 = 512;
/// `--model-loop-guard-submit-wait`'s default, in seconds a continuation the
/// engine refuses as full waits for room.
pub const DEFAULT_SUBMIT_WAIT_SECS: u32 = 60;
/// `--model-loop-guard-persist`'s default: consecutive loop checks before
/// the guard acts. A loop is absorbing and a legitimate enumeration ends
/// with its content: no list, log or table in the opencode and qwen-code
/// reasoning stays under the ratio for more than 6 checks in a row, while
/// 23 of the 25 harmful Rome loops stay for 9 or more
/// (`docs/findings/2026-10-16-a-loop-persists.md`).
pub const DEFAULT_PERSIST_CHECKS: u32 = 4;
/// `--model-loop-guard-min-kept`'s default, in characters: a rewind must keep
/// at least one, or it is only logged. A reasoning that opens in a loop, or
/// in an enumeration the score takes for one, has no clean window to go back
/// to, and a rewind there would drop all of it.
pub const DEFAULT_MIN_KEPT_CHARS: u32 = 1;
/// `--model-loop-guard-repeat-min`'s default, in scored characters: the
/// least a copy must repeat verbatim, the sentences that vary left out, for
/// the repeat signal.
pub const DEFAULT_REPEAT_MIN_CHARS: u32 = 150;
/// `--model-loop-guard-repeat-max`'s default, in scored characters: the
/// longest copy the repeat signal looks for.
pub const DEFAULT_REPEAT_MAX_CHARS: u32 = 16_000;
/// `--model-loop-guard-repeat-copies`'s default: the copies back to back
/// that make a repeat, the last one complete.
pub const DEFAULT_REPEAT_COPIES: u32 = 2;
/// `--model-loop-guard-repeat-varying`'s default: the sentences a copy may
/// have that the one before it does not.
pub const DEFAULT_REPEAT_VARYING: u32 = 1;
/// `--model-loop-guard-nudge`'s default: the model's own words when it
/// caught itself (2026-10-12 seed 109: "I'm repeating. Let me stop the
/// repetition and just write the content directly.").
pub const DEFAULT_NUDGE: &str = "Wait, I'm starting to repeat myself. Let me stop and move on with what I already have.";
/// `--model-loop-guard-nudge-again`'s default: the sentence of every rewind
/// after the first.
pub const DEFAULT_NUDGE_AGAIN: &str =
    "I keep repeating myself. I already have enough to write the answer, so I'll wrap up my reasoning now.";

/// The guard's knobs, as the live config holds them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopGuardParams {
    pub mode: LoopGuardMode,
    /// A window whose zlib ratio is under this many thousandths is a loop.
    pub ratio_permille: u32,
    /// The window, in characters of scored reasoning.
    pub window_chars: u32,
    /// Score the reasoning [`Normalized`]; off, as written.
    pub normalize: bool,
    /// A loop's last `window` raw characters must also score under this
    /// many thousandths; 0, no gate.
    pub raw_gate_permille: u32,
    /// A window scoring at least this many thousandths is clean: a rewind
    /// goes back to the start of the last one before the loop.
    pub clean_permille: u32,
    /// How many characters before that window's start a rewind cuts.
    pub margin_chars: u32,
    /// The nudges a looping reasoning gets before it is closed.
    pub retries: u32,
    /// How many checks in a row must score a loop before the guard acts;
    /// 1, the first.
    pub persist_checks: u32,
    /// A rewind that would keep fewer characters of reasoning than this is
    /// only logged; 0, never.
    pub min_kept_chars: u32,
    /// In `rewind` mode, how many characters of reasoning the client is
    /// kept behind the model: a rewind whose cut falls in them reaches the
    /// client with no tag. 0, none.
    pub holdback_chars: u32,
    /// The window is scored again after this many characters of new
    /// reasoning.
    pub check_every_chars: u32,
    /// The cut moves back to the line break before it, when one is within
    /// this many characters. 0, never.
    pub snap_chars: u32,
    /// The least of `max_tokens` a continuation must have left to be worth
    /// one: with less, the loop is only logged.
    pub min_continuation_tokens: u32,
    /// How long a continuation the engine refuses as full waits for room,
    /// in seconds, before the loop is left as it is. 0, it is tried once.
    pub submit_wait_secs: u32,
    /// The repeat signal: the least a copy repeats verbatim, in scored
    /// characters, the sentences that vary left out.
    pub repeat_min_chars: u32,
    /// The repeat signal's longest copy, in scored characters. 0, no repeat
    /// signal.
    pub repeat_max_chars: u32,
    /// The copies back to back that make a repeat.
    pub repeat_copies: u32,
    /// The sentences a copy may have that the one before it does not.
    pub repeat_varying: u32,
    /// The sentence the first rewind puts after the kept reasoning.
    pub nudge: String,
    /// The sentence every later rewind puts there, until the close.
    pub nudge_again: String,
}

impl LoopGuardParams {
    /// The nudge of the rewind that follows `rewinds` earlier ones: `nudge`
    /// for the first, `nudge_again` for every later one, as a paragraph of
    /// its own. After a kept text that ends in a line break, it goes in
    /// without its leading ones ([`sentence_after`]).
    pub fn nudge_text(&self, rewinds: u32) -> String {
        let sentence = if rewinds == 0 { &self.nudge } else { &self.nudge_again };
        format!("\n\n{}\n\n", sentence.trim())
    }

    /// Whether the guard watches `input`, which starts inside its reasoning
    /// when `starts_in_reasoning`: in `log` or `rewind` mode, on a request
    /// that reasons and generates (never a warm-up).
    pub fn watches(&self, input: &RequestInput, starts_in_reasoning: bool) -> bool {
        self.mode != LoopGuardMode::Off && starts_in_reasoning && !input.warm_up
    }

    /// Whether the guard may rewind `input`, and so hold its reasoning back:
    /// in `rewind` mode, on a request with no images, which a continuation
    /// cannot carry yet.
    pub fn rewinds(&self, input: &RequestInput) -> bool {
        self.mode == LoopGuardMode::Rewind && input.multimodal.is_none()
    }
}

impl Default for LoopGuardParams {
    fn default() -> Self {
        Self {
            mode: LoopGuardMode::default(),
            ratio_permille: DEFAULT_RATIO_PERMILLE,
            window_chars: DEFAULT_WINDOW_CHARS,
            normalize: DEFAULT_NORMALIZE,
            raw_gate_permille: DEFAULT_RAW_GATE_PERMILLE,
            clean_permille: DEFAULT_CLEAN_PERMILLE,
            margin_chars: DEFAULT_MARGIN_CHARS,
            retries: DEFAULT_RETRIES,
            persist_checks: DEFAULT_PERSIST_CHECKS,
            min_kept_chars: DEFAULT_MIN_KEPT_CHARS,
            holdback_chars: DEFAULT_HOLDBACK_CHARS,
            check_every_chars: DEFAULT_CHECK_EVERY_CHARS,
            snap_chars: DEFAULT_SNAP_CHARS,
            min_continuation_tokens: DEFAULT_MIN_CONTINUATION_TOKENS,
            submit_wait_secs: DEFAULT_SUBMIT_WAIT_SECS,
            repeat_min_chars: DEFAULT_REPEAT_MIN_CHARS,
            repeat_max_chars: DEFAULT_REPEAT_MAX_CHARS,
            repeat_copies: DEFAULT_REPEAT_COPIES,
            repeat_varying: DEFAULT_REPEAT_VARYING,
            nudge: DEFAULT_NUDGE.to_owned(),
            nudge_again: DEFAULT_NUDGE_AGAIN.to_owned(),
        }
    }
}

/// The sentence a continuation's prompt carries after `kept`, the kept
/// reasoning as the client shows it: `sentence` (a nudge, the close) as
/// written, or without its leading line breaks when `kept` already ends in
/// one. Otherwise the kept text's last newline token and the sentence's
/// opening `\n\n` re-tokenize as one (`"." "\n\n" | "\n\n" "Wait"` is
/// `"." "\n\n\n\n" "Wait"` from the text), and the next turn's prompt would
/// leave the retained state at the cut (`examples/loop_retokenize.rs`).
pub fn sentence_after<'a>(kept: &str, sentence: &'a str) -> &'a str {
    if kept.ends_with('\n') {
        sentence.trim_start_matches('\n')
    } else {
        sentence
    }
}

/// The tag's fixed opening, for a client that looks for it.
pub const TAG_PREFIX: &str = "[LOOP DETECTED, REWINDING ";

/// The tag's whole opening as [`tag`] writes it, its line breaks included.
const TAG_OPEN: &str = "\n\n[LOOP DETECTED, REWINDING ";

/// The tag the client reads in the reasoning where the guard took back
/// `struck`, the reasoning it had received since the cut, in `tokens`
/// tokens: `\n\n[LOOP DETECTED, REWINDING <n> TOKENS / <m> CHARS · <h>]`,
/// `m` its length in Unicode scalar values and `h` its [`struck_hash`]. The
/// `m` characters right before the tag's line breaks are no longer the
/// model's context, and [`repair`] takes them out when they hash to `h`. It
/// opens with a line break so that it never joins a character the last token
/// left incomplete; what follows it (a nudge, the close) opens with its own
/// unless the kept text ended in one ([`sentence_after`]).
pub fn tag(tokens: usize, struck: &str) -> String {
    format!("{TAG_OPEN}{tokens} TOKENS / {} CHARS \u{b7} {}]", struck.chars().count(), struck_hash(struck))
}

/// The first 6 lowercase hex digits of the SHA-256 of `text`'s UTF-8: what
/// a [`tag`] carries of its struck text.
pub fn struck_hash(text: &str) -> String {
    Sha256::digest(text.as_bytes())[..3].iter().map(|b| format!("{b:02x}")).collect()
}

/// **History repair** (ADR 0052): `reasoning` as a client sends it back,
/// with every rewind the guard took out of it again, so the model is given
/// the text it saw. For each [`tag`], from the last to the first, the `m`
/// characters before it and the tag itself go, if and only if those
/// characters hash to the tag's `h`. A mismatch, fewer than `m` characters
/// before it, or a tag without a hash (the format before ADR 0052's hash)
/// leaves that text as it is.
pub fn repair(reasoning: &str) -> Cow<'_, str> {
    if !reasoning.contains(TAG_OPEN) {
        return Cow::Borrowed(reasoning);
    }
    let mut text = reasoning.to_owned();
    let mut limit = text.len();
    while let Some(open) = text[..limit].rfind(TAG_OPEN) {
        let struck = parse_tag(&text[open..limit]).and_then(|(len, chars, hash)| {
            let start = start_of_last(&text[..open], chars)?;
            (struck_hash(&text[start..open]) == hash).then_some((start, open + len))
        });
        match struck {
            Some((start, end)) => {
                text.replace_range(start..end, "");
                limit = start;
            }
            None => limit = open,
        }
    }
    Cow::Owned(text)
}

/// [`repair`] over a conversation: the reasoning of each assistant message,
/// and nothing else — never its content, never another role's text.
pub fn repair_history(messages: &mut [ChatMessage]) {
    for message in messages.iter_mut().filter(|message| message.role == "assistant") {
        if let Some(reasoning) = message.reasoning_content.as_mut() {
            if let Cow::Owned(repaired) = repair(reasoning) {
                *reasoning = repaired;
            }
        }
    }
}

/// A [`tag`] at the start of `text`: its length in bytes, its `m` and its
/// `h`. `None` when `text` does not start with one in the hashed format.
fn parse_tag(text: &str) -> Option<(usize, usize, &str)> {
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let rest = text.strip_prefix(TAG_OPEN)?;
    let (tokens, rest) = rest.split_once(" TOKENS / ")?;
    let (chars, rest) = rest.split_once(" CHARS \u{b7} ")?;
    let hash = rest.get(..6)?;
    let hex = hash.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    if !digits(tokens) || !digits(chars) || !hex || !rest[6..].starts_with(']') {
        return None;
    }
    Some((text.len() - rest.len() + 7, chars.parse().ok()?, hash))
}

/// Where the last `chars` characters of `text` start; `None` when it has
/// fewer.
fn start_of_last(text: &str, chars: usize) -> Option<usize> {
    match chars {
        0 => Some(text.len()),
        _ => text.char_indices().rev().nth(chars - 1).map(|(at, _)| at),
    }
}

/// The zlib ratio of `text`: compressed bytes over UTF-8 bytes. Lower is more
/// repetitive; normal reasoning sits at 0.30-0.54 over 3,000 characters.
pub fn compress_ratio(text: &str) -> f64 {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return 1.0;
    }
    miniz_oxide::deflate::compress_to_vec_zlib(bytes, 9).len() as f64 / bytes.len() as f64
}

fn ratio_of(chars: &[char]) -> f64 {
    compress_ratio(&chars.iter().collect::<String>())
}

/// The reasoning as the detector scores it, built one character at a time
/// as it streams. With `normalize` on, fenced code blocks (one still open
/// too: code being written never counts) and inline code are taken out and
/// every run of ASCII digits reads as one `0`: a template loop whose slots
/// are numbers ("the 321 AD, the 322 AD") repeats verbatim here, while code,
/// which repeats by nature, is not scored. Off, it is the reasoning as
/// written.
///
/// A character is kept only once it is decided — a backtick or two that may
/// still open a block, an inline span that may still close on its line,
/// wait — and keeps the offset of the raw character it stands for (a run of
/// digits, its first), so a window found here is cut on the raw reasoning.
/// The steps are those of the offline research scorer the finding checks
/// parity against (`docs/findings/2026-10-14-reasoning-loop-guard.md`), in
/// its order: blocks, then inline code on what is left, then digits. The raw
/// reasoning
/// is kept beside it, for the raw gate and the cut.
pub struct Normalized {
    steps: Vec<Box<dyn Step + Send>>,
    text: Vec<char>,
    raw_at: Vec<usize>,
    raw: Vec<char>,
    /// What one step hands the next.
    pass: Vec<(char, usize)>,
    spare: Vec<(char, usize)>,
    /// The scored text's sentences, for the repeat signal.
    sentences: Sentences,
}

impl Normalized {
    /// Off, every character is kept as written.
    pub fn new(normalize: bool) -> Self {
        let steps: Vec<Box<dyn Step + Send>> = if normalize {
            vec![Box::new(Fences::default()), Box::new(InlineCode::default()), Box::new(Digits::default())]
        } else {
            Vec::new()
        };
        Self {
            steps,
            text: Vec::new(),
            raw_at: Vec::new(),
            raw: Vec::new(),
            pass: Vec::new(),
            spare: Vec::new(),
            sentences: Sentences::default(),
        }
    }

    /// The next character of the raw reasoning.
    pub fn push(&mut self, c: char) {
        self.pass.clear();
        self.pass.push((c, self.raw.len()));
        self.raw.push(c);
        for step in &mut self.steps {
            self.spare.clear();
            for &(c, raw) in &self.pass {
                step.push(c, raw, &mut self.spare);
            }
            std::mem::swap(&mut self.pass, &mut self.spare);
        }
        for &(c, raw) in &self.pass {
            self.text.push(c);
            self.raw_at.push(raw);
            self.sentences.push(&self.text, &self.raw_at, &self.raw);
        }
    }

    /// The sentences of scored text finished so far.
    pub fn sentence_count(&self) -> usize {
        self.sentences.all.len()
    }

    /// The repeat the scored text ends in, both its last copies after the
    /// raw character `floor`, under `params`' `repeat_*` limits; see
    /// [`Repeat`].
    ///
    /// The periods tried, shortest first, are the distances back to an
    /// earlier copy of one of the last sentences: as many of them as may
    /// vary in a copy, and one more, so one of them is a copy's own. At each,
    /// the sentences from the last one back are compared with the ones a
    /// period before them, until one more than `repeat_varying` differ
    /// within a period, or `repeat_max` characters back. The shortest period
    /// whose copies reach `repeat_copies` decides: a repeat when a copy
    /// repeats at least `repeat_min` characters of prose verbatim, none when
    /// a shorter block repeating over and over does (a loop of one short
    /// sentence: the compression score's), the next period when neither, as
    /// two short sentences in a row are. Lines of material take part in the
    /// copies but count for nothing towards `repeat_min`: a listing or a log
    /// transcribed twice, to compare two outputs, is no loop.
    ///
    /// A copy is read from the earliest sentence that matches: when the
    /// sentence that varies opens each copy, the copies start after it, so
    /// the repeat is found one sentence into the third copy and its cut keeps
    /// the second copy's varying sentence. With one sentence free to vary,
    /// `X A B C D A B C D` is as much a loop of `D A B C` whose first sentence
    /// varies as one of `A B C D`; the later reading never cuts into what
    /// came before the loop.
    pub fn repeat(&self, floor: usize, params: &LoopGuardParams) -> Option<Repeat> {
        let sentences = &self.sentences;
        let last = sentences.all.len().checked_sub(1)?;
        let first = sentences.all.partition_point(|sentence| self.raw_at[sentence.span.0] < floor);
        let varying = params.repeat_varying as usize;
        let (min, max) = (params.repeat_min_chars as usize, params.repeat_max_chars as usize);
        let copies = f64::from(params.repeat_copies.max(2));
        let end = sentences.all[last].span.1;
        let seeds: Vec<usize> = (first..=last).rev().filter(|&j| sentences.has_text(j)).take(varying + 1).collect();
        // For each seed, its earlier copies (`by_text` is in order): the
        // next one to propose a period is the last of the slice.
        let mut earlier: Vec<(usize, &[usize])> = seeds
            .iter()
            .map(|&j| {
                let all = &sentences.by_text[&sentences.all[j].hash][..];
                (j, &all[..all.partition_point(|&i| i < j)])
            })
            .collect();
        let mut tried = 0;
        loop {
            let mut period = usize::MAX;
            for (j, before) in &mut earlier {
                while let Some((&i, rest)) = before.split_last() {
                    if i < first {
                        *before = &[];
                    } else if *j - i <= tried || !sentences.same(&self.raw, i, *j) {
                        *before = rest;
                    } else {
                        period = period.min(*j - i);
                        break;
                    }
                }
            }
            if period == usize::MAX || last < first + period {
                return None;
            }
            tried = period;
            let chars = end - sentences.all[last + 1 - period].span.0;
            if chars > max {
                return None;
            }
            let needed = (copies - 1.0) * chars as f64;
            let enough = |span: usize, matched: usize| span as f64 >= needed && matched >= min || span > max;
            let Some((low, matched)) = sentences.walk(&self.raw, period, first, varying, enough) else { continue };
            let span = end - sentences.all[low].span.0;
            if (span as f64) < needed {
                continue;
            }
            if matched as f64 * chars as f64 / span as f64 >= min as f64 {
                let (low, _) = sentences.walk(&self.raw, period, first, varying, |_, _| false)?;
                return Some(Repeat {
                    period: chars,
                    copies: 1.0 + (end - sentences.all[low].span.0) as f64 / chars as f64,
                    first: self.raw_at[sentences.all[low - period].span.0],
                    second: self.raw_at[sentences.all[low].span.0],
                    end: self.raw.len(),
                });
            }
            if matched >= min {
                return None;
            }
        }
    }

    /// The scored text so far.
    pub fn text(&self) -> &[char] {
        &self.text
    }

    /// The raw character the scored character `i` stands for.
    pub fn raw_at(&self, i: usize) -> usize {
        self.raw_at[i]
    }

    /// The raw reasoning so far.
    pub fn raw(&self) -> &[char] {
        &self.raw
    }

    /// The raw characters pushed so far.
    pub fn raw_len(&self) -> usize {
        self.raw.len()
    }
}

/// One step of [`Normalized`]: characters in, each with the raw offset it
/// stands for; out, what the next step sees. Another fold is one more step.
trait Step {
    fn push(&mut self, c: char, raw: usize, out: &mut Vec<(char, usize)>);
}

/// Fenced code blocks out, the fences with them: three backticks in a row
/// open a block, the next three close it (```` ```.*?``` ````). One or two
/// backticks outside a block are held until the next character says whether
/// they open one.
#[derive(Default)]
struct Fences {
    open: bool,
    /// The raw offset of the open block's first backtick.
    opened_at: usize,
    /// The raw offsets of the backticks in a row just seen.
    ticks: Vec<usize>,
}

impl Step for Fences {
    fn push(&mut self, c: char, raw: usize, out: &mut Vec<(char, usize)>) {
        if c == '`' {
            self.ticks.push(raw);
            if self.ticks.len() == 3 {
                self.open = !self.open;
                self.opened_at = self.ticks[0];
                self.ticks.clear();
            }
            return;
        }
        if !self.open {
            out.extend(self.ticks.iter().map(|&at| ('`', at)));
            out.push((c, raw));
        }
        self.ticks.clear();
    }
}

/// The first backtick of the code block `text` ends inside, if it does: by
/// [`Fences`]' own rule.
fn open_block(text: &[char]) -> Option<usize> {
    let mut fences = Fences::default();
    let mut out = Vec::new();
    for (at, &c) in text.iter().enumerate() {
        fences.push(c, at, &mut out);
        out.clear();
    }
    fences.open.then_some(fences.opened_at)
}

/// Inline code out: a backtick and what follows it on its line, up to and
/// with the next backtick (`` `[^`\n]*` ``). A line break before that one
/// makes the span text.
#[derive(Default)]
struct InlineCode {
    open: bool,
    span: Vec<(char, usize)>,
}

impl Step for InlineCode {
    fn push(&mut self, c: char, raw: usize, out: &mut Vec<(char, usize)>) {
        if !self.open {
            if c == '`' {
                self.open = true;
                self.span.push((c, raw));
            } else {
                out.push((c, raw));
            }
            return;
        }
        match c {
            '`' => {
                self.open = false;
                self.span.clear();
            }
            '\n' => {
                self.open = false;
                out.append(&mut self.span);
                out.push((c, raw));
            }
            _ => self.span.push((c, raw)),
        }
    }
}

/// Every run of ASCII digits as one `0`.
#[derive(Default)]
struct Digits {
    run: bool,
}

impl Step for Digits {
    fn push(&mut self, c: char, raw: usize, out: &mut Vec<(char, usize)>) {
        if c.is_ascii_digit() {
            if !self.run {
                out.push(('0', raw));
            }
            self.run = true;
        } else {
            self.run = false;
            out.push((c, raw));
        }
    }
}

/// A sentence's stop: `.`, `!` or `?`, which ends one before white space
/// (so a file name, a decimal or `!=` ends none), past any [`closes`].
fn stops(c: char) -> bool {
    matches!(c, '.' | '!' | '?')
}

/// A full-width stop, which ends a sentence where it stands: the languages
/// that write it put no space after it.
fn stops_wide(c: char) -> bool {
    matches!(c, '。' | '！' | '？')
}

/// What may follow a sentence's stop and still close the sentence: a quote,
/// a bracket, emphasis.
fn closes(c: char) -> bool {
    matches!(c, '"' | '\'' | ')' | ']' | '}' | '*' | '_' | '`' | '”' | '’' | '»' | '」' | '』' | '）')
}

/// `text` ends in a stop, past its closers and white space.
fn ends_in_stop(text: &[char]) -> bool {
    let end = text.iter().rposition(|&c| !c.is_whitespace() && !closes(c));
    end.is_some_and(|end| stops(text[end]) || stops_wide(text[end]))
}

/// The scored text cut into sentences as it streams, for the repeat signal
/// ([`Normalized::repeat`]): a sentence ends at a line break, at a
/// full-width stop, and at white space after a stop. One that ends in a
/// stop is **prose**; a line that ends in none is a line of **material**: a
/// list's item, a log's or a table's line, a heading. Two sentences are the
/// same when the raw text they span, white space trimmed off both ends, is:
/// a table or a list whose numbers differ repeats once its digits fold, not
/// as written.
#[derive(Default)]
struct Sentences {
    /// Every finished sentence, in order.
    all: Vec<Sentence>,
    /// The sentences with text, by their hash, in order.
    by_text: HashMap<u64, Vec<usize>>,
    /// Where the sentence being written starts.
    open: usize,
}

/// One finished sentence of [`Sentences`].
struct Sentence {
    /// Its span of the scored text.
    span: (usize, usize),
    /// The raw text it spans, white space trimmed off both ends.
    trimmed: (usize, usize),
    /// The FNV-1a hash of that trimmed raw text.
    hash: u64,
    /// It ends in a stop.
    prose: bool,
}

impl Sentence {
    /// Its scored characters.
    fn chars(&self) -> usize {
        self.span.1 - self.span.0
    }
}

impl Sentences {
    /// After the scored `text`'s last character was appended, `raw_at` its
    /// characters' raw offsets in `raw`: it may end a sentence, or show that
    /// the one before it ended.
    fn push(&mut self, text: &[char], raw_at: &[usize], raw: &[char]) {
        let at = text.len() - 1;
        let end = if text[at] == '\n' || stops_wide(text[at]) {
            at + 1
        } else if text[at].is_whitespace() && at > self.open && ends_in_stop(&text[self.open..at]) {
            // Ends before this white space, which opens the next one.
            at
        } else {
            return;
        };
        let start = std::mem::replace(&mut self.open, end);
        let (mut from, mut to) = (raw_at[start], raw_at[end - 1] + 1);
        while from < to && raw[from].is_whitespace() {
            from += 1;
        }
        while to > from && raw[to - 1].is_whitespace() {
            to -= 1;
        }
        let hash = raw[from..to]
            .iter()
            .fold(0xcbf2_9ce4_8422_2325u64, |hash, &c| (hash ^ u64::from(u32::from(c))).wrapping_mul(0x0000_0100_0000_01b3));
        if to > from {
            self.by_text.entry(hash).or_default().push(self.all.len());
        }
        self.all.push(Sentence { span: (start, end), trimmed: (from, to), hash, prose: ends_in_stop(&text[start..end]) });
    }

    /// Sentence `i` has text beside white space.
    fn has_text(&self, i: usize) -> bool {
        self.all[i].trimmed.0 < self.all[i].trimmed.1
    }

    /// Sentences `a` and `b` are the same, as `raw` has them.
    fn same(&self, raw: &[char], a: usize, b: usize) -> bool {
        let (a, b) = (&self.all[a], &self.all[b]);
        a.hash == b.hash && raw[a.trimmed.0..a.trimmed.1] == raw[b.trimmed.0..b.trimmed.1]
    }

    /// From the last sentence back, each compared with the one `period`
    /// before it, none of those before `first`: the earliest that is the
    /// same, and the characters of the prose that is, from it on. The walk
    /// stops at one more than `varying` different ones within a period, or
    /// once `enough` says so of the span (scored characters from that
    /// sentence's start to the end) and those characters.
    fn walk(&self, raw: &[char], period: usize, first: usize, varying: usize, enough: impl Fn(usize, usize) -> bool) -> Option<(usize, usize)> {
        let last = self.all.len() - 1;
        let end = self.all[last].span.1;
        let mut differ: VecDeque<usize> = VecDeque::new();
        let (mut found, mut matched) = (None, 0);
        let mut j = last;
        while j >= first + period {
            if self.same(raw, j, j - period) {
                if self.all[j].prose {
                    matched += self.all[j].chars();
                }
                found = Some((j, matched));
                if enough(end - self.all[j].span.0, matched) {
                    break;
                }
            } else {
                while differ.front().is_some_and(|&k| k >= j + period) {
                    differ.pop_front();
                }
                if differ.len() >= varying {
                    break;
                }
                differ.push_back(j);
            }
            j -= 1;
        }
        found
    }
}

/// A repeat the reasoning ends in, which the repeat signal found
/// ([`Normalized::repeat`]): the same sentences back to back, copy after
/// copy, each copy allowed to differ from the one before it in
/// `repeat_varying` sentences. A loop whose block is too long for the
/// compression score's window to hold two copies of it is one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Repeat {
    /// One copy, in scored characters: the repeat's period.
    pub period: usize,
    /// The copies back to back, the last one counted in part.
    pub copies: f64,
    /// The raw reasoning character the first copy starts at.
    pub first: usize,
    /// The raw character the second copy starts at: a rewind keeps the
    /// first copy and cuts here.
    pub second: usize,
    /// The raw reasoning length it was found at.
    pub end: usize,
}

/// What the watch found: a loop score that persisted, or a repeat.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Found {
    /// A loop score, the `persist_checks`-th in a row.
    Score(Score),
    /// A repeat ([`Normalized::repeat`]).
    Repeat(Repeat),
}

impl Found {
    /// The log's name for the signal that found it.
    pub fn signal(&self) -> &'static str {
        match self {
            Self::Score(_) => "compression",
            Self::Repeat(_) => "repeat",
        }
    }

    /// The raw character what it looked at starts at: the window's, or the
    /// repeat's first copy's.
    pub fn start(&self) -> usize {
        match self {
            Self::Score(score) => score.start,
            Self::Repeat(repeat) => repeat.first,
        }
    }

    /// The raw reasoning length it was found at.
    pub fn end(&self) -> usize {
        match self {
            Self::Score(score) => score.end,
            Self::Repeat(repeat) => repeat.end,
        }
    }

    /// The loop score, when a score found it.
    pub fn score(&self) -> Option<&Score> {
        match self {
            Self::Score(score) => Some(score),
            Self::Repeat(_) => None,
        }
    }

    /// The repeat, when the repeat signal found it.
    pub fn repeat(&self) -> Option<&Repeat> {
        match self {
            Self::Repeat(repeat) => Some(repeat),
            Self::Score(_) => None,
        }
    }
}

/// A scored window of reasoning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// Its zlib ratio ([`compress_ratio`]).
    pub ratio: f64,
    /// The zlib ratio of the last `window` raw characters when it was taken.
    pub raw_ratio: f64,
    /// The raw reasoning character it starts at.
    pub start: usize,
    /// The raw reasoning length it was taken at: where it ends.
    pub end: usize,
}

impl Score {
    /// Under `ratio`, and as written under `raw_gate` (when there is one):
    /// a loop.
    pub fn is_loop(&self, params: &LoopGuardParams) -> bool {
        self.ratio * 1000.0 < f64::from(params.ratio_permille)
            && (params.raw_gate_permille == 0 || self.raw_ratio * 1000.0 < f64::from(params.raw_gate_permille))
    }

    /// At `clean` or over: no loop in it.
    pub fn is_clean(&self, params: &LoopGuardParams) -> bool {
        self.ratio * 1000.0 >= f64::from(params.clean_permille)
    }
}

/// When to score the window, from where a loop may be found, and what was
/// scored since.
#[derive(Debug, Clone)]
pub struct Watch {
    /// No window that starts before this raw character is scored: the end
    /// of the last nudge, or of the last logged loop.
    floor: usize,
    /// The raw reasoning length the next score waits for.
    next_check: usize,
    /// Every score taken since the floor, in order.
    history: Vec<Score>,
    /// How many of the last scores in a row were a loop.
    loop_run: u32,
    /// The run of loop scores in progress is not acted on ([`Watch::hold`]).
    held: bool,
    /// The sentences of scored text the repeat signal has looked at.
    sentences: usize,
    /// The repeat in progress is not acted on ([`Watch::hold`]).
    repeat_held: bool,
}

impl Watch {
    /// A watch whose first score is the first full window after `floor`.
    pub fn from(floor: usize, params: &LoopGuardParams) -> Self {
        Self {
            floor,
            next_check: floor + params.window_chars as usize,
            history: Vec::new(),
            loop_run: 0,
            held: false,
            sentences: 0,
            repeat_held: false,
        }
    }

    /// The window's score when the reasoning has grown enough to score it
    /// again: the first time a whole window of scored text lies after the
    /// floor, then every `check_every` raw characters. Its raw
    /// window, the last `window` raw characters, lies after the floor too:
    /// the scored one spans at least as many.
    pub fn score(&mut self, scored: &Normalized, params: &LoopGuardParams) -> Option<Score> {
        if scored.raw_len() < self.next_check {
            return None;
        }
        let window = params.window_chars as usize;
        let start = scored.text().len().checked_sub(window)?;
        if scored.raw_at(start) < self.floor {
            return None;
        }
        self.next_check = scored.raw_len() + params.check_every_chars as usize;
        let raw = scored.raw();
        let score = Score {
            ratio: ratio_of(&scored.text()[start..]),
            raw_ratio: ratio_of(&raw[raw.len() - window..]),
            start: scored.raw_at(start),
            end: raw.len(),
        };
        self.history.push(score);
        Some(score)
    }

    /// What the watch finds after the reasoning grew: a repeat, each time a
    /// sentence ends ([`Normalized::repeat`]), or [`Watch::score`], when it
    /// is a loop and the `persist_checks`-th in a row or later (every one,
    /// at 1), unless what found it is held. A score that is not a loop starts
    /// the count over and ends the scores' hold; a sentence that ends no
    /// repeat ends the repeat's.
    pub fn check(&mut self, scored: &Normalized, params: &LoopGuardParams) -> Option<Found> {
        let repeat = self.repeat(scored, params);
        let score = self.persisted(scored, params);
        repeat.map(Found::Repeat).or(score.map(Found::Score))
    }

    fn repeat(&mut self, scored: &Normalized, params: &LoopGuardParams) -> Option<Repeat> {
        let sentences = scored.sentence_count();
        if sentences == self.sentences || params.repeat_max_chars == 0 {
            return None;
        }
        self.sentences = sentences;
        match scored.repeat(self.floor, params) {
            Some(repeat) => (!self.repeat_held).then_some(repeat),
            None => {
                self.repeat_held = false;
                None
            }
        }
    }

    fn persisted(&mut self, scored: &Normalized, params: &LoopGuardParams) -> Option<Score> {
        let score = self.score(scored, params)?;
        if !score.is_loop(params) {
            (self.loop_run, self.held) = (0, false);
            return None;
        }
        self.loop_run += 1;
        // `max(1)`: a configured value is at least 1, a hand-built one may be 0.
        (self.loop_run >= params.persist_checks.max(1) && !self.held).then_some(score)
    }

    /// Act on no more loop scores until one that is not a loop, and on no
    /// more repeats until a sentence that ends none: what is in progress is
    /// left as it is, and the scores taken so far stay for a later cut.
    pub fn hold(&mut self) {
        (self.held, self.repeat_held) = (true, true);
    }

    /// Where to take the raw reasoning `text` back to, for what the watch
    /// `found`: for a loop score, [`Watch::cut_point`]; for a repeat, the
    /// start of its second copy, which keeps the first (a sentence's start
    /// already, so no snap and no margin). A cut inside a code block moves
    /// before its fence as [`Watch::cut_point`]'s does, never before the
    /// floor.
    pub fn cut(&self, text: &[char], found: &Found, params: &LoopGuardParams) -> usize {
        match found {
            Found::Score(_) => self.cut_point(text, params),
            Found::Repeat(repeat) => {
                // Both copies lie after the floor ([`Normalized::repeat`]).
                let cut = repeat.second;
                match open_block(&text[..cut]) {
                    Some(fence) => fence.max(self.floor),
                    None => cut,
                }
            }
        }
    }

    /// Where to take the raw reasoning `text` back to, for the loop the last
    /// score found: the start of the last window before it that scored
    /// clean, `margin` characters earlier, then the line break before that
    /// when one is within `snap`; with no clean window since the
    /// floor, the floor. Never before the floor.
    ///
    /// A cut inside a code block moves to just before the block's opening
    /// fence: an open block is never scored, so the continuation would go
    /// unwatched until the model closed it. (Unless the block opened before
    /// the floor, which only a loop logged inside one leaves.)
    pub fn cut_point(&self, text: &[char], params: &LoopGuardParams) -> usize {
        let before = &self.history[..self.history.len().saturating_sub(1)];
        let cut = match before.iter().rev().find(|score| score.is_clean(params)) {
            Some(clean) => clean.start.saturating_sub(params.margin_chars as usize).max(self.floor),
            None => self.floor,
        };
        let low = cut.saturating_sub(params.snap_chars as usize).max(self.floor);
        let cut = match text[low..cut].iter().rposition(|&c| c == '\n') {
            Some(i) => low + i + 1,
            None => cut,
        };
        match open_block(&text[..cut]) {
            Some(fence) => fence.max(self.floor),
            None => cut,
        }
    }
}

/// The generation as the model sees it: its tokens, the reasoning they
/// decoded to (as written and as scored), and the reasoning's length after
/// each token.
struct View {
    decoder: OutputDecoder,
    tokens: Vec<TokenId>,
    chars_after: Vec<usize>,
    scored: Normalized,
}

impl View {
    fn new(decoder: OutputDecoder, normalize: bool) -> Self {
        Self { decoder, tokens: Vec::new(), chars_after: Vec::new(), scored: Normalized::new(normalize) }
    }

    fn push(&mut self, token: TokenId) {
        for delta in self.decoder.push(&[token]) {
            if delta.channel == Channel::Reasoning {
                for c in delta.text.chars() {
                    self.scored.push(c);
                }
            }
        }
        self.tokens.push(token);
        self.chars_after.push(self.scored.raw_len());
    }

    /// The reasoning as written.
    fn text(&self) -> &[char] {
        self.scored.raw()
    }

    fn in_reasoning(&self) -> bool {
        self.decoder.channel() == Channel::Reasoning
    }

    /// The tokens whose text ends at or before character `cut`.
    fn tokens_before(&self, cut: usize) -> usize {
        self.chars_after.partition_point(|&chars| chars <= cut)
    }

    /// The characters of reasoning the first `tokens` tokens decoded to, as
    /// the model reads them: a line break the decoder still holds at the
    /// start counts for nothing.
    fn chars_through(&self, tokens: usize) -> usize {
        tokens.checked_sub(1).map_or(0, |last| self.chars_after[last])
    }
}

/// What the guard needs from the engine and the template: cancel and submit
/// a request, decode and encode text. A trait so a test can stand in for the
/// model thread.
pub trait GuardOps: Send + Sync + 'static {
    /// Abort a request (fire-and-forget).
    fn cancel(&self, request: RequestId);
    /// Submit a continuation; `None` when the engine would not take it. One
    /// the engine refuses as full waits up to `wait` for room (`Duration::ZERO`,
    /// it is tried once); an implementation with its own admission queue
    /// waits there instead.
    fn submit(&self, input: RequestInput, wait: Duration) -> Pin<Box<dyn Future<Output = Option<(RequestId, EventStream)>> + Send + '_>>;
    /// A fresh channel-splitting decoder, starting in the reasoning.
    fn decoder(&self) -> OutputDecoder;
    /// `text` as tokens of the loaded tokenizer.
    fn encode(&self, text: &str) -> Option<Vec<TokenId>>;
}

/// [`GuardOps`] on the live engine and template.
pub struct EngineOps {
    engine: Engine,
    class: RequestClass,
    template: Arc<dyn TemplateProvider>,
}

impl EngineOps {
    pub fn new(engine: Engine, class: RequestClass, template: Arc<dyn TemplateProvider>) -> Self {
        Self { engine, class, template }
    }

    /// The engine a continuation is submitted to.
    pub(crate) fn engine(&self) -> &Engine {
        &self.engine
    }

    /// The class a continuation is submitted in: the request's own.
    pub(crate) fn class(&self) -> RequestClass {
        self.class
    }
}

/// A continuation the engine refused as `Full` is tried again when a
/// request leaves, or after this long, within its `submit_wait`.
const SUBMIT_RETRY_EVERY: Duration = Duration::from_secs(5);

impl GuardOps for EngineOps {
    fn cancel(&self, request: RequestId) {
        self.engine.cancel(request);
    }

    fn submit(&self, input: RequestInput, wait: Duration) -> Pin<Box<dyn Future<Output = Option<(RequestId, EventStream)>> + Send + '_>> {
        Box::pin(async move {
            let deadline = tokio::time::Instant::now() + wait;
            loop {
                // Marked seen before the attempt: a request that leaves
                // between the refusal and the wait still wakes it.
                let mut freed = self.engine.freed();
                freed.borrow_and_update();
                let error = match self.engine.submit_with_notes(input.clone(), self.class, RequestNotes::default()).await {
                    Ok(pair) => return Some(pair),
                    Err(SubmitError::Full) => {
                        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
                        if left.is_zero() {
                            SubmitError::Full
                        } else if let Ok(Err(_)) = tokio::time::timeout(left.min(SUBMIT_RETRY_EVERY), freed.changed()).await {
                            // A model thread that is gone frees nothing.
                            SubmitError::Full
                        } else {
                            continue;
                        }
                    }
                    Err(error) => error,
                };
                tracing::warn!(name: "ignis.loop_guard.submit_refused", error = ?error, wait_secs = wait.as_secs(), "the continuation was refused");
                return None;
            }
        })
    }

    fn decoder(&self) -> OutputDecoder {
        OutputDecoder::new(self.template.token_decoder(), true)
    }

    fn encode(&self, text: &str) -> Option<Vec<TokenId>> {
        self.template.encode_literal(text)
    }
}

/// What the guard's rewinds took back, shared with the handler that builds
/// the usage. The guard adds to it before it sends the `Done` that ends the
/// stream.
#[derive(Debug, Default)]
pub struct RewoundTokens {
    taken: AtomicU32,
    unsent: AtomicU32,
}

impl RewoundTokens {
    /// A count with `taken` tokens taken back, `unsent` of them never sent.
    pub fn new(taken: u32, unsent: u32) -> Self {
        Self { taken: AtomicU32::new(taken), unsent: AtomicU32::new(unsent) }
    }

    /// Every generated token a rewind took back, whether the client received
    /// it or not: `usage.completion_tokens_details.rewound_tokens`. Not the
    /// tags' own tokens, nor the nudges. With no holdback, the sum of the
    /// tags' `N TOKENS`.
    pub fn taken(&self) -> u32 {
        self.taken.load(Ordering::Relaxed)
    }

    /// Of [`RewoundTokens::taken`], those the holdback never sent: generated,
    /// so `completion_tokens` counts them beside the tokens the stream
    /// carried.
    pub fn unsent(&self) -> u32 {
        self.unsent.load(Ordering::Relaxed)
    }

    /// What the usage reports, for a count of `streamed` generated tokens
    /// (all of them, or the reasoning's) the stream carried: that count
    /// with the [`RewoundTokens::unsent`] ones, which were generated in the
    /// reasoning all the same; and `rewound_tokens`, `None` (the field
    /// absent) when nothing was taken back.
    pub fn usage(&self, streamed: u32) -> (u32, Option<u32>) {
        let taken = self.taken();
        (streamed.saturating_add(self.unsent()), (taken > 0).then_some(taken))
    }
}

/// [`RewoundTokens`], shared.
pub type Rewound = Arc<RewoundTokens>;

/// The engine request generating for a guarded request now: the one first
/// submitted, then each continuation in turn. Shared with what cancels the
/// request by its response (`api::CancelOnDrop`): after a rewind the first
/// one is gone, and only the continuation is left to stop. Also how that
/// consumer ends the request before its own end ([`Running::stop`]).
#[derive(Debug, Clone)]
pub struct Running(Arc<RunningState>);

#[derive(Debug)]
struct RunningState {
    request: AtomicU64,
    /// A guard watches the request: it may hold back what the client is owed.
    guarded: AtomicBool,
    /// Set by [`Running::stop`], for good.
    stop: watch::Sender<bool>,
}

impl Running {
    /// `request`, until a rewind replaces it.
    pub fn new(request: RequestId) -> Self {
        Self(Arc::new(RunningState { request: AtomicU64::new(request), guarded: AtomicBool::new(false), stop: watch::Sender::new(false) }))
    }

    /// The engine request generating now.
    pub fn get(&self) -> RequestId {
        self.0.request.load(Ordering::Relaxed)
    }

    fn set(&self, request: RequestId) {
        self.0.request.store(request, Ordering::Relaxed);
    }

    /// End the request before its own end, for a client's cancel or a
    /// timeout: the guard cancels the engine request generating now, sends
    /// the client everything it held back, and ends its stream, which the
    /// caller reads to the end ([`held_back`]). False when no guard watches
    /// the request: nothing is held, and its stream ends with the engine's.
    pub fn stop(&self) -> bool {
        self.0.stop.send_replace(true);
        self.0.guarded.load(Ordering::Relaxed)
    }

    /// Once [`Running::stop`] was called.
    async fn stopped(&self) {
        let _ = self.0.stop.subscribe().wait_for(|&stop| stop).await;
    }
}

/// What the client of a request cut short is still owed (ADR 0052:
/// everything held back flushes on a cancel and a timeout): [`Running::stop`],
/// then every token left on `stream`, the guard's, to its end. None when no
/// guard watches the request.
pub async fn held_back(running: &Running, stream: &mut EventStream) -> Vec<TokenId> {
    let mut tokens = Vec::new();
    if running.stop() {
        while let Some(event) = stream.recv().await {
            if let SchedEvent::Token { token, .. } = event {
                tokens.push(token);
            }
        }
    }
    tokens
}

/// The loop guard on one request (chat completions, `/v1/responses` over
/// HTTP and the WebSocket), taken before the request is submitted, since
/// every continuation is built from the request as submitted.
pub struct Guarding {
    watched: Option<(LoopGuardParams, RequestInput)>,
    rewound: Rewound,
    metrics: Option<Arc<Metrics>>,
}

impl Guarding {
    /// The guard on `input` under `params`, the live `loop_guard` knobs,
    /// when it watches it ([`LoopGuardParams::watches`]), counting what it
    /// does on the server's `metrics` when they are on (GitHub #330).
    pub fn new(params: LoopGuardParams, input: &RequestInput, starts_in_reasoning: bool, metrics: Option<Arc<Metrics>>) -> Self {
        let watched = params.watches(input, starts_in_reasoning).then(|| (params, input.clone()));
        Self { watched, rewound: Rewound::default(), metrics }
    }

    /// What the guard's rewinds take back, for the usage.
    pub fn rewound(&self) -> Rewound {
        self.rewound.clone()
    }

    /// `stream`, the engine's for `request`, behind the guard when it
    /// watches the request, continuations submitted through `ops`; and the
    /// engine request generating for the client, which a stop sequence, a
    /// cancel by response or a timeout ends.
    pub fn guard(self, ops: impl GuardOps, request: RequestId, stream: EventStream) -> (EventStream, Running) {
        let running = Running::new(request);
        let stream = match self.watched {
            Some((params, input)) => guard_counted(params, ops, input, running.clone(), stream, self.rewound, self.metrics),
            None => stream,
        };
        (stream, running)
    }
}

/// Put the guard between `stream` (the engine's, for `request`, submitted
/// from `input`) and its consumer: the stream returned. `Off` returns
/// `stream` itself.
pub fn guard(params: LoopGuardParams, ops: impl GuardOps, input: RequestInput, request: RequestId, stream: EventStream) -> EventStream {
    guard_counted(params, ops, input, Running::new(request), stream, Rewound::default(), None)
}

/// [`guard`] on the request `running` names, adding the tokens each rewind
/// takes back to `rewound`, keeping `running` on the engine request
/// generating now, and counting what the guard does on `metrics` (GitHub
/// #330).
pub fn guard_counted(
    params: LoopGuardParams,
    ops: impl GuardOps,
    input: RequestInput,
    running: Running,
    stream: EventStream,
    rewound: Rewound,
    metrics: Option<Arc<Metrics>>,
) -> EventStream {
    if params.mode == LoopGuardMode::Off {
        return stream;
    }
    running.0.guarded.store(true, Ordering::Relaxed);
    let (tx, rx) = unbounded_channel();
    tokio::spawn(run(params, ops, input, running, stream, tx, rewound, metrics));
    rx
}

/// The reasoning `tokens` decode to as a client that received them shows
/// it once any text follows: what the decoder published and what it still
/// holds (a newline run, a partial `</think>`). Not what
/// [`OutputDecoder::finish`] gives, which drops a partial marker.
fn shown(mut decoder: OutputDecoder, tokens: &[TokenId]) -> String {
    let mut text: String =
        decoder.push(tokens).into_iter().filter(|delta| delta.channel == Channel::Reasoning).map(|delta| delta.text).collect();
    if decoder.channel() == Channel::Reasoning {
        text.push_str(decoder.pending());
    }
    text
}

/// One guarded request's state.
struct Guard<O: GuardOps> {
    params: LoopGuardParams,
    ops: O,
    tx: UnboundedSender<SchedEvent>,
    /// The request as submitted: every continuation is built from it.
    input: RequestInput,
    /// The client's request id: every event it receives carries it.
    original: RequestId,
    /// The engine request generating now, shared with what cancels by
    /// response.
    current: Running,
    view: View,
    watch: Watch,
    /// The guard may rewind this request ([`LoopGuardParams::rewinds`]);
    /// else it only logs.
    rewindable: bool,
    /// The nudges given so far.
    rewinds: u32,
    /// The guard closed the reasoning: nothing more to watch.
    closed: bool,
    /// The characters of reasoning the client is kept behind the view:
    /// `holdback_chars` on a request the guard may rewind, else 0.
    holdback: usize,
    /// The view's tokens the client has been sent, from the first: the rest
    /// are held back.
    sent: usize,
    /// Token events sent to the client.
    forwarded: u32,
    /// The client's token count when the current continuation's first token
    /// arrives: what its `Done` indices are offset by.
    offset: u32,
    rewound: Rewound,
    /// Where the guard counts what it does (GitHub #330), until
    /// [`Guard::finish`] counts the request itself: whether it has looped
    /// yet, and the lowest of its scores so far, across rewinds.
    metrics: Option<Arc<Metrics>>,
    looped: bool,
    lowest: Option<f64>,
}

/// Every way the guard ends without a `Done` (the client left, the request
/// was stopped, the engine's stream ended) still counts the request.
impl<O: GuardOps> Drop for Guard<O> {
    fn drop(&mut self) {
        self.finish();
    }
}

/// A continuation, ready to go: the tag the client receives before it, if
/// the cut took back anything it had received.
struct Continuation {
    request: RequestId,
    stream: EventStream,
    tag: Vec<TokenId>,
}

impl<O: GuardOps> Guard<O> {
    /// After each token: score the window, and act on a loop.
    async fn after_token(&mut self) -> Option<Continuation> {
        if self.closed || !self.view.in_reasoning() {
            return None;
        }
        let scores = self.watch.history.len();
        let found = self.watch.check(&self.view.scored, &self.params);
        if let (Some(_), Some(scored)) = (&self.metrics, self.watch.history.get(scores)) {
            self.lowest = Some(self.lowest.map_or(scored.ratio, |lowest| lowest.min(scored.ratio)));
        }
        let found = found?;
        let at = self.view.text().len();
        // What a rewind keeps: the whole tokens before the cut, counted in
        // the reasoning the model reads.
        let cut = self.watch.cut(self.view.text(), &found, &self.params);
        let kept = self.view.tokens_before(cut);
        let kept_chars = self.view.chars_through(kept);
        let (score, repeat) = (found.score(), found.repeat());
        if kept_chars < self.params.min_kept_chars as usize {
            tracing::info!(
                name: "ignis.loop_guard.too_little_kept",
                request = self.original, engine_request = self.current.get(), at_chars = at, signal = found.signal(),
                ratio = score.map(|s| s.ratio), raw_ratio = score.map(|s| s.raw_ratio), period_chars = repeat.map(|r| r.period),
                copies = repeat.map(|r| r.copies), window_start_chars = found.start(), cut_chars = cut, kept_chars,
                "loop with too little reasoning before it to keep: log only"
            );
            self.count(LoopOutcome::TooLittleKept);
            // The watch goes on: a clean window later gives a later loop
            // something to keep, and what found this one fires no more
            // until it finds nothing ([`Watch::hold`]).
            self.watch.hold();
            return None;
        }
        if !self.rewindable {
            tracing::info!(
                name: "ignis.loop_guard.detected",
                request = self.original, engine_request = self.current.get(), at_chars = at, signal = found.signal(),
                ratio = score.map(|s| s.ratio), raw_ratio = score.map(|s| s.raw_ratio), period_chars = repeat.map(|r| r.period),
                copies = repeat.map(|r| r.copies), window_start_chars = found.start(),
                cut_chars = cut,
                "loop in the reasoning (log only)"
            );
            self.count(LoopOutcome::Detected);
            self.watch = Watch::from(at, &self.params);
            return None;
        }
        self.rewind(at, found, kept).await
    }

    async fn rewind(&mut self, at: usize, found: Found, kept: usize) -> Option<Continuation> {
        let (score, repeat) = (found.score(), found.repeat());
        let generated = self.view.tokens.len();
        // The kept text and what the client received, as the client shows
        // them: a decoder holds back a trailing line break until the text
        // after it arrives, and the tag is that text.
        let kept_text = shown(self.ops.decoder(), &self.view.tokens[..kept]);
        let kept_chars = kept_text.chars().count();
        let struck = (self.sent > kept)
            .then(|| shown(self.ops.decoder(), &self.view.tokens[..self.sent]).chars().skip(kept_chars).collect::<String>());
        let close = self.rewinds >= self.params.retries;
        let full_sentence: Cow<str> =
            if close { Cow::Borrowed(crate::thinking::THINKING_CLOSE_TEXT) } else { Cow::Owned(self.params.nudge_text(self.rewinds)) };
        let sentence = sentence_after(&kept_text, &full_sentence);
        let encoded_tag = match &struck {
            Some(struck) => self.ops.encode(&tag(self.sent - kept, struck)),
            None => Some(Vec::new()),
        };
        let (Some(nudge), Some(encoded_tag)) = (self.ops.encode(sentence), encoded_tag) else {
            tracing::warn!(name: "ignis.loop_guard.no_tokenizer", request = self.original, "cannot encode the nudge: log only");
            self.count(LoopOutcome::NoTokenizer);
            self.watch = Watch::from(at, &self.params);
            return None;
        };
        let used = (kept + nudge.len()) as u32;
        let mut next = self.input.clone();
        if let Some(max_tokens) = next.params.max_tokens {
            let left = max_tokens.saturating_sub(used);
            if left < self.params.min_continuation_tokens {
                tracing::info!(name: "ignis.loop_guard.no_room", request = self.original, at_chars = at, left, "loop, no room left to rewind");
                self.count(LoopOutcome::NoRoom);
                self.watch = Watch::from(at, &self.params);
                return None;
            }
            next.params.max_tokens = Some(left);
        }
        let budget = match self.input.params.max_tokens {
            Some(generation) => ignis_core::thinking_budget::effective(self.input.params.thinking_budget, generation),
            None => self.input.params.thinking_budget,
        };
        next.params.thinking_budget = if close { None } else { budget.map(|b| b.saturating_sub(used)) };
        next.params.starts_in_reasoning = !close;
        next.params.seed = self.input.params.seed.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(u64::from(self.rewinds) + 1));
        next.tokens.extend_from_slice(&self.view.tokens[..kept]);
        next.tokens.extend_from_slice(&nudge);

        let cancelled = self.current.get();
        self.ops.cancel(cancelled);
        // A consumer that leaves while the continuation waits for room takes
        // it out of the wait: a submission dropped before its answer is
        // cancelled (`Engine::submit_with_notes`), a queue ticket leaves.
        let (request, stream) = tokio::select! {
            submitted = self.ops.submit(next, Duration::from_secs(u64::from(self.params.submit_wait_secs))) => match submitted {
                Some(submitted) => submitted,
                // The ops logged it (`submit_refused`).
                None => {
                    self.count(LoopOutcome::SubmitRefused);
                    return None;
                }
            },
            () = self.tx.closed() => return None,
            // Ended before the continuation was admitted: the client is owed
            // the kept reasoning, never the loop.
            () = self.current.stopped() => {
                self.view = self.view_of(&self.view.tokens[..kept]);
                self.sent = self.sent.min(kept);
                return None;
            }
        };
        self.current.set(request);
        tracing::info!(
            name: "ignis.loop_guard.rewound",
            request = self.original, cancelled, continuation = request,
            at_chars = at, signal = found.signal(), ratio = score.map(|s| s.ratio), raw_ratio = score.map(|s| s.raw_ratio),
            period_chars = repeat.map(|r| r.period), copies = repeat.map(|r| r.copies), window_start_chars = found.start(), cut_chars = kept_chars,
            rewound_tokens = generated - kept, rewound_chars = at - kept_chars,
            struck_chars = struck.as_ref().map_or(0, |struck| struck.chars().count()), held_back = struck.is_none(),
            attempt = self.rewinds + 1, closed = close,
            "loop in the reasoning: rewound"
        );
        self.count(if close { LoopOutcome::Closed } else { LoopOutcome::Rewound });
        let taken = (generated - kept) as u32;
        if let Some(metrics) = &self.metrics {
            metrics.record_loop_guard_rewound_tokens(taken);
        }

        self.rewound.taken.fetch_add(taken, Ordering::Relaxed);
        self.rewound.unsent.fetch_add((generated - self.sent.max(kept)) as u32, Ordering::Relaxed);

        // The model's view: the kept tokens, then the nudge. The client has
        // its first `sent` (all of them, when the tag goes out), and gets
        // the rest as the holdback lets them go.
        self.view = self.view_of(&[&self.view.tokens[..kept], &nudge[..]].concat());
        self.sent = self.sent.min(kept);
        self.watch = Watch::from(self.view.text().len(), &self.params);
        self.rewinds += 1;
        self.closed = close;
        Some(Continuation { request, stream, tag: encoded_tag })
    }

    /// Count a loop the guard saw by what it did about it, next to its log
    /// line (GitHub #330).
    fn count(&mut self, outcome: LoopOutcome) {
        if let Some(metrics) = &self.metrics {
            metrics.record_loop_guard_loop(outcome);
        }
        self.looped = true;
    }

    /// Count the request, once: watched, looped when it did, and its lowest
    /// score into the margin histogram, all three together so a window
    /// reads them in step. Before its `Done` goes to the client, else when
    /// the guard ends without one.
    fn finish(&mut self) {
        let Some(metrics) = self.metrics.take() else { return };
        metrics.record_loop_guard_watched();
        if self.looped {
            metrics.record_loop_guard_looped();
        }
        if let Some(lowest) = self.lowest {
            metrics.observe_loop_guard_min_ratio(lowest);
        }
    }

    /// The model's view of `tokens`, its generation so far.
    fn view_of(&self, tokens: &[TokenId]) -> View {
        let mut view = View::new(self.ops.decoder(), self.params.normalize);
        tokens.iter().for_each(|&token| view.push(token));
        view
    }

    fn send_token(&mut self, token: TokenId) -> bool {
        self.forwarded += 1;
        self.tx.send(SchedEvent::Token { request: self.original, token }).is_ok()
    }

    /// Send the client every token the holdback no longer covers: inside the
    /// reasoning, each whose text ends at least `holdback` characters before
    /// the view's; past it, every one. With no holdback, every token at once.
    fn release(&mut self) -> bool {
        let end = self.view.text().len();
        let all = !self.view.in_reasoning();
        while self.sent < self.view.tokens.len() && (all || self.view.chars_after[self.sent] + self.holdback <= end) {
            let token = self.view.tokens[self.sent];
            self.sent += 1;
            if !self.send_token(token) {
                return false;
            }
        }
        true
    }

    /// Send the client everything held back: at the stream's end, and when
    /// the request is stopped.
    fn flush(&mut self) -> bool {
        while self.sent < self.view.tokens.len() {
            let token = self.view.tokens[self.sent];
            self.sent += 1;
            if !self.send_token(token) {
                return false;
            }
        }
        true
    }

    /// The view's tokens the client has not received yet.
    fn held(&self) -> u32 {
        (self.view.tokens.len() - self.sent) as u32
    }
}

#[allow(clippy::too_many_arguments)]
async fn run<O: GuardOps>(
    params: LoopGuardParams,
    ops: O,
    input: RequestInput,
    running: Running,
    mut stream: EventStream,
    tx: UnboundedSender<SchedEvent>,
    rewound: Rewound,
    metrics: Option<Arc<Metrics>>,
) {
    let view = View::new(ops.decoder(), params.normalize);
    let rewindable = params.rewinds(&input);
    let watch = Watch::from(0, &params);
    let holdback = if rewindable { params.holdback_chars as usize } else { 0 };
    let mut guard = Guard {
        params,
        ops,
        tx,
        input,
        original: running.get(),
        current: running,
        view,
        watch,
        rewindable,
        rewinds: 0,
        closed: false,
        holdback,
        sent: 0,
        forwarded: 0,
        offset: 0,
        rewound,
        metrics,
        looped: false,
        lowest: None,
    };
    // Once a continuation runs, only its tokens and its end reach the
    // client: its admission and prompt-reuse events describe a request the
    // client never made.
    let mut continued = false;
    loop {
        // A consumer that left (cancelled by its response, or gone) stops
        // the engine request generating now, also when the holdback has
        // nothing to send it that would fail. One that ends the request
        // ([`Running::stop`]) is sent what was held back first.
        let event = tokio::select! {
            biased;
            () = guard.tx.closed() => {
                guard.ops.cancel(guard.current.get());
                return;
            }
            () = guard.current.stopped() => {
                guard.ops.cancel(guard.current.get());
                guard.flush();
                return;
            }
            event = stream.recv() => event,
        };
        let Some(mut event) = event else { break };
        match event {
            SchedEvent::Token { token, .. } => {
                guard.view.push(token);
                if !guard.release() {
                    guard.ops.cancel(guard.current.get());
                    return;
                }
                if let Some(continuation) = guard.after_token().await {
                    for token in continuation.tag {
                        if !guard.send_token(token) {
                            guard.ops.cancel(continuation.request);
                            return;
                        }
                    }
                    if !guard.release() {
                        guard.ops.cancel(continuation.request);
                        return;
                    }
                    // What the client still waits for comes before the
                    // continuation's first token.
                    guard.offset = guard.forwarded + guard.held();
                    stream = continuation.stream;
                    continued = true;
                }
            }
            SchedEvent::Done { ref mut request, ref mut tokens, ref mut thinking, ref mut reasoning_redirected_at, .. } => {
                if !guard.flush() {
                    return;
                }
                *request = guard.original;
                if continued {
                    let offset = guard.offset;
                    *tokens = guard.forwarded;
                    if let Some(outcome) = thinking {
                        outcome.forced_at = outcome.forced_at.map(|at| at + offset);
                        outcome.closed_at = outcome.closed_at.map(|at| at + offset);
                    }
                    *reasoning_redirected_at = reasoning_redirected_at.map(|at| at + offset);
                }
                guard.finish();
                let _ = guard.tx.send(event);
                return;
            }
            other if !continued => {
                if guard.tx.send(other).is_err() {
                    guard.ops.cancel(guard.current.get());
                    return;
                }
            }
            _ => {}
        }
    }
    // The engine's stream ended with no `Done`: what was held back still
    // reaches the client.
    guard.flush();
}

#[cfg(test)]
#[path = "../tests/support/loop_text.rs"]
mod loop_text;

#[cfg(test)]
mod tests {
    use super::loop_text::{looping, prose, Source};
    use super::*;
    use crate::decoder::TokenDecoder;
    use ignis_core::{DecodeParams, FinishReason};
    use std::sync::Mutex;

    /// One token per character: its code point. A token at [`NEWLINE_AFTER`]
    /// and over is two: a character, then a line break.
    struct CharDecoder;
    impl TokenDecoder for CharDecoder {
        fn push(&mut self, token: TokenId) -> String {
            piece(token)
        }
        fn finish(&mut self) -> String {
            String::new()
        }
    }

    /// The token for `c` followed by a line break: what a real tokenizer's
    /// `.\n` is, a token whose text ends in a newline the decoder holds.
    const NEWLINE_AFTER: TokenId = 0x20_0000;

    fn piece(token: TokenId) -> String {
        match token.checked_sub(NEWLINE_AFTER) {
            Some(c) => char::from_u32(c).map(|c| format!("{c}\n")).unwrap_or_default(),
            None => char::from_u32(token).map(String::from).unwrap_or_default(),
        }
    }

    fn tokens_of(text: &str) -> Vec<TokenId> {
        text.chars().map(u32::from).collect()
    }

    fn text_of(tokens: &[TokenId]) -> String {
        tokens.iter().map(|&t| piece(t)).collect()
    }

    /// The reasoning a client shows for `tokens`: through the decoder the
    /// chat handler uses, to the end.
    fn client_reasoning(tokens: &[TokenId]) -> String {
        let mut decoder = OutputDecoder::new(Box::new(CharDecoder), true);
        let mut deltas = decoder.push(tokens);
        deltas.extend(decoder.finish());
        deltas.into_iter().filter(|d| d.channel == Channel::Reasoning).map(|d| d.text).collect()
    }

    /// A template loop whose slots are numbers, the shape of the loops the
    /// raw score missed ("the 1527 sack, the 1348 plague").
    fn counting(chars: usize) -> String {
        let mut out = String::new();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        while out.chars().count() < chars {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.push_str(&format!("the {} AD (the {}th year, page {}), ", 300 + state % 1700, state / 7 % 90 + 10, state / 11 % 9000));
        }
        out.chars().take(chars).collect()
    }

    /// A table of numbers, the layout of every line the same and every
    /// number new: a template once its digits fold, information as written.
    fn numeric_table(chars: usize) -> String {
        let mut out = String::new();
        let mut n: u64 = 1000;
        while out.chars().count() < chars {
            out.push_str(&format!("{n}^2 = {}, {n}^3 = {}\n", n * n, n * n * n));
            n += 7;
        }
        out.chars().take(chars).collect()
    }

    /// A code block's worth of lines that repeat, as test code does.
    fn code(chars: usize) -> String {
        "    assert_eq!(watch.check(&scored, &p), None);\n".chars().cycle().take(chars).collect()
    }

    fn params(mode: LoopGuardMode) -> LoopGuardParams {
        LoopGuardParams { mode, ..LoopGuardParams::default() }
    }

    /// The default nudges as a continuation writes them
    /// ([`LoopGuardParams::nudge_text`]).
    const NUDGES: [&str; 2] = [
        "\n\nWait, I'm starting to repeat myself. Let me stop and move on with what I already have.\n\n",
        "\n\nI keep repeating myself. I already have enough to write the answer, so I'll wrap up my reasoning now.\n\n",
    ];

    /// The first rewind writes `nudge`, every later one `nudge_again`, each
    /// trimmed and on a paragraph of its own.
    #[test]
    fn the_nudges_are_written_on_a_paragraph_of_their_own() {
        let p = LoopGuardParams::default();
        assert_eq!([p.nudge_text(0), p.nudge_text(1), p.nudge_text(5)], [NUDGES[0], NUDGES[1], NUDGES[1]]);
        let typed = LoopGuardParams { nudge: " Next step.\n".to_owned(), nudge_again: "Wrap up.".to_owned(), ..p };
        assert_eq!((typed.nudge_text(0).as_str(), typed.nudge_text(1).as_str()), ("\n\nNext step.\n\n", "\n\nWrap up.\n\n"));
    }

    fn raw(mode: LoopGuardMode) -> LoopGuardParams {
        LoopGuardParams { normalize: false, ..params(mode) }
    }

    /// `text` as the detector scores it.
    fn normalized(text: &str, normalize: bool) -> Normalized {
        let mut scored = Normalized::new(normalize);
        text.chars().for_each(|c| scored.push(c));
        scored
    }

    fn scored_text(text: &str) -> String {
        normalized(text, true).text().iter().collect()
    }

    /// Push `text` into `scored` up to raw character `n`.
    fn grow(scored: &mut Normalized, text: &[char], n: usize) {
        while scored.raw_len() < n {
            scored.push(text[scored.raw_len()]);
        }
    }

    /// The first loop the guard finds in `text`, looking after every
    /// character: the raw length it was found at, and its score.
    fn first_loop(text: &str, floor: usize, p: &LoopGuardParams) -> Option<(usize, Score)> {
        let mut scored = Normalized::new(p.normalize);
        let mut watch = Watch::from(floor, p);
        for c in text.chars() {
            scored.push(c);
            match watch.check(&scored, p) {
                Some(Found::Score(score)) => return Some((scored.raw_len(), score)),
                Some(found) => panic!("a repeat where a loop score was looked for: {found:?}"),
                None => {}
            }
        }
        None
    }

    /// The watch that found the first loop in `text`.
    fn watch_to_loop(text: &str, p: &LoopGuardParams) -> Watch {
        let mut scored = Normalized::new(p.normalize);
        let mut watch = Watch::from(0, p);
        for c in text.chars() {
            scored.push(c);
            if watch.check(&scored, p).is_some() {
                return watch;
            }
        }
        panic!("no loop in the text");
    }

    /// A watch from `floor` that took `checks`, each (window start, ratio),
    /// the last one the loop.
    fn watch_with(floor: usize, checks: &[(usize, f64)]) -> Watch {
        let history = checks.iter().map(|&(start, ratio)| Score { ratio, raw_ratio: ratio, start, end: start + 1000 }).collect();
        Watch { floor, next_check: 0, history, loop_run: 0, held: false, sentences: 0, repeat_held: false }
    }

    /// Text for the cut to read: no line break, no backtick.
    fn flat(chars: usize) -> Vec<char> {
        vec!['x'; chars]
    }

    /// A loop must score as one on `persist_checks` checks in a row: it is
    /// found that many checks later, and cut at the same place.
    #[test]
    fn a_loop_is_found_once_it_persists() {
        let text = prose(3000) + &looping(6000);
        let chars: Vec<char> = text.chars().collect();
        let once = LoopGuardParams { persist_checks: 1, ..params(LoopGuardMode::Log) };
        let four = LoopGuardParams { persist_checks: 4, ..params(LoopGuardMode::Log) };
        let (at_once, _) = first_loop(&text, 0, &once).expect("a loop at the first check");
        let (at_four, _) = first_loop(&text, 0, &four).expect("a loop that persists");
        assert_eq!(at_four - at_once, 3 * 250, "three checks later");
        assert_eq!(watch_to_loop(&text, &once).cut_point(&chars, &once), watch_to_loop(&text, &four).cut_point(&chars, &four));
    }

    /// A run of loop scores shorter than the persistence, the length of a
    /// list, is no loop; a score that is not one starts the count over.
    #[test]
    fn a_run_of_loop_scores_shorter_than_the_persistence_is_no_loop() {
        // The same prose comes back after each loop: the loop scores alone.
        let scores = LoopGuardParams { repeat_max_chars: 0, ..params(LoopGuardMode::Log) };
        let once = LoopGuardParams { persist_checks: 1, ..scores.clone() };
        let eight = LoopGuardParams { persist_checks: 8, ..scores };
        let short = prose(3000) + &looping(1200) + &prose(3000);
        assert!(first_loop(&short, 0, &once).is_some(), "a loop at the first check");
        assert_eq!(first_loop(&short, 0, &eight), None);
        let twice = prose(3000) + &looping(1200) + &prose(2000) + &looping(1200) + &prose(3000);
        assert_eq!(first_loop(&twice, 0, &eight), None, "the prose between starts the count over");
    }

    /// What a rewind keeps is counted in whole tokens, as the model reads
    /// them: line breaks the decoder holds at the start count for nothing.
    #[test]
    fn the_kept_reasoning_is_counted_as_the_model_reads_it() {
        let mut view = View::new(OutputDecoder::new(Box::new(CharDecoder), true), true);
        tokens_of("\n\nab").into_iter().for_each(|token| view.push(token));
        assert_eq!(view.tokens_before(0), 2, "the held line breaks end at character 0");
        assert_eq!((view.chars_through(0), view.chars_through(2), view.chars_through(3)), (0, 0, 1));
    }

    /// A held run of loop scores fires no more; a score that is not a loop
    /// ends the hold, and the next run fires after its `persist_checks`.
    #[test]
    fn a_held_run_fires_no_more_until_a_score_that_is_not_a_loop() {
        let p = LoopGuardParams { persist_checks: 2, ..params(LoopGuardMode::Log) };
        let text: Vec<char> = (looping(3000) + &prose(3000) + &looping(3000)).chars().collect();
        let fires = |hold: bool| {
            let mut scored = Normalized::new(p.normalize);
            let mut watch = Watch::from(0, &p);
            let mut fires = Vec::new();
            for (i, &c) in text.iter().enumerate() {
                scored.push(c);
                if watch.check(&scored, &p).is_some() {
                    fires.push(i + 1);
                    if hold {
                        watch.hold();
                    }
                }
            }
            fires
        };
        let held = fires(true);
        assert_eq!(held.len(), 2, "one per run of loop scores: {held:?}");
        assert!(held[0] <= 3000 && held[1] > 6000, "{held:?}");
        assert!(fires(false).len() > 4, "unheld, every loop score from the second on fires");
    }

    #[test]
    fn prose_is_not_a_loop_and_a_loop_is() {
        let text = prose(3000);
        assert!(compress_ratio(&text) > 0.2, "prose scored {}", compress_ratio(&text));
        assert!(compress_ratio(&looping(3000)) < 0.05, "loop scored {}", compress_ratio(&looping(3000)));
        for p in [params(LoopGuardMode::Log), raw(LoopGuardMode::Log)] {
            assert_eq!(first_loop(&prose(8000), 0, &p), None, "prose flagged with {p:?}");
        }
    }

    /// The step is `check_every`: 250 by default.
    #[test]
    fn the_window_is_scored_only_once_it_is_full_and_every_step_after() {
        for step in [DEFAULT_CHECK_EVERY_CHARS as usize, 500] {
            let p = LoopGuardParams { check_every_chars: step as u32, persist_checks: 1, ..params(LoopGuardMode::Log) };
            let text: Vec<char> = looping(2000).chars().collect();
            let mut scored = Normalized::new(p.normalize);
            let mut watch = Watch::from(0, &p);
            grow(&mut scored, &text, 999);
            assert_eq!(watch.check(&scored, &p), None, "under one window");
            grow(&mut scored, &text, 1000);
            assert!(watch.check(&scored, &p).is_some());
            grow(&mut scored, &text, 1000 + step - 1);
            assert_eq!(watch.check(&scored, &p), None, "within the step of {step}");
            grow(&mut scored, &text, 1000 + step);
            assert!(watch.check(&scored, &p).is_some(), "a step of {step}");
        }
    }

    /// Code is not scored text: the first score waits for a whole window
    /// of what is, however many raw characters that takes.
    #[test]
    fn the_first_score_waits_for_a_whole_window_of_scored_text() {
        let p = LoopGuardParams { persist_checks: 1, ..params(LoopGuardMode::Log) };
        let block = format!("```rust\n{}```", code(2000));
        let text = block.clone() + &looping(2000);
        let (at, score) = first_loop(&text, 0, &p).expect("the loop after the block");
        let block_chars = block.chars().count();
        assert_eq!(at, block_chars + 1000, "scored as soon as 1,000 characters after the block are");
        assert_eq!(score.start, block_chars, "the window starts where the block ends");
    }

    #[test]
    fn nothing_before_the_floor_is_scored() {
        let p = LoopGuardParams { persist_checks: 1, ..params(LoopGuardMode::Log) };
        let text: Vec<char> = (prose(1000) + &looping(3000)).chars().collect();
        let mut scored = Normalized::new(p.normalize);
        let mut watch = Watch::from(1000, &p);
        grow(&mut scored, &text, 1999);
        assert_eq!(watch.check(&scored, &p), None, "the window would start before the floor");
        grow(&mut scored, &text, 2000);
        assert_eq!(watch.check(&scored, &p).map(|found| found.start()), Some(1000));
    }

    /// The floor is raw: a window of scored text after it is a window
    /// whose first character comes after it, not `window` raw characters.
    #[test]
    fn a_window_reaching_back_past_the_floor_through_code_is_not_scored() {
        let p = LoopGuardParams { persist_checks: 1, ..params(LoopGuardMode::Log) };
        let after = format!("```\n{}```", code(500)) + &looping(3000);
        let text = prose(1000) + &after;
        let (at, score) = first_loop(&text, 1000, &p).expect("found after the floor");
        let block_chars = after.chars().count() - 3000;
        assert_eq!(score.start, 1000 + block_chars);
        assert_eq!(at, 1000 + block_chars + 1000);
    }

    /// On a stream: the last window before the loop that scored clean is
    /// the reasoning just before it, and the cut is at its start or on the
    /// line break just before.
    #[test]
    fn the_cut_lands_before_the_loop_at_the_last_clean_windows_start() {
        let p = params(LoopGuardMode::Rewind);
        let text = prose(6000) + &looping(5000);
        let watch = watch_to_loop(&text, &p);
        let fire = *watch.history.last().expect("the loop's score");
        let clean = *watch.history.iter().rev().skip(1).find(|score| score.is_clean(&p)).expect("a clean window");
        assert!(clean.start < 6000 && clean.end >= 6000 && fire.end > clean.end, "clean {clean:?}, loop {fire:?}");
        let cut = watch.cut_point(&text.chars().collect::<Vec<_>>(), &p);
        assert!(cut <= clean.start && clean.start - cut <= p.snap_chars as usize, "cut {cut}, clean window from {}", clean.start);
    }

    #[test]
    fn the_cut_is_the_last_clean_windows_start_less_the_margin() {
        let p = params(LoopGuardMode::Rewind);
        let text = flat(4000);
        let watch = watch_with(0, &[(1000, 0.45), (1250, 0.41), (1500, 0.38), (1750, 0.30), (2000, 0.15)]);
        assert_eq!(watch.cut_point(&text, &p), 1250);
        assert_eq!(watch.cut_point(&text, &LoopGuardParams { margin_chars: 100, ..p.clone() }), 1150);
        assert_eq!(watch.cut_point(&text, &LoopGuardParams { clean_permille: 420, ..p.clone() }), 1000);
        let lenient = LoopGuardParams { ratio_permille: 500, clean_permille: 100, ..p };
        assert_eq!(watch_with(0, &[(1000, 0.45), (1250, 0.42)]).cut_point(&text, &lenient), 1000, "the loop's own score is never the clean one");
    }

    /// Within `snap` characters: 300 by default, 0 never.
    #[test]
    fn the_cut_moves_back_to_a_line_break_within_the_snap() {
        let p = params(LoopGuardMode::Rewind);
        let watch = watch_with(0, &[(1000, 0.45), (1250, 0.15)]);
        let mut text = flat(4000);
        text[650] = '\n';
        assert_eq!(watch.cut_point(&text, &p), 1000, "the line break is 350 characters back");
        assert_eq!(watch.cut_point(&text, &LoopGuardParams { snap_chars: 400, ..p.clone() }), 651, "within a snap of 400");
        text[750] = '\n';
        text[800] = '\n';
        assert_eq!(watch.cut_point(&text, &p), 801, "after the last line break before it");
        assert_eq!(watch.cut_point(&text, &LoopGuardParams { snap_chars: 0, ..p }), 1000, "a snap of 0 never moves it");
    }

    #[test]
    fn the_cut_never_crosses_the_floor() {
        let p = params(LoopGuardMode::Rewind);
        let mut text = flat(4000);
        text[1100] = '\n';
        let watch = watch_with(1200, &[(1300, 0.45), (1550, 0.15)]);
        assert_eq!(watch.cut_point(&text, &p), 1300, "the line break is before the floor");
        assert_eq!(watch.cut_point(&text, &LoopGuardParams { margin_chars: 500, ..p }), 1200, "the margin stops at the floor");
    }

    #[test]
    fn with_no_clean_window_since_the_floor_the_cut_is_the_floor() {
        let p = params(LoopGuardMode::Rewind);
        let mut text = flat(4000);
        text[600] = '\n';
        let watch = watch_with(700, &[(700, 0.39), (950, 0.25), (1200, 0.12)]);
        assert_eq!(watch.cut_point(&text, &p), 700);
    }

    /// A cut that would keep a code block open moves to just before its
    /// opening fence: the continuation is scored at once, not once the
    /// model closes the block.
    #[test]
    fn a_cut_inside_a_code_block_moves_before_its_fence() {
        let p = params(LoopGuardMode::Rewind);
        // A block from 900 (its fence) to 1411, a line break there.
        let text: Vec<char> =
            format!("{}```rust\n{}```\n{}", "x".repeat(900), code(500), "y".repeat(2000)).chars().collect();
        assert_eq!((&text[900..903], text[1411]), (&['`'; 3][..], '\n'));
        let after_the_block = watch_with(0, &[(1500, 0.45), (1750, 0.15)]);
        assert_eq!(after_the_block.cut_point(&text, &p), 1412, "a block closed before the cut stays");
        let on_its_line_break = watch_with(0, &[(1411, 0.45), (1661, 0.15)]);
        assert_eq!(on_its_line_break.cut_point(&text, &p), 900, "snapped into the block");
        let margin = LoopGuardParams { margin_chars: 300, ..p };
        assert_eq!(after_the_block.cut_point(&text, &margin), 900, "taken back into the block");
        let floor_in_the_block = watch_with(1000, &[(1500, 0.45), (1750, 0.15)]);
        assert_eq!(floor_in_the_block.cut_point(&text, &margin), 1000, "never before the floor");
    }

    /// The raw gate: a table of numbers is a loop to the normalized score
    /// alone, not as written; a counting loop is one both ways.
    #[test]
    fn a_numeric_table_is_not_a_loop_through_the_raw_gate_and_a_counting_loop_is() {
        let p = params(LoopGuardMode::Log);
        let ungated = LoopGuardParams { raw_gate_permille: 0, ..p.clone() };
        let table = prose(3000) + &numeric_table(4000);
        let (at, score) = first_loop(&table, 0, &ungated).expect("ungated, the folded table is a loop");
        assert!(at > 3000 && score.raw_ratio >= 0.30, "flagged at {at}: {score:?}");
        assert_eq!(first_loop(&table, 0, &p), None, "the gate lets the table through");
        let counted = prose(3000) + &counting(4000);
        let (at, score) = first_loop(&counted, 0, &p).expect("a counting loop is one through the gate");
        assert!(at > 3000 && score.raw_ratio < 0.30, "flagged at {at}: {score:?}");
    }

    /// A template loop over numbers compresses like prose as written, and
    /// repeats verbatim once its digits fold.
    #[test]
    fn a_counting_loop_is_a_loop_only_normalized() {
        let loop_text = counting(1000);
        assert!(compress_ratio(&loop_text) > 0.215, "as written it must look like prose: {}", compress_ratio(&loop_text));
        let text = prose(3000) + &counting(4000);
        let (at, _) = first_loop(&text, 0, &params(LoopGuardMode::Log)).expect("normalized, a loop");
        assert!(at > 3000, "flagged at {at}, before the loop");
        assert_eq!(first_loop(&text, 0, &raw(LoopGuardMode::Log)), None, "as written, not one");
    }

    #[test]
    fn a_verbatim_loop_is_a_loop_either_way() {
        let text = prose(3000) + &looping(3000);
        for p in [params(LoopGuardMode::Log), raw(LoopGuardMode::Log)] {
            let (at, _) = first_loop(&text, 0, &p).expect("a loop");
            assert!(at > 3000, "flagged at {at}, before the loop, with {p:?}");
        }
    }

    /// A reasoning that opens with `lead` characters of `source`'s
    /// sentences (none: it opens in the loop), then loops on a block of
    /// whole sentences about `period` characters long, `copies` paragraphs of
    /// it; with `varying`, each copy's middle sentence is its own. The text,
    /// and where each copy's text starts and ends, in characters.
    fn sentence_loop(source: Source, period: usize, lead: usize, copies: usize, varying: bool) -> (String, Vec<(usize, usize)>) {
        let pool = source.sentences(400);
        let mut next = pool.iter();
        let mut text = String::new();
        while text.chars().count() < lead {
            text.push_str(next.next().unwrap());
            text.push(' ');
        }
        if !text.is_empty() {
            text.pop();
            text.push_str("\n\n");
        }
        let mut block: Vec<&String> = Vec::new();
        while block.iter().map(|s| s.chars().count() + 1).sum::<usize>() < period {
            block.push(next.next().unwrap());
        }
        let mut spans = Vec::new();
        for _ in 0..copies {
            let mut sentences = block.clone();
            if varying {
                sentences[block.len() / 2] = next.next().unwrap();
            }
            let start = text.chars().count();
            text.push_str(&sentences.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" "));
            spans.push((start, text.chars().count()));
            text.push_str("\n\n");
        }
        (text, spans)
    }

    /// The first thing the watch finds in `text`, looking after every
    /// character: the raw length then, what it found, and the watch.
    fn first_found(text: &str, p: &LoopGuardParams) -> Option<(usize, Found, Watch)> {
        let mut scored = Normalized::new(p.normalize);
        let mut watch = Watch::from(0, p);
        for c in text.chars() {
            scored.push(c);
            if let Some(found) = watch.check(&scored, p) {
                return Some((scored.raw_len(), found, watch));
            }
        }
        None
    }

    /// `text` as the detector scores it, cut into its sentences: each one's
    /// scored text and whether it is prose.
    fn sentences_of(text: &str) -> Vec<(String, bool)> {
        let scored = normalized(text, true);
        let sentences = &scored.sentences;
        sentences.all.iter().map(|s| (scored.text()[s.span.0..s.span.1].iter().collect(), s.prose)).collect()
    }

    /// A loop whose block repeats back to back is a repeat by the time its
    /// third copy starts, whatever its period from 250 to 8,000 scored
    /// characters, in three sources (one Italian), verbatim or with one
    /// sentence of each copy its own, opening the reasoning or after 3,000
    /// characters of it; and its cut keeps the first copy, at the start of
    /// the second.
    #[test]
    fn a_loop_of_any_period_is_a_repeat_by_its_third_copy_and_cut_at_its_second() {
        let p = params(LoopGuardMode::Log);
        for source in Source::ALL {
            for period in [250, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000] {
                for varying in [false, true] {
                    for lead in [0, 3000] {
                        let (text, copies) = sentence_loop(source, period, lead, 3, varying);
                        let case = format!("{source:?}, period {period}, varying {varying}, lead {lead}");
                        let (at, found, watch) = first_found(&text, &p).unwrap_or_else(|| panic!("{case}: nothing found"));
                        let Found::Repeat(repeat) = found else { panic!("{case}: {found:?}") };
                        assert!(at > copies[1].1 && at <= copies[2].0 + 1, "{case}: found at {at}, copies at {copies:?}");
                        let cut = watch.cut(&text.chars().collect::<Vec<_>>(), &found, &p);
                        assert!(copies[0].1 <= cut && cut <= copies[1].0, "{case}: cut at {cut}, copies at {copies:?}");
                        assert!(repeat.period.abs_diff(copies[1].0 - copies[0].0) <= 100, "{case}: {repeat:?}, copies at {copies:?}");
                        assert!(repeat.copies >= 2.0 && repeat.first <= copies[0].0 + 1, "{case}: {repeat:?}");
                    }
                }
            }
        }
    }

    /// The compression score's window holds no two copies of a long block:
    /// only the repeat signal sees such a loop, and with it off the guard
    /// sees nothing (the issue's `A × 4`).
    #[test]
    fn a_long_block_loops_unseen_by_the_compression_score() {
        let (text, _) = sentence_loop(Source::Agent, 1900, 3000, 4, false);
        let off = LoopGuardParams { repeat_max_chars: 0, ..params(LoopGuardMode::Log) };
        assert_eq!(first_found(&text, &off).map(|(at, found, _)| (at, found)), None);
        assert!(matches!(first_found(&text, &params(LoopGuardMode::Log)), Some((_, Found::Repeat(_), _))));
    }

    /// A pasted log transcribed, twice to compare two runs.
    fn transcribed_log() -> String {
        let prose = Source::Agent.sentences(12);
        let run = |r: u32| {
            let mut out = String::new();
            let ts = |i: u32| format!("2026-11-26T21:{:02}:{:02}.{:03}Z", 51 + r * 3 + i / 60, (i * 7 + r * 13) % 60, (i * 389 + r * 71) % 1000);
            out.push_str(&format!("{} INFO  ignis.server: listening on 127.0.0.1:8000\n", ts(0)));
            out.push_str(&format!("{} INFO  ignis.runtime.load: model qwen3.8-27b ready in {}.{} s\n", ts(1), 41 + r, 2 + r));
            for i in 0..12 {
                out.push_str(&format!("{} INFO  ignis.request: request {} admitted with {} prompt tokens\n", ts(2 + i), 4100 + i + 100 * r, 812 + 37 * i));
                if i % 4 == 3 {
                    out.push_str(&format!("{} INFO  ignis.health: ok\n", ts(2 + i)));
                }
            }
            out.push_str(&format!("{} WARN  ignis.engine: the prefill of request {} failed, retrying.\n", ts(15), 4111 + 100 * r));
            out.push_str("thread 'model' panicked at crates/core/src/engine.rs:812:31:\n");
            out.push_str("called `Result::unwrap()` on an `Err` value: OutOfMemory\n");
            for frame in ["ignis_core::engine::Engine::step", "ignis_core::engine::Engine::run", "ignis_server::model::spawn::{{closure}}"] {
                out.push_str(&format!("   at {frame}\n"));
            }
            out
        };
        format!(
            "The user pasted the log of the failed run. Let me go through it line by line.\n{}{} {}\nNow the log of the run that worked, to compare.\n{}{} {}",
            run(0),
            prose[0],
            prose[1],
            run(1),
            prose[2],
            prose[3]
        )
    }

    /// A plan re-drafted after a review: steps reworded, one added.
    fn redrafted_plan() -> String {
        let first = [
            "Read the failing test and the module it covers, to see what the test expects.",
            "Add a field for the new limit to the config schema, with the measured default.",
            "Thread the field through the start config into the scheduler's admission check.",
            "Refuse a request over the limit with a clear error that names the field.",
            "Write a test that a request at the limit is admitted and one over it is not.",
            "Update the configuration page and the troubleshooting entry for the new error.",
            "Run the crate's tests, then the whole workspace on the CI branch.",
        ];
        let second = [
            "Read the failing test and the module it covers, to see what the test expects.",
            "Add a field for the new limit to the config schema, with the measured default.",
            "Read the field from the live config at each admission, so a patch applies at once.",
            "Refuse a request over the limit with a clear error that names the field.",
            "Log the refusal once per request, with the limit and the request's size.",
            "Write a test that a patched limit holds for the next request without a reload.",
            "Update the configuration page, the telemetry page and the troubleshooting entry.",
            "Run the crate's tests, then the whole workspace on the CI branch.",
        ];
        let steps = |steps: &[&str]| steps.iter().enumerate().map(|(i, s)| format!("{}. {s}\n", i + 1)).collect::<String>();
        format!(
            "Here is my plan for the change.\n{}The reviewer says the limit belongs in the live config, not the start config, so a patch \
             applies without a reload. They also want the refusal logged. Let me write the plan again with both.\n{}Now the first step.",
            steps(&first),
            steps(&second)
        )
    }

    /// A per-item checklist: each item its own first sentence, then the same
    /// two.
    fn checklist() -> String {
        let items = [
            ("core", "Its engine module gained the new admission check."),
            ("server", "Its handler now returns the refusal as a 400 error."),
            ("bench", "Its fixture sets the limit explicitly for every arm."),
            ("artifact", "Its loader does not read the field at all, which is right."),
            ("kernel", "Its build script is untouched, so only the tests matter."),
            ("web", "Its settings view shows the field beside the other limits."),
        ];
        let mut out = String::from("Every crate needs the same checks before the merge.\n");
        for (name, own) in items {
            out.push_str(&format!(
                "- The {name} crate. {own} Its tests pass with the default features and with every feature on. Mark it done in \
                 the ticket once both runs are green.\n"
            ));
        }
        out + "That is the list, so the next thing is the first crate."
    }

    /// A listing two agents returned, quoted twice to compare them.
    fn listing_twice() -> String {
        let files: String = ["Cargo.toml", "Cargo.lock", "README.md", "AGENTS.md", "GLOSSARY.md", "Makefile", "crates", "docs", "kernel", "web"]
            .iter()
            .chain(["tools", "scripts", "mk", "ci", "tests", ".github", ".githooks", ".gitignore", "rustfmt.toml", "LICENSE"].iter())
            .map(|f| format!("- {f}\n"))
            .collect();
        format!(
            "The two agents both listed the root of the checkout. Let me put the lists side by side.\nAgent one:\n{files}Agent two:\n{files}\
             They are the same, so both agents saw the same checkout."
        )
    }

    /// Repetition that is no loop: a pasted log transcribed twice to compare
    /// two runs, a plan re-drafted after a review, a per-item checklist and a
    /// listing quoted twice. Neither signal acts on any of them.
    #[test]
    fn legitimate_repetition_is_no_loop() {
        let p = params(LoopGuardMode::Log);
        for (name, text) in [("log", transcribed_log()), ("plan", redrafted_plan()), ("checklist", checklist()), ("listing", listing_twice())] {
            let text = Source::Agent.sentences(60).join(" ") + "\n\n" + &text;
            assert_eq!(first_found(&text, &p).map(|(at, found, _)| (at, found)), None, "{name}");
        }
    }

    /// A sentence ends at a line break, at a full-width stop, and at white
    /// space after a stop and its closers, not at a stop inside a word; one
    /// that ends in a stop is prose, a line that ends in none is material.
    #[test]
    fn sentences_end_at_a_stop_before_white_space_and_at_a_line_break() {
        let text = "Read engine.rs and 0.5 first. Then stop! He said \"done.\" Why?\n- an item\n## A heading\nfin。下一句。tail";
        let sentences: Vec<(String, bool)> = sentences_of(text);
        let expect = [
            ("Read engine.rs and 0.0 first.", true),
            (" Then stop!", true),
            (" He said \"done.\"", true),
            (" Why?\n", true),
            ("- an item\n", false),
            ("## A heading\n", false),
            ("fin。", true),
            ("下一句。", true),
        ];
        assert_eq!(sentences, expect.map(|(s, prose)| (s.to_owned(), prose)));
    }

    /// The repeat signal compares sentences as written: rows whose numbers
    /// differ repeat once the digits fold, and are no repeat.
    #[test]
    fn rows_whose_numbers_differ_are_no_repeat() {
        let p = params(LoopGuardMode::Log);
        let rows: String = (0..40).map(|n| format!("Row {n} holds the value {} and the checksum {}.\n", n * 37 + 11, n * 7919 % 1000)).collect();
        let lead = Source::History.sentences(40).join(" ");
        let mut scored = Normalized::new(true);
        for c in format!("{lead}\n{rows}").chars() {
            let sentences = scored.sentence_count();
            scored.push(c);
            if scored.sentence_count() > sentences {
                assert_eq!(scored.repeat(0, &p), None, "at {}", scored.raw_len());
            }
        }
        assert_eq!(scored.sentence_count(), 80, "every row a sentence");
    }

    /// Only prose counts towards `repeat_min`: a block of material lines
    /// repeated is no repeat, the same block with a paragraph of prose is.
    #[test]
    fn only_prose_counts_towards_the_least_a_copy_repeats() {
        let p = params(LoopGuardMode::Log);
        let lines: String = (0..12).map(|i| format!("- item {} of the listing\n", ["one", "two", "three", "four"][i % 4].repeat(1 + i / 4))).collect();
        let lead = Source::Agent.sentences(30).join(" ") + "\n";
        assert_eq!(first_found(&format!("{lead}{lines}{lines}{lines}"), &p).map(|(at, found, _)| (at, found)), None);
        let paragraph = Source::Agent.sentences(40)[35..].join(" ") + "\n";
        let block = format!("{lines}{paragraph}");
        assert!(matches!(first_found(&format!("{lead}{block}{block}{block}"), &p), Some((_, Found::Repeat(_), _))));
    }

    /// One short sentence over and over is the compression score's: the
    /// repeat signal leaves it, at every multiple of it that would reach
    /// `repeat_min`, and the loop score finds it.
    #[test]
    fn a_short_sentence_over_and_over_is_the_compression_scores() {
        let p = params(LoopGuardMode::Log);
        let text = prose(3000) + &"I'll write the tool call now. ".repeat(120);
        let (_, found, _) = first_found(&text, &p).expect("a loop");
        assert!(matches!(found, Found::Score(_)), "{found:?}");
    }

    /// Two short sentences in a row inside a long block hide nothing: the
    /// shorter period they repeat at is too short to decide.
    #[test]
    fn two_short_sentences_in_a_row_do_not_hide_a_long_repeat() {
        let p = params(LoopGuardMode::Log);
        let mut sentences = Source::History.sentences(30);
        sentences.insert(8, "Hmm.".to_owned());
        sentences.insert(8, "Hmm.".to_owned());
        let block = sentences[..20].join(" ") + "\n\n";
        let text = Source::Agent.sentences(20).join(" ") + "\n\n" + &block.repeat(3);
        let (_, found, _) = first_found(&text, &p).expect("the repeat");
        assert!(matches!(found, Found::Repeat(repeat) if repeat.period == block.chars().count()), "{found:?}");
    }

    /// A copy may have `repeat_varying` sentences the one before it does
    /// not, and no more.
    #[test]
    fn a_copy_may_differ_in_as_many_sentences_as_may_vary() {
        let pool = Source::Italian.sentences(80);
        let block = |k: usize| {
            let mut sentences = pool[..12].to_vec();
            sentences[3] = pool[20 + k].clone();
            sentences[8] = pool[40 + k].clone();
            sentences.join(" ") + "\n\n"
        };
        let text = pool[60..].join(" ") + "\n\n" + &block(0) + &block(1) + &block(2);
        let one = params(LoopGuardMode::Log);
        assert_eq!(first_found(&text, &one).map(|(at, found, _)| (at, found)), None, "two sentences vary in each copy");
        let two = LoopGuardParams { repeat_varying: 2, ..one.clone() };
        assert!(matches!(first_found(&text, &two), Some((_, Found::Repeat(_), _))));
        let verbatim = LoopGuardParams { repeat_varying: 0, ..one };
        let (same, _) = sentence_loop(Source::Agent, 600, 1000, 3, false);
        let (varying, _) = sentence_loop(Source::Agent, 600, 1000, 3, true);
        assert!(first_found(&same, &verbatim).is_some());
        assert_eq!(first_found(&varying, &verbatim).map(|(at, found, _)| (at, found)), None, "0: every copy verbatim");
    }

    /// `repeat_copies` copies back to back make a repeat; a copy longer than
    /// `repeat_max` is none, and 0 turns the signal off.
    #[test]
    fn the_copies_needed_and_the_longest_copy_are_limits() {
        let (text, copies) = sentence_loop(Source::History, 1200, 2000, 4, false);
        let three = LoopGuardParams { repeat_copies: 3, ..params(LoopGuardMode::Log) };
        let (at, _, _) = first_found(&text, &three).expect("three copies");
        assert!(at > copies[2].1 && at <= copies[3].0 + 1, "found at {at}, copies at {copies:?}");
        for max in [0, 1000] {
            let short = LoopGuardParams { repeat_max_chars: max, ..params(LoopGuardMode::Log) };
            assert_eq!(first_found(&text, &short).map(|(at, found, _)| (at, found)), None, "max {max}");
        }
    }

    /// Both copies a repeat is found from lie after the floor: the end of
    /// the last nudge, as for the compression score.
    #[test]
    fn a_repeat_is_found_only_after_the_floor() {
        let p = params(LoopGuardMode::Log);
        let (text, copies) = sentence_loop(Source::Agent, 1000, 2000, 4, false);
        let chars: Vec<char> = text.chars().collect();
        let mut scored = Normalized::new(true);
        let mut watch = Watch::from(copies[0].0 + 10, &p);
        let mut found = None;
        for (i, &c) in chars.iter().enumerate() {
            scored.push(c);
            if let Some(Found::Repeat(repeat)) = watch.check(&scored, &p) {
                found = Some((i + 1, repeat));
                break;
            }
        }
        // Two periods after the floor end in the third copy, not with the
        // second as they would from 0.
        let (at, repeat) = found.expect("the copies after the floor");
        assert!(at > copies[2].0 && repeat.first >= copies[0].0 + 10, "found at {at}: {repeat:?}, copies at {copies:?}");
    }

    /// A held repeat is acted on no more until a sentence ends none.
    #[test]
    fn a_held_repeat_fires_no_more_until_a_sentence_ends_none() {
        let p = params(LoopGuardMode::Log);
        let (looped, _) = sentence_loop(Source::Agent, 500, 0, 4, false);
        let text: Vec<char> = (looped.clone() + &Source::History.sentences(30).join(" ") + "\n\n" + &looped).chars().collect();
        let mut scored = Normalized::new(true);
        let mut watch = Watch::from(0, &p);
        let mut fires = Vec::new();
        for (i, &c) in text.iter().enumerate() {
            scored.push(c);
            if watch.check(&scored, &p).is_some() {
                fires.push(i + 1);
                watch.hold();
            }
        }
        let first = looped.chars().count();
        assert!(fires.len() == 2 && fires[0] < first && fires[1] > first, "{fires:?}");
    }

    /// The rewind of a repeat keeps the reasoning through the first copy and
    /// strikes from the second's start; the tag hashes what it struck.
    #[tokio::test]
    async fn a_repeat_is_rewound_to_the_end_of_its_first_copy() {
        let ops = ops_with(vec![(2, scripted(2, "Done.</think>\n\nOK.", true))]);
        let (reasoning, copies) = sentence_loop(Source::Agent, 2000, 3000, 3, true);
        let (text, _) =
            drain(guard(params(LoopGuardMode::Rewind), ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
        let continued = text_of(&ops.submitted.lock().unwrap()[0].tokens);
        let kept = continued.strip_prefix('P').unwrap().strip_suffix(NUDGES[0].trim_start_matches('\n')).expect("the nudge last");
        let chars: Vec<char> = reasoning.chars().collect();
        let kept_chars = kept.chars().count();
        assert!(copies[0].1 <= kept_chars && kept_chars <= copies[1].0, "kept {kept_chars}, copies at {copies:?}");
        assert_eq!(kept, chars[..kept_chars].iter().collect::<String>());
        let (m, struck) = first_struck(&client_reasoning(&tokens_of(&text)));
        assert_eq!(struck, chars[kept_chars..kept_chars + m].iter().collect::<String>(), "struck from the cut on");
        assert!(kept_chars + m > copies[1].1, "the whole second copy struck");
    }

    /// `repeat_min` is the least a copy repeats of prose: a loop whose copy
    /// holds less is no repeat.
    #[test]
    fn the_least_a_copy_repeats_is_a_limit() {
        let (text, copies) = sentence_loop(Source::History, 500, 2000, 3, false);
        let copy = copies[1].0 - copies[0].0;
        for (min, found) in [(100, true), (copy as u32 - 10, true), (copy as u32 + 10, false)] {
            let p = LoopGuardParams { repeat_min_chars: min, ..params(LoopGuardMode::Log) };
            assert_eq!(first_found(&text, &p).is_some(), found, "min {min}, copy {copy}");
        }
    }

    /// When the sentence that varies opens each copy, the copies are read
    /// from the sentence after it: the repeat is found one sentence into the
    /// third copy, and its cut keeps the second copy's varying sentence.
    #[test]
    fn a_repeat_whose_opening_sentence_varies_is_read_from_the_sentence_after_it() {
        let pool = Source::Agent.sentences(60);
        let copy = |k: usize| format!("{} {}", pool[40 + k], pool[..6].join(" "));
        let lead = pool[20..32].join(" ");
        let text = format!("{lead} {} {} {} {}", copy(0), copy(1), copy(2), pool[50]);
        let at = |s: &str| text.find(s).unwrap();
        let (found_at, found, watch) = first_found(&text, &params(LoopGuardMode::Log)).expect("the repeat");
        assert_eq!(found_at, at(&copy(2)) + pool[42].len() + 1, "found once the third copy's opening sentence ends");
        let cut = watch.cut(&text.chars().collect::<Vec<_>>(), &found, &params(LoopGuardMode::Log));
        assert_eq!(cut, at(&copy(1)) + pool[41].len(), "cut after the second copy's varying sentence");
    }

    /// A repeat's cut inside a code block moves before its fence, as a loop
    /// score's does: with `normalize` off, code is scored as written.
    #[test]
    fn a_repeats_cut_inside_a_code_block_moves_before_its_fence() {
        let p = params(LoopGuardMode::Rewind);
        let text: Vec<char> = format!("{}```\n{}", "x".repeat(900), "y".repeat(3000)).chars().collect();
        let repeat = Repeat { period: 1000, copies: 2.0, first: 1000, second: 2000, end: 3000 };
        assert_eq!(watch_with(0, &[]).cut(&text, &Found::Repeat(repeat), &p), 900);
        assert_eq!(watch_with(950, &[]).cut(&text, &Found::Repeat(repeat), &p), 950, "never before the floor");
        let closed: Vec<char> = format!("{}```\n```\n{}", "x".repeat(900), "y".repeat(3000)).chars().collect();
        assert_eq!(watch_with(0, &[]).cut(&closed, &Found::Repeat(repeat), &p), 2000, "a block closed before it stays");
    }

    /// In `log` mode a repeat is only logged, the reasoning passes through
    /// untouched; a repeat whose cut would keep too little is only logged in
    /// `rewind` mode too.
    #[tokio::test]
    async fn a_repeat_is_only_logged_in_log_mode_and_with_too_little_to_keep() {
        let (reasoning, _) = sentence_loop(Source::Italian, 1500, 2000, 4, false);
        let too_little = LoopGuardParams { min_kept_chars: 100_000, ..params(LoopGuardMode::Rewind) };
        for p in [params(LoopGuardMode::Log), too_little] {
            let ops = ops_with(Vec::new());
            let (text, done) = drain(guard(p.clone(), ops.clone(), input("p", 40_000), 1, scripted(1, &reasoning, true))).await;
            assert_eq!(text, reasoning, "{:?}", p.mode);
            assert!(done.is_some());
            assert!(ops.cancelled.lock().unwrap().is_empty() && ops.submitted.lock().unwrap().is_empty(), "{:?}", p.mode);
        }
    }

    #[test]
    fn code_in_a_block_is_never_scored_closed_or_still_open() {
        let words = prose(3000);
        let (before, after) = words.split_at(1500);
        let closed = format!("{before}\n```rust\n{}```\n{after}", code(6000));
        let open = format!("{before}\n```python\n{}", code(6000));
        for text in [&closed, &open] {
            assert!(first_loop(text, 0, &raw(LoopGuardMode::Log)).is_some(), "the code alone would flag");
            assert_eq!(first_loop(text, 0, &params(LoopGuardMode::Log)), None, "code was scored");
        }
        assert_eq!(scored_text(&closed), scored_text(&format!("{before}\n\n{after}")));
        assert_eq!(scored_text(&open), scored_text(&format!("{before}\n")), "an open block is not scored as it is written");
    }

    #[test]
    fn fences_follow_the_first_three_backticks_in_a_row() {
        assert_eq!(scored_text("a```code```b"), "ab");
        assert_eq!(scored_text("a````x```b"), "ab", "the fourth backtick is inside the block");
        assert_eq!(scored_text("a```x`````b c"), "ab c", "the two backticks after the close are an empty inline span");
        assert_eq!(scored_text("a```b```c```d"), "ac", "the third fence opens a block still open");
        assert_eq!(scored_text("x ``"), "x ", "two backticks wait for a third");
    }

    #[test]
    fn inline_code_is_taken_out_on_its_line() {
        assert_eq!(scored_text("a `x` b `` c `d\ne` f"), "a  b  c `d\ne", "the span still open waits");
        assert_eq!(scored_text("a `x` b `` c `d\ne` f\n"), "a  b  c `d\ne` f\n", "a line break makes it text");
        assert_eq!(scored_text("a `x ```code``` y` b"), "a  b", "a block is out first, then the span around it");
    }

    #[test]
    fn every_run_of_digits_reads_as_one_zero() {
        assert_eq!(scored_text("the 321 AD, the 322 AD; 1.5e10"), "the 0 AD, the 0 AD; 0.0e0");
        assert_eq!(scored_text("12`x`34 5"), "0 0", "a span taken out joins the run");
        assert_eq!(scored_text("١٢ x"), "١٢ x", "ASCII digits only");
    }

    #[test]
    fn off_scores_the_reasoning_as_written() {
        let text = "a `x` 12 ```c```\n";
        let scored = normalized(text, false);
        assert_eq!(scored.text().iter().collect::<String>(), text);
        assert!((0..scored.text().len()).all(|i| scored.raw_at(i) == i));
    }

    /// Each scored character maps to the raw character it stands for, in
    /// Unicode scalars (the decoder's unit): itself, or a run's first digit.
    #[test]
    fn each_scored_character_maps_to_its_raw_character() {
        let text = "è1é23 —`x` ```c``` 🦀4\n";
        let raw_chars: Vec<char> = text.chars().collect();
        let scored = normalized(text, true);
        assert_eq!(scored.text().iter().collect::<String>(), "è0é0 —  🦀0\n");
        assert_eq!(scored.raw_len(), raw_chars.len());
        let raw_at: Vec<usize> = (0..scored.text().len()).map(|i| scored.raw_at(i)).collect();
        assert_eq!(raw_at, [0, 1, 2, 3, 5, 6, 10, 18, 19, 20, 21]);
        for (i, &c) in scored.text().iter().enumerate() {
            let at = scored.raw_at(i);
            if c == '0' {
                assert!(raw_chars[at].is_ascii_digit() && (at == 0 || !raw_chars[at - 1].is_ascii_digit()), "{i}");
            } else {
                assert_eq!(raw_chars[at], c, "{i}");
            }
        }
    }

    /// What the scored text of `raw` must be, worked out on the whole text
    /// at once the way the offline research scorer's expressions do it,
    /// less what its end leaves undecided: blocks out (one never closed, to
    /// the end; one or two backticks at the very end wait), then on each
    /// line backticks paired in order with what is between them (an odd one
    /// on the last line waits with the rest of it), then digit runs.
    fn reference(raw: &str) -> String {
        let chars: Vec<char> = raw.chars().collect();
        let fence = |i: usize| chars.len() >= i + 3 && chars[i..i + 3] == ['`'; 3];
        let mut blocks_out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            if fence(i) {
                i = (i + 3..chars.len()).find(|&j| fence(j)).map_or(chars.len(), |j| j + 3);
            } else {
                blocks_out.push(chars[i]);
                i += 1;
            }
        }
        if raw.ends_with('`') {
            while blocks_out.last() == Some(&'`') {
                blocks_out.pop();
            }
        }
        let text: String = blocks_out.into_iter().collect();
        let lines: Vec<&str> = text.split('\n').collect();
        let mut spans_out = Vec::new();
        for (n, line) in lines.iter().enumerate() {
            let ticks: Vec<usize> = line.char_indices().filter(|&(_, c)| c == '`').map(|(i, _)| i).collect();
            let mut kept = String::new();
            let mut from = 0;
            for pair in ticks.chunks(2) {
                match *pair {
                    [open, close] => {
                        kept.push_str(&line[from..open]);
                        from = close + 1;
                    }
                    [odd] if n + 1 == lines.len() => {
                        kept.push_str(&line[from..odd]);
                        from = line.len();
                    }
                    _ => {}
                }
            }
            kept.push_str(&line[from..]);
            spans_out.push(kept);
        }
        let mut out = String::new();
        let mut run = false;
        for c in spans_out.join("\n").chars() {
            if c.is_ascii_digit() {
                if !run {
                    out.push('0');
                }
                run = true;
            } else {
                run = false;
                out.push(c);
            }
        }
        out
    }

    /// Every prefix of every text, so every way a token can end inside a
    /// fence or a span: the scored text as it streams is the reference's.
    #[test]
    fn the_streamed_text_is_the_whole_texts_at_every_prefix() {
        let mut texts = vec![
            "a```rust\nfn f() {}\n```b `x` c 12 `d\ne`` ```".to_owned(),
            "x``y```z````w`````v``````u\n`1`2`3`\n``\n".to_owned(),
        ];
        let alphabet = ['`', '`', '`', 'a', '1', '2', '\n', 'é', ' ', '🦀'];
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        for _ in 0..300 {
            let mut text = String::new();
            for _ in 0..40 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                text.push(alphabet[(state % alphabet.len() as u64) as usize]);
            }
            texts.push(text);
        }
        for text in &texts {
            let mut scored = Normalized::new(true);
            let mut prefix = String::new();
            for c in text.chars() {
                scored.push(c);
                prefix.push(c);
                assert_eq!(scored.text().iter().collect::<String>(), reference(&prefix), "prefix {prefix:?}");
            }
        }
    }

    #[test]
    fn the_tag_says_what_was_taken_back_and_hashes_it() {
        assert_eq!(tag(812, "abc"), "\n\n[LOOP DETECTED, REWINDING 812 TOKENS / 3 CHARS \u{b7} ba7816]");
        assert!(tag(1, "x").contains(TAG_PREFIX));
        // SHA-256's own test vectors: "abc" is ba7816bf…, the empty string e3b0c442….
        assert_eq!((struck_hash("abc"), struck_hash("")), ("ba7816".to_owned(), "e3b0c4".to_owned()));
        assert_eq!(tag(2, "è🦀\n").split(" CHARS").next().unwrap().rsplit(' ').next(), Some("3"), "m counts scalar values");
    }

    /// After a kept text that ends in a line break, a sentence goes in
    /// without its own: `"." "\n\n" | "\n\n" "Wait"` re-tokenizes as
    /// `"." "\n\n\n\n" "Wait"`, and the next turn's prompt would leave the
    /// retained state at the cut.
    #[test]
    fn a_sentence_after_a_line_break_opens_without_its_own() {
        for sentence in NUDGES.into_iter().chain([crate::thinking::THINKING_CLOSE_TEXT]) {
            let bare = sentence.trim_start_matches('\n');
            assert_ne!(bare, sentence, "every sentence opens with line breaks");
            assert_eq!(sentence_after("kept.\n", sentence), bare);
            assert_eq!(sentence_after("kept.\n\n", sentence), bare);
            assert_eq!(sentence_after("kept.", sentence), sentence);
            assert_eq!(sentence_after("", sentence), sentence, "nothing kept, no line break to join");
        }
    }

    /// `kept`, a rewind of `struck` tagged after it, then `after`.
    fn rewound_text(kept: &str, struck: &str, after: &str) -> String {
        format!("{kept}{struck}{}{after}", tag(9, struck))
    }

    #[test]
    fn repair_takes_a_tag_and_its_struck_text_out() {
        let text = rewound_text("kept\n", "the loop, the loop, ", NUDGES[0]);
        assert_eq!(repair(&text), format!("kept\n{}", NUDGES[0]));
        assert!(matches!(repair("no tag here"), Cow::Borrowed(_)));
        let multibyte = rewound_text("è ", "🦀🦀 ü\n", "next");
        assert_eq!(repair(&multibyte), "è next", "m counts scalar values, not bytes");
        assert_eq!(repair(&rewound_text("kept", "", "next")), "keptnext", "an empty struck text is the tag alone");
        assert_eq!(repair(&format!("{}{}", tag(1, "abc"), "x")), "\n\n[LOOP DETECTED, REWINDING 1 TOKENS / 3 CHARS \u{b7} ba7816]x", "nothing before it");
    }

    /// Two rewinds, the second's struck text after the first's tag: from
    /// the last to the first, both go.
    #[test]
    fn repair_takes_every_tag_out_from_the_last() {
        let first = rewound_text("A", "loop one ", NUDGES[0]);
        let text = rewound_text(&format!("{first}kept two "), "loop two ", NUDGES[1]);
        assert_eq!(repair(&text), format!("A{}kept two {}", NUDGES[0], NUDGES[1]));
        // The struck text of the second may begin right at the first tag's end.
        let back_to_back = rewound_text(&rewound_text("A", "loop one ", ""), "loop two", "end");
        assert_eq!(repair(&back_to_back), "Aend");
    }

    /// A tag whose hash does not match, one with fewer characters before it
    /// than it names, and one in the format before the hash are all left as
    /// they are; a good tag beside them still goes.
    #[test]
    fn repair_leaves_a_tag_it_cannot_verify() {
        let edited = rewound_text("kept ", "the loop", "next").replace("the loop", "the LOOP");
        assert_eq!(repair(&edited), edited, "a mismatch");
        let short = format!("ab{}", tag(1, "abc"));
        assert_eq!(repair(&short), short, "fewer characters than m");
        let old = "kept the loop\n\n[LOOP DETECTED, REWINDING 2 TOKENS / 8 CHARS]next";
        assert_eq!(repair(old), old, "the old format");
        let upper = rewound_text("k", "abc", "n").replace("ba7816", "BA7816");
        assert_eq!(repair(&upper), upper, "the hash is lowercase");
        for broken in ["\n\n[LOOP DETECTED, REWINDING x TOKENS / 3 CHARS \u{b7} ba7816]", "\n\n[LOOP DETECTED, REWINDING 1 TOKENS / 3 CHARS \u{b7} ba78]"] {
            let text = format!("abc{broken}");
            assert_eq!(repair(&text), text, "{broken:?}");
        }
        let mixed = format!("{old}{}", rewound_text(" k ", "loop", "end"));
        assert_eq!(repair(&mixed), format!("{old} k end"), "the old tag stays, the good one goes");
        let after_a_bad_one = format!("{}{edited}", rewound_text("A", "loop", "B"));
        assert_eq!(repair(&after_a_bad_one), format!("AB{edited}"), "a bad tag does not stop the ones before it");
    }

    #[test]
    fn history_repair_touches_only_assistant_reasoning() {
        let tagged = rewound_text("kept ", "loop", "next");
        let mut messages = vec![
            ChatMessage::text("system", tagged.clone()),
            ChatMessage::text("user", tagged.clone()),
            ChatMessage { reasoning_content: Some(tagged.clone()), ..ChatMessage::text("assistant", tagged.clone()) },
            ChatMessage { reasoning_content: Some(tagged.clone()), ..ChatMessage::text("user", "q") },
            ChatMessage::text("tool", tagged.clone()),
            ChatMessage::text("assistant", "no reasoning"),
        ];
        let before = messages.clone();
        repair_history(&mut messages);
        assert_eq!(messages[2].reasoning_content.as_deref(), Some("kept next"), "assistant reasoning is repaired");
        assert_eq!(messages[2].content, before[2].content, "assistant content is not");
        for i in [0, 1, 3, 4, 5] {
            assert_eq!(messages[i], before[i], "message {i}");
        }
    }

    /// What a client shows: published text and what the decoder holds,
    /// never what `finish` drops.
    #[test]
    fn shown_is_the_text_with_what_the_decoder_holds() {
        let decoder = || OutputDecoder::new(Box::new(CharDecoder), true);
        assert_eq!(shown(decoder(), &tokens_of("ab\n")), "ab\n");
        assert_eq!(shown(decoder(), &tokens_of("ab</thi")), "ab</thi");
        assert_eq!(shown(decoder(), &tokens_of("a</think>b")), "a");
        assert_eq!(shown(decoder(), &[u32::from('a'), NEWLINE_AFTER + u32::from('b')]), "ab\n");
    }

    /// A stand-in engine: records every cancel and submission (and how long
    /// it may wait for room), and answers a submission with the next
    /// scripted stream.
    #[derive(Default)]
    struct FakeOps {
        cancelled: Mutex<Vec<RequestId>>,
        submitted: Mutex<Vec<RequestInput>>,
        waits: Mutex<Vec<Duration>>,
        streams: Mutex<Vec<(RequestId, EventStream)>>,
    }

    impl GuardOps for Arc<FakeOps> {
        fn cancel(&self, request: RequestId) {
            self.cancelled.lock().unwrap().push(request);
        }
        fn submit(&self, input: RequestInput, wait: Duration) -> Pin<Box<dyn Future<Output = Option<(RequestId, EventStream)>> + Send + '_>> {
            self.submitted.lock().unwrap().push(input);
            self.waits.lock().unwrap().push(wait);
            let next = self.streams.lock().unwrap().pop();
            Box::pin(async move { next })
        }
        fn decoder(&self) -> OutputDecoder {
            OutputDecoder::new(Box::new(CharDecoder), true)
        }
        fn encode(&self, text: &str) -> Option<Vec<TokenId>> {
            Some(tokens_of(text))
        }
    }

    fn scripted(request: RequestId, text: &str, done: bool) -> EventStream {
        let (tx, rx) = unbounded_channel();
        for token in tokens_of(text) {
            tx.send(SchedEvent::Token { request, token }).unwrap();
        }
        if done {
            tx.send(done_event(request, text.chars().count() as u32)).unwrap();
        }
        rx
    }

    fn done_event(request: RequestId, tokens: u32) -> SchedEvent {
        done_with(request, tokens, FinishReason::Stop, None, None)
    }

    fn done_with(
        request: RequestId,
        tokens: u32,
        reason: FinishReason,
        thinking: Option<ignis_core::thinking_budget::BudgetOutcome>,
        reasoning_redirected_at: Option<u32>,
    ) -> SchedEvent {
        SchedEvent::Done {
            request,
            tokens,
            reason,
            spec: None,
            readout: None,
            attention: None,
            drawn: None,
            thinking,
            reasoning_redirected_at,
        }
    }

    fn input(prompt: &str, max_tokens: u32) -> RequestInput {
        RequestInput {
            decision: None,
            constrained: None,
            forced_literal: None,
            grammar: None,
            warm_up: false,
            multimodal: None,
            opener_tokens: None,
            user_turn_tokens: None,
            system_block_tokens: None,
            reuse_boundaries: Vec::new(),
            model: "m".into(),
            tokens: tokens_of(prompt),
            params: DecodeParams {
                max_tokens: Some(max_tokens),
                thinking_budget: Some(max_tokens - 2048),
                starts_in_reasoning: true,
                seed: 7,
                ..DecodeParams::default()
            },
        }
    }

    async fn drain(rx: EventStream) -> (String, Option<SchedEvent>) {
        let (tokens, done) = drain_tokens(rx).await;
        (text_of(&tokens), done)
    }

    /// Every token the client received, in order, and the `Done`.
    async fn drain_tokens(mut rx: EventStream) -> (Vec<TokenId>, Option<SchedEvent>) {
        let mut tokens = Vec::new();
        let mut done = None;
        while let Some(event) = rx.recv().await {
            match event {
                SchedEvent::Token { token, request } => {
                    assert_eq!(request, 1, "a token under another id than the client's");
                    tokens.push(token);
                }
                event @ SchedEvent::Done { .. } => done = Some(event),
                _ => {}
            }
        }
        (tokens, done)
    }

    /// [`scripted`], from tokens.
    fn scripted_tokens(request: RequestId, tokens: &[TokenId], done: bool) -> EventStream {
        let (tx, rx) = unbounded_channel();
        for &token in tokens {
            tx.send(SchedEvent::Token { request, token }).unwrap();
        }
        if done {
            tx.send(done_event(request, tokens.len() as u32)).unwrap();
        }
        rx
    }

    fn ops_with(streams: Vec<(RequestId, EventStream)>) -> Arc<FakeOps> {
        Arc::new(FakeOps { streams: Mutex::new(streams), ..FakeOps::default() })
    }

    /// The tokens the model saw after the client's prompt `prompt`: those the
    /// last continuation's prompt carried past it, then `last`, what that
    /// continuation generated.
    fn models_view(ops: &FakeOps, prompt: &str, last: &[TokenId]) -> Vec<TokenId> {
        let submitted = ops.submitted.lock().unwrap();
        let mut view = submitted.last().expect("a continuation").tokens[tokens_of(prompt).len()..].to_vec();
        view.extend_from_slice(last);
        view
    }

    /// History repair's promise on a stream the guard produced: the
    /// reasoning the client received, repaired, is the reasoning the model
    /// saw, which is what the continuation's prompt carried.
    fn assert_repair_gives_the_models_view(received: &[TokenId], ops: &FakeOps, prompt: &str, last: &[TokenId]) {
        let model = client_reasoning(&models_view(ops, prompt, last));
        let received = client_reasoning(received);
        assert_ne!(received, model, "the client was sent a rewind: there is something to repair");
        assert_eq!(repair(&received), model);
    }

    /// A continuation prompt's tail past the client's prompt: the reasoning
    /// it kept from `reasoning`, then `sentence` as [`sentence_after`] joins
    /// it. (A test reasoning never has two line breaks in a row, so only
    /// one split fits.)
    fn kept_then<'a>(tail: &'a str, sentence: &'a str, reasoning: &str) -> (&'a str, &'a str) {
        [sentence.trim_start_matches('\n'), sentence]
            .into_iter()
            .find_map(|joined| {
                let kept = tail.strip_suffix(joined)?;
                (reasoning.starts_with(kept) && sentence_after(kept, sentence) == joined).then_some((kept, joined))
            })
            .unwrap_or_else(|| panic!("no kept reasoning then {sentence:?}"))
    }

    /// The `m` and the struck text of the first tag the client shows.
    fn first_struck(shown: &str) -> (usize, String) {
        let (before, rest) = shown.split_once(TAG_OPEN).expect("the tag");
        let m: usize = rest.split_once(" CHARS").unwrap().0.rsplit(' ').next().unwrap().parse().unwrap();
        (m, before.chars().skip(before.chars().count() - m).collect())
    }

    #[tokio::test]
    async fn off_hands_the_stream_back_untouched() {
        let ops = ops_with(Vec::new());
        let (text, done) = drain(guard(params(LoopGuardMode::Off), ops.clone(), input("p", 40_000), 1, scripted(1, "abc", true))).await;
        assert_eq!(text, "abc");
        assert!(done.is_some());
    }

    #[tokio::test]
    async fn log_mode_passes_a_loop_through_and_acts_on_nothing() {
        let ops = ops_with(Vec::new());
        let reasoning = prose(2000) + &looping(6000);
        let (text, done) =
            drain(guard(params(LoopGuardMode::Log), ops.clone(), input("p", 40_000), 1, scripted(1, &reasoning, true))).await;
        assert_eq!(text, reasoning);
        assert!(done.is_some());
        assert!(ops.cancelled.lock().unwrap().is_empty());
        assert!(ops.submitted.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_loop_is_rewound_nudged_and_continued() {
        let after = "\nRome's story, written.</think>\n\nRoma.";
        let ops = ops_with(vec![(2, scripted(2, after, true))]);
        let head = prose(5000);
        let reasoning = head.clone() + &looping(6000);
        let (text, done) =
            drain(guard(params(LoopGuardMode::Rewind), ops.clone(), input("PROMPT", 40_000), 1, scripted(1, &reasoning, false)))
                .await;

        assert_eq!(*ops.cancelled.lock().unwrap(), vec![1], "the looping request is cancelled");
        let submitted = ops.submitted.lock().unwrap();
        let next = &submitted[0];
        let continued = text_of(&next.tokens);
        let (kept, nudge) = kept_then(continued.strip_prefix("PROMPT").expect("the prompt first"), NUDGES[0], &reasoning);
        assert!(head.starts_with(kept), "the continuation keeps a prefix of the reasoning before the loop");
        assert!(kept.len() > 1000, "kept only {} chars", kept.len());
        let used = (kept.chars().count() + nudge.chars().count()) as u32;
        assert_eq!(next.params.max_tokens, Some(40_000 - used));
        assert_eq!(next.params.thinking_budget, Some(40_000 - 2048 - used));
        assert!(next.params.starts_in_reasoning);
        assert_ne!(next.params.seed, 7, "a fresh draw");

        // The client: everything it was sent, the tag, the nudge, then the continuation.
        let (sent, rest) = text.split_once("\n\n[LOOP DETECTED, REWINDING ").expect("the tag");
        let (counts, rest) = rest.split_once(']').expect("the tag's end");
        assert!(reasoning.starts_with(sent));
        let struck = &sent[kept.len()..];
        let m = struck.chars().count();
        assert_eq!(counts, format!("{m} TOKENS / {m} CHARS \u{b7} {}", struck_hash(struck)), "one token per char here");
        assert_eq!(rest, format!("{nudge}{after}"));
        let Some(SchedEvent::Done { request, tokens, .. }) = done else { panic!("no done") };
        assert_eq!(request, 1);
        assert_eq!(tokens as usize, text.chars().count(), "every token the client received");
    }

    /// A cut on a line break continues with the nudge's words straight
    /// after it, in the continuation's prompt and in what the client gets
    /// after the tag; a cut that is not on one keeps the nudge's line breaks.
    #[tokio::test]
    async fn a_nudge_after_a_kept_line_break_goes_in_without_its_own() {
        for (head, on_a_line_break) in [(prose(5000), true), (prose(5000).replace('\n', " "), false)] {
            let ops = ops_with(vec![(2, scripted(2, "Roma.", true))]);
            let reasoning = head.clone() + &looping(6000);
            let (text, _) =
                drain(guard(params(LoopGuardMode::Rewind), ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
            let continued = text_of(&ops.submitted.lock().unwrap()[0].tokens);
            let sentence = if on_a_line_break { NUDGES[0].trim_start_matches('\n') } else { NUDGES[0] };
            let kept = continued.strip_prefix('P').unwrap().strip_suffix(sentence).expect("the nudge last");
            assert!(head.starts_with(kept));
            assert_eq!(kept.ends_with('\n'), on_a_line_break, "kept ends {:?}", &kept[kept.len() - 20..]);
            let (_, after_tag) = text.split_once(TAG_OPEN).expect("the tag");
            assert_eq!(after_tag.split_once(']').unwrap().1, format!("{sentence}Roma."));
        }
    }

    /// The count the handler reports: the struck tokens of every rewind,
    /// not the tags or the nudges.
    #[tokio::test]
    async fn the_rewound_count_is_the_struck_tokens_of_every_rewind() {
        let ops = ops_with(vec![(3, scripted(3, "Roma.", true)), (2, scripted(2, &looping(4000), false))]);
        let reasoning = prose(5000) + &looping(6000);
        let rewound = Rewound::default();
        let (text, _) = drain(guard_counted(
            params(LoopGuardMode::Rewind),
            ops.clone(),
            input("P", 40_000),
            Running::new(1),
            scripted(1, &reasoning, false),
            rewound.clone(),
            None,
        ))
        .await;
        // One token per char here: each tag reports its struck tokens.
        let tags: u32 = text
            .split("[LOOP DETECTED, REWINDING ")
            .skip(1)
            .map(|rest| rest.split_once(" TOKENS").unwrap().0.parse::<u32>().unwrap())
            .sum();
        assert_eq!(ops.submitted.lock().unwrap().len(), 2);
        assert!(tags > 0);
        assert_eq!((rewound.taken(), rewound.unsent()), (tags, 0), "with no holdback every struck token was sent");
    }

    /// The usage's arithmetic, on both surfaces: the tokens a holdback never
    /// sent are added to what the stream carried, and `rewound_tokens` is
    /// absent when nothing was taken back.
    #[test]
    fn the_usage_counts_what_was_never_sent_and_names_what_was_taken_back() {
        assert_eq!(RewoundTokens::default().usage(40), (40, None));
        assert_eq!(RewoundTokens::new(321, 0).usage(1500), (1500, Some(321)));
        assert_eq!(RewoundTokens::new(321, 40).usage(1500), (1540, Some(321)));
    }

    /// After a rewind the watch starts again at the nudge's end: a loop
    /// straight after it has no clean window since, and is cut there, the
    /// first continuation kept whole.
    #[tokio::test]
    async fn the_watch_starts_again_after_a_rewind() {
        let ops = ops_with(vec![(3, scripted(3, "Roma.", true)), (2, scripted(2, &looping(4000), false))]);
        let reasoning = prose(5000) + &looping(6000);
        let (text, _) =
            drain(guard(params(LoopGuardMode::Rewind), ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
        assert_eq!(*ops.cancelled.lock().unwrap(), vec![1, 2]);
        let submitted = ops.submitted.lock().unwrap();
        assert_eq!(submitted.len(), 2, "two loops, two rewinds");
        let first = text_of(&submitted[0].tokens);
        kept_then(first.strip_prefix('P').unwrap(), NUDGES[0], &reasoning);
        let second = sentence_after(&first, NUDGES[1]);
        assert_eq!(text_of(&submitted[1].tokens), format!("{first}{second}"), "cut at the first nudge's end");
        assert!(text.ends_with(&format!("{second}Roma.")));
    }

    /// The configured nudges are what a rewind writes, in the
    /// continuation's prompt and after the tag the client reads: `nudge`
    /// first, then `nudge_again`.
    #[tokio::test]
    async fn a_rewind_writes_the_configured_nudges() {
        let ops = ops_with(vec![(3, scripted(3, "Roma.", true)), (2, scripted(2, &looping(4000), false))]);
        let reasoning = prose(5000) + &looping(6000);
        let p = LoopGuardParams { nudge: "Next step.".to_owned(), nudge_again: "Wrap up.".to_owned(), ..params(LoopGuardMode::Rewind) };
        let (text, _) = drain(guard(p, ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
        let submitted = ops.submitted.lock().unwrap();
        assert_eq!(submitted.len(), 2, "two loops, two rewinds");
        let first = text_of(&submitted[0].tokens);
        let (_, first_nudge) = kept_then(first.strip_prefix('P').unwrap(), "\n\nNext step.\n\n", &reasoning);
        let second = sentence_after(&first, "\n\nWrap up.\n\n");
        assert_eq!(text_of(&submitted[1].tokens), format!("{first}{second}"));
        let (_, after_tag) = text.split_once(TAG_OPEN).expect("the first tag");
        assert!(after_tag.split_once(']').unwrap().1.starts_with(first_nudge));
        assert!(text.ends_with(&format!("{second}Roma.")));
        assert!(!text.contains("repeat myself"), "no default sentence");
    }

    /// A reasoning that opens in a loop has no clean window to go back to:
    /// the rewind would keep nothing, so the loop is only logged and passes
    /// through, once. At `min_kept_chars` 0 it is rewound from nothing.
    #[tokio::test]
    async fn a_rewind_that_would_keep_nothing_is_only_logged() {
        let reasoning = looping(8000);
        let ops = ops_with(vec![(2, scripted(2, "Roma.", true))]);
        let (text, done) =
            drain(guard(params(LoopGuardMode::Rewind), ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, true))).await;
        assert_eq!(text, reasoning, "passed through untouched");
        assert!(done.is_some());
        assert!(ops.cancelled.lock().unwrap().is_empty() && ops.submitted.lock().unwrap().is_empty(), "never rewound");

        // Line breaks the decoder holds at the start are not reasoning kept.
        let ops = ops_with(vec![(2, scripted(2, "Roma.", true))]);
        let opened = format!("\n\n{reasoning}");
        let (_, done) =
            drain(guard(params(LoopGuardMode::Rewind), ops.clone(), input("P", 40_000), 1, scripted(1, &opened, true))).await;
        assert!(done.is_some());
        assert!(ops.submitted.lock().unwrap().is_empty(), "a held line break is not something to keep");

        let ops = ops_with(vec![(2, scripted(2, "Roma.", true))]);
        let p = LoopGuardParams { min_kept_chars: 0, ..params(LoopGuardMode::Rewind) };
        let (text, _) = drain(guard(p, ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
        assert_eq!(*ops.cancelled.lock().unwrap(), vec![1]);
        assert_eq!(text_of(&ops.submitted.lock().unwrap()[0].tokens), format!("P{}", NUDGES[0]), "nothing kept, the nudge alone");
        assert!(text.ends_with("Roma."));
    }

    /// After a loop with nothing to keep, a clean window gives the next
    /// loop a cut: it is rewound to there, the first loop kept.
    #[tokio::test]
    async fn after_a_loop_with_nothing_to_keep_a_later_loop_is_rewound_to_a_clean_window() {
        let head = looping(3000) + &prose(4000);
        let reasoning = head.clone() + &looping(6000);
        let ops = ops_with(vec![(2, scripted(2, "Roma.", true))]);
        drain(guard(params(LoopGuardMode::Rewind), ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
        let submitted = ops.submitted.lock().unwrap();
        assert_eq!(submitted.len(), 1, "one rewind, the second loop's");
        let continued = text_of(&submitted[0].tokens);
        let (kept, _) = kept_then(continued.strip_prefix('P').unwrap(), NUDGES[0], &reasoning);
        assert!(kept.len() > 3000 && head.starts_with(kept), "kept the first loop and some prose: {} chars", kept.len());
    }

    /// On the normalized score: a counting loop after a code block is found,
    /// and the cut is made on the raw reasoning, after the block.
    #[tokio::test]
    async fn a_counting_loop_after_code_is_rewound_on_the_raw_text() {
        let ops = ops_with(vec![(2, scripted(2, "Roma.", true))]);
        let words = prose(5500);
        let (before, after) = words.split_at(3000);
        let head = format!("{before}\n```rust\n{}```\n{after}", code(3000));
        let reasoning = head.clone() + &counting(4000);
        let (text, _) =
            drain(guard(params(LoopGuardMode::Rewind), ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
        assert!(text.contains(TAG_PREFIX), "the counting loop was not rewound");
        let submitted = ops.submitted.lock().unwrap();
        let continued = text_of(&submitted[0].tokens);
        let (kept, _) = kept_then(continued.strip_prefix("P").expect("the prompt first"), NUDGES[0], &reasoning);
        assert!(head.starts_with(kept), "the cut is inside the loop");
        assert!(kept.len() > head.len() - after.len(), "the cut took the code block back too: kept {}", kept.len());
    }

    #[tokio::test]
    async fn after_the_retries_the_loop_is_closed() {
        let ops = ops_with(vec![(2, scripted(2, "Roma.", true))]);
        let reasoning = prose(5000) + &looping(6000);
        let p = LoopGuardParams { retries: 0, ..params(LoopGuardMode::Rewind) };
        let (text, _) = drain(guard(p, ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
        let submitted = ops.submitted.lock().unwrap();
        let next = &submitted[0];
        let continued = text_of(&next.tokens);
        let (_, close) = kept_then(continued.strip_prefix('P').unwrap(), crate::thinking::THINKING_CLOSE_TEXT, &reasoning);
        assert_eq!(next.params.thinking_budget, None);
        assert!(!next.params.starts_in_reasoning, "the prompt closed the reasoning");
        assert!(text.ends_with(&format!("{close}Roma.")));
    }

    #[tokio::test]
    async fn a_client_that_leaves_cancels_the_continuation() {
        let (cont_tx, cont_rx) = unbounded_channel();
        let ops = ops_with(vec![(2, cont_rx)]);
        let reasoning = prose(5000) + &looping(6000);
        let mut rx = guard(params(LoopGuardMode::Rewind), ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false));
        // Read until the tag, then leave.
        let mut seen = String::new();
        while !seen.contains(TAG_PREFIX) {
            if let Some(SchedEvent::Token { token, .. }) = rx.recv().await {
                seen.push(char::from_u32(token).unwrap());
            }
        }
        drop(rx);
        cont_tx.send(SchedEvent::Token { request: 2, token: u32::from('x') }).unwrap();
        for _ in 0..100 {
            if ops.cancelled.lock().unwrap().contains(&2) {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("the continuation was never cancelled: {:?}", ops.cancelled.lock().unwrap());
    }

    /// Yield until `done` holds, or fail naming `what`.
    async fn until(what: &str, done: impl Fn() -> bool) {
        for _ in 0..1000 {
            if done() {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("{what}");
    }

    /// What cancels by response reads the engine request generating now:
    /// the one submitted first, then the continuation.
    #[tokio::test]
    async fn the_running_request_is_the_continuation_after_a_rewind() {
        let (_cont_tx, cont_rx) = unbounded_channel();
        let ops = ops_with(vec![(2, cont_rx)]);
        let running = Running::new(1);
        let reasoning = prose(5000) + &looping(6000);
        let _rx = guard_counted(
            params(LoopGuardMode::Rewind),
            ops.clone(),
            input("P", 40_000),
            running.clone(),
            scripted(1, &reasoning, false),
            Rewound::default(),
            None,
        );
        assert_eq!(running.get(), 1);
        until("no rewind", || !ops.submitted.lock().unwrap().is_empty()).await;
        assert_eq!(running.get(), 2);
    }

    /// A consumer that leaves stops the engine request generating now even
    /// when there is nothing to send it, so no failed send tells: here the
    /// holdback hid the rewind and holds every token.
    #[tokio::test]
    async fn a_consumer_that_leaves_cancels_the_continuation_with_nothing_to_send() {
        let (_cont_tx, cont_rx) = unbounded_channel();
        let ops = ops_with(vec![(2, cont_rx)]);
        let reasoning = prose(5000) + &looping(6000);
        let p = LoopGuardParams { holdback_chars: 6000, ..params(LoopGuardMode::Rewind) };
        let rx = guard(p, ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false));
        until("no rewind", || !ops.submitted.lock().unwrap().is_empty()).await;
        drop(rx);
        until("the continuation was never cancelled", || ops.cancelled.lock().unwrap().contains(&2)).await;
    }

    /// [`FakeOps`] whose submissions never get an answer: the engine full.
    struct Stalled(Arc<FakeOps>);

    impl GuardOps for Stalled {
        fn cancel(&self, request: RequestId) {
            self.0.cancel(request);
        }
        fn submit(&self, input: RequestInput, _: Duration) -> Pin<Box<dyn Future<Output = Option<(RequestId, EventStream)>> + Send + '_>> {
            self.0.submitted.lock().unwrap().push(input);
            Box::pin(std::future::pending())
        }
        fn decoder(&self) -> OutputDecoder {
            self.0.decoder()
        }
        fn encode(&self, text: &str) -> Option<Vec<TokenId>> {
            self.0.encode(text)
        }
    }

    /// A consumer that leaves while the continuation waits for room takes
    /// it out of the wait: the guard ends, and nothing is left to admit.
    #[tokio::test]
    async fn a_consumer_that_leaves_while_the_continuation_waits_ends_the_wait() {
        let ops = ops_with(Vec::new());
        let reasoning = prose(5000) + &looping(6000);
        let rx = guard(params(LoopGuardMode::Rewind), Stalled(ops.clone()), input("P", 40_000), 1, scripted(1, &reasoning, false));
        until("no rewind", || !ops.submitted.lock().unwrap().is_empty()).await;
        drop(rx);
        until("the guard still waits", || Arc::strong_count(&ops) == 1).await;
        let cancelled = ops.cancelled.lock().unwrap();
        assert!(cancelled.contains(&1) && cancelled.iter().all(|&request| request == 1), "only the looping request: {cancelled:?}");
    }

    /// A loop the guard did not act on, given the client's text and the ops
    /// that would have: nothing cancelled, nothing submitted, the stream
    /// passed through.
    fn assert_passed_through(ops: &FakeOps, text: &str, reasoning: &str) {
        assert_eq!(text, reasoning, "the client got the reasoning as generated");
        assert!(ops.cancelled.lock().unwrap().is_empty(), "nothing cancelled");
        assert!(ops.submitted.lock().unwrap().is_empty(), "nothing submitted");
    }

    /// A loop with too little of `max_tokens` left after the cut for a
    /// continuation to be worth one (`min_continuation`, 512 by default) is
    /// only logged (`no_room`), and one token more is rewound.
    #[tokio::test]
    async fn a_loop_with_no_room_left_to_continue_is_only_logged() {
        let reasoning = prose(5000) + &looping(6000);
        let ops = ops_with(vec![(2, scripted(2, "Roma.", true))]);
        drain(guard(params(LoopGuardMode::Rewind), ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
        let used = 40_000 - ops.submitted.lock().unwrap()[0].params.max_tokens.unwrap();
        for room in [DEFAULT_MIN_CONTINUATION_TOKENS, 2000] {
            let p = LoopGuardParams { min_continuation_tokens: room, ..params(LoopGuardMode::Rewind) };
            for (max_tokens, rewound) in [(used + room - 1, false), (used + room, true)] {
                let ops = ops_with(vec![(2, scripted(2, "Roma.", true))]);
                let (text, done) = drain(guard(p.clone(), ops.clone(), input("P", max_tokens), 1, scripted(1, &reasoning, true))).await;
                assert!(done.is_some());
                match rewound {
                    false => assert_passed_through(&ops, &text, &reasoning),
                    true => assert_eq!(ops.submitted.lock().unwrap()[0].params.max_tokens, Some(room)),
                }
            }
        }
    }

    /// The continuation may wait `submit_wait` for room: 60 s by default, 0
    /// tried once.
    #[tokio::test]
    async fn the_continuation_waits_for_room_as_long_as_submit_wait_says() {
        let reasoning = prose(5000) + &looping(6000);
        for secs in [DEFAULT_SUBMIT_WAIT_SECS, 0] {
            let ops = ops_with(vec![(2, scripted(2, "Roma.", true))]);
            let p = LoopGuardParams { submit_wait_secs: secs, ..params(LoopGuardMode::Rewind) };
            drain(guard(p, ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
            assert_eq!(*ops.waits.lock().unwrap(), [Duration::from_secs(u64::from(secs))]);
        }
    }

    /// A request the stand-in model generates `max_tokens` tokens for.
    fn plain(max_tokens: u32) -> RequestInput {
        RequestInput { params: DecodeParams { max_tokens: Some(max_tokens), ..DecodeParams::default() }, ..input("P", 40_000) }
    }

    /// [`EngineOps`] on a real engine with one place, which a request that
    /// runs until it is cancelled holds: a continuation finds it full.
    async fn full_engine() -> (EngineOps, RequestId, EventStream) {
        let scheduler = ignis_core::ConcreteScheduler::with_config(
            ignis_core::SchedulerConfig {
                model: "m".into(),
                max_in_flight: 1,
                max_sequence_tokens: 2 << 20,
                kv_capacity_pages: (2 << 20) / 16 * 2,
                ..ignis_core::SchedulerConfig::default()
            },
            Arc::new(ignis_core::MockCompute::new()),
        );
        let engine = Engine::new(Box::new(scheduler));
        let (holder, stream) = engine.submit(plain(1_000_000), RequestClass::Interactive).await.expect("the one place");
        (EngineOps::new(engine, RequestClass::Interactive, Arc::new(crate::template::SimpleTemplateProvider)), holder, stream)
    }

    /// On the live engine, a continuation refused as full is tried again
    /// until `submit_wait` runs out: with 0, once; with more, no longer than
    /// that (not the five seconds a retry waits for a request to leave), and
    /// admitted as soon as one leaves within it.
    #[tokio::test]
    async fn a_continuation_the_engine_finds_full_is_retried_within_the_submit_wait() {
        let (ops, holder, _stream) = full_engine().await;
        let bounded = |wait: Duration| tokio::time::timeout(Duration::from_secs(4), ops.submit(plain(4), wait));
        assert!(bounded(Duration::ZERO).await.expect("tried once, at once").is_none());
        let started = std::time::Instant::now();
        assert!(bounded(Duration::from_millis(300)).await.expect("given up when the wait ran out").is_none());
        assert!(started.elapsed() >= Duration::from_millis(300), "it waited: {:?}", started.elapsed());

        let engine = ops.engine().clone();
        let leave = async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            engine.cancel(holder);
        };
        let (admitted, ()) = tokio::join!(bounded(Duration::from_secs(60)), leave);
        let (_, mut stream) = admitted.expect("admitted when the place freed").expect("admitted");
        assert!(stream.recv().await.is_some(), "it runs");
    }

    /// An engine whose model thread is gone frees no place: the wait ends
    /// at once instead of trying again until `submit_wait` runs out.
    #[tokio::test]
    async fn a_continuation_on_a_shut_down_engine_is_given_up_at_once() {
        let (ops, _, _stream) = full_engine().await;
        ops.engine().shutdown();
        let given_up = tokio::time::timeout(Duration::from_secs(4), ops.submit(plain(4), Duration::from_secs(60))).await;
        assert!(given_up.expect("given up before the wait ran out").is_none());
    }

    /// [`FakeOps`] with no tokenizer: `encode` gives nothing.
    struct NoTokenizer(Arc<FakeOps>);

    impl GuardOps for NoTokenizer {
        fn cancel(&self, request: RequestId) {
            self.0.cancel(request);
        }
        fn submit(&self, input: RequestInput, wait: Duration) -> Pin<Box<dyn Future<Output = Option<(RequestId, EventStream)>> + Send + '_>> {
            self.0.submit(input, wait)
        }
        fn decoder(&self) -> OutputDecoder {
            self.0.decoder()
        }
        fn encode(&self, _: &str) -> Option<Vec<TokenId>> {
            None
        }
    }

    /// With no tokenizer to write the nudge in, a loop is only logged
    /// (`no_tokenizer`).
    #[tokio::test]
    async fn a_loop_with_no_tokenizer_for_the_nudge_is_only_logged() {
        let reasoning = prose(5000) + &looping(6000);
        let ops = ops_with(vec![(2, scripted(2, "Roma.", true))]);
        let (text, done) =
            drain(guard(params(LoopGuardMode::Rewind), NoTokenizer(ops.clone()), input("P", 40_000), 1, scripted(1, &reasoning, true))).await;
        assert!(done.is_some());
        assert_passed_through(&ops, &text, &reasoning);
    }

    /// The tokens the client has once the guard has taken in everything
    /// sent on `tx` so far: a marker event goes after them, and the guard
    /// passes it on only when it reaches it, every release before it made.
    async fn received_by(tx: &UnboundedSender<SchedEvent>, out: &mut EventStream, got: &mut Vec<TokenId>) {
        tx.send(SchedEvent::Evicted { request: 1, snapshot_micros: 0 }).unwrap();
        loop {
            match out.recv().await.expect("the marker") {
                SchedEvent::Token { token, .. } => got.push(token),
                SchedEvent::Evicted { .. } => return,
                other => panic!("only tokens and the marker: {other:?}"),
            }
        }
    }

    /// A request with images is never rewound, since a continuation cannot
    /// carry its media: a loop is only logged, and no holdback keeps the
    /// client behind.
    #[tokio::test]
    async fn a_request_with_images_is_never_rewound_nor_held_back() {
        let reasoning = prose(5000) + &looping(6000);
        let images = RequestInput {
            multimodal: Some(Arc::new(ignis_core::vision::Multimodal { positions: Vec::new(), rope_delta: 0, media: Vec::new() })),
            ..input("P", 40_000)
        };
        let p = LoopGuardParams { holdback_chars: 1000, ..params(LoopGuardMode::Rewind) };
        let ops = ops_with(vec![(2, scripted(2, "Roma.", true))]);
        let (tx, rx) = unbounded_channel();
        let mut out = guard(p, ops.clone(), images, 1, rx);
        for token in tokens_of(&reasoning) {
            tx.send(SchedEvent::Token { request: 1, token }).unwrap();
        }
        let mut got = Vec::new();
        received_by(&tx, &mut out, &mut got).await;
        assert_passed_through(&ops, &text_of(&got), &reasoning);
    }

    /// A request ended before its own end (a cancel, a timeout) gets what
    /// the holdback held: the engine request is cancelled, and the stream
    /// carries the rest of the reasoning, then ends.
    #[tokio::test]
    async fn a_stopped_request_is_sent_what_was_held_back() {
        let reasoning = prose(3000);
        let ops = ops_with(Vec::new());
        let (tx, rx) = unbounded_channel();
        let running = Running::new(1);
        let p = LoopGuardParams { holdback_chars: 1000, ..params(LoopGuardMode::Rewind) };
        let mut out = guard_counted(p, ops.clone(), input("P", 40_000), running.clone(), rx, Rewound::default(), None);
        for token in tokens_of(&reasoning) {
            tx.send(SchedEvent::Token { request: 1, token }).unwrap();
        }
        let mut got = Vec::new();
        received_by(&tx, &mut out, &mut got).await;
        assert!(got.len() < 2100, "the holdback keeps the client behind: {}", got.len());
        got.extend(held_back(&running, &mut out).await);
        assert_eq!(text_of(&got), reasoning, "the rest, then the end");
        assert_eq!(*ops.cancelled.lock().unwrap(), vec![1]);
    }

    /// Stopped while its continuation waits for room: the client is owed
    /// the kept reasoning, never the loop the rewind took back, and nothing
    /// is left to admit.
    #[tokio::test]
    async fn a_request_stopped_while_its_continuation_waits_is_sent_the_kept_reasoning() {
        let ops = ops_with(Vec::new());
        let reasoning = prose(5000) + &looping(6000);
        let running = Running::new(1);
        let p = LoopGuardParams { holdback_chars: 6000, ..params(LoopGuardMode::Rewind) };
        let mut out = guard_counted(p, Stalled(ops.clone()), input("P", 40_000), running.clone(), scripted(1, &reasoning, false), Rewound::default(), None);
        until("no rewind", || !ops.submitted.lock().unwrap().is_empty()).await;
        let kept = text_of(&held_back(&running, &mut out).await);
        let continued = text_of(&ops.submitted.lock().unwrap()[0].tokens);
        assert!(continued.strip_prefix('P').unwrap().starts_with(&kept) && kept.len() > 1000, "kept {} chars", kept.len());
        assert!(prose(5000).starts_with(&kept), "none of the loop");
        until("the guard still waits", || Arc::strong_count(&ops) == 1).await;
    }

    /// With no guard on the request, stopping it holds nothing back.
    #[tokio::test]
    async fn a_request_no_guard_watches_holds_nothing_back() {
        let running = Running::new(1);
        let (_tx, rx) = unbounded_channel();
        let mut out = guard_counted(params(LoopGuardMode::Off), ops_with(Vec::new()), input("P", 40_000), running.clone(), rx, Rewound::default(), None);
        assert!(!running.stop());
        assert!(held_back(&running, &mut out).await.is_empty(), "nothing read from the engine's own stream");
    }

    // ── history repair on the guard's own streams ──────────────────────

    #[tokio::test]
    async fn repair_gives_back_the_models_view_after_one_rewind() {
        let after = "\nRome's story, written.</think>\n\nRoma.";
        let ops = ops_with(vec![(2, scripted(2, after, true))]);
        let reasoning = prose(5000) + &looping(6000);
        let (received, _) =
            drain_tokens(guard(params(LoopGuardMode::Rewind), ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
        assert_eq!(text_of(&received).matches(TAG_PREFIX).count(), 1);
        assert_repair_gives_the_models_view(&received, &ops, "P", &tokens_of(after));
    }

    /// Two rewinds, the second cut at the first nudge's end; and a nudge
    /// then the close, which ends the reasoning.
    #[tokio::test]
    async fn repair_gives_back_the_models_view_after_two_rewinds_and_after_a_close() {
        for retries in [2, 1] {
            let ops = ops_with(vec![(3, scripted(3, "Roma.", true)), (2, scripted(2, &looping(4000), false))]);
            let reasoning = prose(5000) + &looping(6000);
            let p = LoopGuardParams { retries, ..params(LoopGuardMode::Rewind) };
            let (received, _) = drain_tokens(guard(p, ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
            let text = text_of(&received);
            assert_eq!(text.matches(TAG_PREFIX).count(), 2, "retries {retries}");
            let close = crate::thinking::THINKING_CLOSE_TEXT.trim_start_matches('\n');
            assert_eq!(text.contains(close), retries == 1, "retries {retries}");
            assert_repair_gives_the_models_view(&received, &ops, "P", &tokens_of("Roma."));
        }
    }

    /// The token the guard fires on ends in a line break the decoder holds:
    /// the client shows it once the tag arrives, so the tag's `m` counts it,
    /// and repair still gives back the model's view exactly.
    #[tokio::test]
    async fn a_line_break_held_when_the_guard_fires_is_struck_with_the_rest() {
        let reasoning = prose(5000) + &looping(6000);
        let mut tokens = tokens_of(&reasoning);
        // Where the guard fires: the first token the client gets that the
        // reasoning does not have is the tag's.
        let (plain, _) =
            drain_tokens(guard(params(LoopGuardMode::Rewind), ops_with(vec![(2, scripted(2, "Roma.", true))]), input("P", 40_000), 1, scripted(1, &reasoning, false)))
                .await;
        let fired = plain.iter().zip(&tokens).position(|(a, b)| a != b).expect("a tag");
        let fired_on = char::from_u32(tokens[fired - 1]).unwrap();
        tokens[fired - 1] += NEWLINE_AFTER;

        let ops = ops_with(vec![(2, scripted(2, "Roma.", true))]);
        let (received, _) =
            drain_tokens(guard(params(LoopGuardMode::Rewind), ops.clone(), input("P", 40_000), 1, scripted_tokens(1, &tokens, false))).await;
        assert_eq!(received[fired - 1], tokens[fired - 1], "the guard fired on the same token");
        let shown = client_reasoning(&received);
        let (m, struck) = first_struck(&shown);
        assert!(struck.ends_with(&format!("{fired_on}\n")), "the held line break is struck: {struck:?}");
        let (_, rest) = shown.split_once(TAG_OPEN).unwrap();
        assert!(rest.starts_with(&format!("{} TOKENS / {m} CHARS \u{b7} {}]", m - 1, struck_hash(&struck))), "one token for two characters");
        assert_repair_gives_the_models_view(&received, &ops, "P", &tokens_of("Roma."));
    }

    // ── holdback ─────────────────────────────────────────────────────────

    /// A cut inside what the client was sent: the tag strikes only that, the
    /// rest it never received is dropped, and the counts say so.
    #[tokio::test]
    async fn with_a_holdback_a_cut_in_what_was_sent_strikes_only_that() {
        let ops = ops_with(vec![(2, scripted(2, "Roma.", true))]);
        let reasoning = prose(5000) + &looping(6000);
        let p = LoopGuardParams { holdback_chars: 500, ..params(LoopGuardMode::Rewind) };
        let rewound = Rewound::default();
        let (received, done) =
            drain_tokens(guard_counted(p, ops.clone(), input("P", 40_000), Running::new(1), scripted(1, &reasoning, false), rewound.clone(), None)).await;
        let text = text_of(&received);
        let (sent, _) = text.split_once(TAG_OPEN).expect("the cut is in what was sent: a tag");
        let kept = kept_then(&text_of(&models_view(&ops, "P", &[])), NUDGES[0], &reasoning).0.chars().count();
        let generated = kept + rewound.taken() as usize;
        assert!(reasoning.starts_with(sent));
        assert_eq!(sent.chars().count(), generated - rewound.unsent() as usize, "the client got all it was sent, then the tag");
        assert!((499..=501).contains(&rewound.unsent()), "about the holdback was never sent: {}", rewound.unsent());
        let (m, struck) = first_struck(&client_reasoning(&received));
        assert_eq!((m, struck.as_str()), (sent.chars().count() - kept, &sent[kept..]), "only what was sent is struck");
        let Some(SchedEvent::Done { tokens, .. }) = done else { panic!("no done") };
        assert_eq!(tokens as usize, received.len());
        assert_repair_gives_the_models_view(&received, &ops, "P", &tokens_of("Roma."));
    }

    /// A cut inside what is still held back: no tag, the loop never reaches
    /// the client, and what it receives is the model's view token for token.
    /// The continuation's indices land where its tokens do.
    #[tokio::test]
    async fn with_a_holdback_past_the_cut_the_client_never_sees_the_loop() {
        let after = "Roma, written.";
        let (cont_tx, cont_rx) = unbounded_channel();
        for token in tokens_of(after) {
            cont_tx.send(SchedEvent::Token { request: 2, token }).unwrap();
        }
        let outcome = ignis_core::thinking_budget::BudgetOutcome { budget: 100, forced_at: Some(3), closed_at: Some(4) };
        cont_tx.send(done_with(2, after.len() as u32, FinishReason::Stop, Some(outcome), Some(2))).unwrap();
        let ops = ops_with(vec![(2, cont_rx)]);
        let reasoning = prose(5000) + &looping(6000);
        let p = LoopGuardParams { holdback_chars: 6000, ..params(LoopGuardMode::Rewind) };
        let rewound = Rewound::default();
        let (received, done) =
            drain_tokens(guard_counted(p, ops.clone(), input("P", 40_000), Running::new(1), scripted(1, &reasoning, false), rewound.clone(), None)).await;
        assert_eq!(ops.submitted.lock().unwrap().len(), 1, "rewound");
        assert!(!text_of(&received).contains(TAG_PREFIX), "no tag");
        assert_eq!(received, models_view(&ops, "P", &tokens_of(after)), "the model's view, token for token");
        assert!(matches!(repair(&client_reasoning(&received)), Cow::Borrowed(_)), "nothing to repair");
        assert!(rewound.taken() > 0 && rewound.unsent() == rewound.taken(), "every struck token was held back");
        let Some(SchedEvent::Done { tokens, thinking, reasoning_redirected_at, .. }) = done else { panic!("no done") };
        let before = (received.len() - after.len()) as u32;
        assert_eq!(tokens as usize, received.len());
        let thinking = thinking.unwrap();
        assert_eq!((thinking.forced_at, thinking.closed_at, reasoning_redirected_at), (Some(before + 3), Some(before + 4), Some(before + 2)));
    }

    /// The client stays `holdback` characters behind, and gets everything
    /// at the reasoning's close; in `log` mode nothing is held.
    #[tokio::test]
    async fn a_holdback_keeps_the_client_behind_until_the_reasoning_closes() {
        let reasoning = prose(3000);
        for (mode, behind) in [(LoopGuardMode::Rewind, 1000), (LoopGuardMode::Log, 0)] {
            let (tx, rx) = unbounded_channel();
            for token in tokens_of(&reasoning) {
                tx.send(SchedEvent::Token { request: 1, token }).unwrap();
            }
            let p = LoopGuardParams { holdback_chars: 1000, ..params(mode) };
            let mut out = guard(p, ops_with(Vec::new()), input("P", 40_000), 1, rx);
            let mut got = Vec::new();
            received_by(&tx, &mut out, &mut got).await;
            let lag = 3000 - got.len();
            assert!(lag.abs_diff(behind) <= 1, "{mode:?}: {lag} characters behind");
            assert!(reasoning.starts_with(&text_of(&got)));
            for token in tokens_of("</think>answer") {
                tx.send(SchedEvent::Token { request: 1, token }).unwrap();
            }
            received_by(&tx, &mut out, &mut got).await;
            assert_eq!(text_of(&got), format!("{reasoning}</think>answer"), "{mode:?}: all of it at the close");
        }
    }

    /// Everything held back goes out ahead of the `Done`, an error's
    /// included, and when the engine's stream ends with none.
    #[tokio::test]
    async fn a_holdback_lets_everything_go_at_the_end_and_on_an_error() {
        let reasoning = prose(3000);
        let p = LoopGuardParams { holdback_chars: 1000, ..params(LoopGuardMode::Rewind) };
        let (text, done) = drain(guard(p.clone(), ops_with(Vec::new()), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
        assert_eq!((text.as_str(), done.is_none()), (reasoning.as_str(), true), "the stream ended with no Done");
        let (tx, rx) = unbounded_channel();
        for token in tokens_of(&reasoning) {
            tx.send(SchedEvent::Token { request: 1, token }).unwrap();
        }
        tx.send(done_with(1, 3000, FinishReason::Error, None, None)).unwrap();
        let (text, done) = drain(guard(p, ops_with(Vec::new()), input("P", 40_000), 1, rx)).await;
        assert_eq!(text, reasoning);
        let Some(SchedEvent::Done { reason, tokens, .. }) = done else { panic!("no done") };
        assert_eq!((reason, tokens), (FinishReason::Error, 3000));
    }

    /// The guard on `stream` counting on a fresh [`Metrics`] (GitHub #330),
    /// drained: what the client got, and the projection once the guard ended.
    async fn metered(p: LoopGuardParams, ops: impl GuardOps, input: RequestInput, stream: EventStream) -> (String, Arc<Metrics>) {
        let metrics = Arc::new(Metrics::new());
        let out = guard_counted(p, ops, input, Running::new(1), stream, Rewound::default(), Some(Arc::clone(&metrics)));
        let (text, _) = drain(out).await;
        (text, metrics)
    }

    /// `series`' sample in `metrics`' exposition, when it has one.
    fn sample(metrics: &Metrics, series: &str) -> Option<u64> {
        metrics.render().lines().find_map(|line| line.strip_prefix(series)?.strip_prefix(' ')?.parse().ok())
    }

    fn loops(metrics: &Metrics, outcome: &str) -> u64 {
        sample(metrics, &format!("ignis_loop_guard_loops_total{{outcome=\"{outcome}\"}}")).expect("the loop family")
    }

    /// GitHub #330: every loop the guard sees counts once by what it did
    /// about it, and the request it is in counts as looped once however
    /// often it looped.
    #[tokio::test]
    async fn every_loop_is_counted_by_its_outcome_and_its_request_as_looped_once() {
        let reasoning = prose(5000) + &looping(6000);
        let rewind = params(LoopGuardMode::Rewind);
        let roma = || ops_with(vec![(2, scripted(2, "Roma.", true))]);

        // Rewound: the tokens it took back are the usage's.
        let (metrics, rewound) = (Arc::new(Metrics::new()), Rewound::default());
        let out = guard_counted(rewind.clone(), roma(), input("P", 40_000), Running::new(1), scripted(1, &reasoning, false), rewound.clone(), Some(Arc::clone(&metrics)));
        drain(out).await;
        assert_eq!((loops(&metrics, "rewound"), loops(&metrics, "closed")), (1, 0));
        assert!(rewound.taken() > 0);
        assert_eq!(sample(&metrics, "ignis_loop_guard_rewound_tokens_total"), Some(u64::from(rewound.taken())));
        assert_eq!(sample(&metrics, "ignis_loop_guard_watched_requests_total"), Some(1));
        assert_eq!(sample(&metrics, "ignis_loop_guard_looped_requests_total"), Some(1));

        // Closed, its nudges spent.
        let (_, metrics) = metered(LoopGuardParams { retries: 0, ..rewind.clone() }, roma(), input("P", 40_000), scripted(1, &reasoning, false)).await;
        assert_eq!((loops(&metrics, "rewound"), loops(&metrics, "closed")), (0, 1));
        assert!(sample(&metrics, "ignis_loop_guard_rewound_tokens_total").unwrap() > 0, "a close takes the loop back too");

        // Logged, every time the loop goes on: one looped request.
        let (_, metrics) = metered(params(LoopGuardMode::Log), roma(), input("P", 40_000), scripted(1, &reasoning, true)).await;
        assert!(loops(&metrics, "detected") > 1, "{}", metrics.render());
        assert_eq!(sample(&metrics, "ignis_loop_guard_looped_requests_total"), Some(1));
        assert_eq!(sample(&metrics, "ignis_loop_guard_rewound_tokens_total"), Some(0));

        // Held: nothing before the loop to keep.
        let (_, metrics) = metered(rewind.clone(), roma(), input("P", 40_000), scripted(1, &looping(8000), true)).await;
        assert_eq!(loops(&metrics, "too_little_kept"), 1);
        assert_eq!(loops(&metrics, "rewound"), 0);

        // No tokenizer for the nudge.
        let (_, metrics) = metered(rewind.clone(), NoTokenizer(roma()), input("P", 40_000), scripted(1, &reasoning, true)).await;
        assert!(loops(&metrics, "no_tokenizer") >= 1);

        // No room left: `max_tokens` one short of a continuation's worth.
        let ops = roma();
        drain(guard(rewind.clone(), ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
        let used = 40_000 - ops.submitted.lock().unwrap()[0].params.max_tokens.unwrap();
        let (_, metrics) = metered(rewind.clone(), roma(), input("P", used + DEFAULT_MIN_CONTINUATION_TOKENS - 1), scripted(1, &reasoning, true)).await;
        assert!(loops(&metrics, "no_room") >= 1);
        assert_eq!(loops(&metrics, "rewound"), 0);

        // Cancelled, and its continuation refused.
        let (_, metrics) = metered(rewind.clone(), ops_with(Vec::new()), input("P", 40_000), scripted(1, &reasoning, true)).await;
        assert!(loops(&metrics, "submit_refused") >= 1);
        assert_eq!((loops(&metrics, "rewound"), sample(&metrics, "ignis_loop_guard_rewound_tokens_total")), (0, Some(0)));
        assert_eq!(sample(&metrics, "ignis_loop_guard_looped_requests_total"), Some(1));

        // Off: no guard, nothing counted.
        let (_, metrics) = metered(params(LoopGuardMode::Off), roma(), input("P", 40_000), scripted(1, &reasoning, true)).await;
        assert!(!metrics.render().contains("ignis_loop_guard_"));
    }

    /// GitHub #330: a watched request's lowest score goes into the margin
    /// histogram once, before its `Done` reaches the client: prose stays
    /// well above the threshold, a loop's request goes under it, and a
    /// request too short for a window is watched and never scored.
    #[tokio::test]
    async fn a_watched_request_s_lowest_score_is_observed_once_before_its_done() {
        let metrics = Arc::new(Metrics::new());
        let mut out = guard_counted(params(LoopGuardMode::Rewind), ops_with(Vec::new()), input("P", 40_000), Running::new(1), scripted(1, &prose(5000), true), Rewound::default(), Some(Arc::clone(&metrics)));
        while !matches!(out.recv().await.expect("a done"), SchedEvent::Done { .. }) {}
        assert_eq!(sample(&metrics, "ignis_loop_guard_min_ratio_count"), Some(1), "observed by the time the Done arrives");
        assert_eq!(sample(&metrics, "ignis_loop_guard_min_ratio_bucket{le=\"0.4\"}"), Some(0), "prose scores 0.45-0.49");
        assert_eq!(sample(&metrics, "ignis_loop_guard_looped_requests_total"), Some(0));
        assert!(out.recv().await.is_none());
        assert_eq!(sample(&metrics, "ignis_loop_guard_min_ratio_count"), Some(1), "once");

        let reasoning = prose(5000) + &looping(6000);
        let (_, metrics) = metered(params(LoopGuardMode::Rewind), ops_with(vec![(2, scripted(2, "Roma.", true))]), input("P", 40_000), scripted(1, &reasoning, false)).await;
        assert_eq!(sample(&metrics, "ignis_loop_guard_min_ratio_count"), Some(1), "one request, across its rewind");
        assert_eq!(sample(&metrics, "ignis_loop_guard_min_ratio_bucket{le=\"0.25\"}"), Some(1));

        let (_, metrics) = metered(params(LoopGuardMode::Rewind), ops_with(Vec::new()), input("P", 40_000), scripted(1, &prose(500), true)).await;
        assert_eq!(sample(&metrics, "ignis_loop_guard_watched_requests_total"), Some(1));
        assert_eq!(sample(&metrics, "ignis_loop_guard_min_ratio_count"), Some(0));
    }

    /// GitHub #330: a loop counts as soon as the guard sees it; its request
    /// counts, watched and looped together, only when it ends, so a window
    /// never reads more looped requests than watched ones.
    #[tokio::test]
    async fn a_loop_counts_at_once_and_its_request_when_it_ends() {
        let (continuation, rx) = unbounded_channel();
        let metrics = Arc::new(Metrics::new());
        let reasoning = prose(5000) + &looping(6000);
        let ops = ops_with(vec![(2, rx)]);
        let out = guard_counted(params(LoopGuardMode::Rewind), ops, input("P", 40_000), Running::new(1), scripted(1, &reasoning, false), Rewound::default(), Some(Arc::clone(&metrics)));
        until("the rewind counted", || sample(&metrics, "ignis_loop_guard_loops_total{outcome=\"rewound\"}") == Some(1)).await;
        assert_eq!(sample(&metrics, "ignis_loop_guard_watched_requests_total"), Some(0), "still generating");
        assert_eq!(sample(&metrics, "ignis_loop_guard_looped_requests_total"), Some(0));
        continuation.send(done_event(2, 0)).unwrap();
        drain(out).await;
        assert_eq!(sample(&metrics, "ignis_loop_guard_watched_requests_total"), Some(1));
        assert_eq!(sample(&metrics, "ignis_loop_guard_looped_requests_total"), Some(1));
    }
}
