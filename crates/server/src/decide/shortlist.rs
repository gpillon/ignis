//! The **shortlist** (spec `docs/specs/decide/22-locate-by-copy-over-a-folded-state.md`,
//! GitHub #278, ADR 0042): a `locate` whose heads narrow a very long text to
//! a few candidates and whose labelled `choice` decides among them, with no
//! token generated.
//!
//! Three routes, by the kind and the compression a question resolved to:
//!
//! - **folded** (`log` + `template_fold`, and `records` + `template_fold`
//!   over the records' spaced JSON): the target's segments with content are
//!   folded ([`ignis_core::locate::fold`]); past five templates the end
//!   heads read the level-1 text and keep five; a labelled `choice` picks
//!   one (skipped when there is one); past sixteen of its level-2 rows the
//!   end heads read them and keep sixteen; a last labelled `choice` over
//!   **those rows' original lines** picks the line;
//! - **read** (`log`, `prose` or `records` + `none`): the heads read the
//!   whole target — window by window past `LOCATE_WINDOW_KEYS` — the end
//!   heads for a log or records, the sum heads for prose, and keep sixteen;
//!   a last labelled `choice` over them — prose inside its paragraphs,
//!   records as their spaced JSON, log lines as sent;
//! - in the same request as the last `choice`, on the three measured routes,
//!   that `choice` again with a "none" option and, for a folded log, a
//!   yes/no: [`ignis_core::locate::found_log`] and
//!   [`ignis_core::locate::found_by_none`].
//!
//! Every step before the first prefill — the target, the fold, every
//! reading prompt a question can be asked before it has read anything — is
//! rendered while the request can still be refused ([`plan`]). A step that
//! depends on an earlier one's answer (a fold's level 2, every `choice`)
//! is rendered after it, and a fault there is that question's error, not a
//! 422: the siblings' prefills are already spent.
//!
//! A heads reading is the vote's render and readout ([`super::render_reading`])
//! over a **window**: the whole state when the target fits one, else the
//! window's segments alone, as their own state. Each window is read with
//! its content-free baseline, shared by every question over it; the baseline goes
//! first and every question over the window claims the prefix it keeps.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::Arc;

use ignis_core::locate::fold::{Fold, fold};
use ignis_core::locate::reading::{Merge, Reading, cut_windows, merge, shortlist, window_scores};
use ignis_core::locate::render::{in_paragraphs, labelled, rows_first_lines};
use ignis_core::locate::{
    Compression, FOUND_QUESTION, FOUND_THRESHOLD, Kind, LocateCalibration, NONE_LINE, NONE_SENTENCE, POINTER_SHARE,
    RANKING_LEN, SHORTLIST_LEN, SHORTLIST_TEMPLATES, found_by_none, found_log,
};
use ignis_core::pointing::AttentionScores;

use super::{
    Answer, Collect, Evidence, LocateCompression, LocateKind, LocateMethod, LocatePointer, LocatePrompt, LocateRank,
    OrderedValue, PreparedOption, PreparedQuestion, QuestionKind, Refusal, Rendered, Reply, failed, in_waves, led,
    render, render_reading, too_few_segments, within_context,
};

/// Keys a window leaves unused below `LOCATE_WINDOW_KEYS` — at most an
/// eighth of it: a window rendered alone can tokenize its edges a little
/// differently from the whole text it was cut from.
const WINDOW_MARGIN: u32 = 256;

/// A text the heads read, cut into windows (spec 22 § Windows), each with
/// its content-free baseline — shared by every question that reads it.
struct Text {
    reading: Reading,
    merge: Merge,
    heads: usize,
    /// The segments of the text, windowed or not.
    segments: usize,
    windows: Vec<Window>,
}

/// One window of a [`Text`]: the state rendered for it, which segments of
/// the text it holds, and where they sit.
struct Window {
    first: usize,
    state: OrderedValue,
    within: String,
    span: Range<usize>,
    keys: Vec<Option<Range<usize>>>,
    /// The baseline's prompt until it is asked, then its rows.
    baseline: Option<Rendered>,
    rows: Option<Result<AttentionScores, Answer>>,
}

/// Every text this request's shortlists read, and every fold, once.
#[derive(Default)]
pub(super) struct Shared {
    texts: Vec<Text>,
    by_key: HashMap<String, usize>,
    folds: HashMap<String, Arc<Folded>>,
}

/// A target folded (spec 22 § `template_fold`), once per `within`.
struct Folded {
    fold: Fold,
    /// The folded lines: each segment with content, as text.
    lines: Vec<String>,
    /// Each folded line's segment of the target.
    segment: Vec<usize>,
}

/// One shortlist `locate`, from its plan to its answer.
pub(super) struct Planned {
    pub(super) slot: usize,
    id: String,
    instruction: OrderedValue,
    within: String,
    kind: Kind,
    compression: Compression,
    /// The target's segments as the caller sent them, and as text.
    values: Vec<OrderedValue>,
    texts: Vec<String>,
    steps: Steps,
    /// The text read before anything is answered — the target, or a fold's
    /// level 1 — and this question's prompt per window of it, until asked.
    first: Option<usize>,
    prompts: Vec<Option<Rendered>>,
    /// A fold's level-2 text, when its chosen template has too many rows.
    second: Option<usize>,
    /// This question's rows, by (text, window).
    rows: HashMap<(usize, usize), Result<AttentionScores, Answer>>,
    /// Prompt tokens this question's steps spent.
    pub(super) tokens: u32,
    failure: Option<Answer>,
    stats: Stats,
}

enum Steps {
    Folded(Arc<Folded>),
    Read,
}

/// What the request log says of one shortlist (spec 22 § Observability).
#[derive(Default)]
pub(super) struct Stats {
    /// Windows read, over every text the question's readings read.
    pub(super) windows: usize,
    /// A fold's templates, the one its level-1 `choice` picked, and that
    /// template's level-2 rows.
    pub(super) templates: Option<usize>,
    pub(super) template: Option<usize>,
    pub(super) rows: Option<usize>,
    /// Each `choice`'s candidates: a fold's templates, then segments.
    pub(super) candidates: Vec<Vec<usize>>,
    pub(super) p_none: Option<f64>,
    pub(super) p_yes: Option<f64>,
    /// The plan's host time — the fold, when this question made it, and
    /// every render asked before the first prefill — in milliseconds.
    pub(super) plan_ms: f64,
}

impl Planned {
    fn reading(&self) -> Reading {
        match self.kind {
            Kind::Prose => Reading::Sum,
            Kind::Log | Kind::Records => Reading::End,
        }
    }

    fn merge(&self) -> Merge {
        match self.kind {
            Kind::Prose => Merge::Prose,
            Kind::Log | Kind::Records => Merge::Records,
        }
    }

    /// Whether this route carries `found` (spec 22 § Not found): the three
    /// measured ones.
    fn finds(&self) -> bool {
        matches!(
            (self.kind, self.compression),
            (Kind::Log, Compression::TemplateFold) | (Kind::Prose, Compression::None) | (Kind::Records, Compression::None)
        )
    }

    fn fail(&mut self, answer: Answer) {
        if self.failure.is_none() {
            self.failure = Some(answer);
        }
    }
}

/// A text shown in a labelled line: a string element may hold a line break,
/// which would split its line in two.
fn one_line(text: &str) -> String {
    text.replace('\n', " ")
}

/// The content-free instruction a baseline is read with.
fn content_free() -> OrderedValue {
    OrderedValue::String(crate::locate::CONTENT_FREE.to_owned())
}

/// Plan one shortlist `locate` (GitHub #278): its target, its fold, and every
/// reading prompt it is asked before it has read anything — refused, like
/// every fault a caller can commit, before the first prefill.
#[allow(clippy::too_many_arguments)]
pub(super) async fn plan(
    server: &crate::Server,
    state: &OrderedValue,
    question: &PreparedQuestion,
    slot: usize,
    kind: Kind,
    compression: Compression,
    calibration: LocateCalibration,
    model: Option<String>,
    shared: &mut Shared,
) -> Result<Planned, Refusal> {
    let started = std::time::Instant::now();
    let id = question.id.as_str();
    let within = question.within.clone().unwrap_or_default();
    // A last `choice` names up to sixteen candidates and "none".
    if server.active().alphabet.len() < SHORTLIST_LEN + 1 {
        return Err(Refusal::new(
            "alphabet_exhausted",
            format!(
                "question {id:?}: a locate's labelled `choice` names up to {} options, and this model's tokenizer can name only {}",
                SHORTLIST_LEN + 1,
                server.active().alphabet.len()
            ),
        ));
    }
    let target = crate::locate::target_within(state, &within)
        .map_err(|error| Refusal::new(error.code(), format!("question {id:?}: {error}")))?;
    let values = match target {
        OrderedValue::String(text) => text.split('\n').map(|line| OrderedValue::String(line.to_owned())).collect(),
        OrderedValue::Array(items) => items.clone(),
        _ => Vec::new(),
    };
    let texts = crate::locate::segment_texts(target);
    let mut planned = Planned {
        slot,
        id: id.to_owned(),
        instruction: question.instructions.clone(),
        within: within.clone(),
        kind,
        compression,
        values,
        texts,
        steps: Steps::Read,
        first: None,
        prompts: Vec::new(),
        second: None,
        rows: HashMap::new(),
        tokens: 0,
        failure: None,
        stats: Stats::default(),
    };
    match compression {
        Compression::TemplateFold => {
            let folded = match shared.folds.get(&within) {
                Some(folded) => folded.clone(),
                None => {
                    let content = crate::locate::content_flags(target);
                    let segment: Vec<usize> = (0..planned.texts.len()).filter(|&i| content[i]).collect();
                    too_few_segments(id, &segment.iter().map(|&i| Some(i..i + 1)).collect::<Vec<_>>(), 2)?;
                    let lines: Vec<String> = segment.iter().map(|&i| planned.texts[i].clone()).collect();
                    let folded = Arc::new(Folded { fold: fold(&lines), lines, segment });
                    shared.folds.insert(within.clone(), folded.clone());
                    folded
                }
            };
            planned.stats.templates = Some(folded.fold.clusters.len());
            if folded.fold.clusters.len() > SHORTLIST_TEMPLATES {
                let lines: Vec<String> = folded.fold.level1.iter().map(|line| one_line(line)).collect();
                let key = format!("level1\u{0}{within}");
                let text = open_level(server, shared, key, &lines, true, calibration, model.clone(), id).await?;
                planned.prompts = prompts_for(server, &shared.texts[text], &planned.instruction, calibration, model, id).await?;
                planned.first = Some(text);
            }
            planned.steps = Steps::Folded(folded);
        }
        Compression::None => {
            let key = format!("read\u{0}{within}\u{0}{}", kind.label());
            let text = match shared.by_key.get(&key) {
                Some(&text) => text,
                None => {
                    let reading = planned.reading();
                    let opened =
                        open_text(server, state, &within, reading, planned.merge(), 2, false, calibration, model.clone(), id)
                            .await?;
                    shared.texts.push(opened);
                    shared.by_key.insert(key, shared.texts.len() - 1);
                    shared.texts.len() - 1
                }
            };
            planned.prompts = prompts_for(server, &shared.texts[text], &planned.instruction, calibration, model, id).await?;
            planned.first = Some(text);
        }
    }
    planned.stats.plan_ms = started.elapsed().as_secs_f64() * 1e3;
    Ok(planned)
}

/// A fold's level-1 or level-2 text, opened once per key: its lines as one
/// string state, read by the end heads. Level 1 is refused past the context
/// (spec 22 § Refusals); a level-2 text past a window is windowed.
#[allow(clippy::too_many_arguments)]
async fn open_level(
    server: &crate::Server,
    shared: &mut Shared,
    key: String,
    lines: &[String],
    within_the_context: bool,
    calibration: LocateCalibration,
    model: Option<String>,
    id: &str,
) -> Result<usize, Refusal> {
    if let Some(&text) = shared.by_key.get(&key) {
        return Ok(text);
    }
    let state = OrderedValue::String(lines.join("\n"));
    let opened =
        open_text(server, &state, "", Reading::End, Merge::Records, 1, within_the_context, calibration, model, id).await?;
    shared.texts.push(opened);
    shared.by_key.insert(key, shared.texts.len() - 1);
    Ok(shared.texts.len() - 1)
}

/// Open a text the heads read (spec 22 § Windows): its content-free baseline
/// over the whole target, and — when that spans more than one window — the
/// target cut into windows at segment boundaries, each rendered alone with
/// its own baseline. A window is `LOCATE_WINDOW_KEYS` at most, and less on a
/// load whose context would not hold that many keys with the prompt around
/// them: the text's length is bounded by the request, not the context.
/// `within_the_context` refuses a text whose whole render is past the
/// context instead of windowing it.
#[allow(clippy::too_many_arguments)]
async fn open_text(
    server: &crate::Server,
    state: &OrderedValue,
    within: &str,
    reading: Reading,
    merge_rule: Merge,
    least: usize,
    within_the_context: bool,
    calibration: LocateCalibration,
    model: Option<String>,
    id: &str,
) -> Result<Text, Refusal> {
    let heads = calibration.heads_for(reading);
    let whole = render_reading(server, state, within, &content_free(), id, model.clone(), heads).await?;
    too_few_segments(id, &whole.keys, least)?;
    if within_the_context {
        within_context(server, id, &whole)?;
    }
    let segments = whole.keys.len();
    let around = (whole.ready.prompt_tokens as usize).saturating_sub(whole.span.len());
    let window = (calibration.window_keys as usize).min((server.active().engine.max_model_len() as usize).saturating_sub(around));
    if whole.span.len() <= window {
        within_context(server, id, &whole)?;
        let LocatePrompt { ready, span, keys, .. } = whole;
        let window = Window { first: 0, state: state.clone(), within: within.to_owned(), span, keys, baseline: Some(ready), rows: None };
        return Ok(Text { reading, merge: merge_rule, heads: heads.len(), segments, windows: vec![window] });
    }
    // Each segment's keys in the whole render, its separator included: what
    // a window of it will cost.
    let mut costs = vec![0u64; segments];
    let mut next = whole.span.len();
    let owned: Vec<bool> = whole.keys.iter().map(Option::is_some).collect();
    for (index, keys) in whole.keys.iter().enumerate().rev() {
        if let Some(keys) = keys {
            costs[index] = (next - keys.start) as u64;
            next = keys.start;
        }
    }
    drop(whole);
    let target = crate::locate::target_within(state, within)
        .map_err(|error| Refusal::new(error.code(), format!("question {id:?}: {error}")))?;
    let empty: Vec<bool> = match target {
        OrderedValue::String(text) => text.split('\n').map(str::is_empty).collect(),
        OrderedValue::Array(items) => items.iter().map(|item| item.as_str() == Some("")).collect(),
        _ => vec![false; segments],
    };
    let margin = WINDOW_MARGIN.min(window as u32 / 8) as usize;
    let budget = (window - margin) as u64;
    let lines: Vec<&str> = match target {
        OrderedValue::String(text) => text.split('\n').collect(),
        _ => Vec::new(),
    };
    let mut windows = Vec::new();
    for range in cut_windows(&costs, &empty, budget) {
        // A run of empty segments reads nothing.
        if !owned[range.clone()].iter().any(|&owns| owns) {
            continue;
        }
        let window_state = match target {
            OrderedValue::String(_) => OrderedValue::String(lines[range.clone()].join("\n")),
            OrderedValue::Array(items) => OrderedValue::Array(items[range.clone()].to_vec()),
            _ => unreachable!("a segmented target is a string or an array"),
        };
        let baseline = render_reading(server, &window_state, "", &content_free(), id, model.clone(), heads).await?;
        if baseline.span.len() > window {
            // A window holds one segment past the budget only when that
            // segment alone is: nothing smaller can be cut from it.
            return Err(Refusal::new(
                "locate_segment_too_long",
                format!(
                    "question {id:?}: {} of the target span {} keys, past the {} a `locate` reads in one window, so it cannot be read",
                    match range.len() {
                        1 => format!("segment {}", range.start),
                        _ => format!("segments {}..{}", range.start, range.end),
                    },
                    baseline.span.len(),
                    window
                ),
            ));
        }
        within_context(server, id, &baseline)?;
        debug_assert_eq!(baseline.keys.len(), range.len());
        let LocatePrompt { ready, span, keys, .. } = baseline;
        windows.push(Window { first: range.start, state: window_state, within: String::new(), span, keys, baseline: Some(ready), rows: None });
    }
    Ok(Text { reading, merge: merge_rule, heads: heads.len(), segments, windows })
}

/// A question's reading prompt over every window of `text`, each checked
/// against its window's baseline: the two share every byte up to the
/// instruction, so a question whose keys are not its baseline's is answered
/// with that error and never submitted.
async fn prompts_for(
    server: &crate::Server,
    text: &Text,
    instruction: &OrderedValue,
    calibration: LocateCalibration,
    model: Option<String>,
    id: &str,
) -> Result<Vec<Option<Rendered>>, Refusal> {
    let heads = calibration.heads_for(text.reading);
    let mut prompts = Vec::with_capacity(text.windows.len());
    for window in &text.windows {
        let mut prompt = render_reading(server, &window.state, &window.within, instruction, id, model.clone(), heads).await?;
        within_context(server, id, &prompt)?;
        if (&prompt.span, &prompt.keys) != (&window.span, &window.keys) {
            prompt.ready.answered = Some(failed(
                "locate_baseline_misaligned",
                "the content-free baseline's prompt maps the text onto other keys than this question's, so the two cannot be read against each other".to_owned(),
            ));
        }
        prompts.push(Some(prompt.ready));
    }
    Ok(prompts)
}

/// Put a text's windows to the engine, window by window: each window's baseline
/// (if not read yet) leading, then the prompt each question in `prompts`
/// holds for it, which claim the prefix the baseline kept. Every question's rows
/// land in its own map, by (text, window); the prompt tokens spent are
/// returned.
async fn read_windows(
    server: &crate::Server,
    shared: &mut Shared,
    text: usize,
    questions: &mut [Planned],
    mut prompts: Vec<(usize, Vec<Option<Rendered>>)>,
    class: ignis_core::types::RequestClass,
) -> u32 {
    let mut spent = 0u32;
    let windows = shared.texts[text].windows.len();
    for w in 0..windows {
        // Slot 0 is the baseline, slot 1 + i the i-th question's prompt.
        let mut items = Vec::new();
        if let Some(baseline) = shared.texts[text].windows[w].baseline.take() {
            spent = spent.saturating_add(baseline.prompt_tokens);
            items.push((0usize, baseline, Collect::Rows));
        }
        for (i, (who, per_window)) in prompts.iter_mut().enumerate() {
            if let Some(ready) = per_window.get_mut(w).and_then(Option::take) {
                spent = spent.saturating_add(ready.prompt_tokens);
                questions[*who].tokens = questions[*who].tokens.saturating_add(ready.prompt_tokens);
                items.push((1 + i, ready, Collect::Rows));
            }
        }
        for (slot, reply) in led(server, items, class).await {
            let rows = rows_of(reply);
            match slot {
                0 => shared.texts[text].windows[w].rows = Some(rows),
                i => {
                    questions[prompts[i - 1].0].rows.insert((text, w), rows);
                }
            }
        }
    }
    spent
}

/// A reading prompt's reply as rows, or the failure that stands in for them.
fn rows_of(reply: Reply) -> Result<AttentionScores, Answer> {
    match reply {
        Reply::Rows(rows) => rows,
        Reply::Answer(answer) => Err(answer),
    }
}

/// A question's merged reading of `text`: every window's rows read against
/// its baseline's ([`window_scores`]), merged ([`merge`]); `None` for a segment
/// no window read. Or the failure of the first window that did not read.
fn scores_of(
    text: &Text,
    index: usize,
    rows: &HashMap<(usize, usize), Result<AttentionScores, Answer>>,
) -> Result<(Vec<f64>, Vec<bool>), Answer> {
    let mut windows = Vec::with_capacity(text.windows.len());
    let mut owned = vec![false; text.segments];
    for (w, window) in text.windows.iter().enumerate() {
        let asked = match rows.get(&(index, w)) {
            Some(Ok(asked)) => asked,
            Some(Err(failure)) => return Err(failure.clone()),
            None => return Err(failed("not_completed", "a window of this `locate`'s text was never read".to_owned())),
        };
        let baseline = match &window.rows {
            Some(Ok(baseline)) => baseline,
            Some(Err(Answer::Error { code, message })) => {
                return Err(Answer::Error { code: code.clone(), message: format!("its content-free baseline: {message}") });
            }
            Some(Err(other)) => return Err(other.clone()),
            None => return Err(failed("not_completed", "this `locate`'s content-free baseline was never read".to_owned())),
        };
        let read = match (asked.set_rows.as_deref(), baseline.set_rows.as_deref()) {
            (Some(q), Some(na)) => window_scores(q, na, text.heads, &window.keys, text.reading),
            _ => None,
        };
        let Some(scores) = read else {
            return Err(failed(
                "attention_malformed",
                format!(
                    "the heads' rows were not {} whole rows over a window's key span in both this question's prefill and its content-free baseline's",
                    text.heads
                ),
            ));
        };
        for (j, keys) in window.keys.iter().enumerate() {
            owned[window.first + j] = keys.is_some();
        }
        windows.push((window.first, scores));
    }
    Ok((merge(text.merge, text.segments, &windows), owned))
}

/// A labelled `choice` over `count` candidates the evidence names by
/// label, with the "none" option when `none` names one (GitHub #278).
fn choice_step(
    id: &str,
    instruction: &OrderedValue,
    alphabet: &ignis_core::decision::AnswerAlphabet,
    count: usize,
    none: Option<&str>,
) -> PreparedQuestion {
    let answers = alphabet.take(count + usize::from(none.is_some())).expect("the alphabet was checked at plan").to_vec();
    let mut options: Vec<PreparedOption> = answers[..count]
        .iter()
        .map(|answer| PreparedOption { name: answer.label.clone(), description: answer.label.clone() })
        .collect();
    if let Some(none) = none {
        options.push(PreparedOption { name: "none".to_owned(), description: none.to_owned() });
    }
    PreparedQuestion {
        id: id.to_owned(),
        kind: QuestionKind::Choice,
        instructions: instruction.clone(),
        options,
        answers,
        plan: None,
        scalar: None,
        digits: 0,
        head: None,
        within: None,
        locate: None,
    }
}

/// The yes/no a folded log's last request asks (spec 22 § Not found).
fn found_step(id: &str, instruction: &OrderedValue, alphabet: &ignis_core::decision::AnswerAlphabet) -> PreparedQuestion {
    let asked = match instruction {
        OrderedValue::String(text) => text.clone(),
        other => other.to_text(),
    };
    PreparedQuestion {
        id: id.to_owned(),
        kind: QuestionKind::Noul,
        instructions: OrderedValue::String(format!("{FOUND_QUESTION}{asked}")),
        options: super::NOUL_DEFAULT
            .iter()
            .map(|(name, description)| PreparedOption { name: (*name).to_owned(), description: (*description).to_owned() })
            .collect(),
        answers: alphabet.take(2).expect("the alphabet was checked at plan").to_vec(),
        plan: None,
        scalar: None,
        digits: 0,
        head: None,
        within: None,
        locate: None,
    }
}

/// The probability of each label of a `choice` answer, in label order.
fn probabilities(answer: &Answer, labels: &[String]) -> Result<(Vec<f64>, Option<f64>), Answer> {
    match answer {
        Answer::Choice { probabilities, .. } => Ok((
            labels.iter().map(|label| probabilities.get(label).copied().unwrap_or(0.0)).collect(),
            probabilities.get("none").copied(),
        )),
        Answer::Error { .. } => Err(answer.clone()),
        _ => Err(failed("not_a_readout", "a locate's labelled `choice` came back as something else".to_owned())),
    }
}

/// The position of the most probable label: the first on a tie.
fn pick(probabilities: &[f64]) -> usize {
    let mut best = 0;
    for (i, &p) in probabilities.iter().enumerate() {
        if p > probabilities[best] {
            best = i;
        }
    }
    best
}

/// One labelled request's prompts, ready: the plain `choice`, and — on a
/// route with `found` — its "none" variant and, for a folded log, the yes/no.
struct Last {
    labels: Vec<String>,
    /// Each label's segment of the target.
    segments: Vec<usize>,
    steps: Vec<PreparedQuestion>,
    prompts: Vec<Rendered>,
}

/// Answer every planned shortlist `locate` (GitHub #278): the readings its
/// plan rendered, a fold's level-1 `choice` and level 2, and the last
/// request. Every question's answer comes back under its slot with what the
/// request log says of it, and the prompt tokens every step spent.
pub(super) async fn answer_all(
    server: &crate::Server,
    mut questions: Vec<Planned>,
    mut shared: Shared,
    calibration: LocateCalibration,
    model: Option<String>,
    class: ignis_core::types::RequestClass,
) -> (Vec<(usize, Answer, Summary)>, u32) {
    let mut spent: u32 = 0;
    let count = questions.len();

    // 1. The readings planned before any prefill, text by text.
    let mut by_text: BTreeMap<usize, Vec<(usize, Vec<Option<Rendered>>)>> = BTreeMap::new();
    for (index, question) in questions.iter_mut().enumerate() {
        if let Some(text) = question.first {
            by_text.entry(text).or_default().push((index, std::mem::take(&mut question.prompts)));
        }
    }
    for (text, prompts) in by_text {
        spent = spent.saturating_add(read_windows(server, &mut shared, text, &mut questions, prompts, class).await);
    }

    // 2. A fold's level 1: the heads' first five templates, or all of them,
    //    and a labelled `choice` among them — skipped over one template.
    let mut scale = vec![1.0f64; count];
    let mut template = vec![0usize; count];
    let mut asked: Vec<(usize, Vec<usize>, PreparedQuestion, Rendered)> = Vec::new();
    for (index, question) in questions.iter_mut().enumerate() {
        let Steps::Folded(folded) = &question.steps else {
            continue;
        };
        let folded = folded.clone();
        let kept: Vec<usize> = match question.first {
            Some(text) => match shortlisted(&shared, text, &question.rows, SHORTLIST_TEMPLATES) {
                Ok(kept) => {
                    question.stats.windows += shared.texts[text].windows.len();
                    kept
                }
                Err(failure) => {
                    question.fail(failure);
                    continue;
                }
            },
            None => (0..folded.fold.clusters.len()).collect(),
        };
        question.stats.candidates.push(kept.clone());
        if kept.len() < 2 {
            template[index] = kept.first().copied().unwrap_or(0);
            continue;
        }
        let step = choice_step(&question.id, &question.instruction, &server.active().alphabet, kept.len(), None);
        let labels: Vec<String> = step.options.iter().map(|option| option.name.clone()).collect();
        let lines: Vec<String> = kept.iter().map(|&t| one_line(&folded.fold.level1[t])).collect();
        let evidence = Evidence::Json(OrderedValue::String(labelled(&lines, &labels)));
        match render(server, &evidence, &step, model.clone(), false).await {
            Ok(ready) => {
                spent = spent.saturating_add(ready.prompt_tokens);
                question.tokens = question.tokens.saturating_add(ready.prompt_tokens);
                asked.push((index, kept, step, ready));
            }
            Err(refusal) => question.fail(failed(refusal.code, refusal.message)),
        }
    }
    let mut steps = Vec::with_capacity(asked.len());
    let mut items = Vec::with_capacity(asked.len());
    for (index, kept, step, ready) in asked {
        steps.push((index, kept, step));
        items.push((index, ready));
    }
    let items: Vec<(usize, Rendered, Collect<'_>)> = items
        .into_iter()
        .map(|(index, ready)| {
            let step = &steps.iter().find(|(i, _, _)| *i == index).expect("asked").2;
            (index, ready, Collect::Step(step))
        })
        .collect();
    for (index, reply) in in_waves(server, items, class).await {
        let (_, kept, step) = steps.iter().find(|(i, _, _)| *i == index).expect("asked");
        let labels: Vec<String> = step.options.iter().map(|option| option.name.clone()).collect();
        match probabilities(&answer_of(reply), &labels) {
            Ok((p, _)) => {
                let at = pick(&p);
                template[index] = kept[at];
                scale[index] = p[at];
            }
            Err(failure) => questions[index].fail(failure),
        }
    }

    // 3. A fold's level 2: the chosen template's rows — past sixteen, the
    //    end heads' first sixteen.
    let mut kept_rows: Vec<Vec<usize>> = vec![Vec::new(); count];
    let mut members: Vec<Vec<Vec<usize>>> = vec![Vec::new(); count];
    let mut second: BTreeMap<usize, Vec<(usize, Vec<Option<Rendered>>)>> = BTreeMap::new();
    for (index, question) in questions.iter_mut().enumerate() {
        let Steps::Folded(folded) = &question.steps else {
            continue;
        };
        if question.failure.is_some() {
            continue;
        }
        let folded = folded.clone();
        let chosen = template[index];
        let rows = folded.fold.level2(&folded.lines, chosen);
        question.stats.template = Some(chosen);
        question.stats.rows = Some(rows.texts.len());
        if rows.texts.len() > SHORTLIST_LEN {
            let lines: Vec<String> = rows.texts.iter().map(|line| one_line(line)).collect();
            let key = format!("level2\u{0}{}\u{0}{chosen}", question.within);
            let opened =
                open_level(server, &mut shared, key, &lines, false, calibration, model.clone(), &question.id).await;
            let prompts = match opened {
                Ok(text) => prompts_for(server, &shared.texts[text], &question.instruction, calibration, model.clone(), &question.id)
                    .await
                    .map(|prompts| (text, prompts)),
                Err(refusal) => Err(refusal),
            };
            match prompts {
                Ok((text, prompts)) => {
                    second.entry(text).or_default().push((index, prompts));
                    question.second = Some(text);
                }
                Err(refusal) => question.fail(failed(refusal.code, refusal.message)),
            }
        } else {
            kept_rows[index] = (0..rows.texts.len()).collect();
        }
        members[index] = rows.members;
    }
    for (text, prompts) in second {
        spent = spent.saturating_add(read_windows(server, &mut shared, text, &mut questions, prompts, class).await);
    }
    for (index, question) in questions.iter_mut().enumerate() {
        let Some(text) = question.second.filter(|_| question.failure.is_none()) else {
            continue;
        };
        match shortlisted(&shared, text, &question.rows, SHORTLIST_LEN) {
            Ok(kept) => {
                question.stats.windows += shared.texts[text].windows.len();
                kept_rows[index] = kept;
            }
            Err(failure) => question.fail(failure),
        }
    }

    // 4. The last request: the plain `choice` over the candidates, with its
    //    "none" variant and the yes/no where the route carries `found`.
    let mut lasts: Vec<Option<Last>> = Vec::with_capacity(count);
    for (index, question) in questions.iter_mut().enumerate() {
        if question.failure.is_some() {
            lasts.push(None);
            continue;
        }
        match last_request(server, question, &shared, &kept_rows[index], &members[index], model.clone()).await {
            Ok(last) => {
                for ready in &last.prompts {
                    spent = spent.saturating_add(ready.prompt_tokens);
                    question.tokens = question.tokens.saturating_add(ready.prompt_tokens);
                }
                lasts.push(Some(last));
            }
            Err(failure) => {
                question.fail(failure);
                lasts.push(None);
            }
        }
    }
    // Every question's plain `choice` first — each keeps its candidates'
    // text — then its "none" variant and its yes/no, which claim it.
    let mut leaders = Vec::new();
    let mut followers = Vec::new();
    for (index, last) in lasts.iter_mut().enumerate() {
        let Some(last) = last else {
            continue;
        };
        for (n, ready) in std::mem::take(&mut last.prompts).into_iter().enumerate() {
            match n {
                0 => leaders.push((index * STEPS + n, ready)),
                _ => followers.push((index * STEPS + n, ready)),
            }
        }
    }
    let step_of = |slot: usize| &lasts[slot / STEPS].as_ref().expect("asked").steps[slot % STEPS];
    let leaders: Vec<_> = leaders.into_iter().map(|(slot, ready)| (slot, ready, Collect::Step(step_of(slot)))).collect();
    let followers: Vec<_> = followers.into_iter().map(|(slot, ready)| (slot, ready, Collect::Step(step_of(slot)))).collect();
    let mut replies: HashMap<(usize, usize), Answer> = HashMap::new();
    for (slot, reply) in in_waves(server, leaders, class).await {
        replies.insert((slot / STEPS, slot % STEPS), answer_of(reply));
    }
    for (slot, reply) in in_waves(server, followers, class).await {
        replies.insert((slot / STEPS, slot % STEPS), answer_of(reply));
    }

    let mut out = Vec::with_capacity(count);
    for (index, (mut question, last)) in questions.into_iter().zip(lasts).enumerate() {
        let answer = match (question.failure.take(), last) {
            (Some(failure), _) => failure,
            (None, Some(last)) => assemble(&mut question, &last, &replies, index, scale[index]),
            (None, None) => failed("not_completed", "this `locate`'s last request was never asked".to_owned()),
        };
        let summary = Summary {
            kind: question.kind,
            compression: question.compression,
            stats: std::mem::take(&mut question.stats),
            tokens: question.tokens,
        };
        out.push((question.slot, answer, summary));
    }
    (out, spent)
}

/// The most steps a last request asks: the plain `choice`, its "none" variant
/// and the yes/no.
const STEPS: usize = 3;

/// A reply that stands for an answer.
fn answer_of(reply: Reply) -> Answer {
    match reply {
        Reply::Answer(answer) => answer,
        Reply::Rows(_) => failed("not_a_readout", "a locate's labelled `choice` came back as rows".to_owned()),
    }
}

/// Render a question's last request: its candidates as segments of the
/// target in document order, shown as its route shows them, and the steps
/// asked over them.
async fn last_request(
    server: &crate::Server,
    question: &mut Planned,
    shared: &Shared,
    kept_rows: &[usize],
    members: &[Vec<usize>],
    model: Option<String>,
) -> Result<Last, Answer> {
    let segments: Vec<usize> = match &question.steps {
        // Each kept row's first line, as the caller sent it (the map from
        // folded lines to segments keeps their order).
        Steps::Folded(folded) => {
            rows_first_lines(members, kept_rows).into_iter().map(|line| folded.segment[line]).collect()
        }
        Steps::Read => {
            let text = question.first.expect("a read route reads its target");
            let (scores, owned) = scores_of(&shared.texts[text], text, &question.rows)?;
            question.stats.windows += shared.texts[text].windows.len();
            let prose = question.kind == Kind::Prose;
            let candidate: Vec<bool> =
                (0..scores.len()).map(|i| owned[i] && !(prose && question.texts[i].starts_with("# "))).collect();
            shortlist(&scores, &candidate, SHORTLIST_LEN)
        }
    };
    question.stats.candidates.push(segments.clone());
    let labels: Vec<String> =
        server.active().alphabet.take(segments.len()).expect("checked at plan").iter().map(|answer| answer.label.clone()).collect();
    let (text, segments) = match (question.kind, &question.steps) {
        (Kind::Prose, Steps::Read) => in_paragraphs(&question.texts, &segments, &labels),
        _ => {
            let lines: Vec<String> = segments.iter().map(|&s| one_line(&question.texts[s])).collect();
            (labelled(&lines, &labels), segments)
        }
    };
    let mut steps = vec![choice_step(&question.id, &question.instruction, &server.active().alphabet, segments.len(), None)];
    if question.finds() {
        let none = if question.kind == Kind::Prose { NONE_SENTENCE } else { NONE_LINE };
        steps.push(choice_step(&question.id, &question.instruction, &server.active().alphabet, segments.len(), Some(none)));
        if question.compression == Compression::TemplateFold {
            steps.push(found_step(&question.id, &question.instruction, &server.active().alphabet));
        }
    }
    let evidence = Evidence::Json(OrderedValue::String(text));
    let mut prompts = Vec::with_capacity(steps.len());
    for step in &steps {
        let ready = render(server, &evidence, step, model.clone(), false)
            .await
            .map_err(|refusal| failed(refusal.code, refusal.message))?;
        prompts.push(ready);
    }
    Ok(Last { labels: labels[..segments.len()].to_vec(), segments, steps, prompts })
}

/// The first `k` candidates of a question's merged reading of text
/// `index` (every segment a window read and owning keys), in document order.
fn shortlisted(
    shared: &Shared,
    index: usize,
    rows: &HashMap<(usize, usize), Result<AttentionScores, Answer>>,
    k: usize,
) -> Result<Vec<usize>, Answer> {
    let (scores, owned) = scores_of(&shared.texts[index], index, rows)?;
    Ok(shortlist(&scores, &owned, k))
}

/// What the request log and the metrics say of one shortlist.
pub(super) struct Summary {
    pub(super) kind: Kind,
    pub(super) compression: Compression,
    pub(super) stats: Stats,
    /// Prompt tokens this question's own steps spent (its baselines aside).
    pub(super) tokens: u32,
}

/// A question's answer from its last request (spec 22 § The wire): the pick
/// and its value, the ranking and the pointers on the scale of `scale` —
/// the fold's level-1 pick probability, 1 without one — and `found` where
/// the route carries it. Below [`FOUND_THRESHOLD`] nothing is named and the
/// ranking stays.
fn assemble(
    question: &mut Planned,
    last: &Last,
    replies: &HashMap<(usize, usize), Answer>,
    index: usize,
    scale: f64,
) -> Answer {
    let reply = |n: usize| {
        replies
            .get(&(index, n))
            .cloned()
            .unwrap_or_else(|| failed("not_completed", "a step of this `locate`'s last request was never asked".to_owned()))
    };
    let plain = match probabilities(&reply(0), &last.labels) {
        Ok((p, _)) => p,
        Err(failure) => return failure,
    };
    let found = match question.finds() {
        false => None,
        true => {
            let p_none = match probabilities(&reply(1), &last.labels) {
                Ok((_, p_none)) => p_none.unwrap_or(0.0),
                Err(failure) => return failure,
            };
            question.stats.p_none = Some(p_none);
            match question.compression {
                Compression::TemplateFold => {
                    let p_yes = match reply(2) {
                        Answer::Noul { noul } => noul,
                        failure @ Answer::Error { .. } => return failure,
                        _ => return failed("not_a_readout", "a locate's yes/no came back as something else".to_owned()),
                    };
                    question.stats.p_yes = Some(p_yes);
                    Some(found_log(p_none, p_yes))
                }
                Compression::None => Some(found_by_none(p_none)),
            }
        }
    };
    // By probability, the earlier candidate first on a tie.
    let mut order: Vec<usize> = (0..plain.len()).collect();
    order.sort_by(|&a, &b| plain[b].total_cmp(&plain[a]));
    let share = |at: usize| plain[at] * scale;
    let ranking: Vec<LocateRank> =
        order.iter().take(RANKING_LEN).map(|&at| LocateRank { segment: last.segments[at], share: share(at) }).collect();
    let best = order[0];
    let named = found.is_none_or(|found| found >= FOUND_THRESHOLD);
    let pointers: Vec<LocatePointer> = match named {
        false => Vec::new(),
        true => order
            .iter()
            .enumerate()
            .filter(|&(rank, &at)| rank == 0 || share(at) >= POINTER_SHARE)
            .map(|(_, &at)| LocatePointer {
                segment: last.segments[at],
                value: question.values[last.segments[at]].clone(),
                share: share(at),
            })
            .collect(),
    };
    let segment = last.segments[best];
    Answer::Locate {
        kind: LocateKind::from(question.kind),
        method: LocateMethod::Shortlist,
        compression: LocateCompression::from(question.compression),
        found,
        segment: named.then_some(segment),
        value: named.then(|| question.values[segment].clone()),
        confidence: named.then(|| share(best)),
        ranking,
        pointers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn question(kind: Kind, compression: Compression, values: &[&str]) -> Planned {
        Planned {
            slot: 0,
            id: "q".to_owned(),
            instruction: OrderedValue::String("which".to_owned()),
            within: String::new(),
            kind,
            compression,
            values: values.iter().map(|v| OrderedValue::String((*v).to_owned())).collect(),
            texts: values.iter().map(|v| (*v).to_owned()).collect(),
            steps: Steps::Read,
            first: None,
            prompts: Vec::new(),
            second: None,
            rows: HashMap::new(),
            tokens: 0,
            failure: None,
            stats: Stats::default(),
        }
    }

    fn last(segments: &[usize]) -> Last {
        let labels: Vec<String> = ["A", "B", "C", "D"][..segments.len()].iter().map(|l| (*l).to_owned()).collect();
        Last { labels, segments: segments.to_vec(), steps: Vec::new(), prompts: Vec::new() }
    }

    fn choice(pairs: &[(&str, f64)]) -> Answer {
        Answer::Choice {
            choice: pairs[0].0.to_owned(),
            probabilities: pairs.iter().map(|(k, p)| ((*k).to_owned(), *p)).collect(),
            confidence: pairs[0].1,
        }
    }

    /// Spec 22 § The wire: the pick first, every share on the scale of the
    /// fold's level-1 pick, pointers at 0.05 and up, `found` from the "none"
    /// baseline and the yes/no.
    #[test]
    fn a_found_answer_names_its_pick_its_ranking_and_its_pointers() {
        let mut q = question(Kind::Log, Compression::TemplateFold, &["a", "b", "c", "d"]);
        let last = last(&[1, 2, 3]);
        let replies: HashMap<(usize, usize), Answer> = [
            ((0, 0), choice(&[("A", 0.1), ("B", 0.85), ("C", 0.05)])),
            ((0, 1), choice(&[("A", 0.1), ("B", 0.8), ("C", 0.05), ("none", 0.05)])),
            ((0, 2), Answer::Noul { noul: 0.9 }),
        ]
        .into();
        let answer = serde_json::to_value(assemble(&mut q, &last, &replies, 0, 0.5)).unwrap();
        assert_eq!(answer["kind"], "log");
        assert_eq!(answer["method"], "shortlist");
        assert_eq!(answer["compression"], "template_fold");
        assert!((answer["found"].as_f64().unwrap() - (1.0 - 0.05 + 0.9) / 2.0).abs() < 1e-12);
        assert_eq!(answer["segment"], 2, "label B names segment 2");
        assert_eq!(answer["value"], "c");
        assert!((answer["confidence"].as_f64().unwrap() - 0.425).abs() < 1e-12, "0.85 times the level-1 pick's 0.5");
        let ranking: Vec<(u64, f64)> = answer["ranking"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| (r["segment"].as_u64().unwrap(), r["share"].as_f64().unwrap()))
            .collect();
        assert_eq!(ranking.iter().map(|r| r.0).collect::<Vec<_>>(), [2, 1, 3], "by probability");
        // 0.425 and 0.05 reach the threshold; 0.025 does not.
        let pointers: Vec<u64> = answer["pointers"].as_array().unwrap().iter().map(|p| p["segment"].as_u64().unwrap()).collect();
        assert_eq!(pointers, [2, 1]);
        assert_eq!(answer["pointers"][1]["value"], "b");
        assert_eq!(q.stats.p_none, Some(0.05));
        assert_eq!(q.stats.p_yes, Some(0.9));
    }

    /// Spec 22 § Not found: below 0.5 nothing is named, no pointer, and the
    /// ranking stays for a caller who wants the nearest candidate anyway.
    #[test]
    fn a_not_found_answer_names_nothing_and_keeps_its_ranking() {
        let mut q = question(Kind::Prose, Compression::None, &["x", "y"]);
        let last = last(&[0, 1]);
        let replies: HashMap<(usize, usize), Answer> = [
            ((0, 0), choice(&[("A", 0.7), ("B", 0.3)])),
            ((0, 1), choice(&[("A", 0.2), ("B", 0.1), ("none", 0.7)])),
        ]
        .into();
        let answer = serde_json::to_value(assemble(&mut q, &last, &replies, 0, 1.0)).unwrap();
        assert!((answer["found"].as_f64().unwrap() - 0.3).abs() < 1e-12);
        assert_eq!(answer["segment"], serde_json::Value::Null);
        assert_eq!(answer["value"], serde_json::Value::Null);
        assert_eq!(answer["confidence"], serde_json::Value::Null);
        assert_eq!(answer["pointers"], serde_json::json!([]));
        assert_eq!(answer["ranking"][0]["segment"], 0, "the ranking is kept");
    }

    /// A route with no measured `found` names its pick always, and its answer
    /// carries no `found` at all; the pick is a pointer even under 0.05.
    #[test]
    fn a_route_without_found_names_its_pick_and_omits_found() {
        let mut q = question(Kind::Log, Compression::None, &["x", "y", "z"]);
        let last = last(&[0, 2]);
        let replies: HashMap<(usize, usize), Answer> = [((0, 0), choice(&[("A", 0.5), ("B", 0.5)]))].into();
        let answer = serde_json::to_value(assemble(&mut q, &last, &replies, 0, 0.04)).unwrap();
        assert!(answer.get("found").is_none(), "{answer}");
        assert_eq!(answer["segment"], 0, "a tie keeps the earlier candidate");
        assert_eq!(answer["pointers"].as_array().unwrap().len(), 1, "0.02 each: the pick alone");
    }

    /// A step that failed is the question's failure, never an answer.
    #[test]
    fn a_failed_step_is_the_questions_failure() {
        let mut q = question(Kind::Records, Compression::None, &["x", "y"]);
        let last = last(&[0, 1]);
        let replies: HashMap<(usize, usize), Answer> = [
            ((0, 0), choice(&[("A", 0.6), ("B", 0.4)])),
            ((0, 1), failed("not_completed", "late".to_owned())),
        ]
        .into();
        let answer = serde_json::to_value(assemble(&mut q, &last, &replies, 0, 1.0)).unwrap();
        assert_eq!(answer["code"], "not_completed");
    }

    #[test]
    fn a_label_line_holds_no_line_break() {
        assert_eq!(one_line("a\nb"), "a b");
    }
}
