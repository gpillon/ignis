//! `locate`'s prompt and its segments (spec
//! `docs/specs/decide/18-locate-by-attention.md`, GitHub #275, ADR 0041):
//! the pure host functions that turn a caller's `state` into the prompt the
//! vote was calibrated on, and each **segment** of it into the keys its
//! heads are read over.
//!
//! **The render is layout L1** (spec 17): the system message is
//! `{"evidence": …}` alone, the user message is the kind text and then
//! `{"instruction": …}`, and the assistant turn opens with the forced copy
//! scaffold `{"quote":"`. The state's bytes are the endpoint's own
//! ([`OrderedValue`]) — the reason to read attention instead of labels is
//! that nothing is written into the state — and each segment's byte range is
//! recorded as the evidence is written, so nothing is ever re-found by
//! searching text: two identical lines are two segments.
//!
//! Spec 18 phase A measured with these functions from
//! `crates/server/tests/support/locate.rs`; phase B moved them here, and the
//! calibration harness now reads them from the server. The prompt-pinning
//! test holds the endpoint's bytes to the renders set D was dumped with
//! (`crates/server/tests/decide_locate_prompt.rs`).

use std::fmt;
use std::ops::Range;

use crate::decide::{OrderedValue, quoted};

/// What a segment is called: a string's lines, an array's items.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Line,
    Item,
}

impl Unit {
    pub fn name(self) -> &'static str {
        match self {
            Self::Line => "line",
            Self::Item => "item",
        }
    }
}

/// The forced opening a `locate` is read at: the **copy scaffold**. The
/// model is about to copy the segment, which is where copying and retrieval
/// heads look at their source (arXiv 2404.15574), and where every head of
/// the vote was chosen.
pub const COPY_OPENING: &str = "{\"quote\":\"";

/// The kind text of a `locate` under the copy scaffold, first in the user
/// turn. The words the vote was calibrated on: a reworded kind text is an
/// unmeasured prompt (ADR 0034).
pub fn kind_text(unit: Unit) -> String {
    let unit = unit.name();
    format!(
        "Find the one {unit} of the evidence that the instruction asks for. Answer with only \
         a JSON object {{\"quote\": \"<that {unit}, copied exactly>\"}}."
    )
}

/// The user message: the kind text, then the instruction as the endpoint
/// writes one (`{"instruction": …}`), one blank line between them.
pub fn user_text(unit: Unit, instruction: &OrderedValue) -> String {
    format!("{}\n\n{{\"instruction\":{}}}", kind_text(unit), instruction.to_text())
}

/// The content-free instruction the baseline prefill reads (ICR's, QRHead's
/// and contextual calibration's "N/A"): what the heads do with this state
/// and scaffold when nothing is asked.
pub const CONTENT_FREE: &str = "N/A";

/// One segment of the evidence: where its content sits in the system text,
/// and whether it owns keys at all. An empty or all-whitespace segment keeps
/// its index and owns none.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub bytes: Range<usize>,
    pub owns: bool,
}

/// The system message's text, and every segment of the target in it.
#[derive(Clone, Debug, PartialEq)]
pub struct Evidence {
    /// `{"evidence": state}`, the whole state, as the endpoint writes it.
    pub system: String,
    pub unit: Unit,
    pub segments: Vec<Segment>,
    /// Each segment as the caller sent it: a line as a string, an element as
    /// its JSON.
    pub values: Vec<OrderedValue>,
}

/// Why a `within` target cannot be segmented.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetError {
    /// Not an RFC 6901 pointer: it does not start with `/`, or it holds a
    /// `~` that is not `~0` or `~1`.
    Malformed(String),
    /// A pointer that names nothing in this state.
    NotFound(String),
    /// A pointer through an object key the state writes more than once:
    /// which copy it means is a parser's choice.
    Ambiguous(String),
    /// A target that is not a string or a non-empty array, with what it is.
    Unsegmentable { pointer: String, found: &'static str },
}

impl TargetError {
    /// The code `/v1/decide` refuses the request with.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Malformed(_) => "locate_within_malformed",
            Self::NotFound(_) => "locate_within_not_found",
            Self::Ambiguous(_) => "locate_within_ambiguous",
            Self::Unsegmentable { .. } => "locate_target_unsegmentable",
        }
    }
}

impl fmt::Display for TargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(pointer) => write!(
                f,
                "`within` {pointer:?} is not a JSON Pointer (RFC 6901): it must be empty or start with `/`, and `~` may only be written `~0` or `~1`"
            ),
            Self::NotFound(pointer) => write!(f, "`within` {pointer:?} names nothing in the `state`"),
            Self::Ambiguous(pointer) => write!(
                f,
                "`within` {pointer:?} passes through a key the `state` writes more than once, so it names no one value"
            ),
            Self::Unsegmentable { pointer, found } => write!(
                f,
                "`within` {pointer:?} names {found}; a `locate` reads the lines of a string or the elements of a non-empty array"
            ),
        }
    }
}

/// One step of a pointer, resolved against the state: an object entry's
/// index among the entries, or an array index.
type Path = Vec<usize>;

/// Resolve `pointer` against `state` (RFC 6901), refusing a key written
/// twice rather than choosing a copy.
fn resolve(state: &OrderedValue, pointer: &str) -> Result<Path, TargetError> {
    if pointer.is_empty() {
        return Ok(Vec::new());
    }
    let Some(rest) = pointer.strip_prefix('/') else {
        return Err(TargetError::Malformed(pointer.to_owned()));
    };
    let mut path = Vec::new();
    let mut at = state;
    for raw in rest.split('/') {
        let token = unescape(raw).ok_or_else(|| TargetError::Malformed(pointer.to_owned()))?;
        let (step, next) = match at {
            OrderedValue::Object(entries) => {
                let mut matches = entries.iter().enumerate().filter(|(_, (key, _))| *key == token);
                let Some((index, (_, value))) = matches.next() else {
                    return Err(TargetError::NotFound(pointer.to_owned()));
                };
                if matches.next().is_some() {
                    return Err(TargetError::Ambiguous(pointer.to_owned()));
                }
                (index, value)
            }
            OrderedValue::Array(items) => {
                // Digits only, no leading zero, and never `-` (the element
                // past the end, which exists for writing, not reading).
                let index = (!token.is_empty()
                    && token.bytes().all(|b| b.is_ascii_digit())
                    && (token == "0" || !token.starts_with('0')))
                .then(|| token.parse::<usize>().ok())
                .flatten()
                .filter(|&index| index < items.len())
                .ok_or_else(|| TargetError::NotFound(pointer.to_owned()))?;
                (index, &items[index])
            }
            _ => return Err(TargetError::NotFound(pointer.to_owned())),
        };
        path.push(step);
        at = next;
    }
    Ok(path)
}

/// A pointer's reference token, unescaped: `~1` is `/`, `~0` is `~`, any
/// other `~` makes it no pointer.
fn unescape(raw: &str) -> Option<String> {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        match c {
            '~' => match chars.next() {
                Some('0') => out.push('~'),
                Some('1') => out.push('/'),
                _ => return None,
            },
            other => out.push(other),
        }
    }
    Some(out)
}

/// What a value is, for a refusal.
fn kind_of(value: &OrderedValue) -> &'static str {
    match value {
        OrderedValue::Null => "null",
        OrderedValue::Bool(_) => "a boolean",
        OrderedValue::Number(_) => "a number",
        OrderedValue::String(_) => "a string",
        OrderedValue::Array(items) if items.is_empty() => "an empty array",
        OrderedValue::Array(_) => "an array",
        OrderedValue::Object(_) => "an object",
    }
}

/// Where [`write`] records the target's segments.
struct Target {
    unit: Unit,
    segments: Vec<Segment>,
    values: Vec<OrderedValue>,
}

/// Write `value` as [`OrderedValue::write`] does, and when `path` runs out
/// here, segment it: a string's lines split on `\n` exactly (a `\r` stays
/// part of its line), or a non-empty array's elements. A line's range is its
/// escaped text, without the `\n` escapes between lines or the string's
/// quotes; an element's is its JSON text, without the quotes of a string
/// element or the commas. Those separator bytes belong to no segment.
fn write(value: &OrderedValue, path: &[usize], out: &mut String, target: &mut Option<Target>) {
    match (value, path.split_first()) {
        (OrderedValue::String(text), None) => {
            out.push('"');
            let mut segments = Vec::new();
            let mut values = Vec::new();
            for (index, line) in text.split('\n').enumerate() {
                if index > 0 {
                    out.push_str("\\n");
                }
                let escaped = quoted(line);
                let start = out.len();
                out.push_str(&escaped[1..escaped.len() - 1]);
                segments.push(Segment { bytes: start..out.len(), owns: !line.trim().is_empty() });
                values.push(OrderedValue::String(line.to_owned()));
            }
            out.push('"');
            *target = Some(Target { unit: Unit::Line, segments, values });
        }
        (OrderedValue::Array(items), None) => {
            out.push('[');
            let mut segments = Vec::new();
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                let start = out.len();
                item.write(out);
                let (bytes, owns) = match item {
                    OrderedValue::String(text) => ((start + 1)..(out.len() - 1), !text.trim().is_empty()),
                    _ => (start..out.len(), true),
                };
                segments.push(Segment { bytes, owns });
            }
            out.push(']');
            *target = Some(Target { unit: Unit::Item, segments, values: items.clone() });
        }
        (OrderedValue::Array(items), Some((&step, rest))) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                match index == step {
                    true => write(item, rest, out, target),
                    false => item.write(out),
                }
            }
            out.push(']');
        }
        (OrderedValue::Object(entries), Some((&step, rest))) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&quoted(key));
                out.push(':');
                match index == step {
                    true => write(item, rest, out, target),
                    false => item.write(out),
                }
            }
            out.push('}');
        }
        // `resolve` only hands over paths that run through objects and
        // arrays; anything else is written as it is and segments nothing.
        (other, _) => other.write(out),
    }
}

/// Write `{"evidence": state}` as `/v1/decide` writes it, and segment the
/// target `within` names (the whole state when it is empty).
pub fn evidence_within(state: &OrderedValue, within: &str) -> Result<Evidence, TargetError> {
    let path = resolve(state, within)?;
    segmentable_target(state, within)?;
    let mut system = String::from("{\"evidence\":");
    let mut found = None;
    write(state, &path, &mut system, &mut found);
    system.push('}');
    // Two sources for one text: the segmented writer and the endpoint's.
    debug_assert_eq!(system, format!("{{\"evidence\":{}}}", state.to_text()));
    let Target { unit, segments, values } = found.expect("a segmentable target is segmented");
    Ok(Evidence { system, unit, segments, values })
}

/// [`evidence_within`] of the whole state, as spec 18 phase A measured it:
/// a message rather than a [`TargetError`].
pub fn evidence(state: &OrderedValue) -> Result<Evidence, String> {
    evidence_within(state, "").map_err(|error| match error {
        TargetError::Unsegmentable { found: "an empty array", .. } => {
            "the state is an empty array: it has no segments".to_owned()
        }
        _ => format!("the state must be a string or a non-empty array to be segmented, not {}", state.to_text()),
    })
}

/// Which segment each token belongs to: the one holding most of its bytes,
/// the earlier one on a tie, and none when no segment that owns keys holds
/// any of them. `offsets` and the segments' ranges are in one coordinate.
///
/// Segments are in order and disjoint, so each token looks only at those its
/// range reaches.
pub fn owners(offsets: &[(usize, usize)], segments: &[Segment]) -> Vec<Option<u32>> {
    offsets
        .iter()
        .map(|&(start, end)| {
            let first = segments.partition_point(|segment| segment.bytes.end <= start);
            let mut best: Option<(u32, usize)> = None;
            for (index, segment) in segments.iter().enumerate().skip(first) {
                if segment.bytes.start >= end {
                    break;
                }
                if !segment.owns {
                    continue;
                }
                let overlap = end.min(segment.bytes.end).saturating_sub(start.max(segment.bytes.start));
                if overlap > 0 && best.is_none_or(|(_, most)| overlap > most) {
                    best = Some((index as u32, overlap));
                }
            }
            best.map(|(index, _)| index)
        })
        .collect()
}

/// The keys a readout reads: from the first segment-owned token to the last.
/// Keys inside it that belong to no segment are read and credit nobody.
pub fn key_span(owners: &[Option<u32>]) -> Option<Range<usize>> {
    let first = owners.iter().position(Option::is_some)?;
    let last = owners.iter().rposition(Option::is_some)?;
    Some(first..last + 1)
}

/// Each segment's keys as a range of `span`'s positions, relative to its
/// start: `None` for a segment that owns no key. `None` altogether when a
/// segment's tokens are not contiguous — which majority ownership over
/// disjoint, ordered segments rules out for any tokenization with no empty
/// token, and which is refused rather than read if it ever happens.
pub fn segment_keys(owners: &[Option<u32>], span: &Range<usize>, segments: usize) -> Option<Vec<Option<Range<usize>>>> {
    let mut keys: Vec<Option<Range<usize>>> = vec![None; segments];
    for position in span.clone() {
        if let Some(segment) = owners[position] {
            let at = position - span.start;
            let slot = keys.get_mut(segment as usize)?;
            match slot {
                None => *slot = Some(at..at + 1),
                Some(range) if range.end == at => range.end = at + 1,
                Some(_) => return None,
            }
        }
    }
    Some(keys)
}

/// Where the evidence's segments landed in one rendered prompt: the key span
/// and each segment's keys within it ([`segment_keys`]), from the render's
/// text and its tokens' byte offsets. The evidence must appear in the render
/// exactly once — the system message is written into it verbatim.
pub fn map_segments(
    rendered: &str,
    offsets: &[(usize, usize)],
    evidence: &Evidence,
) -> Result<(Range<usize>, Vec<Option<Range<usize>>>), String> {
    let found = rendered.matches(&evidence.system).count();
    if found != 1 {
        return Err(format!("the evidence appears {found} times in the render, not once"));
    }
    let base = rendered.find(&evidence.system).expect("counted above");
    let shifted: Vec<Segment> = evidence
        .segments
        .iter()
        .map(|s| Segment { bytes: s.bytes.start + base..s.bytes.end + base, owns: s.owns })
        .collect();
    let owned = owners(offsets, &shifted);
    let span = key_span(&owned).ok_or("no token of the render belongs to a segment")?;
    let keys = segment_keys(&owned, &span, shifted.len()).ok_or("a segment's tokens are not contiguous")?;
    Ok((span, keys))
}

// ---------------------------------------------------------------------------
// The shortlist's view of a target (spec 22, GitHub #278)
// ---------------------------------------------------------------------------

/// The value `within` names in `state` (the whole state when it is empty),
/// refused as [`evidence_within`] refuses it.
pub fn target_within<'a>(state: &'a OrderedValue, within: &str) -> Result<&'a OrderedValue, TargetError> {
    let mut target = state;
    for step in resolve(state, within)? {
        target = match target {
            OrderedValue::Object(entries) => &entries[step].1,
            OrderedValue::Array(items) => &items[step],
            _ => unreachable!("`resolve` walks objects and arrays only"),
        };
    }
    Ok(target)
}

/// [`target_within`], refused as [`evidence_within`] refuses a target that
/// is not a string or a non-empty array.
pub fn segmentable_target<'a>(state: &'a OrderedValue, within: &str) -> Result<&'a OrderedValue, TargetError> {
    let target = target_within(state, within)?;
    let segmentable = matches!(target, OrderedValue::String(_))
        || matches!(target, OrderedValue::Array(items) if !items.is_empty());
    match segmentable {
        true => Ok(target),
        false => Err(TargetError::Unsegmentable { pointer: within.to_owned(), found: kind_of(target) }),
    }
}

/// Whether `target` is a **records array** (spec 22 § `auto`): a JSON array
/// of at least two elements, every one an object, and not the shape
/// `Evidence::read` reads as content parts (every element an object with a
/// string `type`).
pub fn is_records_array(target: &OrderedValue) -> bool {
    let OrderedValue::Array(items) = target else {
        return false;
    };
    let objects = items.len() >= 2 && items.iter().all(|item| matches!(item, OrderedValue::Object(_)));
    let parts_shaped = items.iter().all(|item| match item {
        OrderedValue::Object(entries) => entries.iter().any(|(key, value)| key == "type" && value.as_str().is_some()),
        _ => false,
    });
    objects && !parts_shaped
}

/// A value as Python's `json.dumps(value, ensure_ascii=False)` writes it:
/// `", "` between items, `": "` after a key, keys in the order sent, and a
/// string's escapes as the endpoint writes them — the **spaced JSON** a
/// record is folded and shown as (spec 22).
///
/// Numbers are written as the endpoint writes them; a float Python would
/// spell with an exponent (`1e+16`, `1e-05`) is spelled `serde_json`'s way.
pub fn spaced_json(value: &OrderedValue) -> String {
    let mut out = String::new();
    write_spaced(value, &mut out);
    out
}

fn write_spaced(value: &OrderedValue, out: &mut String) {
    match value {
        OrderedValue::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_spaced(item, out);
            }
            out.push(']');
        }
        OrderedValue::Object(entries) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                out.push_str(&quoted(key));
                out.push_str(": ");
                write_spaced(item, out);
            }
            out.push('}');
        }
        other => other.write(out),
    }
}

/// Each segment of `target` as text: a string's lines, an array's string
/// elements as their text and every other element as its [`spaced_json`].
/// What the fold folds and a labelled `choice` shows.
pub fn segment_texts(target: &OrderedValue) -> Vec<String> {
    match target {
        OrderedValue::String(text) => text.split('\n').map(str::to_owned).collect(),
        OrderedValue::Array(items) => items
            .iter()
            .map(|item| match item {
                OrderedValue::String(text) => text.clone(),
                other => spaced_json(other),
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Whether each segment of `target` has **content**: what owns keys in a
/// `locate`'s render ([`Segment::owns`]) — a line or string element that is
/// not blank, or any other element. One pass: a long log is read once.
pub fn content_flags(target: &OrderedValue) -> Vec<bool> {
    match target {
        OrderedValue::String(text) => text.split('\n').map(|line| !line.trim().is_empty()).collect(),
        OrderedValue::Array(items) => items
            .iter()
            .map(|item| match item {
                OrderedValue::String(text) => !text.trim().is_empty(),
                _ => true,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// `auto` (spec 22 § `auto`): `records` for a records array, else the fold
/// of the target's first [`ignis_core::locate::AUTO_SEGMENTS`] segments with
/// content — `log` when at least [`ignis_core::locate::AUTO_LOG_SHARE`] of
/// them fall in templates of two or more, `prose` otherwise — with the
/// share it read (`None` for a records array).
pub fn auto_kind(target: &OrderedValue) -> (ignis_core::locate::Kind, Option<f64>) {
    use ignis_core::locate::{AUTO_LOG_SHARE, AUTO_SEGMENTS, Kind};
    if is_records_array(target) {
        return (Kind::Records, None);
    }
    let texts = segment_texts(target);
    let content: Vec<&String> = texts
        .iter()
        .zip(content_flags(target))
        .filter_map(|(text, content)| content.then_some(text))
        .take(AUTO_SEGMENTS)
        .collect();
    let share = ignis_core::locate::fold::fold(&content).templated_share();
    let kind = if share >= AUTO_LOG_SHARE { Kind::Log } else { Kind::Prose };
    (kind, Some(share))
}
