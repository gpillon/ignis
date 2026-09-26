//! The `locate` prompt and its segment-to-key map, as spec 18 phase A
//! measures them (`docs/specs/decide/18-locate-by-attention.md`, GitHub #274).
//!
//! Pure host functions, shared by the calibration harness
//! (`attention_head_locate_gpu.rs`) and their table-driven tests
//! (`locate_segments.rs`). They live beside the tests and not in the server
//! because phase A changes no server behaviour: phase B (GitHub #275) moves
//! them into `/v1/decide`, and its prompt-pinning test holds the endpoint's
//! bytes to the ones measured here.
//!
//! **The render is layout L1** (spec 17): the system message is
//! `{"evidence": …}` alone, the user message is the kind text and then
//! `{"instruction": …}`, and the assistant turn opens with the forced
//! scaffold. The state's bytes are the endpoint's own (`OrderedValue`), so
//! nothing about a segment is re-found by searching text: each segment's byte
//! range is recorded as the evidence is written.

#![allow(dead_code)]

use std::ops::Range;

use ignis_server::decide::OrderedValue;

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

/// The two answer scaffolds phase A measures, each with its own kind text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaffold {
    /// S1, index-shaped: the model is about to name the segment.
    Index,
    /// S2, copy-shaped: the model is about to copy the segment, where copying
    /// and retrieval heads look at their source (arXiv 2404.15574).
    Quote,
}

impl Scaffold {
    pub const ALL: [Self; 2] = [Self::Index, Self::Quote];

    pub fn name(self) -> &'static str {
        match self {
            Self::Index => "s1",
            Self::Quote => "s2",
        }
    }

    /// The assistant turn's forced opening. The query is its last token.
    pub fn opening(self, unit: Unit) -> &'static str {
        match (self, unit) {
            (Self::Index, Unit::Line) => "{\"line\":",
            (Self::Index, Unit::Item) => "{\"item\":",
            (Self::Quote, _) => "{\"quote\":\"",
        }
    }

    /// The kind text, first in the user turn.
    pub fn kind_text(self, unit: Unit) -> String {
        let unit = unit.name();
        match self {
            Self::Index => format!(
                "Find the one {unit} of the evidence that the instruction asks for. The evidence's \
                 {unit}s are numbered from 0. Answer with only a JSON object {{\"{unit}\": <its number>}}."
            ),
            Self::Quote => format!(
                "Find the one {unit} of the evidence that the instruction asks for. Answer with only \
                 a JSON object {{\"quote\": \"<that {unit}, copied exactly>\"}}."
            ),
        }
    }
}

/// The content-free instruction the baseline prefill reads (ICR's, QRHead's
/// and contextual calibration's "N/A").
pub const CONTENT_FREE: &str = "N/A";

/// One segment of the evidence: where its content sits in the system text,
/// and whether it owns keys at all. An empty or all-whitespace segment keeps
/// its index and owns none.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub bytes: Range<usize>,
    pub owns: bool,
}

/// The system message's text, and every segment's place in it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Evidence {
    pub system: String,
    pub unit: Unit,
    pub segments: Vec<Segment>,
}

/// A JSON string literal, escaped as `OrderedValue` escapes one.
fn quoted(text: &str) -> String {
    serde_json::to_string(text).expect("a string always serializes")
}

/// Write `{"evidence": state}` as `/v1/decide` writes it, recording each
/// segment's byte range as it goes: a string's lines split on `\n` exactly
/// (a `\r` stays part of its line), or a non-empty array's elements.
///
/// A line's range is its escaped text, without the `\n` escapes between
/// lines or the string's quotes; an array element's is its JSON text, without
/// the quotes of a string element or the commas. Those separator bytes
/// belong to no segment.
pub fn evidence(state: &OrderedValue) -> Result<Evidence, String> {
    let mut system = String::from("{\"evidence\":");
    let (unit, segments) = match state {
        OrderedValue::String(text) => {
            system.push('"');
            let mut segments = Vec::new();
            for (index, line) in text.split('\n').enumerate() {
                if index > 0 {
                    system.push_str("\\n");
                }
                let escaped = quoted(line);
                let start = system.len();
                system.push_str(&escaped[1..escaped.len() - 1]);
                segments.push(Segment { bytes: start..system.len(), owns: !line.trim().is_empty() });
            }
            system.push('"');
            (Unit::Line, segments)
        }
        OrderedValue::Array(items) if !items.is_empty() => {
            system.push('[');
            let mut segments = Vec::new();
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    system.push(',');
                }
                let text = item.to_text();
                let start = system.len();
                system.push_str(&text);
                let (bytes, owns) = match item {
                    OrderedValue::String(value) => ((start + 1)..(system.len() - 1), !value.trim().is_empty()),
                    _ => (start..system.len(), true),
                };
                segments.push(Segment { bytes, owns });
            }
            system.push(']');
            (Unit::Item, segments)
        }
        OrderedValue::Array(_) => return Err("the state is an empty array: it has no segments".to_owned()),
        other => {
            return Err(format!(
                "the state must be a string or a non-empty array to be segmented, not {}",
                other.to_text()
            ));
        }
    };
    system.push('}');
    // Two sources for one text: the segmented writer and the endpoint's.
    let whole = format!("{{\"evidence\":{}}}", state.to_text());
    if system != whole {
        return Err("the segmented evidence is not the bytes the endpoint writes".to_owned());
    }
    Ok(Evidence { system, unit, segments })
}

/// The user message: the kind text, then the instruction as the endpoint
/// writes one (`{"instruction": …}`), one blank line between them as L0
/// puts one between its kind text and the evidence.
pub fn user_text(scaffold: Scaffold, unit: Unit, instruction: &str) -> String {
    format!(
        "{}\n\n{{\"instruction\":{}}}",
        scaffold.kind_text(unit),
        OrderedValue::String(instruction.to_owned()).to_text()
    )
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
/// start: `None` for a segment that owns no key. A segment's tokens are
/// contiguous, which is asserted rather than assumed.
pub fn segment_keys(owners: &[Option<u32>], span: &Range<usize>, segments: usize) -> Vec<Option<Range<usize>>> {
    let mut keys: Vec<Option<Range<usize>>> = vec![None; segments];
    for position in span.clone() {
        if let Some(segment) = owners[position] {
            let at = position - span.start;
            let slot = &mut keys[segment as usize];
            match slot {
                None => *slot = Some(at..at + 1),
                Some(range) => {
                    assert_eq!(range.end, at, "segment {segment}'s tokens are not contiguous");
                    range.end = at + 1;
                }
            }
        }
    }
    keys
}

/// Where the evidence's segments landed in one rendered prompt: the key span
/// and each segment's keys within it (`segment_keys`), from the render's
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
    let keys = segment_keys(&owned, &span, shifted.len());
    Ok((span, keys))
}

/// The chunks a prompt is prefilled in, as `ConcreteScheduler` cuts a
/// decision's text prompt with an attention readout (`chunk_take`): at most
/// `chunk` tokens each, the last one kept at least `tail` wide so that the
/// hq prompt route materializes the keys it is read over.
pub fn chunks(total: u32, chunk: u32, tail: u32) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let mut start = 0u32;
    while start < total {
        let remaining = total - start;
        let mut take = remaining.min(chunk);
        let left = remaining - take;
        if left > 0 && left < tail && remaining > tail {
            take = remaining - tail;
        }
        out.push((u64::from(start), u64::from(take)));
        start += take;
    }
    out
}

// ── spec 19 phase 1: spans (GitHub #276) ─────────────────────────────────

/// The kind text of a span question: the exact answer, copied, under the
/// copy scaffold `{"quote":"`. `tools/locate-sets/generation.py` asks the
/// generation route with the same words (its `SPAN_KIND`).
pub const SPAN_KIND: &str = "Find the exact answer to the instruction in the evidence. Answer with only a JSON \
                             object {\"quote\": \"<the answer, copied exactly from the evidence>\"}.";

/// The user message of a span question: the span kind text, then the
/// instruction as the endpoint writes one, as [`user_text`] lays out spec
/// 18's.
pub fn span_user_text(instruction: &str) -> String {
    format!("{SPAN_KIND}\n\n{{\"instruction\":{}}}", OrderedValue::String(instruction.to_owned()).to_text())
}

/// What a prompt token is, for reading where the attention goes (spec 19
/// Q4): the state, the kind text, the instruction's own words, or the
/// template around them (role headers, special tokens, the think block,
/// the `{"instruction":` wrapper). The forced scaffold and quote are
/// labelled by the caller, which appends them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Region {
    Evidence,
    Kind,
    Instruction,
    Template,
}

impl Region {
    pub fn name(self) -> &'static str {
        match self {
            Self::Evidence => "evidence",
            Self::Kind => "kind",
            Self::Instruction => "instruction",
            Self::Template => "template",
        }
    }
}

/// Every token of a render labelled by where its first byte sits, as runs
/// `(positions, region)`. The evidence and the user message each appear in
/// the render exactly once; the instruction is the JSON string's content
/// inside `{"instruction":"…"}` (its quotes and the wrapper are template).
pub fn regions(
    rendered: &str,
    offsets: &[(usize, usize)],
    evidence: &Evidence,
    kind_text: &str,
    user: &str,
) -> Result<Vec<(Range<usize>, Region)>, String> {
    let once = |needle: &str, what: &str| -> Result<usize, String> {
        match rendered.matches(needle).count() {
            1 => Ok(rendered.find(needle).expect("counted")),
            n => Err(format!("the {what} appears {n} times in the render, not once")),
        }
    };
    let ev = once(&evidence.system, "evidence")?;
    let us = once(user, "user message")?;
    if !user.starts_with(kind_text) {
        return Err("the user message does not open with the kind text".to_owned());
    }
    let wrapper = "{\"instruction\":\"";
    let at = user.rfind(wrapper).ok_or("the user message holds no instruction object")?;
    let instruction = (us + at + wrapper.len())..(us + user.len() - 2);
    let label = |byte: usize| {
        if (ev..ev + evidence.system.len()).contains(&byte) {
            Region::Evidence
        } else if (us..us + kind_text.len()).contains(&byte) {
            Region::Kind
        } else if instruction.contains(&byte) {
            Region::Instruction
        } else {
            Region::Template
        }
    };
    let mut runs: Vec<(Range<usize>, Region)> = Vec::new();
    for (position, &(start, _)) in offsets.iter().enumerate() {
        let region = label(start);
        match runs.last_mut() {
            Some((range, last)) if *last == region && range.end == position => range.end = position + 1,
            _ => runs.push((position..position + 1, region)),
        }
    }
    Ok(runs)
}

/// The byte range of each key of `span` relative to the evidence's first
/// byte in the render: how a scorer maps a gold character span of the state
/// onto keys without tokenizing anything itself.
pub fn key_bytes(
    rendered: &str,
    offsets: &[(usize, usize)],
    evidence: &Evidence,
    span: &Range<usize>,
) -> Result<Vec<[usize; 2]>, String> {
    let base = rendered.find(&evidence.system).ok_or("the evidence is not in the render")?;
    Ok(offsets[span.clone()]
        .iter()
        .map(|&(start, end)| [start.saturating_sub(base), end.saturating_sub(base)])
        .collect())
}
