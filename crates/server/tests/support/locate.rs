//! The `locate` prompt and its segment-to-key map, as spec 18 phase A
//! measures them (`docs/specs/decide/18-locate-by-attention.md`, GitHub #274).
//!
//! Shared by the calibration harnesses (`attention_head_locate_gpu.rs`,
//! `attention_span_locate_gpu.rs`) and their table-driven tests
//! (`locate_segments.rs`). Phase B (GitHub #275) moved the served half into
//! the server — the copy scaffold's render, the segments and their keys
//! (`ignis_server::locate`) — so what a harness measures and what
//! `/v1/decide` serves are one function. What stays here is the study's
//! own: the index scaffold, the chunk cut, and spec 19's span tooling.
//!
//! **The render is layout L1** (spec 17): the system message is
//! `{"evidence": …}` alone, the user message is the kind text and then
//! `{"instruction": …}`, and the assistant turn opens with the forced
//! scaffold.

#![allow(dead_code, unused_imports)]

use std::ops::Range;

use ignis_server::decide::OrderedValue;

pub use ignis_server::locate::{CONTENT_FREE, Evidence, Segment, Unit, evidence, key_span, map_segments, owners};

/// The two answer scaffolds phase A measures, each with its own kind text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaffold {
    /// S1, index-shaped: the model is about to name the segment.
    Index,
    /// S2, copy-shaped: the model is about to copy the segment, where copying
    /// and retrieval heads look at their source (arXiv 2404.15574) — the one
    /// `/v1/decide` serves (`ignis_server::locate::COPY_OPENING`).
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
            (Self::Quote, _) => ignis_server::locate::COPY_OPENING,
        }
    }

    /// The kind text, first in the user turn.
    pub fn kind_text(self, unit: Unit) -> String {
        match self {
            Self::Index => {
                let unit = unit.name();
                format!(
                    "Find the one {unit} of the evidence that the instruction asks for. The evidence's \
                     {unit}s are numbered from 0. Answer with only a JSON object {{\"{unit}\": <its number>}}."
                )
            }
            Self::Quote => ignis_server::locate::kind_text(unit),
        }
    }
}

/// The user message: the kind text, then the instruction as the endpoint
/// writes one (`{"instruction": …}`), one blank line between them as L0
/// puts one between its kind text and the evidence.
pub fn user_text(scaffold: Scaffold, unit: Unit, instruction: &str) -> String {
    format!(
        "{}

{{\"instruction\":{}}}",
        scaffold.kind_text(unit),
        OrderedValue::String(instruction.to_owned()).to_text()
    )
}

/// Each segment's keys as a range of `span`'s positions, relative to its
/// start (`ignis_server::locate::segment_keys`), asserted contiguous.
pub fn segment_keys(owners: &[Option<u32>], span: &Range<usize>, segments: usize) -> Vec<Option<Range<usize>>> {
    ignis_server::locate::segment_keys(owners, span, segments).expect("a segment's tokens are contiguous")
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
