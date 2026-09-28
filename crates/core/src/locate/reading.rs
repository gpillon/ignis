//! The heads' **readings** of a long text, window by window, and the
//! **shortlist** they keep (spec 22, GitHub #278, ADR 0042).
//!
//! A reading is a **lift**, as the vote's: each head's attention row at the
//! copy scaffold, less the content-free twin's, **standardized over the
//! window's segments** and summed over the heads. Two readings:
//!
//! - the **end reading** (logs, records) scores a segment by where it
//!   closes: its last key, the keys between it and the next segment (its
//!   separator) and the next segment's first key — where the end heads mark
//!   a line at length (`docs/findings/2026-09-28-zero-decode-locate-exploration.md`);
//! - the **sum reading** (prose) by all its keys, as the vote does.
//!
//! A text longer than one window is cut into windows at segment boundaries
//! ([`cut_windows`]), each read as its own prefill; the windows' scores are
//! standardized and merged ([`merge`]), and the first few candidates are
//! kept in document order ([`shortlist`]).
//!
//! Ported from `tools/locate-sets/zd_cache.py` (`key_features`, read in f64
//! as `zd_logpipe.py` reads it), `zd_offline.py` (`zsum`), `zd_windows.py`
//! (`sub_windows`), `zd_records.py` (`cut`, `rank-windows`) and
//! `zd_prose.py` (`rank`), and held to golden cases
//! (`tools/locate-sets/golden22.py` → `crates/core/tests/locate_readings.rs`).

use std::ops::Range;

/// How a segment's attention is read (spec 22 § The heads and their readings).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reading {
    /// Its last key, its separator and the next segment's first key.
    End,
    /// All its keys.
    Sum,
}

/// What a segment that owns no key scores per head: `zsum`'s stand-in for
/// a missing lift, so such a segment sinks below every one that owns keys.
pub const UNOWNED_HEAD_SCORE: f64 = -1e3;

/// One head's softmax over the span, in f64.
fn softmax(scores: &[f32]) -> Vec<f64> {
    let top = scores.iter().fold(f64::NEG_INFINITY, |a, &s| a.max(f64::from(s)));
    let exp: Vec<f64> = scores.iter().map(|&s| (f64::from(s) - top).exp()).collect();
    let total: f64 = exp.iter().sum();
    exp.into_iter().map(|e| e / total).collect()
}

/// One head's feature per segment (`key_features`): the attention mass the
/// reading counts for it, `None` for a segment that owns no key.
fn features(row: &[f32], keys: &[Option<Range<usize>>], reading: Reading) -> Vec<Option<f64>> {
    let span = row.len();
    let p = softmax(row);
    let mut cum = Vec::with_capacity(span + 1);
    cum.push(0.0f64);
    let mut running = 0.0f64;
    for &mass in &p {
        running += mass;
        cum.push(running);
    }
    let owned: Vec<usize> = (0..keys.len()).filter(|&j| keys[j].is_some()).collect();
    let mut out = vec![None; keys.len()];
    for (n, &j) in owned.iter().enumerate() {
        let range = keys[j].as_ref().expect("owned");
        let (a, b) = (range.start, range.end);
        out[j] = Some(match reading {
            Reading::Sum => cum[b] - cum[a],
            Reading::End => {
                let next = owned.get(n + 1).map_or(span, |&k| keys[k].as_ref().expect("owned").start);
                let last = p[b - 1];
                let separator = cum[next] - cum[b];
                let next_first = if next < span { p[next] } else { 0.0 };
                last + separator + next_first
            }
        });
    }
    out
}

/// One window's reading (`zsum(lift).sum(heads)`): per segment, each head's
/// lift of the question over the content-free twin, standardized over the
/// window's segments that own keys, summed over the heads — and
/// [`UNOWNED_HEAD_SCORE`] per head for a segment that owns none.
///
/// `question` and `baseline` are `[heads][span]` rows, row-major; `keys`
/// each segment's keys as a range of the span. `None` when the rows are not
/// whole: no heads, rows that are not `heads` by one span, the two prefills
/// disagreeing, a score that is not finite, a segment past the span, or no
/// segment owning a key.
pub fn window_scores(
    question: &[f32],
    baseline: &[f32],
    heads: usize,
    keys: &[Option<Range<usize>>],
    reading: Reading,
) -> Option<Vec<f64>> {
    if heads == 0 || question.len() != baseline.len() || question.len() % heads != 0 {
        return None;
    }
    let span = question.len() / heads;
    if span == 0 || question.iter().chain(baseline).any(|s| !s.is_finite()) {
        return None;
    }
    if keys.iter().flatten().any(|r| r.start >= r.end || r.end > span) || keys.iter().all(Option::is_none) {
        return None;
    }
    let mut total = vec![0.0f64; keys.len()];
    for (asked, content_free) in question.chunks_exact(span).zip(baseline.chunks_exact(span)) {
        let lifted = features(asked, keys, reading);
        let prior = features(content_free, keys, reading);
        let lift: Vec<Option<f64>> = lifted.iter().zip(&prior).map(|(q, na)| Some((*q)? - (*na)?)).collect();
        let present: Vec<f64> = lift.iter().flatten().copied().collect();
        let count = present.len() as f64;
        let mean = present.iter().sum::<f64>() / count;
        let deviation = (present.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / count).sqrt() + 1e-12;
        for (slot, value) in total.iter_mut().zip(&lift) {
            *slot += match value {
                Some(x) => (x - mean) / deviation,
                None => UNOWNED_HEAD_SCORE,
            };
        }
    }
    Some(total)
}

/// Cut a text's segments into **windows** of at most `budget` keys (spec 22
/// § Windows), as `[first, end)` ranges in order: at the last **empty**
/// segment within the window when there is one — prose's paragraph break,
/// which then belongs to no window — else before the segment that would not
/// fit (a record, a line). `costs` is each segment's keys, its separator
/// included; `empty` whether it is an empty segment. A segment alone past
/// the budget gets a window of its own.
///
/// `zd_windows.sub_windows` on a text with breaks, `zd_records.cut` on one
/// without.
pub fn cut_windows(costs: &[u64], empty: &[bool], budget: u64) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let (mut start, mut total, mut last_break) = (0usize, 0u64, None::<usize>);
    for (i, &cost) in costs.iter().enumerate() {
        if empty.get(i).copied().unwrap_or(false) {
            last_break = Some(i);
        }
        if total + cost > budget && i > start {
            if let Some(at) = last_break.filter(|&at| at > start) {
                out.push(start..at);
                start = at + 1;
                // The segments since the break, this one included: none when
                // this one is the break.
                total = costs[start..i + 1].iter().sum();
                last_break = None;
                continue;
            }
            out.push(start..i);
            (start, total, last_break) = (i, 0, None);
        }
        total += cost;
    }
    out.push(start..costs.len());
    out
}

/// How windows' readings are standardized and merged over the whole text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Merge {
    /// `zd_prose.py rank`: a window's sum under -1e2 (or not finite) set to
    /// -1e4 and kept in its standardization; a segment no window read at
    /// -1e4.
    Prose,
    /// `zd_records.py rank-windows`: a window's scores standardized over all
    /// its segments; a segment no window read at -1e9. Also a log read
    /// without a fold.
    Records,
}

/// Merge windows' readings (`(first segment, scores)`, in order) into one
/// score per segment of a text of `segments`.
pub fn merge(rule: Merge, segments: usize, windows: &[(usize, Vec<f64>)]) -> Vec<f64> {
    let unread = match rule {
        Merge::Prose => -1e4,
        Merge::Records => -1e9,
    };
    let mut merged = vec![unread; segments];
    for (first, scores) in windows {
        let standardized = match rule {
            Merge::Prose => {
                let v: Vec<f64> =
                    scores.iter().map(|&x| if x.is_finite() && x > -1e2 { x } else { -1e4 }).collect();
                // `std`: over the values above -1e5, which the -1e4 above are.
                let kept: Vec<f64> = v.iter().copied().filter(|&x| x > -1e5).collect();
                let (mean, deviation) = moments(&kept);
                v.iter().map(|&x| if x > -1e5 { (x - mean) / (deviation + 1e-12) } else { -1e3 }).collect()
            }
            Merge::Records => {
                let v: Vec<f64> = scores.iter().map(|&x| if x.is_finite() { x } else { -1e9 }).collect();
                let (mean, deviation) = moments(&v);
                v.iter().map(|&x| (x - mean) / (deviation + 1e-12)).collect::<Vec<f64>>()
            }
        };
        for (at, value) in standardized.into_iter().enumerate() {
            if let Some(slot) = merged.get_mut(first + at) {
                *slot = value;
            }
        }
    }
    merged
}

/// The mean and the population standard deviation.
fn moments(values: &[f64]) -> (f64, f64) {
    let count = values.len().max(1) as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / count;
    (mean, variance.sqrt())
}

/// The first `k` candidates by score — the earlier segment keeps a tie —
/// in **document order**: what a labelled `choice` is shown.
pub fn shortlist(scores: &[f64], candidate: &[bool], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    // Stable: equal scores keep their order.
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let mut kept: Vec<usize> = order.into_iter().filter(|&i| candidate.get(i).copied().unwrap_or(false)).take(k).collect();
    kept.sort_unstable();
    kept
}
