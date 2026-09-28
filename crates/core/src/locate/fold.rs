//! **`template_fold`** (spec `docs/specs/decide/22-locate-by-copy-over-a-folded-state.md`,
//! GitHub #278, ADR 0042): a log's lines grouped into **templates**,
//! Drain-style, so a `locate` over a very long log reads a short text first.
//!
//! Ported from `tools/locate-sets/compress.py` (`fold` with `values=True`,
//! `summarize`, `level2`, `_common_affixes`) at the settings set R2 was
//! judged with, and held to golden cases it writes
//! (`tools/locate-sets/golden22.py` → `crates/core/tests/locate_fold.rs`).
//! Python's arithmetic is kept: lengths and cuts in code points, its
//! whitespace (which counts `\x1c`-`\x1f` as whitespace where Rust's does
//! not), its Unicode classes, and first-seen cluster order.
//!
//! - **Clustering**: lines with the same source label (a bracket-opened
//!   prefix, `[svc-a]`) and the same token count, whose tokens agree on at
//!   least [`SIM`] of the positions, share a cluster; every position where
//!   they differ becomes `<*>`. The line's time is removed and obvious
//!   variables (numbers with units, hex, UUIDs, addresses) are masked `<v>`
//!   before the comparison, so they never split a cluster.
//! - **Level 1**: one line per template — its tokens, each variable slot
//!   showing its distinct values while they fit a budget, and `(xN)`.
//! - **Level 2**: one template's lines as their values only — values first,
//!   the slots' shared prefixes and suffixes cut, the line's time last —
//!   exact repeats folded into one row with `(xN)`; each row maps back to
//!   the lines it stands for.
//!
//! Kept as measured, and documented for callers: a bracket-opened line is
//! read as a source label unless the bracket holds a time; level 1 drops
//! every time; folding removes lines' order and neighbours.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

/// The share of a line's positions two lines must agree on to share a
/// cluster (`compress.SIM`).
pub const SIM: f64 = 0.5;

/// A slot's values shown at level 1 are cut to this many code points.
pub const VALUE_WIDTH: usize = 24;
/// Past [`VALUE_BUDGET`], a slot shows this many of its values and `|+N`.
pub const VALUE_CAP: usize = 6;
/// A slot shows every distinct value while they fit this many code points
/// (each value and its separator).
pub const VALUE_BUDGET: usize = 600;

/// Python's `\S+`: a run of anything `str.isspace` does not call whitespace,
/// which is Unicode's `White_Space` and the four separators `\x1c`-`\x1f`.
static TOKEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^\s\x1c-\x1f]+").expect("TOKEN"));
/// A timestamp: a date and a time in the common shapes, or a bare
/// `hh:mm:ss[.f]`.
static TIME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\d{4}[-/]\d\d[-/]\d\d[T ]\d\d:\d\d(?::\d\d(?:\.\d+)?)?(?:Z|[+-]\d\d:?\d\d)?|\b\d\d:\d\d:\d\d(?:\.\d+)?\b",
    )
    .expect("TIME")
});
/// A token that is obviously a variable, once its surrounding quotes and
/// punctuation are stripped.
static VARIABLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^(?:\d+(?:[.,:/]\d+)*(?:ms|s|µs|us|m|h|%|kb|mb|b)?|[0-9a-f]{8,}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|\d+\.\d+\.\d+\.\d+(?::\d+)?)$",
    )
    .expect("VARIABLE")
});
/// A source label: the line opens with `[` and runs to the first `]`.
static LABEL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\[[^\]]*\]").expect("LABEL"));

/// A template position that differs between the cluster's lines.
const WILD: &str = "<*>";
/// A position masked as an obvious variable in every line so far.
const MASKED: &str = "<v>";
/// What a token's ends are stripped of before [`VARIABLE`] reads it.
const PUNCTUATION: &[char] = &['"', ',', ';', '(', ')', '[', ']', '{', '}', '='];

/// Python's `str.isspace`.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// Python's `str.strip()`.
fn strip(text: &str) -> &str {
    text.trim_matches(is_space)
}

fn tokens(text: &str) -> Vec<String> {
    TOKEN.find_iter(text).map(|m| m.as_str().to_owned()).collect()
}

/// The line's label and the rest of it. A bracket-opened prefix is the
/// line's source label (`[svc-a]`) — unless it holds a time: then it is the
/// line's timestamp (`[Sun Dec 04 04:47:44 2005]`, `[10.30 16:49:06]`), and
/// a label read from it would put every line in a template of its own
/// (GitHub #278, found on set R3's first run).
fn split_label(line: &str) -> (&str, &str) {
    match LABEL.find(line) {
        Some(m) if !TIME.is_match(m.as_str()) => (m.as_str(), &line[m.end()..]),
        _ => ("", line),
    }
}

/// The first timestamp of `line`, or nothing.
pub fn stamp_of(line: &str) -> &str {
    TIME.find(line).map_or("", |m| m.as_str())
}

/// A body's tokens with its time removed and obvious variables masked —
/// what clustering compares — and its tokens as they are, for the values.
///
/// The reference removes the time twice before taking the raw tokens (once
/// for both, again for the raw); kept, as measured.
fn tokens_of(body: &str) -> (Vec<String>, Vec<String>) {
    let once = TIME.replace_all(body, " ");
    let masked = tokens(&once)
        .into_iter()
        .map(|token| match VARIABLE.is_match(token.trim_matches(PUNCTUATION)) {
            true => MASKED.to_owned(),
            false => token,
        })
        .collect();
    let raw = tokens(&TIME.replace_all(&once, " "));
    (masked, raw)
}

/// One template: the lines that share it.
#[derive(Debug, Clone, PartialEq)]
pub struct Cluster {
    /// The lines' source label, or empty.
    pub label: String,
    /// The template's tokens: a line's tokens where every member agrees,
    /// `<*>` where they differ, `<v>` where every member held a variable.
    pub template: Vec<String>,
    /// The member lines, as indices into the folded lines, in order.
    pub members: Vec<usize>,
    /// Each member's tokens as written (time removed), for its values.
    raw: Vec<Vec<String>>,
}

impl Cluster {
    /// Whether template position `k` is a slot that carries values.
    fn is_slot(&self, k: usize) -> bool {
        matches!(self.template[k].as_str(), WILD | MASKED)
    }
}

/// A folded text: its templates in first-seen order and their level-1 lines.
#[derive(Debug, Clone, PartialEq)]
pub struct Fold {
    pub clusters: Vec<Cluster>,
    /// One line per cluster, parallel to `clusters`.
    pub level1: Vec<String>,
}

/// One template's lines at level 2: a row per distinct run of values, each
/// with the lines it stands for (in order; the first is the answer when the
/// row is chosen — several lines identical but for their time are one row).
#[derive(Debug, Clone, PartialEq)]
pub struct Level2 {
    pub texts: Vec<String>,
    pub members: Vec<Vec<usize>>,
}

/// Fold `lines` into templates (`compress.fold(lines, SIM, values=True)`).
///
/// The caller leaves empty lines out and keeps the map back to its own
/// segments: every index here is into `lines`.
pub fn fold<S: AsRef<str>>(lines: &[S]) -> Fold {
    let mut clusters: Vec<Cluster> = Vec::new();
    let mut by_key: HashMap<(String, usize), Vec<usize>> = HashMap::new();
    for (index, line) in lines.iter().enumerate() {
        let (label, body) = split_label(line.as_ref());
        let (masked, raw) = tokens_of(body);
        let key = (label.to_owned(), masked.len());
        let denominator = masked.len().max(1) as f64;
        let mut best: Option<(usize, f64)> = None;
        for &candidate in by_key.get(&key).map(Vec::as_slice).unwrap_or(&[]) {
            let template = &clusters[candidate].template;
            let same = template
                .iter()
                .zip(&masked)
                .filter(|(a, b)| a.as_str() != WILD && a == b)
                .count();
            let score = same as f64 / denominator;
            // Strictly greater: the first cluster keeps a tie.
            if best.is_none_or(|(_, most)| score > most) {
                best = Some((candidate, score));
            }
        }
        match best {
            Some((at, score)) if score >= SIM => {
                let cluster = &mut clusters[at];
                for (slot, token) in cluster.template.iter_mut().zip(&masked) {
                    if slot != token {
                        *slot = WILD.to_owned();
                    }
                }
                cluster.members.push(index);
                cluster.raw.push(raw);
            }
            _ => {
                by_key.entry(key).or_default().push(clusters.len());
                clusters.push(Cluster { label: label.to_owned(), template: masked, members: vec![index], raw: vec![raw] });
            }
        }
    }
    let level1 = clusters.iter().map(level1_line).collect();
    Fold { clusters, level1 }
}

/// A cluster's level-1 line: its template with each slot's distinct values
/// (`summarize`), and `(xN)` for a cluster of several lines.
fn level1_line(cluster: &Cluster) -> String {
    let shown: Vec<String> = summarize(cluster)
        .into_iter()
        .map(|token| if token == MASKED { WILD.to_owned() } else { token })
        .collect();
    let count = match cluster.members.len() {
        1 => String::new(),
        n => format!(" (x{n})"),
    };
    strip(&format!("{} {}{count}", cluster.label, shown.join(" "))).to_owned()
}

/// A cluster's template with each variable slot showing its distinct
/// values — every one while they fit [`VALUE_BUDGET`], else the first
/// [`VALUE_CAP`] and `|+N` — each cut to [`VALUE_WIDTH`] code points, so a
/// word the question names stays visible in its template.
fn summarize(cluster: &Cluster) -> Vec<String> {
    cluster
        .template
        .iter()
        .enumerate()
        .map(|(k, token)| {
            if !cluster.is_slot(k) {
                return token.clone();
            }
            let mut seen: Vec<String> = Vec::new();
            let mut known = std::collections::HashSet::new();
            for raw in &cluster.raw {
                let value: String = raw.get(k).map_or(String::new(), |v| v.chars().take(VALUE_WIDTH).collect());
                if known.insert(value.clone()) {
                    seen.push(value);
                }
            }
            let fits = seen.iter().map(|v| v.chars().count() + 1).sum::<usize>() <= VALUE_BUDGET;
            let keep = if fits { seen.len() } else { seen.len().min(VALUE_CAP) };
            match seen.len() {
                0 => WILD.to_owned(),
                1 => seen[0].clone(),
                _ => {
                    let mut shown = seen[..keep].join("|");
                    if seen.len() > keep {
                        shown.push_str(&format!("|+{}", seen.len() - keep));
                    }
                    format!("{{{shown}}}")
                }
            }
        })
        .collect()
}

/// The prefix and suffix, in code points, every value of a slot shares
/// (`application=`, a closing quote), so a row shows only what differs.
fn common_affixes(values: &[Vec<char>]) -> (usize, usize) {
    if values.len() < 2 {
        return (0, 0);
    }
    let mut pre = 0;
    while values.iter().all(|v| v.len() > pre) && values.iter().all(|v| v[pre] == values[0][pre]) {
        pre += 1;
    }
    let mut suf = 0;
    while values.iter().all(|v| v.len() - pre > suf)
        && values.iter().all(|v| v[v.len() - 1 - suf] == values[0][values[0].len() - 1 - suf])
    {
        suf += 1;
    }
    (pre, suf)
}

impl Fold {
    /// Cluster `index`'s lines at level 2 (`compress.level2(values_first=True)`).
    /// `lines` are the lines that were folded, for each row's time.
    pub fn level2<S: AsRef<str>>(&self, lines: &[S], index: usize) -> Level2 {
        let cluster = &self.clusters[index];
        let slots: Vec<usize> = (0..cluster.template.len()).filter(|&k| cluster.is_slot(k)).collect();
        let cuts: HashMap<usize, (usize, usize)> = slots
            .iter()
            .map(|&k| {
                let values: Vec<Vec<char>> =
                    cluster.raw.iter().filter_map(|raw| raw.get(k)).map(|v| v.chars().collect()).collect();
                (k, common_affixes(&values))
            })
            .collect();
        let mut rows: Vec<(String, Vec<usize>)> = Vec::new();
        let mut index_of: HashMap<String, usize> = HashMap::new();
        for (&member, raw) in cluster.members.iter().zip(&cluster.raw) {
            let mut parts: Vec<String> = Vec::new();
            for &k in &slots {
                let Some(value) = raw.get(k) else {
                    continue;
                };
                let (pre, suf) = cuts[&k];
                let chars: Vec<char> = value.chars().collect();
                let shown: String = match chars.len() > pre + suf {
                    true => chars[pre..chars.len() - suf].iter().collect(),
                    false => value.clone(),
                };
                if !shown.is_empty() {
                    parts.push(shown);
                }
            }
            let values = parts.join(" | ");
            if let Some(&row) = index_of.get(&values) {
                rows[row].1.push(member);
                continue;
            }
            let stamp = stamp_of(lines[member].as_ref());
            let text = match stamp.is_empty() {
                true => values.clone(),
                false => format!("{values} @ {stamp}"),
            };
            index_of.insert(values, rows.len());
            rows.push((text, vec![member]));
        }
        let texts = rows
            .iter()
            .map(|(text, members)| match members.len() {
                1 => text.clone(),
                n => format!("{text} (x{n})"),
            })
            .collect();
        Level2 { texts, members: rows.into_iter().map(|(_, members)| members).collect() }
    }

    /// The share of the folded lines that fall in a template of two or more
    /// — what `auto` tells a log from prose by (spec 22 § `auto`).
    pub fn templated_share(&self) -> f64 {
        let lines: usize = self.clusters.iter().map(|c| c.members.len()).sum();
        let templated: usize = self.clusters.iter().map(|c| c.members.len()).filter(|&n| n >= 2).sum();
        templated as f64 / lines.max(1) as f64
    }
}
