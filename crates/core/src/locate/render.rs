//! What a `locate`'s labelled `choice` is shown (spec 22 § The labelled
//! `choice`, GitHub #278): the candidates, each prefixed with its answer
//! label, in a text that makes the answer checkable.
//!
//! Ported from `tools/locate-sets/zd_notfound.py` (`ask`: `label: line`
//! lines) and `zd_prose.py` (`render`: each candidate inside its paragraph),
//! and held to golden cases (`tools/locate-sets/golden22.py` →
//! `crates/core/tests/locate_renders.rs`).

/// The candidates as `label: text` lines, one per candidate, in the order
/// given: a fold's templates or rows, a log's lines, records as their
/// spaced JSON.
pub fn labelled<S: AsRef<str>, L: AsRef<str>>(lines: &[S], labels: &[L]) -> String {
    lines
        .iter()
        .zip(labels)
        .map(|(line, label)| format!("{}: {}", label.as_ref(), line.as_ref()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Prose's render (`zd_prose.render`): each candidate inside its
/// **paragraph** — the lines between the empty lines around it — the
/// paragraphs in document order and one blank line apart, the candidates
/// labelled and every other line of their paragraphs indented three spaces
/// and unlabelled.
///
/// Returns the text and, per label in order, the line it names. A paragraph
/// holding several candidates is shown once.
pub fn in_paragraphs<S: AsRef<str>, L: AsRef<str>>(lines: &[S], candidates: &[usize], labels: &[L]) -> (String, Vec<usize>) {
    let line = |i: usize| lines[i].as_ref();
    let mut chosen: Vec<usize> = candidates.to_vec();
    chosen.sort_unstable();
    chosen.dedup();
    let mut paragraphs: Vec<(usize, usize)> = Vec::new();
    for &c in &chosen {
        let mut a = c;
        while a > 0 && !line(a - 1).is_empty() {
            a -= 1;
        }
        let mut b = c;
        while b + 1 < lines.len() && !line(b + 1).is_empty() {
            b += 1;
        }
        if paragraphs.last().is_none_or(|&(_, end)| end < a) {
            paragraphs.push((a, b));
        }
    }
    let mut out: Vec<String> = Vec::new();
    let mut named = Vec::new();
    for (a, b) in paragraphs {
        for i in a..=b {
            match chosen.binary_search(&i) {
                Ok(_) => {
                    out.push(format!("{}: {}", labels[named.len()].as_ref(), line(i)));
                    named.push(i);
                }
                Err(_) => out.push(format!("   {}", line(i))),
            }
        }
        out.push(String::new());
    }
    (out.join("\n").trim_end_matches('\n').to_owned(), named)
}
