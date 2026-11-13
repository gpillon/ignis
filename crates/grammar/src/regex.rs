//! `pattern` and `format`: a string's decoded value held to a regular
//! language by a DFA of our own table, built from regex-automata's dense DFA.
//!
//! JSON Schema's `pattern` is ECMA-262 and unanchored: the value is valid when
//! the pattern matches anywhere in it. The DFA runs the pattern as an
//! unanchored search with every match reported (`MatchKind::All`); the first
//! time a match ends, the rest of the value is free, which our table spells as
//! the absorbing state [`FREE`]. A state from which no match can be reached
//! any more is cut to [`DEAD`], so the grammar never lets the model into a
//! string it cannot finish.
//!
//! The pattern's ECMA syntax is translated to regex-syntax's: `\d`, `\w`, `\b`
//! are ASCII as in ECMA (regex-syntax's are Unicode), `.` excludes ECMA's four
//! line terminators. Backreferences and lookaround are not regular: they are
//! refused by name, as is anything regex-syntax cannot parse.

use std::collections::HashMap;

use regex_automata::dfa::{dense, Automaton, StartKind};
use regex_automata::nfa::thompson;
use regex_automata::util::{start, syntax};
use regex_automata::{Anchored, MatchKind};

pub(crate) const DEAD: u32 = u32::MAX;
/// A match has been seen: every further byte keeps the value valid.
pub(crate) const FREE: u32 = 0;

/// [`crate::GrammarLimits::max_pattern_states`]'s default: the largest DFA a
/// pattern may compile to.
pub const DEFAULT_MAX_PATTERN_STATES: usize = 20_000;

/// Why a pattern has no DFA.
#[derive(Debug)]
pub(crate) enum DfaError {
    /// It compiles to more states than the bound, named.
    TooLarge(usize),
    /// It does not compile, or matches nothing.
    Invalid(String),
}

impl std::fmt::Display for DfaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DfaError::TooLarge(most) => write!(f, "the pattern's automaton exceeds {most} states"),
            DfaError::Invalid(e) => f.write_str(e),
        }
    }
}

pub(crate) struct ByteDfa {
    trans: Vec<u32>,
    eoi_accept: Vec<bool>,
    pub start: u32,
}

impl ByteDfa {
    #[inline]
    pub fn next(&self, s: u32, b: u8) -> u32 {
        self.trans[(s as usize) << 8 | b as usize]
    }
    #[inline]
    pub fn accepts_at_end(&self, s: u32) -> bool {
        self.eoi_accept[s as usize]
    }
    pub fn state_count(&self) -> usize {
        self.eoi_accept.len()
    }
    pub fn heap_bytes(&self) -> usize {
        self.trans.len() * 4 + self.eoi_accept.len()
    }

    /// Compile a regex-syntax pattern (already translated) to our table, of
    /// at most `max_states` states.
    pub fn build(pattern: &str, max_states: usize) -> Result<Self, DfaError> {
        Self::build_table(pattern, max_states).map_err(|e| match e {
            None => DfaError::TooLarge(max_states),
            Some(e) => DfaError::Invalid(e),
        })
    }

    /// [`ByteDfa::build`]: `Err(None)` past `max_states`.
    fn build_table(pattern: &str, max_states: usize) -> Result<Self, Option<String>> {
        let dfa = dense::Builder::new()
            .configure(
                dense::DFA::config()
                    .match_kind(MatchKind::All)
                    .start_kind(StartKind::Unanchored)
                    .dfa_size_limit(Some(64 << 20))
                    .determinize_size_limit(Some(64 << 20)),
            )
            .syntax(syntax::Config::new().unicode(true).utf8(true))
            .thompson(thompson::Config::new().utf8(true))
            .build(pattern)
            .map_err(|e| Some(e.to_string()))?;
        let start_rs = dfa
            .start_state(&start::Config::new().anchored(Anchored::No))
            .map_err(|e| Some(e.to_string()))?;

        // Our ids: 0 = FREE, then every live regex-automata state reached.
        let mut ids: HashMap<regex_automata::util::primitives::StateID, u32> = HashMap::new();
        let mut order = vec![start_rs];
        ids.insert(start_rs, 1);
        let mut raw: Vec<[RawEdge; 256]> = Vec::new();
        let mut eoi = vec![true];
        let mut i = 0;
        while i < order.len() {
            let s = order[i];
            let mut row = [RawEdge::Dead; 256];
            for b in 0..=255u8 {
                let t = dfa.next_state(s, b);
                row[b as usize] = if dfa.is_match_state(t) {
                    RawEdge::Free
                } else if dfa.is_dead_state(t) || dfa.is_quit_state(t) {
                    RawEdge::Dead
                } else {
                    let next_id = ids.len() as u32 + 1;
                    let id = *ids.entry(t).or_insert_with(|| {
                        order.push(t);
                        next_id
                    });
                    RawEdge::To(id)
                };
            }
            raw.push(row);
            eoi.push(dfa.is_match_state(dfa.next_eoi_state(s)));
            i += 1;
            if order.len() > max_states {
                return Err(None);
            }
        }
        // Liveness: a state is kept when a match is still reachable from it,
        // over the bytes a JSON string can feed the DFA: raw bytes from 0x20,
        // and the five C0 controls the short escapes decode to.
        let feedable = |b: usize| b >= 0x20 || matches!(b, 0x08 | 0x09 | 0x0A | 0x0C | 0x0D);
        let n = raw.len() + 1;
        let mut live = vec![false; n];
        live[0] = true;
        loop {
            let mut changed = false;
            for s in 1..n {
                if live[s] {
                    continue;
                }
                let reach = eoi[s]
                    || raw[s - 1].iter().enumerate().any(|(b, e)| {
                        feedable(b)
                            && match e {
                                RawEdge::Free => true,
                                RawEdge::To(t) => live[*t as usize],
                                RawEdge::Dead => false,
                            }
                    });
                if reach {
                    live[s] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        if !live[1] {
            return Err(Some("the pattern matches no string".into()));
        }
        let mut trans = vec![FREE; 256];
        for row in &raw {
            for e in row.iter() {
                trans.push(match e {
                    RawEdge::Free => FREE,
                    RawEdge::To(t) if live[*t as usize] => *t,
                    _ => DEAD,
                });
            }
        }
        Ok(Self { trans, eoi_accept: eoi, start: 1 })
    }
}

#[derive(Clone, Copy)]
enum RawEdge {
    Dead,
    Free,
    To(u32),
}

/// Translate an ECMA-262 pattern to regex-syntax, or refuse it by name.
pub(crate) fn translate_ecma(p: &str) -> Result<String, String> {
    let chars: Vec<char> = p.chars().collect();
    let mut out = String::with_capacity(p.len() + 16);
    let mut in_class = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '\\' => {
                let Some(&n) = chars.get(i + 1) else {
                    return Err("pattern ends in a lone backslash".into());
                };
                i += 2;
                match n {
                    'd' => out.push_str(if in_class { "0-9" } else { "[0-9]" }),
                    'w' => out.push_str(if in_class { "A-Za-z0-9_" } else { "[A-Za-z0-9_]" }),
                    'D' | 'W' if in_class => {
                        return Err(format!("\\{n} inside a character class is not supported"));
                    }
                    'D' => out.push_str("[^0-9]"),
                    'W' => out.push_str("[^A-Za-z0-9_]"),
                    'b' if in_class => out.push_str("\\x08"),
                    'b' => out.push_str("(?-u:\\b)"),
                    'B' => out.push_str("(?-u:\\B)"),
                    '1'..='9' => return Err("a backreference is not regular".into()),
                    'k' => return Err("a named backreference is not regular".into()),
                    'c' => return Err("\\c control escapes are not supported".into()),
                    '0' => out.push_str("\\x00"),
                    '/' => out.push('/'),
                    'u' if chars.get(i) == Some(&'{') => out.push_str("\\x"),
                    'u' => {
                        let hex: String = chars.get(i..i + 4).map(|s| s.iter().collect()).unwrap_or_default();
                        if hex.len() != 4 || !hex.chars().all(|h| h.is_ascii_hexdigit()) {
                            return Err("\\u needs four hex digits".into());
                        }
                        out.push_str(&format!("\\x{{{hex}}}"));
                        i += 4;
                    }
                    'v' => out.push_str("\\x0B"),
                    _ => {
                        out.push('\\');
                        out.push(n);
                    }
                }
                continue;
            }
            '[' if !in_class => {
                in_class = true;
                out.push('[');
                if chars.get(i + 1) == Some(&'^') {
                    out.push('^');
                    i += 1;
                }
                if chars.get(i + 1) == Some(&']') {
                    return Err("an empty character class is not supported".into());
                }
            }
            '[' => out.push_str("\\["),
            ']' if in_class => {
                in_class = false;
                out.push(']');
            }
            '.' if !in_class => out.push_str("[^\\n\\r\\x{2028}\\x{2029}]"),
            '&' | '~' if in_class => {
                out.push('\\');
                out.push(c);
            }
            '-' if in_class && chars.get(i + 1) == Some(&'-') => out.push_str("\\-"),
            '(' if !in_class => {
                let rest: String = chars[i..chars.len().min(i + 4)].iter().collect();
                if rest.starts_with("(?=") || rest.starts_with("(?!") || rest.starts_with("(?<=") || rest.starts_with("(?<!") {
                    return Err("lookaround is not regular".into());
                }
                out.push('(');
            }
            _ => out.push(c),
        }
        i += 1;
    }
    if in_class {
        return Err("unterminated character class".into());
    }
    Ok(out)
}

const D: &str = "[0-9]";

fn year() -> String {
    "(?:[1-9][0-9]{3}|0[1-9][0-9]{2}|00[1-9][0-9]|000[1-9])".into()
}

fn date() -> String {
    let leap = "(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:0[48]|[2468][048]|[13579][26])00)";
    format!(
        "(?:{y}-(?:0[13578]|1[02])-(?:0[1-9]|[12]{D}|3[01])|{y}-(?:0[469]|11)-(?:0[1-9]|[12]{D}|30)|{y}-02-(?:0[1-9]|1{D}|2[0-8])|{leap}-02-29)",
        y = year()
    )
}

fn time() -> String {
    format!("(?:[01]{D}|2[0-3]):[0-5]{D}:[0-5]{D}(?:\\.{D}+)?(?:Z|[+-](?:[01]{D}|2[0-3]):[0-5]{D})")
}

/// The regular language a `format` holds a string to, anchored at both ends.
/// Each is a subset of what the RFC allows, never a superset.
pub(crate) fn format_pattern(format: &str) -> Option<String> {
    let label = "[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?";
    let octet = "(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])";
    let h = "[0-9A-Fa-f]{1,4}";
    let ipv4 = format!("(?:{octet}\\.){{3}}{octet}");
    let body = match format {
        "date" => date(),
        "time" => time(),
        "date-time" => format!("{}T{}", date(), time()),
        // RFC 3339 appendix A: a year is followed only by a month, a month
        // only by a day; an hour only by a minute, a minute only by a second.
        "duration" => {
            let t = format!("T(?:{D}+H(?:{D}+M(?:{D}+S)?)?|{D}+M(?:{D}+S)?|{D}+S)");
            format!("P(?:{D}+W|(?:{D}+Y(?:{D}+M(?:{D}+D)?)?|{D}+M(?:{D}+D)?|{D}+D)(?:{t})?|{t})")
        }
        "email" => format!("[A-Za-z0-9.!#$%&'*+/=?^_`{{|}}~-]+@{label}(?:\\.{label})*"),
        "hostname" => format!("{label}(?:\\.{label})*"),
        "ipv4" => ipv4,
        "ipv6" => [
            format!("(?:{h}:){{7}}{h}"),
            format!("(?:{h}:){{1,7}}:"),
            format!("(?:{h}:){{1,6}}:{h}"),
            format!("(?:{h}:){{1,5}}(?::{h}){{1,2}}"),
            format!("(?:{h}:){{1,4}}(?::{h}){{1,3}}"),
            format!("(?:{h}:){{1,3}}(?::{h}){{1,4}}"),
            format!("(?:{h}:){{1,2}}(?::{h}){{1,5}}"),
            format!("{h}:(?::{h}){{1,6}}"),
            format!(":(?:(?::{h}){{1,7}}|:)"),
            format!("(?:{h}:){{6}}{ipv4}"),
            format!("::(?:[Ff]{{4}}:)?{ipv4}"),
        ]
        .join("|"),
        "uuid" => "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}".into(),
        _ => return None,
    };
    Some(format!("^(?:{body})$"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matches(dfa: &ByteDfa, s: &str) -> bool {
        let mut st = dfa.start;
        for &b in s.as_bytes() {
            st = dfa.next(st, b);
            if st == DEAD {
                return false;
            }
        }
        st == FREE || dfa.accepts_at_end(st)
    }

    fn compile(p: &str) -> ByteDfa {
        ByteDfa::build(&translate_ecma(p).unwrap(), DEFAULT_MAX_PATTERN_STATES).unwrap()
    }

    #[test]
    fn pattern_is_an_unanchored_search() {
        let d = compile("\\d{4}");
        assert!(matches(&d, "ab1234cd") && !matches(&d, "12a34"));
        let a = compile("^\\d{4}-\\d{2}-\\d{2}$");
        assert!(matches(&a, "2026-11-12") && !matches(&a, "2026-11-12x") && !matches(&a, "x2026-11-12"));
        // ECMA \d is ASCII.
        assert!(!matches(&a, "٢٠٢٦-11-12"));
        // The DFA dies on the first byte no completion can save.
        assert_eq!(a.next(a.start, b'x'), DEAD);
    }

    #[test]
    fn non_regular_constructs_are_refused_by_name() {
        assert!(translate_ecma("(a)\\1").unwrap_err().contains("backreference"));
        assert!(translate_ecma("a(?=b)").unwrap_err().contains("lookaround"));
        assert!(translate_ecma("(?<!x)y").unwrap_err().contains("lookaround"));
    }

    #[test]
    fn formats_hold_their_rfc_subsets() {
        let f = |name: &str| ByteDfa::build(&format_pattern(name).unwrap(), DEFAULT_MAX_PATTERN_STATES).unwrap();
        let date = f("date");
        assert!(matches(&date, "2024-02-29") && !matches(&date, "2023-02-29") && !matches(&date, "2000-13-01"));
        assert!(matches(&date, "2000-02-29") && !matches(&date, "1900-02-29") && !matches(&date, "0000-01-01"));
        let dt = f("date-time");
        assert!(matches(&dt, "2026-11-12T09:30:00Z") && matches(&dt, "2026-11-12T09:30:00.5+01:00"));
        assert!(!matches(&dt, "2026-11-12 09:30:00Z"));
        assert!(matches(&f("duration"), "P3DT4H") && !matches(&f("duration"), "P") && !matches(&f("duration"), "PT"));
        assert!(matches(&f("duration"), "P1Y2M3DT4H5M6S") && matches(&f("duration"), "P2W"));
        assert!(!matches(&f("duration"), "P3Y5D") && !matches(&f("duration"), "PT1H5S"), "RFC 3339 skips no unit");
        assert!(matches(&f("email"), "a.b+c@example.co") && !matches(&f("email"), "a@b@c"));
        assert!(matches(&f("ipv4"), "192.168.0.255") && !matches(&f("ipv4"), "192.168.0.256") && !matches(&f("ipv4"), "01.2.3.4"));
        assert!(matches(&f("ipv6"), "2001:db8::1") && matches(&f("ipv6"), "::1") && matches(&f("ipv6"), "::ffff:10.0.0.1"));
        assert!(!matches(&f("ipv6"), "2001:db8:::1"));
        assert!(matches(&f("uuid"), "123e4567-e89b-12d3-a456-426614174000") && !matches(&f("uuid"), "123e4567e89b12d3a456426614174000"));
        assert!(matches(&f("hostname"), "api.example.com") && !matches(&f("hostname"), "-bad.example"));
        assert!(matches(&f("time"), "23:59:59Z") && !matches(&f("time"), "24:00:00Z"));
    }
}
