//! The string lexer: a JSON string's bytes after its opening quote, valid
//! UTF-8, escapes decoded, and (with a DFA) the decoded value held to a
//! `pattern` or `format`.
//!
//! Inside a string a DFA holds, a `\u` escape is not generated: its partial
//! hex digits name thousands of characters, and keeping the DFA alive through
//! them would cost a scan per digit. Raw UTF-8 writes every character a
//! pattern can want except the C0 controls, and the short escapes cover the
//! five a pattern plausibly names (`\b \f \n \r \t`). A backslash is allowed
//! only where some short escape keeps the DFA alive.

use crate::regex::{ByteDfa, DEAD, FREE};

const NORMAL: u8 = 0;
const BACKSLASH: u8 = 1;
const U0: u8 = 2; // ..=5: k hex digits of a \u escape read
const HIGH: u8 = 6; // a high surrogate read: need '\'
const HIGH_BS: u8 = 7; // need 'u'
const LOW0: u8 = 8; // ..=11: k hex digits of the low surrogate read

/// The lexer's state between two bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct StrState {
    pub esc: u8,
    /// UTF-8 continuation bytes still owed, with the first one's range:
    /// 0 none; 1 one (80-BF); 2 two; 3 two, first A0-BF (after E0); 4 two,
    /// first 80-9F (after ED); 5 three; 6 three, first 90-BF (after F0);
    /// 7 three, first 80-8F (after F4).
    pub utf8: u8,
    /// A \u escape's value so far; while a low surrogate is read, the high
    /// one sits in the top 16 bits.
    pub hex: u32,
    /// The DFA's state (0 when the string has none).
    pub dfa: u32,
}

impl StrState {
    pub fn start(dfa: Option<&ByteDfa>) -> Self {
        Self { esc: NORMAL, utf8: 0, hex: 0, dfa: dfa.map_or(FREE, |d| d.start) }
    }
}

pub(crate) enum StrStep {
    Inside(StrState),
    /// The closing quote, at a point where the value is complete and valid.
    Close,
    Reject,
}

#[inline]
fn hexval(b: u8) -> Option<u32> {
    (b as char).to_digit(16)
}

#[inline]
fn feed(dfa: Option<&ByteDfa>, s: &mut StrState, bytes: &[u8]) -> bool {
    if let Some(d) = dfa {
        for &b in bytes {
            s.dfa = d.next(s.dfa, b);
            if s.dfa == DEAD {
                return false;
            }
        }
    }
    true
}

fn feed_char(dfa: Option<&ByteDfa>, s: &mut StrState, cp: u32) -> bool {
    match char::from_u32(cp) {
        Some(c) => {
            let mut buf = [0u8; 4];
            feed(dfa, s, c.encode_utf8(&mut buf).as_bytes())
        }
        None => false,
    }
}

pub(crate) fn closable(dfa: Option<&ByteDfa>, s: &StrState) -> bool {
    s.esc == NORMAL && s.utf8 == 0 && dfa.is_none_or(|d| s.dfa == FREE || d.accepts_at_end(s.dfa))
}

#[inline]
pub(crate) fn step(dfa: Option<&ByteDfa>, st: StrState, b: u8) -> StrStep {
    let mut s = st;
    match st.esc {
        NORMAL => {
            if st.utf8 != 0 {
                let ok = match st.utf8 {
                    3 => (0xA0..=0xBF).contains(&b),
                    4 => (0x80..=0x9F).contains(&b),
                    6 => (0x90..=0xBF).contains(&b),
                    7 => (0x80..=0x8F).contains(&b),
                    _ => (0x80..=0xBF).contains(&b),
                };
                if !ok {
                    return StrStep::Reject;
                }
                s.utf8 = match st.utf8 {
                    1 => 0,
                    2..=4 => 1,
                    _ => 2,
                };
            } else {
                match b {
                    b'"' => return if closable(dfa, &st) { StrStep::Close } else { StrStep::Reject },
                    b'\\' => {
                        if let Some(d) = dfa {
                            let escapes = [b'"', b'\\', b'/', 0x08, 0x0C, b'\n', b'\r', b'\t'];
                            if !escapes.iter().any(|&x| d.next(st.dfa, x) != DEAD) {
                                return StrStep::Reject;
                            }
                        }
                        s.esc = BACKSLASH;
                        return StrStep::Inside(s);
                    }
                    0x00..=0x1F => return StrStep::Reject,
                    0x20..=0x7F => {}
                    0xC2..=0xDF => s.utf8 = 1,
                    0xE0 => s.utf8 = 3,
                    0xE1..=0xEC | 0xEE..=0xEF => s.utf8 = 2,
                    0xED => s.utf8 = 4,
                    0xF0 => s.utf8 = 6,
                    0xF1..=0xF3 => s.utf8 = 5,
                    0xF4 => s.utf8 = 7,
                    _ => return StrStep::Reject,
                }
            }
            if !feed(dfa, &mut s, &[b]) {
                return StrStep::Reject;
            }
        }
        BACKSLASH => {
            let decoded = match b {
                b'"' | b'\\' | b'/' => b,
                b'b' => 0x08,
                b'f' => 0x0C,
                b'n' => b'\n',
                b'r' => b'\r',
                b't' => b'\t',
                b'u' if dfa.is_none() => {
                    s.esc = U0;
                    s.hex = 0;
                    return StrStep::Inside(s);
                }
                _ => return StrStep::Reject,
            };
            s.esc = NORMAL;
            if !feed(dfa, &mut s, &[decoded]) {
                return StrStep::Reject;
            }
        }
        U0..=5 => {
            let Some(h) = hexval(b) else { return StrStep::Reject };
            s.hex = st.hex * 16 + h;
            if st.esc == U0 + 3 {
                let v = s.hex;
                if (0xD800..=0xDBFF).contains(&v) {
                    s.esc = HIGH;
                } else if (0xDC00..=0xDFFF).contains(&v) {
                    return StrStep::Reject;
                } else {
                    s.esc = NORMAL;
                    s.hex = 0;
                    if !feed_char(dfa, &mut s, v) {
                        return StrStep::Reject;
                    }
                }
            } else if st.esc == U0 + 1 && (0xDC..=0xDF).contains(&s.hex) {
                // Every completion of \uDC.. to \uDF.. is a lone low
                // surrogate: refuse now, never strand the model.
                return StrStep::Reject;
            } else {
                s.esc += 1;
            }
        }
        HIGH => {
            if b != b'\\' {
                return StrStep::Reject;
            }
            s.esc = HIGH_BS;
        }
        HIGH_BS => {
            if b != b'u' {
                return StrStep::Reject;
            }
            s.esc = LOW0;
            s.hex = st.hex << 16;
        }
        LOW0..=11 => {
            let Some(h) = hexval(b) else { return StrStep::Reject };
            let high = st.hex >> 16;
            let low = (st.hex & 0xFFFF) * 16 + h;
            // The low surrogate's first digit must be D, its second C-F.
            if (st.esc == LOW0 && h != 0xD) || (st.esc == LOW0 + 1 && h < 0xC) {
                return StrStep::Reject;
            }
            s.hex = (high << 16) | low;
            if st.esc == LOW0 + 3 {
                let cp = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
                s.esc = NORMAL;
                s.hex = 0;
                if !feed_char(dfa, &mut s, cp) {
                    return StrStep::Reject;
                }
            } else {
                s.esc += 1;
            }
        }
        _ => return StrStep::Reject,
    }
    StrStep::Inside(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regex::{translate_ecma, ByteDfa};

    fn run(dfa: Option<&ByteDfa>, body: &[u8]) -> bool {
        let mut s = StrState::start(dfa);
        for (i, &b) in body.iter().enumerate() {
            match step(dfa, s, b) {
                StrStep::Inside(n) => s = n,
                StrStep::Close => return i == body.len() - 1,
                StrStep::Reject => return false,
            }
        }
        false
    }

    #[test]
    fn json_string_syntax() {
        for ok in [&b"abc\""[..], b"\\n\\t\\\"\\u00e9\"", "caf\u{e9} \u{1F600}\"".as_bytes(), b"\\ud83d\\ude00\""] {
            assert!(run(None, ok), "{:?}", String::from_utf8_lossy(ok));
        }
        for bad in [&b"a\nb\""[..], b"\\x\"", b"\\ud83d\"", b"\\ude00\"", b"\xC0\xAF\"", b"\xED\xA0\x80\"", b"\\u12\""] {
            assert!(!run(None, bad), "{:?}", String::from_utf8_lossy(bad));
        }
    }

    #[test]
    fn a_pattern_sees_the_decoded_value() {
        let d = ByteDfa::build(&translate_ecma("^[A-Z]{3}-\\d{4}$").unwrap()).unwrap();
        assert!(run(Some(&d), b"ABC-1234\""));
        assert!(!run(Some(&d), b"\\u0041BC-1234\""), "no \\u escape inside a pattern");
        let nl = ByteDfa::build(&translate_ecma("^a\\nb$").unwrap()).unwrap();
        assert!(run(Some(&nl), b"a\\nb\""), "a short escape is decoded for the pattern");
        assert!(!run(Some(&nl), b"a\\tb\""));
        assert!(!run(Some(&d), b"ABC-123\""), "closing before the match is refused");
    }
}
