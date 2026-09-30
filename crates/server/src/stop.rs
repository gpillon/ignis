//! The caller's `stop` sequences (spec server/09, GitHub #284): parsed off
//! the request, and matched over the content channel's text as it streams.
//!
//! Matching is over decoded text, byte-exact, never over token ids, so a
//! sequence that does not align to a token boundary still fires. The
//! matcher sits after the tool-call scanner (`crate::toolcall`): it only
//! ever sees content outside a `<tool_call>` block, so a sequence that
//! happens to occur inside a call's arguments cannot truncate the call, and
//! it never sees the reasoning channel, whose close belongs to the template
//! and the thinking budget.
//!
//! **Streaming holds back a possible prefix**, the discipline
//! `toolcall.rs` applies to `<tool_call>`: the tail of the text that is a
//! proper prefix of any sequence waits for the next feed to resolve it, so a
//! sequence split across chunks is never missed and no fragment of one ever
//! reaches the client. The matched sequence itself is never emitted.

use serde_json::Value as JsonValue;

/// OpenAI's cap on the number of stop sequences.
pub const MAX_STOP_SEQUENCES: usize = 4;

/// `stop`'s wire value: absent or `null` is no sequence; a string is one; an
/// array is 1 to [`MAX_STOP_SEQUENCES`] of them. An empty string, an empty
/// array, a non-string element or more than four is refused.
pub fn parse_stop(value: Option<&JsonValue>) -> Result<Vec<String>, String> {
    let sequences = match value {
        None | Some(JsonValue::Null) => return Ok(Vec::new()),
        Some(JsonValue::String(one)) => vec![one.clone()],
        Some(JsonValue::Array(many)) => {
            if many.is_empty() || many.len() > MAX_STOP_SEQUENCES {
                return Err(format!(
                    "stop must be a string or an array of 1 to {MAX_STOP_SEQUENCES} strings (got {} entries)",
                    many.len()
                ));
            }
            many.iter()
                .map(|s| s.as_str().map(str::to_owned))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| "stop must be a string or an array of strings".to_owned())?
        }
        Some(_) => return Err("stop must be a string or an array of strings".to_owned()),
    };
    if sequences.iter().any(String::is_empty) {
        return Err("stop sequences must not be empty".to_owned());
    }
    Ok(sequences)
}

/// What one [`StopMatcher::feed`] makes available.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Fed {
    /// Text safe to emit: nothing in it is, or starts, a stop sequence.
    pub text: String,
    /// A sequence matched: `text` is everything before it, and nothing after
    /// it will ever be emitted.
    pub fired: bool,
}

/// Matches stop sequences over text that arrives in pieces.
pub struct StopMatcher {
    sequences: Vec<String>,
    hold: String,
    fired: bool,
}

impl StopMatcher {
    /// A matcher over `sequences` (non-empty strings, as [`parse_stop`]
    /// returns them).
    pub fn new(sequences: Vec<String>) -> Self {
        Self { sequences, hold: String::new(), fired: false }
    }

    /// Whether a sequence has matched.
    pub fn fired(&self) -> bool {
        self.fired
    }

    /// Feed the next piece of content text. Once a sequence has matched,
    /// everything is swallowed.
    pub fn feed(&mut self, text: &str) -> Fed {
        if self.fired {
            return Fed::default();
        }
        self.hold.push_str(text);
        // The earliest match wins: the text before it is the answer.
        let earliest = self.sequences.iter().filter_map(|s| self.hold.find(s.as_str())).min();
        if let Some(at) = earliest {
            self.hold.truncate(at);
            self.fired = true;
            return Fed { text: std::mem::take(&mut self.hold), fired: true };
        }
        let held = self.sequences.iter().map(|s| prefix_holdback(&self.hold, s)).max().unwrap_or(0);
        let safe = self.hold.len() - held;
        Fed { text: self.hold.drain(..safe).collect(), fired: false }
    }

    /// The text held back as a possible prefix, released: the content it
    /// might have continued has ended (a tool call interrupted it, or the
    /// generation finished).
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.hold)
    }
}

/// The length of the longest suffix of `text` that is a proper, non-empty
/// prefix of `sequence` (cut at a character boundary of `sequence`).
fn prefix_holdback(text: &str, sequence: &str) -> usize {
    (1..sequence.len())
        .rev()
        .filter(|&len| sequence.is_char_boundary(len))
        .find(|&len| text.ends_with(&sequence[..len]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn matcher(sequences: &[&str]) -> StopMatcher {
        StopMatcher::new(sequences.iter().map(|s| s.to_string()).collect())
    }

    /// Every piece fed in turn: the emitted text, and whether it fired.
    fn run(m: &mut StopMatcher, pieces: &[&str]) -> (String, bool) {
        let mut out = String::new();
        for piece in pieces {
            out.push_str(&m.feed(piece).text);
        }
        if !m.fired() {
            out.push_str(&m.flush());
        }
        (out, m.fired())
    }

    #[test]
    fn stop_parses_a_string_or_up_to_four_strings() {
        assert_eq!(parse_stop(None), Ok(vec![]));
        assert_eq!(parse_stop(Some(&json!(null))), Ok(vec![]));
        assert_eq!(parse_stop(Some(&json!("END"))), Ok(vec!["END".to_owned()]));
        assert_eq!(parse_stop(Some(&json!(["a", "b", "c", "d"]))).unwrap().len(), 4);
    }

    #[test]
    fn stop_refuses_what_is_not_one_to_four_non_empty_strings() {
        for bad in [json!(""), json!([]), json!(["a", "b", "c", "d", "e"]), json!(["a", 1]), json!([""]), json!(7), json!({})] {
            assert!(parse_stop(Some(&bad)).is_err(), "{bad} must be refused");
        }
    }

    #[test]
    fn the_sequence_and_what_follows_are_never_emitted() {
        let mut m = matcher(&["END"]);
        assert_eq!(run(&mut m, &["one two END three"]), ("one two ".to_owned(), true));
        assert_eq!(m.feed("more").text, "");
    }

    #[test]
    fn a_sequence_split_across_pieces_fires_and_leaks_no_fragment() {
        let mut m = matcher(&["END"]);
        let first = m.feed("one E");
        assert_eq!(first, Fed { text: "one ".to_owned(), fired: false });
        let second = m.feed("N");
        assert_eq!(second, Fed { text: String::new(), fired: false });
        assert_eq!(m.feed("D tail"), Fed { text: String::new(), fired: true });
    }

    #[test]
    fn a_held_prefix_that_does_not_complete_is_released() {
        let mut m = matcher(&["END"]);
        assert_eq!(run(&mut m, &["one E", "Nx", " two"]), ("one ENx two".to_owned(), false));
        let mut m = matcher(&["END"]);
        assert_eq!(run(&mut m, &["one EN"]), ("one EN".to_owned(), false), "flushed at the end");
    }

    #[test]
    fn a_sequence_not_aligned_to_pieces_fires() {
        // A token boundary lands inside the sequence on both sides.
        let mut m = matcher(&["\n\nQ:"]);
        assert_eq!(run(&mut m, &["answer.\n", "\nQ", ": next"]), ("answer.".to_owned(), true));
    }

    #[test]
    fn four_sequences_each_fire_and_the_earliest_wins() {
        for (text, want) in [("a STOP1 b", "a "), ("a b STOP4", "a b "), ("x ### y", "x ")] {
            let mut m = matcher(&["STOP1", "STOP2", "###", "STOP4"]);
            assert_eq!(run(&mut m, &[text]), (want.to_owned(), true), "{text}");
        }
        let mut m = matcher(&["late", "ear"]);
        assert_eq!(run(&mut m, &["an early late"]), ("an ".to_owned(), true));
    }

    #[test]
    fn a_multibyte_sequence_holds_back_at_character_boundaries() {
        let mut m = matcher(&["§§"]);
        assert_eq!(m.feed("a §").text, "a ");
        assert_eq!(m.feed("§ b"), Fed { text: String::new(), fired: true });
    }
}
