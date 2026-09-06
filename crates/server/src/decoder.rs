//! The incremental output decoder (GitHub #68): splits a generated token
//! stream into `reasoning` / `content` channels as text becomes available,
//! instead of decoding each token in isolation and string-splitting the
//! result.
//!
//! Two concerns compose here:
//! - **Byte completeness** ([`TokenDecoder`], provider-specific): turning
//!   token ids into text without ever handing out an incomplete multi-byte
//!   character. The real tokenizer's implementation lives in
//!   `artifact_template.rs`; the placeholder's is trivial (ASCII only).
//! - **Marker splitting** ([`OutputDecoder`], this module, transport- and
//!   tokenizer-independent): watching already-valid text for the `</think>`
//!   marker and switching from the reasoning channel to the content channel
//!   exactly once, holding back only what could still be a marker prefix.
//!
//! [`OutputDecoder`] is fed by [`TokenDecoder`] chunks but only ever sees
//! valid UTF-8 text — its own unit tests (below) pin the marker-holdback
//! contract with plain string chunks, no tokenizer or GPU required.

use ignis_core::TokenId;

/// The marker the chat template emits between the model's reasoning and its
/// answer (Qwen 3.8's wire contract — GitHub #68).
const CLOSE_MARKER: &str = "</think>";

/// Which half of the response a decoded chunk of text belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    Reasoning,
    Content,
}

/// One decoded chunk, ready to append to its channel's accumulated text (or
/// stream as `delta.reasoning_content` / `delta.content`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delta {
    pub channel: Channel,
    pub text: String,
}

/// Turns generated token ids into text, one push at a time, without ever
/// returning an incomplete multi-byte character (provider-specific: the real
/// tokenizer's implementation holds back genuinely incomplete UTF-8 until
/// the completing token arrives).
pub trait TokenDecoder: Send {
    /// Feed the next token id; returns the text it makes newly available
    /// (empty when nothing is confirmed yet).
    fn push(&mut self, token: TokenId) -> String;
    /// Flush whatever is left at the end of generation (a genuinely
    /// incomplete trailing sequence surfaces as the replacement character,
    /// never as dropped or corrupted bytes).
    fn finish(&mut self) -> String;
}

/// The channel-splitting decoder: wraps a [`TokenDecoder`] and emits
/// `(channel, text)` deltas, switching from `Reasoning` to `Content` exactly
/// once, at the `</think>` marker.
///
/// Which channel a request starts in is not guessed from the text — it is
/// told up front, from the request's already-resolved thinking options
/// ([`enabled`] mirrors `ThinkingOptions::enable_thinking`). That is what
/// makes a generation that hits its token budget before ever emitting
/// `</think>` come out as partial *reasoning* with empty content (GitHub #68
/// user story 19), rather than the other way around: whether a `</think>`
/// is even expected is known before the first token arrives, not inferred
/// after the fact from its absence.
///
/// **A documented deviation from the reference's whole-string split rule.**
/// The reference parses a *finished* response as reasoning between the
/// *last* `<think>` and the *first* `</think>`, content after the *last*
/// `</think>`. This decoder never scans for `<think>` at all (the real
/// template only ever emits it as part of the *prompt*, never inside
/// generated tokens, so there is nothing to find) and switches channel at
/// the *first* `</think>`, permanently. Matching "last `</think>`" exactly
/// would require holding every byte until the stream ends before emitting
/// anything as reasoning — directly at odds with story 19 (a truncated
/// generation must still show its partial reasoning) and story 20 (a
/// streaming client wants a live "thinking…" indicator, not everything at
/// once). For the one `</think>` a well-formed generation actually emits,
/// first and last coincide and this is exactly the reference's rule; it
/// only diverges if the model's own text happens to repeat the literal
/// string `</think>`, which does not happen in practice for this template.
pub struct OutputDecoder {
    tokens: Box<dyn TokenDecoder>,
    channel: Channel,
    /// Text observed but not yet safe to emit: it could still be a prefix of
    /// `</think>`, or (while still in `Reasoning`, before any non-newline
    /// text has been confirmed, or right before a marker) a run of
    /// newlines pending the leading/surrounding-newline trim.
    hold: String,
    /// Whether real (non-newline) text has already been emitted in the
    /// current channel — gates the leading-newline trim, which only applies
    /// to the *start* of a channel's text.
    started: bool,
}

impl OutputDecoder {
    /// A decoder starting in the `Reasoning` channel when `enabled` (the
    /// request's resolved `enable_thinking`), `Content` otherwise — a
    /// thinking-disabled request's output never contains a `</think>`
    /// marker to watch for, so no scanning happens at all.
    pub fn new(tokens: Box<dyn TokenDecoder>, enabled: bool) -> Self {
        Self {
            tokens,
            channel: if enabled {
                Channel::Reasoning
            } else {
                Channel::Content
            },
            hold: String::new(),
            started: false,
        }
    }

    /// Feed the next generated tokens; returns the deltas they make
    /// available (zero, one, or more — a single token can both close out
    /// held-back reasoning and open the content channel).
    pub fn push(&mut self, tokens: &[TokenId]) -> Vec<Delta> {
        let mut deltas = Vec::new();
        for &token in tokens {
            let text = self.tokens.push(token);
            if !text.is_empty() {
                self.feed(&text, &mut deltas);
            }
        }
        deltas
    }

    /// Flush the token decoder and this decoder's own held-back text.
    pub fn finish(&mut self) -> Vec<Delta> {
        let mut deltas = Vec::new();
        let tail = self.tokens.finish();
        if !tail.is_empty() {
            self.feed(&tail, &mut deltas);
        }
        // Whatever is still held: a genuine partial `</think>` prefix is
        // dropped (never leaked as garbage text); anything else — ordinary
        // text, including trailing newlines that turned out not to precede
        // a marker — is published as-is in the current channel (no more
        // trimming: a truncated stream has no confirmed boundary to trim
        // against).
        let held = std::mem::take(&mut self.hold);
        let text = drop_trailing_marker_prefix(held);
        if !text.is_empty() {
            deltas.push(Delta {
                channel: self.channel,
                text,
            });
        }
        deltas
    }

    /// Only scans for the marker while still in `Reasoning` — once switched
    /// to `Content` there is nothing left to hold back for, so new text
    /// streams straight through (after the one-time leading-newline trim).
    fn feed(&mut self, text: &str, deltas: &mut Vec<Delta>) {
        if self.channel == Channel::Content {
            self.emit_content(text, deltas);
            return;
        }
        self.hold.push_str(text);
        if let Some(idx) = self.hold.find(CLOSE_MARKER) {
            let before = self.hold[..idx].to_string();
            let after = self.hold[idx + CLOSE_MARKER.len()..].to_string();
            self.hold.clear();
            let reasoning = trim_surrounding_newlines(&before);
            if !reasoning.is_empty() {
                deltas.push(Delta {
                    channel: Channel::Reasoning,
                    text: reasoning,
                });
            }
            self.channel = Channel::Content;
            self.started = false;
            if !after.is_empty() {
                self.emit_content(&after, deltas);
            }
            return;
        }
        // No marker yet: hold back the longest suffix that could still
        // become one, or that could still be leading/trailing newlines
        // pending confirmation. Everything before that is safe to publish
        // now as ordinary reasoning text.
        let holdback = holdback_len(&self.hold);
        let safe_len = self.hold.len() - holdback;
        if safe_len == 0 {
            return;
        }
        let safe: String = self.hold.drain(..safe_len).collect();
        let text = if !self.started {
            let trimmed = safe.trim_start_matches('\n');
            if !trimmed.is_empty() {
                self.started = true;
            }
            trimmed.to_string()
        } else {
            safe
        };
        if !text.is_empty() {
            deltas.push(Delta {
                channel: Channel::Reasoning,
                text,
            });
        }
    }

    fn emit_content(&mut self, text: &str, deltas: &mut Vec<Delta>) {
        let text = if !self.started {
            let trimmed = text.trim_start_matches('\n');
            if trimmed.is_empty() {
                return;
            }
            self.started = true;
            trimmed
        } else {
            text
        };
        deltas.push(Delta {
            channel: Channel::Content,
            text: text.to_string(),
        });
    }
}

/// The length of the trailing suffix of `text` that must be held back: the
/// longer of (a) the longest suffix that is a proper prefix of the close
/// marker, and (b) the longest all-newline suffix.
fn holdback_len(text: &str) -> usize {
    let marker_prefix = (1..CLOSE_MARKER.len())
        .rev()
        .find(|&len| text.ends_with(&CLOSE_MARKER[..len]))
        .unwrap_or(0);
    let newline_run = text.len() - text.trim_end_matches('\n').len();
    marker_prefix.max(newline_run)
}

/// Trim newlines from both ends (the reasoning channel's "surrounding
/// newlines trimmed" rule).
fn trim_surrounding_newlines(text: &str) -> String {
    text.trim_matches('\n').to_string()
}

/// Drop a trailing proper-prefix-of-the-marker suffix, if `text` ends with
/// one (the finish-time "genuine partial marker is dropped" rule).
fn drop_trailing_marker_prefix(mut text: String) -> String {
    for len in (1..CLOSE_MARKER.len()).rev() {
        if text.ends_with(&CLOSE_MARKER[..len]) {
            text.truncate(text.len() - len);
            break;
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test-double [`TokenDecoder`] driven by a fixed script of
    /// per-token chunks (simulating whatever byte/marker split a real
    /// tokenizer might produce) — the marker-holdback contract is pinned
    /// against this, with no real tokenizer needed.
    struct ScriptedDecoder {
        chunks: std::collections::VecDeque<String>,
        tail: String,
    }

    impl ScriptedDecoder {
        fn new(chunks: &[&str]) -> Self {
            Self {
                chunks: chunks.iter().map(|s| s.to_string()).collect(),
                tail: String::new(),
            }
        }

        fn with_tail(chunks: &[&str], tail: &str) -> Self {
            let mut d = Self::new(chunks);
            d.tail = tail.to_string();
            d
        }
    }

    impl TokenDecoder for ScriptedDecoder {
        fn push(&mut self, _token: TokenId) -> String {
            self.chunks.pop_front().unwrap_or_default()
        }
        fn finish(&mut self) -> String {
            std::mem::take(&mut self.tail)
        }
    }

    /// Drive a scripted decoder through one push per chunk, then finish;
    /// returns the deltas in order.
    fn run(chunks: &[&str], enabled: bool) -> Vec<Delta> {
        run_with_tail(chunks, "", enabled)
    }

    fn run_with_tail(chunks: &[&str], tail: &str, enabled: bool) -> Vec<Delta> {
        let inner = Box::new(ScriptedDecoder::with_tail(chunks, tail));
        let mut decoder = OutputDecoder::new(inner, enabled);
        let ids: Vec<TokenId> = (0..chunks.len() as u32).collect();
        let mut deltas = decoder.push(&ids);
        deltas.extend(decoder.finish());
        deltas
    }

    fn concat(deltas: &[Delta], channel: Channel) -> String {
        deltas
            .iter()
            .filter(|d| d.channel == channel)
            .map(|d| d.text.as_str())
            .collect()
    }

    #[test]
    fn no_marker_at_all_is_entirely_reasoning_when_enabled() {
        // A degenerate case (the model never closes): per story 19, an
        // unclosed reasoning span stays reasoning, not content.
        let deltas = run(&["hello ", "world"], true);
        assert_eq!(concat(&deltas, Channel::Reasoning), "hello world");
        assert_eq!(concat(&deltas, Channel::Content), "");
    }

    #[test]
    fn thinking_disabled_is_entirely_content_from_the_first_token() {
        let deltas = run(&["hello ", "world"], false);
        assert_eq!(concat(&deltas, Channel::Content), "hello world");
        assert_eq!(concat(&deltas, Channel::Reasoning), "");
    }

    #[test]
    fn a_marker_in_a_single_chunk_switches_channel_once() {
        let deltas = run(&["thinking\n</think>\n\nthe answer"], true);
        assert_eq!(concat(&deltas, Channel::Reasoning), "thinking");
        assert_eq!(concat(&deltas, Channel::Content), "the answer");
    }

    #[test]
    fn the_marker_never_leaks_into_either_channel() {
        let deltas = run(&["a</think>b"], true);
        for d in &deltas {
            assert!(!d.text.contains("<think"), "{d:?}");
            assert!(!d.text.contains("think>"), "{d:?}");
        }
    }

    #[test]
    fn a_marker_split_across_two_chunks_is_still_found() {
        let deltas = run(&["thinking</thi", "nk>answer"], true);
        assert_eq!(concat(&deltas, Channel::Reasoning), "thinking");
        assert_eq!(concat(&deltas, Channel::Content), "answer");
    }

    #[test]
    fn a_marker_split_three_ways_is_still_found() {
        let deltas = run(&["thinking</", "thi", "nk>answer"], true);
        assert_eq!(concat(&deltas, Channel::Reasoning), "thinking");
        assert_eq!(concat(&deltas, Channel::Content), "answer");
    }

    #[test]
    fn a_marker_at_the_very_first_chunk() {
        let deltas = run(&["</think>answer"], true);
        assert_eq!(concat(&deltas, Channel::Reasoning), "");
        assert_eq!(concat(&deltas, Channel::Content), "answer");
    }

    #[test]
    fn a_marker_at_the_very_last_chunk() {
        let deltas = run(&["reasoning ", "text</think>"], true);
        assert_eq!(concat(&deltas, Channel::Reasoning), "reasoning text");
        assert_eq!(concat(&deltas, Channel::Content), "");
    }

    #[test]
    fn surrounding_newlines_around_the_marker_are_trimmed() {
        let deltas = run(&["reasoning\n\n</think>\n\nanswer"], true);
        assert_eq!(concat(&deltas, Channel::Reasoning), "reasoning");
        assert_eq!(concat(&deltas, Channel::Content), "answer");
    }

    #[test]
    fn leading_newlines_of_reasoning_itself_are_trimmed() {
        let deltas = run(&["\n\nreasoning</think>answer"], true);
        assert_eq!(concat(&deltas, Channel::Reasoning), "reasoning");
    }

    #[test]
    fn interior_newlines_are_not_mistaken_for_trailing_ones() {
        // The newline run right after "one" is interior (more reasoning
        // text follows), not trailing — it must survive in the output.
        let deltas = run(&["one\n\ntwo</think>answer"], true);
        assert_eq!(concat(&deltas, Channel::Reasoning), "one\n\ntwo");
    }

    #[test]
    fn a_multi_byte_character_split_across_two_tokens_arrives_whole() {
        // The token decoder layer is what guarantees this in production
        // (it never hands out a fragment); this pins that OutputDecoder
        // does not itself further fragment whatever whole text it is
        // given — an empty chunk (held below it) followed by the
        // completed character.
        let deltas = run(&["résumé\n\n", "", "\u{00e9}</think>ok"], true);
        assert!(concat(&deltas, Channel::Reasoning).ends_with('\u{00e9}'));
    }

    #[test]
    fn a_stream_that_ends_mid_marker_drops_the_partial_marker() {
        // Truncated right after "</thi" — never confirmed as a real
        // marker, so it must not leak as text, and reasoning stays
        // reasoning (no </think> was ever seen).
        let deltas = run(&["reasoning</thi"], true);
        assert_eq!(concat(&deltas, Channel::Reasoning), "reasoning");
        assert!(deltas.iter().all(|d| !d.text.contains('<')));
    }

    #[test]
    fn finish_flushes_a_genuine_trailing_newline_as_text_not_a_trim() {
        // No marker ever arrives, so the trailing newline was never
        // "surrounding a marker" — it must be published, not trimmed away.
        let deltas = run(&["reasoning\n"], true);
        assert_eq!(concat(&deltas, Channel::Reasoning), "reasoning\n");
    }

    #[test]
    fn finish_flushes_the_token_decoders_own_tail() {
        let deltas = run_with_tail(&["reasoning</think>ans"], "wer", true);
        assert_eq!(concat(&deltas, Channel::Content), "answer");
    }

    #[test]
    fn streaming_and_non_streaming_agree_by_construction() {
        // The whole point of one shared decoder (story 24): feed it token
        // by token, or all at once — same deltas, concatenated.
        let mut incremental = OutputDecoder::new(
            Box::new(ScriptedDecoder::new(&["think", "ing</thi", "nk>answer"])),
            true,
        );
        let mut streamed = Vec::new();
        for i in 0..3u32 {
            streamed.extend(incremental.push(&[i]));
        }
        streamed.extend(incremental.finish());

        let mut whole = OutputDecoder::new(
            Box::new(ScriptedDecoder::new(&["think", "ing</thi", "nk>answer"])),
            true,
        );
        let mut batched = whole.push(&[0, 1, 2]);
        batched.extend(whole.finish());

        assert_eq!(concat(&streamed, Channel::Reasoning), concat(&batched, Channel::Reasoning));
        assert_eq!(concat(&streamed, Channel::Content), concat(&batched, Channel::Content));
    }
}
