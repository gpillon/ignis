//! Where a `/v1/decide` state will be resumed (GitHub #270, spec
//! `docs/specs/decide/16-reuse-boundaries.md`, ADR 0029 as amended).
//!
//! A hybrid resumes only where its prefill captured state, so the positions a
//! later request will resume from have to be predicted before the prefill.
//! The scheduler publishes whatever **reuse boundaries** a request carries;
//! this module holds the two predictions that need more than one request to
//! make:
//!
//! - the **observed fork** — [`ForkHistory`], the match keys of recent parts
//!   states at every run end, and nothing else;
//! - the **fan-out head** — [`common_head`], the longest common prefix of one
//!   fan-out's rendered questions.
//!
//! Neither decides what a request may *claim*. Matching stays by content (ADR
//! 0029): these only decide where state is kept.

use std::collections::VecDeque;

use ignis_core::identity::{MatchKey, MediaKey, PromptContent};
use ignis_core::types::RequestInput;

/// How many run ends the fork history remembers. An implementation constant,
/// not an operator flag (spec 16): a key is 20 bytes, and 4,096 of them is a
/// few hundred states of a few parts each — far more than the eight retained
/// slots could ever keep a boundary for.
pub const FORK_HISTORY_KEYS: usize = 4096;

/// The match keys of recent parts states at each **run end** — the token
/// position where `parts[0..=i]` ends, for every `i` (GitHub #270). No state,
/// no text: a key cannot be turned back into the prompt it names.
///
/// Keyed by the match key *at* the run end, so everything before the run —
/// the template's head, the question kind's system text — is part of the key,
/// and a run seen under another instruction does not count.
#[derive(Debug, Default)]
pub struct ForkHistory {
    /// Least recently seen first.
    seen: VecDeque<RunEnd>,
}

/// One run end: where it is, and what the prompt up to there is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunEnd {
    pub tokens: u32,
    pub key: MatchKey,
}

impl ForkHistory {
    /// The longest of `run_ends` a recent request also had — the end of the
    /// longest run of leading parts it shares with one — or `None`.
    pub fn longest_seen(&self, run_ends: &[RunEnd]) -> Option<u32> {
        run_ends
            .iter()
            .filter(|end| self.seen.contains(end))
            .map(|end| end.tokens)
            .max()
    }

    /// Remember every one of `run_ends` as just seen, forgetting the least
    /// recently seen beyond [`FORK_HISTORY_KEYS`].
    pub fn record(&mut self, run_ends: &[RunEnd]) {
        for end in run_ends {
            if let Some(at) = self.seen.iter().position(|seen| seen == end) {
                self.seen.remove(at);
            }
            self.seen.push_back(*end);
        }
        while self.seen.len() > FORK_HISTORY_KEYS {
            self.seen.pop_front();
        }
    }
}

/// `input`'s run ends at `ends` — each an exact token prefix of its prompt —
/// named by the prompt's match key there, images included.
pub fn run_ends(input: &RequestInput, ends: &[u32]) -> Vec<RunEnd> {
    let media = media_keys(input);
    let keys = PromptContent::new(&input.tokens, &media).keys_at(ends);
    ends.iter()
        .zip(keys)
        .map(|(&tokens, key)| RunEnd { tokens, key })
        .collect()
}

/// The longest common prefix of `prompts`, on match keys: token ids *and*
/// media identity, so two prompts whose pictures differ share nothing from the
/// first placeholder of the first picture that does.
///
/// 0 for fewer than two prompts: a head only one question would ever claim is
/// a slot spent for nothing.
pub fn common_head(prompts: &[&RequestInput]) -> u32 {
    let [first, rest @ ..] = prompts else {
        return 0;
    };
    if rest.is_empty() {
        return 0;
    }
    let first_media = media_keys(first);
    let mut head = first.tokens.len();
    for other in rest {
        let shared = first
            .tokens
            .iter()
            .zip(&other.tokens)
            .take_while(|(a, b)| a == b)
            .count();
        head = head.min(shared);
        // The first item the two disagree about ends what they share where it
        // begins: its placeholders are the same ids whatever the picture.
        let other_media = media_keys(other);
        if let Some(differs) = first_media
            .iter()
            .zip(other_media.iter().map(Some).chain(std::iter::repeat(None)))
            .find(|(mine, theirs)| *theirs != Some(*mine))
        {
            head = head.min(differs.0.begin as usize);
        }
    }
    u32::try_from(head).unwrap_or(u32::MAX)
}

fn media_keys(input: &RequestInput) -> Vec<MediaKey> {
    input
        .multimodal
        .as_ref()
        .map(|multimodal| multimodal.media.iter().map(MediaKey::from).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn end(tokens: u32, key: u8) -> RunEnd {
        RunEnd { tokens, key: MatchKey::from_bytes([key; 16]) }
    }

    #[test]
    fn the_longest_run_end_seen_before_is_the_fork() {
        let mut history = ForkHistory::default();
        assert_eq!(history.longest_seen(&[end(10, 1), end(20, 2)]), None, "nothing seen yet");
        history.record(&[end(10, 1), end(20, 2)]);
        assert_eq!(history.longest_seen(&[end(10, 1), end(20, 3)]), Some(10), "the first part repeated");
        assert_eq!(history.longest_seen(&[end(10, 1), end(20, 2)]), Some(20), "the whole state repeated");
        assert_eq!(
            history.longest_seen(&[end(10, 9), end(20, 2)]),
            Some(20),
            "a key names the whole prefix, so a hit past a miss is still a hit"
        );
    }

    #[test]
    fn the_least_recently_seen_is_forgotten_first() {
        let mut history = ForkHistory::default();
        let many: Vec<RunEnd> = (0..FORK_HISTORY_KEYS as u32)
            .map(|i| RunEnd { tokens: i, key: MatchKey::from_bytes(i.to_le_bytes().repeat(4).try_into().unwrap()) })
            .collect();
        history.record(&many);
        // Seeing the oldest again makes it the newest.
        history.record(&many[..1]);
        history.record(&[end(7, 0xEE)]);
        assert_eq!(history.longest_seen(&many[..1]), Some(0), "refreshed, so kept");
        assert_eq!(history.longest_seen(&many[1..2]), None, "the least recently seen went");
        assert_eq!(history.longest_seen(&[end(7, 0xEE)]), Some(7));
    }
}
