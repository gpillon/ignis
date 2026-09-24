//! The **thinking budget**: a cap on how many tokens a request may spend
//! inside its reasoning block before the scheduler closes the block for it.
//!
//! Qwen3.8 at its default effort can reason past any `max_tokens` a coding
//! agent sets and never answer (2026-09-24, `docs/findings/
//! 2026-09-24-reasoning-effort-on-coding-tasks.md`: 7 of 16 runs at `xhigh`
//! still thinking at 16K). The model card's remedy is to close the block by
//! hand once a budget is spent: append a short hand-off sentence and
//! `</think>`, then let the model answer from what it has. This module is
//! that, on the scheduler's side of the seam.
//!
//! Nothing new crosses the `Compute` seam. The close is forced the way a
//! constrained decode forces its digits ([`crate::constrained`]): one
//! single-token permitted set per round, with the same one-round lag — a
//! round returns the token the previous call drew and draws the next under
//! the set it carries. So forcing starts on the round *after* the budget is
//! reached, and the one token already drawn freely in between is emitted as
//! the model wrote it.
//!
//! A request that closes its block by itself is never forced, and one that
//! closes it in the gap between the budget and the forced `</think>` stops
//! being forced from that token on. The draw already made in that same round
//! was the close's next token, so at most one token of the close follows a
//! natural `</think>`; the close starts with a line break for that reason.

use std::sync::Arc;

use crate::constrained::PermittedSet;
use crate::types::TokenId;

/// The tokens of a request's generation a budget always leaves for the
/// answer (spec server/08): a close forced so late that the answer after it
/// is cut by `max_tokens` is a turn with no answer all the same.
///
/// Measured (#265, `docs/findings/2026-09-24-thinking-budget-default.md`):
/// the longest answer after a close on the coding sweep was ~1,650 tokens,
/// and the reserve also holds the close itself (~25 tokens) and a
/// speculative round's overshoot past the budget (up to its width). 2,048
/// covers them with a quarter to spare.
pub const ANSWER_RESERVE: u32 = 2048;

/// The budget a request actually runs under: what it asked for, clamped so
/// that [`ANSWER_RESERVE`] tokens of its `generation` (the tokens it may
/// generate at all) stay for the answer. `None` when there is no room above
/// the reserve — no budget, rather than a close forced at the first token.
pub fn effective(budget: Option<u32>, generation: u32) -> Option<u32> {
    let room = generation.checked_sub(ANSWER_RESERVE).filter(|&room| room > 0)?;
    budget.map(|budget| budget.min(room))
}

/// What a request's thinking budget did, reported once on its finish event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetOutcome {
    /// The budget it ran under ([`effective`]).
    pub budget: u32,
    /// The tokens it had emitted when the forced close began — `None` when
    /// the close was not forced (the model closed its block itself, or never
    /// reached the budget).
    pub forced_at: Option<u32>,
}

/// The model's own way to close a reasoning block: the tokens forced, in
/// order, once a budget is spent. `think_end` is the block's closing
/// marker and must appear in `tokens`; whatever follows it (the line break
/// before the answer) is forced too.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinkingClose {
    tokens: Arc<[TokenId]>,
    think_end: TokenId,
    think_end_at: usize,
}

impl ThinkingClose {
    /// A close sequence ending the block at `think_end`. Refused when the
    /// marker is not in the sequence: a close that never closes would force
    /// its tokens and leave the model still reasoning.
    pub fn new(tokens: Vec<TokenId>, think_end: TokenId) -> Result<Self, String> {
        let think_end_at = tokens
            .iter()
            .position(|&t| t == think_end)
            .ok_or_else(|| format!("the thinking close does not contain its end marker {think_end}"))?;
        Ok(Self {
            tokens: tokens.into(),
            think_end,
            think_end_at,
        })
    }

    /// The reasoning block's closing marker.
    pub fn think_end(&self) -> TokenId {
        self.think_end
    }

    /// The forced tokens, in order.
    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }
}

/// One request's progress against its budget.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BudgetState {
    /// The request has emitted its reasoning block's closing marker.
    closed: bool,
    /// The emitted-token index the close's first token is drawn for, once
    /// forcing has started.
    forced_from: Option<u32>,
}

impl BudgetState {
    /// The permitted set for the draw this round makes, for a request that
    /// has emitted `emitted` tokens and may spend `budget` of them
    /// reasoning — or `None` to draw freely.
    ///
    /// The draw is for token index `emitted + 1`: this round emits index
    /// `emitted`, which the previous round drew. Forcing starts at the first
    /// round that finds the budget spent and the block still open.
    pub fn permitted(&mut self, budget: Option<u32>, close: &ThinkingClose, emitted: u32) -> Option<PermittedSet> {
        let budget = budget?;
        let draw = emitted + 1;
        let from = match self.forced_from {
            Some(from) => from,
            None if !self.closed && emitted >= budget => {
                self.forced_from = Some(draw);
                draw
            }
            None => return None,
        };
        let step = draw.checked_sub(from)? as usize;
        close.tokens.get(step).map(|&token| PermittedSet::from([token]))
    }

    /// Record the token emitted at `index` (0-based over the request's
    /// output).
    pub fn commit(&mut self, close: &ThinkingClose, index: u32, token: TokenId) {
        if token != close.think_end || self.closed {
            return;
        }
        self.closed = true;
        // A natural close before the forced one: stop forcing.
        if let Some(from) = self.forced_from {
            if (index as usize) < from as usize + close.think_end_at {
                self.forced_from = None;
            }
        }
    }

    /// Whether this request's close was forced (it reached its budget with
    /// the block open).
    pub fn forced(&self) -> bool {
        self.forced_from.is_some()
    }

    /// The tokens emitted before the close's first forced token, when the
    /// close was forced.
    pub fn forced_at(&self) -> Option<u32> {
        self.forced_from
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const END: TokenId = 99;

    fn close() -> ThinkingClose {
        // "\n\n", "hand", "off", "</think>", "\n\n"
        ThinkingClose::new(vec![10, 11, 12, END, 10], END).unwrap()
    }

    /// Drive a request round by round: each round emits the token the
    /// previous round drew (free draws come from `model`), exactly the
    /// one-round lag the leaf has.
    fn run(budget: Option<u32>, model: impl Fn(u32) -> TokenId, rounds: u32) -> (Vec<TokenId>, BudgetState) {
        let close = close();
        let mut state = BudgetState::default();
        let mut out = Vec::new();
        // The prefill drew token 0 freely.
        let mut pending = model(0);
        for _ in 0..rounds {
            let emitted = out.len() as u32;
            let set = state.permitted(budget, &close, emitted);
            state.commit(&close, emitted, pending);
            out.push(pending);
            pending = match set {
                Some(set) => set[0],
                None => model(emitted + 1),
            };
        }
        (out, state)
    }

    #[test]
    fn a_close_without_its_end_marker_is_refused() {
        assert!(ThinkingClose::new(vec![1, 2, 3], END).is_err());
    }

    #[test]
    fn no_budget_never_forces() {
        let (out, state) = run(None, |_| 5, 20);
        assert!(out.iter().all(|&t| t == 5));
        assert!(!state.forced());
    }

    #[test]
    fn a_spent_budget_forces_the_whole_close_one_round_later() {
        let (out, state) = run(Some(4), |_| 5, 12);
        // Tokens 0..=4 are the model's (index 4 was drawn freely before the
        // round that found the budget spent), then the close, then free.
        assert_eq!(out, vec![5, 5, 5, 5, 5, 10, 11, 12, END, 10, 5, 5]);
        assert!(state.forced());
    }

    #[test]
    fn a_block_closed_before_the_budget_is_never_forced() {
        let (out, state) = run(Some(6), |i| if i == 2 { END } else { 5 }, 12);
        assert_eq!(out.iter().filter(|&&t| t == END).count(), 1);
        assert!(!state.forced());
    }

    #[test]
    fn a_natural_close_inside_the_lag_stops_the_forcing() {
        // Budget 4: the round that emits index 4 starts forcing (draws the
        // close's first token for index 5). The model's own index 4 is
        // `</think>`, so the close stops after that one drawn token.
        let (out, state) = run(Some(4), |i| if i == 4 { END } else { 5 }, 10);
        assert_eq!(out, vec![5, 5, 5, 5, END, 10, 5, 5, 5, 5]);
        assert!(!state.forced());
    }
}
