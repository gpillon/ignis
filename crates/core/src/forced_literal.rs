//! A **forced literal** (GitHub #286, spec server/11): tokens the scheduler
//! makes a request generate, one single-token permitted set per round, and
//! then **releases** — the request goes on generating freely.
//!
//! `tool_choice` is what asks for one: the opening of the tool-call dialect,
//! `<tool_call>\n<function` for `"required"` or `<tool_call>\n<function=NAME>\n`
//! for a named function, after which the arguments and the closing tags are
//! the model's own. The thinking budget's forced close
//! ([`crate::thinking_budget`]) is the same shape at a position a budget
//! picks. It is not a [`crate::constrained::Schedule`]: a schedule ends its
//! run when it is spent, and a forced literal must not end anything.
//!
//! # Where it starts
//!
//! [`ForcedLiteral::at_generation`] forces the generation's first tokens.
//! The first is drawn by the prefill ([`ForcedLiteral::prefill_step`]), for
//! the one-round lag [`crate::constrained`] describes: a round returns the
//! token the previous call drew and draws the next under the set it carries.
//!
//! [`ForcedLiteral::after_reasoning`] forces the tokens right after the
//! reasoning block closes, for a request that generates with thinking on.
//! The close may be the model's own `</think>` or the thinking budget's
//! forced one; nothing is forced while the block is open.
//!
//! # After the model's own close: the joiner
//!
//! The round that commits the model's `</think>` has already drawn the token
//! after it freely, and a speculative round may have committed several more.
//! So the first forced draw follows a token nobody saw when the draw's set
//! was chosen. If that token was the literal's own first — the model opening
//! its call itself — forcing the first again would write it twice, and a
//! call opened twice does not parse.
//!
//! So the first forced token after an unseen one is the literal's
//! **second**, its **joiner**: for the tool-call opener, the line break after
//! `<tool_call>`, which is the right token after `<tool_call>` and a line
//! break in the text after anything else. The round after, the unseen token
//! has been emitted: the literal resumes from its third token when that
//! token was its first, and from its first otherwise. The literal's second
//! token must be one that may follow anything;
//! [`ForcedLiteral::after_reasoning`] refuses a literal with no second token.
//!
//! A close the budget forced leaves no unseen token — its own tokens were
//! forced up to the literal's first draw — so the literal starts there,
//! with its first token. And when the model writes the literal's first token
//! itself after the close, before forcing could start (a speculative round
//! that committed `</think>` and the opener together), nothing is forced:
//! the request is already writing what the literal would have.

use std::sync::Arc;

use crate::constrained::PermittedSet;
use crate::types::TokenId;

/// Where a [`ForcedLiteral`] starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForcedStart {
    /// At the generation's first token.
    Generation,
    /// Right after the reasoning block closes at `think_end`, the block's
    /// closing marker.
    AfterReasoning { think_end: TokenId },
}

/// The tokens a request is made to generate, and where they start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForcedLiteral {
    tokens: Arc<[TokenId]>,
    start: ForcedStart,
}

impl ForcedLiteral {
    /// A literal forced from the generation's first token. Refused when
    /// empty: a forced literal that forces nothing is a caller who believes
    /// a constraint is in effect.
    pub fn at_generation(tokens: Vec<TokenId>) -> Result<Self, String> {
        if tokens.is_empty() {
            return Err("a forced literal must carry at least one token".to_owned());
        }
        Ok(Self { tokens: tokens.into(), start: ForcedStart::Generation })
    }

    /// A literal forced right after the reasoning block closes at
    /// `think_end`. Refused without a second token, which is the joiner a
    /// close by the model itself needs (module docs).
    pub fn after_reasoning(tokens: Vec<TokenId>, think_end: TokenId) -> Result<Self, String> {
        if tokens.len() < 2 {
            return Err("a literal forced after the reasoning block needs a second token to join on".to_owned());
        }
        Ok(Self { tokens: tokens.into(), start: ForcedStart::AfterReasoning { think_end } })
    }

    /// The set the prefill's draw is restricted to: the literal's first
    /// token when it starts at the generation, else `None`.
    pub fn prefill_step(&self) -> Option<PermittedSet> {
        match self.start {
            ForcedStart::Generation => Some(PermittedSet::from([self.tokens[0]])),
            ForcedStart::AfterReasoning { .. } => None,
        }
    }
}

/// Where a literal forced after the reasoning block stands.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Phase {
    /// The block is open: nothing is forced.
    #[default]
    Reasoning,
    /// The block has closed and forcing has not started.
    Closed,
    /// Forcing started with the draw for output index `from`. `joined`: the
    /// run opens with the joiner, the token before it having been drawn
    /// unseen; `unseen_opened`: that token was the literal's first.
    Forcing { from: u32, joined: bool, unseen_opened: bool },
    /// The model wrote the literal's first token itself after the close:
    /// nothing is forced.
    Released,
}

/// One request's progress through its [`ForcedLiteral`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ForcedState {
    phase: Phase,
    /// Whether the last round's draw was a single forced token, so that the
    /// token this round emits was seen when it was chosen.
    last_draw_forced: bool,
}

impl ForcedState {
    /// The permitted set for the draw this round makes, for a request that
    /// has emitted `emitted` tokens — or `None` to draw freely.
    ///
    /// `budget` is the set the thinking budget forces this round, which
    /// wins: the budget only forces inside the reasoning block, and nothing
    /// of the literal is due before the block has closed.
    pub fn permitted(
        &mut self,
        literal: &ForcedLiteral,
        emitted: u32,
        budget: Option<&PermittedSet>,
    ) -> Option<PermittedSet> {
        let set = match budget {
            Some(_) => None,
            None => self.step(literal, emitted + 1),
        };
        self.last_draw_forced = budget.or(set.as_ref()).is_some_and(|set| set.len() == 1);
        set
    }

    fn step(&mut self, literal: &ForcedLiteral, draw: u32) -> Option<PermittedSet> {
        let at = match (literal.start, self.phase) {
            (ForcedStart::Generation, _) => draw as usize,
            (_, Phase::Reasoning | Phase::Released) => return None,
            (_, Phase::Closed) => {
                let joined = !self.last_draw_forced;
                self.phase = Phase::Forcing { from: draw, joined, unseen_opened: false };
                if joined { 1 } else { 0 }
            }
            (_, Phase::Forcing { from, joined, unseen_opened }) => match (joined, draw.checked_sub(from)? as usize) {
                (false, step) => step,
                (true, 0) => 1,
                // The joiner was the literal's second token: after an
                // unseen first, the run resumes at its third.
                (true, step) if unseen_opened => step + 1,
                (true, step) => step - 1,
            },
        };
        literal.tokens.get(at).map(|&token| PermittedSet::from([token]))
    }

    /// Record the token emitted at `index` (0-based over the request's
    /// output).
    pub fn commit(&mut self, literal: &ForcedLiteral, index: u32, token: TokenId) {
        let ForcedStart::AfterReasoning { think_end } = literal.start else {
            return;
        };
        match &mut self.phase {
            phase @ Phase::Reasoning if token == think_end => *phase = Phase::Closed,
            phase @ Phase::Closed if token == literal.tokens[0] => *phase = Phase::Released,
            Phase::Forcing { from, joined: true, unseen_opened } if index + 1 == *from => {
                *unseen_opened = token == literal.tokens[0];
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const END: TokenId = 99;
    const OPEN: TokenId = 50;
    const NL: TokenId = 51;
    /// `<tool_call>`, `\n`, `<`, `function`.
    const OPENER: [TokenId; 4] = [OPEN, NL, 52, 53];

    /// Drive a request round by round with the leaf's one-round lag: each
    /// round emits the token the previous round drew (free draws come from
    /// `model`), and the prefill drew token 0 under the literal's prefill
    /// step. `budget` is the set the thinking budget forces at a given draw.
    fn run(
        literal: &ForcedLiteral,
        model: impl Fn(u32) -> TokenId,
        budget: impl Fn(u32) -> Option<TokenId>,
        rounds: u32,
    ) -> Vec<TokenId> {
        let mut state = ForcedState::default();
        let mut out = Vec::new();
        let mut pending = literal.prefill_step().map_or_else(|| model(0), |set| set[0]);
        for _ in 0..rounds {
            let emitted = out.len() as u32;
            let budget = budget(emitted + 1).map(|token| PermittedSet::from([token]));
            let set = state.permitted(literal, emitted, budget.as_ref()).or(budget);
            state.commit(literal, emitted, pending);
            out.push(pending);
            pending = match set {
                Some(set) => set[0],
                None => model(emitted + 1),
            };
        }
        out
    }

    fn free(_: u32) -> Option<TokenId> {
        None
    }

    #[test]
    fn an_empty_literal_and_one_with_no_joiner_are_refused() {
        assert!(ForcedLiteral::at_generation(Vec::new()).is_err());
        assert!(ForcedLiteral::after_reasoning(vec![OPEN], END).is_err());
        assert!(ForcedLiteral::after_reasoning(OPENER.to_vec(), END).is_ok());
    }

    #[test]
    fn at_the_generation_the_prefill_draws_the_first_token_and_the_run_is_then_released() {
        let literal = ForcedLiteral::at_generation(OPENER.to_vec()).unwrap();
        assert_eq!(literal.prefill_step().as_deref(), Some(&[OPEN][..]));
        let out = run(&literal, |_| 5, free, 8);
        assert_eq!(out, vec![OPEN, NL, 52, 53, 5, 5, 5, 5]);
    }

    #[test]
    fn after_reasoning_nothing_is_forced_while_the_block_is_open() {
        let literal = ForcedLiteral::after_reasoning(OPENER.to_vec(), END).unwrap();
        assert_eq!(literal.prefill_step(), None);
        assert!(run(&literal, |_| 5, free, 12).iter().all(|&t| t == 5));
    }

    #[test]
    fn after_the_models_own_close_the_joiner_follows_the_unseen_token_and_the_literal_follows_it() {
        // The model closes at 3 and writes `\n\n` (7) at 4, drawn in the
        // round that emitted the close: unseen. The joiner comes next, then
        // the whole literal, then the model again.
        let literal = ForcedLiteral::after_reasoning(OPENER.to_vec(), END).unwrap();
        let out = run(&literal, |i| match i { 3 => END, 4 => 7, _ => 5 }, free, 12);
        assert_eq!(out, vec![5, 5, 5, END, 7, NL, OPEN, NL, 52, 53, 5, 5]);
    }

    #[test]
    fn an_unseen_token_that_opened_the_literal_is_not_opened_again() {
        // The model writes `<tool_call>` itself at 4, the token nobody saw:
        // the joiner is its line break, and the literal resumes at `<`.
        let literal = ForcedLiteral::after_reasoning(OPENER.to_vec(), END).unwrap();
        let out = run(&literal, |i| match i { 3 => END, 4 => OPEN, _ => 5 }, free, 10);
        assert_eq!(out, vec![5, 5, 5, END, OPEN, NL, 52, 53, 5, 5]);
        assert_eq!(out.iter().filter(|&&t| t == OPEN).count(), 1);
    }

    #[test]
    fn a_close_the_budget_forced_is_followed_by_the_literal_with_no_joiner() {
        // The budget forces `</think>`, `\n\n` (7) for draws 4 and 5: the
        // token before the literal's first draw was forced, so seen.
        let literal = ForcedLiteral::after_reasoning(OPENER.to_vec(), END).unwrap();
        let budget = |draw: u32| match draw { 4 => Some(END), 5 => Some(7), _ => None };
        let out = run(&literal, |_| 5, budget, 12);
        assert_eq!(out, vec![5, 5, 5, 5, END, 7, OPEN, NL, 52, 53, 5, 5]);
    }

    #[test]
    fn the_models_own_opener_before_forcing_starts_releases_the_literal() {
        // One speculative round committed `</think>` and `<tool_call>`
        // together; the state sees them in order before its next set.
        let literal = ForcedLiteral::after_reasoning(OPENER.to_vec(), END).unwrap();
        let mut state = ForcedState::default();
        assert_eq!(state.permitted(&literal, 0, None), None);
        for (index, token) in [5, END, 7, OPEN, NL].into_iter().enumerate() {
            state.commit(&literal, index as u32, token);
        }
        assert!((5..20).all(|emitted| state.permitted(&literal, emitted, None).is_none()));
    }

    #[test]
    fn the_budget_set_wins_and_the_literal_waits_for_it() {
        let literal = ForcedLiteral::at_generation(OPENER.to_vec()).unwrap();
        let mut state = ForcedState::default();
        let budget = PermittedSet::from([END]);
        assert_eq!(state.permitted(&literal, 0, Some(&budget)), None);
    }
}
