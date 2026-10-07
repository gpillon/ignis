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
//! with its first token.
//!
//! # When the model opens the call itself
//!
//! A speculative round may commit `</think>` and the start of the model's
//! own call together. The literal's first **common** tokens are the ones
//! every call writes alike — for the tool-call opener `<tool_call>`, `\n`,
//! `<`, `function`, before the name — so a model that stopped inside them is
//! continued: the token after its last one, unseen, is taken to be the
//! literal's next, and forcing resumes at the one after that. A model that
//! wrote all of them, or wrote something else after `<tool_call>`, is left
//! alone: the name, or whatever it wrote, is its own.

use std::sync::Arc;

use crate::constrained::PermittedSet;
use crate::types::TokenId;

/// Where a [`ForcedLiteral`] starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForcedStart {
    /// At the generation's first token.
    Generation,
    /// Right after the reasoning block closes at `think_end`, the block's
    /// closing marker. The literal's first `common` tokens are the ones
    /// every call writes alike (module docs).
    AfterReasoning { think_end: TokenId, common: usize },
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
    /// `think_end`, whose first `common` tokens every call writes alike.
    /// Refused without a second token, which is the joiner a close by the
    /// model itself needs, and with `common` past the literal (module docs).
    pub fn after_reasoning(tokens: Vec<TokenId>, think_end: TokenId, common: usize) -> Result<Self, String> {
        if tokens.len() < 2 {
            return Err("a literal forced after the reasoning block needs a second token to join on".to_owned());
        }
        if common > tokens.len() {
            return Err(format!("{common} common tokens in a literal of {}", tokens.len()));
        }
        Ok(Self { tokens: tokens.into(), start: ForcedStart::AfterReasoning { think_end, common } })
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
    /// The block has closed and forcing has not started. `matched`: how many
    /// of the literal's tokens the model has written itself since, in order.
    Closed { matched: usize },
    /// Forcing started with the draw for output index `from`, at literal
    /// index `at`. `joined`: the run opens with the joiner instead, the
    /// token before it having been drawn unseen; `unseen_opened`: that token
    /// was the literal's first.
    Forcing { from: u32, at: usize, joined: bool, unseen_opened: bool },
    /// The model is writing the call itself: nothing is forced.
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
    /// `budget` is the set the thinking budget forces this round, and wins:
    /// the budget only forces inside the reasoning block, and nothing of the
    /// literal is due before the block has closed.
    pub fn permitted(
        &mut self,
        literal: &ForcedLiteral,
        emitted: u32,
        budget: Option<PermittedSet>,
    ) -> Option<PermittedSet> {
        let set = match budget {
            Some(budget) => Some(budget),
            None => self.step(literal, emitted + 1),
        };
        self.last_draw_forced = set.as_ref().is_some_and(|set| set.len() == 1);
        set
    }

    fn step(&mut self, literal: &ForcedLiteral, draw: u32) -> Option<PermittedSet> {
        let at = match (literal.start, self.phase) {
            (ForcedStart::Generation, _) => draw as usize,
            (_, Phase::Reasoning | Phase::Released) => return None,
            // Nothing of the call yet: the joiner after an unseen token, the
            // first token after a seen one.
            (_, Phase::Closed { matched: 0 }) => {
                let joined = !self.last_draw_forced;
                self.phase = Phase::Forcing { from: draw, at: 0, joined, unseen_opened: false };
                if joined { 1 } else { 0 }
            }
            // The model's own call, stopped inside the common tokens: the
            // unseen token after it is the literal's next.
            (_, Phase::Closed { matched }) => {
                self.phase = Phase::Forcing { from: draw, at: matched + 1, joined: false, unseen_opened: false };
                matched + 1
            }
            (_, Phase::Forcing { from, at, joined, unseen_opened }) => match (joined, draw.checked_sub(from)? as usize) {
                (false, step) => at + step,
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
        let ForcedStart::AfterReasoning { think_end, common } = literal.start else {
            return;
        };
        self.phase = match self.phase {
            Phase::Reasoning if token == think_end => Phase::Closed { matched: 0 },
            Phase::Closed { matched } => match (matched, token == literal.tokens[matched]) {
                (0, false) => Phase::Closed { matched: 0 },
                // Written past the common tokens, or something else after
                // its own `<tool_call>`: the call is its own.
                (_, true) if matched + 1 >= common => Phase::Released,
                (_, true) => Phase::Closed { matched: matched + 1 },
                (_, false) => Phase::Released,
            },
            // The unseen token before the joiner, now seen.
            Phase::Forcing { from, at, joined: true, .. } if index + 1 == from => {
                Phase::Forcing { from, at, joined: true, unseen_opened: token == literal.tokens[0] }
            }
            phase => phase,
        };
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
            let set = state.permitted(literal, emitted, budget);
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
    fn an_empty_literal_one_with_no_joiner_and_common_tokens_past_it_are_refused() {
        assert!(ForcedLiteral::at_generation(Vec::new()).is_err());
        assert!(ForcedLiteral::after_reasoning(vec![OPEN], END, 1).is_err());
        assert!(ForcedLiteral::after_reasoning(OPENER.to_vec(), END, 5).is_err());
        assert!(ForcedLiteral::after_reasoning(OPENER.to_vec(), END, 4).is_ok());
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
        let literal = ForcedLiteral::after_reasoning(OPENER.to_vec(), END, 4).unwrap();
        assert_eq!(literal.prefill_step(), None);
        assert!(run(&literal, |_| 5, free, 12).iter().all(|&t| t == 5));
    }

    #[test]
    fn after_the_models_own_close_the_joiner_follows_the_unseen_token_and_the_literal_follows_it() {
        // The model closes at 3 and writes `\n\n` (7) at 4, drawn in the
        // round that emitted the close: unseen. The joiner comes next, then
        // the whole literal, then the model again.
        let literal = ForcedLiteral::after_reasoning(OPENER.to_vec(), END, 4).unwrap();
        let out = run(&literal, |i| match i { 3 => END, 4 => 7, _ => 5 }, free, 12);
        assert_eq!(out, vec![5, 5, 5, END, 7, NL, OPEN, NL, 52, 53, 5, 5]);
    }

    #[test]
    fn an_unseen_token_that_opened_the_literal_is_not_opened_again() {
        // The model writes `<tool_call>` itself at 4, the token nobody saw:
        // the joiner is its line break, and the literal resumes at `<`.
        let literal = ForcedLiteral::after_reasoning(OPENER.to_vec(), END, 4).unwrap();
        let out = run(&literal, |i| match i { 3 => END, 4 => OPEN, _ => 5 }, free, 10);
        assert_eq!(out, vec![5, 5, 5, END, OPEN, NL, 52, 53, 5, 5]);
        assert_eq!(out.iter().filter(|&&t| t == OPEN).count(), 1);
    }

    #[test]
    fn a_close_the_budget_forced_is_followed_by_the_literal_with_no_joiner() {
        // The budget forces `</think>`, `\n\n` (7) for draws 4 and 5: the
        // token before the literal's first draw was forced, so seen.
        let literal = ForcedLiteral::after_reasoning(OPENER.to_vec(), END, 4).unwrap();
        let budget = |draw: u32| match draw { 4 => Some(END), 5 => Some(7), _ => None };
        let out = run(&literal, |_| 5, budget, 12);
        assert_eq!(out, vec![5, 5, 5, 5, END, 7, OPEN, NL, 52, 53, 5, 5]);
    }

    /// One speculative round's tokens, committed in order before the next
    /// round asks for a set.
    fn one_round(literal: &ForcedLiteral, tokens: &[TokenId]) -> ForcedState {
        let mut state = ForcedState::default();
        assert_eq!(state.permitted(literal, 0, None), None);
        for (index, &token) in tokens.iter().enumerate() {
            state.commit(literal, index as u32, token);
        }
        state
    }

    #[test]
    fn the_models_own_call_stopped_inside_the_common_tokens_is_continued() {
        // `</think>`, `\n\n`, `<tool_call>`, `\n` in one round: the unseen
        // token after them is the `<` every call writes next, so the draw
        // after it is the literal's `function`, and then it is spent.
        let literal = ForcedLiteral::after_reasoning(OPENER.to_vec(), END, 4).unwrap();
        let mut state = one_round(&literal, &[5, END, 7, OPEN, NL]);
        assert_eq!(state.permitted(&literal, 5, None).as_deref(), Some(&[53][..]));
        assert_eq!(state.permitted(&literal, 6, None), None);
    }

    #[test]
    fn the_models_own_call_past_the_common_tokens_or_off_the_dialect_is_its_own() {
        let literal = ForcedLiteral::after_reasoning(OPENER.to_vec(), END, 4).unwrap();
        for tokens in [&[END, 7, OPEN, NL, 52, 53][..], &[END, OPEN, 77][..]] {
            let mut state = one_round(&literal, tokens);
            assert!((6..20).all(|emitted| state.permitted(&literal, emitted, None).is_none()), "{tokens:?}");
        }
    }

    #[test]
    fn the_budget_set_wins_and_the_literal_waits_for_it() {
        let literal = ForcedLiteral::at_generation(OPENER.to_vec()).unwrap();
        let mut state = ForcedState::default();
        let budget = PermittedSet::from([END]);
        assert_eq!(state.permitted(&literal, 0, Some(budget.clone())), Some(budget));
    }
}
