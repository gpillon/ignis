//! **Structured output** on the scheduler's side (GitHub #587, spec #287,
//! ADR 0089): a request's answer held to a **grammar**, check-first.
//!
//! The model samples freely. Each draw of the answer waits host-side as the
//! sequence's pending token until the next round consumes it, and the
//! scheduler **checks** it against the request's grammar in between: the
//! prefill's own draw right after the prefill, a plain round's right after
//! the round ([`crate::scheduler::DecodeOutcome::pending`]). A **miss** is
//! **repaired** there and then: the position is drawn again from the scores
//! the call kept, restricted to the grammar's permitted set
//! ([`crate::scheduler::Compute::repair`]), and the next round consumes the
//! repaired token as if it had been drawn. Under greedy the answer is the
//! one masking every step gives; a check costs about a microsecond and only
//! a miss sends anything down.
//!
//! # Speculation stays on
//!
//! A verify round's drafts are checked before they are committed (GitHub
//! #590, [`GrammarState::check_run`]): the host's cut keeps the run up to its
//! first draft that does not pass, which then waits as the round's draw and
//! is held like any other -- repaired from the verify pass's scores on a
//! miss, or the request ended on a stop where the answer is complete.
//!
//! # Where the answer starts
//!
//! With the generation starting inside the reasoning block, at the token
//! after the block's `</think>`: the reasoning stays free. Otherwise at the
//! first token, which the prefill draws. A draw a permitted set forced (the
//! thinking budget's close, whose tail is the line break after `</think>`)
//! is the server's own text and is neither checked nor fed to the grammar.
//!
//! # A tool call instead of the answer
//!
//! With tools beside the grammar (GitHub #592), a turn either calls a tool or
//! answers under the grammar. Until the answer starts -- until anything but
//! leading whitespace has passed -- the grammar also admits the tool call's
//! opener ([`AnswerGrammar::admitting_tool_call`]), a control token no
//! grammar accepts: it passes, and the turn is **released** from the grammar,
//! so the call is written as the model writes it, with no check, miss or
//! repair. A repair before the answer starts may draw the opener too: it is
//! in the permitted set, as masking every step would have it.
//!
//! # Where it ends
//!
//! A stop passes only where the answer is complete; the request then ends
//! with `stop` at once, the EOS never consumed. Nothing else ends it but its
//! cap (`length`) or its deadline: the grammar never strands the model in a
//! state it cannot finish, and its whitespace runs are bounded.

use std::sync::Arc;

use ignis_grammar::{Grammar, Matcher, TokenSet};

use crate::constrained::PermittedMask;
use crate::scheduler::PendingToken;
use crate::types::TokenId;

/// What a request asked its answer to be: the type its `response_format`
/// named, and the label the server's telemetry and its log event carry
/// (#593).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredKind {
    /// Any JSON object (`json_object`).
    JsonObject,
    /// A JSON value valid under the caller's schema (`json_schema`, #588).
    JsonSchema,
}

impl StructuredKind {
    /// The wire name: `response_format`'s `type`.
    pub fn name(self) -> &'static str {
        match self {
            StructuredKind::JsonObject => "json_object",
            StructuredKind::JsonSchema => "json_schema",
        }
    }
}

/// The grammar a request's answer is held to
/// ([`crate::types::RequestInput::grammar`]): a compiled grammar, shared by
/// every request that sent the same shape, and where the answer starts.
#[derive(Clone)]
pub struct AnswerGrammar {
    grammar: Arc<Grammar>,
    kind: StructuredKind,
    think_end: Option<TokenId>,
    tool_opener: Option<TokenId>,
}

impl AnswerGrammar {
    /// An answer held to `grammar`, for a request that asked for `kind`.
    /// With the generation starting inside the reasoning block it starts
    /// after the block's close, which [`AnswerGrammar::after`] names.
    pub fn new(grammar: Arc<Grammar>, kind: StructuredKind) -> Self {
        Self { grammar, kind, think_end: None, tool_opener: None }
    }

    /// The same answer, starting after the reasoning block's `think_end` when
    /// the generation starts inside the block
    /// ([`crate::types::DecodeParams::starts_in_reasoning`]).
    pub fn after(self, think_end: TokenId) -> Self {
        Self { think_end: Some(think_end), ..self }
    }

    /// The same answer, which the turn may skip for a tool call (GitHub
    /// #592): until the answer starts, `opener` -- the tool call's opening
    /// token, which no grammar accepts -- passes too, and releases the turn
    /// from the grammar.
    pub fn admitting_tool_call(self, opener: TokenId) -> Self {
        Self { tool_opener: Some(opener), ..self }
    }

    /// What the request asked for.
    pub fn kind(&self) -> StructuredKind {
        self.kind
    }
}

/// A compiled grammar holds thousands of states; what a log line needs is
/// which kind it is and where it starts.
impl std::fmt::Debug for AnswerGrammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnswerGrammar")
            .field("kind", &self.kind)
            .field("think_end", &self.think_end)
            .field("tool_opener", &self.tool_opener)
            .finish()
    }
}

/// What the grammar says of a pending token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Valid where the answer stands; the grammar has taken it.
    Passed,
    /// A stop where the answer is complete: the request ends here.
    Finished,
    /// A **miss**: the position is repaired before the next round.
    Miss,
    /// The tool call's opener before the answer started (GitHub #592): the
    /// turn is a call, released from the grammar.
    Released,
}

/// One request's progress through its grammar
/// ([`crate::request::Request::grammar`]).
#[derive(Clone)]
pub struct GrammarState {
    matcher: Matcher,
    /// The answer has started: thinking off, or the reasoning block closed.
    open: bool,
    think_end: Option<TokenId>,
    tool_opener: Option<TokenId>,
    /// The turn opened a tool call (GitHub #592): nothing more is checked.
    released: bool,
    /// The permitted set's bits, built on the first miss and reused.
    set: Option<TokenSet>,
}

impl GrammarState {
    /// A request's state before its first token: the answer open at once
    /// unless the generation starts inside the reasoning block.
    pub fn new(grammar: &AnswerGrammar, starts_in_reasoning: bool) -> Self {
        Self {
            matcher: Matcher::new(grammar.grammar.clone()),
            open: !starts_in_reasoning,
            think_end: grammar.think_end,
            tool_opener: grammar.tool_opener,
            released: false,
            set: None,
        }
    }

    /// Whether the answer has started, so a draw is the grammar's to check:
    /// never again once the turn opened a tool call.
    pub fn holds(&self) -> bool {
        self.open && !self.released
    }

    /// Whether the turn opened a tool call and is released from the grammar
    /// (GitHub #592): its rounds are no longer checked.
    pub fn released(&self) -> bool {
        self.released
    }

    /// Whether `token` is the tool call's opener where it may still open one:
    /// before the answer has started.
    fn opens_tool_call(&self, token: TokenId) -> bool {
        self.tool_opener == Some(token) && !self.matcher.answer_started()
    }

    /// The tokens a call committed: the reasoning block's close among them
    /// opens the answer at the token after it.
    pub fn commit(&mut self, tokens: &[TokenId]) {
        if !self.open && self.think_end.is_some_and(|close| tokens.contains(&close)) {
            self.open = true;
        }
    }

    /// Check a verify round's run before it is committed (GitHub #590): its
    /// anchor -- the last pending token, already held -- then the drafts the
    /// target accepted. Returns how many of the run's tokens the commit
    /// keeps, at least the anchor: every draft that passes as an ordinary
    /// token is the grammar's from then on, and the first that does not --
    /// a miss, or a stop -- is where the run is cut. That draft then waits
    /// as the round's draw, and is held as every pending token is
    /// ([`GrammarState::check`]): repaired, or the request ended.
    ///
    /// Before the answer starts nothing is checked, but a reasoning close in
    /// the run opens it at the token after; a stop there is the leaf's to cut
    /// or redirect, and the rest of the run is left to it whole. So is the
    /// rest of a run whose draft opens a tool call before the answer started
    /// (GitHub #592), which releases the turn, and every run after.
    pub fn check_run(&mut self, run: &[PendingToken]) -> usize {
        if self.released {
            return run.len();
        }
        for (index, &draw) in run.iter().enumerate() {
            let token = match draw {
                PendingToken::Token(token) => token,
                PendingToken::Stop if self.open && index > 0 => return index,
                PendingToken::Stop => return run.len(),
            };
            if !self.open {
                self.open = self.think_end == Some(token);
                continue;
            }
            // The anchor was held when it was drawn.
            if index == 0 {
                continue;
            }
            if self.opens_tool_call(token) {
                self.released = true;
                return run.len();
            }
            if self.matcher.grammar().vocab().is_stop(token) || !self.matcher.accept(token) {
                return index;
            }
        }
        run.len()
    }

    /// Check a pending token of the answer, and take it when it passes. The
    /// tool call's opener before the answer started releases the turn.
    pub fn check(&mut self, draw: PendingToken) -> Verdict {
        match draw {
            PendingToken::Token(token) if self.opens_tool_call(token) => {
                self.released = true;
                Verdict::Released
            }
            PendingToken::Stop if self.matcher.is_accepting() => Verdict::Finished,
            PendingToken::Stop => Verdict::Miss,
            PendingToken::Token(token) if !self.matcher.accept(token) => Verdict::Miss,
            PendingToken::Token(_) if self.matcher.is_finished() => Verdict::Finished,
            PendingToken::Token(_) => Verdict::Passed,
        }
    }

    /// The permitted set where the answer stands: what a repair restricts
    /// the missed position to, the tool call's opener among it until the
    /// answer starts. Never empty while the answer is not finished (no state
    /// of a grammar is a dead end); an empty one is refused here.
    pub fn permitted(&mut self) -> Result<PermittedMask, String> {
        let vocab = self.matcher.grammar().vocab().len();
        let opener = self.tool_opener.filter(|&opener| self.opens_tool_call(opener) && (opener as usize) < vocab);
        let set = self.set.get_or_insert_with(|| TokenSet::new(vocab));
        self.matcher.mask(set);
        if let Some(opener) = opener {
            set.insert(opener);
        }
        PermittedMask::from_words(vocab as u32, set.words().to_vec())
    }

    /// Take the token a repair drew: `true` when it was a stop, and the
    /// answer is complete. The tool call's opener releases the turn. Refused
    /// when the token is not one the grammar allows there, which a repair
    /// under its permitted set never draws.
    pub fn repaired(&mut self, token: TokenId) -> Result<bool, String> {
        if self.opens_tool_call(token) {
            self.released = true;
            return Ok(false);
        }
        match self.matcher.accept(token) {
            true => Ok(self.matcher.is_finished()),
            false => Err(format!("the repair drew token {token}, which the grammar does not allow there")),
        }
    }
}

impl std::fmt::Debug for GrammarState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrammarState")
            .field("open", &self.open)
            .field("released", &self.released)
            .field("accepting", &self.matcher.is_accepting())
            .finish()
    }
}
