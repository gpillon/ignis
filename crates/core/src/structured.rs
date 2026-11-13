//! **Structured output** on the scheduler's side (GitHub #587, spec #287,
//! ADR 0089): a request's answer held to a **grammar**, check-first.
//!
//! The model samples freely. Each draw of the answer waits host-side as the
//! sequence's pending token until the next round consumes it, and the
//! scheduler **checks** it against the request's grammar in between: the
//! prefill's own draw right after the prefill, a plain round's right after
//! the round ([`crate::scheduler::DecodeOutcome::waiting`]). A **miss** is
//! **repaired** there and then: the position is drawn again from the scores
//! the call kept, restricted to the grammar's permitted set
//! ([`crate::scheduler::Compute::repair`]), and the next round consumes the
//! repaired token as if it had been drawn. Under greedy the answer is the
//! one masking every step gives; a check costs about a microsecond and only
//! a miss sends anything down.
//!
//! # Where the answer starts
//!
//! With the generation starting inside the reasoning block, at the token
//! after the block's `</think>`: the reasoning stays free. Otherwise at the
//! first token, which the prefill draws. A draw a permitted set forced (the
//! thinking budget's close, whose tail is the line break after `</think>`)
//! is the server's own text and is neither checked nor fed to the grammar.
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
use crate::scheduler::WaitingDraw;
use crate::types::TokenId;

/// What a request asked its answer to be: the type its `response_format`
/// named, and the label its metrics and its log event carry (#593).
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

/// What a request's grammar did over its life, kept on the request and
/// reported on its finish ([`StructuredSummary`], #593).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StructuredTally {
    /// Draws the grammar rejected.
    pub misses: u32,
    /// How long each repair took, in microseconds, in order. A miss the
    /// engine could not repair has no entry.
    pub repair_micros: Vec<u32>,
}

/// What a constrained request's finish reports about its grammar (#593): the
/// type it asked for, its misses, and the time of each repair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredSummary {
    /// The type the request's `response_format` named.
    pub kind: StructuredKind,
    /// Draws the grammar rejected.
    pub misses: u32,
    /// How long each repair took, in microseconds.
    pub repair_micros: Vec<u32>,
}

impl StructuredSummary {
    /// Misses the engine repaired.
    pub fn repairs(&self) -> u32 {
        self.repair_micros.len() as u32
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
}

impl AnswerGrammar {
    /// An answer held to `grammar`, for a request that asked for `kind`.
    /// With the generation starting inside the reasoning block it starts
    /// after the block's close, which [`AnswerGrammar::after`] names.
    pub fn new(grammar: Arc<Grammar>, kind: StructuredKind) -> Self {
        Self { grammar, kind, think_end: None }
    }

    /// The same answer, starting after the reasoning block's `think_end` when
    /// the generation starts inside the block
    /// ([`crate::types::DecodeParams::starts_in_reasoning`]).
    pub fn after(self, think_end: TokenId) -> Self {
        Self { think_end: Some(think_end), ..self }
    }

    /// What the request asked for.
    pub fn kind(&self) -> StructuredKind {
        self.kind
    }

    /// The compiled grammar.
    pub fn grammar(&self) -> &Arc<Grammar> {
        &self.grammar
    }

    /// The reasoning block's close the answer starts after, if one was named.
    pub fn think_end(&self) -> Option<TokenId> {
        self.think_end
    }
}

/// A compiled grammar holds thousands of states; what a log line needs is
/// which kind it is and where it starts.
impl std::fmt::Debug for AnswerGrammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnswerGrammar")
            .field("kind", &self.kind)
            .field("think_end", &self.think_end)
            .finish()
    }
}

/// What the grammar says of a waiting draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Valid where the answer stands; the grammar has taken it.
    Passed,
    /// A stop where the answer is complete: the request ends here.
    Finished,
    /// A **miss**: the position is repaired before the next round.
    Miss,
}

/// One request's progress through its grammar
/// ([`crate::request::Request::grammar`]).
#[derive(Clone)]
pub struct GrammarState {
    matcher: Matcher,
    /// The answer has started: thinking off, or the reasoning block closed.
    open: bool,
    think_end: Option<TokenId>,
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
            set: None,
        }
    }

    /// Whether the answer has started, so a draw is the grammar's to check.
    pub fn holds(&self) -> bool {
        self.open
    }

    /// The tokens a call committed: the reasoning block's close among them
    /// opens the answer at the token after it.
    pub fn commit(&mut self, tokens: &[TokenId]) {
        if !self.open && self.think_end.is_some_and(|close| tokens.contains(&close)) {
            self.open = true;
        }
    }

    /// Check a waiting draw of the answer, and take it when it passes.
    pub fn check(&mut self, draw: WaitingDraw) -> Verdict {
        match draw {
            WaitingDraw::Stop if self.matcher.is_accepting() => Verdict::Finished,
            WaitingDraw::Stop => Verdict::Miss,
            WaitingDraw::Token(token) if !self.matcher.accept(token) => Verdict::Miss,
            WaitingDraw::Token(_) if self.matcher.is_finished() => Verdict::Finished,
            WaitingDraw::Token(_) => Verdict::Passed,
        }
    }

    /// The permitted set where the answer stands: what a repair restricts
    /// the missed position to. Never empty while the answer is not finished
    /// (no state of a grammar is a dead end); an empty one is refused here.
    pub fn permitted(&mut self) -> Result<PermittedMask, String> {
        let vocab = self.matcher.grammar().vocab().len();
        let set = self.set.get_or_insert_with(|| TokenSet::new(vocab));
        self.matcher.mask(set);
        PermittedMask::from_words(vocab as u32, set.words().to_vec())
    }

    /// Take the token a repair drew: `true` when it was a stop, and the
    /// answer is complete. Refused when the token is not one the grammar
    /// allows there, which a repair under its permitted set never draws.
    pub fn repaired(&mut self, token: TokenId) -> Result<bool, String> {
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
            .field("accepting", &self.matcher.is_accepting())
            .finish()
    }
}
