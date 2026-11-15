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
//! in the permitted set, as masking every step would have it. Beside the
//! request's **strict tools** (GitHub #615,
//! [`AnswerGrammar::holding_calls`]) the call is held instead: from the
//! opener on the turn is read by their tool calls' grammar, which holds that
//! call and every one after it, the text between them free, and never
//! returns to the answer's.
//!
//! # Where it ends
//!
//! A stop passes only where the answer is complete; the request then ends
//! with `stop` at once, the EOS never consumed. Nothing else ends it but its
//! cap (`length`) or its deadline: the grammar never strands the model in a
//! state it cannot finish, and its whitespace runs are bounded.
//!
//! # Tool calls held to their tools
//!
//! A request with a **strict tool** (GitHub #611, spec #582, ADR 0094) holds
//! its content to its tool calls' grammar ([`AnswerGrammar::tool_calls`])
//! instead: free text, in which each call the model opens is checked as it
//! is written, a held tool's arguments to its parameters. The grammar arms
//! at the call's opener and is released at its confirmed close inside its
//! own automaton, so every draw of the content is checked -- free text
//! passes whatever it is -- and a verified run is cut at its first miss
//! inside a call. The reasoning stays free: the content starts after the
//! block's `</think>`, as an answer does.
//!
//! A call `tool_choice` forces (GitHub #615) is opened by its **forced
//! literal** ([`crate::forced_literal`]), whose tokens are the server's: each
//! is fed to the grammar as it is forced ([`GrammarState::forced`]), never
//! checked nor counted a miss, and the model's own tokens after it are held
//! as any call's. A literal token the grammar does not take where the
//! content stands ([`GrammarState::admits`]) is not forced: the literal
//! yields there, and the grammar holds what the model writes instead.

use std::sync::Arc;

use ignis_grammar::{Grammar, Matcher, TokenSet, ToolCallGrammar, ToolCallMatcher};

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
    /// Tool calls held to the request's tools: a request with a **strict
    /// tool** (GitHub #611).
    Tool,
}

impl StructuredKind {
    /// The wire name: `response_format`'s `type`, and `tool` for a request's
    /// held tool calls.
    pub fn name(self) -> &'static str {
        match self {
            StructuredKind::JsonObject => "json_object",
            StructuredKind::JsonSchema => "json_schema",
            StructuredKind::Tool => "tool",
        }
    }
}

/// The grammar a request's answer is held to
/// ([`crate::types::RequestInput::grammar`]): a compiled grammar, shared by
/// every request that sent the same shape, and where the answer starts.
#[derive(Clone)]
pub struct AnswerGrammar {
    grammar: Compiled,
    kind: StructuredKind,
    think_end: Option<TokenId>,
    tool_opener: Option<TokenId>,
    /// The tool calls' grammar the call the opener starts is held to (#615).
    calls: Option<Arc<ToolCallGrammar>>,
}

/// What an [`AnswerGrammar`] holds the content to.
#[derive(Clone)]
enum Compiled {
    /// A structured answer (#287).
    Answer(Arc<Grammar>),
    /// The request's tool calls (#611).
    Calls(Arc<ToolCallGrammar>),
}

impl AnswerGrammar {
    /// An answer held to `grammar`, for a request that asked for `kind`.
    /// With the generation starting inside the reasoning block it starts
    /// after the block's close, which [`AnswerGrammar::after`] names.
    pub fn new(grammar: Arc<Grammar>, kind: StructuredKind) -> Self {
        Self { grammar: Compiled::Answer(grammar), kind, think_end: None, tool_opener: None, calls: None }
    }

    /// The content held to a request's tool calls' grammar (GitHub #611):
    /// free text whose calls name the request's tools, a held tool's held to
    /// its parameters. [`AnswerGrammar::after`] applies as to an answer;
    /// [`AnswerGrammar::admitting_tool_call`] has no use here, the grammar
    /// reading the opener itself.
    pub fn tool_calls(grammar: Arc<ToolCallGrammar>) -> Self {
        Self { grammar: Compiled::Calls(grammar), kind: StructuredKind::Tool, think_end: None, tool_opener: None, calls: None }
    }

    /// Whether the content's first draw can be a miss, so the draw a
    /// prefill makes must keep its scores for a repair: always for an
    /// answer, and for tool calls only on a vocabulary where a token holds
    /// the call's opener with more after it.
    pub fn checks_first_draw(&self) -> bool {
        match &self.grammar {
            Compiled::Answer(_) => true,
            Compiled::Calls(grammar) => grammar.first_draw_may_miss(),
        }
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

    /// The same answer, beside the request's **strict tools** (GitHub #615):
    /// the call a turn opens instead of the answer
    /// ([`AnswerGrammar::admitting_tool_call`]) is held to their tool calls'
    /// grammar, which then holds the rest of the turn -- its text free, each
    /// further call held as the first. The turn stays released from the
    /// answer's grammar, and the request keeps the answer's kind.
    pub fn holding_calls(self, calls: Arc<ToolCallGrammar>) -> Self {
        Self { calls: Some(calls), ..self }
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
    /// turn is a call, released from the grammar -- unless the request's
    /// strict tools hold the call (#615), and the opener then passes.
    Released,
}

/// One request's progress through its grammar
/// ([`crate::request::Request::grammar`]).
#[derive(Clone)]
pub struct GrammarState {
    matcher: Held,
    /// The answer has started: thinking off, or the reasoning block closed.
    open: bool,
    think_end: Option<TokenId>,
    tool_opener: Option<TokenId>,
    /// The tool calls' grammar the call the opener starts is held to (#615).
    calls: Option<Arc<ToolCallGrammar>>,
    /// The turn opened a tool call (GitHub #592), or the grammar could not
    /// follow a token the server forced (#615): nothing more is checked.
    released: bool,
    /// The permitted set's bits, built on the first miss and reused.
    set: Option<TokenSet>,
}

impl GrammarState {
    /// A request's state before its first token: the answer open at once
    /// unless the generation starts inside the reasoning block.
    pub fn new(grammar: &AnswerGrammar, starts_in_reasoning: bool) -> Self {
        let matcher = match &grammar.grammar {
            Compiled::Answer(grammar) => Held::Answer(Matcher::new(grammar.clone())),
            Compiled::Calls(grammar) => Held::Calls(ToolCallMatcher::new(grammar.clone())),
        };
        Self {
            matcher,
            open: !starts_in_reasoning,
            think_end: grammar.think_end,
            tool_opener: grammar.tool_opener,
            calls: grammar.calls.clone(),
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

    /// The turn opens a tool call instead of the answer, at `opener`: it is
    /// released from the answer's grammar (#592), and with the request's
    /// tool calls' grammar beside it (#615) the call is held to that from
    /// the opener on, for the rest of the turn. Returns whether the turn is
    /// released, nothing more checked: with no tool calls' grammar, or on a
    /// vocabulary whose opener has no text to arm it.
    fn open_tool_call(&mut self, opener: TokenId) -> bool {
        let Some(calls) = self.calls.take() else {
            self.released = true;
            return true;
        };
        let mut matcher = ToolCallMatcher::new(calls);
        self.released = !(matcher.accept(opener) && matcher.in_call());
        self.matcher = Held::Calls(matcher);
        self.tool_opener = None;
        self.released
    }

    /// Whether the grammar takes `token` where the content stands, taking
    /// nothing: true before the content starts, where nothing is held. A
    /// forced literal's token is forced only where it does (#615).
    pub fn admits(&mut self, token: TokenId) -> bool {
        !self.holds() || self.opens_tool_call(token) || self.matcher.check(token)
    }

    /// Take a token the server forced, a forced literal's (#615): the grammar
    /// follows the server's own text, which is never a miss. Only where the
    /// content is held; a token the grammar does not take there, which
    /// [`GrammarState::admits`] keeps from being forced, releases the turn.
    pub fn forced(&mut self, token: TokenId) {
        if !self.holds() {
            return;
        }
        if self.opens_tool_call(token) {
            self.open_tool_call(token);
        } else if !self.matcher.accept(token) {
            self.released = true;
        }
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
    /// (GitHub #592), which releases the turn, and every run after -- unless
    /// the request's strict tools hold the call (#615), whose drafts are then
    /// checked as the answer's were.
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
                if self.open_tool_call(token) {
                    return run.len();
                }
                continue;
            }
            if self.matcher.is_stop(token) || !self.matcher.accept(token) {
                return index;
            }
        }
        run.len()
    }

    /// Check a pending token of the answer, and take it when it passes. The
    /// tool call's opener before the answer started releases the turn, or
    /// passes into the call the request's strict tools hold (#615).
    pub fn check(&mut self, draw: PendingToken) -> Verdict {
        match draw {
            PendingToken::Token(token) if self.opens_tool_call(token) => match self.open_tool_call(token) {
                true => Verdict::Released,
                false => Verdict::Passed,
            },
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
    /// of a grammar is a dead end); an empty one is refused here, and the
    /// miss then ends its request in an error, unrepaired.
    pub fn permitted(&mut self) -> Result<PermittedMask, String> {
        let vocab = self.matcher.vocab_len();
        let opener = self.tool_opener.filter(|&opener| self.opens_tool_call(opener) && (opener as usize) < vocab);
        let set = self.set.get_or_insert_with(|| TokenSet::new(vocab));
        self.matcher.mask(set);
        if let Some(opener) = opener {
            set.insert(opener);
        }
        PermittedMask::from_words(vocab as u32, set.words().to_vec())
    }

    /// Take the token a repair drew: `true` when it was a stop, and the
    /// answer is complete. The tool call's opener releases the turn, or
    /// opens the call the request's strict tools hold (#615). Refused when
    /// the token is not one the grammar allows there, which a repair under
    /// its permitted set never draws.
    pub fn repaired(&mut self, token: TokenId) -> Result<bool, String> {
        if self.opens_tool_call(token) {
            self.open_tool_call(token);
            return Ok(false);
        }
        match self.matcher.accept(token) {
            true => Ok(self.matcher.is_finished()),
            false => Err(format!("the repair drew token {token}, which the grammar does not allow there")),
        }
    }
}

/// The matcher a [`GrammarState`] reads by: one request's position in its
/// grammar.
#[derive(Clone)]
enum Held {
    Answer(Matcher),
    Calls(ToolCallMatcher),
}

impl Held {
    fn accept(&mut self, token: TokenId) -> bool {
        match self {
            Held::Answer(m) => m.accept(token),
            Held::Calls(m) => m.accept(token),
        }
    }

    fn check(&mut self, token: TokenId) -> bool {
        match self {
            Held::Answer(m) => m.check(token),
            Held::Calls(m) => m.check(token),
        }
    }

    fn is_accepting(&self) -> bool {
        match self {
            Held::Answer(m) => m.is_accepting(),
            Held::Calls(m) => m.is_accepting(),
        }
    }

    fn is_finished(&self) -> bool {
        match self {
            Held::Answer(m) => m.is_finished(),
            Held::Calls(m) => m.is_finished(),
        }
    }

    fn is_stop(&self, token: TokenId) -> bool {
        match self {
            Held::Answer(m) => m.grammar().vocab().is_stop(token),
            Held::Calls(m) => m.grammar().vocab().is_stop(token),
        }
    }

    fn vocab_len(&self) -> usize {
        match self {
            Held::Answer(m) => m.grammar().vocab().len(),
            Held::Calls(m) => m.grammar().vocab().len(),
        }
    }

    /// Whether the answer has started (#592); a request's tool calls have
    /// no answer to start, and admit no opener beside their grammar's.
    fn answer_started(&self) -> bool {
        match self {
            Held::Answer(m) => m.answer_started(),
            Held::Calls(_) => true,
        }
    }

    fn mask(&mut self, set: &mut TokenSet) {
        match self {
            Held::Answer(m) => {
                m.mask(set);
            }
            Held::Calls(m) => {
                m.mask(set);
            }
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
